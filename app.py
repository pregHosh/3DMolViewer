from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import os
# Optional dependency for keyboard shortcuts (legacy API)
try:
    from st_hotkeys import st_hotkeys as legacy_st_hotkeys  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    legacy_st_hotkeys = None

# Optional dependency for keyboard shortcuts (modern API)
try:
    import streamlit_hotkeys as hotkeys  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    hotkeys = None

# Optional dependency for memory introspection
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - best effort
    psutil = None

if psutil is not None:
    _PROCESS = psutil.Process(os.getpid())
else:  # pragma: no cover - fallback path
    _PROCESS = None


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_XYZ_DIR = str(PACKAGE_ROOT / "test_xyz")


BASE_COLUMNS = {
    "identifier",
    "label",
    "path",
    "source",
    "db_path",
    "db_id",
    "selection_id",
    "__index",
    "has_geometry",
}


# Minimum characters before running expensive search/filter logic over large datasets.
MIN_SEARCH_CHARS = 2

SESSION_DEBUG_KEY = "molviewer_debug_enabled"

SIDEBAR_SELECT_LIMIT = 1500
STATS_AUTO_LIMIT = 500


# Import refactored modules
from src.theme_config import THEMES, inject_theme_css
from src.data_loader import load_xyz_metadata, load_ase_metadata, get_atoms, load_atoms_raw
from src.viewer_utils import render_ngl_view, render_3dmol_view, filter_hydrogens, SNAPSHOT_QUALITY_OPTIONS
from src.plot_utils import build_scatter_figure, plot_controls_panel, plot_and_select, pick_numeric_columns, sanitize_plot_config
from src.ui_components import sidebar_controls, show_measurement_panel, show_details, navigation_controls


def _log_perf(
    perf_log: Optional[List[Dict[str, Any]]], event: str, elapsed: float, **metadata: Any
) -> Optional[Dict[str, Any]]:
    if perf_log is None:
        return None
    entry: Dict[str, Any] = {"event": event, "ms": round(elapsed * 1000, 3)}
    if _PROCESS is not None:
        try:
            mem_bytes = _PROCESS.memory_info().rss
        except Exception:  # pragma: no cover - defensive
            mem_bytes = None
        if mem_bytes is not None:
            entry["rss_mb"] = round(mem_bytes / (1024 * 1024), 2)
    if metadata:
        entry.update(metadata)
    perf_log.append(entry)
    return entry


def _precompute_search_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return cached lower-case identifier/label columns for search ranking."""
    cache_key = "_molviewer_search_cache"
    cache = df.attrs.get(cache_key)
    if cache and cache.get("size") == len(df):
        return cache["identifier"], cache["label"]

    identifier = df["identifier"].astype(str).str.lower()
    label = df["label"].astype(str).str.lower()
    cache = {"identifier": identifier, "label": label, "size": len(df)}
    df.attrs[cache_key] = cache
    return identifier, label


def _label_lookup(df: pd.DataFrame) -> Dict[str, str]:
    cache_key = "_molviewer_label_lookup"
    cache = df.attrs.get(cache_key)
    if cache and cache.get("size") == len(df):
        return cache["mapping"]

    identifiers = df["identifier"].astype(str)
    labels = df["label"].astype(str)
    display = np.where(identifiers == labels, identifiers, identifiers + " — " + labels)
    mapping = dict(zip(df["selection_id"], display))
    df.attrs[cache_key] = {"mapping": mapping, "size": len(df)}
    return mapping


def _limit_options(options: Iterable[str], current: Optional[str], limit: int = SIDEBAR_SELECT_LIMIT) -> Tuple[List[str], bool]:
    opts = list(options)
    truncated = False
    if len(opts) > limit:
        truncated = True
        opts = opts[:limit]
        if current and current not in opts:
            opts.insert(0, current)
    return opts, truncated


def _rank_xyz_matches(df: pd.DataFrame, query: str, limit: int = 200) -> pd.DataFrame:
    """
    Return up to `limit` best rows whose identifier/label match `query`.
    Ranking rule (simple, dependency-free):
      - Case-insensitive
      - prefix match > substring
      - shorter match position is better
    """
    if not query:
        return df.head(limit)

    q = str(query).strip().lower()
    if not q:
        return df.head(limit)
    if len(q) < MIN_SEARCH_CHARS:
        return df.head(limit)

    identifier_lower, label_lower = _precompute_search_series(df)

    startswith_id = identifier_lower.str.startswith(q)
    startswith_label = label_lower.str.startswith(q)
    contains_id = identifier_lower.str.contains(q, regex=False)
    contains_label = label_lower.str.contains(q, regex=False)

    match_mask = startswith_id | startswith_label | contains_id | contains_label
    if not match_mask.any():
        return df.iloc[0:0]

    matches = df.loc[match_mask].copy()

    # Prefix matches outrank substring matches.
    prefix_mask = (startswith_id | startswith_label)[match_mask]
    matches["_molviewer_score"] = prefix_mask.astype(int) * 2 + (~prefix_mask).astype(int)

    # Determine earliest match position across identifier/label for tie-breaking.
    id_pos = identifier_lower[match_mask].str.find(q)
    lbl_pos = label_lower[match_mask].str.find(q)
    id_pos = id_pos.mask(id_pos < 0, np.nan)
    lbl_pos = lbl_pos.mask(lbl_pos < 0, np.nan)
    matches["_molviewer_pos"] = pd.concat([id_pos, lbl_pos], axis=1).min(axis=1)

    matches.sort_values(
        by=["_molviewer_score", "_molviewer_pos", "__index"],
        ascending=[False, True, True],
        inplace=True,
    )

    trimmed = matches.head(limit).drop(columns=["_molviewer_score", "_molviewer_pos"], errors="ignore")
    return trimmed


def xyz_navbar(
    df: pd.DataFrame,
    selected_id: Optional[str],
    *,
    debug: bool = False,
    perf_log: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Top sticky navigation bar for XYZ mode:
      - ◀ Prev | type-to-filter selectbox | ▶ Next
      - Search over filename (identifier) and label with ranking.
    Returns possibly updated selected_id.
    """
    st.markdown('<div class="molviewer-navbar">', unsafe_allow_html=True)

    # Current position for prev/next logic
    all_ids = df["selection_id"].tolist()
    if not all_ids:
        st.markdown("</div>", unsafe_allow_html=True)
        return selected_id
    if selected_id not in all_ids:
        selected_id = all_ids[0]
    cur_idx = all_ids.index(selected_id)

    c_prev, c_search, c_next = st.columns([1, 8, 1])

    # Prev
    with c_prev:
        st.markdown('<span class="small-label">Navigate</span>', unsafe_allow_html=True)
        if st.button("◀", use_container_width=True, key="xyz_nav_prev"):
            cur_idx = (cur_idx - 1) % len(all_ids)
            selected_id = all_ids[cur_idx]

    # Search + autocomplete
    with c_search:
        # Persist user query between reruns
        query_key = "xyz_search_query"
        query = st.text_input(
            "Find file (type to filter)",
            key=query_key,
            placeholder="e.g., 000123.xyz or 'benzene'",
        )
        trimmed_query = (query or "").strip()
        entry: Optional[Dict[str, Any]] = None
        # Rank + filter candidates
        if query is not None:
            start = perf_counter()
            sub = _rank_xyz_matches(df, query, limit=500)
            entry = _log_perf(
                perf_log,
                "xyz_search",
                perf_counter() - start,
                query=trimmed_query,
                results=int(len(sub)),
                total=int(len(df)),
            )
            if debug and entry is not None and trimmed_query and len(trimmed_query) < MIN_SEARCH_CHARS:
                entry["note"] = "min_chars"
        else:
            sub = df.head(200)

        if debug and entry is not None:
            note_suffix = f" · {entry['note']}" if "note" in entry else ""
            st.caption(
                f"Search time: {entry['ms']:.3f} ms · results={entry['results']} of {entry['total']}{note_suffix}"
            )

        # Build pretty labels and a mapping for selectbox
        label_map = _label_lookup(sub)
        opts = list(label_map.keys())
        # Keep current selection in filtered list if possible
        idx = opts.index(selected_id) if selected_id in opts else (0 if opts else -1)

        chosen = st.selectbox(
            "Matches (autocomplete)",
            options=opts,
            index=idx if idx >= 0 else 0,
            format_func=lambda sid: label_map.get(sid, sid),
            help="Type to autocomplete; matches filename and label.",
            label_visibility="collapsed",
            key="xyz_autocomplete_select",
        )
        if chosen:
            selected_id = chosen

    # Next
    with c_next:
        st.markdown('<span class="small-label">Navigate</span>', unsafe_allow_html=True)
        if st.button("▶", use_container_width=True, key="xyz_nav_next"):
            # Recompute in case selected_id changed via search
            cur_idx = all_ids.index(selected_id)
            cur_idx = (cur_idx + 1) % len(all_ids)
            selected_id = all_ids[cur_idx]
    st.caption("Tip: Press **Backspace** for previous, **Enter** for next. (Also works with ◀ / ▶ buttons.)")

    st.markdown("</div>", unsafe_allow_html=True)

    # Persist selection
    st.session_state["selected_id"] = selected_id
    return selected_id


def arrow_key_listener() -> Optional[str]:
    """
    Waits for Streamlit to be ready, then attaches a robust keyboard listener.
    """
    from streamlit.components.v1 import html
    return html(
        """
        <script>
        (function(){
            // Use a unique name for the guard flag to ensure this runs only once.
            if (window.parent.molViewerGlobalKeyListenerAttached) {
                return;
            }

            function initializeListener(Streamlit) {
                // Set the guard flag only after we successfully get the Streamlit object.
                if (window.parent.molViewerGlobalKeyListenerAttached) return;
                window.parent.molViewerGlobalKeyListenerAttached = true;

                console.log("3DMolViewer: Streamlit is ready. Attaching key listener.");

                let lastSend = 0;
                function sendValueToStreamlit(val) {
                    const now = Date.now();
                    if (now - lastSend < 150) return; // Throttle events
                    lastSend = now;
                    Streamlit.setComponentValue(val);
                }

                function isUserTyping() {
                    const el = window.parent.document.activeElement;
                    if (!el) return false;
                    const tagName = el.tagName.toUpperCase();
                    return tagName === 'INPUT' || tagName === 'TEXTAREA' || el.isContentEditable;
                }

                window.parent.document.addEventListener("keydown", (e) => {
                    if (isUserTyping()) return;

                    let intent = null;
                    switch (e.key) {
                        case "Enter":
                        case "ArrowRight":
                        case ">":
                            intent = "next";
                            break;
                        case "Backspace":
                        case "ArrowLeft":
                        case "<":
                            intent = "prev";
                            break;
                    }

                    if (intent) {
                        e.preventDefault();
                        e.stopPropagation();
                        sendValueToStreamlit(intent);
                    }
                });

                // Finally, signal that the component is ready.
                Streamlit.setComponentReady();
            }

            // This function waits for the Streamlit object to be available on the parent window.
            function waitForStreamlit() {
                const streamlitParent = window.parent;
                if (streamlitParent && streamlitParent.Streamlit) {
                    initializeListener(streamlitParent.Streamlit);
                } else {
                    // If not found, check again in 50 milliseconds.
                    setTimeout(waitForStreamlit, 50);
                }
            }

            // Start the waiting process.
            waitForStreamlit();
        })();
        </script>
        """,
        height=0,
    )

@st.cache_data
def compute_dataset_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    summary_rows = []
    element_counter = Counter()
    failures: list[str] = []
    for _, record in df.iterrows():
        try:
            atoms = load_atoms_raw(record)
        except Exception:
            failures.append(record["selection_id"])
            continue
        summary_rows.append(
            {
                "selection_id": record["selection_id"],
                "label": record["label"],
                "num_atoms": int(len(atoms)),
                "mass_amu": float(atoms.get_masses().sum()),
            }
        )
        element_counter.update(atoms.get_chemical_symbols())
    summary_df = pd.DataFrame(summary_rows)
    elements_df = pd.DataFrame(
        {"element": elem, "count": count} for elem, count in element_counter.most_common()
    )
    return summary_df, elements_df, failures


def render_distribution_charts(
    summary_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    theme: ThemeConfig,
    df: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    import plotly.express as px
    import seaborn as sns

    numeric_cols: list[str] = []
    if df is not None:
        for col in pick_numeric_columns(df):
            series = df[col]
            if series.notna().any():
                numeric_cols.append(col)

    options: list[str] = []
    if not summary_df.empty:
        options.extend(["Number of atoms", "Total mass"])
    if not elements_df.empty:
        options.append("Atom types")
    if numeric_cols:
        options.extend(numeric_cols)

    if not options:
        st.info("Unable to compute distributions for the current dataset.")
        return

    container = st.container()
    with container:
        st.markdown(
            "<div style='padding:10px 12px;border:1px solid rgba(15,23,42,0.15);"
            "border-radius:10px;background:rgba(241,243,249,0.6);margin-bottom:12px;'>"
            "<strong>Distribution settings</strong></div>",
            unsafe_allow_html=True,
        )

        choice = st.selectbox(
            "Distribution",
            options,
            index=min(options.index(st.session_state.get("distribution_choice", options[0])) if "distribution_choice" in st.session_state and st.session_state["distribution_choice"] in options else 0, len(options) - 1),
            key="distribution_choice",
        )

        mode_options = ["Histogram"] if choice == "Atom types" else ["Histogram", "KDE"]
        if "distribution_mode" not in st.session_state or st.session_state["distribution_mode"] not in mode_options:
            st.session_state["distribution_mode"] = mode_options[0]
        mode = st.radio(
            "Plot type",
            mode_options,
            index=mode_options.index(st.session_state["distribution_mode"]),
            key="distribution_mode",
            horizontal=True,
        )

        cols = st.columns(2)
        with cols[0]:
            grid = st.toggle(
                "Show gridlines",
                value=st.session_state.get("distribution_grid", True),
                key="distribution_grid",
            )
        with cols[1]:
            text_size = st.slider(
                "Text size",
                min_value=8,
                max_value=36,
                value=int(st.session_state.get("distribution_text_size", 14)),
                key="distribution_text_size",
            )

        label_key = f"distribution_label_{choice}"
        default_label = st.session_state.get(label_key, choice)
        label = st.text_input(
            "Axis label",
            value=default_label,
            key=label_key,
        )
        if not label.strip():
            label = choice

    color_sequence = [theme.highlight]
    fig = None
    plot_mode = "plotly"

    if choice == "Number of atoms" and not summary_df.empty:
        fig = px.histogram(
            summary_df,
            x="num_atoms",
            nbins=min(30, max(5, summary_df["num_atoms"].nunique())),
            template=theme.plot_template,
            color_discrete_sequence=color_sequence,
        )
        fig.update_xaxes(title_text=label, showgrid=grid)
        fig.update_yaxes(title_text="Count", showgrid=grid)
    elif choice == "Total mass" and not summary_df.empty:
        fig = px.histogram(
            summary_df,
            x="mass_amu",
            nbins=30,
            template=theme.plot_template,
            color_discrete_sequence=color_sequence,
        )
        fig.update_xaxes(title_text=label, showgrid=grid)
        fig.update_yaxes(title_text="Count", showgrid=grid)
    elif choice == "Atom types":
        if elements_df.empty:
            st.info("Element histogram unavailable for this dataset.")
            return
        fig = px.bar(
            elements_df,
            x="element",
            y="count",
            template=theme.plot_template,
            color="element",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_xaxes(title_text=label, showgrid=grid)
        fig.update_yaxes(title_text="Frequency", showgrid=grid)
    else:
        if choice not in df.columns:
            st.info("Selected property unavailable for distribution plotting.")
            return
        col_data = df[choice].dropna()
        if col_data.empty:
            st.info("No values available for this property.")
            return
        values = pd.to_numeric(col_data, errors="coerce").dropna().astype(float)
        if values.empty:
            st.info("No numeric values available for this property.")
            return
        plot_mode = "plotly"
        if mode == "Histogram":
            fig = px.histogram(
                values.to_frame(name=choice),
                x=choice,
                nbins=min(40, max(5, values.nunique() or 1)),
                template=theme.plot_template,
                color_discrete_sequence=color_sequence,
            )
        else:  # KDE via matplotlib/seaborn
            plot_mode = "matplotlib"
            sns.set_theme(style="white")
            fig, ax = plt.subplots(figsize=(12, 8))
            kde = sns.kdeplot(
                values.to_numpy(),
                label=label,
                color=theme.highlight,
                fill=False,
                linewidth=3,
                ax=ax,
            )
            if kde.lines:
                line = kde.lines[-1]
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                ax.fill_between(
                    x_data,
                    0,
                    y_data,
                    color=to_rgba(theme.highlight, alpha=0.35),
                    zorder=1,
                )
            _, y_max = ax.get_ylim()
            if y_max:
                ax.set_ylim(0, y_max * 1.05)
            ax.set_title(f"{label} Distribution", fontsize=max(14, text_size + 4), color=theme.text_color)
            ax.set_xlabel(label, fontsize=max(12, text_size), color=theme.text_color)
            ax.set_ylabel("Density", fontsize=max(12, text_size), color=theme.text_color)
            ax.tick_params(axis="both", labelsize=max(10, text_size - 2), colors=theme.text_color)
            if grid:
                ax.grid(color="#CBD5F5" if theme.background.lower() == "#fafbff" else "#334155", alpha=0.4)
            else:
                ax.grid(False)
            ax.set_facecolor(theme.background)
            fig.patch.set_facecolor(theme.background)
            for spine in ax.spines.values():
                spine.set_color(theme.text_color)
            fig.tight_layout()

    if fig is None:
        st.info("Unable to render the requested distribution.")
        return

    if plot_mode == "plotly":
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor=theme.plot_bg,
            paper_bgcolor=theme.background,
            font=dict(size=max(8, int(text_size)), color=theme.text_color),
            height=600,
            legend=dict(font=dict(color=theme.text_color)),
        )
        fig.update_traces(marker_color=theme.highlight, selector={"type": "histogram"})
        fig.update_xaxes(
            showgrid=grid,
            title_font=dict(color=theme.text_color, size=max(8, int(text_size))),
            tickfont=dict(color=theme.text_color, size=max(8, int(text_size) - 2)),
            gridcolor="rgba(15,23,42,0.12)" if grid else "rgba(0,0,0,0)",
            linecolor="rgba(15,23,42,0.2)",
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=grid,
            title_font=dict(color=theme.text_color, size=max(8, int(text_size))),
            tickfont=dict(color=theme.text_color, size=max(8, int(text_size) - 2)),
            gridcolor="rgba(15,23,42,0.12)" if grid else "rgba(0,0,0,0)",
            linecolor="rgba(15,23,42,0.2)",
            zeroline=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="3DMolViewer (NGL)", layout="wide")
    theme_name = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=1)
    theme = THEMES[theme_name]
    inject_theme_css(theme)
    st.title("3DMolViewer × NGL: Structure-Property Explorer")

    debug_enabled = st.sidebar.checkbox(
        "Performance diagnostics",
        value=st.session_state.get(SESSION_DEBUG_KEY, False),
        key=SESSION_DEBUG_KEY,
    )
    perf_log: Optional[List[Dict[str, Any]]] = [] if debug_enabled else None

    st.sidebar.header("Data Source")
    source_type = st.sidebar.radio("Source", ["XYZ Directory", "ASE Database"], index=0)
    try:
        if source_type == "XYZ Directory":
            default_xyz_dir = DEFAULT_XYZ_DIR if Path(DEFAULT_XYZ_DIR).exists() else ""
            xyz_dir = st.sidebar.text_input("XYZ directory", value=default_xyz_dir)
            default_csv = PACKAGE_ROOT / "test_xyz" / "xyz_properties.csv"
            csv_default_value = str(default_csv) if default_csv.exists() else ""
            csv_path = st.sidebar.text_input(
                "Optional CSV with properties", value=csv_default_value
            )
            if not xyz_dir:
                st.info("Provide a directory containing .xyz files to begin.")
                return
            start = perf_counter()
            df = load_xyz_metadata(xyz_dir, csv_path or None)
 
            _log_perf(
                perf_log,
                "load_xyz_metadata",
                perf_counter() - start,
                directory=str(xyz_dir),
                records=int(len(df)),
            )
            if df.attrs.get("csv_properties_ignored"):
                ignored_path = df.attrs.get("csv_properties_path") or csv_path or "property CSV"
                st.warning(
                    f"Ignored properties from '{ignored_path}' because none of its filenames matched the XYZ files."
                )
            if df.attrs.get("csv_xyz_filtered"):
                skipped = df.attrs["csv_xyz_filtered"]
                st.info(
                    f"Skipped {skipped} structure{'s' if skipped != 1 else ''} without matching properties from the CSV."
                )
            if df.attrs.get("csv_only_count"):
                csv_only = df.attrs["csv_only_count"]
                st.info(
                    f"Loaded {csv_only} CSV-only entr{'y' if csv_only == 1 else 'ies'} without geometry; 3D view will be unavailable."
                )
        else:
            db_path = st.sidebar.text_input("ASE database path")
            if not db_path:
                st.info("Provide a path to an ASE database (.db) file to begin.")
                return
            start = perf_counter()
            df = load_ase_metadata(db_path)
            _log_perf(
                perf_log,
                "load_ase_metadata",
                perf_counter() - start,
                database=str(db_path),
                records=int(len(df)),
            )
    except Exception as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.warning("No records found with the provided inputs.")
        return

    if "__index" not in df.columns:
        df["__index"] = np.arange(len(df))

    if debug_enabled and df.attrs.get("load_timings"):
        with st.sidebar.expander("Load timings", expanded=False):
            for item in df.attrs["load_timings"]:
                meta = {
                    k: v
                    for k, v in item.items()
                    if k not in {"phase", "seconds"} and v not in (None, "")
                }
                suffix = (
                    " (" + ", ".join(f"{k}={v}" for k, v in meta.items()) + ")"
                    if meta
                    else ""
                )
                st.markdown(
                    f"**{item['phase']}** — {item['seconds']:.3f} s{suffix}"
                )

    has_numeric = bool(pick_numeric_columns(df))
    viewer_config = sidebar_controls(
        df, enable_scatter=has_numeric, show_scatter_controls=False
    )

    label_lookup = _label_lookup(df)

    selected_id: Optional[str] = None

    if has_numeric:
        plot_config_state_key = "plot_config_state"
        current_defaults = sanitize_plot_config(
            st.session_state.get(plot_config_state_key), df
        )
        st.session_state[plot_config_state_key] = current_defaults

        settings_visible_key = "plot_settings_visible"
        if settings_visible_key not in st.session_state:
            st.session_state[settings_visible_key] = False

        left_col, right_col = st.columns((1, 1))
        with left_col:
            toggle_label = (
                "Hide plot settings"
                if st.session_state[settings_visible_key]
                else "Show plot settings"
            )
            if st.button(
                f"🎛️ {toggle_label}",
                key="plot_settings_toggle",
                type="primary",
            ):
                st.session_state[settings_visible_key] = (
                    not st.session_state[settings_visible_key]
                )

            if st.session_state[settings_visible_key]:
                plot_config = plot_controls_panel(
                    df, defaults=st.session_state[plot_config_state_key]
                )
            else:
                plot_config = st.session_state[plot_config_state_key]
                summary_bits = [
                    plot_config["mode"],
                    f"X={plot_config['x_axis']}",
                    f"Y={plot_config['y_axis']}",
                ]
                if plot_config["mode"] == "3D" and plot_config.get("z_axis"):
                    summary_bits.append(f"Z={plot_config['z_axis']}")
                if plot_config.get("color_by"):
                    summary_bits.append(f"color={plot_config['color_by']}")
                if plot_config.get("size_by"):
                    summary_bits.append(f"size={plot_config['size_by']}")
                st.caption(" · ".join(summary_bits))
                quick_grid = st.checkbox(
                    "Show gridlines",
                    value=plot_config.get("grid", True),
                    key="plot_quick_grid",
                )
                plot_config["grid"] = bool(quick_grid)

            plot_config = sanitize_plot_config(plot_config, df)
            st.session_state[plot_config_state_key] = plot_config

            build_start = perf_counter()
            fig = build_scatter_figure(
                df,
                x_axis=plot_config["x_axis"],
                y_axis=plot_config["y_axis"],
                z_axis=plot_config.get("z_axis"),
                color_by=plot_config.get("color_by"),
                size_by=plot_config.get("size_by"),
                theme=theme,
                grid=plot_config.get("grid", True),
                text_size=plot_config.get("text_size", 14),
                axis_labels=plot_config.get("axis_labels"),
            )
            _log_perf(
                perf_log,
                "build_scatter_figure",
                perf_counter() - build_start,
                rows=int(len(df)),
            )
            downloads: List[Dict[str, Any]] = []
            download_error: Optional[str] = None
            filename_base = f"{plot_config['y_axis']}_vs_{plot_config['x_axis']}"
            for fmt, label, mime in (
                ("png", "Save PNG", "image/png"),
                ("pdf", "Save PDF", "application/pdf"),
            ):
                try:
                    data = fig.to_image(format=fmt, scale=2)
                except Exception as exc:  # pragma: no cover - depends on kaleido
                    if download_error is None and "kaleido" in str(exc).lower():
                        download_error = "Install `kaleido` to enable PNG/PDF downloads."
                    elif download_error is None:
                        download_error = f"Unable to export plot as {fmt.upper()}."
                    data = None
                if data:
                    downloads.append(
                        {
                            "key": fmt,
                            "label": label,
                            "data": data,
                            "filename": f"{filename_base}.{fmt}",
                            "mime": mime,
                        }
                    )
            if downloads:
                download_error = None

            select_start = perf_counter()
            selected_id = plot_and_select(
                df,
                fig,
                downloads=downloads,
                download_error=download_error,
            )
            _log_perf(
                perf_log,
                "plot_and_select",
                perf_counter() - select_start,
                rows=int(len(df)),
            )
        st.sidebar.divider()
        sidebar_options, truncated = _limit_options(df["selection_id"], selected_id)
        selected_id = st.sidebar.selectbox(
            "Jump directly to a structure",
            sidebar_options,
            index=sidebar_options.index(selected_id)
            if selected_id in sidebar_options
            else 0,
            format_func=lambda sid: label_lookup.get(sid, sid),
        )
        if truncated:
            st.sidebar.caption(
                f"Showing first {SIDEBAR_SELECT_LIMIT} entries. Use the search bar above the viewer for more."
            )
        st.session_state["selected_id"] = selected_id
        viewer_container = right_col
    else:
        st.sidebar.header("Structure Selection")
        prev_selected = st.session_state.get("selected_id")
        sidebar_options, truncated = _limit_options(df["selection_id"], prev_selected)
        selected_id = st.sidebar.selectbox(
            "Choose a structure",
            sidebar_options,
            format_func=lambda sid: label_lookup.get(sid, sid),
            index=sidebar_options.index(prev_selected)
            if prev_selected in sidebar_options
            else 0,
        )
        if truncated:
            st.sidebar.caption(
                f"Showing first {SIDEBAR_SELECT_LIMIT} entries. Use the search bar above the viewer for more."
            )
        st.session_state["selected_id"] = selected_id
        # Center the standalone viewer when no numeric properties are available
        _, viewer_container, _ = st.columns([1, 2, 1])

    if selected_id is None:
        st.info("Select a structure to view its 3D geometry.")
        return

    # Keyboard navigation (global)

    # Keyboard navigation via optional hotkeys component
    nav_evt = None
    if legacy_st_hotkeys is not None:
        hotkey_pressed = legacy_st_hotkeys([
            ("enter", "Next", ["enter", "ArrowRight", ">"]),
            ("backspace", "Previous", ["backspace", "ArrowLeft", "<"]),
        ])
        if hotkey_pressed == "Next":
            nav_evt = "next"
        elif hotkey_pressed == "Previous":
            nav_evt = "prev"
    elif hotkeys is not None:
        # streamlit-hotkeys >=0.5.0 API
        bindings = []
        for binding_id, help_label, keys in (
            ("next", "Next", ("Enter", "ArrowRight", ">")),
            ("prev", "Previous", ("Backspace", "ArrowLeft", "<")),
        ):
            for key_name in keys:
                bindings.append(
                    hotkeys.hk(binding_id, key=key_name, help=f"{help_label} ({key_name})")
                )
        manager_key = "molviewer-navigation"
        hotkeys.activate(bindings, key=manager_key)
        if hotkeys.pressed("next", key=manager_key):
            nav_evt = "next"
        elif hotkeys.pressed("prev", key=manager_key):
            nav_evt = "prev"
    # The rest of the logic to change the selection remains the same
    if nav_evt:
        should_rerun = False
        options = df["selection_id"].tolist()
        if options:
            current_id = st.session_state.get("selected_id", options[0])
            new_id = current_id
            try:
                current_index = options.index(current_id)
                if nav_evt == "prev":
                    new_index = (current_index - 1 + len(options)) % len(options)
                else:  # "next"
                    new_index = (current_index + 1) % len(options)
                new_id = options[new_index]
            except ValueError:
                new_id = options[0]
            if st.session_state.get("selected_id") != new_id:
                st.session_state["selected_id"] = new_id
                should_rerun = True

        if should_rerun:
            st.rerun()

    with viewer_container:

        if source_type == "XYZ Directory":
            selected_id = xyz_navbar(
                df,
                selected_id,
                debug=debug_enabled,
                perf_log=perf_log,
            )
        else:
            selected_id = navigation_controls(df, selected_id)

        record = df.loc[df["selection_id"] == selected_id].iloc[0]
        has_geometry = bool(record.get("has_geometry", True))

        if not has_geometry:
            st.info(
                "No XYZ geometry available for this CSV entry. Nothing to show in the 3D viewer."
            )
            show_details(record, None, theme)
        else:
            load_start = perf_counter()
            atoms = get_atoms(record)
            load_entry = _log_perf(
                perf_log,
                "load_atoms",
                perf_counter() - load_start,
                selection=str(record["selection_id"]),
                source=str(record["source"]),
            )
            if atoms is None:
                if load_entry is not None:
                    load_entry["status"] = "failed"
                st.error("Unable to load atoms for the selected entry.")
                return
            display_atoms = filter_hydrogens(
                atoms, show_hydrogens=viewer_config["show_hydrogens"]
            )
            if len(display_atoms) == 0:
                st.warning("No atoms remain after hiding hydrogens for this structure.")
            else:
                if viewer_config["viewer_engine"] == "3Dmol":
                    html = render_3dmol_view(
                        display_atoms,
                        record.label,
                        theme=theme,
                        height=600,
                        width=700,
                        threedmol_style=viewer_config["threedmol_style"],
                        threedmol_atom_radius=viewer_config["threedmol_atom_radius"],
                        threedmol_bond_radius=viewer_config["threedmol_bond_radius"],
                    )
                    st.components.v1.html(html, height=600, width=700)
                else:
                    try:
                        quality_map = {label: factor for label, factor in SNAPSHOT_QUALITY_OPTIONS}
                        snapshot_factor = quality_map.get(
                            viewer_config.get("snapshot_quality", ""), 1
                        )
                        snapshot_params = {
                            "transparent": bool(viewer_config.get("snapshot_transparent", False)),
                            "factor": snapshot_factor,
                        }
                        render_start = perf_counter()
                        html = render_ngl_view(
                            display_atoms,
                            record.label,
                            theme=theme,
                            sphere_radius=viewer_config["sphere_radius"],
                            bond_radius=viewer_config["bond_radius"],
                            interaction_mode=viewer_config["viewer_mode"],
                            height=600,
                            width=700,
                            representation_style=viewer_config["representation_style"],
                            label_mode=viewer_config["atom_label"],
                            snapshot=snapshot_params,
                        )
                        _log_perf(
                            perf_log,
                            "render_ngl_view",
                            perf_counter() - render_start,
                            atoms=int(len(display_atoms)),
                            label=str(record.label),
                        )
                        st.components.v1.html(html, height=600, width=700)
                    except Exception as exc:  # pragma: no cover - defensive
                        st.error(str(exc))
                        return

            show_measurement_panel(
                display_atoms,
                viewer_config["viewer_mode"],
                key_prefix=f"measure_{selected_id}",
            )
            show_details(record, display_atoms if len(display_atoms) else atoms, theme)

    with st.expander("Dataset distributions", expanded=False):
        stats_allowed = True
        stats_toggle_key = "molviewer_stats_enabled"
        if len(df) > STATS_AUTO_LIMIT:
            stats_allowed = st.checkbox(
                f"Compute dataset statistics (may read all {len(df)} files)",
                value=st.session_state.get(stats_toggle_key, False),
                key=stats_toggle_key,
            )
            if not stats_allowed:
                st.caption(
                    f"Statistics skipped automatically for large datasets (> {STATS_AUTO_LIMIT}). "
                    "Enable the checkbox above to compute them on demand."
                )
        if stats_allowed:
            with st.spinner("Computing dataset statistics..."):
                stats_start = perf_counter()
                summary_df, elements_df, failures = compute_dataset_statistics(df)
                _log_perf(
                    perf_log,
                    "compute_dataset_statistics",
                    perf_counter() - stats_start,
                    summaries=int(len(summary_df)),
                    failures=int(len(failures)),
                )
            render_distribution_charts(summary_df, elements_df, theme, df)
            if failures:
                st.caption("Skipped structures without available geometries: " + ", ".join(failures))

    if debug_enabled and perf_log:
        with st.sidebar.expander("Performance diagnostics log", expanded=False):
            for entry in perf_log:
                meta_items = []
                for key, value in entry.items():
                    if key in {"event", "ms"}:
                        continue
                    if value in (None, ""):
                        continue
                    meta_items.append(f"{key}={value}")
                suffix = f" ({', '.join(meta_items)})" if meta_items else ""
                st.markdown(
                    f"**{entry['event']}** — {float(entry['ms']):.3f} ms{suffix}"
                )


if __name__ == "__main__":
    main()
