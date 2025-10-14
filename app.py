from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import io
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple
from streamlit.components.v1 import html
import numpy as np
import pandas as pd
import streamlit as st
from ase import Atoms
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

DATASET_STATE_KEY = "_molviewer_dataset_state"
LITE_DATASET_STATE_KEY = "_molviewer_lite_dataset_state"
PLOT_CACHE_KEY = "_molviewer_plot_cache"
PLOT_CACHE_LIMIT = 8
PLOT_SIGNATURE_KEYS = (
    "x_axis",
    "y_axis",
    "z_axis",
    "color_by",
    "size_by",
    "grid",
    "marker_size",
    "color_scheme",
    "x_scale",
    "x_min",
    "x_max",
    "y_scale",
    "y_min",
    "y_max",
    "z_scale",
    "z_min",
    "z_max",
)


# Import refactored modules
from src.theme_config import THEMES, ThemeConfig, inject_theme_css
from src.data_loader import load_xyz_metadata, load_ase_metadata, get_atoms, load_atoms_raw
from src.viewer_utils import render_ngl_view, render_3dmol_view, filter_hydrogens, SNAPSHOT_QUALITY_OPTIONS
from src.plot_utils import build_scatter_figure, plot_controls_panel, plot_and_select, pick_numeric_columns, sanitize_plot_config
from src.ui_components import (
    sidebar_controls,
    show_measurement_panel,
    show_details,
    navigation_controls,
    show_multi_details,
)


def _normalize_path(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:  # pragma: no cover - defensive
        return str(Path(path).expanduser())


def _make_dataset_key(
    source_type: str,
    *,
    xyz_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Tuple[str, str, str]:
    if source_type == "XYZ Directory":
        return ("xyz", _normalize_path(xyz_dir), _normalize_path(csv_path))
    return ("ase", _normalize_path(db_path), "")


def _get_dataset_state() -> Dict[str, Any]:
    return st.session_state.setdefault(DATASET_STATE_KEY, {})


def _make_version_key(version: float) -> str:
    return f"{version:.6f}".replace(".", "_")


def _ensure_dataset_loaded(
    source_type: str,
    *,
    xyz_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    db_path: Optional[str] = None,
    force_reload: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    dataset_state = _get_dataset_state()
    dataset_key = _make_dataset_key(
        source_type,
        xyz_dir=xyz_dir,
        csv_path=csv_path,
        db_path=db_path,
    )
    if (
        not force_reload
        and dataset_state.get("key") == dataset_key
        and "df" in dataset_state
    ):
        if "version_key" not in dataset_state and "version" in dataset_state:
            dataset_state["version_key"] = _make_version_key(dataset_state["version"])
        return dataset_state["df"], dataset_state, True

    load_start = perf_counter()
    if source_type == "XYZ Directory":
        df = load_xyz_metadata(xyz_dir or "", csv_path or None)
        source_meta = {"directory": str(xyz_dir or "")}
    else:
        df = load_ase_metadata(db_path or "")
        source_meta = {"database": str(db_path or "")}
    duration = perf_counter() - load_start

    dataset_state.clear()
    dataset_state.update(
        {
            "key": dataset_key,
            "df": df,
            "version": perf_counter(),
            "load_duration": duration,
            "stats": {},
            "source_meta": source_meta,
        }
    )
    dataset_state["version_key"] = _make_version_key(dataset_state["version"])
    st.session_state.pop(PLOT_CACHE_KEY, None)
    st.session_state.pop("selected_id", None)
    st.session_state.pop("selected_ids", None)
    st.session_state.pop("geometry_cache", None)
    return df, dataset_state, False


def _load_lite_dataset(
    uploaded_file: Any,
    *,
    separator: str,
    force_reload: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    """Load a lightweight CSV dataset uploaded by the user."""
    lite_state = st.session_state.setdefault(LITE_DATASET_STATE_KEY, {})
    if force_reload:
        lite_state.clear()
        st.session_state.pop("geometry_cache", None)
    if uploaded_file is None:
        return pd.DataFrame(), lite_state, False

    content = uploaded_file.getvalue()
    if not content:
        raise ValueError("Uploaded file is empty.")

    cache_key = hashlib.sha256(content + separator.encode("utf-8")).hexdigest()
    if lite_state.get("file_hash") == cache_key and "df" in lite_state:
        return lite_state["df"], lite_state, True

    load_start = perf_counter()
    try:
        text = content.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = content.decode("latin-1")
        encoding = "latin-1"

    try:
        df = pd.read_csv(io.StringIO(text), sep=separator)
    except Exception as exc:  # pragma: no cover - depends on input file
        raise ValueError(f"Unable to parse uploaded CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Uploaded CSV contains no rows.")

    df = df.copy()
    original_cols = df.columns.tolist()

    if "__index" not in df.columns:
        df["__index"] = np.arange(len(df))

    if "selection_id" in df.columns:
        df.loc[:, "selection_id"] = df["selection_id"].astype(str)
        if df["selection_id"].duplicated().any() or df["selection_id"].eq("").any():
            df.loc[:, "selection_id"] = [f"lite_{idx}" for idx in range(len(df))]
    else:
        df["selection_id"] = [f"lite_{idx}" for idx in range(len(df))]

    label_source: Optional[str] = None
    for col in original_cols:
        if col in BASE_COLUMNS or col == "selection_id":
            continue
        if df[col].dtype == object:
            label_source = col
            break
    if label_source is None:
        label_source = original_cols[0] if original_cols else "__index"

    if "label" in df.columns:
        df.loc[:, "label"] = df["label"].fillna("").replace({pd.NA: ""}).astype(str)
    else:
        df["label"] = df[label_source].fillna("").replace({pd.NA: ""}).astype(str)

    if "identifier" in df.columns:
        df.loc[:, "identifier"] = (
            df["identifier"].fillna("").replace({pd.NA: ""}).astype(str)
        )
    else:
        df["identifier"] = df["label"]

    if "has_geometry" in df.columns:
        df.loc[:, "has_geometry"] = df["has_geometry"].fillna(False).astype(bool)
    else:
        df["has_geometry"] = False

    if "source" in df.columns:
        df.loc[:, "source"] = df["source"].fillna("").replace({pd.NA: ""}).astype(str)
    else:
        df["source"] = "lite"

    if "path" in df.columns:
        df.loc[:, "path"] = df["path"].fillna("").replace({pd.NA: ""}).astype(str)
    else:
        df["path"] = ""

    if "db_path" in df.columns:
        df.loc[:, "db_path"] = df["db_path"].fillna("").replace({pd.NA: ""}).astype(str)
    else:
        df["db_path"] = ""

    if "db_id" in df.columns:
        df.loc[:, "db_id"] = df["db_id"].fillna("").replace({pd.NA: ""}).astype(str)
    else:
        df["db_id"] = ""

    duration = perf_counter() - load_start
    version = perf_counter()

    lite_state.clear()
    lite_state.update(
        {
            "df": df,
            "file_hash": cache_key,
            "load_duration": duration,
            "version": version,
            "key": ("lite", uploaded_file.name or "<uploaded>", ""),
            "stats": {},
            "source_meta": {
                "filename": uploaded_file.name or "",
                "encoding": encoding,
                "separator": separator,
                "rows": len(df),
                "columns": len(df.columns),
            },
        }
    )
    lite_state["version_key"] = _make_version_key(version)
    st.session_state.pop("geometry_cache", None)
    return df, lite_state, False


def _resolve_geometry_path(raw_value: Any, base_dir: str) -> tuple[str, bool]:
    """Resolve a geometry file path from a CSV value."""
    if raw_value is None:
        return "", False
    value = str(raw_value).strip()
    if not value or value.lower() in {"nan", "none"}:
        return "", False
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and base_dir:
        candidate = Path(base_dir).expanduser() / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except Exception:  # pragma: no cover - defensive
        resolved = candidate
    exists = resolved.exists()
    return str(resolved), exists


def _configure_lite_geometry(
    df: pd.DataFrame, dataset_state: Dict[str, Any]
) -> pd.DataFrame:
    """Allow users to map lite CSV columns to geometry files."""
    string_columns = [
        col
        for col in df.select_dtypes(include=["object", "string"]).columns
        if col not in BASE_COLUMNS and col != "selection_id"
    ]
    geometry_state: Dict[str, Any] = dataset_state.setdefault(
        "lite_geometry_config", {}
    )
    previous_config = {
        "column": geometry_state.get("column"),
        "base_dir": geometry_state.get("base_dir", ""),
    }
    with st.sidebar.expander("Lite geometry mapping", expanded=False):
        if not string_columns:
            st.caption("No text columns detected for geometry mapping.")
            selected_column = None
            base_directory = ""
        else:
            options = ["None"] + sorted(string_columns)
            current_column = geometry_state.get("column")
            if current_column not in string_columns:
                current_column = None
            default_index = options.index(current_column) if current_column else 0
            selected_option = st.selectbox(
                "Column containing geometry file paths",
                options,
                index=default_index,
            )
            selected_column = (
                selected_option if selected_option != "None" else None
            )
            base_directory = st.text_input(
                "Base directory (optional)",
                value=geometry_state.get("base_dir", ""),
                help="Prepended to relative paths before file lookup.",
            )
            summary = dataset_state.get("lite_geometry_report")
            if summary and selected_column:
                st.caption(
                    f"Found {summary.get('available', 0)} geometries "
                    f"({summary.get('missing', 0)} missing files)."
                )

    new_config = {"column": selected_column, "base_dir": base_directory.strip()}
    geometry_changed = new_config != previous_config
    geometry_state.update(new_config)

    df_modified = df.copy()

    # Always reset previously injected lite geometry rows
    previous_mask = df_modified["source"] == "lite_xyz"
    if previous_mask.any():
        df_modified.loc[previous_mask, "source"] = "lite"
        df_modified.loc[previous_mask, "path"] = ""
        df_modified.loc[previous_mask, "has_geometry"] = False

    report = {"enabled": False, "available": 0, "missing": 0, "total": len(df_modified)}

    if selected_column:
        report["enabled"] = True
        resolved_series = df_modified[selected_column].apply(
            lambda value: _resolve_geometry_path(value, new_config["base_dir"])
        )
        resolved_paths = resolved_series.apply(lambda pair: pair[0])
        exists_series = resolved_series.apply(lambda pair: bool(pair[1]))
        nonempty_mask = resolved_paths.astype(str).str.len() > 0

        if nonempty_mask.any():
            df_modified.loc[nonempty_mask, "path"] = resolved_paths[nonempty_mask]
            df_modified.loc[nonempty_mask, "source"] = "lite_xyz"
            df_modified.loc[nonempty_mask, "has_geometry"] = exists_series[nonempty_mask]
            report["available"] = int(exists_series[nonempty_mask].sum())
            missing_mask = nonempty_mask & ~exists_series
            report["missing"] = int(missing_mask.sum())

    dataset_state["lite_geometry_report"] = report
    dataset_state["df"] = df_modified

    if geometry_changed:
        dataset_state["version"] = perf_counter()
        dataset_state["version_key"] = _make_version_key(dataset_state["version"])
        dataset_state["stats"] = {}
        st.session_state.pop(PLOT_CACHE_KEY, None)
        st.session_state.pop("geometry_cache", None)

    return df_modified


def _load_atoms_cached(
    record: pd.Series,
    dataset_state: Dict[str, Any],
    perf_log: Optional[List[Dict[str, Any]]],
    event_base: str,
) -> Optional[Atoms]:
    """Load atoms, caching results per dataset version and selection."""
    cache: Dict[Tuple[str, str], Optional[Atoms]] = st.session_state.setdefault(
        "geometry_cache", {}
    )
    version_key = dataset_state.get("version_key", "global")
    selection_key = str(record.get("selection_id"))
    cache_key = (version_key, selection_key)

    if cache_key in cache:
        _log_perf(
            perf_log,
            f"{event_base}_cached",
            0.0,
            selection=selection_key,
            source=str(record.get("source")),
        )
        return cache[cache_key]

    load_start = perf_counter()
    atoms = get_atoms(record)
    duration = perf_counter() - load_start
    entry = _log_perf(
        perf_log,
        event_base,
        duration,
        selection=selection_key,
        source=str(record.get("source")),
    )
    if atoms is None and entry is not None:
        entry["status"] = "failed"
    cache[cache_key] = atoms
    return atoms


def _normalize_plot_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


def _make_plot_cache_key(
    dataset_state: Dict[str, Any],
    plot_config: Dict[str, Any],
    theme_name: str,
) -> Tuple[Any, ...]:
    signature = tuple(
        (key, _normalize_plot_value(plot_config.get(key)))
        for key in PLOT_SIGNATURE_KEYS
    )
    return (dataset_state.get("key"), dataset_state.get("version"), theme_name, signature)


def _get_scatter_figure(
    df: pd.DataFrame,
    plot_config: Dict[str, Any],
    theme: ThemeConfig,
    theme_name: str,
    dataset_state: Dict[str, Any],
) -> Tuple[Any, bool, float]:
    cache: OrderedDict = st.session_state.setdefault(PLOT_CACHE_KEY, OrderedDict())
    cache_key = _make_plot_cache_key(dataset_state, plot_config, theme_name)
    cached_fig = cache.get(cache_key)
    if cached_fig is not None:
        cache.move_to_end(cache_key)
        return cached_fig, True, 0.0

    build_start = perf_counter()
    fig = build_scatter_figure(df, theme=theme, **plot_config)
    duration = perf_counter() - build_start
    cache[cache_key] = fig
    cache.move_to_end(cache_key)
    while len(cache) > PLOT_CACHE_LIMIT:
        cache.popitem(last=False)
    return fig, False, duration


def _get_dataset_statistics(
    df: pd.DataFrame,
    dataset_state: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], bool]:
    stats_bucket = dataset_state.setdefault("stats", {})
    dataset_version = dataset_state.get("version")
    if stats_bucket.get("version") == dataset_version:
        return (
            stats_bucket.get("summary", pd.DataFrame()),
            stats_bucket.get("elements", pd.DataFrame()),
            stats_bucket.get("failures", []),
            True,
        )

    summary_df, elements_df, failures = compute_dataset_statistics(df)
    stats_bucket.update(
        {
            "version": dataset_version,
            "summary": summary_df,
            "elements": elements_df,
            "failures": failures,
        }
    )
    return summary_df, elements_df, failures, False


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
            "Choose a structure",
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
    source_options = ["Lite CSV Upload", "XYZ Directory", "ASE Database"]
    source_type = st.sidebar.radio("Source", source_options, index=1)
    reload_requested = st.sidebar.button("Reload data", key="molviewer_reload")
    if reload_requested:
        st.session_state.pop(DATASET_STATE_KEY, None)
        st.session_state.pop(LITE_DATASET_STATE_KEY, None)
        st.session_state.pop(PLOT_CACHE_KEY, None)
        st.session_state.pop("selected_id", None)
        st.session_state.pop("selected_ids", None)
        st.session_state.pop("geometry_cache", None)

    dataset_state: Dict[str, Any] = {}

    try:
        if source_type == "Lite CSV Upload":
            st.sidebar.caption(
                "Upload a CSV file for quick property exploration. "
                "A `label` column will be inferred if missing."
            )
            separator_options = {
                "Comma (,)": ",",
                "Semicolon (;)": ";",
                "Tab (\\t)": "\t",
                "Pipe (|)": "|",
            }
            separator_choice = st.sidebar.selectbox(
                "Separator",
                list(separator_options.keys()),
                index=0,
            )
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV file",
                type=["csv", "txt"],
                key="lite_dataset_upload",
            )
            if uploaded_file is None:
                st.info("Upload a CSV file to begin.")
                return

            df, dataset_state, dataset_cache_hit = _load_lite_dataset(
                uploaded_file,
                separator=separator_options[separator_choice],
                force_reload=reload_requested,
            )
            if perf_log is not None:
                event_name = (
                    "load_lite_dataset_cached"
                    if dataset_cache_hit
                    else "load_lite_dataset"
                )
                duration = (
                    0.0 if dataset_cache_hit else dataset_state.get("load_duration", 0.0)
                )
                _log_perf(
                    perf_log,
                    event_name,
                    duration,
                    filename=getattr(uploaded_file, "name", ""),
                    records=int(len(df)),
                )
            df = _configure_lite_geometry(df, dataset_state)
        elif source_type == "XYZ Directory":
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
            df, dataset_state, dataset_cache_hit = _ensure_dataset_loaded(
                source_type,
                xyz_dir=xyz_dir,
                csv_path=csv_path,
                force_reload=reload_requested,
            )
            if perf_log is not None:
                event_name = "load_xyz_metadata_cached" if dataset_cache_hit else "load_xyz_metadata"
                duration = 0.0 if dataset_cache_hit else dataset_state.get("load_duration", 0.0)
                _log_perf(
                    perf_log,
                    event_name,
                    duration,
                    directory=_normalize_path(xyz_dir),
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
            df, dataset_state, dataset_cache_hit = _ensure_dataset_loaded(
                source_type,
                db_path=db_path,
                force_reload=reload_requested,
            )
            if perf_log is not None:
                event_name = "load_ase_metadata_cached" if dataset_cache_hit else "load_ase_metadata"
                duration = 0.0 if dataset_cache_hit else dataset_state.get("load_duration", 0.0)
                _log_perf(
                    perf_log,
                    event_name,
                    duration,
                    database=_normalize_path(db_path),
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

    version_key = dataset_state.get("version_key", "global")

    has_numeric = bool(pick_numeric_columns(df))
    viewer_config = sidebar_controls(
        df, enable_scatter=has_numeric, show_scatter_controls=False
    )

    label_lookup = _label_lookup(df)

    existing_selected_id = st.session_state.get("selected_id")
    selected_id: Optional[str] = (
        existing_selected_id if existing_selected_id in df["selection_id"].values else None
    )
    selected_ids: List[str] = [
        sid for sid in st.session_state.get("selected_ids", []) if sid in df["selection_id"].values
    ]
    multi_select_enabled = False

    if has_numeric:
        plot_config_state_key = f"plot_config_state_{version_key}"
        current_defaults = sanitize_plot_config(
            st.session_state.get(plot_config_state_key), df
        )
        st.session_state[plot_config_state_key] = current_defaults

        left_col, right_col = st.columns((1, 1))
        with left_col:
            plot_config = plot_controls_panel(
                df,
                key_prefix=f"plot_{version_key}",
                defaults=st.session_state.get(plot_config_state_key),
            )

            plot_config = sanitize_plot_config(plot_config, df)
            st.session_state[plot_config_state_key] = plot_config

            selection_mode_options = ("Single selection", "Multi selection")
            selection_mode_value = st.radio(
                "Selection mode",
                selection_mode_options,
                key="molviewer_selection_mode",
                horizontal=True,
            )
            multi_select_enabled = selection_mode_value == selection_mode_options[1]
            if multi_select_enabled:
                st.caption(
                    "Tip: use box/lasso selection or click multiple points repeatedly to build your list."
                )

            fig, fig_cached, build_duration = _get_scatter_figure(
                df,
                plot_config,
                theme,
                theme_name,
                dataset_state,
            )
            if perf_log is not None:
                event_name = "build_scatter_figure_cached" if fig_cached else "build_scatter_figure"
                _log_perf(
                    perf_log,
                    event_name,
                    build_duration,
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
            selected_id, selected_ids = plot_and_select(
                df,
                fig,
                downloads=downloads,
                download_error=download_error,
                selection_mode="multi" if multi_select_enabled else "single",
            )
            _log_perf(
                perf_log,
                "plot_and_select",
                perf_counter() - select_start,
                rows=int(len(df)),
            )
        st.sidebar.divider()
        st.session_state["selected_id"] = selected_id
        st.session_state["selected_ids"] = selected_ids
        if multi_select_enabled:
            viewer_container = st.container()
        else:
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
        selected_ids = [selected_id] if selected_id else []
        st.session_state["selected_ids"] = selected_ids
        # Center the standalone viewer when no numeric properties are available
        _, viewer_container, _ = st.columns([1, 2, 1])

    if not selected_ids:
        st.info("Select a structure to view its 3D geometry.")
        return
    if selected_id not in selected_ids:
        selected_id = selected_ids[-1]

    # Keyboard navigation (global)

    # Keyboard navigation via optional hotkeys component
    nav_evt = None
    if not multi_select_enabled:
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
        if multi_select_enabled:
            st.markdown("**Selected structures**")
            manage_cols = st.columns([4, 1])
            removal_requests: List[str] = []
            with manage_cols[0]:
                for idx, sid in enumerate(selected_ids, start=1):
                    human_label = label_lookup.get(sid, sid)
                    st.caption(f"{idx}. {human_label}")
            with manage_cols[1]:
                if st.button("Clear selection", key="clear-multi-selection", use_container_width=True):
                    st.session_state["selected_ids"] = []
                    st.session_state["selected_id"] = None
                    st.rerun()

            if len(selected_ids) > 1:
                layout_choice = st.radio(
                    "Viewer layout",
                    ("Grid", "Tabs"),
                    key="molviewer_multi_layout",
                    horizontal=True,
                )
            else:
                layout_choice = "Grid"
                st.session_state.setdefault("molviewer_multi_layout", layout_choice)

            st.caption(
                "Enable the 3D viewer below to load structures on demand. Switch back to single selection to access NGL tools and measurements."
            )

            dataset_has_geometry = (
                bool(df["has_geometry"].astype(bool).any())
                if "has_geometry" in df.columns
                else False
            )
            viewer_state_key = f"viewer_enabled_{version_key}"
            default_viewer_enabled = st.session_state.get(viewer_state_key, False)
            if not dataset_has_geometry and viewer_state_key in st.session_state:
                st.session_state[viewer_state_key] = False
            viewer_checkbox = st.checkbox(
                "Enable 3D viewer",
                value=default_viewer_enabled if dataset_has_geometry else False,
                key=viewer_state_key,
                disabled=not dataset_has_geometry,
                help="Load molecular geometries on demand."
                + ("" if dataset_has_geometry else " No geometries detected for this dataset."),
            )
            viewer_enabled = bool(viewer_checkbox and dataset_has_geometry)

            viewer_style = viewer_config.get("threedmol_style") or "stick"
            viewer_atom_radius = viewer_config.get("threedmol_atom_radius")
            viewer_bond_radius = viewer_config.get("threedmol_bond_radius")
            if viewer_style == "Ball and Stick":
                viewer_atom_radius = viewer_atom_radius or 0.25
                viewer_bond_radius = viewer_bond_radius or 0.08

            label_modes = (
                [viewer_config["atom_label"]] if viewer_config["atom_label"] else None
            )
            viewer_height = 420
            viewer_width = 420

            geometry_message: Optional[str] = None
            if not dataset_has_geometry:
                geometry_message = "No geometries were detected for this dataset. Use the tables below for metadata instead."
            elif not viewer_enabled:
                geometry_message = "Enable the 3D viewer above to fetch geometries for the current selection."

            records_for_tables: List[pd.Series] = []
            atoms_for_tables: Dict[str, Optional[Atoms]] = {}
            viewer_entries: List[Dict[str, Any]] = []

            for order, sid in enumerate(selected_ids):
                row = df.loc[df["selection_id"] == sid]
                if row.empty:
                    continue
                record = row.iloc[0]
                records_for_tables.append(record)
                atoms_for_tables.setdefault(sid, None)

                if geometry_message is not None:
                    continue

                has_geometry = bool(record.get("has_geometry", True))
                if not has_geometry:
                    viewer_entries.append(
                        {
                            "selection_id": sid,
                            "label": record.label,
                            "error": "No XYZ geometry available for this entry.",
                            "order": order,
                        }
                    )
                    continue
                atoms = _load_atoms_cached(
                    record,
                    dataset_state,
                    perf_log,
                    "load_atoms_multi",
                )
                if atoms is None:
                    viewer_entries.append(
                        {
                            "selection_id": sid,
                            "label": record.label,
                            "error": "Unable to load atoms for this entry.",
                            "order": order,
                        }
                    )
                    continue
                filtered_atoms = filter_hydrogens(
                    atoms, show_hydrogens=viewer_config["show_hydrogens"]
                )
                atoms_for_preview = filtered_atoms if len(filtered_atoms) else atoms
                viewer_entries.append(
                    {
                        "selection_id": sid,
                        "label": record.label,
                        "record": record,
                        "atoms": atoms,
                        "display_atoms": atoms_for_preview,
                        "hydrogen_only": len(filtered_atoms) == 0
                        and not viewer_config["show_hydrogens"],
                        "order": order,
                    }
                )
                atoms_for_tables[sid] = atoms_for_preview

            if geometry_message is not None:
                st.info(geometry_message)
            else:
                if not viewer_entries:
                    st.info(
                        "No geometries could be loaded for the current selection. Try picking different points."
                    )
                else:
                    def _render_entry(entry: Dict[str, Any]) -> None:
                        header_cols = st.columns([4, 1])
                        with header_cols[0]:
                            st.markdown(f"**{entry['label']}**")
                            st.caption(f"Selection ID: {entry['selection_id']}")
                        with header_cols[1]:
                            if st.button(
                                "Remove",
                                key=f"remove-viewer-{entry['selection_id']}",
                                use_container_width=True,
                            ):
                                removal_requests.append(entry["selection_id"])
                        if entry.get("error"):
                            st.warning(entry["error"])
                            return
                        if entry.get("hydrogen_only"):
                            st.caption(
                                "All atoms were filtered by the hydrogen toggle; showing the full structure instead."
                            )
                        display_atoms = entry["display_atoms"]
                        render_start = perf_counter()
                        html = render_3dmol_view(
                            display_atoms,
                            entry["label"],
                            theme=theme,
                            height=viewer_height,
                            width=viewer_width,
                            threedmol_style=viewer_style,
                            threedmol_atom_radius=viewer_atom_radius,
                            threedmol_bond_radius=viewer_bond_radius,
                            label_modes=label_modes,
                            container_id=f"threedmol-{entry['selection_id']}-{entry['order']}",
                        )
                        _log_perf(
                            perf_log,
                            "render_3dmol_view_multi",
                            perf_counter() - render_start,
                            atoms=int(len(display_atoms)),
                            label=str(entry["label"]),
                        )
                        st.components.v1.html(html, height=viewer_height, width=viewer_width)

                    if layout_choice == "Tabs" and len(viewer_entries) > 1:
                        tabs = st.tabs([entry["label"] for entry in viewer_entries])
                        for tab, entry in zip(tabs, viewer_entries):
                            with tab:
                                _render_entry(entry)
                    else:
                        per_row = 2
                        for start in range(0, len(viewer_entries), per_row):
                            cols = st.columns(
                                min(per_row, len(viewer_entries) - start)
                            )
                            for col, entry in zip(
                                cols, viewer_entries[start : start + per_row]
                            ):
                                with col:
                                    _render_entry(entry)

                if removal_requests:
                    st.session_state["selected_ids"] = [
                        sid for sid in selected_ids if sid not in removal_requests
                    ]
                    st.session_state["selected_id"] = (
                        st.session_state["selected_ids"][-1]
                        if st.session_state["selected_ids"]
                        else None
                    )
                    st.rerun()

            if records_for_tables:
                records_df = pd.DataFrame(records_for_tables)
                records_df = records_df.set_index("selection_id")
                index_order = [
                    sid for sid in selected_ids if sid in records_df.index
                ]
                records_df = records_df.loc[index_order].reset_index()
                show_multi_details(records_df, atoms_for_tables, theme)
            if geometry_message is not None:
                return
        else:
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

            dataset_has_geometry = (
                bool(df["has_geometry"].astype(bool).any())
                if "has_geometry" in df.columns
                else False
            )
            viewer_state_key = f"viewer_enabled_{version_key}"
            default_viewer_enabled = st.session_state.get(viewer_state_key, False)
            if not dataset_has_geometry and viewer_state_key in st.session_state:
                st.session_state[viewer_state_key] = False
            viewer_checkbox = st.checkbox(
                "Enable 3D viewer",
                value=default_viewer_enabled if dataset_has_geometry else False,
                key=viewer_state_key,
                disabled=not dataset_has_geometry,
                help="Load molecular geometries on demand."
                + ("" if dataset_has_geometry else " No geometries detected for this dataset."),
            )
            viewer_enabled = bool(viewer_checkbox and dataset_has_geometry)

            if not dataset_has_geometry:
                st.info(
                    "No geometries were detected for this dataset. Use the metadata below instead."
                )
                show_details(record, None, theme)
            elif not viewer_enabled:
                st.info(
                    "Enable the 3D viewer above to fetch geometries for the selected structure."
                )
                show_details(record, None, theme)
            else:
                has_geometry = bool(record.get("has_geometry", True))

                if not has_geometry:
                    st.info(
                        "No XYZ geometry available for this CSV entry. Nothing to show in the 3D viewer."
                    )
                    show_details(record, None, theme)
                else:
                    atoms = _load_atoms_cached(
                        record,
                        dataset_state,
                        perf_log,
                        "load_atoms",
                    )
                    if atoms is None:
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
                                height=720,
                                width=840,
                                threedmol_style=viewer_config["threedmol_style"],
                                threedmol_atom_radius=viewer_config["threedmol_atom_radius"],
                                threedmol_bond_radius=viewer_config["threedmol_bond_radius"],
                                label_modes=[viewer_config["atom_label"]] if viewer_config["atom_label"] else None,
                            )
                            st.components.v1.html(html, height=720, width=840)
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
                                    height=720,
                                    width=840,
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
                                st.components.v1.html(html, height=720, width=840)
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
            compute_needed = dataset_state.setdefault("stats", {}).get("version") != dataset_state.get("version")
            stats_start = perf_counter()
            if compute_needed:
                with st.spinner("Computing dataset statistics..."):
                    summary_df, elements_df, failures, stats_cached = _get_dataset_statistics(df, dataset_state)
                duration = perf_counter() - stats_start
            else:
                summary_df, elements_df, failures, stats_cached = _get_dataset_statistics(df, dataset_state)
                duration = 0.0
            if perf_log is not None:
                event_name = "dataset_statistics_cache_hit" if stats_cached else "compute_dataset_statistics"
                _log_perf(
                    perf_log,
                    event_name,
                    duration,
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
