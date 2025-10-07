from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.theme_config import ThemeConfig


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


def build_scatter_figure(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    z_axis: Optional[str],
    color_by: Optional[str],
    size_by: Optional[str],
    theme: ThemeConfig,
    grid: bool = True,
    text_size: int = 14,
    axis_labels: Optional[Dict[str, str]] = None,
):
    plot_df = df.replace({pd.NA: None})
    plot_df = plot_df.where(~plot_df.isna(), None)
    hover_data = [col for col in plot_df.columns if col not in BASE_COLUMNS]
    custom_data = plot_df[["selection_id"]]
    if z_axis:
        fig = px.scatter_3d(
            plot_df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color=color_by,
            size=size_by,
            hover_name="label",
            hover_data=hover_data,
            custom_data=custom_data,
            template=theme.plot_template,
            render_mode="webgl"
        )
    else:
        fig = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            size=size_by,
            hover_name="label",
            hover_data=hover_data,
            custom_data=custom_data,
            template=theme.plot_template,
            render_mode="webgl"
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=theme.plot_bg,
        paper_bgcolor=theme.background,
        height=600,
        font=dict(size=max(8, int(text_size))),
    )
    axis_labels = axis_labels or {}
    if z_axis:
        fig.update_scenes(
            xaxis=dict(
                title=str(axis_labels.get("x") or x_axis),
                showgrid=bool(grid),
            ),
            yaxis=dict(
                title=str(axis_labels.get("y") or y_axis),
                showgrid=bool(grid),
            ),
            zaxis=dict(
                title=str(axis_labels.get("z") or z_axis),
                showgrid=bool(grid),
            ),
        )
    else:
        fig.update_layout(
            xaxis=dict(
                title=str(axis_labels.get("x") or x_axis),
                showgrid=bool(grid),
            ),
            yaxis=dict(
                title=str(axis_labels.get("y") or y_axis),
                showgrid=bool(grid),
            ),
        )
    return fig


def pick_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def pick_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLUMNS and not pd.api.types.is_numeric_dtype(df[c])]


def default_plot_config(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = pick_numeric_columns(df)
    if not numeric_cols:
        numeric_cols = ["__index"]
    mode = "2D"
    x_axis = numeric_cols[0]
    y_axis = numeric_cols[min(1, len(numeric_cols) - 1)]
    z_axis = numeric_cols[min(2, len(numeric_cols) - 1)] if len(numeric_cols) >= 3 else None
    return {
        "mode": mode,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "color_by": None,
        "size_by": None,
        "grid": True,
        "text_size": 14,
        "axis_labels": {
            "x": x_axis,
            "y": y_axis,
            "z": z_axis or "",
        },
    }


def sanitize_plot_config(config: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    base = default_plot_config(df)
    if not config:
        return base

    numeric_cols = pick_numeric_columns(df)
    if not numeric_cols:
        numeric_cols = ["__index"]
    categorical_cols = pick_categorical_columns(df)

    sanitized: Dict[str, Any] = {}
    mode = str(config.get("mode", base["mode"]))
    if mode not in {"2D", "3D"}:
        mode = base["mode"]
    if mode == "3D" and len(numeric_cols) < 3:
        mode = "2D"
    sanitized["mode"] = mode

    def pick_numeric(value: Optional[str], default: str) -> str:
        return value if value in numeric_cols else default

    x_axis = pick_numeric(config.get("x_axis"), base["x_axis"])
    y_axis = pick_numeric(config.get("y_axis"), base["y_axis"])
    if mode == "3D":
        default_z = base["z_axis"] if base["z_axis"] in numeric_cols else (
            numeric_cols[min(2, len(numeric_cols) - 1)] if len(numeric_cols) >= 3 else None
        )
        z_candidate = config.get("z_axis")
        z_axis = z_candidate if z_candidate in numeric_cols else default_z
    else:
        z_axis = None
    sanitized.update({
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
    })

    color_options = set(numeric_cols) | set(categorical_cols)
    color_choice = config.get("color_by")
    sanitized["color_by"] = color_choice if color_choice in color_options else None

    size_choice = config.get("size_by")
    sanitized["size_by"] = size_choice if size_choice in numeric_cols else None

    sanitized["grid"] = bool(config.get("grid", base["grid"]))

    try:
        text_size = int(config.get("text_size", base["text_size"]))
    except (TypeError, ValueError):
        text_size = base["text_size"]
    sanitized["text_size"] = max(8, min(48, text_size))

    labels = config.get("axis_labels", {}) or {}
    sanitized["axis_labels"] = {
        "x": str(labels.get("x") or x_axis),
        "y": str(labels.get("y") or y_axis),
        "z": str(labels.get("z") or (z_axis or "")),
    }

    return sanitized


def plot_controls_panel(
    df: pd.DataFrame,
    *,
    key_prefix: str = "plot",
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    defaults = sanitize_plot_config(defaults or default_plot_config(df), df)
    numeric_cols = pick_numeric_columns(df)
    if not numeric_cols:
        numeric_cols = ["__index"]
    categorical_cols = pick_categorical_columns(df)

    if len(numeric_cols) < 3:
        scatter_options = ["2D"]
    else:
        scatter_options = ["2D", "3D"]

    prefixed = lambda name: f"{key_prefix}_{name}"

    def ensure_state(key: str, value: Any) -> None:
        state_key = prefixed(key)
        if state_key not in st.session_state:
            st.session_state[state_key] = value

    ensure_state("dim", defaults["mode"] if defaults["mode"] in scatter_options else scatter_options[0])
    ensure_state("x", defaults["x_axis"])
    ensure_state("y", defaults["y_axis"])
    ensure_state("z", defaults.get("z_axis"))
    ensure_state("color", defaults.get("color_by"))
    ensure_state("size", defaults.get("size_by"))
    ensure_state("grid", defaults.get("grid", True))
    ensure_state("text_size", defaults.get("text_size", 14))
    axis_labels = defaults.get("axis_labels", {})
    ensure_state("label_x", axis_labels.get("x", defaults["x_axis"]))
    ensure_state("label_y", axis_labels.get("y", defaults["y_axis"]))
    ensure_state("label_z", axis_labels.get("z", defaults.get("z_axis") or ""))

    with st.container():
        st.markdown(
            "<div style='padding:10px 12px;border:1px solid rgba(15,23,42,0.15);"
            "border-radius:10px;background:rgba(241,243,249,0.6);margin-bottom:8px;'>"
            "<strong>Plot settings</strong></div>",
            unsafe_allow_html=True,
        )
        scatter_mode = st.radio(
            "Scatter dimensionality",
            scatter_options,
            index=scatter_options.index(st.session_state[prefixed("dim")]),
            horizontal=True,
            key=prefixed("dim"),
        )
        axis_cols = st.columns(2)
        with axis_cols[0]:
            x_axis = st.selectbox(
                "X axis",
                numeric_cols,
                index=numeric_cols.index(
                    st.session_state[prefixed("x")] if st.session_state[prefixed("x")] in numeric_cols else defaults["x_axis"]
                ),
                key=prefixed("x"),
            )
        with axis_cols[1]:
            y_axis = st.selectbox(
                "Y axis",
                numeric_cols,
                index=numeric_cols.index(
                    st.session_state[prefixed("y")] if st.session_state[prefixed("y")] in numeric_cols else defaults["y_axis"]
                ),
                key=prefixed("y"),
            )

        z_axis = None
        if scatter_mode == "3D":
            z_default = (
                st.session_state[prefixed("z")]
                if st.session_state[prefixed("z")] in numeric_cols
                else defaults.get("z_axis")
            )
            if z_default is None and len(numeric_cols) >= 3:
                z_default = numeric_cols[min(2, len(numeric_cols) - 1)]
            z_index = numeric_cols.index(z_default) if z_default in numeric_cols else min(2, len(numeric_cols) - 1)
            z_axis = st.selectbox(
                "Z axis",
                numeric_cols,
                index=z_index,
                key=prefixed("z"),
            )

        color_options = ["None", *numeric_cols, *categorical_cols]
        current_color = st.session_state[prefixed("color")]
        color_index = color_options.index(current_color) if current_color in color_options else 0
        color_choice = st.selectbox(
            "Color column",
            color_options,
            index=color_index,
            key=prefixed("color"),
        )
        color_by = None if color_choice == "None" else color_choice

        size_options = ["Uniform", *numeric_cols]
        current_size = st.session_state[prefixed("size")]
        size_index = size_options.index(current_size) if current_size in size_options else 0
        size_choice = st.selectbox(
            "Size column",
            size_options,
            index=size_index,
            key=prefixed("size"),
        )
        size_by = None if size_choice == "Uniform" else size_choice

        grid = st.toggle(
            "Show gridlines",
            value=bool(st.session_state[prefixed("grid")]),
            key=prefixed("grid"),
        )
        text_size = st.slider(
            "Text size",
            min_value=8,
            max_value=36,
            value=int(st.session_state[prefixed("text_size")]),
            key=prefixed("text_size"),
        )

        label_cols = st.columns(2)
        with label_cols[0]:
            x_label = st.text_input(
                "X label",
                value=st.session_state[prefixed("label_x")],
                key=prefixed("label_x"),
            )
        with label_cols[1]:
            y_label = st.text_input(
                "Y label",
                value=st.session_state[prefixed("label_y")],
                key=prefixed("label_y"),
            )
        if scatter_mode == "3D":
            z_label = st.text_input(
                "Z label",
                value=st.session_state[prefixed("label_z")],
                key=prefixed("label_z"),
            )
        else:
            z_label = st.session_state.get(prefixed("label_z"), defaults.get("axis_labels", {}).get("z", ""))

    config = sanitize_plot_config(
        {
            "mode": scatter_mode,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
            "color_by": color_by,
            "size_by": size_by,
            "grid": grid,
            "text_size": text_size,
            "axis_labels": {
                "x": (x_label or "").strip() or x_axis,
                "y": (y_label or "").strip() or y_axis,
                "z": (z_label or "").strip() or (z_axis or ""),
            },
        },
        df,
    )

    st.session_state[prefixed("config")] = config
    return config


def plot_and_select(
    df: pd.DataFrame,
    fig: Any,
    *,
    downloads: Optional[List[Dict[str, Any]]] = None,
    download_error: Optional[str] = None,
) -> Optional[str]:
    downloads = downloads or []
    st.subheader("Data Exploration")
    if downloads:
        cols = st.columns(len(downloads))
        for idx, item in enumerate(downloads):
            with cols[idx]:
                st.download_button(
                    label=item.get("label", "Download"),
                    data=item["data"],
                    file_name=item.get("filename", "plot"),
                    mime=item.get("mime", "application/octet-stream"),
                    key=f"plot-download-{item.get('key', idx)}",   # <-- use single quotes here
                    use_container_width=True,
                )
    elif download_error:
        st.caption(download_error)
    if df.empty:
        st.info("No data points to display.")
        return st.session_state.get("selected_id")
    selected_id: Optional[str] = st.session_state.get("selected_id")
    label_lookup = _label_lookup(df)

    plot_state = st.plotly_chart(
        fig,
        use_container_width=True,
        key="scatter_plot",
        on_select="rerun",
        selection_mode=("points",),
    )

    def _normalize_points(state: Any) -> List[Dict[str, Any]]:
        if not state:
            return []
        selection = None
        if isinstance(state, dict):
            selection = state.get("selection")
        elif hasattr(state, "selection"):
            selection = getattr(state, "selection")
        elif hasattr(state, "get"):
            try:
                selection = state.get("selection")  # type: ignore[call-arg]
            except Exception:
                selection = None
        if selection is None:
            return []
        points = None
        if isinstance(selection, dict):
            points = selection.get("points")
        elif hasattr(selection, "points"):
            points = selection.points  # type: ignore[attr-defined]
        elif hasattr(selection, "get"):
            try:
                points = selection.get("points")  # type: ignore[call-arg]
            except Exception:
                points = None
        return points or []

    points = _normalize_points(plot_state)
    if points:
        first_point = points[0]
        candidate: Optional[str] = None
        custom_data = first_point.get("customdata")
        if isinstance(custom_data, (list, tuple)) and custom_data:
            candidate = custom_data[0]
        point_index = first_point.get("point_number") or first_point.get("pointNumber")
        if candidate is None and isinstance(point_index, (int, float)):
            try:
                candidate = df["selection_id"].iloc[int(point_index)]
            except Exception:
                candidate = None
        if candidate and candidate in df["selection_id"].values:
            selected_id = candidate

    # Provide a dropdown fallback in case users prefer manual selection.
    fallback_options = df["selection_id"]
    selectbox_key = "plot_selectbox_fallback"
    if selected_id and st.session_state.get(selectbox_key) != selected_id:
        st.session_state[selectbox_key] = selected_id
    selection = st.selectbox(
        "Choose a structure",
        fallback_options,
        format_func=lambda sid: label_lookup.get(sid, sid),
        index=fallback_options.tolist().index(selected_id)
        if selected_id in fallback_options.tolist()
        else 0,
        key=selectbox_key,
    )
    selected_id = selection

    if selected_id is None and not df.empty:
        selected_id = df["selection_id"].iloc[0]
    st.session_state["selected_id"] = selected_id
    return selected_id

def _label_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return pd.Series(df["label"].values, index=df["selection_id"]).to_dict()
