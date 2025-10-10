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


@st.cache_data
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
    marker_size: int = 10,
    color_scheme: Optional[str] = "Viridis",
    x_scale: str = "linear",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_scale: str = "linear",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    z_scale: str = "linear",
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    **kwargs,
):
    plot_df = df.replace({pd.NA: None})
    plot_df = plot_df.where(~plot_df.isna(), None)
    hover_data = [col for col in plot_df.columns if col not in BASE_COLUMNS]
    custom_data = plot_df[["selection_id"]]

    is_3d = z_axis is not None
    scatter_fn = px.scatter_3d if is_3d else px.scatter
    
    plot_args = {
        "x": x_axis,
        "y": y_axis,
        "color": color_by,
        "hover_name": "label",
        "hover_data": hover_data,
        "custom_data": custom_data,
        "template": theme.plot_template,
    }
    if size_by:
        plot_args["size"] = size_by
    if color_by:
        plot_args["color_continuous_scale"] = color_scheme
    if is_3d:
        plot_args["z"] = z_axis
    else:
        plot_args["render_mode"] = "webgl"
    
    fig = scatter_fn(plot_df, **plot_args)

    if not size_by:
        fig.update_traces(marker=dict(size=marker_size))

    fig.update_layout(
        margin=dict(l=50, r=10, t=40, b=10),
        plot_bgcolor=theme.plot_bg,
        paper_bgcolor=theme.background,
        height=840,
        font=dict(size=12),
        coloraxis_colorbar=dict(len=1, y=0.5, yanchor='middle'),
    )

    xaxis_config = dict(title=x_axis, showgrid=grid, type=x_scale)
    if x_min is not None or x_max is not None:
        xaxis_config['range'] = [x_min, x_max]

    yaxis_config = dict(title=y_axis, showgrid=grid, type=y_scale)
    if y_min is not None or y_max is not None:
        yaxis_config['range'] = [y_min, y_max]

    if is_3d:
        zaxis_config = dict(title=z_axis, showgrid=grid, type=z_scale)
        if z_min is not None or z_max is not None:
            zaxis_config['range'] = [z_min, z_max]
        
        fig.update_scenes(
            xaxis=xaxis_config,
            yaxis=yaxis_config,
            zaxis=zaxis_config,
        )
    else:
        fig.update_layout(
            xaxis=xaxis_config,
            yaxis=yaxis_config,
        )
        
    return fig


def pick_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def pick_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLUMNS and not pd.api.types.is_numeric_dtype(df[c])]


def default_plot_config(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = pick_numeric_columns(df)
    if not numeric_cols:
        numeric_cols = ["__index__"]
    
    x_axis = numeric_cols[0]
    y_axis = numeric_cols[min(1, len(numeric_cols) - 1)]
    
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": None,
        "color_by": None,
        "size_by": "fixed",
        "grid": True,
        "marker_size": 10,
        "color_scheme": "Viridis",
        "x_scale": "linear", "x_min": None, "x_max": None,
        "y_scale": "linear", "y_min": None, "y_max": None,
        "z_scale": "linear", "z_min": None, "z_max": None,
        "text_size": 10,
        "axis_labels": {"x": x_axis, "y": y_axis, "z": ""},
    }

def sanitize_plot_config(config: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    base = default_plot_config(df)
    if not config:
        return base

    numeric_cols = pick_numeric_columns(df)
    if not numeric_cols:
        numeric_cols = ["__index__"]
    categorical_cols = pick_categorical_columns(df)

    sanitized: Dict[str, Any] = {}

    def pick_numeric(value: Optional[str], default: str) -> str:
        return value if value in numeric_cols else default

    sanitized["x_axis"] = pick_numeric(config.get("x_axis"), base["x_axis"])
    sanitized["y_axis"] = pick_numeric(config.get("y_axis"), base["y_axis"])
    
    z_candidate = config.get("z_axis")
    sanitized["z_axis"] = z_candidate if z_candidate in numeric_cols and z_candidate != "None" else None
    
    color_options = set(numeric_cols) | set(categorical_cols)
    color_choice = config.get("color_by")
    sanitized["color_by"] = color_choice if color_choice in color_options and color_choice != "None" else None
    
    size_options = set(numeric_cols)
    size_choice = config.get("size_by")
    sanitized["size_by"] = size_choice if size_choice in size_options and size_choice != "fixed" else None

    sanitized["grid"] = bool(config.get("grid", base["grid"]))
    sanitized["color_scheme"] = config.get("color_scheme", base["color_scheme"])
    
    try:
        marker_size = int(config.get("marker_size") or config.get("text_size") or base["marker_size"])
    except (TypeError, ValueError):
        marker_size = base["marker_size"]
    sanitized["marker_size"] = max(1, min(48, marker_size))

    for axis in ["x", "y", "z"]:
        scale_key = f"{axis}_scale"
        sanitized[scale_key] = "log" if config.get(scale_key) == "log" else "linear"
        for bound in ["min", "max"]:
            bound_key = f"{axis}_{bound}"
            try:
                sanitized[bound_key] = float(config[bound_key]) if config.get(bound_key) is not None else None
            except (ValueError, TypeError):
                sanitized[bound_key] = None

    sanitized["text_size"] = sanitized["marker_size"]
    sanitized["axis_labels"] = {
        "x": sanitized["x_axis"],
        "y": sanitized["y_axis"],
        "z": sanitized.get("z_axis") or "",
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
        numeric_cols = ["__index__"]
    categorical_cols = pick_categorical_columns(df)

    prefixed = lambda name: f"{key_prefix}_{name}"

    def ensure_state(key: str, value: Any) -> None:
        st.session_state.setdefault(prefixed(key), value)

    # Initialize state for all settings to ensure widgets have controlled values
    for key, value in defaults.items():
        ensure_state(key, value)

    config = {}

    with st.container():
        st.markdown(
            "<div style='padding:10px 12px;border:1px solid rgba(15,23,42,0.15);border-radius:10px;background:rgba(241,243,249,0.6);margin-bottom:8px;'>"
            "<strong>Plot Settings</strong></div>",
            unsafe_allow_html=True,
        )

        with st.expander("Data Settings", expanded=True):
            axis_cols = st.columns(2)
            with axis_cols[0]:
                config["x_axis"] = st.selectbox(
                    "X-axis", numeric_cols,
                    key=prefixed("x_axis")
                )
            with axis_cols[1]:
                config["y_axis"] = st.selectbox(
                    "Y-axis", numeric_cols,
                    key=prefixed("y_axis")
                )

            z_options = ["None"] + numeric_cols
            z_selection = st.selectbox(
                "Z-axis", z_options,
                key=prefixed("z_axis")
            )
            config["z_axis"] = None if z_selection == "None" else z_selection
            
            st.markdown("---")
            axes_to_configure = [("X", "x", config["x_axis"]), ("Y", "y", config["y_axis"])]
            if config["z_axis"]:
                axes_to_configure.append(("Z", "z", config["z_axis"]))

            for axis_name, axis_key, col_name in axes_to_configure:
                st.markdown(f"**{axis_name}-Axis Controls**")
                
                default_min, default_max = (None, None)
                if col_name and col_name in df.columns:
                    col_data = pd.to_numeric(df[col_name], errors="coerce").dropna()
                    if not col_data.empty:
                        default_min = float(col_data.min())
                        default_max = float(col_data.max())

                min_state_key = prefixed(f"{axis_key}_min")
                if st.session_state.get(min_state_key) is None:
                    st.session_state[min_state_key] = default_min
                
                max_state_key = prefixed(f"{axis_key}_max")
                if st.session_state.get(max_state_key) is None:
                    st.session_state[max_state_key] = default_max

                c1, c2, c3 = st.columns(3)
                

                if st.session_state.get(min_state_key) == 0.0:
                    st.session_state[min_state_key] = default_min
                if st.session_state.get(max_state_key) == 0.0:
                    st.session_state[max_state_key] = default_max
                
                if st.session_state[max_state_key] < st.session_state[min_state_key]:
                    st.warning(f"Min value for {axis_name} cannot be greater than Max value. Reverting to previous values.")

                with c1:
                    config[f"{axis_key}_scale"] = st.radio(f"Scale", ["linear", "log"], key=prefixed(f"{axis_key}_scale"), horizontal=True, label_visibility="collapsed")
                with c2:
                    config[f"{axis_key}_min"] = st.number_input(f"Min",
                                                                # min_value=5,
                                                                key=min_state_key, 
                                                                format="%g")
                with c3:
                    config[f"{axis_key}_max"] = st.number_input(f"Max", 
                                                                key=max_state_key, 
                                                                format="%g")

        with st.expander("Style Settings", expanded=False):
            color_options = ["None"] + numeric_cols + categorical_cols
            color_selection = st.selectbox(
                "Color by", color_options,
                key=prefixed("color_by")
            )
            config["color_by"] = None if color_selection == "None" else color_selection

            if config["color_by"]:
                config["color_scheme"] = st.selectbox(
                    "Color Scheme",
                    ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Rainbow", "Jet"],
                    key=prefixed("color_scheme")
                )
            else:
                config["color_scheme"] = None
            
            size_options = ["fixed"] + numeric_cols
            size_selection = st.selectbox(
                "Size by", size_options,
                key=prefixed("size_by")
            )
            config["size_by"] = size_selection

            config["grid"] = st.toggle("Show grid", key=prefixed("grid"))
            if config["size_by"] == "fixed":
                config["marker_size"] = st.slider("Marker Size", min_value=1, max_value=30, key=prefixed("marker_size"))
            else:
                config["marker_size"] = defaults["marker_size"]

    # Fill in missing keys for conditional UI
    config.setdefault("z_scale", "linear")
    config.setdefault("z_min", None)
    config.setdefault("z_max", None)
    config.setdefault("color_scheme", "Viridis")

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

    def get_image_data(fig_obj, fmt, scale):
        try:
            return fig_obj.to_image(format=fmt, scale=scale)
        except Exception as exc:
            st.error(f"Error generating {fmt.upper()} image: {exc}")
            return None

    if downloads:
        cols = st.columns(len(downloads))
        for idx, item in enumerate(downloads):
            with cols[idx]:
                # Pass a callable to data to defer image generation
                st.download_button(
                    label=item.get("label", "Download"),
                    data=lambda: get_image_data(fig, item["key"], 2), # Assuming scale 2 for downloads
                    file_name=item.get("filename", "plot"),
                    mime=item.get("mime", "application/octet-stream"),
                    key=f"plot-download-{item.get('key', idx)}",
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



    if selected_id is None and not df.empty:
        selected_id = df["selection_id"].iloc[0]
    st.session_state["selected_id"] = selected_id
    return selected_id

def _label_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return pd.Series(df["label"].values, index=df["selection_id"]).to_dict()
