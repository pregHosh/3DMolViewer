from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ase import Atoms

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

SNAPSHOT_QUALITY_OPTIONS = [
    ("Standard (1x)", 1),
    ("High (2x)", 2),
    ("Ultra (4x)", 4),
]


def sidebar_controls(
    df: pd.DataFrame, *, enable_scatter: bool, show_scatter_controls: bool = True
) -> Dict[str, Any]:
    st.sidebar.header("Data")
    x_axis = y_axis = z_axis = color_by = size_by = None

    if enable_scatter:
        if show_scatter_controls:
            scatter_mode = st.sidebar.radio("Scatter dimensionality", ["2D", "3D"], index=0)
            numeric_cols = pick_numeric_columns(df)
            categorical_cols = pick_categorical_columns(df)
            if not numeric_cols:
                numeric_cols = ["__index"]
            x_axis = st.sidebar.selectbox("X axis", numeric_cols, index=0)
            y_axis = st.sidebar.selectbox(
                "Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1)
            )
            if scatter_mode == "3D":
                z_axis = st.sidebar.selectbox(
                    "Z axis", numeric_cols, index=min(2, len(numeric_cols) - 1)
                )
            color_choice = st.sidebar.selectbox(
                "Color by",
                ["None", *numeric_cols, *categorical_cols],
                index=0,
            )
            color_by = None if color_choice == "None" else color_choice
            size_choice = st.sidebar.selectbox("Size by", ["Uniform", *numeric_cols], index=0)
            size_by = None if size_choice == "Uniform" else size_choice
        else:
            numeric_cols = pick_numeric_columns(df)
            if not numeric_cols:
                numeric_cols = ["__index"]
            x_axis = numeric_cols[0]
            y_axis = numeric_cols[min(1, len(numeric_cols) - 1)]
            if len(numeric_cols) > 2:
                z_axis = numeric_cols[min(2, len(numeric_cols) - 1)]
            st.sidebar.caption("Configure plotting in the main panel below the chart.")
    elif show_scatter_controls:
        st.sidebar.info("Add numeric properties to enable scatter plotting.")

    with st.sidebar.expander("3D Viewer", expanded=True):
        viewer_engine = st.selectbox("Viewer Engine", ["NGL", "3Dmol"], index=0)

        atom_label = st.selectbox(
            "Atom label",
            ["None", "Symbol", "Atomic number", "Atom index"],
            index=0,
            help="Choose a single annotation to display in the 3D viewer",
        )

        if viewer_engine == "NGL":
            mode_label_to_key = {
                "Rotate / navigate": "rotate",
                "Select atom (inspect properties)": "select",
                "Measurement (auto)": "measurement",
            }
            viewer_mode_label = st.selectbox(
                "Mouse mode",
                list(mode_label_to_key.keys()),
                index=0,
                help="Choose how mouse clicks interact with the 3D viewer.",
            )
            viewer_mode = mode_label_to_key[viewer_mode_label]
            sphere_radius = st.slider("Atom radius", min_value=0.1, max_value=0.8, value=0.3, step=0.05)
            bond_radius = st.slider("Bond radius", min_value=0.05, max_value=0.4, value=0.12, step=0.01)
            representation_style = st.selectbox(
                "Rendering style",
                [
                    "Ball + Stick",
                    "Licorice",
                    "Spacefilling",
                    "Hyperball",
                    "Line",
                    "Point Cloud",
                    "Surface",
                ],
                index=0,
                help="Choose how the molecule is drawn in the 3D viewer.",
            )
            snapshot_transparent = st.checkbox(
                "Transparent snapshot background",
                value=False,
                help="When enabled, PNG exports use a transparent backdrop instead of the theme color.",
            )
            snapshot_quality_labels = [label for label, _ in SNAPSHOT_QUALITY_OPTIONS]
            snapshot_quality = st.selectbox(
                "Snapshot quality",
                snapshot_quality_labels,
                index=0,
                help="Choose a resolution multiplier for PNG exports.",
            )
            threedmol_style = None
            threedmol_atom_radius = None
            threedmol_bond_radius = None
        else:  # 3Dmol
            viewer_mode = "rotate"
            sphere_radius = None
            bond_radius = None
            representation_style = None
            snapshot_transparent = None
            snapshot_quality = None
            threedmol_style = st.selectbox(
                "Style",
                ["stick", "sphere", "line", "cross", "Ball and Stick"],
                index=4,
            )
            if threedmol_style == "Ball and Stick":
                threedmol_atom_radius = st.slider("Atom radius", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
                threedmol_bond_radius = st.slider("Bond radius", min_value=0.05, max_value=0.5, value=0.1, step=0.01)
            else:
                threedmol_atom_radius = None
                threedmol_bond_radius = None

        show_hydrogens = st.checkbox(
            "Show hydrogen atoms",
            value=True,
            help="Uncheck to hide hydrogens across the viewer, measurements, and metadata tables.",
        )

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "color_by": color_by,
        "size_by": size_by,
        "viewer_engine": viewer_engine,
        "sphere_radius": sphere_radius,
        "bond_radius": bond_radius,
        "representation_style": representation_style,
        "threedmol_style": threedmol_style,
        "threedmol_atom_radius": threedmol_atom_radius,
        "threedmol_bond_radius": threedmol_bond_radius,
        "atom_label": None if atom_label == "None" else atom_label,
        "viewer_mode": viewer_mode,
        "show_hydrogens": show_hydrogens,
        "snapshot_transparent": snapshot_transparent,
        "snapshot_quality": snapshot_quality,
    }


def _format_atom_option(idx: int, symbols: Iterable[str], numbers: Iterable[int]) -> str:
    return f"{idx + 1} - {symbols[idx]} (Z={int(numbers[idx])})"


def _compute_distance(coords: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(coords[a] - coords[b]))


def _compute_angle(coords: np.ndarray, a: int, b: int, c: int) -> float:
    vec1 = coords[a] - coords[b]
    vec2 = coords[c] - coords[b]
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if not norm1 or not norm2:
        return float("nan")
    cos_theta = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _compute_dihedral(coords: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    b0 = coords[b] - coords[a]
    b1 = coords[c] - coords[b]
    b2 = coords[d] - coords[c]
    b1_norm = np.linalg.norm(b1)
    if not b1_norm:
        return float("nan")
    b1_unit = b1 / b1_norm
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if not n1_norm or not n2_norm:
        return float("nan")
    n1_unit = n1 / n1_norm
    n2_unit = n2 / n2_norm
    m1 = np.cross(n1_unit, b1_unit)
    x = np.dot(n1_unit, n2_unit)
    y = np.dot(m1, n2_unit)
    return float(np.degrees(np.arctan2(y, x)))


def show_measurement_panel(atoms: Atoms, mode: str, *, key_prefix: str) -> None:
    num_atoms = len(atoms)
    if num_atoms == 0:
        st.info("No atoms available for measurements.")
        return
    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()
    coords = atoms.get_positions()
    masses = atoms.get_masses()
    options = list(range(num_atoms))
    fmt = lambda idx: _format_atom_option(idx, symbols, numbers)
    st.markdown("**Measurement tools**")
    if mode == "select":
        idx = st.selectbox("Atom", options, format_func=fmt, key=f"{key_prefix}_sel")
        st.markdown(f"- Symbol: `{symbols[idx]}`  Z={int(numbers[idx])}")
        st.markdown("- Coordinates (Angstrom): ({:.4f}, {:.4f}, {:.4f})".format(*coords[idx]))
        st.markdown(f"- Mass (amu): {float(masses[idx]):.4f}")
        return
    if mode == "measurement":
        st.caption(
            "Select 2–4 atoms in the 3D view to measure distances, angles, or dihedrals automatically. "
            "Use the pickers below for manual calculations."
        )
        if num_atoms >= 2:
            col1, col2 = st.columns(2)
            a = col1.selectbox("Atom 1", options, format_func=fmt, key=f"{key_prefix}_dist_a")
            b = col2.selectbox(
                "Atom 2",
                options,
                format_func=fmt,
                key=f"{key_prefix}_dist_b",
                index=1 if num_atoms > 1 else 0,
            )
            dist = _compute_distance(coords, a, b)
            st.markdown(f"**Distance:** {dist:.4f} Angstrom")
        else:
            st.info("At least two atoms are required for distance measurements.")

        if num_atoms >= 3:
            col1, col2, col3 = st.columns(3)
            a = col1.selectbox("Atom 1", options, format_func=fmt, key=f"{key_prefix}_angle_a")
            b = col2.selectbox(
                "Atom 2 (vertex)",
                options,
                format_func=fmt,
                key=f"{key_prefix}_angle_b",
                index=1 if num_atoms > 1 else 0,
            )
            c = col3.selectbox(
                "Atom 3",
                options,
                format_func=fmt,
                key=f"{key_prefix}_angle_c",
                index=2 if num_atoms > 2 else 0,
            )
            angle = _compute_angle(coords, a, b, c)
            st.markdown(f"**Angle (deg):** {angle:.4f}")
            st.caption(
                f"d(1-2) = {_compute_distance(coords, a, b):.4f} Angstrom · "
                f"d(2-3) = {_compute_distance(coords, b, c):.4f} Angstrom"
            )
        else:
            st.info("At least three atoms are required for angle measurements.")

        if num_atoms >= 4:
            cols = st.columns(4)
            a = cols[0].selectbox("Atom 1", options, format_func=fmt, key=f"{key_prefix}_dih_a")
            b = cols[1].selectbox(
                "Atom 2",
                options,
                format_func=fmt,
                key=f"{key_prefix}_dih_b",
                index=1 if num_atoms > 1 else 0,
            )
            c = cols[2].selectbox(
                "Atom 3",
                options,
                format_func=fmt,
                key=f"{key_prefix}_dih_c",
                index=2 if num_atoms > 2 else 0,
            )
            d = cols[3].selectbox(
                "Atom 4",
                options,
                format_func=fmt,
                key=f"{key_prefix}_dih_d",
                index=3 if num_atoms > 3 else 0,
            )
            dihedral = _compute_dihedral(coords, a, b, c, d)
            st.markdown(f"**Dihedral (deg):** {dihedral:.4f}")
        else:
            st.info("At least four atoms are required for dihedral measurements.")
        return

    st.info("Switch mouse mode to enable backend measurements.")


def summarize_atoms(atoms: Optional[Atoms]) -> Dict[str, Any]:
    if atoms is None or len(atoms) == 0:
        return {
            "formula": "—",
            "num_atoms": 0,
            "mass_amu": None,
            "center_x": None,
            "center_y": None,
            "center_z": None,
            "cell_volume": None,
        }
    cell_volume = None
    if atoms.cell is not None and atoms.cell.volume > 0:
        cell_volume = float(atoms.cell.volume)
    com = atoms.get_center_of_mass()
    return {
        "formula": atoms.get_chemical_formula(),
        "num_atoms": int(len(atoms)),
        "mass_amu": float(atoms.get_masses().sum()),
        "center_x": float(com[0]),
        "center_y": float(com[1]),
        "center_z": float(com[2]),
        "cell_volume": cell_volume,
    }


def _render_dataframe(df: pd.DataFrame, theme: ThemeConfig) -> None:
    if theme.background.lower() == "#fafbff":
        styled = (
            df.style.set_properties(
                **{
                    "background-color": "#FFFFFF",
                    "color": "#0F172A",
                    "border-color": "rgba(15, 23, 42, 0.12)",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", "#0F172A"),
                            ("border-color", "rgba(15, 23, 42, 0.12)"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [("border-color", "rgba(15, 23, 42, 0.12)")],
                    },
                ]
            )
        )
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def show_details(record: pd.Series, atoms: Optional[Atoms], theme: ThemeConfig) -> None:
    st.markdown(
        f"<div class='molviewer-card'><strong>Selected structure:</strong> `{record.label}`</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='molviewer-card'><strong>Basic information</strong></div>", unsafe_allow_html=True)
    _render_dataframe(pd.DataFrame([summarize_atoms(atoms)]), theme)
    metadata = {k: v for k, v in record.items() if k not in BASE_COLUMNS}
    if metadata:
        st.markdown("<div class='molviewer-card'><strong>Metadata</strong></div>", unsafe_allow_html=True)
        _render_dataframe(pd.DataFrame([metadata]), theme)


def navigation_controls(df: pd.DataFrame, selected_id: Optional[str]) -> Optional[str]:
    options = df["selection_id"].tolist()
    if not options:
        return selected_id
    if selected_id not in options:
        selected_id = options[0]
    idx = options.index(selected_id)
    prev_col, center_col, next_col = st.columns([1, 3, 1])
    if prev_col.button("◀ Previous", use_container_width=True):
        idx = (idx - 1) % len(options)
    if next_col.button("Next ▶", use_container_width=True):
        idx = (idx + 1) % len(options)
    new_id = options[idx]
    with center_col:
        st.markdown(
            f"**{df.loc[df['selection_id'] == new_id, 'label'].iloc[0]} ({idx + 1}/{len(options)})**"
        )
    st.session_state["selected_id"] = new_id
    return new_id

# Import from plot_utils for pick_numeric_columns and pick_categorical_columns
from src.plot_utils import pick_numeric_columns, pick_categorical_columns
