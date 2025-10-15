"""Simplified molecule visualization utilities."""

import streamlit as st
import streamlit.components.v1 as components
from ase.db import connect
from ase.io import read
from ase import Atoms
import os

def _atoms_to_xyz(atoms: Atoms) -> str:
    xyz_lines = [str(len(atoms)), ""]
    for atom in atoms:
        xyz_lines.append(f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}")
    return "\n".join(xyz_lines)

def _create_3d_viewer(xyz_data: str) -> None:
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <style>#viewer {{ width: 100%; height: 350px; border: 1px solid #ddd; position: relative; }}</style>
    </head>
    <body>
        <div id="viewer"></div>
        <script>
            const viewer = $3Dmol.createViewer('viewer');
            viewer.addModel(`{xyz_data}`, 'xyz');
            viewer.setStyle({{}}, {{ "stick": {{"radius": 0.1}}, "sphere": {{"scale": 0.3}} }});
            viewer.zoomTo();
            viewer.render();
        </script>
    </body>
    </html>
    """
    components.html(html_content, height=400)

def display_molecule_from_file(data_source: str, molecule_name: str) -> None:
    """Display molecule visualization by reading from a file source."""
    try:
        atoms = None
        if os.path.isdir(data_source):
            file_path = os.path.join(data_source, molecule_name)
            if not os.path.exists(file_path) and not molecule_name.endswith('.xyz'):
                file_path = f"{file_path}.xyz"
            if not os.path.exists(file_path):
                st.error(f"Molecule file not found: {file_path}")
                return
            atoms = read(file_path)
        elif os.path.isfile(data_source) and data_source.endswith('.db'):
            db = connect(data_source)
            row = db.get(name=molecule_name)
            if row:
                atoms = row.toatoms()
        else:
            st.error(f"Invalid source for visualization: {data_source}")
            return

        if atoms:
            xyz_data = _atoms_to_xyz(atoms)
            _create_3d_viewer(xyz_data)
        else:
            st.error(f"Could not load molecule '{molecule_name}' from source.")
    except Exception as e:
        st.error(f"Error visualizing molecule from source: {e}")

def display_molecule_from_data(atoms: Atoms) -> None:
    """Display molecule visualization directly from an Atoms object."""
    try:
        xyz_data = _atoms_to_xyz(atoms)
        _create_3d_viewer(xyz_data)
    except Exception as e:
        st.error(f"Error visualizing molecule from data: {e}")