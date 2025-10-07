# 3DMolViewer Toolkit

3DMolViewer is a standalone Streamlit application for browsing molecular structures with an NGL.js viewport. It is designed as a lightweight companion for model-development workflows, so you can inspect generated geometries without firing up the full MolecularDiffusion stack.

## Highlights
- Interactive scatter plots (Plotly) for exploring scalar properties alongside structures.
- NGL-powered molecule window with snapshot export, keyboard navigation, and multiple rendering modes.
- Bundled sample XYZ files for immediate smoke testing (`test_xyz/`).
- Graceful fallbacks when optional extras (streamlit hotkeys, Plotly event callbacks) are missing.

## Quick Start
1. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r 3DMolViewer/requirements.txt
   ```
2. Launch Streamlit:
   ```bash
   streamlit run 3DMolViewer/app.py
   ```
3. Choose an XYZ directory or ASE database from the sidebar to begin exploring molecules.

## Project Structure
```
3DMolViewer/
├── app.py                # Streamlit app entry point
├── requirements.txt      # Runtime dependencies
├── components/
│   └── key_listener/     # Custom Streamlit component for arrow-key navigation
├── test_xyz/             # Sample XYZ files for demos/tests
└── __init__.py
```

## Contributing
Issues and pull requests are welcome. Keep commits focused, add docstrings for new configuration options, and consider adding a test XYZ file when introducing new rendering behaviors.

## License
Released under the MIT License.
