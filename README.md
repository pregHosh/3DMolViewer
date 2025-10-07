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

## Data Preparation: Using an ASE Database

For large datasets (thousands of molecules), the application will perform much better if the molecule data is consolidated into a single file. The "XYZ Directory" source is convenient but can be slow as it requires scanning many individual files.

This project includes a script to convert a directory of `.xyz`/`.pdb` files and an associated properties CSV into a single, highly-efficient ASE database (`.db`) file.

**How to Create the Database:**

Use the `scripts/create_database.py` script. It requires a CSV file containing a `filename` column that points to the molecule files.

**Usage:**

```bash
python scripts/create_database.py [path/to/your.csv] [path/to/xyz/dir] [output_database.db]
```

**Example:**

To convert the sample data included in this repository into a database named `molecules.db`:

```bash
python scripts/create_database.py test_xyz/xyz_properties.csv test_xyz/ molecules.db --overwrite
```

Once the script is finished, select the "ASE Database" option in the application sidebar and provide the path to your newly created `.db` file.

## Project Structure
```
3DMolViewer/
├── app.py                # Streamlit app entry point
├── requirements.txt      # Runtime dependencies
├── scripts/
│   └── create_database.py # Utility to convert XYZ files to an ASE database
├── components/
│   └── key_listener/     # Custom Streamlit component for arrow-key navigation
├── test_xyz/             # Sample XYZ files for demos/tests
└── __init__.py
```

## Contributing
Issues and pull requests are welcome. Keep commits focused, add docstrings for new configuration options, and consider adding a test XYZ file when introducing new rendering behaviors.

## License
Released under the MIT License.
