# 🧬 Lite Molecular Viz

Lite Molecular Viz is a standalone Streamlit application for interactively browsing and analyzing molecular structures and their associated properties. It's designed as a lightweight, easy-to-use tool for researchers and developers in computational chemistry and drug discovery.


Intended to be used with [![Code](https://img.shields.io/badge/MolCraftDiffusion-GitHub-red)](https://github.com/pregHosh/MolCraftDiffusion)

The paper: [![arXiv](https://img.shields.io/badge/PDF-arXiv-blue)](https://chemrxiv.org/engage/chemrxiv/article-details/6909e50fef936fb4a23df237)


## Features

-   **Multiple Data Sources**: Load your data with flexibility.
    -   **ASE Database**: For large datasets, use a highly efficient `.db` file.
    -   **CSV Files**: Upload directly, or provide a local file path.
    -   **XYZ Directory**: Point to a directory of `.xyz` or `.pdb` files.
-   **Dynamic Visualization Modes**:
    -   **Plotting Interface**: If your data includes numeric properties, explore it with interactive plots (scatter, line, histogram, box, parity). Selecting data points on a plot instantly displays the corresponding 3D molecule.
    -   **Molecule Gallery**: If your dataset only contains molecule identifiers, the app provides a clean, grid-based gallery of all 3D structures.
-   **Interactive 3D Viewer**:
    -   Powered by `3Dmol.js` for high-quality rendering.
    -   View molecules in stick, sphere, or line styles.
    -   Zoom, pan, and rotate structures.
-   **Extensible Plugin System**:
    -   Enhance the app's functionality with optional plugins for on-demand data analysis.
    -   **Dataset Statistics Plugin**: Compute and visualize summary statistics, element distributions, and property histograms for your dataset.

## Quick Start

1.  **Set up the environment**:
    ```bash
    # Clone the repository (if you haven't already)
    # git clone ...
    # cd 3DMolViewer

    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the app**:
    ```bash
    streamlit run app.py
    ```

4.  Once the app is open in your browser, use the sidebar to select your data source and start exploring.

## Data Sources

The application supports several methods for loading molecular data.

### ASE Database (Recommended for large datasets)

For the best performance with large datasets, consolidate your molecules and properties into a single ASE database (`.db`) file.

A utility script is provided to make this easy. It takes a directory of molecule files (`.xyz`, `.pdb`, etc.) and a CSV file with properties, and merges them into a `.db` file. The CSV must contain a column that matches the molecule filenames.

**Usage:**
```bash
python scripts/create_database.py [path/to/your.csv] [path/to/molecule/dir] [output_database.db]
```

**Example:**
```bash
# Create a database from the example data
python scripts/create_database.py examples/sampled_molecules_descriptors.csv examples/ molecules.db --overwrite
```
After creating the database, select "ASE Database" in the app's data source options and provide the path to your `.db` file.

### CSV File

You can load properties from a CSV file by either uploading it directly in the app or by providing a local file path. If the CSV contains a column with molecule names, you can link it to a separate **Molecule Data Source** (like an XYZ directory or an ASE DB) in the sidebar to visualize the structures.

### XYZ Directory

For smaller datasets, you can simply point the application to a directory containing your molecule files (e.g., `.xyz`, `.pdb`). The app will read the files from the directory.

## Project Structure

```
.
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── core/                 # Core application logic
│   ├── molecule_viz.py   # 3Dmol.js visualization
│   ├── plotting.py       # Plotly interactive plots
│   └── utils.py          # Data processing and configuration
├── plugins/              # Extensible plugin modules
│   ├── base_plugin.py
│   └── statistics_plugin.py
├── scripts/              # Utility scripts
│   └── create_database.py
├── examples/             # Example molecule files and data
└── tests/                # Application tests
```

## Contributing

Issues and pull requests are welcome. Please ensure that new features are documented and that the code follows the existing style.

## License

This project is released under the MIT License.