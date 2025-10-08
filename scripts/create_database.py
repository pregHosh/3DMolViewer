from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from ase.data import chemical_symbols
from ase.db import connect
from ase.io import read as ase_read
from tqdm import tqdm


def create_database(
    csv_path: str,
    xyz_dir: str,
    db_path: str,
    *, overwrite: bool
) -> None:
    """
    Creates an ASE database by iterating through a CSV file, loading the
    referenced molecule files, and saving their structures and properties.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"\nError: CSV file not found at '{csv_path}'", file=sys.stderr)
        sys.exit(1)

    xyz_root = Path(xyz_dir)
    if not xyz_root.is_dir():
        print(f"\nError: XYZ directory not found at '{xyz_dir}'", file=sys.stderr)
        sys.exit(1)

    db_file = Path(db_path)
    if db_file.exists():
        if overwrite:
            print(f"Database file '{db_path}' already exists. Deleting it as requested.")
            db_file.unlink()
        else:
            print(
                f"\nError: Database file '{db_path}' already exists.", file=sys.stderr
            )
            print("Use the --overwrite flag to replace it.", file=sys.stderr)
            sys.exit(1)

    print(f"Reading source CSV from '{csv_path}'...")
    try:
        df = pd.read_csv(csv_file)
        if "filename" not in df.columns:
            print("\nError: CSV must contain a 'filename' column.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"\nError reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Build the full list of reserved keys, including all chemical symbols
    reserved_keys = {
        "natoms",
        "energy",
        "forces",
        "stress",
        "magmom",
        "charge",
        "filename",
    }
    for symbol in chemical_symbols:
        reserved_keys.add(symbol)

    print(f"Creating new database at '{db_path}'...")
    structures_written = 0
    with connect(db_path) as db:
        for _, record in tqdm(
            df.iterrows(), total=len(df), desc="Processing rows", ncols=80
        ):
            filename = record["filename"] + ".xyz"
            file_path = xyz_root / filename

            if not file_path.exists():
                print(f"Warning: Could not find file '{filename}' in '{xyz_dir}'. Skipping.")
                continue

            try:
                atoms = ase_read(file_path)
            except Exception as e:
                print(
                    f"Warning: Failed to read atoms from '{file_path}': {e}. Skipping."
                )
                continue

            # Sanitize keys and filter out reserved/redundant ones
            props_to_save = {}
            raw_props = record.to_dict()

            for key, value in raw_props.items():
                if key in reserved_keys:
                    continue  # Skip reserved keys

                # Sanitize the key for ASE DB compatibility
                sanitized_key = re.sub(r'[^a-zA-Z0-9_]+', '_', key)
                props_to_save[sanitized_key] = value

            db.write(atoms, key_value_pairs=props_to_save)
            structures_written += 1

    print("\nDatabase creation complete.")
    print(f"Total structures written: {structures_written} (out of {len(df)} rows in CSV)")
    if structures_written < len(df):
        print("Some rows were skipped because the corresponding files were not found.")
    print(
        f"You can now use '{db_path}' as the 'ASE Database' source in the application."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a directory of XYZ files and a properties CSV into an ASE database.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "csv_path", type=str, help="Path to the CSV file containing filenames and properties."
    )
    parser.add_argument(
        "xyz_dir",
        type=str,
        help="Path to the directory where the .xyz/.pdb files are located.",
    )
    parser.add_argument(
        "db_path",
        type=str,
        help="Path for the output ASE database file (e.g., 'data/molecules.db').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite the database file if it already exists.",
    )
    args = parser.parse_args()

    try:
        create_database(
            args.csv_path, args.xyz_dir, args.db_path, overwrite=args.overwrite
        )
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()