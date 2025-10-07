from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from time import perf_counter

import pandas as pd
import numpy as np
import streamlit as st

from ase import Atoms
from ase.db import connect
from ase.io import read as ase_read


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


@st.cache_data(show_spinner=False)
def load_xyz_metadata(dir_path: str, csv_path: Optional[str]) -> pd.DataFrame:
    directory = Path(dir_path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    timings: List[Dict[str, Any]] = []

    def add_timing(phase: str, start_time: float, **meta: Any) -> None:
        timings.append({"phase": phase, "seconds": perf_counter() - start_time, **meta})

    scan_start = perf_counter()
    xyz_files = list(directory.glob("*.xyz"))
    pdb_files = list(directory.glob("*.pdb"))
    file_map: Dict[str, Path] = {}
    for path in sorted(xyz_files + pdb_files):
        key = path.stem
        if key in file_map:
            if file_map[key].suffix.lower() == ".xyz":
                continue
        file_map[key] = path
    xyz_files = sorted(file_map.values())
    total_geometries = len(xyz_files)
    filtered_xyz_missing_props = 0
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {directory}")
    add_timing("scan_geometries", scan_start, geometries=total_geometries)

    properties: Dict[str, Dict[str, Any]] = {}
    csv_only_payload: Dict[str, Dict[str, Any]] = {}
    csv_ignored = False
    csv_metadata_path: Optional[str] = None
    csv_only_names: list[str] = []
    matched_names: set[str] = set()
    xyz_map = {p.name: p for p in xyz_files}
    xyz_basename_map = {p.stem: p for p in xyz_files}

    if csv_path:
        csv_file = Path(csv_path).expanduser().resolve()
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        csv_metadata_path = str(csv_file)
        csv_records: Dict[str, Dict[str, Any]] = {}

        filename_to_name = {name: name for name in xyz_map.keys()}
        basename_to_name = {stem: path.name for stem, path in xyz_basename_map.items()}

        ROW_THRESHOLD = 5000
        preview_start = perf_counter()
        try:
            preview_df = pd.read_csv(csv_file, nrows=ROW_THRESHOLD + 1)
        except Exception:
            preview_df = None
        add_timing(
            "preview_csv",
            preview_start,
            rows=int(len(preview_df))) if preview_df is not None else 0

        small_dataset = preview_df is not None and len(preview_df) <= ROW_THRESHOLD

        if small_dataset:
            df_props = preview_df.copy() if preview_df is not None else pd.DataFrame()
            if "filename" not in df_props.columns:
                raise ValueError("CSV file must contain a 'filename' column")
            df_props = df_props.copy()
            df_props["filename"] = df_props["filename"].astype(str)
            df_props["_clean_name"] = df_props["filename"].str.replace("\.xyz$|\.pdb$", "", regex=True)
            value_columns = [col for col in df_props.columns if col != "filename"]
            if value_columns:
                df_props = df_props.dropna(
                    subset=[col for col in df_props.columns if col not in {"filename", "_clean_name"}],
                    how="all",
                )
            match_start = perf_counter()
            for _, row in df_props.iterrows():
                fname = str(row["filename"])
                clean = str(row.get("_clean_name") or Path(fname).stem)
                if fname in filename_to_name:
                    target_name = filename_to_name[fname]
                elif clean in basename_to_name:
                    target_name = basename_to_name[clean]
                else:
                    target_name = clean
                payload: Dict[str, Any] = {}
                for col in value_columns:
                    if col in {"_clean_name"}:
                        continue
                    value = row[col]
                    if pd.isna(value):
                        continue
                    payload[col] = value
                csv_records[target_name] = payload
            add_timing(
                "match_properties_small",
                match_start,
                matched=int(len(csv_records)),
                rows=int(len(df_props)),
            )
        else:
            chunk_start = perf_counter()
            chunk_iter = pd.read_csv(csv_file, chunksize=50000)
            value_columns: List[str] = []
            first_chunk = True
            chunk_count = 0
            total_rows = 0
            for chunk in chunk_iter:
                if "filename" not in chunk.columns:
                    raise ValueError("CSV file must contain a 'filename' column")
                chunk = chunk.copy()
                chunk["filename"] = chunk["filename"].astype(str)
                chunk["_clean_name"] = chunk["filename"].str.replace("\.xyz$|\.pdb$", "", regex=True)
                if first_chunk:
                    value_columns = [col for col in chunk.columns if col not in {"filename"}]
                    first_chunk = False
                data_cols = [col for col in chunk.columns if col not in {"filename", "_clean_name"}]
                if data_cols:
                    chunk = chunk.dropna(subset=data_cols, how="all")
                if chunk.empty:
                    continue
                match_full = chunk["filename"].map(filename_to_name)
                match_clean = chunk["_clean_name"].map(basename_to_name)
                chunk["_match_name"] = match_full.fillna(match_clean).fillna(chunk["_clean_name"])
                payload_cols = [col for col in data_cols if col != "_match_name"]
                if payload_cols:
                    chunk[payload_cols] = chunk[payload_cols].astype(object)
                    chunk[payload_cols] = chunk[payload_cols].where(~chunk[payload_cols].isna(), None)
                    chunk_records = chunk.set_index("_match_name")[payload_cols].to_dict(orient="index")
                else:
                    chunk_records = {name: {} for name in chunk["_match_name"].tolist()}
                csv_records.update(chunk_records)
                chunk_count += 1
                total_rows += len(chunk)
            add_timing(
                "match_properties_chunk",
                chunk_start,
                matched=int(len(csv_records)),
                chunks=chunk_count,
                rows=total_rows,
            )

        csv_names = set(csv_records.keys())
        xyz_names = set(list(xyz_map.keys()) + list(xyz_basename_map.keys()))
        matched_names = {name for name in csv_names if name in xyz_map or name in xyz_basename_map}
        csv_only_names = sorted(name for name in csv_names if name not in matched_names)
        if matched_names:
            properties = {}
            filtered_files: list[Path] = []
            for path in xyz_files:
                key = path.name
                base = path.stem
                if key in csv_records:
                    properties[key] = csv_records[key]
                    filtered_files.append(path)
                elif base in csv_records:
                    properties[key] = csv_records[base]
                    filtered_files.append(path)
            xyz_files = filtered_files
            filtered_xyz_missing_props = total_geometries - len(filtered_files)
        else:
            xyz_files = []
            filtered_xyz_missing_props = total_geometries
        csv_only_payload = {name: csv_records[name] for name in csv_only_names}
        csv_ignored = not matched_names and not csv_only_names

    build_start = perf_counter()
    records = []
    for xyz_file in xyz_files:
        metadata = properties.get(xyz_file.name, {})
        record: Dict[str, Any] = {
            "identifier": xyz_file.name,
            "label": metadata.get("label", xyz_file.stem),
            "path": str(xyz_file),
            "source": "xyz",
            "selection_id": f"xyz::{xyz_file.name}",
            "has_geometry": True,
        }
        record.update(metadata)
        record["__index"] = len(records)
        records.append(record)

    for name in csv_only_names:
        metadata = csv_only_payload.get(name, {})
        record = {
            "identifier": name,
            "label": metadata.get("label", Path(name).stem),
            "path": None,
            "source": "csv_only",
            "selection_id": f"csv::{name}",
            "has_geometry": False,
        }
        record.update(metadata)
        record["__index"] = len(records)
        records.append(record)

    df = pd.DataFrame(records).convert_dtypes()
    add_timing("build_dataframe", build_start, records=len(records))
    if csv_metadata_path:
        df.attrs["csv_properties_path"] = csv_metadata_path
    df.attrs["csv_properties_ignored"] = csv_ignored
    if filtered_xyz_missing_props:
        df.attrs["csv_xyz_filtered"] = filtered_xyz_missing_props
    if csv_only_names:
        df.attrs["csv_only_count"] = len(csv_only_names)
        df.attrs["csv_only_names"] = csv_only_names
    if timings:
        df.attrs["load_timings"] = timings
    return df

@st.cache_data(show_spinner=False)
def load_ase_metadata(db_path: str) -> pd.DataFrame:
    database = Path(db_path).expanduser().resolve()
    if not database.exists():
        raise FileNotFoundError(f"ASE database not found: {database}")

    records = []
    with connect(str(database)) as handle:
        for row in handle.select():
            props = extract_row_properties(row)
            label = props.get("label") or row.get("name") or row.formula
            records.append(
                {
                    "identifier": str(row.id),
                    "label": label or f"id_{row.id}",
                    "db_path": str(database),
                    "db_id": row.id,
                    "source": "ase_db",
                    "selection_id": f"ase::{row.id}",
                    "__index": row.id,
                    **props,
                }
            )

    if not records:
        raise ValueError(f"No rows found in ASE database {database}")

    return pd.DataFrame(records).convert_dtypes()


def extract_row_properties(row: Any) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    if hasattr(row, "key_value_pairs"):
        props.update({k: v for k, v in row.key_value_pairs.items() if is_scalar(v)})
    if hasattr(row, "data"):
        props.update({k: v for k, v in row.data.items() if is_scalar(v)})
    for attr in ("energy", "charge", "magmom"):
        if hasattr(row, attr):
            value = getattr(row, attr)
            if is_scalar(value):
                props.setdefault(attr, value)
    return props


def is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool, np.number))


@st.cache_resource(show_spinner=False)
def load_atoms_from_xyz(path: str) -> Atoms:
    return ase_read(path)


@st.cache_resource(show_spinner=False)
def load_atoms_from_ase(db_path: str, row_id: int) -> Atoms:
    with connect(db_path) as handle:
        return handle.get(id=row_id).toatoms()

def load_atoms_raw(record: pd.Series) -> Atoms:
    if record.source == "xyz":
        return load_atoms_from_xyz(record.path)
    if record.source == "ase_db":
        return load_atoms_from_ase(record.db_path, int(record.db_id))
    raise ValueError(f"Unsupported source type: {record.source}")

def get_atoms(record: pd.Series) -> Optional[Atoms]:
    try:
        return load_atoms_raw(record)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load structure for {record.label}: {exc}")
    return None
