"""Utility functions for CSV handling and data processing."""

from typing import Any
import io

import pandas as pd
import streamlit as st
import numpy as np
from ase.db import connect

@st.cache_data
def load_ase_db_cached(db_path: str) -> pd.DataFrame | None:
    """Process an ASE database file and cache the result."""
    try:
        db = connect(db_path)
        records = []
        for row in db.select():
            record = {'filename': row.get('name', f'mol_{row.id}')}
            record.update(row.key_value_pairs)
            if hasattr(row, 'data') and row.data:
                record.update(row.data)
            
            # Store atomic data directly in the dataframe
            atoms = row.toatoms()
            record["atom_object"] = atoms

            records.append(record)
        
        if not records:
            st.warning("No molecules found in the database.")
            return None
        
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        st.error(f"Error reading ASE database: {e}")
        return None

class CSVHandler:
    """Handles CSV file upload and processing."""

    def __init__(self) -> None:
        """Initialize the CSV handler."""
        self.supported_separators = {
            "Comma (,)": ",",
            "Semicolon (;)": ";",
            "Tab": "\t",
            "Pipe (|)": "|",
        }

    def create_upload_widget(
        self,
        key: str = "csv_upload"
    ) -> pd.DataFrame | None:
        """Create CSV upload widget with options.
        
        Args:
            key: Unique key for the widget
            
        Returns:
            Loaded DataFrame or None
        """
        st.subheader("📁 Data Source")
        
        input_method = st.radio(
            "Select Data Source",
            ("ASE Database", "CSV File Path", "CSV File Upload"),
            horizontal=True
        )

        if input_method == "CSV File Upload":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=["csv", "txt"],
                key=key,
                help="Upload a CSV file with your experimental data"
            )
            if uploaded_file:
                return self._process_uploaded_file(uploaded_file)
        elif input_method == "CSV File Path":
            file_path = st.text_input(
                "Enter CSV file path",
                "test_xyz/xyz_properties.csv",
                key=f"{key}_path",
                help="Enter the local path to a CSV file"
            )
            if file_path:
                return self._process_file_path(file_path)
        elif input_method == "ASE Database":
            db_path = st.text_input(
                "Enter ASE database path",
                "examples/sampled_molecules.db",
                key=f"{key}_db_path",
                help="Enter the local path to an ASE database file"
            )
            if db_path:
                return self._process_ase_db(db_path)
        
        return None

    def _process_ase_db(self, db_path: str) -> pd.DataFrame | None:
        """Process an ASE database file."""
        return load_ase_db_cached(db_path)

    def _process_uploaded_file(self, uploaded_file) -> pd.DataFrame | None:
        """Process the uploaded CSV file with options."""
        try:
            content = uploaded_file.getvalue()
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                content_str = content.decode('latin-1')
            
            st.write("**File Preview (first 5 lines):**")
            preview_lines = content_str.split('\n')[:5]
            st.code('\n'.join(preview_lines))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                separator = st.selectbox("Separator", list(self.supported_separators.keys()))
                sep_char = self.supported_separators[separator]
            with col2:
                header_row = st.number_input("Header Row", 0, 10, 0)
            with col3:
                skip_rows = st.number_input("Skip Rows", 0, 20, 0)
            
            df = pd.read_csv(io.StringIO(content_str), sep=sep_char, header=header_row, skiprows=skip_rows)
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None

    def _process_file_path(self, file_path: str) -> pd.DataFrame | None:
        """Process a local CSV file path."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_str = f.read()
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return None
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content_str = f.read()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

        try:
            st.write("**File Preview (first 5 lines):**")
            preview_lines = content_str.split('\n')[:5]
            st.code('\n'.join(preview_lines))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                separator = st.selectbox("Separator", list(self.supported_separators.keys()))
                sep_char = self.supported_separators[separator]
            with col2:
                header_row = st.number_input("Header Row", 0, 10, 0)
            with col3:
                skip_rows = st.number_input("Skip Rows", 0, 20, 0)
            
            df = pd.read_csv(io.StringIO(content_str), sep=sep_char, header=header_row, skiprows=skip_rows)
            return df
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
            return None

    def _display_dataframe_info(self, df: pd.DataFrame) -> None:
        """Display basic information about the DataFrame."""
        st.success(f"✅ Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        with st.expander("📊 Dataset Info"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Columns:**")
                for i, col in enumerate(df.columns):
                    st.write(f"{i+1}. `{col}` ({df[col].dtype}) - {df[col].count()} non-null")
            with col2:
                st.write("**Sample Data:**")
                # Create a copy of the head for display purposes to avoid Arrow errors
                df_head_display = df.head(3).copy()
                if 'atomic_numbers' in df_head_display.columns:
                    df_head_display['atomic_numbers'] = df_head_display['atomic_numbers'].astype(str)
                if 'positions' in df_head_display.columns:
                    df_head_display['positions'] = df_head_display['positions'].astype(str)
                if 'atom_object' in df_head_display.columns:
                    df_head_display.pop('atom_object')  # Remove complex objects for display
                st.dataframe(df_head_display, use_container_width=True)


class DataProcessor:
    """Handles data processing and cleaning operations."""

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() <= max_unique:
                categorical_cols.append(col)
        return categorical_cols

    @staticmethod
    def get_text_columns(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=['object', 'string']).columns.tolist()

    @staticmethod
    def detect_molecule_name_column(df: pd.DataFrame) -> str | None:
        molecule_keywords = ['molecule', 'mol', 'name', 'id', 'compound', 'smiles', 'structure', 'csd', 'ccdc', 'entry', 'label', 'filename']
        for col in df.columns:
            col_lower = col.lower()
            for keyword in molecule_keywords:
                if keyword in col_lower:
                    return col
        text_cols = DataProcessor.get_text_columns(df)
        for col in text_cols:
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                alphanumeric_count = sum(1 for val in sample_values if isinstance(val, str) and val.replace('-', '').replace('_', '').isalnum())
                if alphanumeric_count >= len(sample_values) * 0.8:
                    return col
        return None

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
        text_cols = DataProcessor.get_text_columns(df_clean)
        for col in text_cols:
            if col in ['atomic_numbers', 'positions', 'atom_object']:
                continue
            df_clean[col] = df_clean[col].astype(str).str.strip()
        return df_clean

    @staticmethod
    def get_column_stats(df: pd.DataFrame, column: str) -> dict[str, Any]:
        stats = {}
        col_data = df[column]
        stats['dtype'] = str(col_data.dtype)
        stats['count'] = len(col_data)
        stats['non_null'] = col_data.count()
        stats['null_count'] = col_data.isnull().sum()
        stats['unique_count'] = col_data.nunique()
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({'mean': col_data.mean(), 'std': col_data.std(), 'min': col_data.min(), 'max': col_data.max(), 'median': col_data.median()})
        else:
            value_counts = col_data.value_counts()
            stats.update({'most_common': value_counts.index[0] if not value_counts.empty else None, 'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0, 'sample_values': col_data.dropna().head(5).tolist()})
        return stats


class ConfigManager:
    """Manages app configuration and settings."""

    @staticmethod
    def get_plot_config_widget(df: pd.DataFrame) -> dict[str, Any]:
        st.subheader("📊 Plot Configuration")
        numeric_cols = DataProcessor.get_numeric_columns(df)
        categorical_cols = DataProcessor.get_categorical_columns(df)
        all_cols = df.columns.tolist()
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_type = st.selectbox("Plot Type", ["scatter", "line", "histogram", "box", "parity"], help="Choose the type of plot to create")
        with col2:
            x_col = None
            if plot_type in ["scatter", "line", "parity"]:
                x_col = st.selectbox("X-axis Column", numeric_cols if numeric_cols else all_cols, help="Choose column for X-axis")
        with col3:
            y_col = None
            if plot_type in ["scatter", "line", "box", "parity"]:
                y_col_options = [col for col in numeric_cols if col != x_col] if x_col else numeric_cols
                y_col = st.selectbox("Y-axis Column", y_col_options if y_col_options else all_cols, help="Choose column for Y-axis")
            elif plot_type == "histogram":
                y_col = st.selectbox("Column to Plot", numeric_cols if numeric_cols else all_cols, help="Choose column for histogram")
        config = {"type": plot_type}
        if plot_type in ["scatter", "line"]:
            config.update({"x_col": x_col, "y_col": y_col})
        elif plot_type == "histogram":
            config["col"] = y_col
        elif plot_type == "box":
            config["y_col"] = y_col
        elif plot_type == "parity":
            config.update({"true_col": x_col, "pred_col": y_col})
        return config

    @staticmethod  
    def get_molecule_config_widget(df: pd.DataFrame) -> dict[str, Any]:
        st.subheader("🧬 Molecule Visualization")
        config = {"enabled": False}
        suggested_col = DataProcessor.detect_molecule_name_column(df)
        if not suggested_col:
            st.warning("Could not automatically detect a molecule name column.")
            return config
        enable_mol_viz = st.checkbox("Enable molecule visualization", value=True, help="Show molecular structures when clicking on plot points")
        if enable_mol_viz:
            text_cols = DataProcessor.get_text_columns(df)
            default_idx = text_cols.index(suggested_col) if suggested_col in text_cols else 0
            molecule_col = st.selectbox("Molecule Name Column", text_cols, index=default_idx, help="Column containing molecule names/identifiers")
            config.update({"enabled": True, "molecule_col": molecule_col})
        return config


def fix_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df_fixed = df.copy()
    for col in df_fixed.columns:
        if pd.api.types.is_numeric_dtype(df_fixed[col]):
            df_fixed[col] = df_fixed[col].replace([np.inf, -np.inf], np.nan)
        elif df_fixed[col].dtype == "object":
            df_fixed[col] = df_fixed[col].astype(str).replace(["nan", "None", "<NA>"], "")
    return df_fixed