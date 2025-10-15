"""Lightweight Molecular Visualization App - Main Application."""

import streamlit as st
import pandas as pd

from core.plotting import create_plot_from_config
from core.molecule_viz import display_molecule_from_file, display_molecule_from_data
from core.utils import CSVHandler, DataProcessor, ConfigManager, fix_dataframe_for_display
from ase import Atoms

class LiteVizApp:
    """Main application class for the lightweight visualization app."""

    def __init__(self) -> None:
        """Initialize the app."""
        self.csv_handler = CSVHandler()
        
        # Initialize session state
        if "df" not in st.session_state:
            st.session_state.df = None
        if "selected_indices" not in st.session_state:
            st.session_state.selected_indices = []

    def run(self) -> None:
        """Run the main application."""
        st.set_page_config(
            page_title="Lite Molecular Viz",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🧬 Lightweight Molecular Visualization App")
        st.markdown(
            "Upload your CSV data and create interactive plots with integrated molecular visualization"
        )

        self._setup_sidebar()

        with st.expander("📂 Upload Data", expanded=True):
            st.session_state.df = self._handle_data_upload()
        
        if st.session_state.df is not None:
            self.csv_handler._display_dataframe_info(st.session_state.df)
            self._show_main_interface(st.session_state.df)
        else:
            self._show_welcome_screen()

    def _setup_sidebar(self) -> None:
        """Set up the sidebar configuration."""
        st.sidebar.title("⚙️ Configuration")
        
        st.sidebar.subheader("Molecule Data Source")
        self.data_source = st.sidebar.text_input(
            "Source (XYZ dir or .db file)",
            "test_xyz",
            help="Path to XYZ directory or .db file for visualization."
        )
        
        st.sidebar.subheader("Display Options")
        self.n_columns = st.sidebar.slider("Number of Molecule Columns", 1, 5, 3)

        with st.sidebar.expander("ℹ️ About This App"):
            st.markdown("""
            **Lightweight Molecular Visualization App**
            - Upload CSV or ASE database files.
            - Create interactive plots from any columns.  
            - Visualize molecules from a specified source directory or DB.
            """)

    def _handle_data_upload(self) -> pd.DataFrame | None:
        """Handle CSV file upload and processing."""
        df = self.csv_handler.create_upload_widget("main_csv")
        
        if df is not None:
            df = DataProcessor.clean_dataframe(df)
            st.session_state.df = df
            return df
        
        return st.session_state.df

    def _show_welcome_screen(self) -> None:
        """Show welcome screen when no data is loaded."""
        st.info("Select a data source to get started.")

    def _show_main_interface(self, df: pd.DataFrame) -> None:
        """Show the main interface with data and plots."""
        tab1, tab2 = st.tabs(["📊 Interactive Plots", "📋 Data Explorer"])
        with tab1:
            self._show_plotting_interface(df)
        with tab2:
            self._show_data_explorer(df)

    def _show_plotting_interface(self, df: pd.DataFrame) -> None:
        """Show the interactive plotting interface."""
        plot_config = ConfigManager.get_plot_config_widget(df)
        mol_config = ConfigManager.get_molecule_config_widget(df)

        left_panel, right_panel = st.columns([2, 1])

        with left_panel:
            self._create_and_display_plot(df, plot_config, mol_config)
            if st.session_state.selected_indices:
                st.subheader("Selected Data")
                if st.button("Clear Selection"):
                    st.session_state.selected_indices = []
                    st.rerun()
                selected_df = df.loc[st.session_state.selected_indices]
                st.dataframe(fix_dataframe_for_display(selected_df))

        with right_panel:
            st.subheader("3D Molecules")
            if st.session_state.selected_indices:
                if mol_config.get("enabled"):
                    selected_df = df.loc[st.session_state.selected_indices]
                    self._display_selected_molecules(selected_df, mol_config)
                else:
                    st.info("Molecule display is not enabled in the configuration.")
            else:
                st.info("Select data points on the plot to see molecules.")

    def _create_and_display_plot(self, df: pd.DataFrame, plot_config: dict, mol_config: dict) -> None:
        """Create and display the plot and handle selections."""
        try:
            if mol_config.get("enabled") and mol_config.get("molecule_col"):
                if "hover_data" not in plot_config:
                    plot_config["hover_data"] = []
                plot_config["hover_data"].append(mol_config["molecule_col"])
            
            fig = create_plot_from_config(df, plot_config)
            
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
            self._handle_plot_selection_events(event)

        except Exception as e:
            st.error(f"Error creating plot: {e}")

    def _handle_plot_selection_events(self, event) -> None:
        """Handle plot selection events to update session state."""
        if event.selection and event.selection["points"]:
            st.session_state.selected_indices = [p["customdata"][0] for p in event.selection["points"]]
        else:
            # This case handles deselection (e.g., double-clicking the plot)
            st.session_state.selected_indices = []

    def _display_selected_molecules(self, selected_df: pd.DataFrame, mol_config: dict):
        st.subheader("🧬 Selected Molecule Structures")
        
        cols = st.columns(self.n_columns)
        for i, (idx, row) in enumerate(selected_df.iterrows()):
            with cols[i % self.n_columns]:
                molecule_name = row[mol_config["molecule_col"]]
                st.write(f"**{molecule_name}**")
                try:
                    if 'atom_object' in row and isinstance(row['atom_object'], Atoms):
                        display_molecule_from_data(row['atom_object'])
                    else:
                        display_molecule_from_file(self.data_source, str(molecule_name))
                except Exception as e:
                    st.error(f"Could not display {molecule_name}: {e}")

    def _show_data_explorer(self, df: pd.DataFrame) -> None:
        """Show the data exploration interface."""
        st.subheader("📋 Data Explorer")
        display_df = fix_dataframe_for_display(df.copy())
        st.dataframe(display_df, use_container_width=True, height=600)

def main() -> None:
    """Main entry point."""
    app = LiteVizApp()
    app.run()

if __name__ == "__main__":
    main()