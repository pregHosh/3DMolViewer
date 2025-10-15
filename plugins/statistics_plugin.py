"""Example plugin for statistical analysis."""

from typing import Any
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

from .base_plugin import BasePlugin


class StatisticsPlugin(BasePlugin):
    """Plugin for basic statistical analysis and correlation plots."""

    def __init__(self) -> None:
        """Initialize the statistics plugin."""
        super().__init__(
            name="Statistical Analysis",
            description="Provides correlation analysis, distribution plots, and statistical summaries for numeric data"
        )

    def is_compatible(self, data: pd.DataFrame) -> bool:
        """Check if the dataset has numeric columns for analysis."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return len(numeric_cols) >= 2

    def render_ui(self, data: pd.DataFrame) -> None:
        """Render the statistical analysis interface."""
        st.subheader("📊 Statistical Analysis")
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for statistical analysis")
            return
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Correlation Analysis", "Distribution Analysis", "Summary Statistics"],
            help="Choose the type of statistical analysis"
        )
        
        if analysis_type == "Correlation Analysis":
            self._render_correlation_analysis(data, numeric_cols)
        elif analysis_type == "Distribution Analysis":
            self._render_distribution_analysis(data, numeric_cols)
        else:
            self._render_summary_statistics(data, numeric_cols)

    def process_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """Process data and return statistical results."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        results = {
            "correlation_matrix": data[numeric_cols].corr().to_dict(),
            "summary_stats": data[numeric_cols].describe().to_dict(),
            "column_count": len(numeric_cols)
        }
        
        return results

    def _render_correlation_analysis(self, data: pd.DataFrame, numeric_cols: list[str]) -> None:
        """Render correlation analysis interface."""
        st.write("**Correlation Matrix**")
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            zmin=-1,
            zmax=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation pairs
        st.write("**Strongest Correlations**")
        
        # Get upper triangle of correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if mask[i, j]:
                    corr_pairs.append({
                        "Variable 1": corr_matrix.columns[i],
                        "Variable 2": corr_matrix.columns[j],
                        "Correlation": corr_matrix.iloc[i, j]
                    })
        
        # Sort by absolute correlation
        corr_pairs.sort(key=lambda x: abs(x["Correlation"]), reverse=True)
        
        # Display top correlations
        top_corr = pd.DataFrame(corr_pairs[:10])  # Top 10
        if not top_corr.empty:
            top_corr["Correlation"] = top_corr["Correlation"].round(3)
            st.dataframe(top_corr, use_container_width=True)

    def _render_distribution_analysis(self, data: pd.DataFrame, numeric_cols: list[str]) -> None:
        """Render distribution analysis interface."""
        st.write("**Distribution Analysis**")
        
        selected_col = st.selectbox(
            "Select column for distribution analysis",
            numeric_cols,
            help="Choose a numeric column to analyze its distribution"
        )
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig_hist = px.histogram(
                    data,
                    x=selected_col,
                    title=f"Distribution of {selected_col}",
                    marginal="box"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(
                    data,
                    y=selected_col,
                    title=f"Box Plot of {selected_col}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Distribution statistics
            st.write(f"**Statistics for {selected_col}**")
            stats = data[selected_col].describe()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{stats['mean']:.3f}")
                st.metric("Std Dev", f"{stats['std']:.3f}")
            with col2:
                st.metric("Minimum", f"{stats['min']:.3f}")
                st.metric("Maximum", f"{stats['max']:.3f}")
            with col3:
                st.metric("Median", f"{stats['50%']:.3f}")
                st.metric("Skewness", f"{data[selected_col].skew():.3f}")

    def _render_summary_statistics(self, data: pd.DataFrame, numeric_cols: list[str]) -> None:
        """Render summary statistics interface."""
        st.write("**Summary Statistics**")
        
        # Overall summary
        summary = data[numeric_cols].describe()
        st.dataframe(summary, use_container_width=True)
        
        # Missing data analysis
        st.write("**Missing Data Analysis**")
        missing_data = data[numeric_cols].isnull().sum()
        missing_pct = (missing_data / len(data)) * 100
        
        missing_df = pd.DataFrame({
            "Column": missing_data.index,
            "Missing Count": missing_data.values,
            "Missing %": missing_pct.values.round(2)
        })
        
        st.dataframe(missing_df, use_container_width=True)
        
        # Data type information
        st.write("**Data Types**")
        dtype_info = pd.DataFrame({
            "Column": numeric_cols,
            "Data Type": [str(data[col].dtype) for col in numeric_cols],
            "Unique Values": [data[col].nunique() for col in numeric_cols],
            "Memory Usage (MB)": [data[col].memory_usage(deep=True) / 1024 / 1024 for col in numeric_cols]
        })
        
        dtype_info["Memory Usage (MB)"] = dtype_info["Memory Usage (MB)"].round(3)
        st.dataframe(dtype_info, use_container_width=True)

    def get_requirements(self) -> list[str]:
        """Get additional requirements for statistics plugin."""
        return ["scipy>=1.9.0"]


# Example of how to extend towards full version
class AdvancedAnalysisPlugin(BasePlugin):
    """Example plugin showing how to extend functionality."""

    def __init__(self) -> None:
        """Initialize the advanced analysis plugin."""
        super().__init__(
            name="Advanced Analysis",
            description="Advanced statistical analysis including PCA, clustering, and regression analysis"
        )

    def is_compatible(self, data: pd.DataFrame) -> bool:
        """Check compatibility - needs numeric data and sufficient samples."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return len(numeric_cols) >= 3 and len(data) >= 10

    def render_ui(self, data: pd.DataFrame) -> None:
        """Render advanced analysis UI."""
        st.subheader("🔬 Advanced Analysis")
        
        st.info(
            "This plugin demonstrates how the lite app can be extended "
            "to include advanced features from the full version."
        )
        
        method = st.selectbox(
            "Analysis Method",
            ["Principal Component Analysis", "Clustering Analysis", "Regression Analysis"],
            help="Choose advanced analysis method"
        )
        
        if method == "Principal Component Analysis":
            st.write("**PCA Analysis**")
            st.info("Would perform PCA on numeric columns and show explained variance")
            
        elif method == "Clustering Analysis":
            st.write("**Clustering Analysis**")
            st.info("Would perform K-means clustering and visualize results")
            
        else:
            st.write("**Regression Analysis**")
            st.info("Would perform regression analysis between selected variables")

    def process_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """Process data for advanced analysis."""
        return {
            "message": "Advanced analysis plugin - extend with actual implementations",
            "numeric_columns": len(data.select_dtypes(include=[np.number]).columns)
        }

    def get_requirements(self) -> list[str]:
        """Get requirements for advanced analysis."""
        return ["scikit-learn>=1.1.0", "scipy>=1.9.0", "seaborn>=0.11.0"]