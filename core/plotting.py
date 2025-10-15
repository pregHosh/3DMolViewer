"""Interactive plotting utilities for CSV data visualization."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class InteractivePlotter:
    """Handles interactive plot creation from CSV data."""

    def __init__(self) -> None:
        """Initialize the plotter."""
        self.supported_plot_types = [
            "scatter",
            "line",
            "histogram",
            "box",
            "violin",
            "bar",
        ]

    def create_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str | None = None,
        size_col: str | None = None,
        hover_data: list[str] | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Create an interactive scatter plot.
        
        Args:
            df: DataFrame containing the data
            x_col: Column name for x-axis
            y_col: Column name for y-axis  
            color_col: Optional column for color coding
            size_col: Optional column for size coding
            hover_data: Additional columns to show on hover
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Prepare plot arguments
        plot_kwargs = {
            "data_frame": df,
            "x": x_col,
            "y": y_col,
            "hover_data": hover_data or [],
            "custom_data": [df.index]
        }

        if color_col and color_col in df.columns:
            plot_kwargs["color"] = color_col

        if size_col and size_col in df.columns:
            plot_kwargs["size"] = size_col

        if title:
            plot_kwargs["title"] = title

        # Create the plot
        fig = px.scatter(**plot_kwargs)

        # Add trendline if both axes are numeric
        if (
            pd.api.types.is_numeric_dtype(df[x_col])
            and pd.api.types.is_numeric_dtype(df[y_col])
        ):
            self._add_trendline(fig, df, x_col, y_col)

        # Add correlation coefficient if applicable
        if (
            pd.api.types.is_numeric_dtype(df[x_col])
            and pd.api.types.is_numeric_dtype(df[y_col])
        ):
            self._add_correlation_annotation(fig, df, x_col, y_col)

        # Customize layout
        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            height=500,
            dragmode="select"
        )

        return fig

    def create_histogram(
        self,
        df: pd.DataFrame,
        col: str,
        bins: int = 30,
        color_col: str | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Create a histogram.
        
        Args:
            df: DataFrame containing the data
            col: Column to create histogram for
            bins: Number of bins
            color_col: Optional column for color grouping
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        plot_kwargs = {
            "data_frame": df,
            "x": col,
            "nbins": bins,
            "title": title or f"Distribution of {col}",
        }

        if color_col and color_col in df.columns:
            plot_kwargs["color"] = color_col

        fig = px.histogram(**plot_kwargs)

        # Add statistics annotation
        self._add_distribution_stats(fig, df, col)

        return fig

    def create_box_plot(
        self,
        df: pd.DataFrame,
        x_col: str | None,
        y_col: str,
        color_col: str | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Create a box plot.
        
        Args:
            df: DataFrame containing the data
            x_col: Optional column for grouping (categorical)
            y_col: Column for values (numeric)
            color_col: Optional column for color coding
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        plot_kwargs = {
            "data_frame": df,
            "y": y_col,
            "title": title or f"Box plot of {y_col}",
        }

        if x_col and x_col in df.columns:
            plot_kwargs["x"] = x_col

        if color_col and color_col in df.columns:
            plot_kwargs["color"] = color_col

        return px.box(**plot_kwargs)

    def create_line_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Create a line plot.
        
        Args:
            df: DataFrame containing the data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            color_col: Optional column for line grouping
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        plot_kwargs = {
            "data_frame": df,
            "x": x_col,
            "y": y_col,
            "title": title or f"{y_col} vs {x_col}",
        }

        if color_col and color_col in df.columns:
            plot_kwargs["color"] = color_col

        return px.line(**plot_kwargs)

    def create_parity_plot(
        self,
        df: pd.DataFrame,
        true_col: str,
        pred_col: str,
        label_col: str | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Create a parity plot (predicted vs true values).
        
        Args:
            df: DataFrame containing the data
            true_col: Column with true values
            pred_col: Column with predicted values  
            label_col: Optional column for point labels
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Add scatter plot
        scatter_kwargs = {
            "x": df[true_col],
            "y": df[pred_col],
            "mode": "markers",
            "marker": {"size": 8, "opacity": 0.6},
            "name": "Data points",
        }

        if label_col and label_col in df.columns:
            scatter_kwargs["text"] = df[label_col]
            scatter_kwargs["hovertemplate"] = (
                "True: %{x}<br>Pred: %{y}<br>Label: %{text}<extra></extra>"
            )

        fig.add_trace(go.Scatter(**scatter_kwargs))

        # Add perfect prediction line
        min_val = min(df[true_col].min(), df[pred_col].min())
        max_val = max(df[true_col].max(), df[pred_col].max())

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line={"color": "red", "dash": "dash", "width": 2},
                name="Perfect prediction",
            )
        )

        # Calculate and add metrics
        metrics = self._calculate_regression_metrics(
            df[true_col].values, df[pred_col].values
        )
        self._add_metrics_annotation(fig, metrics)

        fig.update_layout(
            title=title or "Parity Plot",
            xaxis_title=f"True {true_col}",
            yaxis_title=f"Predicted {pred_col}",
            showlegend=True,
            height=500,
        )

        return fig

    def _add_trendline(
        self, fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str
    ) -> None:
        """Add a trendline to the scatter plot."""
        # Remove NaN values for trendline calculation
        clean_df = df[[x_col, y_col]].dropna()
        min_data_points = 2
        if len(clean_df) < min_data_points:
            return

        # Calculate linear regression
        coeffs = np.polyfit(clean_df[x_col], clean_df[y_col], 1)
        x_trend = np.array([clean_df[x_col].min(), clean_df[x_col].max()])
        y_trend = coeffs[0] * x_trend + coeffs[1]

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                line={"color": "red", "width": 2},
                name=f"Trendline (slope={coeffs[0]:.3f})",
            )
        )

    def _add_correlation_annotation(
        self, fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str
    ) -> None:
        """Add correlation coefficient annotation."""
        clean_df = df[[x_col, y_col]].dropna()
        min_data_points = 2
        if len(clean_df) < min_data_points:
            return

        corr = clean_df[x_col].corr(clean_df[y_col])
        if not np.isnan(corr):
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Correlation: {corr:.3f}",
                showarrow=False,
                bgcolor="wheat",
                bordercolor="black",
                borderwidth=1,
            )

    def _add_distribution_stats(
        self, fig: go.Figure, df: pd.DataFrame, col: str
    ) -> None:
        """Add distribution statistics annotation."""
        stats = df[col].describe()
        stats_text = (
            f"Mean: {stats['mean']:.3f}<br>"
            f"Std: {stats['std']:.3f}<br>"
            f"Median: {stats['50%']:.3f}<br>"
            f"Count: {int(stats['count'])}"
        )

        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="lightblue",
            bordercolor="black",
            borderwidth=1,
            align="left",
        )

    def _calculate_regression_metrics(
        self, true_values: np.ndarray, pred_values: np.ndarray
    ) -> dict[str, float]:
        """Calculate regression metrics."""
        mae = np.mean(np.abs(true_values - pred_values))
        mse = np.mean((true_values - pred_values) ** 2)
        rmse = np.sqrt(mse)

        # R² score
        ss_res = np.sum((true_values - pred_values) ** 2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Pearson correlation
        corr = np.corrcoef(true_values, pred_values)[0, 1]

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2,
            "Correlation": corr,
        }

    def _add_metrics_annotation(
        self, fig: go.Figure, metrics: dict[str, float]
    ) -> None:
        """Add metrics annotation to the plot."""
        metrics_text = "<br>".join(
            [f"{key}: {value:.3f}" for key, value in metrics.items()]
        )

        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=metrics_text,
            showarrow=False,
            bgcolor="wheat",
            bordercolor="black",
            borderwidth=1,
            font={"size": 10},
        )


def create_plot_from_config(
    df: pd.DataFrame, plot_config: dict[str, Any]
) -> go.Figure:
    """Create a plot from a configuration dictionary.
    
    Args:
        df: DataFrame containing the data
        plot_config: Dictionary with plot configuration
        
    Returns:
        Plotly figure object
    """
    plotter = InteractivePlotter()
    plot_type = plot_config.get("type", "scatter")

    if plot_type == "scatter":
        return plotter.create_scatter_plot(
            df,
            plot_config["x_col"],
            plot_config["y_col"],
            plot_config.get("color_col"),
            plot_config.get("size_col"),
            plot_config.get("hover_data"),
            plot_config.get("title"),
        )
    if plot_type == "histogram":
        return plotter.create_histogram(
            df,
            plot_config["col"],
            plot_config.get("bins", 30),
            plot_config.get("color_col"),
            plot_config.get("title"),
        )
    if plot_type == "box":
        return plotter.create_box_plot(
            df,
            plot_config.get("x_col"),
            plot_config["y_col"],
            plot_config.get("color_col"),
            plot_config.get("title"),
        )
    if plot_type == "line":
        return plotter.create_line_plot(
            df,
            plot_config["x_col"],
            plot_config["y_col"],
            plot_config.get("color_col"),
            plot_config.get("title"),
        )
    if plot_type == "parity":
        return plotter.create_parity_plot(
            df,
            plot_config["true_col"],
            plot_config["pred_col"],
            plot_config.get("label_col"),
            plot_config.get("title"),
        )
    msg = f"Unsupported plot type: {plot_type}"
    raise ValueError(msg)
