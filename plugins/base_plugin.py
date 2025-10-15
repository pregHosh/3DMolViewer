"""Base plugin interface for extending the lite app functionality."""

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import streamlit as st


class BasePlugin(ABC):
    """Abstract base class for all plugins."""

    def __init__(self, name: str, description: str) -> None:
        """Initialize the plugin.
        
        Args:
            name: Plugin name
            description: Plugin description
        """
        self.name = name
        self.description = description
        self.enabled = False

    @abstractmethod
    def render_ui(self, df: pd.DataFrame) -> None:
        """Render the plugin's user interface.
        
        Args:
            df: The loaded DataFrame
        """
        pass

    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> dict[str, Any]:
        """Process data and return results.
        
        Args:
            df: The input DataFrame
            
        Returns:
            Dictionary with processing results
        """
        pass

    def get_requirements(self) -> list[str]:
        """Get additional package requirements for this plugin.
        
        Returns:
            List of package names
        """
        return []

    def is_compatible(self, df: pd.DataFrame) -> bool:
        """Check if plugin is compatible with the given DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if compatible, False otherwise
        """
        return True


class PluginManager:
    """Manages plugin registration and execution."""

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self.plugins: dict[str, BasePlugin] = {}

    def register_plugin(self, plugin: BasePlugin) -> None:
        """Register a new plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        self.plugins[plugin.name] = plugin

    def get_compatible_plugins(self, df: pd.DataFrame) -> list[BasePlugin]:
        """Get plugins compatible with the given DataFrame.
        
        Args:
            df: DataFrame to check compatibility
            
        Returns:
            List of compatible plugins
        """
        return [
            plugin for plugin in self.plugins.values() 
            if plugin.is_compatible(df)
        ]

    def render_plugin_selector(self, df: pd.DataFrame) -> BasePlugin | None:
        """Render plugin selection UI.
        
        Args:
            df: DataFrame for compatibility checking
            
        Returns:
            Selected plugin or None
        """
        compatible_plugins = self.get_compatible_plugins(df)
        
        if not compatible_plugins:
            st.info("No compatible plugins available for this dataset")
            return None

        st.subheader("🔌 Available Plugins")
        
        plugin_names = [plugin.name for plugin in compatible_plugins]
        selected_name = st.selectbox(
            "Select a plugin",
            ["None"] + plugin_names,
            help="Choose a plugin to extend functionality"
        )
        
        if selected_name == "None":
            return None
        
        selected_plugin = self.plugins[selected_name]
        
        # Show plugin info
        with st.expander(f"ℹ️ About {selected_plugin.name}"):
            st.write(selected_plugin.description)
            
            requirements = selected_plugin.get_requirements()
            if requirements:
                st.write("**Additional requirements:**")
                for req in requirements:
                    st.code(req)
        
        return selected_plugin