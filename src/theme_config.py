from dataclasses import dataclass
from typing import Dict

import streamlit as st


@dataclass
class ThemeConfig:
    background: str
    plot_bg: str
    text_color: str
    highlight: str
    plot_template: str


THEMES: Dict[str, ThemeConfig] = {
    "Light": ThemeConfig(
        background="#FAFBFF",
        plot_bg="#F1F3F9",
        text_color="#0F172A",
        highlight="#2E86DE",
        plot_template="plotly_white",
    ),
    "Dark": ThemeConfig(
        background="#0E1117",
        plot_bg="#131722",
        text_color="#EBEBF5",
        highlight="#F39C12",
        plot_template="plotly_dark",
    ),
}


def inject_theme_css(theme: ThemeConfig) -> None:
    extra_light_css = ""
    if theme.background.lower() == "#fafbff":
        extra_light_css = """
            /* Light theme input overrides */
            .stApp input[type='text'],
            .stApp input[type='number'],
            .stApp input[type='search'],
            .stApp textarea,
            .stApp .stTextInput > div > div,
            .stApp .stNumberInput > div > div,
            .stApp .stSelectbox > div > div,
            .stApp .stMultiSelect > div > div,
            .stApp [data-baseweb='select'] > div,
            .stApp [data-baseweb='textarea'] > div {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                border: 1px solid rgba(15, 23, 42, 0.18) !important;
                box-shadow: none !important;
            }
            .stApp [data-baseweb='select'] div[role='button'] {
                color: #0F172A !important;
            }
            .stApp .stDownloadButton button,
            .stApp .stDownloadButton button:hover,
            .stApp .stButton button,
            .stApp .stButton button:focus {
                color: #0F172A !important;
            }
            .stApp label, .stApp .stMarkdown p {
                color: #0F172A !important;
            }
        """

    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {theme.background};
                color: {theme.text_color};
            }}
            .stApp div, .stApp span, .stApp label, .stApp textarea, .stApp p {{
                color: {theme.text_color} !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: {theme.plot_bg};
            }}
            /* Sticky top navbar */
            .molviewer-navbar {{
                position: sticky;
                top: 0;
                z-index: 999;
                background: {theme.background};
                padding: 8px 12px;
                border-radius: 10px;
                border: 1px solid rgba(15,23,42,0.08);
                box-shadow: 0 6px 18px rgba(15,23,42,0.08);
                margin-bottom: 12px;
            }}
            .molviewer-navbar .small-label {{
                font-size: 12px;
                opacity: 0.7;
                margin-bottom: 4px;
                display: block;
            }}
            {extra_light_css}
        </style>
        """,
        unsafe_allow_html=True,
    )
