"""UI styling and custom CSS"""

import streamlit as st
from ..utils.config import (
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    BACKGROUND_COLOR,
    PRIMARY_FONT,
    SECONDARY_FONT,
    TITLE_FONT_SIZE,
    HEADER_FONT_SIZE,
)


def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app"""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family={PRIMARY_FONT}&family={SECONDARY_FONT}&display=swap');

            .stApp {{
                background-color: {BACKGROUND_COLOR};
                font-family: '{PRIMARY_FONT}', sans-serif;
            }}

            h1 {{
                color: {PRIMARY_COLOR};
                font-family: '{SECONDARY_FONT}', serif;
                font-size: {TITLE_FONT_SIZE};
                font-weight: bold;
                text-align: center;
                margin-top: 30px;
                margin-bottom: 30px;
            }}

            .stButton > button {{
                background-color: {PRIMARY_COLOR};
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px 20px;
                font-family: '{PRIMARY_FONT}', sans-serif;
            }}

            .stButton > button:hover {{
                background-color: {SECONDARY_COLOR};
            }}

            .stImage > img {{
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }}

            .streamlit-expanderHeader {{
                font-size: {HEADER_FONT_SIZE};
                font-weight: bold;
                color: {PRIMARY_COLOR};
                font-family: '{SECONDARY_FONT}', serif;
            }}

            .stAlert > div {{
                background-color: #f1c40f;
                color: #333;
                border-radius: 5px;
                padding: 10px;
                font-family: '{PRIMARY_FONT}', sans-serif;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_about_section():
    """Render the about section at the bottom of the page"""
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 50px; font-family: '{PRIMARY_FONT}', sans-serif;">
            <h2 style="color: {PRIMARY_COLOR};">About Arabic Sign Language Detection</h2>
            <p style="color:#000000;">This application uses advanced AI to detect and interpret Arabic sign language in real-time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
