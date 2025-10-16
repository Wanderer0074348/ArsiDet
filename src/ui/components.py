"""UI components for the Streamlit app"""

import streamlit as st
from typing import Tuple, List, Dict, Optional


class InputModeSelector:
    """Manages input mode selection (local camera, upload, or browser camera)"""

    @staticmethod
    def render_mode_selector() -> str:
        """
        Render input mode selector

        Returns:
            Selected mode: 'webrtc', 'browser_camera', 'upload', or 'local_camera'
        """
        with st.sidebar:
            st.header("Input Mode")
            mode = st.radio(
                "Select Input Method",
                options=['webrtc', 'browser_camera', 'local_camera'],
                format_func=lambda x: {
                    'webrtc': '🌐 WebRTC Live (RECOMMENDED for Cloud)',
                    'browser_camera': '📸 Browser Camera Snapshot',
                    'local_camera': '🎥 Local Webcam (Local Only)'
                }[x],
                help="WebRTC provides LIVE video streaming and works on all cloud platforms!"
            )

            if mode == 'webrtc':
                st.success("✅ **Best choice for deployment!**\nReal-time live video streaming")
            elif mode == 'local_camera':
                st.warning("⚠️ Local Webcam only works when running locally, not on cloud platforms")

            return mode


class CameraControls:
    """Manages camera control buttons and state"""

    @staticmethod
    def initialize_state():
        """Initialize session state for camera control"""
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
        if 'interpretation_history' not in st.session_state:
            st.session_state.interpretation_history = []

    @staticmethod
    def render_controls() -> Tuple[bool, bool]:
        """
        Render start/stop camera buttons

        Returns:
            Tuple of (camera_on, state_changed)
        """
        CameraControls.initialize_state()

        col1, col2 = st.columns(2)

        state_changed = False

        with col1:
            if not st.session_state.camera_on:
                if st.button("Start Camera", key="start_camera"):
                    st.session_state.camera_on = True
                    state_changed = True

        with col2:
            if st.session_state.camera_on:
                if st.button("Stop Camera", key="stop_camera"):
                    st.session_state.camera_on = False
                    state_changed = True

        return st.session_state.camera_on, state_changed


def create_video_placeholder():
    """
    Create a placeholder for video display

    Returns:
        Streamlit empty placeholder
    """
    return st.empty()


def create_interpretation_placeholder():
    """
    Create a placeholder for AI interpretations

    Returns:
        Streamlit empty placeholder
    """
    return st.empty()


def render_api_key_input() -> str:
    """
    Render input field for OpenAI API key

    Returns:
        API key entered by user
    """
    with st.sidebar:
        st.header("AI Settings")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable AI-powered sentence interpretation",
            placeholder="sk-..."
        )

        if api_key:
            st.success("API Key configured!")
            st.info("AI will interpret signs every 20 seconds")
        else:
            st.warning("Add your OpenAI API key to enable AI interpretation")

    return api_key


def render_interpretation_history():
    """Render the interpretation history"""
    # Initialize if not exists
    if 'interpretation_history' not in st.session_state:
        st.session_state.interpretation_history = []

    if st.session_state.interpretation_history:
        st.subheader("Interpretation History")

        for i, interpretation in enumerate(reversed(st.session_state.interpretation_history)):
            with st.expander(f"Interpretation {len(st.session_state.interpretation_history) - i} - {interpretation['timestamp'].strftime('%H:%M:%S')}", expanded=(i == 0)):
                if interpretation['arabic']:
                    st.markdown(f"**Arabic:** {interpretation['arabic']}")
                if interpretation['english']:
                    st.markdown(f"**English:** {interpretation['english']}")
                st.caption(f"Detected words: {', '.join(interpretation['detected_words'])}")
