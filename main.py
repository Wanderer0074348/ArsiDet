"""
Arabic Sign Language Detection Application

This is the main entry point for the Streamlit application with AI-powered interpretation.
"""

import streamlit as st

from src.core.model import get_model_manager
from src.core.video_processor import VideoProcessor
from src.core.browser_camera_processor import BrowserCameraProcessor
from src.core.ai_agent import SignLanguageInterpreter
from src.ui.styles import apply_custom_styles, render_about_section
from src.ui.components import (
    InputModeSelector,
    CameraControls,
    create_video_placeholder,
    create_interpretation_placeholder,
    render_api_key_input,
    render_interpretation_history,
)
from src.utils.config import APP_TITLE


def main():
    """Main application entry point"""
    # Apply custom styling
    apply_custom_styles()

    # Set page title
    st.title(APP_TITLE)

    # Render API key input in sidebar
    api_key = render_api_key_input()

    # Render input mode selector
    input_mode = InputModeSelector.render_mode_selector()

    # Load model manager (cached)
    model_manager = get_model_manager()

    # Initialize AI interpreter if API key is provided
    ai_interpreter = None
    if api_key:
        ai_interpreter = SignLanguageInterpreter(api_key=api_key)

    # Create interpretation placeholder if AI is enabled
    interpretation_placeholder = None
    if ai_interpreter:
        st.markdown("---")
        st.subheader("AI Interpretation")
        interpretation_placeholder = create_interpretation_placeholder()
        st.markdown("---")

    # Handle different input modes
    if input_mode == 'browser_camera':
        # Browser camera mode (works on cloud)
        st.subheader("📸 Browser Camera Mode")
        st.info("✅ This mode works on cloud platforms like Render, Streamlit Cloud, and Hugging Face")

        browser_processor = BrowserCameraProcessor(
            model_manager=model_manager,
            ai_interpreter=ai_interpreter
        )
        browser_processor.run_camera_input(
            interpretation_placeholder=interpretation_placeholder
        )

    elif input_mode == 'local_camera':
        # Local webcam mode (OpenCV - only works locally)
        st.subheader("🎥 Local Webcam Mode")

        # Create placeholders
        video_placeholder = create_video_placeholder()

        # Render camera controls
        camera_on, _ = CameraControls.render_controls()

        # Run detection if camera is on
        if camera_on:
            video_processor = VideoProcessor(
                model_manager=model_manager,
                ai_interpreter=ai_interpreter
            )
            video_processor.run_detection_loop(
                video_placeholder=video_placeholder,
                interpretation_placeholder=interpretation_placeholder
            )

    # Render interpretation history (persists after camera is stopped)
    if ai_interpreter:
        st.markdown("---")
        render_interpretation_history()

    # Render about section
    render_about_section()


if __name__ == "__main__":
    main()
