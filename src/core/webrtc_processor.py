"""WebRTC video processing for cloud deployment with live camera feed"""

import av
import cv2
import numpy as np
import streamlit as st
from typing import Optional, List
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import time
import logging

from .model import ModelManager
from .ai_agent import SignLanguageInterpreter
from ..utils.config import CONFIDENCE_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebRTC configuration for cloud deployment with multiple STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ],
        "iceTransportPolicy": "all",
    }
)


class VideoProcessor(VideoProcessorBase):
    """Video processor for WebRTC stream - runs in async context"""

    def __init__(self, model_manager: ModelManager, conf_threshold: float = CONFIDENCE_THRESHOLD):
        self.model_manager = model_manager
        self.conf_threshold = conf_threshold
        self.detected_labels = []
        self.last_labels = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process incoming video frame

        Args:
            frame: Input video frame

        Returns:
            Processed video frame with detections
        """
        # Convert from av.VideoFrame to numpy array
        img = frame.to_ndarray(format="bgr24")

        try:
            # Run model prediction
            results = self.model_manager.predict(img, conf=self.conf_threshold)

            # Extract detected labels
            detected_labels = []
            if results and len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    if hasattr(box, 'cls') and hasattr(results[0], 'names'):
                        class_id = int(box.cls[0])
                        label = results[0].names.get(class_id, "")
                        if label:
                            detected_labels.append(label)

            # Store detected labels (thread-safe)
            if detected_labels:
                self.detected_labels.extend(detected_labels)
                self.last_labels = detected_labels

            # Get annotated frame
            annotated_frame = results[0].plot()

            # Convert back to av.VideoFrame
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame


class WebRTCProcessor:
    """Handles WebRTC video streaming and processing for cloud deployment"""

    def __init__(
        self,
        model_manager: ModelManager,
        ai_interpreter: Optional[SignLanguageInterpreter] = None,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize WebRTC processor

        Args:
            model_manager: ModelManager instance for running predictions
            ai_interpreter: Optional AI interpreter for sentence generation
            conf_threshold: Confidence threshold for detections
        """
        self.model_manager = model_manager
        self.ai_interpreter = ai_interpreter
        self.conf_threshold = conf_threshold

        # Initialize session state for frame processing
        if 'webrtc_detected_labels' not in st.session_state:
            st.session_state.webrtc_detected_labels = []
        if 'webrtc_last_update' not in st.session_state:
            st.session_state.webrtc_last_update = time.time()

    def run_webrtc_stream(self, interpretation_placeholder=None):
        """
        Run WebRTC video streaming with live detection

        Args:
            interpretation_placeholder: Optional placeholder for AI interpretations
        """
        st.info("🌐 **WebRTC Live Camera Mode** - Works on all cloud platforms!")
        st.caption("Allow camera access when prompted by your browser")

        # Create WebRTC streamer with processor factory
        ctx = webrtc_streamer(
            key="sign-language-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                },
                "audio": False,
            },
            video_processor_factory=lambda: VideoProcessor(
                model_manager=self.model_manager,
                conf_threshold=self.conf_threshold
            ),
            async_processing=True,
        )

        # Show status
        if ctx.state.playing:
            st.success("✅ Camera is active - Detection running!")

            # Get detected labels from video processor
            if ctx.video_processor:
                detected_labels = ctx.video_processor.detected_labels.copy()
                last_labels = ctx.video_processor.last_labels.copy()

                # Show last detected labels
                if last_labels:
                    st.info(f"🔍 Currently detecting: {', '.join(set(last_labels))}")

                # Add detected signs to AI interpreter (respects 2-second interval)
                if self.ai_interpreter and detected_labels:
                    for label in detected_labels:
                        added = self.ai_interpreter.add_detected_sign(label)
                        if added:
                            logger.info(f"Added sign to buffer: {label}")

                    # Clear the processor's buffer after processing
                    ctx.video_processor.detected_labels.clear()

            # Check if it's time for AI interpretation
            if self.ai_interpreter and interpretation_placeholder:
                if self.ai_interpreter.should_interpret():
                    interpretation = self.ai_interpreter.interpret_signs()
                    if interpretation:
                        # Save to history
                        if 'interpretation_history' not in st.session_state:
                            st.session_state.interpretation_history = []
                        st.session_state.interpretation_history.append(interpretation)

                        # Display the interpretation
                        with interpretation_placeholder.container():
                            if interpretation['arabic']:
                                st.success(f"**Arabic:** {interpretation['arabic']}")
                            if interpretation['english']:
                                st.info(f"**English:** {interpretation['english']}")

                            # Show buffer stats
                            stats = self.ai_interpreter.get_buffer_stats()
                            st.caption(f"Detected {stats['total_words']} signs ({stats['unique_words']} unique)")

                # Show buffer stats
                if self.ai_interpreter:
                    stats = self.ai_interpreter.get_buffer_stats()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Words in Buffer", stats['total_words'])
                    with col2:
                        st.metric("Next Interpretation", f"{stats['time_until_next_interpretation']:.1f}s")
                    with col3:
                        st.metric("Next Buffer Push", f"{stats['time_until_next_push']:.1f}s")
        else:
            st.warning("⏸️ Camera stopped or not started")
            st.info("Click 'START' above to begin detection")

        # Add a button to clear buffer
        if st.button("🗑️ Clear Buffer"):
            if self.ai_interpreter:
                self.ai_interpreter.clear_buffer()
            st.session_state.webrtc_detected_labels = []
            st.success("Buffer cleared!")
            st.rerun()
