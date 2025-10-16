"""Video processing and camera management"""

import cv2
import numpy as np
import streamlit as st
from typing import Optional, Tuple, List

from .model import ModelManager
from .ai_agent import SignLanguageInterpreter
from ..utils.config import CAMERA_INDEX, CONFIDENCE_THRESHOLD


class VideoProcessor:
    """Handles video capture and frame processing"""

    def __init__(
        self,
        model_manager: ModelManager,
        ai_interpreter: Optional[SignLanguageInterpreter] = None,
        camera_index: int = CAMERA_INDEX,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize video processor

        Args:
            model_manager: ModelManager instance for running predictions
            ai_interpreter: Optional AI interpreter for sentence generation
            camera_index: Index of the camera to use
            conf_threshold: Confidence threshold for detections
        """
        self.model_manager = model_manager
        self.ai_interpreter = ai_interpreter
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.cap: Optional[cv2.VideoCapture] = None

    def open_camera(self) -> bool:
        """
        Open the camera

        Returns:
            True if camera opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()

    def release_camera(self):
        """Release the camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def validate_frame(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Validate a frame has valid dimensions

        Args:
            frame: Frame to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if frame is None:
            return False, "Frame is None"

        if frame.size == 0:
            return False, "Frame is empty"

        if len(frame.shape) < 2 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return False, f"Invalid frame dimensions: {frame.shape}"

        return True, ""

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Process a frame through the model and return annotated result

        Args:
            frame: Input frame from camera

        Returns:
            Tuple of (RGB frame with detections drawn, list of detected labels)
        """
        # Validate input frame
        is_valid, error_msg = self.validate_frame(frame)
        if not is_valid:
            st.warning(f"Input frame validation failed: {error_msg}")
            return None, []

        try:
            # Run model prediction
            results = self.model_manager.predict(frame, conf=self.conf_threshold)

            # Extract detected labels
            detected_labels = []
            if results and len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    if hasattr(box, 'cls') and hasattr(results[0], 'names'):
                        class_id = int(box.cls[0])
                        label = results[0].names.get(class_id, "")
                        if label:
                            detected_labels.append(label)

            # Get annotated frame
            result_frame = results[0].plot()

            # Validate result frame
            is_valid, error_msg = self.validate_frame(result_frame)
            if not is_valid:
                st.warning(f"Result frame validation failed: {error_msg}")
                return None, detected_labels

            # Convert to RGB for display
            result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

            # Final validation
            is_valid, error_msg = self.validate_frame(result_frame_rgb)
            if not is_valid:
                st.warning(f"RGB frame validation failed: {error_msg}")
                return None, detected_labels

            return result_frame_rgb, detected_labels

        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            return None, []

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def run_detection_loop(self, video_placeholder, interpretation_placeholder=None):
        """
        Run the main detection loop

        Args:
            video_placeholder: Streamlit placeholder for video display
            interpretation_placeholder: Optional placeholder for AI interpretations
        """
        if not self.open_camera():
            st.error("Failed to open camera. Please check if your camera is connected and accessible.")
            st.session_state.camera_on = False
            return

        try:
            while st.session_state.camera_on:
                ret, frame = self.read_frame()

                if not ret or frame is None:
                    st.error("Failed to grab frame from camera")
                    break

                # Validate and process frame
                is_valid, error_msg = self.validate_frame(frame)
                if not is_valid:
                    st.error(f"Invalid frame: {error_msg}")
                    break

                # Process frame through model
                result_frame, detected_labels = self.process_frame(frame)

                # Add detected signs to AI interpreter
                if self.ai_interpreter and detected_labels:
                    for label in detected_labels:
                        self.ai_interpreter.add_detected_sign(label)

                # Check if it's time for AI interpretation
                if self.ai_interpreter and interpretation_placeholder and self.ai_interpreter.should_interpret():
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

                # Display result
                if result_frame is not None:
                    video_placeholder.image(result_frame, channels="RGB")

        except Exception as e:
            st.error(f"Error in detection loop: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

        finally:
            self.release_camera()
