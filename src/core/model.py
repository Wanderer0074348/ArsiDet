"""Model loading and management"""

import torch
from ultralytics import YOLO
import streamlit as st
from pathlib import Path
from typing import Optional

from ..utils.config import MODEL_PATH


class ModelManager:
    """Manages YOLO model loading and inference"""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the model manager

        Args:
            model_path: Path to the YOLO model file. If None, uses default from config
        """
        self.model_path = model_path or MODEL_PATH
        self._model = None
        self._device = None

    @property
    def device(self) -> torch.device:
        """Get the device (CPU/CUDA) for model inference"""
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def load_model(self) -> YOLO:
        """
        Load the YOLO model with caching

        Returns:
            Loaded YOLO model
        """
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. "
                    f"Please ensure the model is in the correct location."
                )

            self._model = YOLO(str(self.model_path))
            self._model.to(self.device)

        return self._model

    def predict(self, frame, conf: float = 0.5):
        """
        Run prediction on a frame

        Args:
            frame: Input image frame
            conf: Confidence threshold for predictions

        Returns:
            YOLO results object
        """
        model = self.load_model()
        return model(frame, conf=conf)


@st.cache_resource
def get_model_manager(model_path: Optional[str] = None) -> ModelManager:
    """
    Get cached model manager instance

    Args:
        model_path: Optional path to model file

    Returns:
        Cached ModelManager instance
    """
    path = Path(model_path) if model_path else None
    return ModelManager(model_path=path)
