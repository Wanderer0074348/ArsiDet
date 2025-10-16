"""Configuration settings for the application"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

# Model settings
MODEL_PATH = PROJECT_ROOT / "models" / "ArabicSignLanguage60.pt"
CONFIDENCE_THRESHOLD = 0.50

# Camera settings
CAMERA_INDEX = 0

# AI Agent settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AI_INTERPRETATION_INTERVAL = 20  # seconds
AI_BUFFER_PUSH_INTERVAL = 2.0  # seconds between adding words to buffer
AI_MODEL = "gpt-4o-mini"  # Using gpt-4o-mini for cost-effectiveness
MAX_WORDS_BUFFER = 100  # Maximum words to keep in buffer

# UI settings
APP_TITLE = "Arabic Sign Language Detection"
PRIMARY_COLOR = "#16a085"
SECONDARY_COLOR = "#1abc9c"
BACKGROUND_COLOR = "#f0f0f0"

# Font settings
PRIMARY_FONT = "Cairo"
SECONDARY_FONT = "Amiri"
TITLE_FONT_SIZE = "40px"
HEADER_FONT_SIZE = "18px"
