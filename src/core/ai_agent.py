"""AI Agent for interpreting detected signs into meaningful sentences"""

import time
from typing import List, Optional, Dict
from collections import deque
from datetime import datetime
import streamlit as st
from openai import OpenAI

from ..utils.config import (
    OPENAI_API_KEY,
    AI_MODEL,
    AI_INTERPRETATION_INTERVAL,
    AI_BUFFER_PUSH_INTERVAL,
    MAX_WORDS_BUFFER,
)


class SignLanguageInterpreter:
    """AI-powered interpreter for converting detected signs to sentences"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = AI_MODEL,
        interpretation_interval: int = AI_INTERPRETATION_INTERVAL,
        buffer_push_interval: float = AI_BUFFER_PUSH_INTERVAL,
    ):
        """
        Initialize the AI interpreter

        Args:
            api_key: OpenAI API key. If None, uses environment variable
            model: OpenAI model to use
            interpretation_interval: Seconds between interpretations
            buffer_push_interval: Seconds between adding words to buffer
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.interpretation_interval = interpretation_interval
        self.buffer_push_interval = buffer_push_interval
        self.client = None

        # Word buffer to store detected signs
        self.word_buffer: deque = deque(maxlen=MAX_WORDS_BUFFER)

        # Timing
        self.last_interpretation_time = time.time()
        self.last_buffer_push_time = time.time()

        # Initialize OpenAI client if API key is available
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {e}")

    def add_detected_sign(self, sign_label: str) -> bool:
        """
        Add a detected sign to the buffer (only if enough time has passed)

        Args:
            sign_label: The label/word detected from sign language

        Returns:
            True if the sign was added to buffer, False otherwise
        """
        if not sign_label or not sign_label.strip():
            return False

        current_time = time.time()
        time_since_last_push = current_time - self.last_buffer_push_time

        # Only add to buffer if enough time has passed
        if time_since_last_push >= self.buffer_push_interval:
            self.word_buffer.append({
                'word': sign_label.strip(),
                'timestamp': datetime.now()
            })
            self.last_buffer_push_time = current_time
            return True

        return False

    def get_buffered_words(self) -> List[str]:
        """
        Get list of words from the buffer

        Returns:
            List of detected words
        """
        return [item['word'] for item in self.word_buffer]

    def should_interpret(self) -> bool:
        """
        Check if enough time has passed for interpretation

        Returns:
            True if interpretation should run
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_interpretation_time
        return time_elapsed >= self.interpretation_interval and len(self.word_buffer) > 0

    def interpret_signs(self) -> Optional[Dict]:
        """
        Use AI to interpret the buffered signs into a meaningful sentence

        Returns:
            Dictionary with 'arabic' and 'english' interpretations or None if interpretation failed
        """
        if not self.client:
            return None

        words = self.get_buffered_words()
        if not words:
            return None

        try:
            # Create a prompt for the AI
            word_sequence = " | ".join(words)

            prompt = f"""You are an AI assistant helping to interpret Arabic Sign Language.

I have detected the following signs/words in sequence (separated by |):
{word_sequence}

Please analyze these detected signs and:
1. Combine them into a coherent, meaningful sentence or phrase
2. Account for possible detection errors or repetitions
3. Consider that signs might be detected multiple times for the same word
4. Provide the interpretation in both Arabic and English

Respond in this exact format:
Arabic: [interpretation in Arabic]
English: [interpretation in English]"""

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in Arabic Sign Language interpretation. You excel at converting sequences of detected signs into coherent, meaningful sentences in both Arabic and English."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=200,
            )

            result = response.choices[0].message.content.strip()

            # Parse the result
            arabic_text = ""
            english_text = ""

            for line in result.split('\n'):
                if line.startswith('Arabic:'):
                    arabic_text = line.replace('Arabic:', '').strip()
                elif line.startswith('English:'):
                    english_text = line.replace('English:', '').strip()

            # Update last interpretation time
            self.last_interpretation_time = time.time()

            return {
                'arabic': arabic_text,
                'english': english_text,
                'timestamp': datetime.now(),
                'detected_words': words.copy()
            }

        except Exception as e:
            st.error(f"AI interpretation error: {e}")
            return None

    def clear_buffer(self):
        """Clear the word buffer"""
        self.word_buffer.clear()

    def get_buffer_stats(self) -> Dict:
        """
        Get statistics about the current buffer

        Returns:
            Dictionary with buffer statistics
        """
        current_time = time.time()
        return {
            'total_words': len(self.word_buffer),
            'unique_words': len(set(self.get_buffered_words())),
            'time_until_next_interpretation': max(0, self.interpretation_interval - (current_time - self.last_interpretation_time)),
            'time_until_next_push': max(0, self.buffer_push_interval - (current_time - self.last_buffer_push_time))
        }


@st.cache_resource
def get_sign_interpreter(api_key: Optional[str] = None) -> SignLanguageInterpreter:
    """
    Get cached sign language interpreter instance

    Args:
        api_key: Optional OpenAI API key

    Returns:
        Cached SignLanguageInterpreter instance
    """
    return SignLanguageInterpreter(api_key=api_key)
