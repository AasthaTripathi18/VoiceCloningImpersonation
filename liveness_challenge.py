"""
liveness_challenge.py

The "Verify" layer: issues a random spoken-phrase challenge and checks
whether the caller actually said it (ASR transcription + fuzzy match).
(Unchanged from the training notebook.)
"""

import random
import string
from difflib import SequenceMatcher

from faster_whisper import WhisperModel

WORD_POOL = [
    "orange", "river", "tiger", "pencil", "cloud", "guitar", "window",
    "basket", "silver", "mountain", "candle", "pepper", "jacket", "engine",
]
DIGIT_POOL = list(string.digits)


def generate_random_phrase(num_words: int = 2, num_digits: int = 3) -> str:
    words = random.sample(WORD_POOL, k=num_words)
    digits = [random.choice(DIGIT_POOL) for _ in range(num_digits)]
    parts = words + digits
    random.shuffle(parts)
    return " ".join(parts)


class LivenessVerifier:
    def __init__(self, whisper_model_size: str = "base", device: str = "cpu"):
        self.asr_model = WhisperModel(whisper_model_size, device=device, compute_type="int8")

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.asr_model.transcribe(audio_path, language="en")
        return " ".join(seg.text for seg in segments).strip().lower()

    def verify(self, audio_path: str, expected_phrase: str, similarity_threshold: float = 0.75) -> dict:
        transcribed = self.transcribe(audio_path)
        expected_norm = expected_phrase.strip().lower()

        similarity = SequenceMatcher(None, transcribed, expected_norm).ratio()
        passed = similarity >= similarity_threshold

        return {
            "expected_phrase": expected_phrase,
            "transcribed_text": transcribed,
            "similarity": similarity,
            "passed": passed,
        }
