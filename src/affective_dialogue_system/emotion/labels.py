"""Shared emotion labels."""

from __future__ import annotations

from enum import Enum


class Emotion(str, Enum):
    ANGER = "ANGER"
    DISGUST = "DISGUST"
    FEAR = "FEAR"
    HAPPINESS = "HAPPINESS"
    NEUTRAL = "NEUTRAL"
    SADNESS = "SADNESS"
    SURPRISE = "SURPRISE"

    @classmethod
    def normalize(cls, value: str | "Emotion") -> "Emotion":
        if isinstance(value, Emotion):
            return value
        normalized = value.strip().replace(" ", "_").upper()
        return cls(normalized)


EMOTION_LABELS = tuple(emotion.value.lower() for emotion in Emotion)

