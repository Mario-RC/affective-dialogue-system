"""Shared dialogue context for Selecting Strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from affective_dialogue_system.emotion.labels import Emotion
from affective_dialogue_system.toxicity.schemas import ToxicityResult


@dataclass
class DialogueContext:
    """Centralized memory used by the Selecting Strategy."""

    current_message: str = ""
    user_emotion: Emotion | None = None
    robot_emotion: Emotion = Emotion.NEUTRAL
    language: str = "es"
    topic: str | None = None
    topic_turn_count: int = 0
    turn_index: int = 0
    glucose_level: float | None = None
    last_user_input_at: datetime | None = None
    dialogue_history: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.language not in {"en", "es"}:
            raise ValueError("language must be en or es")
        if self.topic_turn_count < 0 or self.turn_index < 0:
            raise ValueError("Turn counts must not be negative")
        if self.glucose_level is not None and (
            not math.isfinite(self.glucose_level) or self.glucose_level <= 0
        ):
            raise ValueError("glucose_level must be finite and positive")
        if self.user_emotion is not None:
            self.user_emotion = Emotion.normalize(self.user_emotion)
        self.robot_emotion = Emotion.normalize(self.robot_emotion)


@dataclass(frozen=True)
class StrategyResult:
    response: str
    source: str
    emotional_response: str | None = None
    follow_up_question: str | None = None
    should_end_session: bool = False
    toxicity: ToxicityResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        *,
        response: str,
        source: str,
        emotional_response: str | None = None,
        follow_up_question: str | None = None,
        should_end_session: bool = False,
        toxicity: ToxicityResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyResult:
        return cls(
            response=response,
            source=source,
            emotional_response=emotional_response,
            follow_up_question=follow_up_question,
            should_end_session=should_end_session,
            toxicity=toxicity,
            metadata=metadata or {},
        )
