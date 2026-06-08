"""Dialogue data structures."""

from __future__ import annotations

from dataclasses import dataclass

from affective_dialogue_system.emotion.labels import Emotion


@dataclass(frozen=True)
class AssistantResponse:
    first_emotion: Emotion
    first_text: str
    second_emotion: Emotion
    second_text: str
    third_emotion: Emotion
    third_text: str

    @classmethod
    def placeholder(
        cls,
        user_emotion: Emotion,
        second_emotion: Emotion = Emotion.NEUTRAL,
    ) -> "AssistantResponse":
        return cls(user_emotion, "", second_emotion, "", Emotion.NEUTRAL, "")

    def format(self) -> str:
        return (
            f"({self.first_emotion.value}) {self.first_text} "
            f"({self.second_emotion.value}) {self.second_text} "
            f"({self.third_emotion.value}) {self.third_text}"
        ).strip()


@dataclass(frozen=True)
class DialogueTurn:
    user_emotion: Emotion
    user_text: str
    assistant: AssistantResponse | None = None

    @classmethod
    def from_values(
        cls,
        user_emotion: str | Emotion,
        user_text: str,
        assistant: AssistantResponse | None = None,
    ) -> "DialogueTurn":
        return cls(Emotion.normalize(user_emotion), user_text, assistant)

