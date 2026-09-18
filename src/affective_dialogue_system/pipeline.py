"""High-level orchestration for the affective dialogue system."""

from __future__ import annotations

from affective_dialogue_system.dialogue.engine import AffectiveDialogueEngine
from affective_dialogue_system.dialogue.schemas import AssistantResponse, DialogueTurn
from affective_dialogue_system.emotion.classifier import EmotionClassifier
from affective_dialogue_system.emotion.labels import Emotion


class AffectiveDialogueSystem:
    """Combine emotion classification and dialogue generation."""

    def __init__(
        self,
        dialogue_engine: AffectiveDialogueEngine,
        emotion_classifier: EmotionClassifier | None = None,
        *,
        max_history_turns: int = 20,
    ) -> None:
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be positive")
        self.dialogue_engine = dialogue_engine
        self.emotion_classifier = emotion_classifier
        self.max_history_turns = max_history_turns
        self.turns: list[DialogueTurn] = []

    def reply(
        self,
        text: str,
        *,
        user_emotion: Emotion | str | None = None,
        second_emotion: Emotion | str = Emotion.NEUTRAL,
    ) -> AssistantResponse:
        if not text.strip():
            raise ValueError("text must not be empty")
        if user_emotion is None:
            if self.emotion_classifier is None:
                user_emotion = Emotion.NEUTRAL
            else:
                user_emotion = Emotion.normalize(self.emotion_classifier.predict(text).label)
        else:
            user_emotion = Emotion.normalize(user_emotion)

        turn = DialogueTurn(
            user_emotion,
            text,
            AssistantResponse.placeholder(user_emotion, Emotion.normalize(second_emotion)),
        )
        history = [*self.turns, turn][-self.max_history_turns :]
        response = self.dialogue_engine.generate(history)
        self.turns = [*self.turns, DialogueTurn(user_emotion, text, response)][
            -self.max_history_turns :
        ]
        return response
