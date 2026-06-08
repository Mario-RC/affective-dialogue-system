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
    ) -> None:
        self.dialogue_engine = dialogue_engine
        self.emotion_classifier = emotion_classifier
        self.turns: list[DialogueTurn] = []

    def reply(
        self,
        text: str,
        *,
        user_emotion: Emotion | str | None = None,
        second_emotion: Emotion | str = Emotion.NEUTRAL,
    ) -> AssistantResponse:
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
        self.turns.append(turn)
        response = self.dialogue_engine.generate(self.turns)
        self.turns[-1] = DialogueTurn(user_emotion, text, response)
        return response

