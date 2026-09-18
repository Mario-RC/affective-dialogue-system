"""Adapters that connect dialogue engines to Selecting Strategy."""

from __future__ import annotations

from dataclasses import dataclass

from affective_dialogue_system.dialogue import AssistantResponse, DialogueTurn, seed_dialogue
from affective_dialogue_system.dialogue.engine import AffectiveDialogueEngine
from affective_dialogue_system.emotion.labels import Emotion
from affective_dialogue_system.strategy.context import DialogueContext


@dataclass
class DialogueEngineGenerator:
    """Use ``AffectiveDialogueEngine`` as an LLM generator hook."""

    engine: AffectiveDialogueEngine
    second_emotion: Emotion = Emotion.NEUTRAL
    seed_turns: list[DialogueTurn] | None = None

    def __call__(self, context: DialogueContext) -> AssistantResponse:
        user_emotion = context.user_emotion or Emotion.NEUTRAL
        turns = list(
            self.seed_turns if self.seed_turns is not None else seed_dialogue(context.language)
        )
        turns.extend(_history_as_turns(context))
        turns.append(
            DialogueTurn(
                user_emotion,
                context.current_message,
                AssistantResponse.placeholder(user_emotion, self.second_emotion),
            )
        )
        return self.engine.generate(turns)


def _history_as_turns(context: DialogueContext) -> list[DialogueTurn]:
    turns: list[DialogueTurn] = []
    for item in context.dialogue_history:
        if isinstance(item, DialogueTurn):
            turns.append(item)
    return turns
