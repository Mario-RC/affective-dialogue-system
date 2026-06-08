"""Parse model outputs with explicit emotion tags."""

from __future__ import annotations

from dataclasses import dataclass
import re

from affective_dialogue_system.dialogue.schemas import AssistantResponse
from affective_dialogue_system.emotion.labels import Emotion


_TAG_RE = re.compile(r"\(([^)]+)\)")


@dataclass(frozen=True)
class ParseFallback:
    first_emotion: Emotion = Emotion.NEUTRAL
    second_emotion: Emotion = Emotion.NEUTRAL
    language: str = "es"


def parse_emotional_response(
    text: str,
    *,
    fallback: ParseFallback | None = None,
) -> AssistantResponse:
    """Parse ``(EMOTION) text`` triples from generated text.

    Raises ValueError if the output cannot be parsed and no fallback is provided.
    """

    matches = list(_TAG_RE.finditer(text))
    if len(matches) >= 3:
        try:
            first_emotion = Emotion.normalize(matches[0].group(1))
            second_emotion = Emotion.normalize(matches[1].group(1))
            third_emotion = Emotion.normalize(matches[2].group(1))
            first_text = _segment(text, matches, 0)
            second_text = _segment(text, matches, 1)
            third_text = _segment(text, matches, 2)
            return AssistantResponse(
                first_emotion,
                first_text,
                second_emotion,
                second_text,
                third_emotion,
                third_text,
            )
        except (IndexError, ValueError):
            pass

    if fallback is None:
        raise ValueError(f"Could not parse emotional response: {text!r}")
    return _fallback_response(fallback)


def _segment(text: str, matches: list[re.Match[str]], index: int) -> str:
    start = matches[index].end()
    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
    return text[start:end].strip().strip(".").strip()


def _fallback_response(fallback: ParseFallback) -> AssistantResponse:
    if fallback.language == "en":
        return AssistantResponse(
            fallback.first_emotion,
            "I am sorry.",
            fallback.second_emotion,
            "I did not understand that clearly.",
            Emotion.NEUTRAL,
            "Could you tell me a little more about what you mean?",
        )
    return AssistantResponse(
        fallback.first_emotion,
        "Lo siento.",
        fallback.second_emotion,
        "No te he entendido con claridad.",
        Emotion.NEUTRAL,
        "Podrias contarme un poco mas sobre lo que quieres decir?",
    )

