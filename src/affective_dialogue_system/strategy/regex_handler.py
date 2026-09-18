"""RegEx handler for fast deterministic dialogue responses."""

from __future__ import annotations

import re
from dataclasses import dataclass

from affective_dialogue_system.strategy.context import DialogueContext, StrategyResult
from affective_dialogue_system.strategy.responses import REGEX_RESPONSES_EN, REGEX_RESPONSES_ES


@dataclass(frozen=True)
class RegexRule:
    name: str
    language: str
    pattern: str


DEFAULT_REGEX_RULES = (
    RegexRule("greeting", "es", r"^\s*(hola|buenas|buenos dias|buenas tardes)\b"),
    RegexRule("greeting", "en", r"^\s*(hi|hello|hey|good morning|good afternoon)\b"),
    RegexRule("identity", "es", r"\b(quien eres|como te llamas|que eres)\b"),
    RegexRule("identity", "en", r"\b(who are you|what are you|what is your name)\b"),
    RegexRule("health", "es", r"\b(glucosa|mareo|dolor|diabetes|me encuentro mal)\b"),
    RegexRule("health", "en", r"\b(glucose|dizzy|pain|diabetes|i feel sick)\b"),
)


@dataclass
class RegexHandler:
    rules: tuple[RegexRule, ...] = DEFAULT_REGEX_RULES

    def handle(self, context: DialogueContext) -> StrategyResult | None:
        text = context.current_message.lower()
        responses = REGEX_RESPONSES_EN if context.language == "en" else REGEX_RESPONSES_ES
        for rule in self.rules:
            if rule.language != context.language:
                continue
            if re.search(rule.pattern, text):
                return StrategyResult(
                    response=responses[rule.name],
                    source="regex",
                    metadata={"rule": rule.name},
                )
        return None
