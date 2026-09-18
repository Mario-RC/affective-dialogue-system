"""Legacy-compatible rule-based template handler."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from affective_dialogue_system.strategy.context import DialogueContext, StrategyResult
from affective_dialogue_system.strategy.rule_based_templates import (
    pairs_en,
    pairs_es,
    reflections_en,
    reflections_es,
)


@dataclass
class RuleBasedTemplateEngine:
    pairs: list[tuple[re.Pattern[str], tuple[str, ...] | list[str]]]
    reflections: dict[str, str] = field(default_factory=dict)
    allow_catch_all: bool = False

    def __post_init__(self) -> None:
        for pattern, responses in self.pairs:
            if not responses:
                raise ValueError(f"Template has no responses: {pattern.pattern}")
            for response in responses:
                if any(int(group) > pattern.groups for group in re.findall(r"%(\d+)", response)):
                    raise ValueError(f"Invalid capture group in template: {pattern.pattern}")

    @classmethod
    def for_language(cls, language: str) -> RuleBasedTemplateEngine:
        if language not in {"en", "es"}:
            raise ValueError(f"Unsupported language: {language}")
        if language == "en":
            return cls(
                pairs=[
                    (re.compile(pattern, re.IGNORECASE), responses)
                    for pattern, responses in pairs_en
                ],
                reflections=reflections_en,
            )
        return cls(
            pairs=[
                (re.compile(pattern, re.IGNORECASE), responses) for pattern, responses in pairs_es
            ],
            reflections=reflections_es,
        )

    def respond(self, message: str) -> str | None:
        message = message.strip()
        for pattern, responses in self.pairs:
            if not self.allow_catch_all and pattern.pattern == "(.*)":
                continue
            match = pattern.match(message)
            if not match:
                continue
            response = random.choice(tuple(responses))
            response = self._wildcards(response, match)
            return _fix_punctuation(response)
        return None

    def _wildcards(self, response: str, match: re.Match[str]) -> str:
        return re.sub(
            r"%(\d+)",
            lambda placeholder: self._substitute_reflections(
                match.group(int(placeholder.group(1))) or ""
            ),
            response,
        )

    def _substitute_reflections(self, message: str) -> str:
        if not self.reflections:
            return message
        sorted_keys = sorted(self.reflections.keys(), key=len, reverse=True)
        pattern = re.compile(
            r"\b({})\b".format("|".join(map(re.escape, sorted_keys))), re.IGNORECASE
        )
        return pattern.sub(
            lambda mo: self.reflections.get(mo.group(0).lower(), mo.group(0)), message.lower()
        )


@dataclass
class RuleBasedHandler:
    """Return template responses before falling back to LLMs."""

    allow_catch_all: bool = False
    _engines: dict[str, RuleBasedTemplateEngine] = field(
        default_factory=dict, init=False, repr=False
    )

    def handle(self, context: DialogueContext) -> StrategyResult | None:
        if context.language not in self._engines:
            self._engines[context.language] = RuleBasedTemplateEngine.for_language(context.language)
        engine = self._engines[context.language]
        engine.allow_catch_all = self.allow_catch_all
        response = engine.respond(context.current_message)
        if not response:
            return None
        return StrategyResult.from_parts(
            response=response,
            source="rule_based",
            metadata={"handler": "legacy_rule_based"},
        )


def _fix_punctuation(response: str) -> str:
    replacements = {
        "..": ".",
        "?.": "?",
        "??": "?",
        ".?": "?",
    }
    for source, target in replacements.items():
        if response.endswith(source):
            return response[: -len(source)] + target
    return response
