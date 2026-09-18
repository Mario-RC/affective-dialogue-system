"""Safety classification data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SafetyCategory(str, Enum):
    SAFE = "safe"
    OBSCENE = "obscene"
    THREAT = "threat"
    INSULT = "insult"
    IDENTITY_ATTACK = "identity_attack"
    SEXUAL_EXPLICIT = "sexual_explicit"
    VIOLENCE_AND_HATE = "violence_and_hate"
    CRIMINAL_PLANNING = "criminal_planning"
    GUNS_AND_ILLEGAL_WEAPONS = "guns_and_illegal_weapons"
    REGULATED_OR_CONTROLLED_SUBSTANCES = "regulated_or_controlled_substances"
    SELF_HARM = "self_harm"
    PROHIBITED_TERM = "palabra_no_adecuada"


@dataclass(frozen=True)
class SafetyResult:
    flagged: bool
    category: SafetyCategory = SafetyCategory.SAFE
    source: str = "none"
    score: float = 0.0
    labels: dict[str, float] = field(default_factory=dict)
    matched_text: str | None = None

    @classmethod
    def safe(cls) -> SafetyResult:
        return cls(flagged=False)

    @classmethod
    def flagged_result(
        cls,
        category: SafetyCategory,
        *,
        source: str,
        score: float = 1.0,
        labels: dict[str, float] | None = None,
        matched_text: str | None = None,
    ) -> SafetyResult:
        return cls(
            flagged=True,
            category=category,
            source=source,
            score=score,
            labels=labels or {},
            matched_text=matched_text,
        )
