"""Toxicity filter that combines local rules and optional model detectors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from affective_dialogue_system.toxicity.local_rules import LocalRuleToxicityDetector
from affective_dialogue_system.toxicity.schemas import ToxicityResult


class ToxicityDetector(Protocol):
    def check(self, text: str) -> ToxicityResult:
        """Return a toxicity result for ``text``."""


@dataclass
class ToxicityFilter:
    """Run toxicity detectors in fixed priority order."""

    detectors: list[ToxicityDetector] = field(default_factory=list)

    @classmethod
    def default(cls) -> ToxicityFilter:
        return cls(detectors=[LocalRuleToxicityDetector.from_package_data()])

    def check(self, text: str, *, role: str = "user") -> ToxicityResult:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        for detector in self.detectors:
            check = (
                getattr(detector, "check_response", detector.check)
                if role == "assistant"
                else detector.check
            )
            result = check(text)
            if result.flagged:
                return result
        return ToxicityResult.safe()

    def filter_candidates(self, candidates: Iterable[str]) -> list[str]:
        return [
            candidate
            for candidate in candidates
            if not self.check(candidate, role="assistant").flagged
        ]
