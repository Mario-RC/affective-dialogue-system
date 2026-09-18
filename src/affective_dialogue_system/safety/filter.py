"""Safety filter that combines local rules and optional model detectors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from affective_dialogue_system.safety.local_rules import LocalRuleSafetyDetector
from affective_dialogue_system.safety.schemas import SafetyResult


class SafetyDetector(Protocol):
    def check(self, text: str) -> SafetyResult:
        """Return a safety result for ``text``."""


@dataclass
class SafetyFilter:
    """Run safety detectors in fixed priority order."""

    detectors: list[SafetyDetector] = field(default_factory=list)

    @classmethod
    def default(cls) -> SafetyFilter:
        return cls(detectors=[LocalRuleSafetyDetector.from_package_data()])

    def check(self, text: str, *, role: str = "user") -> SafetyResult:
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
        return SafetyResult.safe()

    def filter_candidates(self, candidates: Iterable[str]) -> list[str]:
        return [
            candidate
            for candidate in candidates
            if not self.check(candidate, role="assistant").flagged
        ]
