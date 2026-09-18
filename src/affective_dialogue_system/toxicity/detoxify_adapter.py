"""Optional Detoxify adapter.

This adapter is intentionally lazy so the project can be installed and tested
without the Detoxify package or model checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from affective_dialogue_system.toxicity.schemas import ToxicityCategory, ToxicityResult

DEFAULT_DETOXIFY_THRESHOLDS = {
    "toxicity": 0.2,
    "severe_toxicity": 0.2,
    "obscene": 0.25,
    "identity_attack": 0.1,
    "insult": 0.4,
    "threat": 0.3,
    "sexual_explicit": 0.1,
}

DETOXIFY_CATEGORY_MAP = {
    "obscene": ToxicityCategory.OBSCENE,
    "identity_attack": ToxicityCategory.IDENTITY_ATTACK,
    "insult": ToxicityCategory.INSULT,
    "threat": ToxicityCategory.THREAT,
    "sexual_explicit": ToxicityCategory.SEXUAL_EXPLICIT,
    "severe_toxicity": ToxicityCategory.VIOLENCE_AND_HATE,
    "toxicity": ToxicityCategory.PROHIBITED_TERM,
}


@dataclass
class DetoxifyToxicityDetector:
    model_type: str = "multilingual"
    checkpoint: str | None = None
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DETOXIFY_THRESHOLDS))

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in self.thresholds.values()
        ):
            raise ValueError("Detoxify thresholds must be finite and between 0 and 1")
        from detoxify import Detoxify

        self.model = Detoxify(model_type=self.model_type, checkpoint=self.checkpoint)

    def check(self, text: str) -> ToxicityResult:
        raw = self.model.predict([text])
        labels = {label: float(values[0]) for label, values in raw.items()}
        if not labels or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in labels.values()
        ):
            raise ValueError("Detoxify returned invalid scores")
        exceeded = [
            (label, value)
            for label, value in labels.items()
            if value > self.thresholds.get(label, 1.0)
        ]
        if not exceeded:
            return ToxicityResult.safe()

        label, score = max(exceeded, key=lambda item: item[1])
        category = DETOXIFY_CATEGORY_MAP.get(label, ToxicityCategory.PROHIBITED_TERM)
        return ToxicityResult.flagged_result(
            category,
            source="detoxify",
            score=score,
            labels=labels,
            matched_text=label,
        )
