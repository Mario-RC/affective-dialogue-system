"""Safety-first speech-level toxicity handler."""

from __future__ import annotations

from dataclasses import dataclass

from affective_dialogue_system.safety import SafetyFilter, safety_response
from affective_dialogue_system.strategy.context import DialogueContext, StrategyResult


@dataclass
class ToxicityFilterHandler:
    safety_filter: SafetyFilter

    def handle(self, context: DialogueContext) -> StrategyResult | None:
        result = self.safety_filter.check(context.current_message)
        if not result.flagged:
            return None
        return StrategyResult(
            response=safety_response(result.category, context.language),
            source="toxicity_filter",
            safety=result,
            metadata={"category": result.category.value, "safety_source": result.source},
        )
