"""Speech-level toxicity filtering and response selection."""

from __future__ import annotations

from dataclasses import dataclass

from affective_dialogue_system.strategy.context import DialogueContext, StrategyResult
from affective_dialogue_system.toxicity import ToxicityFilter, toxicity_response


@dataclass
class ToxicityFilterHandler:
    toxicity_filter: ToxicityFilter

    def handle(self, context: DialogueContext) -> StrategyResult | None:
        result = self.toxicity_filter.check(context.current_message)
        if not result.flagged:
            return None
        return StrategyResult(
            response=toxicity_response(result.category, context.language),
            source="toxicity_filter",
            toxicity=result,
            metadata={"category": result.category.value, "toxicity_source": result.source},
        )
