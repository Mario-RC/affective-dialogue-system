"""Central Selecting Strategy implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from affective_dialogue_system.safety import SafetyFilter
from affective_dialogue_system.strategy.context import DialogueContext, StrategyResult
from affective_dialogue_system.strategy.llm_handler import (
    LLMHandler,
    ResponseGenerator,
    Translator,
    no_answer_result,
)
from affective_dialogue_system.strategy.protocols import (
    GlucoseMonitoringProtocol,
    TimeoutProtocol,
    TopicMonitoringProtocol,
)
from affective_dialogue_system.strategy.rule_based_handler import RuleBasedHandler
from affective_dialogue_system.strategy.toxicity_handler import ToxicityFilterHandler


@dataclass
class SelectingStrategy:
    """Orchestrate protocol-level and speech-level response selection.

    Priority order:
    1. Glucose monitoring protocol.
    2. Timeout prompt for user silence.
    3. Topic monitoring.
    4. Toxicity filter on user input.
    5. Rule-based deterministic template responses.
    6. Parallel LLM generation with output safety filtering.
    7. Deterministic fallback response.
    """

    safety_filter: SafetyFilter = field(default_factory=SafetyFilter.default)
    glucose_protocol: GlucoseMonitoringProtocol = field(default_factory=GlucoseMonitoringProtocol)
    timeout_protocol: TimeoutProtocol = field(default_factory=TimeoutProtocol)
    topic_protocol: TopicMonitoringProtocol = field(default_factory=TopicMonitoringProtocol)
    rule_based_handler: RuleBasedHandler = field(default_factory=RuleBasedHandler)
    primary_generator: ResponseGenerator | None = None
    fallback_generator: ResponseGenerator | None = None
    fallback_translator: Translator | None = None
    llm_timeout_seconds: float = 2.0
    _llm_handler: LLMHandler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._llm_handler = LLMHandler(
            safety_filter=self.safety_filter,
            primary_generator=self.primary_generator,
            fallback_generator=self.fallback_generator,
            fallback_translator=self.fallback_translator,
            timeout_seconds=self.llm_timeout_seconds,
        )

    def close(self, *, wait_for_running: bool = False) -> None:
        self._llm_handler.close(wait_for_running=wait_for_running)

    def __enter__(self) -> SelectingStrategy:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def select_response(
        self,
        context: DialogueContext,
        *,
        now: datetime | None = None,
    ) -> StrategyResult:
        result = self.glucose_protocol.handle(context)
        if result is not None:
            return result
        result = self.timeout_protocol.handle(context, now=now)
        if result is not None:
            return result
        result = self.topic_protocol.handle(context)
        if result is not None:
            return result

        toxicity = ToxicityFilterHandler(self.safety_filter).handle(context)
        if toxicity:
            return toxicity

        rule_based = self.rule_based_handler.handle(context)
        if rule_based:
            safety = self.safety_filter.check(rule_based.response, role="assistant")
            if not safety.flagged:
                return StrategyResult.from_parts(
                    response=rule_based.response,
                    source=rule_based.source,
                    safety=safety,
                    metadata=rule_based.metadata,
                )

        llm = self._llm_handler.handle(context)
        if llm:
            return llm

        return no_answer_result(context)
