"""Bounded parallel generation with a deadline and isolated request contexts."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import Lock
from time import monotonic

from affective_dialogue_system.dialogue.schemas import AssistantResponse
from affective_dialogue_system.strategy.context import DialogueContext, StrategyResult
from affective_dialogue_system.strategy.responses import fallback_response
from affective_dialogue_system.toxicity import ToxicityFilter

logger = logging.getLogger(__name__)
GeneratedResponse = str | AssistantResponse | StrategyResult
ResponseGenerator = Callable[[DialogueContext], GeneratedResponse | None]
Translator = Callable[[str, str], str]


@dataclass
class LLMHandler:
    """Race at most two candidates, including translation and output checks.

    A running Python thread cannot be forcibly cancelled. Each source may have
    only one outstanding job; subsequent calls skip busy sources. Late results
    are discarded, and each job receives an independent copy of the context.
    Call ``close`` when the owning conversation/service is finished.
    """

    toxicity_filter: ToxicityFilter
    primary_generator: ResponseGenerator | None = None
    fallback_generator: ResponseGenerator | None = None
    fallback_translator: Translator | None = None
    timeout_seconds: float = 2.0
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _inflight: dict[str, Future[StrategyResult | None]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

    def handle(self, context: DialogueContext) -> StrategyResult | None:
        deadline = monotonic() + self.timeout_seconds
        tasks: dict[Future[StrategyResult | None], str] = {}
        with self._lock:
            if self._closed:
                raise RuntimeError("LLMHandler is closed")
            for source, generator in (
                ("primary_llm", self.primary_generator),
                ("fallback_llm", self.fallback_generator),
            ):
                previous = self._inflight.get(source)
                if generator is None or (previous is not None and not previous.done()):
                    continue
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=2, thread_name_prefix="affective-dialogue"
                    )
                future = self._executor.submit(
                    self._generate_checked,
                    generator,
                    deepcopy(context),
                    source,
                    self.fallback_translator if source == "fallback_llm" else None,
                )
                self._inflight[source] = future
                tasks[future] = source

        pending = set(tasks)
        try:
            while pending:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                # Prefer primary only if both valid results are ready together.
                for future in sorted(done, key=lambda item: tasks[item] != "primary_llm"):
                    if future.cancelled():
                        continue
                    result = future.result()
                    if result is not None:
                        return result
            return None
        finally:
            for future in pending:
                future.cancel()

    def _generate_checked(
        self,
        generator: ResponseGenerator,
        context: DialogueContext,
        source: str,
        translator: Translator | None,
    ) -> StrategyResult | None:
        try:
            generated = generator(context)
            if generated is None:
                return None
            result = _coerce_strategy_result(generated, source)
            if translator and context.language == "es":
                result = replace(
                    result,
                    response=translator(result.response, context.language),
                    emotional_response=(
                        translator(result.emotional_response, context.language)
                        if result.emotional_response
                        else None
                    ),
                    follow_up_question=(
                        translator(result.follow_up_question, context.language)
                        if result.follow_up_question
                        else None
                    ),
                )
            if not isinstance(result.response, str) or not result.response.strip():
                return None
            toxicity = self.toxicity_filter.check(result.response, role="assistant")
            if toxicity.flagged:
                return None
            for part in (result.emotional_response, result.follow_up_question):
                if part and self.toxicity_filter.check(part, role="assistant").flagged:
                    return None
            return replace(result, source=source, toxicity=toxicity)
        except Exception as exc:
            # Avoid logging user messages or raw model output.
            logger.warning("%s candidate failed (%s)", source, type(exc).__name__)
            return None

    def close(self, *, wait_for_running: bool = False) -> None:
        """Stop accepting jobs; optionally wait for running generators to finish."""
        with self._lock:
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown(wait=wait_for_running, cancel_futures=True)


def no_answer_result(context: DialogueContext) -> StrategyResult:
    return StrategyResult(response=fallback_response(context.language), source="fallback")


def _coerce_strategy_result(generated: GeneratedResponse, source: str) -> StrategyResult:
    if isinstance(generated, StrategyResult):
        return generated
    if isinstance(generated, AssistantResponse):
        emotional_response = " ".join(
            part.strip() for part in (generated.first_text, generated.second_text) if part.strip()
        )
        return StrategyResult.from_parts(
            response=generated.format(),
            source=source,
            emotional_response=emotional_response or None,
            follow_up_question=generated.third_text or None,
            metadata={"structured": True},
        )
    if not isinstance(generated, str):
        raise TypeError("Generators must return text, AssistantResponse, StrategyResult, or None")
    return StrategyResult.from_parts(response=generated, source=source)
