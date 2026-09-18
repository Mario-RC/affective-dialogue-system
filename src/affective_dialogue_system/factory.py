"""Construct services from validated configuration without eager model imports."""

from __future__ import annotations

from affective_dialogue_system.config import AppConfig
from affective_dialogue_system.dialogue.engine import AffectiveDialogueEngine, GenerationSettings
from affective_dialogue_system.safety import SafetyFilter
from affective_dialogue_system.strategy import SelectingStrategy
from affective_dialogue_system.strategy.llm_handler import ResponseGenerator
from affective_dialogue_system.strategy.protocols import (
    GlucoseMonitoringProtocol,
    TimeoutProtocol,
    TopicMonitoringProtocol,
)
from affective_dialogue_system.strategy.rule_based_handler import RuleBasedHandler


def create_safety_filter(config: AppConfig) -> SafetyFilter:
    settings = config.safety
    if not settings.enabled:
        return SafetyFilter()
    result = SafetyFilter.default() if settings.local_rules else SafetyFilter()
    if settings.detoxify.enabled:
        from affective_dialogue_system.safety.detoxify_adapter import DetoxifySafetyDetector

        result.detectors.append(
            DetoxifySafetyDetector(
                model_type=settings.detoxify.model_type,
                checkpoint=settings.detoxify.checkpoint,
            )
        )
    if settings.llama_guard.enabled:
        from affective_dialogue_system.safety.llama_guard_adapter import LlamaGuardSafetyDetector

        result.detectors.append(
            LlamaGuardSafetyDetector(
                model_id=settings.llama_guard.model,
                device=config.runtime.device,
                cache_dir=config.runtime.cache_dir,
            )
        )
    return result


def create_dialogue_engine(config: AppConfig) -> AffectiveDialogueEngine:
    settings = config.dialogue
    return AffectiveDialogueEngine(
        model_id=config.models.dialogue_model,
        template=settings.template,
        language=settings.language,
        device=config.runtime.device,
        cache_dir=config.runtime.cache_dir,
        torch_dtype=config.runtime.torch_dtype,
        trust_remote_code=config.runtime.trust_remote_code,
        max_history_turns=settings.max_history_turns,
        human_name=settings.human_name,
        assistant_name=settings.assistant_name,
        generation=GenerationSettings(
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            do_sample=settings.do_sample,
        ),
    )


def create_strategy(
    config: AppConfig, *, primary_generator: ResponseGenerator | None = None
) -> SelectingStrategy:
    settings = config.selecting_strategy
    return SelectingStrategy(
        safety_filter=create_safety_filter(config),
        glucose_protocol=GlucoseMonitoringProtocol(settings.glucose_threshold_mg_dl),
        timeout_protocol=TimeoutProtocol(settings.timeout_seconds),
        topic_protocol=TopicMonitoringProtocol(settings.topic_max_consecutive_turns),
        rule_based_handler=RuleBasedHandler(allow_catch_all=not settings.skip_rule_based_catch_all),
        primary_generator=primary_generator,
        llm_timeout_seconds=settings.llm_timeout_seconds,
    )
