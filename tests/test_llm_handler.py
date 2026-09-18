from threading import Event
from time import monotonic

import pytest

from affective_dialogue_system.strategy import DialogueContext, StrategyResult
from affective_dialogue_system.strategy.llm_handler import LLMHandler
from affective_dialogue_system.toxicity import ToxicityFilter


def test_timeout_returns_while_generator_is_still_running_and_bounds_jobs():
    release, entered = Event(), Event()
    calls = []

    def slow(context):
        calls.append(context)
        entered.set()
        release.wait(2)
        return "late response"

    handler = LLMHandler(ToxicityFilter(), primary_generator=slow, timeout_seconds=0.03)
    try:
        started = monotonic()
        assert handler.handle(DialogueContext()) is None
        assert monotonic() - started < 0.5
        assert entered.is_set()
        for _ in range(5):
            assert handler.handle(DialogueContext()) is None
        assert len(calls) == 1
    finally:
        release.set()
        handler.close(wait_for_running=True)


def test_fast_candidate_does_not_wait_for_slow_loser():
    release = Event()

    def slow(_):
        release.wait(2)
        return "slow"

    handler = LLMHandler(
        ToxicityFilter(), primary_generator=slow, fallback_generator=lambda _: "ready"
    )
    try:
        started = monotonic()
        assert handler.handle(DialogueContext(language="en")).response == "ready"
        assert monotonic() - started < 0.5
    finally:
        release.set()
        handler.close(wait_for_running=True)


def test_translation_is_included_in_the_deadline():
    release = Event()

    def translate(*_):
        release.wait(2)
        return "traducido"

    handler = LLMHandler(
        ToxicityFilter(),
        fallback_generator=lambda _: "ready",
        fallback_translator=translate,
        timeout_seconds=0.03,
    )
    try:
        started = monotonic()
        assert handler.handle(DialogueContext()) is None
        assert monotonic() - started < 0.5
    finally:
        release.set()
        handler.close(wait_for_running=True)


def test_candidate_contexts_are_isolated_from_caller_and_each_other():
    entered, release = Event(), Event()
    context = DialogueContext(metadata={"nested": {"value": "original"}})

    def primary(copy):
        copy.metadata["nested"]["value"] = "changed"
        entered.set()
        release.wait(2)
        return None

    def fallback(copy):
        assert entered.wait(1)
        return copy.metadata["nested"]["value"]

    handler = LLMHandler(ToxicityFilter(), primary, fallback)
    try:
        assert handler.handle(context).response == "original"
        assert context.metadata["nested"]["value"] == "original"
    finally:
        release.set()
        handler.close(wait_for_running=True)


@pytest.mark.parametrize("generated", [None, " ", 42])
def test_empty_or_invalid_candidates_are_discarded(generated):
    handler = LLMHandler(ToxicityFilter(), primary_generator=lambda _: generated)
    try:
        assert handler.handle(DialogueContext()) is None
    finally:
        handler.close(wait_for_running=True)


def test_generator_and_translator_failures_allow_other_candidate():
    def broken(*_):
        raise RuntimeError("failure")

    handler = LLMHandler(
        ToxicityFilter(), primary_generator=broken, fallback_generator=lambda _: "valid"
    )
    try:
        assert handler.handle(DialogueContext()).response == "valid"
    finally:
        handler.close(wait_for_running=True)
    handler = LLMHandler(
        ToxicityFilter(), fallback_generator=lambda _: "valid", fallback_translator=broken
    )
    try:
        assert handler.handle(DialogueContext()) is None
    finally:
        handler.close(wait_for_running=True)


def test_result_preserves_session_end_and_translates_visible_parts():
    result = StrategyResult(
        "bye",
        "custom",
        emotional_response="calm",
        follow_up_question="more?",
        should_end_session=True,
    )
    handler = LLMHandler(
        ToxicityFilter(),
        fallback_generator=lambda _: result,
        fallback_translator=lambda text, _: "ES:" + text,
    )
    try:
        actual = handler.handle(DialogueContext())
        assert actual.should_end_session
        assert actual.response == "ES:bye"
        assert actual.emotional_response == "ES:calm"
        assert actual.follow_up_question == "ES:more?"
    finally:
        handler.close(wait_for_running=True)


def test_unsafe_separate_fields_are_not_delivered():
    handler = LLMHandler(
        ToxicityFilter.default(),
        primary_generator=lambda _: StrategyResult(
            "Hello", "x", follow_up_question="Eres un idiota"
        ),
    )
    try:
        assert handler.handle(DialogueContext()) is None
    finally:
        handler.close(wait_for_running=True)


def test_closed_handler_rejects_new_jobs():
    handler = LLMHandler(ToxicityFilter())
    handler.close()
    with pytest.raises(RuntimeError, match="closed"):
        handler.handle(DialogueContext())
