"""Regression coverage for dialogue state, response formats, and template handling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from affective_dialogue_system.dialogue import (
    DialogueTurn,
    build_prompt,
    parse_emotional_response,
)
from affective_dialogue_system.dialogue.parser import ParseFallback
from affective_dialogue_system.emotion import Emotion
from affective_dialogue_system.interest import InterestScorer, score_interest
from affective_dialogue_system.pipeline import AffectiveDialogueSystem
from affective_dialogue_system.strategy import DialogueContext, SelectingStrategy, StrategyResult
from affective_dialogue_system.strategy.protocols import GlucoseMonitoringProtocol
from affective_dialogue_system.strategy.rule_based_handler import RuleBasedTemplateEngine

VALID = "(ANGER) Entiendo.\n(SADNESS) Es difícil (a veces).\n(NEUTRAL) ¿Qué ocurrió?"


@pytest.mark.parametrize(
    "text",
    [
        "(ANGER) (SADNESS) Algo. (NEUTRAL) Pregunta?",
        "(ANGER) Uno. (SADNESS) Dos. (ANGER) Tres.",
        "(ANGER) Uno. (SADNESS) Dos. (NEUTRAL) Tres. (FEAR) Cuatro.",
        "Prefacio. (ANGER) Uno. (SADNESS) Dos. (NEUTRAL) Tres.",
        "(UNKNOWN) Uno. (SADNESS) Dos. (NEUTRAL) Tres.",
        "",
        "texto sin estructura",
    ],
)
def test_parser_rejects_invalid_structure(text):
    with pytest.raises(ValueError):
        parse_emotional_response(text)


def test_parser_preserves_punctuation_and_parenthetical_text():
    response = parse_emotional_response(VALID)
    assert response.first_text == "Entiendo."
    assert response.second_text == "Es difícil (a veces)."
    assert response.third_text == "¿Qué ocurrió?"


def test_wrong_emotions_use_requested_fallback():
    response = parse_emotional_response(
        VALID,
        expected_emotions=(Emotion.FEAR, Emotion.NEUTRAL),
        fallback=ParseFallback(Emotion.FEAR, Emotion.NEUTRAL, "es"),
    )
    assert response.first_emotion == Emotion.FEAR
    assert response.second_emotion == Emotion.NEUTRAL


def test_failed_generation_does_not_pollute_history():
    engine = SimpleNamespace(generate=Mock(side_effect=RuntimeError("unavailable")))
    system = AffectiveDialogueSystem(engine)
    with pytest.raises(RuntimeError):
        system.reply("hola")
    assert system.turns == []


def test_successful_history_is_bounded_and_passed_to_generator():
    response = parse_emotional_response(VALID)
    engine = SimpleNamespace(generate=Mock(return_value=response))
    system = AffectiveDialogueSystem(engine, max_history_turns=2)
    for text in ["uno", "dos", "tres"]:
        system.reply(text, user_emotion="ANGER")
    assert [turn.user_text for turn in system.turns] == ["dos", "tres"]
    assert len(engine.generate.call_args.args[0]) == 2
    assert all(turn.assistant is response for turn in system.turns)


def test_missing_history_response_is_rejected():
    with pytest.raises(ValueError, match="Historical"):
        build_prompt(
            [DialogueTurn(Emotion.NEUTRAL, "primero"), DialogueTurn(Emotion.NEUTRAL, "segundo")]
        )


@pytest.mark.parametrize(
    "template,start",
    [
        ("gemma", "<bos><start_of_turn>user\n"),
        ("llama3", "<|begin_of_text|><|start_header_id|>system"),
    ],
)
def test_prompt_starts_in_supported_role(template, start):
    assert build_prompt([DialogueTurn(Emotion.NEUTRAL, "hola")], template=template).startswith(
        start
    )


def test_protocol_priority_is_short_circuited():
    first = Mock(handle=Mock(return_value=StrategyResult("stop", "first")))
    later = Mock(handle=Mock(side_effect=AssertionError("must not run")))
    with SelectingStrategy(
        glucose_protocol=first, timeout_protocol=later, topic_protocol=later
    ) as strategy:
        assert strategy.select_response(DialogueContext()).source == "first"
    later.handle.assert_not_called()


@pytest.mark.parametrize("message", ["no lo sé", "espera 5 minutos", "-50", "0", "6 mmol/l"])
def test_protocol_does_not_reuse_old_or_ambiguous_measurement(message):
    context = DialogueContext(
        current_message=message,
        glucose_level=60,
        metadata={"glucose_protocol_state": "ask_first_measure"},
    )
    result = GlucoseMonitoringProtocol().handle(context)
    assert result.metadata["next_state"] == "ask_first_measure"
    assert "glucose_initial_level" not in context.metadata


def test_recovery_updates_context_and_does_not_retrigger_protocol():
    context = DialogueContext(
        current_message="72,5 mg/dL",
        glucose_level=60,
        metadata={"glucose_protocol_state": "ask_second_measure"},
    )
    protocol = GlucoseMonitoringProtocol()
    assert protocol.handle(context).metadata["completed"]
    assert context.glucose_level == 72.5
    context.current_message = "hola"
    assert protocol.handle(context) is None


@pytest.mark.parametrize("language", ["en", "es"])
def test_all_template_placeholders_are_valid(language):
    assert RuleBasedTemplateEngine.for_language(language).pairs


def test_percent_in_user_capture_is_not_interpreted_as_template():
    engine = RuleBasedTemplateEngine.for_language("en")
    assert "50%" in engine.respond("I need 50% more time")


def test_spanish_reflections_support_accents():
    engine = RuleBasedTemplateEngine.for_language("es")
    assert engine._substitute_reflections("tú eres amable") == "yo soy amable"
    assert engine.respond("hola")
    assert engine.respond("holanda") is None


def test_optional_capture_group_can_be_empty():
    engine = RuleBasedTemplateEngine.for_language("es")
    assert engine.respond("de qué color es mi coche rojo")


@pytest.mark.parametrize("text", ["avión", "aviones", "lugares", "viajar"])
def test_interest_handles_accents_and_word_forms(text):
    assert score_interest(text) == 20


def test_interest_does_not_match_arbitrary_substrings():
    assert score_interest("luz") == 0
    assert score_interest("ni") == 0


def test_interest_instances_do_not_share_mutable_keywords():
    first, second = InterestScorer(), InterestScorer()
    first.ranges[100].add("inventado")
    assert second.score("inventado") == 0
