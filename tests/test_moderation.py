import pytest

from affective_dialogue_system.safety import SafetyCategory, SafetyFilter, SafetyResult
from affective_dialogue_system.safety.llama_guard_adapter import parse_llama_guard_output


@pytest.mark.parametrize("output", ["safe", " SAFE \n"])
def test_explicit_guard_safe_result(output):
    assert not parse_llama_guard_output(output).flagged


@pytest.mark.parametrize(
    "output", ["", "unknown", "unsafe", "unsafe\nS99", "unsafe\nO7", "safe\nS1", "safe or unsafe"]
)
def test_malformed_guard_output_never_passes_as_safe(output):
    assert parse_llama_guard_output(output).flagged


@pytest.mark.parametrize(
    "code,category",
    [
        ("S1", SafetyCategory.VIOLENCE_AND_HATE),
        ("S2", SafetyCategory.CRIMINAL_PLANNING),
        ("S4", SafetyCategory.SEXUAL_EXPLICIT),
        ("S8", SafetyCategory.GUNS_AND_ILLEGAL_WEAPONS),
        ("S9", SafetyCategory.IDENTITY_ATTACK),
        ("S10", SafetyCategory.SELF_HARM),
        ("S11", SafetyCategory.SEXUAL_EXPLICIT),
    ],
)
def test_guard_2_taxonomy(code, category):
    result = parse_llama_guard_output(f"unsafe\n{code}")
    assert result.flagged and result.category == category


def test_multiple_guard_categories_remain_flagged():
    result = parse_llama_guard_output("unsafe\nS1,S10")
    assert result.flagged and result.matched_text == "S1,S10"


@pytest.mark.parametrize("title", ["Suicide Squad", "To Kill a Mockingbird", "Sex Education"])
def test_allowlisted_titles_do_not_trigger_category_patterns(title):
    assert not SafetyFilter.default().check(f"I watched {title}").flagged


def test_allowlist_does_not_hide_a_separate_threat():
    assert (
        SafetyFilter.default().check("Suicide Squad. Te voy a matar.").category
        == SafetyCategory.THREAT
    )


def test_assistant_moderation_uses_response_adapter():
    class Detector:
        def check(self, _):
            return SafetyResult.safe()

        def check_response(self, _):
            return SafetyResult.flagged_result(SafetyCategory.THREAT, source="response")

    safety = SafetyFilter([Detector()])
    assert not safety.check("text").flagged
    assert safety.check("text", role="assistant").flagged


def test_detoxify_preserves_model_type_with_custom_checkpoint(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from affective_dialogue_system.safety.detoxify_adapter import DetoxifySafetyDetector

    constructor = Mock()
    monkeypatch.setitem(sys.modules, "detoxify", SimpleNamespace(Detoxify=constructor))
    DetoxifySafetyDetector(model_type="multilingual", checkpoint="custom.ckpt")
    constructor.assert_called_once_with(model_type="multilingual", checkpoint="custom.ckpt")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 2])
def test_invalid_detoxify_scores_do_not_pass_as_safe(score):
    from types import SimpleNamespace

    from affective_dialogue_system.safety.detoxify_adapter import DetoxifySafetyDetector

    detector = DetoxifySafetyDetector.__new__(DetoxifySafetyDetector)
    detector.model = SimpleNamespace(predict=lambda _: {"toxicity": [score]})
    with pytest.raises(ValueError, match="scores"):
        detector.check("hello")
