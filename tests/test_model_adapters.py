"""Exercise adapter contracts with small fakes, without model downloads or a GPU."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from affective_dialogue_system.asr import WhisperASR
from affective_dialogue_system.dialogue import AffectiveDialogueEngine, DialogueTurn
from affective_dialogue_system.dialogue.engine import GenerationSettings
from affective_dialogue_system.emotion import Emotion, EmotionClassifier
from affective_dialogue_system.generation import EmotionalGPT2Generator
from affective_dialogue_system.runtime import pipeline_device_index
from affective_dialogue_system.tts import CoquiTTSService


class Batch(dict):
    def to(self, _):
        return self


def fake_engine(output):
    engine = AffectiveDialogueEngine.__new__(AffectiveDialogueEngine)
    engine.template, engine.language, engine.device = "gemma", "es", "cpu"
    engine.human_name, engine.assistant_name = "Human", "Ray"
    engine.max_history_turns = 20
    engine.generation = GenerationSettings(do_sample=False)
    engine._torch = SimpleNamespace(inference_mode=nullcontext)
    engine.tokenizer = Mock(return_value=Batch(input_ids=SimpleNamespace(shape=(1, 3))))
    engine.tokenizer.decode.return_value = output
    engine.model = Mock(config=SimpleNamespace(max_position_embeddings=512))
    engine.model.generate.return_value = [[10, 11, 12, 13, 14]]
    return engine


def test_engine_parses_multiline_output_and_only_decodes_continuation():
    engine = fake_engine("(ANGER) Uno.\n(NEUTRAL) Dos.\n(NEUTRAL) ¿Tres?")
    result = engine.generate([DialogueTurn(Emotion.ANGER, "hola")])
    assert result.third_text == "¿Tres?"
    engine.tokenizer.decode.assert_called_once_with([13, 14], skip_special_tokens=True)
    assert engine.tokenizer.call_args.kwargs["add_special_tokens"] is False
    assert "temperature" not in engine.model.generate.call_args.kwargs


def test_engine_rejects_context_overflow_before_generation():
    engine = fake_engine("")
    engine.model.config.max_position_embeddings = 4
    with pytest.raises(ValueError, match="context window"):
        engine.generate([DialogueTurn(Emotion.NEUTRAL, "hola")])
    engine.model.generate.assert_not_called()


def test_engine_rejects_empty_dialogue():
    with pytest.raises(ValueError):
        fake_engine("").generate([])


@pytest.mark.parametrize(
    "settings",
    [{"temperature": 0}, {"temperature": float("nan")}, {"top_p": 1.5}, {"max_new_tokens": 0}],
)
def test_invalid_generation_settings(settings):
    with pytest.raises(ValueError):
        GenerationSettings(**settings)


def test_asr_returns_entire_transcription_not_first_chunk(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()
    asr = WhisperASR.__new__(WhisperASR)
    asr.pipe = Mock(
        return_value={
            "text": "first second",
            "chunks": [{"text": "first", "language": "es"}, {"text": "second", "language": "es"}],
        }
    )
    result = asr.transcribe(audio)
    assert result.text == "first second" and result.language == "es"
    asr.pipe.return_value.pop("text")
    assert asr.transcribe(audio).text == "first second"


def test_asr_missing_file_is_rejected(tmp_path):
    asr = WhisperASR.__new__(WhisperASR)
    asr.pipe = Mock()
    with pytest.raises(FileNotFoundError):
        asr.transcribe(tmp_path / "missing.wav")
    asr.pipe.assert_not_called()


def test_classifier_label_mapping_supports_nonstandard_class_counts():
    classifier = EmotionClassifier.__new__(EmotionClassifier)
    classifier.model = SimpleNamespace(
        config=SimpleNamespace(num_labels=8, id2label={str(i): f"class{i}" for i in range(8)})
    )
    assert classifier._labels()[-1] == "class7"
    classifier.model.config.id2label = {}
    with pytest.raises(ValueError, match="id2label"):
        classifier._labels()


def test_classifier_generic_labels_use_project_order():
    classifier = EmotionClassifier.__new__(EmotionClassifier)
    classifier.model = SimpleNamespace(
        config=SimpleNamespace(num_labels=7, id2label={i: f"LABEL_{i}" for i in range(7)})
    )
    assert classifier._labels() == [emotion.value.lower() for emotion in Emotion]


def test_gpt2_does_not_cut_response_at_last_angle_bracket():
    generator = EmotionalGPT2Generator.__new__(EmotionalGPT2Generator)
    generator.generator = Mock(return_value=[{"generated_text": "The result is > 5."}])
    assert generator.generate("hello") == "The result is > 5."
    assert generator.generator.call_args.kwargs["return_full_text"] is False


@pytest.mark.parametrize(
    "device,expected", [("cpu", -1), ("cuda", 0), ("cuda:2", 2), ("mps", "mps")]
)
def test_pipeline_device_preserves_accelerator(device, expected):
    assert pipeline_device_index(device) == expected


def test_tts_requires_matching_custom_model_and_config():
    with pytest.raises(ValueError, match="together"):
        CoquiTTSService(model_path="model.pth")


def test_tts_validates_reference_audio_before_synthesis(tmp_path):
    service = CoquiTTSService.__new__(CoquiTTSService)
    service.speaker_wav = tmp_path / "missing.wav"
    service.tts = Mock()
    with pytest.raises(FileNotFoundError):
        service.synthesize("Hola")
    service.tts.tts.assert_not_called()
