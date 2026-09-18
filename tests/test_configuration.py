from pathlib import Path
from unittest.mock import Mock

import pytest

from affective_dialogue_system.cli import main
from affective_dialogue_system.config import AppConfig, load_config
from affective_dialogue_system.factory import create_strategy
from affective_dialogue_system.strategy import DialogueContext


def test_reference_configuration_matches_defaults():
    path = Path(__file__).resolve().parents[1] / "configs/default.yaml"
    assert load_config(path, environ={}) == AppConfig()


def test_environment_overrides_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "runtime:\n  device: cpu\n  cache_dir: cache\nmodels:\n  dialogue: example/model\n"
    )
    config = load_config(
        config_file,
        environ={
            "ADS_DEVICE": "cuda:1",
            "ADS_CACHE_DIR": "/tmp/model-cache",
            "ADS_SPEAKER_WAV": "/tmp/speaker.wav",
        },
    )
    assert config.runtime.device == "cuda:1"
    assert config.runtime.cache_dir == Path("/tmp/model-cache")
    assert config.models.speaker_wav == Path("/tmp/speaker.wav")
    assert config.models.dialogue_model == "example/model"


@pytest.mark.parametrize(
    "content",
    [
        "unknown: true",
        "runtime: {typo: true}",
        "dialogue: {do_sample: 'false'}",
        "dialogue: {max_new_tokens: true}",
        "dialogue: {temperature: .nan}",
        "dialogue: {temperature: .inf}",
        "dialogue: {top_p: 2}",
        "dialogue: {language: fr}",
        "selecting_strategy: {llm_timeout_seconds: -1}",
        "selecting_strategy: {topic_max_consecutive_turns: 0}",
        "safety: {enabled: true, local_rules: false}",
        "models: {dialogue: ''}",
        "runtime: {cache_dir: 123}",
        "runtime: []",
        "[]",
        "[",
        "false",
        "0",
        "!!python/object/apply:os.system ['false']",
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_config(path, environ={})


def test_configuration_changes_runtime_selection(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("selecting_strategy:\n  topic_max_consecutive_turns: 2\n")
    with create_strategy(load_config(path, environ={})) as strategy:
        assert (
            strategy.select_response(DialogueContext(topic="music", topic_turn_count=2)).source
            == "topic_monitor"
        )


def test_cli_basic_commands(capsys):
    assert main(["interest", "Siento dolor"]) == 0
    assert capsys.readouterr().out.strip() == "100"
    assert main(["safety", "Hola, hablemos de música"]) == 0
    assert capsys.readouterr().out.strip() == "safe"
    assert main(["select", "hola"]) == 0
    assert capsys.readouterr().out.startswith("rule_based\t")


def test_cli_uses_explicit_configuration(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(
        "dialogue: {language: en}\nselecting_strategy: {topic_max_consecutive_turns: 1}\n"
    )
    main(["--config", str(path), "select", "text", "--topic", "music", "--topic-turn-count", "1"])
    assert "same topic" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["chat", "hola", "--emotion", "unknown"],
        ["chat", " "],
        ["select", "hola", "--topic-turn-count", "-1"],
        ["select", "hola", "--glucose-level", "nan"],
        ["select", "hola", "--last-user-input-seconds-ago", "-1"],
    ],
)
def test_cli_invalid_arguments_fail_without_loading_models(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_cli_chat_filters_input_before_loading_model(monkeypatch, capsys):
    engine = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr("affective_dialogue_system.cli.create_dialogue_engine", engine)
    main(["chat", "Eres un idiota"])
    assert "insultos" in capsys.readouterr().out
    engine.assert_not_called()


def test_cli_chat_filters_generated_output(monkeypatch, capsys):
    response = Mock(format=Mock(return_value="Eres un idiota"))
    monkeypatch.setattr(
        "affective_dialogue_system.cli.create_dialogue_engine",
        lambda _: Mock(generate=Mock(return_value=response)),
    )
    main(["chat", "cuéntame algo"])
    assert "insultos" in capsys.readouterr().out


def test_invalid_dtype_is_rejected_before_model_loading(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("runtime: {torch_dtype: nonexistent}")
    with pytest.raises(ValueError, match="torch_dtype"):
        load_config(config, environ={})
