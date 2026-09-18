"""Run documented dialogue examples with an injected engine; no model download."""

import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from affective_dialogue_system.dialogue import AssistantResponse
from affective_dialogue_system.emotion import Emotion

ROOT = Path(__file__).resolve().parents[1]


class ExampleEngine:
    def generate(self, turns):
        latest = turns[-1]
        return AssistantResponse(
            latest.user_emotion,
            "Entiendo cómo te sientes.",
            latest.assistant.second_emotion,
            "Podemos hablarlo con calma.",
            Emotion.NEUTRAL,
            "¿Qué te gustaría explorar?",
        )


def test_readme_dialogue_code_is_executable(monkeypatch, capsys):
    monkeypatch.setattr(
        "affective_dialogue_system.factory.create_dialogue_engine", lambda _: ExampleEngine()
    )
    readme = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    assert len(blocks) == 2
    for block in blocks:
        exec(compile(block, "README.md", "exec"), {})
    output = capsys.readouterr().out
    assert "Selected by: primary_llm" in output
    assert "(FEAR)" in output


@pytest.mark.parametrize("example", ["text_chat.py", "selecting_dialogue.py"])
def test_dialogue_example_scripts(monkeypatch, capsys, example):
    monkeypatch.setattr(
        "affective_dialogue_system.factory.create_dialogue_engine", lambda _: ExampleEngine()
    )
    runpy.run_path(str(ROOT / "examples" / example), run_name="__main__")
    assert "(NEUTRAL)" in capsys.readouterr().out


def test_importing_every_module_does_not_import_optional_models():
    code = """
import importlib, pkgutil, sys
import affective_dialogue_system
for module in pkgutil.walk_packages(affective_dialogue_system.__path__, affective_dialogue_system.__name__ + '.'):
    importlib.import_module(module.name)
assert not {'torch', 'transformers', 'TTS', 'detoxify'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
