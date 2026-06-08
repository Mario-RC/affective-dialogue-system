# Affective Dialogue System

A modular Python project for affective dialogue: automatic speech recognition,
emotion classification, emotionally structured dialogue generation, interest
scoring, and text-to-speech.

The repository is intentionally lightweight. Large checkpoints and pretrained
weights are loaded from Hugging Face or a local cache instead of being committed
to Git.

## Models

| Component | Default model |
| --- | --- |
| Dialogue generation | `mario-rc/emotional-rlaif-dpo-gemma-2-2b-it` |
| Emotional GPT-2 generation | `mario-rc/emotional-gpt2-medium` |
| Emotion classifier, base | `mario-rc/multilingual-emotional-classifier-xlm-roberta-base` |
| Emotion classifier, large | `mario-rc/multilingual-emotional-classifier-xlm-roberta-large` |
| ASR | `openai/whisper-large-v3-turbo` |
| TTS | `tts_models/multilingual/multi-dataset/xtts_v2` |

## Installation

```bash
python -m pip install -e ".[dev]"
```

Install optional runtime groups as needed:

```bash
python -m pip install -e ".[ml]"
python -m pip install -e ".[ml,asr,tts]"
```

## Quick Usage

Interest scoring does not require model downloads:

```bash
affective-dialogue interest "Siento dolor y mareo"
```

Emotion classification downloads the configured Hugging Face model on first use:

```bash
affective-dialogue emotion "I am really happy today"
```

Generate a single emotionally structured response:

```bash
affective-dialogue chat "No puedo creer que hayamos perdido otra vez" --language es --emotion ANGER
```

## Python API

```python
from affective_dialogue_system.dialogue import (
    AffectiveDialogueEngine,
    AssistantResponse,
    DialogueTurn,
    seed_dialogue,
)
from affective_dialogue_system.emotion import Emotion

turns = seed_dialogue("es")
turns.append(
    DialogueTurn(
        Emotion.ANGER,
        "No puedo creer que hayamos perdido otra vez.",
        AssistantResponse.placeholder(Emotion.ANGER, Emotion.SADNESS),
    )
)

engine = AffectiveDialogueEngine(language="es")
response = engine.generate(turns)
print(response.format())
```

## Repository Layout

```text
src/affective_dialogue_system/   importable Python package
configs/                         model and runtime defaults
examples/                        runnable examples
tests/                           lightweight unit tests
docs/                            architecture and model notes
scripts/                         maintenance helpers
assets/                          tiny repo-safe assets only
```

## Large Artifacts

Do not commit model files such as `.pt`, `.pth`, `.bin`, `.safetensors`, `.onnx`,
or generated audio. Keep them in a Hugging Face repository, a local cache, or a
separate artifact directory.

The previous local working copy contained large checkpoints and a vendored
Coqui TTS checkout. Those artifacts should remain outside this Git repository.

