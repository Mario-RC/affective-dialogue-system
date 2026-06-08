"""Project configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DIALOGUE_MODEL = "mario-rc/emotional-rlaif-dpo-gemma-2-2b-it"
DEFAULT_EMOTIONAL_GPT2_MODEL = "mario-rc/emotional-gpt2-medium"
DEFAULT_EMOTION_MODEL_BASE = "mario-rc/multilingual-emotional-classifier-xlm-roberta-base"
DEFAULT_EMOTION_MODEL_LARGE = "mario-rc/multilingual-emotional-classifier-xlm-roberta-large"
DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers used by the system.

    These are intentionally Hugging Face or model-catalog identifiers instead of
    local paths. Local caches can still be used by passing custom paths at
    runtime.
    """

    dialogue_model: str = DEFAULT_DIALOGUE_MODEL
    emotional_gpt2_model: str = DEFAULT_EMOTIONAL_GPT2_MODEL
    emotion_classifier_base: str = DEFAULT_EMOTION_MODEL_BASE
    emotion_classifier_large: str = DEFAULT_EMOTION_MODEL_LARGE
    asr_model: str = DEFAULT_ASR_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    speaker_wav: Path | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings shared by services."""

    device: str | None = None
    torch_dtype: str | None = None
    cache_dir: Path | None = None
    trust_remote_code: bool = False

