"""Project configuration defaults."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints

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

    def __post_init__(self) -> None:
        if self.torch_dtype not in {None, "float16", "float32", "bfloat16", "float64"}:
            raise ValueError("Unsupported floating-point torch_dtype")


@dataclass(frozen=True)
class DialogueConfig:
    template: str = "gemma"
    language: str = "es"
    human_name: str = "Human"
    assistant_name: str = "Ray"
    max_history_turns: int = 20
    max_new_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    def __post_init__(self) -> None:
        if self.template not in {"gemma", "llama3"} or self.language not in {"en", "es"}:
            raise ValueError("dialogue requires template gemma/llama3 and language en/es")
        if self.max_history_turns < 1 or self.max_new_tokens < 1:
            raise ValueError("History and generation limits must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1]")


@dataclass(frozen=True)
class DetoxifyConfig:
    enabled: bool = False
    model_type: str = "multilingual"
    checkpoint: str | None = None


@dataclass(frozen=True)
class LlamaGuardConfig:
    enabled: bool = False
    model: str = "meta-llama/Meta-Llama-Guard-2-8B"


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool = True
    local_rules: bool = True
    detoxify: DetoxifyConfig = field(default_factory=DetoxifyConfig)
    llama_guard: LlamaGuardConfig = field(default_factory=LlamaGuardConfig)

    def __post_init__(self) -> None:
        if self.enabled and not (
            self.local_rules or self.detoxify.enabled or self.llama_guard.enabled
        ):
            raise ValueError("Enabled safety requires at least one detector")


@dataclass(frozen=True)
class StrategyConfig:
    glucose_threshold_mg_dl: float = 69.0
    topic_max_consecutive_turns: int = 5
    timeout_seconds: float = 30.0
    llm_timeout_seconds: float = 2.0
    rule_based_templates: str = "legacy"
    skip_rule_based_catch_all: bool = True

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0
            for value in (
                self.glucose_threshold_mg_dl,
                self.topic_max_consecutive_turns,
                self.timeout_seconds,
                self.llm_timeout_seconds,
            )
        ):
            raise ValueError("Strategy thresholds and timeouts must be positive")
        if self.rule_based_templates != "legacy":
            raise ValueError("Only the legacy rule-based template set is available")


@dataclass(frozen=True)
class AppConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    selecting_strategy: StrategyConfig = field(default_factory=StrategyConfig)


_MODEL_ALIASES = {
    "dialogue": "dialogue_model",
    "emotional_gpt2": "emotional_gpt2_model",
    "emotion_classifier_base": "emotion_classifier_base",
    "emotion_classifier_large": "emotion_classifier_large",
    "asr": "asr_model",
    "tts": "tts_model",
}


def load_config(
    path: str | Path | None = None, *, environ: Mapping[str, str] | None = None
) -> AppConfig:
    """Load explicit YAML, then exported ADS_* overrides. No implicit file loading.

    Unknown keys and incorrect types are errors. YAML uses ``safe_load`` and
    model/cache paths are interpreted relative to the calling working directory.
    """
    data = {}
    if path is not None:
        import yaml

        try:
            with Path(path).open(encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration: {path}") from exc
        if data is None:
            data = {}
    config = _construct_config(AppConfig, data, "config")
    environment = os.environ if environ is None else environ
    runtime_changes = {}
    if environment.get("ADS_DEVICE"):
        runtime_changes["device"] = environment["ADS_DEVICE"]
    if environment.get("ADS_CACHE_DIR"):
        runtime_changes["cache_dir"] = Path(environment["ADS_CACHE_DIR"]).expanduser()
    if runtime_changes:
        config = replace(config, runtime=replace(config.runtime, **runtime_changes))
    if environment.get("ADS_SPEAKER_WAV"):
        config = replace(
            config,
            models=replace(
                config.models, speaker_wav=Path(environment["ADS_SPEAKER_WAV"]).expanduser()
            ),
        )
    return config


def _construct_config(cls: type, data: object, name: str):
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ValueError(f"{name} must be a mapping with string keys")
    types = get_type_hints(cls)
    kwargs = {}
    for key, value in data.items():
        actual_key = _MODEL_ALIASES.get(key, key) if cls is ModelConfig else key
        if actual_key not in types:
            raise ValueError(f"Unknown configuration key: {name}.{key}")
        if actual_key in kwargs:
            raise ValueError(f"Duplicate configuration key: {name}.{key}")
        kwargs[actual_key] = _config_value(value, types[actual_key], f"{name}.{key}")
    return cls(**kwargs)


def _config_value(value: object, expected: type, name: str):
    if is_dataclass(expected):
        return _construct_config(expected, value, name)
    if get_origin(expected) is UnionType:
        if value is None and type(None) in get_args(expected):
            return None
        expected = next(item for item in get_args(expected) if item is not type(None))
    if expected is Path and isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    if expected is float and type(value) in {int, float} and math.isfinite(value):
        return float(value)
    if expected is not float and type(value) is expected and (expected is not str or value.strip()):
        return value
    raise ValueError(
        f"Invalid value for {name}: expected {getattr(expected, '__name__', expected)}"
    )
