"""Dialogue generation engine with bounded history and validated output."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from affective_dialogue_system.config import DEFAULT_DIALOGUE_MODEL
from affective_dialogue_system.dialogue.parser import ParseFallback, parse_emotional_response
from affective_dialogue_system.dialogue.prompts import build_prompt
from affective_dialogue_system.dialogue.schemas import AssistantResponse, DialogueTurn
from affective_dialogue_system.emotion.labels import Emotion
from affective_dialogue_system.runtime import resolve_device, resolve_torch_dtype


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


class AffectiveDialogueEngine:
    """Generate emotional triples. Toxicity filtering belongs to SelectingStrategy."""

    def __init__(
        self,
        model_id: str = DEFAULT_DIALOGUE_MODEL,
        *,
        template: str = "gemma",
        language: str = "en",
        device: str | None = None,
        trust_remote_code: bool = False,
        generation: GenerationSettings | None = None,
        max_history_turns: int = 20,
        cache_dir: str | Path | None = None,
        torch_dtype: str | None = None,
        human_name: str = "Human",
        assistant_name: str = "Ray",
    ) -> None:
        if template not in {"gemma", "llama3"} or language not in {"en", "es"}:
            raise ValueError("Supported templates: gemma, llama3; languages: en, es")
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be positive")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id, self.template, self.language = model_id, template, language
        self.device = resolve_device(device)
        self.generation = generation or GenerationSettings()
        self.max_history_turns = max_history_turns
        self.human_name, self.assistant_name = human_name, assistant_name
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=resolve_torch_dtype(self.device, torch_dtype),
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()

    def generate(self, turns: Sequence[DialogueTurn]) -> AssistantResponse:
        turns = list(turns)[-self.max_history_turns :]
        prompt = build_prompt(
            turns,
            template=self.template,
            language=self.language,
            human_name=self.human_name,
            assistant_name=self.assistant_name,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.device
        )
        input_length = inputs["input_ids"].shape[-1]
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if (
            isinstance(context_limit, int)
            and input_length + self.generation.max_new_tokens > context_limit
        ):
            raise ValueError(
                "Dialogue exceeds the model context window; shorten the history or message"
            )
        kwargs = {
            "max_new_tokens": self.generation.max_new_tokens,
            "do_sample": self.generation.do_sample,
        }
        if self.generation.do_sample:
            kwargs.update(temperature=self.generation.temperature, top_p=self.generation.top_p)
        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        raw_response = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()
        latest = turns[-1]
        second = latest.assistant.second_emotion if latest.assistant else Emotion.NEUTRAL
        return parse_emotional_response(
            raw_response,
            fallback=ParseFallback(latest.user_emotion, second, self.language),
            expected_emotions=(latest.user_emotion, second),
        )
