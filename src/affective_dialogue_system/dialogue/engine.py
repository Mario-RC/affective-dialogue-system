"""Dialogue generation engine."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from affective_dialogue_system.config import DEFAULT_DIALOGUE_MODEL
from affective_dialogue_system.dialogue.parser import ParseFallback, parse_emotional_response
from affective_dialogue_system.dialogue.prompts import build_prompt
from affective_dialogue_system.dialogue.schemas import AssistantResponse, DialogueTurn
from affective_dialogue_system.runtime import resolve_device


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class AffectiveDialogueEngine:
    """Generate emotionally structured assistant responses."""

    def __init__(
        self,
        model_id: str = DEFAULT_DIALOGUE_MODEL,
        *,
        template: str = "gemma",
        language: str = "en",
        device: str | None = None,
        trust_remote_code: bool = False,
        generation: GenerationSettings | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.template = template
        self.language = language
        self.device = resolve_device(device)
        self.generation = generation or GenerationSettings()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

    def generate(self, turns: Sequence[DialogueTurn]) -> AssistantResponse:
        prompt = build_prompt(turns, template=self.template, language=self.language)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation.max_new_tokens,
            temperature=self.generation.temperature,
            top_p=self.generation.top_p,
            do_sample=self.generation.do_sample,
        )
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        raw_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        first_line = raw_response.splitlines()[0].strip() if raw_response else raw_response
        latest = turns[-1]
        fallback_second = latest.assistant.second_emotion if latest.assistant else latest.user_emotion
        return parse_emotional_response(
            first_line,
            fallback=ParseFallback(latest.user_emotion, fallback_second, self.language),
        )

