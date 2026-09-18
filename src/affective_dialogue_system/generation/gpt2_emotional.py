"""GPT-2 emotional text generation wrapper."""

from __future__ import annotations

from pathlib import Path

from affective_dialogue_system.config import DEFAULT_EMOTIONAL_GPT2_MODEL
from affective_dialogue_system.runtime import pipeline_device_index, resolve_device


class EmotionalGPT2Generator:
    """Generate text with the emotional GPT-2 model."""

    def __init__(
        self,
        model_id: str = DEFAULT_EMOTIONAL_GPT2_MODEL,
        *,
        device: str | None = None,
        trust_remote_code: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.device = resolve_device(device)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=pipeline_device_index(self.device),
        )

    def generate(
        self,
        text: str,
        *,
        user_mood: str = "no emotion",
        assistant_mood: str = "no emotion",
        max_new_tokens: int = 50,
    ) -> str:
        if not text.strip() or max_new_tokens < 1:
            raise ValueError("text must not be empty and max_new_tokens must be positive")
        input_text = f"<bos><{user_mood}>{text}<{assistant_mood}><sep>"
        result = self.generator(
            input_text,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            return_full_text=False,
        )
        generated_text = result[0]["generated_text"]
        return generated_text.strip()
