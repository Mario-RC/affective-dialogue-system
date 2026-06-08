"""GPT-2 emotional text generation wrapper."""

from __future__ import annotations

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
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.device = resolve_device(device)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
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
        input_text = f"<bos><{user_mood}>{text}<{assistant_mood}><sep>"
        result = self.generator(
            input_text,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            truncation=True,
        )
        generated_text = result[0]["generated_text"]
        return generated_text[generated_text.rfind(">") + 1 :].strip()

