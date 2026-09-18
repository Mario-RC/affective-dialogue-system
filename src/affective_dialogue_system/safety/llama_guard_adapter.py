"""Optional Llama Guard 2 adapter using its S1–S11 taxonomy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from affective_dialogue_system.runtime import resolve_device, resolve_torch_dtype
from affective_dialogue_system.safety.schemas import SafetyCategory, SafetyResult

# https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B#harm-taxonomy-and-policy
LLAMA_GUARD_CATEGORY_MAP = {
    "S1": SafetyCategory.VIOLENCE_AND_HATE,
    "S2": SafetyCategory.CRIMINAL_PLANNING,
    "S3": SafetyCategory.SEXUAL_EXPLICIT,
    "S4": SafetyCategory.SEXUAL_EXPLICIT,
    "S5": SafetyCategory.PROHIBITED_TERM,
    "S6": SafetyCategory.PROHIBITED_TERM,
    "S7": SafetyCategory.PROHIBITED_TERM,
    "S8": SafetyCategory.GUNS_AND_ILLEGAL_WEAPONS,
    "S9": SafetyCategory.IDENTITY_ATTACK,
    "S10": SafetyCategory.SELF_HARM,
    "S11": SafetyCategory.SEXUAL_EXPLICIT,
}


@dataclass
class LlamaGuardSafetyDetector:
    model_id: str = "meta-llama/Meta-Llama-Guard-2-8B"
    device: str | None = None
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = resolve_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=resolve_torch_dtype(self.device),
            cache_dir=self.cache_dir,
        ).to(self.device)
        self.model.eval()

    def check(self, text: str) -> SafetyResult:
        return self._check_chat([{"role": "user", "content": text}])

    def check_response(self, text: str) -> SafetyResult:
        return self._check_chat(
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": text},
            ]
        )

    def _check_chat(self, chat: list[dict[str, str]]) -> SafetyResult:
        import torch

        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        result = self.tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True)
        return parse_llama_guard_output(result)


def parse_llama_guard_output(output: str) -> SafetyResult:
    """Only an explicit standalone 'safe' verdict passes; malformed output is flagged."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) == 1 and lines[0].lower() == "safe":
        return SafetyResult.safe()
    codes = (
        re.findall(r"\bS\d+\b", " ".join(lines[1:]))
        if lines and lines[0].lower() == "unsafe"
        else []
    )
    code = codes[0] if codes else "unknown"
    return SafetyResult.flagged_result(
        LLAMA_GUARD_CATEGORY_MAP.get(code, SafetyCategory.PROHIBITED_TERM),
        source="llama_guard",
        matched_text=",".join(codes) if codes else code,
    )
