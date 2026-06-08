"""Whisper ASR service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from affective_dialogue_system.config import DEFAULT_ASR_MODEL
from affective_dialogue_system.runtime import pipeline_device_index, resolve_device, resolve_torch_dtype


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str | None = None


class WhisperASR:
    """Speech recognition wrapper around ``transformers`` Whisper models."""

    def __init__(
        self,
        model_id: str = DEFAULT_ASR_MODEL,
        *,
        device: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.device = resolve_device(device)
        dtype = resolve_torch_dtype(self.device, torch_dtype)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=dtype,
            device=pipeline_device_index(self.device),
            return_language=True,
        )

    def transcribe(self, audio_path: str | Path, *, language: str | None = None) -> Transcription:
        generate_kwargs = {"language": language} if language else None
        result = self.pipe(
            str(audio_path),
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        if "chunks" in result and result["chunks"]:
            chunk = result["chunks"][0]
            return Transcription(
                text=chunk.get("text", "").strip(),
                language=chunk.get("language"),
            )
        return Transcription(text=result.get("text", "").strip(), language=result.get("language"))

