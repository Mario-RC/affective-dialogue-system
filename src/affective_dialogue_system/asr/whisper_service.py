"""Whisper ASR service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from affective_dialogue_system.config import DEFAULT_ASR_MODEL
from affective_dialogue_system.runtime import (
    pipeline_device_index,
    resolve_device,
    resolve_torch_dtype,
)


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
        cache_dir: str | Path | None = None,
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
            cache_dir=cache_dir,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
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
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        generate_kwargs = {"language": language} if language else {}
        result = self.pipe(
            str(audio_path),
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        chunks = result.get("chunks") or []
        text = result.get("text") or " ".join(chunk.get("text", "").strip() for chunk in chunks)
        detected_language = result.get("language") or next(
            (chunk["language"] for chunk in chunks if chunk.get("language")),
            language,
        )
        return Transcription(text=text.strip(), language=detected_language)
