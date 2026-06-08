"""Coqui TTS service wrapper.

The Coqui package is an optional dependency. This module does not vendor Coqui
TTS source code into this repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from affective_dialogue_system.config import DEFAULT_TTS_MODEL
from affective_dialogue_system.runtime import resolve_device


LANGUAGE_CODES = {
    "spanish": "es",
    "es": "es",
    "english": "en",
    "en": "en",
}


class CoquiTTSService:
    """Synthesize speech with Coqui TTS."""

    def __init__(
        self,
        model_name: str = DEFAULT_TTS_MODEL,
        *,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        speaker_wav: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        from TTS.api import TTS

        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        self.device = resolve_device(device)
        self.speaker_wav = Path(speaker_wav) if speaker_wav else None
        if model_path:
            self.tts = TTS(model_path=str(model_path), config_path=str(config_path)).to(self.device)
        else:
            self.tts = TTS(model_name=model_name, progress_bar=False).to(self.device)

    def synthesize(
        self,
        text: str,
        *,
        language: str = "spanish",
        speaker_wav: str | Path | None = None,
    ):
        speaker = Path(speaker_wav) if speaker_wav else self.speaker_wav
        return self.tts.tts(
            text=text,
            speaker_wav=str(speaker) if speaker else None,
            language=_language_code(language),
        )

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        *,
        language: str = "spanish",
        speaker_wav: str | Path | None = None,
    ) -> Path:
        speaker = Path(speaker_wav) if speaker_wav else self.speaker_wav
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=str(speaker) if speaker else None,
            language=_language_code(language),
        )
        return output_path


def _language_code(language: str) -> str:
    try:
        return LANGUAGE_CODES[language.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported language: {language}") from exc

