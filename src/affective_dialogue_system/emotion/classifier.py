"""Text emotion classifier backed by Hugging Face transformers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from affective_dialogue_system.config import DEFAULT_EMOTION_MODEL_BASE
from affective_dialogue_system.emotion.labels import EMOTION_LABELS
from affective_dialogue_system.runtime import resolve_device


@dataclass(frozen=True)
class EmotionPrediction:
    label: str
    confidence: float
    scores: dict[str, float]


class EmotionClassifier:
    """Predict one of the supported emotion labels for a text input."""

    def __init__(
        self,
        model_id: str = DEFAULT_EMOTION_MODEL_BASE,
        *,
        device: str | None = None,
        max_length: int = 128,
        trust_remote_code: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = resolve_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()
        self._torch = torch

    def predict(self, text: str) -> EmotionPrediction:
        if not text.strip():
            raise ValueError("text must not be empty")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = self._torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        labels = self._labels()
        scores = {
            labels[idx]: float(probabilities[idx].detach().cpu()) for idx in range(len(labels))
        }
        predicted_idx = int(self._torch.argmax(probabilities).detach().cpu())
        return EmotionPrediction(
            label=labels[predicted_idx],
            confidence=scores[labels[predicted_idx]],
            scores=scores,
        )

    def _labels(self) -> list[str]:
        count = self.model.config.num_labels
        mapping = getattr(self.model.config, "id2label", None) or {}
        labels = [mapping.get(index, mapping.get(str(index))) for index in range(count)]
        if all(label is not None for label in labels) and not all(
            str(label).upper() == f"LABEL_{index}" for index, label in enumerate(labels)
        ):
            return [str(label).lower() for label in labels]
        if count == len(EMOTION_LABELS):
            return list(EMOTION_LABELS)
        raise ValueError("Model must define id2label or use the seven project emotion labels")
