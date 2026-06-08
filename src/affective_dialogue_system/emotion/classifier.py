"""Text emotion classifier backed by Hugging Face transformers."""

from __future__ import annotations

from dataclasses import dataclass

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
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = resolve_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()
        self._torch = torch

    def predict(self, text: str) -> EmotionPrediction:
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
            labels[idx]: float(probabilities[idx].detach().cpu())
            for idx in range(len(labels))
        }
        predicted_idx = int(self._torch.argmax(probabilities).detach().cpu())
        return EmotionPrediction(
            label=labels[predicted_idx],
            confidence=scores[labels[predicted_idx]],
            scores=scores,
        )

    def _labels(self) -> list[str]:
        id2label = getattr(self.model.config, "id2label", None) or {}
        if id2label and len(id2label) == self.model.config.num_labels:
            return [
                str(id2label.get(idx, id2label.get(str(idx), EMOTION_LABELS[idx]))).lower()
                for idx in range(len(id2label))
            ]
        return list(EMOTION_LABELS)
