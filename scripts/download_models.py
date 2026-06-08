"""Download default Hugging Face models into a local cache."""

from __future__ import annotations

import argparse

from affective_dialogue_system.config import (
    DEFAULT_DIALOGUE_MODEL,
    DEFAULT_EMOTION_MODEL_BASE,
    DEFAULT_EMOTION_MODEL_LARGE,
    DEFAULT_EMOTIONAL_GPT2_MODEL,
)


DEFAULT_MODELS = [
    DEFAULT_DIALOGUE_MODEL,
    DEFAULT_EMOTIONAL_GPT2_MODEL,
    DEFAULT_EMOTION_MODEL_BASE,
    DEFAULT_EMOTION_MODEL_LARGE,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    for model_id in args.models:
        path = snapshot_download(model_id, cache_dir=args.cache_dir)
        print(f"{model_id}\t{path}")


if __name__ == "__main__":
    main()

