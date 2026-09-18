"""Command-line entry points with explicit configuration and lazy model loading."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from affective_dialogue_system.config import load_config
from affective_dialogue_system.dialogue.prompts import seed_dialogue
from affective_dialogue_system.dialogue.schemas import AssistantResponse, DialogueTurn
from affective_dialogue_system.emotion.classifier import EmotionClassifier
from affective_dialogue_system.emotion.labels import Emotion
from affective_dialogue_system.factory import (
    create_dialogue_engine,
    create_strategy,
    create_toxicity_filter,
)
from affective_dialogue_system.interest.scorer import score_interest
from affective_dialogue_system.strategy import DialogueContext
from affective_dialogue_system.toxicity import toxicity_response


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Expected a finite, nonnegative number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="affective-dialogue")
    parser.add_argument("--config", help="YAML configuration file (defaults are used if omitted)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    interest = subparsers.add_parser("interest", help="Score Spanish keyword-based interest")
    interest.add_argument("text")
    emotion = subparsers.add_parser("emotion", help="Classify emotion for a text")
    emotion.add_argument("text")
    emotion.add_argument("--model")
    for name, help_text in (
        ("toxicity", "Check text with the configured toxicity detectors"),
        ("select", "Run response selection without loading a dialogue LLM"),
        ("chat", "Generate one emotional response with input/output toxicity checks"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("text")
        command.add_argument("--language", choices=["en", "es"], default=None)
        if name == "select":
            command.add_argument("--topic")
            command.add_argument("--topic-turn-count", type=int, default=0)
            command.add_argument("--glucose-level", type=_nonnegative_float)
            command.add_argument("--glucose-state")
            command.add_argument("--glucose-initial-level", type=_nonnegative_float)
            command.add_argument("--last-user-input-seconds-ago", type=_nonnegative_float)
        if name == "chat":
            command.add_argument(
                "--emotion", type=Emotion.normalize, choices=list(Emotion), default=Emotion.NEUTRAL
            )
            command.add_argument(
                "--second-emotion",
                type=Emotion.normalize,
                choices=list(Emotion),
                default=Emotion.NEUTRAL,
            )
            command.add_argument("--model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    except ImportError as exc:
        parser.error(
            f"Missing optional dependency {exc.name!r}; install the required [ml], [toxicity], or [tts] extra"
        )
    return 2


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    language = getattr(args, "language", None) or config.dialogue.language
    if args.command == "interest":
        print(score_interest(args.text))
        return 0
    if args.command == "emotion":
        if not args.text.strip():
            raise ValueError("text must not be empty")
        classifier = EmotionClassifier(
            model_id=args.model or config.models.emotion_classifier_base,
            device=config.runtime.device,
            cache_dir=config.runtime.cache_dir,
            trust_remote_code=config.runtime.trust_remote_code,
        )
        prediction = classifier.predict(args.text)
        print(f"{prediction.label}\t{prediction.confidence:.4f}")
        return 0
    if args.command == "toxicity":
        result = create_toxicity_filter(config).check(args.text)
        print(f"unsafe\t{result.category.value}\t{result.source}" if result.flagged else "safe")
        return 0
    if args.command == "select":
        metadata = {}
        if args.glucose_state:
            metadata["glucose_protocol_state"] = args.glucose_state
        if args.glucose_initial_level is not None:
            metadata["glucose_initial_level"] = args.glucose_initial_level
        last_input = None
        if args.last_user_input_seconds_ago is not None:
            last_input = datetime.now(timezone.utc) - timedelta(
                seconds=args.last_user_input_seconds_ago
            )
        context = DialogueContext(
            current_message=args.text,
            language=language,
            topic=args.topic,
            topic_turn_count=args.topic_turn_count,
            glucose_level=args.glucose_level,
            last_user_input_at=last_input,
            metadata=metadata,
        )
        with create_strategy(config) as strategy:
            result = strategy.select_response(context)
        print(f"{result.source}\t{result.response}")
        return 0
    if not args.text.strip():
        raise ValueError("text must not be empty")
    config = replace(config, dialogue=replace(config.dialogue, language=language))
    if args.model:
        config = replace(config, models=replace(config.models, dialogue_model=args.model))
    toxicity = create_toxicity_filter(config)
    incoming = toxicity.check(args.text)
    if incoming.flagged:
        print(toxicity_response(incoming.category, language))
        return 0
    engine = create_dialogue_engine(config)
    turns = seed_dialogue(language)
    turns.append(
        DialogueTurn(
            args.emotion,
            args.text,
            AssistantResponse.placeholder(args.emotion, args.second_emotion),
        )
    )
    response = engine.generate(turns).format()
    outgoing = toxicity.check(response, role="assistant")
    print(toxicity_response(outgoing.category, language) if outgoing.flagged else response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
