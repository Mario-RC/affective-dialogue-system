"""Command-line interface."""

from __future__ import annotations

import argparse

from affective_dialogue_system.dialogue.engine import AffectiveDialogueEngine
from affective_dialogue_system.dialogue.prompts import seed_dialogue
from affective_dialogue_system.dialogue.schemas import AssistantResponse, DialogueTurn
from affective_dialogue_system.emotion.classifier import EmotionClassifier
from affective_dialogue_system.emotion.labels import Emotion
from affective_dialogue_system.interest.scorer import score_interest


def main() -> None:
    parser = argparse.ArgumentParser(prog="affective-dialogue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    interest = subparsers.add_parser("interest", help="Score interest for a text")
    interest.add_argument("text")

    emotion = subparsers.add_parser("emotion", help="Classify emotion for a text")
    emotion.add_argument("text")
    emotion.add_argument("--model", default=None)

    chat = subparsers.add_parser("chat", help="Generate one emotionally structured response")
    chat.add_argument("text")
    chat.add_argument("--emotion", default="NEUTRAL")
    chat.add_argument("--second-emotion", default="NEUTRAL")
    chat.add_argument("--language", choices=["en", "es"], default="es")
    chat.add_argument("--model", default=None)

    args = parser.parse_args()

    if args.command == "interest":
        print(score_interest(args.text))
        return

    if args.command == "emotion":
        classifier = EmotionClassifier(model_id=args.model) if args.model else EmotionClassifier()
        prediction = classifier.predict(args.text)
        print(f"{prediction.label}\t{prediction.confidence:.4f}")
        return

    if args.command == "chat":
        engine = AffectiveDialogueEngine(
            model_id=args.model or "mario-rc/emotional-rlaif-dpo-gemma-2-2b-it",
            language=args.language,
            template="gemma",
        )
        user_emotion = Emotion.normalize(args.emotion)
        second_emotion = Emotion.normalize(args.second_emotion)
        turns = seed_dialogue(args.language)
        turns.append(
            DialogueTurn(
                user_emotion,
                args.text,
                AssistantResponse.placeholder(user_emotion, second_emotion),
            )
        )
        print(engine.generate(turns).format())


if __name__ == "__main__":
    main()

