"""Generate one dialogue turn through the configured response selector."""

from dataclasses import replace

from affective_dialogue_system.config import load_config
from affective_dialogue_system.emotion import Emotion
from affective_dialogue_system.factory import create_dialogue_engine, create_strategy
from affective_dialogue_system.strategy import DialogueContext, DialogueEngineGenerator


def main() -> None:
    config = load_config()
    config = replace(
        config, selecting_strategy=replace(config.selecting_strategy, llm_timeout_seconds=30.0)
    )
    generator = DialogueEngineGenerator(
        create_dialogue_engine(config), second_emotion=Emotion.SADNESS
    )
    context = DialogueContext(
        current_message="No puedo creer que nuestro equipo haya perdido otra vez.",
        user_emotion=Emotion.ANGER,
        language=config.dialogue.language,
    )
    with create_strategy(config, primary_generator=generator) as strategy:
        result = strategy.select_response(context)
        print(result.response)
        print(f"Selected by: {result.source}")


if __name__ == "__main__":
    main()
