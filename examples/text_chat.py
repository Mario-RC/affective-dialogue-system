"""Direct emotional generation with bounded conversation history."""

from affective_dialogue_system.config import load_config
from affective_dialogue_system.factory import create_dialogue_engine
from affective_dialogue_system.pipeline import AffectiveDialogueSystem


def main() -> None:
    config = load_config()
    conversation = AffectiveDialogueSystem(
        create_dialogue_engine(config),
        max_history_turns=config.dialogue.max_history_turns,
    )
    response = conversation.reply(
        "Me preocupa la presentación de mañana.",
        user_emotion="FEAR",
        second_emotion="NEUTRAL",
    )
    print(response.format())


if __name__ == "__main__":
    main()
