from affective_dialogue_system.dialogue import (
    AffectiveDialogueEngine,
    AssistantResponse,
    DialogueTurn,
    seed_dialogue,
)
from affective_dialogue_system.emotion import Emotion


def main() -> None:
    turns = seed_dialogue("es")
    turns.append(
        DialogueTurn(
            Emotion.ANGER,
            "No puedo creer que nuestro equipo haya perdido otra vez.",
            AssistantResponse.placeholder(Emotion.ANGER, Emotion.SADNESS),
        )
    )
    engine = AffectiveDialogueEngine(language="es")
    print(engine.generate(turns).format())


if __name__ == "__main__":
    main()

