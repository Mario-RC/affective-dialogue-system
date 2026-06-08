import unittest

from affective_dialogue_system.dialogue import AssistantResponse, DialogueTurn, build_prompt, seed_dialogue
from affective_dialogue_system.emotion import Emotion


class PromptBuilderTest(unittest.TestCase):
    def test_builds_gemma_prompt_for_latest_user_turn(self) -> None:
        turns = seed_dialogue("es")
        turns.append(
            DialogueTurn(
                Emotion.ANGER,
                "Otra vez hemos perdido.",
                AssistantResponse.placeholder(Emotion.ANGER, Emotion.SADNESS),
            )
        )

        prompt = build_prompt(turns, template="gemma", language="es")

        self.assertIn("<start_of_turn>user", prompt)
        self.assertIn("(ANGER) Otra vez hemos perdido.", prompt)
        self.assertIn("RESPONSE_2 debe tener un tono SADNESS", prompt)

    def test_rejects_unsupported_template(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt(seed_dialogue("en"), template="unknown", language="en")


if __name__ == "__main__":
    unittest.main()

