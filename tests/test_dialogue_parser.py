import unittest

from affective_dialogue_system.dialogue.parser import ParseFallback, parse_emotional_response
from affective_dialogue_system.emotion import Emotion


class DialogueParserTest(unittest.TestCase):
    def test_parses_three_emotional_segments(self) -> None:
        response = parse_emotional_response(
            "(ANGER) Entiendo tu frustracion. (SADNESS) Ha sido una derrota dura. "
            "(NEUTRAL) Que crees que deberia cambiar el equipo?"
        )

        self.assertEqual(response.first_emotion, Emotion.ANGER)
        self.assertEqual(response.second_emotion, Emotion.SADNESS)
        self.assertEqual(response.third_emotion, Emotion.NEUTRAL)
        self.assertIn("frustracion", response.first_text)

    def test_returns_fallback_when_requested(self) -> None:
        response = parse_emotional_response(
            "not parseable",
            fallback=ParseFallback(Emotion.ANGER, Emotion.SADNESS, "es"),
        )

        self.assertEqual(response.first_emotion, Emotion.ANGER)
        self.assertEqual(response.second_emotion, Emotion.SADNESS)


if __name__ == "__main__":
    unittest.main()

