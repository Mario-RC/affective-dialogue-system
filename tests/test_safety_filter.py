import unittest

from affective_dialogue_system.safety import SafetyCategory, SafetyFilter


class SafetyFilterTest(unittest.TestCase):
    def test_safe_text_is_not_flagged(self) -> None:
        result = SafetyFilter.default().check("Hola, me gustaria hablar de musica.")

        self.assertFalse(result.flagged)
        self.assertEqual(result.category, SafetyCategory.SAFE)

    def test_spanish_insult_is_flagged(self) -> None:
        result = SafetyFilter.default().check("Eres un idiota.")

        self.assertTrue(result.flagged)
        self.assertEqual(result.category, SafetyCategory.INSULT)

    def test_threat_is_flagged(self) -> None:
        result = SafetyFilter.default().check("Te voy a matar.")

        self.assertTrue(result.flagged)
        self.assertEqual(result.category, SafetyCategory.THREAT)


if __name__ == "__main__":
    unittest.main()
