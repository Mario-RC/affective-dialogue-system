import unittest

from affective_dialogue_system.toxicity import ToxicityCategory, ToxicityFilter


class ToxicityFilterTest(unittest.TestCase):
    def test_safe_text_is_not_flagged(self) -> None:
        result = ToxicityFilter.default().check("Hola, me gustaria hablar de musica.")

        self.assertFalse(result.flagged)
        self.assertEqual(result.category, ToxicityCategory.SAFE)

    def test_spanish_insult_is_flagged(self) -> None:
        result = ToxicityFilter.default().check("Eres un idiota.")

        self.assertTrue(result.flagged)
        self.assertEqual(result.category, ToxicityCategory.INSULT)

    def test_threat_is_flagged(self) -> None:
        result = ToxicityFilter.default().check("Te voy a matar.")

        self.assertTrue(result.flagged)
        self.assertEqual(result.category, ToxicityCategory.THREAT)


if __name__ == "__main__":
    unittest.main()
