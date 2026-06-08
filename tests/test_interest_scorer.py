import unittest

from affective_dialogue_system.interest import score_interest


class InterestScorerTest(unittest.TestCase):
    def test_scores_medical_keywords_high(self) -> None:
        self.assertEqual(score_interest("Siento dolor y mareo."), 100)

    def test_scores_travel_keywords_low(self) -> None:
        self.assertEqual(score_interest("Me gusta viajar a nuevos lugares."), 20)

    def test_scores_unknown_text_zero(self) -> None:
        self.assertEqual(score_interest("Hoy he leido un libro."), 0)


if __name__ == "__main__":
    unittest.main()

