import unittest
from datetime import datetime, timedelta, timezone

from affective_dialogue_system.dialogue import AssistantResponse
from affective_dialogue_system.emotion import Emotion
from affective_dialogue_system.strategy import DialogueContext, SelectingStrategy


class SelectingStrategyTest(unittest.TestCase):
    def test_glucose_protocol_preempts_dialogue(self) -> None:
        context = DialogueContext(current_message="hola", language="es", glucose_level=60)
        result = SelectingStrategy().select_response(context)

        self.assertEqual(result.source, "glucose_protocol")
        self.assertEqual(context.metadata["glucose_protocol_state"], "ask_feeling")

    def test_glucose_protocol_advances_to_measurement(self) -> None:
        context = DialogueContext(
            current_message="si",
            language="es",
            metadata={"glucose_protocol_state": "ask_feeling"},
        )

        result = SelectingStrategy().select_response(context)

        self.assertEqual(result.source, "glucose_protocol")
        self.assertEqual(context.metadata["glucose_protocol_state"], "ask_first_measure")
        self.assertIn("glucosa", result.response)

    def test_glucose_protocol_completes_after_recovered_measurement(self) -> None:
        context = DialogueContext(
            current_message="72",
            language="es",
            metadata={"glucose_protocol_state": "ask_second_measure", "glucose_initial_level": 60},
        )

        result = SelectingStrategy().select_response(context)

        self.assertEqual(result.source, "glucose_protocol")
        self.assertTrue(result.metadata["completed"])
        self.assertNotIn("glucose_protocol_state", context.metadata)

    def test_timeout_prompts_user_when_silent(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = SelectingStrategy().select_response(
            DialogueContext(
                current_message="",
                language="en",
                last_user_input_at=now - timedelta(seconds=31),
            ),
            now=now,
        )

        self.assertEqual(result.source, "timeout")

    def test_topic_monitor_suggests_topic_switch(self) -> None:
        result = SelectingStrategy().select_response(
            DialogueContext(
                current_message="seguimos hablando",
                language="es",
                topic="basketball",
                topic_turn_count=5,
            )
        )

        self.assertEqual(result.source, "topic_monitor")

    def test_toxicity_handler_preempts_regex_and_llm(self) -> None:
        result = SelectingStrategy(primary_generator=lambda _: "respuesta segura").select_response(
            DialogueContext(current_message="Eres un idiota.", language="es")
        )

        self.assertEqual(result.source, "toxicity_filter")
        self.assertTrue(result.toxicity.flagged)

    def test_regex_handler_handles_greeting(self) -> None:
        result = SelectingStrategy().select_response(
            DialogueContext(current_message="hola, quien eres?", language="es")
        )

        self.assertEqual(result.source, "rule_based")

    def test_generated_unsafe_response_is_discarded_for_safe_fallback_llm(self) -> None:
        strategy = SelectingStrategy(
            primary_generator=lambda _: "Eres un idiota.",
            fallback_generator=lambda _: "I can help you with that calmly.",
            fallback_translator=lambda text, _: "Puedo ayudarte con eso con calma.",
        )

        result = strategy.select_response(
            DialogueContext(current_message="cuentame algo", language="es")
        )

        self.assertEqual(result.source, "fallback_llm")
        self.assertIn("calma", result.response)

    def test_structured_emotional_llm_output_keeps_follow_up_separate(self) -> None:
        strategy = SelectingStrategy(
            primary_generator=lambda _: AssistantResponse(
                Emotion.HAPPINESS,
                "Me alegra escucharte.",
                Emotion.NEUTRAL,
                "Podemos pensarlo con calma.",
                Emotion.NEUTRAL,
                "Que parte quieres explorar primero?",
            )
        )

        result = strategy.select_response(
            DialogueContext(current_message="cuentame algo nuevo", language="es")
        )

        self.assertEqual(result.source, "primary_llm")
        self.assertEqual(result.follow_up_question, "Que parte quieres explorar primero?")
        self.assertIn("calma", result.emotional_response)

    def test_no_valid_handler_returns_fallback(self) -> None:
        result = SelectingStrategy().select_response(
            DialogueContext(current_message="cuentame algo nuevo", language="en")
        )

        self.assertEqual(result.source, "fallback")


if __name__ == "__main__":
    unittest.main()
