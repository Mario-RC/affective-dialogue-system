"""Protocol-level handlers for Selecting Strategy."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

from affective_dialogue_system.strategy.context import DialogueContext, StrategyResult
from affective_dialogue_system.strategy.responses import (
    timeout_response,
    topic_switch_response,
)


@dataclass(frozen=True)
class GlucoseMonitoringProtocol:
    """Research protocol ported from the prototype; not clinically validated."""

    threshold_mg_dl: float = 69.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold_mg_dl) or self.threshold_mg_dl <= 0:
            raise ValueError("threshold_mg_dl must be finite and positive")

    def handle(self, context: DialogueContext) -> StrategyResult | None:
        state = context.metadata.get("glucose_protocol_state")
        if state:
            result = self._continue_protocol(context, state)
            if result is not None:
                return result
        if context.glucose_level is None or context.glucose_level > self.threshold_mg_dl:
            return None

        return self._step(
            context,
            "ask_feeling",
            _t(
                context,
                "Don't worry, you are experiencing low blood sugar. Are you feeling okay?",
                "No te pongas nervioso, estas teniendo una bajada de glucosa. Te encuentras bien?",
            ),
            glucose_level=context.glucose_level,
        )

    def _continue_protocol(self, context: DialogueContext, state: str) -> StrategyResult | None:
        if state == "ask_feeling":
            answer = _yes_no(context.current_message)
            if answer is True:
                return self._ask_first_measure(context)
            if answer is False:
                return self._step(
                    context,
                    "ask_symptoms",
                    _t(
                        context,
                        "Do you have symptoms like dizziness, nausea, mental fog, headache, sweating, weakness, tingling, or tremors?",
                        "Tienes sintomas como mareo, malestar, niebla mental, dolor de cabeza, sudores, debilidad, hormigueo o temblores?",
                    ),
                )
            return self._repeat_yes_no(context)

        if state == "ask_symptoms":
            answer = _yes_no(context.current_message)
            if answer is True:
                return self._step(
                    context,
                    "ask_can_handle",
                    _t(
                        context,
                        "Can you handle the situation?",
                        "Puedes hacerte cargo de la situacion?",
                    ),
                )
            if answer is False:
                return self._ask_first_measure(context)
            return self._repeat_yes_no(context)

        if state == "ask_can_handle":
            answer = _yes_no(context.current_message)
            if answer is True:
                return self._ask_first_measure(context)
            if answer is False:
                return self._step(
                    context,
                    "ask_can_take_sugar",
                    _t(
                        context,
                        "Take 20 to 25 grams of sugar as soon as possible. Can you?",
                        "Toma entre 20 y 25 gramos de azucar lo antes posible. Puedes?",
                    ),
                )
            return self._repeat_yes_no(context)

        if state == "ask_can_take_sugar":
            answer = _yes_no(context.current_message)
            if answer is False:
                return self._complete(
                    context,
                    _t(context, "Call emergency services.", "Avisa a emergencias."),
                    should_end_session=True,
                )
            if answer is True:
                return self._ask_second_measure(
                    context, context.metadata.get("glucose_initial_level")
                )
            return self._repeat_yes_no(context)

        if state == "ask_first_measure":
            level = _parse_glucose_level(context)
            if level is None:
                return self._step(
                    context,
                    "ask_first_measure",
                    _t(
                        context,
                        "Please indicate your glucose level.",
                        "Indicame tu nivel de glucosa, por favor.",
                    ),
                )
            return self._handle_first_measure(context, level)

        if state == "ask_family":
            answer = _yes_no(context.current_message)
            if answer is False:
                return self._complete(
                    context,
                    _t(context, "Call emergency services.", "Avisa a emergencias."),
                    should_end_session=True,
                )
            if answer is True:
                return self._ask_second_measure(
                    context, context.metadata.get("glucose_initial_level")
                )
            return self._repeat_yes_no(context)

        if state == "ask_second_measure":
            level = _parse_glucose_level(context)
            if level is None:
                return self._ask_second_measure(
                    context, context.metadata.get("glucose_initial_level")
                )
            return self._handle_second_measure(context, level)

        context.metadata.pop("glucose_protocol_state", None)
        context.metadata.pop("glucose_initial_level", None)
        return None

    def _handle_first_measure(self, context: DialogueContext, level: float) -> StrategyResult:
        context.glucose_level = level
        context.metadata["glucose_initial_level"] = level
        if level >= 70:
            return self._complete(
                context,
                _t(
                    context,
                    "Great! There is nothing to worry about.",
                    "Enhorabuena, ya no hay nada de lo que preocuparse.",
                ),
            )
        if 65 < level <= 70:
            return self._step(
                context,
                "ask_second_measure",
                _t(
                    context,
                    "Eat a fruit, a glass of milk, or fast-absorbing carbohydrate food. Wait five minutes and check your glucose level again.",
                    "Toma una fruta, un vaso de leche o algun alimento con hidratos de carbono de absorcion rapida. Espera cinco minutos y vuelve a indicarme tu nivel de glucosa.",
                ),
                glucose_level=level,
            )
        if 45 < level <= 65:
            return self._step(
                context,
                "ask_second_measure",
                _t(
                    context,
                    "Take between 10 and 20 grams of sugar diluted in water. Wait five minutes and check your glucose level again.",
                    "Toma entre 10 y 20 gramos de azucar diluido en agua. Espera cinco minutos y vuelve a indicarme tu nivel de glucosa.",
                ),
                glucose_level=level,
            )
        return self._step(
            context,
            "ask_family",
            _t(
                context,
                "Take between 10 and 20 grams of sugar diluted in water. Can you call a family member?",
                "Toma entre 10 y 20 gramos de azucar diluido en agua. Puedes avisar a un familiar?",
            ),
            glucose_level=level,
        )

    def _handle_second_measure(self, context: DialogueContext, level: float) -> StrategyResult:
        context.glucose_level = level
        initial = context.metadata.get("glucose_initial_level")
        if level >= 70:
            return self._complete(
                context,
                _t(
                    context,
                    "Great! There is nothing to worry about.",
                    "Enhorabuena, ya no hay nada de lo que preocuparse.",
                ),
                glucose_level=level,
            )
        if level <= 45:
            return self._step(
                context,
                "ask_family",
                _t(
                    context,
                    "Take between 10 and 20 grams of sugar diluted in water. Can you call a family member?",
                    "Toma entre 10 y 20 gramos de azucar diluido en agua. Puedes avisar a un familiar?",
                ),
                glucose_level=level,
            )
        if initial is None or level > float(initial):
            context.metadata["glucose_initial_level"] = level
            return self._ask_second_measure(context, level)
        return self._step(
            context,
            "ask_second_measure",
            _t(
                context,
                "Your glucose is still low. Take fast-absorbing sugar, wait five minutes, and check it again.",
                "Tu glucosa sigue baja. Toma azucar de absorcion rapida, espera cinco minutos y vuelve a medirla.",
            ),
            glucose_level=level,
        )

    def _ask_first_measure(self, context: DialogueContext) -> StrategyResult:
        return self._step(
            context,
            "ask_first_measure",
            _t(
                context,
                "Please indicate your glucose level.",
                "Indicame tu nivel de glucosa, por favor.",
            ),
        )

    def _ask_second_measure(
        self, context: DialogueContext, initial_level: float | None
    ) -> StrategyResult:
        if initial_level is not None:
            context.metadata["glucose_initial_level"] = initial_level
        return self._step(
            context,
            "ask_second_measure",
            _t(
                context,
                "Wait five minutes and check your glucose level again.",
                "Espera cinco minutos y vuelve a indicarme tu nivel de glucosa.",
            ),
        )

    def _repeat_yes_no(self, context: DialogueContext) -> StrategyResult:
        return self._step(
            context,
            str(context.metadata.get("glucose_protocol_state", "ask_feeling")),
            _t(context, "Please answer yes or no.", "Responde si o no, por favor."),
        )

    def _step(
        self,
        context: DialogueContext,
        next_state: str,
        response: str,
        *,
        glucose_level: float | None = None,
    ) -> StrategyResult:
        context.metadata["glucose_protocol_state"] = next_state
        metadata = {
            "protocol_active": True,
            "next_state": next_state,
            "threshold_mg_dl": self.threshold_mg_dl,
        }
        if glucose_level is not None:
            metadata["glucose_level"] = glucose_level
        return StrategyResult.from_parts(
            response=response,
            source="glucose_protocol",
            metadata=metadata,
        )

    def _complete(
        self,
        context: DialogueContext,
        response: str,
        *,
        glucose_level: float | None = None,
        should_end_session: bool = False,
    ) -> StrategyResult:
        context.metadata.pop("glucose_protocol_state", None)
        context.metadata.pop("glucose_initial_level", None)
        metadata = {"protocol_active": False, "completed": True}
        if glucose_level is not None:
            metadata["glucose_level"] = glucose_level
        return StrategyResult.from_parts(
            response=response,
            source="glucose_protocol",
            should_end_session=should_end_session,
            metadata=metadata,
        )


@dataclass(frozen=True)
class TopicMonitoringProtocol:
    max_consecutive_turns: int = 5

    def __post_init__(self) -> None:
        if self.max_consecutive_turns < 1:
            raise ValueError("max_consecutive_turns must be positive")

    def handle(self, context: DialogueContext) -> StrategyResult | None:
        if context.topic and context.topic_turn_count >= self.max_consecutive_turns:
            return StrategyResult(
                response=topic_switch_response(context.language),
                source="topic_monitor",
                metadata={
                    "topic": context.topic,
                    "topic_turn_count": context.topic_turn_count,
                },
            )
        return None


def _t(context: DialogueContext, english: str, spanish: str) -> str:
    return english if context.language == "en" else spanish


def _yes_no(text: str) -> bool | None:
    normalized = text.lower().strip().strip(".!?¿¡")
    if normalized in {"yes", "y", "si", "sí", "s"}:
        return True
    if normalized in {"no", "n"}:
        return False
    return None


def _parse_glucose_level(context: DialogueContext) -> float | None:
    match = re.fullmatch(
        r"\s*(\d+(?:[.,]\d+)?)\s*(?:mg\s*/\s*dl)?\s*", context.current_message, re.IGNORECASE
    )
    if match:
        value = float(match.group(1).replace(",", "."))
        return value if math.isfinite(value) and value > 0 else None
    # Do not reuse the measurement that originally triggered the protocol.
    return None


@dataclass(frozen=True)
class TimeoutProtocol:
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

    def handle(
        self, context: DialogueContext, *, now: datetime | None = None
    ) -> StrategyResult | None:
        if context.current_message.strip() or context.last_user_input_at is None:
            return None
        now = now or datetime.now(tz=context.last_user_input_at.tzinfo)
        if (now.utcoffset() is None) != (context.last_user_input_at.utcoffset() is None):
            raise ValueError("now and last_user_input_at must use compatible timezones")
        elapsed = (now - context.last_user_input_at).total_seconds()
        if elapsed >= self.timeout_seconds:
            return StrategyResult(
                response=timeout_response(context.language),
                source="timeout",
                metadata={"elapsed_seconds": elapsed, "timeout_seconds": self.timeout_seconds},
            )
        return None
