"""Deterministic responses used by Selecting Strategy."""

from __future__ import annotations

FALLBACK_RESPONSES = {
    "es": "Lo siento, no te he entendido. Podrias repetirmelo?",
    "en": "I'm sorry, I didn't understand. Could you repeat that?",
}

TIMEOUT_RESPONSES = {
    "es": "Estas muy callado... quieres que sigamos hablando?",
    "en": "You are too quiet... would you like to keep talking?",
}

TOPIC_SWITCH_RESPONSES = {
    "es": "Llevamos un rato hablando de lo mismo. Te apetece que cambiemos de tema?",
    "en": "We have been talking about the same topic for a while. Would you like to switch topics?",
}

GLUCOSE_LOW_RESPONSES = {
    "es": "Tu glucosa parece estar baja. Vamos a pausar la conversacion y centrarnos en que estes bien.",
    "en": "Your glucose appears to be low. Let's pause the conversation and focus on keeping you safe.",
}

REGEX_RESPONSES_ES = {
    "greeting": "Hola, soy Ray. Me alegra hablar contigo. Sobre que te gustaria conversar?",
    "identity": "Soy Ray, un sistema de dialogo afectivo disenado para conversar con sensibilidad emocional.",
    "health": "Si te encuentras mal, puedo acompanarte, pero es importante consultar con un profesional sanitario si hay sintomas relevantes.",
}

REGEX_RESPONSES_EN = {
    "greeting": "Hi, I am Ray. I am glad to talk with you. What would you like to discuss?",
    "identity": "I am Ray, an affective dialogue system designed to respond with emotional awareness.",
    "health": "If you feel unwell, I can stay with you, but it is important to contact a healthcare professional for relevant symptoms.",
}


def fallback_response(language: str) -> str:
    return FALLBACK_RESPONSES.get(language, FALLBACK_RESPONSES["es"])


def timeout_response(language: str) -> str:
    return TIMEOUT_RESPONSES.get(language, TIMEOUT_RESPONSES["es"])


def topic_switch_response(language: str) -> str:
    return TOPIC_SWITCH_RESPONSES.get(language, TOPIC_SWITCH_RESPONSES["es"])


def glucose_low_response(language: str) -> str:
    return GLUCOSE_LOW_RESPONSES.get(language, GLUCOSE_LOW_RESPONSES["es"])
