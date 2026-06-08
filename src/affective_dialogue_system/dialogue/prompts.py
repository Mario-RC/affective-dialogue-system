"""Prompt construction for emotionally structured dialogue."""

from __future__ import annotations

from collections.abc import Sequence

from affective_dialogue_system.dialogue.schemas import AssistantResponse, DialogueTurn
from affective_dialogue_system.emotion.labels import Emotion


def seed_dialogue(language: str = "en") -> list[DialogueTurn]:
    """Return a short stable seed conversation for demos and tests."""

    if language == "es":
        return [
            DialogueTurn(
                Emotion.HAPPINESS,
                "Hola, quien eres?",
                AssistantResponse(
                    Emotion.HAPPINESS,
                    "Hola, soy Ray.",
                    Emotion.HAPPINESS,
                    "Estoy aqui para charlar contigo sobre cualquier tema.",
                    Emotion.NEUTRAL,
                    "Sobre que te gustaria hablar?",
                ),
            ),
            DialogueTurn(
                Emotion.HAPPINESS,
                "Me gustaria saber mas sobre ti.",
                AssistantResponse(
                    Emotion.HAPPINESS,
                    "El gusto es mio.",
                    Emotion.NEUTRAL,
                    "Soy un asistente conversacional disenado para responder con sensibilidad emocional.",
                    Emotion.NEUTRAL,
                    "Que tema te interesa explorar con mas detalle?",
                ),
            ),
        ]

    return [
        DialogueTurn(
            Emotion.HAPPINESS,
            "Hello, who are you?",
            AssistantResponse(
                Emotion.HAPPINESS,
                "Hi, I am Ray.",
                Emotion.HAPPINESS,
                "I am here to chat with you about any topic.",
                Emotion.NEUTRAL,
                "What would you like to talk about?",
            ),
        ),
        DialogueTurn(
            Emotion.HAPPINESS,
            "I would like to know more about you.",
            AssistantResponse(
                Emotion.HAPPINESS,
                "The pleasure is mine.",
                Emotion.NEUTRAL,
                "I am a conversational assistant designed to respond with emotional awareness.",
                Emotion.NEUTRAL,
                "What topic would you like to explore in more detail?",
            ),
        ),
    ]


def build_prompt(
    turns: Sequence[DialogueTurn],
    *,
    template: str = "gemma",
    language: str = "en",
    human_name: str = "Human",
    assistant_name: str = "Ray",
) -> str:
    """Build a single-turn generation prompt.

    The final turn should contain the user's latest message. If its assistant
    response is empty or missing, the model is prompted to complete it.
    """

    if not turns:
        raise ValueError("At least one dialogue turn is required")
    if template not in {"gemma", "llama3"}:
        raise ValueError(f"Unsupported template: {template}")
    if language not in {"en", "es"}:
        raise ValueError(f"Unsupported language: {language}")

    if template == "llama3":
        return _build_llama3_prompt(turns, language, human_name, assistant_name)
    return _build_gemma_prompt(turns, language, human_name, assistant_name)


def _context_lines(turns: Sequence[DialogueTurn], language: str) -> str:
    human_label = "Humano" if language == "es" else "Human"
    assistant_label = "Chatbot"
    lines = []
    for turn in turns:
        second_emotion = turn.assistant.second_emotion if turn.assistant else Emotion.NEUTRAL
        lines.append(f"{human_label}: ({turn.user_emotion.value}) PROMPT.")
        lines.append(
            f"{assistant_label}: ({turn.user_emotion.value}) RESPONSE_1. "
            f"({second_emotion.value}) RESPONSE_2. (NEUTRAL) RESPONSE_3."
        )
    return "\n".join(lines)


def _rules(turns: Sequence[DialogueTurn], language: str, human_name: str, assistant_name: str) -> str:
    latest = turns[-1]
    assistant = latest.assistant or AssistantResponse.placeholder(latest.user_emotion)
    if language == "es":
        return f"""Reglas de dialogo:
La respuesta debe ser de dominio abierto, coherente, empatica, atractiva y proactiva.
La respuesta del chatbot se compone de 3 frases diferentes: RESPONSE_1, RESPONSE_2 y RESPONSE_3.
Entre RESPONSE_1, RESPONSE_2 y RESPONSE_3 debe haber una longitud maxima de 20-25 palabras.
RESPONSE_3 debe ser abierta para continuar la conversacion. Evita preguntas de si/no.

Reglas emocionales:
RESPONSE_1 debe tener un tono {latest.user_emotion.value}.
RESPONSE_2 debe tener un tono {assistant.second_emotion.value}.
RESPONSE_3 debe tener un tono NEUTRAL.

El humano se llama {human_name}. El chatbot se llama {assistant_name}.
Responde en un solo turno y sigue exactamente la estructura emocional."""

    return f"""Dialogue rules:
The response must be open-domain, coherent, empathetic, engaging, and proactive.
The chatbot response is composed of 3 different sentences: RESPONSE_1, RESPONSE_2, and RESPONSE_3.
RESPONSE_1, RESPONSE_2, and RESPONSE_3 should have a maximum total length of 20-25 words.
RESPONSE_3 must be open-ended to continue the conversation. Avoid yes/no questions.

Emotional response rules:
RESPONSE_1 must contain a {latest.user_emotion.value} tone.
RESPONSE_2 must contain a {assistant.second_emotion.value} tone.
RESPONSE_3 must contain a NEUTRAL tone.

The human name is {human_name}. The chatbot name is {assistant_name}.
Answer in a single turn and follow exactly the emotional structure."""


def _completion(turns: Sequence[DialogueTurn], *, user_start: str, assistant_start: str, end: str) -> str:
    chunks = []
    for idx, turn in enumerate(turns):
        chunks.append(f"({turn.user_emotion.value}) {turn.user_text}{end}\n{assistant_start}\n")
        if idx != len(turns) - 1 and turn.assistant:
            chunks.append(f"{turn.assistant.format()}{end}\n{user_start}\n")
    return "".join(chunks)


def _build_gemma_prompt(
    turns: Sequence[DialogueTurn],
    language: str,
    human_name: str,
    assistant_name: str,
) -> str:
    system = (
        "<bos>"
        + ("Eres un experto en la creacion de dialogos." if language == "es" else "You are an expert at creating dialogues.")
        + "\n\nDialogue and emotional structure:\n"
        + _context_lines(turns, language)
        + "\n\n"
        + _rules(turns, language, human_name, assistant_name)
        + "<start_of_turn>user\n"
    )
    return system + _completion(
        turns,
        user_start="<start_of_turn>user",
        assistant_start="<start_of_turn>model",
        end="<end_of_turn>",
    )


def _build_llama3_prompt(
    turns: Sequence[DialogueTurn],
    language: str,
    human_name: str,
    assistant_name: str,
) -> str:
    system_text = "Eres un experto en la creacion de dialogos." if language == "es" else "You are an expert at creating dialogues."
    system = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + system_text
        + "\n\nDialogue and emotional structure:\n"
        + _context_lines(turns, language)
        + "\n\n"
        + _rules(turns, language, human_name, assistant_name)
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    )
    return system + _completion(
        turns,
        user_start="<|start_header_id|>user<|end_header_id|>",
        assistant_start="<|start_header_id|>assistant<|end_header_id|>",
        end="<|eot_id|>",
    )

