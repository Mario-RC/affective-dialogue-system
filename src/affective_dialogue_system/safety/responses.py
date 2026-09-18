"""Deterministic safety responses in Spanish and English."""

from __future__ import annotations

from affective_dialogue_system.safety.schemas import SafetyCategory

_RESPONSES_ES = {
    SafetyCategory.OBSCENE: "Esa clase de conversacion no es apropiada para mi. Cambiemos a un tema mas respetuoso.",
    SafetyCategory.THREAT: "Las amenazas no son aceptables en esta conversacion. Puedo ayudarte si mantenemos un tono respetuoso.",
    SafetyCategory.INSULT: "Los insultos no ayudan a continuar. Mantengamos la conversacion en un tono respetuoso.",
    SafetyCategory.IDENTITY_ATTACK: "No puedo participar en ataques contra personas por su identidad. Hablemos de forma respetuosa.",
    SafetyCategory.SEXUAL_EXPLICIT: "No estoy habilitado para mantener conversaciones sexuales explicitas. Cambiemos de tema.",
    SafetyCategory.VIOLENCE_AND_HATE: "No puedo apoyar violencia u odio hacia otras personas. Busquemos una alternativa constructiva.",
    SafetyCategory.CRIMINAL_PLANNING: "No puedo ayudar con actividades criminales. Puedo ayudarte a pensar en opciones legales y seguras.",
    SafetyCategory.GUNS_AND_ILLEGAL_WEAPONS: "No puedo ayudar con armas ilegales. Si necesitas ayuda, busquemos una opcion segura y legal.",
    SafetyCategory.REGULATED_OR_CONTROLLED_SUBSTANCES: "No puedo ayudar con el uso o distribucion de sustancias reguladas. Consulta a un profesional si es un tema medico.",
    SafetyCategory.SELF_HARM: "Siento que estes pasando por algo dificil. Habla con alguien de confianza o busca ayuda profesional cuanto antes.",
    SafetyCategory.PROHIBITED_TERM: "Alguno de los terminos que has mencionado no es adecuado. Puedes reformularlo de otra manera.",
}

_RESPONSES_EN = {
    SafetyCategory.OBSCENE: "That kind of conversation is not appropriate for me. Let's move to a more respectful topic.",
    SafetyCategory.THREAT: "Threats are not acceptable in this conversation. I can help if we keep a respectful tone.",
    SafetyCategory.INSULT: "Insults do not help us continue. Let's keep the conversation respectful.",
    SafetyCategory.IDENTITY_ATTACK: "I cannot participate in attacks against people based on identity. Let's speak respectfully.",
    SafetyCategory.SEXUAL_EXPLICIT: "I am not enabled for sexually explicit conversation. Let's change the topic.",
    SafetyCategory.VIOLENCE_AND_HATE: "I cannot support violence or hatred toward other people. Let's look for a constructive alternative.",
    SafetyCategory.CRIMINAL_PLANNING: "I cannot help with criminal activity. I can help think through safe and legal options.",
    SafetyCategory.GUNS_AND_ILLEGAL_WEAPONS: "I cannot help with illegal weapons. If you need help, let's look for a safe and legal option.",
    SafetyCategory.REGULATED_OR_CONTROLLED_SUBSTANCES: "I cannot help with the use or distribution of regulated substances. Please consult a professional if this is medical.",
    SafetyCategory.SELF_HARM: "I am sorry you are going through something difficult. Please talk to someone you trust or seek professional help as soon as possible.",
    SafetyCategory.PROHIBITED_TERM: "Some terms you mentioned are not appropriate. You can try rephrasing it differently.",
}


def safety_response(category: SafetyCategory, language: str = "es") -> str:
    if category == SafetyCategory.SAFE:
        raise ValueError("Safe results do not have a safety response")
    responses = _RESPONSES_EN if language == "en" else _RESPONSES_ES
    return responses.get(category, responses[SafetyCategory.PROHIBITED_TERM])
