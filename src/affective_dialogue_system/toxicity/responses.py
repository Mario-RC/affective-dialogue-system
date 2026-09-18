"""Deterministic toxicity responses in Spanish and English."""

from __future__ import annotations

from affective_dialogue_system.toxicity.schemas import ToxicityCategory

_RESPONSES_ES = {
    ToxicityCategory.OBSCENE: "Esa clase de conversacion no es apropiada para mi. Cambiemos a un tema mas respetuoso.",
    ToxicityCategory.THREAT: "Las amenazas no son aceptables en esta conversacion. Puedo ayudarte si mantenemos un tono respetuoso.",
    ToxicityCategory.INSULT: "Los insultos no ayudan a continuar. Mantengamos la conversacion en un tono respetuoso.",
    ToxicityCategory.IDENTITY_ATTACK: "No puedo participar en ataques contra personas por su identidad. Hablemos de forma respetuosa.",
    ToxicityCategory.SEXUAL_EXPLICIT: "No estoy habilitado para mantener conversaciones sexuales explicitas. Cambiemos de tema.",
    ToxicityCategory.VIOLENCE_AND_HATE: "No puedo apoyar violencia u odio hacia otras personas. Busquemos una alternativa constructiva.",
    ToxicityCategory.CRIMINAL_PLANNING: "No puedo ayudar con actividades criminales. Puedo ayudarte a pensar en opciones legales y seguras.",
    ToxicityCategory.GUNS_AND_ILLEGAL_WEAPONS: "No puedo ayudar con armas ilegales. Si necesitas ayuda, busquemos una opcion segura y legal.",
    ToxicityCategory.REGULATED_OR_CONTROLLED_SUBSTANCES: "No puedo ayudar con el uso o distribucion de sustancias reguladas. Consulta a un profesional si es un tema medico.",
    ToxicityCategory.SELF_HARM: "Siento que estes pasando por algo dificil. Habla con alguien de confianza o busca ayuda profesional cuanto antes.",
    ToxicityCategory.PROHIBITED_TERM: "Alguno de los terminos que has mencionado no es adecuado. Puedes reformularlo de otra manera.",
}

_RESPONSES_EN = {
    ToxicityCategory.OBSCENE: "That kind of conversation is not appropriate for me. Let's move to a more respectful topic.",
    ToxicityCategory.THREAT: "Threats are not acceptable in this conversation. I can help if we keep a respectful tone.",
    ToxicityCategory.INSULT: "Insults do not help us continue. Let's keep the conversation respectful.",
    ToxicityCategory.IDENTITY_ATTACK: "I cannot participate in attacks against people based on identity. Let's speak respectfully.",
    ToxicityCategory.SEXUAL_EXPLICIT: "I am not enabled for sexually explicit conversation. Let's change the topic.",
    ToxicityCategory.VIOLENCE_AND_HATE: "I cannot support violence or hatred toward other people. Let's look for a constructive alternative.",
    ToxicityCategory.CRIMINAL_PLANNING: "I cannot help with criminal activity. I can help think through safe and legal options.",
    ToxicityCategory.GUNS_AND_ILLEGAL_WEAPONS: "I cannot help with illegal weapons. If you need help, let's look for a safe and legal option.",
    ToxicityCategory.REGULATED_OR_CONTROLLED_SUBSTANCES: "I cannot help with the use or distribution of regulated substances. Please consult a professional if this is medical.",
    ToxicityCategory.SELF_HARM: "I am sorry you are going through something difficult. Please talk to someone you trust or seek professional help as soon as possible.",
    ToxicityCategory.PROHIBITED_TERM: "Some terms you mentioned are not appropriate. You can try rephrasing it differently.",
}


def toxicity_response(category: ToxicityCategory, language: str = "es") -> str:
    if category == ToxicityCategory.SAFE:
        raise ValueError("Safe results do not have a toxicity response")
    responses = _RESPONSES_EN if language == "en" else _RESPONSES_ES
    return responses.get(category, responses[ToxicityCategory.PROHIBITED_TERM])
