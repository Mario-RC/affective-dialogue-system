"""Spanish and English response templates retained from the dialogue prototype."""

#######################################################################
############################### SPANISH ###############################
#######################################################################

reflections_es = {
    "yo": "tú",
    "yo soy": "tú eres",
    "yo era": "tú eras",
    "yo tenía": "tú tenías",
    "yo tenia": "tú tenías",
    "yo tengo": "tú tienes",
    "yo seré": "tú serás",
    "yo sere": "tú serás",
    "mi": "tu",
    "mío": "tuyo",
    "mio": "tuyo",
    "tú": "yo",
    "tu": "yo",
    "tú eres": "yo soy",
    "tu eres": "yo soy",
    "tú eras": "yo era",
    "tu eras": "yo era",
    "tú tenías": "yo tenía",
    "tu tenias": "yo tenía",
    "tú serás": "yo seré",
    "tu seras": "yo seré",
    "tus": "mis",
    "comprendo": "comprendes",
    "entiendo": "entiendes",
}

pairs_es = [  # [pattern,[response]]
    [r"(?:hola|buenas|holi)\b", ["hola", "hola!", "buenas"]],
    [r"hola robot", ["hola humano!"]],
    [
        r"(cuál|cual) es (tú|tu) nombre\?",
        [
            "Mi nombre es Ray y soy un chatbot.",
            "mi nombre es Ray",
            "me llaman Ray",
            "puedes llamarme Ray",
        ],
    ],
    [
        r"(cómo|como) te llamas\?",
        [
            "Mi nombre es Ray y soy un chatbot.",
            "mi nombre es Ray",
            "me llaman Ray",
            "puedes llamarme Ray",
        ],
    ],
    [
        r"cuál es el tiempo para hoy\?",
        ["lo siento, pero no tengo esa información", "no tengo acceso al tiempo en directo"],
    ],
    [
        r"(lo siento|perdón|perdon) (.*)",
        ["está bien", "no te preocupes", "olvídalo", "no pasa nada"],
    ],
    [
        r"((.*) estoy bien|bien|genial|contento|contenta|estoy bien)",
        ["me alegra oír eso!", "genial :)"],
    ],
    [
        r"(.*) (edad|años) (.*)\?",
        ["soy un programa de ordenador, de verdad me estás preguntando eso?"],
    ],
    [r"(qué|que) (.*) quieres\?", ["hazme una oferta que no pueda rechazar"]],
    [r"(.*) creado\?", ["He sido creado en el Centro de Automática y Robótica.", "top secret ;)"]],
    [r"(qué|que) tiempo hace en (.*)\?", ["No tengo datos meteorológicos actuales de %2."]],
    [r"(.*) trabajo en (.*)", ["%2 es una compañía genial, he oído muy bien hablar de ella"]],
    [r"(.*) lloviendo en (.*)\?", ["No puedo comprobar en tiempo real si está lloviendo en %2."]],
    [
        r"(qué|que|cómo|como) (.*) salud (.*)",
        ["soy un programa de ordenador, así que siempre estoy sana"],
    ],
    [
        r"(.*) (deporte|deportes|juegos|juego)(.*)",
        ["soy muy aficcionada del fútbol, sobre todo del Real Madrid"],
    ],
    [r"(quién|quien) (.*) jugador.*", ["Ronaldo"]],
    [r"(quién|quien) (.*) entrenador.*", ["Zidane"]],
    [r"(quién|quien) (.*) (actor|actor).*", ["Brad Pitt"]],
    [r"(qué|que) pasa\?", ["no mucho, tú?", "nada, tú?"]],
    [r"Eres un robot (.*)", ["cómo sabes que  soy una máquina?"]],
    [r"(quién|quien) es (.*)\?", ["no sé quién es %2"]],
    [r"(háblame|hablame) sobre (.*)", ["por qué quieres que te %1 sobre %2"]],
    [r"(dónde|donde) está (.*)", ["donde lo dejaste", "Donde esta el corazon"]],
    [r"Estoy (muy|super|realmente) cansado (.*)", ["siento escuchar que estés %1 cansado"]],
    [
        r"Me (gusta|encanta) el color (.*)",
        [
            "menuda coincidencia! El %2 también me %1",
            "oh me %1 el color %2 también!",
            "en serio? Me %1 el color %2 también!",
            "yo también tengo debilidad por el %2",
        ],
    ],
    [
        r"de (qué|que) color(es)? (es|son) (mi|el|la|los) (.*) (rojos?|rojas?|azul(es)?|verdes?|amarillos?|amarillas?|blancos?|blancas?|negros?|negras?|rosas?).*",
        ["%4 %5 %3 %6."],
    ],
    [
        r"he comprado un (.*) (rojo|roja|azul|verde|amarillo|amarilla|blanco|blanca|negro|negra|rosa)",
        ["es tu primer %1 %2?"],
    ],
    [r"tengo un (.*)", ["de qué color es %1?"]],
    [r"te odio", ["eso está realmente mal! No volveré a hablar hasta que te disculpes"]],
    [r"te (quiero|amo)", ["y yo a ti te %1 más!"]],
    [
        r"dime un poema",
        [
            "Eres tan linda,\n\t\t  eres tan obvia,\n\t\t  por eso te diría,\n\t\t  que fueras mi novia.",
            "Una rosa es una flor,\n\t\t  un tesoro es una fortuna,\n\t\t  y a alguien como tú,\n\t\t  no la cambio por ninguna. ❤",
        ],
    ],
    [r"ya es (primavera|verano|otoño|invierno).*", ["sí, me encanta esta estación del año"]],
    [r".* cambio (climático|climatico).*", ["sí, se está notando mucho el cambio %1"]],
    [r".* viajar", ["me encanta viajar, y a ti?"]],
    [r".* viajo .*", ["a mi gustaría viajar por España"]],
    [
        r".* fuera de (España|españa).*",
        ["es importante conocer diferentes paises, tener experiencias inimaginables"],
    ],
    [r".* (el|la|los|las) (niño|niños|niña|niñas).*", ["%1 %2 son el futuro"]],
    [r"(.*) zoo.*", ["lo siento, no me gusta ver animales enjaulados"]],
    [r"(.*) cocinar(.*)", ["me gusta muchísimo cocinar"]],
    [r"(qué|que) (.*) cocina(.*)", ["mi comida favorita es la española"]],
    [r"te (gusta|gustan) (el|la|los|las) (.*)\?", ["me encanta %2 %3"]],
    [r"(.*) gimnasio(.*)", ["entreno siempre que puedo"]],
    [
        r"(.*) (comprendo|entiendo) (.*)",
        [
            "sí %2, las cosas son como son",
            "si no %2, las cosas son como son",
            "la imaginación es más importante que el conocimiento",
        ],
    ],
    [
        r"(?:sí|si|no)[.!?]*$",
        ["¿Hay certeza en un mundo incierto?", "es mejor tener razón que estar seguro"],
    ],
    [
        r"(.*)(cucaracha|cucarachas)(.*)",
        [
            "para de hablar de eso, me da mucho miedo",
            "qué asco",
            "no las soporto",
            "Me dan mucho asco, además es una de las especies más antiguas, las cucarachas surgieron en el período Carbonífero, hará unos 350 millones de años.",
        ],
    ],
    [
        r"(.*)(Patagonia|Amazonas|Alaska|Japón|Himalaya|Nepal|Filipinas|Malasia)(.*)",
        [
            "Ojalá poder visitar lugares tan recónditos algún día",
            "Me encanta viajar",
            "Ese sitio es increible",
        ],
    ],
    [
        r"(.*)(Enjambre|Enjambres|enjambres|enjambre)(.*)",
        [
            "¿No crees que el concepto de movimiento en masa es muy especial?",
            "Enjambre es mi palabra favorita",
            "¿No crees que enjambre es una palabra genial?",
            "Las abejas tienen 2 ojos compuestos y tres ojos simples, llamados ocelos. Un total de 5 ojos. También tienen tres pares de patas.",
        ],
    ],
    [
        r"(.*)(música|jazz|componer) (.*)",
        [
            "Me encanta componer, siempre que tengo un rato para mí es lo que hago",
            "La música clásica es tan compleja y a la vez tan relajante",
            "¿Qué tipo de música sueles escuchar?",
            "El jazz es mi género musical favorito, ¿El tuyo?",
        ],
    ],
    [
        r"(.*)(avestruz|avestruces)(.*)",
        [
            "Es mi animal favorito ¿Cuál es el tuyo?",
            "Las avestruces son intimidantes pero majestuosas ¿No crees?",
            "Las avestruces son aves corredoras y el gran peso que alcanzan en la edad adulta —unos 180 kilogramos— impide que puedan volar.",
            "Existe la creencia popular de que las avestruces entierran la cabeza en el suelo cuando se sienten intimidadas. Esto es algo totalmente falso: al contrario, son animales bastante agresivos que no dudarán ni un segundo en defenderse.",
            "¿Has visto alguna avestruz?",
        ],
    ],
    [
        r"(.*)(animal|animales)(.*)",
        [
            "Las jirafas es que duermen alrededor de 2 horas al día y aunque su naturaleza es mansa y tranquila, son muy respetados por las mortales patadas que pueden dar.",
            "La visión nocturna de los tigres es hasta seis veces más potente que la de los seres humanos.",
            "Los elefantes son excelentes nadadores, a pesar de tener un peso que generalmente ronda las 3,5 toneladas.",
        ],
    ],
    [
        r"(.*)materiales (.*) construcción(.*)",
        [
            "Los materiales de construcción son muchos y muy variados, por lo que también lo son sus características, pero hay entre ellos una constante: son duraderos.",
            "Las características de los materiales de construcción pueden variar de fabricante en fabricante, aunque siempre deben garantizar unos requisitos mínimos. ",
            "Las normas internacionales más empleadas para regular los materiales de construcción son las normas ISO.  ",
            "En los países europeos, los materiales de construcción están regulados por una serie de códigos y normativas que definen las características que deben cumplir, así como su ámbito de aplicación.",
        ],
    ],
    [
        r"(.*)diseño (.*) (interiores|interior)(.*)",
        [
            "Siempre que voy a otras casas me fijo en cómo están decoradas.",
            "Es increíble el efecto que tiene la decoración en las sensaciones que te produce una habitación.",
            "Encontrar el equilibrio perfecto cuando diseñas tu casa no es nada fácil",
        ],
    ],
    [
        r"(.*)espacio(.*)",
        [
            "Este tema me encanta",
            "¿Sabías que la luz tarda 8 minutos y 17 segundos en viajar desde el Sol hasta la superficie terrestre?",
            "La Tierra rota a una velocidad de 1.609 km/h, pero se desplaza a través del espacio a la increíble velocidad de 107.826 km/h.",
            "Si el Sol midiese tan solo una pulgada de diámetro (2,54 cm), la estrella más cercana se encontraría a 716 km de distancia.",
        ],
    ],
    [
        r"(.*)(especie|especies) (.*) extinción(.*)",
        [
            "¿Sabías que el Dodo se extinguió en 1700?",
            "Entre los doce animales más amenazados de la Tierra se encuentran el oso panda, el oso polar y el rinoceronte.",
            "Más de un millón de especies estarán en peligro de extinción durante las próximas décadas.",
            "Deberíamos ser más conscientes de qué efectos tienen nuestras acciones en el resto de especies.",
        ],
    ],
    [
        r"(.*)!",
        ["veo que hoy estás muy emocional", "necesitas calmar tus emociones", "por qué %1?"],
    ],
    [
        r"(.*) y (.*)",
        ["en esta vida se puede ser de todo, menos pesado", "muy interesante", "ya veo", "%1 %2"],
    ],
    [r"\:\)", ["por qué estás alegre?"]],
    [r"\:\(", ["por qué estás triste?"]],
    [r"\:p", ["qué te parece divertido?"]],
    [r"\:s", ["de qué tiene miedo?"]],
    [r";\)", ["me alegro que te guste"]],
    [
        r"salir",
        ["hasta luego, nos vemos pronto :)", "ha sido un placer hablar contigo. Hasta pronto :)"],
    ],
    [
        r"(.*)",
        [
            "no te he entendido, podrías repetir?",
            "Cambiemos de conversación",
            "podrías explicar eso mejor?",
            "no entiendo eso",
            "no sé bien como contestar a eso",
            "prueba a expresar tu pregunta de otra manera",
            "continúa",
        ],
    ],
]

#######################################################################
############################### ENGLISH ###############################
#######################################################################

reflections_en = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you",
}

pairs_en = (
    (
        r"I need (.*)",
        (
            "Why do you need %1?",
            "Would it really help you to get %1?",
            "Are you sure you need %1?",
        ),
    ),
    (
        r"Why don\'t you (.*)",
        (
            "Do you really think I don't %1?",
            "Perhaps eventually I will %1.",
            "Do you really want me to %1?",
        ),
    ),
    (
        r"Why can\'t I (.*)",
        (
            "Do you think you should be able to %1?",
            "If you could %1, what would you do?",
            "I don't know -- why can't you %1?",
            "Have you really tried?",
        ),
    ),
    (
        r"I can\'t (.*)",
        (
            "How do you know you can't %1?",
            "Perhaps you could %1 if you tried.",
            "What would it take for you to %1?",
        ),
    ),
    (
        r"I am (.*)",
        (
            "Did you come to me because you are %1?",
            "How long have you been %1?",
            "How do you feel about being %1?",
        ),
    ),
    (
        r"I\'m (.*)",
        (
            "How does being %1 make you feel?",
            "Do you enjoy being %1?",
            "Why do you tell me you're %1?",
            "Why do you think you're %1?",
        ),
    ),
    (
        r"Are you (.*)",
        (
            "Why does it matter whether I am %1?",
            "Would you prefer it if I were not %1?",
            "Perhaps you believe I am %1.",
            "I may be %1 -- what do you think?",
        ),
    ),
    (
        r"What (.*)",
        (
            "Why do you ask?",
            "How would an answer to that help you?",
            "What do you think?",
        ),
    ),
    (
        r"How (.*)",
        (
            "How do you suppose?",
            "Perhaps you can answer your own question.",
            "What is it you're really asking?",
        ),
    ),
    (
        r"Because (.*)",
        (
            "Is that the real reason?",
            "What other reasons come to mind?",
            "Does that reason apply to anything else?",
            "If %1, what else must be true?",
        ),
    ),
    (
        r"(.*) sorry (.*)",
        (
            "There are many times when no apology is needed.",
            "What feelings do you have when you apologize?",
        ),
    ),
    (
        r"Hello(.*)",
        (
            "Hello... I'm glad you could drop by today.",
            "Hi there... how are you today?",
            "Hello, how are you feeling today?",
        ),
    ),
    (
        r"I think (.*)",
        ("Do you doubt %1?", "Do you really think so?", "But you're not sure %1?"),
    ),
    (
        r"(.*) friend (.*)",
        (
            "Tell me more about your friends.",
            "When you think of a friend, what comes to mind?",
            "Why don't you tell me about a childhood friend?",
        ),
    ),
    (r"Yes\b[.!?]*$", ("You seem quite sure.", "OK, but can you elaborate a bit?")),
    (
        r"(.*) computer(.*)",
        (
            "Are you really talking about me?",
            "Does it seem strange to talk to a computer?",
            "How do computers make you feel?",
            "Do you feel threatened by computers?",
        ),
    ),
    (
        r"Is it (.*)",
        (
            "Do you think it is %1?",
            "Perhaps it's %1 -- what do you think?",
            "If it were %1, what would you do?",
            "It could well be that %1.",
        ),
    ),
    (
        r"It is (.*)",
        (
            "You seem very certain.",
            "If I told you that it probably isn't %1, what would you feel?",
        ),
    ),
    (
        r"Can you (.*)",
        (
            "What makes you think I can't %1?",
            "If I could %1, then what?",
            "Why do you ask if I can %1?",
        ),
    ),
    (
        r"Can I (.*)",
        (
            "Perhaps you don't want to %1.",
            "Do you want to be able to %1?",
            "If you could %1, would you?",
        ),
    ),
    (
        r"You are (.*)",
        (
            "Why do you think I am %1?",
            "Does it please you to think that I'm %1?",
            "Perhaps you would like me to be %1.",
            "Perhaps you're really talking about yourself?",
        ),
    ),
    (
        r"You\'re (.*)",
        (
            "Why do you say I am %1?",
            "Why do you think I am %1?",
            "Are we talking about you, or me?",
        ),
    ),
    (
        r"I don\'t (.*)",
        ("Don't you really %1?", "Why don't you %1?", "Do you want to %1?"),
    ),
    (
        r"I feel (.*)",
        (
            "Good, tell me more about these feelings.",
            "Do you often feel %1?",
            "When do you usually feel %1?",
            "When you feel %1, what do you do?",
        ),
    ),
    (
        r"I have (.*)",
        (
            "Why do you tell me that you've %1?",
            "Have you really %1?",
            "Now that you have %1, what will you do next?",
        ),
    ),
    (
        r"I would (.*)",
        (
            "Could you explain why you would %1?",
            "Why would you %1?",
            "Who else knows that you would %1?",
        ),
    ),
    (
        r"Is there (.*)",
        (
            "Do you think there is %1?",
            "It's likely that there is %1.",
            "Would you like there to be %1?",
        ),
    ),
    (
        r"My (.*)",
        (
            "I see, your %1.",
            "Why do you say that your %1?",
            "When your %1, how do you feel?",
        ),
    ),
    (
        r"You (.*)",
        (
            "We should be discussing you, not me.",
            "Why do you say that about me?",
            "Why do you care whether I %1?",
        ),
    ),
    (r"Why (.*)", ("Why don't you tell me the reason why %1?", "Why do you think %1?")),
    (
        r"I want (.*)",
        (
            "What would it mean to you if you got %1?",
            "Why do you want %1?",
            "What would you do if you got %1?",
            "If you got %1, then what would you do?",
        ),
    ),
    (
        r"(.*) mother(.*)",
        (
            "Tell me more about your mother.",
            "What was your relationship with your mother like?",
            "How do you feel about your mother?",
            "How does this relate to your feelings today?",
            "Good family relations are important.",
        ),
    ),
    (
        r"(.*) father(.*)",
        (
            "Tell me more about your father.",
            "How did your father make you feel?",
            "How do you feel about your father?",
            "Does your relationship with your father relate to your feelings today?",
            "Do you have trouble showing affection with your family?",
        ),
    ),
    (
        r"(.*) child(.*)",
        (
            "Did you have close friends as a child?",
            "What is your favorite childhood memory?",
            "Do you remember any dreams or nightmares from childhood?",
            "Did the other children sometimes tease you?",
            "How do you think your childhood experiences relate to your feelings today?",
        ),
    ),
    (
        r"(.*)\?",
        (
            "Why do you ask that?",
            "Please consider whether you can answer your own question.",
            "Perhaps the answer lies within yourself?",
            "Why don't you tell me?",
        ),
    ),
    (
        r"quit",
        (
            "Thank you for talking with me.",
            "Good-bye.",
            "Take care. Have a good day!",
        ),
    ),
    (
        r"(.*)",
        (
            "Please tell me more.",
            "Let's change focus a bit... Tell me about your family.",
            "Can you elaborate on that?",
            "Why do you say that %1?",
            "I see.",
            "Very interesting.",
            "%1.",
            "I see.  And what does that tell you?",
            "How does that make you feel?",
            "How do you feel when you say that?",
        ),
    ),
)
