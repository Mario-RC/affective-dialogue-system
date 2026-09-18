"""Local multilingual rule-based toxicity checks.

The local rule set is adapted from the thesis-era Ray toxicity prototype. It is
intended as a fast safety gate and as a deterministic fallback when heavier
moderation models are unavailable.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass, field
from importlib import resources

from affective_dialogue_system.safety.schemas import SafetyCategory, SafetyResult

SPECIAL_CHARS = '!"#$%&()+,./:;<=>?[\\]^_`{|}~'

WHITELIST_PHRASES = {
    "kill bill",
    "to kill a mockingbird",
    "moby dick",
    "dick van dyke",
    "sex education",
    "sex and the city",
    "suicide squad",
    "hells kitchen",
    "hell's kitchen",
    "die hard",
}

REMOVE_FROM_BLACKLIST = {
    "god",
    "gods",
    "ginger",
    "gay",
    "lesbian",
    "lesbians",
    "balls",
    "crabs",
    "dammit",
    "damn",
    "darn",
    "dirty",
    "murder",
    "omg",
    "organ",
    "organs",
    "queer",
    "sexes",
    "psycho",
    "ugly",
    "trashy",
    "crappy",
    "shitty",
    "sucks",
    "stupid",
}

ADD_TO_BLACKLIST = {
    "pornographic",
    "prostitution",
    "blackface",
    "misogyny",
    "talk dirty",
    "dirty talk",
}

CATEGORY_PATTERNS: tuple[tuple[SafetyCategory, tuple[str, ...]], ...] = (
    (
        SafetyCategory.SELF_HARM,
        (
            r"\b(self[- ]?harm|kill myself|hurt myself|suicide)\b",
            r"\b(suicidarme|hacerme dano|quitarme la vida|autolesion)\b",
        ),
    ),
    (
        SafetyCategory.THREAT,
        (
            r"\b(i will kill|i am going to kill|i will hurt|i will attack)\b",
            r"\b(te voy a matar|voy a matarte|te hare dano|voy a atacarte)\b",
        ),
    ),
    (
        SafetyCategory.CRIMINAL_PLANNING,
        (
            r"\b(how to rob|how to steal|plan a robbery|hide a body)\b",
            r"\b(como robar|planear un robo|ocultar un cuerpo)\b",
        ),
    ),
    (
        SafetyCategory.GUNS_AND_ILLEGAL_WEAPONS,
        (
            r"\b(illegal weapon|unregistered gun|make a bomb)\b",
            r"\b(arma ilegal|pistola ilegal|fabricar una bomba)\b",
        ),
    ),
    (
        SafetyCategory.REGULATED_OR_CONTROLLED_SUBSTANCES,
        (
            r"\b(sell drugs|make cocaine|make meth)\b",
            r"\b(vender droga|fabricar cocaina|fabricar metanfetamina)\b",
        ),
    ),
    (
        SafetyCategory.SEXUAL_EXPLICIT,
        (
            r"\b(explicit sexual|pornographic|sexual act)\b",
            r"\b(sexual explicito|pornografico|acto sexual)\b",
        ),
    ),
    (
        SafetyCategory.VIOLENCE_AND_HATE,
        (
            r"\b(exterminate|violent hate|hate all)\b",
            r"\b(exterminar|odio violento|odio a todos)\b",
        ),
    ),
    (
        SafetyCategory.INSULT,
        (
            r"\b(idiot|moron|shut up)\b",
            r"\b(idiota|imbecil|callate|gilipollas)\b",
        ),
    ),
)


@dataclass
class LocalRuleSafetyDetector:
    offensive_phrases: set[str] = field(default_factory=set)
    prohibited_terms: set[str] = field(default_factory=set)

    @classmethod
    def from_package_data(cls) -> LocalRuleSafetyDetector:
        offensive = _load_resource_lines("offensive_phrases_preprocessed.txt")
        prohibited = _load_resource_lines("prohibited_terms.txt")
        offensive = (offensive - {_normalize_text(item) for item in REMOVE_FROM_BLACKLIST}) | {
            _normalize_text(item) for item in ADD_TO_BLACKLIST
        }
        return cls(offensive_phrases=offensive, prohibited_terms=prohibited)

    def check(self, text: str) -> SafetyResult:
        normalized = _remove_whitelisted_phrases(_normalize_text(text))
        if not normalized:
            return SafetyResult.safe()

        for category, patterns in CATEGORY_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, normalized):
                    return SafetyResult.flagged_result(
                        category,
                        source="local_pattern",
                        matched_text=pattern,
                    )

        matched_prohibited = self._contains_phrase(normalized, self.prohibited_terms)
        if matched_prohibited:
            return SafetyResult.flagged_result(
                SafetyCategory.PROHIBITED_TERM,
                source="local_prohibited_terms",
                matched_text=matched_prohibited,
            )

        matched_offensive = self._contains_phrase(normalized, self.offensive_phrases)
        if matched_offensive:
            return SafetyResult.flagged_result(
                SafetyCategory.PROHIBITED_TERM,
                source="local_offensive_phrases",
                matched_text=matched_offensive,
            )

        return SafetyResult.safe()

    def _contains_phrase(self, text: str, phrases: set[str]) -> str | None:
        text_variants = _text_variants(text)
        max_phrase_len = max((len(phrase.split()) for phrase in phrases), default=1)
        for variant in text_variants:
            variant = _remove_whitelisted_phrases(variant)
            tokens = variant.split()
            if not tokens:
                continue
            for ngram_len in range(1, max_phrase_len + 1):
                for index in range(0, len(tokens) - ngram_len + 1):
                    ngram = " ".join(tokens[index : index + ngram_len])
                    if ngram in phrases:
                        return ngram
        return None


def _load_resource_lines(filename: str) -> set[str]:
    data = resources.files("affective_dialogue_system.safety.data").joinpath(filename)
    with data.open("r", encoding="utf-8") as handle:
        return {_normalize_text(line) for line in handle if _normalize_text(line)}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _text_variants(text: str) -> tuple[str, ...]:
    no_special = text.translate({ord(char): "" for char in SPECIAL_CHARS})
    punctuation_as_space = " ".join(
        text.translate({ord(char): " " for char in string.punctuation}).split()
    )
    alpha_numeric = " ".join(re.sub(r"[^\w]", " ", text).split())
    return tuple(dict.fromkeys((text, no_special, punctuation_as_space, alpha_numeric)))


def _remove_whitelisted_phrases(text: str) -> str:
    for phrase in sorted(WHITELIST_PHRASES, key=lambda item: (-len(item), item)):
        text = re.sub(r"(?<!\w)" + re.escape(_normalize_text(phrase)) + r"(?!\w)", " ", text)
    return " ".join(text.split())
