"""Lightweight keyword-based interest scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


DEFAULT_INTEREST_RANGES: dict[int, set[str]] = {
    20: {"viajes", "lugar", "avion"},
    50: {"animal"},
    70: {"cucaracha", "enjambre"},
    100: {"diabetes", "glucosa", "dolor", "pinchazo", "mareo"},
}

DEFAULT_IGNORE_WORDS = {
    "me",
    "es",
    "si",
    "si",
    "yo",
    "la",
    "el",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "y",
    "o",
    "a",
    "ante",
    "bajo",
    "con",
    "contra",
    "de",
    "desde",
    "en",
    "entre",
    "hacia",
    "hasta",
    "para",
    "por",
    "segun",
    "sin",
    "sobre",
    "tras",
    "durante",
    "mediante",
    "excepto",
}


@dataclass
class InterestScorer:
    ranges: dict[int, set[str]] = field(default_factory=lambda: dict(DEFAULT_INTEREST_RANGES))
    ignore_words: set[str] = field(default_factory=lambda: set(DEFAULT_IGNORE_WORDS))

    def score(self, sentence: str) -> int:
        score = 0
        for word in self._tokenize(sentence):
            stemmed_word = self._stem(word)
            for candidate_score, words_set in self.ranges.items():
                if any(stemmed_word in token for phrase in words_set for token in re.findall(r"\w+", phrase)):
                    score = max(score, candidate_score)
        return score

    def _tokenize(self, sentence: str) -> list[str]:
        normalized = sentence.lower()
        ignore = set(self.ignore_words)
        ignore.update(word for word in normalized.split() if len(word) == 2)
        words = re.sub(r"[^\w]", " ", normalized).split()
        return sorted({word for word in words if word not in ignore})

    @staticmethod
    def _stem(word: str) -> str:
        suffixes = (
            "iendo",
            "ando",
            "ado",
            "ido",
            "ar",
            "er",
            "ir",
            "as",
            "os",
            "es",
            "s",
            "a",
            "o",
            "e",
        )
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix):
                return word[: -len(suffix)]
        return word


def score_interest(sentence: str) -> int:
    return InterestScorer().score(sentence)

