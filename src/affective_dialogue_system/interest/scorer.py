"""Lightweight keyword-based interest scoring."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

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
    ranges: dict[int, set[str]] = field(
        default_factory=lambda: {
            score: set(words) for score, words in DEFAULT_INTEREST_RANGES.items()
        }
    )
    ignore_words: set[str] = field(default_factory=lambda: set(DEFAULT_IGNORE_WORDS))

    def score(self, sentence: str) -> int:
        score = 0
        for word in self._tokenize(sentence):
            stemmed_word = self._stem(word)
            for candidate_score, words_set in self.ranges.items():
                if any(
                    stemmed_word == self._stem(token)
                    for phrase in words_set
                    for token in self._tokenize(phrase)
                ):
                    score = max(score, candidate_score)
        return score

    def _tokenize(self, sentence: str) -> list[str]:
        normalized = unicodedata.normalize("NFKD", sentence.lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        return sorted(
            {
                word
                for word in re.findall(r"\w+", normalized)
                if len(word) > 2 and word not in self.ignore_words
            }
        )

    @staticmethod
    def _stem(word: str) -> str:
        # Deliberately small Spanish heuristic; match whole normalized stems.
        if word.endswith("es") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 3:
            word = word[:-1]
        for suffix in ("iendo", "ando", "ado", "ido", "ar", "er", "ir", "a", "o", "e"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word


def score_interest(sentence: str) -> int:
    return InterestScorer().score(sentence)
