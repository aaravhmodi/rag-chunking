from __future__ import annotations

import math
import re
from collections import Counter


WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts if parts else [text.strip()] if text.strip() else []


def split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
    return parts if parts else [text.strip()] if text.strip() else []


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def token_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))
