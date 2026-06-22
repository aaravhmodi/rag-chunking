from __future__ import annotations

import math

from rag_chunking.models import QuestionExample, RetrievalResult


def is_relevant(result: RetrievalResult, question: QuestionExample) -> bool:
    haystack = result.chunk.text.lower()
    evidence_match = question.gold_evidence.lower() in haystack
    answer_match = question.answer.lower() in haystack
    source_match = result.chunk.doc_id == question.source_doc
    return source_match and (evidence_match or answer_match)


def recall_at_k(results: list[RetrievalResult], question: QuestionExample) -> float:
    return 1.0 if any(is_relevant(result, question) for result in results) else 0.0


def reciprocal_rank(results: list[RetrievalResult], question: QuestionExample) -> float:
    for rank, result in enumerate(results, start=1):
        if is_relevant(result, question):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[RetrievalResult], question: QuestionExample) -> float:
    dcg = 0.0
    for rank, result in enumerate(results, start=1):
        rel = 1.0 if is_relevant(result, question) else 0.0
        dcg += rel / math.log2(rank + 1)
    ideal_dcg = 1.0
    return dcg / ideal_dcg if ideal_dcg else 0.0


def answer_exact_match(results: list[RetrievalResult], question: QuestionExample) -> float:
    combined = " ".join(result.chunk.text for result in results).lower()
    return 1.0 if question.answer.lower() in combined else 0.0
