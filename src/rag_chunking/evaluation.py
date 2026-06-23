from __future__ import annotations

import math
from collections.abc import Iterable

from rag_chunking.models import QuestionExample, RetrievalResult


def _normalize_candidates(values: Iterable[str]) -> list[str]:
    return [value.lower() for value in values if value]


def is_answerable(question: QuestionExample) -> bool:
    return bool(question.answer or question.alternative_answers)


def has_evidence_span(question: QuestionExample) -> bool:
    return (
        question.evidence_start is not None
        and question.evidence_end is not None
        and question.source_doc != ""
        and question.evidence_end > question.evidence_start
    )


def is_relevant(result: RetrievalResult, question: QuestionExample) -> bool:
    if question.relevant_doc_ids:
        return result.chunk.doc_id in set(question.relevant_doc_ids)

    haystack = result.chunk.text.lower()
    evidence_match = bool(question.gold_evidence) and question.gold_evidence.lower() in haystack
    answer_candidates = _normalize_candidates([question.answer, *question.alternative_answers])
    answer_match = any(candidate in haystack for candidate in answer_candidates)
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
    if not is_answerable(question):
        return 0.0
    combined = " ".join(result.chunk.text for result in results).lower()
    candidates = _normalize_candidates([question.answer, *question.alternative_answers])
    return 1.0 if any(candidate in combined for candidate in candidates) else 0.0


def evidence_span_recall_at_k(results: list[RetrievalResult], question: QuestionExample) -> float:
    if not has_evidence_span(question):
        return 0.0
    for result in results:
        if result.chunk.doc_id != question.source_doc:
            continue
        if result.chunk.start_char <= question.evidence_start and result.chunk.end_char >= question.evidence_end:
            return 1.0
    return 0.0
