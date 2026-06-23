from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QuestionExample:
    question_id: str
    question: str
    answer: str = ""
    source_doc: str = ""
    gold_evidence: str = ""
    alternative_answers: list[str] = field(default_factory=list)
    relevant_doc_ids: list[str] = field(default_factory=list)
    evidence_start: int | None = None
    evidence_end: int | None = None
    question_type: str = "unknown"
    difficulty: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass(slots=True)
class ExperimentResult:
    strategy: str
    top_k: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    answer_exact_match: float
    answerable_question_count: int
    total_question_count: int
    avg_chunk_count: float
    avg_chunk_length_chars: float
    chunking_latency_ms: float
    retrieval_latency_ms: float
