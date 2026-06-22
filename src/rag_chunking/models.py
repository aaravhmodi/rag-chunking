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
    question: str
    answer: str
    source_doc: str
    gold_evidence: str
    question_type: str = "unknown"
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
    avg_chunk_count: float
    avg_chunk_length_chars: float
    chunking_latency_ms: float
    retrieval_latency_ms: float
