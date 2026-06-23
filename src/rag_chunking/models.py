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
    backend: str = ""


@dataclass(slots=True)
class ExperimentResult:
    strategy: str
    retriever: str
    top_k: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    answer_exact_match: float
    answerable_question_count: int
    evidence_span_recall_at_k: float
    evidence_question_count: int
    llm_judged_question_count: int
    llm_answer_score: float
    hallucination_rate: float
    total_question_count: int
    avg_chunk_count: float
    avg_chunk_length_chars: float
    chunking_latency_ms: float
    retrieval_latency_ms: float


@dataclass(slots=True)
class QuestionDiagnostic:
    strategy: str
    retriever: str
    question_id: str
    question: str
    dataset: str
    split: str
    source_doc: str
    top_k: int
    total_retrieved: int
    top1_doc_id: str
    top1_score: float
    top1_relevant: bool
    first_relevant_rank: int | None
    relevant_retrieved: bool
    answer_exact_match: float
    llm_answer_score: float
    hallucination_score: float
    generated_answer: str
    evidence_question: bool
    first_evidence_rank: int | None
    evidence_span_covered: bool
    failure_mode: str


@dataclass(slots=True)
class LLMGrade:
    answer_score: float
    hallucination_score: float
    generated_answer: str
    rationale: str = ""


@dataclass(slots=True)
class SignificanceComparison:
    metric: str
    baseline_strategy: str
    candidate_strategy: str
    dataset: str
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    p_value: float
    effect_size: float
    significant: bool
    test_name: str
