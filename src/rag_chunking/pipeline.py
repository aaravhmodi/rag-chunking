from __future__ import annotations

import statistics
import time
from collections import defaultdict
from pathlib import Path

from rag_chunking.cache import load_cached_chunks, save_cached_chunks
from rag_chunking.chunkers import build_chunks
from rag_chunking.evaluation import (
    answer_exact_match,
    evidence_span_recall_at_k,
    has_evidence_span,
    is_answerable,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_chunking.models import Chunk, Document, ExperimentResult, QuestionExample, RetrievalResult
from rag_chunking.retrieval import LexicalRetriever


def run_experiment(
    documents: list[Document],
    questions: list[QuestionExample],
    strategy: str,
    top_k: int = 5,
    cache_dir: str | Path | None = None,
) -> ExperimentResult:
    chunks, chunking_latency_ms = chunk_documents(documents, strategy, cache_dir=cache_dir)
    return evaluate_experiment(chunks, questions, strategy, top_k=top_k, chunking_latency_ms=chunking_latency_ms)


def run_grouped_experiments(
    documents: list[Document],
    questions: list[QuestionExample],
    strategy: str,
    top_k: int = 5,
    group_fields: tuple[str, ...] = ("dataset", "split"),
    cache_dir: str | Path | None = None,
) -> dict[str, ExperimentResult]:
    chunks, chunking_latency_ms = chunk_documents(documents, strategy, cache_dir=cache_dir)
    retriever = LexicalRetriever(chunks)
    retrievals = {question.question_id: retriever.retrieve(question.question, top_k=top_k) for question in questions}
    chunk_counts_by_doc = defaultdict(int)
    for chunk in chunks:
        chunk_counts_by_doc[chunk.doc_id] += 1

    grouped_questions: defaultdict[str, list[QuestionExample]] = defaultdict(list)
    grouped_questions["overall"].extend(questions)
    for question in questions:
        for field in group_fields:
            value = question.metadata.get(field)
            if value:
                grouped_questions[f"{field}={value}"].append(question)

    return {
        label: summarize_experiment(
            strategy=strategy,
            questions=group_questions,
            retrievals=retrievals,
            chunk_counts=list(chunk_counts_by_doc.values()),
            chunk_lengths=[len(chunk.text) for chunk in chunks],
            chunking_latency_ms=chunking_latency_ms,
            top_k=top_k,
        )
        for label, group_questions in grouped_questions.items()
    }


def chunk_documents(
    documents: list[Document],
    strategy: str,
    cache_dir: str | Path | None = None,
) -> tuple[list[Chunk], float]:
    if cache_dir is not None:
        cached_chunks = load_cached_chunks(cache_dir, documents, strategy)
        if cached_chunks is not None:
            return cached_chunks, 0.0

    chunking_start = time.perf_counter()
    chunks: list[Chunk] = []
    for document in documents:
        doc_chunks = build_chunks(document, strategy)
        chunks.extend(doc_chunks)
    chunking_latency_ms = (time.perf_counter() - chunking_start) * 1000
    if cache_dir is not None:
        save_cached_chunks(cache_dir, documents, strategy, chunks)
    return chunks, chunking_latency_ms


def evaluate_experiment(
    chunks: list[Chunk],
    questions: list[QuestionExample],
    strategy: str,
    top_k: int,
    chunking_latency_ms: float,
) -> ExperimentResult:
    retriever = LexicalRetriever(chunks)
    retrievals = {question.question_id: retriever.retrieve(question.question, top_k=top_k) for question in questions}
    chunk_counts_by_doc = defaultdict(int)
    for chunk in chunks:
        chunk_counts_by_doc[chunk.doc_id] += 1
    return summarize_experiment(
        strategy=strategy,
        questions=questions,
        retrievals=retrievals,
        chunk_counts=list(chunk_counts_by_doc.values()),
        chunk_lengths=[len(chunk.text) for chunk in chunks],
        chunking_latency_ms=chunking_latency_ms,
        top_k=top_k,
    )


def summarize_experiment(
    strategy: str,
    questions: list[QuestionExample],
    retrievals: dict[str, list[RetrievalResult]],
    chunk_counts: list[int],
    chunk_lengths: list[int],
    chunking_latency_ms: float,
    top_k: int,
) -> ExperimentResult:
    retrieval_start = time.perf_counter()
    recall_scores = []
    rr_scores = []
    ndcg_scores = []
    answer_scores = []
    answerable_question_count = 0
    evidence_scores = []
    evidence_question_count = 0
    for question in questions:
        results = retrievals[question.question_id]
        recall_scores.append(recall_at_k(results, question))
        rr_scores.append(reciprocal_rank(results, question))
        ndcg_scores.append(ndcg_at_k(results, question))
        if is_answerable(question):
            answer_scores.append(answer_exact_match(results, question))
            answerable_question_count += 1
        if has_evidence_span(question):
            evidence_scores.append(evidence_span_recall_at_k(results, question))
            evidence_question_count += 1
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    return ExperimentResult(
        strategy=strategy,
        top_k=top_k,
        recall_at_k=statistics.fmean(recall_scores) if recall_scores else 0.0,
        mrr=statistics.fmean(rr_scores) if rr_scores else 0.0,
        ndcg_at_k=statistics.fmean(ndcg_scores) if ndcg_scores else 0.0,
        answer_exact_match=statistics.fmean(answer_scores) if answer_scores else 0.0,
        answerable_question_count=answerable_question_count,
        evidence_span_recall_at_k=statistics.fmean(evidence_scores) if evidence_scores else 0.0,
        evidence_question_count=evidence_question_count,
        total_question_count=len(questions),
        avg_chunk_count=statistics.fmean(chunk_counts) if chunk_counts else 0.0,
        avg_chunk_length_chars=statistics.fmean(chunk_lengths) if chunk_lengths else 0.0,
        chunking_latency_ms=chunking_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
    )
