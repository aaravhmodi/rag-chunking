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
    first_evidence_rank,
    first_relevant_rank,
    has_evidence_span,
    is_answerable,
    is_relevant,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_chunking.llm import build_llm_judge
from rag_chunking.models import Chunk, Document, ExperimentResult, QuestionDiagnostic, QuestionExample, RetrievalResult
from rag_chunking.retrieval import build_retriever


def run_experiment(
    documents: list[Document],
    questions: list[QuestionExample],
    strategy: str,
    top_k: int = 5,
    cache_dir: str | Path | None = None,
    retriever_backend: str = "lexical",
    embedding_backend: str | None = None,
    llm_judge_backend: str | None = None,
) -> ExperimentResult:
    chunks, chunking_latency_ms = chunk_documents(documents, strategy, cache_dir=cache_dir, embedding_backend=embedding_backend)
    return evaluate_experiment(
        chunks,
        questions,
        strategy,
        top_k=top_k,
        chunking_latency_ms=chunking_latency_ms,
        retriever_backend=retriever_backend,
        embedding_backend=embedding_backend,
        llm_judge_backend=llm_judge_backend,
    )


def run_grouped_experiments(
    documents: list[Document],
    questions: list[QuestionExample],
    strategy: str,
    top_k: int = 5,
    group_fields: tuple[str, ...] = ("dataset", "split"),
    cache_dir: str | Path | None = None,
    retriever_backend: str = "lexical",
    embedding_backend: str | None = None,
    llm_judge_backend: str | None = None,
) -> dict[str, ExperimentResult]:
    chunks, chunking_latency_ms = chunk_documents(documents, strategy, cache_dir=cache_dir, embedding_backend=embedding_backend)
    retriever = build_retriever(chunks, retriever_spec=retriever_backend, embedding_backend=embedding_backend)
    llm_judge = build_llm_judge(llm_judge_backend)
    retrievals = {question.question_id: retriever.retrieve(question.question, top_k=top_k) for question in questions}
    llm_grades = {question.question_id: llm_judge.grade(question, retrievals[question.question_id]) for question in questions} if llm_judge else {}
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
            llm_grades=llm_grades,
            chunk_counts=list(chunk_counts_by_doc.values()),
            chunk_lengths=[len(chunk.text) for chunk in chunks],
            chunking_latency_ms=chunking_latency_ms,
            top_k=top_k,
            retriever_name=retriever.name,
        )
        for label, group_questions in grouped_questions.items()
    }


def chunk_documents(
    documents: list[Document],
    strategy: str,
    cache_dir: str | Path | None = None,
    embedding_backend: str | None = None,
) -> tuple[list[Chunk], float]:
    cache_strategy = strategy if not embedding_backend or strategy != "semantic" else f"{strategy}:{embedding_backend}"
    if cache_dir is not None:
        cached_chunks = load_cached_chunks(cache_dir, documents, cache_strategy)
        if cached_chunks is not None:
            return cached_chunks, 0.0

    chunking_start = time.perf_counter()
    chunks: list[Chunk] = []
    for document in documents:
        doc_chunks = build_chunks(document, strategy, embedding_backend=embedding_backend)
        chunks.extend(doc_chunks)
    chunking_latency_ms = (time.perf_counter() - chunking_start) * 1000
    if cache_dir is not None:
        save_cached_chunks(cache_dir, documents, cache_strategy, chunks)
    return chunks, chunking_latency_ms


def evaluate_experiment(
    chunks: list[Chunk],
    questions: list[QuestionExample],
    strategy: str,
    top_k: int,
    chunking_latency_ms: float,
    retriever_backend: str = "lexical",
    embedding_backend: str | None = None,
    llm_judge_backend: str | None = None,
) -> ExperimentResult:
    retriever = build_retriever(chunks, retriever_spec=retriever_backend, embedding_backend=embedding_backend)
    retrievals = {question.question_id: retriever.retrieve(question.question, top_k=top_k) for question in questions}
    llm_judge = build_llm_judge(llm_judge_backend)
    llm_grades = {question.question_id: llm_judge.grade(question, retrievals[question.question_id]) for question in questions} if llm_judge else {}
    chunk_counts_by_doc = defaultdict(int)
    for chunk in chunks:
        chunk_counts_by_doc[chunk.doc_id] += 1
    return summarize_experiment(
        strategy=strategy,
        questions=questions,
        retrievals=retrievals,
        llm_grades=llm_grades,
        chunk_counts=list(chunk_counts_by_doc.values()),
        chunk_lengths=[len(chunk.text) for chunk in chunks],
        chunking_latency_ms=chunking_latency_ms,
        top_k=top_k,
        retriever_name=retriever.name,
    )


def collect_question_diagnostics(
    chunks: list[Chunk],
    questions: list[QuestionExample],
    strategy: str,
    top_k: int,
    retriever_backend: str = "lexical",
    embedding_backend: str | None = None,
    llm_judge_backend: str | None = None,
) -> list[QuestionDiagnostic]:
    retriever = build_retriever(chunks, retriever_spec=retriever_backend, embedding_backend=embedding_backend)
    llm_judge = build_llm_judge(llm_judge_backend)
    diagnostics: list[QuestionDiagnostic] = []
    for question in questions:
        results = retriever.retrieve(question.question, top_k=top_k)
        grade = llm_judge.grade(question, results) if llm_judge else None
        top1 = results[0] if results else None
        relevant_rank = first_relevant_rank(results, question)
        evidence_rank = first_evidence_rank(results, question)
        evidence_question = has_evidence_span(question)
        evidence_covered = evidence_rank is not None
        relevant_retrieved = relevant_rank is not None
        answer_match = answer_exact_match(results, question)
        if evidence_question:
            failure_mode = "success_evidence" if evidence_covered else "missed_evidence_span" if relevant_retrieved else "missed_relevant"
        else:
            failure_mode = "success_relevant" if relevant_retrieved else "missed_relevant"
        diagnostics.append(
            QuestionDiagnostic(
                strategy=strategy,
                retriever=retriever.name,
                question_id=question.question_id,
                question=question.question,
                dataset=str(question.metadata.get("dataset", "")),
                split=str(question.metadata.get("split", "")),
                source_doc=question.source_doc,
                top_k=top_k,
                total_retrieved=len(results),
                top1_doc_id=top1.chunk.doc_id if top1 else "",
                top1_score=top1.score if top1 else 0.0,
                top1_relevant=is_relevant(top1, question) if top1 else False,
                first_relevant_rank=relevant_rank,
                relevant_retrieved=relevant_retrieved,
                answer_exact_match=answer_match,
                llm_answer_score=grade.answer_score if grade else 0.0,
                hallucination_score=grade.hallucination_score if grade else 0.0,
                generated_answer=grade.generated_answer if grade else "",
                evidence_question=evidence_question,
                first_evidence_rank=evidence_rank,
                evidence_span_covered=evidence_covered,
                failure_mode=failure_mode,
            )
        )
    return diagnostics


def summarize_experiment(
    strategy: str,
    questions: list[QuestionExample],
    retrievals: dict[str, list[RetrievalResult]],
    llm_grades: dict[str, object],
    chunk_counts: list[int],
    chunk_lengths: list[int],
    chunking_latency_ms: float,
    top_k: int,
    retriever_name: str,
) -> ExperimentResult:
    retrieval_start = time.perf_counter()
    recall_scores = []
    rr_scores = []
    ndcg_scores = []
    answer_scores = []
    answerable_question_count = 0
    evidence_scores = []
    evidence_question_count = 0
    llm_answer_scores = []
    hallucination_scores = []
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
        grade = llm_grades.get(question.question_id)
        if grade is not None:
            llm_answer_scores.append(grade.answer_score)
            hallucination_scores.append(grade.hallucination_score)
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    return ExperimentResult(
        strategy=strategy,
        retriever=retriever_name,
        top_k=top_k,
        recall_at_k=statistics.fmean(recall_scores) if recall_scores else 0.0,
        mrr=statistics.fmean(rr_scores) if rr_scores else 0.0,
        ndcg_at_k=statistics.fmean(ndcg_scores) if ndcg_scores else 0.0,
        answer_exact_match=statistics.fmean(answer_scores) if answer_scores else 0.0,
        answerable_question_count=answerable_question_count,
        evidence_span_recall_at_k=statistics.fmean(evidence_scores) if evidence_scores else 0.0,
        evidence_question_count=evidence_question_count,
        llm_judged_question_count=len(llm_answer_scores),
        llm_answer_score=statistics.fmean(llm_answer_scores) if llm_answer_scores else 0.0,
        hallucination_rate=statistics.fmean(hallucination_scores) if hallucination_scores else 0.0,
        total_question_count=len(questions),
        avg_chunk_count=statistics.fmean(chunk_counts) if chunk_counts else 0.0,
        avg_chunk_length_chars=statistics.fmean(chunk_lengths) if chunk_lengths else 0.0,
        chunking_latency_ms=chunking_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
    )
