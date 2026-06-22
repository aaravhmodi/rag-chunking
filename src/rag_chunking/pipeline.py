from __future__ import annotations

import statistics
import time

from rag_chunking.chunkers import build_chunks
from rag_chunking.evaluation import answer_exact_match, ndcg_at_k, recall_at_k, reciprocal_rank
from rag_chunking.models import Document, ExperimentResult, QuestionExample
from rag_chunking.retrieval import LexicalRetriever


def run_experiment(
    documents: list[Document],
    questions: list[QuestionExample],
    strategy: str,
    top_k: int = 5,
) -> ExperimentResult:
    chunking_start = time.perf_counter()
    chunks = []
    chunk_counts: list[int] = []
    chunk_lengths: list[int] = []
    for document in documents:
        doc_chunks = build_chunks(document, strategy)
        chunks.extend(doc_chunks)
        chunk_counts.append(len(doc_chunks))
        chunk_lengths.extend(len(chunk.text) for chunk in doc_chunks)
    chunking_latency_ms = (time.perf_counter() - chunking_start) * 1000

    retriever = LexicalRetriever(chunks)
    retrieval_start = time.perf_counter()
    recall_scores = []
    rr_scores = []
    ndcg_scores = []
    answer_scores = []
    for question in questions:
        results = retriever.retrieve(question.question, top_k=top_k)
        recall_scores.append(recall_at_k(results, question))
        rr_scores.append(reciprocal_rank(results, question))
        ndcg_scores.append(ndcg_at_k(results, question))
        answer_scores.append(answer_exact_match(results, question))
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    return ExperimentResult(
        strategy=strategy,
        top_k=top_k,
        recall_at_k=statistics.fmean(recall_scores) if recall_scores else 0.0,
        mrr=statistics.fmean(rr_scores) if rr_scores else 0.0,
        ndcg_at_k=statistics.fmean(ndcg_scores) if ndcg_scores else 0.0,
        answer_exact_match=statistics.fmean(answer_scores) if answer_scores else 0.0,
        avg_chunk_count=statistics.fmean(chunk_counts) if chunk_counts else 0.0,
        avg_chunk_length_chars=statistics.fmean(chunk_lengths) if chunk_lengths else 0.0,
        chunking_latency_ms=chunking_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
    )
