from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_chunking.loaders import load_documents, load_questions
from rag_chunking.pipeline import collect_question_diagnostics, run_experiment, run_grouped_experiments, chunk_documents
from rag_chunking.models import Chunk
from rag_chunking.reporting import render_markdown_report, write_csv_results, write_diagnostics_csv, write_svg_plots
from rag_chunking.significance import collect_dataset_metric_vectors, compare_strategies_by_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG chunking benchmark experiments.")
    parser.add_argument("--documents", required=True, help="Directory containing .txt documents.")
    parser.add_argument("--questions", required=True, help="Path to JSONL question file.")
    parser.add_argument("--strategies", nargs="+", required=True, help="Chunking strategies to evaluate.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--output", help="Optional path to write JSON results.")
    parser.add_argument("--csv-output", help="Optional path to write CSV results.")
    parser.add_argument("--report-output", help="Optional path to write a research-style Markdown report.")
    parser.add_argument("--plots-dir", help="Optional directory to write SVG charts.")
    parser.add_argument("--title", default="RAG Chunking Benchmark Report", help="Title for the Markdown report.")
    parser.add_argument("--group-by", nargs="*", default=["dataset", "split"], help="Question metadata fields to slice results by.")
    parser.add_argument("--question-split", action="append", help="Restrict evaluation to one or more metadata split values.")
    parser.add_argument("--question-dataset", action="append", help="Restrict evaluation to one or more metadata dataset values.")
    parser.add_argument("--max-documents", type=int, help="Optional cap on the number of loaded documents after filtering.")
    parser.add_argument("--max-questions", type=int, help="Optional cap on the number of loaded questions after filtering.")
    parser.add_argument("--cache-dir", help="Optional directory for persisted chunk caches keyed by strategy and document contents.")
    parser.add_argument("--diagnostics-output", help="Optional path to write per-question diagnostics as CSV.")
    parser.add_argument("--retriever-backend", default="lexical", choices=["lexical", "embedding", "hybrid"], help="Retriever backend to use.")
    parser.add_argument("--embedding-backend", default="hash", help="Embedding backend spec for embedding retrievers or semantic chunking.")
    parser.add_argument("--llm-judge-backend", help="Optional LLM grading backend. Use `heuristic` or `openai[:model]`.")
    parser.add_argument("--significance-output", help="Optional path to write cross-dataset significance comparisons as JSON.")
    parser.add_argument("--significance-baseline", help="Optional baseline strategy for significance testing. Defaults to the first strategy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _status("loading documents and questions")
    documents = load_documents(args.documents)
    questions = load_questions(args.questions)
    questions = [
        question
        for question in questions
        if (not args.question_split or question.metadata.get("split") in args.question_split)
        and (not args.question_dataset or question.metadata.get("dataset") in args.question_dataset)
    ]
    if args.question_dataset:
        documents = [document for document in documents if _document_matches_dataset_filter(document.doc_id, set(args.question_dataset))]
    if args.max_documents is not None:
        documents = documents[: args.max_documents]
        allowed_doc_ids = {document.doc_id for document in documents}
        questions = [question for question in questions if _question_matches_loaded_documents(question, allowed_doc_ids)]
    if args.max_questions is not None:
        questions = questions[: args.max_questions]
    _status(f"loaded {len(documents)} documents and {len(questions)} questions")
    strategy_chunks: dict[str, tuple[list[Chunk], float]] = {}
    for strategy in args.strategies:
        _status(f"chunking strategy {strategy}")
        strategy_chunks[strategy] = chunk_documents(documents, strategy, cache_dir=args.cache_dir, embedding_backend=args.embedding_backend)

    _status("running overall evaluations")
    results = [
        run_experiment(
            documents,
            questions,
            strategy,
            top_k=args.top_k,
            cache_dir=args.cache_dir,
            retriever_backend=args.retriever_backend,
            embedding_backend=args.embedding_backend,
            llm_judge_backend=args.llm_judge_backend,
            chunks=strategy_chunks[strategy][0],
            chunking_latency_ms=strategy_chunks[strategy][1],
        )
        for strategy in args.strategies
    ]
    _status("running grouped evaluations")
    grouped_results = {
        strategy: run_grouped_experiments(
            documents,
            questions,
            strategy,
            top_k=args.top_k,
            group_fields=tuple(args.group_by),
            cache_dir=args.cache_dir,
            retriever_backend=args.retriever_backend,
            embedding_backend=args.embedding_backend,
            llm_judge_backend=args.llm_judge_backend,
            chunks=strategy_chunks[strategy][0],
            chunking_latency_ms=strategy_chunks[strategy][1],
        )
        for strategy in args.strategies
    }
    diagnostics = []
    if args.diagnostics_output or args.report_output:
        _status("collecting diagnostics")
        for strategy in args.strategies:
            chunks, chunking_latency_ms = strategy_chunks[strategy]
            diagnostics.extend(
                collect_question_diagnostics(
                    chunks,
                    questions,
                    strategy,
                    top_k=args.top_k,
                    retriever_backend=args.retriever_backend,
                    embedding_backend=args.embedding_backend,
                    llm_judge_backend=args.llm_judge_backend,
                    chunking_latency_ms=chunking_latency_ms,
                )
            )
    significance = []
    if args.significance_output and diagnostics:
        _status("computing significance tests")
        diagnostics_by_strategy: dict[str, list[dict[str, object]]] = {}
        for strategy in args.strategies:
            diagnostics_by_strategy[strategy] = [
                {
                    "dataset": item.dataset or "overall",
                    "recall_at_k": 1.0 if item.relevant_retrieved else 0.0,
                    "answer_exact_match": item.answer_exact_match,
                    "llm_answer_score": item.llm_answer_score,
                    "hallucination_score": item.hallucination_score,
                    "evidence_span_recall_at_k": 1.0 if item.evidence_span_covered else 0.0,
                }
                for item in diagnostics
                if item.strategy == strategy
            ]
        baseline = args.significance_baseline or args.strategies[0]
        for metric_name in ("recall_at_k", "answer_exact_match", "llm_answer_score", "hallucination_score", "evidence_span_recall_at_k"):
            significance.extend(
                compare_strategies_by_dataset(
                    metric_name,
                    collect_dataset_metric_vectors(diagnostics_by_strategy, metric_name),
                    baseline_strategy=baseline,
                )
            )
    payload = [asdict(result) for result in results]
    text = json.dumps(payload, indent=2)
    if args.output:
        _status(f"writing JSON output to {args.output}")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "overall": payload,
                    "grouped": {
                        strategy: {label: asdict(result) for label, result in slices.items()}
                        for strategy, slices in grouped_results.items()
                    },
                    "significance": [asdict(item) for item in significance],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.csv_output:
        _status(f"writing CSV output to {args.csv_output}")
        write_csv_results(results, args.csv_output)
    if args.diagnostics_output:
        _status(f"writing diagnostics output to {args.diagnostics_output}")
        write_diagnostics_csv(diagnostics, args.diagnostics_output)
    if args.significance_output:
        _status(f"writing significance output to {args.significance_output}")
        significance_path = Path(args.significance_output)
        significance_path.parent.mkdir(parents=True, exist_ok=True)
        significance_path.write_text(json.dumps([asdict(item) for item in significance], indent=2), encoding="utf-8")
    if args.report_output:
        _status(f"writing report to {args.report_output}")
        report = render_markdown_report(
            title=args.title,
            documents=documents,
            questions=questions,
            results=results,
            grouped_results=grouped_results,
            diagnostics=diagnostics,
            significance=significance,
        )
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    if args.plots_dir:
        _status(f"writing plots to {args.plots_dir}")
        write_svg_plots(results, args.plots_dir, diagnostics=diagnostics, grouped_results=grouped_results)
    _status("done")
    print(text)


def _document_matches_dataset_filter(doc_id: str, datasets: set[str]) -> bool:
    if doc_id.startswith("qasper:"):
        return "qasper" in datasets
    if doc_id.startswith("beir:"):
        parts = doc_id.split(":", maxsplit=2)
        return len(parts) >= 2 and parts[1] in datasets
    return True


def _question_matches_loaded_documents(question, allowed_doc_ids: set[str]) -> bool:
    if question.relevant_doc_ids:
        return any(doc_id in allowed_doc_ids for doc_id in question.relevant_doc_ids)
    if question.source_doc:
        return question.source_doc in allowed_doc_ids
    return True


def _status(message: str) -> None:
    print(f"[rag-benchmark] {message}", flush=True)


if __name__ == "__main__":
    main()
