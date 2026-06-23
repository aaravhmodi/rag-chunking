from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_chunking.loaders import load_documents, load_questions
from rag_chunking.pipeline import run_experiment, run_grouped_experiments
from rag_chunking.reporting import render_markdown_report, write_csv_results, write_svg_plots


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.documents)
    questions = load_questions(args.questions)
    questions = [
        question
        for question in questions
        if (not args.question_split or question.metadata.get("split") in args.question_split)
        and (not args.question_dataset or question.metadata.get("dataset") in args.question_dataset)
    ]
    results = [run_experiment(documents, questions, strategy, top_k=args.top_k) for strategy in args.strategies]
    grouped_results = {
        strategy: run_grouped_experiments(documents, questions, strategy, top_k=args.top_k, group_fields=tuple(args.group_by))
        for strategy in args.strategies
    }
    payload = [asdict(result) for result in results]
    text = json.dumps(payload, indent=2)
    if args.output:
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
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.csv_output:
        write_csv_results(results, args.csv_output)
    if args.report_output:
        report = render_markdown_report(
            title=args.title,
            documents=documents,
            questions=questions,
            results=results,
            grouped_results=grouped_results,
        )
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    if args.plots_dir:
        write_svg_plots(results, args.plots_dir)
    print(text)


if __name__ == "__main__":
    main()
