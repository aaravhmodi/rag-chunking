from __future__ import annotations

import csv
import statistics
from dataclasses import asdict
from pathlib import Path

from rag_chunking.models import Document, ExperimentResult, QuestionExample
from rag_chunking.text_utils import tokenize


def write_csv_results(results: list[ExperimentResult], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_report(
    title: str,
    documents: list[Document],
    questions: list[QuestionExample],
    results: list[ExperimentResult],
    grouped_results: dict[str, dict[str, ExperimentResult]] | None = None,
) -> str:
    total_doc_tokens = sum(len(tokenize(document.text)) for document in documents)
    avg_doc_tokens = total_doc_tokens / len(documents) if documents else 0.0
    question_types = sorted({question.question_type for question in questions})
    dataset_counts = _metadata_counts(questions, "dataset")
    split_counts = _metadata_counts(questions, "split")
    best_recall = max(results, key=lambda result: result.recall_at_k) if results else None
    best_mrr = max(results, key=lambda result: result.mrr) if results else None
    fastest = min(results, key=lambda result: result.retrieval_latency_ms) if results else None
    most_compact = min(results, key=lambda result: result.avg_chunk_count) if results else None
    chunk_counts = [result.avg_chunk_count for result in results]
    chunk_lengths = [result.avg_chunk_length_chars for result in results]

    lines = [
        f"# {title}",
        "",
        "## Abstract",
        "",
        (
            "This report evaluates how chunking strategy changes retrieval effectiveness and efficiency in a "
            "dependency-light RAG benchmark. Strategies are compared on recall, ranking quality, answer coverage, "
            "evidence-span coverage, chunk count, chunk length, and retrieval latency."
        ),
        "",
        "## Dataset",
        "",
        f"- Documents: {len(documents)}",
        f"- Questions: {len(questions)}",
        f"- Average document length (tokens): {avg_doc_tokens:.1f}",
        f"- Question types: {', '.join(question_types) if question_types else 'unknown'}",
        f"- Datasets: {_format_count_summary(dataset_counts)}",
        f"- Splits: {_format_count_summary(split_counts)}",
        "",
        "## Method",
        "",
        (
            "Each strategy chunks the same document collection, indexes chunk text with a lexical retriever, and "
            "retrieves the top-k chunks for each question. Relevance is counted when the correct source document is "
            "retrieved and the chunk contains either the gold evidence string or the answer string. For questions "
            "with annotated character spans, evidence-span recall@k counts whether any retrieved chunk fully covers "
            "the labeled evidence span."
        ),
        "",
        "## Results",
        "",
        _results_table(results),
        "",
        "## Findings",
        "",
    ]

    if best_recall:
        lines.append(f"- Highest recall@k: `{best_recall.strategy}` at {best_recall.recall_at_k:.3f}.")
    best_evidence = max(results, key=lambda result: result.evidence_span_recall_at_k) if results else None
    if best_mrr:
        lines.append(f"- Best MRR: `{best_mrr.strategy}` at {best_mrr.mrr:.3f}.")
    if best_evidence and best_evidence.evidence_question_count:
        lines.append(f"- Highest evidence-span recall@k: `{best_evidence.strategy}` at {best_evidence.evidence_span_recall_at_k:.3f}.")
    if fastest:
        lines.append(f"- Lowest retrieval latency: `{fastest.strategy}` at {fastest.retrieval_latency_ms:.3f} ms.")
    if most_compact:
        lines.append(f"- Lowest average chunk count per document: `{most_compact.strategy}` at {most_compact.avg_chunk_count:.2f}.")
    if chunk_counts and chunk_lengths:
        lines.append(
            f"- Strategy spread: chunk count ranges from {min(chunk_counts):.2f} to {max(chunk_counts):.2f}, "
            f"and average chunk length ranges from {min(chunk_lengths):.2f} to {max(chunk_lengths):.2f} characters."
        )
    if grouped_results:
        lines.extend(["", "## Slice Analysis", ""])
        for slice_label in _ordered_slice_labels(grouped_results):
            lines.append(f"### {slice_label}")
            lines.append("")
            lines.append(_results_table([slices[slice_label] for slices in grouped_results.values() if slice_label in slices]))
            lines.append("")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Retrieval is lexical rather than embedding-based, so semantic recall is under-measured.",
            "- Relevance uses exact string containment for evidence and answers, which is stricter than human judgment.",
            "- Results are descriptive; this scaffold does not run significance tests.",
            "",
            "## Reproducibility",
            "",
            "- Install with `python -m pip install -e .`.",
            "- Run `rag-benchmark` with the same document, question, and strategy arguments used for this report.",
        ]
    )
    return "\n".join(lines)


def write_svg_plots(results: list[ExperimentResult], output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("quality.svg").write_text(
        _bar_chart_svg(
            title="Recall@k by Strategy",
            labels=[result.strategy for result in results],
            values=[result.recall_at_k for result in results],
            value_format="{:.3f}",
            max_value=1.0,
            fill="#1f6feb",
        ),
        encoding="utf-8",
    )
    root.joinpath("latency.svg").write_text(
        _bar_chart_svg(
            title="Retrieval Latency (ms) by Strategy",
            labels=[result.strategy for result in results],
            values=[result.retrieval_latency_ms for result in results],
            value_format="{:.3f}",
            max_value=max((result.retrieval_latency_ms for result in results), default=1.0),
            fill="#d97706",
        ),
        encoding="utf-8",
    )


def _results_table(results: list[ExperimentResult]) -> str:
    header = (
        "| Strategy | Recall@k | MRR | nDCG@k | Answer EM | Evidence R@k | Avg chunks/doc | Avg chunk chars | Chunking ms | Retrieval ms |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [
        (
            f"| {result.strategy} | {result.recall_at_k:.3f} | {result.mrr:.3f} | {result.ndcg_at_k:.3f} | "
            f"{_format_answer_metric(result)} | { _format_evidence_metric(result)} | {result.avg_chunk_count:.2f} | {result.avg_chunk_length_chars:.2f} | "
            f"{result.chunking_latency_ms:.3f} | {result.retrieval_latency_ms:.3f} |"
        )
        for result in sorted(results, key=lambda item: (item.recall_at_k, item.mrr, -item.retrieval_latency_ms), reverse=True)
    ]
    return "\n".join([header, *rows]) if rows else header


def _format_answer_metric(result: ExperimentResult) -> str:
    if result.answerable_question_count == 0:
        return "n/a"
    return f"{result.answer_exact_match:.3f}"


def _format_evidence_metric(result: ExperimentResult) -> str:
    if result.evidence_question_count == 0:
        return "n/a"
    return f"{result.evidence_span_recall_at_k:.3f}"


def _metadata_counts(questions: list[QuestionExample], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in questions:
        value = question.metadata.get(field)
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _format_count_summary(counts: dict[str, int]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _ordered_slice_labels(grouped_results: dict[str, dict[str, ExperimentResult]]) -> list[str]:
    labels = set()
    for slices in grouped_results.values():
        labels.update(label for label in slices if label != "overall")
    return sorted(labels)


def _bar_chart_svg(
    title: str,
    labels: list[str],
    values: list[float],
    value_format: str,
    max_value: float,
    fill: str,
) -> str:
    width = 900
    height = 420
    margin_left = 160
    margin_top = 60
    chart_width = 680
    row_height = 44
    bar_height = 24
    max_value = max(max_value, 1e-9)
    svg_rows: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: Arial, sans-serif; fill: #111827; } .label { font-size: 14px; } .title { font-size: 20px; font-weight: 700; } .value { font-size: 12px; }</style>',
        f'<text class="title" x="{margin_left}" y="32">{title}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + index * row_height
        bar_width = 0 if not values else (value / max_value) * chart_width
        svg_rows.append(f'<text class="label" x="12" y="{y + 17}">{label}</text>')
        svg_rows.append(f'<rect x="{margin_left}" y="{y}" width="{chart_width}" height="{bar_height}" fill="#e5e7eb" rx="4"/>')
        svg_rows.append(f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" fill="{fill}" rx="4"/>')
        svg_rows.append(f'<text class="value" x="{margin_left + bar_width + 8:.2f}" y="{y + 17}">{value_format.format(value)}</text>')
    svg_rows.append("</svg>")
    return "\n".join(svg_rows)
