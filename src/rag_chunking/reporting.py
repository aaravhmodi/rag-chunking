from __future__ import annotations

import csv
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rag_chunking.models import Document, ExperimentResult, QuestionDiagnostic, QuestionExample
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


def write_diagnostics_csv(diagnostics: list[QuestionDiagnostic], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in diagnostics]
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
    diagnostics: list[QuestionDiagnostic] | None = None,
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
    if diagnostics:
        lines.extend(["", "## Diagnostics", ""])
        for strategy in sorted({item.strategy for item in diagnostics}):
            strategy_diagnostics = [item for item in diagnostics if item.strategy == strategy]
            failure_counts = Counter(item.failure_mode for item in strategy_diagnostics)
            lines.append(f"### {strategy}")
            lines.append("")
            lines.append(f"- Failure modes: {_format_failure_counts(failure_counts)}")
            missed_evidence = [item for item in strategy_diagnostics if item.failure_mode == "missed_evidence_span"][:5]
            if missed_evidence:
                lines.append("- Sample missed evidence questions:")
                for item in missed_evidence:
                    lines.append(f"  - `{item.question_id}` first relevant rank={item.first_relevant_rank}, first evidence rank={item.first_evidence_rank}, top1 doc=`{item.top1_doc_id}`")
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


def write_svg_plots(
    results: list[ExperimentResult],
    output_dir: str | Path,
    diagnostics: list[QuestionDiagnostic] | None = None,
    grouped_results: dict[str, dict[str, ExperimentResult]] | None = None,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    charts: list[tuple[str, str, str]] = []

    _write_chart(
        root,
        "quality.jpg",
        "Overall Recall@k by Strategy",
        _save_bar_chart(
            root / "quality.jpg",
            title="Recall@k by Strategy",
            labels=[result.strategy for result in results],
            values=[result.recall_at_k for result in results],
            color="#1f6feb",
            y_max=1.0,
        ),
        charts,
    )
    _write_chart(
        root,
        "latency.jpg",
        "Retrieval latency by strategy",
        _save_bar_chart(
            root / "latency.jpg",
            title="Retrieval Latency (ms) by Strategy",
            labels=[result.strategy for result in results],
            values=[result.retrieval_latency_ms for result in results],
            color="#d97706",
        ),
        charts,
    )
    _write_chart(
        root,
        "mrr.jpg",
        "MRR comparison by strategy",
        _save_bar_chart(root / "mrr.jpg", "MRR by Strategy", [r.strategy for r in results], [r.mrr for r in results], "#0f766e", y_max=1.0),
        charts,
    )
    _write_chart(
        root,
        "ndcg.jpg",
        "nDCG@k comparison by strategy",
        _save_bar_chart(root / "ndcg.jpg", "nDCG@k by Strategy", [r.strategy for r in results], [r.ndcg_at_k for r in results], "#047857", y_max=1.0),
        charts,
    )
    _write_chart(
        root,
        "answer_em.jpg",
        "Answer exact match by strategy",
        _save_bar_chart(root / "answer_em.jpg", "Answer Exact Match by Strategy", [r.strategy for r in results], [r.answer_exact_match for r in results], "#2563eb", y_max=1.0),
        charts,
    )
    _write_chart(
        root,
        "evidence.jpg",
        "Evidence-span recall by strategy",
        _save_bar_chart(root / "evidence.jpg", "Evidence Span Recall@k by Strategy", [r.strategy for r in results], [r.evidence_span_recall_at_k for r in results], "#7c3aed", y_max=1.0),
        charts,
    )
    _write_chart(
        root,
        "chunking_latency.jpg",
        "Chunking latency by strategy",
        _save_bar_chart(root / "chunking_latency.jpg", "Chunking Latency (ms) by Strategy", [r.strategy for r in results], [r.chunking_latency_ms for r in results], "#dc2626"),
        charts,
    )
    _write_chart(
        root,
        "chunk_count.jpg",
        "Average chunk count per document by strategy",
        _save_bar_chart(root / "chunk_count.jpg", "Average Chunks per Document", [r.strategy for r in results], [r.avg_chunk_count for r in results], "#ea580c"),
        charts,
    )
    _write_chart(
        root,
        "chunk_length.jpg",
        "Average chunk length by strategy",
        _save_bar_chart(root / "chunk_length.jpg", "Average Chunk Length (chars)", [r.strategy for r in results], [r.avg_chunk_length_chars for r in results], "#9333ea"),
        charts,
    )
    _write_chart(
        root,
        "tradeoff_quality_latency.jpg",
        "Recall versus retrieval latency tradeoff",
        _save_scatter_chart(
            root / "tradeoff_quality_latency.jpg",
            "Recall@k vs Retrieval Latency",
            [(r.retrieval_latency_ms, r.recall_at_k, r.strategy) for r in results],
            "Retrieval latency (ms)",
            "Recall@k",
            "#1f6feb",
        ),
        charts,
    )
    _write_chart(
        root,
        "tradeoff_evidence_latency.jpg",
        "Evidence-span coverage versus retrieval latency tradeoff",
        _save_scatter_chart(
            root / "tradeoff_evidence_latency.jpg",
            "Evidence Recall@k vs Retrieval Latency",
            [(r.retrieval_latency_ms, r.evidence_span_recall_at_k, r.strategy) for r in results],
            "Retrieval latency (ms)",
            "Evidence Recall@k",
            "#7c3aed",
        ),
        charts,
    )
    _write_chart(
        root,
        "tradeoff_chunkcount_recall.jpg",
        "Recall versus chunk-count tradeoff",
        _save_scatter_chart(
            root / "tradeoff_chunkcount_recall.jpg",
            "Recall@k vs Average Chunks per Document",
            [(r.avg_chunk_count, r.recall_at_k, r.strategy) for r in results],
            "Average chunks per document",
            "Recall@k",
            "#ea580c",
        ),
        charts,
    )

    if diagnostics:
        _write_chart(
            root,
            "failure_modes.jpg",
            "Failure-mode composition by strategy",
            _save_stacked_bar_chart(root / "failure_modes.jpg", "Failure Modes by Strategy", sorted({item.strategy for item in diagnostics}), _failure_mode_series(diagnostics)),
            charts,
        )
        _write_chart(
            root,
            "top1_relevance.jpg",
            "Top-1 relevance rate by strategy",
            _save_rate_bar_from_diagnostics(root / "top1_relevance.jpg", diagnostics, "Top-1 Relevant Rate", lambda item: item.top1_relevant, "#16a34a"),
            charts,
        )
        _write_chart(
            root,
            "relevant_retrieved.jpg",
            "Any relevant chunk retrieved by strategy",
            _save_rate_bar_from_diagnostics(root / "relevant_retrieved.jpg", diagnostics, "Relevant Retrieved Rate", lambda item: item.relevant_retrieved, "#2563eb"),
            charts,
        )
        _write_chart(
            root,
            "evidence_coverage.jpg",
            "Evidence-span coverage rate by strategy",
            _save_rate_bar_from_diagnostics(
                root / "evidence_coverage.jpg",
                diagnostics,
                "Evidence Span Coverage Rate",
                lambda item: item.evidence_span_covered,
                "#7c3aed",
                predicate=lambda item: item.evidence_question,
            ),
            charts,
        )
        _write_chart(
            root,
            "first_relevant_rank.jpg",
            "Distribution of first relevant rank by strategy",
            _save_rank_distribution(root / "first_relevant_rank.jpg", "First Relevant Rank Distribution", diagnostics, lambda item: item.first_relevant_rank, lambda item: item.relevant_retrieved),
            charts,
        )
        _write_chart(
            root,
            "first_evidence_rank.jpg",
            "Distribution of first evidence-covering rank by strategy",
            _save_rank_distribution(root / "first_evidence_rank.jpg", "First Evidence Rank Distribution", diagnostics, lambda item: item.first_evidence_rank, lambda item: item.evidence_span_covered),
            charts,
        )
        _write_chart(
            root,
            "answer_hit_rate.jpg",
            "Answer exact-match rate by strategy from per-question diagnostics",
            _save_rate_bar_from_diagnostics(root / "answer_hit_rate.jpg", diagnostics, "Answer Exact Match Rate", lambda item: item.answer_exact_match >= 1.0, "#0f766e"),
            charts,
        )
        _write_chart(
            root,
            "missed_evidence_vs_relevant.jpg",
            "Missed-evidence and missed-relevant counts side by side",
            _save_grouped_count_chart(
                root / "missed_evidence_vs_relevant.jpg",
                "Missed Evidence vs Missed Relevant",
                sorted({item.strategy for item in diagnostics}),
                {
                    "missed_relevant": [sum(1 for item in diagnostics if item.strategy == label and item.failure_mode == "missed_relevant") for label in sorted({item.strategy for item in diagnostics})],
                    "missed_evidence_span": [sum(1 for item in diagnostics if item.strategy == label and item.failure_mode == "missed_evidence_span") for label in sorted({item.strategy for item in diagnostics})],
                },
                {"missed_relevant": "#dc2626", "missed_evidence_span": "#f59e0b"},
            ),
            charts,
        )

    if grouped_results:
        slice_labels = _ordered_slice_labels(grouped_results)
        if slice_labels:
            _write_chart(
                root,
                "slice_recall.jpg",
                "Recall by strategy across dataset and split slices",
                _save_slice_metric_chart(root / "slice_recall.jpg", grouped_results, slice_labels, "recall_at_k", "Recall@k Across Slices", "#1f6feb"),
                charts,
            )
            _write_chart(
                root,
                "slice_evidence.jpg",
                "Evidence recall by strategy across dataset and split slices",
                _save_slice_metric_chart(root / "slice_evidence.jpg", grouped_results, slice_labels, "evidence_span_recall_at_k", "Evidence Recall@k Across Slices", "#7c3aed"),
                charts,
            )
            _write_chart(
                root,
                "slice_latency.jpg",
                "Retrieval latency across slices",
                _save_slice_metric_chart(root / "slice_latency.jpg", grouped_results, slice_labels, "retrieval_latency_ms", "Retrieval Latency Across Slices", "#d97706"),
                charts,
            )

    _write_plot_index(root, charts)


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


def _format_failure_counts(counts: Counter[str]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{label}={counts[label]}" for label in sorted(counts))


def _write_chart(root: Path, filename: str, description: str, _unused_result, charts: list[tuple[str, str, str]]) -> None:
    charts.append((filename, filename.removesuffix(".jpg").replace("_", " "), description))


def _write_plot_index(root: Path, charts: list[tuple[str, str, str]]) -> None:
    lines = ["# Plot Index", ""]
    for filename, title, description in charts:
        lines.append(f"- [{title}]({filename}): {description}")
    root.joinpath("index.md").write_text("\n".join(lines), encoding="utf-8")


def _save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="jpeg", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_bar_chart(path: Path, title: str, labels: list[str], values: list[float], color: str, y_max: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    if y_max is not None:
        ax.set_ylim(0, max(y_max, max(values, default=0)))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}" if value < 10 else f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    _save_figure(fig, path)


def _save_scatter_chart(path: Path, title: str, points: list[tuple[float, float, str]], x_label: str, y_label: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.scatter(xs, ys, c=color, s=80)
    for x, y, label in points:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)
    _save_figure(fig, path)


def _save_stacked_bar_chart(path: Path, title: str, labels: list[str], series: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "success_evidence": "#16a34a",
        "success_relevant": "#22c55e",
        "missed_evidence_span": "#f59e0b",
        "missed_relevant": "#dc2626",
    }
    bottoms = [0] * len(labels)
    for name in ["success_evidence", "success_relevant", "missed_evidence_span", "missed_relevant"]:
        values = series.get(name, [0] * len(labels))
        ax.bar(labels, values, bottom=bottoms, label=name, color=colors[name])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, path)


def _save_rate_bar_from_diagnostics(path: Path, diagnostics: list[QuestionDiagnostic], title: str, getter, color: str, predicate=None) -> None:
    strategies = sorted({item.strategy for item in diagnostics})
    values = []
    for strategy in strategies:
        rows = [item for item in diagnostics if item.strategy == strategy and (predicate(item) if predicate else True)]
        values.append((sum(1 for item in rows if getter(item)) / len(rows)) if rows else 0.0)
    _save_bar_chart(path, title, strategies, values, color, y_max=1.0)


def _save_rank_distribution(path: Path, title: str, diagnostics: list[QuestionDiagnostic], rank_getter, predicate) -> None:
    strategies = sorted({item.strategy for item in diagnostics})
    labels = ["1", "2", "3", "4", "5", ">5/na"]
    series: dict[str, list[float]] = {}
    for strategy in strategies:
        rows = [item for item in diagnostics if item.strategy == strategy]
        counts = [0, 0, 0, 0, 0, 0]
        for item in rows:
            rank = rank_getter(item)
            if predicate(item) and rank is not None and 1 <= rank <= 5:
                counts[rank - 1] += 1
            else:
                counts[5] += 1
        series[strategy] = counts
    _save_grouped_count_chart(path, title, labels, series, None)


def _save_grouped_count_chart(path: Path, title: str, labels: list[str], series: dict[str, list[float]], colors: dict[str, str] | None) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    series_names = list(series.keys())
    width = 0.8 / max(len(series_names), 1)
    x_positions = list(range(len(labels)))
    palette = ["#1f6feb", "#dc2626", "#7c3aed", "#0f766e", "#ea580c", "#2563eb"]
    for index, name in enumerate(series_names):
        values = series[name]
        offset_positions = [x + (index - (len(series_names) - 1) / 2) * width for x in x_positions]
        ax.bar(offset_positions, values, width=width, label=name, color=(colors[name] if colors and name in colors else palette[index % len(palette)]))
    ax.set_xticks(x_positions, labels)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, path)


def _save_slice_metric_chart(path: Path, grouped_results: dict[str, dict[str, ExperimentResult]], slice_labels: list[str], metric_name: str, title: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    strategies = sorted(grouped_results)
    palette = ["#1f6feb", "#dc2626", "#7c3aed", "#0f766e", "#ea580c", "#2563eb"]
    y_positions = list(range(len(slice_labels)))
    for index, strategy in enumerate(strategies):
        values = [float(getattr(grouped_results[strategy][label], metric_name)) if label in grouped_results[strategy] else 0.0 for label in slice_labels]
        ax.plot(values, y_positions, marker="o", label=strategy, color=palette[index % len(palette)])
    ax.set_yticks(y_positions, slice_labels)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=8)
    _save_figure(fig, path)


def _failure_mode_series(diagnostics: list[QuestionDiagnostic]) -> dict[str, list[float]]:
    strategies = sorted({item.strategy for item in diagnostics})
    modes = ["success_evidence", "success_relevant", "missed_evidence_span", "missed_relevant"]
    return {
        mode: [sum(1 for item in diagnostics if item.strategy == strategy and item.failure_mode == mode) for strategy in strategies]
        for mode in modes
    }


def _rate_bar_from_diagnostics_svg(
    diagnostics: list[QuestionDiagnostic],
    title: str,
    getter,
    fill: str,
    predicate=None,
) -> str:
    strategies = sorted({item.strategy for item in diagnostics})
    values = []
    for strategy in strategies:
        rows = [item for item in diagnostics if item.strategy == strategy and (predicate(item) if predicate else True)]
        if not rows:
            values.append(0.0)
            continue
        values.append(sum(1 for item in rows if getter(item)) / len(rows))
    return _bar_chart_svg(title, strategies, values, "{:.3f}", 1.0, fill)


def _rank_distribution_svg(title: str, diagnostics: list[QuestionDiagnostic], rank_getter, predicate) -> str:
    strategies = sorted({item.strategy for item in diagnostics})
    rank_labels = ["1", "2", "3", "4", "5", ">5/na"]
    series: dict[str, list[float]] = {}
    for strategy in strategies:
        rows = [item for item in diagnostics if item.strategy == strategy]
        counts = [0, 0, 0, 0, 0, 0]
        for item in rows:
            rank = rank_getter(item)
            if predicate(item) and rank is not None and 1 <= rank <= 5:
                counts[rank - 1] += 1
            else:
                counts[5] += 1
        series[strategy] = counts
    return _grouped_count_chart_svg(title=title, labels=rank_labels, series=series, colors=None)


def _slice_metric_chart_svg(
    grouped_results: dict[str, dict[str, ExperimentResult]],
    slice_labels: list[str],
    metric_name: str,
    title: str,
    color: str,
) -> str:
    strategies = sorted(grouped_results)
    width = 1200
    height = 460
    margin_left = 220
    margin_top = 60
    chart_width = 920
    row_height = 34
    bar_height = 18
    max_value = max(
        (float(getattr(grouped_results[strategy][label], metric_name)) for strategy in strategies for label in slice_labels if label in grouped_results[strategy]),
        default=1.0,
    )
    max_value = max(max_value, 1e-9)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: Arial, sans-serif; fill: #111827; } .label { font-size: 12px; } .title { font-size: 20px; font-weight: 700; } .sub { font-size: 11px; fill: #6b7280; }</style>',
        f'<text class="title" x="{margin_left}" y="30">{title}</text>',
    ]
    y = margin_top
    palette = ["#1f6feb", "#dc2626", "#7c3aed", "#0f766e", "#ea580c", "#2563eb"]
    for slice_label in slice_labels:
        rows.append(f'<text class="label" x="12" y="{y + 14}">{slice_label}</text>')
        offset = 0
        for index, strategy in enumerate(strategies):
            if slice_label not in grouped_results[strategy]:
                continue
            value = float(getattr(grouped_results[strategy][slice_label], metric_name))
            width_px = (value / max_value) * chart_width
            color_fill = palette[index % len(palette)]
            rows.append(f'<rect x="{margin_left}" y="{y + offset}" width="{width_px:.2f}" height="{bar_height}" fill="{color_fill}" rx="3"/>')
            rows.append(f'<text class="sub" x="{margin_left + width_px + 8:.2f}" y="{y + offset + 13}">{strategy} {value:.3f}</text>')
            offset += bar_height + 3
        y += row_height + 34
        if y > height - 40:
            break
    rows.append("</svg>")
    return "\n".join(rows)


def _scatter_chart_svg(title: str, points: list[tuple[float, float, str]], x_label: str, y_label: str, point_fill: str) -> str:
    width = 900
    height = 520
    margin_left = 90
    margin_right = 40
    margin_top = 60
    margin_bottom = 80
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    x_values = [point[0] for point in points] or [0.0]
    y_values = [point[1] for point in points] or [0.0]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: Arial, sans-serif; fill: #111827; } .title { font-size: 20px; font-weight: 700; } .axis { font-size: 12px; fill: #4b5563; } .label { font-size: 12px; }</style>',
        f'<text class="title" x="{margin_left}" y="30">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#9ca3af"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#9ca3af"/>',
        f'<text class="axis" x="{margin_left + chart_width / 2}" y="{height - 20}">{x_label}</text>',
        f'<text class="axis" x="14" y="{margin_top + chart_height / 2}" transform="rotate(-90 14,{margin_top + chart_height / 2})">{y_label}</text>',
    ]
    for x, y, label in points:
        px = margin_left + ((x - x_min) / x_span) * chart_width
        py = margin_top + chart_height - ((y - y_min) / y_span) * chart_height
        rows.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="7" fill="{point_fill}" fill-opacity="0.85"/>')
        rows.append(f'<text class="label" x="{px + 10:.2f}" y="{py - 8:.2f}">{label}</text>')
    rows.append("</svg>")
    return "\n".join(rows)


def _stacked_bar_chart_svg(title: str, labels: list[str], series: dict[str, list[float]]) -> str:
    width = 960
    height = 440
    margin_left = 150
    margin_top = 60
    chart_width = 720
    row_height = 52
    bar_height = 24
    colors = {
        "success_evidence": "#16a34a",
        "success_relevant": "#22c55e",
        "missed_evidence_span": "#f59e0b",
        "missed_relevant": "#dc2626",
    }
    totals = [sum(series[name][i] for name in series) for i in range(len(labels))]
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: Arial, sans-serif; fill: #111827; } .title { font-size: 20px; font-weight: 700; } .label { font-size: 14px; } .legend { font-size: 12px; }</style>',
        f'<text class="title" x="{margin_left}" y="30">{title}</text>',
    ]
    for index, label in enumerate(labels):
        y = margin_top + index * row_height
        rows.append(f'<text class="label" x="12" y="{y + 17}">{label}</text>')
        x = margin_left
        total = totals[index] if totals[index] else 1.0
        for name in ["success_evidence", "success_relevant", "missed_evidence_span", "missed_relevant"]:
            value = series.get(name, [0] * len(labels))[index]
            width_px = (value / total) * chart_width
            rows.append(f'<rect x="{x:.2f}" y="{y}" width="{width_px:.2f}" height="{bar_height}" fill="{colors[name]}" rx="3"/>')
            x += width_px
        rows.append(f'<text class="legend" x="{margin_left + chart_width + 10}" y="{y + 17}">{int(totals[index])} questions</text>')
    legend_x = margin_left
    for index, name in enumerate(["success_evidence", "success_relevant", "missed_evidence_span", "missed_relevant"]):
        rows.append(f'<rect x="{legend_x + index * 170}" y="{height - 30}" width="14" height="14" fill="{colors[name]}"/>')
        rows.append(f'<text class="legend" x="{legend_x + index * 170 + 20}" y="{height - 18}">{name}</text>')
    rows.append("</svg>")
    return "\n".join(rows)


def _grouped_count_chart_svg(title: str, labels: list[str], series: dict[str, list[float]], colors: dict[str, str] | None) -> str:
    width = 1100
    height = 460
    margin_left = 100
    margin_top = 70
    margin_bottom = 100
    chart_width = 920
    chart_height = 260
    palette = ["#1f6feb", "#dc2626", "#7c3aed", "#0f766e", "#ea580c", "#2563eb"]
    all_values = [value for values in series.values() for value in values] or [0.0]
    max_value = max(all_values) or 1.0
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: Arial, sans-serif; fill: #111827; } .title { font-size: 20px; font-weight: 700; } .axis { font-size: 12px; fill: #4b5563; } .legend { font-size: 12px; }</style>',
        f'<text class="title" x="{margin_left}" y="30">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#9ca3af"/>',
    ]
    group_width = chart_width / max(len(labels), 1)
    bar_width = group_width / max(len(series), 1) * 0.65
    for index, label in enumerate(labels):
        group_x = margin_left + index * group_width
        rows.append(f'<text class="axis" x="{group_x + group_width / 2 - 10:.2f}" y="{margin_top + chart_height + 24}">{label}</text>')
        for series_index, (name, values) in enumerate(series.items()):
            value = values[index]
            height_px = (value / max_value) * (chart_height - 10)
            x = group_x + series_index * (bar_width + 10) + 10
            y = margin_top + chart_height - height_px
            fill = colors[name] if colors and name in colors else palette[series_index % len(palette)]
            rows.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height_px:.2f}" fill="{fill}" rx="3"/>')
            rows.append(f'<text class="axis" x="{x:.2f}" y="{y - 6:.2f}">{int(value)}</text>')
    legend_x = margin_left
    for index, name in enumerate(series):
        fill = colors[name] if colors and name in colors else palette[index % len(palette)]
        rows.append(f'<rect x="{legend_x + index * 170}" y="{height - 28}" width="14" height="14" fill="{fill}"/>')
        rows.append(f'<text class="legend" x="{legend_x + index * 170 + 20}" y="{height - 16}">{name}</text>')
    rows.append("</svg>")
    return "\n".join(rows)


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
