from __future__ import annotations

import math
import random
from collections import defaultdict

from rag_chunking.models import SignificanceComparison


def paired_randomization_test(
    baseline_scores: list[float],
    candidate_scores: list[float],
    *,
    iterations: int = 2000,
    seed: int = 13,
) -> tuple[float, float, float]:
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("paired_randomization_test requires paired score vectors of equal length")
    if not baseline_scores:
        return 1.0, 0.0, 0.0

    observed = _mean(candidate_scores) - _mean(baseline_scores)
    differences = [candidate - baseline for baseline, candidate in zip(baseline_scores, candidate_scores)]
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        randomized = [difference if rng.random() >= 0.5 else -difference for difference in differences]
        simulated = _mean(randomized)
        if abs(simulated) >= abs(observed):
            extreme += 1
    pooled_std = _stddev(differences)
    effect_size = observed / pooled_std if pooled_std else 0.0
    return (extreme + 1) / (iterations + 1), observed, effect_size


def compare_strategies_by_dataset(
    metric_name: str,
    strategy_scores: dict[str, dict[str, list[float]]],
    baseline_strategy: str,
    *,
    alpha: float = 0.05,
    iterations: int = 2000,
) -> list[SignificanceComparison]:
    comparisons: list[SignificanceComparison] = []
    baseline_by_dataset = strategy_scores.get(baseline_strategy, {})
    for candidate_strategy, dataset_scores in strategy_scores.items():
        if candidate_strategy == baseline_strategy:
            continue
        for dataset, baseline_scores in baseline_by_dataset.items():
            candidate_vector = dataset_scores.get(dataset, [])
            paired_length = min(len(baseline_scores), len(candidate_vector))
            if paired_length == 0:
                continue
            baseline_vector = baseline_scores[:paired_length]
            candidate_trimmed = candidate_vector[:paired_length]
            p_value, mean_delta, effect_size = paired_randomization_test(
                baseline_vector,
                candidate_trimmed,
                iterations=iterations,
            )
            comparisons.append(
                SignificanceComparison(
                    metric=metric_name,
                    baseline_strategy=baseline_strategy,
                    candidate_strategy=candidate_strategy,
                    dataset=dataset,
                    baseline_mean=_mean(baseline_vector),
                    candidate_mean=_mean(candidate_trimmed),
                    mean_delta=mean_delta,
                    p_value=p_value,
                    effect_size=effect_size,
                    significant=p_value < alpha,
                    test_name="paired_randomization",
                )
            )
    return comparisons


def collect_dataset_metric_vectors(
    diagnostics_by_strategy: dict[str, list[dict[str, float | str | bool]]],
    metric_name: str,
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for strategy, rows in diagnostics_by_strategy.items():
        for row in rows:
            dataset = str(row.get("dataset", "overall") or "overall")
            grouped[strategy][dataset].append(float(row.get(metric_name, 0.0)))
    return {strategy: dict(dataset_rows) for strategy, dataset_rows in grouped.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
