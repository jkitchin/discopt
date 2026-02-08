"""Performance metrics computation for benchmark runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SolverMetrics:
    """Aggregated metrics from a benchmark run."""

    n_instances: int
    n_solved: int
    n_optimal: int
    n_incorrect: int
    total_wall_time: float
    median_wall_time: float
    geomean_wall_time: float
    node_throughput: float
    mean_gap: float
    layer_fractions: dict[str, float]


@dataclass
class BatchMetrics:
    """Metrics from batch evaluation scaling tests."""

    batch_sizes: list[int] = field(default_factory=list)
    throughputs: list[float] = field(default_factory=list)
    scaling_efficiency: float = 0.0


def shifted_geometric_mean(times: list[float], shift: float = 1.0) -> float:
    """Shifted geometric mean: exp(mean(log(t + shift))) - shift.

    Args:
        times: List of non-negative time values.
        shift: Shift parameter to handle near-zero values (default 1.0).

    Returns:
        The shifted geometric mean.
    """
    if not times:
        return 0.0
    n = len(times)
    log_sum = sum(math.log(t + shift) for t in times)
    return math.exp(log_sum / n) - shift


def compute_solver_metrics(results: list[dict]) -> SolverMetrics:
    """Compute aggregate metrics from a list of solve results.

    Each result dict has keys: status, objective, wall_time, node_count,
    rust_time, jax_time, python_time, gap, expected_objective (optional).

    Args:
        results: List of result dicts from benchmark runs.

    Returns:
        Aggregated SolverMetrics.
    """
    n = len(results)
    if n == 0:
        return SolverMetrics(
            n_instances=0,
            n_solved=0,
            n_optimal=0,
            n_incorrect=0,
            total_wall_time=0.0,
            median_wall_time=0.0,
            geomean_wall_time=0.0,
            node_throughput=0.0,
            mean_gap=0.0,
            layer_fractions={"rust": 0.0, "jax": 0.0, "python": 0.0},
        )

    n_solved = sum(1 for r in results if r["status"] in ("optimal", "feasible"))
    n_optimal = sum(1 for r in results if r["status"] == "optimal")

    # Count incorrect: solved but objective differs from expected
    n_incorrect = 0
    for r in results:
        expected = r.get("expected_objective")
        if expected is not None and r["objective"] is not None:
            if r["status"] in ("optimal", "feasible"):
                abs_err = abs(r["objective"] - expected)
                rel_err = abs_err / max(abs(expected), 1e-10)
                if abs_err > 1e-4 and rel_err > 1e-4:
                    n_incorrect += 1

    wall_times = [r["wall_time"] for r in results]
    total_wall_time = sum(wall_times)
    sorted_times = sorted(wall_times)
    if n % 2 == 1:
        median_wall_time = sorted_times[n // 2]
    else:
        median_wall_time = (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2.0
    geomean_wall_time = shifted_geometric_mean(wall_times)

    # Node throughput: median nodes/sec across instances with nonzero time
    throughputs = []
    for r in results:
        wt = r["wall_time"]
        if wt > 0:
            throughputs.append(r["node_count"] / wt)
    if throughputs:
        sorted_tp = sorted(throughputs)
        nt = len(sorted_tp)
        if nt % 2 == 1:
            node_throughput = sorted_tp[nt // 2]
        else:
            node_throughput = (sorted_tp[nt // 2 - 1] + sorted_tp[nt // 2]) / 2.0
    else:
        node_throughput = 0.0

    # Mean gap (only for solved instances with finite gap)
    gaps = [r["gap"] for r in results if r["gap"] is not None and math.isfinite(r["gap"])]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0

    # Layer fractions
    total_rust = sum(r["rust_time"] for r in results)
    total_jax = sum(r["jax_time"] for r in results)
    total_python = sum(r["python_time"] for r in results)
    total_layer = total_rust + total_jax + total_python
    if total_layer > 0:
        layer_fractions = {
            "rust": total_rust / total_layer,
            "jax": total_jax / total_layer,
            "python": total_python / total_layer,
        }
    else:
        layer_fractions = {"rust": 0.0, "jax": 0.0, "python": 0.0}

    return SolverMetrics(
        n_instances=n,
        n_solved=n_solved,
        n_optimal=n_optimal,
        n_incorrect=n_incorrect,
        total_wall_time=total_wall_time,
        median_wall_time=median_wall_time,
        geomean_wall_time=geomean_wall_time,
        node_throughput=node_throughput,
        mean_gap=mean_gap,
        layer_fractions=layer_fractions,
    )


def compute_batch_metrics(batch_sizes: list[int], throughputs: list[float]) -> BatchMetrics:
    """Compute batch scaling metrics.

    Scaling efficiency is the ratio of throughput doubling per batch size doubling,
    computed as the median of log2(throughput[i+1]/throughput[i]) / log2(size[i+1]/size[i])
    across consecutive pairs.

    Args:
        batch_sizes: List of batch sizes tested.
        throughputs: Corresponding throughputs (evaluations/sec).

    Returns:
        BatchMetrics with scaling efficiency.
    """
    if len(batch_sizes) < 2 or len(throughputs) < 2:
        return BatchMetrics(
            batch_sizes=list(batch_sizes),
            throughputs=list(throughputs),
            scaling_efficiency=0.0,
        )

    ratios = []
    for i in range(len(batch_sizes) - 1):
        if batch_sizes[i] > 0 and batch_sizes[i + 1] > 0:
            if throughputs[i] > 0 and throughputs[i + 1] > 0:
                size_ratio = math.log2(batch_sizes[i + 1] / batch_sizes[i])
                if size_ratio > 0:
                    tp_ratio = math.log2(throughputs[i + 1] / throughputs[i])
                    ratios.append(tp_ratio / size_ratio)

    if ratios:
        sorted_ratios = sorted(ratios)
        nr = len(sorted_ratios)
        if nr % 2 == 1:
            scaling_efficiency = sorted_ratios[nr // 2]
        else:
            scaling_efficiency = (sorted_ratios[nr // 2 - 1] + sorted_ratios[nr // 2]) / 2.0
    else:
        scaling_efficiency = 0.0

    return BatchMetrics(
        batch_sizes=list(batch_sizes),
        throughputs=list(throughputs),
        scaling_efficiency=scaling_efficiency,
    )
