"""discopt.benchmarks - Performance measurement and benchmark infrastructure."""

from discopt.benchmarks.metrics import (
    BatchMetrics,
    SolverMetrics,
    compute_batch_metrics,
    compute_solver_metrics,
    shifted_geometric_mean,
)
from discopt.benchmarks.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    get_smoke_instances,
)

__all__ = [
    "BatchMetrics",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "SolverMetrics",
    "compute_batch_metrics",
    "compute_solver_metrics",
    "get_smoke_instances",
    "shifted_geometric_mean",
]
