"""Benchmark runner that exercises the discopt solver."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from discopt.modeling.core import Model, SolveResult


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    suite: str  # "smoke", "phase1", "phase2"
    time_limit: float = 60.0
    max_nodes: int = 100_000
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32, 64, 128])


@dataclass
class BenchmarkResult:
    """Result of a single benchmark instance."""

    name: str
    status: str
    objective: Optional[float]
    expected_objective: Optional[float]
    wall_time: float
    node_count: int
    rust_time: float
    jax_time: float
    python_time: float
    gap: Optional[float]

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "status": self.status,
            "objective": self.objective,
            "expected_objective": self.expected_objective,
            "wall_time": self.wall_time,
            "node_count": self.node_count,
            "rust_time": self.rust_time,
            "jax_time": self.jax_time,
            "python_time": self.python_time,
            "gap": self.gap,
        }


class BenchmarkRunner:
    """Run benchmark suites and collect performance metrics."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config

    @property
    def config(self) -> BenchmarkConfig:
        return self._config

    def run_instance(
        self,
        model: Model,
        name: str,
        expected_obj: Optional[float] = None,
    ) -> BenchmarkResult:
        """Run a single benchmark instance.

        Args:
            model: Model to solve.
            name: Instance name for reporting.
            expected_obj: Known optimal objective (for correctness checks).

        Returns:
            BenchmarkResult with solve statistics.
        """
        result: SolveResult = model.solve(
            time_limit=self._config.time_limit,
        )

        return BenchmarkResult(
            name=name,
            status=result.status,
            objective=result.objective,
            expected_objective=expected_obj,
            wall_time=result.wall_time,
            node_count=result.node_count,
            rust_time=result.rust_time,
            jax_time=result.jax_time,
            python_time=result.python_time,
            gap=result.gap,
        )

    def run_suite(
        self,
        instances: list[tuple[Model, str, Optional[float]]],
    ) -> list[BenchmarkResult]:
        """Run all instances in a suite.

        Args:
            instances: List of (model, name, expected_obj) tuples.

        Returns:
            List of BenchmarkResult for each instance.
        """
        results = []
        for model, name, expected_obj in instances:
            result = self.run_instance(model, name, expected_obj)
            results.append(result)
        return results

    def export_json(self, results: list[BenchmarkResult], path: str) -> None:
        """Export results as JSON.

        Args:
            results: List of benchmark results.
            path: Output file path.
        """
        data = {
            "config": {
                "suite": self._config.suite,
                "time_limit": self._config.time_limit,
                "max_nodes": self._config.max_nodes,
            },
            "results": [r.to_dict() for r in results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def get_smoke_instances() -> list[tuple[Model, str, Optional[float]]]:
    """Get smoke test instances for quick benchmarking.

    Returns:
        List of (model, name, expected_obj) tuples:
        1. Simple MINLP from examples
        2. Continuous NLP: min x^2 + y^2 s.t. x+y >= 1
        3. Small MINLP: min x^2 + z s.t. x+y >= 1, z binary
    """
    from discopt.modeling.examples import example_simple_minlp

    instances: list[tuple[Model, str, Optional[float]]] = []

    # 1. Simple MINLP from examples (expected: 0.5 at x1=x2=0.5, x3=0)
    m1 = example_simple_minlp()
    instances.append((m1, "simple_minlp", 0.5))

    # 2. Continuous NLP: min x^2 + y^2 s.t. x + y >= 1
    # Optimal: x=y=0.5, obj=0.5
    m2 = Model("continuous_nlp")
    x = m2.continuous("x", lb=-5.0, ub=5.0)
    y = m2.continuous("y", lb=-5.0, ub=5.0)
    m2.minimize(x**2 + y**2)
    m2.subject_to(x + y >= 1)
    instances.append((m2, "continuous_nlp", 0.5))

    # 3. Small MINLP: min x^2 + z s.t. x + y >= 1, z binary
    # Optimal: x=0.5, y=0.5, z=0, obj=0.25
    m3 = Model("small_minlp")
    x3 = m3.continuous("x", lb=-5.0, ub=5.0)
    y3 = m3.continuous("y", lb=-5.0, ub=5.0)
    z3 = m3.binary("z")
    m3.minimize(x3**2 + z3)
    m3.subject_to(x3 + y3 >= 1)
    instances.append((m3, "small_minlp", 0.25))

    return instances
