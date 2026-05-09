"""Tests for discopt.benchmarks performance measurement infrastructure."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import json
import math
import tempfile

import pytest
from discopt.benchmarks.metrics import (
    BatchMetrics,
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
from discopt.modeling.core import Model

# ─────────────────────────────────────────────────────────────
# TestShiftedGeometricMean
# ─────────────────────────────────────────────────────────────


class TestShiftedGeometricMean:
    def test_empty_list(self):
        assert shifted_geometric_mean([]) == 0.0

    def test_single_value(self):
        # sgm([5.0], shift=1) = exp(log(6)) - 1 = 5.0
        result = shifted_geometric_mean([5.0], shift=1.0)
        assert abs(result - 5.0) < 1e-10

    def test_identical_values(self):
        # sgm([3, 3, 3], shift=1) = exp(log(4)) - 1 = 3.0
        result = shifted_geometric_mean([3.0, 3.0, 3.0], shift=1.0)
        assert abs(result - 3.0) < 1e-10

    def test_known_values(self):
        # sgm([1, 4], shift=1) = exp(mean(log(2), log(5))) - 1
        # = exp(log(sqrt(10))) - 1 = sqrt(10) - 1 ~ 2.162
        result = shifted_geometric_mean([1.0, 4.0], shift=1.0)
        expected = math.sqrt(10.0) - 1.0
        assert abs(result - expected) < 1e-10

    def test_zero_shift(self):
        # With shift=0, it's a standard geometric mean
        # geomean([2, 8]) = sqrt(16) = 4
        result = shifted_geometric_mean([2.0, 8.0], shift=0.0)
        assert abs(result - 4.0) < 1e-10

    def test_zero_times_with_shift(self):
        # sgm([0, 0], shift=1) = exp(mean(log(1), log(1))) - 1 = 0
        result = shifted_geometric_mean([0.0, 0.0], shift=1.0)
        assert abs(result) < 1e-10


# ─────────────────────────────────────────────────────────────
# TestSolverMetrics
# ─────────────────────────────────────────────────────────────


def _make_fake_results() -> list[dict]:
    """Create fake solve results for testing metrics computation."""
    return [
        {
            "status": "optimal",
            "objective": 10.0,
            "wall_time": 1.0,
            "node_count": 100,
            "rust_time": 0.3,
            "jax_time": 0.5,
            "python_time": 0.2,
            "gap": 0.0,
            "expected_objective": 10.0,
        },
        {
            "status": "optimal",
            "objective": 20.5,
            "wall_time": 3.0,
            "node_count": 500,
            "rust_time": 1.0,
            "jax_time": 1.5,
            "python_time": 0.5,
            "gap": 0.0,
            "expected_objective": 20.5,
        },
        {
            "status": "feasible",
            "objective": 15.0,
            "wall_time": 5.0,
            "node_count": 1000,
            "rust_time": 1.5,
            "jax_time": 2.5,
            "python_time": 1.0,
            "gap": 0.05,
            "expected_objective": 14.0,
        },
        {
            "status": "infeasible",
            "objective": None,
            "wall_time": 0.5,
            "node_count": 10,
            "rust_time": 0.1,
            "jax_time": 0.2,
            "python_time": 0.2,
            "gap": None,
        },
    ]


class TestSolverMetrics:
    def test_basic_counts(self):
        results = _make_fake_results()
        metrics = compute_solver_metrics(results)
        assert metrics.n_instances == 4
        assert metrics.n_solved == 3  # optimal + feasible
        assert metrics.n_optimal == 2

    def test_incorrect_count(self):
        results = _make_fake_results()
        metrics = compute_solver_metrics(results)
        # Instance 3 has objective=15, expected=14 -> incorrect
        assert metrics.n_incorrect == 1

    def test_wall_times(self):
        results = _make_fake_results()
        metrics = compute_solver_metrics(results)
        assert abs(metrics.total_wall_time - 9.5) < 1e-10
        # Sorted times: [0.5, 1.0, 3.0, 5.0], median = (1.0 + 3.0) / 2 = 2.0
        assert abs(metrics.median_wall_time - 2.0) < 1e-10

    def test_geomean_wall_time(self):
        results = _make_fake_results()
        metrics = compute_solver_metrics(results)
        expected = shifted_geometric_mean([1.0, 3.0, 5.0, 0.5])
        assert abs(metrics.geomean_wall_time - expected) < 1e-10

    def test_layer_fractions_sum_to_one(self):
        results = _make_fake_results()
        metrics = compute_solver_metrics(results)
        total = sum(metrics.layer_fractions.values())
        assert abs(total - 1.0) < 1e-10

    def test_mean_gap(self):
        results = _make_fake_results()
        metrics = compute_solver_metrics(results)
        # Gaps: 0.0, 0.0, 0.05 (None is excluded)
        assert abs(metrics.mean_gap - 0.05 / 3) < 1e-10

    def test_node_throughput(self):
        results = _make_fake_results()
        metrics = compute_solver_metrics(results)
        # Throughputs: 100/1=100, 500/3=166.7, 1000/5=200, 10/0.5=20
        # Sorted: [20, 100, 166.7, 200], median = (100 + 166.7) / 2
        expected = (100.0 + 500.0 / 3.0) / 2.0
        assert abs(metrics.node_throughput - expected) < 1e-6

    def test_empty_results(self):
        metrics = compute_solver_metrics([])
        assert metrics.n_instances == 0
        assert metrics.n_solved == 0
        assert metrics.total_wall_time == 0.0


# ─────────────────────────────────────────────────────────────
# TestBatchMetrics
# ─────────────────────────────────────────────────────────────


class TestBatchMetrics:
    def test_perfect_linear_scaling(self):
        # If throughput doubles when batch doubles -> efficiency = 1.0
        batch_sizes = [1, 2, 4, 8]
        throughputs = [100.0, 200.0, 400.0, 800.0]
        metrics = compute_batch_metrics(batch_sizes, throughputs)
        assert abs(metrics.scaling_efficiency - 1.0) < 1e-10

    def test_no_scaling(self):
        # Same throughput regardless of batch -> efficiency = 0.0
        batch_sizes = [1, 2, 4, 8]
        throughputs = [100.0, 100.0, 100.0, 100.0]
        metrics = compute_batch_metrics(batch_sizes, throughputs)
        assert abs(metrics.scaling_efficiency) < 1e-10

    def test_single_batch_size(self):
        metrics = compute_batch_metrics([32], [1000.0])
        assert metrics.scaling_efficiency == 0.0

    def test_stores_input_data(self):
        batch_sizes = [1, 8, 64]
        throughputs = [10.0, 50.0, 200.0]
        metrics = compute_batch_metrics(batch_sizes, throughputs)
        assert metrics.batch_sizes == batch_sizes
        assert metrics.throughputs == throughputs


# ─────────────────────────────────────────────────────────────
# TestBenchmarkResult
# ─────────────────────────────────────────────────────────────


class TestBenchmarkResult:
    def test_to_dict(self):
        r = BenchmarkResult(
            name="test",
            status="optimal",
            objective=42.0,
            expected_objective=42.0,
            wall_time=1.5,
            node_count=100,
            rust_time=0.3,
            jax_time=0.8,
            python_time=0.4,
            gap=0.0,
        )
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "optimal"
        assert d["objective"] == 42.0
        assert d["expected_objective"] == 42.0
        assert d["wall_time"] == 1.5
        assert d["node_count"] == 100
        assert d["gap"] == 0.0

    def test_to_dict_with_none(self):
        r = BenchmarkResult(
            name="infeasible",
            status="infeasible",
            objective=None,
            expected_objective=None,
            wall_time=0.1,
            node_count=0,
            rust_time=0.0,
            jax_time=0.0,
            python_time=0.1,
            gap=None,
        )
        d = r.to_dict()
        assert d["objective"] is None
        assert d["gap"] is None


# ─────────────────────────────────────────────────────────────
# TestBenchmarkRunner
# ─────────────────────────────────────────────────────────────


class TestBenchmarkRunner:
    def test_smoke_instances_exist(self):
        instances = get_smoke_instances()
        assert len(instances) == 3
        for model, name, expected_obj in instances:
            assert isinstance(model, Model)
            assert isinstance(name, str)

    @pytest.mark.slow
    def test_run_smoke_suite(self):
        config = BenchmarkConfig(suite="smoke", time_limit=30.0)
        runner = BenchmarkRunner(config)
        instances = get_smoke_instances()
        results = runner.run_suite(instances)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, BenchmarkResult)
            assert r.wall_time >= 0.0
            assert r.status in ("optimal", "feasible", "infeasible", "time_limit", "node_limit")

    def test_run_single_instance(self):
        config = BenchmarkConfig(suite="smoke", time_limit=30.0)
        runner = BenchmarkRunner(config)
        m = Model("quick_test")
        x = m.continuous("x", lb=0.0, ub=10.0)
        y = m.continuous("y", lb=0.0, ub=10.0)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)
        result = runner.run_instance(m, "quick", expected_obj=0.5)
        assert result.name == "quick"
        assert result.wall_time > 0.0

    def test_config_stored(self):
        config = BenchmarkConfig(suite="phase1", time_limit=120.0, max_nodes=50_000)
        runner = BenchmarkRunner(config)
        assert runner.config.suite == "phase1"
        assert runner.config.time_limit == 120.0
        assert runner.config.max_nodes == 50_000


# ─────────────────────────────────────────────────────────────
# TestBenchmarkExport
# ─────────────────────────────────────────────────────────────


class TestBenchmarkExport:
    def test_json_export(self):
        config = BenchmarkConfig(suite="smoke", time_limit=30.0)
        runner = BenchmarkRunner(config)

        results = [
            BenchmarkResult(
                name="prob1",
                status="optimal",
                objective=10.0,
                expected_objective=10.0,
                wall_time=1.0,
                node_count=50,
                rust_time=0.2,
                jax_time=0.5,
                python_time=0.3,
                gap=0.0,
            ),
            BenchmarkResult(
                name="prob2",
                status="feasible",
                objective=20.0,
                expected_objective=19.0,
                wall_time=3.0,
                node_count=200,
                rust_time=1.0,
                jax_time=1.2,
                python_time=0.8,
                gap=0.03,
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            runner.export_json(results, path)
            with open(path) as f:
                data = json.load(f)

            assert data["config"]["suite"] == "smoke"
            assert data["config"]["time_limit"] == 30.0
            assert len(data["results"]) == 2
            assert data["results"][0]["name"] == "prob1"
            assert data["results"][1]["status"] == "feasible"
        finally:
            os.unlink(path)

    def test_json_roundtrip(self):
        """Verify exported JSON can be read back and produces valid metrics."""
        config = BenchmarkConfig(suite="smoke")
        runner = BenchmarkRunner(config)
        results = [
            BenchmarkResult(
                name="test",
                status="optimal",
                objective=5.0,
                expected_objective=5.0,
                wall_time=0.5,
                node_count=10,
                rust_time=0.1,
                jax_time=0.3,
                python_time=0.1,
                gap=0.0,
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            runner.export_json(results, path)
            with open(path) as f:
                data = json.load(f)

            metrics = compute_solver_metrics(data["results"])
            assert metrics.n_instances == 1
            assert metrics.n_optimal == 1
            assert metrics.n_incorrect == 0
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────
# TestBatchScaling
# ─────────────────────────────────────────────────────────────


class TestBatchScaling:
    def test_batch_scaling_positive_throughputs(self):
        """Run batch scaling on a simple model and verify throughputs > 0."""
        from discopt.modeling.examples import example_simple_minlp

        model = example_simple_minlp()
        config = BenchmarkConfig(
            suite="smoke",
            batch_sizes=[1, 4, 16],
        )
        runner = BenchmarkRunner(config)
        metrics = runner.run_batch_scaling(model)

        assert isinstance(metrics, BatchMetrics)
        assert len(metrics.throughputs) == 3
        for tp in metrics.throughputs:
            assert tp > 0.0

    def test_batch_metrics_type(self):
        """Verify the return type of run_batch_scaling."""
        model = Model("tiny")
        x = model.continuous("x", lb=0.0, ub=5.0)
        model.minimize(x**2)

        config = BenchmarkConfig(suite="smoke", batch_sizes=[1, 2])
        runner = BenchmarkRunner(config)
        metrics = runner.run_batch_scaling(model)
        assert isinstance(metrics, BatchMetrics)
        assert len(metrics.batch_sizes) == 2
