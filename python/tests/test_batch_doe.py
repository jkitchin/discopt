"""
Tests for discopt.doe batch / parallel experimental design.

Test classes:
  - TestGreedyBatch: greedy strategy correctness and edge cases
  - TestJointBatch: joint strategy (Galvanina-style) correctness
  - TestPenalizedBatch: distance-penalized diversity
  - TestBatchDesignResult: dataclass invariants
  - TestSequentialBatch: sequential_doe integration with experiments_per_round
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    BatchDesignResult,
    BatchStrategy,
    DesignCriterion,
    DesignResult,
    batch_optimal_experiment,
    optimal_experiment,
    sequential_doe,
)
from discopt.estimate import Experiment, ExperimentModel

# Each batch DOE test runs several real NLP solves; the suite is a research
# feature that rarely regresses from typical PRs.  Keep it on the nightly
# `slow` lane and out of the PR fast path.
pytestmark = pytest.mark.slow

# ──────────────────────────────────────────────────────────
# Helper experiments (mirrored from test_doe.py, kept local
# to avoid cross-file imports)
# ──────────────────────────────────────────────────────────


class LinearExperiment(Experiment):
    """y = a*x + b at design point x. Two parameters."""

    def create_model(self, **kwargs):
        m = dm.Model("linear")
        a = m.continuous("a", lb=-20, ub=20)
        b = m.continuous("b", lb=-20, ub=20)
        x = m.continuous("x", lb=0.0, ub=1.0)

        return ExperimentModel(
            model=m,
            unknown_parameters={"a": a, "b": b},
            design_inputs={"x": x},
            responses={"y": a * x + b},
            measurement_error={"y": 0.1},
        )


class ExpDecayExperiment(Experiment):
    """y = A*exp(-k*t). One design (t), one unknown parameter (k)."""

    def create_model(self, **kwargs):
        m = dm.Model("expdecay")
        k = m.continuous("k", lb=0.01, ub=5)
        t = m.continuous("t", lb=0.1, ub=10)

        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={"t": t},
            responses={"y": 5.0 * dm.exp(-k * t)},
            measurement_error={"y": 0.05},
        )


# ──────────────────────────────────────────────────────────
# TestGreedyBatch
# ──────────────────────────────────────────────────────────


class TestGreedyBatch:
    @pytest.mark.slow
    @pytest.mark.integration
    def test_returns_requested_count(self):
        exp = ExpDecayExperiment()
        res = batch_optimal_experiment(
            exp,
            param_values={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
            n_experiments=3,
            strategy=BatchStrategy.GREEDY,
        )
        assert isinstance(res, BatchDesignResult)
        assert res.n_experiments == 3
        assert len(res.designs) == 3
        assert len(res.fim_results) == 3
        assert res.joint_fim.shape == (1, 1)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_fim_additivity(self):
        """joint_fim == sum(per-point FIMs) + prior when prior=None."""
        exp = LinearExperiment()
        res = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=3,
            strategy=BatchStrategy.GREEDY,
        )
        summed = sum(r.fim for r in res.fim_results)
        np.testing.assert_allclose(res.joint_fim, summed, rtol=1e-9, atol=1e-12)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_per_round_monotone_d_optimal(self):
        """log det of running FIM is non-decreasing when adding experiments."""
        exp = LinearExperiment()
        res = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=4,
            strategy=BatchStrategy.GREEDY,
            criterion=DesignCriterion.D_OPTIMAL,
        )
        deltas = np.diff(res.per_round_criterion)
        assert np.all(deltas >= -1e-9)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_n_equals_one_matches_optimal_experiment(self):
        exp = ExpDecayExperiment()
        single = optimal_experiment(
            exp,
            param_values={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
        )
        batch = batch_optimal_experiment(
            exp,
            param_values={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
            n_experiments=1,
            strategy=BatchStrategy.GREEDY,
        )
        assert batch.criterion_value == pytest.approx(single.criterion_value, abs=1e-4)
        dr = batch.to_design_result()
        assert isinstance(dr, DesignResult)
        assert dr.design["t"] == pytest.approx(single.design["t"], abs=1e-2)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_prior_fim_respected(self):
        """A huge prior changes what greedy picks."""
        exp = ExpDecayExperiment()
        no_prior = batch_optimal_experiment(
            exp,
            param_values={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
            n_experiments=2,
            strategy=BatchStrategy.GREEDY,
            prior_fim=None,
        )
        big_prior = batch_optimal_experiment(
            exp,
            param_values={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
            n_experiments=2,
            strategy=BatchStrategy.GREEDY,
            prior_fim=np.array([[1e6]]),
        )
        # Joint FIM with the huge prior must reflect it.
        assert big_prior.joint_fim[0, 0] > no_prior.joint_fim[0, 0] + 1e5

    @pytest.mark.slow
    @pytest.mark.integration
    def test_linear_two_param_joint_fim_invertible(self):
        """For y=a*x+b, 2 greedy picks make the joint FIM well-posed."""
        exp = LinearExperiment()
        res = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=2,
            strategy=BatchStrategy.GREEDY,
            criterion=DesignCriterion.D_OPTIMAL,
        )
        # Each single point gives a singular 2×2 FIM; greedy should still
        # recover enough information that the joint FIM is invertible.
        assert np.linalg.matrix_rank(res.joint_fim) == 2
        for d in res.designs:
            assert 0.0 <= d["x"] <= 1.0

    def test_invalid_strategy_raises(self):
        exp = ExpDecayExperiment()
        with pytest.raises(ValueError, match="Unknown batch strategy"):
            batch_optimal_experiment(
                exp,
                param_values={"k": 0.5},
                design_bounds={"t": (0.1, 10.0)},
                n_experiments=2,
                strategy="not_a_strategy",
            )

    def test_invalid_n_experiments_raises(self):
        exp = ExpDecayExperiment()
        with pytest.raises(ValueError, match="n_experiments"):
            batch_optimal_experiment(
                exp,
                param_values={"k": 0.5},
                design_bounds={"t": (0.1, 10.0)},
                n_experiments=0,
            )


class TestBatchDesignResult:
    @pytest.mark.slow
    @pytest.mark.integration
    def test_summary_runs(self):
        exp = LinearExperiment()
        res = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=2,
            strategy=BatchStrategy.GREEDY,
        )
        s = res.summary()
        assert "Batch Optimal Design" in s
        assert "Experiment 1" in s
        assert "Experiment 2" in s

    @pytest.mark.slow
    @pytest.mark.integration
    def test_covariance_psd(self):
        exp = LinearExperiment()
        res = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=2,
            strategy=BatchStrategy.GREEDY,
        )
        cov = res.parameter_covariance
        eigs = np.linalg.eigvalsh(cov)
        assert np.all(eigs >= -1e-9)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_to_design_result_requires_one(self):
        exp = LinearExperiment()
        res = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=2,
            strategy=BatchStrategy.GREEDY,
        )
        with pytest.raises(ValueError, match="n_experiments == 1"):
            res.to_design_result()


# ──────────────────────────────────────────────────────────
# TestJointBatch (filled in by commit 4)
# ──────────────────────────────────────────────────────────


class TestJointBatch:
    @pytest.mark.slow
    @pytest.mark.integration
    def test_joint_ge_greedy_linear(self):
        exp = LinearExperiment()
        greedy = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=2,
            strategy=BatchStrategy.GREEDY,
            criterion=DesignCriterion.D_OPTIMAL,
        )
        joint = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=2,
            strategy=BatchStrategy.JOINT,
            criterion=DesignCriterion.D_OPTIMAL,
        )
        assert joint.criterion_value >= greedy.criterion_value - 1e-6

    def test_joint_linear_picks_boundaries(self):
        exp = LinearExperiment()
        joint = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=2,
            strategy=BatchStrategy.JOINT,
            criterion=DesignCriterion.D_OPTIMAL,
        )
        xs = sorted(d["x"] for d in joint.designs)
        assert xs[0] == pytest.approx(0.0, abs=0.05)
        assert xs[1] == pytest.approx(1.0, abs=0.05)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_joint_n_equals_one_matches_single(self):
        exp = ExpDecayExperiment()
        single = optimal_experiment(
            exp,
            param_values={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
        )
        joint = batch_optimal_experiment(
            exp,
            param_values={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
            n_experiments=1,
            strategy=BatchStrategy.JOINT,
        )
        assert joint.criterion_value == pytest.approx(single.criterion_value, abs=1e-3)


# ──────────────────────────────────────────────────────────
# TestPenalizedBatch (filled in by commit 5)
# ──────────────────────────────────────────────────────────


class TestPenalizedBatch:
    @pytest.mark.slow
    @pytest.mark.integration
    def test_min_distance_enforced(self):
        exp = LinearExperiment()
        res = batch_optimal_experiment(
            exp,
            param_values={"a": 1.0, "b": 0.0},
            design_bounds={"x": (0.0, 1.0)},
            n_experiments=3,
            strategy=BatchStrategy.PENALIZED,
            criterion=DesignCriterion.D_OPTIMAL,
            min_distance=0.2,
        )
        xs = sorted(d["x"] for d in res.designs)
        # Normalised by (1.0-0.0)=1.0, so distances match absolute diffs
        for i in range(len(xs) - 1):
            assert xs[i + 1] - xs[i] >= 0.2 - 1e-9

    @pytest.mark.slow
    @pytest.mark.integration
    def test_penalized_n_equals_one(self):
        exp = ExpDecayExperiment()
        single = optimal_experiment(
            exp,
            param_values={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
        )
        pen = batch_optimal_experiment(
            exp,
            param_values={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
            n_experiments=1,
            strategy=BatchStrategy.PENALIZED,
            min_distance=0.0,
        )
        assert pen.criterion_value == pytest.approx(single.criterion_value, abs=1e-4)


# ──────────────────────────────────────────────────────────
# TestSequentialBatch
# ──────────────────────────────────────────────────────────


class TestSequentialBatch:
    @pytest.mark.slow
    @pytest.mark.integration
    def test_batched_sequential_records_batch_and_calls_runner(self):
        exp = ExpDecayExperiment()
        k_true = 0.7

        calls: list[dict[str, float]] = []

        def run(design: dict[str, float]) -> dict[str, float]:
            calls.append(dict(design))
            t = design["t"]
            return {"y": float(5.0 * np.exp(-k_true * t))}

        history = sequential_doe(
            exp,
            initial_data={"y": np.array([5.0, 3.0])},
            initial_guess={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
            n_rounds=3,
            experiments_per_round=2,
            batch_strategy=BatchStrategy.GREEDY,
            run_experiment=run,
        )
        assert len(history) == 3
        assert len(calls) == 2 * 3  # 2 experiments per round, 3 rounds
        for r in history:
            assert isinstance(r.design, BatchDesignResult)
            assert r.data_collected is not None
            assert len(np.atleast_1d(r.data_collected["y"])) == 2

    @pytest.mark.slow
    @pytest.mark.integration
    def test_single_experiment_per_round_preserves_behavior(self):
        exp = ExpDecayExperiment()

        def run(design: dict[str, float]) -> dict[str, float]:
            t = design["t"]
            return {"y": float(5.0 * np.exp(-0.7 * t))}

        history = sequential_doe(
            exp,
            initial_data={"y": np.array([5.0, 3.0])},
            initial_guess={"k": 0.5},
            design_bounds={"t": (0.1, 10.0)},
            n_rounds=2,
            experiments_per_round=1,
            run_experiment=run,
        )
        assert len(history) == 2
        for r in history:
            assert isinstance(r.design, DesignResult)

    def test_invalid_experiments_per_round_raises(self):
        exp = ExpDecayExperiment()
        with pytest.raises(ValueError, match="experiments_per_round"):
            sequential_doe(
                exp,
                initial_data={"y": np.array([5.0])},
                initial_guess={"k": 0.5},
                design_bounds={"t": (0.1, 10.0)},
                experiments_per_round=0,
            )
