"""
Tests for discopt.doe.design and discopt.doe.exploration.

Test classes:
  - TestOptimalExperiment: design optimization
  - TestExploreDesignSpace: grid evaluation
  - TestDesignResult: result object properties
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    DesignCriterion,
    DesignResult,
    ExplorationResult,
    compute_fim,
    explore_design_space,
    optimal_experiment,
)
from discopt.estimate import Experiment, ExperimentModel

# ──────────────────────────────────────────────────────────
# Helper experiments with design variables
# ──────────────────────────────────────────────────────────


class DesignableExperiment(Experiment):
    """y = k * x at design point x. Estimate k, choose x."""

    def create_model(self, **kwargs):
        m = dm.Model("designable")
        k = m.continuous("k", lb=0.01, ub=20)
        x = m.continuous("x", lb=0.1, ub=10)

        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={"x": x},
            responses={"y": k * x},
            measurement_error={"y": 0.1},
        )


class TwoDesignExperiment(Experiment):
    """y = A * exp(-k * t) at design (t, A0). Estimate k, choose t."""

    def create_model(self, **kwargs):
        m = dm.Model("two_design")
        k = m.continuous("k", lb=0.01, ub=5)
        t = m.continuous("t", lb=0.1, ub=10)

        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={"t": t},
            responses={"y": 5.0 * dm.exp(-k * t)},
            measurement_error={"y": 0.05},
        )


class TwoParamDesignExperiment(Experiment):
    """y = a*x + b at design point x. Estimate a and b."""

    def create_model(self, **kwargs):
        m = dm.Model("two_param_design")
        a = m.continuous("a", lb=-20, ub=20)
        b = m.continuous("b", lb=-20, ub=20)
        x = m.continuous("x", lb=0.1, ub=10)

        return ExperimentModel(
            model=m,
            unknown_parameters={"a": a, "b": b},
            design_inputs={"x": x},
            responses={"y": a * x + b},
            measurement_error={"y": 0.1},
        )


# ──────────────────────────────────────────────────────────
# TestOptimalExperiment
# ──────────────────────────────────────────────────────────


class TestOptimalExperiment:
    def test_d_optimal_beats_random(self):
        """Optimal design has higher det(FIM) than midpoint design."""
        exp = DesignableExperiment()
        design = optimal_experiment(
            exp,
            param_values={"k": 2.0},
            design_bounds={"x": (0.1, 10.0)},
            criterion=DesignCriterion.D_OPTIMAL,
        )

        # Compare to midpoint design
        mid_fim = compute_fim(exp, {"k": 2.0}, {"x": 5.05})
        assert design.fim_result.d_optimal >= mid_fim.d_optimal - 0.01

    def test_design_within_bounds(self):
        """Optimal design respects bounds."""
        exp = DesignableExperiment()
        design = optimal_experiment(
            exp,
            param_values={"k": 2.0},
            design_bounds={"x": (1.0, 5.0)},
        )
        assert 1.0 <= design.design["x"] <= 5.0

    def test_all_criteria_produce_results(self):
        """All four criteria produce valid DesignResult objects."""
        exp = DesignableExperiment()
        for criterion in [
            DesignCriterion.D_OPTIMAL,
            DesignCriterion.A_OPTIMAL,
            DesignCriterion.E_OPTIMAL,
            DesignCriterion.ME_OPTIMAL,
        ]:
            result = optimal_experiment(
                exp,
                param_values={"k": 2.0},
                design_bounds={"x": (0.5, 5.0)},
                criterion=criterion,
            )
            assert isinstance(result, DesignResult)
            assert "x" in result.design

    def test_prior_fim_shifts_design(self):
        """Adding strong prior can change the optimal design."""
        exp = DesignableExperiment()
        optimal_experiment(
            exp,
            param_values={"k": 2.0},
            design_bounds={"x": (0.5, 10.0)},
        )
        # With very strong prior, the design should still be valid
        prior = np.array([[1e6]])
        design_with_prior = optimal_experiment(
            exp,
            param_values={"k": 2.0},
            design_bounds={"x": (0.5, 10.0)},
            prior_fim=prior,
        )
        assert isinstance(design_with_prior, DesignResult)

    def test_linear_d_optimal_at_boundary(self):
        """For y=k*x with single measurement, D-optimal is at max |x|."""
        exp = DesignableExperiment()
        design = optimal_experiment(
            exp,
            param_values={"k": 2.0},
            design_bounds={"x": (0.1, 10.0)},
            criterion=DesignCriterion.D_OPTIMAL,
        )
        # For y = k*x, FIM = x^2/σ^2, so max x => max FIM
        assert design.design["x"] == pytest.approx(10.0, abs=0.5)


# ──────────────────────────────────────────────────────────
# TestExploreDesignSpace
# ──────────────────────────────────────────────────────────


class TestExploreDesignSpace:
    def test_1d_exploration(self):
        """Single design variable sweep."""
        exp = DesignableExperiment()
        result = explore_design_space(
            exp,
            param_values={"k": 2.0},
            design_ranges={"x": np.linspace(1.0, 5.0, 5)},
        )
        assert isinstance(result, ExplorationResult)
        assert result.metrics["log_det_fim"].shape == (5,)
        assert not np.any(np.isnan(result.metrics["log_det_fim"]))

    def test_best_point_is_at_max_x(self):
        """For y=k*x, best D-optimal point is at max x."""
        exp = DesignableExperiment()
        result = explore_design_space(
            exp,
            param_values={"k": 2.0},
            design_ranges={"x": np.array([1.0, 2.0, 5.0, 10.0])},
        )
        best = result.best_point("log_det_fim")
        assert best["x"] == pytest.approx(10.0)

    def test_monotonic_d_optimal_for_linear(self):
        """For y=k*x, D-optimality increases monotonically with |x|."""
        exp = DesignableExperiment()
        result = explore_design_space(
            exp,
            param_values={"k": 2.0},
            design_ranges={"x": np.linspace(1.0, 10.0, 10)},
        )
        d_opt = result.metrics["log_det_fim"]
        # Should be monotonically increasing
        assert np.all(np.diff(d_opt) >= -1e-10)

    def test_metrics_all_present(self):
        """All four metrics are computed."""
        exp = DesignableExperiment()
        result = explore_design_space(
            exp,
            param_values={"k": 2.0},
            design_ranges={"x": np.array([1.0, 5.0])},
        )
        assert "log_det_fim" in result.metrics
        assert "trace_fim_inv" in result.metrics
        assert "min_eigenvalue" in result.metrics
        assert "condition_number" in result.metrics


# ──────────────────────────────────────────────────────────
# TestDesignResult
# ──────────────────────────────────────────────────────────


class TestDesignResult:
    def test_summary_string(self):
        exp = DesignableExperiment()
        design = optimal_experiment(
            exp,
            param_values={"k": 2.0},
            design_bounds={"x": (1.0, 5.0)},
        )
        s = design.summary()
        assert isinstance(s, str)
        assert "Optimal Experimental Design" in s

    def test_predicted_standard_errors(self):
        exp = DesignableExperiment()
        design = optimal_experiment(
            exp,
            param_values={"k": 2.0},
            design_bounds={"x": (1.0, 10.0)},
        )
        se = design.predicted_standard_errors
        assert len(se) == 1
        assert se[0] > 0

    def test_parameter_covariance_psd(self):
        exp = DesignableExperiment()
        design = optimal_experiment(
            exp,
            param_values={"k": 2.0},
            design_bounds={"x": (1.0, 5.0)},
        )
        cov = design.parameter_covariance
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_metrics_dict(self):
        exp = DesignableExperiment()
        design = optimal_experiment(
            exp,
            param_values={"k": 2.0},
            design_bounds={"x": (1.0, 5.0)},
        )
        m = design.metrics
        assert "log_det_fim" in m
        assert "trace_fim_inv" in m
