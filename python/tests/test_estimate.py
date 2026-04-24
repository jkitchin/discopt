"""
Tests for discopt.estimate -- parameter estimation module.

Test classes:
  - TestExperimentModel: metadata validation
  - TestEstimateParameters: parameter recovery from synthetic data
  - TestEstimationResult: covariance, CI, correlation matrix
  - TestAnalyticVerification: problems with known closed-form solutions
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.estimate import (
    Experiment,
    ExperimentModel,
    estimate_parameters,
)

# ──────────────────────────────────────────────────────────
# Helper experiments
# ──────────────────────────────────────────────────────────


class LinearExperiment(Experiment):
    """y = a*x + b, estimate a and b."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("linear_est")
        a = m.continuous("a", lb=-20, ub=20)
        b = m.continuous("b", lb=-20, ub=20)

        responses = {}
        errors = {}
        for i, xi in enumerate(self.x_data):
            responses[f"y_{i}"] = a * xi + b
            errors[f"y_{i}"] = 0.1

        return ExperimentModel(
            model=m,
            unknown_parameters={"a": a, "b": b},
            design_inputs={},
            responses=responses,
            measurement_error=errors,
        )


class ExponentialExperiment(Experiment):
    """y = A * exp(-k * t), estimate A and k."""

    def __init__(self, t_data):
        self.t_data = t_data

    def create_model(self, **kwargs):
        m = dm.Model("exp_est")
        A = m.continuous("A", lb=0.1, ub=20)
        k = m.continuous("k", lb=0.01, ub=5)

        responses = {}
        errors = {}
        for i, ti in enumerate(self.t_data):
            responses[f"y_{i}"] = A * dm.exp(-k * ti)
            errors[f"y_{i}"] = 0.05

        return ExperimentModel(
            model=m,
            unknown_parameters={"A": A, "k": k},
            design_inputs={},
            responses=responses,
            measurement_error=errors,
        )


class SingleParamExperiment(Experiment):
    """y = k * x, estimate k only."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("single_est")
        k = m.continuous("k", lb=0.01, ub=20)

        responses = {}
        errors = {}
        for i, xi in enumerate(self.x_data):
            responses[f"y_{i}"] = k * xi
            errors[f"y_{i}"] = 0.1

        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={},
            responses=responses,
            measurement_error=errors,
        )


# ──────────────────────────────────────────────────────────
# TestExperimentModel
# ──────────────────────────────────────────────────────────


class TestExperimentModel:
    def test_create_valid(self):
        m = dm.Model()
        k = m.continuous("k", lb=0, ub=10)
        m.minimize(k)
        em = ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={},
            responses={"y": k * 2},
            measurement_error={"y": 0.1},
        )
        assert em.n_parameters == 1
        assert em.n_responses == 1
        assert em.parameter_names == ["k"]
        assert em.response_names == ["y"]

    def test_mismatched_keys_raises(self):
        m = dm.Model()
        k = m.continuous("k", lb=0, ub=10)
        with pytest.raises(ValueError, match="keys must match"):
            ExperimentModel(
                model=m,
                unknown_parameters={"k": k},
                design_inputs={},
                responses={"y1": k, "y2": k * 2},
                measurement_error={"y1": 0.1},  # missing y2
            )

    def test_extra_error_key_raises(self):
        m = dm.Model()
        k = m.continuous("k", lb=0, ub=10)
        with pytest.raises(ValueError, match="keys must match"):
            ExperimentModel(
                model=m,
                unknown_parameters={"k": k},
                design_inputs={},
                responses={"y": k},
                measurement_error={"y": 0.1, "z": 0.2},
            )


# ──────────────────────────────────────────────────────────
# TestEstimateParameters
# ──────────────────────────────────────────────────────────


class TestEstimateParameters:
    def test_linear_regression(self):
        """y = 2*x + 1: recover a=2, b=1 from exact data."""
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 2.0 * x_data + 1.0  # exact, no noise

        exp = LinearExperiment(x_data)
        data = {f"y_{i}": y_data[i] for i in range(len(x_data))}

        result = estimate_parameters(exp, data)
        assert result.parameters["a"] == pytest.approx(2.0, abs=1e-3)
        assert result.parameters["b"] == pytest.approx(1.0, abs=1e-3)

    def test_exponential_decay(self):
        """y = 5*exp(-0.3*t): recover A=5, k=0.3."""
        t_data = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        y_data = 5.0 * np.exp(-0.3 * t_data)

        exp = ExponentialExperiment(t_data)
        data = {f"y_{i}": y_data[i] for i in range(len(t_data))}

        result = estimate_parameters(exp, data)
        assert result.parameters["A"] == pytest.approx(5.0, abs=0.1)
        assert result.parameters["k"] == pytest.approx(0.3, abs=0.05)

    def test_single_parameter(self):
        """y = 3*x: recover k=3."""
        x_data = np.array([1.0, 2.0, 3.0])
        y_data = 3.0 * x_data

        exp = SingleParamExperiment(x_data)
        data = {f"y_{i}": y_data[i] for i in range(len(x_data))}

        result = estimate_parameters(exp, data)
        assert result.parameters["k"] == pytest.approx(3.0, abs=1e-3)

    def test_data_key_mismatch_raises(self):
        """Data keys not matching responses raises ValueError."""
        exp = SingleParamExperiment(np.array([1.0]))
        with pytest.raises(ValueError, match="not in response"):
            estimate_parameters(exp, {"wrong_key": 1.0})

    def test_n_observations_correct(self):
        """n_observations matches number of data points."""
        x_data = np.array([1.0, 2.0, 3.0, 4.0])
        y_data = 2.0 * x_data

        exp = SingleParamExperiment(x_data)
        data = {f"y_{i}": y_data[i] for i in range(len(x_data))}

        result = estimate_parameters(exp, data)
        assert result.n_observations == 4


# ──────────────────────────────────────────────────────────
# TestEstimationResult
# ──────────────────────────────────────────────────────────


class TestEstimationResult:
    def _get_result(self):
        """Helper: fit y = k*x, k_true = 2.0."""
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 2.0 * x_data
        exp = SingleParamExperiment(x_data)
        data = {f"y_{i}": y_data[i] for i in range(len(x_data))}
        return estimate_parameters(exp, data)

    def test_covariance_positive_semidefinite(self):
        result = self._get_result()
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert np.all(eigenvalues >= -1e-10)

    def test_fim_symmetric(self):
        result = self._get_result()
        np.testing.assert_allclose(result.fim, result.fim.T, atol=1e-10)

    def test_fim_positive_semidefinite(self):
        result = self._get_result()
        eigenvalues = np.linalg.eigvalsh(result.fim)
        assert np.all(eigenvalues >= -1e-10)

    def test_correlation_diagonal_ones(self):
        """Diagonal of correlation matrix should be 1.0."""
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 2.0 * x_data + 1.0
        exp = LinearExperiment(x_data)
        data = {f"y_{i}": y_data[i] for i in range(len(x_data))}
        result = estimate_parameters(exp, data)
        corr = result.correlation_matrix
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_confidence_intervals_contain_true_value(self):
        """95% CI should contain the true parameter value for exact data."""
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 2.0 * x_data + 1.0
        exp = LinearExperiment(x_data)
        data = {f"y_{i}": y_data[i] for i in range(len(x_data))}
        result = estimate_parameters(exp, data)
        ci = result.confidence_intervals
        lo_a, hi_a = ci["a"]
        lo_b, hi_b = ci["b"]
        assert lo_a <= 2.0 <= hi_a
        assert lo_b <= 1.0 <= hi_b

    def test_summary_returns_string(self):
        result = self._get_result()
        s = result.summary()
        assert isinstance(s, str)
        assert "Parameter Estimation Results" in s
        assert "k" in s

    def test_standard_errors_dict(self):
        result = self._get_result()
        se = result.standard_errors
        assert "k" in se
        assert se["k"] >= 0


# ──────────────────────────────────────────────────────────
# TestAnalyticVerification
# ──────────────────────────────────────────────────────────


class TestAnalyticVerification:
    def test_linear_fim_equals_XtSigmaX(self):
        """For y = k*x, FIM = sum(x_i^2) / σ^2."""
        x_data = np.array([1.0, 2.0, 3.0])
        sigma = 0.1
        y_data = 2.0 * x_data

        exp = SingleParamExperiment(x_data)
        data = {f"y_{i}": y_data[i] for i in range(len(x_data))}

        result = estimate_parameters(exp, data)

        # Analytic FIM for y = k*x: FIM = sum(x_i^2) / σ^2
        fim_analytic = np.sum(x_data**2) / sigma**2
        np.testing.assert_allclose(result.fim[0, 0], fim_analytic, rtol=1e-4)

    def test_linear_2param_fim(self):
        """For y = a*x + b, FIM = X^T Σ^{-1} X where X = [[x1,1],[x2,1],...]."""
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sigma = 0.1
        y_data = 2.0 * x_data + 1.0

        exp = LinearExperiment(x_data)
        data = {f"y_{i}": y_data[i] for i in range(len(x_data))}

        result = estimate_parameters(exp, data)

        # Analytic: design matrix X = [[x1,1], [x2,1], ...]
        X = np.column_stack([x_data, np.ones_like(x_data)])
        fim_analytic = X.T @ X / sigma**2
        np.testing.assert_allclose(result.fim, fim_analytic, rtol=1e-3)

    def test_single_param_variance(self):
        """Var(k) = σ^2 / sum(x_i^2) for y = k*x."""
        x_data = np.array([1.0, 2.0, 3.0])
        sigma = 0.1

        exp = SingleParamExperiment(x_data)
        y_data = 2.0 * x_data
        data = {f"y_{i}": y_data[i] for i in range(len(x_data))}

        result = estimate_parameters(exp, data)

        var_analytic = sigma**2 / np.sum(x_data**2)
        np.testing.assert_allclose(result.covariance[0, 0], var_analytic, rtol=1e-3)

    def test_array_observations_narrow_covariance(self):
        """Repeated observations of the same response tighten the covariance.

        Regression test: ``estimate_parameters`` used to silently drop all
        but the first element of an array-valued observation, which stopped
        :func:`sequential_doe` from narrowing CIs as data accumulated.
        Under the fix, for ``y = k*x`` with ``n`` repeated measurements per
        design point, Var(k) scales like ``1/n``.
        """
        x_data = np.array([1.0, 2.0, 3.0])
        exp = SingleParamExperiment(x_data)

        # Single-observation baseline: the analytic variance.
        baseline_data = {f"y_{i}": 2.0 * x_data[i] for i in range(len(x_data))}
        var_single = estimate_parameters(exp, baseline_data).covariance[0, 0]

        # Five identical replicates per design point should give variance
        # exactly 1/5 of the single-observation variance.
        n_rep = 5
        rep_data = {f"y_{i}": np.full(n_rep, 2.0 * x_data[i]) for i in range(len(x_data))}
        result_rep = estimate_parameters(exp, rep_data)

        np.testing.assert_allclose(result_rep.covariance[0, 0], var_single / n_rep, rtol=1e-3)
        assert result_rep.n_observations == n_rep * len(x_data)
