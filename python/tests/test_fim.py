"""
Tests for discopt.doe.fim -- Fisher Information Matrix computation.

Test classes:
  - TestFIMComputation: basic FIM computation and properties
  - TestFIMMetrics: D/A/E/ME optimality criteria
  - TestFIMAutodiffVsFiniteDifference: cross-validation
  - TestIdentifiability: parameter identifiability analysis
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import check_identifiability, compute_fim
from discopt.estimate import Experiment, ExperimentModel

# ──────────────────────────────────────────────────────────
# Helper experiments
# ──────────────────────────────────────────────────────────


class LinearExperiment(Experiment):
    """y_i = a*x_i + b for multiple x_i."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("linear")
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


class SingleParamExperiment(Experiment):
    """y = k * x."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("single")
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


class ExponentialExperiment(Experiment):
    """y = A * exp(-k * t)."""

    def __init__(self, t_data):
        self.t_data = t_data

    def create_model(self, **kwargs):
        m = dm.Model("exponential")
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


class UnidentifiableExperiment(Experiment):
    """y = (a * b) * x: a and b not individually identifiable."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("unidentifiable")
        a = m.continuous("a", lb=0.1, ub=10)
        b = m.continuous("b", lb=0.1, ub=10)

        responses = {}
        errors = {}
        for i, xi in enumerate(self.x_data):
            responses[f"y_{i}"] = a * b * xi
            errors[f"y_{i}"] = 0.1

        return ExperimentModel(
            model=m,
            unknown_parameters={"a": a, "b": b},
            design_inputs={},
            responses=responses,
            measurement_error=errors,
        )


# ──────────────────────────────────────────────────────────
# TestFIMComputation
# ──────────────────────────────────────────────────────────


class TestFIMComputation:
    def test_linear_fim_matches_analytic(self):
        """For y = a*x + b, FIM = X^T Σ^{-1} X."""
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sigma = 0.1
        exp = LinearExperiment(x_data)

        result = compute_fim(exp, {"a": 2.0, "b": 1.0})

        # Analytic FIM
        X = np.column_stack([x_data, np.ones_like(x_data)])
        fim_analytic = X.T @ X / sigma**2
        np.testing.assert_allclose(result.fim, fim_analytic, rtol=1e-3)

    def test_single_param_fim(self):
        """For y = k*x, FIM = sum(x_i^2) / σ^2."""
        x_data = np.array([1.0, 2.0, 3.0])
        sigma = 0.1
        exp = SingleParamExperiment(x_data)

        result = compute_fim(exp, {"k": 2.0})
        fim_analytic = np.sum(x_data**2) / sigma**2
        np.testing.assert_allclose(result.fim[0, 0], fim_analytic, rtol=1e-4)

    def test_fim_symmetric(self):
        """FIM is always symmetric."""
        exp = LinearExperiment(np.array([1.0, 2.0, 3.0]))
        result = compute_fim(exp, {"a": 2.0, "b": 1.0})
        np.testing.assert_allclose(result.fim, result.fim.T, atol=1e-10)

    def test_fim_positive_semidefinite(self):
        """FIM eigenvalues are non-negative."""
        exp = ExponentialExperiment(np.array([0.5, 1.0, 2.0, 5.0]))
        result = compute_fim(exp, {"A": 5.0, "k": 0.3})
        eigenvalues = np.linalg.eigvalsh(result.fim)
        assert np.all(eigenvalues >= -1e-10)

    def test_fim_with_prior(self):
        """Prior FIM is additive."""
        exp = SingleParamExperiment(np.array([1.0, 2.0]))
        result_no_prior = compute_fim(exp, {"k": 2.0})
        prior = np.array([[100.0]])
        result_with_prior = compute_fim(exp, {"k": 2.0}, prior_fim=prior)
        np.testing.assert_allclose(
            result_with_prior.fim,
            result_no_prior.fim + prior,
            atol=1e-10,
        )

    def test_more_measurements_larger_fim(self):
        """More data points => larger FIM determinant."""
        exp_3 = SingleParamExperiment(np.array([1.0, 2.0, 3.0]))
        exp_5 = SingleParamExperiment(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        fim_3 = compute_fim(exp_3, {"k": 2.0})
        fim_5 = compute_fim(exp_5, {"k": 2.0})
        assert fim_5.d_optimal > fim_3.d_optimal

    def test_smaller_sigma_larger_fim(self):
        """Smaller measurement error => larger FIM."""

        class SmallSigmaExperiment(Experiment):
            def create_model(self, **kwargs):
                m = dm.Model()
                k = m.continuous("k", lb=0.01, ub=20)
                return ExperimentModel(
                    model=m,
                    unknown_parameters={"k": k},
                    design_inputs={},
                    responses={"y": k * 2.0},
                    measurement_error={"y": 0.01},  # 10x smaller
                )

        class LargeSigmaExperiment(Experiment):
            def create_model(self, **kwargs):
                m = dm.Model()
                k = m.continuous("k", lb=0.01, ub=20)
                return ExperimentModel(
                    model=m,
                    unknown_parameters={"k": k},
                    design_inputs={},
                    responses={"y": k * 2.0},
                    measurement_error={"y": 0.1},
                )

        fim_small = compute_fim(SmallSigmaExperiment(), {"k": 2.0})
        fim_large = compute_fim(LargeSigmaExperiment(), {"k": 2.0})
        assert fim_small.d_optimal > fim_large.d_optimal

    def test_jacobian_shape(self):
        """Jacobian has shape (n_responses, n_params)."""
        exp = LinearExperiment(np.array([1.0, 2.0, 3.0]))
        result = compute_fim(exp, {"a": 2.0, "b": 1.0})
        assert result.jacobian.shape == (3, 2)

    def test_result_names(self):
        """FIMResult has correct parameter and response names."""
        exp = LinearExperiment(np.array([1.0, 2.0]))
        result = compute_fim(exp, {"a": 2.0, "b": 1.0})
        assert result.parameter_names == ["a", "b"]
        assert result.response_names == ["y_0", "y_1"]


# ──────────────────────────────────────────────────────────
# TestFIMMetrics
# ──────────────────────────────────────────────────────────


class TestFIMMetrics:
    def _get_fim_result(self):
        exp = LinearExperiment(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        return compute_fim(exp, {"a": 2.0, "b": 1.0})

    def test_d_optimal_equals_log_det(self):
        r = self._get_fim_result()
        assert r.d_optimal == pytest.approx(np.log(np.linalg.det(r.fim)))

    def test_a_optimal_equals_trace_inv(self):
        r = self._get_fim_result()
        assert r.a_optimal == pytest.approx(np.trace(np.linalg.inv(r.fim)))

    def test_e_optimal_equals_min_eigenvalue(self):
        r = self._get_fim_result()
        assert r.e_optimal == pytest.approx(np.min(np.linalg.eigvalsh(r.fim)))

    def test_me_optimal_equals_condition_number(self):
        r = self._get_fim_result()
        assert r.me_optimal == pytest.approx(np.linalg.cond(r.fim))

    def test_metrics_dict_keys(self):
        r = self._get_fim_result()
        m = r.metrics
        assert set(m.keys()) == {
            "log_det_fim",
            "trace_fim_inv",
            "min_eigenvalue",
            "condition_number",
        }


# ──────────────────────────────────────────────────────────
# TestFIMAutodiffVsFiniteDifference
# ──────────────────────────────────────────────────────────


class TestFIMAutodiffVsFiniteDifference:
    def test_linear_model(self):
        """Autodiff and FD FIM agree for linear model."""
        exp = LinearExperiment(np.array([1.0, 2.0, 3.0]))
        pv = {"a": 2.0, "b": 1.0}
        fim_auto = compute_fim(exp, pv, method="autodiff")
        fim_fd = compute_fim(exp, pv, method="finite_difference")
        np.testing.assert_allclose(fim_auto.fim, fim_fd.fim, rtol=1e-4)

    def test_exponential_model(self):
        """Autodiff and FD FIM agree for nonlinear model."""
        exp = ExponentialExperiment(np.array([0.5, 1.0, 2.0, 5.0]))
        pv = {"A": 5.0, "k": 0.3}
        fim_auto = compute_fim(exp, pv, method="autodiff")
        fim_fd = compute_fim(exp, pv, method="finite_difference")
        np.testing.assert_allclose(fim_auto.fim, fim_fd.fim, rtol=1e-3)

    def test_jacobian_agreement(self):
        """Autodiff and FD Jacobians agree."""
        exp = ExponentialExperiment(np.array([1.0, 2.0]))
        pv = {"A": 5.0, "k": 0.3}
        r_auto = compute_fim(exp, pv, method="autodiff")
        r_fd = compute_fim(exp, pv, method="finite_difference")
        np.testing.assert_allclose(r_auto.jacobian, r_fd.jacobian, rtol=1e-3)


# ──────────────────────────────────────────────────────────
# TestIdentifiability
# ──────────────────────────────────────────────────────────


class TestIdentifiability:
    def test_identifiable_model(self):
        """Well-posed linear model is identifiable."""
        exp = LinearExperiment(np.array([1.0, 2.0, 3.0]))
        result = check_identifiability(exp, {"a": 2.0, "b": 1.0})
        assert result.is_identifiable
        assert result.fim_rank == 2
        assert result.n_parameters == 2
        assert result.problematic_parameters == []

    def test_unidentifiable_product(self):
        """y = (a*b)*x: a and b not individually identifiable."""
        exp = UnidentifiableExperiment(np.array([1.0, 2.0, 3.0]))
        result = check_identifiability(exp, {"a": 2.0, "b": 3.0})
        assert not result.is_identifiable
        assert result.fim_rank == 1
        assert len(result.problematic_parameters) == 1

    def test_single_param_always_identifiable(self):
        """Single parameter model with data is always identifiable."""
        exp = SingleParamExperiment(np.array([1.0, 2.0]))
        result = check_identifiability(exp, {"k": 2.0})
        assert result.is_identifiable
        assert result.fim_rank == 1
