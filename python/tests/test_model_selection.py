"""Tests for the post-experiment selection layer.

Covers AIC / AICc / BIC scoring, AIC weights softmax, the nested
likelihood-ratio test (Wilks 1938), and the Vuong (1989) non-nested
test.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    ModelSelectionResult,
    likelihood_ratio_test,
    model_selection,
    vuong_test,
)
from discopt.estimate import Experiment, ExperimentModel, estimate_parameters

# ─────────────────────────────────────────────────────────────────────
# Helper experiments
# ─────────────────────────────────────────────────────────────────────


class InterceptOnly(Experiment):
    """``y_i = b`` for every observation (1-parameter constant model)."""

    def __init__(self, n_pts):
        self.n = n_pts

    def create_model(self, **kw):
        m = dm.Model("intercept")
        b = m.continuous("b", lb=-50, ub=50)
        responses = {f"y_{i}": b + 0.0 * b for i in range(self.n)}
        errors = {k: 0.5 for k in responses}
        return ExperimentModel(m, {"b": b}, {}, responses, errors)


class LinearAB(Experiment):
    """``y_i = a*x_i + b`` (2-parameter linear)."""

    def __init__(self, xs):
        self.xs = list(xs)

    def create_model(self, **kw):
        m = dm.Model("linear_ab")
        a = m.continuous("a", lb=-50, ub=50)
        b = m.continuous("b", lb=-50, ub=50)
        responses = {f"y_{i}": a * x + b for i, x in enumerate(self.xs)}
        errors = {k: 0.5 for k in responses}
        return ExperimentModel(m, {"a": a, "b": b}, {}, responses, errors)


class QuadraticABC(Experiment):
    """``y_i = a*x_i² + b*x_i + c`` (3-parameter)."""

    def __init__(self, xs):
        self.xs = list(xs)

    def create_model(self, **kw):
        m = dm.Model("quad_abc")
        a = m.continuous("a", lb=-50, ub=50)
        b = m.continuous("b", lb=-50, ub=50)
        c = m.continuous("c", lb=-50, ub=50)
        responses = {f"y_{i}": a * x * x + b * x + c for i, x in enumerate(self.xs)}
        errors = {k: 0.5 for k in responses}
        return ExperimentModel(m, {"a": a, "b": b, "c": c}, {}, responses, errors)


@pytest.fixture
def linear_data():
    np.random.seed(11)
    xs = np.linspace(0, 5, 10)
    a, b = 2.0, 1.0
    y = a * xs + b + np.random.normal(0, 0.5, 10)
    data = {f"y_{i}": y[i] for i in range(len(xs))}
    return xs, data


# ─────────────────────────────────────────────────────────────────────
# AIC / BIC / AICc
# ─────────────────────────────────────────────────────────────────────


class TestModelSelection:
    def test_aic_picks_correct_model(self, linear_data):
        """Linear data should rank linear above the redundant quadratic."""
        xs, data = linear_data
        est_lin = estimate_parameters(LinearAB(xs), data)
        est_quad = estimate_parameters(QuadraticABC(xs), data)
        res = model_selection({"linear": est_lin, "quadratic": est_quad}, method="aic")
        assert res.best_model == "linear"
        assert res.weights["linear"] > res.weights["quadratic"]

    def test_aic_score_formula(self, linear_data):
        """AIC = 2p + objective (one-line derivation from deviance)."""
        xs, data = linear_data
        est_lin = estimate_parameters(LinearAB(xs), data)
        res = model_selection(
            {"linear": est_lin, "intercept": estimate_parameters(InterceptOnly(len(xs)), data)},
            method="aic",
        )
        np.testing.assert_allclose(res.scores["linear"], 2 * 2 + est_lin.objective, atol=1e-10)

    def test_bic_score_formula(self, linear_data):
        xs, data = linear_data
        est_lin = estimate_parameters(LinearAB(xs), data)
        est_int = estimate_parameters(InterceptOnly(len(xs)), data)
        res = model_selection({"linear": est_lin, "intercept": est_int}, method="bic")
        np.testing.assert_allclose(
            res.scores["linear"], 2 * np.log(len(xs)) + est_lin.objective, atol=1e-10
        )

    def test_aic_weights_sum_to_one(self, linear_data):
        xs, data = linear_data
        est_lin = estimate_parameters(LinearAB(xs), data)
        est_quad = estimate_parameters(QuadraticABC(xs), data)
        res = model_selection({"linear": est_lin, "quadratic": est_quad}, method="aic")
        assert sum(res.weights.values()) == pytest.approx(1.0, abs=1e-12)

    def test_aicc_correction(self, linear_data):
        xs, data = linear_data
        n = len(xs)
        est_lin = estimate_parameters(LinearAB(xs), data)
        p = 2
        expected = 2 * p + est_lin.objective + (2 * p * (p + 1)) / (n - p - 1)
        res = model_selection(
            {"linear": est_lin, "intercept": estimate_parameters(InterceptOnly(n), data)},
            method="aicc",
        )
        np.testing.assert_allclose(res.scores["linear"], expected, atol=1e-10)

    def test_single_model_raises(self, linear_data):
        xs, data = linear_data
        est = estimate_parameters(LinearAB(xs), data)
        with pytest.raises(ValueError, match="at least 2"):
            model_selection({"only": est}, method="aic")

    def test_mismatched_n_obs_raises(self):
        xs1 = np.linspace(0, 1, 5)
        xs2 = np.linspace(0, 1, 8)
        data1 = {f"y_{i}": float(x) for i, x in enumerate(xs1)}
        data2 = {f"y_{i}": float(x) for i, x in enumerate(xs2)}
        est1 = estimate_parameters(LinearAB(xs1), data1)
        est2 = estimate_parameters(LinearAB(xs2), data2)
        with pytest.raises(ValueError, match="n_observations"):
            model_selection({"a": est1, "b": est2})


# ─────────────────────────────────────────────────────────────────────
# Likelihood-ratio test
# ─────────────────────────────────────────────────────────────────────


class TestLikelihoodRatio:
    def test_rejects_nested_when_full_truly_better(self, linear_data):
        """Linear data: intercept-only (1 param) nested in linear (2 params).
        LRT should overwhelmingly reject H₀."""
        xs, data = linear_data
        n = len(xs)
        est_int = estimate_parameters(InterceptOnly(n), data)
        est_lin = estimate_parameters(LinearAB(xs), data)
        res = likelihood_ratio_test(est_int, est_lin, nested_name="intercept", full_name="linear")
        assert res.method == "lrt"
        assert res.p_value < 1e-10
        assert res.best_model == "linear"
        assert res.nested_pair == ("intercept", "linear")

    def test_does_not_reject_when_linear_truth(self, linear_data):
        """Linear data: linear (2) nested in quadratic (3). Quadratic
        adds a useless parameter, so LRT should NOT reject H₀
        (p-value should be reasonably large)."""
        xs, data = linear_data
        est_lin = estimate_parameters(LinearAB(xs), data)
        est_quad = estimate_parameters(QuadraticABC(xs), data)
        res = likelihood_ratio_test(est_lin, est_quad, nested_name="linear", full_name="quadratic")
        assert res.p_value > 0.1
        assert res.best_model == "linear"

    def test_raises_when_not_nested(self, linear_data):
        """LRT requires nested.parameter_names ⊂ full.parameter_names."""
        xs, data = linear_data
        est_int = estimate_parameters(InterceptOnly(len(xs)), data)
        # intercept's only param is 'b'; quadratic is 'a, b, c' so 'b' IS a subset.
        # Use mismatched parameter sets instead: linear has {a,b} but the "nested"
        # candidate's parameters are not a subset of linear.
        est_lin = estimate_parameters(LinearAB(xs), data)

        # Construct a fake "non-nested" pair by swapping who's full vs nested.
        with pytest.raises(ValueError, match="more parameters"):
            likelihood_ratio_test(est_lin, est_int)  # full has fewer params -> raises

    @pytest.mark.slow
    def test_uniform_p_value_under_h0(self):
        """If we sample many datasets from the nested model, the LRT
        p-values should be approximately uniform on [0, 1]; check by a
        Kolmogorov-style summary stat (mean ≈ 0.5 ± slack).
        """
        rng = np.random.default_rng(42)
        xs = np.linspace(0, 5, 20)
        # Generate from intercept-only truth: y = 1.5 + noise
        p_values = []
        for trial in range(40):  # keep modest for runtime
            y = 1.5 + rng.normal(0, 0.5, 20)
            data = {f"y_{i}": y[i] for i in range(len(xs))}
            est_int = estimate_parameters(InterceptOnly(len(xs)), data)
            est_lin = estimate_parameters(LinearAB(xs), data)
            res = likelihood_ratio_test(est_int, est_lin)
            p_values.append(res.p_value)
        mean_p = float(np.mean(p_values))
        # Should be near 0.5 under H0 (chi^2_{1,1-α}/2 nominal coverage).
        assert 0.3 < mean_p < 0.7, f"mean p={mean_p} far from 0.5"


# ─────────────────────────────────────────────────────────────────────
# Vuong test
# ─────────────────────────────────────────────────────────────────────


class TestVuong:
    def test_indistinguishable_when_truths_match(self, linear_data):
        """Linear and quadratic-with-zero-a fit identically -> |z| < z_crit."""
        xs, data = linear_data
        est_lin = estimate_parameters(LinearAB(xs), data)
        est_quad = estimate_parameters(QuadraticABC(xs), data)
        res = vuong_test(
            est_lin,
            est_quad,
            data,
            experiments={"lin": LinearAB(xs), "quad": QuadraticABC(xs)},
            name_a="lin",
            name_b="quad",
        )
        assert res.method == "vuong"
        assert res.best_model == "indistinguishable"
        assert abs(res.z_statistic) < 1.96

    def test_picks_better_model_when_different(self):
        """Generate strongly nonlinear data; linear and quadratic should
        give a meaningful Vuong z."""
        rng = np.random.default_rng(7)
        xs = np.linspace(0, 5, 20)
        y = 0.3 * xs * xs + 0.5 + rng.normal(0, 0.3, 20)
        data = {f"y_{i}": y[i] for i in range(len(xs))}
        est_lin = estimate_parameters(LinearAB(xs), data)
        est_quad = estimate_parameters(QuadraticABC(xs), data)
        res = vuong_test(
            est_lin,
            est_quad,
            data,
            experiments={"lin": LinearAB(xs), "quad": QuadraticABC(xs)},
            name_a="lin",
            name_b="quad",
        )
        # Quadratic should win: z < 0 (quad is "b" and m_n = ll_a - ll_b is
        # negative when quad fits better).
        assert res.z_statistic < 0
        assert res.best_model == "quad"

    def test_returns_z_and_p(self, linear_data):
        xs, data = linear_data
        est_lin = estimate_parameters(LinearAB(xs), data)
        est_int = estimate_parameters(InterceptOnly(len(xs)), data)
        res = vuong_test(
            est_lin,
            est_int,
            data,
            experiments={"lin": LinearAB(xs), "int": InterceptOnly(len(xs))},
            name_a="lin",
            name_b="int",
        )
        assert res.z_statistic is not None
        assert res.p_value is not None
        assert isinstance(res, ModelSelectionResult)
