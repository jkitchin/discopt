"""Edge-case and cross-consistency tests for the identifiability /
estimability / profile-likelihood stack.

Focused on scenarios that the three main test modules only touch
lightly:

- ``fixed_parameters`` kwarg on ``estimate_parameters`` works standalone.
- ``initial_estimate`` shortcut in ``profile_likelihood``.
- Confidence-level sensitivity: 68% CI is tighter than 95%.
- Single-parameter models (degenerate shapes).
- ``d_optimal_subset`` with extreme k (1 and p).
- Log-reparameterization: profile CI on log-parameter transforms back
  consistently (profile likelihood is reparameterization-invariant).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    d_optimal_subset,
    diagnose_identifiability,
    estimability_rank,
    profile_likelihood,
)
from discopt.estimate import Experiment, ExperimentModel, estimate_parameters


class LinearExp(Experiment):
    def __init__(self, xs):
        self.xs = xs

    def create_model(self, **kwargs):
        m = dm.Model("linear")
        a = m.continuous("a", lb=-50, ub=50)
        b = m.continuous("b", lb=-50, ub=50)
        responses = {f"y_{i}": a + b * x for i, x in enumerate(self.xs)}
        errors = {k: 1.0 for k in responses}
        return ExperimentModel(m, {"a": a, "b": b}, {}, responses, errors)


class SingleParamExp(Experiment):
    def __init__(self, xs):
        self.xs = xs

    def create_model(self, **kwargs):
        m = dm.Model("single")
        k = m.continuous("k", lb=0.01, ub=20.0)
        responses = {f"y_{i}": k * x for i, x in enumerate(self.xs)}
        errors = {kname: 0.1 for kname in responses}
        return ExperimentModel(m, {"k": k}, {}, responses, errors)


class LogKineticsExp(Experiment):
    """y = exp(log_k) * t; parameterized in log_k so profile likelihood
    of log_k is symmetric around log(k_true) and the CI in k is a
    geometric interval.
    """

    def __init__(self, ts):
        self.ts = ts

    def create_model(self, **kwargs):
        m = dm.Model("log_kinetics")
        log_k = m.continuous("log_k", lb=-5.0, ub=5.0)
        responses = {f"y_{i}": dm.exp(log_k) * t for i, t in enumerate(self.ts)}
        errors = {k: 0.05 for k in responses}
        return ExperimentModel(m, {"log_k": log_k}, {}, responses, errors)


@pytest.fixture
def linear_fit():
    np.random.seed(123)
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    exp = LinearExp(xs)
    a_true, b_true = 1.5, 0.8
    data = {f"y_{i}": a_true + b_true * x + np.random.normal(0, 1) for i, x in enumerate(xs)}
    return exp, data, a_true, b_true


# ─────────────────────────────────────────────────────────────────────
# fixed_parameters kwarg standalone behavior
# ─────────────────────────────────────────────────────────────────────


class TestFixedParametersKwarg:
    def test_fixing_pins_the_value(self, linear_fit):
        exp, data, _, _ = linear_fit
        result = estimate_parameters(exp, data, fixed_parameters={"b": 0.5})
        # Value is pinned exactly.
        assert result.parameters["b"] == pytest.approx(0.5, abs=1e-10)
        # 'a' remains free and near the least-squares value given b=0.5.
        assert result.parameters["a"] != 0.5

    def test_fixing_unknown_parameter_raises_keyerror(self, linear_fit):
        exp, data, _, _ = linear_fit
        with pytest.raises(KeyError, match="does_not_exist"):
            estimate_parameters(exp, data, fixed_parameters={"does_not_exist": 1.0})

    def test_fixing_overrides_initial_guess(self, linear_fit):
        """If the same parameter is in both dicts, fixed_parameters wins
        (and the fitted value equals the fixed value, not the guess).
        """
        exp, data, _, _ = linear_fit
        result = estimate_parameters(
            exp,
            data,
            initial_guess={"b": 10.0},  # would be a bad starting point
            fixed_parameters={"b": 2.0},
        )
        assert result.parameters["b"] == pytest.approx(2.0, abs=1e-10)

    def test_fixing_all_parameters(self, linear_fit):
        """Fixing every parameter gives a degenerate 'fit' at the
        supplied values; the objective just evaluates the residual SSR.
        """
        exp, data, _, _ = linear_fit
        result = estimate_parameters(exp, data, fixed_parameters={"a": 1.5, "b": 0.8})
        assert result.parameters["a"] == pytest.approx(1.5, abs=1e-10)
        assert result.parameters["b"] == pytest.approx(0.8, abs=1e-10)
        # Objective is non-negative.
        assert result.objective >= 0.0


# ─────────────────────────────────────────────────────────────────────
# profile_likelihood options
# ─────────────────────────────────────────────────────────────────────


class TestProfileLikelihoodOptions:
    def test_initial_estimate_skips_refit(self, linear_fit):
        """Passing a pre-computed EstimationResult should not cause
        a second fit; the returned theta_hat should equal the supplied
        one byte-for-byte.
        """
        exp, data, _, _ = linear_fit
        est = estimate_parameters(exp, data)
        prof = profile_likelihood(exp, data, "b", initial_estimate=est)
        assert prof.theta_hat == est.parameters["b"]
        assert prof.objective_hat == est.objective

    def test_confidence_level_monotonicity(self, linear_fit):
        """Higher confidence level -> wider interval."""
        exp, data, _, _ = linear_fit
        est = estimate_parameters(exp, data)
        prof_68 = profile_likelihood(exp, data, "b", confidence_level=0.68, initial_estimate=est)
        prof_95 = profile_likelihood(exp, data, "b", confidence_level=0.95, initial_estimate=est)
        prof_99 = profile_likelihood(exp, data, "b", confidence_level=0.99, initial_estimate=est)
        assert prof_68.ci_upper < prof_95.ci_upper < prof_99.ci_upper
        assert prof_68.ci_lower > prof_95.ci_lower > prof_99.ci_lower

    def test_initial_step_override(self, linear_fit):
        """User can override the curvature-based initial step."""
        exp, data, _, _ = linear_fit
        prof = profile_likelihood(exp, data, "b", initial_step=0.1, max_steps=10)
        assert prof.theta_values.size >= 1  # some points visited


class TestDiagnoseIdentifiabilityApi:
    """The new ``estimation_result=`` shortcut on
    :func:`diagnose_identifiability`.
    """

    def test_estimation_result_kwarg(self, linear_fit):
        exp, data, _, _ = linear_fit
        est = estimate_parameters(exp, data)

        diag_a = diagnose_identifiability(exp, est.parameters)
        diag_b = diagnose_identifiability(exp, estimation_result=est)

        assert diag_a.fim_rank == diag_b.fim_rank
        assert diag_a.vif == diag_b.vif
        np.testing.assert_allclose(diag_a.singular_values, diag_b.singular_values)

    def test_missing_both_raises(self, linear_fit):
        exp, _, _, _ = linear_fit
        with pytest.raises(TypeError, match="either param_values or estimation_result"):
            diagnose_identifiability(exp)


class TestSingleParameterModel:
    def test_diagnose_identifiability_trivial(self):
        exp = SingleParamExp([1.0, 2.0, 3.0])
        diag = diagnose_identifiability(exp, {"k": 1.0})
        assert diag.is_identifiable is True
        assert diag.fim_rank == 1
        assert diag.n_parameters == 1
        assert diag.vif["k"] == pytest.approx(1.0, abs=1e-12)  # no other regressor
        assert diag.variance_decomposition.shape == (1, 1)
        # Only one direction -> all variance attributed to it.
        assert diag.variance_decomposition[0, 0] == pytest.approx(1.0, abs=1e-12)

    def test_profile_likelihood_single_param(self):
        """Single-parameter model should still produce a bounded CI
        when the data are informative."""
        np.random.seed(7)
        xs = [1.0, 2.0, 3.0, 4.0]
        exp = SingleParamExp(xs)
        data = {f"y_{i}": 2.5 * x + np.random.normal(0, 0.1) for i, x in enumerate(xs)}
        prof = profile_likelihood(exp, data, "k")
        assert prof.shape == "bounded"
        assert prof.ci_lower is not None and prof.ci_upper is not None
        assert prof.ci_lower < 2.5 < prof.ci_upper


# ─────────────────────────────────────────────────────────────────────
# d_optimal_subset extremes
# ─────────────────────────────────────────────────────────────────────


class TestDOptimalSubsetExtremes:
    def test_k_equals_one(self, linear_fit):
        exp, _, _, _ = linear_fit
        S = d_optimal_subset(exp, {"a": 1.5, "b": 0.8}, k=1, method="enumerate")
        assert len(S) == 1
        assert S[0] in {"a", "b"}

    def test_k_equals_p(self, linear_fit):
        exp, _, _, _ = linear_fit
        S = d_optimal_subset(exp, {"a": 1.5, "b": 0.8}, k=2, method="enumerate")
        assert set(S) == {"a", "b"}

    def test_k_zero_raises(self, linear_fit):
        exp, _, _, _ = linear_fit
        with pytest.raises(ValueError):
            d_optimal_subset(exp, {"a": 1.5, "b": 0.8}, k=0)

    def test_auto_agrees_with_enumerate_on_small_p(self, linear_fit):
        exp, _, _, _ = linear_fit
        pvals = {"a": 1.5, "b": 0.8}
        assert set(d_optimal_subset(exp, pvals, k=1, method="auto")) == set(
            d_optimal_subset(exp, pvals, k=1, method="enumerate")
        )


# ─────────────────────────────────────────────────────────────────────
# Reparameterization invariance smoke test
# ─────────────────────────────────────────────────────────────────────


class TestReparameterizationInvariance:
    def test_profile_ci_on_log_parameter_matches_geometric_in_original(self):
        """For ``y = exp(log_k) * t`` with data generated at k_true,
        the profile CI on log_k should transform to exp(ci) being a
        geometric CI around k_true. This exercises the claim that
        profile likelihood is reparameterization-invariant.
        """
        np.random.seed(11)
        ts = np.linspace(0.5, 5.0, 10)
        k_true = 2.0
        exp_log = LogKineticsExp(ts)
        data = {f"y_{i}": k_true * t + np.random.normal(0, 0.05) for i, t in enumerate(ts)}

        prof = profile_likelihood(exp_log, data, "log_k")
        assert prof.shape == "bounded"
        # The geometric CI around k_true.
        k_ci_lo = float(np.exp(prof.ci_lower))
        k_ci_hi = float(np.exp(prof.ci_upper))
        assert 0 < k_ci_lo < k_true < k_ci_hi
        # Geometric CI should be "symmetric on the log scale" — i.e.
        # (log(k_hi) - log(k_hat)) ~ (log(k_hat) - log(k_lo)).
        k_hat = float(np.exp(prof.theta_hat))
        up = np.log(k_ci_hi) - np.log(k_hat)
        dn = np.log(k_hat) - np.log(k_ci_lo)
        np.testing.assert_allclose(up, dn, rtol=1e-2)


# ─────────────────────────────────────────────────────────────────────
# Coverage-rate sanity check (probabilistic; few seeds)
# ─────────────────────────────────────────────────────────────────────


class TestCoverageRate:
    def test_profile_ci_covers_truth_on_linear_regression(self):
        """Across a handful of synthetic replicates, the 95% profile CI
        should cover the true slope in most of them. We use 10 seeds
        and require >= 7 cover -- not a rigorous coverage test but a
        quick sanity signal that catches gross errors.
        """
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        b_true = 2.0
        covered = 0
        for seed in range(10):
            rng = np.random.default_rng(seed)
            exp = LinearExp(xs)
            data = {f"y_{i}": 1.0 + b_true * x + rng.normal(0, 1) for i, x in enumerate(xs)}
            prof = profile_likelihood(exp, data, "b")
            if (
                prof.ci_lower is not None
                and prof.ci_upper is not None
                and prof.ci_lower <= b_true <= prof.ci_upper
            ):
                covered += 1
        assert covered >= 7, f"95% profile CI covered truth in only {covered}/10 seeds"


# ─────────────────────────────────────────────────────────────────────
# estimability_rank edge cases
# ─────────────────────────────────────────────────────────────────────


class TestEstimabilityRankEdgeCases:
    def test_single_parameter_ranking(self):
        exp = SingleParamExp([1.0, 2.0, 3.0])
        res = estimability_rank(exp, {"k": 1.0})
        assert res.ranking == ["k"]
        assert res.recommended_subset == ["k"]
        # Collinearity of a single-column Z_K (after unit-normalizing) is 1.
        assert res.collinearity_index == pytest.approx(1.0, abs=1e-10)

    def test_custom_cutoff_shrinks_subset(self):
        """A strict cutoff (0.99) should drop everything except the
        top parameter even when the problem is well-conditioned.
        """
        exp = LinearExp([1.0, 2.0, 3.0, 4.0, 5.0])
        res = estimability_rank(exp, {"a": 1.0, "b": 1.0}, cutoff=0.99)
        assert len(res.recommended_subset) == 1
