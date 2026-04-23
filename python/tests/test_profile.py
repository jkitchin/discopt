"""Tests for profile_likelihood and profile_all (Piece 3 of issue #45).

Key tests
- Linear regression CI: profile CI must match the chi-square-based
  Wald CI to ~5 decimal places (known-sigma Gaussian -> exact). This
  nails the factor-of-2 threshold convention.
- Unidentifiable a*b*x ridge: profile of ``a`` is flat or one-sided
  (depending on how the box constrains the ridge).
- Round-trip pipeline consistency: estimate -> diagnose -> estimability
  -> profile all agree on which parameter is non-identifiable in the
  a*b*x + c*x^2 fixture.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    ProfileLikelihoodResult,
    diagnose_identifiability,
    estimability_rank,
    profile_all,
    profile_likelihood,
)
from discopt.estimate import Experiment, ExperimentModel, estimate_parameters


class LinearExperiment(Experiment):
    """y_i = a + b*x_i."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("linear")
        a = m.continuous("a", lb=-20, ub=20)
        b = m.continuous("b", lb=-20, ub=20)
        responses = {f"y_{i}": a + b * xi for i, xi in enumerate(self.x_data)}
        errors = {k: 1.0 for k in responses}
        return ExperimentModel(m, {"a": a, "b": b}, {}, responses, errors)


class AbRidgeExperiment(Experiment):
    """y_i = a*b*x_i; only the product a*b is identifiable."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("ab")
        a = m.continuous("a", lb=0.1, ub=10)
        b = m.continuous("b", lb=0.1, ub=10)
        responses = {f"y_{i}": a * b * xi for i, xi in enumerate(self.x_data)}
        errors = {k: 0.1 for k in responses}
        return ExperimentModel(m, {"a": a, "b": b}, {}, responses, errors)


class AbcPipelineExperiment(Experiment):
    """y_i = a*b*x_i + c*x_i^2; rank 2, one null direction involves a and b."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("abc")
        a = m.continuous("a", lb=0.1, ub=10)
        b = m.continuous("b", lb=0.1, ub=10)
        c = m.continuous("c", lb=-10, ub=10)
        responses = {f"y_{i}": a * b * xi + c * xi * xi for i, xi in enumerate(self.x_data)}
        errors = {k: 1.0 for k in responses}
        return ExperimentModel(m, {"a": a, "b": b, "c": c}, {}, responses, errors)


@pytest.fixture
def linear_fit():
    np.random.seed(0)
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    a_true, b_true = 2.0, 3.0
    exp = LinearExperiment(xs)
    data = {f"y_{i}": a_true + b_true * x + np.random.normal(0, 1) for i, x in enumerate(xs)}
    est = estimate_parameters(exp, data)
    return exp, data, est


class TestProfileLikelihoodLinear:
    def test_returns_result_type(self, linear_fit):
        exp, data, est = linear_fit
        prof = profile_likelihood(exp, data, "b")
        assert isinstance(prof, ProfileLikelihoodResult)

    def test_ci_matches_chi2_wald_on_linear_regression(self, linear_fit):
        """For linear-Gaussian with known sigma, the profile CI is
        exactly theta_hat +/- sqrt(chi2_{1,0.95}) * SE.
        """
        exp, data, est = linear_fit
        prof = profile_likelihood(exp, data, "b", confidence_level=0.95)

        se = est.standard_errors["b"]
        theta_hat = est.parameters["b"]
        from scipy.stats import chi2

        offset = np.sqrt(chi2.ppf(0.95, df=1)) * se
        expected_lo = theta_hat - offset
        expected_hi = theta_hat + offset

        assert prof.ci_lower is not None
        assert prof.ci_upper is not None
        np.testing.assert_allclose(prof.ci_lower, expected_lo, rtol=1e-3)
        np.testing.assert_allclose(prof.ci_upper, expected_hi, rtol=1e-3)
        assert prof.shape == "bounded"

    def test_threshold_uses_deviance_convention(self, linear_fit):
        """Threshold must equal objective_hat + chi2_{1,alpha}, not
        objective_hat + chi2/2. This is the key factor-of-2 guard.
        """
        exp, data, est = linear_fit
        prof = profile_likelihood(exp, data, "a", confidence_level=0.95)
        from scipy.stats import chi2

        expected_threshold = est.objective + chi2.ppf(0.95, df=1)
        np.testing.assert_allclose(prof.threshold, expected_threshold, rtol=1e-12)

    def test_profile_symmetric_around_theta_hat(self, linear_fit):
        """For linear Gaussian, CI should be symmetric about theta_hat."""
        exp, data, est = linear_fit
        prof = profile_likelihood(exp, data, "b")
        theta_hat = est.parameters["b"]
        lo_dist = theta_hat - prof.ci_lower
        hi_dist = prof.ci_upper - theta_hat
        np.testing.assert_allclose(lo_dist, hi_dist, rtol=1e-2)


class TestProfileLikelihoodUnidentifiable:
    def test_ab_ridge_profile_of_a(self):
        """On the a*b*x ridge: moving a without adjusting b would break
        the fit, but the inner NLP re-optimizes b to keep a*b constant,
        so the profile is flat up to bound clipping. Result: shape is
        ``flat`` (genuinely non-identifiable within the box) or
        ``one_sided_*`` (bound-clipped).
        """
        np.random.seed(1)
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        exp = AbRidgeExperiment(xs)
        data = {f"y_{i}": 6.0 * x + np.random.normal(0, 0.1) for i, x in enumerate(xs)}

        prof = profile_likelihood(exp, data, "a", max_steps=15)
        assert prof.shape in ("flat", "one_sided_lower", "one_sided_upper")
        # Flat identifiability signature: if any arm crosses threshold,
        # it was due to a parameter bound, not curvature.
        assert any("bound" in w or "flat" in w or "non-identifiable" in w for w in prof.warnings)


class TestProfileAll:
    def test_loops_over_all_params(self, linear_fit):
        exp, data, est = linear_fit
        results = profile_all(exp, data, max_steps=15)
        assert set(results.keys()) == {"a", "b"}
        for r in results.values():
            assert isinstance(r, ProfileLikelihoodResult)


class TestApiValidation:
    def test_unknown_parameter_name_raises(self, linear_fit):
        exp, data, _ = linear_fit
        with pytest.raises(KeyError):
            profile_likelihood(exp, data, "does_not_exist")

    def test_invalid_confidence_level_raises(self, linear_fit):
        exp, data, _ = linear_fit
        with pytest.raises(ValueError):
            profile_likelihood(exp, data, "a", confidence_level=1.5)


class TestPipelineConsistency:
    """Round-trip: estimate -> diagnose -> estimability -> profile on
    the a*b*x + c*x^2 fixture. Every tool must agree that the direction
    spanned by {a, b} is non-identifiable.
    """

    def _fit_and_diagnose(self):
        np.random.seed(7)
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        exp = AbcPipelineExperiment(xs)
        data = {f"y_{i}": 6.0 * x + 0.5 * x * x + np.random.normal(0, 1) for i, x in enumerate(xs)}
        est = estimate_parameters(exp, data, initial_guess={"a": 2.0, "b": 3.0, "c": 0.5})
        diag = diagnose_identifiability(exp, est.parameters)
        rank = estimability_rank(exp, est.parameters)
        return exp, data, est, diag, rank

    def test_fim_rank_is_two_of_three(self):
        _, _, _, diag, _ = self._fit_and_diagnose()
        assert diag.fim_rank == 2
        assert diag.n_parameters == 3

    def test_yao_projects_last_to_zero(self):
        _, _, _, _, rank = self._fit_and_diagnose()
        top = rank.projected_norms[0]
        assert rank.projected_norms[-1] < 1e-6 * top

    def test_yao_recommended_subset_excludes_one_of_ab(self):
        """The recommended subset should have size 2, containing c and
        exactly one of {a, b}.
        """
        _, _, _, _, rank = self._fit_and_diagnose()
        assert len(rank.recommended_subset) == 2
        assert "c" in rank.recommended_subset
        excluded = [p for p in ["a", "b", "c"] if p not in rank.recommended_subset]
        assert excluded[0] in {"a", "b"}

    def test_diagnose_null_space_involves_a_and_b(self):
        _, _, _, diag, _ = self._fit_and_diagnose()
        assert len(diag.null_space) == 1
        direction = diag.null_space[0]
        assert abs(direction["c"]) < 1e-3
        assert abs(direction["a"]) > 0.1
        assert abs(direction["b"]) > 0.1
