"""Analytic and closed-form tests for the five discrimination criteria.

Each criterion has at least one *quantitative* test that either
matches a closed-form value or reduces to a known limit.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    DesignCriterion,
    DiscriminationCriterion,
    DiscriminationDesignResult,
    discriminate_compound,
    discriminate_design,
)
from discopt.estimate import Experiment, ExperimentModel

# ─────────────────────────────────────────────────────────────────────
# Fixtures: linear vs quadratic on a single design variable
# ─────────────────────────────────────────────────────────────────────


class LinearExp(Experiment):
    """``y = a * x``."""

    def create_model(self, **kw):
        m = dm.Model("linear")
        a = m.continuous("a", lb=0.1, ub=5.0)
        x = m.continuous("x", lb=0.0, ub=2.0)
        return ExperimentModel(m, {"a": a}, {"x": x}, {"y": a * x}, {"y": 0.5})


class QuadraticExp(Experiment):
    """``y = a * x^2``."""

    def create_model(self, **kw):
        m = dm.Model("quadratic")
        a = m.continuous("a", lb=0.1, ub=5.0)
        x = m.continuous("x", lb=0.0, ub=2.0)
        return ExperimentModel(m, {"a": a}, {"x": x}, {"y": a * x * x}, {"y": 0.5})


EXPS = {"linear": LinearExp(), "quadratic": QuadraticExp()}
PE = {"linear": {"a": 1.0}, "quadratic": {"a": 1.0}}
BOUNDS = {"x": (0.0, 2.0)}


# ─────────────────────────────────────────────────────────────────────
# Hunter-Reiner
# ─────────────────────────────────────────────────────────────────────


class TestHunterReiner:
    def test_returns_discrimination_result(self):
        r = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.HR, n_starts=5, seed=0
        )
        assert isinstance(r, DiscriminationDesignResult)
        assert set(r.model_names) == {"linear", "quadratic"}
        assert r.criterion is DiscriminationCriterion.HR

    def test_picks_upper_bound(self):
        """``|a·x − a·x²|² = a²(x − x²)²`` on [0, 2] is maximised at x=2
        (value 4) for a=1; x=0 gives 0; x=1 gives 0. Argmax = x=2.
        """
        r = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.HR, n_starts=10, seed=0
        )
        assert r.design["x"] == pytest.approx(2.0, abs=1e-4)

    def test_closed_form_value_at_upper_bound(self):
        """At x=2: (2 − 4)² = 4; with w_i=w_j=0.5, HR = 0.5·0.5·4 = 1.0."""
        r = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.HR, n_starts=3, seed=0
        )
        assert r.criterion_value == pytest.approx(1.0, abs=1e-3)

    def test_pairwise_matrix_is_symmetric(self):
        r = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.HR, n_starts=3, seed=0
        )
        assert r.pairwise_divergence is not None
        np.testing.assert_allclose(r.pairwise_divergence, r.pairwise_divergence.T, atol=1e-12)
        assert np.all(np.diag(r.pairwise_divergence) == 0.0)


# ─────────────────────────────────────────────────────────────────────
# Buzzi-Ferraris-Forzatti
# ─────────────────────────────────────────────────────────────────────


class TestBuzziFerraris:
    def test_picks_upper_bound(self):
        r = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.BF, n_starts=10, seed=0
        )
        assert r.design["x"] == pytest.approx(2.0, abs=1e-4)

    def test_bf_argmax_matches_hr_argmax_on_symmetric_problem(self):
        """On a symmetric problem where prediction covariances are
        equal across models, the BF criterion and HR criterion share
        the same argmax.
        """
        r_hr = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.HR, n_starts=5, seed=0
        )
        r_bf = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.BF, n_starts=5, seed=0
        )
        assert r_bf.design["x"] == pytest.approx(r_hr.design["x"], abs=1e-3)


# ─────────────────────────────────────────────────────────────────────
# Jensen-Rényi
# ─────────────────────────────────────────────────────────────────────


class TestJensenRenyi:
    def test_picks_upper_bound(self):
        r = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.JR, n_starts=10, seed=0
        )
        assert r.design["x"] == pytest.approx(2.0, abs=1e-4)

    def test_nonnegative(self):
        """Jensen-Rényi divergence is non-negative."""
        r = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.JR, n_starts=5, seed=0
        )
        assert r.criterion_value >= -1e-10  # numerical slack

    def test_vanishes_for_identical_models(self):
        """If both models make the same predictions with the same
        covariance, JR should be ~0."""
        # Two copies of the same model.
        exps = {"a": LinearExp(), "b": LinearExp()}
        pe = {"a": {"a": 1.0}, "b": {"a": 1.0}}
        r = discriminate_design(
            exps, pe, BOUNDS, criterion=DiscriminationCriterion.JR, n_starts=5, seed=0
        )
        assert abs(r.criterion_value) < 1e-8


# ─────────────────────────────────────────────────────────────────────
# Mutual information
# ─────────────────────────────────────────────────────────────────────


class TestMutualInformation:
    def test_picks_upper_bound_at_moderate_samples(self):
        """MI argmax should agree with HR/BF on this problem even at
        modest sample counts (noise averages out at the optimum)."""
        r = discriminate_design(
            EXPS,
            PE,
            BOUNDS,
            criterion=DiscriminationCriterion.MI,
            n_starts=5,
            mi_samples=1000,
            seed=0,
        )
        assert r.design["x"] == pytest.approx(2.0, abs=0.05)

    def test_returns_no_pairwise_matrix(self):
        r = discriminate_design(
            EXPS,
            PE,
            BOUNDS,
            criterion=DiscriminationCriterion.MI,
            n_starts=3,
            mi_samples=500,
            seed=0,
        )
        assert r.pairwise_divergence is None

    def test_nonnegative(self):
        r = discriminate_design(
            EXPS,
            PE,
            BOUNDS,
            criterion=DiscriminationCriterion.MI,
            n_starts=3,
            mi_samples=500,
            seed=0,
        )
        # MC has ±noise; allow small negative.
        assert r.criterion_value > -0.1


# ─────────────────────────────────────────────────────────────────────
# DT-compound
# ─────────────────────────────────────────────────────────────────────


class TestDTCompound:
    def test_lambda_one_matches_bf(self):
        """``λ = 1`` should produce the same design and value as pure BF."""
        r_bf = discriminate_design(
            EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.BF, n_starts=5, seed=0
        )
        r_dt = discriminate_compound(
            EXPS,
            PE,
            BOUNDS,
            discrimination_weight=1.0,
            discrimination_criterion=DiscriminationCriterion.BF,
            precision_model="linear",
            n_starts=5,
            seed=0,
        )
        assert r_dt.criterion is DiscriminationCriterion.DT
        np.testing.assert_allclose(r_dt.criterion_value, r_bf.criterion_value, atol=1e-4)
        assert r_dt.design["x"] == pytest.approx(r_bf.design["x"], abs=1e-3)

    def test_lambda_zero_maximises_single_model_precision(self):
        """``λ = 0`` should put the experiment where ``log_det(FIM_linear)``
        is largest, ignoring discrimination."""
        r_dt = discriminate_compound(
            EXPS,
            PE,
            BOUNDS,
            discrimination_weight=0.0,
            precision_criterion=DesignCriterion.D_OPTIMAL,
            precision_model="linear",
            n_starts=5,
            seed=0,
        )
        # For y = a*x (a=1), FIM_a = x^2/sigma^2. Maximising log(det) == log(x^2/sigma^2),
        # so argmax over [0, 2] is x=2.
        assert r_dt.design["x"] == pytest.approx(2.0, abs=1e-4)

    def test_bad_weight_raises(self):
        with pytest.raises(ValueError, match="discrimination_weight"):
            discriminate_compound(
                EXPS, PE, BOUNDS, discrimination_weight=-0.1, precision_model="linear"
            )

    def test_unknown_precision_model_raises(self):
        with pytest.raises(KeyError, match="precision_model"):
            discriminate_compound(
                EXPS, PE, BOUNDS, discrimination_weight=0.5, precision_model="does_not_exist"
            )

    def test_default_precision_model_warns(self):
        r = discriminate_compound(EXPS, PE, BOUNDS, discrimination_weight=0.5, n_starts=3, seed=0)
        assert any("precision_model" in w for w in r.warnings)


# ─────────────────────────────────────────────────────────────────────
# API validation
# ─────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_single_model_raises(self):
        with pytest.raises(ValueError, match="at least 2 candidate models"):
            discriminate_design({"only": LinearExp()}, {"only": {"a": 1.0}}, BOUNDS)

    def test_mismatched_param_estimates_keys(self):
        with pytest.raises(ValueError, match="share keys"):
            discriminate_design(
                EXPS,
                {"linear": {"a": 1.0}, "other": {"a": 1.0}},
                BOUNDS,
            )

    def test_empty_design_bounds(self):
        with pytest.raises(ValueError, match="non-empty"):
            discriminate_design(EXPS, PE, {})

    def test_dt_not_callable_as_plain_criterion(self):
        with pytest.raises(ValueError, match="discriminate_compound"):
            discriminate_design(EXPS, PE, BOUNDS, criterion=DiscriminationCriterion.DT)

    def test_model_priors_wrong_keys(self):
        with pytest.raises(ValueError, match="model_priors"):
            discriminate_design(EXPS, PE, BOUNDS, model_priors={"linear": 0.5, "other": 0.5})

    def test_model_priors_zero_sum(self):
        with pytest.raises(ValueError, match="positive"):
            discriminate_design(EXPS, PE, BOUNDS, model_priors={"linear": 0.0, "quadratic": 0.0})


# ─────────────────────────────────────────────────────────────────────
# Three-model stress test
# ─────────────────────────────────────────────────────────────────────


class CubicExp(Experiment):
    def create_model(self, **kw):
        m = dm.Model("cubic")
        a = m.continuous("a", lb=0.1, ub=5.0)
        x = m.continuous("x", lb=0.0, ub=2.0)
        return ExperimentModel(m, {"a": a}, {"x": x}, {"y": a * x * x * x}, {"y": 0.5})


class TestThreeModels:
    """Three rival models at once: BF and JR should both support this
    natively, and pairwise_divergence should be (3, 3)."""

    def _exps_pe(self):
        exps = {"linear": LinearExp(), "quadratic": QuadraticExp(), "cubic": CubicExp()}
        pe = {k: {"a": 1.0} for k in exps}
        return exps, pe

    def test_bf_three_models(self):
        exps, pe = self._exps_pe()
        r = discriminate_design(
            exps, pe, BOUNDS, criterion=DiscriminationCriterion.BF, n_starts=10, seed=0
        )
        assert r.design["x"] == pytest.approx(2.0, abs=1e-3)
        assert r.pairwise_divergence.shape == (3, 3)

    def test_jr_three_models(self):
        exps, pe = self._exps_pe()
        r = discriminate_design(
            exps, pe, BOUNDS, criterion=DiscriminationCriterion.JR, n_starts=10, seed=0
        )
        assert r.design["x"] == pytest.approx(2.0, abs=1e-3)
