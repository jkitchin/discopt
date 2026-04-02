"""Tests for discopt.ro uncertainty set classes.

Covers:
- BoxUncertaintySet: construction, shape validation, bounds
- EllipsoidalUncertaintySet: construction, Sigma_sqrt, radius
- PolyhedralUncertaintySet: construction, shape checks
- budget_uncertainty_set: encoding of Bertsimas-Sim set
- RobustCounterpart: API validation (formulate once, mixed types rejected)
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

import discopt.modeling as dm
from discopt.ro import (
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    PolyhedralUncertaintySet,
    RobustCounterpart,
    budget_uncertainty_set,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scalar_param():
    m = dm.Model()
    return m.parameter("p", value=5.0)


@pytest.fixture()
def vector_param():
    m = dm.Model()
    return m.parameter("c", value=[10.0, 15.0, 8.0])


@pytest.fixture()
def matrix_param():
    m = dm.Model()
    return m.parameter("A", value=np.eye(2))


# ---------------------------------------------------------------------------
# BoxUncertaintySet
# ---------------------------------------------------------------------------


class TestBoxUncertaintySet:
    def test_scalar_broadcast(self, vector_param):
        unc = BoxUncertaintySet(vector_param, delta=1.0)
        assert unc.delta.shape == (3,)
        np.testing.assert_array_equal(unc.delta, [1.0, 1.0, 1.0])

    def test_vector_delta(self, vector_param):
        delta = np.array([0.5, 1.0, 0.3])
        unc = BoxUncertaintySet(vector_param, delta=delta)
        np.testing.assert_array_equal(unc.delta, delta)

    def test_lower_upper_bounds(self, vector_param):
        delta = np.array([1.0, 2.0, 0.5])
        unc = BoxUncertaintySet(vector_param, delta=delta)
        np.testing.assert_allclose(unc.lower, [9.0, 13.0, 7.5])
        np.testing.assert_allclose(unc.upper, [11.0, 17.0, 8.5])

    def test_zero_delta_allowed(self, scalar_param):
        unc = BoxUncertaintySet(scalar_param, delta=0.0)
        np.testing.assert_allclose(unc.lower, [5.0])
        np.testing.assert_allclose(unc.upper, [5.0])

    def test_negative_delta_rejected(self, scalar_param):
        with pytest.raises(ValueError, match="non-negative"):
            BoxUncertaintySet(scalar_param, delta=-1.0)

    def test_wrong_shape_delta_rejected(self, vector_param):
        with pytest.raises(ValueError, match="shape"):
            BoxUncertaintySet(vector_param, delta=np.array([1.0, 2.0]))

    def test_non_parameter_rejected(self):
        with pytest.raises(TypeError, match="Parameter"):
            BoxUncertaintySet("not_a_param", delta=1.0)

    def test_kind(self, scalar_param):
        unc = BoxUncertaintySet(scalar_param, delta=0.5)
        assert unc.kind == "box"

    def test_scalar_parameter_works(self, scalar_param):
        unc = BoxUncertaintySet(scalar_param, delta=0.5)
        np.testing.assert_allclose(unc.lower, [4.5])
        np.testing.assert_allclose(unc.upper, [5.5])


# ---------------------------------------------------------------------------
# EllipsoidalUncertaintySet
# ---------------------------------------------------------------------------


class TestEllipsoidalUncertaintySet:
    def test_isotropic_sigma(self, vector_param):
        unc = EllipsoidalUncertaintySet(vector_param, rho=2.0)
        np.testing.assert_allclose(unc.Sigma, np.eye(3))
        np.testing.assert_allclose(unc.Sigma_sqrt, np.eye(3), atol=1e-12)

    def test_custom_sigma(self, vector_param):
        Sigma = np.diag([4.0, 1.0, 9.0])
        unc = EllipsoidalUncertaintySet(vector_param, rho=1.0, Sigma=Sigma)
        # Sigma_sqrt should satisfy Sigma_sqrt @ Sigma_sqrt.T ≈ Sigma
        S = unc.Sigma_sqrt
        np.testing.assert_allclose(S @ S.T, Sigma, atol=1e-10)

    def test_rho_stored(self, vector_param):
        unc = EllipsoidalUncertaintySet(vector_param, rho=3.5)
        assert unc.rho == pytest.approx(3.5)

    def test_zero_rho_rejected(self, vector_param):
        with pytest.raises(ValueError, match="positive"):
            EllipsoidalUncertaintySet(vector_param, rho=0.0)

    def test_wrong_sigma_shape_rejected(self, vector_param):
        with pytest.raises(ValueError, match="Sigma"):
            EllipsoidalUncertaintySet(vector_param, rho=1.0, Sigma=np.eye(2))

    def test_kind(self, vector_param):
        unc = EllipsoidalUncertaintySet(vector_param, rho=1.0)
        assert unc.kind == "ellipsoidal"

    def test_psd_sigma_fallback(self, vector_param):
        # Near-singular PSD matrix (rank-deficient)
        Sigma = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        unc = EllipsoidalUncertaintySet(vector_param, rho=1.0, Sigma=Sigma)
        # Should not raise; Sigma_sqrt must be real
        assert np.all(np.isfinite(unc.Sigma_sqrt))


# ---------------------------------------------------------------------------
# PolyhedralUncertaintySet
# ---------------------------------------------------------------------------


class TestPolyhedralUncertaintySet:
    def test_basic_construction(self, vector_param):
        k = 3
        A = np.vstack([np.eye(k), -np.eye(k)])
        b = np.ones(2 * k)
        unc = PolyhedralUncertaintySet(vector_param, A=A, b=b)
        assert unc.kind == "polyhedral"
        np.testing.assert_array_equal(unc.A, A)
        np.testing.assert_array_equal(unc.b, b)

    def test_wrong_A_cols_rejected(self, vector_param):
        with pytest.raises(ValueError, match="A must have shape"):
            PolyhedralUncertaintySet(vector_param, A=np.eye(2), b=np.ones(2))

    def test_wrong_b_length_rejected(self, vector_param):
        k = 3
        A = np.eye(k)
        with pytest.raises(ValueError, match="b must have shape"):
            PolyhedralUncertaintySet(vector_param, A=A, b=np.ones(2))

    def test_kind(self, vector_param):
        k = 3
        unc = PolyhedralUncertaintySet(vector_param, A=np.eye(k), b=np.ones(k))
        assert unc.kind == "polyhedral"


# ---------------------------------------------------------------------------
# budget_uncertainty_set
# ---------------------------------------------------------------------------


class TestBudgetUncertaintySet:
    def test_produces_polyhedral(self, vector_param):
        unc = budget_uncertainty_set(vector_param, delta=1.0, gamma=2.0)
        assert isinstance(unc, PolyhedralUncertaintySet)

    def test_budget_attributes(self, vector_param):
        unc = budget_uncertainty_set(vector_param, delta=1.0, gamma=2.0)
        assert unc._is_budget is True
        assert unc._gamma == pytest.approx(2.0)
        np.testing.assert_allclose(unc._delta, [1.0, 1.0, 1.0])

    def test_gamma_zero_gives_nominal(self, vector_param):
        # Γ=0: only nominal solution is feasible, but should construct fine.
        unc = budget_uncertainty_set(vector_param, delta=1.0, gamma=0.0)
        assert unc._gamma == 0.0

    def test_gamma_out_of_range_rejected(self, vector_param):
        with pytest.raises(ValueError, match="gamma"):
            budget_uncertainty_set(vector_param, delta=1.0, gamma=10.0)

    def test_non_positive_delta_rejected(self, vector_param):
        with pytest.raises(ValueError, match="positive"):
            budget_uncertainty_set(vector_param, delta=0.0, gamma=1.0)

    def test_A_b_shapes(self, vector_param):
        k = 3
        unc = budget_uncertainty_set(vector_param, delta=1.0, gamma=1.0)
        # A should have shape (2k + 2, k): k box-upper, k box-lower, 2 budget
        assert unc.A.shape == (2 * k + 2, k)
        assert unc.b.shape == (2 * k + 2,)


# ---------------------------------------------------------------------------
# RobustCounterpart API
# ---------------------------------------------------------------------------


class TestRobustCounterpartAPI:
    def _simple_model(self):
        m = dm.Model()
        x = m.continuous("x", lb=0)
        c = m.parameter("c", value=5.0)
        m.minimize(c * x)
        m.subject_to(x >= 1.0)
        return m, x, c

    def test_single_set_accepted(self):
        m, x, c = self._simple_model()
        unc = BoxUncertaintySet(c, delta=0.5)
        rc = RobustCounterpart(m, unc)  # single set, not a list
        assert rc.kind == "box"

    def test_list_of_sets_accepted(self):
        m = dm.Model()
        x = m.continuous("x", lb=0)
        c1 = m.parameter("c1", value=5.0)
        c2 = m.parameter("c2", value=3.0)
        m.minimize(c1 * x + c2 * x)
        m.subject_to(x >= 1.0)
        unc1 = BoxUncertaintySet(c1, delta=0.5)
        unc2 = BoxUncertaintySet(c2, delta=0.3)
        rc = RobustCounterpart(m, [unc1, unc2])
        assert rc.kind == "box"

    def test_empty_list_rejected(self):
        m, x, c = self._simple_model()
        with pytest.raises(ValueError, match="empty"):
            RobustCounterpart(m, [])

    def test_mixed_types_rejected(self):
        m = dm.Model()
        x = m.continuous("x", lb=0)
        c1 = m.parameter("c1", value=[1.0, 2.0])
        c2 = m.parameter("c2", value=[3.0, 4.0])
        m.minimize(dm.sum(c1 * x))
        unc_box = BoxUncertaintySet(c1, delta=0.1)
        unc_ell = EllipsoidalUncertaintySet(c2, rho=1.0)
        with pytest.raises(ValueError, match="same type"):
            RobustCounterpart(m, [unc_box, unc_ell])

    def test_formulate_once(self):
        m, x, c = self._simple_model()
        unc = BoxUncertaintySet(c, delta=0.5)
        rc = RobustCounterpart(m, unc)
        rc.formulate()
        with pytest.raises(RuntimeError, match="already been called"):
            rc.formulate()

    def test_kind_property(self):
        m, x, c = self._simple_model()
        unc = BoxUncertaintySet(c, delta=0.5)
        rc = RobustCounterpart(m, unc)
        assert rc.kind == "box"
