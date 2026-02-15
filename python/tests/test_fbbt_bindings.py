"""Tests for FBBT PyO3 bindings (C3: FBBT with incumbent cutoff)."""

import discopt.modeling as dm
import numpy as np
from discopt._rust import model_to_repr


def _make_linear_model():
    """min x s.t. x + y <= 10, x in [0, 100], y in [0, 100]."""
    m = dm.Model("linear")
    x = m.continuous("x", lb=0.0, ub=100.0)
    y = m.continuous("y", lb=0.0, ub=100.0)
    m.minimize(x)
    m.subject_to(x + y <= 10.0)
    return m


def _make_exp_model():
    """min exp(x) s.t. x in [-10, 10]."""
    m = dm.Model("exp")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize(dm.exp(x))
    return m


class TestFBBTBindings:
    """Test the PyO3 FBBT bindings on PyModelRepr."""

    def test_basic_fbbt(self):
        """Plain FBBT (no cutoff) tightens bounds correctly."""
        model = _make_linear_model()
        repr_ = model_to_repr(model)
        lbs, ubs = repr_.fbbt(max_iter=10, tol=1e-8)
        lbs = np.asarray(lbs)
        ubs = np.asarray(ubs)
        # x + y <= 10, x >= 0, y >= 0 => ub tightened to 10
        np.testing.assert_allclose(lbs, [0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(ubs, [10.0, 10.0], atol=1e-8)

    def test_cutoff_tightens(self):
        """FBBT with cutoff produces tighter bounds than without."""
        model = _make_linear_model()
        repr_ = model_to_repr(model)

        # Without cutoff
        lbs0, ubs0 = repr_.fbbt(max_iter=10, tol=1e-8)
        # With cutoff=7 (objective x <= 7)
        lbs1, ubs1 = repr_.fbbt_with_cutoff(max_iter=10, tol=1e-8, incumbent_bound=7.0)

        lbs0, ubs0 = np.asarray(lbs0), np.asarray(ubs0)
        lbs1, ubs1 = np.asarray(lbs1), np.asarray(ubs1)

        # x bound should be tighter: 7 < 10
        assert ubs1[0] <= ubs0[0]
        np.testing.assert_allclose(ubs1[0], 7.0, atol=1e-8)
        # y bound should be same (FBBT from constraints)
        np.testing.assert_allclose(ubs1[1], 10.0, atol=1e-8)

    def test_none_matches_fbbt(self):
        """FBBT with incumbent_bound=None matches plain FBBT."""
        model = _make_linear_model()
        repr_ = model_to_repr(model)

        lbs0, ubs0 = repr_.fbbt(max_iter=10, tol=1e-8)
        lbs1, ubs1 = repr_.fbbt_with_cutoff(max_iter=10, tol=1e-8, incumbent_bound=None)

        np.testing.assert_allclose(np.asarray(lbs0), np.asarray(lbs1), atol=1e-14)
        np.testing.assert_allclose(np.asarray(ubs0), np.asarray(ubs1), atol=1e-14)

    def test_correct_shape(self):
        """Returned arrays have one element per variable block."""
        model = _make_linear_model()
        repr_ = model_to_repr(model)
        lbs, ubs = repr_.fbbt(max_iter=10, tol=1e-8)
        assert len(np.asarray(lbs)) == repr_.n_var_blocks
        assert len(np.asarray(ubs)) == repr_.n_var_blocks

    def test_infeasibility(self):
        """FBBT detects infeasibility from impossible cutoff."""
        model = _make_linear_model()
        repr_ = model_to_repr(model)

        # cutoff=-1 means x <= -1, but x >= 0 => infeasible
        lbs, ubs = repr_.fbbt_with_cutoff(max_iter=10, tol=1e-8, incumbent_bound=-1.0)
        lbs, ubs = np.asarray(lbs), np.asarray(ubs)

        # Infeasible => lbs > ubs (empty intervals)
        for i in range(len(lbs)):
            assert lbs[i] > ubs[i], f"Expected empty interval for var {i}"
