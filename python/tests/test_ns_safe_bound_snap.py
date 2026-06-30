"""Unit tests for the reduced-cost *snap* in ``_ns_safe_lp_lower_bound``.

The snap lets the Neumaier-Shcherbina safe bound consume an interior-point dual
(POUNCE) rather than a vertex dual. A column with an infinite binding bound must
have reduced cost exactly 0 in any dual-feasible point; an IPM reports a small
residual instead, and ``rc * inf`` would blow the bound to ``-inf`` (returned as
``None``). Snapping ``|rc| <= rc_snap_tol`` to 0 *only* where the binding bound
is infinite recovers the KKT value. These tests are solver-free: they feed the
function hand-built duals so the arithmetic is exercised deterministically.
"""

from __future__ import annotations

import numpy as np
from discopt._jax.obbt import _ns_safe_lp_lower_bound


def _problem_with_free_column():
    """min x0  s.t.  x0 + x1 <= 4,  x0 in [0,10],  x1 free.

    True min is 0 (x0=0; x1 free absorbs the row). With a tiny row dual the free
    column x1 gets a near-zero *positive* reduced cost against lo = -inf -- the
    trap the snap targets.
    """
    c = np.array([1.0, 0.0])
    A_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([4.0])
    lo = np.array([0.0, -np.inf])
    hi = np.array([10.0, np.inf])
    # HiGHS convention: y = max(-row_dual, 0). Pick row_dual = -1e-9 so y = 1e-9,
    # giving rc1 = y = 1e-9 > 0 on the free (lo=-inf) column.
    dual_values = np.array([-1e-9])
    return c, dual_values, A_ub, b_ub, lo, hi


def test_snap_disabled_by_default_drops_trapped_bound():
    """Default rc_snap_tol=0.0: the infinite-bound residual traps -> None."""
    c, dv, A_ub, b_ub, lo, hi = _problem_with_free_column()
    assert _ns_safe_lp_lower_bound(c, dv, A_ub, b_ub, lo, hi) is None


def test_snap_recovers_finite_sound_bound():
    """rc_snap_tol>0: bound becomes finite and remains a sound under-estimate."""
    c, dv, A_ub, b_ub, lo, hi = _problem_with_free_column()
    g = _ns_safe_lp_lower_bound(c, dv, A_ub, b_ub, lo, hi, rc_snap_tol=1e-6)
    assert g is not None
    assert np.isfinite(g)
    # True min is 0; a safe bound must be <= 0 (a touch below, never above).
    assert g <= 0.0 + 1e-7


def test_snap_does_not_mask_genuine_unboundedness():
    """A reduced cost *above* the tol on an infinite-bound column is real
    unboundedness, must NOT be snapped, and still yields None."""
    c = np.array([1.0, 0.0])
    A_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([4.0])
    lo = np.array([0.0, -np.inf])
    hi = np.array([10.0, np.inf])
    dual_values = np.array([-0.5])  # y=0.5 -> rc1=0.5 >> tol on a free column
    assert _ns_safe_lp_lower_bound(c, dual_values, A_ub, b_ub, lo, hi, rc_snap_tol=1e-6) is None


def test_snap_is_noop_on_finite_bounds():
    """With all bounds finite there is no trap; snap on/off agree exactly."""
    c = np.array([2.0, -1.0])
    A_ub = np.array([[1.0, 1.0], [-1.0, 2.0]])
    b_ub = np.array([5.0, 3.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([10.0, 10.0])
    dual_values = np.array([-0.3, -0.1])
    g0 = _ns_safe_lp_lower_bound(c, dual_values, A_ub, b_ub, lo, hi)
    g1 = _ns_safe_lp_lower_bound(c, dual_values, A_ub, b_ub, lo, hi, rc_snap_tol=1e-6)
    assert g0 is not None and g1 is not None
    assert g0 == g1
