"""#2013: the HiGHS ``lp_nlp_bb`` master separates at its LP relaxation
(``DISCOPT_HIGHS_MASTER_LP_CUTS``).

HiGHS has no hook at fractional nodes, so the routed convex-MINLP master received
OA rows only at integer incumbents and kept its initial linearization everywhere
else (smallinvDAXr2b200-220: 21,709 nodes uncertified at 30 s). Under the flag the
separator also runs at the master's LP relaxation before each tree. These tests pin,
on a toy master ``min t  s.t.  t >= (x - 1.3)**2 + (y - 2.6)**2``, ``x, y``
integer, whose separator returns the tangent of the convex function:

* the rounds call the separator at a FRACTIONAL point and add rows;
* the answer is the true optimum either way (the rows are valid everywhere);
* ``node_callback`` without rounds is still refused, never silently dropped.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solvers.milp_highs import solve_milp_with_lazy_cuts

A, B = 1.3, 2.6
OPT = (1 - A) ** 2 + (3 - B) ** 2  # x = 1, y = 3


def _tangent(p):
    x, y = float(p[0]), float(p[1])
    g = (x - A) ** 2 + (y - B) ** 2
    gx, gy = 2 * (x - A), 2 * (y - B)
    # t >= g + gx (X - x) + gy (Y - y)  <=>  gx X + gy Y - t <= gx x + gy y - g
    return np.array([gx, gy, -1.0]), gx * x + gy * y - g


def _separator(points):
    def sep(p):
        p = np.asarray(p, dtype=np.float64)
        points.append(p.copy())
        x, y, t = p
        if t >= (x - A) ** 2 + (y - B) ** 2 - 1e-9:
            return []
        return [_tangent(p)]

    return sep


def _solve(node_rounds, lazy_pts, node_pts):
    a0, b0 = _tangent(np.array([0.0, 0.0]))
    return solve_milp_with_lazy_cuts(
        c=np.array([0.0, 0.0, 1.0]),
        A_ub=a0[None, :],
        b_ub=np.array([b0]),
        bounds=[(0, 5), (0, 5), (0, 100)],
        integrality=np.array([1, 1, 0]),
        time_limit=30,
        lazy_callback=_separator(lazy_pts),
        node_callback=_separator(node_pts) if node_rounds else None,
        node_lp_rounds=node_rounds,
    )


@pytest.mark.smoke
def test_lp_rounds_separate_at_a_fractional_point_and_keep_the_optimum():
    lazy_pts: list = []
    node_pts: list = []
    r = _solve(10, lazy_pts, node_pts)
    assert r.objective == pytest.approx(OPT, abs=1e-6)
    assert node_pts, "the LP-relaxation rounds never called the separator"
    frac = [p for p in node_pts if np.any(np.abs(p[:2] - np.round(p[:2])) > 1e-6)]
    assert frac, "every LP point was integral: the rounds measured nothing fractional"
    plain = _solve(0, [], [])
    assert plain.objective == pytest.approx(OPT, abs=1e-6)


@pytest.mark.smoke
def test_node_callback_without_rounds_is_refused():
    with pytest.raises(NotImplementedError, match="node_lp_rounds"):
        solve_milp_with_lazy_cuts(
            c=np.array([1.0]),
            bounds=[(0, 1)],
            integrality=np.array([1]),
            lazy_callback=lambda x: [],
            node_callback=lambda x: [],
        )
