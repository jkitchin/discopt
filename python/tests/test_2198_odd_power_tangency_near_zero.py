"""#2198 — the odd-power hull used the secant as a facet when the tangency lay near 0.

``_odd_power_tangent_point`` bracketed the tangency of ``t**p`` from
``(lo, lo**p)`` starting at ``1e-12·max(hi, 1)``. The root is ``c_p·|lo|``
(``c_3 = 1/2``), so for a tiny negative ``lo`` it lies LEFT of that start, the
bracket showed no sign change, and the function returned ``None`` -- which
``_emit_odd_power_hull`` reads as "the tangency is beyond ``hi``" and answers with
the secant ``lo -> hi``. On such a box that secant lies ABOVE the graph and cuts
feasible points. The trigger is ordinary: ``1 - x`` with ``x in [0, 1]`` has the
outward-rounded enclosure ``[-2.2e-16, 1]``, so ``(1 - x)**3`` got the bad facet at
every node whose box contains ``x = 1``. The upper facet had the mirror bug.

Measured before the fix on the issue's knapsack: ``optimal`` objective/bound
``-109`` after one node, while ``-121`` is feasible. After: ``-121``, bound
``-121.00000008``.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.uniform_relax import (
    LinForm,
    _emit_odd_power_hull,
    _odd_power_tangent_point,
)


class _RowSink:
    """Minimal stand-in for the builder: records ``coeffs·cols <= rhs`` rows."""

    def __init__(self) -> None:
        self.rows: list[tuple[dict[int, float], float]] = []

    def add_row(self, coeffs: dict[int, float], rhs: float) -> None:
        self.rows.append((dict(coeffs), float(rhs)))


_T, _W = 0, 1  # column ids: the base t and the aux w = t**p

BOXES = [
    (-(2.0**-50), 2.0),  # the issue's lower-facet case
    (-2.220446049250313e-16, 1.0),  # the (1 - x) enclosure that triggers it in a solve
    (-1.0, 1e-16),  # the mirrored upper-facet case
    (-1e-13, 5.0),
    (-3.0, 1e-14),
    (-1e-300, 1.0),  # lo**p underflows: the slope-0 facet must still be valid
    (-1.0, 2.0),  # ordinary tangency inside the box
    (-1.0, 0.25),  # tangency beyond hi: the secant is the exact lower facet
    (-0.25, 1.0),  # tangency beyond lo: the secant is the exact upper facet
]


def _hull_rows(lo: float, hi: float, p: int):
    sink = _RowSink()
    assert _emit_odd_power_hull(sink, _W, LinForm.col(_T), lo, hi, p)
    assert len(sink.rows) == 2
    return sink.rows


@pytest.mark.parametrize("p", [3, 5, 7])
@pytest.mark.parametrize("box", BOXES)
def test_odd_power_hull_cuts_no_graph_point(box, p):
    lo, hi = box
    rows = _hull_rows(lo, hi, p)
    # Graph points across the box, densely near 0 where the missed tangency lives.
    ts = np.concatenate(
        [
            np.linspace(lo, hi, 2001),
            np.linspace(lo, min(hi, 4.0 * abs(lo)), 2001),
            np.linspace(max(lo, -4.0 * hi), hi, 2001),
            [lo, 0.0, hi, 1.0 if hi >= 1.0 else hi, -1.0 if lo <= -1.0 else lo],
        ]
    )
    checked = 0
    for t in ts:
        t = float(min(max(t, lo), hi))
        w = t**p
        for coeffs, rhs in rows:
            lhs = coeffs.get(_T, 0.0) * t + coeffs.get(_W, 0.0) * w
            # Allow only rounding of the row's own evaluation, not a facet-level cut.
            tol = 1e-12 * (1.0 + abs(rhs) + abs(w) + abs(coeffs.get(_T, 0.0) * t))
            assert lhs <= rhs + tol, (box, p, t, coeffs, rhs, lhs - rhs)
            checked += 1
    assert checked > 0


def test_issue_secant_would_have_cut_the_graph():
    """The pre-fix facet was the secant; pin that it is NOT what is emitted now."""
    lo, hi = -(2.0**-50), 2.0
    secant = (hi**3 - lo**3) / (hi - lo)
    assert lo**3 + secant * (1.0 - lo) > 1.0 + 1.0  # the secant at t=1 exceeds 1**3 by ~3
    (under, _over) = _hull_rows(lo, hi, 3)
    coeffs, rhs = under  # -w + m t <= rhs
    assert -1.0 + coeffs[_T] * 1.0 <= rhs + 1e-12


@pytest.mark.parametrize(
    "p, endpoint, lo, hi, expected",
    [
        (3, -(2.0**-50), -(2.0**-50), 2.0, 2.0**-51),  # exact root -lo/2
        (3, 1e-16, -1.0, 1e-16, -0.5e-16),  # mirrored upper facet
        (3, -1.0, -1.0, 2.0, 0.5),
    ],
)
def test_tangent_point_is_found_near_zero(p, endpoint, lo, hi, expected):
    t = _odd_power_tangent_point(p, endpoint, lo, hi)
    assert t is not None
    assert math.isclose(t, expected, rel_tol=1e-12)
    # The returned point is on the 0 side of the root, so the slope never overshoots.
    assert abs(t) <= abs(expected) * (1.0 + 1e-15)


def test_tangent_point_none_only_when_certified_outside():
    assert _odd_power_tangent_point(3, -1.0, -1.0, 0.25) is None  # root 0.5 > hi
    assert _odd_power_tangent_point(3, 1.0, -0.25, 1.0) is None  # root -0.5 < lo
    # Underflow of endpoint**p is inconclusive: the slope-0 bound row, never the secant.
    assert _odd_power_tangent_point(5, -1e-200, -1e-200, 3.0) == 0.0


def _knapsack(affine_base: bool):
    w = [33, 35, 47, 57, 11]
    v = [43, 45, 57, 67, 21]
    m = dm.Model("issue_2198")
    z = [m.binary(f"z{i}") for i in range(len(w))]
    terms = []
    for i in range(len(w)):
        if affine_base:
            x = m.continuous(f"x{i}", lb=0, ub=1)
            m.subject_to(x >= 1 - z[i])
            s = 1 - x
        else:
            s = m.continuous(f"s{i}", lb=0, ub=1)
            m.subject_to(s <= z[i])
        terms.append(v[i] * 6.75 * (s**3 - s**2))
    m.subject_to(sum(w[i] * z[i] for i in range(len(w))) <= 91)
    m.minimize(sum(terms))
    return m


@pytest.mark.parametrize("affine_base", [True, False])
def test_issue_knapsack_certifies_the_true_optimum(affine_base):
    # z = (1,0,1,0,1), s = 2/3 on those items is feasible with objective -121.
    r = _knapsack(affine_base).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-121.0, abs=1e-5)
    assert r.bound <= -121.0 + 1e-6
