"""#1413: alpha-BB must cover every curved variable, not the ones above a threshold.

``generate_alphabb_quadratic_oa_cuts_from_evaluator`` emits the tangent of

    q_under(x) = q(x) - sum_i alpha_i (x_i - lb_i)(ub_i - x_i)

and that tangent underestimates ``q`` only where ``q_under`` is convex. ``alpha`` was
placed on ``curved``, selected by ``|hess| > hessian_tol`` with ``hessian_tol = 1e-8``,
while ``under_grad`` is the FULL Jacobian row. A variable whose entire Hessian row and
column sat below 1e-8 was therefore dropped, kept its negative curvature, and left
``q_under`` *concave* in that direction -- where a tangent is an OVERestimator. The
emitted "underestimator" then removed points satisfying ``q(x) <= 0``.

Measured on ``cedf1f31`` before the fix, with ``5*x0*x1 - eps*x2^2 <= 0`` and
``eps = 4.9e-9`` (x2's whole Hessian row is ``-9.8e-9``, just inside the threshold):

    x2 box width   1e4 -> cut removes a feasible point by 3.175e-01
                   1e5 -> 3.670e+01
                   1e6 -> 3.675e+03

matching the predicted ``0.75 * eps * W^2`` to four significant figures. The firing
condition needs ``x2*`` at a box extreme, which is where an LP relaxation solution sits.

These tests pin the *class*: no emitted cut may remove a point that satisfies the row it
was derived from, at any scale. They are not tied to the particular instance below, which
is only the smallest reproducer.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax._numeric import is_effectively_finite
from discopt._relax.cutting_planes import (
    _constraint_row_quadratic_hessian,
    generate_alphabb_quadratic_oa_cuts_from_evaluator,
)
from discopt._relax.model_utils import flat_variable_bounds
from discopt._relax.nlp_evaluator import NLPEvaluator

FEAS_TOL = 1e-6
BILIN = 5.0
# Just inside the old 1e-8 exclusion threshold: 2*EPS = 9.8e-9.
EPS = 4.9e-9


def _model(eps: float, width: float, delta: float = 0.1):
    """``BILIN*x0*x1 - eps*x2^2 <= 0``; x2's only Hessian entry is ``-2*eps``.

    ``delta`` is the half-width of x0/x1. It matters because the alpha term on the
    bilinear pair is what keeps the cut valid and is bounded by ``alpha*delta^2``; the
    uncovered curvature on x2 has to exceed that to actually cut a feasible point. A
    narrow box on the bilinear pair is ordinary after a few rounds of branching.
    """
    m = dm.Model()
    m.continuous("x0", lb=-delta, ub=delta)
    m.continuous("x1", lb=-delta, ub=delta)
    m.continuous("x2", lb=-width / 2.0, ub=width / 2.0)
    x0, x1, x2 = m._variables
    m.minimize(x0 + x1 + x2)
    m.subject_to(BILIN * x0 * x1 - eps * x2 * x2 <= 0.0)
    return m


def _row(x, eps: float) -> float:
    return BILIN * x[0] * x[1] - eps * x[2] * x[2]


def _cuts_at_upper_x2(m, eps: float):
    """Cuts at an ``x*`` with x2 AT its upper bound -- where the exposure is maximal.

    With ``x2* = 0`` the tangent of the concave ``-eps*x2^2`` is flat, the cut carries no
    x2 term, and the defect cannot express itself. An earlier version of this probe used
    that point and reported the site clean at every scale; the evaluation point is part
    of the test, not an incidental detail.
    """
    ev = NLPEvaluator(m)
    lb, ub = flat_variable_bounds(m)
    x_star = np.array([0.0, 0.0, ub[2]], dtype=np.float64)
    cuts = generate_alphabb_quadratic_oa_cuts_from_evaluator(
        ev,
        x_star,
        lb,
        ub,
        constraint_senses=[c.sense for c in m._constraints],
        convex_mask=[False],
    )
    return cuts, lb, ub


def _worst_violation_of_a_feasible_point(cuts, lb, ub, eps: float) -> float:
    """Largest amount by which any emitted cut removes a point satisfying the row."""
    rng = np.random.default_rng(1413)
    pts = np.column_stack(
        [
            rng.uniform(lb[0], ub[0], 4000),
            rng.uniform(lb[1], ub[1], 4000),
            rng.uniform(lb[2], ub[2], 4000),
        ]
    )
    # The worst case is at a box corner, where the alpha term that buys validity
    # vanishes; uniform sampling essentially never lands there.
    corners = np.array(
        [[a, b, c] for a in (lb[0], ub[0]) for b in (lb[1], ub[1]) for c in (lb[2], ub[2], 0.0)],
        dtype=np.float64,
    )
    derived = np.array([[0.0, 0.0, lb[2]]], dtype=np.float64)
    pts = np.vstack([derived, corners, pts])
    feas = pts[np.array([_row(p, eps) <= 0.0 for p in pts])]
    assert feas.shape[0] > 0, "no feasible sample -- the assertion below would be vacuous"

    worst = 0.0
    for cut in cuts:
        vals = feas @ np.asarray(cut.coeffs, dtype=np.float64)
        if cut.sense == "<=":
            viol = vals - cut.rhs
        elif cut.sense == ">=":
            viol = cut.rhs - vals
        else:  # pragma: no cover - the generator only emits "<="
            raise AssertionError(f"unexpected cut sense {cut.sense!r}")
        worst = max(worst, float(np.max(viol)))
    return worst


def test_the_dropped_variable_really_was_below_the_old_threshold():
    """Guard the premise: without this, every assertion below is vacuous."""
    m = _model(EPS, 1e4)
    ev = NLPEvaluator(m)
    lb, _ = flat_variable_bounds(m)
    hess = _constraint_row_quadratic_hessian(ev, 0, lb.size)
    assert hess is not None, "row must be recognised as quadratic"
    assert abs(hess[2, 2]) < 1e-8, (
        f"x2's Hessian entry {hess[2, 2]:.3e} must sit inside the old 1e-8 threshold, "
        "or this fixture no longer reproduces the class"
    )
    assert hess[2, 2] < 0.0, "x2's curvature must be NEGATIVE for the defect to bite"
    assert np.count_nonzero(hess) > 0


@pytest.mark.parametrize("width", [1e4, 1e5, 1e6])
def test_no_cut_removes_a_point_that_satisfies_its_own_row(width):
    """The bug, swept over the scale that drives it (#1397 parameter sweep).

    Before the fix: 3.175e-01 / 3.670e+01 / 3.675e+03 at these three widths.
    """
    m = _model(EPS, width)
    cuts, lb, ub = _cuts_at_upper_x2(m, EPS)
    worst = _worst_violation_of_a_feasible_point(cuts, lb, ub, EPS)
    assert worst <= FEAS_TOL, (
        f"a cut removed a point satisfying the row it was derived from, by {worst:.3e} "
        f"(x2 box width {width:.0e}); alpha did not cover every curved variable"
    )


@pytest.mark.parametrize("width", [1e2, 1e3])
def test_narrow_boxes_were_already_safe_and_stay_safe(width):
    """No regression on the scales where the old code happened to be sound."""
    m = _model(EPS, width)
    cuts, lb, ub = _cuts_at_upper_x2(m, EPS)
    assert _worst_violation_of_a_feasible_point(cuts, lb, ub, EPS) <= FEAS_TOL


def test_an_ordinary_nonconvex_row_still_gets_its_cut():
    """No capability loss: the generator must still fire where it always did."""
    m = _model(1e-3, 1e2)
    cuts, lb, ub = _cuts_at_upper_x2(m, 1e-3)
    assert cuts, "lost the cut on an ordinary nonconvex quadratic row"
    assert _worst_violation_of_a_feasible_point(cuts, lb, ub, 1e-3) <= FEAS_TOL


def test_a_structurally_absent_variable_is_not_pulled_into_the_support():
    """Exactly-zero is the test, so a variable absent from the row stays absent.

    ``x3`` appears only linearly. If the support selection treated "absent" as
    "small", x3 would be pulled in and its (here finite) bounds would start
    mattering -- and with an INFINITE bound it would refuse the row outright.
    """
    m = dm.Model()
    m.continuous("x0", lb=-1.0, ub=1.0)
    m.continuous("x1", lb=-1.0, ub=1.0)
    m.continuous("x3")  # no bounds: the default +-9.999e19 is not effectively finite
    x0, x1, x3 = m._variables
    m.minimize(x0 + x1 + x3)
    m.subject_to(BILIN * x0 * x1 + x3 <= 0.0)

    ev = NLPEvaluator(m)
    lb, ub = flat_variable_bounds(m)
    assert not is_effectively_finite(float(ub[2])), (
        "x3 must be unbounded for this test to distinguish absent from tiny"
    )
    hess = _constraint_row_quadratic_hessian(ev, 0, lb.size)
    assert hess is not None
    assert hess[2, 2] == 0.0, "x3 is linear; its curvature must be an EXACT zero"

    cuts = generate_alphabb_quadratic_oa_cuts_from_evaluator(
        ev,
        np.array([0.0, 0.0, 0.0]),
        lb,
        ub,
        constraint_senses=[c.sense for c in m._constraints],
        convex_mask=[False],
    )
    assert cuts, (
        "a linearly-appearing variable with an unbounded box must not block the cut; "
        "structurally absent is not the same as computed-and-tiny"
    )
