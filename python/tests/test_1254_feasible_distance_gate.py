"""A feasibility gate must ask how FAR the point is, not just how big the residual (#1254).

``10**y1 + 10**y2 <= 10**z`` over ``y in [-9,-6]``, ``z in [-20,0]`` has its
optimum at ``z* = log10(2e-7) = -6.69897``. At ``y1 = y2 = -7, z = -20`` the row
is violated by 2.0e-07, which every absolute tolerance in the solver (1e-6 in the
incumbent verifier, 1e-4 in the heuristic gates) accepted — so the point became
the incumbent, met the root relaxation bound, and ``solve`` returned
``status="optimal"``, ``gap_certified=True`` at ``z = -20``. A false certificate,
13 orders from the truth, produced silently.

The residual was never the right question. Every partial derivative of that row
at that point is ~2.3e-07 or smaller, so the nearest point that satisfies it is
``Δy ~ 0.87`` away — a third of ``y``'s whole box. The gates now also require the
first-order distance ``violation / ||grad g||_inf`` to stay inside 1e-4, which
rejects this point by nine orders while leaving an ordinary converged local point
(a row with an ordinary gradient) exactly where it was.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.validation.feasibility import (
    FEASIBLE_DISTANCE_TOL,
    feasible_distance_cap,
    jacobian_row_gradient_norms,
    verify_point,
)


def _unscaled_epigraph():
    """The #1254 model. Optimum ``z = log10(2e-7) = -6.69897`` at ``y1 = y2 = -7``."""
    m = dm.Model("unscaled_epigraph")
    y1 = m.continuous("y1", lb=-9.0, ub=-6.0)
    y2 = m.continuous("y2", lb=-9.0, ub=-6.0)
    z = m.continuous("z", lb=-20.0, ub=0.0)
    m.subject_to(y1 + y2 >= -14.0)
    m.subject_to(10.0**y1 + 10.0**y2 <= 10.0**z)
    m.minimize(z)
    return m


TRUE_OPTIMUM = np.log10(2e-7)


@pytest.mark.unit
def test_cap_is_a_distance_not_a_magnitude():
    """The cap must separate #1254's point from the converged-MPEC point.

    Both are relatively-large violations of a tiny row; only the distance to the
    row's surface tells them apart, and it must be read off the row's GRADIENT.
    """
    checked = 0
    # #1254's row at the bad point: every partial is ~2.3e-07.
    bad_grad = 2.302585e-07
    bad_viol = 2.0e-07
    assert bad_viol > feasible_distance_cap(bad_grad), "the #1254 row must be capped out"
    checked += 1
    # The Scholtes-regularized MPEC row ``x*y <= t`` at ``x=1, y=1.67e-08``:
    # relatively a WORSE violation, but ``dg/dx = 1`` — a converged local point.
    ok_grad, ok_viol = 1.0, 6.7e-09
    assert ok_viol <= feasible_distance_cap(ok_grad), "an ordinary gradient must not be capped"
    checked += 1
    # Inert on every row whose gradient is of ordinary size. The cap carries the
    # 1e-12 noise floor additively since #1392, so it is the allowance to within
    # eight orders rather than bit-for-bit.
    assert feasible_distance_cap(1.0) == pytest.approx(FEASIBLE_DISTANCE_TOL, rel=1e-6)
    assert feasible_distance_cap(1e3) > 1e-4
    checked += 2
    # A non-finite gradient is unestimatable and must not manufacture a cap.
    assert np.isinf(feasible_distance_cap(np.inf)), "an unknown gradient must not cap"
    checked += 1
    assert checked == 5, f"only {checked} assertions executed"


@pytest.mark.unit
def test_gradient_norms_are_read_per_row():
    """``jacobian_row_gradient_norms`` is ``max_j |J_ij|``, with ``inf`` for a bad row."""
    J = np.array([[1.0, -3.0], [0.0, 0.0], [np.inf, 1.0]])
    got = jacobian_row_gradient_norms(J)
    assert got[0] == 3.0
    assert got[1] == 0.0
    assert np.isinf(got[2])
    assert len(got) == 3, "one norm per row"


@pytest.mark.unit
def test_verifier_rejects_the_far_point():
    """``verify_point`` accepted the bad point at 2e-7; it must not."""
    m = _unscaled_epigraph()
    bad = np.array([-7.0, -7.0, -20.0])
    res = verify_point(m, bad, with_objective=True)
    assert not res.ok, f"the far-from-feasible point must not verify, got {res!r}"
    assert "row" in res.reason, f"the rejection must name the row, got {res.reason!r}"
    # …while the model's true optimum, which satisfies the row, still verifies.
    good = np.array([-7.0, -7.0, float(TRUE_OPTIMUM)])
    ok = verify_point(m, good, with_objective=True)
    assert ok.ok, f"the true optimum must verify, got {ok!r}"


@pytest.mark.smoke
def test_solve_does_not_certify_the_floor():
    """End to end: no certificate at ``z = -20``, and the answer is the real one."""
    r = _unscaled_epigraph().solve(time_limit=60)
    x = {k: float(v) for k, v in (r.x or {}).items()}
    assert r.objective is not None and x, "the solve must return a point"
    # The reported point must actually satisfy the row it is reported for.
    violation = 10 ** x["y1"] + 10 ** x["y2"] - 10 ** x["z"]
    assert violation <= 1e-9, f"reported point violates its own row by {violation:.3e}"
    assert r.objective <= TRUE_OPTIMUM + 1e-4, (
        f"objective {r.objective!r} is worse than the true optimum {TRUE_OPTIMUM!r}"
    )
    assert r.objective >= TRUE_OPTIMUM - 1e-4, (
        f"objective {r.objective!r} is BELOW the true optimum {TRUE_OPTIMUM!r}"
    )
    if r.gap_certified:
        # Certifying is fine — certifying the wrong point is not.
        assert r.bound is not None and r.bound <= TRUE_OPTIMUM + 1e-6, (
            f"certified bound {r.bound!r} is above the true optimum"
        )
