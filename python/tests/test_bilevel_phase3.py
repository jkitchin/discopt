"""Bilevel Phase 3: end-to-end solve behavior (the audit's missing coverage).

The phase-1/2 suites validate the KKT/strong-duality *reformulation* against a
scipy oracle but never call ``model.solve()`` — so a false optimal in the actual
global solve went uncaught. This file closes that gap and pins the audit fix:

* The KKT ``gdp``/``sos1`` big-M encodings **refuse loudly** when the follower's
  multipliers are unbounded (the common case) — a sentinel-sized big-M is
  numerically vacuous and would certify a follower-infeasible point (a false
  optimum). This is the regression for the shared GDP big-M fix
  (``_jax/gdp_reformulate._BIGM_SENTINEL``).
* The ``strong_duality`` reduction (a single bilinear equality, no big-M) **solves**
  a linear bilevel to the true optimistic optimum, and the returned follower ``y``
  is genuinely the follower's argmin at the returned leader ``x`` (scipy oracle).

The solve needs the Rust extension + an NLP backend; the refusal tests are
solver-free.
"""

from __future__ import annotations

import pytest

scipy_opt = pytest.importorskip("scipy.optimize")
from discopt.bilevel import BilevelProblem  # noqa: E402
from discopt.modeling.core import Model  # noqa: E402


def _bard_lp():
    """Linear bilevel with a known optimistic optimum.

    Leader:   min_{x,y} x - 4y
    Follower: min_y  y   s.t.  x + y >= 3,  y <= 2x,  y in [0, 10],  x in [0, 10].

    The follower minimises y, so at leader x it picks y = max(0, 3 - x) subject to
    y <= 2x (feasible only for x >= 1). Leader objective x - 4y over x in [1, 3] is
    5x - 12, minimised at x = 1 (y = 2) → obj = -7. (For x > 3 the follower gives
    y = 0 and the leader objective only grows.)
    """
    m = Model("bard")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - 4 * y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[x + y >= 3, y <= 2 * x],
        lower_sense="min",
    )
    return m, x, y, bl


def _follower_argmin(xv: float) -> float:
    res = scipy_opt.linprog(
        c=[1.0],
        A_ub=[[-1.0], [1.0]],
        b_ub=[xv - 3.0, 2.0 * xv],
        bounds=[(0.0, 10.0)],
        method="highs",
    )
    assert res.success, res.message
    return float(res.x[0])


# ---------------------------------------------------------------------------
# 1. Unbounded KKT multipliers → the big-M encodings refuse loudly.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("mpec_method", ["gdp", "sos1"])
def test_kkt_bigm_refuses_unbounded_multipliers(mpec_method):
    """Certifying an unbounded KKT multiplier via big-M would be a false optimum."""
    _m, _x, _y, bl = _bard_lp()
    with pytest.raises(NotImplementedError, match="unbounded|strong_duality"):
        bl.formulate(method="kkt", mpec_method=mpec_method)


# ---------------------------------------------------------------------------
# 2. strong_duality solves the linear bilevel to the true optimum.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_strong_duality_solves_linear_bilevel_to_true_optimum():
    pytest.importorskip("discopt._rust")
    m, x, y, bl = _bard_lp()
    bl.formulate(method="strong_duality")
    r = m.solve(time_limit=90)
    assert r.status in ("optimal", "feasible"), f"unexpected status {r.status}"
    assert r.objective is not None, "no incumbent returned"
    xv, yv = float(r.value(x)), float(r.value(y))
    # True optimistic optimum.
    assert r.objective == pytest.approx(-7.0, abs=1e-3)
    assert xv == pytest.approx(1.0, abs=1e-2)
    assert yv == pytest.approx(2.0, abs=1e-2)
    # The returned y is genuinely the follower's argmin at the returned x
    # (guards against certifying a follower-infeasible point).
    assert yv == pytest.approx(_follower_argmin(xv), abs=1e-2)
