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

import warnings

import pytest

scipy_opt = pytest.importorskip("scipy.optimize")
from discopt.bilevel import BilevelProblem  # noqa: E402
from discopt.modeling.core import Model  # noqa: E402


def _bard_lp(**kw):
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
        **kw,
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


# ---------------------------------------------------------------------------
# 3. Certified solve via a user-supplied multiplier bound.
# ---------------------------------------------------------------------------


def test_default_still_refuses_without_multiplier_ub():
    """No multiplier_ub → the big-M path still refuses (unbounded multipliers)."""
    _m, _x, _y, bl = _bard_lp()
    with pytest.raises(NotImplementedError, match="unbounded|multiplier_ub"):
        bl.formulate(method="kkt", mpec_method="gdp")


@pytest.mark.parametrize("bad", [0.0, -5.0, float("inf")])
def test_invalid_multiplier_ub_raises(bad):
    _m, _x, _y, bl = _bard_lp(multiplier_ub=bad)
    with pytest.raises(ValueError, match="finite positive"):
        bl.formulate(method="kkt", mpec_method="gdp")


@pytest.mark.timeout(120)
def test_user_multiplier_ub_gives_certified_solve():
    """A valid user-supplied multiplier bound makes kkt+gdp gap-certified (LP follower).

    True follower duals here are O(1), so multiplier_ub=50 is valid. The solve must
    reach the true optimum with gap_certified=True and a follower-optimal y.
    """
    pytest.importorskip("discopt._rust")
    m, x, y, bl = _bard_lp(multiplier_ub=50.0)
    bl.formulate(method="kkt", mpec_method="gdp")
    r = m.solve(time_limit=90)
    assert r.status == "optimal", f"unexpected status {r.status}"
    assert r.objective == pytest.approx(-7.0, abs=1e-3)
    xv, yv = float(r.value(x)), float(r.value(y))
    assert xv == pytest.approx(1.0, abs=1e-2)
    assert yv == pytest.approx(2.0, abs=1e-2)
    assert yv == pytest.approx(_follower_argmin(xv), abs=1e-2)
    assert getattr(r, "gap_certified", False) is True


def test_multiplier_bound_active_warning_is_best_effort():
    """The solve() guard warns when a multiplier sits at its supplied bound.

    Driven with a stub result (deterministic, solver-free): the guard is a
    best-effort signal that a too-small multiplier_ub may be cutting the true
    follower response. It does not catch every unsound bound, only the at-bound case.
    """
    import numpy as np

    m, _x, _y, bl = _bard_lp(multiplier_ub=10.0)
    bl.build_kkt_system()
    bl._apply_multiplier_ub()
    mu0 = bl.kkt.comp_pairs[0].f

    class _Stub:
        def __init__(self, at_bound):
            self._at = at_bound

        def value(self, v):
            return np.asarray(10.0 if (v is mu0 and self._at) else 0.0)

    with pytest.warns(UserWarning, match="multiplier_ub"):
        bl._warn_if_multiplier_bound_active(_Stub(at_bound=True))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning expected when well below the bound
        bl._warn_if_multiplier_bound_active(_Stub(at_bound=False))


# ---------------------------------------------------------------------------
# 4. Convex-NLP follower solves end-to-end via strong_duality.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(180)
def test_convex_nlp_follower_solves_via_strong_duality():
    """Follower min exp(y) - x*y (convex in y) → y* = ln(x); leader min (y-1)^2.

    The optimistic optimum is y=1, x=e (obj 0). Confirms the convex-NLP lower level
    (accepted by the interval-Hessian-in-y certifier) reduces and solves end-to-end,
    with the returned y satisfying the follower's first-order condition exp(y) = x.
    """
    import numpy as np
    from discopt.modeling.core import FunctionCall, Model

    pytest.importorskip("discopt._rust")
    m = Model("nlp_bilevel")
    x = m.continuous("x", lb=0.5, ub=8.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize((y - 1.0) * (y - 1.0))
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=FunctionCall("exp", y) - x * y,
        lower_constraints=[],
    )
    bl.formulate(method="strong_duality")
    r = m.solve(time_limit=150)
    assert r.status in ("optimal", "feasible"), f"unexpected status {r.status}"
    assert r.objective is not None
    xv, yv = float(r.value(x)), float(r.value(y))
    assert yv == pytest.approx(1.0, abs=2e-2)
    assert xv == pytest.approx(np.e, abs=5e-2)
    # Follower first-order optimality: exp(y) - x == 0.
    assert abs(np.exp(yv) - xv) < 5e-2
