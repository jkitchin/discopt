"""#866: the repr QP extractor dropped the quadratic term at large coefficient scale.

`_extract_qp_data_from_repr` recovers the objective by unit probes::

    d = f(0);   Q[j,j] = f(e_j) + f(-e_j) - 2d;   c_j = f(e_j) - d - 0.5*Q[j,j]

Each identity is exact in real arithmetic and a difference of nearly-equal floats
in practice. On ``min (x - 1e10)**2`` the constant term is 1e20, whose ulp is
~16384, so the ``+1`` from ``x**2`` is lost in rounding and the extractor returns
**Q = 0** — the objective silently becomes linear.

Measured before the fix::

    min (x - 1e10)**2  s.t.  x in [1, 1e11]
      -> status=optimal, objective=-8.9999989761e+20, gap_certified=True
      -> returned x = 5.0e10, whose TRUE objective is 1.6e21
      -> true optimum is 0.0 at x = 1e10

A **certified negative objective for a sum of squares** (CLAUDE.md §1), on the
default path. At wider boxes the same corruption instead produced ``unbounded``.

The fix does not try to make the cancelling probes accurate — it checks them. The
recovered ``(Q, c, d)`` is re-evaluated against the model's own objective at
box-scale points, and a disagreement raises so the dispatcher falls through to the
autodiff extractor, which recovers ``Q=2, c=-2e10`` exactly here. A bad extraction
now degrades to a slower-but-correct one instead of a wrong answer.

Residual (documented, not a regression): the reported objective can still be off by
about one ulp of the constant term (±16384 against 1e20) because the *expanded*
quadratic form cancels in float64. The returned point is correct — its true
objective is ~1e-11 — so this is report-level noise at 1.6e-16 relative, not a
dropped term.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.problem_classifier import extract_qp_data  # noqa: E402


def _model(ub: float):
    """``min (x - 1e10)**2`` on ``[1, ub]`` — convex, true optimum 0.0 at x=1e10."""
    m = dm.Model("shifted_square")
    x = m.continuous("x", lb=1.0, ub=ub)
    m.minimize((x - 1e10) ** 2)
    return m, x


def test_extraction_recovers_the_quadratic_term():
    """The dispatched extractor must not lose ``x**2``. Pre-fix this returned
    ``Q=[0.]`` and a linear objective."""
    data = extract_qp_data(_model(1e11)[0])
    q = float(np.asarray(data.Q).ravel()[0])
    c = float(np.asarray(data.c).ravel()[0])
    assert q == pytest.approx(2.0, rel=1e-9), f"quadratic term dropped: Q={q}"
    assert c == pytest.approx(-2e10, rel=1e-9), f"linear term corrupted: c={c}"


@pytest.mark.parametrize("ub", [2e10, 1e11, 1e12, 1e14, 1e16, 1e18, 1e19])
def test_no_false_optimum_and_no_false_unbounded(ub):
    """Across the scale sweep: never ``unbounded`` (the model is bounded and
    feasible), and never a grossly negative objective for a sum of squares."""
    m, _x = _model(ub)
    r = m.solve(time_limit=30)
    assert r.status != "unbounded", (
        f"ub={ub:.0e}: FALSE UNBOUNDED on a bounded, feasible model (true optimum 0.0)"
    )
    if r.objective is not None:
        # A square is >= 0. Allow ulp-of-1e20 noise from the expanded QP form, but
        # nothing on the scale of the pre-fix -9e20.
        assert r.objective > -1e6, f"ub={ub:.0e}: certified {r.objective!r} for a sum of squares"


@pytest.mark.parametrize("ub", [2e10, 1e11, 1e12, 1e16, 1e19])
def test_returned_point_is_the_true_minimizer(ub):
    """The strongest check, and the one the pre-fix solver failed hardest: the
    returned point must actually minimize the true objective. Pre-fix it returned
    x=5e10 (true objective 1.6e21) for ub=1e11."""
    m, x = _model(ub)
    r = m.solve(time_limit=30)
    if r.objective is None:
        pytest.skip("no incumbent returned")
    x_val = float(np.asarray(r.value(x)).ravel()[0])
    true_obj = (x_val - 1e10) ** 2
    assert true_obj < 1.0, (
        f"ub={ub:.0e}: returned x={x_val!r} has true objective {true_obj!r}, "
        "not the minimizer (true optimum 0.0 at x=1e10)"
    )
