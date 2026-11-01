"""#1917 G4.1: why native LOA / RIC did not converge where Pyomo GDPopt does.

Three defects, each measured on GDPlib before the fix:

* **A linear row divided by a constant was not read as linear.**
  ``_extract_body_coeffs`` folded ``+``, ``-``, ``*`` and negation, but not
  ``/``, so ``cost == (a*flow + b)*k/8000`` was left out of every master that
  extracts its rows through it (LOA, GLOA's fallback, RIC, LBB's node
  relaxations). On syngas 57 rows went missing, the master's objective was tied
  to nothing, and RIC's master bound sat at 0 (LOA's at 19.9) against an
  optimum of 4669: 27 configurations in 300 s, no certificate. With the rows,
  both certify in 2 iterations, with Pyomo's master bounds to the digit
  (4667.8407, 4668.8090).
* **LOA linearized rows its master already holds exactly.** An affine row is
  convex, so it got an OA cut every iteration on top of its exact master row
  (med_term_purchasing: 708 per iteration, 35449 after 50). This is the defect
  ``solvers/oa.py``'s ``_oa_cut_eligible_mask`` fixed for OA, where the
  floating-point re-derivation of an ``==`` row made a master infeasible.
* **The HiGHS master withdrew every bound of a model with undeclared bounds.**
  The ``+-9.999e19`` default box is passed to HiGHS as finite, and HiGHS's
  presolve-free cross-solve (#1634) calls the master infeasible at that scale,
  so the claim was withdrawn. The cross-check now re-runs with those sides
  opened -- a relaxation, so its bound and its infeasibility are valid.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.gdp_reformulate import _extract_body_coeffs

pytestmark = pytest.mark.smoke


def _n(m):
    return sum(int(v.size) for v in m._variables)


# ── linear / constant ─────────────────────────────────────────────────────


def test_linear_over_a_constant_is_extracted():
    m = dm.Model("div")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    got = _extract_body_coeffs((2 * x + 3 * y + 6) / 4, m, _n(m))
    assert got is not None, "a linear expression over a constant is linear"
    c, off = got
    np.testing.assert_allclose(c, [0.5, 0.75])
    assert off == pytest.approx(1.5)


def test_division_by_a_variable_or_zero_is_declined():
    m = dm.Model("div")
    x = m.continuous("x", lb=1, ub=10)
    y = m.continuous("y", lb=1, ub=10)
    assert _extract_body_coeffs(x / y, m, _n(m)) is None
    assert _extract_body_coeffs(x / (y - y + 0.0), m, _n(m)) is None


def _cost_by_division():
    """``cost == (80 x + 40)/8``, a disjunction bounding ``x`` below by 3 or 5,
    and a nonlinear row so the model is not a MILP."""
    m = dm.Model("cost_div")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    cost = m.continuous("cost", lb=0, ub=1000)
    m.subject_to(cost == (80 * x + 40) / 8)
    m.subject_to(x * y >= 1)
    m.either_or([[x >= 3], [x >= 5]], name="d")
    m.minimize(cost)
    return m


@pytest.mark.parametrize("method", ["ric", "loa"])
def test_the_master_keeps_a_row_divided_by_a_constant(method):
    """Before the fix the cost row was not in the master: its bound was 0."""
    from discopt.gdp.algorithms.loa import LOA
    from discopt.gdp.algorithms.ric import RIC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        algo = (RIC if method == "ric" else LOA)(_cost_by_division(), time_limit=60)
        r = algo.master.solve(30.0)
    assert r.bound is not None
    assert r.bound == pytest.approx(35.0, abs=1e-6)  # (80*3 + 40)/8


# ── LOA's OA cuts skip exact master rows ──────────────────────────────────


def test_loa_cuts_no_row_the_master_holds_exactly():
    from discopt.gdp.algorithms.loa import LOA

    m = dm.Model("convex_gdp")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x + y <= 6)
    m.subject_to(2 * x - y == 1)
    m.subject_to(x**2 + y**2 <= 12)
    m.either_or([[x >= 1], [y >= 2]], name="d")
    m.minimize(x + 2 * y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        algo = LOA(m, time_limit=60)
    convex = algo.oa_rows.convex_mask
    cuts = algo.oa_cut_rows.convex_mask
    assert sum(convex) > sum(cuts) >= 1, (convex, cuts)
    n_before = len(algo.master.cuts)
    algo._cuts_at(np.full(algo.n, 1.0))
    added = len(algo.master.cuts) - n_before
    # Only the convex nonlinear row x^2 + y^2 <= 12 is linearized.
    assert added == sum(cuts) == 1
    # ``oa_rows`` stays the pure convexity certificate.
    assert sum(convex) >= 3


# ── HiGHS master over the default box ─────────────────────────────────────


def test_highs_master_bound_survives_the_default_box():
    """Reduced from GDPlib med_term_purchasing's first master (two columns of
    498): HiGHS with presolve off calls it infeasible; presolve on, optimal 0."""
    pytest.importorskip("highspy")
    from discopt.constants import DEFAULT_VARIABLE_BOUND
    from discopt.solvers import SolveStatus
    from discopt.solvers.milp_highs import _solve_milp_once, solve_milp

    kw = dict(
        c=np.zeros(2),
        A_ub=None,
        b_ub=None,
        A_eq=sp.csr_matrix(np.array([[0.0, -0.136]])),
        b_eq=np.zeros(1),
        bounds=[(0.0, 1.0), (0.0, DEFAULT_VARIABLE_BOUND)],
        integrality=np.array([1, 0], dtype=np.int32),
        gap_tolerance=1e-4,
    )
    # The precondition this test guards: the presolve-free solve over the
    # declared box is wrong. If a HiGHS upgrade fixes it, the test still passes
    # below but no longer exercises the retry.
    declared = _solve_milp_once(time_limit=10.0, presolve=False, mip_start=None, **kw)
    r = solve_milp(time_limit=10.0, **kw)
    diag = r.callback_stats["presolve_cross_check"]
    if declared.status == SolveStatus.INFEASIBLE:
        assert diag.get("default_box_opened") is True
    assert r.status == SolveStatus.OPTIMAL
    assert r.bound is not None and r.bound == pytest.approx(0.0, abs=1e-9)


def test_default_box_opening_keeps_declared_sides():
    from discopt.constants import DEFAULT_VARIABLE_BOUND
    from discopt.solvers.milp_highs import _default_box_opened
    from discopt.solvers.milp_simplex import _INF

    B = DEFAULT_VARIABLE_BOUND
    kw = {"c": np.zeros(4), "bounds": [(-B, B), (-5.0, B), (0.0, 1e15), (-np.inf, 3.0)]}
    got = _default_box_opened(kw)
    assert got == [(-_INF, _INF), (-5.0, _INF), (0.0, 1e15), (-np.inf, 3.0)]
    assert _default_box_opened({"c": np.zeros(1), "bounds": [(0.0, 1e15)]}) is None


# ── LOA re-solves a configuration its subproblem cap stopped ──────────────


def test_loa_raises_the_cap_instead_of_stopping_on_a_capped_repeat(monkeypatch):
    """A global subproblem stopped by its cap proves nothing, so the master
    proposes the configuration again; LOA used to stop there (C-35's repeat
    rule) with the budget unspent. GDPlib syngas at ``time_limit=60``: 6 s cap,
    28 s needed, stopped at 20 s with no incumbent. Simulated here: a subproblem
    resolves only with a cap of at least 2 s, and the fixed-integer NLP fails."""
    import discopt.gdp.algorithms.subproblem as sub
    from discopt.gdp.algorithms.loa import LOA

    real = sub.solve_subproblem
    caps: list = []

    def capped(*a, time_cap=None, **k):
        caps.append(time_cap)
        if time_cap is not None and time_cap < 2.0:
            return sub.SubproblemResult("unresolved", reason="cap (test)")
        return real(*a, time_cap=time_cap, **k)

    monkeypatch.setattr(sub, "solve_subproblem", capped)
    monkeypatch.setattr(LOA, "_fixed_integer_nlp", lambda self, x: (None, False))

    m = dm.Model("capped")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x * y >= 2)
    m.either_or([[x >= 3], [y >= 4]], name="d")
    m.minimize(x + y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = LOA(m, time_limit=60, subproblem_time_limit=1.0).run()
    assert 1.0 in caps and 4.0 in caps, caps  # the probe fired, then escalated
    assert r.status == "optimal", (r.status, caps)
    assert r.objective == pytest.approx(3 + 2 / 3, rel=1e-4)  # x = 3, y = 2/3
