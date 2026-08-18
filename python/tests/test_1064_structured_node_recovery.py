"""#1064: node-bound recovery must use POUNCE's structured convex engine.

``_pounce_recover_node_bound`` re-solves a stalled node relaxation. It routed
through ``qp_pounce.solve_qp``, which wraps ``pounce.Problem(problem_obj=
_QPCallbacks(...))`` -- the generic **callback TNLP** path. That path hides the
linear structure, so POUNCE's presolve cannot engage and its IPM runs ~100
iterations instead of ~20. ``_solve_node_lp_pounce`` was migrated off it for
exactly this reason (see its docstring); recovery was left behind.

Measured cost of being left behind (#1064 entry experiment 4 -- both arms run on
the SAME fixing at the SAME node, 4 comparisons per instance):

    squfl015-060 (n=915)   21.8x     verdicts agreed 4/4
    squfl025-040 (n=1025)  19.4x     verdicts agreed 4/4
    squfl020-150 (n=3020)   5.9x     callback returned NO answer on all 4
                                     (~32 s limit); structured settled every
                                     one in ~5.5 s

These tests pin the routing and the soundness gates, not the timing -- a wall
-clock assertion would be flaky and CLAUDE.md §9 would require a load gate and a
spread for it to mean anything.
"""

import numpy as np
import pytest
from discopt import solver as S
from discopt.solver_tuning import SolverTuning, reset_current, set_current

pounce = pytest.importorskip("pounce")


def _tiny_qp():
    """A small strictly-convex QP with both row types and a known optimum.

    min  1/2 (x0^2 + x1^2 + x2^2) - x0 - x1     s.t.  x0 + x1 + x2 = 1,
                                                      x0 - x1 <= 0.5,
                                                      0 <= x <= 1
    Strictly convex over a nonempty compact box, so the optimum is unique and
    both engines must agree on it.
    """
    Q = np.eye(3)
    c = np.array([-1.0, -1.0, 0.0])
    A_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([1.0])
    A_ub = np.array([[1.0, -1.0, 0.0]])
    b_ub = np.array([0.5])
    lb = np.zeros(3)
    ub = np.ones(3)
    return lb, ub, c, 0.0, A_ub, b_ub, A_eq, b_eq, Q


def _flag(on: bool):
    return set_current(SolverTuning(structured_node_recovery=on))


def test_flag_on_routes_through_structured_engine(monkeypatch):
    """Flag ON must call ``pounce.solve_qp`` and must NOT use the callback path."""
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q = _tiny_qp()

    structured_calls = {"n": 0}
    orig = pounce.solve_qp

    def counting(*a, **kw):
        structured_calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(pounce, "solve_qp", counting)

    callback_calls = {"n": 0}
    import discopt.solvers.qp_pounce as qpp

    orig_cb = qpp.solve_qp

    def counting_cb(*a, **kw):
        callback_calls["n"] += 1
        return orig_cb(*a, **kw)

    monkeypatch.setattr(qpp, "solve_qp", counting_cb)

    tok = _flag(True)
    try:
        rec = S._pounce_recover_node_bound(lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, 0.0, 30.0, Q=Q)
    finally:
        reset_current(tok)

    assert rec is not None, "structured recovery declined a feasible strictly-convex QP"
    assert rec[0] == "optimal", rec[0]
    assert structured_calls["n"] >= 1, "pounce.solve_qp was never called -- flag is a no-op"
    assert callback_calls["n"] == 0, "callback TNLP path ran even though the flag is ON"


def test_both_engines_agree_on_the_optimum():
    """Same problem, both engines: same verdict, same objective, same point.

    This is the anti-vacuity control (§6): if the structured arm answered a
    different question, the speedup would be worthless. Comparing the two arms
    directly is the only way to see that.
    """
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q = _tiny_qp()

    tok = _flag(True)
    try:
        on = S._pounce_recover_node_bound(lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, 0.0, 30.0, Q=Q)
    finally:
        reset_current(tok)

    tok = _flag(False)
    try:
        off = S._pounce_recover_node_bound(lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, 0.0, 30.0, Q=Q)
    finally:
        reset_current(tok)

    assert on is not None and off is not None
    assert on[0] == off[0] == "optimal"
    assert on[1] == pytest.approx(off[1], rel=1e-6), f"{on[1]} vs {off[1]}"
    np.testing.assert_allclose(on[2][:3], off[2][:3], atol=1e-5)


def test_infeasible_box_is_certified_by_both_engines():
    """An empty box must come back ``infeasible`` from either engine.

    Recovery's ``infeasible`` verdict PRUNES, so a structured path that reported
    infeasible where the callback path did not would be a soundness regression,
    not a speedup.
    """
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q = _tiny_qp()
    # x0+x1+x2 = 1 is unsatisfiable once every variable is pinned to 0.
    lb = np.zeros(3)
    ub = np.zeros(3)

    verdicts = []
    for on in (True, False):
        tok = _flag(on)
        try:
            rec = S._pounce_recover_node_bound(lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, 0.0, 30.0, Q=Q)
        finally:
            reset_current(tok)
        verdicts.append(None if rec is None else rec[0])

    assert verdicts[0] == "infeasible", f"structured arm gave {verdicts[0]}"
    assert verdicts[0] == verdicts[1], f"engines disagree on an empty box: {verdicts}"


@pytest.mark.parametrize(
    "bad_x, why",
    [
        (np.array([5.0, 0.0, 0.0]), "outside the variable box"),
        (np.array([0.9, 0.9, 0.9]), "violates the equality row"),
    ],
)
def test_structured_recovery_rejects_an_infeasible_iterate(monkeypatch, bad_x, why):
    """A drifted 'optimal' iterate must be rejected, never returned as a bound.

    The recovery result becomes a node bound or an incumbent, so accepting a
    point that violates its own box or rows would seed a false certificate
    (CLAUDE.md §1). Forcing POUNCE to hand back a bad point is the only way to
    exercise the gate.
    """
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q = _tiny_qp()

    class _Bogus:
        status = "optimal"
        x = bad_x
        obj = -1.0

    monkeypatch.setattr(pounce, "solve_qp", lambda *a, **kw: _Bogus())

    rec = S._structured_node_recovery(lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q=Q)
    assert rec is None, f"accepted a point that {why}: {bad_x}"


def test_lp_case_needs_no_quadratic_term():
    """``P=None`` must work: recovery serves the MILP path too, not just MIQP."""
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, _Q = _tiny_qp()
    rec = S._structured_node_recovery(lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q=None)
    assert rec is not None and rec[0] == "optimal"
    # min -x0 - x1 over the simplex with x0 - x1 <= 0.5 puts mass on x0,x1.
    assert rec[1] < 0.0
