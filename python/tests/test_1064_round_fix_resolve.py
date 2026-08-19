"""#1064: a fractional relaxation point must still be able to yield an incumbent.

``_pounce_snap_incumbent`` only purifies points already integral to within
``_SNAP_TOL`` = 1e-4. A search that never lands near-integral therefore never
produces an incumbent at all -- which is #1064's actual symptom, not a slow
re-solve: squfl020-150 and squfl025-040 each run a full 120 s budget with **zero**
snap re-solves and finish with no primal bound whatsoever.

``_pounce_round_incumbent`` rounds every integer coordinate to the nearest
integer inside its node box, fixes them, and asks for the continuous completion.
Measured end-to-end (interleaved A/B, 120 s, flags via the environment):

    squfl020-150   OFF time_limit obj=None  ->  ON feasible obj=4584.01
    squfl025-040   OFF time_limit obj=None  ->  ON feasible obj=1190.69
    squfl015-060   optimal in both arms (unchanged answer)

Both incumbents sit above their reference optimum and both bounds below it, so
nothing false was introduced.

These tests pin the rounding arithmetic and the retry, not the timing -- a
wall-clock assertion would need a load gate and a spread to mean anything (§9).
"""

import numpy as np
import pytest
from discopt import solver as S
from discopt.solver_tuning import SolverTuning

pytest.importorskip("pounce")


def _fix_or_free_qp():
    """min 1/2(x0^2 + x1^2)  s.t.  x0 + x1 == 1,  x0 in [0,1],  x1 in [0.6,1].

    x0 is the integer coordinate. Fixing x0 = 0 leaves x1 = 1, inside its box;
    fixing x0 = 1 leaves x1 = 0, **outside** it. So the rounding direction
    decides feasibility, which is what makes this a usable retry probe.
    """
    Q = np.eye(2)
    c = np.zeros(2)
    A_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([1.0])
    lb = np.array([0.0, 0.6])
    ub = np.array([1.0, 1.0])
    return lb, ub, c, 0.0, None, None, A_eq, b_eq, Q


def _call(x_relax, *, int_offsets=(0,), int_sizes=(1,)):
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q = _fix_or_free_qp()
    return S._pounce_round_incumbent(
        np.asarray(x_relax, dtype=np.float64),
        list(int_offsets),
        list(int_sizes),
        lb,
        ub,
        c,
        k,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
        0.0,
        30.0,
        Q=Q,
    )


def test_rounds_a_fractional_point_into_a_feasible_incumbent():
    """The whole point: a point the snap gate rejects still becomes an incumbent."""
    x = np.array([0.4, 0.6])
    # Precondition: the snap path must genuinely decline this point, or the test
    # would be exercising purification rather than rounding.
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q = _fix_or_free_qp()
    assert (
        S._pounce_snap_incumbent(x, [0], [1], lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, 0.0, 30.0, Q=Q)
        is None
    ), "the snap gate accepted this point, so it is not a rounding probe"

    inc = _call(x)
    assert inc is not None, "round-fix-resolve declined a point with a feasible rounding"
    obj, x_sol = inc
    assert x_sol[0] == pytest.approx(0.0, abs=1e-9), "x0 was not fixed to its rounded value"
    assert x_sol[1] == pytest.approx(1.0, abs=1e-6)
    assert obj == pytest.approx(0.5, rel=1e-6)


def test_retry_flips_the_most_fractional_coordinate_when_rounding_is_infeasible():
    """Round-to-nearest is genuinely infeasible here; the single flip must rescue it.

    x0 = 0.55 rounds to 1, which forces x1 = 0 outside its box. Without the retry
    this returns None and the instance keeps its "no incumbent" outcome.
    """
    x = np.array([0.55, 0.45])
    assert round(0.55) == 1, "the probe assumes round-half-even puts 0.55 at 1"

    inc = _call(x)
    assert inc is not None, "the retry did not fire -- an infeasible rounding ended the attempt"
    obj, x_sol = inc
    assert x_sol[0] == pytest.approx(0.0, abs=1e-9), "the flip did not reach the feasible side"
    assert obj == pytest.approx(0.5, rel=1e-6)


def test_retry_is_bounded():
    """The retry budget is one flip, not a dive -- an unbounded search here would
    spend the solve's whole budget on a single node."""
    assert S._ROUND_MAX_TRIES == 2


def test_no_integer_variables_declines():
    """Nothing to round: must return None rather than 'succeed' vacuously."""
    assert _call(np.array([0.4, 0.6]), int_offsets=(), int_sizes=()) is None


def test_rounding_stays_inside_the_node_box():
    """Round *then* clamp would leave a coordinate off an integer on a fractional box.

    ``_round_into_box`` must clamp to ``ceil(lo)``/``floor(hi)``: rounding 2.6 in
    the box [0.3, 1.8] has to give 1.0, not 1.8 (which is not an integer and
    would be fixed as a non-integral 'integer').
    """
    out = S._round_into_box(np.array([2.6]), np.array([0.3]), np.array([1.8]))
    assert out is not None
    assert out[0] == pytest.approx(1.0)
    assert float(out[0]) == float(int(out[0])), "clamped to a non-integer box edge"


def test_box_with_no_integer_declines():
    """[0.2, 0.9] contains no integer. Returning a value anyway would hand the
    re-solve an empty box and read as 'the rounding was infeasible'."""
    assert S._round_into_box(np.array([0.5]), np.array([0.2]), np.array([0.9])) is None


def test_flag_defaults_off_and_is_env_settable(monkeypatch):
    """§5: default OFF pending the panel, with a working opt-in."""
    monkeypatch.delenv("DISCOPT_ROUND_FIX_RESOLVE", raising=False)
    assert SolverTuning().round_fix_resolve is False
    monkeypatch.setenv("DISCOPT_ROUND_FIX_RESOLVE", "1")
    assert SolverTuning().round_fix_resolve is True
    monkeypatch.setenv("DISCOPT_ROUND_FIX_RESOLVE", "0")
    assert SolverTuning().round_fix_resolve is False


def _ufl_model(n_i=6, n_j=12):
    """Uncapacitated-facility-location shape -- the #1064 class.

    Binary ``y_i``, continuous ``x_ij``, VUB links ``x_ij <= y_i``, covering
    equalities ``sum_i x_ij == 1``, convex quadratic objective. This routes to
    ``_solve_miqp_bb`` and reaches the round gate, which is what makes the
    assertions below about the gate meaningful.
    """
    from discopt import Model

    m = Model("ufl")
    y = m.binary("y", shape=(n_i,))
    x = m.continuous("x", shape=(n_i, n_j), lb=0.0, ub=1.0)
    rng = np.random.default_rng(0)
    serve = rng.uniform(1.0, 9.0, size=(n_i, n_j))
    opens = rng.uniform(5.0, 15.0, size=n_i)
    for i in range(n_i):
        for j in range(n_j):
            m.subject_to(x[i, j] <= y[i])
    for j in range(n_j):
        m.subject_to(sum(x[i, j] for i in range(n_i)) == 1)
    m.minimize(
        sum(opens[i] * y[i] for i in range(n_i))
        + sum(serve[i, j] * x[i, j] * x[i, j] for i in range(n_i) for j in range(n_j))
    )
    return m


_STUB_SLEEP = 0.3
_STUB_TIME_LIMIT = 10.0


def _run_with_declining_stub(monkeypatch, frac):
    """Solve the UFL fixture with a stub that always declines and costs time.

    Standing in for the expensive ``_pounce_recover_node_bound`` re-solve keeps
    the bound under test the *budget* rather than POUNCE's convergence.
    """
    import time as _time

    seen = {"calls": 0, "limits": []}

    def _stub(*args, **kwargs):
        seen["calls"] += 1
        # Signature: (..., t_start, time_limit, Q=None) -- time_limit is arg 13.
        seen["limits"].append(float(args[12]))
        _time.sleep(_STUB_SLEEP)
        return None

    monkeypatch.setattr(S, "_pounce_round_incumbent", _stub)
    monkeypatch.setattr(S, "_ROUND_TIME_FRAC", frac)
    monkeypatch.setenv("DISCOPT_ROUND_FIX_RESOLVE", "1")

    t0 = _time.perf_counter()
    S.solve_model(_ufl_model(), time_limit=_STUB_TIME_LIMIT)
    seen["wall"] = _time.perf_counter() - t0
    # The probe must have fired, or every assertion downstream is vacuous.
    assert seen["calls"] > 0, (
        "round-fix-resolve never ran: this model no longer reaches the MIQP "
        "round gate, so the budget is untested -- fix the fixture, not the bound"
    )
    return seen


def test_round_fix_resolve_is_bounded_by_a_time_budget(monkeypatch):
    """The heuristic may not spend the solve.

    ``_ROUND_ATTEMPT_CAP`` caps the *number* of attempts, but each attempt is up
    to two ``_pounce_recover_node_bound`` calls -- a full POUNCE re-solve whose
    cost varies by orders of magnitude across instances -- so a count cap bounds
    nothing. Measured on slay05h at T=60 s before this budget existed: 31
    attempts, 0 hits, 66.5 s = 98.3% of the wall, turning a certified optimum
    (1335 nodes, 15.5 s) into a time_limit with no incumbent (63 nodes).
    """
    budget = S._ROUND_TIME_FRAC * _STUB_TIME_LIMIT
    seen = _run_with_declining_stub(monkeypatch, S._ROUND_TIME_FRAC)

    assert seen["calls"] < _ROUND_ATTEMPT_CAP_REF
    spent = seen["calls"] * _STUB_SLEEP
    assert spent <= budget + _STUB_SLEEP + 0.5, (
        f"spent {spent:.2f}s of a {budget:.2f}s budget over {seen['calls']} attempts"
    )
    # Each attempt is handed the budget's deadline, not the solve's: passing the
    # global limit lets a single attempt run all the way to the deadline.
    assert max(seen["limits"]) < _STUB_TIME_LIMIT
    assert seen["wall"] < _STUB_TIME_LIMIT + 5.0


def test_the_time_budget_is_what_bounds_the_spend(monkeypatch):
    """Neutralising only the budget restores the unbounded behaviour.

    Without this arm the test above would pass on any solve that happens to make
    few attempts, and would not show that the *budget* is the binding constraint
    rather than the attempt cap or the search finishing early.
    """
    default_frac = S._ROUND_TIME_FRAC
    budgeted = _run_with_declining_stub(monkeypatch, default_frac)
    unbudgeted = _run_with_declining_stub(monkeypatch, 1e6)

    assert unbudgeted["calls"] > 2 * budgeted["calls"], (
        f"budget frac made no difference: {budgeted['calls']} attempts budgeted "
        f"vs {unbudgeted['calls']} unbudgeted"
    )
    # Unbudgeted, the stub alone outruns the solve's own time limit.
    assert unbudgeted["calls"] * _STUB_SLEEP > _STUB_TIME_LIMIT * default_frac


_ROUND_ATTEMPT_CAP_REF = 64


def test_attempt_cap_is_still_a_backstop():
    assert S._ROUND_ATTEMPT_CAP == _ROUND_ATTEMPT_CAP_REF
    assert 0.0 < S._ROUND_TIME_FRAC < 1.0
