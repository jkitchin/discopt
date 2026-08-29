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

import time

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
    """The ladder is a few fixings, not a dive -- an unbounded search here would
    spend the solve's whole budget on a single node.

    Raised 2 -> 3 when the all-fractional-up rung was added (see the ladder tests
    below); the cap is what bounds the loop, so it is pinned rather than derived.
    """
    assert S._ROUND_MAX_TRIES == 3


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


def test_flag_defaults_on_and_the_opt_out_still_works(monkeypatch):
    """§5: default ON since the graduation panel, and ``=0`` must still opt out.

    The opt-out arm is the load-bearing half. §5 graduates a flag by flipping
    the default while *keeping the legacy path intact*, so a regression that
    silently made ``=0`` a no-op would remove the escape hatch the policy
    requires -- and would do it invisibly, since the ON arm would keep passing.
    """
    monkeypatch.delenv("DISCOPT_ROUND_FIX_RESOLVE", raising=False)
    assert SolverTuning().round_fix_resolve is True
    monkeypatch.setenv("DISCOPT_ROUND_FIX_RESOLVE", "1")
    assert SolverTuning().round_fix_resolve is True
    monkeypatch.setenv("DISCOPT_ROUND_FIX_RESOLVE", "0")
    assert SolverTuning().round_fix_resolve is False


def test_structured_node_recovery_defaults_on_and_the_opt_out_still_works(monkeypatch):
    """The other half of the graduated pair, same contract.

    ``round_fix_resolve`` alone cannot answer on the largest #1064 instance --
    all three ladder rungs return ``None`` on the callback path -- so the pair
    is what graduated, and both defaults have to hold for the measured result
    to be the one that ships.
    """
    monkeypatch.delenv("DISCOPT_STRUCTURED_NODE_RECOVERY", raising=False)
    assert SolverTuning().structured_node_recovery is True
    monkeypatch.setenv("DISCOPT_STRUCTURED_NODE_RECOVERY", "0")
    assert SolverTuning().structured_node_recovery is False


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


_ROUND_ATTEMPT_CAP_REF = 64
_STUB_COST = 0.3
_STUB_TIME_LIMIT = 10.0


# --- The spend rule: exercised directly, not raced to -------------------------
#
# These two tests used to run the UFL fixture below through two full solves and
# compare how many times each reached the round gate. That measures the machine,
# not the budget: the gate sits inside a wall-clock-bounded search, and ten
# identical runs on one box (2026-08-29) gave 0, 7 or 15 gate visits, 0/31/97
# nodes, and a status flipping between ``optimal`` and ``feasible``. On CI both
# arms clamped at 7 and the comparison failed on main while the budget was
# working correctly; on this developer box the fixture usually reached the gate
# zero times, so the invariant was not tested at all. The rule now lives in
# ``_RoundBudget``/``_round_fix_resolve_attempt`` and is asserted on directly --
# which also pins the two conjuncts (opt-out, no-incumbent) nothing tested
# before. ``test_the_miqp_search_wires_the_budget_in`` keeps the end-to-end
# claim that the solver actually uses the rule.


def _drive_gate(monkeypatch, budget, *, cost=_STUB_COST, enabled=True, has_incumbent=False):
    """Call the gate until it declines, with an attempt that costs ``cost``.

    The stub charges the budget itself rather than sleeping, so the rule is
    exercised at its own arithmetic instead of at the clock -- deterministic,
    and it does not spend ``cost`` seconds of the suite per attempt. The real
    elapsed time the gate also charges is microseconds, far below ``cost``.
    """
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q = _fix_or_free_qp()
    seen = {"calls": 0, "limits": []}

    def _stub(*args, **kwargs):
        seen["calls"] += 1
        # Signature: (..., t_start, time_limit, Q=None) -- time_limit is arg 13.
        seen["limits"].append(float(args[12]))
        budget.add_spend(cost)
        return None

    monkeypatch.setattr(S, "_pounce_round_incumbent", _stub)
    # The solver passes a perf_counter reading; 0.0 would make the elapsed
    # term the machine's whole uptime and clamp every deadline to the solve.
    t_start = time.perf_counter()
    for _ in range(_ROUND_ATTEMPT_CAP_REF + 5):
        before = seen["calls"]
        S._round_fix_resolve_attempt(
            np.array([0.4, 0.6]),
            [0],
            [1],
            lb,
            ub,
            c,
            k,
            A_ub,
            b_ub,
            A_eq,
            b_eq,
            t_start,
            _STUB_TIME_LIMIT,
            Q,
            budget,
            enabled=enabled,
            has_incumbent=has_incumbent,
        )
        if seen["calls"] == before:
            break  # the gate declined; the stub was not reached
    else:  # pragma: no cover - runaway guard
        pytest.fail("the gate never declined -- the rule bounds nothing")
    return seen


def test_the_budget_not_the_attempt_cap_is_what_bounds_the_spend(monkeypatch):
    """The binding constraint must be *time*, and neutralising it must show that.

    ``_ROUND_ATTEMPT_CAP`` caps the number of attempts, but each attempt is up
    to two ``_pounce_recover_node_bound`` calls -- a full POUNCE re-solve whose
    cost varies by orders of magnitude across instances -- so a count cap bounds
    nothing. Measured on slay05h at T=60 s before this budget existed: 31
    attempts, 0 hits, 66.5 s = 98.3 % of the wall, turning a certified optimum
    (1335 nodes, 15.5 s) into a time_limit with no incumbent (63 nodes).
    """
    budget = S._ROUND_TIME_FRAC * _STUB_TIME_LIMIT
    budgeted = _drive_gate(monkeypatch, S._RoundBudget(_STUB_TIME_LIMIT))
    # Stopped by the budget, well short of the cap.
    assert budgeted["calls"] < _ROUND_ATTEMPT_CAP_REF
    spent = budgeted["calls"] * _STUB_COST
    assert spent <= budget + _STUB_COST, (
        f"spent {spent:.2f}s of a {budget:.2f}s budget over {budgeted['calls']} attempts"
    )

    # Neutralise only the budget: now nothing but the backstop stops it.
    unbudgeted = _drive_gate(monkeypatch, S._RoundBudget(_STUB_TIME_LIMIT, frac=1e6))
    assert unbudgeted["calls"] == _ROUND_ATTEMPT_CAP_REF
    assert unbudgeted["calls"] > 2 * budgeted["calls"], (
        f"budget made no difference: {budgeted['calls']} attempts budgeted "
        f"vs {unbudgeted['calls']} unbudgeted"
    )
    # Unbudgeted, the attempts alone outrun the solve's own time limit.
    assert unbudgeted["calls"] * _STUB_COST > _STUB_TIME_LIMIT * S._ROUND_TIME_FRAC


def test_each_attempt_gets_the_budget_deadline_not_the_solve_deadline(monkeypatch):
    """Handing an attempt the global limit lets one attempt run to the deadline.

    ``_pounce_recover_node_bound`` derives its own limit as
    ``time_limit - (now - t_start)``, so the per-attempt deadline has to be the
    budget's remainder, and it has to shrink as the budget is spent.
    """
    seen = _drive_gate(monkeypatch, S._RoundBudget(_STUB_TIME_LIMIT))
    assert seen["limits"], "no attempt was made, so no deadline was handed out"
    assert max(seen["limits"]) < _STUB_TIME_LIMIT
    assert seen["limits"] == sorted(seen["limits"], reverse=True), (
        f"per-attempt deadline did not shrink with the budget: {seen['limits']}"
    )
    assert seen["limits"][0] == pytest.approx(S._ROUND_TIME_FRAC * _STUB_TIME_LIMIT, abs=1e-3)


def test_the_gate_is_spent_only_while_the_tree_has_no_incumbent(monkeypatch):
    """With a primal bound in hand a rounded completion can add nothing.

    ``_pounce_round_incumbent`` only ever supplies an upper bound, so spending
    the budget once the tree already has one is pure loss.
    """
    budget = S._RoundBudget(_STUB_TIME_LIMIT)
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q = _fix_or_free_qp()
    called = {"n": 0}

    def _stub(*args, **kwargs):
        called["n"] += 1
        return None

    monkeypatch.setattr(S, "_pounce_round_incumbent", _stub)
    for has_inc, enabled in ((True, True), (False, False)):
        assert (
            S._round_fix_resolve_attempt(
                np.array([0.4, 0.6]),
                [0],
                [1],
                lb,
                ub,
                c,
                k,
                A_ub,
                b_ub,
                A_eq,
                b_eq,
                time.perf_counter(),
                _STUB_TIME_LIMIT,
                Q,
                budget,
                enabled=enabled,
                has_incumbent=has_inc,
            )
            is None
        )
    assert called["n"] == 0, "the gate ran an attempt it should have declined"
    assert budget.attempts == 0 and budget.secs == 0.0
    # And the same budget still works, so the declines above were the gate's
    # doing rather than an exhausted budget.
    assert _drive_gate(monkeypatch, budget)["calls"] > 0


def test_a_raising_attempt_still_charges_the_budget(monkeypatch):
    """Otherwise a rounding that always throws buys unlimited retries."""
    budget = S._RoundBudget(_STUB_TIME_LIMIT)
    lb, ub, c, k, A_ub, b_ub, A_eq, b_eq, Q = _fix_or_free_qp()

    def _boom(*args, **kwargs):
        raise RuntimeError("rounding blew up")

    monkeypatch.setattr(S, "_pounce_round_incumbent", _boom)
    with pytest.raises(RuntimeError, match="blew up"):
        S._round_fix_resolve_attempt(
            np.array([0.4, 0.6]),
            [0],
            [1],
            lb,
            ub,
            c,
            k,
            A_ub,
            b_ub,
            A_eq,
            b_eq,
            time.perf_counter(),
            _STUB_TIME_LIMIT,
            Q,
            budget,
            enabled=True,
            has_incumbent=False,
        )
    assert budget.attempts == 1, "a raising attempt was not counted"
    assert budget.secs > 0.0, "a raising attempt was not charged any time"


def _ufl_model(n_i=6, n_j=12):
    """Uncapacitated-facility-location shape -- the #1064 class.

    Binary ``y_i``, continuous ``x_ij``, VUB links ``x_ij <= y_i``, covering
    equalities ``sum_i x_ij == 1``, convex quadratic objective. This routes to
    ``_solve_miqp_bb``, which is what makes the wiring test below meaningful.
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


def test_the_miqp_search_wires_the_budget_in():
    """The unit tests above are worthless if the search does not use the rule.

    ``_solve_miqp_bb`` is entered directly rather than through ``solve_model``:
    whether a given model *routes* to the convex-MIQP engine is itself not
    reproducible (measured 2026-08-29 -- five identical ``solve_model`` runs of
    the fixture below entered it zero times, other runs of the same build
    entered it and reached the round gate 7 or 15 times). Calling the engine
    makes the wiring claim deterministic; the ``_RoundBudget`` the engine builds
    is the one the gate consults, and it is built from the solve's own limit.
    """
    built, asked = [], []
    real = S._RoundBudget

    class _Recording(real):
        def __init__(self, time_limit, frac=None):
            built.append(float(time_limit))
            super().__init__(time_limit, frac)

        def may_attempt(self):
            asked.append(True)
            return super().may_attempt()

    saved, S._RoundBudget = S._RoundBudget, _Recording
    try:
        S._solve_miqp_bb(
            _ufl_model(),
            _STUB_TIME_LIMIT,
            1e-4,
            1,
            "best_first",
            10_000,
            time.perf_counter(),
            prefer_pounce=True,
        )
    finally:
        S._RoundBudget = saved
    assert built, "the MIQP search never built a round budget -- the rule is unwired"
    assert built[0] == pytest.approx(_STUB_TIME_LIMIT)
    # Building it is not using it: the gate has to actually consult the rule.
    assert asked, "the round gate never consulted the budget -- the rule is bypassed"
    assert real is saved, "the recording subclass leaked out of the test"


# --- The candidate ladder (#1064: switch structure) ---------------------------
#
# Round-to-nearest fails *systematically* on switch structure. A row
# ``x - u*y <= 0`` with binary ``y`` drives ``y`` to ``max_j x_j`` in the
# relaxation, so every switch comes back small and fractional; nearest sends the
# whole vector to zero, which forces ``x = 0`` and contradicts any covering row.
# Measured on all three squfl instances of #1064: every binary fractional in
# [0.010, 0.263], nearest -> 0 open -> primal_infeasible; rounding up -> feasible
# on all three (squfl015-060 obj 1025.73, squfl025-040 obj 1139.53).


def test_ladder_offers_an_all_up_rung_for_a_fractional_switch_vector():
    """The rung that rescues squfl must exist: every fractional switch -> 1."""
    vals = np.array([0.0431, 0.2629, 0.0103, 0.1698])
    lo = np.zeros(4)
    hi = np.ones(4)
    cands = list(S._round_candidates(vals, lo, hi))
    assert cands, "ladder produced nothing"
    # Round-to-nearest opens none of them -- the failure mode under test.
    assert np.array_equal(cands[0], np.zeros(4))
    # Some rung opens all of them. Without it squfl has no feasible fixing.
    assert any(np.array_equal(c, np.ones(4)) for c in cands), (
        f"no all-up rung in ladder: {[c.tolist() for c in cands]}"
    )


def test_ladder_respects_the_node_box():
    """The all-up rung must not leave the node box."""
    vals = np.array([0.4, 0.6])
    lo = np.zeros(2)
    hi = np.array([0.0, 1.0])  # first coordinate is fixed to 0 by the box
    for c in S._round_candidates(vals, lo, hi):
        assert c[0] == 0.0, f"candidate left the box: {c}"
        assert lo[1] <= c[1] <= hi[1]


def test_ladder_suppresses_duplicate_candidates():
    """A duplicate rung would burn an attempt (and a slice of the budget)."""
    # Already integral: nearest, ceil and any flip all coincide.
    vals = np.array([1.0, 0.0])
    cands = list(S._round_candidates(vals, np.zeros(2), np.ones(2)))
    seen = {tuple(c.tolist()) for c in cands}
    assert len(seen) == len(cands), f"duplicates in ladder: {cands}"


def test_ladder_is_bounded_by_max_tries():
    """An unbounded ladder would be a dive, which is a different mechanism."""
    rng = np.random.default_rng(0)
    vals = rng.uniform(0.05, 0.95, size=40)
    cands = list(S._round_candidates(vals, np.zeros(40), np.ones(40)))
    # The generator may offer a few more than the cap; the consumer stops at
    # _ROUND_MAX_TRIES, so what matters is that it is finite and small.
    assert 0 < len(cands) <= 2 + S._ROUND_MAX_TRIES


def test_all_up_rung_produces_an_incumbent_where_nearest_is_infeasible():
    """End-to-end on the UFL shape: nearest is infeasible, the ladder recovers.

    This is #1064's floor -- a feasible incumbent -- on the structure that
    produced the bug, without naming an instance.
    """
    import time as _time

    m, data = _ufl_switch_case()
    res = S._pounce_round_incumbent(
        data["x_relax"],
        data["int_offsets"],
        data["int_sizes"],
        data["lb"],
        data["ub"],
        data["c"],
        0.0,
        data["A_ub"],
        data["b_ub"],
        data["A_eq"],
        data["b_eq"],
        _time.perf_counter(),
        60.0,
        Q=data["Q"],
    )
    assert res is not None, "ladder failed to produce any incumbent"
    obj, x = res
    assert np.isfinite(obj)
    # The returned point must actually satisfy the rows it was fixed against.
    if data["A_ub"] is not None and len(data["b_ub"]):
        assert np.all(data["A_ub"] @ x <= data["b_ub"] + 1e-6)
    if data["A_eq"] is not None and len(data["b_eq"]):
        assert np.all(np.abs(data["A_eq"] @ x - data["b_eq"]) <= 1e-6)
    # And the switches must be integral.
    idx = [
        j
        for off, sz in zip(data["int_offsets"], data["int_sizes"])
        for j in range(off, off + int(sz))
    ]
    assert np.all(np.abs(x[idx] - np.round(x[idx])) <= 1e-6)


def _ufl_switch_case(n_i=4, n_j=8):
    """A minimal uncapacitated-facility-location shape in raw matrix form.

    Variables: x_ij (n_i*n_j continuous), then y_i (n_i binary).
    Rows: x_ij - y_i <= 0 (VUB switches), sum_i x_ij == 1 (covering).
    Objective: separable convex quadratic in x plus a linear facility cost.
    The relaxation optimum spreads x evenly, so every y_i is small-fractional --
    the exact configuration where round-to-nearest is infeasible.
    """
    n = n_i * n_j + n_i

    def xi(i, j):
        return i * n_j + j

    def yi(i):
        return n_i * n_j + i

    A_ub = np.zeros((n_i * n_j, n))
    b_ub = np.zeros(n_i * n_j)
    for i in range(n_i):
        for j in range(n_j):
            A_ub[xi(i, j), xi(i, j)] = 1.0
            A_ub[xi(i, j), yi(i)] = -1.0
    A_eq = np.zeros((n_j, n))
    b_eq = np.ones(n_j)
    for j in range(n_j):
        for i in range(n_i):
            A_eq[j, xi(i, j)] = 1.0
    Q = np.zeros((n, n))
    for i in range(n_i):
        for j in range(n_j):
            Q[xi(i, j), xi(i, j)] = 2.0
    c = np.zeros(n)
    for i in range(n_i):
        c[yi(i)] = 1.0
    lb = np.zeros(n)
    ub = np.ones(n)
    # Relaxation point: x spread evenly across facilities, y_i = max_j x_ij.
    x_relax = np.zeros(n)
    for i in range(n_i):
        for j in range(n_j):
            x_relax[xi(i, j)] = 1.0 / n_i
    for i in range(n_i):
        x_relax[yi(i)] = 1.0 / n_i
    return None, {
        "x_relax": x_relax,
        "int_offsets": [yi(i) for i in range(n_i)],
        "int_sizes": [1] * n_i,
        "lb": lb,
        "ub": ub,
        "c": c,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "Q": Q,
    }
