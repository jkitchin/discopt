"""Tests for the NLP-BB root cutting-plane stage (#781, DISCOPT_NLPBB_ROOT_CUTS).

Soundness gates:
  * the GMI separator is validated by EXACT enumeration: on seeded random
    MILPs, every emitted cut must be violated at the LP vertex and satisfied
    by every integer assignment's full continuous completion (LP certificate
    per integer corner — no sampling gap);
  * ``DISCOPT_NLPBB_ROOT_CUTS=0`` is inert (no constraints added, no behavior
    change) -- the legacy path §5 requires graduation to leave intact;
  * flag ON preserves the optimum and never reports an unsound bound
    (min: bound <= opt; max: bound >= opt), on both senses;
  * the slow named regression runs the real convex-synthesis instance.
"""

from __future__ import annotations

import itertools
import os
import types
from pathlib import Path

import numpy as np
import pytest
from discopt import Model
from discopt.solvers._root_cuts import (
    _solve_lp,
    generate_root_cuts,
    nlpbb_root_cuts_enabled,
    separate_gmi,
)

pytest.importorskip("highspy")

BENCH_NL = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))
OPT_RSYN0805M = 1296.120603  # minlplib.solu (maximize)


def _flag(monkeypatch, on: bool) -> None:
    # The flag graduated default-ON (2026-08-20), so "off" has to be the explicit
    # opt-out value -- deleting the variable now selects ON and would have turned
    # every flag-OFF arm below into a second flag-ON arm silently.
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS", "1" if on else "0")


# ── GMI separator: exact validity by enumeration ─────────────────────────────


def test_gmi_cuts_valid_by_exact_enumeration():
    """Every GMI cut must keep every integer-feasible point (LP certificate per
    integer corner). A single positive violation is an unsound cut."""
    from scipy.optimize import linprog

    rng = np.random.default_rng(7)
    n_bad = 0
    n_cuts = 0
    for _trial in range(25):
        n_int = int(rng.integers(2, 5))
        n_cont = int(rng.integers(1, 4))
        n = n_int + n_cont
        m = int(rng.integers(2, 6))
        a_mat = np.round(rng.normal(0, 2, size=(m, n)), 1)
        lb = np.zeros(n)
        ub = np.concatenate([rng.integers(1, 6, n_int).astype(float), rng.uniform(1, 8, n_cont)])
        b = a_mat @ ((lb + ub) / 2) + rng.uniform(0.5, 3.0, m)
        c = np.round(rng.normal(0, 1, n), 2)
        is_int = np.array([True] * n_int + [False] * n_cont)

        root = types.SimpleNamespace(
            n=n,
            lb=lb,
            ub=ub,
            is_int=is_int,
            A_le=a_mat,
            b_le=b,
            A_eq=np.zeros((0, n)),
            b_eq=np.zeros(0),
            c=c,
            sense_max=True,
        )
        _obj, x, _duals, h = _solve_lp(root, [], [])
        if x is None:
            continue
        cuts = separate_gmi(root, h, x, a_mat, b, max_cuts=16)
        n_cuts += len(cuts)
        for alpha, rhs in cuts:
            assert alpha @ x - rhs > 1e-7, "cut not violated at the LP vertex"
            for combo in itertools.product(*[range(int(ub[j]) + 1) for j in range(n_int)]):
                bounds = [(float(v), float(v)) for v in combo] + [
                    (lb[j], ub[j]) for j in range(n_int, n)
                ]
                res = linprog(-alpha, A_ub=a_mat, b_ub=b, bounds=bounds, method="highs")
                if not res.success:
                    continue
                if float(alpha @ res.x - rhs) > 1e-7:
                    n_bad += 1
    assert n_cuts > 0, "enumeration produced no cuts — test lost its teeth"
    assert n_bad == 0, f"{n_bad} UNSOUND GMI cuts (integer-feasible points removed)"


# ── synthetic convex MINLP fixtures (both senses) ────────────────────────────


def _build_convex_minlp(sense: str) -> Model:
    """Fixed-charge network + one convex quadratic row; linear objective.

    Routes to NLP-BB when solved with ``nlp_bb=True`` (convex, integer vars).
    """
    m = Model(f"rc_{sense}")
    f0 = m.continuous("f0", lb=0.0, ub=10.0)
    f1 = m.continuous("f1", lb=0.0, ub=10.0)
    y0 = m.binary("y0")
    y1 = m.binary("y1")
    m.subject_to(f0 - 8.0 * y0 <= 0.0)
    m.subject_to(f1 - 8.0 * y1 <= 0.0)
    m.subject_to(f0 + f1 >= 3.0)
    m.subject_to(f0 * f0 + f1 * f1 <= 16.0)  # convex quadratic
    expr = f0 + f1 - 2.5 * y0 - 2.5 * y1
    if sense == "max":
        m.maximize(expr)
    else:
        m.minimize(f0 + 2.0 * f1 + 2.5 * y0 + 2.5 * y1)
    return m


@pytest.mark.parametrize("sense", ["max", "min"])
def test_flag_off_inert(monkeypatch, sense):
    _flag(monkeypatch, False)
    assert nlpbb_root_cuts_enabled() is False
    m = _build_convex_minlp(sense)
    n_before = len(m._constraints)
    m.solve(time_limit=30, nlp_bb=True)
    assert len(m._constraints) == n_before, "flag OFF must add no constraints"


@pytest.mark.parametrize("sense", ["max", "min"])
def test_flag_on_optimum_and_bound_sound(monkeypatch, sense):
    _flag(monkeypatch, False)
    base = _build_convex_minlp(sense).solve(time_limit=30, nlp_bb=True)
    assert base.objective is not None

    _flag(monkeypatch, True)
    m = _build_convex_minlp(sense)
    r = m.solve(time_limit=30, nlp_bb=True)
    assert r.objective is not None
    assert not getattr(r, "incumbent_verification_failed", False)
    # optimum preserved (the cuts removed no integer-feasible point)
    assert r.objective == pytest.approx(base.objective, abs=1e-4, rel=1e-4)
    # reported dual bound is sound w.r.t. the known optimum (= base objective)
    if r.bound is not None:
        if sense == "max":
            assert r.bound >= base.objective - 1e-4
        else:
            assert r.bound <= base.objective + 1e-4


def test_generate_root_cuts_direct_sound(monkeypatch):
    """Direct call: LP bound must be a valid dual bound (>= opt for max)."""
    from discopt._relax.nlp_evaluator import NLPEvaluator

    _flag(monkeypatch, True)
    m = _build_convex_minlp("max")
    opt = m.solve(time_limit=30, nlp_bb=True).objective
    assert opt is not None

    m2 = _build_convex_minlp("max")
    ev = NLPEvaluator(m2)
    lb = np.array([0.0, 0.0, 0.0, 0.0])
    ub = np.array([10.0, 10.0, 1.0, 1.0])
    is_int = np.array([False, False, True, True])
    res = generate_root_cuts(m2, ev, lb, ub, is_int, is_int.copy())
    assert res.lp_bound is not None
    assert res.lp_bound >= opt - 1e-6, (
        f"root LP bound {res.lp_bound} below the optimum {opt} — unsound"
    )
    # every returned cut keeps the known optimum's integer corners: check the
    # cuts at the optimal incumbent of the flag-off solve
    r = _build_convex_minlp("max").solve(time_limit=30, nlp_bb=True)
    x_opt = np.array([float(np.atleast_1d(r.x[nm])[0]) for nm in ("f0", "f1", "y0", "y1")])
    for alpha, rhs in res.cuts:
        assert float(alpha @ x_opt) <= rhs + 1e-6, "cut removes the optimal solution"


@pytest.mark.smoke
def test_flag_is_default_on_with_an_opt_out():
    """§5: the root-cut stage graduated default-ON, and the opt-out still works.

    It shipped default-OFF pending its differential panel (#781). That panel was
    re-run on 2026-08-20 -- after #1082 unblocked the sub-NLP primal heuristic on
    convex models and #1098 made ``gap_certified`` mean the same thing on both
    solve routes -- over 153 in-repo corpus instances at 60 s each, and passed
    both bars: cert-clean (0 bounds above a reference optimum, 0 certification
    regressions, proven-optimal 69 in both arms) and net-positive (dual bound
    tighter on 22 / looser on 9, primal shortfall better on 6 / worse on 1, total
    nodes -13.3%, wall +0.5%). Recorded in ``docs/dev/performance-plan.md`` §21.

    §5 requires the ``=0`` opt-out and the legacy path to survive graduation, so
    both halves are pinned here, not just the new default.
    """
    saved = os.environ.pop("DISCOPT_NLPBB_ROOT_CUTS", None)
    try:
        assert nlpbb_root_cuts_enabled() is True, "graduated: unset must mean ON"
        for off in ("0", "off", "false", "no", "OFF", " no "):
            os.environ["DISCOPT_NLPBB_ROOT_CUTS"] = off
            assert nlpbb_root_cuts_enabled() is False, f"{off!r} must opt out"
        # Empty is ON, not OFF, and that is the point: ``export X="$UNSET"`` exports
        # an empty string, and a graduated default-ON path must not be switched off
        # by an accident of shell quoting while reading in every log as "not set"
        # (#993, same lesson on DISCOPT_GDP_CONFIG_PRIMAL).
        for on in ("1", "true", "yes", "on", "", "  "):
            os.environ["DISCOPT_NLPBB_ROOT_CUTS"] = on
            assert nlpbb_root_cuts_enabled() is True, f"{on!r} must leave it ON"
    finally:
        os.environ.pop("DISCOPT_NLPBB_ROOT_CUTS", None)
        if saved is not None:
            os.environ["DISCOPT_NLPBB_ROOT_CUTS"] = saved


# ── the real class (benchmark corpus; slow) ──────────────────────────────────


@pytest.mark.slow
def test_rsyn0805m_flag_on_sound_and_tighter(monkeypatch):
    nl = BENCH_NL / "rsyn0805m.nl"
    if not nl.exists():
        pytest.skip("benchmark corpus not available")
    import discopt.modeling as dm

    _flag(monkeypatch, True)
    r = dm.from_nl(str(nl)).solve(time_limit=30)
    assert not getattr(r, "incumbent_verification_failed", False)
    # soundness (maximize): incumbent never super-optimal, bound never below opt
    if r.objective is not None:
        assert r.objective <= OPT_RSYN0805M + 1e-3
    assert r.bound is not None
    assert r.bound >= OPT_RSYN0805M - 1e-3
    # the composed bound is at most the root-cut LP bound (~1577.3 measured);
    # flag-off tree bound at this budget is ~1768 — require a real improvement
    assert r.bound <= 1650.0, f"root-cut bound composition ineffective: {r.bound}"


# ── stall termination (#1062): the loop must not spend its budget for nothing ─


def test_stalling_loop_exits_after_stall_rounds(monkeypatch):
    """A loop that keeps FINDING cuts but never MOVES the bound must give up.

    This is the #1062 regression. `chosen` cannot detect the condition — measured
    on clay0205hfsg the loop chose cuts in 16 of 16 rounds while the root LP
    bound sat at 0.0 the whole time, so it ran to budget exhaustion (7.4 s, a
    quarter of the solve's entire 30 s limit) and the end-of-loop quality gate
    then discarded every cut it had found. The cuts were never the problem — a
    valid cut cannot loosen a valid bound — the wasted wall was.

    Stubbed rather than corpus-driven so the condition is exact: the LP bound is
    pinned to a constant and selection always yields a cut, which is precisely
    "productive by `chosen`, stalled by the bound".
    """
    import discopt.solvers._root_cuts as rc
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = _build_convex_minlp("max")
    ev = NLPEvaluator(m)
    lb = np.array([0.0, 0.0, 0.0, 0.0])
    ub = np.array([10.0, 10.0, 1.0, 1.0])
    is_int = np.array([False, False, True, True])

    x_fixed = np.array([1.0, 1.0, 0.5, 0.5])
    calls = {"lp": 0, "sel": 0}

    def _frozen_lp(root, cuts_a, cuts_b, time_limit=None):
        calls["lp"] += 1
        return 5.0, x_fixed.copy(), None, None  # bound never moves

    def _always_a_cut(candidates, x, **kw):
        calls["sel"] += 1
        a = np.zeros(len(x_fixed))
        a[0] = 1.0
        return [(a, 99.0)]

    monkeypatch.setattr(rc, "_solve_lp", _frozen_lp)
    monkeypatch.setattr(rc, "_select_cuts", _always_a_cut)

    res = rc.generate_root_cuts(m, ev, lb, ub, is_int, is_int.copy(), time_budget_s=30.0)

    assert calls["lp"] > 0 and calls["sel"] > 0, "stubs never ran — probe measured nothing"
    assert res.stop_reason == "stall", f"stopped on {res.stop_reason!r}, not the stall guard"
    assert res.improving_rounds == 0
    assert res.rounds_run == rc.STALL_ROUNDS, (
        f"ran {res.rounds_run} rounds on a frozen bound; the guard should stop it "
        f"at {rc.STALL_ROUNDS}. Before #1062 this ran to ROUNDS or to the budget."
    )
    # ...and it did not burn the budget it was given.
    assert res.cuts == [], "a stalled loop must still hand back nothing"


def test_loop_never_runs_more_than_stall_rounds_past_its_last_gain(monkeypatch):
    """General invariant, asserted on the real separators rather than stubs.

    Whatever the instance, once the bound stops moving the loop is allowed at
    most STALL_ROUNDS more rounds. This is what bounds the stage's wasted wall
    on every instance, not just the ones in the probe set.
    """
    import discopt.solvers._root_cuts as rc
    from discopt._relax.nlp_evaluator import NLPEvaluator

    checked = 0
    for sense in ("max", "min"):
        m = _build_convex_minlp(sense)
        ev = NLPEvaluator(m)
        res = rc.generate_root_cuts(
            m,
            ev,
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 1.0, 1.0]),
            np.array([False, False, True, True]),
            np.array([False, False, True, True]),
            time_budget_s=30.0,
        )
        if res.stop_reason == "stall":
            assert res.rounds_run <= res.improving_rounds + rc.STALL_ROUNDS, (
                f"{sense}: {res.rounds_run} rounds but only {res.improving_rounds} "
                f"improving — more than {rc.STALL_ROUNDS} rounds were wasted"
            )
        # the trace always carries the baseline plus one entry per completed round
        assert len(res.bound_trace) <= res.rounds_run + 1
        checked += 1
    assert checked == 2, "invariant probe ran on nothing"


@pytest.mark.slow
def test_clay0205hfsg_stage_no_longer_burns_the_time_limit(monkeypatch):
    """The measured #1062 regression, on the instance that showed it.

    Before the stall guard: 16 rounds, 7.4 s against a 6.0 s budget — 25.8% of a
    30 s solve's whole limit — and 0 cuts kept. The budget is not the fix, since
    it was already being overrun; giving up is.
    """
    import time

    import discopt.modeling as dm
    import discopt.solvers._root_cuts as rc
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt.modeling.core import VarType

    nl = BENCH_NL / "clay0205hfsg.nl"
    if not nl.exists():
        pytest.skip("benchmark corpus not available")
    m = dm.from_nl(str(nl))
    lb = np.array([float(v.lb) for v in m._variables])
    ub = np.array([float(v.ub) for v in m._variables])
    is_int = np.array([v.var_type in (VarType.INTEGER, VarType.BINARY) for v in m._variables])

    budget = 6.0  # what solver.py hands the stage at time_limit=30
    t0 = time.perf_counter()
    res = rc.generate_root_cuts(
        m,
        NLPEvaluator(m),
        lb,
        ub,
        is_int,
        is_int & (lb == 0.0) & (ub == 1.0),
        time_budget_s=budget,
    )
    wall = time.perf_counter() - t0

    assert res.improving_rounds == 0, "instance changed; it used to never move the bound"
    assert res.stop_reason == "stall"
    assert res.rounds_run <= rc.STALL_ROUNDS
    assert wall < 0.25 * budget, f"stage still burned {wall:.2f}s of its {budget}s budget"


# ── vector variable blocks: the stage must not silently switch itself off ────


def _build_convex_minlp_vector() -> Model:
    """The same fixed-charge model as ``_build_convex_minlp('min')``, written
    with *vector* variable blocks instead of four scalars.

    Identical feasible set and objective, so the two forms must get the same
    optimum -- and, once flat columns are unravelled, the same root-cut stage.
    """
    m = Model("rc_vec")
    f = m.continuous("f", shape=2, lb=0.0, ub=10.0)
    y = m.binary("y", shape=2)
    m.subject_to(f[0] - 8.0 * y[0] <= 0.0)
    m.subject_to(f[1] - 8.0 * y[1] <= 0.0)
    m.subject_to(f[0] + f[1] >= 3.0)
    m.subject_to(f[0] * f[0] + f[1] * f[1] <= 16.0)
    m.minimize(f[0] + 2.0 * f[1] + 2.5 * y[0] + 2.5 * y[1])
    return m


def test_flat_column_terms_covers_blocks_in_column_order():
    """One term per flat column, blocks unravelled in C order.

    The flat layout is ``concatenate([asarray(x[v.name]).ravel() for v in
    model._variables])``, so a ``(2, 3)`` block owns six consecutive columns
    row-major. Asserting the *count* alone would pass for a wrong order, so
    each term is checked to name its own block.
    """
    from discopt.solvers._root_cuts import flat_column_terms

    m = Model("flatmap")
    a = m.continuous("a", lb=0.0, ub=1.0)
    m.continuous("b", shape=(2, 3), lb=0.0, ub=1.0)
    m.binary("c", shape=4)
    m.minimize(a)

    cols = flat_column_terms(m)
    assert len(cols) == 1 + 6 + 4, "one term per flat column"
    assert cols[0] is a, "scalar blocks are passed through unwrapped"
    checks = 0
    for j, expected in (
        [(0, "a")] + [(1 + k, "b") for k in range(6)] + [(7 + k, "c") for k in range(4)]
    ):
        term = cols[j]
        base = getattr(term, "base", term)
        assert getattr(base, "name", None) == expected, (
            f"flat column {j} should belong to block {expected!r}"
        )
        checks += 1
    assert checks == 11, "CHECKS_EXECUTED must cover every flat column"


def test_root_cuts_fire_on_vector_variable_blocks(monkeypatch):
    """Regression: the stage used to be gated on ``all(size == 1)``.

    That gate is a *modeling-style* restriction, not a problem-class one -- it
    silently disabled every root cut for array-API models while reporting a
    normal solve (CLAUDE.md §2, §6). Fails before the unravel fix (0 cuts
    added), passes after.
    """
    _flag(monkeypatch, False)
    base = _build_convex_minlp_vector().solve(time_limit=30, nlp_bb=True)
    assert base.objective is not None
    n_before = len(_build_convex_minlp_vector()._constraints)

    _flag(monkeypatch, True)
    m = _build_convex_minlp_vector()
    r = m.solve(time_limit=30, nlp_bb=True)
    added = len(m._constraints) - n_before
    assert added > 0, "root-cut stage must reach models built from vector blocks"
    # ...and the cuts must be sound: same optimum, dual bound below it (min).
    assert r.objective == pytest.approx(base.objective, abs=1e-4, rel=1e-4)
    if r.bound is not None:
        assert r.bound <= base.objective + 1e-4


def test_vector_and_scalar_forms_agree(monkeypatch):
    """The two spellings of one model must reach the same optimum with cuts on."""
    _flag(monkeypatch, True)
    scalar = _build_convex_minlp("min").solve(time_limit=30, nlp_bb=True)
    vector = _build_convex_minlp_vector().solve(time_limit=30, nlp_bb=True)
    assert scalar.objective is not None and vector.objective is not None
    assert scalar.objective == pytest.approx(vector.objective, abs=1e-4, rel=1e-4)


# ── #1066: the stage budget must bound the LPs, not only the round boundaries ──


def _deadline_flag(monkeypatch, on: bool) -> None:
    monkeypatch.setenv("DISCOPT_ROOT_CUT_DEADLINE", "1" if on else "0")


def _fake_clock(monkeypatch):
    """A clock only the stub advances, so nothing here depends on machine load.

    ``generate_root_cuts`` does ``import time as _time`` at call time, so
    patching the module attribute reaches it. Returns the mutable holder.
    """
    import time as _t

    now = {"t": 1000.0}
    monkeypatch.setattr(_t, "perf_counter", lambda: now["t"])
    return now


def _charging_lp_stub(monkeypatch, now, cost):
    """Stub ``_solve_lp`` that charges ``cost`` seconds per call to the fake clock.

    Records the ``time_limit`` each call was handed. Returns a frozen bound so
    the round loop's own stall guard is the only other thing that can stop it.
    """
    import discopt.solvers._root_cuts as rc

    seen = {"limits": [], "calls": 0}
    x_fixed = np.array([1.0, 1.0, 0.5, 0.5])

    def _stub(root, cuts_a, cuts_b, time_limit=None):
        seen["calls"] += 1
        seen["limits"].append(time_limit)
        now["t"] += cost
        return 5.0, x_fixed.copy(), None, None

    def _always_a_cut(candidates, x, **kw):
        a = np.zeros(len(x_fixed))
        a[0] = 1.0
        return [(a, 99.0)]

    monkeypatch.setattr(rc, "_solve_lp", _stub)
    monkeypatch.setattr(rc, "_select_cuts", _always_a_cut)
    return seen


def _run_stage(time_budget_s):
    import discopt.solvers._root_cuts as rc
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = _build_convex_minlp("max")
    return rc.generate_root_cuts(
        m,
        NLPEvaluator(m),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 1.0, 1.0]),
        np.array([False, False, True, True]),
        np.array([False, False, True, True]),
        time_budget_s=time_budget_s,
    )


def test_the_budget_covers_the_oa_prologue_not_only_the_round_loop(monkeypatch):
    """The #1066 defect, stated exactly.

    ``generate_root_cuts``' docstring promises ``time_budget_s`` bounds the
    stage's wall time. It did not: the clock was started AFTER the initial
    ``oa_converge()``, so the prologue -- up to ``OA_MAX_ITERS`` unbounded LPs --
    ran entirely outside it. Measured on rsyn0830m at default settings: one
    ``_solve_lp`` call burned 81.3 s of a 150 s solve against a 10 s budget,
    which is the whole reason that instance misses the 60 s default.

    Here the very first LP costs more than the entire budget. With the deadline
    on, the stage must notice and stop; with it off it does not, which is the
    bug this reproduces.
    """
    now = _fake_clock(monkeypatch)
    _deadline_flag(monkeypatch, True)
    seen_on = _charging_lp_stub(monkeypatch, now, cost=25.0)
    res_on = _run_stage(time_budget_s=10.0)

    now["t"] = 1000.0
    _deadline_flag(monkeypatch, False)
    seen_off = _charging_lp_stub(monkeypatch, now, cost=25.0)
    _run_stage(time_budget_s=10.0)

    assert seen_on["calls"] > 0 and seen_off["calls"] > 0, (
        "stubs never ran — probe measured nothing"
    )
    assert seen_on["calls"] == 1, (
        f"the deadline arm spent {seen_on['calls']} LPs on a budget the first one "
        f"had already blown; the stage clock must cover the prologue"
    )
    assert res_on.stop_reason == "budget", f"stopped on {res_on.stop_reason!r}, not the budget"
    assert seen_off["calls"] > seen_on["calls"], (
        f"legacy arm made {seen_off['calls']} calls, deadline arm {seen_on['calls']} — "
        f"the flag made no difference, so this test is not measuring the fix"
    )


def test_each_lp_is_handed_what_is_left_of_the_stage_budget(monkeypatch):
    """Every LP gets a deadline, and it shrinks as the budget is spent.

    An unbounded LP defeats a between-rounds budget check no matter how tight
    that check is: the overrun happens inside one call. So the budget has to
    reach the LP itself.
    """
    now = _fake_clock(monkeypatch)
    _deadline_flag(monkeypatch, True)
    seen = _charging_lp_stub(monkeypatch, now, cost=3.0)
    _run_stage(time_budget_s=10.0)

    limits = seen["limits"]
    assert len(limits) >= 2, f"only {len(limits)} LP(s) ran; nothing to compare"
    assert all(v is not None for v in limits), "an LP ran with no deadline"
    assert all(v <= 10.0 + 1e-9 for v in limits), f"a limit exceeded the budget: {limits}"
    assert limits == sorted(limits, reverse=True), f"limits not non-increasing: {limits}"
    assert limits[0] == pytest.approx(10.0), f"first LP got {limits[0]}, not the full budget"
    # Total charged cannot exceed the budget by more than the call in flight.
    assert len(limits) * 3.0 <= 10.0 + 3.0, f"{len(limits)} LPs x 3.0 s overran a 10 s budget"


def test_the_deadline_flag_is_default_off_with_an_opt_in(monkeypatch):
    """CLAUDE.md §5 regime 2: bound-changing, so it ships default-off."""
    import discopt.solvers._root_cuts as rc

    monkeypatch.delenv("DISCOPT_ROOT_CUT_DEADLINE", raising=False)
    assert rc._deadline_enabled() is False
    for on in ("1", "true", "on", "yes"):
        monkeypatch.setenv("DISCOPT_ROOT_CUT_DEADLINE", on)
        assert rc._deadline_enabled() is True, f"{on!r} did not switch it on"
    for off in ("0", "false", "off", "no"):
        monkeypatch.setenv("DISCOPT_ROOT_CUT_DEADLINE", off)
        assert rc._deadline_enabled() is False, f"{off!r} did not switch it off"


def test_the_legacy_lp_call_is_unchanged_when_the_flag_is_off(monkeypatch):
    """Flag OFF hands every LP ``time_limit=None`` — the legacy path, intact."""
    now = _fake_clock(monkeypatch)
    _deadline_flag(monkeypatch, False)
    seen = _charging_lp_stub(monkeypatch, now, cost=0.0)
    _run_stage(time_budget_s=10.0)
    assert seen["calls"] > 0, "stub never ran — probe measured nothing"
    assert all(v is None for v in seen["limits"]), (
        f"the legacy arm passed a deadline to HiGHS: {seen['limits']}"
    )


def test_the_deadline_reaches_highs_not_just_the_python_loop(monkeypatch):
    """``_solve_lp`` must hand the limit to HiGHS itself.

    The tests above stub ``_solve_lp``, so they prove the loop computes a
    deadline but not that HiGHS is told about it -- and a deadline HiGHS never
    sees bounds nothing, which is the exact shape of the bug (a between-rounds
    check cannot stop an overrun that happens inside one ``h.run()``).
    """
    import discopt.solvers._root_cuts as rc
    import highspy
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = _build_convex_minlp("max")
    is_int = np.array([False, False, True, True])
    root = rc._RootLP(
        m,
        NLPEvaluator(m),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 1.0, 1.0]),
        is_int,
        is_int.copy(),
        True,
    )

    opts: list = []
    real_highs = highspy.Highs

    class _Recording(real_highs):  # type: ignore[misc,valid-type]
        def setOptionValue(self, key, val):  # noqa: N802 - HiGHS' own spelling
            opts.append((key, val))
            return super().setOptionValue(key, val)

    monkeypatch.setattr(highspy, "Highs", _Recording)

    rc._solve_lp(root, [], [], 4.25)
    assert ("time_limit", 4.25) in opts, f"HiGHS never got the deadline; options seen: {opts}"

    opts.clear()
    rc._solve_lp(root, [], [], None)
    assert not [k for k, _ in opts if k == "time_limit"], (
        f"a deadline was set on the legacy path: {opts}"
    )

    opts.clear()
    rc._solve_lp(root, [], [], -5.0)
    tl = [v for k, v in opts if k == "time_limit"]
    assert tl and tl[0] > 0.0, (
        f"a non-positive limit reached HiGHS as {tl}; HiGHS reads <= 0 as "
        f"'no limit', which would restore the overrun this bounds"
    )


# ── #1141: a truncated LP must not discard what the stage already proved ──────
#
# The added work item on #1141 is "graduate or delete DISCOPT_ROOT_CUT_DEADLINE".
# Panelling it found the flag failing bar 1 (a certification regression on tls2),
# and the mechanism is entirely in how `generate_root_cuts` handles an LP that
# stops on the deadline: it returns the all-`None` declined tuple, and both
# places that consume it treat "this LP declined" as "the stage knows nothing".
#
# Both paths are fixed only in the deadline arm. With the flag off a decline is a
# structural or numerical LP failure rather than a budget stop, and restoring an
# earlier solve there would change the DEFAULT path's cuts -- a bound-changing
# edit that belongs to its own panel (CLAUDE.md §5). The legacy behaviour is
# pinned by `test_the_legacy_no_lp_exit_still_discards` below.


def _stub_lp_declining_on(monkeypatch, decline_calls, bounds, x_fixed):
    """``_solve_lp`` stub that DECLINES on the given 1-based call indices.

    Returns ``h = None`` so the GMI separator (which needs a real basis) stays
    out of this fixture; the cut supply is `_select_cuts` below.
    """
    import discopt.solvers._root_cuts as rc

    seen = {"calls": 0, "declined": 0}
    decline_calls = set(decline_calls)

    def _stub(root, cuts_a, cuts_b, time_limit=None):
        seen["calls"] += 1
        if seen["calls"] in decline_calls:
            seen["declined"] += 1
            return None, None, None, None
        i = min(seen["calls"] - 1, len(bounds) - 1)
        return float(bounds[i]), np.array(x_fixed, float), None, None

    monkeypatch.setattr(rc, "_solve_lp", _stub)
    return seen


def _stub_two_cuts_one_binding(monkeypatch, x_fixed):
    """Every round offers one cut BINDING at ``x_fixed`` and one far from it.

    The pair is what makes the binding filter observable: a result that kept both
    did not filter, a result that kept only the first did.
    """
    import discopt.solvers._root_cuts as rc

    a = np.zeros(len(x_fixed))
    a[0] = 1.0
    binding_rhs = float(a @ np.array(x_fixed, float))  # slack exactly 0

    def _always_two(candidates, x, **kw):
        return [(a.copy(), binding_rhs), (a.copy(), binding_rhs + 98.0)]

    monkeypatch.setattr(rc, "_select_cuts", _always_two)
    return binding_rhs


def _run_min_stage(time_budget_s=1e6):
    import discopt.solvers._root_cuts as rc
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = _build_convex_minlp("min")
    return rc.generate_root_cuts(
        m,
        NLPEvaluator(m),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 1.0, 1.0]),
        np.array([False, False, True, True]),
        np.array([False, False, True, True]),
        time_budget_s=time_budget_s,
    )


# ``f0^2 + f1^2 <= 16`` holds here, so ``add_oa`` adds nothing and each
# convergence is exactly ONE LP -- which makes the call indices below the round
# indices. The violating point is used by the mid-convergence test instead.
_X_OA_FEASIBLE = [1.0, 1.0, 0.5, 0.5]
_X_OA_VIOLATING = [4.0, 4.0, 0.5, 0.5]


def test_a_declined_lp_at_a_round_boundary_still_filters_to_binding_cuts(monkeypatch):
    """#1141: the ``no_lp`` exit skipped the binding filter and shipped everything.

    ``generate_root_cuts`` promises "the returned cuts are only those BINDING at
    the final LP optimum ... the full applied set (measured: ~170 dense rows on
    rsyn0805m) collapses node NLP throughput". On a ``no_lp`` exit ``x`` was
    ``None``, so the filter was skipped and the whole applied set was returned --
    exactly the row flood the docstring exists to prevent, and measured on tls2
    as 90 rows where the filter yields 19.

    Fails before the fix with 6 cuts (2 per round × 3 rounds, unfiltered).
    """
    _deadline_flag(monkeypatch, True)
    seen = _stub_lp_declining_on(
        monkeypatch, decline_calls={4}, bounds=[5.0, 6.0, 7.0], x_fixed=_X_OA_FEASIBLE
    )
    _stub_two_cuts_one_binding(monkeypatch, _X_OA_FEASIBLE)

    res = _run_min_stage()

    assert seen["declined"] == 1, "the decline never fired; the test proves nothing"
    assert seen["calls"] == 4, seen
    assert res.stop_reason == "no_lp"
    assert res.rounds_run == 3
    # 3 rounds × 2 cuts = 6 applied; exactly half of them bind at the last solved
    # LP optimum, so a filtered result has 3 and an unfiltered one has 6.
    assert len(res.cuts) == 3, f"binding filter skipped: {len(res.cuts)} cuts kept"
    assert all(abs(float(a @ np.array(_X_OA_FEASIBLE)) - r) <= 1e-6 for a, r in res.cuts)


def test_a_declined_lp_mid_convergence_keeps_the_lp_that_did_close(monkeypatch):
    """#1141: one truncated LP threw away an OPTIMAL LP from the same call.

    ``oa_converge`` overwrote ``obj, x, duals, h`` in place, so a decline on the
    second LP of a convergence discarded the first -- and when that happened in
    the prologue, ``generate_root_cuts`` took its ``x is None`` early return and
    the stage contributed nothing at all.

    The retained LP is a relaxation of the one that timed out (it is over a
    SUBSET of the OA rows), so its optimum is a valid root bound and cuts
    separated from it are valid. Fails before the fix: empty result, 0 rounds.
    """
    _deadline_flag(monkeypatch, True)
    seen = _stub_lp_declining_on(
        monkeypatch,
        decline_calls={2},
        bounds=[5.0, 5.0, 6.0, 7.0, 8.0],
        x_fixed=_X_OA_VIOLATING,
    )
    _stub_two_cuts_one_binding(monkeypatch, _X_OA_VIOLATING)

    res = _run_min_stage()

    assert seen["declined"] == 1, "the decline never fired; the test proves nothing"
    assert res.rounds_run >= 1, "prologue decline still aborted the whole stage"
    assert res.lp_bound is not None
    assert res.cuts, "stage returned no cuts despite a converged prologue LP"


def test_the_legacy_no_lp_exit_still_discards(monkeypatch):
    """The flag-OFF path is untouched by both fixes above (CLAUDE.md §5).

    Same stub, same decline, flag off: the prologue decline must still abort the
    stage. If this ever starts passing cuts through, the default path changed
    without a panel.
    """
    _deadline_flag(monkeypatch, False)
    seen = _stub_lp_declining_on(
        monkeypatch,
        decline_calls={2},
        bounds=[5.0, 5.0, 6.0, 7.0, 8.0],
        x_fixed=_X_OA_VIOLATING,
    )
    _stub_two_cuts_one_binding(monkeypatch, _X_OA_VIOLATING)

    res = _run_min_stage()

    assert seen["declined"] == 1, "the decline never fired; the test proves nothing"
    assert res.cuts == []
    assert res.rounds_run == 0
    assert res.lp_bound is None


def test_separate_gmi_refuses_a_basis_from_a_different_row_system():
    """The invariant the retention fix has to preserve, made structural.

    ``separate_gmi`` pairs ``binv[r]`` / ``row_st[r]`` with ``a_all[r]``
    POSITIONALLY, and the basis rows are ``[<= rows..., == rows...]``. Hand it a
    row system one row wider than the LP the basis came from and row ``m_le``
    multiplies an EQUALITY row's basis entry by a cut row -- an invalid cut, with
    nothing to signal it. Keeping a solved LP across an ``add_oa`` that appends
    rows is exactly how that could arise, so the fix rolls the appended rows back
    and this guard is what says so out loud.
    """
    import discopt.solvers._root_cuts as rc
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = _build_convex_minlp("min")
    root = rc._RootLP(
        m,
        NLPEvaluator(m),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 1.0, 1.0]),
        np.array([False, False, True, True]),
        np.array([False, False, True, True]),
        False,
    )
    obj, x, duals, h = rc._solve_lp(root, [], [])
    assert h is not None, "fixture LP did not solve; the guard is untested"

    a_all = np.vstack([root.A_le, np.zeros((1, root.n))])
    b_all = np.concatenate([root.b_le, [1.0]])
    with pytest.raises(ValueError, match="mismatched basis"):
        rc.separate_gmi(root, h, x, a_all, b_all)

    # ...and the matched system is still accepted (the guard is not a blanket no).
    rc.separate_gmi(root, h, x, root.A_le, root.b_le)
