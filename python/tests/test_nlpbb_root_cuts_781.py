"""Tests for the NLP-BB root cutting-plane stage (#781, DISCOPT_NLPBB_ROOT_CUTS).

Soundness gates:
  * the GMI separator is validated by EXACT enumeration: on seeded random
    MILPs, every emitted cut must be violated at the LP vertex and satisfied
    by every integer assignment's full continuous completion (LP certificate
    per integer corner — no sampling gap);
  * flag OFF is inert (no constraints added, no behavior change);
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
    if on:
        monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS", "1")
    else:
        monkeypatch.delenv("DISCOPT_NLPBB_ROOT_CUTS", raising=False)


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

    def _frozen_lp(root, cuts_a, cuts_b):
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
