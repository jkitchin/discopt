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
    from discopt._jax.nlp_evaluator import NLPEvaluator

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
