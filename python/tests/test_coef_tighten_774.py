"""Feasible-set equivalence gate for big-M coefficient tightening (#774, re-do of #770).

The #770 implementation produced a FALSE PRIMAL (#772): its write-back moved the
Savelsbergh slack into ``con.rhs``, violating the documented normalized-form
invariant ``Constraint.rhs == 0.0``. Every consumer (``NLPEvaluator``, the
relaxation compilers, ``_infer_constraint_bounds``) compiles the body and tests
it against 0, so the rewritten row was silently RELAXED by the slack and the
solver accepted integer points infeasible in the original problem.

The #770 test suite checked feasible-set equivalence while *honoring* ``con.rhs``
— mathematically true of the mutated model, but not what any consumer computes —
so it passed on unsound code. This suite therefore reads rows exactly the way
consumers do (``body ⋈ 0``) and enforces:

  * the ``rhs == 0.0`` invariant survives the rewrite;
  * feasible-set EQUIVALENCE in BOTH directions, proven per binary corner with
    exact LPs (no sampling gaps): no integer-feasible point removed AND no
    infeasible point admitted;
  * the evaluator cache is invalidated by the rewrite (stale compiled
    evaluators must not keep serving the un-tightened rows);
  * a continuous variable with [0,1] bounds is NOT treated as binary (the
    ``"IN" in "contINuous"`` misclassification latent in #770);
  * end-to-end flag-ON regression on the real instance class: the reported
    incumbent is feasible in the PRISTINE model and never super-optimal.

Every test here fails against the reverted #770 implementation.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pytest
from discopt import Model
from discopt.solvers._root_presolve import (
    coef_tighten_enabled,
    tighten_bigm_coefficients,
)

DATA_DIR = Path(__file__).parent / "data" / "minlplib_nl"
BENCH_NL = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))

# minlplib.solu oracles (all maximize)
OPT_SYN05HFSG = 837.7324009
OPT_RSYN0805M = 1296.120603
OPT_SYN40M = 67.71325586


def _flag(monkeypatch, on: bool) -> None:
    if on:
        monkeypatch.setenv("DISCOPT_COEF_TIGHTEN", "1")
    else:
        monkeypatch.delenv("DISCOPT_COEF_TIGHTEN", raising=False)


def _consumer_rows(model: Model):
    """Rows exactly as solver consumers read them: ``coeffs·x + const ⋈ 0``.

    ``NLPEvaluator`` compiles ``con.body`` and ``_infer_constraint_bounds``
    derives cl/cu from the sense alone, so ``con.rhs`` is *never* consulted
    downstream. Assert the normalized-form invariant instead of honoring a
    nonzero rhs — a nonzero rhs IS the #772 bug.
    """
    from discopt._jax.problem_classifier import (
        _extract_linear_coefficients,
        _NotLinearError,
    )

    n = len(model._variables)
    rows = []
    for con in model._constraints:
        assert con.rhs == 0.0, (
            f"normalized-form invariant violated: con.rhs={con.rhs!r} (consumers "
            "compile body vs 0 and silently drop rhs — the #772 false-primal bug)"
        )
        try:
            coeffs, const = _extract_linear_coefficients(con.body, model, n)
        except _NotLinearError:
            continue
        rows.append((np.asarray(coeffs, float), float(const), con.sense))
    return rows


def _feasible(rows, point: np.ndarray, tol: float = 1e-6) -> bool:
    for coeffs, const, sense in rows:
        act = float(coeffs @ point) + const
        if sense == "<=" and act > tol:
            return False
        if sense == ">=" and act < -tol:
            return False
        if sense == "==" and abs(act) > tol:
            return False
    return True


def _max_row_violation_lp(target_row, rows, bounds, fixed):
    """Exact max of ``target_row`` violation subject to ``rows`` + box + fixations.

    Returns the LP maximum of ``coeffs·x + const`` (sense ``<=``; callers
    pre-orient) over the polytope defined by ``rows`` (consumer convention),
    variable ``bounds``, and ``fixed`` (index -> value). ``None`` when the LP is
    infeasible (that corner admits no point at all).
    """
    from scipy.optimize import linprog

    t_coeffs, t_const, t_sense = target_row
    obj = np.asarray(t_coeffs, float)
    if t_sense == ">=":  # violation of a >= row is -(coeffs·x + const)
        obj = -obj
    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for coeffs, const, sense in rows:
        if sense == "<=":
            a_ub.append(coeffs)
            b_ub.append(-const)
        elif sense == ">=":
            a_ub.append(-coeffs)
            b_ub.append(const)
        else:
            a_eq.append(coeffs)
            b_eq.append(-const)
    lp_bounds = list(bounds)
    for j, v in fixed.items():
        lp_bounds[j] = (v, v)
    res = linprog(
        -obj,  # maximize obj
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(a_eq) if a_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=lp_bounds,
        method="highs",
    )
    if not res.success:
        return None
    val = float(obj @ res.x)
    return val + (t_const if t_sense != ">=" else -t_const)


def _assert_equivalent_at_integer_corners(orig: Model, tight: Model, tol: float = 1e-6):
    """Exact bidirectional feasible-set equivalence, per binary corner.

    For every assignment of the binaries:
      * ADMIT direction: no point of the tightened polytope violates any
        original row (feasible-in-tightened ⇒ feasible-in-original);
      * REMOVE direction: no point of the original polytope violates any
        tightened row (feasible-in-original ⇒ feasible-in-tightened).
    LP maxima are exact — no sampling gap.
    """
    orig_rows = _consumer_rows(orig)
    tight_rows = _consumer_rows(tight)
    bounds = []
    bins = []
    for j, v in enumerate(orig._variables):
        lb = float(getattr(v, "lb", 0.0))
        ub = float(getattr(v, "ub", np.inf))
        bounds.append((None if not np.isfinite(lb) else lb, None if not np.isfinite(ub) else ub))
        vt = str(getattr(v, "vtype", getattr(v, "var_type", ""))).upper()
        if vt.endswith("BINARY"):
            bins.append(j)
    assert bins, "test model must contain binaries"

    for corner in itertools.product((0.0, 1.0), repeat=len(bins)):
        fixed = dict(zip(bins, corner))
        for row in orig_rows:
            v = _max_row_violation_lp(row, tight_rows, bounds, fixed)
            assert v is None or v <= tol, (
                f"point ADMITTED at corner {corner}: original row violated by {v:.6g} "
                "inside the tightened polytope (feasible set enlarged — the #772 bug class)"
            )
        for row in tight_rows:
            v = _max_row_violation_lp(row, orig_rows, bounds, fixed)
            assert v is None or v <= tol, (
                f"feasible point REMOVED at corner {corner}: tightened row violated by "
                f"{v:.6g} inside the original polytope (unsound tightening)"
            )


def _build_fixed_charge() -> Model:
    """Small fixed-charge network with slack big-Ms in BOTH sign cases.

    ``flow_i <= 5`` caps each flow while the big-M is 40 (8x slack) — the
    ``a_k < 0`` fixed-charge case. The ``40·y0 + f0 + f1 <= 45`` row exercises
    the positive-coefficient Savelsbergh case: FBBT caps the rest-activity at
    10, so slack = 45 − 10 = 35 < a_k = 40 → a_k shrinks to 5.
    """
    m = Model("fc")
    f0 = m.continuous("f0", lb=0.0, ub=float("inf"))
    f1 = m.continuous("f1", lb=0.0, ub=float("inf"))
    y0 = m.binary("y0")
    y1 = m.binary("y1")
    m.subject_to(f0 - 40.0 * y0 <= 0.0)  # fixed-charge (a_k < 0)
    m.subject_to(f1 - 40.0 * y1 <= 0.0)
    m.subject_to(f0 <= 5.0)  # capacity => FBBT ub(f)=5
    m.subject_to(f1 <= 5.0)
    m.subject_to(f0 + f1 >= 3.0)  # demand
    m.subject_to(40.0 * y0 + f0 + f1 <= 45.0)  # Savelsbergh (a_k > 0)
    m.maximize(f0 + f1 - 2.0 * y0 - 2.0 * y1)
    return m


# ── unit: flag gating and mechanism ──────────────────────────────────────────


def test_flag_default_off(monkeypatch):
    _flag(monkeypatch, False)
    assert coef_tighten_enabled() is False
    m = _build_fixed_charge()
    cons_before = list(m._constraints)
    assert tighten_bigm_coefficients(m) == 0
    assert list(m._constraints) == cons_before  # same objects, untouched


def test_fires_and_reduces_bigm(monkeypatch):
    _flag(monkeypatch, True)
    assert coef_tighten_enabled() is True
    m = _build_fixed_charge()
    n = tighten_bigm_coefficients(m)
    assert n >= 2  # both fixed-charge rows tightened
    for coeffs, _const, _sense in _consumer_rows(m):
        assert np.max(np.abs(coeffs)) <= 6.0 + 1e-6, (
            f"a slack big-M coefficient survived: {np.max(np.abs(coeffs))}"
        )


def test_unbounded_activity_row_skipped(monkeypatch):
    """A row whose non-binary activity slack is zero is left untouched."""
    _flag(monkeypatch, True)
    m = Model("unb")
    x = m.continuous("x", lb=0.0, ub=float("inf"))
    y = m.binary("y")
    m.subject_to(x - 40.0 * y <= 0.0)  # only bound on x is via this row
    m.maximize(y)
    # FBBT derives ub(x)=40 from the row itself => M == cap => nothing to do.
    assert tighten_bigm_coefficients(m) == 0


def test_continuous_unit_interval_var_is_not_binary(monkeypatch):
    """The #770 classifier matched "IN" in "contINuous" — a continuous z∈[0,1]
    was tightened as if it could only take values {0,1}, cutting its genuinely
    feasible fractional range."""
    _flag(monkeypatch, True)
    m = Model("cont01")
    x = m.continuous("x", lb=0.0, ub=float("inf"))
    z = m.continuous("z", lb=0.0, ub=1.0)  # continuous, NOT a binary
    m.subject_to(x - 40.0 * z <= 0.0)
    m.subject_to(x <= 5.0)
    m.maximize(x - z)
    assert tighten_bigm_coefficients(m) == 0, (
        "no binaries in the model — any rewrite treated a continuous [0,1] "
        "variable as binary (unsound: removes fractional-z feasible points)"
    )


# ── the gate: exact bidirectional feasible-set equivalence ───────────────────


def test_rhs_invariant_and_bidirectional_equivalence(monkeypatch):
    orig = _build_fixed_charge()

    _flag(monkeypatch, True)
    tight = _build_fixed_charge()
    assert tighten_bigm_coefficients(tight) >= 2
    # _consumer_rows asserts the rhs==0 invariant on every constraint.
    _assert_equivalent_at_integer_corners(orig, tight)


def test_grid_equivalence_consumer_convention(monkeypatch):
    """Redundant belt-and-braces sampling check at the consumer convention."""
    orig = _build_fixed_charge()
    orig_rows = _consumer_rows(orig)

    _flag(monkeypatch, True)
    tight = _build_fixed_charge()
    assert tighten_bigm_coefficients(tight) >= 2
    tight_rows = _consumer_rows(tight)

    grid = np.linspace(0.0, 6.0, 13)  # spans past the ub=5 cap deliberately
    removed = added = 0
    for y0, y1 in itertools.product((0.0, 1.0), repeat=2):
        for f0 in grid:
            for f1 in grid:
                pt = np.array([f0, f1, y0, y1], float)
                fo = _feasible(orig_rows, pt)
                ft = _feasible(tight_rows, pt)
                removed += fo and not ft
                added += ft and not fo
    assert removed == 0, f"{removed} integer-feasible points removed (unsound)"
    assert added == 0, f"{added} points newly admitted (the #772 bug class)"


def test_evaluator_cache_invalidated(monkeypatch):
    """Stale compiled evaluators must not survive the rewrite (identity-keyed
    fingerprint): #770 mutated Constraint objects in place, so evaluators built
    before the tightening kept serving the un-tightened rows."""
    from discopt._jax.nlp_evaluator import cached_evaluator

    _flag(monkeypatch, True)
    m = _build_fixed_charge()
    ev_pre = cached_evaluator(m)
    assert tighten_bigm_coefficients(m) >= 2
    ev_post = cached_evaluator(m)
    assert ev_post is not ev_pre, "evaluator cache served a stale pre-tightening compile"

    # And the post evaluator agrees with the consumer-convention rows.
    rows = _consumer_rows(m)
    pt = np.array([2.0, 3.0, 1.0, 1.0], float)
    g = np.asarray(ev_post.evaluate_constraints(pt))
    manual = np.array([float(c @ pt) + k for c, k, _s in rows])
    np.testing.assert_allclose(g, manual, atol=1e-9)


def test_optimum_unchanged(monkeypatch):
    _flag(monkeypatch, False)
    base = _build_fixed_charge().solve(time_limit=20, gap_tolerance=1e-6)

    _flag(monkeypatch, True)
    m = _build_fixed_charge()
    assert tighten_bigm_coefficients(m) >= 2
    _flag(monkeypatch, False)  # solve itself shouldn't re-tighten
    tuned = m.solve(time_limit=20, gap_tolerance=1e-6)

    assert base.objective is not None and tuned.objective is not None
    assert tuned.objective == pytest.approx(base.objective, abs=1e-4, rel=1e-4)


# ── real instance class (vendored corpus; CI-runnable) ───────────────────────


def test_syn05hfsg_rhs_invariant_and_no_false_primal(monkeypatch):
    """End-to-end flag-ON on the vendored process-synthesis instance: the
    tightening fires, the rhs invariant holds, and the reported incumbent is
    feasible in a PRISTINE model and never super-optimal (maximize)."""
    import discopt.modeling as dm

    nl = DATA_DIR / "syn05hfsg.nl"
    _flag(monkeypatch, True)
    m = dm.from_nl(str(nl))
    n = tighten_bigm_coefficients(m)
    assert n > 0, "tightening did not fire on the real fixed-charge class"
    _consumer_rows(m)  # asserts rhs==0 on every row

    r = dm.from_nl(str(nl)).solve(time_limit=60)
    assert not getattr(r, "incumbent_verification_failed", False), (
        "the #779 guard tripped — the solve produced a false primal"
    )
    assert r.objective is not None
    assert r.objective <= OPT_SYN05HFSG + 1e-3, (
        f"super-optimal incumbent {r.objective} > opt {OPT_SYN05HFSG} (false primal)"
    )
    # Independently verify the incumbent against a pristine parse.
    from discopt._jax.nlp_evaluator import cached_evaluator
    from discopt._jax.primal_heuristics import _check_constraint_feasibility

    pm = dm.from_nl(str(nl))
    flat = np.concatenate(
        [np.atleast_1d(np.asarray(r.x[v.name], float)).ravel() for v in pm._variables]
    )
    assert _check_constraint_feasibility(cached_evaluator(pm), flat, tol=1e-4), (
        "incumbent infeasible in the pristine original model (false primal)"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,opt",
    [("rsyn0805m", OPT_RSYN0805M), ("syn40m", OPT_SYN40M)],
)
def test_772_named_regression(monkeypatch, name, opt):
    """The exact #772 reproduction: flag-ON solve must return an incumbent that
    is feasible in the pristine .nl model and not super-optimal."""
    nl = BENCH_NL / f"{name}.nl"
    if not nl.exists():
        pytest.skip("benchmark corpus not available")
    import discopt.modeling as dm
    from discopt._jax.nlp_evaluator import cached_evaluator
    from discopt._jax.primal_heuristics import _check_constraint_feasibility

    _flag(monkeypatch, True)
    r = dm.from_nl(str(nl)).solve(time_limit=60, verify_incumbent=False)
    assert r.objective is not None
    assert r.objective <= opt + 1e-3, (
        f"super-optimal incumbent {r.objective} > opt {opt} (the #772 false primal)"
    )
    pm = dm.from_nl(str(nl))
    flat = np.concatenate(
        [np.atleast_1d(np.asarray(r.x[v.name], float)).ravel() for v in pm._variables]
    )
    assert _check_constraint_feasibility(cached_evaluator(pm), flat, tol=1e-4), (
        "incumbent infeasible in the pristine original model (the #772 false primal)"
    )
