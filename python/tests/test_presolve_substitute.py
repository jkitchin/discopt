"""Regression tests for the batch substitution aggregator (#844, P2(a′)).

Covers the three properties the Rust pass has to guarantee end to end:

1. **substitute-everywhere** — a variable defined by a linear equality is
   rewritten out of every row *and* the objective, which is exactly what the
   pre-existing ``aggregate``/``eliminate`` passes cannot do (they require the
   eliminated variable to appear in a single expression). ``test_legacy_*``
   pins that pre-existing limitation so the difference is measured, not
   asserted;
2. **postsolve inverts exactly** — including a chain of ≥3 substitutions and a
   rejected cycle;
3. **the solve path agrees** — ``DISCOPT_PRESOLVE_SUBSTITUTE=1`` returns the
   same optimum, in the ORIGINAL variables, as the default path.
"""

import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._rust import model_to_repr


def _repr_of(model):
    return model_to_repr(model, getattr(model, "_builder", None))


def _chain_model():
    """x0 = 2·x1 + 1, x1 = 3·x2 + 2, x2 = 4·x3 + 3 (a depth-3 chain), plus an
    inequality and an objective in which every chained variable appears."""
    m = dm.Model("chain")
    x = [m.continuous(f"x{i}", lb=-1e3, ub=1e3) for i in range(5)]
    m.subject_to(x[0] - 2 * x[1] == 1)
    m.subject_to(x[1] - 3 * x[2] == 2)
    m.subject_to(x[2] - 4 * x[3] == 3)
    m.subject_to(x[0] + x[1] + x[2] + x[3] <= 100)
    m.minimize(x[0] + x[4])
    return m


@pytest.mark.smoke
def test_legacy_aggregate_pass_cannot_reduce_the_chain_model():
    """Fails-before evidence: the pre-existing passes eliminate nothing here."""
    rep = _repr_of(_chain_model())
    reduced, _ = rep.presolve(
        passes=["eliminate", "aggregate", "factorable_elim"],
        max_iterations=8,
        time_limit_ms=5000,
    )
    assert reduced.n_vars == rep.n_vars == 5


@pytest.mark.smoke
def test_chain_of_three_substitutions_is_eliminated_and_inverted():
    m = _chain_model()
    rep = _repr_of(m)
    reduced, chain = rep.substitute(4)

    assert chain.variables_eliminated == 3
    assert reduced.n_vars == 2, "x3 and the unused x4 survive"
    assert reduced.n_constraints == 1, "only the inequality remains"

    # Every definition names the SAME surviving representative: the chain
    # collapsed, which is why postsolve needs no topological ordering.
    recs = chain.records(0)
    sources = {r["source_block"] for r in recs if r["kind"] == "affine"}
    assert len(sources) == 1, f"expected one representative, got {sources}"

    checked = 0
    for t in (-3.5, 0.0, 2.25, 7.0):
        x_red = [t + 0.5 * i for i in range(reduced.n_vars)]
        x_full = np.asarray(chain.postsolve(x_red), float)
        assert x_full.shape == (rep.n_vars,)
        obj_r, _, _ = reduced.evaluate_point(x_red)
        obj_p, con_p, _ = rep.evaluate_point(list(x_full))
        # The three dropped equalities are the ONLY equalities in the model, so
        # a zero constraint violation on the pristine model at the lifted point
        # is exactly the statement that postsolve inverted them.
        assert con_p == pytest.approx(0.0, abs=1e-9), f"dropped rows violated at t={t}"
        assert obj_p == pytest.approx(obj_r, abs=1e-9)
        checked += 1
    assert checked == 4


@pytest.mark.smoke
def test_cycle_is_rejected_and_its_row_is_kept():
    m = dm.Model("cycle")
    x = [m.continuous(f"x{i}", lb=-10, ub=10) for i in range(3)]
    m.subject_to(x[0] - x[1] == 0)
    m.subject_to(x[1] - x[2] == 0)
    m.subject_to(x[2] - x[0] == 0)  # closes the cycle: defines nothing new
    m.minimize(x[0])

    rep = _repr_of(m)
    reduced, chain = rep.substitute(4)
    stats = chain.sweep_stats()[0]

    assert chain.variables_eliminated == 2
    assert stats["cycles_rejected"] == 1
    assert reduced.n_vars == 1
    assert reduced.n_constraints == 1, "the cycle row stays in the model"
    assert chain.postsolve([4.25]) == pytest.approx([4.25, 4.25, 4.25])


@pytest.mark.smoke
def test_infeasible_transfer_is_reported_and_the_model_is_unchanged():
    m = dm.Model("infeas")
    a = m.continuous("a", lb=0.0, ub=1.0)
    b = m.continuous("b", lb=5.0, ub=6.0)
    m.subject_to(a - b == 0)
    m.minimize(a)

    rep = _repr_of(m)
    reduced, chain = rep.substitute(4)
    assert chain.variables_eliminated == 0
    assert reduced.n_vars == rep.n_vars
    assert reduced.n_constraints == rep.n_constraints


@pytest.mark.smoke
def test_integer_source_is_kept_and_continuous_target_is_eliminated():
    m = dm.Model("intsrc")
    x = m.continuous("x", lb=-100, ub=100)
    y = m.integer("y", lb=0, ub=10)
    m.subject_to(x - 2 * y == 1)
    m.minimize(x)

    rep = _repr_of(m)
    reduced, chain = rep.substitute(4)
    assert chain.variables_eliminated == 1
    assert reduced.var_types() == ["integer"]
    assert chain.postsolve([3.0]) == pytest.approx([7.0, 3.0])


@pytest.mark.smoke
def test_solve_path_flag_agrees_with_the_default_path():
    """The reduced solve must return the same optimum, in original variables."""
    prev = os.environ.get("DISCOPT_PRESOLVE_SUBSTITUTE")
    try:
        os.environ["DISCOPT_PRESOLVE_SUBSTITUTE"] = "0"
        base = _chain_model().solve(time_limit=30)
        os.environ["DISCOPT_PRESOLVE_SUBSTITUTE"] = "1"
        sub = _chain_model().solve(time_limit=30)
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_PRESOLVE_SUBSTITUTE", None)
        else:
            os.environ["DISCOPT_PRESOLVE_SUBSTITUTE"] = prev

    assert sub.status == base.status
    assert sub.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)
    # Reported in the ORIGINAL variables, and feasible for the pristine model.
    assert set(sub.x) == {"x0", "x1", "x2", "x3", "x4"}
    rep = _repr_of(_chain_model())
    flat = [float(np.asarray(sub.x[f"x{i}"]).reshape(-1)[0]) for i in range(5)]
    _, con_viol, bnd_viol = rep.evaluate_point(flat)
    assert max(con_viol, bnd_viol) < 1e-6


# ─────────────────────────────────────────────────────────────────────────
# Card 6a — root-causing the measured 2449 % primal-gap incident
#
# ``watercontamination0202`` (the instance the incident was measured on,
# `sota-parity-analysis-2026-07-27.md` §G-G.1) is snapshot-only and not present
# here, so the card was worked by construction: read the transform, decide which
# mechanism could return a wrong answer, and prove the answer with a reproducer
# that runs in this repository.
#
# What the entry experiment found (numbers in the plan's §6 entry):
#
#   * the transform IS exact.  Per-row and per-variable identities over every
#     in-repo instance the pass reduces (13 of 66) — 13,086 surviving-row
#     comparisons, 3,642 dropped-row identity checks, 52 box-soundness and 26
#     box-exactness checks, plus 104 objective / 104 bridge-fidelity comparisons
#     — all clean.  So none of the four candidate mechanisms (postsolve not
#     inverting, variables not restored, eliminated bounds dropped, objective
#     read off the reduced model) is real;
#   * what IS real is that the reduced model is a different problem *for the
#     relaxation engine*, and the effect is large and NOT monotone.
#
# These tests pin both halves so a future change cannot quietly break either.
# ─────────────────────────────────────────────────────────────────────────

_REDUCING_WITH_ORACLE = {
    # instance: (reference optimum, sense) — every value from
    # docs/dev/data/cert-optima.json via reference_optima.oracle_table(); never
    # hand-typed (CLAUDE.md / discopt_benchmarks/data/local_oracle.json README).
    "gkocis": "minimize",
    "st_e11": "minimize",
    "syn05hfsg": "maximize",
}


def _corpus_nl(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", f"{name}.nl")


@pytest.mark.slow
def test_postsolve_is_exact_on_every_reducing_corpus_instance():
    """The reproducer for Card 6a's four candidate mechanisms — all falsified.

    For every in-repo instance the pass actually reduces, sample the reduced box
    and check the two identities that make the transform an equivalence rather
    than a relaxation:

    * the pristine residual multiset at the lifted point equals the reduced
      residual multiset padded with one zero per dropped row (a surviving row
      whose value moved, or a dropped definitional row that does not hold at the
      lifted point, both break this);
    * a point inside the reduced box lifts inside the pristine box, and a point
      outside the reduced box lifts outside the pristine box (soundness *and*
      exactness of the transferred bounds).
    """
    import glob

    rng = np.random.default_rng(20260730)
    n_row_cmp = 0
    n_drop_cmp = 0
    n_box_in = 0
    n_box_out = 0
    reduced_instances = 0
    problems = []

    def boxes(rep):
        lo = np.empty(rep.n_vars)
        hi = np.empty(rep.n_vars)
        k = 0
        for i in range(rep.n_var_blocks):
            a = np.asarray(rep.var_lb(i), dtype=float).reshape(-1)
            b = np.asarray(rep.var_ub(i), dtype=float).reshape(-1)
            lo[k : k + a.size] = a
            hi[k : k + b.size] = b
            k += a.size
        return lo, hi

    def resid(v, sense, rhs):
        if np.isnan(v):
            return float("nan")
        if sense == "==":
            return abs(v - rhs)
        if sense == "<=":
            return max(v - rhs, 0.0)
        return max(rhs - v, 0.0)

    corpus = sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "*.nl"))
    )
    assert corpus, "corpus is empty — the probe would measure nothing"

    for path in corpus:
        m = dm.from_nl(path)
        pristine = _repr_of(m)
        reduced, chain = pristine.substitute(4)
        if chain.refused is not None or chain.variables_eliminated == 0:
            continue
        reduced_instances += 1
        n_drop = pristine.n_constraints - reduced.n_constraints
        assert n_drop >= 0, f"{path}: reduction ADDED rows"

        psense = [pristine.constraint_sense(i) for i in range(pristine.n_constraints)]
        prhs = [pristine.constraint_rhs(i) for i in range(pristine.n_constraints)]
        rsense = [reduced.constraint_sense(i) for i in range(reduced.n_constraints)]
        rrhs = [reduced.constraint_rhs(i) for i in range(reduced.n_constraints)]
        plo, phi = boxes(pristine)
        rlo, rhi = boxes(reduced)
        slo = np.clip(np.nan_to_num(rlo, neginf=-1e3, posinf=1e3), -1e3, 1e3)
        shi = np.maximum(np.clip(np.nan_to_num(rhi, neginf=-1e3, posinf=1e3), -1e3, 1e3), slo)

        for trial in range(4):
            x_red = slo + rng.random(slo.shape) * (shi - slo)
            forced = None
            if trial >= 2:  # deliberately leave the reduced box
                cand = np.flatnonzero(np.isfinite(rhi) & (np.abs(rhi) < 1e19))
                if cand.size:
                    forced = int(cand[rng.integers(cand.size)])
                    x_red[forced] = rhi[forced] + 1.0 + abs(rhi[forced])
            x_full = np.asarray(chain.postsolve(x_red.tolist()), dtype=float)
            assert x_full.size == pristine.n_vars

            pres = np.array(
                [
                    resid(pristine.evaluate_constraint(j, x_full), psense[j], prhs[j])
                    for j in range(pristine.n_constraints)
                ]
            )
            rres = np.array(
                [
                    resid(reduced.evaluate_constraint(i, x_red), rsense[i], rrhs[i])
                    for i in range(reduced.n_constraints)
                ]
            )
            if not (np.isnan(pres).any() or np.isnan(rres).any()):
                n_row_cmp += int(reduced.n_constraints)
                n_drop_cmp += int(n_drop)
                a = np.sort(pres)
                b = np.sort(np.concatenate([rres, np.zeros(n_drop)]))
                scale = 1.0 + np.maximum(np.abs(a), np.abs(b))
                worst = float(np.max(np.abs(a - b) - 1e-7 * scale))
                if worst > 0.0:
                    problems.append(f"{os.path.basename(path)}: residual multiset differs")

            in_red = bool(np.all(x_red >= rlo - 1e-9) and np.all(x_red <= rhi + 1e-9))
            in_pri = bool(np.all(x_full >= plo - 1e-9) and np.all(x_full <= phi + 1e-9))
            if in_red:
                n_box_in += 1
                if not in_pri:
                    problems.append(
                        f"{os.path.basename(path)}: reduced-feasible box point lifts outside"
                    )
            elif forced is not None:
                n_box_out += 1
                if in_pri:
                    problems.append(f"{os.path.basename(path)}: over-tightened transferred bound")

    # CLAUDE.md §6: the probe must be shown to have fired.
    print(
        f"\n[card6a exactness] instances_reduced={reduced_instances} "
        f"surviving_row_comparisons={n_row_cmp} dropped_row_checks={n_drop_cmp} "
        f"box_in={n_box_in} box_out={n_box_out} problems={len(problems)}"
    )
    assert reduced_instances >= 10, f"only {reduced_instances} instances reduced"
    assert n_row_cmp > 1000, n_row_cmp
    assert n_drop_cmp > 100, n_drop_cmp
    assert n_box_in > 0 and n_box_out > 0, (n_box_in, n_box_out)
    assert not problems, problems[:10]


@pytest.mark.smoke
def test_postsolve_guard_rejects_a_lifted_point_with_a_fractional_integer():
    """Card 6a: the #779 postsolve guard used to be integrality-blind.

    ``ModelRepr.evaluate_point`` — the whole of the guard before this test —
    checks rows and variable bounds and nothing else, so a lifted point whose
    *integral* survivor is fractional passed it and became the reported answer.

    **Not reachable through today's substitution pass**: ``substitute.rs``'s
    "Scope (v0)" never eliminates an integral block, so the reduced solve owns
    integrality and honours it. It is reachable through any other repr transform
    that reuses this postsolve entry — Card 3d proposes exactly that — and the
    guard is the last check before a point becomes the answer, so it is completed
    here rather than left leaning on an upstream invariant (CLAUDE.md §3).
    """
    from discopt.modeling.core import SolveResult, model_from_repr
    from discopt.solvers._presolve_substitute import integrality_violation, lift_result

    m = dm.Model("intguard")
    m.continuous("x", lb=-100, ub=100)
    m.integer("y", lb=0, ub=10)
    xs, ys = m._variables
    m.subject_to(xs - 2 * ys == 1)
    m.minimize(xs)

    pristine = _repr_of(m)
    reduced_repr, chain = pristine.substitute(4)
    assert chain.variables_eliminated == 1, "x must be the eliminated block"
    reduced_model = model_from_repr(reduced_repr, "intguard_substituted")
    assert [v.name for v in reduced_model._variables] == ["y"]

    # y = 3.5 lifts to x = 2*3.5 + 1 = 8.0. Both coordinates are inside their
    # pristine bounds and the only row was the (dropped) definition, so the Rust
    # evaluator reports zero constraint and zero bound violation: the guard's two
    # pre-Card-6a arms both PASS on a point that is not integer-feasible.
    x_full = np.asarray(chain.postsolve([3.5]), dtype=float)
    assert x_full == pytest.approx([8.0, 3.5])
    _, con_viol, bnd_viol = pristine.evaluate_point(list(x_full))
    assert max(con_viol, bnd_viol) <= 1e-12, "fails-before evidence: the old guard sees nothing"

    # The new arm sees it, and names the variable and the deviation.
    bad = integrality_violation(m, x_full)
    assert bad is not None and bad[0] == "y"
    assert bad[1] == pytest.approx(0.5)

    result = SolveResult(status="optimal", objective=8.0, x={"y": np.asarray(3.5)})
    assert lift_result(m, reduced_model, chain, pristine, result) is None, (
        "a lifted point with a fractional integer must be discarded, not reported"
    )

    # And the guard does not fire on a genuinely integral point.
    ok = SolveResult(status="optimal", objective=7.0, x={"y": np.asarray(3.0)})
    lifted = lift_result(m, reduced_model, chain, pristine, ok)
    assert lifted is not None
    assert lifted.objective == pytest.approx(7.0)
    assert integrality_violation(m, np.asarray([7.0, 3.0])) is None


@pytest.mark.slow
@pytest.mark.parametrize("instance", sorted(_REDUCING_WITH_ORACLE))
def test_substitution_flag_never_certifies_past_the_reference_optimum(instance):
    """Card 6a's correctness gate for the flag: sound, whatever it costs.

    The card's measurement is that the flag moves the *relaxation* by orders of
    magnitude and in both directions (hda's dual bound 132x looser, 4stufen's
    ~5x tighter, both deterministic — plan §6). A bound that moves is allowed; a
    bound that crosses the reference optimum is not. These three instances are
    the in-repo reducing instances that both certify quickly and have a proven
    reference optimum, so the invariant is checkable rather than asserted.
    """
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "discopt_benchmarks"))
    from utils.reference_optima import oracle_table

    oracle = oracle_table().get(instance)
    assert oracle is not None and oracle.proven, f"no proven oracle for {instance}"
    sense = _REDUCING_WITH_ORACLE[instance]

    prev = os.environ.get("DISCOPT_PRESOLVE_SUBSTITUTE")
    try:
        os.environ["DISCOPT_PRESOLVE_SUBSTITUTE"] = "1"
        m = dm.from_nl(_corpus_nl(instance))
        res = m.solve(time_limit=120)
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_PRESOLVE_SUBSTITUTE", None)
        else:
            os.environ["DISCOPT_PRESOLVE_SUBSTITUTE"] = prev

    tol = 1e-4 * max(1.0, abs(oracle.value))
    checked = 0
    if res.bound is not None:
        checked += 1
        if sense == "minimize":
            assert res.bound <= oracle.value + tol, (
                f"{instance}: dual bound {res.bound!r} is ABOVE the proven optimum "
                f"{oracle.value!r} with DISCOPT_PRESOLVE_SUBSTITUTE=1"
            )
        else:
            assert res.bound >= oracle.value - tol, (
                f"{instance}: dual bound {res.bound!r} is BELOW the proven optimum "
                f"{oracle.value!r} with DISCOPT_PRESOLVE_SUBSTITUTE=1"
            )
    if res.objective is not None:
        checked += 1
        if sense == "minimize":
            assert res.objective >= oracle.value - tol, (
                f"{instance}: incumbent {res.objective!r} beats the proven optimum "
                f"{oracle.value!r} — a false primal"
            )
        else:
            assert res.objective <= oracle.value + tol, (
                f"{instance}: incumbent {res.objective!r} beats the proven optimum "
                f"{oracle.value!r} — a false primal"
            )
    assert checked > 0, f"{instance}: neither a bound nor an incumbent to check"
