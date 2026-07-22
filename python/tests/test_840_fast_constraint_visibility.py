"""#840: fast=True indexed constraints must be visible to the NLPEvaluator, the
feasibility check, and the #772 incumbent-verification guard.

``m.constraint(set, rule, fast=True)`` emits single-variable-affine families as
sparse rows straight into the Rust builder and does NOT append the Constraint
objects to ``model._constraints`` (adding them there would double-count in the native
solve). Before this fix that left the NLPEvaluator — and hence
``_check_constraint_feasibility`` and the #772 guard, which verify incumbents through
it — BLIND to those constraints: ``evaluate_constraints`` returned ``[]`` and any
point was accepted. That is a soundness hole: an infeasible incumbent on a
fast-indexed model would be certified (the #829 trivial-primal seed exposed it by
injecting all-zeros on an assignment MILP → false optimum obj 0 vs the true 9).

The fix makes ``model._builder_linear_blocks`` the single source of truth for the
fast-path linear rows: the NLPEvaluator materializes them on demand via
``Model._builder_linear_constraints()``, so its constraint view is complete for BOTH
fast paths — ``constraint(fast=True)`` AND the direct ``add_linear_constraints`` matrix
API — with no separate ``_fast_constraints`` mirror to fall out of sync or double-count
once the nonlinear path materializes the rows into ``_constraints``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import scipy.sparse as sp
from discopt._jax.nlp_evaluator import cached_evaluator
from discopt._jax.primal_heuristics import _check_constraint_feasibility, is_qubo

_WORKERS = ["w1", "w2", "w3"]
_TASKS = ["a", "b", "c"]
_ACOST = {(w, t): (i + 1) * (j + 1) for i, w in enumerate(_WORKERS) for j, t in enumerate(_TASKS)}


def _assignment(fast: bool):
    m = dm.Model("assign")
    w = m.set("w", _WORKERS)
    t = m.set("t", _TASKS)
    assign = m.binary("assign", over=w * t)
    m.minimize(dm.sum(_ACOST[(i, j)] * assign[i, j] for i in w for j in t))
    m.constraint(w, lambda i: dm.sum(assign[i, j] for j in t) == 1, name="one_task", fast=fast)
    m.constraint(t, lambda j: dm.sum(assign[i, j] for i in w) == 1, name="one_worker", fast=fast)
    return m


def _flat_n(m):
    return sum(v.size for v in m._variables)


def test_840_evaluator_sees_fast_constraints():
    """The evaluator's constraint count for a fast=True model matches the fast=False
    model (both 6) — the fast rows are no longer invisible."""
    ev_fast = cached_evaluator(_assignment(fast=True))
    ev_slow = cached_evaluator(_assignment(fast=False))
    assert ev_fast.n_constraints == ev_slow.n_constraints == 6, (
        f"#840: fast evaluator sees {ev_fast.n_constraints} constraints, "
        f"slow sees {ev_slow.n_constraints} (want 6 each)"
    )


def test_840_feasibility_check_rejects_infeasible_on_fast_model():
    """all-zeros violates every ``==1`` assignment row; the feasibility check (used by
    the #772 guard AND the trivial-primal seed) must now reject it on the fast=True
    model, not accept it as it did when the evaluator had no constraints."""
    ev = cached_evaluator(_assignment(fast=True))
    z = np.zeros(_flat_n(_assignment(fast=True)))
    assert _check_constraint_feasibility(ev, z, tol=1e-6) is False, (
        "#840: all-zeros wrongly accepted on a fast=True assignment model (guard blind spot)"
    )
    # a genuine permutation (w1->a, w2->b, w3->c) satisfies both families
    idx = {(w, t): k for k, (w, t) in enumerate((w, t) for w in _WORKERS for t in _TASKS)}
    perm = np.zeros(_flat_n(_assignment(fast=True)))
    for w, t in zip(_WORKERS, _TASKS):
        perm[idx[(w, t)]] = 1.0
    assert _check_constraint_feasibility(ev, perm, tol=1e-6) is True, (
        "#840: a valid permutation was rejected on the fast=True model"
    )


def test_840_fast_and_slow_solve_agree():
    """End-to-end: the fast and slow formulations certify the SAME optimum (the true
    assignment optimum, 10) — the fix does not perturb the correct solve, only makes
    verification faithful."""
    rf = _assignment(fast=True).solve()
    rs = _assignment(fast=False).solve()
    assert rf.objective == rs.objective, f"fast {rf.objective} != slow {rs.objective}"
    assert abs(rf.objective - 10.0) < 1e-6, f"#840: optimum drifted to {rf.objective}"


def _mixed_fast_nonlinear():
    """A fast=True indexed family mixed with a nonlinear constraint, so the model takes
    the spatial/nonlinear path (which materializes builder rows into ``_constraints``)."""
    m = dm.Model("mixed")
    w = m.set("w", _WORKERS)
    t = m.set("t", _TASKS)
    assign = m.binary("assign", over=w * t)
    c = m.continuous("c", lb=0, ub=5)
    m.minimize(dm.sum(_ACOST[(i, j)] * assign[i, j] for i in w for j in t))
    m.constraint(w, lambda i: dm.sum(assign[i, j] for j in t) == 1, name="one_task", fast=True)
    m.constraint(t, lambda j: dm.sum(assign[i, j] for i in w) == 1, name="one_worker", fast=True)
    m.subject_to(c * c <= 4, name="nl")
    return m


def test_840_no_double_count_after_materialization():
    """Regression: the fast rows must be counted EXACTLY once. When the nonlinear path
    materializes the builder blocks into ``_constraints``, the evaluator must still see
    the same total — not each fast row twice (the double-count a separate mirror would
    have caused)."""
    m = _mixed_fast_nonlinear()
    before = cached_evaluator(m).n_constraints
    assert before == 7, f"#840: pre-materialize evaluator sees {before} (want 7)"
    n = m._materialize_builder_linear_rows()
    assert n == 6, f"#840: materialized {n} rows (want 6)"
    after = cached_evaluator(m).n_constraints
    assert after == 7, f"#840: post-materialize evaluator sees {after} (want 7 — double-counted?)"
    # Idempotent: a second materialization is a no-op and does not re-inflate the count.
    assert m._materialize_builder_linear_rows() == 0
    assert cached_evaluator(m).n_constraints == 7


def _direct_matrix_model():
    """A model whose only linear constraint is added through the direct
    ``add_linear_constraints`` matrix API (no Constraint object, no fast-indexed
    family), plus a nonlinear constraint to force the spatial path."""
    m = dm.Model("direct")
    x = m.continuous("x", shape=(3,), lb=0, ub=5)
    m.minimize(x[0] + x[1] + x[2])
    A = sp.csr_matrix(np.array([[1.0, 1.0, 1.0]]))
    m.add_linear_constraints(A, x, "==", np.array([3.0]), "sumeq")
    m.subject_to(x[0] * x[1] <= 4, name="nl")
    return m


def test_840_direct_matrix_rows_visible_to_guard():
    """The direct ``add_linear_constraints`` matrix API also bypasses ``_constraints``;
    the guard/feasibility evaluator (built on the pristine model BEFORE the solver
    materializes anything) must still see those rows, or all-zeros — which violates
    ``x0+x1+x2==3`` — would be wrongly accepted as feasible (the same false-primal class
    as the fast-indexed hole)."""
    m = _direct_matrix_model()
    ev = cached_evaluator(m)
    assert ev.n_constraints == 2, f"#840: evaluator sees {ev.n_constraints} (want 2: nl + sumeq)"
    assert _check_constraint_feasibility(ev, np.zeros(3), tol=1e-6) is False, (
        "#840: all-zeros wrongly accepted — the direct-matrix row is invisible to the guard"
    )
    # a point on the equality plane inside the nonlinear region is feasible
    assert _check_constraint_feasibility(ev, np.array([1.0, 1.0, 1.0]), tol=1e-6) is True


def test_840_qubo_not_fooled_by_builder_constraints():
    """A QUBO is UNCONSTRAINED. A binary-quadratic model carrying a builder-resident
    linear row must NOT be misclassified as a QUBO (which would let the QUBO local
    search treat any binary point as feasible, ignoring the constraint)."""
    m = dm.Model("q")
    b = m.binary("b", shape=(3,))
    m.minimize(b[0] * b[1] + b[2])
    A = sp.csr_matrix(np.array([[1.0, 1.0, 1.0]]))
    m.add_linear_constraints(A, b, "<=", np.array([2.0]), "card")
    assert is_qubo(m) is False, "#840: a constrained binary-quadratic model misread as QUBO"
