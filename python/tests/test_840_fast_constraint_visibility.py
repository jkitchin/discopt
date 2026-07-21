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

The fix tracks the fast families in ``model._fast_constraints`` and the NLPEvaluator
includes them, so its constraint view is complete.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
from discopt._jax.nlp_evaluator import cached_evaluator
from discopt._jax.primal_heuristics import _check_constraint_feasibility

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
