"""``DISCOPT_CONVEX_NODE_KEEP_PROVED_BOUND``: keep a convex node's proved bound when
it is ABOVE the NLP objective, and re-cap the published bound at the exact objective.

Found through ``test_2262_cmir_integer_bounds.py::test_issue_2262_nlp_bb_certifies_
the_true_optimum``, which certified locally and not on CI. Measured on that instance
(convex IQP, ``nlp_bb=True``): the integral node that closes the search has NLP
objective -6.72e-7 (CI: -1.46e-6) at a point 3e-9 outside its box, and #2067 proved
bound -3.78e-10. #2067 imports ``min(objective, proved)``, the node becomes the
incumbent at that value, the exhausted tree's dual bound IS the incumbent value, and
``Model.solve`` re-values the objective exactly (0). The published gap was the NLP's
error, and it crossed the 1e-6 absolute tolerance on CI.

These tests check:

* the tree contract through the binding: a proved bound above the point value keeps
  the incumbent at the point's value, leaves the capped ``global_lower_bound`` as it
  was, reports the proved bound uncapped, and does not trip #1296 -- while an
  ancestor contradiction is still caught;
* end to end, the CI failure reproduced by shifting every node objective down (the
  shift is how an NLP with a looser terminal residual reads): OFF loses the
  certificate, ON keeps it at the proved bound;
* soundness, CLAUDE.md §5 (differential + feasible-point sampling): on random convex
  IQPs and random integer sub-boxes, the ON bound is never below the OFF bound and
  never above the enumerated optimum, at any shift.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import discopt.solver as S
import numpy as np
import pytest
from discopt import Model
from discopt._rust import PyTreeManager

sys.path.insert(0, str(Path(__file__).parent))
import test_2262_cmir_integer_bounds as T2262  # noqa: E402

FLAG = "DISCOPT_CONVEX_NODE_KEEP_PROVED_BOUND"
# Captured once: ``_shifted_solve`` must wrap THIS, not whatever the attribute is
# when it runs -- ``monkeypatch`` undoes only at test end, so wrapping the current
# attribute stacked one shift per solve and compared arms at different shifts.
_ORIG_APPLY = S._apply_convex_node_safe_bounds


# ── the tree contract ────────────────────────────────────────────────────────


def _branched_tree():
    """A 1-variable integer tree on [0, 10], root bound 3.0, branched: two children."""
    tree = PyTreeManager(1, [0.0], [10.0], [0], [1], "best_first")
    tree.initialize()
    _, _, root, _ = tree.export_batch(1)
    tree.import_results(
        np.asarray(root, dtype=np.int64),
        np.array([3.0]),
        np.array([[3.5]]),
        np.array([False]),
    )
    tree.process_evaluated()
    _, _, ids, _ = tree.export_batch(2)
    ids = np.asarray(ids, dtype=np.int64)
    assert ids.size == 2
    return tree, ids


def _close_both(bound: float, pv: float):
    tree, ids = _branched_tree()
    tree.import_results(
        ids,
        np.array([bound, 9.0]),
        np.array([[3.0], [4.0]]),
        np.array([True, True]),
        point_values=np.array([pv, np.nan]),
    )
    tree.process_evaluated()
    stats = tree.stats()
    assert stats["open_nodes"] == 0
    return tree, stats


def test_tree_reports_the_proved_bound_uncapped():
    pv = 3.5 - 1e-5  # the point is valued below what its box proved
    tree, stats = _close_both(3.5, pv)
    assert tree.incumbent()[1] == pv
    assert stats["global_lower_bound"] == pv  # capped at the incumbent, as before
    assert stats["proved_lower_bound"] == 3.5  # min(closed 3.5, pruned sibling 9.0)
    assert not stats["bound_contradicted"]  # ancestor bound 3.0 < pv


def test_tree_still_catches_an_ancestor_contradiction():
    _, stats = _close_both(3.5, 2.0)  # a point at 2.0 under an ancestor bound of 3.0
    assert stats["bound_contradicted"]


def test_tree_proved_bound_equals_capped_bound_on_the_legacy_import():
    for bound, pv in [(3.5, np.nan), (3.5, 3.5 + 2e-7)]:
        _, stats = _close_both(bound, pv)
        assert stats["proved_lower_bound"] == stats["global_lower_bound"] == 3.5


def test_flag_is_default_on_with_an_opt_out(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert S._convex_node_keep_proved_bound_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not S._convex_node_keep_proved_bound_enabled()


# ── the re-cap at the exact objective (Model._reconcile_objective_with_model) ──


def _reconciled(proved, internal_obj=-2e-6, internal_bound=-2e-6):
    """A certified result whose internal objective sits ``-internal_obj`` below the
    exact value at its point (0), as on the #2262 instance, reconciled."""
    from discopt.modeling.core import SolveResult

    m = Model("recap")
    x = m.integer("x", lb=-2, ub=2)
    m.minimize(x * x)
    r = SolveResult(
        status="optimal",
        objective=internal_obj,
        bound=internal_bound,
        gap=0.0,
        x={"x": np.array(0.0)},
        gap_certified=True,
        _proved_bound=proved,
    )
    m._reconcile_objective_with_model(r)
    return r


def test_recap_keeps_a_certificate_the_proved_bound_supports():
    r = _reconciled(proved=-3.8e-10)
    assert r.objective == 0.0
    assert r.bound == -3.8e-10
    assert r.status == "optimal" and r.gap_certified


def test_recap_never_publishes_a_bound_above_the_objective():
    r = _reconciled(proved=5.0)  # proved above the exact value: capped at it
    assert r.bound == 0.0 and r.objective == 0.0


def test_without_a_proved_bound_the_certificate_is_withheld_as_before():
    r = _reconciled(proved=None)
    assert r.objective == 0.0 and r.bound == -2e-6
    assert r.status == "feasible" and not r.gap_certified


# ── end to end: the #2262 instance under node-objective noise ────────────────


def _shifted_solve(monkeypatch, model: Model, delta: float, on: bool, **kw):
    """Solve with every finite node NLP objective shifted down by ``delta`` before
    the #2067 step. Returns ``(result, n_shifted)``."""
    monkeypatch.setenv(FLAG, "1" if on else "0")
    orig = _ORIG_APPLY
    shifted = [0]

    def shift(ev, blb, bub, lbs, sols, cl, cu, **k):
        for i in range(len(lbs)):
            if np.isfinite(lbs[i]) and lbs[i] < S._SENTINEL_THRESHOLD:
                lbs[i] -= delta
                shifted[0] += 1
        return orig(ev, blb, bub, lbs, sols, cl, cu, **k)

    monkeypatch.setattr(S, "_apply_convex_node_safe_bounds", shift)
    # ``deterministic``: without it the same model solved repeatedly in one process
    # measured 64, 72 and 43 nodes with two different bounds (wall-clock heuristic
    # budgets), so an ON/OFF comparison would compare two noisy samples, not the flag.
    r = model.solve(nlp_bb=True, deterministic=True, **kw)
    return r, shifted[0]


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.parametrize("delta", [8e-7, 1e-4])
def test_issue_2262_certificate_survives_node_objective_noise(monkeypatch, delta):
    """8e-7 reproduces CI (bound -1.47e-6 vs CI's -1.46e-6); OFF loses the
    certificate there, ON keeps it at the proved bound."""
    off, n_off = _shifted_solve(monkeypatch, T2262._issue_model(), delta, False, time_limit=20)
    on, n_on = _shifted_solve(monkeypatch, T2262._issue_model(), delta, True, time_limit=20)
    assert n_off > 0 and n_on > 0  # the shift fired (CLAUDE.md §6)
    assert off.status == "feasible", (off.status, off.bound)  # the probe measures the defect
    assert on.status == "optimal" and on.gap_certified, (on.status, on.bound)
    assert on.objective == 0.0
    assert off.bound <= on.bound <= 0.0, (off.bound, on.bound)


# ── §5: differential bound test + feasible-point sampling ───────────────────


def _random_convex_iqp(seed: int, box=None):
    """``min x'Qx + c'x  s.t.  A x <= b`` over integers; Q PSD. ``box`` overrides
    the integer bounds per column (a fixed sub-box). Returns (model, oracle)."""
    rng = np.random.default_rng(seed)
    n = 2 if seed % 2 else 3
    B = rng.normal(size=(n, n))
    Q = B @ B.T + 0.1 * np.eye(n)
    c = rng.normal(size=n) * 3
    A = rng.normal(size=(2, n))
    b = np.abs(rng.normal(size=2)) + 0.3
    # The full box: [-2, 2]^2 (the #2262 shape) or [-1, 1]^3 -- [-2, 2]^3 measured up
    # to 43 s per ON/OFF pair locally, past the slow lane's per-test budget on CI.
    full = 2 if n == 2 else 1
    lo, hi = (np.full(n, -full), np.full(n, full)) if box is None else box
    m = Model(f"iqp{seed}")
    x = m.integer("x", shape=(n,), lb=lo, ub=hi)
    m.minimize(
        sum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n))
        + sum(c[i] * x[i] for i in range(n))
    )
    for k in range(2):
        m.subject_to(sum(A[k, i] * x[i] for i in range(n)) <= b[k])
    best = np.inf
    for p in itertools.product(*[range(int(lo[i]), int(hi[i]) + 1) for i in range(n)]):
        p = np.asarray(p, float)
        if np.all(A @ p <= b + 1e-12):
            best = min(best, float(p @ Q @ p + c @ p))
    return m, best


def _box(seed: int, sub: bool):
    """The full box, or a random integer sub-box (deterministic per seed)."""
    if not sub:
        return None
    rng = np.random.default_rng([2262, seed])
    n = 2 if seed % 2 else 3
    lo = rng.integers(-2, 1, size=n)
    return lo, lo + rng.integers(1, 3, size=n)


# Built at collection, keeping only (seed, box) pairs with a feasible integer point:
# an infeasible sub-box has nothing to sample and could only ever skip.
_CASES = [
    (seed, sub)
    for seed in range(4)
    for sub in (False, True)
    if np.isfinite(_random_convex_iqp(seed, _box(seed, sub))[1])
]


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.parametrize("seed,sub", _CASES)
def test_on_bound_is_between_the_off_bound_and_the_enumerated_optimum(monkeypatch, seed, sub):
    box = _box(seed, sub)
    checked = 0
    for delta in (0.0, 1e-4):
        m_off, opt = _random_convex_iqp(seed, box)
        off, n_off = _shifted_solve(monkeypatch, m_off, delta, False, time_limit=30)
        m_on, _ = _random_convex_iqp(seed, box)
        on, n_on = _shifted_solve(monkeypatch, m_on, delta, True, time_limit=30)
        tag = (seed, sub, delta, off.status, on.status, off.bound, on.bound, opt)
        assert n_off > 0 and n_on > 0, tag
        assert off.bound is not None and on.bound is not None, tag
        tol = 1e-9 * (1 + abs(opt))
        # The enumerated optimum is the minimum over every feasible integer point
        # of the box: a bound above it would cut a valid point.
        assert on.bound <= opt + tol, tag
        assert on.bound >= off.bound - tol, tag
        assert abs(on.objective - opt) <= 1e-6 * (1 + abs(opt)), tag
        if off.status == "optimal":
            assert on.status == "optimal", tag  # never loses a certificate
        checked += 1
    assert checked == 2  # rule 6: both shifts compared
