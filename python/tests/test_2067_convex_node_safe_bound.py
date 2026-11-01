"""#2067: a convex node's NLP objective is not a bound; NLP-BB imports a proved one.

QPLIB_8153 (convex MINLP) certified a tree bound 1.57e-7 past a reference point
feasible to 2.7e-15. The bound came from NLP-B&B, which imported each convex node's
NLP *objective* -- the value of an approximate minimiser, above the relaxation's
minimum by the interior-point method's terminal residuals -- as the node's lower
bound. The root alone sat 1.49e-7 past the reference.

``discopt.solvers.convex_node_bound`` replaces that by a Neumaier-Shcherbina bound
on the node's linearised LP, which is a relaxation of the node for a certified-convex
model at ANY point of the box. These tests check:

* validity by feasible-point sampling on random convex QCQPs, at linearisation points
  drawn anywhere in the box (the bound may never exceed a feasible point's objective);
* tightness at a KKT point (the bound must not cost a certificate where the NLP is
  accurate);
* abstention (``None``) where the evaluator is not finite, never a guess;
* the tree contract: the proved bound floors the global bound while the point's
  objective stays the incumbent value;
* end to end on the vendored QPLIB_8153: the NLP-B&B root bound no longer crosses the
  reference, and the ``DISCOPT_CONVEX_NODE_SAFE_BOUND=0`` opt-out still does (so the
  probe measures the defect, not nothing).
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._rust import PyTreeManager
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.solvers.convex_node_bound import convex_node_safe_bound
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

_DATA = Path(__file__).parent / "data" / "issue_2067"


def _random_convex_qcqp(seed: int):
    """``min x'Qx + c'x  s.t.  x'P_k x + a_k'x <= r_k (k=1,2),  sum x == 1`` on a box.

    Q and P_k are PSD by construction, so every row is a convex function below a
    finite side -- the shape ``classify_model`` certifies. ``r_k`` is set so the
    box centre is strictly feasible.
    """
    rng = np.random.default_rng(seed)
    n = 4
    m = dm.Model(f"qcqp{seed}")
    x = m.continuous("x", shape=(n,), lb=-1.0, ub=2.0)
    centre = np.full(n, 0.25)

    def psd():
        B = rng.normal(size=(n, n))
        return B @ B.T / n

    Q, c = psd(), rng.normal(size=n)
    m.minimize(sum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n)) + c @ x)
    rows = []
    for _ in range(2):
        P, a = psd(), rng.normal(size=n)
        r = float(centre @ P @ centre + a @ centre) + 0.5
        m.subject_to(sum(P[i, j] * x[i] * x[j] for i in range(n) for j in range(n)) + a @ x <= r)
        rows.append((P, a, r))
    m.subject_to(sum(x[i] for i in range(n)) == 1.0)

    def feasible(z):
        return abs(z.sum() - 1.0) < 1e-12 and all(z @ P @ z + a @ z <= r for P, a, r in rows)

    return m, n, (lambda z: float(z @ Q @ z + c @ z)), feasible


def _bound_at(m, point, lo, hi):
    ev = make_evaluator(m)
    cl, cu = _infer_constraint_bounds(ev)
    return convex_node_safe_bound(ev, point, lo, hi, np.asarray(cl), np.asarray(cu))


def test_safe_bound_never_exceeds_a_feasible_point():
    """Feasible-point sampling: whatever point the bound is linearised at, no
    feasible point of the box may lie below it (CLAUDE.md §5)."""
    checked = 0
    for seed in range(6):
        m, n, f, feasible = _random_convex_qcqp(seed)
        rng = np.random.default_rng(100 + seed)
        lo, hi = np.full(n, -1.0), np.full(n, 2.0)
        # Feasible samples: random box points projected onto sum == 1, kept if the
        # quadratic rows hold.
        samples = []
        for _ in range(4000):
            z = rng.uniform(-1.0, 2.0, size=n)
            z = z + (1.0 - z.sum()) / n
            if np.all(z >= lo) and np.all(z <= hi) and feasible(z):
                samples.append(z)
        assert len(samples) > 50, f"seed {seed}: too few feasible samples"
        best = min(f(z) for z in samples)
        for _ in range(8):
            point = rng.uniform(-1.0, 2.0, size=n)  # need not be feasible
            bound, why = _bound_at(m, point, lo, hi)
            assert bound is not None, why
            assert bound <= best, f"seed {seed}: bound {bound} above feasible {best}"
            checked += 1
    assert checked == 48


def test_safe_bound_is_tight_at_the_optimum():
    """At a KKT point the linearised LP's optimum is the NLP's, so the bound costs
    only the NLP's own tolerance -- and still sits at or below the optimum."""
    checked = 0
    for seed in range(4):
        m, n, f, _ = _random_convex_qcqp(seed)
        r = m.solve()
        assert r.status == "optimal"
        xs = np.array([r.x[v.name] for v in m._variables]).ravel()
        bound, why = _bound_at(m, xs, np.full(n, -1.0), np.full(n, 2.0))
        assert bound is not None, why
        assert r.objective - 1e-5 <= bound <= r.objective + 1e-9 * max(1.0, abs(r.objective))
        checked += 1
    assert checked == 4


def test_rust_simplex_dual_matches_highs_where_highspy_is_missing(monkeypatch):
    """The browser build has no highspy: the dual then comes from the in-house
    simplex, and must give the same tight, valid bound as HiGHS's."""
    import discopt.solvers.lp_milp_highs as lmh

    checked = 0
    for seed in range(3):
        m, n, _, _ = _random_convex_qcqp(seed)
        r = m.solve()
        xs = np.array([r.x[v.name] for v in m._variables]).ravel()
        box = (np.full(n, -1.0), np.full(n, 2.0))
        with_highs, why = _bound_at(m, xs, *box)
        assert with_highs is not None, why

        def _missing():
            raise lmh.HighsUnavailable("blocked for the test")

        with monkeypatch.context() as mp:
            mp.setattr(lmh, "require_highspy", _missing)
            rust, why = _bound_at(m, xs, *box)
        assert rust is not None, why
        assert rust <= r.objective + 1e-9 * max(1.0, abs(r.objective))
        assert rust == pytest.approx(with_highs, abs=1e-6)
        checked += 1
    assert checked == 3


def test_a_row_highs_refuses_still_gets_a_bound():
    """HiGHS rejects ``|a_ij| >= 1e15`` at ``passModel``; the row-equilibrated retry
    must still produce a valid bound (#1955's row, where the only node otherwise
    abstained and the certificate was lost)."""
    m = dm.Model("bigrow")
    x0 = m.continuous("x0", lb=0, ub=1)
    x1 = m.continuous("x1", lb=1, ub=2)
    x2 = m.continuous("x2", lb=0, ub=1)
    x3 = m.continuous("x3", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=1)
    m.subject_to(x0 + 1e16 * x1 - x2 - 1e16 * x3 <= 0)
    m.subject_to(dm.exp(z) <= 2.5)
    m.minimize(-x0 - z)
    exact = -1.0 - np.log(2.5)  # x0 = x1 = x2 = x3 = 1, z = log 2.5
    point = np.array([1.0, 1.0, 1.0, 1.0, np.log(2.5)])
    bound, why = _bound_at(m, point, np.array([0, 1, 0, 0, 0.0]), np.array([1, 2, 1, 1, 1.0]))
    assert bound is not None, why
    assert bound <= exact


def _norm_cone_model():
    """``min t - x0  s.t.  ||x|| <= t`` on a box: optimum 0 (any ``x = (s, 0), t = s``)."""
    m = dm.Model("cone")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    t = m.continuous("t", lb=0.0, ub=2.0)
    m.minimize(t - x[0])
    m.subject_to(dm.sqrt(x[0] * x[0] + x[1] * x[1]) <= t)
    return m, np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 2.0])


def test_safe_bound_at_the_cone_apex_is_still_valid():
    """At the apex the norm has no derivative; the evaluator returns 0, a valid
    subgradient, so the bound must still sit at or below the optimum 0."""
    m, lo, hi = _norm_cone_model()
    bound, why = _bound_at(m, np.zeros(3), lo, hi)
    assert bound is not None, why
    assert bound <= 0.0


def test_safe_bound_abstains_on_a_non_finite_point():
    """No finite point, no bound -- and never a guessed one."""
    m, lo, hi = _norm_cone_model()
    bound, why = _bound_at(m, np.array([np.nan, 0.0, 1.0]), lo, hi)
    assert bound is None
    assert "not a finite vector" in why


def _branched_tree():
    """Root at ``[0, 10]`` (one integer), branched; both children exported."""
    tree = PyTreeManager(1, [0.0], [10.0], [0], [1], "best_first")
    tree.initialize()
    _, _, ids, _ = tree.export_batch(1)
    tree.import_results(
        np.asarray(ids, dtype=np.int64),
        np.array([3.0]),
        np.array([[3.5]]),
        np.array([False]),
    )
    tree.process_evaluated()
    _, _, ids, _ = tree.export_batch(2)
    return tree, np.asarray(ids, dtype=np.int64)


def test_tree_keeps_the_proved_bound_below_the_point_value():
    """The import contract: an integral node's PROVED bound floors the global
    bound, while the incumbent records the point's own objective."""
    tree, ids = _branched_tree()
    assert ids.size == 2
    tree.import_results(
        ids,
        np.array([3.5, 9.0]),
        np.array([[3.0], [4.0]]),
        np.array([True, True]),
        point_values=np.array([3.5 + 2e-7, np.nan]),
    )
    tree.process_evaluated()
    stats = tree.stats()
    assert stats["open_nodes"] == 0
    assert tree.incumbent()[1] == 3.5 + 2e-7
    assert stats["safe_fathom_floor"] == 3.5
    assert stats["global_lower_bound"] == 3.5
    with pytest.raises(ValueError, match="point values"):
        tree2, ids2 = _branched_tree()
        tree2.import_results(
            ids2,
            np.array([3.5, 9.0]),
            np.array([[3.0], [4.0]]),
            np.array([True, True]),
            point_values=np.array([1.0]),
        )


def _qplib_8153_root(monkeypatch, *, safe: bool):
    from discopt.interfaces import qplib

    monkeypatch.setenv("DISCOPT_CONVEX_NODE_SAFE_BOUND", "1" if safe else "0")
    inst = qplib.read_qplib(str(_DATA / "QPLIB_8153.qplib"))
    _, ref = qplib.read_solution(str(_DATA / "QPLIB_8153.sol"), inst)
    assert ref is not None
    r = qplib.to_model(inst).solve(nlp_bb=True, max_nodes=1, time_limit=60)
    assert r.bound is not None and np.isfinite(r.bound)
    return r, float(ref)


@pytest.mark.slow
@pytest.mark.correctness
def test_qplib_8153_root_bound_does_not_cross_the_reference(monkeypatch):
    """The issue's instance: the NLP-B&B root bound is at or below the reference
    optimum (it was 1.49e-7 above it)."""
    r, ref = _qplib_8153_root(monkeypatch, safe=True)
    assert r.bound <= ref, f"bound {r.bound!r} above the reference {ref!r}"
    assert (r.solver_stats or {}).get("nlpbb/convex_safe_bound_nodes", 0.0) >= 1.0


@pytest.mark.slow
@pytest.mark.correctness
def test_qplib_8153_legacy_import_still_crosses(monkeypatch):
    """Sensitivity of the test above (CLAUDE.md §6): with the opt-out the root bound
    is the NLP objective and crosses the reference, so a pass above is the fix and
    not an instance that never crosses."""
    r, ref = _qplib_8153_root(monkeypatch, safe=False)
    assert r.bound > ref


def test_budget_rations_the_exact_rung_and_suspends_a_failing_ladder():
    """#2070 review: one budget per solve, two deterministic counts and no clock.
    A node that abstains under the budget proves nothing either way, so suspension
    can only cost bound strength, never soundness."""
    from discopt.solvers.convex_node_bound import SafeBoundBudget

    m, lo, hi = _norm_cone_model()
    ev = make_evaluator(m)
    cl, cu = (np.asarray(v) for v in _infer_constraint_bounds(ev))
    budget = SafeBoundBudget()
    bad = np.array([np.nan, 0.0, 1.0])
    for _ in range(SafeBoundBudget.SUSPEND_AFTER - 1):
        assert convex_node_safe_bound(ev, bad, lo, hi, cl, cu, budget=budget)[0] is None
    assert not budget.suspended
    # A proved node resets the run of failures.
    good, why = convex_node_safe_bound(ev, np.zeros(3), lo, hi, cl, cu, budget=budget)
    assert good is not None, why
    assert budget.consecutive_failures == 0
    for _ in range(SafeBoundBudget.SUSPEND_AFTER):
        convex_node_safe_bound(ev, bad, lo, hi, cl, cu, budget=budget)
    assert budget.suspended
    bound, why = convex_node_safe_bound(ev, np.zeros(3), lo, hi, cl, cu, budget=budget)
    assert bound is None and "suspended" in why
    # The exact-rung ration never goes negative.
    assert 0 <= budget.exact_left <= SafeBoundBudget.EXACT_CALLS
