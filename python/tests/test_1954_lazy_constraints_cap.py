"""#1954: lazy constraints must reach the node LP, survive the pool, and never fake a certificate.

Three defects combined into wrong certified optima on models with a correct separator:

1. **The node LP never saw a lazy row.** Lazy rows reached only the node NLP, so the
   McCormick LP kept returning the point the row cuts off, the node was requeued to
   ``_LAZY_REQUEUE_CAP`` and then excluded.
2. **The cap exclusion kept the certificate.** A lazy cut rejects one point; it proves
   nothing about the rest of the box (#1365). Excluding the node dropped its region from
   the bound and the solve certified -2.0 on a model whose optimum is -2.9.
3. **``CutPool`` deduplicated lazy rows unsoundly.** Keys were bare ``hash`` values
   (``hash(-1.0) == hash(-2.0)``) of 8-decimal-rounded rows, and protected model rows
   could be purged like generated cuts.
"""

from __future__ import annotations

import discopt
import discopt.solver as solver_mod
import numpy as np
import pytest
from discopt._relax.cutting_planes import CutPool, LinearCut
from discopt.callbacks import CutResult


def _basic():
    m = discopt.Model("lazy_cap")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.subject_to(x * y <= 1)
    m.minimize(-(x + y))

    def lazy(ctx, model):
        xv, yv = float(ctx.x_relaxation[0]), float(ctx.x_relaxation[1])
        if xv > 2.5 + 1e-7:
            return [CutResult(terms=[(x, 1.0)], sense="<=", rhs=2.5)]
        if yv > 2.5 + 1e-7:
            return [CutResult(terms=[(y, 1.0)], sense="<=", rhs=2.5)]
        return []

    return m, lazy, -2.9


def _scaled():
    # Coefficients 1e-9 and 2e-9 round to the same 8-decimal row: the second lazy row
    # was dropped as a "duplicate" of the first.
    S = 1e9
    m = discopt.Model("scaled_lazy")
    x = m.continuous("x", lb=0, ub=3 * S)
    y = m.continuous("y", lb=0, ub=3)
    m.subject_to((x / S) * y <= 1)
    m.minimize(-(x / S + y))

    def lazy(ctx, model):
        xv, yv = map(float, ctx.x_relaxation[:2])
        if xv / S + yv > 2.5 + 1e-7:
            return [CutResult([(x, 1 / S), (y, 1.0)], "<=", 2.5)]
        if 2 * xv / S + yv > 2.5 + 1e-7:
            return [CutResult([(x, 2 / S), (y, 1.0)], "<=", 2.5)]
        return []

    return m, lazy, -2.5


@pytest.mark.smoke
@pytest.mark.parametrize("build", [_basic, _scaled], ids=["basic", "scaled"])
def test_separating_lazy_rows_certify_true_optimum(build):
    m, lazy, opt = build()
    r = m.solve(lazy_constraints=lazy, time_limit=30, threads=1)
    # Minimisation: a certified bound above the true optimum is a false certificate.
    assert r.bound <= opt + 1e-6, (r.status, r.objective, r.bound)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(opt, abs=1e-5)


@pytest.mark.smoke
def test_requeue_cap_withdraws_the_certificate(monkeypatch):
    # A "separator" that rejects every point with x > 2.5 but returns a row the point
    # already satisfies: it can never make progress, so the node hits the cap and is
    # excluded without proof. That must not certify anything over its region.
    monkeypatch.setattr(solver_mod, "_LAZY_REQUEUE_CAP", 3)
    # Asymmetric objective: the only points better than the accepted ones (x <= 2.5,
    # best -(2.5 + 0.5*0.4) = -2.7) lie in the rejected region (x = 3, y = 1/3 gives
    # -3.1667), so the excluded node's bound keeps the gap open.
    m = discopt.Model("lazy_noop")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.subject_to(x * y <= 1)
    m.minimize(-(x + 0.5 * y))

    def lazy(ctx, model):
        if float(ctx.x_relaxation[0]) > 2.5 + 1e-7:
            return [CutResult(terms=[(x, 1.0)], sense="<=", rhs=10.0)]
        return []

    r = m.solve(lazy_constraints=lazy, time_limit=30, threads=1)
    assert not r.gap_certified, (r.status, r.objective, r.bound)
    assert r.status != "optimal"
    assert r.bound <= -3.1666 + 1e-6  # the rejected region stays in the reported bound


@pytest.mark.smoke
def test_protected_rows_are_keyed_exactly():
    pool = CutPool()
    a = LinearCut(np.array([1e-9, 1.0]), 2.5, "<=")
    b = LinearCut(np.array([2e-9, 1.0]), 2.5, "<=")
    assert pool.add(a, protected=True)
    assert pool.add(b, protected=True)
    assert pool.add(LinearCut(np.array([1e-9, 1.0]), 2.5, "<="), protected=True) is False
    assert len(pool.protected_rows()) == 2


@pytest.mark.smoke
def test_rhs_hash_collision_is_not_a_duplicate():
    # hash(-1.0) == hash(-2.0) in CPython; the old set held only hashes.
    assert hash(-1.0) == hash(-2.0)
    for protected in (False, True):
        pool = CutPool()
        assert pool.add(LinearCut(np.array([1.0]), -1.0, "<="), protected=protected)
        assert pool.add(LinearCut(np.array([1.0]), -2.0, "<="), protected=protected)
        assert len(pool) == 2


@pytest.mark.smoke
def test_protected_rows_survive_both_purges():
    pool = CutPool(max_cuts=4, purge_fraction=0.5)
    lazy = LinearCut(np.array([1.0, 0.0]), 1.0, "<=")
    pool.add(lazy, protected=True)
    for k in range(10):
        pool.add(LinearCut(np.array([0.0, 1.0]), float(k), "<="))
    assert lazy in pool.protected_rows()
    for _ in range(20):
        pool.age_cuts(np.array([-100.0, -100.0]))  # nothing binding
    pool.purge_inactive(max_age=5)
    assert len(pool) == 1 and pool.protected_rows() == [lazy]
    # A pool of only model rows may exceed max_cuts rather than drop one.
    for k in range(6):
        pool.add(LinearCut(np.array([float(k + 2), 1.0]), 0.0, "<="), protected=True)
    assert len(pool.protected_rows()) == 7


@pytest.mark.smoke
def test_lazy_rows_upper_form():
    cuts = [
        LinearCut(np.array([1.0, 2.0]), 3.0, "<="),
        LinearCut(np.array([1.0, 0.0]), 1.0, ">="),
        LinearCut(np.array([0.0, 1.0]), 2.0, "=="),
    ]
    A, b = solver_mod._lazy_rows_upper_form(cuts, 2)
    np.testing.assert_array_equal(A, [[1, 2], [-1, 0], [0, 1], [0, -1]])
    np.testing.assert_array_equal(b, [3, -1, 2, -2])
