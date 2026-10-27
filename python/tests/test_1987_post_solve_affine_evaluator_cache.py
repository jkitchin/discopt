"""#1987: a small MILP's ``solve()`` spent most of its time rebuilding one affine map.

Three post-solve consumers (objective reconciliation, incumbent repair, the
certified-incumbent backstop) each called ``AffineEvaluator.try_build``, and each
build walked every row with its own memo, rebuilding the shared variable and
element nodes as sparse matrices once per row. On the issue's facility-location
MILP (488 columns, 572 rows, HiGHS route, 1 node) that was ~1.4 s of a 2.2 s
solve. The evaluator is now memoized per model structure and one walk memo is
shared across a model's rows. Both are bound-neutral: the maps are identical.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import _affine_evaluator as ae
from discopt._relax.affine_array import affine_of

pytestmark = pytest.mark.smoke


def _milp(n_w: int = 4, n_c: int = 6):
    rng = np.random.default_rng(3)
    cost = rng.uniform(1, 5, (n_w, n_c))
    dem = rng.uniform(1, 4, n_c)
    m = dm.Model("fl")
    y = m.binary("y", shape=(n_w,))
    x = m.continuous("x", shape=(n_w, n_c), lb=0, ub=10)
    m.minimize(
        dm.sum(lambda i: 3.0 * y[i], over=range(n_w))
        + dm.sum(
            lambda i: dm.sum(lambda j: float(cost[i, j]) * x[i, j], over=range(n_c)),
            over=range(n_w),
        )
    )
    for j in range(n_c):
        m.subject_to(dm.sum(lambda i: x[i, j], over=range(n_w)) >= float(dem[j]))
    for i in range(n_w):
        for j in range(n_c):
            m.subject_to(x[i, j] <= float(dem[j]) * y[i])
    return m


@pytest.fixture
def builds(monkeypatch):
    calls = []
    real = ae.AffineEvaluator.try_build.__func__

    def counted(cls, model):
        calls.append(model)
        return real(cls, model)

    monkeypatch.setattr(ae.AffineEvaluator, "try_build", classmethod(counted))
    return calls


def test_one_build_per_solve(builds):
    m = _milp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve()
    assert r.status == "optimal"
    assert len(builds) == 1, f"the post-solve evaluator was built {len(builds)} times"


def test_cache_follows_structure(builds):
    m = dm.Model("s")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.subject_to(x >= 2)
    ev = ae.post_solve_evaluator(m)
    assert ae.post_solve_evaluator(m) is ev and len(builds) == 1
    m.subject_to(x <= 7)  # a new row: a new fingerprint, a new build
    ev2 = ae.post_solve_evaluator(m)
    assert ev2 is not ev and ev2.n_constraints == 2 and len(builds) == 2
    m.minimize(2 * x)  # a new objective
    ev3 = ae.post_solve_evaluator(m)
    assert ev3 is not ev2 and ev3.evaluate_objective(np.array([3.0])) == pytest.approx(6.0)


def test_parametric_model_is_declined():
    """The cache key leaves out ``Parameter.value``; that is safe only because a
    model with a parameter in a row is never given an AffineEvaluator."""
    m = dm.Model("p")
    x = m.continuous("x", lb=0, ub=10)
    p = m.parameter("p", value=2.0)
    m.minimize(x)
    m.subject_to(x >= p)
    assert ae.AffineEvaluator.try_build(m) is None
    assert ae._affine_evaluator(m) is None


def test_a_declined_build_is_cached_too(builds):
    m = dm.Model("nl")
    x = m.continuous("x", lb=0.5, ub=2)
    m.minimize(x)
    m.subject_to(x * x >= 1)
    for _ in range(3):
        assert ae._affine_evaluator(m) is None
    assert len(builds) == 1


def test_shared_memo_gives_the_same_maps():
    m = _milp()
    n = sum(v.size for v in m._variables)
    memo: dict = {}
    for con in m._constraints:
        alone = affine_of(con.body, m._flat_var_offset, n)
        shared = affine_of(con.body, m._flat_var_offset, n, memo=memo)
        assert np.array_equal(alone.b, shared.b)
        assert (alone.M != shared.M).nnz == 0


def test_scalar_scale_fast_path_is_exact():
    m = dm.Model("s")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    n = 3
    for c in (0.1, -3.7, 1e-300, 7.0):
        a = affine_of(c * x[1], m._flat_var_offset, n)
        assert a.M.toarray().tolist() == [[0.0, c, 0.0]]
