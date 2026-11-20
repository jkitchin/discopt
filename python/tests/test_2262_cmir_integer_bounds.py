"""#2262: c-MIR complemented integer columns against bounds that were not integral.

``nlp_bb=True`` certified 332.28 on a 2-variable convex integer QP whose optimum is
0 at x = (0, 0). The NLP-BB root-cut stage (#781) hands c-MIR the box from interval
FBBT over the linear rows, and that box is fractional on integer columns (``x1 <=
0.445``). c-MIR substitutes ``y = u - x``, which is an integer only for an integral
``u``, so the MIR rounding was invalid and cut off the optimum. The same caller also
substituted 0 / 1e5 for a missing bound, so a cut was valid only inside a box the
model never declared.

The fix is in the separator, for every caller: ``separate_cmir`` rounds integer
columns' bounds inward and complements each column against a finite bound it really
has, declining a row that needs a missing one.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from discopt import Model
from discopt._relax.cmir_cuts import separate_cmir

# The issue's instance, verbatim.
_Q = np.array([[34.51854932465685, 33.56202029633967], [33.56202029633967, 52.85501810747605]])
_C = np.array([-5.804167155655379, 4.502570137120227])
_A = np.array([[0.09128248617581755, 1.3019471481756761], [1.2688434632505559, 0.9014167063512275]])
_B = np.array([0.3967991619490152, 0.4662377078789963])


def _issue_model() -> Model:
    m = Model("convex_iqp_2262")
    x = m.integer("x", shape=(2,), lb=-2, ub=2)
    quad = sum(_Q[i, j] * x[i] * x[j] for i in range(2) for j in range(2))
    m.minimize(quad + _C[0] * x[0] + _C[1] * x[1])
    for k in range(2):
        m.subject_to(_A[k, 0] * x[0] + _A[k, 1] * x[1] <= _B[k])
    return m


def _oracle() -> float:
    best = np.inf
    for p in itertools.product(range(-2, 3), repeat=2):
        p = np.array(p, float)
        if np.all(_A @ p <= _B):
            best = min(best, float(p @ _Q @ p + _C @ p))
    return best


def test_issue_2262_nlp_bb_certifies_the_true_optimum():
    opt = _oracle()
    assert opt == 0.0
    r = _issue_model().solve(time_limit=20, nlp_bb=True)
    # The certificate here sits near the absolute gap tolerance (1e-6; the optimum
    # is 0, so the relative one adds nothing): locally bound = -6.72e-7. Report the
    # numbers, not just the status, so a withheld certificate can be told apart
    # from a wrong answer.
    assert r.status == "optimal", (
        r.status,
        r.objective,
        r.bound,
        r.gap,
        r.node_count,
        r.algorithm_route,
    )
    tol = 1e-6
    # The certificate: the bound may not exceed a feasible objective (min sense).
    assert r.bound is not None and r.bound <= opt + tol, (r.bound, opt)
    assert abs(r.objective - opt) <= tol, (r.objective, opt)
    np.testing.assert_allclose(r.x["x"], [0.0, 0.0], atol=1e-6)


def _int_points(lb, ub):
    """Every integer point of the box (bounds rounded inward)."""
    rngs = [range(int(np.ceil(lo - 1e-9)), int(np.floor(hi + 1e-9)) + 1) for lo, hi in zip(lb, ub)]
    return [np.array(p, float) for p in itertools.product(*rngs)]


def test_cmir_single_row_fractional_integer_upper_bound():
    """One integer column on ``[-2, 0.445]`` and the row ``x <= 0.3``.

    Complementing against ``u = 0.445`` gives ``-y <= -0.145``, which MIR rounds to
    ``y >= 1``, i.e. ``x <= -0.555``: that removes the feasible ``x = 0``. Measured
    on the pre-fix separator.
    """
    cuts = separate_cmir(
        np.array([[1.0]]),
        np.array([0.3]),
        np.array([0.3]),
        np.array([-2.0]),
        np.array([0.445]),
        np.array([True]),
    )
    for a, r in cuts:
        for x in (0.0, -1.0, -2.0):
            assert a[0] * x <= r + 1e-9, (a, r, x)


def test_cmir_valid_by_enumeration_on_fractional_integer_boxes():
    """Pure-integer rows on boxes with fractional (FBBT-like) bounds: every cut keeps
    every integer point that satisfies the row. Enumeration is exact, no sampling."""
    rng = np.random.default_rng(2262)
    n_cuts = 0
    n_checks = 0
    for _trial in range(300):
        n = int(rng.integers(1, 4))
        lb = -rng.integers(0, 4, n) + rng.uniform(0.0, 0.95, n) * rng.integers(0, 2, n)
        ub = rng.integers(0, 4, n) - rng.uniform(0.0, 0.95, n) * rng.integers(0, 2, n)
        ub = np.maximum(ub, lb)
        m = int(rng.integers(1, 3))
        a_mat = rng.normal(0, 2, size=(m, n))
        b = rng.normal(0, 1, size=m)
        xstar = lb + rng.uniform(0, 1, n) * (ub - lb)
        feas = [p for p in _int_points(lb, ub) if np.all(a_mat @ p <= b + 1e-12)]
        cuts = separate_cmir(a_mat, b, xstar, lb, ub, np.ones(n, bool), duals=np.ones(m))
        n_cuts += len(cuts)
        for a, r in cuts:
            for p in feas:
                n_checks += 1
                assert float(a @ p) <= r + 1e-9, (a, r, p, lb, ub)
    # Prove the probe fired: cuts were produced and checked against feasible points.
    assert n_cuts > 20 and n_checks > 100, (n_cuts, n_checks)


def test_cmir_infinite_bounds_complement_against_the_finite_one():
    """An integer column with only ``ub = 5`` and the row ``x <= 2.5``: the cut must
    hold for every integer ``x <= 2``, far below any placeholder lower bound."""
    cuts = separate_cmir(
        np.array([[1.0]]),
        np.array([2.5]),
        np.array([2.5]),
        np.array([-np.inf]),
        np.array([5.0]),
        np.array([True]),
    )
    assert cuts, "the row x <= 2.5 has the MIR cut x <= 2"
    for a, r in cuts:
        for x in (2.0, 0.0, -1.0, -1e6):
            assert a[0] * x <= r + 1e-9 * (1 + abs(x)), (a, r, x)
        assert a[0] * 2.5 > r  # it separates xstar


def test_cmir_row_on_a_free_column_is_declined():
    """A row with weight on a column that has no finite bound has no valid
    substitution, so it yields no cut (it used to need a placeholder box)."""
    cuts = separate_cmir(
        np.array([[1.0, 1.0]]),
        np.array([2.5]),
        np.array([2.5, 0.0]),
        np.array([0.0, -np.inf]),
        np.array([5.0, np.inf]),
        np.array([True, True]),
    )
    assert cuts == []


@pytest.mark.parametrize("seed", [3, 11, 17])
def test_random_convex_integer_qp_nlp_bb_bound_is_sound(seed):
    """The issue's generator class (random convex integer QP, two ``<=`` rows,
    brute-force oracle): the NLP-BB bound never crosses the oracle."""
    rng = np.random.default_rng(seed)
    n = 2
    L = 2
    M = rng.normal(size=(n, n)) * 4.0
    Q = M @ M.T
    c = rng.normal(size=n) * 5
    A = rng.normal(size=(2, n))
    b = rng.uniform(0.1, 1.0, size=2)
    best = min(
        float(p @ Q @ p + c @ p)
        for p in (np.array(t, float) for t in itertools.product(range(-L, L + 1), repeat=n))
        if np.all(A @ p <= b)
    )
    m = Model("rq")
    x = m.integer("x", shape=(n,), lb=-L, ub=L)
    quad = sum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n))
    m.minimize(quad + c[0] * x[0] + c[1] * x[1])
    for k in range(2):
        m.subject_to(A[k, 0] * x[0] + A[k, 1] * x[1] <= b[k])
    r = m.solve(time_limit=20, nlp_bb=True)
    tol = 1e-4 * max(1.0, abs(best))
    assert r.bound is None or r.bound <= best + tol, (r.bound, best)
    if r.gap_certified and r.status == "optimal":
        assert abs(r.objective - best) <= tol, (r.objective, best)
