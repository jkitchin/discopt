"""Soundness guards for Lagrangian relaxation.

Headline invariant: the Lagrangian dual ``bound`` is a *valid lower bound* on
the optimum (for minimization) — it never exceeds the true optimum, and never
exceeds the incumbent objective on an ``optimal`` exit. L(λ) is a lower bound
for every λ >= 0, so the best L(λ) reported must respect these.
"""

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition.lagrangian import solve_lagrangian

try:
    from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
except ImportError:
    POUNCE_AVAILABLE = False
try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = [
    pytest.mark.skipif(not (POUNCE_AVAILABLE or HAS_HIGHS), reason="no LP/MILP backend available"),
    pytest.mark.correctness,
]


def _brute_force_min(cost, demand_rows, coupling_cap, n):
    """Enumerate binary assignments; return the feasible minimum."""
    best = np.inf
    for bits in itertools.product([0, 1], repeat=n):
        b = np.array(bits)
        if any(b @ row < d - 1e-9 for row, d in demand_rows):
            continue
        if b.sum() > coupling_cap + 1e-9:
            continue
        best = min(best, float(cost @ b))
    return best


@pytest.mark.parametrize("method", ["subgradient", "bundle"])
def test_bound_never_exceeds_true_optimum(method):
    rng = np.random.default_rng(2024)
    n = 5
    cap = 3
    for _ in range(15):
        cost = rng.uniform(1, 6, n)
        m = dm.Model("s")
        x = m.binary("x", shape=(n,))
        m.minimize(sum(cost[i] * x[i] for i in range(n)))
        # two "demand" block constraints + a coupling cardinality cap.
        row0 = np.zeros(n)
        row0[: n // 2] = 1
        row1 = np.zeros(n)
        row1[n // 2 :] = 1
        m.subject_to(sum(row0[i] * x[i] for i in range(n)) >= 1)
        m.subject_to(sum(row1[i] * x[i] for i in range(n)) >= 1)
        cpl = sum(x[i] for i in range(n)) <= cap
        m.subject_to(cpl)
        m.mark_coupling(cpl)

        r = solve_lagrangian(m, method=method, time_limit=30)
        opt = _brute_force_min(cost, [(row0, 1), (row1, 1)], cap, n)
        if r.bound is not None:
            assert r.bound <= opt + 1e-3, f"unsound bound {r.bound} > optimum {opt}"
        if r.status == "optimal":
            assert r.objective == pytest.approx(opt, abs=1e-3)


def test_no_false_optimal_certification():
    """On 'optimal', the lower bound must not exceed the incumbent objective."""
    m = dm.Model("c")
    x = m.binary("x", shape=(4,))
    m.minimize(2 * x[0] + 3 * x[1] + 2 * x[2] + 4 * x[3])
    m.subject_to(x[0] + x[1] >= 1)
    m.subject_to(x[2] + x[3] >= 1)
    conf = x[0] + x[2] <= 1
    m.subject_to(conf)
    m.mark_coupling(conf)
    r = solve_lagrangian(m, time_limit=30)
    if r.status == "optimal":
        assert r.bound is not None
        assert r.bound <= r.objective + 1e-4
        assert r.gap_certified
