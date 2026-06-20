"""Regression: MILP B&B node solutions must respect their variable bounds.

The pure-Rust simplex adapter (and the POUNCE IPM) can occasionally return a
basic point that violates the variable box it was given on mixed
equality/inequality nodes. Such a point can be *integral but off-bound* (e.g. a
binary at -1) and pass the per-node row-feasibility check, so the B&B tree would
accept it as a spurious integer incumbent with a too-low objective.

The node-solve soundness gate now also checks the variable bounds. These tests
pin the regression on a generalized-assignment instance (knapsack `<=` blocks +
`==` coupling) that previously returned 16 with a binary at -1 instead of the
true optimum 17.
"""

import itertools

import discopt.modeling as dm
import numpy as np
import pytest


def _gap(seed, K=3, N=6):
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 10, (K, N))
    w = rng.integers(3, 9, (K, N))
    cap = [int(w[k].sum() * 0.45) for k in range(K)]
    m = dm.Model(f"gap{seed}")
    x = m.binary("x", shape=(K * N,))

    def xi(k, i):
        return x[k * N + i]

    m.minimize(sum(int(cost[k, i]) * xi(k, i) for k in range(K) for i in range(N)))
    for k in range(K):
        m.subject_to(sum(int(w[k, i]) * xi(k, i) for i in range(N)) <= cap[k])
    for i in range(N):
        m.subject_to(sum(xi(k, i) for k in range(K)) == 1)
    return m, cost, w, cap, K, N


def _brute(cost, w, cap, K, N):
    best = np.inf
    for bits in itertools.product([0, 1], repeat=K * N):
        z = np.array(bits)
        if all(
            sum(int(w[k, i]) * z[k * N + i] for i in range(N)) <= cap[k] for k in range(K)
        ) and all(sum(z[k * N + i] for k in range(K)) == 1 for i in range(N)):
            best = min(
                best, float(sum(int(cost[k, i]) * z[k * N + i] for k in range(K) for i in range(N)))
            )
    return best


def test_no_off_bound_binary_in_incumbent():
    """The exact instance that exposed the bug: optimum 17, no binary at -1."""
    m, cost, w, cap, K, N = _gap(0)
    r = m.solve(time_limit=60)
    assert r.status in ("optimal", "feasible")
    z = r.x["x"]
    assert np.all(z >= -1e-6), f"binary below 0: {z[z < -1e-6]}"
    assert np.all(z <= 1 + 1e-6)
    assert r.objective == pytest.approx(17.0, abs=1e-3)


@pytest.mark.correctness
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5])
def test_milp_matches_bruteforce_with_mixed_eq_ineq(seed):
    m, cost, w, cap, K, N = _gap(seed)
    opt = _brute(cost, w, cap, K, N)
    if not np.isfinite(opt):
        pytest.skip("infeasible instance")
    r = m.solve(time_limit=60)
    assert r.status in ("optimal", "feasible")
    z = r.x["x"]
    assert np.all((z >= -1e-6) & (z <= 1 + 1e-6)), "binary variable outside [0, 1]"
    assert r.objective == pytest.approx(opt, abs=1e-3)
