"""Tests for the Lagrangian B&B node-bound hook (`lagrangian_bound=True`).

The hook combines a per-node Lagrangian dual bound with the LP relaxation bound
inside the MILP branch-and-bound (`_solve_milp_bb`). Soundness is verified
**directly against brute-force enumeration** rather than `model.solve()`, because
the monolithic MILP solver is itself the thing the bound feeds into.

Headline invariant: the Lagrangian bound (root and per-node) never exceeds the
true optimum — feeding it via `max()` therefore can never prune the optimal node.
"""

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition.lagrangian.node_bounder import LagrangianNodeBounder
from discopt.decomposition.structure import flat_bounds

try:
    from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
except ImportError:
    POUNCE_AVAILABLE = False
try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = pytest.mark.skipif(
    not (POUNCE_AVAILABLE or HAS_HIGHS), reason="no LP/MILP backend available"
)


# ── GAP instance builder (1D-flattened; blocks = knapsack capacity, ──
# ── coupling = each task assigned once; subproblems lack integrality) ──


def _build_gap(seed, K=2, N=6):
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 10, (K, N))
    w = rng.integers(3, 9, (K, N))
    cap = [int(w[k].sum() * 0.6) for k in range(K)]
    m = dm.Model(f"gap{seed}")
    x = m.binary("x", shape=(K * N,))

    def xi(k, i):
        return x[k * N + i]

    m.minimize(sum(int(cost[k, i]) * xi(k, i) for k in range(K) for i in range(N)))
    for k in range(K):
        m.subject_to(sum(int(w[k, i]) * xi(k, i) for i in range(N)) <= cap[k])
    for i in range(N):
        c = sum(xi(k, i) for k in range(K)) == 1
        m.subject_to(c)
        m.mark_coupling(c)
    return m, cost, w, cap, K, N


def _brute_force(cost, w, cap, K, N):
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


# ── Soundness: bound never exceeds the true optimum ──


@pytest.mark.correctness
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 7, 11, 14])
def test_node_bound_never_exceeds_true_optimum(seed):
    m, cost, w, cap, K, N = _build_gap(seed)
    opt = _brute_force(cost, w, cap, K, N)
    if not np.isfinite(opt):
        pytest.skip("infeasible instance")
    b = LagrangianNodeBounder.try_build(m, prefer_pounce=True)
    assert b is not None
    lb, ub = flat_bounds(m)
    root = b.solve_root_dual(lb, ub)
    node = b.node_bound(lb, ub)
    assert root is not None and root <= opt + 1e-3, f"root bound {root} > opt {opt}"
    assert node is not None and node <= opt + 1e-3, f"node bound {node} > opt {opt}"


# ── Effectiveness: the bound can strictly beat the LP relaxation ──


@pytest.mark.slow
def test_bound_can_beat_lp_relaxation():
    """On at least one gap instance the Lagrangian root bound > LP root bound.

    Confirms the hook is not a silent no-op (Lagrangian dominates LP exactly when
    the block subproblems lack the integrality property — here, knapsacks).
    """
    from discopt.decomposition._linear import extract_linear
    from discopt.solvers.lp_backend import get_lp_solver

    solve_lp = get_lp_solver(prefer_pounce=True)
    beat = False
    for seed in range(40):
        m, cost, w, cap, K, N = _build_gap(seed, K=3, N=6)
        b = LagrangianNodeBounder.try_build(m, prefer_pounce=True)
        if b is None:
            continue
        lb, ub = flat_bounds(m)
        lin = extract_linear(m)
        n = lin.n
        A = np.array(lin.rows_coeff)
        rhs = np.array(lin.rows_rhs)
        lp = solve_lp(
            lin.c, A_ub=A, b_ub=rhs, bounds=[(float(lb[i]), float(ub[i])) for i in range(n)]
        )
        if lp.objective is None:
            continue
        lp_bound = lp.objective + lin.c_offset
        root = b.solve_root_dual(lb, ub)
        opt = _brute_force(cost, w, cap, K, N)
        # Validity must always hold; tightening sometimes.
        assert root <= opt + 1e-3
        if root > lp_bound + 1e-3:
            beat = True
            break
    assert beat, "Lagrangian bound never beat LP across the search — expected on knapsack blocks"


# ── No-op safety: the hook cleanly disables where it doesn't apply ──


def test_try_build_disabled_without_coupling():
    m = dm.Model("nocouple")
    x = m.binary("x", shape=(3,))
    m.minimize(x[0] + x[1] + x[2])
    m.subject_to(x[0] + x[1] + x[2] >= 1)
    assert LagrangianNodeBounder.try_build(m) is None


def test_try_build_disabled_for_nonlinear():
    m = dm.Model("nl")
    x = m.continuous("x", lb=0, ub=5)
    y = m.binary("y")
    m.minimize(x**2 + y)
    c = x + y <= 3
    m.subject_to(c)
    m.mark_coupling(c)
    assert LagrangianNodeBounder.try_build(m) is None


def test_try_build_disabled_for_maximize():
    m = dm.Model("max")
    x = m.binary("x", shape=(3,))
    m.maximize(x[0] + x[1] + x[2])
    c = x[0] + x[1] + x[2] <= 2
    m.subject_to(c)
    m.mark_coupling(c)
    assert LagrangianNodeBounder.try_build(m) is None


@pytest.mark.correctness
def test_hook_flag_noops_on_maximize_end_to_end():
    """End-to-end: ``lagrangian_bound=True`` on a *maximize* model is a harmless
    no-op (the bounder is minimization-only) and the optimum is still correct."""
    val = [3, 5, 4, 2]
    w = [3, 4, 2, 3]
    cap = 6

    def build():
        m = dm.Model("maxhook")
        x = m.binary("x", shape=(4,))
        m.maximize(sum(val[i] * x[i] for i in range(4)))
        c = sum(w[i] * x[i] for i in range(4)) <= cap
        m.subject_to(c)
        m.mark_coupling(c)
        return m

    best = max(
        sum(val[i] * b[i] for i in range(4))
        for b in itertools.product([0, 1], repeat=4)
        if sum(w[i] * b[i] for i in range(4)) <= cap
    )
    off = build().solve(time_limit=30)
    on = build().solve(time_limit=30, lagrangian_bound=True)
    assert off.objective == pytest.approx(best, abs=1e-3)
    assert on.objective == pytest.approx(best, abs=1e-3)


def test_try_build_disabled_for_multidim_index():
    """2D-indexed models aren't supported by the linear extractor -> clean no-op."""
    m = dm.Model("twod")
    x = m.binary("x", shape=(2, 3))
    m.minimize(sum(x[k, i] for k in range(2) for i in range(3)))
    c = sum(x[0, i] for i in range(3)) <= 1
    m.subject_to(c)
    m.mark_coupling(c)
    assert LagrangianNodeBounder.try_build(m) is None


# ── End-to-end on a clean instance (base solver correct) ──


def _block_conflict():
    m = dm.Model("conflict")
    x = m.binary("x", shape=(4,))
    m.minimize(2 * x[0] + 3 * x[1] + 2 * x[2] + 4 * x[3])
    m.subject_to(x[0] + x[1] >= 1)
    m.subject_to(x[2] + x[3] >= 1)
    conf = x[0] + x[2] <= 1
    m.subject_to(conf)
    m.mark_coupling(conf)
    return m


def test_end_to_end_hook_matches_hook_off():
    off = _block_conflict().solve(time_limit=60)
    on = _block_conflict().solve(time_limit=60, lagrangian_bound=True)
    assert on.status == "optimal"
    assert on.objective == pytest.approx(off.objective, abs=1e-6)
    assert on.objective == pytest.approx(5.0, abs=1e-3)


def test_hook_flag_noop_on_coupling_free_model():
    """Enabling the flag on a model with no coupling structure is harmless."""
    m = dm.Model("plain")
    x = m.binary("x", shape=(3,))
    m.minimize(x[0] + 2 * x[1] + 3 * x[2])
    m.subject_to(x[0] + x[1] + x[2] >= 1)
    r = m.solve(time_limit=30, lagrangian_bound=True)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-3)


def _branching_instance():
    """A covering instance that makes the MILP B&B actually branch (node_count
    > 1), so the hook's *in-tree* per-node bound path is exercised, not just the
    root. min c.x s.t. sum (i+1) x_i >= 35, coupling sum x_i <= 5, x binary."""
    n = 11
    rng = np.random.default_rng(2)
    c = rng.integers(1, 7, n)
    m = dm.Model("branch")
    x = m.binary("x", shape=(n,))
    m.minimize(sum(int(c[i]) * x[i] for i in range(n)))
    m.subject_to(sum((i + 1) * x[i] for i in range(n)) >= 35)
    cpl = sum(x[i] for i in range(n)) <= 5
    m.subject_to(cpl)
    m.mark_coupling(cpl)
    return m, c, n


def _branching_brute(c, n, cap=5, target=35):
    best = np.inf
    for bits in itertools.product([0, 1], repeat=n):
        z = np.array(bits)
        if sum((i + 1) * z[i] for i in range(n)) >= target and z.sum() <= cap:
            best = min(best, float(sum(int(c[i]) * z[i] for i in range(n))))
    return best


@pytest.mark.correctness
def test_hook_sound_and_helpful_on_a_branching_tree():
    """In-tree coverage: the instance branches deeply; the hook must stay sound
    (same optimum as hook-off and brute force) and must not *increase* nodes."""
    m_off, c, n = _branching_instance()
    opt = _branching_brute(c, n)
    off = m_off.solve(time_limit=60)
    on = _branching_instance()[0].solve(time_limit=60, lagrangian_bound=True)
    # The hook path is genuinely exercised in the tree, not just at the root.
    assert off.node_count > 1, "instance no longer branches; pick a harder one"
    assert off.objective == pytest.approx(opt, abs=1e-3)
    assert on.objective == pytest.approx(opt, abs=1e-3)
    # A valid extra lower bound can only prune more (or tie), never explore more.
    assert on.node_count <= off.node_count
