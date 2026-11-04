"""Partition-based (Tsay et al. 2021) and ideal (Anderson et al. 2020) ReLU MILPs (#2134).

What is checked, on small random networks:

* **Exactness.** ``relu_bigm``, ``relu_partition`` (P = 1, 2, 3, both split
  strategies), ``relu_extended`` and big-M + Anderson cuts certify the same MILP
  optimum, and that optimum matches dense enumeration of ``net.forward`` (an
  independent oracle, not another formulation).
* **Tightness order.** The LP relaxation (integrality dropped) bound obeys
  big-M == partition(P=1) <= partition(P=2) <= partition(P=3) <= extended, and
  the big-M LP closed under exact Anderson separation reaches the extended LP
  bound (the projection of the per-neuron hull, Anderson et al. Thm 1). On at
  least one instance the order is *strict*, so the test cannot pass on three
  copies of big-M.
* **Cut validity.** No separated cut is violated by an exactly-evaluated network
  point sampled from the input box (feasible-point sampling, CLAUDE.md §5).

Each test counts the comparisons it made and fails if that count is zero (§6).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from discopt._relax.problem_classifier import dense_A, extract_lp_data
from discopt.callbacks import cut_result_to_dense
from discopt.ml import DenseLayer, NetworkDefinition, NNFormulation, OffsetScaling, add_predictor
from discopt.ml.formulations.relu_partition import partition_indices
from discopt.modeling import Model
from scipy.optimize import linprog

SIZES = [[2, 4, 1], [3, 5, 3, 1], [2, 6, 1]]
SEEDS = [0, 3]

STRATEGIES = [
    ("relu_bigm", {}),
    ("relu_partition", {"n_partitions": 1}),
    ("relu_partition", {"n_partitions": 2}),
    ("relu_partition", {"n_partitions": 3}),
    ("relu_partition", {"n_partitions": 2, "partition_strategy": "contiguous"}),
    ("relu_extended", {}),
]


def _net(sizes, seed):
    rng = np.random.default_rng(seed)
    layers = [
        DenseLayer(
            weights=rng.normal(0, 1.0, (sizes[i], sizes[i + 1])),
            biases=rng.normal(0, 0.3, sizes[i + 1]),
            activation="relu" if i < len(sizes) - 2 else "linear",
        )
        for i in range(len(sizes) - 1)
    ]
    n = sizes[0]
    return NetworkDefinition(layers, input_bounds=(np.full(n, -1.0), np.full(n, 1.0)))


def _build(net, strategy, opts, sense="min", scaling=None):
    m = Model("nn")
    nn = NNFormulation(m, net, strategy=strategy, prefix="p", scaling=scaling, **opts)
    nn.formulate()
    if sense == "min":
        m.minimize(nn.outputs[0])
    else:
        m.maximize(nn.outputs[0])
    return m, nn


def _lp_bound(m, cuts=()):
    """LP relaxation value (integrality dropped) of a min-sense linear model."""
    d = extract_lp_data(m)
    A = dense_A(d.A_eq)
    n = A.shape[1]
    n_orig = sum(v.size for v in m._variables)
    A_ub, b_ub = [], []
    for coeffs, rhs, sense in cuts:
        row = np.zeros(n)
        row[:n_orig] = coeffs
        assert sense == "<="
        A_ub.append(row)
        b_ub.append(rhs)
    res = linprog(
        np.asarray(d.c),
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=A,
        b_eq=np.asarray(d.b_eq),
        bounds=list(zip(np.asarray(d.x_l), np.asarray(d.x_u))),
        method="highs",
    )
    assert res.status == 0, res.message
    return float(res.fun) + float(d.obj_const), res.x[:n_orig]


def _grid_opt(net, sense, n=81):
    lb, ub = net.input_bounds
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(lb, ub)]
    pts = np.array(list(itertools.product(*axes)))
    vals = net.forward(pts)[:, 0]
    return float(vals.min() if sense == "min" else vals.max())


# --------------------------------------------------------------------------- #
# partition_indices
# --------------------------------------------------------------------------- #


def test_partition_indices_cover_nonzero_weights_exactly_once():
    w = np.array([0.5, 0.0, -2.0, 1.5, 0.0, -0.1, 3.0])
    nz = {0, 2, 3, 5, 6}
    n_checked = 0
    for P in (1, 2, 3, 5, 9, None):
        for strat in ("equal_size", "contiguous"):
            parts = partition_indices(w, P, strat)
            flat = np.concatenate(parts).tolist()
            assert sorted(flat) == sorted(nz)
            assert len(parts) == (len(nz) if P is None else min(P, len(nz)))
            n_checked += 1
    # equal_size groups by sorted weight value
    assert [p.tolist() for p in partition_indices(w, 2, "equal_size")] == [[2, 5, 0], [3, 6]]
    assert [p.tolist() for p in partition_indices(w, 2, "contiguous")] == [[0, 2, 3], [5, 6]]
    assert n_checked == 12


def test_partition_indices_validates_callable_and_args():
    w = np.array([1.0, 2.0, 3.0])
    assert [p.tolist() for p in partition_indices(w, 2, lambda w, P: [[0, 2], [1]])] == [
        [0, 2],
        [1],
    ]
    with pytest.raises(ValueError, match="exactly once"):
        partition_indices(w, 2, lambda w, P: [[0], [1]])
    with pytest.raises(ValueError, match="exactly once"):
        partition_indices(w, 2, lambda w, P: [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match=">= 1"):
        partition_indices(w, 0)
    with pytest.raises(ValueError, match="Unknown partition_strategy"):
        partition_indices(w, 2, "random")


def test_option_validation():
    net = _net([2, 3, 1], 0)
    with pytest.raises(ValueError, match="relu_partition"):
        NNFormulation(Model("a"), net, strategy="relu_bigm", n_partitions=2)
    with pytest.raises(ValueError, match="ideal_cuts"):
        NNFormulation(Model("a"), net, strategy="full_space", ideal_cuts=True)
    with pytest.raises(ValueError, match="n_partitions"):
        NNFormulation(Model("a"), net, strategy="relu_partition", n_partitions=0)
    nn = NNFormulation(Model("a"), net, strategy="relu_bigm")
    nn.formulate()
    with pytest.raises(RuntimeError, match="ideal_cuts=True"):
        _ = nn.cut_callback


def test_p1_is_exactly_bigm_structure():
    """P = 1 emits big-M's rows and no auxiliaries (Tsay et al.: P=1 is big-M)."""
    net = _net([3, 5, 3, 1], 0)
    mb, _ = _build(net, "relu_bigm", {})
    mp, _ = _build(net, "relu_partition", {"n_partitions": 1})
    assert [v.name for v in mb._variables] == [v.name for v in mp._variables]
    assert [str(c.name) for c in mb._constraints] == [str(c.name) for c in mp._constraints]


def test_stable_neurons_get_no_binary():
    rng = np.random.default_rng(1)
    W = rng.normal(0, 1, (2, 4))
    b = np.array([10.0, -10.0, 0.1, -0.1])  # neurons 0/1 stable, 2/3 mixed
    net = NetworkDefinition(
        [DenseLayer(W, b, "relu"), DenseLayer(np.ones((4, 1)), np.zeros(1), "linear")],
        input_bounds=(np.full(2, -1.0), np.full(2, 1.0)),
    )
    for strategy in ("relu_partition", "relu_extended"):
        m, _ = _build(net, strategy, {})
        names = {v.name for v in m._variables}
        assert "p_q_0_0" not in names and "p_q_0_1" not in names
        assert "p_q_0_2" in names and "p_q_0_3" in names
        assert "p_zb_0_0" not in names and "p_zb_0_2" in names


# --------------------------------------------------------------------------- #
# Exactness: same certified MILP optimum, matching enumeration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sizes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("sense", ["min", "max"])
def test_milp_optimum_identical_across_strategies(sizes, seed, sense):
    net = _net(sizes, seed)
    grid = _grid_opt(net, sense, n=81 if sizes[0] == 2 else 25)
    objs = []
    for strategy, opts in STRATEGIES:
        m, _ = _build(net, strategy, opts, sense)
        r = m.solve(time_limit=60)
        assert r.status == "optimal", (strategy, opts, r.status)
        objs.append(float(r.objective))
    np.testing.assert_allclose(objs, objs[0], atol=1e-5, rtol=1e-6)
    # The grid only samples the box, so it is an inner bound of the true optimum.
    if sense == "min":
        assert objs[0] <= grid + 1e-6
    else:
        assert objs[0] >= grid - 1e-6
    assert len(objs) == len(STRATEGIES)


@pytest.mark.parametrize("strategy,opts", STRATEGIES[1:])
def test_fixed_inputs_reproduce_forward_with_scaling(strategy, opts):
    net = _net([3, 5, 3, 1], 3)
    sc = OffsetScaling(
        x_offset=np.full(3, 0.25),
        x_factor=np.full(3, 2.0),
        y_offset=np.full(1, -0.5),
        y_factor=np.full(1, 3.0),
    )
    rng = np.random.default_rng(11)
    n_checked = 0
    for x in rng.uniform(-1, 1, (4, 3)):
        m, nn = _build(net, strategy, opts, "min", scaling=sc)
        m.subject_to(nn.inputs == x, name="pin")
        r = m.solve(time_limit=60)
        assert r.status == "optimal"
        ref = 3.0 * net.forward((x - 0.25) / 2.0)[0] - 0.5
        assert abs(float(r.objective) - ref) < 1e-5
        n_checked += 1
    assert n_checked == 4


def test_add_predictor_forwards_options():
    net = _net([2, 4, 1], 0)
    m = Model("ap")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    out, form = add_predictor(m, x, net, method="relu_partition", n_partitions=2, ideal_cuts=True)
    assert any(v.name.startswith("pred_zb_") for v in m._variables)
    assert form.cut_callback is not None


# --------------------------------------------------------------------------- #
# Tightness: LP relaxation ordering, and the cut closure reaching the hull
# --------------------------------------------------------------------------- #


def _cut_closure_bound(net, rounds=50):
    m, nn = _build(net, "relu_bigm", {"ideal_cuts": True})
    sep = nn.cut_callback
    cuts = []
    bound, x = _lp_bound(m)
    for _ in range(rounds):
        new = sep.separate(x, m)
        if not new:
            return bound, len(cuts)
        cuts.extend(cut_result_to_dense(c, m) for c in new)
        bound, x = _lp_bound(m, cuts)
    raise AssertionError("cut loop did not converge")


@pytest.mark.parametrize("sizes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_root_lp_bound_ordering(sizes, seed):
    net = _net(sizes, seed)
    b = {}
    for key, strategy, opts in [
        ("bigm", "relu_bigm", {}),
        ("P1", "relu_partition", {"n_partitions": 1}),
        ("P2", "relu_partition", {"n_partitions": 2}),
        ("P3", "relu_partition", {"n_partitions": 3}),
        ("ext", "relu_extended", {}),
    ]:
        m, _ = _build(net, strategy, opts)
        b[key] = _lp_bound(m)[0]
    tol = 1e-7
    assert abs(b["P1"] - b["bigm"]) <= tol
    # min sense: a tighter relaxation has a HIGHER bound. P2 and P3 are not
    # nested partitions in general, so each is compared to the two ends only.
    assert b["bigm"] <= b["P2"] + tol <= b["ext"] + 2 * tol
    assert b["bigm"] <= b["P3"] + tol <= b["ext"] + 2 * tol
    closure, _ = _cut_closure_bound(net)
    assert abs(closure - b["ext"]) <= 1e-6 * max(1.0, abs(b["ext"]))


def test_ordering_is_strict_somewhere():
    """The ordering test must not pass on formulations that are all big-M."""
    gains = []
    for sizes in SIZES:
        for seed in SEEDS:
            net = _net(sizes, seed)
            lo = _lp_bound(_build(net, "relu_bigm", {})[0])[0]
            p2 = _lp_bound(_build(net, "relu_partition", {"n_partitions": 2})[0])[0]
            hi = _lp_bound(_build(net, "relu_extended", {})[0])[0]
            gains.append((p2 - lo, hi - lo))
    assert any(g[0] > 1e-4 for g in gains), gains
    assert any(g[1] > 1e-4 for g in gains), gains


# --------------------------------------------------------------------------- #
# Ideal cuts: never cut a feasible network point; usable through solve()
# --------------------------------------------------------------------------- #


def _feasible_flat(m, net, x):
    """The exact (x, zhat, z, q) point of the big-M embedding at input x."""
    vals = {}
    vals["p_input"] = np.asarray(x, dtype=np.float64)
    h = vals["p_input"]
    for k, layer in enumerate(net.layers):
        pre = h @ layer.weights + layer.biases
        vals[f"p_zhat_{k}"] = pre
        if layer.activation.value == "relu":
            vals[f"p_z_{k}"] = np.maximum(pre, 0.0)
            for j in range(pre.size):
                vals[f"p_q_{k}_{j}"] = np.array([1.0 if pre[j] > 0 else 0.0])
            h = vals[f"p_z_{k}"]
        else:
            h = pre
    flat = []
    for v in m._variables:
        flat.append(np.broadcast_to(vals[v.name], (v.size,)) if v.name in vals else None)
        assert flat[-1] is not None, v.name
    return np.concatenate([np.ravel(f) for f in flat])


@pytest.mark.parametrize("sizes", SIZES)
def test_separated_cuts_never_cut_a_feasible_point(sizes):
    net = _net(sizes, 0)
    m, nn = _build(net, "relu_bigm", {"ideal_cuts": True})
    sep = nn.cut_callback
    rng = np.random.default_rng(5)
    n = sum(v.size for v in m._variables)
    # Cuts from fractional points: random points of the relaxed box + the LP optimum.
    cuts = []
    _, x_lp = _lp_bound(m)
    cuts += sep.separate(x_lp, m)
    box_lo = np.concatenate([np.ravel(np.broadcast_to(v.lb, (v.size,))) for v in m._variables])
    box_hi = np.concatenate([np.ravel(np.broadcast_to(v.ub, (v.size,))) for v in m._variables])
    for _ in range(200):
        cuts += sep.separate(rng.uniform(box_lo, box_hi, n), m)
    assert cuts, "separator produced no cuts to test"
    dense = [cut_result_to_dense(c, m) for c in cuts]
    n_checks = 0
    for x in rng.uniform(-1, 1, (300, sizes[0])):
        pt = _feasible_flat(m, net, x)
        for coeffs, rhs, sense in dense:
            assert coeffs @ pt <= rhs + 1e-9 * max(1.0, abs(rhs)), (coeffs, rhs)
            n_checks += 1
        # an exact point is in the hull: nothing to separate
        assert sep.separate(pt, m) == []
    assert n_checks > 0


def test_ideal_cuts_through_solve():
    # big-M LP -0.300 vs MILP -0.083 on this instance: the cuts have work to do.
    net = _net([2, 6, 1], 0)
    ref, _ = _build(net, "relu_bigm", {})
    r0 = ref.solve(time_limit=60)
    m, nn = _build(net, "relu_bigm", {"ideal_cuts": True})
    r = m.solve(time_limit=120, cut_callback=nn.cut_callback)
    assert r.status == "optimal"
    assert abs(float(r.objective) - float(r0.objective)) <= 1e-5
    assert nn.cut_callback.n_calls > 0
    assert nn.cut_callback.n_cuts > 0
    assert r.solver_stats["cut_validation/cuts"] == nn.cut_callback.n_cuts
