"""Native (Rust) separation of Anderson et al.'s ideal ReLU inequalities (#2157).

Before #2157, ``model.solve(cut_callback=nn.cut_callback)`` on a ReLU big-M MILP
moved the solve off HiGHS onto the Python spatial tree: ~200x slower than plain
big-M on a trained 2-32-32-1 network, the root gap unchanged, and one network
stopping at ``feasible`` with a 0.70 gap. Now the solver recognises the
separator and separates the same inequalities with the Rust
``ReluIdealSeparator``: over repeated root LP solves on the HiGHS route, at
every fractional node on the Rust MILP engine.

What is checked:

* **Parity.** The Rust separator returns exactly the Python
  ``AndersonCutSeparator``'s cut (same neurons, same coefficients) at random
  fractional points -- the two are independent implementations of one formula.
* **Validity.** No Rust cut is violated by an exactly evaluated network point
  (feasible-point sampling, CLAUDE.md §5), and none is returned at such a point.
* **Routing and certification.** The solve takes the native route, certifies
  the same optimum as plain big-M (and as dense enumeration of the network),
  executes the separator (counts > 0, §6), and its root bound is strictly
  tighter than plain big-M's on an instance where the cuts have work to do.
* **The Rust-engine route** certifies the same optimum with node separation.
* **The Python-callback route stays available**: the separator wrapped in a
  function is an ordinary callback and still reaches the spatial tree.

Each test counts the comparisons it made and fails if that count is zero (§6).
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.callbacks import cut_result_to_dense
from discopt.ml import DenseLayer, NetworkDefinition, NNFormulation
from discopt.modeling import Model

_rust = pytest.importorskip("discopt._rust")


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


def _build(net, ideal_cuts):
    m = Model("nn")
    nn = NNFormulation(m, net, strategy="relu_bigm", prefix="p", ideal_cuts=ideal_cuts)
    nn.formulate()
    m.minimize(nn.outputs[0])
    return m, nn


def _box(m):
    lo = np.concatenate([np.ravel(np.broadcast_to(v.lb, (v.size,))) for v in m._variables])
    hi = np.concatenate([np.ravel(np.broadcast_to(v.ub, (v.size,))) for v in m._variables])
    return lo.astype(np.float64), hi.astype(np.float64)


def _rust_sep(m, nn):
    spec = nn.cut_callback.native_milp_spec(m)
    assert spec is not None
    lo, hi = _box(m)
    sep = _rust.ReluIdealSeparator(
        int(lo.size),
        spec["y"],
        spec["q"],
        spec["bias"],
        spec["ptr"],
        spec["x"],
        spec["w"],
        np.ascontiguousarray(lo[spec["x"]]),
        np.ascontiguousarray(hi[spec["x"]]),
    )
    return sep, spec


def _rust_dense(cuts, n):
    out = []
    for cols, vals, rhs in cuts:
        row = np.zeros(n)
        np.add.at(row, cols, vals)
        out.append((row, rhs))
    return out


def _feasible_flat(m, net, x):
    """The exact (x, zhat, z, q) point of the big-M embedding at input x."""
    vals = {"p_input": np.asarray(x, dtype=np.float64)}
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
    return np.concatenate(
        [np.ravel(np.broadcast_to(vals[v.name], (v.size,))) for v in m._variables]
    )


@pytest.mark.parametrize("sizes,seed", [([2, 4, 1], 0), ([3, 5, 3, 1], 3), ([2, 6, 1], 0)])
def test_rust_separator_matches_python(sizes, seed):
    net = _net(sizes, seed)
    m, nn = _build(net, True)
    sep, spec = _rust_sep(m, nn)
    assert sep.n_neurons == len(spec["y"]) > 0
    lo, hi = _box(m)
    n = lo.size
    rng = np.random.default_rng(11)
    compared = 0
    for _ in range(300):
        x = rng.uniform(lo, hi)
        py = sorted(
            (tuple(np.round(c, 10)), round(r, 10))
            for c, r, _s in (cut_result_to_dense(c, m) for c in nn.cut_callback.separate(x, m))
        )
        rs = sorted(
            (tuple(np.round(c, 10)), round(r, 10))
            for c, r in _rust_dense(sep.separate(np.ascontiguousarray(x)), n)
        )
        assert len(py) == len(rs)
        for (cp, rp), (cr, rr) in zip(py, rs):
            np.testing.assert_allclose(cr, cp, rtol=1e-9, atol=1e-9)
            assert abs(rr - rp) <= 1e-9 * max(1.0, abs(rp))
            compared += 1
    assert compared > 0
    assert sep.calls == 300


@pytest.mark.parametrize("sizes", [[2, 4, 1], [3, 5, 3, 1]])
def test_rust_cuts_never_cut_a_feasible_point(sizes):
    net = _net(sizes, 0)
    m, nn = _build(net, True)
    sep, _ = _rust_sep(m, nn)
    lo, hi = _box(m)
    n = lo.size
    rng = np.random.default_rng(5)
    cuts = []
    for _ in range(200):
        cuts += _rust_dense(sep.separate(rng.uniform(lo, hi)), n)
    assert cuts, "separator produced no cuts to test"
    n_checks = 0
    for x in rng.uniform(-1, 1, (300, sizes[0])):
        pt = _feasible_flat(m, net, x)
        for row, rhs in cuts:
            assert row @ pt <= rhs + 1e-9 * max(1.0, abs(rhs))
            n_checks += 1
        assert sep.separate(pt) == []
    assert n_checks > 0


def test_rust_separator_refuses_unbounded_input():
    net = _net([2, 4, 1], 0)
    m, nn = _build(net, True)
    spec = nn.cut_callback.native_milp_spec(m)
    lo, hi = _box(m)
    x_lo = np.ascontiguousarray(lo[spec["x"]])
    x_lo[0] = -np.inf
    with pytest.raises(ValueError, match="finite box"):
        _rust.ReluIdealSeparator(
            int(lo.size),
            spec["y"],
            spec["q"],
            spec["bias"],
            spec["ptr"],
            spec["x"],
            spec["w"],
            x_lo,
            np.ascontiguousarray(hi[spec["x"]]),
        )


def _enumerated_min(net, k=201):
    g = np.linspace(-1.0, 1.0, k)
    pts = np.array(np.meshgrid(*([g] * net.input_size))).reshape(net.input_size, -1).T
    return float(np.min(net.forward(pts)[:, 0]))


def test_native_route_certifies_and_tightens_the_root(monkeypatch):
    # big-M LP -0.300 vs MILP -0.083 on this instance: the cuts have work to do.
    # The HiGHS route's root cut loop (HiGHS is opt-in since PR #2196); the Rust
    # engine's native separator is covered below.
    pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    net = _net([2, 6, 1], 0)
    ref, _ = _build(net, False)
    r0 = ref.solve(time_limit=60)
    m, nn = _build(net, True)
    r = m.solve(time_limit=60, cut_callback=nn.cut_callback)
    assert r.status == "optimal" and r0.status == "optimal"
    assert "native-relu-cuts" in (r.algorithm_route or ""), r.algorithm_route
    assert "HiGHS" in r.algorithm_route
    assert abs(float(r.objective) - float(r0.objective)) <= 1e-5
    # An independent oracle, not another formulation: the grid minimum is an
    # upper bound on the true minimum, and within a grid cell of it.
    assert float(r.objective) <= _enumerated_min(net) + 1e-6
    assert nn.cut_callback.n_calls > 0
    assert nn.cut_callback.n_cuts > 0
    assert r.solver_stats["ml/relu_ideal_native_cuts"] == nn.cut_callback.n_cuts
    # The root bound reported is the cut-strengthened one (the issue measured it
    # unchanged on the spatial tree). Min sense: tighter means higher.
    assert r.root_bound is not None and r0.root_bound is not None
    assert float(r.root_bound) > float(r0.root_bound) + 1e-4
    assert float(r.root_bound) <= float(r.objective) + 1e-6


def test_rust_engine_route_separates_at_nodes(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    # This test is about the native separator firing AT NODES. Since
    # `DISCOPT_MILP_ROOT_GMI_REPAIR` went default ON (highs-parity P4.4) the root's
    # GMI cuts close this small network at the root, so no node -- and no separator
    # call -- exists. Pin the legacy root so the node path is still exercised.
    monkeypatch.setenv("DISCOPT_MILP_ROOT_GMI_REPAIR", "0")
    # Likewise `DISCOPT_MILP_ALIGNED_SLACKS` (highs-parity D2, default ON 2026-11-07):
    # with the slacks aligned the root cut loop closes this network before any
    # fractional point reaches the separator (0 calls vs 3; both certify the same
    # optimum in 1 node). Pin the legacy layout so the separator path is exercised.
    monkeypatch.setenv("DISCOPT_MILP_ALIGNED_SLACKS", "0")
    net = _net([2, 6, 1], 0)
    ref, _ = _build(net, False)
    r0 = ref.solve(time_limit=60)
    m, nn = _build(net, True)
    r = m.solve(time_limit=60, cut_callback=nn.cut_callback)
    assert r.status == "optimal"
    assert "native-relu-cuts" in (r.algorithm_route or "")
    assert "Rust MILP engine" in r.algorithm_route
    assert abs(float(r.objective) - float(r0.objective)) <= 1e-5
    assert nn.cut_callback.n_calls > 0
    assert nn.cut_callback.n_cuts > 0
    assert r.root_bound is not None and float(r.root_bound) <= float(r.objective) + 1e-6


def test_wrapped_separator_keeps_the_python_callback_route():
    net = _net([2, 4, 1], 0)
    ref, _ = _build(net, False)
    r0 = ref.solve(time_limit=60)
    m, nn = _build(net, True)

    def cb(ctx, model):
        return nn.cut_callback(ctx, model)

    r = m.solve(time_limit=120, cut_callback=cb)
    assert r.status == "optimal"
    assert "native-relu-cuts" not in (r.algorithm_route or "")
    assert abs(float(r.objective) - float(r0.objective)) <= 1e-5
    assert nn.cut_callback.n_calls > 0
    assert "ml/relu_ideal_native_cuts" not in (r.solver_stats or {})


def test_relu_inputs_boxed_declines_the_default_box():
    """The default box (9.999e19) means "no bound": both routes must decline the cuts.

    The previous guards were ``np.isfinite`` (Rust route) and ``|x| < 1e20`` (HiGHS
    route); both accepted the default box, which is finite and below the sentinel.
    """
    from discopt.constants import DEFAULT_VARIABLE_BOUND
    from discopt.solver import _relu_inputs_boxed

    lo = np.array([-1.0, -2.0])
    hi = np.array([1.0, 3.0])
    checks = 0
    assert _relu_inputs_boxed(lo, hi)
    checks += 1
    for bad in (DEFAULT_VARIABLE_BOUND, 1e20, np.inf):
        assert not _relu_inputs_boxed(lo, np.array([1.0, bad]))
        assert not _relu_inputs_boxed(np.array([-bad, -2.0]), hi)
        checks += 2
    assert not _relu_inputs_boxed(lo, np.array([1.0, np.nan]))
    checks += 1
    # A large but genuine bound below the default box is kept.
    assert _relu_inputs_boxed(lo, np.array([1.0, 1e6]))
    checks += 1
    assert checks == 9
