"""T-N3.2: LP-tightened ReLU big-M constants (``DISCOPT_NN_LP_BIGM``).

``discopt.ml.bounds.lp_tighten_bounds`` replaces interval bounds with the proved
LP bounds over the triangle relaxation of the earlier layers (Tjeng, Xiao &
Tedrake 2019; Grimstad & Andersson 2019). Entry experiment E1,
docs/dev/nn-module-plan.md §6, prompted by PWLNet (Sanyal et al. 2026).

It is a bound-changing change (CLAUDE.md §5), so the tests are:

* differential bounds -- never looser than interval arithmetic, and never
  tighter than the true min / max of the pre-activation over the input box
  (solved exactly as a MILP on small nets);
* feasible-point sampling -- with the flag on, fixing the input to any sampled
  point leaves the embedding feasible and its output equal to ``forward``;
* ``=0`` is the interval path, byte for byte; unset is ON for ``relu_bigm``
  and OFF for ``relu_partition`` and ``relu_extended``.

Each test counts what it checked and fails if it checked nothing (§6).
"""

from __future__ import annotations

import discopt
import numpy as np
import pytest
from discopt.ml.bounds import lp_tighten_bounds, propagate_bounds
from discopt.ml.formulations.base import NNFormulation
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition

pytestmark = pytest.mark.unit


def _relu_net(widths, seed, d=2, box=2.0):
    rng = np.random.default_rng(seed)
    sizes = [d, *widths, 1]
    layers = []
    for i in range(len(sizes) - 1):
        act = Activation.RELU if i < len(sizes) - 2 else Activation.LINEAR
        W = rng.normal(0, 1.0 / np.sqrt(sizes[i]), (sizes[i], sizes[i + 1]))
        b = rng.normal(0, 0.3, sizes[i + 1])
        layers.append(DenseLayer(W, b, act))
    return NetworkDefinition(layers, input_bounds=(np.full(d, -box), np.full(d, box)))


NETS = [((8, 8), 0), ((12, 8, 6), 1), ((16, 16), 2), ((10, 10, 10), 3)]


@pytest.mark.parametrize("widths, seed", NETS)
def test_never_looser_than_interval_and_fires(widths, seed):
    net = _relu_net(widths, seed)
    iv = propagate_bounds(net)
    st: dict = {}
    lp = lp_tighten_bounds(net, stats=st)
    assert st["lps"] > 0 and st["declined"] == 0
    checked = 0
    for a, b in zip(iv, lp, strict=True):
        assert np.all(b.pre_lb >= a.pre_lb) and np.all(b.pre_ub <= a.pre_ub)
        assert np.all(b.pre_lb <= b.pre_ub)
        checked += a.pre_lb.size
    assert checked
    # Layer 0 is exact under interval arithmetic and is not re-solved.
    np.testing.assert_array_equal(lp[0].pre_lb, iv[0].pre_lb)
    widths_iv = np.concatenate([b.pre_ub - b.pre_lb for b in iv[1:]])
    widths_lp = np.concatenate([b.pre_ub - b.pre_lb for b in lp[1:]])
    assert np.sum(widths_lp) < np.sum(widths_iv), "LP tightened nothing on a deep net"


ORACLE_NETS = [((6, 6), 0), ((5, 4, 3), 1), ((8, 6), 2)]


@pytest.mark.parametrize("widths, seed", ORACLE_NETS)
def test_bounds_hold_the_exact_box_optimum(widths, seed):
    """Differential bound test: each LP bound is on the right side of the
    exact min / max of its pre-activation (a MILP over the interval big-M
    encoding, which is exact for the network)."""
    net = _relu_net(widths, seed)
    lp = lp_tighten_bounds(net)
    checked = 0
    for k in range(1, len(net.layers)):
        for j in range(net.layers[k].weights.shape[1]):
            for sense in (1, -1):
                m = discopt.Model("oracle")
                f = NNFormulation(m, net, strategy="relu_bigm", prefix="nn")
                f.formulate()
                zh = next(v for v in m._variables if v.name == f"nn_zhat_{k}")
                m.minimize(sense * zh[j])
                r = m.solve(time_limit=60)
                assert r.status == "optimal"
                exact = sense * r.objective
                if sense > 0:
                    assert lp[k].pre_lb[j] <= exact + 1e-7
                else:
                    assert lp[k].pre_ub[j] >= exact - 1e-7
                checked += 1
    assert checked


@pytest.mark.parametrize("widths, seed", NETS)
def test_sampled_points_stay_inside(widths, seed):
    net = _relu_net(widths, seed)
    lp = lp_tighten_bounds(net)
    rng = np.random.default_rng(seed + 100)
    lo, hi = net.input_bounds
    X = np.vstack(
        [
            rng.uniform(lo, hi, size=(20000, 2)),
            rng.integers(0, 2, size=(400, 2)) * (hi - lo) + lo,  # box vertices
        ]
    )
    checked = 0
    h = X
    for k, layer in enumerate(net.layers):
        pre = h @ layer.weights + layer.biases
        assert np.all(pre >= lp[k].pre_lb - 1e-9) and np.all(pre <= lp[k].pre_ub + 1e-9)
        checked += pre.size
        h = np.maximum(pre, 0.0) if layer.activation == Activation.RELU else pre
    assert checked


@pytest.mark.parametrize("strategy", ["relu_bigm", "relu_partition", "relu_extended"])
def test_fixed_input_reproduces_forward_with_the_flag(monkeypatch, strategy):
    """Feasible-point sampling through the embedding: tightened constants must
    not cut any input's true network output."""
    monkeypatch.setenv("DISCOPT_NN_LP_BIGM", "1")
    net = _relu_net((10, 8), 5)
    rng = np.random.default_rng(9)
    checked = 0
    for x0 in rng.uniform(-2.0, 2.0, size=(4, 2)):
        m = discopt.Model("fix")
        f = NNFormulation(m, net, strategy=strategy, prefix="nn")
        f.formulate()
        m.subject_to(f.inputs == x0)
        m.minimize(f.outputs[0])
        r = m.solve(time_limit=60)
        assert r.status == "optimal"
        assert r.objective == pytest.approx(float(net.forward(x0[None, :])[0, 0]), abs=1e-6)
        checked += 1
    assert checked == 4


def _strategy_obj(net, strategy):
    m = discopt.Model("f")
    f = NNFormulation(m, net, strategy=strategy, prefix="nn")
    f.formulate()
    return f._strategy


def _uses_lp(obj, net):
    """True if ``obj`` was built on LP-tightened bounds, False if on interval
    bounds (asserted byte for byte)."""
    if hasattr(obj, "lp_bound_stats"):
        assert obj.lp_bound_stats["lps"] > 0
        return True
    for a, b in zip(obj.layer_bounds, propagate_bounds(net), strict=True):
        np.testing.assert_array_equal(a.pre_lb, b.pre_lb)
        np.testing.assert_array_equal(a.pre_ub, b.pre_ub)
    return False


# (env value, strategy) -> LP-tightened? Unset is each formulation's default:
# ON for relu_bigm, OFF for relu_partition and relu_extended (their panels).
FLAG_TABLE = [
    (None, "relu_bigm", True),
    (None, "relu_partition", False),
    ("1", "relu_partition", True),
    (None, "relu_extended", False),
    ("1", "relu_bigm", True),
    ("1", "relu_extended", True),
    ("0", "relu_bigm", False),
    ("0", "relu_partition", False),
    ("0", "relu_extended", False),
]


def test_flag_default_and_overrides_per_formulation(monkeypatch):
    net = _relu_net((12, 8, 6), 1)
    checked = 0
    for env, strategy, expect in FLAG_TABLE:
        if env is None:
            monkeypatch.delenv("DISCOPT_NN_LP_BIGM", raising=False)
        else:
            monkeypatch.setenv("DISCOPT_NN_LP_BIGM", env)
        assert _uses_lp(_strategy_obj(net, strategy), net) is expect, (env, strategy)
        checked += 1
    assert checked == len(FLAG_TABLE)


def test_on_and_off_certify_the_same_optimum(monkeypatch):
    net = _relu_net((12, 8, 6), 1)
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_NN_LP_BIGM", flag)
        m = discopt.Model("s")
        f = NNFormulation(m, net, strategy="relu_bigm", prefix="nn")
        f.formulate()
        m.minimize(f.outputs[0])
        out[flag] = m.solve(time_limit=60)
    assert out["0"].status == out["1"].status == "optimal"
    assert out["0"].objective == pytest.approx(out["1"].objective, abs=1e-6)


def test_work_budget_stops_cleanly_on_interval_bounds():
    net = _relu_net((10, 10, 10), 3)
    st: dict = {}
    lp = lp_tighten_bounds(net, work_budget=1, stats=st)
    assert st["lps"] == 0 and st["stopped_at"] == 1
    for a, b in zip(propagate_bounds(net), lp, strict=True):
        np.testing.assert_array_equal(a.pre_lb, b.pre_lb)
        np.testing.assert_array_equal(a.pre_ub, b.pre_ub)


def test_scaled_input_box_is_respected(monkeypatch):
    """The tightening runs over the box the layers consume (F1 / T-N0.2)."""
    from discopt.ml.scaling import OffsetScaling

    monkeypatch.setenv("DISCOPT_NN_LP_BIGM", "1")
    net = _relu_net((8, 8), 4)
    sc = OffsetScaling(
        x_offset=np.array([1.0, -1.0]),
        x_factor=np.array([2.0, 0.5]),
        y_offset=np.array([0.0]),
        y_factor=np.array([1.0]),
    )
    m = discopt.Model("sc")
    raw_box = (np.array([-3.0, -2.0]), np.array([5.0, 0.0]))
    net.input_bounds = raw_box
    f = NNFormulation(m, net, strategy="relu_bigm", prefix="nn", scaling=sc)
    f.formulate()
    x0 = np.array([4.0, -1.5])
    m.subject_to(f.inputs == x0)
    m.minimize(f.outputs[0])
    r = m.solve(time_limit=60)
    s = (x0 - sc.x_offset) / sc.x_factor
    assert r.status == "optimal"
    assert r.objective == pytest.approx(float(net.forward(s[None, :])[0, 0]), abs=1e-6)
