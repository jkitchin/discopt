"""#2135: beta-parameterized softplus, squareplus/CHKS, Swish/SiLU and GELU
activations for embedded networks, and the torch reader dropping
``nn.Softplus(beta=...)``.

Covers, per the issue:

- ``NetworkDefinition.forward`` against an independent scalar reference
  (``math`` module, written from the formulas in the issue) for every new
  activation and several ``beta``;
- interval bounds containing dense samples of the activation, including
  intervals that straddle the Swish/GELU minimizer (where the old
  "monotone: [f(lb), f(ub)]" rule is wrong), and being tight there;
- a certified global solve of a small network with each activation matching
  dense enumeration, in every formulation that accepts it;
- the torch reader carrying ``beta`` (and refusing a ``threshold`` or GELU
  approximation it cannot embed), exercised without torch through a stand-in
  ``nn`` namespace and, when torch is installed, against a real
  ``nn.Softplus(beta=10)`` network.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
from discopt.ml.activations import (
    activation_interval,
    activation_minimizer,
    evaluate,
    symbolic,
)
from discopt.ml.bounds import propagate_bounds
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition

NEW_SMOOTH = [Activation.SOFTPLUS, Activation.SQUAREPLUS, Activation.SWISH, Activation.GELU]
BETAS = [0.5, 1.0, 3.0, 10.0]


# ── independent scalar references (from the issue's formulas) ──────────────


def _ref_scalar(act: Activation, z: float, beta: float) -> float:
    if act == Activation.SOFTPLUS:
        # (1/beta) log(1 + e^{beta z}), stable form
        u = beta * z
        return (max(u, 0.0) + math.log1p(math.exp(-abs(u)))) / beta
    if act == Activation.SQUAREPLUS:
        b = 4.0 / beta**2  # b = 4 mu^2 with mu = 1/beta
        return 0.5 * (z + math.sqrt(z * z + b))
    if act == Activation.SWISH:
        u = beta * z
        sig = 1.0 / (1.0 + math.exp(-u)) if u >= 0 else math.exp(u) / (1.0 + math.exp(u))
        return z * sig
    if act == Activation.GELU:
        return z * 0.5 * (1.0 + math.erf(beta * z / math.sqrt(2.0)))
    raise AssertionError(act)


# ── forward ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("act", NEW_SMOOTH)
@pytest.mark.parametrize("beta", BETAS)
def test_forward_matches_reference(act, beta):
    rng = np.random.default_rng(0)
    W1 = rng.normal(size=(2, 4))
    b1 = rng.normal(size=4)
    W2 = rng.normal(size=(4, 1))
    b2 = rng.normal(size=1)
    net = NetworkDefinition(
        [DenseLayer(W1, b1, act, beta=beta), DenseLayer(W2, b2, Activation.LINEAR)]
    )
    n_checked = 0
    for _ in range(25):
        x = rng.uniform(-4, 4, size=2)
        h = x @ W1 + b1
        h = np.array([_ref_scalar(act, float(v), beta) for v in h])
        ref = h @ W2 + b2
        np.testing.assert_allclose(net.forward(x), ref, rtol=1e-12, atol=1e-12)
        n_checked += 1
    assert n_checked == 25


def test_extreme_inputs_are_finite_and_correct():
    z = np.array([-np.inf, -1e300, -800.0, 0.0, 800.0, 1e300, np.inf])
    for act in NEW_SMOOTH:
        for beta in BETAS:
            v = evaluate(act, z, beta)
            assert not np.any(np.isnan(v)), (act, beta, v)
            # relu-like at both tails
            assert abs(v[0]) == 0.0 and abs(v[1]) < 1e-200
            assert v[-1] == np.inf and v[-2] == pytest.approx(1e300)


def test_softplus_beta_one_is_unchanged():
    z = np.linspace(-30, 30, 121)
    np.testing.assert_array_equal(evaluate(Activation.SOFTPLUS, z), np.logaddexp(z, 0))


def test_silu_alias_and_string_activation():
    assert Activation("silu") is Activation.SWISH
    layer = DenseLayer(np.eye(1), np.zeros(1), "gelu", beta=2.0)
    assert layer.activation is Activation.GELU and layer.beta == 2.0


@pytest.mark.parametrize(
    "act", [Activation.LINEAR, Activation.RELU, Activation.SIGMOID, Activation.TANH]
)
def test_beta_refused_on_unparameterized_activation(act):
    with pytest.raises(ValueError, match="takes no beta"):
        DenseLayer(np.eye(1), np.zeros(1), act, beta=2.0)


@pytest.mark.parametrize("beta", [0.0, -1.0, float("inf"), float("nan"), "x"])
def test_invalid_beta_refused(beta):
    with pytest.raises(ValueError, match="positive finite"):
        DenseLayer(np.eye(1), np.zeros(1), Activation.SOFTPLUS, beta=beta)


# ── bounds ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "act,u_star,g_star",
    [
        (Activation.SWISH, -1.2784645427610738, -0.2784645427610738),
        (Activation.GELU, -0.7517915, -0.1699712),
    ],
)
def test_minimizer_values(act, u_star, g_star):
    for beta in BETAS:
        z, f = activation_minimizer(act, beta)
        assert z == pytest.approx(u_star / beta, rel=1e-6)
        assert f == pytest.approx(g_star / beta, rel=1e-6)
        # It really is the minimum of the sampled function.
        grid = np.linspace(z - 5 / beta, z + 5 / beta, 200001)
        assert evaluate(act, grid, beta).min() >= f - 1e-15


@pytest.mark.parametrize("act", NEW_SMOOTH)
@pytest.mark.parametrize("beta", BETAS)
def test_interval_contains_samples(act, beta):
    rng = np.random.default_rng(1)
    n_checked = 0
    n_straddle = 0
    z_star = activation_minimizer(act, beta)[0] if act in (Activation.SWISH, Activation.GELU) else 0
    los = list(rng.uniform(-6, 2, size=60)) + [z_star - 0.1, z_star - 3.0, z_star + 1e-9]
    widths = list(rng.uniform(0.01, 8, size=60)) + [0.2, 2.9, 1.0]
    for lo, w in zip(los, widths):
        hi = lo + w
        lb, ub = activation_interval(act, np.array([lo]), np.array([hi]), beta)
        grid = np.linspace(lo, hi, 4001)
        if lo <= z_star <= hi:
            grid = np.append(grid, z_star)  # the true minimum, for the tightness check
        samples = evaluate(act, grid, beta)
        assert lb[0] <= samples.min() + 1e-15, (act, beta, lo, hi)
        assert ub[0] >= samples.max() - 1e-15, (act, beta, lo, hi)
        # Exact image (the lower end is only lowered by the 1e-12 safety margin).
        assert samples.min() - lb[0] <= 1e-10 * max(1.0, abs(samples.min()))
        assert ub[0] - samples.max() <= 1e-12 * max(1.0, abs(samples.max()))
        if lo <= z_star <= hi:
            n_straddle += 1
        n_checked += 1
    assert n_checked == 63
    if act in (Activation.SWISH, Activation.GELU):
        assert n_straddle >= 3


@pytest.mark.parametrize("act", [Activation.SWISH, Activation.GELU])
def test_monotone_rule_would_be_wrong_when_straddling(act):
    """The pre-#2135 rule [f(lb), f(ub)] excludes reachable values."""
    lo, hi = -2.0, 0.0
    lb, _ = activation_interval(act, np.array([lo]), np.array([hi]))
    naive_lb = min(evaluate(act, lo), evaluate(act, hi))
    true_min = evaluate(act, np.linspace(lo, hi, 100001)).min()
    assert naive_lb > true_min + 1e-2  # the old rule cuts the minimum off
    assert lb[0] <= true_min


def test_propagate_bounds_uses_beta():
    W = np.array([[1.0]])
    b = np.zeros(1)
    for act in NEW_SMOOTH:
        for beta in BETAS:
            net = NetworkDefinition(
                [DenseLayer(W, b, act, beta=beta)],
                input_bounds=(np.array([-3.0]), np.array([2.0])),
            )
            lb = propagate_bounds(net)[0]
            samples = net.forward(np.linspace(-3, 2, 20001)[:, None]).ravel()
            assert lb.post_lb[0] <= samples.min() + 1e-15
            assert lb.post_ub[0] >= samples.max() - 1e-15
            assert lb.post_ub[0] == pytest.approx(samples.max(), rel=1e-12)


# ── symbolic form ──────────────────────────────────────────────────────────


def test_symbolic_beta_one_softplus_is_the_plain_intrinsic():
    """beta = 1 must emit the expression formulations emitted before #2135."""
    import discopt.modeling as dm

    m = dm.Model("s")
    x = m.continuous("x", lb=-1, ub=1)
    assert repr(symbolic(Activation.SOFTPLUS, x, 1.0)) == repr(dm.softplus(x))
    assert repr(symbolic(Activation.SIGMOID, x)) == repr(dm.sigmoid(x))


# ── global solves vs dense enumeration ─────────────────────────────────────


def _tiny_net(act: Activation, beta: float) -> NetworkDefinition:
    # 1 input -> 2 hidden (act, beta) -> 1 linear output, with weights chosen
    # so the hidden pre-activations straddle the Swish/GELU minimizer and the
    # output is non-monotone in x.
    W1 = np.array([[1.5, -1.0]])
    b1 = np.array([0.3, -0.4])
    W2 = np.array([[1.0], [0.7]])
    b2 = np.array([0.1])
    return NetworkDefinition(
        [DenseLayer(W1, b1, act, beta=beta), DenseLayer(W2, b2, Activation.LINEAR)],
        input_bounds=(np.array([-2.5]), np.array([2.5])),
    )


@pytest.mark.parametrize("strategy", ["full_space", "reduced_space", "relu_bigm"])
@pytest.mark.parametrize(
    "act,beta",
    [
        (Activation.SOFTPLUS, 1.0),
        (Activation.SOFTPLUS, 10.0),
        (Activation.SQUAREPLUS, 1.0),
        (Activation.SQUAREPLUS, 4.0),
        (Activation.SWISH, 1.0),
        (Activation.SWISH, 3.0),
        (Activation.GELU, 1.0),
        (Activation.GELU, 3.0),
    ],
)
def test_global_solve_matches_dense_sampling(act, beta, strategy):
    import discopt.modeling as dm
    from discopt.ml.formulations.base import NNFormulation

    net = _tiny_net(act, beta)
    m = dm.Model("nn2135")
    form = NNFormulation(m, net, strategy=strategy, prefix="p")
    form.formulate()
    x = form.inputs
    m.minimize(form.outputs[0])
    r = m.solve(time_limit=120)
    assert r.status == "optimal", r.status

    xs = np.linspace(-2.5, 2.5, 200001)
    vals = net.forward(xs[:, None]).ravel()
    ref_min = float(vals.min())
    x_star = float(np.asarray(r.value(x)).ravel()[0])
    # The incumbent re-evaluates through the network to the reported objective,
    assert float(net.forward(np.array([x_star]))[0]) == pytest.approx(r.objective, abs=1e-5)
    # and the certified optimum is the enumerated one (not above, not below).
    assert r.objective == pytest.approx(ref_min, abs=1e-5)


# ── torch reader ───────────────────────────────────────────────────────────


def _fake_nn():
    """A stand-in for ``torch.nn`` with only what ``_torch_activation`` reads."""

    class ReLU: ...

    class Sigmoid: ...

    class Tanh: ...

    class SiLU: ...

    class Softplus:
        def __init__(self, beta=1.0, threshold=20.0):
            self.beta = beta
            self.threshold = threshold

    class GELU:
        def __init__(self, approximate="none"):
            self.approximate = approximate

    return types.SimpleNamespace(
        ReLU=ReLU, Sigmoid=Sigmoid, Tanh=Tanh, SiLU=SiLU, Softplus=Softplus, GELU=GELU
    )


def test_torch_softplus_beta_is_carried():
    from discopt.ml.readers.torch_reader import _torch_activation

    nn = _fake_nn()
    assert _torch_activation(nn.Softplus(beta=10), nn) == (Activation.SOFTPLUS, 10.0)
    assert _torch_activation(nn.Softplus(), nn) == (Activation.SOFTPLUS, 1.0)
    assert _torch_activation(nn.SiLU(), nn) == (Activation.SWISH, 1.0)
    assert _torch_activation(nn.GELU(), nn) == (Activation.GELU, 1.0)
    assert _torch_activation(nn.ReLU(), nn) == (Activation.RELU, 1.0)


def test_torch_unrepresentable_parameters_refused():
    from discopt.ml.readers.torch_reader import _torch_activation

    nn = _fake_nn()
    with pytest.raises(ValueError, match="threshold"):
        _torch_activation(nn.Softplus(beta=1, threshold=5), nn)
    with pytest.raises(ValueError, match="approximate"):
        _torch_activation(nn.GELU(approximate="tanh"), nn)
    # beta scales the threshold gap: large beta tolerates a smaller threshold
    # only if log1p(exp(-t))/beta stays tiny; t=20 is fine for any beta >= 1.
    assert _torch_activation(nn.Softplus(beta=100, threshold=20), nn)[1] == 100.0


def test_torch_softplus_beta10_roundtrip():
    torch = pytest.importorskip("torch")
    from discopt.ml.readers.torch_reader import load_torch_sequential

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 5), torch.nn.Softplus(beta=10), torch.nn.Linear(5, 1)
    ).double()
    net = load_torch_sequential(model)
    assert net.layers[0].activation is Activation.SOFTPLUS
    assert net.layers[0].beta == 10.0
    x = np.random.default_rng(0).uniform(-2, 2, size=(50, 2))
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(net.forward(x), ref, rtol=1e-9, atol=1e-9)


# ── trainable path carries beta ────────────────────────────────────────────


def test_trainable_freeze_and_from_definition_carry_beta():
    import discopt.modeling as dm
    from discopt.ml.trainable import TrainableNetwork

    m = dm.Model("t")
    net = TrainableNetwork(m, [1, 3, 1], activation="swish", beta=4.0, name="tn")
    assert [ly.beta for ly in net.layers] == [4.0, 1.0]
    vals = {}
    rng = np.random.default_rng(0)
    for ly in net.layers:
        vals[ly.W] = rng.normal(size=(ly.n_in, ly.n_out))
        vals[ly.b] = rng.normal(size=ly.n_out)
    frozen = net.freeze(vals)
    assert frozen.layers[0].activation is Activation.SWISH
    assert frozen.layers[0].beta == 4.0

    m2 = dm.Model("t2")
    net2 = TrainableNetwork.from_definition(m2, frozen, name="tn2")
    assert [ly.beta for ly in net2.layers] == [4.0, 1.0]
