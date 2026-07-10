"""Tests for the trainable ML surrogate API (``discopt.nn.trainable``).

Covers the HM1 acceptance criteria from
``docs/dev/hybrid-ml-implementation-plan.md`` §5:

1. shape/broadcast contract + actionable errors;
2. ReLU refusal on the trainable path;
3. end-to-end neural-DAE training (smoke) reproduces the prototype;
4. kernel-surrogate variant;
5. freeze / from_definition round-trip;
6. Gauss-Newton detector fires on the training objective.

Tests 3, 4, 6 build a small hybrid batch-reactor model (physics mass balances +
a trainable rate law) and solve it, so they are marked ``smoke``.
"""

import numpy as np
import pytest
from discopt.dae import ContinuousSet, DAEBuilder
from discopt.modeling import Model
from discopt.nn import (
    TrainableKernelExpansion,
    TrainableNetwork,
    train,
)
from discopt.nn.network import Activation

# ── ground truth for the reactor problem (mirrors scripts/hybrid_ml) ────────

CA0, CB0, TF = 1.0, 0.0, 2.0
NOISE = 0.01


def _r_true(c):
    return 1.5 * c**2 / (0.3 + c)


def _make_data(seed=0, n_obs=15):
    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(seed)
    t_data = np.linspace(0.05, TF, n_obs)
    sol = solve_ivp(
        lambda t, y: [-_r_true(y[0]), _r_true(y[0])],
        (0, TF),
        [CA0, CB0],
        t_eval=t_data,
        rtol=1e-10,
        atol=1e-12,
    )
    yA = sol.y[0] + NOISE * rng.standard_normal(n_obs)
    yB = sol.y[1] + NOISE * rng.standard_normal(n_obs)
    return t_data, yA, yB


def _build_hybrid(surrogate_factory, *, nfe=20, ncp=2):
    """Build the reactor hybrid model given a factory ``m -> (surrogate, rate_fn)``."""
    t_data, yA, yB = _make_data()
    m = Model()
    surrogate, rate = surrogate_factory(m)

    cs = ContinuousSet("t", (0.0, TF), nfe=nfe, ncp=ncp, scheme="radau")
    dae = DAEBuilder(m, cs)
    dae.add_state("cA", bounds=(0.0, 1.5), initial=CA0)
    dae.add_state("cB", bounds=(0.0, 1.5), initial=CB0)
    dae.set_ode(lambda t, s, a, u: {"cA": -rate(s["cA"]), "cB": rate(s["cA"])})
    dvars = dae.discretize()

    lsq = dae.least_squares("cA", t_data, yA) + dae.least_squares("cB", t_data, yB)
    m.minimize(lsq + 1e-4 * surrogate.l2_penalty())

    tp = dae._element_points()
    cA0 = np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CA0], yA]))
    cB0 = np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CB0], yB]))
    state_init = {dvars["cA"]: cA0, dvars["cB"]: cB0}
    return m, dae, surrogate, state_init, (t_data, yA, yB)


# ── test 1: shape / broadcast contract ──────────────────────────────────────


def test_dense_and_network_shape_contract():
    m = Model()

    # 1-feature network accepts a bare (nfe, ncp) expression.
    net1 = TrainableNetwork(m, [1, 4, 1], activation="tanh", name="n1")
    c = m.continuous("c", shape=(5, 2), lb=0.0, ub=1.0)
    out = net1(c[:, :])
    assert out is not None
    assert net1.n_parameters() == (1 * 4 + 4) + (4 * 1 + 1)

    # 2-feature network: scalar-per-point OR a single pre-stacked input.
    net2 = TrainableNetwork(m, [2, 3, 1], activation="softplus", name="n2")
    assert net2(c[:, 0], c[:, 1]) is not None  # two scalar-per-point args
    d = m.continuous("d", shape=(5, 2), lb=0.0, ub=1.0)
    assert net2(d) is not None  # one pre-stacked arg (trailing axis == 2)

    # scalar input.
    s = m.continuous("s", lb=0.0, ub=1.0)
    assert net1(s) is not None


def test_wrong_arg_count_raises_with_layer_name():
    m = Model()
    net = TrainableNetwork(m, [3, 4, 1], activation="tanh", name="badcall")
    c = m.continuous("c", shape=(5,), lb=0, ub=1)
    # 3-feature net given 2 args: neither n_in scalar-per-point nor 1 pre-stacked.
    with pytest.raises(ValueError, match=r"badcall_L0"):
        net(c, c)


# ── test 2: ReLU refusal ────────────────────────────────────────────────────


def test_relu_refused_on_trainable_path():
    m = Model()
    with pytest.raises(ValueError, match=r"RELU is not supported.*softplus"):
        TrainableNetwork(m, [1, 3, 1], activation="relu", name="r")
    with pytest.raises(ValueError, match=r"RELU is not supported"):
        from discopt.nn import TrainableDense

        TrainableDense(m, 1, 3, activation=Activation.RELU, name="d")


def test_unknown_activation_raises():
    m = Model()
    with pytest.raises(ValueError, match=r"Unknown activation"):
        TrainableNetwork(m, [1, 3, 1], activation="gelu", name="g")


# ── test 3: end-to-end neural-DAE training (smoke) ──────────────────────────


@pytest.mark.smoke
def test_neural_dae_training_end_to_end():
    """Rebuild scripts/hybrid_ml/neural_dae_prototype.py on the new API.

    Thresholds carry >= 2x margin on the measured values (15-19 iters, rate
    RMSE 0.0107, resim < 1x noise): they gate regressions, not luck.
    """
    from scipy.integrate import solve_ivp

    def factory(m):
        net = TrainableNetwork(m, [1, 6, 1], activation="tanh", weight_bounds=(-8, 8), name="rate")
        return net, net

    m, dae, net, state_init, (t_data, yA, yB) = _build_hybrid(factory)
    res = train(
        m,
        initial_solution=state_init | net.initial_values(seed=0),
        options={"max_iter": 3000, "tol": 1e-8},
    )
    assert res.status.name == "OPTIMAL"
    assert res.iterations <= 40

    frozen = net.freeze(res)
    c_grid = np.linspace(float(yA.min()), CA0, 100)
    r_hat = frozen.forward(c_grid.reshape(-1, 1)).ravel()
    rate_rmse = float(np.sqrt(np.mean((r_hat - _r_true(c_grid)) ** 2)))
    assert rate_rmse <= 0.03, f"rate RMSE {rate_rmse:.4f} exceeds 0.03"

    # Honest generalization: integrate the learned ODE and compare to truth.
    def _r_hat(c):
        return float(frozen.forward(np.array([[c]]))[0, 0])

    truth = solve_ivp(
        lambda t, y: [-_r_true(y[0]), _r_true(y[0])],
        (0, TF),
        [CA0, CB0],
        t_eval=np.linspace(0, TF, 40),
        rtol=1e-10,
        atol=1e-12,
    )
    learned = solve_ivp(
        lambda t, y: [-_r_hat(y[0]), _r_hat(y[0])],
        (0, TF),
        [CA0, CB0],
        t_eval=truth.t,
        rtol=1e-9,
        atol=1e-11,
    )
    resim_rmse = float(np.sqrt(np.mean((learned.y[0] - truth.y[0]) ** 2)))
    assert resim_rmse <= 2 * NOISE, f"resim RMSE {resim_rmse:.4f} exceeds {2 * NOISE}"


# ── test 4: kernel surrogate variant ────────────────────────────────────────


@pytest.mark.smoke
def test_kernel_surrogate_training():
    """Linear-in-alpha kernel surrogate: zero init, few iterations."""

    def factory(m):
        ke = TrainableKernelExpansion(m, np.linspace(0.0, 1.05, 12), lengthscale=0.12, name="k")
        return ke, ke

    m, dae, ke, state_init, _ = _build_hybrid(factory)
    res = train(
        m,
        initial_solution=state_init | ke.initial_values(),
        options={"max_iter": 3000, "tol": 1e-8},
    )
    assert res.status.name == "OPTIMAL"
    assert res.iterations <= 25


# ── test 5: freeze / from_definition round-trip ─────────────────────────────


def test_freeze_matches_symbolic_and_from_definition_round_trips():
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.warm_start import unflatten_solution, validate_initial_solution

    m = Model()
    net = TrainableNetwork(m, [1, 5, 3, 1], activation="softplus", name="net")

    # A concrete point in weight space (the model has only weights, no other vars).
    vals = net.initial_values(seed=3)
    x = validate_initial_solution(m, vals)
    var_vals = unflatten_solution(m, x)

    frozen = net.freeze(var_vals)

    # freeze().forward() must match a numpy re-evaluation of the symbolic net at
    # 10 random points to 1e-8.
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.0, 1.0, size=(10, 1))

    def _np_forward(xin):
        h = xin
        acts = {
            Activation.SOFTPLUS: lambda z: np.logaddexp(z, 0.0),
            Activation.LINEAR: lambda z: z,
        }
        for layer in net.layers:
            W = np.asarray(var_vals[layer.W]).reshape(layer.n_in, layer.n_out)
            b = np.asarray(var_vals[layer.b]).reshape(layer.n_out)
            h = acts[layer.activation](h @ W + b)
        return h

    np.testing.assert_allclose(frozen.forward(pts), _np_forward(pts), atol=1e-8)

    # from_definition rebuilds a trainable net whose initial values reproduce the
    # frozen weights exactly.
    m2 = Model()
    net2 = TrainableNetwork.from_definition(m2, frozen, name="net2")
    assert net2.sizes == net.sizes
    seed_vals = net2.initial_values_from_definition()
    for layer2, (W, b) in zip(net2.layers, [(dl.weights, dl.biases) for dl in frozen.layers]):
        np.testing.assert_allclose(seed_vals[layer2.W], W, atol=1e-12)
        np.testing.assert_allclose(seed_vals[layer2.b], b, atol=1e-12)

    # NLPResult-shaped freeze also works (duck-typed .x).
    class _FakeNLPResult:
        def __init__(self, xflat):
            self.x = xflat

    frozen_from_result = net.freeze(_FakeNLPResult(x))
    np.testing.assert_allclose(frozen_from_result.forward(pts), frozen.forward(pts), atol=1e-12)
    # silence unused-evaluator import (kept to document the training entry type)
    assert NLPEvaluator is not None


# ── test 6: Gauss-Newton detector fires on the training objective ───────────


@pytest.mark.smoke
def test_gauss_newton_detector_fires_on_training_objective():
    from discopt._jax.nlp_evaluator import NLPEvaluator

    def factory(m):
        net = TrainableNetwork(m, [1, 6, 1], activation="tanh", name="rate")
        return net, net

    m, dae, net, state_init, _ = _build_hybrid(factory)
    ev = NLPEvaluator(m, gauss_newton=True)
    assert ev.is_gauss_newton is True
