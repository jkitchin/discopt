"""``strategy="relu_icnn"``: input-convex ReLU networks as an LP, no binaries.

From J. Kitchin, "MIP surrogates" (2026-10-28), raised by PWLNet (Sanyal et al.,
arXiv:2611.25969); ICNNs are Amos, Xu & Kolter (ICML 2017). Entry experiment E2
(docs/dev/nn-module-plan.md §6) measured it at a median of 23x the big-M MILP.

The epigraph rows are a relaxation of the network's graph, exact only where a
smaller output is preferred, so the tests cover both directions:

* exact uses (output minimised, or bounded above with another objective)
  certify the same optimum as the exact ``relu_bigm`` encoding, with the
  reported point on the graph;
* uses that reward a larger output (maximised, bounded below) are refused
  with ``ICNNExactnessError``, never answered;
* a network that is not input-convex is refused at build time.

Each test counts its checks and fails if none ran (CLAUDE.md §6).
"""

from __future__ import annotations

import pickle

import discopt
import numpy as np
import pytest
from discopt.ml import add_predictor
from discopt.ml.formulations.base import NNFormulation
from discopt.ml.formulations.relu_icnn import ICNNExactnessError, input_convexity_violation
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition
from discopt.ml.scaling import OffsetScaling
from discopt.modeling.core import VarType

pytestmark = pytest.mark.unit


def _icnn(d, hidden, seed, n_out=1, box=2.0):
    rng = np.random.default_rng(seed)
    sizes = [d, *hidden, n_out]
    layers = []
    for i in range(len(sizes) - 1):
        W = rng.normal(0, 1.0 / np.sqrt(sizes[i]), (sizes[i], sizes[i + 1]))
        if i > 0:
            W = np.abs(W)
        act = Activation.RELU if i < len(sizes) - 2 else Activation.LINEAR
        layers.append(DenseLayer(W, rng.normal(0, 0.3, sizes[i + 1]), act))
    return NetworkDefinition(layers, input_bounds=(np.full(d, -box), np.full(d, box)))


def _embed(net, strategy, scaling=None):
    m = discopt.Model(strategy)
    f = NNFormulation(m, net, strategy=strategy, prefix="nn", scaling=scaling)
    f.formulate()
    return m, f


def _on_graph(f, net, r, scaling=None):
    """The reported point's network variables equal ``forward`` (the repair ran)."""
    x = np.asarray(r.x[f.inputs.name], float)
    s = x if scaling is None else (x - scaling.x_offset) / scaling.x_factor
    y = net.forward(s[None, :])[0]
    if scaling is not None:
        y = y * scaling.y_factor + scaling.y_offset
    np.testing.assert_allclose(r.x[f.outputs.name], y, rtol=0, atol=1e-9)
    h = s
    for k, layer in enumerate(net.layers):
        pre = h @ layer.weights + layer.biases
        np.testing.assert_allclose(r.x[f"nn_zhat_{k}"], pre, atol=1e-9)
        h = np.maximum(pre, 0.0) if layer.activation == Activation.RELU else pre
        if layer.activation == Activation.RELU:
            np.testing.assert_allclose(r.x[f"nn_z_{k}"], h, atol=1e-9)


def _consistent(exact, r):
    """Each certified objective lies in the other's [bound, objective] window:
    relu_bigm stops at the default 1e-4 gap, the LP closes exactly (measured on
    an E2 net, 2026-11-23: 7.5e-5 apart, both certified)."""
    tol = 1e-6 * max(1.0, abs(exact.objective))
    assert exact.bound - tol <= r.objective <= exact.objective + tol
    assert r.bound - tol <= exact.objective


NETS = [(2, (16, 16), 0), (3, (12, 8, 6), 1), (5, (20, 10), 2)]


@pytest.mark.parametrize("d, hidden, seed", NETS)
def test_minimum_matches_the_exact_milp_with_no_binaries(d, hidden, seed):
    net = _icnn(d, hidden, seed)
    res = {}
    for strategy in ("relu_bigm", "relu_icnn"):
        m, f = _embed(net, strategy)
        m.minimize(f.outputs[0])
        res[strategy] = (m, f, m.solve(time_limit=60))
    m, f, r = res["relu_icnn"]
    assert not any(v.var_type == VarType.BINARY for v in m._variables)
    assert any(v.var_type == VarType.BINARY for v in res["relu_bigm"][0]._variables)
    exact = res["relu_bigm"][2]
    assert exact.status == r.status == "optimal"
    _consistent(exact, r)
    _on_graph(f, net, r)


def test_bounded_above_with_another_objective_is_exact_after_repair():
    """``y <= c`` with a linear objective in x: the LP may leave z above
    relu(zhat) where the row is slack, and the repair puts it back."""
    net = _icnn(2, (16, 16), 3)
    checked = 0
    c = None
    for strategy in ("relu_bigm", "relu_icnn"):
        m, f = _embed(net, strategy)
        if c is None:
            m0, f0 = _embed(net, "relu_bigm")
            m0.minimize(f0.outputs[0])
            c = m0.solve(time_limit=60).objective + 0.5
        m.subject_to(f.outputs[0] <= c)
        m.minimize(f.inputs[0] + 0.3 * f.inputs[1])
        r = m.solve(time_limit=60)
        assert r.status == "optimal"
        if strategy == "relu_bigm":
            exact = r
        else:
            _consistent(exact, r)
            _on_graph(f, net, r)
            assert float(r.x[f.outputs.name][0]) <= c + 1e-7
        checked += 1
    assert checked == 2


def test_maximising_the_output_is_refused():
    net = _icnn(2, (16, 16), 0)
    m, f = _embed(net, "relu_icnn")
    m.maximize(f.outputs[0])
    with pytest.raises(ICNNExactnessError, match="relu_bigm"):
        m.solve(time_limit=60)


def test_bounding_the_output_below_is_answered_correctly_or_refused():
    """``y >= c`` rewards a larger output: the relaxation satisfies it anywhere by
    lifting z. The check judges the answer, not the use: a point whose repair
    still satisfies the model is the network's optimum too (it is optimal for a
    relaxation and feasible for the network), so it stands; any other is refused.
    Never a wrong answer, and the refusal must actually fire somewhere."""
    net = _icnn(2, (16, 16), 0)
    answered = refused = 0
    for c in (3.0, 5.0, 6.0, 7.0):
        for a in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.4, -1.0)):
            m0, f0 = _embed(net, "relu_bigm")
            m0.minimize(a[0] * f0.inputs[0] + a[1] * f0.inputs[1])
            m0.subject_to(f0.outputs[0] >= c)
            exact = m0.solve(time_limit=60)
            m, f = _embed(net, "relu_icnn")
            m.minimize(a[0] * f.inputs[0] + a[1] * f.inputs[1])
            m.subject_to(f.outputs[0] >= c)
            try:
                r = m.solve(time_limit=60)
            except ICNNExactnessError:
                refused += 1
                continue
            assert r.status == exact.status
            if r.status == "optimal":
                _consistent(exact, r)
                _on_graph(f, net, r)
            answered += 1
    assert refused > 0 and answered > 0, (refused, answered)


def test_an_infeasible_relaxation_proves_the_network_model_infeasible():
    net = _icnn(2, (16, 16), 0)
    checked = 0
    for strategy in ("relu_bigm", "relu_icnn"):
        m, f = _embed(net, strategy)
        m.subject_to(f.outputs[0] <= -100.0)
        m.minimize(f.inputs[0])
        assert m.solve(time_limit=60).status == "infeasible"
        checked += 1
    assert checked == 2


def test_scaled_network_matches_the_exact_milp():
    net = _icnn(2, (12, 12), 4)
    sc = OffsetScaling(
        x_offset=np.array([1.0, -1.0]),
        x_factor=np.array([2.0, 0.5]),
        y_offset=np.array([0.5]),
        y_factor=np.array([3.0]),
    )
    net.input_bounds = (np.array([-3.0, -2.0]), np.array([5.0, 0.0]))
    out = {}
    for strategy in ("relu_bigm", "relu_icnn"):
        m, f = _embed(net, strategy, scaling=sc)
        m.minimize(f.outputs[0])
        out[strategy] = (f, m.solve(time_limit=60))
    f, r = out["relu_icnn"]
    assert r.status == out["relu_bigm"][1].status == "optimal"
    _consistent(out["relu_bigm"][1], r)
    _on_graph(f, net, r, scaling=sc)


def _non_icnn_cases():
    neg = _icnn(2, (8, 8), 0)
    neg.layers[1].weights[0, 0] = -0.1
    neg_out = _icnn(2, (8,), 0)
    neg_out.layers[1].weights[3, 0] = -0.2
    smooth = _icnn(2, (8,), 0)
    smooth.layers[0].activation = Activation.SIGMOID
    flip = OffsetScaling(
        x_offset=np.zeros(2), x_factor=np.ones(2), y_offset=np.zeros(1), y_factor=-np.ones(1)
    )
    return [
        (neg, None, "negative weight"),
        (neg_out, None, "negative weight"),
        (smooth, None, "sigmoid"),
        (_icnn(2, (8,), 0), flip, "factor <= 0"),
    ]


def test_a_network_that_is_not_input_convex_is_refused_at_build():
    checked = 0
    for net, sc, why in _non_icnn_cases():
        assert why in input_convexity_violation(net, sc)
        with pytest.raises(ValueError, match="input-convex"):
            _embed(net, "relu_icnn", scaling=sc)
        checked += 1
    assert checked == 4
    assert input_convexity_violation(_icnn(2, (8, 8), 0)) is None


def test_the_check_survives_clone_and_pickle():
    net = _icnn(2, (8, 8), 0)
    m, f = _embed(net, "relu_icnn")
    m.maximize(f.outputs[0])
    checked = 0
    for copy in (m.clone(), pickle.loads(pickle.dumps(m))):
        assert len(copy._solve_result_checks) == 1
        with pytest.raises(ICNNExactnessError):
            copy.solve(time_limit=60)
        checked += 1
    assert checked == 2


def test_serializing_the_model_is_refused():
    from discopt.serialize import SerializationError, dumps

    m, f = _embed(_icnn(2, (8,), 0), "relu_icnn")
    m.minimize(f.outputs[0])
    with pytest.raises(SerializationError, match="result checks"):
        dumps(m)


def test_add_predictor_accepts_the_method():
    net = _icnn(2, (8, 8), 5)
    m = discopt.Model("ap")
    x = m.continuous("x", shape=(2,), lb=-2.0, ub=2.0)
    y, form = add_predictor(m, x, net, method="relu_icnn")
    m.minimize(y[0])
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    xs = np.asarray(r.x["x"], float)
    assert r.objective == pytest.approx(float(net.forward(xs[None, :])[0, 0]), abs=1e-6)
    assert not any(v.var_type == VarType.BINARY for v in m._variables)


def test_solve_model_directly_runs_the_check():
    """#2323 review: the check was only on ``Model.solve``; a direct
    ``solve_model`` (the route the differentiable-solve paths and the primal
    heuristics take) returned the maximised relaxation as a certified
    ``optimal``, above the network's true maximum."""
    from discopt.solver import solve_model

    net = _icnn(2, (6,), 0)
    m, f = _embed(net, "relu_icnn")
    m.maximize(f.outputs[0])
    with pytest.raises(ICNNExactnessError, match="relu_bigm"):
        solve_model(m, time_limit=60)


def test_solve_model_directly_answers_an_exact_use():
    """The boundary judges, it does not refuse: a minimised output solved
    through ``solve_model`` is repaired onto the graph and stands, with
    ``objective`` evaluated at the reported point."""
    from discopt.solver import solve_model

    net = _icnn(2, (16, 16), 0)
    m, f = _embed(net, "relu_icnn")
    m.minimize(f.outputs[0])
    r = solve_model(m, time_limit=60)
    assert r.status == "optimal"
    _on_graph(f, net, r)
    xs = np.asarray(r.x[f.inputs.name], float)
    assert r.objective == pytest.approx(float(net.forward(xs[None, :])[0, 0]), rel=0, abs=1e-9)


@pytest.mark.parametrize("fmt", ["to_nl", "to_lp", "to_mps", "to_gams"])
def test_exporting_the_model_is_refused(fmt):
    """#2323 review: a file holds the epigraph rows without the check, so
    another solver reading it would answer the bare relaxation."""
    m, f = _embed(_icnn(2, (8,), 0), "relu_icnn")
    m.maximize(f.outputs[0])
    with pytest.raises(ValueError, match="result checks"):
        getattr(m, fmt)()
