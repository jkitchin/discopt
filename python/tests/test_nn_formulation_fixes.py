"""Regression tests for the nn-module correctness/design fixes.

Covers:
- T-N0.2 (F1): scaling / bound-propagation domain mismatch — bounds and big-Ms
  must be computed on the *scaled* input box the layers actually consume.
- T-N0.3 (F2): tree big-M validity for out-of-box thresholds.
- T-N0.4: free (finite) output-variable bounds.
- T-N2.1: reduced_space honesty — bounded intermediates.
- T-N2.2: NNPresolvePass abstains loudly-but-safely (narrowed except).

The F1/F2 cases (``test_*scaling*`` and ``test_tree_out_of_box*``) fail on the
pre-fix code: the model is either infeasible or certifies/reads back a wrong
optimum because the wrong box drives the variable bounds and big-M constants.
They pass after the fix. See ``nn-module-plan.md`` §3.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import discopt.modeling as dm
from discopt.nn.bounds import propagate_bounds
from discopt.nn.formulations.base import NNFormulation, TreeFormulation
from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
from discopt.nn.presolve import NNPresolvePass, tighten_network
from discopt.nn.scaling import OffsetScaling
from discopt.nn.tree import DecisionTree, TreeEnsembleDefinition
from test_nn_equivalence import assert_embedding_matches, assert_optimum_matches

_SENTINEL = 9.999e19  # Model.continuous unbounded default magnitude


# ---------------------------------------------------------------------------
# Tiny predictors (1 input, finite user-domain box around x=100)
# ---------------------------------------------------------------------------


def _sigmoid_net(lb=99.0, ub=101.0):
    """1-input sigmoid -> linear net; input box in the *user* (unscaled) domain."""
    W1 = np.array([[0.8]])
    b1 = np.array([0.1])
    W2 = np.array([[1.2]])
    b2 = np.array([-0.3])
    return NetworkDefinition(
        [
            DenseLayer(W1, b1, Activation.SIGMOID),
            DenseLayer(W2, b2, Activation.LINEAR),
        ],
        input_bounds=(np.array([lb]), np.array([ub])),
    )


def _relu_net(lb=99.0, ub=101.0):
    """1-input ReLU -> linear net; input box in the user domain."""
    rng = np.random.RandomState(11)
    W1 = rng.randn(1, 3)
    b1 = rng.randn(3)
    W2 = rng.randn(3, 1)
    b2 = rng.randn(1)
    return NetworkDefinition(
        [
            DenseLayer(W1, b1, Activation.RELU),
            DenseLayer(W2, b2, Activation.LINEAR),
        ],
        input_bounds=(np.array([lb]), np.array([ub])),
    )


def _scaling(x_factor):
    """OffsetScaling mapping the user box around 100 into scaled [-2, 2]."""
    return OffsetScaling(
        x_offset=np.array([100.0]),
        x_factor=np.array([x_factor]),
        y_offset=np.array([5.0]),
        y_factor=np.array([2.0]),
    )


# ---------------------------------------------------------------------------
# T-N0.2 (F1): scaled-domain bound propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x_factor", [0.5, -0.5])
def test_full_space_scaling_fixed_input(x_factor):
    assert_embedding_matches(
        _sigmoid_net(), strategy="full_space", scaling=_scaling(x_factor), tol=1e-4
    )


@pytest.mark.parametrize("x_factor", [0.5, -0.5])
def test_full_space_scaling_optimum(x_factor):
    assert_optimum_matches(
        _sigmoid_net(), strategy="full_space", scaling=_scaling(x_factor), sense="min", tol=1e-4
    )


@pytest.mark.parametrize("x_factor", [0.5, -0.5])
def test_relu_bigm_scaling_fixed_input(x_factor):
    assert_embedding_matches(
        _relu_net(), strategy="relu_bigm", scaling=_scaling(x_factor), tol=1e-6
    )


@pytest.mark.parametrize("x_factor", [0.5, -0.5])
def test_relu_bigm_scaling_optimum(x_factor):
    assert_optimum_matches(
        _relu_net(), strategy="relu_bigm", scaling=_scaling(x_factor), sense="min", tol=1e-4
    )


# ---------------------------------------------------------------------------
# T-N0.3 (F2): tree big-M validity for out-of-box thresholds
# ---------------------------------------------------------------------------


def _out_of_box_ensemble():
    """Ensemble whose thresholds sit outside the declared feature box.

    Feature box is [0, 1] for feature 0. Tree 1 splits at 1.5 (> ub - eps):
    every point in the box goes left. Tree 2 splits at -0.5 (< lb): every
    point goes right. With the old per-feature big-M ``ub - lb`` the
    non-selected leaves' constraints cut feasible points.
    """
    tree1 = DecisionTree(
        n_features=1,
        feature=np.array([0, -1, -1]),
        threshold=np.array([1.5, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, 2.0, 7.0]),
    )
    tree2 = DecisionTree(
        n_features=1,
        feature=np.array([0, -1, -1]),
        threshold=np.array([-0.5, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, -3.0, 4.0]),
    )
    return TreeEnsembleDefinition(
        trees=[tree1, tree2],
        n_features=1,
        base_score=0.5,
        input_bounds=(np.array([0.0]), np.array([1.0])),
    )


def test_tree_out_of_box_fixed_input():
    assert_embedding_matches(_out_of_box_ensemble(), tol=1e-5)


def test_tree_out_of_box_optimum():
    assert_optimum_matches(_out_of_box_ensemble(), sense="min", tol=1e-4)
    assert_optimum_matches(_out_of_box_ensemble(), sense="max", tol=1e-4)


def test_tree_in_box_unchanged():
    """In-box thresholds: the tighter per-constraint big-M is a no-op for validity."""
    tree = DecisionTree(
        n_features=2,
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, 1.0, 3.0]),
    )
    ens = TreeEnsembleDefinition(
        trees=[tree],
        n_features=2,
        base_score=0.0,
        input_bounds=(np.array([0.0, -1.0]), np.array([1.0, 1.0])),
    )
    assert_embedding_matches(ens, tol=1e-5)
    assert_optimum_matches(ens, sense="min", tol=1e-4)


# ---------------------------------------------------------------------------
# T-N0.4: free output bounds contain every reachable prediction
# ---------------------------------------------------------------------------


def _build_nn(net, strategy, scaling):
    m = dm.Model("t")
    form = NNFormulation(m, net, strategy=strategy, prefix="e", scaling=scaling)
    form.formulate()
    return form


def _assert_output_bounds_contain(form, obj, scaling, n_points=8, seed=1):
    lb_out = np.asarray(form.outputs.lb, dtype=np.float64).ravel()
    ub_out = np.asarray(form.outputs.ub, dtype=np.float64).ravel()
    # Bounds must be finite (strictly inside the unbounded sentinel).
    assert np.all(np.abs(lb_out) < _SENTINEL)
    assert np.all(np.abs(ub_out) < _SENTINEL)

    ib = obj.input_bounds
    lb, ub = np.asarray(ib[0], dtype=np.float64), np.asarray(ib[1], dtype=np.float64)
    n = len(lb)
    rng = np.random.RandomState(seed)
    for _ in range(n_points):
        x = lb + rng.random_sample(n) * (ub - lb)
        if scaling is None:
            pred = np.asarray(obj.predict(x) if hasattr(obj, "predict") else obj.forward(x))
        else:
            xs = (x - scaling.x_offset) / scaling.x_factor
            pred = scaling.y_factor * np.asarray(obj.forward(xs)) + scaling.y_offset
        pred = np.atleast_1d(pred).astype(np.float64)
        assert np.all(pred >= lb_out - 1e-9), f"pred {pred} below out lb {lb_out}"
        assert np.all(pred <= ub_out + 1e-9), f"pred {pred} above out ub {ub_out}"


def test_output_bounds_full_space_scaled():
    net = _sigmoid_net()
    sc = _scaling(0.5)
    form = _build_nn(net, "full_space", sc)
    _assert_output_bounds_contain(form, net, sc)


def test_output_bounds_relu_bigm_scaled():
    net = _relu_net()
    sc = _scaling(0.5)
    form = _build_nn(net, "relu_bigm", sc)
    _assert_output_bounds_contain(form, net, sc)


def test_output_bounds_reduced_space_scaled():
    net = _sigmoid_net()
    sc = _scaling(0.5)
    form = _build_nn(net, "reduced_space", sc)
    _assert_output_bounds_contain(form, net, sc)


def test_output_bounds_tree():
    ens = _out_of_box_ensemble()
    m = dm.Model("t")
    form = TreeFormulation(m, ens, prefix="e")
    form.formulate()
    _assert_output_bounds_contain(form, ens, None)


# ---------------------------------------------------------------------------
# T-N2.1: reduced_space intermediate variables are bounded
# ---------------------------------------------------------------------------


def _reduced_net():
    """2-layer sigmoid net so there is a genuine intermediate z variable."""
    W1 = np.array([[0.7]])
    b1 = np.array([0.05])
    W2 = np.array([[0.9]])
    b2 = np.array([-0.2])
    return NetworkDefinition(
        [
            DenseLayer(W1, b1, Activation.SIGMOID),
            DenseLayer(W2, b2, Activation.SIGMOID),
        ],
        input_bounds=(np.array([99.0]), np.array([101.0])),
    )


@pytest.mark.parametrize("scaling", [None, _scaling(0.5)])
def test_reduced_space_intermediates_bounded(scaling):
    net = _reduced_net()
    m = dm.Model("t")
    form = NNFormulation(m, net, strategy="reduced_space", prefix="e", scaling=scaling)
    form.formulate()
    # Every created variable whose name marks an intermediate/output must be
    # finitely bounded when the net has input_bounds.
    for var in m._variables:
        if var.name.startswith("e_z_") or var.name == "e_output":
            assert np.all(np.abs(var.lb) < _SENTINEL), f"{var.name} lb not finite"
            assert np.all(np.abs(var.ub) < _SENTINEL), f"{var.name} ub not finite"


def test_reduced_space_scaled_equivalence():
    # Also re-checks F1 through the reduced_space path.
    assert_embedding_matches(
        _sigmoid_net(), strategy="reduced_space", scaling=_scaling(0.5), tol=1e-4
    )


# ---------------------------------------------------------------------------
# T-N2.2: presolve refactor + narrowed abstention
# ---------------------------------------------------------------------------


def test_tighten_network_override_matches_and_no_mutation():
    net = _relu_net()
    box = (np.array([-2.0]), np.array([2.0]))
    expected = propagate_bounds(net, input_bounds=box)
    result = tighten_network(net, input_lb=box[0], input_ub=box[1])
    assert len(result.layer_bounds) == len(expected)
    for got, exp in zip(result.layer_bounds, expected):
        np.testing.assert_allclose(got.pre_lb, exp.pre_lb)
        np.testing.assert_allclose(got.pre_ub, exp.pre_ub)
    # The network's declared box must be untouched (no mutate-and-restore).
    np.testing.assert_allclose(net.input_bounds[0], [99.0])
    np.testing.assert_allclose(net.input_bounds[1], [101.0])


def test_nnpresolve_abstains_without_bounds():
    net = NetworkDefinition(
        [DenseLayer(np.array([[1.0]]), np.array([0.0]), Activation.RELU)],
        input_bounds=None,
    )
    delta = NNPresolvePass(net).run(object())
    # Abstained: no nn implications surfaced, orchestrator gets an inert delta.
    assert "nn_implications" not in delta
