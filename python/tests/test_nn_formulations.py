"""Tests for discopt.nn formulations, readers, and predictor API.

Covers NNFormulation (full_space, relu_bigm, reduced_space), TreeFormulation,
reader modules (sklearn, torch, onnx), and the add_predictor convenience API.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt.nn.bounds import LayerBounds, propagate_bounds
from discopt.nn.formulations.base import NNFormulation, TreeFormulation
from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
from discopt.nn.scaling import OffsetScaling
from discopt.nn.tree import DecisionTree, TreeEnsembleDefinition

# ---------------------------------------------------------------------------
# Helpers: small synthetic networks and trees
# ---------------------------------------------------------------------------


def _make_linear_net(n_in=2, n_out=1, with_bounds=True):
    """Single-layer linear network."""
    rng = np.random.RandomState(0)
    W = rng.randn(n_in, n_out)
    b = rng.randn(n_out)
    bounds = None
    if with_bounds:
        bounds = (-np.ones(n_in), np.ones(n_in))
    return NetworkDefinition(
        [DenseLayer(W, b, Activation.LINEAR)],
        input_bounds=bounds,
    )


def _make_relu_net(with_bounds=True):
    """Two-layer ReLU -> LINEAR network (2 inputs, 3 hidden, 1 output)."""
    rng = np.random.RandomState(42)
    W1 = rng.randn(2, 3)
    b1 = rng.randn(3)
    W2 = rng.randn(3, 1)
    b2 = rng.randn(1)
    bounds = None
    if with_bounds:
        bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    return NetworkDefinition(
        [
            DenseLayer(W1, b1, Activation.RELU),
            DenseLayer(W2, b2, Activation.LINEAR),
        ],
        input_bounds=bounds,
    )


def _make_sigmoid_net():
    """Two-layer sigmoid -> LINEAR network (2 inputs, 4 hidden, 1 output)."""
    rng = np.random.RandomState(7)
    W1 = rng.randn(2, 4)
    b1 = rng.randn(4)
    W2 = rng.randn(4, 1)
    b2 = rng.randn(1)
    return NetworkDefinition(
        [
            DenseLayer(W1, b1, Activation.SIGMOID),
            DenseLayer(W2, b2, Activation.LINEAR),
        ],
        input_bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
    )


def _make_tanh_net():
    """Single-layer tanh network (2 inputs, 2 outputs)."""
    W = np.array([[1.0, 0.5], [-0.5, 1.0]])
    b = np.array([0.0, 0.0])
    return NetworkDefinition(
        [DenseLayer(W, b, Activation.TANH)],
        input_bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
    )


def _make_simple_tree():
    """A single decision tree: split on feature 0 at threshold 0.5."""
    # Node 0: split on feature 0, threshold 0.5
    # Node 1 (left leaf): value 1.0
    # Node 2 (right leaf): value 3.0
    return DecisionTree(
        n_features=2,
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, 1.0, 3.0]),
    )


def _make_tree_ensemble():
    """A two-tree ensemble with 2 features."""
    tree1 = _make_simple_tree()
    # Second tree: split on feature 1 at threshold 0.0
    tree2 = DecisionTree(
        n_features=2,
        feature=np.array([1, -1, -1]),
        threshold=np.array([0.0, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, -1.0, 2.0]),
    )
    return TreeEnsembleDefinition(
        trees=[tree1, tree2],
        n_features=2,
        base_score=0.5,
        input_bounds=(np.array([0.0, -1.0]), np.array([1.0, 1.0])),
    )


# ---------------------------------------------------------------------------
# NNFormulation: full_space strategy
# ---------------------------------------------------------------------------


class TestFullSpaceFormulation:
    def test_builds_without_error(self):
        """Full-space formulation adds variables and constraints to Model."""
        import discopt.modeling as dm

        net = _make_linear_net()
        m = dm.Model("fs_linear")
        nn = NNFormulation(m, net, strategy="full_space", prefix="fs")
        nn.formulate()
        assert nn.inputs is not None
        assert nn.outputs is not None

    def test_sigmoid_builds(self):
        import discopt.modeling as dm

        net = _make_sigmoid_net()
        m = dm.Model("fs_sigmoid")
        nn = NNFormulation(m, net, strategy="full_space", prefix="sig")
        nn.formulate()
        assert nn.inputs is not None
        assert nn.outputs is not None

    def test_rejects_relu(self):
        import discopt.modeling as dm

        net = _make_relu_net()
        m = dm.Model("fs_relu")
        with pytest.raises(ValueError, match="does not support"):
            NNFormulation(m, net, strategy="full_space").formulate()

    def test_no_input_bounds(self):
        """Full-space should still work without input bounds."""
        import discopt.modeling as dm

        net = _make_linear_net(with_bounds=False)
        m = dm.Model("fs_no_bounds")
        nn = NNFormulation(m, net, strategy="full_space")
        nn.formulate()
        assert nn.inputs is not None

    def test_double_formulate_raises(self):
        import discopt.modeling as dm

        net = _make_linear_net()
        m = dm.Model("fs_double")
        nn = NNFormulation(m, net, strategy="full_space")
        nn.formulate()
        with pytest.raises(RuntimeError, match="already been called"):
            nn.formulate()

    def test_access_before_formulate_raises(self):
        import discopt.modeling as dm

        net = _make_linear_net()
        m = dm.Model("fs_noaccess")
        nn = NNFormulation(m, net, strategy="full_space")
        with pytest.raises(RuntimeError, match="Call formulate"):
            _ = nn.inputs
        with pytest.raises(RuntimeError, match="Call formulate"):
            _ = nn.outputs

    def test_with_scaling(self):
        import discopt.modeling as dm

        net = _make_linear_net()
        scaling = OffsetScaling(
            x_offset=np.array([0.0, 0.0]),
            x_factor=np.array([1.0, 1.0]),
            y_offset=np.array([10.0]),
            y_factor=np.array([2.0]),
        )
        m = dm.Model("fs_scaled")
        nn = NNFormulation(m, net, strategy="full_space", scaling=scaling)
        nn.formulate()
        assert nn.inputs is not None
        assert nn.outputs is not None


# ---------------------------------------------------------------------------
# NNFormulation: relu_bigm strategy
# ---------------------------------------------------------------------------


class TestReluBigMFormulation:
    def test_builds_without_error(self):
        import discopt.modeling as dm

        net = _make_relu_net()
        m = dm.Model("bigm_relu")
        nn = NNFormulation(m, net, strategy="relu_bigm", prefix="bm")
        nn.formulate()
        assert nn.inputs is not None
        assert nn.outputs is not None

    def test_requires_input_bounds(self):
        import discopt.modeling as dm

        net = _make_relu_net(with_bounds=False)
        m = dm.Model("bigm_nobounds")
        with pytest.raises(ValueError, match="requires input_bounds"):
            NNFormulation(m, net, strategy="relu_bigm").formulate()

    def test_with_scaling(self):
        import discopt.modeling as dm

        net = _make_relu_net()
        scaling = OffsetScaling(
            x_offset=np.array([0.0, 0.0]),
            x_factor=np.array([1.0, 1.0]),
            y_offset=np.array([0.0]),
            y_factor=np.array([1.0]),
        )
        m = dm.Model("bigm_scaled")
        nn = NNFormulation(m, net, strategy="relu_bigm", scaling=scaling)
        nn.formulate()
        assert nn.inputs is not None


# ---------------------------------------------------------------------------
# NNFormulation: reduced_space strategy
# ---------------------------------------------------------------------------


class TestReducedSpaceFormulation:
    def test_builds_without_error(self):
        import discopt.modeling as dm

        net = _make_tanh_net()
        m = dm.Model("rs_tanh")
        nn = NNFormulation(m, net, strategy="reduced_space", prefix="rs")
        nn.formulate()
        assert nn.inputs is not None
        assert nn.outputs is not None

    def test_relu_via_reduced(self):
        """Reduced-space supports ReLU via dm.maximum."""
        import discopt.modeling as dm

        net = _make_relu_net()
        m = dm.Model("rs_relu")
        nn = NNFormulation(m, net, strategy="reduced_space")
        nn.formulate()
        assert nn.inputs is not None

    def test_no_bounds(self):
        import discopt.modeling as dm

        net = _make_linear_net(with_bounds=False)
        m = dm.Model("rs_nobounds")
        nn = NNFormulation(m, net, strategy="reduced_space")
        nn.formulate()
        assert nn.inputs is not None

    def test_with_scaling(self):
        import discopt.modeling as dm

        net = _make_tanh_net()
        scaling = OffsetScaling(
            x_offset=np.array([0.0, 0.0]),
            x_factor=np.array([1.0, 1.0]),
            y_offset=np.array([0.0, 0.0]),
            y_factor=np.array([1.0, 1.0]),
        )
        m = dm.Model("rs_scaled")
        nn = NNFormulation(m, net, strategy="reduced_space", scaling=scaling)
        nn.formulate()
        assert nn.inputs is not None


# ---------------------------------------------------------------------------
# NNFormulation: invalid strategy
# ---------------------------------------------------------------------------


class TestNNFormulationValidation:
    def test_invalid_strategy(self):
        import discopt.modeling as dm

        net = _make_linear_net()
        m = dm.Model("bad_strategy")
        with pytest.raises(ValueError, match="Unknown strategy"):
            NNFormulation(m, net, strategy="nonexistent")


# ---------------------------------------------------------------------------
# TreeFormulation
# ---------------------------------------------------------------------------


class TestTreeFormulation:
    def test_builds_without_error(self):
        import discopt.modeling as dm

        ensemble = _make_tree_ensemble()
        m = dm.Model("tree_test")
        tf = TreeFormulation(m, ensemble, prefix="tf")
        tf.formulate()
        assert tf.inputs is not None
        assert tf.outputs is not None

    def test_requires_input_bounds(self):
        import discopt.modeling as dm

        tree = _make_simple_tree()
        ensemble = TreeEnsembleDefinition(trees=[tree], n_features=2)
        m = dm.Model("tree_nobounds")
        tf = TreeFormulation(m, ensemble)
        with pytest.raises(ValueError, match="input_bounds is required"):
            tf.formulate()

    def test_double_formulate_raises(self):
        import discopt.modeling as dm

        ensemble = _make_tree_ensemble()
        m = dm.Model("tree_double")
        tf = TreeFormulation(m, ensemble)
        tf.formulate()
        with pytest.raises(RuntimeError, match="already been called"):
            tf.formulate()

    def test_access_before_formulate_raises(self):
        import discopt.modeling as dm

        ensemble = _make_tree_ensemble()
        m = dm.Model("tree_noaccess")
        tf = TreeFormulation(m, ensemble)
        with pytest.raises(RuntimeError, match="Call formulate"):
            _ = tf.inputs
        with pytest.raises(RuntimeError, match="Call formulate"):
            _ = tf.outputs

    def test_tree_predict(self):
        """DecisionTree.predict gives correct leaf values."""
        tree = _make_simple_tree()
        assert tree.predict(np.array([0.3, 0.0])) == pytest.approx(1.0)
        assert tree.predict(np.array([0.7, 0.0])) == pytest.approx(3.0)

    def test_tree_ensemble_predict(self):
        """TreeEnsembleDefinition.predict sums trees + base_score."""
        ens = _make_tree_ensemble()
        # x=[0.3, 0.5]: tree1 -> left (1.0), tree2 -> right (2.0), base=0.5
        assert ens.predict(np.array([0.3, 0.5])) == pytest.approx(3.5)
        # x=[0.7, -0.5]: tree1 -> right (3.0), tree2 -> left (-1.0), base=0.5
        assert ens.predict(np.array([0.7, -0.5])) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Bound propagation
# ---------------------------------------------------------------------------


class TestBoundPropagationMultiLayer:
    def test_relu_net_bounds(self):
        """Bounds propagate correctly through a multi-layer ReLU network."""
        net = _make_relu_net()
        bounds = propagate_bounds(net)
        assert len(bounds) == 2
        for lb_obj in bounds:
            assert isinstance(lb_obj, LayerBounds)
            assert lb_obj.post_lb.shape == lb_obj.post_ub.shape
            assert np.all(lb_obj.post_ub >= lb_obj.post_lb)

    def test_sigmoid_net_bounds(self):
        """Sigmoid bounds should be in [0, 1]."""
        net = _make_sigmoid_net()
        bounds = propagate_bounds(net)
        # First layer is sigmoid
        assert np.all(bounds[0].post_lb >= 0.0)
        assert np.all(bounds[0].post_ub <= 1.0)


# ---------------------------------------------------------------------------
# predictor.py: add_predictor API
# ---------------------------------------------------------------------------


class TestAddPredictor:
    def test_with_network_definition(self):
        import discopt.modeling as dm
        from discopt.nn.predictor import add_predictor

        net = _make_relu_net()
        m = dm.Model("pred_net")
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        outputs, form = add_predictor(m, x, net, prefix="p")
        assert outputs is not None
        assert isinstance(form, NNFormulation)

    def test_with_network_definition_custom_method(self):
        import discopt.modeling as dm
        from discopt.nn.predictor import add_predictor

        net = _make_tanh_net()
        m = dm.Model("pred_custom")
        x = m.continuous("x", shape=(2,), lb=-2.0, ub=2.0)
        outputs, form = add_predictor(m, x, net, method="reduced_space", prefix="rc")
        assert outputs is not None

    def test_with_tree_ensemble(self):
        import discopt.modeling as dm
        from discopt.nn.predictor import add_predictor

        ens = _make_tree_ensemble()
        m = dm.Model("pred_tree")
        x = m.continuous("x", shape=(2,), lb=np.array([0.0, -1.0]), ub=np.array([1.0, 1.0]))
        outputs, form = add_predictor(m, x, ens, prefix="te")
        assert outputs is not None
        assert isinstance(form, TreeFormulation)

    def test_unsupported_predictor_raises(self):
        import discopt.modeling as dm
        from discopt.nn.predictor import add_predictor

        m = dm.Model("pred_bad")
        x = m.continuous("x", shape=(2,))
        with pytest.raises(TypeError, match="Cannot auto-detect"):
            add_predictor(m, x, {"not": "a model"})

    def test_auto_selects_relu_bigm(self):
        """Auto method should pick relu_bigm for networks with ReLU."""
        import discopt.modeling as dm
        from discopt.nn.predictor import add_predictor

        net = _make_relu_net()
        m = dm.Model("pred_auto_relu")
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        outputs, form = add_predictor(m, x, net, prefix="ar")
        assert outputs is not None

    def test_auto_selects_full_space_for_smooth(self):
        """Auto method should pick full_space for smooth activations."""
        import discopt.modeling as dm
        from discopt.nn.predictor import add_predictor

        net = _make_sigmoid_net()
        m = dm.Model("pred_auto_smooth")
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        outputs, form = add_predictor(m, x, net, prefix="as")
        assert outputs is not None

    def test_input_bounds_override(self):
        """add_predictor should accept input_bounds override."""
        import discopt.modeling as dm
        from discopt.nn.predictor import add_predictor

        net = _make_linear_net(with_bounds=False)
        m = dm.Model("pred_bounds")
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        outputs, form = add_predictor(m, x, net, input_bounds=bounds, prefix="bo")
        assert outputs is not None


# ---------------------------------------------------------------------------
# Reader: sklearn
# ---------------------------------------------------------------------------


class TestSklearnReader:
    @pytest.fixture(autouse=True)
    def _check_sklearn(self):
        pytest.importorskip("sklearn")

    def test_load_mlp_regressor(self):
        from discopt.nn.readers.sklearn_reader import load_sklearn_mlp
        from sklearn.neural_network import MLPRegressor

        rng = np.random.RandomState(0)
        X = rng.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        mlp = MLPRegressor(hidden_layer_sizes=(4,), max_iter=10, random_state=0)
        mlp.fit(X, y)

        net = load_sklearn_mlp(mlp, input_bounds=(-np.ones(2), np.ones(2)))
        assert net.input_size == 2
        assert net.output_size == 1
        assert net.n_layers == 2  # hidden + output

    def test_mlp_formulate(self):
        """Converted sklearn MLP can be formulated."""
        import discopt.modeling as dm
        from discopt.nn.readers.sklearn_reader import load_sklearn_mlp
        from sklearn.neural_network import MLPRegressor

        rng = np.random.RandomState(0)
        X = rng.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        mlp = MLPRegressor(hidden_layer_sizes=(3,), max_iter=10, random_state=0)
        mlp.fit(X, y)

        net = load_sklearn_mlp(mlp, input_bounds=(-np.ones(2), np.ones(2)))
        m = dm.Model("sklearn_mlp")
        nn = NNFormulation(m, net, strategy="relu_bigm")
        nn.formulate()
        assert nn.inputs is not None

    def test_load_decision_tree(self):
        from discopt.nn.readers.sklearn_reader import load_sklearn_tree
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.RandomState(0)
        X = rng.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        dt = DecisionTreeRegressor(max_depth=3, random_state=0)
        dt.fit(X, y)

        ens = load_sklearn_tree(dt, input_bounds=(-np.ones(2), np.ones(2)))
        assert isinstance(ens, TreeEnsembleDefinition)
        assert ens.n_features == 2
        assert len(ens.trees) == 1

    def test_load_gradient_boosting(self):
        from discopt.nn.readers.sklearn_reader import load_sklearn_ensemble
        from sklearn.ensemble import GradientBoostingRegressor

        rng = np.random.RandomState(0)
        X = rng.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        gbr = GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=0)
        gbr.fit(X, y)

        ens = load_sklearn_ensemble(gbr, input_bounds=(-np.ones(2), np.ones(2)))
        assert isinstance(ens, TreeEnsembleDefinition)
        assert len(ens.trees) == 3
        assert ens.base_score != 0.0

    def test_load_random_forest(self):
        from discopt.nn.readers.sklearn_reader import load_sklearn_ensemble
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(0)
        X = rng.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        rf = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0)
        rf.fit(X, y)

        ens = load_sklearn_ensemble(rf, input_bounds=(-np.ones(2), np.ones(2)))
        assert isinstance(ens, TreeEnsembleDefinition)
        assert len(ens.trees) == 3

    def test_add_predictor_with_sklearn_mlp(self):
        """add_predictor auto-detects sklearn MLP."""
        import discopt.modeling as dm
        from discopt.nn.predictor import add_predictor
        from sklearn.neural_network import MLPRegressor

        rng = np.random.RandomState(0)
        X = rng.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        mlp = MLPRegressor(hidden_layer_sizes=(3,), max_iter=10, random_state=0)
        mlp.fit(X, y)

        m = dm.Model("sklearn_pred")
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        bounds = (-np.ones(2), np.ones(2))
        outputs, form = add_predictor(m, x, mlp, input_bounds=bounds, prefix="sk")
        assert outputs is not None

    def test_add_predictor_with_sklearn_tree(self):
        """add_predictor auto-detects sklearn DecisionTree."""
        import discopt.modeling as dm
        from discopt.nn.predictor import add_predictor
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.RandomState(0)
        X = rng.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        dt = DecisionTreeRegressor(max_depth=2, random_state=0)
        dt.fit(X, y)

        m = dm.Model("sklearn_tree_pred")
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        bounds = (-np.ones(2), np.ones(2))
        outputs, form = add_predictor(m, x, dt, input_bounds=bounds, prefix="skt")
        assert outputs is not None


# ---------------------------------------------------------------------------
# Reader: torch
# ---------------------------------------------------------------------------


class TestTorchReader:
    @pytest.fixture(autouse=True)
    def _check_torch(self):
        pytest.importorskip("torch")

    def test_load_sequential(self):
        import torch.nn as nn
        from discopt.nn.readers.torch_reader import load_torch_sequential

        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        net = load_torch_sequential(model, input_bounds=(-np.ones(2), np.ones(2)))
        assert net.input_size == 2
        assert net.output_size == 1
        assert net.n_layers == 2

    def test_torch_formulate(self):
        """Converted torch model can be formulated."""
        import discopt.modeling as dm
        import torch.nn as nn
        from discopt.nn.readers.torch_reader import load_torch_sequential

        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        net = load_torch_sequential(model, input_bounds=(-np.ones(2), np.ones(2)))
        m = dm.Model("torch_test")
        form = NNFormulation(m, net, strategy="relu_bigm")
        form.formulate()
        assert form.inputs is not None

    def test_sigmoid_activation(self):
        import torch.nn as nn
        from discopt.nn.readers.torch_reader import load_torch_sequential

        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
        )
        net = load_torch_sequential(model)
        assert net.layers[0].activation == Activation.SIGMOID
        assert net.layers[1].activation == Activation.LINEAR

    def test_trailing_linear(self):
        """Model ending in Linear (no activation) gets LINEAR activation."""
        import torch.nn as nn
        from discopt.nn.readers.torch_reader import load_torch_sequential

        model = nn.Sequential(nn.Linear(3, 2))
        net = load_torch_sequential(model)
        assert net.n_layers == 1
        assert net.layers[0].activation == Activation.LINEAR

    def test_unsupported_layer_raises(self):
        import torch.nn as nn
        from discopt.nn.readers.torch_reader import load_torch_sequential

        model = nn.Sequential(nn.Linear(2, 3), nn.GELU(), nn.Linear(3, 1))
        with pytest.raises(ValueError, match="Unsupported layer type"):
            load_torch_sequential(model)

    def test_add_predictor_with_torch(self):
        """add_predictor auto-detects torch Sequential."""
        import discopt.modeling as dm
        import torch.nn as nn
        from discopt.nn.predictor import add_predictor

        model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
        m = dm.Model("torch_pred")
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        bounds = (-np.ones(2), np.ones(2))
        outputs, form = add_predictor(m, x, model, input_bounds=bounds, prefix="tp")
        assert outputs is not None


# ---------------------------------------------------------------------------
# Reader: ONNX
# ---------------------------------------------------------------------------


class TestOnnxReader:
    @pytest.fixture(autouse=True)
    def _check_onnx(self):
        pytest.importorskip("onnx")

    def test_load_simple_onnx(self, tmp_path):
        """Create a minimal ONNX model and load it."""
        import onnx
        from discopt.nn.readers.onnx_reader import load_onnx
        from onnx import TensorProto, helper, numpy_helper

        # Build a single Gemm node: y = x @ W + b
        W = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([0.5, -0.5], dtype=np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])

        W_init = numpy_helper.from_array(W, name="W")
        b_init = numpy_helper.from_array(b, name="b")

        gemm_node = helper.make_node("Gemm", ["X", "W", "b"], ["Y"], transB=0)
        graph = helper.make_graph([gemm_node], "test_graph", [X], [Y], [W_init, b_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        path = str(tmp_path / "test.onnx")
        onnx.save(model, path)

        net = load_onnx(path, input_bounds=(-np.ones(2), np.ones(2)))
        assert net.input_size == 2
        assert net.output_size == 2
        assert net.n_layers == 1

    def test_onnx_with_relu(self, tmp_path):
        """ONNX model with Gemm + Relu activation."""
        import onnx
        from discopt.nn.readers.onnx_reader import load_onnx
        from onnx import TensorProto, helper, numpy_helper

        W = np.array([[1.0], [1.0]], dtype=np.float32)
        b = np.array([0.0], dtype=np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])

        W_init = numpy_helper.from_array(W, name="W")
        b_init = numpy_helper.from_array(b, name="b")

        gemm_node = helper.make_node("Gemm", ["X", "W", "b"], ["Z"], transB=0)
        relu_node = helper.make_node("Relu", ["Z"], ["Y"])

        graph = helper.make_graph([gemm_node, relu_node], "relu_graph", [X], [Y], [W_init, b_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        path = str(tmp_path / "relu.onnx")
        onnx.save(model, path)

        net = load_onnx(path)
        assert net.n_layers == 1
        assert net.layers[0].activation == Activation.RELU

    def test_add_predictor_with_onnx_path(self, tmp_path):
        """add_predictor accepts a file path to an ONNX model."""
        import discopt.modeling as dm
        import onnx
        from discopt.nn.predictor import add_predictor
        from onnx import TensorProto, helper, numpy_helper

        W = np.array([[1.0], [1.0]], dtype=np.float32)
        b = np.array([0.0], dtype=np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])

        gemm_node = helper.make_node("Gemm", ["X", "W", "b"], ["Y"], transB=0)
        graph = helper.make_graph(
            [gemm_node],
            "pred_graph",
            [X],
            [Y],
            [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        path = str(tmp_path / "pred.onnx")
        onnx.save(model, path)

        m = dm.Model("onnx_pred")
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        bounds = (-np.ones(2), np.ones(2))
        outputs, form = add_predictor(m, x, path, input_bounds=bounds, prefix="ox")
        assert outputs is not None


# ---------------------------------------------------------------------------
# NetworkDefinition.forward consistency
# ---------------------------------------------------------------------------


class TestForwardConsistency:
    def test_softplus(self):
        """Forward pass with softplus activation."""
        W = np.array([[1.0]])
        b = np.array([0.0])
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.SOFTPLUS)],
            input_bounds=(np.array([-2.0]), np.array([2.0])),
        )
        result = net.forward(np.array([0.0]))
        assert result[0] == pytest.approx(np.log(2.0), abs=1e-6)

    def test_multi_layer_forward(self):
        """Multi-layer network forward should match manual computation."""
        net = _make_relu_net()
        x = np.array([0.5, -0.3])
        result = net.forward(x)
        # Manual: layer1 = relu(x @ W1 + b1), layer2 = layer1 @ W2 + b2
        h = np.maximum(x @ net.layers[0].weights + net.layers[0].biases, 0)
        expected = h @ net.layers[1].weights + net.layers[1].biases
        np.testing.assert_allclose(result, expected)
