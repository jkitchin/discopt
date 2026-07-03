"""Regression tests for NN reader hardening (plan tasks T-N1.1 / T-N1.2 / T-N1.3).

These lock in the "refuse loudly, never approximate silently" contract for the
ONNX, sklearn, and torch readers: unsupported structure must raise a naming
error rather than being silently mis-read, and previously-crashing edges must
load faithfully. ONNX correctness is checked against onnxruntime as the oracle;
sklearn/torch against the trained model's own inference.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# T-N1.1 — ONNX reader hardening
# ---------------------------------------------------------------------------


class TestOnnxReaderHardening:
    @pytest.fixture(autouse=True)
    def _check_onnx(self):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    @staticmethod
    def _save(model, tmp_path, name):
        import onnx

        path = str(tmp_path / name)
        onnx.save(model, path)
        return path

    @staticmethod
    def _ort_forward(path, x):
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: x.reshape(1, -1).astype(np.float32)})
        return np.asarray(out[0], dtype=np.float64).reshape(-1)

    def test_gemm_alpha_beta_applied(self, tmp_path):
        """Gemm alpha/beta fold into weight/bias; forward matches onnxruntime."""
        from discopt.nn.readers.onnx_reader import load_onnx
        from onnx import TensorProto, helper, numpy_helper

        rng = np.random.RandomState(0)
        W = rng.randn(2, 3).astype(np.float32)
        b = rng.randn(3).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Gemm", ["X", "W", "b"], ["Y"], alpha=2.0, beta=0.5, transB=0)
        graph = helper.make_graph(
            [node],
            "gemm_ab",
            [X],
            [Y],
            [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        path = self._save(model, tmp_path, "gemm_ab.onnx")

        net = load_onnx(path, input_bounds=(-np.ones(2), np.ones(2)))
        for _ in range(10):
            x = rng.uniform(-1.0, 1.0, size=2)
            np.testing.assert_allclose(
                net.forward(x), self._ort_forward(path, x), atol=1e-5, rtol=1e-5
            )

    def test_transa_raises(self, tmp_path):
        """Gemm with transA=1 (data-input transpose) is refused."""
        from discopt.nn.readers.onnx_reader import load_onnx
        from onnx import TensorProto, helper, numpy_helper

        W = np.eye(2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
        node = helper.make_node("Gemm", ["X", "W", "b"], ["Y"], transA=1, transB=0)
        graph = helper.make_graph(
            [node],
            "transa",
            [X],
            [Y],
            [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        path = self._save(model, tmp_path, "transa.onnx")

        with pytest.raises(ValueError, match="transA"):
            load_onnx(path)

    def test_two_branch_add_raises(self, tmp_path):
        """A graph whose two branches join in an Add is refused as non-sequential."""
        from discopt.nn.readers.onnx_reader import load_onnx
        from onnx import TensorProto, helper, numpy_helper

        W1 = np.eye(2, dtype=np.float32)
        W2 = (2.0 * np.eye(2)).astype(np.float32)
        b = np.zeros(2, dtype=np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
        g1 = helper.make_node("Gemm", ["X", "W1", "b1"], ["A"], transB=0)
        g2 = helper.make_node("Gemm", ["X", "W2", "b2"], ["B"], transB=0)
        add = helper.make_node("Add", ["A", "B"], ["Y"])
        graph = helper.make_graph(
            [g1, g2, add],
            "two_branch",
            [X],
            [Y],
            [
                numpy_helper.from_array(W1, "W1"),
                numpy_helper.from_array(b, "b1"),
                numpy_helper.from_array(W2, "W2"),
                numpy_helper.from_array(b, "b2"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        path = self._save(model, tmp_path, "two_branch.onnx")

        with pytest.raises(ValueError, match="non-sequential"):
            load_onnx(path)

    def test_transb_roundtrips(self, tmp_path):
        """transB=1 stores W transposed; forward matches onnxruntime."""
        from discopt.nn.readers.onnx_reader import load_onnx
        from onnx import TensorProto, helper, numpy_helper

        rng = np.random.RandomState(1)
        # With transB=1, ONNX computes X @ W.T, so store W as (n_out, n_in).
        W = rng.randn(3, 2).astype(np.float32)
        b = rng.randn(3).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Gemm", ["X", "W", "b"], ["Y"], transB=1)
        graph = helper.make_graph(
            [node],
            "transb",
            [X],
            [Y],
            [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        path = self._save(model, tmp_path, "transb.onnx")

        net = load_onnx(path, input_bounds=(-np.ones(2), np.ones(2)))
        for _ in range(10):
            x = rng.uniform(-1.0, 1.0, size=2)
            np.testing.assert_allclose(
                net.forward(x), self._ort_forward(path, x), atol=1e-5, rtol=1e-5
            )


# ---------------------------------------------------------------------------
# T-N1.2 — sklearn semantics + tree edge
# ---------------------------------------------------------------------------


class TestSklearnReaderFixes:
    @pytest.fixture(autouse=True)
    def _check_sklearn(self):
        pytest.importorskip("sklearn")

    def test_binary_mlp_classifier_sigmoid_matches_proba(self):
        """Binary MLPClassifier embeds with a SIGMOID head matching predict_proba."""
        from discopt.nn.network import Activation
        from discopt.nn.readers.sklearn_reader import load_sklearn_mlp
        from sklearn.neural_network import MLPClassifier

        rng = np.random.RandomState(0)
        X = rng.randn(80, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=300, random_state=0)
        clf.fit(X, y)
        assert clf.out_activation_ == "logistic"

        net = load_sklearn_mlp(clf, input_bounds=(-2 * np.ones(2), 2 * np.ones(2)))
        assert net.layers[-1].activation == Activation.SIGMOID

        proba = clf.predict_proba(X)[:, 1]
        got = np.array([net.forward(X[i])[0] for i in range(len(X))])
        np.testing.assert_allclose(got, proba, atol=1e-6)

    def test_multiclass_mlp_classifier_raises(self):
        """Softmax (multi-class) MLPClassifier has no scalar embedding -> ValueError."""
        from discopt.nn.readers.sklearn_reader import load_sklearn_mlp
        from sklearn.neural_network import MLPClassifier

        rng = np.random.RandomState(0)
        X = rng.randn(90, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 0] > 0.5).astype(int)
        clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=200, random_state=0)
        clf.fit(X, y)
        assert clf.out_activation_ == "softmax"

        with pytest.raises(ValueError, match="softmax"):
            load_sklearn_mlp(clf)

    def test_gradient_boosting_classifier_raises(self):
        from discopt.nn.readers.sklearn_reader import load_sklearn_ensemble
        from sklearn.ensemble import GradientBoostingClassifier

        rng = np.random.RandomState(0)
        X = rng.randn(60, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=0)
        clf.fit(X, y)
        with pytest.raises(TypeError, match="regressor"):
            load_sklearn_ensemble(clf)

    def test_random_forest_classifier_raises(self):
        from discopt.nn.readers.sklearn_reader import load_sklearn_ensemble
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(0)
        X = rng.randn(60, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=0)
        clf.fit(X, y)
        with pytest.raises(TypeError, match="regressor"):
            load_sklearn_ensemble(clf)

    def test_decision_tree_classifier_raises(self):
        from discopt.nn.readers.sklearn_reader import load_sklearn_tree
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.RandomState(0)
        X = rng.randn(60, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        clf.fit(X, y)
        with pytest.raises(TypeError, match="regressor"):
            load_sklearn_tree(clf)

    def test_constant_target_regressor_single_leaf(self):
        """A constant-target regressor (single leaf) loads and predicts (no 0-d crash)."""
        from discopt.nn.readers.sklearn_reader import load_sklearn_tree
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.RandomState(0)
        X = rng.randn(40, 2)
        y = np.full(40, 3.5)  # constant target -> a single leaf
        dt = DecisionTreeRegressor(max_depth=3, random_state=0)
        dt.fit(X, y)
        assert dt.tree_.node_count == 1  # sanity: really a single-leaf tree

        ens = load_sklearn_tree(dt, input_bounds=(-np.ones(2), np.ones(2)))
        assert ens.n_features == 2
        for i in range(5):
            assert ens.predict(X[i]) == pytest.approx(3.5, abs=1e-9)


# ---------------------------------------------------------------------------
# T-N1.3 — torch bias=False
# ---------------------------------------------------------------------------


class TestTorchReaderBiasFalse:
    @pytest.fixture(autouse=True)
    def _check_torch(self):
        pytest.importorskip("torch")

    def test_linear_bias_false_matches_forward(self):
        """A Linear(bias=False) layer loads (zero bias) and matches torch forward."""
        import torch
        import torch.nn as nn
        from discopt.nn.readers.torch_reader import load_torch_sequential

        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(2, 3, bias=False),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        model.eval()

        net = load_torch_sequential(model, input_bounds=(-np.ones(2), np.ones(2)))
        # First layer has no bias -> zeros.
        np.testing.assert_allclose(net.layers[0].biases, np.zeros(3), atol=0.0)

        rng = np.random.RandomState(0)
        for _ in range(10):
            x = rng.uniform(-1.0, 1.0, size=2)
            with torch.no_grad():
                ref = model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
            np.testing.assert_allclose(net.forward(x), ref, atol=1e-5, rtol=1e-5)
