"""Tests for tree ensemble embedding (Tier 2 of issue #1)."""

from __future__ import annotations

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────


def _simple_tree():
    """A 2-feature tree: if x[0] <= 0.5 -> 1.0, else 2.0."""
    from discopt.ml.tree import DecisionTree

    return DecisionTree(
        n_features=2,
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, 1.0, 2.0]),
    )


def _deeper_tree():
    """A deeper tree with 3 leaves on feature 0 and 1.

    Node 0: x[0] <= 0.5 -> left=1, right=2
    Node 1: leaf, value=1.0
    Node 2: x[1] <= 0.3 -> left=3, right=4
    Node 3: leaf, value=3.0
    Node 4: leaf, value=5.0
    """
    from discopt.ml.tree import DecisionTree

    return DecisionTree(
        n_features=2,
        feature=np.array([0, -1, 1, -1, -1]),
        threshold=np.array([0.5, 0.0, 0.3, 0.0, 0.0]),
        left_child=np.array([1, -1, 3, -1, -1]),
        right_child=np.array([2, -1, 4, -1, -1]),
        value=np.array([0.0, 1.0, 0.0, 3.0, 5.0]),
    )


class TestDecisionTree:
    def test_leaves(self):
        tree = _simple_tree()
        np.testing.assert_array_equal(tree.leaves, [1, 2])
        assert tree.n_leaves == 2

    def test_leaves_deeper(self):
        tree = _deeper_tree()
        np.testing.assert_array_equal(tree.leaves, [1, 3, 4])
        assert tree.n_leaves == 3

    def test_predict_left(self):
        tree = _simple_tree()
        assert tree.predict(np.array([0.3, 0.0])) == 1.0

    def test_predict_right(self):
        tree = _simple_tree()
        assert tree.predict(np.array([0.7, 0.0])) == 2.0

    def test_predict_deeper(self):
        tree = _deeper_tree()
        assert tree.predict(np.array([0.3, 0.0])) == 1.0
        assert tree.predict(np.array([0.7, 0.2])) == 3.0
        assert tree.predict(np.array([0.7, 0.5])) == 5.0

    def test_leaf_ancestors(self):
        tree = _simple_tree()
        # Leaf 1 is left child of node 0
        assert tree.leaf_ancestors(1) == [(0, "left")]
        # Leaf 2 is right child of node 0
        assert tree.leaf_ancestors(2) == [(0, "right")]

    def test_leaf_ancestors_deeper(self):
        tree = _deeper_tree()
        # Leaf 3: node 0 right, node 2 left
        assert tree.leaf_ancestors(3) == [(0, "right"), (2, "left")]
        # Leaf 4: node 0 right, node 2 right
        assert tree.leaf_ancestors(4) == [(0, "right"), (2, "right")]

    def test_validation(self):
        from discopt.ml.tree import DecisionTree

        with pytest.raises(ValueError):
            DecisionTree(
                n_features=1,
                feature=np.array([0, -1]),
                threshold=np.array([0.5]),  # wrong length
                left_child=np.array([1, -1]),
                right_child=np.array([2, -1]),
                value=np.array([0.0, 1.0]),
            )


class TestTreeEnsembleDefinition:
    def test_predict_single_tree(self):
        from discopt.ml.tree import TreeEnsembleDefinition

        tree = _simple_tree()
        ens = TreeEnsembleDefinition(trees=[tree], n_features=2)
        assert ens.predict(np.array([0.3, 0.0])) == 1.0

    def test_predict_two_trees(self):
        from discopt.ml.tree import TreeEnsembleDefinition

        t1 = _simple_tree()
        t2 = _simple_tree()
        ens = TreeEnsembleDefinition(trees=[t1, t2], n_features=2)
        # Both predict 1.0 for x[0]=0.3
        assert ens.predict(np.array([0.3, 0.0])) == 2.0

    def test_base_score(self):
        from discopt.ml.tree import TreeEnsembleDefinition

        tree = _simple_tree()
        ens = TreeEnsembleDefinition(trees=[tree], n_features=2, base_score=10.0)
        assert ens.predict(np.array([0.3, 0.0])) == 11.0


# ─────────────────────────────────────────────────────────────
# MILP formulation
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestTreeEnsembleFormulation:
    def test_single_tree_fixed_input(self):
        """Fix input to left branch, verify leaf value."""
        import discopt.modeling as dm
        from discopt.ml import TreeEnsembleDefinition, TreeFormulation

        tree = _simple_tree()
        ens = TreeEnsembleDefinition(
            trees=[tree],
            n_features=2,
            input_bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )

        m = dm.Model("tree_fixed")
        tf = TreeFormulation(m, ens)
        tf.formulate()

        # Fix input to left branch
        m.subject_to(tf.inputs[0] == 0.3)
        m.subject_to(tf.inputs[1] == 0.0)
        m.minimize(tf.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        np.testing.assert_allclose(result.objective, 1.0, atol=1e-4)

    def test_single_tree_optimize(self):
        """Minimize over a tree should select the leaf with minimum value."""
        import discopt.modeling as dm
        from discopt.ml import TreeEnsembleDefinition, TreeFormulation

        tree = _simple_tree()
        ens = TreeEnsembleDefinition(
            trees=[tree],
            n_features=2,
            input_bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )

        m = dm.Model("tree_opt")
        tf = TreeFormulation(m, ens)
        tf.formulate()
        m.minimize(tf.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        # Minimum leaf value is 1.0 (left branch)
        np.testing.assert_allclose(result.objective, 1.0, atol=1e-4)

    def test_two_tree_ensemble(self):
        """Ensemble of 2 trees, verify output matches predict()."""
        import discopt.modeling as dm
        from discopt.ml import TreeEnsembleDefinition, TreeFormulation

        t1 = _simple_tree()
        t2 = _deeper_tree()
        ens = TreeEnsembleDefinition(
            trees=[t1, t2],
            n_features=2,
            input_bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )

        m = dm.Model("ensemble")
        tf = TreeFormulation(m, ens)
        tf.formulate()

        # Fix inputs and verify output
        m.subject_to(tf.inputs[0] == 0.3)
        m.subject_to(tf.inputs[1] == 0.5)
        m.minimize(tf.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"

        expected = ens.predict(np.array([0.3, 0.5]))
        np.testing.assert_allclose(result.objective, expected, atol=1e-4)

    def test_requires_input_bounds(self):
        from discopt.ml import TreeEnsembleDefinition, TreeFormulation

        tree = _simple_tree()
        ens = TreeEnsembleDefinition(trees=[tree], n_features=2)
        import discopt.modeling as dm

        m = dm.Model("no_bounds")
        with pytest.raises(ValueError, match="input_bounds"):
            tf = TreeFormulation(m, ens)
            tf.formulate()


# ─────────────────────────────────────────────────────────────
# add_predictor dispatcher
# ─────────────────────────────────────────────────────────────


class TestAddPredictor:
    def test_auto_detect_network(self):
        import discopt.modeling as dm
        from discopt.ml import NetworkDefinition, add_predictor
        from discopt.ml.network import Activation, DenseLayer

        W = np.array([[1.0], [-1.0]], dtype=np.float64)
        b = np.array([0.0], dtype=np.float64)
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.LINEAR)],
            input_bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        )

        m = dm.Model("ap_nn")
        x = m.continuous("x", shape=(2,), lb=-1, ub=1)
        outputs, form = add_predictor(m, x, net, prefix="nn")
        assert outputs is not None
        assert form is not None

    def test_auto_detect_tree(self):
        import discopt.modeling as dm
        from discopt.ml import TreeEnsembleDefinition, add_predictor

        tree = _simple_tree()
        ens = TreeEnsembleDefinition(
            trees=[tree],
            n_features=2,
            input_bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )

        m = dm.Model("ap_tree")
        x = m.continuous("x", shape=(2,), lb=0, ub=1)
        outputs, form = add_predictor(m, x, ens, prefix="tf")
        assert outputs is not None
        assert form is not None

    def test_unknown_type_raises(self):
        import discopt.modeling as dm
        from discopt.ml import add_predictor

        m = dm.Model("ap_bad")
        x = m.continuous("x", shape=(1,), lb=0, ub=1)
        with pytest.raises(TypeError):
            add_predictor(m, x, 12345)


# ─────────────────────────────────────────────────────────────
# sklearn readers (requires scikit-learn)
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSklearnReaders:
    @pytest.fixture(autouse=True)
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_load_decision_tree(self):
        from discopt.ml import load_sklearn_tree
        from sklearn.tree import DecisionTreeRegressor

        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([1.0, 2.0, 3.0])
        dt = DecisionTreeRegressor(max_depth=2, random_state=0)
        dt.fit(X, y)

        ens = load_sklearn_tree(dt, input_bounds=(np.array([0.0]), np.array([1.0])))
        assert ens.n_features == 1
        assert len(ens.trees) == 1

        # Verify predict matches
        for xi in [0.0, 0.25, 0.75, 1.0]:
            x = np.array([xi])
            np.testing.assert_allclose(ens.predict(x), dt.predict(x.reshape(1, -1))[0], atol=1e-10)

    def test_load_gradient_boosting(self):
        from discopt.ml import load_sklearn_ensemble
        from sklearn.ensemble import GradientBoostingRegressor

        rng = np.random.RandomState(42)
        X = rng.rand(50, 2)
        y = X[:, 0] + 0.5 * X[:, 1]
        gbr = GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=42)
        gbr.fit(X, y)

        ens = load_sklearn_ensemble(gbr, input_bounds=(np.zeros(2), np.ones(2)))
        assert ens.n_features == 2
        assert len(ens.trees) == 5

        # Verify predict matches sklearn
        for i in range(5):
            x = X[i]
            np.testing.assert_allclose(ens.predict(x), gbr.predict(x.reshape(1, -1))[0], atol=1e-6)

    def test_load_sklearn_mlp(self):
        from discopt.ml import load_sklearn_mlp
        from sklearn.neural_network import MLPRegressor

        rng = np.random.RandomState(0)
        X = rng.rand(100, 2)
        y = X[:, 0] + X[:, 1]
        mlp = MLPRegressor(
            hidden_layer_sizes=(4,),
            activation="relu",
            max_iter=500,
            random_state=0,
        )
        mlp.fit(X, y)

        net = load_sklearn_mlp(mlp, input_bounds=(np.zeros(2), np.ones(2)))
        assert net.input_size == 2
        assert net.output_size == 1

        # Verify forward pass matches sklearn
        for i in range(5):
            x = X[i]
            np.testing.assert_allclose(net.forward(x), mlp.predict(x.reshape(1, -1))[0], atol=1e-6)

    def test_sklearn_tree_milp_roundtrip(self):
        """Train sklearn tree, embed in discopt, solve, verify."""
        import discopt.modeling as dm
        from discopt.ml import TreeFormulation, load_sklearn_tree
        from sklearn.tree import DecisionTreeRegressor

        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        y = np.array([1.0, 2.0, 4.0])
        dt = DecisionTreeRegressor(max_depth=2, random_state=0)
        dt.fit(X, y)

        ens = load_sklearn_tree(dt, input_bounds=(np.zeros(2), np.ones(2)))

        m = dm.Model("sklearn_rt")
        tf = TreeFormulation(m, ens)
        tf.formulate()

        # Fix input and verify output matches sklearn
        m.subject_to(tf.inputs[0] == 0.5)
        m.subject_to(tf.inputs[1] == 0.5)
        m.minimize(tf.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"

        expected = dt.predict(np.array([[0.5, 0.5]]))[0]
        np.testing.assert_allclose(result.objective, expected, atol=1e-4)
