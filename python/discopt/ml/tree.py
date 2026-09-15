"""Tree ensemble data structures for MILP embedding.

Provides :class:`DecisionTree` and :class:`TreeEnsembleDefinition` for
representing trained decision tree ensembles (random forests, gradient
boosting) in a solver-agnostic format suitable for MILP encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LEAF_MARKER = -1


@dataclass
class DecisionTree:
    """A single decision tree stored as parallel arrays.

    Node ``i`` is a leaf when ``feature[i] == -1``. Otherwise it is a
    split node that branches on ``x[feature[i]] <= threshold[i]``.

    Parameters
    ----------
    n_features : int
        Number of input features.
    feature : np.ndarray
        ``(n_nodes,)`` int array. Feature index for each split node,
        ``-1`` for leaves.
    threshold : np.ndarray
        ``(n_nodes,)`` float array. Split threshold (ignored at leaves).
    left_child : np.ndarray
        ``(n_nodes,)`` int array. Index of left child (``-1`` at leaves).
    right_child : np.ndarray
        ``(n_nodes,)`` int array. Index of right child (``-1`` at leaves).
    value : np.ndarray
        ``(n_nodes,)`` float array. Prediction value at each leaf.
    """

    n_features: int
    feature: np.ndarray
    threshold: np.ndarray
    left_child: np.ndarray
    right_child: np.ndarray
    value: np.ndarray

    def __post_init__(self):
        n = len(self.feature)
        for name in ("threshold", "left_child", "right_child", "value"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != feature length {n}")

    @property
    def leaves(self) -> np.ndarray:
        """Indices of leaf nodes."""
        idx: np.ndarray = np.where(self.feature == _LEAF_MARKER)[0]
        return idx

    @property
    def n_leaves(self) -> int:
        return int(np.sum(self.feature == _LEAF_MARKER))

    def leaf_ancestors(self, leaf_idx: int) -> list[tuple[int, str]]:
        """Return the root-to-leaf path as ``(node, 'left'|'right')`` pairs.

        Parameters
        ----------
        leaf_idx : int
            Index of a leaf node.

        Returns
        -------
        list of (int, str)
            Each entry is ``(split_node_index, direction)`` where direction
            indicates which branch was taken to reach the leaf.
        """
        parent = np.full(len(self.feature), -1, dtype=int)
        direction = np.empty(len(self.feature), dtype=object)
        for i in range(len(self.feature)):
            if self.left_child[i] != _LEAF_MARKER:
                parent[self.left_child[i]] = i
                direction[self.left_child[i]] = "left"
            if self.right_child[i] != _LEAF_MARKER:
                parent[self.right_child[i]] = i
                direction[self.right_child[i]] = "right"

        path = []
        node = leaf_idx
        while parent[node] != -1:
            path.append((int(parent[node]), direction[node]))
            node = parent[node]
        path.reverse()
        return path

    def predict(self, x: np.ndarray) -> float:
        """Evaluate the tree on a single input vector."""
        node = 0
        while self.feature[node] != _LEAF_MARKER:
            if x[self.feature[node]] <= self.threshold[node]:
                node = self.left_child[node]
            else:
                node = self.right_child[node]
        return float(self.value[node])


@dataclass
class TreeEnsembleDefinition:
    """An ensemble of decision trees (random forest, gradient boosting, etc.).

    Parameters
    ----------
    trees : list of DecisionTree
        Individual trees in the ensemble.
    n_features : int
        Number of input features.
    base_score : float
        Additive intercept (e.g. XGBoost ``base_score``).
    input_bounds : tuple of np.ndarray, optional
        ``(lower, upper)`` bounds on each feature. Required for MILP
        formulation.
    """

    trees: list[DecisionTree]
    n_features: int
    base_score: float = 0.0
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def predict(self, x: np.ndarray) -> float:
        """Evaluate the ensemble on a single input vector."""
        return sum(t.predict(x) for t in self.trees) + self.base_score
