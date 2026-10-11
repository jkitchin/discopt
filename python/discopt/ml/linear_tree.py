"""Linear model decision trees (oblique splits, affine leaves) for MILP embedding.

A linear model decision tree partitions the input box with hyperplane splits
``a . x <= t`` and predicts with an affine model ``y = A_l x + b_l`` in each
leaf ``l``. It is piecewise affine, so a model containing one can be solved as
a MILP (or MIQP/MIQCP) instead of an MINLP.

:class:`LinearTreeDefinition` stores the tree as one polytope per leaf in the
original input units,

    P_l = {x : G_l x <= h_l},   y = A_l x + b_l  on P_l,

where the rows of ``G_l`` are the root-to-leaf splits (taken left or right),
followed by the input box. The polytopes cover the box, overlapping only on
shared split facets.

The definition is read from the solver-neutral *v1 hyperplane-tree spec*
written by jax-ldt and by jax-difftree (``difftree.export.to_spec``), so
discopt does not depend on either package. See :meth:`from_spec`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LinearLeaf:
    """One leaf: the polytope ``{x : G x <= h}`` and its model ``y = A x + b``.

    Parameters
    ----------
    G : np.ndarray
        ``(n_rows, n_features)``. The first ``n_split_rows`` rows are the
        root-to-leaf splits; the remaining rows are the input box
        (``x <= ub`` then ``-x <= -lb``).
    h : np.ndarray
        ``(n_rows,)``.
    A : np.ndarray
        ``(n_outputs, n_features)`` leaf coefficients.
    b : np.ndarray
        ``(n_outputs,)`` leaf intercepts.
    n_split_rows : int
        Number of leading rows of ``G`` that are splits (the rest is the box).
    split_sides : list of str
        ``"left"`` or ``"right"`` for each split row, in order.
    """

    G: np.ndarray
    h: np.ndarray
    A: np.ndarray
    b: np.ndarray
    n_split_rows: int
    split_sides: list[str] = field(default_factory=list)


@dataclass
class LinearTreeDefinition:
    """A linear model decision tree as a list of leaf polytopes.

    Parameters
    ----------
    leaves : list of LinearLeaf
        One entry per leaf.
    n_features : int
        Number of inputs.
    n_outputs : int
        Number of outputs.
    input_bounds : tuple of np.ndarray
        ``(lower, upper)`` finite bounds on each input. Required: the hull
        encoding needs bounded leaf polytopes and the big-M encoding needs the
        box for its constants.
    """

    leaves: list[LinearLeaf]
    n_features: int
    n_outputs: int
    input_bounds: tuple[np.ndarray, np.ndarray]

    def __post_init__(self):
        lb, ub = (np.asarray(v, dtype=np.float64).reshape(-1) for v in self.input_bounds)
        if lb.shape != (self.n_features,) or ub.shape != (self.n_features,):
            raise ValueError(
                f"input_bounds must be two arrays of length {self.n_features}, "
                f"got shapes {lb.shape} and {ub.shape}"
            )
        if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
            raise ValueError("input_bounds must be finite for a linear tree formulation")
        if np.any(lb > ub):
            raise ValueError("input_bounds has lower > upper")
        self.input_bounds = (lb, ub)
        if not self.leaves:
            raise ValueError("a linear tree needs at least one leaf")
        for k, leaf in enumerate(self.leaves):
            if leaf.G.shape[1] != self.n_features or leaf.G.shape[0] != leaf.h.shape[0]:
                raise ValueError(f"leaf {k}: G is {leaf.G.shape}, h is {leaf.h.shape}")
            if leaf.A.shape != (self.n_outputs, self.n_features) or leaf.b.shape != (
                self.n_outputs,
            ):
                raise ValueError(f"leaf {k}: A is {leaf.A.shape}, b is {leaf.b.shape}")

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    def leaf_of(self, x: np.ndarray) -> int:
        """Index of the leaf the tree routes ``x`` to (left iff ``a . x <= t``).

        Leaves are tested in order with the spec's tie rule: a split row taken
        *left* is ``a . x <= t`` (closed), a row taken *right* is ``a . x > t``
        (open). Exactly one leaf matches any ``x`` in the box.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        for k, leaf in enumerate(self.leaves):
            r = leaf.G[: leaf.n_split_rows] @ x - leaf.h[: leaf.n_split_rows]
            ok = True
            for v, side in zip(r, leaf.split_sides):
                # left rows are a.x - t <= 0; right rows are stored as
                # -a.x + t <= 0 and must hold strictly (a.x > t).
                if (side == "left" and v > 0.0) or (side == "right" and v >= 0.0):
                    ok = False
                    break
            if ok:
                return k
        raise ValueError(f"no leaf routes x = {x}; the spec's splits do not cover it")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Tree prediction at a single input, shape ``(n_outputs,)``."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        leaf = self.leaves[self.leaf_of(x)]
        return np.asarray(leaf.A @ x + leaf.b, dtype=np.float64)

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(
        cls,
        spec: dict | str | Path,
        input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> LinearTreeDefinition:
        """Build from a v1 hyperplane-tree spec (dict, JSON string, or file path).

        The spec (written by jax-ldt and by ``difftree.export.to_spec``) stores
        a transform ``x' = x @ T`` with ``T`` of shape ``(n_features,
        n_transformed)``; internal node ``i`` goes left iff ``x'[feature[i]] <=
        threshold[i]``; leaf ``i`` predicts ``[1, x'] @ leaf_params[i]`` with
        ``leaf_params[i]`` of shape ``(1 + n_transformed, n_targets)``.

        Parameters
        ----------
        spec : dict, str, or Path
            The spec, its JSON text, or a path to a ``.json`` file.
        input_bounds : tuple of np.ndarray, optional
            ``(lower, upper)``. Overrides the bounds stored in the spec
            (``spec["difftree"]["bounds"]``); one of the two is required.
        """
        data: dict
        if isinstance(spec, dict):
            data = spec
        elif isinstance(spec, str) and spec.lstrip().startswith("{"):
            data = json.loads(spec)
        else:
            data = json.loads(Path(spec).read_text())
        spec_d = data
        if spec_d.get("version") != 1 or spec_d.get("kind") != "hyperplane_tree":
            raise ValueError(
                f"expected a v1 'hyperplane_tree' spec, got version={spec_d.get('version')!r} "
                f"kind={spec_d.get('kind')!r}"
            )
        if spec_d.get("categorical_features"):
            raise NotImplementedError(
                "categorical features in a linear tree spec are not supported yet"
            )

        p = int(spec_d["n_features_in"])
        d_y = int(spec_d["n_targets"])
        T = np.asarray(spec_d["transform_matrix"], dtype=np.float64).reshape(p, -1)
        tree = spec_d["tree"]
        is_leaf = [bool(v) for v in tree["is_leaf"]]
        feature = [int(v) for v in tree["feature"]]
        threshold = tree["threshold"]
        left = [int(v) for v in tree["left"]]
        right = [int(v) for v in tree["right"]]
        params = np.asarray(tree["leaf_params"], dtype=np.float64)

        if input_bounds is None:
            stored = spec_d.get("difftree", {}).get("bounds")
            if stored is None:
                raise ValueError(
                    "input_bounds is required: the spec carries no bounds "
                    "(spec_d['difftree']['bounds'])"
                )
            input_bounds = (np.asarray(stored[0], float), np.asarray(stored[1], float))
        lb = np.asarray(input_bounds[0], dtype=np.float64).reshape(-1)
        ub = np.asarray(input_bounds[1], dtype=np.float64).reshape(-1)
        box_G = np.vstack([np.eye(p), -np.eye(p)])
        box_h = np.concatenate([ub, -lb])

        leaves: list[LinearLeaf] = []

        def walk(node: int, rows: list[np.ndarray], rhs: list[float], sides: list[str]):
            if is_leaf[node]:
                P = params[node]  # (1 + n_t, d_y)
                A = (T @ P[1:]).T  # (d_y, p)
                b = P[0].copy()
                G = np.vstack(rows + [box_G]) if rows else box_G.copy()
                h = np.concatenate([np.asarray(rhs, float), box_h]) if rhs else box_h.copy()
                leaves.append(LinearLeaf(G, h, A, b, len(rows), list(sides)))
                return
            a = T[:, feature[node]]
            t = float(threshold[node])
            walk(left[node], rows + [a], rhs + [t], sides + ["left"])
            walk(right[node], rows + [-a], rhs + [-t], sides + ["right"])

        walk(0, [], [], [])
        return cls(leaves=leaves, n_features=p, n_outputs=d_y, input_bounds=(lb, ub))
