"""MILP encodings of a linear model decision tree (oblique splits, affine leaves).

Two encodings are provided for a tree with leaves ``l`` whose polytopes are
``P_l = {x : G_l x <= h_l}`` (splits, then the input box) and whose models are
``y = A_l x + b_l``:

``"hull"`` (disaggregated, convex-hull / Balas)
    One copy ``x_l`` of the inputs per leaf, and one binary ``z_l``::

        x = sum_l x_l,        sum_l z_l = 1,
        G_l x_l <= h_l z_l,   y = sum_l (A_l x_l + b_l z_l).

    No big-M constants. Its LP relaxation is the convex hull of the union of
    the leaf graphs, so it is the tightest encoding; it has ``L * n`` extra
    continuous variables. Because the box rows are in every ``G_l``, ``z_l =
    0`` forces ``x_l = 0``.

``"bigm"``
    One binary per leaf and no disaggregation::

        sum_l z_l = 1,
        G_l x <= h_l + M_l (1 - z_l),
        -Mo_l (1 - z_l) <= y - (A_l x + b_l) <= Mo_l (1 - z_l).

    Each constant is the exact interval maximum of its row over the input box,
    so a deselected leaf never cuts a feasible point. Smaller, weaker
    relaxation; kept for comparison.

Splits. A split row taken left is ``a . x <= t`` and taken right is ``a . x >
t``. A MILP cannot express the strict side, so both sides are written as
closed half-spaces shifted by ``split_eps * ||a||``: ``a . x <= t - eps`` on
the left, ``a . x >= t + eps`` on the right. This follows
:class:`~discopt.ml.formulations.tree_ensemble.TreeEnsembleFormulation`
(#1447): a tree is discontinuous at its splits, so a point the solver puts on
a split within its feasibility tolerance could otherwise be assigned to the
leaf ``predict`` does not route it to. ``split_eps`` must exceed the solver's
feasibility tolerance (1e-6); the cost is that the open band of half-width
``eps`` around each split is excluded. Set ``split_eps=0`` only for a tree
that is continuous across its splits (e.g. difftree with tied splits), where
either leaf gives the same prediction on the facet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import discopt.modeling as dm
from discopt.ml.linear_tree import LinearTreeDefinition

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

_ENCODINGS = ("hull", "bigm")


def _shifted_rows(leaf, split_eps: float) -> tuple[np.ndarray, np.ndarray]:
    """``(G, h)`` with each split row tightened by ``split_eps * ||a||``."""
    G = leaf.G
    h = leaf.h.astype(np.float64).copy()
    if split_eps > 0.0 and leaf.n_split_rows:
        norms = np.linalg.norm(G[: leaf.n_split_rows], axis=1)
        h[: leaf.n_split_rows] -= split_eps * norms
    return G, h


def _box_range(c: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> tuple[float, float]:
    """Exact ``(min, max)`` of ``c . x`` over the box ``[lb, ub]``."""
    lo = float(np.sum(np.minimum(c * lb, c * ub)))
    hi = float(np.sum(np.maximum(c * lb, c * ub)))
    return lo, hi


class LinearTreeFormulation:
    """Embed a :class:`~discopt.ml.linear_tree.LinearTreeDefinition` as MILP constraints.

    Parameters
    ----------
    model : discopt.Model
        Model to add variables and constraints to.
    definition : LinearTreeDefinition
        The tree.
    prefix : str
        Name prefix for created variables and constraints.
    encoding : str
        ``"hull"`` (default) or ``"bigm"``; see the module docstring.
    split_eps : float
        Separation half-width at the splits (default 1e-5); see the module
        docstring.
    """

    def __init__(
        self,
        model: Model,
        definition: LinearTreeDefinition,
        prefix: str = "ltree",
        encoding: str = "hull",
        split_eps: float = 1e-5,
    ) -> None:
        if encoding not in _ENCODINGS:
            raise ValueError(f"Unknown encoding '{encoding}'. Choose from: {list(_ENCODINGS)}")
        if split_eps < 0.0:
            raise ValueError("split_eps must be non-negative")
        self._model = model
        self._def = definition
        self._prefix = prefix
        self._encoding = encoding
        self._split_eps = float(split_eps)
        self._inputs: Variable | None = None
        self._outputs: Variable | None = None
        self._leaf: Variable | None = None
        self._formulated = False

    @property
    def inputs(self) -> Variable:
        """Input variables, shape ``(n_features,)``."""
        if self._inputs is None:
            raise RuntimeError("Call formulate() before accessing inputs")
        return self._inputs

    @property
    def outputs(self) -> Variable:
        """Output variables, shape ``(n_outputs,)``."""
        if self._outputs is None:
            raise RuntimeError("Call formulate() before accessing outputs")
        return self._outputs

    @property
    def leaf_indicators(self) -> Variable:
        """Leaf binaries ``z``, shape ``(n_leaves,)``; exactly one is 1."""
        if self._leaf is None:
            raise RuntimeError("Call formulate() before accessing leaf_indicators")
        return self._leaf

    def output_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Valid ``(lower, upper)`` on each output: box ranges of every leaf model."""
        lb, ub = self._def.input_bounds
        lo = np.full(self._def.n_outputs, np.inf)
        hi = np.full(self._def.n_outputs, -np.inf)
        for leaf in self._def.leaves:
            for k in range(self._def.n_outputs):
                a, c = _box_range(leaf.A[k], lb, ub)
                lo[k] = min(lo[k], a + leaf.b[k])
                hi[k] = max(hi[k], c + leaf.b[k])
        return lo, hi

    def formulate(self) -> None:
        """Add all variables and constraints to the model (once)."""
        if self._formulated:
            raise RuntimeError("formulate() has already been called")
        if self._encoding == "hull":
            self._build_hull()
        else:
            self._build_bigm()
        self._formulated = True

    # ------------------------------------------------------------------

    def _build_hull(self) -> None:
        m, d, pfx = self._model, self._def, self._prefix
        lb, ub = d.input_bounds
        L, p, q = d.n_leaves, d.n_features, d.n_outputs
        y_lo, y_hi = self.output_bounds()

        x = m.continuous(f"{pfx}_input", shape=(p,), lb=lb, ub=ub)
        y = m.continuous(f"{pfx}_output", shape=(q,), lb=y_lo, ub=y_hi)
        z = m.binary(f"{pfx}_leaf", shape=(L,))
        # x_l lies in [min(lb,0), max(ub,0)] because it is either x or 0.
        xl = m.continuous(
            f"{pfx}_xdis",
            shape=(L, p),
            lb=np.repeat(np.minimum(lb, 0.0)[None], L, axis=0),
            ub=np.repeat(np.maximum(ub, 0.0)[None], L, axis=0),
        )

        m.subject_to(dm.sum(z) == 1, name=f"{pfx}_one_leaf")
        for j in range(p):
            m.subject_to(
                x[j] == dm.sum(lambda i, _j=j: xl[i][_j], over=range(L)),
                name=f"{pfx}_xsum_{j}",
            )
        for l_idx, leaf in enumerate(d.leaves):
            G, h = _shifted_rows(leaf, self._split_eps)
            m.subject_to(G @ xl[l_idx] <= h * z[l_idx], name=f"{pfx}_poly_{l_idx}")
        for k in range(q):
            m.subject_to(
                y[k]
                == dm.sum(
                    lambda i, _k=k: d.leaves[i].A[_k] @ xl[i] + float(d.leaves[i].b[_k]) * z[i],
                    over=range(L),
                ),
                name=f"{pfx}_out_{k}",
            )
        self._inputs, self._outputs, self._leaf = x, y, z

    def _build_bigm(self) -> None:
        m, d, pfx = self._model, self._def, self._prefix
        lb, ub = d.input_bounds
        L, p, q = d.n_leaves, d.n_features, d.n_outputs
        y_lo, y_hi = self.output_bounds()

        x = m.continuous(f"{pfx}_input", shape=(p,), lb=lb, ub=ub)
        y = m.continuous(f"{pfx}_output", shape=(q,), lb=y_lo, ub=y_hi)
        z = m.binary(f"{pfx}_leaf", shape=(L,))
        m.subject_to(dm.sum(z) == 1, name=f"{pfx}_one_leaf")

        for l_idx, leaf in enumerate(d.leaves):
            G, h = _shifted_rows(leaf, self._split_eps)
            # Split rows only: the box rows hold for every x already.
            ns = leaf.n_split_rows
            if ns:
                M = np.array([max(_box_range(G[r], lb, ub)[1] - h[r], 0.0) for r in range(ns)])
                m.subject_to(
                    G[:ns] @ x <= h[:ns] + M * (1 - z[l_idx]),
                    name=f"{pfx}_split_{l_idx}",
                )
            for k in range(q):
                f_lo, f_hi = _box_range(leaf.A[k], lb, ub)
                f_lo += float(leaf.b[k])
                f_hi += float(leaf.b[k])
                # y - f <= y_hi - f_lo and f - y <= f_hi - y_lo when deselected.
                M_up = max(float(y_hi[k]) - f_lo, 0.0)
                M_dn = max(f_hi - float(y_lo[k]), 0.0)
                f = leaf.A[k] @ x + float(leaf.b[k])
                m.subject_to(y[k] - f <= M_up * (1 - z[l_idx]), name=f"{pfx}_oup_{l_idx}_{k}")
                m.subject_to(f - y[k] <= M_dn * (1 - z[l_idx]), name=f"{pfx}_odn_{l_idx}_{k}")
        self._inputs, self._outputs, self._leaf = x, y, z
