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

On/off (semicontinuous) units. With ``active`` set, the tree models a unit
that can be switched off. ``sum_l z_l = a`` replaces ``sum_l z_l = 1`` for a
binary ``a``: when ``a = 1`` exactly one leaf is active and the input lies in
the box, as before; when ``a = 0`` no leaf is active and the inputs and
outputs are all zero, even if zero lies outside the input box (a minimum
turndown above zero, for example). In the hull encoding this follows from the
disaggregation (``x = sum_l x_l`` with every ``x_l = 0``). In the big-M
encoding it is enforced by ``lb a <= x <= ub a`` and ``y_lo a <= y <= y_hi
a``, and every big-M constant is taken over the box extended to contain the
origin so that the deselected rows stay valid at ``x = 0``. Without
``active`` the formulation is unchanged.

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

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

import discopt.modeling as dm
from discopt.ml.linear_tree import LinearTreeDefinition

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

_ENCODINGS = ("hull", "bigm")


class _Box(NamedTuple):
    """Input box and output bounds, each as ``(lower, upper)`` arrays."""

    x_lo: np.ndarray
    x_hi: np.ndarray
    y_lo: np.ndarray
    y_hi: np.ndarray


def _check_active(active, model) -> None:
    """Accept ``None``, ``True``, or a scalar binary of ``model`` (``on`` or ``on[t]``).

    Anything else would silently weaken the formulation: a continuous or
    general-integer indicator makes the leaf sum fractional or larger than 1,
    and a constant (``0``, ``np.False_``) fixes the unit off.
    """
    from discopt.modeling.core import IndexExpression, Variable, VarType

    if active is None or active is True:
        return
    if isinstance(active, (bool, np.bool_)):
        raise ValueError(f"active={active!r} is not meaningful; pass None for an always-on tree")
    base = active.base if isinstance(active, IndexExpression) else active
    if not isinstance(base, Variable) or base.var_type is not VarType.BINARY:
        raise TypeError(
            "active must be None, True, or a binary variable (or an element of one); "
            f"got {type(active).__name__}"
        )
    if base.model is not model:
        raise ValueError("active belongs to a different model than the formulation")
    if active is base and base.size != 1:
        raise ValueError(f"active must be a scalar binary; '{base.name}' has shape {base.shape}")
    if isinstance(active, IndexExpression) and getattr(active, "_shape", None) != ():
        raise ValueError("active must index a single element of the binary variable")


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
    active : None, True, or a binary expression
        ``None`` (default): exactly one leaf is always active. ``True``: create
        a binary ``{prefix}_active`` that switches the unit on and off. A
        binary variable or element of one (e.g. ``on[t]``): use it as the
        on/off indicator. See "On/off units" in the module docstring.
    """

    def __init__(
        self,
        model: Model,
        definition: LinearTreeDefinition,
        prefix: str = "ltree",
        encoding: str = "hull",
        split_eps: float = 1e-5,
        active=None,
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
        _check_active(active, model)
        self._active_arg = active
        self._active: Any = None
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
    def active(self):
        """The on/off indicator, or ``None`` for an always-on tree."""
        if not self._formulated:
            raise RuntimeError("Call formulate() before accessing active")
        return self._active

    @property
    def leaf_indicators(self) -> Variable:
        """Leaf binaries ``z``, shape ``(n_leaves,)``; they sum to 1 (or to ``active``)."""
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
        if self._active_arg is True:
            self._active = self._model.binary(f"{self._prefix}_active")
        elif self._active_arg is not None:
            self._active = self._active_arg
        if self._encoding == "hull":
            self._build_hull()
        else:
            self._build_bigm()
        self._formulated = True

    # ------------------------------------------------------------------

    def _boxes(self) -> tuple[_Box, _Box]:
        """``(box, ext)``: the box and output bounds, and their extension to the origin.

        Without ``active`` ``ext`` is ``box``, so the always-on formulation is
        unchanged.
        """
        lb, ub = self._def.input_bounds
        y_lo, y_hi = self.output_bounds()
        box = _Box(lb, ub, y_lo, y_hi)
        if self._active is None:
            return box, box
        ext = _Box(
            np.minimum(lb, 0.0),
            np.maximum(ub, 0.0),
            np.minimum(y_lo, 0.0),
            np.maximum(y_hi, 0.0),
        )
        return box, ext

    def _one_leaf(self, z) -> None:
        rhs = 1 if self._active is None else self._active
        self._model.subject_to(dm.sum(z) == rhs, name=f"{self._prefix}_one_leaf")

    def _build_hull(self) -> None:
        m, d, pfx = self._model, self._def, self._prefix
        box, ext = self._boxes()
        lb, ub = box.x_lo, box.x_hi
        L, p, q = d.n_leaves, d.n_features, d.n_outputs

        # With ``active`` the variable bounds include 0; the leaf polytopes
        # (which contain the box rows) still confine an active unit to the box.
        x = m.continuous(f"{pfx}_input", shape=(p,), lb=ext.x_lo, ub=ext.x_hi)
        y = m.continuous(f"{pfx}_output", shape=(q,), lb=ext.y_lo, ub=ext.y_hi)
        z = m.binary(f"{pfx}_leaf", shape=(L,))
        # x_l lies in [min(lb,0), max(ub,0)] because it is either x or 0.
        xl = m.continuous(
            f"{pfx}_xdis",
            shape=(L, p),
            lb=np.repeat(np.minimum(lb, 0.0)[None], L, axis=0),
            ub=np.repeat(np.maximum(ub, 0.0)[None], L, axis=0),
        )

        self._one_leaf(z)
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
        box, ext = self._boxes()
        lb, ub, y_lo, y_hi = box
        L, p, q = d.n_leaves, d.n_features, d.n_outputs

        x = m.continuous(f"{pfx}_input", shape=(p,), lb=ext.x_lo, ub=ext.x_hi)
        y = m.continuous(f"{pfx}_output", shape=(q,), lb=ext.y_lo, ub=ext.y_hi)
        z = m.binary(f"{pfx}_leaf", shape=(L,))
        self._one_leaf(z)
        a = self._active
        if a is not None:
            # On: lb <= x <= ub and y_lo <= y <= y_hi. Off: x = 0 and y = 0.
            # Valid for any signs because lb <= ub and y_lo <= y_hi.
            for j in range(p):
                m.subject_to(x[j] >= float(lb[j]) * a, name=f"{pfx}_xon_lo_{j}")
                m.subject_to(x[j] <= float(ub[j]) * a, name=f"{pfx}_xon_hi_{j}")
            for k in range(q):
                m.subject_to(y[k] >= float(y_lo[k]) * a, name=f"{pfx}_yon_lo_{k}")
                m.subject_to(y[k] <= float(y_hi[k]) * a, name=f"{pfx}_yon_hi_{k}")

        for l_idx, leaf in enumerate(d.leaves):
            G, h = _shifted_rows(leaf, self._split_eps)
            # Split rows only: the box rows hold for every x already (or are
            # imposed through ``active`` above). Constants are taken over the
            # box extended to the origin, which is the box itself without
            # ``active``.
            ns = leaf.n_split_rows
            if ns:
                M = np.array(
                    [max(_box_range(G[r], ext.x_lo, ext.x_hi)[1] - h[r], 0.0) for r in range(ns)]
                )
                m.subject_to(
                    G[:ns] @ x <= h[:ns] + M * (1 - z[l_idx]),
                    name=f"{pfx}_split_{l_idx}",
                )
            for k in range(q):
                f_lo, f_hi = _box_range(leaf.A[k], ext.x_lo, ext.x_hi)
                f_lo += float(leaf.b[k])
                f_hi += float(leaf.b[k])
                # y - f <= y_hi - f_lo and f - y <= f_hi - y_lo when deselected.
                M_up = max(float(ext.y_hi[k]) - f_lo, 0.0)
                M_dn = max(f_hi - float(ext.y_lo[k]), 0.0)
                f = leaf.A[k] @ x + float(leaf.b[k])
                m.subject_to(y[k] - f <= M_up * (1 - z[l_idx]), name=f"{pfx}_oup_{l_idx}_{k}")
                m.subject_to(f - y[k] <= M_dn * (1 - z[l_idx]), name=f"{pfx}_odn_{l_idx}_{k}")
        self._inputs, self._outputs, self._leaf = x, y, z
