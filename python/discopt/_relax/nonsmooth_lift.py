"""Exact epigraph lifting of ``abs`` / ``max`` / ``min`` atoms (#1501).

A gradient-based NLP carries one subgradient at the kink of ``|u|``,
``max(a, b)`` or ``min(a, b)``, so neither a single local NLP solve nor an OA
tangent taken there is a certificate (#1297, #1501). Wherever such an atom sits
in a *monotone* position, it can be replaced exactly by a fresh variable ``t``:

* ``max(a, b)`` (and ``|u| = max(u, -u)``) where the enclosing row or objective
  only ever wants the atom SMALL -- a minimised objective, or the body of a
  ``<=`` row, reached from the root through ``+``, ``-``, negation, sums and
  multiplication / division by constants of known sign -- becomes ``t`` with
  ``t >= a`` and ``t >= b``;
* ``min(a, b)`` where the context wants it LARGE becomes ``t`` with ``t <= a``
  and ``t <= b``.

**Why this is exact.** Let ``g(x, s)`` be the row body (or objective) with the
atom's value replaced by ``s``, nondecreasing in ``s`` by the polarity walk.
Then ``{x : exists t >= max(a,b)(x) with g(x, t) <= 0}`` is
``{x : g(x, max(a,b)(x)) <= 0}`` (take ``t = max``; any larger ``t`` only makes
``g`` larger), and ``min_{t >= max} F(x, t) = F(x, max)``. Neither argument
uses convexity: the lifting is exact for any ``a``, ``b``. What convexity buys
is only that the lifted model is *smooth* and convex whenever the original was
convex, so the MIP-NLP family can certify it. An atom in a non-monotone
position (inside ``exp``, a product with a variable, an equality row, ``|.|``
of an ``|.|``, or the wrong polarity -- the concave ``-|u|``) is left alone: the
model then still contains a nonsmooth node and every certificate guard that
keys on :func:`discopt.solver._model_contains_nonsmooth_node` still applies.

``t`` gets the interval enclosure of the atom as its box (sound: ``t = atom(x)``
always lies in it, and the monotone context never needs ``t`` outside it).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    _known_shape,
    carry_builder_objective,
    rewritten_objective,
)

__all__ = ["lift_nonsmooth_atoms", "complete_lifted_point"]


def _const_sign(node: Expression) -> int:
    """``+1`` / ``-1`` if *node* is a :class:`Constant` of uniform strict sign, else 0."""
    if type(node) is not Constant:
        return 0
    v = np.asarray(node.value, dtype=np.float64)
    if v.size == 0 or not np.all(np.isfinite(v)):
        return 0
    if np.all(v > 0):
        return 1
    if np.all(v < 0):
        return -1
    return 0


def _atom(node: Expression) -> Optional[tuple[str, tuple[Expression, ...]]]:
    """``("max"|"min", args)`` for a liftable atom, else ``None``."""
    if isinstance(node, UnaryOp) and node.op == "abs":
        u = node.operand
        return "max", (u, UnaryOp("neg", u))
    if isinstance(node, FunctionCall):
        if node.func_name == "abs" and len(node.args) == 1:
            u = node.args[0]
            return "max", (u, UnaryOp("neg", u))
        if node.func_name in ("max", "min") and len(node.args) == 2:
            return node.func_name, tuple(node.args)
    return None


class _Lifter:
    def __init__(self, source: Model, target: Model) -> None:
        self.source = source
        self.target = target
        self.aux: list[tuple[Variable, Expression]] = []
        self.rows: list[Constraint] = []
        self._k = 0

    def _new_var(self, atom: Expression, shape: tuple[int, ...]) -> Variable:
        from discopt._relax.convexity.interval_eval import evaluate_interval

        # ``evaluate_interval`` returns ``[-inf, inf]`` for an atom it cannot
        # enclose rather than raising, so there is nothing to catch here.
        enc = evaluate_interval(atom, self.source)
        elo = np.broadcast_to(np.asarray(enc.lo, dtype=np.float64), shape)
        ehi = np.broadcast_to(np.asarray(enc.hi, dtype=np.float64), shape)
        # The enclosure is used as is, infinite ends included: clamping it to the
        # default box would cut the points where the atom exceeds 9.999e19 (an
        # ``|x + y|`` over two default-boxed columns reaches 2e20).
        lo, hi = elo, ehi
        taken = {v.name for v in self.target._variables}
        name = f"_nsl_t{self._k}"
        while name in taken:
            self._k += 1
            name = f"_nsl_t{self._k}"
        self._k += 1
        if shape == ():
            lo_s, hi_s = float(lo), float(hi)
            t = self.target.continuous(name, lb=lo_s, ub=hi_s)
        else:
            t = self.target.continuous(name, shape=shape, lb=lo, ub=hi)
        assert isinstance(t, Variable)
        self.aux.append((t, atom))
        return t

    def rewrite(self, node: Expression, e: int) -> Expression:
        """*node* with every liftable atom under polarity *e* replaced.

        ``e = +1``: the context wants *node* small; ``-1``: large; ``0``: unknown
        (no lifting below this point). Returns *node* itself when unchanged.
        """
        if e == 0:
            return node
        at = _atom(node)
        if at is not None:
            kind, args = at
            shape = _known_shape(node)
            if shape is None or (kind == "max" and e != 1) or (kind == "min" and e != -1):
                return node
            t = self._new_var(node, shape)
            for a in args:
                if kind == "max":
                    # a <= t: in ``a - t <= 0`` the argument keeps polarity +1.
                    body = BinaryOp("-", self.rewrite(a, 1), t)
                else:
                    # t <= a: in ``t - a <= 0`` the argument has polarity -1.
                    body = BinaryOp("-", t, self.rewrite(a, -1))
                self.rows.append(Constraint(body, "<=", 0.0))
            return t
        if isinstance(node, BinaryOp):
            op = node.op
            if op in ("+", "-"):
                left = self.rewrite(node.left, e)
                right = self.rewrite(node.right, e if op == "+" else -e)
            elif op == "*":
                sl, sr = _const_sign(node.left), _const_sign(node.right)
                left = self.rewrite(node.left, e * sr)
                right = self.rewrite(node.right, e * sl)
            elif op == "/":
                left = self.rewrite(node.left, e * _const_sign(node.right))
                right = node.right
            else:
                return node
            if left is node.left and right is node.right:
                return node
            return BinaryOp(op, left, right)
        if isinstance(node, UnaryOp) and node.op == "neg":
            inner = self.rewrite(node.operand, -e)
            return node if inner is node.operand else UnaryOp("neg", inner)
        if type(node) is SumExpression:
            inner = self.rewrite(node.operand, e)
            return node if inner is node.operand else SumExpression(inner, axis=node.axis)
        if type(node) is SumOverExpression:
            terms = [self.rewrite(t, e) for t in node.terms]
            if all(a is b for a, b in zip(terms, node.terms)):
                return node
            return SumOverExpression(terms)
        return node


def lift_nonsmooth_atoms(model: Model) -> Model:
    """Return *model* with every monotone-position ``abs``/``max``/``min`` lifted.

    Functional: *model* is not modified. Returns *model* itself when there is
    nothing to lift. The lifted model shares *model*'s variables (the auxiliary
    columns are appended after them, so the original flat offsets are
    unchanged) and records ``_nsl_source_model``, ``_nsl_n_orig_flat`` and
    ``_nsl_aux`` (``[(aux_variable, atom_expression)]``) so a caller can map a
    point back (:func:`complete_lifted_point` goes the other way).
    """
    from discopt.modeling.core import ObjectiveSense
    from discopt.mpec import carry_complementarities
    from discopt.solver import _model_contains_nonsmooth_node

    if not _model_contains_nonsmooth_node(model):
        return model
    new_model = Model(model.name)
    new_model._variables = list(model._variables)
    new_model._parameters = list(model._parameters)
    new_model._rebuild_name_index()
    lifter = _Lifter(model, new_model)

    rebuilt: list = []
    for c in model._constraints:
        if not isinstance(c, Constraint) or c.sense == "==":
            rebuilt.append(c)
            continue
        if c.sense not in ("<=", ">="):
            rebuilt.append(c)
            continue
        body = lifter.rewrite(c.body, 1 if c.sense == "<=" else -1)
        rebuilt.append(c if body is c.body else Constraint(body, c.sense, c.rhs, c.name))

    new_model._objective = model._objective
    obj = model._objective
    if obj is not None:
        e = 1 if obj.sense == ObjectiveSense.MINIMIZE else -1
        expr = lifter.rewrite(obj.expression, e)
        if expr is not obj.expression:
            new_model._objective = rewritten_objective(obj, expr)

    if not lifter.aux:
        return model
    new_model._constraints = rebuilt + lifter.rows
    carry_complementarities(model, new_model, pass_name="nonsmooth epigraph lifting")
    carry_builder_objective(model, new_model)  # fast-API objective; after the last aux column
    new_model._nsl_source_model = model  # type: ignore[attr-defined]
    new_model._nsl_n_orig_flat = sum(v.size for v in model._variables)  # type: ignore[attr-defined]
    new_model._nsl_aux = list(lifter.aux)  # type: ignore[attr-defined]
    return new_model


def complete_lifted_point(lifted: Model, x_orig: np.ndarray) -> np.ndarray:
    """Extend a flat point of the source model with each aux column's atom value.

    At ``t = atom(x)`` every lifting row is satisfied, so the completed point is
    feasible for the lifted model exactly when *x_orig* is for the source.
    """
    # The POUNCE tape, not the JAX DAG compiler (#1873): this needs values only.
    from discopt._point_eval import evaluate_at

    source: Model = lifted._nsl_source_model  # type: ignore[attr-defined]
    x_orig = np.asarray(x_orig, dtype=np.float64).reshape(-1)
    n_orig = int(lifted._nsl_n_orig_flat)  # type: ignore[attr-defined]
    if x_orig.size != n_orig:
        raise ValueError(f"point has {x_orig.size} entries; the source model has {n_orig} columns")
    parts = [x_orig]
    for t, atom in lifted._nsl_aux:  # type: ignore[attr-defined]
        val = evaluate_at(atom, source, x_orig)
        val = np.broadcast_to(val, t.shape if t.shape else ()).reshape(-1)
        lo = np.broadcast_to(np.asarray(t.lb, dtype=np.float64), val.shape)
        hi = np.broadcast_to(np.asarray(t.ub, dtype=np.float64), val.shape)
        if not np.all(np.isfinite(val)):
            val = np.where(np.isfinite(val), val, np.clip(0.0, lo, hi))
        parts.append(np.clip(val, lo, hi))
    return np.concatenate(parts)
