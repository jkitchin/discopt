"""Split multilinear products into chains of bilinear auxiliaries (#1990).

The native spatial kernel (``bnb/spatial_tree.rs``) relaxes bilinear, monomial,
affine-square and ``sqrt`` terms, and ``spatial_producer.build_spatial_kernel_spec``
declines any model whose relaxation carries a trilinear or multilinear term. Such a
model runs on the Python spatial tree, which on ex1252a
(``(c + 1800*x15)*x0*x3*x18`` per line) managed ~280 nodes in 600 s.

This pass writes each product of three or more distinct scalar variables
``f_1 * f_2 * ... * f_n`` as

    w_1 = f_1 * f_2,   w_2 = w_1 * f_3,   ...,   w_{n-1} = w_{n-2} * f_n

and replaces the product by ``w_{n-1}``. Every ``w_k`` is a new continuous column
bounded by the interval product of its two factors over the given box, so the
rewrite is an exact reformulation: a point of the original model extends to
exactly one point of the split model, with the same objective. Only the
relaxation changes -- the recursive bilinear McCormick envelope instead of the
trilinear/multilinear one -- and the kernel can then take the model.

Discrete factors are multiplied first (binary and integer, then continuous, each
group in the order written): an integer-times-integer-times-binary prefix is
exactly representable on every integer-fixed box, so the continuous factor meets a
column the tree pins by branching rather than a relaxed product. A product prefix
shared between several products is lifted once.

The pass is used for the kernel attempt only (``solver.py``,
``DISCOPT_KERNEL_MULTILINEAR_SPLIT``); the Python tree keeps the original model and
its trilinear hull.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    Model,
    Variable,
    VarType,
    carry_validation_guards,
)
from discopt.mpec import carry_complementarities

from .factorable_reform import _collect_mul_factors
from .integer_product_reform import _elem_ref, _scalar_var_ref

#: A bound at or beyond this magnitude is "no bound": discopt stores an unbounded
#: side as the finite sentinel 1e20 (CLAUDE.md, Key Constraints), so a finiteness
#: test alone would let an unbounded factor through and give its aux a 1e20-scale
#: interval.
_NO_BOUND = 1e19


class _Splitter:
    def __init__(self, model: Model, lb: np.ndarray, ub: np.ndarray, offsets: dict[int, int]):
        self.model = model
        self.lb = lb
        self.ub = ub
        self.offsets = offsets
        self.n_cols = int(lb.shape[0])
        self.aux_constraints: list[Constraint] = []
        self.aux_lb: list[float] = []
        self.aux_ub: list[float] = []
        # Per aux column, in append order: ``(left, right)`` as flat column indices
        # of the split model, so ``aux = x[left] * x[right]`` rebuilds it from a
        # point (aux columns refer only to earlier columns).
        self.aux_spec: list[tuple[int, int]] = []
        # Key: the factor sequence of a chain prefix, as ``(var._index, elem)``
        # tuples; value: the aux column and its interval.
        self._cache: dict[tuple, tuple[Expression, float, float, int]] = {}
        self.n_products = 0

    def _interval(self, var: Variable, elem: int) -> tuple[float, float]:
        flat = self.offsets[var._index] + elem
        return float(self.lb[flat]), float(self.ub[flat])

    def chain(self, refs: list[tuple[Variable, int]]) -> Optional[Expression]:
        """The aux column equal to the product of ``refs``, or ``None`` when an
        interval is not finite (the kernel declines an unbounded aux anyway)."""

        def rank(r: tuple[Variable, int]) -> tuple[int, int]:
            vt = r[0].var_type
            group = 0 if vt == VarType.BINARY else 1 if vt == VarType.INTEGER else 2
            return group, self.offsets[r[0]._index] + r[1]

        refs = sorted(refs, key=rank)  # canonical, so equal prefixes are shared
        for var, elem in refs:
            flo, fhi = self._interval(var, elem)
            if not (abs(flo) < _NO_BOUND and abs(fhi) < _NO_BOUND):
                return None
        key: tuple = ((refs[0][0]._index, refs[0][1]),)
        cur: Expression = _elem_ref(*refs[0])
        cur_flat = self.offsets[refs[0][0]._index] + refs[0][1]
        lo, hi = self._interval(*refs[0])
        for var, elem in refs[1:]:
            key = key + ((var._index, elem),)
            flo, fhi = self._interval(var, elem)
            hit = self._cache.get(key)
            if hit is not None:
                cur, lo, hi, cur_flat = hit
                continue
            corners = [lo * flo, lo * fhi, hi * flo, hi * fhi]
            if not all(math.isfinite(c) for c in corners):
                return None
            lo, hi = min(corners), max(corners)
            w = Variable(
                self.model._fresh_name(f"_mls_w{len(self.aux_lb)}"),
                VarType.CONTINUOUS,
                (),
                lo,
                hi,
                self.model,
            )
            self.model._register_generated_variable(w)
            w_flat = self.n_cols + len(self.aux_lb)
            self.aux_spec.append((cur_flat, self.offsets[var._index] + elem))
            self.aux_lb.append(lo)
            self.aux_ub.append(hi)
            self.aux_constraints.append(
                Constraint(BinaryOp("-", w, BinaryOp("*", cur, _elem_ref(var, elem))), "==", 0.0)
            )
            self._cache[key] = (w, lo, hi, w_flat)
            cur, cur_flat = w, w_flat
        return cur

    def rewrite(self, expr: Expression) -> Optional[Expression]:
        """``expr`` with every splittable product replaced; ``None`` aborts the pass."""
        if isinstance(expr, BinaryOp):
            if expr.op == "*":
                factors = _collect_mul_factors(expr)
                coeff = 1.0
                refs: list[tuple[Variable, int]] = []
                for f in factors:
                    if isinstance(f, Constant) and np.ndim(f.value) == 0:
                        coeff *= float(f.value)
                        continue
                    r = _scalar_var_ref(f)
                    if r is None:
                        refs = []
                        break
                    refs.append(r)
                distinct = len({(v._index, e) for v, e in refs}) == len(refs)
                if len(refs) >= 3 and distinct:
                    w = self.chain(refs)
                    if w is None:
                        return None
                    self.n_products += 1
                    return w if coeff == 1.0 else BinaryOp("*", Constant(coeff), w)
            left = self.rewrite(expr.left)
            right = self.rewrite(expr.right)
            if left is None or right is None:
                return None
            if left is expr.left and right is expr.right:
                return expr
            return BinaryOp(expr.op, left, right)
        for attr in ("operand",):
            child = getattr(expr, attr, None)
            if isinstance(child, Expression):
                new = self.rewrite(child)
                if new is None:
                    return None
                if new is not child:
                    clone = copy.copy(expr)
                    setattr(clone, attr, new)
                    return clone
        for attr in ("args", "terms"):
            seq = getattr(expr, attr, None)
            if isinstance(seq, (list, tuple)):
                new_seq = []
                for c in seq:
                    n = self.rewrite(c) if isinstance(c, Expression) else c
                    if n is None:
                        return None
                    new_seq.append(n)
                if any(n is not o for n, o in zip(new_seq, seq)):
                    clone = copy.copy(expr)
                    setattr(clone, attr, type(seq)(new_seq))
                    return clone
        return expr


def split_multilinear_products(
    model: Model, lb: np.ndarray, ub: np.ndarray
) -> Optional[tuple[Model, np.ndarray, np.ndarray]]:
    """The split model and its box ``(lb, ub)`` extended by the aux columns, or
    ``None`` when there is no product to split or a factor is unbounded.

    ``lb``/``ub`` are the flat box over ``model``'s columns (the box the kernel is
    handed). The split model shares ``model``'s ``Variable`` objects; the aux
    columns are appended after them, so its first ``len(lb)`` flat columns are
    exactly ``model``'s, and ``new_model._mls_aux_spec`` lists each aux column's
    ``(left, right)`` factor columns in append order (:func:`extend_point`).
    """
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    offsets: dict[int, int] = {}
    off = 0
    for v in model._variables:
        offsets[v._index] = off
        off += int(v.size)
    if off != lb.shape[0] or off != ub.shape[0]:
        return None
    new_model = Model(model.name)
    new_model._variables = list(model._variables)
    new_model._parameters = list(model._parameters)
    new_model._rebuild_name_index()
    sp = _Splitter(new_model, lb, ub, offsets)
    rebuilt: list = []
    for c in model._constraints:
        if not isinstance(c, Constraint):
            rebuilt.append(c)
            continue
        body = sp.rewrite(c.body)
        if body is None:
            return None
        rebuilt.append(c if body is c.body else Constraint(body, c.sense, c.rhs, c.name))
    new_model._objective = model._objective
    if model._objective is not None:
        expr = sp.rewrite(model._objective.expression)
        if expr is None:
            return None
        if expr is not model._objective.expression:
            obj = copy.copy(model._objective)
            obj.expression = expr
            new_model._objective = obj
    if sp.n_products == 0:
        return None
    new_model._constraints = rebuilt + sp.aux_constraints
    new_model._mls_aux_spec = tuple(sp.aux_spec)  # type: ignore[attr-defined]
    carry_complementarities(model, new_model, pass_name="multilinear split")
    carry_validation_guards(model, new_model)
    return (
        new_model,
        np.concatenate([lb, np.asarray(sp.aux_lb, dtype=np.float64)]),
        np.concatenate([ub, np.asarray(sp.aux_ub, dtype=np.float64)]),
    )


def extend_point(split_model: Model, x: np.ndarray) -> np.ndarray:
    """A point over the original columns extended by the split's aux columns."""
    out = list(np.asarray(x, dtype=np.float64))
    for left, right in split_model._mls_aux_spec:  # type: ignore[attr-defined]
        out.append(out[left] * out[right])
    return np.asarray(out, dtype=np.float64)
