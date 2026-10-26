"""Fathom the unbounded outer piece of a model by a 1/T rescaling (#1975).

A model whose feasible set keeps a column open after FBBT (``x_k in [r_k, inf)``
or ``(-inf, r_k]``) can have a root relaxation that is unbounded below along a
recession ray of those columns even though the model itself is bounded: the
McCormick LP lifts each product ``x_i x_k`` against an infinite factor, and the
coupling the nonlinear model keeps between those products is lost (prolog:
``c^T d = -1.7e-13`` exactly, performance-plan §90). Branching never closes such a
node -- an outer child ``x_k >= p`` keeps the same ray for every ``p``.

This module covers the model with two pieces. With ``T = sum_k sigma_k (x_k - r_k)``
over the open columns (``sigma_k = +1`` for a column open above, ``-1`` below):

* the **inner piece** ``T <= M`` is the model with every open column boxed to
  ``[r_k, r_k + M]`` (mirrored for ``sigma = -1``) plus the row ``T <= M``. It is a
  bounded model and is solved by the ordinary pipeline;
* the **outer piece** ``T >= M`` is rewritten, exactly, through
  ``x_k = r_k + sigma_k u_k / s`` with ``s = 1/T in (0, 1/M]`` and
  ``u >= 0, sum u = 1``. A row of degree ``D`` in the open columns is multiplied by
  ``s^D >= 0`` (direction kept), and the objective of degree ``d`` becomes
  ``(M s)^d f``. Every resulting expression is a polynomial in ``s`` and ``u`` with
  the bounded block unchanged, so the piece is a BOUNDED model in which no product
  is lifted against an infinite factor. Its closure ``s in [0, 1/M]`` is a
  relaxation of the piece (the image of every feasible point satisfies every
  rewritten row), so its certified minimum ``c`` is a lower bound of
  ``(M/T)^d f`` there. Hence on ``T >= M``: ``f >= c (T/M)^d >= c`` when ``c >= 0``
  or ``d = 0``; for ``c < 0`` and ``d >= 1`` the piece has no finite bound.

The model's bound is ``min(inner bound, outer bound)`` and the incumbent is the
inner piece's (re-verified against the original model). For ``c > 0`` the same
``c`` bounds every ``M' >= M`` by ``c (M'/M)^d``, so when the incumbent sits above
the outer bound, ``M`` is enlarged once without re-solving the outer piece.

The rewrite accepts sums, products, divisions by an expression free of open
columns, positive constant powers of any degree (``(s^D a)^p = s^{Dp} a^p``, padded
by ``s^(ceil(Dp) - Dp)``; :meth:`_Homogenizer._power`), ``sqrt``, ``abs``
(``|s^D a| = s^D |a|``), and any function or other node of an argument free of
open columns. Anything else (``log`` of an open column, a division by one, a
negative power of one) raises :class:`OuterPieceUnsupported` and the model is
solved exactly as with the flag OFF. A free column (no finite side) is measured
from 0 and enters ``T`` as ``|x|`` (:func:`build_outer`).

Gated by ``DISCOPT_OUTER_PIECE`` (default OFF; bound-changing, CLAUDE.md §5).
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Objective,
    ObjectiveSense,
    Parameter,
    SolveResult,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

logger = logging.getLogger(__name__)

FLAG = "DISCOPT_OUTER_PIECE"

#: The piece sizes tried for the outer piece, in order; the first whose certified
#: bound is usable (finite, and ``>= 0`` when the objective grows with ``T``) is
#: kept. #1975's entry-experiment grid, middle first: a small ``M`` leaves
#: near-optimal points in the outer piece (prolog: ``f ~ 0`` at ``T ~ 230``, plan
#: §91), so its bound cannot be ``>= 0`` and the attempt only spends budget; a
#: larger ``M`` also widens the bound's margin against the absolute gap tolerance
#: (``c' = M^d c``), at the price of a wider inner box.
M_GRID = (1e4, 1e6, 1e2)

#: Share of ``time_limit`` the outer-piece solves may take together (after the
#: probe), and how it is split over :data:`M_GRID`: front-loaded, as the grid is
#: ordered by how likely each size is to give a usable bound. prolog's piece at
#: ``M = 1e4`` needs ~5 s; an even three-way split of 0.4 x 30 s gave it 3 s.
OUTER_BUDGET_FRACTION = 0.5
M_GRID_SHARE = (0.6, 0.25, 0.15)

#: Share of ``time_limit`` the root probe may take. The pieces are built only when
#: the model as written gets NO dual bound from a root-only solve: firing on every
#: model with an open column cost the models that certify anyway their wall
#: (autocorr_bern20-03 0.4 -> 2.4 s, clay0303hfsg 4.9 -> 13.0 s) and weakened the
#: bound where the inner piece got less time than the whole solve had
#: (autocorr_bern25-06 -3648 -> -4792); first 67 rows of the always-fire panel, plan §92.
PROBE_FRACTION = 0.1

#: A declared or FBBT bound at or beyond this magnitude is open (``Variable`` stores
#: an infinite bound as ``9.999e19``).
_OPEN = 1e19

_REFUSED_FIELDS = (
    "_piecewise_domains",
    "_complementarities",
    "_lowered_complementarities",
    "_decomp_stages",
    "_decomp_blocks",
    "_coupling_keys",
    "_block_labels_var",
    "_block_labels_con",
)


class OuterPieceUnsupported(ValueError):
    """The model has structure the rewrite does not cover; it is solved as usual."""


def outer_piece_enabled() -> bool:
    """``DISCOPT_OUTER_PIECE`` gate (#1975). Default OFF: bound-changing (the outer
    piece's certified bound replaces the original relaxation's on ``T >= M``), so it
    stays OFF until the CLAUDE.md §5 ON/OFF corpus panel passes. Audit row in
    ``docs/dev/flag-retirement-audit.md``; measurement in performance-plan §92."""
    return os.environ.get(FLAG, "0") == "1"


# ── which columns are open ───────────────────────────────────────────────────


@dataclass(frozen=True)
class OpenColumn:
    """One open scalar component: ``x = ref + sign * (T-share)``."""

    var_index: int  # position in ``model._variables``
    flat: int  # flat element of that variable
    ref: float  # the finite side (0 for a free column)
    sign: float  # +1: open above, -1: open below, 0: free (open both ways)
    in_T: bool = True  # a generator: a term of T (see :func:`generators`)


def _flat_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    from discopt.tightening import fbbt_box

    box = fbbt_box(model)
    lb = np.asarray(box.lb, dtype=np.float64).ravel()
    ub = np.asarray(box.ub, dtype=np.float64).ravel()
    n = sum(int(np.prod(v.shape)) if v.shape else 1 for v in model._variables)
    if lb.size != n or ub.size != n:
        raise OuterPieceUnsupported(f"FBBT box has {lb.size} entries for {n} columns")
    # FBBT does not loosen a declared bound; intersect anyway, so the box used here
    # is never wider than what the user declared.
    dlb = np.concatenate([np.ravel(np.asarray(v.lb, dtype=np.float64)) for v in model._variables])
    dub = np.concatenate([np.ravel(np.asarray(v.ub, dtype=np.float64)) for v in model._variables])
    return np.maximum(lb, dlb), np.minimum(ub, dub)


def open_columns(model: Model, lb: np.ndarray, ub: np.ndarray) -> list[OpenColumn]:
    """Every scalar component with an infinite side in ``[lb, ub]``.

    A free component (both sides open) is measured from 0 and enters ``T`` as
    ``|x|``; see :func:`build_outer`."""
    cols: list[OpenColumn] = []
    off = 0
    for vi, v in enumerate(model._variables):
        size = int(np.prod(v.shape)) if v.shape else 1
        for e in range(size):
            lo, hi = lb[off + e], ub[off + e]
            lo_open = not (np.isfinite(lo) and abs(lo) < _OPEN)
            hi_open = not (np.isfinite(hi) and abs(hi) < _OPEN)
            if lo_open and hi_open:
                cols.append(OpenColumn(vi, e, 0.0, 0.0))
            elif hi_open:
                cols.append(OpenColumn(vi, e, float(lo), 1.0))
            elif lo_open:
                cols.append(OpenColumn(vi, e, float(hi), -1.0))
        off += size
    return cols


# ── the rewrite  e(x) -> s^D e(r + sigma u / s) ─────────────────────────────


def _scalar_const(node: Expression) -> Optional[float]:
    if isinstance(node, Constant):
        val = np.asarray(node.value)
        if val.ndim == 0:
            return float(val)
    return None


def _spow(s: Variable, k: int) -> Optional[Expression]:
    """``s**k`` as an expression; ``None`` for ``k == 0`` (the factor 1)."""
    if k == 0:
        return None
    return s if k == 1 else BinaryOp("**", s, Constant(float(k)))


def _times(a: Optional[Expression], b: Expression) -> Expression:
    return b if a is None else BinaryOp("*", a, b)


class _Homogenizer:
    """``H(e) = (E, D)`` with ``E = s^D e(r + sigma u / s)``, built without division.

    ``D`` is an upper bound of ``e``'s degree in the open columns; any ``D`` at least
    that keeps ``E`` a polynomial in ``s``. Sub-DAGs free of open columns are rebuilt
    on the new model's variables and returned with ``D = 0``.
    """

    def __init__(self, var_map: dict[int, Variable], open_leaf: dict[tuple, Expression]):
        self.var_map = var_map  # id(old var) -> new var (bounded block)
        self.open_leaf = open_leaf  # (id(old var), flat or None) -> s * x_k, degree 1
        self.open_vars = {k[0] for k in open_leaf}
        self.s: Variable = None  # type: ignore[assignment]  # set by the builder
        self.memo: dict[int, tuple[Expression, int]] = {}

    def __call__(self, expr: Expression) -> tuple[Expression, int]:
        key = id(expr)
        hit = self.memo.get(key)
        if hit is None:
            hit = self._h(expr)
            self.memo[key] = hit
        return hit

    def lift(self, expr: Expression, d: int, target: int) -> Expression:
        """``s^(target - d) * expr``."""
        return _times(_spow(self.s, target - d), expr)

    def _leaf_key(self, node: IndexExpression) -> Optional[tuple]:
        from discopt.modeling._recentre import _scalar_index_key

        k = _scalar_index_key(node)
        return None if k is None else (k[0], k[1])

    def _h(self, expr: Expression) -> tuple[Expression, int]:
        if isinstance(expr, Constant):
            return expr, 0
        if isinstance(expr, Parameter):
            return expr, 0
        if isinstance(expr, Variable):
            if id(expr) in self.open_vars:
                if expr.shape:
                    raise OuterPieceUnsupported(f"array {expr.name} with an open element")
                return self.open_leaf[(id(expr), None)], 1
            return self.var_map[id(expr)], 0
        if isinstance(expr, IndexExpression):
            if isinstance(expr.base, Variable):
                base = expr.base
                if id(base) in self.open_vars:
                    k = self._leaf_key(expr)
                    if k is None:
                        raise OuterPieceUnsupported(f"non-scalar index into {base.name}")
                    leaf = self.open_leaf.get((k[0], k[1]))
                    if leaf is not None:
                        return leaf, 1
                return IndexExpression(self.var_map[id(base)], expr.index), 0
            inner, d = self(expr.base)
            if d:
                raise OuterPieceUnsupported("index into an array expression of open columns")
            return IndexExpression(inner, expr.index), 0
        if isinstance(expr, UnaryOp):
            a, d = self(expr.operand)
            if expr.op in ("neg", "abs") or d == 0:
                # neg is linear; abs is positively homogeneous: |s^d a| = s^d |a|.
                return UnaryOp(expr.op, a), d
            raise OuterPieceUnsupported(f"{expr.op}() of open columns")
        if isinstance(expr, BinaryOp) and expr.op in ("+", "-"):
            return self._sum_chain(expr)
        if isinstance(expr, BinaryOp):
            return self._binary(expr)
        if isinstance(expr, SumOverExpression):
            parts = [self(t) for t in expr.terms]
            dmax = max(d for _, d in parts)
            return SumOverExpression([self.lift(e, d, dmax) for e, d in parts]), dmax
        if type(expr) is FunctionCall and len(expr.args) == 1:
            a, d = self(expr.args[0])
            if d and expr.func_name == "sqrt":
                return self._power(a, d, 0.5, Constant(0.5))
            if d and expr.func_name == "abs":
                return FunctionCall("abs", a), d
        if type(expr) is FunctionCall:
            args = [self(a) for a in expr.args]
            if any(d for _, d in args):
                raise OuterPieceUnsupported(f"{expr.func_name}() of open columns")
            return FunctionCall(expr.func_name, *(e for e, _ in args)), 0
        if isinstance(expr, MatMulExpression):
            (a, da), (b, db) = self(expr.left), self(expr.right)
            if da or db:
                raise OuterPieceUnsupported("matmul over open columns")
            return MatMulExpression(a, b), 0
        if isinstance(expr, SumExpression):
            a, d = self(expr.operand)
            if d:
                raise OuterPieceUnsupported("array sum over open columns")
            return SumExpression(a, axis=expr.axis), 0
        raise OuterPieceUnsupported(f"no rewrite rule for {type(expr).__name__}")

    def _sum_chain(self, expr: BinaryOp) -> tuple[Expression, int]:
        """A ``+``/``-`` chain, walked along its left spine without recursion: a
        model built by ``e = e + x[i]`` in a loop is as deep as the loop is long."""
        terms: list[tuple[float, Expression]] = []
        node: Expression = expr
        while (
            isinstance(node, BinaryOp)
            and node.op in ("+", "-")
            and (node is expr or id(node) not in self.memo)
        ):
            terms.append((1.0 if node.op == "+" else -1.0, node.right))
            node = node.left
        terms.append((1.0, node))
        parts = [(sg, *self(t)) for sg, t in reversed(terms)]
        dmax = max(d for _, _, d in parts)
        out = [self.lift(e if sg > 0 else UnaryOp("neg", e), d, dmax) for sg, e, d in parts]
        return SumOverExpression(out), dmax

    def _power(
        self, a: Expression, da: int, p: float, p_node: Expression
    ) -> tuple[Expression, int]:
        """``a^p`` for a constant ``p > 0`` and ``a`` of degree ``da >= 1``.

        ``(s^da a)^p = s^(da p) a^p`` (``s >= 0``; a fractional power's domain is
        ``a >= 0``, where ``s^da a >= 0`` too), so with ``D = ceil(da p)``
        ``s^D a^p = (s^da a)^p * s^(D - da p)``, the second factor a power of ``s``
        in ``[0, 1)``. Exact for an integer ``da p`` (every integer power, and
        ``sqrt``); otherwise the stored exponent ``D - fl(da p)`` is off by at most
        an ulp of ``da p``, a factor ``s^delta`` with ``|delta| ~ 1e-16``.
        """
        if not (p > 0.0 and math.isfinite(p)):
            raise OuterPieceUnsupported(f"power {p!r} of open columns")
        D = math.ceil(da * p)
        rest = D - da * p
        powered = BinaryOp("**", a, p_node)
        if rest == 0.0:
            return powered, D
        return BinaryOp("*", powered, BinaryOp("**", self.s, Constant(rest))), D

    def _binary(self, expr: BinaryOp) -> tuple[Expression, int]:
        op = expr.op
        a, da = self(expr.left)
        if op == "**":
            p = _scalar_const(expr.right)
            if da == 0:
                b, db = self(expr.right)
                if db:
                    raise OuterPieceUnsupported("an exponent with open columns")
                return BinaryOp("**", a, b), 0
            if p is None:
                raise OuterPieceUnsupported("a non-constant power of open columns")
            return self._power(a, da, p, expr.right)
        b, db = self(expr.right)
        if op == "*":
            return BinaryOp("*", a, b), da + db
        if op == "/":
            if db:
                raise OuterPieceUnsupported("a division by open columns")
            return BinaryOp("/", a, b), da
        raise OuterPieceUnsupported(f"binary op {op!r}")


# ── the two pieces ───────────────────────────────────────────────────────────


@dataclass
class Plan:
    model: Model
    cols: list[OpenColumn]
    lb: np.ndarray  # FBBT box (flat), original columns
    ub: np.ndarray
    sign: float  # +1 minimize, -1 maximize (the outer piece always minimizes sign*f)
    stats: dict[str, Any] = field(default_factory=dict)
    # Set by build_outer: each open column's share slots in ``_outer_u`` (the
    # second is u- of a free generator), and each row's degree D.
    slots: list[tuple[int, Optional[int]]] = field(default_factory=list)
    row_degrees: list[int] = field(default_factory=list)


def _check_model(model: Model) -> None:
    obj = model._objective
    if obj is None or getattr(obj, "_is_placeholder", False):
        raise OuterPieceUnsupported("no expression objective")
    if model._builder_linear_objective is not None or model._builder_quadratic_objective:
        raise OuterPieceUnsupported("a builder objective")
    if model._builder_linear_blocks:
        raise OuterPieceUnsupported("builder linear blocks")
    for con in model._constraints:
        if type(con) is not Constraint:
            raise OuterPieceUnsupported(f"a {type(con).__name__} row")
    for attr in _REFUSED_FIELDS:
        if getattr(model, attr, None):
            raise OuterPieceUnsupported(f"model carries {attr}")


def plan(model: Model) -> Optional[Plan]:
    """``None`` when no column is open (nothing to do); raises when unsupported."""
    _check_model(model)
    lb, ub = _flat_bounds(model)
    if np.any(lb > ub):
        raise OuterPieceUnsupported("FBBT box is empty")
    cols = open_columns(model, lb, ub)
    if not cols:
        return None
    cols = generators(model, cols)
    sign = -1.0 if model._objective.sense == ObjectiveSense.MAXIMIZE else 1.0
    return Plan(model=model, cols=cols, lb=lb, ub=ub, sign=sign)


#: Width of the trial box :func:`generators` puts on a generator. Only whether
#: FBBT reaches a finite bound matters, not the bound itself.
_TRIAL_WIDTH = 1e4


def _box_columns(model: Model, cols: list[OpenColumn], width: float) -> list[Expression]:
    """Box each column of ``cols`` to ``width`` from its finite side, in place, and
    return the terms ``sigma (x - ref)`` of ``T`` over them."""
    terms: list[Expression] = []
    for c in cols:
        v = model._variables[c.var_index]
        lb = np.array(v.lb, dtype=np.float64, copy=True)
        ub = np.array(v.ub, dtype=np.float64, copy=True)
        fl, fu = lb.reshape(-1), ub.reshape(-1)
        if c.sign >= 0:
            fu[c.flat] = min(fu[c.flat], c.ref + width)
        if c.sign <= 0:
            fl[c.flat] = max(fl[c.flat], c.ref - width)
        if v.var_type is VarType.INTEGER:
            fl[c.flat] = math.ceil(fl[c.flat] - 1e-9)
            fu[c.flat] = math.floor(fu[c.flat] + 1e-9)
        v.lb, v.ub = lb, ub
        leaf = v[np.unravel_index(c.flat, v.shape)] if v.shape else v
        if c.sign != 0:  # |x - ref| is not a linear term; see build_inner
            terms.append(leaf - c.ref if c.sign > 0 else c.ref - leaf)
    return terms


def generators(model: Model, cols: list[OpenColumn]) -> list[OpenColumn]:
    """Mark a minimal set of open columns whose boxing lets FBBT bound all others.

    ``T`` sums only these. A column whose growth is forced by others' (prolog's
    ``x18 <= 3340.8 x7 + 500 x8``) is left out of ``T``: summing it in measures the
    piece in a unit the objective does not grow with, which drives the outer
    piece's ``min f/T`` to the feasibility tolerance (#1975, plan §92). Greedy, in
    model order: a column is dropped when FBBT still bounds every open column with
    the remaining generators boxed. The result is minimal (no single further drop
    works), not necessarily minimum.
    """
    from discopt.tightening import fbbt_box

    base = model.clone()
    saved = [(np.array(v.lb, copy=True), np.array(v.ub, copy=True)) for v in base._variables]
    offs = _offsets(base)

    def all_bounded(gen: list[OpenColumn]) -> bool:
        for v, (lb, ub) in zip(base._variables, saved):
            v.lb, v.ub = lb.copy(), ub.copy()
        _box_columns(base, gen, _TRIAL_WIDTH)
        box = fbbt_box(base)
        lo = np.asarray(box.lb, dtype=np.float64).ravel()
        hi = np.asarray(box.ub, dtype=np.float64).ravel()
        if np.any(lo > hi):
            return False  # empty: proves nothing about boundedness
        for c in cols:
            j = offs[c.var_index] + c.flat
            if not (np.isfinite(lo[j]) and np.isfinite(hi[j])):
                return False
            if max(abs(lo[j]), abs(hi[j])) >= _OPEN:
                return False
        return True

    keep = list(range(len(cols)))
    for k in range(len(cols)):
        trial = [i for i in keep if i != k]
        if trial and all_bounded([cols[i] for i in trial]):
            keep = trial
    kept = set(keep)
    return [
        OpenColumn(c.var_index, c.flat, c.ref, c.sign, in_T=(i in kept)) for i, c in enumerate(cols)
    ]


def _offsets(model: Model) -> list[int]:
    offs, off = [], 0
    for v in model._variables:
        offs.append(off)
        off += int(np.prod(v.shape)) if v.shape else 1
    return offs


def build_outer(p: Plan, M: float) -> tuple[Model, int]:
    """The outer piece ``T >= M`` as a bounded model; returns it and the objective's
    degree ``d`` (its minimum ``c`` bounds ``sign * f >= c (T/M)^d`` on the piece)."""
    model = p.model
    offs = _offsets(model)
    new = Model(f"{model.name}__outer_piece")
    open_at = {(c.var_index, c.flat): c for c in p.cols}
    var_map: dict[int, Variable] = {}
    for vi, v in enumerate(model._variables):
        shape = v.shape if v.shape else ()
        size = int(np.prod(shape)) if shape else 1
        lo = p.lb[offs[vi] : offs[vi] + size].copy()
        hi = p.ub[offs[vi] : offs[vi] + size].copy()
        for e in range(size):
            if (vi, e) in open_at:  # unused placeholder: the leaf is rewritten to u/s
                lo[e] = hi[e] = 0.0
        lo_a = lo.reshape(shape) if shape else lo[0]
        hi_a = hi.reshape(shape) if shape else hi[0]
        if v.var_type is VarType.CONTINUOUS:
            z = new.continuous(v.name, shape=shape, lb=lo_a, ub=hi_a)
        elif v.var_type is VarType.BINARY:
            z = new.binary(v.name, shape=shape)
        else:
            z = new.integer(v.name, shape=shape, lb=np.ceil(lo_a - 1e-9), ub=np.floor(hi_a + 1e-9))
        var_map[id(v)] = z
    for prm in model._parameters:
        new._parameters.append(prm)
        new._names.add(prm.name)

    s = new.continuous("_outer_s", lb=0.0, ub=1.0 / M)
    # Shares. ``s * x_k = ref_k s + (share)``, with
    #   one-sided generator:      sign * u,         u in [0, 1]
    #   free generator:           u+ - u-,          u+, u- in [0, 1]
    #   one-sided non-generator:  sign * u,         u in [0, inf)
    #   free non-generator:       w,                w free
    # The generators' shares sum to 1 (``T = sum sign (x - ref) + sum |x|`` over
    # them, so ``u+ = max(x, 0)/T``, ``u- = max(-x, 0)/T`` is one admissible
    # split). A non-generator's share is bounded only through the rewritten rows,
    # which :func:`_require_bounded_shares` checks.
    lo_u: list[float] = []
    hi_u: list[float] = []
    slots: list[tuple[int, Optional[int]]] = []
    for c in p.cols:
        if c.in_T:
            slots.append((len(lo_u), len(lo_u) + 1 if c.sign == 0 else None))
            lo_u += [0.0, 0.0] if c.sign == 0 else [0.0]
            hi_u += [1.0, 1.0] if c.sign == 0 else [1.0]
        else:
            slots.append((len(lo_u), None))
            lo_u.append(-np.inf if c.sign == 0 else 0.0)
            hi_u.append(np.inf)
    u = new.continuous("_outer_u", shape=(len(lo_u),), lb=np.array(lo_u), ub=np.array(hi_u))
    open_leaf: dict[tuple, Expression] = {}
    simplex: list[Expression] = []
    for c, (kp, km) in zip(p.cols, slots):
        v = model._variables[c.var_index]
        share: Expression
        if km is not None:
            share = u[kp] - u[km]
            simplex += [u[kp], u[km]]
        else:
            share = -u[kp] if c.sign < 0 else u[kp]
            if c.in_T:
                simplex.append(u[kp])
        open_leaf[(id(v), c.flat if v.shape else None)] = (
            share if c.ref == 0.0 else Constant(c.ref) * s + share
        )
    h = _Homogenizer(var_map, open_leaf)
    h.s = s

    p.slots = slots
    p.row_degrees = []
    max_deg = 0
    for con in model._constraints:
        body = con.body if con.rhs == 0.0 else con.body - Constant(con.rhs)
        e, d = h(body)
        p.row_degrees.append(d)
        max_deg = max(max_deg, d)
        new._constraints.append(Constraint(body=e, sense=con.sense, rhs=0.0, name=con.name))
    new.subject_to(SumOverExpression(simplex) == 1.0, name="_outer_simplex")
    f, d = h(model._objective.expression)
    obj = f if p.sign > 0 else UnaryOp("neg", f)
    if d:
        # obj is already s^d f; M^d puts it in the objective's units at T = M.
        obj = BinaryOp("*", Constant(float(M) ** d), obj)
    new._objective = Objective(expression=obj, sense=ObjectiveSense.MINIMIZE)
    p.stats["outer_piece/max_row_degree"] = float(max_deg)
    _require_bounded_shares(new, u, len(lo_u))
    return new, d


def outer_point(p: Plan, outer: Model, x_flat: np.ndarray) -> np.ndarray:
    """The image in ``outer``'s columns of a point ``x`` of the original model with
    ``T(x) > 0``: ``s = 1/T``, each share ``(x - ref)/T`` (split ``u+ - u-`` for a
    free generator). For tests and verification; :func:`build_outer` must have run
    on ``p``."""
    offs = _offsets(p.model)
    T = 0.0
    for c in p.cols:
        if c.in_T:
            dx = float(x_flat[offs[c.var_index] + c.flat]) - c.ref
            T += abs(dx) if c.sign == 0 else c.sign * dx
    if not T > 0.0:
        raise ValueError(f"T(x) = {T} is not positive")
    n_u = max(max(kp, km if km is not None else kp) for kp, km in p.slots) + 1
    u = np.zeros(n_u)
    for c, (kp, km) in zip(p.cols, p.slots):
        dx = float(x_flat[offs[c.var_index] + c.flat]) - c.ref
        if km is not None:
            u[kp], u[km] = max(dx, 0.0) / T, max(-dx, 0.0) / T
        else:
            u[kp] = (dx if c.sign >= 0 else -dx) / T
    head = np.array(x_flat, dtype=np.float64, copy=True)
    for c in p.cols:
        head[offs[c.var_index] + c.flat] = 0.0  # the unused placeholder
    out = np.concatenate([head, [1.0 / T], u])
    n = sum(int(np.prod(v.shape)) if v.shape else 1 for v in outer._variables)
    if out.size != n:
        raise ValueError(f"mapped point has {out.size} entries for {n} columns")
    return out


def _require_bounded_shares(new: Model, u: Variable, n: int) -> None:
    """Every share ``u_k`` must be finite after FBBT, or the outer piece keeps the
    recession it was built to remove."""
    from discopt.tightening import fbbt_box

    box = fbbt_box(new)
    offs = _offsets(new)
    j0 = offs[[i for i, v in enumerate(new._variables) if v is u][0]]
    lo = np.asarray(box.lb, dtype=np.float64).ravel()[j0 : j0 + n]
    hi = np.asarray(box.ub, dtype=np.float64).ravel()[j0 : j0 + n]
    finite = np.isfinite(lo) & np.isfinite(hi) & (np.abs(lo) < _OPEN) & (np.abs(hi) < _OPEN)
    if not np.all(finite):
        raise OuterPieceUnsupported("an open column's share stays unbounded in the outer piece")


def build_inner(p: Plan, M: float) -> Model:
    """The model with every generator boxed by ``M`` and, when no generator is free,
    the row ``T <= M``.

    Inner and outer piece must cover the model, not partition it. A point outside
    the box has a generator with ``|x - ref| > M``, so ``T > M`` and the outer piece
    holds it; the row only shrinks the inner piece further where ``T`` is linear
    (``|x|`` of a free generator is not)."""
    inner = p.model.clone()
    gens = [c for c in p.cols if c.in_T]
    terms = _box_columns(inner, gens, M)
    if all(c.sign != 0 for c in gens):
        inner.subject_to(SumOverExpression(terms) <= M, name="_outer_piece_T")
    return inner


# ── the solve ────────────────────────────────────────────────────────────────


def _usable_outer_bound(r: SolveResult, d: int) -> Optional[float]:
    """The outer piece's bound on ``sign * f`` (``+inf``: piece infeasible), or None."""
    if r.status == "infeasible":
        return math.inf
    if r.status not in ("optimal", "feasible", "time_limit", "node_limit") or r.bound is None:
        return None
    c = float(r.bound)
    if not math.isfinite(c):
        return None
    if d == 0 or c >= 0.0:
        return c
    return None


def solve_with_outer_piece(
    model: Model, p: Plan, solve_args: dict[str, Any]
) -> Optional[SolveResult]:
    """Solve ``model`` as inner + outer piece. ``None``: no usable outer bound was
    found (the caller then solves ``model`` as usual with what is left)."""
    from discopt.modeling.core import _withhold_false_primal
    from discopt.solvers._gap import reported_gap
    from discopt.validation.feasibility import verify_point

    t0 = time.perf_counter()
    time_limit = float(solve_args.get("time_limit", 3600.0))
    stats = p.stats
    stats["outer_piece/open_columns"] = float(len(p.cols))

    def left() -> float:
        return max(0.0, time_limit - (time.perf_counter() - t0))

    sub = dict(solve_args)
    for derived in ("llm", "sensitivity", "validate"):
        sub[derived] = False
    sub.pop("initial_solution", None)
    sub.pop("warm_start", None)

    # Root probe: the model as written, root only. A certificate is returned as is;
    # a finite bound means the default relaxation already bounds the model, and the
    # caller solves it as usual with what is left.
    kw = dict(sub)
    kw["max_nodes"] = min(1, int(kw.get("max_nodes", 1)))
    kw["time_limit"] = PROBE_FRACTION * time_limit
    model._outer_piece_child = True  # type: ignore[attr-defined]
    try:
        probe = model.solve(**kw)
    finally:
        del model._outer_piece_child  # type: ignore[attr-defined]
    stats["outer_piece/probe"] = f"{probe.status} bound={probe.bound}"
    if probe.gap_certified or probe.status == "infeasible":
        probe.solver_stats = {**(probe.solver_stats or {}), **stats}
        probe.solver_stats["outer_piece/result"] = "probe certified"
        return probe
    if probe.bound is not None:
        stats["outer_piece/result"] = "probe has a dual bound"
        logger.info("outer piece not needed on %r: %s", model.name, stats)
        return None

    outer_budget = OUTER_BUDGET_FRACTION * time_limit
    B_out: Optional[float] = None
    M_used = math.nan
    d_used = 0
    outer_nodes = 0
    outer_t0 = time.perf_counter()
    for i, M in enumerate(M_GRID):
        left_outer = outer_budget - (time.perf_counter() - outer_t0)
        budget = left_outer * (M_GRID_SHARE[i] / sum(M_GRID_SHARE[i:]))
        if budget <= 0.5:
            break
        outer, d = build_outer(p, M)
        outer._outer_piece_child = True  # type: ignore[attr-defined]
        r = outer.solve(**{**sub, "time_limit": budget})
        outer_nodes += int(r.node_count or 0)
        stats[f"outer_piece/M{M:g}"] = f"{r.status} bound={r.bound} obj={r.objective}"
        b = _usable_outer_bound(r, d)
        if b is not None:
            B_out, M_used, d_used = b, M, d
            break
    if B_out is None:
        stats["outer_piece/result"] = "no usable outer bound"
        logger.info("outer piece declined on %r: %s", model.name, stats)
        return None
    stats["outer_piece/M"] = M_used
    stats["outer_piece/degree"] = float(d_used)
    stats["outer_piece/outer_bound"] = B_out

    abs_tol = solve_args.get("abs_gap_tolerance")
    abs_tol = 1e-6 if abs_tol is None else float(abs_tol)
    rel_tol = float(solve_args.get("gap_tolerance", 1e-4))

    M_inner = M_used
    result: Optional[SolveResult] = None
    for attempt in range(2):
        inner = build_inner(p, M_inner)
        inner._outer_piece_child = True  # type: ignore[attr-defined]
        init = solve_args.get("initial_solution")
        args = dict(sub)
        if init:
            by_name = {v.name: v for v in inner._variables}
            args["initial_solution"] = {
                by_name[getattr(k, "name", k)]: val for k, val in init.items()
            }
        args["time_limit"] = left()
        result = inner.solve(**args)
        z = None if result.objective is None else p.sign * float(result.objective)
        if attempt == 0 and z is not None and B_out is not None and d_used >= 1:
            needed = z - abs_tol
            if 0.0 < B_out < needed and left() > 1.0:
                # c bounds every M' >= M by c (M'/M)^d: enlarge once, keep the bound.
                scale = (needed / B_out) ** (1.0 / d_used) * (1.0 + 1e-6)
                M_inner = M_used * scale
                B_out = B_out * scale**d_used
                stats["outer_piece/M_enlarged"] = M_inner
                continue
        break
    assert result is not None
    stats["outer_piece/M_inner"] = M_inner
    stats["outer_piece/outer_bound_final"] = B_out

    # Combine in minimization space.
    if result.status == "infeasible":
        b_in = math.inf
    elif result.bound is None or not math.isfinite(float(result.bound)):
        b_in = -math.inf
    else:
        b_in = p.sign * float(result.bound)
    bound = min(b_in, B_out)
    z = None if result.objective is None or result.x is None else p.sign * float(result.objective)

    if z is None:
        if bound == math.inf:
            result.status = "infeasible"
            result.bound = None
            result.gap_certified = True
            stats["outer_piece/result"] = "both pieces infeasible"
        else:
            stats["outer_piece/result"] = "no incumbent"
            logger.info("outer piece declined on %r: %s", model.name, stats)
            return None
    else:
        outer_names = [v.name for v in model._variables]
        flat = np.concatenate(
            [np.ravel(np.asarray(result.x[n], dtype=np.float64)) for n in outer_names]
        )
        check = verify_point(model, flat)
        result.bound = p.sign * bound if math.isfinite(bound) else None
        result.gap = (
            reported_gap(result.objective, result.bound) if result.bound is not None else None
        )
        closed = math.isfinite(bound) and (
            z - bound <= abs_tol or (result.gap is not None and result.gap <= rel_tol)
        )
        if closed:
            result.status = "optimal"
            result.gap_certified = True
            stats["gap_criterion"] = "absolute" if z - bound <= abs_tol else "relative"
            stats["outer_piece/result"] = "certified"
        else:
            if result.status == "optimal":
                result.status = "feasible"
            result.gap_certified = False
            stats.pop("gap_criterion", None)
            stats["outer_piece/result"] = "open"
        if not check.ok:
            logger.error(
                "outer piece: the inner incumbent fails the original model (%s); withheld",
                check.reason,
            )
            _withhold_false_primal(
                result,
                "the incumbent of the inner piece is infeasible in the original model "
                f"({check.reason}); it was withheld (false-primal guard, #772)",
            )
    result._model = model
    result.node_count = int(result.node_count or 0) + outer_nodes
    result.wall_time = time.perf_counter() - t0
    if result.solver_stats is None:
        result.solver_stats = {}
    result.solver_stats.update(stats)
    return result
