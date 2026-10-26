"""Array-native affine extraction: an expression DAG to ``(M, b, shape)`` (#1720).

Every other JAX-free LP extractor in :mod:`discopt._relax.problem_classifier` either
reduces a constraint to one scalar row (the arena walk, the algebraic walk) or
expands an array-valued body into per-element tape nodes (the tape rung, capped at
``_MAX_TAPE_NODES``). A dense vectorized LP -- ``dm.sum(C * x)`` with an ndarray
``C`` and ``dm.sum(x, axis=k) <= v`` over a 1000x1000 ``x`` -- defeats all of them,
and the next rung down is ``jax.jacobian``, which materialises a dense
``(rows, n)`` Jacobian: 2000 x 1,000,000 float64 is 16 GB before the dense ``A``
rows are stacked, and the solve was SIGKILLed.

This module evaluates the DAG *symbolically over arrays*. Every node becomes an
:class:`Affine` -- the exact affine map ``node(x) = M @ x + b``, flattened in C
order, with ``M`` a scipy CSR matrix of shape ``(node.size, n)`` -- so each
numpy-style operation is one sparse-matrix operation on the node's nonzeros:

* ``C * x`` with an ndarray constant ``C`` scales the rows of ``M`` by ``C.ravel()``;
* ``sum(e, axis=k)`` left-multiplies ``M`` by a 0/1 reduction matrix;
* ``e[idx]`` and broadcasting select (and repeat) rows of ``M``;
* ``A @ e`` / ``e @ B`` with a constant matrix are Kronecker products with an
  identity, applied to ``M``.

Cost is O(nnz) in memory and time, with no per-element Python and no dense
intermediate. The result is exact -- no probing, no evaluation, no tolerance -- and
the walk refuses loudly (:class:`NotAffineError`) on anything it cannot represent
exactly: a product of two variable-carrying operands, a nonlinear function, an
unknown node type, an index it cannot apply. A refusal is a decline, never a guess.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Optional

import numpy as np
import scipy.sparse as sp

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    IndexExpression,
    MatMulExpression,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)


class NotAffineError(Exception):
    """The expression is not affine, or uses a construct this walk cannot represent."""


class Affine(NamedTuple):
    """``value(x) = M @ x + b`` for a node of shape ``shape``, flattened in C order.

    ``M is None`` marks a node with no variable dependence at all (a constant). It
    is kept structurally distinct from an all-zero ``M`` so a ``(10**6, 10**6)``
    zero matrix is never allocated for a constant operand.
    """

    M: Optional[Any]  # scipy.sparse.csr_matrix of shape (size, n), or None
    b: np.ndarray  # shape (size,)
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(self.b.shape[0])


def _size(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _const(value: np.ndarray) -> Affine:
    v = np.asarray(value, dtype=np.float64)
    return Affine(None, v.reshape(-1).copy(), tuple(int(d) for d in v.shape))


def _select(a: Affine, idx: np.ndarray, shape: tuple[int, ...]) -> Affine:
    """Rows ``idx`` (flat, C order) of ``a``, as a node of shape ``shape``."""
    idx = np.asarray(idx, dtype=np.intp).reshape(-1)
    M = None if a.M is None else a.M[idx]
    return Affine(M, a.b[idx], shape)


def _broadcast_to(a: Affine, shape: tuple[int, ...]) -> Affine:
    if a.shape == shape:
        return a
    try:
        idx = np.broadcast_to(np.arange(a.size).reshape(a.shape), shape)
    except ValueError as exc:
        raise NotAffineError(f"cannot broadcast {a.shape} to {shape}") from exc
    return _select(a, idx, shape)


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    try:
        return tuple(int(d) for d in np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise NotAffineError(f"operand shapes {shapes} do not broadcast") from exc


def _add_many(terms: list[Affine], signs: list[float], n: int) -> Affine:
    """``sum(sign_k * term_k)`` with numpy broadcasting, in one sparse assembly.

    One COO concatenation for all ``k`` rather than ``k - 1`` pairwise sparse
    additions, so an n-ary sum costs O(total nnz) however many terms it has.
    """
    shape = _broadcast_shape(*(t.shape for t in terms))
    size = _size(shape)
    b = np.zeros(size, dtype=np.float64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for t, s in zip(terms, signs):
        t = _broadcast_to(t, shape)
        b += s * t.b
        if t.M is not None:
            coo = t.M.tocoo()
            rows.append(coo.row)
            cols.append(coo.col)
            vals.append(s * coo.data)
    if not rows:
        return Affine(None, b, shape)
    M = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(size, n)
    )
    M.sum_duplicates()
    return Affine(M, b, shape)


def _scale(a: Affine, c: Affine) -> Affine:
    """Element-wise ``a * c`` for a constant ``c`` (``c.M is None``)."""
    shape = _broadcast_shape(a.shape, c.shape)
    a = _broadcast_to(a, shape)
    cv = _broadcast_to(c, shape).b
    if a.M is None:
        M = None
    elif cv.size == 1:
        # A one-row node: ``diag(cv) @ M`` multiplies every stored entry by the one
        # value, exactly what the scalar product does, without building ``diags``.
        M = (a.M * float(cv[0])).tocsr()
    else:
        M = (sp.diags(cv, format="csr") @ a.M).tocsr()
    return Affine(M, a.b * cv, shape)


def _left_mul(K: Any, a: Affine, shape: tuple[int, ...]) -> Affine:
    """``K @ a`` for a constant (sparse or dense) ``K`` of shape ``(size(shape), a.size)``."""
    K = sp.csr_matrix(K)
    M = None if a.M is None else (K @ a.M).tocsr()
    return Affine(M, np.asarray(K @ a.b, dtype=np.float64).reshape(-1), shape)


def _sum_axis(a: Affine, axis: Any) -> Affine:
    """``numpy.sum(a, axis=axis)`` as a 0/1 reduction matrix applied to ``a``."""
    ndim = len(a.shape)
    axes: tuple[int, ...]
    if axis is None:
        axes = tuple(range(ndim))
    else:
        raw = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
        picked: list[int] = []
        for ax in raw:
            ax = int(ax)
            if not -ndim <= ax < ndim:
                raise NotAffineError(f"sum axis {axis!r} out of range for shape {a.shape}")
            picked.append(ax % ndim)
        if len(set(picked)) != len(picked):
            raise NotAffineError(f"repeated sum axis {axis!r}")
        axes = tuple(picked)
    out_shape = tuple(d for i, d in enumerate(a.shape) if i not in axes)
    keep_shape = tuple(1 if i in axes else d for i, d in enumerate(a.shape))
    out_size = _size(out_shape)
    label = np.broadcast_to(np.arange(out_size).reshape(keep_shape), a.shape).reshape(-1)
    R = sp.csr_matrix(
        (np.ones(a.size, dtype=np.float64), (label, np.arange(a.size))), shape=(out_size, a.size)
    )
    return _left_mul(R, a, out_shape)


def _matmul(left: Affine, right: Affine) -> Affine:
    """``left @ right`` where at least one side is constant, numpy semantics, ndim <= 2."""
    if left.M is None and right.M is None:
        out = np.matmul(left.b.reshape(left.shape), right.b.reshape(right.shape))
        return _const(out)
    if len(left.shape) not in (1, 2) or len(right.shape) not in (1, 2):
        raise NotAffineError(f"matmul of shapes {left.shape} @ {right.shape}")
    if left.shape[-1] != right.shape[0]:
        raise NotAffineError(f"matmul shape mismatch {left.shape} @ {right.shape}")
    if left.M is None:
        A = left.b.reshape(left.shape)
        A2 = A.reshape(1, -1) if A.ndim == 1 else A
        if len(right.shape) == 1:
            # (p, q) @ (q,) -> (p,);  (q,) @ (q,) -> ()
            out_shape: tuple[int, ...] = () if A.ndim == 1 else (A.shape[0],)
            return _left_mul(A2, right, out_shape)
        r = right.shape[1]
        out_shape = (r,) if A.ndim == 1 else (A.shape[0], r)
        # result[i, k] = sum_j A[i, j] X[j, k]; X's row j*r + k
        return _left_mul(sp.kron(sp.csr_matrix(A2), sp.identity(r)), right, out_shape)
    if right.M is None:
        B = right.b.reshape(right.shape)
        B2 = B.reshape(-1, 1) if B.ndim == 1 else B
        if len(left.shape) == 1:
            # (q,) @ (q, r) -> (r,);  (q,) @ (q,) -> ()
            out_shape = () if B.ndim == 1 else (B.shape[1],)
            return _left_mul(B2.T, left, out_shape)
        p = left.shape[0]
        out_shape = (p,) if B.ndim == 1 else (p, B.shape[1])
        # result[i, k] = sum_j X[i, j] B[j, k]; X's row i*q + j
        return _left_mul(sp.kron(sp.identity(p), sp.csr_matrix(B2.T)), left, out_shape)
    raise NotAffineError("matmul between two variable-carrying operands")


def _children(node: Expression) -> list[Expression]:
    if isinstance(node, (Constant, Variable, Parameter)):
        return []
    if isinstance(node, IndexExpression):
        return [node.base]
    if isinstance(node, (BinaryOp, MatMulExpression)):
        return [node.left, node.right]
    if isinstance(node, (UnaryOp, SumExpression)):
        return [node.operand]
    if isinstance(node, SumOverExpression):
        return list(node.terms)
    raise NotAffineError(f"unhandled expression type {type(node).__name__}")


def _combine(
    node: Expression,
    kids: list[Affine],
    offset_of: Callable[[Variable], int],
    n: int,
    allow_parameters: bool,
) -> Affine:
    if isinstance(node, Constant):
        return _const(node.value)
    if isinstance(node, Parameter):
        if not allow_parameters:
            raise NotAffineError(f"parameter {node.name} (caller needs a value-free map)")
        # A parameter is fixed for the duration of a solve; the tape reads its
        # current value in exactly this way.
        return _const(node.value)
    if isinstance(node, Variable):
        size = int(node.size)
        offset = int(offset_of(node))
        if offset < 0 or offset + size > n:
            raise NotAffineError(f"variable {node.name} slots outside the model's {n}")
        M = sp.csr_matrix(
            (np.ones(size), (np.arange(size), offset + np.arange(size))), shape=(size, n)
        )
        return Affine(M, np.zeros(size, dtype=np.float64), tuple(int(d) for d in node.shape))
    if isinstance(node, IndexExpression):
        (base,) = kids
        index = node.index
        parts = index if isinstance(index, tuple) else (index,)
        if any(isinstance(p, Expression) for p in parts):
            raise NotAffineError("index containing an expression")
        try:
            idx = np.arange(base.size).reshape(base.shape)[index]
        except (IndexError, TypeError, ValueError) as exc:
            raise NotAffineError(f"cannot apply index {index!r} to shape {base.shape}") from exc
        idx = np.asarray(idx)
        return _select(base, idx, tuple(int(d) for d in idx.shape))
    if isinstance(node, BinaryOp):
        left, right = kids
        op = node.op
        if op == "+":
            return _add_many([left, right], [1.0, 1.0], n)
        if op == "-":
            return _add_many([left, right], [1.0, -1.0], n)
        both_const = left.M is None and right.M is None
        if both_const and op in ("*", "/", "**"):
            lv = left.b.reshape(left.shape)
            rv = right.b.reshape(right.shape)
            with np.errstate(all="ignore"):
                out = lv * rv if op == "*" else (lv / rv if op == "/" else lv**rv)
            return _const(np.asarray(out))
        if op == "*":
            if left.M is None:
                return _scale(right, left)
            if right.M is None:
                return _scale(left, right)
            raise NotAffineError("product of two variable-carrying operands")
        if op == "/":
            if right.M is None:
                if np.any(right.b == 0.0):
                    raise NotAffineError("division by a constant containing zero")
                return _scale(left, Affine(None, 1.0 / right.b, right.shape))
            raise NotAffineError("division by a variable-carrying operand")
        if op == "**":
            if right.M is None and right.size == 1 and float(right.b[0]) == 1.0:
                return left
            raise NotAffineError("power of a variable-carrying operand")
        raise NotAffineError(f"non-affine binary operator {op!r}")
    if isinstance(node, UnaryOp):
        (operand,) = kids
        if node.op == "neg":
            return Affine(
                None if operand.M is None else (-operand.M).tocsr(), -operand.b, operand.shape
            )
        if operand.M is None and node.op == "abs":
            return Affine(None, np.abs(operand.b), operand.shape)
        raise NotAffineError(f"non-affine unary operator {node.op!r}")
    if isinstance(node, SumExpression):
        (operand,) = kids
        return _sum_axis(operand, node.axis)
    if isinstance(node, MatMulExpression):
        left, right = kids
        return _matmul(left, right)
    if isinstance(node, SumOverExpression):
        if not kids:
            return _const(np.zeros(()))
        return _add_many(kids, [1.0] * len(kids), n)
    raise NotAffineError(f"unhandled expression type {type(node).__name__}")


def affine_of(
    expr: Expression,
    offset_of: Callable[[Variable], int],
    n: int,
    *,
    allow_parameters: bool = True,
    memo: Optional[dict[int, tuple[Expression, Affine]]] = None,
) -> Affine:
    """The exact affine map of ``expr`` over the ``n`` flat variable slots.

    ``offset_of(var)`` returns the first flat slot of ``var``. A ``Parameter`` is
    read at its current value; ``allow_parameters=False`` refuses one instead, for
    a caller that caches the map across parameter updates. Raises
    :class:`NotAffineError` when ``expr`` is not affine or uses a construct this
    walk does not represent.

    Iterative post-order with a memo keyed by node identity, so a shared
    sub-DAG is evaluated once and a deep left-leaning ``((a + b) + c) + ...``
    chain (the ``.nl`` shape, #1064) costs no Python recursion.

    ``memo`` lets a caller share that memo across several calls with the same
    ``offset_of``, ``n`` and ``allow_parameters`` -- the rows of one model, whose
    bodies share their variable and element nodes (``x[i, j]`` is one cached
    node, #1215). Each entry holds its node, so no id in it can be recycled onto
    another object while the memo lives (#1329). The maps are the same either way.
    """
    if memo is None:
        memo = {}
    stack: list[tuple[Expression, bool]] = [(expr, False)]
    while stack:
        node, expanded = stack.pop()
        key = id(node)
        if key in memo:
            continue
        kids = _children(node)
        if not expanded and kids:
            stack.append((node, True))
            for kid in reversed(kids):
                if id(kid) not in memo:
                    stack.append((kid, False))
            continue
        memo[key] = (
            node,
            _combine(node, [memo[id(k)][1] for k in kids], offset_of, n, allow_parameters),
        )
    return memo[id(expr)][1]
