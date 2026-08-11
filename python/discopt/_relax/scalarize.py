"""Symbolic scalarization of array-valued expressions (issue #981).

Why this module exists
----------------------

A :class:`~discopt.modeling.core.Constraint` body may be *array-valued*: ``A @ x
== b`` on a 3-vector is ONE ``Constraint`` object standing for three scalar
rows, and ``(-k) * X[:, 1:]`` is one expression node standing for a whole matrix
of scalar products. The modeling layer, the NLP evaluator (one row per flat
element — see :mod:`discopt.validation.feasibility`) and the Rust IR all handle
that. The *relaxation* engine did not:
:func:`~discopt._relax.uniform_relax.build_uniform_relaxation`
emitted one LP row per ``Constraint`` object and its interval walk assumed every
node enclosed to a scalar. On any model with an array-valued nonlinear body the
build raised ``TypeError: only 0-dimensional arrays can be converted to Python
scalars`` deep inside ``_Builder.bounds``, the caller degraded to no relaxation
at all, and the dual bound collapsed to the trivial objective floor — ``0.0`` for
a sum of squares (measured on ``docs/notebooks/tutorial_dae.ipynb``: relative gap
100 %, so *no* time limit could ever certify).

That is a bound-quality defect, not a soundness one — dropping relaxation rows
only enlarges the feasible set — but it silently disabled global optimality for
every vectorized model: DAE collocation, the ``nn`` layer formulations, and any
user model written in numpy style. Scalarizing at the *expression* level is the
class fix: the product ``(-k) * X[:, 1:]`` becomes ``len(X[:, 1:])`` genuine
scalar bilinear atoms that McCormick can envelope, and the term classifier sees
them as partition candidates so spatial branching can close the gap.

Contract
--------

:func:`scalar_elements` returns the row-major list of scalar
:class:`~discopt.modeling.core.Expression` objects that an array-valued
expression stands for, or ``None`` when static scalarization is not possible
(unknown shape, an opaque :class:`~discopt.modeling.core.CustomCall`, an exotic
index, a >2-D matmul, …). ``None`` means "caller keeps its previous behaviour";
it never means "assume scalar". Callers must treat ``None`` as unknown, never as
an empty row set — dropping rows silently would weaken a bound with no record.

The rewrite is *value-preserving by construction*: every element is built from
the same operator nodes with numpy's own broadcasting index arithmetic, so
``scalar_elements(e)[flat_i]`` evaluates to ``np.ravel(e)[flat_i]`` at every
point. ``test_scalarize.py`` asserts exactly that numerically against the NLP
evaluator on each supported node type.
"""

from __future__ import annotations

import logging
from typing import Optional, cast

import numpy as np

from discopt.modeling.core import (
    _ELEMENTWISE_FUNCS,
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    _index_result_shape,
    _known_shape,
)

logger = logging.getLogger(__name__)

#: Sentinel for "no memoized shape on this node" (``None`` is a real answer).
_MISS = object()

__all__ = ["scalar_elements", "static_shape", "MAX_SCALAR_ELEMENTS"]

#: Refuse to expand a single expression into more scalar elements than this.
#: A relaxation row per element is the point of the exercise, but an accidental
#: 10^7-element broadcast would exhaust memory building expression objects. Above
#: the cap :func:`scalar_elements` returns ``None`` (caller keeps its old path)
#: and logs at WARNING — never a silent truncation, which would drop rows and
#: weaken the bound with no record.
MAX_SCALAR_ELEMENTS = 200_000

#: Element-wise binary operators. ``@`` is handled separately (contraction).
_ELEMENTWISE_BINOPS = frozenset({"+", "-", "*", "/", "**"})


class _Unscalarizable(Exception):
    """Internal: this node cannot be expanded element-wise."""


def static_shape(expr: Expression) -> Optional[tuple[int, ...]]:
    """Static numpy shape of ``expr``, or ``None`` when it is not known.

    Deliberately *not* a call to :func:`discopt.modeling.core._known_shape`, which
    caches ``None`` on any node built over a matmul or a reduction and therefore
    reports "unknown" for the whole tree above it. ``A @ x - k * y`` is exactly
    that shape: its ``BinaryOp`` cached ``None`` at construction because matmul's
    shape is not part of the M8 guard. This function recomputes bottom-up, adding
    the cases this module resolves exactly — ``@`` over 1-D/2-D operands and a
    full reduction (scalar) — so a composite over them keeps a concrete shape.

    Memoized on the node (``_scalarize_shape``): the walk is called once per
    element of every expansion, and a shared subtree would otherwise be re-walked
    combinatorially.
    """
    cached = getattr(expr, "_scalarize_shape", _MISS)
    if cached is not _MISS:
        return cast(Optional[tuple[int, ...]], cached)
    shape = _static_shape_uncached(expr)
    try:
        expr._scalarize_shape = shape  # type: ignore[attr-defined]
    except AttributeError:
        # A ``__slots__`` node cannot carry the memo. Correctness is unaffected;
        # only the walk is slower. Narrow on purpose — any other failure escapes.
        pass
    return shape


def _static_shape_uncached(expr: Expression) -> Optional[tuple[int, ...]]:
    if isinstance(expr, Constant):
        return tuple(int(d) for d in expr.value.shape)

    if isinstance(expr, (Variable, Parameter)):
        return _known_shape(expr)

    if isinstance(expr, IndexExpression):
        base_shape = static_shape(expr.base)
        if base_shape is None:
            return None
        return _index_result_shape(base_shape, expr.index)

    if isinstance(expr, UnaryOp):
        return static_shape(expr.operand)

    if isinstance(expr, BinaryOp):
        if expr.op not in _ELEMENTWISE_BINOPS:
            return None
        return _broadcast(static_shape(expr.left), static_shape(expr.right))

    if isinstance(expr, FunctionCall):
        if expr.func_name not in _ELEMENTWISE_FUNCS:
            return None
        shape: Optional[tuple[int, ...]] = ()
        for arg in expr.args:
            shape = _broadcast(shape, static_shape(arg))
            if shape is None:
                return None
        return shape

    if isinstance(expr, MatMulExpression):
        return _matmul_shape(static_shape(expr.left), static_shape(expr.right))

    if isinstance(expr, SumExpression):
        # A full reduction is scalar. An axis-reduction's shape depends on the
        # evaluator's axis semantics for a partially-known operand — report
        # unknown rather than re-derive them here.
        return () if expr.axis is None else None

    if isinstance(expr, SumOverExpression):
        shapes = [static_shape(t) for t in expr.terms]
        if shapes and all(s == () for s in shapes):
            return ()
        return None

    return _known_shape(expr)


def _broadcast(
    ls: Optional[tuple[int, ...]], rs: Optional[tuple[int, ...]]
) -> Optional[tuple[int, ...]]:
    """numpy broadcast of two shapes; ``None`` if either is unknown or they clash."""
    if ls is None or rs is None:
        return None
    if ls == rs or rs == ():
        return ls
    if ls == ():
        return rs
    try:
        return tuple(int(d) for d in np.broadcast_shapes(ls, rs))
    except ValueError:
        return None


def scalar_elements(expr: Expression) -> Optional[list[Expression]]:
    """Row-major scalar expansion of ``expr``, or ``None`` if not possible.

    A scalar expression returns ``[expr]`` unchanged (identity, not a copy), so
    callers can route every expression through this function without paying for
    the common case.
    """
    shape = static_shape(expr)
    if shape is None:
        return None
    if shape == ():
        return [expr]
    n = int(np.prod(shape)) if shape else 1
    if n > MAX_SCALAR_ELEMENTS:
        logger.warning(
            "scalarize: refusing to expand a %s expression into %d scalar elements "
            "(cap %d); caller keeps its unexpanded path",
            shape,
            n,
            MAX_SCALAR_ELEMENTS,
        )
        return None
    try:
        return [_elem(expr, idx) for idx in np.ndindex(*shape)]
    except _Unscalarizable as exc:
        logger.debug("scalarize: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _matmul_shape(
    ls: Optional[tuple[int, ...]], rs: Optional[tuple[int, ...]]
) -> Optional[tuple[int, ...]]:
    """numpy ``@`` result shape for 1-D/2-D operands, else ``None``."""
    if ls is None or rs is None:
        return None
    if not (1 <= len(ls) <= 2 and 1 <= len(rs) <= 2):
        return None  # stacked/batched matmul — not handled
    if ls[-1] != rs[0]:
        return None  # inner dimensions disagree; let the evaluator report it
    if len(ls) == 1 and len(rs) == 1:
        return ()
    if len(ls) == 2 and len(rs) == 1:
        return (ls[0],)
    if len(ls) == 1 and len(rs) == 2:
        return (rs[1],)
    return (ls[0], rs[1])


def _broadcast_index(idx: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    """Map a result index onto an operand of ``shape`` under numpy broadcasting.

    Right-aligned; a length-1 operand axis is stretched, so it always reads 0.
    """
    if shape == ():
        return ()
    offset = len(idx) - len(shape)
    return tuple(0 if shape[a] == 1 else idx[offset + a] for a in range(len(shape)))


def _index_key(shape: tuple[int, ...], idx: tuple[int, ...]):
    """The ``[]`` key that selects element ``idx`` of a leaf of ``shape``."""
    return int(idx[0]) if len(shape) == 1 else tuple(int(i) for i in idx)


def _elem(expr: Expression, idx: tuple[int, ...]) -> Expression:
    """Scalar expression for element ``idx`` of ``expr``.

    ``idx`` is indexed against ``expr``'s own shape (already broadcast-resolved by
    the caller). Raises :class:`_Unscalarizable` for any node this module cannot
    expand — never returns an approximation.
    """
    shape = static_shape(expr)
    if shape is None:
        raise _Unscalarizable(f"unknown static shape for {type(expr).__name__}")
    if shape == ():
        return expr  # scalar operand broadcast against a shaped sibling

    if isinstance(expr, Constant):
        return Constant(float(expr.value[idx]))

    if isinstance(expr, (Variable, Parameter)):
        return cast(Expression, expr[_index_key(shape, idx)])

    if isinstance(expr, IndexExpression):
        base_shape = static_shape(expr.base)
        if base_shape is None or base_shape == ():
            raise _Unscalarizable("indexed base has no shaped static form")
        # Resolve the (possibly sliced / fancy) index with numpy itself rather
        # than re-deriving its semantics: walk the base's flat positions through
        # the same key the evaluator applies, then read off which base element
        # this result element came from.
        try:
            positions = np.arange(int(np.prod(base_shape))).reshape(base_shape)[expr.index]
            base_flat = int(np.asarray(positions)[idx])
        except Exception as exc:  # index numpy cannot resolve statically
            raise _Unscalarizable(f"unresolvable index {expr.index!r}: {exc}") from exc
        return _elem(expr.base, tuple(int(i) for i in np.unravel_index(base_flat, base_shape)))

    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _elem_broadcast(expr.operand, idx))

    if isinstance(expr, BinaryOp):
        if expr.op not in _ELEMENTWISE_BINOPS:
            raise _Unscalarizable(f"binary op {expr.op!r} is not element-wise")
        return BinaryOp(
            expr.op,
            _elem_broadcast(expr.left, idx),
            _elem_broadcast(expr.right, idx),
        )

    if isinstance(expr, FunctionCall):
        if expr.func_name not in _ELEMENTWISE_FUNCS:
            raise _Unscalarizable(f"function {expr.func_name!r} is not element-wise")
        return FunctionCall(expr.func_name, *[_elem_broadcast(a, idx) for a in expr.args])

    if isinstance(expr, MatMulExpression):
        return _elem_matmul(expr, idx)

    raise _Unscalarizable(f"cannot scalarize {type(expr).__name__}")


def _elem_broadcast(operand: Expression, idx: tuple[int, ...]) -> Expression:
    """``_elem`` of ``operand`` at the broadcast-resolved position of ``idx``."""
    shape = static_shape(operand)
    if shape is None:
        raise _Unscalarizable(f"unknown static shape for {type(operand).__name__}")
    if shape == ():
        return operand
    return _elem(operand, _broadcast_index(idx, shape))


def _elem_matmul(expr: MatMulExpression, idx: tuple[int, ...]) -> Expression:
    """Element ``idx`` of ``left @ right`` as an explicit contraction sum."""
    ls = static_shape(expr.left)
    rs = static_shape(expr.right)
    if ls is None or rs is None:
        raise _Unscalarizable("matmul operand shape unknown")
    inner = ls[-1]

    def left_at(k: int) -> Expression:
        pos = (k,) if len(ls) == 1 else (int(idx[0]), k)
        return _elem(expr.left, pos)

    def right_at(k: int) -> Expression:
        pos: tuple[int, ...]
        if len(rs) == 1:
            pos = (k,)
        else:
            # Column index is the LAST result axis; for (1-D @ 2-D) the result has
            # a single axis which IS the column.
            pos = (k, int(idx[-1]))
        return _elem(expr.right, pos)

    acc: Optional[Expression] = None
    for k in range(int(inner)):
        term = BinaryOp("*", left_at(k), right_at(k))
        acc = term if acc is None else BinaryOp("+", acc, term)
    if acc is None:  # inner dimension 0 — an empty contraction is exactly zero
        return Constant(0.0)
    return acc
