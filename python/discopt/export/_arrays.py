"""Array expansion shared by every exporter (issue #1215).

A discopt model may carry **array-structured** expressions: a matrix product, an
axis reduction, an unindexed shaped variable, a shaped parameter. One such
constraint is many scalar rows, and one such objective is a scalar value that no
scalar-oriented writer can walk directly.

Every exporter needs the same expansion, and for a while each had its own answer:
``nl.py`` expanded constraint bodies (but not objectives), while ``gams.py``,
``lp.py`` and ``mps.py`` had none at all and died inside their DAG walk with
``TypeError: only 0-dimensional arrays can be converted to Python scalars``. That
is four implementations of one idea, three of them empty -- and it mattered more
than a tidiness complaint, because the vectorised form of a model is the one that
reaches solve-ready 36x faster and 28x lighter than the per-element idiom
(performance-plan §41), so refusing it confines every large model to the slow
path.

This module is that one implementation. It is pure: nothing here reads model
state, so a writer supplies only its own ``resolve_index`` callback to
:func:`needs_scalarize`, which decides whether an expression needs expanding at
all (a purely scalar body skips the recursive pass, which also keeps deep
``sum()`` chains off the recursion limit).

Soundness note: reductions are **not** element-wise. ``norm``-p and ``prod``
fold over their argument, so :func:`scalarize` routes them through
:func:`scalarize_reduction` rather than broadcasting them -- broadcasting turned
``norm(x)`` on a 3-vector into three rows of ``norm2(x[i])``, which is a
different model.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import numpy as np

from discopt.modeling.core import (
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
)


def elem(arr: np.ndarray, idx: Any) -> Expression:
    """Read one element out of an ``object``-dtype array of expressions.

    ``ndarray`` is not generic over its ``object`` payload, so numpy's stubs type
    every element access as another array. These arrays only ever hold
    :class:`Expression` nodes; the cast records that once rather than silencing
    the checker at each call site.
    """
    return cast(Expression, arr[idx])


def obj0(x) -> np.ndarray:
    """Wrap a single expression in a 0-d object array."""
    out = np.empty((), dtype=object)
    out[()] = x
    return out


def sum_terms(terms: list[Expression]) -> Expression:
    """Left-fold a list of scalar expressions into a sum (``+``)."""
    if not terms:
        return Constant(0.0)
    result: Expression = terms[0]
    for t in terms[1:]:
        result = BinaryOp("+", result, t)
    return result


def matmul_scalar(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Symbolic matmul of two object arrays of scalar expressions."""
    if left.ndim == 1 and right.ndim == 1:
        (k,) = left.shape
        return obj0(sum_terms([BinaryOp("*", left[i], right[i]) for i in range(k)]))
    if left.ndim == 2 and right.ndim == 1:
        m, k = left.shape
        out = np.empty((m,), dtype=object)
        for i in range(m):
            out[i] = sum_terms([BinaryOp("*", left[i, p], right[p]) for p in range(k)])
        return out
    if left.ndim == 1 and right.ndim == 2:
        k, n = right.shape
        out = np.empty((n,), dtype=object)
        for j in range(n):
            out[j] = sum_terms([BinaryOp("*", left[p], right[p, j]) for p in range(k)])
        return out
    if left.ndim == 2 and right.ndim == 2:
        m, k = left.shape
        _, n = right.shape
        out = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                out[i, j] = sum_terms([BinaryOp("*", left[i, p], right[p, j]) for p in range(k)])
        return out
    raise ValueError(f"Unsupported matmul of shapes {left.shape} @ {right.shape}")


def sum_axis(arr: np.ndarray, axis: int | None) -> np.ndarray:
    """Symbolic reduction (``+``) of an object array along ``axis``."""
    if axis is None:
        return obj0(sum_terms([elem(arr, idx) for idx in np.ndindex(arr.shape)]))
    moved = np.moveaxis(arr, axis, 0)
    out_shape = moved.shape[1:]
    out = np.empty(out_shape, dtype=object)
    for idx in np.ndindex(out_shape):
        out[idx] = sum_terms([moved[(p, *idx)] for p in range(moved.shape[0])])
    if out.ndim == 0:
        return obj0(out[()])
    return out


def scalarize_reduction(expr: FunctionCall) -> Expression:
    """Expand a vector reduction (``norm``-p / ``prod``) into plain algebra.

    ``.nl`` has no norm or product opcode, so the reduction is rewritten in
    terms of ones it does have. Each rewrite is EXACT, not an approximation:

    * ``norm1(x)``   -> ``|x_0| + |x_1| + ...``
    * ``norm2(x)``   -> ``sqrt(x_0^2 + x_1^2 + ...)``
    * ``norm``-p     -> ``(|x_0|^p + ...)^(1/p)``
    * ``prod(x)``    -> ``x_0 * x_1 * ...``

    ``norminf`` is refused rather than rewritten: it is a max-chain, and the
    writer already refuses ``min``/``max`` because they need the DNLP model
    type. A 2-D argument is refused too -- ``jnp.linalg.norm`` of a matrix is
    the induced/spectral norm, which is not a fold over elements, so
    expanding it entrywise would export different mathematics (the same
    refusal ``_nl_expr_compiler._lower_norm`` makes on the tape path).
    """
    if len(expr.args) != 1:
        raise ValueError(
            f"{expr.func_name}() takes one argument in .nl export, got {len(expr.args)}"
        )
    arr = scalarize(expr.args[0])
    if arr.ndim > 1:
        raise ValueError(
            f"{expr.func_name}() of a {arr.ndim}-D argument is the induced "
            f"(matrix) norm, which is not a fold over elements and has no .nl "
            f"equivalent; reduce to a vector first."
        )
    if arr.ndim == 0:
        raise ValueError(
            f"{expr.func_name}() of a scalar is not a vector reduction; pass a shaped expression."
        )
    elems = [elem(arr, i) for i in range(arr.shape[0])]

    name = expr.func_name
    if name == "prod":
        out: Expression = elems[0]
        for e in elems[1:]:
            out = BinaryOp("*", out, e)
        return out

    suffix = name[len("norm") :]
    if suffix == "inf":
        raise ValueError(
            "norminf() is a max-chain and requires the DNLP model type; not supported in .nl export"
        )
    order = 2.0 if suffix == "" else float(suffix)
    if order == 1.0:
        return sum_terms([UnaryOp("abs", e) for e in elems])
    if order == 2.0:
        squares = [BinaryOp("*", e, e) for e in elems]
        return FunctionCall("sqrt", sum_terms(squares))
    powers = [BinaryOp("**", UnaryOp("abs", e), Constant(order)) for e in elems]
    return BinaryOp("**", sum_terms(powers), Constant(1.0 / order))


def scalarize(expr: Expression) -> np.ndarray:
    """Expand an expression into an object ndarray of scalar expressions.

    Indexing, broadcasting, and matrix products are pushed through the DAG
    so each output element becomes a plain scalar expression that the rest
    of the (scalar-oriented) writer can decompose. Scalars are returned as
    0-d object arrays. Node types without a known array structure (e.g.
    parameters) are treated as opaque scalar leaves.
    """
    if isinstance(expr, Constant):
        v = expr.value
        if v.ndim == 0:
            return obj0(Constant(float(v)))
        out = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            out[idx] = Constant(float(v[idx]))
        return out

    if isinstance(expr, Parameter):
        # A Parameter holds a value fixed for the solve, so at export time it
        # is a constant -- expand it element-wise exactly like `Constant`
        # above. Before this, parameters were the "no known array structure"
        # case named in the docstring and became opaque 0-d leaves, so `p[i]`
        # on a shaped parameter raised `IndexError: too many indices` from
        # `base[expr.index]` below rather than exporting.
        v = np.asarray(expr.value)
        if v.ndim == 0:
            return obj0(Constant(float(v)))
        out = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            out[idx] = Constant(float(v[idx]))
        return out

    if isinstance(expr, Variable):
        if expr.shape == () or expr.shape == (1,):
            return obj0(expr)
        out = np.empty(expr.shape, dtype=object)
        ndim = len(expr.shape)
        for idx in np.ndindex(expr.shape):
            out[idx] = IndexExpression(expr, idx if ndim > 1 else idx[0])
        return out

    if isinstance(expr, IndexExpression):
        base = scalarize(expr.base)
        if base.ndim == 0:
            # A shape-`(1,)` leaf scalarizes to a 0-d array (see the `Variable`
            # branch above, which treats `(1,)` as scalar-like), so there is
            # nothing left for `x[0]` to index and `base[0]` raised
            # "too many indices for array: array is 0-dimensional". Index 0 --
            # the only valid index into a length-1 axis -- selects that element.
            flat = expr.index
            if isinstance(flat, tuple):
                flat = flat[0] if len(flat) == 1 else None
            if flat == 0:
                return base
            raise ValueError(
                f"index {expr.index!r} is out of range for "
                f"'{getattr(expr.base, 'name', expr.base)}', which has one element"
            )
        sub = base[expr.index]
        return sub if isinstance(sub, np.ndarray) else obj0(sub)

    if isinstance(expr, UnaryOp):
        operand = scalarize(expr.operand)
        out = np.empty(operand.shape, dtype=object)
        for idx in np.ndindex(operand.shape):
            out[idx] = UnaryOp(expr.op, elem(operand, idx))
        return out

    if isinstance(expr, BinaryOp):
        left, right = np.broadcast_arrays(scalarize(expr.left), scalarize(expr.right))
        out = np.empty(left.shape, dtype=object)
        for idx in np.ndindex(left.shape):
            out[idx] = BinaryOp(expr.op, elem(left, idx), elem(right, idx))
        return out

    if isinstance(expr, FunctionCall):
        if expr.func_name.startswith("norm") or expr.func_name == "prod":
            # REDUCTIONS, not element-wise operations. Falling through to the
            # broadcast below turned `norm(x)` on a 3-vector into THREE rows
            # of `norm2(x[i])` -- a different model, silently. It only failed
            # loudly because `norm2` has no `.nl` opcode; giving it one would
            # have made the export wrong instead of refused.
            return obj0(scalarize_reduction(expr))
        args = [scalarize(a) for a in expr.args]
        bargs = list(np.broadcast_arrays(*args)) if len(args) > 1 else args
        shape = bargs[0].shape
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            out[idx] = FunctionCall(expr.func_name, *[elem(b, idx) for b in bargs])
        return out

    if isinstance(expr, MatMulExpression):
        return matmul_scalar(scalarize(expr.left), scalarize(expr.right))

    if isinstance(expr, SumExpression):
        return sum_axis(scalarize(expr.operand), expr.axis)

    if isinstance(expr, SumOverExpression):
        terms = [scalarize(t) for t in expr.terms]
        bterms = list(np.broadcast_arrays(*terms)) if len(terms) > 1 else terms
        shape = bterms[0].shape
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            out[idx] = sum_terms([elem(b, idx) for b in bterms])
        return out

    # Opaque scalar leaf (e.g. Parameter): leave intact.
    return obj0(expr)


def needs_scalarize(
    expr: Expression,
    resolve_index: Optional[Callable[[IndexExpression], Optional[int]]] = None,
) -> bool:
    """True if *expr* has array structure requiring element expansion.

    Walks the DAG iteratively (no recursion). Returns ``True`` on the first node
    that introduces array shape -- an unindexed array ``Variable``, a shaped
    ``Parameter``, a ``MatMulExpression``, an axis-reducing ``SumExpression``, a
    reduction ``FunctionCall``, or an ``IndexExpression`` that does not resolve
    to a single scalar variable. Scalar-only bodies return ``False`` and bypass
    :func:`scalarize` entirely -- which is not only faster: the downstream
    linear/nonlinear split and the writers already traverse scalar nodes
    iteratively, so skipping the recursive pass is what keeps a deep ``sum()``
    chain off the recursion limit.

    *resolve_index* is the caller's "does this index reduce to one scalar
    variable slot?" test, in its own coordinate system. Omitted, every
    ``IndexExpression`` is treated as needing expansion, which is conservative:
    expansion of an already-scalar index is a no-op.
    """
    stack: list[Expression] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Variable):
            if node.shape not in ((), (1,)):
                return True
        elif isinstance(node, Parameter):
            if np.asarray(node.value).shape not in ((), (1,)):
                return True
        elif isinstance(node, Constant):
            # A non-scalar `Constant` is array-structured and must be expanded --
            # every downstream consumer does `float(node.value)` on it, which
            # raises for anything but a 0-d array.
            #
            # Note the threshold differs from `Variable`/`Parameter` above, which
            # treat shape `(1,)` as scalar-like: `scalarize` maps a `(1,)`
            # Variable to a 0-d leaf, so nothing downstream ever sees its shape,
            # while a `(1,)` Constant keeps its array and reaches `float()`. So
            # `ndim != 0` here rather than `shape not in ((), (1,))`.
            #
            # Missed, this made a body of only shape-(1,) leaves -- e.g.
            # `out == prev * y_factor + y_offset` for a single-output embedded
            # network with output scaling -- skip expansion entirely and fail in
            # `_collect_linear` with "only 0-dimensional arrays can be converted
            # to Python scalars" (#1215).
            if np.asarray(node.value).ndim != 0:
                return True
        elif isinstance(node, (MatMulExpression, SumExpression)):
            return True
        elif isinstance(node, IndexExpression):
            if resolve_index is None or resolve_index(node) is None:
                return True
        elif isinstance(node, BinaryOp):
            stack.append(node.left)
            stack.append(node.right)
        elif isinstance(node, UnaryOp):
            stack.append(node.operand)
        elif isinstance(node, FunctionCall):
            # A reduction changes shape, so it needs the expansion pass even when
            # every operand below it is already scalar.
            if node.func_name.startswith("norm") or node.func_name == "prod":
                return True
            stack.extend(node.args)
        elif isinstance(node, SumOverExpression):
            stack.extend(node.terms)
        # Constant, scalar Variable, resolved IndexExpression, opaque leaves: fine
    return False


def scalarize_body(
    expr: Expression,
    resolve_index: Optional[Callable[[IndexExpression], Optional[int]]] = None,
) -> list[Expression]:
    """The scalar rows a (possibly array) constraint body expands to.

    A scalar body yields a single-element list; an array body yields one scalar
    expression per element, in row-major order -- the same order the AD tape and
    the JAX evaluator use, so row *k* means the same row everywhere.
    """
    if not needs_scalarize(expr, resolve_index):
        return [expr]
    arr = scalarize(expr)
    if arr.ndim == 0:
        return [cast(Expression, arr[()])]
    return list(arr.ravel())


def scalarize_objective(
    expr: Expression,
    resolve_index: Optional[Callable[[IndexExpression], Optional[int]]] = None,
) -> Expression:
    """The single scalar expression an objective expands to.

    An objective that is scalar-VALUED but array-STRUCTURED (``-dm.sum(x)``,
    ``dm.sum(A @ x)``) still needs the expansion; skipping it was why six of eight
    array-shaped construct families could not be exported at all. An expansion
    yielding more than one element means the objective is a vector, which no
    format represents -- refused here rather than silently optimising element
    zero.
    """
    bodies = scalarize_body(expr, resolve_index)
    if len(bodies) != 1:
        raise ValueError(
            f"Objective expands to {len(bodies)} scalar expressions; an objective "
            f"must be scalar. Reduce it first, e.g. dm.sum(expr) or dm.norm(expr)."
        )
    return bodies[0]
