"""Compile a discopt expression DAG into a POUNCE ``NlExpr`` tape expression.

This is the shared prerequisite for taking JAX off the solve path (issue #75).
Both remaining JAX jobs need the same thing — the value and gradient of a scalar
expression at a point — and POUNCE's Rust AD tape provides exactly that:

* **separation tangents** (``_relax/uniform_relax.py``) need ``g(x0)`` and
  ``grad g(x0)`` for the Kelley cutting-plane loop;
* **NLP subsolve derivatives** (``_relax/nlp_evaluator.py``) need ``f``, ``grad f``,
  ``g``, ``J``, and the Lagrangian Hessian.

Previously those were headed for two *different* replacement backends. They no
longer need to be: ``pounce.NlExpr`` covers discopt's *scalar* DAG operators — 20
natively and 9 by the exact rewrites below — where the in-tree interval-AD engine
covers 6, and it agrees with analytic truth exactly on expressions the interval
engine cannot evaluate at all (``tanh``, ``erf``).

A tape node is a scalar, but a discopt DAG node is not: ``x`` on a shaped
:class:`Variable`, ``A @ x``, ``dm.sum(x)`` and ``x[1:]`` are all array-valued.
Those are lowered by carrying a **numpy object array of tape nodes** through the
walk — one scalar ``NlExpr`` per element, in the flat ``x`` vector's own C order
— and letting numpy supply broadcasting and indexing. The scalar entry point
:func:`compile_to_nl_expr` then requires a size-1 result, and
:func:`compile_to_nl_array` returns the whole array for callers that want the
rows (an array-valued constraint body is one ``Constraint`` and *many* rows — see
``_tape_nlp_evaluator``).

The **reductions** are numpy's shapes but not numpy's algorithms. ``np.sum`` and
``np.matmul`` reduce object arrays with ``add.reduce`` — a left-leaning ``+``
chain whose depth is the number of terms — and POUNCE *refuses* an expression
nested past ``NlExpr.max_depth`` (10000), raising a ``ValueError`` that
:func:`try_compile` does not catch, so it escapes the JAX fallback rather than
degrading to it. Every sum here is therefore one **n-ary** ``E.sum([...])`` node
of depth 1: :func:`_sum_along` for ``SumExpression``, one per output entry in
:func:`_matmul`, and the same inside ``norm1``/``norm2``/``norm``\\ *p*. The two
reductions with no n-ary opcode — ``prod`` (a ``*`` chain) and ``norminf`` (a
``max`` chain) — fold in :func:`_fold`, which refuses at the depth limit instead.

Two things this deliberately does NOT do. It does not fold an array reduction
into a shape it was not: an earlier revision lowered ``prod`` as a variadic ``*``
chain over ``args``, which silently computed a different function (``prod`` takes
*one array* argument). And it does not materialize an array where a scalar will
do: ``x[i]`` / ``y[i, j]`` on a shaped :class:`Variable` names a single entry of
the flat vector, so :func:`_static_scalar_slot` resolves it arithmetically
without ever building the base's ``size`` nodes. That fast path is what keeps
scalar indexing O(1) per leaf (issue #654); the general path below is the
fallback for the forms it cannot name.

The layout identity the array path rests on is the JAX path's own:
``_relax/dag_compiler`` materializes a variable as ``x_flat[off : off + size]
.reshape(shape)`` in C order, and ``_relax/nlp_evaluator`` concatenates constraint
rows as ``jnp.reshape(body, (-1,))`` — also C order. ``np.ndarray.reshape(-1)``
over the object array reproduces both exactly, so row *k* of a tape-backed
constraint is row *k* of the JAX-backed one.

The **matrix norms** are the one array form still refused rather than
approximated: ``jnp.linalg.norm`` of a 2-D argument is the induced/spectral norm
(``ord=2`` is the largest singular value), which is not a fold over elements.
1-D vector norms are lowered.

*Coverage claims here must be checked against operators, not instances.* ``.nl``
has no opcode for ``sigmoid``/``softplus``/``entropy``/``centropy``/``signpower``,
so a MINLPLib corpus sweep — measured across 316 instances — exercises exactly six
of these (``log``, ``sqrt``, ``exp``, ``abs``, ``sin``, ``cos``) and can never
reach the rest. Three defects lived in the unreached rewrites: ``entropy`` had an
inverted sign, ``_sign`` passed ``compare``'s arguments in the wrong order, and
``prod`` was the wrong function entirely. See
``python/tests/test_75_nl_expr_compiler.py`` for the per-operator differential
that catches this class.

Why not ``.nl`` as the intermediate? Because it is lossy in the wrong direction:
discopt's ``.nl`` writer refuses ``atan2``, ``min``, ``max``, ``erf`` and ``sign``
(``export/nl.py``), several of which the tape *does* differentiate. Building the
tape expression directly keeps them.

Deliberately NOT under ``_relax/``: nothing here touches JAX.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)


class UnsupportedForTape(Exception):
    """``expr`` contains a node the tape cannot represent.

    Raised rather than returned so a caller cannot mistake "no tape" for "a tape
    that evaluates to nothing". Callers are expected to catch this and fall back
    to their existing path; the only expected trigger is :class:`CustomCall`,
    whose body is an opaque JAX callable by contract (``modeling/core.py``) and
    therefore has no tape equivalent.
    """


#: Ceiling on the number of tape nodes one lowering may materialize.
#:
#: The array path trades one vectorized XLA op for ``size`` Python-level
#: ``NlExpr`` objects, so a dense enough body is cheaper on the JAX path however
#: correct the tape is. Refusing past this point degrades that one model to the
#: JAX evaluator — the sound direction, and the same fallback every other refusal
#: here takes.
#:
#: 2_000_000 by measurement, not by feel. ``dm.sum(x * x)`` over a shape-(N,)
#: variable builds ``2N + 1`` nodes; timed at N = 1e3, 1e4, 1e5, 5e5 and 1e6 the
#: rate is **0.10 us/node**, flat (0.111, 0.097, 0.099, 0.106, 0.101) — linear,
#: with no knee. So the cap bounds one build at ~0.2 s and ~2e6 live Python
#: objects; every model in the corpus is orders of magnitude below it, and a body
#: dense enough to reach it is one whose vectorized form is the right shape
#: anyway. The number is a blowup guard, not a tuning parameter: a dense
#: 3000x3000 matmul would be 1.8e7 nodes.
_MAX_TAPE_NODES = 2_000_000


class _Budget:
    """Running count of materialized tape nodes, shared across one lowering."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def charge(self, n: int, what: str) -> None:
        self.count += int(n)
        if self.count > _MAX_TAPE_NODES:
            raise UnsupportedForTape(
                f"lowering {what} would materialize more than {_MAX_TAPE_NODES} tape "
                "nodes; a body this dense is cheaper on the vectorized path"
            )


def _wrap_scalar(node: Any) -> np.ndarray:
    """A single tape node as a 0-d object array.

    ``np.asarray(node, dtype=object)`` would do the same for ``NlExpr``, but only
    because it defines neither ``__len__`` nor ``__iter__``; building the array
    explicitly does not depend on that.
    """
    out = np.empty((), dtype=object)
    out[()] = node
    return out


def _const_array(value: Any, E: Any, budget: _Budget) -> np.ndarray:
    """A numeric constant (scalar or array) as an object array of ``const_`` nodes.

    A non-numeric or ragged value is a representability limit, not a bug: numpy
    raises ``ValueError``/``TypeError`` there, neither of which
    :func:`try_compile` catches, so it would escape the fallback and crash the
    caller instead of degrading to JAX.
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise UnsupportedForTape(
            f"non-numeric constant of type {type(value).__name__} ({exc})"
        ) from exc
    if arr.ndim == 0:
        return _wrap_scalar(E.const_(float(arr[()])))
    budget.charge(arr.size, "an array constant")
    out = np.empty(arr.shape, dtype=object)
    flat_in = arr.reshape(-1)
    flat_out = out.reshape(-1)
    for k in range(arr.size):
        flat_out[k] = E.const_(float(flat_in[k]))
    return out


def _map(fn: Any, *arrays: np.ndarray) -> np.ndarray:
    """Apply ``fn`` elementwise over broadcast object arrays — ONE OR TWO of them.

    ``np.frompyfunc`` returns a bare object (not a 0-d array) when every input is
    0-d, so the result is re-wrapped; ``np.asarray(..., dtype=object)`` on an
    ``NlExpr`` gives the 0-d array wanted.

    The arity is not free: ``np.frompyfunc`` refuses past 64 operands, and its
    ``ValueError`` is not one :func:`try_compile` catches, so it escapes the JAX
    fallback and crashes the caller. Every call site is a fixed-arity DAG operator
    (unary or binary); a variadic reduction belongs in :func:`_sum_along`, which
    loops instead. ``UnsupportedForTape`` here rather than that ``ValueError``, so
    a future variadic caller degrades instead of crashing.
    """
    if len(arrays) > 2:
        raise UnsupportedForTape(
            f"_map over {len(arrays)} operands; np.frompyfunc caps at 64 and this "
            "path is for fixed-arity operators only"
        )
    out = np.frompyfunc(fn, len(arrays), 1)(*arrays)
    return np.asarray(out, dtype=object)


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Start of ``var`` in the flat ``x`` vector.

    Delegates to the model's memoized prefix-sum table, so per-leaf resolution
    during the DAG walk is O(1). A linear scan here would be O(n) per leaf and
    O(n^2) over the build — the regression issue #654 fixed, and the reason
    ``dag_compiler`` routes through the same table.
    """
    return model._flat_var_offset(var)


def _static_scalar_slot(expr: IndexExpression, model: Model) -> int | None:
    """Flat ``x`` index of ``expr``, or ``None`` when it is not one static scalar slot.

    ``x[i]`` / ``y[i, j]`` on a shaped :class:`Variable` names a *single* entry of
    the flat vector, so it lowers to an ordinary scalar tape variable — no array
    machinery required. The identity it rests on is the JAX path's own layout:
    ``_relax/dag_compiler`` materializes a variable as ``x_flat[off : off + size]
    .reshape(shape)`` (C order) and then applies ``a[index]``. For a full-rank
    all-integer index that composition is exactly
    ``x_flat[off + ravel_multi_index(index, shape)]``.

    This is a **fast path, not a gate**. Returning ``None`` sends the caller to
    the general array lowering, which materializes the base and indexes it with
    numpy; the point of resolving here is that it costs O(1) instead of O(size)
    per leaf, and scalar indexing is the overwhelmingly common form. Returning
    ``None`` is therefore always safe — it can cost time, never correctness.

    The forms it declines:

    * a **slice or partial** index (``x[1:]``, ``y[0]`` on a 2-D ``y``), which
      names many slots rather than one;
    * a **non-Variable base** (``(x + y)[0]``, or a chained ``x[0][1]``), which
      has no flat slot to name at all;
    * a **non-integer / exotic** index (boolean mask, fancy index, ``Ellipsis``,
      a symbolic index);
    * an **out-of-range** index. numpy raises where ``jnp`` silently clamps, so
      the general path will refuse it too rather than resolve it to a slot the
      JAX path would not have used.

    ``bool`` is rejected explicitly: it is a subclass of ``int`` in Python, but
    numpy reads ``x[True]`` as a mask, not as ``x[1]``.

    Deliberately arithmetic rather than probe-based. Indexing an
    ``np.arange(size).reshape(shape)`` probe would match numpy semantics by
    construction, but allocating one per index node is O(size) per leaf and
    O(size·leaves) over the build — the same quadratic root-setup cost issue #654
    removed from :func:`_compute_var_offset`.
    """
    base = expr.base
    if not isinstance(base, Variable):
        return None
    shape = tuple(base.shape or ())
    if not shape:
        return None

    index = expr.index
    idx = index if isinstance(index, tuple) else (index,)
    if len(idx) != len(shape):
        # Partial indexing leaves an array; over-indexing is invalid. Either way
        # this is not one slot.
        return None

    flat = 0
    for i, dim in zip(idx, shape):
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            return None
        i = int(i)
        if i < 0:
            i += dim
        if not 0 <= i < dim:
            return None
        flat = flat * dim + i
    return _compute_var_offset(base, model) + flat


def compile_to_nl_array(expr: Expression, model: Model) -> np.ndarray:
    """Compile ``expr`` to a numpy object array of ``pounce.NlExpr`` tape nodes.

    The array carries ``expr``'s own shape (0-d for a scalar), and each element
    supports ``.eval(x)`` / ``.gradient(x)`` for a flat ``x`` ordered exactly like
    the JAX path's ``x_flat``, so the two are directly comparable point-for-point.

    ``result.reshape(-1)`` is the row order the JAX evaluator concatenates
    (``jnp.reshape(body, (-1,))``), which is what makes an array-valued constraint
    body's rows line up across the two backends.

    Raises:
        UnsupportedForTape: if the DAG contains a node with no tape equivalent.
    """
    import pounce

    E = pounce.NlExpr
    # id(node) -> object array. The DAG is a DAG, not a tree: a node reachable by
    # k references must be built once, or a linear DAG lowers in time exponential
    # in its sharing depth (the same trap `dag_compiler` documents as issue #383).
    memo: dict[int, Any] = {}
    return _lower(expr, model, E, memo, _Budget())


def compile_to_nl_expr(expr: Expression, model: Model) -> Any:
    """:func:`compile_to_nl_array` for a scalar ``expr``, returning the bare node.

    An array-valued ``expr`` is refused here rather than silently reduced: the
    callers of this entry point (objective, separation tangent, Gauss-Newton
    residual) each need exactly one scalar, and picking an element or a sum for
    them would compute a different function.
    """
    arr = compile_to_nl_array(expr, model)
    if arr.size != 1:
        raise UnsupportedForTape(
            f"expression is array-valued (shape {arr.shape}); this caller needs a scalar"
        )
    return arr.reshape(-1)[0]


def _children(expr: Expression, model: Model) -> tuple[Expression, ...]:
    """The sub-expressions :func:`_lower_uncached` will ``rec()`` into, in order.

    Kept deliberately in lockstep with the ``rec`` calls below — including
    :class:`IndexExpression`, where the base is lowered *only* when the static
    scalar-slot fast path misses. Pre-lowering it unconditionally would rebuild
    the whole base array for every ``x[i]`` leaf, which is the O(size * leaves)
    cost issue #654 removed.

    A node type missing here is not a correctness bug: :func:`_lower` still
    reaches its children through ``rec``, only recursively. It costs depth
    robustness for that shape, nothing else.
    """
    if isinstance(expr, BinaryOp):
        return (expr.left, expr.right)
    if isinstance(expr, UnaryOp):
        return (expr.operand,)
    if isinstance(expr, FunctionCall):
        return tuple(expr.args)
    if isinstance(expr, SumExpression):
        return (expr.operand,)
    if isinstance(expr, SumOverExpression):
        return tuple(expr.terms)
    if isinstance(expr, IndexExpression):
        return () if _static_scalar_slot(expr, model) is not None else (expr.base,)
    if isinstance(expr, MatMulExpression):
        return (expr.left, expr.right)
    return ()


def _lower(
    expr: Expression, model: Model, E: Any, memo: dict[int, Any], budget: _Budget
) -> np.ndarray:
    """Lower ``expr``, memoizing every node, with an EXPLICIT stack.

    Not recursion. An ``.nl`` file's objective is one left-deep operator chain
    with a term per variable, so the DAG's depth grows with the model rather than
    with the modeler's nesting: ``squfl015-060``'s objective measures 903 levels
    over 4556 nodes. ``_lower``/``_lower_uncached``/``rec`` is three frames per
    level, so CPython's 1000-frame default aborts the build at roughly 330 —
    ``RecursionError``, not ``UnsupportedForTape``, so it escaped
    :func:`try_build`'s fallback and took down the whole caller. That made
    ``make_evaluator`` unusable on the very models the OA path is for (#1063),
    while the JAX evaluator, which lowers the same DAG without a Python-frame
    chain, built it in 0.37 s.

    Raising the recursion limit is not the fix: the frames are real C-stack
    frames, so a limit high enough for a 100k-term ``.nl`` objective trades a
    clean exception for a hard interpreter crash. The traversal below is
    post-order — children complete before their parent — so every ``rec`` call
    inside :func:`_lower_uncached` is a memo hit and the frame chain stays flat.
    Sibling order matches the recursive form exactly, which keeps ``budget``
    charging (and so the node named in an overflow message) unchanged.
    """
    key = id(expr)
    hit = memo.get(key)
    if hit is not None:
        return cast(np.ndarray, hit)

    stack: list[tuple[Expression, bool]] = [(expr, False)]
    while stack:
        node, expanded = stack.pop()
        node_key = id(node)
        if node_key in memo:
            continue
        if not expanded:
            # Push the node back under its children so it is built after them;
            # `reversed` so siblings pop left-to-right, as `rec` visits them.
            stack.append((node, True))
            for child in reversed(_children(node, model)):
                if id(child) not in memo:
                    stack.append((child, False))
            continue
        built = _lower_uncached(node, model, E, memo, budget)
        if not isinstance(built, np.ndarray):
            # numpy returns a bare scalar, not a 0-d array, whenever every operand
            # of an elementwise op is 0-d -- so `x[0] * x[1]` comes back as an
            # `NlExpr`. Normalizing here rather than at each of the dozen
            # construction sites is what lets every branch above assume `.shape`
            # and `.size` exist.
            built = _wrap_scalar(built)
        memo[node_key] = built

    return cast(np.ndarray, memo[key])


def _lower_uncached(
    expr: Expression, model: Model, E: Any, memo: dict[int, Any], budget: _Budget
) -> np.ndarray:
    def rec(child: Expression) -> np.ndarray:
        return _lower(child, model, E, memo, budget)

    if isinstance(expr, Constant):
        return _const_array(expr.value, E, budget)

    if isinstance(expr, Variable):
        offset = _compute_var_offset(expr, model)
        if expr.size == 1:
            return _wrap_scalar(E.var(offset))
        # The flat-layout identity: `dag_compiler` materializes this variable as
        # `x_flat[offset : offset + size].reshape(shape)` in C order, so element
        # k of the C-order flattening is slot `offset + k`.
        shape = tuple(expr.shape or (expr.size,))
        budget.charge(expr.size, f"array variable {expr.name!r}")
        out = np.empty(shape, dtype=object)
        flat = out.reshape(-1)
        for k in range(expr.size):
            flat[k] = E.var(offset + k)
        return out

    if isinstance(expr, Parameter):
        # Parameters are constants at compile time, matching the legacy
        # `compile_expression` behaviour (the params-as-runtime-args variant has
        # no tape analogue -- a tape is built for fixed structure). Because the
        # value is BAKED IN, a caller holding a tape across a `Parameter.value`
        # re-bind gets stale derivatives; `_tape_nlp_evaluator` rebuilds on
        # change, and `evaluator_fingerprint` deliberately does NOT cover this.
        return _const_array(expr.value, E, budget)

    if isinstance(expr, BinaryOp):
        left, right, op = rec(expr.left), rec(expr.right), expr.op
        # numpy supplies the broadcasting, which is the same rule `jnp` applies to
        # the same two operands on the JAX path.
        budget.charge(_broadcast_size(left, right, op), f"a {op!r} over arrays")
        # `cast`: numpy types an object-dtype operator result as `Any`.
        if op == "+":
            return cast(np.ndarray, left + right)
        if op == "-":
            return cast(np.ndarray, left - right)
        if op == "*":
            return cast(np.ndarray, left * right)
        if op == "/":
            return cast(np.ndarray, left / right)
        if op == "**":
            return cast(np.ndarray, left**right)
        raise UnsupportedForTape(f"binary operator {op!r}")

    if isinstance(expr, UnaryOp):
        operand, op = rec(expr.operand), expr.op
        budget.charge(operand.size, f"a unary {op!r}")
        if op == "neg":
            return -operand
        if op == "abs":
            return _map(lambda a: _abs(E, a), operand)
        raise UnsupportedForTape(f"unary operator {op!r}")

    if isinstance(expr, FunctionCall):
        return _lower_function(expr, E, [rec(a) for a in expr.args], budget)

    if isinstance(expr, SumExpression):
        # An ARRAY reduction (``jnp.sum(operand, axis=...)``), not a list of scalar
        # terms -- ``.operand``/``.axis``, per ``dag_compiler``.
        operand = rec(expr.operand)
        if operand.size == 0:
            # `np.sum` of an empty object array returns the integer 0, not a tape
            # node, and that would poison every node built on top of it.
            raise UnsupportedForTape("SumExpression over an empty array has no tape node")
        return _sum_along(operand, expr.axis, E)

    if isinstance(expr, SumOverExpression):
        terms = [rec(t) for t in expr.terms]
        if not terms:
            return _wrap_scalar(E.const_(0.0))
        if len(terms) == 1:
            return terms[0]
        shape = terms[0].shape
        for t in terms[1:]:
            shape = np.broadcast_shapes(shape, t.shape)
        budget.charge(int(np.prod(shape, dtype=np.int64)) * len(terms), "a SumOverExpression")
        # `E.sum` over all terms at once rather than a `+` chain -- see
        # :func:`_sum_along` for why the chain shape is not merely slower.
        #
        # An explicit loop, NOT `_map` over `len(terms)` arrays: `np.frompyfunc`
        # refuses past 64 operands, and its `ValueError` is not one
        # `try_compile` catches, so a wide `dm.sum_over` escaped the fallback and
        # crashed the caller outright (adversarial suite,
        # `test_large_dense_jacobian_no_crash`, at 1100 terms).
        flat_terms = [np.broadcast_to(t, shape).reshape(-1) for t in terms]
        out = np.empty(shape, dtype=object)
        flat_out = out.reshape(-1)
        for k in range(flat_out.size):
            flat_out[k] = E.sum([ft[k] for ft in flat_terms])
        return out

    if isinstance(expr, IndexExpression):
        # Fast path FIRST: `x[i]` / `y[i, j]` on a shaped Variable names one flat
        # slot arithmetically, so the base's `size` nodes are never built. Falling
        # through to the general path instead would be correct but O(size) per
        # leaf and O(size * leaves) over the build -- the issue #654 cost.
        slot = _static_scalar_slot(expr, model)
        if slot is not None:
            return _wrap_scalar(E.var(slot))
        base = rec(expr.base)
        try:
            out = base[expr.index]
        except (IndexError, TypeError, ValueError) as exc:
            # Includes the out-of-range case, which numpy raises on and `jnp`
            # silently CLAMPS. Refusing hands it to the JAX path unchanged rather
            # than resolving it to a slot that path would not have used.
            raise UnsupportedForTape(
                f"IndexExpression: index {expr.index!r} on a "
                f"{type(expr.base).__name__} of shape {base.shape} is not "
                f"statically resolvable ({type(exc).__name__}: {exc})"
            ) from exc
        return np.asarray(out, dtype=object)

    if isinstance(expr, MatMulExpression):
        # The charge is inside `_matmul`, where the contracted dimension is known.
        return _matmul(rec(expr.left), rec(expr.right), E, budget)

    if isinstance(expr, CustomCall):
        # By contract (`modeling/core.py`) a dm.custom body is an opaque callable
        # that must be JAX-differentiable. There is no tape equivalent, and there
        # should not be a silent one -- relaxing an opaque callable needs AD
        # *through* it, which is exactly what the tape cannot do.
        raise UnsupportedForTape("CustomCall (dm.custom) has no tape equivalent")

    raise UnsupportedForTape(f"expression node {type(expr).__name__}")


def _matmul(left: np.ndarray, right: np.ndarray, E: Any, budget: _Budget) -> np.ndarray:
    """``left @ right`` for 1-D/2-D operands, one n-ary ``E.sum`` per output entry.

    Not ``np.matmul``: its object-dtype path accumulates each dot product with
    ``+``, so an inner dimension of K gives a chain of depth K and POUNCE refuses
    past ``max_depth`` — the same trap :func:`_sum_along` documents.

    Stacked (ndim > 2) operands are refused rather than looped over. ``jnp``
    broadcasts the leading batch dimensions there, and reproducing that rule by
    hand is exactly the kind of silent divergence this module refuses; nothing in
    discopt's modeling layer emits one today.
    """
    if left.ndim == 0 or right.ndim == 0:
        raise UnsupportedForTape("MatMulExpression with a scalar operand is not a matmul")
    if left.ndim > 2 or right.ndim > 2:
        raise UnsupportedForTape(
            f"MatMulExpression of stacked operands (shapes {left.shape} @ {right.shape}); "
            "only 1-D/2-D matmul is lowered"
        )
    # numpy's own promotion rule: a 1-D left operand is a row prepended with 1, a
    # 1-D right operand is a column appended with 1, and the added axis is dropped
    # from the result.
    a = left.reshape(1, -1) if left.ndim == 1 else left
    b = right.reshape(-1, 1) if right.ndim == 1 else right
    if a.shape[1] != b.shape[0]:
        raise UnsupportedForTape(
            f"MatMulExpression shape mismatch: {left.shape} @ {right.shape} "
            f"({a.shape[1]} vs {b.shape[0]} over the contracted axis)"
        )

    n, k = a.shape
    m = b.shape[1]
    # Exactly `n*m*(k+1)` nodes: k products and one n-ary sum per output entry.
    # Charging `left.size * right.size` instead would overstate this by a factor
    # of k (`(n*k)*(k*m)`) and refuse matmuls that are nowhere near the cap --
    # over-refusal is not the safe direction here, it just sends a model back to
    # the JAX path this work exists to leave.
    budget.charge(n * m * (k + 1), f"a {left.shape} @ {right.shape} MatMulExpression")
    out = np.empty((n, m), dtype=object)
    for i in range(n):
        row = a[i]
        for j in range(m):
            col = b[:, j]
            out[i, j] = E.sum([row[t] * col[t] for t in range(k)])

    if left.ndim == 1 and right.ndim == 1:
        return _wrap_scalar(out[0, 0])
    if left.ndim == 1:
        return cast(np.ndarray, out[0])
    if right.ndim == 1:
        return cast(np.ndarray, out[:, 0])
    return out


def _sum_along(arr: np.ndarray, axis: int | None, E: Any) -> np.ndarray:
    """``jnp.sum(arr, axis=axis)`` as ONE n-ary tape node per output element.

    Not ``np.sum``, which reduces object arrays with ``add.reduce`` — a
    left-leaning ``+`` chain of depth N. That is not merely slower: POUNCE
    **refuses** an expression nested past ``NlExpr.max_depth`` (10000), because a
    deeper tree overflows the stack when it is taped, walked or freed, which is a
    hard crash rather than an exception. Measured: ``dm.sum(x * x)`` on a
    50000-element variable raised ``ValueError: expression nesting would reach
    depth 10001``. ``NlExpr.sum`` builds one n-ary node of depth 1 whatever the
    term count, so the reduction's width stops mattering.

    ``ValueError`` is also not an exception :func:`try_compile` catches, so the
    chain form did not even degrade to JAX — it escaped the fallback entirely.
    """
    if axis is None:
        return _wrap_scalar(E.sum(arr.reshape(-1).tolist()))
    moved = np.moveaxis(arr, axis, -1)
    out = np.empty(moved.shape[:-1], dtype=object)
    flat_out = out.reshape(-1)
    flat_in = moved.reshape(-1, moved.shape[-1])
    for k in range(flat_out.size):
        flat_out[k] = E.sum(flat_in[k].tolist())
    return out


def _broadcast_size(left: np.ndarray, right: np.ndarray, op: str) -> int:
    """Element count of ``left op right``, refusing a shape pair that cannot broadcast.

    numpy would raise ``ValueError`` on the operation itself, which is NOT caught
    by :func:`try_compile` and so escapes the fallback and crashes the caller.
    """
    try:
        shape = np.broadcast_shapes(left.shape, right.shape)
    except ValueError as exc:
        raise UnsupportedForTape(
            f"operands of shape {left.shape} and {right.shape} do not broadcast "
            f"under {op!r} ({exc})"
        ) from exc
    return int(np.prod(shape, dtype=np.int64))


def _abs(E: Any, a: Any) -> Any:
    """``abs(a) = max(a, -a)``. The tape has ``max``; ASL/Ipopt treat the kink as a
    non-smooth event and route derivatives through the active branch."""
    return E.max(a, -a)


#: Argument floor for ``entropy``/``centropy``, matching `_relax/dag_compiler.py`
#: (``jnp.maximum(x, 1e-300)``). It regularizes the ``x -> 0+`` limit: the true
#: derivative of ``x*log(x)`` at 0 is ``-inf``, and both backends deliberately
#: report a large finite number instead so a solver evaluating a box pinned at
#: ``[0, 0]`` does not propagate a non-finite into a bound.
_XLOG_FLOOR = 1e-300

#: ``log(_XLOG_FLOOR)``, folded in Python rather than emitted as ``log(const)``.
#:
#: Not cosmetic. Below the floor the clamped branch is ``x * log(floor/y)``, and
#: routing that through a tape ``Log`` node makes its second derivative
#: ``-1/q**2`` with ``q = floor/y = 1e-300``, i.e. ``-1e600`` -- inf. Measured at
#: ``(x, y) = (1e-320, 1.0)``: the ``log(const/y)`` form returns
#: ``[[0, -inf], [-inf, nan]]`` where the truth ``[[0, -1], [-1, 1e-320]]`` is
#: entirely representable. Emitting the constant and subtracting ``log(y)``
#: returns exactly that truth. Same structural trap as the fused opcodes solve --
#: a composite in range built from an intermediate that is not.
_LOG_XLOG_FLOOR = math.log(_XLOG_FLOOR)


def _clamped_xlog(E: Any, x: Any, fused: Any, log_of_floor: Any) -> Any:
    """``entropy``/``centropy`` on the fused opcode, keeping the ``_XLOG_FLOOR`` clamp.

    The two lowerings differ only in which fused opcode carries the unclamped
    branch and what the log factor is once the argument is clamped, so both go
    through here. ``fused`` receives the (clamped) first argument; the authority
    for ``x < floor`` is ``x * log_of_floor`` -- ``x`` times a constant-in-``x``
    factor, NOT the fused op evaluated at the floor.

    That distinction is why this is a ``select`` and not just ``fused(max(x,
    floor))``. Below the floor the authority stays LINEAR in ``x`` with slope
    ``log(1e-300) = -690.78``, while ``xlogx(max(x, floor))`` would be constant
    with slope ``0``. The regularized-but-large derivative is the whole point of
    the clamp -- ``factorable_reform._try_entropy`` refuses only ``lo < 0.0``, so
    a box pinned at ``[0, 0]`` is admitted by design -- and reporting ``0`` there
    would tell the NLP that moving ``x`` off zero does not change the entropy,
    which is false.

    ``max(x, floor)`` still guards the fused argument. The tape's forward sweep
    evaluates EVERY slot, so at ``x < 0`` the inactive branch would otherwise
    compute ``xlogx(-1) = nan``; pounce's
    ``cond_does_not_leak_a_non_finite_from_its_inactive_branch`` measures that all
    three sweeps (``gradient_seed``, ``hessian_accumulate``,
    ``hessian_directional``) route around it, and the ``max`` means this lowering
    does not have to rely on that invariant to stay finite by construction.

    Cost: the ``select`` makes the node opaque to pounce's FBBT translator, which
    emits ``Opaque`` for ``Expr::Cond``. Nothing is actually lost -- ``E.max``
    lowers to ``Expr::MaxList``, which was ALREADY ``Opaque``, so the previous
    ``x * log(max(x, floor))`` had an opaque interior too and could not tighten
    through the log either way.
    """
    floor = E.const_(_XLOG_FLOOR)
    return E.select(E.compare("<", x, floor), x * log_of_floor, fused(E.max(x, floor)))


def _require_fused_xlog(E: Any, name: str) -> None:
    """Refuse ``entropy``/``centropy`` on a POUNCE predating the fused opcodes.

    ``pounce_usable()`` gates on ``NlExpr`` existing at all (pounce #470); these
    two opcodes landed later (pounce #489), so a build in between imports fine and
    then raises ``AttributeError`` deep inside a compile. Raise the
    ``UnsupportedForTape`` the caller already handles, which degrades this one
    model to the JAX evaluator rather than failing the solve.

    Deliberately NOT a fallback to the old ``x * log(x)`` lowering: that lowering
    is the defect this replaces, and quietly routing back to it would turn a stale
    extension into wrong derivatives instead of a visible capability miss.
    """
    missing = [n for n in ("xlogx", "centropy") if not hasattr(E, n)]
    if missing:
        raise UnsupportedForTape(
            f"{name} needs the fused NlExpr.{'/'.join(missing)} opcode(s) from "
            "pounce #489; rebuild the pounce extension to use the tape here"
        )


#: Below this ``|a|``, ``log1p`` uses a truncated series rather than Kahan's
#: compensated form. Both the term count and the crossover are set by the SECOND
#: derivative, which is the binding constraint here -- the value and gradient are
#: at one ulp across this whole region either way.
#:
#: The series runs to ``a**5/5``, not ``a**4/4``. Truncating at ``a**4`` leaves a
#: second derivative of ``-1 + 2a - 3a**2`` against a true expansion
#: ``-1 + 2a - 3a**2 + 4a**3 - ...``, an error of ``~4a**3``; the extra term moves
#: the leading error to ``~5a**4``. Measured at a=5e-5: 5.00e-13 before, 7.61e-19
#: after.
#:
#: The crossover is 5e-4 by MEASUREMENT, not algebra. Kahan's second-derivative
#: error decays like ``eps/a`` (its ``u - 1`` differs from ``a`` by a rounding gap
#: that double differentiation amplifies) while the series' grows like ``5a**4``,
#: so the worst case over the line is minimized where they cross. Scanning
#: candidate crossovers over a 69-point grid spanning 1e-17..1, worst-case
#: relative Hessian error was:
#:
#:     1e-4 -> 1.092e-11   2e-4 -> 4.552e-13   5e-4 -> 2.281e-13
#:     1e-3 -> 5.001e-13   2e-3 -> 5.007e-11
#:
#: with value/gradient flat at 1.336e-16 / 2.979e-16 for every candidate up to
#: 1e-3 and degrading at 2e-3 (3.002e-15 / 1.773e-14) -- so 5e-4 is the minimax
#: point and the scan would have shown a bad trade if one existed.
_LOG1P_TAYLOR = 5e-4


def _log1p(E: Any, a: Any) -> Any:
    """``log1p(a)`` accurately for small ``|a|``; the tape has no ``log1p`` opcode.

    ``log(1 + a)`` loses every significant digit once ``a`` falls below the
    rounding gap of 1.0: measured, ``a = 1e-17`` returns exactly ``0.0`` where
    ``jnp.log1p`` returns ``1e-17``. This is Kahan's compensated form — with
    ``u = 1 + a`` computed in floating point, ``log(u) / (u - 1)`` is the
    correction factor for the representation error in ``u``, and it is accurate
    precisely where the naive form is not.

    ``select`` is a data-flow node: BOTH arms are evaluated and both contribute
    partials to the reverse sweep, so it is not enough for the selected arm to be
    finite — each arm must be fed an argument inside its own safe range. Two
    guards, both load-bearing, both from a measurement:

    * ``d_safe`` is never zero, so the correction never forms ``0/0``.
    * the Kahan arm sees ``a_small``, clamped to ``[-0.5, 0.5]``. Applying the
      correction at large ``a`` overflows in the *derivative*, not the value: the
      quotient rule squares ``d``, and at ``a = 1e300`` that term is ``1e600 ->
      inf``, which silently drops the second term and returned a gradient of
      6.918e-298 against a true ``1e-300``. Bounding the arm's argument bounds
      ``d <= 1.5``, so ``d**2`` cannot overflow.

    Outside the small range the naive ``log(1 + a)`` is already accurate, and it
    is likewise fed a constant when it is not the selected arm.
    """
    one = E.const_(1.0)
    small = E.compare("<", _abs(E, a), E.const_(0.5))

    a_small = E.select(small, a, E.const_(0.0))
    u = one + a_small
    at_one = E.compare("==", u, one)
    d_safe = E.select(at_one, one, u - one)
    # `u == 1` means `a` vanished entirely in the addition, so `log(u)` is 0 and
    # the correction would return 0 rather than `a`. That arm returns `a_small`
    # unchanged -- which IS log1p to full precision once `a` is below the
    # rounding gap of 1.0.
    kahan = E.select(at_one, a_small, a_small * E.log(u) / d_safe)

    naive = E.log(E.select(small, E.const_(2.0), one + a))
    kahan_or_naive = E.select(small, kahan, naive)

    # Below `_LOG1P_TAYLOR`, a truncated series -- NOT the Kahan form. Kahan
    # corrects the VALUE; it does not correct the derivatives AD then takes of
    # it, because `u - 1` differs from `a` in floating point and double
    # differentiation amplifies that gap. Measured against the analytic
    # `-1/(1+a)**2`, the arms above give a second derivative of exactly 0.0 at
    # a=1e-17 (the `at_one` arm is LINEAR, so it has no curvature at all) and
    # -1.0039 at a=1e-13, against a true -1. The series is accurate in all three
    # orders at once: its second derivative is `-1 + 2a - 3a**2 + 4a**3`, which
    # is the leading expansion of the true `-1/(1+a)**2`. See `_LOG1P_TAYLOR` for
    # why it runs to `a**5/5` -- the SECOND derivative sets the term count, not
    # the value.
    #
    # `a_tiny` is clamped for the usual reason -- both arms of a `select`
    # evaluate, and `a**4` at a=1e300 is `inf`, which would poison the sweep.
    a_tiny = E.select(E.compare("<", _abs(E, a), E.const_(_LOG1P_TAYLOR)), a, E.const_(0.0))
    taylor = a_tiny * (
        one
        - a_tiny
        * (
            E.const_(1.0 / 2.0)
            - a_tiny
            * (E.const_(1.0 / 3.0) - a_tiny * (E.const_(1.0 / 4.0) - a_tiny / E.const_(5.0)))
        )
    )
    return E.select(E.compare("<", _abs(E, a), E.const_(_LOG1P_TAYLOR)), taylor, kahan_or_naive)


def _sign(E: Any, a: Any) -> Any:
    """``sign(a)`` via nested selects; zero maps to 0.0 as in ``jnp.sign``.

    ``compare`` takes the operator FIRST (``compare(op, lhs, rhs)``). Passing it
    last raises ``TypeError``, not ``UnsupportedForTape``, so it escapes
    ``try_compile``'s fallback and crashes the caller.
    """
    zero = E.const_(0.0)
    pos = E.compare(">", a, zero)
    neg = E.compare("<", a, zero)
    return E.select(pos, E.const_(1.0), E.select(neg, E.const_(-1.0), zero))


def _lower_function(expr: FunctionCall, E: Any, args: list, budget: _Budget) -> np.ndarray:
    """Lower a ``FunctionCall``. 20 operators are native to the tape; the rest are
    exact rewrites into operators that are.

    Every operator here except ``prod`` and ``norm*`` is *elementwise*, so it maps
    over the object arrays its arguments carry; the two reductions fold instead.
    """
    name = expr.func_name

    def arg0() -> Any:
        """First argument, after ``_require`` has proven it exists."""
        return args[0]

    charged = sum(a.size for a in args) or 1
    budget.charge(charged, f"function {name!r}")

    native_unary = {
        "exp",
        "log",
        "log10",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "erf",
    }
    if name in native_unary:
        _require(args, 1, name)
        return _map(getattr(E, name), arg0())

    if name in ("min", "max", "atan2"):
        _require(args, 2, name)
        return _map(getattr(E, name), args[0], args[1])

    # --- exact rewrites (no tape opcode, but expressible) ---------------------
    if name == "abs":
        _require(args, 1, name)
        return _map(lambda a: _abs(E, a), arg0())
    if name == "sign":
        _require(args, 1, name)
        return _map(lambda a: _sign(E, a), arg0())
    if name == "log1p":
        _require(args, 1, name)
        return _map(lambda a: _log1p(E, a), arg0())
    if name == "log2":
        _require(args, 1, name)
        return _map(lambda a: E.log(a) * E.const_(1.0 / math.log(2.0)), arg0())
    if name == "sigmoid":
        # Branch-stable form. `t = exp(-|a|)` is in (0, 1] for EVERY input, so
        # neither arm can overflow, and each arm is the algebraically exact
        # sigmoid on its own side:
        #
        #     a >= 0:  1/(1+exp(-a))  = 1/(1+t)
        #     a <  0:  exp(a)/(1+exp(a)) = t/(1+t)
        #
        # The `|a|` kink at 0 cancels rather than leaking into the derivative:
        # each arm pairs with the matching sign of `dt/da`, both arms give 1/2
        # and 1/4 there, so the result is smooth across the switch.
        #
        # The naive `1/(1+exp(-a))` this replaces was NOT safe, though its first
        # two orders looked it. A prior version of this comment claimed "measured
        # over -1e300..1e300: zero non-finite values, zero non-finite gradients"
        # and concluded "do not harden this one". That measurement was real but
        # only covered orders 0 and 1. At order 2 the left tail is where it dies:
        # `exp(745)` overflows, the quotient rule then forms `inf/inf`, and the
        # LAGRANGIAN HESSIAN -- which the NLP subsolve actually consumes --
        # came back `nan` at a=-745 and SIGN-FLIPPED at a=-300 (-5.148e-131
        # against a true +5.148e-131). `dm.sigmoid` is the SIGMOID activation in
        # all four `nn/formulations/`, so that is a live user path.
        #
        # The upper-tail accuracy that motivated keeping the naive form IS
        # preserved: at a=40 this still gives 4.248e-18 for the derivative where
        # `jax.nn.sigmoid` underflows to 0.0, because arm 1 is `1/(1+t)` with the
        # same `t`. Rewriting instead to `0.5*(1 + tanh(a/2))` would still be
        # wrong -- `tanh(-20)` rounds to exactly -1.0, giving 0.0 for a value
        # whose true magnitude is 4e-18.
        _require(args, 1, name)

        def _sigmoid(a: Any) -> Any:
            one = E.const_(1.0)
            t = E.exp(-_abs(E, a))
            return E.select(E.compare(">=", a, E.const_(0.0)), one / (one + t), t / (one + t))

        return _map(_sigmoid, arg0())
    if name == "softplus":
        # `log(1 + exp(a))` OVERFLOWS: `exp(710)` is `inf`, so softplus(745)
        # returned `inf` where the true value is 745 (measured: 2 non-finite
        # values over the sampled domain). The shifted form never exponentiates
        # a positive argument, so `exp` is confined to (0, 1] and cannot
        # overflow. It also recovers the lower tail exactly -- at `a = -300`
        # this returns 5.148e-131, matching `jnp.logaddexp`, where the naive
        # form collapsed to 0.0.
        _require(args, 1, name)
        return _map(lambda a: E.max(a, E.const_(0.0)) + _log1p(E, E.exp(-_abs(E, a))), arg0())
    if name == "entropy":
        # x*log(x) -- discopt's DAG semantics, NOT the information-theory
        # convention -x*log(x). The authority is `_relax/dag_compiler.py`, which
        # this must reproduce bit-for-bit: `lambda x: x * jnp.log(jnp.maximum(x,
        # 1e-300))`. Getting the sign from the name instead of the reference
        # cost a silent factor of -1 (reldiff 2.0) that no corpus instance could
        # have caught -- `.nl` has no entropy opcode, so 316 MINLPLib instances
        # exercise this line zero times.
        #
        # The floor is part of the semantics, not a nicety. Without it, `x = 0`
        # gives `0 * log(0) = 0 * -inf = nan` for the value and `-inf` for the
        # derivative, where the authority returns -0.0 and -690.78. That point is
        # REACHABLE: `factorable_reform._try_entropy` refuses only `lo < 0.0`, so
        # a box whose lower bound is exactly 0 is admitted by design. The comment
        # above already named `jnp.maximum(x, 1e-300)` as the authority; the sign
        # was carried across in `08e1e0a1` and the clamp was not.
        #
        # Lowered onto the FUSED `xlogx` opcode (pounce #489), not onto
        # `x * log(x)`. This is a correctness requirement, not a speed one:
        # `(x log x)'' = 1/x` is finite for every positive x down to 1e-308 --
        # at x = 1e-299 it is 1e299, an ordinary number -- but every chain-rule
        # decomposition routes through `log''(x) = -1/x**2 = -1e598`, past
        # `f64::MAX`. A composite in range built from a factor out of range is
        # unreachable however carefully the product and log rules are written;
        # only an opcode that never forms `1/x**2` gets there. Measured: the
        # fused second derivative returns 1e299 where `x * log(x)` returns inf.
        _require(args, 1, name)
        _require_fused_xlog(E, name)
        return _map(
            lambda a: _clamped_xlog(E, a, lambda t: E.xlogx(t), E.const_(_LOG_XLOG_FLOOR)),
            arg0(),
        )
    if name == "centropy":
        # x*log(x/y), matching dag_compiler's GAMS centropy -- including the same
        # floor on the NUMERATOR only (`x * log(max(x, 1e-300) / y)`). Same
        # defect and same fix as entropy above: measured 3 non-finite values and
        # 4 non-finite gradients over the sampled domain before the clamp; 0 and
        # 1 after it.
        #
        # Lowered onto the FUSED `centropy` opcode for entropy's reason
        # (`d2/dx2` is `1/x`, unreachable through `log''`) plus one of its own:
        # `d2/dy2` is `x/y**2`, and the quotient rule materializes `y**2`, which
        # overflows for `|y| > 1.3e154` while `x/y**2` itself stays in range.
        # That was the KNOWN RESIDUAL this comment used to record: the gradient
        # of `centropy(1e300, 1e300)` came back `[1, nan]` because `y**2` is
        # 1e600. The fused rule computes `q = x/y` once and expresses every
        # second-order term as a division by `y`, never by `y**2`.
        #
        # The old note here warned against "fixing" it as
        # `log(max(x, floor)) - log(y)`, because that trades an overflow no real
        # model reaches for catastrophic cancellation at `x ~= y`, which models
        # hit routinely. That warning is now DISCHARGED rather than ignored: the
        # opcode's `ln_ratio` picks among three regimes -- `ln_1p((x-y)/y)` near
        # `x = y` (Sterbenz makes `x - y` exact for `y/2 <= x <= 2y`), `q.ln()`
        # for finite positive q, and the difference form only when the ratio
        # itself leaves f64 range. Measured at `x = 1e10 + 1, y = 1e10`: naive
        # error 8.27e-8, fused error exactly 0.
        _require(args, 2, name)
        _require_fused_xlog(E, name)
        return _map(
            lambda a, y: _clamped_xlog(
                E, a, lambda t: E.centropy(t, y), E.const_(_LOG_XLOG_FLOOR) - E.log(y)
            ),
            args[0],
            args[1],
        )
    if name == "signpower":
        # sign(x) * |x|**p -- the standard smooth-away-from-zero signed power.
        _require(args, 2, name)
        return _map(lambda a, p: _sign(E, a) * (_abs(E, a) ** p), args[0], args[1])
    if name == "prod":
        # NOT a variadic multiply. `dag_compiler` compiles prod as `jnp.prod(arg)`
        # -- ONE argument, which is an ARRAY, reduced to a scalar. Lowering it as
        # a `*` chain over `args` silently computed a different function
        # (measured: reldiff 0.90 on value, 1.70 on gradient). The fold below is
        # over the argument's ELEMENTS, which is that reduction.
        _require(args, 1, name)
        return _wrap_scalar(_fold(arg0(), lambda a, b: a * b, E, name))
    if name.startswith("norm"):
        # `jnp.linalg.norm(a, ord=p)`, per `dag_compiler`. For a 1-D argument that
        # is the vector p-norm and folds over elements. For 2-D it is the
        # INDUCED/matrix norm -- ord=2 is the largest singular value, ord=1 the
        # max column sum -- which is not a fold, and is refused below rather than
        # approximated by the entrywise norm it is not.
        _require(args, 1, name)
        return _wrap_scalar(_lower_norm(name, E, arg0()))

    raise UnsupportedForTape(f"function {name!r}")


def _fold(arr: np.ndarray, combine: Any, E: Any, what: str) -> Any:
    """Left-fold ``combine`` over the C-order elements of ``arr``.

    Only for the two reductions with no n-ary opcode — ``prod`` (a ``*`` chain)
    and ``norminf`` (a ``max`` chain). The chain's depth *is* its element count,
    so the array's width is checked against ``NlExpr.max_depth`` up front rather
    than letting POUNCE raise a ``ValueError`` that :func:`try_compile` does not
    catch and that would therefore escape the fallback instead of degrading to
    JAX. Sums route through :func:`_sum_along` and have no such limit.

    Refuses an empty array rather than inventing an identity element: the tape
    node for "the product of nothing" would be a constant the JAX path never
    produced (``jnp.prod`` of an empty array is 1.0, but reaching that here means
    the DAG shape was not what the caller thought).
    """
    flat = arr.reshape(-1)
    if flat.size == 0:
        raise UnsupportedForTape(f"{what} over an empty array has no tape node")
    limit = int(E.max_depth)
    if flat.size >= limit:
        raise UnsupportedForTape(
            f"{what} over {flat.size} elements would nest past NlExpr.max_depth "
            f"({limit}); this reduction has no n-ary tape opcode"
        )
    acc = flat[0]
    for k in range(1, flat.size):
        acc = combine(acc, flat[k])
    return acc


def _lower_norm(name: str, E: Any, arr: np.ndarray) -> Any:
    """``jnp.linalg.norm(arr, ord=p)`` for a vector ``arr``."""
    suffix = name[len("norm") :]
    if suffix == "inf":
        ord_p: float = math.inf
    elif suffix == "":
        ord_p = 2.0
    else:
        try:
            ord_p = float(suffix)
        except ValueError as exc:
            raise UnsupportedForTape(f"unsupported norm order: {name!r}") from exc

    if arr.ndim > 1:
        raise UnsupportedForTape(
            f"{name} of a {arr.ndim}-D argument is jnp.linalg.norm's MATRIX norm "
            "(induced/spectral), which is not a fold over elements"
        )
    # `jnp.linalg.norm` of a 0-d argument raises; a 1-element vector is fine and
    # `reshape(-1)` makes the 0-d case into one, matching nothing -- so refuse it
    # explicitly instead of silently accepting a shape jnp would have rejected.
    if arr.ndim == 0:
        raise UnsupportedForTape(f"{name} of a scalar argument is not a vector norm")

    absolutes = _map(lambda a: _abs(E, a), arr)
    if ord_p == math.inf:
        return _fold(absolutes, lambda a, b: E.max(a, b), E, name)
    if ord_p == 1.0:
        return E.sum(absolutes.tolist())
    if ord_p == 2.0:
        # `E.sqrt` rather than `** 0.5`: same value, but the tape has a first-class
        # sqrt derivative rule where a general power has to route through
        # `exp(p*log(x))`-style handling at x = 0.
        return E.sqrt(E.sum(_map(lambda a: a * a, absolutes).tolist()))
    # `sum(|x_i|**p) ** (1/p)`. Built from |x| rather than x so a non-integer p
    # never raises a negative base to a fractional power.
    powered = _map(lambda a: a ** E.const_(ord_p), absolutes)
    return E.sum(powered.tolist()) ** E.const_(1.0 / ord_p)


def _require(args: list, n: int, name: str) -> None:
    if len(args) != n:
        raise UnsupportedForTape(f"{name} expects {n} argument(s), got {len(args)}")


def try_compile(expr: Expression, model: Model) -> Any | None:
    """``compile_to_nl_expr`` returning ``None`` instead of raising.

    For call sites that already have a working fallback and want the tape only
    when it is available. The distinction matters: this returns ``None`` only for
    *representability*, never for a numerical failure, so a caller cannot confuse
    "unsupported operator" with "bad point".
    """
    try:
        return compile_to_nl_expr(expr, model)
    except UnsupportedForTape:
        return None
