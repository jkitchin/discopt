"""Build the POUNCE AD tape from the Rust expression arena instead of the Python DAG.

Measured motivation (#1215): at foundation scale (100k rows) the tape build is
**59% of the whole build -> solve-ready pipeline**, and **93-94% of that is
:mod:`discopt._nl_expr_compiler` walking the Python expression DAG**
(57.9-64.6 us/row) while POUNCE's own ``build_nl_problem`` is 4.1-4.4 us/row and
flat. ``model_to_repr`` has *already* walked the same structure into a typed Rust
arena moments earlier, so the Python walk is a second pass over data that is
already in a form a machine can scan.

``PyModelRepr.tape_program()`` returns that arena as six flat numpy arrays. The
arena is append-only, so a child's ``ExprId`` is always lower than its parent's:
lowering is a single forward **array scan** with a ``cache[i]`` table -- no
recursion, no ``id()``-keyed memo, and no exponential blow-up on shared
subtrees.

Why this is allowed to be silent-fallback rather than silent-substitute
----------------------------------------------------------------------

A tape is the derivative source for the whole NLP path. A tape that is merely
*plausible* is worse than no tape: it yields wrong gradients, hence wrong
incumbents, hence a wrong certificate (CLAUDE.md §1). So this module never
guesses:

* ``tape_program`` emits ``OP_UNSUPPORTED`` -- never a skip -- for every arena
  node the flat encoding does not cover (array variables, matmul, axis sums,
  constant arrays, shaped parameters, non-integer indexing, ``MathFunc``\\ s
  outside the mapped set). Unsupportedness propagates up through every parent.
* :func:`try_build_arena_tape` returns ``None`` -- not a partial tape -- if the
  objective or *any* constraint root is unsupported, if the model carries
  builder-resident rows (whose arena order differs from the Python order, see
  below), or if the arena's flat variable offsets disagree with the model's.
  The caller then walks the Python DAG exactly as before.

Bit-identity, not approximate agreement
---------------------------------------

The emission rules below mirror :mod:`discopt._nl_expr_compiler` node for node,
including its additive-chain flattening: a maximal ``+``/``-`` chain of at least
:data:`~discopt._nl_expr_compiler._FLATTEN_MIN_TERMS` leaves becomes one n-ary
``E.sum`` in *left-to-right leaf order*, and a shorter chain keeps its original
binary nesting. That threshold is imported rather than repeated so the two paths
cannot drift. Matching it is not cosmetic: emitting a binary ``+`` chain where
the Python path emits ``E.sum`` would rebuild the depth-N nesting that
``NlExpr.max_depth`` refuses at 10 000 terms, and flattening where Python does
not would reassociate the sum (``a+(b+c)`` is not ``(a+b)+c`` in IEEE).

Row order
---------

``model_to_repr`` clones the builder's constraint vector first and appends the
expression-path rows after it (``expr_bindings.rs``), whereas the evaluator
enumerates ``(*model._constraints, *model._builder_linear_constraints())`` --
the opposite order. Rather than reorder on an invariant that is only true today,
this module refuses any model carrying builder rows, so arena index ``i`` is
``model._constraints[i]`` by construction. Builder rows are the linear
fast-construction path and are cheap to lower on the Python side anyway.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional

import numpy as np

from discopt._nl_expr_compiler import _FLATTEN_MIN_TERMS
from discopt.export._common import has_builder_only_constraint_rows

# Opcodes -- must track `crates/discopt-python/src/expr_bindings.rs::tape_program`.
OP_UNSUPPORTED = 0
OP_CONST = 1
OP_VAR = 2
OP_ADD = 3
OP_SUB = 4
OP_MUL = 5
OP_DIV = 6
OP_POW = 7
OP_NEG = 8
OP_ABS = 9
OP_SUMOVER = 10
OP_FUNC_BASE = 20

#: ``math_func_code`` order in ``expr_bindings.rs``. Code 99 is that function's
#: catch-all for a ``MathFunc`` it does not map; it is deliberately absent here so
#: it resolves to "unsupported" rather than to some neighbouring function.
# MathFunc index -> the NlExpr method of the same meaning. Every name here must
# be a REAL method on `pounce.NlExpr`: the scan calls it through `getattr`, so a
# name that does not exist raises `AttributeError` mid-solve instead of falling
# back. `test_1215_arena_tape_func_table.py` asserts the whole table against the
# live class, which is how the `log2` entry below was caught.
#
# `Log2` (index 2) is deliberately absent: NlExpr has `log` and `log10` but no
# `log2`, so it is lowered in the scan the same way the Python compiler lowers
# it. Indices past the end (Acos, Tanh, Erf, ...) are simply not lowered here --
# `.get()` returns None and the model falls back to the Python DAG path.
_FUNC_LOG2 = 2

_FUNC_METHOD = {
    0: "exp",
    1: "log",
    3: "log10",
    4: "sqrt",
    5: "sin",
    6: "cos",
    7: "tan",
    8: "atan",
    9: "sinh",
    10: "cosh",
    11: "asin",
}

_LOG = logging.getLogger(__name__)

_ENV = "DISCOPT_ARENA_TAPE"


def arena_tape_enabled() -> bool:
    """Whether the arena lowering is used. ``DISCOPT_ARENA_TAPE=0`` opts out.

    Read at call time, not import time, so a test can flip it without reloading.
    The legacy Python-DAG path stays intact and is still the fallback for every
    model this lowering refuses, so the opt-out is a switch between two live
    paths, not a dead flag.
    """
    return os.environ.get(_ENV, "1") not in ("0", "false", "False")


def _num(obj: Any, attr: str) -> Any:
    """Read a ``PyModelRepr`` field that may be exposed as a property or a method."""
    v = getattr(obj, attr)
    return v() if callable(v) else v


class _Program:
    """The six flat arrays of the arena's tape encoding, n-ary operands sliced out.

    Built either from ``tape_program()`` (one slot per arena node; refuses array
    bodies) or from ``tape_program_expanded()`` (fully scalar, array bodies fanned
    out in Rust). Both emit the same opcodes and the same
    operands-have-lower-indices invariant, so :func:`lower` serves both.
    """

    __slots__ = ("op", "a", "b", "k", "_args_flat", "_args_ptr", "n")

    def __init__(self, arrays: Any) -> None:
        op, a, b, k, args_flat, args_ptr = arrays
        self.op = np.asarray(op)
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.k = np.asarray(k)
        self._args_flat = np.asarray(args_flat)
        self._args_ptr = np.asarray(args_ptr)
        self.n = int(self.op.shape[0])

    def args(self, i: int) -> np.ndarray:
        return self._args_flat[self._args_ptr[i] : self._args_ptr[i + 1]]


def _chain_leaves(prog: _Program, root: int) -> Optional[list[tuple[int, int]]]:
    """Signed leaves of the maximal ``+``/``-`` chain at *root*, left-to-right.

    The exact analogue of :func:`discopt._nl_expr_compiler._additive_chain`,
    including its ``None`` return for a chain shorter than
    :data:`_FLATTEN_MIN_TERMS` (so the caller falls through to ordinary binary
    lowering) and its explicit stack -- the shape being flattened is precisely
    the one deep enough to overflow a recursive walk.
    """
    op, a, b = prog.op, prog.a, prog.b
    out: list[tuple[int, int]] = []
    stack: list[tuple[int, int]] = [(root, 1)]
    while stack:
        nid, sign = stack.pop()
        o = op[nid]
        if o == OP_ADD or o == OP_SUB:
            rsign = sign if o == OP_ADD else -sign
            # push right first so the left subtree pops (and so lowers) first
            stack.append((int(b[nid]), rsign))
            stack.append((int(a[nid]), sign))
        else:
            out.append((sign, int(nid)))
    if len(out) < _FLATTEN_MIN_TERMS:
        return None
    return out


def lower(prog: _Program, roots: list[int], E: Any) -> list:
    """Lower the arena to a ``cache`` of tape nodes, ``None`` where unsupported.

    Two passes. The first walks *backwards* from the roots marking which nodes
    are actually reachable and, at each additive chain root, recording its
    flattened leaves and marking those (not the chain's interior nodes) as
    needed -- mirroring ``_nl_expr_compiler._children``, which descends straight
    to a chain's leaves so the interiors are never lowered. Because a child's id
    is always below its parent's, one backward sweep suffices.

    The second pass walks *forwards* and materialises each needed node; every
    operand it reads has a lower id and so is already in ``cache``.
    """
    n = prog.n
    op, a, b, k = prog.op, prog.a, prog.b, prog.k

    needed = np.zeros(n, dtype=bool)
    chains: dict[int, list[tuple[int, int]]] = {}
    for r in roots:
        needed[r] = True

    for i in range(n - 1, -1, -1):
        if not needed[i]:
            continue
        o = op[i]
        if o == OP_ADD or o == OP_SUB:
            ch = _chain_leaves(prog, i)
            if ch is not None:
                chains[i] = ch
                for _, leaf in ch:
                    needed[leaf] = True
                continue
        if a[i] >= 0:
            needed[a[i]] = True
        if b[i] >= 0:
            needed[b[i]] = True
        if o == OP_SUMOVER:
            for t in prog.args(i):
                needed[t] = True

    cache: list = [None] * n
    for i in range(n):
        if not needed[i]:
            continue
        o = int(op[i])

        if o == OP_UNSUPPORTED:
            continue
        if o == OP_CONST:
            cache[i] = E.const_(float(k[i]))
            continue
        if o == OP_VAR:
            cache[i] = E.var(int(k[i]))
            continue

        ch = chains.get(i)
        if ch is not None:
            # A separate flag rather than re-using `parts` as its own "dropped"
            # sentinel: one unsupported leaf drops the whole chain, and a list
            # that is sometimes `None` is both harder to read and untypeable.
            parts: list[Any] = []
            complete = True
            for sign, leaf in ch:
                p = cache[leaf]
                if p is None:
                    complete = False
                    break
                parts.append(p if sign > 0 else -p)
            if complete:
                cache[i] = E.sum(parts)
            continue

        if o == OP_SUMOVER:
            terms = [cache[t] for t in prog.args(i)]
            if any(t is None for t in terms):
                continue
            # Same degenerate-arity rules as `_nl_expr_compiler`: a sum over no
            # terms is the constant 0, a sum over one term is that term.
            if not terms:
                cache[i] = E.const_(0.0)
            elif len(terms) == 1:
                cache[i] = terms[0]
            else:
                cache[i] = E.sum(terms)
            continue

        if o in (OP_NEG, OP_ABS):
            arg = cache[a[i]]
            if arg is None:
                continue
            cache[i] = -arg if o == OP_NEG else abs(arg)
            continue

        if o >= OP_FUNC_BASE:
            fidx = o - OP_FUNC_BASE
            arg = cache[a[i]]
            if arg is None:
                continue
            if fidx == _FUNC_LOG2:
                # Exactly the form `_nl_expr_compiler` uses, because the two
                # paths must build the same tape: `log(a) * (1/ln 2)`.
                cache[i] = E.log(arg) * E.const_(1.0 / math.log(2.0))
                continue
            method = _FUNC_METHOD.get(fidx)
            if method is None:
                continue
            cache[i] = getattr(arg, method)()
            continue

        lhs, rhs = cache[a[i]], cache[b[i]]
        if lhs is None or rhs is None:
            continue
        if o == OP_ADD:
            cache[i] = lhs + rhs
        elif o == OP_SUB:
            cache[i] = lhs - rhs
        elif o == OP_MUL:
            cache[i] = lhs * rhs
        elif o == OP_DIV:
            cache[i] = lhs / rhs
        elif o == OP_POW:
            cache[i] = lhs**rhs

    return cache


def _offsets_agree(repr_: Any, model: Any) -> bool:
    """True if the arena's flat variable slots match the evaluator's ordering.

    ``E.var(slot)`` uses the slot ``tape_program`` resolved from the arena's own
    block offsets, while the evaluator's ``x`` vector is ``model._variables`` in
    declaration order. A disagreement would alias one variable onto another --
    wrong derivatives with no error -- so it is checked rather than assumed, and
    a mismatch falls back instead of raising: the Python path is still correct
    for such a model.
    """
    shapes = _num(repr_, "var_shapes")
    names = _num(repr_, "var_names")
    if len(shapes) != len(model._variables) or len(names) != len(model._variables):
        return False
    for sh, nm, var in zip(shapes, names, model._variables):
        size = int(np.prod(sh)) if len(sh) else 1
        # Size alone would not catch a PERMUTATION of equally-sized blocks, which
        # is the aliasing this guard exists to prevent, so the block name is
        # checked too.
        if size != int(var.size) or str(nm) != str(var.name):
            return False
    return True


def try_build_arena_tape(model: Any, E: Any) -> Optional[tuple[Any, list]]:
    """``(objective_node, constraint_nodes)`` from the arena, or ``None`` to fall back.

    ``None`` is returned -- never a partial or approximate tape -- when the model
    carries builder-resident rows (row-order mismatch, see the module docstring),
    when the arena's variable slots disagree with the evaluator's ordering, or
    when the objective or any constraint root reaches an unsupported arena node.

    The objective is returned *unnegated*; the caller applies the maximise
    negation exactly as it does on the Python path.
    """
    if not arena_tape_enabled():
        return None
    # Builder rows would land ahead of the expression rows in the arena while the
    # evaluator enumerates them after -- refuse rather than reorder.
    #
    # Asked in O(1) off `_builder_linear_blocks`. The obvious spelling,
    # `if model._builder_linear_constraints():`, MATERIALISES ONE `Constraint`
    # OBJECT PER ROW to answer a yes/no question; the same line in the `.nl`
    # writer's refusal measured at 7.12 us/row on a 20 000-row model, against 9.4
    # for the writer it was guarding (performance-plan.md §52).
    if has_builder_only_constraint_rows(model):
        return None
    if getattr(model, "_objective", None) is None:
        return None
    # A builder-resident objective (`add_linear_objective`) is folded into the
    # arena's objective root, but the evaluator lowers `model._objective.expression`
    # alone. Taking the arena root for such a model would silently OPTIMISE A
    # DIFFERENT FUNCTION, so refuse instead.
    from discopt.export._common import builder_objective

    if builder_objective(model) is not None:
        return None

    from discopt._rust import model_to_repr

    try:
        repr_ = model_to_repr(model, getattr(model, "_builder", None))
    except Exception as exc:  # noqa: BLE001 -- see below; deliberately broad
        # "This model has no arena representation" is how `convert_expr` reports
        # several distinct cases, each with its own exception type and wording:
        # `TypeError: Unknown expression type: CustomCall` for an opaque callable,
        # `ValueError: Unknown MathFunc: centropy` for an operator the core IR
        # does not carry, and more as the modelling layer grows. An allowlist of
        # message substrings was tried first and was wrong within one test run --
        # it caught the CustomCall wording and let `centropy` escape, turning a
        # model that used to solve into a crash.
        #
        # This is not a swallowed error (§7). Before this module existed the
        # evaluator never called `model_to_repr` at all, so a failure here is not
        # a regression being hidden -- it is new information about a model the
        # Python walk below still lowers correctly. And nothing is suppressed:
        # the solve path calls `model_to_repr` again, unguarded, so a genuine
        # defect in it still surfaces there with its own traceback. What is
        # caught is only this speculative attempt.
        #
        # `Exception`, not `BaseException`: a pyo3 `PanicException` derives from
        # `BaseException` and must keep propagating.
        _LOG.debug("arena tape declined; model_to_repr refused: %s: %s", type(exc).__name__, exc)
        return None
    if not _offsets_agree(repr_, model):
        return None

    con_ids = [int(c) for c in _num(repr_, "constraint_ids")]
    # One arena row per expression constraint: an array-valued body is ONE arena
    # node and MANY tape rows, so a count mismatch means the fan-out this path
    # cannot do is required. (Such a body also roots at an unsupported node, but
    # the count is checked first so the refusal names the real reason.)
    from discopt.modeling.core import Constraint

    expr_cons = [c for c in model._constraints if isinstance(c, Constraint)]
    if len(con_ids) != len(expr_cons):
        return None

    obj_id = int(_num(repr_, "objective_id"))
    prog = _Program(repr_.tape_program())
    if obj_id < 0 or obj_id >= prog.n:
        return None

    cache = lower(prog, [obj_id, *con_ids], E)
    obj = cache[obj_id]
    if obj is None:
        return None
    cons = [cache[cid] for cid in con_ids]
    if any(c is None for c in cons):
        return None
    return obj, cons


def try_build_expanded_tape(model: Any, E: Any) -> Optional[tuple[Any, list, list[int]]]:
    r"""``(objective, constraint rows, rows-per-constraint)`` from ``expand``, or ``None``.

    **This is the verification harness for** ``discopt_core::expand``\ **, not a
    production lowering path.** Nothing in the solve path calls it, deliberately.

    It exists because `expand` -- the Rust fan-out of an array-valued body into N
    scalar rows -- is written for the ``.nl`` writer, whose consumer is Rust, and
    its correctness has to be provable from Python. Lowering its output to POUNCE
    nodes here lets `issue1215_expanded_tape_differential.py` compare objective,
    gradient, every constraint value and the full Jacobian against the Python DAG
    walk, and pin the ROW ORDER, which a tolerance check on values alone would
    not catch.

    It is not used for the tape because it is **1.77x slower** there, measured:
    ``_nl_expr_compiler`` lowers array-at-a-time -- 242 ``_lower_uncached`` calls
    for 20 000 rows -- and lets numpy's ``frompyfunc`` create the per-element
    POUNCE nodes in C, whereas an expanded program must be consumed one scalar
    instruction at a time from Python. Routing the tape through here replaces a C
    loop with a Python one. See ``docs/dev/performance-plan.md`` §45.

    ``rows_per_constraint`` is returned rather than assumed: it is what would
    drive ``_constraint_flat_sizes``, which attributes duals, the row map and
    feasibility reports back to the ``Constraint`` each row came from.
    """
    if not arena_tape_enabled():
        return None
    # O(1), for the reason given in `try_build_arena_tape` above.
    if has_builder_only_constraint_rows(model):
        return None
    if getattr(model, "_objective", None) is None:
        return None
    from discopt.export._common import builder_objective

    if builder_objective(model) is not None:
        return None

    from discopt._rust import model_to_repr

    try:
        repr_ = model_to_repr(model, getattr(model, "_builder", None))
    except Exception as exc:  # noqa: BLE001 -- see try_build_arena_tape
        _LOG.debug("expanded tape declined; model_to_repr refused: %s", exc)
        return None
    if not _offsets_agree(repr_, model):
        return None

    try:
        (op, a, b, k, args_flat, args_ptr, obj_root, row_roots, rows_per) = (
            repr_.tape_program_expanded()
        )
    except ValueError as exc:
        # `expand` refuses rather than guessing -- an inexact fan-out is a
        # different model. Fall back to the Python walk, which is unchanged.
        _LOG.debug("expanded tape declined: %s", exc)
        return None

    prog = _Program((op, a, b, k, args_flat, args_ptr))
    roots = [int(obj_root), *(int(r) for r in row_roots)]
    cache = lower(prog, roots, E)
    obj = cache[int(obj_root)]
    if obj is None:
        return None
    cons = [cache[int(r)] for r in row_roots]
    if any(c is None for c in cons):
        return None
    return obj, cons, [int(v) for v in rows_per]
