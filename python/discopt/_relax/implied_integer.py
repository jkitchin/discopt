"""Provably-sound implied-integer detection.

A variable declared continuous may be *forced* to integer values by the model.
Marking such a variable integer is value-preserving (it cannot cut off any
feasible — hence any optimal — point) and lets the integer-product reformulation
(``integer_product_reform``) tighten bilinear terms that involve it. Marking a
*non*-implied variable integer would cut off feasible points and is the cardinal
correctness violation, so this detector uses only a **rigorously sound**
sufficient condition and is conservative everywhere else.

**Sound condition (integer-defining equality).** A variable ``x`` is integer at
every feasible point if there is a *linear* equality constraint

    Σ_j a_j x_j + c = 0      (sense "==")

with all ``a_j`` and ``c`` integer, ``x`` appearing with coefficient ``±1``, and
**every other** variable with a nonzero coefficient already known integer
(declared integer/binary, or itself proven implied-integer in an earlier round).
Then ``x = ∓(Σ_{j≠x} a_j x_j + c)`` is an integer combination of integers ⇒ ``x``
is integer.

This is exactly the structure the ``ex126x`` trim-loss models carry: e.g.
``x5 - x35 - 2·x36 - 4·x37 = 0`` with ``x35,x36,x37`` binary ⇒ ``x5`` integer.

Detection iterates to a fixpoint so chains (``x`` integer because ``y`` was just
proven integer) are caught. Range links like ``b ≤ x ≤ b+4`` (inequalities) are
**never** sufficient and are correctly ignored.

**Exact arithmetic (#1952).** Every test is exact: a coefficient within 1e-9 of an
integer is *not* an integer (``(1 + 2**-40)·x - y = 0`` does not make ``x``
integer), and a coefficient of 5e-10 is in the row (dropping it let ``x - y +
5e-10·w = 0`` mark ``x``). Exact tests on the float extraction are still not
enough, because the extractor rounds while it accumulates (``2**53·x + 0.5·x -
2**53·x + x - y`` extracts as ``{x: 1, y: -1}``; its exact ``x`` coefficient is
3/2). So the float extraction is only a prefilter, and each candidate row is
re-derived in ``Fraction`` arithmetic by :func:`_exact_linear_row`; the proof is
read off those exact coefficients, and a row outside the walker's grammar is
skipped.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from discopt._flat_index import resolve_scalar_slot
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    IndexExpression,
    Model,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

from .gdp_reformulate import _is_linear
from .problem_classifier import _extract_linear_coefficients_sparse, _NotLinearError


class _NotExactLinear(Exception):
    """The row is outside the exact walker's grammar; the caller skips it."""


def _exact_const(node) -> Fraction:
    """Exact rational value of a variable-free subtree (#1952).

    Every leaf is a binary64 ``Constant``; ``Fraction(float)`` is exact, and so is
    every ``+ - * /`` on Fractions, so the result is the exact value of the
    expression *as written*, not of its float evaluation.
    """
    if isinstance(node, Constant):
        v = node.value
        if v.ndim == 0 or v.size == 1:
            f = float(v.reshape(()))
            if not math.isfinite(f):
                raise _NotExactLinear("non-finite constant")
            return Fraction(f)
        raise _NotExactLinear("array constant")
    if isinstance(node, UnaryOp) and node.op == "neg":
        return -_exact_const(node.operand)
    if isinstance(node, BinaryOp) and node.op in ("+", "-", "*", "/"):
        lv, rv = _exact_const(node.left), _exact_const(node.right)
        if node.op == "+":
            return lv + rv
        if node.op == "-":
            return lv - rv
        if node.op == "*":
            return lv * rv
        if rv == 0:
            raise _NotExactLinear("division by zero")
        return lv / rv
    raise _NotExactLinear(f"non-constant node {type(node).__name__}")


def _is_var_free(node) -> bool:
    if isinstance(node, Constant):
        return True
    if isinstance(node, UnaryOp):
        return node.op == "neg" and _is_var_free(node.operand)
    if isinstance(node, BinaryOp):
        return _is_var_free(node.left) and _is_var_free(node.right)
    return False


def _exact_linear_row(body, model: Model, n: int) -> tuple[dict[int, Fraction], Fraction]:
    """Exact rational ``(terms, const)`` of a linear row body (#1952).

    The float extractor (``_extract_linear_coefficients_sparse``) rounds while it
    accumulates: ``0.1*x + 0.2*x + 0.7*x - y`` and ``2**53*x + 0.5*x - 2**53*x + x - y``
    both come out as exactly ``{x: 1.0, y: -1.0}``, although the exact coefficients
    of ``x`` are not 1 (the second is 3/2). An integrality proof read off those
    floats is a proof about a different row. This walker does the same sum in
    ``Fraction`` arithmetic, so the coefficients it returns are the exact values of
    the row as stored. It deliberately covers only the plain linear grammar; any
    other node raises :class:`_NotExactLinear` and the row is skipped, which can
    only under-detect.

    Iterative, like the float walker, because a ``.nl`` body is a left-leaning chain
    one node deep per term (#1064).
    """
    terms: dict[int, Fraction] = {}
    const = Fraction(0)

    def add(i: int, v: Fraction) -> None:
        if not 0 <= i < n:
            raise _NotExactLinear(f"slot {i} outside [0, {n})")
        terms[i] = terms.get(i, Fraction(0)) + v

    stack: list[tuple[object, Fraction]] = [(body, Fraction(1))]
    while stack:
        node, scale = stack.pop()
        if _is_var_free(node):
            const += scale * _exact_const(node)
            continue
        if isinstance(node, (Variable, IndexExpression)):
            slot = resolve_scalar_slot(node, model)
            if slot is None:
                raise _NotExactLinear("not a single scalar slot")
            add(slot, scale)
            continue
        if isinstance(node, UnaryOp) and node.op == "neg":
            stack.append((node.operand, -scale))
            continue
        if isinstance(node, BinaryOp):
            if node.op == "+":
                stack.append((node.right, scale))
                stack.append((node.left, scale))
                continue
            if node.op == "-":
                stack.append((node.right, -scale))
                stack.append((node.left, scale))
                continue
            if node.op == "*":
                if _is_var_free(node.left):
                    stack.append((node.right, scale * _exact_const(node.left)))
                    continue
                if _is_var_free(node.right):
                    stack.append((node.left, scale * _exact_const(node.right)))
                    continue
                raise _NotExactLinear("product of two variable expressions")
            if node.op == "/" and _is_var_free(node.right):
                d = _exact_const(node.right)
                if d == 0:
                    raise _NotExactLinear("division by zero")
                stack.append((node.left, scale / d))
                continue
            raise _NotExactLinear(f"operator {node.op}")
        if isinstance(node, SumOverExpression):
            for t in reversed(node.terms):
                stack.append((t, scale))
            continue
        raise _NotExactLinear(f"node {type(node).__name__}")
    return terms, const


def detect_implied_integers(model: Model) -> set[tuple[int, int]]:
    """Return ``{(variable._index, flat_element)}`` for every continuous variable
    the model *provably* forces to integer values (see module docstring).

    Conservative: under-detection is safe (a missed tightening); the returned set
    is sound — constraining any of these variables integer leaves the feasible
    region's relevant projection, and hence the optimum, unchanged.

    **Row representation (#863).** Rows are kept as their nonzeros only. The dense
    predecessor built an ``np.zeros(n)`` per equality *body* and RETAINED one per
    integer-data row, then re-derived each row's support with
    ``np.nonzero(np.abs(a) > tol)`` — an O(n) scan per row per fixpoint round.
    On ``watercontamination0202`` (106,711 variables / 107,209 constraints) this
    function cost **71.1 s and +31.2 GiB RSS**. The dense full-width row was never
    needed, only its support and the coefficients on it.

    The marking is **identical**, not merely similar, and that matters because this
    function marks variables INTEGER: a wrongly-marked variable cuts off feasible
    points, the cardinal correctness violation. Identity holds term by term:

    * ``_extract_linear_coefficients_sparse`` is the accumulator the dense wrapper
      is already built on, so the coefficient values are bit-identical.
    * The integer-coefficient test ran over all ``n`` slots; entries absent from the
      row are exactly ``0.0``, which is an integer, so testing only the stored
      entries accepts and rejects exactly the same rows. (Since #1952 every test is
      exact and the proof is read off the exact rational row; see the module
      docstring.)
    * The support is sorted ASCENDING, which is the order ``np.nonzero`` produced,
      and rows stay in ``model._constraints`` order. The fixpoint loop below is
      order-sensitive in principle, so preserving both orders — rather than merely
      preserving the set of rows — is what makes the marked set identical.
    * The support is now computed once per row instead of once per row per round;
      ``a`` never changes between rounds, so the recomputation was pure waste.

    ``test_863_sparse_implied_integer.py`` checks that identity against an
    independent dense reimplementation of the predecessor over the in-repo ``.nl``
    corpus, not just that it got faster.
    """
    n = sum(v.size for v in model._variables)
    flat = [(v, e) for v in model._variables for e in range(v.size)]
    # Known-integer mask: declared integer/binary to start; grows as we prove more.
    is_int = np.array(
        [flat[i][0].var_type in (VarType.INTEGER, VarType.BINARY) for i in range(n)],
        dtype=bool,
    )

    # Pre-extract integer-data linear equality rows once, as (support, coefficients)
    # with the support ascending. The float extraction is a cheap prefilter; the
    # proof uses the exact rational row (#1952).
    eq_rows: list[tuple[list[int], dict[int, Fraction]]] = []
    for c in model._constraints:
        if not isinstance(c, Constraint) or c.sense != "==":
            continue
        if not _is_linear(c.body):
            continue
        try:
            terms, const = _extract_linear_coefficients_sparse(c.body, model, n)
        except _NotLinearError:
            continue
        if terms:
            vals = np.fromiter(terms.values(), dtype=np.float64, count=len(terms))
            if not np.all(vals == np.round(vals)):
                continue
        if not float(const).is_integer():
            continue
        try:
            exact, exact_const = _exact_linear_row(c.body, model, n)
        except _NotExactLinear:
            continue
        if exact_const.denominator != 1:
            continue
        if any(v.denominator != 1 for v in exact.values()):
            continue
        nz = sorted(i for i, v in exact.items() if v != 0)
        eq_rows.append((nz, exact))

    marked: set[tuple[int, int]] = set()
    changed = True
    while changed:
        changed = False
        for nz, terms in eq_rows:
            for idx in nz:
                if is_int[idx]:
                    continue
                if abs(terms[idx]) != 1:
                    continue  # coefficient must be ±1 for the integer-quotient proof
                if all(is_int[j] for j in nz if j != idx):
                    var, elem = flat[idx]
                    marked.add((var._index, elem))
                    is_int[idx] = True
                    changed = True
    return marked


def mark_implied_integers(model: Model) -> int:
    """Mark every detected implied-integer variable's ``var_type`` as INTEGER,
    in place. Returns the number of (scalar-element) markings applied.

    A variable is promoted to INTEGER only when **all** of its scalar elements are
    implied-integer (the per-element granularity of detection is preserved for the
    common scalar case; array variables are promoted only when fully covered, to
    avoid changing the type of a partially-continuous block)."""
    detected = detect_implied_integers(model)
    if not detected:
        return 0
    by_var: dict[int, set[int]] = {}
    for vidx, elem in detected:
        by_var.setdefault(vidx, set()).add(elem)
    count = 0
    for v in model._variables:
        if v.var_type != VarType.CONTINUOUS:
            continue
        elems = by_var.get(v._index)
        if elems is not None and len(elems) == v.size:
            v.var_type = VarType.INTEGER
            count += v.size
    return count
