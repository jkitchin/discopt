"""Interval-valued evaluator for modeling expressions.

Walks a ``discopt.modeling`` expression DAG with :class:`Interval`
variable values and returns a sound interval enclosure of the
expression's value over the input box. Used as a building block by
the interval-AD Hessian propagator underlying the box-local
convexity certificate.

The evaluator trusts the underlying interval arithmetic primitives
for soundness: each supported atom composes them into a correctly
rounded enclosure. Atoms not in the supported set return an
unbounded interval ``[-inf, +inf]`` so downstream consumers refuse to
certify rather than produce a wrong answer.

References
----------
Moore (1966), *Interval Analysis*.
Neumaier (1990), *Interval Methods for Systems of Equations*.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
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

from . import interval as iv
from .interval import Interval

# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def evaluate_interval(
    expr: Expression,
    model: Optional[Model],
    box: Optional[dict] = None,
    _cache: Optional[dict] = None,
) -> Interval:
    """Return an interval enclosure of ``expr`` over ``box``.

    Args:
        expr: The expression to evaluate.
        model: The :class:`~discopt.modeling.core.Model` the expression
            references. Accepted for symmetry with the rest of the layer and
            threaded through the walk, but never consulted: a variable's
            enclosure comes from ``box`` or from its own ``lb``/``ub``. ``None``
            is therefore valid, and is what a caller with no model in hand --
            e.g. the ``atan2`` sign analysis in
            :mod:`discopt.modeling._atan2`, which runs at expression-build time
            -- passes. Keep it that way: a ``model`` that became load-bearing
            here would silently change those callers' answers
            (``test_enclosure_ignores_model_argument``).
        box: Optional dict ``{Variable: Interval}`` that overrides the
            variable's declared bounds. When ``None`` or a variable is
            missing, the declared ``(lb, ub)`` of the variable is used.
        _cache: Memoization dict keyed by ``id(expr)``.

    Returns:
        An :class:`Interval` that contains ``expr(x)`` for every ``x``
        consistent with the variable box.
    """
    if _cache is None:
        _cache = {}
    return _eval(expr, model, box or {}, _cache)


# ──────────────────────────────────────────────────────────────────────
# Internal dispatch
# ──────────────────────────────────────────────────────────────────────


def _variable_interval(v: Variable, box: dict) -> Interval:
    """Interval enclosure for a (possibly array-shaped) variable."""
    if v in box:
        return box[v]
    lb = np.asarray(v.lb, dtype=np.float64)
    ub = np.asarray(v.ub, dtype=np.float64)
    return Interval(lb, ub)


def _indexed_interval(expr: IndexExpression, box: dict) -> Optional[Interval]:
    """Interval enclosure for ``var[idx]``; ``None`` when the pattern
    is not a direct index of a Variable (caller falls back to a
    general DAG walk)."""
    if not isinstance(expr.base, Variable):
        return None
    base_iv = _variable_interval(expr.base, box)
    try:
        lo = base_iv.lo[expr.index]
        hi = base_iv.hi[expr.index]
    except (IndexError, TypeError):
        return None
    return Interval(np.asarray(lo), np.asarray(hi))


def _reduced_count(arr: np.ndarray, axis: Optional[int]) -> int:
    """How many terms a ``SumExpression`` reduction adds together.

    ``axis=None`` sums every element; an integer axis sums along that axis only.
    A 0-d operand is the identity, so it counts as one term and no widening is
    applied. Used to size the accumulation-error bound, so an over-count would
    only widen (safe) and an under-count would narrow (not safe) -- hence the
    conservative fallback of ``arr.size`` for anything unexpected.
    """
    if arr.ndim == 0:
        return 1
    if axis is None:
        return int(arr.size)
    try:
        return int(arr.shape[axis])
    except (IndexError, TypeError):
        return int(arr.size)


def _eager_children(expr: Expression) -> tuple:
    """The children :func:`_eval_impl` evaluates unconditionally for ``expr``.

    ``SumOverExpression`` terms are not listed: its affine fast path (#1712)
    reads them without evaluating each one, and pre-evaluating 90,000 terms
    would undo it. :func:`_eval` instead tries that path first and pushes the
    terms only when it declines. A child listed here that ``_eval_impl`` would
    not have evaluated only costs time; one missing only costs recursion depth.
    """
    if isinstance(expr, BinaryOp):
        return (expr.left, expr.right)
    if isinstance(expr, UnaryOp):
        return (expr.operand,)
    if isinstance(expr, SumExpression):
        return (expr.operand,)
    if isinstance(expr, FunctionCall):
        return tuple(expr.args)
    if isinstance(expr, MatMulExpression):
        return (expr.left, expr.right)
    if isinstance(expr, IndexExpression) and not isinstance(expr.base, Variable):
        return (expr.base,)
    return ()


def _eval(expr: Expression, model: Optional[Model], box: dict, cache: dict) -> Interval:
    eid = id(expr)
    if eid in cache:
        return cache[eid]
    # #1814: fill the cache children-first with an explicit stack, so every
    # ``_eval`` call ``_eval_impl`` makes on a listed child is a cache hit. The
    # recursion depth then no longer grows with the expression: a loop-built
    # ``x[0] + ... + x[2499]`` is 2500 ``BinaryOp`` levels deep, and the one-frame-
    # per-level walk raised ``RecursionError`` while ``Model.complementarity`` sized
    # its operand at build time, outside any solve-time headroom gate. Every node
    # is still evaluated by ``_eval_impl`` exactly once, so the enclosures are
    # unchanged.
    #
    # A ``SumOverExpression`` is resolved here rather than in ``_eval_impl``: the
    # #1712 affine fast path is tried first, and only when it declines are the
    # terms pushed as children -- a ``dm.sum([e, x[i]])`` chain nests one sum per
    # level, and its terms recursed once per level otherwise.
    stack: list[tuple[Expression, bool]] = [(expr, False)]
    while stack:
        node, expanded = stack.pop()
        nid = id(node)
        if nid in cache:
            continue
        is_sum = isinstance(node, SumOverExpression)
        if not expanded:
            if is_sum:
                fast = _sum_over_fast(node, box)
                if fast is not None:
                    cache[nid] = fast
                    continue
                children: Sequence[Expression] = node.terms
            else:
                children = _eager_children(node)
            pending = [c for c in children if id(c) not in cache]
            if pending:
                stack.append((node, True))
                stack.extend((c, False) for c in reversed(pending))
                continue
        if is_sum:
            cache[nid] = _sum_over_generic(node, model, box, cache)
        else:
            cache[nid] = _eval_impl(node, model, box, cache)
    return cache[eid]


def _eval_impl(expr: Expression, model: Optional[Model], box: dict, cache: dict) -> Interval:
    # --- Leaves -----------------------------------------------------
    if isinstance(expr, Constant):
        v = np.asarray(expr.value, dtype=np.float64)
        return Interval(v, v)
    if isinstance(expr, Parameter):
        v = np.asarray(expr.value, dtype=np.float64)
        return Interval(v, v)
    if isinstance(expr, Variable):
        return _variable_interval(expr, box)
    if isinstance(expr, IndexExpression):
        idx_iv = _indexed_interval(expr, box)
        if idx_iv is not None:
            return idx_iv
        # Fallback: recurse into the base expression (non-variable base).
        base = _eval(expr.base, model, box, cache)
        return Interval(np.asarray(base.lo[expr.index]), np.asarray(base.hi[expr.index]))

    # --- Unary ops --------------------------------------------------
    if isinstance(expr, UnaryOp):
        child = _eval(expr.operand, model, box, cache)
        if expr.op == "neg":
            return -child
        if expr.op == "abs":
            return iv.absolute(child)
        return _unbounded(child.lo.shape)

    # --- Binary ops -------------------------------------------------
    if isinstance(expr, BinaryOp):
        left = _eval(expr.left, model, box, cache)
        right = _eval(expr.right, model, box, cache)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "**":
            return _eval_power(expr, left, right)
        return _unbounded(left.lo.shape)

    # --- Function calls --------------------------------------------
    if isinstance(expr, FunctionCall):
        return _eval_function_call(expr, model, box, cache)

    # --- Aggregations ----------------------------------------------
    if isinstance(expr, SumExpression):
        # ``SumExpression`` IS a reduction: ``dag_compiler`` lowers it to
        # ``jnp.sum(operand, axis=expr.axis)``. This used to return the operand's
        # enclosure UNREDUCED, so ``sum(x)`` over ``x in [0, 10]^2`` came back as
        # ``[0, 10]`` instead of ``[0, 20]`` -- an enclosure that does not contain
        # the value, i.e. unsound in the narrow direction, and silently the wrong
        # shape. Measured via the #1148 residual probe: a point ``x = (6, 6)``
        # against the row ``sum(x) <= 10`` reported a violation of 0.0 where the
        # true violation is 2.0, because the reduction never happened and the
        # elementwise ``6`` cleared the row (#1158 review, HIGH 1).
        #
        # Summation is monotone increasing in every argument, so the enclosure of
        # the sum is the sum of the endpoints -- no interval-arithmetic subtlety.
        # Outward rounding by the ACCUMULATION error, not by one ULP.
        #
        # A single ``nextafter`` per endpoint is the right convention for a binary
        # operation, and it is what ``_eval_matmul`` uses -- but a reduction over
        # n terms accumulates far more than one ULP, so one step does not bound
        # it. Measured on this branch against an exact ``fractions.Fraction``
        # reference (3000 random sums, n in [4, 600], heavy cancellation at ~1e8):
        # the one-ULP form returned an enclosure that did NOT contain the true sum
        # on 2289 of 3000 trials, worst shortfall 1.5e-6. That is the same
        # too-narrow enclosure the reduction fix was written to close, and this
        # evaluator is on the solve path (nonlinear bound tightening, uniform/OA
        # relaxation, the g-convex injection), where it becomes an FBBT tightening
        # that cuts the optimum out of the box (#1158 review 3, MEDIUM 1).
        #
        # The size of that widening is ``_accumulation_factor``'s business, and
        # this reduction shares ``_widen_sum`` with ``_eval_matmul`` (#1161) so the
        # two cannot drift apart -- an earlier revision of this branch carried its
        # own inline copy of the same arithmetic, which is exactly the drift the
        # shared helper exists to prevent. Sharing also picks up the helper's
        # non-finite guard, which the inline copy lacked: an infinite ``|x_i|``
        # produced ``inf - inf = nan`` and silently destroyed the enclosure.
        #
        # Widening is always the SAFE direction: an enclosure may be loose and
        # remain sound, but must never be narrow.
        inner = _eval(expr.operand, model, box, cache)
        lo_arr = np.asarray(inner.lo, dtype=np.float64)
        hi_arr = np.asarray(inner.hi, dtype=np.float64)
        n_terms = _reduced_count(lo_arr, expr.axis)
        lo = _widen_sum(np.sum(lo_arr, axis=expr.axis), lo_arr, n_terms, expr.axis, -1.0)
        hi = _widen_sum(np.sum(hi_arr, axis=expr.axis), hi_arr, n_terms, expr.axis, +1.0)
        return Interval(np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))

    if isinstance(expr, SumOverExpression):
        fast = _sum_over_fast(expr, box)
        if fast is not None:
            return fast
        return _sum_over_generic(expr, model, box, cache)

    if isinstance(expr, MatMulExpression):
        return _eval_matmul(expr, model, box, cache)

    return _unbounded(())


# ──────────────────────────────────────────────────────────────────────
# Per-element linear sums (#1712)
# ──────────────────────────────────────────────────────────────────────


def _sum_over_fast(expr: SumOverExpression, box: dict) -> Optional[Interval]:
    """The enclosure of ``expr`` without evaluating its terms, else ``None``.

    An empty sum is the constant 0; otherwise :func:`_scalar_affine_sum`.
    """
    if not expr.terms:
        return Interval(np.zeros(()), np.zeros(()))
    return _scalar_affine_sum(expr.terms, box)


def _sum_over_generic(
    expr: SumOverExpression, model: Optional[Model], box: dict, cache: dict
) -> Interval:
    """``Σ terms`` by :meth:`Interval.__add__`, folded left to right."""
    total = _eval(expr.terms[0], model, box, cache)
    for t in expr.terms[1:]:
        total = total + _eval(t, model, box, cache)
    return total


def _scalar_leaf_bounds(expr: Expression, box: dict, bases: dict) -> Optional[tuple[float, float]]:
    """``(lo, hi)`` of a scalar variable or ``var[idx]`` leaf, else ``None``.

    The values :func:`_variable_interval` / :func:`_indexed_interval` would wrap,
    read without building an :class:`Interval` per leaf. ``bases`` memoizes each
    variable's enclosure for the duration of one sum. ``None`` -- the generic walk
    decides -- also for an inverted pair, so that walk's ``Interval`` validation
    still raises on it exactly as before.
    """
    if isinstance(expr, IndexExpression):
        var = expr.base
        if not isinstance(var, Variable):
            return None
        index = expr.index
    elif isinstance(expr, Variable):
        if expr.shape != ():
            return None
        var, index = expr, None
    else:
        return None
    pair = bases.get(id(var))
    if pair is None:
        whole = _variable_interval(var, box)
        pair = (whole.lo, whole.hi)
        bases[id(var)] = pair
    if index is None:
        lo, hi = pair
    else:
        try:
            lo = pair[0][index]
            hi = pair[1][index]
        except (IndexError, TypeError):
            return None
    if np.ndim(lo) != 0 or np.ndim(hi) != 0:
        return None
    flo, fhi = float(lo), float(hi)
    if flo > fhi:
        return None
    return flo, fhi


def _scalar_coefficient(expr: Expression) -> Optional[float]:
    """The value of a 0-d ``Constant``/``Parameter``, else ``None``."""
    if not isinstance(expr, (Constant, Parameter)):
        return None
    v = np.asarray(expr.value, dtype=np.float64)
    return float(v) if v.ndim == 0 else None


def _scalar_affine_sum(terms: list, box: dict) -> Optional[Interval]:
    """Enclose ``Σ terms`` when every term is ``leaf`` or ``coef * leaf``; else ``None``.

    #1712: ``dm.sum(c[i, j] * x[i, j] for ...)`` -- the per-element idiom a Pyomo
    port writes -- is a ``SumOverExpression`` with one Python node per term, and
    the generic walk paid four ``Interval`` constructions per term (90k terms:
    ~6 s per call, twice per solve via the #1551 certificate guard).

    This is the generic walk's arithmetic, not an approximation of it, so the
    result is **bit-identical** (``test_interval_eval_affine_sum_fast_path``):

    * each product is :meth:`Interval.__mul__`, applied elementwise -- every step
      of it (corners, ``min``/``max``, the ``0 * ±inf`` NaN cleanup, the #957
      underflow test choosing between ``_round_*`` and ``_round_*_exact0``) is
      elementwise, so a vector of terms gets exactly what each term got alone;
    * the sum is :meth:`Interval.__add__` folded left to right, in the same order
      as the generic ``SumOverExpression`` branch, on Python floats (IEEE-754
      double, so ``+`` and :func:`math.nextafter` match numpy's exactly).

    Any term outside the pattern returns ``None`` and the caller walks generically.
    """
    n = len(terms)
    lo_t: list = [None] * n
    hi_t: list = [None] * n
    prod_pos: list[int] = []
    l_lo: list[float] = []
    l_hi: list[float] = []
    r_lo: list[float] = []
    r_hi: list[float] = []
    bases: dict = {}
    for k, t in enumerate(terms):
        leaf = _scalar_leaf_bounds(t, box, bases)
        if leaf is not None:
            lo_t[k], hi_t[k] = leaf
            continue
        if not (isinstance(t, BinaryOp) and t.op == "*"):
            return None
        # Keep the operand order of ``left * right``: ``__mul__``'s corners are
        # formed from (self, other), and although IEEE products commute, matching
        # the order keeps the equivalence argument trivially line-by-line.
        cl = _scalar_coefficient(t.left)
        if cl is not None:
            leaf = _scalar_leaf_bounds(t.right, box, bases)
            if leaf is None:
                return None
            ll, lh, (rl, rh) = cl, cl, leaf
        else:
            cr = _scalar_coefficient(t.right)
            if cr is None:
                return None
            leaf = _scalar_leaf_bounds(t.left, box, bases)
            if leaf is None:
                return None
            (ll, lh), rl, rh = leaf, cr, cr
        prod_pos.append(k)
        l_lo.append(ll)
        l_hi.append(lh)
        r_lo.append(rl)
        r_hi.append(rh)

    if prod_pos:
        p_lo, p_hi = _elementwise_interval_mul(
            np.asarray(l_lo), np.asarray(l_hi), np.asarray(r_lo), np.asarray(r_hi)
        )
        for k, a, b in zip(prod_pos, p_lo.tolist(), p_hi.tolist()):
            lo_t[k] = a
            hi_t[k] = b

    # ``Interval.__add__``: ``_round_down_exact0(lo + lo')``, ``_round_up_exact0(hi + hi')``.
    nxt = math.nextafter
    ninf, pinf = -math.inf, math.inf
    lo = lo_t[0]
    hi = hi_t[0]
    for a, b in zip(lo_t[1:], hi_t[1:]):
        s = lo + a
        lo = 0.0 if s == 0.0 else nxt(s, ninf)
        s = hi + b
        hi = 0.0 if s == 0.0 else nxt(s, pinf)
    return Interval(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))


def _elementwise_interval_mul(
    s_lo: np.ndarray, s_hi: np.ndarray, o_lo: np.ndarray, o_hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """:meth:`Interval.__mul__` with each element treated as its own scalar interval.

    ``__mul__`` decides its NaN cleanup and its #957 rounding mode with ``any``
    over the whole array; for a 0-d interval that ``any`` is the element itself,
    so here both decisions are taken per element. Kept beside its one caller;
    see :func:`_scalar_affine_sum` for why it must stay line-for-line with
    ``Interval.__mul__``.
    """
    with np.errstate(invalid="ignore"):
        a = s_lo * o_lo
        b = s_lo * o_hi
        c = s_hi * o_lo
        d = s_hi * o_hi
        lo = np.minimum(np.minimum(a, b), np.minimum(c, d))
        hi = np.maximum(np.maximum(a, b), np.maximum(c, d))
        if bool(np.isnan(lo).any()) or bool(np.isnan(hi).any()):
            # A NaN corner can only be ``0 * ±inf``; ``nan_to_num`` is the identity
            # on every other corner, so cleaning all elements equals cleaning the
            # NaN ones.
            a = iv._nan_corner_to_zero(a)
            b = iv._nan_corner_to_zero(b)
            c = iv._nan_corner_to_zero(c)
            d = iv._nan_corner_to_zero(d)
            lo = np.minimum(np.minimum(a, b), np.minimum(c, d))
            hi = np.maximum(np.maximum(a, b), np.maximum(c, d))
    underflowed = (
        ((a == 0.0) & (s_lo != 0.0) & (o_lo != 0.0))
        | ((b == 0.0) & (s_lo != 0.0) & (o_hi != 0.0))
        | ((c == 0.0) & (s_hi != 0.0) & (o_lo != 0.0))
        | ((d == 0.0) & (s_hi != 0.0) & (o_hi != 0.0))
    )
    # Per element: ``exact0`` rounding iff an endpoint is zero and nothing
    # underflowed. Where no endpoint is zero the two roundings coincide.
    exact0 = ((lo == 0.0) | (hi == 0.0)) & ~underflowed
    out_lo = np.where(exact0, iv._round_down_exact0(lo), iv._round_down(lo))
    out_hi = np.where(exact0, iv._round_up_exact0(hi), iv._round_up(hi))
    return np.asarray(out_lo, dtype=np.float64), np.asarray(out_hi, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────
# Accumulation-error bound for reductions
# ──────────────────────────────────────────────────────────────────────


def _reduction_is_pairwise(terms: np.ndarray, axis: Union[int, tuple[int, ...], None]) -> bool:
    """Whether numpy sums this reduction **pairwise** rather than sequentially.

    numpy's pairwise summation is a property of the *inner loop*, not of
    ``np.sum``: it applies when the reduction walks contiguous memory. Reducing
    along a **slow** (strided) axis accumulates sequentially instead, and its
    error grows like ``n`` rather than ``log2(n)``
    (`numpy.sum notes <https://numpy.org/doc/stable/reference/generated/numpy.sum.html#notes>`_).

    Applying the pairwise bound to a sequential reduction is unsound in the
    narrow direction, and it was measured: a C-contiguous ``(10002, 2)`` array of
    ones with the first row ``1e16`` and the last ``-1e16`` has exact column sums
    of ``10000``, but ``sum(v, axis=0)`` returned ``[-67.89, 67.89]`` — an
    enclosure that excludes the truth — while the transposed fast-axis control
    enclosed it (#1158 review 4, blocking 2).

    Answers **False unless pairwise is provable**: over-widening is loose and
    sound, under-widening is not. A tuple axis reduces over several axes in an
    order numpy does not specify, so it takes the sequential bound too.
    """
    if terms.ndim == 0:
        return True
    if axis is None:
        # A full reduction walks one buffer, pairwise, when that buffer is
        # contiguous; a strided one gives no such guarantee.
        return bool(terms.flags.c_contiguous or terms.flags.f_contiguous)
    if not isinstance(axis, (int, np.integer)):
        return False
    a = int(axis) % terms.ndim
    return bool(terms.strides[a] == terms.itemsize)


def _accumulation_factor(n_terms: int, *, pairwise: bool) -> float:
    """Relative widening that bounds the float error of an ``n``-term reduction.

    A single ``np.nextafter`` per endpoint is the right convention for a *binary*
    operation, but a reduction over ``n`` terms accumulates far more than one ULP,
    so one step does not bound it. The forward error is bounded by
    ``gamma_k * sum |x_i|`` with ``u = eps/2`` the unit roundoff (Higham,
    *Accuracy and Stability of Numerical Algorithms*, §4.2), where ``k`` is
    ``log2(n)`` for **pairwise** summation and ``n - 1`` for **sequential**
    summation. Which one applies is :func:`_reduction_is_pairwise`'s business, and
    it is not a free choice — see there for the measured counterexample.
    ``_relax/convexity/eigenvalue.py`` already takes the rigorous route for its
    own accumulation, so the pattern is the repo's, not an invention.

    The factor returned is ``(log2(n) + 2) * eps`` pairwise and ``(n - 1) * eps``
    sequential — in ``u`` units, ``2*log2(n) + 4`` and ``2*(n - 1)``, each at
    least twice its bound. The headroom is deliberate and covers, on top of the
    summation itself:

    * the rounding of the summands, which for ``_eval_matmul`` are themselves
      *rounded products* of interval endpoints (``<= 0.5 * eps * sum |x_i|``);
    * the rounding of the multiplication by this factor.

    ``sum |x_i|`` is itself an ``n``-term reduction computed in floating point,
    so it can come back *below* the exact magnitude sum, and the widening term is
    only as sound as it is. Two things cover that. Its summands are
    **nonnegative**, so there is no cancellation and its relative error is at most
    ``gamma_n = n*u/(1 - n*u)`` — for any ``n`` this evaluator will ever see,
    ``n*u < 1e-9``, so ``fl(sum |x_i|) >= (1 - 1e-9) * sum |x_i|``, which the
    factor-of-two headroom above absorbs many times over. On top of that,
    :func:`_widen_sum` rounds the magnitude sum **up** by one ULP before scaling,
    so the final rounding is not left to the headroom at all. The caller still
    applies a final outward ``np.nextafter`` to absorb the rounding of the
    subtraction/addition of the widening term.

    Widening is always the SAFE direction: an enclosure may be loose and remain
    sound, but must never be narrow. ``n_terms <= 1`` is the identity reduction
    and gets no widening.
    """
    if n_terms <= 1:
        return 0.0
    eps = float(np.finfo(np.float64).eps)
    pairwise_k = math.log2(n_terms) + 2.0
    if pairwise:
        return pairwise_k * eps
    # ``max`` and not plain ``n - 1``: the two cross at n = 5, so for a handful of
    # terms the sequential bound is the SMALLER of the two, and taking it would
    # make this fix *narrow* strided enclosures relative to the code it replaces.
    # Both values are sound (each is at least twice its Higham bound), but a
    # soundness fix that tightens a bound anywhere is a bound-changing change
    # needing its own differential evidence (CLAUDE.md §5), and there is no reason
    # to buy that for <= 5 terms. With the max, the pairwise path is untouched and
    # the strided path only ever widens.
    return max(float(n_terms - 1), pairwise_k) * eps


def _widen_sum(
    total: np.ndarray,
    terms: np.ndarray,
    n_terms: int,
    axis: Union[int, tuple[int, ...], None],
    sign: float,
) -> np.ndarray:
    """Widen a reduced sum outward by its accumulation-error bound.

    ``sign`` is ``-1.0`` for a lower endpoint and ``+1.0`` for an upper one. A
    non-finite error term (any ``|x_i|`` infinite, or the sum overflowing) widens
    the endpoint all the way to the corresponding infinity rather than producing
    ``inf - inf = nan``, which would silently destroy the enclosure.

    ``axis`` mirrors ``np.sum``'s: ``_eval_matmul`` reduces a whole row and passes
    ``None``, while the ``SumExpression`` reduction passes its declared ``axis``
    (an int, a tuple of ints, or ``None`` for a full reduction). Both go through
    this one helper so the two reductions cannot drift apart (#1161).

    The error bound is sized for the reduction numpy will *actually* perform —
    pairwise along contiguous memory, sequential along a strided axis — because
    the pairwise bound does not hold for a sequential accumulation
    (:func:`_reduction_is_pairwise`). ``sum |x_i|`` is rounded **up** before being
    scaled — a magnitude sum rounded down would shrink the very term that makes
    the enclosure sound — and its own accumulation error is bounded by the
    factor's headroom, since its summands are nonnegative and so cannot cancel
    (see :func:`_accumulation_factor`).
    """
    factor = _accumulation_factor(n_terms, pairwise=_reduction_is_pairwise(terms, axis))
    if factor == 0.0:
        return total
    mag = np.sum(np.abs(terms), axis=axis)
    err = np.nextafter(factor * np.nextafter(mag, np.inf), np.inf)
    widened = total + sign * err
    return np.where(np.isfinite(err), widened, sign * np.inf)


def _unbounded(shape) -> Interval:
    return Interval(
        np.full(shape, -np.inf, dtype=np.float64),
        np.full(shape, np.inf, dtype=np.float64),
    )


def _eval_power(expr: BinaryOp, left: Interval, right: Interval) -> Interval:
    """Handle ``base ** exponent`` — constant exponent only for v1."""
    # Require a concrete scalar exponent so we know whether integer or
    # fractional rules apply.
    if not isinstance(expr.right, (Constant, Parameter)):
        return _unbounded(left.lo.shape)
    raw = np.asarray(expr.right.value)
    if raw.ndim != 0:
        return _unbounded(left.lo.shape)
    n = float(raw)
    n_int = int(n)
    if np.isclose(n, float(n_int)):
        if n_int < 0:
            # x**-k = 1 / x**k (a denominator holding 0 gives (-inf, inf)).
            return Interval.point(1.0) / left ** (-n_int)
        return left**n_int
    # Fractional: base must be nonneg; use exp(n log(x)).
    if np.any(left.lo < 0):
        return _unbounded(left.lo.shape)
    return iv.exp(Interval.point(n) * iv.log(left))


def _eval_function_call(
    expr: FunctionCall, model: Optional[Model], box: dict, cache: dict
) -> Interval:
    if not expr.args:
        return _unbounded(())
    args = [_eval(a, model, box, cache) for a in expr.args]

    if expr.func_name == "max" and len(args) >= 2:
        lo = args[0].lo
        hi = args[0].hi
        for a in args[1:]:
            lo = np.maximum(lo, a.lo)
            hi = np.maximum(hi, a.hi)
        return Interval(lo, hi)
    if expr.func_name == "min" and len(args) >= 2:
        lo = args[0].lo
        hi = args[0].hi
        for a in args[1:]:
            lo = np.minimum(lo, a.lo)
            hi = np.minimum(hi, a.hi)
        return Interval(lo, hi)

    if expr.func_name == "centropy" and len(args) == 2:
        return iv.centropy(args[0], args[1])

    if expr.func_name == "atan2" and len(args) == 2:
        # atan2 maps EVERY argument pair into (-pi, pi], so this enclosure is
        # unconditionally sound and needs no sign analysis of the arguments.
        #
        # It matters because the generic multi-argument arm below returns
        # [-inf, +inf], and an unbounded enclosure becomes an unbounded aux
        # floor in `uniform_relax._build_multivar` -- i.e. no finite dual bound
        # at all, so the solve cannot certify even a trivially bounded
        # objective. A raw `atan2` node only reaches here from `serialize.loads`
        # or the GAMS link on a cut-straddling box (`dm.atan2` rewrites into
        # `atan` instead and never builds one), but a finite [-pi, pi] is
        # strictly better than nothing for those.
        shape = np.broadcast_shapes(args[0].lo.shape, args[1].lo.shape)
        return Interval(np.full(shape, -np.pi), np.full(shape, np.pi))

    if len(args) != 1:
        return _unbounded(args[0].lo.shape)

    arg = args[0]
    name = expr.func_name
    if name.startswith("norm"):
        return _norm_interval(name, arg)
    if name == "prod":
        return _prod_interval(arg)
    if name == "exp":
        return iv.exp(arg)
    if name == "log":
        return iv.log(arg)
    if name == "log2":
        return iv.log(arg) / Interval.point(float(np.log(2.0)))
    if name == "log10":
        return iv.log(arg) / Interval.point(float(np.log(10.0)))
    if name == "sqrt":
        return iv.sqrt(arg)
    if name == "abs":
        return iv.absolute(arg)
    if name == "sin":
        return iv.sin(arg)
    if name == "cos":
        return iv.cos(arg)
    if name == "tan":
        return iv.tan(arg)
    if name == "sinh":
        return iv.sinh(arg)
    if name == "cosh":
        return iv.cosh(arg)
    if name == "tanh":
        return iv.tanh(arg)
    # Monotone inverse-trig / inverse-hyperbolic / error / log1p atoms
    # (issue #136): each has a sound endpoint-image enclosure on its domain.
    if name == "atan":
        return iv.atan(arg)
    if name == "asin":
        return iv.asin(arg)
    if name == "acos":
        return iv.acos(arg)
    if name == "asinh":
        return iv.asinh(arg)
    if name == "acosh":
        return iv.acosh(arg)
    if name == "atanh":
        return iv.atanh(arg)
    if name == "erf":
        return iv.erf(arg)
    if name == "log1p":
        return iv.log1p(arg)
    if name == "sigmoid":
        return iv.sigmoid(arg)
    if name == "softplus":
        return iv.softplus(arg)
    if name == "entropy":
        return iv.entropy(arg)
    # A registered domain operator (#1248 A). Its enclosure is, by definition,
    # the enclosure of its lowering — the expression the model actually carries —
    # so this is sound by construction and never looser than the term-by-term
    # bound the same box would give.
    reg = _registered_interval(name, expr, model, box, cache)
    if reg is not None:
        return reg
    # Unsupported atoms return an unbounded enclosure; the certificate
    # will refuse to prove convexity for expressions that hit this
    # path, preserving soundness.
    return _unbounded(arg.lo.shape)


def _prod_interval(arg: Interval) -> Interval:
    """Sound SCALAR enclosure of ``prod(arg)`` (#1582).

    A one-argument ``prod`` is a full reduction -- ``jnp.prod`` over the flattened
    operand, the tape's ``*`` fold over elements -- so its enclosure is 0-d. It is
    the product of the per-element intervals, folded left in C order, each step
    widened one ULP outward (the binary-operation convention of ``_eval_matmul``).
    Before #1582 this fell through to the generic "unsupported" arm, which
    returned an unbounded enclosure of the ARGUMENT's shape: sound, but
    array-shaped, which the relaxation layer then rejected as an invariant
    violation, leaving every node of a ``dm.prod`` model without an LP bound.
    """
    lo_arr = np.ravel(np.asarray(arg.lo, dtype=np.float64))
    hi_arr = np.ravel(np.asarray(arg.hi, dtype=np.float64))
    if lo_arr.size == 0:
        return Interval.point(1.0)
    acc = Interval(lo_arr[0], hi_arr[0])
    for k in range(1, lo_arr.size):
        acc = acc * Interval(lo_arr[k], hi_arr[k])
        acc = Interval(np.nextafter(acc.lo, -np.inf), np.nextafter(acc.hi, np.inf))
    return acc


def _norm_interval(name: str, arg: Interval) -> Interval:
    """Sound SCALAR enclosure of ``norm{p}(arg)`` (#1493).

    A norm is a reduction, so its enclosure is 0-d whatever ``arg``'s shape.
    The generic arm below used to return an unbounded interval of ``arg``'s
    shape, i.e. an ARRAY enclosure for a scalar node, which the relaxation
    builder then failed on with ``TypeError``.

    For a 0-d or 1-D argument (the vector p-norm of the tape lowering) the
    enclosure is composed from outward-rounded interval operations on
    ``|arg_i|``: exact composition for ``p`` in {1, 2, inf}, and the
    norm-equivalence sandwich ``||x||_inf <= ||x||_p <= ||x||_1`` for any other
    ``p >= 1``. A >=2-D argument is ``jnp.linalg.norm``'s induced MATRIX norm,
    which the entrywise bounds above do NOT enclose (``||I_3||_2 = 1`` while the
    entrywise 2-norm is ``sqrt(3)``), so it gets only ``[0, +inf)`` -- true of
    every norm.
    """
    lo_arr = np.asarray(arg.lo, dtype=np.float64)
    if lo_arr.ndim >= 2 or lo_arr.size == 0:
        return Interval(np.asarray(0.0), np.asarray(np.inf))
    suffix = name[len("norm") :]
    if suffix == "inf":
        p = np.inf
    elif suffix == "":
        p = 2.0
    else:
        try:
            p = float(suffix)
        except ValueError:
            return Interval(np.asarray(0.0), np.asarray(np.inf))
    a = iv.absolute(arg)
    a_lo = np.asarray(a.lo, dtype=np.float64).reshape(-1)
    a_hi = np.asarray(a.hi, dtype=np.float64).reshape(-1)
    parts = [Interval(np.asarray(lo), np.asarray(hi)) for lo, hi in zip(a_lo, a_hi)]
    if p == np.inf:
        # max selects existing doubles: exact, no rounding.
        return Interval(np.asarray(np.max(a_lo)), np.asarray(np.max(a_hi)))
    if p == 2.0:
        acc = parts[0] ** 2
        for part in parts[1:]:
            acc = acc + part**2
        return iv.sqrt(acc)
    one = parts[0]
    for part in parts[1:]:
        one = one + part
    if p == 1.0:
        return one
    # ||x||_inf <= ||x||_p <= ||x||_1 for every p >= 1.
    return Interval(np.asarray(np.max(a_lo)), np.asarray(one.hi))


def _registered_interval(name, expr, model, box, cache):
    """Enclosure of a registered atom, via its lowering, or ``None``.

    Reached for a ``FunctionCall`` that ``canonical_expr`` reconstructed from a
    named atom node: the reconstruction carries the name but not the tag, so the
    lowering is rebuilt from the registry on the reconstructed argument.
    """
    from discopt.operators import get_registered

    fn = get_registered(name)
    if fn is None or len(expr.args) != 1:
        return None
    # #1520: kept as a sound fallback. ``interval_expr`` is the user's registered
    # lowering (``discopt.operators``), i.e. user code with no documented failure
    # mode; when it raises, the caller keeps the unbounded enclosure, which contains
    # every value and so can never tighten a bound or prove a curvature. Only the
    # user call is guarded: our own enclosure of the lowering declines by value.
    try:
        lowered = fn.interval_expr(expr.args[0])
    except Exception as exc:  # noqa: BLE001 - user-supplied lowering; see above
        from discopt._relax._fallback import warn_fallback_once

        warn_fallback_once(
            f"registered operator {name!r} interval lowering",
            exc,
            "its enclosure stays unbounded (no tightening, no curvature claim)",
        )
        return None
    return evaluate_interval(lowered, model, box, cache)


def _eval_matmul(
    expr: MatMulExpression, model: Optional[Model], box: dict, cache: dict
) -> Interval:
    """Interval matrix–vector or matrix–matrix product.

    ``discopt`` uses ``MatMulExpression`` primarily for constant-matrix
    times variable-vector; the handling below covers that case plus
    the symmetric one.
    """
    left = _eval(expr.left, model, box, cache)
    right = _eval(expr.right, model, box, cache)

    # For a matmul of interval matrices A (m × k) by B (k × n) the
    # enclosure is formed from the interval dot products. We express
    # the dot product as the sum of element-wise interval products,
    # which the :class:`Interval` operators already propagate soundly.
    A_lo, A_hi = np.asarray(left.lo), np.asarray(left.hi)
    B_lo, B_hi = np.asarray(right.lo), np.asarray(right.hi)
    if A_lo.ndim == 2 and B_lo.ndim == 1:
        # (m, k) @ (k,) → (m,)
        m, k = A_lo.shape
        lo = np.zeros(m, dtype=np.float64)
        hi = np.zeros(m, dtype=np.float64)
        for i in range(m):
            row_lo = A_lo[i]
            row_hi = A_hi[i]
            prods_lo = np.minimum(
                np.minimum(row_lo * B_lo, row_lo * B_hi),
                np.minimum(row_hi * B_lo, row_hi * B_hi),
            )
            prods_hi = np.maximum(
                np.maximum(row_lo * B_lo, row_lo * B_hi),
                np.maximum(row_hi * B_lo, row_hi * B_hi),
            )
            # Outward rounding by the ACCUMULATION error, not by one ULP: the
            # dot product adds ``k`` terms, each itself a rounded product of
            # interval endpoints, so a single ``nextafter`` per endpoint does
            # not bound the error (#1161). Measured before this change against
            # an exact ``fractions.Fraction`` reference (400 random ``(1, k) @
            # (k,)`` products, ``k in [4, 400)``): the one-ULP form returned an
            # enclosure that did NOT contain the true dot product on 166 of 400
            # trials, worst shortfall 8.5e-07. This evaluator is on the solve
            # path (nonlinear bound tightening, uniform/OA relaxation, the
            # g-convex injection, the interval-AD Hessian propagator), where a
            # too-narrow enclosure becomes an FBBT tightening that cuts the
            # optimum out of the box.
            lo[i] = _widen_sum(prods_lo.sum(), prods_lo, k, None, -1.0)
            hi[i] = _widen_sum(prods_hi.sum(), prods_hi, k, None, +1.0)
        return Interval(np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))
    # Other shapes fall through as unbounded for now — not needed by
    # any expression the convexity certificate currently targets.
    return _unbounded(A_lo.shape[:-1] if A_lo.ndim >= 1 else ())


__all__ = ["evaluate_interval"]
