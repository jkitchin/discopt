"""Interval-valued forward-mode automatic differentiation.

Produces a sound enclosure of the gradient and Hessian of a scalar
expression over an input box. Each node carries a triple
``(value, gradient, hessian)`` whose entries are :class:`Interval`
objects; the chain rule is applied with interval arithmetic so that
the resulting matrix ``H`` encloses every pointwise Hessian of the
expression on the box.

This machinery exists to support the sound box-local convexity
certificate. The caller then bounds the minimum eigenvalue of ``H``
using interval Gershgorin (or a tighter test); if that lower bound
is ≥ 0, the expression is convex on the box.

The implementation is pure numpy; no JAX. JAX's autodiff does not
carry interval types, and the per-constraint cost is small enough
that a Python walker is adequate.

Sparse representation
---------------------
Internally the walker does **not** materialise a dense ``n × n``
interval Hessian at every node. A separable expression — a sum of ``N``
terms each touching only one or two variables — has a Hessian whose
non-zero footprint is ``O(N)``, yet a dense per-node representation
forces ``O(n²)`` work (allocation + arithmetic) at every one of the
``N`` nodes, i.e. ``O(N · n²)`` overall. Instead each node carries

* ``grad`` as ``dict[int, Interval]`` — only the non-zero partials,
* ``hess`` as ``dict[(int, int), Interval]`` — only the non-zero
  entries of the *upper triangle* (the Hessian is symmetric), keyed by
  ``(i, j)`` with ``i ≤ j``.

The chain-rule arithmetic then touches only the live entries, so the
walk is ``O(N + nnz)``. The single ``n × n`` allocation happens once,
at the very top, when :func:`interval_hessian` densifies the root node
into the public :class:`IntervalAD` the certificate consumes. Soundness
is identical to the dense path: a missing key denotes a *structural*
zero (the partial is identically zero — no computation, hence no
roundoff to enclose), and every present entry is produced by the same
outward-rounded :class:`Interval` arithmetic.

Limitations
-----------
Current atom table covers ``+``, ``-``, unary ``neg``, ``*``, ``/``
(constant or strictly-signed denominator), integer powers, ``exp``,
``log``, ``sqrt``. Non-smooth atoms (``abs``, ``max``, ``min``) have
undefined Hessians at kink points and are rejected with an unbounded
Hessian, forcing the certificate to abstain.

References
----------
Moore (1966), *Interval Analysis*, §4 (interval derivatives).
Neumaier (1990), *Interval Methods for Systems of Equations*.
Griewank, Walther (2008), *Evaluating Derivatives*, §3 (forward-mode
automatic differentiation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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

from . import interval as iv
from .interval import Interval

# Reused scalar-interval constants (degenerate points) — building these
# once avoids re-allocating tiny 0-d arrays in every chain-rule step.
_ONE = Interval.point(1.0)
_TWO = Interval.point(2.0)


class IntervalHessianTooLarge(ValueError):
    """Raised when an expression's DAG exceeds the interval-Hessian node budget.

    The interval-Hessian walk is a pure-numpy, per-node chain-rule pass; its wall
    cost is ~linear in the DAG node count but with a large numpy-scalar constant
    (~0.5 ms/node), so a body with >100k nodes (e.g. qap's 21 424-term quadratic
    objective, ~124k nodes) runs for over a minute and is uninterruptible — blowing
    the solver's ``time_limit`` (#654). A ``ValueError`` subclass so the existing
    ``except ValueError`` abstention paths (``certify_convex``, the McCormick
    Hessian refinement) catch it and fall back soundly.
    """


# Node-count ceiling for :func:`interval_hessian`. Above this the walk abstains
# (raises :class:`IntervalHessianTooLarge`) rather than run a minute-plus
# uninterruptible pass. The interval Hessian is only ever a *bound tightening*
# (convexity proof / alphaBB / McCormick refinement); refusing it routes callers
# to their sound looser fallback (spatial B&B, term-wise McCormick), so the
# ceiling never affects a dual bound's validity. Set well above any normal
# convex-body DAG (hundreds–low-thousands of nodes) and far below the pathological
# regime. A purely quadratic body of any size still certifies through the exact
# PSD-on-Q fast path in ``certify_convex``, which runs before this walk.
_INTERVAL_HESSIAN_MAX_NODES = 8000


def _expr_node_budget_exceeded(expr: Expression, limit: int) -> bool:
    """True if ``expr``'s DAG has more than ``limit`` distinct nodes.

    Iterative, memoized (shared subexpressions counted once — the same accounting
    :func:`_walk` uses), and early-exits the moment the count crosses ``limit``, so
    the check itself is O(limit) and never becomes the pathology it guards against.
    """
    seen: set[int] = set()
    stack: list[Expression] = [expr]
    count = 0
    while stack:
        e = stack.pop()
        eid = id(e)
        if eid in seen:
            continue
        seen.add(eid)
        count += 1
        if count > limit:
            return True
        if isinstance(e, (BinaryOp, MatMulExpression)):
            stack.append(e.left)
            stack.append(e.right)
        elif isinstance(e, UnaryOp):
            stack.append(e.operand)
        elif isinstance(e, (FunctionCall, CustomCall)):
            stack.extend(e.args)
        elif isinstance(e, SumExpression):
            stack.append(e.operand)
        elif isinstance(e, SumOverExpression):
            stack.extend(e.terms)
        elif isinstance(e, IndexExpression):
            stack.append(e.base)
        # Variable / Constant (and any other leaf) contribute no children.
    return False


# Type aliases for the sparse carriers (documentation only).
GradMap = "dict[int, Interval]"
HessMap = "dict[tuple[int, int], Interval]"


# ──────────────────────────────────────────────────────────────────────
# Public data types (dense — the certificate / tests consume these)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rank1Factor:
    """Sound metadata: this node's Hessian is ``c · v vᵀ``.

    A node carries this when its Hessian is provably rank-1 PSD (or
    NSD, with a sign-bracketed ``c``). The certificate consults it
    for a structural sufficient PSD test that does not depend on the
    entry-wise tightness of the interval matrix — useful on wide
    boxes where the off-diagonal interval blows up but the underlying
    rank-1 structure is intact.

    The ``affine_base_*`` fields are populated when the node arose
    from squaring an expression whose Hessian is identically zero
    (i.e. the base is affine in the variables); a downstream
    division by an affine positive denominator combines them via the
    perspective rule into a tighter rank-1 form.
    """

    c: Interval
    v: Interval
    affine_base_value: Optional[Interval] = None
    affine_base_grad: Optional[Interval] = None


@dataclass(frozen=True)
class IntervalAD:
    """A scalar expression's value, gradient, and Hessian as intervals.

    * ``value`` — scalar interval enclosing ``f(x)`` for ``x`` in the box.
    * ``grad``  — shape-``(n,)`` interval enclosing ``∇f(x)``.
    * ``hess``  — shape-``(n, n)`` symmetric interval enclosing
      ``∇²f(x)``.
    * ``rank1_factor`` — optional rank-1 metadata; see
      :class:`Rank1Factor`. ``None`` for nodes without a known
      rank-1 structure. Soundness is independent of this field —
      it is metadata that lets the certificate dispatch a tighter
      sufficient PSD test. Any op that does not explicitly preserve
      the factorisation drops the field, so a stale value cannot
      mislead.
    """

    value: Interval
    grad: Interval
    hess: Interval
    rank1_factor: Optional[Rank1Factor] = None


# ──────────────────────────────────────────────────────────────────────
# Internal sparse carriers
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _SparseRank1:
    """Sparse counterpart of :class:`Rank1Factor` used during the walk.

    ``v`` and ``affine_base_grad`` are sparse gradient maps; they are
    densified to dense :class:`Interval` vectors only when the root
    node is converted to the public :class:`IntervalAD`.
    """

    c: Interval
    v: "dict[int, Interval]"
    affine_base_value: Optional[Interval] = None
    affine_base_grad: Optional["dict[int, Interval]"] = None


@dataclass(slots=True)
class _SparseAD:
    """A node's value, gradient, and Hessian in sparse form.

    * ``value`` — scalar :class:`Interval`.
    * ``grad`` — ``{flat_index: Interval}``; absent key ⇒ exact-zero
      partial.
    * ``hess`` — ``{(i, j): Interval}`` for ``i ≤ j`` (upper triangle of
      the symmetric Hessian); absent key ⇒ exact-zero entry.
    * ``n`` — flat variable count (for densification).
    * ``unbounded`` — ``True`` when an unsupported / non-smooth atom
      forced an abstention; densifies to a ``±inf`` Hessian so the
      certificate refuses to certify.
    * ``rank1`` — optional sparse rank-1 metadata.
    """

    value: Interval
    grad: "dict[int, Interval]"
    hess: "dict[tuple[int, int], Interval]"
    n: int
    unbounded: bool = False
    rank1: Optional[_SparseRank1] = field(default=None)


# ──────────────────────────────────────────────────────────────────────
# Flat-variable index map
# ──────────────────────────────────────────────────────────────────────


def _offset_map(model: Model) -> list[int]:
    """Prefix-sum flat offsets for ``model``'s variables, cached on the model.

    ``_var_offset`` is called once per variable leaf of every node visited, so
    the naive ``sum(v.size for v in variables[:index])`` made leaf handling
    O(n) and the whole walk O(n²). The prefix sums are a pure function of the
    variable list (which only ever grows by append), so cache them keyed on the
    list identity and length; a length change invalidates the cache.
    """
    variables = model._variables
    cached = model.__dict__.get("_iad_offset_cache")
    if cached is not None and cached[0] is variables and cached[1] == len(variables):
        return cached[2]  # type: ignore[no-any-return]
    offsets: list[int] = []
    acc = 0
    for v in variables:
        offsets.append(acc)
        acc += v.size
    model.__dict__["_iad_offset_cache"] = (variables, len(variables), offsets)
    return offsets


def _var_offset(var: Variable, model: Model) -> int:
    return _offset_map(model)[var._index]


def _flat_size(model: Model) -> int:
    return sum(v.size for v in model._variables)


# ──────────────────────────────────────────────────────────────────────
# Sparse arithmetic helpers
# ──────────────────────────────────────────────────────────────────────
#
# All multiplications below feed scalar :class:`Interval` operators,
# which outward-round (toward ∓inf) on every op — so the maps these
# helpers build are sound enclosures, entry by entry. The whole walk
# runs under a single ``np.errstate`` (see :func:`interval_hessian`) so
# that intentional overflow / ``0 * inf`` on wide boxes — which produce
# the ``±inf`` sentinels the certificate reads as "abstain" — do not
# emit benchmark-visible warnings.


def _dadd(a: dict, b: dict) -> dict:
    """Entry-wise sum of two sparse maps (grad or hess)."""
    if not a:
        return dict(b)
    if not b:
        return dict(a)
    out = dict(a)
    for k, v in b.items():
        cur = out.get(k)
        out[k] = v if cur is None else cur + v
    return out


def _dsub(a: dict, b: dict) -> dict:
    """Entry-wise difference ``a - b`` of two sparse maps."""
    if not b:
        return dict(a)
    out = dict(a)
    for k, v in b.items():
        cur = out.get(k)
        out[k] = -v if cur is None else cur - v
    return out


def _dneg(a: dict) -> dict:
    """Entry-wise negation (exact — sign flip carries no roundoff)."""
    return {k: -v for k, v in a.items()}


def _dscale(s: Interval, a: dict) -> dict:
    """Scale every entry of a sparse map by the scalar interval ``s``."""
    if not a:
        return {}
    return {k: s * v for k, v in a.items()}


def _self_outer(g: dict) -> dict:
    """Upper-triangle of ``g gᵀ`` with dependency-aware tightening.

    The diagonal uses the squaring rule (``gᵢ²`` is nonneg and bracketed
    by ``[0, max(|loᵢ|, |hiᵢ|)²]``), matching the dense self-outer
    specialisation — essential for Hessian enclosures tight enough for
    Gershgorin to certify compositions like ``exp(x²)``. Off-diagonal
    entries are the general corner products ``gᵢ · gⱼ``.
    """
    items = list(g.items())
    out: dict = {}
    for a in range(len(items)):
        i, gi = items[a]
        out[(i, i)] = gi**2  # nonneg squaring special-case in Interval.__pow__
        for b in range(a + 1, len(items)):
            j, gj = items[b]
            out[(i, j) if i < j else (j, i)] = gi * gj
    return out


def _sym_cross(a: dict, b: dict) -> dict:
    """Upper-triangle of the symmetric matrix ``a bᵀ + b aᵀ``.

    Mirrors the dense ``_outer(a, b) + _outer(b, a)`` term: entry
    ``(i, j)`` is ``aᵢ bⱼ + aⱼ bᵢ`` (absent keys count as exact zero).
    When ``a is b`` the result is ``2 · a aᵀ`` and is routed through
    :func:`_self_outer` so the diagonal keeps the squaring tightening.
    """
    if a is b:
        return _dscale(_TWO, _self_outer(a))
    if not a or not b:
        return {}
    keys = sorted(set(a) | set(b))
    out: dict = {}
    for ix in range(len(keys)):
        i = keys[ix]
        ai = a.get(i)
        bi = b.get(i)
        for jx in range(ix, len(keys)):
            j = keys[jx]
            aj = a.get(j)
            bj = b.get(j)
            term: Optional[Interval] = None
            if ai is not None and bj is not None:
                term = ai * bj
            if aj is not None and bi is not None:
                t2 = aj * bi
                term = t2 if term is None else term + t2
            if term is not None:
                out[(i, j)] = term
    return out


_AFFINE_HESS_TOL = 1e-300


def _hess_is_exactly_zero(hess: dict) -> bool:
    """``True`` iff the sparse Hessian has no curvature above ``_AFFINE_HESS_TOL``.

    An empty map is the common affine case (no second-order entry was
    ever produced). A non-empty map can still be "affine" when every
    entry encloses ``0`` within a few ULPs (subnormals from
    outward-rounded cancellations). The tolerance ``1e-300`` sits ~270
    orders of magnitude below any plausible real curvature term, so it
    cannot misclassify genuine curvature as affine.

    Sufficient-only: a false ``False`` merely skips the rank-1 fast path
    and falls through to Gershgorin.
    """
    tol = _AFFINE_HESS_TOL
    for entry in hess.values():
        if abs(float(entry.lo)) > tol or abs(float(entry.hi)) > tol:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Sentinels / densification
# ──────────────────────────────────────────────────────────────────────


def _unbounded(n: int) -> _SparseAD:
    """Abstention sentinel: densifies to a ``±inf`` Hessian."""
    inf = np.float64(np.inf)
    return _SparseAD(
        value=Interval(-inf, inf),
        grad={},
        hess={},
        n=n,
        unbounded=True,
    )


def _dense_unbounded(n: int) -> IntervalAD:
    inf = np.float64(np.inf)
    return IntervalAD(
        value=Interval(-inf, inf),
        grad=Interval(np.full(n, -inf), np.full(n, inf)),
        hess=Interval(np.full((n, n), -inf), np.full((n, n), inf)),
    )


def _dense_vec(d: dict, n: int) -> Interval:
    lo = np.zeros(n, dtype=np.float64)
    hi = np.zeros(n, dtype=np.float64)
    for i, entry in d.items():
        lo[i] = entry.lo
        hi[i] = entry.hi
    return Interval(lo, hi)


def _densify(sad: _SparseAD, n: int) -> IntervalAD:
    """Materialise a sparse node into the public dense :class:`IntervalAD`.

    The only ``O(n²)`` allocation in the whole certificate; the
    scatter loop costs ``O(nnz)``. Absent grad/hess keys denote exact
    structural zeros and are left as ``0.0``.
    """
    if sad.unbounded:
        return _dense_unbounded(n)

    grad = _dense_vec(sad.grad, n)

    hlo = np.zeros((n, n), dtype=np.float64)
    hhi = np.zeros((n, n), dtype=np.float64)
    for (i, j), entry in sad.hess.items():
        lo = entry.lo
        hi = entry.hi
        hlo[i, j] = lo
        hhi[i, j] = hi
        if i != j:
            hlo[j, i] = lo
            hhi[j, i] = hi
    hess = Interval(hlo, hhi)

    rank1: Optional[Rank1Factor] = None
    if sad.rank1 is not None:
        r = sad.rank1
        abg = _dense_vec(r.affine_base_grad, n) if r.affine_base_grad is not None else None
        rank1 = Rank1Factor(
            c=r.c,
            v=_dense_vec(r.v, n),
            affine_base_value=r.affine_base_value,
            affine_base_grad=abg,
        )

    return IntervalAD(value=sad.value, grad=grad, hess=hess, rank1_factor=rank1)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def interval_hessian(
    expr: Expression,
    model: Model,
    box: Optional[dict] = None,
) -> IntervalAD:
    """Interval value, gradient, and Hessian of a scalar expression.

    The returned triple encloses ``(f(x), ∇f(x), ∇²f(x))`` for every
    point ``x`` in the input box. The Hessian enclosure is the
    artifact the convexity certificate consumes.

    Args:
        expr: Scalar expression from :mod:`discopt.modeling.core`.
        model: The model defining the flat variable layout.
        box: Optional ``{Variable: Interval}`` overriding declared
            bounds. Missing variables fall back to ``(v.lb, v.ub)``.

    Raises:
        ValueError: if ``expr`` references array-shaped values that
            the scalar-output AD cannot handle.
        IntervalHessianTooLarge: if ``expr``'s DAG exceeds
            :data:`_INTERVAL_HESSIAN_MAX_NODES` — the walk would blow the solver's
            time budget (#654); callers catch it and fall back soundly.
    """
    n = _flat_size(model)
    if n == 0:
        raise ValueError("Model has no variables; cannot produce Hessian.")
    # #654: refuse the minute-plus uninterruptible walk on a pathologically large
    # body. Cheap O(budget) pre-check with early-exit; abstaining is sound (the
    # interval Hessian is only ever a bound tightening).
    if _expr_node_budget_exceeded(expr, _INTERVAL_HESSIAN_MAX_NODES):
        raise IntervalHessianTooLarge(
            f"expression DAG exceeds {_INTERVAL_HESSIAN_MAX_NODES} nodes; "
            "interval-Hessian walk declined to protect the time budget (#654)"
        )
    box = box or {}
    cache: dict = {}
    # Wide boxes intentionally overflow to ``±inf`` / ``0 * inf`` (the
    # abstention sentinels the certificate reads as UNKNOWN). Those are
    # sound interval results, not numerical bugs — suppress the warnings
    # for the whole walk, mirroring the dense path's per-op errstate.
    with np.errstate(over="ignore", invalid="ignore"):
        sad = _walk(expr, model, box, cache, n)
        return _densify(sad, n)


# ──────────────────────────────────────────────────────────────────────
# Internal DAG walker
# ──────────────────────────────────────────────────────────────────────


def _walk(expr: Expression, model: Model, box: dict, cache: dict, n: int) -> _SparseAD:
    eid = id(expr)
    hit = cache.get(eid)
    if hit is not None:
        return hit
    out = _impl(expr, model, box, cache, n)
    cache[eid] = out
    return out


def _variable_scalar_value(v: Variable, box: dict) -> Interval:
    """Interval enclosure of a *scalar* variable's value.

    The result is always a 0-d (scalar) :class:`Interval`: the sparse
    walker stores one scalar per grad/hess entry, so a box override
    supplied as a shape-``(1,)`` interval (the layout
    :func:`refresh_convex_mask` builds) must be raveled to a scalar
    here rather than flowing as a length-1 vector.
    """
    if v in box:
        bi = box[v]
        lo = float(np.asarray(bi.lo).ravel()[0])
        hi = float(np.asarray(bi.hi).ravel()[0])
        return Interval(np.float64(lo), np.float64(hi))
    lb = float(np.asarray(v.lb).ravel()[0])
    ub = float(np.asarray(v.ub).ravel()[0])
    return Interval(np.float64(lb), np.float64(ub))


def _indexed_scalar_value(expr: IndexExpression, box: dict) -> Interval:
    v = expr.base
    lb = np.asarray(v.lb).ravel()
    ub = np.asarray(v.ub).ravel()
    # Translate the index into a flat position inside the variable.
    idx = expr.index
    if isinstance(idx, tuple):
        # Only 1-D indexing supported for scalar output.
        if len(idx) == 1:
            idx = idx[0]
        else:
            raise ValueError("Multi-dim indexing unsupported in interval AD")
    if v in box:
        # Box override is shape (size,) when variable is array-valued.
        box_iv = box[v]
        return Interval(np.asarray(box_iv.lo).ravel()[idx], np.asarray(box_iv.hi).ravel()[idx])
    return Interval(np.float64(lb[idx]), np.float64(ub[idx]))


def _impl(expr: Expression, model: Model, box: dict, cache: dict, n: int) -> _SparseAD:
    # --- Leaves -----------------------------------------------------
    if isinstance(expr, Constant):
        v = float(np.asarray(expr.value))
        return _SparseAD(value=Interval(np.float64(v), np.float64(v)), grad={}, hess={}, n=n)

    if isinstance(expr, Parameter):
        v = float(np.asarray(expr.value))
        return _SparseAD(value=Interval(np.float64(v), np.float64(v)), grad={}, hess={}, n=n)

    if isinstance(expr, Variable):
        if expr.size != 1:
            raise ValueError(f"Interval Hessian requires scalar variables; got shape {expr.shape}")
        slot = _var_offset(expr, model)
        val = _variable_scalar_value(expr, box)
        return _SparseAD(value=val, grad={slot: _ONE}, hess={}, n=n)

    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        v = expr.base
        raw_idx = expr.index
        if isinstance(raw_idx, tuple):
            if len(raw_idx) != 1:
                raise ValueError("Multi-dim indexing unsupported")
            flat_idx = int(raw_idx[0])
        else:
            flat_idx = int(raw_idx)
        slot = _var_offset(v, model) + flat_idx
        val = _indexed_scalar_value(expr, box)
        return _SparseAD(value=val, grad={slot: _ONE}, hess={}, n=n)

    # --- Unary ops --------------------------------------------------
    if isinstance(expr, UnaryOp):
        child = _walk(expr.operand, model, box, cache, n)
        if child.unbounded:
            return _unbounded(n)
        if expr.op == "neg":
            return _SparseAD(
                value=-child.value,
                grad=_dneg(child.grad),
                hess=_dneg(child.hess),
                n=n,
            )
        # |x| is non-smooth at 0 — no sound Hessian.
        return _unbounded(n)

    # --- Binary ops -------------------------------------------------
    if isinstance(expr, BinaryOp):
        return _binary(expr, model, box, cache, n)

    # --- Function calls --------------------------------------------
    if isinstance(expr, FunctionCall):
        return _function_call(expr, model, box, cache, n)

    if isinstance(expr, SumExpression):
        return _walk(expr.operand, model, box, cache, n)

    if isinstance(expr, SumOverExpression):
        if not expr.terms:
            return _SparseAD(value=Interval.point(0.0), grad={}, hess={}, n=n)
        result = _walk(expr.terms[0], model, box, cache, n)
        if result.unbounded:
            return _unbounded(n)
        value = result.value
        grad = dict(result.grad)
        hess = dict(result.hess)
        for t in expr.terms[1:]:
            other = _walk(t, model, box, cache, n)
            if other.unbounded:
                return _unbounded(n)
            value = value + other.value
            grad = _dadd(grad, other.grad)
            hess = _dadd(hess, other.hess)
        return _SparseAD(value=value, grad=grad, hess=hess, n=n)

    return _unbounded(n)


# ──────────────────────────────────────────────────────────────────────
# Binary-op rules
# ──────────────────────────────────────────────────────────────────────


def _binary(expr: BinaryOp, model: Model, box: dict, cache: dict, n: int) -> _SparseAD:
    left = _walk(expr.left, model, box, cache, n)
    right = _walk(expr.right, model, box, cache, n)

    if left.unbounded or right.unbounded:
        return _unbounded(n)

    if expr.op == "+":
        return _SparseAD(
            value=left.value + right.value,
            grad=_dadd(left.grad, right.grad),
            hess=_dadd(left.hess, right.hess),
            n=n,
        )

    if expr.op == "-":
        return _SparseAD(
            value=left.value - right.value,
            grad=_dsub(left.grad, right.grad),
            hess=_dsub(left.hess, right.hess),
            n=n,
        )

    if expr.op == "*":
        # (f g)'  = g f' + f g'
        # (f g)'' = g f'' + f g'' + f' g'ᵀ + g' f'ᵀ
        fg = left.value * right.value
        grad = _dadd(_dscale(right.value, left.grad), _dscale(left.value, right.grad))
        hess = _dadd(
            _dadd(_dscale(right.value, left.hess), _dscale(left.value, right.hess)),
            _sym_cross(left.grad, right.grad),
        )
        # Rank-1 metadata for ``g * g`` (BinaryOp form of squaring) when
        # ``g`` is affine. The DAG cache makes ``left is right`` for any
        # node that appears twice as the same instance, so this catches
        # ``x * x`` and any ``e * e`` bound to one Python variable. The
        # Hessian already collapsed to ``2 ∇g ∇gᵀ`` here, so the claim is
        # sound.
        rank1: Optional[_SparseRank1] = None
        if expr.left is expr.right and _hess_is_exactly_zero(left.hess):
            rank1 = _SparseRank1(
                c=_TWO,
                v=left.grad,
                affine_base_value=left.value,
                affine_base_grad=left.grad,
            )
        return _SparseAD(value=fg, grad=grad, hess=hess, n=n, rank1=rank1)

    if expr.op == "/":
        return _division(expr, left, right, n)

    if expr.op == "**":
        return _power(expr, left, n)

    return _unbounded(n)


def _division(expr: BinaryOp, left: _SparseAD, right: _SparseAD, n: int) -> _SparseAD:
    """Implement ``f / g`` via the reciprocal chain rule.

    Defined only when ``g`` is strictly sign-determined. Otherwise the
    triple falls to the unbounded enclosure.
    """
    g = right.value
    if g.contains_zero().any():
        return _unbounded(n)

    # Rank-1 fast path: when the numerator is the square of an affine
    # expression and the denominator is itself affine and strictly
    # positive, the quotient's Hessian collapses to a perspective-form
    # rank-1 matrix that the certificate can prove PSD structurally —
    # even on wide boxes where the entry-wise enclosure is too loose for
    # Gershgorin.
    if (
        left.rank1 is not None
        and left.rank1.affine_base_value is not None
        and left.rank1.affine_base_grad is not None
        and bool(np.all(np.asarray(g.lo) > 0.0))
        and _hess_is_exactly_zero(right.hess)
    ):
        return _rank1_quotient(left, right, n)

    # Reciprocal derivatives:
    #   (1/g)'  = -g' / g^2
    #   (1/g)'' = 2 g' g'ᵀ / g^3 - g'' / g^2
    g2 = g * g
    g3 = g2 * g
    inv_g = _ONE / g
    inv_g2 = _ONE / g2
    inv_g3 = _ONE / g3
    recip_value = inv_g
    recip_grad = _dscale(-inv_g2, right.grad)
    recip_hess = _dsub(
        _dscale(_TWO * inv_g3, _self_outer(right.grad)),
        _dscale(inv_g2, right.hess),
    )

    # f / g = f * (1/g) — apply product rule.
    fg_val = left.value * recip_value
    fg_grad = _dadd(_dscale(recip_value, left.grad), _dscale(left.value, recip_grad))
    fg_hess = _dadd(
        _dadd(_dscale(recip_value, left.hess), _dscale(left.value, recip_hess)),
        _sym_cross(left.grad, recip_grad),
    )
    return _SparseAD(value=fg_val, grad=fg_grad, hess=fg_hess, n=n)


def _rank1_quotient(left: _SparseAD, right: _SparseAD, n: int) -> _SparseAD:
    """Specialised ``g² / h`` Hessian when ``g`` and ``h`` are affine.

    For affine ``g`` with gradient ``v_g`` (so ``H_g = 0``) and affine
    ``h`` with gradient ``v_h`` (so ``H_h = 0``) and ``h > 0`` on the
    box, the quotient ``g²/h`` has the exact pointwise Hessian

        ``H = (2/h) · v vᵀ``     where  ``v = v_g − (g/h) · v_h``.

    This is the perspective form: a rank-1 PSD matrix on every point of
    the box. We emit the Hessian via :func:`_self_outer` so the diagonal
    is forced nonneg, and attach a :class:`_SparseRank1` so the
    certificate's structural PSD test fires regardless of off-diagonal
    interval blowup.
    """
    assert left.rank1 is not None
    assert left.rank1.affine_base_value is not None
    assert left.rank1.affine_base_grad is not None

    g_val = left.rank1.affine_base_value
    v_g = left.rank1.affine_base_grad
    h = right.value
    v_h = right.grad

    # Combined rank-1 vector and coefficient.
    g_over_h = g_val / h
    v_combined = _dsub(v_g, _dscale(g_over_h, v_h))
    c_combined = _TWO / h

    hess = _dscale(c_combined, _self_outer(v_combined))

    # Value and gradient via the standard reciprocal-product rule —
    # tightness on those is not required for the convexity verdict but
    # they remain sound and consistent with the generic path.
    inv_h = _ONE / h
    inv_h2 = _ONE / (h * h)
    recip_grad = _dscale(-inv_h2, v_h)
    fg_val = left.value * inv_h
    fg_grad = _dadd(_dscale(inv_h, left.grad), _dscale(left.value, recip_grad))

    return _SparseAD(
        value=fg_val,
        grad=fg_grad,
        hess=hess,
        n=n,
        rank1=_SparseRank1(c=c_combined, v=v_combined),
    )


def _power(expr: BinaryOp, base: _SparseAD, n_vars: int) -> _SparseAD:
    """``g^p`` for a literal exponent ``p``.

    Supports integer ``p`` on any domain and fractional ``p`` on a
    strictly positive base (through ``exp(p log g)`` composition).
    """
    if not isinstance(expr.right, (Constant, Parameter)):
        return _unbounded(n_vars)
    raw = np.asarray(expr.right.value)
    if raw.ndim != 0:
        return _unbounded(n_vars)
    p = float(raw)

    if np.isclose(p, 0.0):
        return _SparseAD(value=_ONE, grad={}, hess={}, n=n_vars)
    if np.isclose(p, 1.0):
        return base

    # Integer exponent path — computed without leaving interval arithmetic.
    p_int = int(p)
    if np.isclose(p, float(p_int)):
        return _integer_power(base, p_int, n_vars)

    # Fractional: require strictly positive base; use exp(p log g).
    if np.any(base.value.lo <= 0):
        return _unbounded(n_vars)
    log_g = _apply_log(base, n_vars)
    if log_g.unbounded:
        return _unbounded(n_vars)
    pt = Interval.point(p)
    scaled = _SparseAD(
        value=pt * log_g.value,
        grad=_dscale(pt, log_g.grad),
        hess=_dscale(pt, log_g.hess),
        n=n_vars,
    )
    return _apply_exp(scaled, n_vars)


def _integer_power(base: _SparseAD, p: int, n: int) -> _SparseAD:
    """Direct chain rule for ``g^p`` with integer ``p``.

    * value   : ``g^p``
    * gradient: ``p g^{p-1} ∇g``
    * hessian : ``p g^{p-1} H_g + p(p-1) g^{p-2} (∇g ⊗ ∇g)``

    Works for any sign of ``g`` when ``p`` is a positive integer;
    negative integer ``p`` goes through the reciprocal path.
    """
    if p < 0:
        # g^(-k) = (g^k)^-1 via the generic reciprocal chain rule.
        return _reciprocal_power(base, -p, n)
    g = base.value
    g_pm1 = g ** (p - 1)
    g_pm2 = g ** (p - 2) if p >= 2 else Interval.point(0.0)
    coeff1 = Interval.point(float(p)) * g_pm1
    coeff2 = Interval.point(float(p * (p - 1))) * g_pm2
    value = g**p
    grad = _dscale(coeff1, base.grad)
    hess = _dadd(_dscale(coeff1, base.hess), _dscale(coeff2, _self_outer(base.grad)))
    # Rank-1 metadata: when p == 2 and H_g is identically zero (g is
    # affine), the second-order term ``p g^{p-1} H_g`` vanishes and the
    # Hessian collapses exactly to ``2 · ∇g ∇gᵀ``. Soundness is
    # independent of this field.
    rank1: Optional[_SparseRank1] = None
    if p == 2 and _hess_is_exactly_zero(base.hess):
        rank1 = _SparseRank1(
            c=_TWO,
            v=base.grad,
            affine_base_value=base.value,
            affine_base_grad=base.grad,
        )
    return _SparseAD(value=value, grad=grad, hess=hess, n=n, rank1=rank1)


def _reciprocal_power(base: _SparseAD, k: int, n: int) -> _SparseAD:
    """``g^(-k) = 1 / g^k`` via the reciprocal rule."""
    if base.value.contains_zero().any():
        return _unbounded(n)
    gk = _integer_power(base, k, n)
    g = gk.value
    g2 = g * g
    g3 = g2 * g
    value = _ONE / g
    grad = _dscale(-(_ONE / g2), gk.grad)
    hess = _dsub(
        _dscale(_TWO / g3, _self_outer(gk.grad)),
        _dscale(_ONE / g2, gk.hess),
    )
    return _SparseAD(value=value, grad=grad, hess=hess, n=n)


# ──────────────────────────────────────────────────────────────────────
# Function-call rules
# ──────────────────────────────────────────────────────────────────────


def _function_call(expr: FunctionCall, model: Model, box: dict, cache: dict, n: int) -> _SparseAD:
    if len(expr.args) != 1:
        return _unbounded(n)
    arg = _walk(expr.args[0], model, box, cache, n)
    if arg.unbounded:
        return _unbounded(n)
    name = expr.func_name
    if name == "exp":
        return _apply_exp(arg, n)
    if name == "log":
        return _apply_log(arg, n)
    if name == "sin":
        return _apply_sin(arg, n)
    if name == "cos":
        return _apply_cos(arg, n)
    if name == "tan":
        return _apply_tan(arg, n)
    if name == "sqrt":
        # sqrt = x^0.5 on the positive domain.
        if np.any(arg.value.lo < 0):
            return _unbounded(n)
        # f = g^0.5, f' = 0.5 g^-0.5 g', f'' = 0.5 g^-0.5 H_g - 0.25 g^-1.5 (∇g ∇gᵀ).
        sqrt_g = iv.sqrt(arg.value)
        inv_sqrt_g = _ONE / sqrt_g
        inv_sqrt_g3 = inv_sqrt_g * inv_sqrt_g * inv_sqrt_g
        coeff1 = Interval.point(0.5) * inv_sqrt_g
        coeff2 = Interval.point(-0.25) * inv_sqrt_g3
        grad = _dscale(coeff1, arg.grad)
        hess = _dadd(_dscale(coeff1, arg.hess), _dscale(coeff2, _self_outer(arg.grad)))
        return _SparseAD(value=sqrt_g, grad=grad, hess=hess, n=n)
    # Other atoms (trig, abs, cosh, ...) are unsupported by the v1
    # certificate; return unbounded to force abstention.
    return _unbounded(n)


def _apply_exp(arg: _SparseAD, n: int) -> _SparseAD:
    """Chain rule through ``exp``.

    * value    : ``exp(g)``
    * gradient : ``exp(g) ∇g``
    * hessian  : ``exp(g) (H_g + ∇g ∇gᵀ)``
    """
    if arg.unbounded:
        return _unbounded(n)
    e = iv.exp(arg.value)
    grad = _dscale(e, arg.grad)
    hess = _dscale(e, _dadd(arg.hess, _self_outer(arg.grad)))
    return _SparseAD(value=e, grad=grad, hess=hess, n=n)


def _apply_log(arg: _SparseAD, n: int) -> _SparseAD:
    """Chain rule through ``log``.

    * value    : ``log(g)``
    * gradient : ``(1/g) ∇g``
    * hessian  : ``(1/g) H_g - (1/g²) (∇g ∇gᵀ)``
    """
    if arg.unbounded:
        return _unbounded(n)
    if np.any(arg.value.lo <= 0):
        return _unbounded(n)
    g = arg.value
    inv_g = _ONE / g
    inv_g2 = inv_g * inv_g
    value = iv.log(g)
    grad = _dscale(inv_g, arg.grad)
    hess = _dsub(_dscale(inv_g, arg.hess), _dscale(inv_g2, _self_outer(arg.grad)))
    return _SparseAD(value=value, grad=grad, hess=hess, n=n)


def _apply_sin(arg: _SparseAD, n: int) -> _SparseAD:
    """Chain rule through ``sin`` (region-aware via the interval of sin/cos).

    * value    : ``sin(g)``
    * gradient : ``cos(g) ∇g``
    * hessian  : ``cos(g) H_g - sin(g) (∇g ∇gᵀ)``

    On a box where ``g`` lies in a constant-curvature region, ``-sin(g)`` keeps
    a constant sign, so the interval Hessian is sign-definite and the
    certificate can prove convexity/concavity; a wide (sign-spanning) box yields
    an indefinite interval and a sound abstention.
    """
    if arg.unbounded:
        return _unbounded(n)
    g = arg.value
    sin_g = iv.sin(g)
    cos_g = iv.cos(g)
    grad = _dscale(cos_g, arg.grad)
    hess = _dsub(_dscale(cos_g, arg.hess), _dscale(sin_g, _self_outer(arg.grad)))
    return _SparseAD(value=sin_g, grad=grad, hess=hess, n=n)


def _apply_cos(arg: _SparseAD, n: int) -> _SparseAD:
    """Chain rule through ``cos``.

    * value    : ``cos(g)``
    * gradient : ``-sin(g) ∇g``
    * hessian  : ``-sin(g) H_g - cos(g) (∇g ∇gᵀ)``
    """
    if arg.unbounded:
        return _unbounded(n)
    g = arg.value
    sin_g = iv.sin(g)
    cos_g = iv.cos(g)
    neg_sin_g = Interval.point(-1.0) * sin_g
    grad = _dscale(neg_sin_g, arg.grad)
    hess = _dsub(_dscale(neg_sin_g, arg.hess), _dscale(cos_g, _self_outer(arg.grad)))
    return _SparseAD(value=cos_g, grad=grad, hess=hess, n=n)


def _apply_tan(arg: _SparseAD, n: int) -> _SparseAD:
    """Chain rule through ``tan``.

    * value    : ``tan(g)``
    * gradient : ``sec²(g) ∇g``           with ``sec²(g) = 1 + tan²(g)``
    * hessian  : ``sec²(g) H_g + 2 tan(g) sec²(g) (∇g ∇gᵀ)``

    A box spanning an asymptote makes ``iv.tan`` (hence ``sec²``) unbounded, so
    the Hessian is unbounded and the certificate abstains — sound.
    """
    if arg.unbounded:
        return _unbounded(n)
    g = arg.value
    tan_g = iv.tan(g)
    sec2 = Interval.point(1.0) + tan_g * tan_g
    grad = _dscale(sec2, arg.grad)
    hess = _dadd(
        _dscale(sec2, arg.hess),
        _dscale(Interval.point(2.0) * tan_g * sec2, _self_outer(arg.grad)),
    )
    return _SparseAD(value=tan_g, grad=grad, hess=hess, n=n)


__all__ = ["IntervalAD", "Rank1Factor", "interval_hessian"]
