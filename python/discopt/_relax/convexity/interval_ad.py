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
* ``hess`` as an :class:`_HMap` — only the non-zero entries of the
  *upper triangle* (the Hessian is symmetric), ``(i, j)`` with ``i ≤ j``,
  held as parallel key / ``lo`` / ``hi`` arrays so a dense outer product
  ``∇g ∇gᵀ`` costs numpy calls, not one Python interval op per entry.

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

import sys
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
    return _expr_node_count_capped(expr, limit) > limit


def _expr_node_count_capped(expr: Expression, limit: int) -> int:
    """Distinct-node count of ``expr``'s DAG, stopping at ``limit + 1``."""
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
            return count
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
    return count


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
    * ``hess`` — :class:`_HMap` of entries ``(i, j)``, ``i ≤ j`` (upper
      triangle of the symmetric Hessian); absent key ⇒ exact-zero entry.
    * ``n`` — flat variable count (for densification).
    * ``unbounded`` — ``True`` when an unsupported / non-smooth atom
      forced an abstention; densifies to a ``±inf`` Hessian so the
      certificate refuses to certify.
    * ``rank1`` — optional sparse rank-1 metadata.
    """

    value: Interval
    grad: "dict[int, Interval]"
    hess: "_HMap"
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
# Gradient maps are ``{slot: Interval}`` dicts of scalar intervals and go
# through the scalar :class:`Interval` operators. Hessian maps are
# :class:`_HMap` — the same sparse upper triangle held as parallel arrays — and
# go through the element-wise kernels ``_v_*`` below, which reproduce the
# scalar :class:`Interval` operators **bit for bit, entry by entry** (same
# corner products, same NaN-corner convention, same per-entry #957 exact-zero
# rule). Both are outward-rounded, so the maps are sound enclosures entry by
# entry. The whole walk runs under a single ``np.errstate`` (see
# :func:`interval_hessian`) so that intentional overflow / ``0 * inf`` on wide
# boxes — which produce the ``±inf`` sentinels the certificate reads as
# "abstain" — do not emit benchmark-visible warnings.
#
# Why the Hessian is vectorised (#1544 follow-up): a chain-rule step through a
# nonlinear atom adds the outer product ``∇g ∇gᵀ``, so ``sin(sum(x))`` over
# ``n`` variables carries ``n(n+1)/2`` Hessian entries. As a dict of scalar
# intervals that was one Python-level interval multiply (~17 µs) per entry per
# chain-rule step — ~1.1 M of them, ~19 s, for a 600-term sum, which put
# ``test_shallow_model_with_wide_sum_certifies`` at the edge of its time limit
# on CI. As arrays it is a handful of numpy calls.


def _dadd(a: dict, b: dict) -> dict:
    """Entry-wise sum of two sparse gradient maps."""
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
    """Entry-wise difference ``a - b`` of two sparse gradient maps."""
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
    """Scale every entry of a sparse gradient map by the scalar interval ``s``."""
    if not a:
        return {}
    return {k: s * v for k, v in a.items()}


# -- element-wise interval kernels (bit-identical to the scalar operators) ----


def _v_add(alo, ahi, blo, bhi):
    """Element-wise :meth:`Interval.__add__`."""
    return iv._round_down_exact0(alo + blo), iv._round_up_exact0(ahi + bhi)


def _v_sub(alo, ahi, blo, bhi):
    """Element-wise :meth:`Interval.__sub__`."""
    return iv._round_down_exact0(alo - bhi), iv._round_up_exact0(ahi - blo)


def _v_mul(alo, ahi, blo, bhi):
    """Element-wise :meth:`Interval.__mul__`, deciding the #957 rule per entry.

    ``Interval.__mul__`` on an *array* takes its exact-zero decision once for the
    whole array (one underflowed entry nudges every exact-zero endpoint). Each
    scalar entry of a sparse map was multiplied on its own, so this kernel
    decides per entry — which is what makes it bit-identical to the scalar path.
    """
    with np.errstate(invalid="ignore"):
        a = alo * blo
        b = alo * bhi
        c = ahi * blo
        d = ahi * bhi
        lo = np.minimum(np.minimum(a, b), np.minimum(c, d))
        hi = np.maximum(np.maximum(a, b), np.maximum(c, d))
        if bool(np.isnan(lo).any()) or bool(np.isnan(hi).any()):
            # ``0 * ±inf -> 0`` (C-36). The map is the identity on every non-NaN
            # corner, so applying it to all entries changes only the NaN ones.
            a = iv._nan_corner_to_zero(a)
            b = iv._nan_corner_to_zero(b)
            c = iv._nan_corner_to_zero(c)
            d = iv._nan_corner_to_zero(d)
            lo = np.minimum(np.minimum(a, b), np.minimum(c, d))
            hi = np.maximum(np.maximum(a, b), np.maximum(c, d))
    underflowed = ((lo == 0.0) | (hi == 0.0)) & (
        ((a == 0.0) & (alo != 0.0) & (blo != 0.0))
        | ((b == 0.0) & (alo != 0.0) & (bhi != 0.0))
        | ((c == 0.0) & (ahi != 0.0) & (blo != 0.0))
        | ((d == 0.0) & (ahi != 0.0) & (bhi != 0.0))
    )
    # Without a zero endpoint the exact0 and plain roundings coincide, so the
    # only entries that take the plain (always-nudge) rounding are underflows.
    out_lo = np.where(underflowed, iv._round_down(lo), iv._round_down_exact0(lo))
    out_hi = np.where(underflowed, iv._round_up(hi), iv._round_up_exact0(hi))
    return out_lo, out_hi


def _v_sq(lo, hi):
    """Element-wise :meth:`Interval.__pow__` with ``n == 2`` (per-entry #957 rule)."""
    zero_in = (lo <= 0) & (hi >= 0)
    lo_sq = lo * lo
    hi_sq = hi * hi
    s_lo = np.where(zero_in, 0.0, np.minimum(lo_sq, hi_sq))
    s_hi = np.maximum(lo_sq, hi_sq)
    underflowed = ((s_lo == 0.0) | (s_hi == 0.0)) & (
        ((lo_sq == 0.0) & (lo != 0.0)) | ((hi_sq == 0.0) & (hi != 0.0))
    )
    out_lo = np.where(underflowed, iv._round_down(s_lo), iv._round_down_exact0(s_lo))
    out_hi = np.where(underflowed, iv._round_up(s_hi), iv._round_up_exact0(s_hi))
    return out_lo, out_hi


# -- sparse Hessian maps --------------------------------------------------------

# Entry ``(i, j)`` (``i <= j``) is stored under the int64 key ``i * _KEY_SHIFT + j``,
# so sorting the keys orders entries row-major and a key round-trips exactly.
_KEY_SHIFT = np.int64(1 << 32)


@dataclass(frozen=True, slots=True)
class _HMap:
    """Sparse upper-triangle Hessian: sorted unique ``keys`` with ``[lo, hi]``.

    An absent key is a structural (exact) zero, exactly as an absent dict key
    was. Instances are never mutated, so they may be shared between nodes.
    """

    keys: np.ndarray  # int64, sorted, unique
    lo: np.ndarray  # float64
    hi: np.ndarray  # float64

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def rows_cols(self) -> tuple[np.ndarray, np.ndarray]:
        return self.keys // _KEY_SHIFT, self.keys % _KEY_SHIFT


_H0 = _HMap(
    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
)


def _hmerge(a: _HMap, b: _HMap, op, only_b) -> _HMap:
    """Union of two maps: ``op`` on shared keys, ``a`` / ``only_b(b)`` elsewhere.

    Entries present in only one operand are carried over untouched, as the dict
    helpers did (combining with an implicit zero would add a rounding step).
    """
    keys = np.union1d(a.keys, b.keys)
    ia = np.searchsorted(keys, a.keys)
    ib = np.searchsorted(keys, b.keys)
    in_a = np.zeros(keys.shape[0], dtype=bool)
    in_a[ia] = True
    in_b = np.zeros(keys.shape[0], dtype=bool)
    in_b[ib] = True
    lo = np.empty(keys.shape[0], dtype=np.float64)
    hi = np.empty(keys.shape[0], dtype=np.float64)
    lo[ia] = a.lo
    hi[ia] = a.hi
    b_lo, b_hi = only_b(b.lo, b.hi)
    b_only = ~in_a[ib]
    lo[ib[b_only]] = b_lo[b_only]
    hi[ib[b_only]] = b_hi[b_only]
    shared_b = ~b_only
    if bool(shared_b.any()):
        pos = ib[shared_b]
        s_lo, s_hi = op(lo[pos], hi[pos], b.lo[shared_b], b.hi[shared_b])
        lo[pos] = s_lo
        hi[pos] = s_hi
    return _HMap(keys, lo, hi)


def _hadd(a: _HMap, b: _HMap) -> _HMap:
    """Entry-wise sum of two sparse Hessian maps."""
    if not len(a):
        return b
    if not len(b):
        return a
    return _hmerge(a, b, _v_add, lambda lo, hi: (lo, hi))


def _hsub(a: _HMap, b: _HMap) -> _HMap:
    """Entry-wise difference ``a - b`` of two sparse Hessian maps."""
    if not len(b):
        return a
    if not len(a):
        return _hneg(b)
    return _hmerge(a, b, _v_sub, lambda lo, hi: (-hi, -lo))


def _hneg(a: _HMap) -> _HMap:
    """Entry-wise negation (exact — sign flip carries no roundoff)."""
    if not len(a):
        return a
    return _HMap(a.keys, -a.hi, -a.lo)


def _hscale(s: Interval, a: _HMap) -> _HMap:
    """Scale every entry of a sparse Hessian map by the scalar interval ``s``."""
    if not len(a):
        return _H0
    lo, hi = _v_mul(s.lo, s.hi, a.lo, a.hi)
    return _HMap(a.keys, lo, hi)


def _grad_arrays(g: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(slots, lo, hi)`` of a sparse gradient map, in insertion order."""
    m = len(g)
    idx = np.fromiter(g.keys(), dtype=np.int64, count=m)
    lo = np.fromiter((float(v.lo) for v in g.values()), dtype=np.float64, count=m)
    hi = np.fromiter((float(v.hi) for v in g.values()), dtype=np.float64, count=m)
    return idx, lo, hi


def _from_pairs(i: np.ndarray, j: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> _HMap:
    """Build an :class:`_HMap` from (unsorted, duplicate-free) entry coordinates."""
    keys = np.minimum(i, j) * _KEY_SHIFT + np.maximum(i, j)
    order = np.argsort(keys, kind="stable")
    return _HMap(keys[order], lo[order], hi[order])


def _self_outer(g: dict) -> _HMap:
    """Upper-triangle of ``g gᵀ`` with dependency-aware tightening.

    The diagonal uses the squaring rule (``gᵢ²`` is nonneg and bracketed
    by ``[0, max(|loᵢ|, |hiᵢ|)²]``), matching the dense self-outer
    specialisation — essential for Hessian enclosures tight enough for
    Gershgorin to certify compositions like ``exp(x²)``. Off-diagonal
    entries are the general corner products ``gᵢ · gⱼ``.
    """
    if not g:
        return _H0
    idx, glo, ghi = _grad_arrays(g)
    pa, pb = np.triu_indices(idx.shape[0])
    lo, hi = _v_mul(glo[pa], ghi[pa], glo[pb], ghi[pb])
    diag = pa == pb
    d_lo, d_hi = _v_sq(glo, ghi)
    lo[diag] = d_lo
    hi[diag] = d_hi
    return _from_pairs(idx[pa], idx[pb], lo, hi)


def _sym_cross(a: dict, b: dict) -> _HMap:
    """Upper-triangle of the symmetric matrix ``a bᵀ + b aᵀ``.

    Mirrors the dense ``_outer(a, b) + _outer(b, a)`` term: entry
    ``(i, j)`` is ``aᵢ bⱼ + aⱼ bᵢ`` (absent keys count as exact zero).
    When ``a is b`` the result is ``2 · a aᵀ`` and is routed through
    :func:`_self_outer` so the diagonal keeps the squaring tightening.
    """
    if a is b:
        return _hscale(_TWO, _self_outer(a))
    if not a or not b:
        return _H0
    keys = np.array(sorted(set(a) | set(b)), dtype=np.int64)
    k = keys.shape[0]

    def _dense(d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        present = np.zeros(k, dtype=bool)
        lo = np.zeros(k, dtype=np.float64)
        hi = np.zeros(k, dtype=np.float64)
        idx, d_lo, d_hi = _grad_arrays(d)
        pos = np.searchsorted(keys, idx)
        present[pos] = True
        lo[pos] = d_lo
        hi[pos] = d_hi
        return present, lo, hi

    pa_, alo, ahi = _dense(a)
    pb_, blo, bhi = _dense(b)
    ix, jx = np.triu_indices(k)
    has1 = pa_[ix] & pb_[jx]  # aᵢ bⱼ
    has2 = pa_[jx] & pb_[ix]  # aⱼ bᵢ
    keep = has1 | has2
    ix, jx, has1, has2 = ix[keep], jx[keep], has1[keep], has2[keep]
    t1_lo, t1_hi = _v_mul(alo[ix], ahi[ix], blo[jx], bhi[jx])
    t2_lo, t2_hi = _v_mul(alo[jx], ahi[jx], blo[ix], bhi[ix])
    s_lo, s_hi = _v_add(t1_lo, t1_hi, t2_lo, t2_hi)
    both = has1 & has2
    lo = np.where(both, s_lo, np.where(has1, t1_lo, t2_lo))
    hi = np.where(both, s_hi, np.where(has1, t1_hi, t2_hi))
    # ``keys`` is sorted and ix <= jx, so these keys are already sorted & unique.
    return _HMap(keys[ix] * _KEY_SHIFT + keys[jx], lo, hi)


_AFFINE_HESS_TOL = 1e-300


def _hess_is_exactly_zero(hess: _HMap) -> bool:
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
    return not bool(np.any((np.abs(hess.lo) > tol) | (np.abs(hess.hi) > tol)))


# ──────────────────────────────────────────────────────────────────────
# Sentinels / densification
# ──────────────────────────────────────────────────────────────────────


def _unbounded(n: int) -> _SparseAD:
    """Abstention sentinel: densifies to a ``±inf`` Hessian."""
    inf = np.float64(np.inf)
    return _SparseAD(
        value=Interval(-inf, inf),
        grad={},
        hess=_H0,
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
    rows, cols = sad.hess.rows_cols()
    hlo[rows, cols] = sad.hess.lo
    hhi[rows, cols] = sad.hess.hi
    off = rows != cols
    hlo[cols[off], rows[off]] = sad.hess.lo[off]
    hhi[cols[off], rows[off]] = sad.hess.hi[off]
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
    n_nodes = _expr_node_count_capped(expr, _INTERVAL_HESSIAN_MAX_NODES)
    if n_nodes > _INTERVAL_HESSIAN_MAX_NODES:
        raise IntervalHessianTooLarge(
            f"expression DAG exceeds {_INTERVAL_HESSIAN_MAX_NODES} nodes; "
            "interval-Hessian walk declined to protect the time budget (#654)"
        )
    box = box or {}
    cache: dict = {}

    def _run() -> IntervalAD:
        # Wide boxes intentionally overflow to ``±inf`` / ``0 * inf`` (the
        # abstention sentinels the certificate reads as UNKNOWN). Those are
        # sound interval results, not numerical bugs — suppress the warnings
        # for the whole walk, mirroring the dense path's per-op errstate.
        with np.errstate(over="ignore", invalid="ignore"):
            sad = _walk(expr, model, box, cache, n)
            return _densify(sad, n)

    # #1520: ``_walk`` recurses once per DAG level (a few frames each), so a
    # left-folded sum of a few hundred terms -- well inside the node budget --
    # exhausted the default 1000-frame limit and raised ``RecursionError``, which
    # callers' catch-alls read as "no verdict". Depth is bounded by the node
    # count, so run deep bodies on the large-stack runner the convexity walk
    # already uses (#266) instead of abstaining by accident.
    from .rules import _run_with_deep_recursion

    return _run_with_deep_recursion(
        _run, depth_need=_stack_depth() + _WALK_FRAMES_PER_NODE * n_nodes + 200
    )


def _stack_depth() -> int:
    """Number of Python frames currently on this thread's stack."""
    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


#: Upper bound on Python frames ``_walk`` uses per DAG level (``_walk`` ->
#: ``_impl`` -> a per-op helper -> ``_walk``), for the recursion-limit estimate.
_WALK_FRAMES_PER_NODE = 4


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


def _scalar_leaf_interval(expr: Expression) -> Interval:
    """Degenerate interval for a *scalar-valued* ``Constant``/``Parameter`` leaf.

    Raises ``ValueError`` — the walker's established abstain signal, which
    :func:`~.certificate.certify_convex` catches — when the leaf holds an
    array rather than a scalar. The vectorized / indexed-summation modeling
    API builds exactly such leaves (``m.parameter("cost", value=np.array(...))``
    multiplied elementwise by an array variable), and ``float(np.asarray(v))``
    on one raises ``TypeError: only 0-dimensional arrays can be converted to
    Python scalars``. That ``TypeError`` escaped ``certify_convex``'s
    ``except ValueError`` and was swallowed by the caller's broad
    ``except Exception``, turning a *structural* refusal ("this walker is
    scalar-only; array leaves are out of scope", the same refusal array
    ``Variable``s already get) into an accidental one that read as
    convexity-unknown (issue #936, Defect 3). Abstaining deliberately keeps
    the verdict identical while making the reason visible in the logs.

    The conversion itself is left exactly as it was — ``float(np.asarray(...))``,
    so every leaf that walked before still walks and none that didn't now does.
    Only the *exception type* on refusal changes.
    """
    arr = np.asarray(expr.value)
    try:
        v = float(arr)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Interval Hessian requires scalar {type(expr).__name__} leaves; got "
            f"shape {arr.shape}. Array-valued leaves come from the vectorized "
            "modeling API and are out of scope for the scalar interval walker."
        ) from exc
    return Interval(np.float64(v), np.float64(v))


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
        return _SparseAD(value=_scalar_leaf_interval(expr), grad={}, hess=_H0, n=n)

    if isinstance(expr, Parameter):
        return _SparseAD(value=_scalar_leaf_interval(expr), grad={}, hess=_H0, n=n)

    if isinstance(expr, Variable):
        if expr.size != 1:
            raise ValueError(f"Interval Hessian requires scalar variables; got shape {expr.shape}")
        slot = _var_offset(expr, model)
        val = _variable_scalar_value(expr, box)
        return _SparseAD(value=val, grad={slot: _ONE}, hess=_H0, n=n)

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
        return _SparseAD(value=val, grad={slot: _ONE}, hess=_H0, n=n)

    # --- Unary ops --------------------------------------------------
    if isinstance(expr, UnaryOp):
        child = _walk(expr.operand, model, box, cache, n)
        if child.unbounded:
            return _unbounded(n)
        if expr.op == "neg":
            return _SparseAD(
                value=-child.value,
                grad=_dneg(child.grad),
                hess=_hneg(child.hess),
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
            return _SparseAD(value=Interval.point(0.0), grad={}, hess=_H0, n=n)
        result = _walk(expr.terms[0], model, box, cache, n)
        if result.unbounded:
            return _unbounded(n)
        value = result.value
        grad = dict(result.grad)
        hess = result.hess
        for t in expr.terms[1:]:
            other = _walk(t, model, box, cache, n)
            if other.unbounded:
                return _unbounded(n)
            value = value + other.value
            grad = _dadd(grad, other.grad)
            hess = _hadd(hess, other.hess)
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
            hess=_hadd(left.hess, right.hess),
            n=n,
        )

    if expr.op == "-":
        return _SparseAD(
            value=left.value - right.value,
            grad=_dsub(left.grad, right.grad),
            hess=_hsub(left.hess, right.hess),
            n=n,
        )

    if expr.op == "*":
        # (f g)'  = g f' + f g'
        # (f g)'' = g f'' + f g'' + f' g'ᵀ + g' f'ᵀ
        fg = left.value * right.value
        grad = _dadd(_dscale(right.value, left.grad), _dscale(left.value, right.grad))
        hess = _hadd(
            _hadd(_hscale(right.value, left.hess), _hscale(left.value, right.hess)),
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
    recip_hess = _hsub(
        _hscale(_TWO * inv_g3, _self_outer(right.grad)),
        _hscale(inv_g2, right.hess),
    )

    # f / g = f * (1/g) — apply product rule.
    fg_val = left.value * recip_value
    fg_grad = _dadd(_dscale(recip_value, left.grad), _dscale(left.value, recip_grad))
    fg_hess = _hadd(
        _hadd(_hscale(recip_value, left.hess), _hscale(left.value, recip_hess)),
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

    hess = _hscale(c_combined, _self_outer(v_combined))

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
        return _SparseAD(value=_ONE, grad={}, hess=_H0, n=n_vars)
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
        hess=_hscale(pt, log_g.hess),
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
    hess = _hadd(_hscale(coeff1, base.hess), _hscale(coeff2, _self_outer(base.grad)))
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
    hess = _hsub(
        _hscale(_TWO / g3, _self_outer(gk.grad)),
        _hscale(_ONE / g2, gk.hess),
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
    if name == "entropy":
        # f = g log g (#1616 A-02): f' = (log g + 1) g',
        # f'' = (log g + 1) H_g + (1/g) (∇g ∇gᵀ). Requires g > 0 strictly on the
        # box (1/g is unbounded at 0), else abstain.
        if np.any(arg.value.lo <= 0):
            return _unbounded(n)
        g = arg.value
        d1 = iv.log(g) + _ONE
        inv_g = _ONE / g
        grad = _dscale(d1, arg.grad)
        hess = _hadd(_hscale(d1, arg.hess), _hscale(inv_g, _self_outer(arg.grad)))
        return _SparseAD(value=iv.entropy(g), grad=grad, hess=hess, n=n)
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
        hess = _hadd(_hscale(coeff1, arg.hess), _hscale(coeff2, _self_outer(arg.grad)))
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
    hess = _hscale(e, _hadd(arg.hess, _self_outer(arg.grad)))
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
    hess = _hsub(_hscale(inv_g, arg.hess), _hscale(inv_g2, _self_outer(arg.grad)))
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
    hess = _hsub(_hscale(cos_g, arg.hess), _hscale(sin_g, _self_outer(arg.grad)))
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
    hess = _hsub(_hscale(neg_sin_g, arg.hess), _hscale(cos_g, _self_outer(arg.grad)))
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
    hess = _hadd(
        _hscale(sec2, arg.hess),
        _hscale(Interval.point(2.0) * tan_g * sec2, _self_outer(arg.grad)),
    )
    return _SparseAD(value=tan_g, grad=grad, hess=hess, n=n)


__all__ = ["IntervalAD", "Rank1Factor", "interval_hessian"]
