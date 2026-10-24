"""Exact recentring of large-offset variables before a solve (#1537 workstream C).

A variable whose box sits far from the origin -- ``x in [1e6, 1e6 + 3]`` -- is the
same model as ``x = z + 1e6, z in [0, 3]``, but not the same *numerics*: rows with
activities of ~1e6, objectives whose constant-free part is ~1e6 at every feasible
point, and polynomial expansions with ~c^k terms. The invariance harness
(``python/tests/test_1537_invariance.py``) measured the cost: under ``x = y - 1e6``
the corpus lost 23 of 69 certificates and four models came back with FALSE ones
(#1542, #1543), plus two crashes (#1544).

This module rebuilds such a model in recentred coordinates and maps the result
back. It is an exact change of variables -- no bound, row or objective is
approximated -- so the certified answer of the recentred model is the certified
answer of the original.

Two requirements came out of the entry experiment, both binding:

* **Constant folding.** Substituting ``x = z + c`` alone leaves ``(z + c) - c``
  wherever the user wrote ``x - c``. On ``st_e36`` that unfolded form certified
  *infeasible* on a feasible model (#1542). Every scalar affine subtree is
  therefore rebuilt as ``sum_i a_i z_i + k`` with the constants folded, so
  ``(z + c) - c`` becomes exactly ``z``. With folding the same experiment showed
  0 false certificates and 0 crashes (69 models).
* **Selectivity.** Only a variable whose |anchor| is at least ``ratio`` times its
  box width (and at least ``ratio``) is moved: recentring a well-scaled model is
  not free (st_e36: 89 -> 93 nodes, 11 -> 19 s), and a model with nothing to move
  is solved exactly as before.

Gated by ``DISCOPT_RECENTRE`` (default ON, ``=0`` opts out; CLAUDE.md §5); the
threshold by ``DISCOPT_RECENTRE_RATIO`` (default 100).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from fractions import Fraction
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
    Parameter,
    SolveResult,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

logger = logging.getLogger(__name__)

FLAG = "DISCOPT_RECENTRE"
RATIO_ENV = "DISCOPT_RECENTRE_RATIO"
DEFAULT_RATIO = 100.0


class RecentreUnsupported(ValueError):
    """The model has structure this pass does not rebuild; solve it unrecentred."""


def recentre_enabled() -> bool:
    """``DISCOPT_RECENTRE`` -- default ON since #1537 (graduated, CLAUDE.md §5);
    ``DISCOPT_RECENTRE=0`` restores the unrecentred solve exactly.

    Graduation panel (``discopt_benchmarks/scripts/recentre_graduation_panel.py``,
    run 2026-10-03 on ``main`` after #1588/#1592 plus the n-ary-sum convexity fix,
    20 s, OFF/ON interleaved per instance, load 5-7, 206 comparisons: the corpus as
    written, the corpus under ``x = y - c`` with c ~ 1e3 / 1e6, and the #1537
    generated families): ON vs OFF false certificates **0 vs 0**, published
    incumbents failing re-verification **0 vs 0**, certified **185 vs 167**, ON
    loses **0** certificates OFF keeps (OFF loses 18 ON keeps), total wall **612 s
    vs 881 s**, bound-neutral drift **0 of 49** untouched instances.

    The three losses that blocked the 2026-10-02 attempt are gone. ``nvs09`` (both
    shifts) was the translated multilinear monomial distributing into 1,024 terms,
    fixed by ``DISCOPT_LIFT_AFFINE_MONOMIALS`` (#1588). ``clay0303hfsg`` (1e3) was the
    convexity recognisers not seeing through the n-ary ``SumOverExpression`` this
    pass's constant folding emits (the perspective denominator ``0.001 + 0.999 y``):
    36 hull rows lost their CONVEX verdict and the model left the OA route. That is
    fixed in the recognisers (``_relax/convexity``), so a user who writes
    ``dm.sum`` gets it too. ``cvxnonsep_psig40r`` (1e3) certifies in both arms on
    this run (13 vs 23 nodes); it was not separately root-caused.
    """
    return os.environ.get(FLAG, "1") != "0"


def recentre_ratio() -> float:
    raw = os.environ.get(RATIO_ENV)
    ratio = DEFAULT_RATIO if raw is None else float(raw)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"{RATIO_ENV} must be a positive finite number, got {raw!r}")
    return ratio


#: A bound at or beyond this magnitude is "effectively open" for the one-sided
#: rule: ``[1e6, 1e15]`` and ``[1e6, inf)`` are both "x >= 1e6".
#:
#: **Limit (documented, not an oversight):** the anchor itself must be below this
#: magnitude, so a variable whose box sits entirely at ``|x| >= 1e10`` --
#: ``[1e15, 1e15 + 64]``, ``[2**52, 2**52 + 10]`` -- is never moved: both its
#: bounds read as open. Such a box is solved
#: exactly as with the flag OFF. Moving it would be exact (the shift is an
#: integer and the narrow rule's subtraction is Sterbenz-exact), but at that
#: magnitude the user's own constants are already below the box's ulp, so the
#: model as written is not the model they meant; nothing in the panel measured
#: it, and lowering the cutoff changes which one-sided boxes move.
_OPEN = 1e10


def plan_shifts(model: Model, ratio: float) -> dict[int, np.ndarray]:
    """``{id(var): c}`` for every variable worth moving, ``c`` shaped like the var.

    A component moves when either

    * **narrow offset** -- both bounds finite and ``|lb| >= ratio * max(1, ub-lb)``
      (anchor ``lb``); or
    * **one-sided offset** -- a finite ``lb >= ratio`` on a box at least as wide
      as that offset (an open ``ub`` counts as infinitely wide): anchor ``lb``;
      or an open ``lb`` with ``ub <= -ratio``: anchor ``ub``. A box that contains
      0 is never moved (#1609).
      Without this, ``x in [1e6, 1e15]`` keeps a user's ``x - 1e6`` unfolded and
      st_miqp4 / alan still certify infeasible / crash under a shift (#1543).

    Measured on the unshifted in-repo corpus at ratio 100: 4 of 66 models have a
    component to move (ex14_1_9 1, hda 2, heatexch_gen2 13, tanksize 12); the
    other 62 are left exactly as they were.

    The shift is the anchor rounded to an integer, so integer and binary domains
    stay integral.

    **The one-sided rule moves modest offsets too.** Its threshold is the
    absolute ``|lb| >= ratio``, not a ratio to a width it does not have, so a
    plain ``x >= 150`` with no upper bound IS moved (to ``z >= 0.25`` for
    ``lb = 150.25``) while ``x in [150, 200]`` is not (150 < 100 * 50). That is
    deliberate -- an open box has no width to compare against, and the #1543
    failures were exactly such boxes -- but it means the flag is not a no-op on a
    model with only moderately offset one-sided bounds. Bounds at or beyond
    :data:`_OPEN` count as open; see there for the large-magnitude limit.
    """
    plan: dict[int, np.ndarray] = {}
    for v in model._variables:
        lb = np.asarray(v.lb, dtype=np.float64)
        ub = np.asarray(v.ub, dtype=np.float64)
        lb_fin = np.isfinite(lb) & (np.abs(lb) < _OPEN)
        ub_fin = np.isfinite(ub) & (np.abs(ub) < _OPEN)
        width = np.where(lb_fin & ub_fin, ub - lb, np.inf)
        narrow = lb_fin & ub_fin & (np.abs(lb) >= ratio * np.maximum(1.0, width))
        # One-sided: the box is at least as wide as its offset (an open bound gives
        # width = inf). A fixed "open" cutoff alone missed x in [-1.3e6, 9.9987e9]
        # (a 1e10 bound after a shift) and left its (y - c)**2 unfolded -- the #1543
        # trigger, certified 262 vs the true 2 on shifted st_miqp2.
        #
        # A box that already contains 0 is never moved (#1609). Its user origin is
        # a point of the box, so there is no offset to remove -- only one to add:
        # j in [-1e3, 1e3] anchored at -1e3 put the optimum j* = 2.5e-4 at
        # z* = 1000.00025, where the QP route's relative tolerance reported j off
        # by 19 % as `optimal`. ``narrow`` already excludes 0 (|lb| >= width), so
        # this bites only the one-sided rules: a finite lb anchors only when
        # lb > 0, an open lb anchors at ub only when ub < 0. The #1543 trigger
        # (shifted st_miqp2, x1 in [-1.3e6, 9.9987e9]) is such a box; it is left
        # alone now and certifies the base optimum 2 unmoved (measured).
        from_lb = lb_fin & ~narrow & (lb >= ratio) & (width >= lb)
        from_ub = ub_fin & ~lb_fin & (ub <= -ratio)
        if np.any(narrow | from_lb | from_ub):
            plan[id(v)] = np.where(
                narrow | from_lb, np.round(lb), np.where(from_ub, np.round(ub), 0.0)
            )
    return plan


# ── affine canonicalisation ───────────────────────────────────────────────────

# An affine form: ({leaf_key: coef}, const, {leaf_key: leaf_node}).
_Aff = tuple[dict, float, dict]


def _scalar_index_key(node: IndexExpression) -> Optional[tuple]:
    base = node.base
    if not isinstance(base, Variable):
        return None
    idx = node.index
    if isinstance(idx, (int, np.integer)):
        idx = (int(idx),)
    if not (isinstance(idx, tuple) and all(isinstance(i, (int, np.integer)) for i in idx)):
        return None
    if len(idx) != len(base.shape):
        return None
    # Normalise negative indices the way numpy does (``T[-1]`` is ``T[2]`` on a
    # length-3 T): ``ravel_multi_index`` rejects them, which crashed the pass
    # (#1609). An index out of range either way is not ours to reinterpret.
    norm = []
    for i, n in zip(idx, base.shape):
        i = int(i)
        if not -n <= i < n:
            return None
        norm.append(i + n if i < 0 else i)
    return (id(base), int(np.ravel_multi_index(tuple(norm), base.shape)))


def _const_value(node) -> Optional[float]:
    if isinstance(node, Constant):
        val = np.asarray(node.value)
        if val.ndim == 0:
            return float(val)
    return None


def _canonical(aff: _Aff) -> Expression:
    coefs, const, leaves = aff
    terms: list[Expression] = []
    for key, a in coefs.items():
        if a == 0.0:
            continue
        leaf = leaves[key]
        terms.append(leaf if a == 1.0 else Constant(a) * leaf)
    if const != 0.0 or not terms:
        terms.append(Constant(const))
    return terms[0] if len(terms) == 1 else SumOverExpression(terms)


#: Stands in for the rebuilt node of an affine subtree, which is never used: an
#: affine child is always emitted as ``_canonical(aff)``.
_AFFINE_ONLY: Any = object()


class _Rewriter:
    """``x -> z + c`` substitution with every scalar affine subtree folded."""

    def __init__(self, var_map: dict[int, Variable], shifts: dict[int, np.ndarray]):
        self.var_map = var_map  # id(old var) -> new var
        self.shifts = shifts  # id(old var) -> c (only moved vars)
        self.memo: dict[int, tuple[Expression, Optional[_Aff]]] = {}
        # One folded node per source node. ``_canonical`` builds a fresh node, so
        # without this a shared subexpression is DE-shared: ``y * y`` (one
        # Variable node on both sides) became ``(z + c) * (z + c)`` with two
        # distinct sums, and the square rule (``left is right``) lost it:
        # ``min x*x + w`` with a moved ``x`` went CONVEX -> not convex under
        # recentring (test_1537_recentre.py::test_shared_square_keeps_its_verdict).
        # The key is ``id(aff)``; every ``aff`` is kept alive by ``self.memo``, so
        # ids are not reused.
        self.emitted: dict[int, Expression] = {}

    def __call__(self, expr: Expression) -> Expression:
        node, aff = self._rw(expr)
        return self._emit(node, aff)

    def _rw(self, expr: Expression) -> tuple[Expression, Optional[_Aff]]:
        key = id(expr)
        hit = self.memo.get(key)
        if hit is None:
            hit = self._rw_impl(expr)
            self.memo[key] = hit
        return hit

    def _leaf_aff(self, node: Expression, leaf_key: tuple, shift: float) -> _Aff:
        return ({leaf_key: 1.0}, float(shift), {leaf_key: node})

    def _rw_impl(self, expr: Expression) -> tuple[Expression, Optional[_Aff]]:
        if isinstance(expr, Constant):
            v = _const_value(expr)
            return expr, (None if v is None else ({}, v, {}))
        if isinstance(expr, Parameter):
            return expr, None  # mutable: never folded into a constant
        if isinstance(expr, Variable):
            z = self.var_map[id(expr)]
            c = self.shifts.get(id(expr))
            if z.shape:
                node = z if c is None else z + Constant(c)
                return node, None  # array-valued: rewritten, not folded
            shift = 0.0 if c is None else float(np.asarray(c))
            node = z if shift == 0.0 else z + Constant(shift)
            return node, self._leaf_aff(z, (id(z), None), shift)
        if isinstance(expr, IndexExpression):
            if isinstance(expr.base, Variable):
                z = self.var_map[id(expr.base)]
                c = self.shifts.get(id(expr.base))
                leaf = IndexExpression(z, expr.index)
                ck = c[expr.index] if c is not None else None
                lkey = _scalar_index_key(leaf)
                if lkey is not None:
                    shift = 0.0 if ck is None else float(np.asarray(ck))
                    node = leaf if shift == 0.0 else leaf + Constant(shift)
                    return node, self._leaf_aff(leaf, lkey, shift)
                if ck is None or not np.any(ck):
                    return leaf, None
                return leaf + Constant(np.asarray(ck, dtype=np.float64)), None
            return IndexExpression(self(expr.base), expr.index), None
        # An affine node is only ever emitted as ``_canonical(aff)`` (``_emit``), so
        # its rebuilt node is not built (#1929 follow-up): building it emitted --
        # canonicalised -- both children, so a sum chain of n terms canonicalised
        # every prefix, O(n^2) nodes (48.7 s of a 60 s limit on gams04, 4.1M
        # products for 122k nonzeros). ``_AFFINE_ONLY`` marks such a node.
        if isinstance(expr, UnaryOp):
            child, caff = self._rw(expr.operand)
            if expr.op == "neg" and caff is not None:
                coefs, const, leaves = caff
                return _AFFINE_ONLY, ({k: -a for k, a in coefs.items()}, -const, leaves)
            return UnaryOp(expr.op, self._emit(child, caff)), None
        if isinstance(expr, BinaryOp):
            left, laff = self._rw(expr.left)
            right, raff = self._rw(expr.right)
            aff = _combine(expr.op, laff, raff)
            if aff is not None:
                return _AFFINE_ONLY, aff
            return BinaryOp(expr.op, self._emit(left, laff), self._emit(right, raff)), None
        if type(expr) is FunctionCall:
            return FunctionCall(expr.func_name, *(self(a) for a in expr.args)), None
        if isinstance(expr, MatMulExpression):
            return MatMulExpression(self(expr.left), self(expr.right)), None
        if isinstance(expr, SumExpression):
            return SumExpression(self(expr.operand), axis=expr.axis), None
        if isinstance(expr, SumOverExpression):
            parts = [self._rw(t) for t in expr.terms]
            if all(a is not None for _, a in parts):
                acc: _Aff = ({}, 0.0, {})
                for _, a in parts:
                    acc = _combine("+", acc, a)  # type: ignore[assignment]
                return _AFFINE_ONLY, acc
            return SumOverExpression([self._emit(n, a) for n, a in parts]), None
        raise RecentreUnsupported(f"no rewrite rule for {type(expr).__name__}")

    def _emit(self, node: Expression, aff: Optional[_Aff]) -> Expression:
        """A child as it appears under a NON-affine parent: folded if affine.

        The same source node always emits the same object (``self.emitted``), so
        the rewrite preserves the DAG's sharing."""
        if aff is None or (node is not _AFFINE_ONLY and _is_leaf(node)):
            return node
        out = self.emitted.get(id(aff))
        if out is None:
            out = _canonical(aff)
            self.emitted[id(aff)] = out
        return out


def _is_leaf(node: Expression) -> bool:
    return isinstance(node, (Constant, Parameter, Variable)) or (
        isinstance(node, IndexExpression) and isinstance(node.base, Variable)
    )


def _combine(op: str, a: Optional[_Aff], b: Optional[_Aff]) -> Optional[_Aff]:
    if a is None or b is None:
        return None
    (ca, ka, la), (cb, kb, lb) = a, b
    if op in ("+", "-"):
        sign = 1.0 if op == "+" else -1.0
        coefs = dict(ca)
        for k, v in cb.items():
            coefs[k] = coefs.get(k, 0.0) + sign * v
        return coefs, ka + sign * kb, {**la, **lb}
    if op == "*":
        if not ca:  # constant * affine
            return {k: ka * v for k, v in cb.items()}, ka * kb, lb
        if not cb:
            return {k: kb * v for k, v in ca.items()}, ka * kb, la
        return None
    if op == "/" and not cb and kb != 0.0:
        return {k: v / kb for k, v in ca.items()}, ka / kb, la
    return None


# ── the rebuilt model ─────────────────────────────────────────────────────────


@dataclass
class Recentring:
    """A recentred model plus the per-variable shifts to map points across."""

    model: Model
    shifts: dict[str, np.ndarray]  # var name -> c, for moved variables only

    def to_inner(self, values: dict[str, Any]) -> dict[str, Any]:
        """A point keyed by variable NAME, original coordinates -> recentred."""
        out = {}
        for name, val in values.items():
            c = self.shifts.get(name)
            out[name] = val if c is None else np.asarray(val, dtype=np.float64) - c
        return out

    def to_outer(self, values: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for name, val in values.items():
            c = self.shifts.get(name)
            out[name] = val if c is None else np.asarray(val, dtype=np.float64) + c
        return out


def _two_sum_error(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Knuth's TwoSum: the exact rounding error ``(a + b) - s`` of ``s = fl(a + b)``.

    Zero exactly when ``s`` is the exact sum. Only meaningful for finite inputs.
    """
    bp = s - a
    ap = s - bp
    err: np.ndarray = (a - ap) + (b - bp)
    return err


def shift_bounds(lb: np.ndarray, ub: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(lb - c, ub - c)`` rounded OUTWARD wherever the subtraction is inexact.

    The narrow rule's ``lb - c`` is exact (Sterbenz: ``c`` is ``lb`` rounded, so
    within a factor of 2), as is any shift of an integer bound below 2**53. The
    one-sided rule is not: ``x in [140.25, 1e17]`` moves by ``c = 140``, and
    ``1e17 - 140`` is not a double (ulp 16); round-to-nearest takes it to
    ``...856``, *inside* the original box. The inner box must contain the outer
    one, so an inexact lower bound steps down one ulp and an inexact upper bound
    steps up one ulp. Infinite bounds stay infinite.
    """
    lo = lb - c
    hi = ub - c
    with np.errstate(invalid="ignore"):
        err_lo = np.where(np.isfinite(lb), _two_sum_error(lb, -c, lo), 0.0)
        err_hi = np.where(np.isfinite(ub), _two_sum_error(ub, -c, hi), 0.0)
    # err = exact - rounded: an exact value below the rounded lower bound -> step down.
    lo = np.where(err_lo < 0.0, np.nextafter(lo, -np.inf), lo)
    hi = np.where(err_hi > 0.0, np.nextafter(hi, np.inf), hi)
    return lo, hi


def _exact(value: Fraction, what: str) -> float:
    """``value`` as a float, or :class:`RecentreUnsupported` if it is not one."""
    f = float(value)
    if Fraction(f) != value:
        raise RecentreUnsupported(
            f"{what} is not exactly representable after the shift (nearest double {f!r})"
        )
    return f


def _translate_rows(A, b: np.ndarray, c: np.ndarray, what: str) -> np.ndarray:
    """``b - A @ c`` row by row, exactly, or :class:`RecentreUnsupported`.

    ``A x <sense> b`` with ``x = z + c`` is ``A z <sense> b - A c``. The new rhs is
    accumulated in exact rational arithmetic over the nonzeros that touch a moved
    column, and refused when the exact value is not a double -- a rounded rhs
    would be a different row, which is the approximation this pass exists to
    avoid. Integer coefficients times integer shifts below 2**53 are always exact.
    """
    A = A.tocsr()
    c_flat = np.ravel(c)
    out = np.array(b, dtype=np.float64, copy=True)
    indptr, indices, data = A.indptr, A.indices, A.data
    for i in range(A.shape[0]):
        acc = Fraction(float(b[i]))
        touched = False
        for k in range(int(indptr[i]), int(indptr[i + 1])):
            cj = float(c_flat[int(indices[k])])
            if cj != 0.0 and data[k] != 0.0:
                acc -= Fraction(float(data[k])) * Fraction(cj)
                touched = True
        if touched:
            out[i] = _exact(acc, f"{what} row {i} rhs")
    return out


def _translate_builder_objective(model: Model, var_map, shifts_by_id, new: Model) -> None:
    """Move ``add_linear_objective`` / ``add_quadratic_objective`` into ``new``.

    Linear ``q'x + k`` becomes ``q'z + (k + q'c)``. Quadratic ``0.5 x'Qx + q'x + k``
    is first put in the builder's own convention -- the Rust builder optimises
    ``0.5 x'Sx`` with ``S = triu(Q) + striu(Q)'`` (``export/_common.py``) -- and
    then becomes ``0.5 z'Sz + (q + S c)'z + (k + q'c + 0.5 c'Sc)``. Every new
    coefficient is computed exactly or the pass refuses.
    """
    import scipy.sparse as sp

    lin_obj = model._builder_linear_objective
    quad_obj = model._builder_quadratic_objective
    if lin_obj is not None:
        q, x, k, sense = lin_obj
        z = var_map[id(x)]
        c = shifts_by_id.get(id(x))
        if c is None:
            new.add_linear_objective(q, z, constant=k, sense=sense)
            return
        q = np.asarray(q, dtype=np.float64).ravel()
        c_flat = np.ravel(c)
        acc = Fraction(float(k))
        for jj in np.nonzero((q != 0.0) & (c_flat != 0.0))[0]:
            acc += Fraction(float(q[jj])) * Fraction(float(c_flat[jj]))
        const = _exact(acc, "linear objective constant")
        new.add_linear_objective(q, z, constant=const, sense=sense)
        return
    if quad_obj is not None:
        Q, q, x, k, sense = quad_obj
        z = var_map[id(x)]
        c = shifts_by_id.get(id(x))
        if c is None:
            new.add_quadratic_objective(Q, q, z, constant=k, sense=sense)
            return
        S = (sp.triu(Q, 0) + sp.triu(Q, 1).T).tocsr()
        q = np.asarray(q, dtype=np.float64).ravel()
        c_flat = np.ravel(c)
        cF = [Fraction(float(v)) for v in c_flat]
        q_new = q.copy()
        acc = Fraction(float(k))
        for jj in np.nonzero((q != 0.0) & (c_flat != 0.0))[0]:
            acc += Fraction(float(q[jj])) * cF[jj]
        cSc = Fraction(0)
        for i in range(S.shape[0]):
            sc_i = Fraction(0)
            touched = False
            for kk in range(int(S.indptr[i]), int(S.indptr[i + 1])):
                j = int(S.indices[kk])
                if c_flat[j] != 0.0 and S.data[kk] != 0.0:
                    sc_i += Fraction(float(S.data[kk])) * cF[j]
                    touched = True
            if touched:
                q_new[i] = _exact(Fraction(float(q[i])) + sc_i, f"quadratic objective c[{i}]")
                cSc += cF[i] * sc_i
        const = _exact(acc + cSc / 2, "quadratic objective constant")
        new.add_quadratic_objective(S, q_new, z, constant=const, sense=sense)
        return
    raise RecentreUnsupported("the objective is a builder placeholder with no recorded block")


#: Model-resident state this pass does not translate. Each is refused when
#: non-empty rather than silently dropped from the inner model (CLAUDE.md §3):
#: piecewise domains and complementarities change the feasible set, and the
#: decomposition / block labels key solver structure by constraint identity.
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


def recentre(model: Model, ratio: Optional[float] = None) -> Optional[Recentring]:
    """The recentred twin of ``model``, or ``None`` when no variable qualifies.

    Every piece of model state that reaches a solve is either translated exactly
    or refused with :class:`RecentreUnsupported` (the caller then solves the model
    unrecentred and says so at WARNING):

    * variables -- bounds shifted, rounded outward when inexact (:func:`shift_bounds`);
    * expression rows and objective -- rewritten with constant folding;
    * builder-resident rows (``add_linear_constraints``, ``Model.constraint(...,
      fast=True)``) -- ``b - A c``, exactly (:func:`_translate_rows`);
    * builder-resident objective (``add_linear_objective`` /
      ``add_quadratic_objective``) -- linear and constant terms picked up exactly;
    * parameters -- shared objects (values stay live, never folded);
    * ``_initial_point`` and ``_gams_initial_values`` -- shifted per element;
    * refused: indicator / disjunctive / SOS / logical rows (any ``Constraint``
      subclass), a missing objective, an expression objective coexisting with a
      builder objective, and the fields in :data:`_REFUSED_FIELDS`.

    Not carried and not needed: ``_atan2_preconditions`` (re-checked on the
    original by ``validate()`` before this runs, and the rewritten expressions are
    the same functions), ``_gdp_factory_*`` (``validate()`` refuses unattached
    blocks), ``_simplex_lowerings`` / ``_sets`` (bookkeeping no solve reads), and
    ``_source_nl_path`` / ``_nl_repr`` (deliberately absent: the inner model is
    not the ``.nl`` file).

    The rewrite recurses once per expression level, so a deep model runs on the
    large-stack worker the solver uses for deep expressions (#266).
    """
    plan = plan_shifts(model, recentre_ratio() if ratio is None else ratio)
    if not plan:
        return None
    from discopt._relax.convexity.rules import _run_with_deep_recursion
    from discopt._relax.factorable_reform import _max_expr_node_count

    depth = _max_expr_node_count(model)
    need = 0 if depth <= 700 else min(4000 + 8 * depth, 1_000_000)
    return _run_with_deep_recursion(lambda: _build(model, plan), depth_need=need)


def _build(model: Model, plan: dict[int, np.ndarray]) -> Recentring:
    obj = model._objective
    if obj is None:
        raise RecentreUnsupported("model has no objective")
    placeholder = bool(getattr(obj, "_is_placeholder", False))
    has_builder_obj = (
        model._builder_linear_objective is not None
        or model._builder_quadratic_objective is not None
    )
    if has_builder_obj and not placeholder:
        raise RecentreUnsupported("an expression objective and a builder objective are both set")
    for con in model._constraints:
        if type(con) is not Constraint:
            raise RecentreUnsupported(f"a {type(con).__name__} row")
    for attr in _REFUSED_FIELDS:
        if getattr(model, attr, None):
            raise RecentreUnsupported(f"model carries {attr}")

    new = Model(f"{model.name}")
    var_map: dict[int, Variable] = {}
    shifts_by_id: dict[int, np.ndarray] = {}
    shifts_by_name: dict[str, np.ndarray] = {}
    for v in model._variables:
        c = plan.get(id(v))
        lb, ub = np.asarray(v.lb, dtype=np.float64), np.asarray(v.ub, dtype=np.float64)
        if c is not None:
            c = np.broadcast_to(c, lb.shape).astype(np.float64)
            lb, ub = shift_bounds(lb, ub, c)
            shifts_by_id[id(v)] = c if v.shape else np.asarray(float(c))
            shifts_by_name[v.name] = shifts_by_id[id(v)]
        shape = v.shape if v.shape else ()
        if v.var_type is VarType.CONTINUOUS:
            z = new.continuous(v.name, shape=shape, lb=lb, ub=ub)
        elif v.var_type is VarType.BINARY and c is None:
            z = new.binary(v.name, shape=shape)
        else:
            z = new.integer(v.name, shape=shape, lb=lb, ub=ub)
        var_map[id(v)] = z
    for p in model._parameters:  # shared objects: values stay live, names unchanged
        new._parameters.append(p)
        new._names.add(p.name)

    rw = _Rewriter(var_map, shifts_by_id)
    for con in model._constraints:
        new._constraints.append(
            Constraint(body=rw(con.body), sense=con.sense, rhs=0.0, name=con.name)
        )
    for A, x, sense, b, name in model._builder_linear_blocks:
        c = shifts_by_id.get(id(x))
        b_new = b if c is None else _translate_rows(A, b, c, f"builder block {name!r}")
        new.add_linear_constraints(A, var_map[id(x)], sense, b_new, name=name)
    if placeholder:
        _translate_builder_objective(model, var_map, shifts_by_id, new)
    else:
        new._objective = Objective(expression=rw(obj.expression), sense=obj.sense)

    # ``_initial_point`` is keyed ``(var name, flat element) -> value`` (set by
    # ``set_initial_point`` / ``from_nl``); move each entry into the new coordinates.
    new._initial_point = {
        (name, elem): float(val) - float(np.ravel(shifts_by_name[name])[elem])
        if name in shifts_by_name
        else val
        for (name, elem), val in getattr(model, "_initial_point", {}).items()
    }
    # ``from_gams`` start: ``name -> float`` (whole variable) or ``{flat: float}``.
    gams_iv = getattr(model, "_gams_initial_values", None)
    if gams_iv:
        moved: dict[str, Any] = {}
        for name, entry in gams_iv.items():
            c = shifts_by_name.get(name)
            if c is None:
                moved[name] = entry
                continue
            c_flat = np.ravel(c)
            if isinstance(entry, dict):
                moved[name] = {i: float(v) - float(c_flat[int(i)]) for i, v in entry.items()}
            elif c_flat.size == 1:
                moved[name] = float(entry) - float(c_flat[0])
            else:
                moved[name] = {i: float(entry) - float(c_flat[i]) for i in range(c_flat.size)}
        new._gams_initial_values = moved  # type: ignore[attr-defined]
    return Recentring(model=new, shifts=shifts_by_name)


def solve_recentred(outer: Model, rc: Recentring, solve_args: dict[str, Any]) -> SolveResult:
    """Solve ``rc.model`` with ``outer.solve``'s arguments; map the result back.

    The inner solve runs the whole ``Model.solve`` pipeline (every route, the
    convex kernel included). Whatever its status, a mapped point is then
    re-verified against the ORIGINAL model with :func:`verify_point`; a point the
    user's model rejects is withheld exactly as the #772 false-primal guard
    withholds one (``x=None``, ``objective=None``, ``status="error"``,
    ``incumbent_verification_failed=True``; the bound is untouched). The derived
    reports -- ``sensitivity``, the ``validate`` examiner and the LLM explanation
    -- are switched off for the inner solve; the caller computes them on the
    ORIGINAL model.
    """
    import dataclasses

    from discopt.modeling.core import _withhold_false_primal
    from discopt.validation.feasibility import verify_point

    args = dict(solve_args)
    for derived in ("llm", "sensitivity", "validate"):
        args[derived] = False
    init = args.get("initial_solution")
    if init:
        by_name = {getattr(k, "name", k): v for k, v in init.items()}
        inner_vars = {v.name: v for v in rc.model._variables}
        args["initial_solution"] = {inner_vars[n]: val for n, val in rc.to_inner(by_name).items()}
    ws = args.get("warm_start")
    if ws is not None and ws.x is not None:
        args["warm_start"] = dataclasses.replace(ws, x=rc.to_inner(ws.x))

    rc.model._recentre_inner = True  # type: ignore[attr-defined]
    result = rc.model.solve(**args)
    if not isinstance(result, SolveResult):  # the call site excludes streaming solves
        raise TypeError(f"recentring expected a SolveResult, got {type(result).__name__}")

    outer_names = [v.name for v in outer._variables]
    if result.x:
        # Inner-only columns (``_fr_aux_*`` and the like) describe the inner model,
        # not the user's; only the declared names are mapped back.
        result.x = rc.to_outer({n: result.x[n] for n in outer_names if n in result.x})
    result._model = outer
    if result.infeasibility_certificate is not None:
        # A row-violation witness of the recentred LP; its rows are the same, but it
        # was computed in z-space, so it is not handed out as the original's.
        result.infeasibility_certificate = None
    stats = result.solver_stats if result.solver_stats is not None else {}
    result.solver_stats = stats
    stats["recentre/variables_moved"] = float(len(rc.shifts))

    if result.x:
        missing = [n for n in outer_names if n not in result.x]
        reason: Optional[str]
        if missing:
            reason = f"the inner result carries no value for {missing[:3]}"
        else:
            flat = np.concatenate(
                [np.ravel(np.asarray(result.x[n], dtype=np.float64)) for n in outer_names]
            )
            check = verify_point(outer, flat)
            reason = None if check.ok else str(check.reason)
        if reason is not None:
            logger.error(
                "recentring: the mapped incumbent fails the original model (%s); "
                "withholding it (false-primal guard, #772)",
                reason,
            )
            _withhold_false_primal(
                result,
                "the incumbent mapped back from the recentred model is infeasible in "
                f"the original model ({reason}); it was withheld (false-primal guard, #772)",
            )
            stats["recentre/mapped_point_refused"] = 1.0
    return result
