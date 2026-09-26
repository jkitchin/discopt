"""``nonlinear_to_pwl``: replace univariate nonlinear terms by piecewise-linear MILP (#1482).

This is the *approximating* counterpart of :meth:`Model.piecewise`. Where
``piecewise`` declares a function that **is** its table, this transformation
takes a model whose terms are nonlinear -- ``exp(x)``, ``x**3 - 2*x``,
``sqrt(q)`` -- and replaces every maximal univariate nonlinear subexpression
``g(x)`` of a bounded scalar input by a piecewise-linear construct, so the model
becomes a MILP (or keeps only its multivariate nonlinear terms).

A PWL approximation is not a relaxation
---------------------------------------
Replacing ``g`` by its chord interpolant cuts off feasible points or admits
infeasible ones depending on curvature, so the approximated model's optimum is
**not** a bound on the original problem. Reporting it as certified would be a
false certificate. The two modes face that head-on:

``mode="outer"`` (the default) -- a sound outer approximation.
    ``g`` is replaced by a variable ``w`` tied to ``x`` by a disaggregated
    convex-combination MILP (one binary per segment) *plus a per-segment error
    band*: on segment ``[a, b]`` with chord ``s``, ``w - s(x)`` is confined to a
    rigorous enclosure of ``g(x) - s(x)``. The enclosure is the intersection of

    * a direct interval evaluation of ``g`` over the segment minus the chord's
      range, and
    * mean-value bounds from both endpoints, ``e(x) in e(a) + (g'([a,b]) - m)(x-a)``
      and ``e(x) in e(b) + (m - g'([a,b]))(b-x)``, with ``g'`` enclosed by
      interval automatic differentiation
      (:func:`discopt._relax.monotonicity.derivative_enclosure`). These bounds are
      *linear in x*, so they taper to the (rounding-sized) sampling error at the
      breakpoints and the band shrinks quadratically under refinement.

    Every point ``(x, g(x))`` satisfies the rows, so the transformed model is a
    relaxation of the original: its certified dual bound is a valid bound on the
    original. The transformed solution is then checked on the **original** model
    (and polished by a local NLP with the integers fixed if it is not already
    feasible) to produce an incumbent. The result reports
    ``status="optimal"``/``gap_certified=True`` only when that verified incumbent
    and the relaxation bound meet within tolerance; otherwise refinement bisects
    each term's widest segments (and adds the relaxed input value where ``w``
    strayed from ``g``), capped per term by ``max_breakpoints``, and repeats. On
    exhaustion the result is ``"feasible"`` with the honest bound, or a limit
    status with the bound and no incumbent. The partition-refinement idea is
    that of adaptive multivariate partitioning :cite:p:`Nagarajan2019` and of
    logarithmic partitioning schemes :cite:p:`Misener2011`, applied here to
    univariate terms.

``mode="approximate"`` -- the chord interpolant, uncertified.
    ``g`` is replaced by its interpolant through the breakpoint samples (via
    :meth:`Model.piecewise`). The result **never** carries a bound:
    ``bound=None``, ``gap_certified=False``, and ``algorithm_route`` states that
    the model solved was an approximation of the one declared. The approximate
    solution is verified on the original model (polished if needed); the status
    is ``"feasible"`` only if a verified point exists, ``"local_infeasible"``
    otherwise -- which, as everywhere in discopt, is *not* an infeasibility proof.

What is transformed
-------------------
A subexpression is replaced when it is scalar, nonlinear, depends on exactly one
scalar variable element (``x`` or ``v[i]``) whose declared bounds are finite, and
contains no :class:`Parameter` (a mutable value would be baked into the table).
The *maximal* such subexpression is taken, so ``exp(x) + x**2`` becomes one term.
Everything else -- multivariate terms such as ``x*y``, vector-valued nodes,
opaque calls -- is left exactly as written; the transformed model then still
contains those nonlinearities and discopt's global solver handles them, which
keeps the result sound. Skipped univariate terms are reported with the reason.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from discopt.modeling.core import Model, SolveResult

__all__ = [
    "PWL_MODES",
    "PWLSkippedTerm",
    "PWLTerm",
    "PWLTransformation",
    "nonlinear_to_pwl",
]

#: Accepted ``mode=`` values.
PWL_MODES: tuple[str, ...] = ("outer", "approximate")

_ABS_GAP_DEFAULT = 1e-6


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PWLTerm:
    """One univariate nonlinear term that was replaced.

    Attributes
    ----------
    index : int
        Position in traversal order (objective first, then constraints).
    location : str
        ``"objective"`` or the constraint name / ``"constraint[i]"``.
    expression : str
        The term as written (truncated).
    input : str
        The scalar input it depends on (``"x"`` or ``"v[3]"``).
    domain : tuple of float
        The input's declared bounds, i.e. the breakpoint span.
    breakpoints : tuple of float
        The breakpoints used in the last build.
    max_band : float or None
        ``mode="outer"`` only: the widest per-segment error band
        (``e_hi - e_lo``) in the last build -- how far ``w`` may sit from ``g(x)``
        anywhere in the relaxation. ``None`` in ``mode="approximate"``.
    """

    index: int
    location: str
    expression: str
    input: str
    domain: tuple[float, float]
    breakpoints: tuple[float, ...]
    max_band: Optional[float] = None


@dataclass(frozen=True)
class PWLSkippedTerm:
    """A univariate nonlinear term that was left exact, and why."""

    location: str
    expression: str
    reason: str


class PWLTransformError(ValueError):
    """The model cannot be transformed (and nothing was approximated)."""


# ---------------------------------------------------------------------------
# Expression analysis
# ---------------------------------------------------------------------------

_FLAG_PARAM = "param"
_FLAG_VECTOR = "vector"
_FLAG_OPAQUE = "opaque"


def _leaf_key(node) -> Optional[tuple]:
    """Key of a scalar variable leaf (``x`` or ``v[i]``), else ``None``."""
    from discopt.modeling.core import IndexExpression, Variable

    if isinstance(node, Variable):
        return ("v", id(node)) if node.shape == () else None
    if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
        idx = node.index
        if isinstance(idx, (int, np.integer)):
            idx_t: tuple = (int(idx),)
        elif isinstance(idx, tuple) and all(isinstance(i, (int, np.integer)) for i in idx):
            idx_t = tuple(int(i) for i in idx)
        else:
            return None
        if len(idx_t) != len(node.base.shape):
            return None
        shape = node.base.shape
        idx_t = tuple(i + s if i < 0 else i for i, s in zip(idx_t, shape))
        return ("i", id(node.base), idx_t)
    return None


def _children(node) -> Optional[list]:
    """Operand list of a node this module knows how to walk; ``None`` otherwise."""
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(node, (BinaryOp, MatMulExpression)):
        return [node.left, node.right]
    if isinstance(node, (UnaryOp, SumExpression)):
        return [node.operand]
    if isinstance(node, FunctionCall):
        return list(node.args)
    if isinstance(node, SumOverExpression):
        return list(node.terms)
    if isinstance(node, IndexExpression):
        return [node.base]
    return None


def _postorder(root, memo: dict):
    """Yield the not-yet-memoized nodes under *root*, children before parents.

    Iterative: expression DAGs from ``.nl`` files and from the sums this module
    emits are left-deep chains thousands of nodes long, far past Python's
    recursion limit.
    """
    from discopt.modeling.core import Variable

    stack = [(root, False)]
    seen: set[int] = set()
    while stack:
        node, expanded = stack.pop()
        key = id(node)
        if key in memo:
            continue
        if expanded:
            yield node
            continue
        if key in seen:
            continue
        seen.add(key)
        stack.append((node, True))
        is_leaf = isinstance(node, Variable) or _leaf_key(node) is not None
        kids = None if is_leaf else _children(node)
        for c in reversed(kids or []):
            if id(c) not in memo:
                stack.append((c, False))


class _Analyzer:
    """Memoized dependency / affinity analysis over an expression DAG."""

    def __init__(self):
        self._deps: dict[int, tuple[frozenset, frozenset]] = {}
        self._aff: dict[int, bool] = {}
        self.leaves: dict[tuple, Any] = {}

    def deps(self, root) -> tuple[frozenset, frozenset]:
        for node in _postorder(root, self._deps):
            self._deps[id(node)] = self._deps_one(node)
        return self._deps[id(root)]

    def _deps_one(self, node) -> tuple[frozenset, frozenset]:
        from discopt.modeling.core import Constant, IndexExpression, Parameter, Variable

        lk = _leaf_key(node)
        if lk is not None:
            self.leaves.setdefault(lk, node)
            return frozenset([lk]), frozenset()
        if isinstance(node, Constant):
            return frozenset(), frozenset()
        if isinstance(node, Parameter):
            return frozenset(), frozenset([_FLAG_PARAM])
        if isinstance(node, Variable):  # a non-scalar variable used whole
            return frozenset(), frozenset([_FLAG_VECTOR])
        kids = _children(node)
        if kids is None:
            return frozenset(), frozenset([_FLAG_OPAQUE])
        keys: set = set()
        flags: set = set()
        for c in kids:
            k, f = self._deps[id(c)]
            keys |= k
            flags |= f
        if isinstance(node, IndexExpression):
            flags.add(_FLAG_VECTOR)  # indexing into a computed (non-variable) array
        return frozenset(keys), frozenset(flags)

    def affine(self, root) -> bool:
        self.deps(root)
        for node in _postorder(root, self._aff):
            self._aff[id(node)] = self._affine_one(node)
        return self._aff[id(root)]

    def _const(self, node) -> bool:
        keys, flags = self._deps[id(node)]
        return not keys and not flags

    def _affine_one(self, node) -> bool:
        from discopt.modeling.core import (
            BinaryOp,
            MatMulExpression,
            Parameter,
            SumExpression,
            SumOverExpression,
            UnaryOp,
            Variable,
        )

        if self._const(node) or _leaf_key(node) is not None:
            return True
        if isinstance(node, (Variable, Parameter)):
            return True  # a whole (vector) variable, or a parameter: linear
        aff = self._aff
        if isinstance(node, UnaryOp):
            return node.op == "neg" and aff[id(node.operand)]
        if isinstance(node, BinaryOp):
            left, right = aff[id(node.left)], aff[id(node.right)]
            if node.op in ("+", "-"):
                return left and right
            if node.op == "*":
                return (self._const(node.left) or self._const(node.right)) and left and right
            if node.op == "/":
                return self._const(node.right) and left
            return False
        if isinstance(node, MatMulExpression):
            both = aff[id(node.left)] and aff[id(node.right)]
            return (self._const(node.left) or self._const(node.right)) and both
        if isinstance(node, SumExpression):
            return aff[id(node.operand)]
        if isinstance(node, SumOverExpression):
            return all(aff[id(t)] for t in node.terms)
        return False


def _rebuild(root, fn):
    """Rebuild *root*, replacing each node ``n`` by ``fn(n)`` when that is not None.

    ``fn`` is tried top-down; a replaced node is not descended into. Otherwise the
    children are rebuilt and the node is reconstructed only if one changed. Node
    types this module does not know are returned unchanged and never descended
    into. Iterative, for the same reason as :func:`_postorder`.
    """
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    memo: dict[int, Any] = {}
    stack = [(root, False)]
    while stack:
        n, expanded = stack.pop()
        key = id(n)
        if key in memo:
            continue
        if not expanded:
            out = fn(n)
            if out is not None:
                memo[key] = out
                continue
            kids = _children(n)
            if not kids:
                memo[key] = n
                continue
            stack.append((n, True))
            for c in reversed(kids):
                if id(c) not in memo:
                    stack.append((c, False))
            continue
        g = lambda c: memo[id(c)]  # noqa: E731
        if isinstance(n, BinaryOp):
            a, b = g(n.left), g(n.right)
            out = n if (a is n.left and b is n.right) else BinaryOp(n.op, a, b)
        elif isinstance(n, UnaryOp):
            a = g(n.operand)
            out = n if a is n.operand else UnaryOp(n.op, a)
        elif isinstance(n, FunctionCall):
            args = [g(a) for a in n.args]
            same = all(x is y for x, y in zip(args, n.args))
            out = n if same else FunctionCall(n.func_name, *args)
        elif isinstance(n, MatMulExpression):
            a, b = g(n.left), g(n.right)
            out = n if (a is n.left and b is n.right) else MatMulExpression(a, b)
        elif isinstance(n, SumExpression):
            a = g(n.operand)
            out = n if a is n.operand else SumExpression(a, n.axis)
        elif isinstance(n, SumOverExpression):
            terms = [g(t) for t in n.terms]
            same = all(x is y for x, y in zip(terms, n.terms))
            out = n if same else SumOverExpression(terms)
        elif isinstance(n, IndexExpression):
            a = g(n.base)
            out = n if a is n.base else IndexExpression(a, n.index)
        else:  # pragma: no cover - _children returned None for these
            out = n
        memo[key] = out
    return memo[id(root)]


# ---------------------------------------------------------------------------
# Univariate term geometry
# ---------------------------------------------------------------------------


class _Univariate:
    """``g`` as a function of one scalar, via a probe variable in a scratch model."""

    def __init__(self, term, leaf, lo: float, hi: float):
        from discopt.modeling.core import Model

        self._probe_model = Model("_pwl_probe")
        self.t = self._probe_model.continuous("t", lb=lo, ub=hi)
        key = _leaf_key(leaf)
        t = self.t
        self.g = _rebuild(term, lambda n: t if _leaf_key(n) == key else None)

    def enclose(self, a: float, b: float):
        from discopt._relax.convexity.interval import Interval
        from discopt._relax.convexity.interval_eval import evaluate_interval

        iv = evaluate_interval(self.g, None, box={self.t: Interval.from_bounds(a, b)})
        return float(np.min(iv.lo)), float(np.max(iv.hi))

    def point(self, a: float) -> tuple[float, float, float]:
        """``(value, err_lo, err_hi)``: a float sample and its rigorous error."""
        lo, hi = self.enclose(a, a)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return math.nan, math.nan, math.nan
        mid = 0.5 * (lo + hi)
        if abs(mid) < _COEF_FLOOR:
            # A rounding-sized sample would be dropped by the LP backend; use 0
            # and let the error enclosure absorb it (it is measured against 0).
            mid = 0.0
        return mid, _down(lo - mid), _up(hi - mid)

    def derivative(self, a: float, b: float) -> Optional[tuple[float, float]]:
        from discopt._relax.convexity.interval import Interval
        from discopt._relax.monotonicity import derivative_enclosure

        d = derivative_enclosure(self.g, self.t, {self.t: Interval.from_bounds(a, b)})
        if d is None:
            return None
        return float(np.min(d.lo)), float(np.max(d.hi))


def _down(x: float) -> float:
    return float(np.nextafter(x, -np.inf))


def _up(x: float) -> float:
    return float(np.nextafter(x, np.inf))


#: Smallest coefficient magnitude emitted in a band row. The rounding-sized
#: sample errors (~1e-16) are below what LP backends keep -- HiGHS drops
#: ``|a_ij| <= 1e-12``, which would *tighten* the row to zero. Moving such a
#: coefficient outward to this magnitude only loosens the relaxation.
_COEF_FLOOR = 1e-11


def _loosen_up(x: float) -> float:
    """Round an upper-bound coefficient up, away from the drop zone near zero."""
    x = _up(x)
    return max(x, _COEF_FLOOR) if -_COEF_FLOOR < x < _COEF_FLOOR else x


def _loosen_down(x: float) -> float:
    """Round a lower-bound coefficient down, away from the drop zone near zero."""
    x = _down(x)
    return min(x, -_COEF_FLOOR) if -_COEF_FLOOR < x < _COEF_FLOOR else x


@dataclass
class _SegmentBand:
    """Rigorous bounds on ``e(x) = g(x) - s(x)`` over one segment (chord ``s``).

    ``lo``/``hi`` hold on the whole segment. The tapers, when finite, give the
    linear bounds ``e <= ua + ka*(x-a)``, ``e <= ub + kb*(b-x)``,
    ``e >= la + ja*(x-a)``, ``e >= lb + jb*(b-x)``.
    """

    lo: float
    hi: float
    taper: Optional[tuple[float, float, float, float, float, float, float, float]] = None


def _segment_band(
    u: _Univariate, a: float, b: float, ga: tuple, gb: tuple
) -> Optional[_SegmentBand]:
    """Band for ``g - chord`` on ``[a, b]``; ``None`` if no finite band exists."""
    from discopt._relax.convexity.interval import Interval

    va, la, ua = ga
    vb, lb_, ub_ = gb
    h = b - a
    m = (vb - va) / h
    # (A) direct: e in G([a,b]) - s([a,b]); s is linear so its range is its endpoints.
    g_lo, g_hi = u.enclose(a, b)
    e_lo, e_hi = -math.inf, math.inf
    if math.isfinite(g_lo) and math.isfinite(g_hi):
        s_iv = Interval.from_bounds(min(va, vb), max(va, vb))
        e_iv = Interval.from_bounds(g_lo, g_hi) - s_iv
        e_lo, e_hi = float(e_iv.lo), float(e_iv.hi)
    taper = None
    d = u.derivative(a, b)
    if d is not None:
        dl, du = d
        # e' = g' - m in [dl - m, du - m]; the chord slope m is a float we chose,
        # so only the arithmetic below needs outward rounding.
        slope = Interval.from_bounds(dl, du) - Interval.point(m)
        hh = Interval.point(h)
        span = slope * hh  # (g' - m) * h
        ka, ja = float(span.hi), float(span.lo)  # from a: e in e(a) + [ja, ka]*(x-a)/h
        kb, jb = -float(span.lo), -float(span.hi)  # from b: e in e(b) + [jb, kb]*(b-x)/h
        # Constant consequences of the tapers over the segment.
        e_hi = min(e_hi, _up(ua + max(0.0, ka)), _up(ub_ + max(0.0, kb)))
        e_lo = max(e_lo, _down(la + min(0.0, ja)), _down(lb_ + min(0.0, jb)))
        taper = (
            _loosen_up(ua),
            _loosen_up(ka),
            _loosen_up(ub_),
            _loosen_up(kb),
            _loosen_down(la),
            _loosen_down(ja),
            _loosen_down(lb_),
            _loosen_down(jb),
        )
    if not (math.isfinite(e_lo) and math.isfinite(e_hi)):
        return None
    # e(a) and e(b) lie in the band by construction; an empty band is a bug in the
    # enclosures, never a numerical nuisance to be clamped away.
    if e_lo > e_hi:
        raise AssertionError(
            f"nonlinear_to_pwl: empty error band [{e_lo}, {e_hi}] on segment [{a}, {b}]"
        )
    return _SegmentBand(e_lo, e_hi, taper)


def _check_band(u: _Univariate, a: float, b: float, va: float, vb: float, bd: _SegmentBand):
    """Self-check a band against rigorous point enclosures inside the segment.

    Not a proof -- :func:`_segment_band` is the proof -- but defence in depth: an
    implementation error that produced a band excluding the graph would make the
    relaxation cut off feasible points, and the resulting false bound or false
    *infeasibility* certificate has no incumbent to be caught against. Raises
    ``AssertionError`` naming the segment; never clamps.
    """
    h = b - a
    for frac in (0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0):
        t = a + frac * h
        g_lo, g_hi = u.enclose(t, t)
        chord = va + (vb - va) * frac
        e_lo, e_hi = g_lo - chord, g_hi - chord
        slack = 1e-9 * max(1.0, abs(g_lo), abs(g_hi))
        bad = e_hi < bd.lo - slack or e_lo > bd.hi + slack
        if bd.taper is not None and not bad:
            ua, ka, ub_, kb, la, ja, lb_, jb = bd.taper
            bad = (
                e_lo > ua + ka * frac + slack
                or e_lo > ub_ + kb * (1 - frac) + slack
                or e_hi < la + ja * frac - slack
                or e_hi < lb_ + jb * (1 - frac) - slack
            )
        if bad:
            raise AssertionError(
                f"nonlinear_to_pwl: the error band on segment [{a}, {b}] excludes the "
                f"term's value at x={t} (g - chord in [{e_lo}, {e_hi}], band "
                f"[{bd.lo}, {bd.hi}]); the outer approximation would not be a relaxation"
            )


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def _lower_outer(model, leaf, w, bps, samples, bands, prefix) -> None:
    """Disaggregated convex combination plus the per-segment error band.

    With ``z_i = 1`` the segment weights satisfy ``x - a = h*wr_i`` and
    ``b - x = h*wl_i``, so the tapers are linear rows in ``(wl_i, wr_i, z_i)``;
    with ``z_i = 0`` every term vanishes and ``d_i = 0``.
    """
    from discopt.modeling._piecewise import _lin, _value_reference

    add = model.subject_to
    s = len(bps) - 1
    wl = model.continuous(f"{prefix}_wl", shape=(s,), lb=0.0, ub=1.0)
    wr = model.continuous(f"{prefix}_wr", shape=(s,), lb=0.0, ub=1.0)
    z = model.binary(f"{prefix}_z", shape=(s,))
    d_lb = np.array([min(-_COEF_FLOOR, _loosen_down(bd.lo)) for bd in bands])
    d_ub = np.array([max(_COEF_FLOOR, _loosen_up(bd.hi)) for bd in bands])
    d = model.continuous(f"{prefix}_d", shape=(s,), lb=d_lb, ub=d_ub)
    zs = [z[i] for i in range(s)]
    for i in range(s):
        add(wl[i] + wr[i] == zs[i], name=f"{prefix}_seg{i}")
    add(_lin(np.ones(s), zs) == 1.0, name=f"{prefix}_choose")
    b = np.asarray(bps)
    v = np.asarray(samples)
    wterms = [wl[i] for i in range(s)] + [wr[i] for i in range(s)]
    # Centred on the first breakpoint and on a central sample (#1495, the #1494
    # class): with sum(wl + wr) = sum(z) = 1 these equal the uncentred
    # ``leaf == sum b w`` rows exactly, but a solver meets that sum only to a
    # tolerance eps, and uncentred the leaf may then drift by |b_0| * eps -- at
    # b_0 = 1e5 enough to certify 2.5e-5 on a problem whose minimum is 0. Centred,
    # the drift is span * eps. ``_value_reference`` keeps a sample equal to the
    # reference up to rounding from becoming a droppable ~1e-16 coefficient.
    cv = _value_reference(v)
    bx = np.concatenate([b[:-1], b[1:]]) - b[0]
    vw = np.concatenate([v[:-1], v[1:]]) - cv
    add(leaf == _lin(bx, wterms, b[0]), name=f"{prefix}_x")
    dterms = [d[i] for i in range(s)]
    add(
        w == _lin(np.concatenate([vw, np.ones(s)]), wterms + dterms, cv),
        name=f"{prefix}_w",
    )
    for i, bd in enumerate(bands):
        add(d[i] <= _loosen_up(bd.hi) * zs[i], name=f"{prefix}_band{i}_hi")
        add(d[i] >= _loosen_down(bd.lo) * zs[i], name=f"{prefix}_band{i}_lo")
        if bd.taper is not None:
            ua, ka, ub_, kb, la, ja, lb_, jb = bd.taper
            add(d[i] <= ua * zs[i] + ka * wr[i], name=f"{prefix}_taper{i}_a_hi")
            add(d[i] <= ub_ * zs[i] + kb * wl[i], name=f"{prefix}_taper{i}_b_hi")
            add(d[i] >= la * zs[i] + ja * wr[i], name=f"{prefix}_taper{i}_a_lo")
            add(d[i] >= lb_ * zs[i] + jb * wl[i], name=f"{prefix}_taper{i}_b_lo")


# ---------------------------------------------------------------------------
# The transformation
# ---------------------------------------------------------------------------


@dataclass
class _Build:
    model: "Model"
    terms: list[PWLTerm]
    skipped: list[PWLSkippedTerm]
    # (leaf, w variable name, g, per-segment band widths or None)
    handles: list[tuple[Any, str, "_Univariate", Optional[list[float]]]]
    fully_linear: bool


def _leaf_label(leaf: Any) -> str:
    from discopt.modeling.core import IndexExpression

    if isinstance(leaf, IndexExpression):
        idx = leaf.index if isinstance(leaf.index, tuple) else (leaf.index,)
        base: Any = leaf.base
        return f"{base.name}[{', '.join(str(int(i)) for i in idx)}]"
    return str(leaf.name)


def _leaf_value(leaf: Any, x: dict) -> float:
    from discopt.modeling.core import IndexExpression

    if isinstance(leaf, IndexExpression):
        base: Any = leaf.base
        return float(np.asarray(x[base.name])[leaf.index])
    return float(np.asarray(x[leaf.name]))


def _leaf_bounds(leaf: Any) -> tuple[float, float, bool]:
    from discopt.modeling.core import IndexExpression, VarType

    v: Any
    if isinstance(leaf, IndexExpression):
        v = leaf.base
        lo, hi = float(np.asarray(v.lb)[leaf.index]), float(np.asarray(v.ub)[leaf.index])
    else:
        v = leaf
        lo, hi = float(np.asarray(v.lb)), float(np.asarray(v.ub))
    return lo, hi, v.var_type in (VarType.INTEGER, VarType.BINARY)


_UNBOUNDED = 1e15  # discopt's "infinite" default bounds are +-9.999e19

#: Largest input or term magnitude the transformation will tabulate. Beyond it
#: the emitted coefficients leave the range LP backends accept (HiGHS refuses
#: |a_ij| >= 1e15) long before they stop being meaningful; the term stays exact.
_MAGNITUDE_LIMIT = 1e9


def _default_breakpoints(lo: float, hi: float, is_int: bool, segments: int) -> np.ndarray:
    if is_int and lo == math.floor(lo) and hi == math.floor(hi):
        if hi - lo <= segments:
            # Every integer is a breakpoint: exact at every integer-feasible input.
            return np.arange(lo, hi + 1.0)
        # Otherwise an integer-valued grid: only integers are feasible inputs.
        return np.unique(np.round(np.linspace(lo, hi, segments + 1)))
    bps = np.linspace(lo, hi, segments + 1)
    # linspace can land a hair off zero (e.g. 1e-17), a coefficient an LP backend
    # drops; interior breakpoints are free to choose, so put it on zero exactly.
    tiny = np.abs(bps) < 1e-12 * max(1.0, hi - lo)
    tiny[0] = tiny[-1] = False
    bps[tiny] = 0.0
    return bps


def _refuse_non_algebraic(model: "Model") -> None:
    from discopt.modeling.core import Constraint

    for c in model._constraints:
        if not isinstance(c, Constraint):
            raise PWLTransformError(
                "nonlinear_to_pwl supports models whose relations are all algebraic "
                f"constraints; found {type(c).__name__}. Indicator, disjunctive, SOS "
                "and logical relations cannot be verified at a point by the incumbent "
                "check this transformation relies on. (A Model.piecewise function "
                "built with method='sos2' is one such relation -- use another method.)"
            )
    if getattr(model, "_complementarities", None):
        raise PWLTransformError(
            "nonlinear_to_pwl does not support models with complementarity relations."
        )


class PWLTransformation:
    """A model with its univariate nonlinear terms replaced by PWL MILP constructs.

    Build with :func:`nonlinear_to_pwl`. The original model is never mutated:
    every build works on a serialized copy.

    Attributes
    ----------
    original : Model
        The model as declared.
    mode : str
        ``"outer"`` or ``"approximate"``.
    model : Model
        The transformed model from the latest build.
    terms : list of PWLTerm
        The terms replaced.
    skipped : list of PWLSkippedTerm
        Univariate nonlinear terms left exact, with the reason.
    fully_linear : bool
        True when no nonlinear term remains in the transformed model (it is a
        pure MILP/LP).
    """

    def __init__(
        self, model: "Model", mode: str, segments: int, method: str, max_breakpoints: int = 256
    ):
        from discopt.modeling._piecewise import normalize_piecewise_method

        if mode not in PWL_MODES:
            raise ValueError(f"nonlinear_to_pwl mode must be one of {PWL_MODES}, got {mode!r}.")
        if isinstance(segments, bool) or not isinstance(segments, (int, np.integer)):
            raise TypeError(f"segments must be an int, got {type(segments).__name__}.")
        if segments < 1:
            raise ValueError(f"segments must be at least 1, got {segments}.")
        if model._objective is None:
            raise ValueError("No objective set. Call m.minimize() or m.maximize().")
        from discopt.modeling.core import ObjectiveSense

        self._maximize = model._objective.sense == ObjectiveSense.MAXIMIZE
        _refuse_non_algebraic(model)
        self.original = model
        self.mode = mode
        self.segments = int(segments)
        if max_breakpoints < segments + 1:
            raise ValueError(
                f"max_breakpoints ({max_breakpoints}) must be at least segments + 1 "
                f"({segments + 1})."
            )
        self.max_breakpoints = int(max_breakpoints)
        self.method = normalize_piecewise_method(method)
        if mode == "approximate" and self.method == "sos2":
            raise ValueError(
                "method='sos2' produces an SOS relation the incumbent check cannot "
                "verify; use 'incremental', 'log' or 'disaggregated'."
            )
        self._breakpoints: Optional[list[np.ndarray]] = None
        self._build = self._make()

    # -- public views ------------------------------------------------------

    @property
    def model(self) -> "Model":
        return self._build.model

    @property
    def terms(self) -> list[PWLTerm]:
        return list(self._build.terms)

    @property
    def skipped(self) -> list[PWLSkippedTerm]:
        return list(self._build.skipped)

    @property
    def fully_linear(self) -> bool:
        return self._build.fully_linear

    def __repr__(self) -> str:
        return (
            f"PWLTransformation(mode={self.mode!r}, terms={len(self._build.terms)}, "
            f"skipped={len(self._build.skipped)}, fully_linear={self.fully_linear})"
        )

    # -- building ----------------------------------------------------------

    def _make(self) -> _Build:
        from discopt.modeling._piecewise import PiecewiseLinear, build_piecewise
        from discopt.modeling.core import Constraint, Objective
        from discopt.serialize import dumps, loads

        work = loads(dumps(self.original))
        names_o = [v.name for v in self.original._variables]
        if [v.name for v in work._variables] != names_o:  # pragma: no cover - invariant
            raise AssertionError("nonlinear_to_pwl: copy reordered the variables")

        an = _Analyzer()
        terms: list[PWLTerm] = []
        skipped: list[PWLSkippedTerm] = []
        handles: list[tuple[Any, str, _Univariate, Optional[list[float]]]] = []
        replaced: dict[int, Any] = {}
        first_build = self._breakpoints is None
        bps_all: list[np.ndarray] = [] if first_build else self._breakpoints  # type: ignore[assignment]
        location = ["objective"]

        from discopt.modeling.core import _known_shape

        def candidate(n):
            if id(n) in replaced:
                return replaced[id(n)]
            keys, flags = an.deps(n)
            if len(keys) != 1 or _known_shape(n) != () or an.affine(n):
                return None
            if flags:
                if _FLAG_PARAM in flags and not (flags - {_FLAG_PARAM}):
                    skipped.append(
                        PWLSkippedTerm(
                            location[0],
                            _short(n),
                            "depends on a Parameter; approximating it would bake in the "
                            "parameter's current value",
                        )
                    )
                    replaced[id(n)] = n
                    return n
                return None
            (key,) = keys
            leaf = an.leaves[key]
            lo, hi, is_int = _leaf_bounds(leaf)
            if not (abs(lo) < _UNBOUNDED and abs(hi) < _UNBOUNDED):
                skipped.append(
                    PWLSkippedTerm(
                        location[0], _short(n), f"input {_leaf_label(leaf)} is unbounded"
                    )
                )
                replaced[id(n)] = n
                return n
            if not hi > lo:
                skipped.append(
                    PWLSkippedTerm(location[0], _short(n), f"input {_leaf_label(leaf)} is fixed")
                )
                replaced[id(n)] = n
                return n
            k = len(terms)
            if first_build:
                bps = _default_breakpoints(lo, hi, is_int, self.segments)
            else:
                if k >= len(bps_all):  # pragma: no cover - invariant
                    raise AssertionError("nonlinear_to_pwl: term count changed between builds")
                bps = bps_all[k]
            u = _Univariate(n, leaf, lo, hi)
            # Decided from the whole domain, not the grid, so the verdict is the
            # same on every (refined) build and the term list cannot shift.
            g_lo, g_hi = u.enclose(lo, hi)
            if max(abs(lo), abs(hi)) > _MAGNITUDE_LIMIT or not (
                abs(g_lo) <= _MAGNITUDE_LIMIT and abs(g_hi) <= _MAGNITUDE_LIMIT
            ):
                skipped.append(
                    PWLSkippedTerm(
                        location[0],
                        _short(n),
                        f"input {_leaf_label(leaf)} in [{lo:g}, {hi:g}] gives a term range "
                        f"enclosed only by [{g_lo:g}, {g_hi:g}]; beyond +-{_MAGNITUDE_LIMIT:g} "
                        "the table's coefficients are not numerically meaningful -- tighten "
                        "the input's bounds",
                    )
                )
                replaced[id(n)] = n
                return n
            pts = [u.point(float(t)) for t in bps]
            if not all(math.isfinite(p[0]) for p in pts):
                skipped.append(
                    PWLSkippedTerm(
                        location[0],
                        _short(n),
                        f"not finite at every breakpoint of {_leaf_label(leaf)} in "
                        f"[{lo:g}, {hi:g}]",
                    )
                )
                replaced[id(n)] = n
                return n
            samples = [p[0] for p in pts]
            prefix = f"_n2p{k}"
            max_band: Optional[float] = None
            if self.mode == "outer":
                bands = []
                for i in range(len(bps) - 1):
                    a_i, b_i = float(bps[i]), float(bps[i + 1])
                    bd = _segment_band(u, a_i, b_i, pts[i], pts[i + 1])
                    if bd is None:
                        break
                    _check_band(u, a_i, b_i, pts[i][0], pts[i + 1][0], bd)
                    bands.append(bd)
                if len(bands) != len(bps) - 1:
                    skipped.append(
                        PWLSkippedTerm(
                            location[0],
                            _short(n),
                            "no finite error enclosure on some segment (interval "
                            "arithmetic could not bound the term there)",
                        )
                    )
                    replaced[id(n)] = n
                    return n
                w_lo = min(min(samples) + min(b.lo for b in bands), min(samples))
                w_hi = max(max(samples) + max(b.hi for b in bands), max(samples))
                w_lo = max(w_lo, g_lo)
                w_hi = min(w_hi, g_hi)
                w = work.continuous(f"{prefix}_w", lb=_down(w_lo), ub=_up(w_hi))
                _lower_outer(work, leaf, w, bps, samples, bands, prefix)
                max_band = max(b.hi - b.lo for b in bands)
            else:
                table = PiecewiseLinear.from_table(bps, samples)
                w = build_piecewise(work, leaf, table, self.method, f"{prefix}_w")
            terms.append(
                PWLTerm(
                    index=k,
                    location=location[0],
                    expression=_short(n),
                    input=_leaf_label(leaf),
                    domain=(lo, hi),
                    breakpoints=tuple(float(t) for t in bps),
                    max_band=max_band,
                )
            )
            if first_build:
                bps_all.append(np.asarray(bps, dtype=np.float64))
            widths = [b.hi - b.lo for b in bands] if self.mode == "outer" else None
            handles.append((leaf, w.name, u, widths))
            replaced[id(n)] = w
            return w

        obj = work._objective
        assert obj is not None  # checked at construction; loads() preserves it
        placeholder = getattr(obj, "_is_placeholder", False)
        originals = list(work._constraints)
        work._constraints = []
        new_obj = obj
        if not placeholder:
            new_obj = Objective(_rebuild(obj.expression, candidate), obj.sense)
        rewritten = []
        for i, c in enumerate(originals):
            location[0] = c.name or f"constraint[{i}]"
            body = _rebuild(c.body, candidate)
            rewritten.append(c if body is c.body else Constraint(body, c.sense, c.rhs, c.name))
        aux = work._constraints
        work._constraints = rewritten + aux
        work._objective = new_obj
        if first_build:
            self._breakpoints = bps_all

        # The emitted rows are linear by construction; only the rewritten model
        # rows can still carry a nonlinearity.
        an2 = _Analyzer()
        fully_linear = (placeholder or an2.affine(new_obj.expression)) and all(
            an2.affine(c.body) for c in rewritten
        )
        return _Build(work, terms, skipped, handles, fully_linear)

    def _refine(self, x: dict, feas_tol: float) -> int:
        """Refine the partition while the gap is open; return breakpoints added.

        Every term gets its widest tenth of segments bisected (at least one): the
        relaxation's looseness can sit anywhere, not only at the relaxed point --
        measured on nvs03, whose relaxed incumbent lay exactly on the graph while
        the surrogate's own bound was still half the optimum, so refining only
        where ``w`` strayed from ``g`` stalled. A term whose ``w`` does stray at the
        relaxed solution also gets that input value and its segment's midpoint.
        Capping the split at a tenth matters as much: bisecting every wide segment
        doubled the table each round on the valve-point dispatch (9 -> 849
        breakpoints in 8 rounds, then an out-of-memory kill). Returns 0 only when
        the per-term cap leaves no room.
        """
        assert self._breakpoints is not None
        added = 0
        for k, (leaf, w_name, u, widths) in enumerate(self._build.handles):
            bps = self._breakpoints[k]
            lo, hi = float(bps[0]), float(bps[-1])
            min_sep = 1e-9 * max(1.0, hi - lo)
            t = _leaf_value(leaf, x)
            w_val = float(np.asarray(x[w_name]))
            g_val, _, _ = u.point(t)
            stray = math.isfinite(g_val) and abs(w_val - g_val) > feas_tol * max(1.0, abs(g_val))
            new: list[float] = []
            if stray:
                j = int(np.searchsorted(bps, t, side="right")) - 1
                j = min(max(j, 0), len(bps) - 2)
                new += [t, 0.5 * (bps[j] + bps[j + 1])]
            if widths:
                n_split = max(1, math.ceil(0.1 * len(widths)))
                worst = np.argsort(widths)[::-1][:n_split]
                new.extend(0.5 * (bps[i] + bps[i + 1]) for i in worst)
            if _leaf_bounds(leaf)[2]:
                # Integer input: only integer points are feasible, and at an integer
                # breakpoint the tapered band collapses to rounding size, so place
                # every new breakpoint on an integer (both neighbours of a midpoint).
                new = [float(v) for p in new for v in (math.floor(p), math.ceil(p))]
                new = [p for p in new if lo < p < hi]
            new = [p for p in new if np.min(np.abs(bps - p)) > min_sep]
            room = self.max_breakpoints - len(bps)
            new = sorted(set(new), key=lambda p: abs(p - t))[: max(0, room)]
            if new:
                self._breakpoints[k] = np.unique(np.concatenate([bps, new]))
                added += len(self._breakpoints[k]) - len(bps)
        if added:
            self._build = self._make()
        return added

    # -- solving -----------------------------------------------------------

    def solve(
        self,
        *,
        time_limit: float = 3600.0,
        gap_tolerance: float = 1e-4,
        abs_gap_tolerance: Optional[float] = None,
        max_rounds: int = 20,
        polish: bool = True,
        **solve_kwargs,
    ) -> "SolveResult":
        """Solve the transformed model and report a result **about the original**.

        Parameters
        ----------
        time_limit : float
            Total wall-clock budget across all rounds.
        gap_tolerance, abs_gap_tolerance : float
            Certification tolerances, as in :meth:`Model.solve`. The absolute
            default is ``1e-6``.
        max_rounds : int
            ``mode="outer"`` only: maximum number of solve/refine rounds.
        polish : bool
            If the transformed solution is not feasible for the original model,
            run a local NLP on the original from it, integers fixed, and verify
            the result. The polished point must still pass the independent check.
        **solve_kwargs
            Forwarded to each :meth:`Model.solve` of the transformed model.

        Returns
        -------
        SolveResult
            ``x`` holds the original model's variables only. See the module
            docstring for the status contract of each mode.
        """
        from discopt.modeling.core import SolveResult

        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        abs_tol = _ABS_GAP_DEFAULT if abs_gap_tolerance is None else float(abs_gap_tolerance)
        t0 = time.perf_counter()
        maximize = self._maximize
        best_obj: Optional[float] = None
        best_x: Optional[np.ndarray] = None
        best_bound: Optional[float] = None
        bound_source: Optional[str] = None
        nodes = 0
        rounds = 0
        history: list[dict] = []
        last_status = None
        polish_notes: list[str] = []
        feas_tol = 1e-6

        def better(a: float, b: Optional[float]) -> bool:
            return b is None or (a > b if maximize else a < b)

        rounds_allowed = max_rounds if self.mode == "outer" else 1
        certified = False
        timed_out = False
        while rounds < rounds_allowed:
            remaining = time_limit - (time.perf_counter() - t0)
            if remaining <= 0:
                timed_out = True
                break
            rounds += 1
            r = self.model.solve(
                time_limit=remaining,
                gap_tolerance=gap_tolerance,
                abs_gap_tolerance=abs_gap_tolerance,
                **solve_kwargs,
            )
            if not isinstance(r, SolveResult):
                raise TypeError(
                    "nonlinear_to_pwl.solve needs a SolveResult from each round; a "
                    "streaming solve (stream=True) is not supported here."
                )
            nodes += int(r.node_count or 0)
            last_status = r.status
            entry: dict[str, Any] = {
                "round": rounds,
                "status": r.status,
                "surrogate_objective": r.objective,
                "surrogate_bound": r.bound,
                "breakpoints": [len(t.breakpoints) for t in self._build.terms],
            }
            history.append(entry)
            if self.mode == "outer" and r.status == "infeasible" and r.gap_certified:
                if best_x is not None:
                    # The same tripwire as below, for the extreme case: an earlier
                    # round produced a point VERIFIED on the original model, so a
                    # relaxation claiming emptiness is not a relaxation (or its
                    # solver's infeasible label is false). Publishing "infeasible"
                    # next to a known feasible point would be a false certificate.
                    raise AssertionError(
                        "nonlinear_to_pwl: the outer approximation was reported "
                        f"infeasible in round {rounds} after round(s) before it produced "
                        f"a point verified on the original model (objective {best_obj}); "
                        "the outer approximation is not a relaxation"
                    )
                # The relaxation is infeasible, so the original is: a certificate.
                return self._result(
                    status="infeasible",
                    gap_certified=True,
                    t0=t0,
                    nodes=nodes,
                    rounds=rounds,
                    history=history,
                    polish_notes=polish_notes,
                )
            if self.mode == "outer" and r.bound is not None and (r.bound_valid or r.gap_certified):
                rb = float(r.bound)
                if best_bound is None or (rb < best_bound if maximize else rb > best_bound):
                    best_bound = rb
                    bound_source = r.bound_source or "bnb_tree"
            if r.x is not None:
                cand = self._verified_candidate(r.x, polish, polish_notes)
                if cand is not None and better(cand[1], best_obj):
                    best_x, best_obj = cand
                entry["verified_objective"] = None if cand is None else cand[1]
            if self.mode != "outer":
                break
            if best_obj is not None and best_bound is not None:
                # Soundness tripwire: a valid bound can never pass a verified point.
                slack = abs_tol + 1e-7 * max(1.0, abs(best_obj))
                if (
                    (best_bound > best_obj + slack)
                    if not maximize
                    else (best_bound < best_obj - slack)
                ):
                    raise AssertionError(
                        f"nonlinear_to_pwl: relaxation bound {best_bound} passes the verified "
                        f"objective {best_obj}; the outer approximation is not a relaxation"
                    )
                if _gap_closed(best_obj, best_bound, gap_tolerance, abs_tol):
                    certified = True
                    break
            if r.x is None or rounds >= rounds_allowed:
                break
            if self._refine(r.x, feas_tol) == 0:
                entry["refined"] = 0
                break

        return self._result(
            status=None,
            certified=certified,
            best_obj=best_obj,
            best_x=best_x,
            best_bound=best_bound,
            bound_source=bound_source,
            last_status=last_status,
            timed_out=timed_out,
            t0=t0,
            nodes=nodes,
            rounds=rounds,
            history=history,
            polish_notes=polish_notes,
        )

    def _verified_candidate(
        self, x: dict, polish: bool, notes: list[str]
    ) -> Optional[tuple[np.ndarray, float]]:
        """Map a transformed solution to the original; return its best verified form.

        Candidates are the mapped point itself and, with *polish*, a local NLP
        solution of the original from it (integers fixed). Each must pass
        :func:`~discopt.validation.feasibility.verify_point` on the ORIGINAL model
        -- the independent check is the only thing that makes a point an
        incumbent. The better verified candidate wins.
        """
        from discopt.validation.feasibility import verify_point

        orig = self.original
        maximize = self._maximize
        flat = np.concatenate(
            [np.asarray(x[v.name], dtype=np.float64).reshape(-1) for v in orig._variables]
        )
        found: list[tuple[np.ndarray, float]] = []
        res = verify_point(orig, flat, with_objective=True)
        if res.ok:
            assert res.objective is not None  # with_objective=True on an ok result
            found.append((flat, float(res.objective)))
        else:
            notes.append(f"transformed point not feasible for the original: {res.reason}")
        if polish:
            polished = _polish(orig, flat, notes)
            if polished is not None:
                res2 = verify_point(orig, polished, with_objective=True)
                if res2.ok:
                    assert res2.objective is not None
                    found.append((polished, float(res2.objective)))
                else:
                    notes.append(f"polished point failed verification: {res2.reason}")
        if not found:
            return None
        return (max if maximize else min)(found, key=lambda c: c[1])

    def _result(self, *, status, t0, nodes, rounds, history, polish_notes, **kw) -> "SolveResult":
        from discopt.modeling.core import SolveResult

        orig = self.original
        stats = {
            "nonlinear_to_pwl": {
                "mode": self.mode,
                "rounds": rounds,
                "terms": len(self._build.terms),
                "skipped": len(self._build.skipped),
                "fully_linear": self.fully_linear,
                "history": history,
                "polish_notes": polish_notes,
            }
        }
        wall = time.perf_counter() - t0
        if status == "infeasible":
            return SolveResult(
                status="infeasible",
                wall_time=wall,
                node_count=nodes,
                gap_certified=True,
                solver_stats=stats,
                algorithm_route="nonlinear_to_pwl(outer): the outer approximation is "
                "infeasible, which proves the original model infeasible",
                _model=orig,
            )
        best_obj = kw["best_obj"]
        best_x = kw["best_x"]
        x_dict = None
        if best_x is not None:
            x_dict = {}
            off = 0
            for v in orig._variables:
                x_dict[v.name] = best_x[off : off + v.size].reshape(v.shape)
                off += v.size
        if self.mode == "approximate":
            return SolveResult(
                status=(
                    "feasible"
                    if best_obj is not None
                    else ("error" if kw["last_status"] == "error" else "local_infeasible")
                ),
                objective=best_obj,
                x=x_dict,
                wall_time=wall,
                node_count=nodes,
                gap_certified=False,
                solver_stats=stats,
                algorithm_route=(
                    "nonlinear_to_pwl(approximate): the model solved was a piecewise-linear "
                    "APPROXIMATION of the one declared; no bound on the declared model is "
                    "claimed" + ("" if best_obj is not None else " and no verified point was found")
                ),
                _model=orig,
            )
        best_bound = kw["best_bound"]
        certified = kw["certified"]
        gap = None
        if best_obj is not None and best_bound is not None:
            gap = abs(best_obj - best_bound) / max(abs(best_obj), abs(best_bound), 1e-10)
        if certified:
            status_s = "optimal"
        elif best_obj is not None:
            status_s = "feasible"
        elif kw["last_status"] == "error" and best_bound is None:
            status_s = "error"
        elif kw["last_status"] == "time_limit" or kw["timed_out"]:
            status_s = "time_limit"
        else:
            # Rounds exhausted (or refinement stalled) without a verified point.
            status_s = "iteration_limit"
        route = (
            "nonlinear_to_pwl(outer): bound from a rigorous piecewise-linear outer "
            "approximation of the univariate nonlinear terms; incumbent verified on the "
            "declared model"
        )
        return SolveResult(
            status=status_s,
            objective=best_obj,
            bound=best_bound,
            gap=gap,
            x=x_dict,
            wall_time=wall,
            node_count=nodes,
            gap_certified=bool(certified),
            bound_valid=best_bound is not None,
            bound_source=kw["bound_source"] if best_bound is not None else None,
            solver_stats=stats,
            algorithm_route=route,
            _model=orig,
        )


def _gap_closed(obj: float, bound: float, rel: float, abs_tol: float) -> bool:
    diff = abs(obj - bound)
    return diff <= abs_tol or diff / max(abs(obj), abs(bound), 1e-10) <= rel


def _polish(model: "Model", x0: np.ndarray, notes: list[str]) -> Optional[np.ndarray]:
    """Local NLP on a copy of *model* from *x0* with integer columns fixed."""
    from discopt.modeling.core import VarType
    from discopt.serialize import dumps, loads
    from discopt.solvers.nlp_pounce import solve_nlp_from_model

    work = loads(dumps(model))
    off = 0
    for v in work._variables:
        sl = x0[off : off + v.size].reshape(v.shape)
        if v.var_type in (VarType.INTEGER, VarType.BINARY):
            fixed = np.clip(np.round(sl), v.lb, v.ub)
            v.lb = np.asarray(fixed, dtype=np.float64)
            v.ub = np.asarray(fixed, dtype=np.float64).copy()
        off += v.size
    start = np.clip(x0, *_flat_bounds(work))
    try:
        res = solve_nlp_from_model(work, x0=start, options={"max_iter": 500})
    except Exception as exc:  # noqa: BLE001 - recorded, and the point is simply not used
        notes.append(f"polish NLP raised {type(exc).__name__}: {exc}")
        return None
    x = getattr(res, "x", None)
    if x is None:
        notes.append("polish NLP returned no point")
        return None
    return np.asarray(x, dtype=np.float64)


def _flat_bounds(model: "Model") -> tuple[np.ndarray, np.ndarray]:
    lbs = [np.asarray(v.lb, dtype=np.float64).reshape(-1) for v in model._variables]
    ubs = [np.asarray(v.ub, dtype=np.float64).reshape(-1) for v in model._variables]
    return np.concatenate(lbs), np.concatenate(ubs)


def _short(node, n: int = 80) -> str:
    s = repr(node)
    return s if len(s) <= n else s[: n - 3] + "..."


def nonlinear_to_pwl(
    model: "Model",
    *,
    mode: str = "outer",
    segments: int = 8,
    method: str = "incremental",
    max_breakpoints: int = 256,
) -> PWLTransformation:
    """Replace a model's univariate nonlinear terms by piecewise-linear MILP constructs.

    Parameters
    ----------
    model : Model
        The model to transform. It is not modified.
    mode : {"outer", "approximate"}, default ``"outer"``
        ``"outer"`` builds a rigorous outer approximation (a relaxation): its
        bound is valid for *model* and :meth:`PWLTransformation.solve` certifies
        only when a verified incumbent meets it. ``"approximate"`` uses the chord
        interpolant and never claims a bound.
    segments : int, default 8
        Initial number of equal segments per term over its input's bounds. An
        integer input whose range spans at most ``segments`` integers gets a
        breakpoint at every integer instead.
    method : str, default ``"incremental"``
        ``mode="approximate"`` only: the :meth:`Model.piecewise` encoding. The
        outer mode always uses the disaggregated form, whose segment binaries
        carry the error band.
    max_breakpoints : int, default 256
        ``mode="outer"`` refinement cap per term. A term that reaches it stops
        refining; if the gap is still open the result says so (``"feasible"``
        with the bound reached), rather than growing the MILP without limit.

    Returns
    -------
    PWLTransformation
        Inspect ``.terms`` / ``.skipped`` / ``.model``; call ``.solve()``.

    Raises
    ------
    PWLTransformError
        If the model holds a non-algebraic relation (indicator, disjunction, SOS,
        logical, complementarity), which the incumbent check cannot verify.
    """
    return PWLTransformation(model, mode, segments, method, max_breakpoints)
