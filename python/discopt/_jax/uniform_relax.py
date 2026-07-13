"""Uniform factorable relaxation engine (issue #632).

One entry point — :func:`build_uniform_relaxation` — that relaxes *every* atom
class of the R1.1 canonical DAG (``canonical_expr.atomize`` taxonomy) soundly and
uniformly, composing via the auxiliary-variable method (AVM). This is the
substrate the federation cutover (``docs/dev/factorable-capability-blueprint.md``)
routes through; it is implemented as a **new module that runs ALONGSIDE** the
default ``build_milp_relaxation`` path — nothing here is wired into the default
build, and no new ``DISCOPT_*`` runtime flag is added. It is exercised only by
tests and the validation harness.

The pipeline (blueprint §3.1)::

    relax(canonical_dag, box) =
        dag   = canonicalize(model)                    # R1.1
        walk each root (objective + constraint bodies) bottom-up:
            for every nonlinear CNode:
                w         = new aux column
                [lo, hi]  = SOUND interval enclosure of the node over the box
                            (AVM: inner-atom bounds flow into the outer envelope)
                rows      = ENVELOPE_LIBRARY[kind](ctx, node, w)   # sound O.A.
        assemble the LP (c, A_ub, b_ub, bounds, integrality) in the SAME shape as
        build_milp_relaxation so it is directly comparable.

Coverage is **0 fallbacks by construction** (blueprint §3.5): every canonical node
is exactly one of the ``atomize`` kinds; each kind has a builder; and the *floor*
every builder guarantees is the sound interval box on the aux column
(``lo <= w <= hi``) — a valid outer relaxation that never cuts a feasible point.
On top of that floor each builder adds tighter sound rows (secant/tangent for a
definite-curvature 1-D atom, McCormick for a product, facets for min/max). There
is therefore no atom for which the objective drops to the separable feasibility
fallback; the "could not linearize the objective" path cannot fire.

Soundness (blueprint §3.4, CLAUDE.md §Development Philosophy) is **absolute and by
construction**: every row a builder emits is a proven valid inequality for the
lifted point ``(x, w=f(x))`` at every feasible ``x`` over the aux box. Each builder
documents its soundness argument inline. Where a class is not yet tightened to its
exact envelope (sign-straddling odd powers, wide multilinear boxes, general
``callN``), the builder emits the sound interval floor (plus any partial tightening)
and is reported ``loose-but-sound`` — never unsound, never a fallback.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional

import numpy as np

from discopt._jax.canonical_expr import (
    CanonicalDAG,
    CNode,
    canonicalize,
    reconstruct,
)
from discopt._jax.convexity.interval import Interval
from discopt._jax.convexity.interval_eval import evaluate_interval
from discopt._jax.milp_relaxation import MilpRelaxationModel
from discopt._jax.model_utils import flat_variable_bounds
from discopt.modeling.core import Model, ObjectiveSense

__all__ = [
    "LinForm",
    "Envelope",
    "UniformRelaxation",
    "ENVELOPE_LIBRARY",
    "build_uniform_relaxation",
    "relaxation_report",
]

# A finite endpoint magnitude beyond which we treat a bound as "not usable" for a
# tangent/secant/McCormick row (an infinite or astronomically large coefficient
# would be numerically meaningless and can inject spurious constraints). The aux
# interval floor still applies, so dropping such a row only loosens — never
# unsound.
_BIG = 1e19
# Below this box width a secant/tangent construction divides by ~0; the aux
# interval floor already pins the (near-degenerate) column, so we skip.
_MIN_WIDTH = 1e-12


def _finite(*vals: float) -> bool:
    return all(math.isfinite(v) and abs(v) < _BIG for v in vals)


# --------------------------------------------------------------------------- #
# LinForm — an affine expression over the relaxation's columns
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class LinForm:
    """``const + sum_j coef_j * col_j`` over relaxation columns (orig ∪ aux).

    Every canonical node's *value in the relaxation* is represented by a
    ``LinForm``: an affine node folds into an affine combination of columns (no
    aux); a nonlinear node is represented by its single aux column (``{w: 1}``).
    """

    coeffs: dict[int, float] = dataclasses.field(default_factory=dict)
    const: float = 0.0

    @classmethod
    def col(cls, j: int) -> "LinForm":
        return cls({j: 1.0}, 0.0)

    @classmethod
    def constant(cls, c: float) -> "LinForm":
        return cls({}, float(c))

    def scaled(self, s: float) -> "LinForm":
        return LinForm({j: c * s for j, c in self.coeffs.items()}, self.const * s)

    def __add__(self, other: "LinForm") -> "LinForm":
        out = dict(self.coeffs)
        for j, c in other.coeffs.items():
            out[j] = out.get(j, 0.0) + c
        return LinForm({j: c for j, c in out.items() if c != 0.0}, self.const + other.const)


# --------------------------------------------------------------------------- #
# Envelope — a builder's contribution (blueprint §3.2)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Envelope:
    """One atom's contribution: rows over (orig ∪ aux) cols + aux metadata.

    ``rows`` are ``(coeffs, rhs)`` meaning ``sum_j coeffs[j]*col_j <= rhs``.
    ``tight`` records whether the builder reached the exact envelope for this atom
    (``True``) or fell to the sound interval floor / a partial tightening
    (``False``) — consumed by :func:`relaxation_report` for the coverage census.
    """

    rows: list[tuple[dict[int, float], float]]
    tight: bool


@dataclasses.dataclass
class UniformRelaxation:
    """The assembled uniform relaxation of a model over a box."""

    model: MilpRelaxationModel
    n_orig: int
    n_aux: int
    obj_sense_sign: float  # +1 minimize native, -1 if maximize was negated
    obj_offset: float
    # per-node coverage: id(CNode) -> (kind, tight)
    coverage: dict[int, tuple[str, bool]]


# --------------------------------------------------------------------------- #
# Node classification for the envelope-library dispatch
# --------------------------------------------------------------------------- #
def _node_kind(node: CNode) -> str:
    """Map a nonlinear CNode to its ENVELOPE_LIBRARY key.

    Per-node (AVM granularity), not the maximal-atom kind: a ``call``/``pow`` is
    always a 1-D atom over its (single) child's LinForm, a ``prod`` is a product
    (``ratio`` when any exponent is negative), a ``callN`` is a multivariable
    intrinsic, and everything unrepresentable is ``opaque``.
    """
    if node.kind == "call":
        return "univariate_call"
    if node.kind == "pow":
        return "power"
    if node.kind == "prod":
        (exps,) = node.payload
        return "ratio" if any(e < 0 for e in exps) else "product"
    if node.kind == "callN":
        return "multivar"
    return "opaque"


# --------------------------------------------------------------------------- #
# 1-D function table: value, derivative, and a SOUND curvature verdict on [lo,hi]
# --------------------------------------------------------------------------- #
# curvature(lo, hi) -> "convex" | "concave" | None (indefinite / abstain).
# Every verdict is rigorous: for a monotone-second-derivative atom the sign of
# f'' is read off the sign of the box; for sin/cos the sign of f'' = -f is read
# off a sound interval enclosure. None => the builder uses the interval floor.
def _curv_const(kind: str) -> Callable[[float, float], Optional[str]]:
    return lambda lo, hi: kind


def _curv_by_sign(convex_when_nonneg: bool) -> Callable[[float, float], Optional[str]]:
    """f'' has the sign of ``x`` (or its negation). Sign-definite box => verdict."""

    def curv(lo: float, hi: float) -> Optional[str]:
        if lo >= 0.0:
            return "convex" if convex_when_nonneg else "concave"
        if hi <= 0.0:
            return "concave" if convex_when_nonneg else "convex"
        return None  # straddles the inflection at 0

    return curv


def _curv_sin(lo: float, hi: float) -> Optional[str]:
    # sin'' = -sin. Enclose sin over [lo,hi]; a sign-definite enclosure certifies.
    enc = Interval(np.asarray(float(lo)), np.asarray(float(hi)))
    from discopt._jax.convexity import interval as iv

    s = iv.sin(enc)
    if float(s.hi) <= 0.0:
        return "convex"
    if float(s.lo) >= 0.0:
        return "concave"
    return None


def _curv_cos(lo: float, hi: float) -> Optional[str]:
    # cos'' = -cos.
    enc = Interval(np.asarray(float(lo)), np.asarray(float(hi)))
    from discopt._jax.convexity import interval as iv

    c = iv.cos(enc)
    if float(c.hi) <= 0.0:
        return "convex"
    if float(c.lo) >= 0.0:
        return "concave"
    return None


# name -> (f, f', curvature, domain_ok(lo) )
_UNIVARIATE_FN: dict[str, tuple[Callable, Callable, Callable, Callable]] = {
    "exp": (np.exp, np.exp, _curv_const("convex"), lambda lo: True),
    "log": (np.log, lambda t: 1.0 / t, _curv_const("concave"), lambda lo: lo > 0.0),
    "log2": (
        np.log2,
        lambda t: 1.0 / (t * math.log(2.0)),
        _curv_const("concave"),
        lambda lo: lo > 0.0,
    ),
    "log10": (
        np.log10,
        lambda t: 1.0 / (t * math.log(10.0)),
        _curv_const("concave"),
        lambda lo: lo > 0.0,
    ),
    "log1p": (np.log1p, lambda t: 1.0 / (1.0 + t), _curv_const("concave"), lambda lo: lo > -1.0),
    "sqrt": (np.sqrt, lambda t: 0.5 / np.sqrt(t), _curv_const("concave"), lambda lo: lo >= 0.0),
    "sin": (np.sin, np.cos, _curv_sin, lambda lo: True),
    "cos": (np.cos, lambda t: -np.sin(t), _curv_cos, lambda lo: True),
    "sinh": (np.sinh, np.cosh, _curv_by_sign(True), lambda lo: True),
    "cosh": (np.cosh, np.sinh, _curv_const("convex"), lambda lo: True),
    "tanh": (np.tanh, lambda t: 1.0 - np.tanh(t) ** 2, _curv_by_sign(False), lambda lo: True),
    "atan": (np.arctan, lambda t: 1.0 / (1.0 + t * t), _curv_by_sign(False), lambda lo: True),
    "asin": (
        np.arcsin,
        lambda t: 1.0 / np.sqrt(1.0 - t * t),
        _curv_by_sign(True),
        lambda lo: lo > -1.0,
    ),
    "acos": (
        np.arccos,
        lambda t: -1.0 / np.sqrt(1.0 - t * t),
        _curv_by_sign(False),
        lambda lo: lo > -1.0,
    ),
    "asinh": (
        np.arcsinh,
        lambda t: 1.0 / np.sqrt(1.0 + t * t),
        _curv_by_sign(False),
        lambda lo: True,
    ),
    "acosh": (
        np.arccosh,
        lambda t: 1.0 / np.sqrt(t * t - 1.0),
        _curv_const("concave"),
        lambda lo: lo > 1.0,
    ),
    "atanh": (np.arctanh, lambda t: 1.0 / (1.0 - t * t), _curv_by_sign(True), lambda lo: lo > -1.0),
    "erf": (
        lambda t: math.erf(t),
        lambda t: 2.0 / math.sqrt(math.pi) * math.exp(-t * t),
        _curv_by_sign(False),
        lambda lo: True,
    ),
}


# --------------------------------------------------------------------------- #
# The builder context — the bottom-up AVM walk state
# --------------------------------------------------------------------------- #
class _Builder:
    """Holds the growing LP + the bottom-up node walk (AVM composition)."""

    def __init__(
        self,
        model: Model,
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        track_aux_exprs: bool = False,
    ):
        self.model = model
        self.n_orig = int(len(flat_lb))
        self.col_lb: list[float] = list(map(float, flat_lb))
        self.col_ub: list[float] = list(map(float, flat_ub))
        self.integrality: list[int] = [0] * self.n_orig
        self.rows: list[tuple[dict[int, float], float]] = []
        # id(CNode) -> LinForm (memoized rep; CSE: shared subexpr relaxed once)
        self._rep: dict[int, LinForm] = {}
        # id(CNode) -> (lo, hi) sound interval enclosure
        self._bounds: dict[int, tuple[float, float]] = {}
        self.coverage: dict[int, tuple[str, bool]] = {}
        self._ivbox = _interval_box(model, flat_lb, flat_ub)
        # Validation-only: aux column -> the modeling Expression whose exact value
        # that column represents (the node itself, a relaxed power, or a McCormick
        # partial product). Lets the soundness harness set every aux to its TRUE
        # value and verify no row cuts the lifted feasible point. Opt-in (each
        # entry costs a reconstruct) so the production build stays lean.
        self.track_aux_exprs = track_aux_exprs
        self.aux_expr: dict[int, object] = {}

    # -- columns / rows ----------------------------------------------------- #
    def new_aux(self, lo: float, hi: float, integ: bool = False) -> int:
        j = len(self.col_lb)
        self.col_lb.append(float(lo))
        self.col_ub.append(float(hi))
        self.integrality.append(1 if integ else 0)
        return j

    def add_row(self, coeffs: dict[int, float], rhs: float) -> None:
        # Drop rows whose payload is not finite/usable — the interval floor stands.
        if not math.isfinite(rhs) or abs(rhs) >= _BIG:
            return
        if any(not (math.isfinite(c) and abs(c) < _BIG) for c in coeffs.values()):
            return
        self.rows.append(({j: float(c) for j, c in coeffs.items() if c != 0.0}, float(rhs)))

    # -- interval enclosure of a node over the box (AVM bound propagation) --- #
    def bounds(self, node: CNode) -> tuple[float, float]:
        cached = self._bounds.get(id(node))
        if cached is not None:
            return cached
        expr = reconstruct(node, self.model)
        # A FRESH per-expr cache: ``evaluate_interval`` memoizes by ``id(expr)``,
        # but each ``reconstruct`` builds a new transient tree and Python reuses
        # ``id()`` of GC'd nodes, so a shared cache would return a STALE interval
        # for a different node — an UNSOUND, nondeterministic bound. Per-node bound
        # results are still memoized by the stable ``id(CNode)`` in ``self._bounds``.
        enc = evaluate_interval(expr, self.model, self._ivbox)
        lo = float(np.asarray(enc.lo))
        hi = float(np.asarray(enc.hi))
        if not (math.isfinite(lo)):
            lo = -math.inf
        if not (math.isfinite(hi)):
            hi = math.inf
        self._bounds[id(node)] = (lo, hi)
        return (lo, hi)

    # -- the recursive representation walk (bottom-up) ---------------------- #
    def rep(self, node: CNode) -> LinForm:
        cached = self._rep.get(id(node))
        if cached is not None:
            return cached
        out = self._rep_impl(node)
        self._rep[id(node)] = out
        return out

    def _rep_impl(self, node: CNode) -> LinForm:
        if node.kind == "var":
            return LinForm.col(node.payload)
        if node.kind == "const":
            return LinForm.constant(node.payload)
        if node.kind == "sum":
            coeffs, const = node.payload
            acc = LinForm.constant(const)
            for coef, child in zip(coeffs, node.children):
                acc = acc + self.rep(child).scaled(float(coef))
            return acc
        # Every remaining kind is a nonlinear atom: allocate an aux with the sound
        # interval floor, then let its builder add tighter sound rows.
        lo, hi = self.bounds(node)
        w = self.new_aux(lo, hi)
        if self.track_aux_exprs:
            self.aux_expr[w] = reconstruct(node, self.model)
        kind = _node_kind(node)
        env = ENVELOPE_LIBRARY[kind](self, node, w)
        for coeffs, rhs in env.rows:
            self.add_row(coeffs, rhs)
        self.coverage[id(node)] = (kind, env.tight)
        return LinForm.col(w)


def _interval_box(model: Model, flat_lb: np.ndarray, flat_ub: np.ndarray) -> dict:
    """Build the ``{Variable: Interval}`` box for :func:`evaluate_interval`."""
    box: dict = {}
    off = 0
    for v in model._variables:
        size = int(v.size)
        shape = tuple(getattr(v, "shape", ()) or ())
        lo = np.asarray(flat_lb[off : off + size], dtype=np.float64).reshape(shape)
        hi = np.asarray(flat_ub[off : off + size], dtype=np.float64).reshape(shape)
        box[v] = Interval(lo, hi)
        off += size
    return box


# --------------------------------------------------------------------------- #
# Shared 1-D secant/tangent emission (the composite-envelope math)
# --------------------------------------------------------------------------- #
def _emit_1d(
    ctx: _Builder,
    w: int,
    lt: LinForm,
    lo: float,
    hi: float,
    f: Callable[[float], float],
    fp: Callable[[float], float],
    curv: Optional[str],
) -> bool:
    """Emit the sound 1-D envelope of ``w = f(t)`` with ``t = lt ∈ [lo, hi]``.

    Soundness. For ``f`` convex on ``[lo,hi]``: (over) the secant chord lies on or
    above the graph, so ``w <= chord(t)`` holds at ``w=f(t)``; (under) every
    tangent line lies on or below the graph, so ``w >= tangent(t)`` holds. The
    concave case is the mirror image. Both use exact endpoint/point evaluations of
    ``f``/``f'`` (the lines are constructed to touch the graph, hence valid). When
    ``curv is None`` no line is sound, so only the aux interval floor stands.

    Returns ``True`` iff the exact two-sided envelope was emitted (``tight``).
    """
    if curv is None:
        return False
    width = hi - lo
    if not _finite(lo, hi) or width < _MIN_WIDTH:
        # Degenerate/unbounded box: the aux interval floor is the sound relaxation.
        return False
    try:
        flo, fhi = float(f(lo)), float(f(hi))
    except (ValueError, ArithmeticError):
        return False
    if not _finite(flo, fhi):
        return False
    slope = (fhi - flo) / width  # secant slope

    def _secant_row(sign: float) -> None:
        # sign*w <= sign*(flo + slope*(t-lo)); secant intercept a = flo - slope*lo.
        a = flo - slope * lo
        coeffs = {w: sign}
        for j, c in lt.scaled(-sign * slope).coeffs.items():
            coeffs[j] = coeffs.get(j, 0.0) + c
        ctx.add_row(coeffs, sign * (a + slope * lt.const))

    def _tangent_row(t0: float, sign: float) -> None:
        # sign*(w - (f(t0) + f'(t0)(t - t0))) >= 0  ->  -sign*w + sign*f'(t0)*t <= ...
        try:
            g, gp = float(f(t0)), float(fp(t0))
        except (ValueError, ArithmeticError):
            return
        if not _finite(g, gp):
            return
        intercept = g - gp * t0  # tangent line: gp*t + intercept
        # sign*w >= sign*(gp*t + intercept), with t = lt (cols) + lt.const:
        #   -sign*w + sign*gp*(cols) <= -sign*intercept - sign*gp*lt.const
        coeffs = {w: -sign}
        for j, c in lt.scaled(sign * gp).coeffs.items():
            coeffs[j] = coeffs.get(j, 0.0) + c
        ctx.add_row(coeffs, -sign * intercept - sign * gp * lt.const)

    mid = 0.5 * (lo + hi)
    if curv == "convex":
        _secant_row(+1.0)  # w <= secant
        for t0 in (lo, mid, hi):
            _tangent_row(t0, +1.0)  # w >= tangent
    else:  # concave
        _secant_row(-1.0)  # w >= secant
        for t0 in (lo, mid, hi):
            _tangent_row(t0, -1.0)  # w <= tangent
    return True


# --------------------------------------------------------------------------- #
# ENVELOPE_LIBRARY builders — one per atom kind, ALL sound (blueprint §3.2)
# --------------------------------------------------------------------------- #
def _build_univariate_call(ctx: _Builder, node: CNode, w: int) -> Envelope:
    """``w = fname(t)`` for a single-argument intrinsic; ``t`` = child LinForm.

    AVM: the argument child is relaxed first (affine => its LinForm; nonlinear =>
    its own aux with sound bounds), so a composite ``f(g(x))`` is uniformly a 1-D
    atom of ``f`` over ``g``'s enclosure — exactly the composition BARON does. The
    row set is the exact secant/tangent envelope when ``f`` has definite curvature
    on the argument box, else the sound interval floor.
    """
    fname: str = node.payload
    arg = node.children[0]
    lt = ctx.rep(arg)
    lo, hi = ctx.bounds(arg)
    if fname == "abs":
        return _build_abs(ctx, w, lt, lo, hi)
    entry = _UNIVARIATE_FN.get(fname)
    if entry is None:
        return Envelope(rows=[], tight=False)  # unknown intrinsic -> interval floor
    f, fp, curv_fn, dom_ok = entry
    if not dom_ok(lo):
        return Envelope(rows=[], tight=False)  # arg box violates domain -> floor
    tight = _emit_1d(ctx, w, lt, lo, hi, f, fp, curv_fn(lo, hi))
    return Envelope(rows=[], tight=tight)


def _build_abs(ctx: _Builder, w: int, lt: LinForm, lo: float, hi: float) -> Envelope:
    """``w = |t|`` — convex, piecewise linear. Exact hull on the box.

    Soundness: ``|t| >= t`` and ``|t| >= -t`` everywhere (under-estimators, valid
    subgradient facets); ``|t| <= secant`` between the endpoint images (over, from
    convexity). Together they are the exact convex hull of ``{(t,|t|)}``.
    """
    # w >= t  ->  t - w <= 0
    c1 = dict(lt.coeffs)
    c1[w] = c1.get(w, 0.0) - 1.0
    ctx.add_row(c1, -lt.const)
    # w >= -t ->  -t - w <= 0
    c2 = lt.scaled(-1.0).coeffs
    c2[w] = c2.get(w, 0.0) - 1.0
    ctx.add_row(c2, lt.const)
    tight = _emit_secant_only(ctx, w, lt, lo, hi, abs, sign=+1.0)
    return Envelope(rows=[], tight=tight)


def _emit_secant_only(ctx, w, lt, lo, hi, f, sign) -> bool:
    if not _finite(lo, hi) or (hi - lo) < _MIN_WIDTH:
        return False
    flo, fhi = float(f(lo)), float(f(hi))
    if not _finite(flo, fhi):
        return False
    slope = (fhi - flo) / (hi - lo)
    a = flo - slope * lo
    coeffs = {w: sign}
    for j, c in lt.scaled(-sign * slope).coeffs.items():
        coeffs[j] = coeffs.get(j, 0.0) + c
    ctx.add_row(coeffs, sign * (a + slope * lt.const))
    return True


def _pow_curv(p: float, lo: float, hi: float) -> Optional[str]:
    """Sound curvature of ``t**p`` on ``[lo,hi]`` (``f'' = p(p-1) t^(p-2)``)."""
    is_int = float(p).is_integer()
    if is_int and int(p) % 2 == 0:
        return "convex"  # even integer power: convex on all of R
    # Sign-definite box required otherwise (the only curvature change is at t=0,
    # and non-integer p needs t>=0 anyway).
    if lo < 0.0 < hi:
        return None
    if hi <= 0.0 and not is_int:
        return None  # negative base, fractional power: not real -> floor
    # On a sign-definite box the sign of f'' is constant; read it at the midpoint.
    mid = 0.5 * (lo + hi)
    if mid == 0.0:
        mid = hi if hi != 0.0 else lo
    if mid == 0.0:
        return None
    try:
        fpp = p * (p - 1.0) * (mid ** (p - 2.0))
    except (ValueError, ArithmeticError):
        return None
    if not math.isfinite(fpp):
        return None
    if fpp > 0.0:
        return "convex"
    if fpp < 0.0:
        return "concave"
    return "convex"  # p in {0,1} degenerate (won't reach: pow collapses those)


def _build_power(ctx: _Builder, node: CNode, w: int) -> Envelope:
    """``w = t**p``; ``t`` = base LinForm, ``p`` = constant exponent.

    AVM: the base is relaxed first, so ``(affine)**2``, ``f(x)**2`` and ``x**p``
    are all uniformly a 1-D power atom over the base's enclosure. Exact
    secant/tangent envelope on a definite-curvature (sign-definite or even-power)
    box; interval floor on a sign-straddling odd/fractional power (loose-but-sound,
    the blueprint's two-piece-hull tightening TODO).
    """
    p = float(node.payload)
    base = node.children[0]
    lt = ctx.rep(base)
    lo, hi = ctx.bounds(base)
    curv = _pow_curv(p, lo, hi)
    f = lambda t: float(t) ** p  # noqa: E731
    fp = lambda t: p * (float(t) ** (p - 1.0))  # noqa: E731
    tight = _emit_1d(ctx, w, lt, lo, hi, f, fp, curv)
    return Envelope(rows=[], tight=tight)


def _emit_mccormick(ctx: _Builder, w: int, la: LinForm, ba, lb_: LinForm, bb) -> None:
    """Emit the 4 McCormick rows for ``w = a*b`` over ``a∈ba``, ``b∈bb``.

    Soundness (McCormick 1976): the four bilinear inequalities
    ``w >= aL b + bL a - aL bL``, ``w >= aH b + bH a - aH bH``,
    ``w <= aH b + bL a - aH bL``, ``w <= aL b + bH a - aL bH`` are each valid for
    every ``(a,b)`` in the box with ``w=a*b`` (they are the convex hull of the
    bilinear graph over a box). Rows are substituted with ``a=la``, ``b=lb_``. A
    row is skipped when an endpoint is non-finite (the aux interval floor stands).
    """
    aL, aH = ba
    bL, bH = bb
    for coef_a, coef_b, cc, sign in (
        (bL, aL, -aL * bL, +1.0),  # w >= aL*b + bL*a - aL*bL
        (bH, aH, -aH * bH, +1.0),  # w >= aH*b + bH*a - aH*bH
        (bL, aH, -aH * bL, -1.0),  # w <= aH*b + bL*a - aH*bL
        (bH, aL, -aL * bH, -1.0),  # w <= aL*b + bH*a - aL*bH
    ):
        if not _finite(coef_a, coef_b, cc):
            continue
        # E = coef_a*a + coef_b*b + cc.  sign=+1 => w >= E (under); sign=-1 => w <= E
        # (over).  As a `<=` row:  (-sign)*w + sign*E <= 0.
        coeffs = {w: -sign}
        for j, c in la.scaled(sign * coef_a).coeffs.items():
            coeffs[j] = coeffs.get(j, 0.0) + c
        for j, c in lb_.scaled(sign * coef_b).coeffs.items():
            coeffs[j] = coeffs.get(j, 0.0) + c
        rhs = -sign * (cc + coef_a * la.const + coef_b * lb_.const)
        ctx.add_row(coeffs, rhs)


def _factor_value(
    ctx: _Builder, base: CNode, e: float
) -> tuple[LinForm, tuple[float, float], object]:
    """Value + bounds + tracking-expr of a product factor ``base**e``.

    Relaxes the power into its own 1-D aux when ``e != 1`` (AVM). The third return
    is the modeling Expression for that factor (validation tracking only).
    """
    lb_base = ctx.rep(base)
    bb = ctx.bounds(base)
    base_expr = reconstruct(base, ctx.model) if ctx.track_aux_exprs else None
    if e == 1.0:
        return lb_base, bb, base_expr
    # Relax base**e as its own 1-D power atom into a fresh aux (AVM).
    lo, hi = _pow_bounds(bb[0], bb[1], e)
    aux = ctx.new_aux(lo, hi)
    factor_expr: object = None
    if ctx.track_aux_exprs and base_expr is not None:
        factor_expr = base_expr**e
        ctx.aux_expr[aux] = factor_expr
    curv = _pow_curv(e, bb[0], bb[1])
    f = lambda t: float(t) ** e  # noqa: E731
    fp = lambda t: e * (float(t) ** (e - 1.0))  # noqa: E731
    _emit_1d(ctx, aux, lb_base, bb[0], bb[1], f, fp, curv)
    return LinForm.col(aux), (lo, hi), factor_expr


def _pow_bounds(lo: float, hi: float, e: float) -> tuple[float, float]:
    enc = Interval(np.asarray(float(lo)), np.asarray(float(hi)))
    from discopt._jax.convexity import interval as iv

    if float(e).is_integer():
        r = enc ** int(e)
    elif lo > 0.0:
        r = iv.exp(Interval.point(float(e)) * iv.log(enc))
    else:
        return (-math.inf, math.inf)
    rl, rh = float(np.asarray(r.lo)), float(np.asarray(r.hi))
    return (rl if math.isfinite(rl) else -math.inf, rh if math.isfinite(rh) else math.inf)


def _build_product(ctx: _Builder, node: CNode, w: int) -> Envelope:
    """``w = prod_i base_i**e_i`` (all exponents positive => product/monomial).

    Recursive pairwise McCormick (sound): each factor value (relaxing any e≠1
    power first) is folded left-to-right, allocating an intermediate aux for every
    partial product except the last, whose aux IS ``w``. Every step is a sound
    bilinear McCormick relaxation, so the composition is a sound outer relaxation
    of the product. Tight for a single bilinear pair on a box; loose-but-sound for
    wide / high-arity multilinear boxes (blueprint TODO: simultaneous multilinear
    envelope / log-space for ``(prod x)**a``).
    """
    (exps,) = node.payload
    factors = [_factor_value(ctx, ch, float(e)) for ch, e in zip(node.children, exps)]
    tight = _fold_product(ctx, w, factors)
    return Envelope(rows=[], tight=tight)


def _build_ratio(ctx: _Builder, node: CNode, w: int) -> Envelope:  # noqa: D401
    """``w = prod_i base_i**e_i`` with a negative exponent (division).

    Same recursive McCormick fold as ``product`` — a negative-exponent factor
    ``base**(-k)`` is relaxed by :func:`_factor_value` as a 1-D power atom (convex
    reciprocal on a sign-definite base), then McCormick-combined. Sign-indefinite
    denominator => that factor's bounds are unbounded => its McCormick rows are
    skipped and the aux interval floor stands (sound; no finite bound, which is an
    orthogonal bound-validity concern, not a soundness one).
    """
    (exps,) = node.payload
    factors = [_factor_value(ctx, ch, float(e)) for ch, e in zip(node.children, exps)]
    tight = _fold_product(ctx, w, factors)
    return Envelope(rows=[], tight=tight and len(factors) == 2)


def _fold_product(ctx: _Builder, w: int, factors: list) -> bool:
    if len(factors) < 2:
        # A product node always has >= 2 factors; guard defensively.
        return False
    acc_lin, acc_b, acc_expr = factors[0]
    for k in range(1, len(factors)):
        fl, fb, fe = factors[k]
        tb = _interval_mul(acc_b, fb)
        if k == len(factors) - 1:
            target = w  # last product IS the node's aux (floor bounds already set)
        else:
            target = ctx.new_aux(tb[0], tb[1])
            if ctx.track_aux_exprs:
                ctx.aux_expr[target] = acc_expr * fe
        _emit_mccormick(ctx, target, acc_lin, acc_b, fl, fb)
        acc_lin, acc_b = LinForm.col(target), tb
        if ctx.track_aux_exprs:
            acc_expr = acc_expr * fe
    # "tight" only for a plain bilinear pair with finite endpoints.
    return len(factors) == 2 and _finite(*factors[0][1], *factors[1][1])


def _interval_mul(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    prods = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    prods = [p for p in prods if not math.isnan(p)]
    if not prods:
        return (-math.inf, math.inf)
    return (min(prods), max(prods))


def _build_multivar(ctx: _Builder, node: CNode, w: int) -> Envelope:
    """Multivariable intrinsic ``callN`` (min / max / centropy / …).

    Soundness:
      * ``max(a_i)``: ``max >= a_i`` for each argument (valid lower facets); the
        aux upper bound caps it. These are the exact convex-hull under-estimators.
      * ``min(a_i)``: ``min <= a_i`` for each argument (valid upper facets).
      * anything else (centropy, general n-ary): the sound aux interval floor.
    """
    fname: str = node.payload
    args = [ctx.rep(c) for c in node.children]
    if fname == "max" and len(args) >= 2:
        for la in args:  # w >= a_i  ->  a_i - w <= 0
            coeffs = dict(la.coeffs)
            coeffs[w] = coeffs.get(w, 0.0) - 1.0
            ctx.add_row(coeffs, -la.const)
        return Envelope(rows=[], tight=True)
    if fname == "min" and len(args) >= 2:
        for la in args:  # w <= a_i  ->  w - a_i <= 0
            coeffs = la.scaled(-1.0).coeffs
            coeffs[w] = coeffs.get(w, 0.0) + 1.0
            ctx.add_row(coeffs, la.const)
        return Envelope(rows=[], tight=True)
    return Envelope(rows=[], tight=False)  # interval floor (loose-but-sound)


def _build_opaque(ctx: _Builder, node: CNode, w: int) -> Envelope:
    """Unrepresentable node (CustomCall / MatMul / array / sign-spanning).

    The aux interval floor ``lo <= w <= hi`` is a sound outer relaxation on this
    node only (never an objective-wide feasibility drop) — the blueprint's
    "uniform-by-refusal" contract (§2 last row).
    """
    return Envelope(rows=[], tight=False)


ENVELOPE_LIBRARY: dict[str, Callable[[_Builder, CNode, int], Envelope]] = {
    "univariate_call": _build_univariate_call,
    "power": _build_power,
    "product": _build_product,
    "ratio": _build_ratio,
    "multivar": _build_multivar,
    "opaque": _build_opaque,
}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_uniform_relaxation(
    model: Model,
    box: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> UniformRelaxation:
    """Build the uniform factorable relaxation of ``model`` over ``box``.

    Parameters
    ----------
    model : Model
    box : optional ``(flat_lb, flat_ub)`` — the B&B node box in flat variable
        order (like ``build_milp_relaxation``'s ``bound_override``). Defaults to
        the model's declared variable bounds.

    Returns
    -------
    UniformRelaxation
        Wraps a :class:`MilpRelaxationModel` (same output contract as
        ``build_milp_relaxation`` — ``.solve(backend="simplex")`` yields a sound LP
        lower bound via discopt's in-house Rust simplex) plus per-node coverage.
    """
    if box is None:
        flat_lb, flat_ub = flat_variable_bounds(model)
    else:
        flat_lb = np.asarray(box[0], dtype=np.float64)
        flat_ub = np.asarray(box[1], dtype=np.float64)

    dag: CanonicalDAG = canonicalize(model)
    ctx = _Builder(model, flat_lb, flat_ub)

    # Objective -> c, obj_offset (minimize convention; maximize is negated so the
    # reported LP bound is a valid lower bound on the minimize-equivalent, matching
    # build_milp_relaxation).
    sign = 1.0
    obj_lin = LinForm()
    if dag.objective is not None:
        obj_lin = ctx.rep(dag.objective)
        if model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
            sign = -1.0
            obj_lin = obj_lin.scaled(-1.0)

    # Constraints (normalized ``body sense 0``) -> rows.
    for con, cnode in zip(model._constraints, dag.constraints):
        lc = ctx.rep(cnode)
        sense = con.sense
        rhs_shift = float(con.rhs)
        if sense == "<=":  # body <= rhs
            ctx.add_row(lc.coeffs, rhs_shift - lc.const)
        elif sense == ">=":  # body >= rhs  -> -body <= -rhs
            ctx.add_row(lc.scaled(-1.0).coeffs, -(rhs_shift) + lc.const)
        elif sense == "==":  # body == rhs (both directions)
            ctx.add_row(lc.coeffs, rhs_shift - lc.const)
            ctx.add_row(lc.scaled(-1.0).coeffs, -(rhs_shift) + lc.const)

    # Assemble the LP.
    n_cols = len(ctx.col_lb)
    c = np.zeros(n_cols, dtype=np.float64)
    for j, coef in obj_lin.coeffs.items():
        c[j] += coef
    obj_offset = obj_lin.const

    if ctx.rows:
        import scipy.sparse as sp

        data: list[float] = []
        rows_idx: list[int] = []
        cols_idx: list[int] = []
        b = np.zeros(len(ctx.rows), dtype=np.float64)
        for i, (coeffs, rhs) in enumerate(ctx.rows):
            b[i] = rhs
            for j, coef in coeffs.items():
                data.append(coef)
                rows_idx.append(i)
                cols_idx.append(j)
        A_ub = sp.csr_matrix(
            (data, (rows_idx, cols_idx)), shape=(len(ctx.rows), n_cols), dtype=np.float64
        )
    else:
        A_ub = None
        b = None

    bounds = list(zip(ctx.col_lb, ctx.col_ub))
    # Pure LP relaxation (integrality relaxed at the root) — a sound lower bound,
    # matching the federation's root-node LP convention.
    milp = MilpRelaxationModel(
        c=c,
        A_ub=A_ub,
        b_ub=b,
        bounds=bounds,
        obj_offset=obj_offset,
        integrality=None,
    )
    return UniformRelaxation(
        model=milp,
        n_orig=ctx.n_orig,
        n_aux=n_cols - ctx.n_orig,
        obj_sense_sign=sign,
        obj_offset=obj_offset,
        coverage=dict(ctx.coverage),
    )


def relaxation_report(
    model: Model,
    box: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    """Per-atom coverage census for a model (blueprint §3.6 metric 1).

    Returns ``{"fallbacks": int, "n_atoms": int, "by_kind": {kind: {"total","tight",
    "loose"}}, "tight": int, "loose": int}``. ``fallbacks`` is the number of atoms
    left with NO sound envelope — **0 by construction** (every atom carries at
    least the sound interval floor). ``loose`` atoms are sound but not yet at their
    exact envelope (deferred tightening).
    """
    rel = build_uniform_relaxation(model, box)
    by_kind: dict[str, dict[str, int]] = {}
    tight = loose = 0
    for _kind, is_tight in rel.coverage.values():
        d = by_kind.setdefault(_kind, {"total": 0, "tight": 0, "loose": 0})
        d["total"] += 1
        if is_tight:
            d["tight"] += 1
            tight += 1
        else:
            d["loose"] += 1
            loose += 1
    return {
        "fallbacks": 0,
        "n_atoms": len(rel.coverage),
        "by_kind": by_kind,
        "tight": tight,
        "loose": loose,
        "n_aux": rel.n_aux,
    }
