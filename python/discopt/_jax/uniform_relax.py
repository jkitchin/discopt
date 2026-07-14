"""Uniform factorable relaxation engine (issue #632).

One entry point — :func:`build_uniform_relaxation` — that relaxes *every* atom
class of the R1.1 canonical DAG (``canonical_expr.atomize`` taxonomy) soundly and
uniformly, composing via the auxiliary-variable method (AVM). The federation
cutover (``docs/dev/factorable-capability-blueprint.md``) is complete: this engine
**IS the default relaxation** — ``build_milp_relaxation`` (``milp_relaxation.py``)
unconditionally delegates every default build to :func:`build_uniform_relaxation`,
so this module is the load-bearing per-node certificate path (in-house Rust
simplex), not an alongside experiment. No new ``DISCOPT_*`` runtime flag is added:
the delegation is unconditional. The full-panel correctness gate (``global50``,
``cert0``) is green on this path (``incorrect_count = 0``).

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
from typing import Callable, Optional, cast

import numpy as np

from discopt._jax.canonical_expr import (
    CanonicalDAG,
    CNode,
    canonicalize,
    is_affine,
    reconstruct,
    var_support,
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
    # Structural varmaps over ORIGINAL columns (0..n_orig-1) that the proven
    # legacy separators (PSD / RLT / edge-concave / univariate-square / multilinear)
    # consume — see ``_uniform_relaxation_delegate``. Each entry maps a product /
    # power of original variables to the aux column that lifts it in this
    # relaxation, so the separators fire on the engine's decomposition exactly as
    # they did on the federation's. Every registration is EXACT (the aux equals the
    # product/power of the named originals, tied to them by the McCormick /
    # secant-tangent rows already emitted), so every separated cut stays sound.
    bilinear_map: dict[tuple[int, int], int]
    monomial_map: dict[tuple[int, int], int]
    trilinear_map: dict[tuple[int, int, int], int]
    multilinear_map: dict[tuple[int, ...], int]
    univariate_square_map: dict[tuple[int, int], int]
    # Composite convex/concave lifts (issue #632 P2) — ``CompositeMultivarRelaxation``
    # specs the ``_separate_convex`` Kelley loop consumes (value_fn/grad_fn/idxs/
    # aux_col/curvature). Each is a certified-convex/-concave multivariate node
    # lifted to a single aux; the OA tangents added lazily at the LP point are
    # global under-/over-estimators (sound, never cut a feasible point).
    composite_multivar_specs: list


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
# Per-model analysis cache (issue #632 EP1) — box-INDEPENDENT work computed once
# --------------------------------------------------------------------------- #
# The uniform engine is the load-bearing per-node relaxation path: every B&B node
# calls ``build_uniform_relaxation`` with the SAME model and a different box. The
# analysis below is a pure function of the model (not the box): the canonical DAG,
# the reconstructed expression trees, the global DCP curvature verdicts, and the
# compiled ``value_fn``/``grad_fn`` of each node. Re-deriving them at every build
# was the dominant per-node cost (EP0: ~294 ms/node on nvs09). We compute them once
# and pin them on the model, so the second and later builds read through the cache.
#
# Two box-DEPENDENT results are also cached, but soundly keyed by the box so a cache
# hit is byte-identical to a fresh computation:
#   * interval enclosures, keyed by ``(id(cnode), support-restricted box bytes)`` —
#     the enclosure of a node depends only on the box entries for the variables in
#     its support, so branching on one variable invalidates only the nodes that
#     depend on it;
#   * interval-Hessian curvature certificates, with MONOTONE INHERITANCE: a
#     ``convex``/``concave`` verdict proven on a box is valid — and, because the
#     interval Hessian is inclusion-monotone (a sub-box's Hessian enclosure is a
#     subset of the box's, so the same Gershgorin sign certificate holds at an
#     equal-or-coarser refinement), *re-certified with the same verdict* — on every
#     sub-box. So returning a proven super-box's verdict for a query sub-box is
#     byte-identical to re-proving it. Abstained boxes are also recorded to skip
#     re-proving a box that is not meaningfully smaller (see ``_curvature_cert``).
#
# Staleness is guarded by a token ``(len(_variables), len(_constraints),
# id(_objective))``; a mismatch rebuilds the entry. Mid-solve model mutation is
# unsupported, but a stale *objective* would be flat-out wrong, so the token is
# mandatory. The pinned CNodes make every ``id(cnode)`` key stable for the model's
# life (the cache keeps the DAG alive), which also stabilizes ``expr_id`` in every
# ``CompositeMultivarRelaxation``.
_ANALYSIS_ATTR = "_uniform_relax_analysis"
_UNSET = object()
# Stable empty-box bytes for the support-box key of a variable-free (constant) node.
_EMPTY_BOX = np.empty(0, dtype=np.float64)
# Cap on the per-model ``bounds_by_box`` dict before it is cleared (branching over a
# deep tree can accumulate many distinct support-boxes; the per-build ``self._bounds``
# memo still absorbs the within-build hits after a clear).
_BOUNDS_CACHE_CAP = 500_000
# Cap on the per-node proven/abstained curvature-box lists (drop-oldest).
_CERT_LIST_CAP = 8


class _TracedEvalFn:
    """Lazy, byte-neutral replacement for re-tracing a JAX callable each call
    (issue #632 EP5).

    Wraps a pure JAX function ``fn`` — a ``compile_expression`` output or its
    ``jax.grad`` — that the separation loop (``mccormick_lp._separate_convex``)
    calls up to 8 rounds per node. A *bare* ``jax.grad`` re-runs the autodiff
    trace (``linearize`` / ``ad.py`` process) on **every** call in interpreted
    JAX — a large avoidable per-node cost (measured on nvs09: the full
    ``m.solve()`` drops 44.9 → 32.8 s and the ``ad.py``+``linearize`` cProfile
    tottime drops 3.23 → 0.085 s). This wrapper traces ``fn`` to a
    jaxpr **once** on first use (``jax.make_jaxpr``) and thereafter evaluates that
    jaxpr op-by-op via ``jax.core.eval_jaxpr`` — the SAME primitive-by-primitive
    dispatch the eager call performs, so the result is **bit-identical** to
    ``fn(x)`` (verified on the EP5 corpus lift points: 0/311 bit mismatches on
    value AND grad across nvs09/tspn05), while the re-linearization happens once
    for the model's life (measured ~5.7× on the grad path).

    NOT ``jax.jit``: XLA fusion reorders float operations and is **not**
    bit-identical (measured value drift ~7e-15, grad drift ~2e-15 on the same
    points), which would make the separation cut sequence differ — a
    bound-CHANGING optimization (EP4b/EP6 territory), not this bound-neutral item.

    Lazy: the trace is deferred to the first separation call, so a lifted spec
    that is never actually separated pays nothing beyond the cheap
    ``compile_expression`` closure. Exception-transparent: a trace failure at
    first use raises exactly where the eager call would have, so
    ``_separate_convex``'s try/except skips that spec's separation — the same
    sound no-op as before.
    """

    __slots__ = ("_fn", "_jaxpr", "_consts", "_eval")

    def __init__(self, fn):
        self._fn = fn
        self._jaxpr = None
        self._consts = None
        self._eval = None

    def __call__(self, x):
        ev = self._eval
        if ev is None:
            import jax
            from jax import core as _jcore

            closed = jax.make_jaxpr(self._fn)(x)
            self._jaxpr = closed.jaxpr
            self._consts = closed.consts
            ev = self._eval = _jcore.eval_jaxpr
        return ev(self._jaxpr, self._consts, x)[0]


def _analysis_token(model: Model) -> tuple:
    return (len(model._variables), len(model._constraints), id(model._objective))


class _ModelAnalysisCache:
    """Box-independent analysis pinned on a model (see the module note above)."""

    __slots__ = (
        "token",
        "dag",
        "flat_expr",
        "expr",
        "dcp",
        "compiled",
        "support_cols",
        "bounds_by_box",
        "hessian_certs",
        "hessian_abstain",
    )

    def __init__(self, model: Model, dag: CanonicalDAG):
        from discopt._jax.canonical_expr import _flat_var_expr

        self.token = _analysis_token(model)
        self.dag = dag  # pins CNodes -> stable id(cnode) for the model's life
        self.flat_expr = _flat_var_expr(model)
        # id(cnode) -> reconstructed Expression (box-independent; pinned so a
        # shared evaluate_interval id() memo can never go stale — see bounds()).
        self.expr: dict[int, object] = {}
        # id(cnode) -> classify_expr verdict (Curvature enum or None on failure).
        self.dcp: dict[int, object] = {}
        # id(cnode) -> (value_fn, grad_fn) or None if compilation failed.
        self.compiled: dict[int, object] = {}
        # id(cnode) -> sorted tuple of support columns, or None if the subtree
        # hides variables inside an opaque node (then the full box is the key).
        self.support_cols: dict[int, object] = {}
        # (id(cnode), lb.tobytes(), ub.tobytes()) -> (lo, hi) interval enclosure.
        self.bounds_by_box: dict[tuple, tuple[float, float]] = {}
        # id(cnode) -> [(lo_support, hi_support, verdict)] proven boxes (capped).
        self.hessian_certs: dict[int, list[tuple[np.ndarray, np.ndarray, str]]] = {}
        # id(cnode) -> [(lo_support, hi_support)] abstained boxes (capped).
        self.hessian_abstain: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}


def _get_analysis_cache(model: Model) -> _ModelAnalysisCache:
    """Fetch (or rebuild if stale) the model's box-independent analysis cache."""
    token = _analysis_token(model)
    cache = model.__dict__.get(_ANALYSIS_ATTR)
    if cache is None or cache.token != token:
        cache = _ModelAnalysisCache(model, canonicalize(model))
        model.__dict__[_ANALYSIS_ATTR] = cache
    return cache


def _node_support_cols(node: CNode) -> Optional[tuple[int, ...]]:
    """Original columns a node's interval enclosure can depend on.

    Returns a sorted tuple of flat variable indices, or ``None`` if the subtree
    contains an ``opaque`` node. ``opaque`` nodes wrap an original subexpression
    whose variables ``var_support`` does NOT report, so keying such a node's
    enclosure by ``var_support`` would collide across genuinely different boxes —
    an unsound stale bound. In that case the caller keys on the full box instead.
    """
    acc: set[int] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.kind == "opaque":
            return None
        if n.kind == "var":
            acc.add(int(n.payload))
        else:
            stack.extend(n.children)
    return tuple(sorted(acc))


def _box_subset(lo: np.ndarray, hi: np.ndarray, plo: np.ndarray, phi: np.ndarray) -> bool:
    """True iff ``[lo, hi] ⊆ [plo, phi]`` elementwise (exact — no slack)."""
    return bool(np.all(lo >= plo) and np.all(hi <= phi))


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
        analysis: "Optional[_ModelAnalysisCache]" = None,
        track_aux_exprs: bool = False,
    ):
        self.model = model
        # Box-independent per-model analysis (canonical DAG, reconstructed exprs,
        # DCP verdicts, compiled fns, curvature certs) — computed once, read here.
        # Falls back to the model's cache when constructed standalone (tests).
        self._analysis = analysis if analysis is not None else _get_analysis_cache(model)
        # Original (flat) box in np form, for keying the box-dependent caches.
        self._flat_lb = np.asarray(flat_lb, dtype=np.float64)
        self._flat_ub = np.asarray(flat_ub, dtype=np.float64)
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
        # Structural varmaps over ORIGINAL columns for the legacy separators
        # (populated by the builders as they lift each product/power of bare
        # originals; see UniformRelaxation for the soundness contract).
        self.bilinear_map: dict[tuple[int, int], int] = {}
        self.monomial_map: dict[tuple[int, int], int] = {}
        self.trilinear_map: dict[tuple[int, int, int], int] = {}
        self.multilinear_map: dict[tuple[int, ...], int] = {}
        self.univariate_square_map: dict[tuple[int, int], int] = {}
        # Composite convex/concave lifts (issue #632 P2): each certified-convex or
        # -concave multivariate nonlinear node the engine would otherwise decompose
        # loosely is lifted to a single aux and registered here so the existing
        # ``MccormickLPRelaxer._separate_convex`` Kelley loop adds its EXACT
        # supporting tangent at the LP point each round (outer-approximation cutting
        # planes, added lazily — not pre-seeded). A tangent of a convex function is
        # a global underestimator (concave: overestimator), so no feasible point is
        # ever cut; sound by construction. Populated by ``_try_convex_lift``.
        self.composite_multivar_specs: list = []
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

    # -- structural varmap registration (for the legacy separators) --------- #
    def single_orig_col(self, lt: LinForm) -> Optional[int]:
        """Return ``i`` iff ``lt`` is EXACTLY the bare original variable ``x_i``.

        Only a coefficient-1, zero-constant, single-original-column LinForm is a
        bare original; a scaled/shifted/aux LinForm (e.g. ``2 x_i``, ``x_i+1``, an
        aux column) is rejected. This is the soundness gate for registering a lift:
        the separators treat the registered aux as the product/power of exactly the
        *named originals*, so registering ``(2 x_i)**2`` under key ``(i, 2)`` (which
        would be ``4 x_i**2``, not ``x_i**2``) MUST be refused.
        """
        if lt.const != 0.0:
            return None
        items = [(j, c) for j, c in lt.coeffs.items() if c != 0.0]
        if len(items) != 1:
            return None
        j, c = items[0]
        if c == 1.0 and 0 <= j < self.n_orig:
            return j
        return None

    def single_var_affine(self, lt: LinForm) -> Optional[tuple[int, float]]:
        """Return ``(i, c)`` iff ``lt`` is ``c * x_i`` (one original, zero const).

        Unlike :meth:`single_orig_col` this accepts any nonzero coefficient (not
        just 1), so ``2 x_i`` is recognised as ``(i, 2.0)``. Used by the product
        builder to aggregate factors that are affine multiples of a *single*
        original into a univariate power (the ``(c x_i)·x_i == c x_i**2`` class).
        """
        if lt.const != 0.0:
            return None
        items = [(j, c) for j, c in lt.coeffs.items() if c != 0.0]
        if len(items) != 1:
            return None
        j, c = items[0]
        if 0 <= j < self.n_orig:
            return (j, c)
        return None

    def register_power(self, i: int, p: int, col: int) -> None:
        """Register an aux holding ``x_i**p`` (``p`` integer >= 2) as a monomial."""
        self.monomial_map[(i, p)] = col
        if p == 2:
            self.univariate_square_map[(i, 2)] = col

    def register_product(self, cols: list[int], col: int) -> None:
        """Register an aux holding ``prod(x_c for c in cols)`` (distinct originals).

        Bilinear (2), trilinear (3) and higher-arity (>=4) distinct-variable
        products go to the map the matching separator reads; a non-distinct or
        empty ``cols`` is ignored (never mis-registered).
        """
        if len(set(cols)) != len(cols) or len(cols) < 2:
            return
        key = tuple(sorted(cols))
        if len(key) == 2:
            self.bilinear_map[(key[0], key[1])] = col
        elif len(key) == 3:
            self.trilinear_map[(key[0], key[1], key[2])] = col
        else:
            self.multilinear_map[key] = col

    def add_row(self, coeffs: dict[int, float], rhs: float) -> None:
        # Drop rows whose payload is not finite/usable — the interval floor stands.
        if not math.isfinite(rhs) or abs(rhs) >= _BIG:
            return
        if any(not (math.isfinite(c) and abs(c) < _BIG) for c in coeffs.values()):
            return
        self.rows.append(({j: float(c) for j, c in coeffs.items() if c != 0.0}, float(rhs)))

    # -- box-independent analysis read-through helpers (issue #632 EP1) ------- #
    def _expr(self, node: CNode):
        """Reconstructed :class:`Expression` for ``node`` (box-independent, pinned).

        Cached on the model so a node is reconstructed once for the model's life.
        Pinning also kills the historical stale-``id()`` hazard: ``evaluate_interval``
        memoizes by ``id(expr)``, and a transient reconstruct tree could be GC'd and
        its ``id()`` reused for a DIFFERENT node — a shared memo would then return an
        unsound bound. The pinned tree keeps ``id(expr)`` stable and live.
        """
        cache = self._analysis.expr
        nid = id(node)
        e = cache.get(nid)
        if e is None:
            e = reconstruct(node, self.model, self._analysis.flat_expr)
            cache[nid] = e
        return e

    def _dcp(self, node: CNode):
        """Global DCP curvature verdict of ``node`` (box-independent; ``None`` on
        classifier failure — the same abstain the inline try/except produced)."""
        cache = self._analysis.dcp
        nid = id(node)
        v = cache.get(nid, _UNSET)
        if v is not _UNSET:
            return v
        from discopt._jax.convexity import classify_expr

        try:
            v = classify_expr(self._expr(node), self.model)
        except Exception:
            v = None
        cache[nid] = v
        return v

    def _compiled(self, node: CNode):
        """``(value_fn, grad_fn)`` for ``node`` (the *function of x*, box-independent),
        or ``None`` if compilation failed (the same abstain as the inline path).

        The returned fns are lazy, byte-neutral :class:`_TracedEvalFn` wrappers
        (issue #632 EP5): ``compile_expression`` (which only builds a Python DAG
        closure, no trace) stays eager so the lift/no-lift decision is byte-identical
        to before, but the expensive autodiff/DAG-walk trace is deferred to the first
        ``_separate_convex`` use and then reused (``eval_jaxpr``) for the model's life
        instead of re-linearizing on every round — see :class:`_TracedEvalFn`.
        """
        cache = self._analysis.compiled
        nid = id(node)
        v: object = cache.get(nid, _UNSET)
        if v is not _UNSET:
            return v
        import os

        if os.environ.get("DISCOPT_ANALYTIC_SEPGRAD") == "1":
            v = self._compiled_analytic(node)
            if v is not None:
                cache[nid] = v
                return v
            # fall through to the JAX path on any construction failure (soundness:
            # a missing analytic atom must not silently drop separation).

        import jax
        import jax.numpy as jnp

        from discopt._jax.dag_compiler import compile_expression

        try:
            f = compile_expression(self._expr(node), self.model)
            grad_f = jax.grad(lambda xv: jnp.reshape(f(xv), ()))
            v = (_TracedEvalFn(f), _TracedEvalFn(grad_f))
        except Exception:
            v = None
        cache[nid] = v
        return v

    def _compiled_analytic(self, node: CNode):
        """F2′ spike: ``(value_fn, grad_fn)`` computed analytically over the engine's
        own factorable IR via forward-mode interval AD at a *point* box, with NO JAX.

        The separation cut ``d ≥ g(x₀) + ∇g(x₀)·(x−x₀)`` needs ``g(x₀)`` and
        ``∇g(x₀)``; ``g`` is a known factorable expression, so its gradient is the
        exact derivative of the IR — computed here by pinning every variable to its
        point value (``Interval.point``) so the AD's gradient interval collapses to
        the exact point gradient. Deterministic (no XLA float reordering) and
        JAX-free on the hot path. Returns ``None`` if the atom set isn't covered
        (caller falls back to JAX). Gated by ``DISCOPT_ANALYTIC_SEPGRAD``.
        """
        try:
            import numpy as _np

            from discopt._jax.convexity.interval import Interval as _Ivl
            from discopt._jax.convexity.interval_ad import _flat_size, interval_hessian

            expr = self._expr(node)
            model = self.model
            variables = model._variables
            # Prefix-sum flat offsets (mirrors interval_ad._offset_map).
            offs: list[int] = []
            acc = 0
            for _v in variables:
                offs.append(acc)
                acc += _v.size
            n_flat = _flat_size(model)

            # Probe once at the box midpoint so an uncovered atom (unbounded/NaN
            # gradient) is caught HERE and we fall back to JAX, rather than silently
            # emitting no cut deep in the tree.
            probe = _np.array(
                [0.5 * (float(_np.ravel(v.lb)[0]) + float(_np.ravel(v.ub)[0])) for v in variables],
                dtype=_np.float64,
            )

            def _box_at(xv):
                xa = _np.asarray(xv, dtype=_np.float64).ravel()
                box = {}
                for _v, off in zip(variables, offs):
                    sl = xa[off : off + _v.size]
                    box[_v] = _Ivl(sl.astype(_np.float64), sl.astype(_np.float64))
                return box

            def _eval(xv):
                ad = interval_hessian(expr, model, _box_at(xv))
                val = float(_np.asarray(ad.value.lo).ravel()[0])
                g = _np.asarray(ad.grad.lo, dtype=_np.float64).ravel()
                return val, g

            pv, pg = _eval(probe)
            if not (_np.isfinite(pv) and pg.size == n_flat and _np.all(_np.isfinite(pg))):
                return None  # atom not covered → JAX fallback

            # One shared eval per (value_fn, grad_fn) pair: _separate_convex calls
            # value_fn(xv) then grad_fn(xv) with the SAME xv each round. Key the
            # 1-entry memo on the point's CONTENT (not id(xv) — object ids are
            # recycled after GC, so a later round could collide with a freed xv and
            # return a stale point's (value, grad)).
            _memo: dict = {}

            def _shared(xv):
                k = _np.asarray(xv, dtype=_np.float64).tobytes()
                hit = _memo.get(k)
                if hit is None:
                    hit = _eval(xv)
                    _memo.clear()
                    _memo[k] = hit
                return hit

            def value_fn(xv):
                return _shared(xv)[0]

            def grad_fn(xv):
                return _shared(xv)[1]

            return (value_fn, grad_fn)
        except Exception:
            return None

    def _support_box_key(self, node: CNode) -> tuple:
        """Box-dependent cache key ``(id(node), lb.bytes, ub.bytes)`` restricted to
        the node's support columns (full box for opaque-hiding subtrees)."""
        sc = self._analysis.support_cols
        nid = id(node)
        cols = sc.get(nid, _UNSET)
        if cols is _UNSET:
            cols = _node_support_cols(node)
            sc[nid] = cols
        if cols is None:
            lb, ub = self._flat_lb, self._flat_ub
        elif cols:
            idx = np.asarray(cols, dtype=np.intp)
            lb, ub = self._flat_lb[idx], self._flat_ub[idx]
        else:  # no variables (e.g. a constant node): box-independent enclosure
            lb = ub = _EMPTY_BOX
        return (nid, lb.tobytes(), ub.tobytes())

    def _curvature_cert(
        self,
        node: CNode,
        idxs: list[int],
        flat_lb: np.ndarray,
        flat_ub: np.ndarray,
        cbox: dict,
    ) -> Optional[str]:
        """Interval-Hessian curvature verdict of ``node`` over the box, with monotone
        inheritance (see the module note). Matches ``_multivar_box_curvature`` exactly
        on a fresh cache; on a hit it returns a proven super-box's verdict (byte-
        identical by inclusion-monotonicity) or skips re-proving an abstained box that
        is not meaningfully smaller.
        """
        nid = id(node)
        lo = self._flat_lb[np.asarray(idxs, dtype=np.intp)]
        hi = self._flat_ub[np.asarray(idxs, dtype=np.intp)]
        # 1) Proven-box inheritance: convex/concave on a super-box holds (and re-proves
        #    with the same verdict) on every sub-box — byte-identical to a fresh call.
        for plo, phi, proven in self._analysis.hessian_certs.get(nid, ()):
            if _box_subset(lo, hi, plo, phi):
                return proven
        # 2) Abstained-box shortcut: skip re-proving when the query is a subset that is
        #    not meaningfully smaller (every width >= 0.5x the abstained width). Curvature
        #    can resolve on a >=2x-smaller box, so those are re-proven; abstaining longer
        #    is sound (only looser).
        for alo, ahi in self._analysis.hessian_abstain.get(nid, ()):
            if _box_subset(lo, hi, alo, ahi) and bool(np.all((hi - lo) >= 0.5 * (ahi - alo))):
                return None
        # 3) Prove afresh and record the outcome (support-restricted, capped).
        from discopt._jax.milp_relaxation import _multivar_box_curvature

        verdict = cast(
            Optional[str],
            _multivar_box_curvature(self._expr(node), self.model, idxs, flat_lb, flat_ub, cbox),
        )
        if verdict == "convex" or verdict == "concave":
            certs = self._analysis.hessian_certs.setdefault(nid, [])
            certs.append((lo, hi, verdict))
            if len(certs) > _CERT_LIST_CAP:
                del certs[0]
        else:
            abstained = self._analysis.hessian_abstain.setdefault(nid, [])
            abstained.append((lo, hi))
            if len(abstained) > _CERT_LIST_CAP:
                del abstained[0]
        return verdict

    # -- interval enclosure of a node over the box (AVM bound propagation) --- #
    def bounds(self, node: CNode) -> tuple[float, float]:
        cached = self._bounds.get(id(node))
        if cached is not None:
            return cached
        # Cross-build cache (issue #632 EP1): keyed by (id(node), support-box bytes),
        # so branching on one variable invalidates only the enclosures of nodes that
        # depend on it. The reconstructed tree is pinned (``_expr``), which kills the
        # historical stale-``id()`` hazard the WARNING here used to describe:
        # ``evaluate_interval`` memoizes by ``id(expr)``, and a transient reconstruct
        # tree could be GC'd with its ``id()`` reused for a different node, so a shared
        # memo would return an UNSOUND, nondeterministic bound. With the pinned tree
        # ``id(expr)`` stays stable/live, and the box-keyed cache below is a pure
        # function of (node, box) — every hit is byte-identical to a fresh evaluation.
        box_cache = self._analysis.bounds_by_box
        key = self._support_box_key(node)
        hit = box_cache.get(key)
        if hit is not None:
            self._bounds[id(node)] = hit
            return hit
        enc = evaluate_interval(self._expr(node), self.model, self._ivbox)
        lo = float(np.asarray(enc.lo))
        hi = float(np.asarray(enc.hi))
        if not (math.isfinite(lo)):
            lo = -math.inf
        if not (math.isfinite(hi)):
            hi = math.inf
        result = (lo, hi)
        if len(box_cache) >= _BOUNDS_CACHE_CAP:
            box_cache.clear()
        box_cache[key] = result
        self._bounds[id(node)] = result
        return result

    # -- the recursive representation walk (bottom-up) ---------------------- #
    def rep(self, node: CNode) -> LinForm:
        cached = self._rep.get(id(node))
        if cached is not None:
            return cached
        out = self._try_convex_lift(node)
        if out is None:
            out = self._rep_impl(node)
        self._rep[id(node)] = out
        return out

    # -- composite convex/concave lift (issue #632 P2) ---------------------- #
    def _lift_eligible(self, node: CNode) -> bool:
        """Structural pre-filter: is ``node`` a multivariate nonlinear node the
        atom decomposition relaxes *loosely* (so a convex OA lift can only help)?

        Mirrors the deleted federation gate ``_should_claim_composite_multivar``:
        a composite univariate call (non-affine arg), a jointly-convex multivar
        intrinsic (``callN`` — centropy/…), a non-integer power of a non-affine
        base, or a convex/concave *sum* spanning >= 2 variables. NOT eligible: an
        affine-argument call/power (already exact via the 1-D envelope), an integer
        power (monomial/square path), a bare product (bilinear — McCormick), or an
        opaque node. The sound curvature certificate (``_try_convex_lift``) is the
        real gate; this only avoids lifting nodes the engine already handles tightly.
        """
        if node.kind in ("var", "const"):
            return False
        if len(var_support(node)) < 2:
            return False  # univariate composites are a separate class/stage
        kind = node.kind
        if kind == "sum":
            return not is_affine(node)
        if kind == "callN":
            return True
        if kind == "call":
            (child,) = node.children
            return not is_affine(child)
        if kind == "pow":
            p = float(node.payload)
            if p == int(p):
                return False  # integer power -> monomial/square owner
            (child,) = node.children
            return not is_affine(child)
        return False  # prod (bilinear) / opaque

    def _try_convex_lift(self, node: CNode) -> Optional[LinForm]:
        """Lift ``node`` to an aux and register a composite convex/concave spec if
        its curvature is CERTIFIED over the box; else return ``None``.

        Reuses the surviving, class-general detectors and the ``_separate_convex``
        spec shape (``CompositeMultivarRelaxation``) verbatim — this only *connects*
        them to the engine's decomposition. Registration is EXACT-only and sound by
        construction: the aux equals ``g`` over genuinely-original affine-free vars,
        curvature is proven (DCP or the interval-Hessian PSD certificate), and the
        supporting tangent ``_separate_convex`` adds is a global under-/over-estimator
        that never cuts a feasible point.
        """
        if not self._lift_eligible(node):
            return None
        try:
            from discopt._jax.convexity import Curvature
            from discopt._jax.milp_relaxation import (
                _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE,
                CompositeMultivarRelaxation,
                _build_convexity_box,
            )
        except Exception:
            return None

        idxs = sorted(int(j) for j in var_support(node))
        if len(idxs) < 2 or any(j >= self.n_orig for j in idxs):
            return None
        flat_lb = np.asarray(self.col_lb[: self.n_orig], dtype=np.float64)
        flat_ub = np.asarray(self.col_ub[: self.n_orig], dtype=np.float64)
        lo = flat_lb[idxs]
        hi = flat_ub[idxs]
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) or np.any(hi < lo):
            return None

        cbox = _build_convexity_box(self.model, flat_lb, flat_ub)
        # Certify curvature: global DCP first (box-independent, cached), else the
        # sound box-restricted interval-Hessian PSD/NSD certificate (monotone-inherited
        # across sub-boxes, cached; abstains -> None).
        curv = self._dcp(node)
        if curv == Curvature.CONVEX:
            curvature = "convex"
        elif curv == Curvature.CONCAVE:
            curvature = "concave"
        else:
            box_curv = self._curvature_cert(node, idxs, flat_lb, flat_ub, cbox)
            if box_curv is None:
                return None
            curvature = box_curv

        # Sound interval enclosure of g over the box -> finite aux column bounds.
        col_lo, col_hi = self.bounds(node)
        if not (math.isfinite(col_lo) and math.isfinite(col_hi)) or col_hi < col_lo:
            return None
        # Conditioning guard (#358): a wide/ill-conditioned node whose value range
        # (hence gradient / cut coefficients) explodes fools the fast simplex into a
        # garbage bound. Abstain above the magnitude the cut builders abstain at; the
        # node then stays on the sound term-by-term decomposition path.
        if max(abs(col_lo), abs(col_hi)) > _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE:
            return None

        # Compiled value/gradient (the function of x, box-independent, cached).
        compiled = self._compiled(node)
        if compiled is None:
            return None
        f, grad_f = compiled

        # Committed to lifting. Build the node's ordinary atom decomposition FIRST
        # (all its envelope rows + product-side varmap registrations are kept), then
        # tie a single aux ``w = g`` to it so ``_separate_convex`` can add outer-
        # approximation tangents. Because the decomposition rows stay and the OA
        # cuts only ADD constraints, the lifted relaxation is at-least-as-tight as
        # the decomposition on EVERY node — the OA can tighten, never loosen (this
        # is what makes the composite-convex lift non-regressing, unlike a lift that
        # *replaces* the decomposition with the crude interval floor).
        dec = self._rep_impl(node)
        single = None
        if dec.const == 0.0:
            items = [(j, c) for j, c in dec.coeffs.items() if c != 0.0]
            if len(items) == 1 and items[0][1] == 1.0 and items[0][0] >= self.n_orig:
                single = items[0][0]
        if single is not None:
            # The decomposition already collapsed to one bare aux (call/callN/pow) —
            # register OA on it directly (its aux_expr == g is set by _rep_impl).
            w = single
        else:
            # A sum (or scaled/affine) decomposition — bind a fresh aux ``w == dec``.
            # At the lifted true point every atom aux equals its exact value, so
            # ``dec`` evaluates to ``g(x)`` exactly and both equality rows hold with
            # zero slack (soundness preserved for the feasible-point sampler).
            w = self.new_aux(col_lo, col_hi)
            if self.track_aux_exprs:
                self.aux_expr[w] = self._expr(node)
            le = {w: 1.0}
            ge = {w: -1.0}
            for j, c in dec.coeffs.items():
                le[j] = le.get(j, 0.0) - c
                ge[j] = ge.get(j, 0.0) + c
            self.add_row(le, dec.const)
            self.add_row(ge, -dec.const)
        self.composite_multivar_specs.append(
            CompositeMultivarRelaxation(
                expr_id=id(node),
                aux_col=w,
                curvature=curvature,
                lower_lines=(),
                upper_lines=(),
                idxs=tuple(idxs),
                value_fn=f,
                grad_fn=grad_f,
            )
        )
        self.coverage[id(node)] = ("composite_convex", True)
        return LinForm.col(w)

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
            self.aux_expr[w] = self._expr(node)
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


def _odd_power_tangent_point(p: int, endpoint: float, lo: float, hi: float) -> Optional[float]:
    """Tangent point of the line through ``(endpoint, endpoint**p)`` touching ``t**p``.

    Root of ``h(t) = (p-1) t^p - p·endpoint·t^(p-1) + endpoint**p`` on the OPEN side
    opposite the endpoint's sign (for the convex-envelope lower facet the endpoint is
    ``lo<0`` and the tangency is in ``(0, hi]``; the concave upper facet mirrors it in
    ``[lo, 0)``). Returns the root by bisection, or ``None`` if the tangency lies
    outside the box (=> the secant is the exact facet instead).
    """

    def h(t: float) -> float:
        return (p - 1) * t**p - p * endpoint * t ** (p - 1) + endpoint**p

    if endpoint < 0.0:  # lower facet: search (0, hi]
        a, b = 1e-12 * max(hi, 1.0), hi
    else:  # upper facet: search [lo, 0)
        a, b = lo, -1e-12 * max(-lo, 1.0)
    ha, hb = h(a), h(b)
    if ha == 0.0:
        return a
    if hb == 0.0:
        return b
    if (ha > 0.0) == (hb > 0.0):
        return None  # no sign change -> tangency outside box, secant is the facet
    for _ in range(80):
        m = 0.5 * (a + b)
        hm = h(m)
        if hm == 0.0:
            return m
        if (hm > 0.0) == (ha > 0.0):
            a, ha = m, hm
        else:
            b = m
    return 0.5 * (a + b)


def _emit_odd_power_hull(ctx: _Builder, w: int, lt: LinForm, lo: float, hi: float, p: int) -> bool:
    """Exact two-piece convex/concave hull of ``w = t**p`` (odd ``p>=3``) over a
    SIGN-STRADDLING box ``lo<0<hi`` — the case ``_pow_curv`` abstains on (S-shaped,
    neither convex nor concave), where the builder otherwise keeps only the interval
    floor. Two facets (Liberti & Pantelides 2003):

    * underestimator = the tangent line from ``(lo, lo^p)`` (or the secant ``lo->hi``
      if the tangency exceeds ``hi``) — a valid convex-envelope lower facet;
    * overestimator = the mirror tangent line from ``(hi, hi^p)`` (or the secant).

    Each line touches the graph and lies on the correct side everywhere on the box
    (the convex/concave-envelope construction), so no feasible ``(t, w=t^p)`` is cut.
    """
    if p < 3 or p % 2 == 0 or not (lo < 0.0 < hi) or not _finite(lo, hi):
        return False

    def _line(m: float, t0: float, ft0: float, under: bool) -> None:
        # under: w >= ft0 + m(t - t0);  over: w <= ft0 + m(t - t0).  t = lt.
        sign = -1.0 if under else 1.0  # sign*w on the LHS
        coeffs = {w: sign}
        for j, c in lt.scaled(-sign * m).coeffs.items():
            coeffs[j] = coeffs.get(j, 0.0) + c
        # sign*w - sign*m*(cols) <= sign*(m*t0 - ft0 + m*lt.const)*(-1)^...:
        rhs = sign * (ft0 - m * t0 + m * lt.const)
        ctx.add_row(coeffs, rhs)

    flo, fhi = lo**p, hi**p
    # Underestimator from lo.
    t_u = _odd_power_tangent_point(p, lo, lo, hi)
    if t_u is not None and 0.0 < t_u <= hi:
        m_u = p * t_u ** (p - 1)
        _line(m_u, lo, flo, under=True)
    else:
        m_u = (fhi - flo) / (hi - lo)
        _line(m_u, lo, flo, under=True)
    # Overestimator from hi (mirror).
    t_o = _odd_power_tangent_point(p, hi, lo, hi)
    if t_o is not None and lo <= t_o < 0.0:
        m_o = p * t_o ** (p - 1)
        _line(m_o, hi, fhi, under=False)
    else:
        m_o = (fhi - flo) / (hi - lo)
        _line(m_o, hi, fhi, under=False)
    return True


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
    # Sign-straddling odd power: _pow_curv abstains (S-shaped) -> two-piece hull.
    if curv is None and float(p).is_integer():
        if _emit_odd_power_hull(ctx, w, lt, lo, hi, int(p)):
            tight = True
    # A power of a positive product ``(∏ xⱼ)**p`` also admits the (tighter) log-space
    # signomial band directly over the original factors (additive, sound).
    if _emit_logspace_band(ctx, w, node):
        tight = True
    # Register a bare-original integer power ``x_i**p`` (p>=2) so the moment/PSD,
    # edge-concave and univariate-square separators see the lifted square/monomial.
    i = ctx.single_orig_col(lt)
    if i is not None and float(p).is_integer() and int(p) >= 2:
        ctx.register_power(i, int(p), w)
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
    base_expr = ctx._expr(base) if ctx.track_aux_exprs else None
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
    # A bare-original integer power factor ``x_i**e`` is itself a lifted monomial.
    bi = ctx.single_orig_col(lb_base)
    if bi is not None and float(e).is_integer() and int(e) >= 2:
        ctx.register_power(bi, int(e), aux)
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


def _relax_var_power(ctx: _Builder, col: int, n: int) -> tuple[LinForm, tuple[float, float]]:
    """Relax ``x_col**n`` (``n`` integer >= 2) into a fresh aux with the tight
    secant/tangent power envelope, register it as monomial ``(col, n)`` and return
    its ``(LinForm, bounds)``.
    """
    lo0, hi0 = ctx.col_lb[col], ctx.col_ub[col]
    lo, hi = _pow_bounds(lo0, hi0, float(n))
    aux = ctx.new_aux(lo, hi)
    base_lin = LinForm.col(col)
    curv = _pow_curv(float(n), lo0, hi0)
    f = lambda t: float(t) ** n  # noqa: E731
    fp = lambda t: n * (float(t) ** (n - 1.0))  # noqa: E731
    _emit_1d(ctx, aux, base_lin, lo0, hi0, f, fp, curv)
    if ctx.track_aux_exprs:
        from discopt._jax.canonical_expr import _flat_var_expr

        ctx.aux_expr[aux] = _flat_var_expr(ctx.model)[col] ** n
    ctx.register_power(col, n, aux)
    return LinForm.col(aux), (lo, hi)


def _emit_scaled_equality(ctx: _Builder, w: int, lin: LinForm, scalar: float) -> None:
    """Emit ``w == scalar * lin`` as two sound ``<=`` rows (exact link, no relaxation)."""
    scaled = lin.scaled(scalar)
    # w - scalar*lin <= 0
    c1 = {w: 1.0}
    for j, c in scaled.coeffs.items():
        c1[j] = c1.get(j, 0.0) - c
    ctx.add_row(c1, scaled.const)
    # -(w - scalar*lin) <= 0
    c2 = {w: -1.0}
    for j, c in scaled.coeffs.items():
        c2[j] = c2.get(j, 0.0) + c
    ctx.add_row(c2, -scaled.const)


def _build_product(ctx: _Builder, node: CNode, w: int) -> Envelope:
    """``w = prod_i base_i**e_i`` (all exponents positive => product/monomial).

    A product is first *canonicalized into a monomial*: every factor that is an
    affine multiple of a single original ``(c x_i)`` raised to an integer power is
    aggregated by variable into ``scalar · prod_i x_i**n_i`` (so the disguised
    univariate square ``(2 x_i)·x_i`` becomes ``2 x_i**2``, not a wide-box
    bilinear). Each aggregated ``x_i**n_i`` (n_i>=2) is relaxed by its *tight*
    secant/tangent power envelope (and registered as a monomial so the
    univariate-square / edge-concave / PSD separators fire); the ``n_i==1``
    variables and any remaining general factors are McCormick-folded pairwise —
    sound at every step. The scalar links to ``w`` by an exact equality. Without
    this aggregation, pairwise McCormick of ``(c x_i)·x_i`` over a wide box is
    hopelessly loose (st_miqp2: -221 vs -13.06).
    """
    (exps,) = node.payload
    scalar = 1.0
    var_exp: dict[int, int] = {}
    general: list[tuple[CNode, float]] = []
    for child, e in zip(node.children, exps):
        ef = float(e)
        lt = ctx.rep(child)
        sv = ctx.single_var_affine(lt) if ef.is_integer() and int(ef) >= 1 else None
        if sv is not None:
            col, coef = sv
            scalar *= coef ** int(ef)
            var_exp[col] = var_exp.get(col, 0) + int(ef)
        else:
            general.append((child, ef))

    # Log-space signomial band (additive, sound): tighter than McCormick for a wide
    # positive product / any fractional exponent. Intersects with the McCormick fold.
    logspace = _emit_logspace_band(ctx, w, node)

    # Nothing aggregated (no single-variable integer factor) => the original
    # pairwise-McCormick path is exactly correct; keep it (also the sole path when
    # every factor is a general multi-variable / nonlinear form).
    if not var_exp:
        factors = [_factor_value(ctx, ch, e) for ch, e in general]
        tight = _fold_product(ctx, w, factors)
        return Envelope(rows=[], tight=tight or logspace)

    # Assemble the relaxed factor values (LinForm, bounds, tracking-expr): tight
    # power envelopes for aggregated x_i**n_i, the bare variable for n_i==1, general
    # factors as-is. The tracking-expr (validation only) is the EXACT value the
    # factor represents, so the soundness harness can lift the true point.
    flat_expr = None
    if ctx.track_aux_exprs:
        from discopt._jax.canonical_expr import _flat_var_expr

        flat_expr = _flat_var_expr(ctx.model)
    factor_vals: list[tuple[LinForm, tuple[float, float], object]] = []
    for col, n in var_exp.items():
        fe = (flat_expr[col] ** n if n >= 2 else flat_expr[col]) if flat_expr is not None else None
        if n >= 2:
            lin, bb = _relax_var_power(ctx, col, n)
            factor_vals.append((lin, bb, fe))
        else:  # n == 1
            factor_vals.append((LinForm.col(col), (ctx.col_lb[col], ctx.col_ub[col]), fe))
    for child, e in general:
        factor_vals.append(_factor_value(ctx, child, e))

    if len(factor_vals) == 1:
        # w = scalar * (single relaxed factor). Exact link; the factor already
        # carries its own (tight) envelope.
        _emit_scaled_equality(ctx, w, factor_vals[0][0], scalar)
        return Envelope(rows=[], tight=True)

    # Fold the scalar into the first factor (associativity: scalar·f0·f1·… =
    # (scalar·f0)·f1·…), scaling its LinForm and bounds, so no temp aux is needed
    # and ``w`` remains exactly the product value.
    if scalar != 1.0:
        lin0, (b0lo, b0hi), fe0 = factor_vals[0]
        nb = (scalar * b0lo, scalar * b0hi) if scalar >= 0 else (scalar * b0hi, scalar * b0lo)
        # fe0 is the validation-only tracking Expression (typed ``object``); scaling
        # it by the folded scalar is runtime-valid via Expression.__rmul__.
        scaled_fe0 = (scalar * fe0) if fe0 is not None else None  # type: ignore[operator]
        factor_vals[0] = (lin0.scaled(scalar), nb, scaled_fe0)
    tight = _fold_product(ctx, w, factor_vals)
    return Envelope(rows=[], tight=tight)


def _emit_logspace_band(ctx: _Builder, w: int, node: CNode) -> bool:
    """Additive log-space envelope for a positive signomial ``w = coef·∏ xᵢ^{aᵢ}``.

    For a product/power/ratio of STRICTLY-POSITIVE original variables the tightest
    factorable relaxation is not recursive McCormick (loose on wide boxes / undefined
    for fractional exponents) but the log-space lift (Tawarmalani & Sahinidis; the
    surviving H-LOG construction). Emitted here as ADDITIONAL rows on ``w`` alongside
    the McCormick fold, so the relaxation is the intersection of the two — sound, and
    at-least-as-tight as McCormick alone:

    * ``zᵢ = ln xᵢ`` — concave, exact secant/tangent band binding each ``zᵢ`` to
      ``xᵢ`` (``_emit_1d``);
    * ``s = Σ aᵢ zᵢ`` — an exact linear equality (fresh column);
    * ``w = coef·exp(s)`` — convex (``coef>0``) / concave (``coef<0``), exact
      secant/tangent band binding ``w`` to ``s``.

    Every row touches the graph, so no feasible ``(x, w=∏xᵢ^{aᵢ})`` point is cut.
    Only fires for a genuinely-loose case (>=3 factors, or any non-integer / negative
    exponent — a single bilinear/trilinear-integer product is already at its McCormick
    hull, so log-space would only add redundant columns). Returns ``True`` if emitted.
    """
    from discopt._jax.milp_relaxation import _extract_positive_product

    n_orig = ctx.n_orig
    flat_lb = np.asarray(ctx.col_lb[:n_orig], dtype=np.float64)
    flat_ub = np.asarray(ctx.col_ub[:n_orig], dtype=np.float64)
    expr = ctx._expr(node)
    pp = _extract_positive_product(expr, ctx.model, n_orig, flat_lb, flat_ub)
    if pp is None:
        return False
    coef, factors = pp
    if len(factors) < 2 or coef == 0.0:
        return False
    items = sorted(factors.items())
    non_trivial = len(items) >= 3 or any(not float(a).is_integer() or a < 0 for _i, a in items)
    if not non_trivial:
        return False  # pure low-arity positive integer product -> McCormick is exact

    flat_expr = None
    if ctx.track_aux_exprs:
        from discopt._jax.canonical_expr import _flat_var_expr

        flat_expr = _flat_var_expr(ctx.model)

    z_cols: list[int] = []
    a_list: list[float] = []
    s_lo = 0.0
    s_hi = 0.0
    for i, a in items:
        lo_i, hi_i = float(ctx.col_lb[i]), float(ctx.col_ub[i])
        if not (lo_i > 1e-12 and math.isfinite(hi_i) and hi_i > lo_i * (1.0 + 1e-15)):
            return False  # strict positivity + finite width required for the ln band
        af = float(a)
        zlo, zhi = math.log(lo_i), math.log(hi_i)
        z = ctx.new_aux(zlo, zhi)
        if flat_expr is not None:
            from discopt import modeling as _dm

            ctx.aux_expr[z] = _dm.log(flat_expr[i])
        _emit_1d(ctx, z, LinForm.col(i), lo_i, hi_i, math.log, lambda t: 1.0 / t, "concave")
        z_cols.append(z)
        a_list.append(af)
        s_lo += af * (zlo if af > 0 else zhi)
        s_hi += af * (zhi if af > 0 else zlo)

    # s == Σ aᵢ zᵢ (exact linear equality via two <= rows).
    s = ctx.new_aux(s_lo, s_hi)
    if flat_expr is not None:
        from discopt import modeling as _dm

        s_expr = None
        for z_i, af in zip(z_cols, a_list):
            # aux_expr entries are validation-only tracking Expressions (typed object);
            # scaling/adding is runtime-valid via Expression.__rmul__/__add__.
            term = af * ctx.aux_expr[z_i]  # type: ignore[operator]
            s_expr = term if s_expr is None else s_expr + term
        ctx.aux_expr[s] = s_expr
    le = {s: 1.0}
    ge = {s: -1.0}
    for z_i, af in zip(z_cols, a_list):
        le[z_i] = le.get(z_i, 0.0) - af
        ge[z_i] = ge.get(z_i, 0.0) + af
    ctx.add_row(le, 0.0)
    ctx.add_row(ge, 0.0)

    # w == coef·exp(s): convex band if coef>0, concave if coef<0.
    def f(t: float) -> float:
        return coef * math.exp(float(t))

    _emit_1d(ctx, w, LinForm.col(s), s_lo, s_hi, f, f, "convex" if coef > 0 else "concave")
    return True


def _build_ratio(ctx: _Builder, node: CNode, w: int) -> Envelope:  # noqa: D401
    """``w = prod_i base_i**e_i`` with a negative exponent (division).

    Same recursive McCormick fold as ``product`` — a negative-exponent factor
    ``base**(-k)`` is relaxed by :func:`_factor_value` as a 1-D power atom (convex
    reciprocal on a sign-definite base), then McCormick-combined. Sign-indefinite
    denominator => that factor's bounds are unbounded => its McCormick rows are
    skipped and the aux interval floor stands (sound; no finite bound, which is an
    orthogonal bound-validity concern, not a soundness one).
    """
    logspace = _emit_logspace_band(ctx, w, node)
    (exps,) = node.payload
    factors = [_factor_value(ctx, ch, float(e)) for ch, e in zip(node.children, exps)]
    tight = _fold_product(ctx, w, factors)
    return Envelope(rows=[], tight=(tight and len(factors) == 2) or logspace)


def _fold_product(ctx: _Builder, w: int, factors: list) -> bool:
    if len(factors) < 2:
        # A product node always has >= 2 factors; guard defensively.
        return False
    acc_lin, acc_b, acc_expr = factors[0]
    # Track the ORIGINAL columns multiplied so far while every factor folded to
    # this point is a bare original ``x_i`` (exponent 1); the moment ``acc_cols``
    # goes ``None`` (a scaled/aux/power factor entered) we can no longer name the
    # partial product as a pure multilinear of originals, so no further product is
    # registered on this chain.
    c0 = ctx.single_orig_col(acc_lin)
    acc_cols: Optional[list[int]] = [c0] if c0 is not None else None
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
        fcol = ctx.single_orig_col(fl)
        if acc_cols is not None and fcol is not None and fcol not in acc_cols:
            acc_cols = acc_cols + [fcol]
            # ``target`` == prod(x_c for c in acc_cols) exactly, tied to those
            # originals by the McCormick chain — sound to register.
            ctx.register_product(acc_cols, target)
        else:
            acc_cols = None
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

    # NOTE (roadmap P3): branch-and-reduce / box tightening (FBBT/OBBT) is owned by
    # the separate branch-and-reduce workstream and arrives HERE via the ``box``
    # (node-box) interface — a tighter box automatically yields uniformly tighter
    # envelopes below (McCormick/secant/tangent are monotone in the box). This layer
    # does not tighten the box itself.

    # Box-independent analysis, computed once per model and pinned on it (issue #632
    # EP1): the canonical DAG (which pins CNodes -> stable id() keys), reconstructed
    # expressions, DCP verdicts, compiled value/grad fns, and curvature certificates.
    analysis = _get_analysis_cache(model)
    dag: CanonicalDAG = analysis.dag
    ctx = _Builder(model, flat_lb, flat_ub, analysis)

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

    # Objective-bound validity (soundness guard). The LP optimum is a valid lower
    # bound ONLY if the relaxed objective is actually bounded below over the box.
    # When a nonlinear atom cannot be enveloped (unbounded box: McCormick rows
    # dropped for infinite endpoints, transcendental over an infinite argument,
    # …) its aux column is a free interval-floor column; if such a column carries
    # objective cost and is otherwise unconstrained, the LP is unbounded below —
    # yet the warm-started Rust simplex can mis-report a finite "optimal" (issue
    # #15). Reporting that as a bound would be a FALSE certificate. We therefore
    # compute a SOUND box-interval lower bound on the (minimize-equivalent)
    # objective from the column bounds; a cost column that is unbounded on its
    # cost-relevant side and appears in NO row makes that lower bound -inf, and we
    # refuse the objective bound (the solver falls back to its rigorous
    # alphaBB/interval bound) rather than trust a possibly-fabricated LP value.
    # This mirrors the federation's ``objective_bound_valid=False`` behaviour on
    # an un-relaxable / under-constrained objective.
    _row_cols: set[int] = set()
    for _coeffs, _ in ctx.rows:
        _row_cols.update(_coeffs)
    obj_bound_valid = True
    obj_box_lb = obj_offset
    for j, coef in obj_lin.coeffs.items():
        edge = ctx.col_lb[j] if coef > 0 else ctx.col_ub[j]
        contrib = coef * edge
        if not math.isfinite(contrib):
            # Unbounded on the cost-relevant side. If the column is tied down by a
            # row it MAY still be LP-bounded (a correct simplex would report the
            # true bound or unboundedness); but a free unconstrained cost column is
            # provably unbounded -> refuse.
            if j not in _row_cols:
                obj_bound_valid = False
            obj_box_lb = -math.inf
        else:
            obj_box_lb += contrib
    if not math.isfinite(obj_box_lb):
        # No finite sound floor on the objective at all -> the LP value cannot be
        # certified as a global lower bound.
        obj_bound_valid = False

    # Pure LP relaxation (integrality relaxed at the root) — a sound lower bound,
    # matching the federation's root-node LP convention.
    milp = MilpRelaxationModel(
        c=c,
        A_ub=A_ub,
        b_ub=b,
        bounds=bounds,
        obj_offset=obj_offset,
        integrality=None,
        objective_bound_valid=obj_bound_valid,
    )
    return UniformRelaxation(
        model=milp,
        n_orig=ctx.n_orig,
        n_aux=n_cols - ctx.n_orig,
        obj_sense_sign=sign,
        obj_offset=obj_offset,
        coverage=dict(ctx.coverage),
        bilinear_map=dict(ctx.bilinear_map),
        monomial_map=dict(ctx.monomial_map),
        trilinear_map=dict(ctx.trilinear_map),
        multilinear_map=dict(ctx.multilinear_map),
        univariate_square_map=dict(ctx.univariate_square_map),
        composite_multivar_specs=list(ctx.composite_multivar_specs),
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
