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
import os
import time
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
from discopt._jax.milp_relaxation import (
    MilpRelaxationModel,
    _expression_lower_bound_for_lift,
    _flat_variable_types,
    _integer_domain_values,
    _linearize_affine_expr,
)
from discopt._jax.model_utils import flat_variable_bounds
from discopt.modeling.core import Model, ObjectiveSense, UnaryOp

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
# Max distinct integer values a finite-domain trig-square selector table may
# enumerate (issue #640 Bucket 1); larger domains keep the loose double-lift.
_MAX_FINITE_DOMAIN_TRIG_TABLE_VALUES = 256


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
    # Optional multiplier applied to the aux column when it becomes the node's
    # representation: ``rep(node) = rep_scale · col(w)``. Lets a builder make ``w``
    # the EXACT product/atom (so the RLT/PSD separators and cut-inheritance identities
    # see one clean lifted column) while a leading scalar rides in the rep, avoiding a
    # second ``w == scalar·w_pure`` binding column (issue #640 Bucket 2/4).
    rep_scale: float = 1.0


@dataclasses.dataclass
class FiniteDomainTrigSquareTable:
    """Exact selector table for ``sin(int-affine)**2`` / ``cos(int-affine)**2``.

    Issue #640 Bucket 1 (recovered). When the trig argument is affine in a SINGLE
    integer/binary variable ``x_i`` over a small finite domain, the square takes
    finitely many EXACT values; a one-hot selector ``λ_v`` per domain value
    (``Σλ_v = 1``, ``λ_v ∈ {0,1}``) with the exact equality links ``x_i = Σ v·λ_v``,
    ``base = Σ trig(v)·λ_v``, ``sq = Σ trig(v)²·λ_v`` reproduces the square exactly
    at every integer point (vs. the loose double-lift ``(trig-envelope)²``). Sound:
    the links are exact equalities at the finitely many feasible integer points,
    and their LP relaxation (continuous ``λ``) is the convex hull of those points —
    a valid relaxation.
    """

    func_name: str
    var_idx: int
    domain_values: list[int]
    trig_values: list[float]
    square_values: list[float]
    selector_cols: list[int]
    base_col: int
    square_col: int


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
    # Affine squares ``(c·x_j+d)**2`` -> ``(var, aux) -> (coeff, const)`` (issue #640
    # Bucket 3), consumed by the incremental McCormick patch to regenerate their
    # box-dependent envelope rows in closed form.
    affine_square_map: dict = dataclasses.field(default_factory=dict)
    # Ratio-of-products lifts ``((num_cols...), (den_cols...)) -> aux`` over bare
    # originals with integer exponents (issue #309); consumed by the integer-ratio
    # partition bound to address the quotient column of its piece LPs.
    ratio_map: dict = dataclasses.field(default_factory=dict)
    # Finite-domain trig-square selector tables (issue #640 Bucket 1). Consumed by
    # the delegate to populate the ``finite_domain_trig_square_tables`` varmap family.
    finite_domain_trig_square_tables: list = dataclasses.field(default_factory=list)
    # Per-column integrality (0/1) over ALL columns (orig ∪ aux). Original-variable
    # integrality is applied by the delegate; this carries the ENGINE-created integer
    # aux (e.g. the trig-square selector binaries) so the delegate marks them too.
    integrality: list = dataclasses.field(default_factory=list)
    # Univariate intrinsic atoms ``w = fname(coeff*x_i + const)`` whose argument is
    # affine in ONE original variable — entries ``(fname, w, var_idx, coeff, const)``.
    # Exposed so the native-kernel producer (#764) can emit the box-independent term
    # descriptor (e.g. ``sqrt``) whose envelope the Rust kernel regenerates per box.
    univariate_atom_specs: list = dataclasses.field(default_factory=list)


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


def clear_analysis_cache(model: Model) -> None:
    """Drop the box-independent relaxation analysis cache pinned on ``model``.

    The cache (canonical DAG, reconstructed exprs, DCP verdicts, compiled fns,
    and interval enclosures in ``bounds_by_box``) is keyed by ``_analysis_token``
    — ``(len(_variables), len(_constraints), id(_objective))`` — which guards
    *structural* changes only and intentionally does NOT hash parameter *values*
    (they can be large arrays; hashing them on the per-node cache lookup would be
    a hot-path cost). Parameters are constant within a solve, so across the B&B
    nodes of one solve the token is stable and the cache is correctly reused (the
    intended win). But changing a :class:`~discopt.modeling.core.Parameter`'s
    ``.value`` *between* solves — the supported use of parameters — leaves the
    token unchanged while the cached interval bounds still embed the OLD value,
    so the stale relaxation would be reused on the spatial B&B path and yield an
    incorrect (unsound) bound on the re-solve (issue #742: NBI's per-weight
    subproblems change the CHIM-target parameters between solves and stalled at
    the dominated ``t = 0`` CHIM point). ``solve_model`` calls this at the start
    of every solve, in lockstep with the convexity-classification reset, so each
    solve rebuilds the analysis against the current parameter values.
    """
    model.__dict__.pop(_ANALYSIS_ATTR, None)


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
        skip_convex_lift: bool = False,
    ):
        self.model = model
        # Skip the composite convex/concave OA lift (#640 Bucket 3): it lifts the
        # whole convex node to a box-dependent epigraph aux the incremental McCormick
        # patch cannot regenerate. Skipping it keeps the objective as the plain sum
        # of its atom columns (same base-LP bound; only the lazy Kelley tangents are
        # forgone), so the fast-path relaxation stays patchable and sound.
        self.skip_convex_lift = bool(skip_convex_lift)
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
        # Affine squares ``(c·x_j + d)**2`` -> aux (issue #640 Bucket 3): keyed
        # ``(var, aux) -> (coeff, const)`` so the incremental McCormick patch can
        # regenerate the box-dependent envelope in closed form.
        self.affine_square_map: dict[tuple[int, int], tuple[float, float]] = {}
        # Ratio-of-products lifts ``w = (Π x_i) / (Π y_j)`` over BARE original
        # variables with integer exponents (issue #309): keyed
        # ``((num_cols_sorted...), (den_cols_sorted...)) -> aux`` with multiplicity
        # (``x**2`` contributes its column twice). Registration is EXACT — the aux
        # equals the named quotient, tied to the originals by the recursive
        # McCormick fold — so the integer-ratio partition bound can address the
        # quotient column directly. Metadata only; no rows are added here.
        self.ratio_map: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        # Composite convex/concave lifts (issue #632 P2): each certified-convex or
        # -concave multivariate nonlinear node the engine would otherwise decompose
        # loosely is lifted to a single aux and registered here so the existing
        # ``MccormickLPRelaxer._separate_convex`` Kelley loop adds its EXACT
        # supporting tangent at the LP point each round (outer-approximation cutting
        # planes, added lazily — not pre-seeded). A tangent of a convex function is
        # a global underestimator (concave: overestimator), so no feasible point is
        # ever cut; sound by construction. Populated by ``_try_convex_lift``.
        self.composite_multivar_specs: list = []
        # Finite-domain trig-square selector tables (issue #640 Bucket 1), populated
        # by ``_build_power`` when it hits a ``trig(int-affine)**2`` over a small
        # finite integer domain.
        self.finite_domain_trig_square_tables: list = []
        # Univariate intrinsic atoms whose argument is affine in ONE original
        # variable (``w = fname(coeff*x_i + const)``) — recorded by
        # ``_build_univariate_call`` so the piecewise-partition refinement (#640 S8
        # AMP ``disc_state`` recovery) can add per-interval secant/tangent envelopes
        # when ``x_i`` is partitioned. Entries: ``(fname, w, var_idx, coeff, const)``.
        self.univariate_atom_specs: list[tuple[str, int, int, float, float]] = []
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
        out = None if self.skip_convex_lift else self._try_convex_lift(node)
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
            # Accumulate into ONE dict rather than chaining ``acc = acc + rep(child)``:
            # ``LinForm.__add__`` copies the growing accumulator each time, so a sum of
            # N children (e.g. a 21k-term quadratic objective, qap) was O(N^2). Folding
            # in place is O(N + total nnz) and produces the byte-identical LinForm (same
            # per-column coefficient sum, zeros dropped), so the relaxation is unchanged.
            acc_coeffs: dict[int, float] = {}
            acc_const = float(const)
            for coef, child in zip(coeffs, node.children):
                s = float(coef)
                lf = self.rep(child)
                acc_const += lf.const * s
                for j, c in lf.coeffs.items():
                    acc_coeffs[j] = acc_coeffs.get(j, 0.0) + c * s
            return LinForm({j: c for j, c in acc_coeffs.items() if c != 0.0}, acc_const)
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
        rep = LinForm.col(w)
        return rep if env.rep_scale == 1.0 else rep.scaled(env.rep_scale)


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
# ATOM-REDUNDANCY-REVIEW (#632) — grep this tag to find the whole cluster.
#
# The gated DCP atoms below (entropy, log-sum-exp, norm, relative-entropy, xexp)
# each recognize a convex composite the *bare* factorable relaxation shatters. But
# the full solver also runs ``_try_convex_lift`` (default ON) + the ``_separate_convex``
# Kelley loop, which lifts any node whose interval-Hessian Gershgorin certificate
# proves it convex/concave — and that ALREADY closes several of these on a real solve.
# Measured node counts (full ``m.solve()``, atom ON vs OFF) on constructed interior-min
# instances:
#     log-sum-exp   103 -> 3   HELPS   (certifier abstains: wide interval Hessian)
#     xexp           33 -> 9   HELPS   (univariate: not eligible for the >=2-var lift)
#     norm            0 -> 0   REDUNDANT
#     relative-entr   0 -> 0   REDUNDANT
#     entropy         0 -> 0   REDUNDANT (as a >=2-var sum; the lift certifies it)
# So log-sum-exp and xexp earn their keep; norm/relent/entropy are (on every case
# reproducible here) redundant with the composite lift. They are kept gated-OFF and
# harmless pending a check against the real centropy ``ex6_2_*`` regression (#19),
# whose specific structure may hit a certifier-abstain case like log-sum-exp does.
# If that check comes back clean, DELETE the three REDUNDANT atoms (their detectors,
# emitters, ``_build_product``/``_build_univariate_call`` interceptions, flags, and
# tests) — each redundant site is tagged ``ATOM-REDUNDANCY-REVIEW: redundant`` below.
# --------------------------------------------------------------------------- #
# Log-sum-exp atom (issue #632 adjacent-atom family). ``log(sum_i exp(t_i))`` is
# CONVEX in the exp arguments ``t_i``, but the factorable path relaxes the outer
# ``log`` as a CONCAVE atom over the (separately-relaxed) exp-sum — the wrong
# curvature for the convex composite, so the underestimator (the bound side for a
# minimization) collapses to a loose floor (measured root gap ~2.7). Recognize the
# atom and emit its exact convex outer approximation: supporting-hyperplane
# (softmax-gradient) tangent cuts ``w >= LSE(t0) + sum_i softmax_i(t0)*(t_i - t0)``.
# ANY reference ``t0`` gives a sound global underestimator (tangent of a convex
# function), so a set at the box center + corners tightens toward the exact
# envelope; the aux interval bound (kept via ``tight=False``) closes the over side.
# Sound by construction. Gated ``DISCOPT_LOGSUMEXP_ATOM`` (default OFF -> the log
# path is byte-identical). Prototype scope: unit-weight ``log(sum exp(affine))``.
def _lse_terms(ctx: "_Builder", node: CNode):
    """If ``node`` is ``log(sum_i exp(t_i))`` (unit weights, no constant), return
    ``[(lt_i, lo_i, hi_i), ...]`` (each exp argument's LinForm + bounds); else None."""
    if node.kind != "call" or node.payload != "log":
        return None
    arg = node.children[0]
    if arg.kind != "sum":
        return None
    coeffs, const = arg.payload
    if not coeffs or not all(abs(float(c) - 1.0) < 1e-12 for c in coeffs):
        return None
    terms = []
    for ch in arg.children:
        if ch.kind != "call" or ch.payload != "exp":
            return None
        (targ,) = ch.children
        lo, hi = ctx.bounds(targ)
        terms.append((ctx.rep(targ), lo, hi))
    # A positive additive constant ``c`` folds in as a fixed term ``exp(log c)``,
    # so ``log(sum exp(t_i) + c)`` = LSE over the args plus ``log c`` (this is what
    # makes softplus ``log(1 + exp(x))`` = LSE(0, x) recognizable). A negative
    # constant can make the inner sum non-positive -> not a clean LSE, so bail.
    cst = float(const)
    if cst > 0.0:
        lc = math.log(cst)
        terms.append((LinForm.constant(lc), lc, lc))
    elif cst != 0.0:
        return None
    return terms if len(terms) >= 2 else None


def _lse_refs(terms: list) -> list:
    """Reference points in t-space for the softmax tangent set: box center + box
    corners (capped for many terms). Any ``t0`` yields a sound tangent."""
    n = len(terms)
    los = [lo for _, lo, _ in terms]
    his = [hi for _, _, hi in terms]
    refs = [[0.5 * (lo + hi) for _, lo, hi in terms]]
    if n <= 4:
        import itertools

        refs.extend(list(c) for c in itertools.product(*[(lo, hi) for _, lo, hi in terms]))
    else:
        for j in range(n):  # per-axis dominant corner: t_j at hi, rest at lo
            r = list(los)
            r[j] = his[j]
            refs.append(r)
    return refs


def _emit_lse(ctx: "_Builder", w: int, terms: list) -> bool:
    """Emit softmax-gradient tangent underestimators of ``w = log(sum_i exp(t_i))``.
    Returns True iff at least one cut was emitted (over side = the aux floor)."""
    if not all(_finite(lo, hi) for _, lo, hi in terms):
        return False
    lts = [lt for lt, _, _ in terms]
    emitted = False
    for t0 in _lse_refs(terms):
        e = [math.exp(v) for v in t0]
        z = math.fsum(e)
        if not math.isfinite(z) or z <= 0.0:
            continue
        s = [ei / z for ei in e]
        # w >= log(z) + sum_i s_i (t_i - t0_i); t_i = lt_i affine. Rearranged to
        # -w + sum_i s_i lt_i(x) <= sum_i s_i t0_i - log(z) - sum_i s_i lt_i.const.
        coeffs: dict[int, float] = {w: -1.0}
        b = -math.log(z)
        for si, lt, t0i in zip(s, lts, t0):
            for j, c in lt.coeffs.items():
                coeffs[j] = coeffs.get(j, 0.0) + si * c
            b += si * t0i - si * lt.const
        ctx.add_row(coeffs, b)
        emitted = True
    return emitted


# --------------------------------------------------------------------------- #
# ATOM-REDUNDANCY-REVIEW: redundant (norm — measured 0->0 nodes; see the cluster note above).
# Euclidean-norm atom (issue #632 adjacent-atom family). ``sqrt(sum_i t_i^2)`` is
# CONVEX, but the factorable path relaxes the outer ``sqrt`` as a CONCAVE atom over
# the square-sum — the wrong curvature, collapsing the underestimator to a loose
# floor. Recognize the atom and emit its convex OA: ``||t|| >= a . t`` for every
# unit vector ``a`` (Cauchy-Schwarz — the tangent of the norm at ``t0`` is the
# direction ``a = t0/||t0||``), plus the axis facets ``||t|| >= +/- t_i``. Sound for
# ANY direction. Gated ``DISCOPT_NORM_ATOM`` (default OFF -> the sqrt path is
# byte-identical). Prototype scope: ``sqrt(sum_i t_i^2 [+ c])``, ``c >= 0`` folding
# in as a constant coordinate ``sqrt(c)``.
def _norm_terms(ctx: "_Builder", node: CNode):
    """If ``node`` is ``sqrt(sum_i t_i^2 [+ c])`` (unit weights, ``c >= 0``), return
    ``[(lt_i, lo_i, hi_i), ...]`` for each squared base's LinForm + bounds; else
    ``None``."""
    if node.kind != "call" or node.payload != "sqrt":
        return None
    arg = node.children[0]
    if arg.kind != "sum":
        return None
    coeffs, const = arg.payload
    if not coeffs or not all(abs(float(c) - 1.0) < 1e-12 for c in coeffs):
        return None
    terms = []
    for ch in arg.children:
        if ch.kind != "pow" or float(ch.payload) != 2.0:
            return None
        (base,) = ch.children
        lo, hi = ctx.bounds(base)
        terms.append((ctx.rep(base), lo, hi))
    cst = float(const)
    if cst > 0.0:  # sqrt(sum t^2 + c) = ||(t, sqrt(c))||: a fixed coordinate.
        sc = math.sqrt(cst)
        terms.append((LinForm.constant(sc), sc, sc))
    elif cst != 0.0:
        return None
    return terms if terms else None


def _emit_norm(ctx: "_Builder", w: int, terms: list) -> bool:
    """Emit convex OA of ``w = ||t||``: axis facets ``w >= +/- t_i`` and direction
    cuts ``w >= a . t`` at box-corner directions. Returns True iff any cut emitted."""
    if not all(_finite(lo, hi) for _, lo, hi in terms):
        return False
    # ||t|| is nonnegative and box-bounded above by the farthest corner; pin the
    # aux interval so the objective floor is finite (the bound certifies) and sound.
    ctx.col_lb[w] = max(ctx.col_lb[w], 0.0)
    maxsq = math.fsum(max(abs(lo), abs(hi)) ** 2 for _, lo, hi in terms)
    ctx.col_ub[w] = min(ctx.col_ub[w], math.sqrt(maxsq))
    lts = [lt for lt, _, _ in terms]
    emitted = False
    # Axis facets: ||t|| >= |t_i|  ->  w >= t_i and w >= -t_i.
    for lt in lts:
        for sgn in (1.0, -1.0):
            coeffs: dict[int, float] = {w: -1.0}
            for j, c in lt.coeffs.items():
                coeffs[j] = coeffs.get(j, 0.0) + sgn * c
            ctx.add_row(coeffs, -sgn * lt.const)  # -w + sgn*t_i <= 0
            emitted = True
    # Direction cuts a . t <= ||t|| at the box-corner unit directions.
    for p in _lse_refs(terms):
        norm_p = math.sqrt(math.fsum(v * v for v in p))
        if norm_p < 1e-12:
            continue
        a = [v / norm_p for v in p]
        coeffs = {w: -1.0}
        b = 0.0
        for ai, lt in zip(a, lts):
            for j, c in lt.coeffs.items():
                coeffs[j] = coeffs.get(j, 0.0) + ai * c
            b += -ai * lt.const
        ctx.add_row(coeffs, b)
        emitted = True
    return emitted


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
    # Log-sum-exp atom (gated): emit the convex softmax-tangent OA instead of the
    # loose concave ``log(sum exp)`` relaxation. Off by default => byte-identical.
    if fname == "log" and os.environ.get("DISCOPT_LOGSUMEXP_ATOM") == "1":
        _lse = _lse_terms(ctx, node)
        if _lse is not None and _emit_lse(ctx, w, _lse):
            # tight=False keeps the aux interval floor (over side); the tangent
            # cuts tighten the under side toward the exact convex envelope.
            return Envelope(rows=[], tight=False)
    # Euclidean-norm atom (gated): convex OA of sqrt(sum t^2) instead of the loose
    # concave sqrt(square-sum). Off by default => byte-identical.
    if fname == "sqrt" and os.environ.get("DISCOPT_NORM_ATOM") == "1":
        _nt = _norm_terms(ctx, node)
        if _nt is not None and _emit_norm(ctx, w, _nt):
            return Envelope(rows=[], tight=False)
    arg = node.children[0]
    lt = ctx.rep(arg)
    lo, hi = ctx.bounds(arg)
    # Record ``w = fname(coeff*x_i + const)`` for the piecewise-partition refinement
    # (#640 S8 AMP ``disc_state`` recovery) when the argument is affine in a single
    # ORIGINAL variable. This includes intrinsics with no static single-curvature
    # envelope over the box (e.g. ``tan`` on a box straddling its inflection, or any
    # ``fname`` absent from ``_UNIVARIATE_FN``): partitioning splits the box into
    # single-curvature pieces where a sound secant/tangent envelope DOES exist.
    if fname in _PIECEWISE_UNIVARIATE_FN:
        aff = _single_orig_affine(ctx, lt)
        if aff is not None:
            v_idx, v_coeff = aff
            ctx.univariate_atom_specs.append((fname, w, v_idx, v_coeff, float(lt.const)))
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


def _emit_engine_equality(ctx: _Builder, coeffs: dict[int, float], rhs: float) -> None:
    """Emit the two-sided rows for ``sum_j coeffs[j]*col_j == rhs``."""
    ctx.add_row(dict(coeffs), rhs)
    ctx.add_row({j: -c for j, c in coeffs.items()}, -rhs)


def _try_finite_domain_trig_square_table(
    ctx: _Builder, node: CNode, w: int, base: CNode, base_lt: LinForm
) -> bool:
    """Emit the exact selector table for ``trig(int-affine)**2`` (#640 Bucket 1).

    Fires only when the ``sin``/``cos`` argument is affine in a SINGLE integer /
    binary original variable over a small finite domain. Builds one binary selector
    ``λ_v`` per domain value with ``Σλ_v = 1`` and the exact equality links
    ``x_i = Σ v·λ_v``, ``base = Σ trig(v)·λ_v``, ``sq(=w) = Σ trig(v)²·λ_v``. These
    reproduce the square exactly at each integer point; their continuous-``λ`` LP
    relaxation is the convex hull of those points (sound). Returns ``True`` iff the
    table was emitted (caller then treats the atom as tight and skips the loose
    generic power hull). Any structural mismatch → ``False`` (loose path stands).
    """
    fname = base.payload
    if fname not in ("sin", "cos"):
        return False
    # base_col: the single aux column that represents the trig call.
    base_items = [(j, c) for j, c in base_lt.coeffs.items() if c != 0.0]
    if len(base_items) != 1 or base_lt.const != 0.0 or base_items[0][1] != 1.0:
        return False
    base_col = base_items[0][0]
    # Argument must be affine in exactly one integer/binary ORIGINAL variable.
    arg_lt = ctx.rep(base.children[0])
    arg_items = [(j, float(c)) for j, c in arg_lt.coeffs.items() if abs(float(c)) > 1e-12]
    if len(arg_items) != 1:
        return False
    var_idx, arg_coeff = arg_items[0]
    if not (0 <= var_idx < ctx.n_orig):
        return False
    flat_lb = np.asarray(ctx.col_lb[: ctx.n_orig], dtype=np.float64)
    flat_ub = np.asarray(ctx.col_ub[: ctx.n_orig], dtype=np.float64)
    domain = _integer_domain_values(var_idx, _flat_variable_types(ctx.model), flat_lb, flat_ub)
    if domain is None:
        return False
    domain_values = list(domain)
    if not domain_values or len(domain_values) > _MAX_FINITE_DOMAIN_TRIG_TABLE_VALUES:
        return False
    fn = np.sin if fname == "sin" else np.cos
    arg_const = float(arg_lt.const)
    trig_values: list[float] = []
    square_values: list[float] = []
    for v in domain_values:
        tv = float(fn(arg_coeff * float(v) + arg_const))
        if not math.isfinite(tv):
            return False
        trig_values.append(tv)
        square_values.append(tv * tv)

    selector_cols: list[int] = []
    if len(domain_values) > 1:
        selector_cols = [ctx.new_aux(0.0, 1.0, integ=True) for _ in domain_values]
        _emit_engine_equality(ctx, {c: 1.0 for c in selector_cols}, 1.0)  # Σλ == 1
        _emit_engine_equality(
            ctx,
            {var_idx: 1.0, **{c: -float(v) for c, v in zip(selector_cols, domain_values)}},
            0.0,
        )  # x_i == Σ v·λ
        _emit_engine_equality(
            ctx,
            {base_col: 1.0, **{c: -tv for c, tv in zip(selector_cols, trig_values)}},
            0.0,
        )  # base == Σ trig(v)·λ
        _emit_engine_equality(
            ctx,
            {w: 1.0, **{c: -sq for c, sq in zip(selector_cols, square_values)}},
            0.0,
        )  # sq(w) == Σ trig(v)²·λ
    else:
        # Degenerate single-value domain: pin base and square to the exact value.
        _emit_engine_equality(ctx, {base_col: 1.0}, trig_values[0])
        _emit_engine_equality(ctx, {w: 1.0}, square_values[0])

    ctx.finite_domain_trig_square_tables.append(
        FiniteDomainTrigSquareTable(
            func_name=fname,
            var_idx=var_idx,
            domain_values=domain_values,
            trig_values=trig_values,
            square_values=square_values,
            selector_cols=selector_cols,
            base_col=base_col,
            square_col=w,
        )
    )
    return True


def _build_power(ctx: _Builder, node: CNode, w: int) -> Envelope:
    """``w = t**p``; ``t`` = base LinForm, ``p`` = constant exponent.

    AVM: the base is relaxed first, so ``(affine)**2``, ``f(x)**2`` and ``x**p``
    are all uniformly a 1-D power atom over the base's enclosure. Exact
    secant/tangent envelope on a definite-curvature (sign-definite or even-power)
    box; interval floor on a sign-straddling odd/fractional power (loose-but-sound;
    the two-piece-hull tightening is a deferred tightness item tracked in #640).
    """
    p = float(node.payload)
    base = node.children[0]
    lt = ctx.rep(base)
    lo, hi = ctx.bounds(base)
    # Exact finite-domain trig-square selector table (#640 Bucket 1): a
    # ``sin/cos(int-affine)**2`` over a small integer domain relaxes EXACTLY via
    # one-hot selectors, superseding the loose ``(trig-envelope)**2`` double-lift.
    if (
        p == 2.0
        and base.kind == "call"
        and _try_finite_domain_trig_square_table(ctx, node, w, base, lt)
    ):
        return Envelope(rows=[], tight=True)
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
    elif p == 2.0 and tight and curv == "convex":
        # Affine square ``(c·x_j + d)**2`` (single original, not bare): register the
        # affine form so the incremental McCormick patch can regenerate its
        # box-dependent 4-row envelope (#640 Bucket 3). ``t**2`` is convex for every
        # ``t``, so it is the same secant + 3-tangent hull on any (finite) box —
        # unlike an odd/higher monomial, no root-sign gating is needed.
        items = [(j, c) for j, c in lt.coeffs.items() if c != 0.0]
        if len(items) == 1 and 0 <= items[0][0] < ctx.n_orig:
            ctx.affine_square_map[items[0][0], w] = (float(items[0][1]), float(lt.const))
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


# --------------------------------------------------------------------------- #
# ATOM-REDUNDANCY-REVIEW: redundant on constructed cases (entropy — measured 0->0
# nodes as a >=2-var sum), BUT this atom's original driver is the centropy ex6_2_*
# regression (#19), whose structure may still need it; do NOT delete before that
# corpus check. See the cluster note above.
# Entropy atom (issue #632 centropy-tightness). ``x*log(x)`` is convex on x>0
# (``f'' = 1/x > 0``), but the factorable path shatters it into a loose bilinear
# ``x*(log x)`` over the DECOUPLED box (McCormick allows ``x=x_ub, log=log(x_lb)``
# simultaneously -> a floor far below the true convex minimum). The generic
# interval-Hessian certifier cannot recover the convexity of the factored form
# (dependency problem). So we RECOGNIZE the atom and emit its EXACT convex hull
# (tangent under + secant over) via the shared 1-D emitter — the same tight
# treatment ``exp``/``log`` already get from ``_UNIVARIATE_FN``. Sound by
# construction (a tangent of a convex function is a global underestimator; the
# secant chord is a global overestimator). Gated by ``DISCOPT_ENTROPY_ATOM``
# (default OFF, flag-graduation convention); when off, ``_build_product`` is
# byte-identical. Recognizes ``t*log(t)`` for any shared affine form ``t`` (so
# ``x log x``, ``(a x) log(a x)`` and ``(a x + b) log(a x + b)`` all fire); the
# relative-entropy generalization ``x log(x/y)`` has its own gated atom below.
def _xlogx(t: float) -> float:
    return t * math.log(t)


def _xlogx_prime(t: float) -> float:
    return math.log(t) + 1.0


def _linform_eq(a: LinForm, b: LinForm) -> bool:
    """Do two LinForms represent the same affine expression (same nonzero coeffs
    and constant)? Used to recognize ``t*log(t)`` for a shared affine ``t``."""
    if abs(a.const - b.const) > 1e-12:
        return False
    ca = {j: c for j, c in a.coeffs.items() if c != 0.0}
    cb = {j: c for j, c in b.coeffs.items() if c != 0.0}
    return ca.keys() == cb.keys() and all(abs(ca[j] - cb[j]) <= 1e-12 for j in ca)


def _entropy_prod_var(ctx: "_Builder", node: CNode):
    """If ``node`` is ``t*log(t)`` for a shared AFFINE form ``t`` (unit exponents),
    return ``(lt_t, lo, hi)`` for the shared 1-D entropy envelope; else ``None``.
    Covers ``x*log(x)``, ``(a x)*log(a x)``, ``(a x + b)*log(a x + b)`` — all convex
    (``t log t`` composed with an affine ``t``). Only recognized when the base
    factor and the log argument are the *same* affine form, so it is never
    mis-applied to a genuine bilinear ``x*log(y)``."""
    if node.kind != "prod":
        return None
    (exps,) = node.payload
    if len(node.children) != 2 or not all(float(e) == 1.0 for e in exps):
        return None
    for i, ch in enumerate(node.children):
        if ch.kind == "call" and ch.payload == "log":
            other = node.children[1 - i]
            (larg,) = ch.children
            lt_o = ctx.rep(other)
            if _linform_eq(lt_o, ctx.rep(larg)):
                lo, hi = ctx.bounds(other)
                return lt_o, lo, hi
    return None


# xexp atom (issue #632 adjacent-atom family). ``t*exp(t)`` is convex on ``t >= -2``
# (``f''(t) = exp(t)(t+2) >= 0``), but the factorable path shatters it into a bilinear
# ``t * exp(t)`` (``t`` against the convex-relaxed ``exp``) -> loose (root gap ~2.3 on
# an interior-min box). Recognize the atom for a shared affine ``t`` and emit its exact
# 1-D convex envelope (tangent under + secant over). Sound on any box with ``lo >= -2``
# (fully convex there). Gated ``DISCOPT_XEXP_ATOM`` (default OFF; byte-identical off).
# On ``lo < -2`` (box spans the inflection) or an overflow-prone ``hi`` fall through to
# the sound product path.
def _xexp(t: float) -> float:
    return t * math.exp(t)


def _xexp_prime(t: float) -> float:
    return math.exp(t) * (1.0 + t)


def _xexp_prod_var(ctx: "_Builder", node: CNode):
    """If ``node`` is ``t*exp(t)`` for a shared AFFINE form ``t`` (unit exponents),
    return ``(lt_t, lo, hi)`` for the shared 1-D xexp envelope; else ``None``.
    Covers ``x*exp(x)``, ``(a x)*exp(a x)``, ``(a x + b)*exp(a x + b)``. Recognized
    only when the base factor and the exp argument are the SAME affine form, so it is
    never mis-applied to a genuine bilinear ``x*exp(y)``."""
    if node.kind != "prod":
        return None
    (exps,) = node.payload
    if len(node.children) != 2 or not all(float(e) == 1.0 for e in exps):
        return None
    for i, ch in enumerate(node.children):
        if ch.kind == "call" and ch.payload == "exp":
            other = node.children[1 - i]
            (earg,) = ch.children
            lt_o = ctx.rep(other)
            if _linform_eq(lt_o, ctx.rep(earg)):
                lo, hi = ctx.bounds(other)
                return lt_o, lo, hi
    return None


# ATOM-REDUNDANCY-REVIEW: redundant (relent — measured 0->0 nodes; see the cluster note above).
# Relative-entropy atom (issue #632 adjacent-atom family). ``x*log(x/y)`` is
# JOINTLY convex on x,y>0 (the perspective of ``x log x``), but the factorable
# path relaxes it as ``x * log(x/y)`` (a bilinear of x against the concave-relaxed
# ``log(x/y)``) -> loose. Recognize the atom and emit its joint convex OA: tangent
# planes ``w >= D0 + dD/dx (x-x0) + dD/dy (y-y0)`` with dD/dx = log(x0/y0)+1,
# dD/dy = -x0/y0. Sound for ANY (x0,y0) with x0,y0>0 (tangent of a jointly-convex
# function). Gated ``DISCOPT_RELENT_ATOM`` (default OFF; byte-identical off).
def _relent_vars(ctx: "_Builder", node: CNode):
    """If ``node`` is ``x*log(x/y)`` (unit-coeff single vars x != y), return
    ``(col_x, col_y, (lox, hix, loy, hiy))``; else ``None``."""
    if node.kind != "prod":
        return None
    (exps,) = node.payload
    if len(node.children) != 2 or not all(float(e) == 1.0 for e in exps):
        return None
    for i, ch in enumerate(node.children):
        if ch.kind == "call" and ch.payload == "log":
            other = node.children[1 - i]
            (ratio,) = ch.children
            sv = ctx.single_var_affine(ctx.rep(other))
            if sv is None or abs(sv[1] - 1.0) > 1e-12 or ratio.kind != "prod":
                return None
            colx = sv[0]
            (rexps,) = ratio.payload
            if len(ratio.children) != 2:
                return None
            num = den = None
            for rc, re in zip(ratio.children, rexps):
                svr = ctx.single_var_affine(ctx.rep(rc))
                if svr is None or abs(svr[1] - 1.0) > 1e-12:
                    return None
                if abs(float(re) - 1.0) < 1e-12:
                    num = svr[0]
                elif abs(float(re) + 1.0) < 1e-12:
                    den = svr[0]
            if num != colx or den is None or den == colx:
                return None
            lox, hix = ctx.bounds(other)
            return colx, den, (lox, hix, ctx.col_lb[den], ctx.col_ub[den])
    return None


def _emit_relent(ctx: "_Builder", w: int, colx: int, coly: int, box: tuple) -> bool:
    """Emit joint tangent-plane underestimators of ``w = x*log(x/y)``."""
    lox, hix, loy, hiy = box
    if not (lox > 0.0 and loy > 0.0 and _finite(lox, hix) and _finite(loy, hiy)):
        return False
    # Sound aux floor (interval product of x>0 and log(x/y)) so the bound certifies.
    lr, hr = math.log(lox / hiy), math.log(hix / loy)
    ctx.col_lb[w] = max(ctx.col_lb[w], min(xv * lv for xv in (lox, hix) for lv in (lr, hr)))
    ctx.col_ub[w] = min(ctx.col_ub[w], max(xv * lv for xv in (lox, hix) for lv in (lr, hr)))
    refs = [(0.5 * (lox + hix), 0.5 * (loy + hiy))]
    refs += [(a, b) for a in (lox, hix) for b in (loy, hiy)]
    emitted = False
    for x0, y0 in refs:
        d0 = x0 * math.log(x0 / y0)
        gx = math.log(x0 / y0) + 1.0
        gy = -x0 / y0
        # w >= d0 + gx(x-x0) + gy(y-y0) -> -w + gx*x + gy*y <= gx*x0 + gy*y0 - d0
        ctx.add_row({w: -1.0, colx: gx, coly: gy}, gx * x0 + gy * y0 - d0)
        emitted = True
    return emitted


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
    # Entropy atom (gated): recognize ``x*log(x)`` and emit its exact convex
    # envelope instead of the loose decoupled-bilinear fold. Off by default =>
    # byte-identical. ``lo>0`` guards the domain (``x*log(x)`` -> nan at x=0, and
    # its lower tangent slope ``log x + 1`` -> -inf); on ``lo<=0`` fall through to
    # the sound product path.
    if os.environ.get("DISCOPT_ENTROPY_ATOM") == "1":
        _ent = _entropy_prod_var(ctx, node)
        if _ent is not None:
            _lt, _lo, _hi = _ent
            if _lo > 0.0:
                _tight = _emit_1d(ctx, w, _lt, _lo, _hi, _xlogx, _xlogx_prime, "convex")
                return Envelope(rows=[], tight=_tight)
    # xexp atom (gated): recognize ``t*exp(t)`` on its convex region ``t>=-2`` and emit
    # the exact 1-D convex envelope. ``lo<-2`` (spans the inflection) or an overflow-prone
    # ``hi`` => fall through to the sound product path. Off by default => byte-identical.
    if os.environ.get("DISCOPT_XEXP_ATOM") == "1":
        _xe = _xexp_prod_var(ctx, node)
        if _xe is not None:
            _lt, _lo, _hi = _xe
            if _lo >= -2.0 and _hi <= 700.0:
                _tight = _emit_1d(ctx, w, _lt, _lo, _hi, _xexp, _xexp_prime, "convex")
                return Envelope(rows=[], tight=_tight)
    # Relative-entropy atom (gated): joint convex OA of x*log(x/y). Off => identical.
    if os.environ.get("DISCOPT_RELENT_ATOM") == "1":
        _re = _relent_vars(ctx, node)
        if _re is not None and _emit_relent(ctx, w, _re[0], _re[1], _re[2]):
            return Envelope(rows=[], tight=False)

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

    # A non-unit scalar (e.g. the ``3`` the canonical atomizer folds into a factor
    # of ``-3·x0·x1``) must NOT be folded into a factor before the McCormick chain:
    # doing so scales a factor away from its bare original, so ``_fold_product`` can
    # no longer name the partial product and the bilinear/multilinear registration
    # the RLT/PSD separators consume is lost (issue #640 Bucket 2). Instead make the
    # node aux ``w`` the PURE (unscaled) product — ``_fold_product`` folds its
    # McCormick hull into ``w`` and registers it as the exact product of originals —
    # and carry the scalar in the node's REP (``rep_scale``). SOUNDNESS /
    # bound-neutrality: McCormick is 1-homogeneous in a scaled factor, so
    # ``scalar·hull(∏xᵢ) == hull(scalar·∏xᵢ)``; the LP feasible set is identical to
    # the old scaled-factor fold, only re-expressed as ``scalar·w`` with ``w`` the
    # named product column. Reusing ``w`` (rather than a second ``w == scalar·w_pure``
    # binding aux) keeps exactly ONE lifted column per product — what the RLT/PSD
    # separators, the cut-inheritance column identities, and the feasible-point
    # samplers expect (#640 Bucket 4).
    if scalar != 1.0:
        pure_b = factor_vals[0][1]
        for k in range(1, len(factor_vals)):
            pure_b = _interval_mul(pure_b, factor_vals[k][1])
        # ``w`` was allocated with the node (scalar·product) interval; reset it to the
        # PURE product interval — ``rep_scale`` recovers the node interval.
        ctx.col_lb[w], ctx.col_ub[w] = pure_b
        if ctx.track_aux_exprs:
            pe = factor_vals[0][2]
            for k in range(1, len(factor_vals)):
                pe = (pe * factor_vals[k][2]) if pe is not None else None  # type: ignore[operator]
            if pe is not None:
                ctx.aux_expr[w] = pe  # w now holds the PURE product value
        tight = _fold_product(ctx, w, factor_vals)
        return Envelope(rows=[], tight=tight, rep_scale=scalar)
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
    _register_ratio_of_originals(ctx, node, w)
    factors = [_factor_value(ctx, ch, float(e)) for ch, e in zip(node.children, exps)]
    tight = _fold_product(ctx, w, factors)
    return Envelope(rows=[], tight=(tight and len(factors) == 2) or logspace)


def _register_ratio_of_originals(ctx: _Builder, node: CNode, w: int) -> None:
    """Register ``w = (Π x_i) / (Π y_j)`` in ``ctx.ratio_map`` when exact.

    Only a quotient of BARE original variables (coefficient exactly 1, integer
    exponents) is registered — the consumer (the integer-ratio partition bound,
    issue #309) treats the registered aux as the exact quotient of the *named*
    originals, so a scaled/affine/aux factor must be refused (same soundness gate
    as :meth:`_Builder.single_orig_col` for the product maps). Metadata only:
    no rows are emitted and no behavior changes for non-consumers.
    """
    (exps,) = node.payload
    num: list[int] = []
    den: list[int] = []
    for child, e in zip(node.children, exps):
        ef = float(e)
        if not ef.is_integer() or ef == 0.0:
            return
        # Gate on the child KIND before touching its rep: ``ctx.rep`` builds the
        # child's envelope on first call, so probing a composite child here would
        # reorder aux/row creation relative to the pre-change build (a byte-level
        # layout change on the default path). A bare original is always safe.
        if getattr(child, "kind", None) != "var":
            return
        col = ctx.single_orig_col(ctx.rep(child))
        if col is None:
            return
        (num if ef > 0 else den).extend([col] * abs(int(ef)))
    if not num or not den:
        return
    ctx.ratio_map[(tuple(sorted(num)), tuple(sorted(den)))] = w


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
def _fix_single_var_equalities(
    model: Model, flat_lb: np.ndarray, flat_ub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the box of any variable pinned by a single-variable equality.

    Scans ``==`` constraints whose (normalized ``body == rhs``) body is affine in a
    SINGLE original variable, ``a*x_i + c == rhs`` ⇒ ``x_i = (rhs - c)/a``, and
    intersects ``x_i``'s box with the fixed value. This is EXACT, not a relaxation:
    the equality is already a hard model constraint, so fixing the variable removes
    no feasible point; it only lets the per-atom envelopes below see the collapsed
    box (e.g. ``tan(x)`` on ``x == 0`` relaxes to the exact ``0`` instead of the
    loose mixed-curvature interval over the declared box). The fix is applied only
    when the pinned value lies inside the current box; an out-of-box value is left
    for the equality's own LP rows to expose as infeasible (never silently widen or
    empty the box here). Returns fresh arrays; inputs are untouched.
    """
    lb = np.array(flat_lb, dtype=np.float64)
    ub = np.array(flat_ub, dtype=np.float64)
    n = len(lb)
    for con in model._constraints:
        if getattr(con, "sense", None) != "==":
            continue
        try:
            coeff, const = _linearize_affine_expr(con.body, model, n)
        except (ValueError, TypeError):
            continue
        nz = [(j, float(c)) for j, c in enumerate(np.asarray(coeff)) if abs(float(c)) > 1e-12]
        if len(nz) != 1:
            continue
        j, c = nz[0]
        val = (float(con.rhs) - float(const)) / c
        if not math.isfinite(val):
            continue
        # Only collapse within the existing box; leave any infeasibility (val
        # outside [lb, ub]) to the equality's own rows so we never empty the box.
        if lb[j] - 1e-9 <= val <= ub[j] + 1e-9:
            lb[j] = max(lb[j], val)
            ub[j] = min(ub[j], val)
    return lb, ub


def _emit_quadratic_rlt(
    ctx: _Builder, model: Model, flat_lb: np.ndarray, flat_ub: np.ndarray
) -> None:
    """Add quadratic constraint-factor RLT rows (issue #640 Bucket 2, Phase 2).

    For each degree-2 polynomial constraint ``g {<=,==} 0`` and each variable
    ``x_m`` in its support with a finite box, multiply ``(-g)`` (which is ``>= 0``
    on the feasible region) by the bound factors ``(x_m - l_m) >= 0`` and
    ``(u_m - x_m) >= 0``. Expanding yields degree-3 monomials ``x_k x_l x_m`` which
    are lifted on demand (recursive McCormick over the registered lower-degree
    product columns) and linearized. The assembled row (single-sourced through
    ``rlt_quadratic_bound_cut_row``) is a valid inequality in the lifted space —
    it equals the non-negative product at every feasible point, so it never cuts a
    feasible point (sound, only tightens). An equality parent ``g == 0`` also gets
    the reverse row (pinning the product to zero). Selective (only constraints
    touching the model's nonlinear support) and capped (``DISCOPT_RLT_QUAD_MAX``
    new columns) so the LP does not blow up.
    """
    from discopt._jax.milp_relaxation import _quadratic_constraint_forms
    from discopt._jax.rlt_cuts import rlt_quadratic_bound_cut_row
    from discopt.solver_tuning import current as _tuning

    n_orig = ctx.n_orig
    forms = _quadratic_constraint_forms(model, n_orig)
    if not forms:
        return

    # Vars that participate in some registered nonlinear product — the RLT factor
    # is only worth forming where it touches nonlinear structure.
    nonconvex: set[int] = set()
    for a, b in ctx.bilinear_map:
        nonconvex.update((a, b))
    for i, _p in ctx.monomial_map:
        nonconvex.add(i)
    for tri in ctx.trilinear_map:
        nonconvex.update(tri)
    for multi in ctx.multilinear_map:
        nonconvex.update(multi)

    # sorted-multiset -> lifted column, seeded from the base decomposition.
    rlt_prod: dict[tuple[int, ...], int] = {}
    for (a, b), col in ctx.bilinear_map.items():
        rlt_prod[tuple(sorted((a, b)))] = col
    for (i, p), col in ctx.monomial_map.items():
        if p == 2:
            rlt_prod[(i, i)] = col
    for tri, col in ctx.trilinear_map.items():
        rlt_prod[tuple(sorted(tri))] = col

    cap = int(_tuning().rlt_quad_max)
    start_cols = len(ctx.col_lb)

    def _finite_box(c: int) -> bool:
        return _finite(ctx.col_lb[c], ctx.col_ub[c])

    def _lift_pair(cp: int, cq: int) -> int:
        bp = (ctx.col_lb[cp], ctx.col_ub[cp])
        bq = (ctx.col_lb[cq], ctx.col_ub[cq])
        tb = _interval_mul(bp, bq)
        w = ctx.new_aux(tb[0], tb[1])
        _emit_mccormick(ctx, w, LinForm.col(cp), bp, LinForm.col(cq), bq)
        return w

    def _ensure(key: tuple[int, ...]) -> Optional[int]:
        key = tuple(sorted(key))
        hit = rlt_prod.get(key)
        if hit is not None:
            return hit
        if len(key) == 1:
            return key[0]
        if len(ctx.col_lb) - start_cols >= cap:
            return None
        if len(key) == 2:
            a, b = key
            if not (_finite_box(a) and _finite_box(b)):
                return None
            w = _lift_pair(a, b)
            rlt_prod[key] = w
            ctx.register_power(a, 2, w) if a == b else ctx.register_product([a, b], w)
            return w
        if len(key) == 3:
            a, b, c = key
            pair = _ensure((a, b))
            if pair is None or len(ctx.col_lb) - start_cols >= cap or not _finite_box(c):
                return None
            w = _lift_pair(pair, c)
            rlt_prod[key] = w
            if len({a, b, c}) == 3:
                ctx.register_product([a, b, c], w)
            return w
        return None

    specs: list[tuple] = []
    for quad, lin, const, sense in forms:
        support = set(lin) | {k for k, _ in quad} | {ll for _, ll in quad}
        if not (support & nonconvex):
            continue
        for mm in sorted(support):
            if not _finite_box(mm):
                continue
            required: set[tuple[int, ...]] = set()
            for i in lin:
                required.add(tuple(sorted((i, mm))))
            for ka, kb in quad:
                required.add(tuple(sorted((ka, kb))))
                required.add(tuple(sorted((ka, kb, mm))))
            prod_map: dict[tuple[int, ...], int] = {}
            ok = True
            for req in required:
                rc = _ensure(req)
                if rc is None:
                    ok = False
                    break
                prod_map[req] = rc
            if not ok:
                continue
            specs.append((quad, lin, const, mm, ctx.col_lb[mm], ctx.col_ub[mm], sense, prod_map))

    if not specs:
        return
    n_total = len(ctx.col_lb)

    def _orig_col(i: int) -> Optional[int]:
        return i if 0 <= i < n_orig else None

    for quad, lin, const, mm, lm, um, sense, prod_map in specs:

        def _prod_col(key: tuple[int, ...], _pm: dict = prod_map) -> Optional[int]:
            return _pm.get(tuple(sorted(key)))

        for lower, bnd in ((True, lm), (False, um)):
            assembled = rlt_quadratic_bound_cut_row(
                quad, lin, const, mm, float(bnd), lower, _orig_col, _prod_col, n_total
            )
            if assembled is None:
                continue
            coeffs, rhs = assembled
            nz = np.flatnonzero(coeffs)
            # coeffs·z >= rhs  ->  -coeffs·z <= -rhs.
            ctx.add_row({int(j): -float(coeffs[j]) for j in nz}, -float(rhs))
            if sense == "==":
                ctx.add_row({int(j): float(coeffs[j]) for j in nz}, float(rhs))


def _clamped_breakpoints(
    raw: "np.ndarray", lo: float, hi: float, min_intervals: int = 2
) -> Optional[list[float]]:
    """Return sorted breakpoints of ``raw`` clipped to ``[lo, hi]`` with the box
    endpoints included, or ``None`` if fewer than ``min_intervals`` usable
    intervals remain.

    The node box (``lo, hi``) may be tighter than when the AMP partition was
    created, so breakpoints outside it are dropped and the true endpoints are
    re-inserted — the partition must exactly tile ``[lo, hi]`` for the
    disaggregation identity ``x == sum_k x_hat_k`` to be exact.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo <= 1e-12:
        return None
    pts = [float(p) for p in np.asarray(raw, dtype=np.float64) if lo < float(p) < hi]
    merged = sorted({lo, hi, *pts})
    # Drop near-duplicate breakpoints that would create a degenerate interval.
    dedup: list[float] = [merged[0]]
    for p in merged[1:]:
        if p - dedup[-1] > 1e-9:
            dedup.append(p)
    if dedup[-1] < hi - 1e-12:
        dedup[-1] = hi
    if len(dedup) - 1 < min_intervals:
        return None
    return dedup


_MAX_PIECEWISE_CELLS = 144  # cap on grid cells per bilinear (cost guard)


def _add_piecewise_bilinear(
    ctx: "_Builder",
    a: int,
    b: int,
    w: int,
    a_pts: list[float],
    b_pts: Optional[list[float]],
) -> None:
    """Append a sound disaggregated piecewise-McCormick relaxation of ``w == x_a *
    x_b``: 2-D grid over partitions ``a_pts`` (of ``a``) and ``b_pts`` (of ``b``)
    when both are supplied, else 1-D over ``a_pts`` with ``b`` on its full box.

    Piecewise McCormick (Nagarajan et al., CP 2016; Alpine.jl): a cell binary
    ``lam`` per (interval-of-a x interval-of-b) cell, ``sum lam == 1``,
    disaggregates ``a = sum a_hat``, ``b = sum b_hat``, ``w = sum w_hat`` with each
    ``_hat`` copy confined to its cell when active (0 otherwise), and per-cell
    McCormick on ``w_hat`` using the CELL bounds. SOUNDNESS: for the TRUE point
    ``(a, b, w = a*b)`` in cell ``c*``, ``lam_{c*}=1`` (else 0), ``a_hat_{c*}=a``,
    ``b_hat_{c*}=b``, ``w_hat_{c*}=a*b`` (else 0) satisfies every row — no feasible
    point is cut; the full-box McCormick rows on ``w`` remain, so the result is
    their intersection (never looser, strictly tighter once ``lam`` is integral).
    Refining BOTH factors is required to certify a bilinear whose two factor ranges
    are both wide (1-D refinement of one factor plateaus at ``~Δ_other/4``).
    """
    bl = float(ctx.col_lb[b])
    bu = float(ctx.col_ub[b])
    if not (math.isfinite(bl) and math.isfinite(bu)):
        return
    # Build the grid of (a-interval, b-interval) cells. 1-D degenerates to a single
    # b-"interval" spanning the full box.
    a_ivs = list(zip(a_pts[:-1], a_pts[1:]))
    b_ivs = list(zip(b_pts[:-1], b_pts[1:])) if b_pts else [(bl, bu)]
    if len(a_ivs) * len(b_ivs) > _MAX_PIECEWISE_CELLS:
        b_ivs = [(bl, bu)]  # fall back to 1-D on ``a`` to bound the column count
    cells = [(alo, ahi, blo, bhi) for (alo, ahi) in a_ivs for (blo, bhi) in b_ivs]

    lam: list[int] = []
    a_hat: list[int] = []
    b_hat: list[int] = []
    w_hat: list[int] = []
    for alo, ahi, blo, bhi in cells:
        lam.append(ctx.new_aux(0.0, 1.0, integ=True))
        a_hat.append(ctx.new_aux(min(0.0, alo), max(0.0, ahi)))
        b_hat.append(ctx.new_aux(min(0.0, blo), max(0.0, bhi)))
        corners = [alo * blo, alo * bhi, ahi * blo, ahi * bhi, 0.0]
        w_hat.append(ctx.new_aux(min(corners), max(corners)))

    def _eq(coeffs: dict[int, float], rhs: float) -> None:
        ctx.add_row(coeffs, rhs)
        ctx.add_row({j: -c for j, c in coeffs.items()}, -rhs)

    n = len(cells)
    _eq({lam[k]: 1.0 for k in range(n)}, 1.0)
    _eq({a: 1.0, **{a_hat[k]: -1.0 for k in range(n)}}, 0.0)
    _eq({b: 1.0, **{b_hat[k]: -1.0 for k in range(n)}}, 0.0)
    _eq({w: 1.0, **{w_hat[k]: -1.0 for k in range(n)}}, 0.0)
    for k, (alo, ahi, blo, bhi) in enumerate(cells):
        # alo*lam <= a_hat <= ahi*lam ;  blo*lam <= b_hat <= bhi*lam
        ctx.add_row({a_hat[k]: 1.0, lam[k]: -ahi}, 0.0)
        ctx.add_row({a_hat[k]: -1.0, lam[k]: alo}, 0.0)
        ctx.add_row({b_hat[k]: 1.0, lam[k]: -bhi}, 0.0)
        ctx.add_row({b_hat[k]: -1.0, lam[k]: blo}, 0.0)
        # Per-cell McCormick (under):
        ctx.add_row({w_hat[k]: -1.0, b_hat[k]: alo, a_hat[k]: blo, lam[k]: -alo * blo}, 0.0)
        ctx.add_row({w_hat[k]: -1.0, b_hat[k]: ahi, a_hat[k]: bhi, lam[k]: -ahi * bhi}, 0.0)
        # Per-cell McCormick (over):
        ctx.add_row({w_hat[k]: 1.0, b_hat[k]: -alo, a_hat[k]: -bhi, lam[k]: alo * bhi}, 0.0)
        ctx.add_row({w_hat[k]: 1.0, b_hat[k]: -ahi, a_hat[k]: -blo, lam[k]: ahi * blo}, 0.0)


# Univariate intrinsics eligible for piecewise refinement: ``fname -> (f, f',
# curvature(arg_lo, arg_hi))``. Reuses the static-envelope table and adds ``tan``
# (absent there because it has no single-curvature envelope over a box straddling
# its inflection — the exact case piecewise splitting fixes).
_PIECEWISE_UNIVARIATE_FN: dict[str, tuple[Callable, Callable, Callable]] = {
    "tan": (np.tan, lambda t: 1.0 / (np.cos(t) ** 2), _curv_by_sign(True)),
    **{name: (f, fp, cv) for name, (f, fp, cv, _dom) in _UNIVARIATE_FN.items()},
}


def _single_orig_affine(ctx: "_Builder", lt: "LinForm") -> Optional[tuple[int, float]]:
    """Return ``(var_idx, coeff)`` iff ``lt`` is ``coeff*x_i (+ const)`` for a single
    ORIGINAL variable (any nonzero coeff, any const), else ``None``."""
    items = [(j, c) for j, c in lt.coeffs.items() if c != 0.0]
    if len(items) != 1:
        return None
    j, c = items[0]
    if 0 <= j < ctx.n_orig and math.isfinite(c) and c != 0.0:
        return (j, float(c))
    return None


def _univariate_inflection_args(fname: str, alo: float, ahi: float) -> list[float]:
    """Curvature-change (inflection) points of ``fname`` inside ``(alo, ahi)`` in
    ARGUMENT space, so each sub-interval is single-curvature."""
    out: list[float] = []
    # f'' has the sign of the argument -> inflection at 0.
    if fname in ("tan", "tanh", "atan", "asin", "asinh", "atanh", "erf", "sinh"):
        if alo < 0.0 < ahi:
            out.append(0.0)
        return out
    if fname in ("sin", "cos"):
        # sin'' = -sin (inflections at k*pi); cos'' = -cos (inflections at pi/2+k*pi).
        start = 0.0 if fname == "sin" else 0.5 * math.pi
        k_min = math.ceil((alo - start) / math.pi)
        k_max = math.floor((ahi - start) / math.pi)
        for k in range(k_min, k_max + 1):
            p = start + k * math.pi
            if alo < p < ahi:
                out.append(float(p))
    return out


def _tan_branch_safe(alo: float, ahi: float) -> bool:
    """True iff ``[alo, ahi]`` contains no ``tan`` asymptote (pi/2 + k*pi)."""
    margin = 1e-6
    k_min = math.floor((alo - 0.5 * math.pi) / math.pi) - 1
    k_max = math.ceil((ahi - 0.5 * math.pi) / math.pi) + 1
    for k in range(k_min, k_max + 1):
        asymptote = 0.5 * math.pi + k * math.pi
        if alo - margin <= asymptote <= ahi + margin:
            return False
    return True


def _emit_piecewise_1d(
    ctx: "_Builder",
    w_hat: int,
    x_hat: int,
    lam: int,
    coeff: float,
    const: float,
    alo: float,
    ahi: float,
    f: Callable[[float], float],
    fp: Callable[[float], float],
    curv: Optional[str],
) -> None:
    """Append the λ-homogenized secant/tangent envelope of ``w_hat == f(coeff*x_hat +
    const)`` valid on the active interval (arg in ``[alo, ahi]``).

    ``arg_k = coeff*x_hat + const*lam`` is the per-interval argument (== the true
    arg when ``lam == 1``, 0 when ``lam == 0``). A line ``s*arg + b`` valid on the
    interval becomes ``s*arg_k + b*lam`` so it is the true line when the interval is
    active and the trivial ``0 (>=|<=) 0`` when inactive. SOUNDNESS: on a
    curvature-certified piece each tangent is a supporting line (rigorous one-sided
    bound) and the endpoint secant a chord (rigorous opposite bound); homogenizing
    by ``lam`` keeps them exact at the true point and vacuous elsewhere.
    """
    if curv is None or ahi - alo < _MIN_WIDTH:
        return
    try:
        flo, fhi = float(f(alo)), float(f(ahi))
    except (ValueError, ArithmeticError):
        return
    if not _finite(flo, fhi):
        return
    ssl = (fhi - flo) / (ahi - alo)  # secant slope (arg space)
    sint = flo - ssl * alo  # secant intercept

    def _line(slope: float, intercept: float, sign: float) -> None:
        # sign*w_hat (<=|>=) sign*(slope*arg_k + intercept*lam):
        #   arg_k = coeff*x_hat + const*lam
        #   -> row  sign*w_hat - sign*slope*coeff*x_hat - sign*(slope*const+intercept)*lam <= 0
        if not (_finite(slope, intercept)):
            return
        ctx.add_row(
            {
                w_hat: sign,
                x_hat: -sign * slope * coeff,
                lam: -sign * (slope * const + intercept),
            },
            0.0,
        )

    mid = 0.5 * (alo + ahi)
    sec_sign = +1.0 if curv == "convex" else -1.0  # convex: w<=secant; concave: w>=secant
    _line(ssl, sint, sec_sign)
    for t0 in (alo, mid, ahi):
        try:
            g, gp = float(f(t0)), float(fp(t0))
        except (ValueError, ArithmeticError):
            continue
        if not _finite(g, gp):
            continue
        _line(gp, g - gp * t0, -sec_sign)  # tangent: opposite side of the secant


def _add_piecewise_univariate(
    ctx: "_Builder",
    w: int,
    v: int,
    coeff: float,
    const: float,
    f: Callable[[float], float],
    fp: Callable[[float], float],
    curv_fn: Callable[[float, float], Optional[str]],
    pts: list[float],
    branch_safe: Optional[Callable[[float, float], bool]] = None,
) -> None:
    """Append a sound disaggregated piecewise envelope of ``w == f(coeff*x_v +
    const)`` over the partition ``pts`` of ``x_v`` (used for both univariate
    intrinsics and integer powers ``x_v**p``). Aborts (adds nothing) if any interval
    is not soundly relaxable (e.g. a ``tan`` piece crossing an asymptote), leaving
    the atom's existing interval floor intact.
    """
    m = len(pts) - 1
    # Sound generous column box for the disaggregated ``w_hat`` copies: the atom's
    # full-box aux range (rigorous, contains every sub-interval range) plus 0 (the
    # inactive value).
    w_lo = min(0.0, float(ctx.col_lb[w]))
    w_hi = max(0.0, float(ctx.col_ub[w]))
    if not (_finite(w_lo, w_hi)):
        return
    # Pre-check: every interval's argument box must be soundly relaxable.
    specs: list[tuple[float, float, Optional[str]]] = []
    for k in range(m):
        a0 = coeff * pts[k] + const
        a1 = coeff * pts[k + 1] + const
        alo, ahi = (a0, a1) if a0 <= a1 else (a1, a0)
        if branch_safe is not None and not branch_safe(alo, ahi):
            return  # cannot relax this atom soundly over the partition
        specs.append((alo, ahi, curv_fn(alo, ahi)))

    lam: list[int] = []
    x_hat: list[int] = []
    w_hat: list[int] = []
    for k in range(m):
        p_lo, p_hi = pts[k], pts[k + 1]
        lam.append(ctx.new_aux(0.0, 1.0, integ=True))
        x_hat.append(ctx.new_aux(min(0.0, p_lo), max(0.0, p_hi)))
        w_hat.append(ctx.new_aux(w_lo, w_hi))

    def _eq(coeffs: dict[int, float], rhs: float) -> None:
        ctx.add_row(coeffs, rhs)
        ctx.add_row({j: -c for j, c in coeffs.items()}, -rhs)

    _eq({lam[k]: 1.0 for k in range(m)}, 1.0)
    _eq({v: 1.0, **{x_hat[k]: -1.0 for k in range(m)}}, 0.0)
    _eq({w: 1.0, **{w_hat[k]: -1.0 for k in range(m)}}, 0.0)
    for k in range(m):
        p_lo, p_hi = pts[k], pts[k + 1]
        ctx.add_row({x_hat[k]: 1.0, lam[k]: -p_hi}, 0.0)  # x_hat <= p_hi*lam
        ctx.add_row({x_hat[k]: -1.0, lam[k]: p_lo}, 0.0)  # x_hat >= p_lo*lam
        alo, ahi, curv = specs[k]
        _emit_piecewise_1d(ctx, w_hat[k], x_hat[k], lam[k], coeff, const, alo, ahi, f, fp, curv)


def _apply_partition_refinement(ctx: "_Builder", disc_state: object) -> None:
    """Add sound piecewise structure for every atom depending on a partitioned
    variable (#640 S8 AMP ``disc_state`` recovery). No-op when no partitions.
    """
    parts = getattr(disc_state, "partitions", None)
    if not parts:
        return

    def _clamped(v: int, extra: Optional[list[float]] = None) -> Optional[list[float]]:
        raw = parts.get(v)
        if raw is None:
            return None
        merged = np.asarray(raw, dtype=np.float64)
        if extra:
            merged = np.concatenate([merged, np.asarray(extra, dtype=np.float64)])
        return _clamped_breakpoints(merged, float(ctx.col_lb[v]), float(ctx.col_ub[v]))

    for (a, b), w in list(ctx.bilinear_map.items()):
        a_pts = _clamped(a)
        b_pts = _clamped(b)
        if a_pts is None and b_pts is None:
            continue
        # Refine both factors (2-D grid) when both are partitioned — required to
        # certify a bilinear with two wide factor ranges. If only one is
        # partitioned, refine that one (1-D) with the other on its full box.
        if a_pts is not None:
            _add_piecewise_bilinear(ctx, a, b, w, a_pts, b_pts)
        elif b_pts is not None:
            _add_piecewise_bilinear(ctx, b, a, w, b_pts, None)

    for fname, w, v, coeff, const in list(ctx.univariate_atom_specs):
        if v not in parts:
            continue
        entry = _PIECEWISE_UNIVARIATE_FN.get(fname)
        if entry is None:
            continue
        f, fp, curv_fn = entry
        # Inject the function's inflection points (mapped from arg space to x space)
        # so no sub-interval straddles a curvature change; without this a partition
        # refined only near the incumbent leaves the inflection interval loose.
        infl_x: list[float] = []
        _e0 = coeff * float(ctx.col_lb[v]) + const
        _e1 = coeff * float(ctx.col_ub[v]) + const
        for a_infl in _univariate_inflection_args(fname, min(_e0, _e1), max(_e0, _e1)):
            infl_x.append((a_infl - const) / coeff)
        pts = _clamped(v, infl_x)
        if pts is None:
            continue
        branch_safe = _tan_branch_safe if fname == "tan" else None
        _add_piecewise_univariate(ctx, w, v, coeff, const, f, fp, curv_fn, pts, branch_safe)

    # Integer powers ``x_v**p`` (the sphere/square terms): piecewise-refine the aux
    # so a convex square's under-estimating tangents tighten as ``x_v`` is
    # partitioned — without this a wide symmetric box (e.g. ``y**2`` on ``[-s, s]``)
    # keeps its aux pinned to the tangent at 0, decoupling the square from its base.
    for (i, p), w in list(ctx.monomial_map.items()):
        if i not in parts or not isinstance(p, int) or p < 2:
            continue
        pw = p

        def _fpow(t: float, _pw: int = pw) -> float:
            return float(t**_pw)

        def _fppow(t: float, _pw: int = pw) -> float:
            return float(_pw * t ** (_pw - 1))

        # ``x**p`` is convex for even ``p`` (all x) and for odd ``p`` on ``x >= 0``;
        # concave for odd ``p`` on ``x <= 0`` — curvature has the sign of ``x`` when
        # ``p`` is odd (inflection at 0), constant convex when ``p`` is even.
        curv_pow = _curv_const("convex") if pw % 2 == 0 else _curv_by_sign(True)
        infl_x = [0.0] if (pw % 2 == 1 and ctx.col_lb[i] < 0.0 < ctx.col_ub[i]) else []
        pts = _clamped(i, infl_x)
        if pts is None:
            continue
        _add_piecewise_univariate(ctx, w, i, 1.0, 0.0, _fpow, _fppow, curv_pow, pts)


def build_uniform_relaxation(
    model: Model,
    box: Optional[tuple[np.ndarray, np.ndarray]] = None,
    *,
    rlt_quad: bool = False,
    skip_separable_floor: bool = False,
    skip_convex_lift: bool = False,
    disc_state: object = None,
    build_deadline: Optional[float] = None,
) -> UniformRelaxation:
    """Build the uniform factorable relaxation of ``model`` over ``box``.

    Parameters
    ----------
    model : Model
    box : optional ``(flat_lb, flat_ub)`` — the B&B node box in flat variable
        order (like ``build_milp_relaxation``'s ``bound_override``). Defaults to
        the model's declared variable bounds.
    build_deadline : float, optional
        Issue #694 anytime/incremental build. When set (a ``time.perf_counter``
        absolute time), the constraint-row loop stops adding rows once the deadline
        passes and returns the relaxation built *so far*. Default ``None`` (the
        legacy monolithic build; **byte-identical** to before). A truncated build
        is still a **valid outer relaxation**: dropping constraint rows only enlarges
        the feasible region, so its LP minimum is a valid (weaker) lower bound —
        never falsified (baron-gap-plan.md §8 "weaken but never falsify"). The
        objective is fully linearized BEFORE the loop, so ``objective_bound_valid``
        (and the box/separable objective floor) are unaffected by truncation — a
        truncated finite-box relaxation therefore still carries a finite bound (the
        #694 entry experiment: finite by 8–45 % of build on every tested structure).
        The result records ``milp._build_truncated`` / ``_build_constraints_done`` /
        ``_build_constraints_total``. Gated default-off via
        ``SolverTuning.anytime_root_build``; only ``_root_relaxation_lower_bound``
        passes a non-``None`` value today.
    rlt_quad : bool, default False
        Enable the quadratic constraint-factor RLT pass (issue #640 Bucket 2):
        multiply each degree-2 polynomial constraint by variable bound factors,
        lifting the resulting degree-3 monomials on demand, and add the valid RLT
        product rows. Sound (only tightens); the caller gates it on the
        ``rlt_level1`` build option AND ``DISCOPT_RLT_QUAD``.

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

    # NOTE (roadmap P3): general branch-and-reduce / box tightening (FBBT/OBBT) is
    # owned by the separate branch-and-reduce workstream and arrives HERE via the
    # ``box`` (node-box) interface — a tighter box automatically yields uniformly
    # tighter envelopes below (McCormick/secant/tangent are monotone in the box).
    # The one tightening this layer DOES perform is EXACT variable fixing from
    # single-variable equality constraints (issue #640 Bucket 1): ``a*x_i == b``
    # forces ``x_i = b/a``, so relaxing over the collapsed box [b/a, b/a] is exact
    # for that variable — a sound win (the envelope of e.g. ``tan(x)`` on the fixed
    # point is the exact value) that cannot remove a feasible point, since the
    # equality is already a hard constraint of the model.
    flat_lb, flat_ub = _fix_single_var_equalities(model, flat_lb, flat_ub)

    # Box-independent analysis, computed once per model and pinned on it (issue #632
    # EP1): the canonical DAG (which pins CNodes -> stable id() keys), reconstructed
    # expressions, DCP verdicts, compiled value/grad fns, and curvature certificates.
    analysis = _get_analysis_cache(model)
    dag: CanonicalDAG = analysis.dag
    ctx = _Builder(model, flat_lb, flat_ub, analysis, skip_convex_lift=skip_convex_lift)

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
    #
    # Issue #694 anytime build: when ``build_deadline`` is set, stop repping
    # constraints once the deadline passes and keep the rows built so far. The
    # objective (above) is already fully linearized, so a prefix of the constraint
    # rows is a valid weaker outer relaxation (dropping rows only enlarges the
    # feasible set -> the LP min stays a valid lower bound). The deadline is polled
    # BETWEEN whole constraints (never mid-``rep``), so no partial/invalid row is
    # ever emitted. Default ``None`` -> the whole loop runs, byte-identical.
    n_constraints = len(model._constraints)
    constraints_done = 0
    build_truncated = False
    for con, cnode in zip(model._constraints, dag.constraints):
        if build_deadline is not None and time.perf_counter() >= build_deadline:
            build_truncated = True
            break
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
        constraints_done += 1

    # Quadratic constraint-factor RLT (issue #640 Bucket 2, gated). Runs AFTER the
    # base build so every product column the base decomposition registered is
    # available to seed the on-demand degree-3 lifting. Skipped on a truncated
    # anytime build (#694): it only *adds* rows (tightening), so skipping it when
    # the build deadline is already spent is sound (weaker, never falsified) and
    # keeps the truncation honoring the deadline.
    if rlt_quad and not build_truncated:
        _emit_quadratic_rlt(ctx, model, flat_lb, flat_ub)

    obj_offset = obj_lin.const

    # ── Plain box-interval objective floor + validity (soundness guard) ──
    # The LP optimum is a valid lower bound ONLY if the relaxed objective is
    # actually bounded below over the box. When a nonlinear atom cannot be
    # enveloped (unbounded box: McCormick rows dropped for infinite endpoints,
    # transcendental over an infinite argument, …) its aux column is a free
    # interval-floor column; if such a column carries objective cost and is
    # otherwise unconstrained, the LP is unbounded below — yet the warm-started
    # Rust simplex can mis-report a finite "optimal" (issue #15). Reporting that as
    # a bound would be a FALSE certificate. We compute a SOUND box-interval lower
    # bound on the (minimize-equivalent) objective from the column bounds; a cost
    # column that is unbounded on its cost-relevant side and appears in NO row makes
    # that lower bound -inf, and we refuse the objective bound (the solver falls
    # back to its rigorous alphaBB/interval bound) rather than trust a
    # possibly-fabricated LP value. Mirrors the federation's
    # ``objective_bound_valid=False`` on an un-relaxable / under-constrained
    # objective. NOTE: the LP feasible region ⊆ the column box, so the LP optimum is
    # always ≥ ``obj_box_lb`` — this is why the separable floor below only helps
    # (and is only added) when it STRICTLY exceeds ``obj_box_lb``.
    _row_cols: set[int] = set()
    for _coeffs, _ in ctx.rows:
        _row_cols.update(_coeffs)
    obj_bound_valid = True
    obj_box_lb = obj_offset
    for j, coef in obj_lin.coeffs.items():
        edge = ctx.col_lb[j] if coef > 0 else ctx.col_ub[j]
        contrib = coef * edge
        if not math.isfinite(contrib):
            if j not in _row_cols:
                obj_bound_valid = False
            obj_box_lb = -math.inf
        else:
            obj_box_lb += contrib
    if not math.isfinite(obj_box_lb):
        obj_bound_valid = False

    # ── Separable objective lower bound (issue #640 Bucket 1 — federation-parity) ──
    # ``sep_lb`` is a sound constant lower bound on the MINIMIZE-EQUIVALENT
    # objective, derived by separable term analysis (integer-domain enumeration for
    # cos(integer-affine), x*exp(x) >= -1/e, monotone-polynomial vertex min,
    # reciprocal / even-power enclosures). It rescues shapes the static per-atom
    # envelope leaves loose or unbounded on a wide box: an unbounded
    # x*exp(x)+cos(y)+z^3-z^2 objective, or an integer-affine cos whose continuous
    # envelope is the [-1,1] range. We add it as the single cut ``obj_lin >= sep_lb``
    # ONLY when it strictly improves ``obj_box_lb`` (i.e. it would change the LP
    # bound) — a redundant cut on a linear / already-tight objective is skipped so
    # bound-neutral relaxation consumers (e.g. the incremental-McCormick row-exact
    # validator) are untouched. SOUNDNESS: every feasible ORIGINAL point x maps to a
    # lifted point with obj_lin = objective(x) >= sep_lb, so the cut removes NO image
    # of a feasible point — only relaxation-only points where the loose envelope dips
    # below the true objective floor. The LP optimum after the cut is therefore still
    # a valid lower bound (>= before, <= true optimum), and — being a genuine LP row
    # rather than a free-column simplex quirk — it makes the objective certifiably
    # bounded, so ``obj_bound_valid`` becomes True.
    if (
        not skip_separable_floor
        and dag.objective is not None
        and model._objective is not None
        and obj_lin.coeffs
    ):
        obj_expr = model._objective.expression
        min_equiv_expr = (
            UnaryOp("neg", obj_expr)
            if model._objective.sense == ObjectiveSense.MAXIMIZE
            else obj_expr
        )
        sep_lb = _expression_lower_bound_for_lift(min_equiv_expr, model, flat_lb, flat_ub)
        if sep_lb is not None and math.isfinite(sep_lb) and sep_lb > obj_box_lb + 1e-9:
            # obj_lin >= sep_lb  <=>  -(obj_lin variable part) <= obj_lin.const - sep_lb
            ctx.add_row(obj_lin.scaled(-1.0).coeffs, obj_lin.const - sep_lb)
            obj_bound_valid = True

    # Piecewise-McCormick partition refinement (#640 S8 — AMP `disc_state`
    # recovery). When the caller (the AMP adaptive-multivariate-partitioning loop)
    # supplies partition breakpoints, add sound piecewise structure that tightens
    # every atom depending on a partitioned variable. No-op when ``disc_state`` is
    # empty (every non-AMP caller and the root), so the base build is unchanged.
    _apply_partition_refinement(ctx, disc_state)

    # Assemble the LP.
    n_cols = len(ctx.col_lb)
    c = np.zeros(n_cols, dtype=np.float64)
    for j, coef in obj_lin.coeffs.items():
        c[j] += coef

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
        objective_bound_valid=obj_bound_valid,
    )
    # Rigorous box-interval objective floor (issue #640 Bucket 2, nvs22). A valid
    # global lower bound on the (minimize-equivalent) objective, computed from the
    # cost-column bounds alone — independent of the constraint rows and of the
    # node-solve conditioning clamp (the cost columns are never near-inf, so this
    # stays finite even when a free non-cost column drives the clamped LP to a
    # spurious "unbounded"). The node solver falls back to it to report a sound
    # bound instead of declining. ``None`` when no finite floor exists.
    milp._objective_floor = obj_box_lb if math.isfinite(obj_box_lb) else None
    # Issue #694 anytime-build provenance (informational; the LP is sound either
    # way). ``_build_truncated`` is True when the constraint loop stopped early on
    # the ``build_deadline``; the two counters record coverage for diagnostics/tests.
    milp._build_truncated = build_truncated
    milp._build_constraints_done = constraints_done
    milp._build_constraints_total = n_constraints
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
        affine_square_map=dict(ctx.affine_square_map),
        ratio_map=dict(ctx.ratio_map),
        finite_domain_trig_square_tables=list(ctx.finite_domain_trig_square_tables),
        integrality=list(ctx.integrality),
        univariate_atom_specs=list(ctx.univariate_atom_specs),
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
    exact envelope (deferred tightening; tracked in #640).
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
