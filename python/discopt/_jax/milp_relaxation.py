"""
MILP Relaxation Builder for AMP (Adaptive Multivariate Partitioning).

Builds a linear programming relaxation of the original MINLP by:
  1. Replacing bilinear terms x_i*x_j with auxiliary variables w_ij and
     adding standard McCormick envelope constraints.
  2. Replacing monomial terms x_i^n with auxiliary variables s_i and adding
     piecewise tangent-cut underestimators plus partition-activated secant
     overestimators when the variable is discretized.
  3. Linearizing the original objective and constraints.

The LP relaxation gives a valid lower bound:
  LP_opt ≤ global NLP_opt

As the partition becomes finer (more intervals in disc_state), more tangent and
local secant cuts are added for monomials, tightening the lower bound.

Theory: Nagarajan et al., JOGO 2018, Section 4 (piecewise McCormick relaxation).
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt._jax._numeric import EFFECTIVE_INF as _EFFECTIVE_INF
from discopt._jax._numeric import is_effectively_finite as _is_effectively_finite
from discopt._jax.discretization import DiscretizationState
from discopt._jax.embedding import EmbeddingMap, build_embedding_map
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.operator_relaxations import (
    critical_points_in_interval as _critical_points_in_interval,
)
from discopt._jax.operator_relaxations import tan_range as _tan_range
from discopt._jax.operator_relaxations import trig_range as _trig_range
from discopt._jax.operator_relaxations import trig_square_curvature as _trig_square_curvature
from discopt._jax.operator_relaxations import (
    trig_square_grad as _trig_square_grad,
)
from discopt._jax.operator_relaxations import trig_square_range as _trig_square_range
from discopt._jax.operator_relaxations import trig_square_value as _trig_square_value
from discopt._jax.term_classifier import (
    NonlinearTerms,
    _compute_var_offset,
    _get_flat_index,
    distribute_products,
    extract_reciprocal_power,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    ObjectiveSense,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

logger = logging.getLogger(__name__)

# Dedupe identical warnings emitted across repeated relaxation builds (AMP iterates).
_warned_messages: set[str] = set()
_MAX_INTEGER_COS_ENUM = 10000
_MAX_FINITE_EXP_ARG = float(np.log(np.finfo(np.float64).max))
_MAX_TRIG_PIECEWISE_SPAN = 2.0 * math.pi
# Lifted ``1/g`` (issue #154) is convex only on ``g > 0``; require the inner
# interval's lower end to clear this strictly-positive floor so the reciprocal
# and its tangent slopes (``-1/g**2``) stay finite and well-conditioned.
_RECIP_MIN_DENOM = 1e-6
# Lifted ``sqrt(g)`` needs ``g >= 0``; tolerate a tiny negative slack from loose
# interval bounds on a provably-nonnegative argument (sum of squares) and clamp.
_SQRT_NEG_TOL = 1e-9
# Conditioning guard shared by the fractional-power envelopes (issue #158) and the
# lifted reciprocal/sqrt envelopes (issue #154). A power ``x**p`` with ``p<0`` (or
# ``0<p<1``) near a small lower bound, the convex ``1/g`` slope ``-1/g**2``, and the
# concave ``sqrt(g)`` slope ``1/(2*sqrt(g))`` all blow up (reaching ~1e9+) as the
# interval's lower end approaches zero. An LP row carrying such a coefficient against
# an RHS of order 1 is numerically unreliable: HiGHS returns a polytope that EXCLUDES
# feasible points (so OBBT shrinks a variable past its true feasible range and the
# per-node relaxer reports a feasible node "infeasible" — the nvs08 false optimum),
# or stalls at ``iteration_limit`` with a partial objective that is NOT a valid dual
# bound. Refuse to emit/lift any envelope cut whose slope exceeds this limit;
# dropping a cut only ENLARGES the relaxation, so abstention is always sound.
_LIFT_MAX_ENVELOPE_SLOPE = 1e6

# Largest argument magnitude a *cross-term* sqrt/reciprocal lift (issue #154,
# increment 2) may carry. Lifting ``sqrt(g)`` of a cross-term polynomial (e.g.
# nvs05/nvs22 ``sqrt(x4**2 + 2*x4*x5*x7 + x5**2)``) folds the product factors
# into McCormick aux columns whose bounds — and the resulting envelope row
# coefficients — scale with ``|g|``. When that magnitude reaches ~1e9 the LP is
# ill-conditioned: the fast simplex backend returns a wrong "optimal" bound that
# can exceed the true optimum (an unsound dual bound), while HiGHS still solves
# it correctly. Fast-simplex is demonstrated reliable up to ~1e7 on these
# relaxations, so cross-term lifts abstain above this limit. Abstention only
# drops a cut (the constraint is simply left un-lifted), which ENLARGES the
# relaxation and is therefore always sound.
_LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE = 1e7


def _envelope_slope_ok(slope: float) -> bool:
    """True when an envelope cut's slope is well-conditioned enough to emit.

    A finite slope at or below :data:`_LIFT_MAX_ENVELOPE_SLOPE` keeps the LP row
    numerically reliable; anything steeper (e.g. ``p*t**(p-1)`` for ``x**p`` with
    ``p<0`` near a small lower bound) makes HiGHS return an unsound polytope, so
    the caller abstains. Dropping a cut only enlarges the relaxation, so an
    abstention is always sound — at worst the aux column keeps its value bounds.
    """
    return bool(np.isfinite(slope)) and abs(slope) <= _LIFT_MAX_ENVELOPE_SLOPE


_MAX_TRIG_PIECEWISE_INTERVALS = 32
_MAX_TRIG_IMPORTED_BREAKPOINTS = _MAX_TRIG_PIECEWISE_INTERVALS + 1
_MAX_TRIG_PIECEWISE_WIDTH = math.pi / 6.0
_MAX_RELAXATION_PARTITION_INTERVALS = 128
_MAX_OBJECTIVE_LIFT_POWER = 6
_MAX_FINITE_DOMAIN_TRIG_TABLE_VALUES = 256

# A relaxation row coefficient / RHS, or a variable bound, at or above this
# magnitude is "numerically catastrophic": it comes from a McCormick/secant
# envelope over a variable with an enormous (or effectively infinite) range, and
# leaves the LP so ill-conditioned (dynamic range ~1e13+) that the backend
# (HiGHS) returns kSolveError and *no* bound at all. The cap sits above any
# legitimate modeling coefficient (big-M ~1e9, gear4's 1e6 linking term) and
# below the 1e11..1e37 entries such envelopes produce, so well-scaled models are
# never affected. Used only by ``sanitize_relaxation_for_conditioning``, which in
# turn only feeds the last-resort root-relaxation fallback bound — never the main
# solve — so dropping a borderline-large row can at most weaken that fallback,
# never the primary result. The recovered bound is empirically stable for caps
# from 1e11 down to 1e6 on the affected instances.
_RELAX_NUMERIC_CAP = 1e10

# Equilibrate the lifted relaxation before an external LP/MILP solve when its
# coefficient dynamic range exceeds this (matches the Rust simplex's own scaling
# trigger). The lifted McCormick rows of a product over a wide variable box mix
# tiny constants (~1e-9) with large bound-derived coefficients (~1e7), giving a
# >1e15 spread on ex1252's boundary sub-boxes — HiGHS stalls on it (a 452x96 LP
# hits its time limit) while the pure-Rust simplex, which equilibrates, solves it
# in ~0.03s. Geometric-mean row/column scaling brings the spread down so the
# external (HiGHS) path converges instead of timing out (issue #184).
_RELAX_EQUILIBRATE_TRIGGER = 1e6


def equilibrate_relaxation_lp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    bounds: list[tuple[float, float]],
    integrality: Optional[np.ndarray],
    *,
    iters: int = 20,
) -> tuple[
    np.ndarray,
    Optional[sp.spmatrix],
    Optional[np.ndarray],
    list[tuple[float, float]],
    np.ndarray,
]:
    """Geometric-mean (Ruiz) equilibration of ``min c·x s.t. A_ub x <= b_ub``.

    Alternating row/column infinity-norm sweeps drive every row and column to unit
    scale; the factors are snapped to powers of two so the transform is exact in
    floating point. **Integer columns are never scaled** (that would corrupt
    integrality), so the conditioning of the integer part is left to the solver.

    Soundness: this is an exact diagonal rescaling ``x = D·x'`` of the LP. Row
    scaling ``s_r`` rewrites ``A_r x <= b_r`` as ``(s_r A_r) x <= s_r b_r`` — same
    feasible set. Column scaling ``d_j`` substitutes ``x_j = d_j x'_j`` into the
    objective and rows consistently. The optimal objective value is therefore
    **unchanged**, so the LP/MILP bound from the scaled solve is the true bound;
    only the returned point needs ``x = col_scale · x'`` to map back. Returns
    ``(c', A', b', bounds', col_scale)``.
    """
    if A_ub is None or b_ub is None:
        return np.asarray(c, dtype=np.float64), A_ub, b_ub, bounds, np.ones(len(c))

    A = sp.csr_matrix(A_ub).astype(np.float64).copy()
    m, n = A.shape
    col_scale = np.ones(n)
    row_scale = np.ones(m)
    # Columns we are allowed to scale: continuous only (integer cols stay at 1).
    scalable = np.ones(n, dtype=bool) if integrality is None else (np.asarray(integrality) == 0)

    for _ in range(iters):
        absA = A.copy()
        absA.data = np.abs(absA.data)
        rmax = np.asarray(absA.max(axis=1).todense()).ravel()
        rmax[rmax == 0.0] = 1.0
        dr = 2.0 ** np.round(np.log2(1.0 / np.sqrt(rmax)))
        A = sp.diags(dr) @ A
        row_scale *= dr

        absA = A.copy()
        absA.data = np.abs(absA.data)
        cmax = np.asarray(absA.max(axis=0).todense()).ravel()
        cmax[cmax == 0.0] = 1.0
        dc = 2.0 ** np.round(np.log2(1.0 / np.sqrt(cmax)))
        dc[~scalable] = 1.0
        A = A @ sp.diags(dc)
        col_scale *= dc

    b2 = np.asarray(b_ub, dtype=np.float64) * row_scale
    c2 = np.asarray(c, dtype=np.float64) * col_scale
    bounds2 = [
        (
            lo / d if np.isfinite(lo) else lo,
            hi / d if np.isfinite(hi) else hi,
        )
        for (lo, hi), d in zip(bounds, col_scale)
    ]
    return c2, sp.csr_matrix(A), b2, bounds2, col_scale


def _warn_once(msg: str, *args) -> None:
    formatted = msg % args if args else msg
    if formatted in _warned_messages:
        return
    _warned_messages.add(formatted)
    logger.warning("%s", formatted)


# ---------------------------------------------------------------------------
# Result and model wrappers
# ---------------------------------------------------------------------------


@dataclass
class MilpRelaxationResult:
    """Result of solving a MILP relaxation.

    ``objective`` is the relaxation MILP's incumbent. ``bound`` is the rigorous
    dual lower bound on the relaxation optimum (hence on the original problem);
    it is the value AMP/OA-style callers must use as the global lower bound. It
    is ``None`` when no valid dual bound is available (or the relaxation
    objective is not itself a valid bound on the original).
    """

    status: str  # "optimal", "infeasible", "error", "time_limit"
    objective: Optional[float] = None
    bound: Optional[float] = None
    x: Optional[np.ndarray] = None


class MilpRelaxationModel:
    """Wrapper around a MILP that exposes a .solve() method.

    Stores the LP data and delegates solving to solve_milp (HiGHS).
    """

    def __init__(
        self,
        c: np.ndarray,
        A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
        b_ub: Optional[np.ndarray],
        bounds: list[tuple[float, float]],
        obj_offset: float = 0.0,
        integrality: Optional[np.ndarray] = None,
        objective_bound_valid: bool = True,
    ):
        self._c = c
        self._A_ub = A_ub
        self._b_ub = b_ub
        self._bounds = bounds
        self._obj_offset = obj_offset
        self._integrality = integrality
        self._objective_bound_valid = objective_bound_valid

    def solve(
        self,
        time_limit: Optional[float] = None,
        gap_tolerance: float = 1e-4,
        backend: str = "auto",
    ) -> MilpRelaxationResult:
        from discopt.solvers import SolveStatus
        from discopt.solvers.lp_backend import get_milp_solver

        # backend="auto": HiGHS if present, else POUNCE. backend="simplex" routes
        # to the warm-started-simplex B&B (falls back to auto if unavailable).
        solve_milp = get_milp_solver(backend=backend)

        # Equilibrate the LP for the external (HiGHS/POUNCE) backends when it is
        # badly scaled. The pure-Rust simplex already equilibrates internally, so
        # skip the (redundant) Python pre-scaling there; HiGHS does not cope with
        # the lifted relaxation's >1e15 coefficient spread and stalls without it
        # (issue #184). The transform is exact, so the returned bound/objective is
        # unchanged — only the solution point is mapped back through ``col_scale``.
        c_s, A_s, b_s, bounds_s = self._c, self._A_ub, self._b_ub, self._bounds
        col_scale = None
        if backend != "simplex" and self._A_ub is not None:
            data = sp.csr_matrix(self._A_ub).data
            nz = np.abs(data[data != 0.0])
            ill = bool(
                nz.size
                and np.isfinite(nz).all()
                and nz.max() / nz.min() > _RELAX_EQUILIBRATE_TRIGGER
            )
            if ill:
                c_s, A_s, b_s, bounds_s, col_scale = equilibrate_relaxation_lp(
                    self._c, self._A_ub, self._b_ub, self._bounds, self._integrality
                )

        result = solve_milp(
            c=c_s,
            A_ub=A_s,
            b_ub=b_s,
            bounds=bounds_s,
            integrality=self._integrality,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
        )
        # Map the scaled solution point back to original variables (x = D·x').
        if col_scale is not None and result.x is not None:
            result.x = np.asarray(result.x, dtype=np.float64) * col_scale

        # Map SolveStatus enum to string
        status_map = {
            SolveStatus.OPTIMAL: "optimal",
            SolveStatus.INFEASIBLE: "infeasible",
            SolveStatus.UNBOUNDED: "unbounded",
            SolveStatus.TIME_LIMIT: "time_limit",
            SolveStatus.ITERATION_LIMIT: "iteration_limit",
            SolveStatus.ERROR: "error",
        }
        status_str = status_map.get(result.status, str(result.status))

        obj = None
        if result.objective is not None and self._objective_bound_valid:
            obj = float(result.objective) + self._obj_offset

        # The sound lower bound on the original problem is the MILP's dual bound
        # (not its incumbent), and only when this relaxation's objective is a
        # valid bound on the original (``_objective_bound_valid``).
        bound = None
        if result.bound is not None and self._objective_bound_valid:
            bound = float(result.bound) + self._obj_offset

        return MilpRelaxationResult(status=status_str, objective=obj, bound=bound, x=result.x)


def sanitize_relaxation_for_conditioning(
    model: "MilpRelaxationModel",
) -> "MilpRelaxationModel":
    """Return a copy of *model* with numerically catastrophic content removed, so
    the LP backend can produce a (sound, possibly weaker) bound instead of failing.

    Two transforms, both of which only *relax* the feasible set — the LP optimum
    therefore remains a valid lower bound for a minimization (weaker, never
    higher than the true optimum):

    1. Drop any constraint row whose coefficient or RHS is non-finite or has
       magnitude >= ``_RELAX_NUMERIC_CAP``. Removing a constraint enlarges the
       feasible set.
    2. Clamp any variable bound of magnitude >= ``_RELAX_NUMERIC_CAP`` to +/-inf.
       Widening a variable's box enlarges the feasible set. (A clamped objective
       variable can make the LP unbounded -> bound becomes -inf/None, still sound.)

    Both are no-ops on well-scaled models (no row or bound reaches the cap), so
    this is safe to apply unconditionally before a fallback root-bound solve.
    """
    cap = _RELAX_NUMERIC_CAP

    A = model._A_ub
    b = model._b_ub
    if A is not None and b is not None and A.shape[0] > 0:
        A = sp.csr_matrix(A)  # accepts dense or sparse; typed CSR for .data/.indptr
        b = np.asarray(b, dtype=np.float64)
        row_of_nz = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
        bad_nz = ~np.isfinite(A.data) | (np.abs(A.data) >= cap)
        keep = np.isfinite(b) & (np.abs(b) < cap)
        if bad_nz.any():
            keep[row_of_nz[bad_nz]] = False
        if not keep.all():
            logger.debug(
                "relaxation conditioning: dropped %d catastrophic constraint row(s)",
                int((~keep).sum()),
            )
            A = A[keep]
            b = b[keep]
        if A.shape[0] == 0:
            A = None
            b = None
    else:
        A = None
        b = None

    bounds = [
        (
            lo if abs(lo) < cap else (-np.inf if lo < 0 else np.inf),
            hi if abs(hi) < cap else (np.inf if hi > 0 else -np.inf),
        )
        for (lo, hi) in model._bounds
    ]

    return MilpRelaxationModel(
        c=model._c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        obj_offset=model._obj_offset,
        integrality=model._integrality,
        objective_bound_valid=model._objective_bound_valid,
    )


@dataclass
class UnivariateRelaxation:
    """Lifted outer relaxation for a supported univariate operator."""

    expr_id: int
    func_name: str
    aux_col: int
    arg_coeff: np.ndarray
    arg_const: float
    arg_lb: float
    arg_ub: float


@dataclass
class PiecewiseUnivariateInterval:
    """Binary-selected interval for a mixed-curvature univariate relaxation."""

    delta_col: int
    lb: float
    ub: float
    curvature: Optional[str]


@dataclass
class PiecewiseUnivariateRelaxation:
    """Partition-aware relaxation for a lifted univariate operator."""

    relax: UnivariateRelaxation
    intervals: list[PiecewiseUnivariateInterval]


@dataclass
class UnivariateSquareRelaxation:
    """Lifted square of a supported univariate auxiliary."""

    base_col: int
    aux_col: int
    base_lb: float
    base_ub: float


@dataclass
class AffineSquareRelaxation:
    """Univariate square envelope on an affine residual over lifted columns.

    ``r = const + Σ resid[col] * z[col]`` is an affine form whose factors (a
    bilinear/monomial product) are already lifted to auxiliary columns. ``s`` is
    the lifted ``r**2`` and ``[r_lb, r_ub]`` is an interval-arithmetic outer
    bound on ``r`` used to build the tangent/secant square envelope (issue #155).
    """

    aux_col: int
    resid: dict[int, float]
    const: float
    r_lb: float
    r_ub: float


@dataclass
class AffinePowerRelaxation:
    """Univariate power envelope on a scaled single-variable residual.

    ``r = scale * x_var`` (``x_var`` is the original flat column ``var_idx``) and
    ``w`` is the lifted ``r**power`` for an integer ``power >= 3``.  The envelope
    is built in well-conditioned ``r`` space, so a base variable ranging over a
    wide box (e.g. ``x in [0, 2950]`` in ex1252, deliberately scaled by
    ``1/2950`` so ``r in [0, 1]``) does not produce the ``~1e10`` aux column the
    raw ``x**power`` monomial would and the magnitude cap would drop.
    """

    aux_col: int
    var_idx: int
    scale: float
    power: int
    r_lb: float
    r_ub: float


@dataclass
class RatioRelaxation:
    """Linear-fractional envelope for a ratio of products ``coeff·m/q`` (#185).

    ``m`` is the lifted numerator product column and ``q`` the lifted denominator
    product column (each a standard McCormick-enveloped bilinear/multilinear aux,
    or an original variable). ``r`` is a fresh column standing for the *pure*
    ratio ``m/q``; the division node is mapped to ``r`` with the scalar ``coeff``
    applied by the linearizer (keeping a large numerator constant out of the
    envelope coefficients). The McCormick envelope of the bilinear identity
    ``r·q = m`` — emitted over ``[r_lb, r_ub] × [q bounds]`` — outer-approximates
    the quotient: at any true point ``r = m/q`` so ``r·q = m`` holds exactly and
    all four inequalities are satisfied, hence the relaxed feasible set is a
    superset of the true one (sound lower bound). Requires ``q`` strictly
    sign-definite and bounded away from zero on the node box.
    """

    r_col: int
    q_col: int
    m_col: int
    r_lb: float
    r_ub: float


@dataclass
class PiecewiseTrigSquareInterval:
    """Binary-selected interval for a direct trig-square relaxation."""

    delta_col: int
    lb: float
    ub: float
    curvature: Optional[str]


@dataclass
class PiecewiseTrigSquareRelaxation:
    """Partition-aware direct relaxation for sin(arg)^2 or cos(arg)^2."""

    square: UnivariateSquareRelaxation
    func_name: str
    arg_coeff: np.ndarray
    arg_const: float
    arg_lb: float
    arg_ub: float
    intervals: list[PiecewiseTrigSquareInterval]


@dataclass
class FiniteDomainTrigSquareTable:
    """Exact selector table for sin(integer_affine)^2 or cos(integer_affine)^2."""

    square: UnivariateSquareRelaxation
    func_name: str
    var_idx: int
    arg_coeff: float
    arg_const: float
    domain_values: list[int]
    trig_values: list[float]
    square_values: list[float]
    selector_cols: list[int]


@dataclass
class MinMaxObjectiveLift:
    """Epigraph/hypograph lift for a supported objective-level min/max call."""

    func_name: str
    aux_col: int
    branch_exprs: tuple[Expression, ...]
    branch_bounds: tuple[tuple[Optional[float], Optional[float]], ...]
    aux_bounds: tuple[float, float]


# ---------------------------------------------------------------------------
# Helpers: variable bounds
# ---------------------------------------------------------------------------


def _piecewise_product_bounds(
    a_k: float,
    b_k: float,
    y_lb: float,
    y_ub: float,
) -> tuple[list[float], float, float]:
    """Return interval corner products and their min/max values."""
    corners = [a_k * y_lb, a_k * y_ub, b_k * y_lb, b_k * y_ub]
    return corners, min(corners), max(corners)


def _compute_piecewise_big_m(corners: list[float]) -> float:
    """Scale Big-M with the interval magnitude instead of adding a flat constant."""
    max_corner = max(abs(float(c)) for c in corners)
    return max_corner * (1.0 + 1e-4) + max(1e-6, 1e-4 * max_corner)


def _linear_expr_bounds(
    coeff: np.ndarray,
    const: float,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, float]:
    """Return interval bounds for an affine expression over variable bounds."""
    lower = float(const)
    upper = float(const)
    for c_i, lb_i, ub_i in zip(coeff, lb, ub):
        c = float(c_i)
        if c == 0.0:
            # A zero coefficient contributes nothing, regardless of the
            # variable's bounds. Skipping it avoids ``0 * inf = nan`` poisoning
            # the interval when an unrelated variable is unbounded.
            continue
        if c > 0.0:
            lower += c * float(lb_i)
            upper += c * float(ub_i)
        else:
            lower += c * float(ub_i)
            upper += c * float(lb_i)
    return lower, upper


def _constant_value(expr: Expression) -> Optional[float]:
    if not isinstance(expr, Constant):
        return None
    values = np.asarray(expr.value, dtype=np.float64).ravel()
    if values.size != 1:
        return None
    return float(values[0])


def _eval_constant_expr(expr: Expression) -> Optional[float]:
    """Evaluate a *variable-free* subexpression to a scalar, else return None.

    Unlike :func:`_constant_value` (which only recognizes a literal ``Constant``
    node), this folds composite constant subexpressions such as ``neg(2.5)``
    (a unary negation of a literal), ``(-3) * (-3)``, and other arithmetic over
    constants. It is deliberately conservative: it returns ``None`` the moment a
    variable, index, function call, or any unhandled node is encountered, so it
    can never mis-fold an expression that actually depends on a decision
    variable. Folding such factors is exact (value-preserving), so using it in
    the product linearizer only ever tightens the relaxation while staying
    sound.
    """
    direct = _constant_value(expr)
    if direct is not None:
        return direct
    if isinstance(expr, UnaryOp):
        val = _eval_constant_expr(expr.operand)
        if val is None:
            return None
        if expr.op == "neg":
            return -val
        if expr.op == "abs":
            return abs(val)
        return None
    if isinstance(expr, BinaryOp):
        left = _eval_constant_expr(expr.left)
        if left is None:
            return None
        right = _eval_constant_expr(expr.right)
        if right is None:
            return None
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            if right == 0.0:
                return None
            return left / right
        if expr.op == "**":
            try:
                result = left**right
            except (ValueError, OverflowError, ZeroDivisionError):
                return None
            # A negative base to a fractional power yields a complex result;
            # that is not a real constant we can fold, so bail conservatively.
            if isinstance(result, complex):
                return None
            return float(result)
        return None
    return None


def _affine_var_base(expr: Expression, model: Model) -> Optional[tuple[float, int]]:
    """Return ``(coeff, flat_idx)`` if ``expr`` is a single scaled variable.

    Matches ``coeff * x`` in any of: a bare scalar variable, a constant-scaled
    variable (``c*x`` / ``x*c``), a variable divided by a constant (``x/c``), or
    a negation thereof.  Returns ``None`` for anything with additive structure or
    more than one variable.  Used to recognize an affine single-variable power
    base ``(c*x)**n`` so it can be lifted in well-conditioned scaled ``r = c*x``
    space rather than as a raw ``x**n`` monomial.
    """
    flat = _get_flat_index(expr, model)
    if flat is not None:
        return 1.0, flat
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        inner = _affine_var_base(expr.operand, model)
        if inner is not None:
            return -inner[0], inner[1]
        return None
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            lc = _constant_value(expr.left)
            if lc is not None:
                inner = _affine_var_base(expr.right, model)
                return (lc * inner[0], inner[1]) if inner is not None else None
            rc = _constant_value(expr.right)
            if rc is not None:
                inner = _affine_var_base(expr.left, model)
                return (rc * inner[0], inner[1]) if inner is not None else None
            return None
        if expr.op == "/":
            rc = _constant_value(expr.right)
            if rc is not None and rc != 0.0:
                inner = _affine_var_base(expr.left, model)
                return (inner[0] / rc, inner[1]) if inner is not None else None
            return None
    return None


def _finite_bound_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not _is_effectively_finite(value):
        return None
    return value


def _expand_integer_powers_for_relaxation(expr: Expression, model: Model) -> Expression:
    """Expand small integer powers of affine expressions for existing monomial lifts."""

    def visit(node: Expression) -> Expression:
        if isinstance(node, BinaryOp):
            left = visit(node.left)
            right = visit(node.right)
            if node.op == "**":
                exp = _constant_value(right)
                if exp is not None:
                    n = int(exp)
                    if exp == n and 2 <= n <= _MAX_OBJECTIVE_LIFT_POWER:
                        base = left
                        product = base
                        for _ in range(n - 1):
                            product = BinaryOp("*", product, base)
                        return distribute_products(product)
            return distribute_products(BinaryOp(node.op, left, right))
        if isinstance(node, UnaryOp):
            return UnaryOp(node.op, visit(node.operand))
        if isinstance(node, SumExpression):
            return SumExpression(visit(node.operand), axis=node.axis)
        if isinstance(node, SumOverExpression):
            return SumOverExpression([visit(term) for term in node.terms])
        # Preserve FunctionCall object identity so existing univariate lift maps
        # keyed by id(expr) remain usable during branch linearization.
        return node

    return distribute_products(visit(expr))


def _expression_lower_bound_for_lift(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    expanded = _expand_integer_powers_for_relaxation(expr, model)
    lower = _separable_objective_lower_bound(expanded, model, flat_lb, flat_ub)
    return _finite_bound_or_none(lower)


def _expression_upper_bound_for_lift(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    lower_of_negated = _expression_lower_bound_for_lift(
        UnaryOp("neg", expr),
        model,
        flat_lb,
        flat_ub,
    )
    if lower_of_negated is None:
        return None
    return -lower_of_negated


def _collect_monomial_terms_for_lift(expr: Expression, model: Model) -> set[tuple[int, int]]:
    terms: set[tuple[int, int]] = set()

    def visit(node: Expression) -> None:
        if isinstance(node, BinaryOp):
            if node.op == "*":
                decomp = _decompose_product(node, model)
                if decomp is not None:
                    _scalar, indices = decomp
                    unique = list(dict.fromkeys(indices))
                    if len(unique) == 1 and len(indices) >= 2:
                        terms.add((unique[0], len(indices)))
            elif node.op == "**":
                flat = _get_flat_index(node.left, model)
                exp = _constant_value(node.right)
                if flat is not None and exp is not None:
                    n = int(exp)
                    if exp == n and n >= 2:
                        terms.add((flat, n))
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, UnaryOp):
            visit(node.operand)
            return
        if isinstance(node, FunctionCall):
            for arg in node.args:
                visit(arg)
            return
        if isinstance(node, IndexExpression) and not isinstance(node.base, Variable):
            visit(node.base)
            return
        if isinstance(node, SumExpression):
            visit(node.operand)
            return
        if isinstance(node, SumOverExpression):
            for term in node.terms:
                visit(term)

    visit(expr)
    return terms


# Flat-variable monomial: a sorted tuple of original variable indices, repeated
# by power (e.g. ``x1**2 * x0`` → ``(0, 1, 1)``).  An affine-square residual is
# represented as ``(const, [(coeff, monomial), ...])``.
_Monomial = tuple[int, ...]
_AffineSquare = tuple[float, list[tuple[float, _Monomial]]]


def _product_to_monomial(expr: Expression, model: Model) -> tuple[float, _Monomial] | None:
    """Fold a pure product of original variables / integer powers / constants.

    Returns ``(scalar, monomial)`` where ``monomial`` is the sorted tuple of
    original flat variable indices (repeated by power), or ``None`` if the
    product contains a non-polynomial leaf (a function call, a non-constant
    division, a fractional power, …).
    """
    scalar = [1.0]
    idxs: list[int] = []

    def visit(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op == "*":
            return visit(e.left) and visit(e.right)
        if isinstance(e, UnaryOp) and e.op == "neg":
            scalar[0] *= -1.0
            return visit(e.operand)
        if isinstance(e, Constant):
            scalar[0] *= float(e.value)
            return True
        flat = _get_flat_index(e, model)
        if flat is not None:
            idxs.append(flat)
            return True
        if isinstance(e, BinaryOp) and e.op == "**" and isinstance(e.right, Constant):
            base_flat = _get_flat_index(e.left, model)
            if base_flat is not None:
                exp_val = float(e.right.value)
                n = int(exp_val)
                if exp_val == n and n >= 1:
                    idxs.extend([base_flat] * n)
                    return True
        return False

    if not visit(expr):
        return None
    return scalar[0], tuple(sorted(idxs))


def _expr_to_polynomial(expr: Expression, model: Model) -> _AffineSquare | None:
    """Walk a *distributed* expression into ``(const, [(coeff, monomial), ...])``.

    Returns ``None`` if any leaf is not a polynomial in the original variables
    (so the caller falls back to the existing relaxation paths instead of
    misclassifying a transcendental residual).
    """
    const = [0.0]
    terms: list[tuple[float, _Monomial]] = []

    def visit(e: Expression, scale: float) -> bool:
        if isinstance(e, Constant):
            const[0] += scale * float(e.value)
            return True
        flat = _get_flat_index(e, model)
        if flat is not None:
            terms.append((scale, (flat,)))
            return True
        if isinstance(e, UnaryOp):
            if e.op == "neg":
                return visit(e.operand, -scale)
            return False
        if isinstance(e, SumExpression):
            return visit(e.operand, scale)
        if isinstance(e, BinaryOp):
            if e.op == "+":
                return visit(e.left, scale) and visit(e.right, scale)
            if e.op == "-":
                return visit(e.left, scale) and visit(e.right, -scale)
            if e.op == "/":
                if isinstance(e.right, Constant):
                    denom = float(e.right.value)
                    if denom == 0.0:
                        return False
                    return visit(e.left, scale / denom)
                return False
            if e.op == "*":
                decomp = _product_to_monomial(e, model)
                if decomp is None:
                    return False
                coeff, monomial = decomp
                terms.append((scale * coeff, monomial))
                return True
            if e.op == "**":
                decomp = _product_to_monomial(e, model)
                if decomp is None:
                    return False
                coeff, monomial = decomp
                terms.append((scale * coeff, monomial))
                return True
        return False

    if not visit(expr, 1.0):
        return None
    return const[0], terms


def _extract_affine_square(expr: Expression, model: Model) -> _AffineSquare | None:
    """Recognize ``E**2`` where ``E`` is an affine combination of liftable products.

    Returns the residual polynomial ``(const, terms)`` for ``E`` when ``E``
    distributes into a sum of at least two addends with at least one genuinely
    nonlinear term (a product or power ≥ 2) — the shape that, if the square were
    distributed, would blow up into high-degree monomials (issue #155). Returns
    ``None`` for transcendental residuals, pure quadratic forms of a linear
    residual, and single-term squares (handled by the existing pipelines).
    """
    if not (
        isinstance(expr, BinaryOp)
        and expr.op == "**"
        and isinstance(expr.right, Constant)
        and float(expr.right.value) == 2.0
    ):
        return None
    base = distribute_products(expr.left)
    poly = _expr_to_polynomial(base, model)
    if poly is None:
        return None
    const, terms = poly
    if not any(len(monomial) >= 2 for _coeff, monomial in terms):
        return None
    n_addends = len(terms) + (1 if const != 0.0 else 0)
    if n_addends < 2:
        return None
    return const, terms


def _collect_affine_squares(model: Model) -> list[tuple[Expression, _AffineSquare]]:
    """Find every ``E**2`` affine-square node in the objective and constraints."""
    found: list[tuple[Expression, _AffineSquare]] = []
    seen: set[int] = set()

    def visit(e: Expression) -> None:
        info = _extract_affine_square(e, model)
        if info is not None and id(e) not in seen:
            seen.add(id(e))
            found.append((e, info))
            # The residual is lifted wholesale; do not descend into it.
            return
        if isinstance(e, BinaryOp):
            visit(e.left)
            visit(e.right)
        elif isinstance(e, UnaryOp):
            visit(e.operand)
        elif isinstance(e, FunctionCall):
            for arg in e.args:
                visit(arg)
        elif isinstance(e, IndexExpression):
            if not isinstance(e.base, Variable):
                visit(e.base)
        elif isinstance(e, SumExpression):
            visit(e.operand)
        elif isinstance(e, SumOverExpression):
            for term in e.terms:
                visit(term)

    if model._objective is not None:
        visit(model._objective.expression)
    for constraint in model._constraints:
        visit(constraint.body)
    return found


def _collect_affine_powers(
    model: Model, already_lifted: set[int]
) -> list[tuple[Expression, float, int, int]]:
    """Find ``(c*x)**n`` nodes (integer ``n >= 3``, ``c*x`` a scaled variable).

    Returns ``(node, scale, var_idx, power)`` for each distinct node whose base
    is a *non-bare* scaled single variable — a bare ``x**n`` is left to the
    standard monomial machinery (scaling it would not improve the aux-column
    conditioning).  Nodes already claimed by an issue-#155 affine-square lift
    (``already_lifted``) are skipped.
    """
    found: list[tuple[Expression, float, int, int]] = []
    seen: set[int] = set()

    def visit(e: Expression) -> None:
        if id(e) in already_lifted:
            return
        if isinstance(e, BinaryOp) and e.op == "**" and isinstance(e.right, Constant):
            exp_f = float(e.right.value)
            n = int(exp_f)
            # Only a genuinely scaled base benefits: a bare variable base lifts
            # identically in raw space, so leave it to the monomial path.
            if exp_f == n and n >= 3 and _get_flat_index(e.left, model) is None:
                base = _affine_var_base(e.left, model)
                if base is not None and id(e) not in seen:
                    scale, var_idx = base
                    seen.add(id(e))
                    found.append((e, float(scale), int(var_idx), n))
                    return  # the whole node is lifted; do not descend
        if isinstance(e, BinaryOp):
            visit(e.left)
            visit(e.right)
        elif isinstance(e, UnaryOp):
            visit(e.operand)
        elif isinstance(e, FunctionCall):
            for arg in e.args:
                visit(arg)
        elif isinstance(e, IndexExpression):
            if not isinstance(e.base, Variable):
                visit(e.base)
        elif isinstance(e, SumExpression):
            visit(e.operand)
        elif isinstance(e, SumOverExpression):
            for term in e.terms:
                visit(term)

    if model._objective is not None:
        visit(model._objective.expression)
    for constraint in model._constraints:
        visit(constraint.body)
    return found


def _build_minmax_objective_lift(
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[MinMaxObjectiveLift]:
    if model._objective is None:
        return None
    expr = model._objective.expression
    if not isinstance(expr, FunctionCall) or len(expr.args) < 2:
        return None
    if model._objective.sense == ObjectiveSense.MINIMIZE and expr.func_name != "max":
        return None
    if model._objective.sense == ObjectiveSense.MAXIMIZE and expr.func_name != "min":
        return None
    if expr.func_name not in {"min", "max"}:
        return None

    branch_exprs = tuple(_expand_integer_powers_for_relaxation(arg, model) for arg in expr.args)
    branch_bounds = tuple(
        (
            _expression_lower_bound_for_lift(branch, model, flat_lb, flat_ub),
            _expression_upper_bound_for_lift(branch, model, flat_lb, flat_ub),
        )
        for branch in branch_exprs
    )

    lower_bounds = [lb for lb, _ub in branch_bounds if lb is not None]
    upper_bounds = [ub for _lb, ub in branch_bounds if ub is not None]
    aux_lb: Optional[float]
    aux_ub: Optional[float]
    if expr.func_name == "max":
        # max(f_i) is at least any available lower bound on a branch.
        aux_lb = max(lower_bounds) if lower_bounds else None
        # max(f_i) is at most max(ub_i) only when every branch has an upper bound.
        aux_ub = max(upper_bounds) if len(upper_bounds) == len(branch_bounds) else None
        directional_bound = aux_lb
    else:
        # min(f_i) is at least min(lb_i) only when every branch has a lower bound.
        aux_lb = min(lower_bounds) if len(lower_bounds) == len(branch_bounds) else None
        # min(f_i) is at most any available upper bound on a branch.
        aux_ub = min(upper_bounds) if upper_bounds else None
        directional_bound = aux_ub

    directional_bound = _finite_bound_or_none(directional_bound)
    if directional_bound is None:
        return None

    lb = _finite_bound_or_none(aux_lb)
    ub = _finite_bound_or_none(aux_ub)
    aux_bounds = (
        lb if lb is not None else -_EFFECTIVE_INF,
        ub if ub is not None else _EFFECTIVE_INF,
    )
    if aux_bounds[0] > aux_bounds[1] + 1e-9:
        return None
    if aux_bounds[0] > aux_bounds[1]:
        mid = 0.5 * (aux_bounds[0] + aux_bounds[1])
        aux_bounds = (mid, mid)

    return MinMaxObjectiveLift(
        func_name=expr.func_name,
        aux_col=-1,
        branch_exprs=branch_exprs,
        branch_bounds=branch_bounds,
        aux_bounds=aux_bounds,
    )


def _normalize_convhull_formulation(formulation: str) -> str:
    """Normalize accepted bilinear convex-hull mode names."""
    aliases = {
        "disaggregated": "disaggregated",
        "piecewise": "disaggregated",
        "sos2": "sos2",
        "facet": "facet",
        "lambda": "sos2",
    }
    try:
        return aliases[formulation]
    except KeyError as err:
        raise ValueError(
            f"Unsupported convhull_formulation: {formulation!r}. "
            "Choose from 'disaggregated', 'sos2', 'facet', or 'lambda'."
        ) from err


def _sorted_unique_points(points: list[float]) -> list[float]:
    """Return sorted points with near-duplicates removed."""
    unique: list[float] = []
    for point in sorted(float(p) for p in points):
        if not unique or abs(point - unique[-1]) > 1e-12:
            unique.append(point)
    return unique


def _power_tangent_line(t: float, n: int) -> tuple[float, float]:
    """Return slope/intercept for the tangent to x**n at x=t."""
    slope = float(n * (t ** (n - 1)))
    intercept = float((t**n) - slope * t)
    return slope, intercept


def _power_secant_line(lb: float, ub: float, n: int) -> tuple[float, float]:
    """Return slope/intercept for the secant through (lb, lb**n) and (ub, ub**n)."""
    if abs(ub - lb) <= 1e-12:
        return 0.0, float(lb**n)
    slope = float((ub**n - lb**n) / (ub - lb))
    intercept = float(lb**n - slope * lb)
    return slope, intercept


def _power_is_convex_on_box(n: int, lb: float) -> bool:
    """Return True when x**n is convex on the current box."""
    return n % 2 == 0 or lb >= 0.0


def _monomial_breakpoints(
    var_idx: int,
    lb_i: float,
    ub_i: float,
    disc_state: DiscretizationState,
) -> list[float]:
    """Return refinement-aware monomial cut points, including zero when needed."""
    if var_idx in disc_state.partitions and len(disc_state.partitions[var_idx]) >= 2:
        points = [float(p) for p in disc_state.partitions[var_idx]]
    else:
        points = [lb_i, ub_i]
    if lb_i < 0.0 < ub_i:
        points.append(0.0)
    return _sorted_unique_points(points)


def _odd_mixed_tangent_is_valid(
    t: float,
    lb: float,
    ub: float,
    n: int,
    kind: str,
) -> bool:
    """Check whether the tangent at t is a global under/over-estimator on [lb, ub]."""
    slope, intercept = _power_tangent_line(t, n)
    critical_points = [lb, ub, t]
    mirrored = -t
    if lb <= mirrored <= ub:
        critical_points.append(mirrored)

    diffs = [float(x**n - (slope * x + intercept)) for x in _sorted_unique_points(critical_points)]
    tol = 1e-10
    if kind == "under":
        return all(diff >= -tol for diff in diffs)
    if kind == "over":
        return all(diff <= tol for diff in diffs)
    raise ValueError(f"Unknown tangent validity kind: {kind}")


def _choose_trilinear_pair(
    term: tuple[int, int, int],
    partitioned_vars: set[int],
) -> tuple[tuple[int, int], int]:
    """Choose a deterministic trilinear decomposition pair.

    Prefer a pair that includes as many currently partitioned original variables as
    possible so the first or second lifted bilinear term can reuse the stronger
    piecewise relaxation machinery already present for bilinear terms.
    """
    i, j, k = tuple(sorted(term))
    candidates = [((i, j), k), ((i, k), j), ((j, k), i)]
    candidates.sort()
    return max(
        candidates,
        key=lambda item: (
            sum(v in partitioned_vars for v in item[0]),
            item[0][0] in partitioned_vars or item[0][1] in partitioned_vars,
        ),
    )


# ---------------------------------------------------------------------------
# Helpers: expression decomposition
# ---------------------------------------------------------------------------


def _decompose_product(
    expr: Expression,
    model: Model,
    fractional_power_var_map: Optional[dict[tuple[int, float], int]] = None,
    univariate_var_map: Optional[dict[object, int]] = None,
    monomial_var_map: Optional[dict[tuple[int, int], int]] = None,
    composite_var_map: Optional[dict[int, int]] = None,
    composite_coeff_map: Optional[dict[int, float]] = None,
) -> tuple[float, list[int]] | None:
    """Decompose a product expression into (scalar, [flat_or_aux_idx, ...]).

    Returns None if expr contains non-constant, non-variable leaves.
    Constants are accumulated into the scalar; variable references and
    registered lifted sub-expressions are appended to the index list (using
    their MILP column indices).

    When ``monomial_var_map`` is supplied, a *mixed repeated-factor* product
    such as ``x*x*y`` is collapsed: the repeated original-variable group ``x*x``
    is replaced by its monomial aux column (``x**2``), leaving ``[col(x**2), y]``
    — a lifted bilinear pair. This lets the standard McCormick pipeline relax
    ``x**2 * y`` (one monomial envelope + one bilinear envelope) instead of
    rejecting it as an unsupported repeated-factor term.
    """
    scalar: list[float] = [1.0]
    var_indices: list[int] = []

    def visit(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op == "*":
            return visit(e.left) and visit(e.right)
        if isinstance(e, Constant):
            scalar[0] *= float(e.value)
            return True
        flat = _get_flat_index(e, model)
        if flat is not None:
            var_indices.append(flat)
            return True
        if univariate_var_map:
            aux_col = univariate_var_map.get(id(e))
            if aux_col is not None:
                var_indices.append(aux_col)
                return True
        if composite_var_map:
            aux_col = composite_var_map.get(id(e))
            if aux_col is not None:
                # A composite node carrying a non-unit substitution coefficient
                # (e.g. a ratio-of-products aux scaled by the numerator constant,
                # issue #185) cannot be represented as a plain product factor here;
                # abstain so the linearizer's coefficient-aware path handles it.
                if composite_coeff_map and composite_coeff_map.get(id(e), 1.0) != 1.0:
                    return False
                var_indices.append(aux_col)
                return True
        # Recognize var^p (fractional p) when an aux column was allocated.
        if (
            fractional_power_var_map
            and isinstance(e, BinaryOp)
            and e.op == "**"
            and isinstance(e.right, Constant)
        ):
            base_flat = _get_flat_index(e.left, model)
            if base_flat is not None:
                key = (base_flat, float(e.right.value))
                if key in fractional_power_var_map:
                    var_indices.append(fractional_power_var_map[key])
                    return True
        # Fold a *composite* variable-free factor (e.g. ``neg(1e6)`` from a
        # parsed ``-1e6*i1*i2``, or ``(-3)*(-3)``) into the scalar. A bare
        # ``Constant`` is handled above; this catches negations/arithmetic over
        # constants that would otherwise look like an undecomposable factor and
        # cause the whole product (and its constraint) to be dropped from the
        # relaxation. Exact and variable-free, so it only ever tightens.
        cval = _eval_constant_expr(e)
        if cval is not None:
            scalar[0] *= cval
            return True
        return False

    if not visit(expr):
        return None

    # Collapse mixed repeated-factor groups (x*x*y) into monomial aux columns
    # so the product reduces to distinct lifted factors the McCormick pipeline
    # can relax. Pure monomials (x*x with a single unique base) are left intact
    # for the dedicated monomial branch in the linearizer.
    if monomial_var_map:
        n_orig = sum(v.size for v in model._variables)
        counts: dict[int, int] = {}
        for i in var_indices:
            if i < n_orig:
                counts[i] = counts.get(i, 0) + 1
        repeated = {i for i, c in counts.items() if c >= 2}
        if repeated and len(set(var_indices)) >= 2:
            collapsed: list[int] = []
            seen: set[int] = set()
            ok = True
            for i in var_indices:
                if i in repeated:
                    if i in seen:
                        continue
                    col = monomial_var_map.get((i, counts[i]))
                    if col is None:
                        ok = False
                        break
                    collapsed.append(col)
                    seen.add(i)
                else:
                    collapsed.append(i)
            if ok:
                var_indices = collapsed

    return scalar[0], var_indices


def _decompose_signed_monomial(expr: Expression, model: Model) -> tuple[float, list[int]] | None:
    """Decompose a signed product / quotient-by-constant into ``(coeff, indices)``.

    Folds every variable-free subexpression (including composite constants such as
    gear4's ``neg(1e6)``) into ``coeff`` and collects the flat indices of the
    original-variable factors (with repeats for integer powers ``x**n``,
    ``2 ≤ n ≤ 4``). Returns ``None`` the moment a non-constant, non-variable,
    non-product/​power leaf — a transcendental call, an additive operand, or a
    division by a variable — is encountered, so the caller only ever sees a
    genuine signed monomial in original variables.
    """
    const = _eval_constant_expr(expr)
    if const is not None:
        return const, []
    flat = _get_flat_index(expr, model)
    if flat is not None:
        return 1.0, [flat]
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        sub = _decompose_signed_monomial(expr.operand, model)
        if sub is None:
            return None
        c, idx = sub
        return -c, idx
    if isinstance(expr, BinaryOp) and expr.op == "*":
        left = _decompose_signed_monomial(expr.left, model)
        if left is None:
            return None
        right = _decompose_signed_monomial(expr.right, model)
        if right is None:
            return None
        return left[0] * right[0], left[1] + right[1]
    if isinstance(expr, BinaryOp) and expr.op == "/":
        denom = _eval_constant_expr(expr.right)
        if denom is None or denom == 0.0:
            return None
        sub = _decompose_signed_monomial(expr.left, model)
        if sub is None:
            return None
        return sub[0] / denom, sub[1]
    if isinstance(expr, BinaryOp) and expr.op == "**" and isinstance(expr.right, Constant):
        p = _eval_constant_expr(expr.right)
        if p is not None and p.is_integer() and 2 <= int(p) <= 4:
            base = _decompose_signed_monomial(expr.left, model)
            if base is None:
                return None
            bc, bidx = base
            return bc ** int(p), bidx * int(p)
    return None


def _collect_lifted_bilinear_products(
    model: Model,
    fractional_power_var_map: dict[tuple[int, float], int],
    univariate_var_map: dict[object, int],
    n_orig: int,
    monomial_var_map: Optional[dict[tuple[int, int], int]] = None,
    composite_var_map: Optional[dict[int, int]] = None,
) -> list[tuple[int, int]]:
    """Return products between original variables and lifted auxiliary columns."""
    keys: set[tuple[int, int]] = set()

    def visit(expr: Expression) -> None:
        if isinstance(expr, BinaryOp):
            if expr.op == "*":
                decomp = _decompose_product(
                    expr,
                    model,
                    fractional_power_var_map=fractional_power_var_map,
                    univariate_var_map=univariate_var_map,
                    monomial_var_map=monomial_var_map,
                    composite_var_map=composite_var_map,
                )
                if decomp is not None:
                    _scalar, indices = decomp
                    unique = list(dict.fromkeys(indices))
                    if (
                        len(unique) == 2
                        and len(indices) == 2
                        and any(idx >= n_orig for idx in unique)
                    ):
                        i, j = sorted(unique)
                        keys.add((i, j))
            visit(expr.left)
            visit(expr.right)
            return

        if isinstance(expr, UnaryOp):
            visit(expr.operand)
            return

        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                visit(arg)
            return

        if isinstance(expr, IndexExpression):
            if not isinstance(expr.base, Variable):
                visit(expr.base)
            return

        if isinstance(expr, SumExpression):
            visit(expr.operand)
            return

        if isinstance(expr, SumOverExpression):
            for term in expr.terms:
                visit(term)

    if model._objective is not None:
        visit(distribute_products(model._objective.expression))
    for constraint in model._constraints:
        visit(distribute_products(constraint.body))

    return sorted(keys)


def _collect_lifted_higher_products(
    model: Model,
    fractional_power_var_map: dict[tuple[int, float], int],
    univariate_var_map: dict[object, int],
    n_orig: int,
    monomial_var_map: Optional[dict[tuple[int, int], int]] = None,
    composite_var_map: Optional[dict[int, int]] = None,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, ...]]]:
    """Return trilinear/multilinear products that involve a lifted aux column.

    The trilinear/multilinear allocation loops in :func:`build_milp_relaxation`
    are populated only from ``terms.trilinear`` / ``terms.multilinear``, which
    the term classifier records over *original* variables. A product such as
    ``x**2 * y * z`` is dumped into ``general_nl``; after :func:`_decompose_product`
    collapses ``x*x`` into its monomial aux column it becomes a three-distinct-column
    product ``[col(x**2), y, z]`` whose key never appears in those classifier
    sets, so the linearizer raised ``"Trilinear (i,j,k) not in map"`` and the
    whole objective/constraint dropped (issue #139, bucket 2).

    This collector walks the objective and constraints exactly like
    :func:`_collect_lifted_bilinear_products` and returns the distinct-column
    products with no repeated factor (``len(indices) == len(unique)``) where at
    least one factor is a lifted aux column (``idx >= n_orig``). They are split
    by arity into trilinear (three columns) and multilinear (four or more).
    The recursive bilinear chain that relaxes them is already proven sound:
    each stage is a standard McCormick envelope on interval-arithmetic bounds.
    """
    trilinear: set[tuple[int, int, int]] = set()
    multilinear: set[tuple[int, ...]] = set()

    def visit(expr: Expression) -> None:
        if isinstance(expr, BinaryOp):
            if expr.op == "*":
                decomp = _decompose_product(
                    expr,
                    model,
                    fractional_power_var_map=fractional_power_var_map,
                    univariate_var_map=univariate_var_map,
                    monomial_var_map=monomial_var_map,
                    composite_var_map=composite_var_map,
                )
                if decomp is not None:
                    _scalar, indices = decomp
                    unique = list(dict.fromkeys(indices))
                    if (
                        len(unique) == len(indices)
                        and len(unique) >= 3
                        and any(idx >= n_orig for idx in unique)
                    ):
                        ordered = tuple(sorted(unique))
                        if len(ordered) == 3:
                            trilinear.add(ordered)  # type: ignore[arg-type]
                        else:
                            multilinear.add(ordered)
            visit(expr.left)
            visit(expr.right)
            return

        if isinstance(expr, UnaryOp):
            visit(expr.operand)
            return

        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                visit(arg)
            return

        if isinstance(expr, IndexExpression):
            if not isinstance(expr.base, Variable):
                visit(expr.base)
            return

        if isinstance(expr, SumExpression):
            visit(expr.operand)
            return

        if isinstance(expr, SumOverExpression):
            for term in expr.terms:
                visit(term)

    if model._objective is not None:
        visit(distribute_products(model._objective.expression))
    for constraint in model._constraints:
        visit(distribute_products(constraint.body))

    return sorted(trilinear), sorted(multilinear, key=lambda t: (len(t), t))


# Univariate functions whose superposition cuts are supported: smooth on any
# box the lifted aux already validated, so the Chebyshev kernel encloses them
# rigorously. ``abs`` (non-smooth) and ``tan`` (poles) are deliberately omitted.
_SUPERPOSITION_FUNCS = {
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "reciprocal",
    "sin",
    "cos",
}


def _superposition_jax_func(func_name: str):
    """Return a jax-traceable callable for ``func_name``, or ``None`` if unsupported."""
    import jax.numpy as jnp

    table = {
        "exp": jnp.exp,
        "log": jnp.log,
        "log2": lambda t: jnp.log(t) / jnp.log(2.0),
        "log10": lambda t: jnp.log(t) / jnp.log(10.0),
        "sqrt": jnp.sqrt,
        "reciprocal": lambda t: 1.0 / t,
        "sin": jnp.sin,
        "cos": jnp.cos,
    }
    return table.get(func_name)


def _add_superposition_cuts(
    add_row,
    univariate_by_aux_col: dict[int, "UnivariateRelaxation"],
    bilinear_var_map: dict[tuple[int, int], int],
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    n_orig: int,
    n_total: int,
) -> None:
    """Inject rigorous interior-reference cuts for lifted ``w = f(arg)*y`` products.

    For each bilinear aux ``w`` whose factors are a lifted univariate aux
    ``u = f(arg)`` (``arg`` affine in the original variables) and an original
    variable ``y``, emit the superposition cut family from
    :mod:`discopt._jax.superposition`. Each cut is an individually valid global
    bound on the true product surface, so the LP stays a sound lower-bounding
    relaxation. Any term that cannot be handled safely is skipped (never relaxed
    incorrectly): unsupported function, shared variable between ``arg`` and
    ``y``, degenerate box, or a non-finite cut coefficient.
    """
    from discopt._jax.superposition import (
        BilinearNonlinearTerm,
        bilinear_nonlinear_cuts,
        superposition_references,
    )

    for (i, j), w_col in bilinear_var_map.items():
        # Identify the univariate-aux factor and the original-variable factor.
        if i in univariate_by_aux_col and j < n_orig:
            relax = univariate_by_aux_col[i]
            y_col = j
        elif j in univariate_by_aux_col and i < n_orig:
            relax = univariate_by_aux_col[j]
            y_col = i
        else:
            continue

        func = _superposition_jax_func(relax.func_name)
        if func is None:
            continue

        arg_coeff = np.asarray(relax.arg_coeff, dtype=np.float64)
        # ``y`` must be independent of the argument of ``f`` for the bilinear
        # split to be valid; skip if it appears in the affine argument.
        if y_col < len(arg_coeff) and arg_coeff[y_col] != 0.0:
            continue

        x_lb, x_ub = float(relax.arg_lb), float(relax.arg_ub)
        y_lb, y_ub = float(flat_lb[y_col]), float(flat_ub[y_col])
        if not (x_lb < x_ub and y_lb < y_ub):
            continue
        if not (
            np.isfinite(x_lb) and np.isfinite(x_ub) and np.isfinite(y_lb) and np.isfinite(y_ub)
        ):
            continue

        try:
            term = BilinearNonlinearTerm(func, (x_lb, x_ub), (y_lb, y_ub))
            refs = superposition_references((x_lb, x_ub), (y_lb, y_ub))
            cuts = bilinear_nonlinear_cuts(term, refs)
        except (ValueError, ArithmeticError, FloatingPointError):
            continue

        arg_const = float(relax.arg_const)
        for cut in cuts:
            # Local cut over (t, y, w): coeffs = [-ax, -ay, 1] with t the
            # argument of f. Substitute t = arg_coeff·x + arg_const.
            neg_ax, neg_ay, aw = (float(c) for c in cut.coeffs)
            row = np.zeros(n_total, dtype=np.float64)
            row[:n_orig] += neg_ax * arg_coeff[:n_orig]
            row[y_col] += neg_ay
            row[w_col] += aw
            rhs = float(cut.rhs) - neg_ax * arg_const
            if not np.all(np.isfinite(row)) or not np.isfinite(rhs):
                continue
            # ``add_row(coeff, rhs)`` encodes ``coeff·z <= rhs``.
            if cut.sense == "<=":
                add_row(row, rhs)
            elif cut.sense == ">=":
                add_row(-row, -rhs)


def _collect_distinct_multilinear_products(model: Model) -> list[tuple[int, ...]]:
    """Return distinct-variable product terms with four or more factors."""
    terms: set[tuple[int, ...]] = set()

    def visit(expr: Expression) -> None:
        if isinstance(expr, BinaryOp):
            if expr.op == "*":
                decomp = _decompose_product(expr, model)
                if decomp is not None:
                    _, indices = decomp
                    unique = list(dict.fromkeys(indices))
                    if len(unique) >= 4 and len(unique) == len(indices):
                        terms.add(tuple(sorted(unique)))
                        return
            visit(expr.left)
            visit(expr.right)
            return

        if isinstance(expr, UnaryOp):
            visit(expr.operand)
            return

        if isinstance(expr, SumExpression):
            visit(expr.operand)
            return

        if isinstance(expr, SumOverExpression):
            for term in expr.terms:
                visit(term)
            return

        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                visit(arg)

    if model._objective is not None:
        visit(model._objective.expression)
    for constraint in model._constraints:
        visit(constraint.body)

    return sorted(terms)


def _linear_constraint_forms(model: Model, n_vars: int) -> list[tuple[np.ndarray, float]]:
    """Return each *linear* model constraint as ``(coeff, const)`` meaning the
    valid inequality ``coeff . x + const <= 0``.

    Constraints are stored as ``body <= 0`` or ``body == 0`` (``>=`` is
    normalized to ``<=`` by the expression operators), so a ``<=`` constraint
    contributes one form and an ``==`` constraint contributes both ``g <= 0`` and
    ``-g <= 0``. Nonlinear constraint bodies are skipped (they have no affine
    form). These are the factors level-1 RLT multiplies by variable bound
    factors to generate valid product cuts.
    """
    forms: list[tuple[np.ndarray, float]] = []
    for constraint in model._constraints:
        try:
            coeff, const = _linearize_affine_expr(constraint.body, model, n_vars)
        except ValueError:
            continue  # nonlinear body — not an affine factor
        if not (np.all(np.isfinite(coeff)) and np.isfinite(const)):
            continue
        sense = constraint.sense
        if sense == "<=":
            forms.append((coeff, float(const)))
        elif sense == "==":
            forms.append((coeff, float(const)))
            forms.append((-coeff, -float(const)))
    return forms


# A quadratic constraint factor for level-1 RLT (issue #15, Phase 2): the body
# ``g(x) = const + sum_i lin_i x_i + sum_{(k,l)} quad_{kl} x_k x_l`` with the sense
# of the parent constraint. ``quad`` keys are sorted index pairs ``(k, l)``,
# ``k <= l`` (``(k, k)`` is the square ``x_k**2``).
_QuadForm = tuple[dict[tuple[int, int], float], dict[int, float], float, str]


def _quadratic_constraint_forms(model: Model, n_vars: int) -> list[_QuadForm]:
    """Return each *genuinely quadratic* (degree-exactly-2 polynomial) model
    constraint as ``(quad, lin, const, sense)``.

    These are the nonlinear factors that Phase-2 level-1 RLT multiplies by
    variable bound factors. Purely linear bodies are skipped (the affine path in
    :func:`_linear_constraint_forms` already handles them); cubic-or-higher and
    non-polynomial (transcendental, fractional-power) bodies are skipped because
    their RLT products are out of scope for the degree-3 lifting implemented
    here. The parent ``sense`` (``"<="`` or ``"=="``) is carried so an equality
    parent can emit a two-sided equality product row.
    """
    forms: list[_QuadForm] = []
    for constraint in model._constraints:
        if constraint.sense not in ("<=", "=="):
            continue
        poly = _expr_to_polynomial(distribute_products(constraint.body), model)
        if poly is None:
            continue
        const, terms = poly
        quad: dict[tuple[int, int], float] = {}
        lin: dict[int, float] = {}
        const_acc = float(const)
        max_degree = 0
        ok = True
        for coeff, monomial in terms:
            degree = len(monomial)
            max_degree = max(max_degree, degree)
            if degree == 0:
                const_acc += coeff
            elif degree == 1:
                idx = monomial[0]
                if idx >= n_vars:
                    ok = False
                    break
                lin[idx] = lin.get(idx, 0.0) + coeff
            elif degree == 2:
                ka, kb = monomial  # already sorted by _product_to_monomial
                if ka >= n_vars or kb >= n_vars:
                    ok = False
                    break
                quad[(ka, kb)] = quad.get((ka, kb), 0.0) + coeff
            else:
                ok = False  # degree >= 3 body: out of scope for quadratic-factor RLT
                break
        if not ok or max_degree != 2:
            continue
        if not (np.isfinite(const_acc) and all(np.isfinite(v) for v in lin.values())):
            continue
        if not all(np.isfinite(v) for v in quad.values()):
            continue
        forms.append((quad, lin, const_acc, constraint.sense))
    return forms


def _linearize_affine_expr(expr: Expression, model: Model, n_vars: int) -> tuple[np.ndarray, float]:
    """Linearize an affine expression over original variables.

    Raises ValueError when the expression contains nonlinear structure.  This is
    intentionally narrower than _linearize_expr because univariate operator
    relaxations are only soundly supported here for affine arguments.
    """
    coeff = np.zeros(n_vars, dtype=np.float64)
    const_acc: list[float] = [0.0]

    def visit(e: Expression, scale: float) -> None:
        if isinstance(e, Constant):
            const_acc[0] += scale * float(e.value)
            return

        if isinstance(e, Variable):
            offset = _compute_var_offset(e, model)
            if e.size == 1:
                coeff[offset] += scale
                return
            raise ValueError(f"Cannot use array variable as scalar affine argument: {e}")

        if isinstance(e, IndexExpression):
            flat = _get_flat_index(e, model)
            if flat is None:
                raise ValueError(f"Cannot linearize IndexExpression: {e}")
            coeff[flat] += scale
            return

        if isinstance(e, UnaryOp) and e.op == "neg":
            visit(e.operand, -scale)
            return

        if isinstance(e, BinaryOp):
            if e.op == "+":
                visit(e.left, scale)
                visit(e.right, scale)
                return
            if e.op == "-":
                visit(e.left, scale)
                visit(e.right, -scale)
                return
            if e.op == "*":
                if isinstance(e.left, Constant):
                    visit(e.right, scale * float(e.left.value))
                    return
                if isinstance(e.right, Constant):
                    visit(e.left, scale * float(e.right.value))
                    return
                raise ValueError(f"Non-affine product in univariate argument: {e}")
            if e.op == "/":
                if isinstance(e.right, Constant):
                    visit(e.left, scale / float(e.right.value))
                    return
                raise ValueError(f"Non-affine division in univariate argument: {e}")
            if e.op == "**":
                if isinstance(e.right, Constant):
                    exp = float(e.right.value)
                    if exp == 1.0:
                        visit(e.left, scale)
                        return
                    if exp == 0.0:
                        const_acc[0] += scale
                        return
                raise ValueError(f"Non-affine power in univariate argument: {e}")

        if isinstance(e, SumExpression):
            op = e.operand
            if isinstance(op, Variable):
                offset = _compute_var_offset(op, model)
                for k in range(op.size):
                    coeff[offset + k] += scale
                return
            visit(op, scale)
            return

        if isinstance(e, SumOverExpression):
            for term in e.terms:
                visit(term, scale)
            return

        raise ValueError(f"Unsupported affine argument node {type(e).__name__}: {e}")

    visit(expr, 1.0)
    return coeff, const_acc[0]


def _univariate_arg(expr: Expression) -> tuple[str, Expression] | None:
    """Return (operator_name, argument) for supported univariate nodes."""
    if isinstance(expr, FunctionCall) and len(expr.args) == 1:
        name = expr.func_name
        if name in {
            "sqrt",
            "log",
            "log2",
            "log10",
            "exp",
            "abs",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "acosh",
            "entropy",
        }:
            return name, expr.args[0]
    if isinstance(expr, UnaryOp) and expr.op == "abs":
        return "abs", expr.operand
    if isinstance(expr, BinaryOp) and expr.op == "/" and _constant_value(expr.left) is not None:
        return "reciprocal", expr.right
    return None


def _univariate_value(func_name: str, x: float) -> float:
    """Evaluate a supported scalar univariate function."""
    if func_name == "sqrt":
        return float(np.sqrt(x))
    if func_name == "log":
        return float(np.log(x))
    if func_name == "log2":
        return float(np.log2(x))
    if func_name == "log10":
        return float(np.log10(x))
    if func_name == "exp":
        return float(np.exp(x))
    if func_name == "abs":
        return float(abs(x))
    if func_name == "reciprocal":
        return float(1.0 / x)
    if func_name == "sin":
        return float(np.sin(x))
    if func_name == "cos":
        return float(np.cos(x))
    if func_name == "tan":
        return float(np.tan(x))
    if func_name == "asin":
        return float(np.arcsin(x))
    if func_name == "acos":
        return float(np.arccos(x))
    if func_name == "acosh":
        return float(np.arccosh(x))
    if func_name == "entropy":
        # entropy(x) = x*log(x), with the x -> 0+ limit equal to 0.
        if x <= 0.0:
            return 0.0
        return float(x * np.log(x))
    raise ValueError(f"Unsupported univariate function: {func_name}")


def _univariate_grad(func_name: str, x: float) -> float:
    """Evaluate the first derivative of a smooth supported univariate function."""
    if func_name == "sqrt":
        return float(0.5 / np.sqrt(x))
    if func_name == "log":
        return float(1.0 / x)
    if func_name == "log2":
        return float(1.0 / (x * np.log(2.0)))
    if func_name == "log10":
        return float(1.0 / (x * np.log(10.0)))
    if func_name == "exp":
        return float(np.exp(x))
    if func_name == "reciprocal":
        return float(-1.0 / (x * x))
    if func_name == "sin":
        return float(np.cos(x))
    if func_name == "cos":
        return float(-np.sin(x))
    if func_name == "tan":
        c = float(np.cos(x))
        return float(1.0 / (c * c))
    if func_name == "asin":
        return float(1.0 / np.sqrt(1.0 - x * x))
    if func_name == "acos":
        return float(-1.0 / np.sqrt(1.0 - x * x))
    if func_name == "acosh":
        return float(1.0 / np.sqrt(x * x - 1.0))
    if func_name == "entropy":
        # d/dx [x*log(x)] = log(x) + 1 (finite only for x > 0).
        return float(np.log(x) + 1.0)
    raise ValueError(f"No smooth derivative for univariate function: {func_name}")


def _tan_domain_ok(arg_lb: float, arg_ub: float) -> bool:
    """Return True when ``tan`` is finite and continuous on the interval."""
    if not np.isfinite(arg_lb) or not np.isfinite(arg_ub) or arg_lb > arg_ub:
        return False
    half_pi = 0.5 * np.pi
    k = np.ceil((arg_lb - half_pi) / np.pi)
    asymptote = half_pi + k * np.pi
    if arg_lb <= asymptote <= arg_ub:
        return False
    return all(_is_effectively_finite(np.tan(x)) for x in (arg_lb, arg_ub))


def _univariate_domain_ok(func_name: str, arg_lb: float, arg_ub: float) -> bool:
    """Return True when the operator can be relaxed on the interval."""
    if not np.isfinite(arg_lb) or not np.isfinite(arg_ub) or arg_lb > arg_ub:
        return False
    if func_name == "sqrt" and arg_lb < 0.0:
        return False
    if func_name in {"log", "log2", "log10"} and arg_lb <= 0.0:
        return False
    if func_name in {"sqrt", "log", "log2", "log10"}:
        return True
    if func_name == "entropy":
        # entropy(x) = x*log(x) is finite on x >= 0 (limit 0 at x = 0); a smooth
        # tangent underestimator needs at least one strictly positive point.
        return bool(arg_lb >= 0.0 and arg_ub > 0.0)
    if func_name == "exp":
        return bool(arg_ub <= _MAX_FINITE_EXP_ARG)
    if func_name == "abs":
        return True
    if func_name == "reciprocal":
        return bool(arg_lb > 0.0)
    if func_name in {"sin", "cos"}:
        return True
    if func_name == "tan":
        return _tan_range(arg_lb, arg_ub) is not None
    if func_name in {"asin", "acos"}:
        # Defined on [-1, 1]; the relaxation is built on the closed interval and
        # the singular endpoint slopes are skipped (see ``_tangent_points``).
        return bool(arg_lb >= -1.0 and arg_ub <= 1.0)
    if func_name == "acosh":
        # acosh is defined and real on x >= 1 (concave there).
        return bool(arg_lb >= 1.0)
    return False


def _univariate_value_bounds(func_name: str, arg_lb: float, arg_ub: float) -> tuple[float, float]:
    """Return finite bounds for f(x) on [arg_lb, arg_ub]."""
    if func_name == "abs":
        if arg_lb <= 0.0 <= arg_ub:
            return 0.0, max(abs(arg_lb), abs(arg_ub))
        values = [abs(arg_lb), abs(arg_ub)]
        return min(values), max(values)
    if func_name in {"sin", "cos", "tan"}:
        bounds = _trig_range(func_name, arg_lb, arg_ub)
        if bounds is None:
            return np.nan, np.nan
        return bounds
    if func_name == "entropy":
        # entropy(x) = x*log(x) is convex with a single interior minimum at
        # x = 1/e (value -1/e). The maximum is at an endpoint.
        f_lb = _univariate_value("entropy", arg_lb)
        f_ub = _univariate_value("entropy", arg_ub)
        inv_e = float(np.exp(-1.0))
        lo = min(f_lb, f_ub)
        if arg_lb <= inv_e <= arg_ub:
            lo = min(lo, -inv_e)
        return lo, max(f_lb, f_ub)
    values = [_univariate_value(func_name, arg_lb), _univariate_value(func_name, arg_ub)]
    return min(values), max(values)


def _tangent_points(func_name: str, lb: float, ub: float) -> list[float]:
    """Choose deterministic valid tangent points for smooth univariate cuts."""
    raw_points = [lb, 0.5 * (lb + ub), ub]
    points: list[float] = []
    for pt in raw_points:
        if func_name == "sqrt" and pt <= 0.0:
            continue
        if func_name in {"log", "log2", "log10", "reciprocal", "entropy"} and pt <= 0.0:
            continue
        if func_name == "tan" and not _is_effectively_finite(np.tan(pt)):
            continue
        if func_name in {"asin", "acos"} and abs(pt) >= 1.0 - 1e-9:
            # Slope diverges at +/-1; the secant still bounds the convex/concave
            # branch, so we simply omit the singular endpoint tangent.
            continue
        if func_name == "acosh" and pt <= 1.0 + 1e-12:
            # Slope diverges at x = 1; omit the singular endpoint tangent.
            continue
        if not np.isfinite(pt):
            continue
        if all(abs(pt - seen) > 1e-12 for seen in points):
            points.append(float(pt))
    return points


def _univariate_curvature(func_name: str, val_lb: float, val_ub: float) -> Optional[str]:
    """Return certified curvature on the interval, or None for mixed curvature."""
    tol = 1e-12
    if func_name in {"exp", "reciprocal", "entropy"}:
        return "convex"
    if func_name in {"sqrt", "log", "log2", "log10"}:
        return "concave"
    if func_name in {"sin", "cos"}:
        if val_lb >= -tol:
            return "concave"
        if val_ub <= tol:
            return "convex"
        return None
    if func_name == "tan":
        if val_lb >= -tol:
            return "convex"
        if val_ub <= tol:
            return "concave"
    if func_name == "asin":
        # asin is odd and increasing: value >= 0 <=> arg >= 0 (convex branch),
        # value <= 0 <=> arg <= 0 (concave branch); mixed across the inflection.
        if val_lb >= -tol:
            return "convex"
        if val_ub <= tol:
            return "concave"
        return None
    if func_name == "acos":
        # acos is decreasing with an inflection at arg = 0 (value = pi/2):
        # arg >= 0 <=> value <= pi/2 (concave); arg <= 0 <=> value >= pi/2 (convex).
        half_pi = 0.5 * math.pi
        if val_ub <= half_pi + tol:
            return "concave"
        if val_lb >= half_pi - tol:
            return "convex"
        return None
    if func_name == "acosh":
        # acosh is concave on its entire domain x >= 1.
        return "concave"
    return None


def _trig_partition_breakpoints(
    relax: UnivariateRelaxation,
    disc_state: DiscretizationState,
    n_orig: int,
) -> list[float]:
    """Return safe breakpoints for a mixed-curvature trig argument interval."""
    lb = float(relax.arg_lb)
    ub = float(relax.arg_ub)
    if relax.func_name not in {"sin", "cos", "tan"} or not (np.isfinite(lb) and np.isfinite(ub)):
        return [lb, ub]

    points = [lb, ub]
    if relax.func_name == "sin":
        curvature_start, critical_start = 0.0, math.pi / 2.0
        points.extend(_critical_points_in_interval(critical_start, math.pi, lb, ub))
    elif relax.func_name == "cos":
        curvature_start, critical_start = math.pi / 2.0, 0.0
        points.extend(_critical_points_in_interval(critical_start, math.pi, lb, ub))
    else:
        curvature_start = 0.0

    points.extend(_critical_points_in_interval(curvature_start, math.pi, lb, ub))

    nz = np.flatnonzero(np.abs(relax.arg_coeff) > 1e-12)
    if nz.size == 1:
        var_idx = int(nz[0])
        if var_idx < n_orig and var_idx in disc_state.partitions:
            coeff = float(relax.arg_coeff[var_idx])
            partition = np.asarray(disc_state.partitions[var_idx], dtype=np.float64)
            if partition.size <= _MAX_TRIG_IMPORTED_BREAKPOINTS:
                transformed = coeff * partition + relax.arg_const
                points.extend(float(p) for p in transformed)

    # A modest fixed split keeps the dedicated trig relaxation useful even when
    # no AMP variable partition exists for the affine argument.
    base = _sorted_unique_points([p for p in points if lb - 1e-12 <= p <= ub + 1e-12])
    refined: list[float] = []
    for a, b in zip(base[:-1], base[1:]):
        if not refined:
            refined.append(float(a))
        width = float(b - a)
        if width > _MAX_TRIG_PIECEWISE_WIDTH:
            n_chunks = int(math.ceil(width / _MAX_TRIG_PIECEWISE_WIDTH))
            for k in range(1, n_chunks):
                refined.append(float(a + width * k / n_chunks))
        refined.append(float(b))
    return _sorted_unique_points(refined or base)


def _trig_piecewise_interval_specs(
    relax: UnivariateRelaxation,
    disc_state: DiscretizationState,
    n_orig: int,
) -> list[tuple[float, float, Optional[str]]]:
    """Build certified curvature subintervals for mixed-curvature trig functions."""
    if relax.func_name not in {"sin", "cos", "tan"}:
        return []
    if not (np.isfinite(relax.arg_lb) and np.isfinite(relax.arg_ub)):
        return []
    if relax.arg_ub - relax.arg_lb >= _MAX_TRIG_PIECEWISE_SPAN:
        return []
    bounds = _trig_range(relax.func_name, relax.arg_lb, relax.arg_ub)
    if bounds is None:
        return []
    if _univariate_curvature(relax.func_name, bounds[0], bounds[1]) is not None:
        return []

    points = _trig_partition_breakpoints(relax, disc_state, n_orig)
    if len(points) - 1 > _MAX_TRIG_PIECEWISE_INTERVALS:
        return []
    intervals: list[tuple[float, float, Optional[str]]] = []
    for a, b in zip(points[:-1], points[1:]):
        if b <= a + 1e-12:
            continue
        local_bounds = _trig_range(relax.func_name, a, b)
        curvature = None
        if local_bounds is not None:
            curvature = _univariate_curvature(relax.func_name, local_bounds[0], local_bounds[1])
        intervals.append((float(a), float(b), curvature))
    if sum(1 for _a, _b, curvature in intervals if curvature is not None) < 2:
        return []
    return intervals


def _inverse_trig_piecewise_interval_specs(
    relax: UnivariateRelaxation,
) -> list[tuple[float, float, Optional[str]]]:
    """Build certified curvature subintervals for asin/acos across the inflection.

    ``asin``/``acos`` change curvature at the argument value 0 (the inflection
    point).  When the argument interval straddles 0 we split it there into two
    sign-definite, single-curvature pieces; otherwise the single-curvature path
    in the main builder already applies.
    """
    if relax.func_name not in {"asin", "acos"}:
        return []
    lb = float(relax.arg_lb)
    ub = float(relax.arg_ub)
    if not (np.isfinite(lb) and np.isfinite(ub)):
        return []
    if not (lb < -1e-12 and ub > 1e-12):
        return []

    intervals: list[tuple[float, float, Optional[str]]] = []
    for a, b in ((lb, 0.0), (0.0, ub)):
        if b <= a + 1e-12:
            continue
        val_lb, val_ub = _univariate_value_bounds(relax.func_name, a, b)
        curvature = _univariate_curvature(relax.func_name, val_lb, val_ub)
        intervals.append((float(a), float(b), curvature))
    if sum(1 for _a, _b, curvature in intervals if curvature is not None) < 2:
        return []
    return intervals


def _affine_argument_has_continuous_var(arg_coeff: np.ndarray, model: Model) -> bool:
    """Return true if an affine argument depends on at least one continuous variable."""
    offset = 0
    for var in model._variables:
        is_continuous = var.var_type not in (VarType.BINARY, VarType.INTEGER)
        for k in range(var.size):
            if abs(float(arg_coeff[offset + k])) > 1e-12 and is_continuous:
                return True
        offset += var.size
    return False


def _trig_square_partition_breakpoints(
    relax: UnivariateRelaxation,
    disc_state: DiscretizationState,
    n_orig: int,
) -> list[float]:
    """Return safe breakpoints for a mixed-curvature trig-square argument interval."""
    lb = float(relax.arg_lb)
    ub = float(relax.arg_ub)
    points = [lb, ub]
    points.extend(_critical_points_in_interval(0.0, math.pi / 2.0, lb, ub))
    points.extend(_critical_points_in_interval(math.pi / 4.0, math.pi / 2.0, lb, ub))

    nz = np.flatnonzero(np.abs(relax.arg_coeff) > 1e-12)
    if nz.size == 1:
        var_idx = int(nz[0])
        if var_idx < n_orig and var_idx in disc_state.partitions:
            coeff = float(relax.arg_coeff[var_idx])
            partition = np.asarray(disc_state.partitions[var_idx], dtype=np.float64)
            if partition.size <= _MAX_TRIG_IMPORTED_BREAKPOINTS:
                transformed = coeff * partition + relax.arg_const
                points.extend(float(p) for p in transformed)

    base = _sorted_unique_points([p for p in points if lb - 1e-12 <= p <= ub + 1e-12])
    refined: list[float] = []
    for a, b in zip(base[:-1], base[1:]):
        if not refined:
            refined.append(float(a))
        width = float(b - a)
        if width > _MAX_TRIG_PIECEWISE_WIDTH:
            n_chunks = int(math.ceil(width / _MAX_TRIG_PIECEWISE_WIDTH))
            for k in range(1, n_chunks):
                refined.append(float(a + width * k / n_chunks))
        refined.append(float(b))
    return _sorted_unique_points(refined or base)


def _trig_square_piecewise_interval_specs(
    relax: UnivariateRelaxation,
    disc_state: DiscretizationState,
    n_orig: int,
) -> list[tuple[float, float, Optional[str]]]:
    """Build certified curvature subintervals for mixed-curvature trig-square terms."""
    if relax.func_name not in {"sin", "cos"}:
        return []
    if not (np.isfinite(relax.arg_lb) and np.isfinite(relax.arg_ub)):
        return []
    if relax.arg_ub - relax.arg_lb >= _MAX_TRIG_PIECEWISE_SPAN:
        return []
    if _trig_square_range(relax.func_name, relax.arg_lb, relax.arg_ub) is None:
        return []
    if _trig_square_curvature(relax.func_name, relax.arg_lb, relax.arg_ub) is not None:
        return []

    points = _trig_square_partition_breakpoints(relax, disc_state, n_orig)
    if len(points) - 1 > _MAX_TRIG_PIECEWISE_INTERVALS:
        return []

    intervals: list[tuple[float, float, Optional[str]]] = []
    for a, b in zip(points[:-1], points[1:]):
        if b <= a + 1e-12:
            continue
        curvature = _trig_square_curvature(relax.func_name, a, b)
        intervals.append((float(a), float(b), curvature))
    if sum(1 for _a, _b, curvature in intervals if curvature is not None) < 2:
        return []
    return intervals


def _univariate_signature(
    func_name: str,
    arg_coeff: np.ndarray,
    arg_const: float,
) -> tuple[str, tuple[float, ...], float]:
    return func_name, tuple(float(c) for c in arg_coeff.tolist()), float(arg_const)


def _flatten_additive_terms(
    expr: Expression, scale: float, out: list[tuple[float, Expression]]
) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "+":
        _flatten_additive_terms(expr.left, scale, out)
        _flatten_additive_terms(expr.right, scale, out)
        return
    if isinstance(expr, BinaryOp) and expr.op == "-":
        _flatten_additive_terms(expr.left, scale, out)
        _flatten_additive_terms(expr.right, -scale, out)
        return
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        _flatten_additive_terms(expr.operand, -scale, out)
        return
    if isinstance(expr, SumOverExpression):
        for term in expr.terms:
            _flatten_additive_terms(term, scale, out)
        return
    out.append((scale, expr))


def _match_scaled_constant_division(
    expr: Expression,
    scale: float,
) -> Optional[tuple[float, Expression]]:
    """Return (scaled numerator, denominator) for scale * (c / denominator)."""
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_scaled_constant_division(expr.operand, -scale)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_scaled_constant_division(expr.right, scale * left_const)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_scaled_constant_division(expr.left, scale * right_const)
        return None

    if not isinstance(expr, BinaryOp) or expr.op != "/":
        return None
    numerator = _constant_value(expr.left)
    if numerator is None or abs(numerator) <= 1e-12:
        return None
    return scale * numerator, expr.right


def _exact_positive_reciprocal_row(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[tuple[np.ndarray, float]]:
    """Match ``constant - numerator / positive_affine <= 0`` as an exact affine row."""
    terms: list[tuple[float, Expression]] = []
    _flatten_additive_terms(expr, 1.0, terms)

    constant_term = 0.0
    reciprocal_match: Optional[tuple[float, Expression]] = None

    for scale, term in terms:
        const_val = _constant_value(term)
        if const_val is not None:
            constant_term += scale * const_val
            continue

        match = _match_scaled_constant_division(term, scale)
        if match is None or reciprocal_match is not None:
            return None
        reciprocal_match = match

    if reciprocal_match is None or constant_term <= 0.0:
        return None

    scaled_numerator, denominator = reciprocal_match
    if scaled_numerator >= 0.0:
        return None

    try:
        denom_coeff, denom_const = _linearize_affine_expr(denominator, model, len(flat_lb))
        denom_lb, _denom_ub = _linear_expr_bounds(denom_coeff, denom_const, flat_lb, flat_ub)
    except ValueError:
        return None

    if denom_lb <= 0.0:
        return None
    rhs = -scaled_numerator / constant_term
    if not np.isfinite(rhs):
        return None
    return denom_coeff, float(rhs - denom_const)


def _flatten_product_factors(expr: Expression, out: list[Expression]) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "*":
        _flatten_product_factors(expr.left, out)
        _flatten_product_factors(expr.right, out)
        return
    out.append(expr)


def _monomial_power_term(expr: Expression, model: Model) -> Optional[tuple[int, int]]:
    flat = _get_flat_index(expr, model)
    if flat is not None:
        return flat, 1
    if isinstance(expr, BinaryOp) and expr.op == "**" and isinstance(expr.right, Constant):
        base = _get_flat_index(expr.left, model)
        if base is None:
            return None
        exp_val = float(expr.right.value)
        n_int = int(exp_val)
        if exp_val == n_int and n_int >= 1:
            return base, n_int
    return None


def _match_scaled_monomial(expr: Expression, model: Model) -> Optional[tuple[float, int, int]]:
    factors: list[Expression] = []
    _flatten_product_factors(expr, factors)
    scalar = 1.0
    var_idx: Optional[int] = None
    power_total = 0
    for factor in factors:
        const = _constant_value(factor)
        if const is not None:
            scalar *= const
            continue
        power_term = _monomial_power_term(factor, model)
        if power_term is None:
            return None
        factor_var, factor_power = power_term
        if var_idx is None:
            var_idx = factor_var
        elif var_idx != factor_var:
            return None
        power_total += factor_power
    if var_idx is None or power_total < 1:
        return None
    return scalar, var_idx, power_total


def _match_x_exp_product(expr: Expression, model: Model) -> Optional[tuple[float, int]]:
    factors: list[Expression] = []
    _flatten_product_factors(expr, factors)
    scalar = 1.0
    var_idx: Optional[int] = None
    exp_arg_idx: Optional[int] = None
    for factor in factors:
        const = _constant_value(factor)
        if const is not None:
            scalar *= const
            continue
        flat = _get_flat_index(factor, model)
        if flat is not None:
            if var_idx is not None:
                return None
            var_idx = flat
            continue
        if isinstance(factor, FunctionCall) and factor.func_name == "exp" and len(factor.args) == 1:
            arg_idx = _get_flat_index(factor.args[0], model)
            if arg_idx is None or exp_arg_idx is not None:
                return None
            exp_arg_idx = arg_idx
            continue
        return None
    if var_idx is None or exp_arg_idx is None or var_idx != exp_arg_idx:
        return None
    return scalar, var_idx


def _safe_x_exp_value(x: float) -> Optional[float]:
    if not np.isfinite(x) or x > _MAX_FINITE_EXP_ARG:
        return None
    if x < -745.0:
        return 0.0
    return float(x * np.exp(x))


def _x_exp_upper_bound(var_idx: int, flat_lb: np.ndarray, flat_ub: np.ndarray) -> Optional[float]:
    lb = float(flat_lb[var_idx])
    ub = float(flat_ub[var_idx])
    if not (_is_effectively_finite(lb) and _is_effectively_finite(ub)):
        return None
    values = [_safe_x_exp_value(lb), _safe_x_exp_value(ub)]
    finite_values = [value for value in values if value is not None and np.isfinite(value)]
    if len(finite_values) != len(values):
        return None
    return max(finite_values)


def _is_cos_call(expr: Expression) -> bool:
    return isinstance(expr, FunctionCall) and expr.func_name == "cos" and len(expr.args) == 1


def _flat_variable_types(model: Model) -> list[VarType]:
    types: list[VarType] = []
    for var in model._variables:
        types.extend([var.var_type] * var.size)
    return types


def _integer_domain_values(
    var_idx: int,
    flat_types: list[VarType],
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[range]:
    var_type = flat_types[var_idx]
    if var_type not in (VarType.BINARY, VarType.INTEGER):
        return None
    lb_i = float(flat_lb[var_idx])
    ub_i = float(flat_ub[var_idx])
    if not (_is_effectively_finite(lb_i) and _is_effectively_finite(ub_i)):
        return None
    lo = int(np.ceil(lb_i - 1e-9))
    hi = int(np.floor(ub_i + 1e-9))
    if var_type == VarType.BINARY:
        lo = max(lo, 0)
        hi = min(hi, 1)
    if lo > hi:
        return None
    return range(lo, hi + 1)


def _integer_affine_cos_lower_bound(
    expr: Expression,
    scale: float,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Return exact lower bound for scale*cos(integer-affine expr) on a small box."""
    if not isinstance(expr, FunctionCall) or expr.func_name != "cos" or len(expr.args) != 1:
        return None
    try:
        coeff, const = _linearize_affine_expr(expr.args[0], model, len(flat_lb))
    except ValueError:
        return None

    flat_types = _flat_variable_types(model)
    entries: list[tuple[float, range]] = []
    n_values = 1
    for var_idx, c_i in enumerate(coeff):
        c = float(c_i)
        if abs(c) <= 1e-12:
            continue
        values = _integer_domain_values(var_idx, flat_types, flat_lb, flat_ub)
        if values is None:
            return None
        n_values *= len(values)
        if n_values > _MAX_INTEGER_COS_ENUM:
            return None
        entries.append((c, values))

    if not entries:
        value = scale * float(np.cos(const))
        return value if np.isfinite(value) else None

    best = np.inf
    for assignment in itertools.product(*(values for _c, values in entries)):
        arg = float(const)
        for (c, _values), value in zip(entries, assignment):
            arg += c * float(value)
        best = min(best, scale * float(np.cos(arg)))
    return float(best) if np.isfinite(best) else None


def _integer_affine_trig_range(
    func_name: str,
    coeff: np.ndarray,
    const: float,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[tuple[float, float]]:
    """Return exact range for trig(affine integer vars) on small finite domains."""
    if func_name not in {"sin", "cos", "tan"}:
        return None

    flat_types = _flat_variable_types(model)
    entries: list[tuple[float, range]] = []
    n_values = 1
    for var_idx, c_i in enumerate(coeff):
        c = float(c_i)
        if abs(c) <= 1e-12:
            continue
        values = _integer_domain_values(var_idx, flat_types, flat_lb, flat_ub)
        if values is None:
            return None
        n_values *= len(values)
        if n_values > _MAX_INTEGER_COS_ENUM:
            return None
        entries.append((c, values))

    if not entries:
        value = _univariate_value(func_name, float(const))
        return (value, value) if np.isfinite(value) else None

    values_out: list[float] = []
    for assignment in itertools.product(*(values for _c, values in entries)):
        arg = float(const)
        for (c, _values), value in zip(entries, assignment):
            arg += c * float(value)
        value_out = _univariate_value(func_name, arg)
        if not np.isfinite(value_out):
            return None
        values_out.append(value_out)
    if not values_out:
        return None
    return min(values_out), max(values_out)


def _finite_domain_trig_square_table_values(
    relax: UnivariateRelaxation,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[tuple[int, float, float, list[int], list[float], list[float]]]:
    """Return exact finite-domain values for a single integer trig-square argument."""
    if relax.func_name not in {"sin", "cos"}:
        return None

    nz = np.flatnonzero(np.abs(relax.arg_coeff) > 1e-12)
    if nz.size != 1:
        return None

    var_idx = int(nz[0])
    flat_types = _flat_variable_types(model)
    domain = _integer_domain_values(var_idx, flat_types, flat_lb, flat_ub)
    if domain is None:
        return None

    domain_values = list(domain)
    if not domain_values or len(domain_values) > _MAX_FINITE_DOMAIN_TRIG_TABLE_VALUES:
        return None

    arg_coeff = float(relax.arg_coeff[var_idx])
    arg_const = float(relax.arg_const)
    if not (_is_effectively_finite(arg_coeff) and _is_effectively_finite(arg_const)):
        return None

    trig_values: list[float] = []
    square_values: list[float] = []
    for value in domain_values:
        arg = arg_coeff * float(value) + arg_const
        trig_value = _univariate_value(relax.func_name, arg)
        square_value = trig_value * trig_value
        if not (np.isfinite(trig_value) and np.isfinite(square_value)):
            return None
        trig_values.append(float(trig_value))
        square_values.append(float(square_value))

    return var_idx, arg_coeff, arg_const, domain_values, trig_values, square_values


def _scaled_affine_lower_bound(
    expr: Expression,
    scale: float,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    try:
        coeff, const = _linearize_affine_expr(expr, model, len(flat_lb))
    except ValueError:
        return None
    want_lower = scale >= 0.0
    bound = float(const)
    for c_i, lb_i, ub_i in zip(coeff, flat_lb, flat_ub):
        c = float(c_i)
        if abs(c) <= 1e-12:
            continue
        chosen = float(lb_i) if (c >= 0.0) == want_lower else float(ub_i)
        if not _is_effectively_finite(chosen):
            return None
        bound += c * chosen
    return scale * bound


def _evaluate_polynomial(coeffs: dict[int, float], x: float) -> Optional[float]:
    max_power = max(coeffs)
    value = 0.0
    for power in range(max_power, -1, -1):
        value = value * x + float(coeffs.get(power, 0.0))
        if not np.isfinite(value):
            return None
    return float(value)


def _polynomial_lower_bound(
    coeffs: dict[int, float],
    lb: float,
    ub: float,
) -> Optional[float]:
    clean = {power: coeff for power, coeff in coeffs.items() if abs(coeff) > 1e-12}
    if not clean:
        return 0.0
    max_power = max(clean)
    if max_power == 0:
        return float(clean[0])

    leading = float(clean[max_power])
    lo_unbounded = not _is_effectively_finite(lb)
    hi_unbounded = not _is_effectively_finite(ub)
    if hi_unbounded and leading < 0.0:
        return None
    if lo_unbounded:
        if max_power % 2 == 0 and leading < 0.0:
            return None
        if max_power % 2 == 1 and leading > 0.0:
            return None

    candidates: list[float] = []
    if not lo_unbounded:
        candidates.append(float(lb))
    if not hi_unbounded:
        candidates.append(float(ub))

    deriv_coeffs = [power * clean.get(power, 0.0) for power in range(max_power, 0, -1)]
    roots = np.roots(deriv_coeffs) if deriv_coeffs else np.array([])
    for root in roots:
        if abs(float(np.imag(root))) > 1e-9:
            continue
        x = float(np.real(root))
        if (lo_unbounded or x >= lb - 1e-9) and (hi_unbounded or x <= ub + 1e-9):
            candidates.append(x)

    values: list[float] = []
    for x in _sorted_unique_points(candidates):
        value = _evaluate_polynomial(clean, x)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return min(values)


def _reciprocal_term_lower_bound(
    scaled_numerator: float,
    denominator: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Rigorous constant lower bound for ``scaled_numerator / denominator(x)``.

    Encloses ``denominator`` over the box via the same separable-polynomial
    machinery used for objective lifts (``_expression_lower_bound_for_lift`` /
    ``_expression_upper_bound_for_lift``), which recovers ``D_lo``/``D_hi`` from a
    distributed quadratic by per-variable vertex minimization — crucially, this
    works on the live solve's already-distributed denominator (e.g. ``0.1 +
    x0**2 - 8*x0 + 16 + ...``) where a naive interval evaluation of ``x*x`` on an
    unbounded box would collapse to ``[-inf, inf]``. When the enclosure is
    strictly positive (``D_lo > 0``), ``1/D`` is decreasing in ``D`` so the term
    ``k/D`` is minimized at ``D_hi`` when ``k > 0`` and at ``D_lo`` when
    ``k < 0``. Returns ``None`` (caller abstains) if the denominator cannot be
    proven strictly positive or the bound is not finite.
    """
    if abs(scaled_numerator) <= 1e-12:
        return 0.0
    denom_lo = _expression_lower_bound_for_lift(denominator, model, flat_lb, flat_ub)
    denom_hi = _expression_upper_bound_for_lift(denominator, model, flat_lb, flat_ub)
    # A strictly-positive, finite lower end is required for a sound reciprocal;
    # _expression_lower_bound_for_lift returns None when it cannot prove one.
    if denom_lo is None or not np.isfinite(denom_lo) or denom_lo <= 1e-12:
        return None
    if scaled_numerator > 0.0:
        # k/D minimized at the largest D; an unbounded/unknown D_hi drives the
        # positive term toward 0 from above, which is still a valid lower bound.
        bound = (
            0.0
            if (denom_hi is None or not np.isfinite(denom_hi))
            else (scaled_numerator / denom_hi)
        )
    else:
        bound = scaled_numerator / denom_lo
    if not np.isfinite(bound):
        return None
    return float(bound)


def _match_scaled_even_power(
    term: Expression, scale: float
) -> Optional[tuple[float, Expression, int]]:
    """Match ``scale * c * (base)**n`` with ``n`` a positive even integer.

    Returns ``(coeff, base, n)`` where ``coeff`` folds ``scale`` and every
    constant factor of the product. Exactly one even-power factor is allowed;
    any other non-constant factor (or a second power) disqualifies the match.
    """
    factors: list[Expression] = []
    _flatten_product_factors(term, factors)
    coeff = scale
    base: Optional[Expression] = None
    power = 0
    for factor in factors:
        const = _constant_value(factor)
        if const is not None:
            coeff *= const
            continue
        if (
            isinstance(factor, BinaryOp)
            and factor.op == "**"
            and isinstance(factor.right, Constant)
        ):
            exp_val = float(factor.right.value)
            n_int = int(round(exp_val))
            if abs(exp_val - n_int) < 1e-12 and n_int >= 2 and n_int % 2 == 0:
                if base is not None:
                    return None
                base = factor.left
                power = n_int
                continue
        return None
    if base is None:
        return None
    return coeff, base, power


def _count_distinct_scalar_refs(expr: Expression, model: Model) -> int:
    """Count distinct scalar variable columns referenced by ``expr``.

    Used to gate the even-power lower bound to genuinely multivariate bases:
    a single-variable square is handled more tightly by the distribute /
    polynomial path (which combines it with any linear term in the same
    variable), so only multivariate bases — which that path cannot bound at
    all — are routed through the sum-of-squares relaxation.
    """
    seen: set = set()

    def visit(e: Expression) -> None:
        idx = _get_flat_index(e, model)
        if idx is not None:
            seen.add(idx)
            return
        if isinstance(e, Variable):
            seen.add(("var", id(e)))
            return
        if isinstance(e, IndexExpression):
            visit(e.base)
            return
        if isinstance(e, BinaryOp):
            visit(e.left)
            visit(e.right)
        elif isinstance(e, UnaryOp):
            visit(e.operand)
        elif isinstance(e, FunctionCall):
            for a in e.args:
                visit(a)
        elif isinstance(e, SumExpression):
            visit(e.operand)
        elif isinstance(e, SumOverExpression):
            for t in e.terms:
                visit(t)

    visit(expr)
    return len(seen)


def _even_power_term_lower_bound(
    coeff: float,
    base: Expression,
    n: int,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Rigorous constant lower bound for ``coeff * base(x)**n`` with ``n`` even.

    ``base**n >= 0`` always, so for ``coeff >= 0`` the term is nonnegative — a
    valid lower bound of ``0`` even when ``base`` cannot be enclosed. When the
    box yields a finite enclosure ``base in [bl, bh]`` the bound tightens to the
    vertex minimum of ``base**n`` (``0`` if the interval straddles zero, else
    ``min(|bl|, |bh|)**n``). For ``coeff < 0`` the term is maximized in
    magnitude at the larger ``|base|`` endpoint, so a finite enclosure is
    required; ``None`` (caller abstains) when it is unavailable.
    """
    bl = _expression_lower_bound_for_lift(base, model, flat_lb, flat_ub)
    bh = _expression_upper_bound_for_lift(base, model, flat_lb, flat_ub)
    if coeff >= 0.0:
        if bl is None or bh is None or not (np.isfinite(bl) and np.isfinite(bh)):
            return 0.0
        if bl <= 0.0 <= bh:
            pow_min = 0.0
        else:
            pow_min = min(abs(bl), abs(bh)) ** n
        return float(coeff * pow_min)
    if bl is None or bh is None or not (np.isfinite(bl) and np.isfinite(bh)):
        return None
    pow_max = max(abs(bl), abs(bh)) ** n
    bound = coeff * pow_max
    if not np.isfinite(bound):
        return None
    return float(bound)


def _separable_objective_lower_bound(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Compute a conservative constant lower bound for simple separable objectives.

    ``expr`` is flattened additively and matched term-by-term.  Reciprocal terms
    ``k / D(x)`` are matched on the term's ORIGINAL (un-distributed) structure so
    the denominator's square/power shape survives for a sound interval enclosure;
    every other term is distributed individually before the polynomial / affine
    matchers run (the union over terms equals distributing the whole expression,
    so non-reciprocal behavior is unchanged).
    """
    terms: list[tuple[float, Expression]] = []
    _flatten_additive_terms(expr, 1.0, terms)

    total = 0.0
    polynomial_terms: dict[int, dict[int, float]] = {}

    def _accumulate_simple_term(scale: float, term: Expression) -> bool:
        """Fold one already-distributed, non-reciprocal term into the running bound.

        Returns ``False`` (caller abstains entirely) when the term is not one of
        the recognized separable shapes.
        """
        nonlocal total
        if abs(scale) <= 1e-12:
            return True
        const = _constant_value(term)
        if const is not None:
            total += scale * const
            return True

        x_exp = _match_x_exp_product(term, model)
        if x_exp is not None:
            scalar, var_idx = x_exp
            term_scale = scale * scalar
            if abs(term_scale) <= 1e-12:
                return True
            if term_scale > 0.0:
                total += term_scale * (-1.0 / np.e)
                return True
            upper = _x_exp_upper_bound(var_idx, flat_lb, flat_ub)
            if upper is None:
                return False
            total += term_scale * upper
            return True

        if _is_cos_call(term):
            integer_lb = _integer_affine_cos_lower_bound(term, scale, model, flat_lb, flat_ub)
            total += integer_lb if integer_lb is not None else -abs(scale)
            return True

        monomial = _match_scaled_monomial(term, model)
        if monomial is not None:
            scalar, var_idx, power = monomial
            polynomial_terms.setdefault(var_idx, {})
            polynomial_terms[var_idx][power] = (
                polynomial_terms[var_idx].get(power, 0.0) + scale * scalar
            )
            return True

        affine_bound = _scaled_affine_lower_bound(term, scale, model, flat_lb, flat_ub)
        if affine_bound is None:
            return False
        total += affine_bound
        return True

    for scale, term in terms:
        if abs(scale) <= 1e-12:
            continue

        # Reciprocal term ``k / D(x)`` with a strictly-positive denominator (e.g.
        # ex8_1_6's ``-1/(0.1 + (x0-4)**2 + (x1-4)**2)``). The MILP linearizer
        # cannot relax a non-constant division, so without this the whole
        # objective is dropped and AMP can never certify. A rigorous interval
        # enclosure ``D in [D_lo, D_hi]`` with ``D_lo > 0`` yields a valid
        # constant lower bound for the term: ``k/D_hi`` when ``k > 0`` else
        # ``k/D_lo`` (``1/D`` is decreasing in ``D``). The bound tightens as B&B
        # branching shrinks the box, eventually enabling certification. Matched
        # on the un-distributed term so ``D``'s ``(x-a)**2`` shape survives for a
        # tight interval enclosure (distribution would expand it to a polynomial
        # whose naive interval enclosure is uselessly loose on a wide box).
        recip = _match_scaled_constant_division(term, scale)
        if recip is not None:
            recip_bound = _reciprocal_term_lower_bound(recip[0], recip[1], model, flat_lb, flat_ub)
            if recip_bound is None:
                return None
            total += recip_bound
            continue

        # Even-power term ``c * (E(x))**n`` (n even) with a *multivariate* base,
        # e.g. Rosenbrock's ``100 * (x1 - x0**2)**2``. Distributing it yields a
        # bilinear/multivariate polynomial whose cross terms no single-variable
        # matcher accepts, so the whole objective would be dropped. But a square
        # is nonnegative regardless of its argument's structure: for ``c >= 0``
        # the term is ``>= 0`` (tightened to the vertex minimum of ``E**n`` when
        # the box encloses ``E``). Recognizing it on the un-distributed term lets
        # AMP certify sum-of-squares objectives at the root. Single-variable
        # bases are left to the polynomial path, which combines them with any
        # linear term in the same variable for a strictly tighter bound.
        even_pow = _match_scaled_even_power(term, scale)
        if even_pow is not None and _count_distinct_scalar_refs(even_pow[1], model) >= 2:
            coeff, base, power = even_pow
            ep_bound = _even_power_term_lower_bound(coeff, base, power, model, flat_lb, flat_ub)
            if ep_bound is None:
                return None
            total += ep_bound
            continue

        # Distribute this single term and fold each resulting sub-term through the
        # simple-shape matchers (polynomial path needs the expanded form).
        sub_terms: list[tuple[float, Expression]] = []
        _flatten_additive_terms(distribute_products(term), scale, sub_terms)
        for sub_scale, sub_term in sub_terms:
            if not _accumulate_simple_term(sub_scale, sub_term):
                return None

    for var_idx, coeffs in polynomial_terms.items():
        lower = _polynomial_lower_bound(coeffs, float(flat_lb[var_idx]), float(flat_ub[var_idx]))
        if lower is None:
            return None
        total += lower

    if not np.isfinite(total):
        return None
    return float(total)


@dataclass
class CompositeUnivariateRelaxation:
    """Outer relaxation for a single-variable nonlinear node of certified curvature.

    Unlike :class:`UnivariateRelaxation` (a named operator of an *affine*
    argument), this handles a composite node ``f(x_i)`` that depends on a single
    original variable but whose internal structure is non-affine — e.g.
    ``sqrt(x**2 + c)``, ``(a*x + b)**p``, or ``exp(c - k/(d + x))``. Curvature is
    proven sound on the node box (analytic power rule, or a subdivided
    interval-Hessian certificate), so the tangent/secant envelope below is a
    rigorous outer approximation. The aux column ``z`` satisfies
    ``lower_lines ≤ z ≤ upper_lines`` (convex: tangents below, secant above;
    concave: secant below, tangents above).
    """

    expr_id: int
    aux_col: int
    var_idx: int
    curvature: str
    arg_lb: float
    arg_ub: float
    lower_lines: tuple[tuple[float, float], ...]
    upper_lines: tuple[tuple[float, float], ...]
    pin_value: Optional[float]


_COMPOSITE_CURV_TOL = 1e-9
_COMPOSITE_MAX_SUBDIV = 256


def _build_convexity_box(model: Model, flat_lb: np.ndarray, flat_ub: np.ndarray) -> dict:
    """Build the ``{Variable: Interval}`` box the convexity certificate expects."""
    from discopt._jax.convexity.interval import Interval

    box: dict = {}
    offset = 0
    for v in model._variables:
        size = v.size
        shape = v.shape if v.shape else (1,)
        lo = np.asarray(flat_lb[offset : offset + size], dtype=np.float64).reshape(shape)
        hi = np.asarray(flat_ub[offset : offset + size], dtype=np.float64).reshape(shape)
        box[v] = Interval(lo, hi)
        offset += size
    return box


def _composite_referenced_var(expr: Expression, model: Model) -> Optional[tuple[Variable, int]]:
    """Return ``(var, flat_idx)`` if ``expr`` depends on exactly one scalar variable."""
    found: dict[int, Variable] = {}

    def visit(e: Expression) -> None:
        if isinstance(e, Variable):
            found[id(e)] = e
            return
        if isinstance(e, IndexExpression):
            if isinstance(e.base, Variable):
                found[id(e.base)] = e.base
            else:
                visit(e.base)
            return
        if isinstance(e, BinaryOp):
            visit(e.left)
            visit(e.right)
        elif isinstance(e, UnaryOp):
            visit(e.operand)
        elif isinstance(e, FunctionCall):
            for a in e.args:
                visit(a)
        elif isinstance(e, SumExpression):
            visit(e.operand)
        elif isinstance(e, SumOverExpression):
            for t in e.terms:
                visit(t)

    visit(expr)
    if len(found) != 1:
        return None
    var = next(iter(found.values()))
    if var.size != 1:
        return None
    return var, _compute_var_offset(var, model)


def _should_claim_composite(expr: Expression, model: Model, n_orig: int) -> bool:
    """True when ``expr`` is a nonlinear node the *existing* machinery cannot lift.

    The bilinear/monomial/fractional-power/univariate-of-affine builders already
    cover their cases; claiming those here would create duplicate aux columns.
    So claim only:

    * a named univariate call whose argument is *non-affine* (the affine case is
      handled by :func:`_collect_univariate_relaxations`), or
    * ``(base)**p`` with a *non-integer* constant exponent ``p`` and a *non-bare*
      base (a bare variable base is a monomial / fractional power; integer powers
      of a non-bare base are squares/monomials handled by the dedicated square and
      ``distribute_products`` machinery, so claiming them duplicates aux columns).
    """
    if isinstance(expr, FunctionCall) and len(expr.args) == 1:
        op_info = _univariate_arg(expr)
        if op_info is None:
            return False
        _name, arg = op_info
        try:
            _linearize_affine_expr(arg, model, n_orig)
        except ValueError:
            return True  # non-affine argument → existing univariate path can't lift it
        return False
    if isinstance(expr, BinaryOp) and expr.op == "**" and isinstance(expr.right, Constant):
        p = float(expr.right.value)
        if p == int(p):
            return False  # integer powers → monomial / square / distribute machinery
        # Bare variable / indexed base is a fractional power already.
        if _get_flat_index(expr.left, model) is not None:
            return False
        return True
    return False


def _affine_base_power_curvature(expr: Expression, model: Model, box: dict) -> Optional[str]:
    """Analytic curvature of ``(affine_base)**p`` from ``p`` and the base sign.

    ``x**p`` is convex for ``p ≥ 1`` or ``p ≤ 0`` and concave for ``0 ≤ p ≤ 1``
    on ``x > 0``; composition with an affine map preserves curvature. The base
    sign is checked with a sound interval enclosure (Boyd & Vandenberghe §3.1.5).
    """
    if not (isinstance(expr, BinaryOp) and expr.op == "**" and isinstance(expr.right, Constant)):
        return None
    p = float(expr.right.value)
    if p in (0.0, 1.0):
        return None
    from discopt._jax.convexity.interval_eval import evaluate_interval

    try:
        base_iv = evaluate_interval(expr.left, model, box=box)
    except Exception:
        return None
    base_lo = float(np.asarray(base_iv.lo).ravel()[0])
    if not np.isfinite(base_lo):
        return None
    # Outward interval rounding can render a true 0 as a tiny negative; a small
    # tolerance keeps the nonnegativity test from spuriously abstaining.
    tol = 1e-9
    if p < 0.0:
        return "convex" if base_lo > tol else None
    if base_lo < -tol:
        return None  # non-integer / odd powers need a nonnegative base for a clean verdict
    if p > 1.0:
        return "convex"
    return "concave"  # 0 < p < 1


def _subdivision_curvature(
    expr: Expression,
    model: Model,
    var: Variable,
    flat_idx: int,
    lo: float,
    hi: float,
    box: dict,
) -> Optional[str]:
    """Sound curvature via a subdivided interval Hessian.

    ``f`` is C² and depends on one variable, so its Hessian is the scalar
    ``f''`` in entry ``(flat_idx, flat_idx)``. Covering ``[lo, hi]`` with
    sub-intervals and enclosing ``f''`` on each tightens the dependency-induced
    looseness of a single whole-box enclosure: if every piece has ``f'' ≥ 0``
    the function is convex on the union ``[lo, hi]`` (symmetrically for concave).
    """
    from discopt._jax.convexity.interval import Interval
    from discopt._jax.convexity.interval_ad import interval_hessian

    if hi - lo <= _COMPOSITE_CURV_TOL:
        return None  # pinned — handled by the exact pin value, not an envelope
    shape = var.shape if var.shape else (1,)
    n_sub = 4
    while n_sub <= _COMPOSITE_MAX_SUBDIV:
        edges = np.linspace(lo, hi, n_sub + 1)
        all_convex = True
        all_concave = True
        finite = True
        for k in range(n_sub):
            box[var] = Interval(
                np.full(shape, edges[k], dtype=np.float64),
                np.full(shape, edges[k + 1], dtype=np.float64),
            )
            try:
                ad = interval_hessian(expr, model, box=box)
            except Exception:
                finite = False
                break
            h = ad.hess
            h_lo = float(h.lo[flat_idx, flat_idx])
            h_hi = float(h.hi[flat_idx, flat_idx])
            if not (np.isfinite(h_lo) and np.isfinite(h_hi)):
                finite = False
                break
            if h_lo < -_COMPOSITE_CURV_TOL:
                all_convex = False
            if h_hi > _COMPOSITE_CURV_TOL:
                all_concave = False
            if not all_convex and not all_concave:
                break
        box[var] = Interval(
            np.full(shape, lo, dtype=np.float64), np.full(shape, hi, dtype=np.float64)
        )
        if not finite:
            return None
        if all_convex:
            return "convex"
        if all_concave:
            return "concave"
        n_sub *= 4
    return None


def _composite_curvature(
    expr: Expression,
    model: Model,
    var: Variable,
    flat_idx: int,
    lo: float,
    hi: float,
    box: dict,
) -> Optional[str]:
    """Certified curvature of a single-variable composite, or ``None`` to abstain."""
    analytic = _affine_base_power_curvature(expr, model, box)
    if analytic is not None:
        return analytic
    return _subdivision_curvature(expr, model, var, flat_idx, lo, hi, box)


def _composite_envelope(
    curvature: str,
    lo: float,
    hi: float,
    value_fn,
    grad_fn,
) -> Optional[tuple[tuple, tuple, Optional[float], tuple[float, float]]]:
    """Build (lower_lines, upper_lines, pin_value, col_bounds) for a composite.

    Lines are ``(slope, intercept)`` in the single variable. For a convex
    function tangents underestimate and the endpoint secant overestimates; the
    roles swap for concave. Slopes/values come from exact autodiff, so for a
    truly-convex (resp. concave) ``f`` each tangent is a supporting line and the
    secant a chord — a rigorous outer band.

    ``col_bounds`` is a rigorous box on the aux column derived from the same
    certified curvature: a convex ``f`` attains its maximum at an endpoint, and
    its underestimating tangents bound it from below (symmetrically for concave).
    This avoids the interval evaluator, which abstains (``±inf``) on a
    non-integer power whose base interval grazes zero.
    """
    if hi - lo <= _COMPOSITE_CURV_TOL:
        v = value_fn(0.5 * (lo + hi))
        if not np.isfinite(v):
            return None
        return (), (), float(v), (float(v), float(v))

    pts: list[float] = []
    for t in (lo, 0.5 * (lo + hi), hi):
        if all(abs(t - s) > 1e-12 for s in pts):
            pts.append(float(t))

    f_lo = value_fn(lo)
    f_hi = value_fn(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return None
    secant_slope = (f_hi - f_lo) / (hi - lo)
    secant_intercept = f_lo - secant_slope * lo
    if not (np.isfinite(secant_slope) and np.isfinite(secant_intercept)):
        return None

    tangents: list[tuple[float, float]] = []
    for t in pts:
        slope = grad_fn(t)
        val = value_fn(t)
        if not (np.isfinite(slope) and np.isfinite(val)):
            return None
        tangents.append((float(slope), float(val - slope * t)))

    def _line_span(slope: float, intercept: float) -> tuple[float, float]:
        a = slope * lo + intercept
        b = slope * hi + intercept
        return min(a, b), max(a, b)

    secant = (float(secant_slope), float(secant_intercept))
    if curvature == "convex":
        # Max at an endpoint; tangents (underestimators) bound below.
        col_hi = max(f_lo, f_hi)
        col_lo = min(_line_span(s, b)[0] for s, b in tangents)
        return tuple(tangents), (secant,), None, (float(col_lo), float(col_hi))
    # Concave: min at an endpoint; tangents (overestimators) bound above.
    col_lo = min(f_lo, f_hi)
    col_hi = max(_line_span(s, b)[1] for s, b in tangents)
    return (secant,), tuple(tangents), None, (float(col_lo), float(col_hi))


def _collect_composite_univariate_relaxations(
    model: Model,
    n_orig: int,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    start_col: int,
    claimed_ids: set[int],
) -> tuple[list[CompositeUnivariateRelaxation], dict[int, int], list[tuple[float, float]]]:
    """Collect single-variable composite nodes with certified curvature.

    Returns ``(relaxations, var_map, bounds)`` where ``var_map`` maps the node's
    ``id(expr)`` to its aux column. Abstains silently (no entry) whenever
    curvature can't be proven, the value range isn't finite, or autodiff fails —
    abstention drops to the existing warn-and-omit path and never to an unsound
    cut.
    """
    relaxations: list[CompositeUnivariateRelaxation] = []
    var_map: dict[int, int] = {}
    bounds: list[tuple[float, float]] = []
    seen: set[int] = set()
    col_idx = start_col

    try:
        import jax
        import jax.numpy as jnp

        from discopt._jax.dag_compiler import compile_expression
    except Exception:
        return relaxations, var_map, bounds

    box = _build_convexity_box(model, flat_lb, flat_ub)
    base_x = np.zeros(n_orig, dtype=np.float64)

    def maybe_add(expr: Expression) -> None:
        nonlocal col_idx
        eid = id(expr)
        if eid in seen or eid in claimed_ids:
            return
        if not _should_claim_composite(expr, model, n_orig):
            return
        ref = _composite_referenced_var(expr, model)
        if ref is None:
            return
        var, flat_idx = ref
        lo = float(flat_lb[flat_idx])
        hi = float(flat_ub[flat_idx])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
            return
        curvature = _composite_curvature(expr, model, var, flat_idx, lo, hi, box)
        if curvature is None:
            return
        try:
            f = compile_expression(expr, model)
            grad_f = jax.grad(lambda xv: jnp.reshape(f(xv), ()))
        except Exception:
            return

        def value_fn(t: float) -> float:
            x = jnp.asarray(base_x).at[flat_idx].set(t)
            return float(jnp.reshape(f(x), ()))

        def grad_fn(t: float) -> float:
            x = jnp.asarray(base_x).at[flat_idx].set(t)
            return float(np.asarray(grad_f(x)).ravel()[flat_idx])

        env = _composite_envelope(curvature, lo, hi, value_fn, grad_fn)
        if env is None:
            return
        lower_lines, upper_lines, pin_value, col_bounds = env
        seen.add(eid)
        var_map[eid] = col_idx
        relaxations.append(
            CompositeUnivariateRelaxation(
                expr_id=eid,
                aux_col=col_idx,
                var_idx=flat_idx,
                curvature=curvature,
                arg_lb=lo,
                arg_ub=hi,
                lower_lines=lower_lines,
                upper_lines=upper_lines,
                pin_value=pin_value,
            )
        )
        bounds.append(col_bounds)
        col_idx += 1

    def visit(expr: Expression) -> None:
        maybe_add(expr)
        if isinstance(expr, BinaryOp):
            visit(expr.left)
            visit(expr.right)
        elif isinstance(expr, UnaryOp):
            visit(expr.operand)
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                visit(arg)
        elif isinstance(expr, IndexExpression):
            if not isinstance(expr.base, Variable):
                visit(expr.base)
        elif isinstance(expr, SumExpression):
            visit(expr.operand)
        elif isinstance(expr, SumOverExpression):
            for term in expr.terms:
                visit(term)

    if model._objective is not None:
        visit(model._objective.expression)
    for constraint in model._constraints:
        visit(constraint.body)

    return relaxations, var_map, bounds


def _collect_univariate_relaxations(
    model: Model,
    n_orig: int,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    start_col: int,
) -> tuple[list[UnivariateRelaxation], dict[object, int], list[tuple[float, float]]]:
    """Collect supported univariate operator nodes and assign auxiliary columns."""
    relaxations: list[UnivariateRelaxation] = []
    var_map: dict[object, int] = {}
    bounds: list[tuple[float, float]] = []
    seen: set[int] = set()
    col_idx = start_col

    # Lazily-computed FBBT-tightened box, used only to *rescue* a univariate
    # operator whose argument is non-finite (out of domain) under the raw node
    # box and would otherwise be dropped from the relaxation (issue #219).
    # ``None`` until first needed; ``(None, None)`` if FBBT is unavailable or
    # proves infeasibility. The tightened box is a sound outer bound on the
    # *global* feasible set (FBBT only removes provably-infeasible regions and
    # uses no objective cutoff), so every feasible point — at any B&B node — lies
    # within it; an envelope built over it is therefore valid for the whole
    # relaxation. We intersect with the node box for extra tightness.
    fbbt_cache: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []

    def _fbbt_argument_box() -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not fbbt_cache:
            tightened: tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
            try:
                from discopt.tightening import fbbt_box

                res = fbbt_box(model)
                if not res.infeasible and len(res.lb) == len(flat_lb):
                    t_lb = np.maximum(np.asarray(res.lb, dtype=np.float64), flat_lb)
                    t_ub = np.minimum(np.asarray(res.ub, dtype=np.float64), flat_ub)
                    tightened = (t_lb, t_ub)
            except Exception:
                tightened = (None, None)
            fbbt_cache.append(tightened)
        return fbbt_cache[0]

    def maybe_add(expr: Expression) -> None:
        nonlocal col_idx
        expr_id = id(expr)
        if expr_id in seen:
            return
        op_info = _univariate_arg(expr)
        if op_info is None:
            return
        func_name, arg = op_info
        try:
            arg_coeff, arg_const = _linearize_affine_expr(arg, model, n_orig)
            arg_lb, arg_ub = _linear_expr_bounds(arg_coeff, arg_const, flat_lb, flat_ub)
        except ValueError:
            return
        eff_lb, eff_ub = flat_lb, flat_ub
        if not _univariate_domain_ok(func_name, arg_lb, arg_ub):
            # The raw node box leaves the argument out of domain (e.g. ``log`` of
            # an argument with a ``+inf`` upper bound). Implied bounds — big-M
            # rows, other constraints — may finitely bound it; retry over the
            # FBBT-tightened box before giving up and dropping the constraint.
            t_lb, t_ub = _fbbt_argument_box()
            if t_lb is None or t_ub is None:
                return
            arg_lb, arg_ub = _linear_expr_bounds(arg_coeff, arg_const, t_lb, t_ub)
            if not _univariate_domain_ok(func_name, arg_lb, arg_ub):
                return
            eff_lb, eff_ub = t_lb, t_ub
        exact_integer_range = _integer_affine_trig_range(
            func_name,
            arg_coeff,
            arg_const,
            model,
            eff_lb,
            eff_ub,
        )
        if exact_integer_range is not None:
            val_lb, val_ub = exact_integer_range
        else:
            val_lb, val_ub = _univariate_value_bounds(func_name, arg_lb, arg_ub)
        if not np.isfinite(val_lb) or not np.isfinite(val_ub):
            return
        signature = _univariate_signature(func_name, arg_coeff, arg_const)
        if signature in var_map:
            seen.add(expr_id)
            var_map[expr_id] = var_map[signature]
            return
        seen.add(expr_id)
        var_map[expr_id] = col_idx
        var_map[signature] = col_idx
        relaxations.append(
            UnivariateRelaxation(
                expr_id=expr_id,
                func_name=func_name,
                aux_col=col_idx,
                arg_coeff=arg_coeff,
                arg_const=arg_const,
                arg_lb=float(arg_lb),
                arg_ub=float(arg_ub),
            )
        )
        bounds.append((float(val_lb), float(val_ub)))
        col_idx += 1

    def visit(expr: Expression) -> None:
        maybe_add(expr)
        if isinstance(expr, BinaryOp):
            visit(expr.left)
            visit(expr.right)
        elif isinstance(expr, UnaryOp):
            visit(expr.operand)
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                visit(arg)
        elif isinstance(expr, IndexExpression):
            if not isinstance(expr.base, Variable):
                visit(expr.base)
        elif isinstance(expr, SumExpression):
            visit(expr.operand)
        elif isinstance(expr, SumOverExpression):
            for term in expr.terms:
                visit(term)

    if model._objective is not None:
        visit(model._objective.expression)
    for constraint in model._constraints:
        visit(constraint.body)

    return relaxations, var_map, bounds


def _collect_univariate_square_relaxations(
    model: Model,
    univariate_var_map: dict[object, int],
    all_bounds: list[tuple[float, float]],
    start_col: int,
) -> tuple[list[UnivariateSquareRelaxation], dict[tuple[int, int], int], list[tuple[float, float]]]:
    """Collect squares of lifted trig calls and assign auxiliary columns."""
    relaxations: list[UnivariateSquareRelaxation] = []
    var_map: dict[tuple[int, int], int] = {}
    bounds: list[tuple[float, float]] = []
    col_idx = start_col

    def maybe_add(expr: Expression) -> None:
        nonlocal col_idx
        if not (
            isinstance(expr, BinaryOp)
            and expr.op == "**"
            and isinstance(expr.left, FunctionCall)
            and expr.left.func_name in {"sin", "cos", "tan"}
            and isinstance(expr.right, Constant)
            and float(expr.right.value) == 2.0
        ):
            return

        base_col = univariate_var_map.get(id(expr.left))
        if base_col is None:
            return
        key = (base_col, 2)
        if key in var_map:
            return

        base_lb, base_ub = [float(v) for v in all_bounds[base_col]]
        vals = [base_lb * base_lb, base_ub * base_ub]
        if base_lb <= 0.0 <= base_ub:
            vals.append(0.0)
        var_map[key] = col_idx
        bounds.append((float(min(vals)), float(max(vals))))
        relaxations.append(
            UnivariateSquareRelaxation(
                base_col=base_col,
                aux_col=col_idx,
                base_lb=base_lb,
                base_ub=base_ub,
            )
        )
        col_idx += 1

    def visit(expr: Expression) -> None:
        maybe_add(expr)
        if isinstance(expr, BinaryOp):
            visit(expr.left)
            visit(expr.right)
        elif isinstance(expr, UnaryOp):
            visit(expr.operand)
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                visit(arg)
        elif isinstance(expr, IndexExpression):
            if not isinstance(expr.base, Variable):
                visit(expr.base)
        elif isinstance(expr, SumExpression):
            visit(expr.operand)
        elif isinstance(expr, SumOverExpression):
            for term in expr.terms:
                visit(term)

    if model._objective is not None:
        visit(model._objective.expression)
    for constraint in model._constraints:
        visit(constraint.body)

    return relaxations, var_map, bounds


# ---------------------------------------------------------------------------
# Helpers: expression linearizer
# ---------------------------------------------------------------------------


def _linearize_expr(
    expr: Expression,
    model: Model,
    bilinear_var_map: dict[tuple[int, int], int],
    trilinear_var_map: dict[tuple[int, int, int], int],
    multilinear_var_map: dict[tuple[int, ...], int],
    monomial_var_map: dict[tuple[int, int], int],
    univariate_var_map: dict[object, int],
    n_total_vars: int,
    fractional_power_var_map: Optional[dict[tuple[int, float], int]] = None,
    univariate_square_var_map: Optional[dict[tuple[int, int], int]] = None,
    flat_lb: Optional[np.ndarray] = None,
    flat_ub: Optional[np.ndarray] = None,
    composite_var_map: Optional[dict[int, int]] = None,
    composite_coeff_map: Optional[dict[int, float]] = None,
) -> tuple[np.ndarray, float]:
    """Walk expression tree and return (coeff, constant) for linearized form.

    coeff[j] = coefficient of MILP variable j in the linear approximation.
    constant = scalar constant term.

    Nonlinear terms must have a corresponding auxiliary variable in the maps;
    raises ValueError if an unregistered nonlinear term is encountered.

    When ``flat_lb``/``flat_ub`` are supplied, any nonlinear term whose base
    variable is *pinned* at this node (``ub - lb`` within tolerance — produced
    by spatial branching driving a domain to a point) is folded to its exact
    constant value instead of requiring an auxiliary column. The aux-column
    builders intentionally skip degenerate (zero-width) domains, so without
    this fold a pinned monomial/fractional-power term would fail to linearize
    and silently sink the node's objective bound to "feasibility only".
    """
    coeff = np.zeros(n_total_vars, dtype=np.float64)
    const_acc: list[float] = [0.0]
    n_orig = sum(var.size for var in model._variables)

    _PIN_TOL = 1e-9

    def _pinned_value(idx: int) -> Optional[float]:
        """Exact value of variable ``idx`` if pinned at this node, else None."""
        if flat_lb is None or flat_ub is None or idx >= len(flat_lb):
            return None
        lo = float(flat_lb[idx])
        hi = float(flat_ub[idx])
        if hi - lo <= _PIN_TOL:
            return 0.5 * (lo + hi)
        return None

    def visit(e: Expression, scale: float) -> None:  # noqa: C901
        if composite_var_map is not None:
            aux_col = composite_var_map.get(id(e))
            if aux_col is not None:
                sub_coeff = composite_coeff_map.get(id(e), 1.0) if composite_coeff_map else 1.0
                coeff[aux_col] += scale * sub_coeff
                return

        if isinstance(e, Constant):
            const_acc[0] += scale * float(e.value)

        elif isinstance(e, Variable):
            offset = _compute_var_offset(e, model)
            if e.size == 1:
                coeff[offset] += scale
            else:
                # Multi-element variable (unusual in scalar expression)
                for k in range(e.size):
                    coeff[offset + k] += scale

        elif isinstance(e, IndexExpression):
            flat = _get_flat_index(e, model)
            if flat is not None:
                coeff[flat] += scale
            else:
                raise ValueError(f"Cannot linearize IndexExpression: {e}")

        elif isinstance(e, FunctionCall):
            aux_col = univariate_var_map.get(id(e))
            if aux_col is not None:
                coeff[aux_col] += scale
            else:
                raise ValueError(f"Cannot linearize FunctionCall: {e}")

        elif isinstance(e, BinaryOp):
            if e.op == "+":
                visit(e.left, scale)
                visit(e.right, scale)

            elif e.op == "-":
                visit(e.left, scale)
                visit(e.right, -scale)

            elif e.op == "/":
                if isinstance(e.right, Constant):
                    visit(e.left, scale / float(e.right.value))
                elif isinstance(e.left, Constant):
                    # c / (x**p)  →  fractional-power aux column for x**-p
                    # (e.g. 1/(x**3*sqrt(x)) → x**-3.5 in nvs08). Try this before
                    # the reciprocal univariate path so a monomial-product
                    # denominator is relaxed instead of dropped.
                    fp_aux = None
                    if fractional_power_var_map is not None:
                        recip = extract_reciprocal_power(e, model)
                        if recip is not None:
                            fp_idx, fp_exp, fp_coeff = recip
                            pinned = _pinned_value(fp_idx)
                            if pinned is not None:
                                const_acc[0] += scale * fp_coeff * (pinned**fp_exp)
                                return
                            fp_aux = fractional_power_var_map.get((fp_idx, float(fp_exp)))
                            if fp_aux is not None:
                                coeff[fp_aux] += scale * fp_coeff
                                return
                    aux_col = univariate_var_map.get(id(e))
                    if aux_col is None:
                        try:
                            arg_coeff, arg_const = _linearize_affine_expr(e.right, model, n_orig)
                        except ValueError:
                            aux_col = None
                        else:
                            aux_col = univariate_var_map.get(
                                _univariate_signature("reciprocal", arg_coeff, arg_const)
                            )
                    if aux_col is not None:
                        coeff[aux_col] += scale * float(e.left.value)
                    else:
                        raise ValueError(f"Cannot linearize non-constant division: {e}")
                else:
                    raise ValueError(f"Cannot linearize non-constant division: {e}")

            elif e.op == "**":
                flat = _get_flat_index(e.left, model)
                if flat is not None and isinstance(e.right, Constant):
                    exp_val = float(e.right.value)
                    n_int = int(exp_val)
                    if exp_val == n_int:
                        if n_int == 1:
                            coeff[flat] += scale
                            return
                        if n_int == 0:
                            const_acc[0] += scale
                            return
                        if n_int >= 2:
                            key = (flat, n_int)
                            if key in monomial_var_map:
                                coeff[monomial_var_map[key]] += scale
                                return
                            pinned = _pinned_value(flat)
                            if pinned is not None:
                                const_acc[0] += scale * (pinned**n_int)
                                return
                            raise ValueError(f"Monomial {key} not in monomial_var_map")
                    fp_key = (flat, exp_val)
                    if fractional_power_var_map and fp_key in fractional_power_var_map:
                        coeff[fractional_power_var_map[fp_key]] += scale
                        return
                    pinned = _pinned_value(flat)
                    if pinned is not None and pinned >= 0.0:
                        const_acc[0] += scale * (pinned**exp_val)
                        return
                    raise ValueError(f"Fractional power {fp_key} has no aux column")
                raise ValueError(f"Cannot linearize power expression: {e}")

            elif e.op == "*":
                # Constant scaling? Fold a variable-free factor (including
                # composite constants such as ``neg(2.5)`` or ``(-3)*(-3)``)
                # before attempting product decomposition. Without this, a term
                # like ``(-2.5) * x`` (parsed as ``neg(2.5) * x``) is treated as
                # an undecomposable bilinear product and the whole constraint is
                # dropped from the relaxation, weakening the bound.
                lconst = _eval_constant_expr(e.left)
                if lconst is not None:
                    visit(e.right, scale * lconst)
                    return
                rconst = _eval_constant_expr(e.right)
                if rconst is not None:
                    visit(e.left, scale * rconst)
                    return
                # Full product decomposition
                decomp = _decompose_product(
                    e,
                    model,
                    fractional_power_var_map=fractional_power_var_map,
                    univariate_var_map=univariate_var_map,
                    monomial_var_map=monomial_var_map,
                    composite_var_map=composite_var_map,
                    composite_coeff_map=composite_coeff_map,
                )
                if decomp is None:
                    raise ValueError(f"Cannot decompose product: {e}")
                c, indices = decomp
                # Fold pinned factors (lb==ub at this node) into the constant
                # coefficient — their value is exact, lowering the product's
                # arity so a partially-pinned bilinear/trilinear collapses to a
                # lower-order term (or a constant) instead of demanding an aux
                # column the builder skipped for the degenerate domain.
                if (flat_lb is not None or flat_ub is not None) and indices:
                    kept_indices: list[int] = []
                    for idx in indices:
                        pinned = _pinned_value(idx)
                        if pinned is not None:
                            c *= pinned
                        else:
                            kept_indices.append(idx)
                    indices = kept_indices
                unique = list(dict.fromkeys(indices))
                if len(indices) == 0:
                    const_acc[0] += scale * c
                elif len(unique) == 1 and len(indices) == 1:
                    coeff[unique[0]] += scale * c
                elif len(unique) == 1:
                    # x^n monomial
                    n = len(indices)
                    key = (unique[0], n)
                    if key in monomial_var_map:
                        coeff[monomial_var_map[key]] += scale * c
                    elif univariate_square_var_map and key in univariate_square_var_map:
                        coeff[univariate_square_var_map[key]] += scale * c
                    else:
                        raise ValueError(f"Monomial {key} not in map")
                elif len(unique) == 2:
                    if len(unique) != len(indices):
                        raise ValueError("Mixed repeated-factor products are not supported")
                    i_idx, j_idx = unique[0], unique[1]
                    key = (min(i_idx, j_idx), max(i_idx, j_idx))
                    if key in bilinear_var_map:
                        coeff[bilinear_var_map[key]] += scale * c
                    else:
                        raise ValueError(f"Bilinear {key} not in map")
                elif len(unique) == 3:
                    if len(unique) != len(indices):
                        raise ValueError("Mixed repeated-factor products are not supported")
                    ordered = sorted(unique)
                    tri_key = (ordered[0], ordered[1], ordered[2])
                    if tri_key in trilinear_var_map:
                        coeff[trilinear_var_map[tri_key]] += scale * c
                    else:
                        raise ValueError(f"Trilinear {tri_key} not in map")
                else:
                    if len(unique) != len(indices):
                        raise ValueError("Mixed repeated-factor products are not supported")
                    multilinear_key = tuple(sorted(unique))
                    if multilinear_key in multilinear_var_map:
                        coeff[multilinear_var_map[multilinear_key]] += scale * c
                    else:
                        raise ValueError(f"Multilinear {multilinear_key} not in map")

            else:
                raise ValueError(f"Cannot linearize BinaryOp: {e.op}")

        elif isinstance(e, UnaryOp):
            if e.op == "neg":
                visit(e.operand, -scale)
            elif e.op == "abs":
                aux_col = univariate_var_map.get(id(e))
                if aux_col is not None:
                    coeff[aux_col] += scale
                else:
                    raise ValueError(f"Cannot linearize UnaryOp: {e.op}")
            else:
                raise ValueError(f"Cannot linearize UnaryOp: {e.op}")

        elif isinstance(e, SumExpression):
            op = e.operand
            if isinstance(op, Variable):
                offset = _compute_var_offset(op, model)
                for k in range(op.size):
                    coeff[offset + k] += scale
            else:
                visit(op, scale)

        elif isinstance(e, SumOverExpression):
            for term in e.terms:
                visit(term, scale)

        else:
            raise ValueError(f"Cannot linearize {type(e).__name__}: {e}")

    visit(expr, 1.0)
    return coeff, const_acc[0]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_milp_relaxation(
    model: Model,
    terms: NonlinearTerms,
    disc_state: DiscretizationState,
    incumbent: Optional[np.ndarray] = None,
    oa_cuts: Optional[list] = None,
    convhull_formulation: str = "disaggregated",
    convhull_ebd: bool = False,
    convhull_ebd_encoding: str = "gray",
    bound_override: Optional[tuple[np.ndarray, np.ndarray]] = None,
    superposition: bool = False,
    rlt_level1: bool = False,
) -> tuple["MilpRelaxationModel", dict]:
    """Build a MILP relaxation with piecewise McCormick for bilinear/monomial terms.

    For each bilinear term x_i*x_j: adds standard McCormick envelope constraints
    (4 linear inequalities).  These give the convex hull of the bilinear set on the
    bounding box and are independent of the partition (piecewise refinement via binary
    variables is left for future enhancement).

    For each monomial x_i^n (currently n=2 handled precisely):
    - Piecewise tangent underestimators (one per partition interval midpoint) — gets
      tighter as disc_state gains more intervals.
    - Global secant overestimator — bounds s from above.

    The LP objective and constraints are obtained by substituting auxiliary vars for
    all nonlinear terms.

    Parameters
    ----------
    model : Model
    terms : NonlinearTerms
        Output of classify_nonlinear_terms(model).
    disc_state : DiscretizationState
        Current partition; provides intervals for tangent cut placement.
    incumbent : np.ndarray, optional
        Current best NLP solution (flat).  Used to add OA tangent cuts for
        general nonlinear terms; currently unused (reserved for future use).
    convhull_formulation : str, default "disaggregated"
        Piecewise bilinear formulation. ``"disaggregated"`` keeps the existing
        xbar/wbar construction; ``"sos2"`` and ``"facet"`` use a λ-based
        convex-hull reformulation similar to Alpine.jl.
    convhull_ebd : bool, default False
        Replace SOS2 interval binaries with a logarithmic embedded encoding.
        Only supported with ``convhull_formulation="sos2"`` or ``"lambda"``.
    convhull_ebd_encoding : str, default "gray"
        Embedded encoding scheme. ``"gray"`` is the Alpine-style default and
        the only option that remains SOS2-compatible for arbitrary partition
        counts. ``"binary"`` is only valid for two partitions.

    Returns
    -------
    (MilpRelaxationModel, varmap)
        MilpRelaxationModel has a .solve() method returning MilpRelaxationResult.
        varmap maps auxiliary variable keys to MILP column indices.
    """
    if bound_override is None:
        flat_lb, flat_ub = flat_variable_bounds(model)
    else:
        flat_lb = np.asarray(bound_override[0], dtype=np.float64)
        flat_ub = np.asarray(bound_override[1], dtype=np.float64)
    n_orig = len(flat_lb)
    convhull_mode = _normalize_convhull_formulation(convhull_formulation)
    if convhull_ebd and convhull_mode != "sos2":
        raise ValueError(
            "convhull_ebd is only supported with convhull_formulation='sos2' or its 'lambda' alias."
        )
    generation_guardrails: list[str] = []
    generation_guardrail_keys: set[tuple[str, str, int, int]] = set()
    objective_lift = _build_minmax_objective_lift(model, flat_lb, flat_ub)
    objective_lift_monomials: set[tuple[int, int]] = set()
    if objective_lift is not None:
        for branch_expr in objective_lift.branch_exprs:
            objective_lift_monomials.update(_collect_monomial_terms_for_lift(branch_expr, model))
    elif model._objective is not None:
        # Regular (non-minmax) objective: collect monomial sub-terms from
        # repeated-factor products such as ``x**2 * y`` (issue #139, bucket 2).
        # The term classifier punts mixed repeated-factor products to
        # ``general_nl`` without recording the constituent monomial, so the
        # objective term would otherwise drop (objective_bound_valid=False).
        # Lifting the monomial (``x**2``) lets the product relax via one
        # monomial envelope + one lifted bilinear envelope — both rigorous
        # McCormick underestimators — instead of dropping the objective.
        objective_lift_monomials.update(
            _collect_monomial_terms_for_lift(
                distribute_products(model._objective.expression), model
            )
        )
    # ── Square-of-affine-in-lifted-vars residuals (issue #155) ──────────────
    # ``(1.5 - x0*(1-x1))**2``-style terms are lifted wholesale to a univariate
    # square envelope on the affine residual rather than distributed into the
    # catastrophic high-degree monomials (~1e18 on nvs16) that the magnitude cap
    # would otherwise drop. Collect the squares now so every single-variable
    # power factor (``x1**2``, ``x1**3``) the residual needs is allocated through
    # the standard monomial machinery (column + tangent/secant envelope) below.
    affine_squares = _collect_affine_squares(model)
    affine_square_monomials: set[tuple[int, int]] = set()
    for _node, (_const, sq_terms) in affine_squares:
        for _coeff, monomial in sq_terms:
            for var_idx, power in Counter(monomial).items():
                if power >= 2:
                    affine_square_monomials.add((var_idx, power))
    # ── Multi-factor univariate products in CONSTRAINTS (st_e40) ────────────
    # A constraint such as ``(i-1)*(i-2)*...*(i-12) == 0`` (st_e40's degree-7
    # set-membership polynomials forcing ``i in {1,2,3,5,8,10,12}``) exposes its
    # monomials ``i**2 .. i**7`` only AFTER distribution. The objective is
    # already distributed before monomial collection (above), but constraints
    # were not — so those monomials were never registered, the linearizer raised
    # ``"Monomial (i, k) not in map"`` and DROPPED THE WHOLE CONSTRAINT from the
    # relaxation. The result was a uselessly loose dual bound (st_e40: root bound
    # 4.41 against a true optimum of 30.41, so the spatial B&B never prunes).
    # Collecting them here registers the aux columns + rigorous monomial power
    # envelopes so the constraint is KEPT. Sound: re-including a previously
    # dropped constraint only shrinks the relaxed feasible set toward the true
    # one, so the bound can only rise toward (never above) the optimum.
    constraint_lift_monomials: set[tuple[int, int]] = set()
    for _con in model._constraints:
        try:
            constraint_lift_monomials.update(
                _collect_monomial_terms_for_lift(distribute_products(_con.body), model)
            )
        except Exception:  # pragma: no cover - defensive: never break the build
            continue
    monomial_terms = sorted(
        set(terms.monomial)
        | objective_lift_monomials
        | affine_square_monomials
        | constraint_lift_monomials
    )

    def _record_generation_guardrail(
        kind: str,
        target: object,
        interval_count: int,
        limit: int,
    ) -> None:
        key = (kind, repr(target), int(interval_count), int(limit))
        if key in generation_guardrail_keys:
            return
        generation_guardrail_keys.add(key)
        note = (
            f"skipped {kind} refinement for {target}: "
            f"{interval_count} intervals exceeds cap {limit}"
        )
        generation_guardrails.append(note)
        logger.debug("AMP: %s", note)

    def _guarded_partition_points(
        kind: str,
        target: object,
        points: list[float] | np.ndarray,
    ) -> Optional[list[float]]:
        finite_points = [float(p) for p in points if np.isfinite(float(p))]
        guarded = _sorted_unique_points(finite_points)
        interval_count = max(0, len(guarded) - 1)
        if interval_count > _MAX_RELAXATION_PARTITION_INTERVALS:
            _record_generation_guardrail(
                kind,
                target,
                interval_count,
                _MAX_RELAXATION_PARTITION_INTERVALS,
            )
            return None
        return guarded

    def _coarse_monomial_breakpoints(lb_i: float, ub_i: float) -> list[float]:
        points = [lb_i, ub_i]
        if lb_i < 0.0 < ub_i:
            points.append(0.0)
        return _sorted_unique_points(points)

    def _monomial_aux_bounds(lb_i: float, ub_i: float, n: int) -> tuple[float, float]:
        """Return safe auxiliary bounds for ``s = x**n``.

        Effectively infinite original bounds cannot support numerically useful
        tangent/secant rows, but the auxiliary still lets constraints and
        objectives reference the lifted monomial instead of dropping the whole
        expression from the MILP relaxation.
        """
        lb_finite = _is_effectively_finite(lb_i)
        ub_finite = _is_effectively_finite(ub_i)
        if lb_finite and ub_finite:
            vals = [lb_i**n, ub_i**n]
            if n % 2 == 0 and lb_i < 0 < ub_i:
                vals.append(0.0)
            return min(vals), max(vals)

        if n % 2 == 0:
            lower = 0.0
            if lb_finite and lb_i > 0.0:
                lower = lb_i**n
            elif ub_finite and ub_i < 0.0:
                lower = ub_i**n
            return float(lower), np.inf

        lower = lb_i**n if lb_finite else -np.inf
        upper = ub_i**n if ub_finite else np.inf
        return float(lower), float(upper)

    # ── Assign MILP column indices ──────────────────────────────────────────
    # Original variables keep columns 0..n_orig-1. Additional columns are created
    # for lifted bilinear, trilinear, and monomial terms plus any piecewise binaries.
    bilinear_var_map: dict[tuple[int, int], int] = {}
    trilinear_var_map: dict[tuple[int, int, int], int] = {}
    trilinear_stage_map: dict[tuple[int, int, int], dict[str, object]] = {}
    multilinear_var_map: dict[tuple[int, ...], int] = {}
    multilinear_stage_map: dict[tuple[int, ...], list[dict[str, int]]] = {}
    monomial_var_map: dict[tuple[int, int], int] = {}
    univariate_var_map: dict[object, int] = {}
    fractional_power_var_map: dict[tuple[int, float], int] = {}

    col_idx = n_orig
    all_bounds: list[tuple[float, float]] = list(zip(flat_lb.tolist(), flat_ub.tolist()))
    integrality_flags: list[int] = []
    for v in model._variables:
        flag = 1 if v.var_type in (VarType.BINARY, VarType.INTEGER) else 0
        integrality_flags.extend([flag] * v.size)

    bilinear_relation_map: dict[tuple[int, int], int] = {}

    def _ensure_bilinear_aux(lhs_col: int, rhs_col: int) -> int:
        nonlocal col_idx
        key = (min(lhs_col, rhs_col), max(lhs_col, rhs_col))
        if key in bilinear_relation_map:
            return bilinear_relation_map[key]

        lhs_lb, lhs_ub = all_bounds[key[0]]
        rhs_lb, rhs_ub = all_bounds[key[1]]
        corners = [
            lhs_lb * rhs_lb,
            lhs_lb * rhs_ub,
            lhs_ub * rhs_lb,
            lhs_ub * rhs_ub,
        ]
        bilinear_relation_map[key] = col_idx
        all_bounds.append((min(corners), max(corners)))
        integrality_flags.append(0)
        col_idx += 1
        return bilinear_relation_map[key]

    def _ensure_multilinear_aux(term: tuple[int, ...]) -> tuple[int, list[dict[str, int]]]:
        ordered = tuple(sorted(term))
        if len(ordered) < 2:
            raise ValueError("multilinear terms require at least two variables")

        stages: list[dict[str, int]] = []
        current_col = ordered[0]
        for rhs_col in ordered[1:]:
            lhs_col = current_col
            product_col = _ensure_bilinear_aux(lhs_col, rhs_col)
            stages.append(
                {
                    "lhs_col": lhs_col,
                    "rhs_col": rhs_col,
                    "product_col": product_col,
                }
            )
            current_col = product_col
        return current_col, stages

    original_bilinear_keys = sorted({(min(i, j), max(i, j)) for i, j in terms.bilinear})
    for key in original_bilinear_keys:
        bilinear_var_map[key] = _ensure_bilinear_aux(*key)

    partitioned_vars = set(disc_state.partitions)
    trilinear_terms: list[tuple[int, int, int]] = []
    for term in terms.trilinear:
        ordered = sorted(term)
        canonical = (ordered[0], ordered[1], ordered[2])
        if canonical not in trilinear_terms:
            trilinear_terms.append(canonical)

    def _builder_pinned(idx: int) -> bool:
        """True if original variable ``idx`` is pinned (lb==ub) at this node."""
        if idx >= len(flat_lb):
            return False
        return float(flat_ub[idx]) - float(flat_lb[idx]) <= 1e-9

    for term in sorted(trilinear_terms):
        pair, remaining = _choose_trilinear_pair(term, partitioned_vars)
        pair_col = _ensure_bilinear_aux(*pair)
        final_col = _ensure_bilinear_aux(pair_col, remaining)
        trilinear_var_map[term] = final_col
        trilinear_stage_map[term] = {
            "pair": pair,
            "pair_col": pair_col,
            "remaining_var": remaining,
            "product_col": final_col,
        }
        # Pinned-factor collapse: when exactly one factor is pinned at this node
        # (lb==ub from branching/OBBT), ``_linearize_expr`` folds that factor's
        # exact value into the coefficient, leaving a *bilinear* in the other two
        # factors. Pre-allocate that bilinear's aux column + McCormick envelope so
        # the collapsed term still linearizes; otherwise the linearizer raises
        # "Bilinear (i,j) not in map" and the whole objective/constraint drops
        # (issue #139: surfaces on nvs22 once its x**2*y objective term lifts).
        unpinned = [v for v in term if not _builder_pinned(v)]
        if len(unpinned) == 2:
            bkey = (min(unpinned), max(unpinned))
            if bkey not in bilinear_var_map:
                bilinear_var_map[bkey] = _ensure_bilinear_aux(*bkey)

    multilinear_terms = terms.multilinear or _collect_distinct_multilinear_products(model)
    for multi_term in multilinear_terms:
        final_col, stages = _ensure_multilinear_aux(multi_term)
        multilinear_var_map[multi_term] = final_col
        multilinear_stage_map[multi_term] = stages

    # Numerical-conditioning guard: a lifted monomial whose auxiliary bound spans
    # an extreme magnitude (e.g. ``x1**9`` over ``[15, 25]`` → ~3.8e12) injects
    # coefficients ranging over >1e12 into the LP. The relaxation is still
    # logically sound — it contains every true feasible point — but the badly
    # scaled system drives the LP solver to report *spurious* infeasibility
    # (a false infeasibility, as serious as a false bound: a feasible problem
    # would look infeasible and produce no dual bound). Such a monomial is not
    # lifted; the linearizer then raises on the missing aux and the containing
    # constraint is omitted entirely (omission only enlarges the feasible region,
    # so the dual bound stays valid). The cap sits far above every monomial any
    # currently-bounding instance needs (the largest, nvs20/nvs21, peak at
    # ~1.6e9) and triggers only on pathological high-degree polynomials such as
    # st_e36's degree-10 equality constraint (issue #139, bucket 2).
    _MONOMIAL_AUX_BOUND_LIMIT = 1e10
    _kept_monomial_terms: list[tuple[int, int]] = []
    for var_idx, n in monomial_terms:
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        blo, bhi = _monomial_aux_bounds(lb_i, ub_i, n)
        mag = max(
            abs(blo) if np.isfinite(blo) else 0.0,
            abs(bhi) if np.isfinite(bhi) else 0.0,
        )
        if mag > _MONOMIAL_AUX_BOUND_LIMIT:
            logger.debug(
                "AMP: not lifting monomial x%d**%d (aux bound magnitude %.3g exceeds "
                "%.0e); constraints/terms referencing it are omitted to avoid a "
                "numerically degenerate LP relaxation.",
                var_idx,
                n,
                mag,
                _MONOMIAL_AUX_BOUND_LIMIT,
            )
            continue
        _kept_monomial_terms.append((var_idx, n))
    monomial_terms = _kept_monomial_terms

    for var_idx, n in monomial_terms:
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        monomial_var_map[(var_idx, n)] = col_idx
        all_bounds.append(_monomial_aux_bounds(lb_i, ub_i, n))
        integrality_flags.append(0)
        col_idx += 1

    # ── Level-1 RLT product-column setup (issue #175) ───────────────────────
    # Multiplying each linear constraint factor ``g(x) <= 0`` by a variable bound
    # factor (``x_m - l_m >= 0`` and ``u_m - x_m >= 0``) yields a valid product
    # inequality. Linearizing it replaces every ``x_i*x_m`` by a lifted product
    # column (its McCormick envelope auto-emits through ``bilinear_relation_map``);
    # the resulting cut couples the loose product relaxation to the model's linear
    # structure and tightens the root bound for high-degree-product instances whose
    # variable boxes include 0 (nvs20). The cut value at any true ``(x, x∘x)`` point
    # equals ``g(x)·(bound factor) <= 0``, so each row is a valid relaxation cut and
    # can never exclude a feasible point. Resolved here (before the column count is
    # frozen) so the product columns exist; the rows are emitted with the others.
    rlt_cut_specs: list[dict[str, Any]] = []
    rlt_quad_specs: list[dict[str, Any]] = []
    if rlt_level1:

        def _rlt_product_col(i: int, mm: int) -> Optional[int]:
            lo_i, hi_i = float(flat_lb[i]), float(flat_ub[i])
            lo_m, hi_m = float(flat_lb[mm]), float(flat_ub[mm])
            if not (
                _is_effectively_finite(lo_i)
                and _is_effectively_finite(hi_i)
                and _is_effectively_finite(lo_m)
                and _is_effectively_finite(hi_m)
            ):
                return None
            if i == mm:
                # Prefer the tighter convex monomial envelope for x_m**2 when present.
                mono_col = monomial_var_map.get((mm, 2))
                if mono_col is not None:
                    return mono_col
            return _ensure_bilinear_aux(i, mm)

        for coeff_form, const_form in _linear_constraint_forms(model, n_orig):
            support = [i for i in range(n_orig) if coeff_form[i] != 0.0]
            if not support:
                continue
            for mm in range(n_orig):
                lo_m, hi_m = float(flat_lb[mm]), float(flat_ub[mm])
                if not (_is_effectively_finite(lo_m) and _is_effectively_finite(hi_m)):
                    continue
                prod_cols: dict[int, int] = {}
                ok = True
                for i in support:
                    col = _rlt_product_col(i, mm)
                    if col is None:
                        ok = False
                        break
                    prod_cols[i] = col
                if not ok:
                    continue
                rlt_cut_specs.append(
                    {
                        "coeff": coeff_form,
                        "const": float(const_form),
                        "m": mm,
                        "lm": lo_m,
                        "um": hi_m,
                        "support": support,
                        "prod_cols": prod_cols,
                    }
                )

        # ── Phase 2: nonlinear (quadratic) constraint-factor RLT (issue #15) ──
        # Multiply a *quadratic* constraint factor ``g(x) {<=,==} 0`` by a variable
        # bound factor and lift the resulting degree-3 monomials on demand. This is
        # the textbook gap for nonconvex quadratic (e.g. Weymouth) equalities. The
        # product columns (bilinears, mixed ``x_i**2 x_j``, distinct triples, pure
        # cubes) are allocated here while the column count is still growing; the
        # rows are assembled at emission time by the single-sourced
        # ``rlt_quadratic_bound_cut_row`` so the deployed math is exactly what the
        # soundness audit exercises. Selective (only constraints whose variables
        # touch the model's nonlinear support) and capped (bounded new columns) so
        # the LP does not blow up. Disable with ``DISCOPT_RLT_QUAD=0``.
        if os.environ.get("DISCOPT_RLT_QUAD", "1") != "0":
            nonconvex_vars: set[int] = set()
            for bi, bj in terms.bilinear:
                nonconvex_vars.update((bi, bj))
            for tri in terms.trilinear:
                nonconvex_vars.update(tri)
            for multi in terms.multilinear or ():
                nonconvex_vars.update(multi)
            for mono_idx, _mono_pow in monomial_terms:
                nonconvex_vars.add(mono_idx)

            _quad_col_cap = int(os.environ.get("DISCOPT_RLT_QUAD_MAX", "256"))
            _quad_cols_start = col_idx

            def _ensure_monomial_aux(i: int, p: int) -> Optional[int]:
                """Lift ``x_i**p`` on demand (registering its power envelope)."""
                nonlocal col_idx
                key = (i, p)
                existing = monomial_var_map.get(key)
                if existing is not None:
                    return existing
                lb_i = float(flat_lb[i])
                ub_i = float(flat_ub[i])
                if not (_is_effectively_finite(lb_i) and _is_effectively_finite(ub_i)):
                    return None
                blo, bhi = _monomial_aux_bounds(lb_i, ub_i, p)
                mag = max(
                    abs(blo) if np.isfinite(blo) else 0.0,
                    abs(bhi) if np.isfinite(bhi) else 0.0,
                )
                if mag > _MONOMIAL_AUX_BOUND_LIMIT:
                    return None
                if col_idx - _quad_cols_start >= _quad_col_cap:
                    return None
                monomial_var_map[key] = col_idx
                all_bounds.append((blo, bhi))
                integrality_flags.append(0)
                col_idx += 1
                # Register so the secant/tangent power envelope is emitted later.
                monomial_terms.append(key)
                return monomial_var_map[key]

            def _ensure_product_col(idxs: tuple[int, ...]) -> Optional[int]:
                """Column for a degree-1..3 monomial given as a multiset of vars."""
                key = tuple(sorted(idxs))
                if col_idx - _quad_cols_start >= _quad_col_cap:
                    # Still allow products whose columns already exist.
                    pass
                if len(key) == 1:
                    return key[0]
                if len(key) == 2:
                    a, b = key
                    if a == b:
                        return _ensure_monomial_aux(a, 2)
                    return _ensure_bilinear_aux(a, b)
                if len(key) == 3:
                    a, b, c = key
                    if a == b == c:
                        return _ensure_monomial_aux(a, 3)
                    if a == b:  # x_a**2 * x_c
                        sq = _ensure_monomial_aux(a, 2)
                        if sq is None or col_idx - _quad_cols_start >= _quad_col_cap:
                            return None
                        return _ensure_bilinear_aux(sq, c)
                    if b == c:  # x_a * x_b**2
                        sq = _ensure_monomial_aux(b, 2)
                        if sq is None or col_idx - _quad_cols_start >= _quad_col_cap:
                            return None
                        return _ensure_bilinear_aux(a, sq)
                    if col_idx - _quad_cols_start >= _quad_col_cap:
                        return None
                    final_col, _stages = _ensure_multilinear_aux((a, b, c))
                    return final_col
                return None

            for quad, lin, qconst, sense in _quadratic_constraint_forms(model, n_orig):
                support_vars = set(lin) | {k for k, _l in quad} | {_l for _k, _l in quad}
                if not support_vars & nonconvex_vars:
                    continue  # selective: only factors touching nonlinear structure
                for mm in sorted(support_vars):
                    lo_m, hi_m = float(flat_lb[mm]), float(flat_ub[mm])
                    if not (_is_effectively_finite(lo_m) and _is_effectively_finite(hi_m)):
                        continue
                    # Enumerate and lift every product the cut row will reference.
                    required: set[tuple[int, ...]] = set()
                    for i in lin:
                        required.add(tuple(sorted((i, mm))))
                    for ka, kb in quad:
                        required.add((ka, kb))
                        required.add(tuple(sorted((ka, kb, mm))))
                    prod_map: dict[tuple[int, ...], int] = {}
                    ok = True
                    for req in required:
                        col = _ensure_product_col(req)
                        if col is None:
                            ok = False
                            break
                        prod_map[req] = col
                    if not ok:
                        continue
                    rlt_quad_specs.append(
                        {
                            "quad": quad,
                            "lin": lin,
                            "const": qconst,
                            "m": mm,
                            "lm": lo_m,
                            "um": hi_m,
                            "sense": sense,
                            "prod_map": prod_map,
                        }
                    )

    monomial_pw_map: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for var_idx, n in monomial_terms:
        if (var_idx, n) not in monomial_var_map:
            continue
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        if (
            var_idx not in disc_state.partitions
            or not _power_is_convex_on_box(n, lb_i)
            or not _is_effectively_finite(lb_i)
            or not _is_effectively_finite(ub_i)
        ):
            continue

        breakpoints = _guarded_partition_points(
            "monomial piecewise",
            (var_idx, n),
            _monomial_breakpoints(var_idx, lb_i, ub_i, disc_state),
        )
        if breakpoints is None:
            continue
        if len(breakpoints) < 3:
            continue

        monomial_intervals: list[tuple[int, float, float]] = []
        for a_k, b_k in zip(breakpoints[:-1], breakpoints[1:]):
            if b_k <= a_k:
                continue
            delta_col = col_idx
            all_bounds.append((0.0, 1.0))
            integrality_flags.append(1)
            col_idx += 1
            monomial_intervals.append((delta_col, float(a_k), float(b_k)))

        if monomial_intervals:
            monomial_pw_map[(var_idx, n)] = monomial_intervals

    univariate_relaxations, univariate_var_map, univariate_bounds = _collect_univariate_relaxations(
        model,
        n_orig,
        flat_lb,
        flat_ub,
        col_idx,
    )
    for val_bounds in univariate_bounds:
        all_bounds.append(val_bounds)
        integrality_flags.append(0)
        col_idx += 1

    # Single-variable composite nodes (sqrt-of-quadratic, affine-base power,
    # exp-of-rational, …) whose curvature is certified sound on the node box.
    _univariate_claimed_ids = {k for k in univariate_var_map if isinstance(k, int)}
    composite_relaxations, composite_var_map, composite_bounds = (
        _collect_composite_univariate_relaxations(
            model,
            n_orig,
            flat_lb,
            flat_ub,
            col_idx,
            _univariate_claimed_ids,
        )
    )
    for val_bounds in composite_bounds:
        all_bounds.append(val_bounds)
        integrality_flags.append(0)
        col_idx += 1

    # Per-node substitution coefficient for composite auxes whose value is a
    # scalar multiple of the aux column (ratio-of-products lift, issue #185).
    composite_coeff_map: dict[int, float] = {}
    # Linear-fractional ``r·q = m`` envelopes accumulated by the ratio lift.
    ratio_relaxations: list[RatioRelaxation] = []

    univariate_square_relaxations, univariate_square_var_map, univariate_square_bounds = (
        _collect_univariate_square_relaxations(
            model,
            univariate_var_map,
            all_bounds,
            col_idx,
        )
    )
    for val_bounds in univariate_square_bounds:
        all_bounds.append(val_bounds)
        integrality_flags.append(0)
        col_idx += 1

    univariate_by_aux_col = {relax.aux_col: relax for relax in univariate_relaxations}
    finite_domain_trig_square_tables: list[FiniteDomainTrigSquareTable] = []
    for square_relax in univariate_square_relaxations:
        base_relax = univariate_by_aux_col.get(square_relax.base_col)
        if base_relax is None:
            continue
        table_values = _finite_domain_trig_square_table_values(
            base_relax,
            model,
            flat_lb,
            flat_ub,
        )
        if table_values is None:
            continue
        var_idx, arg_coeff, arg_const, domain_values, trig_values, square_values = table_values

        selector_cols: list[int] = []
        if len(domain_values) > 1:
            for _ in domain_values:
                selector_cols.append(col_idx)
                all_bounds.append((0.0, 1.0))
                integrality_flags.append(1)
                col_idx += 1

        finite_domain_trig_square_tables.append(
            FiniteDomainTrigSquareTable(
                square=square_relax,
                func_name=base_relax.func_name,
                var_idx=var_idx,
                arg_coeff=arg_coeff,
                arg_const=arg_const,
                domain_values=domain_values,
                trig_values=trig_values,
                square_values=square_values,
                selector_cols=selector_cols,
            )
        )

    piecewise_trig_square_relaxations: list[PiecewiseTrigSquareRelaxation] = []
    for square_relax in univariate_square_relaxations:
        base_relax = univariate_by_aux_col.get(square_relax.base_col)
        if base_relax is None or base_relax.func_name not in {"sin", "cos"}:
            continue
        if not _affine_argument_has_continuous_var(base_relax.arg_coeff, model):
            continue
        interval_specs = _trig_square_piecewise_interval_specs(base_relax, disc_state, n_orig)
        if not interval_specs:
            continue

        trig_square_intervals: list[PiecewiseTrigSquareInterval] = []
        for a_k, b_k, curvature in interval_specs:
            delta_col = col_idx
            all_bounds.append((0.0, 1.0))
            integrality_flags.append(1)
            col_idx += 1
            trig_square_intervals.append(
                PiecewiseTrigSquareInterval(
                    delta_col=delta_col,
                    lb=float(a_k),
                    ub=float(b_k),
                    curvature=curvature,
                )
            )

        if trig_square_intervals:
            piecewise_trig_square_relaxations.append(
                PiecewiseTrigSquareRelaxation(
                    square=square_relax,
                    func_name=base_relax.func_name,
                    arg_coeff=base_relax.arg_coeff,
                    arg_const=base_relax.arg_const,
                    arg_lb=base_relax.arg_lb,
                    arg_ub=base_relax.arg_ub,
                    intervals=trig_square_intervals,
                )
            )

    piecewise_univariate_relaxations: list[PiecewiseUnivariateRelaxation] = []
    for relax in univariate_relaxations:
        interval_specs = _trig_piecewise_interval_specs(relax, disc_state, n_orig)
        if not interval_specs:
            interval_specs = _inverse_trig_piecewise_interval_specs(relax)
        if not interval_specs:
            continue

        trig_intervals: list[PiecewiseUnivariateInterval] = []
        for a_k, b_k, curvature in interval_specs:
            delta_col = col_idx
            all_bounds.append((0.0, 1.0))
            integrality_flags.append(1)
            col_idx += 1
            trig_intervals.append(
                PiecewiseUnivariateInterval(
                    delta_col=delta_col,
                    lb=float(a_k),
                    ub=float(b_k),
                    curvature=curvature,
                )
            )

        if trig_intervals:
            piecewise_univariate_relaxations.append(
                PiecewiseUnivariateRelaxation(relax=relax, intervals=trig_intervals)
            )

    # ── Fractional-power aux columns: a = x^p with non-integer p ────────────
    # Only handle the cases where the relaxation is well-defined and
    # numerically stable: x ≥ 0 strictly bounded, and either 0 < p < 1
    # (concave) or p > 1 (convex).  Other cases are skipped and remain
    # general_nl, surfacing through the existing warning path.
    fractional_power_specs: list[dict] = []
    for var_idx, p in terms.fractional_power:
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        if lb_i < 0.0 or ub_i <= lb_i:
            continue
        if p == 0.0 or p == 1.0:
            continue
        # Convexity: p ∈ (0,1) → concave on x ≥ 0; p > 1 or p < 0 → convex on x > 0.
        if 0.0 < p < 1.0:
            convexity = "concave"
        elif p > 1.0 or p < 0.0:
            convexity = "convex"
            if p < 0.0 and lb_i <= 0.0:
                continue
        else:
            continue
        try:
            f_lb = lb_i**p
            f_ub = ub_i**p
        except (ValueError, OverflowError):
            continue
        if not (np.isfinite(f_lb) and np.isfinite(f_ub)):
            continue
        col = col_idx
        fractional_power_var_map[(var_idx, float(p))] = col
        all_bounds.append((min(f_lb, f_ub), max(f_lb, f_ub)))
        integrality_flags.append(0)
        col_idx += 1
        fractional_power_specs.append(
            {
                "var": var_idx,
                "p": float(p),
                "col": col,
                "lb": lb_i,
                "ub": ub_i,
                "f_lb": f_lb,
                "f_ub": f_ub,
                "convexity": convexity,
            }
        )

    # ── Bilinear-with-fractional-power: y * x^p  →  McCormick on (y_col, fp_col)
    bilinear_with_fp_keys: list[tuple[int, int]] = []
    for lin_idx, fp_key in terms.bilinear_with_fp:
        if fp_key not in fractional_power_var_map:
            continue
        fp_col = fractional_power_var_map[fp_key]
        pair = (min(lin_idx, fp_col), max(lin_idx, fp_col))
        bilinear_with_fp_keys.append(pair)

    for key in bilinear_with_fp_keys:
        bilinear_var_map[key] = _ensure_bilinear_aux(*key)

    lifted_bilinear_keys = _collect_lifted_bilinear_products(
        model,
        fractional_power_var_map,
        univariate_var_map,
        n_orig,
        monomial_var_map=monomial_var_map,
        composite_var_map=composite_var_map,
    )
    for key in lifted_bilinear_keys:
        bilinear_var_map[key] = _ensure_bilinear_aux(*key)

    # ── Lifted trilinear / multilinear: products such as ``x**2 * y * z`` whose
    # repeated factor collapses to a lifted aux column. The classifier never
    # records their distinct-column key, so allocate the recursive bilinear
    # chain here (after every lifted map — monomial/univariate/fractional-power/
    # composite — is populated so ``_decompose_product`` resolves each factor).
    lifted_trilinear_keys, lifted_multilinear_keys = _collect_lifted_higher_products(
        model,
        fractional_power_var_map,
        univariate_var_map,
        n_orig,
        monomial_var_map=monomial_var_map,
        composite_var_map=composite_var_map,
    )
    for term in lifted_trilinear_keys:
        if term in trilinear_var_map:
            continue
        pair, remaining = _choose_trilinear_pair(term, partitioned_vars)
        pair_col = _ensure_bilinear_aux(*pair)
        final_col = _ensure_bilinear_aux(pair_col, remaining)
        trilinear_var_map[term] = final_col
        trilinear_stage_map[term] = {
            "pair": pair,
            "pair_col": pair_col,
            "remaining_var": remaining,
            "product_col": final_col,
        }
    for multi_term in lifted_multilinear_keys:
        if multi_term in multilinear_var_map:
            continue
        final_col, stages = _ensure_multilinear_aux(multi_term)
        multilinear_var_map[multi_term] = final_col
        multilinear_stage_map[multi_term] = stages

    # ── Square-of-affine-in-lifted-vars envelopes (issue #155) ──────────────
    # For each recognized ``E**2`` residual, lift every product factor to a
    # column (monomial aux for ``x**n``, recursive bilinear chain for products),
    # build the affine residual ``r`` over those columns, and allocate ``s = r^2``
    # with an interval-arithmetic bound on ``r``. The tangent/secant envelope is
    # emitted with the other square rows; the linearizer resolves the protected
    # ``**2`` node to ``s`` through ``composite_var_map``.
    affine_square_relaxations: list[AffineSquareRelaxation] = []
    affine_square_protected_ids: set[int] = set()

    def _lift_monomial_column(monomial: _Monomial) -> Optional[int]:
        """Column for a product of original variables, or None if not liftable."""
        power_cols: list[int] = []
        for var_idx, power in sorted(Counter(monomial).items()):
            if power == 1:
                power_cols.append(var_idx)
            else:
                mono_col = monomial_var_map.get((var_idx, power))
                if mono_col is None:
                    # The monomial was dropped by the magnitude cap; the whole
                    # square cannot be lifted soundly, so fall back (skip it).
                    return None
                power_cols.append(mono_col)
        if not power_cols:
            return None
        col = power_cols[0]
        for nxt in power_cols[1:]:
            col = _ensure_bilinear_aux(col, nxt)
        return col

    for node, (sq_const, sq_terms) in affine_squares:
        resid: dict[int, float] = {}
        residual_const = float(sq_const)
        liftable = True
        for coeff, monomial in sq_terms:
            if not monomial:
                residual_const += coeff
                continue
            lifted_col = _lift_monomial_column(monomial)
            if lifted_col is None:
                liftable = False
                break
            resid[lifted_col] = resid.get(lifted_col, 0.0) + coeff
        if not liftable or not resid:
            continue

        r_lb = residual_const
        r_ub = residual_const
        finite = True
        for col, coeff in resid.items():
            clo, chi = (float(v) for v in all_bounds[col])
            if not (_is_effectively_finite(clo) and _is_effectively_finite(chi)):
                finite = False
                break
            lo_contrib = coeff * clo if coeff >= 0.0 else coeff * chi
            hi_contrib = coeff * chi if coeff >= 0.0 else coeff * clo
            r_lb += lo_contrib
            r_ub += hi_contrib
        if not finite or not (np.isfinite(r_lb) and np.isfinite(r_ub)) or r_ub < r_lb:
            continue

        s_lb = 0.0 if r_lb <= 0.0 <= r_ub else min(r_lb * r_lb, r_ub * r_ub)
        s_ub = max(r_lb * r_lb, r_ub * r_ub)
        # Cap an extreme square upper bound to avoid a numerically degenerate LP
        # (the residual range legitimately reaches ~1.6e9 on nvs16, so s_ub ~1e18).
        # An unbounded-above aux is sound: the lower bound comes from the tangent
        # underestimators and the column's own lower bound s_lb.
        if not np.isfinite(s_ub) or s_ub > _MONOMIAL_AUX_BOUND_LIMIT:
            s_ub = np.inf

        s_col = col_idx
        all_bounds.append((float(s_lb), float(s_ub)))
        integrality_flags.append(0)
        col_idx += 1
        affine_square_relaxations.append(
            AffineSquareRelaxation(
                aux_col=s_col,
                resid=resid,
                const=residual_const,
                r_lb=float(r_lb),
                r_ub=float(r_ub),
            )
        )
        composite_var_map[id(node)] = s_col
        affine_square_protected_ids.add(id(node))

    # ── Scaled single-variable power envelopes ``(c*x)**n`` (n >= 3) ─────────
    # A power of a *scaled* variable (e.g. ``(0.000338983*x6)**3`` with
    # x6 in [0, 2950]) is lifted on the well-conditioned residual ``r = c*x``
    # (here r in [0, 1]) instead of as the raw ``x**3`` monomial whose aux bound
    # (~2.6e10) trips the magnitude cap and, if forced through, yields a
    # numerically degenerate LP. The aux column ``w = r**n`` carries the
    # univariate power envelope below; ``id(node)`` resolves through
    # ``composite_var_map`` so the linearizer references ``w`` directly.
    affine_power_relaxations: list[AffinePowerRelaxation] = []
    for node, scale, var_idx, power in _collect_affine_powers(model, affine_square_protected_ids):
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        if not (np.isfinite(lb_i) and np.isfinite(ub_i)):
            continue
        r_lo = scale * lb_i
        r_hi = scale * ub_i
        r_lb = min(r_lo, r_hi)
        r_ub = max(r_lo, r_hi)
        if not (np.isfinite(r_lb) and np.isfinite(r_ub)) or r_ub - r_lb <= 1e-12:
            continue
        # ``w = r**power`` value range over [r_lb, r_ub] (monotone for odd power;
        # the spanning-zero minimum of an even power is 0).
        w_vals = [r_lb**power, r_ub**power]
        if r_lb < 0.0 < r_ub:
            w_vals.append(0.0)
        w_lb = min(w_vals)
        w_ub = max(w_vals)
        # Cap an extreme upper bound to keep the LP well-conditioned; the
        # tangent/secant rows below still bound ``w`` soundly (an unbounded-above
        # aux only enlarges the feasible region).
        if not np.isfinite(w_ub) or abs(w_ub) > _MONOMIAL_AUX_BOUND_LIMIT:
            w_ub = np.inf
        if not np.isfinite(w_lb) or abs(w_lb) > _MONOMIAL_AUX_BOUND_LIMIT:
            w_lb = -np.inf
        w_col = col_idx
        all_bounds.append((float(w_lb), float(w_ub)))
        integrality_flags.append(0)
        col_idx += 1
        affine_power_relaxations.append(
            AffinePowerRelaxation(
                aux_col=w_col,
                var_idx=var_idx,
                scale=float(scale),
                power=power,
                r_lb=float(r_lb),
                r_ub=float(r_ub),
            )
        )
        composite_var_map[id(node)] = w_col
        affine_square_protected_ids.add(id(node))

    # ── Lifted reciprocal / sqrt of a bounded inner expression (issue #154) ──
    # ``c / g`` and ``sqrt(g)`` are dropped by the affine-only univariate path
    # whenever ``g`` is itself nonlinear (a product or a sum of products), which
    # leaves the defining equality (e.g. ``x4 = 4243.28/(x0*x1)``) omitted and
    # the root bound loose. Here we *force-lift* ``g`` to an affine combination
    # over the extended variable space: every multiplicative factor — including
    # squares ``x*x`` (built as the McCormick bilinear ``x·x``) — becomes a
    # bounded product aux column via ``_ensure_bilinear_aux``, which auto-emits
    # its McCormick envelope through ``bilinear_relation_map``. When ``g`` resolves
    # and its interval is finite with the right sign — ``g > 0`` for ``1/g``
    # (convex), ``g >= 0`` for ``sqrt(g)`` (concave) — we lift an aux column for
    # the outer atom and let the standard convex/concave envelope (tangents +
    # secant) cut it.
    #
    # Soundness: at any true feasible point set each product aux to its exact
    # value (the McCormick envelope contains the true product), so ``g`` is exact
    # and the outer convex/concave envelope encloses the curve; the relaxation
    # feasible set is therefore a superset of the true one and the dual bound
    # stays valid. We deliberately abstain (fall back to warn-and-omit, which only
    # enlarges the feasible region) on anything not provably sound here: a
    # square-of-affine with cross terms ``(a·x+b·y)**2`` (deferred to the
    # square-of-affine envelope, #155), a non-constant numerator, an unbounded or
    # wrong-sign interval, or a product whose aux bound magnitude blows past the
    # monomial limit (ex1252's ~1e15 terms).
    def _extended_affine_interval(arg_coeff: np.ndarray, arg_const: float) -> tuple[float, float]:
        lo = float(arg_const)
        hi = float(arg_const)
        for j in np.flatnonzero(arg_coeff):
            c = float(arg_coeff[j])
            clb, cub = all_bounds[j]
            terms_j = (c * float(clb), c * float(cub))
            lo += min(terms_j)
            hi += max(terms_j)
        return lo, hi

    _MAX_FORCED_POWER = 4

    # Set by ``_lift_inner_to_affine`` whenever the cross-term-monomial fallback
    # (issue #154 increment 2) claims a factor; read by ``_maybe_lift_outer_atom``
    # to apply the cross-term conditioning guard. Reset before each outer lift.
    cross_term_used = [False]

    # Keeps synthetic ``expr_id`` sentinels for the intermediate reciprocal auxes
    # created by the nested-division lift (issue #154 increment 3) alive for the
    # lifetime of this build, so their ``id()`` never collides with a real node's.
    _nested_div_keepalive: list[object] = []

    def _ensure_bounded_bilinear(lhs_col: int, rhs_col: int) -> Optional[int]:
        """Force-create the product aux for two bounded columns (``x*y``, or
        ``x*x`` for a square), or ``None`` if a factor is unbounded or the product
        aux's bound magnitude exceeds the monomial limit (kept numerically tame)."""
        for c in (lhs_col, rhs_col):
            lo, hi = all_bounds[c]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return None
        a_lo, a_hi = (float(v) for v in all_bounds[lhs_col])
        b_lo, b_hi = (float(v) for v in all_bounds[rhs_col])
        corners = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]
        if max(abs(min(corners)), abs(max(corners))) > _MONOMIAL_AUX_BOUND_LIMIT:
            return None
        return _ensure_bilinear_aux(lhs_col, rhs_col)

    def _lift_factor_to_col(expr: Expression) -> Optional[int]:
        """Reduce one multiplicative factor to a single bounded column, creating
        product aux columns as needed. Claims only bounded original variables,
        their pairwise/iterated products, and positive-integer powers of a single
        bounded variable. Abstains (``None``) on affine-base powers/squares so the
        cross-term square-of-affine envelope (#155) stays the path for those."""
        flat = _get_flat_index(expr, model)
        if flat is not None:
            lo, hi = all_bounds[flat]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return None
            return flat
        if isinstance(expr, BinaryOp) and expr.op == "*":
            lcol = _lift_factor_to_col(expr.left)
            if lcol is None:
                return None
            rcol = _lift_factor_to_col(expr.right)
            if rcol is None:
                return None
            return _ensure_bounded_bilinear(lcol, rcol)
        if isinstance(expr, BinaryOp) and expr.op == "**" and isinstance(expr.right, Constant):
            p = float(expr.right.value)
            if p.is_integer() and 2 <= int(p) <= _MAX_FORCED_POWER:
                base = _lift_factor_to_col(expr.left)
                if base is None:
                    return None
                col = base
                for _ in range(int(p) - 1):
                    nxt = _ensure_bounded_bilinear(col, base)
                    if nxt is None:
                        return None
                    col = nxt
                return col
        return None

    def _lift_monomial_with_coeff(expr: Expression) -> Optional[tuple[list[int], float]]:
        """Decompose a multiplicative monomial into ``(variable-factor columns,
        scalar coefficient)``. Unlike :func:`_lift_factor_to_col`, embedded
        constant factors are folded into the coefficient rather than rejected, so
        a cross term such as ``2*x4*x5*x7`` (nvs05/nvs22 constraint C4) resolves to
        ``([x4, x5, x7], 2.0)``. Mirrors ``_lift_factor_to_col``'s claims (bounded
        original variables, products, integer powers) and likewise abstains on
        affine-base powers so the #155 square-of-affine envelope keeps that path.
        Returns ``None`` on any non-monomial / unbounded shape."""
        c = _constant_value(expr)
        if c is not None:
            return [], float(c)
        flat = _get_flat_index(expr, model)
        if flat is not None:
            lo, hi = all_bounds[flat]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return None
            return [flat], 1.0
        if isinstance(expr, UnaryOp) and expr.op == "-":
            sub = _lift_monomial_with_coeff(expr.operand)
            if sub is None:
                return None
            cols, coeff = sub
            return cols, -coeff
        if isinstance(expr, BinaryOp) and expr.op == "*":
            left = _lift_monomial_with_coeff(expr.left)
            if left is None:
                return None
            right = _lift_monomial_with_coeff(expr.right)
            if right is None:
                return None
            lcols, lc = left
            rcols, rc = right
            return lcols + rcols, lc * rc
        if isinstance(expr, BinaryOp) and expr.op == "**" and isinstance(expr.right, Constant):
            p = float(expr.right.value)
            if p.is_integer() and 2 <= int(p) <= _MAX_FORCED_POWER:
                base = _lift_monomial_with_coeff(expr.left)
                if base is None:
                    return None
                bcols, bc = base
                return bcols * int(p), bc ** int(p)
        return None

    def _fold_cols_to_aux(cols: list[int]) -> Optional[int]:
        """Fold a list of bounded columns into a single product aux via iterated
        bilinear lifting, or ``None`` if any factor is unbounded / the product aux
        bound exceeds the monomial limit. Empty list yields ``None`` (the caller
        treats a coefficient-only monomial as a constant)."""
        if not cols:
            return None
        col = cols[0]
        lo, hi = all_bounds[col]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        for nxt in cols[1:]:
            folded = _ensure_bounded_bilinear(col, nxt)
            if folded is None:
                return None
            col = folded
        return col

    def _lift_inner_to_affine(expr: Expression) -> Optional[tuple[dict[int, float], float]]:
        """Resolve ``g`` to ``(coeffs, const)`` over the extended column space,
        force-lifting product factors. ``None`` to abstain on any unclaimed shape."""
        c = _constant_value(expr)
        if c is not None:
            return {}, float(c)
        if isinstance(expr, BinaryOp) and expr.op in ("+", "-"):
            left = _lift_inner_to_affine(expr.left)
            if left is None:
                return None
            right = _lift_inner_to_affine(expr.right)
            if right is None:
                return None
            lc, lconst = left
            rc, rconst = right
            sign = 1.0 if expr.op == "+" else -1.0
            out = dict(lc)
            for k, v in rc.items():
                out[k] = out.get(k, 0.0) + sign * v
            return out, lconst + sign * rconst
        if isinstance(expr, BinaryOp) and expr.op == "*":
            lconst_val = _constant_value(expr.left)
            rconst_val = _constant_value(expr.right)
            scale = lconst_val if lconst_val is not None else rconst_val
            if scale is not None:
                sub = expr.right if lconst_val is not None else expr.left
                inner = _lift_inner_to_affine(sub)
                if inner is None:
                    return None
                sc, sconst = inner
                return {k: scale * v for k, v in sc.items()}, scale * sconst
        if isinstance(expr, UnaryOp) and expr.op == "-":
            inner = _lift_inner_to_affine(expr.operand)
            if inner is None:
                return None
            sc, sconst = inner
            return {k: -v for k, v in sc.items()}, -sconst
        col = _lift_factor_to_col(expr)
        if col is not None:
            return {col: 1.0}, 0.0
        # Increment #2 (issue #154): a monomial carrying an embedded constant
        # factor — e.g. the cross term ``2*x4*x5*x7`` in nvs05/nvs22 constraint
        # C4 — abstains above because ``_lift_factor_to_col`` rejects the scalar
        # ``2``. Strip the coefficient and fold the variable factors into one
        # product aux so ``sqrt`` of a cross-term quadratic lifts soundly.
        mono = _lift_monomial_with_coeff(expr)
        if mono is None:
            return None
        cols, coeff = mono
        folded = _fold_cols_to_aux(cols)
        if folded is None:
            return None
        cross_term_used[0] = True
        return {folded: coeff}, 0.0

    def _try_lift_ratio_of_products(div_expr: BinaryOp) -> None:
        """Linear-fractional lift of ``coeff·(Π num_i) / (Π den_j)`` (issue #185).

        Without this, a ratio of products (e.g. gear4's ``-1e6·i1·i2/(i3·i4)``)
        is dumped into ``general_nl`` and the whole enclosing constraint is dropped
        from the relaxation, leaving a trivial dual bound. We relax it via the
        bilinear identity ``r·q = m`` where ``m`` is the lifted numerator product,
        ``q`` the lifted denominator product, and ``r`` a fresh column for the
        *pure* ratio ``m/q``; the division node is mapped to ``r`` with the scalar
        ``coeff`` applied by the linearizer (keeping the large numerator constant
        out of the envelope coefficients, so the relaxation LP stays
        well-conditioned).

        Soundness: ``r``'s interval is the sign-aware interval product
        ``[m] · (1/[q])`` which contains the true ``m/q`` at every feasible point
        (``1/q`` is monotone on a sign-definite ``q``), so the McCormick envelope
        of ``r·q`` set equal to ``m`` is a valid outer approximation. We *decline*
        (leaving the constraint to drop, which only enlarges the relaxation — always
        sound) when the denominator is not strictly sign-definite / bounded away
        from zero on the node box, when any factor is unbounded, or when the
        envelope magnitudes exceed the conditioning limit (an ill-conditioned LP
        could otherwise yield an unsound bound).
        """
        nonlocal col_idx
        eid = id(div_expr)
        if eid in composite_var_map or eid in univariate_var_map:
            return

        num = _decompose_signed_monomial(div_expr.left, model)
        if num is None:
            return
        coeff, num_idx = num
        if not (np.isfinite(coeff) and coeff != 0.0) or not num_idx:
            return

        den = _decompose_signed_monomial(div_expr.right, model)
        if den is None:
            return
        den_coeff, den_idx = den
        if not (np.isfinite(den_coeff) and den_coeff != 0.0) or not den_idx:
            return
        coeff = coeff / den_coeff
        if not np.isfinite(coeff):
            return

        # Fold each side's variable factors into a single bounded product-aux
        # column (each carries a standard McCormick envelope tying it to the
        # underlying variables); a single-variable factor maps to its own column.
        m_col = _fold_cols_to_aux(num_idx)
        if m_col is None:
            return
        q_col = _fold_cols_to_aux(den_idx)
        if q_col is None or q_col == m_col:
            return

        q_lo, q_hi = (float(v) for v in all_bounds[q_col])
        if not (np.isfinite(q_lo) and np.isfinite(q_hi)):
            return
        # Must-gate: denominator strictly sign-definite and bounded away from 0,
        # else 1/q's interval is invalid (→ unsound r bound → false certification).
        if not (q_lo > _RECIP_MIN_DENOM or q_hi < -_RECIP_MIN_DENOM):
            return
        m_lo, m_hi = (float(v) for v in all_bounds[m_col])
        if not (np.isfinite(m_lo) and np.isfinite(m_hi)):
            return

        # 1/q is monotone on a sign-definite interval, so its range is exactly
        # [1/q_hi, 1/q_lo]; the pure ratio r = m/q lies in the interval product.
        inv_lo, inv_hi = 1.0 / q_hi, 1.0 / q_lo
        corners = [m_lo * inv_lo, m_lo * inv_hi, m_hi * inv_lo, m_hi * inv_hi]
        r_lo, r_hi = float(min(corners)), float(max(corners))
        if not (np.isfinite(r_lo) and np.isfinite(r_hi)):
            return

        # Conditioning guard: keep every envelope coefficient / RHS below the
        # cross-term magnitude limit so the fast simplex backend stays reliable.
        env_mags = [
            abs(r_lo),
            abs(r_hi),
            abs(q_lo),
            abs(q_hi),
            abs(r_lo * q_lo),
            abs(r_hi * q_hi),
            abs(r_hi * q_lo),
            abs(r_lo * q_hi),
        ]
        if max(env_mags) > _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE:
            return

        r_col = col_idx
        all_bounds.append((r_lo, r_hi))
        integrality_flags.append(0)
        col_idx += 1

        ratio_relaxations.append(
            RatioRelaxation(r_col=r_col, q_col=q_col, m_col=m_col, r_lb=r_lo, r_ub=r_hi)
        )
        composite_var_map[eid] = r_col
        composite_coeff_map[eid] = float(coeff)

    def _try_lift_nested_division(div_expr: BinaryOp) -> None:
        """Lift ``num / g`` with a *non-constant* numerator (issue #154 increment
        3). The constant-numerator path scales an existing ``1/g`` aux by the
        numerator (``_maybe_lift_outer_atom`` + the linearizer at the ``c/g`` site);
        a non-constant numerator instead hits ``Cannot linearize non-constant
        division`` and the constraint is dropped. The classic example is nvs05/
        nvs22 ``(0.5*x3)/x6``.

        We relax it as the factorable product ``num/g = n · (ncoeff/g)`` where the
        numerator decomposes into ``ncoeff · n`` (a single bounded column ``n`` and
        a positive scalar ``ncoeff``):

        1. force-lift the reciprocal ``r = ncoeff/g`` with the standard convex
           reciprocal envelope, folding ``ncoeff`` into the affine argument
           (``arg = g/ncoeff``) so ``r``'s value is exactly ``ncoeff/g`` and the
           product below carries coefficient 1;
        2. force the McCormick product ``P = n · r``;
        3. map the division node to ``P`` through ``composite_var_map`` so the
           linearizer substitutes it with coefficient 1.

        Soundness: at any true feasible point ``r = ncoeff/g`` lies in
        ``[ncoeff/gh, ncoeff/gl]`` and the reciprocal envelope contains it, and the
        McCormick envelope of ``n · r`` contains ``n · (ncoeff/g) = num/g`` — so the
        relaxed feasible set is a superset of the true one and the dual bound stays
        valid. We abstain (drop the cut, which only enlarges the relaxation —
        always sound) on every shape not provably handled here: a non-monomial or
        non-positive-coefficient numerator (a sign fold would flip the reciprocal
        interval negative), a denominator that is not strictly positive / is
        ill-conditioned for the reciprocal envelope, or a product aux whose bound
        magnitude is large enough to make the fast simplex backend mis-solve the
        LP (the #158/increment-2 wrong-"optimal" hazard)."""
        nonlocal col_idx
        eid = id(div_expr)
        if eid in composite_var_map or eid in univariate_var_map:
            return

        # Numerator → (single bounded column n, positive scalar ncoeff).
        mono = _lift_monomial_with_coeff(div_expr.left)
        if mono is None:
            return
        ncols, ncoeff = mono
        if not (np.isfinite(ncoeff) and ncoeff > 0.0):
            return
        n_col = _fold_cols_to_aux(ncols)
        if n_col is None:
            return

        # Denominator g → affine over the extended column space.
        g_lifted = _lift_inner_to_affine(div_expr.right)
        if g_lifted is None:
            return
        g_coeffs, g_const = g_lifted
        if not g_coeffs:
            return  # constant denominator is not a genuine division

        # Reciprocal argument t = g / ncoeff, so reciprocal(t) = ncoeff / g.
        arg_coeff = np.zeros(col_idx)
        for col, val in g_coeffs.items():
            arg_coeff[col] = val / ncoeff
        arg_const = g_const / ncoeff

        gl, gh = _extended_affine_interval(arg_coeff, arg_const)
        if not (np.isfinite(gl) and np.isfinite(gh)) or gh < gl:
            return
        # Reciprocal needs a strictly-positive, well-conditioned denominator: the
        # tangent slope ``-1/gl**2`` is the largest coefficient the envelope emits.
        if gl <= _RECIP_MIN_DENOM:
            return
        if 1.0 / (gl * gl) > _LIFT_MAX_ENVELOPE_SLOPE:
            return
        val_lb, val_ub = 1.0 / gh, 1.0 / gl
        if not (np.isfinite(val_lb) and np.isfinite(val_ub)):
            return

        # Conditioning guard (issue #154 increment 2/3): a product aux whose bounds
        # reach the magnitude where the lifted LP is ill-conditioned lets the fast
        # simplex return a wrong "optimal" — an unsound dual bound. Pre-check the
        # n·r corner magnitudes and abstain above the limit (sound: drops the cut).
        n_lo, n_hi = (float(v) for v in all_bounds[n_col])
        if not (np.isfinite(n_lo) and np.isfinite(n_hi)):
            return
        corners = [n_lo * val_lb, n_lo * val_ub, n_hi * val_lb, n_hi * val_ub]
        if max(abs(min(corners)), abs(max(corners))) > _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE:
            return

        # 1. reciprocal aux r = ncoeff / g, with the convex reciprocal envelope.
        sentinel = object()
        _nested_div_keepalive.append(sentinel)
        recip_col = col_idx
        univariate_relaxations.append(
            UnivariateRelaxation(
                expr_id=id(sentinel),
                func_name="reciprocal",
                aux_col=recip_col,
                arg_coeff=arg_coeff,
                arg_const=float(arg_const),
                arg_lb=float(gl),
                arg_ub=float(gh),
            )
        )
        all_bounds.append((float(val_lb), float(val_ub)))
        integrality_flags.append(0)
        col_idx += 1

        # 2. McCormick product P = n · r.
        prod_col = _ensure_bounded_bilinear(n_col, recip_col)
        if prod_col is None:
            return  # the orphan reciprocal aux is bounded + enveloped, just unused

        # 3. substitute the division node with P (coefficient 1).
        composite_var_map[eid] = prod_col

    def _maybe_lift_outer_atom(expr: Expression) -> None:
        nonlocal col_idx
        eid = id(expr)
        if eid in univariate_var_map or eid in composite_var_map:
            return
        if isinstance(expr, BinaryOp) and expr.op == "/":
            num = _constant_value(expr.left)
            if num is None:
                # Non-constant numerator. Try the ratio-of-products lift first
                # (handles signed numerators and product denominators via the
                # r·q = m identity, issue #185); fall back to the reciprocal-based
                # nested-division lift (issue #154 increment 3). The fallback is a
                # no-op once the ratio lift has claimed the node.
                _try_lift_ratio_of_products(expr)
                _try_lift_nested_division(expr)
                return
            func_name, inner = "reciprocal", expr.right
        elif isinstance(expr, FunctionCall) and expr.func_name == "sqrt" and len(expr.args) == 1:
            func_name, inner = "sqrt", expr.args[0]
        else:
            return

        cross_term_used[0] = False
        lifted = _lift_inner_to_affine(inner)
        if lifted is None:
            return
        coeffs, arg_const = lifted

        # A purely-affine inner over original variables is already handled by the
        # standard univariate path; only claim genuinely lifted (aux-referencing)
        # arguments so we never shadow or double-count that path.
        if not any(col >= n_orig for col in coeffs):
            return

        arg_coeff = np.zeros(col_idx)
        for col, val in coeffs.items():
            arg_coeff[col] = val

        gl, gh = _extended_affine_interval(arg_coeff, arg_const)
        if not (np.isfinite(gl) and np.isfinite(gh)) or gh < gl:
            return
        # Cross-term conditioning guard (issue #154 increment 2): a cross-term
        # lift whose argument magnitude is large enough to make the LP
        # ill-conditioned can produce a wrong "optimal" — and therefore an
        # *unsound* dual bound — from the fast simplex backend. Abstain (drop the
        # lift, enlarging the relaxation, which is always sound) above the limit.
        if cross_term_used[0] and max(abs(gl), abs(gh)) > _LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE:
            return
        if func_name == "reciprocal":
            if gl <= _RECIP_MIN_DENOM:
                return
            # Tangent slope ``-1/gl**2`` is the largest coefficient the envelope
            # injects; refuse a numerically degenerate cut.
            if 1.0 / (gl * gl) > _LIFT_MAX_ENVELOPE_SLOPE:
                return
            val_lb, val_ub = 1.0 / gh, 1.0 / gl
        else:  # sqrt
            if gl < -_SQRT_NEG_TOL:
                return
            gl = max(gl, 0.0)
            # Concave-tangent overestimator slope ``1/(2*sqrt(t))`` blows up as
            # ``t→0``. The emission places tangents at ``[lb, mid, ub]`` but skips
            # any ``pt <= 0`` (``_tangent_points``), so when ``gl == 0`` the worst
            # emitted tangent sits at the midpoint ``gh/2`` (well-conditioned);
            # otherwise it sits at ``gl``. Guard that worst emitted slope.
            slope_pt = gl if gl > 0.0 else 0.5 * gh
            if slope_pt <= 0.0 or 1.0 / (2.0 * np.sqrt(slope_pt)) > _LIFT_MAX_ENVELOPE_SLOPE:
                return
            val_lb, val_ub = float(np.sqrt(gl)), float(np.sqrt(gh))
        if not (np.isfinite(val_lb) and np.isfinite(val_ub)):
            return

        aux_col = col_idx
        univariate_var_map[eid] = aux_col
        univariate_relaxations.append(
            UnivariateRelaxation(
                expr_id=eid,
                func_name=func_name,
                aux_col=aux_col,
                arg_coeff=arg_coeff,
                arg_const=float(arg_const),
                arg_lb=float(gl),
                arg_ub=float(gh),
            )
        )
        all_bounds.append((float(val_lb), float(val_ub)))
        integrality_flags.append(0)
        col_idx += 1

    def _walk_lift(expr: Expression) -> None:
        _maybe_lift_outer_atom(expr)
        if isinstance(expr, BinaryOp):
            _walk_lift(expr.left)
            _walk_lift(expr.right)
        elif isinstance(expr, UnaryOp):
            _walk_lift(expr.operand)
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                _walk_lift(arg)
        elif isinstance(expr, IndexExpression):
            if not isinstance(expr.base, Variable):
                _walk_lift(expr.base)
        elif isinstance(expr, SumExpression):
            _walk_lift(expr.operand)
        elif isinstance(expr, SumOverExpression):
            for term in expr.terms:
                _walk_lift(term)

    # Distribute products ONCE and reuse the *same* trees for both this lift walk
    # and the constraint/objective linearization later. ``distribute_products``
    # rebuilds the operator tree (and expands ``(a+b)**2`` into its monomial/
    # bilinear terms), so a second call would yield different node identities and
    # the lifted ``univariate_var_map[id(div_or_sqrt_node)]`` keys would not match
    # the nodes the linearizer visits. Sharing the trees keeps the keys aligned —
    # and the square expansion lets ``sqrt`` of a cross-term quadratic lift too,
    # each term enveloped by its own (sound) McCormick product aux.
    # Protect the #155 affine-square nodes so ``distribute_products`` leaves their
    # ``**2`` intact (the affine-square envelope resolves them through
    # ``composite_var_map`` by original node id); every other ``(a+b)**2`` still
    # expands so the #154 sqrt/reciprocal lift can envelope each product term.
    protected_squares = (
        frozenset(affine_square_protected_ids) if affine_square_protected_ids else None
    )
    distributed_objective = (
        distribute_products(model._objective.expression, protected_squares)
        if model._objective is not None
        else None
    )
    distributed_bodies = {
        id(c): distribute_products(c.body, protected_squares) for c in model._constraints
    }
    if distributed_objective is not None:
        _walk_lift(distributed_objective)
    for _constraint in model._constraints:
        _walk_lift(distributed_bodies[id(_constraint)])

    # ── Multilinear RLT convex-hull cuts (setup) ───────────────────────────
    # The recursive chain w = (((x0*x1)*x2)*...) gives a valid but loose
    # relaxation of a multilinear product. The exact convex/concave hull is the
    # Reformulation-Linearization (RLT) system: for every subset S of the
    # factors with |S| >= 2, the |S|-degree bound-factor product inequalities
    #     prod_{i in S} (s_i*x_i + c_i) >= 0,  (x-xL) -> (+1,-xL), (xU-x) -> (-1,xU)
    # linearized via a lifted column w_T for every sub-product T (|T| >= 2).
    # Each is a product of nonnegative bound factors, hence individually valid
    # (tightens, never invalidates); the full set is the exact hull (Rikun
    # 1997 / Meyer & Floudas 2004), verified to match the box-vertex envelope.
    # |S|=2 cuts are exactly the McCormick envelopes already emitted by
    # _ensure_bilinear_aux, so here we (a) materialize every subset-product
    # column so the higher cuts can reference it, and (b) record the term for
    # the |S|>=3 cut rows emitted below. Capped at DISCOPT_MULTILINEAR_RLT_MAX
    # factors (default 4) to bound the 2^n column/row growth; larger products
    # keep the loose recursive chain.
    rlt_terms: list[tuple[tuple[int, ...], dict[frozenset, int]]] = []
    if os.environ.get("DISCOPT_TRILINEAR_RLT", "1") != "0":
        _rlt_cap = int(os.environ.get("DISCOPT_MULTILINEAR_RLT_MAX", "4"))
        _candidate_terms = set(trilinear_stage_map.keys()) | set(multilinear_stage_map.keys())
        for _term in _candidate_terms:
            _vars = tuple(sorted(set(_term)))
            _n = len(_vars)
            # Distinct factors only; repeated-factor products (x*x*y) collapse
            # onto a monomial aux and are not handled here.
            if _n < 3 or _n > _rlt_cap or _n != len(_term):
                continue
            subset_cols: dict[frozenset, int] = {frozenset([c]): c for c in _vars}
            for _k in range(2, _n + 1):
                for _comb in itertools.combinations(_vars, _k):
                    _m = max(_comb)
                    _rest = frozenset(c for c in _comb if c != _m)
                    subset_cols[frozenset(_comb)] = _ensure_bilinear_aux(subset_cols[_rest], _m)
            rlt_terms.append((_vars, subset_cols))

    bilinear_pw_map: dict[tuple[int, int], list] = {}
    bilinear_lambda_map: dict[tuple[int, int], dict] = {}

    for (lhs_col, rhs_col), _w_col in bilinear_relation_map.items():
        part_var: Optional[int] = None
        if lhs_col < n_orig and lhs_col in disc_state.partitions:
            part_var = lhs_col
            other_var = rhs_col
        elif rhs_col < n_orig and rhs_col in disc_state.partitions:
            part_var = rhs_col
            other_var = lhs_col
        else:
            continue

        pts = _guarded_partition_points(
            "bilinear piecewise",
            (lhs_col, rhs_col),
            disc_state.partitions[part_var],
        )
        if pts is None:
            continue
        other_lb, other_ub = all_bounds[other_var]

        if convhull_mode == "disaggregated":
            intervals = []
            for k in range(len(pts) - 1):
                a_k = float(pts[k])
                b_k = float(pts[k + 1])
                _, wk_lo, wk_hi = _piecewise_product_bounds(
                    a_k,
                    b_k,
                    float(other_lb),
                    float(other_ub),
                )

                delta_col = col_idx
                all_bounds.append((0.0, 1.0))
                integrality_flags.append(1)
                col_idx += 1

                xbar_col = col_idx
                all_bounds.append((min(a_k, 0.0), max(abs(a_k), abs(b_k))))
                integrality_flags.append(0)
                col_idx += 1

                wbar_col = col_idx
                all_bounds.append((min(wk_lo, 0.0), max(wk_hi, 0.0)))
                integrality_flags.append(0)
                col_idx += 1

                intervals.append((delta_col, xbar_col, wbar_col, a_k, b_k))

            bilinear_pw_map[(lhs_col, rhs_col)] = intervals
        else:
            breakpoints = [float(p) for p in pts]
            lambda_cols: list[int] = []
            alpha_cols: list[int] = []
            theta_cols: list[int] = []
            embedding_cols: list[int] = []
            embedding_info: Optional[EmbeddingMap] = None
            theta_lb = min(0.0, float(other_lb), float(other_ub))
            theta_ub = max(0.0, float(other_lb), float(other_ub))

            for _ in breakpoints:
                lambda_cols.append(col_idx)
                all_bounds.append((0.0, 1.0))
                integrality_flags.append(0)
                col_idx += 1

            if convhull_mode == "sos2" and convhull_ebd and len(breakpoints) > 2:
                embedding_info = build_embedding_map(
                    len(breakpoints),
                    encoding=convhull_ebd_encoding,
                )
                for _ in range(embedding_info.bit_count):
                    embedding_cols.append(col_idx)
                    all_bounds.append((0.0, 1.0))
                    integrality_flags.append(1)
                    col_idx += 1
            else:
                for _ in range(len(breakpoints) - 1):
                    alpha_cols.append(col_idx)
                    all_bounds.append((0.0, 1.0))
                    integrality_flags.append(1)
                    col_idx += 1

            for _ in breakpoints:
                theta_cols.append(col_idx)
                all_bounds.append((theta_lb, theta_ub))
                integrality_flags.append(0)
                col_idx += 1

            bilinear_lambda_map[(lhs_col, rhs_col)] = {
                "part_var": part_var,
                "other_var": other_var,
                "breakpoints": breakpoints,
                "lambda_cols": lambda_cols,
                "alpha_cols": alpha_cols,
                "theta_cols": theta_cols,
                "embedding_cols": embedding_cols,
                "embedding_info": embedding_info,
                "mode": convhull_mode,
            }

    # ── Piecewise aux columns for shared disaggregated structure ───────────
    # When a variable has partition breakpoints AND is the base of a concave
    # fractional power ``z = x^p`` with p in (0, 1), replace the global secant
    # with a piecewise version.  Monomial secants use their existing AMP
    # interval-selector formulation below.
    pw_candidate_vars: set[int] = set()
    for spec_pre in terms.fractional_power:
        var_idx, p = spec_pre
        if 0.0 < float(p) < 1.0 and var_idx in disc_state.partitions:
            pw_candidate_vars.add(var_idx)

    piecewise_var_map: dict[int, list[tuple[int, int, float, float]]] = {}
    for var_idx in sorted(pw_candidate_vars):
        pw_pts = _guarded_partition_points(
            "fractional-power piecewise",
            var_idx,
            disc_state.partitions[var_idx],
        )
        if pw_pts is None:
            continue
        if len(pw_pts) < 3:
            # With only 2 breakpoints there's just one interval; the global
            # secant already coincides with the piecewise secant.
            continue
        pw_intervals_list: list[tuple[int, int, float, float]] = []
        for k in range(len(pw_pts) - 1):
            p_lo = float(pw_pts[k])
            p_hi = float(pw_pts[k + 1])
            delta_col = col_idx
            all_bounds.append((0.0, 1.0))
            integrality_flags.append(1)
            col_idx += 1
            xbar_col = col_idx
            xbar_lb = min(p_lo, 0.0)
            xbar_ub = max(p_hi, 0.0)
            all_bounds.append((xbar_lb, xbar_ub))
            integrality_flags.append(0)
            col_idx += 1
            pw_intervals_list.append((delta_col, xbar_col, p_lo, p_hi))
        piecewise_var_map[var_idx] = pw_intervals_list

    if objective_lift is not None:
        objective_lift.aux_col = col_idx
        all_bounds.append(objective_lift.aux_bounds)
        integrality_flags.append(0)
        col_idx += 1

    n_total = col_idx

    # ── Constraint rows (A_ub @ z ≤ b_ub) ───────────────────────────────────
    A_data: list[float] = []
    A_row_indices: list[int] = []
    A_col_indices: list[int] = []
    b_rows: list[float] = []

    def _add_row(coeff: np.ndarray, rhs: float) -> None:
        coeff_arr = np.asarray(coeff, dtype=np.float64).ravel()
        row_idx = len(b_rows)
        nz = np.flatnonzero(coeff_arr)
        if nz.size:
            A_row_indices.extend([row_idx] * int(nz.size))
            A_col_indices.extend(nz.tolist())
            A_data.extend(coeff_arr[nz].tolist())
        b_rows.append(float(rhs))

    # ── Piecewise structural constraints (once per partitioned variable) ────
    # For each var_idx with a piecewise structure we enforce:
    #   sum δ_k = 1, x = Σ x̄_k, p_k δ_k ≤ x̄_k ≤ p_{k+1} δ_k.
    # Both monomial-secant and concave-fp-secant rows reference these aux
    # columns, so emitting structural rows once avoids duplicate constraints.
    for var_idx, pw_intervals in piecewise_var_map.items():
        # 1) Σ δ_k = 1 (encoded as ≤ 1 and ≥ 1)
        row = np.zeros(n_total)
        for delta_col, _xbar_col, _plo, _phi in pw_intervals:
            row[delta_col] = 1.0
        _add_row(row, 1.0)
        _add_row(-row, -1.0)
        # 2) x = Σ x̄_k  →  x − Σ x̄_k = 0
        row = np.zeros(n_total)
        row[var_idx] = 1.0
        for _delta_col, xbar_col, _plo, _phi in pw_intervals:
            row[xbar_col] = -1.0
        _add_row(row, 0.0)
        _add_row(-row, 0.0)
        # 3) Per-interval bounds: p_k δ_k ≤ x̄_k ≤ p_{k+1} δ_k
        for delta_col, xbar_col, p_lo, p_hi in pw_intervals:
            row = np.zeros(n_total)
            row[xbar_col] = 1.0
            row[delta_col] = -p_hi
            _add_row(row, 0.0)
            row = np.zeros(n_total)
            row[xbar_col] = -1.0
            row[delta_col] = p_lo
            _add_row(row, 0.0)

    # McCormick constraints for each lifted bilinear relation
    for (i, j), w_col in bilinear_relation_map.items():
        xi_lb_g, xi_ub_g = [float(v) for v in all_bounds[i]]
        xj_lb_g, xj_ub_g = [float(v) for v in all_bounds[j]]

        # A McCormick envelope needs finite bounds on both factors: the four
        # inequalities use ``x_lb``/``x_ub`` as coefficients, so an unbounded
        # factor injects ``±inf`` into the constraint matrix and the LP solver
        # errors out (e.g. nvs22's free auxiliary variables x4-x7, which only
        # appear in omitted division/sqrt constraints). Skip the envelope: the
        # aux ``w`` stays unconstrained, which only enlarges the feasible region
        # and therefore keeps the dual bound valid. This mirrors the finite-bound
        # guard already applied to monomial envelopes below.
        if not (
            _is_effectively_finite(xi_lb_g)
            and _is_effectively_finite(xi_ub_g)
            and _is_effectively_finite(xj_lb_g)
            and _is_effectively_finite(xj_ub_g)
        ):
            continue

        if (i, j) in bilinear_lambda_map:
            lambda_info = bilinear_lambda_map[(i, j)]
            part_var = int(lambda_info["part_var"])
            other_var = int(lambda_info["other_var"])
            breakpoints = list(lambda_info["breakpoints"])
            lambda_cols = list(lambda_info["lambda_cols"])
            alpha_cols = list(lambda_info["alpha_cols"])
            theta_cols = list(lambda_info["theta_cols"])
            embedding_cols = list(lambda_info.get("embedding_cols", []))
            embedding_info = lambda_info.get("embedding_info")
            mode = str(lambda_info["mode"])
            yj_lb, yj_ub = [float(v) for v in all_bounds[other_var]]

            row_sum_lambda = np.zeros(n_total)
            for lambda_col in lambda_cols:
                row_sum_lambda[lambda_col] = -1.0
            _add_row(row_sum_lambda, -1.0)
            _add_row(-row_sum_lambda, 1.0)

            if alpha_cols:
                row_sum_alpha = np.zeros(n_total)
                for alpha_col in alpha_cols:
                    row_sum_alpha[alpha_col] = -1.0
                _add_row(row_sum_alpha, -1.0)
                _add_row(-row_sum_alpha, 1.0)

            row_x = np.zeros(n_total)
            row_x[part_var] = 1.0
            for p_j, lambda_col in zip(breakpoints, lambda_cols):
                row_x[lambda_col] -= float(p_j)
            _add_row(row_x, 0.0)
            _add_row(-row_x, 0.0)

            row_y = np.zeros(n_total)
            row_y[other_var] = 1.0
            for theta_col in theta_cols:
                row_y[theta_col] -= 1.0
            _add_row(row_y, 0.0)
            _add_row(-row_y, 0.0)

            row_w = np.zeros(n_total)
            row_w[w_col] = 1.0
            for p_j, theta_col in zip(breakpoints, theta_cols):
                row_w[theta_col] -= float(p_j)
            _add_row(row_w, 0.0)
            _add_row(-row_w, 0.0)

            if mode == "sos2":
                assert alpha_cols or embedding_cols, (
                    "Expected either alpha or embedding columns for SOS2 linking"
                )

            if mode == "sos2" and embedding_info is not None:
                for bit_col, positive_set, negative_set in zip(
                    embedding_cols,
                    embedding_info.positive_sets,
                    embedding_info.negative_sets,
                ):
                    row = np.zeros(n_total)
                    for lambda_idx in positive_set:
                        row[lambda_cols[lambda_idx]] = 1.0
                    row[bit_col] = -1.0
                    _add_row(row, 0.0)

                    row = np.zeros(n_total)
                    for lambda_idx in negative_set:
                        row[lambda_cols[lambda_idx]] = 1.0
                    row[bit_col] = 1.0
                    _add_row(row, 1.0)
            elif mode == "sos2":
                for idx, lambda_col in enumerate(lambda_cols):
                    row = np.zeros(n_total)
                    row[lambda_col] = 1.0
                    if idx == 0:
                        row[alpha_cols[0]] = -1.0
                    elif idx == len(lambda_cols) - 1:
                        row[alpha_cols[-1]] = -1.0
                    else:
                        row[alpha_cols[idx - 1]] = -1.0
                        row[alpha_cols[idx]] = -1.0
                    _add_row(row, 0.0)
            else:
                for idx in range(len(alpha_cols) - 1):
                    row = np.zeros(n_total)
                    for alpha_col in alpha_cols[: idx + 1]:
                        row[alpha_col] -= 1.0
                    for lambda_col in lambda_cols[: idx + 1]:
                        row[lambda_col] += 1.0
                    _add_row(row, 0.0)

                    row = np.zeros(n_total)
                    for alpha_col in alpha_cols[: idx + 1]:
                        row[alpha_col] += 1.0
                    for lambda_col in lambda_cols[: idx + 2]:
                        row[lambda_col] -= 1.0
                    _add_row(row, 0.0)

            for lambda_col, theta_col in zip(lambda_cols, theta_cols):
                row = np.zeros(n_total)
                row[theta_col] = -1.0
                row[lambda_col] = yj_lb
                _add_row(row, 0.0)

                row = np.zeros(n_total)
                row[theta_col] = -1.0
                row[other_var] = 1.0
                row[lambda_col] = yj_ub
                _add_row(row, yj_ub)

                row = np.zeros(n_total)
                row[theta_col] = 1.0
                row[other_var] = -1.0
                row[lambda_col] = -yj_lb
                _add_row(row, -yj_lb)

                row = np.zeros(n_total)
                row[theta_col] = 1.0
                row[lambda_col] = -yj_ub
                _add_row(row, 0.0)

        elif (i, j) in bilinear_pw_map and bilinear_pw_map[(i, j)]:
            # ── Piecewise McCormick with binary partition selection ──────────
            intervals = bilinear_pw_map[(i, j)]
            if i < n_orig and i in disc_state.partitions:
                part_var, other_var = i, j
            else:
                part_var, other_var = j, i

            yj_lb, yj_ub = [float(v) for v in all_bounds[other_var]]

            # Constraint: Σ δ_k = 1 (select exactly one partition)
            row_sum = np.zeros(n_total)
            for delta_col, _, _, _, _ in intervals:
                row_sum[delta_col] = -1.0
            _add_row(row_sum, -1.0)  # -Σδ_k ≤ -1
            _add_row(-row_sum, 1.0)  # Σδ_k ≤ 1

            # Constraint: x_part = Σ x̄_k (reconstruct partition variable)
            row_recon = np.zeros(n_total)
            row_recon[part_var] = 1.0
            for _, xbar_col, _, _, _ in intervals:
                row_recon[xbar_col] = -1.0
            _add_row(row_recon, 0.0)  # x_part - Σ x̄_k ≤ 0
            _add_row(-row_recon, 0.0)  # -(x_part - Σ x̄_k) ≤ 0

            # Constraint: w = Σ w̄_k
            row_wsum = np.zeros(n_total)
            row_wsum[w_col] = 1.0
            for _, _, wbar_col, _, _ in intervals:
                row_wsum[wbar_col] = -1.0
            _add_row(row_wsum, 0.0)
            _add_row(-row_wsum, 0.0)

            for delta_col, xbar_col, wbar_col, a_k, b_k in intervals:
                corners, wk_lo, wk_hi = _piecewise_product_bounds(
                    a_k,
                    b_k,
                    yj_lb,
                    yj_ub,
                )

                def _inactive_big_m(other_coeff: float, rhs: float) -> float:
                    violations = [
                        float(other_coeff) * yj_lb - float(rhs),
                        float(other_coeff) * yj_ub - float(rhs),
                    ]
                    max_violation = max(0.0, *violations)
                    if max_violation <= 0.0:
                        return 0.0
                    return max_violation * (1.0 + 1e-4) + max(
                        1e-9,
                        1e-9 * max_violation,
                    )

                # x̄_k ≥ a_k * δ_k  (x̄_k is in [a_k, b_k] when δ_k=1)
                row = np.zeros(n_total)
                row[xbar_col] = -1.0
                row[delta_col] = a_k
                _add_row(row, 0.0)  # -x̄_k + a_k*δ_k ≤ 0  → x̄_k ≥ a_k*δ_k

                # x̄_k ≤ b_k * δ_k
                row = np.zeros(n_total)
                row[xbar_col] = 1.0
                row[delta_col] = -b_k
                _add_row(row, 0.0)

                # w̄_k ≤ wk_hi * δ_k  → w̄_k=0 when δ_k=0
                # This forces the bilinear product to 0 when interval k is inactive.
                row = np.zeros(n_total)
                row[wbar_col] = 1.0
                row[delta_col] = -wk_hi
                _add_row(row, 0.0)

                # w̄_k ≥ wk_lo * δ_k. Together with the upper row this forces
                # w̄_k=0 when the interval is inactive, even for negative products.
                row = np.zeros(n_total)
                row[wbar_col] = -1.0
                row[delta_col] = wk_lo
                _add_row(row, 0.0)

                # Per-interval McCormick with big-M relaxation.
                # The big-M term LOOSENS the constraint when δ_k=0 (interval inactive).
                #
                # cv1: w̄_k ≥ a_k*y + x̄_k*y_lb - a_k*y_lb - M*(1-δ_k)
                #   → -w̄_k + a_k*y + x̄_k*y_lb + M*δ_k ≤ a_k*y_lb + M
                rhs = a_k * yj_lb
                big_m = _inactive_big_m(a_k, rhs)
                row = np.zeros(n_total)
                row[wbar_col] = -1.0
                row[other_var] += a_k
                row[xbar_col] += yj_lb
                row[delta_col] = big_m
                _add_row(row, rhs + big_m)

                # cv2: w̄_k ≥ b_k*y + x̄_k*y_ub - b_k*y_ub - M*(1-δ_k)
                #   → -w̄_k + b_k*y + x̄_k*y_ub + M*δ_k ≤ b_k*y_ub + M
                rhs = b_k * yj_ub
                big_m = _inactive_big_m(b_k, rhs)
                row = np.zeros(n_total)
                row[wbar_col] = -1.0
                row[other_var] += b_k
                row[xbar_col] += yj_ub
                row[delta_col] = big_m
                _add_row(row, rhs + big_m)

                # cc1: w̄_k ≤ b_k*y + x̄_k*y_lb - b_k*y_lb + M*(1-δ_k)
                #   → w̄_k - b_k*y - x̄_k*y_lb + M*δ_k ≤ M - b_k*y_lb
                rhs = -b_k * yj_lb
                big_m = _inactive_big_m(-b_k, rhs)
                row = np.zeros(n_total)
                row[wbar_col] = 1.0
                row[other_var] -= b_k
                row[xbar_col] -= yj_lb
                row[delta_col] = big_m
                _add_row(row, rhs + big_m)

                # cc2: w̄_k ≤ a_k*y + x̄_k*y_ub - a_k*y_ub + M*(1-δ_k)
                #   → w̄_k - a_k*y - x̄_k*y_ub + M*δ_k ≤ M - a_k*y_ub
                rhs = -a_k * yj_ub
                big_m = _inactive_big_m(-a_k, rhs)
                row = np.zeros(n_total)
                row[wbar_col] = 1.0
                row[other_var] -= a_k
                row[xbar_col] -= yj_ub
                row[delta_col] = big_m
                _add_row(row, rhs + big_m)

        else:
            # ── Standard (global) McCormick ──────────────────────────────────
            # cv1: w ≥ xi_lb*xj + xi*xj_lb - xi_lb*xj_lb
            #   →  -w + xj_lb*xi + xi_lb*xj ≤ xi_lb*xj_lb
            row = np.zeros(n_total)
            row[w_col] = -1.0
            row[i] += xj_lb_g
            row[j] += xi_lb_g
            _add_row(row, xi_lb_g * xj_lb_g)

            # cv2: w ≥ xi_ub*xj + xi*xj_ub - xi_ub*xj_ub
            row = np.zeros(n_total)
            row[w_col] = -1.0
            row[i] += xj_ub_g
            row[j] += xi_ub_g
            _add_row(row, xi_ub_g * xj_ub_g)

            # cc1: w ≤ xi_ub*xj + xi*xj_lb - xi_ub*xj_lb
            row = np.zeros(n_total)
            row[w_col] = 1.0
            row[i] -= xj_lb_g
            row[j] -= xi_ub_g
            _add_row(row, -xi_ub_g * xj_lb_g)

            # cc2: w ≤ xi_lb*xj + xi*xj_ub - xi_lb*xj_ub
            row = np.zeros(n_total)
            row[w_col] = 1.0
            row[i] -= xj_ub_g
            row[j] -= xi_lb_g
            _add_row(row, -xi_lb_g * xj_ub_g)

    # ── Ratio-of-products linear-fractional envelopes (issue #185) ──────────
    # Each lifted ratio carries the bilinear identity ``r·q = m`` (m the lifted
    # numerator product, q the lifted denominator product, r the pure ratio).
    # Emitting the four McCormick inequalities of the product ``r·q`` with the
    # product value substituted by the linear term ``m`` outer-approximates the
    # quotient: at any true point r = m/q so r·q = m holds and every row is
    # satisfied, so the relaxed region is a superset of the true one (sound).
    for rr in ratio_relaxations:
        r_col, q_col, m_col = rr.r_col, rr.q_col, rr.m_col
        r_lb, r_ub = (float(v) for v in all_bounds[r_col])
        q_lb, q_ub = (float(v) for v in all_bounds[q_col])
        if not (
            _is_effectively_finite(r_lb)
            and _is_effectively_finite(r_ub)
            and _is_effectively_finite(q_lb)
            and _is_effectively_finite(q_ub)
        ):
            continue
        # m ≥ r_lb·q + q_lb·r − r_lb·q_lb  →  −m + r_lb·q + q_lb·r ≤ r_lb·q_lb
        row = np.zeros(n_total)
        row[m_col] += -1.0
        row[q_col] += r_lb
        row[r_col] += q_lb
        _add_row(row, r_lb * q_lb)
        # m ≥ r_ub·q + q_ub·r − r_ub·q_ub
        row = np.zeros(n_total)
        row[m_col] += -1.0
        row[q_col] += r_ub
        row[r_col] += q_ub
        _add_row(row, r_ub * q_ub)
        # m ≤ r_ub·q + q_lb·r − r_ub·q_lb  →  m − r_ub·q − q_lb·r ≤ −r_ub·q_lb
        row = np.zeros(n_total)
        row[m_col] += 1.0
        row[q_col] += -r_ub
        row[r_col] += -q_lb
        _add_row(row, -r_ub * q_lb)
        # m ≤ r_lb·q + q_ub·r − r_lb·q_ub
        row = np.zeros(n_total)
        row[m_col] += 1.0
        row[q_col] += -r_lb
        row[r_col] += -q_ub
        _add_row(row, -r_lb * q_ub)

    # ── Multilinear RLT convex-hull cuts (emission) ────────────────────────
    # For each recorded term, emit the |S|-degree bound-factor product cuts for
    # every factor-subset S with |S| >= 3 (the |S|=2 cuts are the McCormick
    # envelopes already emitted above). With factor (x_i - xL_i) encoded as
    # (s_i=+1, c_i=-xL_i) and (xU_i - x_i) as (s_i=-1, c_i=+xU_i),
    #   prod_{i in S} (s_i*x_i + c_i) >= 0
    # expands to  sum_{T subset of S} (prod_{i in T} s_i)(prod_{i in S\T} c_i)*w_T
    # (w_T the lifted column of sub-product T, w_{empty}=1, w_{single i}=x_i),
    # which linearizes to a valid cut. The full subset family is the exact hull;
    # each cut alone is sound, so any subset only tightens.
    for _vars, subset_cols in rlt_terms:
        _bnd: dict[int, tuple[float, float]] = {}
        _finite = True
        for c in _vars:
            lo, hi = (float(v) for v in all_bounds[c])
            if not (_is_effectively_finite(lo) and _is_effectively_finite(hi)):
                _finite = False
                break
            _bnd[c] = (lo, hi)
        if not _finite:
            continue
        for k in range(3, len(_vars) + 1):
            for comb in itertools.combinations(_vars, k):
                for corner in itertools.product((0, 1), repeat=k):
                    sc = {
                        i: ((1.0, -_bnd[i][0]) if corner[pos] == 0 else (-1.0, _bnd[i][1]))
                        for pos, i in enumerate(comb)
                    }
                    row = np.zeros(n_total)
                    const = 0.0
                    # iterate every subset T of comb (include empty -> constant)
                    for r in range(k + 1):
                        for tcomb in itertools.combinations(comb, r):
                            tset = frozenset(tcomb)
                            coef = 1.0
                            for i in comb:
                                coef *= sc[i][0] if i in tset else sc[i][1]
                            if r == 0:
                                const += coef
                            else:
                                row[subset_cols[tset]] += coef
                    # product >= 0  ->  -(linear part) <= constant
                    _add_row(-row, const)

    # Binary interval selectors for partitioned convex monomial overestimators.
    # A local secant is valid only on its own interval, so the selector links the
    # original variable to one active interval before applying that secant.
    for (var_idx, n), monomial_intervals in monomial_pw_map.items():
        if not monomial_intervals:
            continue
        s_col = monomial_var_map[(var_idx, n)]
        x_lb, x_ub = [float(v) for v in all_bounds[var_idx]]
        _s_lb, s_ub = [float(v) for v in all_bounds[s_col]]

        row_sum = np.zeros(n_total)
        for delta_col, _, _ in monomial_intervals:
            row_sum[delta_col] = -1.0
        _add_row(row_sum, -1.0)
        _add_row(-row_sum, 1.0)

        for delta_col, a_k, b_k in monomial_intervals:
            lower_m = max(0.0, a_k - x_lb)
            row = np.zeros(n_total)
            row[var_idx] = -1.0
            row[delta_col] = lower_m
            _add_row(row, lower_m - a_k)

            upper_m = max(0.0, x_ub - b_k)
            row = np.zeros(n_total)
            row[var_idx] = 1.0
            row[delta_col] = upper_m
            _add_row(row, b_k + upper_m)

            slope, intercept = _power_secant_line(a_k, b_k, n)
            line_at_lb = slope * x_lb + intercept
            line_at_ub = slope * x_ub + intercept
            line_min = min(line_at_lb, line_at_ub)
            secant_m = max(0.0, s_ub - line_min)

            row = np.zeros(n_total)
            row[s_col] = 1.0
            row[var_idx] = -slope
            row[delta_col] = secant_m
            _add_row(row, intercept + secant_m)

    # ── β-driven piecewise McCormick on bilinear-with-fp ────────────────────
    # For pairs y = w * z where z = β^p is a fractional-power aux, the standard
    # bilinear McCormick uses z's GLOBAL bounds, so it stays loose even after w
    # is heavily partitioned.  When β has a piecewise structure we can derive
    # per-β-interval tight bounds on z (z ∈ [p_k^p, p_{k+1}^p] when β ∈
    # [p_k, p_{k+1}]) and add per-interval big-M McCormick on top of the
    # existing standard or w-piecewise relaxation.  Their intersection is at
    # least as tight, and is dramatically tighter inside each β cell.
    for lin_idx, fp_key in terms.bilinear_with_fp:
        if fp_key not in fractional_power_var_map:
            continue
        fp_col = fractional_power_var_map[fp_key]
        beta_var, p_exp = fp_key
        beta_var = int(beta_var)
        p_exp = float(p_exp)
        pw_intervals = piecewise_var_map.get(beta_var, [])
        if not pw_intervals:
            continue
        pair_key = (min(lin_idx, fp_col), max(lin_idx, fp_col))
        if pair_key not in bilinear_var_map:
            continue
        y_col = bilinear_var_map[pair_key]
        w_lb, w_ub = [float(v) for v in all_bounds[lin_idx]]
        z_lb_global, z_ub_global = [float(v) for v in all_bounds[fp_col]]
        for delta_col, _xbar_col, p_lo, p_hi in pw_intervals:
            try:
                z_at_lo = p_lo**p_exp
                z_at_hi = p_hi**p_exp
            except (ValueError, OverflowError):
                continue
            if not (np.isfinite(z_at_lo) and np.isfinite(z_at_hi)):
                continue
            z_lb_k = min(z_at_lo, z_at_hi)
            z_ub_k = max(z_at_lo, z_at_hi)
            # Skip degenerate intervals.
            if z_ub_k - z_lb_k < 1e-12:
                continue
            corners = [w_lb * z_lb_k, w_lb * z_ub_k, w_ub * z_lb_k, w_ub * z_ub_k]
            # Big-M sized to dominate the global y range when δ_k = 0; use the
            # max global corner so the relaxation is automatically slack on
            # inactive intervals.
            global_corners = [
                w_lb * z_lb_global,
                w_lb * z_ub_global,
                w_ub * z_lb_global,
                w_ub * z_ub_global,
            ]
            M_k = _compute_piecewise_big_m(global_corners + corners)
            # cv1: y ≥ z_lb_k*w + w_lb*z - z_lb_k*w_lb  (relaxed by M when δ=0)
            #   →  -y + z_lb_k*w + w_lb*z + M*δ_k ≤ z_lb_k*w_lb + M
            row = np.zeros(n_total)
            row[y_col] = -1.0
            row[lin_idx] += z_lb_k
            row[fp_col] += w_lb
            row[delta_col] = M_k
            _add_row(row, z_lb_k * w_lb + M_k)
            # cv2: y ≥ z_ub_k*w + w_ub*z - z_ub_k*w_ub
            row = np.zeros(n_total)
            row[y_col] = -1.0
            row[lin_idx] += z_ub_k
            row[fp_col] += w_ub
            row[delta_col] = M_k
            _add_row(row, z_ub_k * w_ub + M_k)
            # cc1: y ≤ z_ub_k*w + w_lb*z - z_ub_k*w_lb
            #   →  y - z_ub_k*w - w_lb*z + M*δ_k ≤ M - z_ub_k*w_lb
            row = np.zeros(n_total)
            row[y_col] = 1.0
            row[lin_idx] -= z_ub_k
            row[fp_col] -= w_lb
            row[delta_col] = M_k
            _add_row(row, M_k - z_ub_k * w_lb)
            # cc2: y ≤ z_lb_k*w + w_ub*z - z_lb_k*w_ub
            row = np.zeros(n_total)
            row[y_col] = 1.0
            row[lin_idx] -= z_lb_k
            row[fp_col] -= w_ub
            row[delta_col] = M_k
            _add_row(row, M_k - z_lb_k * w_ub)

    # Monomial constraints
    for var_idx, n in monomial_terms:
        if (var_idx, n) not in monomial_var_map:
            continue
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        if not (_is_effectively_finite(lb_i) and _is_effectively_finite(ub_i)):
            continue
        s_col = monomial_var_map[(var_idx, n)]
        breakpoints = _guarded_partition_points(
            "monomial tangent",
            (var_idx, n),
            _monomial_breakpoints(var_idx, lb_i, ub_i, disc_state),
        )
        if breakpoints is None:
            breakpoints = _coarse_monomial_breakpoints(lb_i, ub_i)

        def _add_under_tangent(t: float) -> None:
            slope, intercept = _power_tangent_line(t, n)
            row = np.zeros(n_total)
            row[s_col] = -1.0
            row[var_idx] = slope
            _add_row(row, -intercept)

        def _add_over_tangent(t: float) -> None:
            slope, intercept = _power_tangent_line(t, n)
            row = np.zeros(n_total)
            row[s_col] = 1.0
            row[var_idx] = -slope
            _add_row(row, intercept)

        def _add_under_secant(a: float, b: float) -> None:
            slope, intercept = _power_secant_line(a, b, n)
            row = np.zeros(n_total)
            row[s_col] = -1.0
            row[var_idx] = slope
            _add_row(row, -intercept)

        def _add_over_secant(a: float, b: float) -> None:
            slope, intercept = _power_secant_line(a, b, n)
            row = np.zeros(n_total)
            row[s_col] = 1.0
            row[var_idx] = -slope
            _add_row(row, intercept)

        if _power_is_convex_on_box(n, lb_i):
            # Convex on the full domain: tangents underestimate and the secant
            # overestimates. Using all breakpoints makes the relaxation tighten
            # monotonically as the partition is refined.
            for t in breakpoints:
                _add_under_tangent(t)
            _add_over_secant(lb_i, ub_i)
        elif ub_i <= 0.0:
            # Concave on the full domain: the secant underestimates and tangents
            # overestimate.
            _add_under_secant(lb_i, ub_i)
            for t in breakpoints:
                _add_over_tangent(t)
        else:
            # Mixed-sign odd powers change curvature at zero. Keep only tangents that
            # are globally valid on the current box so the relaxation remains sound.
            for t in breakpoints:
                if _odd_mixed_tangent_is_valid(t, lb_i, ub_i, n, "under"):
                    _add_under_tangent(t)
                if _odd_mixed_tangent_is_valid(t, lb_i, ub_i, n, "over"):
                    _add_over_tangent(t)

    # Supported univariate operator graph relaxations.
    def _add_lower_line(relax: UnivariateRelaxation, slope: float, intercept: float) -> None:
        """Add t >= slope * arg + intercept.

        ``arg_coeff`` is normally affine over the original variables
        (``n_orig`` wide), but a lifted reciprocal/sqrt relaxation expresses its
        argument over the *extended* space (original + product aux columns), so
        the slice width follows ``arg_coeff`` rather than a fixed ``n_orig``.
        """
        row = np.zeros(n_total)
        row[: relax.arg_coeff.shape[0]] = slope * relax.arg_coeff
        row[relax.aux_col] = -1.0
        _add_row(row, -intercept - slope * relax.arg_const)

    def _add_upper_line(relax: UnivariateRelaxation, slope: float, intercept: float) -> None:
        """Add t <= slope * arg + intercept."""
        row = np.zeros(n_total)
        row[: relax.arg_coeff.shape[0]] = -slope * relax.arg_coeff
        row[relax.aux_col] = 1.0
        _add_row(row, intercept + slope * relax.arg_const)

    def _add_aux_equality(relax: UnivariateRelaxation, coeff: np.ndarray, rhs: float) -> None:
        """Add equality t + coeff @ x = rhs as two inequality rows."""
        row = np.zeros(n_total)
        row[: coeff.shape[0]] = coeff
        row[relax.aux_col] = 1.0
        _add_row(row, rhs)
        _add_row(-row, -rhs)

    def _add_gated_row(row: np.ndarray, rhs: float, delta_col: int, big_m: float) -> None:
        """Add ``row @ z <= rhs`` active when ``delta_col`` is one."""
        gated = row.copy()
        gated[delta_col] += max(0.0, float(big_m))
        _add_row(gated, rhs + max(0.0, float(big_m)))

    def _linear_line_bounds(
        slope: float,
        intercept: float,
        lb: float,
        ub: float,
    ) -> tuple[float, float]:
        values = [slope * lb + intercept, slope * ub + intercept]
        return min(values), max(values)

    def _add_gated_lower_line(
        relax: UnivariateRelaxation,
        interval: PiecewiseUnivariateInterval,
        slope: float,
        intercept: float,
    ) -> None:
        """Add t >= slope * arg + intercept on one active interval."""
        t_lb, _t_ub = [float(v) for v in all_bounds[relax.aux_col]]
        _line_lb, line_ub = _linear_line_bounds(slope, intercept, relax.arg_lb, relax.arg_ub)
        big_m = max(0.0, line_ub - t_lb)

        row = np.zeros(n_total)
        row[:n_orig] = slope * relax.arg_coeff
        row[relax.aux_col] = -1.0
        rhs = -intercept - slope * relax.arg_const
        _add_gated_row(row, rhs, interval.delta_col, big_m)

    def _add_gated_upper_line(
        relax: UnivariateRelaxation,
        interval: PiecewiseUnivariateInterval,
        slope: float,
        intercept: float,
    ) -> None:
        """Add t <= slope * arg + intercept on one active interval."""
        _t_lb, t_ub = [float(v) for v in all_bounds[relax.aux_col]]
        line_lb, _line_ub = _linear_line_bounds(slope, intercept, relax.arg_lb, relax.arg_ub)
        big_m = max(0.0, t_ub - line_lb)

        row = np.zeros(n_total)
        row[:n_orig] = -slope * relax.arg_coeff
        row[relax.aux_col] = 1.0
        rhs = intercept + slope * relax.arg_const
        _add_gated_row(row, rhs, interval.delta_col, big_m)

    def _add_gated_trig_square_lower_line(
        relax: PiecewiseTrigSquareRelaxation,
        interval: PiecewiseTrigSquareInterval,
        slope: float,
        intercept: float,
    ) -> None:
        """Add q >= slope * arg + intercept on one active interval."""
        q_lb, _q_ub = [float(v) for v in all_bounds[relax.square.aux_col]]
        _line_lb, line_ub = _linear_line_bounds(slope, intercept, relax.arg_lb, relax.arg_ub)
        big_m = max(0.0, line_ub - q_lb)

        row = np.zeros(n_total)
        row[:n_orig] = slope * relax.arg_coeff
        row[relax.square.aux_col] = -1.0
        rhs = -intercept - slope * relax.arg_const
        _add_gated_row(row, rhs, interval.delta_col, big_m)

    def _add_gated_trig_square_upper_line(
        relax: PiecewiseTrigSquareRelaxation,
        interval: PiecewiseTrigSquareInterval,
        slope: float,
        intercept: float,
    ) -> None:
        """Add q <= slope * arg + intercept on one active interval."""
        _q_lb, q_ub = [float(v) for v in all_bounds[relax.square.aux_col]]
        line_lb, _line_ub = _linear_line_bounds(slope, intercept, relax.arg_lb, relax.arg_ub)
        big_m = max(0.0, q_ub - line_lb)

        row = np.zeros(n_total)
        row[:n_orig] = -slope * relax.arg_coeff
        row[relax.square.aux_col] = 1.0
        rhs = intercept + slope * relax.arg_const
        _add_gated_row(row, rhs, interval.delta_col, big_m)

    for relax in univariate_relaxations:
        lb_u = relax.arg_lb
        ub_u = relax.arg_ub
        if abs(ub_u - lb_u) <= 1e-12:
            val = _univariate_value(relax.func_name, lb_u)
            row = np.zeros(n_total)
            row[relax.aux_col] = 1.0
            _add_row(row, val)
            _add_row(-row, -val)
            continue

        if relax.func_name == "abs":
            if lb_u >= 0.0:
                # t = arg
                _add_aux_equality(relax, -relax.arg_coeff, relax.arg_const)
            elif ub_u <= 0.0:
                # t = -arg
                _add_aux_equality(relax, relax.arg_coeff, -relax.arg_const)
            else:
                # t >= arg, t >= -arg, and t below the endpoint secant.
                _add_lower_line(relax, 1.0, 0.0)
                _add_lower_line(relax, -1.0, 0.0)
                f_lb = abs(lb_u)
                f_ub = abs(ub_u)
                slope = (f_ub - f_lb) / (ub_u - lb_u)
                intercept = f_lb - slope * lb_u
                _add_upper_line(relax, slope, intercept)
            continue

        f_lb = _univariate_value(relax.func_name, lb_u)
        f_ub = _univariate_value(relax.func_name, ub_u)
        secant_slope = (f_ub - f_lb) / (ub_u - lb_u)
        secant_intercept = f_lb - secant_slope * lb_u
        if relax.func_name in {"sin", "cos", "tan"}:
            continuous_bounds = _trig_range(relax.func_name, lb_u, ub_u)
            if continuous_bounds is None:
                continue
            val_lb, val_ub = continuous_bounds
        else:
            val_lb, val_ub = [float(v) for v in all_bounds[relax.aux_col]]
        curvature = _univariate_curvature(relax.func_name, val_lb, val_ub)

        if curvature == "convex":
            # Convex: tangents are lower bounds; secant is an upper bound.
            for pt in _tangent_points(relax.func_name, lb_u, ub_u):
                slope = _univariate_grad(relax.func_name, pt)
                intercept = _univariate_value(relax.func_name, pt) - slope * pt
                _add_lower_line(relax, slope, intercept)
            _add_upper_line(relax, secant_slope, secant_intercept)
        elif curvature == "concave":
            # Concave: secant is a lower bound; tangents are upper bounds.
            _add_lower_line(relax, secant_slope, secant_intercept)
            for pt in _tangent_points(relax.func_name, lb_u, ub_u):
                slope = _univariate_grad(relax.func_name, pt)
                intercept = _univariate_value(relax.func_name, pt) - slope * pt
                _add_upper_line(relax, slope, intercept)

    # ── Single-variable composite relaxations (certified curvature) ─────────
    # z is the aux column for f(x_i); each (slope, intercept) line is in x_i.
    # lower_lines give ``z ≥ slope·x_i + intercept`` and upper_lines give
    # ``z ≤ slope·x_i + intercept``. A pinned variable yields an exact equality.
    for crelax in composite_relaxations:
        if crelax.pin_value is not None:
            row = np.zeros(n_total)
            row[crelax.aux_col] = 1.0
            _add_row(row, crelax.pin_value)
            _add_row(-row, -crelax.pin_value)
            continue
        for slope, intercept in crelax.lower_lines:
            row = np.zeros(n_total)
            row[crelax.var_idx] += slope
            row[crelax.aux_col] = -1.0
            _add_row(row, -intercept)
        for slope, intercept in crelax.upper_lines:
            row = np.zeros(n_total)
            row[crelax.var_idx] += -slope
            row[crelax.aux_col] = 1.0
            _add_row(row, intercept)

    for table in finite_domain_trig_square_tables:
        base_col = table.square.base_col
        square_col = table.square.aux_col

        if not table.selector_cols:
            trig_value = table.trig_values[0]
            square_value = table.square_values[0]

            row = np.zeros(n_total)
            row[base_col] = 1.0
            _add_row(row, trig_value)
            _add_row(-row, -trig_value)

            row = np.zeros(n_total)
            row[square_col] = 1.0
            _add_row(row, square_value)
            _add_row(-row, -square_value)
            continue

        row_sum = np.zeros(n_total)
        for selector_col in table.selector_cols:
            row_sum[selector_col] = 1.0
        _add_row(row_sum, 1.0)
        _add_row(-row_sum, -1.0)

        row = np.zeros(n_total)
        row[table.var_idx] = 1.0
        for domain_value, selector_col in zip(table.domain_values, table.selector_cols):
            row[selector_col] -= float(domain_value)
        _add_row(row, 0.0)
        _add_row(-row, 0.0)

        row = np.zeros(n_total)
        row[base_col] = 1.0
        for trig_value, selector_col in zip(table.trig_values, table.selector_cols):
            row[selector_col] -= trig_value
        _add_row(row, 0.0)
        _add_row(-row, 0.0)

        row = np.zeros(n_total)
        row[square_col] = 1.0
        for square_value, selector_col in zip(table.square_values, table.selector_cols):
            row[selector_col] -= square_value
        _add_row(row, 0.0)
        _add_row(-row, 0.0)

    for pw_relax in piecewise_univariate_relaxations:
        relax = pw_relax.relax

        row_sum = np.zeros(n_total)
        for interval in pw_relax.intervals:
            row_sum[interval.delta_col] = -1.0
        _add_row(row_sum, -1.0)
        _add_row(-row_sum, 1.0)

        for interval in pw_relax.intervals:
            arg_lb = float(relax.arg_lb)
            arg_ub = float(relax.arg_ub)

            # arg >= interval.lb when selected.
            lower_m = max(0.0, interval.lb - arg_lb)
            row = np.zeros(n_total)
            row[:n_orig] = -relax.arg_coeff
            rhs = relax.arg_const - interval.lb
            _add_gated_row(row, rhs, interval.delta_col, lower_m)

            # arg <= interval.ub when selected.
            upper_m = max(0.0, arg_ub - interval.ub)
            row = np.zeros(n_total)
            row[:n_orig] = relax.arg_coeff
            rhs = interval.ub - relax.arg_const
            _add_gated_row(row, rhs, interval.delta_col, upper_m)

            if interval.curvature not in {"convex", "concave"}:
                continue

            f_lb = _univariate_value(relax.func_name, interval.lb)
            f_ub = _univariate_value(relax.func_name, interval.ub)
            secant_slope = (f_ub - f_lb) / (interval.ub - interval.lb)
            secant_intercept = f_lb - secant_slope * interval.lb

            if interval.curvature == "convex":
                for pt in _tangent_points(relax.func_name, interval.lb, interval.ub):
                    slope = _univariate_grad(relax.func_name, pt)
                    intercept = _univariate_value(relax.func_name, pt) - slope * pt
                    _add_gated_lower_line(relax, interval, slope, intercept)
                _add_gated_upper_line(relax, interval, secant_slope, secant_intercept)
            else:
                _add_gated_lower_line(relax, interval, secant_slope, secant_intercept)
                for pt in _tangent_points(relax.func_name, interval.lb, interval.ub):
                    slope = _univariate_grad(relax.func_name, pt)
                    intercept = _univariate_value(relax.func_name, pt) - slope * pt
                    _add_gated_upper_line(relax, interval, slope, intercept)

    for trig_square_relax in piecewise_trig_square_relaxations:
        row_sum = np.zeros(n_total)
        for trig_square_interval in trig_square_relax.intervals:
            row_sum[trig_square_interval.delta_col] = -1.0
        _add_row(row_sum, -1.0)
        _add_row(-row_sum, 1.0)

        for trig_square_interval in trig_square_relax.intervals:
            arg_lb = float(trig_square_relax.arg_lb)
            arg_ub = float(trig_square_relax.arg_ub)

            # arg >= interval.lb when selected.
            lower_m = max(0.0, trig_square_interval.lb - arg_lb)
            row = np.zeros(n_total)
            row[:n_orig] = -trig_square_relax.arg_coeff
            rhs = trig_square_relax.arg_const - trig_square_interval.lb
            _add_gated_row(row, rhs, trig_square_interval.delta_col, lower_m)

            # arg <= interval.ub when selected.
            upper_m = max(0.0, arg_ub - trig_square_interval.ub)
            row = np.zeros(n_total)
            row[:n_orig] = trig_square_relax.arg_coeff
            rhs = trig_square_interval.ub - trig_square_relax.arg_const
            _add_gated_row(row, rhs, trig_square_interval.delta_col, upper_m)

            if trig_square_interval.curvature not in {"convex", "concave"}:
                continue

            f_lb = _trig_square_value(trig_square_relax.func_name, trig_square_interval.lb)
            f_ub = _trig_square_value(trig_square_relax.func_name, trig_square_interval.ub)
            secant_slope = (f_ub - f_lb) / (trig_square_interval.ub - trig_square_interval.lb)
            secant_intercept = f_lb - secant_slope * trig_square_interval.lb

            tangent_points = [
                trig_square_interval.lb,
                0.5 * (trig_square_interval.lb + trig_square_interval.ub),
                trig_square_interval.ub,
            ]
            if trig_square_interval.curvature == "convex":
                for pt in _sorted_unique_points(tangent_points):
                    slope = _trig_square_grad(trig_square_relax.func_name, pt)
                    intercept = _trig_square_value(trig_square_relax.func_name, pt) - slope * pt
                    _add_gated_trig_square_lower_line(
                        trig_square_relax,
                        trig_square_interval,
                        slope,
                        intercept,
                    )
                _add_gated_trig_square_upper_line(
                    trig_square_relax,
                    trig_square_interval,
                    secant_slope,
                    secant_intercept,
                )
            else:
                _add_gated_trig_square_lower_line(
                    trig_square_relax,
                    trig_square_interval,
                    secant_slope,
                    secant_intercept,
                )
                for pt in _sorted_unique_points(tangent_points):
                    slope = _trig_square_grad(trig_square_relax.func_name, pt)
                    intercept = _trig_square_value(trig_square_relax.func_name, pt) - slope * pt
                    _add_gated_trig_square_upper_line(
                        trig_square_relax,
                        trig_square_interval,
                        slope,
                        intercept,
                    )

    for square_relax in univariate_square_relaxations:
        lb_i = square_relax.base_lb
        ub_i = square_relax.base_ub
        tangent_pts = [lb_i, ub_i]
        if lb_i <= 0.0 <= ub_i:
            tangent_pts.append(0.0)
        for t in _sorted_unique_points(tangent_pts):
            row = np.zeros(n_total)
            row[square_relax.aux_col] = -1.0
            row[square_relax.base_col] = 2.0 * t
            _add_row(row, t * t)
        if abs(ub_i - lb_i) > 1e-12:
            row = np.zeros(n_total)
            row[square_relax.aux_col] = 1.0
            row[square_relax.base_col] = -(lb_i + ub_i)
            _add_row(row, -lb_i * ub_i)

    # ── Square-of-affine-in-lifted-vars envelopes (issue #155) ──────────────
    # ``s = r**2`` over the affine residual ``r = const + Σ resid[col]*z[col]``,
    # ``r ∈ [r_lb, r_ub]``. ``s`` convex → tangent under-estimators and a secant
    # over-estimator. Every row is an individually valid bound, so any row whose
    # coefficients or rhs reach an extreme magnitude (the residual range hits
    # ~1.6e9 on nvs16, so a tangent at the far endpoint carries a ~1e18 constant)
    # is dropped to keep the LP well-conditioned — omitting an over/under-estimator
    # only enlarges the feasible region and keeps the dual bound sound. The
    # tangent at ``r = 0`` (and the column lower bound ``s_lb``) still pins
    # ``s ≥ 0``, recovering the trivial sum-of-squares bound.
    def _affine_square_row_ok(row: np.ndarray, rhs: float) -> bool:
        if not np.isfinite(rhs) or abs(rhs) > _MONOMIAL_AUX_BOUND_LIMIT:
            return False
        nz = np.flatnonzero(row)
        return bool(nz.size) and float(np.max(np.abs(row[nz]))) <= _MONOMIAL_AUX_BOUND_LIMIT

    for sq in affine_square_relaxations:
        r_lb = sq.r_lb
        r_ub = sq.r_ub
        tangent_pts = [r_lb, r_ub]
        if r_lb <= 0.0 <= r_ub:
            tangent_pts.append(0.0)
        # Tangent under-estimators: s ≥ 2t*r - t^2  →  -s + 2t*r ≤ t^2.
        for t in _sorted_unique_points(tangent_pts):
            row = np.zeros(n_total)
            row[sq.aux_col] = -1.0
            for col, coeff in sq.resid.items():
                row[col] += 2.0 * t * coeff
            rhs = t * t - 2.0 * t * sq.const
            if _affine_square_row_ok(row, rhs):
                _add_row(row, rhs)
        # Secant over-estimator: s ≤ (r_lb+r_ub)*r - r_lb*r_ub.
        if abs(r_ub - r_lb) > 1e-12:
            slope = r_lb + r_ub
            row = np.zeros(n_total)
            row[sq.aux_col] = 1.0
            for col, coeff in sq.resid.items():
                row[col] += -slope * coeff
            rhs = slope * sq.const - r_lb * r_ub
            if _affine_square_row_ok(row, rhs):
                _add_row(row, rhs)

    # ── Scaled single-variable power envelopes ``w = (scale*x)**n`` ─────────
    # ``w = r**n`` over the residual ``r = scale*x`` with ``r ∈ [r_lb, r_ub]``.
    # All rows are written in the original ``x`` column (``r = scale*x``), so the
    # underestimator ``w ≥ a·r + b`` becomes ``a·scale·x − w ≤ −b`` and the
    # overestimator ``w ≤ a·r + b`` becomes ``w − a·scale·x ≤ b``. Each row is an
    # individually valid global bound, so any that exceeds the magnitude cap is
    # dropped (only enlarging the feasible region) and the dual bound stays sound.
    for ap in affine_power_relaxations:
        n = ap.power
        r_lb, r_ub, s, w_col, v_idx = ap.r_lb, ap.r_ub, ap.scale, ap.aux_col, ap.var_idx
        pts = _sorted_unique_points([r_lb, r_ub] + ([0.0] if r_lb < 0.0 < r_ub else []))

        under_lines: list[tuple[float, float]] = []
        over_lines: list[tuple[float, float]] = []
        if _power_is_convex_on_box(n, r_lb):
            # Convex: tangent under-estimators, secant over-estimator.
            under_lines = [_power_tangent_line(t, n) for t in pts]
            over_lines = [_power_secant_line(r_lb, r_ub, n)]
        elif r_ub <= 0.0:
            # Concave (odd power, all-nonpositive box): secant under, tangent over.
            under_lines = [_power_secant_line(r_lb, r_ub, n)]
            over_lines = [_power_tangent_line(t, n) for t in pts]
        else:
            # Odd power spanning zero: neither convex nor concave — keep only the
            # tangents that are globally valid under/over-estimators on the box.
            under_lines = [
                _power_tangent_line(t, n)
                for t in pts
                if _odd_mixed_tangent_is_valid(t, r_lb, r_ub, n, "under")
            ]
            over_lines = [
                _power_tangent_line(t, n)
                for t in pts
                if _odd_mixed_tangent_is_valid(t, r_lb, r_ub, n, "over")
            ]

        for slope, intercept in under_lines:
            row = np.zeros(n_total)
            row[w_col] = -1.0
            row[v_idx] += slope * s
            if _affine_square_row_ok(row, -intercept):
                _add_row(row, -intercept)
        for slope, intercept in over_lines:
            row = np.zeros(n_total)
            row[w_col] = 1.0
            row[v_idx] += -slope * s
            if _affine_square_row_ok(row, intercept):
                _add_row(row, intercept)

    # ── Product-of-squares RLT: w = x_i^2 * x_j^2 = (x_i * x_j)^2 ────────────
    # The McCormick lift of a degree-4 ``x_i^2 * x_j^2`` term as the product of
    # the two square columns ``col(i,2) * col(j,2)`` has the trivial convex
    # under-estimator ``w >= 0`` (both squares are nonnegative), which is very
    # loose. Coupling ``w`` to the bilinear ``p = x_i * x_j`` through the exact
    # identity ``w = p**2`` adds the univariate square envelope on ``p`` —
    # tangent under-estimators ``w >= 2*t*p - t**2`` and a secant over-estimator
    # — every row of which is valid on the true manifold ``w = p**2`` and
    # strictly tighter wherever ``p`` is driven away from zero. Both ``p`` and
    # ``w`` are already lifted columns, so no auxiliary is introduced (nvs20's
    # objective is a sum of such products of one-variable quadratics, issue #175).
    _sq_base_by_col = {
        col: var_idx for (var_idx, power), col in monomial_var_map.items() if power == 2
    }
    for (ci, cj), w_col in list(bilinear_var_map.items()):
        base_i = _sq_base_by_col.get(ci)
        base_j = _sq_base_by_col.get(cj)
        if base_i is None or base_j is None or base_i == base_j:
            continue
        p_col = bilinear_var_map.get((min(base_i, base_j), max(base_i, base_j)))
        if p_col is None:
            continue
        p_lb, p_ub = (float(v) for v in all_bounds[p_col])
        if not (np.isfinite(p_lb) and np.isfinite(p_ub)) or p_ub - p_lb <= 1e-12:
            continue
        tangent_pts = _sorted_unique_points([p_lb, p_ub] + ([0.0] if p_lb < 0.0 < p_ub else []))
        for t in tangent_pts:
            slope, intercept = _power_tangent_line(t, 2)
            row = np.zeros(n_total)
            row[p_col] += slope
            row[w_col] += -1.0
            if _affine_square_row_ok(row, -intercept):
                _add_row(row, -intercept)
        slope, intercept = _power_secant_line(p_lb, p_ub, 2)
        row = np.zeros(n_total)
        row[w_col] += 1.0
        row[p_col] += -slope
        if _affine_square_row_ok(row, intercept):
            _add_row(row, intercept)

    # ── Level-1 RLT product cuts (issue #175) ───────────────────────────────
    # For a linear factor ``g(x) = const + coeff·x <= 0`` and variable ``x_m`` with
    # finite box ``[l_m, u_m]``, both bound-factor products are valid:
    #   g·(x_m - l_m) <= 0  and  g·(u_m - x_m) <= 0
    # (a nonpositive times a nonnegative). Expanding and substituting the lifted
    # product column for each ``x_i*x_m`` gives a linear cut whose value at any true
    # ``(x, x∘x)`` equals the product above, so it never excludes a feasible point.
    for spec in rlt_cut_specs:
        coeff_form = spec["coeff"]
        const_form = spec["const"]
        mm = spec["m"]
        lm = spec["lm"]
        um = spec["um"]
        support = spec["support"]
        prod_cols = spec["prod_cols"]

        # Lower factor (x_m - l_m): const·x_m + Σ coeff_i·w_{i,m}
        #                            − l_m·Σ coeff_i·x_i ≤ const·l_m
        row = np.zeros(n_total)
        row[mm] += const_form
        for i in support:
            coef_i = float(coeff_form[i])
            row[prod_cols[i]] += coef_i
            row[i] += -lm * coef_i
        if _affine_square_row_ok(row, const_form * lm):
            _add_row(row, const_form * lm)

        # Upper factor (u_m - x_m): −const·x_m + u_m·Σ coeff_i·x_i
        #                            − Σ coeff_i·w_{i,m} ≤ −const·u_m
        row = np.zeros(n_total)
        row[mm] += -const_form
        for i in support:
            coef_i = float(coeff_form[i])
            row[i] += um * coef_i
            row[prod_cols[i]] += -coef_i
        if _affine_square_row_ok(row, -const_form * um):
            _add_row(row, -const_form * um)

    # ── Phase 2: quadratic constraint-factor RLT product rows (issue #15) ────
    # Each spec is a quadratic factor ``g {<=,==} 0`` times a bound factor on
    # ``x_m``. ``rlt_quadratic_bound_cut_row`` returns the valid inequality
    # ``coeffs·z ≥ rhs`` (the product ``(-g)·factor ≥ 0``); for ``A_ub z ≤ b`` we
    # add ``-coeffs·z ≤ -rhs``. An equality parent (``g == 0``) also gets the
    # reverse row, pinning the product to zero (strictly tighter than one-sided).
    if rlt_quad_specs:
        from discopt._jax.rlt_cuts import rlt_quadratic_bound_cut_row

        def _orig_col(i: int) -> Optional[int]:
            return i if 0 <= i < n_orig else None

        for spec in rlt_quad_specs:
            prod_map = spec["prod_map"]

            def _prod_col(
                key: tuple[int, ...], _pm: dict[tuple[int, ...], int] = prod_map
            ) -> Optional[int]:
                return _pm.get(tuple(sorted(key)))

            for lower, bnd in ((True, spec["lm"]), (False, spec["um"])):
                assembled = rlt_quadratic_bound_cut_row(
                    spec["quad"],
                    spec["lin"],
                    spec["const"],
                    spec["m"],
                    float(bnd),
                    lower,
                    _orig_col,
                    _prod_col,
                    n_total,
                )
                if assembled is None:
                    continue
                coeffs, rhs = assembled
                # coeffs·z ≥ rhs  ->  -coeffs·z ≤ -rhs.
                if _affine_square_row_ok(-coeffs, -rhs):
                    _add_row(-coeffs, -rhs)
                # Equality parent: also enforce coeffs·z ≤ rhs (two-sided = 0).
                if spec["sense"] == "==" and _affine_square_row_ok(coeffs, rhs):
                    _add_row(coeffs, rhs)

    # ── Fractional-power envelope constraints ──────────────────────────────
    # For a = x^p with x in [lb, ub], lb ≥ 0:
    #   - 0 < p < 1 (concave on x ≥ 0):
    #         secant under-estimator, tangent over-estimators (refined by partition).
    #   - p > 1 or p < 0 with lb > 0 (convex):
    #         tangent under-estimators, secant over-estimator.
    for spec in fractional_power_specs:
        var_idx = spec["var"]
        p = spec["p"]
        a_col = spec["col"]
        lb_i = spec["lb"]
        ub_i = spec["ub"]
        f_lb = spec["f_lb"]
        f_ub = spec["f_ub"]
        convexity = spec["convexity"]

        # Tangent points: include partition breakpoints when available so the
        # relaxation tightens monotonically as AMP refines.
        if var_idx in disc_state.partitions and len(disc_state.partitions[var_idx]) >= 2:
            guarded_tangent_pts = _guarded_partition_points(
                "fractional-power tangent",
                (var_idx, p),
                disc_state.partitions[var_idx],
            )
            if guarded_tangent_pts is None:
                tangent_pts = [lb_i, ub_i]
            else:
                tangent_pts = [float(t) for t in guarded_tangent_pts]
        else:
            tangent_pts = [lb_i, ub_i]
        # Avoid degenerate tangents at zero when the slope or value is undefined.
        tangent_pts = [t for t in tangent_pts if t > 0.0 or (t == 0.0 and p > 1.0)]
        if not tangent_pts:
            tangent_pts = [max(lb_i, 1e-12), ub_i]

        # Secant slope over [lb, ub].
        if abs(ub_i - lb_i) > 1e-12:
            secant_slope = (f_ub - f_lb) / (ub_i - lb_i)
            secant_intercept = f_lb - secant_slope * lb_i
        else:
            secant_slope = 0.0
            secant_intercept = f_lb

        # Conditioning guard (issue #158): for steep powers (``p<0`` near a small
        # ``lb_i``, or ``0<p<1`` near zero) the tangent/secant slopes blow up
        # (``p*t**(p-1)`` and the secant over ``[lb,ub]``), reaching ~1e9+.  An LP
        # row with such a coefficient against an RHS of order 1 is numerically
        # unreliable: HiGHS returns a polytope that EXCLUDES feasible points,
        # which makes OBBT shrink a variable past its true feasible range and the
        # per-node relaxer report a feasible node "infeasible" (nvs08 false
        # optimum).  Dropping any individual cut only ENLARGES the relaxation, so
        # abstaining on an ill-conditioned row is always sound — at worst the aux
        # column degrades to its (exact) value bounds.
        _slope_ok = _envelope_slope_ok

        if convexity == "concave":
            pw_intervals = piecewise_var_map.get(var_idx, [])
            if pw_intervals:
                # Piecewise secant under-estimator: per interval k = [p_k, p_{k+1}],
                # a ≥ p_k^p + slope_k (x − p_k), where slope_k = (p_{k+1}^p − p_k^p) /
                # (p_{k+1} − p_k).  Disaggregated form using shared (δ_k, x̄_k):
                #   −a + Σ_k (slope_k x̄_k + (p_k^p − slope_k p_k) δ_k) ≤ 0.
                row = np.zeros(n_total)
                row[a_col] = -1.0
                _pw_ill_conditioned = False
                for delta_col, xbar_col, p_lo, p_hi in pw_intervals:
                    if abs(p_hi - p_lo) > 1e-12:
                        try:
                            f_plo = p_lo**p
                            f_phi = p_hi**p
                        except (ValueError, OverflowError):
                            continue
                        if not (np.isfinite(f_plo) and np.isfinite(f_phi)):
                            continue
                        slope_k = (f_phi - f_plo) / (p_hi - p_lo)
                        if not _slope_ok(slope_k):
                            _pw_ill_conditioned = True
                            break
                        intercept_k = f_plo - slope_k * p_lo
                        row[xbar_col] += slope_k
                        row[delta_col] += intercept_k
                # A single aggregated row mixes every interval's slope; if any
                # breakpoint is ill-conditioned the whole row is unreliable, so
                # omit it rather than emit a partially-built (unsound) cut.
                if not _pw_ill_conditioned:
                    _add_row(row, 0.0)
            elif _slope_ok(secant_slope):
                # Global secant under-estimator: a ≥ secant_slope*x + secant_intercept
                #   →  -a + secant_slope*x ≤ -secant_intercept
                row = np.zeros(n_total)
                row[a_col] = -1.0
                row[var_idx] = secant_slope
                _add_row(row, -secant_intercept)
            # Tangent over-estimators: a ≤ p*t^(p-1)*(x-t) + t^p
            #   →  a - p*t^(p-1)*x ≤ -((p-1)*t^p) ... derivation below.
            #   t_slope = p*t^(p-1);  t_const = t^p - t_slope*t = (1-p)*t^p
            for t in tangent_pts:
                t_slope = p * (t ** (p - 1.0))
                if not _slope_ok(t_slope):
                    continue
                t_const = (1.0 - p) * (t**p)
                row = np.zeros(n_total)
                row[a_col] = 1.0
                row[var_idx] = -t_slope
                _add_row(row, t_const)
        else:  # convex
            # Tangent under-estimators: a ≥ p*t^(p-1)*(x-t) + t^p
            for t in tangent_pts:
                t_slope = p * (t ** (p - 1.0))
                if not _slope_ok(t_slope):
                    continue
                t_const = (1.0 - p) * (t**p)
                row = np.zeros(n_total)
                row[a_col] = -1.0
                row[var_idx] = t_slope
                _add_row(row, -t_const)
            # Secant over-estimator: a ≤ secant_slope*x + secant_intercept
            if _slope_ok(secant_slope):
                row = np.zeros(n_total)
                row[a_col] = 1.0
                row[var_idx] = -secant_slope
                _add_row(row, secant_intercept)

    if objective_lift is not None:
        for branch_expr in objective_lift.branch_exprs:
            try:
                c_branch, const_branch = _linearize_expr(
                    branch_expr,
                    model,
                    bilinear_var_map,
                    trilinear_var_map,
                    multilinear_var_map,
                    monomial_var_map,
                    univariate_var_map,
                    n_total,
                    fractional_power_var_map=fractional_power_var_map,
                    univariate_square_var_map=univariate_square_var_map,
                    flat_lb=flat_lb,
                    flat_ub=flat_ub,
                    composite_var_map=composite_var_map,
                    composite_coeff_map=composite_coeff_map,
                )
            except ValueError as err:
                logger.debug(
                    "AMP: min/max objective lift uses auxiliary bounds because a branch "
                    "could not be linearized: %s",
                    err,
                )
                continue

            if objective_lift.func_name == "max":
                # minimize max(f_i): f_i <= t  ->  f_i - t <= 0
                row = c_branch.copy()
                row[objective_lift.aux_col] -= 1.0
                _add_row(row, -const_branch)
            else:
                # maximize min(f_i): t <= f_i  ->  t - f_i <= 0
                row = -c_branch
                row[objective_lift.aux_col] += 1.0
                _add_row(row, const_branch)

    # Model constraints
    for constraint in model._constraints:
        body = distributed_bodies[id(constraint)]
        sense = constraint.sense
        if sense == "<=":
            exact_row = _exact_positive_reciprocal_row(body, model, flat_lb, flat_ub)
            if exact_row is not None:
                c_exact, rhs_exact = exact_row
                row_exact = np.zeros(n_total)
                row_exact[:n_orig] = c_exact
                _add_row(row_exact, rhs_exact)
        try:
            crow, const = _linearize_expr(
                body,
                model,
                bilinear_var_map,
                trilinear_var_map,
                multilinear_var_map,
                monomial_var_map,
                univariate_var_map,
                n_total,
                fractional_power_var_map=fractional_power_var_map,
                univariate_square_var_map=univariate_square_var_map,
                flat_lb=flat_lb,
                flat_ub=flat_ub,
                composite_var_map=composite_var_map,
                composite_coeff_map=composite_coeff_map,
            )
            # body ≤ 0  →  c @ z + const ≤ 0  →  c @ z ≤ -const
            if sense == "<=":
                _add_row(crow, -const)
            elif sense == "==":
                _add_row(crow, -const)
                _add_row(-crow, const)
            # (">=" is normalized to "<=" by the Expression operators)
        except ValueError as err:
            # Constraint contains terms we can't linearize (e.g. general nonlinear).
            # Omitting it makes the LP feasible region larger → still a valid lower bound.
            _warn_once(
                "AMP: omitting constraint %s from the MILP relaxation because it cannot "
                "be linearized safely: %s",
                constraint.name or "<unnamed>",
                err,
            )

    # ── OA tangent cuts from NLP incumbent ──────────────────────────────────
    # These are outer-approximation linearizations of the original nonlinear
    # constraints at the incumbent point.  They are in terms of ORIGINAL
    # variables (columns 0..n_orig-1) and tighten the LP relaxation.
    if oa_cuts:
        for coeff, rhs in oa_cuts:
            row = np.zeros(n_total)
            row[: len(coeff)] = coeff[: n_total if len(coeff) > n_total else len(coeff)]
            _add_row(row, rhs)

    # ── Superposition cuts for bilinear-of-nonlinear terms (M8 of #81) ───────
    # For each lifted product w = f(x)*y (a univariate aux times an original
    # variable), add rigorous interior-reference cuts that strictly tighten the
    # compositional McCormick envelope. Every cut is an individually valid bound
    # on the true product surface, so the LP remains a sound lower-bounding
    # relaxation (the rigorous-bound invariant is preserved).
    if superposition:
        _add_superposition_cuts(
            _add_row,
            univariate_by_aux_col,
            bilinear_var_map,
            flat_lb,
            flat_ub,
            n_orig,
            n_total,
        )

    # ── Objective ────────────────────────────────────────────────────────────
    assert model._objective is not None
    assert distributed_objective is not None
    obj_expr = distributed_objective
    try:
        if objective_lift is not None:
            c_obj = np.zeros(n_total)
            c_obj[objective_lift.aux_col] = 1.0
            const_obj = 0.0
        else:
            c_obj, const_obj = _linearize_expr(
                obj_expr,
                model,
                bilinear_var_map,
                trilinear_var_map,
                multilinear_var_map,
                monomial_var_map,
                univariate_var_map,
                n_total,
                fractional_power_var_map=fractional_power_var_map,
                univariate_square_var_map=univariate_square_var_map,
                flat_lb=flat_lb,
                flat_ub=flat_ub,
                composite_var_map=composite_var_map,
                composite_coeff_map=composite_coeff_map,
            )
        objective_bound_valid = True
    except ValueError as err:
        fallback_lb = None
        if model._objective.sense == ObjectiveSense.MINIMIZE:
            # Pass the ORIGINAL (un-distributed) objective: the separable bound
            # matches reciprocal terms on their native ``k/D(x)`` shape so the
            # denominator's ``(x-a)**2`` structure survives for a tight interval
            # enclosure (it distributes the non-reciprocal terms itself).
            fallback_lb = _separable_objective_lower_bound(
                model._objective.expression, model, flat_lb, flat_ub
            )
        if fallback_lb is not None:
            c_obj = np.zeros(n_total)
            const_obj = float(fallback_lb)
            objective_bound_valid = True
            logger.debug(
                "AMP: using separable objective lower bound after linearization failed: %s",
                err,
            )
        else:
            # Keep a feasibility objective so the relaxation can still produce a point,
            # but do not treat the LP value as a sound global bound. Warn loudly:
            # without an objective, AMP's lower-bound machinery is disabled and the
            # solver can only ever return "feasible", never "optimal".
            _warn_once(
                "MILP relaxation could not linearize the objective (%s); falling back to "
                "a feasibility objective. AMP will not be able to produce a lower bound "
                "or certify optimality on this problem.",
                err,
            )
            c_obj = np.zeros(n_total)
            const_obj = 0.0
            objective_bound_valid = False
            logger.debug("AMP: objective is not linearizable; MILP relaxation bound is unavailable")

    # Negate for maximization
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        c_obj = -c_obj
        const_obj = -const_obj

    # ── Assemble and return ──────────────────────────────────────────────────
    if b_rows:
        A_ub_arr = sp.csr_matrix(
            (A_data, (A_row_indices, A_col_indices)),
            shape=(len(b_rows), n_total),
            dtype=np.float64,
        )
        b_ub_arr = np.array(b_rows, dtype=np.float64)
    else:
        A_ub_arr = None
        b_ub_arr = None

    # Build integrality array (1 = integer, 0 = continuous)
    integrality_arr = np.array(integrality_flags, dtype=np.int32)
    has_integers = bool(np.any(integrality_arr > 0))

    milp_model = MilpRelaxationModel(
        c=c_obj,
        A_ub=A_ub_arr,
        b_ub=b_ub_arr,
        bounds=all_bounds,
        obj_offset=const_obj,
        integrality=integrality_arr if has_integers else None,
        objective_bound_valid=objective_bound_valid,
    )

    varmap: dict = {
        "original": {k: k for k in range(n_orig)},
        "bilinear": bilinear_var_map,
        "trilinear": trilinear_var_map,
        "trilinear_stages": trilinear_stage_map,
        "multilinear": multilinear_var_map,
        "multilinear_stages": multilinear_stage_map,
        "monomial": monomial_var_map,
        "monomial_pw": monomial_pw_map,
        "univariate": {k: v for k, v in univariate_var_map.items() if isinstance(k, int)},
        "univariate_signatures": {
            k: v for k, v in univariate_var_map.items() if not isinstance(k, int)
        },
        "univariate_relaxations": univariate_relaxations,
        "composite_relaxations": composite_relaxations,
        "univariate_piecewise_relaxations": piecewise_univariate_relaxations,
        "univariate_square": univariate_square_var_map,
        "univariate_square_relaxations": univariate_square_relaxations,
        "univariate_square_piecewise_relaxations": piecewise_trig_square_relaxations,
        "finite_domain_trig_square_tables": finite_domain_trig_square_tables,
        "fractional_power": fractional_power_var_map,
        "minmax_objective_lift": (
            {
                "func_name": objective_lift.func_name,
                "aux_col": objective_lift.aux_col,
                "branch_bounds": objective_lift.branch_bounds,
                "aux_bounds": objective_lift.aux_bounds,
            }
            if objective_lift is not None
            else None
        ),
        "bilinear_pw": bilinear_pw_map,
        "bilinear_lambda": bilinear_lambda_map,
        "convhull_formulation": convhull_mode,
        "convhull_ebd": convhull_ebd,
        "convhull_ebd_encoding": convhull_ebd_encoding,
        "generation_guardrails": generation_guardrails,
    }

    return milp_model, varmap
