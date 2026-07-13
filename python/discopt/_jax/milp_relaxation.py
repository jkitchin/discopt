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
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt._jax._numeric import EFFECTIVE_INF as _EFFECTIVE_INF
from discopt._jax._numeric import is_effectively_finite as _is_effectively_finite
from discopt._jax.discretization import DiscretizationState
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.operator_relaxations import (
    critical_points_in_interval as _critical_points_in_interval,
)
from discopt._jax.operator_relaxations import tan_range as _tan_range
from discopt._jax.operator_relaxations import trig_range as _trig_range
from discopt._jax.operator_relaxations import trig_square_curvature as _trig_square_curvature
from discopt._jax.operator_relaxations import trig_square_range as _trig_square_range
from discopt._jax.term_classifier import (
    NonlinearTerms,
    _compute_var_offset,
    _get_flat_index,
    distribute_products,
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
from discopt.solver_tuning import current as _tuning

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

# Multi-argument atoms that are *jointly convex* over their domain and therefore
# admit a rigorous supporting-hyperplane (gradient) underestimator via the
# composite-multivariate lift. Maps atom name → required argument count. The
# convexity/domain licence is enforced by ``classify_expr`` in the collector, so
# this table only names the candidate atoms (a structural pre-filter). Currently
# just the GAMS relative-entropy intrinsic ``centropy(x, y) = x·log(x/y)`` (Boyd
# & Vandenberghe §3.2.6), which powers the MINLPLib ``ex6_2_*`` entropy family.
_JOINTLY_CONVEX_MULTIVAR_ATOMS: dict[str, int] = {"centropy": 2}


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

# Coefficient-spread above which an ``infeasible`` verdict from the (numerically
# fragile) Rust simplex is *distrusted* and re-verified with exact equilibration.
# The Rust simplex's internal equilibration is insufficient for the lifted
# relaxation's worst conditioning — RLT cuts on a wide box yield degree-3 monomial
# coefficients with a ~1e5 spread, on which the simplex returns a *false*
# infeasible even though the LP (and HiGHS / the Python-equilibrated simplex) is
# feasible. A false-infeasible at a B&B node would prune the region containing the
# optimum, so re-solving is a soundness guard, not just speed. Set well below the
# (HiGHS-tuned) 1e6 trigger above because the simplex fails at lower spreads.
_RELAX_FALSE_INFEAS_TRIGGER = 1e3


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
    # Pure-Rust certificate side-channel (issue #356), populated only on the
    # warm-started-simplex pure-LP path. ``safe_bound`` is a Neumaier–Shcherbina
    # safe lower bound from the simplex's own row duals — ``<=`` the true optimum
    # at any conditioning, so a caller can fathom on it without an independent
    # (HiGHS) cross-check. ``farkas_certified`` is ``True`` when an ``infeasible``
    # verdict was independently proven by a verified Farkas dual ray. Both default
    # to "unavailable" so the generic / MILP-B&B paths are unaffected.
    safe_bound: Optional[float] = None
    farkas_certified: bool = False


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
        # Warm-start state for the pure-LP simplex fast path (cutting-plane loop):
        # the previous solve's optimal basis and the (structural-cols, rows) it was
        # produced at, so the next ``.solve()`` on the SAME columns with rows only
        # appended can dual-simplex re-optimize from it. See ``_solve_lp_warm``.
        self._warm_basis: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._warm_struct_n: Optional[int] = None
        self._warm_rows: Optional[int] = None

    def solve(
        self,
        time_limit: Optional[float] = None,
        gap_tolerance: float = 1e-4,
        backend: str = "auto",
    ) -> MilpRelaxationResult:
        from discopt.solvers import SolveStatus
        from discopt.solvers.lp_backend import get_milp_solver

        # Warm-startable pure-LP fast path: the spatial cut-separation loop
        # re-solves the SAME structural columns with only rows (cuts) appended, so
        # the previous optimal basis is an ideal dual-simplex warm start. Engage
        # only for the Rust simplex backend on a pure LP (no integrality). A
        # bad/mismatched basis is ignored inside Rust (cold fallback) and the dual
        # simplex converges to the same LP optimum, so the bound is unchanged --
        # warm-start only changes speed. Disable with ``DISCOPT_LP_WARMSTART=0``.
        if (
            backend == "simplex"
            and self._integrality is None
            and self._A_ub is not None
            and _tuning().lp_warmstart
        ):
            warm = self._solve_lp_warm()
            # A warm-start ``infeasible`` on an ill-conditioned LP can be a
            # numerical false-negative; fall through to the equilibrated re-verify
            # below rather than trust it (a false-infeasible would unsoundly prune
            # a B&B node). Any other warm verdict is the true LP optimum.
            if warm is not None and warm.status != "infeasible":
                return warm
            # The bare warm simplex returned ``None`` (iter-limit/numerical) or a
            # possibly-false ``infeasible`` on a badly-scaled LP. Retry with the
            # SAME fast warm simplex on the geometric-mean-equilibrated LP — an
            # exact, feasible-set-preserving rescale that yields the identical
            # optimum (verified equal to the old cold ``solve_milp`` path) at warm
            # speed. This replaces the 170x-slower cold MILP-B&B fallthrough that
            # used to handle these ill-conditioned relaxation solves (nvs21).
            equil = self._solve_lp_warm_equilibrated()
            if equil is not None and equil.status in ("optimal", "infeasible", "unbounded"):
                return equil

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

        # Re-verify a possible *false* infeasible. The Rust simplex can report an
        # ill-conditioned relaxation LP infeasible when it is actually feasible
        # (RLT cuts on a wide box produce a ~1e5 coefficient spread on which the
        # simplex's internal scaling is insufficient). Accepting that verdict at a
        # B&B node would prune a region that may contain the optimum — unsound. The
        # raw solve was NOT Python-equilibrated above for ``backend="simplex"``, so
        # re-solve once with exact geometric-mean equilibration before accepting
        # ``infeasible``. Equilibration preserves the feasible set exactly, so a
        # genuinely infeasible LP stays infeasible; only a numerical false-negative
        # flips to optimal.
        if result.status == SolveStatus.INFEASIBLE and col_scale is None and self._A_ub is not None:
            _nz = np.abs(sp.csr_matrix(self._A_ub).data)
            _nz = _nz[_nz != 0.0]
            if (
                _nz.size
                and np.isfinite(_nz).all()
                and _nz.max() / _nz.min() > _RELAX_FALSE_INFEAS_TRIGGER
            ):
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

    def _solve_lp_warm(self) -> Optional["MilpRelaxationResult"]:
        """Pure-LP warm-started re-solve via the Rust dual simplex.

        Reuses the cached optimal basis from the previous ``.solve()`` when the
        structural column set is unchanged and rows have only grown (the
        cutting-plane case), extending it for the appended slacks. Returns the
        mapped :class:`MilpRelaxationResult`, or ``None`` to defer to the generic
        path (binding unavailable, or an ``iter_limit``/``numerical`` exit). The
        returned objective/bound is the true LP optimum — warm-start is a pure
        speed optimization, never a correctness one.
        """
        from discopt.solvers import SolveStatus

        try:
            from discopt.solvers.milp_simplex import solve_lp_warm_std
        except Exception:  # pragma: no cover - binding absent
            return None

        n_struct = np.asarray(self._c, dtype=np.float64).ravel().shape[0]
        m_now = 0 if self._A_ub is None else sp.csr_matrix(self._A_ub).shape[0]
        in_basis = None
        if (
            self._warm_basis is not None
            and self._warm_struct_n == n_struct
            and self._warm_rows is not None
            and self._warm_rows <= m_now
        ):
            in_basis = self._warm_basis

        try:
            result, out_basis, cert = solve_lp_warm_std(
                self._c, self._A_ub, self._b_ub, self._bounds, in_basis=in_basis, return_cert=True
            )
        except Exception:  # pragma: no cover - defensive; fall back to generic path
            return None
        if result is None:
            # iter_limit / numerical: let the generic path (with its HiGHS option)
            # handle it; drop the stale basis so the next round cold-starts.
            self._warm_basis = None
            return None
        if out_basis is not None:
            self._warm_basis = out_basis
            self._warm_struct_n = n_struct
            self._warm_rows = m_now
        else:
            self._warm_basis = None

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
        bound = None
        if result.bound is not None and self._objective_bound_valid:
            bound = float(result.bound) + self._obj_offset
        safe_bound = None
        if cert.safe_bound is not None and self._objective_bound_valid:
            safe_bound = float(cert.safe_bound) + self._obj_offset
        return MilpRelaxationResult(
            status=status_str,
            objective=obj,
            bound=bound,
            x=result.x,
            safe_bound=safe_bound,
            farkas_certified=bool(cert.farkas_certified),
        )

    def _solve_lp_warm_equilibrated(self) -> Optional["MilpRelaxationResult"]:
        """Warm-simplex re-solve on the *equilibrated* LP.

        The bare warm simplex (:meth:`_solve_lp_warm`) returns ``None`` /
        false-``infeasible`` on a badly-scaled relaxation (the lifted McCormick
        envelope of a high-degree term spans many orders of magnitude — nvs21's
        ``x1**4`` reaches ~1e9). The legacy fallback then cold-solved the same LP
        through the MILP-B&B entry (``solve_milp``) — same Rust engine, no extra
        robustness, ~170x slower. Geometric-mean (Ruiz) equilibration is an exact,
        feasible-set-preserving rescaling, so solving the equilibrated LP with the
        same fast warm simplex yields the *identical* optimum (verified equal to
        the old cold path on nvs21) at warm speed. The objective value is invariant
        under the rescaling; only the returned point maps back via ``col_scale``.
        Returns the result, or ``None`` to defer to the generic path.
        """
        from discopt.solvers import SolveStatus

        if self._A_ub is None:
            return None
        try:
            from discopt.solvers.milp_simplex import solve_lp_warm_std
        except Exception:  # pragma: no cover - binding absent
            return None
        try:
            c_s, A_s, b_s, bounds_s, col_scale = equilibrate_relaxation_lp(
                self._c, self._A_ub, self._b_ub, self._bounds, None
            )
            result, _, cert = solve_lp_warm_std(
                c_s, sp.csr_matrix(A_s), b_s, bounds_s, in_basis=None, return_cert=True
            )
        except Exception:  # pragma: no cover - defensive
            return None
        if result is None:
            return None
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
        bound = None
        if result.bound is not None and self._objective_bound_valid:
            bound = float(result.bound) + self._obj_offset
        # Equilibration is objective-invariant, so the safe bound computed on the
        # rescaled LP is a valid safe bound on the original objective (issue #356).
        safe_bound = None
        if cert.safe_bound is not None and self._objective_bound_valid:
            safe_bound = float(cert.safe_bound) + self._obj_offset
        # Map the scaled solution point back to the original variables (x = D x').
        x_mapped = None
        if result.x is not None:
            x_mapped = np.asarray(result.x, dtype=np.float64) * np.asarray(
                col_scale, dtype=np.float64
            )
        return MilpRelaxationResult(
            status=status_str,
            objective=obj,
            bound=bound,
            x=x_mapped,
            safe_bound=safe_bound,
            farkas_certified=bool(cert.farkas_certified),
        )


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
                    # Register EVERY repeated original-variable factor as a monomial,
                    # not only a pure single-variable product. A multi-repeated mixed
                    # product such as ``x*x*y*y`` decomposes to ``[x, x, y, y]``; the
                    # constraint-linearization collapse needs both ``(x,2)`` and
                    # ``(y,2)`` to rebuild it as the bilinear ``aux(x**2)*aux(y**2)``.
                    # Catching only the ``len(unique)==1`` (pure ``x*x``) case missed
                    # ``(y,2)`` because tree associativity (``((x*x)*y)*y``) never
                    # presents ``y*y`` as a standalone subproduct — so the collapse
                    # failed and the whole constraint dropped from the relaxation.
                    counts: dict[int, int] = {}
                    for i in indices:
                        counts[i] = counts.get(i, 0) + 1
                    for i, c in counts.items():
                        if c >= 2:
                            terms.add((i, c))
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


def _collect_repeated_var_monomials_in_products(
    expr: Expression, model: Model
) -> set[tuple[int, int]]:
    """Register monomials for original variables that repeat as factors of a
    product *even when a sibling factor is an opaque transcendental* (issue #267).

    ``sqrt(x) * y * y`` decomposes (after the univariate lift) to
    ``[col(sqrt), y, y]``; the constraint-linearization collapse then needs the
    ``(y, 2)`` monomial aux to rebuild it as ``aux(sqrt) * aux(y**2)``. The plain
    :func:`_collect_monomial_terms_for_lift` misses it because the whole product
    does not decompose without ``univariate_var_map`` (the ``sqrt`` factor is
    unresolved there). This structural pass walks each maximal ``*`` tree and
    counts the original-variable factors directly, ignoring (but tolerating) the
    opaque factors, so the repeated-variable monomial is registered regardless of
    what the other factors are. Registering an unused monomial only costs one aux
    column and a rigorous power envelope, so it is always sound."""
    terms: set[tuple[int, int]] = set()

    def factor_counts(node: Expression, counts: dict[int, int]) -> None:
        if isinstance(node, BinaryOp) and node.op == "*":
            factor_counts(node.left, counts)
            factor_counts(node.right, counts)
            return
        if isinstance(node, UnaryOp) and node.op == "neg":
            factor_counts(node.operand, counts)
            return
        if isinstance(node, BinaryOp) and node.op == "**" and isinstance(node.right, Constant):
            base = _get_flat_index(node.left, model)
            p = _constant_value(node.right)
            if base is not None and p is not None and p == int(p) and int(p) >= 1:
                counts[base] = counts.get(base, 0) + int(p)
            return
        flat = _get_flat_index(node, model)
        if flat is not None:
            counts[flat] = counts.get(flat, 0) + 1

    def visit(node: Expression) -> None:
        if isinstance(node, BinaryOp):
            if node.op == "*":
                counts: dict[int, int] = {}
                factor_counts(node, counts)
                for v, c in counts.items():
                    if c >= 2:
                        terms.add((v, c))
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
    pinned_value: Optional[Callable[[int], Optional[float]]] = None,
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

    When ``pinned_value`` is supplied (a ``flat_idx -> exact value or None``
    lookup over the node's bounds), a power factor ``x**p`` whose base is pinned
    (lb==ub) is folded into the scalar as the exact constant ``x**p`` rather than
    requiring a fractional-power aux column — which the builder skips on a
    degenerate domain — so a branch/OBBT-pinned ``y * x**p`` term still decomposes
    (to ``[col(y)]`` scaled by ``x**p``) instead of dropping from the relaxation.
    """
    scalar: list[float] = [1.0]
    var_indices: list[int] = []

    def visit(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op == "*":
            return visit(e.left) and visit(e.right)
        if isinstance(e, UnaryOp) and e.op == "neg":
            # A negated factor ``neg(g)`` is ``-1 * g``: peel the sign into the
            # scalar and decompose ``g``. Without this, ``neg(x) * x`` — the
            # internal form a maximize→minimize flip produces for ``-x**2``, and
            # the shape the parser builds for any ``-a*b`` — is an undecomposable
            # product, so the whole term drops from the relaxation and the dual
            # bound freezes. For a pure-integer maximize-of-a-convex objective the
            # spatial B&B then certifies a stationary incumbent (x=0 for x**2) as
            # the optimum: a false-optimal (e.g. ``max x**2`` over integer [-3,3]
            # returned 0 instead of 9). Peeling is exact and sign-only, so it only
            # ever lets the existing envelopes fire.
            scalar[0] *= -1.0
            return visit(e.operand)
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
        # Recognize var^p (fractional p) when an aux column was allocated, or
        # fold it to an exact constant when the base is pinned (lb==ub) at this
        # node. The pinned fold matters because the fractional-power aux column
        # is *skipped* for a degenerate [lb==ub] domain (the builder's bounds
        # guard), so without it a branched/OBBT-pinned base turns ``y * x^p``
        # into an undecomposable product and the whole term — or the objective —
        # drops from the relaxation, sinking the node's dual bound. Folding the
        # pinned power is variable-free and exact, so it only ever tightens.
        if isinstance(e, BinaryOp) and e.op == "**" and isinstance(e.right, Constant):
            base_flat = _get_flat_index(e.left, model)
            if base_flat is not None:
                exp_val = float(e.right.value)
                key = (base_flat, exp_val)
                if fractional_power_var_map and key in fractional_power_var_map:
                    var_indices.append(fractional_power_var_map[key])
                    return True
                # Integer power x**n (n >= 2) → monomial aux column. Without this
                # an integer-power factor inside a product (e.g. ``x**0.5 * y**2``
                # in ex1226's e1) makes the whole product undecomposable and the
                # constraint drops from the relaxation, freezing the dual bound.
                # The monomial aux carries a rigorous power envelope and the
                # bilinear envelope between the two lifted columns is registered by
                # ``_collect_lifted_bilinear_products``, so resolving it here only
                # ever shrinks the relaxed set toward the true one (sound).
                if monomial_var_map and exp_val == int(exp_val) and int(exp_val) >= 2:
                    mono_key = (base_flat, int(exp_val))
                    if mono_key in monomial_var_map:
                        var_indices.append(monomial_var_map[mono_key])
                        return True
                if pinned_value is not None:
                    pv = pinned_value(base_flat)
                    # x^p is real only for x >= 0 (fractional p) or any integer p.
                    if pv is not None and (pv >= 0.0 or exp_val == int(exp_val)):
                        try:
                            scalar[0] *= float(pv) ** exp_val
                        except (ValueError, OverflowError):
                            return False
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

    Raises ValueError when the expression contains nonlinear structure: only
    affine arguments are soundly supported here, since any nonlinear structure
    is relaxed by the uniform engine's atom envelopes rather than linearized.
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


@dataclass
class CompositeMultivarRelaxation:
    """Outer relaxation for a *multivariate* convex/concave nonlinear node.

    Generalizes :class:`CompositeUnivariateRelaxation` to a node ``g(x)`` that
    depends on **more than one** original variable but whose global curvature is
    certified by the DCP classifier — e.g. the Euclidean distance
    ``sqrt((x0-x2)**2 + (x1-x3)**2) = ||A x + b||`` of a TSP-with-neighbourhoods
    objective (MINLPLib ``tspn*``). The aux column ``d`` replaces ``g(x)`` so a
    product such as ``g(x) * x10`` decomposes through the standard McCormick
    bilinear envelope (``d`` registered in ``composite_var_map``).

    Soundness:

    * **CONVEX** ``g`` — each gradient cut ``d ≥ g(x_k) + ∇g(x_k)·(x − x_k)`` is
      a supporting hyperplane, valid *everywhere* for a convex function (no
      finiteness of bounds required for the cut itself), and the column upper
      bound ``d ≤ U`` uses a sound interval over-enclosure of ``g`` on the box.
      Together the tangent cuts (below) and the constant cap (above) form a
      rigorous outer band.
    * **CONCAVE** ``g`` — the roles swap: gradient cuts over-estimate
      (``d ≤ …``) and the constant column lower bound ``d ≥ L`` bounds below.

    Each line is sparse: ``((col, coeff), …), intercept`` meaning
    ``d (≥|≤) Σ coeff·x_col + intercept``.
    """

    expr_id: int
    aux_col: int
    curvature: str
    lower_lines: tuple[tuple[tuple[tuple[int, float], ...], float], ...]
    upper_lines: tuple[tuple[tuple[tuple[int, float], ...], float], ...]
    # Dependent original-variable columns and the compiled value/gradient of the
    # lifted node, so a separator can add the exact supporting hyperplane at the LP
    # point each round (issue #358 Phase 2). ``None`` disables LP-point separation.
    idxs: tuple[int, ...] = ()
    value_fn: Optional[Callable] = None
    grad_fn: Optional[Callable] = None


_COMPOSITE_CURV_TOL = 1e-9
_COMPOSITE_MAX_SUBDIV = 256
# Max sub-boxes a multivariate box-convexity certificate may enumerate across all
# refinement levels of its partition (keeps the interval-Hessian sweep bounded for
# high-dimensional nodes; pinned axes are excluded from the product).
_MULTIVAR_MAX_SUBBOXES = 64


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


def _log_monomial_enabled() -> bool:
    """H-LOG feature flag (``DISCOPT_LOG_MONOMIAL``, default **OFF**).

    Task ``cert:LR-2`` — the narrow positive-product piece F16 left alive
    (``docs/dev/lever-a-root-tightness-plan.md`` §7; ``docs/dev/lr0-logspace-entry``).
    Scoped to a **positive product** ``∏ xᵢ^{aᵢ}`` (every factor lb > 0 on the
    FBBT-tightened box): the log-space relaxation ``zᵢ = ln xᵢ`` (exact concave
    envelope), ``s = Σ aᵢ zᵢ`` (linear), ``t = exp(s)`` (exact convex envelope) is
    far tighter than recursive McCormick for wide boxes. Default-OFF; when OFF the
    relaxation is byte-identical to prior main.
    """
    return os.environ.get("DISCOPT_LOG_MONOMIAL", "0") != "0"


def _expr_has_nonlinear_subterm(expr: Expression) -> bool:
    """True if ``expr`` contains a FunctionCall or a non-unit power — i.e. it is
    not a pure affine combination (which the exact linear path already handles)."""
    if isinstance(expr, FunctionCall):
        return True
    if isinstance(expr, BinaryOp):
        if expr.op == "**":
            return True
        if expr.op == "*":
            # nonlinear unless one side is constant
            if not (isinstance(expr.left, Constant) or isinstance(expr.right, Constant)):
                return True
            return _expr_has_nonlinear_subterm(expr.left) or _expr_has_nonlinear_subterm(expr.right)
        if expr.op in ("+", "-"):
            return _expr_has_nonlinear_subterm(expr.left) or _expr_has_nonlinear_subterm(expr.right)
        return True  # "/" and other ops are nonlinear
    if isinstance(expr, UnaryOp):
        return _expr_has_nonlinear_subterm(expr.operand)
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
    try:
        _linearize_affine_expr(expr.left, model, sum(v.size for v in model._variables))
    except ValueError:
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
    p_int = int(round(p))
    p_is_int = abs(p - p_int) <= 1e-12
    if p_is_int and p_int >= 2 and p_int % 2 == 0:
        return "convex"
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
    """Sound curvature via an *adaptively refined* interval Hessian.

    ``f`` is C² and depends on one variable, so its Hessian is the scalar
    ``f''`` in entry ``(flat_idx, flat_idx)``. Covering ``[lo, hi]`` with
    sub-intervals and enclosing ``f''`` on each tightens the dependency-induced
    looseness of a single whole-box enclosure: if every piece has ``f'' ≥ 0``
    the function is convex on the union ``[lo, hi]`` (symmetrically for concave).

    Rather than re-evaluating a *uniform* grid at escalating densities
    (4, 16, 64, … sub-intervals, each level recomputed from scratch — so a term
    resolving only at the finest level paid for every coarser one), this refines
    adaptively: check the whole box first (a convex/concave term with a tight
    enclosure resolves in a single evaluation), and otherwise bisect *only* the
    sub-intervals whose enclosure is still sign-ambiguous, down to the same
    finest width ``(hi - lo) / _COMPOSITE_MAX_SUBDIV``. A depth-first descent fails
    fast — the first finest-width piece that violates the target curvature refutes
    it immediately. The finest resolution is unchanged, so any term the old
    uniform sweep certified is still certified (no soundness or detection
    regression); localized looseness just costs far fewer Hessian evaluations.
    """
    from discopt._jax.convexity.interval import Interval
    from discopt._jax.convexity.interval_ad import interval_hessian

    if hi - lo <= _COMPOSITE_CURV_TOL:
        return None  # pinned — handled by the exact pin value, not an envelope
    shape = var.shape if var.shape else (1,)
    tol = _COMPOSITE_CURV_TOL
    min_width = (hi - lo) / _COMPOSITE_MAX_SUBDIV
    _ABSTAIN = object()  # sentinel: a non-finite enclosure → give up entirely

    def _hess(a: float, b: float):
        box[var] = Interval(
            np.full(shape, a, dtype=np.float64), np.full(shape, b, dtype=np.float64)
        )
        try:
            ad = interval_hessian(expr, model, box=box)
        except Exception:
            return None
        h = ad.hess
        h_lo = float(h.lo[flat_idx, flat_idx])
        h_hi = float(h.hi[flat_idx, flat_idx])
        if not (np.isfinite(h_lo) and np.isfinite(h_hi)):
            return None
        return h_lo, h_hi

    def _prove(want_convex: bool):
        """True if ``f`` is convex (resp. concave) over ``[lo, hi]``, False if
        refuted at the finest width, ``_ABSTAIN`` on a non-finite enclosure."""
        stack = [(lo, hi)]
        while stack:
            a, b = stack.pop()
            r = _hess(a, b)
            if r is None:
                return _ABSTAIN
            h_lo, h_hi = r
            ok = (h_lo >= -tol) if want_convex else (h_hi <= tol)
            if ok:
                continue  # this piece does not threaten the target curvature
            if (b - a) <= min_width * (1.0 + 1e-9):
                return False  # genuine sign change at the finest resolution
            m = 0.5 * (a + b)
            stack.append((a, m))
            stack.append((m, b))
        return True

    try:
        whole = _hess(lo, hi)
        if whole is None:
            return None
        h_lo, h_hi = whole
        if h_lo >= -tol:
            return "convex"
        if h_hi <= tol:
            return "concave"
        # Genuine straddle on the whole box: refine, trying the leaning direction
        # first (positive-leaning enclosure → convex) then the other.
        lean_convex = (h_lo + h_hi) >= 0.0
        for want_convex in (lean_convex, not lean_convex):
            res = _prove(want_convex)
            if res is _ABSTAIN:
                return None
            if res is True:
                return "convex" if want_convex else "concave"
        return None
    finally:
        box[var] = Interval(
            np.full(shape, lo, dtype=np.float64), np.full(shape, hi, dtype=np.float64)
        )


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


def _alias_equality_defs(model: Model) -> dict[int, Expression]:
    """Map an aux column index to the expression it equals via an ``aux == h``
    equality constraint (the form ``factorable_reform`` emits: ``w - h == 0``).

    Only equalities of the exact shape ``Variable(aux) - h == 0`` (or ``h -
    Variable(aux) == 0``) with ``rhs == 0`` are recognised, so the alias is
    unambiguous. Used by the H-UNI/H-LOG lifts to relax the *aux's* nonlinear
    uses against the underlying ``h`` when the reform has split the structure
    (e.g. nvs09's ``(ln(x-2))**2`` becomes ``aux == ln(x-2)`` + objective
    ``aux**2``). Structure-only — no instance names.
    """
    defs: dict[int, Expression] = {}
    for con in model._constraints:
        if con.sense != "==" or abs(float(con.rhs)) > 1e-12:
            continue
        body = con.body
        if not (isinstance(body, BinaryOp) and body.op == "-"):
            continue
        left, right = body.left, body.right
        # aux - h == 0  (aux is a bare variable/index)
        aux_idx = _get_flat_index(left, model)
        other = right
        if aux_idx is None:
            aux_idx = _get_flat_index(right, model)
            other = left
        if aux_idx is None:
            continue
        # ``other`` must be genuinely nonlinear (an affine alias is already exact).
        if not _expr_has_nonlinear_subterm(other):
            continue
        # First definition wins; a variable defined by two equalities is ambiguous.
        defs.setdefault(aux_idx, other)
    return defs


@dataclass
class LogMonomialRelaxation:
    """Log-space envelope of a positive product ``t == coef·∏ xᵢ^{aᵢ}`` (H-LOG).

    Task ``cert:LR-2``. Binds an existing product column ``t_col`` (the reform's
    ``aux == ∏xᵢ`` alias) to the original factor columns through log space:

    * ``zᵢ = ln xᵢ`` — concave, so tangents overestimate and the endpoint secant
      underestimates (a rigorous outer band on each ``zᵢ`` column).
    * ``s = Σ aᵢ zᵢ`` — exact linear equality.
    * ``t = coef·exp(s)`` — convex, so tangents underestimate and the secant
      overestimates (a rigorous band binding ``t_col``).

    All envelopes are recomputed per node box from ``factor_boxes``/``s_lb,s_ub``;
    strict positivity of every factor is a hard precondition checked on the
    node box (no epsilon shift). Purely additive rows + fresh z/s columns.
    """

    t_col: int
    coef: float
    z_cols: tuple[int, ...]  # one ln-column per distinct factor
    factor_cols: tuple[int, ...]  # the original variable column of each factor
    factor_exps: tuple[float, ...]  # exponent aᵢ (aligned with z_cols/factor_cols)
    factor_boxes: tuple[tuple[float, float], ...]
    s_col: int
    s_lb: float
    s_ub: float


def _extract_positive_product(
    expr: Expression, model: Model, n_orig: int, flat_lb: np.ndarray, flat_ub: np.ndarray
) -> Optional[tuple[float, dict[int, float]]]:
    """Return ``(coef, {orig_var_idx: exponent})`` if ``expr`` is a product/power of
    strictly-positive original variables, else ``None``.

    Strict positivity (lb > 0 on the node box) is required for every factor — the
    log lift is undefined otherwise — and is the H-LOG precondition (no epsilon
    shift). Only *original* variables (index < ``n_orig``) are accepted as factors.
    """
    factors: dict[int, float] = {}
    coef = [1.0]

    def visit(e: Expression, power: float) -> bool:
        if isinstance(e, Constant):
            v = float(e.value)
            if v <= 0.0:
                return False  # a non-positive constant factor breaks the log lift
            coef[0] *= v**power
            return True
        idx = _get_flat_index(e, model)
        if idx is not None:
            if idx >= n_orig:
                return False
            lo = float(flat_lb[idx])
            if not (lo > 1e-9) or not np.isfinite(lo):
                return False  # strict-positivity precondition (no epsilon shift)
            factors[idx] = factors.get(idx, 0.0) + power
            return True
        if isinstance(e, BinaryOp):
            if e.op == "*":
                return visit(e.left, power) and visit(e.right, power)
            if e.op == "/":
                return visit(e.left, power) and visit(e.right, -power)
            if e.op == "**" and isinstance(e.right, Constant):
                return visit(e.left, power * float(e.right.value))
            if e.op in ("+", "-"):
                # Additive-identity passthrough: canonical reconstruct wraps a
                # monomial factor as ``0 + m`` / ``m + 0`` / ``m - 0``. Strip a zero
                # constant operand and recurse into the real factor.
                if isinstance(e.left, Constant) and abs(float(e.left.value)) <= 1e-12:
                    if e.op == "+":
                        return visit(e.right, power)
                elif isinstance(e.right, Constant) and abs(float(e.right.value)) <= 1e-12:
                    return visit(e.left, power)
        # Fallback: a factor that is an affine multiple of a SINGLE positive
        # variable, ``c·x`` with ``c>0`` and zero constant (e.g. ``(30000·x)^-0.48``
        # in a signomial). Then ``(c·x)^power = c^power · x^power`` is still a
        # positive monomial factor. A nonzero constant or >1 variable breaks the
        # monomial form (reject); a non-positive coefficient breaks the log lift.
        try:
            aff_coeffs, aff_const = _linearize_affine_expr(e, model, n_orig)
        except ValueError:
            return False
        if abs(float(aff_const)) > 1e-12:
            return False
        nz = [(j, float(c)) for j, c in enumerate(aff_coeffs) if abs(float(c)) > 0.0]
        if len(nz) != 1:
            return False
        j, c = nz[0]
        if j >= n_orig or c <= 0.0:
            return False
        lo = float(flat_lb[j])
        if not (lo > 1e-9) or not np.isfinite(lo):
            return False
        coef[0] *= c**power
        factors[j] = factors.get(j, 0.0) + power
        return True

    if not visit(expr, 1.0):
        return None
    if not factors:
        return None
    return coef[0], factors


def _referenced_flat_indices(expr: Expression, model: Model) -> list[int]:
    """Sorted distinct flat column indices of the scalar variables in ``expr``."""
    found: set[int] = set()

    def visit(e: Expression) -> None:
        if isinstance(e, Variable):
            offset = _compute_var_offset(e, model)
            for j in range(e.size):
                found.add(offset + j)
            return
        if isinstance(e, IndexExpression):
            if isinstance(e.base, Variable):
                idx = _get_flat_index(e, model)
                if idx is not None:
                    found.add(idx)
                else:
                    offset = _compute_var_offset(e.base, model)
                    for j in range(e.base.size):
                        found.add(offset + j)
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
    return sorted(found)


def _convex_claimer_enabled() -> bool:
    """Whether the convex polynomial subexpression claimer (#358) is on.

    Default OFF while the vertical slice is validated; set ``DISCOPT_CONVEX_CLAIMER``
    to a non-``0`` value to enable. The claim is sound regardless (the collector
    only lifts a node it certifies CONVEX/CONCAVE), so the flag gates *tightness*
    behaviour, not correctness.
    """
    return os.environ.get("DISCOPT_CONVEX_CLAIMER", "0") != "0"


def _multivar_box_curvature(
    expr: Expression,
    model: Model,
    idxs: list[int],
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    box: dict,
) -> Optional[str]:
    """Sound box-restricted ``"convex"``/``"concave"`` certificate for a node.

    The multivariate analogue of :func:`_subdivision_curvature`. ``expr`` is C²,
    so it is convex on the (convex) box iff ``∇²expr ⪰ 0`` at *every* point of the
    box, and concave iff ``∇²expr ⪯ 0`` everywhere. We enclose the Hessian with
    interval AD on each sub-box of an axis-aligned partition and apply a per-row
    interval-Gershgorin eigenvalue bound (identical math to
    :func:`alphabb.rigorous_alpha`): for the dependent submatrix ``H``,

        λ_min(H) ≥ min_i ( H[i,i].lo − Σ_{j≠i} max(|H[i,j].lo|, |H[i,j].hi|) )
        λ_max(H) ≤ max_i ( H[i,i].hi + Σ_{j≠i} max(|H[i,j].lo|, |H[i,j].hi|) ).

    Because PSD-ness is a *pointwise* condition and the sub-boxes cover the box,
    certifying the sign on every sub-box certifies it on the whole box — so a
    function that is only *locally* convex (nonconvex elsewhere, e.g. a ``sqrt`` of
    an indefinite polynomial) is still soundly certified over the region the
    relaxation actually uses. Refines the partition until conclusive or the
    sub-box budget is hit; abstains (returns ``None``) otherwise. This certificate
    is general: it depends on no algebraic shape, only on the interval Hessian, so
    it covers every twice-differentiable multivariate node, not one problem class.

    Only certifies when the dependent axes are finitely bounded (the interval
    Hessian needs a finite box). Off-diagonal couplings to non-dependent variables
    are exactly zero (``expr`` does not depend on them), so restricting Gershgorin
    to the dependent submatrix loses nothing.
    """
    import itertools

    from discopt._jax.convexity.interval import Interval
    from discopt._jax.convexity.interval_ad import interval_hessian

    dep = [int(j) for j in idxs]
    d = len(dep)
    if d == 0:
        return None
    los = np.array([float(flat_lb[j]) for j in dep], dtype=np.float64)
    his = np.array([float(flat_ub[j]) for j in dep], dtype=np.float64)
    if not (np.all(np.isfinite(los)) and np.all(np.isfinite(his))) or np.any(his < los):
        return None
    widths = his - los

    # Locate each dependent flat index within its (possibly vector) Variable.
    var_at: dict[int, tuple[Variable, int]] = {}
    offset = 0
    for v in model._variables:
        for c in range(v.size):
            var_at[offset + c] = (v, c)
        offset += v.size
    if any(j not in var_at for j in dep):
        return None
    loc = {j: var_at[j] for j in dep}
    affected_vars = {loc[j][0] for j in dep}
    saved = {v: box[v] for v in affected_vars}

    tol = _COMPOSITE_CURV_TOL
    non_pinned = [i for i in range(d) if widths[i] > tol]
    ix = np.ix_(dep, dep)

    def _verdict_at(k: int) -> Optional[str]:
        edges = [
            np.linspace(los[i], his[i], k + 1) if widths[i] > tol else np.array([los[i], his[i]])
            for i in range(d)
        ]
        all_convex = True
        all_concave = True
        for combo in itertools.product(*[range(len(e) - 1) for e in edges]):
            lo_arr = {v: np.array(saved[v].lo, dtype=np.float64).reshape(-1) for v in affected_vars}
            hi_arr = {v: np.array(saved[v].hi, dtype=np.float64).reshape(-1) for v in affected_vars}
            for i, j in enumerate(dep):
                v, c = loc[j]
                lo_arr[v][c] = float(edges[i][combo[i]])
                hi_arr[v][c] = float(edges[i][combo[i] + 1])
            for v in affected_vars:
                box[v] = Interval(
                    lo_arr[v].reshape(saved[v].lo.shape), hi_arr[v].reshape(saved[v].hi.shape)
                )
            try:
                ad = interval_hessian(expr, model, box=box)
            except Exception:
                return None
            h_lo = np.asarray(ad.hess.lo, dtype=np.float64)[ix]
            h_hi = np.asarray(ad.hess.hi, dtype=np.float64)[ix]
            if not (np.all(np.isfinite(h_lo)) and np.all(np.isfinite(h_hi))):
                return None
            abs_max = np.maximum(np.abs(h_lo), np.abs(h_hi))
            row_radius = abs_max.sum(axis=1) - np.abs(np.diag(abs_max))
            gersh_lo = np.diag(h_lo) - row_radius  # ≤ λ_min(H)
            gersh_hi = np.diag(h_hi) + row_radius  # ≥ λ_max(H)
            if np.any(gersh_lo < -tol):
                all_convex = False
            if np.any(gersh_hi > tol):
                all_concave = False
            if not all_convex and not all_concave:
                break
        if all_convex:
            return "convex"
        if all_concave:
            return "concave"
        return ""  # inconclusive at this refinement → keep refining

    try:
        if not non_pinned:
            return _verdict_at(1) or None
        k = 1
        while True:
            verdict = _verdict_at(k)
            if verdict is None:
                return None  # non-finite Hessian / AD failure → abstain
            if verdict:
                return verdict
            next_k = k * 2
            if next_k ** len(non_pinned) > _MULTIVAR_MAX_SUBBOXES:
                return None
            k = next_k
    finally:
        for v in affected_vars:
            box[v] = saved[v]


_EMPTY_VARMAP_KEYS: tuple[str, ...] = (
    "bilinear",
    "trilinear",
    "trilinear_stages",
    "multilinear",
    "multilinear_stages",
    "monomial",
    "monomial_pw",
    "univariate",
    "univariate_signatures",
    "univariate_relaxations",
    "composite_relaxations",
    "composite_multivar_relaxations",
    "univariate_piecewise_relaxations",
    "univariate_square",
    "univariate_square_relaxations",
    "univariate_square_piecewise_relaxations",
    "finite_domain_trig_square_tables",
    "fractional_power",
    "bilinear_pw",
    "bilinear_lambda",
    "generation_guardrails",
)


def _empty_varmap(n_orig: int, convhull_mode: str) -> dict:
    """A drop-in ``varmap`` for the engine path: originals mapped, families empty."""
    vm: dict = {k: {} for k in _EMPTY_VARMAP_KEYS}
    vm["original"] = {k: k for k in range(n_orig)}
    vm["minmax_objective_lift"] = None
    vm["convhull_formulation"] = convhull_mode
    vm["convhull_ebd"] = False
    vm["convhull_ebd_encoding"] = "gray"
    vm["generation_guardrails"] = []
    return vm


def _uniform_relaxation_delegate(
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    n_orig: int,
    convhull_mode: str,
) -> tuple["MilpRelaxationModel", dict]:
    """Build the default relaxation through the uniform factorable engine (#632).

    ``build_uniform_relaxation`` (``uniform_relax.py``) relaxes every canonical
    atom class soundly by the auxiliary-variable method and returns a
    :class:`MilpRelaxationModel` with the SAME output contract as the historical
    federated builder — original variables in columns ``0..n_orig-1``, aux columns
    appended after. Soundness is by construction (every emitted row is a valid
    outer inequality at the lifted point); tightness parity with the deleted
    product-side separators (RLT/PSD/finite-domain trig) is the deferred polish
    pass. Original-variable integrality is preserved (aux columns continuous), so
    the integer-aware node solve and every legacy caller keep their contract.
    """
    from discopt._jax.uniform_relax import build_uniform_relaxation

    rel = build_uniform_relaxation(model, box=(flat_lb, flat_ub))
    milp = rel.model
    n_total = int(np.size(milp._c))
    flags = np.zeros(n_total, dtype=np.int32)
    off = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            flags[off : off + v.size] = 1
        off += int(v.size)
    milp._integrality = flags if int(flags.sum()) else None
    # Populate the structural varmap families the proven legacy separators (PSD /
    # RLT / edge-concave / univariate-square / multilinear) consume, from the
    # engine's own decomposition (uniform_relax registered each lifted product /
    # power of ORIGINAL variables to its aux column). This restores product-side
    # tightness parity: the separators now fire on the engine's relaxation exactly
    # as they did on the deleted federation, driven by the uniform factorable
    # decomposition. Every registered aux equals the named product/power and is
    # tied to its originals by the emitted McCormick / secant-tangent rows, so
    # every separated cut is a valid inequality at the lifted feasible point
    # (soundness by construction; see uniform_relax.UniformRelaxation).
    vm = _empty_varmap(n_orig, convhull_mode)
    vm["bilinear"] = dict(rel.bilinear_map)
    vm["monomial"] = dict(rel.monomial_map)
    vm["trilinear"] = dict(rel.trilinear_map)
    vm["multilinear"] = dict(rel.multilinear_map)
    vm["univariate_square"] = dict(rel.univariate_square_map)
    # Composite convex/concave lifts (issue #632 P2): each certified-convex/-concave
    # multivariate node the engine lifted to a single aux is registered here so the
    # existing ``MccormickLPRelaxer._separate_convex`` outer-approximation (Kelley)
    # loop adds its exact supporting tangent at the LP point each round, recovering
    # the composite-convex tightness class generally. Each spec carries a jax value
    # / gradient over the ORIGINAL affine-free variables and its aux column; the
    # tangent of a certified convex (resp. concave) function is a global under-
    # (over-) estimator, so the cut never removes a feasible point (sound by
    # construction; the loop is a sound no-op on any failure).
    vm["composite_multivar_relaxations"] = list(rel.composite_multivar_specs)
    return milp, vm


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

    .. note::
        **#632 cutover — this function now delegates the entire default build to
        the uniform factorable engine** (:func:`uniform_relax.build_uniform_relaxation`),
        which relaxes every canonical atom class soundly via the AVM. As a result
        the following parameters are currently **IGNORED** on the default path and
        kept only for signature compatibility: ``terms``, ``disc_state`` (no
        piecewise-McCormick refinement is fed in — the engine's per-atom envelopes
        supersede it), ``incumbent``, ``oa_cuts`` (OA/Kelley tangents are added
        lazily by the separators at ``solve_at_node``, not pre-seeded here),
        ``convhull_ebd``/``convhull_ebd_encoding``, ``superposition``, and
        ``rlt_level1``. Only ``model``, ``convhull_formulation`` (validated),
        and ``bound_override`` still affect the result. A caller that relied on
        ``disc_state`` piecewise refinement or on feeding back ``oa_cuts`` gets a
        sound but unrefined-by-those-inputs relaxation — the engine is a valid
        outer relaxation by construction, verified ``incorrect_count = 0`` on the
        global50 panel. The docstring below describes the superseded federated
        build and is retained for historical context.

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
    # ── #632 cutover: the uniform factorable engine is the DEFAULT relaxation ──
    # Route the build through build_uniform_relaxation (uniform_relax.py), which
    # relaxes every canonical atom class soundly via the AVM and returns a
    # MilpRelaxationModel with the same column contract. This supersedes the
    # federated collectors/separators below (being deleted stage-by-stage). The
    # engine is a valid outer relaxation by construction; product-side tightness
    # parity is the deferred polish pass.
    return _uniform_relaxation_delegate(model, flat_lb, flat_ub, n_orig, convhull_mode)
