"""General-purpose Outer Approximation (OA) solver for MINLP.

Implements the Duran-Grossmann (1986) / Fletcher-Leyffer (1994) algorithm
with extensions for feasibility cuts, equality relaxation, and ECP mode.

Decomposes MINLP into alternating NLP subproblems (with fixed integers)
and MILP master problems (with accumulated linearization cuts).

The convex-case OA guarantee applies when the minimization objective is
convex, nonlinear inequalities are written in their convex orientation, and
equalities are affine. Nonlinear equalities such as process equations make the
model nonconvex for OA purposes; equality relaxation is a robustness heuristic
and must not be read as restoring the convex-case convergence guarantee.

References:
    Duran & Grossmann, Math. Prog. 36, 1986. DOI: 10.1007/BF02592064
    Fletcher & Leyffer, Math. Prog. 66, 1994. DOI: 10.1007/BF01581153
    Viswanathan & Grossmann, C&CE 14(7), 1990. DOI: 10.1016/0098-1354(90)87085-4
    Westerlund & Pettersson, C&CE 19(S1), 1995. DOI: 10.1016/0098-1354(95)00164-W
"""

from __future__ import annotations

import logging
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from discopt.modeling.core import Constraint, Model, ObjectiveSense, SolveResult, VarType
from discopt.solvers.mip_nlp_options import (
    FP_OPTION_KEYS,
    GOA_AMP_ONLY_OPTION_KEYS,
    GOA_AMP_OPTION_DEFAULTS,
    MIPNLPShotConfig,
)

if TYPE_CHECKING:
    from discopt._jax.nlp_evaluator import NLPEvaluator

logger = logging.getLogger(__name__)

_INIT_STRATEGIES = frozenset({"rNLP", "initial_binary", "max_binary", "fp"})
_REGULARIZATION_MODES = {
    "level_l1": "level_L1",
    "level_l2": "level_L2",
    "level_linfinity": "level_L_infinity",
    "level_l_infinity": "level_L_infinity",
    "level_linf": "level_L_infinity",
    "level_l_inf": "level_L_infinity",
    "grad_lag": "grad_lag",
    "hess_lag": "hess_lag",
    "hess_only_lag": "hess_only_lag",
    "sqp_lag": "sqp_lag",
}
_DERIVATIVE_REGULARIZATION_MODES = frozenset({"grad_lag", "hess_lag", "hess_only_lag", "sqp_lag"})
_HESSIAN_REGULARIZATION_MODES = frozenset({"hess_lag", "hess_only_lag"})
_QP_REGULARIZATION_MODES = frozenset({"level_L2", "hess_lag", "hess_only_lag", "sqp_lag"})
_LINEAR_REGULARIZATION_MODES = frozenset({"level_L1", "level_L_infinity", "grad_lag"})
_FEASIBILITY_NORMS = {
    "l1": "L1",
    "l2": "L2",
    "linfinity": "L_infinity",
    "l_infinity": "L_infinity",
    "l-inf": "L_infinity",
    "l_inf": "L_infinity",
}
_START_BOUND_CLIP = 1e8
_CUT_SOURCE_ORDER = (
    "oa",
    "ecp",
    "objective",
    "objective_rootsearch",
    "esh",
    "feasibility",
    "integer",
    "external",
)


def _float_tuple(values) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1))


def _row_violation(
    coeffs: tuple[float, ...],
    rhs: float,
    supporting_point: Optional[tuple[float, ...]],
) -> Optional[float]:
    if supporting_point is None or len(supporting_point) != len(coeffs):
        return None
    lhs = float(np.dot(np.asarray(coeffs, dtype=np.float64), np.asarray(supporting_point)))
    return max(0.0, lhs - float(rhs))


@dataclass(frozen=True)
class MIPNLPCutRecord:
    """Structured provenance for one generated MIP-NLP master cut."""

    source: str
    global_valid: bool
    local_valid: bool
    supporting_point: Optional[tuple[float, ...]]
    violation: Optional[float]
    constraint_id: Optional[int]
    objective_id: Optional[str]
    coefficients: tuple[float, ...]
    rhs: float
    dedup_key: tuple[tuple[float, ...], float]

    @classmethod
    def from_row(
        cls,
        source: str,
        coeffs,
        rhs: float,
        *,
        global_valid: bool,
        local_valid: bool = True,
        supporting_point=None,
        violation: Optional[float] = None,
        constraint_id: Optional[int] = None,
        objective_id: Optional[str] = None,
    ) -> "MIPNLPCutRecord":
        coeff_tuple = _float_tuple(coeffs)
        rhs_float = float(rhs)
        point_tuple = None if supporting_point is None else _float_tuple(supporting_point)
        violation_float = (
            _row_violation(coeff_tuple, rhs_float, point_tuple)
            if violation is None
            else float(violation)
        )
        return cls(
            source=str(source),
            global_valid=bool(global_valid),
            local_valid=bool(local_valid),
            supporting_point=point_tuple,
            violation=violation_float,
            constraint_id=constraint_id,
            objective_id=objective_id,
            coefficients=coeff_tuple,
            rhs=rhs_float,
            dedup_key=(coeff_tuple, rhs_float),
        )


@dataclass
class MIPNLPCutProvenance:
    """Deduplicated provenance ledger for MIP-NLP master cuts."""

    records: list[MIPNLPCutRecord] = field(default_factory=list)
    _dedup_keys: set[tuple[tuple[float, ...], float]] = field(default_factory=set)

    def add(self, record: MIPNLPCutRecord) -> bool:
        if record.dedup_key in self._dedup_keys:
            return False
        self._dedup_keys.add(record.dedup_key)
        self.records.append(record)
        return True

    def add_row(
        self,
        source: str,
        coeffs,
        rhs: float,
        *,
        global_valid: bool,
        local_valid: bool = True,
        supporting_point=None,
        violation: Optional[float] = None,
        constraint_id: Optional[int] = None,
        objective_id: Optional[str] = None,
    ) -> bool:
        return self.add(
            MIPNLPCutRecord.from_row(
                source,
                coeffs,
                rhs,
                global_valid=global_valid,
                local_valid=local_valid,
                supporting_point=supporting_point,
                violation=violation,
                constraint_id=constraint_id,
                objective_id=objective_id,
            )
        )

    def source_counts(self) -> dict[str, int]:
        counts = Counter(record.source for record in self.records)
        out = {source: int(counts.get(source, 0)) for source in _CUT_SOURCE_ORDER}
        for source, count in sorted(counts.items()):
            out.setdefault(str(source), int(count))
        return out


def _normalize_init_strategy(init_strategy: str) -> str:
    """Normalize and validate the MindtPy-style initialization strategy."""
    if not isinstance(init_strategy, str):
        raise ValueError(f"init_strategy must be a string, got {type(init_strategy).__name__}.")
    key = init_strategy.strip().lower().replace("-", "_")
    if key == "rnlp":
        return "rNLP"
    if key in {"initial_binary", "max_binary", "fp"}:
        return key
    raise ValueError(
        f"Unknown init_strategy={init_strategy!r}. Choose one of: "
        + ", ".join(sorted(_INIT_STRATEGIES))
        + "."
    )


def _normalize_feasibility_norm(feasibility_norm: str) -> str:
    """Normalize and validate the MindtPy-style feasibility norm."""
    if not isinstance(feasibility_norm, str):
        raise ValueError(
            f"feasibility_norm must be a string, got {type(feasibility_norm).__name__}."
        )
    key = feasibility_norm.strip().lower().replace(" ", "_")
    normalized = _FEASIBILITY_NORMS.get(key)
    if normalized is not None:
        return normalized
    raise ValueError(
        f"Unknown feasibility_norm={feasibility_norm!r}. Choose one of: L1, L2, L_infinity."
    )


def _normalize_regularization(add_regularization: Optional[str]) -> Optional[str]:
    """Normalize and validate the supported regularized-OA modes."""
    if add_regularization is None:
        return None
    if not isinstance(add_regularization, str):
        raise ValueError(
            f"add_regularization must be a string or None, got {type(add_regularization).__name__}."
        )
    key = add_regularization.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = _REGULARIZATION_MODES.get(key)
    if normalized is not None:
        return normalized
    raise ValueError(
        "Unknown add_regularization="
        f"{add_regularization!r}. Choose one of: grad_lag, hess_lag, hess_only_lag, "
        "level_L1, level_L2, level_L_infinity, sqp_lag."
    )


def _normalize_positive_float(name: str, value: float) -> float:
    """Validate a strictly positive finite float option."""
    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return out


def _normalize_open_unit_float(name: str, value: float) -> float:
    """Validate a finite float in the open interval ``(0, 1)``."""
    out = float(value)
    if not np.isfinite(out) or out <= 0 or out >= 1:
        raise ValueError(
            f"{name} must be a finite number in the open interval (0, 1), got {value!r}."
        )
    return out


def _normalize_nonnegative_float(name: str, value: float) -> float:
    """Validate a finite nonnegative float option."""
    out = float(value)
    if not np.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}.")
    return out


def _normalize_optional_positive_int(name: str, value: Optional[int]) -> Optional[int]:
    """Validate a positive integer option, allowing None to disable it."""
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}.")
    return out


def _normalize_positive_int(name: str, value: int) -> int:
    """Validate a positive integer option."""
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return out


def _fp_iteration_count(
    max_iterations: int,
    fp_iteration_limit: Optional[int],
    *,
    default_cap: Optional[int] = None,
) -> int:
    """Resolve the FP loop count from explicit and legacy iteration controls."""
    if fp_iteration_limit is not None:
        return _normalize_positive_int("fp_iteration_limit", fp_iteration_limit)
    limit = int(max_iterations) if int(max_iterations) > 0 else 1
    if default_cap is not None:
        limit = min(limit, int(default_cap))
    return max(1, limit)


def _require_solution_pool_backend(milp_solver: str) -> None:
    if not isinstance(milp_solver, str) or milp_solver.strip().lower() != "gurobi":
        raise RuntimeError(
            "OA solution_pool=True requires milp_solver='gurobi' because only "
            "the Gurobi backend currently exposes a MIP solution pool."
        )


def _qp_regularization_backend_error(add_regularization: str) -> RuntimeError:
    return RuntimeError(
        f"OA {add_regularization} regularization requires a QP/MIQP-capable backend "
        "for the regularized master. Install highspy or choose "
        "a linear regularization mode such as add_regularization='level_L1', "
        "'level_L_infinity', or 'grad_lag'."
    )


def _qp_regularization_solve_error(add_regularization: str) -> RuntimeError:
    return RuntimeError(
        f"OA {add_regularization} regularized master was rejected by the QP/MIQP "
        "backend. The regularization Hessian may be nonconvex or indefinite; "
        "use a convex test model, add_regularization='sqp_lag' for a proximal "
        "QP, or a linear derivative mode such as add_regularization='grad_lag'."
    )


def _l2_regularization_backend_error() -> RuntimeError:
    return _qp_regularization_backend_error("level_L2")


def _require_qp_regularization_backend(add_regularization: str) -> None:
    """Raise when the active QP backend cannot solve mixed-integer QPs."""
    try:
        from discopt.solvers.lp_backend import get_qp_solver

        solve_qp = get_qp_solver()
    except ImportError as exc:
        raise _qp_regularization_backend_error(add_regularization) from exc
    if getattr(solve_qp, "__module__", "").endswith("qp_pounce"):
        raise _qp_regularization_backend_error(add_regularization)


def _require_l2_regularization_backend() -> None:
    _require_qp_regularization_backend("level_L2")


def _default_nlp_start(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Return the existing deterministic midpoint NLP start."""
    lb_clip = np.clip(np.asarray(lb, dtype=np.float64), -_START_BOUND_CLIP, _START_BOUND_CLIP)
    ub_clip = np.clip(np.asarray(ub, dtype=np.float64), -_START_BOUND_CLIP, _START_BOUND_CLIP)
    return np.asarray(0.5 * (lb_clip + ub_clip), dtype=np.float64)


def _round_integral_to_bounds(value: float, lb: float, ub: float) -> float:
    """Round half-up, then clamp to the nearest integer-compatible bounds."""
    rounded = float(np.floor(float(value) + 0.5))
    lo = float(np.ceil(lb))
    hi = float(np.floor(ub))
    if lo <= hi:
        return float(np.clip(rounded, lo, hi))
    return float(np.clip(rounded, lb, ub))


def _max_integral_seed(lb: float, ub: float, fallback: float) -> float:
    """Largest practical integer seed; fallback handles effectively unbounded uppers."""
    hi = float(np.floor(ub))
    lo = float(np.ceil(lb))
    if lo <= hi and np.isfinite(hi) and abs(hi) <= _START_BOUND_CLIP:
        return hi
    return _round_integral_to_bounds(fallback, lb, ub)


def _build_initial_strategy_point(
    decomp: _DecomposedProblem,
    init_strategy: str,
    initial_point: Optional[np.ndarray],
) -> np.ndarray:
    """Build the deterministic fixed-integer seed for non-rNLP strategies.

    ``initial_binary`` starts from the user/model start when supplied and rounds
    discrete variables half-up after bound clamping. ``max_binary`` activates
    binary variables at their largest feasible value; for general integers it
    uses the largest practical finite upper-bound value, falling back to the
    rounded clipped midpoint when the upper bound is effectively unbounded.
    """
    x_seed = _default_nlp_start(decomp.lb, decomp.ub)
    if initial_point is not None:
        x0 = np.asarray(initial_point, dtype=np.float64)
        if x0.shape != (decomp.n_vars,):
            raise ValueError(
                f"initial_point has shape {x0.shape}; expected ({decomp.n_vars},) "
                "for MIP-NLP initialization."
            )
        x_seed = np.clip(x0, decomp.lb, decomp.ub)

    if init_strategy == "initial_binary":
        for idx in decomp.int_indices:
            x_seed[idx] = _round_integral_to_bounds(x_seed[idx], decomp.lb[idx], decomp.ub[idx])
        return x_seed

    if init_strategy == "max_binary":
        midpoint = _default_nlp_start(decomp.lb, decomp.ub)
        for idx in decomp.binary_indices:
            x_seed[idx] = _max_integral_seed(decomp.lb[idx], decomp.ub[idx], fallback=1.0)
        for idx in decomp.general_integer_indices:
            x_seed[idx] = _max_integral_seed(
                decomp.lb[idx],
                decomp.ub[idx],
                fallback=midpoint[idx],
            )
        return x_seed

    raise ValueError(
        f"Internal error: non-rNLP initializer received init_strategy={init_strategy!r}."
    )


# ── Configuration ──────────────────────────────────────────────


@dataclass
class OAConfig:
    """Configuration for the OA solver."""

    time_limit: float = 3600.0
    gap_tolerance: float = 1e-4
    max_iterations: int = 100
    nlp_solver: str = "ipm"
    equality_relaxation: bool = False
    ecp_mode: bool = False
    feasibility_cuts: bool = True
    heuristic_nonconvex: bool = False
    add_slack: bool = False
    max_slack: float = 1000.0
    oa_penalty_factor: float = 1000.0
    add_no_good_cuts: bool = False
    feasibility_norm: str = "L_infinity"
    add_regularization: Optional[str] = None
    level_coef: float = 0.5
    stalling_limit: Optional[int] = None
    cycling_check: bool = False
    log_iterations: bool = True


@dataclass(frozen=True)
class _FPConfig:
    """Normalized feasibility-pump option bundle."""

    feasibility_norm: str
    main_norm: str
    add_no_good_cuts: bool
    iteration_limit: Optional[int]
    projzerotol: float
    mipgap: Optional[float]
    discrete_only: bool


def _normalize_fp_config(
    *,
    feasibility_norm: str,
    add_no_good_cuts: bool,
    fp_iteration_limit: Optional[int] = None,
    fp_cutoffdecr: float = 0.0,
    fp_projcuts: Optional[bool] = None,
    fp_transfercuts: bool = False,
    fp_projzerotol: float = 0.0,
    fp_mipgap: Optional[float] = None,
    fp_discrete_only: bool = True,
    fp_main_norm: Optional[str] = None,
    fp_norm_constraint: bool = False,
    fp_norm_constraint_coef: float = 1.0,
) -> _FPConfig:
    """Normalize supported MindtPy-style FP controls and reject unsupported ones."""
    normalized_feasibility_norm = _normalize_feasibility_norm(feasibility_norm)
    normalized_main_norm = _normalize_feasibility_norm(
        normalized_feasibility_norm if fp_main_norm is None else fp_main_norm
    )
    iteration_limit = _normalize_optional_positive_int("fp_iteration_limit", fp_iteration_limit)
    projection_cuts = bool(add_no_good_cuts if fp_projcuts is None else fp_projcuts)
    projzerotol = _normalize_nonnegative_float("fp_projzerotol", fp_projzerotol)
    mipgap = None if fp_mipgap is None else _normalize_nonnegative_float("fp_mipgap", fp_mipgap)
    discrete_only = bool(fp_discrete_only)

    cutoffdecr = _normalize_nonnegative_float("fp_cutoffdecr", fp_cutoffdecr)
    if cutoffdecr > 0.0:
        raise ValueError(
            "Unsupported feasibility-pump option fp_cutoffdecr: discopt does not "
            "currently add improving-objective cutoff constraints during FP. Use "
            "fp_cutoffdecr=0.0."
        )
    if bool(fp_transfercuts):
        raise ValueError(
            "Unsupported feasibility-pump option fp_transfercuts=True: FP projection "
            "cuts are not transferred into OA/GOA master problems. Use "
            "fp_transfercuts=False."
        )
    if bool(fp_norm_constraint):
        raise ValueError(
            "Unsupported feasibility-pump option fp_norm_constraint=True: discopt "
            "does not currently add monotonic norm constraints to FP-NLP subproblems. "
            "Use fp_norm_constraint=False."
        )
    norm_coef = _normalize_positive_float("fp_norm_constraint_coef", fp_norm_constraint_coef)
    if norm_coef != 1.0:
        raise ValueError(
            "Unsupported feasibility-pump option fp_norm_constraint_coef: this option "
            "has no effect unless fp_norm_constraint=True, which discopt does not "
            "currently support. Use fp_norm_constraint_coef=1.0."
        )

    return _FPConfig(
        feasibility_norm=normalized_feasibility_norm,
        main_norm=normalized_main_norm,
        add_no_good_cuts=projection_cuts,
        iteration_limit=iteration_limit,
        projzerotol=projzerotol,
        mipgap=mipgap,
        discrete_only=discrete_only,
    )


@dataclass
class _FeasibilityPumpResult:
    """Best point produced by the MIP-NLP feasibility pump."""

    best_x: Optional[np.ndarray]
    best_obj: Optional[float]
    best_near_x: Optional[np.ndarray]
    best_near_merit: float
    iterations: int = 0
    mip_count: int = 0


@dataclass
class _NLPAttempt:
    """Internal NLP solve result with derivative data retained when available."""

    x: Optional[np.ndarray]
    objective: Optional[float]
    multipliers: Optional[np.ndarray]
    status: Optional[object] = None


@dataclass
class _DerivativeRegularizationData:
    """Lagrangian derivative data used by derivative-based ROA modes."""

    target: np.ndarray
    gradient: np.ndarray
    hessian: Optional[np.ndarray] = None


@dataclass
class _MasterMILPData:
    """Matrix data for an OA-style MILP master."""

    c: np.ndarray
    A_ub: Optional[np.ndarray]
    b_ub: Optional[np.ndarray]
    A_eq: Optional[np.ndarray]
    b_eq: Optional[np.ndarray]
    bounds: list[tuple[float, float]]
    integrality: np.ndarray
    use_objective_epigraph: bool
    slack_index: Optional[int]


# ── Problem Decomposition ─────────────────────────────────────


@dataclass
class _DecomposedProblem:
    """Pre-processed model split into linear and nonlinear parts."""

    evaluator: "NLPEvaluator"
    n_vars: int
    n_cons: int
    lb: np.ndarray
    ub: np.ndarray
    int_indices: list[int]
    binary_indices: list[int]
    general_integer_indices: list[int]
    integrality: np.ndarray
    linear_A_rows: list[np.ndarray]
    linear_b_rows: list[float]
    linear_senses: list[str]
    nonlinear_indices: list[int]
    constraint_senses: list[str]
    obj_coeffs: Optional[tuple] = None
    obj_is_linear: bool = False
    oa_objective_is_convex: bool = True
    oa_constraint_mask: Optional[list[bool]] = None
    master_bound_valid: bool = True
    model: Optional[Model] = None


def _decompose_model(model: Model) -> _DecomposedProblem:
    """Separate model into linear/nonlinear constraints, identify integers."""
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.gdp_reformulate import _extract_body_coeffs, _is_linear
    from discopt._jax.nlp_evaluator import NLPEvaluator

    evaluator = NLPEvaluator(model)
    oa_convexity = classify_oa_cut_convexity(model)
    n_vars = evaluator.n_variables
    n_cons = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds

    # Identify integer/binary variable indices
    int_indices = []
    binary_indices = []
    general_integer_indices = []
    offset = 0
    for v in model._variables:
        if v.var_type == VarType.BINARY:
            for i in range(v.size):
                idx = offset + i
                int_indices.append(idx)
                binary_indices.append(idx)
        elif v.var_type == VarType.INTEGER:
            for i in range(v.size):
                idx = offset + i
                int_indices.append(idx)
                general_integer_indices.append(idx)
        offset += v.size

    integrality = np.zeros(n_vars, dtype=np.int32)
    for idx in int_indices:
        integrality[idx] = 1

    # Classify constraints as linear or nonlinear
    linear_A_rows = []
    linear_b_rows = []
    linear_senses = []
    nonlinear_indices = []

    # Track senses for ALL constraints in evaluator order (nonlinear only)
    all_constraint_senses = []
    eval_idx = 0  # tracks position in evaluator's stacked constraints

    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        if _is_linear(c.body):
            coeffs = _extract_body_coeffs(c.body, model, n_vars)
            if coeffs is not None:
                c_vec, off = coeffs
                linear_A_rows.append(c_vec)
                linear_b_rows.append(-off)
                linear_senses.append(c.sense)
            else:
                nonlinear_indices.append(eval_idx)
        else:
            nonlinear_indices.append(eval_idx)
        all_constraint_senses.append(c.sense)
        eval_idx += 1

    # Check if objective is linear
    raw_obj = model._objective
    obj_coeffs = (
        _extract_body_coeffs(raw_obj.expression, model, n_vars) if raw_obj is not None else None
    )
    obj_is_linear = obj_coeffs is not None
    # The NLPEvaluator works in minimization convention: it negates a MAXIMIZE
    # objective, so the NLP subproblems and the epigraph objective OA cuts all
    # optimize ``-f``. Put the *linear* master objective in the same convention.
    # Without this the master MILP minimizes ``+f`` while the subproblems maximize
    # it, and OA converges to — and certifies as optimal — a wrong point
    # (e.g. syn05m: returned -831 as "optimal" vs the true maximum 837.73).
    if obj_coeffs is not None and raw_obj is not None and raw_obj.sense == ObjectiveSense.MAXIMIZE:
        _c_vec, _c_off = obj_coeffs
        obj_coeffs = (-_c_vec, -_c_off)

    return _DecomposedProblem(
        evaluator=evaluator,
        n_vars=n_vars,
        n_cons=n_cons,
        lb=lb,
        ub=ub,
        int_indices=int_indices,
        binary_indices=binary_indices,
        general_integer_indices=general_integer_indices,
        integrality=integrality,
        linear_A_rows=linear_A_rows,
        linear_b_rows=linear_b_rows,
        linear_senses=linear_senses,
        nonlinear_indices=nonlinear_indices,
        constraint_senses=all_constraint_senses,
        obj_coeffs=obj_coeffs,
        obj_is_linear=obj_is_linear,
        oa_objective_is_convex=oa_convexity.objective_is_convex,
        oa_constraint_mask=oa_convexity.constraint_mask,
        master_bound_valid=(obj_is_linear or oa_convexity.objective_is_convex),
        model=model,
    )


# ── Bounds Proxy ──────────────────────────────────────────────


class _BoundsProxy:
    """Wraps an NLPEvaluator with overridden variable bounds.

    Forwards all attribute access to the underlying evaluator except
    for variable_bounds which returns the overridden bounds.
    """

    def __init__(self, evaluator, new_lb, new_ub):
        self._eval = evaluator
        self._lb = np.asarray(new_lb, dtype=np.float64)
        self._ub = np.asarray(new_ub, dtype=np.float64)

    def __getattr__(self, name):
        # Forward anything not found on self to the underlying evaluator
        return getattr(self._eval, name)

    @property
    def variable_bounds(self):
        return self._lb, self._ub


# ── NLP Subproblem Solvers ────────────────────────────────────


def _is_primal_feasible(evaluator, x, tol: float = 1e-4) -> bool:
    """Return True if x satisfies all constraints within tol."""
    if evaluator.n_constraints == 0:
        return True
    try:
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        cl, cu = _infer_constraint_bounds(evaluator._model)
        cons = np.asarray(evaluator.evaluate_constraints(x))
        return bool(np.all(cons >= cl - tol) and np.all(cons <= cu + tol))
    except Exception:
        return False


def _solve_nlp_attempt(
    evaluator,
    lb,
    ub,
    nlp_solver: str,
    max_iter: int = 200,
    x0=None,
) -> _NLPAttempt:
    """Solve an NLP with given bounds, retaining solver multipliers."""
    if x0 is None:
        x0 = _default_nlp_start(lb, ub)
    else:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != (evaluator.n_variables,):
            raise ValueError(
                f"NLP initial point has shape {x0.shape}; expected ({evaluator.n_variables},)."
            )
        x0 = np.clip(x0, lb, ub)

    try:
        if nlp_solver == "ipopt":
            from discopt.solvers.nlp_ipopt import solve_nlp
        else:
            from discopt.solvers.nlp_pounce import solve_nlp

        result = solve_nlp(evaluator, x0, options={"print_level": 0, "max_iter": max_iter})

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL:
            return _NLPAttempt(
                x=result.x,
                objective=float(evaluator.evaluate_objective(result.x)),
                multipliers=result.multipliers,
                status=result.status,
            )

        # Accept iteration-limited results if the solution is primal feasible.
        # The IPM may not certify dual convergence (code 4: stalled) yet still
        # find a valid primal point, which is sufficient for OA linearization cuts.
        if result.status == SolveStatus.ITERATION_LIMIT and result.x is not None:
            if _is_primal_feasible(evaluator, result.x):
                return _NLPAttempt(
                    x=result.x,
                    objective=float(evaluator.evaluate_objective(result.x)),
                    multipliers=result.multipliers,
                    status=result.status,
                )
    except Exception:
        pass
    return _NLPAttempt(x=None, objective=None, multipliers=None)


def _solve_nlp(evaluator, lb, ub, nlp_solver: str, max_iter: int = 200, x0=None):
    """Solve an NLP with given bounds. Returns (x, obj) or (None, None)."""
    attempt = _solve_nlp_attempt(evaluator, lb, ub, nlp_solver, max_iter=max_iter, x0=x0)
    return attempt.x, attempt.objective


def _maybe_return_nlp_attempt(attempt: _NLPAttempt, return_attempt: bool):
    if return_attempt:
        return attempt
    return attempt.x, attempt.objective


def _solve_nlp_relaxation(
    evaluator,
    lb,
    ub,
    nlp_solver: str,
    initial_point=None,
    return_attempt: bool = False,
):
    """Solve the continuous NLP relaxation (all integers relaxed)."""
    attempt = _solve_nlp_attempt(evaluator, lb, ub, nlp_solver, x0=initial_point)
    return _maybe_return_nlp_attempt(attempt, return_attempt)


def _solve_nlp_subproblem(
    evaluator,
    lb,
    ub,
    int_indices,
    x_master,
    nlp_solver,
    initial_point=None,
    return_attempt: bool = False,
):
    """Fix integers at master values and solve NLP subproblem."""
    sub_lb = lb.copy()
    sub_ub = ub.copy()
    for idx in int_indices:
        val = _round_integral_to_bounds(x_master[idx], lb[idx], ub[idx])
        sub_lb[idx] = val
        sub_ub[idx] = val

    proxy = _BoundsProxy(evaluator, sub_lb, sub_ub)
    attempt = _solve_nlp_attempt(proxy, sub_lb, sub_ub, nlp_solver, x0=initial_point)
    return _maybe_return_nlp_attempt(attempt, return_attempt)


def _regularization_requires_derivatives(add_regularization: Optional[str]) -> bool:
    return add_regularization in _DERIVATIVE_REGULARIZATION_MODES


def _constraint_multipliers_for_regularization(
    decomp: _DecomposedProblem,
    add_regularization: str,
    multipliers: Optional[np.ndarray],
) -> np.ndarray:
    if multipliers is None:
        if decomp.n_cons == 0:
            return np.empty(0, dtype=np.float64)
        raise RuntimeError(
            f"OA {add_regularization} regularization requires NLP dual multipliers, "
            "but the selected NLP backend did not return constraint duals."
        )
    lam = np.asarray(multipliers, dtype=np.float64).reshape(-1)
    if lam.shape != (decomp.n_cons,):
        raise RuntimeError(
            f"OA {add_regularization} regularization received {lam.size} NLP dual "
            f"multipliers for {decomp.n_cons} constraint rows."
        )
    return lam


def _build_derivative_regularization_data(
    decomp: _DecomposedProblem,
    add_regularization: str,
    x_star: np.ndarray,
    multipliers: Optional[np.ndarray],
) -> _DerivativeRegularizationData:
    """Build Lagrangian gradient/Hessian data for derivative ROA modes."""
    x = np.asarray(x_star, dtype=np.float64)
    if x.shape != (decomp.n_vars,):
        raise ValueError(
            f"regularization reference point has shape {x.shape}; expected ({decomp.n_vars},)."
        )

    lam = _constraint_multipliers_for_regularization(decomp, add_regularization, multipliers)
    try:
        grad = np.asarray(decomp.evaluator.evaluate_gradient(x), dtype=np.float64).reshape(-1)
        if decomp.n_cons:
            jac = np.asarray(decomp.evaluator.evaluate_jacobian(x), dtype=np.float64)
            grad = grad + jac.T @ lam
    except Exception as exc:
        raise RuntimeError(
            f"OA {add_regularization} regularization requires NLP gradient and Jacobian "
            "access for the Lagrangian."
        ) from exc
    if grad.shape != (decomp.n_vars,):
        raise RuntimeError(
            f"OA {add_regularization} regularization produced a Lagrangian gradient "
            f"with shape {grad.shape}; expected ({decomp.n_vars},)."
        )

    # A fixed-integer NLP is already first-order stationary in continuous
    # variables up to bound multipliers. Match MindtPy's intent by using the
    # reduced Lagrangian gradient to guide only discrete moves.
    if decomp.int_indices:
        reduced_grad = np.zeros_like(grad)
        reduced_grad[np.asarray(decomp.int_indices, dtype=np.intp)] = grad[decomp.int_indices]
        grad = reduced_grad

    hess = None
    if add_regularization in _HESSIAN_REGULARIZATION_MODES:
        try:
            hess = np.asarray(
                decomp.evaluator.evaluate_lagrangian_hessian(x, 1.0, lam),
                dtype=np.float64,
            )
        except Exception as exc:
            raise RuntimeError(
                f"OA {add_regularization} regularization requires NLP Hessian access "
                "for the Lagrangian."
            ) from exc
        if hess.shape != (decomp.n_vars, decomp.n_vars):
            raise RuntimeError(
                f"OA {add_regularization} regularization received a Lagrangian Hessian "
                f"with shape {hess.shape}; expected ({decomp.n_vars}, {decomp.n_vars})."
            )
        hess = 0.5 * (hess + hess.T)

    return _DerivativeRegularizationData(target=x.copy(), gradient=grad, hessian=hess)


def _constraint_violation_data(evaluator, x) -> tuple[np.ndarray, np.ndarray]:
    """Return nonnegative row violations and active derivative signs."""
    if evaluator.n_constraints == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    vals = np.asarray(evaluator.evaluate_constraints(x), dtype=np.float64)
    cl, cu = _infer_constraint_bounds(evaluator)
    lower = np.zeros_like(vals)
    upper = np.zeros_like(vals)

    finite_lb = cl > -1e19
    finite_ub = cu < 1e19
    lower[finite_lb] = np.maximum(cl[finite_lb] - vals[finite_lb], 0.0)
    upper[finite_ub] = np.maximum(vals[finite_ub] - cu[finite_ub], 0.0)

    use_upper = upper >= lower
    violations = np.where(use_upper, upper, lower)
    signs = np.where(violations > 0, np.where(use_upper, 1.0, -1.0), 0.0)
    return violations, signs


def _constraint_violation_merit(evaluator, x, feasibility_norm: str) -> float:
    """Compute the selected feasibility violation merit at ``x``."""
    violations, _signs = _constraint_violation_data(evaluator, x)
    if violations.size == 0:
        return 0.0
    if feasibility_norm == "L1":
        return float(np.sum(violations))
    if feasibility_norm == "L2":
        return float(np.dot(violations, violations))
    return float(np.max(violations))


class _FeasibilityEvaluator:
    """Bounds-only NLP evaluator that minimizes constraint violation merit."""

    def __init__(self, evaluator, lb, ub, feasibility_norm: str):
        self._eval = evaluator
        self._lb = np.asarray(lb, dtype=np.float64)
        self._ub = np.asarray(ub, dtype=np.float64)
        self._feasibility_norm = feasibility_norm

    @property
    def n_variables(self):
        return self._eval.n_variables

    @property
    def n_constraints(self):
        return 0

    @property
    def variable_bounds(self):
        return self._lb, self._ub

    def evaluate_objective(self, x):
        return _constraint_violation_merit(self._eval, x, self._feasibility_norm)

    def evaluate_gradient(self, x):
        violations, signs = _constraint_violation_data(self._eval, x)
        if violations.size == 0 or np.all(violations <= 0):
            return np.zeros(self.n_variables, dtype=np.float64)

        try:
            jac = np.asarray(self._eval.evaluate_jacobian(x), dtype=np.float64)
        except Exception:
            return np.zeros(self.n_variables, dtype=np.float64)

        if self._feasibility_norm == "L1":
            weights = signs
        elif self._feasibility_norm == "L2":
            weights = 2.0 * violations * signs
        else:
            weights = np.zeros_like(violations)
            weights[int(np.argmax(violations))] = signs[int(np.argmax(violations))]
        return np.asarray(weights @ jac, dtype=np.float64)

    def evaluate_hessian(self, x):
        return np.zeros((self.n_variables, self.n_variables), dtype=np.float64)

    def evaluate_lagrangian_hessian(self, x, obj_factor, lagrange):
        return np.zeros((self.n_variables, self.n_variables), dtype=np.float64)

    def evaluate_constraints(self, x):
        return np.empty(0, dtype=np.float64)

    def evaluate_jacobian(self, x):
        return np.empty((0, self.n_variables), dtype=np.float64)


def _solve_feasibility_subproblem(
    evaluator,
    lb,
    ub,
    int_indices,
    x_master,
    nlp_solver,
    feasibility_norm,
):
    """Solve feasibility problem with fixed integers.

    Minimizes the selected violation norm over the continuous variables with
    the master integer assignment fixed. If that bounded feasibility NLP cannot
    improve the master point, return the clipped master point so OA can still
    generate cuts deterministically.
    """
    sub_lb = lb.copy()
    sub_ub = ub.copy()
    for idx in int_indices:
        val = _round_integral_to_bounds(x_master[idx], lb[idx], ub[idx])
        sub_lb[idx] = val
        sub_ub[idx] = val

    x0 = np.clip(x_master[: evaluator.n_variables], sub_lb, sub_ub)
    best_x = x0
    best_merit = _constraint_violation_merit(evaluator, x0, feasibility_norm)

    try:
        proxy = _FeasibilityEvaluator(evaluator, sub_lb, sub_ub, feasibility_norm)
        x_feas, _obj_feas = _solve_nlp(proxy, sub_lb, sub_ub, nlp_solver, x0=x0)
        if x_feas is not None:
            candidate = np.clip(np.asarray(x_feas, dtype=np.float64), sub_lb, sub_ub)
            candidate_merit = _constraint_violation_merit(evaluator, candidate, feasibility_norm)
            if candidate_merit <= best_merit + 1e-9:
                best_x = candidate
    except Exception:
        pass

    return best_x


def _is_integer_feasible(decomp: _DecomposedProblem, x: np.ndarray, tol: float = 1e-5) -> bool:
    """Return True when all discrete coordinates are integral within tolerance."""
    return all(abs(float(x[idx]) - round(float(x[idx]))) <= tol for idx in decomp.int_indices)


def _snap_integer_assignment(decomp: _DecomposedProblem, x: np.ndarray) -> np.ndarray:
    """Clip a point to bounds and round discrete variables to valid integer values."""
    snapped = np.clip(np.asarray(x, dtype=np.float64), decomp.lb, decomp.ub)
    for idx in decomp.int_indices:
        snapped[idx] = _round_integral_to_bounds(snapped[idx], decomp.lb[idx], decomp.ub[idx])
    return snapped


def _integer_assignment_key(decomp: _DecomposedProblem, x: np.ndarray) -> tuple[float, ...]:
    """Return a stable rounded assignment key for the model's discrete variables."""
    return tuple(
        _round_integral_to_bounds(float(x[idx]), decomp.lb[idx], decomp.ub[idx])
        for idx in decomp.int_indices
    )


def _append_binary_no_good_projection_cut(
    decomp: _DecomposedProblem,
    assignment: tuple[float, ...],
    n_master: int,
    a_rows: list[np.ndarray],
    b_rows: list[float],
) -> bool:
    """Append a binary assignment exclusion cut to the projection MILP."""
    if decomp.general_integer_indices or not decomp.binary_indices:
        # General-integer assignment exclusion needs an orthogonality cut or
        # auxiliary encoding. Keep this projection cut binary-only for now; the
        # pump still detects repeats and returns the best feasible point found.
        return False

    assignment_by_index = dict(zip(decomp.int_indices, assignment))
    coeffs = np.zeros(n_master, dtype=np.float64)
    count_ones = 0
    for idx in decomp.binary_indices:
        val = assignment_by_index[idx]
        if val >= 0.5:
            coeffs[idx] = 1.0
            count_ones += 1
        else:
            coeffs[idx] = -1.0
    a_rows.append(coeffs)
    b_rows.append(float(count_ones - 1))
    return True


def _solve_integer_projection_mip(
    decomp: _DecomposedProblem,
    target: np.ndarray,
    seen_assignments: set[tuple[float, ...]],
    projection_norm: str,
    time_limit: float,
    gap_tolerance: float,
    discrete_only: bool = True,
    projzerotol: float = 0.0,
    milp_solver: str = "auto",
) -> Optional[np.ndarray]:
    """Project the current point to a new integer assignment with a small MILP.

    The projection objective is L1 for ``L1`` and as a MILP-compatible surrogate
    for ``L2``. The fixed-integer feasibility NLP still scores candidates with
    the requested L2 merit. ``L_infinity`` uses one shared deviation variable.
    By default the distance is computed over discrete variables only, matching
    discopt's original FP semantics; ``discrete_only=False`` also penalizes
    continuous-variable movement in the projection MILP.
    """
    try:
        from discopt.solvers import SolveStatus
        from discopt.solvers.lp_backend import get_milp_solver

        solve_milp = get_milp_solver(backend=milp_solver)
    except ImportError:
        return None

    target = np.clip(np.asarray(target, dtype=np.float64), decomp.lb, decomp.ub)
    if projzerotol > 0.0:
        zeroable = (np.abs(target) <= projzerotol) & (decomp.lb <= 0.0) & (decomp.ub >= 0.0)
        target = target.copy()
        target[zeroable] = 0.0
    n_vars = decomp.n_vars
    distance_indices = list(decomp.int_indices if discrete_only else range(n_vars))
    if not distance_indices:
        return _snap_integer_assignment(decomp, target)
    use_linf = projection_norm == "L_infinity"

    a_ub_rows: list[np.ndarray] = []
    b_ub_vals: list[float] = []
    a_eq_rows: list[np.ndarray] = []
    b_eq_vals: list[float] = []

    if use_linf:
        deviation_index = n_vars
        n_master = n_vars + 1
        c = np.zeros(n_master, dtype=np.float64)
        c[deviation_index] = 1.0
        bounds = list(zip(decomp.lb.tolist(), decomp.ub.tolist()))
        bounds.append((0.0, 1e20))

        for idx in distance_indices:
            row = np.zeros(n_master, dtype=np.float64)
            row[idx] = 1.0
            row[deviation_index] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(target[idx]))

            row = np.zeros(n_master, dtype=np.float64)
            row[idx] = -1.0
            row[deviation_index] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(-target[idx]))
    else:
        n_dev = len(distance_indices)
        n_master = n_vars + n_dev
        c = np.zeros(n_master, dtype=np.float64)
        bounds = list(zip(decomp.lb.tolist(), decomp.ub.tolist()))
        for j, idx in enumerate(distance_indices):
            dev_idx = n_vars + j
            c[dev_idx] = 1.0
            width = max(float(decomp.ub[idx] - decomp.lb[idx]), 1.0)
            bounds.append((0.0, width))

            row = np.zeros(n_master, dtype=np.float64)
            row[idx] = 1.0
            row[dev_idx] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(target[idx]))

            row = np.zeros(n_master, dtype=np.float64)
            row[idx] = -1.0
            row[dev_idx] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(-target[idx]))

    for row, rhs, sense in zip(
        decomp.linear_A_rows,
        decomp.linear_b_rows,
        decomp.linear_senses,
    ):
        master_row = np.zeros(n_master, dtype=np.float64)
        master_row[:n_vars] = row
        if sense == "<=":
            a_ub_rows.append(master_row)
            b_ub_vals.append(rhs)
        elif sense == ">=":
            a_ub_rows.append(-master_row)
            b_ub_vals.append(-rhs)
        elif sense == "==":
            a_eq_rows.append(master_row)
            b_eq_vals.append(rhs)

    for assignment in seen_assignments:
        _append_binary_no_good_projection_cut(decomp, assignment, n_master, a_ub_rows, b_ub_vals)

    integrality = np.zeros(n_master, dtype=np.int32)
    integrality[:n_vars] = decomp.integrality
    result = solve_milp(
        c=c,
        A_ub=np.asarray(a_ub_rows, dtype=np.float64) if a_ub_rows else None,
        b_ub=np.asarray(b_ub_vals, dtype=np.float64) if b_ub_vals else None,
        A_eq=np.asarray(a_eq_rows, dtype=np.float64) if a_eq_rows else None,
        b_eq=np.asarray(b_eq_vals, dtype=np.float64) if b_eq_vals else None,
        bounds=bounds,
        integrality=integrality,
        time_limit=max(float(time_limit), 0.0),
        gap_tolerance=gap_tolerance,
    )
    if result.status not in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
        return None
    if result.x is None:
        return None
    return _snap_integer_assignment(decomp, result.x[:n_vars])


def _run_feasibility_pump(
    model: Model,
    decomp: _DecomposedProblem,
    *,
    nlp_solver: str,
    initial_point: Optional[np.ndarray],
    time_limit: float,
    gap_tolerance: float,
    max_iterations: int,
    feasibility_norm: str,
    add_no_good_cuts: bool,
    fp_main_norm: Optional[str] = None,
    fp_mipgap: Optional[float] = None,
    fp_discrete_only: bool = True,
    fp_projzerotol: float = 0.0,
    milp_solver: str = "auto",
) -> _FeasibilityPumpResult:
    """Run a bounded MindtPy-style feasibility pump."""
    t_start = time.perf_counter()
    feasibility_norm = _normalize_feasibility_norm(feasibility_norm)
    projection_norm = _normalize_feasibility_norm(
        feasibility_norm if fp_main_norm is None else fp_main_norm
    )
    projection_gap = (
        float(gap_tolerance)
        if fp_mipgap is None
        else _normalize_nonnegative_float("fp_mipgap", fp_mipgap)
    )
    projzerotol = _normalize_nonnegative_float("fp_projzerotol", fp_projzerotol)
    evaluator = decomp.evaluator
    x_relax, obj_relax = _solve_nlp_relaxation(
        evaluator,
        decomp.lb,
        decomp.ub,
        nlp_solver,
        initial_point=initial_point,
    )
    if x_relax is None:
        current = _default_nlp_start(decomp.lb, decomp.ub)
    else:
        current = np.clip(np.asarray(x_relax, dtype=np.float64), decomp.lb, decomp.ub)

    best_x: Optional[np.ndarray] = None
    best_obj: Optional[float] = None
    best_near_x: Optional[np.ndarray] = current.copy()
    best_near_merit = _constraint_violation_merit(evaluator, current, feasibility_norm)

    def consider(point: np.ndarray, objective: Optional[float] = None) -> bool:
        nonlocal best_x, best_obj, best_near_x, best_near_merit
        x = np.clip(np.asarray(point, dtype=np.float64), decomp.lb, decomp.ub)
        merit = _constraint_violation_merit(evaluator, x, feasibility_norm)
        if merit < best_near_merit - 1e-9:
            best_near_merit = merit
            best_near_x = x.copy()
        if not _is_integer_feasible(decomp, x):
            return False
        if not _is_primal_feasible(evaluator, x):
            return False
        obj = float(evaluator.evaluate_objective(x)) if objective is None else float(objective)
        if best_obj is None or obj < best_obj:
            best_x = x.copy()
            best_obj = obj
        return True

    if x_relax is not None and consider(current, obj_relax):
        return _FeasibilityPumpResult(
            best_x=best_x,
            best_obj=best_obj,
            best_near_x=best_near_x,
            best_near_merit=best_near_merit,
        )

    seen_assignments: set[tuple[float, ...]] = set()
    iterations = 0
    mip_count = 0
    max_rounds = max(1, int(max_iterations))

    for iteration in range(max_rounds):
        if time.perf_counter() - t_start >= time_limit:
            break
        remaining = max(0.0, time_limit - (time.perf_counter() - t_start))
        projected = None
        if add_no_good_cuts:
            projected = _solve_integer_projection_mip(
                decomp,
                current,
                seen_assignments,
                projection_norm,
                remaining,
                projection_gap,
                discrete_only=bool(fp_discrete_only),
                projzerotol=projzerotol,
                milp_solver=milp_solver,
            )
            mip_count += 1
        if projected is None:
            projected = _snap_integer_assignment(decomp, current)

        assignment = _integer_assignment_key(decomp, projected)
        if assignment in seen_assignments:
            break
        seen_assignments.add(assignment)

        x_nlp, obj_nlp = _solve_nlp_subproblem(
            evaluator,
            decomp.lb,
            decomp.ub,
            decomp.int_indices,
            projected,
            nlp_solver,
            initial_point=projected,
        )
        iterations = iteration + 1
        if x_nlp is not None:
            current = x_nlp
            if consider(x_nlp, obj_nlp) and not add_no_good_cuts:
                break
            continue

        x_feas = _solve_feasibility_subproblem(
            evaluator,
            decomp.lb,
            decomp.ub,
            decomp.int_indices,
            projected,
            nlp_solver,
            feasibility_norm,
        )
        if x_feas is not None:
            current = x_feas
            if consider(x_feas) and not add_no_good_cuts:
                break
        else:
            current = projected

    return _FeasibilityPumpResult(
        best_x=best_x,
        best_obj=best_obj,
        best_near_x=best_near_x,
        best_near_merit=best_near_merit,
        iterations=iterations,
        mip_count=mip_count,
    )


# ── Cut Generation ────────────────────────────────────────────


def _constraint_ids_for_generated_oa_cuts(
    evaluator,
    x_point,
    constraint_senses,
    convex_mask,
    *,
    violated_only: bool,
    tol: float = 1e-8,
) -> list[int]:
    """Return constraint ids in the same order as the OA cut generator emits cuts."""
    m = evaluator.n_constraints
    if m == 0:
        return []
    if constraint_senses is None:
        constraint_senses = ["<="] * m
    if not violated_only:
        return [k for k in range(m) if convex_mask is None or bool(convex_mask[k])]

    cons_vals = evaluator.evaluate_constraints(x_point)
    ids: list[int] = []
    for k in range(m):
        if convex_mask is not None and not bool(convex_mask[k]):
            continue
        g_k = float(cons_vals[k])
        sense = constraint_senses[k]
        violated = (
            (sense == "<=" and g_k > tol)
            or (sense == ">=" and g_k < -tol)
            or (sense == "==" and abs(g_k) > tol)
        )
        if violated:
            ids.append(k)
    return ids


def _constraint_cut_global_valid(
    constraint_convex_mask,
    constraint_id: Optional[int],
    original_sense: str,
    equality_relaxation: bool,
) -> bool:
    if equality_relaxation and original_sense == "==":
        return False
    if constraint_convex_mask is None or constraint_id is None:
        return True
    return bool(constraint_convex_mask[constraint_id])


def _append_master_cut(
    oa_A_rows,
    oa_b_rows,
    coeffs,
    rhs,
    oa_cut_relaxable=None,
    relaxable=True,
    cut_provenance: Optional[MIPNLPCutProvenance] = None,
    source: Optional[str] = None,
    global_valid: bool = True,
    local_valid: bool = True,
    supporting_point=None,
    violation: Optional[float] = None,
    constraint_id: Optional[int] = None,
    objective_id: Optional[str] = None,
):
    """Append a master cut and optional slack-relaxability metadata."""
    oa_A_rows.append(coeffs)
    oa_b_rows.append(rhs)
    if oa_cut_relaxable is not None:
        oa_cut_relaxable.append(bool(relaxable))
    if cut_provenance is not None and source is not None:
        cut_provenance.add_row(
            source,
            coeffs,
            rhs,
            global_valid=global_valid,
            local_valid=local_valid,
            supporting_point=supporting_point,
            violation=violation,
            constraint_id=constraint_id,
            objective_id=objective_id,
        )


def _add_oa_cuts(
    evaluator,
    x_star,
    n_vars,
    n_cons,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    obj_is_linear,
    constraint_convex_mask,
    objective_is_convex,
    equality_relaxation=False,
    oa_cut_relaxable=None,
    cut_provenance: Optional[MIPNLPCutProvenance] = None,
):
    """Generate OA cuts at x_star and append to cut lists.

    Constraint cuts have length n_vars.
    Objective cuts (when nonlinear) have length n_vars+1, with the last
    element being the -eta epigraph coefficient.
    """
    from discopt._jax.cutting_planes import (
        generate_oa_cuts_from_evaluator,
        generate_objective_oa_cut,
    )

    if n_cons > 0:
        cuts = generate_oa_cuts_from_evaluator(
            evaluator,
            x_star,
            constraint_senses=constraint_senses,
            convex_mask=constraint_convex_mask,
        )
        constraint_ids = _constraint_ids_for_generated_oa_cuts(
            evaluator,
            x_star,
            constraint_senses,
            constraint_convex_mask,
            violated_only=False,
        )
        for constraint_id, cut in zip(constraint_ids, cuts):
            coeffs = cut.coeffs.copy()
            # Filter degenerate cuts
            if np.linalg.norm(coeffs) < 1e-12:
                continue

            sense = cut.sense
            original_sense = sense
            if equality_relaxation and sense == "==":
                sense = "<="
            global_valid = _constraint_cut_global_valid(
                constraint_convex_mask,
                constraint_id,
                original_sense,
                equality_relaxation,
            )

            if sense == "<=":
                _append_master_cut(
                    oa_A_rows,
                    oa_b_rows,
                    coeffs,
                    cut.rhs,
                    oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                    source="oa",
                    global_valid=global_valid,
                    supporting_point=x_star,
                    constraint_id=constraint_id,
                )
            elif sense == ">=":
                _append_master_cut(
                    oa_A_rows,
                    oa_b_rows,
                    -coeffs,
                    -cut.rhs,
                    oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                    source="oa",
                    global_valid=global_valid,
                    supporting_point=x_star,
                    constraint_id=constraint_id,
                )
            elif sense == "==":
                # Equality: add both <= and >= cuts
                _append_master_cut(
                    oa_A_rows,
                    oa_b_rows,
                    coeffs,
                    cut.rhs,
                    oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                    source="oa",
                    global_valid=global_valid,
                    supporting_point=x_star,
                    constraint_id=constraint_id,
                )
                _append_master_cut(
                    oa_A_rows,
                    oa_b_rows,
                    -coeffs,
                    -cut.rhs,
                    oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                    source="oa",
                    global_valid=global_valid,
                    supporting_point=x_star,
                    constraint_id=constraint_id,
                )

    # Objective OA cut (only if nonlinear): grad^T x - eta <= rhs
    if not obj_is_linear and objective_is_convex:
        n_master = n_vars + 1
        obj_value = float(evaluator.evaluate_objective(x_star))
        obj_support = np.concatenate([np.asarray(x_star, dtype=np.float64), [obj_value]])
        obj_cut = generate_objective_oa_cut(evaluator, x_star, n_master, z_index=n_vars)
        _append_master_cut(
            oa_A_rows,
            oa_b_rows,
            obj_cut.coeffs.copy(),
            obj_cut.rhs,
            oa_cut_relaxable,
            relaxable=False,
            cut_provenance=cut_provenance,
            source="objective",
            global_valid=True,
            supporting_point=obj_support,
            objective_id="objective",
        )


def _add_ecp_cuts(
    evaluator,
    x_master,
    n_vars,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    obj_is_linear,
    constraint_convex_mask,
    objective_is_convex,
    equality_relaxation=False,
    oa_cut_relaxable=None,
    cut_provenance: Optional[MIPNLPCutProvenance] = None,
):
    """Generate ECP cuts: OA cuts only for violated constraints at x_master."""
    from discopt._jax.cutting_planes import (
        generate_objective_oa_cut,
        separate_oa_cuts,
    )

    n_added = 0
    if evaluator.n_constraints > 0:
        cuts = separate_oa_cuts(
            evaluator,
            x_master,
            constraint_senses=constraint_senses,
            convex_mask=constraint_convex_mask,
        )
        constraint_ids = _constraint_ids_for_generated_oa_cuts(
            evaluator,
            x_master,
            constraint_senses,
            constraint_convex_mask,
            violated_only=True,
        )
        for constraint_id, cut in zip(constraint_ids, cuts):
            coeffs = cut.coeffs.copy()
            if np.linalg.norm(coeffs) < 1e-12:
                continue

            sense = cut.sense
            original_sense = sense
            if equality_relaxation and sense == "==":
                sense = "<="
            global_valid = _constraint_cut_global_valid(
                constraint_convex_mask,
                constraint_id,
                original_sense,
                equality_relaxation,
            )

            if sense == "<=":
                _append_master_cut(
                    oa_A_rows,
                    oa_b_rows,
                    coeffs,
                    cut.rhs,
                    oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                    source="ecp",
                    global_valid=global_valid,
                    supporting_point=x_master,
                    constraint_id=constraint_id,
                )
                n_added += 1
            elif sense == ">=":
                _append_master_cut(
                    oa_A_rows,
                    oa_b_rows,
                    -coeffs,
                    -cut.rhs,
                    oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                    source="ecp",
                    global_valid=global_valid,
                    supporting_point=x_master,
                    constraint_id=constraint_id,
                )
                n_added += 1
            elif sense == "==":
                _append_master_cut(
                    oa_A_rows,
                    oa_b_rows,
                    coeffs,
                    cut.rhs,
                    oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                    source="ecp",
                    global_valid=global_valid,
                    supporting_point=x_master,
                    constraint_id=constraint_id,
                )
                _append_master_cut(
                    oa_A_rows,
                    oa_b_rows,
                    -coeffs,
                    -cut.rhs,
                    oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                    source="ecp",
                    global_valid=global_valid,
                    supporting_point=x_master,
                    constraint_id=constraint_id,
                )
                n_added += 2

    if not obj_is_linear and objective_is_convex:
        n_master = n_vars + 1
        obj_value = float(evaluator.evaluate_objective(x_master))
        obj_support = np.concatenate([np.asarray(x_master, dtype=np.float64), [obj_value]])
        obj_cut = generate_objective_oa_cut(evaluator, x_master, n_master, z_index=n_vars)
        _append_master_cut(
            oa_A_rows,
            oa_b_rows,
            obj_cut.coeffs.copy(),
            obj_cut.rhs,
            oa_cut_relaxable,
            relaxable=False,
            cut_provenance=cut_provenance,
            source="objective",
            global_valid=True,
            supporting_point=obj_support,
            objective_id="objective",
        )
        n_added += 1

    return n_added


def _add_no_good_cut(
    x_master,
    binary_indices,
    oa_A_rows,
    oa_b_rows,
    n_vars,
    oa_cut_relaxable=None,
    cut_provenance: Optional[MIPNLPCutProvenance] = None,
):
    """Add a binary-assignment exclusion (no-good) cut.

    sum_{i: y_i*=1} (1-y_i) + sum_{i: y_i*=0} y_i >= 1
    Equivalently in <= form:
    sum_{y_i*=1} y_i - sum_{y_i*=0} y_i <= count(y_i*=1) - 1
    """
    if not binary_indices:
        return False

    coeffs = np.zeros(n_vars, dtype=np.float64)
    count_ones = 0
    for idx in binary_indices:
        val = _round_integral_to_bounds(x_master[idx], 0.0, 1.0)
        if val >= 0.5:
            coeffs[idx] = 1.0
            count_ones += 1
        else:
            coeffs[idx] = -1.0
    _append_master_cut(
        oa_A_rows,
        oa_b_rows,
        coeffs,
        float(count_ones - 1),
        oa_cut_relaxable,
        relaxable=False,
        cut_provenance=cut_provenance,
        source="integer",
        global_valid=True,
        supporting_point=x_master,
    )
    return True


def _add_feasibility_cuts(
    evaluator,
    x_feas,
    n_vars,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    constraint_convex_mask,
    oa_cut_relaxable=None,
    cut_provenance: Optional[MIPNLPCutProvenance] = None,
):
    """Add gradient-based feasibility cuts (Fletcher-Leyffer 1994).

    For each violated constraint g_k(x) <= 0 at x_feas:
        g_k(x_feas) + nabla g_k(x_feas)^T (x - x_feas) <= 0
    """
    from discopt._jax.cutting_planes import separate_oa_cuts

    if evaluator.n_constraints == 0:
        return

    cuts = separate_oa_cuts(
        evaluator,
        x_feas,
        constraint_senses=constraint_senses,
        convex_mask=constraint_convex_mask,
    )
    constraint_ids = _constraint_ids_for_generated_oa_cuts(
        evaluator,
        x_feas,
        constraint_senses,
        constraint_convex_mask,
        violated_only=True,
    )
    for constraint_id, cut in zip(constraint_ids, cuts):
        coeffs = cut.coeffs.copy()
        if np.linalg.norm(coeffs) < 1e-12:
            continue
        global_valid = _constraint_cut_global_valid(
            constraint_convex_mask,
            constraint_id,
            cut.sense,
            equality_relaxation=False,
        )
        # Feasibility cuts are gradient cuts, not hard integer exclusions, so
        # the shared OA slack may relax them to keep the heuristic master robust.
        if cut.sense == "<=":
            _append_master_cut(
                oa_A_rows,
                oa_b_rows,
                coeffs,
                cut.rhs,
                oa_cut_relaxable,
                cut_provenance=cut_provenance,
                source="feasibility",
                global_valid=global_valid,
                supporting_point=x_feas,
                constraint_id=constraint_id,
            )
        elif cut.sense == ">=":
            _append_master_cut(
                oa_A_rows,
                oa_b_rows,
                -coeffs,
                -cut.rhs,
                oa_cut_relaxable,
                cut_provenance=cut_provenance,
                source="feasibility",
                global_valid=global_valid,
                supporting_point=x_feas,
                constraint_id=constraint_id,
            )


# ── MILP Master Problem ──────────────────────────────────────


def _build_master_milp_data(
    linear_A_rows,
    linear_b_rows,
    linear_senses,
    oa_A_rows,
    oa_b_rows,
    n_vars,
    integrality,
    lb,
    ub,
    obj_coeffs,
    obj_is_linear,
    objective_bound_valid,
    add_slack=False,
    max_slack=1000.0,
    oa_penalty_factor=1000.0,
    oa_cut_relaxable=None,
    use_objective_epigraph=None,
) -> _MasterMILPData:
    """Build matrix data for an OA-style MILP master."""
    # ``use_objective_epigraph`` controls master layout when supplied; the
    # certification flag remains only as the compatibility fallback.
    if use_objective_epigraph is None:
        use_objective_epigraph = (not obj_is_linear) and objective_bound_valid
    if oa_cut_relaxable is not None and len(oa_cut_relaxable) != len(oa_A_rows):
        raise ValueError(
            "oa_cut_relaxable must match oa_A_rows length; "
            f"got {len(oa_cut_relaxable)} flags for {len(oa_A_rows)} cuts."
        )
    n_master = n_vars
    if use_objective_epigraph:
        n_master += 1  # epigraph variable eta
    slack_index = None
    if add_slack:
        # A single shared slack intentionally keeps the master compact. It is a
        # MindtPy-inspired heuristic simplification, not a per-cut slack model.
        slack_index = n_master
        n_master += 1

    # Build A_ub, b_ub from linear <= constraints + OA cuts
    A_ub_rows = []
    b_ub_vals = []

    for i, sense in enumerate(linear_senses):
        row = linear_A_rows[i]
        if use_objective_epigraph:
            row = np.append(row, 0.0)
        if add_slack:
            row = np.append(row, 0.0)
        if sense == "<=":
            A_ub_rows.append(row)
            b_ub_vals.append(linear_b_rows[i])
        elif sense == ">=":
            A_ub_rows.append(-row)
            b_ub_vals.append(-linear_b_rows[i])

    # OA cuts (all in <= form already)
    # Constraint cuts have length n_vars; objective cuts carry the eta column.
    for i in range(len(oa_A_rows)):
        row = np.asarray(oa_A_rows[i], dtype=np.float64)
        original_len = len(row)
        if use_objective_epigraph and len(row) == n_vars:
            row = np.append(row, 0.0)  # extend constraint cuts with 0 for eta
        if add_slack:
            # Relax only constraint OA/feasibility cuts. Objective epigraph cuts
            # and hard integer-exclusion cuts must remain unrelaxed.
            if oa_cut_relaxable is None:
                relax_cut = original_len == n_vars
            else:
                relax_cut = bool(oa_cut_relaxable[i])
            slack_coeff = -1.0 if relax_cut else 0.0
            row = np.append(row, slack_coeff)
        A_ub_rows.append(row)
        b_ub_vals.append(oa_b_rows[i])

    # Equality constraints from linear
    A_eq_rows = []
    b_eq_vals = []
    for i, sense in enumerate(linear_senses):
        if sense == "==":
            row = linear_A_rows[i]
            if use_objective_epigraph:
                row = np.append(row, 0.0)
            if add_slack:
                row = np.append(row, 0.0)
            A_eq_rows.append(row)
            b_eq_vals.append(linear_b_rows[i])

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_vals) if b_eq_vals else None

    # Objective
    c = np.zeros(n_master)
    if obj_is_linear:
        c_vec, _off = obj_coeffs
        c[:n_vars] = c_vec
    elif use_objective_epigraph:
        c[n_vars] = 1.0  # minimize eta
    if slack_index is not None:
        c[slack_index] = oa_penalty_factor

    # Bounds
    bounds_list = list(zip(lb.tolist(), ub.tolist()))
    if use_objective_epigraph:
        bounds_list.append((-1e20, 1e20))  # eta unbounded
    if slack_index is not None:
        bounds_list.append((0.0, max_slack))

    # Integrality
    int_vec = np.zeros(n_master, dtype=np.int32)
    int_vec[:n_vars] = integrality

    return _MasterMILPData(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds_list,
        integrality=int_vec,
        use_objective_epigraph=bool(use_objective_epigraph),
        slack_index=slack_index,
    )


def _master_solution_candidates(
    master_result,
    n_vars: int,
    *,
    solution_pool: bool,
    num_solution_iteration: int,
) -> list[np.ndarray]:
    """Return original-variable master candidates for one OA iteration."""
    if master_result.x is None:
        return []

    incumbent = np.asarray(master_result.x, dtype=np.float64).ravel()
    if not solution_pool:
        return [incumbent[:n_vars].copy()]

    raw_pool = list(master_result.solution_pool or [])
    if not raw_pool:
        raw_pool = [incumbent]
    elif not any(
        np.allclose(np.asarray(candidate, dtype=np.float64).ravel()[:n_vars], incumbent[:n_vars])
        for candidate in raw_pool
        if np.asarray(candidate, dtype=np.float64).ravel().size >= n_vars
    ):
        raw_pool.insert(0, incumbent)

    candidates: list[np.ndarray] = []
    for raw_candidate in raw_pool:
        arr = np.asarray(raw_candidate, dtype=np.float64).ravel()
        if arr.size < n_vars:
            continue
        x_candidate = arr[:n_vars].copy()
        if any(np.allclose(x_candidate, existing) for existing in candidates):
            continue
        candidates.append(x_candidate)
        if len(candidates) >= num_solution_iteration:
            break

    return candidates or [incumbent[:n_vars].copy()]


def _solve_master_milp(
    linear_A_rows,
    linear_b_rows,
    linear_senses,
    oa_A_rows,
    oa_b_rows,
    n_vars,
    integrality,
    lb,
    ub,
    obj_coeffs,
    obj_is_linear,
    objective_bound_valid,
    time_limit,
    gap_tolerance,
    add_slack=False,
    max_slack=1000.0,
    oa_penalty_factor=1000.0,
    oa_cut_relaxable=None,
    use_objective_epigraph=None,
    milp_solver="auto",
    solution_pool=False,
    num_solution_iteration=5,
):
    """Build and solve the master MILP."""
    try:
        if solution_pool:
            _require_solution_pool_backend(milp_solver)
            from discopt.solvers.gurobi import solve_milp
        else:
            from discopt.solvers.lp_backend import get_milp_solver

            solve_milp = get_milp_solver(backend=milp_solver)
    except ImportError as e:
        raise ImportError(
            "OA solver requires a MILP backend for the master. Install one of: "
            "pip install highspy  |  pip install pounce-solver  |  pip install gurobipy"
        ) from e

    master = _build_master_milp_data(
        linear_A_rows,
        linear_b_rows,
        linear_senses,
        oa_A_rows,
        oa_b_rows,
        n_vars,
        integrality,
        lb,
        ub,
        obj_coeffs,
        obj_is_linear,
        objective_bound_valid,
        add_slack=add_slack,
        max_slack=max_slack,
        oa_penalty_factor=oa_penalty_factor,
        oa_cut_relaxable=oa_cut_relaxable,
        use_objective_epigraph=use_objective_epigraph,
    )

    return solve_milp(
        c=master.c,
        A_ub=master.A_ub,
        b_ub=master.b_ub,
        A_eq=master.A_eq,
        b_eq=master.b_eq,
        bounds=master.bounds,
        integrality=master.integrality,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        **(
            {
                "options": {
                    "PoolSearchMode": 2,
                    "PoolSolutions": max(1, int(num_solution_iteration)),
                },
                "solution_pool": True,
                "num_solution_iteration": max(1, int(num_solution_iteration)),
            }
            if solution_pool
            else {}
        ),
    )


def _solve_regularized_master(
    decomp: _DecomposedProblem,
    oa_A_rows,
    oa_b_rows,
    *,
    add_regularization: str,
    target: np.ndarray,
    objective_level: float,
    time_limit: float,
    gap_tolerance: float,
    add_slack: bool = False,
    max_slack: float = 1000.0,
    oa_penalty_factor: float = 1000.0,
    oa_cut_relaxable=None,
    use_objective_epigraph: Optional[bool] = None,
    derivative_data: Optional[_DerivativeRegularizationData] = None,
    milp_solver: str = "auto",
) -> Optional[np.ndarray]:
    """Solve the ROA level-set master and return its original-variable point.

    The regularized master keeps the current linear/OA master constraints,
    adds a level constraint on the master objective estimate, and optimizes the
    selected regularization objective. ``level_L1``, ``level_L_infinity``, and
    ``grad_lag`` are MILPs; quadratic modes require a QP backend that supports
    integrality.
    """
    from discopt.solvers import SolveStatus

    if use_objective_epigraph is None:
        use_objective_epigraph = (not decomp.obj_is_linear) and decomp.oa_objective_is_convex
    if not decomp.obj_is_linear and not use_objective_epigraph:
        return None
    if oa_cut_relaxable is not None and len(oa_cut_relaxable) != len(oa_A_rows):
        raise ValueError(
            "oa_cut_relaxable must match oa_A_rows length; "
            f"got {len(oa_cut_relaxable)} flags for {len(oa_A_rows)} cuts."
        )

    n_vars = decomp.n_vars
    target = np.clip(np.asarray(target, dtype=np.float64), decomp.lb, decomp.ub)
    if add_regularization in _DERIVATIVE_REGULARIZATION_MODES and derivative_data is None:
        raise RuntimeError(
            f"OA {add_regularization} regularization requires Lagrangian derivative data."
        )
    eta_index = n_vars if use_objective_epigraph else None
    n_base = n_vars + (1 if use_objective_epigraph else 0)
    slack_index = None
    if add_slack:
        slack_index = n_base
        n_base += 1

    aux_start = None
    if add_regularization == "level_L1":
        aux_start = n_base
        n_master = n_base + n_vars
    elif add_regularization == "level_L_infinity":
        aux_start = n_base
        n_master = n_base + 1
    elif add_regularization in {"level_L2", "grad_lag", "hess_lag", "hess_only_lag", "sqp_lag"}:
        n_master = n_base
    else:  # pragma: no cover - guarded by _normalize_regularization
        raise ValueError(f"Unsupported regularization mode {add_regularization!r}.")

    a_ub_rows: list[np.ndarray] = []
    b_ub_vals: list[float] = []
    a_eq_rows: list[np.ndarray] = []
    b_eq_vals: list[float] = []

    def base_row(coeffs: np.ndarray) -> np.ndarray:
        row = np.zeros(n_master, dtype=np.float64)
        row[: len(coeffs)] = coeffs
        return row

    for row, rhs, sense in zip(
        decomp.linear_A_rows,
        decomp.linear_b_rows,
        decomp.linear_senses,
    ):
        master_row = base_row(np.asarray(row, dtype=np.float64))
        if sense == "<=":
            a_ub_rows.append(master_row)
            b_ub_vals.append(float(rhs))
        elif sense == ">=":
            a_ub_rows.append(-master_row)
            b_ub_vals.append(float(-rhs))
        elif sense == "==":
            a_eq_rows.append(master_row)
            b_eq_vals.append(float(rhs))

    for i, cut_row in enumerate(oa_A_rows):
        row = np.asarray(cut_row, dtype=np.float64)
        original_len = len(row)
        if use_objective_epigraph and original_len == n_vars:
            row = np.append(row, 0.0)
        master_row = base_row(row)
        if slack_index is not None:
            if oa_cut_relaxable is None:
                relax_cut = original_len == n_vars
            else:
                relax_cut = bool(oa_cut_relaxable[i])
            master_row[slack_index] = -1.0 if relax_cut else 0.0
        a_ub_rows.append(master_row)
        b_ub_vals.append(float(oa_b_rows[i]))

    level_row = np.zeros(n_master, dtype=np.float64)
    level_rhs = float(objective_level)
    if decomp.obj_is_linear and decomp.obj_coeffs is not None:
        c_vec, c_off = decomp.obj_coeffs
        level_row[:n_vars] = c_vec
        level_rhs -= float(c_off)
    elif eta_index is not None:
        level_row[eta_index] = 1.0
    else:
        return None
    a_ub_rows.append(level_row)
    b_ub_vals.append(level_rhs)

    if add_regularization == "level_L1":
        assert aux_start is not None
        for idx in range(n_vars):
            dev_idx = aux_start + idx
            row = np.zeros(n_master, dtype=np.float64)
            row[idx] = 1.0
            row[dev_idx] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(target[idx]))

            row = np.zeros(n_master, dtype=np.float64)
            row[idx] = -1.0
            row[dev_idx] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(-target[idx]))
    elif add_regularization == "level_L_infinity":
        assert aux_start is not None
        dev_idx = aux_start
        for idx in range(n_vars):
            row = np.zeros(n_master, dtype=np.float64)
            row[idx] = 1.0
            row[dev_idx] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(target[idx]))

            row = np.zeros(n_master, dtype=np.float64)
            row[idx] = -1.0
            row[dev_idx] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(-target[idx]))

    bounds = list(zip(decomp.lb.tolist(), decomp.ub.tolist()))
    if use_objective_epigraph:
        bounds.append((-1e20, 1e20))
    if slack_index is not None:
        bounds.append((0.0, max_slack))
    if add_regularization == "level_L1":
        bounds.extend((0.0, 1e20) for _ in range(n_vars))
    elif add_regularization == "level_L_infinity":
        bounds.append((0.0, 1e20))

    integrality = np.zeros(n_master, dtype=np.int32)
    integrality[:n_vars] = decomp.integrality
    A_ub = np.asarray(a_ub_rows, dtype=np.float64) if a_ub_rows else None
    b_ub = np.asarray(b_ub_vals, dtype=np.float64) if b_ub_vals else None
    A_eq = np.asarray(a_eq_rows, dtype=np.float64) if a_eq_rows else None
    b_eq = np.asarray(b_eq_vals, dtype=np.float64) if b_eq_vals else None

    if add_regularization in _QP_REGULARIZATION_MODES:
        try:
            from discopt.solvers.lp_backend import get_qp_solver

            solve_qp = get_qp_solver()
            Q = np.zeros((n_master, n_master), dtype=np.float64)
            c = np.zeros(n_master, dtype=np.float64)
            if add_regularization == "level_L2":
                for idx in range(n_vars):
                    Q[idx, idx] = 2.0
                    c[idx] = -2.0 * target[idx]
            elif add_regularization in {"hess_lag", "hess_only_lag"}:
                assert derivative_data is not None  # guarded above
                assert derivative_data.hessian is not None  # guarded by data builder
                hess = derivative_data.hessian
                ref = derivative_data.target
                Q[:n_vars, :n_vars] = hess
                c[:n_vars] = -hess @ ref
                if add_regularization == "hess_lag":
                    c[:n_vars] += derivative_data.gradient
            elif add_regularization == "sqp_lag":
                assert derivative_data is not None  # guarded above
                # First-slice MindtPy-compatible SQP regularization: keep a
                # fixed unit proximal weight until a public tuning option is
                # justified by solver behavior across more benchmark cases.
                rho = 1.0
                for idx in range(n_vars):
                    Q[idx, idx] = 2.0 * rho
                c[:n_vars] = derivative_data.gradient - 2.0 * rho * derivative_data.target
            if slack_index is not None:
                c[slack_index] = oa_penalty_factor
            result = solve_qp(
                Q=Q,
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                integrality=integrality,
                time_limit=max(float(time_limit), 0.0),
                gap_tolerance=gap_tolerance,
            )
        except ImportError as exc:
            raise _qp_regularization_backend_error(add_regularization) from exc
        except ValueError as exc:
            raise _qp_regularization_solve_error(add_regularization) from exc
    else:
        from discopt.solvers.lp_backend import get_milp_solver

        solve_milp = get_milp_solver(backend=milp_solver)
        c = np.zeros(n_master, dtype=np.float64)
        if add_regularization == "level_L1":
            assert aux_start is not None
            c[aux_start : aux_start + n_vars] = 1.0
        elif add_regularization == "level_L_infinity":
            assert aux_start is not None
            c[aux_start] = 1.0
        elif add_regularization == "grad_lag":
            assert derivative_data is not None  # guarded above
            c[:n_vars] = derivative_data.gradient
        if slack_index is not None:
            c[slack_index] = oa_penalty_factor
        result = solve_milp(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            integrality=integrality,
            time_limit=max(float(time_limit), 0.0),
            gap_tolerance=gap_tolerance,
        )

    if result.status not in (
        SolveStatus.OPTIMAL,
        SolveStatus.ITERATION_LIMIT,
        SolveStatus.TIME_LIMIT,
    ):
        return None
    if result.x is None:
        return None
    return np.asarray(result.x[:n_vars], dtype=np.float64)


# ── Result Construction ───────────────────────────────────────


def _build_x_dict(x_flat: np.ndarray, model: Model) -> dict:
    """Convert flat solution vector to {var_name: value} dict."""
    result = {}
    offset = 0
    for v in model._variables:
        result[v.name] = x_flat[offset : offset + v.size].reshape(v.shape)
        offset += v.size
    return result


def _compute_gap(lb: float, ub: float) -> float:
    if ub >= 1e19 or lb <= -1e19:
        return 1.0
    abs_gap = max(0.0, ub - lb)
    if abs_gap <= 1e-9:
        return 0.0
    denom = max(abs(ub), abs(lb), 1.0)
    return abs_gap / denom


def solve_feasibility_pump(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "ipm",
    initial_point: Optional[np.ndarray] = None,
    feasibility_norm: str = "L_infinity",
    add_no_good_cuts: bool = True,
    fp_iteration_limit: Optional[int] = None,
    fp_cutoffdecr: float = 0.0,
    fp_projcuts: Optional[bool] = None,
    fp_transfercuts: bool = False,
    fp_projzerotol: float = 0.0,
    fp_mipgap: Optional[float] = None,
    fp_discrete_only: bool = True,
    fp_main_norm: Optional[str] = None,
    fp_norm_constraint: bool = False,
    fp_norm_constraint_coef: float = 1.0,
    **kwargs,
) -> SolveResult:
    """Run the MIP-NLP feasibility pump as a standalone heuristic method."""
    if kwargs:
        raise ValueError(
            "Unsupported feasibility-pump option(s): "
            + ", ".join(sorted(kwargs))
            + ". Supported FP options are: "
            + ", ".join(sorted(FP_OPTION_KEYS))
            + ", add_no_good_cuts, feasibility_norm."
        )
    t_start = time.perf_counter()
    fp_config = _normalize_fp_config(
        feasibility_norm=feasibility_norm,
        add_no_good_cuts=bool(add_no_good_cuts),
        fp_iteration_limit=fp_iteration_limit,
        fp_cutoffdecr=fp_cutoffdecr,
        fp_projcuts=fp_projcuts,
        fp_transfercuts=fp_transfercuts,
        fp_projzerotol=fp_projzerotol,
        fp_mipgap=fp_mipgap,
        fp_discrete_only=fp_discrete_only,
        fp_main_norm=fp_main_norm,
        fp_norm_constraint=fp_norm_constraint,
        fp_norm_constraint_coef=fp_norm_constraint_coef,
    )
    feasibility_norm = fp_config.feasibility_norm
    decomp = _decompose_model(model)
    evaluator = decomp.evaluator
    obj_sign = (
        -1.0
        if (model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE)
        else 1.0
    )

    if len(decomp.int_indices) == 0:
        x_sol, obj = _solve_nlp_relaxation(
            evaluator,
            decomp.lb,
            decomp.ub,
            nlp_solver,
            initial_point=initial_point,
        )
        wall_time = time.perf_counter() - t_start
        if x_sol is not None:
            return SolveResult(
                status="feasible",
                objective=obj_sign * obj,
                bound=None,
                gap=None,
                x=_build_x_dict(x_sol, model),
                wall_time=wall_time,
                gap_certified=False,
            )
        return SolveResult(
            status="no_feasible_point",
            objective=None,
            bound=None,
            gap=None,
            x={},
            wall_time=wall_time,
            gap_certified=False,
        )

    fp = _run_feasibility_pump(
        model,
        decomp,
        nlp_solver=nlp_solver,
        initial_point=initial_point,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        max_iterations=_fp_iteration_count(max_iterations, fp_config.iteration_limit),
        feasibility_norm=feasibility_norm,
        add_no_good_cuts=fp_config.add_no_good_cuts,
        fp_main_norm=fp_config.main_norm,
        fp_mipgap=fp_config.mipgap,
        fp_discrete_only=fp_config.discrete_only,
        fp_projzerotol=fp_config.projzerotol,
    )
    wall_time = time.perf_counter() - t_start
    if fp.best_x is not None and fp.best_obj is not None:
        return SolveResult(
            status="feasible",
            objective=obj_sign * fp.best_obj,
            bound=None,
            gap=None,
            x=_build_x_dict(fp.best_x, model),
            wall_time=wall_time,
            mip_count=fp.mip_count,
            gap_certified=False,
        )
    return SolveResult(
        status="no_feasible_point",
        objective=None,
        bound=None,
        gap=None,
        x={},
        wall_time=wall_time,
        mip_count=fp.mip_count,
        gap_certified=False,
    )


def _require_lp_nlp_bb_gurobi_backend(milp_solver: str) -> None:
    if not isinstance(milp_solver, str) or milp_solver.strip().lower() != "gurobi":
        raise RuntimeError(
            "mip_nlp_method='lp_nlp_bb' requires milp_solver='gurobi' because "
            "LP/NLP branch-and-bound uses single-tree lazy constraint callbacks. "
            "Backends 'auto', 'highs', 'pounce', and 'simplex' do not expose the "
            "required persistent lazy-cut capability."
        )


def _format_lazy_master_cut(
    row,
    *,
    n_vars: int,
    master: _MasterMILPData,
    relaxable: bool,
) -> np.ndarray:
    """Extend an OA cut row to the active master layout for Gurobi cbLazy."""
    cut = np.asarray(row, dtype=np.float64).ravel()
    if master.use_objective_epigraph and len(cut) == n_vars:
        cut = np.append(cut, 0.0)
    if master.slack_index is not None:
        cut = np.append(cut, -1.0 if relaxable else 0.0)
    if len(cut) != len(master.c):
        raise ValueError(
            f"lazy OA cut has {len(cut)} coefficients but master has {len(master.c)} variables"
        )
    return cut


def solve_lp_nlp_bb(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "ipm",
    init_strategy: str = "rNLP",
    initial_point: Optional[np.ndarray] = None,
    equality_relaxation: bool = False,
    feasibility_cuts: bool = True,
    heuristic_nonconvex: bool = False,
    add_slack: bool = False,
    max_slack: float = 1000.0,
    oa_penalty_factor: float = 1000.0,
    add_no_good_cuts: bool = False,
    feasibility_norm: str = "L_infinity",
    milp_solver: str = "gurobi",
    **kwargs,
) -> SolveResult:
    """Solve a convex MINLP with the LP/NLP branch-and-bound variant.

    This is a single-tree OA method: the MILP master is solved once, and each
    integer incumbent triggers a fixed-integer NLP solve inside a Gurobi lazy
    constraint callback. Lazy rows are generated through the same OA and
    feasibility-cut helpers used by the multi-tree OA method. ``max_iterations``
    only caps the optional feasibility-pump initializer on this path; the main
    single-tree search is delegated to Gurobi and is controlled by
    ``time_limit`` and ``gap_tolerance``.
    """
    if kwargs:
        raise ValueError(
            "Unsupported LP/NLP BB option(s): "
            + ", ".join(sorted(kwargs))
            + ". Supported options are: add_no_good_cuts, add_slack, equality_relaxation, "
            "feasibility_cuts, feasibility_norm, heuristic_nonconvex, init_strategy, "
            "max_slack, milp_solver, oa_penalty_factor."
        )
    _require_lp_nlp_bb_gurobi_backend(milp_solver)
    t_start = time.perf_counter()
    init_strategy = _normalize_init_strategy(init_strategy)
    feasibility_norm = _normalize_feasibility_norm(feasibility_norm)
    fp_config = _normalize_fp_config(
        feasibility_norm=feasibility_norm,
        add_no_good_cuts=True,
    )
    max_slack = _normalize_positive_float("max_slack", max_slack)
    oa_penalty_factor = _normalize_positive_float("oa_penalty_factor", oa_penalty_factor)
    heuristic_nonconvex = bool(heuristic_nonconvex)
    if heuristic_nonconvex:
        equality_relaxation = True
        add_slack = True
    add_slack = bool(add_slack)
    add_no_good_cuts = bool(add_no_good_cuts)

    decomp = _decompose_model(model)
    evaluator = decomp.evaluator
    n_vars = decomp.n_vars
    n_cons = decomp.n_cons
    obj_sign = (
        -1.0
        if (model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE)
        else 1.0
    )
    if decomp.oa_constraint_mask is not None and not all(decomp.oa_constraint_mask):
        logger.warning(
            "LP/NLP BB: generating OA cuts only for %d of %d constraints classified convex",
            sum(1 for is_convex in decomp.oa_constraint_mask if is_convex),
            len(decomp.oa_constraint_mask),
        )
    if not decomp.obj_is_linear and not decomp.oa_objective_is_convex:
        logger.warning(
            "LP/NLP BB: nonlinear objective is not convex in the optimization sense; "
            "disabling certified bound/gap reporting and skipping objective OA cuts"
        )
    master_bound_valid = decomp.master_bound_valid and not heuristic_nonconvex

    if len(decomp.int_indices) == 0:
        x_sol, obj = _solve_nlp_relaxation(
            evaluator,
            decomp.lb,
            decomp.ub,
            nlp_solver,
            initial_point=initial_point,
        )
        wall_time = time.perf_counter() - t_start
        if x_sol is not None:
            return SolveResult(
                status="optimal",
                objective=obj_sign * obj,
                bound=obj_sign * obj,
                gap=0.0,
                x=_build_x_dict(x_sol, model),
                wall_time=wall_time,
                mip_count=0,
            )
        return SolveResult(
            status="infeasible",
            objective=None,
            bound=None,
            gap=None,
            x={},
            wall_time=wall_time,
            mip_count=0,
        )

    oa_A_rows: list[np.ndarray] = []
    oa_b_rows: list[float] = []
    oa_cut_relaxable: list[bool] = []
    incumbent: Optional[np.ndarray] = None
    incumbent_obj: Optional[float] = None
    nlp_subproblem_count = 0

    def accept_incumbent(x: np.ndarray, obj: float) -> None:
        nonlocal incumbent, incumbent_obj
        if incumbent_obj is None or obj < incumbent_obj:
            incumbent = np.asarray(x, dtype=np.float64).copy()
            incumbent_obj = float(obj)

    def add_oa_cuts_at(x_cut: np.ndarray) -> None:
        _add_oa_cuts(
            evaluator,
            x_cut,
            n_vars,
            n_cons,
            decomp.constraint_senses,
            oa_A_rows,
            oa_b_rows,
            decomp.obj_is_linear,
            decomp.oa_constraint_mask,
            decomp.oa_objective_is_convex,
            equality_relaxation=equality_relaxation,
            oa_cut_relaxable=oa_cut_relaxable,
        )

    if init_strategy == "rNLP":
        x_relax, obj_relax = _solve_nlp_relaxation(
            evaluator,
            decomp.lb,
            decomp.ub,
            nlp_solver,
            initial_point=initial_point,
        )
        if x_relax is not None:
            add_oa_cuts_at(x_relax)
            if _is_integer_feasible(decomp, x_relax) and obj_relax is not None:
                accept_incumbent(x_relax, obj_relax)
        else:
            add_oa_cuts_at(_default_nlp_start(decomp.lb, decomp.ub))
    elif init_strategy == "fp":
        fp_iterations = _fp_iteration_count(
            max_iterations,
            fp_config.iteration_limit,
            default_cap=10,
        )
        fp_result = _run_feasibility_pump(
            model,
            decomp,
            nlp_solver=nlp_solver,
            initial_point=initial_point,
            time_limit=max(time_limit - (time.perf_counter() - t_start), 0.0),
            gap_tolerance=gap_tolerance,
            max_iterations=fp_iterations,
            feasibility_norm=fp_config.feasibility_norm,
            add_no_good_cuts=fp_config.add_no_good_cuts,
            fp_main_norm=fp_config.main_norm,
            fp_mipgap=fp_config.mipgap,
            fp_discrete_only=fp_config.discrete_only,
            fp_projzerotol=fp_config.projzerotol,
            milp_solver=milp_solver,
        )
        x_cut = fp_result.best_x if fp_result.best_x is not None else fp_result.best_near_x
        add_oa_cuts_at(x_cut if x_cut is not None else _default_nlp_start(decomp.lb, decomp.ub))
        if fp_result.best_x is not None and fp_result.best_obj is not None:
            accept_incumbent(fp_result.best_x, fp_result.best_obj)
    else:
        x_seed = _build_initial_strategy_point(decomp, init_strategy, initial_point)
        x_init, obj_init = _solve_nlp_subproblem(
            evaluator,
            decomp.lb,
            decomp.ub,
            decomp.int_indices,
            x_seed,
            nlp_solver,
            initial_point=x_seed,
        )
        add_oa_cuts_at(x_init if x_init is not None else x_seed)
        if x_init is not None and obj_init is not None:
            accept_incumbent(x_init, obj_init)

    elapsed = time.perf_counter() - t_start
    remaining = max(float(time_limit) - elapsed, 0.0)
    master = _build_master_milp_data(
        decomp.linear_A_rows,
        decomp.linear_b_rows,
        decomp.linear_senses,
        oa_A_rows,
        oa_b_rows,
        n_vars,
        decomp.integrality,
        decomp.lb,
        decomp.ub,
        decomp.obj_coeffs,
        decomp.obj_is_linear,
        master_bound_valid,
        add_slack=add_slack,
        max_slack=max_slack,
        oa_penalty_factor=oa_penalty_factor,
        oa_cut_relaxable=oa_cut_relaxable,
        use_objective_epigraph=(not decomp.obj_is_linear and decomp.oa_objective_is_convex),
    )

    def collect_new_lazy_cuts(start: int, master_x: np.ndarray) -> list[tuple[np.ndarray, float]]:
        rows: list[tuple[np.ndarray, float]] = []
        for idx in range(start, len(oa_A_rows)):
            relaxable = bool(oa_cut_relaxable[idx]) if idx < len(oa_cut_relaxable) else True
            row = _format_lazy_master_cut(
                oa_A_rows[idx],
                n_vars=n_vars,
                master=master,
                relaxable=relaxable,
            )
            rhs = float(oa_b_rows[idx])
            if float(np.dot(row, master_x)) > rhs + 1e-6:
                rows.append((row, rhs))
        return rows

    def lazy_callback(master_x: np.ndarray) -> list[tuple[np.ndarray, float]]:
        nonlocal nlp_subproblem_count
        x_master = np.asarray(master_x[:n_vars], dtype=np.float64)
        start = len(oa_A_rows)
        nlp_subproblem_count += 1
        x_nlp, obj_nlp = _solve_nlp_subproblem(
            evaluator,
            decomp.lb,
            decomp.ub,
            decomp.int_indices,
            x_master,
            nlp_solver,
            initial_point=x_master,
        )
        if x_nlp is not None:
            if obj_nlp is not None:
                accept_incumbent(x_nlp, obj_nlp)
            add_oa_cuts_at(x_nlp)
        else:
            if feasibility_cuts:
                x_feas = _solve_feasibility_subproblem(
                    evaluator,
                    decomp.lb,
                    decomp.ub,
                    decomp.int_indices,
                    x_master,
                    nlp_solver,
                    feasibility_norm,
                )
                if x_feas is not None:
                    _add_feasibility_cuts(
                        evaluator,
                        x_feas,
                        n_vars,
                        decomp.constraint_senses,
                        oa_A_rows,
                        oa_b_rows,
                        decomp.oa_constraint_mask,
                        oa_cut_relaxable=oa_cut_relaxable,
                    )
            if add_no_good_cuts and not decomp.general_integer_indices:
                _add_no_good_cut(
                    x_master,
                    decomp.binary_indices,
                    oa_A_rows,
                    oa_b_rows,
                    n_vars,
                    oa_cut_relaxable=oa_cut_relaxable,
                )
            add_oa_cuts_at(x_master)

        return collect_new_lazy_cuts(start, np.asarray(master_x, dtype=np.float64))

    from discopt.solvers import SolveStatus
    from discopt.solvers.gurobi import solve_milp_with_lazy_cuts

    master_result = solve_milp_with_lazy_cuts(
        c=master.c,
        A_ub=master.A_ub,
        b_ub=master.b_ub,
        A_eq=master.A_eq,
        b_eq=master.b_eq,
        bounds=master.bounds,
        integrality=master.integrality,
        time_limit=remaining,
        gap_tolerance=gap_tolerance,
        lazy_callback=lazy_callback,
    )
    wall_time = time.perf_counter() - t_start

    bound = None
    if master_bound_valid and master_result.bound is not None:
        bound = float(master_result.bound)
        if decomp.obj_is_linear and decomp.obj_coeffs is not None:
            bound += float(decomp.obj_coeffs[1])
    gap = (
        _compute_gap(bound, incumbent_obj)
        if bound is not None and incumbent_obj is not None
        else None
    )
    status = "feasible"
    if master_result.status == SolveStatus.INFEASIBLE:
        status = "infeasible"
    elif master_result.status == SolveStatus.TIME_LIMIT:
        status = "time_limit" if incumbent is None else "feasible"
    elif master_result.status == SolveStatus.ITERATION_LIMIT:
        status = "iteration_limit" if incumbent is None else "feasible"
    elif master_result.status == SolveStatus.OPTIMAL and incumbent is not None:
        status = "optimal" if gap is not None and gap <= gap_tolerance else "feasible"
    elif incumbent is None:
        status = "no_feasible_point"

    if incumbent is not None and incumbent_obj is not None:
        return SolveResult(
            status=status,
            objective=obj_sign * incumbent_obj,
            bound=(obj_sign * bound if bound is not None else None),
            gap=gap,
            x=_build_x_dict(incumbent, model),
            wall_time=wall_time,
            node_count=master_result.node_count,
            mip_count=1,
            subnlp_calls=nlp_subproblem_count,
            gap_certified=master_bound_valid,
        )

    return SolveResult(
        status=status,
        objective=None,
        bound=(obj_sign * bound if bound is not None else None),
        gap=None,
        x={},
        wall_time=wall_time,
        node_count=master_result.node_count,
        mip_count=1,
        subnlp_calls=nlp_subproblem_count,
        gap_certified=master_bound_valid,
    )


def solve_goa(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "ipm",
    initial_point: Optional[np.ndarray] = None,
    add_no_good_cuts: bool = True,
    **amp_options,
) -> SolveResult:
    """Solve a MINLP through the global OA/relaxation stack.

    Convexity-certified MINLPs are handed to the OA algorithm, whose master
    lower bounds are globally valid in that case. Other models use the
    AMP/McCormick global-relaxation path. The MIP-NLP feasibility pump, with
    no-good cuts enabled by default, is only an incumbent-start heuristic for
    the nonconvex AMP path, so its exclusions never taint certified bounds.
    AMP-only options are honored on the nonconvex path; if supplied for a model
    that certifies convex and is handed to OA, they are ignored with a warning.
    """
    t_start = time.perf_counter()
    provided_option_keys = frozenset(amp_options)

    rel_gap = amp_options.pop("rel_gap", gap_tolerance)
    max_iter = amp_options.pop("max_iter", max_iterations)
    init_strategy = _normalize_init_strategy(amp_options.pop("init_strategy", "fp"))
    feasibility_norm = _normalize_feasibility_norm(
        amp_options.pop("feasibility_norm", "L_infinity")
    )
    fp_kwargs = {key: amp_options.pop(key) for key in FP_OPTION_KEYS if key in amp_options}
    fp_config = _normalize_fp_config(
        feasibility_norm=feasibility_norm,
        add_no_good_cuts=bool(add_no_good_cuts),
        fp_iteration_limit=fp_kwargs.get("fp_iteration_limit"),
        fp_cutoffdecr=fp_kwargs.get("fp_cutoffdecr", 0.0),
        fp_projcuts=fp_kwargs.get("fp_projcuts"),
        fp_transfercuts=fp_kwargs.get("fp_transfercuts", False),
        fp_projzerotol=fp_kwargs.get("fp_projzerotol", 0.0),
        fp_mipgap=fp_kwargs.get("fp_mipgap"),
        fp_discrete_only=fp_kwargs.get("fp_discrete_only", True),
        fp_main_norm=fp_kwargs.get("fp_main_norm"),
        fp_norm_constraint=fp_kwargs.get("fp_norm_constraint", False),
        fp_norm_constraint_coef=fp_kwargs.get("fp_norm_constraint_coef", 1.0),
    )
    amp_kwargs = dict(GOA_AMP_OPTION_DEFAULTS)
    for key in GOA_AMP_OPTION_DEFAULTS:
        if key in amp_options:
            amp_kwargs[key] = amp_options.pop(key)
    use_start_as_incumbent = bool(amp_kwargs["use_start_as_incumbent"])
    if amp_options:
        raise ValueError(
            "Unsupported GOA option(s): "
            + ", ".join(sorted(amp_options))
            + ". Pass AMP/global-relaxation options supported by solve_goa."
        )

    from discopt._jax.convexity import classify_oa_cut_convexity

    oa_convexity = classify_oa_cut_convexity(model)
    if oa_convexity.objective_is_convex and all(oa_convexity.constraint_mask):
        ignored_amp_options = sorted(provided_option_keys.intersection(GOA_AMP_ONLY_OPTION_KEYS))
        if ignored_amp_options:
            warnings.warn(
                "GOA routed a convexity-certified model to OA; AMP-only GOA "
                "option(s) are ignored on this path: " + ", ".join(ignored_amp_options),
                UserWarning,
                stacklevel=2,
            )
        elapsed = time.perf_counter() - t_start
        remaining_time = max(0.0, float(time_limit) - elapsed)
        result = solve_oa(
            model,
            time_limit=remaining_time,
            gap_tolerance=rel_gap,
            max_iterations=max_iter,
            nlp_solver=nlp_solver,
            init_strategy=init_strategy,
            initial_point=initial_point,
            add_no_good_cuts=bool(add_no_good_cuts),
            feasibility_norm=feasibility_norm,
            **fp_kwargs,
        )
        result.wall_time += elapsed
        return result

    goa_initial_point = initial_point
    pre_amp_mip_count = 0

    decomp: Optional[_DecomposedProblem] = None
    if init_strategy in {"fp", "initial_binary", "max_binary"}:
        decomp = _decompose_model(model)

    if init_strategy == "fp" and decomp is not None and decomp.int_indices:
        elapsed = time.perf_counter() - t_start
        remaining = max(0.0, float(time_limit) - elapsed)
        if np.isfinite(remaining) and np.isfinite(time_limit):
            pump_budget = min(remaining, max(0.0, 0.1 * float(time_limit)), 10.0)
        else:
            pump_budget = 10.0
        fp_iterations = _fp_iteration_count(
            max_iterations,
            fp_config.iteration_limit,
            default_cap=10,
        )
        if pump_budget > 0.0:
            fp_result = _run_feasibility_pump(
                model,
                decomp,
                nlp_solver=nlp_solver,
                initial_point=initial_point,
                time_limit=pump_budget,
                gap_tolerance=gap_tolerance,
                max_iterations=fp_iterations,
                feasibility_norm=fp_config.feasibility_norm,
                add_no_good_cuts=fp_config.add_no_good_cuts,
                fp_main_norm=fp_config.main_norm,
                fp_mipgap=fp_config.mipgap,
                fp_discrete_only=fp_config.discrete_only,
                fp_projzerotol=fp_config.projzerotol,
            )
            pre_amp_mip_count += fp_result.mip_count
            if fp_result.best_x is not None:
                goa_initial_point = fp_result.best_x
                use_start_as_incumbent = True
            elif fp_result.best_near_x is not None:
                goa_initial_point = fp_result.best_near_x
    elif init_strategy in {"initial_binary", "max_binary"} and decomp is not None:
        goa_initial_point = _build_initial_strategy_point(decomp, init_strategy, initial_point)

    elapsed = time.perf_counter() - t_start
    remaining_time = max(0.0, float(time_limit) - elapsed)
    if remaining_time <= 0.0:
        if goa_initial_point is not None and decomp is not None:
            candidate = np.asarray(goa_initial_point, dtype=np.float64)
            if _is_integer_feasible(decomp, candidate) and _is_primal_feasible(
                decomp.evaluator, candidate
            ):
                obj = float(decomp.evaluator.evaluate_objective(candidate))
                obj_sign = (
                    -1.0
                    if (
                        model._objective is not None
                        and model._objective.sense == ObjectiveSense.MAXIMIZE
                    )
                    else 1.0
                )
                return SolveResult(
                    status="feasible",
                    objective=obj_sign * obj,
                    bound=None,
                    gap=None,
                    x=_build_x_dict(candidate, model),
                    wall_time=elapsed,
                    mip_count=pre_amp_mip_count,
                    gap_certified=False,
                )
        return SolveResult(
            status="time_limit",
            objective=None,
            bound=None,
            gap=None,
            x=None,
            wall_time=elapsed,
            mip_count=pre_amp_mip_count,
            gap_certified=False,
        )

    from discopt.solvers.amp import solve_amp

    amp_kwargs["use_start_as_incumbent"] = use_start_as_incumbent
    result = solve_amp(
        model,
        rel_gap=rel_gap,
        time_limit=remaining_time,
        max_iter=max_iter,
        nlp_solver=nlp_solver,
        initial_point=goa_initial_point,
        **amp_kwargs,
    )
    result.wall_time += elapsed
    result.mip_count += pre_amp_mip_count
    return result


# ── Main Algorithm ────────────────────────────────────────────


def solve_oa(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "ipm",
    init_strategy: str = "rNLP",
    initial_point: Optional[np.ndarray] = None,
    equality_relaxation: bool = False,
    ecp_mode: bool = False,
    feasibility_cuts: bool = True,
    heuristic_nonconvex: bool = False,
    add_slack: bool = False,
    max_slack: float = 1000.0,
    oa_penalty_factor: float = 1000.0,
    add_no_good_cuts: bool = False,
    feasibility_norm: str = "L_infinity",
    fp_iteration_limit: Optional[int] = None,
    fp_cutoffdecr: float = 0.0,
    fp_projcuts: Optional[bool] = None,
    fp_transfercuts: bool = False,
    fp_projzerotol: float = 0.0,
    fp_mipgap: Optional[float] = None,
    fp_discrete_only: bool = True,
    fp_main_norm: Optional[str] = None,
    fp_norm_constraint: bool = False,
    fp_norm_constraint_coef: float = 1.0,
    add_regularization: Optional[str] = None,
    level_coef: float = 0.5,
    stalling_limit: Optional[int] = None,
    cycling_check: bool = False,
    milp_solver: str = "auto",
    solution_pool: bool = False,
    num_solution_iteration: int = 5,
    mip_nlp_profile: str = "default",
    mip_nlp_shot_config: Optional[MIPNLPShotConfig] = None,
    **kwargs,
) -> SolveResult:
    """Solve a MINLP via Outer Approximation.

    Decomposes the problem into alternating NLP subproblems (continuous
    optimization with fixed integers) and MILP master problems (linear
    relaxation with accumulated OA cuts).

    Parameters
    ----------
    model : Model
        MINLP model with continuous, binary, and/or integer variables.
    time_limit : float
        Wall-clock time limit in seconds.
    gap_tolerance : float
        Relative optimality gap for convergence.
    max_iterations : int
        Maximum OA iterations.
    nlp_solver : str
        NLP backend: ``"ipm"``, ``"ipopt"``, ``"pounce"``.
    init_strategy : {"rNLP", "initial_binary", "max_binary", "fp"}
        Initialization strategy for the first master cuts and fixed-integer
        NLP seed. ``"rNLP"`` solves the continuous relaxation and generates
        cuts at that point. ``"initial_binary"`` rounds and clamps discrete
        variables from ``initial_point`` when supplied, otherwise from the
        deterministic midpoint start. ``"max_binary"`` starts binary variables
        at their largest feasible values; general integers use their largest
        practical finite upper-bound value, or the rounded clipped midpoint
        when no practical finite upper bound exists. ``"fp"`` runs a bounded
        feasibility pump and generates cuts at its best feasible or near-feasible
        point.
    initial_point : numpy.ndarray, optional
        Flat model start produced from ``Model.solve(initial_solution=...)``.
        Used to warm-start the continuous relaxation for ``init_strategy="rNLP"``,
        by ``init_strategy="initial_binary"``, and as the continuous part of
        ``"max_binary"``.
    equality_relaxation : bool
        Relax nonlinear equalities to inequalities in OA cuts
        (Viswanathan & Grossmann 1990). Helps when nonlinear equalities
        cause the MILP master to become infeasible. This is a robustness
        heuristic; nonlinear equalities do not satisfy the convex MINLP OA
        guarantee unless they are affine.
    ecp_mode : bool
        Extended Cutting Plane mode (Westerlund & Pettersson 1995):
        skip NLP subproblems entirely, only add cuts at MILP master
        solutions for violated constraints. Simpler but slower convergence.
    feasibility_cuts : bool
        Use gradient-based feasibility cuts (Fletcher & Leyffer 1994)
        when the NLP subproblem is infeasible. Stronger than no-good cuts.
    heuristic_nonconvex : bool
        Enable MindtPy-style heuristic handling for nonconvex cases. This turns
        on equality relaxation and slack handling and suppresses certified
        bound/gap reporting.
    add_slack : bool
        Relax OA constraint cuts with one nonnegative master slack variable.
    max_slack : float
        Upper bound for the OA master slack variable.
    oa_penalty_factor : float
        Positive objective penalty applied to the OA master slack variable.
    add_no_good_cuts : bool
        Add integer-exclusion cuts after infeasible fixed-integer NLP solves.
    feasibility_norm : {"L1", "L2", "L_infinity"}
        Violation norm minimized by the feasibility subproblem heuristic.
    fp_iteration_limit : int, optional
        Iteration cap for the feasibility-pump initializer. When omitted,
        ``init_strategy="fp"`` keeps the legacy cap of ``min(max_iterations, 10)``.
    fp_main_norm : {"L1", "L2", "L_infinity"}, optional
        Distance norm used by the FP projection MILP. When omitted, this follows
        ``feasibility_norm``. ``L2`` uses the current MILP-compatible L1
        projection surrogate while the feasibility subproblem still scores
        nonlinear violation with squared L2 merit.
    fp_discrete_only : bool
        When true, the FP projection distance is computed only on discrete
        variables. When false, continuous-variable deviations are penalized too.
    fp_projcuts : bool, optional
        Explicit control for discopt's FP projection-MILP path with binary
        no-good cuts. When false, FP falls back to direct integer rounding.
        When omitted, FP initialization enables this path by default.
    fp_projzerotol : float
        Projection target entries with absolute value at or below this tolerance
        are treated as zero when zero lies within that variable's bounds.
    fp_mipgap : float, optional
        Gap tolerance for FP projection MILPs. Defaults to ``gap_tolerance``.
    fp_cutoffdecr, fp_transfercuts, fp_norm_constraint, fp_norm_constraint_coef
        MindtPy FP controls that are explicitly unsupported in discopt's current
        FP implementation unless set to their no-op defaults. Non-default values
        raise ``ValueError`` rather than being silently ignored.
    add_regularization : {None, "level_L1", "level_L2", "level_L_infinity",
            "grad_lag", "hess_lag", "hess_only_lag", "sqp_lag"}
        Optional level-set regularized OA master before fixed-integer NLP solves.
        L1, L-infinity, and ``grad_lag`` are solved as MILPs; quadratic modes
        require a MIQP-capable QP backend. Derivative modes require NLP duals,
        and Hessian modes require Lagrangian Hessian access.
    level_coef : float
        Coefficient in the open interval ``(0, 1)`` for the regularization
        level constraint. The level is
        ``(1 - level_coef) * incumbent_UB + level_coef * master_LB``.
    stalling_limit : int, optional
        Stop after this many consecutive incumbent-objective records without
        material progress.
    cycling_check : bool
        Stop when the master repeats a fixed-integer assignment.
    milp_solver : str
        MILP backend for OA master problems: ``"auto"``, ``"highs"``,
        ``"pounce"``, ``"simplex"``, or ``"gurobi"``.
    solution_pool : bool
        When true, request multiple Gurobi master MILP solutions per OA
        iteration and solve fixed-NLP subproblems for each selected integer
        assignment. Currently requires ``milp_solver="gurobi"``.
    num_solution_iteration : int
        Maximum number of master solution-pool candidates to process per OA
        iteration when ``solution_pool=True``.

    Returns
    -------
    SolveResult
    """
    t_start = time.perf_counter()
    init_strategy = _normalize_init_strategy(init_strategy)
    feasibility_norm = _normalize_feasibility_norm(feasibility_norm)
    fp_config = _normalize_fp_config(
        feasibility_norm=feasibility_norm,
        add_no_good_cuts=True,
        fp_iteration_limit=fp_iteration_limit,
        fp_cutoffdecr=fp_cutoffdecr,
        fp_projcuts=fp_projcuts,
        fp_transfercuts=fp_transfercuts,
        fp_projzerotol=fp_projzerotol,
        fp_mipgap=fp_mipgap,
        fp_discrete_only=fp_discrete_only,
        fp_main_norm=fp_main_norm,
        fp_norm_constraint=fp_norm_constraint,
        fp_norm_constraint_coef=fp_norm_constraint_coef,
    )
    add_regularization = _normalize_regularization(add_regularization)
    max_slack = _normalize_positive_float("max_slack", max_slack)
    oa_penalty_factor = _normalize_positive_float("oa_penalty_factor", oa_penalty_factor)
    level_coef = _normalize_open_unit_float("level_coef", level_coef)
    stalling_limit = _normalize_optional_positive_int("stalling_limit", stalling_limit)
    num_solution_iteration = _normalize_positive_int(
        "num_solution_iteration",
        num_solution_iteration,
    )
    heuristic_nonconvex = bool(heuristic_nonconvex)
    solution_pool = bool(solution_pool)
    if solution_pool:
        _require_solution_pool_backend(milp_solver)
    if add_regularization is not None and ecp_mode:
        raise ValueError("add_regularization is only supported for OA, not ECP mode.")
    if heuristic_nonconvex:
        equality_relaxation = True
        add_slack = True
    add_slack = bool(add_slack)
    add_no_good_cuts = bool(add_no_good_cuts)
    cycling_check = bool(cycling_check)

    # 1. Decompose model
    decomp = _decompose_model(model)
    evaluator = decomp.evaluator
    n_vars = decomp.n_vars
    n_cons = decomp.n_cons
    derivative_regularization = _regularization_requires_derivatives(add_regularization)
    if derivative_regularization and init_strategy == "fp" and n_cons > 0:
        raise ValueError(
            "OA derivative regularization needs an NLP-based initialization that returns "
            "duals; init_strategy='fp' does not provide constraint multipliers."
        )
    if add_regularization in _QP_REGULARIZATION_MODES and decomp.int_indices:
        _require_qp_regularization_backend(add_regularization)
    # The whole OA loop runs in the evaluator's minimization convention (it
    # negates a MAXIMIZE objective). Un-negate the user-facing objective/bound at
    # the return sites with this sign; the gap is convention-invariant.
    _obj_sign = (
        -1.0
        if (model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE)
        else 1.0
    )
    if decomp.oa_constraint_mask is not None and not all(decomp.oa_constraint_mask):
        logger.warning(
            "OA: generating OA cuts only for %d of %d constraints classified convex",
            sum(1 for is_convex in decomp.oa_constraint_mask if is_convex),
            len(decomp.oa_constraint_mask),
        )
    if not decomp.obj_is_linear and not decomp.oa_objective_is_convex:
        logger.warning(
            "OA: nonlinear objective is not convex in the optimization sense; "
            "disabling master lower-bound updates and skipping objective OA cuts"
        )
    master_bound_valid = decomp.master_bound_valid and not heuristic_nonconvex

    # 2. Generate initial linearization cuts.
    oa_A_rows: list[np.ndarray] = []
    oa_b_rows: list[float] = []
    oa_cut_relaxable: list[bool] = []
    cut_provenance = MIPNLPCutProvenance()

    UB = 1e20
    LB = -1e20
    incumbent = None
    incumbent_obj = None
    integer_assignments_seen: set[tuple[float, ...]] = set()
    incumbent_progress: list[float] = []
    termination_reason = None
    incumbent_derivative_data: Optional[_DerivativeRegularizationData] = None

    method_name = "ecp" if ecp_mode else "oa"
    trace_iterations: list[dict[str, object]] = []
    mip_count = 0
    nlp_subproblem_count = 0
    feasibility_subproblem_count = 0
    solution_pool_candidate_count = 0

    def _trace_value(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        out = float(value)
        if not np.isfinite(out) or abs(out) >= 1e19:
            return None
        return out

    def _trace_status(status) -> str:
        name = getattr(status, "name", None)
        if isinstance(name, str):
            return name.lower()
        return str(status).lower()

    def _cut_source_delta(before: dict[str, int]) -> dict[str, int]:
        after = cut_provenance.source_counts()
        return {source: int(after.get(source, 0) - before.get(source, 0)) for source in after}

    def _build_mip_nlp_trace(final_reason: Optional[str]) -> dict[str, object]:
        final_lb = _trace_value(LB)
        final_ub = _trace_value(UB)
        bound_valid = bool(master_bound_valid and final_lb is not None)
        final_gap = (
            _trace_value(_compute_gap(LB, UB)) if bound_valid and final_ub is not None else None
        )
        gap_certified = bool(bound_valid and final_gap is not None)
        return {
            "schema_version": 1,
            "solver": "mip-nlp",
            "method": method_name,
            "profile": mip_nlp_profile,
            "shot_options": (
                mip_nlp_shot_config.as_trace_dict() if mip_nlp_shot_config is not None else {}
            ),
            "iterations": trace_iterations,
            "summary": {
                "mip_count": int(mip_count),
                "nlp_subproblem_count": int(nlp_subproblem_count),
                "feasibility_subproblem_count": int(feasibility_subproblem_count),
                "cut_count": int(len(oa_A_rows)),
                "provenance_cut_count": int(len(cut_provenance.records)),
                "cut_source_counts": cut_provenance.source_counts(),
                "solution_pool_candidates": int(solution_pool_candidate_count),
            },
            "termination_reason": final_reason,
            "master_bound_valid": bool(master_bound_valid),
            "gap_certified": gap_certified,
            "bound_validity": "global" if bound_valid else "heuristic",
            "final_lb": final_lb,
            "final_ub": final_ub,
            "final_gap": final_gap,
        }

    # If no integer variables, just solve the NLP directly
    if len(decomp.int_indices) == 0:
        x_sol, obj = _solve_nlp_relaxation(
            evaluator,
            decomp.lb,
            decomp.ub,
            nlp_solver,
            initial_point=initial_point,
        )
        wall_time = time.perf_counter() - t_start
        if x_sol is not None:
            LB = float(obj)
            UB = float(obj)
            return SolveResult(
                status="optimal",
                objective=_obj_sign * obj,
                bound=_obj_sign * obj,
                gap=0.0,
                x=_build_x_dict(x_sol, model),
                wall_time=wall_time,
                mip_nlp_trace=_build_mip_nlp_trace("continuous_nlp_optimal"),
            )
        return SolveResult(
            status="infeasible",
            objective=None,
            bound=None,
            gap=None,
            x={},
            wall_time=wall_time,
            mip_nlp_trace=_build_mip_nlp_trace("continuous_nlp_infeasible"),
        )

    def accept_incumbent(
        x: np.ndarray,
        obj: float,
        multipliers: Optional[np.ndarray],
    ) -> None:
        nonlocal UB, incumbent, incumbent_obj, incumbent_derivative_data
        UB = float(obj)
        incumbent = np.asarray(x, dtype=np.float64).copy()
        incumbent_obj = float(obj)
        if derivative_regularization:
            assert add_regularization is not None
            incumbent_derivative_data = _build_derivative_regularization_data(
                decomp,
                add_regularization,
                incumbent,
                multipliers,
            )

    if init_strategy == "rNLP":
        relax_attempt = None
        if derivative_regularization:
            relax_attempt = _solve_nlp_relaxation(
                evaluator,
                decomp.lb,
                decomp.ub,
                nlp_solver,
                initial_point=initial_point,
                return_attempt=True,
            )
            x_relax, obj_relax = relax_attempt.x, relax_attempt.objective
        else:
            x_relax, obj_relax = _solve_nlp_relaxation(
                evaluator,
                decomp.lb,
                decomp.ub,
                nlp_solver,
                initial_point=initial_point,
            )

        if x_relax is not None:
            _add_oa_cuts(
                evaluator,
                x_relax,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
                cut_provenance=cut_provenance,
            )
            # Check if relaxation solution is already integer-feasible.
            is_int_feasible = all(
                abs(x_relax[idx] - round(x_relax[idx])) < 1e-5 for idx in decomp.int_indices
            )
            if is_int_feasible and obj_relax is not None:
                multipliers = relax_attempt.multipliers if relax_attempt is not None else None
                accept_incumbent(x_relax, obj_relax, multipliers)
        else:
            # NLP relaxation failed; generate initial cuts at the deterministic midpoint.
            x_mid = _default_nlp_start(decomp.lb, decomp.ub)
            _add_oa_cuts(
                evaluator,
                x_mid,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
                cut_provenance=cut_provenance,
            )
    elif init_strategy == "fp":
        fp_iterations = _fp_iteration_count(
            max_iterations,
            fp_config.iteration_limit,
            default_cap=10,
        )
        fp_result = _run_feasibility_pump(
            model,
            decomp,
            nlp_solver=nlp_solver,
            initial_point=initial_point,
            time_limit=max(time_limit - (time.perf_counter() - t_start), 0.0),
            gap_tolerance=gap_tolerance,
            max_iterations=fp_iterations,
            feasibility_norm=fp_config.feasibility_norm,
            add_no_good_cuts=fp_config.add_no_good_cuts,
            fp_main_norm=fp_config.main_norm,
            fp_mipgap=fp_config.mipgap,
            fp_discrete_only=fp_config.discrete_only,
            fp_projzerotol=fp_config.projzerotol,
            milp_solver=milp_solver,
        )
        x_cut = fp_result.best_x if fp_result.best_x is not None else fp_result.best_near_x
        if x_cut is None:
            x_cut = _default_nlp_start(decomp.lb, decomp.ub)
        _add_oa_cuts(
            evaluator,
            x_cut,
            n_vars,
            n_cons,
            decomp.constraint_senses,
            oa_A_rows,
            oa_b_rows,
            decomp.obj_is_linear,
            decomp.oa_constraint_mask,
            decomp.oa_objective_is_convex,
            equality_relaxation=equality_relaxation,
            oa_cut_relaxable=oa_cut_relaxable,
            cut_provenance=cut_provenance,
        )
        if fp_result.best_x is not None and fp_result.best_obj is not None:
            accept_incumbent(fp_result.best_x, fp_result.best_obj, None)
    else:
        x_seed = _build_initial_strategy_point(decomp, init_strategy, initial_point)
        if ecp_mode:
            _add_oa_cuts(
                evaluator,
                x_seed,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
                cut_provenance=cut_provenance,
            )
            if _is_primal_feasible(evaluator, x_seed):
                accept_incumbent(x_seed, float(evaluator.evaluate_objective(x_seed)), None)
        else:
            init_attempt = None
            if derivative_regularization:
                nlp_subproblem_count += 1
                init_attempt = _solve_nlp_subproblem(
                    evaluator,
                    decomp.lb,
                    decomp.ub,
                    decomp.int_indices,
                    x_seed,
                    nlp_solver,
                    initial_point=x_seed,
                    return_attempt=True,
                )
                x_init, obj_init = init_attempt.x, init_attempt.objective
            else:
                nlp_subproblem_count += 1
                x_init, obj_init = _solve_nlp_subproblem(
                    evaluator,
                    decomp.lb,
                    decomp.ub,
                    decomp.int_indices,
                    x_seed,
                    nlp_solver,
                    initial_point=x_seed,
                )
            x_cut = x_init if x_init is not None else x_seed
            _add_oa_cuts(
                evaluator,
                x_cut,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
                cut_provenance=cut_provenance,
            )
            if x_init is not None and obj_init is not None:
                multipliers = init_attempt.multipliers if init_attempt is not None else None
                accept_incumbent(x_init, obj_init, multipliers)

    # 3. Main OA loop
    for iteration in range(max_iterations):
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            logger.info("OA: Time limit reached at iteration %d", iteration)
            termination_reason = "time_limit"
            break

        # a. Solve master MILP
        cuts_before = len(oa_A_rows)
        provenance_before = len(cut_provenance.records)
        cut_source_counts_before = cut_provenance.source_counts()
        nlp_before = nlp_subproblem_count
        feasibility_before = feasibility_subproblem_count
        lb_before = _trace_value(LB)
        ub_before = _trace_value(UB)
        master_result = _solve_master_milp(
            decomp.linear_A_rows,
            decomp.linear_b_rows,
            decomp.linear_senses,
            oa_A_rows,
            oa_b_rows,
            n_vars,
            decomp.integrality,
            decomp.lb,
            decomp.ub,
            decomp.obj_coeffs,
            decomp.obj_is_linear,
            master_bound_valid,
            time_limit=time_limit - elapsed,
            gap_tolerance=gap_tolerance,
            add_slack=add_slack,
            max_slack=max_slack,
            oa_penalty_factor=oa_penalty_factor,
            oa_cut_relaxable=oa_cut_relaxable,
            use_objective_epigraph=(not decomp.obj_is_linear and decomp.oa_objective_is_convex),
            milp_solver=milp_solver,
            solution_pool=solution_pool,
            num_solution_iteration=num_solution_iteration,
        )
        mip_count += 1

        from discopt.solvers import SolveStatus

        if master_result is None:
            logger.info("OA: Master MILP failed at iteration %d", iteration)
            termination_reason = "master_error"
            trace_iterations.append(
                {
                    "index": int(iteration),
                    "master_status": "error",
                    "lb_before": lb_before,
                    "ub_before": ub_before,
                    "lb": _trace_value(LB),
                    "ub": _trace_value(UB),
                    "gap": _trace_value(_compute_gap(LB, UB)),
                    "cuts_added": int(len(oa_A_rows) - cuts_before),
                    "cuts_total": int(len(oa_A_rows)),
                    "provenance_cuts_added": int(len(cut_provenance.records) - provenance_before),
                    "provenance_cuts_total": int(len(cut_provenance.records)),
                    "cuts_added_by_source": _cut_source_delta(cut_source_counts_before),
                    "nlp_subproblem_count": int(nlp_subproblem_count - nlp_before),
                    "feasibility_subproblem_count": int(
                        feasibility_subproblem_count - feasibility_before
                    ),
                    "solution_pool_candidates": 0,
                    "node_count": 0,
                    "repair_actions": [],
                    "termination_reason": termination_reason,
                }
            )
            break

        if master_result.status == SolveStatus.INFEASIBLE:
            logger.info("OA: Master MILP infeasible at iteration %d", iteration)
            termination_reason = "master_infeasible"
            trace_iterations.append(
                {
                    "index": int(iteration),
                    "master_status": _trace_status(master_result.status),
                    "lb_before": lb_before,
                    "ub_before": ub_before,
                    "lb": _trace_value(LB),
                    "ub": _trace_value(UB),
                    "gap": _trace_value(_compute_gap(LB, UB)),
                    "cuts_added": int(len(oa_A_rows) - cuts_before),
                    "cuts_total": int(len(oa_A_rows)),
                    "provenance_cuts_added": int(len(cut_provenance.records) - provenance_before),
                    "provenance_cuts_total": int(len(cut_provenance.records)),
                    "cuts_added_by_source": _cut_source_delta(cut_source_counts_before),
                    "nlp_subproblem_count": int(nlp_subproblem_count - nlp_before),
                    "feasibility_subproblem_count": int(
                        feasibility_subproblem_count - feasibility_before
                    ),
                    "solution_pool_candidates": 0,
                    "node_count": int(getattr(master_result, "node_count", 0) or 0),
                    "repair_actions": [],
                    "termination_reason": termination_reason,
                }
            )
            break

        if master_result.status == SolveStatus.UNBOUNDED or master_result.x is None:
            # Master unbounded → need more OA cuts. Generate at midpoint.
            logger.info("OA: Master MILP unbounded at iteration %d, adding cuts", iteration)
            lb_clip = np.clip(decomp.lb, -1e8, 1e8)
            ub_clip = np.clip(decomp.ub, -1e8, 1e8)
            x_mid = 0.5 * (lb_clip + ub_clip)
            _add_oa_cuts(
                evaluator,
                x_mid,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
                cut_provenance=cut_provenance,
            )
            trace_iterations.append(
                {
                    "index": int(iteration),
                    "master_status": _trace_status(master_result.status),
                    "lb_before": lb_before,
                    "ub_before": ub_before,
                    "lb": _trace_value(LB),
                    "ub": _trace_value(UB),
                    "gap": _trace_value(_compute_gap(LB, UB)),
                    "cuts_added": int(len(oa_A_rows) - cuts_before),
                    "cuts_total": int(len(oa_A_rows)),
                    "provenance_cuts_added": int(len(cut_provenance.records) - provenance_before),
                    "provenance_cuts_total": int(len(cut_provenance.records)),
                    "cuts_added_by_source": _cut_source_delta(cut_source_counts_before),
                    "nlp_subproblem_count": int(nlp_subproblem_count - nlp_before),
                    "feasibility_subproblem_count": int(
                        feasibility_subproblem_count - feasibility_before
                    ),
                    "solution_pool_candidates": 0,
                    "node_count": int(getattr(master_result, "node_count", 0) or 0),
                    "repair_actions": [],
                    "termination_reason": "master_unbounded",
                }
            )
            continue

        # The master gives a valid LB only via its dual ``bound`` (never the
        # incumbent ``objective``, which is an upper bound on a limited solve).
        if master_bound_valid and master_result.bound is not None:
            LB = max(LB, master_result.bound)

        nlp_initial_point = None
        if (
            add_regularization is not None
            and incumbent is not None
            and incumbent_obj is not None
            and master_bound_valid
            and np.isfinite(LB)
            and np.isfinite(UB)
            and LB > -1e19
            and UB < 1e19
        ):
            regularization_lb = LB
            if decomp.obj_is_linear and decomp.obj_coeffs is not None:
                regularization_lb += float(decomp.obj_coeffs[1])
            objective_level = (1.0 - level_coef) * float(UB) + level_coef * float(regularization_lb)
            remaining_time = max(0.0, time_limit - (time.perf_counter() - t_start))
            derivative_data = None
            if derivative_regularization:
                if incumbent_derivative_data is None:
                    raise RuntimeError(
                        f"OA {add_regularization} regularization requires Lagrangian "
                        "derivative data from an incumbent NLP solve."
                    )
                derivative_data = incumbent_derivative_data
            x_regularized = _solve_regularized_master(
                decomp,
                oa_A_rows,
                oa_b_rows,
                add_regularization=add_regularization,
                target=incumbent,
                objective_level=objective_level,
                time_limit=remaining_time,
                gap_tolerance=gap_tolerance,
                add_slack=add_slack,
                max_slack=max_slack,
                oa_penalty_factor=oa_penalty_factor,
                oa_cut_relaxable=oa_cut_relaxable,
                use_objective_epigraph=(not decomp.obj_is_linear and decomp.oa_objective_is_convex),
                derivative_data=derivative_data,
                milp_solver=milp_solver,
            )
            if x_regularized is not None:
                nlp_initial_point = x_regularized
                logger.info(
                    "OA: %s regularized master selected fixed-NLP initial point",
                    add_regularization,
                )

        master_candidates = _master_solution_candidates(
            master_result,
            n_vars,
            solution_pool=solution_pool,
            num_solution_iteration=num_solution_iteration,
        )
        solution_pool_candidate_count += len(master_candidates)
        iteration_record: dict[str, object] = {
            "index": int(iteration),
            "master_status": _trace_status(master_result.status),
            "lb_before": lb_before,
            "ub_before": ub_before,
            "solution_pool_candidates": int(len(master_candidates)),
            "node_count": int(getattr(master_result, "node_count", 0) or 0),
            "repair_actions": [],
        }
        stop_after_master_pool = False
        pool_integer_assignments_seen: set[tuple[float, ...]] = set()

        for candidate_index, x_master in enumerate(master_candidates):
            elapsed = time.perf_counter() - t_start
            if elapsed >= time_limit:
                logger.info("OA: Time limit reached during iteration %d", iteration)
                termination_reason = "time_limit"
                stop_after_master_pool = True
                break

            int_assignment = tuple(
                _round_integral_to_bounds(x_master[idx], decomp.lb[idx], decomp.ub[idx])
                for idx in decomp.int_indices
            )
            if solution_pool:
                if int_assignment in pool_integer_assignments_seen:
                    logger.info(
                        "OA: skipping duplicate pooled integer assignment %s",
                        int_assignment,
                    )
                    continue
                pool_integer_assignments_seen.add(int_assignment)
            if cycling_check:
                if int_assignment in integer_assignments_seen:
                    logger.info(
                        "OA: cycling detected at iteration %d for integer assignment %s",
                        iteration,
                        int_assignment,
                    )
                    termination_reason = "cycling"
                    stop_after_master_pool = True
                    break
                integer_assignments_seen.add(int_assignment)

            # b. ECP mode: add cuts at master point, skip NLP
            if ecp_mode:
                n_violated = _add_ecp_cuts(
                    evaluator,
                    x_master,
                    n_vars,
                    decomp.constraint_senses,
                    oa_A_rows,
                    oa_b_rows,
                    decomp.obj_is_linear,
                    decomp.oa_constraint_mask,
                    decomp.oa_objective_is_convex,
                    equality_relaxation=equality_relaxation,
                    oa_cut_relaxable=oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                )
                # In ECP, use master objective as heuristic UB
                master_obj = float(evaluator.evaluate_objective(x_master))
                cons_vals = evaluator.evaluate_constraints(x_master)
                is_feasible = all(cons_vals[k] <= 1e-6 for k in range(n_cons))
                if is_feasible and master_obj < UB:
                    UB = master_obj
                    incumbent = x_master.copy()
                    incumbent_obj = master_obj

                gap = _compute_gap(LB, UB)
                logger.info(
                    "OA-ECP iter %d: LB=%.6f UB=%.6f gap=%.4f%% cuts=%d violated=%d",
                    iteration,
                    LB,
                    UB,
                    gap * 100,
                    len(oa_A_rows),
                    n_violated,
                )

                if n_violated == 0:
                    termination_reason = "ecp_feasible"
                    stop_after_master_pool = True
                    break
                if master_bound_valid and gap <= gap_tolerance:
                    termination_reason = "gap"
                    stop_after_master_pool = True
                    break
                continue

            # c. Fix integers, solve NLP subproblem
            nlp_attempt = None
            if derivative_regularization:
                nlp_subproblem_count += 1
                nlp_attempt = _solve_nlp_subproblem(
                    evaluator,
                    decomp.lb,
                    decomp.ub,
                    decomp.int_indices,
                    x_master,
                    nlp_solver,
                    initial_point=nlp_initial_point,
                    return_attempt=True,
                )
                x_nlp, obj_nlp = nlp_attempt.x, nlp_attempt.objective
            else:
                nlp_subproblem_count += 1
                x_nlp, obj_nlp = _solve_nlp_subproblem(
                    evaluator,
                    decomp.lb,
                    decomp.ub,
                    decomp.int_indices,
                    x_master,
                    nlp_solver,
                    initial_point=nlp_initial_point,
                )

            if x_nlp is not None:
                if obj_nlp < UB:
                    multipliers = nlp_attempt.multipliers if nlp_attempt is not None else None
                    accept_incumbent(x_nlp, obj_nlp, multipliers)

                # Generate OA cuts at NLP solution
                _add_oa_cuts(
                    evaluator,
                    x_nlp,
                    n_vars,
                    n_cons,
                    decomp.constraint_senses,
                    oa_A_rows,
                    oa_b_rows,
                    decomp.obj_is_linear,
                    decomp.oa_constraint_mask,
                    decomp.oa_objective_is_convex,
                    equality_relaxation=equality_relaxation,
                    oa_cut_relaxable=oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                )
            else:
                # NLP infeasible for this integer assignment
                if feasibility_cuts:
                    feasibility_subproblem_count += 1
                    x_feas = _solve_feasibility_subproblem(
                        evaluator,
                        decomp.lb,
                        decomp.ub,
                        decomp.int_indices,
                        x_master,
                        nlp_solver,
                        feasibility_norm,
                    )
                    if x_feas is not None:
                        _add_feasibility_cuts(
                            evaluator,
                            x_feas,
                            n_vars,
                            decomp.constraint_senses,
                            oa_A_rows,
                            oa_b_rows,
                            decomp.oa_constraint_mask,
                            oa_cut_relaxable=oa_cut_relaxable,
                            cut_provenance=cut_provenance,
                        )

                if add_no_good_cuts and not decomp.general_integer_indices:
                    _add_no_good_cut(
                        x_master,
                        decomp.binary_indices,
                        oa_A_rows,
                        oa_b_rows,
                        n_vars,
                        oa_cut_relaxable=oa_cut_relaxable,
                        cut_provenance=cut_provenance,
                    )

                # Also add OA cuts at master point
                _add_oa_cuts(
                    evaluator,
                    x_master,
                    n_vars,
                    n_cons,
                    decomp.constraint_senses,
                    oa_A_rows,
                    oa_b_rows,
                    decomp.obj_is_linear,
                    decomp.oa_constraint_mask,
                    decomp.oa_objective_is_convex,
                    equality_relaxation=equality_relaxation,
                    oa_cut_relaxable=oa_cut_relaxable,
                    cut_provenance=cut_provenance,
                )

            # d. Check convergence
            gap = _compute_gap(LB, UB)
            logger.info(
                "OA iter %d: LB=%.6f UB=%.6f gap=%.4f%% cuts=%d",
                iteration,
                LB,
                UB,
                gap * 100,
                len(oa_A_rows),
            )

            if incumbent_obj is not None:
                incumbent_progress.append(float(UB))
                if stalling_limit is not None and len(incumbent_progress) >= stalling_limit:
                    prev = incumbent_progress[-stalling_limit]
                    if abs(incumbent_progress[-1] - prev) <= 1e-12:
                        logger.info(
                            "OA: stalling detected after %d incumbent records; best objective %.6f",
                            stalling_limit,
                            UB,
                        )
                        termination_reason = "stalling"
                        stop_after_master_pool = True
                        break

            if master_bound_valid and gap <= gap_tolerance:
                termination_reason = "gap"
                stop_after_master_pool = True
                break

        if stop_after_master_pool:
            iteration_record["termination_reason"] = termination_reason
        iteration_record.update(
            {
                "lb": _trace_value(LB),
                "ub": _trace_value(UB),
                "gap": _trace_value(_compute_gap(LB, UB)),
                "cuts_added": int(len(oa_A_rows) - cuts_before),
                "cuts_total": int(len(oa_A_rows)),
                "provenance_cuts_added": int(len(cut_provenance.records) - provenance_before),
                "provenance_cuts_total": int(len(cut_provenance.records)),
                "cuts_added_by_source": _cut_source_delta(cut_source_counts_before),
                "nlp_subproblem_count": int(nlp_subproblem_count - nlp_before),
                "feasibility_subproblem_count": int(
                    feasibility_subproblem_count - feasibility_before
                ),
            }
        )
        trace_iterations.append(iteration_record)
        if stop_after_master_pool:
            break

    # 4. Build result
    wall_time = time.perf_counter() - t_start
    gap = _compute_gap(LB, UB)
    bound_certified = master_bound_valid
    bound = LB if bound_certified and LB > -1e19 else None
    reported_gap = gap if bound is not None and UB < 1e19 else None
    final_reason = termination_reason
    if final_reason is None:
        if wall_time >= time_limit:
            final_reason = "time_limit"
        elif incumbent is not None and bound_certified and gap <= gap_tolerance:
            final_reason = "gap"
        else:
            final_reason = "iteration_limit"

    if incumbent is not None and incumbent_obj is not None:
        status = "optimal" if bound_certified and gap <= gap_tolerance else "feasible"
        if termination_reason in {"cycling", "stalling"}:
            status = "feasible"
        return SolveResult(
            status=status,
            objective=_obj_sign * incumbent_obj,
            bound=(_obj_sign * bound if bound is not None else None),
            gap=reported_gap,
            x=_build_x_dict(incumbent, model),
            wall_time=wall_time,
            mip_count=mip_count,
            subnlp_calls=nlp_subproblem_count,
            mip_nlp_trace=_build_mip_nlp_trace(final_reason),
        )

    return SolveResult(
        status="infeasible",
        objective=None,
        bound=(_obj_sign * bound if bound is not None else None),
        gap=None,
        x={},
        wall_time=wall_time,
        mip_count=mip_count,
        subnlp_calls=nlp_subproblem_count,
        mip_nlp_trace=_build_mip_nlp_trace(final_reason),
    )
