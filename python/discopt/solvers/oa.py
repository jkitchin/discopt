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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from discopt.modeling.core import Constraint, Model, ObjectiveSense, SolveResult, VarType
from discopt.solvers.mip_nlp_candidates import FixedNLPCandidate, FixedNLPCandidateManager
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
    "initial_poa",
    "relaxation_phase",
    "objective",
    "objective_rootsearch",
    "reduction",
    "esh",
    "feasibility",
    "integer",
    "external",
)
_INITIAL_POA_PHASES = frozenset({"auto", "initial"})
_PERIODIC_RELAXATION_PHASES = frozenset({"periodic"})
_SHOT_MASTER_FEATURE_BACKEND = "gurobi"


def _normalize_optional_hook(name: str, hook: Any) -> Any:
    if hook is not None and not callable(hook):
        raise ValueError(f"{name} must be callable or None, got {type(hook).__name__}.")
    return hook


def _validate_hook_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}.")
    return bool(value)


def _finite_hook_float(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}.") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return out


def _external_hook_items(value: Any, *, hook_name: str, item_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(
            f"{hook_name} must return None or an iterable of {item_name} payloads."
        ) from exc


def _validate_external_primal_candidates(
    value: Any,
    *,
    n_vars: int,
    hook_name: str = "external_primal_candidate_hook",
) -> list[dict[str, object]]:
    if value is None:
        return []
    items: list[Any]
    if isinstance(value, Mapping):
        items = [value]
    else:
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            items = _external_hook_items(value, hook_name=hook_name, item_name="candidate")
        else:
            if arr.ndim == 1:
                items = [arr]
            elif arr.ndim == 2:
                items = [row for row in arr]
            else:
                raise ValueError(
                    f"{hook_name} returned candidate array with {arr.ndim} dimensions; "
                    "expected a 1-D point or 2-D point matrix."
                )

    out: list[dict[str, object]] = []
    for idx, item in enumerate(items):
        if isinstance(item, Mapping):
            if "point" not in item:
                raise ValueError(f"{hook_name} candidate {idx} must include a 'point' entry.")
            raw_point = item["point"]
            objective = item.get("objective")
            provider = item.get("provider")
            nlp_source = item.get("nlp_source", "active")
        else:
            raw_point = item
            objective = None
            provider = None
            nlp_source = "active"

        point = np.asarray(raw_point, dtype=np.float64).reshape(-1)
        if point.shape != (int(n_vars),):
            raise ValueError(
                f"{hook_name} candidate {idx} point has length {point.size}; "
                f"expected {int(n_vars)}."
            )
        if not np.all(np.isfinite(point)):
            raise ValueError(f"{hook_name} candidate {idx} point must contain only finite values.")
        payload: dict[str, object] = {
            "point": point.copy(),
            "source": "external",
            "nlp_source": str(nlp_source),
        }
        if objective is not None:
            payload["objective"] = _finite_hook_float(
                f"{hook_name} candidate {idx} objective",
                objective,
            )
        if provider is not None:
            payload["provider"] = str(provider)
        out.append(payload)
    return out


def _validate_external_hyperplanes(
    value: Any,
    *,
    n_vars: int,
    hook_name: str = "external_hyperplane_hook",
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for idx, item in enumerate(
        _external_hook_items(value, hook_name=hook_name, item_name="hyperplane")
    ):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{hook_name} hyperplane {idx} must be a dict with coefficients and rhs."
            )
        coeffs_raw = item.get("coefficients", item.get("coeffs"))
        if coeffs_raw is None:
            raise ValueError(
                f"{hook_name} hyperplane {idx} must include 'coefficients' or 'coeffs'."
            )
        if "rhs" not in item:
            raise ValueError(f"{hook_name} hyperplane {idx} must include 'rhs'.")
        coeffs = np.asarray(coeffs_raw, dtype=np.float64).reshape(-1)
        if coeffs.shape != (int(n_vars),):
            raise ValueError(
                f"{hook_name} hyperplane {idx} has {coeffs.size} coefficients; "
                f"expected {int(n_vars)}."
            )
        if not np.all(np.isfinite(coeffs)):
            raise ValueError(
                f"{hook_name} hyperplane {idx} coefficients must contain only finite values."
            )
        if np.linalg.norm(coeffs) < 1e-12:
            raise ValueError(f"{hook_name} hyperplane {idx} coefficients must be nonzero.")
        rhs = _finite_hook_float(f"{hook_name} hyperplane {idx} rhs", item["rhs"])
        support = None
        if item.get("supporting_point") is not None:
            support = np.asarray(item["supporting_point"], dtype=np.float64).reshape(-1)
            if support.shape != (int(n_vars),):
                raise ValueError(
                    f"{hook_name} hyperplane {idx} supporting_point has length "
                    f"{support.size}; expected {int(n_vars)}."
                )
            if not np.all(np.isfinite(support)):
                raise ValueError(
                    f"{hook_name} hyperplane {idx} supporting_point must contain "
                    "only finite values."
                )
        constraint_id = item.get("constraint_id")
        if constraint_id is not None:
            constraint_id = int(constraint_id)
            if constraint_id < 0:
                raise ValueError(f"{hook_name} hyperplane {idx} constraint_id must be nonnegative.")
        objective_id = item.get("objective_id")
        violation = item.get("violation")
        payload: dict[str, object] = {
            "coefficients": coeffs.copy(),
            "rhs": rhs,
            "relaxable": _validate_hook_bool(
                f"{hook_name} hyperplane {idx} relaxable",
                item.get("relaxable", True),
            ),
            "global_valid": _validate_hook_bool(
                f"{hook_name} hyperplane {idx} global_valid",
                item.get("global_valid", True),
            ),
            "local_valid": _validate_hook_bool(
                f"{hook_name} hyperplane {idx} local_valid",
                item.get("local_valid", True),
            ),
            "supporting_point": None if support is None else support.copy(),
            "violation": (
                None
                if violation is None
                else _finite_hook_float(f"{hook_name} hyperplane {idx} violation", violation)
            ),
            "constraint_id": constraint_id,
            "objective_id": None if objective_id is None else str(objective_id),
        }
        out.append(payload)
    return out


def _validate_external_dual_bound(
    value: Any,
    *,
    hook_name: str = "external_dual_bound_hook",
) -> Optional[dict[str, object]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "bound" not in value:
            raise ValueError(f"{hook_name} must return a number or a dict with a 'bound' entry.")
        bound = value["bound"]
        global_valid = value.get("global_valid", True)
        provider = value.get("provider")
    else:
        bound = value
        global_valid = True
        provider = None
    payload: dict[str, object] = {
        "bound": _finite_hook_float(f"{hook_name} bound", bound),
        "global_valid": _validate_hook_bool(f"{hook_name} global_valid", global_valid),
    }
    if provider is not None:
        payload["provider"] = str(provider)
    return payload


def _validate_external_termination(
    value: Any,
    *,
    hook_name: str = "termination_hook",
) -> bool:
    return _validate_hook_bool(f"{hook_name} return value", value)


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

    def remove_source(self, source: str) -> int:
        """Drop records for an inactive cut source and rebuild dedup state."""
        keep: list[MIPNLPCutRecord] = []
        removed = 0
        for record in self.records:
            if record.source == source:
                removed += 1
            else:
                keep.append(record)
        if removed:
            self.records = keep
            self._dedup_keys = {record.dedup_key for record in keep}
        return removed


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
    integer_binary_expansion: Optional["_IntegerBinaryExpansion"] = None
    integer_binary_start: Optional[int] = None


@dataclass(frozen=True)
class _IntegerBinaryVariable:
    """Binary expansion metadata for one original general-integer variable."""

    index: int
    lower: int
    upper: int
    bit_start: int
    bit_count: int


@dataclass(frozen=True)
class _IntegerBinaryExpansion:
    """Logical binary tail used by OA master no-good cuts.

    Stored cut rows use ``[original variables, eta-slot, expansion bits]``. The
    eta slot disambiguates expanded rows from legacy objective-epigraph rows.
    """

    n_vars: int
    variables: tuple[_IntegerBinaryVariable, ...]
    bit_count: int

    @property
    def logical_width(self) -> int:
        return self.n_vars + 1 + self.bit_count

    @property
    def logical_binary_indices(self) -> list[int]:
        return [self.n_vars + 1 + idx for idx in range(self.bit_count)]

    def bit_values_for_point(self, point) -> np.ndarray:
        x = np.asarray(point, dtype=np.float64).ravel()
        values = np.zeros(self.bit_count, dtype=np.float64)
        for spec in self.variables:
            if spec.bit_count <= 0:
                continue
            val = _round_integral_to_bounds(x[spec.index], spec.lower, spec.upper)
            offset = int(round(val)) - spec.lower
            for bit in range(spec.bit_count):
                values[spec.bit_start + bit] = 1.0 if (offset & (1 << bit)) else 0.0
        return values

    def logical_point(self, point) -> np.ndarray:
        x = np.asarray(point, dtype=np.float64).ravel()
        if self.bit_count <= 0:
            return cast(np.ndarray, x[: self.n_vars].copy())
        return cast(
            np.ndarray,
            np.concatenate([x[: self.n_vars], np.array([0.0]), self.bit_values_for_point(x)]),
        )


def _build_integer_binary_expansion(
    decomp: "_DecomposedProblem",
    *,
    enabled: bool,
) -> Optional[_IntegerBinaryExpansion]:
    """Return binary expansion metadata for bounded general-integer variables."""
    if not enabled or not decomp.general_integer_indices:
        return None

    variables: list[_IntegerBinaryVariable] = []
    bit_start = 0
    for idx in decomp.general_integer_indices:
        raw_lb = float(decomp.lb[idx])
        raw_ub = float(decomp.ub[idx])
        if (
            not np.isfinite(raw_lb)
            or not np.isfinite(raw_ub)
            or abs(raw_lb) >= _START_BOUND_CLIP
            or abs(raw_ub) >= _START_BOUND_CLIP
        ):
            raise ValueError(
                "integer_to_binary=True requires finite practical bounds for every "
                f"general-integer variable; variable index {idx} has bounds "
                f"({raw_lb}, {raw_ub})."
            )
        lower = int(np.ceil(raw_lb))
        upper = int(np.floor(raw_ub))
        if lower > upper:
            raise ValueError(
                "integer_to_binary=True found no integer value inside bounds for "
                f"general-integer variable index {idx}: ({raw_lb}, {raw_ub})."
            )
        domain_width = upper - lower
        bit_count = int(domain_width).bit_length()
        variables.append(
            _IntegerBinaryVariable(
                index=int(idx),
                lower=lower,
                upper=upper,
                bit_start=bit_start,
                bit_count=bit_count,
            )
        )
        bit_start += bit_count

    return _IntegerBinaryExpansion(
        n_vars=int(decomp.n_vars),
        variables=tuple(variables),
        bit_count=int(bit_start),
    )


def _warn_integer_to_binary_noop(
    solver_name: str,
    *,
    integer_to_binary: bool,
    add_no_good_cuts: bool,
) -> None:
    if integer_to_binary and not add_no_good_cuts:
        logger.warning(
            "%s: integer_to_binary=True ignored because add_no_good_cuts=False; "
            "integer-to-binary expansion is only used for no-good cuts.",
            solver_name,
        )


def _stored_row_uses_integer_binary_expansion(
    row: np.ndarray,
    n_vars: int,
    integer_binary_expansion: Optional[_IntegerBinaryExpansion],
) -> bool:
    return (
        integer_binary_expansion is not None
        and integer_binary_expansion.bit_count > 0
        and len(row) == n_vars + 1 + integer_binary_expansion.bit_count
    )


def _stored_row_to_master_layout(
    row,
    *,
    n_vars: int,
    n_master: int,
    use_objective_epigraph: bool,
    slack_index: Optional[int],
    relaxable: bool,
    integer_binary_expansion: Optional[_IntegerBinaryExpansion] = None,
    integer_binary_start: Optional[int] = None,
) -> np.ndarray:
    """Copy a stored OA row into the active MILP master column layout."""
    raw = np.asarray(row, dtype=np.float64).ravel()
    out = np.zeros(n_master, dtype=np.float64)
    if _stored_row_uses_integer_binary_expansion(raw, n_vars, integer_binary_expansion):
        assert integer_binary_expansion is not None
        if integer_binary_start is None:
            raise ValueError("integer-binary cut row requires expansion columns in master")
        out[:n_vars] = raw[:n_vars]
        if use_objective_epigraph:
            out[n_vars] = raw[n_vars]
        out[integer_binary_start : integer_binary_start + integer_binary_expansion.bit_count] = raw[
            n_vars + 1 :
        ]
    else:
        if use_objective_epigraph and len(raw) == n_vars:
            out[:n_vars] = raw
        else:
            if len(raw) > n_master:
                raise ValueError(
                    f"OA cut has {len(raw)} coefficients but master has {n_master} variables"
                )
            out[: len(raw)] = raw
    if slack_index is not None:
        out[slack_index] = -1.0 if relaxable else 0.0
    return out


def _append_integer_binary_link_rows(
    a_eq_rows: list[np.ndarray],
    b_eq_vals: list[float],
    *,
    n_master: int,
    integer_binary_expansion: Optional[_IntegerBinaryExpansion],
    integer_binary_start: Optional[int],
) -> None:
    if (
        integer_binary_expansion is None
        or integer_binary_expansion.bit_count <= 0
        or integer_binary_start is None
    ):
        return
    for spec in integer_binary_expansion.variables:
        if spec.bit_count <= 0:
            continue
        row = np.zeros(n_master, dtype=np.float64)
        row[spec.index] = 1.0
        for bit in range(spec.bit_count):
            row[integer_binary_start + spec.bit_start + bit] = -float(1 << bit)
        a_eq_rows.append(row)
        b_eq_vals.append(float(spec.lower))


@dataclass
class _ShotMIPSolutionLimitState:
    """Small state machine for SHOT-style early MIP incumbent limits."""

    strategy: str
    capacity: int
    backend: str
    current_limit: Optional[int] = None
    updates: int = 0
    last_update_reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.capacity = max(1, int(self.capacity))
        if self.strategy in {"auto", "adaptive"}:
            self.current_limit = 1
            self.last_update_reason = "initial"
        elif self.strategy == "static":
            self.current_limit = self.capacity
            self.last_update_reason = "static"
        else:
            self.current_limit = None
            self.last_update_reason = "disabled"

    @property
    def enabled(self) -> bool:
        return self.strategy in {"auto", "adaptive", "static"}

    @property
    def supported(self) -> bool:
        return self.backend == _SHOT_MASTER_FEATURE_BACKEND

    @property
    def requested_limit(self) -> Optional[int]:
        if not self.enabled or not self.supported or self.current_limit is None:
            return None
        return max(1, int(self.current_limit))

    @property
    def degraded_reason(self) -> Optional[str]:
        if self.enabled and not self.supported:
            return "mip_solution_limit_strategy requires milp_solver='gurobi'"
        return None

    def as_trace_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "enabled": bool(self.enabled),
            "supported": bool(self.supported),
            "limit": self.requested_limit,
            "raw_limit": self.current_limit,
            "capacity": int(self.capacity),
            "updates": int(self.updates),
            "last_update_reason": self.last_update_reason,
            "degraded_reason": self.degraded_reason,
        }

    def observe_iteration(
        self,
        *,
        incumbent_improved: bool,
        cuts_added: int,
        master_status: str,
    ) -> dict[str, object]:
        before = self.current_limit
        reason = "unchanged"
        if self.strategy in {"auto", "adaptive"}:
            if incumbent_improved:
                self.current_limit = 1
                reason = "incumbent_improved"
            elif master_status in {"optimal", "iteration_limit", "time_limit"} and cuts_added <= 0:
                self.current_limit = min(self.capacity, max(1, int(self.current_limit or 1) + 1))
                reason = "no_new_cuts"
            else:
                reason = "cuts_added"
        elif self.strategy == "static":
            self.current_limit = self.capacity
            reason = "static"
        elif self.strategy == "force_optimal":
            self.current_limit = None
            reason = "force_optimal"
        else:
            self.current_limit = None
            reason = "disabled"

        if before != self.current_limit:
            self.updates += 1
        self.last_update_reason = reason
        out = self.as_trace_dict()
        out["previous_raw_limit"] = before
        out["update_reason"] = reason
        return out


def _shot_master_feature_supported(milp_solver: str) -> bool:
    return str(milp_solver).lower() == _SHOT_MASTER_FEATURE_BACKEND


def _extend_master_mip_start(
    master: _MasterMILPData,
    *,
    n_vars: int,
    mip_start,
    mip_start_objective: Optional[float],
) -> Optional[np.ndarray]:
    if mip_start is None:
        return None
    start = np.asarray(mip_start, dtype=np.float64).ravel()
    if start.size < n_vars:
        return None
    full = np.zeros(len(master.c), dtype=np.float64)
    for idx in range(n_vars):
        lo, hi = master.bounds[idx]
        full[idx] = min(max(float(start[idx]), float(lo)), float(hi))
    next_index = n_vars
    if master.use_objective_epigraph:
        if mip_start_objective is None or not np.isfinite(float(mip_start_objective)):
            return None
        lo, hi = master.bounds[next_index]
        full[next_index] = min(max(float(mip_start_objective), float(lo)), float(hi))
    if master.slack_index is not None:
        lo, hi = master.bounds[master.slack_index]
        full[master.slack_index] = min(max(0.0, float(lo)), float(hi))
    if (
        master.integer_binary_expansion is not None
        and master.integer_binary_expansion.bit_count > 0
        and master.integer_binary_start is not None
    ):
        bits = master.integer_binary_expansion.bit_values_for_point(start[:n_vars])
        full[
            master.integer_binary_start : master.integer_binary_start
            + master.integer_binary_expansion.bit_count
        ] = bits
    return full


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


def _coerce_nlp_attempt(result) -> _NLPAttempt:
    if isinstance(result, _NLPAttempt):
        return result
    if isinstance(result, tuple) and len(result) >= 2:
        return _NLPAttempt(x=result[0], objective=result[1], multipliers=None)
    raise TypeError(f"Expected _NLPAttempt or (x, objective), got {type(result).__name__}.")


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


def _solve_fixed_nlp_subproblem_attempt(
    evaluator,
    lb,
    ub,
    int_indices,
    x_master,
    nlp_solver,
    *,
    initial_point=None,
) -> _NLPAttempt:
    """Call the fixed-NLP helper and retain status when the implementation supports it."""
    try:
        result = _solve_nlp_subproblem(
            evaluator,
            lb,
            ub,
            int_indices,
            x_master,
            nlp_solver,
            initial_point=initial_point,
            return_attempt=True,
        )
    except TypeError as exc:
        if "return_attempt" not in str(exc):
            raise
        result = _solve_nlp_subproblem(
            evaluator,
            lb,
            ub,
            int_indices,
            x_master,
            nlp_solver,
            initial_point=initial_point,
        )
    return _coerce_nlp_attempt(result)


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
    constraint_source: str = "oa",
    objective_source: str = "objective",
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
                    source=constraint_source,
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
                    source=constraint_source,
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
                    source=constraint_source,
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
                    source=constraint_source,
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
            source=objective_source,
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
    include_local_cuts: bool = False,
    incumbent=None,
    local_cut_trace: Optional[dict[str, object]] = None,
):
    """Generate ECP cuts: OA cuts only for violated constraints at x_master."""
    from discopt._jax.cutting_planes import (
        generate_objective_oa_cut,
        separate_oa_cuts,
    )

    n_added = 0
    local_added = 0
    local_rejected = 0

    def reject_if_incumbent_excluded(coeffs, rhs: float, global_valid: bool) -> bool:
        nonlocal local_rejected
        if (
            not global_valid
            and incumbent is not None
            and _candidate_cut_excludes_point(coeffs, rhs, incumbent)
        ):
            local_rejected += 1
            return True
        return False

    def trace_counter_value(key: str) -> int:
        if local_cut_trace is None:
            return 0
        value = local_cut_trace.get(key, 0)
        if isinstance(value, (int, float)):
            return int(value)
        return 0

    if evaluator.n_constraints > 0:
        ecp_convex_mask = None if include_local_cuts else constraint_convex_mask
        cuts = separate_oa_cuts(
            evaluator,
            x_master,
            constraint_senses=constraint_senses,
            convex_mask=ecp_convex_mask,
        )
        constraint_ids = _constraint_ids_for_generated_oa_cuts(
            evaluator,
            x_master,
            constraint_senses,
            ecp_convex_mask,
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
                if reject_if_incumbent_excluded(coeffs, cut.rhs, global_valid):
                    continue
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
                if not global_valid:
                    local_added += 1
                n_added += 1
            elif sense == ">=":
                if reject_if_incumbent_excluded(-coeffs, -cut.rhs, global_valid):
                    continue
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
                if not global_valid:
                    local_added += 1
                n_added += 1
            elif sense == "==":
                if not reject_if_incumbent_excluded(coeffs, cut.rhs, global_valid):
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
                    if not global_valid:
                        local_added += 1
                    n_added += 1
                if reject_if_incumbent_excluded(-coeffs, -cut.rhs, global_valid):
                    continue
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
                if not global_valid:
                    local_added += 1
                n_added += 1

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

    if local_cut_trace is not None and local_rejected:
        previous = trace_counter_value("local_cuts_rejected")
        local_cut_trace["local_cuts_rejected"] = previous + int(local_rejected)
    if local_cut_trace is not None and local_added:
        previous = trace_counter_value("local_cuts_added")
        local_cut_trace["local_cuts_added"] = previous + int(local_added)

    return n_added


def _candidate_cut_violation(coeffs, rhs: float, point) -> float:
    lhs = float(np.dot(np.asarray(coeffs, dtype=np.float64), np.asarray(point, dtype=np.float64)))
    return max(0.0, lhs - float(rhs))


def _candidate_cut_excludes_point(coeffs, rhs: float, point, *, tol: float = 1e-8) -> bool:
    lhs = float(np.dot(np.asarray(coeffs, dtype=np.float64), np.asarray(point, dtype=np.float64)))
    return lhs > float(rhs) + float(tol)


@dataclass(frozen=True)
class _ESHHyperplaneCandidate:
    coeffs: np.ndarray
    rhs: float
    relaxable: bool
    source: str
    global_valid: bool
    local_valid: bool
    supporting_point: np.ndarray
    violation: float
    constraint_id: Optional[int] = None
    objective_id: Optional[str] = None


def _select_hyperplane_candidates(
    candidates: list[_ESHHyperplaneCandidate],
    *,
    max_per_iter: Optional[int],
    selection_factor: float,
) -> list[_ESHHyperplaneCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item.violation, reverse=True)
    keep_count = max(1, int(np.ceil(len(ordered) * float(selection_factor))))
    if max_per_iter is not None:
        keep_count = min(keep_count, int(max_per_iter))
    return ordered[:keep_count]


def _add_esh_cuts(
    evaluator,
    x_master,
    n_vars,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    obj_is_linear,
    constraint_convex_mask,
    objective_is_convex,
    interior_point_store,
    *,
    rootsearch_strategy: str,
    equality_relaxation: bool = False,
    oa_cut_relaxable=None,
    cut_provenance: Optional[MIPNLPCutProvenance] = None,
    incumbent=None,
    incumbent_obj=None,
    objective_epigraph_available: Optional[bool] = None,
    hyperplane_max_per_iter: Optional[int] = None,
    hyperplane_selection_factor: float = 1.0,
) -> tuple[int, dict[str, object]]:
    """Generate SHOT-style extended supporting hyperplanes with ECP fallback."""
    from discopt._jax.cutting_planes import (
        generate_oa_cuts_from_evaluator,
        generate_objective_oa_cut,
    )
    from discopt.solvers.mip_nlp_rootsearch import (
        MIPNLPRootSearchStatus,
        rootsearch_from_store,
    )

    x_master = np.asarray(x_master, dtype=np.float64).reshape(-1)
    if objective_epigraph_available is None:
        # Production masters only carry an objective epigraph for convex objectives.
        # Tests and future heuristic masters may explicitly opt in to exercise the
        # local-objective guard without changing the current master layout.
        objective_epigraph_available = bool(objective_is_convex)
    trace: dict[str, object] = {
        "attempted": True,
        "fallback_used": False,
        "fallback_reason": None,
        "rootsearch": None,
        "candidate_hyperplanes": 0,
        "selected_hyperplanes": 0,
        "cuts_added": 0,
        "local_cuts_added": 0,
        "local_cuts_rejected": 0,
    }

    def fallback(reason: str) -> tuple[int, dict[str, object]]:
        trace["fallback_used"] = True
        trace["fallback_reason"] = reason
        added = _add_ecp_cuts(
            evaluator,
            x_master,
            n_vars,
            constraint_senses,
            oa_A_rows,
            oa_b_rows,
            obj_is_linear,
            constraint_convex_mask,
            objective_is_convex,
            equality_relaxation=equality_relaxation,
            oa_cut_relaxable=oa_cut_relaxable,
            cut_provenance=cut_provenance,
            include_local_cuts=True,
            incumbent=incumbent,
            local_cut_trace=trace,
        )
        trace["cuts_added"] = int(added)
        return added, trace

    if interior_point_store is None:
        return fallback("missing_interior_point_store")

    root_result = rootsearch_from_store(
        evaluator,
        x_master,
        interior_point_store,
        strategy=rootsearch_strategy,
        fixed_discrete=True,
        constraint_senses=constraint_senses,
    )
    trace["rootsearch"] = root_result.as_trace_dict()
    if root_result.status is MIPNLPRootSearchStatus.CANDIDATE_FEASIBLE:
        return fallback(root_result.status.value)
    if root_result.status is not MIPNLPRootSearchStatus.CONVERGED or root_result.point is None:
        return fallback(root_result.status.value)

    support = np.asarray(root_result.point, dtype=np.float64).reshape(-1)
    master_violations, _master_signs = _constraint_violation_data(evaluator, x_master)
    generated = generate_oa_cuts_from_evaluator(
        evaluator,
        support,
        constraint_senses=constraint_senses,
        convex_mask=None,
    )
    constraint_ids = _constraint_ids_for_generated_oa_cuts(
        evaluator,
        support,
        constraint_senses,
        convex_mask=None,
        violated_only=False,
    )
    candidates: list[_ESHHyperplaneCandidate] = []
    local_rejected = 0

    def add_constraint_candidate(
        coeffs,
        rhs: float,
        *,
        constraint_id: int,
        global_valid: bool,
    ) -> None:
        nonlocal local_rejected
        coeffs_arr = np.asarray(coeffs, dtype=np.float64).copy()
        if np.linalg.norm(coeffs_arr) < 1e-12:
            return
        violation = _candidate_cut_violation(coeffs_arr, rhs, x_master)
        if violation <= 1e-8:
            return
        if (
            not global_valid
            and incumbent is not None
            and _candidate_cut_excludes_point(coeffs_arr, rhs, incumbent)
        ):
            local_rejected += 1
            return
        candidates.append(
            _ESHHyperplaneCandidate(
                coeffs=coeffs_arr,
                rhs=float(rhs),
                relaxable=True,
                source="esh",
                global_valid=bool(global_valid),
                local_valid=True,
                supporting_point=support,
                violation=float(violation),
                constraint_id=int(constraint_id),
            )
        )

    for constraint_id, cut in zip(constraint_ids, generated):
        if constraint_id >= len(master_violations) or master_violations[constraint_id] <= 1e-8:
            continue
        original_sense = cut.sense
        sense = "<=" if equality_relaxation and cut.sense == "==" else cut.sense
        global_valid = _constraint_cut_global_valid(
            constraint_convex_mask,
            constraint_id,
            original_sense,
            equality_relaxation,
        )
        if sense == "<=":
            add_constraint_candidate(
                cut.coeffs,
                cut.rhs,
                constraint_id=constraint_id,
                global_valid=global_valid,
            )
        elif sense == ">=":
            add_constraint_candidate(
                -cut.coeffs,
                -cut.rhs,
                constraint_id=constraint_id,
                global_valid=global_valid,
            )
        elif sense == "==":
            add_constraint_candidate(
                cut.coeffs,
                cut.rhs,
                constraint_id=constraint_id,
                global_valid=global_valid,
            )
            add_constraint_candidate(
                -cut.coeffs,
                -cut.rhs,
                constraint_id=constraint_id,
                global_valid=global_valid,
            )

    if not obj_is_linear and objective_epigraph_available:
        n_master = n_vars + 1
        objective_global_valid = bool(objective_is_convex)
        obj_value = float(evaluator.evaluate_objective(support))
        obj_support = np.concatenate([support, [obj_value]])
        obj_cut = generate_objective_oa_cut(evaluator, support, n_master, z_index=n_vars)
        tangent_at_master = float(np.dot(obj_cut.coeffs[:n_vars], x_master) - obj_cut.rhs)
        objective_gap = max(0.0, float(evaluator.evaluate_objective(x_master)) - tangent_at_master)
        if np.linalg.norm(obj_cut.coeffs[:n_vars]) >= 1e-12 and objective_gap > 1e-8:
            incumbent_point = None
            if not objective_global_valid and incumbent is not None:
                incumbent_arr = np.asarray(incumbent, dtype=np.float64).reshape(-1)
                if incumbent_arr.size == n_vars:
                    if incumbent_obj is None:
                        incumbent_obj_value = float(evaluator.evaluate_objective(incumbent_arr))
                    else:
                        incumbent_obj_value = float(incumbent_obj)
                    incumbent_point = np.concatenate([incumbent_arr, [incumbent_obj_value]])
            if (
                not objective_global_valid
                and incumbent_point is not None
                and _candidate_cut_excludes_point(obj_cut.coeffs, obj_cut.rhs, incumbent_point)
            ):
                local_rejected += 1
            else:
                candidates.append(
                    _ESHHyperplaneCandidate(
                        coeffs=obj_cut.coeffs.copy(),
                        rhs=float(obj_cut.rhs),
                        relaxable=False,
                        source="objective_rootsearch",
                        global_valid=objective_global_valid,
                        local_valid=True,
                        supporting_point=obj_support,
                        violation=float(objective_gap),
                        objective_id="objective",
                    )
                )

    trace["candidate_hyperplanes"] = int(len(candidates))
    trace["local_cuts_rejected"] = int(local_rejected)
    selected = _select_hyperplane_candidates(
        candidates,
        max_per_iter=hyperplane_max_per_iter,
        selection_factor=hyperplane_selection_factor,
    )
    trace["selected_hyperplanes"] = int(len(selected))

    local_added = 0
    for item in selected:
        _append_master_cut(
            oa_A_rows,
            oa_b_rows,
            item.coeffs,
            item.rhs,
            oa_cut_relaxable,
            relaxable=item.relaxable,
            cut_provenance=cut_provenance,
            source=item.source,
            global_valid=item.global_valid,
            local_valid=item.local_valid,
            supporting_point=item.supporting_point,
            violation=item.violation,
            constraint_id=item.constraint_id,
            objective_id=item.objective_id,
        )
        if not item.global_valid:
            local_added += 1

    trace["cuts_added"] = int(len(selected))
    trace["local_cuts_added"] = int(local_added)
    return len(selected), trace


def _add_no_good_cut(
    x_master,
    binary_indices,
    oa_A_rows,
    oa_b_rows,
    n_vars,
    oa_cut_relaxable=None,
    cut_provenance: Optional[MIPNLPCutProvenance] = None,
    integer_binary_expansion: Optional[_IntegerBinaryExpansion] = None,
):
    """Add a binary-assignment exclusion (no-good) cut.

    sum_{i: y_i*=1} (1-y_i) + sum_{i: y_i*=0} y_i >= 1
    Equivalently in <= form:
    sum_{y_i*=1} y_i - sum_{y_i*=0} y_i <= count(y_i*=1) - 1
    """
    if integer_binary_expansion is not None and integer_binary_expansion.bit_count > 0:
        cut_point = integer_binary_expansion.logical_point(x_master)
        encoded_binary_indices = (
            list(binary_indices) + integer_binary_expansion.logical_binary_indices
        )
        n_cut_vars = integer_binary_expansion.logical_width
    else:
        cut_point = np.asarray(x_master, dtype=np.float64).ravel()
        encoded_binary_indices = list(binary_indices)
        n_cut_vars = n_vars

    if not encoded_binary_indices:
        return False

    coeffs = np.zeros(n_cut_vars, dtype=np.float64)
    count_ones = 0
    for idx in encoded_binary_indices:
        val = _round_integral_to_bounds(cut_point[idx], 0.0, 1.0)
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
        supporting_point=cut_point,
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


def _primal_reduction_cutoff(value: float) -> float:
    """Return a strict improvement cutoff in the internal minimization convention."""
    return float(value) - 1e-6 * (1.0 + abs(float(value)))


def _add_primal_reduction_cut(
    decomp: "_DecomposedProblem",
    incumbent,
    incumbent_obj: Optional[float],
    oa_A_rows,
    oa_b_rows,
    oa_cut_relaxable=None,
    cut_provenance: Optional[MIPNLPCutProvenance] = None,
) -> dict[str, object]:
    """Add a SHOT-style objective reduction cut when the master row is exact."""
    trace: dict[str, object] = {
        "status": "skipped",
        "reason": None,
        "source": "reduction",
        "global_valid": False,
        "local_valid": True,
        "cutoff": None,
        "incumbent_objective": None,
    }
    if incumbent is None or incumbent_obj is None:
        trace["reason"] = "no_incumbent"
        return trace
    if not decomp.obj_is_linear or decomp.obj_coeffs is None:
        trace["reason"] = "nonlinear_objective_without_certified_epigraph"
        return trace

    c_vec, obj_offset = decomp.obj_coeffs
    coeffs = np.asarray(c_vec, dtype=np.float64).reshape(-1).copy()
    if coeffs.size != decomp.n_vars or np.linalg.norm(coeffs) < 1e-12:
        trace["reason"] = "missing_linear_objective_row"
        return trace

    incumbent_master_obj = float(incumbent_obj) - float(obj_offset)
    cutoff = _primal_reduction_cutoff(incumbent_master_obj)
    _append_master_cut(
        oa_A_rows,
        oa_b_rows,
        coeffs,
        cutoff,
        oa_cut_relaxable,
        relaxable=False,
        cut_provenance=cut_provenance,
        source="reduction",
        global_valid=False,
        local_valid=True,
        supporting_point=np.asarray(incumbent, dtype=np.float64).reshape(-1)[: decomp.n_vars],
        objective_id="objective_cutoff",
    )
    trace.update(
        {
            "status": "added",
            "reason": None,
            "cutoff": float(cutoff),
            "incumbent_objective": float(incumbent_master_obj),
        }
    )
    return trace


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
    integer_binary_expansion: Optional[_IntegerBinaryExpansion] = None,
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
    integer_binary_start = None
    if integer_binary_expansion is not None and integer_binary_expansion.bit_count > 0:
        integer_binary_start = n_master
        n_master += integer_binary_expansion.bit_count

    # Build A_ub, b_ub from linear <= constraints + OA cuts
    A_ub_rows = []
    b_ub_vals = []

    for i, sense in enumerate(linear_senses):
        row = _stored_row_to_master_layout(
            linear_A_rows[i],
            n_vars=n_vars,
            n_master=n_master,
            use_objective_epigraph=bool(use_objective_epigraph),
            slack_index=None,
            relaxable=False,
            integer_binary_expansion=integer_binary_expansion,
            integer_binary_start=integer_binary_start,
        )
        if sense == "<=":
            A_ub_rows.append(row)
            b_ub_vals.append(linear_b_rows[i])
        elif sense == ">=":
            A_ub_rows.append(-row)
            b_ub_vals.append(-linear_b_rows[i])

    # OA cuts (all in <= form already)
    # Constraint cuts have length n_vars; objective cuts carry the eta column.
    for i in range(len(oa_A_rows)):
        original_len = len(np.asarray(oa_A_rows[i], dtype=np.float64).ravel())
        # Relax only constraint OA/feasibility cuts. Objective epigraph cuts and
        # hard integer-exclusion cuts must remain unrelaxed.
        if oa_cut_relaxable is None:
            relax_cut = original_len == n_vars
        else:
            relax_cut = bool(oa_cut_relaxable[i])
        row = _stored_row_to_master_layout(
            oa_A_rows[i],
            n_vars=n_vars,
            n_master=n_master,
            use_objective_epigraph=bool(use_objective_epigraph),
            slack_index=slack_index,
            relaxable=relax_cut,
            integer_binary_expansion=integer_binary_expansion,
            integer_binary_start=integer_binary_start,
        )
        A_ub_rows.append(row)
        b_ub_vals.append(oa_b_rows[i])

    # Equality constraints from linear
    A_eq_rows = []
    b_eq_vals = []
    for i, sense in enumerate(linear_senses):
        if sense == "==":
            row = _stored_row_to_master_layout(
                linear_A_rows[i],
                n_vars=n_vars,
                n_master=n_master,
                use_objective_epigraph=bool(use_objective_epigraph),
                slack_index=None,
                relaxable=False,
                integer_binary_expansion=integer_binary_expansion,
                integer_binary_start=integer_binary_start,
            )
            A_eq_rows.append(row)
            b_eq_vals.append(linear_b_rows[i])
    _append_integer_binary_link_rows(
        A_eq_rows,
        b_eq_vals,
        n_master=n_master,
        integer_binary_expansion=integer_binary_expansion,
        integer_binary_start=integer_binary_start,
    )

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
    if integer_binary_expansion is not None and integer_binary_expansion.bit_count > 0:
        bounds_list.extend((0.0, 1.0) for _ in range(integer_binary_expansion.bit_count))

    # Integrality
    int_vec = np.zeros(n_master, dtype=np.int32)
    int_vec[:n_vars] = integrality
    if (
        integer_binary_expansion is not None
        and integer_binary_expansion.bit_count > 0
        and integer_binary_start is not None
    ):
        for spec in integer_binary_expansion.variables:
            bit_integrality = int(integrality[spec.index])
            int_vec[
                integer_binary_start + spec.bit_start : integer_binary_start
                + spec.bit_start
                + spec.bit_count
            ] = bit_integrality

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
        integer_binary_expansion=integer_binary_expansion,
        integer_binary_start=integer_binary_start,
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
    mip_start=None,
    mip_start_objective: Optional[float] = None,
    objective_cutoff: Optional[float] = None,
    mip_solution_limit: Optional[int] = None,
    integer_binary_expansion: Optional[_IntegerBinaryExpansion] = None,
):
    """Build and solve the master MILP."""
    try:
        gurobi_controls = (
            objective_cutoff is not None or mip_solution_limit is not None or mip_start is not None
        )
        if solution_pool or (gurobi_controls and _shot_master_feature_supported(milp_solver)):
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
        integer_binary_expansion=integer_binary_expansion,
    )

    full_mip_start = None
    if _shot_master_feature_supported(milp_solver):
        full_mip_start = _extend_master_mip_start(
            master,
            n_vars=n_vars,
            mip_start=mip_start,
            mip_start_objective=mip_start_objective,
        )
    gurobi_options: dict[str, object] = {}
    if _shot_master_feature_supported(milp_solver):
        if objective_cutoff is not None:
            gurobi_options["Cutoff"] = float(objective_cutoff)
        if mip_solution_limit is not None:
            gurobi_options["SolutionLimit"] = max(1, int(mip_solution_limit))
        if solution_pool:
            gurobi_options.update(
                {
                    "PoolSearchMode": 2,
                    "PoolSolutions": max(1, int(num_solution_iteration)),
                }
            )

    solve_kwargs: dict[str, Any] = {
        "c": master.c,
        "A_ub": master.A_ub,
        "b_ub": master.b_ub,
        "A_eq": master.A_eq,
        "b_eq": master.b_eq,
        "bounds": master.bounds,
        "integrality": master.integrality,
        "time_limit": time_limit,
        "gap_tolerance": gap_tolerance,
    }
    if gurobi_options:
        solve_kwargs["options"] = gurobi_options
    if full_mip_start is not None:
        solve_kwargs["mip_start"] = full_mip_start
    if solution_pool:
        solve_kwargs["solution_pool"] = True
        solve_kwargs["num_solution_iteration"] = max(1, int(num_solution_iteration))

    solve_milp_any: Any = solve_milp
    return solve_milp_any(**solve_kwargs)


def _global_valid_master_cut_rows(
    cut_provenance: MIPNLPCutProvenance,
) -> tuple[list[np.ndarray], list[float], int, int]:
    """Return globally valid provenance rows for the certified-bound master."""
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    local_excluded = 0
    integer_excluded = 0
    for record in cut_provenance.records:
        if not record.global_valid:
            local_excluded += 1
            continue
        if record.source == "integer":
            integer_excluded += 1
            continue
        coeffs = np.asarray(record.coefficients, dtype=np.float64)
        rows.append(coeffs)
        rhs.append(float(record.rhs))
    return rows, rhs, local_excluded, integer_excluded


def _solve_initial_poa_master(
    decomp: _DecomposedProblem,
    oa_A_rows,
    oa_b_rows,
    *,
    master_bound_valid: bool,
    time_limit: float,
    gap_tolerance: float,
    add_slack: bool,
    max_slack: float,
    oa_penalty_factor: float,
    oa_cut_relaxable,
    milp_solver: str,
    integer_binary_expansion: Optional[_IntegerBinaryExpansion] = None,
):
    """Solve the current OA master with integrality relaxed for initial POA seeding."""
    relaxed_integrality = np.zeros_like(decomp.integrality, dtype=np.int32)
    return _solve_master_milp(
        decomp.linear_A_rows,
        decomp.linear_b_rows,
        decomp.linear_senses,
        oa_A_rows,
        oa_b_rows,
        decomp.n_vars,
        relaxed_integrality,
        decomp.lb,
        decomp.ub,
        decomp.obj_coeffs,
        decomp.obj_is_linear,
        master_bound_valid,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        add_slack=add_slack,
        max_slack=max_slack,
        oa_penalty_factor=oa_penalty_factor,
        oa_cut_relaxable=oa_cut_relaxable,
        use_objective_epigraph=(not decomp.obj_is_linear and decomp.oa_objective_is_convex),
        milp_solver=milp_solver,
        solution_pool=False,
        num_solution_iteration=1,
        integer_binary_expansion=integer_binary_expansion,
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
    integer_binary_expansion: Optional[_IntegerBinaryExpansion] = None,
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
    integer_binary_start = None
    if integer_binary_expansion is not None and integer_binary_expansion.bit_count > 0:
        integer_binary_start = n_base
        n_base += integer_binary_expansion.bit_count

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

    def base_row(coeffs, *, relaxable: bool = False, slack: bool = False) -> np.ndarray:
        return _stored_row_to_master_layout(
            coeffs,
            n_vars=n_vars,
            n_master=n_master,
            use_objective_epigraph=bool(use_objective_epigraph),
            slack_index=slack_index if slack else None,
            relaxable=relaxable,
            integer_binary_expansion=integer_binary_expansion,
            integer_binary_start=integer_binary_start,
        )

    for row, rhs, sense in zip(
        decomp.linear_A_rows,
        decomp.linear_b_rows,
        decomp.linear_senses,
    ):
        master_row = base_row(row)
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
        original_len = len(np.asarray(cut_row, dtype=np.float64).ravel())
        if oa_cut_relaxable is None:
            relax_cut = original_len == n_vars
        else:
            relax_cut = bool(oa_cut_relaxable[i])
        master_row = base_row(cut_row, relaxable=relax_cut, slack=True)
        a_ub_rows.append(master_row)
        b_ub_vals.append(float(oa_b_rows[i]))

    _append_integer_binary_link_rows(
        a_eq_rows,
        b_eq_vals,
        n_master=n_master,
        integer_binary_expansion=integer_binary_expansion,
        integer_binary_start=integer_binary_start,
    )

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
    if integer_binary_expansion is not None and integer_binary_expansion.bit_count > 0:
        bounds.extend((0.0, 1.0) for _ in range(integer_binary_expansion.bit_count))
    if add_regularization == "level_L1":
        bounds.extend((0.0, 1e20) for _ in range(n_vars))
    elif add_regularization == "level_L_infinity":
        bounds.append((0.0, 1e20))

    integrality = np.zeros(n_master, dtype=np.int32)
    integrality[:n_vars] = decomp.integrality
    if (
        integer_binary_expansion is not None
        and integer_binary_expansion.bit_count > 0
        and integer_binary_start is not None
    ):
        for spec in integer_binary_expansion.variables:
            bit_integrality = int(decomp.integrality[spec.index])
            integrality[
                integer_binary_start + spec.bit_start : integer_binary_start
                + spec.bit_start
                + spec.bit_count
            ] = bit_integrality
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
    return _stored_row_to_master_layout(
        row,
        n_vars=n_vars,
        n_master=len(master.c),
        use_objective_epigraph=master.use_objective_epigraph,
        slack_index=master.slack_index,
        relaxable=relaxable,
        integer_binary_expansion=master.integer_binary_expansion,
        integer_binary_start=master.integer_binary_start,
    )


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
    integer_to_binary: bool = False,
    mip_nlp_profile: str = "default",
    mip_nlp_shot_config: Optional[MIPNLPShotConfig] = None,
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
            "integer_to_binary, "
            "max_slack, milp_solver, mip_nlp_profile, mip_nlp_shot_config, "
            "oa_penalty_factor."
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
    integer_to_binary = bool(integer_to_binary)
    _warn_integer_to_binary_noop(
        "LP/NLP BB",
        integer_to_binary=integer_to_binary,
        add_no_good_cuts=add_no_good_cuts,
    )

    decomp = _decompose_model(model)
    integer_binary_expansion = _build_integer_binary_expansion(
        decomp,
        enabled=bool(integer_to_binary and add_no_good_cuts),
    )
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
    shot_config = mip_nlp_shot_config if mip_nlp_profile == "shot" else None
    shot_profile = shot_config is not None
    shot_cut_strategy = shot_config.cut_strategy if shot_config is not None else "oa"
    cut_provenance = MIPNLPCutProvenance()
    callback_events: list[dict[str, object]] = []

    if shot_profile:
        from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

        interior_point_store = MIPNLPInteriorPointStore(
            n_vars,
            int_indices=decomp.int_indices,
            lb=decomp.lb,
            ub=decomp.ub,
        )
    else:
        interior_point_store = None

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

    def _linear_objective_offset() -> float:
        if decomp.obj_is_linear and decomp.obj_coeffs is not None:
            return float(decomp.obj_coeffs[1])
        return 0.0

    def _master_objective_from_evaluator(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return float(value) - _linear_objective_offset()

    def _record_interior_point(
        point,
        source: str,
        metadata: Optional[dict[str, object]] = None,
        *,
        require_feasible: bool = False,
    ) -> None:
        if interior_point_store is None:
            return
        interior_point_store.add(
            point,
            source=source,
            metadata=metadata,
            evaluator=evaluator,
            constraint_senses=decomp.constraint_senses,
            require_feasible=require_feasible,
        )

    def _record_callback_event(
        *,
        context: str,
        cuts_start: int,
        provenance_start: int,
        cuts_returned: int,
        fixed_nlp_status: Optional[str] = None,
        rootsearch_trace: Optional[dict[str, object]] = None,
        integer_cut_added: bool = False,
    ) -> None:
        event: dict[str, object] = {
            "context": context,
            "cuts_generated": int(len(oa_A_rows) - cuts_start),
            "cuts_returned": int(cuts_returned),
            "provenance_cuts_added": int(len(cut_provenance.records) - provenance_start),
            "provenance_cuts_total": int(len(cut_provenance.records)),
            "fixed_nlp_status": fixed_nlp_status,
            "integer_cut_added": bool(integer_cut_added),
            "cut_source_counts": cut_provenance.source_counts(),
        }
        if rootsearch_trace is not None:
            event["rootsearch"] = rootsearch_trace
        callback_events.append(event)

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
            _record_interior_point(
                incumbent,
                "callback_incumbent",
                {"objective": float(obj)},
                require_feasible=True,
            )

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
            cut_provenance=cut_provenance,
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
            if obj_relax is not None:
                _record_interior_point(
                    x_relax,
                    "nlp_relaxation",
                    {"objective": float(obj_relax)},
                    require_feasible=True,
                )
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
            _record_interior_point(
                x_init,
                "initial_fixed_nlp",
                {"objective": float(obj_init)},
                require_feasible=True,
            )
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
        integer_binary_expansion=integer_binary_expansion,
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
        cuts_start = len(oa_A_rows)
        provenance_start = len(cut_provenance.records)
        start = len(oa_A_rows)
        rootsearch_trace = None
        if (
            shot_profile
            and shot_cut_strategy in {"auto", "esh"}
            and interior_point_store is not None
        ):
            assert shot_config is not None
            _esh_added, rootsearch_trace = _add_esh_cuts(
                evaluator,
                x_master,
                n_vars,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                interior_point_store,
                rootsearch_strategy=shot_config.rootsearch_strategy,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
                cut_provenance=cut_provenance,
                incumbent=incumbent,
                incumbent_obj=incumbent_obj,
                hyperplane_max_per_iter=shot_config.hyperplane_max_per_iter,
                hyperplane_selection_factor=shot_config.hyperplane_selection_factor,
            )
        nlp_subproblem_count += 1
        fixed_nlp_status = "failed"
        integer_cut_added = False
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
            fixed_nlp_status = "feasible"
            if obj_nlp is not None:
                accept_incumbent(x_nlp, obj_nlp)
                _record_interior_point(
                    x_nlp,
                    "callback_fixed_nlp",
                    {"objective": float(obj_nlp)},
                    require_feasible=True,
                )
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
                        cut_provenance=cut_provenance,
                    )
            if add_no_good_cuts and (
                not decomp.general_integer_indices or integer_binary_expansion is not None
            ):
                integer_cut_added = _add_no_good_cut(
                    x_master,
                    decomp.binary_indices,
                    oa_A_rows,
                    oa_b_rows,
                    n_vars,
                    oa_cut_relaxable=oa_cut_relaxable,
                    integer_binary_expansion=integer_binary_expansion,
                    cut_provenance=cut_provenance,
                )
            add_oa_cuts_at(x_master)

        rows = collect_new_lazy_cuts(start, np.asarray(master_x, dtype=np.float64))
        _record_callback_event(
            context="mipsol",
            cuts_start=cuts_start,
            provenance_start=provenance_start,
            cuts_returned=len(rows),
            fixed_nlp_status=fixed_nlp_status,
            rootsearch_trace=rootsearch_trace,
            integer_cut_added=integer_cut_added,
        )
        return rows

    def node_callback(master_x: np.ndarray) -> list[tuple[np.ndarray, float]]:
        full_master_x = np.asarray(master_x, dtype=np.float64)
        x_master = full_master_x[:n_vars]
        cuts_start = len(oa_A_rows)
        provenance_start = len(cut_provenance.records)
        start = len(oa_A_rows)
        _add_ecp_cuts(
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
        rows = collect_new_lazy_cuts(start, full_master_x)
        _record_callback_event(
            context="mipnode",
            cuts_start=cuts_start,
            provenance_start=provenance_start,
            cuts_returned=len(rows),
        )
        return rows

    from discopt.solvers import SolveStatus
    from discopt.solvers.gurobi import solve_milp_with_lazy_cuts

    master_mip_start = None
    if shot_profile and incumbent is not None:
        master_mip_start = _extend_master_mip_start(
            master,
            n_vars=n_vars,
            mip_start=incumbent,
            mip_start_objective=_master_objective_from_evaluator(incumbent_obj),
        )

    def callback_terminate(_snapshot: dict[str, object]) -> bool:
        return (time.perf_counter() - t_start) >= float(time_limit)

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
        node_callback=node_callback if shot_profile else None,
        terminate_callback=callback_terminate,
        mip_start=master_mip_start,
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
    callback_stats = dict(master_result.callback_stats or {})
    callback_terminated = bool(callback_stats.get("terminated"))
    status = "feasible"
    termination_reason: Optional[str] = None
    if callback_terminated:
        termination_reason = "time_limit"
        status = "time_limit" if incumbent is None else "feasible"
    elif master_result.status == SolveStatus.INFEASIBLE:
        status = "infeasible"
    elif master_result.status == SolveStatus.TIME_LIMIT:
        status = "time_limit" if incumbent is None else "feasible"
    elif master_result.status == SolveStatus.ITERATION_LIMIT:
        status = "iteration_limit" if incumbent is None else "feasible"
    elif master_result.status == SolveStatus.OPTIMAL and incumbent is not None:
        status = "optimal" if gap is not None and gap <= gap_tolerance else "feasible"
    elif incumbent is None:
        status = "no_feasible_point"
    if termination_reason is None:
        termination_reason = status

    trace_bound_validity = (
        "global"
        if master_bound_valid and bound is not None
        else ("heuristic" if bound is not None else "unavailable")
    )
    single_tree_trace: dict[str, object] = {
        "schema_version": 1,
        "solver": "mip-nlp",
        "method": "lp_nlp_bb",
        "profile": mip_nlp_profile,
        "shot_options": (
            mip_nlp_shot_config.as_trace_dict() if mip_nlp_shot_config is not None else {}
        ),
        "iterations": [
            {
                "index": 0,
                "master_status": _trace_status(master_result.status),
                "lb": _trace_value(bound),
                "ub": _trace_value(incumbent_obj),
                "gap": _trace_value(gap),
                "cuts_total": int(len(oa_A_rows)),
                "provenance_cuts_total": int(len(cut_provenance.records)),
                "cut_source_counts": cut_provenance.source_counts(),
                "callback_events": callback_events,
                "callback_stats": callback_stats,
                "node_count": int(master_result.node_count),
                "mip_start_applied": bool(master_mip_start is not None),
            }
        ],
        "summary": {
            "mip_count": 1,
            "nlp_subproblem_count": int(nlp_subproblem_count),
            "cut_count": int(len(oa_A_rows)),
            "provenance_cut_count": int(len(cut_provenance.records)),
            "cut_source_counts": cut_provenance.source_counts(),
            "callback_event_count": int(len(callback_events)),
            "callback_stats": callback_stats,
            "node_count": int(master_result.node_count),
        },
        "termination_reason": termination_reason,
        "master_bound_valid": bool(master_bound_valid),
        "gap_certified": bool(master_bound_valid),
        "bound_validity": trace_bound_validity,
        "final_lb": _trace_value(bound),
        "final_ub": _trace_value(incumbent_obj),
        "final_gap": _trace_value(gap),
    }

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
            mip_nlp_trace=single_tree_trace,
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
        mip_nlp_trace=single_tree_trace,
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
    integer_to_binary: bool = False,
    mip_nlp_profile: str = "default",
    mip_nlp_shot_config: Optional[MIPNLPShotConfig] = None,
    external_primal_candidate_hook: Any = None,
    external_hyperplane_hook: Any = None,
    external_dual_bound_hook: Any = None,
    termination_hook: Any = None,
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
    integer_to_binary : bool
        When true, bounded general-integer variables get a linked binary
        expansion in the OA master so no-good cuts can exclude assignments over
        generated binary variables. Unbounded or impractically bounded general
        integers raise a diagnostic when this option is combined with
        ``add_no_good_cuts``.
    external_primal_candidate_hook, external_hyperplane_hook, external_dual_bound_hook,
    termination_hook : callable, optional
        Opt-in event hooks for the multi-tree OA loop. Hooks receive a read-only
        context dictionary with iteration, elapsed time, current bound/incumbent
        data, and candidate points where relevant. Returned payloads are
        validated before they can add external fixed-NLP candidates, master cuts,
        dual-bound updates, or request user termination.

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
    external_primal_candidate_hook = _normalize_optional_hook(
        "external_primal_candidate_hook",
        external_primal_candidate_hook,
    )
    external_hyperplane_hook = _normalize_optional_hook(
        "external_hyperplane_hook",
        external_hyperplane_hook,
    )
    external_dual_bound_hook = _normalize_optional_hook(
        "external_dual_bound_hook",
        external_dual_bound_hook,
    )
    termination_hook = _normalize_optional_hook("termination_hook", termination_hook)
    heuristic_nonconvex = bool(heuristic_nonconvex)
    solution_pool = bool(solution_pool)
    shot_solution_pool_degraded_reason: Optional[str] = None
    if mip_nlp_profile == "shot" and mip_nlp_shot_config is not None:
        if mip_nlp_shot_config.solution_pool_capacity is not None:
            num_solution_iteration = int(mip_nlp_shot_config.solution_pool_capacity)
        shot_pool_requested = (
            mip_nlp_shot_config.fixed_nlp_strategy == "solution_pool"
            or mip_nlp_shot_config.solution_pool_capacity is not None
        )
        if shot_pool_requested and not solution_pool:
            if _shot_master_feature_supported(milp_solver):
                solution_pool = True
            else:
                shot_solution_pool_degraded_reason = (
                    "fixed_nlp_strategy='solution_pool' requires milp_solver='gurobi'"
                )
                logger.warning(
                    "OA: SHOT solution-pool request ignored for milp_solver=%r; "
                    "only the Gurobi backend exposes solution-pool candidates.",
                    milp_solver,
                )
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
    integer_to_binary = bool(integer_to_binary)
    _warn_integer_to_binary_noop(
        "OA",
        integer_to_binary=integer_to_binary,
        add_no_good_cuts=add_no_good_cuts,
    )

    # 1. Decompose model
    decomp = _decompose_model(model)
    integer_binary_expansion = _build_integer_binary_expansion(
        decomp,
        enabled=bool(integer_to_binary and add_no_good_cuts),
    )
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
    local_cut_added = False

    # 2. Generate initial linearization cuts.
    oa_A_rows: list[np.ndarray] = []
    oa_b_rows: list[float] = []
    oa_cut_relaxable: list[bool] = []
    cut_provenance = MIPNLPCutProvenance()

    UB = 1e20
    LB = -1e20
    certified_LB = -1e20
    heuristic_LB = -1e20
    certified_bound_source: Optional[str] = None
    heuristic_bound_source: Optional[str] = None
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
    initial_poa_trace: Optional[dict[str, object]] = None
    interior_point_store = None
    shot_cut_strategy = "auto"
    shot_solution_limit_state: Optional[_ShotMIPSolutionLimitState] = None
    shot_unsupported_backend_features: set[str] = set()
    shot_master_backend_supported = _shot_master_feature_supported(milp_solver)
    if shot_solution_pool_degraded_reason is not None:
        shot_unsupported_backend_features.add("solution_pool")
    if mip_nlp_profile == "shot":
        from discopt.solvers.mip_nlp_rootsearch import MIPNLPInteriorPointStore

        if mip_nlp_shot_config is not None:
            shot_cut_strategy = mip_nlp_shot_config.cut_strategy
            shot_solution_limit_state = _ShotMIPSolutionLimitState(
                strategy=mip_nlp_shot_config.mip_solution_limit_strategy,
                capacity=num_solution_iteration,
                backend=str(milp_solver).lower(),
            )
            if shot_solution_limit_state.degraded_reason is not None:
                shot_unsupported_backend_features.add("mip_solution_limit")
        interior_point_store = MIPNLPInteriorPointStore(
            n_vars,
            int_indices=decomp.int_indices,
            lb=decomp.lb,
            ub=decomp.ub,
        )
        phase = mip_nlp_shot_config.relaxation_phase if mip_nlp_shot_config is not None else "off"
        initial_poa_enabled = mip_nlp_shot_config is not None and phase in _INITIAL_POA_PHASES
        initial_poa_trace = {
            "enabled": bool(initial_poa_enabled),
            "phase": phase,
            "attempted": False,
            "status": "pending" if initial_poa_enabled else "disabled",
            "fallback_reason": None if initial_poa_enabled else f"relaxation_phase={phase}",
            "cuts_added": 0,
            "provenance_cuts_added": 0,
            "objective_bound": None,
            "objective_bound_valid": False,
            "interior_point_candidates": 0,
            "interior_points_stored": 0,
            "node_count": 0,
        }
    fixed_nlp_strategy = "always"
    if mip_nlp_profile == "shot" and mip_nlp_shot_config is not None:
        fixed_nlp_strategy = mip_nlp_shot_config.fixed_nlp_strategy
    fixed_nlp_manager = FixedNLPCandidateManager(
        n_vars=n_vars,
        int_indices=decomp.int_indices,
        lb=decomp.lb,
        ub=decomp.ub,
        strategy=fixed_nlp_strategy,
        candidate_limit=num_solution_iteration,
        deduplicate_used_assignments=(mip_nlp_profile == "shot"),
    )
    fixed_nlp_call_count = 0
    fixed_nlp_call_source_counts: Counter[str] = Counter()
    fixed_nlp_call_status_counts: Counter[str] = Counter()
    external_hook_call_counts: Counter[str] = Counter()
    external_hook_accept_counts: Counter[str] = Counter()
    external_hook_reject_counts: Counter[str] = Counter()
    external_hook_error_counts: Counter[str] = Counter()
    repaired_assignment_keys: set[tuple[float, ...]] = set()
    active_reduction_cut_indices: set[int] = set()
    reduction_cut_incumbent_key: Optional[float] = None

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

    def _linear_objective_offset() -> float:
        if decomp.obj_is_linear and decomp.obj_coeffs is not None:
            return float(decomp.obj_coeffs[1])
        return 0.0

    def _master_objective_from_evaluator(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return float(value) - _linear_objective_offset()

    def _evaluator_objective_from_master(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return float(value) + _linear_objective_offset()

    def _promote_certified_bound(value: Optional[float], source: str) -> bool:
        nonlocal LB, certified_LB, certified_bound_source
        traced = _trace_value(value)
        if traced is None:
            return False
        previous = certified_LB
        if traced > certified_LB:
            certified_LB = float(traced)
            certified_bound_source = str(source)
        if traced > LB:
            LB = float(traced)
        return certified_LB > previous + 1e-12

    def _record_heuristic_bound(value: Optional[float], source: str) -> bool:
        nonlocal LB, heuristic_LB, heuristic_bound_source
        traced = _trace_value(value)
        if traced is None:
            return False
        previous = heuristic_LB
        if traced > heuristic_LB:
            heuristic_LB = float(traced)
            heuristic_bound_source = str(source)
        if traced > LB:
            LB = float(traced)
        return heuristic_LB > previous + 1e-12

    def _certified_gap_value() -> Optional[float]:
        if _trace_value(certified_LB) is None or _trace_value(UB) is None:
            return None
        return _compute_gap(certified_LB, UB)

    def _certified_gap_converged() -> bool:
        gap_value = _certified_gap_value()
        return bool(gap_value is not None and gap_value <= gap_tolerance)

    def _cut_source_delta(before: dict[str, int]) -> dict[str, int]:
        after = cut_provenance.source_counts()
        return {source: int(after.get(source, 0) - before.get(source, 0)) for source in after}

    def _user_bound(value: Optional[float]) -> Optional[float]:
        traced = _trace_value(value)
        return None if traced is None else float(_obj_sign * traced)

    def _absolute_gap_value() -> Optional[float]:
        if _trace_value(LB) is None or _trace_value(UB) is None:
            return None
        return abs(float(UB) - float(LB))

    def _external_hook_context(
        event: str,
        *,
        iteration: int,
        master_point: Optional[np.ndarray] = None,
        solution_points: Optional[list[np.ndarray]] = None,
    ) -> dict[str, object]:
        points = [] if solution_points is None else solution_points
        return {
            "event": event,
            "iteration": int(iteration),
            "elapsed": float(time.perf_counter() - t_start),
            "is_minimization": bool(_obj_sign > 0),
            "current_dual_bound": _user_bound(LB),
            "current_primal_bound": _user_bound(UB),
            "relative_gap": _trace_value(_compute_gap(LB, UB)),
            "absolute_gap": _trace_value(_absolute_gap_value()),
            "incumbent": None if incumbent is None else incumbent.copy(),
            "incumbent_objective": _user_bound(incumbent_obj),
            "master_point": None if master_point is None else master_point.copy(),
            "solution_points": [point.copy() for point in points],
            "n_vars": int(n_vars),
            "n_constraints": int(n_cons),
            "mip_count": int(mip_count),
            "nlp_subproblem_count": int(nlp_subproblem_count),
            "feasibility_subproblem_count": int(feasibility_subproblem_count),
            "cut_count": int(len(oa_A_rows)),
            "provenance_cut_count": int(len(cut_provenance.records)),
        }

    def _call_external_hook(name: str, hook: Any, context: dict[str, object]) -> Any:
        external_hook_call_counts[name] += 1
        try:
            return hook(context)
        except Exception as exc:
            external_hook_error_counts[name] += 1
            raise RuntimeError(f"{name} failed during MIP-NLP solve: {exc}") from exc

    def _external_hook_counter_dict(counter: Counter[str]) -> dict[str, int]:
        return {str(key): int(value) for key, value in sorted(counter.items())}

    def _external_hooks_summary() -> dict[str, object]:
        return {
            "call_counts": _external_hook_counter_dict(external_hook_call_counts),
            "accepted_counts": _external_hook_counter_dict(external_hook_accept_counts),
            "rejected_counts": _external_hook_counter_dict(external_hook_reject_counts),
            "error_counts": _external_hook_counter_dict(external_hook_error_counts),
        }

    def _maybe_run_termination_hook(iteration: int) -> Optional[dict[str, object]]:
        if termination_hook is None:
            return None
        event_name = "termination"
        raw = _call_external_hook(
            event_name,
            termination_hook,
            _external_hook_context(event_name, iteration=iteration),
        )
        requested = _validate_external_termination(raw)
        if requested:
            external_hook_accept_counts[event_name] += 1
        else:
            external_hook_reject_counts[event_name] += 1
        return {
            "hook": event_name,
            "status": "terminate" if requested else "continue",
            "requested": bool(requested),
        }

    def _maybe_update_external_dual_bound(
        iteration: int,
        *,
        master_point: Optional[np.ndarray],
    ) -> Optional[dict[str, object]]:
        if external_dual_bound_hook is None:
            return None
        event_name = "external_dual_bound"
        raw = _call_external_hook(
            event_name,
            external_dual_bound_hook,
            _external_hook_context(
                event_name,
                iteration=iteration,
                master_point=master_point,
            ),
        )
        payload = _validate_external_dual_bound(raw)
        if payload is None:
            external_hook_reject_counts[event_name] += 1
            return {"hook": event_name, "status": "no_output", "bound": None}

        user_bound = float(cast(float, payload["bound"]))
        internal_bound = float(_obj_sign * user_bound)
        comparison_bound = certified_LB if bool(payload["global_valid"]) else heuristic_LB
        if internal_bound <= comparison_bound + 1e-12:
            external_hook_reject_counts[event_name] += 1
            return {
                "hook": event_name,
                "status": "not_improving",
                "bound": user_bound,
                "global_valid": bool(payload["global_valid"]),
                "provider": payload.get("provider"),
            }

        if bool(payload["global_valid"]):
            updated = _promote_certified_bound(internal_bound, "external")
        else:
            updated = _record_heuristic_bound(internal_bound, "external")
        if updated:
            external_hook_accept_counts[event_name] += 1
        else:
            external_hook_reject_counts[event_name] += 1
        return {
            "hook": event_name,
            "status": "bound_updated" if updated else "not_improving",
            "bound": user_bound,
            "global_valid": bool(payload["global_valid"]),
            "provider": payload.get("provider"),
        }

    def _maybe_add_external_hyperplanes(
        iteration: int,
        *,
        solution_points: list[np.ndarray],
    ) -> Optional[dict[str, object]]:
        nonlocal local_cut_added
        if external_hyperplane_hook is None:
            return None
        event_name = "external_hyperplane"
        master_point = solution_points[0] if solution_points else None
        raw = _call_external_hook(
            event_name,
            external_hyperplane_hook,
            _external_hook_context(
                event_name,
                iteration=iteration,
                master_point=master_point,
                solution_points=solution_points,
            ),
        )
        payloads = _validate_external_hyperplanes(raw, n_vars=n_vars)
        if not payloads:
            external_hook_reject_counts[event_name] += 1
            return {"hook": event_name, "status": "no_output", "cuts_added": 0}

        cuts_before = len(oa_A_rows)
        local_added = 0
        for payload in payloads:
            support = payload["supporting_point"]
            if support is None:
                support = master_point
            global_valid = bool(payload["global_valid"])
            _append_master_cut(
                oa_A_rows,
                oa_b_rows,
                payload["coefficients"],
                float(cast(float, payload["rhs"])),
                oa_cut_relaxable,
                relaxable=bool(payload["relaxable"]),
                cut_provenance=cut_provenance,
                source="external",
                global_valid=global_valid,
                local_valid=bool(payload["local_valid"]),
                supporting_point=support,
                violation=cast(Optional[float], payload["violation"]),
                constraint_id=cast(Optional[int], payload["constraint_id"]),
                objective_id=cast(Optional[str], payload["objective_id"]),
            )
            if not global_valid:
                local_added += 1
        added = int(len(oa_A_rows) - cuts_before)
        if local_added:
            local_cut_added = True
        external_hook_accept_counts[event_name] += added
        return {
            "hook": event_name,
            "status": "cuts_added" if added else "no_new_cuts",
            "cuts_added": added,
            "local_cuts_added": int(local_added),
        }

    def _maybe_add_external_primal_candidates(
        iteration: int,
        *,
        master_point: Optional[np.ndarray],
        solution_points: list[np.ndarray],
    ) -> Optional[dict[str, object]]:
        if external_primal_candidate_hook is None:
            return None
        event_name = "external_primal_candidate"
        raw = _call_external_hook(
            event_name,
            external_primal_candidate_hook,
            _external_hook_context(
                event_name,
                iteration=iteration,
                master_point=master_point,
                solution_points=solution_points,
            ),
        )
        payloads = _validate_external_primal_candidates(raw, n_vars=n_vars)
        if not payloads:
            external_hook_reject_counts[event_name] += 1
            return {"hook": event_name, "status": "no_output", "candidates_added": 0}
        manager_payloads: list[dict[str, object]] = []
        for payload in payloads:
            manager_payload = dict(payload)
            if manager_payload.get("objective") is not None:
                objective_hint = cast(float, manager_payload["objective"])
                manager_payload["objective"] = float(_obj_sign * objective_hint)
            manager_payloads.append(manager_payload)
        added = fixed_nlp_manager.add_external_candidates(
            manager_payloads,
            iteration=int(iteration),
            provider="external_primal_candidate_hook",
        )
        rejected = int(len(payloads) - added)
        external_hook_accept_counts[event_name] += int(added)
        if rejected:
            external_hook_reject_counts[event_name] += rejected
        return {
            "hook": event_name,
            "status": "candidates_added" if added else "no_new_candidates",
            "candidates_requested": int(len(payloads)),
            "candidates_added": int(added),
            "candidates_rejected": rejected,
        }

    def _build_mip_nlp_trace(final_reason: Optional[str]) -> dict[str, object]:
        final_lb = _trace_value(certified_LB)
        final_heuristic_lb = _trace_value(heuristic_LB)
        final_ub = _trace_value(UB)
        bound_valid = bool(final_lb is not None)
        final_gap = (
            _trace_value(_compute_gap(certified_LB, UB))
            if bound_valid and final_ub is not None
            else None
        )
        heuristic_gap = (
            _trace_value(_compute_gap(heuristic_LB, UB))
            if final_heuristic_lb is not None and final_ub is not None
            else None
        )
        local_cut_count = sum(1 for record in cut_provenance.records if not record.global_valid)
        gap_certified = bool(bound_valid and final_gap is not None)
        summary = {
            "mip_count": int(mip_count),
            "nlp_subproblem_count": int(nlp_subproblem_count),
            "feasibility_subproblem_count": int(feasibility_subproblem_count),
            "cut_count": int(len(oa_A_rows)),
            "provenance_cut_count": int(len(cut_provenance.records)),
            "local_cut_count": int(local_cut_count),
            "cut_source_counts": cut_provenance.source_counts(),
            "solution_pool_candidates": int(solution_pool_candidate_count),
        }
        if shot_solution_limit_state is not None:
            summary["mip_solution_limit"] = shot_solution_limit_state.as_trace_dict()
        if shot_unsupported_backend_features:
            summary["unsupported_backend_features"] = sorted(shot_unsupported_backend_features)
        if external_hook_call_counts:
            summary["external_hooks"] = _external_hooks_summary()
        fixed_nlp_candidates_added = sum(fixed_nlp_manager.added_source_counts.values())
        summary["fixed_nlp_candidate_count"] = int(fixed_nlp_candidates_added)
        summary["fixed_nlp_candidate_source_counts"] = {
            str(source): int(count)
            for source, count in sorted(fixed_nlp_manager.added_source_counts.items())
        }
        summary["fixed_nlp_call_count"] = int(fixed_nlp_call_count)
        summary["fixed_nlp_call_source_counts"] = {
            str(source): int(count)
            for source, count in sorted(fixed_nlp_call_source_counts.items())
        }
        summary["fixed_nlp_call_status_counts"] = {
            str(status): int(count)
            for status, count in sorted(fixed_nlp_call_status_counts.items())
        }
        summary["fixed_nlp_scheduler"] = fixed_nlp_manager.scheduler_trace()
        if interior_point_store is not None:
            interior_counts = Counter(record.source for record in interior_point_store.records)
            summary["interior_point_count"] = int(len(interior_point_store.records))
            summary["interior_point_source_counts"] = {
                str(source): int(count) for source, count in sorted(interior_counts.items())
            }
        if initial_poa_trace is not None:
            poa_cuts = initial_poa_trace.get("cuts_added", 0)
            poa_provenance_cuts = initial_poa_trace.get("provenance_cuts_added", 0)
            summary["initial_poa_cuts"] = int(poa_cuts) if isinstance(poa_cuts, int) else 0
            summary["initial_poa_provenance_cuts"] = (
                int(poa_provenance_cuts) if isinstance(poa_provenance_cuts, int) else 0
            )
        repair_actions: list[dict[str, object]] = []
        reduction_events: list[dict[str, object]] = []
        for iteration_record in trace_iterations:
            raw_repair_actions = iteration_record.get("repair_actions", [])
            if isinstance(raw_repair_actions, list):
                repair_actions.extend(
                    action for action in raw_repair_actions if isinstance(action, dict)
                )
            raw_reduction_events = iteration_record.get("reduction_cuts", [])
            if isinstance(raw_reduction_events, list):
                reduction_events.extend(
                    event for event in raw_reduction_events if isinstance(event, dict)
                )
        summary["master_repair_attempt_count"] = sum(
            1 for action in repair_actions if action.get("attempted")
        )
        summary["master_repair_success_count"] = sum(
            1 for action in repair_actions if action.get("status") == "repaired"
        )
        summary["master_repair_failure_count"] = sum(
            1 for action in repair_actions if action.get("status") == "failed"
        )
        summary["master_repair_loop_count"] = sum(
            1 for action in repair_actions if action.get("status") == "loop_detected"
        )
        summary["reduction_cut_added_count"] = sum(
            1 for event in reduction_events if event.get("status") == "added"
        )
        summary["reduction_cut_skipped_count"] = sum(
            1 for event in reduction_events if event.get("status") == "skipped"
        )
        convex_bounding_records: list[dict[str, object]] = []
        for iteration_record in trace_iterations:
            raw_convex_bounding = iteration_record.get("convex_bounding")
            if isinstance(raw_convex_bounding, dict):
                convex_bounding_records.append(cast(dict[str, object], raw_convex_bounding))
        summary["convex_bounding_solve_count"] = sum(
            1 for record in convex_bounding_records if bool(record.get("attempted"))
        )
        summary["convex_bounding_bound_update_count"] = sum(
            1 for record in convex_bounding_records if bool(record.get("bound_updated"))
        )
        if final_lb is not None:
            bound_validity = "global"
        elif final_heuristic_lb is not None:
            bound_validity = "heuristic"
        else:
            bound_validity = "unavailable"
        trace = {
            "schema_version": 1,
            "solver": "mip-nlp",
            "method": method_name,
            "profile": mip_nlp_profile,
            "shot_options": (
                mip_nlp_shot_config.as_trace_dict() if mip_nlp_shot_config is not None else {}
            ),
            "iterations": trace_iterations,
            "summary": summary,
            "termination_reason": final_reason,
            "master_bound_valid": bound_valid,
            "gap_certified": gap_certified,
            "bound_validity": bound_validity,
            "final_lb": final_lb,
            "final_ub": final_ub,
            "final_gap": final_gap,
            "heuristic_lb": final_heuristic_lb,
            "heuristic_gap": heuristic_gap,
            "certified_bound_source": certified_bound_source,
            "heuristic_bound_source": heuristic_bound_source,
        }
        if shot_solution_pool_degraded_reason is not None:
            trace["solution_pool_degraded_reason"] = shot_solution_pool_degraded_reason
        if initial_poa_trace is not None:
            trace["initial_poa"] = dict(initial_poa_trace)
        return trace

    def _fixed_nlp_status_name(attempt: _NLPAttempt) -> str:
        status = attempt.status
        if status is not None:
            return _trace_status(status)
        if attempt.x is not None:
            return "feasible"
        return "failed"

    def _fixed_nlp_warm_start(
        candidate: FixedNLPCandidate,
        preferred_start: Optional[np.ndarray],
    ) -> tuple[np.ndarray, str]:
        if preferred_start is not None:
            return np.asarray(preferred_start, dtype=np.float64), "regularized_master"
        return candidate.point.copy(), candidate.source

    def _record_fixed_nlp_trace(
        iteration_record: dict[str, object],
        candidate: FixedNLPCandidate,
        *,
        status: str,
        objective: Optional[float],
        incumbent_update: str,
        warm_start_source: str,
    ) -> None:
        calls = iteration_record.get("fixed_nlp_calls")
        if not isinstance(calls, list):
            calls = []
            iteration_record["fixed_nlp_calls"] = calls
        trace = candidate.trace_dict()
        trace.update(
            {
                "status": status,
                "objective": _trace_value(objective),
                "incumbent_update": incumbent_update,
                "warm_start_source": warm_start_source,
            }
        )
        calls.append(trace)

    def _record_interior_point(
        x: np.ndarray,
        source: str,
        metadata: Optional[dict[str, object]] = None,
    ) -> bool:
        if interior_point_store is None:
            return False
        record = interior_point_store.add(
            x,
            source=source,
            metadata=metadata,
            evaluator=evaluator,
            constraint_senses=decomp.constraint_senses,
            require_feasible=True,
        )
        return record is not None

    def _shot_disabled_relaxation_trace() -> Optional[dict[str, object]]:
        if mip_nlp_profile != "shot" or mip_nlp_shot_config is None:
            return None
        phase = mip_nlp_shot_config.relaxation_phase
        return {
            "phase": phase,
            "enabled": False,
            "attempted": False,
            "status": "disabled",
            "fallback_reason": f"relaxation_phase={phase}",
            "cuts_added": 0,
            "provenance_cuts_added": 0,
            "objective_bound": None,
            "objective_bound_valid": False,
            "node_count": 0,
        }

    def _shot_objective_cutoff() -> Optional[float]:
        if incumbent_obj is None or not master_bound_valid:
            return None
        cutoff = _master_objective_from_evaluator(incumbent_obj)
        if cutoff is None:
            return None
        if not np.isfinite(cutoff):
            return None
        return cutoff + 1e-8 * (1.0 + abs(cutoff))

    def _shot_master_controls() -> tuple[
        dict[str, object], Optional[np.ndarray], Optional[float], Optional[int], Optional[float]
    ]:
        if mip_nlp_profile != "shot" or mip_nlp_shot_config is None:
            return {}, None, None, None, None

        start_requested = incumbent is not None
        cutoff_requested = _shot_objective_cutoff() is not None
        objective_cutoff = _shot_objective_cutoff() if shot_master_backend_supported else None
        mip_start = incumbent if start_requested and shot_master_backend_supported else None
        mip_start_objective = _master_objective_from_evaluator(incumbent_obj)
        limit = shot_solution_limit_state.requested_limit if shot_solution_limit_state else None

        if start_requested and not shot_master_backend_supported:
            shot_unsupported_backend_features.add("mip_start")
        if cutoff_requested and not shot_master_backend_supported:
            shot_unsupported_backend_features.add("objective_cutoff")

        unsupported_reason = None
        if not shot_master_backend_supported:
            unsupported_reason = "requires milp_solver='gurobi'"
        trace = {
            "backend": str(milp_solver),
            "backend_supported": bool(shot_master_backend_supported),
            "mip_start": {
                "requested": bool(start_requested),
                "supported": bool(shot_master_backend_supported),
                "applied": bool(mip_start is not None),
                "degraded_reason": unsupported_reason if start_requested else None,
            },
            "objective_cutoff": {
                "requested": bool(cutoff_requested),
                "supported": bool(shot_master_backend_supported),
                "applied": bool(objective_cutoff is not None),
                "value": _trace_value(objective_cutoff),
                "degraded_reason": unsupported_reason if cutoff_requested else None,
            },
            "mip_solution_limit": (
                shot_solution_limit_state.as_trace_dict()
                if shot_solution_limit_state is not None
                else {
                    "strategy": "none",
                    "enabled": False,
                    "supported": bool(shot_master_backend_supported),
                    "limit": None,
                    "raw_limit": None,
                    "capacity": 0,
                    "updates": 0,
                    "last_update_reason": "disabled",
                    "degraded_reason": None,
                }
            ),
        }
        return trace, mip_start, objective_cutoff, limit, mip_start_objective

    def _maybe_update_convex_bounding_bound(iteration: int, elapsed: float) -> dict[str, object]:
        nonlocal mip_count
        enabled = bool(
            mip_nlp_profile == "shot" and not (master_bound_valid and not local_cut_added)
        )
        rows, rhs, local_excluded, integer_excluded = _global_valid_master_cut_rows(cut_provenance)
        trace: dict[str, object] = {
            "iteration": int(iteration),
            "enabled": enabled,
            "attempted": False,
            "status": "disabled" if not enabled else "pending",
            "reason": "primary_master_certified" if not enabled else None,
            "global_cut_count": int(len(rows)),
            "local_cut_excluded_count": int(local_excluded),
            "integer_cut_excluded_count": int(integer_excluded),
            "bound_before": _trace_value(certified_LB),
            "objective_bound": None,
            "bound_after": _trace_value(certified_LB),
            "bound_updated": False,
            "master_status": None,
            "node_count": 0,
        }
        if not enabled:
            return trace
        if not decomp.master_bound_valid:
            trace.update(
                {
                    "status": "unavailable",
                    "reason": "objective_not_globally_boundable",
                }
            )
            return trace
        remaining = max(float(time_limit) - float(elapsed), 0.0)
        if remaining <= 0.0:
            trace.update({"status": "skipped", "reason": "time_limit"})
            return trace
        trace.update({"attempted": True, "status": "running"})
        try:
            result = _solve_master_milp(
                decomp.linear_A_rows,
                decomp.linear_b_rows,
                decomp.linear_senses,
                rows,
                rhs,
                n_vars,
                decomp.integrality,
                decomp.lb,
                decomp.ub,
                decomp.obj_coeffs,
                decomp.obj_is_linear,
                decomp.master_bound_valid,
                time_limit=remaining,
                gap_tolerance=gap_tolerance,
                add_slack=False,
                max_slack=max_slack,
                oa_penalty_factor=oa_penalty_factor,
                oa_cut_relaxable=None,
                use_objective_epigraph=(not decomp.obj_is_linear and decomp.oa_objective_is_convex),
                milp_solver=milp_solver,
                solution_pool=False,
                num_solution_iteration=1,
                mip_start=None,
                mip_start_objective=None,
                objective_cutoff=None,
                mip_solution_limit=None,
                integer_binary_expansion=integer_binary_expansion,
            )
            mip_count += 1
        except Exception as exc:
            trace.update(
                {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "bound_after": _trace_value(certified_LB),
                }
            )
            return trace

        status_name = "none" if result is None else _trace_status(result.status)
        trace["master_status"] = status_name
        trace["node_count"] = int(getattr(result, "node_count", 0) or 0)
        if result is None or result.bound is None:
            trace.update(
                {
                    "status": "no_bound",
                    "reason": f"master_status={status_name}",
                    "bound_after": _trace_value(certified_LB),
                }
            )
            return trace

        objective_bound = _evaluator_objective_from_master(result.bound)
        updated = _promote_certified_bound(objective_bound, "convex_bounding")
        trace.update(
            {
                "status": "bound_updated" if updated else "no_bound_update",
                "reason": None,
                "objective_bound": _trace_value(objective_bound),
                "bound_after": _trace_value(certified_LB),
                "bound_updated": bool(updated),
            }
        )
        return trace

    def _shot_reduction_cuts_enabled() -> bool:
        return bool(
            mip_nlp_profile == "shot"
            and mip_nlp_shot_config is not None
            and mip_nlp_shot_config.reduction_cuts
            and not master_bound_valid
        )

    def _drop_active_reduction_cuts(reason: str) -> int:
        nonlocal reduction_cut_incumbent_key
        if not active_reduction_cut_indices:
            return 0
        keep_indices = [
            idx for idx in range(len(oa_A_rows)) if idx not in active_reduction_cut_indices
        ]
        removed = len(oa_A_rows) - len(keep_indices)
        oa_A_rows[:] = [oa_A_rows[idx] for idx in keep_indices]
        oa_b_rows[:] = [oa_b_rows[idx] for idx in keep_indices]
        if oa_cut_relaxable:
            oa_cut_relaxable[:] = [oa_cut_relaxable[idx] for idx in keep_indices]
        cut_provenance.remove_source("reduction")
        active_reduction_cut_indices.clear()
        reduction_cut_incumbent_key = None
        logger.info("OA: dropped %d active primal reduction cut(s): %s", removed, reason)
        return removed

    def _maybe_add_primal_reduction_cut(iteration: int) -> Optional[dict[str, object]]:
        nonlocal local_cut_added, reduction_cut_incumbent_key
        if not _shot_reduction_cuts_enabled():
            return None
        event = {
            "iteration": int(iteration),
            "enabled": True,
        }
        if incumbent_obj is None:
            trace = _add_primal_reduction_cut(
                decomp,
                incumbent,
                incumbent_obj,
                oa_A_rows,
                oa_b_rows,
                oa_cut_relaxable=oa_cut_relaxable,
                cut_provenance=cut_provenance,
            )
            trace.update(event)
            return trace

        incumbent_key = float(incumbent_obj)
        if reduction_cut_incumbent_key is not None and abs(
            reduction_cut_incumbent_key - incumbent_key
        ) <= 1e-9 * (1.0 + abs(incumbent_key)):
            return {
                **event,
                "status": "skipped",
                "reason": "already_active_for_incumbent",
                "source": "reduction",
                "global_valid": False,
                "local_valid": True,
                "cutoff": None,
                "incumbent_objective": _trace_value(incumbent_key),
            }

        dropped = _drop_active_reduction_cuts("incumbent_changed")
        row_index = len(oa_A_rows)
        trace = _add_primal_reduction_cut(
            decomp,
            incumbent,
            incumbent_obj,
            oa_A_rows,
            oa_b_rows,
            oa_cut_relaxable=oa_cut_relaxable,
            cut_provenance=cut_provenance,
        )
        trace.update(event)
        trace["dropped_previous"] = int(dropped)
        if trace.get("status") == "added":
            active_reduction_cut_indices.add(row_index)
            reduction_cut_incumbent_key = incumbent_key
            local_cut_added = True
        return trace

    def _attempt_master_repair(
        *,
        iteration: int,
        master_objective_cutoff: Optional[float],
        master_solution_limit: Optional[int],
        elapsed: float,
    ):
        nonlocal mip_count
        action: dict[str, object] = {
            "iteration": int(iteration),
            "attempted": False,
            "status": "disabled",
            "reason": "master_repair_disabled",
            "reset_objective_cutoff": False,
            "reset_mip_solution_limit": False,
            "dropped_reduction_cuts": 0,
            "master_status": None,
            "node_count": 0,
        }
        if not (
            mip_nlp_profile == "shot"
            and mip_nlp_shot_config is not None
            and mip_nlp_shot_config.master_repair
        ):
            return None, action

        action.update(
            {
                "attempted": True,
                "status": "running",
                "reason": None,
                "reset_objective_cutoff": bool(master_objective_cutoff is not None),
                "reset_mip_solution_limit": bool(master_solution_limit is not None),
                "dropped_reduction_cuts": int(_drop_active_reduction_cuts("master_infeasible")),
            }
        )
        try:
            repaired = _solve_master_milp(
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
                time_limit=max(time_limit - elapsed, 0.0),
                gap_tolerance=gap_tolerance,
                add_slack=True,
                max_slack=max_slack,
                oa_penalty_factor=oa_penalty_factor,
                oa_cut_relaxable=oa_cut_relaxable,
                use_objective_epigraph=(not decomp.obj_is_linear and decomp.oa_objective_is_convex),
                milp_solver=milp_solver,
                solution_pool=False,
                num_solution_iteration=1,
                mip_start=None,
                mip_start_objective=None,
                objective_cutoff=None,
                mip_solution_limit=None,
                integer_binary_expansion=integer_binary_expansion,
            )
            mip_count += 1
        except Exception as exc:
            action.update(
                {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            return None, action

        status_name = "none" if repaired is None else _trace_status(repaired.status)
        action["master_status"] = status_name
        action["node_count"] = int(getattr(repaired, "node_count", 0) or 0)
        if repaired is None or repaired.x is None:
            action.update({"status": "failed", "reason": f"master_status={status_name}"})
            return None, action
        if status_name not in {"optimal", "iteration_limit", "time_limit"}:
            action.update({"status": "failed", "reason": f"master_status={status_name}"})
            return None, action

        repaired_x = np.asarray(repaired.x, dtype=np.float64).reshape(-1)
        if repaired_x.size < n_vars:
            action.update(
                {
                    "status": "failed",
                    "reason": f"master_solution_size={repaired_x.size}",
                }
            )
            return None, action

        assignment_key = _integer_assignment_key(decomp, repaired_x[:n_vars])
        action["integer_assignment"] = list(assignment_key)
        if assignment_key in repaired_assignment_keys:
            action.update(
                {
                    "status": "loop_detected",
                    "reason": "repaired_integer_assignment_repeated",
                }
            )
            return None, action

        repaired_assignment_keys.add(assignment_key)
        action.update({"status": "repaired", "reason": None})
        return repaired, action

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
            _promote_certified_bound(obj, "continuous_nlp")
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
        _record_interior_point(incumbent, "incumbent", {"objective": float(obj)})

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
            _record_interior_point(x_relax, "nlp_relaxation", {"objective": float(obj_relax)})
            if mip_nlp_profile == "shot":
                fixed_nlp_manager.add(
                    x_relax,
                    source="lp_relaxation",
                    objective=obj_relax,
                    iteration=-1,
                )
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

    if initial_poa_trace is not None and initial_poa_trace["enabled"]:
        if not oa_A_rows:
            initial_poa_trace.update(
                {
                    "status": "skipped",
                    "fallback_reason": "no_initial_polyhedral_cuts",
                }
            )
        else:
            cuts_before = len(oa_A_rows)
            provenance_before = len(cut_provenance.records)
            bound_before = _trace_value(LB)
            initial_poa_trace.update(
                {
                    "attempted": True,
                    "status": "running",
                    "fallback_reason": None,
                    "bound_before": bound_before,
                }
            )
            try:
                poa_result = _solve_initial_poa_master(
                    decomp,
                    oa_A_rows,
                    oa_b_rows,
                    master_bound_valid=master_bound_valid,
                    time_limit=max(time_limit - (time.perf_counter() - t_start), 0.0),
                    gap_tolerance=gap_tolerance,
                    add_slack=add_slack,
                    max_slack=max_slack,
                    oa_penalty_factor=oa_penalty_factor,
                    oa_cut_relaxable=oa_cut_relaxable,
                    milp_solver=milp_solver,
                    integer_binary_expansion=integer_binary_expansion,
                )
                mip_count += 1
            except Exception as exc:
                initial_poa_trace.update(
                    {
                        "status": "fallback",
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                        "bound_after": _trace_value(LB),
                    }
                )
            else:
                status_name = "none" if poa_result is None else _trace_status(poa_result.status)
                if poa_result is None or poa_result.x is None:
                    initial_poa_trace.update(
                        {
                            "status": "fallback",
                            "fallback_reason": f"master_status={status_name}",
                            "bound_after": _trace_value(LB),
                        }
                    )
                elif status_name not in {"optimal", "iteration_limit"}:
                    initial_poa_trace.update(
                        {
                            "status": "fallback",
                            "fallback_reason": f"master_status={status_name}",
                            "node_count": int(getattr(poa_result, "node_count", 0) or 0),
                            "bound_after": _trace_value(LB),
                        }
                    )
                else:
                    poa_x = np.asarray(poa_result.x, dtype=np.float64).reshape(-1)
                    if poa_x.size < n_vars:
                        initial_poa_trace.update(
                            {
                                "status": "fallback",
                                "fallback_reason": f"master_solution_size={poa_x.size}",
                                "node_count": int(getattr(poa_result, "node_count", 0) or 0),
                                "bound_after": _trace_value(LB),
                            }
                        )
                    else:
                        x_poa = np.clip(
                            poa_x[:n_vars],
                            decomp.lb,
                            decomp.ub,
                        )
                        objective_bound = None
                        if master_bound_valid and poa_result.bound is not None:
                            objective_bound = _evaluator_objective_from_master(poa_result.bound)
                            if objective_bound is not None:
                                if not local_cut_added:
                                    _promote_certified_bound(objective_bound, "initial_poa")
                                else:
                                    _record_heuristic_bound(objective_bound, "initial_poa")
                        stored_poa_interior = _record_interior_point(
                            x_poa,
                            "initial_poa",
                            {
                                "objective_bound": _trace_value(objective_bound),
                                "node_count": int(getattr(poa_result, "node_count", 0) or 0),
                            },
                        )
                        fixed_nlp_manager.add(
                            x_poa,
                            source="lp_relaxation",
                            objective=objective_bound,
                            iteration=-1,
                        )
                        interior_candidates = 1 if stored_poa_interior else 0
                        _add_oa_cuts(
                            evaluator,
                            x_poa,
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
                            constraint_source="initial_poa",
                        )
                        cuts_added = int(len(oa_A_rows) - cuts_before)
                        provenance_added = int(len(cut_provenance.records) - provenance_before)
                        initial_poa_trace.update(
                            {
                                "status": "seeded" if cuts_added else "no_new_cuts",
                                "fallback_reason": None,
                                "cuts_added": cuts_added,
                                "provenance_cuts_added": provenance_added,
                                "objective_bound": _trace_value(objective_bound),
                                "objective_bound_valid": bool(
                                    master_bound_valid and objective_bound is not None
                                ),
                                "bound_after": _trace_value(LB),
                                "interior_point_candidates": int(interior_candidates),
                                "interior_points_stored": int(interior_candidates),
                                "node_count": int(getattr(poa_result, "node_count", 0) or 0),
                            }
                        )

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
        external_hook_events: list[dict[str, object]] = []
        termination_event = _maybe_run_termination_hook(iteration)
        if termination_event is not None:
            external_hook_events.append(termination_event)
            if termination_event["requested"]:
                termination_reason = "user_termination"
                trace_iterations.append(
                    {
                        "index": int(iteration),
                        "master_status": "not_run",
                        "lb_before": lb_before,
                        "ub_before": ub_before,
                        "lb": _trace_value(LB),
                        "ub": _trace_value(UB),
                        "gap": _trace_value(_compute_gap(LB, UB)),
                        "cuts_added": 0,
                        "cuts_total": int(len(oa_A_rows)),
                        "provenance_cuts_added": 0,
                        "provenance_cuts_total": int(len(cut_provenance.records)),
                        "cuts_added_by_source": _cut_source_delta(cut_source_counts_before),
                        "nlp_subproblem_count": 0,
                        "feasibility_subproblem_count": 0,
                        "solution_pool_candidates": 0,
                        "node_count": 0,
                        "repair_actions": [],
                        "reduction_cuts": [],
                        "relaxation_phase": _shot_disabled_relaxation_trace(),
                        "convex_bounding": None,
                        "master_controls": {},
                        "external_hooks": external_hook_events,
                        "termination_reason": termination_reason,
                    }
                )
                break
        reduction_cut_events: list[dict[str, object]] = []
        reduction_cut_event = _maybe_add_primal_reduction_cut(iteration)
        if reduction_cut_event is not None:
            reduction_cut_events.append(reduction_cut_event)
        relaxation_phase_record = _shot_disabled_relaxation_trace()
        if (
            mip_nlp_profile == "shot"
            and mip_nlp_shot_config is not None
            and mip_nlp_shot_config.relaxation_phase in _PERIODIC_RELAXATION_PHASES
        ):
            relaxation_phase_record = {
                "phase": mip_nlp_shot_config.relaxation_phase,
                "enabled": True,
                "attempted": True,
                "status": "running",
                "fallback_reason": None,
                "bound_before": _trace_value(LB),
                "cuts_added": 0,
                "provenance_cuts_added": 0,
                "objective_bound": None,
                "objective_bound_valid": False,
                "interior_point_candidates": 0,
                "interior_points_stored": 0,
                "node_count": 0,
            }
            relax_cuts_before = len(oa_A_rows)
            relax_provenance_before = len(cut_provenance.records)
            try:
                relax_result = _solve_initial_poa_master(
                    decomp,
                    oa_A_rows,
                    oa_b_rows,
                    master_bound_valid=master_bound_valid,
                    time_limit=max(time_limit - elapsed, 0.0),
                    gap_tolerance=gap_tolerance,
                    add_slack=add_slack,
                    max_slack=max_slack,
                    oa_penalty_factor=oa_penalty_factor,
                    oa_cut_relaxable=oa_cut_relaxable,
                    milp_solver=milp_solver,
                    integer_binary_expansion=integer_binary_expansion,
                )
                mip_count += 1
            except Exception as exc:
                relaxation_phase_record.update(
                    {
                        "status": "fallback",
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                        "bound_after": _trace_value(LB),
                    }
                )
            else:
                status_name = "none" if relax_result is None else _trace_status(relax_result.status)
                if relax_result is None or relax_result.x is None:
                    relaxation_phase_record.update(
                        {
                            "status": "fallback",
                            "fallback_reason": f"master_status={status_name}",
                            "bound_after": _trace_value(LB),
                        }
                    )
                elif status_name not in {"optimal", "iteration_limit"}:
                    relaxation_phase_record.update(
                        {
                            "status": "fallback",
                            "fallback_reason": f"master_status={status_name}",
                            "node_count": int(getattr(relax_result, "node_count", 0) or 0),
                            "bound_after": _trace_value(LB),
                        }
                    )
                else:
                    relax_x_raw = np.asarray(relax_result.x, dtype=np.float64).reshape(-1)
                    if relax_x_raw.size < n_vars:
                        relaxation_phase_record.update(
                            {
                                "status": "fallback",
                                "fallback_reason": f"master_solution_size={relax_x_raw.size}",
                                "node_count": int(getattr(relax_result, "node_count", 0) or 0),
                                "bound_after": _trace_value(LB),
                            }
                        )
                    else:
                        x_relax_phase = np.clip(relax_x_raw[:n_vars], decomp.lb, decomp.ub)
                        objective_bound = None
                        if master_bound_valid and relax_result.bound is not None:
                            objective_bound = _evaluator_objective_from_master(relax_result.bound)
                            if objective_bound is not None:
                                if not local_cut_added:
                                    _promote_certified_bound(objective_bound, "relaxation_phase")
                                else:
                                    _record_heuristic_bound(objective_bound, "relaxation_phase")
                        stored_relax_interior = _record_interior_point(
                            x_relax_phase,
                            "relaxation_phase",
                            {
                                "objective_bound": _trace_value(objective_bound),
                                "iteration": int(iteration),
                                "node_count": int(getattr(relax_result, "node_count", 0) or 0),
                            },
                        )
                        fixed_nlp_manager.add(
                            x_relax_phase,
                            source="lp_relaxation",
                            objective=objective_bound,
                            iteration=int(iteration),
                        )
                        _add_oa_cuts(
                            evaluator,
                            x_relax_phase,
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
                            constraint_source="relaxation_phase",
                        )
                        cuts_added = int(len(oa_A_rows) - relax_cuts_before)
                        provenance_added = int(
                            len(cut_provenance.records) - relax_provenance_before
                        )
                        relaxation_phase_record.update(
                            {
                                "status": "seeded" if cuts_added else "no_new_cuts",
                                "fallback_reason": None,
                                "cuts_added": cuts_added,
                                "provenance_cuts_added": provenance_added,
                                "objective_bound": _trace_value(objective_bound),
                                "objective_bound_valid": bool(
                                    master_bound_valid and objective_bound is not None
                                ),
                                "bound_after": _trace_value(LB),
                                "interior_point_candidates": int(1 if stored_relax_interior else 0),
                                "interior_points_stored": int(1 if stored_relax_interior else 0),
                                "node_count": int(getattr(relax_result, "node_count", 0) or 0),
                            }
                        )

        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            logger.info("OA: Time limit reached after relaxation phase at iteration %d", iteration)
            termination_reason = "time_limit"
            trace_iterations.append(
                {
                    "index": int(iteration),
                    "master_status": "not_run",
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
                    "reduction_cuts": reduction_cut_events,
                    "relaxation_phase": relaxation_phase_record,
                    "master_controls": {},
                    "external_hooks": external_hook_events,
                    "termination_reason": termination_reason,
                }
            )
            break

        (
            master_control_trace,
            master_mip_start,
            master_objective_cutoff,
            master_solution_limit,
            master_mip_start_objective,
        ) = _shot_master_controls()
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
            mip_start=master_mip_start,
            mip_start_objective=master_mip_start_objective,
            objective_cutoff=master_objective_cutoff,
            mip_solution_limit=master_solution_limit,
            integer_binary_expansion=integer_binary_expansion,
        )
        mip_count += 1
        convex_bounding_record = _maybe_update_convex_bounding_bound(
            iteration,
            time.perf_counter() - t_start,
        )

        from discopt.solvers import SolveStatus

        repair_actions: list[dict[str, object]] = []
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
                    "reduction_cuts": reduction_cut_events,
                    "relaxation_phase": relaxation_phase_record,
                    "convex_bounding": convex_bounding_record,
                    "master_controls": master_control_trace,
                    "external_hooks": external_hook_events,
                    "termination_reason": termination_reason,
                }
            )
            break

        if master_result.status == SolveStatus.CUTOFF:
            cutoff_bound = master_result.bound
            if cutoff_bound is None:
                cutoff_bound = master_objective_cutoff
            if master_bound_valid and cutoff_bound is not None:
                evaluator_cutoff_bound = _evaluator_objective_from_master(cutoff_bound)
                if evaluator_cutoff_bound is not None:
                    if not local_cut_added:
                        _promote_certified_bound(evaluator_cutoff_bound, "primary_master_cutoff")
                    else:
                        _record_heuristic_bound(evaluator_cutoff_bound, "primary_master_cutoff")
            gap = _compute_gap(LB, UB)
            termination_reason = "gap" if _certified_gap_converged() else "master_cutoff"
            trace_iterations.append(
                {
                    "index": int(iteration),
                    "master_status": _trace_status(master_result.status),
                    "lb_before": lb_before,
                    "ub_before": ub_before,
                    "lb": _trace_value(LB),
                    "ub": _trace_value(UB),
                    "gap": _trace_value(gap),
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
                    "reduction_cuts": reduction_cut_events,
                    "relaxation_phase": relaxation_phase_record,
                    "convex_bounding": convex_bounding_record,
                    "master_controls": master_control_trace,
                    "external_hooks": external_hook_events,
                    "termination_reason": termination_reason,
                }
            )
            break

        if master_result.status == SolveStatus.INFEASIBLE:
            logger.info("OA: Master MILP infeasible at iteration %d", iteration)
            repaired_result, repair_action = _attempt_master_repair(
                iteration=iteration,
                master_objective_cutoff=master_objective_cutoff,
                master_solution_limit=master_solution_limit,
                elapsed=time.perf_counter() - t_start,
            )
            repair_actions = [repair_action] if repair_action.get("attempted") else []
            if repaired_result is None:
                termination_reason = (
                    "master_repair_loop"
                    if repair_action.get("status") == "loop_detected"
                    else (
                        "master_infeasible_unrepaired"
                        if repair_action.get("attempted")
                        else "master_infeasible"
                    )
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
                        "provenance_cuts_added": int(
                            len(cut_provenance.records) - provenance_before
                        ),
                        "provenance_cuts_total": int(len(cut_provenance.records)),
                        "cuts_added_by_source": _cut_source_delta(cut_source_counts_before),
                        "nlp_subproblem_count": int(nlp_subproblem_count - nlp_before),
                        "feasibility_subproblem_count": int(
                            feasibility_subproblem_count - feasibility_before
                        ),
                        "solution_pool_candidates": 0,
                        "node_count": int(getattr(master_result, "node_count", 0) or 0),
                        "repair_actions": repair_actions,
                        "reduction_cuts": reduction_cut_events,
                        "relaxation_phase": relaxation_phase_record,
                        "convex_bounding": convex_bounding_record,
                        "master_controls": master_control_trace,
                        "external_hooks": external_hook_events,
                        "termination_reason": termination_reason,
                    }
                )
                break
            master_result = repaired_result

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
                    "reduction_cuts": reduction_cut_events,
                    "relaxation_phase": relaxation_phase_record,
                    "convex_bounding": convex_bounding_record,
                    "master_controls": master_control_trace,
                    "external_hooks": external_hook_events,
                    "termination_reason": "master_unbounded",
                }
            )
            continue

        # The master gives a valid LB only via its dual ``bound`` (never the
        # incumbent ``objective``, which is an upper bound on a limited solve).
        if master_result.bound is not None:
            master_bound = _evaluator_objective_from_master(master_result.bound)
            if master_bound is not None:
                if master_bound_valid and not local_cut_added:
                    _promote_certified_bound(master_bound, "primary_master")
                else:
                    _record_heuristic_bound(master_bound, "primary_master")

        master_solution_points = _master_solution_candidates(
            master_result,
            n_vars,
            solution_pool=solution_pool,
            num_solution_iteration=num_solution_iteration,
        )
        primary_master_point = master_solution_points[0] if master_solution_points else None
        external_dual_event = _maybe_update_external_dual_bound(
            iteration,
            master_point=primary_master_point,
        )
        if external_dual_event is not None:
            external_hook_events.append(external_dual_event)
        external_hyperplane_event = _maybe_add_external_hyperplanes(
            iteration,
            solution_points=master_solution_points,
        )
        if external_hyperplane_event is not None:
            external_hook_events.append(external_hyperplane_event)

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
                integer_binary_expansion=integer_binary_expansion,
            )
            if x_regularized is not None:
                nlp_initial_point = x_regularized
                logger.info(
                    "OA: %s regularized master selected fixed-NLP initial point",
                    add_regularization,
                )

        if ecp_mode:
            fixed_nlp_candidates = [
                FixedNLPCandidate(
                    point=point,
                    source="mip_optimum" if idx == 0 else "solution_pool",
                    objective=None,
                    iteration=int(iteration),
                    sequence=idx,
                    integer_assignment=fixed_nlp_manager.assignment_key(point),
                )
                for idx, point in enumerate(master_solution_points)
            ]
        else:
            external_primal_event = _maybe_add_external_primal_candidates(
                iteration,
                master_point=primary_master_point,
                solution_points=master_solution_points,
            )
            if external_primal_event is not None:
                external_hook_events.append(external_primal_event)
            fixed_nlp_manager.add_master_result(
                master_result,
                iteration=int(iteration),
                solution_pool=solution_pool,
                limit=num_solution_iteration,
            )
            fixed_nlp_candidates = fixed_nlp_manager.take_ready(
                iteration=int(iteration),
                elapsed=time.perf_counter() - t_start,
                has_solution_pool_candidate=bool(solution_pool),
            )
        processed_master_candidates = sum(
            1 for cand in fixed_nlp_candidates if cand.source in {"mip_optimum", "solution_pool"}
        )
        solution_pool_candidate_count += processed_master_candidates
        incumbent_obj_before_iteration = incumbent_obj
        iteration_record: dict[str, object] = {
            "index": int(iteration),
            "master_status": _trace_status(master_result.status),
            "lb_before": lb_before,
            "ub_before": ub_before,
            "solution_pool_candidates": int(processed_master_candidates),
            "fixed_nlp_candidates": int(len(fixed_nlp_candidates)),
            "fixed_nlp_scheduler": fixed_nlp_manager.scheduler_trace(),
            "node_count": int(getattr(master_result, "node_count", 0) or 0),
            "repair_actions": repair_actions,
            "reduction_cuts": reduction_cut_events,
            "relaxation_phase": relaxation_phase_record,
            "convex_bounding": convex_bounding_record,
            "master_controls": master_control_trace,
            "external_hooks": external_hook_events,
        }
        stop_after_master_pool = False
        pool_integer_assignments_seen: set[tuple[float, ...]] = set()

        if not ecp_mode and not fixed_nlp_candidates:
            x_master = np.asarray(master_result.x, dtype=np.float64).reshape(-1)[:n_vars].copy()
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
            iteration_record["fixed_nlp_skipped"] = {
                "reason": f"fixed_nlp_strategy={fixed_nlp_strategy}",
                "ecp_cuts_added": int(n_violated),
            }

        for candidate_index, candidate in enumerate(fixed_nlp_candidates):
            x_master = candidate.point
            elapsed = time.perf_counter() - t_start
            if elapsed >= time_limit:
                logger.info("OA: Time limit reached during iteration %d", iteration)
                termination_reason = "time_limit"
                stop_after_master_pool = True
                break

            int_assignment = candidate.integer_assignment
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
                if (
                    mip_nlp_profile == "shot"
                    and mip_nlp_shot_config is not None
                    and shot_cut_strategy in {"auto", "esh"}
                ):
                    n_violated, esh_trace = _add_esh_cuts(
                        evaluator,
                        x_master,
                        n_vars,
                        decomp.constraint_senses,
                        oa_A_rows,
                        oa_b_rows,
                        decomp.obj_is_linear,
                        decomp.oa_constraint_mask,
                        decomp.oa_objective_is_convex,
                        interior_point_store,
                        rootsearch_strategy=mip_nlp_shot_config.rootsearch_strategy,
                        equality_relaxation=equality_relaxation,
                        oa_cut_relaxable=oa_cut_relaxable,
                        cut_provenance=cut_provenance,
                        incumbent=incumbent,
                        incumbent_obj=incumbent_obj,
                        hyperplane_max_per_iter=mip_nlp_shot_config.hyperplane_max_per_iter,
                        hyperplane_selection_factor=(
                            mip_nlp_shot_config.hyperplane_selection_factor
                        ),
                    )
                    esh_events = iteration_record.get("esh")
                    if not isinstance(esh_events, list):
                        esh_events = []
                        iteration_record["esh"] = esh_events
                    esh_events.append(esh_trace)
                    local_cuts_added_obj = esh_trace.get("local_cuts_added", 0)
                    if (
                        isinstance(local_cuts_added_obj, (int, float))
                        and int(local_cuts_added_obj) > 0
                    ):
                        local_cut_added = True
                else:
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
                    _record_interior_point(
                        incumbent,
                        "ecp_candidate",
                        {"objective": float(master_obj)},
                    )

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
                if _certified_gap_converged():
                    termination_reason = "gap"
                    stop_after_master_pool = True
                    break
                continue

            # c. Fix integers, solve NLP subproblem
            warm_start, warm_start_source = _fixed_nlp_warm_start(candidate, nlp_initial_point)
            nlp_subproblem_count += 1
            fixed_nlp_call_count += 1
            fixed_nlp_call_source_counts[candidate.source] += 1
            nlp_attempt = _solve_fixed_nlp_subproblem_attempt(
                evaluator,
                decomp.lb,
                decomp.ub,
                decomp.int_indices,
                x_master,
                nlp_solver,
                initial_point=warm_start,
            )
            x_nlp, obj_nlp = nlp_attempt.x, nlp_attempt.objective
            nlp_status_name = _fixed_nlp_status_name(nlp_attempt)
            fixed_nlp_call_status_counts[nlp_status_name] += 1
            incumbent_update = "not_feasible"

            if x_nlp is not None:
                if obj_nlp is not None and obj_nlp < UB:
                    multipliers = nlp_attempt.multipliers
                    accept_incumbent(x_nlp, obj_nlp, multipliers)
                    incumbent_update = "improved"
                else:
                    incumbent_update = "not_improved"

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

                safe_integer_cut_status = nlp_status_name in {
                    "failed",
                    "infeasible",
                    "unbounded",
                    "error",
                }
                if (
                    safe_integer_cut_status
                    and add_no_good_cuts
                    and (not decomp.general_integer_indices or integer_binary_expansion is not None)
                ):
                    _add_no_good_cut(
                        x_master,
                        decomp.binary_indices,
                        oa_A_rows,
                        oa_b_rows,
                        n_vars,
                        oa_cut_relaxable=oa_cut_relaxable,
                        cut_provenance=cut_provenance,
                        integer_binary_expansion=integer_binary_expansion,
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

            fixed_nlp_manager.record_call_result(
                candidate,
                iteration=int(iteration),
                elapsed=time.perf_counter() - t_start,
                success=x_nlp is not None,
            )
            _record_fixed_nlp_trace(
                iteration_record,
                candidate,
                status=nlp_status_name,
                objective=obj_nlp,
                incumbent_update=incumbent_update,
                warm_start_source=warm_start_source,
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

            if _certified_gap_converged():
                termination_reason = "gap"
                stop_after_master_pool = True
                break

        if stop_after_master_pool:
            iteration_record["termination_reason"] = termination_reason
        iteration_cuts_added = int(len(oa_A_rows) - cuts_before)
        incumbent_improved = incumbent_obj is not None and (
            incumbent_obj_before_iteration is None
            or float(incumbent_obj) < float(incumbent_obj_before_iteration) - 1e-12
        )
        if shot_solution_limit_state is not None:
            iteration_record["mip_solution_limit_update"] = (
                shot_solution_limit_state.observe_iteration(
                    incumbent_improved=incumbent_improved,
                    cuts_added=iteration_cuts_added,
                    master_status=str(iteration_record["master_status"]),
                )
            )
        iteration_record.update(
            {
                "lb": _trace_value(LB),
                "ub": _trace_value(UB),
                "gap": _trace_value(_compute_gap(LB, UB)),
                "cuts_added": iteration_cuts_added,
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
    certified_gap = _certified_gap_value()
    bound = certified_LB if _trace_value(certified_LB) is not None else None
    reported_gap = certified_gap if bound is not None and UB < 1e19 else None
    final_reason = termination_reason
    if final_reason is None:
        if wall_time >= time_limit:
            final_reason = "time_limit"
        elif incumbent is not None and _certified_gap_converged():
            final_reason = "gap"
        else:
            final_reason = "iteration_limit"

    if incumbent is not None and incumbent_obj is not None:
        status = "optimal" if _certified_gap_converged() else "feasible"
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
            gap_certified=bool(reported_gap is not None),
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
        gap_certified=False,
    )
