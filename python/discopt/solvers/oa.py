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
import math
import os
import time
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import numpy as np

from discopt.modeling.core import Constraint, Model, ObjectiveSense, SolveResult, VarType
from discopt.solvers import pounce_incumbent_options, pounce_option_defaults
from discopt.solvers._gap import bound_inversion_tolerance, optimality_gap
from discopt.solvers.mip_nlp_candidates import FixedNLPCandidate, FixedNLPCandidateManager
from discopt.solvers.mip_nlp_options import (
    FP_OPTION_KEYS,
    GOA_AMP_ONLY_OPTION_KEYS,
    GOA_AMP_OPTION_DEFAULTS,
    MIPNLPShotConfig,
)

if TYPE_CHECKING:
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt.solvers import SolveStatus

logger = logging.getLogger(__name__)

#: While OA holds no incumbent, the fraction of the remaining budget the master
#: MILP is allowed to consume (#1062). OA's product with no incumbent is nothing
#: at all, and the master is entitled to stop at its own time limit and still
#: return a perfectly good integer assignment — but the fixed-integer NLP that
#: turns that assignment into an incumbent runs *after* the candidate loop's
#: ``elapsed >= time_limit`` check. Handing the master 100% of the budget
#: therefore makes OA discard the answer it is holding. Measured on ``rsyn0840m``
#: at 60 s: master 60.34 s (100% of the solve), fixed-NLP calls 0, obj None —
#: against ``rsyn0805m`` on the same route, whose master takes 0.14 s and which
#: reaches a proved optimum in 1.2 s.
_MASTER_NO_INCUMBENT_BUDGET_FRAC = 0.9


def _master_time_budget(
    remaining: float,
    *,
    has_incumbent: bool,
    checkin_remaining: Optional[float] = None,
) -> float:
    """Time limit for one master MILP solve, reserving room for the fixed NLP.

    Once an incumbent exists OA has something to return, so the master may use
    everything that is left. Until then it may not, or a master that exhausts
    the budget leaves OA reporting ``status=unknown, obj=None`` while a usable
    integer assignment sits in ``master_result.x``.

    The reserve is a *fraction* rather than a constant so it never starves as
    the budget shrinks, and it is a no-op whenever the master finishes early —
    which is the common case. Stopping the master early costs only master
    optimality, not soundness: a MILP master truncated at its time limit still
    yields a valid dual bound for a relaxation of the MINLP.

    ``checkin_remaining`` is the caller's *soft* deadline: seconds from now by
    which the OA loop must return control to the top of the iteration, where the
    ``termination_hook`` runs. Without it a caller that budgets OA by progress
    cannot act, because the budget it granted is exactly what the first master
    expands to fill -- measured on ``rsyn0840m`` at a 60 s route budget, the
    master ran to ~55 s and the hook's second call arrived too late to leave the
    caller's fallback anything to work with (#1066). This is not the per-iteration
    master cap falsified in ``docs/dev/performance-plan.md`` §22.2: that one
    shrank *every* master to force more rounds and produced weaker bounds; this
    truncates at most the one master that would cross the caller's deadline.
    """
    budget = remaining
    if not has_incumbent and math.isfinite(remaining) and remaining > 0.0:
        budget = remaining * _MASTER_NO_INCUMBENT_BUDGET_FRAC
    if (
        checkin_remaining is not None
        and math.isfinite(checkin_remaining)
        and checkin_remaining > 0.0
    ):
        budget = min(budget, float(checkin_remaining))
    return budget


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
            if arr.ndim == 1 and arr.size == 0:
                return []
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
    """The row as a tuple of Python floats.

    ``ndarray.tolist()`` is the same conversion the element-wise ``float(v)``
    generator did -- a ``float64`` becomes a Python ``float``, with identical
    value, equality and hash -- but it runs in C over the whole buffer instead of
    building one generator frame per element. That matters because this is the
    per-cut cost of the provenance ledger, paid on the FULL master width for every
    row: profiled on ``squfl020-150`` (3020 columns) at a 60 s limit, the
    generator alone was 159.5 M calls and 11.5 s, and ``add_row`` 20.8 s
    cumulative -- 31% of the solve spent recording cuts rather than generating
    them. The values are unchanged, so this is bound-neutral by construction
    (CLAUDE.md §5): same coefficients, same dedup keys, same tree.
    """
    return tuple(np.asarray(values, dtype=np.float64).reshape(-1).tolist())


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
        "for integer regularized masters. Choose a linear regularization mode such "
        "as add_regularization='level_L1', 'level_L_infinity', or 'grad_lag'."
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
    #: The subsolver's own terminal code (``NLPResult.raw_status``), kept because
    #: ``status`` collapses several distinct Ipopt outcomes onto ``ERROR`` and the
    #: collapse is not reversible. Diagnosing #1141 needed exactly this
    #: distinction: 401 fixed-NLP solves behind one OA separator returned 281
    #: successes, 60 ``Infeasible_Problem_Detected`` and 57
    #: ``Error_In_Step_Computation``, and the callback trace recorded all 118
    #: failures identically as "failed". A genuinely infeasible integer assignment
    #: and a subsolver that fell over are different problems with different fixes.
    raw_status: Optional[int] = None
    #: The point the subsolver exited at, even when the attempt was NOT accepted.
    #: Never an incumbent candidate and never used as one -- ``x`` is the accepted
    #: point and stays ``None`` on a failure. This is for *verifying* a failure
    #: verdict (e.g. re-measuring the constraint violation at the exit point of an
    #: "infeasible" solve) rather than for using it.
    raw_x: Optional[np.ndarray] = None


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
    #: #1066: first column of the disaggregated perspective epigraph block, one
    #: column per term in ``perspective_terms``. ``None`` means the master
    #: carries the single aggregate epigraph and nothing else.
    perspective_start: Optional[int] = None
    perspective_terms: tuple[tuple[int, int, float], ...] = ()


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
        eta_value = float(mip_start_objective)
        if master.perspective_start is not None:
            # eta carries the residual objective in a disaggregated master, so a
            # start that put the whole objective here would violate the very
            # epigraph rows it is meant to satisfy.
            term_values = np.array(
                [q * start[xc] * start[xc] for xc, _yc, q in master.perspective_terms],
                dtype=np.float64,
            )
            eta_value -= float(term_values.sum())
            for k, value in enumerate(term_values):
                lo_k, hi_k = master.bounds[master.perspective_start + k]
                full[master.perspective_start + k] = min(
                    max(float(value), float(lo_k)), float(hi_k)
                )
        lo, hi = master.bounds[next_index]
        full[next_index] = min(max(eta_value, float(lo)), float(hi))
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
    from discopt._relax.convexity import classify_oa_cut_convexity
    from discopt._relax.gdp_reformulate import _extract_body_coeffs, _is_linear
    from discopt._tape_nlp_evaluator import make_evaluator

    # #1063: go through the canonical funnel, NOT ``NLPEvaluator(model)``.
    # ``_relax/nlp_evaluator`` imports jax at module scope, so constructing it
    # directly puts JAX in ``sys.modules`` and pays a cold XLA compile — measured
    # at 3m0.5s for ``jit_concat_constraints`` on squfl015-060 against a 60 s
    # budget, i.e. the OA path could not return at all. ``make_evaluator`` hands
    # back the tape evaluator (default-ON since #75) and only falls back to JAX
    # when the model is not tape-representable.
    evaluator = make_evaluator(model)
    # #1064: the perspective strengthening of the objective epigraph cut needs
    # the model's ``q*x**2``-over-semicontinuous-``x`` structure, which is a
    # property of the model and not of any node, so it is read once here and
    # carried on the evaluator -- ``_add_oa_cuts`` is reached from a dozen call
    # sites and receives the evaluator but not the model. ``_BoundsProxy``
    # forwards unknown attributes to the evaluator it wraps, so a bounds-scoped
    # copy sees the same table. See ``_relax.perspective.perspective_objective_terms``.
    if _perspective_oa_terms_enabled():
        from discopt._relax.perspective import perspective_objective_terms

        evaluator._perspective_oa_terms = perspective_objective_terms(model)
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


#: Smallest wall budget handed to a subsolve. A budget of zero (or a negative
#: one, once the deadline has already passed) is not "run instantly" to POUNCE
#: -- it is an option value the backend may reject, which silently restores the
#: UNBOUNDED behaviour this floor exists to prevent. Call sites that can skip
#: the subsolve entirely check the deadline themselves; this floor only keeps a
#: subsolve that *must* run from being handed a degenerate limit.
_NLP_WALL_FLOOR_S = 0.1


def _time_left(t_start: float, time_limit: float) -> float:
    """Unfloored seconds left before ``t_start + time_limit`` (negative once past)."""
    return float(time_limit) - (time.perf_counter() - float(t_start))


def _remaining_wall(t_start: float, time_limit: float) -> float:
    """Seconds left before ``t_start + time_limit``, floored at ``_NLP_WALL_FLOOR_S``.

    The single source of the MIP-NLP deadline contract (#1105): every subsolve
    launched inside a time-limited MIP-NLP region takes its wall budget from
    here, so ``time_limit`` covers the whole run rather than each phase
    separately. Before this existed the fixed-NLP, feasibility-restoration and
    initialization NLPs ran with no budget at all -- on ``kondili_recipe_pr46``
    a single feasibility subproblem launched at t=22 s ran 69.85 s against a
    60 s limit, and the solve returned at 92.4 s (1.54x).
    """
    return max(_time_left(t_start, time_limit), _NLP_WALL_FLOOR_S)


def _solve_nlp_attempt(
    evaluator,
    lb,
    ub,
    nlp_solver: str,
    max_iter: int = 200,
    x0=None,
    max_wall_time: Optional[float] = None,
) -> _NLPAttempt:
    """Solve an NLP with given bounds, retaining solver multipliers.

    ``max_wall_time`` is the subsolve's share of the caller's deadline, in
    seconds. Measured on POUNCE 0.10 (``scratchpad/issue1105/probe_maxwall.py``,
    400-var nonconvex NLP): unbounded 25.72 s, ``max_wall_time=0.5`` -> 0.517 s,
    ``max_wall_time=2.0`` -> 2.073 s, both returning ``TIME_LIMIT``. Leaving it
    ``None`` keeps the historical unbounded behaviour for callers with no
    deadline of their own.
    """
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

        # This point becomes OA's incumbent and its reported objective, so it takes
        # the incumbent options (#945). Without them the fixed-NLP subproblem returns
        # a point outside its declared box and OA certifies a super-optimal incumbent.
        opts = pounce_option_defaults()
        opts.update(pounce_incumbent_options())
        opts.update({"max_iter": max_iter})
        if max_wall_time is not None:
            opts["max_wall_time"] = max(float(max_wall_time), _NLP_WALL_FLOOR_S)
        result = solve_nlp(evaluator, x0, options=opts)

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL:
            return _NLPAttempt(
                x=result.x,
                objective=float(evaluator.evaluate_objective(result.x)),
                multipliers=result.multipliers,
                status=result.status,
                raw_status=getattr(result, "raw_status", None),
            )

        # Accept iteration- and time-limited results if the solution is primal
        # feasible. The IPM may not certify dual convergence (code 4: stalled)
        # yet still find a valid primal point, which is sufficient for OA
        # linearization cuts. ``TIME_LIMIT`` joined this set with #1105: once a
        # subsolve carries a wall budget, stopping on the clock is the ordinary
        # outcome, and dropping a feasible point purely because the clock (not
        # the iteration counter) stopped it would lose incumbents the unbounded
        # code path used to find. The point is screened by
        # ``_is_primal_feasible`` either way, so nothing infeasible is admitted.
        if (
            result.status in (SolveStatus.ITERATION_LIMIT, SolveStatus.TIME_LIMIT)
            and result.x is not None
        ):
            if _is_primal_feasible(evaluator, result.x):
                return _NLPAttempt(
                    x=result.x,
                    objective=float(evaluator.evaluate_objective(result.x)),
                    multipliers=result.multipliers,
                    status=result.status,
                    # Carried on the ACCEPTED path too (#1141): without it a
                    # restoration that converged and one that merely stopped at the
                    # iteration limit both record ``raw=None``, and the outcome
                    # tally cannot tell them apart — the exact ambiguity this
                    # field exists to remove.
                    raw_status=getattr(result, "raw_status", None),
                )
        # Not accepted as a usable point — but WHY is not thrown away (#1141).
        # ``status``/``raw_status`` ride along on the empty attempt so the caller
        # (and the callback trace) can tell "this assignment is infeasible" from
        # "the subsolver fell over", which the single empty attempt could not.
        return _NLPAttempt(
            x=None,
            objective=None,
            multipliers=None,
            status=result.status,
            raw_status=getattr(result, "raw_status", None),
            raw_x=None if result.x is None else np.asarray(result.x, dtype=np.float64),
        )
    except Exception as exc:  # noqa: BLE001 - an empty attempt is handled by the caller
        # Capability-disabling: no NLP point means no OA linearization from this
        # node, which reads as "OA cuts don't help" rather than as a failure.
        logger.debug("OA NLP subproblem raised: %s: %s", type(exc).__name__, exc)
    return _NLPAttempt(x=None, objective=None, multipliers=None)


def _solve_nlp(
    evaluator,
    lb,
    ub,
    nlp_solver: str,
    max_iter: int = 200,
    x0=None,
    max_wall_time: Optional[float] = None,
):
    """Solve an NLP with given bounds. Returns (x, obj) or (None, None)."""
    attempt = _solve_nlp_attempt(
        evaluator, lb, ub, nlp_solver, max_iter=max_iter, x0=x0, max_wall_time=max_wall_time
    )
    return attempt.x, attempt.objective


def _maybe_return_nlp_attempt(attempt: _NLPAttempt, return_attempt: bool):
    if return_attempt:
        return attempt
    return attempt.x, attempt.objective


def _fixed_nlp_status_label(attempt: _NLPAttempt) -> str:
    """Name a fixed-NLP subproblem outcome for the callback trace (#1141).

    ``"feasible"`` when the attempt produced a usable point. Otherwise the
    subsolver's own verdict, so the record distinguishes the outcomes that used to
    collapse into one ``"failed"``:

    * ``"infeasible_local"`` — Ipopt/POUNCE code 2, restoration converged to a
      local minimizer of the constraint violation. On a **convex** subproblem that
      is a genuine infeasibility proof; on a nonconvex one it proves nothing, which
      is why it is labelled ``local`` here and is NOT mapped to
      :attr:`SolveStatus.INFEASIBLE` upstream (see
      :data:`discopt.solvers.nlp_ipopt.IPOPT_LOCALLY_INFEASIBLE`). Nothing in this
      driver prunes on the label; it is a diagnostic.
    * ``"failed:<code>"`` — any other terminal code, named rather than erased.
    * ``"failed"`` — the subsolver raised, so there is no code to report.
    """
    if attempt.x is not None:
        return "feasible"
    raw = attempt.raw_status
    if raw is None:
        return "failed"
    from discopt.solvers.nlp_ipopt import IPOPT_LOCALLY_INFEASIBLE

    if int(raw) == IPOPT_LOCALLY_INFEASIBLE:
        return "infeasible_local"
    return f"failed:{int(raw)}"


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
    max_wall_time: Optional[float] = None,
):
    """Solve the continuous NLP relaxation (all integers relaxed)."""
    attempt = _solve_nlp_attempt(
        evaluator, lb, ub, nlp_solver, x0=initial_point, max_wall_time=max_wall_time
    )
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
    max_wall_time: Optional[float] = None,
):
    """Fix integers at master values and solve NLP subproblem."""
    sub_lb = lb.copy()
    sub_ub = ub.copy()
    for idx in int_indices:
        val = _round_integral_to_bounds(x_master[idx], lb[idx], ub[idx])
        sub_lb[idx] = val
        sub_ub[idx] = val

    proxy = _BoundsProxy(evaluator, sub_lb, sub_ub)
    attempt = _solve_nlp_attempt(
        proxy, sub_lb, sub_ub, nlp_solver, x0=initial_point, max_wall_time=max_wall_time
    )
    return _maybe_return_nlp_attempt(attempt, return_attempt)


#: Optional ``_solve_nlp_subproblem`` keywords, in the order they are dropped
#: when the bound implementation does not accept them (a stub or an older
#: signature). Richest first: losing the wall budget only costs the deadline
#: contract, losing ``return_attempt`` costs the status, losing
#: ``initial_point`` costs the warm start.
_FIXED_NLP_OPTIONAL_KWARGS = ("max_wall_time", "return_attempt", "initial_point")


def _solve_fixed_nlp_subproblem_attempt(
    evaluator,
    lb,
    ub,
    int_indices,
    x_master,
    nlp_solver,
    *,
    initial_point=None,
    max_wall_time: Optional[float] = None,
) -> _NLPAttempt:
    """Call the fixed-NLP helper and retain status when the implementation supports it.

    Degrades one keyword at a time rather than all-or-nothing: a
    ``_solve_nlp_subproblem`` that predates any of these keywords still runs,
    and a ``TypeError`` that does not name one of the *remaining* optional
    keywords is a real error and propagates.
    """
    args = (evaluator, lb, ub, int_indices, x_master, nlp_solver)
    optional: dict[str, Any] = {"initial_point": initial_point, "return_attempt": True}
    if max_wall_time is not None:
        optional["max_wall_time"] = float(max_wall_time)
    for dropped in range(len(_FIXED_NLP_OPTIONAL_KWARGS) + 1):
        skip = _FIXED_NLP_OPTIONAL_KWARGS[:dropped]
        kwargs = {name: value for name, value in optional.items() if name not in skip}
        try:
            result = _solve_nlp_subproblem(*args, **kwargs)
            break
        except TypeError as exc:
            remaining = _FIXED_NLP_OPTIONAL_KWARGS[dropped:]
            if not any(name in str(exc) for name in remaining):
                raise
    return _coerce_nlp_attempt(result)


def _infeasible_nogood_enabled() -> bool:
    """``DISCOPT_OA_INFEASIBLE_NOGOOD``: exclude an integer assignment the fixed NLP
    *proved* infeasible, even when ``add_no_good_cuts`` is off.

    Default-OFF. It adds rows to the master, so it changes the master's dual bound
    (CLAUDE.md §5 regime 2) and ships behind a flag until a corpus panel clears
    both bars. ``=1`` turns it on.
    """
    return os.environ.get("DISCOPT_OA_INFEASIBLE_NOGOOD", "0") not in (
        "0",
        "",
        "false",
        "False",
    )


def _assignment_proven_infeasible(
    attempt: _NLPAttempt,
    evaluator,
    constraint_convex_mask: Optional[list[bool]],
    tol: float = 1e-6,
) -> bool:
    """Is the fixed-integer subproblem PROVEN infeasible, not merely unsolved?

    A no-good cut deletes an integer assignment from the master permanently, so it
    is sound only against a proof. "The NLP did not return a point" is not one:
    #1141 measured 401 fixed-NLP solves behind one separator returning 281
    successes, 60 ``Infeasible_Problem_Detected`` and **57**
    ``Error_In_Step_Computation`` — and excluding an assignment on the strength of
    a step-computation failure would delete a subtree that may hold the optimum.

    Two conditions, both required:

    * the subsolver's own verdict is
      :data:`~discopt.solvers.nlp_ipopt.IPOPT_LOCALLY_INFEASIBLE` (Ipopt code 2:
      restoration converged to a local minimizer of the constraint violation, with
      the violation still positive), and
    * **every** constraint of the model defines a convex feasible set
      (``constraint_convex_mask`` all true). Fixing the integers narrows the box,
      which preserves convexity, so the subproblem's feasible set is convex — and
      on a convex set the violation measure is convex, so a *local* minimizer of it
      is a *global* one and a positive value is a genuine emptiness proof. Without
      that certificate code 2 says only that the algorithm got stuck, which proves
      nothing (see ``IPOPT_LOCALLY_INFEASIBLE``).

    Plus one independent check that costs one evaluation: the point the subsolver
    exited at must still violate a constraint by more than ``tol``. A code 2
    reported from an essentially feasible point is a solver artefact, not a proof,
    and this is the cheapest way to refuse it without trusting the code alone.
    """
    from discopt.solvers.nlp_ipopt import IPOPT_LOCALLY_INFEASIBLE

    if attempt.raw_status is None or int(attempt.raw_status) != IPOPT_LOCALLY_INFEASIBLE:
        return False
    if not constraint_convex_mask or not all(constraint_convex_mask):
        return False
    if attempt.raw_x is None:
        return False
    violations, _signs = _constraint_violation_data(evaluator, attempt.raw_x)
    return bool(violations.size and float(np.max(violations)) > tol)


def _fixed_subproblem_rigorously_infeasible(
    evaluator,
    lb,
    ub,
    int_indices,
    x_master,
    tol: float = 1e-6,
) -> bool:
    """Return true only when the fixed-integer NLP is provably infeasible.

    A local NLP failure is not enough to justify excluding the integer
    assignment. This check certifies infeasibility only for the all-fixed case,
    where evaluating the single remaining point is a complete feasibility test.
    """
    sub_lb = lb.copy()
    sub_ub = ub.copy()
    for idx in int_indices:
        val = _round_integral_to_bounds(x_master[idx], lb[idx], ub[idx])
        sub_lb[idx] = val
        sub_ub[idx] = val

    if not np.all(sub_lb >= sub_ub - 1e-12):
        return False

    if evaluator.n_constraints == 0:
        return False

    try:
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        cl, cu = _infer_constraint_bounds(evaluator)
        x_point = np.array(
            [0.5 * (sub_lb[i] + sub_ub[i]) for i in range(evaluator.n_variables)],
            dtype=np.float64,
        )
        cons = np.asarray(evaluator.evaluate_constraints(x_point), dtype=np.float64)
        if not np.all(np.isfinite(cons)):
            return False
        return bool(np.any(cons < cl - tol) or np.any(cons > cu + tol))
    except Exception:
        return False


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


def _elastic_restoration_enabled() -> bool:
    """``DISCOPT_OA_ELASTIC_RESTORATION``: pose the OA feasibility subproblem as a
    constrained elastic NLP instead of an unconstrained violation merit.

    Default-**ON** since the #1141 graduation panel; ``=0`` is the opt-out. It
    changes which point the feasibility cut is built at, hence the master's rows and
    its dual bound — CLAUDE.md §5 regime 2 — so it shipped default-OFF until that
    panel. Over the 119 vendored MINLPLib instances it is **outcome-neutral**:
    0 soundness violations, certificates 23/23, 0 bounds moved, wall −1.5 %, and on
    all 7 rows where it actually runs the status, objective and wall are unchanged
    while ``Error_In_Step_Computation`` failures turn into convergences
    (``m3`` 313 -> 0, ``clay0303hfsg`` 46 -> 0). It pays on the class the issue is
    about: 2.3x on ``meanvarx`` (the one real corpus row of that class, same
    certificate) and 5–8x on reconstructions of ``portfol_classical050_1``.

    Applies **only on a model whose constraints are all certified convex** (the
    caller passes the mask). That gate is not a tuning knob, it is the condition
    under which the elastic subproblem means anything: on a convex feasible set it
    is a convex NLP, so its solution is the *global* minimum-violation point and
    the restoration actually certifies something. On a nonconvex model it is one
    more local solve — and, measured, a more expensive one. Every corpus row where
    the elastic form was slower was nonconvex and produced no incumbent in either
    arm (`bchoco06/07/08`, `beuster`, `heatexch_gen2`: +3.6 to +14.1 s each,
    `constraints_convex=False`, `incumbent=False`); every convex row was neutral or
    faster. See ``docs/dev/performance-plan.md`` §24.
    """
    return os.environ.get("DISCOPT_OA_ELASTIC_RESTORATION", "1") not in (
        "0",
        "",
        "false",
        "False",
    )


class _ElasticFeasibilityEvaluator:
    """The OA feasibility subproblem as a *constrained* elastic NLP (#1141).

    :class:`_FeasibilityEvaluator` poses restoration as an **unconstrained**
    minimization of a violation merit, and reports a **zero** Hessian for it. That
    is not a modelling nicety — it is the reason restoration never converges. With
    no constraints, an interior-point method's KKT matrix is ``σ_f ∇²f + Σ``, and
    ``σ_f ∇²f`` is identically zero here; away from the variable bounds ``Σ`` is
    tiny too, so the matrix is numerically singular, inertia correction runs out,
    and the solve exits ``Error_In_Step_Computation``. #1141 measured exactly that
    on ``portfol_classical050_1``: **0 of 60** restorations converged, 57 of them
    with code −3, and switching the merit norm (L1 / L2 / L∞) changed nothing —
    because the norm is not what is zero, the Hessian is.

    This is the textbook elastic formulation instead (Fletcher & Leyffer 1994;
    what BONMIN and SHOT solve), over ``z = [x | u]``::

        min  ‖u‖   s.t.   cl ≤ g(x) ± u ≤ cu,   u ≥ 0,   integers fixed

    with one slack per constraint row for ``L1``/``L2`` and a single shared slack
    for ``L_infinity``. Three properties matter:

    * it is **smooth** — the ``max``/``abs`` of the merit moves into the
      constraints, where an IPM handles it natively;
    * it has **real constraints and a real Hessian** — the ``x`` block is the
      original problem's Lagrangian Hessian and the ``u`` block is ``2I`` (L2) or
      structurally carried by the constraint Jacobian (L1/L∞), so the KKT system
      is the one the solver was designed for rather than a bordered zero;
    * its start is **feasible by construction** — ``u`` is initialised to the
      violation at the master point, so the IPM begins inside its own feasible
      set and any progress it makes strictly reduces the violation.

    The point handed back is the ``x`` block. Soundness is unchanged either way:
    the caller only uses the point to *linearize at*, and an OA cut taken at any
    point is a supporting hyperplane of a convex constraint.
    """

    def __init__(self, evaluator, lb, ub, feasibility_norm: str):
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        self._eval = evaluator
        self._lb = np.asarray(lb, dtype=np.float64)
        self._ub = np.asarray(ub, dtype=np.float64)
        self._norm = feasibility_norm
        self._n = int(evaluator.n_variables)
        cl, cu = _infer_constraint_bounds(evaluator)
        self._cl = np.asarray(cl, dtype=np.float64)
        self._cu = np.asarray(cu, dtype=np.float64)
        # One elastic row per FINITE constraint bound. An equality contributes
        # both, which is what makes ``|g| <= u`` come out of a formulation that
        # only ever writes ``<=``.
        rows: list[tuple[int, int]] = []  # (original row, +1 upper / -1 lower)
        for i in range(self._cl.shape[0]):
            if self._cu[i] < 1e19:
                rows.append((i, +1))
            if self._cl[i] > -1e19:
                rows.append((i, -1))
        self._rows = rows
        # Slack layout: one per original row, or a single shared one for L∞.
        self._shared = feasibility_norm not in ("L1", "L2")
        self._k = 1 if self._shared else self._cl.shape[0]

    # -- layout ----------------------------------------------------------------
    @property
    def n_variables(self) -> int:
        return self._n + self._k

    @property
    def n_constraints(self) -> int:
        return len(self._rows)

    @property
    def variable_bounds(self):
        return (
            np.concatenate([self._lb, np.zeros(self._k)]),
            np.concatenate([self._ub, np.full(self._k, 1e20)]),
        )

    def constraint_bounds(self) -> list[tuple[float, float]]:
        """Row bounds for the elastic rows, in this evaluator's own row order."""
        out: list[tuple[float, float]] = []
        for i, side in self._rows:
            if side > 0:
                out.append((-1e20, float(self._cu[i])))
            else:
                out.append((float(self._cl[i]), 1e20))
        return out

    def _slack_of(self, i: int) -> int:
        return 0 if self._shared else i

    def start_point(self, x0: np.ndarray) -> np.ndarray:
        """``[x0 | violation(x0)]`` — elastic-feasible by construction."""
        violations, _signs = _constraint_violation_data(self._eval, x0)
        if self._shared:
            u = np.array([float(np.max(violations)) if violations.size else 0.0])
        else:
            u = np.asarray(violations, dtype=np.float64).copy()
        return np.concatenate([np.asarray(x0, dtype=np.float64), np.maximum(u, 0.0)])

    # -- objective -------------------------------------------------------------
    def evaluate_objective(self, z):
        u = np.asarray(z, dtype=np.float64)[self._n :]
        if self._norm == "L2":
            return float(np.dot(u, u))
        return float(np.sum(u))

    def evaluate_gradient(self, z):
        g = np.zeros(self.n_variables, dtype=np.float64)
        if self._norm == "L2":
            g[self._n :] = 2.0 * np.asarray(z, dtype=np.float64)[self._n :]
        else:
            g[self._n :] = 1.0
        return g

    def evaluate_hessian(self, z):
        h = np.zeros((self.n_variables, self.n_variables), dtype=np.float64)
        if self._norm == "L2":
            idx = np.arange(self._n, self.n_variables)
            h[idx, idx] = 2.0
        return h

    def evaluate_lagrangian_hessian(self, z, obj_factor, lagrange):
        h = np.zeros((self.n_variables, self.n_variables), dtype=np.float64)
        if self._norm == "L2":
            idx = np.arange(self._n, self.n_variables)
            h[idx, idx] = 2.0 * float(obj_factor)
        x = np.asarray(z, dtype=np.float64)[: self._n]
        lam = np.asarray(lagrange, dtype=np.float64).ravel()
        # Both sides write ``+g_i``, so a row's multiplier accumulates with the
        # same sign whichever bound it came from.
        lam_orig = np.zeros(self._cl.shape[0], dtype=np.float64)
        for r, (i, _side) in enumerate(self._rows):
            if r < lam.shape[0]:
                lam_orig[i] += lam[r]
        # ``obj_factor = 0``: the elastic objective touches only the ``u`` block,
        # which is filled above; the ``x`` block is purely the constraints'.
        hx = np.asarray(self._eval.evaluate_lagrangian_hessian(x, 0.0, lam_orig), dtype=np.float64)
        h[: self._n, : self._n] = hx
        return h

    # -- constraints -----------------------------------------------------------
    def evaluate_constraints(self, z):
        z = np.asarray(z, dtype=np.float64)
        x, u = z[: self._n], z[self._n :]
        g = np.asarray(self._eval.evaluate_constraints(x), dtype=np.float64)
        out = np.empty(len(self._rows), dtype=np.float64)
        for r, (i, side) in enumerate(self._rows):
            out[r] = g[i] - side * u[self._slack_of(i)]
        return out

    def evaluate_jacobian(self, z):
        z = np.asarray(z, dtype=np.float64)
        x = z[: self._n]
        jg = np.asarray(self._eval.evaluate_jacobian(x), dtype=np.float64)
        j = np.zeros((len(self._rows), self.n_variables), dtype=np.float64)
        for r, (i, side) in enumerate(self._rows):
            j[r, : self._n] = jg[i]
            j[r, self._n + self._slack_of(i)] = -float(side)
        return j


def _solve_feasibility_subproblem(
    evaluator,
    lb,
    ub,
    int_indices,
    x_master,
    nlp_solver,
    feasibility_norm,
    max_wall_time: Optional[float] = None,
    constraint_convex_mask: Optional[list[bool]] = None,
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
        if (
            _elastic_restoration_enabled()
            and evaluator.n_constraints > 0
            and bool(constraint_convex_mask)
            and all(constraint_convex_mask or ())
        ):
            x_feas = _solve_elastic_restoration(
                evaluator, sub_lb, sub_ub, x0, feasibility_norm, max_wall_time
            )
        else:
            proxy = _FeasibilityEvaluator(evaluator, sub_lb, sub_ub, feasibility_norm)
            # ``_solve_nlp_attempt`` rather than ``_solve_nlp``: the two differ only
            # in whether the subsolver's verdict survives, and #1141's whole
            # diagnosis turned on that verdict. Restoration falls back to the
            # clipped master point on failure, so a run where it NEVER converges
            # looks, from the outside, exactly like one where it always did.
            attempt = _solve_nlp_attempt(
                proxy, sub_lb, sub_ub, nlp_solver, x0=x0, max_wall_time=max_wall_time
            )
            raw = attempt.raw_status
            _RESTORATION_OUTCOMES[("merit", "ok" if raw in (0, 1) else f"raw={raw}")] += 1
            x_feas = attempt.x
        if x_feas is not None:
            candidate = np.clip(np.asarray(x_feas, dtype=np.float64), sub_lb, sub_ub)
            candidate_merit = _constraint_violation_merit(evaluator, candidate, feasibility_norm)
            if candidate_merit <= best_merit + 1e-9:
                best_x = candidate
    except Exception as exc:  # noqa: BLE001 - falls back to the clipped master point
        logger.debug(
            "OA feasibility restoration failed, keeping the clipped master point: %s: %s",
            type(exc).__name__,
            exc,
        )

    return best_x


#: Restoration outcomes of the LAST solve, by subsolver status label. Read by the
#: callback trace and by the panels; reset per ``solve_lp_nlp_bb`` call.
#:
#: #1141's diagnosis needed this and had to add it by hand: the shipped code
#: swallowed the restoration's outcome entirely (it falls back to the clipped
#: master point either way), so "restoration converged 0 of 60 times" was
#: invisible from the outside and the loop reported cuts as if they had been
#: built at a converged point.
_RESTORATION_OUTCOMES: Counter = Counter()


def _solve_elastic_restoration(
    evaluator,
    sub_lb,
    sub_ub,
    x0,
    feasibility_norm: str,
    max_wall_time: Optional[float],
):
    """Solve the elastic feasibility subproblem; return the ``x`` block or ``None``.

    Records the subsolver's own status in :data:`_RESTORATION_OUTCOMES` so a run
    that never converges says so instead of looking like one that did.
    """
    from discopt.solvers.nlp_pounce import solve_nlp

    proxy = _ElasticFeasibilityEvaluator(evaluator, sub_lb, sub_ub, feasibility_norm)
    if proxy.n_constraints == 0:
        return None
    opts = pounce_option_defaults()
    opts.update({"max_iter": 200})
    if max_wall_time is not None:
        opts["max_wall_time"] = max(float(max_wall_time), _NLP_WALL_FLOOR_S)
    result = solve_nlp(
        # ``solve_nlp`` is annotated against the concrete ``NLPEvaluator`` but
        # consumes it structurally (``_IpoptCallbacks`` calls the same seven
        # methods on whatever it is given) -- which is exactly how the sibling
        # ``_FeasibilityEvaluator`` reaches it, via the untyped ``_solve_nlp``
        # helper. The cast records that this class satisfies the same duck-typed
        # contract rather than widening it.
        cast(Any, proxy),
        proxy.start_point(x0),
        constraint_bounds=proxy.constraint_bounds(),
        options=opts,
    )
    raw = getattr(result, "raw_status", None)
    _RESTORATION_OUTCOMES[("elastic", "ok" if raw in (0, 1) else f"raw={raw}")] += 1
    if result.x is None:
        return None
    return np.asarray(result.x, dtype=np.float64)[: evaluator.n_variables]


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
        max_wall_time=_remaining_wall(t_start, time_limit),
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
            max_wall_time=_remaining_wall(t_start, time_limit),
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
            max_wall_time=_remaining_wall(t_start, time_limit),
            constraint_convex_mask=decomp.oa_constraint_mask,
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


#: #1064 §6 telemetry: how many perspective strengthenings were actually
#: applied to objective epigraph rows this process. A panel arm that reports
#: zero here measured the control twice, whatever the flag said.
_PERSPECTIVE_OA_CUT_APPLIED: list[int] = [0]


def _perspective_oa_terms_enabled() -> bool:
    """Thin re-export so ``oa.py`` does not import ``solver.py`` (cycle)."""
    from discopt._relax.perspective import perspective_oa_cut_enabled

    return perspective_oa_cut_enabled()


def _strengthen_objective_cut_perspective(coeffs, rhs, x_star, n_vars, terms):
    """Perspective-strengthen an OA objective epigraph row in place-safe form.

    ``coeffs``/``rhs`` are the row ``grad^T x - eta <= rhs``. For each
    ``(x_col, y_col, q)`` whose ``x`` is semicontinuous with indicator ``y``,
    the tangent's own constant ``-q*xbar**2`` is replaced by ``-q*xbar**2 * y``:
    subtract ``q*xbar**2`` from the ``y`` coefficient and from ``rhs``. At
    ``y = 1`` the two cancel and the row is unchanged; at ``y = 0``
    semicontinuity forces ``x = 0`` and the row correctly reads ``eta >= 0``
    instead of the tangent's slack ``eta >= -q*xbar**2``.

    Returns ``(coeffs, rhs, n_applied)``. ``n_applied`` is the CLAUDE.md §6
    executed count: a silent zero here is a strengthening that never happened,
    which is exactly how a no-op reads as a pass.
    """
    applied = 0
    for x_col, y_col, q in terms:
        if not (0 <= x_col < n_vars and 0 <= y_col < n_vars):
            # The master row is laid out over the model's flat columns; an index
            # past them means the two layouts disagree and the shift would land
            # on the wrong variable. Refuse the term rather than guess.
            continue
        xbar = float(x_star[x_col])
        if not np.isfinite(xbar):
            continue
        shift = q * xbar * xbar
        if not np.isfinite(shift) or shift <= 0.0:
            continue  # xbar == 0: the tangent is already the perspective cut
        coeffs[y_col] -= shift
        rhs -= shift
        applied += 1
    return coeffs, rhs, applied


#: #1066 §6 telemetry, the disaggregated counterparts of the counter above:
#: how many per-term epigraph rows were emitted, and how many objective cuts
#: were *refused* because a reference coordinate was not finite. A panel arm
#: reporting zero rows ran the aggregate master whatever the flag said.
_PERSPECTIVE_DISAGG_ROWS: list[int] = [0]
_PERSPECTIVE_DISAGG_REFUSED: list[int] = [0]
#: References dropped because their row carries no information (see
#: ``_PERSPECTIVE_MIN_ROW_COEFF``). Counted so "none were dropped" is
#: distinguishable from "the check never ran" (CLAUDE.md §6).
_PERSPECTIVE_DISAGG_NEAR_ZERO: list[int] = [0]

#: Smallest ``2*q*|z|`` worth a perspective row. The row is
#: ``2*q*z*x_k - q*z**2*y_k - s_k <= 0`` against a ``-1`` on ``s_k``, so once
#: ``2*q*|z|`` falls to the master backend's matrix floor the backend drops the
#: ``x_k`` term outright and what lands is ``s_k >= 0`` -- which the column's own
#: lower bound already says. This is HiGHS's default ``small_matrix_value``, the
#: value at which that discard provably happens.
#:
#: Measured on ``squfl020-150`` (#1066): references at ``z ~ 1.2e-16`` produced
#: 43 500 such rows, 63% of the master's 69 330. Because their ``q*z**2`` term
#: (~3e-32) also drove the row-scaling in ``milp_highs._prepare_cut_row``, each
#: reached HiGHS multiplied by ``2**46``, taking the master's coefficient range
#: to 6.4e18 -- and HiGHS then reported dual bounds ABOVE the master's own
#: provable optimum (558.044 against 557.84865).
_PERSPECTIVE_MIN_ROW_COEFF = 1e-9


def _perspective_disagg_enabled() -> bool:
    """Thin re-export so ``oa.py`` does not import ``solver.py`` (cycle)."""
    from discopt._relax.perspective import perspective_disaggregation_enabled

    return perspective_disaggregation_enabled()


@dataclass
class _PerspectiveEpigraph:
    """One epigraph column per separable perspective term (#1066).

    The master's aggregate ``eta`` is left covering only the *residual*
    objective ``f - sum_k q_k x_k**2``; each term gets its own column ``s_k``
    and its own Frangioni-Gentile rows

        ``2*q*z*x_k - q*z**2*y_k - s_k <= 0``

    one per reference ``z`` the search visits. The master objective is
    ``eta + sum_k s_k``, so the sum is still a valid underestimator of ``f``
    (each row is valid on both integral values of ``y_k`` -- see
    ``_relax.perspective``), and the master's LP relaxation is free to pick a
    *different* reference for each term at a fractional point, which is the one
    thing the aggregate row cannot do.
    """

    terms: tuple[tuple[int, int, float], ...]
    rows: list[tuple[int, float]] = field(default_factory=list)
    _seen: set[tuple[int, float]] = field(default_factory=set)
    #: Column-wise copies of ``terms`` and of the pooled references, so a whole
    #: pool can be scored in one vectorised pass. The pool reaches tens of
    #: thousands of rows on the squfl family (69 NLP solves x ~930 live terms,
    #: measured on squfl025-040); materialising a dense master row per pool
    #: entry just to test it for violation cost more than the cuts bought.
    _x_cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    _y_cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    _q: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    _row_k: list[int] = field(default_factory=list)
    _row_z: list[float] = field(default_factory=list)
    #: Row indices created since the last drain. A per-term row born with the
    #: aggregate cut it was split out of must reach the master *with* it: it is
    #: exactly tight at the point it was generated from, so a violation filter
    #: scores it at 0 and drops it, and the term's ``s_k`` is then left free at
    #: its lower bound while the residual row it was subtracted from is already
    #: in. That asymmetry -- and not the disaggregation itself -- is what made
    #: the first cut of this report a *looser* dual bound than the aggregate
    #: row on squfl025-040 (120.652 vs 142.510).
    _pending: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._x_cols = np.array([t[0] for t in self.terms], dtype=np.intp)
        self._y_cols = np.array([t[1] for t in self.terms], dtype=np.intp)
        self._q = np.array([t[2] for t in self.terms], dtype=np.float64)

    def add(self, k: int, z: float) -> bool:
        """Record a reference for term ``k``; False if it is already carried."""
        key = (int(k), float(z))
        if key in self._seen:
            return False
        self._seen.add(key)
        self.rows.append(key)
        self._row_k.append(int(k))
        self._row_z.append(float(z))
        self._pending.append(len(self.rows) - 1)
        _PERSPECTIVE_DISAGG_ROWS[0] += 1
        return True

    def drain_pending(self) -> list[int]:
        """Row indices created since the last call, and clear the list."""
        pending = self._pending
        self._pending = []
        return pending

    def violations(self, master_x, *, perspective_start: int) -> np.ndarray:
        """``a^T x`` of every pooled row (rhs is 0), vectorised over the pool."""
        if not self.rows:
            return np.zeros(0, dtype=np.float64)
        x = np.asarray(master_x, dtype=np.float64).ravel()
        ks = np.asarray(self._row_k, dtype=np.intp)
        zs = np.asarray(self._row_z, dtype=np.float64)
        q = self._q[ks]
        scores = (
            2.0 * q * zs * x[self._x_cols[ks]]
            - q * zs * zs * x[self._y_cols[ks]]
            - x[perspective_start + ks]
        )
        return np.asarray(scores, dtype=np.float64)

    def row_for(self, index: int, *, n_master: int, perspective_start: int) -> np.ndarray:
        k, z = self.rows[index]
        x_col, y_col, q = self.terms[k]
        row = np.zeros(n_master, dtype=np.float64)
        row[x_col] = 2.0 * q * z
        row[y_col] = -q * z * z
        row[perspective_start + k] = -1.0
        return row

    def term_values(self, x) -> np.ndarray:
        """``q_k * x_k**2`` per term: the epigraph values at a point."""
        arr = np.asarray(x, dtype=np.float64).ravel()
        return np.array([q * arr[xc] * arr[xc] for xc, _yc, q in self.terms], dtype=np.float64)


def _perspective_epigraph_for(terms, n_vars: int) -> Optional[_PerspectiveEpigraph]:
    """Build the epigraph, or ``None`` when the term table cannot carry one.

    Every term must be usable: the split below removes *all* of them from every
    aggregate row, so a term that can be removed from one row and not another
    would leave the master double-counting it -- an over-estimate of ``f``, i.e.
    an invalid bound. Refuse the whole disaggregation rather than part of it.
    """
    if not terms:
        return None
    for x_col, y_col, q in terms:
        if not (0 <= int(x_col) < n_vars and 0 <= int(y_col) < n_vars):
            return None
        if not np.isfinite(q) or q <= 0.0:
            return None
    return _PerspectiveEpigraph(terms=tuple((int(a), int(b), float(c)) for a, b, c in terms))


def _disaggregate_objective_cut(coeffs, rhs, x_star, epigraph: _PerspectiveEpigraph):
    """Move every perspective term out of an aggregate epigraph row.

    ``coeffs``/``rhs`` are ``grad^T x - eta <= rhs``; term ``k`` contributes
    ``2*q*z`` to ``coeffs[x_col]`` and ``q*z**2`` to ``rhs`` (the tangent's
    ``grad.z - f(z)``), so removing it is exact arithmetic and leaves the row a
    valid tangent of the residual objective. The term is re-added as its own
    perspective row against ``s_k``.

    Returns ``(coeffs, rhs)``, or ``None`` when a reference coordinate is not
    finite -- in which case the caller must drop the cut entirely rather than
    emit a row that removes some terms and not others.
    """
    for x_col, _y_col, _q in epigraph.terms:
        if not np.isfinite(float(x_star[x_col])):
            _PERSPECTIVE_DISAGG_REFUSED[0] += 1
            return None
    for k, (x_col, _y_col, q) in enumerate(epigraph.terms):
        z = float(x_star[x_col])
        coeffs[x_col] -= 2.0 * q * z
        rhs -= q * z * z
        if 2.0 * q * abs(z) > _PERSPECTIVE_MIN_ROW_COEFF:
            epigraph.add(k, z)
        else:
            # The row would say ``s_k >= 0``, which the column bound already
            # says -- exactly true at z == 0 and true to the master's own
            # arithmetic below the floor. The removal above still has to happen
            # either way; it subtracts ``2*q*z`` and ``q*z**2``, both negligible
            # here, and dropping an underestimator row can only weaken the
            # bound, never invalidate it.
            _PERSPECTIVE_DISAGG_NEAR_ZERO[0] += 1
    return coeffs, rhs


def _split_or_strengthen_objective_cut(
    evaluator, coeffs, rhs, x_star, n_vars, *, strengthen_aggregate: bool = True
):
    """Apply whichever perspective treatment this master is configured for.

    ``strengthen_aggregate=False`` opts a call site out of the #1064 in-place
    strengthening while keeping the #1066 split, which every site that writes
    into a disaggregated master must take. It is how the ECP site stays
    bound-neutral when disaggregation is off: #1064 graduated on a panel that
    measured the OA site only.

    Returns ``(coeffs, rhs)`` or ``None`` to mean "do not emit this cut".
    """
    epigraph = getattr(evaluator, "_perspective_epigraph", None)
    if epigraph is not None:
        return _disaggregate_objective_cut(coeffs, rhs, x_star, epigraph)
    terms = getattr(evaluator, "_perspective_oa_terms", None)
    if terms and strengthen_aggregate:
        coeffs, rhs, n_applied = _strengthen_objective_cut_perspective(
            coeffs, rhs, x_star, n_vars, terms
        )
        if n_applied:
            _PERSPECTIVE_OA_CUT_APPLIED[0] += n_applied
    return coeffs, rhs


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
    from discopt._relax.cutting_planes import (
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
        obj_coeffs_row = obj_cut.coeffs.copy()
        obj_rhs = float(obj_cut.rhs)
        # #1064/#1066: treat the perspective of every separable convex square
        # over a semicontinuous variable -- either strengthening this aggregate
        # row in place, or splitting the terms out into their own epigraph
        # columns. Globally valid either way (see ``_relax.perspective``), so
        # ``global_valid=True`` below is unchanged, and a no-op when the model
        # has no such structure.
        _split = _split_or_strengthen_objective_cut(
            evaluator, obj_coeffs_row, obj_rhs, x_star, n_vars
        )
        if _split is None:
            return
        obj_coeffs_row, obj_rhs = _split
        _append_master_cut(
            oa_A_rows,
            oa_b_rows,
            obj_coeffs_row,
            obj_rhs,
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
    from discopt._relax.cutting_planes import (
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

    _ecp_obj_coeffs = None
    _ecp_obj_rhs = None
    obj_support = None
    if not obj_is_linear and objective_is_convex:
        n_master = n_vars + 1
        obj_value = float(evaluator.evaluate_objective(x_master))
        obj_support = np.concatenate([np.asarray(x_master, dtype=np.float64), [obj_value]])
        obj_cut = generate_objective_oa_cut(evaluator, x_master, n_master, z_index=n_vars)
        # Same perspective treatment as the OA site: an ECP objective row lands
        # in the *same* master, so leaving a term in this row while the OA rows
        # split it out would double-count it against the epigraph columns.
        _split = _split_or_strengthen_objective_cut(
            evaluator,
            obj_cut.coeffs.copy(),
            float(obj_cut.rhs),
            x_master,
            n_vars,
            strengthen_aggregate=False,
        )
        _ecp_obj_coeffs, _ecp_obj_rhs = _split if _split is not None else (None, None)
    if _ecp_obj_coeffs is not None:
        _append_master_cut(
            oa_A_rows,
            oa_b_rows,
            _ecp_obj_coeffs,
            _ecp_obj_rhs,
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
    from discopt._relax.cutting_planes import (
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
        if getattr(evaluator, "_perspective_epigraph", None) is not None:
            # An ESH objective hyperplane is built, filtered and possibly
            # discarded downstream, so it cannot carry the all-or-nothing term
            # split a disaggregated master requires. The two are never combined
            # (``solve_lp_nlp_bb`` refuses to disaggregate under a SHOT
            # profile); refuse loudly rather than emit a row that double-counts
            # every perspective term against its own epigraph column.
            raise RuntimeError(
                "ESH objective hyperplanes cannot be generated into a master "
                "with a disaggregated perspective epigraph (#1066)"
            )
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


def _int_config_key(x_master, int_indices) -> tuple[int, ...]:
    """Return a canonical key for the integer part of a master solution."""
    return tuple(int(round(float(x_master[idx]))) for idx in int_indices)


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
    from discopt._relax.cutting_planes import separate_oa_cuts

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
    perspective_epigraph: Optional[_PerspectiveEpigraph] = None,
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
    # #1066: the disaggregated perspective epigraph is appended *after* every
    # existing block, so ``n_vars``, the eta column, the slack column and the
    # binary-expansion offsets all keep the values the rest of this module is
    # written against, and a stored row shorter than the master still maps by
    # prefix copy with zeros in the new columns.
    perspective_start = None
    n_perspective = 0
    if perspective_epigraph is not None:
        if not use_objective_epigraph:
            raise ValueError(
                "a disaggregated perspective epigraph needs the objective "
                "epigraph column: the split leaves eta covering the residual "
                "objective, and without eta the residual has nowhere to go"
            )
        n_perspective = len(perspective_epigraph.terms)
        perspective_start = n_master
        n_master += n_perspective

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

    if perspective_start is not None:
        assert perspective_epigraph is not None
        for row_index in range(len(perspective_epigraph.rows)):
            A_ub_rows.append(
                perspective_epigraph.row_for(
                    row_index, n_master=n_master, perspective_start=perspective_start
                )
            )
            b_ub_vals.append(0.0)

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
    if perspective_start is not None:
        # eta now underestimates only the residual objective; the split-out
        # terms are minimised through their own columns.
        c[perspective_start : perspective_start + n_perspective] = 1.0
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
    if perspective_start is not None:
        # q_k * x_k**2 >= 0, and the lower bound is what makes the z == 0
        # reference row redundant rather than missing.
        bounds_list.extend((0.0, 1e20) for _ in range(n_perspective))

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
        perspective_start=perspective_start,
        perspective_terms=(perspective_epigraph.terms if perspective_epigraph is not None else ()),
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
            "pip install pounce-solver  |  pip install gurobipy"
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


#: Consecutive OA iterations with **no** movement in either bound after which the
#: cut loop is abandoned as non-converging.
#:
#: OA's existing ``stalling_limit`` watches only the incumbent and defaults to
#: ``None`` (off), so a run whose master bound never moves had no exit at all and
#: spun until its time limit. Measured on ``alan`` (MINLPLib) with a 30 s budget::
#:
#:     OA iter    0: LB=0.000000 UB=3.000000 gap=100.0000% cuts=18
#:     OA iter    1: LB=0.000000 UB=3.000000 gap=100.0000% cuts=27
#:     ...  ~7500 iterations, +9 cuts each, neither bound ever moving
#:
#: The whole budget bought nothing: the caller's fallback then reproduced the
#: default path's answer exactly (same objective, same bound, 21 nodes) having
#: paid 30 s for the privilege.
#:
#: Aborting cannot weaken the reported bound -- by construction the bound has not
#: moved for ``_OA_NO_PROGRESS_ITERATIONS`` iterations, so the value returned is
#: the one the loop would still be holding. It only stops paying for cuts that
#: demonstrably change nothing. Deliberately generous: a genuine OA run plateaus
#: for a few iterations before a cut binds (``fac2`` sits at LB=2.6e8 for two
#: iterations, then jumps to the optimum on the third).
_OA_NO_PROGRESS_ITERATIONS = 50


def _bounds_moved(
    before: tuple[Optional[float], Optional[float]],
    after: tuple[Optional[float], Optional[float]],
) -> bool:
    """Did either OA bound move by more than floating-point noise?

    Scale-relative so a 1-ulp wobble at ``|obj| ~ 1e8`` does not read as
    progress and reset the no-progress counter forever, which would make
    :data:`_OA_NO_PROGRESS_ITERATIONS` unreachable on exactly the large-objective
    models where a stalled loop is most expensive.
    """
    for old_value, new_value in zip(before, after):
        if old_value is None or new_value is None:
            return True
        if abs(float(new_value) - float(old_value)) > 1e-9 * max(1.0, abs(float(old_value))):
            return True
    return False


def _compute_gap(lb: float, ub: float) -> float:
    """OA's optimality gap — see :mod:`discopt.solvers._gap` for the semantics.

    ``denom_floor=1.0``: below unit objective scale OA reports the absolute gap
    rather than dividing by a vanishing denominator.
    """
    return optimality_gap(lb, ub, denom_floor=1.0)


def _lp_nlp_bb_exit_status(
    *,
    converged_early: bool,
    callback_terminated: bool,
    master_status: "SolveStatus",
    has_incumbent: bool,
    master_bound_valid: bool,
    gap: Optional[float],
    gap_tolerance: float,
    hook_stopped_before_wall: bool,
) -> tuple[str, Optional[str]]:
    """The status and termination reason :func:`solve_lp_nlp_bb` exits with.

    Split out of the driver so the decision can be tested at every combination of
    its inputs rather than only at whichever ones a 60-second solve happens to
    produce.

    The last clause is the one with teeth. A certificate the driver already holds
    does not stop being a certificate because the *clock* -- rather than the gap
    test -- is what ended the search. ``master_bound_valid`` says the master is a
    relaxation of the MINLP, and a MIP tree's dual bound bounds its own optimum
    whether the tree finished or was cut off mid-search, so ``gap <=
    gap_tolerance`` on that bound is the same certificate a converged run reports.
    Before this, a run that ended on the wall was reported ``feasible`` even with
    the gap closed: ``squfl015-060`` at default settings finished with dual bound
    366.6218167383147 against incumbent 366.62181673996474 -- a relative gap of
    4.5e-14 against a 1e-4 tolerance -- and reported ``feasible``.

    ``termination_reason`` is deliberately NOT rewritten to ``gap_tolerance`` when
    that clause fires: the search really did stop on the clock, and the trace
    should keep saying so. Status answers "is the answer proven"; the reason
    answers "why did the loop end". They are different questions.
    """
    from discopt.solvers import SolveStatus

    status = "feasible"
    termination_reason: Optional[str] = None
    if converged_early and has_incumbent:
        # Stopped on the certificate, not on the clock or the caller's option.
        termination_reason = "gap_tolerance"
        status = "optimal" if gap is not None and gap <= gap_tolerance else "feasible"
    elif callback_terminated:
        # The clock and the hook both come back through the same callback, so name
        # whichever one it was rather than reporting a time limit that may not have
        # been reached.
        termination_reason = "termination_hook" if hook_stopped_before_wall else "time_limit"
        status = "time_limit" if not has_incumbent else "feasible"
    elif master_status == SolveStatus.INFEASIBLE:
        status = "infeasible"
    elif master_status == SolveStatus.TIME_LIMIT:
        status = "time_limit" if not has_incumbent else "feasible"
    elif master_status == SolveStatus.ITERATION_LIMIT:
        status = "iteration_limit" if not has_incumbent else "feasible"
    elif master_status == SolveStatus.OPTIMAL and has_incumbent:
        status = "optimal" if gap is not None and gap <= gap_tolerance else "feasible"
    elif not has_incumbent:
        status = "no_feasible_point"
    if (
        status == "feasible"
        and has_incumbent
        and master_bound_valid
        and gap is not None
        and gap <= gap_tolerance
    ):
        # ``gap`` comes from :func:`_compute_gap`, which reports 1.0 -- never a
        # small number -- when the bound ordering is materially inverted, so a
        # broken certificate cannot reach this branch (see ``solvers._gap``).
        status = "optimal"
    if termination_reason is None:
        termination_reason = status
    return status, termination_reason


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
            max_wall_time=_remaining_wall(t_start, time_limit),
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
            x=None,  # no incumbent -- see SolveResult.x's contract (#1105)
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
        x=None,  # no incumbent -- see SolveResult.x's contract (#1105)
        wall_time=wall_time,
        mip_count=fp.mip_count,
        gap_certified=False,
    )


def _continuous_model_is_certified_convex(decomp: "_DecomposedProblem") -> bool:
    """Is the *continuous* model provably convex — objective AND every constraint?

    The gate on certifying a single local NLP solve as the global optimum. Without
    it, ``solve_oa``/``solve_lp_nlp_bb`` returned ``status="optimal"`` with
    ``bound = objective`` and ``gap = 0.0`` for ANY integer-free model, because an
    integer-free OA "loop" is one NLP solve and the code read that as a proof.

    Measured on MINLPLib ``trig`` (one continuous variable on ``[-2, 5]``, one
    nonconvex row): ``mip_nlp_method="lp_nlp_bb"`` returned ``optimal`` at
    ``-2.479027828`` with ``bound = -2.479027828``, while the true minimum over the
    declared box is ``-3.762500358`` (brute force at ``x = 2.667``, and MINLPLib's
    own value). A local NLP found a local minimum and the caller was handed it as a
    certificate. Found by #1141's corpus panel; it predates that work and is
    identical with the panel's flag on or off.

    Convex objective + convex constraints is exactly the condition under which a
    local minimum is global, which is what the certificate claims.
    """
    if not (decomp.obj_is_linear or decomp.oa_objective_is_convex):
        return False
    mask = decomp.oa_constraint_mask
    if decomp.n_cons == 0:
        return True
    if not mask:
        return False
    return all(mask)


def _oa_node_cuts_enabled() -> bool:
    """``DISCOPT_OA_NODE_CUTS``: separate ECP cuts at *fractional* master nodes.

    Default-**ON** since the #1141 graduation panel. Adding a supporting hyperplane
    at a fractional node is sound (a convex constraint's tangent underestimates it
    everywhere, so the row is globally valid) but it CHANGES the master's dual
    bound — CLAUDE.md §5 regime 2 — so it shipped default-OFF until a corpus-wide
    differential panel cleared both bars.

    That panel, over the 119 vendored MINLPLib instances (``lp_nlp_bb`` on the
    in-house simplex master, 30 s per arm, interleaved, incumbents
    feasibility-verified): 37/119 rows exercised it, 189 soundness checks,
    **0 violations**, certificates **42 -> 44** with none lost, total wall
    601.9 s -> 536.8 s (**−10.8 %**), **10 dual bounds tighter and 2 looser**
    (``clay0303hfsg`` and ``fac2``, neither certified by either arm). On the class
    the issue is about it is 6–16x. Recorded in
    ``docs/dev/performance-plan.md`` §24.

    ``DISCOPT_OA_NODE_CUTS=0`` is the opt-out and restores the pre-#1141 master
    exactly. Ignored on the SHOT profile, which is *defined* by fractional-point
    hyperplane generation and therefore always separates at nodes.
    """
    return os.environ.get("DISCOPT_OA_NODE_CUTS", "1") not in ("0", "", "false", "False")


def _oa_node_cut_rounds() -> int:
    """Separation rounds one master node may run (``DISCOPT_OA_NODE_CUT_ROUNDS``).

    Each round is one gradient evaluation plus one warm LP re-solve of that node.
    The rows are global, so what one node does not separate the next one will —
    a small budget is the design, not a compromise.
    """
    return max(1, int(os.environ.get("DISCOPT_OA_NODE_CUT_ROUNDS", "2")))


def _oa_node_cut_cap() -> int:
    """Cap on fractional-node rows per solve (``DISCOPT_OA_NODE_CUT_CAP``).

    A fractional cut only tightens, so it is optional and budgeted: an unbounded
    stream of gradient cuts densifies every node LP and trades the node win back
    for wall time.
    """
    return max(1, int(os.environ.get("DISCOPT_OA_NODE_CUT_CAP", "500")))


def _resolve_lp_nlp_bb_backend(milp_solver: str, *, shot_profile: bool) -> str:
    """Pick the single-tree MILP backend for LP/NLP branch-and-bound.

    Returns ``"gurobi"``, ``"simplex"`` or ``"highs"``. Until #1060 this raised for
    anything but Gurobi, because the single tree needs a *persistent
    lazy-constraint callback* and only Gurobi's had one. The in-house Rust MILP
    driver now has one too (``solve_milp_lazy_csc_py``), so ``"auto"`` and
    ``"simplex"`` resolve to it and the method no longer requires a commercial
    license.

    ``"highs"`` resolves to the HiGHS separate-and-restart master
    (:func:`discopt.solvers.milp_highs.solve_milp_with_lazy_cuts`). HiGHS cannot
    inject a row from inside its tree -- 1.12 declares
    ``kCallbackMipDefineLazyConstraints`` but its callback input struct has no
    field to hand one back -- so it separates at ``kCallbackMipImprovingSolution``
    (the QG trigger: every integer-feasible incumbent) and rebuilds the tree when,
    and only when, a cut is actually needed. That is not a true single tree and
    the docstring above says so, but it is what makes the free path *finish*: the
    in-house driver's root loop closes 0% of the ``rsyn0840m`` master's gap where
    HiGHS closes 86.1%, and the master engine, not the tree topology, is the
    measured gap.

    ``"auto"`` deliberately does **not** pick HiGHS. Routing it there would make
    the default depend on an optional package and would move every existing
    caller's node counts; the opt-in keeps #356 and the current defaults intact.

    ``"pounce"`` still refuses: the POUNCE matrix-MILP B&B exposes no separator
    hook, and silently substituting a different backend would hide that the
    caller's choice was ignored.

    The SHOT profile no longer requires Gurobi (#1141). Its ESH/hyperplane
    strategy adds user cuts at *fractional* node relaxations, and until #1141 the
    Rust hook fired only at integer-feasible points, so there was nothing to map
    it onto; the driver now has a fractional-node hook
    (``solve_milp_lazy_csc_py(node_callback=...)``) and ``"simplex"``/``"auto"``
    serve SHOT too. ``"highs"`` is still refused for SHOT: that master separates
    only at integer-feasible incumbents, and accepting the request while dropping
    the fractional cuts would report a SHOT run that never ran SHOT's cut
    generation.
    """
    backend = milp_solver.strip().lower() if isinstance(milp_solver, str) else ""
    if backend == "gurobi":
        return "gurobi"
    if backend in {"auto", "simplex"}:
        return "simplex"
    if shot_profile:
        # The SHOT profile separates hyperplanes at fractional node relaxations.
        # Only Gurobi and the in-house simplex driver (#1141) expose that hook;
        # the HiGHS master does not, and accepting the request while dropping
        # those cuts would report a SHOT run that never ran SHOT's cut generation.
        raise RuntimeError(
            "mip_nlp_method='lp_nlp_bb' with mip_nlp_profile='shot' requires a "
            "backend with a fractional-node cut hook: 'gurobi' or 'simplex' (also "
            "reachable as 'auto'). The HiGHS master separates only at "
            f"integer-feasible incumbents. Got milp_solver={milp_solver!r}."
        )
    if backend == "highs":
        return "highs"
    raise RuntimeError(
        "mip_nlp_method='lp_nlp_bb' requires a MILP backend with a lazy-constraint "
        "callback: 'gurobi', 'simplex' (also reachable as 'auto') or 'highs'. "
        f"Backend {milp_solver!r} exposes no separator hook."
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
    milp_solver: str = "auto",
    integer_to_binary: bool = False,
    mip_nlp_profile: str = "default",
    mip_nlp_shot_config: Optional[MIPNLPShotConfig] = None,
    termination_hook: Any = None,
    **kwargs,
) -> SolveResult:
    """Solve a convex MINLP with the LP/NLP branch-and-bound variant.

    This is a single-tree OA method: the MILP master is solved once, and each
    integer incumbent triggers a fixed-integer NLP solve inside a lazy
    constraint callback. Lazy rows are generated through the same OA and
    feasibility-cut helpers used by the multi-tree OA method. ``max_iterations``
    only caps the optional feasibility-pump initializer on this path; the main
    single-tree search is delegated to the MILP backend and is controlled by
    ``time_limit`` and ``gap_tolerance``.

    ``milp_solver`` selects that backend: ``"simplex"`` (the in-house Rust MILP
    driver, also reached by the default ``"auto"``) or ``"gurobi"``. See
    :func:`_resolve_lp_nlp_bb_backend` for what each one supports.

    ``termination_hook`` is the same contract :func:`solve_oa` offers -- a callable
    handed a context mapping, returning true to stop -- so a caller that budgets a
    solve by progress can budget this method too. It is honoured on the backends
    that have a check-in point: ``"highs"`` consults it at every master restart and
    ``"gurobi"`` on its wall-clock poll. The ``"simplex"`` driver has neither, and
    passing a hook it cannot call would make any budget built on it a fiction, so
    that combination is refused rather than silently ignored (CLAUDE.md §3).

    The context carries the keys :func:`solve_oa`'s hook sees for a bound-driven
    decision -- ``event``, ``elapsed``, ``is_minimization``, ``current_dual_bound``,
    ``current_primal_bound``, ``relative_gap``, ``absolute_gap``,
    ``incumbent_objective`` -- plus ``restarts`` and ``lazy_cuts``. The per-iteration
    OA keys have no meaning in a single tree and are not invented: this method has
    no ``iteration``.
    """
    if kwargs:
        raise ValueError(
            "Unsupported LP/NLP BB option(s): "
            + ", ".join(sorted(kwargs))
            + ". Supported options are: add_no_good_cuts, add_slack, equality_relaxation, "
            "feasibility_cuts, feasibility_norm, heuristic_nonconvex, init_strategy, "
            "integer_to_binary, "
            "max_slack, milp_solver, mip_nlp_profile, mip_nlp_shot_config, "
            "oa_penalty_factor, termination_hook."
        )
    t_start = time.perf_counter()
    shot_config = mip_nlp_shot_config if mip_nlp_profile == "shot" else None
    shot_profile = shot_config is not None
    shot_cut_strategy = shot_config.cut_strategy if shot_config is not None else "oa"
    # Resolve (and refuse) the single-tree backend before any model work, so an
    # unsupported request costs a message rather than a decomposition.
    lazy_backend = _resolve_lp_nlp_bb_backend(milp_solver, shot_profile=shot_profile)
    hook = _normalize_optional_hook("termination_hook", termination_hook)
    if hook is not None and lazy_backend == "simplex":
        raise NotImplementedError(
            "termination_hook is not available on the in-house simplex master: the "
            "driver enforces time_limit itself and offers no check-in point, so the "
            "hook could never be called and any budget built on it would be a "
            "fiction. Use milp_solver='highs' (checks in at every master restart) "
            "or milp_solver='gurobi'."
        )
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

    # Restoration outcomes are per solve, not per process (#1141).
    _RESTORATION_OUTCOMES.clear()
    # Read once per solve, not per callback: a flag that changed mid-search would
    # make the run's own record unreadable.
    _oa_infeasible_nogood = _infeasible_nogood_enabled()
    #: Assignments excluded because the fixed NLP PROVED them infeasible. The
    #: anti-vacuity counter for the flag (§6): zero means it never fired, which is
    #: not the same as "there was nothing to exclude".
    proven_infeasible_count = [0]
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
            max_wall_time=_remaining_wall(t_start, time_limit),
        )
        wall_time = time.perf_counter() - t_start
        certified = _continuous_model_is_certified_convex(decomp) and not heuristic_nonconvex
        if x_sol is not None:
            # An integer-free OA "loop" is a single local NLP solve. That is the
            # global optimum only on a convex model; on a nonconvex one it is a
            # local minimum, and reporting it with ``bound = objective, gap = 0``
            # is a false certificate (see ``_continuous_model_is_certified_convex``:
            # ``trig`` returned -2.479 as optimal against a true -3.7625).
            return SolveResult(
                status="optimal" if certified else "feasible",
                objective=obj_sign * obj,
                bound=obj_sign * obj if certified else None,
                gap=0.0 if certified else None,
                gap_certified=certified,
                x=_build_x_dict(x_sol, model),
                wall_time=wall_time,
                mip_count=0,
            )
        # A local NLP that found no point has not PROVED the model infeasible
        # either -- the same asymmetry, in the other direction.
        return SolveResult(
            status="infeasible" if certified else "no_feasible_point",
            objective=None,
            bound=None,
            gap=None,
            x=None,  # no incumbent -- see SolveResult.x's contract (#1105)
            wall_time=wall_time,
            mip_count=0,
        )

    oa_A_rows: list[np.ndarray] = []
    oa_b_rows: list[float] = []
    oa_cut_relaxable: list[bool] = []
    incumbent: Optional[np.ndarray] = None
    incumbent_obj: Optional[float] = None
    nlp_subproblem_count = 0

    # #1066: give every separable perspective term its own epigraph column, so
    # the master's LP relaxation can combine each term's best reference at a
    # fractional point instead of being held to one common reference per row.
    # Only on this driver's own master: it builds exactly one, which is what
    # makes the all-or-nothing term split checkable. Not under a SHOT profile,
    # whose ESH hyperplanes cannot carry the split (they are filtered after
    # generation), and not without the epigraph column the residual needs.
    perspective_epigraph: Optional[_PerspectiveEpigraph] = None
    if (
        _perspective_disagg_enabled()
        and not shot_profile
        and not decomp.obj_is_linear
        and decomp.oa_objective_is_convex
    ):
        perspective_epigraph = _perspective_epigraph_for(
            getattr(evaluator, "_perspective_oa_terms", None) or [], n_vars
        )
        if perspective_epigraph is not None:
            evaluator._perspective_epigraph = perspective_epigraph  # type: ignore[attr-defined]

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
            max_wall_time=_remaining_wall(t_start, time_limit),
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
            max_wall_time=_remaining_wall(t_start, time_limit),
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
        perspective_epigraph=perspective_epigraph,
    )
    #: Perspective epigraph rows already in the master. The build below takes
    #: whatever the pool holds at that moment; everything the search adds after
    #: it has to reach the tree as a lazy row like any other cut. Rows that are
    #: not violated yet stay in the pool and are offered again later rather than
    #: being dropped -- a term's reference earns its place at a *fractional*
    #: point, which is the whole reason the epigraph is disaggregated.
    perspective_rows_emitted = set(
        range(len(perspective_epigraph.rows)) if perspective_epigraph is not None else ()
    )

    def collect_new_perspective_rows(master_x: np.ndarray) -> list[tuple[np.ndarray, float]]:
        if perspective_epigraph is None or master.perspective_start is None:
            return []
        rows: list[tuple[np.ndarray, float]] = []
        scores = perspective_epigraph.violations(
            master_x, perspective_start=master.perspective_start
        )
        # Freshly split rows go in unconditionally (see ``_pending``); older
        # pooled rows only when the master point actually violates them.
        candidates = perspective_epigraph.drain_pending()
        candidates.extend(np.flatnonzero(scores > 1e-6).tolist())
        for row_index in candidates:
            if row_index in perspective_rows_emitted:
                continue
            rows.append(
                (
                    perspective_epigraph.row_for(
                        row_index,
                        n_master=len(master.c),
                        perspective_start=master.perspective_start,
                    ),
                    0.0,
                )
            )
            perspective_rows_emitted.add(row_index)
        return rows

    def collect_new_lazy_cuts(start: int, master_x: np.ndarray) -> list[tuple[np.ndarray, float]]:
        rows: list[tuple[np.ndarray, float]] = list(collect_new_perspective_rows(master_x))
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
        integer_cut_added = False
        fixed_attempt = _solve_fixed_nlp_subproblem_attempt(
            evaluator,
            decomp.lb,
            decomp.ub,
            decomp.int_indices,
            x_master,
            nlp_solver,
            initial_point=x_master,
            max_wall_time=_remaining_wall(t_start, time_limit),
        )
        x_nlp, obj_nlp = fixed_attempt.x, fixed_attempt.objective
        # #1141: the trace used to record every non-success as the single word
        # "failed", which made a genuinely infeasible integer assignment and a
        # subsolver that fell over indistinguishable in the record — and those are
        # different problems with different fixes. Name the outcome instead.
        fixed_nlp_status = _fixed_nlp_status_label(fixed_attempt)
        if x_nlp is not None:
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
                    max_wall_time=_remaining_wall(t_start, time_limit),
                    constraint_convex_mask=decomp.oa_constraint_mask,
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
            # #1141 item 4: 7 of 172 integer assignments were re-proposed on
            # `portfol_classical050_1`, one of them six times, because an OA cut
            # excludes the *point* it was taken at, not the *assignment* -- with a
            # linear objective and no epigraph there is nothing forcing the master
            # away from the same integer point at a different continuous one. A
            # no-good cut is what excludes the assignment, and it is sound exactly
            # when the assignment is PROVEN infeasible (see
            # `_assignment_proven_infeasible`; "the NLP returned nothing" is not a
            # proof and must never be treated as one).
            proven_infeasible = _oa_infeasible_nogood and _assignment_proven_infeasible(
                fixed_attempt, evaluator, decomp.oa_constraint_mask
            )
            if proven_infeasible:
                proven_infeasible_count[0] += 1
            if (add_no_good_cuts or proven_infeasible) and (
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

    from discopt.solvers import MILPResult

    solve_milp_with_lazy_cuts: Callable[..., MILPResult]
    if lazy_backend == "gurobi":
        from discopt.solvers.gurobi import (
            solve_milp_with_lazy_cuts as solve_milp_with_lazy_cuts,
        )
    elif lazy_backend == "highs":
        from discopt.solvers.milp_highs import (
            solve_milp_with_lazy_cuts as solve_milp_with_lazy_cuts,
        )
    else:
        from discopt.solvers.milp_simplex import (
            solve_milp_with_lazy_cuts as solve_milp_with_lazy_cuts,
        )

    master_mip_start = None
    if shot_profile and incumbent is not None:
        master_mip_start = _extend_master_mip_start(
            master,
            n_vars=n_vars,
            mip_start=incumbent,
            mip_start_objective=_master_objective_from_evaluator(incumbent_obj),
        )

    hook_calls = [0]
    #: The internal-sense dual bound that closed the gap, once one does.
    converged_at: list[Optional[float]] = [None]
    #: How many check-ins carried a usable dual bound. Zero means the early exit
    #: never had anything to judge, which is NOT the same as "it judged and said
    #: keep going" (CLAUDE.md §6); it is exported in ``callback_stats``.
    bound_observations = [0]

    def _master_bound_internal(raw) -> Optional[float]:
        """The master's dual bound in the internal minimization sense, or None."""
        if not master_bound_valid or raw is None or not np.isfinite(float(raw)):
            return None
        value = float(raw)
        if decomp.obj_is_linear and decomp.obj_coeffs is not None:
            value += float(decomp.obj_coeffs[1])
        return value

    def callback_terminate(snapshot: dict[str, object]) -> bool:
        if (time.perf_counter() - t_start) >= float(time_limit):
            return True
        lb = _master_bound_internal(snapshot.get("dual_bound"))
        ub = incumbent_obj
        if lb is not None and ub is not None:
            bound_observations[0] += 1
            if _compute_gap(lb, ub) <= gap_tolerance:
                # The master's running dual bound has met the NLP incumbent, which
                # is the same certificate this driver tests for after the master
                # returns -- just read at the moment it becomes true rather than
                # whenever the separator happens to run dry. The bound is valid:
                # the master is a relaxation of the MINLP (its cuts are supporting
                # hyperplanes of convex functions, and ``master_bound_valid`` is
                # what says so), and a MIP tree's dual bound is a valid bound on
                # its own optimum at every instant of the search, interrupted or
                # not. Without this the loop runs past its own certificate:
                # ``rsyn0820m02m`` had bound 1092.1600 against incumbent 1092.0911
                # -- gap 6.3e-5, inside the 1e-4 default -- at 5.1 s of a 60 s
                # limit, then rebuilt its tree five more times and was reported
                # ``feasible`` at the wall (measured 2026-08-29).
                converged_at[0] = lb if converged_at[0] is None else max(converged_at[0], lb)
                return True
        if hook is None:
            return False
        context: dict[str, object] = {
            "event": "termination",
            "elapsed": float(time.perf_counter() - t_start),
            "is_minimization": bool(obj_sign > 0),
            "current_dual_bound": None if lb is None else obj_sign * lb,
            "current_primal_bound": None if ub is None else obj_sign * ub,
            "relative_gap": _compute_gap(lb, ub) if lb is not None and ub is not None else None,
            "absolute_gap": (None if lb is None or ub is None else abs(ub - lb)),
            "incumbent_objective": None if ub is None else obj_sign * ub,
            "n_vars": int(n_vars),
            "n_constraints": int(n_cons),
            "restarts": int(cast(int, snapshot.get("restarts", 0) or 0)),
            "lazy_cuts": int(cast(int, snapshot.get("lazy_cuts", 0) or 0)),
        }
        hook_calls[0] += 1
        try:
            raw = hook(context)
        except Exception as exc:  # never swallowed (CLAUDE.md §7)
            raise RuntimeError(f"termination_hook failed during LP/NLP BB solve: {exc}") from exc
        return _validate_external_termination(raw)

    lazy_kwargs: dict[str, object] = dict(
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
        mip_start=master_mip_start,
    )
    if lazy_backend == "gurobi":
        lazy_kwargs["node_callback"] = node_callback if shot_profile else None
        lazy_kwargs["terminate_callback"] = callback_terminate
    elif lazy_backend == "simplex" and (shot_profile or _oa_node_cuts_enabled()):
        # #1141: the in-house driver now has a fractional-node hook, so the ECP
        # cuts that used to require Gurobi's MIPNODE run here too. This is the
        # capability the issue measured as missing: without it every node either
        # pays a full NLP (38.9 ms/node against SCIP's 3.0) or gets a relaxation
        # that ignores the nonlinear constraint. The SHOT profile *needs* it (its
        # hyperplane strategy is defined at fractional points), so it is wired
        # unconditionally there; for the plain convex path it is a bound-CHANGING
        # knob and ships behind ``DISCOPT_OA_NODE_CUTS`` (CLAUDE.md §5 regime 2).
        lazy_kwargs["node_callback"] = node_callback
        lazy_kwargs["node_hook_rounds"] = _oa_node_cut_rounds()
        lazy_kwargs["node_hook_cut_cap"] = _oa_node_cut_cap()
    elif lazy_backend == "highs":
        # The HiGHS master checks in at every restart and, between them, on an
        # interval. That is what lets a caller budget this method by progress
        # (#1066) -- and, hook or no hook, what lets the driver notice its own
        # certificate the moment the bound meets the incumbent instead of at
        # whatever later instant the separator runs dry.
        lazy_kwargs["terminate_callback"] = callback_terminate
    master_result = solve_milp_with_lazy_cuts(**lazy_kwargs)  # type: ignore[arg-type]
    wall_time = time.perf_counter() - t_start

    bound = None
    # The master bound as it stood before the inversion guard below, kept so the
    # trace still carries the number a diagnosis needs after ``bound`` is
    # suppressed. ``None`` means the guard did not fire.
    inverted_master_bound: Optional[float] = None
    if master_bound_valid and master_result.bound is not None:
        bound = float(master_result.bound)
        if decomp.obj_is_linear and decomp.obj_coeffs is not None:
            bound += float(decomp.obj_coeffs[1])
    if converged_at[0] is not None:
        # Report the bound the stop was actually taken on. Both readings are valid
        # bounds in the internal sense, so the tighter one is the honest one -- and
        # if the post-hoc read came back weaker, reporting it would contradict the
        # very test that ended the search and turn a certificate into "feasible".
        bound = converged_at[0] if bound is None else max(bound, converged_at[0])
    # The master's dual bound crossed the incumbent it is reported against.
    # ``solve_oa`` has refused to publish that since ``fac2`` (see
    # ``_certified_bound_inverted``); this path -- the single-tree LP/NLP-BB
    # driver, a different function -- never got the same guard. Measured on
    # ``squfl020-150`` (MINLPLib ``=best=`` 557.84865) at default settings: the
    # HiGHS lazy master returned ``bound=557.9460019817818`` against this
    # driver's own incumbent ``557.848649973387``, i.e. a *lower* bound 0.097
    # ABOVE the true optimum, and the run reported it. ``_compute_gap`` already
    # declines to call that converged (it returns 1.0, documented as "nothing
    # proved"), but suppressing only the gap still hands the caller ``bound >
    # objective``, and #1059's route merge republishes exactly that number.
    #
    # One of the two is wrong and this code cannot tell which, so neither is
    # reported as proved. The incumbent survives -- it is an independently
    # feasibility-checked point -- and the offending number is kept in the
    # trace as ``inverted_master_bound``, where a diagnostic belongs and nothing
    # reads it as a dual certificate (CLAUDE.md §1).
    if (
        bound is not None
        and incumbent_obj is not None
        and np.isfinite(bound)
        and np.isfinite(incumbent_obj)
        and float(bound) - float(incumbent_obj)
        > bound_inversion_tolerance(float(bound), float(incumbent_obj))
    ):
        logger.warning(
            "lp_nlp_bb: master dual bound %.12g is above the incumbent %.12g by "
            "more than rounding -- one of the two is wrong, so neither the bound "
            "nor the gap is reported. This is a bound-validity defect worth "
            "investigating, not a tolerance to widen.",
            float(bound),
            float(incumbent_obj),
        )
        inverted_master_bound = float(bound)
        bound = None
        master_bound_valid = False
    gap = (
        _compute_gap(bound, incumbent_obj)
        if bound is not None and incumbent_obj is not None
        else None
    )
    callback_stats = dict(master_result.callback_stats or {})
    # How many times the caller's hook actually ran. Zero with a hook installed
    # means it never got a check-in -- a master that never separated -- and NOT
    # that it kept saying continue (CLAUDE.md §6).
    callback_stats["termination_hook_calls"] = int(hook_calls[0])
    # Distinguish "the early exit never had a bound to judge" from "it judged and
    # kept going" -- only the HiGHS backend reports a dual bound at check-in, so
    # on any other master this count is 0 and the exit is inert by construction.
    callback_stats["dual_bound_observations"] = int(bound_observations[0])
    callback_stats["converged_early"] = bool(converged_at[0] is not None)
    callback_terminated = bool(callback_stats.get("terminated"))
    status, termination_reason = _lp_nlp_bb_exit_status(
        converged_early=converged_at[0] is not None,
        callback_terminated=callback_terminated,
        master_status=master_result.status,
        has_incumbent=incumbent is not None,
        master_bound_valid=bool(master_bound_valid),
        gap=gap,
        gap_tolerance=gap_tolerance,
        hook_stopped_before_wall=(
            hook is not None and (time.perf_counter() - t_start) < float(time_limit)
        ),
    )

    trace_bound_validity = (
        "global"
        if master_bound_valid and bound is not None
        else ("heuristic" if bound is not None else "unavailable")
    )
    # ``gap_certified`` means the gap is CLOSED and rests on a valid bound -- not
    # merely that the printed gap is arithmetically well-formed. The looser
    # reading (``master_bound_valid`` alone) reported ``gap_certified=True``
    # alongside a 424% open gap on ``syn40m``, which ``result_io.summary_text``
    # renders with no "(uncertified)" marker, and which the NLP-BB path spells
    # the opposite way (``solver.py``: any ``feasible`` exit clears it). One
    # field with two meanings is what made a cross-route comparison meaningless.
    # Bound validity keeps its own signal in ``master_bound_valid`` /
    # ``bound_validity``; this narrows ``gap_certified`` only, and only ever
    # from True to False.
    gap_is_certified = bool(master_bound_valid and gap is not None and gap <= gap_tolerance)
    single_tree_trace: dict[str, object] = {
        "schema_version": 1,
        "solver": "mip-nlp",
        "method": "lp_nlp_bb",
        "milp_backend": lazy_backend,
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
            # #1141: how the feasibility restorations actually ended, by
            # formulation and subsolver code. Restoration falls back to the
            # clipped master point on failure, so without this a run where it
            # never converged is indistinguishable from one where it always did.
            "restoration_outcomes": {
                f"{form}/{status}": int(n) for (form, status), n in _RESTORATION_OUTCOMES.items()
            },
            # Assignments the fixed NLP PROVED infeasible and that were therefore
            # excluded by a no-good cut. Zero means the mechanism never fired --
            # not that there was nothing to exclude (CLAUDE.md §6).
            "proven_infeasible_assignments": int(proven_infeasible_count[0]),
        },
        "termination_reason": termination_reason,
        "master_bound_valid": bool(master_bound_valid),
        "gap_certified": gap_is_certified,
        "bound_validity": trace_bound_validity,
        "final_lb": _trace_value(bound),
        "final_ub": _trace_value(incumbent_obj),
        "final_gap": _trace_value(gap),
        # Only set when the master handed back a bound past the incumbent and
        # the guard above suppressed it -- the number to investigate, kept out
        # of ``final_lb`` so nothing reads it as a dual certificate.
        "inverted_master_bound": _trace_value(inverted_master_bound),
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
            gap_certified=gap_is_certified,
            mip_nlp_trace=single_tree_trace,
        )

    return SolveResult(
        status=status,
        objective=None,
        bound=(obj_sign * bound if bound is not None else None),
        gap=None,
        x=None,  # no incumbent -- see SolveResult.x's contract (#1105)
        wall_time=wall_time,
        node_count=master_result.node_count,
        mip_count=1,
        subnlp_calls=nlp_subproblem_count,
        gap_certified=gap_is_certified,
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

    from discopt._relax.convexity import classify_oa_cut_convexity

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
    master_checkin_deadline: Optional[float] = None,
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
    add_regularization : str or None
        Optional level-set regularized OA master before fixed-integer NLP solves.
        One of ``None``, ``"level_L1"``, ``"level_L2"``, ``"level_L_infinity"``,
        ``"grad_lag"``, ``"hess_lag"``, ``"hess_only_lag"``, or ``"sqp_lag"``.
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
        MILP backend for OA master problems: ``"auto"``, ``"pounce"``,
        ``"simplex"``, or ``"gurobi"``.
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
    external_primal_candidate_hook : callable, optional
        Opt-in event hook for the multi-tree OA loop (see notes below).
    external_hyperplane_hook : callable, optional
        Opt-in event hook for the multi-tree OA loop (see notes below).
    external_dual_bound_hook : callable, optional
        Opt-in event hook for the multi-tree OA loop (see notes below).
    termination_hook : callable, optional
        Opt-in event hook for the multi-tree OA loop. All four hooks receive a
        read-only context dictionary with iteration, elapsed time, current
        bound/incumbent data, and candidate points where relevant. Returned
        payloads are validated before they can add external fixed-NLP candidates,
        master cuts, dual-bound updates, or request user termination.
    master_checkin_deadline : float, optional
        Soft deadline, in seconds of elapsed OA time, by which the loop must
        return to the top of an iteration so ``termination_hook`` can run. Only
        the master MILP that would cross it is shortened, and only until the
        deadline passes; after that the ordinary budget applies. Pointless
        without ``termination_hook``, and validated as such.

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
    if master_checkin_deadline is not None:
        master_checkin_deadline = float(master_checkin_deadline)
        if not math.isfinite(master_checkin_deadline) or master_checkin_deadline <= 0.0:
            raise ValueError(
                "master_checkin_deadline must be a finite positive number of seconds, "
                f"got {master_checkin_deadline!r}"
            )
        if termination_hook is None:
            raise ValueError(
                "master_checkin_deadline only has an effect alongside termination_hook; "
                "passing it without one shortens a master solve for no reader"
            )
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
    # Local NLP failures do not prove fixed-integer infeasibility. Track those
    # assignments so final certification is downgraded instead of adding an
    # unsound no-good cut.
    unresolved_int_configs: set[tuple[int, ...]] = set()
    incumbent_progress: list[float] = []
    # ``_OA_NO_PROGRESS_ITERATIONS`` bookkeeping: the (LB, UB) pair as of the end
    # of the previous iteration, and how many consecutive iterations have left it
    # untouched.
    _last_progress_bounds: Optional[tuple[Optional[float], Optional[float]]] = None
    no_progress_iterations = 0
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

    def _certified_bound_inverted() -> bool:
        """Has the certified lower bound risen *above* the incumbent?

        A dual bound past the incumbent is a broken certificate: one of the two
        is wrong and OA cannot tell which, so neither may be reported as proved.
        :func:`optimality_gap` already refuses to call this converged -- it
        returns ``1.0``, documented as "nothing proved ... let the caller's
        certification logic decline" -- but the caller did not decline: the
        finalizer stamped ``gap_certified=bool(reported_gap is not None)``, and
        ``1.0`` is not ``None``, so an inverted bound shipped as a certificate.

        Measured on ``fac2`` (MINLPLib, optimum 331837498.2) at a 30 s limit::

            OA iter 23: LB=331845337.439688 UB=331845161.424879 gap=100.0000%

        The run returned ``bound=331845337.44`` with ``gap_certified=True`` --
        a lower bound 7839 above the true optimum and 176 above its own
        incumbent, violating the ``bound <= incumbent`` invariant. The
        inversion threshold is :func:`bound_inversion_tolerance`, shared with
        AMP and the LOA path so all three agree on what counts as rounding
        rather than a real crossing.
        """
        lb, ub = _trace_value(certified_LB), _trace_value(UB)
        if lb is None or ub is None:
            return False
        return float(lb) - float(ub) > bound_inversion_tolerance(float(lb), float(ub))

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
        has_unresolved = bool(unresolved_int_configs)
        bound_valid = bool(
            final_lb is not None and not has_unresolved and not _certified_bound_inverted()
        )
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
        # Same narrowing as the single-tree builder above: a valid-but-OPEN
        # gap is not a certificate. ``bound_valid`` remains the validity
        # signal, still exported as ``master_bound_valid``/``bound_validity``.
        gap_certified = bool(bound_valid and final_gap is not None and final_gap <= gap_tolerance)
        summary = {
            "mip_count": int(mip_count),
            "nlp_subproblem_count": int(nlp_subproblem_count),
            "feasibility_subproblem_count": int(feasibility_subproblem_count),
            "unresolved_integer_config_count": int(len(unresolved_int_configs)),
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
        if has_unresolved:
            bound_validity = "uncertified"
        elif final_lb is not None:
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
            max_wall_time=_remaining_wall(t_start, time_limit),
        )
        wall_time = time.perf_counter() - t_start
        certified = _continuous_model_is_certified_convex(decomp) and not heuristic_nonconvex
        if x_sol is not None:
            LB = float(obj)
            UB = float(obj)
            if certified:
                _promote_certified_bound(obj, "continuous_nlp")
            # See ``_continuous_model_is_certified_convex``: one local NLP solve is
            # a global proof only on a convex model.
            return SolveResult(
                status="optimal" if certified else "feasible",
                objective=_obj_sign * obj,
                bound=_obj_sign * obj if certified else None,
                gap=0.0 if certified else None,
                gap_certified=certified,
                x=_build_x_dict(x_sol, model),
                wall_time=wall_time,
                mip_nlp_trace=_build_mip_nlp_trace(
                    "continuous_nlp_optimal" if certified else "continuous_nlp_local"
                ),
            )
        return SolveResult(
            status="infeasible" if certified else "no_feasible_point",
            objective=None,
            bound=None,
            gap=None,
            x=None,  # no incumbent -- see SolveResult.x's contract (#1105)
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
                max_wall_time=_remaining_wall(t_start, time_limit),
            )
            x_relax, obj_relax = relax_attempt.x, relax_attempt.objective
        else:
            x_relax, obj_relax = _solve_nlp_relaxation(
                evaluator,
                decomp.lb,
                decomp.ub,
                nlp_solver,
                initial_point=initial_point,
                max_wall_time=_remaining_wall(t_start, time_limit),
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
                    max_wall_time=_remaining_wall(t_start, time_limit),
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
                    max_wall_time=_remaining_wall(t_start, time_limit),
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
            time_limit=_master_time_budget(
                time_limit - elapsed,
                has_incumbent=incumbent is not None,
                checkin_remaining=(
                    None if master_checkin_deadline is None else master_checkin_deadline - elapsed
                ),
            ),
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
                max_wall_time=_remaining_wall(t_start, time_limit),
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
                # NLP returned no feasible point for this integer assignment.
                # A no-good cut is sound only when the fixed subproblem is
                # rigorously infeasible; local NLP failures stay unresolved.
                rigorously_infeasible = _fixed_subproblem_rigorously_infeasible(
                    evaluator,
                    decomp.lb,
                    decomp.ub,
                    decomp.int_indices,
                    x_master,
                )
                # The fixed NLP may have consumed the rest of the budget on its
                # way to failing, so re-check the deadline here rather than
                # relying on the per-candidate check above: restoration is a
                # SECOND full NLP solve, and running it past the deadline is
                # exactly the #1105 overrun (69.85 s of a 60 s limit, launched
                # with 38 s left). Its only product is cuts for a NEXT iteration
                # that the deadline has already ruled out, so skipping it costs
                # nothing.
                restoration_time_left = _time_left(t_start, time_limit)
                if feasibility_cuts and restoration_time_left <= _NLP_WALL_FLOOR_S:
                    logger.info(
                        "OA: skipping feasibility restoration at iteration %d; "
                        "%.3f s left of the %.3f s limit",
                        iteration,
                        restoration_time_left,
                        time_limit,
                    )
                    termination_reason = "time_limit"
                    stop_after_master_pool = True
                elif feasibility_cuts:
                    feasibility_subproblem_count += 1
                    x_feas = _solve_feasibility_subproblem(
                        evaluator,
                        decomp.lb,
                        decomp.ub,
                        decomp.int_indices,
                        x_master,
                        nlp_solver,
                        feasibility_norm,
                        max_wall_time=_remaining_wall(t_start, time_limit),
                        constraint_convex_mask=decomp.oa_constraint_mask,
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
                    and rigorously_infeasible
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
                elif safe_integer_cut_status and not rigorously_infeasible:
                    config_key = _int_config_key(x_master, decomp.int_indices)
                    already_seen = config_key in unresolved_int_configs
                    unresolved_int_configs.add(config_key)
                    if already_seen:
                        logger.info(
                            "OA: integer configuration %s unresolved by NLP "
                            "(non-rigorous failure) and re-proposed; stopping without "
                            "a no-good cut",
                            config_key,
                        )
                        termination_reason = "nonrigorous_nlp_failure"
                        stop_after_master_pool = True

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

            if stop_after_master_pool:
                break

            # Neither bound moved this iteration. Cuts are still being added, so
            # if that holds for long enough the master is not being changed in
            # any way that matters and the remaining budget is better spent by
            # the caller. See ``_OA_NO_PROGRESS_ITERATIONS``.
            _lb_now, _ub_now = _trace_value(LB), _trace_value(UB)
            if (
                _last_progress_bounds is not None
                and _lb_now is not None
                and _ub_now is not None
                and _last_progress_bounds[0] is not None
                and _last_progress_bounds[1] is not None
                and not _bounds_moved(_last_progress_bounds, (_lb_now, _ub_now))
            ):
                no_progress_iterations += 1
            else:
                no_progress_iterations = 0
            _last_progress_bounds = (_lb_now, _ub_now)
            if no_progress_iterations >= _OA_NO_PROGRESS_ITERATIONS:
                logger.info(
                    "OA: no bound movement in %d consecutive iterations "
                    "(LB=%s UB=%s, %d cuts); abandoning the cut loop",
                    no_progress_iterations,
                    _lb_now,
                    _ub_now,
                    len(oa_A_rows),
                )
                termination_reason = "stalling"
                stop_after_master_pool = True
                break

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

    # The dual bound crossed the incumbent. Report *nothing* proved rather than
    # a bound we know to be wrong: suppressing only the gap would still hand the
    # caller `bound > objective`, and the #1059 route then propagated exactly
    # that into the user's SolveResult. See ``_certified_bound_inverted``.
    if _certified_bound_inverted():
        logger.warning(
            "OA: certified lower bound %s is above the incumbent %s by more "
            "than rounding -- one of the two is wrong, so neither the bound nor "
            "the gap is reported as certified. This is a bound-validity defect "
            "worth investigating, not a tolerance to widen.",
            _trace_value(certified_LB),
            _trace_value(UB),
        )
        bound = None
        reported_gap = None
    final_reason = termination_reason
    if final_reason is None:
        if wall_time >= time_limit:
            final_reason = "time_limit"
        elif incumbent is not None and _certified_gap_converged():
            final_reason = "gap"
        else:
            final_reason = "iteration_limit"

    # C-35: if any integer configuration was left unresolved by a non-rigorous
    # NLP failure, the search is incomplete - we deliberately did NOT exclude
    # those configurations, so neither optimality nor infeasibility is proved.
    # Downgrade certification; never report a *certified* "infeasible" in this
    # state (an unresolved configuration might be the feasible/optimal one).
    has_unresolved = len(unresolved_int_configs) > 0
    if has_unresolved:
        reported_gap = None
        # #1105: ``bound`` and the trace must not contradict each other. The
        # trace publishes ``master_bound_valid``/``bound_validity`` computed as
        # ``final_lb is not None and not has_unresolved and not inverted`` --
        # so an unresolved configuration made the trace say
        # ``master_bound_valid=false, bound_validity="uncertified"`` while the
        # ``bound`` field still handed the caller the master number (reported
        # on ``kondili_recipe_pr46`` as ``bound=0.0`` beside exactly those two
        # trace fields). A consumer has no way to know which of the two to
        # believe, and the general ``bound`` field is the one read as a dual
        # certificate. Suppress it here and keep the number in the trace's
        # ``final_lb``, which is where a diagnostic belongs. This only ever
        # reports *less*, so it cannot manufacture a certificate.
        bound = None

    if incumbent is not None and incumbent_obj is not None:
        status = "optimal" if _certified_gap_converged() and not has_unresolved else "feasible"
        if termination_reason in {"cycling", "stalling"}:
            status = "feasible"
        # ``gap_certified`` must agree with ``status``: it is the field a user
        # reads (and ``result_io.summary_text`` renders) to decide whether the
        # reported gap is a certificate. Deriving it from ``reported_gap is not
        # None`` instead made it True on any run that merely *had* a gap --
        # ``syn40m`` returned ``status="feasible"`` with an 84% gap and
        # ``gap_certified=True``, printed with no "(uncertified)" marker. The
        # NLP-BB path already spells this the strict way (``solver.py``: a
        # ``feasible`` exit clears it), so the loose reading also made
        # ``gap_certified`` incomparable across the two routes. Bound validity
        # is a separate question and keeps its own signal in the trace's
        # ``master_bound_valid`` / ``bound_validity``.
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
            gap_certified=(status == "optimal"),
        )

    # No incumbent was found. A no-incumbent terminal state is only a *proof* of
    # infeasibility when the master MILP — a valid relaxation of the integer
    # feasible set carrying globally-valid OA and (rigorous) no-good cuts — was
    # itself proven infeasible: an infeasible relaxation implies the original is
    # infeasible. Those are the ``master_infeasible`` reasons below (including the
    # repair-attempted-and-failed variant). Every OTHER no-incumbent exit is
    # INCONCLUSIVE: a resource limit (``time_limit`` / ``iteration_limit``), user
    # termination, a master solver error, an unbounded master, cycling/stalling,
    # or a repair loop that gave up all leave part of the integer feasible set
    # unexplored, so a feasible (possibly optimal) assignment may still exist.
    # Reporting "infeasible" in those states is a false infeasibility certificate
    # (CLAUDE.md §1) — a solver that merely ran out of budget must never claim the
    # model is infeasible. Report "unknown" there, exactly as the C-35
    # unresolved-config path above does.
    #
    # C-35: if any integer configuration was left unresolved by a non-rigorous
    # NLP failure we deliberately did NOT exclude it, so infeasibility is not
    # proved either -- ``has_unresolved`` vetoes the "infeasible" arm as well.
    _PROVEN_INFEASIBLE_REASONS = {"master_infeasible", "master_infeasible_unrepaired"}
    # #1105: "unknown" is right for an exit whose cause we cannot name, but a
    # solve that simply ran out of its wall-clock or iteration budget has a
    # perfectly nameable cause, and it is the one fact the caller needs to
    # decide whether to retry with a larger budget. Both spellings are in the
    # ``SolveResult.status`` vocabulary and are what the B&B route already
    # returns for the same exits, so reporting them here also makes the two
    # routes comparable. Neither claims anything about feasibility.
    _LIMIT_REASON_STATUS = {"time_limit": "time_limit", "iteration_limit": "iteration_limit"}
    if final_reason in _PROVEN_INFEASIBLE_REASONS and not has_unresolved:
        status = "infeasible"
    else:
        status = _LIMIT_REASON_STATUS.get(final_reason or "", "unknown")

    # ``x=None``, not ``x={}``: ``SolveResult.x`` is documented as "None if no
    # feasible solution found", and an empty dict is truthy-checked as an
    # *existing* incumbent by every downstream consumer. ``Model.solve``'s
    # false-primal screen indexed ``result.x["x0"]`` and raised ``KeyError``,
    # which it then swallowed and warned that an "incumbent" was unscreened
    # even though ``objective`` was None; the #1059 dual-recovery block reads
    # the same dict and silently recovered nothing. (#1105)
    return SolveResult(
        status=status,
        objective=None,
        bound=(_obj_sign * bound if bound is not None else None),
        gap=None,
        x=None,
        wall_time=wall_time,
        mip_count=mip_count,
        subnlp_calls=nlp_subproblem_count,
        mip_nlp_trace=_build_mip_nlp_trace(final_reason),
        gap_certified=False,
    )
