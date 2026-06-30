"""Interior-point storage and rootsearch helpers for MIP-NLP methods."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import numpy as np

_CONSTRAINT_INF = 1e19
VectorLike = Sequence[float] | np.ndarray


class MIPNLPRootSearchStatus(str, Enum):
    """Structured rootsearch outcomes safe to place in traces."""

    DISABLED = "disabled"
    MISSING_INTERIOR_POINT = "missing_interior_point"
    INCOMPATIBLE_DISCRETE = "incompatible_discrete"
    INTERIOR_INFEASIBLE = "interior_infeasible"
    CANDIDATE_FEASIBLE = "candidate_feasible"
    NO_BRACKET = "no_bracket"
    CONVERGED = "converged"
    TIME_LIMIT = "time_limit"
    MAX_ITERATIONS = "max_iterations"
    EVALUATION_ERROR = "evaluation_error"


@dataclass
class MIPNLPInteriorPointRecord:
    """One stored MIP-NLP interior-point candidate with provenance metadata."""

    point: np.ndarray
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    discrete_signature: tuple[float, ...] = ()
    max_residual: Optional[float] = None

    def as_trace_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "metadata": dict(self.metadata),
            "discrete_signature": self.discrete_signature,
            "max_residual": self.max_residual,
        }


@dataclass
class MIPNLPRootSearchResult:
    """Result of line rootsearch between an interior point and a candidate."""

    status: MIPNLPRootSearchStatus
    point: Optional[np.ndarray] = None
    t: Optional[float] = None
    residual: Optional[float] = None
    iterations: int = 0
    strategy: str = "bisection"
    interior_source: Optional[str] = None
    message: Optional[str] = None

    def as_trace_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "t": self.t,
            "residual": self.residual,
            "iterations": int(self.iterations),
            "strategy": self.strategy,
            "interior_source": self.interior_source,
            "message": self.message,
        }


class MIPNLPInteriorPointStore:
    """Small provenance-aware store for feasible MIP-NLP reference points."""

    def __init__(
        self,
        n_vars: int,
        *,
        int_indices: Iterable[int] = (),
        lb: Optional[VectorLike] = None,
        ub: Optional[VectorLike] = None,
        integer_tol: float = 1e-6,
    ) -> None:
        self.n_vars = int(n_vars)
        self.int_indices = tuple(int(i) for i in int_indices)
        self.integer_tol = float(integer_tol)
        self.lb = None if lb is None else _as_vector("lb", lb, self.n_vars)
        self.ub = None if ub is None else _as_vector("ub", ub, self.n_vars)
        self.records: list[MIPNLPInteriorPointRecord] = []

    def add(
        self,
        point: VectorLike,
        *,
        source: str,
        metadata: Optional[dict[str, Any]] = None,
        evaluator=None,
        constraint_bounds=None,
        constraint_senses: Optional[Sequence[str]] = None,
        require_feasible: bool = False,
        feasibility_tol: float = 1e-7,
    ) -> Optional[MIPNLPInteriorPointRecord]:
        """Store a point, optionally only when it satisfies evaluator constraints."""
        x = _as_vector("point", point, self.n_vars)
        max_residual = None
        if evaluator is not None:
            try:
                max_residual = _max_constraint_residual(
                    evaluator,
                    x,
                    constraint_bounds=constraint_bounds,
                    constraint_senses=constraint_senses,
                )
            except Exception:
                if require_feasible:
                    return None
            else:
                if require_feasible and max_residual > float(feasibility_tol):
                    return None

        record = MIPNLPInteriorPointRecord(
            point=x.copy(),
            source=str(source),
            metadata=dict(metadata or {}),
            discrete_signature=self.discrete_signature(x),
            max_residual=max_residual,
        )
        self.records.append(record)
        return record

    def discrete_signature(self, point: VectorLike) -> tuple[float, ...]:
        x = _as_vector("point", point, self.n_vars)
        return tuple(
            _rounded_discrete_value(x[idx], self._lb_at(idx), self._ub_at(idx))
            for idx in self.int_indices
        )

    def is_compatible(
        self,
        record: MIPNLPInteriorPointRecord,
        candidate: VectorLike,
        *,
        fixed_discrete: bool = True,
    ) -> bool:
        if not fixed_discrete or not self.int_indices:
            return True
        return record.discrete_signature == self.discrete_signature(candidate)

    def select(
        self,
        candidate: VectorLike,
        *,
        fixed_discrete: bool = True,
    ) -> Optional[MIPNLPInteriorPointRecord]:
        for record in reversed(self.records):
            if self.is_compatible(record, candidate, fixed_discrete=fixed_discrete):
                return record
        return None

    def _lb_at(self, idx: int) -> Optional[float]:
        return None if self.lb is None else float(self.lb[idx])

    def _ub_at(self, idx: int) -> Optional[float]:
        return None if self.ub is None else float(self.ub[idx])


def rootsearch_from_store(
    evaluator,
    candidate: VectorLike,
    store: MIPNLPInteriorPointStore,
    *,
    strategy: str = "bisection",
    fixed_discrete: bool = True,
    constraint_bounds=None,
    constraint_senses: Optional[Sequence[str]] = None,
    feasibility_tol: float = 1e-7,
    residual_tol: float = 1e-7,
    x_tol: float = 1e-8,
    max_iterations: int = 60,
    time_limit: Optional[float] = None,
) -> MIPNLPRootSearchResult:
    """Run rootsearch using the newest compatible interior point in ``store``."""
    record = store.select(candidate, fixed_discrete=fixed_discrete)
    if record is None:
        status = (
            MIPNLPRootSearchStatus.INCOMPATIBLE_DISCRETE
            if fixed_discrete and store.records and store.int_indices
            else MIPNLPRootSearchStatus.MISSING_INTERIOR_POINT
        )
        return MIPNLPRootSearchResult(status=status, strategy=_normalize_strategy(strategy))

    result = rootsearch_between_points(
        evaluator,
        record.point,
        candidate,
        strategy=strategy,
        int_indices=store.int_indices,
        fixed_discrete=fixed_discrete,
        lb=store.lb,
        ub=store.ub,
        constraint_bounds=constraint_bounds,
        constraint_senses=constraint_senses,
        feasibility_tol=feasibility_tol,
        residual_tol=residual_tol,
        x_tol=x_tol,
        max_iterations=max_iterations,
        time_limit=time_limit,
    )
    result.interior_source = record.source
    return result


def rootsearch_between_points(
    evaluator,
    interior_point: VectorLike,
    candidate: VectorLike,
    *,
    strategy: str = "bisection",
    int_indices: Iterable[int] = (),
    fixed_discrete: bool = False,
    lb: Optional[VectorLike] = None,
    ub: Optional[VectorLike] = None,
    constraint_bounds=None,
    constraint_senses: Optional[Sequence[str]] = None,
    feasibility_tol: float = 1e-7,
    residual_tol: float = 1e-7,
    x_tol: float = 1e-8,
    max_iterations: int = 60,
    time_limit: Optional[float] = None,
) -> MIPNLPRootSearchResult:
    """Find the first nonlinear-constraint boundary point on a segment.

    ``interior_point`` is expected to satisfy the model constraints and
    ``candidate`` is usually a MIP/NLP candidate outside the nonlinear feasible
    region. With ``fixed_discrete=True``, integer slots are held at the
    candidate values for the entire segment.
    """
    strategy_name = _normalize_strategy(strategy)
    if strategy_name == "none":
        return MIPNLPRootSearchResult(status=MIPNLPRootSearchStatus.DISABLED, strategy="none")

    n_vars = int(getattr(evaluator, "n_variables", len(np.asarray(candidate).reshape(-1))))
    start = _as_vector("interior_point", interior_point, n_vars)
    end = _as_vector("candidate", candidate, n_vars)
    int_idx = tuple(int(i) for i in int_indices)
    lb_vec = None if lb is None else _as_vector("lb", lb, n_vars)
    ub_vec = None if ub is None else _as_vector("ub", ub, n_vars)

    if fixed_discrete and int_idx:
        start_sig = tuple(
            _rounded_discrete_value(start[idx], _bound_at(lb_vec, idx), _bound_at(ub_vec, idx))
            for idx in int_idx
        )
        end_sig = tuple(
            _rounded_discrete_value(end[idx], _bound_at(lb_vec, idx), _bound_at(ub_vec, idx))
            for idx in int_idx
        )
        if start_sig != end_sig:
            return MIPNLPRootSearchResult(
                status=MIPNLPRootSearchStatus.INCOMPATIBLE_DISCRETE,
                strategy=strategy_name,
            )
        for idx in int_idx:
            start[idx] = end[idx]

    deadline = None if time_limit is None else time.perf_counter() + max(0.0, float(time_limit))
    if _timed_out(deadline):
        return MIPNLPRootSearchResult(
            status=MIPNLPRootSearchStatus.TIME_LIMIT,
            strategy=strategy_name,
        )

    try:
        phi_start = _max_constraint_residual(
            evaluator,
            start,
            constraint_bounds=constraint_bounds,
            constraint_senses=constraint_senses,
        )
        phi_end = _max_constraint_residual(
            evaluator,
            end,
            constraint_bounds=constraint_bounds,
            constraint_senses=constraint_senses,
        )
    except Exception as exc:
        return MIPNLPRootSearchResult(
            status=MIPNLPRootSearchStatus.EVALUATION_ERROR,
            strategy=strategy_name,
            message=f"{type(exc).__name__}: {exc}",
        )

    if phi_start > float(feasibility_tol):
        return MIPNLPRootSearchResult(
            status=MIPNLPRootSearchStatus.INTERIOR_INFEASIBLE,
            point=start,
            t=0.0,
            residual=float(phi_start),
            strategy=strategy_name,
        )
    if phi_end <= float(feasibility_tol):
        return MIPNLPRootSearchResult(
            status=MIPNLPRootSearchStatus.CANDIDATE_FEASIBLE,
            point=end,
            t=1.0,
            residual=float(phi_end),
            strategy=strategy_name,
        )

    def point_at(t: float) -> np.ndarray:
        return np.asarray(start + float(t) * (end - start), dtype=np.float64)

    def phi(t: float) -> float:
        if _timed_out(deadline):
            raise _RootSearchTimeout
        return _max_constraint_residual(
            evaluator,
            point_at(t),
            constraint_bounds=constraint_bounds,
            constraint_senses=constraint_senses,
        )

    if strategy_name == "toms748":
        toms_result = _try_toms748(
            phi,
            point_at,
            residual_tol=float(residual_tol),
            x_tol=float(x_tol),
            max_iterations=int(max_iterations),
        )
        if toms_result.status is not MIPNLPRootSearchStatus.NO_BRACKET:
            return toms_result

    return _bisect_root(
        phi,
        point_at,
        residual_tol=float(residual_tol),
        x_tol=float(x_tol),
        max_iterations=int(max_iterations),
        strategy="bisection" if strategy_name == "auto" else strategy_name,
    )


def _bisect_root(
    phi,
    point_at,
    *,
    residual_tol: float,
    x_tol: float,
    max_iterations: int,
    strategy: str,
) -> MIPNLPRootSearchResult:
    lo = 0.0
    hi = 1.0
    last_t = 0.5
    last_residual = None
    for iteration in range(1, max(1, max_iterations) + 1):
        last_t = 0.5 * (lo + hi)
        try:
            last_residual = float(phi(last_t))
        except _RootSearchTimeout:
            return MIPNLPRootSearchResult(
                status=MIPNLPRootSearchStatus.TIME_LIMIT,
                point=point_at(lo),
                t=lo,
                residual=last_residual,
                iterations=iteration - 1,
                strategy=strategy,
            )
        if abs(last_residual) <= residual_tol or (hi - lo) <= x_tol:
            return MIPNLPRootSearchResult(
                status=MIPNLPRootSearchStatus.CONVERGED,
                point=point_at(last_t),
                t=last_t,
                residual=last_residual,
                iterations=iteration,
                strategy=strategy,
            )
        if last_residual <= 0.0:
            lo = last_t
        else:
            hi = last_t

    return MIPNLPRootSearchResult(
        status=MIPNLPRootSearchStatus.MAX_ITERATIONS,
        point=point_at(last_t),
        t=last_t,
        residual=last_residual,
        iterations=max(1, max_iterations),
        strategy=strategy,
    )


def _try_toms748(
    phi,
    point_at,
    *,
    residual_tol: float,
    x_tol: float,
    max_iterations: int,
) -> MIPNLPRootSearchResult:
    try:
        from scipy.optimize import toms748

        root = float(toms748(phi, 0.0, 1.0, xtol=x_tol, rtol=x_tol, maxiter=max_iterations))
        residual = float(phi(root))
    except _RootSearchTimeout:
        return MIPNLPRootSearchResult(
            status=MIPNLPRootSearchStatus.TIME_LIMIT,
            strategy="toms748",
        )
    except Exception as exc:
        return MIPNLPRootSearchResult(
            status=MIPNLPRootSearchStatus.NO_BRACKET,
            strategy="toms748",
            message=f"toms748_fallback: {type(exc).__name__}: {exc}",
        )
    return MIPNLPRootSearchResult(
        status=MIPNLPRootSearchStatus.CONVERGED,
        point=point_at(root),
        t=root,
        residual=residual,
        iterations=0,
        strategy="toms748",
    )


def _max_constraint_residual(
    evaluator,
    x: np.ndarray,
    *,
    constraint_bounds=None,
    constraint_senses: Optional[Sequence[str]] = None,
) -> float:
    residuals = _constraint_residuals(
        evaluator,
        x,
        constraint_bounds=constraint_bounds,
        constraint_senses=constraint_senses,
    )
    if residuals.size == 0:
        return float("-inf")
    return float(np.max(residuals))


def _constraint_residuals(
    evaluator,
    x: np.ndarray,
    *,
    constraint_bounds=None,
    constraint_senses: Optional[Sequence[str]] = None,
) -> np.ndarray:
    m = int(getattr(evaluator, "n_constraints", 0))
    if m == 0:
        return np.empty(0, dtype=np.float64)
    cons = np.asarray(evaluator.evaluate_constraints(x), dtype=np.float64).reshape(-1)
    if cons.size != m:
        raise ValueError(f"evaluator returned {cons.size} constraints; expected {m}.")
    cl, cu = _constraint_bounds(
        evaluator,
        constraint_bounds=constraint_bounds,
        constraint_senses=constraint_senses,
    )
    residual_parts: list[np.ndarray] = []
    upper_mask = cu < _CONSTRAINT_INF
    lower_mask = cl > -_CONSTRAINT_INF
    if np.any(upper_mask):
        residual_parts.append(cons[upper_mask] - cu[upper_mask])
    if np.any(lower_mask):
        residual_parts.append(cl[lower_mask] - cons[lower_mask])
    if not residual_parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(residual_parts)


def _constraint_bounds(
    evaluator,
    *,
    constraint_bounds=None,
    constraint_senses: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    m = int(getattr(evaluator, "n_constraints", 0))
    if constraint_bounds is not None:
        if isinstance(constraint_bounds, tuple) and len(constraint_bounds) == 2:
            cl = np.asarray(constraint_bounds[0], dtype=np.float64).reshape(-1)
            cu = np.asarray(constraint_bounds[1], dtype=np.float64).reshape(-1)
        else:
            bounds = np.asarray(constraint_bounds, dtype=np.float64)
            if bounds.shape != (m, 2):
                raise ValueError(f"constraint_bounds must have shape ({m}, 2).")
            cl = bounds[:, 0]
            cu = bounds[:, 1]
        if cl.shape != (m,) or cu.shape != (m,):
            raise ValueError(f"constraint bounds must have length {m}.")
        return cl, cu

    if constraint_senses is not None:
        if len(constraint_senses) != m:
            raise ValueError(
                f"constraint_senses has length {len(constraint_senses)}; expected {m}."
            )
        cl = np.full(m, -_CONSTRAINT_INF, dtype=np.float64)
        cu = np.full(m, _CONSTRAINT_INF, dtype=np.float64)
        for i, sense in enumerate(constraint_senses):
            if sense == "<=":
                cu[i] = 0.0
            elif sense == ">=":
                cl[i] = 0.0
            elif sense == "==":
                cl[i] = 0.0
                cu[i] = 0.0
            else:
                raise ValueError(f"Unknown constraint sense: {sense!r}.")
        return cl, cu

    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    return _infer_constraint_bounds(evaluator)


def _normalize_strategy(strategy: str) -> str:
    key = str(strategy).strip().lower().replace("-", "_")
    if key in {"", "auto"}:
        return "bisection"
    if key in {"none", "bisection", "toms748"}:
        return key
    raise ValueError("rootsearch strategy must be one of: auto, none, bisection, toms748.")


def _as_vector(name: str, values: VectorLike, n_vars: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if n_vars is not None and arr.shape != (int(n_vars),):
        raise ValueError(f"{name} has shape {arr.shape}; expected ({int(n_vars)},).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _rounded_discrete_value(
    value: float,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
) -> float:
    rounded = float(np.floor(float(value) + 0.5))
    if lb is None or ub is None:
        return rounded
    lo = float(np.ceil(lb))
    hi = float(np.floor(ub))
    if lo <= hi:
        return float(np.clip(rounded, lo, hi))
    return rounded


def _bound_at(values: Optional[np.ndarray], idx: int) -> Optional[float]:
    return None if values is None else float(values[idx])


def _timed_out(deadline: Optional[float]) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


class _RootSearchTimeout(Exception):
    pass
