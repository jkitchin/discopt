"""Fixed-integer NLP candidate management for MIP-NLP decomposition."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

_SOURCE_ORDER = {
    "mip_optimum": 0,
    "lp_relaxation": 1,
    "solution_pool": 2,
    "rootsearch": 3,
    "external": 4,
}


def _round_integral_to_bounds(value: float, lb: float, ub: float) -> float:
    rounded = float(np.floor(float(value) + 0.5))
    lo = float(np.ceil(lb))
    hi = float(np.floor(ub))
    if lo <= hi:
        return float(np.clip(rounded, lo, hi))
    return float(np.clip(rounded, lb, ub))


@dataclass(frozen=True)
class FixedNLPCandidate:
    """One candidate integer assignment for a fixed-integer NLP solve."""

    point: np.ndarray
    source: str
    objective: Optional[float] = None
    iteration: Optional[int] = None
    nlp_source: str = "active"
    provider: Optional[str] = None
    sequence: int = 0
    integer_assignment: tuple[float, ...] = field(default_factory=tuple)

    def trace_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "objective": None if self.objective is None else float(self.objective),
            "iteration": self.iteration,
            "nlp_source": self.nlp_source,
            "provider": self.provider,
            "integer_assignment": [float(v) for v in self.integer_assignment],
            "sequence": int(self.sequence),
        }


class FixedNLPCandidateManager:
    """Queue and schedule fixed-NLP candidates.

    The manager keeps ordering deterministic, deduplicates integer assignments,
    and exposes a small scheduler that mirrors SHOT's always/iteration/time/pool
    activation modes without coupling to a particular MIP or NLP backend.
    """

    def __init__(
        self,
        *,
        n_vars: int,
        int_indices: list[int],
        lb,
        ub,
        strategy: str = "always",
        iteration_frequency: int = 1,
        time_frequency: float = 0.0,
        candidate_limit: Optional[int] = None,
        deduplicate_used_assignments: bool = False,
    ) -> None:
        self.n_vars = int(n_vars)
        self.int_indices = list(int_indices)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.strategy = self._normalize_strategy(strategy)
        self.iteration_frequency = max(1, int(iteration_frequency))
        self.time_frequency = max(0.0, float(time_frequency))
        self.candidate_limit = None if candidate_limit is None else max(1, int(candidate_limit))
        self.deduplicate_used_assignments = bool(deduplicate_used_assignments)
        self._pending: list[FixedNLPCandidate] = []
        self._pending_keys: set[tuple[float, ...]] = set()
        self._used_keys: set[tuple[float, ...]] = set()
        self._sequence = 0
        self._last_call_iteration: Optional[int] = None
        self._last_call_time: Optional[float] = None
        self._last_success: Optional[bool] = None
        self.added_source_counts: Counter[str] = Counter()
        self.skipped_duplicate_count = 0

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        key = str(strategy).strip().lower().replace("-", "_")
        if key in {"auto", "adaptive", "always", "iteration", "time", "solution_pool", "none"}:
            return key
        raise ValueError(
            "fixed_nlp_strategy must be one of: adaptive, always, auto, "
            "iteration, none, solution_pool, time."
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @staticmethod
    def _sort_key(candidate: FixedNLPCandidate) -> tuple[int, float, int]:
        return (
            _SOURCE_ORDER.get(candidate.source, 50),
            float("inf") if candidate.objective is None else float(candidate.objective),
            candidate.sequence,
        )

    def assignment_key(self, point) -> tuple[float, ...]:
        x = np.asarray(point, dtype=np.float64).reshape(-1)
        return tuple(
            _round_integral_to_bounds(float(x[idx]), float(self.lb[idx]), float(self.ub[idx]))
            for idx in self.int_indices
        )

    def add(
        self,
        point,
        *,
        source: str,
        objective: Optional[float] = None,
        iteration: Optional[int] = None,
        nlp_source: str = "active",
        provider: Optional[str] = None,
    ) -> bool:
        x = np.asarray(point, dtype=np.float64).reshape(-1)
        if x.size < self.n_vars:
            return False
        x = np.clip(x[: self.n_vars].copy(), self.lb, self.ub)
        key = self.assignment_key(x)
        candidate = FixedNLPCandidate(
            point=x,
            source=str(source),
            objective=None if objective is None else float(objective),
            iteration=iteration,
            nlp_source=str(nlp_source),
            provider=provider,
            sequence=self._sequence,
            integer_assignment=key,
        )
        if key in self._pending_keys:
            for idx, existing in enumerate(self._pending):
                if existing.integer_assignment == key:
                    if self._sort_key(candidate) < self._sort_key(existing):
                        self._pending[idx] = candidate
                        self._sequence += 1
                        self.added_source_counts[str(source)] += 1
                        return True
                    break
            self.skipped_duplicate_count += 1
            return False
        if self.deduplicate_used_assignments and key in self._used_keys:
            self.skipped_duplicate_count += 1
            return False
        self._sequence += 1
        self._pending.append(candidate)
        self._pending_keys.add(key)
        self.added_source_counts[str(source)] += 1
        return True

    def add_external_candidates(
        self,
        candidates,
        *,
        iteration: Optional[int] = None,
        provider: Optional[str] = None,
        nlp_source: str = "active",
    ) -> int:
        added = 0
        for candidate in candidates or []:
            objective = None
            point = candidate
            source = "external"
            if isinstance(candidate, dict):
                point = candidate.get("point")
                objective = candidate.get("objective")
                source = str(candidate.get("source", "external"))
                provider = candidate.get("provider", provider)
                nlp_source = str(candidate.get("nlp_source", nlp_source))
            if point is None:
                continue
            if self.add(
                point,
                source=source,
                objective=objective,
                iteration=iteration,
                nlp_source=nlp_source,
                provider=provider,
            ):
                added += 1
        return added

    def add_master_result(
        self,
        master_result: Any,
        *,
        iteration: int,
        solution_pool: bool,
        limit: int,
        nlp_source: str = "active",
    ) -> int:
        if getattr(master_result, "x", None) is None:
            return 0
        before = self.pending_count
        incumbent = np.asarray(master_result.x, dtype=np.float64).reshape(-1)
        self.add(
            incumbent,
            source="mip_optimum",
            objective=getattr(master_result, "objective", None),
            iteration=iteration,
            nlp_source=nlp_source,
        )
        if solution_pool:
            pool = list(getattr(master_result, "solution_pool", None) or [])
            pool_objectives = list(getattr(master_result, "solution_pool_objectives", None) or [])
            for idx, raw in enumerate(pool):
                objective = pool_objectives[idx] if idx < len(pool_objectives) else None
                self.add(
                    raw,
                    source="solution_pool",
                    objective=objective,
                    iteration=iteration,
                    nlp_source=nlp_source,
                )
                if self.candidate_limit is not None and self.pending_count - before >= limit:
                    break
        return self.pending_count - before

    def should_call(
        self,
        *,
        iteration: int,
        elapsed: Optional[float] = None,
        has_solution_pool_candidate: bool = False,
    ) -> bool:
        if self.strategy == "none" or not self._pending:
            return False
        if self.strategy == "always":
            return True
        if self.strategy == "solution_pool":
            return bool(has_solution_pool_candidate)
        if self.strategy == "iteration":
            if self._last_call_iteration is None:
                return True
            return (int(iteration) - self._last_call_iteration) >= self.iteration_frequency
        if self.strategy == "time":
            now = time.perf_counter() if elapsed is None else float(elapsed)
            if self._last_call_time is None:
                return True
            return (now - self._last_call_time) >= self.time_frequency
        # adaptive/auto: call immediately after a productive solve, and otherwise
        # fall back to the iteration/time gates.
        if self._last_success is True:
            return True
        if self._last_call_iteration is None:
            return True
        iter_ready = (int(iteration) - self._last_call_iteration) >= self.iteration_frequency
        now = time.perf_counter() if elapsed is None else float(elapsed)
        time_ready = (now - float(self._last_call_time or 0.0)) >= self.time_frequency
        return bool(iter_ready or time_ready)

    def take_ready(
        self,
        *,
        iteration: int,
        elapsed: Optional[float] = None,
        has_solution_pool_candidate: bool = False,
    ) -> list[FixedNLPCandidate]:
        if not self.should_call(
            iteration=iteration,
            elapsed=elapsed,
            has_solution_pool_candidate=has_solution_pool_candidate,
        ):
            return []
        ordered = sorted(
            self._pending,
            key=self._sort_key,
        )
        if self.candidate_limit is not None:
            ready = ordered[: self.candidate_limit]
            keep = ordered[self.candidate_limit :]
        else:
            ready = ordered
            keep = []
        self._pending = keep
        self._pending_keys = {cand.integer_assignment for cand in keep}
        for cand in ready:
            self._used_keys.add(cand.integer_assignment)
        return ready

    def record_call_result(
        self,
        candidate: FixedNLPCandidate,
        *,
        iteration: int,
        elapsed: Optional[float],
        success: bool,
    ) -> None:
        del candidate
        self._last_call_iteration = int(iteration)
        self._last_call_time = time.perf_counter() if elapsed is None else float(elapsed)
        self._last_success = bool(success)

    def scheduler_trace(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "iteration_frequency": int(self.iteration_frequency),
            "time_frequency": float(self.time_frequency),
            "candidate_limit": self.candidate_limit,
            "pending": int(self.pending_count),
            "deduplicate_used_assignments": bool(self.deduplicate_used_assignments),
            "skipped_duplicates": int(self.skipped_duplicate_count),
            "added_source_counts": {
                str(source): int(count)
                for source, count in sorted(self.added_source_counts.items())
            },
        }
