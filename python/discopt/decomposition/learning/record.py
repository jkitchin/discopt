"""Telemetry schema for the Decomposition Advisor (design §8, §10.2).

The advisor improves over time only if every solve is recorded. This module
defines the immutable record written after a solve — the instance's features, the
method the advisor chose, what it *predicted*, and what was *observed*. Recording
``predicted`` next to ``observed`` is the whole point: it makes the advisor's own
error measurable, which is what later calibrates the scoring weights and
confidences (design §10.2).

Everything here is dependency-light and JSON-serializable so the store
(:mod:`discopt.decomposition.learning.store`) can be a plain append-only file —
no database required, nothing leaves the machine unless the user exports it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum


class Outcome(Enum):
    """Terminal status of a solve."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InstanceFeatures:
    """Cheap scalar features describing an instance (design §10.3).

    These feed the instance-based / portfolio learners. Graph embeddings for the
    neural options are a later addition; the scalar features come straight from
    the :class:`~discopt.decomposition.advisor.analyzer.StructureReport` with no
    extra passes.
    """

    num_vars: int
    num_constraints: int
    num_incidences: int
    integer_fraction: float
    coupling_density: float
    num_blocks: int
    blocks_after_integer_projection: int
    nonlinear: bool

    @staticmethod
    def names() -> list[str]:
        """Names of the numeric feature-vector components, in order."""
        return [
            "log_num_vars",
            "log_num_constraints",
            "log_num_incidences",
            "integer_fraction",
            "coupling_density",
            "log_num_blocks",
            "log_blocks_after_projection",
            "nonlinear",
        ]

    def vector(self) -> list[float]:
        """Numeric feature vector for distance/retrieval.

        Counts are ``log1p``-scaled so a large model does not dominate the
        distance; fractions and the nonlinear flag are already in ``[0, 1]``.
        """
        return [
            math.log1p(self.num_vars),
            math.log1p(self.num_constraints),
            math.log1p(self.num_incidences),
            self.integer_fraction,
            self.coupling_density,
            math.log1p(self.num_blocks),
            math.log1p(self.blocks_after_integer_projection),
            1.0 if self.nonlinear else 0.0,
        ]

    def to_dict(self) -> dict:
        """JSON-serializable form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceFeatures":
        """Reconstruct from :meth:`to_dict`."""
        return cls(**d)


@dataclass(frozen=True)
class ObservedPerformance:
    """What actually happened when the chosen method ran."""

    wall_clock_s: float
    iterations: int | None = None
    converged: bool = True
    gap: float | None = None
    peak_mem_mb: float | None = None
    master_time_s: float | None = None
    subproblem_time_s: float | None = None

    def to_dict(self) -> dict:
        """JSON-serializable form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObservedPerformance":
        """Reconstruct from :meth:`to_dict`."""
        return cls(**d)


@dataclass(frozen=True)
class SolveRecord:
    """One solve's telemetry (design §10.2), the unit the store persists.

    Attributes
    ----------
    instance_fingerprint : str
        Stable hash of the instance's structure, for grouping re-solves.
    features : InstanceFeatures
        The instance's scalar features.
    chosen : str
        ``MethodKind.value`` the advisor recommended / that was run.
    considered : list[tuple[str, float]]
        ``(method, aggregate)`` for every scored candidate — the runners-up too,
        so a learner can see what was *not* picked.
    predicted_speedup, predicted_confidence : float | None
        What the advisor forecast for the chosen method.
    observed : ObservedPerformance | None
        Measured performance (``None`` if only the prediction is being logged).
    solver_config : dict
        Tolerances / threads / backend in effect.
    outcome : str
        ``Outcome.value``.
    timestamp : float
        Unix seconds; supplied by the caller (kept out of the record builder so
        results stay reproducible in tests).
    """

    instance_fingerprint: str
    features: InstanceFeatures
    chosen: str
    considered: list[tuple[str, float]] = field(default_factory=list)
    predicted_speedup: float | None = None
    predicted_confidence: float | None = None
    observed: ObservedPerformance | None = None
    solver_config: dict = field(default_factory=dict)
    outcome: str = Outcome.UNKNOWN.value
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """JSON-serializable form."""
        return {
            "instance_fingerprint": self.instance_fingerprint,
            "features": self.features.to_dict(),
            "chosen": self.chosen,
            "considered": [list(c) for c in self.considered],
            "predicted_speedup": self.predicted_speedup,
            "predicted_confidence": self.predicted_confidence,
            "observed": None if self.observed is None else self.observed.to_dict(),
            "solver_config": self.solver_config,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SolveRecord":
        """Reconstruct from :meth:`to_dict`."""
        obs = d.get("observed")
        return cls(
            instance_fingerprint=d["instance_fingerprint"],
            features=InstanceFeatures.from_dict(d["features"]),
            chosen=d["chosen"],
            considered=[tuple(c) for c in d.get("considered", [])],
            predicted_speedup=d.get("predicted_speedup"),
            predicted_confidence=d.get("predicted_confidence"),
            observed=None if obs is None else ObservedPerformance.from_dict(obs),
            solver_config=d.get("solver_config", {}),
            outcome=d.get("outcome", Outcome.UNKNOWN.value),
            timestamp=d.get("timestamp", 0.0),
        )


__all__ = [
    "InstanceFeatures",
    "ObservedPerformance",
    "Outcome",
    "SolveRecord",
]
