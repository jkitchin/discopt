"""Build and persist :class:`SolveRecord` telemetry from an advisor run.

The bridge between the advisor (which produced scores + a recommendation) and the
store. ``build_record`` assembles the record; ``record_outcome`` builds it from a
:class:`~discopt.decomposition.advisor.DecompositionAdvisor` and appends it. The
caller supplies the timestamp so records stay reproducible in tests (design §10.2
keeps ``Date.now``-style nondeterminism out of the builder).
"""

from __future__ import annotations

from discopt.decomposition.advisor.analyzer import StructureReport
from discopt.decomposition.advisor.scoring import ScoreVector
from discopt.decomposition.advisor.types import Candidate, MethodKind
from discopt.decomposition.learning.features import extract_features, fingerprint
from discopt.decomposition.learning.record import ObservedPerformance, Outcome, SolveRecord
from discopt.decomposition.learning.store import RecordStore


def build_record(
    scored: list[tuple[Candidate, ScoreVector]],
    report: StructureReport,
    chosen: MethodKind,
    *,
    observed: ObservedPerformance | None = None,
    solver_config: dict | None = None,
    outcome: Outcome | str = Outcome.UNKNOWN,
    timestamp: float = 0.0,
) -> SolveRecord:
    """Assemble a :class:`SolveRecord` from scored candidates + the chosen method."""
    considered = [(c.method.value, sv.aggregate) for c, sv in scored]
    sv_by_method = {c.method: sv for c, sv in scored}
    chosen_sv = sv_by_method.get(chosen)
    predicted_speedup = (
        chosen_sv.performance.estimated_speedup
        if chosen_sv is not None and chosen_sv.performance is not None
        else None
    )
    predicted_conf = chosen_sv.confidence if chosen_sv is not None else None
    return SolveRecord(
        instance_fingerprint=fingerprint(report),
        features=extract_features(report),
        chosen=chosen.value,
        considered=considered,
        predicted_speedup=predicted_speedup,
        predicted_confidence=predicted_conf,
        observed=observed,
        solver_config=solver_config or {},
        outcome=outcome.value if isinstance(outcome, Outcome) else outcome,
        timestamp=timestamp,
    )


def record_outcome(
    advisor,
    store: RecordStore,
    *,
    observed: ObservedPerformance | None = None,
    chosen: MethodKind | None = None,
    solver_config: dict | None = None,
    outcome: Outcome | str = Outcome.OPTIMAL,
    timestamp: float = 0.0,
) -> SolveRecord:
    """Build a record from *advisor* and append it to *store*.

    Defaults *chosen* to the advisor's recommendation. Pass *observed* once the
    solve has run so predicted-vs-observed can later be compared.
    """
    scored = advisor.scores()
    report = advisor.structure()
    if chosen is None:
        chosen = advisor.recommendation().recommendation
    record = build_record(
        scored,
        report,
        chosen,
        observed=observed,
        solver_config=solver_config,
        outcome=outcome,
        timestamp=timestamp,
    )
    store.append(record)
    return record


__all__ = ["build_record", "record_outcome"]
