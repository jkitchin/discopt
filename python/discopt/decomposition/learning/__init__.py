"""Learning-from-experience for the Decomposition Advisor (design §8, §10).

Phase 7 foundation: record every solve, and let past records improve future
recommendations. The pieces:

- :mod:`record` — the :class:`SolveRecord` telemetry schema (features + what was
  predicted + what was observed).
- :mod:`features` — scalar feature extraction and instance fingerprinting.
- :mod:`store` — an append-only, local-first (JSONL) :class:`RecordStore` with
  nearest-neighbor retrieval.
- :mod:`recorder` — build/persist a record from an advisor run.
- :mod:`policies` — :class:`InstanceBasedPolicy`, a learned selection policy that
  drops in behind the advisor's :class:`~discopt.decomposition.advisor.selection.Policy`
  seam and safely defers to the rule-based default until enough data accrues.

All local-first and dependency-light; nothing leaves the machine unless the user
exports it.
"""

from __future__ import annotations

from discopt.decomposition.learning.features import extract_features, fingerprint
from discopt.decomposition.learning.policies import InstanceBasedPolicy
from discopt.decomposition.learning.record import (
    InstanceFeatures,
    ObservedPerformance,
    Outcome,
    SolveRecord,
)
from discopt.decomposition.learning.recorder import build_record, record_outcome
from discopt.decomposition.learning.store import RecordStore

__all__ = [
    "InstanceBasedPolicy",
    "InstanceFeatures",
    "ObservedPerformance",
    "Outcome",
    "RecordStore",
    "SolveRecord",
    "build_record",
    "extract_features",
    "fingerprint",
    "record_outcome",
]
