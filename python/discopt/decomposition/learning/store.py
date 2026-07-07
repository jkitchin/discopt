"""Append-only store for solve telemetry (design §10.4).

A dependency-light, local-first record store: JSON Lines on disk (one
:class:`~discopt.decomposition.learning.record.SolveRecord` per line), plus an
in-memory mode for tests. Nothing leaves the machine; a shareable/anonymized
benchmark export is a separate, opt-in step (a research artifact, design §10.4).

The store also does the one retrieval the instance-based learner needs:
``nearest`` — the k records whose feature vectors are closest to a query, by
range-normalized Euclidean distance.
"""

from __future__ import annotations

import json
import math
import os

from discopt.decomposition.learning.record import InstanceFeatures, SolveRecord


class RecordStore:
    """Append-only collection of :class:`SolveRecord`, JSONL-backed or in-memory.

    Parameters
    ----------
    path : str | None
        JSONL file to append to / load from. ``None`` keeps records in memory
        only (useful for tests and ephemeral sessions).
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._records: list[SolveRecord] = []
        if path is not None and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._records.append(SolveRecord.from_dict(json.loads(line)))

    def append(self, record: SolveRecord) -> None:
        """Add *record*, persisting to disk if the store is file-backed."""
        self._records.append(record)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict()) + "\n")

    def all(self) -> list[SolveRecord]:
        """All records, in insertion order."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def nearest(self, features: InstanceFeatures, k: int = 5) -> list[tuple[SolveRecord, float]]:
        """Return up to *k* ``(record, distance)`` pairs closest to *features*.

        Distance is Euclidean on the feature vectors after per-component
        range-normalization across the stored set (so no single component
        dominates). An empty store returns ``[]``.
        """
        if not self._records:
            return []
        vectors = [r.features.vector() for r in self._records]
        query = features.vector()
        dim = len(query)
        # Per-component range for normalization (guard zero range).
        lo = [min(v[i] for v in vectors + [query]) for i in range(dim)]
        hi = [max(v[i] for v in vectors + [query]) for i in range(dim)]
        span = [(hi[i] - lo[i]) or 1.0 for i in range(dim)]

        def dist(v: list[float]) -> float:
            return math.sqrt(sum(((v[i] - query[i]) / span[i]) ** 2 for i in range(dim)))

        scored = [(r, dist(v)) for r, v in zip(self._records, vectors)]
        scored.sort(key=lambda rd: rd[1])
        return scored[:k]


__all__ = ["RecordStore"]
