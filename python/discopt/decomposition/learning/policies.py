"""Learned selection policies (design §10.1).

The first — and most robust — learning option is *instance-based*: retrieve the
nearest past instances from the store and let their observed winners bias the
recommendation. :class:`InstanceBasedPolicy` implements the same
:class:`~discopt.decomposition.advisor.selection.Policy` interface as the
rule-based default, so it drops into the advisor with no other change — the whole
point of isolating selection behind a policy seam. With too little data it defers
entirely to the rule-based policy, so it is safe from day one.

Bandit / portfolio / learned-ranker / GNN policies (design §10.1) are future
additions behind the same interface.
"""

from __future__ import annotations

from dataclasses import replace

from discopt.decomposition.advisor.analyzer import StructureReport
from discopt.decomposition.advisor.scoring import ScoreVector
from discopt.decomposition.advisor.selection import (
    Policy,
    Ranked,
    RuleBasedPolicy,
    SelectionContext,
)
from discopt.decomposition.advisor.types import Candidate, MethodKind
from discopt.decomposition.learning.features import extract_features
from discopt.decomposition.learning.store import RecordStore

_NEG_INF = float("-inf")


class InstanceBasedPolicy:
    """Bias selection toward methods that won on nearby past instances.

    Parameters
    ----------
    store : RecordStore
        Source of past solve records.
    base : Policy | None
        Fallback policy (default :class:`RuleBasedPolicy`); used verbatim when
        there is too little evidence.
    k : int
        Neighbors to retrieve.
    min_records : int
        Minimum neighbors required before overriding the base policy.
    """

    def __init__(
        self,
        store: RecordStore,
        base: Policy | None = None,
        *,
        k: int = 5,
        min_records: int = 3,
    ) -> None:
        self.store = store
        self.base = base or RuleBasedPolicy()
        self.k = k
        self.min_records = min_records

    def rank(
        self,
        scored: list[tuple[Candidate, ScoreVector]],
        ctx: SelectionContext,
    ) -> list[Ranked]:
        base_ranked = self.base.rank(scored, ctx)
        preferred = self._preferred_method(ctx.report)
        if preferred is None:
            return base_ranked

        # Override only if the learner's pick is a *viable* candidate here:
        # present, sound (not vetoed), and beating the baseline.
        target = next(
            (
                r
                for r in base_ranked
                if r.candidate.method is preferred
                and r.score.aggregate > 0.0
                and r.score.aggregate != _NEG_INF
            ),
            None,
        )
        if target is None:
            return base_ranked
        return [replace(r, recommended=(r is target)) for r in base_ranked]

    def _preferred_method(self, report: StructureReport) -> MethodKind | None:
        """Distance-weighted vote over nearest neighbors' chosen methods."""
        neighbors = self.store.nearest(extract_features(report), self.k)
        if len(neighbors) < self.min_records:
            return None
        votes: dict[str, float] = {}
        for rec, d in neighbors:
            weight = 1.0 / (1.0 + d)
            votes[rec.chosen] = votes.get(rec.chosen, 0.0) + weight
        best = max(votes, key=lambda m: votes[m])
        try:
            return MethodKind(best)
        except ValueError:
            return None


__all__ = ["InstanceBasedPolicy"]
