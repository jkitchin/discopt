"""Selection: turn scored candidates into a single recommendation.

Scoring (Phase 3) produces a ranked list; selection decides *which* candidate the
advisor actually recommends. The two are separate on purpose (design §3.6, §7):
the score is a number, but the recommendation applies policy — most importantly
the correctness-first tie-break that prefers a *proven-equivalent* method over a
slightly-higher-scoring *relaxation*, and the rule that nothing is recommended
over the monolith unless it beats the baseline.

The :class:`Policy` protocol is the swappable seam. :class:`RuleBasedPolicy`
encodes the design's decision tree; a learned policy (Phase 7) implements the
same interface and drops in without touching analysis, scoring, or explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from discopt.decomposition.advisor.analyzer import StructureReport
from discopt.decomposition.advisor.scoring import ScoreVector
from discopt.decomposition.advisor.types import Candidate, MethodKind, Soundness

# Preference order among equally-scored candidates: exactness wins ties.
_SOUNDNESS_RANK: dict[Soundness, int] = {
    Soundness.PROVEN_EQUIVALENT: 3,
    Soundness.RELAXATION: 2,
    Soundness.UNKNOWN: 1,
    Soundness.HEURISTIC: 0,
}

# Candidates whose aggregate scores are within this band are treated as a tie and
# broken by soundness (design §7: prefer the exact method when it is nearly as good).
_TIE_EPSILON = 0.15


@dataclass(frozen=True)
class SelectionContext:
    """Context passed to a :class:`Policy` for ranking/selection."""

    report: StructureReport


@dataclass(frozen=True)
class Ranked:
    """A candidate with its score, final rank, and whether it is the pick."""

    candidate: Candidate
    score: ScoreVector
    rank: int
    recommended: bool


@runtime_checkable
class Policy(Protocol):
    """Rank scored candidates and mark the recommended one."""

    def rank(
        self,
        scored: list[tuple[Candidate, ScoreVector]],
        ctx: SelectionContext,
    ) -> list[Ranked]:
        """Order candidates and flag exactly one (or the baseline) as recommended."""
        ...


class RuleBasedPolicy:
    """The default policy: the design's §7 decision tree over scored candidates.

    Rules, in order:

    1. Discard vetoed candidates (score ``-inf``).
    2. The recommendation must *beat the no-decomposition baseline* (aggregate
       ``> 0``); otherwise recommend ``NONE``.
    3. Among candidates within a small score band of the best, prefer the one
       with the strongest soundness (proven-equivalent over relaxation over
       unknown) — correctness-first tie-breaking.
    """

    def rank(
        self,
        scored: list[tuple[Candidate, ScoreVector]],
        ctx: SelectionContext,
    ) -> list[Ranked]:
        # Score order first (already sorted, but be robust).
        ordered = sorted(
            scored,
            key=lambda cs: (-cs[1].aggregate, -cs[1].confidence),
        )

        recommended_idx = self._pick(ordered)
        return [
            Ranked(candidate=c, score=sv, rank=i, recommended=(i == recommended_idx))
            for i, (c, sv) in enumerate(ordered)
        ]

    def _pick(self, ordered: list[tuple[Candidate, ScoreVector]]) -> int:
        """Index of the recommended candidate in *ordered* (score-descending)."""
        # Non-baseline candidates that beat the monolith and are not vetoed.
        viable = [
            (i, c, sv)
            for i, (c, sv) in enumerate(ordered)
            if c.method is not MethodKind.NONE
            and sv.aggregate > 0.0
            and sv.aggregate != float("-inf")
        ]
        if not viable:
            # Nothing beats solving the model as written → recommend NONE.
            for i, (c, _sv) in enumerate(ordered):
                if c.method is MethodKind.NONE:
                    return i
            return 0

        best_score = viable[0][2].aggregate
        # Tie band: candidates nearly as good as the best.
        contenders = [t for t in viable if best_score - t[2].aggregate <= _TIE_EPSILON]
        # Prefer strongest soundness, then higher score, then higher confidence.
        best = max(
            contenders,
            key=lambda t: (
                _SOUNDNESS_RANK.get(t[1].est_soundness, 0),
                t[2].aggregate,
                t[2].confidence,
            ),
        )
        return best[0]


__all__ = [
    "Policy",
    "Ranked",
    "RuleBasedPolicy",
    "SelectionContext",
]
