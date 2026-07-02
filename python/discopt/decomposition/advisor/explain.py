"""Explainability: turn a recommendation into an evidence-backed rationale.

Every recommendation ships a machine-readable *and* human-readable explanation
(design §9), built from the same numbers the scorer used — never a post-hoc
rationalization. The :class:`Explainer` assembles an :class:`Explanation` from the
ranked candidates and the structure report, and renders it as text, markdown, or
JSON. It also answers *counterfactual* "why not method X?" queries (§9.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from discopt.decomposition.advisor.analyzer import StructureReport
from discopt.decomposition.advisor.scoring import ScoreVector
from discopt.decomposition.advisor.selection import (
    Policy,
    Ranked,
    RuleBasedPolicy,
    SelectionContext,
)
from discopt.decomposition.advisor.types import Candidate, MethodKind, Soundness


class Severity(Enum):
    """Severity of a :class:`Concern`."""

    INFO = "info"
    WARN = "warn"
    BLOCKER = "blocker"


@dataclass(frozen=True)
class Rationale:
    """A single evidence-backed reason for the recommendation."""

    claim: str
    evidence: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class Concern:
    """A risk or caveat attached to the recommendation."""

    issue: str
    severity: Severity
    mitigation: str | None = None


@dataclass(frozen=True)
class RankedAlternative:
    """A runner-up method and why it was not chosen."""

    method: MethodKind
    aggregate: float
    reason: str


@dataclass(frozen=True)
class Explanation:
    """A complete, renderable explanation of a recommendation (design §9.1)."""

    recommendation: MethodKind
    headline: str
    reasons: list[Rationale]
    concerns: list[Concern]
    score: ScoreVector | None
    alternatives: list[RankedAlternative]
    confidence: float

    # ── rendering ────────────────────────────────────────────────

    def render(self, fmt: str = "text") -> str:
        """Render the explanation as ``"text"``, ``"markdown"``, or ``"json"``."""
        fmt = fmt.lower()
        if fmt == "json":
            return json.dumps(self.to_dict(), indent=2)
        if fmt == "markdown":
            return self._render_markdown()
        if fmt == "text":
            return self._render_text()
        raise ValueError(f"unknown format {fmt!r}; expected text|markdown|json")

    def to_dict(self) -> dict:
        """Machine-readable form for tooling / the LLM layer."""
        return {
            "recommendation": self.recommendation.value,
            "headline": self.headline,
            "confidence": self.confidence,
            "reasons": [{"claim": r.claim, "evidence": r.evidence} for r in self.reasons],
            "concerns": [
                {"issue": c.issue, "severity": c.severity.value, "mitigation": c.mitigation}
                for c in self.concerns
            ],
            "score": None if self.score is None else self.score.metrics,
            "performance": (
                None
                if self.score is None or self.score.performance is None
                else {
                    "estimated_speedup": self.score.performance.estimated_speedup,
                    "parallel_efficiency": self.score.performance.parallel_efficiency,
                    "num_blocks": self.score.performance.num_blocks,
                    "est_iterations": self.score.performance.est_iterations,
                }
            ),
            "alternatives": [
                {"method": a.method.value, "aggregate": a.aggregate, "reason": a.reason}
                for a in self.alternatives
            ],
        }

    def _render_text(self) -> str:
        lines = [self.headline, "─" * max(len(self.headline), 40)]
        if self.reasons:
            lines.append("Why:")
            for r in self.reasons:
                lines.append(f"  • {r.claim}")
                if r.evidence:
                    ev = ", ".join(f"{k}={v}" for k, v in r.evidence.items())
                    lines.append(f"        evidence: {ev}")
        if self.score is not None and self.score.performance is not None:
            lines.append("Estimated:")
            lines.append(f"  • {self.score.performance.summary()}")
        if self.concerns:
            lines.append("Potential issues:")
            for c in self.concerns:
                mit = f"  (mitigation: {c.mitigation})" if c.mitigation else ""
                lines.append(f"  • [{c.severity.value}] {c.issue}{mit}")
        if self.alternatives:
            lines.append("Alternatives considered:")
            for a in self.alternatives:
                lines.append(f"  • {a.method.label} (score {a.aggregate:+.2f}) — {a.reason}")
        return "\n".join(lines)

    def _render_markdown(self) -> str:
        lines = [f"### {self.headline}", ""]
        if self.reasons:
            lines.append("**Why:**")
            for r in self.reasons:
                ev = (
                    f" _({', '.join(f'{k}={v}' for k, v in r.evidence.items())})_"
                    if r.evidence
                    else ""
                )
                lines.append(f"- {r.claim}{ev}")
            lines.append("")
        if self.score is not None and self.score.performance is not None:
            lines.append(f"**Estimated:** {self.score.performance.summary()}")
            lines.append("")
        if self.concerns:
            lines.append("**Potential issues:**")
            for c in self.concerns:
                mit = f" — _mitigation: {c.mitigation}_" if c.mitigation else ""
                lines.append(f"- `{c.severity.value}` {c.issue}{mit}")
            lines.append("")
        if self.alternatives:
            lines.append("**Alternatives considered:**")
            for a in self.alternatives:
                lines.append(f"- {a.method.label} (score {a.aggregate:+.2f}) — {a.reason}")
        return "\n".join(lines)


class Explainer:
    """Build :class:`Explanation` objects from ranked candidates + a report."""

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or RuleBasedPolicy()

    def explain(
        self,
        scored: list[tuple[Candidate, ScoreVector]],
        report: StructureReport,
    ) -> Explanation:
        """Assemble the explanation for the policy's recommended candidate."""
        ranked = self.policy.rank(scored, SelectionContext(report))
        pick = next((r for r in ranked if r.recommended), ranked[0] if ranked else None)
        if pick is None:
            return Explanation(
                recommendation=MethodKind.NONE,
                headline="Recommended: No decomposition (no candidates)",
                reasons=[],
                concerns=[],
                score=None,
                alternatives=[],
                confidence=1.0,
            )
        cand = pick.candidate
        headline = f"Recommended: {cand.method.label}"
        reasons = self._reasons(cand, pick.score, report)
        concerns = self._concerns(cand, report)
        alternatives = self._alternatives(ranked, pick)
        return Explanation(
            recommendation=cand.method,
            headline=headline,
            reasons=reasons,
            concerns=concerns,
            score=pick.score if cand.method is not MethodKind.NONE else None,
            alternatives=alternatives,
            confidence=pick.score.confidence,
        )

    def _reasons(
        self, cand: Candidate, sv: ScoreVector, report: StructureReport
    ) -> list[Rationale]:
        if cand.method is MethodKind.NONE:
            return [
                Rationale(
                    "No decomposition is estimated to beat solving the model as written.",
                    {
                        "coupling_density": round(report.coupling_density, 3),
                        "blocks": report.num_blocks,
                    },
                )
            ]
        reasons: list[Rationale] = []
        if cand.method in (MethodKind.BENDERS, MethodKind.GENERALIZED_BENDERS):
            reasons.append(
                Rationale(
                    f"Fixing {report.num_integer} integer variable(s) disconnects the "
                    f"model into {report.blocks_after_integer_projection} recourse blocks.",
                    {
                        "integer_localization": sv.metrics.get("integer_localization", 0.0),
                        "components_after_projection": report.blocks_after_integer_projection,
                    },
                )
            )
            recourse = (
                "nonlinear (convex → GBD)" if report.model_is_nonlinear else "linear (Benders)"
            )
            reasons.append(Rationale(f"Recourse subproblems are {recourse}."))
        elif cand.method is MethodKind.INDEPENDENT_BLOCKS:
            reasons.append(
                Rationale(
                    f"The model already splits into {cand.num_blocks} independent blocks "
                    "with no coupling.",
                    {"blocks": cand.num_blocks},
                )
            )
        elif cand.method is MethodKind.LAGRANGIAN:
            n_coupling = (
                len(cand.structure.coupling_constraints) if cand.structure is not None else 0
            )
            reasons.append(
                Rationale(
                    f"{n_coupling} coupling constraint(s) link {cand.num_blocks} "
                    "otherwise-independent blocks.",
                    {"coupling_density": round(report.coupling_density, 3)},
                )
            )
        if report.coupling_density <= 0.1:
            reasons.append(
                Rationale(
                    f"Coupling density is only {report.coupling_density:.1%}.",
                    {"coupling_density": round(report.coupling_density, 3)},
                )
            )
        return reasons

    def _concerns(self, cand: Candidate, report: StructureReport) -> list[Concern]:
        concerns: list[Concern] = []
        if cand.est_soundness is Soundness.UNKNOWN:
            concerns.append(
                Concern(
                    "Exactness depends on convex subproblems; GBD's bound is only rigorous "
                    "if the recourse is convex.",
                    Severity.WARN,
                    mitigation="run convexity detection / OBBT on the recourse; else use "
                    "spatial branch-and-bound inside the subproblem",
                )
            )
        if cand.est_soundness is Soundness.RELAXATION:
            concerns.append(
                Concern(
                    "Yields a valid dual bound, not the exact optimum without primal recovery.",
                    Severity.INFO,
                )
            )
        if cand.method in (MethodKind.BENDERS, MethodKind.GENERALIZED_BENDERS) and (
            report.model_is_nonlinear
        ):
            concerns.append(
                Concern(
                    "Nonlinear recourse may give weaker cuts / slower convergence.",
                    Severity.INFO,
                )
            )
        return concerns

    def _alternatives(self, ranked: list[Ranked], pick: Ranked) -> list[RankedAlternative]:
        out: list[RankedAlternative] = []
        for r in ranked:
            if r is pick:
                continue
            c, sv = r.candidate, r.score
            if sv.aggregate == float("-inf"):
                reason = "ruled out: unsound on this structure"
            elif c.method is MethodKind.NONE:
                reason = "baseline (solve as written)"
            elif sv.aggregate <= 0.0:
                reason = "does not beat the baseline here"
            elif c.est_soundness is not Soundness.PROVEN_EQUIVALENT:
                reason = f"feasible but {c.est_soundness.value}; ranked lower"
            else:
                reason = "lower estimated benefit"
            out.append(RankedAlternative(c.method, sv.aggregate, reason))
        return out

    def explain_counterfactual(
        self,
        method: MethodKind,
        scored: list[tuple[Candidate, ScoreVector]],
        report: StructureReport,
    ) -> str:
        """Answer 'why (not) method X?' for an arbitrary method (design §9.3)."""
        by_method = {c.method: (c, sv) for c, sv in scored}
        if method in by_method:
            c, sv = by_method[method]
            verdict = "recommended" if _is_recommended(method, scored, report) else "considered"
            return (
                f"{method.label} was {verdict}: score {sv.aggregate:+.2f}, "
                f"soundness {c.est_soundness.value}, from {c.provenance}."
            )
        # Not generated — explain the missing precondition.
        return f"{method.label} was not applicable: {_missing_precondition(method, report)}"


def _is_recommended(
    method: MethodKind,
    scored: list[tuple[Candidate, ScoreVector]],
    report: StructureReport,
) -> bool:
    ranked = RuleBasedPolicy().rank(scored, SelectionContext(report))
    pick = next((r for r in ranked if r.recommended), None)
    return pick is not None and pick.candidate.method is method


def _missing_precondition(method: MethodKind, report: StructureReport) -> str:
    """Human explanation of why a method's structural precondition is unmet."""
    if method in (MethodKind.BENDERS, MethodKind.GENERALIZED_BENDERS):
        if report.num_integer == 0:
            return "no integer/binary complicating variables to fix in a master"
        return "fixing the integer variables does not disconnect the model into ≥2 recourse blocks"
    if method is MethodKind.INDEPENDENT_BLOCKS:
        return (
            "the model is not already block-diagonal (removing no constraints, it stays connected)"
        )
    if method is MethodKind.LAGRANGIAN:
        return "no coupling constraints were detected or annotated (mark_coupling)"
    return "this method has no generator yet (planned in a later phase)"


__all__ = [
    "Concern",
    "Explainer",
    "Explanation",
    "RankedAlternative",
    "Rationale",
    "Severity",
]
