"""The Decomposition Advisor — analysis, candidates, and (later) recommendation.

Phase 2 of the advisor (see ``docs/design/decomposition-advisor.md``): automatic
block detection. Given a model, the advisor analyzes its structure
(:class:`StructureReport`), discovers candidate decompositions
(:class:`Candidate` via the generator cascade), and exposes them through the
:class:`DecompositionAdvisor` façade — while reusing the Phase 1 graph layer and
the existing :func:`~discopt.decomposition.structure.detect_decomposition`
contract.

Scoring, recommendation, and automatic reformulation arrive in later phases
behind the same façade.
"""

from __future__ import annotations

from discopt.decomposition.advisor.advisor import (
    DecompositionAdvisor,
    analyze_decomposition,
)
from discopt.decomposition.advisor.analyzer import StructureAnalyzer, StructureReport
from discopt.decomposition.advisor.candidates import (
    DEFAULT_GENERATORS,
    CandidateGenerator,
    generate_candidates,
)
from discopt.decomposition.advisor.explain import (
    Concern,
    Explainer,
    Explanation,
    RankedAlternative,
    Rationale,
    Severity,
)
from discopt.decomposition.advisor.scoring import (
    PerformanceEstimate,
    Scorer,
    ScoreVector,
    ScoringWeights,
)
from discopt.decomposition.advisor.selection import (
    Policy,
    Ranked,
    RuleBasedPolicy,
    SelectionContext,
)
from discopt.decomposition.advisor.types import Candidate, MethodKind, Soundness

__all__ = [
    "DEFAULT_GENERATORS",
    "Candidate",
    "CandidateGenerator",
    "Concern",
    "DecompositionAdvisor",
    "Explainer",
    "Explanation",
    "MethodKind",
    "PerformanceEstimate",
    "Policy",
    "Ranked",
    "RankedAlternative",
    "Rationale",
    "RuleBasedPolicy",
    "ScoreVector",
    "Scorer",
    "ScoringWeights",
    "SelectionContext",
    "Severity",
    "Soundness",
    "StructureAnalyzer",
    "StructureReport",
    "analyze_decomposition",
    "generate_candidates",
]
