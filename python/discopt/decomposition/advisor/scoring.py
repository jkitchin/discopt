"""Scoring: rank decomposition candidates by estimated benefit.

Given the candidates from Phase 2, the scorer produces a total order. Following
the design (§6), scoring is deliberately *two-stage and method-aware*:

1. **Gatekeepers (veto, not weigh).** A candidate that cannot be sound on this
   structure is removed (score ``-inf``), never merely down-weighted — the
   correctness-first stance (design goal #1).
2. **Benefit − cost, versus a no-decomposition baseline.** The monolithic
   ``NONE`` candidate anchors the scale at ``0``; a decomposition must *beat*
   solving the model as written, not merely be feasible. Dense coupling collapses
   the benefit below the baseline so the advisor says "don't".

Every score carries a **confidence** and an analytic :class:`PerformanceEstimate`
(speedup, parallel efficiency, iteration count). These are honest first-cut
estimates — the predicted-vs-observed logging of Phase 7 is what will calibrate
the weights and confidences over time. They should be read as *ranking signals*,
not promises.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from discopt.decomposition.advisor.analyzer import StructureReport
from discopt.decomposition.advisor.types import Candidate, MethodKind, Soundness
from discopt.decomposition.graph import kernels
from discopt.decomposition.graph.base import ModelGraph, build_graph

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class ScoringWeights:
    """Weights and thresholds for the scalarized score (design §6.3).

    Defaults are hand-tuned first-cut values; they live here (not hard-coded in
    the scorer) so a config or a learned policy (Phase 7) can override them.
    """

    w_parallel: float = 1.0
    """Weight on log-speedup (the dominant benefit term)."""
    w_localize: float = 0.5
    """Weight on how well the hard part (integers / nonlinearity) is isolated."""
    w_cuts: float = 0.3
    """Weight on expected cut / bound strength."""
    w_comm: float = 0.5
    """Weight (penalty) on estimated communication cost."""
    w_warm: float = 0.2
    """Weight (penalty) on poor warm-start / load balance."""
    tau_couple: float = 0.35
    """Coupling-density above which decomposition is penalized below baseline."""
    penalty: float = 1.0
    """Size of the sub-baseline penalty for dense coupling."""


@dataclass(frozen=True)
class PerformanceEstimate:
    """Analytic performance prediction for a candidate (design §6.4).

    An Amdahl-style model: the coupling is the serial fraction, and load
    imbalance across blocks caps the effective parallelism.
    """

    num_blocks: int
    """Effective parallel blocks for this method (recourse blocks for Benders,
    dualized blocks for Lagrangian, components for independent blocks)."""
    effective_parallelism: float
    """total_work / max_block_work — the speedup cap from load balance."""
    parallel_efficiency: float
    """effective_parallelism / num_blocks, in ``(0, 1]`` (1.0 = balanced)."""
    estimated_speedup: float
    """Amdahl speedup versus the monolith (``≥ 1``)."""
    est_iterations: int
    """Rough coordination iteration count (method-specific)."""
    confidence: float
    """How much to trust this estimate, in ``[0, 1]``."""

    def summary(self) -> str:
        """One-line human-readable estimate."""
        return (
            f"~{self.estimated_speedup:.1f}x speedup, "
            f"{self.parallel_efficiency:.0%} parallel efficiency "
            f"over {self.num_blocks} blocks, ~{self.est_iterations} iters "
            f"[confidence {self.confidence:.2f}]"
        )


@dataclass(frozen=True)
class ScoreVector:
    """A candidate's scored metrics plus its scalar rank key.

    Attributes
    ----------
    metrics : dict[str, float]
        Named metric values (design §6.2), for explanation and debugging.
    aggregate : float
        The scalar rank key; higher is better. ``0.0`` is the no-decomposition
        baseline, ``-inf`` a vetoed (unsound) candidate.
    confidence : float
        Confidence in the aggregate, in ``[0, 1]``.
    performance : PerformanceEstimate | None
        Analytic speedup/efficiency estimate (``None`` for the monolith).
    """

    metrics: dict[str, float] = field(default_factory=dict)
    aggregate: float = 0.0
    confidence: float = 1.0
    performance: PerformanceEstimate | None = None

    def summary(self) -> str:
        """One-line human-readable score."""
        perf = f" — {self.performance.summary()}" if self.performance else ""
        return f"score={self.aggregate:+.2f} (confidence {self.confidence:.2f}){perf}"


# Per-method constants: expected cut/bound strength and base confidence. These
# encode domain knowledge (linear Benders cuts are strong; GBD depends on
# convexity; Lagrangian gives a dual bound) until learning refines them.
_CUT_STRENGTH: dict[MethodKind, float] = {
    MethodKind.INDEPENDENT_BLOCKS: 1.0,
    MethodKind.BENDERS: 0.9,
    MethodKind.GENERALIZED_BENDERS: 0.6,
    MethodKind.LAGRANGIAN: 0.5,
}
_BASE_CONFIDENCE: dict[MethodKind, float] = {
    MethodKind.NONE: 1.0,
    MethodKind.INDEPENDENT_BLOCKS: 0.9,
    MethodKind.BENDERS: 0.75,
    MethodKind.GENERALIZED_BENDERS: 0.55,
    MethodKind.LAGRANGIAN: 0.6,
}
_FIXES_COMPLICATING = (MethodKind.BENDERS, MethodKind.GENERALIZED_BENDERS)


def _effective_block_sizes(cand: Candidate, graph: ModelGraph) -> list[int]:
    """Variable-count of each parallel block, appropriate to the method.

    - Benders/GBD: recourse components after *removing the complicating vars*
      (that is where the parallelism lives once the master fixes them).
    - Lagrangian / independent blocks: the partition's blocks directly.
    """
    s = cand.structure
    if s is None:
        return []
    if cand.method in _FIXES_COMPLICATING:
        name_to_idx = {nm: i for i, nm in enumerate(graph.var_names)}
        remove = {name_to_idx[nm] for nm in s.complicating_vars if nm in name_to_idx}
        projected = [[j for j in clique if j not in remove] for clique in graph.constraint_cliques]
        block_of, _ = kernels.connected_components(graph.num_vars, projected)
        bearing = {block_of[c[0]] for c in projected if c}
        sizes: Counter[int] = Counter()
        for j in range(graph.num_vars):
            if j in remove:
                continue
            b = block_of[j]
            if b in bearing:
                sizes[b] += 1
        return sorted(sizes.values(), reverse=True)
    return sorted((len(b) for b in s.blocks), reverse=True)


def _performance_estimate(
    cand: Candidate, report: StructureReport, sizes: list[int]
) -> PerformanceEstimate:
    """Amdahl-style speedup / efficiency from block sizes and coupling."""
    b = len(sizes)
    base_conf = _BASE_CONFIDENCE.get(cand.method, 0.5)
    if b <= 1:
        # No exploitable parallelism.
        return PerformanceEstimate(
            num_blocks=b,
            effective_parallelism=1.0,
            parallel_efficiency=1.0,
            estimated_speedup=1.0,
            est_iterations=1,
            confidence=base_conf,
        )
    total = float(sum(sizes))
    mx = float(max(sizes))
    p_eff = total / mx  # speedup cap from load balance (== b when balanced)
    parallel_efficiency = p_eff / b

    # Serial coordination fraction: none for independent blocks, else driven by
    # coupling density (with a small floor for the master solve).
    if cand.method is MethodKind.INDEPENDENT_BLOCKS:
        f_coord = 0.0
    else:
        f_coord = min(0.9, max(0.02, report.coupling_density))
    speedup = 1.0 / (f_coord + (1.0 - f_coord) / p_eff)
    speedup = max(1.0, speedup)

    # Rough method-specific iteration count.
    if cand.method is MethodKind.INDEPENDENT_BLOCKS:
        iters = 1
    elif cand.method in _FIXES_COMPLICATING:
        n_comp = len(cand.structure.complicating_vars) if cand.structure else 0
        iters = min(200, 5 + 2 * n_comp)
    else:  # Lagrangian / other coordination
        iters = 50

    # Extrapolation lowers confidence when blocks are very imbalanced.
    conf = base_conf * (0.5 + 0.5 * parallel_efficiency)
    return PerformanceEstimate(
        num_blocks=b,
        effective_parallelism=p_eff,
        parallel_efficiency=parallel_efficiency,
        estimated_speedup=speedup,
        est_iterations=iters,
        confidence=round(conf, 3),
    )


def _block_size_cv(sizes: list[int]) -> float:
    """Coefficient of variation of block sizes (0 = perfectly balanced)."""
    if len(sizes) <= 1:
        return 0.0
    mean = sum(sizes) / len(sizes)
    if mean == 0:
        return 0.0
    var = sum((s - mean) ** 2 for s in sizes) / len(sizes)
    return math.sqrt(var) / mean


class Scorer:
    """Score and rank decomposition candidates (design §6)."""

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score(
        self,
        model,
        cand: Candidate,
        report: StructureReport,
        graph: ModelGraph | None = None,
    ) -> ScoreVector:
        """Score a single candidate into a :class:`ScoreVector`."""
        w = self.weights

        # The monolithic baseline anchors the scale at 0.
        if cand.method is MethodKind.NONE:
            return ScoreVector(
                metrics={"coupling_density": report.coupling_density},
                aggregate=0.0,
                confidence=1.0,
                performance=None,
            )

        # Gatekeeper: a candidate known to be unsound is vetoed outright.
        if cand.est_soundness is Soundness.HEURISTIC:
            return ScoreVector(
                metrics={"veto": 1.0},
                aggregate=_NEG_INF,
                confidence=1.0,
                performance=None,
            )

        if graph is None:
            graph = build_graph(model)
        sizes = _effective_block_sizes(cand, graph)
        perf = _performance_estimate(cand, report, sizes)
        cv = _block_size_cv(sizes)

        localization = self._localization(cand, report)
        cut_strength = _CUT_STRENGTH.get(cand.method, 0.5)
        # Communication cost proxy: separator size relative to the model.
        if cand.method in _FIXES_COMPLICATING and cand.structure is not None:
            sep = len(cand.structure.complicating_vars)
            comm_norm = sep / report.num_vars if report.num_vars else 0.0
        else:
            comm_norm = report.coupling_density
        warm = perf.parallel_efficiency  # balanced blocks warm-start best

        metrics = {
            "num_blocks": float(perf.num_blocks),
            "block_size_cv": round(cv, 4),
            "coupling_density": round(report.coupling_density, 4),
            "parallel_efficiency": round(perf.parallel_efficiency, 4),
            "estimated_speedup": round(perf.estimated_speedup, 4),
            "integer_localization": round(localization, 4),
            "expected_cut_strength": cut_strength,
            "est_communication_cost": round(comm_norm, 4),
            "warm_start_potential": round(warm, 4),
        }

        # Dense coupling ⇒ decomposition does not pay: drop below the baseline.
        if report.coupling_density > w.tau_couple:
            return ScoreVector(
                metrics=metrics,
                aggregate=-w.penalty,
                confidence=0.9,
                performance=perf,
            )

        benefit = (
            w.w_parallel * math.log2(max(perf.estimated_speedup, 1.0))
            + w.w_localize * localization
            + w.w_cuts * cut_strength
        )
        cost = w.w_comm * comm_norm + w.w_warm * (1.0 - warm)
        aggregate = benefit - cost

        # UNKNOWN soundness (e.g. GBD pending convexity) trims confidence.
        conf = perf.confidence
        if cand.est_soundness is Soundness.UNKNOWN:
            conf *= 0.8
        elif cand.est_soundness is Soundness.RELAXATION:
            conf *= 0.9

        return ScoreVector(
            metrics=metrics,
            aggregate=round(aggregate, 4),
            confidence=round(conf, 3),
            performance=perf,
        )

    def _localization(self, cand: Candidate, report: StructureReport) -> float:
        """How well the method isolates the hard part of the model."""
        if cand.method is MethodKind.INDEPENDENT_BLOCKS:
            return 1.0
        if cand.method in _FIXES_COMPLICATING:
            return 1.0 if report.integer_localizes else 0.5
        # Lagrangian: more balanced blocks localize better.
        if cand.structure is not None and cand.structure.num_blocks > 1:
            return 0.6
        return 0.4

    def score_all(
        self,
        model,
        candidates: list[Candidate],
        report: StructureReport,
        graph: ModelGraph | None = None,
    ) -> list[tuple[Candidate, ScoreVector]]:
        """Score every candidate and return them ranked best-first.

        Ties break by confidence, then method name, for determinism. The graph is
        built once and shared across candidates.
        """
        if graph is None:
            graph = build_graph(model)
        scored = [(c, self.score(model, c, report, graph)) for c in candidates]
        scored.sort(key=lambda cs: (-cs[1].aggregate, -cs[1].confidence, cs[0].method.value))
        return scored


__all__ = [
    "PerformanceEstimate",
    "ScoreVector",
    "Scorer",
    "ScoringWeights",
]
