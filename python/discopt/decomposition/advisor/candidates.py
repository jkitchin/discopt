"""Candidate generation: map discovered structure to decomposition candidates.

Each :class:`CandidateGenerator` inspects the :class:`StructureReport` and, when
applicable, proposes one or more :class:`Candidate` decompositions. Generators
are the extensibility seam of Phase 2: a new method is a new generator plus (in
a later phase) a scoring rule, touching no analysis code.

The generators run cheap-first (the cascade of design doc §5.1) and their output
is de-duplicated. Ranking is *not* done here — every applicable candidate is
returned, including the monolithic ``NONE`` baseline, for scoring/selection to
order later.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from discopt.decomposition.advisor.analyzer import StructureReport
from discopt.decomposition.advisor.types import Candidate, MethodKind, Soundness
from discopt.decomposition.structure import DecompositionStructure, detect_decomposition


@runtime_checkable
class CandidateGenerator(Protocol):
    """Proposes decomposition candidates from a model and its structure report."""

    name: str

    def applicable(self, report: StructureReport) -> bool:
        """Cheap gate: could this generator produce anything for *report*?"""
        ...

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        """Produce candidates (possibly empty) for *model*."""
        ...


class MonolithicGenerator:
    """Always proposes the no-decomposition baseline.

    Keeping ``NONE`` in the candidate set guarantees the ranker always has the
    honest competitor to beat (design doc §6.3): a decomposition must be *better*
    than solving the monolith, not merely feasible.
    """

    name = "monolithic"

    def applicable(self, report: StructureReport) -> bool:
        return True

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        return [
            Candidate(
                method=MethodKind.NONE,
                structure=None,
                provenance="baseline",
                est_soundness=Soundness.PROVEN_EQUIVALENT,
                notes=("solve the model as written",),
            )
        ]


class IndependentBlockGenerator:
    """Block-diagonal models: solve the independent blocks with no coordination."""

    name = "independent-blocks"

    def applicable(self, report: StructureReport) -> bool:
        return report.is_block_diagonal and not report.coupling_constraints

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        structure = detect_decomposition(model)
        return [
            Candidate(
                method=MethodKind.INDEPENDENT_BLOCKS,
                structure=structure,
                provenance=f"connected components ({structure.num_blocks} blocks)",
                est_soundness=Soundness.PROVEN_EQUIVALENT,
                notes=(
                    f"{structure.num_blocks} independent blocks, no coupling",
                    "exact if the objective is separable across blocks",
                ),
            )
        ]


class IntegerProjectionGenerator:
    """Fixing the integer variables exposes independent recourse → Benders.

    This generalizes the shipping default (complicating vars = integer/binary
    vars): it only fires when *removing* those variables actually disconnects the
    model into ≥2 recourse blocks, which is the structural precondition for
    Benders paying off. Nonlinear recourse selects Generalized Benders.
    """

    name = "integer-projection"

    def applicable(self, report: StructureReport) -> bool:
        return report.integer_localizes

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        structure = detect_decomposition(model)
        if not structure.complicating_vars:
            return []
        if report.model_is_nonlinear:
            method = MethodKind.GENERALIZED_BENDERS
            soundness = Soundness.UNKNOWN
            notes = (
                f"fixing {len(structure.complicating_vars)} integer vars exposes "
                f"{report.blocks_after_integer_projection} recourse blocks",
                "nonlinear recourse → GBD; exact only if subproblems are convex "
                "(checked during scoring)",
            )
        else:
            method = MethodKind.BENDERS
            soundness = Soundness.PROVEN_EQUIVALENT
            notes = (
                f"fixing {len(structure.complicating_vars)} integer vars exposes "
                f"{report.blocks_after_integer_projection} recourse blocks",
                "linear recourse → classical Benders (exact)",
            )
        return [
            Candidate(
                method=method,
                structure=structure,
                provenance="integer-variable projection",
                est_soundness=soundness,
                notes=notes,
            )
        ]


class CouplingGenerator:
    """A few coupling constraints link otherwise-independent blocks → Lagrangian.

    Dualizing the coupling rows separates the model into blocks solvable in
    parallel and yields a valid dual bound. Fires on annotated or bridge-detected
    coupling.
    """

    name = "coupling"

    def applicable(self, report: StructureReport) -> bool:
        return bool(report.coupling_constraints)

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        structure = detect_decomposition(model)
        if not structure.coupling_constraints:
            return []
        provenance = "user annotations" if report.has_annotations else "bridge-constraint scan"
        return [
            Candidate(
                method=MethodKind.LAGRANGIAN,
                structure=structure,
                provenance=provenance,
                est_soundness=Soundness.RELAXATION,
                notes=(
                    f"{len(structure.coupling_constraints)} coupling constraint(s) "
                    f"dualized; {structure.num_blocks} blocks",
                    "yields a valid dual bound; primal recovery may need extra work",
                ),
            )
        ]


#: Default generator cascade, cheap-first (design doc §5.1).
DEFAULT_GENERATORS: tuple[CandidateGenerator, ...] = (
    MonolithicGenerator(),
    IndependentBlockGenerator(),
    IntegerProjectionGenerator(),
    CouplingGenerator(),
)


def _dedup_key(cand: Candidate) -> tuple:
    """Identity of a candidate for de-duplication: method + partition signature."""
    s: DecompositionStructure | None = cand.structure
    if s is None:
        return (cand.method,)
    return (
        cand.method,
        tuple(sorted(s.complicating_vars)),
        tuple(sorted(s.coupling_constraints)),
    )


def generate_candidates(
    model,
    report: StructureReport,
    generators: tuple[CandidateGenerator, ...] = DEFAULT_GENERATORS,
) -> list[Candidate]:
    """Run the generator cascade and return de-duplicated candidates.

    Order follows the cascade (monolithic baseline first, then cheap structural
    detectors); duplicates from different generators collapse to the
    first-produced one. No ranking is applied.
    """
    out: list[Candidate] = []
    seen: set[tuple] = set()
    for gen in generators:
        if not gen.applicable(report):
            continue
        for cand in gen.generate(model, report):
            key = _dedup_key(cand)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


__all__ = [
    "DEFAULT_GENERATORS",
    "CandidateGenerator",
    "CouplingGenerator",
    "IndependentBlockGenerator",
    "IntegerProjectionGenerator",
    "MonolithicGenerator",
    "generate_candidates",
]
