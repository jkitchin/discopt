"""Core value types for the Decomposition Advisor.

These are the immutable artifacts that flow between the advisor's stages
(analyze → generate candidates → score → recommend). Phase 2 introduces the
first three: :class:`MethodKind`, :class:`Soundness`, :class:`StructureReport`,
and :class:`Candidate`. Scoring (``ScoreVector``) and explanation
(``Explanation``) arrive in later phases behind the same package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from discopt.decomposition.structure import DecompositionStructure


class MethodKind(Enum):
    """A decomposition method the advisor can recommend.

    Only ``NONE``, ``INDEPENDENT_BLOCKS``, ``BENDERS``, ``GENERALIZED_BENDERS``
    and ``LAGRANGIAN`` are *generated* today (they map to shipping solvers). The
    remaining members are registered so downstream code (scoring, selection,
    reformulation) has stable identifiers as those methods come online.
    """

    NONE = "none"
    INDEPENDENT_BLOCKS = "independent_blocks"
    BENDERS = "benders"
    GENERALIZED_BENDERS = "generalized_benders"
    OUTER_APPROXIMATION = "outer_approximation"
    LAGRANGIAN = "lagrangian"
    # Registered for later phases (see design doc §5):
    DANTZIG_WOLFE = "dantzig_wolfe"
    COLUMN_GENERATION = "column_generation"
    ADMM = "admm"
    SCHUR = "schur"
    PROGRESSIVE_HEDGING = "progressive_hedging"
    NESTED_BENDERS = "nested_benders"

    @property
    def label(self) -> str:
        """Human-readable name for explanations and summaries."""
        return {
            MethodKind.NONE: "No decomposition",
            MethodKind.INDEPENDENT_BLOCKS: "Independent blocks",
            MethodKind.BENDERS: "Classical Benders",
            MethodKind.GENERALIZED_BENDERS: "Generalized Benders",
            MethodKind.OUTER_APPROXIMATION: "Outer approximation",
            MethodKind.LAGRANGIAN: "Lagrangian relaxation",
            MethodKind.DANTZIG_WOLFE: "Dantzig-Wolfe",
            MethodKind.COLUMN_GENERATION: "Column generation",
            MethodKind.ADMM: "ADMM",
            MethodKind.SCHUR: "Schur complement",
            MethodKind.PROGRESSIVE_HEDGING: "Progressive hedging",
            MethodKind.NESTED_BENDERS: "Nested Benders",
        }[self]


class Soundness(Enum):
    """How faithfully a candidate's reformulation represents the original model.

    The advisor's correctness-first stance (design goal #1) uses this as a
    gatekeeper: a candidate that is not at least a valid relaxation is never
    silently applied.
    """

    PROVEN_EQUIVALENT = "proven_equivalent"
    """Reformulation has the same optimal value as the monolith (e.g. classical
    Benders on a MILP, independent blocks of a separable model)."""

    RELAXATION = "relaxation"
    """Reformulation yields a valid bound (e.g. Lagrangian dual), not necessarily
    the exact optimum without further work."""

    UNKNOWN = "unknown"
    """Equivalence depends on a property not yet checked (e.g. GBD is exact only
    on a convex model; convexity is resolved during scoring)."""

    HEURISTIC = "heuristic"
    """No equivalence guarantee; use only as a search accelerator."""


@dataclass(frozen=True)
class ElementView:
    """The element-level partition behind a candidate the array view cannot see.

    Attached by the element generators (``candidates.py``) when a decomposition
    exists only element by element -- e.g. scenarios sharing one array variable.
    The scorer and the IR read their block sizes, separator and coupling from
    here instead of from the array-level graph, which for such a candidate
    reports one block. ``None`` on every array-level candidate.

    Attributes
    ----------
    block_columns : tuple[tuple[int, ...], ...]
        Flat columns of each independent block (constraint-bearing blocks only).
    block_labels : tuple[tuple[str, ...], ...]
        Element labels (``x[0, 3]``) of the same columns, for the IR.
    separator_fraction : float
        Master (complicating) columns over all columns; ``0`` when nothing is fixed.
    coupling_density : float
        Dualized constraints over all constraints; ``0`` when nothing is dualized.
    """

    block_columns: tuple[tuple[int, ...], ...]
    block_labels: tuple[tuple[str, ...], ...]
    separator_fraction: float = 0.0
    coupling_density: float = 0.0

    @property
    def num_blocks(self) -> int:
        return len(self.block_columns)


@dataclass(frozen=True)
class Candidate:
    """A proposed decomposition: a method applied to a discovered structure.

    Candidates are *unranked* in Phase 2 — scoring and selection order them in
    later phases. ``structure`` reuses the existing
    :class:`~discopt.decomposition.structure.DecompositionStructure` contract, so
    a candidate feeds the shipping ``solve_benders`` / ``solve_lagrangian``
    drivers with no adapter.

    Attributes
    ----------
    method : MethodKind
        The decomposition method this candidate applies.
    structure : DecompositionStructure | None
        The partition (blocks, complicating vars, coupling rows). ``None`` for
        the monolithic ``NONE`` candidate.
    provenance : str
        Which generator/algorithm produced this candidate, for diagnostics.
    est_soundness : Soundness
        Best-effort soundness class before scoring refines it.
    notes : tuple[str, ...]
        Short human-readable reasons/caveats; seed material for the Phase 4
        explanation.
    element_view : ElementView | None
        The element-level partition, for a candidate only the element graph
        reveals; ``None`` otherwise.
    """

    method: MethodKind
    structure: DecompositionStructure | None
    provenance: str
    est_soundness: Soundness
    notes: tuple[str, ...] = field(default_factory=tuple)
    element_view: ElementView | None = None

    @property
    def num_blocks(self) -> int:
        """Number of blocks in the candidate's partition (0 if monolithic)."""
        if self.element_view is not None:
            return self.element_view.num_blocks
        return self.structure.num_blocks if self.structure is not None else 0

    def summary(self) -> str:
        """One-line summary of the candidate."""
        extra = f", {self.num_blocks} blocks" if self.structure is not None else ""
        return f"{self.method.label} [{self.est_soundness.value}{extra}] — {self.provenance}"


__all__ = ["Candidate", "ElementView", "MethodKind", "Soundness"]
