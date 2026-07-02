"""Soundness certificates and variable mappings for reformulations.

The correctness spine of the reformulation IR (design §8.2). A
:class:`SoundnessCertificate` records *why* a decomposition represents the
original model — exactly, as a relaxation, or not at all — and refuses to run a
decomposition that cannot be justified. A :class:`VariableMapping` records which
original variables live in the master versus which subproblem block, so results
come back in the user's variable space and the partition is inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from discopt.decomposition.advisor.types import MethodKind, Soundness


@dataclass(frozen=True)
class VariableMapping:
    """Original-variable ↔ (master | subproblem-block) mapping.

    Attributes
    ----------
    master_vars : tuple[str, ...]
        Variables held in the coordinating master (complicating vars for Benders;
        empty for Lagrangian / independent-block decompositions).
    block_of_var : dict[str, int]
        Subproblem block id per variable; ``-1`` for a master variable.
    num_blocks : int
        Number of subproblem blocks.
    """

    master_vars: tuple[str, ...]
    block_of_var: dict[str, int]
    num_blocks: int

    def role(self, var: str) -> str:
        """Human label for a variable's role: ``"master"`` or ``"block{i}"``."""
        if var in self.master_vars:
            return "master"
        b = self.block_of_var.get(var, -1)
        return f"block{b}" if b >= 0 else "master"


@dataclass(frozen=True)
class SoundnessCertificate:
    """Why a reformulation is (or is not) a faithful stand-in for the monolith.

    Attributes
    ----------
    method : MethodKind
        The decomposition method certified.
    level : Soundness
        ``PROVEN_EQUIVALENT`` (same optimum), ``RELAXATION`` (valid bound),
        ``UNKNOWN`` (property-dependent, e.g. GBD needs convexity), or
        ``HEURISTIC`` (no guarantee).
    rationale : str
        One-line justification.
    caveats : tuple[str, ...]
        Conditions the guarantee depends on.
    """

    method: MethodKind
    level: Soundness
    rationale: str
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def is_sound(self) -> bool:
        """True unless the certificate is merely heuristic.

        A relaxation (valid bound) and an unknown-but-not-refuted certificate are
        both allowed to run; only a purely heuristic reformulation is refused by
        default (correctness-first, design goal #1).
        """
        return self.level is not Soundness.HEURISTIC

    def assert_sound(self) -> None:
        """Raise :class:`ValueError` if the reformulation is unsound to run."""
        if not self.is_sound():
            raise ValueError(
                f"refusing to build a {self.method.label} reformulation: {self.rationale}"
            )

    def summary(self) -> str:
        """One-line human-readable certificate."""
        caveat = f" (caveats: {'; '.join(self.caveats)})" if self.caveats else ""
        return f"{self.method.label}: {self.level.value} — {self.rationale}{caveat}"


__all__ = ["SoundnessCertificate", "VariableMapping"]
