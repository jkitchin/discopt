"""Master and subproblem descriptors for the reformulation IR.

Light, immutable *views* of a decomposition's partition (design §8.2) — which
variables coordinate (master) and which independent block each remaining variable
belongs to. These describe the reformulation for inspection and reporting; the
actual coordinated solve is delegated to the shipping drivers by
:class:`~discopt.decomposition.ir.reformulation.DecomposedModel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from discopt.decomposition.advisor.types import MethodKind


@dataclass(frozen=True)
class SubproblemModel:
    """One independent block of a decomposition."""

    block_id: int
    variables: tuple[str, ...]
    constraint_indices: tuple[int, ...] = field(default_factory=tuple)

    @property
    def size(self) -> int:
        """Number of variables in the block."""
        return len(self.variables)

    def summary(self) -> str:
        """One-line human-readable descriptor."""
        return f"subproblem[{self.block_id}]: {self.size} vars, {len(self.constraint_indices)} cons"


@dataclass(frozen=True)
class MasterModel:
    """The coordinating problem: complicating vars and/or dualized coupling rows."""

    method: MethodKind
    variables: tuple[str, ...]
    coupling_constraints: tuple[int, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """One-line human-readable descriptor."""
        return (
            f"master ({self.method.label}): {len(self.variables)} complicating vars, "
            f"{len(self.coupling_constraints)} coupling constraint(s)"
        )


__all__ = ["MasterModel", "SubproblemModel"]
