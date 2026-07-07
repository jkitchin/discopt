"""Reformulation intermediate representation for the Decomposition Advisor.

Phase 5 (design §8): turn a chosen candidate into a solvable
:class:`DecomposedModel` — a master/subproblem partition with a soundness
certificate and a variable mapping — whose ``solve()`` dispatches to the shipping
decomposition drivers. The IR wraps the existing solvers behind one uniform
interface rather than reimplementing them, giving later phases (parallel
execution, nesting) a stable object to build on.
"""

from __future__ import annotations

from discopt.decomposition.ir.certificate import SoundnessCertificate, VariableMapping
from discopt.decomposition.ir.models import MasterModel, SubproblemModel
from discopt.decomposition.ir.reformulation import DecomposedModel, build_decomposition

__all__ = [
    "DecomposedModel",
    "MasterModel",
    "SoundnessCertificate",
    "SubproblemModel",
    "VariableMapping",
    "build_decomposition",
]
