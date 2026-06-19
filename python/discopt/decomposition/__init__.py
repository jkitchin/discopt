"""Decomposition methods for structured (MI)NLP.

This package provides decomposition algorithms that exploit problem structure:

- **Benders decomposition** (:mod:`discopt.decomposition.benders`) — for
  block-angular / two-stage problems with complicating variables.
- **Lagrangian relaxation** (:mod:`discopt.decomposition.lagrangian`) — dual
  bounds and primal recovery for problems with coupling constraints.

Both build on the shared structure layer (:mod:`discopt.decomposition.structure`),
which resolves user annotations (``model.first_stage(...)``,
``model.mark_coupling(...)``) and auto-detects block structure.
"""

from __future__ import annotations

from discopt.decomposition.structure import (
    DecompositionStructure as DecompositionStructure,
)
from discopt.decomposition.structure import (
    detect_decomposition as detect_decomposition,
)
from discopt.decomposition.structure import (
    flat_bounds as flat_bounds,
)
from discopt.decomposition.structure import (
    restricted_bounds as restricted_bounds,
)

__all__ = [
    "DecompositionStructure",
    "detect_decomposition",
    "flat_bounds",
    "restricted_bounds",
]
