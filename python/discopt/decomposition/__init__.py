"""Decomposition methods for structured (MI)NLP.

This package provides decomposition algorithms that exploit problem structure:

- **Benders decomposition** (:mod:`discopt.decomposition.benders`) — for
  block-angular / two-stage problems with complicating variables. Classical
  Benders handles a linear recourse LP; **Generalized Benders** (``solve_gbd``)
  handles a convex nonlinear recourse subproblem and is dispatched automatically
  when the model is nonlinear.
- **Lagrangian relaxation** (:mod:`discopt.decomposition.lagrangian`) — dual
  bounds and primal recovery for problems with coupling constraints.

Both build on the shared structure layer (:mod:`discopt.decomposition.structure`),
which resolves user annotations (``model.first_stage(...)``,
``model.mark_coupling(...)``) and auto-detects block structure.
"""

from __future__ import annotations

from discopt.decomposition.benders import (
    BendersConfig as BendersConfig,
)
from discopt.decomposition.benders import (
    solve_benders as solve_benders,
)
from discopt.decomposition.benders import (
    solve_gbd as solve_gbd,
)
from discopt.decomposition.lagrangian import (
    LagrangianConfig as LagrangianConfig,
)
from discopt.decomposition.lagrangian import (
    solve_lagrangian as solve_lagrangian,
)
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
    "BendersConfig",
    "DecompositionStructure",
    "LagrangianConfig",
    "detect_decomposition",
    "flat_bounds",
    "restricted_bounds",
    "solve_benders",
    "solve_gbd",
    "solve_lagrangian",
]
