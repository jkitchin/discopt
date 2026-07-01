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

from discopt.decomposition.advisor import (
    Candidate as Candidate,
)
from discopt.decomposition.advisor import (
    DecompositionAdvisor as DecompositionAdvisor,
)
from discopt.decomposition.advisor import (
    Explanation as Explanation,
)
from discopt.decomposition.advisor import (
    MethodKind as MethodKind,
)
from discopt.decomposition.advisor import (
    ScoreVector as ScoreVector,
)
from discopt.decomposition.advisor import (
    Soundness as Soundness,
)
from discopt.decomposition.advisor import (
    StructureReport as StructureReport,
)
from discopt.decomposition.advisor import (
    analyze_decomposition as analyze_decomposition,
)
from discopt.decomposition.benders import (
    BendersConfig as BendersConfig,
)
from discopt.decomposition.benders import (
    solve_benders as solve_benders,
)
from discopt.decomposition.benders import (
    solve_gbd as solve_gbd,
)
from discopt.decomposition.ir import (
    DecomposedModel as DecomposedModel,
)
from discopt.decomposition.ir import (
    SoundnessCertificate as SoundnessCertificate,
)
from discopt.decomposition.ir import (
    build_decomposition as build_decomposition,
)
from discopt.decomposition.lagrangian import (
    LagrangianConfig as LagrangianConfig,
)
from discopt.decomposition.lagrangian import (
    solve_lagrangian as solve_lagrangian,
)
from discopt.decomposition.learning import (
    InstanceBasedPolicy as InstanceBasedPolicy,
)
from discopt.decomposition.learning import (
    RecordStore as RecordStore,
)
from discopt.decomposition.learning import (
    SolveRecord as SolveRecord,
)
from discopt.decomposition.learning import (
    record_outcome as record_outcome,
)
from discopt.decomposition.parallel import (
    SchedulingGraph as SchedulingGraph,
)
from discopt.decomposition.parallel import (
    SequentialComm as SequentialComm,
)
from discopt.decomposition.parallel import (
    ThreadPoolComm as ThreadPoolComm,
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
    "Candidate",
    "DecompositionAdvisor",
    "DecompositionStructure",
    "DecomposedModel",
    "Explanation",
    "InstanceBasedPolicy",
    "LagrangianConfig",
    "MethodKind",
    "RecordStore",
    "ScoreVector",
    "SchedulingGraph",
    "SequentialComm",
    "SolveRecord",
    "Soundness",
    "SoundnessCertificate",
    "StructureReport",
    "ThreadPoolComm",
    "analyze_decomposition",
    "build_decomposition",
    "detect_decomposition",
    "flat_bounds",
    "record_outcome",
    "restricted_bounds",
    "solve_benders",
    "solve_gbd",
    "solve_lagrangian",
]
