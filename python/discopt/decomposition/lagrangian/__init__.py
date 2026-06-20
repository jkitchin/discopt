"""Lagrangian relaxation for problems with coupling constraints.

Dualizing the coupling (linking) constraints decouples the model into
independent blocks and yields a **dual lower bound** that, for problems whose
blocks lack the integrality property, dominates the LP relaxation. The dual is
maximized by a subgradient method or a cutting-plane (bundle-style) method, and
a Lagrangian heuristic recovers a feasible primal incumbent.

v1 targets linear (mixed-integer) models; the recovered ``bound`` is a rigorous
lower bound. Dantzig-Wolfe / column generation / branch-and-price are planned.
"""

from __future__ import annotations

from discopt.decomposition.lagrangian.node_bounder import (
    LagrangianNodeBounder as LagrangianNodeBounder,
)
from discopt.decomposition.lagrangian.solver import (
    LagrangianConfig as LagrangianConfig,
)
from discopt.decomposition.lagrangian.solver import (
    solve_lagrangian as solve_lagrangian,
)

__all__ = ["LagrangianConfig", "LagrangianNodeBounder", "solve_lagrangian"]
