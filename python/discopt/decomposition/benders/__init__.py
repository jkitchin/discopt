"""Benders decomposition for block-angular / two-stage (mixed-integer) LPs.

The master problem holds the *complicating* (first-stage) variables; for a
fixed master point the recourse subproblem is a linear program whose optimal
dual yields a Benders **optimality cut**, and whose infeasibility yields a
**feasibility cut**. Iterating tightens the master's lower bound until it meets
the incumbent's upper bound.

**Classical Benders** handles linear constraints and a linear objective, with
all integer variables first-stage (so the recourse is a continuous LP). Every
cut is a valid global underestimator (LP weak duality), so the master objective
is a sound lower bound.

**Generalized Benders Decomposition** (``gbd.solve_gbd``, Geoffrion 1972)
extends this to a *convex nonlinear* recourse subproblem: cuts come from the
recourse NLP's KKT multipliers via the envelope theorem. ``solve_benders``
detects nonlinearity and dispatches to GBD automatically. The reported lower
bound is rigorous when the model is convex; on a nonconvex model GBD runs
heuristically and reports ``bound=None``.
"""

from __future__ import annotations

from discopt.decomposition.benders.gbd import (
    solve_gbd as solve_gbd,
)
from discopt.decomposition.benders.solver import (
    BendersConfig as BendersConfig,
)
from discopt.decomposition.benders.solver import (
    solve_benders as solve_benders,
)

__all__ = ["BendersConfig", "solve_benders", "solve_gbd"]
