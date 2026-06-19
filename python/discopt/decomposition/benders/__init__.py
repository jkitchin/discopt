"""Benders decomposition for block-angular / two-stage (mixed-integer) LPs.

The master problem holds the *complicating* (first-stage) variables; for a
fixed master point the recourse subproblem is a linear program whose optimal
dual yields a Benders **optimality cut**, and whose infeasibility yields a
**feasibility cut**. Iterating tightens the master's lower bound until it meets
the incumbent's upper bound.

v1 implements **classical Benders** for problems with linear constraints and a
linear objective, where all integer variables are first-stage (so the recourse
is a continuous LP). This is rigorous: every cut is a valid global
underestimator (LP weak duality), so the master objective is a sound lower
bound. Generalized Benders (convex-NLP subproblems) is a planned extension.
"""

from __future__ import annotations

from discopt.decomposition.benders.solver import (
    BendersConfig as BendersConfig,
)
from discopt.decomposition.benders.solver import (
    solve_benders as solve_benders,
)

__all__ = ["BendersConfig", "solve_benders"]
