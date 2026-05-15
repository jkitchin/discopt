"""Solver backends for discopt."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class SolveStatus(Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ITERATION_LIMIT = "iteration_limit"
    TIME_LIMIT = "time_limit"
    ERROR = "error"


@dataclass
class LPResult:
    """Result of solving a linear program.

    ``dual_values`` are constraint marginals (one per row).
    ``reduced_costs`` are variable marginals (one per column). Both are in
    the sign convention of the LP as passed to the solver (i.e. the
    internal minimization form).
    """

    status: SolveStatus
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    dual_values: Optional[np.ndarray] = None
    reduced_costs: Optional[np.ndarray] = None
    basis: Optional[object] = None
    iterations: int = 0
    wall_time: float = 0.0


@dataclass
class MILPResult:
    """Result of solving a mixed-integer linear program."""

    status: SolveStatus
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    gap: Optional[float] = None
    node_count: int = 0
    iterations: int = 0
    wall_time: float = 0.0


@dataclass
class QPResult:
    """Result of solving a quadratic program."""

    status: SolveStatus
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    dual_values: Optional[np.ndarray] = None
    reduced_costs: Optional[np.ndarray] = None
    node_count: int = 0
    iterations: int = 0
    wall_time: float = 0.0


@dataclass
class NLPResult:
    """Result of solving a nonlinear program.

    ``multipliers`` are constraint Lagrange multipliers (one per constraint
    row). ``bound_multipliers_lower`` and ``bound_multipliers_upper`` are
    the multipliers on the variable lower- and upper-bound constraints
    (one per variable, ≥ 0 at active bounds, ≈ 0 elsewhere). All are in
    the sign convention of the problem as passed to the solver
    (i.e. the internal minimization form).
    """

    status: SolveStatus
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    multipliers: Optional[np.ndarray] = None
    bound_multipliers_lower: Optional[np.ndarray] = None
    bound_multipliers_upper: Optional[np.ndarray] = None
    iterations: int = 0
    wall_time: float = 0.0
