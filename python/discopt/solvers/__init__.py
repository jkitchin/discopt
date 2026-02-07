"""Solver backends for discopt."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    """Result of solving a linear program."""

    status: SolveStatus
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    dual_values: Optional[np.ndarray] = None
    basis: Optional[object] = None
    iterations: int = 0
    wall_time: float = 0.0
