"""Solver backends for discopt."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class SolveStatus(Enum):
    """Terminal status of a solve (optimal, infeasible, unbounded, ...)."""

    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ITERATION_LIMIT = "iteration_limit"
    TIME_LIMIT = "time_limit"
    CUTOFF = "cutoff"
    ERROR = "error"


@dataclass
class InfeasibilityCertificate:
    """Constructive (minimal-violation) witness that an LP is infeasible.

    Produced by the elastic Phase-1 LP that minimizes total constraint
    violation. A positive ``total_violation`` is, by LP duality, a Farkas
    certificate: no point satisfies all constraints and bounds simultaneously.
    ``ineq_violations`` and ``eq_violations`` give the minimal violation each
    row must incur (an entry ``> 0`` marks a conflicting constraint) — an
    IIS-like diagnosis, though not guaranteed to be a minimal irreducible
    subsystem. Rows are in the order the matrices were passed (inequalities
    then equalities).
    """

    total_violation: float
    ineq_violations: np.ndarray
    eq_violations: np.ndarray


@dataclass
class LPResult:
    """Result of solving a linear program.

    ``dual_values`` are constraint marginals (one per row).
    ``reduced_costs`` are variable marginals (one per column). Both are in
    the sign convention of the LP as passed to the solver (i.e. the
    internal minimization form).

    ``infeasibility_certificate`` is populated (when available) on an
    ``INFEASIBLE`` result to witness *why* the LP is infeasible.
    """

    status: SolveStatus
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    dual_values: Optional[np.ndarray] = None
    reduced_costs: Optional[np.ndarray] = None
    basis: Optional[object] = None
    iterations: int = 0
    wall_time: float = 0.0
    infeasibility_certificate: Optional[InfeasibilityCertificate] = None


@dataclass
class MILPResult:
    """Result of solving a mixed-integer linear program.

    ``objective`` is the incumbent value (an *upper* bound for a minimization on
    a non-optimal exit). ``bound`` is the rigorous dual *lower* bound on the
    optimum (for minimization): it equals ``objective`` once the solve is proven
    optimal and remains a valid lower bound on a time/node-limited exit.

    Callers that need a sound lower bound (AMP / OA / GDP-LOA master relaxations)
    MUST read ``bound``, never ``objective`` — using the incumbent as a lower
    bound can inflate the global LB past the true optimum and falsely certify
    optimality. ``bound`` is ``None`` when no valid dual bound is available.
    ``solution_pool`` and ``solution_pool_objectives`` are populated only by
    backends that explicitly expose multiple incumbent solutions.
    """

    status: SolveStatus
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    bound: Optional[float] = None
    gap: Optional[float] = None
    node_count: int = 0
    iterations: int = 0
    wall_time: float = 0.0
    solution_pool: Optional[list[np.ndarray]] = None
    solution_pool_objectives: Optional[list[float]] = None
    callback_stats: Optional[dict[str, object]] = None


@dataclass
class QPResult:
    """Result of solving a quadratic program.

    ``infeasibility_certificate`` is populated (when available) on an
    ``INFEASIBLE`` result to witness *why* the QP is infeasible.

    ``kkt_error`` is the solver's final KKT residual when it reports one
    (interior-point backends only; ``None`` for vertex solvers like HiGHS that
    reach an exact optimum). Consumers gate a POUNCE-first default on it: a
    nonzero residual flags an unconverged "optimal" (issue #145) so the caller
    can degrade to a vertex solver instead of trusting a drifted objective.
    """

    status: SolveStatus
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    bound: Optional[float] = None
    gap: Optional[float] = None
    dual_values: Optional[np.ndarray] = None
    reduced_costs: Optional[np.ndarray] = None
    node_count: int = 0
    iterations: int = 0
    wall_time: float = 0.0
    infeasibility_certificate: Optional[InfeasibilityCertificate] = None
    kkt_error: Optional[float] = None


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
