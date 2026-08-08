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


# ---------------------------------------------------------------------------
# Shared POUNCE/Ipopt option defaults (issue #940)
# ---------------------------------------------------------------------------

# Per-build measurement behind these two values (issue #940). A 49-LP battery
# (n=5..40, data scale 1e0..1e7, plus the tutorial diet LP), counting rejections
# by discopt's own feasibility guard / non-convergence, with EVERY arm pinning
# BOTH options explicitly:
#
#                                  pounce @ main        PyPI 0.9.0
#   POUNCE's own defaults          38 trips, 3 nonopt   23 trips, 3 nonopt
#   constr_viol_tol=1e-8 alone     39 trips, 2 nonopt    0 trips, 3 nonopt
#   bound_relax_factor=0 alone      0 trips, 2 nonopt    (not measured alone)
#   BOTH (what ships)               0 trips, 2 nonopt    0 trips, 2 nonopt
#
# Which mechanism dominates depends on the build, so both are set:
# constr_viol_tol carries the published wheel and is a no-op on main;
# bound_relax_factor carries main. `pyproject.toml` admits `pounce-solver>=0.9`
# while CI tracks `main`, so neither may be dropped on the grounds that the
# other covers it. Naming only one in an experiment arm leaves the other at the
# shipped value and silently mislabels the arm — that cost one round of this fix.
#
# The residual 2 non-convergences are pre-existing stalls at data scale ~1e7,
# present under POUNCE's own defaults too. `constr_viol_tol=1e-12` was measured
# and REJECTED: it reaches the tolerance but destroys convergence on main (18 of
# 49 instances fail to reach OPTIMAL, against 2 at the default).
#
# None of this relaxes any discopt guard: the guards remain the arbiter
# (CLAUDE.md §1), and a point that still fails one is still rejected.
# Requested absolute cap on the max-norm of the UNSCALED constraint violation at
# termination. POUNCE inherits Ipopt's default of 1e-4, two orders LOOSER than
# discopt's own 1e-6 constraint tolerance
# (``solver._matrix_solution_feasible``, ``_jax.primal_heuristics.
# _check_constraint_feasibility``), so a converged point can sit 1e-4 on the
# infeasible side of a row and still be reported OPTIMAL. Dominant on the
# published pounce-solver wheel; a no-op on current ``main``.
POUNCE_CONSTR_VIOL_TOL = 1e-8

# Ipopt's ``bound_relax_factor`` (default 1e-8) deliberately relaxes every
# bound — including the slack bounds standing in for inequality rows — by
# ``1e-8*(1 + |bound|)``. The solve then converges honestly, but to a RELAXED
# box, so the returned point sits OUTSIDE the box the caller declared and the
# error grows with the data. That is what discopt's feasibility guards were
# rejecting POUNCE points for, and no convergence tolerance can remove it: the
# solver has already converged; the box it converged to is the wrong one.
#
# Pinning it to 0 keeps iterates inside the declared box. This matters well
# beyond the LP/QP fast path — an NLP iterate stepping below a bound of 0 is
# exactly how ``log``/``sqrt``/``1/x`` produce the NaN failures discopt warns
# about — which is why it lives here rather than in one backend (CLAUDE.md §2:
# fix the class, not the instance).
POUNCE_BOUND_RELAX_FACTOR = 0.0


def pounce_option_defaults() -> dict:
    """discopt's baseline POUNCE options: quiet, and inside the declared box.

    THE single source of truth for EVERY POUNCE entry point: the matrix-form
    backends (``lp_pounce``, ``qp_pounce``, #940) and the NLP path
    (``nlp_pounce.solve_nlp`` and ``solver.py``'s batch path, #945). Seed it
    *before* merging a caller's options so an explicit request still wins::

        opts = pounce_option_defaults()
        opts.update(caller_options or {})

    Do not re-spell these values at a call site — a second copy is how one entry
    point silently keeps Ipopt's defaults while the rest move.

    Extending the seed to the NLP path was held back until #945 because it is a
    certification-semantics change, not just an option: with incumbents genuinely
    inside their boxes, ``incumbent - bound`` stops being ``<= 0``, and 12
    ``gap == 0 @ 1e-9`` assertions across OA / GDPopt / MindtPy turned out to be
    satisfiable only by a solver returning slightly-infeasible incumbents. #945
    settled that by giving gap closure an absolute criterion at discopt's own
    ``1e-6`` — see :mod:`discopt.solvers._gap`.
    """
    return {
        "print_level": 0,
        "constr_viol_tol": POUNCE_CONSTR_VIOL_TOL,
    }


def pounce_incumbent_options() -> dict:
    """Extra options for a POUNCE call whose returned POINT is the product (#945).

    ``bound_relax_factor = 0`` lives here and deliberately NOT in
    :func:`pounce_option_defaults`, because the split is not "LP versus NLP" — it
    is **what the caller consumes**:

    * A call site whose ``x`` becomes a reported solution or incumbent needs the
      point inside the box the model declared. Ipopt's default relaxes every
      bound (and every slack bound standing in for an inequality row) by
      ``1e-8*(1 + |bound|)``, and a squared row takes the square root of that: on
      the MindtPy constraint-qualification fixture ``(x-3)^2 <= 0`` admitted every
      ``x`` within 1e-4 of 3, and discopt certified ``optimal`` 1e-4 below an exact
      optimum of 3.0.
    * A call site whose **multipliers** are the product must NOT set it. A
      degenerate feasible set (Slater failing, e.g. ``x0^2 + x1^2 <= 0``) has no
      finite multiplier; Ipopt's relaxation hands it an artificial interior and
      keeps the dual finite. Pinning it to 0 makes the multiplier diverge and the
      cut built from it useless — measured at 9.8e7 with a cut slope of 7.9e8 on
      GBD, and it is why the same option applied backend-wide cost the Benders
      dual LP its convergence (#940: two correctness-lane tests, 1.6s -> 79s).

    So: use this at incumbent/solution producers, never at dual consumers
    (Benders and GBD recourse, OBBT). Seed it after
    :func:`pounce_option_defaults` and before a caller's own options::

        opts = pounce_option_defaults()
        opts.update(pounce_incumbent_options())
        opts.update(caller_options or {})
    """
    return {"bound_relax_factor": POUNCE_BOUND_RELAX_FACTOR}
