"""GDPopt-style Logic-based Outer Approximation (LOA) solver for GDP.

Decomposes a GDP/MINLP into alternating MILP master + NLP subproblems.
Requires a matrix MILP backend for the master problem solver.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

# The master MILP lives in the GDP algorithm framework (#1775 G3.2); LOA imports
# it from there unchanged.
from discopt.gdp.algorithms.master import (  # noqa: F401 - re-exported for callers and tests
    _add_no_good_cut,
    _build_no_good_augmentation,
    _discrete_vars_all_binary,
    _int_config_key,
    _solve_master_milp,
)
from discopt.modeling.core import (
    Model,
    SolveResult,
)
from discopt.solvers import pounce_incumbent_options, pounce_option_defaults
from discopt.solvers._gap import optimality_gap

if TYPE_CHECKING:  # pragma: no cover - typing only
    # #1063: type-only. ``_relax/nlp_evaluator`` imports jax at module scope, so a
    # runtime import here put JAX in ``sys.modules`` merely by importing this
    # module — before any evaluator was even built.
    from discopt._relax.nlp_evaluator import NLPEvaluator

logger = logging.getLogger(__name__)


def solve_gdpopt_loa(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "ipm",
    milp_solver: str = "auto",
    algorithm: str = "loa",
    **options,
) -> SolveResult:
    """Solve a GDP model via Logic-based Outer Approximation.

    A thin entry point over :class:`discopt.gdp.algorithms.loa.LOA` (#1775 G4.1),
    the framework's LOA; ``options`` are its ``init_algorithm``,
    ``set_cover_iterlim`` and ``subproblem``.

    Parameters
    ----------
    model : Model
        The original model (before GDP reformulation).
    time_limit : float
        Wall-clock time limit in seconds.
    gap_tolerance : float
        Relative optimality gap tolerance.
    max_iterations : int
        Maximum LOA iterations (master solves).
    nlp_solver : str
        NLP solver backend for subproblems (``"ipm"``, ``"pounce"``, ``"ipopt"``).
    milp_solver : str
        MILP backend for LOA master problems: ``"auto"``, ``"pounce"``,
        ``"simplex"``, ``"highs"`` or ``"gurobi"``. ``"highs"`` is an explicit
        opt-in only (#356 took HiGHS off the default MILP path; #1060 restored
        it as a named master engine); ``"auto"`` never selects it.
    algorithm : str
        ``"loa"`` (:class:`~discopt.gdp.algorithms.loa.LOA`) or ``"gloa"``
        (:class:`~discopt.gdp.algorithms.gloa.GLOA`, #1775 G4.3: a relaxation
        master and global subproblems), ``"ric"``
        (:class:`~discopt.gdp.algorithms.ric.RIC`, #1775 G4.2: linear master and
        global subproblems), ``"enumerate"``
        (:class:`~discopt.gdp.algorithms.enumerate.Enumerate`: every
        configuration, global subproblems; takes ``max_configurations``) or
        ``"lbb"`` (:class:`~discopt.gdp.algorithms.lbb.LBB`, #1775 G4.4:
        branch and bound over disjunctions; takes ``propagate``) or ``"ldsda"``
        (:class:`~discopt.gdp.algorithms.ldsda.LDSDA`, #1775 G4.5: a local
        discrete-steepest descent; no bound, never ``optimal``).

    Returns
    -------
    SolveResult
    """
    cls = native_gdp_class(algorithm)
    if algorithm == "enumerate":
        # Every configuration is solved (or the run refuses up front): no master,
        # no NLP, and no iteration limit to truncate the enumeration.
        result: SolveResult = cls(
            model, time_limit=time_limit, gap_tolerance=gap_tolerance, **options
        ).run()
        return result
    if algorithm in ("lbb", "ldsda"):
        # Node / point solves are discopt's own: no master and no NLP solver to pass.
        result = cls(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            iteration_limit=max_iterations,
            **options,
        ).run()
        return result
    result = cls(
        model,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        iteration_limit=max_iterations,
        nlp_solver=nlp_solver,
        milp_solver=milp_solver,
        **options,
    ).run()
    return result


def native_gdp_class(algorithm: str) -> Any:
    """The :mod:`discopt.gdp.algorithms` class behind a native ``gdp_method``
    (its constructor's options differ by method, hence ``Any``)."""
    from discopt.gdp.algorithms.enumerate import Enumerate
    from discopt.gdp.algorithms.gloa import GLOA
    from discopt.gdp.algorithms.lbb import LBB
    from discopt.gdp.algorithms.ldsda import LDSDA
    from discopt.gdp.algorithms.loa import LOA
    from discopt.gdp.algorithms.ric import RIC

    classes = {
        "loa": LOA,
        "gloa": GLOA,
        "lbb": LBB,
        "ric": RIC,
        "enumerate": Enumerate,
        "ldsda": LDSDA,
    }
    if algorithm not in classes:
        raise ValueError(f"algorithm must be one of {sorted(classes)}, got {algorithm!r}")
    return classes[algorithm]


def _repair_incumbent(reformulated: Model, evaluator, x: np.ndarray) -> np.ndarray:
    """The NLP subproblem point, repaired onto its rows when that verifies (#1678).

    The fixed-integer NLP converges to its working accuracy, and on the big-M
    reformulation a residual of ~1e-8 relative to a row's term scale is enough to
    buy an objective BELOW the master's valid dual bound (measured: a convex
    process-selection GDP with unit capacity 100 returned an incumbent 1.2e-5 under
    the bound, which :func:`optimality_gap` correctly reads as a crossed, unclosed
    gap, so LOA reported ``feasible``; ``Model.solve`` then repaired the point onto
    an objective that closes the gap but, being downgrade-only, never restored
    ``optimal``; at capacity 1e4 the published point also failed the declared
    disjunction and logged UNVERIFIED CERTIFICATE).

    :func:`~discopt.validation.feasibility.repair_point` takes minimum-norm Newton
    steps onto the violated rows with integer columns frozen, so the disjunct rows
    the master's assignment activates are met to float noise. The repaired point is
    adopted only if :func:`~discopt.validation.feasibility.verify_point` accepts it
    on the reformulated model; otherwise the raw point is kept and judged exactly as
    before. Only the incumbent moves -- the dual bound is untouched -- and its
    objective is re-evaluated at the repaired point, so ``UB`` can only become more
    honest (it rises by what the residual had bought).
    """
    from discopt.validation.feasibility import repair_point, verify_point

    rep = repair_point(reformulated, x, evaluator=evaluator)
    if rep.x is None:
        logger.debug("LOA: incumbent repair declined (%s)", rep.reason)
        return x
    if not verify_point(reformulated, rep.x, evaluator=evaluator).ok:
        logger.debug("LOA: repaired incumbent did not verify; keeping the raw point")
        return x
    return np.asarray(rep.x, dtype=np.float64)


def _compute_gap(lb: float, ub: float) -> float:
    """LOA's optimality gap — see :mod:`discopt.solvers._gap` for the semantics.

    ``denom_floor=1e-10`` keeps LOA's stricter reporting scale for an
    *unconverged* run (the gap relative to the objective however small it is).
    It no longer degenerates on a converged near-zero optimum, because the shared
    absolute criterion closes the gap before the denominator is reached.
    """
    return optimality_gap(lb, ub, denom_floor=1e-10)


def _solve_nlp_relaxation(
    evaluator, lb, ub, nlp_solver: str, time_limit: float | None = None
) -> np.ndarray | None:
    """Solve the continuous NLP relaxation (within ``time_limit`` seconds, if given)."""
    lb_clip = np.clip(lb, -1e8, 1e8)
    ub_clip = np.clip(ub, -1e8, 1e8)
    x0 = 0.5 * (lb_clip + ub_clip)

    try:
        if nlp_solver == "ipopt":
            from discopt.solvers.nlp_ipopt import solve_nlp
        else:
            from discopt.solvers.nlp_pounce import solve_nlp

        options: dict = {"print_level": 0}
        if time_limit is not None:
            options["max_wall_time"] = max(float(time_limit), 0.05)
        result = solve_nlp(evaluator, x0, options=options)

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL and result.x is not None:
            return np.asarray(result.x, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - no relaxation point is handled by the caller
        # Capability-disabling: without this point LOA starts from no relaxation
        # solution at all, so a silent skip looks like "the relaxation is useless".
        logger.debug("GDP NLP relaxation solve failed: %s: %s", type(exc).__name__, exc)
    return None


def _solve_nlp_subproblem(
    evaluator, sub_lb, sub_ub, nlp_solver: str, time_limit: float | None = None
) -> np.ndarray | None:
    """Solve NLP subproblem with fixed integer bounds (within ``time_limit`` seconds)."""
    # IPM log-barriers require strict lb < x < ub; relax fixed vars by tiny slack
    ipm_lb = sub_lb.copy()
    ipm_ub = sub_ub.copy()
    fixed = ipm_lb == ipm_ub
    ipm_lb[fixed] -= 1e-8
    ipm_ub[fixed] += 1e-8

    lb_clip = np.clip(ipm_lb, -1e8, 1e8)
    ub_clip = np.clip(ipm_ub, -1e8, 1e8)
    x0 = np.clip(0.5 * (lb_clip + ub_clip), sub_lb, sub_ub)

    # Create a bounds-overriding evaluator proxy
    proxy = _BoundsProxy(evaluator, ipm_lb, ipm_ub)

    try:
        if nlp_solver == "ipopt":
            from discopt.solvers.nlp_ipopt import solve_nlp
        else:
            from discopt.solvers.nlp_pounce import solve_nlp

        # LOA's incumbent comes from here, so it takes the incumbent options
        # (#945): pre-#945 this returned a point below its own lb and below its own
        # dual bound, and LOA reported that as `optimal`.
        opts = pounce_option_defaults()
        opts.update(pounce_incumbent_options())
        opts.update({"max_iter": 200})
        if time_limit is not None:
            # #1775 G4.1: the caller's deadline remainder, as for every subsolve.
            opts["max_wall_time"] = max(float(time_limit), 0.05)
        result = solve_nlp(cast("NLPEvaluator", proxy), x0, options=opts)

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL and result.x is not None:
            return np.asarray(result.x, dtype=np.float64)

        # Accept ITERATION_LIMIT if the solution is primal-feasible.
        # The IPM may fail to certify dual optimality (e.g. on LPs or
        # near-degenerate NLPs) while still finding the correct primal
        # solution.  Any feasible point is a valid LOA upper bound.
        if result.status == SolveStatus.ITERATION_LIMIT and result.x is not None:
            x_sol = np.asarray(result.x, dtype=np.float64)
            atol = 1e-6
            if np.all(x_sol >= sub_lb - atol) and np.all(x_sol <= sub_ub + atol):
                feas = True
                if evaluator.n_constraints > 0:
                    g_val = evaluator.evaluate_constraints(x_sol)
                    model = getattr(evaluator, "_model", None)
                    if model is None:
                        model = proxy._eval._model
                    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

                    c_lb, c_ub = _infer_constraint_bounds(model)
                    for ci in range(len(c_lb)):
                        if g_val[ci] < c_lb[ci] - atol or g_val[ci] > c_ub[ci] + atol:
                            feas = False
                            break
                if feas:
                    return x_sol
    except Exception as exc:  # noqa: BLE001 - no subproblem point is handled by the caller
        # Capability-disabling: a swallowed failure here means LOA silently never
        # gets an upper bound from this configuration.
        logger.debug("GDP fixed-integer NLP subproblem failed: %s: %s", type(exc).__name__, exc)
    return None


def _fixed_subproblem_rigorously_infeasible(evaluator, sub_lb, sub_ub, tol: float = 1e-6) -> bool:
    """RIGOROUSLY decide whether the fixed-integer subproblem is infeasible.

    C-35: a no-good cut is only sound when the excluded configuration is proved
    infeasible/dominated. A local NLP that merely failed to converge is not such
    a proof. Sufficient rigorous condition: when there are NO free continuous
    variables (every variable pinned, ``sub_lb == sub_ub``), the subproblem has
    a single candidate point; a constraint violation there proves infeasibility.
    Otherwise return False (do not claim rigorous infeasibility).
    """
    sub_lb = np.asarray(sub_lb, dtype=np.float64)
    sub_ub = np.asarray(sub_ub, dtype=np.float64)
    if not np.all(sub_lb >= sub_ub - 1e-12):
        return False
    if evaluator.n_constraints == 0:
        return False
    try:
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        model = getattr(evaluator, "_model", None)
        c_lb, c_ub = _infer_constraint_bounds(model)
        x_point = 0.5 * (sub_lb + sub_ub)
        cons = np.asarray(evaluator.evaluate_constraints(x_point), dtype=np.float64)
        if not np.all(np.isfinite(cons)):
            return False
        return bool(np.any(cons < c_lb - tol) or np.any(cons > c_ub + tol))
    except Exception:
        return False


class _BoundsProxy:
    """Wraps an NLPEvaluator with overridden variable bounds."""

    def __init__(self, evaluator: "NLPEvaluator", new_lb, new_ub) -> None:
        self._eval = evaluator
        self._lb = np.asarray(new_lb, dtype=np.float64)
        self._ub = np.asarray(new_ub, dtype=np.float64)

    @property
    def n_variables(self):
        return self._eval.n_variables

    @property
    def n_constraints(self):
        return self._eval.n_constraints

    @property
    def variable_bounds(self):
        return self._lb, self._ub

    @property
    def timing_bucket(self):
        """Forward the wrapped evaluator's layer (issue #74).

        This class enumerates its members explicitly rather than delegating via
        ``__getattr__`` (unlike ``oa._BoundsProxy``), so without this property a
        GDP-LOA sub-NLP falls to ``_IpoptCallbacks``'s "undeclared" arm: its
        derivative callbacks are charged to nothing, the enclosing solver region
        absorbs them, and the run emits a `timing-bucket-unknown` warning into
        whatever is reading its output (measured in
        docs/notebooks/tutorial_gdp.ipynb, #1362).
        """
        return self._eval.timing_bucket

    @property
    def _model(self):
        return self._eval._model

    @property
    def _obj_fn(self):
        return self._eval._obj_fn

    @property
    def _cons_fn(self):
        return self._eval._cons_fn

    @property
    def _source_constraints(self):
        return self._eval._source_constraints

    @property
    def _constraint_flat_sizes(self):
        return self._eval._constraint_flat_sizes

    def evaluate_objective(self, x):
        return self._eval.evaluate_objective(x)

    def evaluate_gradient(self, x):
        return self._eval.evaluate_gradient(x)

    def evaluate_hessian(self, x):
        return self._eval.evaluate_hessian(x)

    def evaluate_constraints(self, x):
        return self._eval.evaluate_constraints(x)

    def evaluate_jacobian(self, x):
        return self._eval.evaluate_jacobian(x)

    def evaluate_lagrangian_hessian(self, x, obj_factor, lam):
        return self._eval.evaluate_lagrangian_hessian(x, obj_factor, lam)


def _add_oa_cuts(
    evaluator,
    x_star,
    n_vars,
    n_cons,
    oa_A_rows,
    oa_b_rows,
    obj_is_linear,
    oa_rows,
    objective_is_convex,
):
    """Generate OA cuts at x_star and append to cut lists."""
    from discopt._relax.cutting_planes import (
        generate_oa_cuts_from_evaluator,
        generate_objective_oa_cut,
    )

    # Constraint OA cuts
    if n_cons > 0:
        cuts = generate_oa_cuts_from_evaluator(
            evaluator,
            x_star,
            constraint_senses=oa_rows.senses,
            convex_mask=oa_rows.convex_mask,
        )
        for cut in cuts:
            if cut.sense == "<=":
                oa_A_rows.append(cut.coeffs.copy())
                oa_b_rows.append(cut.rhs)
            elif cut.sense == ">=":
                oa_A_rows.append(-cut.coeffs.copy())
                oa_b_rows.append(-cut.rhs)

    # Objective OA cut (only if nonlinear)
    if not obj_is_linear and objective_is_convex:
        obj_cut = generate_objective_oa_cut(evaluator, x_star, n_vars + 1, z_index=n_vars)
        # `None`: no finite tangent at x_star (e.g. sqrt at 0) -- add no row (#1491).
        if obj_cut is not None:
            # This cut is: coeffs @ x <= rhs (underestimates the objective)
            oa_A_rows.append(obj_cut.coeffs.copy())
            oa_b_rows.append(obj_cut.rhs)


def _build_x_dict(x_flat: np.ndarray, model: Model) -> dict:
    """Convert flat solution vector to {var_name: value} dict."""
    result = {}
    offset = 0
    for v in model._variables:
        result[v.name] = x_flat[offset : offset + v.size].reshape(v.shape)
        offset += v.size
    return result
