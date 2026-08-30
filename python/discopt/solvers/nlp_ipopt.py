"""
NLP solver wrapper using cyipopt (Python binding for Ipopt).

Phase 1 scaffolding: uses cyipopt for continuous relaxation solves.
Will be replaced by direct Rust Ipopt bindings later.

Maps NLPEvaluator callbacks to cyipopt.Problem interface:
  - objective, gradient, constraints, jacobian, hessian
  - Variable and constraint bounds
  - Ipopt status codes to SolveStatus enum
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from discopt.modeling.core import Constraint, Model
from discopt.solvers import NLPResult, SolveStatus

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    # #75: module-scope import pulled jax onto every solve via nlp_pounce ->
    # nlp_backend. Annotation-only at runtime.
    from discopt._relax.nlp_evaluator import NLPEvaluator

# Ipopt status code mapping
# See: https://coin-or.github.io/Ipopt/IpReturnCodes_8inc.html
#
# NOTE: Status 2 ("Infeasible_Problem_Detected") is mapped to ERROR rather
# than INFEASIBLE because IPOPT can only detect *local* infeasibility.
# For non-convex NLPs the problem may still be feasible from a different
# starting point.  Mapping to ERROR prevents the solver from confidently
# reporting "infeasible" when the problem is merely hard to solve.
_IPOPT_STATUS_MAP: dict[int, SolveStatus] = {
    0: SolveStatus.OPTIMAL,  # Solve_Succeeded
    1: SolveStatus.OPTIMAL,  # Solved_To_Acceptable_Level
    2: SolveStatus.ERROR,  # Infeasible_Problem_Detected (local only)
    # Ipopt code 3 = Search_Direction_Becomes_Too_Small: the IPM stalled on a tiny
    # step, typically AT/near a local solution it cannot certify to tolerance — NOT
    # unboundedness (that is code 4, Diverging_Iterates). Mapping it to UNBOUNDED made
    # a converged-but-uncertified node look like an unbounded relaxation, which a local
    # NLP can never prove; on jit1's B&B nodes this produced a false UNBOUNDED cascade
    # (pounce#257/#258) that forced a cyipopt-retry workaround. Treat it as a
    # non-converged limit — the B&B handles it soundly (non-pruning, uncertified),
    # never as a false unbounded verdict.
    3: SolveStatus.ITERATION_LIMIT,  # Search_Direction_Becomes_Too_Small (stalled, not unbounded)
    4: SolveStatus.ERROR,  # Diverging_Iterates
    5: SolveStatus.ERROR,  # User_Requested_Stop
    6: SolveStatus.ERROR,  # Feasible_Point_Found (not optimal)
    -1: SolveStatus.ITERATION_LIMIT,  # Maximum_Iterations_Exceeded
    -2: SolveStatus.ERROR,  # Restoration_Failed
    -3: SolveStatus.ERROR,  # Error_In_Step_Computation
    -4: SolveStatus.TIME_LIMIT,  # Maximum_CpuTime_Exceeded
    -5: SolveStatus.TIME_LIMIT,  # Maximum_WallTime_Exceeded
}


def _charge_evaluator(method):
    """Charge a derivative callback's time to the evaluator's own layer.

    This adapter is the *only* path from the NLP subsolver (POUNCE or cyipopt,
    both native) back into the Python evaluator, so it is the correct seam for
    attribution. Without it, the callbacks run inside ``charge("pounce")`` and
    POUNCE's bucket absorbs the derivative cost — the same kind of cross-layer
    inflation the layer profile exists to expose.

    The bucket is read from the evaluator (``timing_bucket``) rather than fixed
    here. It used to be hardcoded ``"jax"`` on the premise that "today's
    evaluator is JAX-backed"; #75's tape evaluator made that false, and because
    ``charge`` records *self* time the error was two-sided — a JAX-free tape
    solve reported fabricated ``jax_time`` **and** an equally understated
    ``pounce_time``, i.e. both halves of the rust/python partition were wrong on
    the very measurement that judges the JAX removal. Measured before this
    change on a bilinear MINLP: ``jax_time = 0.00217 s`` with
    ``"jax" not in sys.modules``.
    """
    import functools

    from discopt import _timing

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._bucket is None:
            return method(self, *args, **kwargs)
        with _timing.charge(self._bucket):
            return method(self, *args, **kwargs)

    return wrapper


class _IpoptCallbacks:
    """Adapter mapping NLPEvaluator methods to cyipopt.Problem callbacks."""

    def __init__(self, evaluator) -> None:
        self._ev = evaluator
        self._n = evaluator.n_variables
        self._m = evaluator.n_constraints
        self._use_sparse = (
            hasattr(evaluator, "has_sparse_structure") and evaluator.has_sparse_structure()
        )
        # Resolve once per solve, not once per callback. An evaluator that does
        # not declare its layer gets charged to *nothing* rather than to a
        # guessed bucket: an unknown backend's time then stays with the enclosing
        # region, which is merely coarse, whereas guessing invents a number about
        # a library that may never have been loaded. The in-tree evaluators both
        # declare one (the proxies in ``solver.py`` forward it), so the warning
        # below fires only for a duck-typed evaluator from outside the package.
        self._bucket = getattr(evaluator, "timing_bucket", None)
        if self._bucket is None:
            logger.warning(
                "Evaluator %s declares no `timing_bucket`; its derivative-callback "
                "time will be left with the enclosing solver region and the layer "
                "profile will over-report that layer [timing-bucket-unknown].",
                type(evaluator).__name__,
            )

    @_charge_evaluator
    def objective(self, x: np.ndarray) -> float:
        return self._ev.evaluate_objective(x)

    @_charge_evaluator
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self._ev.evaluate_gradient(x)

    @_charge_evaluator
    def constraints(self, x: np.ndarray) -> np.ndarray:
        return self._ev.evaluate_constraints(x)

    @_charge_evaluator
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        if self._use_sparse:
            return self._ev.evaluate_jacobian_values(x)
        # cyipopt wants the Jacobian flattened in the order given by jacobianstructure
        jac = self._ev.evaluate_jacobian(x)
        return jac.flatten()

    @_charge_evaluator
    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        if self._use_sparse:
            return self._ev.jacobian_structure()
        # Dense structure: all (row, col) pairs
        rows, cols = np.meshgrid(np.arange(self._m), np.arange(self._n), indexing="ij")
        return (rows.flatten(), cols.flatten())

    @_charge_evaluator
    def hessian(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float) -> np.ndarray:
        if self._use_sparse:
            return self._ev.evaluate_hessian_values(x, obj_factor, lagrange)
        # Hessian of the Lagrangian = obj_factor * H_obj + sum(lagrange[i] * H_c[i])
        if hasattr(self._ev, "evaluate_lagrangian_hessian"):
            h = self._ev.evaluate_lagrangian_hessian(x, obj_factor, lagrange)
        else:
            h = obj_factor * self._ev.evaluate_hessian(x)

        # Extract lower triangle in row-major order matching hessianstructure
        rows, cols = self.hessianstructure()
        return h[rows, cols]

    @_charge_evaluator
    def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        if self._use_sparse:
            return self._ev.hessian_structure()
        # Lower triangle (including diagonal)
        rows, cols = np.tril_indices(self._n)
        return (rows, cols)


_BOUNDS_CACHE_ATTR = "_discopt_constraint_bounds_cache"


def _constraint_bounds_fingerprint(constraints, sizes) -> tuple:
    """An O(1) identity of the constraint list the bounds were derived from.

    ``constraints`` and ``sizes`` are owned by the evaluator, which also owns
    the cache entry, so neither object can be garbage-collected (and neither
    ``id`` recycled) while the entry is reachable. Rebuilding either list --
    the only way a compiled evaluator's senses can change -- changes the
    identity and misses the cache.
    """
    return (id(constraints), len(constraints), id(sizes), int(np.sum(sizes)))


def _cached_constraint_bounds(source, constraints, sizes):
    entry = getattr(source, _BOUNDS_CACHE_ATTR, None)
    if entry is None:
        return None
    fingerprint, cl, cu = entry
    if fingerprint != _constraint_bounds_fingerprint(constraints, sizes):
        return None
    return cl, cu


def _store_constraint_bounds(source, constraints, sizes, cl, cu) -> None:
    # Evaluators defined with ``__slots__`` simply go uncached rather than
    # raising; the bounds are recomputed exactly as they were before.
    if not hasattr(source, "__dict__"):
        return
    entry = (_constraint_bounds_fingerprint(constraints, sizes), cl, cu)
    setattr(source, _BOUNDS_CACHE_ATTR, entry)


def _infer_constraint_bounds(source) -> tuple[np.ndarray, np.ndarray]:
    """Infer constraint bounds (cl, cu) from model constraint senses.

    Accepts either a :class:`Model` or an :class:`NLPEvaluator`.

    The NLPEvaluator compiles constraints as ``body - rhs``, so we need:
      - For ``<=`` constraints: ``body - rhs <= 0``, so ``cl=-inf, cu=0``.
      - For ``==`` constraints: ``body - rhs == 0``, so ``cl=0, cu=0``.
      - For ``>=`` constraints: ``body - rhs >= 0``, so ``cl=0, cu=inf``.

    When an NLPEvaluator is supplied, each source Constraint's bounds are
    repeated to match the evaluator's ``_constraint_flat_sizes`` (needed
    for vector-valued bodies such as DAEBuilder's vectorized collocation
    residual). When a Model is supplied directly, each source Constraint
    contributes exactly one row (legacy scalar behavior).
    """
    if isinstance(source, Model):
        constraints = [c for c in source._constraints if isinstance(c, Constraint)]
        sizes = np.ones(len(constraints), dtype=np.intp)
    else:
        constraints = source._source_constraints
        sizes = source._constraint_flat_sizes
        # An NLPEvaluator compiles its constraint list once in ``__init__`` and
        # never mutates it, but the bounds were rebuilt on every call: the OA
        # feasibility phase calls this 45k times per solve through
        # ``_constraint_violation_data``, and each call allocates ``2 * n_cons``
        # ``np.full`` arrays plus two concatenates. Measured on
        # ``portfol_classical050_1`` (103 constraints): 111.4 us/call, ~5.1 s of
        # a 32.8 s solve. A ``Model`` can gain constraints between calls, so
        # only the evaluator branch is cached.
        cached = _cached_constraint_bounds(source, constraints, sizes)
        if cached is not None:
            return cached[0].copy(), cached[1].copy()

    cl_parts: list[np.ndarray] = []
    cu_parts: list[np.ndarray] = []
    for c, sz in zip(constraints, sizes):
        sz_int = int(sz)
        if c.sense == "<=":
            lo, hi = -1e20, 0.0
        elif c.sense == "==":
            lo, hi = 0.0, 0.0
        elif c.sense == ">=":
            lo, hi = 0.0, 1e20
        else:
            raise ValueError(f"Unknown constraint sense: {c.sense}")
        cl_parts.append(np.full(sz_int, lo, dtype=np.float64))
        cu_parts.append(np.full(sz_int, hi, dtype=np.float64))

    if not cl_parts:
        cl = np.empty(0, dtype=np.float64)
        cu = np.empty(0, dtype=np.float64)
    else:
        cl = np.concatenate(cl_parts)
        cu = np.concatenate(cu_parts)

    if isinstance(source, Model):
        # Never cached, so these arrays are already the caller's alone.
        return cl, cu

    _store_constraint_bounds(source, constraints, sizes, cl, cu)
    # Callers receive their own arrays, exactly as before this was cached, so a
    # caller that writes into the result cannot corrupt the cache.
    return cl.copy(), cu.copy()


def solve_nlp(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]] = None,
    options: Optional[dict] = None,
) -> NLPResult:
    """
    Solve an NLP using cyipopt with JAX-compiled callbacks.

    Args:
        evaluator: NLPEvaluator providing objective/gradient/Hessian/constraint/Jacobian.
        x0: Initial point (n,).
        constraint_bounds: List of (cl, cu) per constraint. None to infer from model.
        options: Ipopt options dict (e.g., {'max_iter': 1000, 'tol': 1e-8}).

    Returns:
        NLPResult with solution
    """
    try:
        import cyipopt
    except ImportError:
        raise ImportError(
            "cyipopt is required for solve_nlp. Install it with:\n"
            "  pip install cyipopt\n"
            "Note: cyipopt requires the Ipopt C library to be installed."
        )

    opts = dict(options) if options else {}
    opts.setdefault("print_level", 0)

    n = evaluator.n_variables
    m = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds

    # Constraint bounds
    if constraint_bounds is not None:
        cl = np.array([b[0] for b in constraint_bounds], dtype=np.float64)
        cu = np.array([b[1] for b in constraint_bounds], dtype=np.float64)
    elif m > 0:
        cl, cu = _infer_constraint_bounds(evaluator)
    else:
        cl = np.empty(0, dtype=np.float64)
        cu = np.empty(0, dtype=np.float64)

    callbacks = _IpoptCallbacks(evaluator)

    problem = cyipopt.Problem(
        n=n,
        m=m,
        problem_obj=callbacks,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    # cyipopt requires native Python types (rejects numpy scalars).
    # Some options (e.g. max_wall_time) may not exist in older Ipopt versions.
    import logging as _logging

    import numpy as _np

    _logger = _logging.getLogger(__name__)
    for key, value in opts.items():
        try:
            if isinstance(value, (_np.floating, float)):
                problem.add_option(key, float(value))
            elif isinstance(value, (_np.integer, int)):
                problem.add_option(key, int(value))
            else:
                problem.add_option(key, value)
        except TypeError:
            _logger.debug("Ipopt option '%s' not accepted, skipping", key)

    t0 = time.perf_counter()
    x, info = problem.solve(x0.astype(np.float64))
    wall_time = time.perf_counter() - t0

    status_code = info["status"]
    status = _IPOPT_STATUS_MAP.get(status_code, SolveStatus.ERROR)

    multipliers = info.get("mult_g", None)
    if multipliers is not None and len(multipliers) == 0:
        multipliers = None
    mult_x_L = info.get("mult_x_L", None)
    if mult_x_L is not None and len(mult_x_L) == 0:
        mult_x_L = None
    mult_x_U = info.get("mult_x_U", None)
    if mult_x_U is not None and len(mult_x_U) == 0:
        mult_x_U = None

    return NLPResult(
        status=status,
        x=np.asarray(x),
        objective=float(info["obj_val"]),
        multipliers=np.asarray(multipliers) if multipliers is not None else None,
        bound_multipliers_lower=np.asarray(mult_x_L) if mult_x_L is not None else None,
        bound_multipliers_upper=np.asarray(mult_x_U) if mult_x_U is not None else None,
        iterations=0,  # Ipopt doesn't expose iteration count via this API
        wall_time=wall_time,
    )


def solve_nlp_from_model(
    model: Model,
    x0: Optional[np.ndarray] = None,
    options: Optional[dict] = None,
) -> NLPResult:
    """Convenience: create NLPEvaluator from model and solve.

    Args:
        model: A Model with objective and constraints set.
        x0: Initial point (n,). If None, uses midpoint of bounds clipped to [-100, 100].
        options: Ipopt options dict.

    Returns:
        NLPResult with solution.
    """
    from discopt._tape_nlp_evaluator import make_evaluator

    evaluator = make_evaluator(model)

    if x0 is None:
        lb, ub = evaluator.variable_bounds
        lb_clipped = np.clip(lb, -100.0, 100.0)
        ub_clipped = np.clip(ub, -100.0, 100.0)
        x0 = 0.5 * (lb_clipped + ub_clipped)

    return solve_nlp(evaluator, x0, options=options)
