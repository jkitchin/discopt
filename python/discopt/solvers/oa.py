"""General-purpose Outer Approximation (OA) solver for MINLP.

Implements the Duran-Grossmann (1986) / Fletcher-Leyffer (1994) algorithm
with extensions for feasibility cuts, equality relaxation, and ECP mode.

Decomposes MINLP into alternating NLP subproblems (with fixed integers)
and MILP master problems (with accumulated linearization cuts).

References:
    Duran & Grossmann, Math. Prog. 36, 1986. DOI: 10.1007/BF02592064
    Fletcher & Leyffer, Math. Prog. 66, 1994. DOI: 10.1007/BF01581153
    Viswanathan & Grossmann, C&CE 14(7), 1990. DOI: 10.1016/0098-1354(90)87085-4
    Westerlund & Pettersson, C&CE 19(S1), 1995. DOI: 10.1016/0098-1354(95)00164-W
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from discopt.modeling.core import Constraint, Model, ObjectiveSense, SolveResult, VarType

if TYPE_CHECKING:
    from discopt._jax.nlp_evaluator import NLPEvaluator

logger = logging.getLogger(__name__)

_INIT_STRATEGIES = frozenset({"rNLP", "initial_binary", "max_binary"})
_FEASIBILITY_NORMS = {
    "l1": "L1",
    "l2": "L2",
    "linfinity": "L_infinity",
    "l_infinity": "L_infinity",
    "l-inf": "L_infinity",
    "l_inf": "L_infinity",
}
_START_BOUND_CLIP = 1e8


def _normalize_init_strategy(init_strategy: str) -> str:
    """Normalize and validate the MindtPy-style initialization strategy."""
    if not isinstance(init_strategy, str):
        raise ValueError(f"init_strategy must be a string, got {type(init_strategy).__name__}.")
    key = init_strategy.strip().lower().replace("-", "_")
    if key == "rnlp":
        return "rNLP"
    if key in {"initial_binary", "max_binary"}:
        return key
    raise ValueError(
        f"Unknown init_strategy={init_strategy!r}. Choose one of: "
        + ", ".join(sorted(_INIT_STRATEGIES))
        + "."
    )


def _normalize_feasibility_norm(feasibility_norm: str) -> str:
    """Normalize and validate the MindtPy-style feasibility norm."""
    if not isinstance(feasibility_norm, str):
        raise ValueError(
            f"feasibility_norm must be a string, got {type(feasibility_norm).__name__}."
        )
    key = feasibility_norm.strip().lower().replace(" ", "_")
    normalized = _FEASIBILITY_NORMS.get(key)
    if normalized is not None:
        return normalized
    raise ValueError(
        f"Unknown feasibility_norm={feasibility_norm!r}. Choose one of: L1, L2, L_infinity."
    )


def _normalize_positive_float(name: str, value: float) -> float:
    """Validate a strictly positive finite float option."""
    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return out


def _normalize_optional_positive_int(name: str, value: Optional[int]) -> Optional[int]:
    """Validate a positive integer option, allowing None to disable it."""
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}.")
    return out


def _default_nlp_start(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Return the existing deterministic midpoint NLP start."""
    lb_clip = np.clip(np.asarray(lb, dtype=np.float64), -_START_BOUND_CLIP, _START_BOUND_CLIP)
    ub_clip = np.clip(np.asarray(ub, dtype=np.float64), -_START_BOUND_CLIP, _START_BOUND_CLIP)
    return np.asarray(0.5 * (lb_clip + ub_clip), dtype=np.float64)


def _round_integral_to_bounds(value: float, lb: float, ub: float) -> float:
    """Round half-up, then clamp to the nearest integer-compatible bounds."""
    rounded = float(np.floor(float(value) + 0.5))
    lo = float(np.ceil(lb))
    hi = float(np.floor(ub))
    if lo <= hi:
        return float(np.clip(rounded, lo, hi))
    return float(np.clip(rounded, lb, ub))


def _max_integral_seed(lb: float, ub: float, fallback: float) -> float:
    """Largest practical integer seed; fallback handles effectively unbounded uppers."""
    hi = float(np.floor(ub))
    lo = float(np.ceil(lb))
    if lo <= hi and np.isfinite(hi) and abs(hi) <= _START_BOUND_CLIP:
        return hi
    return _round_integral_to_bounds(fallback, lb, ub)


def _build_initial_strategy_point(
    decomp: _DecomposedProblem,
    init_strategy: str,
    initial_point: Optional[np.ndarray],
) -> np.ndarray:
    """Build the deterministic fixed-integer seed for non-rNLP strategies.

    ``initial_binary`` starts from the user/model start when supplied and rounds
    discrete variables half-up after bound clamping. ``max_binary`` activates
    binary variables at their largest feasible value; for general integers it
    uses the largest practical finite upper-bound value, falling back to the
    rounded clipped midpoint when the upper bound is effectively unbounded.
    """
    x_seed = _default_nlp_start(decomp.lb, decomp.ub)
    if initial_point is not None:
        x0 = np.asarray(initial_point, dtype=np.float64)
        if x0.shape != (decomp.n_vars,):
            raise ValueError(
                f"initial_point has shape {x0.shape}; expected ({decomp.n_vars},) "
                "for MIP-NLP initialization."
            )
        x_seed = np.clip(x0, decomp.lb, decomp.ub)

    if init_strategy == "initial_binary":
        for idx in decomp.int_indices:
            x_seed[idx] = _round_integral_to_bounds(x_seed[idx], decomp.lb[idx], decomp.ub[idx])
        return x_seed

    if init_strategy == "max_binary":
        midpoint = _default_nlp_start(decomp.lb, decomp.ub)
        for idx in decomp.binary_indices:
            x_seed[idx] = _max_integral_seed(decomp.lb[idx], decomp.ub[idx], fallback=1.0)
        for idx in decomp.general_integer_indices:
            x_seed[idx] = _max_integral_seed(
                decomp.lb[idx],
                decomp.ub[idx],
                fallback=midpoint[idx],
            )
        return x_seed

    raise ValueError(
        f"Internal error: non-rNLP initializer received init_strategy={init_strategy!r}."
    )


# ── Configuration ──────────────────────────────────────────────


@dataclass
class OAConfig:
    """Configuration for the OA solver."""

    time_limit: float = 3600.0
    gap_tolerance: float = 1e-4
    max_iterations: int = 100
    nlp_solver: str = "ipm"
    equality_relaxation: bool = False
    ecp_mode: bool = False
    feasibility_cuts: bool = True
    heuristic_nonconvex: bool = False
    add_slack: bool = False
    max_slack: float = 1000.0
    oa_penalty_factor: float = 1000.0
    add_no_good_cuts: bool = False
    feasibility_norm: str = "L_infinity"
    stalling_limit: Optional[int] = None
    cycling_check: bool = False
    log_iterations: bool = True


# ── Problem Decomposition ─────────────────────────────────────


@dataclass
class _DecomposedProblem:
    """Pre-processed model split into linear and nonlinear parts."""

    evaluator: "NLPEvaluator"
    n_vars: int
    n_cons: int
    lb: np.ndarray
    ub: np.ndarray
    int_indices: list[int]
    binary_indices: list[int]
    general_integer_indices: list[int]
    integrality: np.ndarray
    linear_A_rows: list[np.ndarray]
    linear_b_rows: list[float]
    linear_senses: list[str]
    nonlinear_indices: list[int]
    constraint_senses: list[str]
    obj_coeffs: Optional[tuple] = None
    obj_is_linear: bool = False
    oa_objective_is_convex: bool = True
    oa_constraint_mask: Optional[list[bool]] = None
    master_bound_valid: bool = True
    model: Optional[Model] = None


def _decompose_model(model: Model) -> _DecomposedProblem:
    """Separate model into linear/nonlinear constraints, identify integers."""
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.gdp_reformulate import _extract_body_coeffs, _is_linear
    from discopt._jax.nlp_evaluator import NLPEvaluator

    evaluator = NLPEvaluator(model)
    oa_convexity = classify_oa_cut_convexity(model)
    n_vars = evaluator.n_variables
    n_cons = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds

    # Identify integer/binary variable indices
    int_indices = []
    binary_indices = []
    general_integer_indices = []
    offset = 0
    for v in model._variables:
        if v.var_type == VarType.BINARY:
            for i in range(v.size):
                idx = offset + i
                int_indices.append(idx)
                binary_indices.append(idx)
        elif v.var_type == VarType.INTEGER:
            for i in range(v.size):
                idx = offset + i
                int_indices.append(idx)
                general_integer_indices.append(idx)
        offset += v.size

    integrality = np.zeros(n_vars, dtype=np.int32)
    for idx in int_indices:
        integrality[idx] = 1

    # Classify constraints as linear or nonlinear
    linear_A_rows = []
    linear_b_rows = []
    linear_senses = []
    nonlinear_indices = []

    # Track senses for ALL constraints in evaluator order (nonlinear only)
    all_constraint_senses = []
    eval_idx = 0  # tracks position in evaluator's stacked constraints

    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        if _is_linear(c.body):
            coeffs = _extract_body_coeffs(c.body, model, n_vars)
            if coeffs is not None:
                c_vec, off = coeffs
                linear_A_rows.append(c_vec)
                linear_b_rows.append(-off)
                linear_senses.append(c.sense)
            else:
                nonlinear_indices.append(eval_idx)
        else:
            nonlinear_indices.append(eval_idx)
        all_constraint_senses.append(c.sense)
        eval_idx += 1

    # Check if objective is linear
    raw_obj = model._objective
    obj_coeffs = (
        _extract_body_coeffs(raw_obj.expression, model, n_vars) if raw_obj is not None else None
    )
    obj_is_linear = obj_coeffs is not None
    # The NLPEvaluator works in minimization convention: it negates a MAXIMIZE
    # objective, so the NLP subproblems and the epigraph objective OA cuts all
    # optimize ``-f``. Put the *linear* master objective in the same convention.
    # Without this the master MILP minimizes ``+f`` while the subproblems maximize
    # it, and OA converges to — and certifies as optimal — a wrong point
    # (e.g. syn05m: returned -831 as "optimal" vs the true maximum 837.73).
    if obj_coeffs is not None and raw_obj is not None and raw_obj.sense == ObjectiveSense.MAXIMIZE:
        _c_vec, _c_off = obj_coeffs
        obj_coeffs = (-_c_vec, -_c_off)

    return _DecomposedProblem(
        evaluator=evaluator,
        n_vars=n_vars,
        n_cons=n_cons,
        lb=lb,
        ub=ub,
        int_indices=int_indices,
        binary_indices=binary_indices,
        general_integer_indices=general_integer_indices,
        integrality=integrality,
        linear_A_rows=linear_A_rows,
        linear_b_rows=linear_b_rows,
        linear_senses=linear_senses,
        nonlinear_indices=nonlinear_indices,
        constraint_senses=all_constraint_senses,
        obj_coeffs=obj_coeffs,
        obj_is_linear=obj_is_linear,
        oa_objective_is_convex=oa_convexity.objective_is_convex,
        oa_constraint_mask=oa_convexity.constraint_mask,
        master_bound_valid=(obj_is_linear or oa_convexity.objective_is_convex),
        model=model,
    )


# ── Bounds Proxy ──────────────────────────────────────────────


class _BoundsProxy:
    """Wraps an NLPEvaluator with overridden variable bounds.

    Forwards all attribute access to the underlying evaluator except
    for variable_bounds which returns the overridden bounds.
    """

    def __init__(self, evaluator, new_lb, new_ub):
        self._eval = evaluator
        self._lb = np.asarray(new_lb, dtype=np.float64)
        self._ub = np.asarray(new_ub, dtype=np.float64)

    def __getattr__(self, name):
        # Forward anything not found on self to the underlying evaluator
        return getattr(self._eval, name)

    @property
    def variable_bounds(self):
        return self._lb, self._ub


# ── NLP Subproblem Solvers ────────────────────────────────────


def _is_primal_feasible(evaluator, x, tol: float = 1e-4) -> bool:
    """Return True if x satisfies all constraints within tol."""
    if evaluator.n_constraints == 0:
        return True
    try:
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        cl, cu = _infer_constraint_bounds(evaluator._model)
        cons = np.asarray(evaluator.evaluate_constraints(x))
        return bool(np.all(cons >= cl - tol) and np.all(cons <= cu + tol))
    except Exception:
        return False


def _solve_nlp(evaluator, lb, ub, nlp_solver: str, max_iter: int = 200, x0=None):
    """Solve an NLP with given bounds. Returns (x, obj) or (None, None)."""
    if x0 is None:
        x0 = _default_nlp_start(lb, ub)
    else:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != (evaluator.n_variables,):
            raise ValueError(
                f"NLP initial point has shape {x0.shape}; expected ({evaluator.n_variables},)."
            )
        x0 = np.clip(x0, lb, ub)

    try:
        if nlp_solver == "ipopt":
            from discopt.solvers.nlp_ipopt import solve_nlp
        else:
            from discopt.solvers.nlp_pounce import solve_nlp

        result = solve_nlp(evaluator, x0, options={"print_level": 0, "max_iter": max_iter})

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL:
            return result.x, float(evaluator.evaluate_objective(result.x))

        # Accept iteration-limited results if the solution is primal feasible.
        # The IPM may not certify dual convergence (code 4: stalled) yet still
        # find a valid primal point, which is sufficient for OA linearization cuts.
        if result.status == SolveStatus.ITERATION_LIMIT and result.x is not None:
            if _is_primal_feasible(evaluator, result.x):
                return result.x, float(evaluator.evaluate_objective(result.x))
    except Exception:
        pass
    return None, None


def _solve_nlp_relaxation(evaluator, lb, ub, nlp_solver: str, initial_point=None):
    """Solve the continuous NLP relaxation (all integers relaxed)."""
    return _solve_nlp(evaluator, lb, ub, nlp_solver, x0=initial_point)


def _solve_nlp_subproblem(evaluator, lb, ub, int_indices, x_master, nlp_solver, initial_point=None):
    """Fix integers at master values and solve NLP subproblem."""
    sub_lb = lb.copy()
    sub_ub = ub.copy()
    for idx in int_indices:
        val = _round_integral_to_bounds(x_master[idx], lb[idx], ub[idx])
        sub_lb[idx] = val
        sub_ub[idx] = val

    proxy = _BoundsProxy(evaluator, sub_lb, sub_ub)
    return _solve_nlp(proxy, sub_lb, sub_ub, nlp_solver, x0=initial_point)


def _constraint_violation_data(evaluator, x) -> tuple[np.ndarray, np.ndarray]:
    """Return nonnegative row violations and active derivative signs."""
    if evaluator.n_constraints == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    vals = np.asarray(evaluator.evaluate_constraints(x), dtype=np.float64)
    cl, cu = _infer_constraint_bounds(evaluator)
    lower = np.zeros_like(vals)
    upper = np.zeros_like(vals)

    finite_lb = cl > -1e19
    finite_ub = cu < 1e19
    lower[finite_lb] = np.maximum(cl[finite_lb] - vals[finite_lb], 0.0)
    upper[finite_ub] = np.maximum(vals[finite_ub] - cu[finite_ub], 0.0)

    use_upper = upper >= lower
    violations = np.where(use_upper, upper, lower)
    signs = np.where(violations > 0, np.where(use_upper, 1.0, -1.0), 0.0)
    return violations, signs


def _constraint_violation_merit(evaluator, x, feasibility_norm: str) -> float:
    """Compute the selected feasibility violation merit at ``x``."""
    violations, _signs = _constraint_violation_data(evaluator, x)
    if violations.size == 0:
        return 0.0
    if feasibility_norm == "L1":
        return float(np.sum(violations))
    if feasibility_norm == "L2":
        return float(np.dot(violations, violations))
    return float(np.max(violations))


class _FeasibilityEvaluator:
    """Bounds-only NLP evaluator that minimizes constraint violation merit."""

    def __init__(self, evaluator, lb, ub, feasibility_norm: str):
        self._eval = evaluator
        self._lb = np.asarray(lb, dtype=np.float64)
        self._ub = np.asarray(ub, dtype=np.float64)
        self._feasibility_norm = feasibility_norm

    @property
    def n_variables(self):
        return self._eval.n_variables

    @property
    def n_constraints(self):
        return 0

    @property
    def variable_bounds(self):
        return self._lb, self._ub

    def evaluate_objective(self, x):
        return _constraint_violation_merit(self._eval, x, self._feasibility_norm)

    def evaluate_gradient(self, x):
        violations, signs = _constraint_violation_data(self._eval, x)
        if violations.size == 0 or np.all(violations <= 0):
            return np.zeros(self.n_variables, dtype=np.float64)

        try:
            jac = np.asarray(self._eval.evaluate_jacobian(x), dtype=np.float64)
        except Exception:
            return np.zeros(self.n_variables, dtype=np.float64)

        if self._feasibility_norm == "L1":
            weights = signs
        elif self._feasibility_norm == "L2":
            weights = 2.0 * violations * signs
        else:
            weights = np.zeros_like(violations)
            weights[int(np.argmax(violations))] = signs[int(np.argmax(violations))]
        return np.asarray(weights @ jac, dtype=np.float64)

    def evaluate_hessian(self, x):
        return np.zeros((self.n_variables, self.n_variables), dtype=np.float64)

    def evaluate_lagrangian_hessian(self, x, obj_factor, lagrange):
        return np.zeros((self.n_variables, self.n_variables), dtype=np.float64)

    def evaluate_constraints(self, x):
        return np.empty(0, dtype=np.float64)

    def evaluate_jacobian(self, x):
        return np.empty((0, self.n_variables), dtype=np.float64)


def _solve_feasibility_subproblem(
    evaluator,
    lb,
    ub,
    int_indices,
    x_master,
    nlp_solver,
    feasibility_norm,
):
    """Solve feasibility problem with fixed integers.

    Minimizes the selected violation norm over the continuous variables with
    the master integer assignment fixed. If that bounded feasibility NLP cannot
    improve the master point, return the clipped master point so OA can still
    generate cuts deterministically.
    """
    sub_lb = lb.copy()
    sub_ub = ub.copy()
    for idx in int_indices:
        val = _round_integral_to_bounds(x_master[idx], lb[idx], ub[idx])
        sub_lb[idx] = val
        sub_ub[idx] = val

    x0 = np.clip(x_master[: evaluator.n_variables], sub_lb, sub_ub)
    best_x = x0
    best_merit = _constraint_violation_merit(evaluator, x0, feasibility_norm)

    try:
        proxy = _FeasibilityEvaluator(evaluator, sub_lb, sub_ub, feasibility_norm)
        x_feas, _obj_feas = _solve_nlp(proxy, sub_lb, sub_ub, nlp_solver, x0=x0)
        if x_feas is not None:
            candidate = np.clip(np.asarray(x_feas, dtype=np.float64), sub_lb, sub_ub)
            candidate_merit = _constraint_violation_merit(evaluator, candidate, feasibility_norm)
            if candidate_merit <= best_merit + 1e-9:
                best_x = candidate
    except Exception:
        pass

    return best_x


# ── Cut Generation ────────────────────────────────────────────


def _append_master_cut(
    oa_A_rows,
    oa_b_rows,
    coeffs,
    rhs,
    oa_cut_relaxable=None,
    relaxable=True,
):
    """Append a master cut and optional slack-relaxability metadata."""
    oa_A_rows.append(coeffs)
    oa_b_rows.append(rhs)
    if oa_cut_relaxable is not None:
        oa_cut_relaxable.append(bool(relaxable))


def _add_oa_cuts(
    evaluator,
    x_star,
    n_vars,
    n_cons,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    obj_is_linear,
    constraint_convex_mask,
    objective_is_convex,
    equality_relaxation=False,
    oa_cut_relaxable=None,
):
    """Generate OA cuts at x_star and append to cut lists.

    Constraint cuts have length n_vars.
    Objective cuts (when nonlinear) have length n_vars+1, with the last
    element being the -eta epigraph coefficient.
    """
    from discopt._jax.cutting_planes import (
        generate_oa_cuts_from_evaluator,
        generate_objective_oa_cut,
    )

    if n_cons > 0:
        cuts = generate_oa_cuts_from_evaluator(
            evaluator,
            x_star,
            constraint_senses=constraint_senses,
            convex_mask=constraint_convex_mask,
        )
        for cut in cuts:
            coeffs = cut.coeffs.copy()
            # Filter degenerate cuts
            if np.linalg.norm(coeffs) < 1e-12:
                continue

            sense = cut.sense
            if equality_relaxation and sense == "==":
                sense = "<="

            if sense == "<=":
                _append_master_cut(oa_A_rows, oa_b_rows, coeffs, cut.rhs, oa_cut_relaxable)
            elif sense == ">=":
                _append_master_cut(oa_A_rows, oa_b_rows, -coeffs, -cut.rhs, oa_cut_relaxable)
            elif sense == "==":
                # Equality: add both <= and >= cuts
                _append_master_cut(oa_A_rows, oa_b_rows, coeffs, cut.rhs, oa_cut_relaxable)
                _append_master_cut(oa_A_rows, oa_b_rows, -coeffs, -cut.rhs, oa_cut_relaxable)

    # Objective OA cut (only if nonlinear): grad^T x - eta <= rhs
    if not obj_is_linear and objective_is_convex:
        n_master = n_vars + 1
        obj_cut = generate_objective_oa_cut(evaluator, x_star, n_master, z_index=n_vars)
        _append_master_cut(
            oa_A_rows,
            oa_b_rows,
            obj_cut.coeffs.copy(),
            obj_cut.rhs,
            oa_cut_relaxable,
            relaxable=False,
        )


def _add_ecp_cuts(
    evaluator,
    x_master,
    n_vars,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    obj_is_linear,
    constraint_convex_mask,
    objective_is_convex,
    equality_relaxation=False,
    oa_cut_relaxable=None,
):
    """Generate ECP cuts: OA cuts only for violated constraints at x_master."""
    from discopt._jax.cutting_planes import (
        generate_objective_oa_cut,
        separate_oa_cuts,
    )

    n_added = 0
    if evaluator.n_constraints > 0:
        cuts = separate_oa_cuts(
            evaluator,
            x_master,
            constraint_senses=constraint_senses,
            convex_mask=constraint_convex_mask,
        )
        for cut in cuts:
            coeffs = cut.coeffs.copy()
            if np.linalg.norm(coeffs) < 1e-12:
                continue

            sense = cut.sense
            if equality_relaxation and sense == "==":
                sense = "<="

            if sense == "<=":
                _append_master_cut(oa_A_rows, oa_b_rows, coeffs, cut.rhs, oa_cut_relaxable)
                n_added += 1
            elif sense == ">=":
                _append_master_cut(oa_A_rows, oa_b_rows, -coeffs, -cut.rhs, oa_cut_relaxable)
                n_added += 1
            elif sense == "==":
                _append_master_cut(oa_A_rows, oa_b_rows, coeffs, cut.rhs, oa_cut_relaxable)
                _append_master_cut(oa_A_rows, oa_b_rows, -coeffs, -cut.rhs, oa_cut_relaxable)
                n_added += 2

    if not obj_is_linear and objective_is_convex:
        n_master = n_vars + 1
        obj_cut = generate_objective_oa_cut(evaluator, x_master, n_master, z_index=n_vars)
        _append_master_cut(
            oa_A_rows,
            oa_b_rows,
            obj_cut.coeffs.copy(),
            obj_cut.rhs,
            oa_cut_relaxable,
            relaxable=False,
        )
        n_added += 1

    return n_added


def _add_no_good_cut(
    x_master,
    int_indices,
    oa_A_rows,
    oa_b_rows,
    n_vars,
    oa_cut_relaxable=None,
):
    """Add an integer-exclusion (no-good) cut.

    sum_{i: y_i*=1} (1-y_i) + sum_{i: y_i*=0} y_i >= 1
    Equivalently in <= form:
    sum_{y_i*=1} y_i - sum_{y_i*=0} y_i <= count(y_i*=1) - 1
    """
    coeffs = np.zeros(n_vars)
    count_ones = 0
    for idx in int_indices:
        val = _round_integral_to_bounds(x_master[idx], 0.0, 1.0)
        if val >= 0.5:
            coeffs[idx] = 1.0
            count_ones += 1
        else:
            coeffs[idx] = -1.0
    _append_master_cut(
        oa_A_rows,
        oa_b_rows,
        coeffs,
        float(count_ones - 1),
        oa_cut_relaxable,
        relaxable=False,
    )


def _add_feasibility_cuts(
    evaluator,
    x_feas,
    n_vars,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    constraint_convex_mask,
    oa_cut_relaxable=None,
):
    """Add gradient-based feasibility cuts (Fletcher-Leyffer 1994).

    For each violated constraint g_k(x) <= 0 at x_feas:
        g_k(x_feas) + nabla g_k(x_feas)^T (x - x_feas) <= 0
    """
    from discopt._jax.cutting_planes import separate_oa_cuts

    if evaluator.n_constraints == 0:
        return

    cuts = separate_oa_cuts(
        evaluator,
        x_feas,
        constraint_senses=constraint_senses,
        convex_mask=constraint_convex_mask,
    )
    for cut in cuts:
        coeffs = cut.coeffs.copy()
        if np.linalg.norm(coeffs) < 1e-12:
            continue
        # Feasibility cuts are gradient cuts, not hard integer exclusions, so
        # the shared OA slack may relax them to keep the heuristic master robust.
        if cut.sense == "<=":
            _append_master_cut(oa_A_rows, oa_b_rows, coeffs, cut.rhs, oa_cut_relaxable)
        elif cut.sense == ">=":
            _append_master_cut(oa_A_rows, oa_b_rows, -coeffs, -cut.rhs, oa_cut_relaxable)


# ── MILP Master Problem ──────────────────────────────────────


def _solve_master_milp(
    linear_A_rows,
    linear_b_rows,
    linear_senses,
    oa_A_rows,
    oa_b_rows,
    n_vars,
    integrality,
    lb,
    ub,
    obj_coeffs,
    obj_is_linear,
    objective_bound_valid,
    time_limit,
    gap_tolerance,
    add_slack=False,
    max_slack=1000.0,
    oa_penalty_factor=1000.0,
    oa_cut_relaxable=None,
    use_objective_epigraph=None,
):
    """Build and solve the master MILP."""
    try:
        from discopt.solvers.lp_backend import get_milp_solver

        # HiGHS if present, else POUNCE (self-hosted B&B) — HiGHS-free path.
        solve_milp = get_milp_solver()
    except ImportError as e:
        raise ImportError(
            "OA solver requires a MILP backend for the master. Install one of: "
            "pip install highspy  |  pip install pounce-solver"
        ) from e

    # ``use_objective_epigraph`` controls master layout when supplied; the
    # certification flag remains only as the compatibility fallback.
    if use_objective_epigraph is None:
        use_objective_epigraph = (not obj_is_linear) and objective_bound_valid
    if oa_cut_relaxable is not None and len(oa_cut_relaxable) != len(oa_A_rows):
        raise ValueError(
            "oa_cut_relaxable must match oa_A_rows length; "
            f"got {len(oa_cut_relaxable)} flags for {len(oa_A_rows)} cuts."
        )
    n_master = n_vars
    if use_objective_epigraph:
        n_master += 1  # epigraph variable eta
    slack_index = None
    if add_slack:
        # A single shared slack intentionally keeps the master compact. It is a
        # MindtPy-inspired heuristic simplification, not a per-cut slack model.
        slack_index = n_master
        n_master += 1

    # Build A_ub, b_ub from linear <= constraints + OA cuts
    A_ub_rows = []
    b_ub_vals = []

    for i, sense in enumerate(linear_senses):
        row = linear_A_rows[i]
        if use_objective_epigraph:
            row = np.append(row, 0.0)
        if add_slack:
            row = np.append(row, 0.0)
        if sense == "<=":
            A_ub_rows.append(row)
            b_ub_vals.append(linear_b_rows[i])
        elif sense == ">=":
            A_ub_rows.append(-row)
            b_ub_vals.append(-linear_b_rows[i])

    # OA cuts (all in <= form already)
    # Constraint cuts have length n_vars; objective cuts carry the eta column.
    for i in range(len(oa_A_rows)):
        row = np.asarray(oa_A_rows[i], dtype=np.float64)
        original_len = len(row)
        if use_objective_epigraph and len(row) == n_vars:
            row = np.append(row, 0.0)  # extend constraint cuts with 0 for eta
        if add_slack:
            # Relax only constraint OA/feasibility cuts. Objective epigraph cuts
            # and hard integer-exclusion cuts must remain unrelaxed.
            if oa_cut_relaxable is None:
                relax_cut = original_len == n_vars
            else:
                relax_cut = bool(oa_cut_relaxable[i])
            slack_coeff = -1.0 if relax_cut else 0.0
            row = np.append(row, slack_coeff)
        A_ub_rows.append(row)
        b_ub_vals.append(oa_b_rows[i])

    # Equality constraints from linear
    A_eq_rows = []
    b_eq_vals = []
    for i, sense in enumerate(linear_senses):
        if sense == "==":
            row = linear_A_rows[i]
            if use_objective_epigraph:
                row = np.append(row, 0.0)
            if add_slack:
                row = np.append(row, 0.0)
            A_eq_rows.append(row)
            b_eq_vals.append(linear_b_rows[i])

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_vals) if b_eq_vals else None

    # Objective
    c = np.zeros(n_master)
    if obj_is_linear:
        c_vec, _off = obj_coeffs
        c[:n_vars] = c_vec
    elif use_objective_epigraph:
        c[n_vars] = 1.0  # minimize eta
    if slack_index is not None:
        c[slack_index] = oa_penalty_factor

    # Bounds
    bounds_list = list(zip(lb.tolist(), ub.tolist()))
    if use_objective_epigraph:
        bounds_list.append((-1e20, 1e20))  # eta unbounded
    if slack_index is not None:
        bounds_list.append((0.0, max_slack))

    # Integrality
    int_vec = np.zeros(n_master, dtype=np.int32)
    int_vec[:n_vars] = integrality

    return solve_milp(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds_list,
        integrality=int_vec,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
    )


# ── Result Construction ───────────────────────────────────────


def _build_x_dict(x_flat: np.ndarray, model: Model) -> dict:
    """Convert flat solution vector to {var_name: value} dict."""
    result = {}
    offset = 0
    for v in model._variables:
        result[v.name] = x_flat[offset : offset + v.size].reshape(v.shape)
        offset += v.size
    return result


def _compute_gap(lb: float, ub: float) -> float:
    if ub >= 1e19 or lb <= -1e19:
        return 1.0
    abs_gap = max(0.0, ub - lb)
    if abs_gap <= 1e-9:
        return 0.0
    denom = max(abs(ub), abs(lb), 1e-10)
    return abs_gap / denom


# ── Main Algorithm ────────────────────────────────────────────


def solve_oa(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "ipm",
    init_strategy: str = "rNLP",
    initial_point: Optional[np.ndarray] = None,
    equality_relaxation: bool = False,
    ecp_mode: bool = False,
    feasibility_cuts: bool = True,
    heuristic_nonconvex: bool = False,
    add_slack: bool = False,
    max_slack: float = 1000.0,
    oa_penalty_factor: float = 1000.0,
    add_no_good_cuts: bool = False,
    feasibility_norm: str = "L_infinity",
    stalling_limit: Optional[int] = None,
    cycling_check: bool = False,
    **kwargs,
) -> SolveResult:
    """Solve a MINLP via Outer Approximation.

    Decomposes the problem into alternating NLP subproblems (continuous
    optimization with fixed integers) and MILP master problems (linear
    relaxation with accumulated OA cuts).

    Parameters
    ----------
    model : Model
        MINLP model with continuous, binary, and/or integer variables.
    time_limit : float
        Wall-clock time limit in seconds.
    gap_tolerance : float
        Relative optimality gap for convergence.
    max_iterations : int
        Maximum OA iterations.
    nlp_solver : str
        NLP backend: ``"ipm"``, ``"ipopt"``, ``"pounce"``.
    init_strategy : {"rNLP", "initial_binary", "max_binary"}
        Initialization strategy for the first master cuts and fixed-integer
        NLP seed. ``"rNLP"`` solves the continuous relaxation and generates
        cuts at that point. ``"initial_binary"`` rounds and clamps discrete
        variables from ``initial_point`` when supplied, otherwise from the
        deterministic midpoint start. ``"max_binary"`` starts binary variables
        at their largest feasible values; general integers use their largest
        practical finite upper-bound value, or the rounded clipped midpoint
        when no practical finite upper bound exists.
    initial_point : numpy.ndarray, optional
        Flat model start produced from ``Model.solve(initial_solution=...)``.
        Used to warm-start the continuous relaxation for ``init_strategy="rNLP"``,
        by ``init_strategy="initial_binary"``, and as the continuous part of
        ``"max_binary"``.
    equality_relaxation : bool
        Relax nonlinear equalities to inequalities in OA cuts
        (Viswanathan & Grossmann 1990). Helps when nonlinear equalities
        cause the MILP master to become infeasible.
    ecp_mode : bool
        Extended Cutting Plane mode (Westerlund & Pettersson 1995):
        skip NLP subproblems entirely, only add cuts at MILP master
        solutions for violated constraints. Simpler but slower convergence.
    feasibility_cuts : bool
        Use gradient-based feasibility cuts (Fletcher & Leyffer 1994)
        when the NLP subproblem is infeasible. Stronger than no-good cuts.
    heuristic_nonconvex : bool
        Enable MindtPy-style heuristic handling for nonconvex cases. This turns
        on equality relaxation and slack handling and suppresses certified
        bound/gap reporting.
    add_slack : bool
        Relax OA constraint cuts with one nonnegative master slack variable.
    max_slack : float
        Upper bound for the OA master slack variable.
    oa_penalty_factor : float
        Positive objective penalty applied to the OA master slack variable.
    add_no_good_cuts : bool
        Add integer-exclusion cuts after infeasible fixed-integer NLP solves.
    feasibility_norm : {"L1", "L2", "L_infinity"}
        Violation norm minimized by the feasibility subproblem heuristic.
    stalling_limit : int, optional
        Stop after this many consecutive incumbent-objective records without
        material progress.
    cycling_check : bool
        Stop when the master repeats a fixed-integer assignment.

    Returns
    -------
    SolveResult
    """
    t_start = time.perf_counter()
    init_strategy = _normalize_init_strategy(init_strategy)
    feasibility_norm = _normalize_feasibility_norm(feasibility_norm)
    max_slack = _normalize_positive_float("max_slack", max_slack)
    oa_penalty_factor = _normalize_positive_float("oa_penalty_factor", oa_penalty_factor)
    stalling_limit = _normalize_optional_positive_int("stalling_limit", stalling_limit)
    heuristic_nonconvex = bool(heuristic_nonconvex)
    if heuristic_nonconvex:
        equality_relaxation = True
        add_slack = True
    add_slack = bool(add_slack)
    add_no_good_cuts = bool(add_no_good_cuts)
    cycling_check = bool(cycling_check)

    # 1. Decompose model
    decomp = _decompose_model(model)
    evaluator = decomp.evaluator
    n_vars = decomp.n_vars
    n_cons = decomp.n_cons
    # The whole OA loop runs in the evaluator's minimization convention (it
    # negates a MAXIMIZE objective). Un-negate the user-facing objective/bound at
    # the return sites with this sign; the gap is convention-invariant.
    _obj_sign = (
        -1.0
        if (model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE)
        else 1.0
    )
    if decomp.oa_constraint_mask is not None and not all(decomp.oa_constraint_mask):
        logger.warning(
            "OA: generating OA cuts only for %d of %d constraints classified convex",
            sum(1 for is_convex in decomp.oa_constraint_mask if is_convex),
            len(decomp.oa_constraint_mask),
        )
    if not decomp.obj_is_linear and not decomp.oa_objective_is_convex:
        logger.warning(
            "OA: nonlinear objective is not convex in the optimization sense; "
            "disabling master lower-bound updates and skipping objective OA cuts"
        )
    master_bound_valid = decomp.master_bound_valid and not heuristic_nonconvex

    # If no integer variables, just solve the NLP directly
    if len(decomp.int_indices) == 0:
        x_sol, obj = _solve_nlp_relaxation(
            evaluator,
            decomp.lb,
            decomp.ub,
            nlp_solver,
            initial_point=initial_point,
        )
        wall_time = time.perf_counter() - t_start
        if x_sol is not None:
            return SolveResult(
                status="optimal",
                objective=_obj_sign * obj,
                bound=_obj_sign * obj,
                gap=0.0,
                x=_build_x_dict(x_sol, model),
                wall_time=wall_time,
            )
        return SolveResult(
            status="infeasible",
            objective=None,
            bound=None,
            gap=None,
            x={},
            wall_time=wall_time,
        )

    # 2. Generate initial linearization cuts.
    oa_A_rows: list[np.ndarray] = []
    oa_b_rows: list[float] = []
    oa_cut_relaxable: list[bool] = []

    UB = 1e20
    LB = -1e20
    incumbent = None
    incumbent_obj = None
    integer_assignments_seen: set[tuple[float, ...]] = set()
    incumbent_progress: list[float] = []
    termination_reason = None

    if init_strategy == "rNLP":
        x_relax, obj_relax = _solve_nlp_relaxation(
            evaluator,
            decomp.lb,
            decomp.ub,
            nlp_solver,
            initial_point=initial_point,
        )

        if x_relax is not None:
            _add_oa_cuts(
                evaluator,
                x_relax,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
            )
            # Check if relaxation solution is already integer-feasible.
            is_int_feasible = all(
                abs(x_relax[idx] - round(x_relax[idx])) < 1e-5 for idx in decomp.int_indices
            )
            if is_int_feasible and obj_relax is not None:
                UB = obj_relax
                incumbent = x_relax.copy()
                incumbent_obj = obj_relax
        else:
            # NLP relaxation failed; generate initial cuts at the deterministic midpoint.
            x_mid = _default_nlp_start(decomp.lb, decomp.ub)
            _add_oa_cuts(
                evaluator,
                x_mid,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
            )
    else:
        x_seed = _build_initial_strategy_point(decomp, init_strategy, initial_point)
        if ecp_mode:
            _add_oa_cuts(
                evaluator,
                x_seed,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
            )
            if _is_primal_feasible(evaluator, x_seed):
                UB = float(evaluator.evaluate_objective(x_seed))
                incumbent = x_seed.copy()
                incumbent_obj = UB
        else:
            x_init, obj_init = _solve_nlp_subproblem(
                evaluator,
                decomp.lb,
                decomp.ub,
                decomp.int_indices,
                x_seed,
                nlp_solver,
                initial_point=x_seed,
            )
            x_cut = x_init if x_init is not None else x_seed
            _add_oa_cuts(
                evaluator,
                x_cut,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
            )
            if x_init is not None and obj_init is not None:
                UB = obj_init
                incumbent = x_init.copy()
                incumbent_obj = obj_init

    # 3. Main OA loop
    for iteration in range(max_iterations):
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            logger.info("OA: Time limit reached at iteration %d", iteration)
            break

        # a. Solve master MILP
        master_result = _solve_master_milp(
            decomp.linear_A_rows,
            decomp.linear_b_rows,
            decomp.linear_senses,
            oa_A_rows,
            oa_b_rows,
            n_vars,
            decomp.integrality,
            decomp.lb,
            decomp.ub,
            decomp.obj_coeffs,
            decomp.obj_is_linear,
            master_bound_valid,
            time_limit=time_limit - elapsed,
            gap_tolerance=gap_tolerance,
            add_slack=add_slack,
            max_slack=max_slack,
            oa_penalty_factor=oa_penalty_factor,
            oa_cut_relaxable=oa_cut_relaxable,
            use_objective_epigraph=(not decomp.obj_is_linear and decomp.oa_objective_is_convex),
        )

        from discopt.solvers import SolveStatus

        if master_result is None:
            logger.info("OA: Master MILP failed at iteration %d", iteration)
            break

        if master_result.status == SolveStatus.INFEASIBLE:
            logger.info("OA: Master MILP infeasible at iteration %d", iteration)
            termination_reason = "master_infeasible"
            break

        if master_result.status == SolveStatus.UNBOUNDED or master_result.x is None:
            # Master unbounded → need more OA cuts. Generate at midpoint.
            logger.info("OA: Master MILP unbounded at iteration %d, adding cuts", iteration)
            lb_clip = np.clip(decomp.lb, -1e8, 1e8)
            ub_clip = np.clip(decomp.ub, -1e8, 1e8)
            x_mid = 0.5 * (lb_clip + ub_clip)
            _add_oa_cuts(
                evaluator,
                x_mid,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
            )
            continue

        x_master = master_result.x[:n_vars]
        if cycling_check:
            int_assignment = tuple(
                _round_integral_to_bounds(x_master[idx], decomp.lb[idx], decomp.ub[idx])
                for idx in decomp.int_indices
            )
            if int_assignment in integer_assignments_seen:
                logger.info(
                    "OA: cycling detected at iteration %d for integer assignment %s",
                    iteration,
                    int_assignment,
                )
                termination_reason = "cycling"
                break
            integer_assignments_seen.add(int_assignment)

        # The master gives a valid LB only via its dual ``bound`` (never the
        # incumbent ``objective``, which is an upper bound on a limited solve).
        if master_bound_valid and master_result.bound is not None:
            LB = max(LB, master_result.bound)

        # b. ECP mode: add cuts at master point, skip NLP
        if ecp_mode:
            n_violated = _add_ecp_cuts(
                evaluator,
                x_master,
                n_vars,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
            )
            # In ECP, use master objective as heuristic UB
            master_obj = float(evaluator.evaluate_objective(x_master))
            cons_vals = evaluator.evaluate_constraints(x_master)
            is_feasible = all(cons_vals[k] <= 1e-6 for k in range(n_cons))
            if is_feasible and master_obj < UB:
                UB = master_obj
                incumbent = x_master.copy()
                incumbent_obj = master_obj

            gap = _compute_gap(LB, UB)
            logger.info(
                "OA-ECP iter %d: LB=%.6f UB=%.6f gap=%.4f%% cuts=%d violated=%d",
                iteration,
                LB,
                UB,
                gap * 100,
                len(oa_A_rows),
                n_violated,
            )

            if n_violated == 0:
                termination_reason = "ecp_feasible"
                break
            if master_bound_valid and gap <= gap_tolerance:
                termination_reason = "gap"
                break
            continue

        # c. Fix integers, solve NLP subproblem
        x_nlp, obj_nlp = _solve_nlp_subproblem(
            evaluator,
            decomp.lb,
            decomp.ub,
            decomp.int_indices,
            x_master,
            nlp_solver,
        )

        if x_nlp is not None:
            if obj_nlp < UB:
                UB = obj_nlp
                incumbent = x_nlp.copy()
                incumbent_obj = obj_nlp

            # Generate OA cuts at NLP solution
            _add_oa_cuts(
                evaluator,
                x_nlp,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
            )
        else:
            # NLP infeasible for this integer assignment
            if feasibility_cuts:
                x_feas = _solve_feasibility_subproblem(
                    evaluator,
                    decomp.lb,
                    decomp.ub,
                    decomp.int_indices,
                    x_master,
                    nlp_solver,
                    feasibility_norm,
                )
                if x_feas is not None:
                    _add_feasibility_cuts(
                        evaluator,
                        x_feas,
                        n_vars,
                        decomp.constraint_senses,
                        oa_A_rows,
                        oa_b_rows,
                        decomp.oa_constraint_mask,
                        oa_cut_relaxable=oa_cut_relaxable,
                    )

            if add_no_good_cuts:
                _add_no_good_cut(
                    x_master,
                    decomp.int_indices,
                    oa_A_rows,
                    oa_b_rows,
                    n_vars,
                    oa_cut_relaxable=oa_cut_relaxable,
                )

            # Also add OA cuts at master point
            _add_oa_cuts(
                evaluator,
                x_master,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
                oa_cut_relaxable=oa_cut_relaxable,
            )

        # d. Check convergence
        gap = _compute_gap(LB, UB)
        logger.info(
            "OA iter %d: LB=%.6f UB=%.6f gap=%.4f%% cuts=%d",
            iteration,
            LB,
            UB,
            gap * 100,
            len(oa_A_rows),
        )

        if incumbent_obj is not None:
            incumbent_progress.append(float(UB))
            if stalling_limit is not None and len(incumbent_progress) >= stalling_limit:
                prev = incumbent_progress[-stalling_limit]
                if abs(incumbent_progress[-1] - prev) <= 1e-12:
                    logger.info(
                        "OA: stalling detected after %d incumbent records; best objective %.6f",
                        stalling_limit,
                        UB,
                    )
                    termination_reason = "stalling"
                    break

        if master_bound_valid and gap <= gap_tolerance:
            termination_reason = "gap"
            break

    # 4. Build result
    wall_time = time.perf_counter() - t_start
    gap = _compute_gap(LB, UB)
    bound_certified = master_bound_valid
    bound = LB if bound_certified and LB > -1e19 else None
    reported_gap = gap if bound is not None and UB < 1e19 else None

    if incumbent is not None and incumbent_obj is not None:
        status = "optimal" if bound_certified and gap <= gap_tolerance else "feasible"
        if termination_reason in {"cycling", "stalling"}:
            status = "feasible"
        return SolveResult(
            status=status,
            objective=_obj_sign * incumbent_obj,
            bound=(_obj_sign * bound if bound is not None else None),
            gap=reported_gap,
            x=_build_x_dict(incumbent, model),
            wall_time=wall_time,
        )

    return SolveResult(
        status="infeasible",
        objective=None,
        bound=(_obj_sign * bound if bound is not None else None),
        gap=None,
        x={},
        wall_time=wall_time,
    )
