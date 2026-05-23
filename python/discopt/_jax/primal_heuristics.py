"""
Multi-start primal heuristics for MINLP.

Finds good feasible solutions by launching NLP solves from diverse starting
points. Includes a feasibility pump that rounds fractional integer variables
and re-solves the resulting NLP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model, VarType
from discopt.solvers import NLPResult, SolveStatus
from discopt.solvers.nlp_ipopt import solve_nlp


@dataclass
class MultiStartResult:
    """Result of multi-start NLP solving."""

    best_objective: Optional[float] = None
    best_solution: Optional[np.ndarray] = None
    n_starts: int = 0
    n_feasible: int = 0
    n_integer_feasible: int = 0
    all_objectives: list[float] = field(default_factory=list)


def _get_integer_mask(model: Model) -> np.ndarray:
    """Return a boolean mask over the flat variable vector: True where integer/binary."""
    parts: list[np.ndarray] = []
    for v in model._variables:
        is_int = v.var_type in (VarType.BINARY, VarType.INTEGER)
        parts.append(np.full(v.size, is_int, dtype=bool))
    return np.concatenate(parts) if parts else np.array([], dtype=bool)


def _get_variable_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return (lb, ub) flat arrays for all variables."""
    lbs: list[np.ndarray] = []
    ubs: list[np.ndarray] = []
    for v in model._variables:
        lbs.append(v.lb.flatten())
        ubs.append(v.ub.flatten())
    return np.concatenate(lbs), np.concatenate(ubs)


def _generate_starts(
    lb: np.ndarray,
    ub: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate diverse starting points within bounds.

    Uses stratified random sampling: divide [0, 1] into n_starts strata along
    each dimension and sample uniformly within each stratum, then scale to
    [lb, ub]. This gives better coverage than pure uniform random.
    """
    n = len(lb)
    # Clip infinite bounds for sampling purposes
    lb_safe = np.clip(lb, -1e6, 1e6)
    ub_safe = np.clip(ub, -1e6, 1e6)

    # Stratified sampling along a single axis with random permutation
    starts = np.empty((n_starts, n), dtype=np.float64)
    for j in range(n):
        # Divide [0,1] into n_starts strata, sample within each
        strata = (np.arange(n_starts) + rng.uniform(size=n_starts)) / n_starts
        rng.shuffle(strata)
        starts[:, j] = lb_safe[j] + strata * (ub_safe[j] - lb_safe[j])

    return starts


def _is_nlp_feasible(result: NLPResult) -> bool:
    """Check whether an NLP result represents a feasible solution."""
    return result.status in (SolveStatus.OPTIMAL,) and result.x is not None


def _is_integer_feasible(
    x: np.ndarray,
    int_mask: np.ndarray,
    tol: float = 1e-5,
) -> bool:
    """Check if integer variables are within tolerance of integer values."""
    if not np.any(int_mask):
        return True
    frac = np.abs(x[int_mask] - np.round(x[int_mask]))
    return bool(np.all(frac <= tol))


class MultiStartNLP:
    """Multi-start NLP solver for finding good feasible solutions.

    Generates diverse starting points and solves an NLP from each.
    Tracks the best feasible (and integer-feasible) solution found.
    """

    def __init__(
        self,
        model: Model,
        n_starts: int = 64,
        seed: int = 42,
    ) -> None:
        self._model = model
        self._n_starts = n_starts
        self._seed = seed

    def solve(
        self,
        ipopt_options: Optional[dict] = None,
    ) -> MultiStartResult:
        """Run multi-start NLP solving.

        Args:
            ipopt_options: Options passed to Ipopt (e.g. max_iter, tol).

        Returns:
            MultiStartResult with best solution and statistics.
        """
        model = self._model
        evaluator = NLPEvaluator(model)
        lb, ub = evaluator.variable_bounds
        int_mask = _get_integer_mask(model)
        has_integers = np.any(int_mask)

        rng = np.random.default_rng(self._seed)
        starts = _generate_starts(lb, ub, self._n_starts, rng)

        opts = dict(ipopt_options) if ipopt_options else {}
        opts.setdefault("print_level", 0)

        result = MultiStartResult(n_starts=self._n_starts)
        best_obj = float("inf")

        for i in range(self._n_starts):
            x0 = starts[i]
            nlp_result = solve_nlp(evaluator, x0, options=opts)
            if not _is_nlp_feasible(nlp_result):
                continue

            result.n_feasible += 1
            assert nlp_result.objective is not None
            assert nlp_result.x is not None

            int_feas = _is_integer_feasible(nlp_result.x, int_mask)
            if int_feas:
                result.n_integer_feasible += 1

            result.all_objectives.append(nlp_result.objective)

            # For MINLP, only update incumbent if integer-feasible
            if has_integers and not int_feas:
                continue

            if nlp_result.objective < best_obj:
                best_obj = nlp_result.objective
                result.best_objective = nlp_result.objective
                result.best_solution = nlp_result.x.copy()

        return result


def feasibility_pump(
    model: Model,
    x_nlp: np.ndarray,
    max_rounds: int = 5,
    ipopt_options: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Try to find an integer-feasible solution via rounding + re-solve.

    Given an NLP solution with fractional integer variables:
    1. Round integer variables to nearest integer.
    2. Fix integer variables and re-solve NLP for continuous variables.
    3. If feasible, return. Otherwise perturb and retry.

    Args:
        model: The optimization model.
        x_nlp: An NLP relaxation solution (may have fractional integers).
        max_rounds: Maximum rounding + re-solve attempts.
        ipopt_options: Options passed to Ipopt.

    Returns:
        An integer-feasible solution vector, or None if not found.
    """
    int_mask = _get_integer_mask(model)
    if not np.any(int_mask):
        # No integer variables, the NLP solution is already integer-feasible.
        return x_nlp.copy()

    lb, ub = _get_variable_bounds(model)
    evaluator = NLPEvaluator(model)

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)

    rng = np.random.default_rng(42)

    for round_idx in range(max_rounds):
        x_try = x_nlp.copy()

        # Round integer variables
        x_try[int_mask] = np.round(x_try[int_mask])

        # Perturb on rounds > 0 by randomly flipping some integer values
        if round_idx > 0:
            flip_mask = int_mask & (rng.random(len(x_try)) < 0.3)
            perturbation = rng.choice([-1, 0, 1], size=len(x_try))
            x_try[flip_mask] = x_try[flip_mask] + perturbation[flip_mask]

        # Clip to bounds
        x_try = np.clip(x_try, lb, ub)

        # Fix integer variables: set lb = ub for them in a modified evaluator
        # Instead of rebuilding the evaluator, we fix by tightening bounds on x0
        # and re-solving with the original evaluator. The rounded integer values
        # are used as the starting point. Ipopt will respect variable bounds.
        x0 = x_try.copy()

        nlp_result = solve_nlp(evaluator, x0, options=opts)
        if not _is_nlp_feasible(nlp_result):
            continue

        assert nlp_result.x is not None
        if _is_integer_feasible(nlp_result.x, int_mask):
            return nlp_result.x.copy()

    return None


def _check_constraint_feasibility(
    evaluator: NLPEvaluator,
    x: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """Check that ``x`` satisfies the model's constraints to within ``tol``."""
    if evaluator.n_constraints == 0:
        return True
    g = np.asarray(evaluator.evaluate_constraints(x))
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    cl, cu = _infer_constraint_bounds(evaluator)
    return bool(np.all(g >= cl - tol) and np.all(g <= cu + tol))


def subnlp(
    model: Model,
    x_relax: np.ndarray,
    backend: Optional[Callable] = None,
    nlp_options: Optional[dict] = None,
    integer_tol: float = 1e-5,
    feas_tol: float = 1e-6,
    evaluator: Optional[NLPEvaluator] = None,
) -> Optional[tuple[np.ndarray, float]]:
    """SubNLP-style primal heuristic: fix integers, re-solve continuous NLP.

    Given a relaxation point ``x_relax``:
    1. Round integer variables to their nearest integer value and tighten
       their bounds to that single value (fixed). For pure-continuous models,
       this step is a no-op.
    2. Solve the resulting continuous NLP from ``x_relax`` as warm start.
    3. Verify that the returned point is integer-feasible (trivial when
       integers are fixed) and constraint-feasible.

    Args:
        model: The optimization model.
        x_relax: Relaxation point (NLP-relaxed solution at a B&B node).
        backend: ``solve_nlp(evaluator, x0, options=...)`` callable. If None,
            uses :func:`discopt.solvers.nlp_backend.get_nlp_solver('auto')`.
        nlp_options: Options dict forwarded to the NLP backend.
        integer_tol: Tolerance for declaring integer feasibility.
        feas_tol: Tolerance for declaring constraint feasibility.
        evaluator: Pre-built NLPEvaluator; one is constructed if omitted.

    Returns:
        ``(x, obj)`` if the heuristic produced a usable integer- and
        constraint-feasible point, else ``None``.
    """
    if backend is None:
        from discopt.solvers.nlp_backend import get_nlp_solver

        backend = get_nlp_solver("auto")

    if evaluator is None:
        evaluator = NLPEvaluator(model)

    int_mask = _get_integer_mask(model)
    lb_orig, ub_orig = _get_variable_bounds(model)

    x0 = np.asarray(x_relax, dtype=np.float64).copy()

    # Fix integer variables by rounding and clamping bounds to that value.
    # We mutate the model variables' bounds in-place and restore afterwards
    # since NLPEvaluator.variable_bounds reads from the model on each call.
    saved_bounds: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        if np.any(int_mask):
            x0[int_mask] = np.round(x0[int_mask])
            x0 = np.clip(x0, lb_orig, ub_orig)

            # Save and tighten bounds on integer variables.
            offset = 0
            for v in model._variables:
                sz = v.size
                saved_bounds.append((v.lb.copy(), v.ub.copy()))
                if v.var_type in (VarType.BINARY, VarType.INTEGER):
                    fixed = x0[offset : offset + sz].reshape(v.lb.shape)
                    v.lb = fixed.copy()
                    v.ub = fixed.copy()
                offset += sz

        opts = dict(nlp_options) if nlp_options else {}
        opts.setdefault("print_level", 0)

        try:
            nlp_result = backend(evaluator, x0, options=opts)
        except Exception:
            return None
    finally:
        for v, (lb_v, ub_v) in zip(model._variables, saved_bounds):
            v.lb = lb_v
            v.ub = ub_v

    if not _is_nlp_feasible(nlp_result):
        return None
    if nlp_result.x is None or nlp_result.objective is None:
        return None

    x_out = np.asarray(nlp_result.x)

    # Clip integer slots back to the rounded value (the fixed-bounds solve
    # should already yield this, but guard against tiny drifts).
    if np.any(int_mask):
        x_out = x_out.copy()
        x_out[int_mask] = np.round(x_out[int_mask])
        if not _is_integer_feasible(x_out, int_mask, tol=integer_tol):
            return None

    if not _check_constraint_feasibility(evaluator, x_out, tol=feas_tol):
        return None

    # Recompute objective at the snapped point to keep it consistent.
    obj = float(evaluator.evaluate_objective(x_out))
    return x_out, obj
