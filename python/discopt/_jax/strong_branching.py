"""
Strong branching data collection for GNN training labels.

Evaluates the dual bound improvement for each candidate branching variable
by solving two child NLP relaxations (left: x_i <= floor(val), right:
x_i >= ceil(val)) and recording the resulting bound changes.

The collected (graph, scores) pairs serve as imitation-learning targets
for the GNN branching policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from discopt._jax.problem_graph import INTEGRALITY_TOL

# Type alias for an NLP solve function
# Signature: solve_fn(lb, ub) -> (objective: float, feasible: bool)
SolveFn = Callable[[np.ndarray, np.ndarray], tuple[float, bool]]


@dataclass
class StrongBranchResult:
    """Result of strong branching evaluation for one variable.

    Attributes:
        var_index: Flat variable index.
        score: Combined branching score (product of dual improvements).
        left_bound: Dual bound from left child (x_i <= floor(val)).
        right_bound: Dual bound from right child (x_i >= ceil(val)).
        left_feasible: Whether left child was feasible.
        right_feasible: Whether right child was feasible.
    """

    var_index: int
    score: float
    left_bound: float
    right_bound: float
    left_feasible: bool
    right_feasible: bool


@dataclass
class StrongBranchData:
    """Collection of strong branching evaluations at a single B&B node.

    Used as training data for the GNN policy (imitation learning).

    Attributes:
        parent_bound: Objective value from the parent relaxation.
        results: List of StrongBranchResult for each candidate variable.
        best_var_index: Variable index with the highest strong branching score.
        scores_array: Normalized scores array (n_vars,), with 0 for
            non-candidate variables.
    """

    parent_bound: float
    results: list[StrongBranchResult]
    best_var_index: Optional[int]
    scores_array: np.ndarray


def _sb_score(
    left_bound: float,
    right_bound: float,
    parent_bound: float,
    left_feasible: bool,
    right_feasible: bool,
) -> float:
    """Compute strong branching score from child dual bounds.

    Uses the product scoring rule from Achterberg et al. (2005):
        score = max(dl, eps) * max(dr, eps)
    where dl, dr are dual bound improvements in each child.

    If a child is infeasible, its improvement is treated as +inf
    (large value), since that subtree is pruned.
    """
    eps = 1e-6
    if not left_feasible:
        dl = 1e6
    else:
        dl = max(left_bound - parent_bound, eps)

    if not right_feasible:
        dr = 1e6
    else:
        dr = max(right_bound - parent_bound, eps)

    return max(dl, eps) * max(dr, eps)


def evaluate_strong_branching(
    solution: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    parent_bound: float,
    is_integer: np.ndarray,
    solve_fn: SolveFn,
    n_vars: int,
    max_candidates: int = 20,
) -> StrongBranchData:
    """Evaluate strong branching scores for all fractional integer variables.

    For each candidate variable, solves two child NLP relaxations and
    computes the dual bound improvement.

    Args:
        solution: Current relaxation solution (n_vars,).
        node_lb: Current node lower bounds (n_vars,).
        node_ub: Current node upper bounds (n_vars,).
        parent_bound: Objective value of the parent relaxation.
        is_integer: Boolean array (n_vars,) marking integer variables.
        solve_fn: Function(lb, ub) -> (objective, feasible) to solve
            an NLP relaxation with given bounds.
        n_vars: Total number of variables.
        max_candidates: Maximum number of candidates to evaluate
            (pre-select by fractionality for efficiency).

    Returns:
        StrongBranchData with per-variable scores.
    """
    # Identify candidate variables (fractional integer)
    candidates = []
    for i in range(n_vars):
        if not is_integer[i]:
            continue
        val = float(solution[i])
        frac = val - np.floor(val)
        if frac < INTEGRALITY_TOL or frac > 1.0 - INTEGRALITY_TOL:
            continue
        fractionality = 0.5 - abs(frac - 0.5)
        candidates.append((i, val, fractionality))

    # Sort by fractionality (most fractional first) and limit
    candidates.sort(key=lambda t: -t[2])
    candidates = candidates[:max_candidates]

    results = []
    for var_idx, val, _ in candidates:
        floor_val = np.floor(val)
        ceil_val = floor_val + 1.0

        # Left child: x_i <= floor(val)
        left_ub = node_ub.copy()
        left_ub[var_idx] = floor_val
        left_obj, left_feas = solve_fn(node_lb, left_ub)

        # Right child: x_i >= ceil(val)
        right_lb = node_lb.copy()
        right_lb[var_idx] = ceil_val
        right_obj, right_feas = solve_fn(right_lb, node_ub)

        score = _sb_score(left_obj, right_obj, parent_bound, left_feas, right_feas)

        results.append(
            StrongBranchResult(
                var_index=var_idx,
                score=score,
                left_bound=left_obj,
                right_bound=right_obj,
                left_feasible=left_feas,
                right_feasible=right_feas,
            )
        )

    # Build scores array
    scores_array = np.zeros(n_vars, dtype=np.float64)
    best_var = None
    best_score = -1.0

    for r in results:
        scores_array[r.var_index] = r.score
        if r.score > best_score:
            best_score = r.score
            best_var = r.var_index

    # Normalize scores to [0, 1]
    max_score = scores_array.max()
    if max_score > 0:
        scores_array = scores_array / max_score

    return StrongBranchData(
        parent_bound=parent_bound,
        results=results,
        best_var_index=best_var,
        scores_array=scores_array,
    )
