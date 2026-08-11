"""
Discretization State Management for AMP (Adaptive Multivariate Partitioning).

Manages the partition breakpoints for each selected variable.  The core insight
from CP 2016 is that adding breakpoints near the current MILP solution concentrates
refinement where the relaxation gap is largest, converging faster than uniform
refinement.

Key properties guaranteed by this module:
- Partitions are monotonically finer: once a breakpoint is added, it is never removed.
- Bounds are always preserved: partitions[v][0] == lb[v], partitions[v][-1] == ub[v].
- Convergence check: all partition widths < abs_width_tol.

Theory references:
  - Nagarajan et al., CP 2016: http://harshangrjn.github.io/pdf/CP_2016.pdf
  - Alpine.jl add_adaptive_partition() in bounding_model.jl
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DiscretizationState:
    """Breakpoint dictionaries for AMP domain partitioning.

    Attributes
    ----------
    partitions : dict[int, np.ndarray]
        Maps flat variable index → sorted 1-D array of breakpoints.
        For a variable with n_init=2 initial intervals:
            partitions[v] = [lb, midpoint, ub]  (3 elements)
        Finer partitions have more elements.
    scaling_factor : float
        Controls the width of new partitions added during adaptive refinement.
        New breakpoints are placed at x* ± (partition_width / scaling_factor).
        Default 10.0 (Alpine's default).
    abs_width_tol : float
        Convergence threshold.  When all partition widths are below this value,
        check_partition_convergence() returns True.
        Default 1e-3.
    """

    partitions: dict[int, np.ndarray] = field(default_factory=dict)
    scaling_factor: float = 10.0
    abs_width_tol: float = 1e-3


def initialize_partitions(
    var_indices: list[int],
    lb: list[float],
    ub: list[float],
    n_init: int = 2,
    scaling_factor: float = 10.0,
    abs_width_tol: float = 1e-3,
) -> DiscretizationState:
    """Create n_init uniform intervals for each partition variable.

    Parameters
    ----------
    var_indices : list[int]
        Flat variable indices to partition.
    lb : list[float]
        Lower bound for each variable (same order as var_indices).
    ub : list[float]
        Upper bound for each variable (same order as var_indices).
    n_init : int, default 2
        Number of initial intervals per variable.  n_init=2 gives 3 breakpoints:
        [lb, midpoint, ub].
    scaling_factor : float, default 10.0
        Refinement ratio stored on the returned state.
    abs_width_tol : float, default 1e-3
        Convergence tolerance stored on the returned state.

    Returns
    -------
    DiscretizationState
        Initial discretization with uniform breakpoints.
    """
    partitions: dict[int, np.ndarray] = {}
    for k, v_idx in enumerate(var_indices):
        partitions[v_idx] = np.linspace(float(lb[k]), float(ub[k]), n_init + 1)
    return DiscretizationState(
        partitions=partitions,
        scaling_factor=float(scaling_factor),
        abs_width_tol=float(abs_width_tol),
    )


def add_adaptive_partition(
    state: DiscretizationState,
    solution: dict[int, float],
    var_indices: list[int],
    lb: list[float],
    ub: list[float],
) -> DiscretizationState:
    """Add breakpoints near solution[v] for each partition variable v.

    For each variable v in var_indices:
    1. Identify the active partition [p_lo, p_hi] containing solution[v].
    2. Compute width = p_hi - p_lo.
    3. Add new breakpoints at:
         p_lo + width / scaling_factor
         p_hi - width / scaling_factor
    4. Merge with existing breakpoints, sort, deduplicate.

    This mirrors Alpine.jl's add_adaptive_partition() in bounding_model.jl.

    Parameters
    ----------
    state : DiscretizationState
        Current partition state.
    solution : dict[int, float]
        Current MILP solution values, keyed by flat variable index.
    var_indices : list[int]
        Variables to refine.
    lb : list[float]
        Lower bounds (same order as var_indices).
    ub : list[float]
        Upper bounds (same order as var_indices).

    Returns
    -------
    DiscretizationState
        New state with refined partitions (old breakpoints preserved).
    """
    new_partitions = {k: v.copy() for k, v in state.partitions.items()}

    for k, v_idx in enumerate(var_indices):
        if v_idx not in new_partitions:
            # Variable not yet partitioned — initialize with 2 intervals
            new_partitions[v_idx] = np.linspace(float(lb[k]), float(ub[k]), 3)

        pts = new_partitions[v_idx]
        x_star = float(solution.get(v_idx, 0.5 * (pts[0] + pts[-1])))

        # Clip x_star to [lb, ub]
        x_star = float(np.clip(x_star, pts[0], pts[-1]))

        # Find active partition interval
        p_lo_raw = np.searchsorted(pts, x_star, side="right") - 1
        p_lo_idx = int(np.clip(p_lo_raw, 0, len(pts) - 2))
        p_lo = float(pts[p_lo_idx])
        p_hi = float(pts[p_lo_idx + 1])
        width = p_hi - p_lo

        if width <= 0:
            continue  # degenerate partition — skip

        delta = width / state.scaling_factor

        # New breakpoints: two new points around the solution value
        # (mirrors Alpine.jl add_adaptive_partition)
        new1 = x_star - delta
        new2 = x_star + delta

        # Only add if they create a meaningful improvement and lie strictly inside
        eps = 1e-12
        candidates = []
        if new1 > p_lo + eps and new1 < p_hi - eps:
            candidates.append(new1)
        if new2 > p_lo + eps and new2 < p_hi - eps and abs(new2 - new1) > eps:
            candidates.append(new2)
        if not candidates:
            midpoint = 0.5 * (p_lo + p_hi)
            if midpoint > p_lo + eps and midpoint < p_hi - eps:
                candidates.append(midpoint)

        if candidates:
            merged = np.sort(np.unique(np.concatenate([pts, candidates])))
            new_partitions[v_idx] = merged

    return DiscretizationState(
        partitions=new_partitions,
        scaling_factor=state.scaling_factor,
        abs_width_tol=state.abs_width_tol,
    )


def add_uniform_partition(
    state: DiscretizationState,
    _solution: dict[int, float],
    var_indices: list[int],
    lb: list[float],
    ub: list[float],
) -> DiscretizationState:
    """Uniformly refine every current interval for the selected variables.

    Unlike adaptive refinement, this does not use the current solution value.
    Each interval is split at its midpoint, preserving all existing breakpoints.
    """
    new_partitions = {k: v.copy() for k, v in state.partitions.items()}

    for k, v_idx in enumerate(var_indices):
        if v_idx not in new_partitions:
            new_partitions[v_idx] = np.linspace(float(lb[k]), float(ub[k]), 3)

        pts = new_partitions[v_idx]
        if len(pts) < 2:
            continue

        mids = 0.5 * (pts[:-1] + pts[1:])
        merged = np.sort(np.unique(np.concatenate([pts, mids])))
        new_partitions[v_idx] = merged

    return DiscretizationState(
        partitions=new_partitions,
        scaling_factor=state.scaling_factor,
        abs_width_tol=state.abs_width_tol,
    )


def check_partition_convergence(state: DiscretizationState) -> bool:
    """Return True if every partition width is below state.abs_width_tol.

    When this returns True, further partitioning will not significantly improve
    the relaxation bound.

    Parameters
    ----------
    state : DiscretizationState
        Current partition state.

    Returns
    -------
    bool
        True if all partition widths < abs_width_tol.
    """
    for pts in state.partitions.values():
        if len(pts) < 2:
            continue
        widths = np.diff(pts)
        if np.any(widths >= state.abs_width_tol):
            return False
    return True
