"""Centralised numerical constants for the discopt solver.

All magic numbers, tolerances, and sentinel values used across the solver
pipeline are defined here so that they can be tuned from a single location
and referenced by name in the code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Infeasibility / failure sentinels
# ---------------------------------------------------------------------------
# Value stored as the lower bound for nodes whose NLP relaxation failed or
# was declared infeasible.  Must be larger than any realistic objective.
INFEASIBILITY_SENTINEL: float = 1e30

# Threshold for filtering bogus incumbents.  Any incumbent with objective
# >= this value is treated as invalid.  Set slightly below the sentinel so
# that legitimate large objectives (e.g. 1e25) are never filtered out.
SENTINEL_THRESHOLD: float = 1e29

# ---------------------------------------------------------------------------
# Constraint bound "infinity" — used where solvers need finite bounds in
# place of +/- inf (e.g. Ipopt, HiGHS).  NOT a sentinel — this is a
# legitimate large number representing an inactive bound.
# ---------------------------------------------------------------------------
CONSTRAINT_INF: float = 1e20

# ---------------------------------------------------------------------------
# Default box for a variable declared with no bounds.
#
# Deliberately just BELOW ``CONSTRAINT_INF`` (#850), so the box is a finite
# number every engine can carry. It is *read* as "no bound" (#1678 b): the
# LP/MILP routes hand a side of this magnitude to their engines as the 1e20
# infinity (so ``min -x`` is proved ``unbounded``), and
# ``Model._withhold_default_box_certificate`` refuses any certificate whose
# point sits on it. A declared bound below it, however large, is finite as
# posed. Reports still distinguish it from ``CONSTRAINT_INF`` (#1387): it tells
# the user they declared no bound, not that they wrote 1e20.
# ---------------------------------------------------------------------------
DEFAULT_VARIABLE_BOUND: float = 9.999e19

# ---------------------------------------------------------------------------
# Starting-point generation
# ---------------------------------------------------------------------------
# When variable bounds are infinite, clip to this range for midpoint /
# multi-start starting-point generation.
STARTING_POINT_CLIP: float = 100.0


def clip_start_box(lb, ub):
    """A finite window inside ``[lb, ub]`` for midpoint / multi-start generation.

    ``np.clip(b, -STARTING_POINT_CLIP, STARTING_POINT_CLIP)`` on each bound is the
    window wherever the box meets ``[-STARTING_POINT_CLIP, STARTING_POINT_CLIP]``,
    and this returns exactly that there. A box outside it, or touching it (#1542:
    ``[1.3e6, 1.3e6 + 5]``, a model written in shifted coordinates) collapses under
    the bare clip to the single point ``±STARTING_POINT_CLIP`` -- outside the box,
    with span 0. The NLP-BB node pre-screen read that as "every variable pinned",
    evaluated the rows at a point the box does not contain, and returned a false
    ``infeasible`` certificate on a feasible MINLP. Such a box gets a window of the
    same width ``2 * STARTING_POINT_CLIP`` anchored at its near end instead, so the
    result always satisfies ``lb <= lo <= hi <= ub`` for a non-empty box, and
    ``lo < hi`` whenever ``lb < ub``.
    """
    import numpy as np

    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    c = STARTING_POINT_CLIP
    lo = np.clip(lb, -c, c)
    hi = np.clip(ub, -c, c)
    # ``>=`` / ``<=``, not strict: a box TOUCHING the window (``[-105, -100]``)
    # meets it in one point only, so the bare clip collapses it just the same.
    above = lb >= c
    below = ub <= -c
    if np.any(above) or np.any(below):
        lo = np.where(above, lb, np.where(below, np.maximum(lb, ub - 2.0 * c), lo))
        hi = np.where(above, np.minimum(ub, lb + 2.0 * c), np.where(below, ub, hi))
    return lo, hi


# Fractions along the [lb, ub] interval for multi-start seeds.
MULTISTART_FRACTIONS: tuple[float, ...] = (0.25, 0.75)

# ---------------------------------------------------------------------------
# Solver tolerances
# ---------------------------------------------------------------------------
# Default Ipopt / IPM convergence tolerance.
DEFAULT_OPTIMALITY_TOL: float = 1e-7

# AlphaBB: eigenvalue threshold — alphas below this are treated as zero
# (i.e. the function is already convex in that direction).
ALPHABB_EPS: float = 1e-8

# AlphaBB: safety margin added to alpha to ensure strict underestimation.
ALPHABB_SAFETY: float = 1e-6
