"""Public bound-tightening utilities (feasibility-based / row-activity).

discopt's solver already runs a sophisticated bound-tightening stack during
presolve and at the root node — feasibility-based bound tightening (FBBT,
iterated to a fixpoint), implied row-activity propagation, duality-based
tightening (DBBT), and optimization-based bound tightening (OBBT) over the
McCormick relaxation. This module exposes the *sound, LP-free* core of that
stack — FBBT — as a small public API for analysis and for building custom
workflows (e.g. conflict detection).

FBBT propagates the **row activity** of each constraint under the current
variable bounds: for a constraint ``g(x) ⋈ rhs`` it computes the interval the
body can take, then isolates each variable to derive an implied bound, iterating
across constraints until a fixpoint. It only ever derives *valid* bounds, so the
tightened box always contains the entire feasible region — it never removes a
feasible point, and in particular never excludes the optimum. If propagation
produces an empty interval, the (sub)problem is proven infeasible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from discopt.modeling.core import Model

__all__ = ["BoundTightening", "fbbt_box"]


@dataclass
class BoundTightening:
    """Result of an FBBT bound-tightening pass.

    Attributes
    ----------
    lb, ub : np.ndarray
        Tightened flat (per-scalar) variable bounds. Always a subset of the
        input box (each bound intersected, never loosened).
    infeasible : bool
        True if propagation derived an empty interval for some variable, which
        proves the problem infeasible.
    n_tightened : int
        Number of scalar variables whose lower or upper bound was strictly
        improved.
    """

    lb: np.ndarray
    ub: np.ndarray
    infeasible: bool
    n_tightened: int


def _block_sizes(shapes: list[list[int]]) -> list[int]:
    sizes = []
    for shp in shapes:
        n = 1
        for d in shp:
            n *= d
        sizes.append(n)
    return sizes


def fbbt_box(model: Model, *, max_iter: int = 20, tol: float = 1e-9) -> BoundTightening:
    """Tighten ``model``'s variable bounds with feasibility-based bound tightening.

    Runs FBBT (iterated to a fixpoint, up to ``max_iter`` sweeps) over the model's
    constraints and returns the tightened flat bounds. The per-block bounds from
    the Rust FBBT engine are expanded to per-scalar bounds and intersected with
    the model's current bounds, so the result is always a sound subset of the
    input box.

    Parameters
    ----------
    model : Model
        The model whose bounds to tighten.
    max_iter : int
        Maximum FBBT sweeps (cross-constraint propagation reaches a fixpoint when
        a sweep changes nothing).
    tol : float
        Numerical tolerance for empty-interval (infeasibility) detection.

    Returns
    -------
    BoundTightening
    """
    from discopt._rust import model_to_repr

    repr_ = model_to_repr(model)
    n_blocks = repr_.n_var_blocks
    shapes = repr_.var_shapes()
    sizes = _block_sizes(shapes)

    # Original per-scalar bounds (block-by-block), and the FBBT per-block result.
    orig_lb: list[float] = []
    orig_ub: list[float] = []
    for i in range(n_blocks):
        orig_lb.extend(float(v) for v in repr_.var_lb(i))
        orig_ub.extend(float(v) for v in repr_.var_ub(i))
    orig_lb_arr = np.asarray(orig_lb, dtype=np.float64)
    orig_ub_arr = np.asarray(orig_ub, dtype=np.float64)

    block_lb, block_ub = repr_.fbbt(max_iter=max_iter, tol=tol)
    block_lb = np.asarray(block_lb, dtype=np.float64)
    block_ub = np.asarray(block_ub, dtype=np.float64)

    # Expand each block's [lo, hi] to its scalar slots; intersect with originals.
    # The Rust FBBT engine seeds each block from the element-wise UNION of its
    # bounds (`seed_block_interval`, C-31), so the returned block interval is a
    # valid *outer* bound for EVERY element of the block — stamping it onto every
    # scalar slot and intersecting with the originals can only tighten, never
    # loosen, and never excludes a feasible point. (Pre-C-31 the engine seeded
    # from element 0 only, so on a heterogeneous array block this stamp cut the
    # feasible region of the other elements and could reach the certified LP
    # dual bound via `_fbbt_argument_box`.)
    new_lb = orig_lb_arr.copy()
    new_ub = orig_ub_arr.copy()
    offset = 0
    for i in range(n_blocks):
        size = sizes[i]
        lo = block_lb[i] if i < len(block_lb) else -math.inf
        hi = block_ub[i] if i < len(block_ub) else math.inf
        sl = slice(offset, offset + size)
        new_lb[sl] = np.maximum(new_lb[sl], lo)
        new_ub[sl] = np.minimum(new_ub[sl], hi)
        offset += size

    infeasible = bool(np.any(new_lb > new_ub + tol))
    tightened = int(np.count_nonzero((new_lb > orig_lb_arr + tol) | (new_ub < orig_ub_arr - tol)))

    return BoundTightening(lb=new_lb, ub=new_ub, infeasible=infeasible, n_tightened=tightened)
