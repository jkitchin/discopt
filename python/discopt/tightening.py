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

__all__ = ["BoundTightening", "fbbt_box", "probe_box"]


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


def probe_box(
    model: Model,
    *,
    max_iter: int = 8,
    tol: float = 1e-9,
    max_vars: int = 64,
    incumbent: float | None = None,
) -> BoundTightening:
    """Tighten ``model``'s bounds with FBBT **plus per-node probing** (P3).

    This is the sound, LP-free branch-and-reduce reduction: after an FBBT sweep,
    each discrete variable is tentatively fixed at a bound and FBBT is re-run; a
    *proven-infeasible* fixing contracts the domain (binaries are forced to the
    surviving value, integer endpoints are peeled). It is strictly at least as
    tight as :func:`fbbt_box` and, like it, only ever derives valid bounds — the
    returned box always contains the entire feasible region, so it never removes
    a feasible point and in particular never excludes the optimum. An empty
    interval proves the (sub)problem infeasible.

    Parameters
    ----------
    model : Model
        The model whose bounds to tighten.
    max_iter : int
        FBBT inner-loop iteration cap (used both for the outer sweep and for the
        per-fixing FBBT solves inside probing).
    tol : float
        Numerical tolerance for tightening / empty-interval detection.
    max_vars : int
        Budget: at most this many discrete variables are probed.
    incumbent : float, optional
        A valid incumbent objective (upper bound on the optimum). When given,
        probing becomes optimality-aware via a cutoff constraint
        (``objective ⋈ incumbent``); a fixing whose sub-box cannot reach the
        incumbent is discarded. Sound because the incumbent bounds the optimum.

    Returns
    -------
    BoundTightening

    Notes
    -----
    Reductions are applied at *block* granularity through the Rust in-tree
    kernel (exact for scalar variables); the returned per-scalar bounds are
    always intersected with the model's current bounds, so the result is a sound
    subset of the input box regardless of block structure.
    """
    from discopt._rust import model_to_repr

    repr_ = model_to_repr(model, getattr(model, "_builder", None))
    n_blocks = repr_.n_var_blocks
    shapes = repr_.var_shapes()
    sizes = _block_sizes(shapes)

    # Original per-scalar bounds, and a valid per-block outer seed (union of the
    # block's scalar bounds — a sound outer interval for every element; the final
    # intersect with the scalar originals restores per-element tightness).
    orig_lb: list[float] = []
    orig_ub: list[float] = []
    block_seed_lb: list[float] = []
    block_seed_ub: list[float] = []
    for i in range(n_blocks):
        lbs = [float(v) for v in repr_.var_lb(i)]
        ubs = [float(v) for v in repr_.var_ub(i)]
        orig_lb.extend(lbs)
        orig_ub.extend(ubs)
        block_seed_lb.append(min(lbs) if lbs else -math.inf)
        block_seed_ub.append(max(ubs) if ubs else math.inf)
    orig_lb_arr = np.asarray(orig_lb, dtype=np.float64)
    orig_ub_arr = np.asarray(orig_ub, dtype=np.float64)

    delta = repr_.in_tree_presolve(
        np.asarray(block_seed_lb, dtype=np.float64),
        np.asarray(block_seed_ub, dtype=np.float64),
        node_depth=0,
        depth_stride=1,
        max_iter=max_iter,
        tol=tol,
        incumbent=(float(incumbent) if incumbent is not None and np.isfinite(incumbent) else None),
        probing=True,
        probe_max_vars=max_vars,
    )

    if delta["infeasible"]:
        return BoundTightening(lb=orig_lb_arr, ub=orig_ub_arr, infeasible=True, n_tightened=0)

    block_lb = np.asarray(delta["lb"], dtype=np.float64)
    block_ub = np.asarray(delta["ub"], dtype=np.float64)

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
