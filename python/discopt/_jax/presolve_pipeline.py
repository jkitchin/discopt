"""Root-stage presolve orchestrator (wires M4+M5 and M10 of issue #51).

The Rust-side passes ``eliminate_variables`` and ``reformulate_polynomial``
are pure ``ModelRepr -> ModelRepr`` transforms and are exposed individually
through PyO3. This module sequences them at root, together with the
existing FBBT pass, and returns a tightened ``PyModelRepr`` plus an
aggregate stats dict suitable for logging / regression assertions.

Sequencing rationale (cheapest-and-most-shrinking first):

1. ``eliminate_variables`` (M10) — pins continuous scalar variables that
   are uniquely determined by a singleton equality and drops the
   determining equality. Pure tightening, monotone, idempotent. Cheap.
2. ``reformulate_polynomial`` (M4 + M5) — rewrites high-degree monomials
   to nested bilinear products with auxiliary variables; derives
   McCormick-style bounds for the new aux vars from interval AD. This
   *changes the model topology* (adds aux vars and constraints), so it
   is opt-in by default.
3. ``fbbt`` — forward/backward bound propagation. Runs last so it sees
   the tightened bounds and (when enabled) the new aux variables.

All passes produce a *new* ``ModelRepr``; nothing in-place. The caller is
responsible for swapping in the returned object as the active repr.
"""

from __future__ import annotations

from typing import Any


def run_root_presolve(
    model_repr,
    *,
    eliminate: bool = True,
    polynomial: bool = False,
    fbbt: bool = True,
    fbbt_max_iter: int = 20,
    fbbt_tol: float = 1e-8,
) -> tuple[Any, dict]:
    """Run the root-stage presolve sequence on ``model_repr``.

    Args:
        model_repr: a ``PyModelRepr`` produced by ``model_to_repr``.
        eliminate: run ``eliminate_variables`` (M10). Safe by default.
        polynomial: run ``reformulate_polynomial`` (M4 + M5). Off by
            default because it adds auxiliary variables and changes the
            shape of the variable index space; downstream code that
            assumes a fixed n_vars must opt in.
        fbbt: run FBBT after the structural rewrites.
        fbbt_max_iter, fbbt_tol: forwarded to ``fbbt`` / ``fbbt_with_cutoff``.

    Returns:
        ``(new_model_repr, stats)`` where ``stats`` is a dict with keys
        ``elimination`` (or absent when skipped), ``polynomial`` (ditto),
        and ``fbbt`` containing per-block ``lb``/``ub`` arrays. Empty
        dict means no pass actually ran.
    """
    stats: dict = {}
    repr_ = model_repr

    if eliminate:
        repr_, elim_stats = repr_.eliminate_variables()
        stats["elimination"] = dict(elim_stats)

    if polynomial:
        repr_, poly_stats = repr_.reformulate_polynomial()
        stats["polynomial"] = dict(poly_stats)

    if fbbt:
        lbs, ubs = repr_.fbbt_with_cutoff(
            max_iter=fbbt_max_iter, tol=fbbt_tol, incumbent_bound=None
        )
        stats["fbbt"] = {"lb": lbs, "ub": ubs}

    return repr_, stats


def propagate_bounds_to_model(model, model_repr) -> int:
    """Push tightened per-element bounds from ``model_repr`` back into ``model``.

    This is the bridge that lets the Rust-side presolve outcome influence
    the Python-side relaxation compiler / branch-and-bound initialisation.
    Only blocks whose count and flat element count are unchanged from the
    original model are touched. Aux variables introduced by
    ``reformulate_polynomial`` have no Python-side counterpart and are
    silently skipped.

    Returns the number of *scalar elements* whose ``lb`` or ``ub`` was
    strictly tightened. Equal-or-looser updates are ignored.
    """
    import numpy as np

    n_tightened = 0
    py_blocks = list(model._variables)
    n_blocks = min(len(py_blocks), model_repr.n_var_blocks)
    for bi in range(n_blocks):
        block = py_blocks[bi]
        py_lb = np.asarray(block.lb, dtype=np.float64)
        py_ub = np.asarray(block.ub, dtype=np.float64)
        rust_lb = np.asarray(model_repr.var_lb(bi), dtype=np.float64)
        rust_ub = np.asarray(model_repr.var_ub(bi), dtype=np.float64)
        if rust_lb.size != py_lb.size or rust_ub.size != py_ub.size:
            continue
        flat_py_lb = py_lb.reshape(-1).copy()
        flat_py_ub = py_ub.reshape(-1).copy()
        changed = False
        for k in range(flat_py_lb.size):
            new_lo = max(flat_py_lb[k], rust_lb[k])
            new_hi = min(flat_py_ub[k], rust_ub[k])
            if new_lo > flat_py_lb[k] + 1e-12:
                flat_py_lb[k] = new_lo
                n_tightened += 1
                changed = True
            if new_hi < flat_py_ub[k] - 1e-12:
                flat_py_ub[k] = new_hi
                n_tightened += 1
                changed = True
        if changed:
            block.lb = flat_py_lb.reshape(py_lb.shape)
            block.ub = flat_py_ub.reshape(py_ub.shape)
    return n_tightened


def run_reverse_ad_tightening(
    model,
    *,
    max_iter: int = 25,
    tol: float = 1e-9,
) -> int:
    """Tighten ``model`` variable bounds via reverse-mode interval AD (M9).

    Walks every constraint as ``(body, target_interval)`` where ``target``
    is derived from the constraint sense and rhs, then iterates the
    Gauss-Seidel reverse-AD propagation in ``interval_ad_reverse.tighten_box``
    to a fixed point. The returned tightened intervals are intersected
    with the current Python-side ``lb``/``ub`` and written back, so that
    the relaxation compiler and B&B initialisation see the tighter box.

    Returns the number of variable blocks whose bounds were strictly
    tightened. Skips quietly when no constraints have polynomial /
    differentiable bodies.
    """
    import numpy as np

    from discopt._jax.convexity.interval import Interval
    from discopt._jax.convexity.interval_ad_reverse import tighten_box

    constraints: list = []
    for c in model._constraints:
        sense = getattr(c, "sense", None)
        rhs = float(getattr(c, "rhs", 0.0))
        body = getattr(c, "body", None)
        if body is None or sense is None:
            continue
        if sense == "<=":
            target = Interval(
                np.asarray(-np.inf, dtype=np.float64),
                np.asarray(rhs, dtype=np.float64),
            )
        elif sense == ">=":
            target = Interval(
                np.asarray(rhs, dtype=np.float64),
                np.asarray(np.inf, dtype=np.float64),
            )
        elif sense == "==":
            target = Interval(
                np.asarray(rhs, dtype=np.float64),
                np.asarray(rhs, dtype=np.float64),
            )
        else:
            continue
        constraints.append((body, target))
    if not constraints:
        return 0

    try:
        new_box = tighten_box(constraints, model, max_iter=max_iter, tol=tol)
    except Exception:
        return 0

    n_tightened = 0
    for v, iv in new_box.items():
        lo_arr = np.asarray(iv.lo, dtype=np.float64).ravel()
        hi_arr = np.asarray(iv.hi, dtype=np.float64).ravel()
        if lo_arr.size != 1 or hi_arr.size != 1:
            # Vector / matrix interval bounds aren't supported by this
            # propagation path yet.
            continue
        new_lo = float(lo_arr[0])
        new_hi = float(hi_arr[0])
        if not (np.isfinite(new_lo) and np.isfinite(new_hi)):
            continue
        py_lb = np.asarray(v.lb, dtype=np.float64).ravel()
        py_ub = np.asarray(v.ub, dtype=np.float64).ravel()
        if py_lb.size != 1 or py_ub.size != 1:
            continue
        cur_lo = float(py_lb[0])
        cur_hi = float(py_ub[0])
        if new_lo > cur_lo + 1e-12 or new_hi < cur_hi - 1e-12:
            v.lb = np.broadcast_to(np.asarray(max(cur_lo, new_lo)), v.shape)
            v.ub = np.broadcast_to(np.asarray(min(cur_hi, new_hi)), v.shape)
            n_tightened += 1
    return n_tightened


__all__ = [
    "run_root_presolve",
    "propagate_bounds_to_model",
    "run_reverse_ad_tightening",
]
