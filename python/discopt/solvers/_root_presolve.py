"""Root-node presolve helpers shared by global solvers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from discopt.modeling.core import Model

logger = logging.getLogger(__name__)


def _round_integral_bounds(
    lb: np.ndarray,
    ub: np.ndarray,
    int_offsets: list[int],
    int_sizes: list[int],
) -> None:
    """Round integer/binary flat bounds in-place."""
    for offset, size in zip(int_offsets, int_sizes):
        sl = slice(offset, offset + size)
        finite_lb = np.isfinite(lb[sl])
        finite_ub = np.isfinite(ub[sl])
        lb_view = lb[sl]
        ub_view = ub[sl]
        lb_view[finite_lb] = np.ceil(lb_view[finite_lb] - 1e-9)
        ub_view[finite_ub] = np.floor(ub_view[finite_ub] + 1e-9)


def _flat_fbbt_bounds(
    model: Model,
    fbbt_lbs: np.ndarray,
    fbbt_ubs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map block-level FBBT bounds to flat solver bounds."""
    tightened_lb = lb.copy()
    tightened_ub = ub.copy()

    if len(fbbt_lbs) != len(model._variables) or len(fbbt_ubs) != len(model._variables):
        return tightened_lb, tightened_ub

    offset = 0
    for block_idx, var in enumerate(model._variables):
        size = var.size
        if size != 1:
            offset += size
            continue
        block_lb = float(fbbt_lbs[block_idx])
        block_ub = float(fbbt_ubs[block_idx])
        sl = slice(offset, offset + size)
        if np.isfinite(block_lb):
            tightened_lb[sl] = np.maximum(tightened_lb[sl], block_lb)
        if np.isfinite(block_ub):
            tightened_ub[sl] = np.minimum(tightened_ub[sl], block_ub)
        offset += size

    return tightened_lb, tightened_ub


def tighten_root_bounds_with_fbbt(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    int_offsets: list[int],
    int_sizes: list[int],
    *,
    model_repr: Any | None = None,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Run root FBBT and integer-bound rounding for solver tree bounds.

    Returns ``(tightened_lb, tightened_ub, infeasible, changed)``. If the Rust
    FBBT binding is unavailable, this still performs sound integer rounding.
    """
    orig_lb = np.asarray(lb, dtype=np.float64)
    orig_ub = np.asarray(ub, dtype=np.float64)
    tightened_lb = orig_lb.copy()
    tightened_ub = orig_ub.copy()

    if model_repr is None:
        try:
            from discopt._rust import model_to_repr

            model_repr = model_to_repr(model, getattr(model, "_builder", None))
        except Exception as exc:
            logger.debug("Root FBBT model conversion skipped: %s", exc)
            model_repr = None

    if model_repr is not None:
        try:
            fbbt_lbs, fbbt_ubs = model_repr.fbbt(max_iter=max_iter, tol=tol)
            tightened_lb, tightened_ub = _flat_fbbt_bounds(
                model,
                np.asarray(fbbt_lbs, dtype=np.float64),
                np.asarray(fbbt_ubs, dtype=np.float64),
                tightened_lb,
                tightened_ub,
            )
        except Exception as exc:
            logger.debug("Root FBBT bound tightening skipped: %s", exc)

    _round_integral_bounds(tightened_lb, tightened_ub, int_offsets, int_sizes)

    infeasible = bool(np.any(tightened_lb > tightened_ub + tol))
    close = (tightened_lb > tightened_ub) & (tightened_lb <= tightened_ub + tol)
    if np.any(close):
        midpoint = 0.5 * (tightened_lb[close] + tightened_ub[close])
        tightened_lb[close] = midpoint
        tightened_ub[close] = midpoint

    changed = bool(np.any(tightened_lb > orig_lb + tol) or np.any(tightened_ub < orig_ub - tol))
    return tightened_lb, tightened_ub, infeasible, changed
