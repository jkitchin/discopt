"""Reverse-mode interval AD presolve pass (B2 of issue #53).

Wraps ``discopt._jax.convexity.interval_ad_reverse.tighten_box`` as a
:class:`PresolvePass` so reverse-AD bound tightening runs inside the
orchestrator's fixed-point loop alongside Rust kernels (FBBT, OBBT,
implied bounds, ...) instead of as a one-shot step.

The pass walks every constraint of the Python ``Model`` to build the
``(body, target_interval)`` list, then iterates reverse-AD propagation
to quiescence on the *current* variable box (read from
``model_repr.var_lb``/``var_ub``, which reflects whatever upstream
sweeps already produced). Tightened endpoints are written back into the
Rust IR via ``tighten_var_bounds`` so the next Rust sweep sees them.

This is the canonical Schichl & Neumaier (2005) reverse-AD construction
on the JAX DAG: forward-evaluate intervals once, then propagate every
parent's target back through the DAG to children using interval-arith
inverse rules.

Dependencies: A3 (Python passes participate in the orchestrator).

References
----------
- Schichl, Neumaier (2005), *Interval analysis on directed acyclic
  graphs for global optimization*, J. Global Optim. 33.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import make_python_delta


class ReverseADPass:
    """Reverse-mode interval AD bound tightening as a presolve pass.

    Args:
        model: the Python ``Model`` whose constraints define the
            ``(body, target)`` propagation list.
        max_iter: cap on inner Gauss-Seidel sweeps inside ``tighten_box``.
        tol: convergence threshold for the inner loop.
    """

    name = "reverse_ad"

    def __init__(self, model: Any, *, max_iter: int = 25, tol: float = 1e-9):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol

    def run(self, model_repr: Any) -> dict:
        from discopt._jax.convexity.interval import Interval
        from discopt._jax.convexity.interval_ad_reverse import tighten_box

        delta = make_python_delta(self.name)

        # Build the constraint list. Constraints whose body or sense
        # are missing are silently skipped — those don't have a
        # well-defined polynomial body for reverse propagation.
        constraints: list = []
        for c in self.model._constraints:
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
            return delta

        # Build initial box from the *current* Rust IR bounds (which may
        # already have been tightened by upstream passes). Falls back to
        # the Python-side declared bounds for blocks whose flat element
        # count differs (aux vars from polynomial reformulation).
        box = _box_from_model_repr(self.model, model_repr)

        try:
            new_box = tighten_box(
                constraints,
                self.model,
                box=box,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        except Exception:
            # Reverse AD does not support every operator combination
            # (multidimensional indexing, certain transcendentals). On
            # an unsupported atom abstain — the orchestrator will catch
            # this as a no-op delta.
            return delta

        n_tightened = 0
        for v, iv in new_box.items():
            lo_arr = np.asarray(iv.lo, dtype=np.float64).ravel()
            hi_arr = np.asarray(iv.hi, dtype=np.float64).ravel()
            if lo_arr.size != 1 or hi_arr.size != 1:
                # Vector / matrix interval bounds are out of scope for
                # the per-block ``tighten_var_bounds`` writeback.
                continue
            new_lo = float(lo_arr[0])
            new_hi = float(hi_arr[0])
            if not (np.isfinite(new_lo) and np.isfinite(new_hi)):
                continue
            block_idx = _model_block_index(self.model, v)
            if block_idx is None:
                continue
            cur_lb = list(model_repr.var_lb(block_idx))
            cur_ub = list(model_repr.var_ub(block_idx))
            if len(cur_lb) != 1 or len(cur_ub) != 1:
                continue
            tighten_lb = max(cur_lb[0], new_lo)
            tighten_ub = min(cur_ub[0], new_hi)
            if tighten_lb > tighten_ub + 1e-12:
                # Empty interval — write a clearly-empty signal so
                # the orchestrator's empty-interval check fires.
                model_repr.tighten_var_bounds(block_idx, [tighten_lb], [tighten_lb - 1.0])
                n_tightened += 2
                continue
            n = model_repr.tighten_var_bounds(block_idx, [tighten_lb], [tighten_ub])
            n_tightened += int(n)

        delta["bounds_tightened"] = n_tightened
        delta["work_units"] = len(constraints)
        return delta


def _box_from_model_repr(model: Any, model_repr: Any) -> dict:
    """Read the current Rust IR bounds back into a ``{Variable: Interval}``.

    Used to seed reverse-AD tightening with the orchestrator's running
    box rather than the Python ``Model``'s static declared bounds.
    """
    from discopt._jax.convexity.interval import Interval

    box: dict = {}
    py_blocks = list(model._variables)
    n = min(len(py_blocks), model_repr.n_var_blocks)
    for bi in range(n):
        v = py_blocks[bi]
        rust_lb = np.asarray(model_repr.var_lb(bi), dtype=np.float64)
        rust_ub = np.asarray(model_repr.var_ub(bi), dtype=np.float64)
        py_lb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        py_ub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        if rust_lb.size != py_lb.size or rust_ub.size != py_ub.size:
            continue
        eff_lb = np.maximum(py_lb, rust_lb)
        eff_ub = np.minimum(py_ub, rust_ub)
        if eff_lb.size == 1:
            box[v] = Interval(
                np.asarray(eff_lb[0], dtype=np.float64),
                np.asarray(eff_ub[0], dtype=np.float64),
            )
        else:
            box[v] = Interval(
                eff_lb.reshape(np.asarray(v.lb).shape),
                eff_ub.reshape(np.asarray(v.ub).shape),
            )
    return box


def _model_block_index(model: Any, var: Any) -> int | None:
    """Return the variable block index of ``var`` in ``model``, or ``None``."""
    for i, v in enumerate(model._variables):
        if v is var:
            return i
    return None


__all__ = ["ReverseADPass"]
