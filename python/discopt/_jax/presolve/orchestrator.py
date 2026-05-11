"""Python wrapper that interleaves Python passes with Rust orchestrator
sweeps to a fixed point (A3 of the presolve roadmap).

The Rust orchestrator (``discopt._rust.PyModelRepr.presolve``) runs a
fixed-point loop over Rust kernels. To allow Python passes (JAX-driven
convexity, NN-embedded MINLP presolve, reverse-mode interval AD, GDP
reformulation choice) to participate, we wrap the Rust loop:

1. Run the Rust orchestrator one full sweep at a time
   (``max_iterations=1``).
2. After each sweep, run every registered Python pass in order.
   Each pass mutates the ``PyModelRepr`` (typically by tightening
   variable bounds via ``tighten_var_bounds``) and returns a delta
   dict.
3. The combined progress flag (any Rust delta or any Python delta with
   ``delta_made_progress`` true) decides whether to iterate again.
4. Loop until no progress, ``max_iterations`` exceeded, or a wall-time
   budget is hit.

This is the canonical entry point for any setup that needs Python passes
to interact with Rust-side bound tightening. ``run_root_presolve`` in
``presolve_pipeline`` remains the single-call legacy entry point and
delegates here when ``python_passes`` is non-empty.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from .protocol import PresolvePass, delta_made_progress, make_python_delta


def run_orchestrated_presolve(
    model_repr: Any,
    *,
    rust_passes: list[str] | None = None,
    python_passes: Sequence[PresolvePass] = (),
    max_iterations: int = 16,
    rust_kwargs: dict | None = None,
    time_limit_ms: int = 0,
) -> tuple[Any, dict]:
    """Drive a fixed-point presolve loop interleaving Rust and Python passes.

    Args:
        model_repr: ``PyModelRepr`` to presolve. Mutated in place via
            ``tighten_var_bounds`` when Python passes write back.
        rust_passes: List of Rust pass names to run each sweep, or
            ``None`` for the default order. Empty list ⇒ Python-only.
        python_passes: Sequence of objects implementing
            :class:`PresolvePass`. Run after each Rust sweep, in order.
        max_iterations: Cap on the number of Rust+Python sweeps.
        rust_kwargs: Extra kwargs forwarded to
            ``PyModelRepr.presolve`` (e.g. ``fbbt_tol``,
            ``reduced_cost_info``).
        time_limit_ms: Wall-clock cap (0 disables).

    Returns:
        ``(model_repr, stats)`` matching the shape returned by
        ``run_root_presolve``: ``stats`` carries ``iterations``,
        ``terminated_by``, ``deltas``, ``bounds_lo``, ``bounds_hi``.
    """
    if rust_kwargs is None:
        rust_kwargs = {}
    if rust_passes is None:
        rust_passes = [
            "eliminate",
            "aggregate",
            "redundancy",
            "simplify",
            "implied_bounds",
            "fbbt",
            "probing",
        ]

    started = time.monotonic()
    deltas: list[dict] = []
    terminated_by = "IterationCap"
    last_iter = 0
    last_bounds_lo = None
    last_bounds_hi = None

    for sweep in range(max_iterations):
        last_iter = sweep + 1
        sweep_progress = False

        # ── 1. One Rust sweep ────────────────────────────────────────
        if rust_passes:
            new_repr, raw = model_repr.presolve(
                passes=rust_passes,
                max_iterations=1,
                time_limit_ms=time_limit_ms,
                work_unit_budget=0,
                **rust_kwargs,
            )
            model_repr = new_repr
            last_bounds_lo = raw["bounds_lo"]
            last_bounds_hi = raw["bounds_hi"]
            for d in raw["deltas"]:
                deltas.append(d)
                if delta_made_progress(d):
                    sweep_progress = True
            # Rust orchestrator may itself terminate early (Infeasible,
            # TimeBudget). Honour that by exiting outer loop.
            if raw["terminated_by"] == "Infeasible":
                terminated_by = "Infeasible"
                break
            if raw["terminated_by"] == "TimeBudget":
                terminated_by = "TimeBudget"
                break

        # ── 2. Python passes ────────────────────────────────────────
        for p in python_passes:
            pass_started = time.monotonic()
            try:
                d = p.run(model_repr)
            except Exception as exc:  # pragma: no cover - diagnostic
                # Python-pass failures should not silently break the
                # solve. Record an empty delta tagged with the error
                # and continue; the caller can inspect deltas to see
                # which pass failed.
                d = make_python_delta(getattr(p, "name", "python_pass"), pass_iter=sweep)
                d["error"] = repr(exc)
            if d is None:
                continue
            # Stamp wall-time + iter if the pass didn't.
            elapsed_ms = (time.monotonic() - pass_started) * 1000.0
            if d.get("wall_time_ms", 0.0) == 0.0:
                d["wall_time_ms"] = elapsed_ms
            d.setdefault("pass_iter", sweep)
            deltas.append(d)
            if delta_made_progress(d):
                sweep_progress = True

        # ── 3. Convergence + budget check ───────────────────────────
        if not sweep_progress:
            terminated_by = "NoProgress"
            break
        if time_limit_ms > 0 and (time.monotonic() - started) * 1000.0 >= time_limit_ms:
            terminated_by = "TimeBudget"
            break

    # Rebuild bounds arrays if Python passes ran but Rust did not.
    if last_bounds_lo is None:
        import numpy as np

        n = model_repr.n_var_blocks
        last_bounds_lo = np.array(
            [model_repr.var_lb(i)[0] if model_repr.var_lb(i) else 0.0 for i in range(n)],
            dtype=np.float64,
        )
        last_bounds_hi = np.array(
            [model_repr.var_ub(i)[0] if model_repr.var_ub(i) else 0.0 for i in range(n)],
            dtype=np.float64,
        )

    stats = {
        "iterations": last_iter,
        "terminated_by": terminated_by,
        "deltas": deltas,
        "bounds_lo": last_bounds_lo,
        "bounds_hi": last_bounds_hi,
    }
    return model_repr, stats
