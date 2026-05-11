"""Root-stage presolve orchestrator (P1 of issue #53).

This module is the Python-side entry point into the Rust presolve
pipeline. It used to drive the kernels (`eliminate_variables`,
`reformulate_polynomial`, `fbbt_with_cutoff`) one-shot in a fixed
sequence; that role has moved into the Rust orchestrator
(`crates/discopt-core/src/presolve/orchestrator.rs`), exposed via
``PyModelRepr.presolve``.

The public API on this module is preserved exactly so that
``solver.py`` does not have to change:

- :func:`run_root_presolve` — runs the orchestrator at the root and
  returns ``(new_model_repr, stats)`` with the same dict shape callers
  already log against.
- :func:`propagate_bounds_to_model` — pushes tightened per-element
  bounds back into the Python ``Model`` object.
- :func:`run_reverse_ad_tightening` — Python-side reverse-mode interval
  AD pass (M9 of #51); kept here unchanged because it operates on the
  Python ``Model`` object, not the Rust ``ModelRepr``.

Sequencing (matches the orchestrator's default pass order):

1. ``eliminate`` — singleton-equality variable fixing.
2. ``polynomial_reform`` — opt-in; rewrites high-degree monomials to
   nested bilinear products with auxiliary variables.
3. ``simplify`` — integer rounding, big-M strengthening, redundant-
   constraint detection.
4. ``fbbt`` — forward/backward bound propagation.
5. ``probing`` — binary-variable probing.

The orchestrator iterates the list to a fixed point under a global
budget rather than running each pass once.
"""

from __future__ import annotations

from typing import Any


def run_root_presolve(
    model_repr,
    *,
    eliminate: bool = True,
    aggregate: bool = True,
    redundancy: bool = True,
    polynomial: bool = False,
    implied_bounds: bool = True,
    scaling: bool = False,
    cliques: bool = False,
    reduced_cost: bool = False,
    reduced_cost_info: dict | None = None,
    reduction_constraints: bool = False,
    coefficient_strengthening: bool = True,
    factorable_elim: bool = True,
    fbbt: bool = True,
    fbbt_fixed_point: bool = False,
    fbbt_max_iter: int = 20,
    fbbt_tol: float = 1e-8,
    simplify: bool = True,
    probing: bool = True,
    max_iterations: int = 16,
    time_limit_ms: int = 0,
    python_passes=None,
) -> tuple[Any, dict]:
    """Run the root-stage presolve sequence on ``model_repr``.

    Args:
        model_repr: a ``PyModelRepr`` produced by ``model_to_repr``.
        eliminate: enable singleton-equality variable elimination (M10).
        polynomial: enable polynomial reformulation (M4 + M5). Off by
            default because it adds auxiliary variables and changes the
            shape of the variable index space; downstream code that
            assumes a fixed n_vars must opt in.
        fbbt: enable forward/backward bound tightening.
        fbbt_max_iter, fbbt_tol: forwarded to the FBBT kernel.
        simplify: enable integer rounding / big-M / redundancy pass.
        probing: enable binary-variable probing.
        max_iterations: cap on full sweeps over the pass list. The
            orchestrator stops earlier if a sweep makes no progress.
        time_limit_ms: wall-clock cap (0 disables).

    Returns:
        ``(new_model_repr, stats)`` where ``stats`` is a dict mirroring
        the historical keys (``elimination``, ``polynomial``, ``fbbt``)
        plus the new orchestrator metadata (``iterations``,
        ``terminated_by``, ``deltas``). Callers that only inspect the
        legacy keys keep working unchanged.
    """
    pass_names: list[str] = []
    if eliminate:
        pass_names.append("eliminate")
    if factorable_elim:
        pass_names.append("factorable_elim")
    if aggregate:
        pass_names.append("aggregate")
    if redundancy:
        pass_names.append("redundancy")
    if polynomial:
        pass_names.append("polynomial_reform")
    if simplify:
        pass_names.append("simplify")
    if coefficient_strengthening:
        pass_names.append("coefficient_strengthening")
    if scaling:
        pass_names.append("scaling")
    if implied_bounds:
        pass_names.append("implied_bounds")
    if fbbt:
        pass_names.append("fbbt")
    if fbbt_fixed_point:
        pass_names.append("fbbt_fixed_point")
    if probing:
        pass_names.append("probing")
    if cliques:
        pass_names.append("cliques")
    if reduced_cost:
        pass_names.append("reduced_cost_fixing")
    if reduction_constraints:
        pass_names.append("reduction_constraints")

    if not pass_names and not python_passes:
        return model_repr, {}

    if python_passes:
        # A3 handshake path: interleave Python passes between Rust
        # orchestrator sweeps to a fixed point.
        from discopt._jax.presolve.orchestrator import run_orchestrated_presolve

        new_repr, raw = run_orchestrated_presolve(
            model_repr,
            rust_passes=pass_names,
            python_passes=python_passes,
            max_iterations=max_iterations,
            time_limit_ms=time_limit_ms,
            rust_kwargs={
                "fbbt_max_iter": fbbt_max_iter,
                "fbbt_tol": fbbt_tol,
                "reduced_cost_info": reduced_cost_info,
            },
        )
    else:
        new_repr, raw = model_repr.presolve(
            passes=pass_names,
            max_iterations=max_iterations,
            time_limit_ms=time_limit_ms,
            work_unit_budget=0,
            fbbt_max_iter=fbbt_max_iter,
            fbbt_tol=fbbt_tol,
            reduced_cost_info=reduced_cost_info,
        )

    stats: dict = {
        "iterations": raw["iterations"],
        "terminated_by": raw["terminated_by"],
        "deltas": list(raw["deltas"]),
    }

    # Synthesize legacy per-pass dicts from the delta log for backward
    # compatibility with callers that grep for "elimination",
    # "polynomial", "fbbt".
    elim_total = {"variables_fixed": 0, "constraints_removed": 0, "candidates_examined": 0}
    poly_total = {
        "constraints_rewritten": 0,
        "constraints_skipped": 0,
        "aux_variables_introduced": 0,
        "aux_constraints_introduced": 0,
        "aux_bounds_derived": 0,
    }
    agg_total: dict = {
        "variables_aggregated": 0,
        "equalities_dropped": 0,
        "candidates_examined": 0,
        "aggregations": [],
    }
    redundancy_total = {
        "constraints_removed": 0,
        "pairs_examined": 0,
    }
    implied_total = {
        "bounds_tightened": 0,
        "linear_rows_examined": 0,
    }
    scaling_total: dict = {
        "row_scales": None,
        "col_scales": None,
        "linear_rows_sampled": 0,
    }
    cliques_total: dict = {
        "edges": [],
        "linear_rows_scanned": 0,
    }
    rcf_total: dict = {
        "bounds_tightened": 0,
        "vars_fixed": [],
        "blocks_examined": 0,
    }
    rc_total: dict = {
        "bounds_tightened": 0,
        "vars_fixed_to_zero": [],
        "constraints_made_redundant": [],
    }
    for d in raw["deltas"]:
        if d["pass_name"] == "eliminate":
            elim_total["constraints_removed"] += len(d.get("constraints_removed", []))
            elim_total["candidates_examined"] += int(d.get("work_units", 0))
            elim_total["variables_fixed"] += int(d.get("bounds_tightened", 0)) // 2
        elif d["pass_name"] == "polynomial_reform":
            poly_total["aux_variables_introduced"] += int(d.get("aux_vars_introduced", 0))
            poly_total["aux_constraints_introduced"] += int(d.get("aux_constraints_introduced", 0))
        elif d["pass_name"] == "aggregate":
            entries = d.get("vars_aggregated", []) or []
            agg_total["variables_aggregated"] += len(entries)
            agg_total["equalities_dropped"] += len(d.get("constraints_removed", []))
            agg_total["candidates_examined"] += int(d.get("work_units", 0))
            agg_total["aggregations"].extend(entries)
        elif d["pass_name"] == "redundancy":
            redundancy_total["constraints_removed"] += len(d.get("constraints_removed", []))
            redundancy_total["pairs_examined"] += int(d.get("work_units", 0))
        elif d["pass_name"] == "implied_bounds":
            implied_total["bounds_tightened"] += int(d.get("bounds_tightened", 0))
            implied_total["linear_rows_examined"] += int(d.get("work_units", 0))
        elif d["pass_name"] == "scaling":
            if d.get("row_scales") is not None:
                scaling_total["row_scales"] = list(d["row_scales"])
            if d.get("col_scales") is not None:
                scaling_total["col_scales"] = list(d["col_scales"])
            scaling_total["linear_rows_sampled"] = int(d.get("work_units", 0))
        elif d["pass_name"] == "cliques":
            edges = d.get("cliques", []) or []
            cliques_total["edges"] = list(edges)
            cliques_total["linear_rows_scanned"] = int(d.get("work_units", 0))
        elif d["pass_name"] == "reduced_cost_fixing":
            rcf_total["bounds_tightened"] += int(d.get("bounds_tightened", 0))
            rcf_total["blocks_examined"] += int(d.get("work_units", 0))
            rcf_total["vars_fixed"].extend(d.get("vars_fixed", []) or [])
        elif d["pass_name"] == "reduction_constraints":
            rc_total["bounds_tightened"] += int(d.get("bounds_tightened", 0))
            rc_total["vars_fixed_to_zero"].extend(idx for idx, _ in (d.get("vars_fixed", []) or []))
            rc_total["constraints_made_redundant"].extend(d.get("constraints_removed", []) or [])
    if eliminate:
        stats["elimination"] = elim_total
    if aggregate:
        stats["aggregation"] = agg_total
    if redundancy:
        stats["redundancy"] = redundancy_total
    if implied_bounds:
        stats["implied_bounds"] = implied_total
    if scaling:
        stats["scaling"] = scaling_total
    if cliques:
        stats["cliques"] = cliques_total
    if reduced_cost:
        stats["reduced_cost_fixing"] = rcf_total
    if reduction_constraints:
        stats["reduction_constraints"] = rc_total
    if polynomial:
        stats["polynomial"] = poly_total
    if fbbt:
        stats["fbbt"] = {"lb": raw["bounds_lo"], "ub": raw["bounds_hi"]}

    return new_repr, stats


def propagate_bounds_to_model(model, model_repr) -> int:
    """Push tightened per-element bounds from ``model_repr`` back into ``model``.

    This is the bridge that lets the Rust-side presolve outcome influence
    the Python-side relaxation compiler / branch-and-bound initialisation.
    Only blocks whose count and flat element count are unchanged from the
    original model are touched. Aux variables introduced by polynomial
    reformulation have no Python-side counterpart and are silently
    skipped.

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
