#!/usr/bin/env python
"""PYPROF-1 coverage map: which solve path + which capabilities fire per instance.

Measurement-only. Monkeypatches (does NOT edit) the solver.py dispatch entry
points and the recent-capability functions with call counters + a tighten
detector, then solves one .nl instance and prints a coverage row:

  instance | path | run_root_fixpoint(calls/tightened) | reduce_node(calls) |
  PSD(calls) | wall | nodes

"Path" is the first dispatch function actually entered (convex/GP fast path,
QP/MIQP fast path, MILP driver _solve_milp_bb, or the general spatial
_solve_nlp_bb McCormick LP-relaxer path).

Run with DISCOPT_ROOT_FIXPOINT=1 DISCOPT_NODE_REDUCE=1 DISCOPT_PSD_COST_GATE=1
to see whether the capabilities fire when *enabled* (they are default-OFF).

Usage
-----
  python coverage_map.py INST.nl --time-limit 60 --json out.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def _tightened(lb0, ub0, lb1, ub1) -> bool:
    try:
        return bool(np.any(np.asarray(lb1) > np.asarray(lb0) + 1e-12)) or bool(
            np.any(np.asarray(ub1) < np.asarray(ub0) - 1e-12)
        )
    except Exception:
        return False


def run(args: argparse.Namespace) -> dict:
    import discopt.solver as solver_mod
    from discopt.modeling.core import from_nl

    counters = {
        "path": None,
        "root_fixpoint_calls": 0,
        "root_fixpoint_tightened": 0,
        "reduce_node_calls": 0,
        "reduce_node_tightened": 0,
        "psd_calls": 0,
    }

    # --- wrap dispatch entry points; first one hit records the path ---
    def wrap_path(name, fn):
        def w(*a, **k):
            if counters["path"] is None:
                counters["path"] = name
            return fn(*a, **k)

        return w

    for pathname, attr in [
        ("spatial:_solve_nlp_bb", "_solve_nlp_bb"),
        ("milp:_solve_milp_bb", "_solve_milp_bb"),
        ("milp:_solve_milp_simplex", "_solve_milp_simplex"),
        ("qp:_solve_qp", "_solve_qp"),
        ("lp:_solve_lp", "_solve_lp"),
    ]:
        if hasattr(solver_mod, attr):
            setattr(solver_mod, attr, wrap_path(pathname, getattr(solver_mod, attr)))

    # convex fast path may be a nested closure; also detect via classify result
    # by wrapping the convex solver if present.
    for pathname, attr in [
        ("convex:_solve_convex_nlp", "_solve_convex_nlp"),
        ("convex:_solve_convex", "_solve_convex"),
        ("gp:_solve_gp", "_solve_gp"),
    ]:
        if hasattr(solver_mod, attr):
            setattr(solver_mod, attr, wrap_path(pathname, getattr(solver_mod, attr)))

    # --- wrap capabilities ---
    try:
        import discopt._jax.root_reduce as root_reduce_mod

        _orig_rf = root_reduce_mod.run_root_fixpoint

        def wrap_rf(model, lb, ub, *a, **k):
            counters["root_fixpoint_calls"] += 1
            lb0, ub0 = np.array(lb, float), np.array(ub, float)
            res = _orig_rf(model, lb, ub, *a, **k)
            if _tightened(lb0, ub0, getattr(res, "lb", lb0), getattr(res, "ub", ub0)):
                counters["root_fixpoint_tightened"] += 1
            return res

        root_reduce_mod.run_root_fixpoint = wrap_rf
        # solver.py may have imported the name directly
        if hasattr(solver_mod, "run_root_fixpoint"):
            solver_mod.run_root_fixpoint = wrap_rf
    except Exception as e:
        counters["_rf_wrap_err"] = str(e)

    try:
        import discopt._jax.node_reduce as node_reduce_mod

        _orig_nr = node_reduce_mod.reduce_node

        def wrap_nr(model, node_lb, node_ub, *a, **k):
            counters["reduce_node_calls"] += 1
            lb0, ub0 = np.array(node_lb, float), np.array(node_ub, float)
            res = _orig_nr(model, node_lb, node_ub, *a, **k)
            if _tightened(lb0, ub0, getattr(res, "lb", lb0), getattr(res, "ub", ub0)):
                counters["reduce_node_tightened"] += 1
            return res

        node_reduce_mod.reduce_node = wrap_nr
        if hasattr(solver_mod, "reduce_node"):
            solver_mod.reduce_node = wrap_nr
    except Exception as e:
        counters["_nr_wrap_err"] = str(e)

    try:
        import discopt._jax.mccormick_lp as mccormick_lp_mod

        if hasattr(mccormick_lp_mod, "MccormickLPRelaxer") and hasattr(
            mccormick_lp_mod.MccormickLPRelaxer, "_separate_psd"
        ):
            _orig_psd = mccormick_lp_mod.MccormickLPRelaxer._separate_psd

            def wrap_psd(self, *a, **k):
                counters["psd_calls"] += 1
                return _orig_psd(self, *a, **k)

            mccormick_lp_mod.MccormickLPRelaxer._separate_psd = wrap_psd
    except Exception as e:
        counters["_psd_wrap_err"] = str(e)

    model = from_nl(args.instance)
    t0 = time.perf_counter()
    result = model.solve(time_limit=args.time_limit, gap_tolerance=1e-4)
    wall = time.perf_counter() - t0

    return {
        "instance": args.instance,
        "path": counters["path"],
        "run_root_fixpoint_calls": counters["root_fixpoint_calls"],
        "run_root_fixpoint_tightened": counters["root_fixpoint_tightened"],
        "reduce_node_calls": counters["reduce_node_calls"],
        "reduce_node_tightened": counters["reduce_node_tightened"],
        "psd_calls": counters["psd_calls"],
        "wall_s": wall,
        "node_count": result.node_count,
        "status": str(result.status),
        "objective": None if result.objective is None else float(result.objective),
        "_errs": {k: v for k, v in counters.items() if k.startswith("_")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    rec = run(args)
    text = json.dumps(rec, indent=1, default=str)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
