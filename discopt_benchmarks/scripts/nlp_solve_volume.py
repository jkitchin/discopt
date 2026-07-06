#!/usr/bin/env python3
"""VOLUME-1 — attribute POUNCE NLP-solve volume to its call-site SOURCES.

Instruments the single shared POUNCE NLP entry point
(``discopt.solvers.nlp_pounce.solve_nlp``) and buckets every NLP solve by the
nearest identifying caller frame on the Python stack (root multistart,
feasibility_pump, diving, rins, rens, subnlp, local_branching, per-node
incumbent NLP, OBBT probe, ...).  For each bucket it records #solves and total
wall.  It also wraps the primal-heuristic / multistart *entry* functions to
measure the per-source **incumbent-improvement hit rate**: how often a call from
that source returns a feasible candidate strictly better than the best-so-far.

Pure monkeypatch — no production edit.  Measurement-only.

Usage:
    PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python nlp_solve_volume.py <instance.nl> [--time-limit 60] [--gap 1e-4]

Env knobs for the prototype cut (all default-OFF, read by solver via env, not
here) are echoed in the report so the caller can confirm the cut fired.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import time
from collections import defaultdict

import numpy as np

# ---- source buckets: caller-frame function names we attribute solves to -----
# Ordered by specificity; the *nearest* (deepest) match on the stack wins so a
# subnlp launched from inside rens is charged to subnlp/rens correctly.
HEURISTIC_SOURCES = [
    "feasibility_pump",
    "subnlp",
    "diving",
    "fractional_diving",
    "objective_diving",
    "rins",
    "rens",
    "local_branching",
    "_local_branching_submip",
    "integer_local_search",
    "integer_box_search",
    "enumerate_binary_seeds_subnlp",
]
# node / root engine sources
ENGINE_SOURCES = [
    "_solve_root_node_multistart",  # root multistart diversification
    "_solve_node_nlp_pounce",  # per-node incumbent/bound NLP + _attempt retries
    "_polish",  # OBBT / polishing probes (best-effort name match)
    "obbt",
]
ALL_SOURCE_NAMES = HEURISTIC_SOURCES + ENGINE_SOURCES


def _classify_stack() -> str:
    """Walk the live stack and return the nearest identifying source label.

    The deepest (closest to solve_nlp) matching frame wins.  When both a
    heuristic frame and an engine frame are present (e.g. a diving call routes
    through _solve_node_nlp_pounce), the HEURISTIC label wins because it is the
    semantic source of the solve — the engine wrapper is shared plumbing.  We do
    this by scanning outward and recording the first heuristic hit; if none,
    fall back to the first engine hit; else 'other'.
    """
    # inner helpers whose true source is a heuristic further out on the stack
    # (e.g. subnlp's inner _objective_improve loop actually belongs to
    # integer_local_search).  Skip subnlp when a wider heuristic frame exists.
    stack = inspect.stack()
    engine_hit = None
    outer_heuristic = None  # first (widest) heuristic that owns the solve
    subnlp_seen = False
    try:
        for fr in stack:
            name = fr.function
            if name == "subnlp":
                subnlp_seen = True
            if name in HEURISTIC_SOURCES:
                # remember the widest heuristic; but a direct subnlp with no
                # wider heuristic is charged to subnlp itself
                outer_heuristic = name
            if engine_hit is None and name in ENGINE_SOURCES:
                engine_hit = name
    finally:
        del stack
    if outer_heuristic is not None:
        # prefer the widest non-subnlp heuristic (integer_local_search etc.)
        return outer_heuristic
    if subnlp_seen:
        return "subnlp"
    if engine_hit is not None:
        return engine_hit
    return "other"


class VolumeTracker:
    def __init__(self):
        self.count = defaultdict(int)  # source -> #solves
        self.wall = defaultdict(float)  # source -> total wall (s)
        self.iters = defaultdict(int)  # source -> total IPM iters
        # hit-rate bookkeeping (per heuristic/multistart entry)
        self.entry_calls = defaultdict(int)  # source -> #entry invocations
        self.entry_hits = defaultdict(int)  # source -> #that improved best
        self.best_obj = np.inf  # running best feasible incumbent (min sense)
        self.total_solves = 0
        self.total_wall = 0.0


def install(tracker: VolumeTracker, sense_min: bool = True):
    """Monkeypatch solve_nlp and the heuristic entry functions."""
    import discopt._jax.primal_heuristics as ph
    import discopt.solver as solver_mod
    import discopt.solvers.nlp_pounce as npc

    orig_solve_nlp = npc.solve_nlp

    def wrapped_solve_nlp(*a, **k):
        src = _classify_stack()
        t0 = time.perf_counter()
        res = orig_solve_nlp(*a, **k)
        dt = time.perf_counter() - t0
        tracker.count[src] += 1
        tracker.wall[src] += dt
        with contextlib.suppress(Exception):
            tracker.iters[src] += int(getattr(res, "iterations", 0) or 0)
        tracker.total_solves += 1
        tracker.total_wall += dt
        return res

    npc.solve_nlp = wrapped_solve_nlp
    # solver.py imports solve_nlp lazily as solve_nlp_pounce inside functions, so
    # patching the module attribute is enough (the lazy `from ... import` runs at
    # call time and picks up the patched attribute).

    patched = [("nlp_pounce.solve_nlp", orig_solve_nlp)]

    # ---- hit-rate wrappers on heuristic + multistart ENTRY functions --------
    def _extract_obj(ret):
        """Best-effort feasible objective from a heuristic return value.

        Heuristics return (x, obj), or a candidate object, or None.  We only
        need the objective (min sense) to test improvement.
        """
        if ret is None:
            return None
        if isinstance(ret, tuple) and len(ret) >= 2:
            cand_x, cand_obj = ret[0], ret[1]
            if cand_x is None:
                return None
            try:
                return float(cand_obj)
            except (TypeError, ValueError):
                return None
        # NLPResult-like
        obj = getattr(ret, "objective", None)
        status = getattr(ret, "status", None)
        if obj is not None:
            from discopt.solvers import SolveStatus

            if status is not None and status not in (
                SolveStatus.OPTIMAL,
                SolveStatus.ITERATION_LIMIT,
            ):
                return None
            try:
                return float(obj)
            except (TypeError, ValueError):
                return None
        return None

    def make_hit_wrapper(name, orig):
        def wrapper(*a, **k):
            tracker.entry_calls[name] += 1
            ret = orig(*a, **k)
            obj = _extract_obj(ret)
            if obj is not None and np.isfinite(obj):
                # min sense: strictly better = smaller
                better = obj < tracker.best_obj - 1e-9
                if better:
                    tracker.entry_hits[name] += 1
                    tracker.best_obj = obj
            return ret

        return wrapper

    for name in [
        "feasibility_pump",
        "subnlp",
        "diving",
        "fractional_diving",
        "objective_diving",
        "rins",
        "rens",
        "local_branching",
        "integer_local_search",
        "integer_box_search",
        "enumerate_binary_seeds_subnlp",
    ]:
        if hasattr(ph, name):
            o = getattr(ph, name)
            patched.append((f"primal_heuristics.{name}", o))
            setattr(ph, name, make_hit_wrapper(name, o))

    # multistart entry: hit = did the root relaxation produce the FIRST feasible
    # incumbent (best_obj was inf -> finite).  Wrap _solve_root_node_multistart.
    if hasattr(solver_mod, "_solve_root_node_multistart"):
        o = solver_mod._solve_root_node_multistart
        patched.append(("solver._solve_root_node_multistart", o))

        def ms_wrapper(*a, **k):
            tracker.entry_calls["_solve_root_node_multistart"] += 1
            ret = o(*a, **k)
            obj = _extract_obj(ret)
            if obj is not None and np.isfinite(obj) and obj < tracker.best_obj - 1e-9:
                tracker.entry_hits["_solve_root_node_multistart"] += 1
                tracker.best_obj = obj
            return ret

        solver_mod._solve_root_node_multistart = ms_wrapper

    return patched


def run(path: str, time_limit: float, gap: float) -> dict:
    from discopt.modeling.core import from_nl

    tracker = VolumeTracker()
    model = from_nl(path)
    install(tracker)

    t0 = time.perf_counter()
    result = model.solve(time_limit=time_limit, gap_tolerance=gap)
    wall = time.perf_counter() - t0

    # assemble report
    rows = []
    for src in sorted(tracker.count, key=lambda s: -tracker.wall[s]):
        rows.append(
            {
                "source": src,
                "solves": tracker.count[src],
                "wall_s": round(tracker.wall[src], 3),
                "iters": tracker.iters[src],
            }
        )
    hit_rows = []
    for src in sorted(tracker.entry_calls, key=lambda s: -tracker.entry_calls[s]):
        calls = tracker.entry_calls[src]
        hits = tracker.entry_hits[src]
        hit_rows.append(
            {
                "source": src,
                "entry_calls": calls,
                "improved_incumbent": hits,
                "hit_rate": round(hits / calls, 4) if calls else 0.0,
            }
        )

    status = getattr(result, "status", None)
    obj = getattr(result, "objective", None)
    bound = getattr(result, "bound", None)
    nodes = getattr(result, "node_count", getattr(result, "nodes", None))
    return {
        "instance": path.split("/")[-1],
        "wall_s": round(wall, 3),
        "status": str(status),
        "objective": obj,
        "bound": bound,
        "nodes": nodes,
        "total_nlp_solves": tracker.total_solves,
        "total_nlp_wall_s": round(tracker.total_wall, 3),
        "attribution": rows,
        "hit_rates": hit_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    rep = run(args.instance, args.time_limit, args.gap)
    print(json.dumps(rep, indent=2, default=str))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rep, f, indent=2, default=str)


if __name__ == "__main__":
    main()
