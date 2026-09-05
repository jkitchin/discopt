#!/usr/bin/env python
"""#1180 deliverable 3 -- separate the three node-NLP candidates that survived the
engine swap.

#1026 listed four candidates for the residual per-node gap. The fourth
("per-iterate JAX dispatch latency") no longer exists -- the tape evaluator
replaced the JAX one. The three that survive, and how each is measured here:

1. **POUNCE iteration count per node-NLP.** Every ``solve_nlp`` on the POUNCE
   path is recorded: iterations, wall, status, size, and the callback counts it
   drove. Root-phase solves (the multistart, which runs before the first node
   callback) are separated from tree-phase node solves, because they are
   different work and averaging them hides both.

2. **Python frame overhead in ``_IpoptCallbacks`` / ``_BoundOverrideEvaluator``.**
   The callback path is five Python frames deep before any arithmetic happens:
   ``_IpoptCallbacks.<cb>`` -> ``_charge_evaluator.wrapper`` -> ``_timing.charge``
   (a generator-based context manager) -> ``_BoundOverrideEvaluator.__getattr__``
   -> ``TapeNLPEvaluator.evaluate_*`` -> ``_x`` (a per-call Python list
   comprehension over n) -> the native tape. Each layer is timed separately,
   interleaved, with a standard deviation (CLAUDE.md §9), against the pure native
   call at the bottom.

3. **Warm-start quality -- is every node re-solving from scratch?** discopt hands
   POUNCE a primal point only (the parent's solution, clipped into the child
   box). POUNCE 0.11 accepts a full ``WarmStart`` (multipliers, bound
   multipliers, barrier parameter); nothing in ``discopt`` constructs one. The
   A/B re-solves each captured node subproblem three ways -- production warm x0,
   cold box-midpoint x0, and the preceding solve's full ``WarmStart`` -- and
   compares iteration counts.

Nothing here changes solver behavior: the A/B re-solves happen after the solve,
on captured problem objects, and their results are discarded.

Measurement discipline: executed-assertion count printed, non-zero exit at zero;
no swallowed exceptions; ``discopt.__file__`` plus a version-unique marker checked
before any number is believed; per-item progress printed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

ASSERTS = {"n": 0}


def check(cond: bool, msg: str) -> None:
    ASSERTS["n"] += 1
    if not cond:
        raise AssertionError(msg)


CB_NAMES = ("objective", "gradient", "constraints", "jacobian", "hessian")


def instrument():
    """Wrap the callback class and the POUNCE solve entry point.

    Returns ``(state, restore)``. The wrappers only count and record; they never
    catch an exception (CLAUDE.md §7).
    """
    import pounce

    from discopt.solvers import nlp_ipopt, nlp_pounce

    state: dict = {
        "phase": "root",
        "cb_counts": dict.fromkeys(CB_NAMES, 0),
        "solves": [],
        "last_callbacks": None,
        "last_x": None,
        "captures": [],  # (problem, info, x0, phase)
    }

    originals = {n: getattr(nlp_ipopt._IpoptCallbacks, n) for n in CB_NAMES}
    orig_init = nlp_ipopt._IpoptCallbacks.__init__
    orig_solve_nlp = nlp_pounce.solve_nlp
    orig_problem_solve = pounce.Problem.solve

    def make_cb(name, fn):
        def wrapper(self, *a, **k):
            state["cb_counts"][name] += 1
            if name == "objective" and a:
                state["last_x"] = np.array(a[0], dtype=np.float64)
            return fn(self, *a, **k)

        wrapper.__name__ = fn.__name__
        return wrapper

    def init_wrapper(self, evaluator):
        orig_init(self, evaluator)
        state["last_callbacks"] = self

    def problem_solve_wrapper(self, *a, **k):
        out = orig_problem_solve(self, *a, **k)
        state["_pending"] = (self, out[1], np.array(a[0]) if a else None)
        return out

    def solve_nlp_wrapper(evaluator, x0, *a, **k):
        before = dict(state["cb_counts"])
        state.pop("_pending", None)
        t0 = time.perf_counter()
        res = orig_solve_nlp(evaluator, x0, *a, **k)
        wall = time.perf_counter() - t0
        delta = {n: state["cb_counts"][n] - before[n] for n in CB_NAMES}
        lb, ub = evaluator.variable_bounds
        rec = {
            "phase": state["phase"],
            "iterations": int(res.iterations or 0),
            "status": str(res.status),
            "wall_s": wall,
            "n": int(evaluator.n_variables),
            "m": int(evaluator.n_constraints),
            "callbacks": delta,
            "callbacks_total": sum(delta.values()),
        }
        pending = state.get("_pending")
        if pending is not None:
            problem, info, px0 = pending
            mid = 0.5 * (np.clip(lb, -1e6, 1e6) + np.clip(ub, -1e6, 1e6))
            rec["x0_is_midpoint"] = bool(
                px0 is not None and np.allclose(px0, mid, rtol=0, atol=1e-9)
            )
            rec["x0_dist_to_midpoint"] = (
                float(np.linalg.norm(px0 - mid)) if px0 is not None else None
            )
            state["captures"].append(
                {"problem": problem, "info": info, "x0": px0, "mid": mid, "phase": state["phase"]}
            )
        state["solves"].append(rec)
        return res

    for n, fn in originals.items():
        setattr(nlp_ipopt._IpoptCallbacks, n, make_cb(n, fn))
    nlp_ipopt._IpoptCallbacks.__init__ = init_wrapper
    nlp_pounce.solve_nlp = solve_nlp_wrapper
    pounce.Problem.solve = problem_solve_wrapper

    def restore():
        for n, fn in originals.items():
            setattr(nlp_ipopt._IpoptCallbacks, n, fn)
        nlp_ipopt._IpoptCallbacks.__init__ = orig_init
        nlp_pounce.solve_nlp = orig_solve_nlp
        pounce.Problem.solve = orig_problem_solve

    return state, restore


def timed(fn, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def callback_layer_costs(callbacks, x, reps: int, rounds: int) -> dict:
    """Cost of each layer between POUNCE and the tape, interleaved with a spread.

    Interleaved (all arms per round, `rounds` rounds) rather than sequential, so a
    frequency or load excursion hits every arm alike (CLAUDE.md §9).
    """
    proxy = callbacks._ev
    tape = getattr(proxy, "_evaluator", proxy)
    check(hasattr(tape, "_problem"), "evaluator is not the tape evaluator; wrong backend")
    problem = tape._problem
    xl = [float(v) for v in np.asarray(x, dtype=float).ravel()]
    m = int(tape.n_constraints)

    arms = {
        "full_callback_objective": lambda: callbacks.objective(x),
        "proxy_evaluate_objective": lambda: proxy.evaluate_objective(x),
        "tape_evaluate_objective": lambda: tape.evaluate_objective(x),
        "native_plus_listbuild": lambda: problem.objective(tape._x(x)),
        "native_only": lambda: problem.objective(xl),
    }
    if m > 0:
        arms.update(
            {
                "full_callback_constraints": lambda: callbacks.constraints(x),
                "tape_evaluate_constraints": lambda: tape.evaluate_constraints(x),
                "native_constraints_only": lambda: problem.constraints(xl),
            }
        )
    for fn in arms.values():  # warm up every arm before any is timed
        fn()
    samples: dict[str, list[float]] = {k: [] for k in arms}
    for _ in range(rounds):
        for label, fn in arms.items():
            samples[label].append(timed(fn, reps))
    out = {}
    for label, s in samples.items():
        out[label] = {
            "median_us": statistics.median(s) * 1e6,
            "sd_us": (statistics.stdev(s) * 1e6) if len(s) > 1 else 0.0,
            "rounds": rounds,
            "reps": reps,
        }
    return out


def warm_start_ab(captures: list, max_cases: int) -> dict:
    """Re-solve captured node subproblems from three starts and compare iterations."""
    import pounce

    cases = []
    tree = [c for c in captures if c["phase"] == "tree"]
    pool = tree if tree else captures
    prev_info = None
    n_ws_ok = 0
    for c in pool[:max_cases]:
        problem, info, x0, mid = c["problem"], c["info"], c["x0"], c["mid"]
        row: dict = {"phase": c["phase"], "n": problem.n, "m": problem.m}
        t0 = time.perf_counter()
        _, i_warm = problem.solve(x0=np.asarray(x0, dtype=np.float64))
        row["warm_x0"] = {
            "iters": int(i_warm.get("iter_count", -1)),
            "status": int(i_warm.get("status", -100)),
            "wall_s": time.perf_counter() - t0,
        }
        t0 = time.perf_counter()
        _, i_cold = problem.solve(x0=np.asarray(mid, dtype=np.float64))
        row["cold_midpoint"] = {
            "iters": int(i_cold.get("iter_count", -1)),
            "status": int(i_cold.get("status", -100)),
            "wall_s": time.perf_counter() - t0,
        }
        if prev_info is not None:
            # Unsigned state (no ``problem=``): dimensions are still checked
            # against the arrays, the rest is deliberately unverified because the
            # child box differs from the parent's by construction.
            ws = pounce.WarmStart.from_info(prev_info)
            t0 = time.perf_counter()
            _, i_ws = problem.solve(x0=np.asarray(x0, dtype=np.float64), warm_start=ws)
            row["prev_warm_start"] = {
                "iters": int(i_ws.get("iter_count", -1)),
                "status": int(i_ws.get("status", -100)),
                "wall_s": time.perf_counter() - t0,
            }
            n_ws_ok += 1
        prev_info = info
        cases.append(row)

    def med(key, field="iters"):
        vals = [c[key][field] for c in cases if key in c and c[key][field] >= 0]
        return statistics.median(vals) if vals else None

    return {
        "n_cases": len(cases),
        "n_with_prev_warm_start": n_ws_ok,
        "median_iters_warm_x0": med("warm_x0"),
        "median_iters_cold_midpoint": med("cold_midpoint"),
        "median_iters_prev_warm_start": med("prev_warm_start"),
        "median_wall_warm_x0_s": med("warm_x0", "wall_s"),
        "median_wall_cold_midpoint_s": med("cold_midpoint", "wall_s"),
        "median_wall_prev_warm_start_s": med("prev_warm_start", "wall_s"),
        "cases": cases,
    }


def run_instance(nl: str, time_limit: float, reps: int, rounds: int, max_cases: int) -> dict:
    from discopt.modeling.core import from_nl

    state, restore = instrument()
    try:
        model = from_nl(nl)

        def node_cb(_ctx, _model):
            state["phase"] = "tree"

        t0 = time.perf_counter()
        result = model.solve(time_limit=time_limit, gap_tolerance=1e-4, node_callback=node_cb)
        wall = time.perf_counter() - t0

        solves = state["solves"]
        check(len(solves) > 0, f"{nl}: no POUNCE NLP solve was observed -- probe measured nothing")
        root = [s for s in solves if s["phase"] == "root"]
        tree = [s for s in solves if s["phase"] == "tree"]

        cb_costs = None
        if state["last_callbacks"] is not None and state["last_x"] is not None:
            cb_costs = callback_layer_costs(
                state["last_callbacks"], state["last_x"], reps, rounds
            )
        ab = warm_start_ab(state["captures"], max_cases)
    finally:
        restore()

    def summarize(rows):
        if not rows:
            return None
        return {
            "n_solves": len(rows),
            "median_iterations": statistics.median(r["iterations"] for r in rows),
            "mean_iterations": statistics.mean(r["iterations"] for r in rows),
            "max_iterations": max(r["iterations"] for r in rows),
            "median_wall_ms": statistics.median(r["wall_s"] for r in rows) * 1e3,
            "total_wall_s": sum(r["wall_s"] for r in rows),
            "median_callbacks_per_solve": statistics.median(r["callbacks_total"] for r in rows),
            "total_callbacks": sum(r["callbacks_total"] for r in rows),
            "statuses": sorted({r["status"] for r in rows}),
            "n": rows[0]["n"],
            "m": rows[0]["m"],
        }

    return {
        "instance": os.path.basename(nl),
        "wall_s": wall,
        "nodes": int(result.node_count),
        "status": str(result.status),
        "objective": None if result.objective is None else float(result.objective),
        "bound": None if result.bound is None else float(result.bound),
        "nlp_root_multistart": summarize(root),
        "nlp_tree_nodes": summarize(tree),
        "nlp_all": summarize(solves),
        "callback_layer_costs_us": cb_costs,
        "warm_start_ab": ab,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", default="nvs05")
    ap.add_argument("--nl-dir", default="python/tests/data/minlplib_nl")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--max-ab-cases", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import discopt

    check(
        os.path.abspath(discopt.__file__).startswith(os.path.abspath("python/discopt")),
        f"discopt imported from {discopt.__file__}, not the worktree under test",
    )
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    check(TapeNLPEvaluator.timing_bucket == "rust", "build predates the POUNCE-tape default")

    records = []
    for name in [s.strip() for s in args.instances.split(",") if s.strip()]:
        nl = os.path.join(args.nl_dir, f"{name}.nl")
        check(os.path.exists(nl), f"missing instance {nl}")
        print(f"[{name}] solving with instrumentation ...", flush=True)
        rec = run_instance(nl, args.time_limit, args.reps, args.rounds, args.max_ab_cases)
        records.append(rec)
        for key in ("nlp_root_multistart", "nlp_tree_nodes"):
            s = rec[key]
            if s:
                print(
                    f"[{name}] {key}: {s['n_solves']} solves, median {s['median_iterations']:.0f} "
                    f"iters, {s['median_callbacks_per_solve']:.0f} callbacks/solve, "
                    f"{s['median_wall_ms']:.0f} ms/solve",
                    flush=True,
                )
        ab = rec["warm_start_ab"]
        print(
            f"[{name}] warm-start A/B on {ab['n_cases']} subproblems: "
            f"warm x0 {ab['median_iters_warm_x0']}, cold midpoint "
            f"{ab['median_iters_cold_midpoint']}, prev WarmStart "
            f"{ab['median_iters_prev_warm_start']} (median iterations)",
            flush=True,
        )
        c = rec["callback_layer_costs_us"]
        if c:
            for k in sorted(c):
                print(f"    {k:32s} {c[k]['median_us']:8.2f} us  (sd {c[k]['sd_us']:.2f})",
                      flush=True)

    out = {
        "probe": "issue1180_node_nlp_candidates",
        "discopt_file": discopt.__file__,
        "time_limit_s": args.time_limit,
        "records": records,
        "executed_assertions": ASSERTS["n"],
    }
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(out, indent=1))
    print(f"\nexecuted assertions: {ASSERTS['n']}")
    if ASSERTS["n"] == 0:
        print("PROBE MEASURED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
