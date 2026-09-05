#!/usr/bin/env python
"""#1180 deliverable 3 -- separate the three node-NLP candidates that survived the
engine swap.

#1026 listed four candidates for the residual per-node gap. The fourth
("per-iterate JAX dispatch latency") no longer exists -- the tape evaluator
replaced the JAX one. The three that survive, and how each is measured here:

1. **POUNCE iteration count per node-NLP.** Every ``solve_nlp`` on the POUNCE
   path is recorded: iterations, wall, status, size, and the callback counts it
   drove. The root multistart and the tree's node solves are separated by a
   companion ``max_nodes=1`` run rather than by a ``node_callback`` phase flag:
   attaching a callback is NOT observation-neutral, it is a documented routing
   signal, and it measures a different engine (see ``run_instance``).

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
import functools
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
    """Wrap the callback class and BOTH POUNCE entry points.

    Two entry points, not one: the serial ``pounce.Problem.solve`` and the batch
    ``pounce.solve_nlp_batch`` (``solver._solve_batch_pounce``). An earlier
    version of this probe wrapped ``nlp_pounce.solve_nlp`` only and asserted it
    fired -- which it did not on ``alan``, whose node NLPs all go through the
    batch path. The assertion caught it instead of reporting a vacuous zero,
    which is the point of having one.

    Returns ``(state, restore)``. The wrappers only count and record; they never
    catch an exception (CLAUDE.md §7).
    """
    import pounce

    from discopt.solvers import nlp_ipopt

    state: dict = {
        # The post-solve A/B re-solves the captured problems through the SAME
        # patched entry points. Without this flag they land in ``solves`` and
        # inflate both the solve count and the total NLP wall past the solve's
        # own wall -- which is exactly how a probe reports 89.7 s of NLP inside a
        # 26.4 s solve and reads as a finding.
        "recording": True,
        "cb_counts": dict.fromkeys(CB_NAMES, 0),
        "solves": [],
        "last_callbacks": None,
        "last_x": None,
        "captures": [],
        "batch_calls": 0,
        "non_problem_batches": 0,
    }

    originals = {n: getattr(nlp_ipopt._IpoptCallbacks, n) for n in CB_NAMES}
    orig_init = nlp_ipopt._IpoptCallbacks.__init__
    orig_problem_solve = pounce.Problem.solve
    orig_batch = pounce.solve_nlp_batch

    def make_cb(name, fn):
        @functools.wraps(fn)
        def wrapper(self, *a, **k):
            state["cb_counts"][name] += 1
            if name == "objective" and a:
                state["last_x"] = np.array(a[0], dtype=np.float64)
            return fn(self, *a, **k)

        return wrapper

    @functools.wraps(orig_init)
    def init_wrapper(self, evaluator):
        orig_init(self, evaluator)
        state["last_callbacks"] = self

    def _record(problem, x_star, info, x0, cb_delta, wall, source):
        lb, ub = problem.get_bounds()[:2] if hasattr(problem, "get_bounds") else (None, None)
        mid = None
        if lb is not None and ub is not None:
            mid = 0.5 * (
                np.clip(np.asarray(lb, dtype=float), -1e6, 1e6)
                + np.clip(np.asarray(ub, dtype=float), -1e6, 1e6)
            )
        rec = {
            "source": source,
            "iterations": int(info.get("iter_count", -1)),
            "status": int(info.get("status", -100)),
            "wall_s": wall,
            "n": int(problem.n),
            "m": int(problem.m),
            "callbacks": cb_delta,
            "callbacks_total": sum(cb_delta.values()),
        }
        if mid is not None and x0 is not None:
            rec["x0_is_midpoint"] = bool(np.allclose(np.asarray(x0), mid, rtol=0, atol=1e-9))
            rec["x0_dist_to_midpoint"] = float(np.linalg.norm(np.asarray(x0) - mid))
        state["solves"].append(rec)
        state["captures"].append(
            {
                "problem": problem,
                "x": None if x_star is None else np.asarray(x_star, dtype=np.float64),
                "info": info,
                "x0": x0,
                "mid": mid,
                }
        )

    def problem_solve_wrapper(self, *a, **k):
        if not state["recording"]:
            return orig_problem_solve(self, *a, **k)
        before = dict(state["cb_counts"])
        t0 = time.perf_counter()
        out = orig_problem_solve(self, *a, **k)
        wall = time.perf_counter() - t0
        delta = {n: state["cb_counts"][n] - before[n] for n in CB_NAMES}
        x0 = k.get("x0", a[0] if a else None)
        _record(
            self,
            out[0],
            out[1],
            np.asarray(x0) if x0 is not None else None,
            delta,
            wall,
            "serial",
        )
        return out

    def batch_wrapper(problems, *a, **k):
        if not state["recording"]:
            return orig_batch(problems, *a, **k)
        state["batch_calls"] += 1
        before = dict(state["cb_counts"])
        t0 = time.perf_counter()
        out = orig_batch(problems, *a, **k)
        wall = time.perf_counter() - t0
        delta = {n: state["cb_counts"][n] - before[n] for n in CB_NAMES}
        probs = list(problems)
        x0s = k.get("x0s") or (a[0] if a else None)
        n_inst = max(len(probs), 1)
        share = {n: v // n_inst for n, v in delta.items()}
        for i, p in enumerate(probs):
            if not isinstance(p, pounce.Problem):
                # The POUNCE-native .nl path hands NlProblem instances; it is
                # default-OFF, so this is recorded rather than handled.
                state["non_problem_batches"] += 1
                continue
            x0 = None
            if x0s is not None and i < len(x0s) and x0s[i] is not None:
                x0 = np.asarray(x0s[i], dtype=np.float64)
            _record(p, out[i][0], out[i][1], x0, share, wall / n_inst, "batch")
        return out

    for n, fn in originals.items():
        setattr(nlp_ipopt._IpoptCallbacks, n, make_cb(n, fn))
    nlp_ipopt._IpoptCallbacks.__init__ = init_wrapper
    pounce.Problem.solve = problem_solve_wrapper
    pounce.solve_nlp_batch = batch_wrapper

    def restore():
        for n, fn in originals.items():
            setattr(nlp_ipopt._IpoptCallbacks, n, fn)
        nlp_ipopt._IpoptCallbacks.__init__ = orig_init
        pounce.Problem.solve = orig_problem_solve
        pounce.solve_nlp_batch = orig_batch

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
    pool = [c for c in captures if c["mid"] is not None and c["x0"] is not None]
    prev = None
    n_ws_ok = 0
    n_dim_mismatch = 0
    for c in pool[:max_cases]:
        problem, info, x0, mid = c["problem"], c["info"], c["x0"], c["mid"]
        x_star = c["x"]
        row: dict = {"n": problem.n, "m": problem.m}
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
        if prev is not None and prev[2] == (problem.n, problem.m):
            # Unsigned state (no ``problem=``): dimensions are still checked
            # against the arrays, the rest is deliberately unverified because the
            # child box differs from the parent's by construction.
            ws = pounce.WarmStart.from_info(prev[0], prev[1])
            t0 = time.perf_counter()
            _, i_ws = problem.solve(x0=np.asarray(x0, dtype=np.float64), warm_start=ws)
            row["prev_warm_start"] = {
                "iters": int(i_ws.get("iter_count", -1)),
                "status": int(i_ws.get("status", -100)),
                "wall_s": time.perf_counter() - t0,
            }
            n_ws_ok += 1
        elif prev is not None:
            # A consecutive pair whose (n, m) differ cannot replay a dual state at
            # all: the cut pool changes the row count between node NLPs, so this
            # is a *structural* limit on warm-starting across nodes, not a probe
            # shortcoming. Counted, not hidden.
            n_dim_mismatch += 1
        prev = (x_star, info, (problem.n, problem.m)) if x_star is not None else None
        cases.append(row)

    def med(key, field="iters"):
        vals = [c[key][field] for c in cases if key in c and c[key][field] >= 0]
        return statistics.median(vals) if vals else None

    return {
        "n_cases": len(cases),
        "n_with_prev_warm_start": n_ws_ok,
        "n_prev_dim_mismatch": n_dim_mismatch,
        "median_iters_warm_x0": med("warm_x0"),
        "median_iters_cold_midpoint": med("cold_midpoint"),
        "median_iters_prev_warm_start": med("prev_warm_start"),
        "median_wall_warm_x0_s": med("warm_x0", "wall_s"),
        "median_wall_cold_midpoint_s": med("cold_midpoint", "wall_s"),
        "median_wall_prev_warm_start_s": med("prev_warm_start", "wall_s"),
        "cases": cases,
    }


def count_root_only(nl: str, time_limit: float) -> dict:
    """How many NLP solves a ``max_nodes=1`` (root-only) solve runs.

    The full run minus this is the tree's share -- the same construction the
    layer-split probe's ``--root-arm`` uses, and the reason neither needs a
    ``node_callback``.
    """
    from discopt.modeling.core import from_nl

    state, restore = instrument()
    try:
        model = from_nl(nl)
        t0 = time.perf_counter()
        result = model.solve(time_limit=time_limit, gap_tolerance=1e-4, max_nodes=1)
        wall = time.perf_counter() - t0
    finally:
        restore()
    return {
        "wall_s": wall,
        "nodes": int(result.node_count),
        "n_solves": len(state["solves"]),
        "total_iterations": sum(r["iterations"] for r in state["solves"] if r["iterations"] > 0),
    }


def run_instance(nl: str, time_limit: float, reps: int, rounds: int, max_cases: int) -> dict:
    from discopt.modeling.core import from_nl

    state, restore = instrument()
    try:
        model = from_nl(nl)
        # NO ``node_callback``: attaching one is not observation-neutral. It is a
        # documented routing signal (``_MIP_NLP_IGNORED_OPTIONS``, the GP probe,
        # the substitution-presolve gate all refuse to auto-route when a caller
        # asked to watch nodes), and it therefore measures a DIFFERENT ENGINE.
        # Measured on ``alan`` in fresh subprocesses, both orders, same 13 nodes
        # and same objective 2.925: without a callback the solve runs 54 POUNCE
        # NLP solves and 11 130 tape evaluations; with one it runs 1 and 0.
        t0 = time.perf_counter()
        result = model.solve(time_limit=time_limit, gap_tolerance=1e-4)
        wall = time.perf_counter() - t0

        solves = state["solves"]
        check(
            len(solves) > 0,
            f"{nl}: no POUNCE NLP solve was observed on either entry point "
            "-- probe measured nothing",
        )
        state["recording"] = False
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
            "sources": sorted({r["source"] for r in rows}),
            "n": rows[0]["n"],
            "m": rows[0]["m"],
        }

    root_only = count_root_only(nl, time_limit)

    return {
        "instance": os.path.basename(nl),
        "wall_s": wall,
        "root_only": root_only,
        "nodes": int(result.node_count),
        "status": str(result.status),
        "objective": None if result.objective is None else float(result.objective),
        "bound": None if result.bound is None else float(result.bound),
        "nlp_all": summarize(solves),
        "batch_calls": state["batch_calls"],
        "non_problem_batch_instances": state["non_problem_batches"],
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
        s = rec["nlp_all"]
        if s:
            print(
                f"[{name}] {s['n_solves']} NLP solves ({rec['nodes']} nodes, "
                f"n={s['n']} m={s['m']}): median {s['median_iterations']:.0f} iters, "
                f"{s['median_callbacks_per_solve']:.0f} callbacks/solve, "
                f"{s['median_wall_ms']:.1f} ms/solve, {s['total_wall_s']:.1f}s total "
                f"of {rec['wall_s']:.1f}s wall",
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
