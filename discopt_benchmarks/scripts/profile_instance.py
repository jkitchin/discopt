#!/usr/bin/env python
"""Measurement-only per-phase profiler for a single MINLPLib .nl instance.

Produces a JSON record attributing wall clock to solver phases:
parse, root processing, per-node LP, POUNCE NLP subsolves, OBBT/FBBT,
cut separation, relaxation (re)construction, lazy imports, and JAX tracing.

Modes
-----
clean     : no profiler; wall-clock phases from result fields + a bound trace
            sampled via ``node_callback`` (best_bound vs time/nodes).
cprofile  : same solve under cProfile; dumps .pstats and extracts cumulative
            times for a fixed set of attribution buckets. cProfile adds
            measurable overhead (~10-50% on JAX-heavy code); use the clean
            run's wall clock for totals and this run for *shares*.

Usage
-----
  python profile_instance.py INST.nl --time-limit 60 --mode clean --json out.json
  python profile_instance.py INST.nl --mode cprofile --pstats out.pstats --json out.json
  python profile_instance.py INST.nl --mode clean --second-solve   # JIT/setup tax
  python profile_instance.py INST.nl --mode clean --dump-stack-after 70  # deadline overrun

This script never modifies solver behavior beyond attaching the (optional)
``node_callback`` trace; ``--no-trace`` disables even that for a fully
uninstrumented run.
"""

from __future__ import annotations

import argparse
import cProfile
import faulthandler
import json
import pstats
import sys
import threading
import time


class StackSampler:
    """In-process sampling profiler (py-spy needs root on macOS; this doesn't).

    Samples the main thread's Python stack at ``hz`` from a daemon thread and
    aggregates collapsed stacks (``a;b;c count`` lines, flamegraph-compatible).
    Native (Rust/C) time is attributed to the Python frame that called it.
    Overhead is a few percent at 100-200 Hz.
    """

    def __init__(self, hz: float = 100.0):
        self.interval = 1.0 / hz
        self.counts: dict[str, int] = {}
        self.n_samples = 0
        self._stop = threading.Event()
        self._target = threading.current_thread().ident
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            frames = sys._current_frames()
            f = frames.get(self._target)
            stack = []
            while f is not None:
                code = f.f_code
                fn = code.co_filename
                # keep paths short: last two components
                parts = fn.replace("\\", "/").split("/")
                short = "/".join(parts[-2:])
                stack.append(f"{short}:{code.co_name}")
                f = f.f_back
            if stack:
                key = ";".join(reversed(stack))
                self.counts[key] = self.counts.get(key, 0) + 1
                self.n_samples += 1
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def dump(self, path: str):
        with open(path, "w") as fh:
            for k, v in sorted(self.counts.items(), key=lambda kv: -kv[1]):
                fh.write(f"{k} {v}\n")


def run(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    from discopt.modeling.core import from_nl

    t_import = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = from_nl(args.instance)
    t_parse = time.perf_counter() - t0

    trace: list[dict] = []

    def node_cb(ctx, _model):
        trace.append(
            {
                "t": ctx.elapsed_time,
                "nodes": ctx.node_count,
                "best_bound": _f(ctx.best_bound),
                "incumbent": _f(ctx.incumbent_obj),
                "gap": _f(ctx.gap),
            }
        )

    kwargs: dict = {"time_limit": args.time_limit, "gap_tolerance": 1e-4}
    if not args.no_trace:
        kwargs["node_callback"] = node_cb

    if args.dump_stack_after:
        faulthandler.dump_traceback_later(args.dump_stack_after, repeat=True, exit=False)

    prof = None
    if args.mode == "cprofile":
        prof = cProfile.Profile()
        prof.enable()
    sampler = None
    if args.sample_hz:
        sampler = StackSampler(hz=args.sample_hz)
        sampler.start()
    t0 = time.perf_counter()
    result = model.solve(**kwargs)
    t_solve = time.perf_counter() - t0
    if sampler is not None:
        sampler.stop()
        if args.sample_out:
            sampler.dump(args.sample_out)
    if prof is not None:
        prof.disable()
    if args.dump_stack_after:
        faulthandler.cancel_dump_traceback_later()

    rec: dict = {
        "instance": args.instance,
        "mode": args.mode,
        "time_limit": args.time_limit,
        "import_s": t_import,
        "parse_s": t_parse,
        "solve_s": t_solve,
        "status": str(result.status),
        "objective": _f(result.objective),
        "bound": _f(result.bound),
        "gap": _f(result.gap),
        "node_count": result.node_count,
        "root_time_s": _f(result.root_time),
        "root_bound": _f(result.root_bound),
        "root_gap": _f(result.root_gap),
        "rust_time": _f(result.rust_time),
        "jax_time": _f(result.jax_time),
        "python_time": _f(result.python_time),
        "solver_stats": result.solver_stats,
        "trace": trace if not args.no_trace else None,
    }

    if args.second_solve:
        t0 = time.perf_counter()
        r2 = model.solve(time_limit=args.time_limit, gap_tolerance=1e-4)
        rec["second_solve_s"] = time.perf_counter() - t0
        rec["second_solve_nodes"] = r2.node_count

    if prof is not None:
        if args.pstats:
            prof.dump_stats(args.pstats)
        rec["buckets"] = extract_buckets(prof)

    return rec


def _f(v):
    """JSON-safe float (None for None/inf/nan kept as-is where representable)."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return v if v == v and abs(v) != float("inf") else str(v)


# Attribution buckets: (label, match on (filename, lineno, funcname), use_cumtime).
# Builtin/C functions are keyed as ('~', 0, '<name>') by pstats. For C entries
# cumtime includes Python callbacks re-entered from Rust (pounce evaluates
# objective/jacobian via JAX callbacks), so both cum and tot are reported.
BUCKETS = [
    ("pounce_problem_solve", lambda f, n: f == "~" and "pounce" in n and "solve" in n),
    ("pounce_module_fn", lambda f, n: f == "~" and "pounce._pounce" in n and "solve" not in n),
    ("rust_lp_solve", lambda f, n: f == "~" and "discopt._rust.solve_lp" in n),
    ("rust_other", lambda f, n: f == "~" and "discopt._rust" in n and "solve_lp" not in n),
    ("build_milp_relaxation", lambda f, n: n == "build_milp_relaxation"),
    ("solve_at_node", lambda f, n: n == "solve_at_node"),
    ("lp_relaxer_solve", lambda f, n: "milp_relaxation" in f and n == "solve"),
    ("obbt_root", lambda f, n: "obbt.py" in f and n == "obbt_tighten_root"),
    ("obbt_node", lambda f, n: "obbt.py" in f and n in ("obbt_tighten_node", "obbt_tighten")),
    ("feasibility_pump", lambda f, n: n == "feasibility_pump"),
    ("primal_heuristics_mod", lambda f, n: "primal_heuristics" in f),
    ("nlp_pounce_solve_nlp", lambda f, n: "nlp_pounce" in f and n == "solve_nlp"),
    ("qp_pounce", lambda f, n: "qp_pounce" in f),
    ("jax_trace", lambda f, n: "partial_eval" in f and n == "trace_to_jaxpr"),
    ("jax_pjit_cache_miss", lambda f, n: "pjit" in f and n == "cache_miss"),
    ("lazy_imports", lambda f, n: "importlib" in f and n == "_find_and_load"),
    ("scipy_sparse", lambda f, n: "scipy/sparse" in f),
]


def extract_buckets(prof: cProfile.Profile) -> dict:
    st = pstats.Stats(prof)
    out: dict[str, dict] = {}
    for (fname, _lineno, funcname), (_cc, nc, tt, ct, _callers) in st.stats.items():
        for label, match in BUCKETS:
            try:
                hit = match(fname, funcname)
            except Exception:
                hit = False
            if hit:
                b = out.setdefault(label, {"ncalls": 0, "tottime": 0.0, "cumtime": 0.0})
                b["ncalls"] += nc
                b["tottime"] += tt
                # cumtime across multiple matched frames of one bucket can
                # double-count nesting within the bucket; still the best
                # available upper estimate per bucket.
                if label == "rust_lp_solve":
                    b["cumtime"] += ct
                else:
                    b["cumtime"] = max(b["cumtime"], ct)
    total = pstats.Stats(prof).total_tt
    out["_total_profiled_s"] = total
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--mode", choices=["clean", "cprofile"], default="clean")
    ap.add_argument("--pstats", default=None, help="dump raw pstats here (cprofile mode)")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--no-trace", action="store_true", help="do not attach node_callback")
    ap.add_argument("--second-solve", action="store_true", help="solve twice; JIT/setup tax")
    ap.add_argument("--sample-hz", type=float, default=None, help="in-process stack sampler rate")
    ap.add_argument("--sample-out", default=None, help="collapsed-stack output file")
    ap.add_argument(
        "--dump-stack-after",
        type=float,
        default=None,
        help="faulthandler stack dump every N seconds (deadline-overrun diagnosis)",
    )
    args = ap.parse_args()

    rec = run(args)
    text = json.dumps(rec, indent=1)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text)
    # Keep stdout terse: summary line only (trace can be thousands of entries).
    summary = {k: v for k, v in rec.items() if k not in ("trace", "buckets")}
    print(json.dumps(summary, indent=1))
    if "buckets" in rec:
        print(json.dumps(rec["buckets"], indent=1))
    if rec.get("trace"):
        tr = rec["trace"]
        print(f"trace: {len(tr)} nodes; first={tr[0]} last={tr[-1]}", file=sys.stderr)


if __name__ == "__main__":
    main()
