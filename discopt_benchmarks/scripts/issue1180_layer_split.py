#!/usr/bin/env python
"""#1180 deliverable 1 -- the per-node layer split at current ``main``, post-tape.

Re-runs the ``baron-gap-plan.md`` §1.3 measurement. That table
(``python 82.5 % / jax 12.3 % / rust 3.4 % / pounce-native 0.1 %``) measured the
JAX evaluator, which no longer runs: ``DISCOPT_NLP_EVAL=tape`` is default-ON since
``a2fb90d2`` and a default solve imports zero ``jax`` modules.

Method (§8 item 5 -- never a single-layer profiler label):

* **Arm A, clean run.** No profiler. Wall, node count, and the FFI-boundary
  split that :mod:`discopt._timing` accumulates (``rust`` / ``python`` partition
  the wall; ``pounce`` ⊆ ``rust``, ``jax`` ⊆ ``python``). Undistorted totals.
* **Arm B, cProfile run.** Same solve under cProfile; every profile entry is
  classified by the *layer its code lives in* and aggregated by **self** time
  (``tottime``), so a native frame that calls back into Python is not credited
  with the callback. This is the binding-boundary aggregation §1.3 used.
  cProfile inflates Python frame cost, so arm B is read for *shares* and arm A
  for totals -- the two are reported side by side rather than blended.

Arm B additionally carves out the **evaluator callback path** (the
``_IpoptCallbacks`` wrapper, ``_timing.charge``, ``_BoundOverrideEvaluator``
attribute forwarding, and the tape evaluator's own ``_x`` marshaling) as its own
bucket, because #1180 deliverable 3 asks specifically what that path costs.

Measurement discipline (CLAUDE.md §6-§11): the probe prints an executed-assertion
count and **exits non-zero when it is zero**; it asserts ``discopt.__file__`` and
a version-unique marker before trusting anything; it never swallows an exception;
and it prints per-instance progress as it goes.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
import time

# Executed-assertion counter (CLAUDE.md §6). Every real check bumps it.
ASSERTS = {"n": 0}


def check(cond: bool, msg: str) -> None:
    ASSERTS["n"] += 1
    if not cond:
        raise AssertionError(msg)


# --- layer classification --------------------------------------------------
#
# cProfile keys are ``(filename, lineno, funcname)``; C functions use the
# sentinel filename ``"~"`` and carry the module in the funcname, e.g.
# ``<built-in method discopt._rust.solve_lp_warm_csc_py>`` or
# ``<method 'solve' of 'pounce._pounce.Problem' objects>``.

LAYERS = (
    "pounce_native",
    "rust_discopt",
    "jax",
    "callback_path",
    "python_discopt",
    "python_numpy_scipy",
    "python_other",
)

# Frames that exist only to carry one evaluator callback from POUNCE's native
# solve loop back into the tape. Matched on (file-tail, funcname).
_CALLBACK_FRAMES = {
    ("nlp_ipopt.py", "wrapper"),
    ("_timing.py", "charge"),
    ("_timing.py", "_totals"),
    ("_timing.py", "_stack"),
    ("solver.py", "__getattr__"),
    ("_tape_nlp_evaluator.py", "_x"),
    ("_tape_nlp_evaluator.py", "_ensure_fresh"),
    ("_tape_nlp_evaluator.py", "evaluate_objective"),
    ("_tape_nlp_evaluator.py", "evaluate_gradient"),
    ("_tape_nlp_evaluator.py", "evaluate_constraints"),
    ("_tape_nlp_evaluator.py", "evaluate_jacobian_values"),
    ("_tape_nlp_evaluator.py", "evaluate_hessian_values"),
    ("nlp_ipopt.py", "objective"),
    ("nlp_ipopt.py", "gradient"),
    ("nlp_ipopt.py", "constraints"),
    ("nlp_ipopt.py", "jacobian"),
    ("nlp_ipopt.py", "hessian"),
    ("contextlib.py", "__enter__"),
    ("contextlib.py", "__exit__"),
    ("contextlib.py", "helper"),
}


def classify(filename: str, funcname: str) -> str:
    if filename == "~":
        n = funcname
        # pounce >= 0.11 rebinds ``Problem.solve`` to a Python warm-start shim
        # (``pounce/_warm_start.py``) that calls the ORIGINAL native method; the
        # profiler records that inner call under the bare repr of the saved
        # function object, with no module path in the key. Verified from its
        # caller edge (``_solve_with_warm_start``) before adding this rule --
        # left unclassified it lands in ``python_other`` and silently invents a
        # 27 % "Python" share that is really the native IPM (§8 item 5).
        if "Problem.solve" in n:
            return "pounce_native"
        if "pounce" in n:
            return "pounce_native"
        if "discopt._rust" in n or "discopt._core" in n:
            return "rust_discopt"
        if "jax" in n:
            return "jax"
        return "python_other"
    f = filename.replace("\\", "/")
    tail = f.rsplit("/", 1)[-1]
    if "/jax/" in f or "/jaxlib/" in f:
        return "jax"
    if (tail, funcname) in _CALLBACK_FRAMES:
        return "callback_path"
    if "/pounce/" in f:
        return "pounce_native"
    if "/discopt/" in f:
        return "python_discopt"
    if "/numpy/" in f or "/scipy/" in f:
        return "python_numpy_scipy"
    return "python_other"


# Named components, read from the profile by (file-tail, funcname). ``cum`` is
# cumulative time (an UPPER bound for a component that can nest inside itself or
# another listed component -- read them as "this seam accounts for at most X",
# never summed); ``self`` is self time and IS additive across components.
COMPONENTS = {
    "obbt_probe_lp": ("obbt.py", "_solve_probe"),
    "obbt_root": ("obbt.py", "obbt_tighten_root"),
    "obbt_node": ("obbt.py", "obbt_tighten_node"),
    "node_nlp_solve": ("nlp_pounce.py", "solve_nlp"),
    "lp_relaxer_solve": ("milp_relaxation.py", "solve"),
    "build_milp_relaxation": ("milp_relaxation.py", "build_milp_relaxation"),
    "mccormick_node_solve": ("mccormick_lp.py", "_solve_at_node_impl"),
    "primal_heuristics": ("primal_heuristics.py", "run_primal_heuristics"),
}

# Native primitives, matched on the C-entry funcname.
NATIVE = {
    "rust_lp_warm": "discopt._rust.solve_lp_warm_csc_py",
    "rust_milp": "discopt._rust.solve_milp_csc_py",
    "pounce_ipm": "Problem.solve",
    "tape_objective": "'objective' of 'pounce.NlProblem'",
    "tape_gradient": "'gradient' of 'pounce.NlProblem'",
    "tape_constraints": "'constraints' of 'pounce.NlProblem'",
    "tape_jacobian": "'jacobian' of 'pounce.NlProblem'",
    "tape_hessian": "'hessian' of 'pounce.NlProblem'",
}


def extract_components(st: pstats.Stats) -> dict:
    """Per-seam ncalls/self/cum. Counts what it matched so a typo cannot pass."""
    comp: dict = {}
    matched = 0
    for (fname, _lineno, func), (_cc, nc, tt, ct, _callers) in st.stats.items():
        tail = fname if fname == "~" else fname.replace("\\", "/").rsplit("/", 1)[-1]
        for label, (want_tail, want_func) in COMPONENTS.items():
            if tail == want_tail and func == want_func:
                b = comp.setdefault(label, {"ncalls": 0, "self_s": 0.0, "cum_s": 0.0})
                b["ncalls"] += nc
                b["self_s"] += tt
                b["cum_s"] = max(b["cum_s"], ct)
                matched += 1
        if fname == "~":
            for label, needle in NATIVE.items():
                if needle in func:
                    b = comp.setdefault(label, {"ncalls": 0, "self_s": 0.0, "cum_s": 0.0})
                    b["ncalls"] += nc
                    b["self_s"] += tt
                    b["cum_s"] += ct
                    matched += 1
    comp["_matched_entries"] = matched
    return comp


def split_profile(prof: cProfile.Profile) -> dict:
    st = pstats.Stats(prof)
    per_layer = dict.fromkeys(LAYERS, 0.0)
    entries = 0
    hot: list[tuple[float, str, int]] = []
    for (fname, _lineno, func), (_cc, nc, tt, _ct, _callers) in st.stats.items():
        layer = classify(fname, func)
        per_layer[layer] += tt
        entries += 1
        short = fname if fname == "~" else fname.replace("\\", "/").rsplit("/", 1)[-1]
        hot.append((tt, f"{layer}|{short}:{func}", nc))
    total = sum(per_layer.values())
    check(entries > 0, "cProfile produced no entries -- the profiler never fired")
    check(total > 0.0, "cProfile total self time is zero -- nothing was measured")
    hot.sort(reverse=True)
    return {
        "entries": entries,
        "components": extract_components(st),
        "total_self_s": total,
        "by_layer_s": per_layer,
        "by_layer_pct": {k: 100.0 * v / total for k, v in per_layer.items()},
        "hot20": [{"self_s": round(t, 4), "what": w, "ncalls": n} for t, w, n in hot[:20]],
    }


def clean_run(nl: str, time_limit: float) -> dict:
    from discopt.modeling.core import from_nl

    model = from_nl(nl)
    t0 = time.perf_counter()
    result = model.solve(time_limit=time_limit, gap_tolerance=1e-4)
    wall = time.perf_counter() - t0
    check(result.node_count >= 0, "solve returned no node count")
    return {
        "wall_s": wall,
        "nodes": int(result.node_count),
        "nodes_per_s": result.node_count / wall if wall > 0 else 0.0,
        "status": str(result.status),
        "objective": None if result.objective is None else float(result.objective),
        "bound": None if result.bound is None else float(result.bound),
        "root_time_s": None if result.root_time is None else float(result.root_time),
        "ffi_rust_s": float(result.rust_time or 0.0),
        "ffi_python_s": float(result.python_time or 0.0),
        "ffi_jax_s": float(result.jax_time or 0.0),
        "jax_imported": "jax" in sys.modules,
    }


def profiled_run(
    nl: str, time_limit: float, pstats_out: str | None, max_nodes: int | None = None
) -> dict:
    from discopt.modeling.core import from_nl

    model = from_nl(nl)
    kw: dict = {"time_limit": time_limit, "gap_tolerance": 1e-4}
    if max_nodes is not None:
        kw["max_nodes"] = max_nodes
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    result = model.solve(**kw)
    prof.disable()
    wall = time.perf_counter() - t0
    if pstats_out:
        prof.dump_stats(pstats_out)
    rec = split_profile(prof)
    rec["wall_s"] = wall
    rec["nodes"] = int(result.node_count)
    rec["jax_imported"] = "jax" in sys.modules
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", default="nvs05")
    ap.add_argument("--nl-dir", default="python/tests/data/minlplib_nl")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pstats-dir", default=None)
    ap.add_argument(
        "--root-arm",
        action="store_true",
        help="also profile a max_nodes=1 (root-only) solve; the difference isolates the tree",
    )
    args = ap.parse_args()

    import discopt

    # §8 item: verify which code was actually loaded before trusting a number.
    check(
        os.path.abspath(discopt.__file__).startswith(os.path.abspath("python/discopt")),
        f"discopt imported from {discopt.__file__}, not the worktree under test",
    )
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator  # version-unique marker

    check(
        TapeNLPEvaluator.timing_bucket == "rust",
        "tape evaluator marker missing: this build predates the POUNCE-tape default",
    )
    check("jax" not in sys.modules, "jax entered sys.modules merely on import discopt")

    records = []
    for name in [s.strip() for s in args.instances.split(",") if s.strip()]:
        nl = os.path.join(args.nl_dir, f"{name}.nl")
        check(os.path.exists(nl), f"missing instance file {nl}")
        print(f"[{name}] clean arm ...", flush=True)
        clean = clean_run(nl, args.time_limit)
        print(
            f"[{name}] clean: {clean['nodes']} nodes in {clean['wall_s']:.2f}s "
            f"({clean['nodes_per_s']:.2f} nodes/s), jax_imported={clean['jax_imported']}",
            flush=True,
        )
        check(not clean["jax_imported"], f"{name}: jax imported on the default solve path")
        ps = None
        if args.pstats_dir:
            os.makedirs(args.pstats_dir, exist_ok=True)
            ps = os.path.join(args.pstats_dir, f"{name}.pstats")
        print(f"[{name}] cProfile arm ...", flush=True)
        prof = profiled_run(nl, args.time_limit, ps)
        pct = prof["by_layer_pct"]
        print(
            f"[{name}] layers: "
            + "  ".join(f"{k}={pct[k]:.1f}%" for k in LAYERS if pct[k] >= 0.05),
            flush=True,
        )
        rec = {"instance": name, "clean": clean, "cprofile": prof}
        if args.root_arm:
            print(f"[{name}] root-only arm ...", flush=True)
            ps_r = os.path.join(args.pstats_dir, f"{name}.root.pstats") if args.pstats_dir else None
            root = profiled_run(nl, args.time_limit, ps_r, max_nodes=1)
            print(f"[{name}] root-only: {root['wall_s']:.2f}s, {root['nodes']} nodes", flush=True)
            rec["cprofile_root_only"] = root
        records.append(rec)

    out = {
        "probe": "issue1180_layer_split",
        "discopt_file": discopt.__file__,
        "time_limit_s": args.time_limit,
        "python": sys.version.split()[0],
        "records": records,
        "executed_assertions": ASSERTS["n"],
    }
    text = json.dumps(out, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(f"\nexecuted assertions: {ASSERTS['n']}")
    if ASSERTS["n"] == 0:
        print("PROBE MEASURED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
