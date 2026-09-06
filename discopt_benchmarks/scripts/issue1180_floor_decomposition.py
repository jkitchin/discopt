#!/usr/bin/env python
"""#1180 deliverable 2 -- the fresh-subprocess per-process floor, re-measured post-tape.

``baron-gap-plan.md`` §1.1 reports, on a trivial instance (alan):

    import jax 299 | import pounce 148 | import discopt 66 | parse 2 | solve 80 | total 595 ms

The 299 ms row no longer happens on the default path (``DISCOPT_NLP_EVAL=tape``,
default-ON since ``a2fb90d2``), and the issue is explicit that the remaining rows
must be **re-measured**, not back-subtracted: they were measured alongside an
import that no longer runs, so the arithmetic would carry that error forward.

Each measurement is a genuinely fresh ``python`` process (no module cache, no
warm allocator). The parent times the whole process with ``subprocess.run`` so
interpreter startup is inside the total; the child reports its own phase splits
and whether ``jax`` ever entered ``sys.modules``.

**Control arm (this is what makes the default arm non-vacuous).** The same
decomposition is run with ``DISCOPT_NLP_EVAL=jax``. That arm MUST show jax in
``sys.modules`` and a non-zero jax-import cost; if it does not, the probe is
blind to imports and its "no jax" finding on the default arm proves nothing.
The probe asserts exactly that and exits non-zero when it fails.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ASSERTS = {"n": 0}


def check(cond: bool, msg: str) -> None:
    ASSERTS["n"] += 1
    if not cond:
        raise AssertionError(msg)


CHILD = r"""
import json, sys, time
t_start = time.perf_counter()
t0 = time.perf_counter(); import discopt; t_discopt = time.perf_counter() - t0
t0 = time.perf_counter(); import pounce; t_pounce = time.perf_counter() - t0
t0 = time.perf_counter()
from discopt.modeling.core import from_nl
m = from_nl(sys.argv[1])
t_parse = time.perf_counter() - t0
t0 = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[2]), gap_tolerance=1e-4)
t_solve = time.perf_counter() - t0
t_jax = None
if "jax" in sys.modules:
    t_jax = 0.0   # already paid inside another phase; located by the phase splits
print("PHASES " + json.dumps({
    "import_discopt_s": t_discopt,
    "import_pounce_s": t_pounce,
    "parse_s": t_parse,
    "solve_s": t_solve,
    "child_total_s": time.perf_counter() - t_start,
    "nodes": int(r.node_count),
    "status": str(r.status),
    "objective": None if r.objective is None else float(r.objective),
    "jax_imported": "jax" in sys.modules,
    "n_jax_modules": sum(1 for k in sys.modules if k == "jax" or k.startswith("jax.")),
}))
"""

# The jax arm needs the import located rather than buried in `solve`, so it runs
# a variant that times `import jax` explicitly right after the evaluator picks it.
CHILD_JAX_IMPORT = r"""
import json, time
t0 = time.perf_counter()
import jax  # noqa: F401
print("JAXIMPORT " + json.dumps({"s": time.perf_counter() - t0}))
"""


def run_child(code: str, argv: list[str], env_extra: dict, python: str) -> tuple[float, dict]:
    env = dict(os.environ)
    env.update(env_extra)
    env.setdefault("PYTHONWARNINGS", "ignore")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [python, "-c", code, *argv], capture_output=True, text=True, env=env, timeout=600
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        # Never swallowed (CLAUDE.md §7): a child that died is a broken probe.
        raise RuntimeError(f"child failed rc={proc.returncode}\nSTDERR:\n{proc.stderr[-4000:]}")
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("PHASES ") or line.startswith("JAXIMPORT "):
            payload = json.loads(line.split(" ", 1)[1])
    check(payload is not None, "child produced no phase record")
    return wall, payload


def median_of(runs: list[dict], key: str) -> float:
    return statistics.median(r[key] for r in runs)


def arm(name: str, nl: str, time_limit: float, env_extra: dict, reps: int, python: str) -> dict:
    runs = []
    for i in range(reps):
        wall, rec = run_child(CHILD, [nl, str(time_limit)], env_extra, python)
        rec["process_wall_s"] = wall
        runs.append(rec)
        print(
            f"  [{name}] rep {i + 1}/{reps}: total {wall * 1000:.0f} ms "
            f"(discopt {rec['import_discopt_s'] * 1000:.0f}, pounce "
            f"{rec['import_pounce_s'] * 1000:.0f}, parse {rec['parse_s'] * 1000:.0f}, "
            f"solve {rec['solve_s'] * 1000:.0f}), jax={rec['jax_imported']}",
            flush=True,
        )
    med = {
        k: median_of(runs, k)
        for k in (
            "process_wall_s",
            "import_discopt_s",
            "import_pounce_s",
            "parse_s",
            "solve_s",
            "child_total_s",
        )
    }
    med["startup_and_teardown_s"] = med["process_wall_s"] - med["child_total_s"]
    return {
        "arm": name,
        "reps": reps,
        "median": med,
        "spread_total_ms": (
            statistics.stdev([r["process_wall_s"] for r in runs]) * 1000 if reps > 1 else None
        ),
        "nodes": runs[0]["nodes"],
        "status": runs[0]["status"],
        "objective": runs[0]["objective"],
        "jax_imported": runs[0]["jax_imported"],
        "n_jax_modules": runs[0]["n_jax_modules"],
        "runs": runs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", default="python/tests/data/minlplib_nl/alan.nl")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    check(os.path.exists(args.instance), f"missing instance {args.instance}")

    # Bare-interpreter baseline: what the process costs before discopt exists.
    bare = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        subprocess.run([args.python, "-c", "pass"], capture_output=True, timeout=120)
        bare.append(time.perf_counter() - t0)
    bare_med = statistics.median(bare)
    print(f"bare interpreter: {bare_med * 1000:.0f} ms (median of {args.reps})", flush=True)

    print("default arm (tape evaluator, the shipped path):", flush=True)
    default = arm("tape", args.instance, args.time_limit, {}, args.reps, args.python)
    check(not default["jax_imported"], "default arm imported jax -- the tape default is not live")
    check(default["n_jax_modules"] == 0, "default arm loaded jax submodules")

    print("control arm (DISCOPT_NLP_EVAL=jax) -- proves the probe can SEE an import:", flush=True)
    control = arm(
        "jax", args.instance, args.time_limit, {"DISCOPT_NLP_EVAL": "jax"}, args.reps, args.python
    )
    check(
        control["jax_imported"],
        "control arm did NOT import jax: the probe cannot see imports, so the default "
        "arm's 'no jax' result is vacuous",
    )
    check(control["n_jax_modules"] > 10, "control arm loaded suspiciously few jax modules")

    _, jaximp = run_child(CHILD_JAX_IMPORT, [], {}, args.python)
    print(f"standalone `import jax`: {jaximp['s'] * 1000:.0f} ms", flush=True)

    out = {
        "probe": "issue1180_floor_decomposition",
        "instance": args.instance,
        "bare_interpreter_ms": bare_med * 1000,
        "import_jax_standalone_ms": jaximp["s"] * 1000,
        "default_arm": default,
        "control_arm_jax": control,
        "executed_assertions": ASSERTS["n"],
    }
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(out, indent=1))

    m = default["median"]
    print(f"\n--- default (tape) floor, median of {args.reps} ---")
    print(f"  interpreter startup+teardown : {m['startup_and_teardown_s'] * 1000:7.0f} ms")
    print(f"  import discopt               : {m['import_discopt_s'] * 1000:7.0f} ms")
    print(f"  import pounce                : {m['import_pounce_s'] * 1000:7.0f} ms")
    print(f"  parse .nl                    : {m['parse_s'] * 1000:7.0f} ms")
    print(f"  solve ({default['nodes']} nodes){'':13s}: {m['solve_s'] * 1000:7.0f} ms")
    sd = default["spread_total_ms"]
    sd_txt = "n/a (1 rep)" if sd is None else f"{sd:.0f} ms"
    print(f"  TOTAL process wall           : {m['process_wall_s'] * 1000:7.0f} ms  (sd {sd_txt})")
    cm = control["median"]
    print(f"  [control jax arm total       : {cm['process_wall_s'] * 1000:7.0f} ms]")
    print(f"\nexecuted assertions: {ASSERTS['n']}")
    if ASSERTS["n"] == 0:
        print("PROBE MEASURED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
