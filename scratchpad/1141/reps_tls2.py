"""Reps on the instance that failed bar 1: `tls2`, both arms, interleaved.

The #1141 panel found DISCOPT_ROOT_CUT_DEADLINE regressing tls2's certificate
6/6. This re-runs that comparison after the discard fix. Interleaved OFF/ON per
rep (never arm-blocked), load gate printed before and after, spread reported --
CLAUDE.md §9.

Prints an executed-rep count and exits non-zero if it is zero.
"""
import os, sys, time, json, statistics, pathlib
import numpy as np

os.environ.setdefault("DISCOPT_CONVEX_MINLP_ROUTE", "0")
from discopt.modeling.core import from_nl                    # noqa: E402
import discopt.solvers._root_cuts as rc                      # noqa: E402

assert "n_le_basis" in pathlib.Path(rc.__file__).read_text(), "fix not loaded"
print("module:", rc.__file__, "(fix marker present)", flush=True)

name = sys.argv[1] if len(sys.argv) > 1 else "tls2"
tl = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
p = pathlib.Path("python/tests/data/minlplib_nl") / f"{name}.nl"

print(f"load before {os.getloadavg()}", flush=True)
out = {"off": [], "on": []}
n = 0
for rep in range(reps):
    for arm in ("off", "on"):
        os.environ["DISCOPT_ROOT_CUT_DEADLINE"] = "1" if arm == "on" else "0"
        m = from_nl(str(p))
        t = time.perf_counter()
        r = m.solve(time_limit=tl, gap_tolerance=1e-4)
        w = time.perf_counter() - t
        out[arm].append({"status": str(r.status), "obj": r.objective, "bound": r.bound,
                         "wall": w})
        n += 1
        print(f"rep{rep} {arm:3s} {str(r.status):12s} obj={r.objective!r:>22} "
              f"bound={r.bound!r:>22} wall={w:.2f}", flush=True)
print(f"load after {os.getloadavg()}")
for arm in ("off", "on"):
    ws = [d["wall"] for d in out[arm]]
    st = [d["status"] for d in out[arm]]
    ob = [d["obj"] for d in out[arm]]
    sd = statistics.stdev(ws) if len(ws) > 1 else 0.0
    print(f"{arm:3s}: wall {statistics.mean(ws):.2f} ± {sd:.2f}  statuses={st}  objs={ob}")
print(f"EXECUTED REPS: {n}")
pathlib.Path(f"scratchpad/1141/reps_{name}_{int(tl)}.json").write_text(json.dumps(out, indent=1))
sys.exit(0 if n else 1)
