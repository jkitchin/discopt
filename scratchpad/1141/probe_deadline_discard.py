"""Does the deadline arm actually (a) discard a converged stage and (b) hand
`separate_gmi` a STALE basis?  #1141's added work item.

Both are claims about `_root_cuts` internals, so both are measured from outside
the module by wrapping its own functions -- the same probe runs unchanged on the
pre-fix tree (where the guard does not exist) and on the fixed one.

Anti-vacuity (CLAUDE.md §6): the probe counts stage entries, deadline-carrying
LPs and declined LPs, and exits non-zero if the stage never ran a deadline LP --
"0 mismatches" out of 0 opportunities is not a result.
"""
import os, sys, time, json, pathlib
import numpy as np

os.environ.setdefault("DISCOPT_ROOT_CUT_DEADLINE", "1")
os.environ.setdefault("DISCOPT_CONVEX_MINLP_ROUTE", "0")

from discopt.modeling.core import from_nl                    # noqa: E402
import discopt.solvers._root_cuts as rc                      # noqa: E402

print("module under test:", rc.__file__, flush=True)
# Marker must be a single LINE of the source: the first attempt used a string
# the fix splits across two lines, so it read False on BOTH trees (§8).
MARKER = "n_le_basis"
HAS_FIX = MARKER in pathlib.Path(rc.__file__).read_text()
print("fix marker present:", HAS_FIX, flush=True)

S = {"stage": 0, "lp": 0, "deadline_lp": 0, "declined": 0,
     "gmi_calls": 0, "gmi_mismatch": 0, "gmi_extra_rows": [],
     "results": []}

_lp, _gmi, _gen = rc._solve_lp, rc.separate_gmi, rc.generate_root_cuts


def lp(root, ca, cb, time_limit=None):
    S["lp"] += 1
    if time_limit is not None:
        S["deadline_lp"] += 1
    out = _lp(root, ca, cb, time_limit)
    if out[1] is None:
        S["declined"] += 1
    return out


def gmi(root, h, x, a_all, b_all, **kw):
    """Measure the basis/row-system mismatch BEFORE delegating.

    No try/except: an instrument that swallows is an instrument that lies (§7).
    """
    S["gmi_calls"] += 1
    n_le_basis = int(h.getNumRow()) - int(root.A_eq.shape[0])
    extra = int(a_all.shape[0]) - n_le_basis
    if extra != 0:
        S["gmi_mismatch"] += 1
        S["gmi_extra_rows"].append(extra)
    return _gmi(root, h, x, a_all, b_all, **kw)


def gen(*args, **kwargs):
    S["stage"] += 1
    out = _gen(*args, **kwargs)
    S["results"].append({"cuts": len(out.cuts), "lp_bound": out.lp_bound,
                         "rounds": out.rounds_run, "stop": out.stop_reason,
                         "trace_len": len(out.bound_trace)})
    return out


rc._solve_lp, rc.separate_gmi, rc.generate_root_cuts = lp, gmi, gen
# solver.py imports the names into its own frame at call time (`from ... import`
# inside the function body), so patching the module attributes is enough --
# asserted below by `S["stage"] > 0`.

name = sys.argv[1] if len(sys.argv) > 1 else "tls2"
tl = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
p = pathlib.Path("python/tests/data/minlplib_nl") / f"{name}.nl"
m = from_nl(str(p))
t = time.perf_counter()
r = m.solve(time_limit=tl, gap_tolerance=1e-4)
wall = time.perf_counter() - t
print(f"\n{name}: status={r.status} obj={r.objective!r} bound={r.bound!r} wall={wall:.2f}s")
print("stage entries      :", S["stage"])
print("stage LPs          :", S["lp"], " with a deadline:", S["deadline_lp"],
      " declined:", S["declined"])
print("separate_gmi calls :", S["gmi_calls"], " with a STALE basis:", S["gmi_mismatch"],
      " extra rows:", sorted(set(S["gmi_extra_rows"]))[:10])
print("stage results      :", json.dumps(S["results"]))
if S["deadline_lp"] == 0:
    print("PROBE MEASURED NOTHING: no LP ever carried a deadline", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
