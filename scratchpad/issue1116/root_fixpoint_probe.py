"""#1116 E7: does the deadline-driven ROOT FIXPOINT return a different box per run?

``run_root_fixpoint`` (solver.py:14370, flag ``root_fixpoint`` default **ON**) is
handed ``deadline=perf_counter() + min(max(time_limit*0.10, 1.0), remaining)`` --
360 s at the default ``time_limit=3600``. Its docstring: "The loop stops the
moment it is reached", and inside, S3 OBBT gets 85% of the remaining budget and is
itself deadline-clamped. That is a role-2 clock in the #912 sense: it does not
answer "when do we stop?" (the user's ``time_limit``), it answers "how much
tightening do we do?" -- so the returned box, and therefore the whole downstream
relaxation, is a function of machine speed.

``build_component_bisect.py`` showed the first root LP has a different number of
COLUMNS between two reps in one process, and ``build_truncation_probe.py`` showed
the build itself was NOT truncated (``cons_done == cons_total``, 404/404). A
different box entering an untruncated build explains both.

This probe records what the fixpoint returned -- rounds, bounds tightened, and a
hash of the (lb, ub) box -- for every call, across reps.

Kill criterion: if the box hash is IDENTICAL across reps while the bound still
varies, the fixpoint is not the source and this direction is dead.

§6 executed-comparison count with non-zero exit; §7 no swallowed exceptions;
§8 module identity printed; §10 per-rep flush.
"""

import hashlib
import json
import sys

import discopt
import numpy as np
from discopt._relax import root_reduce
from discopt.modeling.core import from_nl

print(f"discopt.__file__={discopt.__file__}", flush=True)


def _h(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:12]


runs: list[list[dict]] = []
_cur: list[dict] = []
_real = root_reduce.run_root_fixpoint


def _traced(model, lb, ub, **kw):
    res = _real(model, lb, ub, **kw)
    rec = {
        "call": len(_cur),
        "in_box": _h(np.concatenate([np.asarray(lb, float), np.asarray(ub, float)])),
        "out_box": _h(np.concatenate([np.asarray(res.lb, float), np.asarray(res.ub, float)])),
        "n_tightened": int(res.n_tightened),
        "n_rounds": int(res.n_rounds),
        "infeasible": bool(res.infeasible),
        "stage_time": {k: round(v, 2) for k, v in dict(res.stage_time).items()},
    }
    _cur.append(rec)
    print(json.dumps({"rep": len(runs), **rec}), flush=True)
    return res


root_reduce.run_root_fixpoint = _traced

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem, max_nodes, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

bounds = []
for rep in range(reps):
    _cur = []
    r = from_nl(NL.format(stem)).solve(max_nodes=max_nodes)
    runs.append(_cur)
    bounds.append(repr(float(r.bound)) if r.bound is not None else None)
    print(
        f"rep={rep} fixpoint_calls={len(_cur)} nodes={r.node_count} bound={bounds[-1]}",
        flush=True,
    )

comparisons = 0
n_calls = min(len(r) for r in runs) if runs else 0
for k in range(n_calls):
    recs = [r[k] for r in runs]
    for key in ("in_box", "out_box", "n_tightened", "n_rounds"):
        distinct = sorted({repr(x[key]) for x in recs})
        comparisons += len(recs) - 1
        print(
            f"CALL {k} {key:12s} {'STABLE' if len(distinct) == 1 else 'VARIES'} {distinct}",
            flush=True,
        )

print(
    f"bound across reps: {'STABLE' if len(set(bounds)) == 1 else 'VARIES'} {sorted(set(bounds))}",
    flush=True,
)
print(f"comparisons={comparisons} fixpoint_calls_per_rep={[len(r) for r in runs]}", flush=True)
if comparisons == 0:
    print("PROBE FIRED NOTHING (run_root_fixpoint was never called)", flush=True)
    sys.exit(2)
