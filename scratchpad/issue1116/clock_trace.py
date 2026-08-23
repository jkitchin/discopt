"""#1116 localization: WHICH wall-clock-gated decision differs between reps?

Usage: python -u clock_trace.py <instance-stem> <max_nodes> <reps>

Wraps ``time.perf_counter`` with a tracer that tallies calls per (file, line)
call site. A loop whose iteration count is decided by the wall clock shows up as
a call site whose tally differs across otherwise identical repetitions -- which
localizes the divergence without changing the solve route (no callbacks, no
flags; CLAUDE.md §8). Prints per-rep progress (§10), the number of executed
comparisons (§6), and exits non-zero if it compared nothing. Exceptions are not
caught (§7).
"""

import collections
import json
import sys
import time

_orig_perf = time.perf_counter
_counts: collections.Counter = collections.Counter()
_on = [False]


def _traced() -> float:
    if _on[0]:
        f = sys._getframe(1)
        _counts[(f.f_code.co_filename.rsplit("/", 1)[-1], f.f_lineno)] += 1
    return _orig_perf()


time.perf_counter = _traced

import discopt  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

print(f"discopt.__file__={discopt.__file__}", flush=True)

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem, max_nodes, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

per_rep = []
results = []
for rep in range(reps):
    model = from_nl(NL.format(stem))
    _counts.clear()
    _on[0] = True
    t0 = _orig_perf()
    r = model.solve(max_nodes=max_nodes)
    _on[0] = False
    wall = _orig_perf() - t0
    per_rep.append(dict(_counts))
    results.append(
        {
            "rep": rep,
            "nodes": int(r.node_count or 0),
            "bound": None if r.bound is None else float(r.bound),
            "objective": None if r.objective is None else float(r.objective),
            "status": r.status,
            "clock_calls": sum(_counts.values()),
            "clock_sites": len(_counts),
            "wall": round(wall, 2),
        }
    )
    print(json.dumps(results[-1]), flush=True)

sites = sorted({s for c in per_rep for s in c})
comparisons = 0
differing = []
for s in sites:
    vals = [c.get(s, 0) for c in per_rep]
    comparisons += len(vals) - 1
    if len(set(vals)) > 1:
        differing.append((s, vals))
print(f"--- {len(differing)} differing call sites of {len(sites)} ---", flush=True)
for s, vals in sorted(differing, key=lambda kv: -max(kv[1])):
    print(f"{s[0]}:{s[1]:<6d} {vals}", flush=True)
print(f"comparisons={comparisons}", flush=True)
if comparisons == 0:
    sys.exit(2)
