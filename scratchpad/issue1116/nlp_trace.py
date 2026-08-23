"""#1116 localization, stage 2: is the FIRST divergence in a local NLP solve?

Usage: python -u nlp_trace.py <instance-stem> <max_nodes> <reps>

The baseline probe showed the INCUMBENT objective moves in its last digits across
repetitions, not just the dual bound, so the local NLP is a prime suspect: several
call sites clamp it with ``max_wall_time`` (e.g. ``solver.py:1227`` caps a seed
solve at 4 s), and a wall-clamped NLP returns a different iterate on a differently
loaded machine.

This probe wraps every NLP entry point (``nlp_pounce.solve_nlp`` and POUNCE's
``solve_nlp_batch``) and records, per call, a hash of the INPUTS and of the
OUTPUTS. Comparing the two repetitions' call sequences gives the first index where
they diverge and says whether the inputs at that index were already different
(divergence upstream) or identical (the NLP itself is not reproducible).

Nothing about the solve route changes (no callbacks, no flags; §8). Exceptions are
not caught (§7); the executed-comparison count is printed and a zero count exits
non-zero (§6).
"""

import hashlib
import json
import sys
import time

import discopt
import numpy as np
import pounce
from discopt.modeling.core import from_nl
from discopt.solvers import nlp_pounce

print(f"discopt.__file__={discopt.__file__}", flush=True)
print(f"pounce.__file__={pounce.__file__}", flush=True)

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem, max_nodes, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

_calls: list = []
_on = [False]


def _h(obj) -> str:
    m = hashlib.sha1()

    def feed(o):
        if isinstance(o, np.ndarray):
            m.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, (list, tuple)):
            for e in o:
                feed(e)
        elif isinstance(o, dict):
            for k in sorted(o, key=repr):
                m.update(repr(k).encode())
                feed(o[k])
        else:
            m.update(repr(o).encode())

    feed(obj)
    return m.hexdigest()[:12]


_orig_solve = nlp_pounce.solve_nlp
_orig_batch = getattr(pounce, "solve_nlp_batch", None)


def _traced_solve(evaluator, x0, constraint_bounds=None, options=None, **kw):
    if not _on[0]:
        return _orig_solve(evaluator, x0, constraint_bounds, options, **kw)
    t0 = time.perf_counter()
    r = _orig_solve(evaluator, x0, constraint_bounds, options, **kw)
    dt = time.perf_counter() - t0
    _calls.append(
        {
            "kind": "solve_nlp",
            "in": _h([np.asarray(x0, dtype=np.float64), constraint_bounds, options]),
            "wall_cap": None if not options else options.get("max_wall_time"),
            "elapsed": round(dt, 4),
            "status": getattr(r, "status", None),
            "out": _h(np.asarray(getattr(r, "x", []), dtype=np.float64)),
            "obj": None if getattr(r, "objective", None) is None else float(r.objective),
            "iters": int(getattr(r, "iterations", 0) or 0),
        }
    )
    return r


nlp_pounce.solve_nlp = _traced_solve

if _orig_batch is not None:

    def _traced_batch(*args, **kw):
        if not _on[0]:
            return _orig_batch(*args, **kw)
        t0 = time.perf_counter()
        rs = _orig_batch(*args, **kw)
        dt = time.perf_counter() - t0
        _calls.append(
            {
                "kind": "solve_nlp_batch",
                "in": _h([args, kw]),
                "wall_cap": None,
                "elapsed": round(dt, 4),
                "status": _h([getattr(r, "status", None) for r in rs]),
                "out": _h([np.asarray(getattr(r, "x", []), dtype=np.float64) for r in rs]),
                "obj": _h([getattr(r, "objective", None) for r in rs]),
                "iters": _h([getattr(r, "iterations", None) for r in rs]),
            }
        )
        return rs

    pounce.solve_nlp_batch = _traced_batch

per_rep = []
for rep in range(reps):
    model = from_nl(NL.format(stem))
    _calls.clear()
    _on[0] = True
    t0 = time.perf_counter()
    r = model.solve(max_nodes=max_nodes)
    _on[0] = False
    per_rep.append(list(_calls))
    print(
        json.dumps(
            {
                "rep": rep,
                "nodes": int(r.node_count or 0),
                "bound": None if r.bound is None else float(r.bound),
                "objective": None if r.objective is None else float(r.objective),
                "nlp_calls": len(per_rep[-1]),
                "wall_capped_calls": sum(1 for c in per_rep[-1] if c["wall_cap"] is not None),
                "wall": round(time.perf_counter() - t0, 2),
            }
        ),
        flush=True,
    )

comparisons = 0
first_div = None
a, b = per_rep[0], per_rep[1]
for i in range(min(len(a), len(b))):
    comparisons += 1
    if a[i]["in"] != b[i]["in"] or a[i]["out"] != b[i]["out"] or a[i]["obj"] != b[i]["obj"]:
        first_div = i
        break
print(f"calls rep0={len(a)} rep1={len(b)}", flush=True)
if first_div is None:
    print("NO DIVERGENCE in the NLP call sequence prefix", flush=True)
else:
    print(f"FIRST DIVERGENCE at call index {first_div}", flush=True)
    for tag, c in (("rep0", a[first_div]), ("rep1", b[first_div])):
        print(f"  {tag}: {json.dumps(c)}", flush=True)
    print(
        "  inputs identical -> the NLP itself is not reproducible"
        if a[first_div]["in"] == b[first_div]["in"]
        else "  inputs ALREADY differ -> divergence is upstream of this call",
        flush=True,
    )
    for j in range(max(0, first_div - 3), first_div):
        print(f"  prior[{j}] rep0={json.dumps(a[j])}", flush=True)
        print(f"  prior[{j}] rep1={json.dumps(b[j])}", flush=True)
print(f"comparisons={comparisons}", flush=True)
if comparisons == 0:
    sys.exit(2)
