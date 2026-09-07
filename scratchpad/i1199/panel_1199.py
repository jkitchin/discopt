"""Panel for #1199: does the reported incumbent pass the repo's ACCEPTANCE arbiter?

For every in-repo MINLPLib instance: solve, then re-verify the reported point
against a freshly parsed ORIGINAL model with
``primal_heuristics._check_constraint_feasibility``'s own combined test, and
report the normalized ratio  max_i viol_i / (tol + rtol*scale_i)  (<= 1 passes).

Prints per-instance progress (CLAUDE.md §10) and an executed-comparison count,
exiting non-zero if nothing was actually compared (§6).
"""

import glob
import json
import os
import sys
import time

import numpy as np
from discopt._relax.nlp_evaluator import cached_evaluator
from discopt.modeling.core import from_nl
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

MARKER_EXPECTED = os.environ.get("PANEL_MARKER")  # §8: assert the code under test
if MARKER_EXPECTED is not None:
    import discopt.solver as _S

    src = open(_S.__file__).read()
    present = MARKER_EXPECTED in src
    want = os.environ.get("PANEL_MARKER_PRESENT", "1") == "1"
    print(f"# {_S.__file__}  marker={MARKER_EXPECTED!r} present={present} want={want}", flush=True)
    assert present is want, "wrong code loaded"

TL = float(os.environ.get("PANEL_TL", "20"))
MAXN = int(os.environ.get("PANEL_NODES", "5000"))
out_path = sys.argv[1]

files = sorted(glob.glob("python/tests/data/minlplib_nl/*.nl"))
rows = []
compared = 0
for k, f in enumerate(files, 1):
    name = os.path.basename(f)[:-3]
    rec = {"name": name}
    try:
        m = from_nl(f)
        t0 = time.perf_counter()
        r = m.solve(max_nodes=MAXN, time_limit=TL)
        rec["wall"] = round(time.perf_counter() - t0, 3)
        rec["status"] = r.status
        rec["obj"] = None if r.objective is None else float(r.objective)
        rec["bound"] = None if getattr(r, "bound", None) is None else float(r.bound)
        rec["nodes"] = getattr(r, "node_count", None)
        rec["cert"] = bool(getattr(r, "gap_certified", False))
        if r.x:
            om = from_nl(f)
            ev = cached_evaluator(om)
            if ev.n_constraints:
                cl, cu = (np.asarray(b, float) for b in _infer_constraint_bounds(ev))
                x = np.concatenate(
                    [np.atleast_1d(np.asarray(r.x[v.name], float)).ravel() for v in om._variables]
                )
                g = np.asarray(ev.evaluate_constraints(x))
                viol = np.maximum(np.maximum(cl - g, 0.0), np.maximum(g - cu, 0.0))
                jac = np.abs(np.asarray(ev.evaluate_jacobian(x), float))
                scale = jac @ np.abs(x)
                thr = 1e-6 + 1e-9 * scale
                ratio = float(np.max(viol / thr))
                rec["ratio"] = ratio
                rec["maxviol"] = float(np.max(viol))
                rec["accepts"] = ratio <= 1.0
                compared += 1
    except Exception as exc:  # recorded, never swallowed
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rows.append(rec)
    print(f"[{k}/{len(files)}] {name}: {json.dumps(rec)}", flush=True)

json.dump(rows, open(out_path, "w"), indent=1)
bad = [r for r in rows if r.get("accepts") is False]
print(f"\nincumbents compared: {compared}")
print(f"FAIL acceptance arbiter: {len(bad)} -> {[(r['name'], round(r['ratio'], 2)) for r in bad]}")
if compared == 0:
    print("PANEL MEASURED NOTHING")
    sys.exit(1)
