"""One instance, one subprocess. Modes:
  screen <nl>         -- replay the #827 seed's candidate ladder, no solve
  solve <nl> <flag> <tl> -- default solve with DISCOPT_TRIVIAL_PRIMAL=<flag>
Prints one line ``RESULT {json}``. Never swallows an exception (CLAUDE.md 7).
"""

import json
import os
import sys
import time

mode, nl = sys.argv[1], sys.argv[2]
if mode == "solve":
    os.environ["DISCOPT_TRIVIAL_PRIMAL"] = sys.argv[3]

import discopt  # noqa: E402
import discopt.solver as S  # noqa: E402
import numpy as np  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

assert os.environ["TP_EXPECT_ROOT"] in discopt.__file__, discopt.__file__
assert hasattr(S, "_trivial_primal_enabled"), "flag absent: wrong tree loaded"

name = os.path.basename(nl)[:-3]
t0 = time.time()
model = from_nl(nl)
if mode == "screen":
    from discopt._relax.primal_heuristics import _check_constraint_feasibility as cc

    ev = S._make_evaluator(model)
    lb, ub = (np.asarray(b, dtype=np.float64) for b in ev.variable_bounds)
    lo, hi = S._clip_start_box(lb, ub)
    hits, checks = [], 0
    for label, x in (
        ("origin", np.clip(np.zeros_like(lb), lb, ub)),
        ("center", 0.5 * (lo + hi)),
        ("lower", lo),
        ("upper", hi),
    ):
        x = np.asarray(x, dtype=np.float64)
        checks += 1
        if np.all(np.isfinite(x)) and cc(ev, x, tol=1e-6):
            o = float(ev.evaluate_objective(x))
            if np.isfinite(o):
                hits.append((label, o))
    print("RESULT " + json.dumps({"name": name, "checks": checks, "hits": hits,
                                   "t": time.time() - t0}), flush=True)
else:
    from discopt.validation.feasibility import verify_point

    r = model.solve(time_limit=float(sys.argv[4]), gap_tolerance=1e-4)
    ok = None
    if r.x is not None and r.objective is not None:
        fresh = from_nl(nl)
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in fresh._variables]
        )
        ok = bool(verify_point(fresh, flat).ok)
    print("RESULT " + json.dumps({
        "name": name, "flag": sys.argv[3], "status": str(r.status),
        "obj": r.objective, "bound": r.bound, "cert": bool(getattr(r, "gap_certified", False)),
        "verified": ok, "t": time.time() - t0}), flush=True)
