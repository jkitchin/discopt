"""#860 entry experiment: is an LP-per-node McCormick relaxation usable on MIXED
(continuous+integer) and MAXIMIZE models at all?

The issue names ``gastrans040`` / ``rsyn0805m04hfsg`` as gate probes; neither is
vendored in this repo (the full MINLPLib snapshot lives outside it and the
container has no network route to minlplib.org), so the experiment runs over the
REAL in-repo corpus instead -- which is dominated by exactly this class: 54 of the
66 ``minlplib_nl`` instances are mixed, and 4 are mixed + MAXIMIZE (``bchoco06/07/08``,
``syn05hfsg`` -- the same ``syn``/``rsyn`` family as the issue's maximize probe).

For each instance it measures, at the ROOT box only (no engine work, no tree):

  bound      the McCormick LP optimum over the mixed box (minimize-equivalent);
             finite => a usable dual bound exists for this class.
  round      is the point (integers rounded, continuous at their LP values)
             feasible for the TRUE nonlinear constraints?
  complete   is the point (integers rounded and FIXED, continuous re-solved by the
             node LP) feasible for the TRUE nonlinear constraints? -- the natural
             mixed-integer generalization of the pure-integer engine's
             "collapse the box and the relaxation is exact" primal.

Kill criterion (issue #860): if NO mixed instance yields a verified feasible point
at any budget, LP-per-node is not the lever for this class and the engine should
not be widened.
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "..", "python", "tests", "data")


def _load(path):
    from discopt.modeling.core import from_nl

    return from_nl(path)


def probe(path, time_budget=60.0):
    from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt._relax.term_classifier import classify_nonlinear_terms
    from discopt.modeling.core import ObjectiveSense, VarType
    from discopt.solver import _check_constraint_feasibility, _infer_constraint_bounds

    m = _load(path)
    n = len(m._variables)
    is_int = np.array(
        [v.var_type in (VarType.INTEGER, VarType.BINARY) for v in m._variables], dtype=bool
    )
    sense = (
        "MAX"
        if (m._objective is not None and m._objective.sense == ObjectiveSense.MAXIMIZE)
        else "MIN"
    )
    lb = np.array([float(v.lb) for v in m._variables])
    ub = np.array([float(v.ub) for v in m._variables])
    out = {
        "name": os.path.basename(path)[:-3],
        "n": n,
        "nint": int(is_int.sum()),
        "sense": sense,
        "finite_box": bool(np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))),
    }
    if not out["finite_box"]:
        out["note"] = "infinite root box"
        return out

    ev = NLPEvaluator(m)
    cl, cu = _infer_constraint_bounds(m, ev)

    def feasible(x):
        return bool(_check_constraint_feasibility(ev, x, cl, cu, tol=1e-6))

    terms = classify_nonlinear_terms(m)
    t0 = time.perf_counter()
    inc = IncrementalMcCormickLP(m, terms, deadline=t0 + time_budget)
    out["inc_ok"] = bool(inc.ok)
    if not inc.ok:
        out["note"] = "no incremental structure"
        return out
    b, x, _ = inc.solve(lb, ub)
    out["root_lp_s"] = round(time.perf_counter() - t0, 3)
    if b is None:
        out["note"] = "root LP failed"
        return out
    out["bound"] = float(b)

    xr = np.array(x[:n], dtype=float)
    xr[is_int] = np.round(xr[is_int])
    xr = np.minimum(np.maximum(xr, lb), ub)
    out["round"] = feasible(xr)

    lo, hi = lb.copy(), ub.copy()
    lo[is_int] = xr[is_int]
    hi[is_int] = xr[is_int]
    b2, x2, _ = inc.solve(lo, hi)
    if x2 is not None:
        xc = np.array(x2[:n], dtype=float)
        xc[is_int] = xr[is_int]
        xc = np.minimum(np.maximum(xc, lb), ub)
        out["complete"] = feasible(xc)
        if out["complete"]:
            try:
                out["obj"] = float(ev.evaluate_objective(xc))
            except Exception:
                pass
    else:
        out["complete"] = False
    out["total_s"] = round(time.perf_counter() - t0, 3)
    return out


def main():
    only = sys.argv[1:] or None
    paths = sorted(glob.glob(os.path.join(_DATA, "minlplib_nl", "*.nl"))) + sorted(
        glob.glob(os.path.join(_DATA, "minlplib", "*.nl"))
    )
    seen = set()
    for p in paths:
        name = os.path.basename(p)[:-3]
        if name in seen:
            continue
        seen.add(name)
        if only and name not in only:
            continue
        try:
            r = probe(p)
        except Exception as exc:  # keep the sweep going; report the failure
            r = {"name": name, "note": f"{type(exc).__name__}: {exc}"[:80]}
        print(r, flush=True)


if __name__ == "__main__":
    main()
