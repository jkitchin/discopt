"""#860 entry experiment, round 2 -- under the WIDENED gate.

Round 1 (``issue860_entry_experiment.py``) measured the engine's *current* gate on
the in-repo corpus and found the binding blocker is not mixed-ness at all:

  * 30 / 71 mixed instances are rejected by ``_is_in_scope``'s all-variables-finite
    box test (real MINLPs give continuous columns a ``+inf`` bound);
  * 28 / 71 are rejected because ``IncrementalMcCormickLP`` declines (term types its
    closed-form patch does not map);
  * only 12 reach a root LP at all.

Round 2 therefore measures the gate this issue proposes:

  * accept a partially infinite root box -- the COLD ``build_milp_relaxation``
    already drops any row whose payload is non-finite (``uniform_relax._Builder.add_row``),
    so it self-guards; the INCREMENTAL patch does not, so it is used only when every
    mapped product factor has finite bounds;
  * accept any variable mix with at least one integer, and either objective sense.

Reported per instance:

  path      ``inc`` (incremental patch) or ``cold`` (per-node builder)
  bound     minimize-equivalent McCormick LP optimum at the root -- finite => usable
  round     integers rounded, continuous at their LP values: feasible for the TRUE
            nonlinear constraints?
  complete  integers rounded and FIXED, continuous re-solved by the node LP:
            feasible for the TRUE constraints? (the mixed generalization of the
            pure-integer engine's collapsed-box primal)

Kill criterion (issue #860): no verified feasible point anywhere in the mixed class
=> LP-per-node is not the lever and the engine should not be widened.
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "..", "python", "tests", "data")


def _root_lp(model, terms, lb, ub, budget):
    """(path, bound, x) at the root box, preferring the incremental patch."""
    from discopt._relax.discretization import DiscretizationState
    from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._relax.milp_relaxation import build_milp_relaxation

    t0 = time.perf_counter()
    inc = IncrementalMcCormickLP(model, terms, deadline=t0 + budget)
    if inc.ok:
        cols = set()
        for i, j in inc.bilinear:
            cols.update((int(i), int(j)))
        for i, _p in inc.monomial:
            cols.add(int(i))
        for j, _a in inc.affine_square:
            cols.add(int(j))
        idx = np.fromiter(cols, dtype=int) if cols else np.zeros(0, dtype=int)
        if idx.size == 0 or (np.all(np.isfinite(lb[idx])) and np.all(np.isfinite(ub[idx]))):
            b, x, _ = inc.solve(lb, ub)
            return "inc", b, x, inc
    relax, info = build_milp_relaxation(
        model, terms, DiscretizationState(), bound_override=(lb, ub)
    )
    if not relax._objective_bound_valid:
        return "cold", None, None, None
    relax._integrality = None
    res = relax.solve()
    if res is None or res.bound is None or res.x is None:
        return "cold", None, None, None
    return "cold", float(res.bound), np.asarray(res.x, dtype=float), None


def _resolve_fixed(model, terms, lo, hi, inc):
    if inc is not None:
        _b, x, _ = inc.solve(lo, hi)
        return x
    from discopt._relax.discretization import DiscretizationState
    from discopt._relax.milp_relaxation import build_milp_relaxation

    relax, _info = build_milp_relaxation(
        model, terms, DiscretizationState(), bound_override=(lo, hi)
    )
    relax._integrality = None
    res = relax.solve()
    return None if res is None or res.x is None else np.asarray(res.x, dtype=float)


def probe(path, budget=60.0):
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt._relax.term_classifier import classify_nonlinear_terms
    from discopt.modeling.core import ObjectiveSense, VarType, from_nl
    from discopt.solver import _check_constraint_feasibility, _infer_constraint_bounds

    m = from_nl(path)
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
        "inf_box": not bool(np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))),
    }
    ev = NLPEvaluator(m)
    cl, cu = _infer_constraint_bounds(m, ev)

    def feasible(x):
        return bool(_check_constraint_feasibility(ev, x, cl, cu, tol=1e-6))

    terms = classify_nonlinear_terms(m)
    t0 = time.perf_counter()
    kind, b, x, inc = _root_lp(m, terms, lb, ub, budget)
    out["path"] = kind
    out["root_s"] = round(time.perf_counter() - t0, 2)
    if b is None or not np.isfinite(b):
        out["note"] = "no finite root bound"
        return out
    out["bound"] = float(b)

    xr = np.array(x[:n], dtype=float)
    xr[is_int] = np.round(xr[is_int])
    xr = np.minimum(np.maximum(xr, lb), ub)
    out["round"] = feasible(xr)

    lo, hi = lb.copy(), ub.copy()
    lo[is_int] = xr[is_int]
    hi[is_int] = xr[is_int]
    x2 = _resolve_fixed(m, terms, lo, hi, inc)
    if x2 is None:
        out["complete"] = False
    else:
        xc = np.array(x2[:n], dtype=float)
        xc[is_int] = xr[is_int]
        xc = np.minimum(np.maximum(xc, lb), ub)
        out["complete"] = feasible(xc)
        if out["complete"]:
            try:
                out["obj"] = float(ev.evaluate_objective(xc))
            except Exception:
                pass
    out["total_s"] = round(time.perf_counter() - t0, 2)
    return out


def main():
    only = set(sys.argv[1:]) or None
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
        except Exception as exc:
            r = {"name": name, "note": f"{type(exc).__name__}: {exc}"[:90]}
        print(r, flush=True)


if __name__ == "__main__":
    main()
