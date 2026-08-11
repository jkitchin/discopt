"""#860 §5 differential panel for the widened LP-per-node scope.

Two panels, one process per solve (isolated: the solver stashes per-solve state on
the model and a crash in one instance must not take the sweep with it).

PANEL A -- engine soundness on the widened class. Runs ``solve_lp_spatial_bb``
directly on every in-repo instance now in scope (>=1 integer variable, either sense,
any continuous mix) and checks the certificate invariants that do not need an
external oracle:

  * every reported incumbent is INDEPENDENTLY re-verified feasible (fresh evaluator,
    fresh constraint bounds) and its objective re-evaluated to match;
  * ``bound <= objective`` for a minimize, ``bound >= objective`` for a maximize;
  * the bound never crosses the best verified feasible point found by ANY run of that
    instance (a dual bound above a known feasible objective is unsound, full stop);
  * a ``status="optimal"`` run is never beaten by another run's verified incumbent
    (that would be a false optimality certificate).

PANEL B -- graduation gate for ``DISCOPT_LP_SPATIAL_MIXED`` (whether the DEFAULT
path reserves budget for the #844 no-incumbent fallback on mixed / maximize models).
Runs ``Model.solve`` with the flag off and on and requires BOTH:

  (1) cert-clean: no ``gap_certified`` True -> False, no objective drift beyond
      tolerance, no incumbent better than the panel's best verified point, no
      ``incumbent_verification_failed``;
  (2) net-positive: incumbents gained where the default path had none, without a
      broad wall-clock regression.

Usage:  python scratchpad/panel860.py [--tl SECONDS] [--panel A|B|AB] [--out FILE]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORPORA = [
    os.path.join(_ROOT, "python", "tests", "data", "minlplib_nl"),
    os.path.join(_ROOT, "python", "tests", "data", "minlplib"),
]

# --------------------------------------------------------------------------- #
# workers (run in a fresh interpreter per instance)
# --------------------------------------------------------------------------- #

_COMMON = r'''
import os, sys, json, time, warnings
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
warnings.filterwarnings("ignore")
import numpy as np
from discopt.modeling.core import from_nl, ObjectiveSense, VarType

def load(path):
    m = from_nl(path)
    sense = -1.0 if (m._objective is not None
                     and m._objective.sense == ObjectiveSense.MAXIMIZE) else 1.0
    return m, sense

def verify_point(model, x_flat):
    """Independent re-verification: fresh evaluator, fresh constraint bounds.
    Returns (feasible, true_objective) -- the objective in the MODEL's own sense."""
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt.solver import _check_constraint_feasibility, _infer_constraint_bounds
    ev = NLPEvaluator(model)
    cl, cu = _infer_constraint_bounds(model, ev)
    ok = bool(_check_constraint_feasibility(ev, np.asarray(x_flat, float), cl, cu, tol=1e-5))
    # NLPEvaluator returns the MINIMIZE-EQUIVALENT objective; undo that for reporting.
    sgn = -1.0 if (model._objective is not None
                   and model._objective.sense == ObjectiveSense.MAXIMIZE) else 1.0
    obj = sgn * float(ev.evaluate_objective(np.asarray(x_flat, float)))
    return ok, obj
'''

_WORKER_ENGINE = (
    _COMMON
    + r"""
tl = float(sys.argv[1])
for path in sys.argv[2:]:
    out = {"path": path}
    try:
        m, sgn = load(path)
        from discopt._relax.lp_spatial_bb import _is_in_scope, solve_lp_spatial_bb
        out["in_scope"] = bool(_is_in_scope(m))
        out["in_scope_legacy"] = bool(_is_in_scope(m, mixed=False))
        if out["in_scope"]:
            t0 = time.perf_counter()
            r = solve_lp_spatial_bb(m, time_limit=tl, gap_tolerance=1e-4)
            out["wall"] = time.perf_counter() - t0
            out["declined"] = r is None
            if r is not None:
                out.update(status=r.status, obj=r.objective, bound=r.bound,
                           gap=r.gap, nodes=r.node_count)
                if r.x is not None:
                    ok, trueobj = verify_point(from_nl(path), np.asarray(r.x, float))
                    out["x_feasible"] = ok
                    out["x_true_obj"] = trueobj
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    print("RESULT" + json.dumps(out), flush=True)
"""
)

_WORKER_SOLVE = (
    _COMMON
    + r"""
tl, flag = float(sys.argv[1]), sys.argv[2]
os.environ["DISCOPT_LP_SPATIAL_MIXED"] = flag
for path in sys.argv[3:]:
    out = {"path": path}
    try:
        m, sgn = load(path)
        t0 = time.perf_counter()
        r = m.solve(time_limit=tl)
        out.update(wall=time.perf_counter() - t0, status=r.status, obj=r.objective,
                   bound=r.bound, gapc=bool(getattr(r, "gap_certified", False)),
                   nodes=getattr(r, "node_count", None),
                   ivf=bool(getattr(r, "incumbent_verification_failed", False)))
        if r.x is not None:
            names = [v.name for v in m._variables]
            flat = np.concatenate([np.atleast_1d(np.asarray(r.x[k], float)).ravel() for k in names])
            ok, trueobj = verify_point(from_nl(path), flat)
            out["x_feasible"] = ok
            out["x_true_obj"] = trueobj
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    print("RESULT" + json.dumps(out), flush=True)
"""
)


def _run_batch(worker, prefix, paths, timeout):
    """Run a BATCH of instances in one interpreter and return {path: result}.

    One process per solve is the cleanest isolation, but the fixed startup cost
    (JAX import + parse + evaluator build) dominated at ~2.3 min per instance
    against ~1 s of actual solving, which made the panel a 10-hour run. Batching
    amortizes it; the batch is kept small so a crash or a memory leak costs only
    that batch, and every instance still builds a FRESH model.
    """
    out = {p: {"error": "no_result"} for p in paths}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker, *prefix, *paths],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_ROOT,
        )
        for ln in proc.stdout.splitlines():
            if ln.startswith("RESULT"):
                r = json.loads(ln[6:])
                out[r.pop("path")] = r
        for p in paths:
            if out[p].get("error") == "no_result":
                out[p]["stderr"] = proc.stderr[-300:]
    except subprocess.TimeoutExpired:
        for p in paths:
            if out[p].get("error") == "no_result":
                out[p] = {"error": "harness_timeout"}
    return out


def instances():
    seen, out = set(), []
    for d in _CORPORA:
        for p in sorted(glob.glob(os.path.join(d, "*.nl"))):
            name = os.path.basename(p)[:-3]
            if name in seen:
                continue
            seen.add(name)
            out.append((name, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", type=float, default=30.0)
    ap.add_argument("--panel", default="AB")
    ap.add_argument("--out", default="scratchpad/panel860_results.json")
    ap.add_argument("--only", default="")
    ap.add_argument("--batch", type=int, default=6)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    rows = {}
    inst = [(n, p) for n, p in instances() if not only or n in only]
    done = 0
    for s0 in range(0, len(inst), a.batch):
        chunk = inst[s0 : s0 + a.batch]
        paths = [p for _n, p in chunk]
        hard = a.tl * len(chunk) + 240.0
        eng = off = on = {}
        if "A" in a.panel:
            eng = _run_batch(_WORKER_ENGINE, [str(a.tl)], paths, hard)
        if "B" in a.panel:
            off = _run_batch(_WORKER_SOLVE, [str(a.tl), "0"], paths, hard)
            on = _run_batch(_WORKER_SOLVE, [str(a.tl), "1"], paths, hard)
        for name, path in chunk:
            row = {"name": name}
            if "A" in a.panel:
                row["engine"] = eng.get(path, {"error": "missing"})
            if "B" in a.panel:
                row["off"] = off.get(path, {"error": "missing"})
                row["on"] = on.get(path, {"error": "missing"})
            rows[name] = row
            done += 1
            print(f"[{done}/{len(inst)}] {json.dumps(row)}", flush=True)
        with open(os.path.join(_ROOT, a.out), "w") as fh:
            json.dump(rows, fh, indent=1)
    print("done")


if __name__ == "__main__":
    main()
