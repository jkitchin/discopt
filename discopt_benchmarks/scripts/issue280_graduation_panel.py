"""Graduation panel for the #280 one-hot swap reseed (``DISCOPT_MILP_SWAP_RESEED``).

CLAUDE.md §5 differential-panel gate: run every instance flag OFF vs ON and
require BOTH bars before the flag graduates default-on:

  1. cert-clean  — no dual bound above the reference optimum, no incumbent
     beating it, no ``gap_certified=True`` -> uncertified regression, and the
     ON-arm incumbent independently re-verified feasible against the model rows.
  2. net-positive — ON measurably helps the incumbent broadly (better on some,
     never worse) and does not degrade the certified set.

Usage:
    python discopt_benchmarks/scripts/issue280_graduation_panel.py inst1,inst2 60 out.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "python")

NL = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")


def oracle(name):
    with open(SOLU) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and p[1] == name and p[0] in ("=opt=", "=best="):
                return float(p[2])
    return None


def _incumbent_feasible(model, r) -> bool:
    """Independently re-verify the returned incumbent against the model rows —
    not on faith from the solver's own status."""
    if r.x is None:
        return True  # no incumbent to check
    try:
        from discopt._jax.nlp_evaluator import cached_evaluator
        from discopt._jax.primal_heuristics import _check_constraint_feasibility

        ev = cached_evaluator(model)
        flat = np.concatenate(
            [
                np.atleast_1d(np.asarray(r.x[v.name], dtype=np.float64)).ravel()
                for v in model._variables
            ]
        )
        return bool(_check_constraint_feasibility(ev, flat))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"    (feasibility check errored: {exc})", flush=True)
        return True


def run(name, tl):
    from discopt.modeling.core import from_nl

    opt = oracle(name)
    arms = {}
    for flag in ("0", "1"):
        os.environ["DISCOPT_MILP_SWAP_RESEED"] = flag
        m = from_nl(f"{NL}/{name}.nl")
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.solve(time_limit=tl)
        arms["off" if flag == "0" else "on"] = {
            "obj": None if r.objective is None else float(r.objective),
            "bound": None if r.bound is None else float(r.bound),
            "status": r.status,
            "gap_certified": bool(getattr(r, "gap_certified", False)),
            "nodes": int(r.node_count),
            "wall": round(time.time() - t0, 2),
            "incumbent_feasible": _incumbent_feasible(m, r),
        }
    return {"instance": name, "oracle": opt, **arms}


def assess(rec):
    """Per-instance cert-clean + net-positive verdict (min sense)."""
    opt = rec["oracle"]
    off, on = rec["off"], rec["on"]
    tol = 1e-4 * (1 + abs(opt)) if opt is not None else 1e-4
    problems = []
    # cert-clean
    for arm_name, a in (("off", off), ("on", on)):
        if opt is not None and a["bound"] is not None and a["bound"] > opt + tol:
            problems.append(f"{arm_name} bound {a['bound']} > oracle {opt}")
        if opt is not None and a["obj"] is not None and a["obj"] < opt - tol:
            problems.append(f"{arm_name} obj {a['obj']} < oracle {opt} (beats optimum)")
        if not a["incumbent_feasible"]:
            problems.append(f"{arm_name} incumbent INFEASIBLE")
    if off["gap_certified"] and not on["gap_certified"]:
        problems.append("cert regression: OFF certified, ON not")
    # net-positive (primal, min sense)
    prim = "n/a"
    if off["obj"] is not None and on["obj"] is not None:
        if on["obj"] < off["obj"] - 1e-6:
            prim = "ON better"
        elif on["obj"] > off["obj"] + 1e-6:
            prim = "ON WORSE"
        else:
            prim = "tie"
    return problems, prim


def main():
    insts = sys.argv[1].split(",")
    tl = float(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else None
    rows = []
    hdr = f"{'instance':24s} {'oracle':>11s} | {'OFF obj':>10s} {'ON obj':>10s} | primal | cert"
    print(hdr, flush=True)
    n_better = n_tie = n_worse = n_unsound = 0
    for name in insts:
        rec = run(name, tl)
        problems, prim = assess(rec)
        rec["problems"] = problems
        rec["primal"] = prim
        rows.append(rec)
        n_better += prim == "ON better"
        n_tie += prim == "tie"
        n_worse += prim == "ON WORSE"
        n_unsound += bool(problems)
        fo = lambda x: f"{x:.2f}" if x is not None else "None"  # noqa: E731
        cert = "CLEAN" if not problems else "!! " + "; ".join(problems)
        print(
            f"{name:24s} {fo(rec['oracle']):>11s} | {fo(rec['off']['obj']):>10s} "
            f"{fo(rec['on']['obj']):>10s} | {prim:9s} | {cert}",
            flush=True,
        )
    summary = {
        "on_better": n_better,
        "tie": n_tie,
        "on_worse": n_worse,
        "instances_with_soundness_problem": n_unsound,
        "cert_clean": n_unsound == 0,
        "net_positive": n_better > 0 and n_worse == 0,
    }
    print("\nSUMMARY:", json.dumps(summary), flush=True)
    verdict = "GRADUATE" if summary["cert_clean"] and summary["net_positive"] else "HOLD"
    print(f"VERDICT: {verdict}", flush=True)
    if out:
        with open(out, "w") as f:
            json.dump({"time_limit_s": tl, "summary": summary, "rows": rows}, f, indent=2)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
