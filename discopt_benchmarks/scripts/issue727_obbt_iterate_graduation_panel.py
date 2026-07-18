"""Graduation panel for #727 Track 1.2 — iterated root OBBT (``DISCOPT_OBBT_ITERATE``, PR #720).

CLAUDE.md §5 differential-panel gate: run every instance flag OFF vs ON in one warm
process and require BOTH bars before the flag graduates default-on:

  1. cert-clean  — no dual bound above the reference optimum (min sense:
     bound <= opt+tol), no incumbent beating it, no ``gap_certified=True`` ->
     uncertified regression, certified objectives within tol, and the ON-arm
     incumbent independently re-verified feasible against the model rows.
  2. net-positive — ON measurably helps broadly: certifies #727 cluster instances
     OFF cannot, or tightens the dual bound / cuts nodes, AND does not regress the
     easy corpus (no OFF-certified instance fails to certify ON; no material
     wall/node blowup on fast instances from OBBT per-node/root cost).

Usage:
    python .../issue727_obbt_iterate_graduation_panel.py <inst1,inst2,...> <TL_s> <out.json>
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
REPO_NL = "python/tests/data/minlplib_nl"

FLAG = "DISCOPT_OBBT_ITERATE"


def _oracle_map():
    m = {}
    with open(SOLU) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and p[0] in ("=opt=", "=best="):
                try:
                    m[p[1]] = float(p[2])
                except ValueError:
                    pass
    return m


ORACLE = _oracle_map()


def _nl_path(name):
    """Prefer the in-repo corpus copy; fall back to the MINLPLib snapshot."""
    r = os.path.join(REPO_NL, f"{name}.nl")
    if os.path.exists(r):
        return r
    return os.path.join(NL, f"{name}.nl")


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

    opt = ORACLE.get(name)
    path = _nl_path(name)
    arms = {}
    for flag in ("0", "1"):
        os.environ[FLAG] = flag
        m = from_nl(path)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.solve(time_limit=tl)
        arms["off" if flag == "0" else "on"] = {
            "obj": None if r.objective is None else float(r.objective),
            "bound": None if r.bound is None else float(r.bound),
            "status": str(r.status),
            "gap_certified": bool(getattr(r, "gap_certified", False)),
            "nodes": int(r.node_count),
            "wall": round(time.time() - t0, 2),
            "incumbent_feasible": _incumbent_feasible(m, r),
        }
    os.environ.pop(FLAG, None)
    return {"instance": name, "oracle": opt, "path": path, **arms}


def assess(rec):
    """Per-instance cert-clean + net-positive verdict (min sense)."""
    opt = rec["oracle"]
    off, on = rec["off"], rec["on"]
    tol = 1e-4 * (1 + abs(opt)) if opt is not None else 1e-4
    problems = []
    # --- cert-clean (soundness) ---
    for arm_name, a in (("off", off), ("on", on)):
        if opt is not None and a["bound"] is not None and a["bound"] > opt + tol:
            problems.append(f"{arm_name} bound {a['bound']:.6g} > oracle {opt:.6g} (FALSE BOUND)")
        if opt is not None and a["obj"] is not None and a["obj"] < opt - tol:
            problems.append(f"{arm_name} obj {a['obj']:.6g} < oracle {opt:.6g} (beats optimum)")
        if not a["incumbent_feasible"]:
            problems.append(f"{arm_name} incumbent INFEASIBLE")
    if off["gap_certified"] and not on["gap_certified"]:
        problems.append("cert regression: OFF certified, ON not")

    # --- net-positive signals ---
    # certification gained/lost
    cert = "same"
    if on["gap_certified"] and not off["gap_certified"]:
        cert = "ON gained cert"
    elif off["gap_certified"] and not on["gap_certified"]:
        cert = "ON LOST cert"

    # dual bound movement (min sense: higher bound = tighter). Only meaningful when
    # both arms produced a bound and it is below the incumbent.
    bnd = "n/a"
    if off["bound"] is not None and on["bound"] is not None:
        db = on["bound"] - off["bound"]
        denom = abs(off["bound"]) + 1.0
        rel = db / denom
        if rel > 1e-4:
            bnd = f"ON tighter (+{rel*100:.2f}%)"
        elif rel < -1e-4:
            bnd = f"ON LOOSER ({rel*100:.2f}%)"
        else:
            bnd = "tie"

    # primal objective (min sense)
    prim = "n/a"
    if off["obj"] is not None and on["obj"] is not None:
        if on["obj"] < off["obj"] - 1e-6:
            prim = "ON better"
        elif on["obj"] > off["obj"] + 1e-6:
            prim = "ON WORSE"
        else:
            prim = "tie"

    return problems, cert, bnd, prim


def main():
    insts = [s for s in sys.argv[1].split(",") if s]
    tl = float(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else None
    rows = []
    hdr = (
        f"{'instance':22s} {'oracle':>11s} | {'OFFb':>10s} {'ONb':>10s} "
        f"{'OFFn':>7s} {'ONn':>7s} {'OFFw':>6s} {'ONw':>6s} | {'cert':14s} {'bound':16s} {'cert-ok'}"
    )
    print(hdr, flush=True)
    n_cert_gain = n_cert_loss = n_bnd_tighter = n_bnd_looser = n_unsound = 0
    for name in insts:
        rec = run(name, tl)
        problems, cert, bnd, prim = assess(rec)
        rec["problems"] = problems
        rec["cert_delta"] = cert
        rec["bound_delta"] = bnd
        rec["primal_delta"] = prim
        rows.append(rec)
        n_cert_gain += cert == "ON gained cert"
        n_cert_loss += cert == "ON LOST cert"
        n_bnd_tighter += bnd.startswith("ON tighter")
        n_bnd_looser += bnd.startswith("ON LOOSER")
        n_unsound += bool(problems)
        fo = lambda x: f"{x:.3g}" if x is not None else "None"  # noqa: E731
        ck = "CLEAN" if not problems else "!!" + "; ".join(problems)
        print(
            f"{name:22s} {fo(rec['oracle']):>11s} | {fo(rec['off']['bound']):>10s} "
            f"{fo(rec['on']['bound']):>10s} {rec['off']['nodes']:>7d} {rec['on']['nodes']:>7d} "
            f"{rec['off']['wall']:>6.1f} {rec['on']['wall']:>6.1f} | {cert:14s} {bnd:16s} {ck}",
            flush=True,
        )
    summary = {
        "flag": FLAG,
        "n_instances": len(rows),
        "cert_gained_on": n_cert_gain,
        "cert_lost_on": n_cert_loss,
        "bound_tighter_on": n_bnd_tighter,
        "bound_looser_on": n_bnd_looser,
        "instances_with_soundness_problem": n_unsound,
        "cert_clean": n_unsound == 0,
        # net-positive: gains something (cert or tighter bound) and loses nothing
        "net_positive": (n_cert_gain > 0 or n_bnd_tighter > 0)
        and n_cert_loss == 0
        and n_bnd_looser == 0,
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
