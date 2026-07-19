#!/usr/bin/env python3
r"""#282 Workstream C entry experiment C.0 — per-cut-family standalone root gain.

For the convex NLP-BB panel (``rsyn0805m/0810m/0815m``, ``syn40m``), measure how far
each available cut mechanism moves the ROOT dual bound toward the proven ``=opt=``
optimum, and compare against the discopt->SCIP root spread. A family is dead for #282
(CLAUDE.md §4 + the #727 sound!=helpful law) unless it closes **>= 10 % of the
discopt->SCIP root spread** on at least one convex instance.

The metric (plan §C.0):  relative root gain = (bound_with - bound_root)/(opt - bound_root),
reported here in *excess points of opt* so it is directly comparable to the SCIP spread.

Cut mechanisms measured, using **trusted existing infrastructure only** (no
hand-rolled separators — a buggy separator is a worse measurement than none):

  * **B0  = NLP-BB root** (``nlp_bb=True``): the convex *continuous* relaxation the
    default path already solves at the root. This is the baseline ``bound_root``.
  * **OA cuts (path a)**: NOT measured numerically here — for a convex MINLP the
    NLP-BB root IS the exact continuous convex relaxation, so an OA tangent of any
    convex row at the root point is a *valid but redundant* supporting hyperplane
    (it is a tangent to a constraint the relaxation already satisfies exactly).
    Adding it cannot move B0. Path (a) root gain is therefore 0 by construction —
    an outer-approximation LP converges UP to B0 from the loose side, never past it.
  * **Spatial LP root** (``nlp_bb=False``, 0 cut rounds): discopt's McCormick-LP
    relaxation of the big-M reformulation, no cuts. A *different* (>= as loose)
    relaxation than B0.
  * **Spatial LP + discopt cuts** (``nlp_bb=False``, ``root_cut_rounds=R``): the LP
    root after discopt's real root cut loop — Gomory / single-row MIR / c-MIR
    aggregation (``DISCOPT_CMIR_AGGREGATION``) / knapsack-cover / clique. This is the
    trusted measurement of MIR + knapsack-cover (path b) on the real models.
  * **flow-cover**: does not exist anywhere in the tree (plan §C entry). Cannot be
    measured without a prototype; reported as UNMEASURED with a structural note.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=python \
      python discopt_benchmarks/scripts/issue282_c0_cut_family_root_gain.py --time-limit 20
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

SNAPSHOT = Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
NL_DIR = SNAPSHOT / "minlplib" / "nl"
SOLU = SNAPSHOT / "minlplib.solu"

# The convex half of the #282 panel (routes to _solve_nlp_bb).
CONVEX = ["rsyn0805m", "rsyn0810m", "rsyn0815m", "syn40m"]

# discopt vs SCIP root excess (% of opt), from the round-2 diagnosis §R2-2. Used
# only to define the >=10%-of-spread kill threshold; not an input to the solve.
SCIP_ROOT_EXCESS = {
    "rsyn0805m": 16.1,
    "rsyn0810m": 9.5,
    "rsyn0815m": 17.9,
    "syn40m": 5.2,
}


def load_opt(names):
    out = {}
    with open(SOLU) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 3 and p[0] == "=opt=" and p[1] in names:
                with contextlib.suppress(ValueError):
                    out[p[1]] = float(p[2])
    return out


def _f(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if abs(xf) < 1e29 else None


def _excess(bound, opt, is_max):
    """Root dual-bound excess in % of opt (>=0 when sound)."""
    if bound is None or opt is None:
        return None
    raw = (bound - opt) if is_max else (opt - bound)
    return raw / max(1.0, abs(opt)) * 100.0


def _root_bound(name, is_max, *, nlp_bb, cut_rounds, tl, cmir=False):
    from discopt.modeling.core import from_nl

    prev = os.environ.get("DISCOPT_CMIR_AGGREGATION")
    if cmir:
        os.environ["DISCOPT_CMIR_AGGREGATION"] = "1"
    else:
        os.environ.pop("DISCOPT_CMIR_AGGREGATION", None)
    try:
        m = from_nl(str(NL_DIR / f"{name}.nl"))
        kw = {"time_limit": tl, "gap_tolerance": 1e-4, "nlp_bb": nlp_bb}
        if cut_rounds is not None:
            kw["root_cut_rounds"] = cut_rounds
        res = m.solve(**kw)
        return {
            "root_bound": _f(getattr(res, "root_bound", None)),
            "bound": _f(getattr(res, "bound", None)),
            "status": getattr(res, "status", None),
            "nodes": getattr(res, "node_count", None),
        }
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_CMIR_AGGREGATION", None)
        else:
            os.environ["DISCOPT_CMIR_AGGREGATION"] = prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--cut-rounds", type=int, default=20)
    ap.add_argument("--instances", default=",".join(CONVEX))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from discopt.modeling.core import ObjectiveSense, from_nl

    names = [s for s in args.instances.split(",") if s.strip()]
    opts = load_opt(set(names))
    rows = []
    for name in names:
        opt = opts.get(name)
        m = from_nl(str(NL_DIR / f"{name}.nl"))
        is_max = getattr(m._objective, "sense", None) == ObjectiveSense.MAXIMIZE
        spread = None
        print(f"\n=== {name} (opt={opt}, {'max' if is_max else 'min'}) ===", flush=True)

        b0 = _root_bound(name, is_max, nlp_bb=True, cut_rounds=None, tl=args.time_limit)
        sp0 = _root_bound(name, is_max, nlp_bb=False, cut_rounds=0, tl=args.time_limit)
        spc = _root_bound(
            name, is_max, nlp_bb=False, cut_rounds=args.cut_rounds, tl=args.time_limit
        )
        spcm = _root_bound(
            name, is_max, nlp_bb=False, cut_rounds=args.cut_rounds, tl=args.time_limit, cmir=True
        )

        e_b0 = _excess(b0["root_bound"], opt, is_max)
        e_sp0 = _excess(sp0["root_bound"], opt, is_max)
        e_spc = _excess(spc["root_bound"], opt, is_max)
        e_spcm = _excess(spcm["root_bound"], opt, is_max)
        scip = SCIP_ROOT_EXCESS.get(name)
        if e_b0 is not None and scip is not None:
            spread = e_b0 - scip  # discopt->SCIP root spread in excess pts
        kill = None if spread is None else 0.10 * spread

        rec = {
            "name": name,
            "opt": opt,
            "is_max": bool(is_max),
            "nlpbb_root_excess": e_b0,  # B0 baseline (bound_root)
            "spatial_lp_root_excess": e_sp0,  # McCormick LP, no cuts
            "spatial_lp_cuts_root_excess": e_spc,  # + Gomory/MIR/cover
            "spatial_lp_cuts_cmir_root_excess": e_spcm,  # + c-MIR aggregation
            "scip_root_excess": scip,
            "discopt_scip_spread_pts": spread,
            "kill_threshold_pts": kill,
            "nlpbb_root_bound": b0["root_bound"],
            "spatial_lp_root_bound": sp0["root_bound"],
            "spatial_lp_cuts_root_bound": spc["root_bound"],
            "spatial_lp_cuts_cmir_root_bound": spcm["root_bound"],
            # Gain of discopt's real MIR/cover cut loop vs the *same* LP relaxation
            # (spatial LP, no cuts). This is the trusted "do cuts move the root" delta.
            "mir_cover_gain_pts": (None if (e_sp0 is None or e_spc is None) else (e_sp0 - e_spc)),
            "cmir_extra_gain_pts": (
                None if (e_spc is None or e_spcm is None) else (e_spc - e_spcm)
            ),
        }
        # Best root excess any discopt mechanism reaches (min over the excesses).
        cands = [x for x in (e_b0, e_sp0, e_spc, e_spcm) if x is not None]
        rec["best_discopt_root_excess"] = min(cands) if cands else None
        if rec["best_discopt_root_excess"] is not None and e_b0 is not None:
            rec["best_gain_vs_nlpbb_pts"] = e_b0 - rec["best_discopt_root_excess"]
        rows.append(rec)
        print(
            f"  NLP-BB(B0)  root excess: {e_b0}\n"
            f"  spatial LP  (0 cuts):    {e_sp0}\n"
            f"  spatial LP + cuts:       {e_spc}\n"
            f"  spatial LP + cuts+cMIR:  {e_spcm}\n"
            f"  SCIP root excess:        {scip}  | spread={spread} pts  kill>={kill} pts",
            flush=True,
        )

    out = args.out or (
        Path(__file__).resolve().parents[1]
        / "results"
        / "issue282"
        / f"c0_cut_family_root_gain_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(
            {
                "rows": rows,
                "opts": opts,
                "note": (
                    "OA (path a) gain is 0 by construction on convex models (redundant tangent "
                    "of the exact continuous relaxation B0). flow-cover does not exist; UNMEASURED."
                ),
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
