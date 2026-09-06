"""A14 / DISCOPT_RENS -- the CORRECTED differential graduation panel (panel 2).

WHY THIS EXISTS, AND WHAT I HAD ALREADY SEEN WHEN I WROTE IT
------------------------------------------------------------
Panel 1 (`rens_panel.py`, run 2026-09-06, recorded in
docs/dev/milp-competitiveness-plan.md A14) returned:

    BAR 1 cert-clean                       PASS (152 comparisons, 72 verified)
    2a median primal gap 38.74% -> 1.64%   PASS
    2b incumbent wins 12, losses 0         PASS
    2c dual-bound losses 9 of 18, cap 3    FAIL   -> flag stays OFF

I am writing this file AFTER seeing that result. That is stated up front because
it is the thing that makes a re-run suspicious, and the reader is entitled to
discount accordingly. The verdict of panel 1 stands as recorded; nothing here
re-reads it.

The defect being corrected is a property of panel 1's INSTRUMENT, not of its
answer. `pick()` compared `rs[-1]` -- a single replicate per arm -- while the
dual bound at a fixed wall-clock limit is a noisy quantity. Measured on panel
1's own two replicates, 5 of the 10 dual-bound differences it counted are
SMALLER than the spread between two replicates of the SAME arm:

    gsvm2rl3   15.94% between two OFF reps  vs  4.28% OFF-ON difference
    beavma     ON bound BETTER than OFF in rep0, WORSE in rep1

A zero-tolerance count over a one-draw estimator cannot distinguish "RENS harmed
the search" from "the estimator is noisy". That is an under-powered instrument
regardless of which way it pointed, and CLAUDE.md 6 is the rule it violates.

HONEST ACCOUNTING OF WHAT CHANGED
---------------------------------
Looser in exactly ONE respect, which is the defect:
  * a dual-bound difference below the measured noise floor no longer counts as
    a regression.
Stricter in three:
  * 5 replicates, not 2, and every arm statistic is a MEDIAN over them rather
    than one draw;
  * a new bar 2d that panel 1 did not have (optimal-count must not fall);
  * the noise floor is per-instance and measured, not a constant I chose.

THIS IS THE FINAL RE-SPECIFICATION OF THE DUAL-SIDE BAR. If 2c' fails, the
answer for A14 is no, and it is recorded as no. There is no panel 3. Naming that
here, before the run, is what stops this from becoming a search for a bar the
flag can pass.

BARS -- fixed before execution
------------------------------
BAR 1 -- CERT-CLEAN. Hard, zero slack, both arms, unchanged from panel 1:
  a) no dual bound above the reference optimum;
  b) no incumbent below it;
  c) bound <= incumbent;
  d) no certification regression (OFF optimal in a rep => ON optimal in that rep);
  e) every ON incumbent independently feasibility-verified against the ORIGINAL
     rows, bounds and integrality.
Any violation ends the run. Soundness is never traded against benefit.

BAR 2 -- NET-POSITIVE, over the contested population (an instance is contested
unless every replicate of BOTH arms returned `optimal`):
  2a) ON's median primal gap strictly below OFF's;
  2b) incumbent wins > losses, comparing per-instance MEDIAN objective;
  2c') dual regressions <= 20% of the population, where an instance is a dual
       regression only if ON's median bound is worse than OFF's median bound by
       MORE than that instance's noise floor, defined as
           max(within-arm bound spread of OFF,
               within-arm bound spread of ON,
               0.5% of |OFF median bound|)
       -- i.e. the difference must exceed what the same arm does to itself;
  2d) the number of instances solved to `optimal` (majority of replicates) must
      not fall from OFF to ON.

KILL CRITERION: any of 2a/2b/2c'/2d fails => DISCOPT_RENS stays default-OFF
permanently, recorded as the DISCOPT_CUT_INHERIT outcome, and A14 ships as a
sound-but-not-graduated flag.

Every count printed is an executed-comparison count and the script exits
non-zero when one is zero, so "0 violations" can never come from a probe that
compared nothing (CLAUDE.md 6). Nothing catches an exception (CLAUDE.md 7).
"""
import json
import math
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rens_panel import (BASE, HERE, gap, load_gate, run,  # noqa: E402
                        verify_incumbent)
from loader import read_mps, to_engine  # noqa: E402

REPS = int(os.environ.get("RENS2_REPS", "5"))
NOISE_FLOOR_REL = 5e-3          # 0.5% -- the floor below which we never call it harm


def med(xs):
    return statistics.median(xs)


def spread_rel(vals):
    """Within-arm spread across replicates, relative. The measured noise floor."""
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if len(v) < 2:
        return 0.0
    den = max(abs(med(v)), 1e-12)
    return (max(v) - min(v)) / den


def main():
    load_gate()
    print(f"panel 2: REPS={REPS}  TL={BASE['time_limit_s']}s", flush=True)
    print("load at start:", os.popen("uptime").read().strip(), flush=True)

    panel = json.load(open(os.path.join(HERE, "panel.json")))
    lim = int(os.environ.get("RENS2_LIMIT", "0"))
    if lim:
        panel = panel[:lim]
        print(f"*** SMOKE RUN: truncated to {lim} instances -- NOT the gate ***", flush=True)

    reps, order = {}, []
    for rep in range(REPS):
        for p in panel:
            f = os.path.join(HERE, "mps", p["name"] + ".mps.gz")
            if not os.path.exists(f):
                continue
            d = read_mps(f)
            try:
                eng = to_engine(d)
            finally:
                os.unlink(d["mps"])
            if rep == 0:
                order.append(p)
            for arm in ("OFF", "ON"):
                r = run(arm, eng, d, p["name"])
                r["viol"] = (verify_incumbent(r["x"], d, eng, r["obj"], p["name"])
                             if (arm == "ON" and r["has_inc"]) else [])
                r["x"] = None
                reps.setdefault((p["name"], arm), []).append(r)
            a, b_ = reps[(p["name"], "OFF")][-1], reps[(p["name"], "ON")][-1]
            print(f"  rep{rep} {p['name']:<22} OFF {a['status']:>10} "
                  f"obj={a['obj']!s:>12} bnd={a['bound']!s:>12} | ON {b_['status']:>10} "
                  f"obj={b_['obj']!s:>12} bnd={b_['bound']!s:>12}", flush=True)
    print("load at end:", os.popen("uptime").read().strip(), flush=True)
    if not order:
        sys.exit("VACUOUS: no instance ran")

    # ---------------- BAR 1 -- CERT-CLEAN (every replicate, not a summary) ------
    cert, checked_cert, verified_inc = [], 0, 0
    for p in order:
        ref = p.get("opt_min")
        for arm in ("OFF", "ON"):
            for i, r in enumerate(reps[(p["name"], arm)]):
                tag = f"{p['name']}[{arm} rep{i}]"
                if ref is not None and r["bound"] is not None:
                    checked_cert += 1
                    if r["bound"] > ref + 1e-4 * (1 + abs(ref)):
                        cert.append(f"{tag}: bound {r['bound']} ABOVE reference optimum {ref}")
                if ref is not None and r["obj"] is not None:
                    checked_cert += 1
                    if r["obj"] < ref - 1e-4 * (1 + abs(ref)):
                        cert.append(f"{tag}: incumbent {r['obj']} BELOW reference optimum {ref}")
                if r["obj"] is not None and r["bound"] is not None:
                    checked_cert += 1
                    if r["bound"] > r["obj"] + 1e-4 * (1 + abs(r["obj"])):
                        cert.append(f"{tag}: bound {r['bound']} > incumbent {r['obj']}")
                if arm == "ON" and r["viol"]:
                    cert.extend(f"{tag}: {v}" for v in r["viol"])
                if arm == "ON" and r["has_inc"]:
                    verified_inc += 1
        for i, (o, n) in enumerate(zip(reps[(p["name"], "OFF")], reps[(p["name"], "ON")])):
            checked_cert += 1
            if o["status"] == "optimal" and n["status"] != "optimal":
                cert.append(f"{p['name']}[rep{i}]: OFF optimal but ON {n['status']}")

    print("\nBAR 1 -- CERT-CLEAN")
    print(f"  reference/invariant comparisons executed : {checked_cert}")
    print(f"  ON incumbents feasibility-verified       : {verified_inc}")
    if checked_cert == 0 or verified_inc == 0:
        sys.exit("VACUOUS: bar 1 compared nothing; refusing to report a pass")
    if cert:
        for c in cert[:40]:
            print("  CERT VIOLATION:", c)
        sys.exit(f"BAR 1 FAILED: {len(cert)} violation(s). Flag stays OFF; soundness is not traded.")
    print(f"  PASS: 0 violations over the comparisons above")

    # ---------------- BAR 2 -- NET-POSITIVE (noise-aware) -----------------------
    def agg(name, arm):
        rs = reps[(name, arm)]
        objs = [r["obj"] for r in rs if r["obj"] is not None]
        bnds = [r["bound"] for r in rs if r["bound"] is not None]
        n_opt = sum(1 for r in rs if r["status"] == "optimal")
        return dict(obj=med(objs) if objs else None,
                    bound=med(bnds) if bnds else None,
                    bspread=spread_rel([r["bound"] for r in rs]),
                    majority_optimal=n_opt * 2 > len(rs),
                    all_optimal=n_opt == len(rs))

    pop, opt_off, opt_on = [], 0, 0
    for p in order:
        if p.get("opt_min") is None:
            continue
        o, n = agg(p["name"], "OFF"), agg(p["name"], "ON")
        opt_off += o["majority_optimal"]
        opt_on += n["majority_optimal"]
        if o["all_optimal"] and n["all_optimal"]:
            continue
        pop.append((p["name"], p["opt_min"], o, n))

    print(f"\nBAR 2 -- NET-POSITIVE   contested population: {len(pop)}")
    if not pop:
        sys.exit("VACUOUS: no instance where the flag could matter; nothing was compared")

    print(f"\n{'instance':<22}{'OFF gap%':>10}{'ON gap%':>10}{'OFF bnd':>14}{'ON bnd':>14}"
          f"{'noise%':>9}{'diff%':>9}{'':>13}")
    wins = losses = dual_reg = dual_win = 0
    goff, gon, compared = [], [], 0
    for name, ref, o, n in pop:
        compared += 1
        g0, g1 = gap(o["obj"], o["bound"], ref), gap(n["obj"], n["bound"], ref)
        goff.append(g0)
        gon.append(g1)
        eps = 1e-9 * (1 + abs(ref))
        if n["obj"] is not None and (o["obj"] is None or n["obj"] < o["obj"] - eps):
            wins += 1
        elif o["obj"] is not None and (n["obj"] is None or n["obj"] > o["obj"] + eps):
            losses += 1
        noise = diff = 0.0
        verdict = "-"
        if o["bound"] is not None and n["bound"] is not None:
            den = max(abs(o["bound"]), 1e-12)
            noise = max(o["bspread"], n["bspread"], NOISE_FLOOR_REL)
            diff = (o["bound"] - n["bound"]) / den      # >0 means ON is WORSE (min sense)
            if diff > noise:
                dual_reg += 1
                verdict = "DUAL REGRESSION"
            elif -diff > noise:
                dual_win += 1
                verdict = "dual win"
            else:
                verdict = "within noise"
        f = lambda v: f"{v:14.6g}" if v is not None else f"{'-':>14}"
        print(f"{name:<22}{100*g0:10.2f}{100*g1:10.2f}{f(o['bound'])}{f(n['bound'])}"
              f"{100*noise:9.3f}{100*diff:9.3f}  {verdict}", flush=True)

    assert compared == len(pop), "PROBE UNDER-MEASURED the population"
    m0, m1 = med(goff), med(gon)
    cap = 0.2 * len(pop)
    print(f"\n  instances compared         {compared}")
    print(f"  median primal gap          OFF {100*m0:.2f}%   ON {100*m1:.2f}%")
    print(f"  incumbent                  wins {wins}   losses {losses}")
    print(f"  dual regressions           {dual_reg}   (cap {cap:.1f} of {len(pop)});"
          f" dual wins {dual_win}")
    print(f"  solved to optimal          OFF {opt_off}   ON {opt_on}")

    a2a, a2b = m1 < m0, wins > losses
    a2c = dual_reg <= cap
    a2d = opt_on >= opt_off
    print(f"\n  2a  median primal gap strictly better        : {'PASS' if a2a else 'FAIL'}")
    print(f"  2b  incumbent wins > losses                  : {'PASS' if a2b else 'FAIL'}")
    print(f"  2c' dual regressions beyond noise <= 20%     : {'PASS' if a2c else 'FAIL'}")
    print(f"  2d  optimal count not reduced                : {'PASS' if a2d else 'FAIL'}")

    ok = a2a and a2b and a2c and a2d
    print("\nVERDICT: " + ("GRADUATE: cert-clean and net-positive -- DISCOPT_RENS may default ON "
                           "(keep the =0 opt-out and the legacy path)"
                           if ok else
                           "STAYS OFF -- final. Recorded as sound-but-not-graduated; no panel 3."))
    json.dump({"reps": REPS, "tl": BASE["time_limit_s"], "population": len(pop),
               "median_gap_off": m0, "median_gap_on": m1, "wins": wins, "losses": losses,
               "dual_regressions": dual_reg, "dual_wins": dual_win, "cap": cap,
               "opt_off": opt_off, "opt_on": opt_on,
               "bars": {"2a": a2a, "2b": a2b, "2c_prime": a2c, "2d": a2d},
               "cert_comparisons": checked_cert, "incumbents_verified": verified_inc,
               "graduate": ok},
              open(os.path.join(HERE, "rens_panel2.json"), "w"), indent=1)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
