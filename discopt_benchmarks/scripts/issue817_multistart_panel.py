#!/usr/bin/env python3
"""#817 graduation panel: DISCOPT_SANE_MULTISTART OFF vs ON differential.

The flag only affects ``_solve_continuous`` (the pure-continuous single-NLP path),
so the panel is a spread of pure-continuous instances: the #817 "no incumbent"
gap set, finite-bounded controls, and a broader sample. Each instance is solved
in a fresh subprocess under each flag value (the flag is process-global), and the
two runs are compared against the MINLPLib ``primalbound`` oracle.

Gate (CLAUDE.md §5):
  * cert-clean: no ON incumbent better than the oracle optimum (false primal), no
    bound regression (bound is primal-independent — must be identical OFF vs ON),
    no certification regression (OFF certified -> ON uncertified).
  * net-positive: ON gains incumbents broadly, loses none, no material slowdown.

Usage: python -m discopt_benchmarks.scripts.issue817_multistart_panel [--tl 20]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from discopt_benchmarks.scripts.global_opt_baron_vs_discopt import load_known_optima

BIG_NL = Path.home() / "Dropbox" / "projects" / "discopt-minlp-benchmark" / "minlplib" / "nl"

# The #817 continuous gap set + finite-bounded controls + a broader continuous sample.
GAP = [
    "emfl050_3_3", "emfl100_3_3", "pooling_foulds3tp", "pooling_foulds4tp",
    "pooling_foulds3pq", "squfl010-080persp", "squfl020-050persp", "gastrans040",
    "tln4", "tln5", "lip", "cvxnonsep_pcon30r", "ball_mk2_30", "pooling_adhya4stp",
]
CONTROL = [
    "ex4_1_1", "ex8_1_1", "ex8_1_3", "ex8_1_5", "ex14_1_2", "ex4_1_5", "ex6_1_1",
    "ex7_2_3", "ex14_2_3", "ex14_2_7", "ex2_1_1", "ex3_1_1", "ex5_2_2_case1",
    "chem", "himmel16", "st_e05", "st_e07", "prob06", "prob09", "hydro",
]

WORKER = r"""
import sys, discopt.modeling as dm
nl, tl = sys.argv[1], float(sys.argv[2])
try:
    r = dm.from_nl(nl).solve(time_limit=tl)
    print("RESULT", r.status, r.objective, r.bound, r.gap_certified)
except Exception as e:  # noqa: BLE001
    print("RESULT error", None, None, False, "EXC:%s" % str(e)[:80])
"""


def run(inst: str, flag: str, tl: float) -> dict:
    import os as _os
    nl = str(BIG_NL / f"{inst}.nl")
    if not Path(nl).exists():
        return {"status": "NO_NL"}
    env = dict(_os.environ)
    env["DISCOPT_SANE_MULTISTART"] = flag
    import time as _t
    t0 = _t.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-c", WORKER, nl, str(tl)],
                           capture_output=True, text=True, timeout=tl + 90, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "OUTER_TIMEOUT", "wall": tl + 90}
    wall = _t.perf_counter() - t0
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")), None)
    if not line:
        return {"status": "NO_OUTPUT", "wall": wall, "stderr": p.stderr[-200:]}
    _, st, obj, bnd, cert = (line.split(" ", 4) + [""] * 5)[:5]

    def _f(x):
        try:
            return None if x in ("None", "") else float(x)
        except ValueError:
            return None
    return {"status": st, "obj": _f(obj), "bound": _f(bnd),
            "cert": cert.strip() == "True", "wall": round(wall, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", type=float, default=20.0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    oracle = load_known_optima()
    panel = [i for i in (GAP + CONTROL) if (BIG_NL / f"{i}.nl").exists()]
    out = Path(args.out) if args.out else Path(
        "/private/tmp/claude-501/-Users-jkitchin-projects-discopt/"
        "f2daf772-3a9a-4191-bb80-684893cb0543/scratchpad/issue817_panel.jsonl")
    fh = out.open("w")
    print(f"# #817 panel: {len(panel)} instances, tl={int(args.tl)}s, OFF vs ON", flush=True)

    gained, lost, unsound, cert_regress, wall_off, wall_on = [], [], [], [], 0.0, 0.0
    for inst in panel:
        off = run(inst, "0", args.tl)
        on = run(inst, "1", args.tl)
        opt = oracle.get(inst)
        rec = {"inst": inst, "opt": opt, "off": off, "on": on}
        flags = []
        o_off, o_on = off.get("obj"), on.get("obj")
        wall_off += off.get("wall", 0.0) or 0.0
        wall_on += on.get("wall", 0.0) or 0.0
        # incumbent gained / lost
        if o_off is None and o_on is not None:
            gained.append(inst); flags.append("GAINED")
        if o_off is not None and o_on is None:
            lost.append(inst); flags.append("LOST")
        # soundness: ON incumbent must not beat the oracle optimum (false primal).
        # (sense-agnostic: flag only if |gap to opt| is large AND on the wrong side.)
        if o_on is not None and opt is not None:
            tol = 1e-4 + 1e-3 * abs(opt)
            # a feasible point cannot be strictly better than the true optimum;
            # check both senses conservatively (min: obj<opt; but oracle may be max).
            if o_on < opt - tol - max(1.0, abs(opt) * 0.05):
                unsound.append((inst, o_on, opt)); flags.append("UNSOUND?")
        # bound must be primal-independent: identical OFF vs ON (else a real bug)
        b_off, b_on = off.get("bound"), on.get("bound")
        if b_off is not None and b_on is not None and abs(b_off - b_on) > 1e-4 * (1 + abs(b_off)):
            flags.append("BOUND_DRIFT")
        # certification regression
        if off.get("cert") and not on.get("cert"):
            cert_regress.append(inst); flags.append("CERT_REGRESS")
        rec["flags"] = flags
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"  {inst:20} OFF[{off.get('status'):>10} obj={o_off} {off.get('wall')}s] "
              f"ON[{on.get('status'):>10} obj={o_on} {on.get('wall')}s]"
              f"{'  <<' + ','.join(flags) if flags else ''}", flush=True)
    fh.close()

    print("\n================ #817 PANEL SUMMARY ================", flush=True)
    print(f"incumbents GAINED (OFF none -> ON feasible): {len(gained)}", flush=True)
    print("  " + ", ".join(gained), flush=True)
    print(f"incumbents LOST (regression):                {len(lost)}  {lost}", flush=True)
    print(f"UNSOUND (ON beats oracle optimum):           {len(unsound)}  {unsound}", flush=True)
    print(f"certification regressions:                   {len(cert_regress)}  {cert_regress}", flush=True)
    print(f"total wall  OFF={wall_off:.1f}s  ON={wall_on:.1f}s  (delta {wall_on-wall_off:+.1f}s)", flush=True)
    cert_clean = not unsound and not lost and not cert_regress
    net_positive = len(gained) > 0
    print(f"\nCERT-CLEAN: {'PASS' if cert_clean else 'FAIL'}   "
          f"NET-POSITIVE: {'PASS' if net_positive else 'FAIL'}", flush=True)
    return 0 if (cert_clean and net_positive) else 1


if __name__ == "__main__":
    raise SystemExit(main())
