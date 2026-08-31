"""Is the OFF-arm dual bound on a time-limited row STABLE? (#1141)

A "regression" from 1700.0 to 3.98e-12 only means something if 1700.0 is a
property of the algorithm rather than of where the wall happened to fall. This
perturbs the budget (a change that cannot affect soundness) and reports how far
the OFF arm's own bound moves. Prints an executed-run count (§6).
"""
import os, sys, time
from discopt.modeling.core import from_nl

NAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["clay0303hfsg", "fac2"]
DIR = "python/tests/data/minlplib_nl"
LIMITS = [25.0, 30.0, 35.0]

runs = 0
for name in NAMES:
    for arm, cuts in (("off", "0"), ("on", "1")):
        os.environ["DISCOPT_OA_NODE_CUTS"] = cuts
        line = []
        for tl in LIMITS:
            m = from_nl(f"{DIR}/{name}.nl")
            t = time.perf_counter()
            r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb",
                        milp_solver="simplex", time_limit=tl, gap_tolerance=1e-4)
            runs += 1
            line.append(f"tl={tl:g}: bound={r.bound!r} obj={r.objective!r} "
                        f"({time.perf_counter()-t:.1f}s)")
        print(f"{name:16s} {arm:3s}  " + " | ".join(line), flush=True)
print(f"\nEXECUTED RUNS: {runs}")
sys.exit(1 if runs == 0 else 0)
