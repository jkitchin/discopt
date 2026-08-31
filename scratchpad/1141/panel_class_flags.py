"""All three #1141 flags on the class the issue is about, plus the real MINLPLib
portfolio rows the vendored corpus does have (`meanvarx`).

The corpus panels judge the flags over 119 instances; this judges them over the
convex-MIQCP portfolio family, which the corpus barely represents. Interleaved,
with an executed-run count (§6).
"""
import argparse, os, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import portfolio2
from discopt.modeling.core import from_nl

FLAGS = ["DISCOPT_OA_NODE_CUTS", "DISCOPT_OA_ELASTIC_RESTORATION", "DISCOPT_OA_INFEASIBLE_NOGOOD"]
ARMS = [("base", {}),
        ("node", {"DISCOPT_OA_NODE_CUTS": "1"}),
        ("elastic", {"DISCOPT_OA_ELASTIC_RESTORATION": "1"}),
        ("nogood", {"DISCOPT_OA_INFEASIBLE_NOGOOD": "1"}),
        ("all", {f: "1" for f in FLAGS})]

CASES = [("meanvarx", None),
         ("portfolio n=40 K=6", dict(n=40, K=6, spread=0.001, cap_scale=0.7)),
         ("portfolio n=50 K=5", dict(n=50, K=5, spread=0.002)),
         ("portfolio n=60 K=6", dict(n=60, K=6, spread=0.001))]

ap = argparse.ArgumentParser()
ap.add_argument("--time-limit", type=float, default=60.0)
a = ap.parse_args()

runs = 0
for name, kw in CASES:
    print(f"\n{name}")
    for arm, env in ARMS:
        for f in FLAGS:
            os.environ[f] = "0"
        os.environ.update(env)
        m = from_nl("python/tests/data/minlplib/meanvarx.nl") if kw is None else portfolio2.build(**kw)
        t = time.perf_counter()
        r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
                    time_limit=a.time_limit, gap_tolerance=1e-4)
        runs += 1
        summ = ((r.mip_nlp_trace or {}).get("summary") or {})
        cb = summ.get("callback_stats") or {}
        print(f"  {arm:8s} {str(r.status):16s} obj={r.objective!r:24s} bound={r.bound!r:24s} "
              f"{time.perf_counter()-t:6.2f}s  mipnode={cb.get('mipnode_calls')} "
              f"proven_inf={summ.get('proven_infeasible_assignments')} "
              f"restoration={summ.get('restoration_outcomes')}", flush=True)
print(f"\nEXECUTED RUNS: {runs}")
sys.exit(1 if runs == 0 else 0)
