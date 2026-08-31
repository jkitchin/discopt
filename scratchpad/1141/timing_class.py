"""Interleaved, repeated timing of the #1141 class win (CLAUDE.md §9).

Three arms per instance -- HiGHS master (what the convex route targets), simplex
master, simplex master + fractional separation -- run A/B/C interleaved, REPS
times, reporting mean and standard deviation rather than a single number. Prints
an executed-run count (§6).
"""
import argparse, os, statistics, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import portfolio2

ARMS = [("highs", "highs", "0"), ("simplex", "simplex", "0"), ("simplex+node", "simplex", "1")]

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--time-limit", type=float, default=60.0)
a = ap.parse_args()

CASES = [
    dict(n=50, K=5, spread=0.002),
    dict(n=40, K=6, spread=0.001, cap_scale=0.7),
    dict(n=60, K=6, spread=0.001),
]

runs = 0
print(f"load before: {os.getloadavg()}")
for kw in CASES:
    walls = {name: [] for name, _, _ in ARMS}
    last = {}
    for _ in range(a.reps):
        for name, backend, nodecuts in ARMS:  # interleaved, not arm-by-arm
            os.environ["DISCOPT_OA_NODE_CUTS"] = nodecuts
            m = portfolio2.build(**kw)
            t = time.perf_counter()
            r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver=backend,
                        time_limit=a.time_limit, gap_tolerance=1e-4)
            walls[name].append(time.perf_counter() - t)
            last[name] = (str(r.status), r.objective, r.bound)
            runs += 1
    print(f"\n{kw}")
    for name, _, _ in ARMS:
        w = walls[name]
        sd = statistics.stdev(w) if len(w) > 1 else 0.0
        st, ob, bd = last[name]
        print(f"  {name:14s} {statistics.mean(w):7.2f} s  sd {sd:5.2f}  n={len(w)}  "
              f"{st:10s} obj={ob!r} bound={bd!r}")
print(f"\nload after: {os.getloadavg()}")
print(f"EXECUTED RUNS: {runs}")
sys.exit(1 if runs == 0 else 0)
