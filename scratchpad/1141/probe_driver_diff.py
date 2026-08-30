"""Driver-level differential test for the #1141 node hook.

The node separator returns rows that are valid BY CONSTRUCTION -- nonnegative
combinations of the model's own `A_ub` rows -- so any change in the certified
optimum is the driver's, not the cut's. 100 random MILPs; the certified objective
must be identical with and without the hook.

Prints an executed-comparison count and exits non-zero on any mismatch (§6).
"""
import sys
import numpy as np
from discopt.solvers.milp_simplex import solve_milp, solve_milp_with_lazy_cuts

rng = np.random.default_rng(12345)
compared = 0
bad = []

for trial in range(100):
    n = int(rng.integers(4, 12))
    m = int(rng.integers(3, 10))
    A = rng.normal(size=(m, n)).round(2)
    b = (A @ rng.random(n) + rng.random(m) * 2).round(3)
    c = rng.normal(size=n).round(2)
    bounds = [(0.0, float(rng.integers(1, 6))) for _ in range(n)]
    integrality = (rng.random(n) < 0.7).astype(int)

    base = solve_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integrality=integrality,
                      time_limit=20.0, gap_tolerance=1e-9)

    calls = {"n": 0}

    def node_cb(x, _A=A, _b=b):
        calls["n"] += 1
        lam = rng.random(_A.shape[0]) * (rng.random(_A.shape[0]) < 0.4)
        if lam.sum() <= 0:
            lam = np.ones(_A.shape[0]) / _A.shape[0]
        # A nonnegative combination of `A_ub x <= b_ub` rows: valid everywhere.
        return [(lam @ _A, float(lam @ _b))]

    hooked = solve_milp_with_lazy_cuts(
        c=c, A_ub=A, b_ub=b, bounds=bounds, integrality=integrality,
        time_limit=20.0, gap_tolerance=1e-9,
        lazy_callback=lambda x: [],
        node_callback=node_cb, node_hook_rounds=3, node_hook_cut_cap=200,
    )
    compared += 1
    st = dict(hooked.callback_stats or {})
    ob, oh = base.objective, hooked.objective
    bb, bh = base.bound, hooked.bound
    same_obj = (ob is None and oh is None) or (
        ob is not None and oh is not None and abs(ob - oh) <= 1e-6 * max(1.0, abs(ob))
    )
    # The hooked bound may only be TIGHTER on the same problem, never looser than
    # the true optimum: a bound above the plain solve's certified optimum is false.
    bound_bad = (
        bh is not None and ob is not None and bh > ob + 1e-6 * max(1.0, abs(ob))
    )
    if not same_obj or bound_bad:
        bad.append((trial, base.status, hooked.status, ob, oh, bb, bh, st.get("mipnode_calls")))
        print(f"trial {trial}: MISMATCH base={base.status.name} obj={ob!r} bound={bb!r} | "
              f"hooked={hooked.status.name} obj={oh!r} bound={bh!r} mipnode={st.get('mipnode_calls')}")

print(f"\nEXECUTED COMPARISONS: {compared}   MISMATCHES: {len(bad)}")
if compared == 0:
    print("PROBE MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if bad else 0)
