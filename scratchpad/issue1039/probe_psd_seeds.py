"""#1039 bucket D: derive the PSD / auto node-reduction threshold from a measurement.

The two failing tests each encode ``node_count < base/2`` on a *single* synthetic
6-var box-QP (seed 0). That threshold was never derived from anything -- it is a
round number on one instance, which CLAUDE.md §2 rejects. Measure the reduction
across seeds so a threshold can be set from the distribution.

No exception is swallowed (§7); prints an executed-comparison count and exits
non-zero if it is zero (§6).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np

assert "/Users/jkitchin/projects/discopt/python/discopt" in dm.__file__, dm.__file__


def qcqp(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = (A + A.T) / 2
    m = dm.Model(f"qcqp_n{n}_s{seed}")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    expr = None
    for i in range(n):
        for j in range(n):
            term = float(Q[i, j]) * x[i] * x[j]
            expr = term if expr is None else expr + term
    m.minimize(expr)
    return m


checks = 0
ratios = []
for n, seed in [(8,s) for s in range(6)] + [(10,s) for s in range(4)]:
    base = qcqp(n, seed).solve(cuts="manual", time_limit=120)
    psd = qcqp(n, seed).solve(psd_cuts=True, time_limit=120)
    auto = qcqp(n, seed).solve(cuts="auto", time_limit=120)
    ok = (
        abs(float(base.objective) - float(psd.objective)) < 1e-3
        and abs(float(base.objective) - float(auto.objective)) < 1e-3
    )
    r_psd = psd.node_count / base.node_count
    r_auto = auto.node_count / base.node_count
    ratios.append((n, seed, base.node_count, psd.node_count, auto.node_count, r_psd, r_auto))
    print(
        f"n={n} seed={seed} base={base.node_count:5d} psd={psd.node_count:5d} "
        f"auto={auto.node_count:5d} psd/base={r_psd:.3f} auto/base={r_auto:.3f} "
        f"obj_match={ok}",
        flush=True,
    )
    checks += 1

print()
print(f"worst psd/base  = {max(r[6] for r in ratios):.3f}")
print(f"worst auto/base = {max(r[6] for r in ratios):.3f}")
print(f"EXECUTED COMPARISONS: {checks}")
if checks == 0:
    sys.exit(1)
