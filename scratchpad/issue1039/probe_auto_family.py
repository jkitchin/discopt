"""#1039 bucket D: does ``cuts="auto"`` actually match the family it selects?

``test_auto_matches_best_family_and_preserves_optimum`` is named for that claim but
never asserts it -- both halves compare ``auto`` against the *cut-free* baseline via
an invented ``< base/2`` ratio. Measure the claim the name makes: auto == PSD on
box-QP, auto == RLT on constrained QCQP.

No exception swallowed (§7); executed-comparison count printed, non-zero exit at
zero (§6).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np

assert "/Users/jkitchin/projects/discopt/python/discopt" in dm.__file__, dm.__file__


def qcqp(n, seed, constrained):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = (A + A.T) / 2
    m = dm.Model("q")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    expr = None
    for i in range(n):
        for j in range(n):
            term = float(Q[i, j]) * x[i] * x[j]
            expr = term if expr is None else expr + term
    m.minimize(expr)
    if constrained:
        m.subject_to(dm.sum([x[i] for i in range(n)]) <= 0.6 * n)
        m.subject_to(x[0] + x[1] <= 1.2)
    return m


checks = 0
for n, seed in [(6, 0), (6, 3), (10, 0), (10, 2)]:
    for constrained in (False, True):
        base = qcqp(n, seed, constrained).solve(cuts="manual", time_limit=120)
        auto = qcqp(n, seed, constrained).solve(cuts="auto", time_limit=120)
        psd = qcqp(n, seed, constrained).solve(psd_cuts=True, time_limit=120)
        rlt = qcqp(n, seed, constrained).solve(rlt_cuts=True, time_limit=120)
        fam = "RLT" if constrained else "PSD"
        target = rlt if constrained else psd
        print(
            f"n={n} seed={seed} constrained={constrained!s:5s} base={base.node_count:4d} "
            f"auto={auto.node_count:4d} psd={psd.node_count:4d} rlt={rlt.node_count:4d} "
            f"| auto=={fam}: {auto.node_count == target.node_count} "
            f"obj_ok={abs(float(base.objective) - float(auto.objective)) < 1e-3}",
            flush=True,
        )
        checks += 1

print(f"\nEXECUTED COMPARISONS: {checks}")
if checks == 0:
    sys.exit(1)
