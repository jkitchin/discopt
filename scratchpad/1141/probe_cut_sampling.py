"""Feasible-point SAMPLING against every row the separators returned (#1141).

One reference optimum is one point; a cut can be invalid without touching it.
This samples many genuinely MINLP-feasible points and tests every recorded row at
each. Prints an executed-check count (§6).
"""
import os, sys, pathlib
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import portfolio2
import discopt.solvers.milp_simplex as ms
import discopt.solvers.oa as oa

KW = dict(n=40, K=6, spread=0.001, cap_scale=0.7)
N, K = KW["n"], KW["K"]

# rebuild Sigma / cap exactly as portfolio2.build does
rng0 = np.random.default_rng(0)
F = rng0.normal(scale=0.1, size=(N, 3 if False else 4))
Sigma = F @ F.T / 4 + np.diag(0.01 + 0.02 * rng0.random(N))
_mu = 0.10 + KW["spread"] * rng0.random(N)
CAP = KW["cap_scale"] * float(np.mean(np.diag(Sigma)) / K)

rows = []
_orig = ms.solve_milp_with_lazy_cuts


def wrapped(**kw):
    for key, tag in (("node_callback", "node"), ("lazy_callback", "lazy")):
        cb = kw.get(key)
        if cb is None:
            continue

        def spy(x, _cb=cb, _tag=tag):
            out = _cb(x)
            for co, rhs in out or []:
                rows.append((_tag, np.asarray(co, float).copy(), float(rhs)))
            return out

        kw[key] = spy
    return _orig(**kw)


ms.solve_milp_with_lazy_cuts = wrapped
oa.solve_milp_with_lazy_cuts = wrapped
os.environ["DISCOPT_OA_NODE_CUTS"] = "1"
m = portfolio2.build(**KW)
r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
            time_limit=180, gap_tolerance=1e-4)
print(f"ON obj={r.objective!r} bound={r.bound!r}; recorded rows={len(rows)}")

rng = np.random.default_rng(99)
pts = []
tries = 0
while len(pts) < 3000 and tries < 60000:
    tries += 1
    S = rng.choice(N, size=rng.integers(1, K + 1), replace=False)
    w = rng.random(len(S)) ** rng.uniform(0.3, 3.0)
    w = w / w.sum()
    x = np.zeros(2 * N)
    x[S] = w
    x[N + S] = 1.0
    if float(x[:N] @ Sigma @ x[:N]) <= CAP + 1e-12:
        pts.append(x)
print(f"sampled {len(pts)} MINLP-feasible points (of {tries} tries)")
assert pts, "no feasible sample -- cannot judge the cuts"

P = np.array(pts)
checks = 0
bad = 0
worst = 0.0
for tag, co, rhs in rows:
    d = co.shape[0]
    lhs = P[:, :d] @ co
    checks += P.shape[0]
    v = float(np.max(lhs - rhs))
    if v > 1e-6:
        bad += 1
        worst = max(worst, v)
        if bad <= 6:
            print(f"  INVALID {tag} row: cuts a feasible point by {v:.4e}")

print(f"\nEXECUTED CHECKS: {checks}   INVALID ROWS: {bad}/{len(rows)}   worst violation {worst:.3e}")
sys.exit(1 if (bad or checks == 0) else 0)
