"""Scaled-up brute-force OA differential (#1141): binaries only, so the integer
box is enumerable, but the trees are big enough to exercise dives, heuristics,
reduced-cost fixing and the root cut loop.

`--opts k=v,...` overrides driver options so the culprit can be bisected.
Prints an executed-comparison count (§6).
"""
import argparse, itertools, sys
import numpy as np
from discopt.solvers.milp_simplex import solve_milp_with_lazy_cuts

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=40)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
compared = 0
bad = 0

for trial in range(a.trials):
    n = a.n
    kfac = 3
    F = rng.normal(size=(n, kfac))
    Q = F @ F.T / kfac + np.diag(0.5 + rng.random(n))   # PSD, dense cross terms
    q = rng.normal(size=n)
    rhs = float(np.trace(Q) / n * rng.uniform(1.5, 4.0))
    c = rng.normal(size=n).round(3)
    A = rng.normal(size=(3, n)).round(3)
    b = (np.abs(A) @ np.full(n, 0.6)).round(3)

    def g(x, _Q=Q, _q=q, _r=rhs):
        x = np.asarray(x, float)
        return float(x @ _Q @ x + _q @ x - _r)

    def cut(x, _Q=Q, _q=q):
        x = np.asarray(x, float)
        gv = g(x)
        if gv <= 1e-9:
            return []
        grad = 2.0 * (_Q @ x) + _q
        return [(grad, float(grad @ x - gv))]

    best = None
    for pt in itertools.product((0.0, 1.0), repeat=n):
        p = np.array(pt)
        if np.any(A @ p > b + 1e-9) or g(p) > 1e-9:
            continue
        v = float(c @ p)
        if best is None or v < best:
            best = v

    for arm in ("off", "on"):
        kw = dict(c=c, A_ub=A, b_ub=b, bounds=[(0.0, 1.0)] * n,
                  integrality=np.ones(n, int), lazy_callback=cut,
                  time_limit=30.0, gap_tolerance=1e-9)
        if arm == "on":
            kw.update(node_callback=cut, node_hook_rounds=a.rounds, node_hook_cut_cap=2000)
        r = solve_milp_with_lazy_cuts(**kw)
        compared += 1
        msg = None
        if best is None:
            if r.objective is not None and r.status.name == "OPTIMAL":
                msg = f"expected infeasible, got obj={r.objective!r}"
        else:
            if r.status.name == "OPTIMAL" and (
                r.objective is None or abs(r.objective - best) > 1e-6 * max(1.0, abs(best))
            ):
                msg = f"certified obj={r.objective!r} but brute force says {best!r}"
            if r.bound is not None and r.bound > best + 1e-6 * max(1.0, abs(best)):
                msg = (msg or "") + f"; bound {r.bound!r} ABOVE true optimum {best!r}"
        if msg:
            bad += 1
            st = dict(r.callback_stats or {})
            print(f"trial {trial} arm={arm}: {msg} (status {r.status.name}, "
                  f"mipnode={st.get('mipnode_calls')}, nodes={r.node_count})")

print(f"\nEXECUTED COMPARISONS: {compared}   MISMATCHES: {bad}")
sys.exit(1 if (bad or compared == 0) else 0)
