"""Differential probe: do the LP/QP-path Neumaier-Shcherbina safe-bound
implementations agree, and are they each sound (g <= p*)?

Compares, on the SAME LP and the SAME HiGHS duals:
  A = discopt._relax.obbt._ns_safe_lp_lower_bound        (inequality form)
  B = discopt.solvers.milp_simplex._safe_lp_lower_bound_std  (equality std form)

Prints an executed-comparison count and exits non-zero if it is zero
(CLAUDE.md measurement discipline #6).
"""

import sys

import numpy as np
import scipy.sparse as sp
from discopt._relax.obbt import _ns_safe_lp_lower_bound as ns_obbt
from discopt.solvers.milp_simplex import _safe_lp_lower_bound_std as ns_milp
from scipy.optimize import linprog

rng = np.random.default_rng(20260912)

n_cmp = 0  # executed comparisons
unsound_A = unsound_B = 0
disagree = []
abstain_A = abstain_B = 0


def run(n, m, scale, finite_box=True, tag=""):
    global n_cmp, unsound_A, unsound_B, abstain_A, abstain_B
    c = rng.normal(size=n)
    A = rng.normal(size=(m, n))
    # conditioning knob: widen the coefficient range by `scale` orders
    if scale > 0:
        A = A * np.power(10.0, rng.integers(-scale, scale + 1, size=(m, n)).astype(float))
    lo = np.full(n, -1.0)
    hi = np.full(n, 1.0)
    if not finite_box:
        hi[: max(1, n // 4)] = np.inf
    b = A @ np.clip(rng.normal(size=n), -0.5, 0.5) + np.abs(rng.normal(size=m)) + 0.1
    res = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=list(zip(lo.tolist(), [None if np.isinf(h) else h for h in hi])),
        method="highs",
    )
    if not res.success:
        return
    p_star = float(res.fun)
    y = np.asarray(res.ineqlin.marginals, dtype=np.float64)  # <= 0, HiGHS convention

    gA = ns_obbt(c, y, A, b, lo, hi)

    # same LP in equality standard form [A | I] z = b, slacks in [0, inf)
    a_std = sp.hstack([sp.csr_matrix(A), sp.identity(m, format="csr")], format="csr")
    c_std = np.concatenate([c, np.zeros(m)])
    lb_std = np.concatenate([lo, np.zeros(m)])
    ub_std = np.concatenate([hi, np.full(m, np.inf)])
    gB = ns_milp(y, c_std, a_std, b, lb_std, ub_std)

    if gA is None:
        abstain_A += 1
    if gB is None:
        abstain_B += 1
    if gA is None or gB is None:
        return

    n_cmp += 1
    tol = 1e-6 * (1.0 + abs(p_star))
    if gA > p_star + tol:
        unsound_A += 1
    if gB > p_star + tol:
        unsound_B += 1
    rel = abs(gA - gB) / max(1.0, abs(p_star))
    if rel > 1e-9:
        disagree.append((tag, n, m, scale, p_star, gA, gB, rel))


for trial in range(60):
    run(12, 8, 0, True, "well-cond/finite")
for trial in range(60):
    run(12, 8, 6, True, "ill-cond/finite")
for trial in range(60):
    run(25, 15, 3, False, "mixed/infinite-side")

print(f"executed comparisons: {n_cmp}")
print(f"abstained (None): obbt={abstain_A}  milp_simplex={abstain_B}")
print(f"UNSOUND (g > p*): obbt={unsound_A}  milp_simplex={unsound_B}")
print(f"disagreements (rel > 1e-9): {len(disagree)}")
disagree.sort(key=lambda d: -d[-1])
for d in disagree[:8]:
    tag, n, m, s, p, ga, gb, rel = d
    print(f"  {tag:22s} n={n} m={m} p*={p:+.8e} obbt={ga:+.8e} milp={gb:+.8e} rel={rel:.2e}")
if disagree:
    rels = np.array([d[-1] for d in disagree])
    print(f"  disagreement rel: median={np.median(rels):.2e} max={rels.max():.2e}")

if n_cmp == 0:
    print("PROBE FIRED ZERO COMPARISONS", file=sys.stderr)
    sys.exit(2)
