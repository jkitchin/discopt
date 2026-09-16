"""Entry experiment for #1248: is there anything for a custom envelope to WIN?

#1248 wants a relaxation-extension API so a plugin can supply CALPHAD-specific
envelopes, on the hypothesis that they beat the generic factorable relaxation.
The #727 RLT lesson says measure that on the REAL class before building the
mechanism: a synthetic root gain of 0.68 there turned out to be 0.0 on real
instances.

For a UNIVARIATE composite on a box, the best possible envelope is the convex
envelope, whose minimum over the box IS the true global minimum. So the gain a
perfect custom envelope could deliver at the root is exactly

    generic_root_bound  vs  true_optimum

and the node count the generic arm needs to close the gap is what a perfect
envelope would save. No API needed to measure it.

KILL CRITERION: if the generic root bound is already at the true optimum (gap
<= 1e-6 relative) on the issue's own `rk_binary` example, a custom univariate
envelope for that class buys nothing and component A's motivating case is
falsified.
"""

import sys

import discopt.modeling as dm
import numpy as np

R_GAS = 8.314


def rk_binary_expr(_m, x, L0, L1, rt=1.0):
    """x(1-x)(L0 + L1(2x-1)) + RT[x ln x + (1-x) ln(1-x)] — the issue's example."""
    return x * (1 - x) * (L0 + L1 * (2 * x - 1)) + rt * (dm.xlogx(x) + dm.xlogx(1 - x))


def true_min_univariate(f, lo, hi, n=2_000_001):
    xs = np.linspace(lo, hi, n)
    return float(np.min(f(xs)))


def numeric_rk(L0, L1, rt):
    def f(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)
            ent = ent + np.where(1 - x > 0, (1 - x) * np.log(np.where(1 - x > 0, 1 - x, 1.0)), 0.0)
        return x * (1 - x) * (L0 + L1 * (2 * x - 1)) + rt * ent

    return f


rows = 0
print(
    f"{'L0':>8} {'L1':>8} {'RT':>6} | {'true min':>12} {'root bound':>12} "
    f"{'root gap':>10} {'nodes':>6} {'status':>10}"
)
for L0, L1, rt in [
    (0.0, 0.0, 1.0),  # entropy only — convex, should be exact
    (3.0, 0.0, 1.0),  # regular solution, miscibility gap
    (5.0, 0.0, 1.0),  # deeper gap
    (3.0, 1.5, 1.0),  # asymmetric RK
    (8.0, -4.0, 1.0),  # strongly asymmetric
    (20000.0, 5000.0, R_GAS * 1000.0),  # J/mol units at 1000 K
]:
    m = dm.Model("rk_binary")
    x = m.continuous("x", lb=1e-9, ub=1 - 1e-9)
    m.minimize(rk_binary_expr(m, x, L0, L1, rt))

    root = m.solve(time_limit=60, max_nodes=1)
    full = m.solve(time_limit=60)
    truth = true_min_univariate(numeric_rk(L0, L1, rt), 1e-9, 1 - 1e-9)

    rb = root.root_bound if root.root_bound is not None else root.bound
    gap = None if rb is None else abs(truth - rb) / max(1.0, abs(truth))
    print(
        f"{L0:8.1f} {L1:8.1f} {rt:6.1f} | {truth:12.6f} "
        f"{(rb if rb is not None else float('nan')):12.6f} "
        f"{(gap if gap is not None else float('nan')):10.2e} "
        f"{full.node_count:6d} {full.status:>10}"
    )
    rows += 1

assert rows == 6, rows
print(f"EXECUTED_CASES={rows}")

# ------------------------------------------------------------------ #
# The AFTER arm: name the composite and let the relaxer envelope it whole.
# ------------------------------------------------------------------ #
print()
print(f"{'L0':>9} {'L1':>9} {'RT':>8} | {'primitive':>14} | {'registered':>14} | ratio")
pairs = 0
for i, (L0, L1, rt) in enumerate(
    [
        (3.0, 0.0, 1.0),
        (5.0, 0.0, 1.0),
        (3.0, 1.5, 1.0),
        (8.0, -4.0, 1.0),
        (20000.0, 5000.0, R_GAS * 1000.0),
    ]
):
    mp = dm.Model("primitive")
    xp = mp.continuous("x", lb=1e-9, ub=1 - 1e-9)
    mp.minimize(rk_binary_expr(mp, xp, L0, L1, rt))
    a = mp.solve(time_limit=120)

    fn = dm.register_function(
        f"rk_binary_{i}",
        lambda x, A=L0, B=L1, C=rt: rk_binary_expr(None, x, A, B, C),
        replace=True,
    )
    mr = dm.Model("registered")
    xr = mr.continuous("x", lb=1e-9, ub=1 - 1e-9)
    mr.minimize(fn(xr))
    b = mr.solve(time_limit=120)

    same = abs(a.objective - b.objective) <= 1e-6 * max(1.0, abs(a.objective))
    ratio = (a.node_count / b.node_count) if b.node_count else float("inf")
    print(
        f"{L0:9.1f} {L1:9.1f} {rt:8.1f} | {a.node_count:9d} nodes | {b.node_count:9d} nodes "
        f"| {ratio:6.1f}x  same_optimum={same}"
    )
    assert same, (a.objective, b.objective)
    pairs += 1

assert pairs == 5, pairs
print(f"EXECUTED_PAIRS={pairs}")
if rows == 0 or pairs == 0:
    print("FAIL: an arm measured nothing", file=sys.stderr)
    raise SystemExit(1)
