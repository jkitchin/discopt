"""#1151: what the stricter verifier costs in certification time.

Panel B showed that on ``balance``/``ratio3`` at a 30 s budget the OFF arm
reports ``optimal`` — at a value BELOW the true optimum — while the ON arm
reports ``feasible`` at the true optimum. OFF's certificate is manufactured by
its own too-low incumbent (the tree stops when ``bound >= incumbent - gap_tol``),
so this measures how long an HONEST certificate takes on the same models.

Interleaved arms, 3 reps, load-gated by the caller; reports mean and spread.
"""
import statistics
import sys
import time

import numpy as np

from discopt.modeling.core import Model
from discopt.validation import feasibility as F

_NEW = F._row_scales


def _old_row_scales(evaluator, x_flat, rows):
    try:
        J = np.asarray(evaluator.evaluate_jacobian(x_flat), dtype=np.float64)
    except Exception:
        return None
    if J.ndim != 2 or J.shape[0] <= int(rows.max()):
        return None
    xw = np.maximum(1.0, np.abs(np.asarray(x_flat, dtype=np.float64)))
    sub = np.abs(J[rows, :]) * xw[None, :]
    if not np.all(np.isfinite(sub)):
        return None
    return np.asarray(sub.max(axis=1), dtype=np.float64)


def build(floor):
    m = Model("balance")
    x = m.continuous("x", lb=floor, ub=1e3)
    y = m.continuous("y", lb=floor, ub=1e3)
    m.minimize(x / y + y / x)
    return m


REPS = 3
BUDGET = 300.0
runs = 0
print(f"{'floor':>8}{'arm':>5}{'rep':>4}{'status':>11}{'objective':>22}{'wall':>9}", flush=True)
acc = {}
for floor in (1e-3, 1e-2):
    for rep in range(REPS):
        for arm in ("on", "off"):
            F._row_scales = _old_row_scales if arm == "off" else _NEW
            t0 = time.perf_counter()
            r = build(floor).solve(solver="bb", time_limit=BUDGET)
            w = time.perf_counter() - t0
            runs += 1
            acc.setdefault((floor, arm), []).append((r.status, r.objective, w))
            print(f"{floor:>8.0e}{arm:>5}{rep:>4}{r.status:>11}{r.objective:>22.15g}{w:>9.2f}",
                  flush=True)

print()
for (floor, arm), rows in sorted(acc.items()):
    walls = [w for _, _, w in rows]
    print(f"floor {floor:.0e} {arm:>3}: status={ {s for s, _, _ in rows} } "
          f"obj={ {round(o, 12) for _, o, _ in rows} } "
          f"wall mean {statistics.mean(walls):.2f}s sd {statistics.pstdev(walls):.2f}")
print(f"executed solves: {runs}")
if runs == 0:
    sys.exit("MEASURED NOTHING")
