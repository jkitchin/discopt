"""#1151 §6 instrument: does the reported objective equal the objective evaluated
at the solver's own reported point?

Families: affine / bilinear / separable-square / division, each swept over the
positive box floor so the 1/denominator law from the issue is visible (or absent).
Prints an executed-comparison count and exits non-zero if it is zero.
"""
import sys
import numpy as np
from discopt.modeling.core import Model

COMPARISONS = 0


def build(family, floor):
    POS = dict(lb=floor, ub=1e3)
    m = Model(family)
    x = m.continuous("x", **POS)
    y = m.continuous("y", **POS)
    if family == "affine":
        m.minimize(2.0 * x + 3.0 * y)
        f = lambda xv, yv: 2.0 * xv + 3.0 * yv
    elif family == "bilinear":
        m.minimize(x * y + 4.0 / (x * y) if False else x * y)
        f = lambda xv, yv: xv * yv
    elif family == "square":
        m.minimize((x - 1.0) ** 2 + (y - 2.0) ** 2)
        f = lambda xv, yv: (xv - 1.0) ** 2 + (yv - 2.0) ** 2
    elif family == "division":
        m.minimize(x / y + y / x)
        f = lambda xv, yv: xv / yv + yv / xv
    else:
        raise ValueError(family)
    return m, f


def main():
    global COMPARISONS
    worst = {}
    rows = []
    for family in ("affine", "bilinear", "square", "division"):
        for floor in (1e-3, 1e-2, 1e-1, 1e0):
            m, f = build(family, floor)
            r = m.solve(solver="bb")
            if r.x is None or r.objective is None:
                rows.append((family, floor, r.status, None, None, None, None))
                continue
            xv = float(r.x["x"])
            yv = float(r.x["y"])
            oracle = f(xv, yv)
            delta = r.objective - oracle
            denom = min(xv, yv)
            COMPARISONS += 1
            worst[family] = max(worst.get(family, 0.0), abs(delta))
            rows.append((family, floor, r.status, r.objective, oracle, delta, abs(delta) * denom))
    print(f"{'family':<10}{'floor':>8}{'status':>12}{'reported':>22}{'oracle':>22}"
          f"{'delta':>14}{'|d|*denom':>13}")
    for fam, floor, st, rep, orc, d, prod in rows:
        if rep is None:
            print(f"{fam:<10}{floor:>8.0e}{st:>12}{'-':>22}{'-':>22}{'-':>14}{'-':>13}")
        else:
            print(f"{fam:<10}{floor:>8.0e}{st:>12}{rep:>22.15g}{orc:>22.15g}"
                  f"{d:>14.3e}{prod:>13.3e}")
    print()
    print("max |reported - oracle at returned point| per family:")
    for fam, w in worst.items():
        print(f"  {fam:<10} {w:.3e}")
    print(f"executed comparisons: {COMPARISONS}")
    if COMPARISONS == 0:
        print("PROBE MEASURED NOTHING", file=sys.stderr)
        sys.exit(1)


main()
