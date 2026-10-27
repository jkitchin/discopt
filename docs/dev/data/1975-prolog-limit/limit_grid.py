"""#1975 step 1: does prolog's NONLINEAR model follow the ray?

Scaled limit (divide every open column by T = x7 + x8, alpha = x7/T, u_k = x_k/T):
the bounded block (x2..x5, x9..x17) stays O(1); C7..C10 lose their O(1) terms;
C0/C1/C11/C12 give  u0*x2 + u1*x4 <= y18 <= 3340.8 a + 500 (1-a), etc.
Asymptotic objective slope = cap(a) - P,  cap = 3712 a + 5000 (1-a),
P = u0 (x2+x3) + u1 (x4+x5). The ray exists in the model iff max(P - cap) >= 0.

For fixed (u0, u1) the problem is an LP in (a, u6, y18, y19, x2..x5, x9..x17).
"""

import sys

import numpy as np
from scipy.optimize import linprog

# variable layout: a, u6, y18, y19, x2, x3, x4, x5, x9..x17  -> 17 columns
A_, U6, Y18, Y19, X2, X3, X4, X5 = range(8)
XS = list(range(8, 17))  # x9..x17
N = 17


def lp_for(u0, u1):
    rows, rhs = [], []

    def row(coefs, b):
        r = np.zeros(N)
        for k, v in coefs.items():
            r[k] += v
        rows.append(r)
        rhs.append(b)

    # bilinear rows (u fixed): u0 x2 + u1 x4 - y18 <= 0 ; u0 x3 + u1 x5 - y19 <= 0
    row({X2: u0, X4: u1, Y18: -1}, 0)
    row({X3: u0, X5: u1, Y19: -1}, 0)
    # C11/C12 limits: y18 <= 3340.8 a + 500 (1-a)
    row({Y18: 1, A_: -(3340.8 - 500)}, 500)
    row({Y19: 1, A_: -(371.2 - 4500)}, 4500)
    # C7..C10 homogeneous parts
    row({Y18: -0.564264890180399}, -(356.474947137148 * u0 + 53.7083537310174 * u1))
    row({Y19: -0.405939876920766}, -(339.983422262764 * u0 + 43.5418249774113 * u1))
    row({Y18: -0.507117039797071}, -(106.946746119538 * u0 + 145.018955433089 * u1))
    row({Y19: -0.578889145413521}, -(173.929713444361 * u0 + 203.031384299627 * u1))
    # C13..C21 (x7 -> a, x8 -> 1-a):  lin(u) - c7 a - c8 (1-a) <= 0
    for c0, c1, c6, rows78 in [
        (0.94, -0.048, -0.064, [(1, 2), (1.2, 1.8), (0.8, 2.2)]),
        (0.0, -0.247, 0.58, [(2, 3), (1.8, 3.5), (2.4, 2.3)]),
        (-0.244, 0.916, -0.172, [(3, 3), (2.7, 3.2), (3.2, 2.7)]),
    ]:
        for c7, c8 in rows78:
            row({U6: c6, A_: -(c7 - c8)}, c8 - c0 * u0 - c1 * u1)
    # bounded block, unscaled (C2..C6)
    s1, s2, s3 = XS[0:3], XS[3:6], XS[6:9]
    row({X2: 1, X3: 1, **{k: -0.94 for k in s1}, **{k: 0.244 for k in s3}}, 0)
    row({**{k: 0.064 for k in s1}, **{k: -0.58 for k in s2}, **{k: 0.172 for k in s3}}, 0)
    row({X4: 1, X5: 1, **{k: 0.048 for k in s1}, **{k: 0.247 for k in s2},
         **{k: -0.916 for k in s3}}, 0)
    row(dict(zip(XS, [1, 1.2, 0.8, 2, 1.8, 2.4, 3, 2.7, 3.2])), 3712)
    row(dict(zip(XS, [2, 1.8, 2.2, 3, 3.5, 2.3, 3, 3.2, 2.7])), 5000)
    # maximise P - cap  ->  minimise cap - P
    c = np.zeros(N)
    c[X2] = c[X3] = -u0
    c[X4] = c[X5] = -u1
    c[A_] = 3712 - 5000  # cap = 5000 + (3712-5000) a ; constant 5000 added below
    bounds = [(0, 1)] + [(0, None)] * (N - 1)
    r = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method="highs")
    if r.status == 2:
        return None
    assert r.status == 0, r.message
    return r.fun + 5000


def main():
    # u0, u1 upper bounds from C7..C10 at their loosest alpha
    u0max = max(0.564264890180399 * 3340.8 / 356.474947137148, 0.405939876920766 * 4500 / 339.983422262764)
    u1max = max(0.507117039797071 * 3340.8 / 145.018955433089, 0.578889145413521 * 4500 / 203.031384299627)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    best, arg, solved = np.inf, None, 0
    for u0 in np.linspace(0, u0max, n):
        for u1 in np.linspace(0, u1max, n):
            v = lp_for(u0, u1)
            if v is None:
                continue
            solved += 1
            if v < best:
                best, arg = v, (u0, u1)
    print(f"u0max={u0max:.4f} u1max={u1max:.4f} grid={n}x{n} feasible LPs={solved}")
    print(f"grid min asymptotic slope (cap - P) = {best:.6g} at u={arg}")
    # The minimum sits on a thin ridge the coarse grid misses: zoom in three times.
    for r in (0.05, 0.005, 0.0005):
        c0, c1 = arg
        for u0 in np.linspace(c0 - r, c0 + r, 41):
            for u1 in np.linspace(c1 - r, c1 + r, 41):
                v = lp_for(u0, u1)
                if v is not None and v < best:
                    best, arg = v, (u0, u1)
                    solved += 1
        print(f"  zoom +-{r}: min (cap - P) = {best:.6g} at u=({arg[0]:.6f}, {arg[1]:.6f})")
    if solved == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
