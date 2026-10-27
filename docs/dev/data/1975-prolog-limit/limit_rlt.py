"""#1975 step 2 in the scaled limit: max (P - cap) of the root LP + RLT rows.

V0: McCormick only (w0j <= ub_j u0)                       -- the current LP
V1: + issue's H: C2 x (x0 - .2), C4 x (x1 - .2), products x0*x9..17 McCormick
V2: + every bounded-block row C2..C6 x (x0 - .2) and x (x1 - .2)
max(P - cap) == 0 (to LP tolerance) means a zero/negative-cost ray survives;
< 0 strictly means the relaxation's recession cone has positive cost.
"""
import sys
import numpy as np
from scipy.optimize import linprog

B = [2, 3, 4, 5] + list(range(9, 18))         # bounded block
# box upper bounds: FBBT (x2,x3 <= 7097.47, x4,x5 <= 3455.29); x9..17 from C5/C6
C5 = dict(zip(range(9, 18), [1, 1.2, .8, 2, 1.8, 2.4, 3, 2.7, 3.2]))
C6 = dict(zip(range(9, 18), [2, 1.8, 2.2, 3, 3.5, 2.3, 3, 3.2, 2.7]))
UB = {2: 7097.47, 3: 7097.47, 4: 3455.29, 5: 3455.29}
UB.update({j: min(3712 / C5[j], 5000 / C6[j]) for j in range(9, 18)})
OBBT = {2: 1918.37, 3: 1918.37, 4: 861.73, 5: 861.73}
S1, S2, S3 = (9, 10, 11), (12, 13, 14), (15, 16, 17)
QROWS = [  # (coefs, rhs) of C2..C6 in the bounded block
    ({2: 1, 3: 1, **{k: -.94 for k in S1}, **{k: .244 for k in S3}}, 0),
    ({**{k: .064 for k in S1}, **{k: -.58 for k in S2}, **{k: .172 for k in S3}}, 0),
    ({4: 1, 5: 1, **{k: .048 for k in S1}, **{k: .247 for k in S2}, **{k: -.916 for k in S3}}, 0),
    (C5, 3712), (C6, 5000)]

names = ["a", "u0", "u1", "u6", "y18", "y19"] + [f"W0_{j}" for j in B] + [f"W1_{j}" for j in B]
ix = {n: i for i, n in enumerate(names)}


def solve(variant, ub):
    rows, rhs = [], []
    def row(c, b):
        r = np.zeros(len(names))
        for k, v in c.items(): r[ix[k]] += v
        rows.append(r); rhs.append(b)
    row({"W0_2": 1, "W1_4": 1, "y18": -1}, 0); row({"W0_3": 1, "W1_5": 1, "y19": -1}, 0)
    row({"y18": 1, "a": -(3340.8 - 500)}, 500); row({"y19": 1, "a": -(371.2 - 4500)}, 4500)
    row({"u0": 356.474947137148, "u1": 53.7083537310174, "y18": -.564264890180399}, 0)
    row({"u0": 339.983422262764, "u1": 43.5418249774113, "y19": -.405939876920766}, 0)
    row({"u0": 106.946746119538, "u1": 145.018955433089, "y18": -.507117039797071}, 0)
    row({"u0": 173.929713444361, "u1": 203.031384299627, "y19": -.578889145413521}, 0)
    for c0, c1, c6, r in [(.94, -.048, -.064, [(1, 2), (1.2, 1.8), (.8, 2.2)]),
                          (0, -.247, .58, [(2, 3), (1.8, 3.5), (2.4, 2.3)]),
                          (-.244, .916, -.172, [(3, 3), (2.7, 3.2), (3.2, 2.7)])]:
        for c7, c8 in r:
            row({"u0": c0, "u1": c1, "u6": c6, "a": c8 - c7}, c8)
    for k, u in (("0", "u0"), ("1", "u1")):          # McCormick over-estimators, limit form
        for j in B:
            row({f"W{k}_{j}": 1, u: -ub[j]}, 0)
    sel = {"V0": [], "V1": [(0, "0"), (2, "1")], "V2": [(i, k) for i in range(5) for k in "01"]}[variant]
    for i, k in sel:                                   # (x_k - lb) * (rhs - row) >= 0, limit form
        c, b = QROWS[i]
        row({**{f"W{k}_{j}": v for j, v in c.items()}, f"u{k}": -b}, 0)
    obj = np.zeros(len(names))                         # minimise cap - P
    for n in ("W0_2", "W0_3", "W1_4", "W1_5"): obj[ix[n]] = -1
    obj[ix["a"]] = 3712 - 5000
    bounds = [(0, 1)] + [(0, None)] * (len(names) - 1)
    r = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method="highs")
    assert r.status == 0, r.message
    return r.fun + 5000, {n: round(v, 3) for n, v in zip(names, r.x) if abs(v) > 1e-9 and n[0] != "W"}


if __name__ == '__main__':
  n = 0
  for boxname, ub in (("FBBT", UB), ("OBBT", {**UB, **OBBT})):
      for v in ("V0", "V1", "V2"):
          val, pt = solve(v, ub); n += 1
          print(f"{boxname} {v}: min (cap - P) = {val:.6g}   {pt}")
  print("LPs solved:", n); sys.exit(0 if n == 6 else 1)
