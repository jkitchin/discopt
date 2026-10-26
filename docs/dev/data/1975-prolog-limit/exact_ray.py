"""#1975 step 2, exact: a rational recession ray of prolog's root LP + RLT rows.

Recession cone of the LP over its open columns (x0, x1, x6, x7, x8, x18, x19 and
the lifted products w0j = x0*xj, w1j = x1*xj); bounded columns get direction 0.
Each row is the homogenised LP row with its DOUBLE coefficients:
  McCormick  w0j <= ub_j*x0 + lb0*xj - lb0*ub_j      ->  d_w0j - ub_j d_x0 <= 0
  RLT        (x0 - lb0)(b - a.x) >= 0                ->  sum a_j d_w0j - b d_x0 <= 0
  model rows C0, C1, C7..C13..C21 (their open columns).
C0, C1, C11, C12 are imposed with equality, d_x7 = 1. Then
  c.d = 3712 + 5000 d8 - (d18 + d19) = 3712 - fl(3340.8) - fl(371.2) < 0 exactly.
Floating point proposes a vertex; the vertex is recomputed in Fractions from its
active set and every row is checked exactly.
"""
import sys
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, ".")
from limit_rlt import B, QROWS, UB, OBBT  # noqa: E402

def build(variant, ub):
    names = ["x0", "x1", "x6", "x7", "x8", "x18", "x19"] + [f"w{k}_{j}" for k in "01" for j in B]
    ix = {n: i for i, n in enumerate(names)}
    ineq, eq = [], []
    def r(c, rhs=0.0, to=ineq):
        to.append(({ix[k]: float(v) for k, v in c.items()}, float(rhs)))
    r({"w0_2": 1, "w1_4": 1, "x18": -1}, to=eq); r({"w0_3": 1, "w1_5": 1, "x19": -1}, to=eq)
    r({"x18": 1, "x7": -3340.8, "x8": -500}, to=eq); r({"x19": 1, "x7": -371.2, "x8": -4500}, to=eq)
    r({"x7": 1}, 1.0, to=eq)
    r({"x0": 356.474947137148, "x1": 53.7083537310174, "x18": -0.564264890180399})
    r({"x0": 339.983422262764, "x1": 43.5418249774113, "x19": -0.405939876920766})
    r({"x0": 106.946746119538, "x1": 145.018955433089, "x18": -0.507117039797071})
    r({"x0": 173.929713444361, "x1": 203.031384299627, "x19": -0.578889145413521})
    for c0, c1, c6, rr in [(.94, -.048, -.064, [(1, 2), (1.2, 1.8), (.8, 2.2)]),
                           (0, -.247, .58, [(2, 3), (1.8, 3.5), (2.4, 2.3)]),
                           (-.244, .916, -.172, [(3, 3), (2.7, 3.2), (3.2, 2.7)])]:
        for c7, c8 in rr:
            r({"x0": c0, "x1": c1, "x6": c6, "x7": -c7, "x8": -c8})
    for k in "01":
        for j in B:
            r({f"w{k}_{j}": 1, f"x{k}": -ub[j]})
    sel = {"V0": [], "V1": [(0, "0"), (2, "1")], "V2": [(i, k) for i in range(5) for k in "01"]}[variant]
    for i, k in sel:
        c, b = QROWS[i]
        r({**{f"w{k}_{j}": v for j, v in c.items()}, f"x{k}": -b})
    return names, ineq, eq

def dense(rows, n):
    A = np.zeros((len(rows), n)); b = np.zeros(len(rows))
    for i, (c, rhs) in enumerate(rows):
        for j, v in c.items(): A[i, j] += v
        b[i] = rhs
    return A, b

def exact_vertex(A_rows, b_rows, n):
    """Solve the square-or-tall rational system exactly (Gauss-Jordan); None if inconsistent/singular."""
    M = [[F(v) for v in row] + [F(b)] for row, b in zip(A_rows, b_rows)]
    r, piv = 0, []
    for c in range(n):
        p = next((k for k in range(r, len(M)) if M[k][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]; inv = 1 / M[r][c]; M[r] = [v * inv for v in M[r]]
        for k in range(len(M)):
            if k != r and M[k][c] != 0:
                f = M[k][c]; M[k] = [a - f * bb for a, bb in zip(M[k], M[r])]
        piv.append(c); r += 1
    if len(piv) != n or any(row[n] != 0 for row in M[r:]): return None
    return [M[i][n] for i in range(n)]

checks = 0
for boxname, ub in (("FBBT", UB), ("OBBT", {**UB, **OBBT})):
    for variant in ("V0", "V1", "V2"):
        names, ineq, eq = build(variant, ub); n = len(names)
        Ai, bi = dense(ineq, n); Ae, be = dense(eq, n)
        lp = linprog(np.zeros(n), A_ub=Ai, b_ub=bi, A_eq=Ae, b_eq=be, bounds=[(0, None)] * n, method="highs-ds")
        if lp.status != 0:
            print(f"{boxname} {variant}: tight-limit system infeasible in floating point ({lp.message})"); continue
        x = lp.x
        # active set: equalities, tight inequalities, zero columns
        act_A = [list(r) for r in Ae] ; act_b = list(be)
        for row, b in zip(Ai, bi):
            if abs(row @ x - b) <= 1e-9 * (1 + np.abs(row).sum()): act_A.append(list(row)); act_b.append(b)
        for j in range(n):
            if x[j] <= 1e-12:
                e = [0.0] * n; e[j] = 1.0; act_A.append(e); act_b.append(0.0)
        d = exact_vertex(act_A, act_b, n)
        assert d is not None, f"{boxname} {variant}: active set is not a vertex"
        assert all(v >= 0 for v in d)
        for c, rhs in ineq:
            assert sum(F(v) * d[j] for j, v in c.items()) <= F(rhs); checks += 1
        for c, rhs in eq:
            assert sum(F(v) * d[j] for j, v in c.items()) == F(rhs); checks += 1
        dd = dict(zip(names, d))
        cost = (F(3712) * dd["x7"] + F(5000) * dd["x8"]
                - (dd["w0_2"] + dd["w0_3"] + dd["w1_4"] + dd["w1_5"]))
        assert cost < 0
        print(f"{boxname} {variant}: EXACT ray, c.d = {float(cost):.6g}  "
              f"(x0={float(dd['x0']):.4g} x1={float(dd['x1']):.4g} x8={float(dd['x8']):.4g})")
print("exact row checks:", checks); sys.exit(0 if checks else 1)
