"""#2142: what SCIP's presolve does to the homog-hull kernel spec (120 aggregations).
Builds the model as run.py does, runs presolve only, writes the transformed problem as
.cip and classifies the aggregated variables. Executed-assertion count at the end."""
import re, sys
import numpy as np
import pyscipopt as scip

SPEC = "/home/user/discopt/docs/dev/data/2142-exact-hull/scip/cstr_homog_spec.npz"
d = np.load(SPEC)
n = int(d["n_cols"]); n_orig = int(d["n_orig"])
c = d["c"]; lo = d["global_lo"].astype(float); hi = d["global_hi"].astype(float); integ = d["integrality"]
m = scip.Model("cstr_homog_spec"); m.hideOutput()
x = []
for j in range(n):
    x.append(m.addVar(name=f"x{j}", vtype="I" if integ[j] else "C", lb=None if lo[j] <= -1e19 else float(lo[j]), ub=None if hi[j] >= 1e19 else float(hi[j])))
m.setObjective(scip.quicksum(float(c[j]) * x[j] for j in range(n) if c[j] != 0.0), "minimize")
ptr, cols, coeffs, rhs = d["fixed_row_ptr"], d["fixed_cols"], d["fixed_coeffs"], d["fixed_rhs"]
for r in range(len(rhs)):
    p0, p1 = int(ptr[r]), int(ptr[r + 1])
    m.addCons(scip.quicksum(float(coeffs[k]) * x[int(cols[k])] for k in range(p0, p1)) <= float(rhs[r]), name=f"fix{r}")
tk, ti, tj, tout, tp, tco, tcs = (d[k] for k in ("term_kind", "term_i", "term_j", "term_out", "term_p", "term_coeff", "term_cst"))
for k in range(len(tk)):
    i, j, w, p, co, cs = int(ti[k]), int(tj[k]), int(tout[k]), int(tp[k]), float(tco[k]), float(tcs[k])
    if tk[k] == 0: m.addCons(x[w] == x[i] * x[j], name=f"term{k}")
    elif tk[k] == 1: m.addCons(x[w] == x[i] ** p, name=f"term{k}")
    elif tk[k] == 2: m.addCons(x[w] == (co * x[i] + cs) ** 2, name=f"term{k}")
    elif tk[k] == 3: m.addCons(x[w] == scip.sqrt(co * x[i] + cs), name=f"term{k}")
    elif tk[k] == 4: m.addCons(x[w] == (co * x[i] + cs) ** p, name=f"term{k}")
bw, ap_, ac, aco, acst, bp_, bc, bco, bcst = (d[k] for k in ("blf_w", "blf_a_ptr", "blf_a_cols", "blf_a_coeffs", "blf_a_const", "blf_b_ptr", "blf_b_cols", "blf_b_coeffs", "blf_b_const"))
for k in range(len(bw)):
    A = float(acst[k]) + scip.quicksum(float(aco[q]) * x[int(ac[q])] for q in range(int(ap_[k]), int(ap_[k + 1])))
    B = float(bcst[k]) + scip.quicksum(float(bco[q]) * x[int(bc[q])] for q in range(int(bp_[k]), int(bp_[k + 1])))
    m.addCons(x[int(bw[k])] == A * B, name=f"blf{k}")
m.setParam("presolving/maxrounds", int(sys.argv[1]) if len(sys.argv) > 1 else -1)
m.presolve()
print("presolve status:", m.getStatus(), "nvars", m.getNVars(), "nconss", m.getNConss())
m.writeProblem("presolved_homog.cip", trans=True)
txt = open("presolved_homog.cip").read()
fixed = txt.split("FIXED")[1].split("CONSTRAINTS")[0] if "FIXED" in txt else ""
kinds = {}
rows = []
for line in fixed.splitlines():
    line = line.strip()
    mm = re.match(r"\[(\w+)\] <t_x(\d+)>:(.*)", line)
    if not mm: continue
    kind, j, rest = mm.group(1), int(mm.group(2)), mm.group(3)
    cls = "aux" if j >= n_orig else ("bin" if integ[j] else "cont")
    kinds[(kind, cls)] = kinds.get((kind, cls), 0) + 1
    rows.append((kind, j, cls, rest.strip()))
print("aggregation census (kind, class): ", kinds)
for r in rows: print("  ", r)
n_assert = 0
assert rows, "no FIXED section parsed"; n_assert += 1
print(f"executed assertions: {n_assert}")
