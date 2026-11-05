"""#2142 item 3 step 3 -- ask SCIP which variable / point it branches on for each captured box.

  python -u s3_scip.py CAPTURE.npz OUT.json [--sep]

For each sampled node: SCIP on discopt's own spec (built as in e20_scip.py), every column's
bounds set to that node's post-propagation box, objective limit = the node's cutoff,
presolve off (so the transformed variables are the spec's columns 1:1), heuristics off,
limits/nodes=1, root separation off unless --sep (an in-tree node separates rarely). The
decision is read from the root's children (Node.getParentBranchings); SCIP's last root LP
value of the branched column is recorded by an LPSOLVED event handler. Prints an
executed-assertion count."""
import json, sys
import numpy as np
import pyscipopt as scip

cap, out = sys.argv[1], sys.argv[2]
SEP = "--sep" in sys.argv
NOLIM = "--no-objlimit" in sys.argv
PARAMS = [a.split("=", 1) for a in sys.argv[3:] if "=" in a and not a.startswith("--")]
ONLY = None
for a in sys.argv[3:]:
    if a.startswith("--only-status="):
        ONLY = (a.split("=", 1)[1], json.load(open(sys.argv[sys.argv.index(a) + 1])))
d = np.load(cap)
n = int(d["n_cols"]); integ = d["integrality"].astype(bool)


class LPRec(scip.Eventhdlr):
    def eventinit(self):
        self.model.catchEvent(scip.SCIP_EVENTTYPE.LPSOLVED, self)
        self.last = None

    def eventexit(self):
        self.model.dropEvent(scip.SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexec(self, event):
        if self.model.getLPSolstat() == scip.SCIP_LPSOLSTAT.OPTIMAL:
            self.last = [self.model.getSolVal(None, v) for v in self.model.getVars(transformed=True)]


def build(lo, hi, cutoff):
    m = scip.Model("cstr_bigm_spec")
    m.hideOutput()
    x = []
    for j in range(n):
        lbj = None if lo[j] <= -1e19 else float(lo[j])
        ubj = None if hi[j] >= 1e19 else float(hi[j])
        x.append(m.addVar(name=f"x{j}", vtype="I" if integ[j] else "C", lb=lbj, ub=ubj))
    c = d["c"]
    m.setObjective(scip.quicksum(float(c[j]) * x[j] for j in range(n) if c[j] != 0.0), "minimize")
    ptr, cols, coeffs, rhs = d["fixed_row_ptr"], d["fixed_cols"], d["fixed_coeffs"], d["fixed_rhs"]
    for r in range(len(rhs)):
        p0, p1 = int(ptr[r]), int(ptr[r + 1])
        m.addCons(scip.quicksum(float(coeffs[k]) * x[int(cols[k])] for k in range(p0, p1)) <= float(rhs[r]))
    tk, ti, tj, tout, tp, tco, tcs = (d[k] for k in ("term_kind", "term_i", "term_j", "term_out", "term_p", "term_coeff", "term_cst"))
    for k in range(len(tk)):
        i, j, w, p, co, cs = int(ti[k]), int(tj[k]), int(tout[k]), int(tp[k]), float(tco[k]), float(tcs[k])
        if tk[k] == 0:
            m.addCons(x[w] == x[i] * x[j])
        elif tk[k] == 1:
            m.addCons(x[w] == x[i] ** p)
        elif tk[k] == 2:
            m.addCons(x[w] == (co * x[i] + cs) ** 2)
        elif tk[k] == 3:
            m.addCons(x[w] == scip.sqrt(co * x[i] + cs))
        elif tk[k] == 4:
            m.addCons(x[w] == (co * x[i] + cs) ** p)
        else:
            raise ValueError(tk[k])
    bw = d["blf_w"]
    for k in range(len(bw)):
        A = float(d["blf_a_const"][k]) + scip.quicksum(float(d["blf_a_coeffs"][q]) * x[int(d["blf_a_cols"][q])] for q in range(int(d["blf_a_ptr"][k]), int(d["blf_a_ptr"][k + 1])))
        B = float(d["blf_b_const"][k]) + scip.quicksum(float(d["blf_b_coeffs"][q]) * x[int(d["blf_b_cols"][q])] for q in range(int(d["blf_b_ptr"][k]), int(d["blf_b_ptr"][k + 1])))
        m.addCons(x[int(bw[k])] == A * B)
    m.setParam("limits/nodes", 1)
    m.setParam("presolving/maxrounds", 0)
    m.setParam("parallel/maxnthreads", 1)
    m.setParam("lp/threads", 1)
    m.setParam("randomization/randomseedshift", 0)
    m.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    if not SEP:
        m.setParam("separating/maxroundsroot", 0)
    if np.isfinite(cutoff) and not NOLIM:
        m.setObjlimit(float(cutoff))
    for k, v in PARAMS:
        vv = {"true": True, "false": False}.get(v.lower())
        if vv is None:
            vv = float(v) if "." in v or "e" in v else int(v)
        m.setParam(k, vv)
    return m


rows = []
n_assert = 0
keep = None
if ONLY:
    keep = {r["id"] for r in ONLY[1] if r["status"] == ONLY[0]}
for s in range(len(d["s_id"])):
    if keep is not None and int(d["s_id"][s]) not in keep:
        continue
    lo, hi = d["s_lo"][s], d["s_hi"][s]
    m = build(lo, hi, d["s_cutoff"][s])
    ev = LPRec()
    m.includeEventhdlr(ev, "lprec", "")
    m.optimize()
    st = m.getStatus()
    row = {"id": int(d["s_id"][s]), "status": st, "our_col": int(d["s_col"][s]), "our_val": float(d["s_val"][s]),
           "our_int": bool(integ[d["s_col"][s]]), "lo_c": float(lo[d["s_col"][s]]), "hi_c": float(hi[d["s_col"][s]]),
           "scip_dual": m.getDualbound(), "our_bound": float(d["s_bound"][s])}
    if st == "nodelimit":
        leaves, children, siblings = m.getOpenNodes()
        br = []
        for ch in children:
            pb = ch.getParentBranchings()
            if pb:
                vs, bds, bts = pb
                br.append((vs[0].name, bds[0], int(bts[0])))
        if br:
            name = br[0][0]
            j = int(name.split("x")[-1])
            vals = sorted({b[1] for b in br})
            # integer: down child ub=floor, up child lb=ceil -> the point is floor+0.5
            p = (vals[0] + 0.5) if (integ[j] and len(vals) == 2) else vals[0]
            row.update({"scip_col": j, "scip_val": float(p), "scip_kinds": [b[2] for b in br],
                        "scip_lp_val": None if ev.last is None else float(ev.last[j]),
                        "scip_lo": float(lo[j]), "scip_hi": float(hi[j]), "n_children": len(br)})
            n_assert += 1
    rows.append(row)
    print(json.dumps(row), flush=True)
    m.freeProb()
json.dump(rows, open(out, "w"))
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
