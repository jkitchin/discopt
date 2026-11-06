"""Python GMI separator inside SCIP, to isolate *where the cuts live* from *how they are made*.

Modes (cut generation code identical; only the bounds used for complementation and the
lifetime/sharing of the row differ):
  L-sub : node LOCAL bounds, row local -> lives only in the subtree (discopt B1' design)
  G-sub : GLOBAL bounds,     row local -> lives only in the subtree
  G-pool: GLOBAL bounds,     row global, also added to SCIP's global cut pool (SCIP design)
All SCIP separators off. Presolve off. Probe counters printed; validity probe checks every
G cut against the known optimum (L cuts are only checked when the optimum is in the box).
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyscipopt as ps
from pyscipopt import SCIP_RESULT

P = "/home/user/discopt/ref/miplib2017/{}.mps.gz"
OPT = json.load(open(__file__.rsplit("/", 1)[0] + "/optsols.json"))
FRAC_MIN = 1e-3


class GMI(ps.Sepa):
    def __init__(self, mode, maxcuts, maxrounds, optsol):
        self.mode, self.maxcuts, self.maxrounds = mode, maxcuts, maxrounds
        self.optsol = optsol
        self.ctr = dict(calls=0, cuts=0, nodes_sep=0, xb_mismatch=0, opt_violated_global=0,
                        opt_violated_local_inbox=0, checked=0)
        self.rounds_at = {}
        self.k = 0

    def sepaexeclp(self):
        m = self.model
        if m.getLPSolstat() != ps.SCIP_LPSOLSTAT.OPTIMAL:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        node = m.getCurrentNode().getNumber()
        r = self.rounds_at.get(node, 0)
        if r >= self.maxrounds:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.rounds_at[node] = r + 1
        if r == 0:
            self.ctr["nodes_sep"] += 1
        self.ctr["calls"] += 1
        cols = m.getLPColsData()
        rows = m.getLPRowsData()
        n, mr = len(cols), len(rows)
        vars_ = [c.getVar() for c in cols]
        # LP matrix rows: s_r = a_r x + const_r, lhs<=s_r<=rhs ; M=[A | -I] (x,s) = -const
        I, J, V = [], [], []
        const = np.zeros(mr)
        lhs = np.zeros(mr); rhs = np.zeros(mr)
        for ri, row in enumerate(rows):
            for col, v in zip(row.getCols(), row.getVals()):
                I.append(ri); J.append(col.getLPPos()); V.append(v)
            const[ri] = row.getConstant(); lhs[ri] = row.getLhs(); rhs[ri] = row.getRhs()
        A = sp.csc_matrix((V, (I, J)), shape=(mr, n))
        M = sp.hstack([A, -sp.identity(mr, format="csc")], format="csc")
        # status/bounds over (x,s)
        st = [c.getBasisStatus() for c in cols] + [row.getBasisStatus() for row in rows]
        if self.mode == "L-sub":
            lb = np.array([v.getLbLocal() for v in vars_]); ub = np.array([v.getUbLocal() for v in vars_])
        else:
            lb = np.array([v.getLbGlobal() for v in vars_]); ub = np.array([v.getUbGlobal() for v in vars_])
        lbl = np.array([v.getLbLocal() for v in vars_]); ubl = np.array([v.getUbLocal() for v in vars_])
        LB = np.concatenate([lb, lhs]); UB = np.concatenate([ub, rhs])
        LBl = np.concatenate([lbl, lhs]); UBl = np.concatenate([ubl, rhs])
        isint = np.array([v.vtype() in ("BINARY", "INTEGER") for v in vars_] + [False] * mr)
        xs = np.array([c.getPrimsol() for c in cols])
        act = A @ xs + const
        val = np.concatenate([xs, act])
        basic = [j for j, s in enumerate(st) if s == "basic"]
        if len(basic) != mr or any(s == "zero" for s in st):
            return {"result": SCIP_RESULT.DIDNOTRUN}
        Bm = M[:, basic].tocsc()
        try:
            lu = spla.splu(Bm)
        except RuntimeError:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        nb = [j for j, s in enumerate(st) if s != "basic"]
        # nonbasic at (local) bound value as SCIP has it
        nbval = np.array([LBl[j] if st[j] == "lower" else UBl[j] for j in nb])
        rhsv = -const - M[:, nb] @ nbval
        xb = lu.solve(rhsv)
        if np.max(np.abs(xb - val[basic])) > 1e-6:
            self.ctr["xb_mismatch"] += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}
        Mnb = M[:, nb].tocsr()
        cands = [(i, bj) for i, bj in enumerate(basic) if isint[bj]
                 and FRAC_MIN <= xb[i] - np.floor(xb[i]) <= 1 - FRAC_MIN]
        cands.sort(key=lambda t: -min(xb[t[0]] - np.floor(xb[t[0]]), np.ceil(xb[t[0]]) - xb[t[0]]))
        added = 0
        local_row = self.mode != "G-pool"
        for i, bj in cands:
            if added >= self.maxcuts:
                break
            e = np.zeros(mr); e[i] = 1.0
            w = lu.solve(e, trans="T")
            abar = Mnb.T @ w  # tableau coefs over nonbasic
            # complement with chosen bounds (LB/UB): local for L, global for G
            at_up = np.array([st[j] == "upper" for j in nb])
            bnd = np.where(at_up, [UB[j] for j in nb], [LB[j] for j in nb])
            if not np.all(np.isfinite(bnd)) or np.any(np.abs(bnd) >= 1e19):
                continue
            alpha = np.where(at_up, -abar, abar)
            # beta = x_B + sum alpha * xtilde* ; xtilde* = |val - bnd|
            xt = np.where(at_up, bnd - val[nb], val[nb] - bnd)
            beta = xb[i] + alpha @ xt
            f0 = beta - np.floor(beta)
            if not (FRAC_MIN <= f0 <= 1 - FRAC_MIN):
                continue
            nbint = np.array([isint[j] and abs(bnd[k] - round(bnd[k])) < 1e-9 for k, j in enumerate(nb)])
            fj = alpha - np.floor(alpha)
            psi_i = np.where(fj <= f0, fj / f0, (1 - fj) / (1 - f0))
            psi_c = np.where(alpha >= 0, alpha / f0, -alpha / (1 - f0))
            psi = np.where(nbint, psi_i, psi_c)
            psi[np.abs(psi) < 1e-12] = 0.0
            # cut sum psi xtilde >= 1 ; xtilde = v - bnd (lower) or bnd - v (upper)
            g = np.zeros(n + mr); rhs_c = 1.0
            for k, j in enumerate(nb):
                if psi[k] == 0.0:
                    continue
                if at_up[k]:
                    g[j] -= psi[k]; rhs_c -= psi[k] * bnd[k]
                else:
                    g[j] += psi[k]; rhs_c += psi[k] * bnd[k]
            # substitute s = A x + const
            gx = g[:n] + A.T @ g[n:]
            rhs_c -= g[n:] @ const
            gx[np.abs(gx) < 1e-11] = 0.0
            nz = np.nonzero(gx)[0]
            if len(nz) == 0:
                continue
            mx, mn = np.max(np.abs(gx[nz])), np.min(np.abs(gx[nz]))
            if mx / mn > 1e7:
                continue
            viol = rhs_c - gx @ xs
            if viol <= 1e-6 * max(1.0, abs(rhs_c)):
                continue
            # validity probe against the known optimum
            o = np.array([self.optsol[v.name[2:] if v.name.startswith("t_") else v.name] for v in vars_])
            inbox = np.all(o >= lbl - 1e-9) and np.all(o <= ubl + 1e-9)
            if self.mode != "L-sub" or inbox:
                self.ctr["checked"] += 1
                if gx @ o < rhs_c - 1e-6 * max(1, abs(rhs_c)):
                    self.ctr["opt_violated_global" if self.mode != "L-sub" else "opt_violated_local_inbox"] += 1
            self.k += 1
            row = m.createEmptyRowSepa(self, f"pgmi{self.k}", lhs=rhs_c, rhs=None,
                                       local=local_row, modifiable=False, removable=True)
            m.cacheRowExtensions(row)
            for j in nz:
                m.addVarToRow(row, vars_[j], float(gx[j]))
            m.flushRowExtensions(row)
            if m.isCutEfficacious(row):
                m.addCut(row, forcecut=False)
                if self.mode == "G-pool":
                    m.addPoolCut(row)
                added += 1
            m.releaseRow(row)
        self.ctr["cuts"] += added
        return {"result": SCIP_RESULT.SEPARATED if added else SCIP_RESULT.DIDNOTFIND}


def run(nm, mode, freq, maxcuts, maxrounds, tl, extra):
    m = ps.Model(); m.hideOutput(); m.readProblem(P.format(nm))
    m.setParam("limits/time", tl); m.setParam("limits/gap", 1e-4); m.setParam("presolving/maxrounds", 0)
    for f in sorted({k.split("/")[1] for k in m.getParams() if k.startswith("separating/") and k.endswith("/freq")}):
        m.setParam(f"separating/{f}/freq", -1)
    for k, v in extra.items():
        m.setParam(k, v)
    s = GMI(mode, maxcuts, maxrounds, OPT[nm])
    m.includeSepa(s, "pygmi", "", priority=-1000, freq=freq, maxbounddist=1.0)
    t = time.time(); m.optimize()
    return dict(status=m.getStatus(), nodes=m.getNNodes(), t=round(time.time() - t, 1),
                bound=round(m.getDualbound(), 4), **s.ctr)


if __name__ == "__main__":
    names = sys.argv[1].split(","); arms = json.loads(sys.argv[2]); tl = float(sys.argv[3])
    n = 0
    for nm in names:
        for a in arms:
            r = run(nm, a["mode"], a.get("freq", 10), a.get("maxcuts", 50), a.get("maxrounds", 5), tl,
                    a.get("extra", {}))
            n += 1
            print(f"{nm[-5:]} {json.dumps(a)} -> {r}", flush=True)
    print("executed", n)
    sys.exit(0 if n else 1)
