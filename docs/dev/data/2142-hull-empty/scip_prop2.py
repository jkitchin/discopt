"""#2142 §114: copy of 2142-propagation/scip_prop.py with ONE change -- `big` (default 1e19, the
§113 behaviour) sets both SCIP's numerics/infinity (when > 1e20) and the threshold above which a
spec bound is passed as infinite. §113 used 1e19/1e20, under which the eps-perspective products
(1e8)^3 = 1e24 of a perspective binary at 0 are unrepresentable.

SCIP root domain propagation on a spatial-kernel spec at a given box (#2142 item 3).

The spec is loaded exactly as e20_scip.py loads it (fixed rows + every lifted term as an
exact nonlinear equation). Per box: bounds set on the ORIGINAL problem, objective limit =
cutoff, presolve / separation / LP / heuristics / symmetry / conflict / OBBT off,
limits/nodes=1 -> only root domain propagation runs; the transformed variables' global
bounds are read back. Extra SCIP params (e.g. one propagator off) via `params`."""
from __future__ import annotations

import math

import numpy as np
import pyscipopt as scip

BASE_PARAMS = {
    "presolving/maxrounds": 0,
    "separating/maxrounds": 0,
    "separating/maxroundsroot": 0,
    "lp/solvefreq": -1,
    "misc/usesymmetry": 0,
    "conflict/enable": False,
    "propagating/obbt/freq": -1,
    "propagating/nlobbt/freq": -1,
    "propagating/redcost/freq": -1,
    "limits/nodes": 1,
    "parallel/maxnthreads": 1,
    "randomization/randomseedshift": 0,
}


class ScipProp:
    def __init__(self, d, params=None, big=1e19):
        self.d = d
        self.big = big
        n = int(d["n_cols"])
        self.n = n
        m = scip.Model("spec")
        m.hideOutput()
        if big > 1e20:
            m.setParam("numerics/infinity", big * 10)
        m.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
        for k, v in {**BASE_PARAMS, **(params or {})}.items():
            m.setParam(k, v)
        lo, hi, integ, c = d["global_lo"], d["global_hi"], d["integrality"], d["c"]
        x = [m.addVar(name=f"x{j}", vtype="I" if integ[j] else "C",
                      lb=None if lo[j] <= -self.big else float(lo[j]), ub=None if hi[j] >= self.big else float(hi[j]))
             for j in range(n)]
        m.setObjective(scip.quicksum(float(c[j]) * x[j] for j in range(n) if c[j] != 0.0), "minimize")
        ptr, cols, co, rhs = d["fixed_row_ptr"], d["fixed_cols"], d["fixed_coeffs"], d["fixed_rhs"]
        for r in range(len(rhs)):
            m.addCons(scip.quicksum(float(co[k]) * x[int(cols[k])] for k in range(int(ptr[r]), int(ptr[r + 1]))) <= float(rhs[r]))
        for k in range(len(d["term_kind"])):
            kind, i, j, w, p = (int(d[q][k]) for q in ("term_kind", "term_i", "term_j", "term_out", "term_p"))
            co_, cs_ = float(d["term_coeff"][k]), float(d["term_cst"][k])
            if kind == 0:
                m.addCons(x[w] == x[i] * x[j])
            elif kind == 1:
                m.addCons(x[w] == x[i] ** p)
            elif kind == 2:
                m.addCons(x[w] == (co_ * x[i] + cs_) ** 2)
            elif kind == 3:
                m.addCons(x[w] == scip.sqrt(co_ * x[i] + cs_))
            elif kind == 4:
                m.addCons(x[w] == (co_ * x[i] + cs_) ** p)
            else:
                raise ValueError(kind)
        bw = d["blf_w"]
        for k in range(len(bw)):
            A = float(d["blf_a_const"][k]) + scip.quicksum(float(d["blf_a_coeffs"][q]) * x[int(d["blf_a_cols"][q])] for q in range(int(d["blf_a_ptr"][k]), int(d["blf_a_ptr"][k + 1])))
            B = float(d["blf_b_const"][k]) + scip.quicksum(float(d["blf_b_coeffs"][q]) * x[int(d["blf_b_cols"][q])] for q in range(int(d["blf_b_ptr"][k]), int(d["blf_b_ptr"][k + 1])))
            m.addCons(x[int(bw[k])] == A * B)
        self.m, self.x = m, x

    def propagate(self, lo, hi, cutoff):
        """Returns (status, lo, hi). status 'infeasible' = SCIP proved the box empty."""
        m, x = self.m, self.x
        m.freeTransform()
        inf = m.infinity()
        for j, v in enumerate(x):
            m.chgVarLb(v, -inf if lo[j] <= -self.big else float(lo[j]))
            m.chgVarUb(v, inf if hi[j] >= self.big else float(hi[j]))
        m.setObjlimit(float(cutoff) if cutoff is not None and math.isfinite(cutoff) else inf)
        m.optimize()
        st = m.getStatus()
        if st == "infeasible":
            return st, None, None
        L = np.empty(self.n)
        H = np.empty(self.n)
        for j, v in enumerate(x):
            tv = m.getTransformedVar(v)
            L[j] = tv.getLbGlobal()
            H[j] = tv.getUbGlobal()
        L[L <= -inf] = -np.inf
        H[H >= inf] = np.inf
        return st, L, H
