"""#2142 item 3 attribution -- SCIP on discopt's own captured big-M spatial-kernel spec
(the lifted MINLP: fixed rows + every lifted term as an exact nonlinear equation), with
SCIP components switched off one at a time, so the gap between our kernel and SCIP can
be attributed to cuts / propagation (incl. OBBT) / presolve / branching.

  --capture <npz>   solve cstr big-M on the baseline tree for 1 deterministic node with a
                    spy on build_spatial_kernel_spec and save the spec the kernel got
                    (presolved bounds included).
  --run <npz> --arm {default,nocuts,noobbt,noprop,nopresolve,nocuts_noobbt,noheur} [--tl 120]

Prints an executed-assertion count (CLAUDE.md §6)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--capture")
ap.add_argument("--run")
ap.add_argument("--arm", default="default")
ap.add_argument("--tl", type=float, default=120.0)
ap.add_argument("--node-limit", type=int, default=0)
ap.add_argument("--param", nargs="*", default=[])
args = ap.parse_args()
n_assert = 0

if args.capture:
    import discopt
    import discopt._relax.spatial_producer as sp
    import discopt.solver as solver
    from discopt.pyomo.gdp import convert_pyomo_gdp

    print("tree:", discopt.__file__)
    import gdplib.cstr as cstr

    orig = sp.build_spatial_kernel_spec
    got = []

    def spy(*a, **k):
        out = orig(*a, **k)
        if out is not None:
            got.append(dict(out))
        return out

    sp.build_spatial_kernel_spec = spy
    for name in dir(solver):
        if getattr(solver, name, None) is orig:
            setattr(solver, name, spy)
    model = convert_pyomo_gdp(cstr.build_model()).model
    res = model.solve(gdp_method=os.environ.get("GDP","big-m"), time_limit=60, deterministic=True, max_nodes=1)
    print("capture solve:", res.status, res.objective, res.bound, res.node_count)
    assert got, "the kernel producer never built a spec"
    n_assert += 1
    spec = got[0]
    np.savez(args.capture, **{k: np.asarray(v) for k, v in spec.items()})
    print(f"captured spec: n_cols={spec['n_cols']} n_orig={spec['n_orig']} terms={len(spec['term_kind'])} "
          f"kinds={sorted(set(spec['term_kind'].tolist()))} blf={len(spec['blf_w'])} fixed={len(spec['fixed_rhs'])} "
          f"sign={spec['meta_obj_sense_sign']} off={spec['meta_obj_offset']}")

if args.run:
    import pyscipopt as scip

    d = np.load(args.run)
    n = int(d["n_cols"])
    c = d["c"]
    lo = d["global_lo"].astype(float)
    hi = d["global_hi"].astype(float)
    integ = d["integrality"]
    sign, off = float(d["meta_obj_sense_sign"]), float(d["meta_obj_offset"])
    m = scip.Model("cstr_bigm_spec")
    m.hideOutput()
    x = []
    for j in range(n):
        lbj = None if lo[j] <= -1e19 else float(lo[j])
        ubj = None if hi[j] >= 1e19 else float(hi[j])
        x.append(m.addVar(name=f"x{j}", vtype="I" if integ[j] else "C", lb=lbj, ub=ubj))
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
    bw, ap_, ac, aco, acst, bp_, bc, bco, bcst = (d[k] for k in ("blf_w", "blf_a_ptr", "blf_a_cols", "blf_a_coeffs", "blf_a_const", "blf_b_ptr", "blf_b_cols", "blf_b_coeffs", "blf_b_const"))
    for k in range(len(bw)):
        A = float(acst[k]) + scip.quicksum(float(aco[q]) * x[int(ac[q])] for q in range(int(ap_[k]), int(ap_[k + 1])))
        B = float(bcst[k]) + scip.quicksum(float(bco[q]) * x[int(bc[q])] for q in range(int(bp_[k]), int(bp_[k + 1])))
        m.addCons(x[int(bw[k])] == A * B)
    m.setParam("limits/time", args.tl)
    if args.node_limit > 0:
        m.setParam("limits/nodes", args.node_limit)
    m.setParam("parallel/maxnthreads", 1)
    m.setParam("lp/threads", 1)
    m.setParam("randomization/randomseedshift", 0)
    arm = args.arm
    if "nocuts" in arm:
        m.setParam("separating/maxrounds", 0)
        m.setParam("separating/maxroundsroot", 0)
    if "noobbt" in arm:
        m.setParam("propagating/obbt/freq", -1)
    if "noprop" in arm:
        m.setParam("propagating/maxrounds", 0)
        m.setParam("propagating/maxroundsroot", 0)
    if "nopresolve" in arm:
        m.setParam("presolving/maxrounds", 0)
    if "noheur" in arm:
        m.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    if "preferbinary" in arm:
        m.setParam("branching/preferbinary", True)
    for kv in args.param:
        k, v = kv.split("=", 1)
        vv = {"true": True, "false": False}.get(v.lower())
        if vv is None:
            vv = float(v) if "." in v else int(v)
        m.setParam(k, vv)
    if "nostrong" in arm:
        m.setParam("branching/relpscost/priority", -1)
        m.setParam("branching/mostinf/priority", 100000)
    t0 = time.perf_counter()
    m.optimize()
    wall = time.perf_counter() - t0
    st = m.getStatus()
    pb = m.getPrimalbound()
    db = m.getDualbound()
    out = {
        "arm": arm,
        "tl": args.tl,
        "node_limit": args.node_limit,
        "status": st,
        "primal_internal": pb,
        "dual_internal": db,
        "primal_model": sign * (pb + off) if np.isfinite(pb) else None,
        "dual_model": sign * (db + off) if np.isfinite(db) else None,
        "nodes": m.getNNodes(),
        "wall": round(wall, 1),
        "ncuts_applied": m.getNCutsApplied() if hasattr(m, "getNCutsApplied") else None,
        "vars": n,
        "fixed_rows": int(len(rhs)),
        "terms": int(len(tk)),
        "blf": int(len(bw)),
    }
    # branching-variable census (per SCIP variable, both directions), by column class
    n_orig = int(d["n_orig"])
    BD = scip.SCIP_BRANCHDIR
    counts = []
    for j, v in enumerate(x):
        try:
            nb = v.getNBranchings(BD.DOWNWARDS) + v.getNBranchings(BD.UPWARDS)
        except Exception as exc:  # recorded, never hidden
            nb = f"EXC {type(exc).__name__}"
        counts.append(nb)
    cls = {"binary_orig": 0, "cont_orig": 0, "aux": 0}
    for j, nb in enumerate(counts):
        if not isinstance(nb, int):
            continue
        key = "aux" if j >= n_orig else ("binary_orig" if integ[j] else "cont_orig")
        cls[key] += nb
    top = sorted(((nb, j) for j, nb in enumerate(counts) if isinstance(nb, int) and nb > 0), reverse=True)[:12]
    out["branchings_by_class"] = cls
    out["top_branch_cols"] = [(j, nb, "aux" if j >= n_orig else ("bin" if integ[j] else "cont")) for nb, j in top]
    print(json.dumps(out), flush=True)
    n_assert += 1
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert > 0 else 1)
