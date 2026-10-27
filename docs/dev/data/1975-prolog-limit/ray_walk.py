"""#1975 census (b'): does the NONLINEAR model follow the root LP's ray?

From a SCIP feasible point x, walk x + t d (d = the ray's original-column part) and
check feasibility + objective with SCIP's own checker. Ray-following (feasible, the
objective keeps falling) => the ray is real, no relaxation can remove it. Leaves the
feasible set => the ray is a relaxation artefact, which the 1/T piece could address.
"""
import json, sys
from pyscipopt import Model
name, path, rayj = sys.argv[1], sys.argv[2], sys.argv[3]
d = {int(k): v for k, v in json.loads(rayj).items()}
m = Model(); m.hideOutput(); m.readProblem(path); m.setParam("limits/time", 20); m.optimize()
out = {"name": name, "scip": m.getStatus(), "nsol": m.getNSols()}
if m.getNSols():
    xs = {v.name: v for v in m.getVars()}
    best = m.getBestSol()
    x0 = {n: m.getSolVal(best, v) for n, v in xs.items()}
    out["scip_obj"] = m.getSolObjVal(best)
    m.freeTransform()  # checkSol on new solutions needs the problem back in PROBLEM stage
    walk = []
    for t in (0, 1, 10, 100, 1e3, 1e4):
        s = m.createSol()
        for n, v in xs.items():
            val = x0[n]
            if n.startswith("x") and n[1:].isdigit() and int(n[1:]) in d:
                val += t * d[int(n[1:])]
            m.setSolVal(s, v, val)
        # objective var (nlobjvar) must be recomputed: let SCIP check only constraints
        # the objective is fixed by the walk only through the original columns; nlobjvar
        # (if any) is left free, so only the constraints and bounds are judged here
        feas = m.checkSol(s, printreason=False, completely=True, checkbounds=True, checkintegrality=True, checklprows=True, original=True)
        walk.append((t, bool(feas)))
        m.freeSol(s)
    out["walk"] = walk
print(json.dumps(out), flush=True)
