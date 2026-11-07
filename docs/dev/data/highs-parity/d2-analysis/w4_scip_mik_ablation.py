"""W4 entry: which SCIP cut family certifies mik-250-20-75-*? presolve off, 30 s."""
import sys, json, time
sys.path.insert(0, "/home/user/discopt/docs/dev/data/highs-parity/d2-analysis")
from d2_probe import instance_path
import pyscipopt
ARMS = {
    "default": {},
    "no-gomory": {"separating/gomory/freq": -1},
    "no-aggregation": {"separating/aggregation/freq": -1},
    "no-gomory-no-aggr": {"separating/gomory/freq": -1, "separating/aggregation/freq": -1},
    "root-only-both": {"separating/gomory/freq": 0, "separating/aggregation/freq": 0},
    "aggr-intree-only": {"separating/gomory/freq": 0},
}
n = 0
for name in sys.argv[1].split(","):
    for arm, prm in ARMS.items():
        m = pyscipopt.Model(); m.hideOutput()
        m.readProblem(str(instance_path(name)))
        m.setParam("limits/time", 30.0); m.setParam("limits/gap", 1e-4)
        m.setParam("presolving/maxrounds", 0); m.setParam("parallel/maxnthreads", 1)
        for k, v in prm.items(): m.setParam(k, v)
        t = time.time(); m.optimize()
        print(json.dumps({"name": name, "arm": arm, "status": m.getStatus(), "nodes": m.getNNodes(),
                          "time": round(time.time() - t, 1), "gap": m.getGap(), "primal": m.getPrimalbound(),
                          "dual": m.getDualbound()}), flush=True)
        n += 1
print("executed", n); sys.exit(0 if n else 1)
