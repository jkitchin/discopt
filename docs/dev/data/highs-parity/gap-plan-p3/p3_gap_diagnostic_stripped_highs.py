"""HiGHS 'discopt_like' (presolve off, restarts off, symmetry off, conflict off) on the rows we lose:
root bound, nodes, status. Prints the rows run (§6)."""
import json, sys, os
sys.argv = ["x", ".", "/dev/null"]
src = open("diag.py").read().split("jobs = [")[0]
exec(src)
OPT2 = "hsab/opt_diag_like.txt"
open(OPT2, "w").write("threads = 1\ntime_limit = 20\npresolve = off\nmip_allow_restart = false\nmip_detect_symmetry = false\n")
lost = ['bppc8-02','issue-2446','mik-250-20-75-1','mik-250-20-75-2','mik-250-20-75-3','mik-250-20-75-5','neos-911970','neos17','b-ball','bell5','enlight8','enlight_hard']
OPT = OPT2
os.environ["HS_NO_CONFLICT"] = "1"
STRIP.remove("HS_NO_CONFLICT") if "HS_NO_CONFLICT" in STRIP else None
n = 0
with open("diag_like.jsonl", "w") as out:
    for m in [p for p in pop if p["name"] in lost]:
        r = run_h(m); opt = m["oracle_objective"]
        g = lambda v: None if v is None or abs(v) == float("inf") else round(abs(opt - v) / max(1, abs(opt)) * 100, 2)
        row = dict(name=m["name"], status=r["status"], nodes=r["nodes"], root=g(r["root_bound"]), dual=g(r["bound"]), restarts=r["restarts"], wall=round(r["wall"], 1))
        out.write(json.dumps(row) + "\n"); out.flush(); n += 1
        print(row, flush=True)
print("rows:", n); sys.exit(0 if n else 1)
