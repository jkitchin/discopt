"""S4 entry experiment: seed discopt with HiGHS's optimum on the 16 gap rows.

Arms (20 s, 1 thread, interleaved per row, one solve at a time):
  base   -- default discopt (from_mps loader)
  nopre  -- DISCOPT_MILP_PRESOLVE=0 (a seed file skips MILP presolve; matched control)
  seed   -- DISCOPT_MILP_SEED_POINT=<HiGHS optimum in driver column order>
A seeded run counts only if stderr shows "DISCOPT_MILP_SEED_POINT accepted" (§6).
Rows that certify under `seed` but not `nopre` are primal-bound."""
import json, os, subprocess, sys
import numpy as np
S, OLD = sys.argv[1], sys.argv[2]
REPO = "/home/user/discopt"
rows = open(f"{S}/s3/gap16.txt").read().split()
inst = {r: next(f"{OLD}/popc/inst/{f}" for f in os.listdir(f"{OLD}/popc/inst") if f.split(".")[0] == r) for r in rows}
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
def optimum(r):
    import highspy
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", 120.0); h.setOptionValue("mip_rel_gap", 0.0)
    assert h.readModel(inst[r]) == highspy.HighsStatus.kOk
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal, (r, h.getModelStatus())
    return np.asarray(h.getSolution().col_value), h.getInfo().objective_function_value
def seed_file(r, x):
    from discopt._mps import read_mps
    integ = read_mps(inst[r])["integrality"]
    perm = np.concatenate([np.flatnonzero(integ == 1), np.flatnonzero(integ != 1)])
    xs = x[perm]
    xs[integ[perm] == 1] = np.round(xs[integ[perm] == 1])
    p = f"{S}/s3/seed_{r}.txt"; np.savetxt(p, xs, fmt="%.17g"); return p
def worker(r, extra):
    p = subprocess.run([sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "worker", inst[r], "R", "20", "1e-4", "100000000"],
                       capture_output=True, text=True, env=dict(ENV, **extra), timeout=300)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    d = json.loads(line[7:]) if line else {"status": "ERR", "err": p.stderr[-500:]}
    d["seed_accepted"] = "DISCOPT_MILP_SEED_POINT accepted" in p.stderr
    return d
out = open(f"{S}/s3/s4_entry.jsonl", "w"); n = 0; table = []
for r in rows:
    x, opt = optimum(r)
    sf = seed_file(r, x)
    res = {}
    for arm, extra in (("base", {}), ("nopre", {"DISCOPT_MILP_PRESOLVE": "0"}), ("seed", {"DISCOPT_MILP_SEED_POINT": sf})):
        d = worker(r, extra); n += 1; res[arm] = d
        out.write(json.dumps({"name": r, "arm": arm, "opt": opt, **{k: d.get(k) for k in ("status", "objective", "bound", "nodes", "wall", "seed_accepted", "err")}}) + "\n"); out.flush()
    print(f"{r:28} opt={opt:<14.6g} " + "  ".join(f"{a}:{res[a]['status'][:4]} b={res[a].get('bound')} " for a in res) + f" accepted={res['seed']['seed_accepted']}", flush=True)
    table.append((r, res))
acc = sum(res["seed"]["seed_accepted"] for _, res in table)
primal = [r for r, res in table if res["seed"]["status"] == "optimal" and res["nopre"]["status"] != "optimal"]
print(f"\nseeds accepted {acc}/{len(table)}; certify only when seeded (vs nopre): {primal}")
print("solves:", n); sys.exit(0 if n and acc else 1)
