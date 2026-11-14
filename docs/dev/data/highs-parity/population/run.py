"""Population check for the HiGHS-parity close-out (plan C, highs-mip-replacement-plan.md).

Usage: python -u run.py <population.tsv> <instance_dir> <out.jsonl>

For each instance (name<TAB>optimum, chosen by select.py before any solve ran) it runs
  ours  -- discopt's graduated default (Rust MILP engine), lp_milp_panel.py worker arm R
  highs -- highspy on the same .mps.gz, threads=1, time_limit=20, mip_rel_gap=1e-4,
           every other option at its default (presolve and restarts on)
20 s each, two solves in parallel, arm order alternating per row. Soundness (CLAUDE.md §1):
a certified optimum must match the MIPLIB optimum (abs 1e-6 + rel 1e-4) and no dual bound may
pass it; our incumbent's max constraint violation is recorded. Exits non-zero when no solve ran
(§6) or on any violation.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
TL = 20.0
pop = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
inst_dir, out_path = sys.argv[2], sys.argv[3]
# Extra discopt arms: ours with these variables set (a flag's panel against "ours"),
# e.g. {"ours_flag": {"DISCOPT_SOME_FLAG": "1"}} with POP_ARMS=ours,ours_flag. Empty: the
# one flag panelled this way so far (DISCOPT_LP_DUAL_SPARSE_SOLVES, 2026-11-13) was
# retired (highs-mip-replacement-plan.md §8.8).
ARM_ENV: dict = {"ours_nlexpr": {"DISCOPT_NL_LINEAR_MATRIX": "0"},
                 "ours_fj": {"DISCOPT_MILP_FEAS_JUMP": "1", "DISCOPT_MILP_TRIVIAL_HEUR": "1"}}
HIGHS = r'''
import json, sys, time, highspy
h = highspy.Highs()
for k, v in (("output_flag", False), ("threads", 1), ("time_limit", float(sys.argv[2])),
             ("mip_rel_gap", 1e-4)):
    assert h.setOptionValue(k, v) == highspy.HighsStatus.kOk, k
assert h.readModel(sys.argv[1]) == highspy.HighsStatus.kOk
t0 = time.perf_counter(); h.run(); wall = time.perf_counter() - t0
ms = h.getModelStatus(); info = h.getInfo()
opt = ms == highspy.HighsModelStatus.kOptimal
sense = h.getLp().sense_
print("RESULT " + json.dumps({
    "status": "optimal" if opt else str(ms).split(".")[-1],
    "objective": info.objective_function_value if info.primal_solution_status == 2 else None,
    "bound": info.mip_dual_bound, "gap_certified": opt, "nodes": int(info.mip_node_count),
    "wall": wall, "maximize": sense == highspy.ObjSense.kMaximize}))
'''


def run(job):
    k, arm = job
    name = pop[k][0]
    path = os.path.join(inst_dir, f"{name}.mps.gz")
    env = {a: b for a, b in os.environ.items() if not a.startswith("DISCOPT_")}
    env.update(PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="1", DISCOPT_LP_MILP_BACKEND="rust")
    if arm in ARM_ENV:
        env.update(ARM_ENV[arm])
    if arm == "ours" or arm in ARM_ENV:
        cmd = [sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "worker",
               path, "R", str(TL), "1e-4", "100000000"]
    else:
        cmd = [sys.executable, "-I", "-c", HIGHS, path, str(TL)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else {"status": "ERR", "err": p.stderr[-400:]}
    except subprocess.TimeoutExpired:
        r = {"status": "ERR", "err": "process timeout 600 s"}
    r["proc_wall"] = time.perf_counter() - t0
    return k, arm, r


# POP_ARMS=ours re-runs one arm and takes the other's rows from POP_REUSE (a previous
# out.jsonl): a change to discopt does not re-time HiGHS.
ARMS = tuple(os.environ.get("POP_ARMS", "ours,highs").split(","))
reuse = {}
if os.environ.get("POP_REUSE"):
    for l in open(os.environ["POP_REUSE"]):
        r = json.loads(l)
        if r["arm"] not in ARMS:
            reuse[(r["name"], r["arm"])] = {k: v for k, v in r.items() if k not in ("name", "arm", "opt")}
ALL = ("ours", "highs") + tuple(ARM_ENV)
jobs = [(k, a) for k in range(len(pop))
        for a in (ALL[k % len(ALL):] + ALL[:k % len(ALL)]) if a in ARMS]
res, viol = {}, []
for k in range(len(pop)):
    for a in ("ours", "highs"):
        if a not in ARMS:
            if (pop[k][0], a) not in reuse:
                sys.exit(f"POP_REUSE has no {a} row for {pop[k][0]}")
            res[(k, a)] = reuse[(pop[k][0], a)]
with open(out_path, "w") as out, ThreadPoolExecutor(2) as ex:
    for k, arm, r in ex.map(run, jobs):
        name, opt = pop[k][0], float(pop[k][1])
        res[(k, arm)] = r
        out.write(json.dumps({"name": name, "arm": arm, "opt": opt, **r}) + "\n")
        out.flush()
        if all((k, a) in res for a in ARMS) and (k, "ours") in res and (k, "highs") in res:
            # MIPLIB optima are in the file's own sense; the HiGHS worker reports the sense.
            sg = -1 if res[(k, "highs")].get("maximize") else 1
            tol = 1e-6 + 1e-4 * abs(opt)
            for a in [a for a in ALL if (k, a) in res]:
                x = res[(k, a)]
                if x.get("status") == "ERR":
                    continue
                if x.get("gap_certified") and (x.get("objective") is None or abs(x["objective"] - opt) > tol):
                    viol.append((name, a, "certified objective", x.get("objective"), opt))
                b = x.get("bound")
                if b is not None and abs(b) < 1e19 and sg * (b - opt) > tol:
                    viol.append((name, a, "bound past optimum", b, opt))
                if x.get("status") == "infeasible":
                    viol.append((name, a, "infeasible verdict", None, opt))
                if a != "highs" and x.get("objective") is not None and (x.get("max_viol") or 0) > 1e-5:
                    viol.append((name, a, "incumbent violation", x.get("max_viol"), opt))
            c = ["OPT " if res[(k, a)].get("gap_certified") else f"{res[(k, a)].get('status', '?')[:4]:4}"
                 for a in ("ours", "highs")]
            print(f"{k:3d} {name[:28]:28} ours {c[0]} {res[(k,'ours')].get('wall') or 0:5.1f}  "
                  f"highs {c[1]} {res[(k,'highs')].get('wall') or 0:5.1f}", flush=True)
n = sum(1 for r in res.values() if r.get("status") != "ERR")
for a in [a for a in ALL if any((k, a) in res for k in range(len(pop)))]:
    rs = [res[(k, a)] for k in range(len(pop)) if (k, a) in res]
    print(f"{a:6} certified {sum(bool(r.get('gap_certified')) for r in rs):4d} of {len(rs)}"
          f"  ERR {sum(r.get('status') == 'ERR' for r in rs)}"
          f"  total_wall {sum(r.get('wall') or 0 for r in rs):8.1f}")
print("violations:", len(viol))
for v in viol:
    print("  VIOLATION", v)
print("solves:", n)
sys.exit(0 if n and not viol else 1)
