"""Root-gap entry measurement (highs-mip-replacement-plan.md §8.4 rule 3).

Usage: python -u measure.py <names.txt> <instance_dir> <out.jsonl>

Per instance (the population rows where discopt reported no dual bound at 20 s):
  lp_highs / lp_highs_nopre -- HiGHS 1.15.1 on the LP relaxation, threads=1, 60 s cap,
                               presolve default / off (simplex, the default solver)
  lp_ours   -- discopt's LP route on the same relaxation (Rust simplex), 60 s cap; the
               time inside discopt._rust.solve_* calls and the whole Model.solve wall
  milp      -- the 20 s MILP solve under DISCOPT_PROFILE=1 in a child process: the Python
               set-up before the first engine call, the engine time, and the root phases
               (RootCutLoop contains RootSolve and the root's SubMip calls: nested shares,
               not a sum)
Exits non-zero if no instance was measured (CLAUDE.md section 6).
"""
import json, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../../.."))
names = [l.strip() for l in open(sys.argv[1]) if l.strip()]
inst_dir, out_path = sys.argv[2], sys.argv[3]
CAP = 60.0

HIGHS_LP = r'''
import json, sys, time, highspy
h = highspy.Highs()
for k, v in (("output_flag", False), ("threads", 1), ("time_limit", float(sys.argv[2])),
             ("presolve", sys.argv[3])):
    assert h.setOptionValue(k, v) == highspy.HighsStatus.kOk, k
assert h.readModel(sys.argv[1]) == highspy.HighsStatus.kOk
lp = h.getLp(); lp.integrality_ = []
assert h.passModel(lp) == highspy.HighsStatus.kOk
t0 = time.perf_counter(); h.run(); wall = time.perf_counter() - t0
info = h.getInfo()
print("RESULT " + json.dumps({"status": str(h.getModelStatus()).split(".")[-1], "wall": wall,
      "obj": info.objective_function_value, "iters": int(info.simplex_iteration_count)}))
'''

OURS_LP = r'''
import json, os, sys, tempfile, time
sys.path.insert(0, sys.argv[3] + "/discopt_benchmarks/scripts")
import lp_milp_loader as L
import discopt._rust as R
from discopt.modeling.core import from_nl
inst = L.read_instance(sys.argv[1]); inst.is_int[:] = False
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); mdl = from_nl(nl)
spent = [0.0, 0]
for nm in [n for n in dir(R) if n.startswith("solve_")]:
    f = getattr(R, nm)
    def g(*a, _f=f, **k):
        t = time.perf_counter()
        try:
            return _f(*a, **k)
        finally:
            spent[0] += time.perf_counter() - t; spent[1] += 1
    setattr(R, nm, g)
t0 = time.perf_counter(); r = mdl.solve(time_limit=float(sys.argv[2])); wall = time.perf_counter() - t0
print("RESULT " + json.dumps({"status": r.status, "wall": wall, "engine": spent[0],
      "engine_calls": spent[1], "obj": r.objective, "route": r.algorithm_route}))
'''

MILP = r'''
import json, os, sys, tempfile, time
sys.path.insert(0, sys.argv[2] + "/discopt_benchmarks/scripts")
import lp_milp_loader as L
import discopt._rust as R
from discopt.modeling.core import from_nl
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); mdl = from_nl(nl)
marks = []
real = R.solve_milp_csc_py
def spy(*a, **k):
    t = time.perf_counter(); r = real(*a, **k); marks.append((t, time.perf_counter())); return r
R.solve_milp_csc_py = spy
t0 = time.perf_counter(); r = mdl.solve(time_limit=20.0); t1 = time.perf_counter()
print("RESULT " + json.dumps({"status": r.status, "wall": t1 - t0,
      "prework": (marks[0][0] - t0) if marks else None,
      "engine": sum(b - a for a, b in marks), "engine_calls": len(marks)}), flush=True)
'''

PHASES = ("MilpPresolve", "RootCutLoop", "RootSolve", "SubMip", "RandRound", "SearchLoop",
          "NodeLpSolve", "DualPivotLoop", "LuNumeric")
COUNTERS = ("RootCutRounds", "DualExitDeadline", "LpVerdictIterLimit")


def run(cmd, env=None, timeout=400):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env=env or os.environ.copy())
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    return (json.loads(line[-1][7:]) if line else {"status": "ERR", "err": p.stderr[-300:]}), p


def first_block(stderr):
    """Phase totals (ms) and counters of the first profile block: the top-level solve."""
    ph, ct, blk = {}, {}, 0
    for l in stderr.splitlines():
        if l.startswith("--- MILP phase profile"):
            blk += 1
            if blk > 1:
                break
            continue
        if blk != 1:
            continue
        t = l.split()
        if len(t) >= 5 and t[0] in PHASES and t[2] == "calls":
            ph[t[0]] = float(t[3]) / 1000.0
        elif len(t) == 2 and t[0] in COUNTERS:
            ct[t[0]] = int(t[1])
    return ph, ct


measured = 0
with open(out_path, "w") as out:
    for name in names:
        path = os.path.join(inst_dir, f"{name}.mps.gz")
        row = {"name": name}
        row["lp_highs"], _ = run([sys.executable, "-I", "-c", HIGHS_LP, path, str(CAP), "on"])
        row["lp_highs_nopre"], _ = run([sys.executable, "-I", "-c", HIGHS_LP, path, str(CAP), "off"])
        env = {k: v for k, v in os.environ.items() if not k.startswith("DISCOPT_")}
        env.update(RAYON_NUM_THREADS="1")
        row["lp_ours"], _ = run([sys.executable, "-c", OURS_LP, path, str(CAP), REPO], env=env)
        env2 = dict(env, DISCOPT_PROFILE="1")
        row["milp"], p = run([sys.executable, "-c", MILP, path, REPO], env=env2)
        row["phases"], row["counters"] = first_block(p.stderr)
        out.write(json.dumps(row) + "\n"); out.flush()
        measured += 1
        lh, lo, m = row["lp_highs"], row["lp_ours"], row["milp"]
        print(f"{name:28} LP highs {lh.get('wall', 0):6.2f}s ({lh.get('status')}) nopre "
              f"{row['lp_highs_nopre'].get('wall', 0):6.2f}s | ours engine {lo.get('engine', 0):6.2f}s "
              f"wall {lo.get('wall', 0):6.2f}s ({lo.get('status')}) | MILP prework "
              f"{m.get('prework') or 0:5.2f} engine {m.get('engine') or 0:5.2f} "
              f"phases {{{', '.join(f'{k}: {v:.1f}' for k, v in row['phases'].items())}}} "
              f"{row['counters']}", flush=True)
print("measured", measured)
sys.exit(0 if measured else 1)
