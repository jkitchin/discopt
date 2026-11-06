"""Entry: does the Rust route certify the HiGHS-PRESOLVED model of rows it fails?"""
import json, os, subprocess, sys
sys.path.insert(0, "discopt_benchmarks/scripts")
import highspy, p42_rens_panel as R
S = sys.argv[1]; names = sys.argv[2].split(","); TL = sys.argv[3]
pop = {m["name"]: m for m in R.population()}
env = dict(os.environ, DISCOPT_LP_MILP_BACKEND="rust", PYTHONUNBUFFERED="1")
n = 0
for nm in names:
    src = pop[nm]["path"]; dst = os.path.join(S, nm + ".pre.mps")
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.readModel(src); h.presolve()
    h.writePresolvedModel(dst)
    out = {}
    for tag, path in (("orig", src), ("pre", dst)):
        p = subprocess.run([sys.executable, "discopt_benchmarks/scripts/lp_milp_panel.py", "worker", path, "R", TL, "1e-4", "100000000"],
                           capture_output=True, text=True, env=env, timeout=4 * float(TL) + 120)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        r = json.loads(line[0][7:]) if line else {"status": "ERR " + p.stderr[-200:]}
        out[tag] = (r.get("status"), r.get("gap_certified"), r.get("objective"), r.get("bound"), r.get("nodes"))
    n += 1
    print(f"{nm:22s} opt={pop[nm]['oracle_objective']:.6g} ORIG {out['orig']} | PRESOLVED {out['pre']}", flush=True)
print("executed", n)
