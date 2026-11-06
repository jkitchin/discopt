import json, os, subprocess, sys
sys.path.insert(0, "discopt_benchmarks/scripts")
import d2_direct_driver_panel as D, p42_rens_panel as R
pop = {m["name"]: m for m in R.population()}
n = 0
for nm in sys.argv[1].split(","):
    for arm in ("0", "1"):
        env = dict(os.environ, DISCOPT_MILP_NODE_GMI=arm, DISCOPT_PROFILE="1")
        p = subprocess.run([sys.executable, "-c", D.CHILD, str(D.REPO / pop[nm]["path"]), str(D.HERE.parent), "20"],
                           capture_output=True, text=True, env=env)
        r = json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT")][0][7:])
        lifted = [l.split()[-1] for l in p.stderr.splitlines() if "NodeGmiLifted" in l]
        n += 1
        print(f"{nm:22s} gmi={arm} {r['status']:9s} bound={r['bound']:.4f} obj={r['obj']} nodes={r['nodes']} lifted={lifted}", flush=True)
print("executed", n)
