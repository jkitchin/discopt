import json, os, re, subprocess, sys
sys.path.insert(0, "discopt_benchmarks/scripts")
import d2_direct_driver_panel as D, miplib_panel as M
names = sys.argv[1].split(","); TL = sys.argv[2]
mem = {m["name"]: m for m in M._load_panel(M.PANEL_JSON)["members"]}
env = dict(os.environ, DISCOPT_PROFILE="1")
done = 0
for n in names:
    m = mem[n]
    p = subprocess.run([sys.executable, "-c", D.CHILD, str(D.REPO / m["path"]), str(D.HERE.parent), TL],
                       capture_output=True, text=True, env=env, timeout=600)
    ctr = {}
    for ln in (p.stdout + p.stderr).splitlines():
        mm = re.search(r"\b(NodeLp\w+)\b\D*?(\d+)", ln)
        if mm: ctr[mm.group(1)] = ctr.get(mm.group(1), 0) + int(mm.group(2))
    res = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
    r = json.loads(res[0][7:]) if res else {}
    it, ito = ctr.get("NodeLpIters", 0), ctr.get("NodeLpItersOverInc", 0)
    print(json.dumps({"name": n, "status": r.get("status"), "nodes": r.get("nodes"), **ctr,
                      "iter_share_over_inc": ito / it if it else None}), flush=True)
    done += bool(ctr)
print("executed with counters:", done); sys.exit(0 if done else 1)
