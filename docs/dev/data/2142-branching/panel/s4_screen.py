"""#2142 step 4 fire screen: which .nl instances does DISCOPT_SPATIAL_BRANCH_CLAMP act on?

  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u s4_screen.py OUT.jsonl TL MAX_NODES file.nl ...

ON arm only (flag =1), DISCOPT_PROFILE=1, deterministic, one fresh process per instance;
records SpatialSplits / SpatialSplitMoved. Prints an executed-assertion count."""
import json, os, subprocess, sys
WT = os.environ["DISCOPT_TREE"]
out, tl, mn, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
CHILD = r"""
import json, sys
import discopt, discopt.modeling as dm
from discopt._rust import profile_counters_py, profile_reset_py
assert discopt.__file__.startswith(sys.argv[4]), discopt.__file__
m = dm.from_nl(sys.argv[1]); profile_reset_py()
r = m.solve(time_limit=float(sys.argv[2]), deterministic=True, max_nodes=int(sys.argv[3]))
c = dict(profile_counters_py())
print("SCREEN " + json.dumps({"status": str(r.status), "nodes": int(r.node_count or 0), "route": str(getattr(r, "algorithm_route", "")),
      "SpatialSplits": int(c["SpatialSplits"]), "SpatialSplitMoved": int(c["SpatialSplitMoved"])}), flush=True)
"""
n = 0
for f in files:
    env = {**os.environ, "DISCOPT_SPATIAL_BRANCH_CLAMP": "1", "DISCOPT_PROFILE": "1", "PYTHONHASHSEED": "0"}
    p = subprocess.run([sys.executable, "-c", CHILD, f, tl, mn, WT], env=env, capture_output=True, text=True, timeout=float(tl) + 300)
    line = [l for l in p.stdout.splitlines() if l.startswith("SCREEN ")]
    rec = {"instance": os.path.basename(f)[:-3]}
    if line:
        rec.update(json.loads(line[0][7:])); n += 1
    else:
        rec["error"] = (p.stderr or "")[-400:]
    print(json.dumps(rec), flush=True)
    open(out, "a").write(json.dumps(rec) + "\n")
print(f"executed assertions: {n}")
sys.exit(0 if n else 1)
