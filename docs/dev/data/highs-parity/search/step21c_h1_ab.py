"""H1 A/B at a fixed node budget. usage: ab.py <names> <arms> <max_nodes> [par]
Each row: name arm status dual-gap% nodes wall + profile counters; BOUND>OPT flags a P0."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
W = "/home/user/discopt/.claude/worktrees/agent-a79ed5165031c37b8"
sys.path.insert(0, f"{W}/discopt_benchmarks/scripts")
import p42_rens_panel as R

BASE = {"DISCOPT_HIGHS_SEARCH_CUT_LP": "1", "DISCOPT_HIGHS_SEPA_NODE": "1"}
ARMS = {
    "off": {},  # default behaviour (flags unset)
    "fixed": {"DISCOPT_HIGHS_SEARCH_CUT_LP": "0", "DISCOPT_HIGHS_SEPA_NODE": "1"},
    "noseed": {**BASE, "DISCOPT_XP_NO_SEED": "1", "DISCOPT_HIGHS_SEARCH_POOL": "adhoc"},
    "seed": {**BASE, "DISCOPT_HIGHS_SEARCH_POOL": "adhoc"},
    "h1": {**BASE, "DISCOPT_HIGHS_SEARCH_POOL": "highs"},
    "h1noseed": {**BASE, "DISCOPT_HIGHS_SEARCH_POOL": "highs", "DISCOPT_XP_NO_SEED": "1"},
}
CTRS = ["HighsSearchCutRowsAdded", "HighsSearchCutRowsDeleted", "HighsSearchPoolSeeded",
        "HighsSearchPoolSelected", "HighsSearchPoolLpRemoved", "NodeGmiRaw"]
tgt = sys.argv[1].split(","); arms = sys.argv[2].split(","); mn = sys.argv[3]
par = int(sys.argv[4]) if len(sys.argv) > 4 else 2
pop = {m["name"]: m for m in R.population() if m["name"] in tgt}
fired = 0


def mx(err, key):
    v = [int(l.split()[-1]) for l in err.splitlines() if l.strip().startswith(key + " ")]
    return max(v or [0])


def run(job):
    name, arm = job
    mem = pop[name]
    env = {k: v for k, v in os.environ.items() if not k.startswith("DISCOPT_")}
    env.update(PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="1", DISCOPT_LP_MILP_BACKEND="rust",
               DISCOPT_PROFILE="1", DISCOPT_PANEL_DETERMINISTIC="1", PYTHONPATH=f"{W}/python",
               **ARMS[arm])
    p = subprocess.run([sys.executable, f"{W}/discopt_benchmarks/scripts/lp_milp_panel.py",
                        "worker", f"{W}/{mem['path']}", "R", "600", "1e-4", mn],
                       capture_output=True, text=True, env=env, timeout=1200)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        raise RuntimeError(f"{name} {arm}: no RESULT\n{p.stderr[-3000:]}")
    r = json.loads(line[-1][7:])
    o = mem["oracle_objective"]; b = r.get("bound")
    dg = None if b is None else round(abs(o - b) / max(1, abs(o)) * 100, 4)
    bad = b is not None and b > o + 1e-6 + 1e-4 * abs(o)
    c = {k: mx(p.stderr, k) for k in CTRS}
    return (name, arm, r.get("status"), dg, r.get("nodes"), round(r.get("wall") or 0, 1),
            r.get("objective"), "bound=%r" % b, r.get("termination"), json.dumps(c), "BOUND>OPT" if bad else "")


jobs = [(n, a) for n in tgt for a in arms]
with ThreadPoolExecutor(par) as ex:
    for res in ex.map(run, jobs):
        fired += 1
        print(*res, flush=True)
print("rows", fired, flush=True)
sys.exit(0 if fired == len(jobs) else 1)
