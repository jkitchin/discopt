"""Root dual bound, HiGHS (presolve off, root only) vs discopt (flag bundle, root only), on given rows.
Gap = |opt - bound| / max(1,|opt|) %. Prints per row and the count compared (CLAUDE.md §6)."""
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; S = sys.argv[1]; rows = sys.argv[2].split(",")
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
pop = {m["name"]: m for m in R.population()}
H = f"{S}/hsab/build/bin/highs"
open(f"{S}/hsab/opt_root.txt", "w").write("threads = 1\ntime_limit = 20\npresolve = off\nmip_max_nodes = 1\n")
def highs(n):
    p = subprocess.run([H, "--options_file", f"{S}/hsab/opt_root.txt", f"{REPO}/{pop[n]['path']}"], capture_output=True, text=True, timeout=200)
    b = None; t = None
    for ln in p.stdout.splitlines():
        if ln.strip().startswith("Dual bound"): b = float(ln.split()[-1])
        if ln.strip().startswith("Timing"): t = float(ln.split()[-1])
    return b, t
def ours(n):
    env = dict(os.environ, DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_MILP_ROOT_CMIR="1", DISCOPT_MILP_RAND_ROUND="1", DISCOPT_MILP_RUN1_CONTINUE="1")
    p = subprocess.run([sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "worker", f"{REPO}/{pop[n]['path']}", "R", "20", "1e-4", "2"], capture_output=True, text=True, env=env, timeout=400)
    ln = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r = json.loads(ln[-1][7:]) if ln else {}
    return r.get("bound"), r.get("wall")
def g(n, b):
    o = pop[n]["oracle_objective"]
    return None if b is None else abs(o - b) / max(1, abs(o)) * 100
with ThreadPoolExecutor(2) as ex:
    hs = list(ex.map(highs, rows)); us = list(ex.map(ours, rows))
n = 0
print(f"{'row':22} {'HiGHS root gap':>15} {'t':>6} {'discopt root gap':>17} {'t':>6}")
for r, (hb, ht), (ub, ut) in zip(rows, hs, us):
    hg, ug = g(r, hb), g(r, ub); n += hg is not None and ug is not None
    f = lambda v: "-" if v is None else f"{v:.2f}%"
    print(f"{r:22} {f(hg):>15} {ht or 0:6.1f} {f(ug):>17} {ut or 0:6.1f}")
print("compared:", n); sys.exit(0 if n else 1)
