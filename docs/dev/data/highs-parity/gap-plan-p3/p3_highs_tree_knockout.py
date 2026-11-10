"""Which HiGHS component closes bell5 (and b-ball, neos17) in few nodes? HiGHS stripped to a
cut-free tree (presolve, restarts, symmetry, conflict, all separators off), then one more
component off per arm. 20 s, 1 thread, sequential. Prints rows run (§6)."""
import json, os, re, subprocess, sys
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
S = os.path.dirname(os.path.abspath(__file__)); HB = f"{S}/hsab/build/bin/highs"
names = sys.argv[1].split(",")
pop = {m["name"]: m for m in R.population()}
BASE = {"threads": "1", "time_limit": "20", "presolve": "off", "mip_allow_restart": "false",
        "mip_detect_symmetry": "false", "mip_allow_cut_separation_at_nodes": "false"}
ENV = {"HS_NO_CONFLICT": "1", "HS_NO_TABLEAU": "1", "HS_NO_PATH": "1", "HS_NO_MODK": "1", "HS_NO_IMPLB": "1", "HS_NO_CLIQUE": "1"}
HEUR = {"mip_heuristic_effort": "0", "mip_heuristic_run_feasibility_jump": "false", "mip_heuristic_run_root_reduced_cost": "false",
        "mip_heuristic_run_rens": "false", "mip_heuristic_run_rins": "false", "mip_heuristic_run_shifting": "false", "mip_heuristic_run_zi_round": "false"}
BARE = dict(HEUR, mip_pscost_minreliable="0")
ARMS = {"bare": (BARE, {}), "no_objprop": (BARE, {"HS_NO_OBJPROP": "1"}), "no_redcost": (BARE, {"HS_NO_REDCOST": "1"}),
        "no_nodeprop": (BARE, {"HS_NO_NODEPROP": "1"}), "pscost_only": (BARE, {"HS_PSCOST_ONLY": "1"}),
        "no_prop_no_rc": (BARE, {"HS_NO_NODEPROP": "1", "HS_NO_REDCOST": "1", "HS_NO_OBJPROP": "1"})}
n = 0
for name in names:
    for arm, (extra, envx) in ARMS.items():
        of = f"{S}/hsab/opt_tree_{arm}.txt"
        open(of, "w").write("".join(f"{a} = {b}\n" for a, b in {**BASE, **extra}.items()))
        env = dict(os.environ, **ENV, **envx)
        try:
            p = subprocess.run([HB, "--options_file", of, f"/home/user/discopt/{pop[name]['path']}"], capture_output=True, text=True, env=env, timeout=60)
        except subprocess.TimeoutExpired:
            print(f"{name:12} {arm:12} HUNG (>60 s with a 20 s limit)", flush=True); n += 1; continue
        g = {}
        for ln in p.stdout.splitlines():
            s = ln.strip()
            for key in ("Status", "Nodes", "Dual bound", "Primal bound", "LP iterations", "Timing"):
                if s.startswith(key) and key not in g and (key != "Nodes" or s.split()[-1].isdigit()):
                    g[key] = s[len(key):].strip()
        n += 1
        print(f"{name:12} {arm:12} status={g.get('Status')} nodes={g.get('Nodes')} lpit={g.get('LP iterations')} t={g.get('Timing')} dual={g.get('Dual bound')} primal={g.get('Primal bound')}", flush=True)
print("runs:", n); sys.exit(0 if n else 1)
