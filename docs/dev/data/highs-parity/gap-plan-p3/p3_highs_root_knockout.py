"""mik root gap: HiGHS root only (mip_max_nodes=1), stripped (presolve, restarts, symmetry,
conflict off), one more component off per arm. Root gap % vs the oracle. Prints runs (§6)."""
import os, re, subprocess, sys
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
S = os.path.dirname(os.path.abspath(__file__)); HB = f"{S}/hsab/build/bin/highs"
pop = {m["name"]: m for m in R.population()}
names = sys.argv[1].split(",")
BASE = {"threads": "1", "time_limit": "60", "presolve": "off", "mip_allow_restart": "false",
        "mip_detect_symmetry": "false", "mip_max_nodes": "1"}
HEUR = {"mip_heuristic_effort": "0", "mip_heuristic_run_feasibility_jump": "false", "mip_heuristic_run_root_reduced_cost": "false",
        "mip_heuristic_run_rens": "false", "mip_heuristic_run_rins": "false", "mip_heuristic_run_shifting": "false", "mip_heuristic_run_zi_round": "false"}
ARMS = {"stripped": ({}, {}), "no_heur": (HEUR, {}), "no_root_rc": ({}, {"HS_NO_ROOT_REDCOST": "1"}),
        "no_tableau": ({}, {"HS_NO_TABLEAU": "1"}), "no_path": ({}, {"HS_NO_PATH": "1"}), "no_modk": ({}, {"HS_NO_MODK": "1"}),
        "no_implb": ({}, {"HS_NO_IMPLB": "1"}), "no_clique": ({}, {"HS_NO_CLIQUE": "1"}),
        "no_heur_no_rc": (HEUR, {"HS_NO_ROOT_REDCOST": "1"})}
ROW = re.compile(r"^\s*([A-Za-z])?\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\S+)\s+(\S+)")
n = 0
print(f"{'row':18}" + "".join(f"{a:>14}" for a in ARMS))
for name in names:
    opt = pop[name]["oracle_objective"]; cells = []
    for arm, (o, e) in ARMS.items():
        of = f"{S}/hsab/opt_root_{arm}.txt"
        open(of, "w").write("".join(f"{a} = {b}\n" for a, b in {**BASE, **o}.items()))
        env = {a: b for a, b in os.environ.items() if not a.startswith("HS_")}; env.update({"HS_NO_CONFLICT": "1", **e})
        p = subprocess.run([HB, "--options_file", of, f"/home/user/discopt/{pop[name]['path']}"], capture_output=True, text=True, env=env, timeout=120)
        rb = None
        for ln in p.stdout.splitlines():
            m = ROW.match(ln)
            if m and int(m.group(2)) == 0:
                try: rb = float(m.group(6))
                except ValueError: pass
        n += 1
        cells.append("-" if rb is None else f"{abs(opt - rb) / max(1, abs(opt)) * 100:.2f}%")
    print(f"{name:18}" + "".join(f"{c:>14}" for c in cells), flush=True)
print("runs:", n); sys.exit(0 if n else 1)
