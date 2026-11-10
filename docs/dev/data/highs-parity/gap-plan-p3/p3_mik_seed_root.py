"""Entry experiment (gap plan §12.27): does an incumbent at the start of discopt's root loop,
with root reduced-cost fixing + propagation, close the mik root gap the way it does in HiGHS?
Root only (max_nodes=2). Arms: base / seed / seed+prop. Prints root gap % and whether the seed
was accepted; exits 1 if a seeded arm's seed was not accepted or nothing ran (§6)."""
import json, os, subprocess, sys
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
S = os.path.dirname(os.path.abspath(__file__)); pop = {m["name"]: m for m in R.population()}
B = {"DISCOPT_MILP_RAND_ROUND": "1", "DISCOPT_MILP_RUN1_CONTINUE": "1", "DISCOPT_MILP_ROOT_CMIR": "highs", "DISCOPT_LP_MILP_BACKEND": "rust"}
rows = sys.argv[1].split(","); n = 0; bad = 0
print(f"{'row':18}{'base':>10}{'seed':>10}{'seed+prop':>11}")
for name in rows:
    cells = []
    for arm in ("base", "seed", "seed+prop"):
        env = {a: b for a, b in os.environ.items() if not a.startswith("DISCOPT_")}; env.update(B)
        if arm != "base": env["DISCOPT_MILP_SEED_POINT"] = f"{S}/mikseed/{name}.0"
        env["DISCOPT_HIGHS_SEPA_PROPAGATE"] = "1" if arm == "seed+prop" else "0"
        p = subprocess.run([sys.executable, "/home/user/discopt/discopt_benchmarks/scripts/lp_milp_panel.py", "worker",
                            f"/home/user/discopt/{pop[name]['path']}", "R", "60", "1e-4", "2"], capture_output=True, text=True, env=env)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else {}
        acc = "DISCOPT_MILP_SEED_POINT accepted" in p.stderr
        if arm != "base" and not acc: bad += 1
        opt = pop[name]["oracle_objective"]; b = r.get("bound")
        n += 1; cells.append("-" if b is None else f"{abs(opt - b) / max(1, abs(opt)) * 100:.2f}%{'' if arm == 'base' or acc else '!'}")
    print(f"{name:18}" + "".join(f"{c:>10}" for c in cells), flush=True)
print("runs:", n, "seeds not accepted:", bad); sys.exit(0 if n and not bad else 1)
