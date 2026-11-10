"""Soundness re-score of an A/B jsonl with each instance's sense from the panel loader.
Checks: certified objective == oracle; dual bound never past the oracle; objective never better
than the oracle; no infeasible verdict. Prints the check count (CLAUDE.md §6); exit 1 on any violation."""
import json, sys
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R, lp_milp_loader as L
pop = {m["name"]: m for m in R.population()}
rows = [json.loads(l) for l in open(sys.argv[1])]
sense = {}
viol = []; checks = 0
for r in rows:
    n = r["name"]
    if n not in sense:
        sense[n] = -1 if L.read_instance("/home/user/discopt/" + pop[n]["path"]).maximize else 1
    sg, opt = sense[n], r["opt"]
    if opt is None or r["status"] in ("ERR", None): continue
    tol = 1e-6 + 1e-4 * abs(opt)
    if r["status"] == "optimal" and r["gap_certified"]:
        checks += 1
        if r["objective"] is None or abs(r["objective"] - opt) > tol: viol.append((n, r["arm"], "cert obj", r["objective"], opt))
    if r["bound"] is not None:
        checks += 1
        if sg * (r["bound"] - opt) > tol: viol.append((n, r["arm"], "bound past oracle", r["bound"], opt))
    if r["objective"] is not None:
        checks += 1
        if sg * (opt - r["objective"]) > tol: viol.append((n, r["arm"], "objective better than oracle", r["objective"], opt))
    if r["status"] == "infeasible": checks += 1; viol.append((n, r["arm"], "infeasible verdict", None, opt))
print("maximisation:", sorted(k for k, v in sense.items() if v < 0))
print("checks:", checks, "violations:", len(viol))
for v in viol: print("  VIOLATION", v)
sys.exit(1 if viol or not checks else 0)
