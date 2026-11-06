"""Report for the #2170 panel: old rule vs new gate, paired per (instance, method)."""
import json, math, sys
from pathlib import Path
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
from benchmarks import gdplib_runner as gr
import gdp_baseline as gb
oracle = gr.panel_oracle()
rows = [json.loads(l) for p in sys.argv[1:] for l in Path(p).read_text().splitlines() if l.strip()]
bad_marker = [r for r in rows if not r.get("marker_ok")]
cells = {}
for r in rows:
    cells.setdefault((r["instance"], r["method"]), {})[r["panel_arm"]] = r
n_checks = incorrect = crossings = 0
solved = {"old": 0, "new": 0}
lines = []
for (n, m), arms in sorted(cells.items()):
    if set(arms) != {"old", "new"}:
        lines.append(f"INCOMPLETE {n} {m}"); continue
    o = oracle.get(n)
    ref = getattr(o, "value", None) if o is not None else None
    for a in ("old", "new"):
        r = arms[a]
        if r.get("status") == "optimal":
            solved[a] += 1
        obj, bnd, mn = r.get("objective"), r.get("bound"), r.get("minimize", True)
        if obj is not None and r.get("in_pyomo") is not None:
            n_checks += 1
            if not gb.verified_feasible(r["in_pyomo"]):
                incorrect += 1; lines.append(f"UNVERIFIED POINT {n} {m} {a}")
        if ref is not None and bnd is not None and math.isfinite(ref):
            tol = 1e-6 + 1e-4 * abs(ref)
            if (mn and bnd > ref + tol) or ((not mn) and bnd < ref - tol):
                crossings += 1; lines.append(f"BOUND CROSSES ORACLE {n} {m} {a} bound={bnd} ref={ref}")
        if r.get("status") == "optimal" and ref is not None and obj is not None:
            if abs(obj - ref) > 1e-6 + 1e-4 * abs(ref):
                incorrect += 1; lines.append(f"WRONG OPTIMUM {n} {m} {a} obj={obj} ref={ref}")
    ro, rn = arms["old"], arms["new"]
    diff = (ro.get("status") != rn.get("status")) or (
        (ro.get("objective") is None) != (rn.get("objective") is None)) or (
        ro.get("objective") is not None and rn.get("objective") is not None
        and abs(ro["objective"] - rn["objective"]) > 1e-6 + 1e-4 * abs(ro["objective"]))
    gate_old, gate_new = ro.get("gate_log") or [], rn.get("gate_log") or []
    fired = any("skipped" in g for g in gate_old) or any("runs, the incumbent" in g for g in gate_new)
    if diff or fired:
        lines.append(f"{'DIFF ' if diff else 'same '}{n:22s} {m:6s} old: {ro.get('status')} obj={ro.get('objective')} "
                     f"bnd={ro.get('bound')} nodes={ro.get('node_count')} | new: {rn.get('status')} obj={rn.get('objective')} "
                     f"bnd={rn.get('bound')} nodes={rn.get('node_count')}\n      gate old={gate_old}\n      gate new={gate_new}")
print(f"rows={len(rows)} cells={len(cells)} marker_mismatch={len(bad_marker)}")
print(f"incumbent checks={n_checks} incorrect={incorrect} bound_crossings={crossings} solved old={solved['old']} new={solved['new']}")
print("\n".join(lines))
assert n_checks > 0, "no incumbent was checked"
