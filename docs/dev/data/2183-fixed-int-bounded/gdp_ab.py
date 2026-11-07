"""#2183: gdp_parity big-M/hull cells, old vs new arm, interleaved, in-process.
old = _fixed_integer_lp_cannot_refute patched to False (exactly the pre-#2183 path)."""
from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[4])
import json, os, sys, time, warnings
import discopt
assert discopt.__file__.startswith(REPO + "/python"), discopt.__file__
from discopt.solvers import lp_milp_highs as L
sys.path.insert(0, REPO + "/discopt_benchmarks"); sys.path.insert(0, REPO + "/discopt_benchmarks/scripts")
from gdp_baseline import _spec
from benchmarks.pyomo_gdp_examples import builder as pbuilder
from discopt.pyomo.gdp import convert_pyomo_gdp
warnings.simplefilter("ignore")
REAL = L._fixed_integer_lp_cannot_refute
fired = {"true": 0, "calls": 0}
def spy(*a, **k):
    r = REAL(*a, **k); fired["calls"] += 1; fired["true"] += bool(r); return r
milp_calls = {"n": 0}
orig_milp = L.solve_milp_std
def milp_spy(*a, **k):
    milp_calls["n"] += 1; return orig_milp(*a, **k)
L.solve_milp_std = milp_spy
GDPLIB = ['biofuel', 'gdp_col', 'pyomo.constrained_layout_clay0203', 'pyomo.eight_process',
          'pyomo.eight_process_logical', 'pyomo.ex_633_trespalacios', 'pyomo.jobshop_small',
          'pyomo.nonconvex_hen', 'pyomo.strip_packing_concrete', 'stranded_gas', 'syngas']
VENDORED = ['pyomo.batch_processing', 'pyomo.disease_model', 'pyomo.med_term_purchasing']
def build(name):
    return pbuilder(name)() if name in VENDORED else _spec(name).builder()
excluded = {}
for name in list(GDPLIB):
    try:
        convert_pyomo_gdp(build(name))
    except Exception as e:  # screening only; recorded and printed, never silent
        excluded[name] = f"{type(e).__name__}: {e}"[:160]
        GDPLIB.remove(name)
print("EXCLUDED (builder fails on this gdplib install):", json.dumps(excluded, indent=1), flush=True)
TL = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
rows = []; cmp = 0
for name in GDPLIB + VENDORED:
    for method in ("big-m", "hull"):
        row = {"instance": name, "method": method}
        for arm in ("old", "new"):
            L._fixed_integer_lp_cannot_refute = (lambda *a, **k: False) if arm == "old" else spy
            f0, m0 = fired["true"], milp_calls["n"]
            mdl = build(name)
            conv = convert_pyomo_gdp(mdl)
            t = time.perf_counter()
            try:
                r = conv.model.solve(gdp_method=method, time_limit=TL)
            except ValueError as e:  # a loud reformulation refusal: recorded per cell, both arms
                row[arm] = dict(status="refused", cert=False, obj=None, bound=None, nodes=0,
                                wall=0.0, milp_route_calls=milp_calls["n"] - m0,
                                bounded_fired=fired["true"] - f0, route=str(e)[:160])
                cmp += 1
                continue
            row[arm] = dict(status=r.status, cert=bool(r.gap_certified), obj=r.objective, bound=r.bound,
                            nodes=r.node_count, wall=round(time.perf_counter() - t, 2),
                            milp_route_calls=milp_calls["n"] - m0, bounded_fired=fired["true"] - f0,
                            route=(r.algorithm_route or "")[:120])
            cmp += 1
        L._fixed_integer_lp_cannot_refute = REAL
        rows.append(row)
        o, n = row["old"], row["new"]
        print(f"{name:40s} {method:6s} old={o['status']}/{o['cert']}/{o['obj']}/{o['bound']} "
              f"new={n['status']}/{n['cert']}/{n['obj']}/{n['bound']} milp_calls={n['milp_route_calls']} "
              f"fired={n['bounded_fired']}", flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
print(f"SUMMARY cells={len(rows)} comparisons={cmp} cert old={sum(r['old']['cert'] for r in rows)} "
      f"new={sum(r['new']['cert'] for r in rows)} fired_cells={sum(r['new']['bounded_fired']>0 for r in rows)}")
sys.exit(0 if cmp else 1)
