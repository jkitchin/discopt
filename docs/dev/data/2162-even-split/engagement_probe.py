"""PR 2162 engagement probe: on how many instances does the root fixpoint's round-1 OBBT
use more than its even share (round-1 allotment / max_rounds), legacy arm, time_limit 120?
Only there can DISCOPT_ROOT_FIXPOINT_EVEN_SPLIT change anything.
Usage: engage2162.py REPO OUT JOBS"""
import json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
CHILD = r'''
import json, sys, time, warnings
repo, kind, name, method = sys.argv[1:5]
sys.path.insert(0, repo + "/discopt_benchmarks")
import discopt; assert discopt.__file__.startswith(repo), discopt.__file__
import discopt._relax.root_reduce as RR
assert not hasattr(RR, "_even_split_enabled"), "legacy arm must not carry the PR"
calls = []
class _Stop(BaseException): pass
orig_fp, orig_obbt = RR.run_root_fixpoint, RR._stage_obbt
cur = {}
def fp(*a, **k):
    dl = k.get("deadline"); budget = None if dl is None else dl - time.perf_counter()
    rec = {"budget": budget, "max_rounds": k.get("max_rounds", 2), "rounds": []}
    calls.append(rec); prev = dict(cur); cur.clear(); cur["rec"] = rec
    t = time.perf_counter()
    try:
        return orig_fp(*a, **k)
    finally:
        rec["wall"] = time.perf_counter() - t
        cur.clear(); cur.update(prev)
        if budget is not None and budget > 5.0:   # the outer solve's fixpoint (10% of 120 s)
            raise _Stop()
def ob(*a, **k):
    dl = k.get("deadline"); allot = None if dl is None else dl - time.perf_counter()
    t = time.perf_counter()
    out = orig_obbt(*a, **k)
    if "rec" in cur:
        cur["rec"]["rounds"].append({"allot": allot, "used": time.perf_counter() - t, "tightened": int(out[3])})
    return out
RR.run_root_fixpoint, RR._stage_obbt = fp, ob
import discopt.solver as SV
SV_names = [n for n in dir(SV) if getattr(SV, n) is orig_fp]
for n in SV_names: setattr(SV, n, fp)
warnings.simplefilter("ignore")
if kind == "nl":
    from discopt.modeling.core import from_nl
    m = from_nl(name); kw = {}
else:
    from benchmarks import gdplib_runner as gr
    from discopt.pyomo.gdp import convert_pyomo_gdp
    m = convert_pyomo_gdp(gr.discover_models(include=[name])[0].builder()).model
    kw = {"gdp_method": method}
stopped = False
t0 = time.perf_counter()
try:
    m.solve(time_limit=120, **kw)
except _Stop:
    stopped = True
print("@@" + json.dumps({"calls": calls, "stopped": stopped, "wall": time.perf_counter() - t0}))
'''
def run(repo, kind, name, method):
    load = os.getloadavg()[0]
    p = subprocess.run([sys.executable, "-c", CHILD, repo, kind, name, method], env=dict(os.environ, PYTHONPATH=repo + "/python"),
                       capture_output=True, text=True, timeout=900)
    r = next((json.loads(l[2:]) for l in p.stdout.splitlines() if l.startswith("@@")), {"error": p.stderr[-400:]})
    r.update(kind=kind, name=Path(name).stem if kind == "nl" else name, method=method, load1=round(load, 1))
    return r
def main():
    repo, out, jobs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sys.path.insert(0, repo + "/discopt_benchmarks")
    from benchmarks import gdplib_runner as gr
    cases = [("nl", str(p), "-") for p in sorted(Path(repo + "/python/tests/data/minlplib_nl").glob("*.nl"))]
    gdp = ["batch_processing","biofuel","cstr","disease_model","ex1_linan_2023","gdp_col","hda","kaibel","med_term_purchasing","methanol","mod_hens","modprodnet","positioning","pyomo.constrained_layout_clay0203","pyomo.eight_process","pyomo.eight_process_logical","pyomo.ex_633_trespalacios","pyomo.jobshop_small","pyomo.nonconvex_hen","pyomo.strip_packing_concrete","small_batch","spectralog","stranded_gas","syngas"]
    cases += [("gdp", g, mth) for g in gdp for mth in ("hull", "big-m")]
    n = 0
    with open(out, "w") as fh, ProcessPoolExecutor(jobs) as ex:
        for f in as_completed([ex.submit(run, repo, *c) for c in cases]):
            r = f.result(); n += 1; fh.write(json.dumps(r) + "\n"); fh.flush()
            outer = [c for c in r.get("calls", []) if (c.get("budget") or 0) > 5.0]
            desc = "no outer fixpoint" if not outer else " ".join(f"r{i+1}:{x['used']:.2f}/{x['allot']:.2f}s,{x['tightened']}t" for i, x in enumerate(outer[0]["rounds"]))
            print(f"[{n}/{len(cases)}] {r['name']:34s} {r['method']:5s} {desc} {r.get('error','')[:120]}", flush=True)
    print("executed", n)
if __name__ == "__main__":
    main()
