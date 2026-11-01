"""#1917 G4.1 entry experiment: does binary coefficient strengthening of LOA's OA cuts
(Pyomo's per-cut big-M) change the master? Arms interleaved per instance."""
import json, sys, time, warnings
import numpy as np
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
warnings.simplefilter("ignore")
import discopt
assert discopt.__file__.startswith("/home/user/discopt/python/")
from discopt.gdp.algorithms import loa as loa_mod
from discopt.pyomo.gdp import convert_pyomo_gdp
from benchmarks import gdplib_runner as gr

STATS = {"cuts_seen": 0, "coefs_tightened": 0}
orig_cuts_at = loa_mod.LOA._cuts_at

def strengthen(row, rhs, lb, ub, binaries):
    row = row.copy(); n = 0
    for j in binaries:
        b = row[j]
        if b == 0.0: continue
        rest = row.copy(); rest[j] = 0.0
        hi = np.where(rest > 0, rest * ub, rest * lb)
        hi = np.where(rest == 0, 0.0, hi)
        if not np.all(np.isfinite(hi)): continue
        U = float(np.sum(hi)); U += 1e-9 * (1 + abs(U))  # outward
        if b > 0 and U < rhs:
            d = rhs - U; row[j] = b - d; rhs = rhs - d; n += 1
        elif b < 0 and U < rhs - b:
            d = (rhs - b) - U; row[j] = b + d; n += 1
    return row, rhs, n

def patched(self, x):
    k = len(self.master.cuts)
    orig_cuts_at(self, x)
    if not MODE["on"]: 
        STATS["cuts_seen"] += len(self.master.cuts) - k; return
    binaries = [i for i in self.master.int_indices if self.lb_box[i] == 0 and self.ub_box[i] == 1]
    for i in range(k, len(self.master.cuts)):
        row = self.master.cuts[i]
        if row.shape != (self.n,): continue  # epigraph cuts left alone
        STATS["cuts_seen"] += 1
        new, rhs, n = strengthen(row, self.master.cut_rhs[i], self.lb_box, self.ub_box, binaries)
        self.master.cuts[i] = new; self.master.cut_rhs[i] = rhs; STATS["coefs_tightened"] += n

MODE = {"on": False}
loa_mod.LOA._cuts_at = patched
names = sys.argv[1].split(","); tl = float(sys.argv[2])
for name in names:
    for on in (False, True) if hash(name) % 2 else (True, False):
        MODE["on"] = on; STATS.update(cuts_seen=0, coefs_tightened=0)
        conv = convert_pyomo_gdp(gr.discover_models(include=[name])[0].builder())
        t = time.perf_counter()
        try:
            r = conv.model.solve(gdp_method="loa", time_limit=tl)
            s = r.solver_stats or {}
            out = dict(status=r.status, obj=r.objective, bound=r.bound, it=s.get("iterations"), nmaster=s.get("n_master"), nsub=s.get("n_subproblems"))
        except Exception as e:
            out = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
        out.update(instance=name, on=on, wall=round(time.perf_counter() - t, 2), **STATS)
        print(json.dumps(out), flush=True)
print("probe done; rows:", 2 * len(names))
