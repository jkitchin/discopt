"""Validity check for the strengthening probe: every master point of the OFF run
satisfies all original cuts; does it satisfy their strengthened versions?"""
import sys, warnings
import numpy as np
sys.path.insert(0, "docs/dev/data/1917-loa-cut-strengthening")
warnings.simplefilter("ignore")
sys.argv = ["x", "", "60"]
import importlib.util
spec = importlib.util.spec_from_file_location("ls", "docs/dev/data/1917-loa-cut-strengthening/loa_strengthen.py")
src = open(spec.origin).read().split("names = sys.argv")[0]
g = {"__name__": "ls"}; exec(src, g)
from discopt.gdp.algorithms import master as master_mod, loa as loa_mod
pts, pairs, box = [], [], {}
orig_solve = master_mod.MasterMILP.solve
def solve(self, *a, **k):
    r = orig_solve(self, *a, **k)
    if getattr(r, "x", None) is not None: pts.append(np.asarray(r.x, float)[: self.n_vars])
    return r
master_mod.MasterMILP.solve = solve
orig = g["orig_cuts_at"]
def cuts_at(self, x):
    k = len(self.master.cuts); orig(self, x)
    bins = [i for i in self.master.int_indices if self.lb_box[i] == 0 and self.ub_box[i] == 1]
    box["lb"], box["ub"] = self.lb_box, self.ub_box
    for i in range(k, len(self.master.cuts)):
        row = self.master.cuts[i]
        if row.shape != (self.n,): continue
        new, rhs, n = g["strengthen"](row, self.master.cut_rhs[i], self.lb_box, self.ub_box, bins)
        if n: pairs.append((row.copy(), self.master.cut_rhs[i], new, rhs))
loa_mod.LOA._cuts_at = cuts_at
from discopt.pyomo.gdp import convert_pyomo_gdp
from benchmarks import gdplib_runner as gr
conv = convert_pyomo_gdp(gr.discover_models(include=["pyomo.eight_process"])[0].builder())
r = conv.model.solve(gdp_method="loa", time_limit=60)
print("OFF run:", r.status, r.objective, "master points", len(pts), "strengthened pairs", len(pairs))
checks = bad = 0
for p in pts:
    inbox = np.all(p >= box["lb"] - 1e-7) and np.all(p <= box["ub"] + 1e-7)
    for a, b, a2, b2 in pairs:
        checks += 1
        if a @ p <= b + 1e-6 and a2 @ p > b2 + 1e-6:
            bad += 1
            if bad <= 3:
                print("  INVALID: orig slack", b - a @ p, "new slack", b2 - a2 @ p, "point in box:", inbox)
print("checks", checks, "invalidated", bad); sys.exit(0 if checks else 3)
