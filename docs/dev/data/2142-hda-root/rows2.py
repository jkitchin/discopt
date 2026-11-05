"""Print hda's degree-2 homogenized rows with their variables' disjunct boxes."""
import os, sys
os.environ["DISCOPT_HULL_EXACT_PERSPECTIVE"] = "1"
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts"); sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
import discopt._relax.gdp_reformulate as G
from discopt import transformations as Tr
seen = []
orig = G._perspective_mccormick_rows
def spy(con, y, copies, bounds_k, *a):
    mono = G._polynomial_monomials(con.body)
    seen.append((con.name, {k: round(c, 4) for k, c in mono.items()}, {n: bounds_k.get(n) for key in mono for n in key}))
    return orig(con, y, copies, bounds_k, *a)
G._perspective_mccormick_rows = spy
m = convert_pyomo_gdp(_spec("hda").builder()).model
Tr.create_using("gdp.hull", m)
for s in seen[:25]: print(s)
print("executed assertions:", 1 if seen else 0)
