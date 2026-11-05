"""#2142 item 1: list the disjunct rows the homogenized hull rewrites on an instance."""
import os, sys
os.environ["DISCOPT_HULL_EXACT_PERSPECTIVE"] = "1"
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts"); sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
import discopt._relax.gdp_reformulate as G
from discopt import transformations as Tr
name = sys.argv[1]
seen = []
orig = G._homogenized_hull_row
def spy(con, y, copies, model):
    out = orig(con, y, copies, model)
    mono = G._polynomial_monomials(con.body)
    deg = max((len(k) for k in mono), default=0) if mono else None
    seen.append((con.name, con.sense, deg, len(mono) if mono else None, out is not None, str(con.body)[:160], con.rhs))
    return out
G._homogenized_hull_row = spy
m = convert_pyomo_gdp(_spec(name).builder()).model
Tr.create_using("gdp.hull", m) if hasattr(Tr, "create_using") else None
from collections import Counter
print(Counter((s[2], s[4]) for s in seen))
for s in seen[:60]: print(s)
print("executed assertions:", 1 if seen else 0)
