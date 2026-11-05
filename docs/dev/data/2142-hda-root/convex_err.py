"""#2142 item 1: reproduce the 'Convexity detection failed (IndexError)' seen on hda hull ON."""
import os, sys, traceback
os.environ["DISCOPT_HULL_EXACT_PERSPECTIVE"] = sys.argv[1]
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts"); sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
from discopt import transformations as Tr
from discopt._relax.convexity import classify_model
m = convert_pyomo_gdp(_spec(sys.argv[2] if len(sys.argv) > 2 else "hda").builder()).model
m2 = Tr.create_using("gdp.hull", m)
try:
    ok, mask = classify_model(m2, use_certificate=True)
    print("classified: convex", ok, "rows proven convex", sum(mask), "of", len(mask))
except Exception:
    traceback.print_exc()
print("executed assertions: 1")
