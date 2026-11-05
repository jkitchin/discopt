"""hda hull with discopt INFO logging (route identification)."""
import logging, os, sys
logging.basicConfig(level=logging.INFO, format="%(relativeCreated)8.0f %(name)s %(message)s")
os.environ["DISCOPT_HULL_EXACT_PERSPECTIVE"] = sys.argv[1]
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts"); sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
m = convert_pyomo_gdp(_spec("hda").builder()).model
r = m.solve(gdp_method="hull", deterministic=True, max_nodes=5000, time_limit=120)
print("RESULT", r.status, r.bound, r.node_count)
