import sys

sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
import discopt.solver as S
from benchmarks import gdplib_runner as gr
from discopt.pyomo.gdp import convert_pyomo_gdp
from discopt.transformations import get

pm = gr.discover_models(include=["batch_processing"])[0].builder()
conv = convert_pyomo_gdp(pm)
m = conv.model
print("source:", S._convex_minlp_auto_route(m)[:2])
for meth in ("big-m", "hull"):
    w = get("gdp").apply(m.clone() if hasattr(m, "clone") else m, method=meth)
    print(meth, S._convex_minlp_auto_route(w)[:2])
