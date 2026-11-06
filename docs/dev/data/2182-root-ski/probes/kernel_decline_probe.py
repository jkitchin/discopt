"""#2182: why does the native spatial kernel decline the small GDP lowerings?

Wraps spatial_producer.build_spatial_kernel_spec during a real Model.solve and
records, via sys.setprofile, every `return None` executed inside that module.
"""
import sys, warnings, json, linecache
sys.path.insert(0, "discopt_benchmarks")
import discopt
import discopt._relax.spatial_producer as SP
from benchmarks import pyomo_gdp_examples as px
from discopt.pyomo.gdp import convert_pyomo_gdp

assert "wt_feat" in discopt.__file__, discopt.__file__
warnings.simplefilter("ignore")
F = SP.__file__
events = []
orig = SP.build_spatial_kernel_spec

def prof(frame, ev, arg):
    if ev == "return" and arg is None and frame.f_code.co_filename == F:
        events.append((frame.f_code.co_name, frame.f_lineno,
                       linecache.getline(F, frame.f_lineno - 1).strip()[:110]))

def wrapped(*a, **k):
    sys.setprofile(prof)
    try:
        out = orig(*a, **k)
    finally:
        sys.setprofile(None)
    events.append(("<result>", 0, "spec" if out is not None else "None"))
    return out

SP.build_spatial_kernel_spec = wrapped
n = 0
for name in sys.argv[1:]:
    for m in ("big-m", "hull"):
        events.clear()
        r = convert_pyomo_gdp(px.builder(name)()).model.solve(gdp_method=m, time_limit=60)
        n += 1
        tops = [e for e in events if e[0] == "build_spatial_kernel_spec"]
        print(json.dumps({"instance": name, "method": m, "route": (r.algorithm_route or "")[:40],
                          "producer_called": any(e[0] == "<result>" for e in events),
                          "result": [e[2] for e in events if e[0] == "<result>"],
                          "top_return": tops[-1][1:] if tops else None,
                          "inner_none_returns": sorted({(e[0], e[1]) for e in events if e[0] not in ("build_spatial_kernel_spec", "<result>")})[:6]}))
print("executed:", n)
assert n > 0
