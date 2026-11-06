"""#2182 task 2 entry experiment: where does a small big-m / hull solve spend its wall?

Usage: python -u phase_probe.py <instance> <method> [--profile out.prof]

Prints, in one fresh process: import, build, convert, solve #1 (cold), and
convert+solve #2 (warm: same process, fresh conversion). cold - warm is the
one-time cost (lazy imports, first-call warm-up); warm is the per-solve cost.
"""

import cProfile
import json
import sys
import time
import warnings

t0 = time.perf_counter()
import discopt  # noqa: E402

t_import = time.perf_counter() - t0
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
from benchmarks import pyomo_gdp_examples as px  # noqa: E402

from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402

name, method = sys.argv[1], sys.argv[2]


class _S:
    def __init__(self, b):
        self.builder = b


def _spec(n):
    if n.startswith(px.PREFIX):
        return _S(px.builder(n))
    from gdp_baseline import _spec as s

    return s(n)
prof_out = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--profile" else None

assert "/home/user/discopt/python/discopt" in discopt.__file__, discopt.__file__
n_jax_before = sum(1 for m in sys.modules if m == "jax" or m.startswith("jax."))


def one(profile=None):
    tb = time.perf_counter()
    pm = _spec(name).builder()
    tc = time.perf_counter()
    conv = convert_pyomo_gdp(pm)
    ts = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if profile is not None:
            profile.enable()
        r = conv.model.solve(gdp_method=method, time_limit=300)
        if profile is not None:
            profile.disable()
    te = time.perf_counter()
    return {
        "build": tc - tb,
        "convert": ts - tc,
        "solve": te - ts,
        "status": r.status,
        "obj": r.objective,
        "nodes": getattr(r, "node_count", None),
    }


prof = cProfile.Profile() if prof_out else None
cold = one(prof)
warm = one()
if prof_out:
    prof.dump_stats(prof_out)
n_jax = sum(1 for m in sys.modules if m == "jax" or m.startswith("jax."))
row = {
    "instance": name,
    "method": method,
    "import": round(t_import, 3),
    "cold": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cold.items()},
    "warm": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in warm.items()},
    "jax_modules": n_jax - n_jax_before,
}
print(json.dumps(row), flush=True)
assert cold["status"] == warm["status"], "cold/warm disagree"
print("executed-assertions: 2", flush=True)
