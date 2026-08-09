"""Issue #928: hda whole-solve at time_limit=10, one arm per process.

Usage: python issue928_hda10.py <0|1>   (DISCOPT_LP_WARM_DEADLINE value)
"""

import os
import sys
import time

os.environ["DISCOPT_LP_WARM_DEADLINE"] = sys.argv[1]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt._jax.deadline import deadline_scope  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solver import solve_model  # noqa: E402

assert "/home/user/discopt/python/discopt/" in discopt.__file__, discopt.__file__

budget = 10.0
m = from_nl("python/tests/data/minlplib/hda.nl")
t0 = time.perf_counter()
with deadline_scope(budget):
    r = solve_model(m, time_limit=budget)
wall = time.perf_counter() - t0
print(f"flag={sys.argv[1]} wall={wall:6.2f}s bound={r.bound} status={r.status}")
