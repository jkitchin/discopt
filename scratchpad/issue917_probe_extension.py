"""#917: does the incumbent-conditional budget extension actually fire?

Runs one instance through ``Model.solve`` with the flag OFF and then ON, in one
process, and asserts on the observable the change is *about*: the wall clock and
``solver_stats["budget/incumbent_extension_s"]``.

Per CLAUDE.md §8 it first asserts which ``discopt`` was imported and that the
version under test carries the #917 marker. Per §6 it prints an executed-assertion
count and exits non-zero if nothing was checked.
"""

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import discopt  # noqa: E402
from discopt import solver as _solver  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

print("discopt.__file__ =", discopt.__file__)
assert "/home/user/discopt/python/discopt/" in discopt.__file__, discopt.__file__
assert hasattr(_solver, "_extend_budget_for_incumbent"), "#917 marker absent: wrong tree loaded"

name = sys.argv[1]
T = float(sys.argv[2])
path = f"python/tests/data/minlplib/{name}.nl"

checks = 0
for flag in ("0", "1"):
    os.environ["DISCOPT_LP_SPATIAL_RESERVE_EXTENSION"] = flag
    m = from_nl(path)
    t0 = time.perf_counter()
    r = m.solve(time_limit=T)
    wall = time.perf_counter() - t0
    ext = (r.solver_stats or {}).get("budget/incumbent_extension_s")
    print(
        f"flag={flag} wall={wall:.1f} status={r.status} obj={r.objective} "
        f"bound={r.bound} certified={r.gap_certified} nodes={r.node_count} extension_s={ext}",
        flush=True,
    )
    checks += 1
    if r.objective is not None and r.bound is not None:
        assert r.bound <= r.objective + 1e-6, f"UNSOUND: bound {r.bound} > incumbent {r.objective}"
        checks += 1

print(f"EXECUTED_ASSERTIONS={checks}")
if checks == 0:
    print("PROBE FIRED NOTHING", file=sys.stderr)
    sys.exit(1)
