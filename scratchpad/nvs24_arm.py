"""One nvs24 primary solve under a chosen instrumentation arm.

Every instrument tried so far seemed to flip nvs24 out of its ~52 s slow mode, but
that was never measured interleaved or repeated. This runs ONE arm per process so
the arms can be interleaved by the driver (CLAUDE.md §9).

Arms:
  plain    -- no wrappers at all
  warm     -- wrap solvers.milp_simplex.solve_lp_warm_std (a timing no-op)
  usq      -- wrap MccormickLPRelaxer._separate_univariate_square
  relax    -- wrap MilpRelaxationModel.solve

Usage: python nvs24_arm.py <arm> [time_limit]
"""

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt._jax.deadline import deadline_scope  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solver import solve_model  # noqa: E402

assert "/python/discopt/" in discopt.__file__, discopt.__file__

arm = sys.argv[1]
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 3.9
wrapped = 0


def _wrap(owner, name):
    orig = getattr(owner, name)

    def spy(*a, **kw):
        global wrapped
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        _ = time.perf_counter() - t0
        wrapped += 1
        return out

    setattr(owner, name, spy)


if arm == "warm":
    import discopt.solvers.milp_simplex as MS

    _wrap(MS, "solve_lp_warm_std")
elif arm == "usq":
    import discopt._jax.mccormick_lp as M

    _wrap(M.MccormickLPRelaxer, "_separate_univariate_square")
elif arm == "relax":
    import discopt._jax.milp_relaxation as MR

    _wrap(MR.MilpRelaxationModel, "solve")
elif arm != "plain":
    raise SystemExit(f"unknown arm {arm}")

m = from_nl("python/tests/data/minlplib/nvs24.nl")
t0 = time.perf_counter()
with deadline_scope(budget):
    r = solve_model(m, time_limit=budget)
wall = time.perf_counter() - t0
usq = (r.solver_stats or {}).get("separate/univariate_square")
print(
    f"arm={arm:6s} wall={wall:6.1f} ({wall / budget:5.1f}x) nodes={r.node_count} "
    f"bound={r.bound} usq={None if usq is None else round(usq, 1)} wrapped_calls={wrapped}"
)
if arm != "plain" and wrapped == 0:
    print("PROBE FIRED NOTHING: wrapper never called", file=sys.stderr)
    sys.exit(1)
