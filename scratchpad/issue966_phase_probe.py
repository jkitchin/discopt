"""#966: attribute a run's wall to phases, and catch in-flight blowups.

Instruments three seams on one ``solve_model`` run:

* ``MccormickLPRelaxer.solve_at_node``  — per-round wall vs the round's grant
  (``time_limit``), i.e. the WHOLE round including its cold build+separation;
* ``milp_relaxation.build_milp_relaxation`` — the round's non-LP build cost;
* ``MilpRelaxationModel.solve``          — the LP-solve share.

A ``faulthandler.dump_traceback_later`` watchdog fires at 2x budget and again
at 4x budget (CLAUDE.md §10: a wrapper that prints on return never fires for a
call that does not return), so a severe-mode run prints the in-flight stack.

CLAUDE.md §6: prints executed-instrument counts and exits non-zero when the
probes saw nothing.

Usage: python scratchpad/issue966_phase_probe.py <instance> <budget> <on|off>
"""

import faulthandler
import os
import sys
import time

name, budget, arm = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.environ["DISCOPT_LP_WARM_DEADLINE"] = "1" if arm == "on" else "0"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import discopt  # noqa: E402
import discopt._jax.mccormick_lp as MC  # noqa: E402
import discopt._jax.milp_relaxation as MR  # noqa: E402
from discopt._jax.deadline import deadline_scope  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solver import solve_model  # noqa: E402

assert "/home/user/discopt/python/discopt/" in discopt.__file__, discopt.__file__

node_calls: list[tuple[float, float | None]] = []  # (wall, granted time_limit)
build_calls: list[float] = []
lp_calls: list[float] = []

_orig_node = MC.MccormickLPRelaxer.solve_at_node
_orig_build = MC.build_milp_relaxation
_orig_lp = MR.MilpRelaxationModel.solve


def spy_node(self, lb, ub, time_limit=None, **kw):
    t0 = time.perf_counter()
    out = _orig_node(self, lb, ub, time_limit=time_limit, **kw)
    w = time.perf_counter() - t0
    node_calls.append((w, time_limit))
    if time_limit is not None and w > time_limit + 1.0:
        print(f"  ROUND OVERRUN: wall={w:.2f}s grant={time_limit:.2f}s", flush=True)
    return out


def spy_build(*a, **kw):
    t0 = time.perf_counter()
    out = _orig_build(*a, **kw)
    build_calls.append(time.perf_counter() - t0)
    return out


def spy_lp(self, time_limit=None, gap_tolerance=1e-4, backend="auto", *, want_marginals=False):
    t0 = time.perf_counter()
    out = _orig_lp(
        self,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        backend=backend,
        want_marginals=want_marginals,
    )
    lp_calls.append(time.perf_counter() - t0)
    return out


MC.MccormickLPRelaxer.solve_at_node = spy_node
MC.build_milp_relaxation = spy_build
MR.MilpRelaxationModel.solve = spy_lp

m = from_nl(f"python/tests/data/minlplib_nl/{name}.nl")
faulthandler.dump_traceback_later(2.0 * budget + 10.0, exit=False, file=sys.stderr)
t0 = time.perf_counter()
with deadline_scope(budget):
    r = solve_model(m, time_limit=budget)
wall = time.perf_counter() - t0
faulthandler.cancel_dump_traceback_later()

print(
    f"RUN {name} arm={arm} budget={budget}: wall={wall:.1f}s status={r.status} "
    f"bound={r.bound}\n"
    f"  rounds(solve_at_node)={len(node_calls)} round_time={sum(w for w, _ in node_calls):.1f}s\n"
    f"  builds={len(build_calls)} build_time={sum(build_calls):.1f}s\n"
    f"  lp_solves={len(lp_calls)} lp_time={sum(lp_calls):.1f}s\n"
    f"  non-node remainder={wall - sum(w for w, _ in node_calls):.1f}s",
    flush=True,
)
overruns = sum(1 for w, tl in node_calls if tl is not None and w > tl + 1.0)
print(f"executed-comparison count: {len(node_calls) + len(build_calls) + len(lp_calls)}")
print(f"round overruns (>grant+1s): {overruns}")
if not (node_calls or build_calls or lp_calls):
    print("PROBE FIRED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(2 if wall > 1.8 * budget else 0)
