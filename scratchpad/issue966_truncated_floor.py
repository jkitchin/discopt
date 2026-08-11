"""#966 entry experiment 2: can a round that cannot afford its build still bank
a floor?

H1 (falsified, see issue966_inc_scope.py): the cheap tier a declined round could
fall back to is the incremental fast path.  Measured: ``_inc`` is unavailable on
all five instances the round-budget flag hurts (nvs05, tspn10, casctanks,
contvar, tls2), so that tier does not exist for this class.

H2 (this probe): the #694 anytime build already IS the cheap tier.  A build whose
``build_deadline`` is already spent emits zero constraint rows but still
linearizes the objective, so it yields (a) a valid, weaker LP relaxation and (b)
``milp._objective_floor`` -- the rigorous box-interval objective floor the node
solver falls back to.  If a fully-truncated round is much cheaper than a full
round AND still reports a finite bound, then "decline the round" (bank nothing)
can be replaced by "yield the round" (bank a floor), which is what #966's
remaining item asks for.

Kill criterion: a fully-truncated round costing more than 25% of the full round's
wall on these instances means truncation is not a cheap floor producer, and the
decline branch has to stay as it is.

Counted output; exits non-zero if nothing was compared (CLAUDE.md §6).
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
from discopt._relax import mccormick_lp as mc_mod  # noqa: E402
from discopt._relax.mccormick_lp import MccormickLPRelaxer  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

SEEN: list = []
_orig_build = mc_mod.build_milp_relaxation


def _spy(*a, **kw):
    t0 = time.perf_counter()
    out = _orig_build(*a, **kw)
    wall = time.perf_counter() - t0
    milp = out[0] if isinstance(out, tuple) else out
    inner = getattr(milp, "model", milp)
    SEEN.append(
        {
            "wall": round(wall, 4),
            "truncated": bool(getattr(inner, "_build_truncated", False)),
            "rows_done": getattr(inner, "_build_constraints_done", None),
            "rows_total": getattr(inner, "_build_constraints_total", None),
            "objective_floor": getattr(inner, "_objective_floor", None),
        }
    )
    return out


mc_mod.build_milp_relaxation = _spy

REPS = int(os.environ.get("PROBE_REPS", "2"))
compared = 0
for name in sys.argv[1:]:
    m = from_nl(f"python/tests/data/minlplib_nl/{name}.nl")
    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)

    for rep in range(REPS):
        SEEN.clear()
        t0 = time.perf_counter()
        full = relaxer.solve_at_node(np.asarray(lb), np.asarray(ub), time_limit=30.0)
        full_wall = time.perf_counter() - t0
        full_builds = list(SEEN)

        SEEN.clear()
        t0 = time.perf_counter()
        cut = relaxer.solve_at_node(
            np.asarray(lb),
            np.asarray(ub),
            time_limit=30.0,
            build_deadline=time.perf_counter(),  # already spent: full truncation
        )
        cut_wall = time.perf_counter() - t0
        cut_builds = list(SEEN)
        compared += 1

        print(
            f"{name:12s} rep{rep} FULL round={full_wall:7.3f}s status={full.status:12s} "
            f"lb={full.lower_bound} x={full.x is not None}\n"
            f"{'':12s}      CUT  round={cut_wall:7.3f}s status={cut.status:12s} "
            f"lb={cut.lower_bound} x={cut.x is not None} ratio={cut_wall / full_wall:.3f}\n"
            f"{'':12s}      full builds={full_builds}\n"
            f"{'':12s}      cut  builds={cut_builds}",
            flush=True,
        )

print(f"INSTANCES_COMPARED={compared}")
if compared == 0:
    sys.exit(2)
