"""#945: why does GBD stop converging on test_gbd::test_linear_objective_nonlinear_constraint?

`min 3y - x0 - x1  s.t.  x0^2 + x1^2 <= 8y`, x in [0,5]^2, y binary.
y=1 -> x0=x1=2 -> objective -1.

Runs the two arms interleaved and dumps the per-iteration LB/UB trace plus the
subproblem point, so the divergence is attributed rather than guessed at.
"""

from __future__ import annotations

import inspect
import logging
import sys

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as NLPP
from discopt.decomposition.benders import solve_benders
from discopt.solvers import pounce_option_defaults

assert "opts = pounce_option_defaults()" in inspect.getsource(NLPP.solve_nlp), "not the #945 tree"

_PRE = {"print_level": 0}
_REAL = pounce_option_defaults


def build():
    m = dm.Model("linnl")
    y = m.binary("y")
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)
    return m


logging.basicConfig(level=logging.INFO, format="    %(message)s", stream=sys.stdout)
logging.getLogger("discopt.decomposition.benders.gbd").setLevel(logging.INFO)

runs = 0
for arm in ("pre", "post"):
    NLPP.pounce_option_defaults = (lambda: dict(_PRE)) if arm == "pre" else _REAL
    print(f"\n===== arm={arm} =====", flush=True)
    r = solve_benders(build(), time_limit=30)
    runs += 1
    print(
        f"  RESULT arm={arm} status={r.status} obj={r.objective!r} bound={r.bound!r} "
        f"gap={r.gap!r}",
        flush=True,
    )
NLPP.pounce_option_defaults = _REAL

print(f"\nexecuted_runs={runs}")
if runs == 0:
    sys.exit(2)
