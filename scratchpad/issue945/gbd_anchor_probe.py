"""#945 / GBD: dump the ACTUAL (anchor, s) each recourse call produces.

A paper derivation of these numbers is not evidence. ``_recourse`` is a closure
inside ``solve_gbd`` with no injection seam, so this traps its real return value
with a return-event trace hook rather than reimplementing its arithmetic (a
reimplementation can diverge from the code it claims to measure).

Both arms are run; the per-iteration cut sequence is printed for each.
"""

from __future__ import annotations

import inspect
import sys

import numpy as np

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as NLPP
from discopt.decomposition.benders import solve_benders
from discopt.solvers import pounce_option_defaults

assert "opts = pounce_option_defaults()" in inspect.getsource(NLPP.solve_nlp), "not the #945 tree"

_PRE = {"print_level": 0}
_REAL = pounce_option_defaults

records: list[tuple] = []
_ARM = "pre"


def _tracer(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "_recourse":
        return _local
    return None


def _local(frame, event, arg):
    if event == "return" and isinstance(arg, tuple) and len(arg) == 6:
        kind, v, x_full, anchor, s, rig = arg
        records.append((_ARM, kind, float(v), np.asarray(x_full).copy(),
                        float(anchor), np.asarray(s).copy(), bool(rig)))
    return None


def build():
    m = dm.Model("linnl")
    y = m.binary("y")
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)
    return m


for arm in ("pre", "post"):
    _ARM = arm
    NLPP.pounce_option_defaults = (lambda: dict(_PRE)) if arm == "pre" else _REAL
    sys.settrace(_tracer)
    try:
        r = solve_benders(build(), time_limit=30)
    finally:
        sys.settrace(None)
    print(f"=== arm={arm}: status={r.status} obj={r.objective!r} bound={r.bound!r}", flush=True)
NLPP.pounce_option_defaults = _REAL

print(f"\n{'arm':5s} {'kind':6s} {'v':>14s} {'anchor':>16s} {'s (master cols)':>22s} {'rig':>5s}")
print("-" * 74)
shown: set = set()
for arm, kind, v, x_full, anchor, s, rig in records:
    key = (arm, kind, round(v, 12), round(anchor, 12))
    if key in shown:
        continue
    shown.add(key)
    print(f"{arm:5s} {kind:6s} {v:14.6e} {anchor:16.6e} "
          f"{np.array2string(s, precision=4):>22s} {rig!s:>5s}")

print(f"\nexecuted_recourse_returns={len(records)}  distinct_shown={len(shown)}")
if not records:
    print("PROBE TRAPPED NOTHING — the trace hook never fired", file=sys.stderr)
    sys.exit(2)
