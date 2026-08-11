"""Which nvs24 root phase burns the budget-independent ~46 s?

Every in-process instrument tried so far PERTURBS the answer away: a Python stack
sampler, py-spy --native and cProfile each drop the primary from ~53 s to 7-12 s and
change the reported bound (-56272 -> -103844), i.e. the expensive phase stops
running. ``faulthandler.dump_traceback_later(repeat=True)`` uses a C watchdog thread
that does not contend for the GIL and costs one dump per interval, so it is the one
instrument light enough to leave the slow mode intact (CLAUDE.md §10).

Prints the dump count and exits non-zero if the watchdog never fired (§6).

RETRACTED PREMISE (CLAUDE.md §11). This probe was written believing that adding an
instrument flipped nvs24 out of its ~52 s slow mode -- a stack sampler, py-spy
--native, cProfile and faulthandler each appeared to drop the primary to 7-12 s.
That reading was WRONG: it rested on single, non-interleaved runs. ``nvs24_arm.py``
re-ran the arms interleaved, 3 reps, and every arm is slow (~53 s) with
``separate/univariate_square`` at 47-48 s in all of them. Instrumentation has no
such effect; the early fast runs were unexplained outliers. Kept as the record of
the wrong turn -- use ``nvs24_arm.py`` for the real measurement.
"""

import faulthandler
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt._relax.deadline import deadline_scope  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solver import solve_model  # noqa: E402

assert "/python/discopt/" in discopt.__file__, discopt.__file__

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 3.9
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

dump_path = open(os.path.join(os.path.dirname(__file__), "nvs24_faulthandler.txt"), "w")
m = from_nl("python/tests/data/minlplib/nvs24.nl")

t0 = time.perf_counter()
faulthandler.dump_traceback_later(INTERVAL, repeat=True, file=dump_path, exit=False)
try:
    with deadline_scope(BUDGET):
        r = solve_model(m, time_limit=BUDGET)
finally:
    faulthandler.cancel_dump_traceback_later()
    dump_path.flush()
    dump_path.close()
wall = time.perf_counter() - t0

print(
    f"WALL={wall:.1f} ({wall / BUDGET:.1f}x) status={r.status} nodes={r.node_count} bound={r.bound}"
)

text = open(os.path.join(os.path.dirname(__file__), "nvs24_faulthandler.txt")).read()
dumps = text.count("Timeout (")
print(f"WATCHDOG_DUMPS={dumps}")
if dumps == 0:
    print("PROBE FIRED NOTHING: watchdog never dumped", file=sys.stderr)
    sys.exit(1)
