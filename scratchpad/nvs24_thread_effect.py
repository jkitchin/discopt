"""Does the mere presence of a periodically-waking Python thread change nvs24's wall?

A stack sampler waking every 250 ms made a ``time_limit=6`` solve of nvs24 run in
11.7 s instead of 52.4 s. If a thread that does NOTHING but sleep reproduces that,
the cause is GIL / thread-scheduling, not the sampling.

Arms are run INTERLEAVED (CLAUDE.md §9), one solve per subprocess so nothing carries
over, and the arm is chosen by argv.

Usage: python nvs24_thread_effect.py <arm: none|idle|sample> <time_limit>

VERDICT: no effect. All arms ~54-59 s over 2 interleaved reps. This is the
control that falsified the "instrumentation flips the mode" reading; see
``nvs24_arm.py`` for the 4-arm, 3-rep confirmation.
"""

import os
import sys
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

assert "/python/discopt/" in discopt.__file__, discopt.__file__

arm = sys.argv[1]
budget = float(sys.argv[2])
assert arm in ("none", "idle", "sample"), arm

stop = threading.Event()
ticks = 0
main_id = threading.get_ident()


def idle_loop():
    global ticks
    while not stop.wait(0.25):
        ticks += 1


def sample_loop():
    global ticks
    import traceback

    while not stop.wait(0.25):
        f = sys._current_frames().get(main_id)
        if f is not None:
            traceback.extract_stack(f)
        ticks += 1


th = None
if arm != "none":
    th = threading.Thread(target=idle_loop if arm == "idle" else sample_loop, daemon=True)
    th.start()

m = from_nl("python/tests/data/minlplib/nvs24.nl")
t0 = time.perf_counter()
r = m.solve(time_limit=budget)
wall = time.perf_counter() - t0
stop.set()
if th is not None:
    th.join(timeout=2)

print(
    f"arm={arm:6s} budget={budget:5.1f} wall={wall:7.1f} ratio={wall / budget:5.2f}x "
    f"status={r.status} nodes={r.node_count} bound={r.bound} thread_ticks={ticks}"
)
if arm != "none" and ticks == 0:
    print("PROBE FIRED NOTHING: the thread never woke", file=sys.stderr)
    sys.exit(1)
