"""Where does nvs24's budget-independent ~55 s go?

Samples the main thread's stack on a timer while a `time_limit=6` solve runs, so a
phase that never returns is still visible (CLAUDE.md §10: a timing wrapper that
prints on return never fires for a call that does not return).

Prints an executed-sample count and exits non-zero if it sampled nothing (§6).

RETRACTED PREMISE (CLAUDE.md §11). This probe was written believing that adding an
instrument flipped nvs24 out of its ~52 s slow mode -- a stack sampler, py-spy
--native, cProfile and faulthandler each appeared to drop the primary to 7-12 s.
That reading was WRONG: it rested on single, non-interleaved runs. ``nvs24_arm.py``
re-ran the arms interleaved, 3 reps, and every arm is slow (~53 s) with
``separate/univariate_square`` at 47-48 s in all of them. Instrumentation has no
such effect; the early fast runs were unexplained outliers. Kept as the record of
the wrong turn -- use ``nvs24_arm.py`` for the real measurement.
"""

import os
import sys
import threading
import time
import traceback
from collections import Counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

assert "/python/discopt/" in discopt.__file__, discopt.__file__

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
INTERVAL = 0.25

samples = 0
leaf = Counter()
frame_in_solver = Counter()
timeline = []
stop = threading.Event()
main_id = threading.get_ident()
t0 = time.perf_counter()


def sampler():
    global samples
    while not stop.wait(INTERVAL):
        frames = sys._current_frames()
        f = frames.get(main_id)
        if f is None:
            continue
        stack = traceback.extract_stack(f)
        samples += 1
        top = stack[-1]
        leaf[f"{os.path.basename(top.filename)}:{top.lineno} {top.name}"] += 1
        # Deepest frame that is still inside discopt itself — the phase name.
        for fr in reversed(stack):
            if "/discopt/" in fr.filename:
                key = f"{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"
                frame_in_solver[key] += 1
                timeline.append((time.perf_counter() - t0, key))
                break


th = threading.Thread(target=sampler, daemon=True)
th.start()

m = from_nl("python/tests/data/minlplib/nvs24.nl")
r = m.solve(time_limit=BUDGET)
wall = time.perf_counter() - t0
stop.set()
th.join(timeout=2)

print(
    f"\nbudget={BUDGET}  wall={wall:.1f}  ratio={wall / BUDGET:.2f}x  "
    f"status={r.status} nodes={r.node_count}"
)
print(f"\n--- deepest discopt frame (top 15 of {samples} samples) ---")
for k, v in frame_in_solver.most_common(15):
    print(f"{v * INTERVAL:7.1f}s  {100 * v / max(samples, 1):5.1f}%  {k}")
print("\n--- leaf frame (top 10) ---")
for k, v in leaf.most_common(10):
    print(f"{v * INTERVAL:7.1f}s  {100 * v / max(samples, 1):5.1f}%  {k}")
print("\n--- timeline (phase changes) ---")
prev = None
for t, k in timeline:
    if k != prev:
        print(f"  t={t:6.1f}s  {k}")
        prev = k

print(f"\nEXECUTED_SAMPLES={samples}")
if samples == 0:
    print("PROBE FIRED NOTHING", file=sys.stderr)
    sys.exit(1)
