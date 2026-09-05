"""#1165 regression evidence: the OLD recipe fails under load, the NEW one does not.

Runs the failing case (``kelley``/seed 4) under both recipes, N reps, and counts
how often ``seq.bound == thr.bound and seq.objective == thr.objective`` holds:

  old: two solves at ``time_limit=15``            (what shipped)
  new: two solves at ``max_iterations=50, time_limit=300``  (this change)

Intended to be run WITH deliberate CPU contention -- that is the condition the
old assertion is sensitive to and the whole point of the comparison. Prints an
executed-comparison count and exits non-zero when it is zero (§6); catches
nothing (§7).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "tests"))

import discopt  # noqa: E402

print(f"[§8] discopt.__file__ = {discopt.__file__}", flush=True)
assert "/home/user/discopt/python/discopt/" in discopt.__file__, "wrong tree loaded"

from discopt.decomposition.lagrangian import solve_lagrangian  # noqa: E402
from test_decomposition_adversarial import _rand_gap  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
RECIPES = {
    "old (time_limit=15)": dict(time_limit=15),
    "new (max_iterations=50, time_limit=300)": dict(max_iterations=50, time_limit=300),
}

comparisons = 0
agree = {k: 0 for k in RECIPES}
truncated = {k: 0 for k in RECIPES}
for rep in range(REPS):
    for label, kw in RECIPES.items():  # interleaved, not sequential (§9)
        t0 = time.time()
        seq = solve_lagrangian(_rand_gap(4), method="kelley", backend="sequential", **kw)
        thr = solve_lagrangian(_rand_gap(4), method="kelley", backend="threads", **kw)
        dt = time.time() - t0
        comparisons += 1
        same = seq.bound == thr.bound and seq.objective == thr.objective
        agree[label] += same
        truncated[label] += ("time_limit" in (seq.status, thr.status))
        print(
            f"  rep{rep} {label:42s} {dt:6.1f}s seq[{seq.status},{seq.bound!r}] "
            f"thr[{thr.status},{thr.bound!r}] {'SAME' if same else 'DIFFER'}",
            flush=True,
        )

print(f"\nexecuted comparisons = {comparisons}")
for label in RECIPES:
    print(f"  {label:42s} agreed {agree[label]}/{REPS}, "
          f"wall-truncated {truncated[label]}/{REPS}")
if comparisons == 0:
    sys.exit("PROBE MADE ZERO COMPARISONS")
