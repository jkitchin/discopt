"""#1165 entry experiment (A): is the Lagrangian determinism assertion comparing
two wall-clock-TRUNCATED runs, and does a fixed iteration budget remove the noise?

Arm "time"  reproduces the shipped test: two 15 s time-limited solves per case.
Arm "iters" replaces the wall cap with a fixed ``max_iterations`` and a time
limit far larger than the observed runtime, so both arms execute the same work.

Prints an executed-comparison count and exits non-zero if it is zero (CLAUDE.md
§6). No exception is swallowed (§7): a failure must crash, not read as a pass.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "tests"))

import discopt  # noqa: E402
from discopt.decomposition.lagrangian import solve_lagrangian  # noqa: E402
from test_decomposition_adversarial import _rand_gap  # noqa: E402

print(f"[§8] discopt.__file__ = {discopt.__file__}")
assert "/home/user/discopt/python/discopt/" in discopt.__file__, "wrong tree loaded"

MODE = sys.argv[1] if len(sys.argv) > 1 else "time"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MAX_ITERS = int(os.environ.get("PROBE_MAX_ITERS", "40"))

comparisons = 0
disagreements = 0
truncated = 0
t_start = time.time()
for rep in range(REPS):
    for seed in range(6):
        for method in ("subgradient", "bundle", "kelley"):
            mono = _rand_gap(seed).solve(time_limit=30)
            if mono.status != "optimal":
                print(f"  rep{rep} seed={seed} {method}: degenerate, skipped", flush=True)
                continue
            if MODE == "time":
                kw = dict(time_limit=15)
            else:
                kw = dict(time_limit=600, max_iterations=MAX_ITERS)
            t0 = time.time()
            seq = solve_lagrangian(_rand_gap(seed), method=method, backend="sequential", **kw)
            thr = solve_lagrangian(_rand_gap(seed), method=method, backend="threads", **kw)
            dt = time.time() - t0
            comparisons += 1
            same = seq.bound == thr.bound and seq.objective == thr.objective
            if not same:
                disagreements += 1
            if "time_limit" in (seq.status, thr.status):
                truncated += 1
            print(
                f"  rep{rep} seed={seed} {method:12s} {dt:6.1f}s "
                f"seq[{seq.status},{seq.bound!r}] thr[{thr.status},{thr.bound!r}] "
                f"{'SAME' if same else 'DIFFER'}",
                flush=True,
            )

print(f"\n[{MODE}] comparisons={comparisons} disagreements={disagreements} "
      f"time_limit_truncated={truncated} wall={time.time() - t_start:.1f}s")
if comparisons == 0:
    sys.exit("PROBE MADE ZERO COMPARISONS")
