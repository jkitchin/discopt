"""#1165: which of the remaining ON/OFF and seq/thr comparisons can be made to
terminate on WORK affordably?

hda cannot (measured: OFF 305 s, ON >2095 s under ``deterministic=True,
max_nodes=1``). These are the other comparisons of the same shape in the same
two files:

  A. ``test_inert_on_cleanly_certifying_instances`` -- alan / ex1221, today a
     bare ``time_limit=20`` per arm with ``off.bound == on.bound``.
  B. ``test_rand_multicut_sound_and_deterministic`` -- Benders seq vs threads,
     today a bare ``time_limit=30`` per arm with bit-equality.

Reports, per case, the terminal status and whether the arms are bit-identical,
under both the shipped wall budget and a work budget. Prints an executed
comparison count and exits non-zero when it is zero (§6); catches nothing (§7).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "tests"))

import discopt  # noqa: E402
import discopt.modeling as dm  # noqa: E402

print(f"[§8] discopt.__file__ = {discopt.__file__}")
assert "/home/user/discopt/python/discopt/" in discopt.__file__, "wrong tree loaded"

from discopt.decomposition.benders import solve_benders  # noqa: E402
from discopt.decomposition.benders.solver import BendersConfig  # noqa: E402
from test_decomposition_adversarial import _rand_two_stage  # noqa: E402

_NL = os.path.join("python", "tests", "data", "minlplib_nl")
_FLAG = "DISCOPT_NODE_NUMERICAL_DUAL_BOUND"
# The #1039 recipe already in test_relax_row_filter.py.
_WORK_KW = {"deterministic": True, "max_nodes": 25, "time_limit": 120}

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
comparisons = 0

print("\n=== A. #517 flag inertness on cleanly-certifying instances ===")
for name in ("alan", "ex1221"):
    path = os.path.join(_NL, f"{name}.nl")
    for tag, kw in (("shipped tl=20", {"time_limit": 20}), ("work-terminated", _WORK_KW)):
        for rep in range(REPS):
            os.environ[_FLAG] = "0"
            t0 = time.time()
            off = dm.from_nl(path).solve(**kw)
            os.environ[_FLAG] = "1"
            on = dm.from_nl(path).solve(**kw)
            dt = time.time() - t0
            comparisons += 1
            print(
                f"  {name:8s} {tag:16s} rep{rep} {dt:6.1f}s "
                f"off[{off.status},n={off.node_count},{off.bound!r}] "
                f"on[{on.status},n={on.node_count},{on.bound!r}] "
                f"{'SAME' if off.bound == on.bound and off.objective == on.objective else 'DIFFER'}",
                flush=True,
            )
del os.environ[_FLAG]

print("\n=== B. Benders multicut backend determinism ===")
for seed in range(8):
    for rep in range(REPS):
        t0 = time.time()
        seq = solve_benders(
            _rand_two_stage(seed), config=BendersConfig(time_limit=30, multicut=True,
                                                        backend="sequential")
        )
        thr = solve_benders(
            _rand_two_stage(seed), config=BendersConfig(time_limit=30, multicut=True,
                                                        backend="threads")
        )
        dt = time.time() - t0
        comparisons += 1
        print(
            f"  seed={seed} rep{rep} {dt:6.1f}s seq[{seq.status},{seq.objective!r},{seq.bound!r}] "
            f"thr[{thr.status},{thr.objective!r},{thr.bound!r}] "
            f"{'SAME' if seq.bound == thr.bound and seq.objective == thr.objective else 'DIFFER'}",
            flush=True,
        )

print(f"\nexecuted comparisons = {comparisons}")
if comparisons == 0:
    sys.exit("PROBE MADE ZERO COMPARISONS")
