"""Default-path neutrality panel for the root DIRECT probe, on a NODE budget.

From the review of PR #1006 (finding 4).

The default-path neutrality check ran under ``time_limit``, so three instances
(bchoco07, bchoco08, clay0303hfsg) showed a difference that a wall-clock budget
can produce on its own -- a busier machine stops at a different node and reports
a different bound with no code change involved. ``max_nodes`` is deterministic:
if the difference was a timing artifact it vanishes, if it was real it survives.

Arms per instance: DISCOPT_DIRECT_HEURISTIC unset (the default path) / "0"
(explicit opt-out) / "1" (flag on). unset vs =0 MUST be bit-identical -- the
bound-neutral regime of CLAUDE.md 5 -- and the probe must fire ZERO times on
either.

A ``time_limit`` backstop is set so the panel terminates, and whether it BOUND is
reported per arm. A bound backstop reintroduces exactly the nondeterminism this
run exists to remove, so an arm that hits it is reported as inconclusive rather
than counted (CLAUDE.md 6: the probe must say when it measured nothing).
``faulthandler`` dumps a traceback if any single arm exceeds the dump interval,
because a step that may never return is invisible to a wrapper that prints on
return (CLAUDE.md 10).
"""

import faulthandler
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "0"

from pathlib import Path  # noqa: E402

import discopt  # noqa: E402
from discopt.heuristic_governor import governor  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402

# CLAUDE.md 8: prove which code is loaded, and that it is the version under test.
print(f"discopt from {discopt.__file__}", flush=True)
import discopt.solver as _s  # noqa: E402

assert hasattr(_s, "_direct_root_primal"), "marker absent: this is not the branch under test"
print("marker _direct_root_primal present", flush=True)

CORPUS = Path("/home/user/discopt/python/tests/data/minlplib_nl")
MAX_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
TL = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
# Cheapest first, so partial output is still useful if the panel is cut short.
NAMES = sys.argv[3:] or ["clay0303hfsg", "bchoco08", "bchoco07"]

faulthandler.enable()


def run(name, flag):
    if flag is None:
        os.environ.pop("DISCOPT_DIRECT_HEURISTIC", None)
    else:
        os.environ["DISCOPT_DIRECT_HEURISTIC"] = flag
    governor().reset()
    m = from_nl(str(CORPUS / f"{name}.nl"))
    faulthandler.dump_traceback_later(TL + 60.0, exit=False)
    try:
        r = m.solve(max_nodes=MAX_NODES, time_limit=TL)
    finally:
        faulthandler.cancel_dump_traceback_later()
    return r, int(governor().snapshot().get("direct", {}).get("calls", 0))


def key(r):
    return (r.status, r.node_count, r.objective, r.bound, r.gap_certified)


compared = 0
inconclusive = 0
drift = []
print(f"panel: max_nodes={MAX_NODES} time_limit={TL}s backstop, instances={NAMES}", flush=True)

for name in NAMES:
    p = CORPUS / f"{name}.nl"
    if not p.exists():
        print(f"{name}: MISSING, skipped", flush=True)
        continue
    print(f"\n{name}", flush=True)
    got = {}
    for label, flag in (("unset", None), ("=0", "0"), ("=1", "1")):
        t0 = time.perf_counter()
        r, c = run(name, flag)
        wall = time.perf_counter() - t0
        got[label] = (r, c, wall)
        # The backstop binding is the thing that makes an arm uncomparable.
        hit_tl = r.status == "time_limit" or wall >= TL * 0.98
        print(
            f"   {label:6s} nodes={r.node_count} status={r.status} "
            f"obj={r.objective!r} bnd={r.bound!r} cert={r.gap_certified} "
            f"direct_calls={c} wall={wall:.1f}s"
            f"{'  <-- TIME BACKSTOP BOUND, arm not comparable' if hit_tl else ''}",
            flush=True,
        )
    (unset, c_unset, w0), (off, c_off, w1), (on, c_on, w2) = got["unset"], got["=0"], got["=1"]

    bound_by_time = [
        lbl for lbl, (r, _c, w) in got.items() if r.status == "time_limit" or w >= TL * 0.98
    ]
    if bound_by_time:
        inconclusive += 1
        print(
            f"   INCONCLUSIVE: arms {bound_by_time} hit the wall-clock backstop, so this "
            f"instance is NOT a deterministic comparison -- the same artifact as the "
            f"original run. Not counted.",
            flush=True,
        )
        continue

    compared += 1
    if key(unset) != key(off):
        drift.append(f"{name}: default path is NOT identical to the explicit =0 opt-out")
    if c_unset or c_off:
        drift.append(f"{name}: the direct probe FIRED on the default path ({c_unset}/{c_off})")
    # Flag ON may change node counts -- that is what a primal heuristic does. It
    # may not move a bound the wrong way or lose a certificate.
    if unset.bound is not None and on.bound is not None:
        if on.bound > unset.bound + 1e-6 * max(1.0, abs(unset.bound)):
            drift.append(f"{name}: bound rose with the flag on {unset.bound!r} -> {on.bound!r}")
    if unset.gap_certified and not on.gap_certified:
        drift.append(f"{name}: lost certification with the flag on")
    if on.bound is not None and on.objective is not None and on.bound > on.objective + 1e-6:
        drift.append(f"{name}: bound {on.bound!r} above incumbent {on.objective!r} with flag on")

print(f"\ncompared={compared} deterministic instances, inconclusive={inconclusive}")
for d in drift:
    print(f"DRIFT {d}")
if compared == 0:
    print("PROBE MEASURED NOTHING DETERMINISTIC")
    sys.exit(2)
print("no drift" if not drift else f"{len(drift)} drift findings")
sys.exit(1 if drift else 0)
