"""#1165 entry experiment (B): is the hda differential assertion comparing two
wall-clock-TRUNCATED solves, and does a deterministic node budget remove it?

Arm "time"  reproduces the shipped test: two 25 s time-limited solves.
Arm "nodes" replaces the wall cap with ``max_nodes=N`` + ``deterministic=True``
and a time limit far larger than the observed runtime, so both arms terminate on
*work* and the reported bound is a function of the model, not of machine speed
(``SolverTuning.deterministic``, #1116).

Repeats each arm REPS times so a spread is reported, not a single number (§9).
Prints an executed-comparison count and exits non-zero if it is zero (§6);
nothing is caught (§7).
"""

import os
import statistics
import sys
import time

import discopt  # noqa: E402
import discopt.modeling as dm  # noqa: E402

print(f"[§8] discopt.__file__ = {discopt.__file__}")
assert "/home/user/discopt/python/discopt/" in discopt.__file__, "wrong tree loaded"

HDA = os.path.join("python", "tests", "data", "minlplib_nl", "hda.nl")
FLAG = "DISCOPT_NODE_NUMERICAL_DUAL_BOUND"

MODE = sys.argv[1] if len(sys.argv) > 1 else "time"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NODES = int(os.environ.get("PROBE_MAX_NODES", "3"))
TL = float(os.environ.get("PROBE_TIME_LIMIT", "600"))


def run(flag_value):
    os.environ["DISCOPT_RELAX_ROW_FILTER"] = "0"
    os.environ[FLAG] = flag_value
    t0 = time.time()
    if MODE == "time":
        r = dm.from_nl(HDA).solve(time_limit=25)
    else:
        r = dm.from_nl(HDA).solve(time_limit=TL, max_nodes=NODES, deterministic=True)
    return r, time.time() - t0


runs = 0
bounds = {"0": [], "1": []}
for rep in range(REPS):
    # Interleaved, not sequential (§9).
    for flag in ("0", "1"):
        r, dt = run(flag)
        runs += 1
        bounds[flag].append(r.bound)
        print(
            f"  rep{rep} {FLAG}={flag} {dt:7.1f}s status={r.status} "
            f"nodes={r.node_count} bound={r.bound!r} obj={r.objective!r}",
            flush=True,
        )

print(f"\n[{MODE}] executed solves = {runs}")
for flag, vals in bounds.items():
    finite = [v for v in vals if v is not None]
    sd = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    print(f"  flag={flag}: {vals}  sd={sd:.6g}")
same = len(set(bounds["0"])) == 1 and len(set(bounds["1"])) == 1
print(f"  within-arm bit-reproducible: {same}")
if runs == 0:
    sys.exit("PROBE RAN NO SOLVE")
