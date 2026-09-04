"""#1039: bchoco07 was byte-identical within the flag=0 arm (3/3 reps). If the
flag=1 arm is equally deterministic at a DIFFERENT bound, the ON-vs-OFF
difference is real and flag-caused, not nondeterminism.
CLAUDE.md 6: executed-comparison count, non-zero exit at zero. 7: nothing swallowed.
"""
import os, sys
import discopt.modeling as dm
FLAG = "DISCOPT_RELAX_ROW_FILTER"
path = os.path.join("python", "tests", "data", "minlplib_nl", "bchoco07.nl")
assert os.path.exists(path), path
print("discopt from:", dm.__file__, flush=True)
comparisons = 0
seen = {}
for arm in ("1", "0", "1", "0", "1"):          # interleaved (CLAUDE.md 9)
    os.environ[FLAG] = arm
    r = dm.from_nl(path).solve(time_limit=20)
    seen.setdefault(arm, []).append(r.bound)
    print(f"  flag={arm}: bound={r.bound!r} wall={r.wall_time:.2f} "
          f"nodes={r.node_count} status={r.status}", flush=True)
for arm, bs in seen.items():
    for b in bs[1:]:
        comparisons += 1
        print(f"  arm {arm} self-consistent: {b == bs[0]}")
comparisons += 1
print(f"  ON vs OFF identical: {set(seen['1']) == set(seen['0'])}")
print(f"EXECUTED COMPARISONS: {comparisons}")
if comparisons == 0:
    sys.exit("probe compared nothing")
