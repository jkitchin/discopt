"""#1039 bucket E/A: is the `byte-identical` bound drift caused by the flag, or by
the 20s wall-clock deadline landing at a different point? Runs the SAME arm
(flag=0) repeatedly and compares bounds within the arm. Any within-arm spread
means the assertion is measuring timer jitter, not the flag.

CLAUDE.md 6: prints an executed-comparison count and exits non-zero at zero.
CLAUDE.md 7: no exception is swallowed.
"""
import os, sys
import discopt.modeling as dm

MARKER = "test_failure_triggered_is_byte_identical_on_solving_instances"
src = os.path.join("python", "tests", "test_relax_row_filter.py")
assert MARKER in open(src).read(), "version marker absent: wrong tree"
print("discopt from:", dm.__file__, flush=True)

FLAG = "DISCOPT_RELAX_ROW_FILTER"
DATA = os.path.join("python", "tests", "data", "minlplib_nl")
comparisons = 0
for name in ("bchoco07", "casctanks"):
    path = os.path.join(DATA, f"{name}.nl")
    assert os.path.exists(path), path
    bounds = []
    for rep in range(3):
        os.environ[FLAG] = "0"          # same arm every time
        r = dm.from_nl(path).solve(time_limit=20)
        bounds.append((r.bound, r.wall_time, r.node_count, r.status))
        print(f"  {name} flag=0 rep{rep}: bound={r.bound!r} "
              f"wall={r.wall_time:.2f} nodes={r.node_count} status={r.status}", flush=True)
    for i in range(1, len(bounds)):
        comparisons += 1
        if bounds[i][0] != bounds[0][0]:
            print(f"  -> WITHIN-ARM DRIFT on {name}: {bounds[0][0]!r} vs {bounds[i][0]!r}")
    print(f"  {name}: within-arm identical = "
          f"{len({b[0] for b in bounds}) == 1}", flush=True)

print(f"EXECUTED COMPARISONS: {comparisons}")
if comparisons == 0:
    sys.exit("probe compared nothing")
