"""#1039 bucket A: find an instance that still provokes an uncertified node LP.

The issue's own retraction comment asks for exactly this: the three opt-out tests
assert that hda's root LP false-fails without the filter, hda no longer does, and
the tests need re-pointing at an instance that still opens the failure branch.
The #1039 ``row_filter/{invocations,rows_dropped}`` counters make that searchable.

The distinction that matters: ``invocations`` counts the branch OPENING (the node
LP produced no certified verdict); ``rows_dropped`` counts rows the filter actually
removed. Only when rows are dropped do the ON and OFF paths genuinely differ -- an
invocation that drops nothing leaves the arms identical and the opt-out still has
nothing to compare.

No exception swallowed (§7); executed-solve count printed, non-zero exit at zero (§6).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm

assert "/Users/jkitchin/projects/discopt/python/discopt" in dm.__file__, dm.__file__

DATA = "python/tests/data/minlplib_nl"
names = sorted(f[:-3] for f in os.listdir(DATA) if f.endswith(".nl"))

checks = 0
hits = []
for name in names:
    os.environ["DISCOPT_RELAX_ROW_FILTER"] = "1"
    r = dm.from_nl(os.path.join(DATA, f"{name}.nl")).solve(time_limit=15)
    st = r.solver_stats or {}
    inv = st.get("row_filter/invocations", 0)
    drop = st.get("row_filter/rows_dropped", 0)
    checks += 1
    if inv:
        hits.append((name, inv, drop))
        print(
            f"HIT {name:22s} invocations={inv:6.0f} rows_dropped={drop:8.0f} "
            f"status={r.status} bound={r.bound}",
            flush=True,
        )

print()
print("instances opening the filter branch:")
for name, inv, drop in hits:
    print(f"  {name:22s} invocations={inv:6.0f} rows_dropped={drop:8.0f}")
print(f"\nEXECUTED SOLVES: {checks}")
if checks == 0:
    sys.exit(1)
