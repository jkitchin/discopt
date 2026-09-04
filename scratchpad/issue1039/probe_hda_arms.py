"""#1039 bucket A: settle whether the hda opt-out is live, at the comment's conditions.

The issue's retraction comment measured ``hda`` at ``time_limit=60`` and reported
``filter_invocations=0`` in BOTH arms, concluding the mechanism is dormant and the
opt-out therefore untestable on this instance. The #1039 counters say otherwise at
time_limit=15 (2 invocations, 356 rows dropped). Re-run at the comment's exact
conditions, both arms, 2 reps interleaved.

No exception swallowed (§7); executed-solve count printed, non-zero exit at zero (§6).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm

assert "/Users/jkitchin/projects/discopt/python/discopt" in dm.__file__, dm.__file__
# §8: assert the marker unique to the version under test is PRESENT.
import discopt._relax.mccormick_lp as _mc

assert "_row_filter_stats" in open(_mc.__file__).read(), "counter not in the loaded module"

checks = 0
for rep in range(2):
    for arm in ("0", "1"):
        os.environ["DISCOPT_RELAX_ROW_FILTER"] = arm
        r = dm.from_nl("python/tests/data/minlplib_nl/hda.nl").solve(time_limit=60)
        st = r.solver_stats or {}
        print(
            f"rep={rep} FLAG={arm} status={r.status} bound={r.bound} "
            f"invocations={st.get('row_filter/invocations', 0)} "
            f"rows_dropped={st.get('row_filter/rows_dropped', 0)}",
            flush=True,
        )
        checks += 1

print(f"\nEXECUTED SOLVES: {checks}")
if checks == 0:
    sys.exit(1)
