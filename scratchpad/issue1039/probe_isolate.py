"""#1039 bucket A: isolate each opt-out flag from the LATER-graduated row filter.

Both remaining bucket-A failures have one root cause: the test varies the flag it
names but leaves ``DISCOPT_RELAX_ROW_FILTER`` at its graduated default (ON since
#671, 2026-07-18), and the row filter supplies hda's tight bound on its own. So
the "legacy baseline" each test describes is not the configuration it actually
runs. Measure the configuration each test MEANS.

No exception swallowed (§7); executed-solve count printed, non-zero exit at zero (§6).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm

assert "/Users/jkitchin/projects/discopt/python/discopt" in dm.__file__, dm.__file__

HDA = "python/tests/data/minlplib_nl/hda.nl"

CONFIGS = [
    # (label, {env}, time_limit)
    (
        "517 OFF, row filter OFF  (legacy no-rescue baseline)",
        {"DISCOPT_NODE_NUMERICAL_DUAL_BOUND": "0", "DISCOPT_RELAX_ROW_FILTER": "0"},
        25,
    ),
    (
        "517 ON,  row filter OFF  (candidate-A floor only)",
        {"DISCOPT_NODE_NUMERICAL_DUAL_BOUND": "1", "DISCOPT_RELAX_ROW_FILTER": "0"},
        25,
    ),
    (
        "671refine OFF, candA ON, row filter OFF",
        {
            "DISCOPT_LP_ITERATIVE_REFINEMENT": "0",
            "DISCOPT_NODE_NUMERICAL_DUAL_BOUND": "1",
            "DISCOPT_RELAX_ROW_FILTER": "0",
        },
        90,
    ),
    (
        "671refine ON,  candA ON, row filter OFF",
        {
            "DISCOPT_LP_ITERATIVE_REFINEMENT": "1",
            "DISCOPT_NODE_NUMERICAL_DUAL_BOUND": "1",
            "DISCOPT_RELAX_ROW_FILTER": "0",
        },
        90,
    ),
]

checks = 0
for label, env, tl in CONFIGS:
    for k in (
        "DISCOPT_NODE_NUMERICAL_DUAL_BOUND",
        "DISCOPT_RELAX_ROW_FILTER",
        "DISCOPT_LP_ITERATIVE_REFINEMENT",
    ):
        os.environ.pop(k, None)
    os.environ.update(env)
    r = dm.from_nl(HDA).solve(time_limit=tl)
    st = r.solver_stats or {}
    print(
        f"{label:42s} status={r.status:11s} bound={r.bound} "
        f"rf_inv={st.get('row_filter/invocations', 0)}",
        flush=True,
    )
    checks += 1

print(f"\nEXECUTED SOLVES: {checks}")
if checks == 0:
    sys.exit(1)
