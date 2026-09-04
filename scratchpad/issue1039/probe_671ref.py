"""#1039 bucket A: does DISCOPT_NODE_NUMERICAL_DUAL_BOUND=1 LOOSEN hda's bound?

First pass (probe_isolate.py), row filter pinned OFF, time_limit=90:
    517 OFF -> -141697.43   517 ON -> -13992288065.86
Both are sound lower bounds, but ON is ~5 orders of magnitude looser. That is a
strong claim about a graduated flag, so it is re-measured here with the arms
INTERLEAVED and repeated (CLAUDE.md §9) before being written down anywhere.

No exception swallowed (§7); executed-solve count printed, non-zero exit at zero (§6).
"""

import os
import statistics
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm

assert "/Users/jkitchin/projects/discopt/python/discopt" in dm.__file__, dm.__file__

HDA = "python/tests/data/minlplib_nl/hda.nl"
HDA_OPT = -5964.534084

arms: dict[str, list] = {"0": [], "1": []}
checks = 0
for rep in range(3):
    for arm in ("0", "1"):
        os.environ["DISCOPT_RELAX_ROW_FILTER"] = "0"
        os.environ["DISCOPT_LP_ITERATIVE_REFINEMENT"] = arm
        os.environ["DISCOPT_NODE_NUMERICAL_DUAL_BOUND"] = "1"
        r = dm.from_nl(HDA).solve(time_limit=90)
        arms[arm].append(r.bound)
        sound = r.bound is None or r.bound <= HDA_OPT + 1e-2
        print(
            f"rep={rep} 671ref={arm} status={r.status} bound={r.bound} sound={sound}",
            flush=True,
        )
        assert sound, f"UNSOUND bound {r.bound} > opt {HDA_OPT}"
        checks += 1

for arm, vals in arms.items():
    finite = [v for v in vals if v is not None]
    sd = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    print(f"\n671ref={arm}: values={vals} sd={sd:.6g}")

print(f"\nEXECUTED SOLVES: {checks}")
if checks == 0:
    sys.exit(1)
