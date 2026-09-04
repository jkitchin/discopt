"""#1039 bucket A: test_hda_optout_restores_loose_candidate_a_floor now passes.
Its asserts sit under `if r.bound is not None and math.isfinite(r.bound)`, so a
None/non-finite bound makes it pass VACUOUSLY (CLAUDE.md 6). Determine which.
Also record the OFF/ON bound for the two flags the #1039 retraction never measured.
"""
import math, os, sys
import discopt.modeling as dm
path = os.path.join("python", "tests", "data", "minlplib_nl", "hda.nl")
assert os.path.exists(path), path
print("discopt from:", dm.__file__, flush=True)
checks = 0
for flag in ("DISCOPT_RELAX_ROW_FILTER",
             "DISCOPT_NODE_NUMERICAL_DUAL_BOUND",
             "DISCOPT_LP_ITERATIVE_REFINEMENT"):
    for arm in ("0", "1"):
        os.environ.pop("DISCOPT_RELAX_ROW_FILTER", None)
        os.environ.pop("DISCOPT_NODE_NUMERICAL_DUAL_BOUND", None)
        os.environ.pop("DISCOPT_LP_ITERATIVE_REFINEMENT", None)
        os.environ[flag] = arm
        r = dm.from_nl(path).solve(time_limit=60)
        finite = r.bound is not None and math.isfinite(r.bound)
        checks += 1
        print(f"  {flag}={arm}: bound={r.bound!r} finite={finite} "
              f"status={r.status} nodes={r.node_count} wall={r.wall_time:.1f}", flush=True)
        if flag == "DISCOPT_RELAX_ROW_FILTER" and arm == "0":
            print(f"    -> asserts would {'RUN' if finite else 'BE SKIPPED (vacuous pass)'}")
print(f"EXECUTED CHECKS: {checks}")
if checks == 0:
    sys.exit("probe checked nothing")
