"""#1039 bucket C: does the C-42 pool-drop retry still fire on nvs06, and does
nvs06 still certify? Distinguishes 'mechanism broken' from 'mechanism no longer
needed'. CLAUDE.md 6: executed-check count, non-zero exit at zero. 7: no swallow.
"""
import os, sys
from pathlib import Path
from discopt.modeling import from_nl
from discopt import SolverTuning
DATA = Path("python/tests/data/minlplib_nl")
checks = 0
for name, oracle in (("nvs06", 1.7703125),):
    p = DATA / f"{name}.nl"
    assert p.exists(), p
    for inherit in (True, False):
        res = from_nl(str(p)).solve(time_limit=20, gap_tolerance=1e-4,
                                    tuning=SolverTuning(cut_inherit=inherit))
        st = res.solver_stats or {}
        pool = {k: v for k, v in st.items() if k.startswith("pool/")}
        checks += 1
        print(f"{name} cut_inherit={inherit}: status={res.status} obj={res.objective} "
              f"bound={res.bound} nodes={res.node_count}", flush=True)
        print(f"    pool stats: {pool}", flush=True)
print(f"EXECUTED CHECKS: {checks}")
if checks == 0:
    sys.exit("probe checked nothing")
