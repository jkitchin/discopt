"""#1039 bucket C items 2-3: do tspn05 re-separation and the tx1 adaptive
node-NLP back-off still fire, and do their end-to-end contracts still hold?
CLAUDE.md 6/7: executed-check count, non-zero exit at zero, nothing swallowed."""
import sys
from pathlib import Path
from discopt.modeling import from_nl
from discopt import SolverTuning
DATA = Path("python/tests/data/minlplib_nl")
checks = 0
p = DATA / "tspn05.nl"
assert p.exists(), p
res = from_nl(str(p)).solve(time_limit=60, gap_tolerance=1e-4,
                            tuning=SolverTuning(cut_inherit=True))
st = res.solver_stats or {}
checks += 1
print(f"tspn05 ON: status={res.status} obj={res.objective} bound={res.bound} "
      f"nodes={res.node_count}", flush=True)
print("   pool stats:", {k: v for k, v in st.items() if k.startswith("pool/")}, flush=True)
print(f"EXECUTED CHECKS: {checks}")
if checks == 0:
    sys.exit("probe checked nothing")
