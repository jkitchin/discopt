"""#1039 bucket C item 3: why does the TX1 adaptive back-off not fire on nvs09?
Counts fired batches and back-off/reset events. CLAUDE.md 6/7."""
import logging, os, sys
from discopt.modeling import from_nl
os.environ["DISCOPT_ADAPTIVE_NLP"] = "1"
recs = []
class Cap(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "TX1 adaptive node-NLP" in m:
            recs.append(m)
lg = logging.getLogger("discopt.solver")
lg.setLevel(logging.DEBUG); lg.addHandler(Cap())
res = from_nl("python/tests/data/minlplib_nl/nvs09.nl").solve(time_limit=30)
print(f"status={res.status} obj={res.objective} bound={res.bound} nodes={res.node_count}")
backoff = [m for m in recs if "back off" in m]
reset = [m for m in recs if "reset" in m]
print(f"TX1 messages total={len(recs)} back-off={len(backoff)} reset={len(reset)}")
for m in recs[:8]:
    print("   ", m)
print(f"EXECUTED CHECKS: {1}")
