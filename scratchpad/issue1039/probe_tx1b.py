"""#1039 bucket C item 3: find an instance whose search is long enough to
exercise the TX1 adaptive back-off (needs >=2 consecutive non-improving FIRED
batches). nvs09 now certifies in 31 nodes and fires nothing. CLAUDE.md 6/7."""
import logging, os, sys
from discopt.modeling import from_nl
os.environ["DISCOPT_ADAPTIVE_NLP"] = "1"
DATA = "python/tests/data/minlplib_nl"
checks = 0
for name in ("nvs17", "casctanks", "bchoco07", "tspn05"):
    p = os.path.join(DATA, f"{name}.nl")
    if not os.path.exists(p):
        print(f"{name}: not vendored"); continue
    recs = []
    class Cap(logging.Handler):
        def emit(self, r):
            m = r.getMessage()
            if "TX1 adaptive node-NLP" in m:
                recs.append(m)
    lg = logging.getLogger("discopt.solver")
    lg.setLevel(logging.DEBUG)
    h = Cap(); lg.addHandler(h)
    res = from_nl(p).solve(time_limit=30)
    lg.removeHandler(h)
    checks += 1
    print(f"{name:11s} status={res.status:10s} nodes={res.node_count:6d} "
          f"obj={res.objective} bound={res.bound} | TX1 total={len(recs)} "
          f"backoff={sum('back off' in m for m in recs)} "
          f"reset={sum('reset' in m for m in recs)}", flush=True)
print(f"EXECUTED CHECKS: {checks}")
if checks == 0:
    sys.exit("probe checked nothing")
