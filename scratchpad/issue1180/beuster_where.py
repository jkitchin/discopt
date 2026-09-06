"""What is beuster running at t=150 s and t=190 s, past a 120 s time_limit?

faulthandler, not a timing wrapper: a wrapper that prints on return never fires
for a call that does not return (CLAUDE.md §10).
"""
import faulthandler, sys, time
sys.path.insert(0, "discopt_benchmarks/scripts")
import issue1180_callback_ab as AB

arm = sys.argv[1]
arms = AB.Arms(); arms.install(arm); arms.verify(arm)
from discopt.modeling.core import from_nl
m = from_nl("python/tests/data/minlplib_nl/beuster.nl")
faulthandler.dump_traceback_later(150, repeat=True, exit=False, file=sys.stderr)
t0 = time.perf_counter()
r = m.solve(time_limit=120.0, gap_tolerance=1e-4, deterministic=True, max_nodes=20)
faulthandler.cancel_dump_traceback_later()
print(f"DONE arm={arm} wall={time.perf_counter()-t0:.1f}s nodes={r.node_count} status={r.status}")
