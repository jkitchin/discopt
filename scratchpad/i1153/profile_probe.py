"""#1153: the solver's own layer profile per rung — where does the wall go?"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt
from discopt.modeling.core import from_nl
print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
inst = sys.argv[1]
n = 0
for tl in [float(x) for x in sys.argv[2].split(",")]:
    t = time.perf_counter()
    r = from_nl(inst).solve(time_limit=tl, gap_tolerance=1e-4)
    n += 1
    print(f"\n=== tl={tl} nodes={r.node_count} obj={r.objective!r} bound={r.bound!r} "
          f"status={r.status} route={r.algorithm_route!r} wall={r.wall_time:.2f} "
          f"(measured {time.perf_counter()-t:.2f})", flush=True)
    for f in ("rust_time", "python_time", "jax_time", "pounce_time", "root_time",
              "mip_count", "root_bound", "convex_fast_path", "nlp_bb"):
        print(f"    {f:18s} = {getattr(r, f, None)!r}", flush=True)
    st = r.solver_stats or {}
    for k, v in sorted(st.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else 0))[:14]:
        print(f"    stat {k:34s} = {v!r}", flush=True)
print(f"\n# executed rungs: {n}", flush=True)
raise SystemExit(1 if n == 0 else 0)
