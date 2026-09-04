"""#1153: which NLP call gets the big wall grant, and what does it spend?

Wraps the POUNCE entry point and records (granted max_wall_time, wall spent,
caller). Exits non-zero if no NLP call was observed (CLAUDE.md §6).
"""
import os, sys, time, traceback
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt
from discopt.modeling.core import from_nl
import discopt.solvers.nlp_pounce as NP
import discopt.solver as S

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
calls = []
_real = NP.solve_nlp


def wrapped(evaluator, x0, **kw):
    grant = (kw.get("options") or {}).get("max_wall_time")
    stack = [f.name for f in traceback.extract_stack()[:-1]
             if f.filename.endswith(("solver.py", "primal_heuristics.py"))]
    t = time.perf_counter()
    try:
        return _real(evaluator, x0, **kw)
    finally:
        calls.append((grant, time.perf_counter() - t, "<-".join(reversed(stack[-3:]))))


NP.solve_nlp = wrapped
S.solve_nlp_pounce = wrapped

inst = sys.argv[1]
seen = 0
for tl in [float(x) for x in sys.argv[2].split(",")]:
    calls.clear()
    r = from_nl(inst).solve(time_limit=tl, gap_tolerance=1e-4)
    seen += len(calls)
    tot = sum(c[1] for c in calls)
    print(f"\n=== tl={tl} nodes={r.node_count} obj={r.objective!r} bound={r.bound!r} "
          f"wall={r.wall_time:.2f} root_time={r.root_time!r} pounce={r.pounce_time:.2f} "
          f"| {len(calls)} NLP calls totalling {tot:.2f}s", flush=True)
    for grant, spent, where in sorted(calls, key=lambda c: -c[1])[:10]:
        print(f"    grant={grant!r:>10} spent={spent:7.2f}s  {where}", flush=True)
print(f"\n# observed NLP calls: {seen}", flush=True)
raise SystemExit(1 if seen == 0 else 0)
