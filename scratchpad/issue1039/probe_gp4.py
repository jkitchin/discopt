"""#1039: find WHERE the understated objective enters.  Trap every call that
returns the bad value 1.998683979470214 by tracing frames that assign it.
CLAUDE.md §7: no exception swallowed."""
import sys, warnings, traceback
import discopt
from discopt import Model

assert "/Users/jkitchin/projects/discopt/python/discopt" in discopt.__file__

BAD = 1.998683979470214
hits = []

def tracer(frame, event, arg):
    if event != "return":
        return
    try:
        if isinstance(arg, float) and abs(arg - BAD) < 1e-12:
            co = frame.f_code
            if "discopt" in co.co_filename:
                hits.append(f"{co.co_filename}:{frame.f_lineno} {co.co_name}")
    except Exception:
        raise
    return

m = Model("div")
x = m.continuous("x", lb=1e-3, ub=1e3)
y = m.continuous("y", lb=1e-3, ub=1e3)
m.minimize(x / y + y / x)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sys.settrace(tracer)
    try:
        r = m.solve(solver="bb", time_limit=5.0)
    finally:
        sys.settrace(None)

print(f"objective={r.objective!r}")
seen = []
for h in hits:
    if h not in seen:
        seen.append(h)
print(f"\nDISTINCT FRAMES RETURNING THE BAD VALUE: {len(seen)}")
for h in seen[:40]:
    print("  ", h)
sys.exit(0 if seen else 1)
