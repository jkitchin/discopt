"""Entry experiment (#1141): why does build_convex_spec refuse each corpus instance?

Prints a per-instance verdict and an executed-check count; exits non-zero if the
count is zero (CLAUDE.md §6 -- a probe that measured nothing must not read as a pass).
"""
import sys, traceback, collections, pathlib
from discopt.modeling.core import from_nl
from discopt.solvers import _convex_kernel as ck

DATA = pathlib.Path("python/tests/data/minlplib_nl")
checked = 0
reasons = collections.Counter()
rows = []
for nl in sorted(DATA.glob("*.nl")):
    name = nl.stem
    try:
        model = from_nl(str(nl))
    except Exception as exc:
        rows.append((name, "LOAD_FAIL", f"{type(exc).__name__}: {exc}"))
        reasons["<load failed>"] += 1
        checked += 1
        continue
    try:
        spec = ck._build(model, None)
        verdict, why = "ROUTED", ""
    except ck.NotConvexKernel as exc:
        verdict, why = "REFUSED", str(exc)
    except Exception as exc:
        verdict, why = "ERROR", f"{type(exc).__name__}: {exc}"
    checked += 1
    reasons[why if verdict != "ROUTED" else "<routed>"] += 1
    rows.append((name, verdict, why))
    print(f"{name:28s} {verdict:8s} {why}", flush=True)

print()
print("=== refusal reasons ===")
for why, n in reasons.most_common():
    print(f"{n:4d}  {why}")
print(f"\nEXECUTED CHECKS: {checked}")
if checked == 0:
    print("PROBE MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
