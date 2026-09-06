"""Reproduce the 3 blocking findings of review 5122926860 on PR #1158.

Run from the repository root. Prints an executed-assertion count and exits
non-zero if it made none, or if any probe raised (CLAUDE.md §6/§7).

Kept as the record of what each finding actually did, since all three are now
fixed and the regression tests assert the *fixed* behaviour rather than showing
the failure. Measured values:

===  ==========================================  ==========================================
     before (f9f4a2b)                            after (1347c2b)
===  ==========================================  ==========================================
B1   accept_local_incumbent -> -0.0002           -> None
     (true optimum 0, source residual 1e-4)      (recomputed at the boundary)
B2   slow axis [-67.8924, 67.8924],              [-44413.4, 44413.4], contains it
     excludes the exact column sum 10000         (fast-axis control stays tight)
B3   status 'local_optimal' at the untouched     status 'local_limit'
     starting point, product row violated by 24
===  ==========================================  ==========================================
"""
import sys
import traceback

import discopt
import discopt._rust as _rust
import discopt.modeling.core as dm
import discopt.mpec_report as mr
import numpy as np

print("discopt:", discopt.__file__)
print("_rust:  ", _rust.__file__)
print("marker accept_local_incumbent:", hasattr(mr, "accept_local_incumbent"))
checks = 0

# ── B1: accept_local_incumbent trusts a report measured at a DIFFERENT point ──
def b1():
    global checks
    from discopt.mpec import complementarity, reformulate_scholtes
    m = dm.Model("b1")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(-x - y)
    m.subject_to(x == y)
    pair = complementarity(x, y, "p")
    reformulate_scholtes(m, [pair], 1e-8)
    m._complementarities.append(pair)

    good = np.array([0.0, 0.0])
    report_at_origin = mr.source_residual_report(m, x_flat=good)
    checks += 1
    print(f"B1 report at (0,0): source_satisfied={report_at_origin.source_satisfied}")

    class R:
        x = {"x": np.array(0.0), "y": np.array(0.0)}
        objective = 0.0
        mpec_report = report_at_origin

    bad = np.array([1e-4, 1e-4])
    fresh = mr.source_residual_report(m, x_flat=bad)
    checks += 1
    print(f"B1 fresh report at (1e-4,1e-4): source_satisfied={fresh.source_satisfied} "
          f"complementarity={fresh.complementarity.value:.3e}")

    verdict = mr.accept_local_incumbent(m, R(), x_flat=bad)
    checks += 1
    print(f"B1 accept_local_incumbent(x_flat=(1e-4,1e-4)) -> {verdict}   "
          f"(true optimum 0; anything < 0 is a false incumbent)")

# ── B2: axis-sum enclosure can exclude the exact sum on a slow memory axis ──
def b2():
    global checks
    from discopt._relax.convexity.interval import Interval
    from discopt._relax.convexity.interval_eval import evaluate_interval
    n = 10002
    a = np.ones((n, 2), dtype=np.float64, order="C")
    a[0, :] = 1e16
    a[-1, :] = -1e16
    exact = 10000.0  # each column: 1e16 + 10000*1 - 1e16 in exact arithmetic

    m = dm.Model("b2")
    v = m.continuous("v", shape=(n, 2), lb=-1e17, ub=1e17)
    iv = evaluate_interval(dm.sum(v, axis=0), m, {v: Interval(a.copy(), a.copy())})
    lo = np.asarray(iv.lo).ravel()
    hi = np.asarray(iv.hi).ravel()
    contains = bool(np.all(lo <= exact) and np.all(exact <= hi))
    checks += 1
    print(f"B2 slow-axis (axis=0 on C-contiguous): lo={lo[0]:.6g} hi={hi[0]:.6g} "
          f"contains exact {exact:g}? {contains}")

    at = np.ascontiguousarray(a.T)  # fast-axis control
    m2 = dm.Model("b2c")
    v2 = m2.continuous("v2", shape=at.shape, lb=-1e17, ub=1e17)
    iv2 = evaluate_interval(dm.sum(v2, axis=1), m2, {v2: Interval(at.copy(), at.copy())})
    lo2 = np.asarray(iv2.lo).ravel()
    hi2 = np.asarray(iv2.hi).ravel()
    contains2 = bool(np.all(lo2 <= exact) and np.all(exact <= hi2))
    checks += 1
    print(f"B2 fast-axis control:                 lo={lo2[0]:.6g} hi={hi2[0]:.6g} "
          f"contains exact? {contains2}")

# ── B3: a stage that performs no optimization still yields local_optimal ──
def b3():
    global checks
    from discopt.mpec import complementarity, solve_mpec
    m = dm.Model("b3")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 5) ** 2 + (y - 5) ** 2 - 18)
    pair = complementarity(x, y, "p")
    res = solve_mpec(m, [pair], method="scholtes", max_iter=1,
                     x0=np.array([5.0, 5.0]), nlp_options={"max_iter": 0})
    checks += 1
    print(f"B3 zero-iteration solve -> status={getattr(res,'status',None)!r} "
          f"objective={getattr(res,'objective',None)} "
          f"gap_certified={getattr(res,'gap_certified',None)}")

# Each probe is caught so one failure does not hide the other two findings, but a
# raise is a BROKEN PROBE, never a pass: it is printed with its traceback and the
# script exits non-zero. A bare ``except`` here would turn "this path is broken"
# into "this path is fine", which is the CLAUDE.md §7 failure applied to the very
# instrument used to judge the fixes.
raised: list[str] = []
for fn in (b1, b2, b3):
    print(f"--- {fn.__name__} ---")
    try:
        fn()
    except Exception as exc:
        traceback.print_exc()
        print(f"{fn.__name__} RAISED {type(exc).__name__}: {exc}")
        raised.append(fn.__name__)

print("EXECUTED_ASSERTIONS:", checks)
if checks == 0:
    sys.exit("probe measured nothing")
if raised:
    sys.exit(f"probe(s) raised, so their findings were NOT measured: {', '.join(raised)}")
