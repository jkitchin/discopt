"""#1141: the DEFAULT solve, end to end — route, guard, fallback and all.

`panel_route_target.py` forced `mip_nlp_method`/`milp_solver`, so it measured the
MASTER and the route's progress guard never ran. That is the wrong configuration
to draw a default-path conclusion from, and it hid a regression: on a plain
`solve(time_limit=30)` of tls2 the retargeted route is abandoned by the #1066
guard at 15.1 s and the run ends `time_limit` with NO incumbent, where the
pre-change route returned `optimal` 5.3 in 0.58 s.

The guard was calibrated on the HiGHS master, which certified in a small fraction
of its budget; the in-house master is ~3x slower on this population, so both the
guard's trailing-gap criterion AND the fixed 50% fallback reserve cut it off
before it can finish.

Arms, all plain `solve(time_limit=...)` with no kwargs, interleaved per instance:

  pre        -- the pre-#1141 default, emulated IN-PROCESS by patching the route
                to return {"milp_solver": "highs"} again, so the guard, the
                reserve and the fallback all behave exactly as they did.
  inhouse    -- the current default (route -> in-house master, guard on).
  noguard    -- current default with DISCOPT_CONVEX_ROUTE_GUARD=0 (fixed 50%
                split), to separate "the guard's criterion is wrong here" from
                "the reserve is too small here".

Incumbents feasibility-verified; every arm's bound checked against every arm's
verified incumbent and the reference optimum. Executed-check count printed (§6).
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "python/tests")
import discopt.solver as _solver  # noqa: E402
from _optima import known_optimum, optima_registry  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt._tape_nlp_evaluator import make_evaluator  # noqa: E402
from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: E402

DIRS = [pathlib.Path("python/tests/data/minlplib_nl"),
        pathlib.Path("python/tests/data/minlplib")]
TL = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
_real_route = _solver._convex_minlp_auto_route


def _route_as_highs(model):
    """The pre-#1141 route, verbatim: same method, same reason shape, highs pinned."""
    method, reason, opts = _real_route(model)
    if method != "lp_nlp_bb":
        return method, reason, opts
    return method, reason, {**opts, "milp_solver": "highs"}


def _route_as_oa(model):
    """Route to `"oa"`, the other discopt-native target.

    Not an afterthought: the #1066 guard's `master_checkin_deadline` limb was
    built FOR `"oa"`, so `"oa"` may survive the budget policy that the in-house
    `lp_nlp_bb` master does not -- and on the master-only panel `"oa"` looked
    worst, which is exactly the kind of ranking a default-path measurement can
    invert.
    """
    method, reason, opts = _real_route(model)
    if method != "lp_nlp_bb":
        return method, reason, opts
    return "oa", reason, {k: v for k, v in opts.items() if k != "milp_solver"}


ARMS = {"pre": ("1", _route_as_highs), "inhouse": ("1", _real_route),
        "noguard": ("0", _real_route), "oa": ("1", _route_as_oa)}


def flat_point(model, xd):
    if xd is None:
        return None
    lb, _ = flat_variable_bounds(model)
    out, k = np.zeros(len(lb)), 0
    for v in model._variables:
        if v.name not in xd:
            return None
        out[k:k + v.size] = np.atleast_1d(np.asarray(xd[v.name], float)).ravel()
        k += v.size
    return out


def max_violation(model, x):
    g = np.asarray(make_evaluator(model).evaluate_constraints(x), float)
    senses = [c.sense if isinstance(c.sense, str) else c.sense.value
              for c in model._constraints]
    worst = 0.0
    for gi, s in zip(g, senses):
        worst = max(worst, gi if s == "<=" else (-gi if s == ">=" else abs(gi)))
    lb, ub = flat_variable_bounds(model)
    if len(x):
        worst = max(worst, float(np.max(np.maximum(lb - x, x - ub))))
    return float(worst)


paths = {}
for d in DIRS:
    for p_ in sorted(d.glob("*.nl")):
        paths.setdefault(p_.stem, p_)
os.environ["DISCOPT_CONVEX_MINLP_ROUTE"] = "1"
routed = [n for n, p_ in paths.items()
          if _real_route(from_nl(str(p_)))[0] is not None]
print(f"instances the router diverts: {len(routed)}/{len(paths)}", flush=True)
if not routed:
    print("PANEL MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)

print(f"load before {os.getloadavg()}", flush=True)
print(f"{'instance':22s} {'arm':8s} {'status':12s} {'objective':>17s} {'bound':>17s} "
      f"{'wall':>7s} {'feas':>9s}", flush=True)
rows, checks, viol = [], 0, []
for name in routed:
    rec = {"instance": name}
    for arm, (guard, route_fn) in ARMS.items():
        os.environ["DISCOPT_CONVEX_ROUTE_GUARD"] = guard
        _solver._convex_minlp_auto_route = route_fn
        try:
            m = from_nl(str(paths[name]))
            rec.setdefault("sense",
                           1.0 if m._objective.sense == ObjectiveSense.MINIMIZE else -1.0)
            t = time.perf_counter()
            r = m.solve(time_limit=TL, gap_tolerance=1e-4)   # DEFAULT: no kwargs
            wall = time.perf_counter() - t
            x = flat_point(m, getattr(r, "x", None))
            rec[arm] = {"status": str(r.status), "objective": r.objective,
                        "bound": r.bound, "wall": wall,
                        "feas": None if x is None else max_violation(m, x),
                        "route": str(getattr(r, "algorithm_route", None))[:150]}
        except Exception as exc:
            rec[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name:22s} {arm:8s} RAISED {type(exc).__name__}: {exc}", flush=True)
            continue
        finally:
            _solver._convex_minlp_auto_route = _real_route
        d = rec[arm]
        ob = "None" if d["objective"] is None else f"{d['objective']:.10g}"
        bd = "None" if d["bound"] is None else f"{d['bound']:.10g}"
        fs = "None" if d["feas"] is None else f"{d['feas']:.2e}"
        print(f"{name:22s} {arm:8s} {d['status']:12s} {ob:>17s} {bd:>17s} "
              f"{d['wall']:7.2f} {fs:>9s}", flush=True)

    s = rec["sense"]
    refs = []
    if name in optima_registry():
        refs.append(("ORACLE", float(known_optimum(name))))
    for a2 in ARMS:
        d2 = rec.get(a2) or {}
        if d2.get("objective") is not None and (d2.get("feas") or 1.0) <= 1e-6:
            refs.append((f"verified incumbent ({a2})", float(d2["objective"])))
    for arm in ARMS:
        b = (rec.get(arm) or {}).get("bound")
        if b is None:
            continue
        for label, ref in refs:
            checks += 1
            if s * b > s * ref + 1e-6 * max(1.0, abs(ref)):
                viol.append(f"{name} [{arm}]: bound {b!r} past {label} {ref!r}")
    rows.append(rec)

pathlib.Path("scratchpad/1141/panel_route_default.json").write_text(
    json.dumps(rows, indent=1, default=str))
print(f"\nload after {os.getloadavg()}")
print(f"{'arm':8s} {'certificates':>12s} {'incumbents':>11s} {'total wall':>11s}")
for arm in ARMS:
    cert = sum(1 for r in rows if str((r.get(arm) or {}).get("status")) == "optimal")
    inc = sum(1 for r in rows if (r.get(arm) or {}).get("objective") is not None)
    tw = sum((r.get(arm) or {}).get("wall", 0.0) for r in rows)
    print(f"{arm:8s} {cert:12d} {inc:11d} {tw:10.1f}s")
lost = [r["instance"] for r in rows
        if str((r.get("pre") or {}).get("status")) == "optimal"
        and str((r.get("inhouse") or {}).get("status")) != "optimal"]
noinc = [r["instance"] for r in rows
         if (r.get("pre") or {}).get("objective") is not None
         and (r.get("inhouse") or {}).get("objective") is None]
print(f"\nCERTIFICATES LOST vs pre : {len(lost)}  {lost}")
print(f"INCUMBENTS LOST vs pre   : {len(noinc)}  {noinc}")
print(f"EXECUTED SOUNDNESS CHECKS: {checks}   VIOLATIONS: {len(viol)}")
for v in viol:
    print("  !!", v)
if checks == 0:
    print("PANEL MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if viol else 0)
