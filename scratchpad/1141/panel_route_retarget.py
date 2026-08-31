"""#1141's headline row: should the certified-convex route still target HiGHS?

`_convex_minlp_auto_route` sends a certified-convex MIQP/MIQCP/MINLP to
`lp_nlp_bb` on the **HiGHS** master (`{"milp_solver": "highs"}`). That choice was
made in #1066 on rsyn0840m -- the in-house master's root loop closed 0% of the
gap where HiGHS closed 86.1%. It was measured BEFORE the fractional-node hook
existed, and a master's ability to close its own gap is exactly what that hook
changes, so the evidence behind the route is stale by construction. #1141's
headline complaint (the per-node gap on convex MINLPs) lives on this route, and
`DISCOPT_OA_NODE_CUTS` is INERT here while it points at HiGHS -- the HiGHS master
has no fractional-node hook at all.

Population is asked of the router itself rather than guessed: every corpus
instance is put through `_convex_minlp_auto_route` and only the rows it actually
diverts to `lp_nlp_bb`/highs are measured. Three arms, interleaved per instance:

  highs         -- what the route does today
  simplex       -- the in-house master, no fractional separation (the #1066 arm)
  simplex+node  -- the in-house master WITH the #1141 hook (the proposal)

Soundness is checked on every arm: incumbents are feasibility-verified from the
model's own evaluator, and no arm's dual bound may pass any arm's verified
incumbent or the reference optimum. Prints an executed-check count and exits
non-zero if it is zero (CLAUDE.md §6).
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "python/tests")
from _optima import known_optimum, optima_registry  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt._tape_nlp_evaluator import make_evaluator  # noqa: E402
from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: E402
from discopt.solver import _convex_minlp_auto_route  # noqa: E402

ARMS = {"highs": ("highs", "0"), "simplex": ("simplex", "0"),
        "simplex+node": ("simplex", "1")}
DIRS = [pathlib.Path("python/tests/data/minlplib_nl"),
        pathlib.Path("python/tests/data/minlplib")]
TL = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0


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

# Ask the router, do not guess the population.
routed = []
for name, p_ in paths.items():
    try:
        m = from_nl(str(p_))
        method, why, kw = _convex_minlp_auto_route(m)
    except Exception as exc:
        print(f"{name:22s} route probe raised {type(exc).__name__}: {exc}", flush=True)
        continue
    if method == "lp_nlp_bb" and kw.get("milp_solver") == "highs":
        routed.append(name)
print(f"instances the router diverts to lp_nlp_bb/highs: {len(routed)}/{len(paths)}")
print(" ", ", ".join(routed), flush=True)
if not routed:
    print("PANEL MEASURED NOTHING: router diverts no vendored instance", file=sys.stderr)
    sys.exit(1)

print(f"\nload before {os.getloadavg()}", flush=True)
print(f"{'instance':22s} {'arm':13s} {'status':11s} {'objective':>17s} {'bound':>17s} "
      f"{'wall':>7s} {'feas':>9s}", flush=True)
rows, checks, viol = [], 0, []
for name in routed:
    rec = {"instance": name}
    for arm, (backend, nodecuts) in ARMS.items():
        os.environ["DISCOPT_OA_NODE_CUTS"] = nodecuts
        m = from_nl(str(paths[name]))
        rec.setdefault("sense", 1.0 if m._objective.sense == ObjectiveSense.MINIMIZE
                       else -1.0)
        t = time.perf_counter()
        try:
            r = m.solve(time_limit=TL, gap_tolerance=1e-4,
                        mip_nlp_method="lp_nlp_bb", milp_solver=backend)
            wall = time.perf_counter() - t
            x = flat_point(m, getattr(r, "x", None))
            rec[arm] = {"status": str(r.status), "objective": r.objective,
                        "bound": r.bound, "wall": wall,
                        "feas": None if x is None else max_violation(m, x)}
        except Exception as exc:
            rec[arm] = {"error": f"{type(exc).__name__}: {exc}",
                        "wall": time.perf_counter() - t}
            print(f"{name:22s} {arm:13s} RAISED {type(exc).__name__}: {exc}", flush=True)
            continue
        d = rec[arm]
        ob = "None" if d["objective"] is None else f"{d['objective']:.10g}"
        bd = "None" if d["bound"] is None else f"{d['bound']:.10g}"
        fs = "None" if d["feas"] is None else f"{d['feas']:.2e}"
        print(f"{name:22s} {arm:13s} {d['status']:11s} {ob:>17s} {bd:>17s} "
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

pathlib.Path("scratchpad/1141/panel_route_retarget.json").write_text(
    json.dumps(rows, indent=1, default=str))
print(f"\nload after {os.getloadavg()}")
print(f"{'arm':13s} {'certificates':>12s} {'total wall':>11s}")
for arm in ARMS:
    cert = sum(1 for r in rows if str((r.get(arm) or {}).get("status")) == "optimal")
    tw = sum((r.get(arm) or {}).get("wall", 0.0) for r in rows)
    print(f"{arm:13s} {cert:12d} {tw:10.1f}s")
print(f"\nEXECUTED SOUNDNESS CHECKS: {checks}   VIOLATIONS: {len(viol)}")
for v in viol:
    print("  !!", v)
if checks == 0:
    print("PANEL MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if viol else 0)
