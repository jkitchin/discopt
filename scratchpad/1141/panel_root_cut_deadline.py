"""Graduation panel for `DISCOPT_ROOT_CUT_DEADLINE` (#1141's added work item).

Unlike the other #1141 flags this one lives on the **default** solve path (the
root cutting-plane stage that `nlpbb_root_cuts_enabled()` turns on by default), so
the panel drives `model.solve()` rather than `mip_nlp_method="lp_nlp_bb"`.

Both §5 bars:

* cert-clean -- the ON arm's dual bound against BOTH arms' independently
  feasibility-verified incumbents and against `known_optima.toml` where an oracle
  exists, sense-aware; no certification regression;
* net-positive -- wall, bounds, certificates.

Anti-vacuity (§6) is measured, not assumed: `_root_cuts._solve_lp` is wrapped to
count the stage's LP calls, how many carried a deadline, and how many were
DECLINED (returned all-`None`, i.e. the limit actually bit). A row with zero
deadline-carrying calls did not exercise the flag and is reported as such.

Prints an executed-check count and exits non-zero if it is zero.
"""
import argparse, json, os, sys, time, pathlib
import numpy as np

sys.path.insert(0, "python/tests")
DATA_DIRS = [
    pathlib.Path("python/tests/data/minlplib_nl"),
    pathlib.Path("python/tests/data/minlplib"),
]

from discopt.modeling.core import from_nl                    # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt._tape_nlp_evaluator import make_evaluator       # noqa: E402
import discopt.solvers._root_cuts as rc                      # noqa: E402

try:
    from _optima import known_optimum
except Exception:
    def known_optimum(name, full=False):
        return None

STATS = {"stage_calls": 0, "lp_calls": 0, "deadline_calls": 0, "declined": 0}
_orig_solve_lp = rc._solve_lp
_orig_generate = rc.generate_root_cuts


def _wrapped_generate(*args, **kwargs):
    """Count STAGE entries too: `lp_calls == 0` alone cannot tell "the stage ran
    and solved nothing" from "the stage never ran", and the second is the case
    that makes a whole panel vacuous (CLAUDE.md §6)."""
    STATS["stage_calls"] += 1
    return _orig_generate(*args, **kwargs)


rc.generate_root_cuts = _wrapped_generate


def _wrapped_solve_lp(root, cuts_a, cuts_b, time_limit=None):
    STATS["lp_calls"] += 1
    if time_limit is not None:
        STATS["deadline_calls"] += 1
    out = _orig_solve_lp(root, cuts_a, cuts_b, time_limit)
    if out[1] is None:
        STATS["declined"] += 1
    return out


rc._solve_lp = _wrapped_solve_lp


def flat_point(model, xdict):
    if xdict is None:
        return None
    lb, _ = flat_variable_bounds(model)
    out = np.zeros(len(lb))
    k = 0
    for v in model._variables:
        if v.name not in xdict:
            return None
        out[k:k + v.size] = np.atleast_1d(np.asarray(xdict[v.name], float)).ravel()
        k += v.size
    return out


def max_violation(model, x):
    ev = make_evaluator(model)
    g = np.asarray(ev.evaluate_constraints(x), float)
    senses = [c.sense if isinstance(c.sense, str) else c.sense.value for c in model._constraints]
    worst = 0.0
    for gi, s in zip(g, senses):
        worst = max(worst, gi if s == "<=" else (-gi if s == ">=" else abs(gi)))
    lb, ub = flat_variable_bounds(model)
    if len(x):
        worst = max(worst, float(np.max(np.maximum(lb - x, x - ub))))
    return float(worst)


ap = argparse.ArgumentParser()
ap.add_argument("--time-limit", type=float, default=30.0)
ap.add_argument("--out", default="scratchpad/1141/panel_root_cut_deadline.json")
ap.add_argument("--only", default="")
a = ap.parse_args()

paths = {}
for d in DATA_DIRS:
    for p_ in sorted(d.glob("*.nl")):
        paths.setdefault(p_.stem, p_)
names = sorted(paths)
if a.only:
    names = [n for n in names if n in set(a.only.split(","))]
print(f"corpus: {len(names)} instances; flag DISCOPT_ROOT_CUT_DEADLINE; "
      f"DEFAULT solve path; load before {os.getloadavg()}", flush=True)
print(f"{'instance':22s} {'arm':4s} {'status':16s} {'objective':>18s} {'bound':>18s} "
      f"{'wall':>7s} {'feas':>9s}  stage-LPs", flush=True)

checks = 0
violations = []
rows = []
for name in names:
    rec = {"instance": name}
    for arm in ("off", "on"):
        os.environ["DISCOPT_ROOT_CUT_DEADLINE"] = "1" if arm == "on" else "0"
        try:
            model = from_nl(str(paths[name]))
        except Exception as exc:
            rec[arm] = {"error": f"load: {type(exc).__name__}: {exc}"}
            continue
        rec.setdefault("sense", 1.0 if model._objective.sense.name == "MINIMIZE" else -1.0)
        STATS.update(stage_calls=0, lp_calls=0, deadline_calls=0, declined=0)
        t = time.perf_counter()
        try:
            r = model.solve(time_limit=a.time_limit, gap_tolerance=1e-4)
            wall = time.perf_counter() - t
            x = flat_point(model, getattr(r, "x", None))
            rec[arm] = {
                "status": str(r.status), "objective": r.objective, "bound": r.bound,
                "wall": wall, "feas": None if x is None else max_violation(model, x),
                "nodes": getattr(r, "node_count", None), "stage": dict(STATS),
            }
        except Exception as exc:
            rec[arm] = {"error": f"{type(exc).__name__}: {exc}",
                        "wall": time.perf_counter() - t, "stage": dict(STATS)}
            print(f"{name:22s} {arm:4s} RAISED {type(exc).__name__}: {exc}", flush=True)
            continue
        d = rec[arm]
        ob = "None" if d["objective"] is None else f"{d['objective']:.10g}"
        bd = "None" if d["bound"] is None else f"{d['bound']:.10g}"
        fs = "None" if d["feas"] is None else f"{d['feas']:.2e}"
        print(f"{name:22s} {arm:4s} {d['status']:16s} {ob:>18s} {bd:>18s} "
              f"{d['wall']:7.2f} {fs:>9s}  {d['stage']}", flush=True)

    off, on = rec.get("off", {}), rec.get("on", {})
    sense = rec.get("sense", 1.0)
    for src, d in (("off", off), ("on", on)):
        obj, feas = d.get("objective"), d.get("feas")
        if obj is None or feas is None or feas > 1e-6 or on.get("bound") is None:
            continue
        checks += 1
        if sense * on["bound"] > sense * obj + 1e-6 * max(1.0, abs(obj)):
            violations.append(f"{name}: ON bound {on['bound']!r} past verified {src} "
                              f"incumbent {obj!r}")
    try:
        opt = known_optimum(name)
    except KeyError:
        opt = None
    if opt is not None and on.get("bound") is not None:
        checks += 1
        if sense * on["bound"] > sense * float(opt) + 1e-6 * max(1.0, abs(float(opt))):
            violations.append(f"{name}: ON bound {on['bound']!r} past ORACLE {opt!r}")
    if off.get("status") == "optimal":
        checks += 1
        if on.get("status") != "optimal":
            violations.append(f"{name}: certification regressed optimal -> {on.get('status')}")
    rows.append(rec)

pathlib.Path(a.out).write_text(json.dumps(rows, indent=1, default=str))
ran = sum(1 for r in rows if (r.get("on", {}).get("stage") or {}).get("stage_calls", 0) > 0)
fired = sum(1 for r in rows if (r.get("on", {}).get("stage") or {}).get("deadline_calls", 0) > 0)
print(f"\nROWS WHERE THE ROOT-CUT STAGE RAN AT ALL: {ran}/{len(rows)}")
bit = sum(1 for r in rows
          if (r.get("on", {}).get("stage") or {}).get("declined", 0)
          > (r.get("off", {}).get("stage") or {}).get("declined", 0))
print(f"\nload after {os.getloadavg()}")
print(f"ROWS WHERE THE STAGE RAN WITH A DEADLINE: {fired}/{len(rows)}")
print(f"ROWS WHERE THE DEADLINE ACTUALLY BIT (extra declined LP): {bit}/{len(rows)}")
print(f"EXECUTED SOUNDNESS CHECKS: {checks}   VIOLATIONS: {len(violations)}")
for v in violations:
    print("  !!", v)
if checks == 0:
    print("PANEL MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if violations else 0)
