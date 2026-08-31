"""Graduation-gate panel for DISCOPT_OA_NODE_CUTS (#1141), CLAUDE.md §5 regime 2.

For every in-repo MINLPLib instance, runs `mip_nlp_method="lp_nlp_bb"` on the
in-house simplex master with the flag OFF and ON, INTERLEAVED per instance, and
records:

* soundness -- the ON arm's dual bound against BOTH arms' incumbents and against
  `known_optima.toml` where an oracle exists (a bound above a feasible point is a
  false certificate, CLAUDE.md §1);
* incumbent feasibility, verified independently from the model's own evaluator,
  not taken on the solver's word;
* certification -- an instance that was `optimal` OFF must not regress;
* wall and bound, for the net-positive bar.

Prints per-instance rows as it goes (§10) and an executed-check count (§6).
"""
import argparse, json, os, sys, time, pathlib
import numpy as np

sys.path.insert(0, "python/tests")
DATA = pathlib.Path("python/tests/data/minlplib_nl")

from discopt.modeling.core import from_nl                    # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt._tape_nlp_evaluator import make_evaluator       # noqa: E402

try:
    from _optima import known_optimum
except Exception:
    def known_optimum(name, full=False):
        return None


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
    """Worst constraint violation of `x`, from the model's own evaluator."""
    ev = make_evaluator(model)
    g = np.asarray(ev.evaluate_constraints(x), float)
    senses = [c.sense if isinstance(c.sense, str) else c.sense.value for c in model._constraints]
    worst = 0.0
    for gi, s in zip(g, senses):
        worst = max(worst, gi if s == "<=" else (-gi if s == ">=" else abs(gi)))
    lb, ub = flat_variable_bounds(model)
    worst = max(worst, float(np.max(np.maximum(lb - x, x - ub))) if len(x) else 0.0)
    return float(worst)


ap = argparse.ArgumentParser()
ap.add_argument("--time-limit", type=float, default=30.0)
ap.add_argument("--out", default="scratchpad/1141/panel_corpus.json")
ap.add_argument("--only", default="")
a = ap.parse_args()

names = sorted(p.stem for p in DATA.glob("*.nl"))
if a.only:
    keep = set(a.only.split(","))
    names = [n for n in names if n in keep]

checks = 0
violations = []
rows = []
print(f"{'instance':22s} {'arm':4s} {'status':12s} {'objective':>18s} {'bound':>18s} "
      f"{'wall':>7s} {'feas':>9s}", flush=True)

for name in names:
    rec = {"instance": name}
    for arm in ("off", "on"):
        os.environ["DISCOPT_OA_NODE_CUTS"] = "1" if arm == "on" else "0"
        try:
            model = from_nl(str(DATA / f"{name}.nl"))
        except Exception as exc:
            rec[arm] = {"error": f"load: {type(exc).__name__}: {exc}"}
            print(f"{name:22s} {arm:4s} LOAD FAILED {exc}", flush=True)
            continue
        # Sense matters for every bound assertion below: for a MAXIMIZE model the
        # solver's `bound` is an UPPER bound, so `bound >= objective` is the sound
        # relation, not a violation. A sense-blind check flags every `syn*` row.
        sense = 1.0 if model._objective.sense.name == "MINIMIZE" else -1.0
        rec.setdefault("sense", sense)
        t = time.perf_counter()
        try:
            r = model.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb",
                            milp_solver="simplex", time_limit=a.time_limit,
                            gap_tolerance=1e-4)
            wall = time.perf_counter() - t
            x = flat_point(model, getattr(r, "x", None))
            feas = None if x is None else max_violation(model, x)
            cb = ((r.mip_nlp_trace or {}).get("summary", {}) or {}).get("callback_stats", {}) or {}
            rec[arm] = {"status": str(r.status), "objective": r.objective, "bound": r.bound,
                        "wall": wall, "feas": feas, "nodes": getattr(r, "node_count", None),
                        # Anti-vacuity per row (§6): an ON row with mipnode == 0 did
                        # not exercise the flag at all, and must not be counted as
                        # evidence either way.
                        "mipnode": cb.get("mipnode_calls"), "node_cuts": cb.get("driver_node_cuts")}
        except Exception as exc:
            wall = time.perf_counter() - t
            rec[arm] = {"error": f"{type(exc).__name__}: {exc}", "wall": wall}
            print(f"{name:22s} {arm:4s} RAISED {type(exc).__name__}: {exc}", flush=True)
            continue
        d = rec[arm]
        ob = "None" if d["objective"] is None else f"{d['objective']:.10g}"
        bd = "None" if d["bound"] is None else f"{d['bound']:.10g}"
        fs = "None" if d["feas"] is None else f"{d['feas']:.2e}"
        print(f"{name:22s} {arm:4s} {d['status']:12s} {ob:>18s} {bd:>18s} "
              f"{d['wall']:7.2f} {fs:>9s} mipnode={d.get('mipnode')}", flush=True)

    off, on = rec.get("off", {}), rec.get("on", {})
    sense = rec.get("sense", 1.0)
    # --- soundness: the ON bound may not exceed any VERIFIED feasible objective
    for src, d in (("off", off), ("on", on)):
        obj, feas = d.get("objective"), d.get("feas")
        if obj is None or feas is None or feas > 1e-6:
            continue
        if on.get("bound") is not None:
            checks += 1
            tol = 1e-6 * max(1.0, abs(obj))
            if sense * on["bound"] > sense * obj + tol:
                violations.append(f"{name}: ON bound {on['bound']!r} > verified {src} "
                                  f"incumbent {obj!r}")
    try:
        opt = known_optimum(name)
    except KeyError:
        opt = None
    if opt is not None and on.get("bound") is not None:
        checks += 1
        if sense * on["bound"] > sense * float(opt) + 1e-6 * max(1.0, abs(float(opt))):
            violations.append(f"{name}: ON bound {on['bound']!r} > ORACLE optimum {opt!r}")
    # --- certification regression
    if off.get("status") == "optimal":
        checks += 1
        if on.get("status") != "optimal":
            violations.append(f"{name}: certification regressed optimal -> {on.get('status')}")
    rows.append(rec)

pathlib.Path(a.out).write_text(json.dumps(rows, indent=1, default=str))
exercised = sum(1 for r in rows if (r.get("on", {}).get("mipnode") or 0) > 0)
print(f"\nROWS THAT EXERCISED THE FLAG: {exercised}/{len(rows)}")
print(f"EXECUTED SOUNDNESS CHECKS: {checks}   VIOLATIONS: {len(violations)}")
for v in violations:
    print("  !!", v)
if checks == 0:
    print("PANEL MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if violations else 0)
