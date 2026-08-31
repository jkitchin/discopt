"""Bar 2 for DISCOPT_ROOT_CUT_DEADLINE: is `time_budget_s` actually enforced?

The 119-instance corpus panel cannot answer bar 2 for this flag. The flag's
benefit shows only where the OA prologue OUTRUNS the stage budget, and at the
budget solver.py hands the stage (2-10 s) no vendored instance does -- measured,
0/119 deadline bites. #1066 measured the pathology on rsyn0830m (one LP burned
81.3 s of a 150 s solve against a 10 s budget); rsyn* is not vendored here.

So measure the CONTRACT directly, over a budget range, on the real instances
that run the stage. This is a wall-clock contract ("does this function return
when it said it would"), not a bound-quality claim, so it reads the same
wherever it is exercised -- the #727 lesson (a mechanism validated on a
synthetic proxy can be a no-op on the real class) is about gains, not contracts.

ARGUMENTS COME FROM THE REAL CALLER. `generate_root_cuts`' docstring states a
caller contract -- "`lb`/`ub` are the FBBT-tightened root bounds" -- and the
first cut of this probe rebuilt them from `flat_variable_bounds` instead, i.e.
the RAW declared box. On cvxnonsep_psig40r that box leaves 42 of 82 columns
unbounded, the separators substitute a fake 1e5 for an infinite bound, and the
stage returned root bounds up to 32092 against a verified incumbent of 86.5 --
8 "violations" in BOTH arms that were the probe's contract breach, not a defect
in the stage. (The shipped path solves that instance to `optimal` at 86.539.)
So the stage is now patched to capture the arguments its real caller passes and
replay THOSE, which satisfies the contract by construction.

Soundness is still checked on every replay: a truncated stage may only WEAKEN
the root bound, never invalidate it, so each `lp_bound` is checked against the
reference optimum and against the corpus panel's independently
feasibility-verified incumbents.

Prints an executed-check count and exits non-zero if it is zero (CLAUDE.md §6).
"""
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, "python/tests")
import discopt.solvers._root_cuts as rc  # noqa: E402
from _optima import known_optimum, optima_registry  # noqa: E402
from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: E402

assert "n_le_basis" in pathlib.Path(rc.__file__).read_text(), "fix not loaded"
print("module:", rc.__file__, "(fix marker present)", flush=True)

NAMES = (sys.argv[1] if len(sys.argv) > 1 else
         "clay0303hfsg,cvxnonsep_nsig30,cvxnonsep_psig30,cvxnonsep_psig40r,"
         "fac2,flay02m,flay03m,m3,nvs03,tls2").split(",")
BUDGETS = [float(b) for b in (sys.argv[2].split(",") if len(sys.argv) > 2
                              else ["0.25", "0.5", "1.0", "2.0", "5.0"])]
D = pathlib.Path("python/tests/data/minlplib_nl")
_orig_generate = rc.generate_root_cuts


def verified_incumbents(path="scratchpad/1141/panel_deadline_fixed.json"):
    """Second oracle: `known_optima.toml` covers 1 of these 10 instances.

    The 30 s corpus panel recorded, per instance, an objective whose point was
    feasibility-verified from the model's own evaluator (`feas <= 1e-6`). A
    feasible objective bounds the optimum, so a root LP bound may never pass it.
    """
    out = {}
    for r in json.loads(pathlib.Path(path).read_text()):
        best, s_ = None, r.get("sense", 1.0)
        for arm in ("off", "on"):
            d = r.get(arm) or {}
            o, f = d.get("objective"), d.get("feas")
            if o is None or f is None or f > 1e-6:
                continue
            if best is None or s_ * o < s_ * best:
                best = float(o)
        if best is not None:
            out[r["instance"]] = best
    return out


INCUMBENTS = verified_incumbents()
print(f"verified incumbents available for {len(INCUMBENTS)} instances", flush=True)
print(f"load before {os.getloadavg()}", flush=True)
print(f"{'instance':18s} {'budget':>7s} {'arm':4s} {'wall':>8s} {'over':>8s} "
      f"{'cuts':>5s} {'stop':>8s} {'lp_bound':>18s}", flush=True)

rows, checks, viol = [], 0, []
overrun = {"off": 0, "on": 0}
attempts = {"off": 0, "on": 0}
replayed = 0


def make_capture(name, sense, opt, inc):
    """Replay the stage at each budget in both arms, with the REAL caller's args."""
    def _capture(model, evaluator, lb, ub, is_int, is_bin, time_budget_s=10.0):
        global replayed, checks
        if _capture.done:                       # top-level solve only, once
            return _orig_generate(model, evaluator, lb, ub, is_int, is_bin,
                                  time_budget_s=time_budget_s)
        _capture.done = True
        for budget in BUDGETS:
            rec = {"instance": name, "budget": budget}
            for arm in ("off", "on"):
                os.environ["DISCOPT_ROOT_CUT_DEADLINE"] = "1" if arm == "on" else "0"
                t = time.perf_counter()
                res = _orig_generate(model, evaluator, lb, ub, is_int, is_bin,
                                     time_budget_s=budget)
                wall = time.perf_counter() - t
                replayed += 1
                attempts[arm] += 1
                over = wall - budget
                # 20% past its own budget is the #1066 defect showing, not jitter.
                if over > 0.2 * budget:
                    overrun[arm] += 1
                d = {"wall": wall, "cuts": len(res.cuts), "lp_bound": res.lp_bound,
                     "rounds": res.rounds_run, "stop": res.stop_reason}
                rec[arm] = d
                bd = "None" if res.lp_bound is None else f"{res.lp_bound:.10g}"
                print(f"{name:18s} {budget:7.2f} {arm:4s} {wall:8.2f} {over:+8.2f} "
                      f"{len(res.cuts):5d} {res.stop_reason:>8s} {bd:>18s}", flush=True)
                if res.lp_bound is None:
                    continue
                for label, ref in (("ORACLE", opt), ("verified incumbent", inc)):
                    if ref is None:
                        continue
                    checks += 1
                    if sense * res.lp_bound > sense * ref + 1e-6 * max(1.0, abs(ref)):
                        viol.append(f"{name}@{budget}s {arm}: root bound "
                                    f"{res.lp_bound!r} past {label} {ref!r}")
            rows.append(rec)
        os.environ["DISCOPT_ROOT_CUT_DEADLINE"] = "0"
        return _orig_generate(model, evaluator, lb, ub, is_int, is_bin,
                              time_budget_s=time_budget_s)
    _capture.done = False
    return _capture


os.environ["DISCOPT_CONVEX_MINLP_ROUTE"] = "0"   # else the stage's owner never runs
for name in NAMES:
    if not (D / f"{name}.nl").exists():
        print(f"{name}: MISSING", flush=True)
        continue
    m = from_nl(str(D / f"{name}.nl"))
    sense = 1.0 if m._objective.sense == ObjectiveSense.MINIMIZE else -1.0
    opt = float(known_optimum(name)) if name in optima_registry() else None
    cap = make_capture(name, sense, opt, INCUMBENTS.get(name))
    rc.generate_root_cuts = cap
    try:
        m.solve(time_limit=30, gap_tolerance=1e-4)
    finally:
        rc.generate_root_cuts = _orig_generate
    if not cap.done:
        print(f"{name}: stage never ran; nothing measured on this row", flush=True)

pathlib.Path("scratchpad/1141/panel_budget_contract.json").write_text(
    json.dumps(rows, indent=1, default=str))
print(f"\nload after {os.getloadavg()}")
print(f"stage replays: {replayed}")
for arm in ("off", "on"):
    print(f"{arm:3s}: stage runs {attempts[arm]:3d}   OVERRAN its budget by >20%: "
          f"{overrun[arm]:3d}")
print(f"EXECUTED SOUNDNESS CHECKS: {checks}   VIOLATIONS: {len(viol)}")
for v in viol:
    print("  !!", v)
if checks == 0 or attempts["on"] == 0:
    print("PANEL MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if viol else 0)
