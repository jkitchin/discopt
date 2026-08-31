"""#1141 item 3 entry experiment: WHY does OA feasibility restoration never converge?

Hypothesis: `_FeasibilityEvaluator` poses restoration as an UNCONSTRAINED
minimization of a violation merit and reports a ZERO Hessian for it. With no
constraints an interior-point method's KKT matrix is `sigma_f * grad^2 f + Sigma`,
and the first term is identically zero, so away from the variable bounds the
matrix is numerically singular -- which is the reported failure,
`Error_In_Step_Computation` (code -3), 57 of 60 on `portfol_classical050_1`. It
also predicts that changing the merit NORM changes nothing, which #1141 measured.

Kill criterion: dead if `_ElasticFeasibilityEvaluator` -- the constrained elastic
reformulation, smooth, with the original Lagrangian Hessian and a feasible start --
converges no more often and reaches no lower violation.

Every restoration the OA loop actually requests, on real corpus instances, is
replayed through both. Prints an executed-comparison count (§6); a replay that
raises propagates (§7).
"""
import collections, json, os, pathlib, sys
import numpy as np

import discopt.solvers.oa as oa
from discopt.modeling.core import from_nl
from discopt.solvers import pounce_option_defaults
from discopt.solvers.nlp_pounce import solve_nlp

DIRS = [pathlib.Path("python/tests/data/minlplib_nl"),
        pathlib.Path("python/tests/data/minlplib")]
paths = {}
for d in DIRS:
    for p in sorted(d.glob("*.nl")):
        paths.setdefault(p.stem, p)

records = []
CUR = {}
_orig = oa._solve_feasibility_subproblem


def replay(evaluator, sub_lb, sub_ub, x0, norm):
    opts = pounce_option_defaults()
    opts.update({"max_iter": 300})

    merit = oa._FeasibilityEvaluator(evaluator, sub_lb, sub_ub, norm)
    a = solve_nlp(merit, x0, options=dict(opts))
    a_x = np.clip(np.asarray(a.x, float), sub_lb, sub_ub)
    a_merit = oa._constraint_violation_merit(evaluator, a_x, norm)

    el = oa._ElasticFeasibilityEvaluator(evaluator, sub_lb, sub_ub, norm)
    b = solve_nlp(el, el.start_point(x0),
                  constraint_bounds=el.constraint_bounds(), options=dict(opts))
    b_x = np.clip(np.asarray(b.x, float)[: evaluator.n_variables], sub_lb, sub_ub)
    b_merit = oa._constraint_violation_merit(evaluator, b_x, norm)
    return a.raw_status, float(a_merit), b.raw_status, float(b_merit)


def wrapped(evaluator, lb, ub, int_indices, x_master, nlp_solver, feasibility_norm,
            max_wall_time=None):
    out = _orig(evaluator, lb, ub, int_indices, x_master, nlp_solver, feasibility_norm,
                max_wall_time=max_wall_time)
    if len(records) >= CUR.get("cap", 10**9):
        return out
    lb_a, ub_a = np.asarray(lb, float), np.asarray(ub, float)
    sub_lb, sub_ub = lb_a.copy(), ub_a.copy()
    xm = np.asarray(x_master, float)
    for idx in int_indices:
        v = oa._round_integral_to_bounds(xm[idx], lb_a[idx], ub_a[idx])
        sub_lb[idx] = v
        sub_ub[idx] = v
    x0 = np.clip(xm[: evaluator.n_variables], sub_lb, sub_ub)
    start = oa._constraint_violation_merit(evaluator, x0, feasibility_norm)
    a_raw, a_merit, b_raw, b_merit = replay(evaluator, sub_lb, sub_ub, x0, feasibility_norm)
    records.append(dict(instance=CUR["name"], norm=feasibility_norm, start=float(start),
                        a_raw=a_raw, a_merit=a_merit, b_raw=b_raw, b_merit=b_merit))
    return out


oa._solve_feasibility_subproblem = wrapped

names = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(paths)
CUR["cap"] = int(os.environ.get("RESTORATION_CAP", "400"))
for name in names:
    if name not in paths or len(records) >= CUR["cap"]:
        continue
    before = len(records)
    CUR["name"] = name
    try:
        m = from_nl(str(paths[name]))
    except Exception:
        continue
    os.environ["DISCOPT_OA_NODE_CUTS"] = "0"
    os.environ["DISCOPT_OA_ELASTIC_RESTORATION"] = "0"  # arm A is the shipped path
    try:
        m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
                time_limit=15, gap_tolerance=1e-4)
    except Exception as exc:
        print(f"{name}: solve raised {type(exc).__name__}: {exc}", flush=True)
    if len(records) > before:
        print(f"{name}: {len(records)-before} restorations replayed", flush=True)

print(f"\nEXECUTED RESTORATION COMPARISONS: {len(records)}")
if not records:
    print("PROBE MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
pathlib.Path("scratchpad/1141/restoration_records.json").write_text(
    json.dumps(records, indent=1, default=str))

a_codes = collections.Counter(r["a_raw"] for r in records)
b_codes = collections.Counter(r["b_raw"] for r in records)
a_ok = sum(1 for r in records if r["a_raw"] in (0, 1))
b_ok = sum(1 for r in records if r["b_raw"] in (0, 1))
print(f"instances contributing: {len(set(r['instance'] for r in records))}")
print(f"merit-formulation status codes:  {dict(a_codes)}")
print(f"elastic-formulation status codes: {dict(b_codes)}")
print(f"CONVERGED: merit {a_ok}/{len(records)}   elastic {b_ok}/{len(records)}")
better = sum(1 for r in records if r["b_merit"] < r["a_merit"] - 1e-9)
worse = sum(1 for r in records if r["b_merit"] > r["a_merit"] + 1e-9)
print(f"violation reached: elastic lower on {better}, higher on {worse}, "
      f"tied on {len(records)-better-worse}")
a_red = sum(1 for r in records if r["a_merit"] < r["start"] - 1e-9)
b_red = sum(1 for r in records if r["b_merit"] < r["start"] - 1e-9)
print(f"strictly improved on the clipped master point: merit {a_red}, elastic {b_red}")
