"""#2142 exact-hull: exactness checks of the prototype lowerings (PREREG.md, "Exactness checks").

  PYTHONHASHSEED=0 PYTHONPATH=/home/user/discopt/python python -u check.py [ARMS...]

1. big-M certified optimum x* of GDPlib cstr (shipped default path).
2. Six fixed selector configurations (j active reactors 1..j, recycle at r), each solved under
   big-M (reference) -- the configuration of x* is one of them.
   (each configuration is a nonconvex NLP that does not certify in the budget, so the
   comparison is between the two incumbents and the two bounds, printed).
3. Per arm: x* mapped to the lowered columns satisfies every lowered row; each configuration's
   optimum equals big-M's and projects to a point feasible for the original rows and the selected
   disjuncts' rows; convex combinations of lowered feasible points from different configurations
   satisfy every LINEAR lowered row.
Every row is evaluated by a numpy walk of the expression DAG written here (no solver code).
Prints an executed-assertion count (CLAUDE.md §6); nothing is caught (§7)."""
import itertools
import os
import re
import sys
import time

import numpy as np

import discopt
import discopt._relax.gdp_reformulate as G
from discopt.modeling.core import (
    BinaryOp, Constant, Constraint, FunctionCall, IndexExpression, UnaryOp, Variable,
)

assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
n_assert = 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_hull_proto as P  # noqa: E402
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
import gdplib.cstr as cstr  # noqa: E402

ARMS = sys.argv[1:] or ["homog", "lifted"]
ORACLE = 3.0620073
CONFIGS = [(5, 5), (1, 1), (2, 1), (3, 3)]  # (active reactors 1..j, recycle r); x* is (5, 5)


def ev(e, vals):
    if isinstance(e, Variable):
        return float(vals[e.name])
    if isinstance(e, Constant):
        v = np.asarray(e.value, dtype=float)
        assert v.size == 1, v.shape
        return float(v.ravel()[0])
    if isinstance(e, IndexExpression):
        raise NotImplementedError("IndexExpression")
    if isinstance(e, BinaryOp):
        a, b = ev(e.left, vals), ev(e.right, vals)
        if e.op == "+":
            return a + b
        if e.op == "-":
            return a - b
        if e.op == "*":
            return a * b
        if e.op == "/":
            return a / b
        if e.op == "**":
            return a**b
        raise NotImplementedError(e.op)
    if isinstance(e, UnaryOp):
        assert e.op == "neg", e.op
        return -ev(e.operand, vals)
    if isinstance(e, FunctionCall):
        raise NotImplementedError(e.func_name)
    terms = G._sumover_terms(e)
    if terms is not None:
        return sum(ev(t, vals) for t in terms)
    raise NotImplementedError(type(e).__name__)


def violation(con: Constraint, vals) -> float:
    r = ev(con.body, vals) - float(np.max(con.rhs))
    if con.sense == "<=":
        return max(r, 0.0)
    if con.sense == ">=":
        return max(-r, 0.0)
    return abs(r)


def max_violation(cons, vals, linear_only=False):
    worst, name = 0.0, None
    for c in cons:
        if not isinstance(c, Constraint):
            continue
        if linear_only and not G._is_linear(c.body):
            continue
        v = violation(c, vals)
        if v > worst:
            worst, name = v, c.name
    return worst, name


def fix_config(model, j, r):
    byname = {v.name: v for v in model._variables}
    for n in range(1, 6):
        byname[f"YP_is_cstr[{n}]_active"].fix(1.0 if n <= j else 0.0)
        byname[f"YP_is_bypass[{n}]_active"].fix(0.0 if n <= j else 1.0)
        byname[f"YR_is_recycle[{n}]_active"].fix(1.0 if n == r else 0.0)
        byname[f"YR_is_not_recycle[{n}]_active"].fix(0.0 if n == r else 1.0)


def values(model, res):
    return {v.name: float(np.asarray(res.value(v)).ravel()[0]) for v in model._variables}


def solve(model, max_nodes=2000, time_limit=120, **kw):
    t = time.perf_counter()
    res = model.solve(time_limit=time_limit, deterministic=True, max_nodes=max_nodes, **kw)
    return res, time.perf_counter() - t


# ---- 1. x* under big-M ----------------------------------------------------------------
import json
m = convert_pyomo_gdp(cstr.build_model()).model
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_ref.json")
cached = json.load(open(REF)) if os.path.exists(REF) else None
if cached is None:
    res_star, w = solve(m, gdp_method="big-m", max_nodes=150000, time_limit=600)
    print(f"big-M default: status={res_star.status} obj={res_star.objective} bound={res_star.bound} "
          f"nodes={res_star.node_count} wall={w:.0f}s", flush=True)
    assert abs(res_star.objective - ORACLE) <= 1e-4, res_star.objective; n_assert += 1
    x_star = values(m, res_star)
else:
    print("big-M references read from check_ref.json (written by an earlier run of this script)")
    x_star = cached["x_star"]
j_star = sum(int(round(x_star[f"YP_is_cstr[{n}]_active"])) for n in range(1, 6))
r_star = [n for n in range(1, 6) if round(x_star[f"YR_is_recycle[{n}]_active"]) == 1]
print(f"x*: {j_star} active reactors, recycle at {r_star}")
assert len(r_star) == 1; n_assert += 1
if (j_star, r_star[0]) not in CONFIGS:
    CONFIGS[0] = (j_star, r_star[0])
viol_star, _ = max_violation(m._constraints, x_star)
print(f"x* max violation on the original plain rows: {viol_star:.2e}")

# ---- 2. the configurations under big-M -----------------------------------------------
ref = {}
for j, r in CONFIGS:
    if cached is not None:
        ref[(j, r)] = tuple(cached["ref"][f"{j},{r}"])
        print(f"big-M config j={j} r={r} (cached): status={ref[(j, r)][0]} obj={ref[(j, r)][1]} "
              f"bound={ref[(j, r)][2]}")
        continue
    mc = m.clone()
    fix_config(mc, j, r)
    res, w = solve(mc, gdp_method="big-m")
    ref[(j, r)] = (res.status, res.objective, res.bound)
    print(f"big-M config j={j} r={r}: status={res.status} obj={res.objective} bound={res.bound} "
          f"nodes={res.node_count} wall={w:.0f}s", flush=True)
if cached is None:
    json.dump({"x_star": x_star, "ref": {f"{j},{r}": list(v) for (j, r), v in ref.items()}},
              open(REF, "w"))
assert ref[(j_star, r_star[0])][1] is not None; n_assert += 1
assert abs(ref[(j_star, r_star[0])][1] - ORACLE) <= 1e-4; n_assert += 1

# ---- 3. per arm ------------------------------------------------------------------------
plan = G._collect_indicator_hull_groups(m)
copy_re = re.compile(r"^_hull_(?P<prefix>.+)_v_(?P<vname>.+)_(?P<k>\d+)$")


def lowered_point(low, x, arm):
    """Map an original point (dict) to the lowered columns: copies, auxes, ratio columns."""
    # the hull sanitizes the group key into the variable names (':' -> '_')
    sels = {key.replace(":", "_"): [s.name for s in sels] for key, (dc, sels) in plan.groups.items()}
    vals = dict(x)
    for v in low._variables:
        if v.name in vals:
            continue
        mm = copy_re.match(v.name)
        if mm:
            key, vname, k = mm["prefix"], mm["vname"], int(mm["k"])
            y = x[sels[key][k]]
            vals[v.name] = x[vname] if round(y) == 1 else 0.0
    # auxes (lifted): w_{a_b_c} = y * prod t = prod of copies when y = 1, else 0
    for v in low._variables:
        if v.name in vals:
            continue
        mm = re.match(r"^_hull_(?P<prefix>.+)_(?P<k>\d+)_w_(?P<mono>.+)$", v.name)
        if mm and arm == "lifted":
            key, k, mono = mm["prefix"], int(mm["k"]), mm["mono"]
            y = x[sels[key][k]]
            names = mono.split("_")
            # variable names carry no '_' in cstr; re-split on the copies' names
            vals[v.name] = float(np.prod([x[n] for n in names])) if round(y) == 1 else 0.0
            continue
        mm = re.match(r"^_hull_(?P<prefix>.+)_(?P<k>\d+)_t0_(?P<vname>.+)$", v.name)
        if mm and arm == "ratio":
            key, k, vname = mm["prefix"], int(mm["k"]), mm["vname"]
            y = x[sels[key][k]]
            lo, hi = float(np.min(v.lb)), float(np.max(v.ub))
            vals[v.name] = x[vname] if round(y) == 1 else 0.5 * (lo + hi)
            continue
    missing = [v.name for v in low._variables if v.name not in vals]
    assert not missing, missing[:5]
    return vals


def disjunct_rows_at(x):
    """The original rows that must hold at x: plain rows + rows of every selected disjunct."""
    rows = [c for c in m._constraints if isinstance(c, Constraint)]
    for key, (dc, sels) in plan.groups.items():
        for k, d in enumerate(dc.disjuncts):
            if round(x[sels[k].name]) == 1:
                rows.extend(it for it in d if isinstance(it, Constraint))
    return rows


for arm in ARMS:
    print(f"\n===== arm {arm} =====", flush=True)
    P.install(arm)
    low = G.reformulate_gdp(m, method="hull")
    print(f"lowered: vars {len(low._variables)} rows {len(low._constraints)} stats {P.STATS}")
    assert P.STATS["rows"] == (5 if arm != "ratio" else 0); n_assert += 1
    # 3a. x* in the lowered model
    vals = lowered_point(low, x_star, arm)
    viol, name = max_violation(low._constraints, vals)
    print(f"[3a] x* on every lowered row: max violation {viol:.2e} ({name})")
    assert viol <= 10 * max(viol_star, 1e-7), (viol, name); n_assert += 1
    # 3b. fixed configurations
    points = {}
    for j, r in CONFIGS:
        lc = low.clone()
        fix_config(lc, j, r)
        res, w = solve(lc)
        st, obj_ref, bnd_ref = ref[(j, r)]
        print(f"[3b] j={j} r={r}: arm status={res.status} obj={res.objective} bound={res.bound} "
              f"nodes={res.node_count} wall={w:.0f}s | big-M status={st} obj={obj_ref} "
              f"bound={bnd_ref}", flush=True)
        if res.status == "infeasible" or st == "infeasible":
            assert res.status == st, (res.status, st); n_assert += 1
            continue
        if res.objective is not None and obj_ref is not None:
            assert abs(res.objective - obj_ref) <= 1e-4 * max(1.0, abs(obj_ref)), (
                res.objective, obj_ref); n_assert += 1
        if res.objective is not None:
            lv = values(lc, res)
            points[(j, r)] = lv
            xp = {v.name: lv[v.name] for v in m._variables}
            viol, name = max_violation(disjunct_rows_at(xp), xp)
            print(f"      projection on the original + selected-disjunct rows: {viol:.2e} ({name})")
            assert viol <= 1e-5, (viol, name); n_assert += 1
    P.uninstall()
    # 3c. convex combinations of lowered feasible points satisfy the linear lowered rows
    lin = [c for c in low._constraints if G._is_linear(c.body)]
    n_combo = 0
    for (a, pa), (b, pb) in itertools.combinations(points.items(), 2):
        for lam in (0.25, 0.5, 0.75):
            mix = {n: lam * pa[n] + (1 - lam) * pb[n] for n in pa}
            viol, name = max_violation(lin, mix)
            assert viol <= 1e-5, (a, b, lam, viol, name); n_assert += 1
            n_combo += 1
    print(f"[3c] {n_combo} convex combinations of {len(points)} lowered points satisfy all "
          f"{len(lin)} linear lowered rows")
    assert n_combo > 0; n_assert += 1

print(f"\nexecuted assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
