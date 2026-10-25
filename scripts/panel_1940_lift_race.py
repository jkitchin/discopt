"""#1940 graduation panel: the integer-bilinear lift race (``DISCOPT_IPX_RACE``) ON vs OFF.

OFF is the legacy rule -- adopt the lift whenever it yields a pure MILP. ON runs
the race in ``solver._ipx_lift_race``: a 5 s lifted slice, a 5 s un-lifted slice,
then the arm with the smaller gap, all three merged.

The panel covers every class #1940 names:

* the 12 in-repo instances the lift is adopted on (``ex1263``-``ex1266``,
  ``nvs02/10/11/12/14/15``, ``prob02/03``);
* dense and sparse objective-only integer QPs
  (``sum_{i<j} Q_ij x_i x_j + sum x``, ``x in [0,10]^n``, ``Q ~ N(0,1)``,
  ``sum x >= n``), and the dense family with a combinatorial linear side;
* constraint-product families (a dense product sum in a ``<=`` or ``>=`` row);
* objective-only integer x continuous products, and a knapsack with an
  objective-only ``k*y`` product;
* the #1937 PWL counterexample (``test_outer_with_integer_inputs_and_remaining_
  bilinear_term``: a PWL MILP plus one ``n*x`` product in the objective);
* array-variable models (written with ``shape=``), which the lift reaches only
  under ``DISCOPT_INTEGER_PRODUCT_INDEXED=1`` -- they are solved with that flag
  set in both arms, so they measure the race on the class #1934 wants to add.

Cert-clean checks, per solve: every incumbent is feasible on the ORIGINAL model
(re-evaluated here, not taken from the solver); every dual bound is no better than
the best objective any certified run reached; every certified objective agrees
with it; and no instance certified by OFF is uncertified by ON.

Usage: ``python -u scripts/panel_1940_lift_race.py [--time-limit 30] [--reps 2]``.
Prints per-solve progress and an executed-check count; exits non-zero on any
failure or when nothing was checked (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
from discopt._point_eval import compile_point_fn
from discopt.modeling import from_nl

# CLAUDE.md §8: the version under test carries the race.
assert hasattr(solver_mod, "_ipx_lift_race"), solver_mod.__file__
print("solver loaded from", solver_mod.__file__, flush=True)

FLAG = "DISCOPT_IPX_RACE"
INDEXED = "DISCOPT_INTEGER_PRODUCT_INDEXED"
DATA = os.path.join(os.path.dirname(__file__), "..", "python", "tests", "data")

CHECKS = 0
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)
        print("  FAIL:", msg, flush=True)


# ---------------------------------------------------------------- instances


def dense_qp(n, seed, pairs_per_var=None, extra_lin=False):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"qp_n{n}_s{seed}")
    x = [m.integer(f"x{i}", lb=0, ub=10) for i in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if pairs_per_var is not None:
        k = int(round(pairs_per_var * n))
        pairs = [pairs[t] for t in rng.choice(len(pairs), size=k, replace=False)]
    m.minimize(sum(float(rng.normal()) * x[i] * x[j] for i, j in pairs) + sum(x))
    m.subject_to(sum(x) >= n)
    if extra_lin:
        z = [m.binary(f"z{i}") for i in range(n)]
        for i in range(n):
            m.subject_to(x[i] <= 10 * z[i])
        w = rng.uniform(1, 5, n)
        m.subject_to(sum(float(w[i]) * z[i] for i in range(n)) <= float(w.sum() * 0.6))
        m.subject_to(sum(z) <= int(0.7 * n))
    return m


def obj_ic(n, seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"objIC{n}")
    h = n // 2
    x = [m.integer(f"x{i}", lb=0, ub=10) for i in range(h)]
    y = [m.continuous(f"y{i}", lb=0, ub=10) for i in range(h)]
    m.minimize(
        sum(float(rng.normal()) * x[i] * y[j] for i in range(h) for j in range(h)) + sum(x) + sum(y)
    )
    m.subject_to(sum(x) + sum(y) >= n)
    return m


def con_le(n, seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"con{n}")
    x = [m.integer(f"x{i}", lb=0, ub=10) for i in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m.subject_to(sum(float(rng.normal()) * x[i] * x[j] for i, j in pairs) <= 10.0 * n)
    m.minimize(-sum(float(rng.uniform(0.5, 2)) * x[i] for i in range(n)))
    return m


def con_ge(n, seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"conge{n}")
    x = [m.integer(f"x{i}", lb=0, ub=10) for i in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m.subject_to(sum(float(rng.uniform(0, 1)) * x[i] * x[j] for i, j in pairs) >= 5.0 * n * n)
    m.minimize(sum(float(rng.uniform(0.5, 2)) * x[i] for i in range(n)))
    return m


def knap_obj(n, seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"knapobj{n}")
    z = [m.binary(f"z{i}") for i in range(n)]
    k = m.integer("k", lb=0, ub=8)
    y = m.continuous("y", lb=0, ub=5)
    w = rng.uniform(1, 10, n)
    v = rng.uniform(1, 10, n)
    m.subject_to(sum(float(w[i]) * z[i] for i in range(n)) <= float(w.sum() / 3) + k)
    m.subject_to(sum(z) >= 3)
    m.subject_to(y >= 1 + 0.1 * sum(z))
    m.minimize(-sum(float(v[i]) * z[i] for i in range(n)) + 2.0 * k * y - 3 * y)
    return m


def pwl_case():
    m = dm.Model("pwl_nx")
    n = m.integer("n", lb=0, ub=5)
    x = m.continuous("x", lb=0.5, ub=3)
    m.minimize((n - 2.3) ** 2 + x * n / 4 + 1 / x)
    return dm.nonlinear_to_pwl(m), m


def array_qp(n, seed):
    """Dense objective-only integer QP written with an array variable."""
    rng = np.random.default_rng(seed)
    m = dm.Model(f"arrqp_n{n}")
    x = m.integer("x", shape=(n,), lb=0, ub=10)
    m.minimize(
        sum(float(rng.normal()) * x[i] * x[j] for i in range(n) for j in range(i + 1, n))
        + sum(x[i] for i in range(n))
    )
    m.subject_to(sum(x[i] for i in range(n)) >= n)
    return m


def array_mixed(n, seed):
    """Integer x continuous array products in rows (the #1934 panel's mixed class)."""
    rng = np.random.default_rng(seed)
    m = dm.Model(f"arrmixed_n{n}")
    x = m.integer("x", shape=(n,), lb=0, ub=5)
    y = m.continuous("y", shape=(n,), lb=0.0, ub=5.0)
    for i in range(n):
        m.subject_to(x[i] * y[i] >= float(rng.uniform(1.0, 4.0)))
    m.subject_to(sum(x[i] * x[(i + 1) % n] for i in range(n)) <= float(n * 5))
    cx = rng.uniform(0.5, 3.0, size=n)
    cy = rng.uniform(0.5, 3.0, size=n)
    m.minimize(sum(float(cx[i]) * x[i] + float(cy[i]) * y[i] for i in range(n)))
    return m


def _nl(name):
    for d in ("minlplib", "minlplib_nl"):
        p = os.path.join(DATA, d, f"{name}.nl")
        if os.path.exists(p):
            return from_nl(p)
    raise FileNotFoundError(name)


CORPUS = "ex1263 ex1264 ex1265 ex1266 nvs02 nvs10 nvs11 nvs12 nvs14 nvs15 prob02 prob03"

#: name -> (builder, needs DISCOPT_INTEGER_PRODUCT_INDEXED). A builder returns
#: (thing to solve, model to verify against).
INSTANCES: dict = {n: ((lambda n=n: (_nl(n),) * 2), False) for n in CORPUS.split()}
for _n in (12, 15, 18, 20):
    for _s in (0, 1):
        INSTANCES[f"dense_n{_n}_s{_s}"] = ((lambda n=_n, s=_s: (dense_qp(n, s),) * 2), False)
for _n in (15, 25, 40):
    INSTANCES[f"sparse_n{_n}"] = ((lambda n=_n: (dense_qp(n, 0, pairs_per_var=2.0),) * 2), False)
for _n in (12, 15, 18):
    INSTANCES[f"denselin_n{_n}"] = ((lambda n=_n: (dense_qp(n, 0, extra_lin=True),) * 2), False)
for _n in (8, 12, 15, 20):
    INSTANCES[f"con{_n}"] = ((lambda n=_n: (con_le(n, 0),) * 2), False)
for _n in (10, 12, 15, 20):
    INSTANCES[f"conge{_n}"] = ((lambda n=_n: (con_ge(n, 0),) * 2), False)
for _n in (10, 14):
    INSTANCES[f"objIC{_n}"] = ((lambda n=_n: (obj_ic(n, 0),) * 2), False)
for _n in (20, 40):
    INSTANCES[f"knapobj{_n}"] = ((lambda n=_n: (knap_obj(n, 0),) * 2), False)
INSTANCES["pwl_nx"] = (pwl_case, False)
for _n in (12, 15):
    INSTANCES[f"arrqp_n{_n}"] = ((lambda n=_n: (array_qp(n, 0),) * 2), True)
for _n in (4, 6):
    INSTANCES[f"arrmixed_n{_n}"] = ((lambda n=_n: (array_mixed(n, 0),) * 2), True)


# ---------------------------------------------------------------- checks


def _x_flat(model, r):
    return np.concatenate(
        [np.asarray(r.x[v.name], dtype=float).reshape(-1) for v in model._variables]
    )


def _infeasibility(model, x, tol=1e-6):
    """Largest violation of the ORIGINAL model's rows, bounds and integrality."""
    worst = 0.0
    off = 0
    for v in model._variables:
        lo = np.broadcast_to(np.asarray(v.lb, dtype=float), v.shape).reshape(-1)
        hi = np.broadcast_to(np.asarray(v.ub, dtype=float), v.shape).reshape(-1)
        xv = x[off : off + v.size]
        off += v.size
        worst = max(worst, float(np.max(lo - xv, initial=0)), float(np.max(xv - hi, initial=0)))
        if v.var_type.name in ("INTEGER", "BINARY"):
            worst = max(worst, float(np.max(np.abs(xv - np.round(xv)), initial=0)))
    for c in model._constraints:
        val = np.asarray(compile_point_fn(c.body, model)(x), dtype=float).reshape(-1)
        rhs = float(np.asarray(c.rhs)) if c.rhs is not None else 0.0
        scale = max(1.0, abs(rhs))
        if c.sense == "<=":
            worst = max(worst, float(np.max(val - rhs)) / scale)
        elif c.sense == ">=":
            worst = max(worst, float(np.max(rhs - val)) / scale)
        else:
            worst = max(worst, float(np.max(np.abs(val - rhs))) / scale)
    return worst


def solve_arm(name, race_on, tl):
    build, indexed = INSTANCES[name]
    os.environ[FLAG] = "1" if race_on else "0"
    os.environ[INDEXED] = "1" if indexed else "0"
    target, model = build()
    t0 = time.perf_counter()
    r = target.solve(time_limit=tl)
    wall = time.perf_counter() - t0
    viol = None
    if r.x is not None and r.objective is not None:
        viol = _infeasibility(model, _x_flat(model, r))
    return dict(
        name=name,
        race=race_on,
        wall=round(wall, 3),
        status=r.status,
        obj=r.objective,
        bound=r.bound,
        cert=bool(r.gap_certified) and r.status == "optimal",
        nodes=int(r.node_count or 0),
        viol=viol,
        maximize=model._objective.sense.name == "MAXIMIZE",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma-separated instance names")
    ap.add_argument("--out", default=None, help="JSON-lines file, one line per solve")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(INSTANCES)
    print("load:", os.getloadavg(), "instances:", len(names), flush=True)
    out = open(args.out, "a") if args.out else None
    rows = []
    for name in names:
        runs = {True: [], False: []}
        # ABBA per pair of reps: ON, OFF, OFF, ON, ...
        order = [on for _ in range(args.reps // 2 or 1) for on in (True, False, False, True)]
        for on in order[: 2 * args.reps]:
            rec = solve_arm(name, on, args.time_limit)
            runs[on].append(rec)
            print(json.dumps(rec, default=str), flush=True)
            if out:
                out.write(json.dumps(rec, default=str) + "\n")
                out.flush()
        allr = runs[True] + runs[False]
        mx = allr[0]["maximize"]
        sgn = -1.0 if mx else 1.0  # internal minimize
        cert_objs = [sgn * a["obj"] for a in allr if a["cert"]]
        incs = [sgn * a["obj"] for a in allr if a["obj"] is not None]
        ref = min(cert_objs) if cert_objs else (min(incs) if incs else None)
        for a in allr:
            tag = f"{name} {'ON' if a['race'] else 'OFF'}"
            if a["viol"] is not None:
                check(a["viol"] <= 1e-5, f"{tag}: incumbent violates the model by {a['viol']:.3g}")
            if ref is not None and a["bound"] is not None:
                b = sgn * a["bound"]
                check(b <= ref + 1e-6 * max(1.0, abs(ref)), f"{tag}: bound {b} above {ref}")
            if a["cert"] and cert_objs:
                o = sgn * a["obj"]
                check(
                    abs(o - min(cert_objs)) <= 1e-4 * max(1.0, abs(o)) + 1e-6,
                    f"{tag}: certified {o} disagrees with {min(cert_objs)}",
                )
        if any(a["cert"] for a in runs[False]):
            check(all(a["cert"] for a in runs[True]), f"{name}: certification regression")
        row = dict(
            name=name,
            on_wall=statistics.mean(a["wall"] for a in runs[True]),
            off_wall=statistics.mean(a["wall"] for a in runs[False]),
            on_sd=statistics.pstdev(a["wall"] for a in runs[True]),
            off_sd=statistics.pstdev(a["wall"] for a in runs[False]),
            on_cert=sum(a["cert"] for a in runs[True]),
            off_cert=sum(a["cert"] for a in runs[False]),
        )
        rows.append(row)
        print(
            f"  {name:16s} OFF {row['off_wall']:6.2f}+-{row['off_sd']:.2f} cert {row['off_cert']}"
            f" | ON {row['on_wall']:6.2f}+-{row['on_sd']:.2f} cert {row['on_cert']}",
            flush=True,
        )
    tot_on = sum(r["on_wall"] for r in rows)
    tot_off = sum(r["off_wall"] for r in rows)
    cert_on = sum(r["on_cert"] > 0 for r in rows)
    cert_off = sum(r["off_cert"] > 0 for r in rows)
    slower = [
        r["name"]
        for r in rows
        if r["on_wall"] > r["off_wall"] + 2 * max(r["on_sd"], r["off_sd"], 0.05)
        and r["on_wall"] > 1.2 * r["off_wall"]
    ]
    faster = [
        r["name"]
        for r in rows
        if r["on_wall"] < r["off_wall"] - 2 * max(r["on_sd"], r["off_sd"], 0.05)
        and r["on_wall"] < r["off_wall"] / 1.2
    ]
    print(
        f"\nSUMMARY instances={len(rows)} total wall OFF {tot_off:.1f}s ON {tot_on:.1f}s; "
        f"certified OFF {cert_off} ON {cert_on}; faster {len(faster)} {faster}; "
        f"slower {len(slower)} {slower}",
        flush=True,
    )
    print(f"executed checks: {CHECKS}; failures: {len(FAILS)}", flush=True)
    for f in FAILS:
        print("  -", f)
    if CHECKS == 0 or FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
