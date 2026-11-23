"""#2309 panel: which root-dual gate should the HiGHS MILP route use?

The route (``lp_milp_highs._solve_milp_std``) checks the root LP's primal/dual pair: if
``box_dual_violation`` exceeds ``dual_violation_tolerance`` it falls through to the
rigorous ``root_pair_ns_gap`` (#1612) and decertifies unless that gap is within
``CERT_ABS + CERT_REL*|obj|``. This panel measures four gates end to end, by running the
real ``Model.solve()`` once per gate with ``box_dual_violation`` patched to return the
gate's value (the reported ``problem_scale_floor=True`` call is passed through):

  a  per-column (today)                         box_dual_violation(sf, x, y)
  b  scale-floored                              box_dual_violation(..., problem_scale_floor=True)
  c  per-column, interior single-entry c_j=0 columns skipped (the #2300 alternative)
  d  no filter: always ask the NS gap           +inf

Truth is independent of every solver: the MILP optimum by exact enumeration of the
integer box (feasibility decided in rationals near the boundary) or by DP (knapsacks);
the root-LP optimum by exact vertex enumeration in rationals. Set covers are too large
to enumerate and take ``scipy.optimize.milp`` as reference (unit coefficients,
numerically benign); they test false refusals only (#1612).

Families (the #1410 / #1612 shapes; generators reconstructed, the originals were never
committed): ``bad1`` / ``bad3`` badly scaled 5x2 integer programs, coefficients
log-uniform over 9 decades, cost scale 1 / 1e-3; ``well`` 6x3 Gaussian; ``knap`` 0/1
knapsacks with a side row; ``cover`` 120x60 set covers.

Run one arm per (highspy version, root-LP engine); the highspy version is whatever
imports, e.g. ``PYTHONPATH=<dir with highspy==1.15.1>``::

    python -u scripts/panel_2309_root_dual_gate.py run --engine highs --out arm.jsonl
    python -u scripts/panel_2309_root_dual_gate.py summary arm1.jsonl arm2.jsonl ...

Output is run data: keep it out of the repo (CLAUDE.md); quote the summary in the issue.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from fractions import Fraction

import numpy as np

CANDIDATES = ("a", "b", "c", "d")
FAMILIES = {"bad1": 2100, "bad3": 2100, "well": 120, "knap": 60, "cover": 40}


# --------------------------------------------------------------------------- instances
def gen(family: str, seed: int) -> dict:
    rng = np.random.default_rng([2309, sum(map(ord, family)), seed])
    if family in ("bad1", "bad3"):
        n, m = 5, 2

        def lgu(lo, hi, size):
            return rng.choice([-1.0, 1.0], size) * 10.0 ** rng.uniform(lo, hi, size)

        a = lgu(-3, 6, (m, n))
        c = lgu(-3, 4.5, n) * (1.0 if family == "bad1" else 1e-3)
        ub = rng.integers(2, 5, n)
        x0 = rng.integers(0, ub + 1)
        b = a @ x0 + np.abs(a) @ (ub * rng.uniform(0.0, 0.3, n))
        return dict(kind="ip", sense="min", c=c.tolist(), A=a.tolist(), b=b.tolist(),
                    ub=ub.tolist())  # fmt: skip
    if family == "well":
        n, m = 6, 3
        c = rng.normal(size=n)
        a = rng.normal(size=(m, n))
        ub = rng.integers(1, 5, n)
        b = a @ (ub / 2) + np.abs(a) @ (ub * 0.3)
        return dict(kind="ip", sense="min", c=c.tolist(), A=a.tolist(), b=b.tolist(),
                    ub=ub.tolist())  # fmt: skip
    if family == "knap":
        n = int(rng.integers(20, 31))
        u = rng.uniform(0.1, 0.7, n)
        w = rng.integers(15, 61, n)
        cap = int(w.sum() // 4)
        return dict(kind="knap", sense="max", u=u.tolist(), w=w.tolist(), cap=cap)
    if family == "cover":
        cover = np.zeros((120, 60), int)
        for i in range(120):
            cover[i, rng.choice(60, 3, replace=False)] = 1
        w = rng.integers(2, 5, 60).astype(float)
        return dict(kind="cover", sense="min", cover=cover.tolist(), w=w.tolist())
    raise ValueError(family)


def build(inst: dict):
    import discopt.modeling as dm

    m = dm.Model("p2309")
    if inst["kind"] == "ip":
        c, a, b, ub = inst["c"], inst["A"], inst["b"], inst["ub"]
        xs = [m.integer(f"x{j}", lb=0, ub=ub[j]) for j in range(len(c))]
        m.minimize(sum(c[j] * xs[j] for j in range(len(c))))
        for i in range(len(b)):
            m.subject_to(sum(a[i][j] * xs[j] for j in range(len(c))) <= b[i])
    elif inst["kind"] == "knap":
        u, w, n = inst["u"], inst["w"], len(inst["u"])
        y = m.binary("y", shape=(n,))
        m.maximize(dm.sum(lambda j: u[j] * y[j], over=range(n)))
        m.subject_to(dm.sum(lambda j: float(w[j]) * y[j], over=range(n)) <= float(inst["cap"]))
        m.subject_to(y[0] + y[1] <= 1)
    else:
        cover, w = np.array(inst["cover"]), inst["w"]
        y = m.binary("y", shape=(60,))
        for i in range(120):
            m.subject_to(dm.sum(y[j] for j in range(60) if cover[i, j]) >= 1)
        m.minimize(dm.sum(w[j] * y[j] for j in range(60)))
    return m


# ------------------------------------------------------------------------------ truth
def _feasible_exact(a, b, x) -> bool:
    for i in range(len(b)):
        if sum(Fraction(a[i][j]) * x[j] for j in range(len(x))) > Fraction(b[i]):
            return False
    return True


def milp_truth(inst: dict) -> float:
    if inst["kind"] == "ip":
        a, b, c, ub = (np.array(inst[k], float) for k in ("A", "b", "c", "ub"))
        pts = np.array(list(itertools.product(*[range(int(u) + 1) for u in ub])), float)
        act = pts @ a.T
        scale = np.abs(pts) @ np.abs(a).T + np.abs(b)
        slack = b - act
        clear_in = np.all(slack > 1e-9 * scale, axis=1)
        clear_out = np.any(slack < -1e-9 * scale, axis=1)
        unsure = ~clear_in & ~clear_out
        feas = clear_in.copy()
        for k in np.flatnonzero(unsure):
            feas[k] = _feasible_exact(inst["A"], inst["b"], [int(v) for v in pts[k]])
        assert feas.any(), "no feasible point: generator broken"
        objs = [sum(Fraction(inst["c"][j]) * int(p[j]) for j in range(len(p))) for p in pts[feas]]
        return float(min(objs))
    if inst["kind"] == "knap":
        u, w, cap = inst["u"], inst["w"], inst["cap"]

        def dp(items):
            best = np.full(cap + 1, -np.inf)
            best[0] = 0.0
            for j in items:
                for ww in range(cap, w[j] - 1, -1):
                    best[ww] = max(best[ww], best[ww - w[j]] + u[j])
            return float(best.max())

        n = len(u)
        return max(dp([j for j in range(n) if j != 1]), dp([j for j in range(n) if j != 0]))
    from scipy.optimize import Bounds, LinearConstraint, milp

    r = milp(np.array(inst["w"]), constraints=LinearConstraint(np.array(inst["cover"]), lb=1),
             integrality=np.ones(60), bounds=Bounds(0, 1))  # fmt: skip
    assert r.status == 0
    return float(r.fun)


def _solve_exact(mat, rhs):
    """Gaussian elimination in rationals; None if singular."""
    k = len(rhs)
    aug = [list(mat[i]) + [rhs[i]] for i in range(k)]
    for col in range(k):
        piv = next((r for r in range(col, k) if aug[r][col] != 0), None)
        if piv is None:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(k):
            if r != col and aug[r][col] != 0:
                f = aug[r][col] / aug[col][col]
                aug[r] = [aug[r][t] - f * aug[col][t] for t in range(k + 1)]
    return [aug[i][k] / aug[i][i] for i in range(k)]


def lp_truth(inst: dict) -> float | None:
    """Exact root-LP optimum of ``min c x, A x <= b, 0 <= x <= ub`` by vertex
    enumeration over the slack form (struct at 0/ub or basic; slacks >= 0)."""
    if inst["kind"] != "ip":
        return None
    a = [[Fraction(v) for v in row] for row in inst["A"]]
    b = [Fraction(v) for v in inst["b"]]
    c = [Fraction(v) for v in inst["c"]]
    ub = [Fraction(int(v)) for v in inst["ub"]]
    m, n = len(b), len(c)
    cols = [[a[i][j] for i in range(m)] for j in range(n)] + [
        [Fraction(int(i == r)) for i in range(m)] for r in range(m)
    ]
    best = None
    for basis in itertools.combinations(range(n + m), m):
        nb_struct = [j for j in range(n) if j not in basis]
        bmat = [[cols[j][i] for j in basis] for i in range(m)]
        for at_ub in itertools.product((False, True), repeat=len(nb_struct)):
            xn = {j: (ub[j] if f else Fraction(0)) for j, f in zip(nb_struct, at_ub)}
            rhs = [b[i] - sum(a[i][j] * v for j, v in xn.items()) for i in range(m)]
            xb = _solve_exact(bmat, rhs)
            if xb is None:
                continue
            ok = True
            x = dict(xn)
            for j, v in zip(basis, xb):
                if v < 0 or (j < n and v > ub[j]):
                    ok = False
                    break
                x[j] = v
            if not ok:
                continue
            obj = sum(c[j] * x.get(j, Fraction(0)) for j in range(n))
            best = obj if best is None or obj < best else best
    assert best is not None, "LP has no vertex: generator broken"
    return float(best)


# ------------------------------------------------------------------------------ gates
def per_column(sf, x, y, *, skip_interior_singleton_zero_cost: bool):
    """Replica of ``box_dual_violation``'s per-column test (asserted equal to it when
    nothing is skipped), with the #2300 alternative as an option."""
    import scipy.sparse as sp
    from discopt.solvers.lp_milp_highs import INF

    y = np.asarray(y, dtype=np.float64).ravel()
    xv = np.asarray(x, dtype=np.float64).ravel()
    d = sf.c - np.asarray(sf.A.T @ y).ravel()
    A = sp.csc_matrix(sf.A)  # noqa: N806
    absA = abs(A)  # noqa: N806
    scale = np.maximum(np.abs(sf.c) + np.asarray(absA.T @ np.abs(y)).ravel(), np.finfo(float).tiny)
    fin_lo, fin_hi = sf.xl > -INF, sf.xu < INF
    lo, hi = np.where(fin_lo, sf.xl, 0.0), np.where(fin_hi, sf.xu, 0.0)
    at_lo = fin_lo & (xv <= lo + 1e-9 * (1.0 + np.abs(lo)))
    at_hi = fin_hi & (xv >= hi - 1e-9 * (1.0 + np.abs(hi)))
    v = np.zeros_like(d)
    lo_only, hi_only = at_lo & ~at_hi, at_hi & ~at_lo
    v[lo_only] = np.maximum(0.0, -d[lo_only])
    v[hi_only] = np.maximum(0.0, d[hi_only])
    interior = ~at_lo & ~at_hi
    v[interior] = np.abs(d[interior])
    if skip_interior_singleton_zero_cost:
        single = np.diff(A.indptr) == 1
        v[interior & single & (sf.c == 0.0)] = 0.0
    return float((v / scale).max()) if v.size else 0.0


_STATE: dict = {}


def _init(engine: str) -> None:
    os.environ["DISCOPT_LP_MILP_BACKEND"] = "highs"
    os.environ["DISCOPT_LP_STD_ENGINE"] = engine
    import discopt
    from discopt.solvers import lp_milp_highs as H

    src = open(H.__file__).read()
    assert "root_dual_violation_scaled" in src, "not the post-#2305 tree"
    orig = H.box_dual_violation
    _STATE.update(H=H, orig=orig, discopt=discopt.__file__)

    def gate(sf, x, y, *, problem_scale_floor=False):
        if problem_scale_floor:  # the reported _scaled stat: unchanged
            return orig(sf, x, y, problem_scale_floor=True)
        raw = orig(sf, x, y)
        rep = per_column(sf, x, y, skip_interior_singleton_zero_cost=False)
        assert rep == raw, ("replica drifted from box_dual_violation", rep, raw)
        vals = {
            "a": raw,
            "b": orig(sf, x, y, problem_scale_floor=True),
            "c": per_column(sf, x, y, skip_interior_singleton_zero_cost=True),
            "d": float("inf"),
        }
        lp_obj = float(sf.c @ np.asarray(x)) + sf.obj_const
        ns = H.root_pair_ns_gap(sf, x, y)
        _STATE["gate"] = dict(
            vals=vals, tol=H.dual_violation_tolerance(sf), lp_obj=lp_obj, ns_gap=ns,
            ns_thr=H.CERT_ABS + H.CERT_REL * abs(lp_obj), calls=_STATE["gate"]["calls"] + 1,
        )  # fmt: skip
        return vals[_STATE["cand"]]

    H.box_dual_violation = gate


def _work(task):
    family, seed = task
    inst = gen(family, seed)
    t0 = time.perf_counter()
    truth = milp_truth(inst)
    lp_true = lp_truth(inst)
    t_truth = time.perf_counter() - t0
    rows = []
    for cand in CANDIDATES:
        _STATE["cand"] = cand
        _STATE["gate"] = {"calls": 0}
        r = build(inst).solve(time_limit=120)
        g = _STATE["gate"]
        st = r.solver_stats or {}
        rows.append(dict(
            family=family, seed=seed, cand=cand, sense=inst["sense"], truth=truth,
            lp_true=lp_true, status=r.status, cert=bool(r.gap_certified),
            obj=r.objective, bound=r.bound, gate_calls=g["calls"],
            gate=None if g["calls"] == 0 else {k: v for k, v in g.items() if k != "calls"},
            ns_ran="milp/root_dual_ns_gap" in st, route=getattr(r, "algorithm_route", None),
            wall=r.wall_time, t_truth=t_truth,
        ))  # fmt: skip
    return rows


def cmd_run(args) -> int:
    import highspy

    hv = highspy.Highs().version()
    tasks = [(f, s) for f in args.families for s in range(args.offset, args.offset + FAMILIES[f])]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"highspy {hv} engine {args.engine} tasks {len(tasks)} procs {args.procs}", flush=True)
    n_rows = n_gate = 0
    t0 = time.perf_counter()
    with (
        open(args.out, "w") as fh,
        mp.get_context("fork").Pool(args.procs, initializer=_init, initargs=(args.engine,)) as pool,
    ):
        for k, rows in enumerate(pool.imap_unordered(_work, tasks, chunksize=4), 1):
            for row in rows:
                row.update(highspy=hv, engine=args.engine)
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
                n_gate += row["gate_calls"] > 0
            fh.flush()
            if k % 200 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} instances  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"rows {n_rows}  rows reaching the gate {n_gate}", flush=True)
    return 0 if n_gate else 1


# ---------------------------------------------------------------------------- summary
def _tol(truth):
    return 1e-7 * (1.0 + abs(truth))


def false_bound(row) -> bool:
    if row["bound"] is None:
        return False
    return (row["bound"] > row["truth"] + _tol(row["truth"])) if row["sense"] == "min" else (
        row["bound"] < row["truth"] - _tol(row["truth"]))  # fmt: skip


def wrong_cert(row) -> bool:
    if not row["cert"]:
        return False
    obj_bad = row["obj"] is None or abs(row["obj"] - row["truth"]) > 1e-4 * (1 + abs(row["truth"]))
    return false_bound(row) or obj_bad


def cmd_summary(args) -> int:
    rows = [json.loads(line) for p in args.files for line in open(p)]
    checked = 0
    arms = sorted({(r["highspy"], r["engine"]) for r in rows})
    for hv, eng in arms:
        ar = [r for r in rows if r["highspy"] == hv and r["engine"] == eng]
        print(f"\n=== highspy {hv}, root-LP engine {eng}: {len(ar) // 4} instances ===")
        # LP level: the root pair against the exact LP optimum (gate reached, ip families).
        a_rows = [r for r in ar if r["cand"] == "a" and r["gate"] and r["lp_true"] is not None]
        lp_wrong = [
            r for r in a_rows
            if abs(r["gate"]["lp_obj"] - r["lp_true"]) > 1e-6 + 1e-9 * abs(r["lp_true"])
        ]  # fmt: skip
        print(f"root pairs vs exact LP: {len(a_rows)} checked, {len(lp_wrong)} wrong")
        for cand in CANDIDATES:

            def refuses(r):
                g = r["gate"]
                if g["vals"][cand] <= g["tol"]:
                    return False
                ns = g["ns_gap"]
                return ns is None or ns > g["ns_thr"]

            caught = sum(refuses(r) for r in lp_wrong)
            print(f"  gate {cand}: refuses {caught}/{len(lp_wrong)} wrong root pairs")
        print("MILP level (end-to-end Model.solve per gate):")
        print("  gate family  n  certified  WRONG-cert  false-bound  gate-trips  ns-runs")
        for cand in CANDIDATES:
            for fam in FAMILIES:
                fr = [r for r in ar if r["cand"] == cand and r["family"] == fam]
                if not fr:
                    continue
                checked += len(fr)
                trips = sum(
                    1 for r in fr if r["gate"] and r["gate"]["vals"][cand] > r["gate"]["tol"]
                )
                print(f"  {cand:4} {fam:6} {len(fr):5} {sum(r['cert'] for r in fr):9} "
                      f"{sum(wrong_cert(r) for r in fr):10} {sum(false_bound(r) for r in fr):11} "
                      f"{trips:11} {sum(r['ns_ran'] for r in fr):8}")  # fmt: skip
        # Paired differences against today's gate.
        by = {(r["family"], r["seed"], r["cand"]): r for r in ar}
        for cand in "bcd":
            lost = gained = 0
            for (fam, seed, cc), r in by.items():
                if cc != "a":
                    continue
                o = by.get((fam, seed, cand))
                if o is None:
                    continue
                lost += r["cert"] and not o["cert"]
                gained += o["cert"] and not r["cert"]
            print(f"  {cand} vs a: certified gained {gained}, lost {lost}")
    print(f"\nexecuted comparisons: {checked}")
    return 0 if checked else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", choices=("highs", "rust"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    r.add_argument("--procs", type=int, default=4)
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    a = p.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_summary(a)


if __name__ == "__main__":
    sys.exit(main())
