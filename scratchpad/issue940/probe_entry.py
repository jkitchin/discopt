"""Entry experiment for issue #940.

Measures how often POUNCE's LP point trips the #850 guard
(``_matrix_solution_feasible``) at the ``_solve_lp_matrix`` call site, and the
distribution of worst-row violation/threshold ratios.

CLAUDE.md §6: prints an executed-check count and exits non-zero if it is zero.
CLAUDE.md §7: no exception is swallowed anywhere in the instrument.
CLAUDE.md §8: asserts module provenance before measuring.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402

# ---- provenance (§8) --------------------------------------------------------
assert S.__file__.startswith("/home/user/discopt/python/"), S.__file__
import discopt.solvers.lp_pounce as LPP  # noqa: E402

assert LPP.__file__.startswith("/home/user/discopt/python/"), LPP.__file__
assert LPP.POUNCE_AVAILABLE, "pounce not installed - the experiment cannot run"

RECORDS = []
_orig_feasible = S._matrix_solution_feasible


def _worst_ratio(x, A_ub, b_ub, A_eq, b_eq, bounds, tol=1e-6, rtol=1e-9):
    """Worst (violation / threshold) over every row and bound. >1 == guard trips."""
    x = np.asarray(x, dtype=np.float64)
    worst = 0.0
    worst_scale = 0.0
    if not np.all(np.isfinite(x)):
        return np.inf, 0.0
    absx = np.abs(x)
    for A, b, signed in ((A_ub, b_ub, True), (A_eq, b_eq, False)):
        if A is None or b is None or not len(b):
            continue
        A = np.asarray(A, dtype=np.float64)
        viol = A @ x - np.asarray(b, dtype=np.float64)
        if not signed:
            viol = np.abs(viol)
        scale = np.abs(A) @ absx
        ratio = viol / (tol + rtol * scale)
        k = int(np.argmax(ratio))
        if ratio[k] > worst:
            worst, worst_scale = float(ratio[k]), float(scale[k])
    if bounds is not None:
        for xi, (lo, hi) in zip(x, bounds):
            thr = tol + rtol * abs(xi)
            v = max(lo - xi, xi - hi)
            if v / thr > worst:
                worst, worst_scale = float(v / thr), float(abs(xi))
    return worst, worst_scale


def _patched(x, A_ub, b_ub, A_eq, b_eq, bounds, tol=1e-6, rtol=1e-9):
    ok = _orig_feasible(x, A_ub, b_ub, A_eq, b_eq, bounds, tol=tol, rtol=rtol)
    ratio, scale = _worst_ratio(x, A_ub, b_ub, A_eq, b_eq, bounds, tol=tol, rtol=rtol)
    RECORDS.append(
        {"engine": _CURRENT["engine"], "model": _CURRENT["model"], "ok": ok,
         "ratio": ratio, "row_scale": scale}
    )
    return ok


S._matrix_solution_feasible = _patched

# Tag each guard call with the engine that produced the point.
_CURRENT = {"engine": "?", "model": "?"}
_orig_solve_lp_matrix = S._solve_lp_matrix


def _tagged_solve_lp_matrix(model, t_start, time_limit, solve_lp_fn, engine, **kw):
    prev = _CURRENT["engine"]
    _CURRENT["engine"] = engine
    try:
        return _orig_solve_lp_matrix(model, t_start, time_limit, solve_lp_fn, engine, **kw)
    finally:
        _CURRENT["engine"] = prev


S._solve_lp_matrix = _tagged_solve_lp_matrix


# ---------------------------------------------------------------- populations
def notebook_lps():
    """The four tutorial_lp.ipynb LPs, verbatim from the notebook source."""
    out = []

    # diet
    cost = np.array([2.0, 3.5, 8.0, 11.0, 25.0])
    A = np.array([[3.0, 8.0, 15.0, 22.0, 31.0],
                  [25.0, 120.0, 200.0, 10.0, 15.0],
                  [1.0, 0.1, 0.5, 3.0, 5.5],
                  [250.0, 150.0, 400.0, 200.0, 450.0]])
    req = np.array([55.0, 800.0, 12.0, 2000.0])
    m = dm.Model("diet")
    x = m.continuous("x", shape=(5,), lb=0, ub=10)
    m.minimize(dm.sum(lambda j: cost[j] * x[j], over=range(5)))
    for i in range(4):
        m.subject_to(dm.sum(lambda j: A[i, j] * x[j], over=range(5)) >= req[i], name=f"n{i}")
    out.append(("nb:diet", m))

    # transport
    supply = np.array([300.0, 400.0, 500.0])
    demand = np.array([200.0, 300.0, 250.0, 450.0])
    sc = np.array([[8.0, 6.0, 10.0, 9.0], [9.0, 12.0, 7.0, 5.0], [14.0, 9.0, 16.0, 4.0]])
    nw, nc = sc.shape
    mt = dm.Model("transport")
    xt = mt.continuous("ship", shape=(nw, nc), lb=0, ub=float(max(supply.max(), demand.max())))
    mt.minimize(dm.sum(lambda i: dm.sum(lambda j: sc[i, j] * xt[i, j], over=range(nc)),
                       over=range(nw)))
    for i in range(nw):
        mt.subject_to(dm.sum(lambda j: xt[i, j], over=range(nc)) <= supply[i], name=f"s{i}")
    for j in range(nc):
        mt.subject_to(dm.sum(lambda i: xt[i, j], over=range(nw)) >= demand[j], name=f"d{j}")
    out.append(("nb:transport", mt))

    # production
    profit = np.array([20.0, 30.0, 25.0])
    usage = np.array([[2.0, 1.0, 3.0], [1.0, 3.0, 2.0], [0.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    avail = np.array([120.0, 150.0, 80.0, 180.0])
    mp = dm.Model("production")
    xp = mp.continuous("prod", shape=(3,), lb=0)
    mp.maximize(dm.sum(lambda j: profit[j] * xp[j], over=range(3)))
    for i in range(4):
        mp.subject_to(dm.sum(lambda j: usage[i, j] * xp[j], over=range(3)) <= avail[i],
                      name=f"r{i}")
    out.append(("nb:production", mp))

    # blending
    quality = np.array([8.0, 6.0, 4.0, 2.0])
    crude_cost = np.array([45.0, 35.0, 25.0, 15.0])
    avail_b = np.array([5000.0] * 4)
    min_q = np.array([6.0, 4.0])
    dem = np.array([3000.0, 4000.0])
    price = np.array([60.0, 40.0])
    nk, np_ = 4, 2
    mbl = dm.Model("blending")
    xb = mbl.continuous("x", shape=(nk, np_), lb=0)
    rev = dm.sum(lambda j: price[j] * dm.sum(lambda i: xb[i, j], over=range(nk)), over=range(np_))
    tc = dm.sum(lambda j: dm.sum(lambda i: crude_cost[i] * xb[i, j], over=range(nk)),
                over=range(np_))
    mbl.maximize(rev - tc)
    for i in range(nk):
        mbl.subject_to(dm.sum(lambda j: xb[i, j], over=range(np_)) <= avail_b[i], name=f"s{i}")
    for j in range(np_):
        mbl.subject_to(dm.sum(lambda i: xb[i, j], over=range(nk)) >= dem[j], name=f"d{j}")
    for j in range(np_):
        mbl.subject_to(
            dm.sum(lambda i: (quality[i] - min_q[j]) * xb[i, j], over=range(nk)) >= 0,
            name=f"q{j}")
    out.append(("nb:blending", mbl))
    return out


def _matrix_lp(name, c, A_le, b_le, A_ge, b_ge, lb, ub, maximize=False):
    m = dm.Model(name)
    n = len(c)
    x = m.continuous("x", shape=(n,), lb=lb, ub=ub)
    obj = dm.sum(lambda j: float(c[j]) * x[j], over=range(n))
    m.maximize(obj) if maximize else m.minimize(obj)
    for i in range(A_le.shape[0]):
        row = A_le[i]
        m.subject_to(dm.sum(lambda j: float(row[j]) * x[j], over=range(n)) <= float(b_le[i]),
                     name=f"le{i}")
    for i in range(A_ge.shape[0]):
        row = A_ge[i]
        m.subject_to(dm.sum(lambda j: float(row[j]) * x[j], over=range(n)) >= float(b_ge[i]),
                     name=f"ge{i}")
    return m


def sweep_lps():
    """Random LPs across size and *data scale*.

    Hypothesis under test: the guard trips as a function of the row term scale
    ``sum_j |A_ij||x_j|``, not of any property peculiar to the tutorial models.
    Threshold is ``1e-6 + 1e-9*scale``, so a relative POUNCE accuracy of ~6e-9
    predicts trips once ``scale`` exceeds a few hundred.
    """
    out = []
    rng = np.random.default_rng(20940)
    for n, mrows in ((5, 3), (10, 6), (20, 10), (40, 20)):
        for scale in (1.0, 1e1, 1e2, 1e3, 1e4, 1e6):
            for rep in range(3):
                A_ge = np.abs(rng.uniform(0.5, 5.0, size=(mrows, n)))
                b_ge = scale * np.abs(rng.uniform(0.5, 2.0, size=mrows)) * n * 0.25
                c = np.abs(rng.uniform(1.0, 10.0, size=n))
                A_le = np.zeros((0, n))
                b_le = np.zeros(0)
                out.append((f"sweep:ge_n{n}_m{mrows}_s{scale:g}_r{rep}",
                            _matrix_lp("sw", c, A_le, b_le, A_ge, b_ge,
                                       lb=0.0, ub=float(10.0 * scale))))
                # <= (resource) form at the same scale
                A_le2 = np.abs(rng.uniform(0.5, 5.0, size=(mrows, n)))
                b_le2 = scale * np.abs(rng.uniform(0.5, 2.0, size=mrows)) * n * 0.25
                out.append((f"sweep:le_n{n}_m{mrows}_s{scale:g}_r{rep}",
                            _matrix_lp("sw", c, A_le2, b_le2, np.zeros((0, n)), np.zeros(0),
                                       lb=0.0, ub=float(10.0 * scale), maximize=True)))
    return out


def corpus_lps():
    """Every ``.nl`` in the in-repo corpus that classifies as an LP."""
    from discopt._jax.problem_classifier import ProblemClass, classify_problem
    from discopt.modeling import from_nl

    out = []
    roots = ["python/tests/data/minlplib_nl", "python/tests/data/minlplib"]
    paths = []
    for root in roots:
        if os.path.isdir(root):
            paths += [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.endswith(".nl")]
    for p in paths:
        model = from_nl(p)
        if classify_problem(model) == ProblemClass.LP:
            out.append((f"nl:{os.path.basename(p)}", model))
    print(f"corpus: scanned {len(paths)} .nl files, {len(out)} classify as LP", flush=True)
    return out


def main():
    pops = [("notebook", notebook_lps()), ("sweep", sweep_lps()), ("corpus", corpus_lps())]
    for label, models in pops:
        for name, model in models:
            _CURRENT["model"] = name
            before = len(RECORDS)
            res = model.solve()
            recs = RECORDS[before:]
            fired = [r for r in recs if r["engine"] == "POUNCE" and not r["ok"]]
            print(f"{label:8s} {name:38s} status={res.status:10s} "
                  f"guard_calls={len(recs)} pounce_trips={len(fired)} "
                  f"worst_ratio={max([r['ratio'] for r in recs], default=float('nan')):.3g} "
                  f"obj={res.objective}", flush=True)

    pounce_recs = [r for r in RECORDS if r["engine"] == "POUNCE"]
    trips = [r for r in pounce_recs if not r["ok"]]
    print("\n===== SUMMARY =====")
    print(f"GUARD_CHECKS_OBSERVED={len(RECORDS)}  POUNCE_CHECKS={len(pounce_recs)}  "
          f"POUNCE_TRIPS={len(trips)}")
    if pounce_recs:
        print(f"trip rate = {100.0 * len(trips) / len(pounce_recs):.1f}% of POUNCE LP solves")
        ratios = np.array([r["ratio"] for r in pounce_recs])
        finite = ratios[np.isfinite(ratios)]
        for q in (50, 75, 90, 95, 99, 100):
            print(f"  ratio p{q:<3d} = {np.percentile(finite, q):.4g}")
        print("\n  trips by row scale decade:")
        import collections
        by = collections.Counter()
        tot = collections.Counter()
        for r in pounce_recs:
            d = int(np.floor(np.log10(max(r["row_scale"], 1e-12))))
            tot[d] += 1
            if not r["ok"]:
                by[d] += 1
        for d in sorted(tot):
            print(f"    1e{d:<3d}: {by[d]:4d}/{tot[d]:<4d} trip")
    print("\n  tripping models:")
    for r in trips:
        print(f"    {r['model']:40s} ratio={r['ratio']:.3g} row_scale={r['row_scale']:.4g}")

    if not RECORDS:
        print("PROBE FIRED ZERO CHECKS - measurement is meaningless", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
