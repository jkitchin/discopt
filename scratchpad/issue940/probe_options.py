"""Issue #940 experiment 2: can POUNCE's own LP convergence be made to imply
discopt's per-row constraint tolerance?

For each candidate option set, solve the tripping LPs directly through
``lp_pounce.solve_lp`` and record (a) status, (b) worst absolute row violation,
(c) worst violation/threshold ratio against the #850 guard, (d) iterations and
wall time. §6: prints an executed-check count and exits non-zero if zero.
"""

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import discopt.solvers.lp_pounce as LPP  # noqa: E402
from discopt.solver import _matrix_solution_feasible  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402

assert LPP.__file__.startswith("/home/user/discopt/python/"), LPP.__file__
assert LPP.POUNCE_AVAILABLE

CHECKS = 0


def worst(x, A_ub, b_ub, A_eq, b_eq, bounds, tol=1e-6, rtol=1e-9):
    x = np.asarray(x, float)
    absx = np.abs(x)
    wv = 0.0
    wr = 0.0
    for A, b, signed in ((A_ub, b_ub, True), (A_eq, b_eq, False)):
        if A is None or b is None or not len(b):
            continue
        A = np.asarray(A, float)
        v = A @ x - np.asarray(b, float)
        if not signed:
            v = np.abs(v)
        s = np.abs(A) @ absx
        wv = max(wv, float(v.max()))
        wr = max(wr, float((v / (tol + rtol * s)).max()))
    for xi, (lo, hi) in zip(x, bounds):
        v = max(lo - xi, xi - hi)
        wv = max(wv, float(v))
        wr = max(wr, float(v / (tol + rtol * abs(xi))))
    return wv, wr


# ---- LP instances (matrix form) reproducing the tripping population ---------
def diet():
    c = np.array([2.0, 3.5, 8.0, 11.0, 25.0])
    A = np.array([[3.0, 8.0, 15.0, 22.0, 31.0],
                  [25.0, 120.0, 200.0, 10.0, 15.0],
                  [1.0, 0.1, 0.5, 3.0, 5.5],
                  [250.0, 150.0, 400.0, 200.0, 450.0]])
    r = np.array([55.0, 800.0, 12.0, 2000.0])
    return "diet", c, -A, -r, [(0.0, 10.0)] * 5


def transport():
    supply = np.array([300.0, 400.0, 500.0])
    demand = np.array([200.0, 300.0, 250.0, 450.0])
    sc = np.array([[8.0, 6.0, 10.0, 9.0], [9.0, 12.0, 7.0, 5.0], [14.0, 9.0, 16.0, 4.0]])
    nw, nc = sc.shape
    n = nw * nc
    rows, rhs = [], []
    for i in range(nw):
        r = np.zeros(n)
        r[i * nc:(i + 1) * nc] = 1.0
        rows.append(r)
        rhs.append(supply[i])
    for j in range(nc):
        r = np.zeros(n)
        r[j::nc] = -1.0
        rows.append(r)
        rhs.append(-demand[j])
    return "transport", sc.ravel(), np.array(rows), np.array(rhs), [(0.0, 500.0)] * n


def rand_lp(tag, n, m, scale, seed):
    rng = np.random.default_rng(seed)
    A = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
    b = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
    c = np.abs(rng.uniform(1.0, 10.0, size=n))
    return tag, c, A, b, [(0.0, 10.0 * scale)] * n


INSTANCES = [diet(), transport()]
for k, (n, m) in enumerate(((5, 3), (10, 6), (20, 10), (40, 20))):
    for scale in (1e2, 1e3, 1e4):
        for rep in range(2):
            INSTANCES.append(rand_lp(f"rand_n{n}_s{scale:g}_r{rep}", n, m, scale,
                                     1000 + 17 * k + rep + int(np.log10(scale))))

CANDIDATES = {
    "Ipopt default 1e-4": {"constr_viol_tol": 1e-4},
    "SHIPPED constr_viol_tol=1e-8": {"constr_viol_tol": 1e-8},
    "constr_viol_tol=1e-9": {"constr_viol_tol": 1e-9},
    "constr_viol_tol=1e-11": {"constr_viol_tol": 1e-11},
    "tol=1e-10": {"tol": 1e-10},
    "cvt=1e-9 + tol=1e-10": {"constr_viol_tol": 1e-9, "tol": 1e-10},
    "cvt=1e-9 + acceptable_cvt=1e-9": {
        "constr_viol_tol": 1e-9, "acceptable_constr_viol_tol": 1e-9},
    "scaling=none": {"nlp_scaling_method": "none"},
    "scaling=none + cvt=1e-9": {"nlp_scaling_method": "none", "constr_viol_tol": 1e-9},
}


def main():
    global CHECKS
    print(f"{'candidate':34s} {'trips':>6s} {'/n':>4s} {'worst_ratio':>12s} "
          f"{'worst_viol':>11s} {'nonopt':>6s} {'mean_iter':>9s} {'mean_ms':>8s}")
    rows = {}
    for label, opts in CANDIDATES.items():
        trips = nonopt = 0
        wr_all, wv_all, iters, times = [], [], [], []
        for tag, c, A_ub, b_ub, bounds in INSTANCES:
            t0 = time.perf_counter()
            res = LPP.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                               options=opts or None)
            dt = (time.perf_counter() - t0) * 1e3
            CHECKS += 1
            if res.status != SolveStatus.OPTIMAL:
                nonopt += 1
                continue
            ok = _matrix_solution_feasible(res.x, A_ub, b_ub, None, None, bounds)
            wv, wr = worst(res.x, A_ub, b_ub, None, None, bounds)
            wr_all.append(wr)
            wv_all.append(wv)
            iters.append(res.iterations)
            times.append(dt)
            if not ok:
                trips += 1
        rows[label] = (trips, len(INSTANCES), max(wr_all, default=float("nan")),
                       max(wv_all, default=float("nan")), nonopt,
                       float(np.mean(iters)) if iters else float("nan"),
                       float(np.mean(times)) if times else float("nan"))
        t, n, wr, wv, no, it, ms = rows[label]
        print(f"{label:34s} {t:6d} {n:4d} {wr:12.4g} {wv:11.4g} {no:6d} {it:9.1f} {ms:8.2f}",
              flush=True)

    print(f"\nCHECKS_EXECUTED={CHECKS}")
    if CHECKS == 0:
        print("PROBE FIRED ZERO CHECKS", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
