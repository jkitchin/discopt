"""Benchmark the Rust-internal MILP simplex driver against SCIP.

Two regimes:
  * sparse set-covering (each column covers ~6 rows) — the regime where dense
    O(m^3) Gomory separation and cold-per-step diving used to dominate the root
    LP; sparse-LU Gomory + warm-start diving fixed both.
  * dense multidimensional knapsack (regime B) — the small/dense guard that must
    not regress.

Solves each instance with ``model.solve(nlp_solver="simplex")`` and, when
``pyscipopt`` is available, the same model exported to MPS and solved by SCIP,
printing a side-by-side wall-time table.

Run:  python discopt_benchmarks/bench_milp_sparse.py
Env:  TL=<seconds> time limit (default 60)
"""

import os
import sys
import time
import tempfile
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np  # noqa: E402
import discopt.modeling as dm  # noqa: E402

TL = float(os.environ.get("TL", "60"))

try:
    from pyscipopt import Model as SCIPModel

    HAVE_SCIP = True
except ImportError:
    HAVE_SCIP = False


def gen_setcover(ncol, nrow, seed, per_col=6):
    rng = np.random.default_rng(seed)
    cols = [rng.choice(nrow, size=per_col, replace=False) for _ in range(ncol)]
    rows_to_cols = {i: [] for i in range(nrow)}
    for j, c in enumerate(cols):
        for i in c:
            rows_to_cols[i].append(j)
    for i in range(nrow):
        if not rows_to_cols[i]:
            j = int(rng.integers(0, ncol))
            rows_to_cols[i].append(j)
    cost = rng.integers(1, 100, ncol).astype(float)
    m = dm.Model(f"sc_{ncol}_{nrow}_{seed}")
    x = m.binary("x", shape=(ncol,))
    m.minimize(dm.sum(lambda j: cost[j] * x[j], over=range(ncol)))
    m.subject_to(
        [dm.sum(lambda j: x[j], over=rows_to_cols[i]) >= 1 for i in range(nrow)],
        name="cov",
    )
    return m


def gen_mdknapsack(n, kdim, seed):
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 50, size=(kdim, n)).astype(float)
    cap = 0.5 * w.sum(axis=1)
    profit = rng.integers(1, 50, size=n).astype(float)
    m = dm.Model(f"mdk_{n}_{kdim}_{seed}")
    x = m.binary("x", shape=(n,))
    m.maximize(dm.sum(lambda j: profit[j] * x[j], over=range(n)))
    m.subject_to(
        [dm.sum(lambda j: w[i, j] * x[j], over=range(n)) <= cap[i] for i in range(kdim)],
        name="cap",
    )
    return m


def solve_discopt(m):
    t0 = time.perf_counter()
    r = m.solve(nlp_solver="simplex", time_limit=TL, gap_tolerance=1e-6, max_nodes=5_000_000)
    return {"status": str(r.status), "obj": r.objective, "nodes": r.node_count,
            "wall": time.perf_counter() - t0}


def solve_scip(mps):
    sm = SCIPModel()
    sm.hideOutput(True)
    sm.readProblem(mps)
    sm.setParam("limits/time", TL)
    sm.setParam("limits/gap", 1e-6)
    t0 = time.perf_counter()
    sm.optimize()
    w = time.perf_counter() - t0
    return {"status": sm.getStatus(), "obj": sm.getObjVal() if sm.getNSols() > 0 else None,
            "nodes": sm.getNNodes(), "wall": w}


def bench(label, model):
    mps = os.path.join(tempfile.gettempdir(), f"{label}.mps")
    model.to_mps(mps)
    d = solve_discopt(model)
    s = solve_scip(mps) if HAVE_SCIP else None
    if s:
        ratio = d["wall"] / s["wall"] if s["wall"] > 0 else float("inf")
        print(f"{label:14} | discopt {d['status']:10} {d['wall']:7.2f}s obj={str(d['obj'])[:8]:>8} "
              f"| scip {s['status']:10} {s['wall']:7.2f}s | {ratio:5.1f}x")
    else:
        print(f"{label:14} | discopt {d['status']:10} {d['wall']:7.2f}s obj={str(d['obj'])[:8]:>8}")
    sys.stdout.flush()


def main():
    print(f"# MILP simplex driver vs SCIP  (TL={TL}s, pyscipopt={'yes' if HAVE_SCIP else 'no'})")
    print("## sparse set-covering (each column covers 6 rows)")
    for nc, nr in [(500, 250), (1000, 500), (2000, 800), (4000, 1500)]:
        bench(f"sc{nc}x{nr}", gen_setcover(nc, nr, seed=3))
    print("## dense multidimensional knapsack (regime B)")
    for n, k in [(50, 5), (150, 8), (250, 10)]:
        bench(f"mdk{n}x{k}", gen_mdknapsack(n, k, seed=7))


if __name__ == "__main__":
    main()
