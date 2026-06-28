"""Engine baseline + A/B harness for the pure-MILP simplex driver (issue #332).

Two measurement layers, on a fixed family of sparse MILP instances plus
dense/guard instances:

  * **LP layer** — the root relaxation LP solved by discopt's engine (RootSolve
    phase) vs **HiGHS** on the *identical* LP. Isolates LP-engine quality from B&B.
  * **MILP layer** — the full solve by discopt's engine (`solve_milp_py` directly,
    bypassing the Python budget cap) vs **SCIP** on the same model.

Per-phase / pivot attribution comes from the `DISCOPT_PROFILE` instrumentation in
the Rust core (RootSolve / NodeLpSolve / StrongBranch / SepGomory + pivot
categorization), captured via an fd-level stderr redirect.

Run:
  python discopt_benchmarks/bench_engine.py                 # full baseline
  python discopt_benchmarks/bench_engine.py --profile sc    # + per-phase dump on covering
Env: TL=<s> engine/SCIP time limit (default 30).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402

TL = float(os.environ.get("TL", "30"))

try:
    import highspy  # noqa: E402

    HAVE_HIGHS = True
except ImportError:
    HAVE_HIGHS = False
try:
    from pyscipopt import Model as SCIPModel  # noqa: E402

    HAVE_SCIP = True
except ImportError:
    HAVE_SCIP = False


# --------------------------------------------------------------------------- #
# Instance families (deterministic). Each returns a discopt Model.
# --------------------------------------------------------------------------- #
def gen_setcover(ncol, nrow, seed, per_col=6):
    rng = np.random.default_rng(seed)
    cols = [rng.choice(nrow, size=per_col, replace=False) for _ in range(ncol)]
    r2c = {i: [] for i in range(nrow)}
    for j, c in enumerate(cols):
        for i in c:
            r2c[i].append(j)
    for i in range(nrow):
        if not r2c[i]:
            r2c[i].append(int(rng.integers(0, ncol)))
    cost = rng.integers(1, 100, ncol).astype(float)
    m = dm.Model(f"sc{ncol}x{nrow}")
    x = m.binary("x", shape=(ncol,))
    m.minimize(dm.sum(lambda j: cost[j] * x[j], over=range(ncol)))
    m.subject_to([dm.sum(lambda j: x[j], over=r2c[i]) >= 1 for i in range(nrow)], name="cov")
    return m


def gen_setpack(ncol, nrow, seed, per_col=6):
    rng = np.random.default_rng(seed)
    cols = [rng.choice(nrow, size=per_col, replace=False) for _ in range(ncol)]
    r2c = {i: [] for i in range(nrow)}
    for j, c in enumerate(cols):
        for i in c:
            r2c[i].append(j)
    val = rng.integers(1, 100, ncol).astype(float)
    m = dm.Model(f"sp{ncol}x{nrow}")
    x = m.binary("x", shape=(ncol,))
    m.maximize(dm.sum(lambda j: val[j] * x[j], over=range(ncol)))
    m.subject_to(
        [dm.sum(lambda j: x[j], over=r2c[i]) <= 1 for i in range(nrow) if r2c[i]], name="pk"
    )
    return m


def gen_gap(nagents, ntasks, seed):
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 50, (nagents, ntasks)).astype(float)
    res = rng.integers(1, 20, (nagents, ntasks)).astype(float)
    cap = res.sum(axis=1) * 0.6
    m = dm.Model(f"gap{nagents}x{ntasks}")
    x = m.binary("x", shape=(nagents, ntasks))
    m.minimize(
        dm.sum(
            lambda at: cost[at[0], at[1]] * x[at[0], at[1]],
            over=[(a, t) for a in range(nagents) for t in range(ntasks)],
        )
    )
    # each task assigned once
    m.subject_to(
        [dm.sum(lambda a, t=t: x[a, t], over=range(nagents)) == 1 for t in range(ntasks)],
        name="assign",
    )
    # capacity per agent
    m.subject_to(
        [
            dm.sum(lambda t, a=a: res[a, t] * x[a, t], over=range(ntasks)) <= cap[a]
            for a in range(nagents)
        ],
        name="cap",
    )
    return m


def gen_knapsack(n, kdim, seed):
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 50, (kdim, n)).astype(float)
    cap = 0.5 * w.sum(axis=1)
    profit = rng.integers(1, 50, n).astype(float)
    m = dm.Model(f"mdk{n}x{kdim}")
    x = m.binary("x", shape=(n,))
    m.maximize(dm.sum(lambda j: profit[j] * x[j], over=range(n)))
    m.subject_to(
        [dm.sum(lambda j, i=i: w[i, j] * x[j], over=range(n)) <= cap[i] for i in range(kdim)],
        name="cap",
    )
    return m


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def _lp_data(model):
    from discopt._jax.problem_classifier import extract_lp_data
    from discopt.solver import _extract_variable_info

    lp = extract_lp_data(model)
    n_orig = sum(v.size for v in model._variables)
    _, _, _, ioff, isz = _extract_variable_info(model)
    int_idx = [j for off, s in zip(ioff, isz, strict=False) for j in range(off, off + int(s))]
    return lp, n_orig, int_idx


def solve_discopt_engine(model, capture_profile=False):
    """Call the Rust MILP engine directly (no Python budget cap)."""
    from discopt._rust import solve_milp_py

    lp, n_orig, int_idx = _lp_data(model)
    A = np.ascontiguousarray(lp.A_eq, dtype=np.float64)
    args = (
        np.ascontiguousarray(lp.c, dtype=np.float64),
        A,
        np.ascontiguousarray(lp.b_eq, dtype=np.float64),
        np.ascontiguousarray(lp.x_l, dtype=np.float64),
        np.ascontiguousarray(lp.x_u, dtype=np.float64),
        np.ascontiguousarray(np.asarray(int_idx, dtype=np.int64)),
        n_orig,
        float(lp.obj_const),
        5_000_000,
        1e-6,
    )
    prof = ""
    t0 = time.perf_counter()
    if capture_profile:
        os.environ["DISCOPT_PROFILE"] = "1"
        old = os.dup(2)
        # fd-level redirect: tf must outlive the dup2/restore dance, not a `with`.
        tf = tempfile.TemporaryFile()  # noqa: SIM115
        os.dup2(tf.fileno(), 2)
        try:
            st, x, obj, bound, nodes, _ = solve_milp_py(*args, time_limit_s=TL)
        finally:
            os.dup2(old, 2)
            os.close(old)
        tf.seek(0)
        prof = tf.read().decode(errors="replace")
        tf.close()
        del os.environ["DISCOPT_PROFILE"]
    else:
        st, x, obj, bound, nodes, _ = solve_milp_py(*args, time_limit_s=TL)
    return {
        "status": st,
        "obj": obj,
        "nodes": nodes,
        "wall": time.perf_counter() - t0,
        "profile": prof,
    }


def solve_highs_lp(model):
    """Solve the root LP relaxation (drop integrality) with HiGHS."""
    if not HAVE_HIGHS:
        return None
    import scipy.sparse as sp

    lp, _, _ = _lp_data(model)
    A = sp.csr_matrix(np.asarray(lp.A_eq, dtype=np.float64))
    m, n = A.shape
    c = np.asarray(lp.c, dtype=np.float64)
    b = np.asarray(lp.b_eq, dtype=np.float64)
    xl = np.asarray(lp.x_l, dtype=np.float64)
    xu = np.asarray(lp.x_u, dtype=np.float64)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", TL)
    h.addCols(n, c, xl, xu, 0, [], [], [])
    h.addRows(m, b, b, A.nnz, A.indptr, A.indices, A.data)
    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0
    info = h.getInfo()
    return {
        "wall": dt,
        "iters": int(getattr(info, "simplex_iteration_count", -1)),
        "obj": h.getObjectiveValue() + float(lp.obj_const),
    }


def solve_scip(model):
    if not HAVE_SCIP:
        return None
    mps = os.path.join(tempfile.gettempdir(), f"{model.name}.mps")
    model.to_mps(mps)
    s = SCIPModel()
    s.hideOutput(True)
    s.readProblem(mps)
    s.setParam("limits/time", TL)
    s.setParam("limits/gap", 1e-6)
    t0 = time.perf_counter()
    s.optimize()
    dt = time.perf_counter() - t0
    return {
        "status": s.getStatus(),
        "obj": s.getObjVal() if s.getNSols() else None,
        "nodes": s.getNNodes(),
        "wall": dt,
    }


def _rootsolve_ms(prof):
    m = re.search(r"RootSolve\s+\d+ calls\s+([\d.]+) ms", prof or "")
    return float(m.group(1)) if m else None


INSTANCES = [
    ("setcover", lambda: gen_setcover(500, 250, 3)),
    ("setcover", lambda: gen_setcover(1000, 500, 3)),
    ("setcover", lambda: gen_setcover(2000, 800, 3)),
    ("setcover", lambda: gen_setcover(4000, 1500, 3)),
    ("setpack", lambda: gen_setpack(1000, 500, 4)),
    ("setpack", lambda: gen_setpack(2000, 800, 4)),
    ("gap", lambda: gen_gap(15, 60, 5)),
    ("gap", lambda: gen_gap(25, 100, 5)),
    ("knapsack", lambda: gen_knapsack(150, 8, 7)),
    ("knapsack", lambda: gen_knapsack(250, 10, 7)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="", help="substring of instance names to deep-profile")
    args = ap.parse_args()

    print(f"# engine baseline  TL={TL}s  (highs={HAVE_HIGHS}, scip={HAVE_SCIP})")
    print(
        f"{'instance':14} | {'LP: discopt-root':>16} {'highs-LP':>10} {'LP x':>5} "
        f"| {'MILP: discopt':>14} {'nodes':>7} {'scip':>10} {'snodes':>7} {'MILP x':>6}"
    )
    print("-" * 118)
    for _fam, gen in INSTANCES:
        m = gen()
        cap = bool(args.profile) and args.profile in m.name
        d = solve_discopt_engine(m, capture_profile=cap)
        hl = solve_highs_lp(m)
        sc = solve_scip(m)
        root_ms = _rootsolve_ms(d["profile"]) if cap else None
        lp_x = (root_ms / 1000.0) / hl["wall"] if (root_ms and hl and hl["wall"] > 1e-9) else None
        milp_x = d["wall"] / sc["wall"] if (sc and sc["wall"] > 1e-9) else None
        root_s = f"{root_ms:.0f}ms" if root_ms else "-"
        highs_s = f"{hl['wall'] * 1000:.0f}ms" if hl else "-"
        lpx_s = f"{lp_x:.1f}" if lp_x else "-"
        disc_s = f"{d['wall']:.2f}s({d['status'][:3]})"
        scip_s = f"{sc['wall']:.2f}s" if sc else "-"
        snodes_s = str(sc["nodes"]) if sc else "-"
        milpx_s = f"{milp_x:.1f}" if milp_x else "-"
        print(
            f"{m.name:14} | "
            f"{root_s:>16} {highs_s:>10} {lpx_s:>5} | "
            f"{disc_s:>14} {d['nodes']:>7} {scip_s:>10} {snodes_s:>7} {milpx_s:>6}"
        )
        if cap and d["profile"]:
            print("    --- profile (" + m.name + ") ---")
            for line in d["profile"].splitlines():
                if line.strip():
                    print("    " + line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
