#!/usr/bin/env python3
"""#2212 throughput panel: cold dual (sign-matched logical start) vs cold primal.

The #1008 question, asked of the cold path #2212 proposes. #2212's entry
experiment (``entry_experiment.py``) showed that a dual simplex started from the
logical basis decides LPs the cold two-phase primal cannot. Before it replaces
the primal as the engine's cold path, it has to be no slower on the LPs the
primal already decides. That is the history ``SimplexOptions::cold_dual_start``
records: the cold *primal* ground to ``max_iter`` on equality-rich relaxations,
where the dual start was ~25x faster (QPLIB_1157), and the dual start stayed OFF
only because nothing had measured it broadly.

**LPs.** Root relaxations built the way #1008 built them
(``scratchpad/i1008/capture.py``): ``build_uniform_relaxation(model, box,
rlt_lineq=r)`` for ``r`` in {False, True}, rows ``A_ub x <= b_ub``. Equalities
reach the LP as two opposing rows, so these LPs are primal-degenerate. Brought to
standard form with one slack per row (``A x + s = b``, ``s >= 0``), so every row
has a logical column.

* ``--corpus nl``: every ``python/tests/data/minlplib_nl/*.nl`` (in-repo).
* ``--corpus qplib --qplib-dir DIR``: #1008's own selection, by filter from
  ``DIR/qplib_manifest.csv`` (usable oracle, continuous, nonconvex,
  ``nvars <= 400``, ``lift >= 300``, the ``--qplib-n`` largest by lift, plus the
  ``QPLIB_1157`` probe), never by hardcoded name (CLAUDE.md §2).

**Arms**, interleaved per LP with the order alternating by rep (§9):

* ``primal``: no start, i.e. today's cold path.
* ``dual``: the logical basis (slacks basic, each structural at the bound its cost
  sign selects). When a costed column's selected bound is open the start does not
  exist and the arm records ``refused``; the proposed cold path then falls back
  to the primal, so a refused LP costs what the primal costs.

Same entry for both (``solve_lp_warm_csc``, default options, ``tol 1e-9``), same
``--max-iter`` cap, no deadline (a deadline changes the options, see
``lp_bindings.rs``). HiGHS (presolve off, 1e-10, ``small_matrix_value`` 1e-12,
the oracle settings ``entry_experiment.py`` established) checks every verdict.

Prints one JSON line per LP as it goes (§10), a summary, and an executed-check
count; exits non-zero on zero checks (§6) or any verdict contradicting HiGHS.
Nothing is caught except a relaxation that cannot be built, which is recorded
as a row with its exception type and counted (§7).

Usage::

    python -u docs/dev/data/2212-cold-path/throughput_panel.py --corpus nl \\
        --reps 3 --max-iter 20000 --out nl.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))
import discopt  # noqa: E402
from discopt._rust import solve_lp_warm_csc_py  # noqa: E402

assert Path(discopt.__file__).resolve().is_relative_to(REPO), discopt.__file__

INF = 1e20


def _basis_consts() -> tuple[int, int, int]:
    """AT_LOWER, AT_UPPER, BASIC read from `crate::lp::basis`, never typed in.

    The first version of this script hard-coded BASIC=2, AT_UPPER=1; the engine's
    values are BASIC=1, AT_UPPER=2. Every start it built marked the wrong columns
    basic (#2212 README, retraction).
    """
    import re

    src = (REPO / "crates/discopt-core/src/lp/basis.rs").read_text()
    val = {k: int(v) for k, v in re.findall(r"pub const (\w+): i8 = (-?\d+);", src)}
    return val["AT_LOWER"], val["AT_UPPER"], val["BASIC"]


AT_LOWER, AT_UPPER, BASIC = _basis_consts()
assert (AT_LOWER, BASIC, AT_UPPER) == (0, 1, 2), (AT_LOWER, BASIC, AT_UPPER)
REL = 1e-6


def nl_models():
    from discopt.modeling.core import from_nl

    for p in sorted((REPO / "python/tests/data/minlplib_nl").glob("*.nl")):
        yield p.stem, (lambda p=p: from_nl(str(p)))


def qplib_models(qdir: str, n: int):
    import pandas as pd
    from discopt.interfaces import qplib

    df = pd.read_csv(os.path.join(qdir, "qplib_manifest.csv"))
    q = df[(df.usable_oracle) & (df.n_integral == 0) & (df.nonconvex)].copy()
    q["lift"] = q.obj_quad_nnz + q.con_quad_nnz
    q = q[(q.nvars <= 400) & (q.lift >= 300)].sort_values("lift", ascending=False)
    names = list(q.name)[:n]
    if "QPLIB_1157" in set(q.name) and "QPLIB_1157" not in names:
        names.append("QPLIB_1157")  # #1008's gate probe
    for nm in names:
        path = os.path.join(qdir, "qplib", f"{nm}.qplib")
        yield nm, (lambda path=path: qplib.to_model(qplib.read_qplib(path)))


def build(make, rlt: bool) -> dict:
    from discopt._relax.uniform_relax import build_uniform_relaxation

    m = make()
    lb = np.concatenate([np.ravel(v.lb) for v in m._variables]).astype(float)
    ub = np.concatenate([np.ravel(v.ub) for v in m._variables]).astype(float)
    M = build_uniform_relaxation(m, box=(lb, ub), rlt_lineq=rlt).model
    A = sp.csc_matrix(M._A_ub)
    rows, n = A.shape
    bl = np.asarray(M._bounds, float)
    lo = np.where(np.isfinite(bl[:, 0]), bl[:, 0], -INF)
    hi = np.where(np.isfinite(bl[:, 1]), bl[:, 1], INF)
    S = sp.hstack([A, sp.identity(rows, format="csc")]).tocsc()  # A x + s = b
    return {
        "A": S,
        "m": rows,
        "n": n + rows,
        "n_struct": n,
        "c": np.r_[np.asarray(M._c, float).ravel(), np.zeros(rows)],
        "b": np.asarray(M._b_ub, float).ravel(),
        "l": np.r_[lo, np.zeros(rows)],
        "u": np.r_[hi, np.full(rows, INF)],
    }


def logical_start(f: dict):
    n, ns = f["n"], f["n_struct"]
    st = np.empty(n, np.int8)
    for j in range(ns):
        cj = f["c"][j]
        if cj >= 0.0 and f["l"][j] > -INF:
            st[j] = AT_LOWER
        elif cj <= 0.0 and f["u"][j] < INF:
            st[j] = AT_UPPER
        else:
            return None
    st[ns:] = BASIC
    return st, np.arange(ns, n, dtype=np.int64)


def highs(f: dict) -> tuple[str, float | None]:
    import highspy

    h = highspy.Highs()
    for k, v in (
        ("output_flag", False),
        ("presolve", "off"),
        ("small_matrix_value", 1e-12),
        # Relaxation coefficients reach 1.4e15 (nvs05); HiGHS rejects the whole model
        # above its default 1e15 instead of solving it.
        ("large_matrix_value", 1e20),
        ("primal_feasibility_tolerance", 1e-10),
        ("dual_feasibility_tolerance", 1e-10),
    ):
        assert h.setOptionValue(k, v) == highspy.HighsStatus.kOk, k
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = f["n"], f["m"]
    lp.col_cost_ = f["c"]
    lp.col_lower_ = np.where(f["l"] <= -INF, -np.inf, f["l"])
    lp.col_upper_ = np.where(f["u"] >= INF, np.inf, f["u"])
    lp.row_lower_ = lp.row_upper_ = f["b"]
    A = f["A"]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    h.passModel(lp)
    h.run()
    st = h.getModelStatus()
    if st == highspy.HighsModelStatus.kOptimal:
        return "optimal", h.getInfo().objective_function_value
    if st == highspy.HighsModelStatus.kInfeasible:
        return "infeasible", None
    return str(st).split(".")[-1], None


def solve(f: dict, start, max_iter: int) -> dict:
    A = f["A"]
    cs, bv = start if start is not None else (None, None)
    t = time.perf_counter()
    status, _x, obj, iters, *_ = solve_lp_warm_csc_py(
        f["c"],
        f["m"],
        f["n"],
        A.indptr.astype(np.int64),
        A.indices.astype(np.int64),
        A.data,
        f["b"],
        f["l"],
        f["u"],
        cs,
        bv,
        1e-9,
        max_iter,
        None,
        False,
    )
    return {
        "status": status,
        "obj": float(obj),
        "iters": int(iters),
        "wall": time.perf_counter() - t,
    }


def verdict(r: dict, ref) -> str:
    """decided-agree | undecided | WRONG (a decided verdict HiGHS contradicts)."""
    if r["status"] not in ("optimal", "infeasible"):
        return "undecided"
    if ref[0] not in ("optimal", "infeasible"):
        return "undecided-oracle"
    if r["status"] != ref[0]:
        return "WRONG"
    if ref[0] == "optimal" and abs(r["obj"] - ref[1]) > REL * (1.0 + abs(ref[1])):
        return "WRONG"
    return "decided-agree"


def summarize(rows: list[dict], reps: int) -> int:
    """Print the summary from recorded rows; exit status 1 on 0 checks or a WRONG row."""
    arms_run = [r[a] for r in rows if "build_error" not in r for a in ("primal", "dual") if a in r]
    n_checks = len(arms_run)
    n_wrong = sum(a["verdict"] == "WRONG" for a in arms_run)
    built = [r for r in rows if "build_error" not in r]
    both = [
        r
        for r in built
        if "dual" in r
        and r["primal"]["verdict"] == "decided-agree"
        and r["dual"]["verdict"] == "decided-agree"
    ]
    ratios = [
        r["dual"]["wall_med"] / r["primal"]["wall_med"] for r in both if r["primal"]["wall_med"] > 0
    ]
    piv = [(r["dual"]["iters"] + 1) / (r["primal"]["iters"] + 1) for r in both]

    def count(arm, v):
        return sum(1 for r in built if arm in r and r[arm]["verdict"] == v)

    def cascade(r):
        """Proposed cold path: the dual from the logical start, else the primal.

        `solve_lp_warm_csc` already falls back to the cold primal inside the dual
        arm when the dual fails (`hda`: both arms report the same 3311 pivots), so
        the dual arm IS the cascade wherever the start exists. Adding a second
        primal run counted `hda` twice in the first version of this summary.
        """
        a = r["dual"] if "dual" in r else r["primal"]
        return a["wall_med"], a["verdict"] == "decided-agree"

    casc = [cascade(r) for r in built]
    print("\n# summary", flush=True)
    print(f"LPs built {len(built)} / {len(rows)} (build errors {len(rows) - len(built)})")
    print(
        f"dual start refused (open costed bound): "
        f"{sum(1 for r in built if r['dual_start'] == 'refused')}"
    )
    for arm in ("primal", "dual"):
        print(
            f"{arm:6s} decided {count(arm, 'decided-agree'):4d}  undecided "
            f"{count(arm, 'undecided'):3d}  WRONG {count(arm, 'WRONG')}"
        )
    print(f"cascade (dual, else primal) decided {sum(d for _, d in casc)} / {len(built)}")
    if ratios:
        g = math.exp(statistics.fmean(math.log(x) for x in ratios))
        print(
            f"both decided: {len(both)} LPs; wall dual/primal geomean {g:.3f}, median "
            f"{statistics.median(ratios):.3f}, min {min(ratios):.3f}, max {max(ratios):.3f}"
        )
        print(
            f"  pivots dual/primal geomean "
            f"{math.exp(statistics.fmean(math.log(x) for x in piv)):.3f}"
        )
        worst = sorted(both, key=lambda r: r["dual"]["wall_med"] / r["primal"]["wall_med"])[-5:]
        for r in worst:
            print(
                f"  slowest-for-dual {r['lp']:28s} m={r['m']:5d} primal "
                f"{r['primal']['wall_med'] * 1e3:8.1f} ms/{r['primal']['iters']} piv  dual "
                f"{r['dual']['wall_med'] * 1e3:8.1f} ms/{r['dual']['iters']} piv"
            )
    tp = sum(r["primal"]["wall_med"] for r in built)
    print(
        f"total wall: primal {tp:.2f} s, cascade {sum(w for w, _ in casc):.2f} s "
        f"(median of {reps} reps per LP)"
    )
    print(f"# load at end: {os.getloadavg()}")
    print(f"EXECUTED CHECKS: {n_checks}  WRONG: {n_wrong}", flush=True)
    return 1 if n_checks == 0 or n_wrong else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("nl", "qplib"), default="nl")
    ap.add_argument(
        "--qplib-dir",
        default=os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/qplib"),
    )
    ap.add_argument("--qplib-n", type=int, default=12)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows", default=None, help="summarize these recorded rows; solve nothing")
    args = ap.parse_args()
    if args.rows:
        rows = [json.loads(line) for line in open(args.rows) if line.startswith("{")]
        return summarize(rows, args.reps)
    assert args.out, "--out is required when solving"
    print(f"# load at start: {os.getloadavg()}", flush=True)

    src = nl_models() if args.corpus == "nl" else qplib_models(args.qplib_dir, args.qplib_n)
    rows, n_checks, n_wrong = [], 0, 0
    with open(args.out, "w") as out:
        for name, make in src:
            for rlt in (False, True):
                tag = f"{name}_rlt{int(rlt)}"
                try:
                    f = build(make, rlt)
                except Exception as exc:  # noqa: BLE001 -- recorded as a row, counted
                    row = {"lp": tag, "build_error": f"{type(exc).__name__}: {exc}"[:200]}
                    print(json.dumps(row), flush=True)
                    out.write(json.dumps(row) + "\n")
                    rows.append(row)
                    continue
                ref = highs(f)
                start = logical_start(f)
                arms = {"primal": [], "dual": []}
                for rep in range(args.reps):
                    order = ("primal", "dual") if rep % 2 == 0 else ("dual", "primal")
                    for arm in order:
                        if arm == "dual" and start is None:
                            continue
                        arms[arm].append(solve(f, start if arm == "dual" else None, args.max_iter))
                row = {
                    "lp": tag,
                    "m": f["m"],
                    "n": f["n"],
                    "highs": ref,
                    "dual_start": "ok" if start is not None else "refused",
                }
                for arm, runs in arms.items():
                    if not runs:
                        continue
                    first = runs[0]
                    # Deterministic per LP: every rep must agree on status/pivots.
                    assert all(
                        r["status"] == first["status"] and r["iters"] == first["iters"]
                        for r in runs
                    ), f"{tag}/{arm} not reproducible: {runs}"
                    v = verdict(first, ref)
                    n_checks += 1
                    n_wrong += v == "WRONG"
                    walls = [r["wall"] for r in runs]
                    row[arm] = {
                        "status": first["status"],
                        "obj": first["obj"],
                        "iters": first["iters"],
                        "verdict": v,
                        "wall_med": statistics.median(walls),
                        "wall_sd": statistics.pstdev(walls),
                    }
                print(json.dumps(row), flush=True)
                out.write(json.dumps(row) + "\n")
                rows.append(row)

    return summarize(rows, args.reps)


if __name__ == "__main__":
    sys.exit(main())
