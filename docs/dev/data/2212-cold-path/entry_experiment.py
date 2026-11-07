#!/usr/bin/env python3
"""#2212 entry experiments (CLAUDE.md §4), on the in-repo captured LP fixtures.

For every fixture under ``crates/discopt-core/src/{bnb,lp/simplex}/testdata/*_lp.json``
(the standard form ``min c'x  s.t.  A x = b,  l <= x <= u``, CSC, slack columns
included) this runs four arms and prints one JSON line per fixture:

* ``highs``   -- the oracle: HiGHS on the same LP (the 1e20 sentinel mapped to inf),
                 presolve off, feasibility tolerances 1e-10, ``small_matrix_value``
                 1e-12 (its floor). With its defaults HiGHS is not an oracle for
                 these LPs, measured: it drops entries below 1e-9, which turns
                 ``ex1252a`` (11 such entries) infeasible, and presolve moves
                 ``nvs01``'s optimum from 9.0833 to 9.1004. A fixture with entries
                 below 1e-12 is still altered, so it is marked ``oracle_exact: false``.
* ``warm``    -- the engine from the fixture's captured start basis (the caller's
                 inherited basis) via ``solve_lp_warm_csc``. That entry falls back to
                 the cold primal on any difficulty, so the profile counters say
                 whether the warm dual itself decided (``DualPrepAccept``,
                 ``DualColdFallbacks``, ``DualExit*``). Question 1 of the issue.
* ``cold``    -- the engine with no start: today's cold path (two-phase primal).
* ``logical`` -- the dual simplex from the sign-matched logical basis (each row's
                 logical column basic, each structural at the bound its cost sign
                 wants), the start #2163 and #2202 each patched into one caller.
                 Question 2 of the issue: is it a sound replacement cold path?
                 This arm uses the fixture's own unit columns and is refused when a
                 row has none (the #2163 ReLU LPs).
* ``augmented`` -- the same start in #2163's general form
                 (``ScaledProblem::logical_basis_dual``): append a logical column
                 ``-e_i`` per row, fixed at ``[b_i, b_i]``, right-hand side 0, so a
                 full logical basis always exists. Refused only when a costed
                 column's selected bound is open. Point and objective are mapped
                 back to the original columns before the checks below.

Every ``optimal`` answer is also checked without HiGHS: the returned point's
max relative row residual and bound violation, and the Neumaier-Shcherbina safe
lower bound from the returned duals (``ns_safe_bound_csc_py``), which must not
exceed the returned objective.

Also recorded per fixture: kappa_1 of the start basis (numpy, independent of the
engine), and whether the logical start exists (it is refused when a costed
column's selected bound is open, or a row has no unit logical column).

Every line asserts the loaded ``discopt`` is this checkout's (CLAUDE.md §8), and the
script ends with an executed-comparison count, exiting non-zero when it is 0 (§6).

Usage:  DISCOPT_PROFILE=1 python -u docs/dev/data/2212-cold-path/entry_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[4]
FIXTURE_DIRS = [
    REPO / "crates/discopt-core/src/bnb/testdata",
    REPO / "crates/discopt-core/src/lp/simplex/testdata",
]
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

sys.path.insert(0, str(REPO / "python"))
import discopt  # noqa: E402
from discopt._rust import (  # noqa: E402
    ns_safe_bound_csc_py,
    profile_counters_py,
    profile_reset_py,
    solve_lp_warm_csc_py,
)

assert Path(discopt.__file__).resolve().is_relative_to(REPO), discopt.__file__
assert os.environ.get("DISCOPT_PROFILE") == "1", "run with DISCOPT_PROFILE=1 (counters)"

COUNTERS = (
    "DualPrepAccept",
    "DualPrepRejectShape",
    "DualPrepRejectSingular",
    "DualPrepRejectDualInf",
    "DualPrepRejectInfBound",
    "DualColdFallbacks",
    "DualStallTrips",
    "WarmVerdictOptimal",
    "WarmVerdictInfeasible",
    "WarmVerdictNumerical",
    "WarmVerdictIterLimit",
    "Phase1EndBoxOk",
    "Phase1EndBoxViolated",
    "Phase1DriftDominates",
    "LpAuditBoundsFail",
    "LpAuditRowsFail",
    "LogicalBasisDualAttempts",
    "LogicalBasisDualDecided",
)


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    m, n = int(d["m"]), int(d["n"])
    A = sp.csc_matrix(
        (np.asarray(d["vals"], float), np.asarray(d["row_idx"]), np.asarray(d["col_ptr"])),
        shape=(m, n),
    )
    out = {
        "name": path.stem,
        "m": m,
        "n": n,
        "A": A,
        "b": np.asarray(d["b"], float),
        "c": np.asarray(d["c"], float),
        "l": np.asarray(d["l"], float),
        "u": np.asarray(d["u"], float),
        "start": None,
    }
    if "basic_vars" in d:
        out["start"] = (
            np.asarray(d["col_status"], np.int8),
            np.asarray(d["basic_vars"], np.int64),
        )
    return out


def highs(f: dict, tight: bool = True) -> tuple[str, float | None]:
    import highspy

    h = highspy.Highs()
    opts = [("output_flag", False), ("presolve", "off"), ("small_matrix_value", 1e-12)]
    if tight:
        opts += [("primal_feasibility_tolerance", 1e-10), ("dual_feasibility_tolerance", 1e-10)]
    for k, v in opts:
        assert h.setOptionValue(k, v) == highspy.HighsStatus.kOk, k
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = f["n"], f["m"]
    lp.col_cost_ = f["c"]
    big = lambda v: np.where(np.abs(v) >= INF, np.copysign(np.inf, v), v)  # noqa: E731
    lp.col_lower_, lp.col_upper_ = big(f["l"]), big(f["u"])
    lp.row_lower_ = lp.row_upper_ = f["b"]
    A = f["A"]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    h.passModel(lp)
    h.run()
    st = h.getModelStatus()
    if st == highspy.HighsModelStatus.kOptimal:
        return "optimal", h.getInfo().objective_function_value
    if st == highspy.HighsModelStatus.kInfeasible:
        return "infeasible", None
    return str(st).split(".")[-1], None


def logical_basis(f: dict) -> tuple[tuple[np.ndarray, np.ndarray] | None, str]:
    """Each row's unit logical column basic; structurals at their cost-sign bound."""
    A, m, n, c, lo, hi = f["A"], f["m"], f["n"], f["c"], f["l"], f["u"]
    nnz = np.diff(A.indptr)
    logical = [-1] * m
    for j in range(n - 1, -1, -1):  # prefer trailing columns (slacks are appended)
        if nnz[j] == 1 and c[j] == 0.0:
            i = A.indices[A.indptr[j]]
            if logical[i] < 0:
                logical[i] = j
    if min(logical) < 0:
        return None, "row without a zero-cost unit column"
    basic = set(logical)
    status = np.full(n, AT_LOWER, np.int8)
    for j in range(n):
        if j in basic:
            status[j] = BASIC
        elif c[j] < 0.0:
            if hi[j] >= INF:
                return None, "costed column with open selected bound"
            status[j] = AT_UPPER
        elif c[j] > 0.0 and lo[j] <= -INF:
            return None, "costed column with open selected bound"
        elif c[j] == 0.0 and lo[j] <= -INF:
            status[j] = AT_UPPER if hi[j] < INF else AT_LOWER  # free zero-cost: no choice
    return (status, np.asarray(logical, np.int64)), "ok"


def augmented(f: dict) -> tuple[dict, tuple[np.ndarray, np.ndarray]] | None:
    """#2163's augmented LP and its all-logical start, or None when refused."""
    m, n, c, lo, hi = f["m"], f["n"], f["c"], f["l"], f["u"]
    status = np.empty(n + m, np.int8)
    for j in range(n):
        if c[j] >= 0.0 and lo[j] > -INF:
            status[j] = AT_LOWER
        elif c[j] <= 0.0 and hi[j] < INF:
            status[j] = AT_UPPER
        else:
            return None
    status[n:] = BASIC
    g = dict(f)
    g["A"] = sp.hstack([f["A"], -sp.identity(m, format="csc")]).tocsc()
    g["n"] = n + m
    g["c"] = np.r_[c, np.zeros(m)]
    g["l"] = np.r_[lo, f["b"]]
    g["u"] = np.r_[hi, f["b"]]
    g["b"] = np.zeros(m)
    g["orig"] = f
    return g, (status, np.arange(n, n + m, dtype=np.int64))


def kappa1(f: dict, basic: np.ndarray) -> float:
    B = f["A"][:, basic].toarray()
    try:
        return float(np.linalg.cond(B, 1))
    except np.linalg.LinAlgError:
        return float("inf")


def run(f: dict, start, flip: bool) -> dict:
    profile_reset_py()
    A = f["A"]
    cs, bv = start if start is not None else (None, None)
    status, x, obj, iters, _cs, _bv, dual, _ray = solve_lp_warm_csc_py(
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
        100_000,
        None,
        flip,
    )
    ctr = profile_counters_py()
    out = {
        "status": status,
        "obj": float(obj) if status == "optimal" else None,
        "iters": int(iters),
        "ctr": {k: int(ctr.get(k, 0)) for k in COUNTERS if ctr.get(k, 0)},
    }
    if status == "optimal":
        x = np.asarray(x)
        if "orig" in f:  # augmented: check the structural part on the original LP
            x, f, A = x[: f["orig"]["n"]], f["orig"], f["orig"]["A"]
            obj = float(f["c"] @ x)
            out["obj"] = obj
        # Scaled by the row's own activity: a column parked on the 1e20 sentinel
        # (qplib2170) makes |b|-scaling report 0.5 for a 5e-21 relative residual.
        r = np.abs(A @ x - f["b"]) / (1.0 + abs(A) @ np.abs(x))
        viol = np.maximum(np.maximum(f["l"] - x, x - f["u"]), 0.0)
        safe = ns_safe_bound_csc_py(
            np.asarray(dual),
            f["c"],
            f["m"],
            f["n"],
            A.indptr.astype(np.int64),
            A.indices.astype(np.int64),
            A.data,
            f["b"],
            f["l"],
            f["u"],
        )
        out["max_rel_row_residual"] = float(r.max())
        out["max_bound_violation"] = float(viol.max())
        out["max_abs_x"] = float(np.abs(x).max())
        out["ns_safe_bound"] = safe
        out["safe_bound_ok"] = safe is None or safe <= obj + 1e-9 * (1.0 + abs(obj))
    return out


def agrees(arm: dict, ref: tuple[str, float | None]) -> bool | None:
    """None = undecided; True/False = decided and (dis)agreeing with HiGHS."""
    if arm["status"] not in ("optimal", "infeasible"):
        return None
    if arm["status"] != ref[0]:
        return False
    if ref[0] == "optimal":
        return abs(arm["obj"] - ref[1]) <= REL * (1.0 + abs(ref[1]))
    return True


def main() -> int:
    paths = sorted(p for d in FIXTURE_DIRS for p in d.glob("*_lp.json"))
    assert paths, "no fixtures found"
    n_cmp = 0
    for p in paths:
        f = load(p)
        flip = f["name"].startswith("nous1_")  # the spatial kernel's warm mode
        ref = highs(f)
        tiny = np.abs(f["A"].data)
        row = {
            "fixture": f["name"],
            "m": f["m"],
            "n": f["n"],
            "highs": ref,
            "highs_default_tol": highs(f, tight=False),
            "oracle_exact": bool(not np.any((tiny > 0) & (tiny < 1e-12))),
        }
        if f["start"] is not None:
            row["kappa1_start"] = kappa1(f, f["start"][1])
            row["warm"] = run(f, f["start"], flip)
            row["warm"]["agrees"] = agrees(row["warm"], ref)
            n_cmp += 1
        row["cold"] = run(f, None, flip)
        row["cold"]["agrees"] = agrees(row["cold"], ref)
        n_cmp += 1
        lb, why = logical_basis(f)
        row["logical_start"] = why
        if lb is not None:
            row["kappa1_logical"] = kappa1(f, lb[1])
            row["logical"] = run(f, lb, flip)
            row["logical"]["agrees"] = agrees(row["logical"], ref)
            n_cmp += 1
        aug = augmented(f)
        row["augmented_start"] = "ok" if aug else "costed column with open selected bound"
        if aug is not None:
            row["augmented"] = run(aug[0], aug[1], flip)
            row["augmented"]["agrees"] = agrees(row["augmented"], ref)
            n_cmp += 1
        print(json.dumps(row), flush=True)
    print(f"EXECUTED COMPARISONS: {n_cmp}", flush=True)
    return 0 if n_cmp else 1


if __name__ == "__main__":
    sys.exit(main())
