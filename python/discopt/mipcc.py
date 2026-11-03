"""MIP Competition 2027 entry point: ``python -m discopt.mipcc INSTANCE SOLUTION``.

Reads an infeasible MIP (``.mps`` / ``.mps.gz``), computes an irreducible
infeasible subsystem with integrality as an element (:func:`discopt._iis_milp.
milp_iis_arrays`, the default deletion-order portfolio) and writes the
competition's solution JSON (``SOLUTION_FORMAT.md`` of the MIPcc27 kit).

The model goes from HiGHS's MPS reader straight into the engine's arrays: no
:class:`~discopt.modeling.core.Model` and no Python expression per row, so the
read costs what HiGHS's parser costs.

The run contract (``PROGRAM_INTERFACE.md``): the budget is ``MIPCC_TIME_LIMIT``
wall-clock seconds from launch, including start-up; a run still going at the
limit is killed and its output discarded; exit 0 once the solution is written.
So:

* the engine gets the budget minus a margin for start-up, writing and exit;
* the file is written atomically (temporary file + rename), so it is never seen
  half-written;
* a watchdog thread exits the process at the hard stop whatever the engine is
  doing. Before the engine returns, the file holds the whole model with every
  element kept -- valid whenever the full model is infeasible, the competition's
  premise -- so even a watchdog exit leaves a checkable answer;
* every certificate is re-checked against the original MPS data with the
  checker's rule (absolute 1e-5 on every row, bound and integrality; the
  certified element violated by more than 1e-5) and dropped if it fails: one
  wrong certificate rejects the whole file, a missing one costs only that
  element's credit.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Optional

import numpy as np

_T_LAUNCH = time.perf_counter()  # earliest point this process can see

# Portfolio grace: once one member completes at elapsed time t, the others may
# run until GRACE * t. Time is a ranking criterion; size is too. Simulated on the
# 6-order smallset sweep (35 instances some member completes, 60 s): waiting for
# every member gives total size 3503 and 33 at-or-below Gurobi, GRACE = 2 gives
# 3507 and 31, GRACE = 1 gives 3526 and 30 -- while waiting for every member
# keeps the run alive to the full budget whenever one member never completes.
GRACE = 2.0

# Competition tolerance (SOLUTION_FORMAT.md, "Certificates"). Judging uses it.
CHECK_TOL = 1e-5
_INF = 1e20  # the engine's unbounded sentinel


def budget(limit: float) -> tuple[float, float]:
    """``(engine_budget, hard_stop)`` in seconds after launch for a run limit of
    ``limit`` seconds. The margin covers container start-up before Python runs
    (unseen here), the final certificate check and write, and interpreter exit.
    """
    margin = min(max(10.0, 0.05 * limit), 120.0)
    hard = max(limit - margin / 2.0, 1.0)
    return max(limit - margin, 0.5), hard


def read_mps(path: str) -> dict:
    """The model's arrays and names, via HiGHS's MPS reader."""
    import highspy
    import scipy.sparse as sp

    tmpdir = None
    src = path
    if path.endswith(".gz"):
        # HiGHS's reader takes a plain file; /tmp is writable in the sandbox.
        tmpdir = tempfile.mkdtemp(prefix="mipcc_")
        src = os.path.join(tmpdir, "model.mps")
        with gzip.open(path, "rb") as f, open(src, "wb") as g:
            shutil.copyfileobj(f, g)
    try:
        h = highspy.Highs()
        h.silent()
        st = h.readModel(src)
        if st == highspy.HighsStatus.kError:
            raise ValueError(f"HiGHS could not read {path!r}")
        lp = h.getLp()
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
    n, m = lp.num_col_, lp.num_row_
    a = lp.a_matrix_
    if a.format_ == highspy.MatrixFormat.kRowwise:
        A = sp.csr_matrix((np.array(a.value_), np.array(a.index_), np.array(a.start_)), (m, n))
    else:
        A = sp.csc_matrix(
            (np.array(a.value_), np.array(a.index_), np.array(a.start_)), (m, n)
        ).tocsr()
    A.sum_duplicates()
    integrality = np.array([int(t) for t in lp.integrality_], dtype=int)
    if integrality.size == 0:
        integrality = np.zeros(n, dtype=int)
    if np.any((integrality != 0) & (integrality != 1)):
        # Semi-continuous / semi-integer columns are not IIS elements this
        # engine models; refuse rather than silently treat them as something else.
        raise NotImplementedError("semi-continuous or semi-integer columns")

    def sentinel(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).copy()
        v[v <= -_INF] = -_INF
        v[v >= _INF] = _INF
        return v

    col_names = list(lp.col_names_) if len(lp.col_names_) == n else [f"C{j}" for j in range(n)]
    row_names = list(lp.row_names_) if len(lp.row_names_) == m else [f"R{i}" for i in range(m)]
    return dict(
        A=A,
        row_lo=sentinel(lp.row_lower_),
        row_hi=sentinel(lp.row_upper_),
        col_lb=sentinel(lp.col_lower_),
        col_ub=sentinel(lp.col_upper_),
        is_int=integrality == 1,
        col_names=col_names,
        row_names=row_names,
    )


def certificate_ok(
    data: dict, rows: np.ndarray, lbk: np.ndarray, ubk: np.ndarray, held: np.ndarray, key, x
) -> bool:
    """The checker's rule for one certificate, on the original data: ``x``
    satisfies every kept element except ``key`` within :data:`CHECK_TOL` (the box
    of every column that stays integer applies too) and violates ``key`` by more
    than :data:`CHECK_TOL`. ``rows``/``lbk``/``ubk``/``held`` are the kept rows,
    kept lower/upper bounds and kept integralities, as boolean masks."""
    A, rlo, rhi = data["A"], data["row_lo"], data["row_hi"]
    clb, cub = data["col_lb"], data["col_ub"]
    act = A @ x
    row_v = np.maximum(np.where(rlo > -_INF, rlo - act, 0.0), np.where(rhi < _INF, act - rhi, 0.0))
    lb_v = np.where(clb > -_INF, clb - x, 0.0)
    ub_v = np.where(cub < _INF, x - cub, 0.0)
    int_v = np.abs(x - np.round(x))
    rows, lbk, ubk, held = rows.copy(), lbk.copy(), ubk.copy(), held.copy()
    kind = key[0]
    if kind == "constraint":
        rows[key[1]] = False
        own = row_v[key[1]]
    elif kind == "bound":
        (lbk if key[2] == "lower" else ubk)[key[1]] = False
        own = (lb_v if key[2] == "lower" else ub_v)[key[1]]
    elif kind == "integrality":
        # Relaxing integrality keeps the column's bounds.
        held[key[1]] = False
        lbk[key[1]] = True
        ubk[key[1]] = True
        own = int_v[key[1]]
    else:
        raise ValueError(f"unknown element kind {kind!r}")
    lbk |= held
    ubk |= held
    others = max(
        float(np.max(row_v[rows], initial=0.0)),
        float(np.max(lb_v[lbk], initial=0.0)),
        float(np.max(ub_v[ubk], initial=0.0)),
        float(np.max(int_v[held], initial=0.0)),
    )
    return others <= CHECK_TOL and float(own) > CHECK_TOL


def solution_json(data: dict, res=None) -> dict:
    """The competition JSON for ``res`` (a :class:`MilpIIS`), or every element
    kept when ``res`` is None. Relaxed elements are listed explicitly (an
    unlisted element defaults to "keep"); certificates that fail
    :func:`certificate_ok` are left out."""
    n, m = data["col_lb"].size, data["row_lo"].size
    rn, cn = data["row_names"], data["col_names"]
    if res is None:
        return {}
    rows = np.zeros(m, dtype=bool)
    rows[list(res.constraints)] = True
    held = np.zeros(n, dtype=bool)
    held[list(res.integrality)] = True
    lbk = np.zeros(n, dtype=bool)
    ubk = np.zeros(n, dtype=bool)
    for j, side in res.bounds:
        (lbk if side == "lower" else ubk)[j] = True
    good = {}
    n_bad = 0
    for key, x in res.certificates.items():
        x = np.asarray(x, dtype=np.float64)
        if certificate_ok(data, rows, lbk, ubk, held, key, x):
            good[key] = x
        else:
            n_bad += 1

    def cert(key) -> Optional[dict]:
        x = good.get(key)
        if x is None:
            return None
        nz = np.flatnonzero(x)
        return {cn[j]: float(x[j]) for j in nz}

    def keep(key) -> dict:
        c = cert(key)
        return {"action": "keep"} if c is None else {"action": "keep", "certificate": c}

    out_rows: dict = {}
    for i in range(m):
        if not rows[i]:
            out_rows[rn[i]] = {"action": "relax"}
        else:
            out_rows[rn[i]] = keep(("constraint", i))
    out_cols: dict = {}
    for j in range(n):
        entry: dict = {}
        if data["is_int"][j]:
            if held[j]:
                entry["integrality"] = keep(("integrality", j))
                # A held column's bounds always apply; they are not elements.
                out_cols[cn[j]] = entry
                continue
            entry["integrality"] = {"action": "relax"}
        for side, kept, finite in (
            ("lower", lbk[j], data["col_lb"][j] > -_INF),
            ("upper", ubk[j], data["col_ub"][j] < _INF),
        ):
            if not finite:
                continue
            if kept:
                entry[f"{side}_bound"] = keep(("bound", j, side))
            else:
                entry[f"{side}_bound"] = {"action": "relax"}
        if entry:
            out_cols[cn[j]] = entry
    solution_json.n_bad = n_bad  # type: ignore[attr-defined]
    return {"rows": out_rows, "columns": out_cols}


def write_atomic(path: str, doc: dict) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".mipcc_", dir=d)
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(tmp, path)


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m discopt.mipcc INSTANCE SOLUTION", file=sys.stderr)
        return 2
    instance, solution = argv
    limit = float(os.environ.get("MIPCC_TIME_LIMIT", "3600"))
    threads = int(os.environ.get("MIPCC_THREADS", str(os.cpu_count() or 1)))
    engine_budget, hard_stop = budget(limit)

    # Safety net: the empty document keeps every element (the whole model).
    write_atomic(solution, {})

    def watchdog() -> None:
        left = hard_stop - (time.perf_counter() - _T_LAUNCH)
        if left > 0:
            time.sleep(left)
        print("mipcc: hard stop reached; exiting with the last written solution", flush=True)
        sys.stdout.flush()
        os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        return _run(instance, solution, threads, engine_budget)
    except Exception:
        # A bug, not a verdict: report it in full. The file on disk is the whole
        # model, a valid answer by the competition's premise, and a non-zero exit
        # would discard it (PROGRAM_INTERFACE.md, "Result").
        import traceback

        traceback.print_exc()
        print("mipcc: engine failed; answer is the full model", flush=True)
        return 0


def _run(instance: str, solution: str, threads: int, engine_budget: float) -> int:
    data = read_mps(instance)
    from discopt._iis_milp import DEFAULT_PORTFOLIO, milp_iis_arrays

    m = data["row_lo"].size
    arrays = (
        data["A"],
        data["row_lo"],
        data["row_hi"],
        {i: np.array([i], dtype=np.int64) for i in range(m)},
        data["col_lb"],
        data["col_ub"],
        data["is_int"],
    )
    elapsed = time.perf_counter() - _T_LAUNCH
    try:
        res = milp_iis_arrays(
            arrays,
            portfolio=DEFAULT_PORTFOLIO,
            workers=max(1, min(threads, len(DEFAULT_PORTFOLIO))),
            total_time_limit=max(engine_budget - elapsed, 0.5),
            t_start=time.perf_counter(),
            grace=GRACE,
        )
    except TimeoutError as exc:
        # No subsystem was confirmed in the budget: the whole model (already
        # written) is the answer. Said loudly; it is a valid but poor result.
        print(f"mipcc: no confirmed subsystem in the budget ({exc}); answer is the full model")
        return 0
    doc = solution_json(data, res)
    write_atomic(solution, doc)
    size = len(res.constraints) + len(res.bounds) + len(res.integrality)
    print(
        "mipcc: "
        + json.dumps(
            dict(
                size=size,
                complete=res.complete,
                certificates=len(res.certificates),
                certificates_dropped=getattr(solution_json, "n_bad", 0),
                order=res.order,
                wall=round(time.perf_counter() - _T_LAUNCH, 2),
            )
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
