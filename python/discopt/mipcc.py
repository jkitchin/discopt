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

# Portfolio wait: the library default, "fast" (see _iis_milp.PORTFOLIO_WAITS:
# the other members stop clamp(t, 10, 30) s after the first completes at t).
PORTFOLIO_WAIT = "fast"

# Competition tolerance (SOLUTION_FORMAT.md, "Certificates"). Judging uses it.
CHECK_TOL = 1e-5
# Cap on the checker-settings re-solve of the final answer (it runs after the
# answer is written; an undecided result keeps the answer).
CHECKER_SOLVE_CAP_S = 60.0
# Our own acceptance is stricter on both sides, so a certificate at the edge of
# the checker's tolerance -- where its arithmetic (summation order) and ours may
# disagree -- is dropped rather than risked: one rejected certificate rejects
# the whole file, a dropped one costs only that element's credit.
_OTHERS_MAX = 0.5 * CHECK_TOL  # every other kept element satisfied within this
_OWN_MIN = 2.0 * CHECK_TOL  # the certified element violated by more than this
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
    from discopt._optional_highs import require_highspy

    highspy = require_highspy("discopt.mipcc (HiGHS MPS reader / checker)")
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
    """The checker's rule for one certificate, on the original data, with a
    safety margin: ``x`` satisfies every kept element except ``key`` within
    ``_OTHERS_MAX`` (the box of every column that stays integer applies too) and
    violates ``key`` by more than ``_OWN_MIN`` (the checker: both 1e-5).
    ``rows``/``lbk``/``ubk``/``held`` are the kept rows, kept lower/upper bounds
    and kept integralities, as boolean masks."""
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
    return others <= _OTHERS_MAX and float(own) > _OWN_MIN


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


def main(argv: Optional[list[str]] = None, t_launch: Optional[float] = None) -> int:
    """Run the entry program. ``t_launch`` (``time.perf_counter()``) is when the
    budget began: the command line passes this module's import time, the closest
    point to process launch it can see; an in-process caller's budget starts at
    the call. (Measured from import time, an in-process call made minutes after
    the import -- a pytest worker -- found its hard stop already past, and the
    watchdog ``os._exit``'ed the worker at once.)"""
    t0 = time.perf_counter() if t_launch is None else t_launch
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
    # What the file on disk holds, for the failure message below: _run updates
    # it after each write.
    on_disk = {"answer": "the full model"}

    finished = threading.Event()

    def watchdog() -> None:
        # Stands down once main() has returned: an in-process caller (the
        # tests) must not be killed later by a watchdog left behind.
        if finished.wait(max(hard_stop - (time.perf_counter() - t0), 0.0)):
            return
        print("mipcc: hard stop reached; exiting with the last written solution", flush=True)
        sys.stdout.flush()
        os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        return _run(instance, solution, threads, engine_budget, t0, hard_stop, on_disk)
    except Exception:
        # A bug, not a verdict: report it in full. The file on disk is a valid
        # answer -- the whole model (valid by the competition's premise) or, when
        # the failure came after it was written, the confirmed subsystem -- and a
        # non-zero exit would discard it (PROGRAM_INTERFACE.md, "Result").
        import traceback

        traceback.print_exc()
        print(f"mipcc: run failed; answer is {on_disk['answer']}", flush=True)
        return 0
    finally:
        finished.set()


def checker_rejects(data: dict, res, time_limit: float) -> bool:
    """Whether the official checker would find the kept subsystem feasible:
    one HiGHS solve with presolve on and both feasibility tolerances at
    :data:`CHECK_TOL`, as the checker runs it, and -- on a point -- the checker's
    1e-5 rule on every kept row, bound and integrality.

    The engine proves its answer infeasible twice (presolve-on HiGHS at HiGHS's
    tighter default tolerances, and an independent route), so a ``True`` here
    means a set infeasible in truth that the checker's looser judgement would
    still reject (#2145 review: the borderline zone). Undecided within
    ``time_limit``, or a point that fails the 1e-5 rule (the checker then
    retries tighter): ``False`` -- the doubly proven answer stands."""
    from discopt._optional_highs import require_highspy

    highspy = require_highspy("discopt.mipcc (HiGHS MPS reader / checker)")
    if time_limit <= 0.5:
        return False
    m, n = data["row_lo"].size, data["col_lb"].size
    rows = np.zeros(m, dtype=bool)
    rows[list(res.constraints)] = True
    held = np.zeros(n, dtype=bool)
    held[list(res.integrality)] = True
    lbk = held.copy()
    ubk = held.copy()
    for j, side in res.bounds:
        (lbk if side == "lower" else ubk)[j] = True
    inf = highspy.kHighsInf
    lb = np.where(lbk & (data["col_lb"] > -_INF), data["col_lb"], -inf)
    ub = np.where(ubk & (data["col_ub"] < _INF), data["col_ub"], inf)
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("presolve", "on")
    h.setOptionValue("mip_feasibility_tolerance", CHECK_TOL)
    h.setOptionValue("primal_feasibility_tolerance", CHECK_TOL)
    h.setOptionValue("time_limit", float(time_limit))
    # Wall-clock guard: HiGHS was measured overrunning its time_limit 7-10x on
    # presolve-free solves; stop it at the limit (plus a little) whatever it does.
    stop = time.perf_counter() + float(time_limit) + 1.0

    def _guard(event) -> None:
        event.interrupt(time.perf_counter() > stop)

    h.cbMipInterrupt.subscribe(_guard)
    h.cbSimplexInterrupt.subscribe(_guard)
    h.cbIpmInterrupt.subscribe(_guard)
    h.addVars(n, lb, ub)
    vt = highspy.HighsVarType
    h.changeColsIntegrality(
        n,
        np.arange(n, dtype=np.int32),
        np.array([vt.kInteger if k else vt.kContinuous for k in held]),
    )
    A = data["A"].tocsr()
    for r in np.flatnonzero(rows):
        s, e = A.indptr[r], A.indptr[r + 1]
        lo, hi = data["row_lo"][r], data["row_hi"][r]
        h.addRow(
            lo if lo > -_INF else -inf,
            hi if hi < _INF else inf,
            e - s,
            A.indices[s:e].astype(np.int32),
            A.data[s:e],
        )
    run = h.run()
    if run not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        raise RuntimeError(f"HiGHS failed on the checker-settings solve: {run}")
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return False
    x = np.asarray(h.getSolution().col_value, dtype=np.float64)
    act = A @ x
    rlo, rhi = data["row_lo"], data["row_hi"]
    viol = max(
        float(np.max(np.where(rows & (rlo > -_INF), rlo - act, 0.0), initial=0.0)),
        float(np.max(np.where(rows & (rhi < _INF), act - rhi, 0.0), initial=0.0)),
        float(
            np.max(np.where(lbk & (data["col_lb"] > -_INF), data["col_lb"] - x, 0.0), initial=0.0)
        ),
        float(
            np.max(np.where(ubk & (data["col_ub"] < _INF), x - data["col_ub"], 0.0), initial=0.0)
        ),
        float(np.max(np.where(held, np.abs(x - np.round(x)), 0.0), initial=0.0)),
    )
    return viol <= CHECK_TOL


def _run(
    instance: str,
    solution: str,
    threads: int,
    engine_budget: float,
    t0: float,
    hard_stop: float,
    on_disk: Optional[dict] = None,
) -> int:
    if on_disk is None:
        on_disk = {}
    data = read_mps(instance)
    from discopt._iis_milp import DEFAULT_PORTFOLIO, PORTFOLIO_WAITS, milp_iis_arrays

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
    elapsed = time.perf_counter() - t0
    try:
        res = milp_iis_arrays(
            arrays,
            portfolio=DEFAULT_PORTFOLIO,
            workers=max(1, min(threads, len(DEFAULT_PORTFOLIO))),
            total_time_limit=max(engine_budget - elapsed, 0.5),
            t_start=time.perf_counter(),
            grace=PORTFOLIO_WAITS[PORTFOLIO_WAIT][0],
            grace_window=PORTFOLIO_WAITS[PORTFOLIO_WAIT][1],
        )
    except TimeoutError as exc:
        # No subsystem was confirmed in the budget: the whole model (already
        # written) is the answer. Said loudly; it is a valid but poor result.
        print(f"mipcc: no confirmed subsystem in the budget ({exc}); answer is the full model")
        return 0
    # Write the doubly proven answer FIRST: whatever happens to the checker-
    # settings solve below (an overrun into the watchdog, an exception), the
    # answer on disk is this one, not the full-model fallback (#2145 review).
    doc = solution_json(data, res)
    write_atomic(solution, doc)
    on_disk["answer"] = "the confirmed subsystem"
    left = min(CHECKER_SOLVE_CAP_S, hard_stop - (time.perf_counter() - t0) - 5.0)
    if checker_rejects(data, res, left):
        # Infeasible in truth (two proofs), feasible within the checker's 1e-5:
        # submitting it would fail the run. The whole model replaces it.
        write_atomic(solution, {})
        on_disk["answer"] = "the full model"
        print(
            "mipcc: the checker's own settings find the subsystem feasible within "
            f"{CHECK_TOL}; answer is the full model",
            flush=True,
        )
        return 0
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
                wall=round(time.perf_counter() - t0, 2),
            )
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(t_launch=_T_LAUNCH))
