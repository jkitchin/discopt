"""MIP Competition 2027 entry point: ``python -m discopt.mipcc INSTANCE SOLUTION``.

Reads an infeasible MIP (``.mps`` / ``.mps.gz``), computes an irreducible
infeasible subsystem with integrality as an element (:func:`discopt._iis_milp.
milp_iis_arrays`, the default deletion-order portfolio) and writes the
competition's solution JSON (``SOLUTION_FORMAT.md`` of the MIPcc27 kit).

The model goes from discopt's MPS reader (``discopt._mps``, Rust, an exact
reproduction of HiGHS's) straight into the engine's arrays: no
:class:`~discopt.modeling.core.Model` and no Python expression per row.

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

import json
import os
import sys
import tempfile
import threading
import time
from typing import Optional

import numpy as np

from discopt._optional_highs import run_checked, wrap_callback

_T_LAUNCH = time.perf_counter()  # earliest point this process can see

# Portfolio wait: the library default, "fast" (see _iis_milp.PORTFOLIO_WAITS:
# the other members stop clamp(t, 10, 30) s after the first completes at t, but
# not before half the budget has elapsed). MIPCC_PORTFOLIO_WAIT=scaled is the
# opt-in that trades size for time (see _iis_milp.PORTFOLIO_WAITS).
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
    """The model's arrays and names, via discopt's own MPS reader (``discopt._mps``,
    an exact reproduction of HiGHS's ``readModel`` -> ``getLp``; highs-parity P6).

    The matrix-entry cap is lifted (``large_matrix_value=inf``) because the
    competition checker sets that option before ``readModel``: a model it scores
    with entries of 1e15 or more (MIPLIB ``control30-5-10-4``) must not be refused
    here, which would leave the whole model as the answer."""
    from discopt._mps import read_mps as _read

    d = _read(path, large_matrix_value=float("inf"))
    n, m = d["n"], d["m"]
    A = d["A"].tocsr()
    A.sum_duplicates()
    integrality = d["integrality"]
    if np.any((integrality != 0) & (integrality != 1)):
        # Semi-continuous / semi-integer columns are not IIS elements this
        # engine models; refuse rather than silently treat them as something else.
        raise NotImplementedError("semi-continuous or semi-integer columns")

    def sentinel(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).copy()
        v[v <= -_INF] = -_INF
        v[v >= _INF] = _INF
        return v

    col_names = d["col_names"] if len(d["col_names"]) == n else [f"C{j}" for j in range(n)]
    row_names = d["row_names"] if len(d["row_names"]) == m else [f"R{i}" for i in range(m)]
    return dict(
        A=A,
        row_lo=sentinel(d["row_lo"]),
        row_hi=sentinel(d["row_hi"]),
        col_lb=sentinel(d["col_lb"]),
        col_ub=sentinel(d["col_ub"]),
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
    solution_json.n_good = len(good)  # type: ignore[attr-defined]
    return {"rows": out_rows, "columns": out_cols}


def write_atomic(path: str, doc: dict) -> None:
    """Write ``doc`` to ``path`` as a regular, world-readable file, atomically.

    ``os.replace`` swaps the directory entry itself, so a symlink already at
    ``path`` is replaced by a regular file, never written through (the harness
    rejects a symlink). ``mkstemp`` creates the file ``0600``; the container runs
    under a uid the organizers choose, and unless that uid maps onto the host user
    of the harness that copies the answer out of ``/out`` (``--userns=keep-id``
    does; a plain ``--user <uid>`` on a Linux host does not), a ``0600`` file is
    unreadable to it. So the mode is set to ``0644`` before the rename."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".mipcc_", dir=d)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        # Leave no stray temporary file next to the answer.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


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


# The official checker's retry tolerance (``RETRY_SOLVER_TOLERANCE`` in its
# ``backend/base.py``): a "feasible" point that misses CHECK_TOL on the original
# rows earns one more solve at this tolerance.
_CHECK_RETRY_TOL = 1e-9


def _masks(data: dict, sub) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, lbk, ubk, held)`` boolean masks of the kept subsystem ``sub``
    (anything with ``constraints`` / ``bounds`` / ``integrality``). A held
    column's bounds always apply, as in the checker (they are not listed for it)."""
    m, n = data["row_lo"].size, data["col_lb"].size
    rows = np.zeros(m, dtype=bool)
    rows[list(sub.constraints)] = True
    held = np.zeros(n, dtype=bool)
    held[list(sub.integrality)] = True
    lbk = held.copy()
    ubk = held.copy()
    for j, side in sub.bounds:
        (lbk if side == "lower" else ubk)[j] = True
    return rows, lbk, ubk, held


def _violations(data: dict, x: np.ndarray) -> tuple[np.ndarray, ...]:
    """``x``'s violation of every element of the model -- rows, lower bounds,
    upper bounds, integralities -- 0 where satisfied or vacuous."""
    act = data["A"] @ x
    rlo, rhi = data["row_lo"], data["row_hi"]
    clb, cub = data["col_lb"], data["col_ub"]
    row_v = np.maximum(np.where(rlo > -_INF, rlo - act, 0.0), np.where(rhi < _INF, act - rhi, 0.0))
    lb_v = np.where(clb > -_INF, clb - x, 0.0)
    ub_v = np.where(cub < _INF, x - cub, 0.0)
    int_v = np.where(data["is_int"], np.abs(x - np.round(x)), 0.0)
    return row_v, lb_v, ub_v, int_v


def _max_kept_violation(data: dict, masks: tuple, x: np.ndarray) -> float:
    """The checker's rule on a point: the largest violation of a kept element."""
    rows, lbk, ubk, held = masks
    row_v, lb_v, ub_v, int_v = _violations(data, x)
    return max(
        float(np.max(row_v[rows], initial=0.0)),
        float(np.max(lb_v[lbk], initial=0.0)),
        float(np.max(ub_v[ubk], initial=0.0)),
        float(np.max(int_v[held], initial=0.0)),
    )


def checker_verdict(data: dict, sub, time_limit: float) -> tuple[str, Optional[np.ndarray]]:
    """The official checker's verdict on the kept subsystem ``sub``, predicted by
    making the checker's own solve (``mipcc_iis_checker``, HiGHS backend,
    ``BaseBackend.solve_subsystem``): a fresh ``HighsLp`` with every column, the
    kept rows, bounds and integralities, default options except both feasibility
    tolerances at :data:`CHECK_TOL`. ``kInfeasible`` is ``"infeasible"``; a
    feasible primal solution that satisfies every kept element within CHECK_TOL
    (absolute) is ``"feasible"``; one that misses earns one more solve at
    :data:`_CHECK_RETRY_TOL`, and a miss there too is ``"unknown"``; anything else
    -- a solve error (chinneck misc06inf: HiGHS's "untransformed violations" ends
    the run in ``kError``), a time limit, no primal solution -- is ``"unknown"``.
    The checker scores ``"unknown"`` as a failure, exactly as it does
    ``"feasible"``: only ``"infeasible"`` passes.

    Returns ``(verdict, point)``, the point being the last primal solution HiGHS
    reported (feasible or not; ``None`` when it reported none): the direction
    :func:`_repair` adds elements back along. ``time_limit`` caps the whole
    verdict (both solves), with a wall-clock guard, as HiGHS was measured
    overrunning its own limit."""
    from discopt._optional_highs import require_highspy

    highspy = require_highspy(
        "discopt.mipcc (the HiGHS re-solve that mirrors the official checker)"
    )
    masks = _masks(data, sub)
    rows, lbk, ubk, held = masks
    n = data["col_lb"].size
    inf = highspy.kHighsInf
    lb = np.where(lbk & (data["col_lb"] > -_INF), data["col_lb"], -inf)
    ub = np.where(ubk & (data["col_ub"] < _INF), data["col_ub"], inf)
    rlo = data["row_lo"][rows]
    rhi = data["row_hi"][rows]
    csc = data["A"][np.flatnonzero(rows)].tocsc()
    vt = highspy.HighsVarType
    stop = time.perf_counter() + float(time_limit)
    point: Optional[np.ndarray] = None
    for tol in (CHECK_TOL, _CHECK_RETRY_TOL):
        left = stop - time.perf_counter()
        if left <= 0.05:
            return "unknown", point
        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = int(rows.sum())
        lp.col_cost_ = np.zeros(n, dtype=np.float64)
        lp.col_lower_ = lb
        lp.col_upper_ = ub
        lp.integrality_ = [vt.kInteger if k else vt.kContinuous for k in held]
        lp.row_lower_ = np.where(rlo > -_INF, rlo, -inf)
        lp.row_upper_ = np.where(rhi < _INF, rhi, inf)
        mat = highspy.HighsSparseMatrix()
        mat.format_ = highspy.MatrixFormat.kColwise
        mat.start_ = csc.indptr.tolist()
        mat.index_ = csc.indices.tolist()
        mat.value_ = csc.data.tolist()
        lp.a_matrix_ = mat
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("primal_feasibility_tolerance", tol)
        h.setOptionValue("mip_feasibility_tolerance", tol)
        h.setOptionValue("time_limit", float(left))
        # Wall-clock guard: HiGHS was measured overrunning its time_limit 7-10x on
        # presolve-free solves; stop it at the limit (plus a little) whatever it does.
        guard_at = time.perf_counter() + float(left) + 1.0

        def _guard(event, guard_at=guard_at) -> None:
            event.interrupt(time.perf_counter() > guard_at)

        # #2221: wrapped so an exception in the guard is re-raised after ``run``.
        guard = wrap_callback(h, _guard)
        h.cbMipInterrupt.subscribe(guard)
        h.cbSimplexInterrupt.subscribe(guard)
        h.cbIpmInterrupt.subscribe(guard)
        if h.passModel(lp) == highspy.HighsStatus.kError:
            return "unknown", point  # the checker could not solve it either
        # The run status is not consulted, as the checker does not consult it: a
        # solve error leaves a model status that is neither infeasible nor a point.
        run_checked(h)
        if h.getModelStatus() == highspy.HighsModelStatus.kInfeasible:
            return "infeasible", point
        sol = h.getSolution()
        if sol.value_valid:
            point = np.asarray(sol.col_value, dtype=np.float64)
        if h.getInfo().primal_solution_status != highspy.SolutionStatus.kSolutionStatusFeasible:
            return "unknown", point
        if point is not None and _max_kept_violation(data, masks, point) <= CHECK_TOL:
            return "feasible", point
    return "unknown", point


def checker_rejects(data: dict, res, time_limit: float) -> bool:
    """Whether the official checker is predicted to fail the kept subsystem:
    :func:`checker_verdict` is anything but ``"infeasible"`` (an error or an
    undecided solve is a predicted failure, as the checker scores it). With no
    time to make the solve (``time_limit <= 0.5``) nothing is predicted and
    ``False`` is returned: the doubly proven answer stands."""
    if time_limit <= 0.5:
        return False
    return checker_verdict(data, res, time_limit)[0] != "infeasible"


def _all_keys(data: dict) -> list[tuple]:
    """Every element of the model: rows, finite bounds, integralities."""
    m, n = data["row_lo"].size, data["col_lb"].size
    keys: list[tuple] = [("constraint", i) for i in range(m)]
    for j in range(n):
        if data["col_lb"][j] > -_INF:
            keys.append(("bound", j, "lower"))
        if data["col_ub"][j] < _INF:
            keys.append(("bound", j, "upper"))
    keys.extend(("integrality", int(j)) for j in np.flatnonzero(data["is_int"]))
    return keys


def _keys_of(data: dict, sub) -> set:
    """``sub``'s elements as keys, each held column's finite bounds listed
    explicitly. They apply whatever the list says, so the checker sees the same
    subsystem; listed, they stay when a deletion step drops the integrality, as
    relaxing an integrality keeps the bounds under the checker's rule."""
    keys: set[tuple] = {("constraint", int(i)) for i in sub.constraints}
    keys |= {("bound", int(j), side) for j, side in sub.bounds}
    for j in sub.integrality:
        keys.add(("integrality", int(j)))
        if data["col_lb"][j] > -_INF:
            keys.add(("bound", int(j), "lower"))
        if data["col_ub"][j] < _INF:
            keys.add(("bound", int(j), "upper"))
    return keys


class _Sub:
    """A kept subsystem as :func:`checker_verdict` and :func:`solution_json` read it."""

    def __init__(self, keys, certificates: dict):
        self.constraints = sorted(k[1] for k in keys if k[0] == "constraint")
        self.bounds = sorted((k[1], k[2]) for k in keys if k[0] == "bound")
        self.integrality = sorted(k[1] for k in keys if k[0] == "integrality")
        self.certificates = certificates


def _violated_keys(data: dict, keys: set, x: np.ndarray) -> list[tuple]:
    """The elements of the model NOT in ``keys`` that ``x`` violates by more than
    CHECK_TOL, most violated first."""
    row_v, lb_v, ub_v, int_v = _violations(data, x)
    out: list[tuple[float, tuple]] = []
    for i in np.flatnonzero(row_v > CHECK_TOL):
        out.append((float(row_v[i]), ("constraint", int(i))))
    for j in np.flatnonzero(lb_v > CHECK_TOL):
        out.append((float(lb_v[j]), ("bound", int(j), "lower")))
    for j in np.flatnonzero(ub_v > CHECK_TOL):
        out.append((float(ub_v[j]), ("bound", int(j), "upper")))
    for j in np.flatnonzero(int_v > CHECK_TOL):
        out.append((float(int_v[j]), ("integrality", int(j))))
    out = [vk for vk in out if vk[1] not in keys]
    out.sort(key=lambda vk: -vk[0])
    return [k for _, k in out]


def _size(data: dict, sub) -> int:
    """Kept elements as the checker counts them: a held column's bounds are not
    elements (:func:`_keys_of` lists them; the engine's answers may not)."""
    held = set(sub.integrality)
    return (
        len(sub.constraints) + sum(1 for j, _ in sub.bounds if j not in held) + len(sub.integrality)
    )


class _NoTime(Exception):
    """The deadline of :func:`_repair` passed."""


def _repair(data: dict, base, x: Optional[np.ndarray], deadline: float, log: dict):
    """A subsystem the checker's solve finds infeasible, grown from ``base`` (a
    confirmed subsystem it does not) by adding elements back, then shrunk by a
    deletion filter that judges with :func:`checker_verdict`; ``None`` if none is
    found before ``deadline`` (``time.perf_counter()``).

    Growing: while the checker's solve returns a point, every element of the whole
    model that the point violates by more than CHECK_TOL is added (the elements
    whose removal let the 1e-5 gap open). Each round adds at least one -- a point
    violating nothing would make the whole model feasible to the checker, against
    the competition's premise -- so the loop ends. With no point (a solve error
    that left none) the rest of the model is added in doubling chunks.
    Shrinking: the added elements, then the base elements whose certificate no
    longer holds in the grown set (one that still holds shows its element is
    needed: dropping it leaves a point the checker accepts), are dropped in
    chunks, halved on failure. Every adopted step keeps a set the checker's solve
    found infeasible, so stopping at the deadline is safe."""

    def verdict(keys):
        left = deadline - time.perf_counter()
        if left <= 0.5:
            raise _NoTime
        log["solves"] = log.get("solves", 0) + 1
        return checker_verdict(data, _Sub(keys, {}), min(CHECKER_SOLVE_CAP_S, left))

    base_keys = _keys_of(data, base)
    keys = set(base_keys)
    added: list[tuple] = []
    try:
        while True:
            more = _violated_keys(data, keys, x) if x is not None else []
            if not more:
                rest = [k for k in _all_keys(data) if k not in keys]
                if not rest:
                    return None  # grown to the whole model: the caller's last resort
                more = rest[: max(1, len(added))]
            keys |= set(more)
            added.extend(more)
            v, x = verdict(keys)
            if v == "infeasible":
                break
    except _NoTime:
        log["grow_cut"] = True
        return None
    log["added"] = len(added)
    certs = {k: np.asarray(c, dtype=np.float64) for k, c in base.certificates.items()}
    rows, lbk, ubk, held = _masks(data, _Sub(keys, {}))
    needed = {k for k, c in certs.items() if certificate_ok(data, rows, lbk, ubk, held, k, c)}
    # Latest added first (the most violated were added first: likeliest needed).
    groups = [added[::-1], sorted(base_keys - needed, key=str)]
    found: dict = {}
    try:
        for group in groups:
            stack = [group]
            while stack:
                # A held column's bounds apply whatever the list says: not elements.
                g = [
                    k
                    for k in stack.pop()
                    if k in keys and not (k[0] == "bound" and ("integrality", k[1]) in keys)
                ]
                if not g:
                    continue
                trial = keys - set(g)
                v, xp = verdict(trial)
                if v == "infeasible":
                    keys = trial
                    log["dropped"] = log.get("dropped", 0) + len(g)
                elif len(g) > 1:
                    half = len(g) // 2
                    stack.append(g[half:])
                    stack.append(g[:half])
                elif v == "feasible" and xp is not None:
                    found[g[0]] = xp  # a certificate candidate; solution_json checks it
    except _NoTime:
        log["filter_cut"] = True
    certs.update(found)
    return _Sub(keys, {k: c for k, c in certs.items() if k in keys})


def _checked_answer(data: dict, res, deadline: float, log: dict, on_reject=None):
    """The answer to write: the first of ``res`` and its confirmed ``fallbacks``
    (ranked) that :func:`checker_verdict` finds infeasible, else a :func:`_repair`
    of ``res``, else ``None`` (the whole model). Every candidate is infeasible by
    the engine's two proofs; this only predicts the checker's looser judgement.
    With no time to check ``res`` at all it is returned unchecked: nothing was
    predicted. ``on_reject()`` is called once ``res`` is predicted to fail (the
    entry then puts the whole model on disk while the search goes on)."""
    cands = [res, *(getattr(res, "fallbacks", None) or [])]
    point = None
    for k, cand in enumerate(cands):
        left = deadline - time.perf_counter()
        if left <= 0.5:
            if k == 0:
                log["unchecked"] = True
                return res
            break
        v, x = checker_verdict(data, cand, min(CHECKER_SOLVE_CAP_S, left))
        log.setdefault("verdicts", []).append(v)
        if v == "infeasible":
            log["chosen"] = "answer" if k == 0 else f"fallback {k}"
            return cand
        if k == 0:
            point = x
            if on_reject is not None:
                on_reject()
    rep = _repair(data, res, point, deadline, log)
    if rep is not None:
        log["chosen"] = "repair"
    return rep


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
    # The IIS engine follows DISCOPT_LP_ENGINE (DISCOPT_IIS_ENGINE overrides). The
    # entry's own HiGHS override (#2225) became redundant with the engine switch.
    from discopt._iis_milp import _resolve_iis_engine

    engine = _resolve_iis_engine(None)
    wait = os.environ.get("MIPCC_PORTFOLIO_WAIT", PORTFOLIO_WAIT)
    if wait not in PORTFOLIO_WAITS:
        raise ValueError(f"MIPCC_PORTFOLIO_WAIT={wait!r}: not one of {sorted(PORTFOLIO_WAITS)}")
    try:
        res = milp_iis_arrays(
            arrays,
            iis_engine=engine,
            portfolio=DEFAULT_PORTFOLIO,
            workers=max(1, min(threads, len(DEFAULT_PORTFOLIO))),
            threads=threads,
            total_time_limit=max(engine_budget - elapsed, 0.5),
            t_start=time.perf_counter(),
            grace=PORTFOLIO_WAITS[wait][0],
            grace_window=PORTFOLIO_WAITS[wait][1],
            grace_floor=PORTFOLIO_WAITS[wait][2],
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
    # The checker-settings verdict and, when it predicts a failure, the search
    # for a subsystem it passes (#2225 held-out run): everything until 5 s before
    # the hard stop.
    deadline = t0 + hard_stop - 5.0
    from discopt._optional_highs import MissingHighsError

    log: dict = {}

    def rejected() -> None:
        # The doubly proven answer is predicted to fail: until something better is
        # found, the whole model (valid by the competition's premise) is on disk.
        write_atomic(solution, {})
        on_disk["answer"] = "the full model"

    try:
        chosen = _checked_answer(data, res, deadline, log, on_reject=rejected)
    except MissingHighsError:
        # The checker-settings re-solve imitates the competition's HiGHS-based
        # checker, so it needs HiGHS (the reader does not, since P6). Without it the
        # answer on disk -- infeasible by two proofs -- stands, and the run says so.
        print(
            "mipcc: highspy is not installed, so the checker-settings re-solve was "
            "skipped; the doubly proven subsystem stands",
            flush=True,
        )
        chosen = res
    if chosen is None:
        # Infeasible in truth (two proofs), but the checker's solve on every
        # candidate errs, is undecided or finds a point within its 1e-5: submitting
        # any of them would fail the run. The whole model (already written) stands.
        print(
            "mipcc: the checker's own settings do not find any confirmed subsystem "
            f"infeasible (tolerance {CHECK_TOL}; {json.dumps(log)}); answer is the full model",
            flush=True,
        )
        return 0
    engine_res = res  # its members/timing describe the run, whichever set is written
    if chosen is not res:
        doc = solution_json(data, chosen)
        write_atomic(solution, doc)
        on_disk["answer"] = "a subsystem the checker's settings find infeasible"
        print(
            "mipcc: the checker's own settings reject the answer; "
            f"{log.get('chosen')} written instead ({json.dumps(log)})",
            flush=True,
        )
        res = chosen
    size = _size(data, res)
    print(
        "mipcc: "
        + json.dumps(
            dict(
                size=size,
                complete=getattr(res, "complete", None),
                certificates=len(res.certificates),
                certificates_dropped=getattr(solution_json, "n_bad", 0),
                order=getattr(res, "order", None),
                engine=engine,
                forced=[getattr(res, "n_forced", 0), getattr(res, "n_forced_certified", 0)],
                add_one=[getattr(res, "n_add_one_tried", 0), getattr(res, "n_add_one_moves", 0)],
                relaxed=getattr(res, "n_relaxed", 0),
                closed=[getattr(res, "n_exchanged", 0), getattr(res, "n_close_dropped", 0)],
                proven=getattr(res, "proven_irreducible", None),
                rehold=getattr(res, "n_rehold", 0),
                checker=log,
                # Kept elements written with no certificate (a complete answer
                # with some is not shown irreducible to the checker).
                n_uncertified=size - getattr(solution_json, "n_good", 0),
                # Where the wall time went: the portfolio's members (order, outcome,
                # size, complete, seconds from the engine's start to their end) and
                # the engine's phase times (MilpIIS.timing).
                members=[
                    [
                        m.get("order"),
                        m.get("outcome"),
                        m.get("size"),
                        m.get("complete"),
                        m.get("t_done"),
                    ]
                    for m in (getattr(engine_res, "members", None) or [])
                ],
                timing=getattr(engine_res, "timing", None),
                wall=round(time.perf_counter() - t0, 2),
            )
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(t_launch=_T_LAUNCH))
