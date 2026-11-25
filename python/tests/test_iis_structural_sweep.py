"""The IIS engine's structural sweep (``_iis_milp._structural_sweep``).

Between filter phases and on the final answer, the engine drops elements of the
current infeasible subsystem that need no solve to judge: rows that hold over
the column boxes, rows that a monotone (pure-literal) column can always satisfy,
and the integrality and bounds of a column no kept row touches. Measured on
MIPcc27's dijle, whose 3961-element answer kept 1176 orphan elements.

These tests pin (1) soundness: the swept system is still infeasible, checked
independently with HiGHS on random infeasible subsystems; (2) each rule's
refusals; (3) the engine wiring, including the ``DISCOPT_IIS_SWEEP=0`` opt-out.
"""

import random

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp

highspy = pytest.importorskip("highspy")
from discopt._iis_milp import _structural_sweep, milp_iis  # noqa: E402

INF = 1e20


@pytest.fixture(autouse=True)
def _pin_highs_engine(monkeypatch):
    monkeypatch.setenv("DISCOPT_IIS_ENGINE", "highs")


def _verdict(A, lo, hi, groups, lb, ub, cons, bounds, ints):
    """The subsystem's status: kept rows, held integralities (with their declared
    box), and the kept bounds of every other column. ``"feasible"`` only with a
    HiGHS point that satisfies the system on independent re-evaluation;
    ``"infeasible"`` only when HiGHS says so with presolve on AND off; else
    ``None``. HiGHS 1.x was measured returning ``kInfeasible`` with presolve off
    on a feasible 5-column MILP from this generator (presolve on: optimal), so a
    single verdict is not an oracle."""
    n = lb.size
    inf = highspy.kHighsInf
    lo_c = np.full(n, -inf)
    hi_c = np.full(n, inf)
    for j in ints:
        lo_c[j] = lb[j] if lb[j] > -INF else -inf
        hi_c[j] = ub[j] if ub[j] < INF else inf
    for j, side in bounds:
        if j in ints:
            continue
        if side == "lower":
            lo_c[j] = lb[j]
        else:
            hi_c[j] = ub[j]
    rows = [r for ci in cons for r in groups[ci]]
    statuses = []
    for presolve in ("on", "off"):
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("presolve", presolve)
        h.setOptionValue("time_limit", 2.0)
        h.addVars(n, lo_c, hi_c)
        for r in rows:
            s, e = A.indptr[r], A.indptr[r + 1]
            h.addRow(
                -inf if lo[r] <= -INF else lo[r],
                inf if hi[r] >= INF else hi[r],
                e - s,
                A.indices[s:e].astype(np.int32),
                A.data[s:e],
            )
        if ints:
            idx = np.array(sorted(ints), dtype=np.int32)
            h.changeColsIntegrality(
                len(idx), idx, np.array([highspy.HighsVarType.kInteger] * len(idx))
            )
        h.run()
        st = h.getModelStatus()
        statuses.append(st)
        if st == highspy.HighsModelStatus.kOptimal:
            x = np.array(h.getSolution().col_value)
            held = np.array(sorted(ints), dtype=int)
            x[held] = np.round(x[held])
            tol = 1e-7
            ok = bool(np.all(x >= lo_c - tol) and np.all(x <= hi_c + tol))
            for r in rows:
                s, e = A.indptr[r], A.indptr[r + 1]
                act = float(A.data[s:e] @ x[A.indices[s:e]])
                ok &= (lo[r] <= -INF or act >= lo[r] - tol) and (hi[r] >= INF or act <= hi[r] + tol)
            if ok:
                return "feasible"
    if all(st == highspy.HighsModelStatus.kInfeasible for st in statuses):
        return "infeasible"
    return None


def _random_instance(rng):
    n, m = rng.randint(2, 6), rng.randint(2, 7)
    rows, cols, vals = [], [], []
    for r in range(m):
        for j in rng.sample(range(n), rng.randint(1, min(n, 3))):
            rows.append(r)
            cols.append(j)
            vals.append(rng.choice([-3, -2, -1, 1, 2, 3, 0.5, -1.5]))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(m, n))
    lo = np.array(
        [rng.choice([-INF, rng.randint(-4, 4), rng.randint(-4, 4) + 0.5]) for _ in range(m)]
    )
    hi = np.array(
        [
            INF if rng.random() < 0.5 else max(lo[r] if lo[r] > -INF else -9, rng.randint(-4, 6))
            for r in range(m)
        ],
        dtype=float,
    )
    hi = np.where((lo <= -INF) & (hi >= INF), float(rng.randint(-2, 3)), hi)
    lb = np.array([rng.choice([-INF, 0.0, -2.0, 1.0]) for _ in range(n)])
    ub = np.array([rng.choice([INF, 1.0, 3.0, 2.5]) for _ in range(n)])
    ub = np.maximum(ub, np.where(lb > -INF, lb, -INF))
    ints = {j for j in range(n) if rng.random() < 0.5}
    return A, lo, hi, lb, ub, ints


@pytest.mark.smoke
def test_swept_subsystem_stays_infeasible():
    """Random infeasible subsystems (each a random infeasible reduction of an
    infeasible model): after the sweep, HiGHS never finds a verified feasible
    point. A feasible verdict would, by the sweep's own argument, extend to a
    point of the subsystem before the sweep, which both HiGHS settings call
    infeasible."""
    rng = random.Random(7)
    checked = fired = removed = undecided = 0
    for _ in range(1000):
        A, lo, hi, lb, ub, ints = _random_instance(rng)
        m, n = A.shape
        groups = {r: np.array([r]) for r in range(m)}
        cons = set(range(m))
        bounds = {(j, "lower") for j in range(n) if lb[j] > -INF}
        bounds |= {(j, "upper") for j in range(n) if ub[j] < INF}
        if _verdict(A, lo, hi, groups, lb, ub, cons, bounds, ints) != "infeasible":
            continue
        elements = [("c", c) for c in cons] + [("b", b) for b in bounds] + [("i", i) for i in ints]
        for kind, e in rng.sample(elements, k=min(len(elements), rng.randint(0, 4))):
            c2, b2, i2 = set(cons), set(bounds), set(ints)
            {"c": c2, "b": b2, "i": i2}[kind].discard(e)
            if _verdict(A, lo, hi, groups, lb, ub, c2, b2, i2) == "infeasible":
                cons, bounds, ints = c2, b2, i2
        nc, nb, ni = _structural_sweep(A, A.tocsc(), lo, hi, groups, lb, ub, cons, bounds, ints)
        checked += 1
        if nc + nb + ni:
            fired += 1
            removed += nc + nb + ni
            after = _verdict(A, lo, hi, groups, lb, ub, cons, bounds, ints)
            assert after != "feasible", (A.toarray(), lo, hi, lb, ub, cons, bounds, ints)
            undecided += after is None
    print(f"[sweep] checked={checked} fired={fired} removed={removed} undecided={undecided}")
    assert checked >= 300 and fired >= 200 and undecided <= fired // 10, (
        checked,
        fired,
        undecided,
    )


def _one(A_rows, lo, hi, lb, ub, ints, bounds=None):
    m = len(A_rows)
    n = len(lb)
    A = sp.csr_matrix(np.array(A_rows, dtype=float).reshape(m, n))
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    if bounds is None:
        bounds = {(j, "lower") for j in range(n) if lb[j] > -INF}
        bounds |= {(j, "upper") for j in range(n) if ub[j] < INF}
    cons = set(range(m))
    ints = set(ints)
    groups = {r: np.array([r]) for r in range(m)}
    out = _structural_sweep(
        A, A.tocsc(), np.array(lo, float), np.array(hi, float), groups, lb, ub, cons, bounds, ints
    )
    return out, cons, bounds, ints


@pytest.mark.smoke
def test_monotone_column_drops_its_rows():
    # x0 >= 5, x0 <= 2 (the conflict); x1 - x2 <= 0 with x1 >= 0 only: x1 -> 0 frees it.
    (nc, _, _), cons, _, _ = _one(
        [[1, 0, 0], [1, 0, 0], [0, 1, -1]],
        [5, -INF, -INF],
        [INF, 2, 0],
        [-INF, 0, 0],
        [INF, INF, INF],
        [],
    )
    assert nc == 1 and cons == {0, 1}


@pytest.mark.smoke
def test_two_sided_row_is_not_monotone():
    # 0 <= x1 - x2 <= 0 is two-sided in x1 and x2: nothing to drop.
    (nc, nb, ni), cons, _, _ = _one(
        [[1, 0, 0], [1, 0, 0], [0, 1, -1]],
        [5, -INF, 0],
        [INF, 2, 0],
        [-INF, -INF, -INF],
        [INF] * 3,
        [],
    )
    assert (nc, nb, ni) == (0, 0, 0) and cons == {0, 1, 2}


@pytest.mark.smoke
def test_monotone_needs_an_integral_value_for_a_held_column():
    # x1 >= 0.5 held integer: x1 - x2 <= 0 must hold at x1 = 0.5 (not integral):
    # the rule refuses; at x1 = 1 the row would not hold for x2 in [0, 0.9].
    (nc, _, _), cons, _, _ = _one(
        [[1, 0, 0], [1, 0, 0], [0, 1, -1]],
        [5, -INF, -INF],
        [INF, 2, 0],
        [-INF, 0.5, 0],
        [INF, 3, 0.9],
        [1],
    )
    assert nc == 0 and 2 in cons


@pytest.mark.smoke
def test_orphan_column_loses_integrality_and_bounds():
    # x0 conflicts alone; x1 is in no row: its integrality and bounds go.
    (nc, nb, ni), cons, bounds, ints = _one(
        [[1, 0], [1, 0]], [5, -INF], [INF, 2], [-INF, 0], [INF, 4], [1]
    )
    assert (nc, ni) == (0, 1) and ints == set() and all(j != 1 for j, _ in bounds)


@pytest.mark.smoke
def test_orphan_held_column_without_an_integer_is_kept():
    # x1 in [0.2, 0.8] held integer: an infeasibility source of its own.
    (_, _, ni), _, _, ints = _one(
        [[1, 0], [1, 0]], [5, -INF], [INF, 2], [-INF, 0.2], [INF, 0.8], [1]
    )
    assert ni == 0 and ints == {1}


@pytest.mark.smoke
def test_orphan_column_with_an_empty_box_is_kept():
    (_, nb, _), _, bounds, _ = _one([[1, 0], [1, 0]], [5, -INF], [INF, 2], [-INF, 3], [INF, 1], [])
    assert nb == 0 and {(1, "lower"), (1, "upper")} <= bounds


def _redundant_rows_model():
    m = dm.Model("many")
    x = m.continuous("x", shape=(40,), lb=0, ub=10)
    m.subject_to(x[0] >= 5, name="big")
    for k in range(1, 39):
        m.subject_to(x[k] + x[k + 1] <= 15, name=f"rh{k}")
    m.subject_to(x[0] <= 2, name="small")
    m.minimize(0 * x[0])
    return m


@pytest.mark.smoke
def test_engine_sweeps_and_the_opt_out_restores_the_filter(monkeypatch):
    on = milp_iis(_redundant_rows_model())
    monkeypatch.setenv("DISCOPT_IIS_SWEEP", "0")
    off = milp_iis(_redundant_rows_model())
    assert on.constraints == off.constraints == [0, 39]
    assert on.bounds == off.bounds and on.integrality == off.integrality
    assert on.proven_irreducible and off.proven_irreducible
    assert off.n_swept == 0 and on.n_swept > 0
    assert on.n_solves < off.n_solves, (on.n_solves, off.n_solves)
    assert len(on.certificates) == len(on.constraints) + len(on.bounds) + len(on.integrality)
