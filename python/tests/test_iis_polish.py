"""The MILP IIS answer's post-pass (``_iis_milp._polish``): the forced-violation
certificate repair and the add-one constraint move.

Certificates are checked here with ``discopt.mipcc.certificate_ok`` (the
competition checker's rule with its safety margin), not with the engine's own
``_check_certificates``, so the check is independent of the code under test.
"""

import time

import numpy as np
import pytest
import scipy.sparse as sp
from discopt import _iis_milp as E
from discopt._iis_milp import MilpIIS, _polish, milp_iis_arrays
from discopt.mipcc import certificate_ok

pytestmark = pytest.mark.requires_highs

INF = 1e20


def _arrays(rows, lo, hi, n, lb, ub, is_int):
    A = sp.csr_matrix(np.array(rows, dtype=float).reshape(len(rows), n))
    m = A.shape[0]
    return (
        A,
        np.array(lo, dtype=float),
        np.array(hi, dtype=float),
        {i: np.array([i], dtype=np.int64) for i in range(m)},
        np.array(lb, dtype=float),
        np.array(ub, dtype=float),
        np.array(is_int, dtype=bool),
    )


def _checked(arrays, res: MilpIIS) -> int:
    """Assert every certificate passes the checker's rule; return how many."""
    A, lo, hi, _, lb, ub, is_int = arrays
    data = dict(A=A, row_lo=lo, row_hi=hi, col_lb=lb, col_ub=ub, is_int=is_int)
    n, m = lb.size, lo.size
    rows = np.zeros(m, dtype=bool)
    rows[list(res.constraints)] = True
    held = np.zeros(n, dtype=bool)
    held[list(res.integrality)] = True
    lbk = np.zeros(n, dtype=bool)
    ubk = np.zeros(n, dtype=bool)
    for j, side in res.bounds:
        (lbk if side == "lower" else ubk)[j] = True
    for key, x in res.certificates.items():
        assert certificate_ok(data, rows, lbk, ubk, held, key, np.asarray(x)), key
    return len(res.certificates)


def _n_elements(res: MilpIIS) -> int:
    return len(res.constraints) + len(res.bounds) + len(res.integrality)


# x + y <= 5, x >= 3, y >= 3 (an IIS of 3 constraints); x + y >= 6 with x + y <= 5
# is one of 2, reachable from the first by adding one constraint.
_ROWS = [[1, 1], [1, 0], [0, 1], [1, 1]]
_LO = [-INF, 3, 3, 6]
_HI = [5, INF, INF, INF]


def _three_row_answer():
    """The 3-constraint IIS, found on the model without the 4th constraint (so
    its ids are the full model's ids)."""
    arr = _arrays(_ROWS[:3], _LO[:3], _HI[:3], 2, [0, 0], [10, 10], [False, False])
    res = milp_iis_arrays(arr, total_time_limit=30.0)
    assert sorted(res.constraints) == [0, 1, 2] and res.complete
    return res


def _full():
    return _arrays(_ROWS, _LO, _HI, 2, [0, 0], [10, 10], [False, False])


def test_add_one_finds_the_smaller_iis_and_certifies_it(monkeypatch):
    monkeypatch.delenv("DISCOPT_IIS_ADD_ONE", raising=False)
    res = _three_row_answer()
    arr = _full()
    out = _polish(
        arr,
        res,
        cert_tol=1e-6,
        time_limit=10.0,
        engine="highs",
        deadline=time.perf_counter() + 60.0,
    )
    assert sorted(out.constraints) == [0, 3]
    assert out.bounds == [] and out.integrality == []
    assert out.n_add_one_moves == 1 and out.n_add_one_tried >= 1
    assert _checked(arr, out) == _n_elements(out) == 2


def test_add_one_move_is_reverted_when_confirmation_fails(monkeypatch):
    """A changed set is adopted only after confirm_infeasible says True."""
    res = _three_row_answer()
    arr = _full()
    calls = []

    def refuse(self, *a, **k):
        calls.append(a)
        return None

    monkeypatch.setattr(E._Engine, "confirm_infeasible", refuse)
    out = _polish(
        arr,
        res,
        cert_tol=1e-6,
        time_limit=10.0,
        engine="highs",
        deadline=time.perf_counter() + 60.0,
    )
    assert len(calls) == 1
    assert sorted(out.constraints) == [0, 1, 2] and out.n_add_one_moves == 0
    assert _checked(arr, out) == 3


def test_add_one_opt_out_and_no_budget_keep_the_answer(monkeypatch):
    res = _three_row_answer()
    arr = _full()
    monkeypatch.setenv("DISCOPT_IIS_ADD_ONE", "0")
    out = _polish(
        arr, res, cert_tol=1e-6, time_limit=10.0, engine="highs", deadline=time.perf_counter() + 60
    )
    assert sorted(out.constraints) == [0, 1, 2] and out.n_add_one_tried == 0
    monkeypatch.delenv("DISCOPT_IIS_ADD_ONE")
    out = _polish(arr, res, cert_tol=1e-6, time_limit=10.0, engine="highs", deadline=None)
    assert sorted(out.constraints) == [0, 1, 2] and out.n_add_one_tried == 0


def test_forced_violation_restores_missing_certificates(monkeypatch):
    """Every certificate stripped from an answer is found again by forcing its
    element violated, for each element kind (constraint, integrality, bound)."""
    monkeypatch.setenv("DISCOPT_IIS_ADD_ONE", "0")
    # z - w >= 0.6 and z <= 0.9 (rows), z integer in [0, 3], w in [0.3, 4]: z = 0
    # forces w <= -0.6 against w >= 0.3. The IIS is both rows, z's integrality
    # and w's lower bound.
    arr = _arrays([[1, -1], [1, 0]], [0.6, -INF], [INF, 0.9], 2, [0, 0.3], [3, 4], [True, False])
    res = milp_iis_arrays(arr, total_time_limit=30.0)
    kinds = {k[0] for k in E._element_keys(res.constraints, res.bounds, res.integrality)}
    assert kinds == {"constraint", "integrality", "bound"}, res
    full = _n_elements(res)
    stripped = MilpIIS(
        constraints=list(res.constraints),
        bounds=list(res.bounds),
        integrality=list(res.integrality),
        certificates={},
    )
    out = _polish(
        arr,
        stripped,
        cert_tol=1e-6,
        time_limit=10.0,
        engine="highs",
        deadline=time.perf_counter() + 60,
    )
    assert out.n_forced == full and out.n_forced_certified == full
    assert _checked(arr, out) == full


def test_weak_certificate_is_replaced_by_a_strong_one(monkeypatch):
    """A point the engine's 1e-6 check accepts but a 1e-5 checker would not (its
    element violated by only 1e-5) counts as missing and is replaced."""
    monkeypatch.setenv("DISCOPT_IIS_ADD_ONE", "0")
    # x >= 3 and x <= 3 - 1e-5: x = 3 - 1e-5 certifies the first only weakly.
    arr = _arrays([[1], [1]], [3, -INF], [INF, 3 - 1e-5], 1, [0], [10], [False])
    weak = np.array([3 - 1e-5])
    key = ("constraint", 0)
    assert E._check_certificates({key: weak}, _engine(arr), {0, 1}, set(), set(), 1e-6), (
        "the test point must pass the engine's own check"
    )
    res = MilpIIS(constraints=[0, 1], bounds=[], integrality=[], certificates={key: weak})
    out = _polish(
        arr, res, cert_tol=1e-6, time_limit=10.0, engine="highs", deadline=time.perf_counter() + 60
    )
    assert out.n_forced == 2 and out.n_forced_certified == 2
    assert out.certificates[key][0] < 3 - 2e-5
    assert _checked(arr, out) == 2


def _engine(arr):
    A, lo, hi, groups, lb, ub, is_int = arr
    return E._Engine(A, lo, hi, lb, ub, is_int, groups, 10.0, False, engine="highs")


def test_add_one_sweep_cut_by_the_clock_is_discarded_and_the_probe_cap_is_deterministic(
    monkeypatch,
):
    """The probe count bounds the sweep (no probe: no move); the time share only
    stops it, and a sweep it stops raises, so _polish keeps the answer whole."""
    res = _three_row_answer()
    arr = _full()
    eng = _engine(arr)
    got, _, _, tried = E._add_one(eng, {0, 1, 2}, set(), set(), time.perf_counter() + 60, 0)
    assert got is None and tried == 0
    with pytest.raises(E._OutOfTime):
        E._add_one(eng, {0, 1, 2}, set(), set(), time.perf_counter() - 1.0)
    monkeypatch.setattr(E, "_ADD_ONE_FRACTION", 0.0)
    out = _polish(
        arr, res, cert_tol=1e-6, time_limit=10.0, engine="highs", deadline=time.perf_counter() + 60
    )
    assert sorted(out.constraints) == [0, 1, 2] and out.n_add_one_moves == 0
    assert _checked(arr, out) == 3
