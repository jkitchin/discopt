"""The MILP IIS answer's post-pass (``_iis_milp._polish``): the forced-violation
certificate repair and the add-one constraint move.

Certificates are checked here with ``discopt.mipcc.certificate_ok`` (the
competition checker's rule with its safety margin), not with the engine's own
``_check_certificates``, so the check is independent of the code under test.
"""

import time
from dataclasses import replace

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


def _anytime(cons, bounds, ints):
    """An answer as the filter returns it when the clock stopped it."""
    return MilpIIS(constraints=cons, bounds=bounds, integrality=ints, complete=False)


def _post(arr, res):
    return _polish(
        arr, res, cert_tol=1e-6, time_limit=10.0, engine="highs", deadline=time.perf_counter() + 60
    )


def test_relax_drops_a_general_integer_from_an_anytime_answer(monkeypatch):
    """x + y >= 3, x + y <= 2 is infeasible without x's integrality: the block
    relaxation drops it (with its box, so one element less)."""
    monkeypatch.delenv("DISCOPT_IIS_RELAX_INT", raising=False)
    arr = _arrays([[1, 1], [1, 1]], [3, -INF], [INF, 2], 2, [0, 0], [10, 10], [True, False])
    out = _post(arr, _anytime([0, 1], [], [0]))
    assert out.integrality == [] and out.bounds == [] and sorted(out.constraints) == [0, 1]
    assert out.n_relaxed == 1 and not out.complete
    assert _checked(arr, out) == 2


def test_relax_trades_an_uncertifiable_binary_integrality_for_one_bound(monkeypatch):
    """x binary, x >= 1.5: x's integrality has no certificate under the checker's
    rule (relaxed, x keeps its box [0, 1]). The swap keeps only x <= 1, which
    has one, so every element of the same-size answer is certified."""
    monkeypatch.delenv("DISCOPT_IIS_RELAX_INT", raising=False)
    arr = _arrays([[1]], [1.5], [INF], 1, [0], [1], [True])
    monkeypatch.setenv("DISCOPT_IIS_RELAX_INT", "0")
    before = _post(arr, _anytime([0], [], [0]))
    assert before.integrality == [0] and _checked(arr, before) == 1
    monkeypatch.setenv("DISCOPT_IIS_RELAX_INT", "1")
    out = _post(arr, _anytime([0], [], [0]))
    assert out.integrality == [] and out.bounds == [(0, "upper")] and out.constraints == [0]
    assert _n_elements(out) == _n_elements(before) and out.n_relaxed == 1
    assert _checked(arr, out) == 2


def test_relax_restores_a_needed_integrality_and_leaves_complete_answers_alone(monkeypatch):
    """2x = 1 with x binary: the relaxed point x = 0.5 names x, which is held
    again, so the answer is unchanged. A complete answer is never relaxed by
    this pass. The answer handed in as complete here holds a redundant
    integrality (its two rows are LP-infeasible), which is the closing pass's
    business: opted out the answer stands, by default it is exchanged, dropped
    and the two rows are returned with a certificate each."""
    monkeypatch.delenv("DISCOPT_IIS_RELAX_INT", raising=False)
    arr = _arrays([[2]], [1], [1], 1, [0], [1], [True])
    out = _post(arr, _anytime([0], [], [0]))
    assert out.integrality == [0] and out.bounds == [] and out.n_relaxed == 0
    arr2 = _arrays([[1, 1], [1, 1]], [3, -INF], [INF, 2], 2, [0, 0], [10, 10], [True, False])

    def done():
        return MilpIIS(constraints=[0, 1], bounds=[], integrality=[0], complete=True)

    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "0")
    kept = _post(arr2, done())
    assert kept.integrality == [0] and kept.n_relaxed == 0
    monkeypatch.delenv("DISCOPT_IIS_CLOSE")
    closed = _post(arr2, done())
    assert closed.n_relaxed == 0 and closed.n_exchanged == 1
    assert closed.integrality == [] and closed.bounds == []
    assert sorted(closed.constraints) == [0, 1]
    assert _checked(arr2, closed) == _n_elements(closed) == 2


def test_relaxed_set_is_kept_only_when_confirmed(monkeypatch):
    monkeypatch.delenv("DISCOPT_IIS_RELAX_INT", raising=False)
    monkeypatch.setattr(E._Engine, "confirm_infeasible", lambda self, *a, **k: None)
    arr = _arrays([[1, 1], [1, 1]], [3, -INF], [INF, 2], 2, [0, 0], [10, 10], [True, False])
    out = _post(arr, _anytime([0, 1], [], [0]))
    assert out.integrality == [0] and out.n_relaxed == 0


def test_relaxation_cut_by_the_clock_is_discarded():
    arr = _arrays([[1, 1], [1, 1]], [3, -INF], [INF, 2], 2, [0, 0], [10, 10], [True, False])
    eng = _engine(arr)
    with pytest.raises(E._OutOfTime):
        E._relax_integrality(eng, {0, 1}, set(), {0}, set(), time.perf_counter() - 1.0)


def test_filter_leaves_the_post_pass_its_reserve(monkeypatch):
    """With a budget, the filter gets (1 - _POLISH_RESERVE_FRACTION) of it and the
    post-pass the whole; the opt-out gives the filter the whole budget again."""
    seen = []
    real = E._milp_iis_arrays

    def spy(*a, **k):
        seen.append(k["total_time_limit"])
        return real(*a, **k)

    monkeypatch.setattr(E, "_milp_iis_arrays", spy)
    arr = _full()
    monkeypatch.delenv("DISCOPT_IIS_RELAX_INT", raising=False)
    milp_iis_arrays(arr, total_time_limit=40.0)
    monkeypatch.setenv("DISCOPT_IIS_RELAX_INT", "0")
    milp_iis_arrays(arr, total_time_limit=40.0)
    assert seen == [pytest.approx(40.0 * (1 - E._POLISH_RESERVE_FRACTION)), 40.0]


def test_post_pass_shares_saturate_with_the_budget(monkeypatch):
    """#1153 role 2: the filter's reserve and the relaxation / add-one stops are
    capped by constants, so a larger budget goes to the filter, not to a longer
    count-bounded post-pass. Each cap equals its fraction of a 600 s budget, so it
    cannot bind at <= 600 s."""
    assert E._POLISH_RESERVE_CAP_S == pytest.approx(E._POLISH_RESERVE_FRACTION * 600.0)
    assert E._RELAX_CAP_S == pytest.approx(E._RELAX_FRACTION * 600.0)
    assert E._ADD_ONE_CAP_S == pytest.approx(E._ADD_ONE_FRACTION * 600.0)
    seen = []
    real = E._milp_iis_arrays

    def spy(*a, **k):
        seen.append(k["total_time_limit"])
        return real(*a, **k)

    monkeypatch.setattr(E, "_milp_iis_arrays", spy)
    monkeypatch.delenv("DISCOPT_IIS_RELAX_INT", raising=False)
    milp_iis_arrays(_full(), total_time_limit=6000.0)
    assert seen == [pytest.approx(6000.0 - E._POLISH_RESERVE_CAP_S)]

    stops = {}
    real_relax, real_add = E._relax_integrality, E._add_one

    def relax_spy(eng, cons, bounds, ints, keep, stop_at, *a, **k):
        stops["relax"] = stop_at - time.perf_counter()
        return real_relax(eng, cons, bounds, ints, keep, stop_at, *a, **k)

    def add_spy(eng, cons, bounds, ints, stop_at, *a, **k):
        stops["add_one"] = stop_at - time.perf_counter()
        return real_add(eng, cons, bounds, ints, stop_at, *a, **k)

    monkeypatch.setattr(E, "_relax_integrality", relax_spy)
    monkeypatch.setattr(E, "_add_one", add_spy)
    monkeypatch.delenv("DISCOPT_IIS_ADD_ONE", raising=False)
    far = time.perf_counter() + 6000.0
    arr = _arrays([[1, 1], [1, 1]], [3, -INF], [INF, 2], 2, [0, 0], [10, 10], [True, False])
    kw = dict(cert_tol=1e-6, time_limit=6000.0, engine="highs", deadline=far)
    _polish(arr, _anytime([0, 1], [], [0]), **kw)
    _polish(_full(), _three_row_answer(), **kw)
    assert set(stops) == {"relax", "add_one"}
    assert 0.0 < stops["relax"] <= E._RELAX_CAP_S
    assert 0.0 < stops["add_one"] <= E._ADD_ONE_CAP_S


def test_add_one_final_refilter_frees_an_integrality_its_box_alone_keeps_infeasible():
    """x integer in [0, 1], y free. {R0: x + y >= 0.5, R1: y <= 0.2,
    R2: x <= 0.8} with x held is an IIS of 4. Adding R3: x >= 1.5 drops all
    three old rows ({R3} with x held, 2 elements). The final re-filter: freeing
    x with its box is feasible at x = 1.5, outside [0, 1]; with the box kept it
    is infeasible, so the integrality goes and only x <= 1 stays. Before the
    fix the integrality was kept with no certificate: not irreducible."""
    arr = _arrays(
        [[1, 1], [0, 1], [1, 0], [1, 0]],
        [0.5, -INF, -INF, 1.5],
        [INF, 0.2, 0.8, INF],
        2,
        [0, -INF],
        [1, INF],
        [True, False],
    )
    eng = _engine(arr)
    got, pend, decided, _ = E._add_one(eng, {0, 1, 2}, set(), {0}, time.perf_counter() + 60)
    assert got == ({3}, {(0, "upper")}, set()) and decided
    assert eng.confirm_infeasible(*got, stop_at=time.perf_counter() + 60) is True
    res = MilpIIS(
        constraints=sorted(got[0]),
        bounds=sorted(got[1]),
        integrality=sorted(got[2]),
        certificates=E._check_certificates(pend, eng, *got, 1e-6),
        complete=True,
    )
    assert _checked(arr, res) == _n_elements(res) == 2


def test_post_pass_integrality_step_tries_the_integrality_alone():
    """The step shared by re-hold and add-one's final re-filter: on {R0} with x
    held, pass 1 (x free of its box) is feasible at x = 1.5, outside [0, 1];
    pass 2 (box kept) is infeasible, so the integrality goes and its box joins
    the bounds. On a second model (x binary, 0.2 <= x <= 0.8) the pass-1 point
    lies in the box: it is the certificate, and the integrality stays."""
    arr = _arrays(
        [[1, 0], [1, 1], [0, 1]],
        [1.5, -INF, 0],
        [INF, 1.2, INF],
        2,
        [0, -INF],
        [1, INF],
        [True, False],
    )
    eng = _engine(arr)
    b, i, pend = set(), {0}, {}
    assert E._filter_integrality(eng.probe, eng, 0, {0}, b, i, pend)
    assert i == set() and b == {(0, "lower"), (0, "upper")} and pend == {}
    # x in {0, 1} with 0.2 <= x <= 0.8 (rows R0, R1 of a second model): the
    # joint point is in the box, a certificate as it is; the integrality stays.
    arr2 = _arrays([[1], [1]], [0.2, -INF], [INF, 0.8], 1, [0], [1], [True])
    eng2 = _engine(arr2)
    b, i, pend = set(), {0}, {}
    assert E._filter_integrality(eng2.probe, eng2, 0, {0, 1}, b, i, pend)
    assert i == {0} and b == set() and set(pend) == {("integrality", 0)}
    assert 0.2 - 1e-9 <= pend[("integrality", 0)][0] <= 0.8 + 1e-9


# --- the closing pass (DISCOPT_IIS_CLOSE) ---------------------------------------
#
# 2x - 2y = 1 with x, y integer and both fixed at 0. {row, int x, int y} is
# infeasible and irreducible by deletion, but the checker relaxes an integrality
# with the column's box kept, and the box alone ([0, 0] twice) leaves the row
# infeasible: neither integrality can have a certificate. The certifiable IIS
# is {row, x <= 0, y >= 0}.


def _parity():
    return _arrays([[2, -2]], [1], [1], 2, [0, 0], [0, 0], [True, True])


def _parity_answer() -> MilpIIS:
    return MilpIIS(constraints=[0], bounds=[], integrality=[0, 1])


def _polish_parity(arr):
    return _polish(
        arr,
        _parity_answer(),
        cert_tol=1e-6,
        time_limit=10.0,
        engine="highs",
        deadline=time.perf_counter() + 60.0,
    )


def test_closing_pass_is_the_default_and_zero_opts_out(monkeypatch):
    monkeypatch.delenv("DISCOPT_IIS_CLOSE", raising=False)
    assert E._close_enabled()
    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "0")
    assert not E._close_enabled()


def test_without_the_closing_pass_the_held_integralities_stay_uncertified(monkeypatch):
    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "0")
    arr = _parity()
    out = _polish_parity(arr)
    assert sorted(out.integrality) == [0, 1] and out.n_exchanged == 0
    assert not any(k[0] == "integrality" for k in out.certificates)
    assert _checked(arr, out) < _n_elements(out)


def test_closing_pass_exchanges_integrality_for_bounds_and_drops_the_slack_ones(monkeypatch):
    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "1")
    arr = _parity()
    out = _polish_parity(arr)
    assert out.integrality == [] and sorted(out.constraints) == [0]
    assert sorted(out.bounds) == [(0, "upper"), (1, "lower")]
    assert out.n_exchanged == 2 and out.n_close_dropped == 2
    assert _checked(arr, out) == _n_elements(out) == 3
    assert out.proven_irreducible


def test_closing_pass_keeps_the_answer_when_confirmation_fails(monkeypatch):
    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "1")
    calls = []

    def refuse(self, *a, **k):
        calls.append(a)
        return None

    monkeypatch.setattr(E._Engine, "confirm_infeasible", refuse)
    out = _polish_parity(_parity())
    assert calls, "the exchanged set was never offered for confirmation"
    assert sorted(out.integrality) == [0, 1] and out.bounds == []
    assert out.n_exchanged == 0 and out.n_close_dropped == 0


def test_closing_pass_leaves_a_certified_answer_alone(monkeypatch):
    """x integer in [0, 1] with 2x = 1: the integrality has a witness (x = 1/2
    inside the box), so nothing is exchanged."""
    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "1")
    arr = _arrays([[2]], [1], [1], 1, [0], [1], [True])
    res = milp_iis_arrays(arr, total_time_limit=30.0)
    assert res.integrality == [0] and res.n_exchanged == 0 and res.n_close_dropped == 0
    assert _checked(arr, res) == _n_elements(res)


# Re-hold move (_rehold). x, y binary, z free continuous:
#   R0: x + y = 1, R1: x - y = 0  -- infeasible only with an integrality (2y = 1);
#   R2: x - z >= 0, R3: z >= 0.6  -- with R0, R1 infeasible even continuously.
# An integrality-first filter without the SAT core (ICB-n) frees both
# integralities, since the rows stay infeasible, and then needs all four rows
# (size 4). Holding the integralities again lets R2 and R3 go: {R0, R1, one int}.
_RH_ROWS = [[1, 1, 0], [1, -1, 0], [1, 0, -1], [0, 0, 1]]
_RH_LO = [1, 0, 0, 0.6]
_RH_HI = [1, 0, INF, INF]


def _rehold_model():
    return _arrays(_RH_ROWS, _RH_LO, _RH_HI, 3, [0, 0, -INF], [1, 1, INF], [True, True, False])


def test_rehold_trades_rows_for_an_integrality(monkeypatch):
    """MIPcc27 stein45inf: the integrality-first answer kept 209 rows + 16
    integralities (225); re-holding gave 190 + 22 (212, Gurobi 213)."""
    arr = _rehold_model()
    monkeypatch.setenv("DISCOPT_IIS_REHOLD", "0")
    off = milp_iis_arrays(arr, order="ICB-n", total_time_limit=60.0)
    assert off.complete and sorted(off.constraints) == [0, 1, 2, 3] and off.integrality == []
    monkeypatch.delenv("DISCOPT_IIS_REHOLD")
    on = milp_iis_arrays(arr, order="ICB-n", total_time_limit=60.0)
    assert on.complete and sorted(on.constraints) == [0, 1]
    assert len(on.integrality) == 1 and on.bounds == []
    assert on.n_rehold == 1
    assert _checked(arr, on) == _n_elements(on) == 3


def test_rehold_is_kept_only_when_confirmed(monkeypatch):
    arr = _rehold_model()
    monkeypatch.setenv("DISCOPT_IIS_REHOLD", "0")
    res = milp_iis_arrays(arr, order="ICB-n", total_time_limit=60.0)
    monkeypatch.delenv("DISCOPT_IIS_REHOLD")
    monkeypatch.setenv("DISCOPT_IIS_ADD_ONE", "0")
    calls = []

    def refuse(self, *a, **k):
        calls.append(a)
        return None

    monkeypatch.setattr(E._Engine, "confirm_infeasible", refuse)
    out = _polish(
        arr, res, cert_tol=1e-6, time_limit=10.0, engine="highs", deadline=time.perf_counter() + 60
    )
    assert len(calls) == 1
    assert sorted(out.constraints) == [0, 1, 2, 3] and out.n_rehold == 0


def test_rehold_probe_cap_and_clock_discard_the_pass(monkeypatch):
    """The probe cap skips the pass before any solve (deterministic); reaching
    the stop raises _OutOfTime, which _polish turns into 'the answer stands'."""
    arr = _rehold_model()
    monkeypatch.setenv("DISCOPT_IIS_REHOLD", "0")
    res = milp_iis_arrays(arr, order="ICB-n", total_time_limit=60.0)
    A, lo, hi, groups, lb, ub, is_int = arr
    eng = E._Engine(A, lo, hi, lb, ub, is_int, groups, 10.0, False, engine="highs")
    c, b, i = set(res.constraints), set(res.bounds), set(res.integrality)
    n0 = eng.n_solves
    got, pend, decided = E._rehold(eng, c, b, i, time.perf_counter() + 60, max_probes=5)
    assert got is None and pend == {} and eng.n_solves == n0
    with pytest.raises(E._OutOfTime):
        E._rehold(eng, c, b, i, time.perf_counter() - 1.0)


def test_rehold_share_saturates(monkeypatch):
    """#1153 role 2: the re-hold stop is capped by a constant."""
    stops = {}
    real = E._rehold

    def spy(eng, cons, bounds, ints, stop_at, *a, **k):
        stops["rehold"] = stop_at - time.perf_counter()
        return real(eng, cons, bounds, ints, stop_at, *a, **k)

    arr = _rehold_model()
    monkeypatch.setenv("DISCOPT_IIS_REHOLD", "0")
    res = milp_iis_arrays(arr, order="ICB-n", total_time_limit=60.0)
    monkeypatch.delenv("DISCOPT_IIS_REHOLD")
    monkeypatch.setattr(E, "_rehold", spy)
    kw = dict(cert_tol=1e-6, time_limit=6000.0, engine="highs")
    out = _polish(arr, res, deadline=time.perf_counter() + 6000.0, **kw)
    assert 0.0 < stops["rehold"] <= E._REHOLD_CAP_S
    assert out.n_rehold == 1


def test_rehold_reuses_the_answers_certificates(monkeypatch):
    """A known point that certifies its element on the re-held set skips that
    element's probe; the result is unchanged."""
    arr = _rehold_model()
    monkeypatch.setenv("DISCOPT_IIS_REHOLD", "0")
    res = milp_iis_arrays(arr, order="ICB-n", total_time_limit=60.0)
    A, lo, hi, groups, lb, ub, is_int = arr
    out = {}
    for known in (None, res.certificates):
        eng = E._Engine(A, lo, hi, lb, ub, is_int, groups, 10.0, False, engine="highs")
        c, b, i = set(res.constraints), set(res.bounds), set(res.integrality)
        got, pend, decided = E._rehold(eng, c, b, i, time.perf_counter() + 60, known=known)
        out[known is None] = (got, eng.n_solves, set(pend))
    assert out[True][0] == out[False][0] == ({0, 1}, set(), {1})
    assert out[False][1] < out[True][1]
    keys = {("constraint", 0), ("constraint", 1), ("integrality", 1)}
    assert out[False][2] == out[True][2] == keys


def test_rehold_frees_an_integrality_its_box_alone_keeps_infeasible():
    """MIPcc27 stein45inf (gate, branch a97e46b): re-hold kept integrality zz02
    because relaxing it jointly with its box was feasible (zz02 = -1), while
    relaxing it alone, box kept, is infeasible: the checker reported the answer
    not minimal (215 of 216 certified). Here x is integer in [0, 1], y free;
    R0: x >= 1.5, R1: x + y <= 1.2, R2: y >= 0. Holding x makes R0 alone
    infeasible; freeing x with its box gives x = 1.5, outside it, so the
    integrality is tried alone: it goes, its box becomes elements, and the
    bound stage keeps only x <= 1."""
    arr = _arrays(
        [[1, 0], [1, 1], [0, 1]],
        [1.5, -INF, 0],
        [INF, 1.2, INF],
        2,
        [0, -INF],
        [1, INF],
        [True, False],
    )
    eng = _engine(arr)
    got, pend, decided = E._rehold(eng, {0, 1, 2}, set(), set(), time.perf_counter() + 60)
    assert got == ({0}, {(0, "upper")}, set()) and decided
    assert eng.confirm_infeasible(*got, stop_at=time.perf_counter() + 60) is True
    res = MilpIIS(
        constraints=sorted(got[0]),
        bounds=sorted(got[1]),
        integrality=sorted(got[2]),
        certificates=E._check_certificates(pend, eng, *got, 1e-6),
        complete=True,
    )
    assert _checked(arr, res) == _n_elements(res) == 2


# Two disjoint infeasible pairs: x >= 2 / x <= 1 (rows 0, 1) and y >= 2 / y <= 1
# (rows 2, 3); row 4 (x + y <= 30) is never needed.
_PAIRS = _arrays(
    [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]],
    [2, -INF, 2, -INF, -INF],
    [INF, 1, INF, 1, 30],
    2,
    [0, 0],
    [10, 10],
    [False, False],
)


def _answer(cons):
    return MilpIIS(constraints=sorted(cons), bounds=[], integrality=[], certificates={})


def test_add_one_also_runs_from_the_answer_the_rehold_move_replaced(monkeypatch):
    """Re-hold and add-one are greedy, so the path matters: from the answer
    {0,1,2,3} the re-hold move goes to {2,3,4}, where add-one finds nothing, but
    add-one from {0,1,2,3} reaches {0,1}. The post-pass keeps the smaller.
    (Measured: domset-hexgrid_03_052 re-hold then add-one 171, add-one alone 159.)
    Before the change the answer was {2,3,4}."""
    for v in ("DISCOPT_IIS_REHOLD", "DISCOPT_IIS_ADD_ONE"):
        monkeypatch.delenv(v, raising=False)
    starts = []

    def fake_rehold(eng, cons, bounds, ints, stop_at, known=None):
        return ({2, 3, 4}, set(), set()), {}, True

    def fake_add_one(eng, cons, bounds, ints, stop_at):
        starts.append(frozenset(cons))
        if set(cons) == {0, 1, 2, 3}:
            return ({0, 1}, set(), set()), {}, True, 1
        return None, {}, True, 1

    monkeypatch.setattr(E, "_rehold", fake_rehold)
    monkeypatch.setattr(E, "_add_one", fake_add_one)
    out = _polish(
        _PAIRS,
        _answer({0, 1, 2, 3}),
        cert_tol=1e-6,
        time_limit=10.0,
        engine="highs",
        deadline=time.perf_counter() + 60.0,
    )
    assert starts == [frozenset({2, 3, 4}), frozenset({0, 1, 2, 3})]
    assert out.constraints == [0, 1] and out.n_rehold == 0 and out.n_add_one_moves == 1
    assert out.n_add_one_tried == 2
    assert _checked(_PAIRS, out) == 2  # the repair certifies the winner


def test_add_one_from_both_starts_keeps_the_rehold_path_on_a_tie(monkeypatch):
    for v in ("DISCOPT_IIS_REHOLD", "DISCOPT_IIS_ADD_ONE"):
        monkeypatch.delenv(v, raising=False)
    # The fakes return a non-minimal {2, 3, 4}; the closing pass would drop row
    # 4 and hide which path was kept.
    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "0")

    def fake_rehold(eng, cons, bounds, ints, stop_at, known=None):
        return ({2, 3, 4}, set(), set()), {}, True

    def fake_add_one(eng, cons, bounds, ints, stop_at):
        if set(cons) == {0, 1, 2, 3}:  # a 3-element answer from the replaced one
            return ({0, 1, 4}, set(), set()), {}, True, 1
        return None, {}, True, 1

    monkeypatch.setattr(E, "_rehold", fake_rehold)
    monkeypatch.setattr(E, "_add_one", fake_add_one)
    out = _polish(
        _PAIRS,
        _answer({0, 1, 2, 3}),
        cert_tol=1e-6,
        time_limit=10.0,
        engine="highs",
        deadline=time.perf_counter() + 60.0,
    )
    assert out.constraints == [2, 3, 4] and out.n_rehold == 1 and out.n_add_one_moves == 0


def test_rank_answers_keeps_distinct_complete_alternatives_smallest_first():
    a = _answer({0, 1})
    b = _answer({0, 1, 4})
    b2 = _answer({0, 1, 4})
    c = _answer({1, 2, 3, 4})
    anytime = replace(_answer({2, 3}), complete=False)
    best = E._rank_answers([(0, c), (1, b), (2, anytime), (3, b2), (4, a)])
    assert best is a
    assert [r.constraints for r in best.alternatives] == [[0, 1, 4], [1, 2, 3, 4]]


def test_rank_answers_keeps_every_other_confirmed_answer_as_a_fallback():
    """#2225: the MIPcc27 entry needs somewhere to go when the checker's own
    settings reject the answer, short of the whole model. Every other distinct
    member answer, incomplete ones too, is confirmed infeasible and is kept."""
    a = _answer({0, 1})
    c = _answer({1, 2, 3, 4})
    anytime = replace(_answer({2, 3}), complete=False)
    dup = _answer({0, 1})
    best = E._rank_answers([(0, c), (1, anytime), (2, dup), (3, a)])
    assert best is dup
    assert [r.constraints for r in best.alternatives] == [[1, 2, 3, 4]]
    assert [r.constraints for r in best.fallbacks] == [[2, 3]]


def test_with_fallbacks_drops_the_answer_and_duplicates_and_ranks_the_rest():
    best = _answer({0, 1})
    pool = [None, _answer({1, 2, 3, 4}), _answer({0, 1}), _answer({2, 3}), _answer({2, 3})]
    pool[1].alternatives = [_answer({5})]
    out = E._with_fallbacks(best, pool)
    assert out.constraints == [0, 1]
    assert [r.constraints for r in out.fallbacks] == [[2, 3], [1, 2, 3, 4]]
    assert all(r.alternatives is None and r.fallbacks is None for r in out.fallbacks)


def test_the_smallest_answer_after_the_post_pass_wins(monkeypatch):
    """The portfolio's smallest answer is not always the smallest after the
    post-pass (stein45inf: 225 -> 212 but 237 -> 205), so the other complete
    answers are polished too and the smallest result is kept. Before the change
    only the smallest member answer was polished."""
    after = {(0, 1, 4): [0, 1, 4], (1, 2, 3, 4): [2, 3]}
    polished = []

    def fake_polish(arrays, res, **kw):
        polished.append(tuple(res.constraints))
        return replace(res, constraints=after[tuple(res.constraints)])

    monkeypatch.setattr(E, "_polish", fake_polish)
    monkeypatch.delenv("DISCOPT_IIS_POLISH_ALTERNATIVES", raising=False)
    res = _answer({0, 1, 4})
    res.members = [{"order": "ICB"}]
    res.alternatives = [_answer({1, 2, 3, 4})]
    kw = dict(cert_tol=1e-6, time_limit=10.0, engine="highs")
    first = E._polish(_PAIRS, res, **kw)
    out = E._polish_alternatives(_PAIRS, res, first, deadline=time.perf_counter() + 60.0, **kw)
    assert out.constraints == [2, 3] and out.members == [{"order": "ICB"}]
    assert out.alternatives is None and polished == [(0, 1, 4), (1, 2, 3, 4)]
    # Opt-out, and no budget: the first answer stands.
    monkeypatch.setenv("DISCOPT_IIS_POLISH_ALTERNATIVES", "0")
    out0 = E._polish_alternatives(_PAIRS, res, first, deadline=time.perf_counter() + 60.0, **kw)
    assert out0.constraints == [0, 1, 4] and out0.alternatives is None
    monkeypatch.delenv("DISCOPT_IIS_POLISH_ALTERNATIVES")
    out1 = E._polish_alternatives(_PAIRS, res, first, deadline=None, **kw)
    assert out1.constraints == [0, 1, 4]


def _counting_moves(monkeypatch, rehold_to=None):
    """Fake re-hold (moves to ``rehold_to`` when given) and add-one (never moves);
    returns the list of sets add-one ran from."""
    for v in ("DISCOPT_IIS_REHOLD", "DISCOPT_IIS_ADD_ONE"):
        monkeypatch.delenv(v, raising=False)
    starts = []

    def fake_rehold(eng, cons, bounds, ints, stop_at, known=None):
        if rehold_to is None:
            return None, {}, True
        return (set(rehold_to), set(), set()), {}, True

    def fake_add_one(eng, cons, bounds, ints, stop_at):
        starts.append(frozenset(cons))
        return None, {}, True, 1

    monkeypatch.setattr(E, "_rehold", fake_rehold)
    monkeypatch.setattr(E, "_add_one", fake_add_one)
    return starts


def _alt_polish(res, **kw):
    return _polish(
        _PAIRS,
        res,
        cert_tol=1e-6,
        time_limit=10.0,
        engine="highs",
        deadline=time.perf_counter() + 60.0,
        **kw,
    )


def test_an_alternative_the_rehold_move_cannot_change_stops_unless_smaller(monkeypatch):
    """An alternative's post-pass pays only when it ends smaller than the
    incumbent. One the re-hold move leaves unchanged and that is not already
    smaller stops before the add-one move (measured, hexgrid: three such
    alternatives spent ~70 s each and ended unchanged). Before the change the
    add-one move ran from every alternative."""
    starts = _counting_moves(monkeypatch)
    res = _answer({1, 2, 3, 4})
    out = _alt_polish(res, beat=3)
    assert out is res and starts == []
    # Already smaller than the incumbent: it continues.
    _alt_polish(res, beat=5)
    assert starts == [frozenset({1, 2, 3, 4})]
    # No incumbent (the first post-pass): it continues.
    _alt_polish(res)
    assert len(starts) == 2


def test_an_alternative_the_rehold_move_changed_continues(monkeypatch):
    """hexgrid's winner: 308 -> 171 by the re-hold move, not smaller than the
    incumbent's 169, then 159 by the add-one move from 308."""
    starts = _counting_moves(monkeypatch, rehold_to={0, 1, 4})
    _alt_polish(_answer({0, 1, 2, 3}), beat=2)
    assert starts == [frozenset({0, 1, 4}), frozenset({0, 1, 2, 3})]


def test_the_add_one_move_does_not_repeat_a_sweep_from_the_same_set(monkeypatch):
    """Two answers the re-hold move takes to the same set would sweep it twice
    (a sweep that finds no move costs its whole share, ~28 s on bf1355-075).
    Before the change the sweep ran from every start."""
    starts = _counting_moves(monkeypatch, rehold_to={0, 1, 4})
    swept: set = set()
    _alt_polish(_answer({0, 1, 2, 3}), swept=swept)
    _alt_polish(_answer({0, 1, 2, 3, 4}), swept=swept)
    assert starts == [
        frozenset({0, 1, 4}),
        frozenset({0, 1, 2, 3}),
        frozenset({0, 1, 2, 3, 4}),
    ]


def test_post_passes_run_in_member_processes_and_the_smallest_wins():
    """The portfolio answer's and the alternatives' post-passes run at once in
    member processes (the portfolio's workers are idle by then); each result is
    confirmed in its process, the smallest is kept, the members record stays.
    Before the change there was no parallel post-pass."""
    pre = _answer({0, 1, 2, 3})
    pre.members = [{"order": "ICB"}]
    pre.alternatives = [_answer({0, 1, 2, 3, 4})]
    kw = dict(cert_tol=1e-6, time_limit=10.0, engine="highs")
    got = E._polish_parallel(
        _PAIRS, [pre, pre.alternatives[0]], [None, 4], 2, time.perf_counter() + 60.0, kw
    )
    assert len(got) == 2 and all(g is not None for g in got)
    # The portfolio answer's post-pass moves (add-one from {0,1,2,3} reaches a pair).
    assert len(got[0].constraints) == 2 and set(got[0].constraints) < {0, 1, 2, 3}
    # Not smaller than beat=4, and re-hold cannot change it: pruned, unchanged.
    assert got[1].constraints == [0, 1, 2, 3, 4]
    out = E._polish_answers_parallel(_PAIRS, pre, 2, time.perf_counter() + 60.0, kw)
    assert len(out.constraints) == 2 and out.members == [{"order": "ICB"}]
    assert out.alternatives is None


def test_post_passed_answers_rank_a_fully_certified_one_first_under_the_closing_pass(
    monkeypatch,
):
    """The alternatives' post-passes pick the winner with the portfolio's rule:
    under the closing pass a smaller answer missing a witness loses to a fully
    certified one (it is not an IIS under the checker's rule)."""
    small = MilpIIS(constraints=[0, 1], bounds=[], integrality=[], certificates={})
    big = MilpIIS(
        constraints=[0, 1, 2],
        bounds=[],
        integrality=[],
        certificates={("constraint", k): np.zeros(2) for k in range(3)},
    )
    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "1")
    assert E._post_key(big) < E._post_key(small)
    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "0")
    assert E._post_key(small) < E._post_key(big)


def _redundant_elements(arr, res: MilpIIS) -> list:
    """Elements whose removal leaves ``res``'s set infeasible (an independent
    removal probe per element): non-empty means the set is not irreducible."""
    eng = _engine(arr)
    out = []
    for key in E._element_keys(res.constraints, res.bounds, res.integrality):
        c, b, i = set(res.constraints), set(res.bounds), set(res.integrality)
        if key[0] == "constraint":
            c.discard(key[1])
        elif key[0] == "bound":
            b.discard((key[1], key[2]))
        else:
            i.discard(key[1])
        if eng.probe(c, b, i)[0] is True:
            out.append(key)
    return out


def test_exchange_does_not_carry_a_proven_flag_onto_the_bounds_it_adds(monkeypatch):
    """x integer in [0, 1] with x >= 1.5, handed in as a complete answer proven
    irreducible whose integrality has no in-box witness. The exchange replaces
    it by both box bounds, and x >= 0 is redundant: no probe tested the bounds
    it added. With the certificate repair off (or cut short by the clock) the
    lower bound stays, so the answer must not claim irreducibility. It used to:
    ``proven`` was the filter's flag, carried across a set the filter never
    saw. With the repair on the redundant bound is dropped and every element is
    certified, which is a proof by itself."""
    monkeypatch.setenv("DISCOPT_IIS_CLOSE", "1")
    arr = _arrays([[1]], [1.5], [INF], 1, [0], [1], [True])

    def answer():
        return MilpIIS(constraints=[0], bounds=[], integrality=[0], proven_irreducible=True)

    def run():
        return _polish(
            arr,
            answer(),
            cert_tol=1e-6,
            time_limit=10.0,
            engine="highs",
            deadline=time.perf_counter() + 60.0,
        )

    monkeypatch.setenv("DISCOPT_IIS_CERT_REPAIR", "0")
    out = run()
    assert out.n_exchanged == 1 and out.integrality == []
    assert sorted(out.bounds) == [(0, "lower"), (0, "upper")]
    assert _redundant_elements(arr, out) == [("bound", 0, "lower")]
    assert not out.proven_irreducible
    monkeypatch.delenv("DISCOPT_IIS_CERT_REPAIR")
    out = run()
    assert out.n_exchanged == 1 and out.n_close_dropped == 1
    assert out.bounds == [(0, "upper")] and _redundant_elements(arr, out) == []
    assert _checked(arr, out) == _n_elements(out) == 2
    assert out.proven_irreducible
