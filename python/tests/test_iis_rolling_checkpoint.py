"""Rolling checkpoints: anytime answers from inside a phase.

Before them, a budget that ran out mid-phase fell back to the last *phase-end*
checkpoint, which inside the first phase is checkpoint 0 -- the whole base set.
On MIPcc27 at 600 s that is what dijle / dingo / dives returned (6282 / 4960 /
9500 elements), although the integrality filter had already dropped half the
integralities and those sets confirmed presolve-free in under a second. Now the
engine confirms the current set after a successful deletion once it has shrunk
by 10%, within a 10% time share, and keeps it as the anytime answer.

The budget running out is simulated through ``_Engine._solve`` raising the
engine's own out-of-time signal, so the tests do not depend on timing.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import _iis_milp
from discopt._iis_milp import milp_iis

highspy = pytest.importorskip("highspy")


def _model(n_noise=27):
    """x0 + x1 + x2 = 1.5 over binaries (infeasible only through integrality),
    plus redundant rows over the same binaries that play no part in it. With
    order "CIB" the row phase runs first, with every integrality held, so each
    dropped row makes the reported subsystem smaller."""
    m = dm.Model("roll")
    x = m.binary("x", shape=(3,))
    m.subject_to(x[0] + x[1] + x[2] == 1.5, name="half")
    for k in range(n_noise):
        m.subject_to(x[k % 3] <= 1, name=f"cap{k}")
    m.minimize(0 * x[0])
    return m


def _cut_after(monkeypatch, n_choose):
    """Presolve-on solves after the first ``n_choose`` raise out-of-time."""
    real = _iis_milp._Engine._solve
    calls = {"choose": 0}

    def solve(self, presolve, stop_at=None):
        if presolve == "choose":
            calls["choose"] += 1
            if calls["choose"] > n_choose:
                raise _iis_milp._OutOfTime
        return real(self, presolve, stop_at)

    monkeypatch.setattr(_iis_milp._Engine, "_solve", solve)
    return calls


def _infeasible(model, res):
    """Independent check of the returned subsystem with HiGHS (presolve on/off)."""
    from discopt.decomposition._linear import extract_linear

    lin = extract_linear(model)
    import scipy.sparse as sp

    A = sp.csr_matrix(lin.A)
    lbs = np.concatenate([np.atleast_1d(np.asarray(v.lb, float)).ravel() for v in model._variables])
    ubs = np.concatenate([np.atleast_1d(np.asarray(v.ub, float)).ravel() for v in model._variables])
    n = lbs.size
    held = set(res.integrality)
    kept_b = set(res.bounds)
    lb = np.array(
        [lbs[j] if (j in held or (j, "lower") in kept_b) else -highspy.kHighsInf for j in range(n)]
    )
    ub = np.array(
        [ubs[j] if (j in held or (j, "upper") in kept_b) else highspy.kHighsInf for j in range(n)]
    )
    verdicts = []
    for presolve in ("on", "off"):
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("presolve", presolve)
        h.addVars(n, lb, ub)
        vt = highspy.HighsVarType
        h.changeColsIntegrality(
            n,
            np.arange(n, dtype=np.int32),
            np.array([vt.kInteger if j in held else vt.kContinuous for j in range(n)]),
        )
        for ci in res.constraints:
            for r in [r for r, src in enumerate(lin.row_source) if src == ci]:
                s, e = A.indptr[r], A.indptr[r + 1]
                lo = lin.b[r] if lin.sense[r] in (">=", "==") else -highspy.kHighsInf
                hi = lin.b[r] if lin.sense[r] in ("<=", "==") else highspy.kHighsInf
                h.addRow(lo, hi, e - s, A.indices[s:e].astype(np.int32), A.data[s:e])
        h.run()
        verdicts.append(h.getModelStatus() == highspy.HighsModelStatus.kInfeasible)
    return all(verdicts)


def test_budget_out_mid_phase_returns_a_rolling_checkpoint(monkeypatch):
    m = _model()
    base = 1 + 27 + 3  # every row + every integrality (bounds are boxes of held ints)
    # Enough presolve-on solves for the integrality filter to drop the first
    # batches of y's, then the budget "runs out" mid-phase.
    _cut_after(monkeypatch, 6)
    res = milp_iis(m, total_time_limit=60.0, sat_candidate=False, order="CIB")
    assert not res.complete
    assert res.n_rolling >= 1
    size = len(res.constraints) + len(res.bounds) + len(res.integrality)
    assert size <= 0.9 * base  # smaller than the base set checkpoint 0 holds
    monkeypatch.undo()
    assert _infeasible(m, res)


def test_no_budget_no_rolling_checkpoints():
    res = milp_iis(_model(), sat_candidate=False, order="CIB")
    assert res.complete and res.n_rolling == 0
    assert res.constraints == [0] and sorted(res.integrality) == [0, 1, 2]


def test_a_refuting_rolling_confirmation_redoes_the_run(monkeypatch):
    """If the presolve-free route finds an exact-integer point for a set the
    filter holds as proven, an earlier verdict was wrong: per-probe rerun."""
    real_conf = _iis_milp._Engine.confirm_infeasible
    state = {"n": 0}

    def confirm(self, *a, **k):
        if k.get("presolve") == ("off",) and k.get("stop_at") is not None:
            state["n"] += 1
            if state["n"] == 2:  # the first rolling attempt (checkpoint 0 was n == 1)
                return False
        return real_conf(self, *a, **k)

    monkeypatch.setattr(_iis_milp._Engine, "confirm_infeasible", confirm)
    res = milp_iis(_model(), total_time_limit=60.0, sat_candidate=False, order="CIB")
    assert state["n"] >= 2
    assert res.fallback and res.complete


def test_an_unfinished_run_answers_with_the_smallest_confirmed_set(monkeypatch):
    """Mid-way through the integrality phase the current set counts MORE
    elements than the base set: each relaxed integrality turns the column's two
    finite bounds into elements. An unfinished run must keep the smaller
    checkpoint, not confirm and return the larger current set (dijle: 8005
    instead of 6282 once the final confirmation began to succeed)."""
    m = dm.Model("grow")
    x = m.binary("x", shape=(3,))
    m.subject_to(x[0] + x[1] + x[2] == 1.5, name="half")
    m.integer("y", shape=(27,), lb=0, ub=3)  # in no row: integrality droppable
    m.minimize(0 * x[0])
    base = 1 + 3 + 27  # one row and every integrality (held boxes are not elements)
    calls = _cut_after(monkeypatch, 9)
    res = milp_iis(m, total_time_limit=60.0, sat_candidate=False)
    assert calls["choose"] > 9 and not res.complete
    size = len(res.constraints) + len(res.bounds) + len(res.integrality)
    assert size <= base


def test_a_phase_end_confirmation_out_of_its_own_budget_does_not_end_the_filter(monkeypatch):
    """#2145 review: phase-end confirmations got their own ``stop_at``; when one
    used it up, ``_OutOfTime`` reached the phase loop's outer handler and every
    remaining phase was skipped, although the overall deadline was far off. Now
    it is undecided, like ``roll()``'s. (Rolling checkpoints off; the second
    budgeted confirmation -- after checkpoint 0 -- is the phase-end one.)"""
    monkeypatch.setattr(_iis_milp, "_ROLL_SHRINK", 0.0)
    real = _iis_milp._Engine.confirm_infeasible
    budgeted = {"n": 0}

    def confirm(self, *a, **k):
        if k.get("stop_at") is not None:
            budgeted["n"] += 1
            if budgeted["n"] == 2:
                raise _iis_milp._OutOfTime  # this confirmation's own budget ran out
        return real(self, *a, **k)

    monkeypatch.setattr(_iis_milp._Engine, "confirm_infeasible", confirm)
    res = milp_iis(_model(), total_time_limit=60.0, sat_candidate=False, order="CIB")
    assert budgeted["n"] >= 2  # the phase-end confirmation was reached and ran out
    assert res.complete
    assert res.constraints == [0] and sorted(res.integrality) == [0, 1, 2]
