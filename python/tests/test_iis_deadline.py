"""``compute_iis(total_time_limit=...)``: an overall budget with an anytime result.

The competition this was built for (MIPcc27) kills a run at its limit and
discards its output, and HiGHS's own ``time_limit`` was measured overrunning
7-10x on presolve-free re-solves. So the matrix engine enforces the budget with
a wall-clock interrupt on every HiGHS run and, when the budget runs out, returns
the subsystem it has -- still proven infeasible, ``complete=False`` -- or raises
``TimeoutError`` if nothing could be confirmed. Never an unconfirmed claim.
"""

import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.infeasibility import compute_iis

highspy = pytest.importorskip("highspy")


def _redundant_rows_model(n_red=60):
    m = dm.Model("many")
    x = m.continuous("x", shape=(n_red + 2,), lb=0, ub=10)
    m.subject_to(x[0] >= 5, name="big")
    for k in range(1, n_red + 1):
        m.subject_to(x[k] + x[k + 1] <= 15, name=f"rh{k}")
    m.subject_to(x[0] <= 2, name="small")
    m.minimize(0 * x[0])
    return m


def _independently_infeasible(model, iis):
    """Solve the returned subsystem with a fresh HiGHS model built from the IIS."""
    from discopt.decomposition._linear import extract_linear

    lin = extract_linear(model)
    keep = {id(c) for c in iis.constraints}
    rows = [r for r, src in enumerate(lin.row_source) if id(model._constraints[src]) in keep]
    n = lin.n
    lb = np.concatenate(
        [np.broadcast_to(np.asarray(v.lb, float), v.shape).ravel() for v in model._variables]
    )
    ub = np.concatenate(
        [np.broadcast_to(np.asarray(v.ub, float), v.shape).ravel() for v in model._variables]
    )
    flat = {}
    off = 0
    for v in model._variables:
        flat[id(v)] = off
        off += v.size
    inf = highspy.kHighsInf
    clb, cub = np.full(n, -inf), np.full(n, inf)
    for v, side, idx in iis.bound_elements:
        j = flat[id(v)] + (int(np.ravel_multi_index(idx, v.shape)) if idx else 0)
        if side == "lower":
            clb[j] = lb[j]
        else:
            cub[j] = ub[j]
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("presolve", "off")
    h.addVars(n, clb, cub)
    A = lin.A.tocsr()
    for r in rows:
        s, e = A.indptr[r], A.indptr[r + 1]
        lo = lin.b[r] if lin.sense[r] in (">=", "==") else -inf
        hi = lin.b[r] if lin.sense[r] in ("<=", "==") else inf
        h.addRow(lo, hi, e - s, A.indices[s:e].astype(np.int32), A.data[s:e])
    h.run()
    return h.getModelStatus() == highspy.HighsModelStatus.kInfeasible


@pytest.mark.smoke
@pytest.mark.parametrize("confirm", ["each", "final"])
def test_budget_returns_a_confirmed_anytime_subsystem(monkeypatch, confirm):
    from discopt import _iis_milp

    real_run = highspy.Highs.run

    def slow_run(self):
        # Inside HiGHS's run, so the engine's own solve timing sees it (its
        # confirmation reserve is sized from those timings).
        time.sleep(0.15)
        return real_run(self)

    monkeypatch.setattr(highspy.Highs, "run", slow_run)
    real_milp_iis = _iis_milp.milp_iis
    monkeypatch.setattr(
        _iis_milp, "milp_iis", lambda m, **kw: real_milp_iis(m, confirm=confirm, **kw)
    )
    m = _redundant_rows_model()
    # ~0.15 s per solve: room for the base probe, a checkpoint confirmation,
    # some filtering and the final two-solve confirmation -- but not the ~20+
    # solves the full filter needs.
    budget = 2.5
    t0 = time.perf_counter()
    iis = compute_iis(m, include_integrality=True, total_time_limit=budget)
    elapsed = time.perf_counter() - t0
    assert not iis.complete and not iis.proven_irreducible
    assert elapsed < budget + 1.0, elapsed
    names = {c.name for c in iis.constraints}
    assert {"big", "small"} <= names  # the conflict is never dropped
    assert len(names) < 62  # and the filter did make progress
    assert _independently_infeasible(m, iis)


@pytest.mark.smoke
def test_default_path_honours_the_budget(monkeypatch):
    from discopt.modeling.core import Model

    real = Model.solve

    def slow(self, *a, **k):
        time.sleep(0.1)
        return real(self, *a, **k)

    monkeypatch.setattr(Model, "solve", slow)
    m = _redundant_rows_model(40)
    t0 = time.perf_counter()
    iis = compute_iis(m, total_time_limit=1.0)
    assert time.perf_counter() - t0 < 2.5
    assert not iis.complete and not iis.proven_irreducible
    names = {c.name for c in iis.constraints}
    assert {"big", "small"} <= names and len(names) < 42


@pytest.mark.smoke
def test_no_time_to_prove_infeasibility_raises():
    with pytest.raises(TimeoutError):
        compute_iis(_redundant_rows_model(), include_integrality=True, total_time_limit=1e-4)


def _market_split(m_rows=4, n=36, seed=7):
    """Cornuejols-Dawande market split: LP-feasible, notoriously hard for B&B."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 100, size=(m_rows, n))
    rhs = A.sum(axis=1) // 2
    mod = dm.Model("ms")
    x = mod.binary("x", shape=(n,))
    for i in range(m_rows):
        mod.subject_to(sum(float(A[i, j]) * x[j] for j in range(n)) == float(rhs[i]))
    mod.minimize(0 * x[0])
    return mod


@pytest.mark.smoke
def test_wall_guard_stops_highs_when_its_own_limit_is_ignored(monkeypatch):
    """With HiGHS's time_limit disabled only the interrupt callback can stop a run."""
    from discopt import _iis_milp

    real_set = highspy.Highs.setOptionValue

    def no_time_limit(self, name, value):
        if name == "time_limit":
            return real_set(self, name, highspy.kHighsInf)
        return real_set(self, name, value)

    monkeypatch.setattr(highspy.Highs, "setOptionValue", no_time_limit)
    statuses = []
    real_solve = _iis_milp._Engine._solve

    def record(self, presolve, stop_at=None):
        st = real_solve(self, presolve, stop_at)
        statuses.append(st)
        return st

    monkeypatch.setattr(_iis_milp._Engine, "_solve", record)
    budget = 2.0
    t0 = time.perf_counter()
    with pytest.raises((TimeoutError, ValueError)):
        _iis_milp.milp_iis(_market_split(), time_limit=600, total_time_limit=budget)
    elapsed = time.perf_counter() - t0
    assert highspy.HighsModelStatus.kInterrupt in statuses, statuses  # the guard fired
    assert elapsed < budget + 2.0, elapsed
