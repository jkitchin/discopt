"""#1926: the post-NLP dual refit (``lsq_linear``) respects the solve's limit.

``_duals_against_declared_box`` refit the multipliers with SciPy's defaults (up
to 100 ``trf`` iterations, each an LSMR solve of up to ``min(m, n)`` inner
iterations) and no deadline: on ``acopf_case1354pegase_qcqp`` one ``trf``
iteration took ~15 s and the refit ran for minutes after a 60 s limit, without
ever reaching stationarity. Now a refit is not started past the deadline, and a
large sparse one runs under fixed iteration caps.
"""

import time

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._dual_recovery import REFIT_LSMR_MAXITER, REFIT_MAX_TRF_ITER, recover_multipliers


def _problem(n=40, m=30, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 1.5, size=n)
    jac = np.zeros((m, n))
    for i in range(m):
        jac[i, rng.choice(n, 3, replace=False)] = rng.normal(size=3)
    return dict(
        grad=rng.normal(size=n),
        jac=jac,
        body=jac @ x,
        sense_arr=np.array(["<="] * m),
        rhs_arr=jac @ x,  # every row active, so the refit is not skipped as empty
        x_flat=x,
        lb=np.full(n, -10.0),
        ub=np.full(n, 10.0),
        is_continuous=np.ones(n, dtype=bool),
    )


def _spy(monkeypatch):
    import discopt._dual_recovery as DR

    calls = []
    orig = DR.lsq_linear

    def spy(A, b, **kw):
        calls.append((A.shape, sp.issparse(A), kw))
        return orig(A, b, **kw)

    monkeypatch.setattr(DR, "lsq_linear", spy)
    return calls


@pytest.mark.unit
def test_no_refit_is_started_past_the_deadline(monkeypatch):
    calls = _spy(monkeypatch)
    rec = recover_multipliers(**_problem(), deadline=time.perf_counter() - 1.0)
    assert not rec.ok
    assert "time limit" in rec.detail
    assert calls == []


@pytest.mark.unit
def test_a_live_deadline_and_a_small_system_keep_scipy_defaults(monkeypatch):
    calls = _spy(monkeypatch)
    p = _problem()
    ref = recover_multipliers(**p)
    rec = recover_multipliers(**p, deadline=time.perf_counter() + 60.0)
    assert rec.ok and ref.ok
    np.testing.assert_array_equal(rec.mu_full, ref.mu_full)
    assert len(calls) == 2 and all("lsmr_maxiter" not in kw for _, _, kw in calls)


@pytest.mark.unit
def test_a_large_sparse_refit_runs_under_the_iteration_caps(monkeypatch):
    calls = _spy(monkeypatch)
    n = m = REFIT_LSMR_MAXITER + 200
    p = _problem(n=n, m=m)
    p["jac"] = sp.csr_matrix(p["jac"])
    recover_multipliers(**p)
    (shape, is_sparse, kw), *_ = calls
    assert is_sparse and min(shape) > REFIT_LSMR_MAXITER
    assert kw["lsmr_maxiter"] == REFIT_LSMR_MAXITER
    assert kw["max_iter"] == REFIT_MAX_TRF_ITER


def test_the_continuous_solve_hands_the_refit_its_deadline(monkeypatch):
    import warnings

    import discopt.modeling as dm
    import discopt.solver as solver_mod

    seen = []
    real = solver_mod._duals_against_declared_box

    def spy(**kw):
        seen.append(kw.get("deadline"))
        return real(**kw)

    monkeypatch.setattr(solver_mod, "_duals_against_declared_box", spy)
    m = dm.Model("refit_deadline")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.subject_to(x**2 + y**2 <= 1)
    m.minimize(-x)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert seen, "the declared-box dual check never ran"
    assert all(d is not None and t0 < d <= t0 + 31 for d in seen), seen
