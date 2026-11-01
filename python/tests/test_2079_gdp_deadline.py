"""#2079: every step of a native GDP solve is charged to ``time_limit``.

Measured on GDPlib ``stranded_gas`` (RIC, 57828 columns x 15241 rows, unloaded
machine): ``time_limit=120`` returned after 216.8 s. The overrun had four parts,
none of which the run's ``Deadline`` saw:

* ``Model.solve``'s passes before the GDP dispatch (34 s), because the
  decomposition was handed the whole ``time_limit`` instead of what was left;
* the GDP lowering (``GDPSpace.from_model``, 20 s), which ran before the
  algorithm's ``Deadline`` was created;
* the master MILP's dense row stack (7909 x 57828, 39 s), built after the limit
  was read, then 6 s of CSC marshaling inside ``milp_simplex.solve_milp`` that
  its own clock did not count;
* a fixed-integer NLP launched after the deadline expired (5 s).

After the fix the same run returns in 121.9 s. These tests pin each part on a
small model with an injected slow step, so they run in well under a second.
"""

from __future__ import annotations

import time
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.gdp.algorithms import GDPAlgorithm
from discopt.gdp.algorithms import init as gdp_init
from discopt.gdp.algorithms import master as gdp_master
from discopt.solvers.gdpopt_loa import native_gdp_class

pytestmark = pytest.mark.smoke

METHODS = ("loa", "gloa", "ric", "lbb", "enumerate", "ldsda")


def _gdp():
    """min (x - 3)^2 over [x <= 2] v [x >= 5], x in [0, 10]: optimum 1 at x = 2."""
    m = dm.Model("g2079")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    m.minimize((x - 3) ** 2)
    return m


class _Clock:
    """A clock the test advances by hand."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


# ── the lowering is charged (GDPSpace.from_model inside the deadline) ───────


@pytest.mark.parametrize("method", METHODS)
def test_the_lowering_is_charged_to_the_deadline(method, monkeypatch):
    clock = _Clock()
    real = gdp_init.GDPSpace.from_model.__func__
    lowerings = []

    def slow_lowering(cls, model):
        lowerings.append(1)
        clock.t += 100.0  # the lowering "takes" 100 s
        return real(cls, model)

    monkeypatch.setattr(gdp_init.GDPSpace, "from_model", classmethod(slow_lowering))
    alg = native_gdp_class(method)(_gdp(), time_limit=60.0, clock=clock)
    assert lowerings == [1], "the probe did not fire"
    assert alg.deadline.elapsed() >= 100.0
    assert alg.deadline.expired()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = alg.run()
    # An expired budget does no work and claims nothing.
    assert r.status == "time_limit"
    assert alg.n_master == 0 and alg.n_subproblems == 0


def test_time_limit_and_deadline_are_exclusive():
    from discopt.gdp.algorithms import Deadline

    with pytest.raises(ValueError, match="not both"):
        GDPAlgorithm(_gdp(), time_limit=5.0, deadline=Deadline(5.0))


# ── Model.solve hands the decomposition what is left of its budget ──────────


@pytest.mark.parametrize("method", METHODS)
def test_model_solve_passes_the_remaining_budget(method, monkeypatch):
    """A pass after ``solve_model``'s clock starts, before the GDP dispatch, is
    charged: here a 0.5 s stand-in for the 34 s ``stranded_gas`` spent there."""
    import discopt.solver as solver
    import discopt.solvers.gdpopt_loa as gl

    seen = {}
    real = gl.solve_gdpopt_loa
    real_const = solver._constant_objective_result

    def slow_pass(*a, **k):
        seen["slow_pass"] = True
        time.sleep(0.5)
        return real_const(*a, **k)

    def spy(model, *, time_limit, **kw):
        seen["time_limit"] = time_limit
        return real(model, time_limit=time_limit, **kw)

    monkeypatch.setattr(solver, "_constant_objective_result", slow_pass)
    monkeypatch.setattr(gl, "solve_gdpopt_loa", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        r = _gdp().solve(gdp_method=method, time_limit=30.0)
        wall = time.perf_counter() - t0
    assert seen.get("slow_pass") and "time_limit" in seen, "the probe did not fire"
    assert seen["time_limit"] <= 29.5
    # The reported wall is the solve's, not the decomposition's alone.
    assert 0.5 <= r.wall_time <= wall


# ── the master: sparse stack, identical rows, build charged ─────────────────


def _dense_reference(rows, rhs, senses, cuts, cut_rhs, n_vars, epigraph, ng_rows, ng_rhs, n_extra):
    """The pre-#2079 dense stack, verbatim in effect."""
    pad = (1 if epigraph else 0) + n_extra
    width = n_vars + pad

    def widen(r):
        r = np.asarray(r, dtype=np.float64)
        return np.concatenate([r, np.zeros(width - r.shape[0])])

    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for r, b, s in zip(rows, rhs, senses):
        if s == "<=":
            a_ub.append(widen(r))
            b_ub.append(b)
        elif s == ">=":
            a_ub.append(-widen(r))
            b_ub.append(-b)
    for r, b in zip(cuts, cut_rhs):
        a_ub.append(widen(r))
        b_ub.append(b)
    for r, b, s in zip(rows, rhs, senses):
        if s == "==":
            a_eq.append(widen(r))
            b_eq.append(b)
    a_ub.extend(ng_rows)
    b_ub.extend(ng_rhs)
    return np.array(a_ub), np.array(b_ub), np.array(a_eq), np.array(b_eq)


@pytest.mark.parametrize("epigraph", [False, True])
@pytest.mark.parametrize("no_goods", [False, True])
def test_master_stack_is_sparse_and_identical(epigraph, no_goods, monkeypatch):
    rng = np.random.default_rng(2079)
    n = 7
    rows = [rng.integers(-2, 3, n).astype(float) for _ in range(6)]
    rows[0][:] = 0.0  # an all-zero row stays a row
    senses = ["<=", ">=", "==", "<=", ">=", "=="]
    rhs = list(rng.normal(size=6))
    cuts = [rng.normal(size=n) for _ in range(2)]
    if epigraph:
        cuts.append(np.append(rng.normal(size=n), -1.0))
    cut_rhs = list(rng.normal(size=len(cuts)))
    integrality = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
    lb, ub = np.zeros(n), np.array([3.0, 3.0] + [10.0] * 5)
    configs = [(1, 2)] if no_goods else None
    captured = {}

    def fake_solver(**kw):
        captured.update(kw)
        return None

    monkeypatch.setattr(
        "discopt.solvers.lp_backend.get_milp_solver", lambda backend="auto": fake_solver
    )
    gdp_master._solve_master_milp(
        rows,
        rhs,
        senses,
        cuts,
        cut_rhs,
        n,
        integrality,
        lb,
        ub,
        None if epigraph else (np.ones(n), 0.0),
        not epigraph,
        epigraph,
        time_limit=10.0,
        gap_tolerance=1e-4,
        no_good_configs=configs,
        int_indices=[0, 1],
    )
    assert captured, "the probe did not fire"
    n_master = n + (1 if epigraph else 0)
    ng_rows, ng_rhs, n_extra = (
        gdp_master._build_no_good_augmentation(configs, [0, 1], lb, ub, n_master)
        if no_goods
        else ([], [], 0)
    )
    a_ub, b_ub, a_eq, b_eq = _dense_reference(
        rows, rhs, senses, cuts, cut_rhs, n, epigraph, ng_rows, ng_rhs, n_extra
    )
    assert sp.issparse(captured["A_ub"]) and sp.issparse(captured["A_eq"])
    got_ub = captured["A_ub"].toarray()
    got_eq = captured["A_eq"].toarray()
    assert got_ub.shape == a_ub.shape and got_eq.shape == a_eq.shape
    np.testing.assert_array_equal(got_ub, a_ub)
    np.testing.assert_array_equal(got_eq, a_eq)
    np.testing.assert_array_equal(captured["b_ub"], b_ub)
    np.testing.assert_array_equal(captured["b_eq"], b_eq)
    # The same nonzeros a dense -> CSR conversion keeps: no stored zeros.
    assert captured["A_ub"].nnz == sp.csr_matrix(a_ub).nnz
    assert captured["A_eq"].nnz == sp.csr_matrix(a_eq).nnz


def test_master_row_wider_than_the_master_is_refused(monkeypatch):
    monkeypatch.setattr(
        "discopt.solvers.lp_backend.get_milp_solver", lambda backend="auto": lambda **kw: None
    )
    with pytest.raises(ValueError, match="entries; the master has"):
        gdp_master._solve_master_milp(
            [np.ones(3)],
            [1.0],
            ["<="],
            [np.ones(4)],  # an epigraph-width cut on a master without one
            [1.0],
            3,
            np.zeros(3, dtype=np.int32),
            np.zeros(3),
            np.ones(3),
            (np.ones(3), 0.0),
            True,
            False,
            time_limit=10.0,
            gap_tolerance=1e-4,
        )


def test_master_build_is_charged_to_its_time_limit(monkeypatch):
    real = gdp_master._stack_sparse
    calls = []

    def slow_stack(*a, **k):
        calls.append(1)
        time.sleep(0.2)
        return real(*a, **k)

    captured = {}

    def fake_solver(**kw):
        captured.update(kw)
        return None

    monkeypatch.setattr(gdp_master, "_stack_sparse", slow_stack)
    monkeypatch.setattr(
        "discopt.solvers.lp_backend.get_milp_solver", lambda backend="auto": fake_solver
    )
    gdp_master._solve_master_milp(
        [np.ones(2)],
        [1.0],
        ["<="],
        [],
        [],
        2,
        np.zeros(2, dtype=np.int32),
        np.zeros(2),
        np.ones(2),
        (np.ones(2), 0.0),
        True,
        False,
        time_limit=1.0,
        gap_tolerance=1e-4,
    )
    assert calls, "the probe did not fire"
    assert captured["time_limit"] <= 0.8


# ── milp_simplex.solve_milp counts from its own entry ───────────────────────


def test_simplex_milp_charges_marshaling(monkeypatch):
    import discopt._rust as rust
    import discopt.solvers.milp_simplex as ms

    real_marshal = ms._marshal_std_form
    real_driver = rust.solve_milp_csc_py
    budgets = []

    def slow_marshal(*a, **k):
        time.sleep(0.2)
        return real_marshal(*a, **k)

    def spy_driver(*a, **k):
        budgets.append(k["time_limit_s"])
        return real_driver(*a, **k)

    monkeypatch.setattr(ms, "_marshal_std_form", slow_marshal)
    monkeypatch.setattr(rust, "solve_milp_csc_py", spy_driver)
    r = ms.solve_milp(
        c=np.array([1.0, 1.0]),
        A_ub=np.array([[-1.0, -1.0]]),
        b_ub=np.array([-1.0]),
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        integrality=np.array([1, 1]),
        time_limit=1.0,
    )
    assert budgets, "the probe did not fire"
    assert all(b <= 0.8 for b in budgets), budgets
    assert r.objective == pytest.approx(1.0)


# ── no NLP is launched once the deadline has expired ────────────────────────


@pytest.mark.parametrize("method", ["loa", "ric"])
def test_no_fixed_integer_nlp_after_the_deadline(method, monkeypatch):
    import discopt.solvers.gdpopt_loa as gl

    clock = _Clock()
    alg = native_gdp_class(method)(_gdp(), time_limit=60.0, clock=clock)

    def must_not_run(*a, **k):
        raise AssertionError("an NLP was launched after the deadline")

    monkeypatch.setattr(gl, "_solve_nlp_subproblem", must_not_run)
    clock.t += 61.0
    assert alg.deadline.expired()
    x_master = np.zeros(alg.n)
    x, rigorous = alg._fixed_integer_nlp(x_master)
    assert x is None
    assert rigorous in (True, False)
    assert "skipped" in alg._step_log
