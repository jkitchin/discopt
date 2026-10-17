"""Regression tests for #1617: crashes and spurious refusals.

Each test pins one witness from the issue that, on the parent of this change,
either crashed, refused a model it can solve, or answered falsely.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# C-15: a zero-length variable crashed the defined-variable FBBT box build.
# ---------------------------------------------------------------------------


def test_zero_length_variable_does_not_crash_the_solve():
    m = dm.Model("zero_len")
    x = m.binary("x", shape=(2,))
    m.continuous("y", shape=(0,))
    m.minimize(x[0] + x[1])
    m.subject_to(x[0] + x[1] >= 1)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.error)
    assert r.objective == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# B-19d: a vector variable with ub=np.inf crashed the same box build (it was
# built as a scalar Interval from the first element).
# ---------------------------------------------------------------------------


def test_vector_variable_with_infinite_upper_bound_solves():
    m = dm.Model("vec_inf")
    s = m.continuous("s", shape=(2,), lb=0, ub=np.inf)
    y = m.continuous("y", lb=0, ub=np.inf)
    m.subject_to(y == s[0] + 1.0)
    m.minimize(y + s[0])
    r = m.solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.error)
    assert r.objective == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# X-32a: a coefficient below HiGHS's small_matrix_value made the MILP route
# return a causeless ``error``. The tiny term is now absorbed into a row range.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [1e-13, 1e-12, 1e-11, 0.0])
def test_tiny_coefficient_milp_is_solved(eps):
    m = dm.Model("tiny")
    x = m.continuous("x", lb=-1, ub=1)
    a = m.continuous("a", lb=0, ub=2)
    q = m.binary("q")
    m.subject_to(a >= x + eps * q)
    m.subject_to(a <= 2 * q)
    m.maximize(a - 0.5 * x)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", (eps, r.status, r.error)
    assert r.objective == pytest.approx(2.5, abs=1e-6)


def test_tiny_coefficient_is_not_proved_infeasible_by_bound_tightening():
    """``1e-13*y >= 5`` with ``y <= 1e14`` is feasible (y = 5e13). The univariate
    tightener treated ``|b| <= 1e-12`` as zero and proved it infeasible."""
    m = dm.Model("tiny_int")
    y = m.integer("y", lb=0, ub=1e14)
    m.subject_to(1e-13 * y >= 5.0)
    m.minimize(y)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.error)
    assert r.objective == pytest.approx(5e13, rel=1e-6)

    m = dm.Model("tiny_cont")
    y = m.continuous("y", lb=0, ub=1e14)
    m.subject_to(1e-13 * y >= 5.0)
    m.minimize(y)
    r = m.solve(time_limit=30)
    # The LP route cannot pass the coefficient to HiGHS exactly; it may refuse,
    # but it must never claim the feasible model infeasible, and a refusal must
    # say why.
    assert r.status != "infeasible"
    if r.status == "optimal":
        assert r.objective == pytest.approx(5e13, rel=1e-6)
    else:
        assert r.error and "without recording a cause" not in r.error, r.error


def test_absorb_tiny_entries_is_a_sound_relaxation():
    from discopt.solvers.lp_milp_highs import (
        INF,
        SMALL_MATRIX_VALUE,
        StdForm,
        _absorb_tiny_entries,
    )

    A = np.array(
        [
            [1.0, 1e-13, -5e-13, 0.0],
            [2.0, 0.0, 0.0, 1e-14],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    b = np.array([1.0, 3.0, 2.0])
    xl = np.array([0.0, -10.0, 0.0, 0.0])
    xu = np.array([10.0, 10.0, 4.0, INF])
    sf = StdForm.from_arrays(np.ones(4), A, b, xl, xu)
    sf2, lo, hi, n = _absorb_tiny_entries(sf)
    assert n == 3
    assert np.all(np.abs(sf2.A.data) > SMALL_MATRIX_VALUE)
    # Row 2 untouched: still an equality.
    assert lo[2] == hi[2] == 2.0
    # Row 1 has a term on an open column side: that side of the range opens.
    assert lo[1] == -np.inf and np.isfinite(hi[1])
    # Every point of the original box/rows satisfies the ranged rows.
    rng = np.random.default_rng(0)
    checks = 0
    for _ in range(2000):
        x = rng.uniform(xl, np.minimum(xu, 1e6))
        # Solve for x0 so that row 0 holds exactly, when it stays in its box.
        x[0] = b[0] - 1e-13 * x[1] + 5e-13 * x[2]
        if not (xl[0] <= x[0] <= xu[0]):
            continue
        r = sf2.A @ x
        assert lo[0] <= r[0] <= hi[0]
        checks += 1
    assert checks > 1000


def test_highs_route_error_records_its_cause():
    """An ``error`` from the HiGHS LP route carries HiGHS's reason, not the generic
    backfill."""
    m = dm.Model("lp_rhs_sentinel")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(2 * x <= 1.94849311702961e20)
    m.minimize(-x)
    r = m.solve(time_limit=20)
    assert r.status == "error"
    assert r.error and "without recording a cause" not in r.error, r.error


# ---------------------------------------------------------------------------
# D-04: AMP refused milp_solver="highs".
# ---------------------------------------------------------------------------


def test_amp_accepts_highs_milp_solver():
    m = dm.Model("bilin")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x * y >= 4)
    m.minimize(x + y)
    r = m.solve(solver="amp", milp_solver="highs", time_limit=30)
    assert r.status in ("optimal", "feasible", "time_limit"), (r.status, r.error)
    assert r.objective == pytest.approx(4.0, abs=1e-3)
    assert r.bound is None or r.bound <= 4.0 + 1e-6


# ---------------------------------------------------------------------------
# B-11: estimate_parameters refused a local solver's "local_optimal" status.
# ---------------------------------------------------------------------------


def test_estimate_parameters_accepts_local_optimal():
    from discopt.estimate import Experiment, ExperimentModel, estimate_parameters

    t = np.linspace(0, 4, 8)
    y = 3 * np.exp(-0.7 * t)

    class Exp(Experiment):
        def create_model(self, **kw):
            m = dm.Model("e")
            A = m.continuous("A", lb=0.1, ub=10)
            k = m.continuous("k", lb=0.01, ub=5)
            return ExperimentModel(
                model=m,
                unknown_parameters={"A": A, "k": k},
                design_inputs={},
                responses={f"y{i}": A * dm.exp(-k * ti) for i, ti in enumerate(t)},
                measurement_error={f"y{i}": 0.05 for i in range(8)},
            )

    est = estimate_parameters(
        Exp(), {f"y{i}": y[i] for i in range(8)}, solver_options={"solver": "pounce"}
    )
    assert est.parameters["A"] == pytest.approx(3.0, abs=1e-3)
    assert est.parameters["k"] == pytest.approx(0.7, abs=1e-3)


# ---------------------------------------------------------------------------
# C-02b: a BooleanVar indicator in ``if_then`` crashed deep in lowering.
# ---------------------------------------------------------------------------


def test_if_then_accepts_a_boolean_var():
    m = dm.Model("b")
    Y = m.boolean("Y")
    x = m.continuous("x", lb=0, ub=10)
    m.if_then(Y, [x >= 5])
    m.minimize(x)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.error)
    assert r.objective == pytest.approx(0.0, abs=1e-6)

    m = dm.Model("b_forced")
    Y = m.boolean("Y")
    x = m.continuous("x", lb=0, ub=10)
    m.if_then(Y, [x >= 5])
    m.subject_to(Y.variable >= 1)
    m.minimize(x)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.error)
    assert r.objective == pytest.approx(5.0, abs=1e-6)


def test_if_then_rejects_a_logical_expression_at_call_time():
    m = dm.Model("b_bad")
    A, B = m.boolean("A"), m.boolean("B")
    x = m.continuous("x", lb=0, ub=10)
    with pytest.raises(TypeError, match="if_then"):
        m.if_then(A & B, [x >= 5])


# ---------------------------------------------------------------------------
# C-02c: the native MILP backend classified the model BEFORE GDP lowering and
# refused an ``either_or`` model as not-an-LP/MILP.
# ---------------------------------------------------------------------------


def test_native_milp_backend_classifies_the_lowered_model():
    m = dm.Model("c")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 6]])
    m.minimize(x - 4)
    r = m.solve(milp_backend="native", milp_cuts=False, time_limit=30)
    assert r.status == "optimal", (r.status, r.error)
    assert r.objective == pytest.approx(-4.0, abs=1e-6)


# ---------------------------------------------------------------------------
# C-12a: hull on an unbounded disjunction variable emitted a 1e15-sized
# coefficient and failed with a causeless ``error``; big-M's refusal sent the
# user to hull. Both now refuse loudly and accurately.
# ---------------------------------------------------------------------------


def _unbounded_gdp():
    m = dm.Model("u")
    x = m.continuous("x", lb=0)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x >= 5 + y, y <= 2], [x <= 3, y >= 6]])
    m.minimize(x + (10 - y))
    return m


def test_hull_refuses_an_unbounded_disjunction_variable_by_name():
    with pytest.raises(ValueError, match=r"hull.*'x'.*finite"):
        _unbounded_gdp().solve(gdp_method="hull", time_limit=30)


def test_big_m_refusal_does_not_promise_hull_handles_unbounded_variables():
    with pytest.raises(ValueError) as ei:
        _unbounded_gdp().solve(gdp_method="big-m", time_limit=30)
    assert "needs a finite bound" in str(ei.value)


# ---------------------------------------------------------------------------
# C-16: on a model deep enough for ``_scoped_deep_recursion`` to move the solve
# to a worker thread, the B&B's callback-enforcement mark (a threading.local) was
# set on the worker and lost, and the caller refused a correctly screened result.
# The same loss hid callback failures from the #1436 refusal.
# ---------------------------------------------------------------------------


def _deep_lazy_model(n=900):
    m = dm.Model("deep_lazy")
    x = m.binary("x", shape=(n,))
    w = np.linspace(1.0, 2.0, n)
    # One long objective expression: deeper than _DEEP_SOLVE_DEPTH_GATE. Python's
    # ``sum`` builds the ``+`` chain itself; ``dm.sum`` may fold these terms into
    # one shallow ``dm.sum(w * x)`` (#1712), which would not engage the worker.
    m.minimize(sum(float(w[k]) * x[k] for k in range(n)))
    m.subject_to(dm.sum(lambda k: x[k], over=range(n)) >= 2)
    return m, x


def test_lazy_constraints_survive_the_deep_recursion_worker_thread():
    from discopt._relax.factorable_reform import _max_expr_node_count
    from discopt.callbacks import CutResult
    from discopt.solver import _DEEP_SOLVE_DEPTH_GATE

    m, x = _deep_lazy_model()
    assert _max_expr_node_count(m) > _DEEP_SOLVE_DEPTH_GATE  # the worker path engages
    calls = [0]

    def lazy(ctx, model):
        calls[0] += 1
        xr = np.asarray(ctx.x_relaxation).ravel()
        # Forbid the two cheapest items together.
        if xr[0] + xr[1] > 1.0 + 1e-6:
            return [CutResult(terms=[(x[0], 1.0), (x[1], 1.0)], sense="<=", rhs=1.0)]
        return []

    r = m.solve(lazy_constraints=lazy, time_limit=120)
    assert calls[0] > 0
    assert r.status == "optimal", (r.status, r.error)
    xv = np.asarray(r.x["x"]).ravel()
    assert xv[0] + xv[1] <= 1.0 + 1e-6
    w = np.linspace(1.0, 2.0, 900)
    assert r.objective == pytest.approx(w[0] + w[2], abs=1e-6)


def test_callback_failure_on_the_deep_recursion_worker_is_refused():
    from discopt.solver import FeasibilityCallbackError

    m, _x = _deep_lazy_model()

    def lazy(ctx, model):
        raise RuntimeError("boom")

    with pytest.raises(FeasibilityCallbackError):
        m.solve(lazy_constraints=lazy, time_limit=120)
