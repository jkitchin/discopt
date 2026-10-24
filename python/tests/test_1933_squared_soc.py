"""Regression tests for #1933: a second-order cone written in squared form.

``sum(a_i**2) + c <= k * t**2`` with ``t >= 0`` is a Lorentz cone, but the function
``sum(a_i**2) + c - k t**2`` is not convex, so the classifier could not prove the
row and the solve fell to spatial branch and bound. The norm form
``sqrt(sum(a_i**2) + c) <= sqrt(k) * t`` was already proven convex.

Covered here:

1. ``Model.convexity()`` and the solve rewrite the squared row (``"squared_soc"``)
   and only when the rewrite is exact: rows that are not cones are left alone;
2. the rewrite is set-exact on the box (sampled);
3. ``sqrt`` of a sum of squares is lowered as a norm in both tape builders, so a
   solve at the cone's apex no longer hits ``Invalid_Number_Detected``;
4. constraint duals are reported for the squared row the caller wrote;
5. a convex MIQCP in squared form takes the convex MINLP route, including when
   it arrives from Pyomo through ``.nl``.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.soc_rewrite import rewrite_squared_soc_rows

pounce = pytest.importorskip("pounce")

_ARENA_MODES = ["1", "0"]  # DISCOPT_ARENA_TAPE: arena lowering, Python-DAG lowering


def _issue_model(form: str) -> dm.Model:
    m = dm.Model(form)
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    s = m.continuous("s", lb=0, ub=10)
    if form == "squared":
        m.subject_to(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 0.09 <= s**2, name="soc")
    else:
        m.subject_to(dm.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 0.09) <= s, name="soc")
    m.minimize(x[0] + s)
    return m


# --------------------------------------------------------------------------- #
# 1. Classification and the rewrite's scope.
# --------------------------------------------------------------------------- #


def test_issue_reproducer_is_proven_convex():
    r = _issue_model("squared").convexity()
    assert r.is_convex is True
    assert r.nonconvex_constraints() == []
    assert "squared_soc" in r.rewrites


def test_issue_reproducer_solves_like_the_norm_form():
    sq = _issue_model("squared").solve(time_limit=30)
    nm = _issue_model("norm").solve(time_limit=30)
    assert sq.status == "optimal" and nm.status == "optimal"
    # The two forms are the same problem, so the same optimum.
    assert sq.objective == pytest.approx(nm.objective, abs=1e-6)
    assert sq.objective == pytest.approx(0.0440306561, abs=1e-6)


def _rewritten(build) -> list:
    m = build()
    _, rows = rewrite_squared_soc_rows(m)
    return rows


def _box(lb_s=0.0):
    m = dm.Model()
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    s = m.continuous("s", lb=lb_s, ub=3)
    return m, x, s


def _ge_weighted():
    m, x, s = _box()
    m.subject_to(4 * s**2 >= (x[0] - 1) ** 2 + 2 * x[1] * x[1])
    return m


def _vector_sum():
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    s = m.continuous("s", lb=0, ub=3)
    m.subject_to(dm.sum(x**2) + 0.09 <= s**2)
    return m


def _affine_t():
    m, x, s = _box()
    m.subject_to(x[0] ** 2 + x[1] ** 2 <= (2 * s + 1) ** 2)
    return m


def _square_of_free_variable():
    # pycse's spelling: the constant enters as a variable fixed by an equality.
    m, x, s = _box()
    c = m.continuous("c", lb=-5, ub=5)
    m.subject_to(c == 0.3)
    m.subject_to(x[0] ** 2 + x[1] ** 2 + c**2 <= s**2)
    return m


@pytest.mark.parametrize("build", [_ge_weighted, _vector_sum, _affine_t, _square_of_free_variable])
def test_cone_spellings_are_rewritten(build):
    m = build()
    assert len(_rewritten(build)) == 1
    r = m.convexity()
    assert r.is_convex is True and "squared_soc" in r.rewrites


def _hyperbola():
    # x^2 - 1 <= s^2 is |x| <= sqrt(1 + s^2): not a convex set.
    m, x, s = _box()
    m.subject_to(x[0] ** 2 - 1 <= s**2)
    return m


def _t_may_be_negative():
    # x^2 <= s^2 with s in [-1, 3] is |x| <= |s|: not convex.
    m, x, s = _box(lb_s=-1.0)
    m.subject_to(x[0] ** 2 <= s**2)
    return m


def _linear_term():
    m, x, s = _box()
    m.subject_to(x[0] ** 2 + x[1] <= s**2)
    return m


def _two_negative_squares():
    m, x, s = _box()
    m.subject_to(x[0] ** 2 <= s**2 + x[1] ** 2)
    return m


def _parameter_weight():
    m, x, s = _box()
    p = m.parameter("p", value=2.0)
    m.subject_to(p * x[0] ** 2 + 1 <= s**2)
    return m


def _parameter_in_t():
    # t = p * s is nonnegative now, but a later p < 0 would make it negative.
    m, x, s = _box()
    p = m.parameter("p", value=2.0)
    m.subject_to(x[0] ** 2 + 1 <= (p * s) ** 2)
    return m


def _constant_t():
    # q <= 2**2 is a ball: convex as written, kept for the quadratic routes.
    m, x, s = _box()
    m.subject_to(x[0] ** 2 + x[1] ** 2 <= (0 * s + 2) ** 2)
    return m


def _equality():
    m, x, s = _box()
    m.subject_to(x[0] ** 2 + x[1] ** 2 == s**2)
    return m


@pytest.mark.parametrize(
    "build",
    [
        _hyperbola,
        _t_may_be_negative,
        _linear_term,
        _two_negative_squares,
        _parameter_weight,
        _parameter_in_t,
        _constant_t,
        _equality,
    ],
)
def test_non_cones_are_left_alone(build):
    assert _rewritten(build) == []
    assert "squared_soc" not in build().convexity().rewrites


def test_deep_nl_style_row_is_rewritten():
    """A 3000-term left-nested ``+`` chain (the ``.nl`` shape, #1520) neither
    overflows the stack nor pays a dense eigendecomposition."""
    m = dm.Model()
    x = m.continuous("x", shape=(3000,), lb=-1, ub=1)
    s = m.continuous("s", lb=0, ub=100)
    body = x[0] ** 2
    for i in range(1, 3000):
        body = body + x[i] ** 2
    m.subject_to(body + 0.09 <= s**2)
    m.minimize(s)
    _, rows = rewrite_squared_soc_rows(m)
    assert len(rows) == 1


def test_rewrite_does_not_mutate_the_callers_model():
    m = _issue_model("squared")
    before = list(m._constraints)
    new, rows = rewrite_squared_soc_rows(m)
    assert rows and new is not m
    assert m._constraints == before
    assert new._constraints[0] is not before[0]
    assert new._constraints[0].name == "soc"


# --------------------------------------------------------------------------- #
# 2. Set exactness on the box.
# --------------------------------------------------------------------------- #


def test_rewrite_is_set_exact_on_sampled_points():
    """q <= k t^2 and sqrt(q) <= sqrt(k) t agree at every sampled box point."""
    m = dm.Model()
    x = m.continuous("x", shape=(2,), lb=-2, ub=2)
    s = m.continuous("s", lb=0, ub=2)
    m.subject_to(0.5 * (x[0] - 0.2) ** 2 + 3 * x[1] ** 2 + 0.1 <= 2.5 * (s + 0.1) ** 2)
    new, rows = rewrite_squared_soc_rows(m)
    assert len(rows) == 1
    from discopt._tape_nlp_evaluator import cached_tape_evaluator

    m.minimize(s)
    new.minimize(s)
    ev_old = cached_tape_evaluator(m)
    ev_new = cached_tape_evaluator(new)
    rng = np.random.default_rng(1933)
    pts = np.column_stack(
        [rng.uniform(-2, 2, 4000), rng.uniform(-2, 2, 4000), rng.uniform(0, 2, 4000)]
    )
    compared = 0
    for p in pts:
        g_old = float(ev_old.evaluate_constraints(p)[0])
        g_new = float(ev_new.evaluate_constraints(p)[0])
        if abs(g_old) < 1e-9:
            continue  # on the boundary the two may round differently
        assert (g_old <= 0.0) == (g_new <= 0.0), (p, g_old, g_new)
        compared += 1
    assert compared > 3000
    # Both sides are well populated, so the comparison is not vacuous.
    inside = sum(float(ev_old.evaluate_constraints(p)[0]) <= 0 for p in pts)
    assert 400 < inside < 3600


# --------------------------------------------------------------------------- #
# 3. sqrt of a sum of squares lowers as a norm (both tape builders).
# --------------------------------------------------------------------------- #


def _apex_model(body):
    m = dm.Model("apex")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    s = m.continuous("s", lb=0, ub=10)
    m.subject_to(body(x) <= s)
    m.minimize(s)
    return m


@pytest.mark.parametrize("arena", _ARENA_MODES)
@pytest.mark.parametrize(
    "body, scalar",
    [
        (lambda x: dm.sqrt(x[0] ** 2 + x[1] ** 2), True),
        (lambda x: dm.sqrt(x[0] * x[0] + 2.0 * x[1] ** 2), True),
        (lambda x: dm.sqrt(x[0] ** 2 + (x[1] - x[0]) ** 2), True),
        # An array body: the arena's scalar program refuses it (Python path only).
        (lambda x: dm.sqrt(dm.sum(x**2)), False),
    ],
)
def test_norm_sqrt_has_zero_subgradient_at_apex(monkeypatch, arena, body, scalar):
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", arena)
    m = _apex_model(body)
    if arena == "1" and scalar:
        # Prove the arena lowering is the one under test, not a silent fallback.
        from discopt._arena_tape import try_build_arena_tape

        assert try_build_arena_tape(m, pounce.NlExpr) is not None
    from discopt._tape_nlp_evaluator import cached_tape_evaluator

    jac = cached_tape_evaluator(m).evaluate_jacobian(np.zeros(3))
    assert np.all(np.isfinite(jac)), jac
    assert jac[0, 0] == 0.0 and jac[0, 1] == 0.0 and jac[0, 2] == -1.0


@pytest.mark.parametrize("arena", _ARENA_MODES)
@pytest.mark.parametrize(
    "body",
    [
        lambda x: dm.sqrt(x[0] ** 2 - x[1] ** 2),  # not a sum of squares
        lambda x: dm.sqrt(x[0] ** 2 + x[1]),  # a linear term
        lambda x: dm.sqrt(x[0] * x[1]),  # a product, not a square
    ],
)
def test_other_sqrt_keeps_the_honest_edge(monkeypatch, arena, body):
    """#1491's edge slope stays for every argument that is not a sum of squares."""
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", arena)
    from discopt._tape_nlp_evaluator import cached_tape_evaluator

    jac = cached_tape_evaluator(_apex_model(body)).evaluate_jacobian(np.zeros(3))
    assert not np.all(np.isfinite(jac[0, :2])), jac


@pytest.mark.parametrize("arena", _ARENA_MODES)
def test_norm_sqrt_is_unchanged_away_from_apex(monkeypatch, arena):
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", arena)
    from discopt._tape_nlp_evaluator import cached_tape_evaluator

    ev = cached_tape_evaluator(_apex_model(lambda x: dm.sqrt(x[0] ** 2 + x[1] ** 2)))
    p = np.array([0.3, -0.4, 1.0])
    assert ev.evaluate_constraints(p)[0] == pytest.approx(0.5 - 1.0, abs=1e-15)
    assert ev.evaluate_jacobian(p)[0] == pytest.approx([0.6, -0.8, -1.0], abs=1e-15)


@pytest.mark.parametrize("form", ["squared", "norm"])
def test_solve_at_the_apex_is_optimal(form):
    """Before #1933 the norm form died with Invalid_Number at x = 0 (obj 5, uncertified)."""
    m = dm.Model("apex")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    s = m.continuous("s", lb=0, ub=10)
    if form == "squared":
        m.subject_to(x[0] ** 2 + x[1] ** 2 <= s**2)
    else:
        m.subject_to(dm.sqrt(x[0] ** 2 + x[1] ** 2) <= s)
    m.minimize(s)
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 4. Duals refer to the row the caller wrote.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sense", ["<=", ">="])
def test_duals_satisfy_stationarity_of_the_squared_row(sense):
    from discopt.validation.examiner import examine

    m = dm.Model()
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    s = m.continuous("s", lb=0, ub=10)
    if sense == "<=":
        m.subject_to(x[0] ** 2 + x[1] ** 2 + 0.25 <= 4 * s**2, name="cone")
    else:
        m.subject_to(4 * s**2 >= x[0] ** 2 + x[1] ** 2 + 0.25, name="cone")
    m.subject_to(x[0] + x[1] >= 1, name="lin")
    m.minimize(s)
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(math.sqrt(0.75) / 2, abs=1e-6)
    assert r.constraint_duals is not None
    # The norm row's multiplier is 0.5; the squared row's is 0.5 / (2 * 2 * s*).
    assert float(r.constraint_duals["cone"]) == pytest.approx(0.5 / (4 * r.objective), rel=1e-5)
    checks = {c.name: c for c in examine(r, m).checks}
    stat = checks["stationarity (solver duals)"]
    assert stat.passed, stat


# --------------------------------------------------------------------------- #
# 5. A convex MIQCP in squared form takes the convex MINLP route.
# --------------------------------------------------------------------------- #


def _miqcp(form: str) -> dm.Model:
    m = dm.Model(form)
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    s = m.continuous("s", lb=0, ub=10)
    y = m.binary("y")
    if form == "squared":
        m.subject_to(x[0] ** 2 + x[1] ** 2 + 0.01 <= s**2)
    else:
        m.subject_to(dm.sqrt(x[0] ** 2 + x[1] ** 2 + 0.01) <= s)
    m.subject_to(x[0] + x[1] >= y)
    m.minimize(s - 0.5 * y)
    return m


def test_squared_miqcp_takes_the_convex_route():
    sq = _miqcp("squared").solve(time_limit=30)
    nm = _miqcp("norm").solve(time_limit=30)
    assert sq.status == "optimal" and sq.objective == pytest.approx(0.1, abs=1e-6)
    assert sq.objective == pytest.approx(nm.objective, abs=1e-6)
    assert sq.algorithm_route is not None and "mip-nlp" in sq.algorithm_route
    assert sq.algorithm_route == nm.algorithm_route


def test_pyomo_squared_cone_through_nl(tmp_path):
    pyo = pytest.importorskip("pyomo.environ")

    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 2))
    c0 = rng.normal(size=4)
    pm = pyo.ConcreteModel()
    pm.x = pyo.Var(range(2), bounds=(-1, 1))
    pm.y = pyo.Var(range(3), within=pyo.Binary)
    pm.r = pyo.Var(range(4))
    pm.rdef = pyo.Constraint(
        range(4),
        rule=lambda b, i: (
            b.r[i] == sum(A[i, j] * b.x[j] for j in range(2)) + c0[i] + 0.3 * b.y[i % 3]
        ),
    )
    pm.std = pyo.Var(within=pyo.NonNegativeReals)
    pm.c = pyo.Var()  # free, fixed by an equality: pycse's ``_add_std`` spelling
    pm.cdef = pyo.Constraint(expr=pm.c == 0.3)
    pm.cone = pyo.Constraint(expr=sum(pm.r[i] ** 2 for i in range(4)) + pm.c**2 <= pm.std**2)
    pm.pick = pyo.Constraint(expr=sum(pm.y[i] for i in range(3)) == 1)
    pm.obj = pyo.Objective(expr=pm.x[0] - pm.x[1] + 2 * pm.std)
    path = tmp_path / "squared_cone.nl"
    pm.write(str(path))

    m = dm.from_nl(str(path))
    report = m.convexity()
    assert report.is_convex is True and "squared_soc" in report.rewrites
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.algorithm_route is not None and "mip-nlp" in r.algorithm_route
    assert r.objective == pytest.approx(1.5683270831, abs=1e-6)


def test_exhausted_work_budget_stops_the_scan(monkeypatch):
    """#1456: the pass runs before ``time_limit`` is armed, so it carries a
    deterministic budget; spending it leaves the model unchanged, never raises."""
    import discopt._relax.soc_rewrite as soc

    m = _issue_model("squared")
    monkeypatch.setattr(soc, "_SOC_SCAN_WORK_BUDGET", 3)
    new, rows = soc.rewrite_squared_soc_rows(m)
    assert rows == [] and new is m
    monkeypatch.setattr(soc, "_SOC_SCAN_WORK_BUDGET", 10_000)
    assert len(soc.rewrite_squared_soc_rows(m)[1]) == 1


def _two_cones() -> dm.Model:
    m = dm.Model("two_cones")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    s = m.continuous("s", shape=(2,), lb=0, ub=10)
    m.subject_to(x[0] ** 2 + 0.09 <= s[0] ** 2, name="cone0")
    m.subject_to(x[1] ** 2 + 0.09 <= s[1] ** 2, name="cone1")
    m.minimize(x[0] + x[1] + s[0] + s[1])
    return m


def test_spent_time_limit_abandons_the_pass_wholesale():
    """#1456 abandon hook: checked per row; when it fires, everything rewritten
    so far is discarded and the model comes back unchanged."""
    m = _two_cones()
    assert len(rewrite_squared_soc_rows(m)[1]) == 2  # the control

    new, rows = rewrite_squared_soc_rows(m, deadline=lambda: True)
    assert rows == [] and new is m

    # Fires only after the first row was rewritten: still nothing is kept.
    calls = {"n": 0}

    def expires_after_first_row() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    new, rows = rewrite_squared_soc_rows(m, deadline=expires_after_first_row)
    assert calls["n"] == 2, "the hook must be read once per row, in the outer loop"
    assert rows == [] and new is m


def test_solve_hands_the_pass_its_presolve_gate(monkeypatch):
    """The solve gates the pass at entry and passes the per-row abandon hook."""
    import discopt._relax.soc_rewrite as soc

    seen = []
    real = soc.rewrite_squared_soc_rows

    def spy(model, deadline=None):
        seen.append(deadline)
        return real(model, deadline)

    monkeypatch.setattr(soc, "rewrite_squared_soc_rows", spy)
    r = _issue_model("squared").solve(time_limit=30)
    assert r.status == "optimal"
    assert seen and callable(seen[0]) and seen[0]() is False


def test_vector_shaped_row_is_declined():
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    s = m.continuous("s", lb=0, ub=10)
    m.subject_to(x**2 <= s**2)
    m.minimize(s)
    assert rewrite_squared_soc_rows(m)[1] == []
