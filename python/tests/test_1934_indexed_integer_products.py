"""#1934: the integer-product reformulation recognises entries of array variables.

``_scalar_var_ref`` read the indexed variable from ``IndexExpression.variable`` /
``.var``; the node stores it in ``.base``, so every ``x[i]`` resolved to ``None``
and the bilinear, square and #707 multilinear passes never fired on a model
written with array variables. Recognising them is bound-changing (CLAUDE.md §5),
so it sits behind ``DISCOPT_INTEGER_PRODUCT_INDEXED`` (default ON; ``=0`` opts
out). It graduated together with the solver's objective-only decline
(``DISCOPT_IPX_OBJECTIVE_ONLY_DECLINE``), which keeps the lift off models whose
integer products are all in the objective, where it was measured to lose.

Covered here: ``=0`` is the legacy behaviour and the default is ON;
flag ON resolves int, tuple and negative indices to the flat element ``_int_factor_range`` reads,
refuses slices/bools/out-of-range; two references to one element share one
expansion; a 2-D variable's expansion links the right element; and the rewrite
is value-preserving against brute-force enumeration (every integer point of the
original, extended across the lift, satisfies every reformed row with the same
objective, and the reformed model's optimum equals the enumerated one).
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import integer_product_reform as ipr
from discopt.modeling.core import IndexExpression

FLAG = "DISCOPT_INTEGER_PRODUCT_INDEXED"


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def _issue_model():
    m = dm.Model("idx")
    x = m.integer("x", shape=(2,), lb=0, ub=7)
    m.subject_to(x[0] * x[1] <= 20)
    m.minimize(-x[0] - x[1])
    return m, x


def test_opt_out_keeps_legacy_behaviour(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    m, x = _issue_model()
    assert ipr._scalar_var_ref(x[0]) is None
    assert ipr.has_nonconvex_integer_bilinear(m) is False
    assert ipr.reformulate_integer_bilinear(m) is m


def test_default_is_on(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    m, x = _issue_model()
    assert ipr._scalar_var_ref(x[0]) == (x, 0)
    assert ipr.has_nonconvex_integer_bilinear(m) is True


def test_issue_reproducer(flag_on):
    """Fails before the fix: False / None."""
    m, x = _issue_model()
    assert ipr._scalar_var_ref(x[0]) == (x, 0)
    assert ipr.has_nonconvex_integer_bilinear(m) is True
    assert ipr.has_reformulation_work(m) is True


def test_index_resolution(flag_on):
    m = dm.Model("res")
    x = m.integer("x", shape=(3,), lb=0, ub=3)
    w = m.integer("w", shape=(2, 3), lb=0, ub=3)
    checked = 0
    for expr, want in (
        (x[2], (x, 2)),
        (x[-1], (x, 2)),
        (w[0, 1], (w, 1)),
        (w[1, 0], (w, 3)),
        (w[-1, -1], (w, 5)),
        (IndexExpression(x, np.int64(1)), (x, 1)),
    ):
        got = ipr._scalar_var_ref(expr)
        assert got is not None and got[0] is want[0] and got[1] == want[1], (expr, got)
        # _elem_ref is the inverse: it names the same flat element.
        back = ipr._scalar_var_ref(ipr._elem_ref(got[0], got[1]))
        assert back is not None and back[0] is got[0] and back[1] == got[1]
        checked += 1
    for expr in (
        x[0:2],
        w[0],  # partial index: a row, not a scalar
        IndexExpression(x, True),
        IndexExpression(x, 7),
        IndexExpression(w, (0, 9)),
        IndexExpression(x * 2, 0),  # base is not a Variable
    ):
        assert ipr._scalar_var_ref(expr) is None, expr
        checked += 1
    assert checked == 12


def _bits_for(m, var, elem):
    spec = m._ipx_aux_spec
    off = m._flat_var_offset(var) + elem
    return [e for e in spec if e[0] == "bit" and e[1] == off]


def test_shared_element_shares_one_expansion(flag_on):
    m = dm.Model("share")
    x = m.integer("x", shape=(3,), lb=0, ub=7)
    m.subject_to(x[0] * x[1] + x[0] * x[2] <= 20)
    m.minimize(-x[0] - x[1] - x[2])
    rm = ipr.reformulate_integer_bilinear(m)
    assert rm is not m
    # x[0] appears in both products (participation 2, equal ranges), so it is
    # the factor expanded -- and exactly once: 3 bits for [0, 7], not 6.
    assert len(_bits_for(rm, x, 0)) == 3
    assert _bits_for(rm, x, 1) == [] and _bits_for(rm, x, 2) == []


def test_two_dimensional_expansion_links_the_right_element(flag_on):
    """The link row must name ``w[1, 0]`` (flat 3), not ``IndexExpression(w, 3)``,
    which on a shape-(2, 2) variable is out of range rather than an element."""
    m = dm.Model("twod")
    w = m.integer("w", shape=(2, 2), lb=0, ub=3)
    m.subject_to(w[1, 0] * w[0, 1] >= 2)
    m.subject_to(w[1, 0] ** 2 <= 4)
    m.minimize(w[1, 0] + w[0, 1] + w[0, 0] + w[1, 1])
    rm = ipr.reformulate_integer_bilinear(m)
    assert rm is not m
    _check_value_preserving(m, rm, [w])
    r = rm.solve(time_limit=30)
    assert r.objective == pytest.approx(_brute_force(m, [w])[0], abs=1e-6)


def _points(vars_):
    axes = []
    for v in vars_:
        lb = np.asarray(v.lb, dtype=float).ravel()
        ub = np.asarray(v.ub, dtype=float).ravel()
        axes += [range(int(lo), int(hi) + 1) for lo, hi in zip(lb, ub)]
    for p in itertools.product(*axes):
        yield np.asarray(p, dtype=float)


_FNS: dict = {}


def _row_values(model, x_full):
    """Every row's body value and the objective at a flat point (numpy tape;
    each expression is lowered once per model and reused across points)."""
    from discopt._point_eval import compile_point_fn

    key = id(model)
    if key not in _FNS:
        _FNS[key] = (
            [(compile_point_fn(c.body, model), c) for c in model._constraints],
            compile_point_fn(model._objective.expression, model),
        )
    rows, obj = _FNS[key]
    vals = [(float(np.asarray(fn(x_full))), c) for fn, c in rows]
    return vals, float(np.asarray(obj(x_full)))


def _feasible(rows, tol=1e-9):
    for val, c in rows:
        rhs = float(np.asarray(c.rhs)) if c.rhs is not None else 0.0
        if c.sense == "<=" and val - rhs > tol:
            return False
        if c.sense == ">=" and val - rhs < -tol:
            return False
        if c.sense == "==" and abs(val - rhs) > tol:
            return False
    return True


def _brute_force(model, vars_):
    best, n = np.inf, 0
    for p in _points(vars_):
        rows, obj = _row_values(model, p)
        n += 1
        if _feasible(rows):
            best = min(best, obj)
    return best, n


def _check_value_preserving(m, rm, vars_):
    """Every integer point of *m*, extended across the lift, is feasible in *rm*
    iff it is feasible in *m*, with the same objective (no valid point cut, no
    infeasible point admitted)."""
    n = 0
    for p in _points(vars_):
        rows0, obj0 = _row_values(m, p)
        ext = ipr.extend_initial_point(rm, p)
        assert ext is not None
        rows1, obj1 = _row_values(rm, ext)
        assert _feasible(rows1) == _feasible(rows0), p
        assert obj1 == pytest.approx(obj0, abs=1e-9)
        n += 1
    assert n > 0
    return n


@pytest.mark.parametrize("seed", range(4))
def test_random_array_models_value_preserving(flag_on, seed):
    rng = np.random.default_rng(1934 + seed)
    m = dm.Model(f"rand{seed}")
    n = 3
    ub = rng.integers(2, 5, size=n)
    x = m.integer("x", shape=(n,), lb=0, ub=ub)
    y = m.continuous("y", lb=-2.0, ub=3.0)
    pairs = [(0, 1), (1, 2), (0, 2)]
    a = rng.uniform(-1.0, 1.0, size=len(pairs))
    body = sum(float(a[k]) * x[i] * x[j] for k, (i, j) in enumerate(pairs))
    m.subject_to(body + y <= float(rng.uniform(1.0, 4.0)))
    m.subject_to(x[0] * y >= -3.0)
    c = rng.uniform(-2.0, 1.0, size=n)
    m.minimize(sum(float(c[i]) * x[i] for i in range(n)) - y)
    rm = ipr.reformulate_integer_bilinear(m)
    assert rm is not m
    from discopt._relax.term_classifier import classify_nonlinear_terms

    nl = classify_nonlinear_terms(rm)
    assert not (nl.bilinear or nl.trilinear or nl.multilinear or nl.monomial or nl.general_nl)

    # Both arms solve to the same certified optimum, and the reformed model's
    # dual bound never crosses it.
    r_off = m.solve(time_limit=30)
    r_on = rm.solve(time_limit=30)
    assert r_off.status == "optimal" and r_on.status == "optimal"
    assert r_on.objective == pytest.approx(r_off.objective, abs=1e-5)
    if r_on.bound is not None:
        assert r_on.bound <= r_off.objective + 1e-6


def test_multilinear_indexed(flag_on):
    m = dm.Model("ml")
    x = m.integer("x", shape=(3,), lb=0, ub=3)
    m.subject_to(x[0] * x[1] * x[2] >= 4)
    m.minimize(x[0] + x[1] + x[2])
    assert ipr.has_integer_multilinear_reformulation_work(m) is True
    rm = ipr.reformulate_integer_multilinear(m)
    assert rm is not m
    _check_value_preserving(m, rm, [x])
    assert rm.solve(time_limit=30).objective == pytest.approx(_brute_force(m, [x])[0])


def test_solve_end_to_end_matches_brute_force(flag_on):
    m, x = _issue_model()
    best, npts = _brute_force(m, [x])
    assert npts == 64
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(best, abs=1e-6)


# --- the objective-only decline (``DISCOPT_IPX_OBJECTIVE_ONLY_DECLINE``) ---

DECLINE = "DISCOPT_IPX_OBJECTIVE_ONLY_DECLINE"


def _objective_only_model(n=6):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n, n))
    m = dm.Model("objonly")
    x = m.integer("x", shape=(n,), lb=0, ub=4)
    m.minimize(
        sum(float(q[i, j]) * x[i] * x[j] for i in range(n) for j in range(i + 1, n))
        + sum(x[i] for i in range(n))
    )
    m.subject_to(sum(x[i] for i in range(n)) >= n)
    return m, x


@pytest.fixture
def lift_calls(monkeypatch):
    """Record every integer-bilinear lift the solver builds. The registry
    resolves ``integer.bilinear`` to this module attribute at call time."""
    calls: list[bool] = []
    real = ipr.reformulate_integer_bilinear

    def spy(model, **kw):
        out = real(model, **kw)
        calls.append(out is not model)
        return out

    monkeypatch.setattr(ipr, "reformulate_integer_bilinear", spy)
    return calls


def test_constraint_witness_ignores_the_objective(flag_on):
    m, _ = _objective_only_model()
    assert ipr.has_nonconvex_integer_bilinear(m) is True
    assert ipr.has_nonconvex_integer_bilinear(m, in_constraints=True) is False
    m2, _ = _issue_model()  # the product is in a constraint
    assert ipr.has_nonconvex_integer_bilinear(m2, in_constraints=True) is True


def test_objective_only_products_are_not_lifted(flag_on, monkeypatch, lift_calls):
    """Fails without the decline: the objective-only model took the lift."""
    monkeypatch.delenv(DECLINE, raising=False)
    m, x = _objective_only_model()
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert lift_calls == []
    best, _ = _brute_force(m, [x])
    assert r.objective == pytest.approx(best, abs=1e-6)

    monkeypatch.setenv(DECLINE, "0")  # opt-out: the old gate lifts it
    m0, _ = _objective_only_model()
    r0 = m0.solve(time_limit=30)
    assert lift_calls == [True]
    assert r0.objective == pytest.approx(best, abs=1e-6)


def test_constraint_products_are_still_lifted(flag_on, monkeypatch, lift_calls):
    monkeypatch.delenv(DECLINE, raising=False)
    m, x = _issue_model()
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and lift_calls == [True]
    assert r.objective == pytest.approx(_brute_force(m, [x])[0], abs=1e-6)
