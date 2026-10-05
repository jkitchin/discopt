"""#1620 remaining items: relax_integrality, to_gams(initial_point=), parameter
folding in to_lp/to_mps (B-21b), solve(pounce_scaling=) (B-04), and the HiGHS
route's slack-form explanation (C-04)."""

from __future__ import annotations

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.export import to_gams, to_lp, to_mps

pytestmark = pytest.mark.smoke


# ── relax_integrality ────────────────────────────────────────────────────────


def _bin_packing():
    loads = np.array([5.8, 5.7, 5.4, 5.3, 5.2, 5.0])
    m = dm.Model("bp")
    x = m.binary("x", shape=(6, 6))
    y = m.binary("y", shape=(6,))
    m.minimize(dm.sum(lambda j: y[j], over=range(6)))
    for i in range(6):
        m.subject_to(dm.sum(lambda j: x[i, j], over=range(6)) == 1)
    for j in range(6):
        m.subject_to(dm.sum(lambda i: loads[i] * x[i, j], over=range(6)) <= 10 * y[j])
    return m


def test_relax_integrality_is_registered_as_inexact():
    t = dt.get("core.relax_integrality")
    assert t.exact is False
    assert "core.relax_integrality" in dt.available()


def test_relax_integrality_gives_the_lp_bound_and_leaves_the_source_alone():
    m = _bin_packing()
    lp = dt.create_using("core.relax_integrality", m)
    assert all(v.var_type == dm.VarType.BINARY for v in m._variables)
    assert all(v.var_type == dm.VarType.CONTINUOUS for v in lp._variables)
    # Binaries keep their [0, 1] box.
    for v in lp._variables:
        assert np.all(v.lb == 0.0) and np.all(v.ub == 1.0)
    r_lp = lp.solve()
    r_ip = m.solve()
    assert r_lp.status == "optimal" and r_ip.status == "optimal"
    # LP relaxation of bin packing: sum(loads) / capacity.
    assert r_lp.objective == pytest.approx(32.4 / 10.0, abs=1e-6)
    assert r_lp.objective <= r_ip.objective + 1e-9
    assert r_ip.objective == pytest.approx(6.0)


def test_relax_integrality_in_place_reaches_the_fast_api_builder():
    m = dm.Model("f")
    x = m.integer("x", shape=(2,), lb=0, ub=10)
    m.add_linear_constraints(sp.csr_matrix([[2.0, 3.0]]), x, "<=", np.array([7.5]))
    m.add_linear_objective(np.array([-1.0, -1.0]), x)
    assert m.solve().objective == pytest.approx(-3.0)
    dt.apply_to("core.relax_integrality", m)
    assert m.solve().objective == pytest.approx(-3.75)


def test_relax_integrality_refuses_indicator_relations():
    m = dm.Model("ind")
    z = m.binary("z")
    x = m.continuous("x", lb=0, ub=10)
    m.if_then(z, [x >= 5])
    m.minimize(x + z)
    with pytest.raises(ValueError, match="binary indicators"):
        dt.create_using("core.relax_integrality", m)


def test_relax_integrality_clears_implied_integer_auxes():
    m = _bin_packing()
    m._implied_integer_auxes = {"w0"}
    dt.apply_to("core.relax_integrality", m)
    assert m._implied_integer_auxes == set()


# ── B-14: to_gams(initial_point=) ────────────────────────────────────────────


def _gams_model():
    m = dm.Model("g")
    x = m.continuous("x", lb=1, ub=3)
    y = m.continuous("y", shape=(2,), lb=0, ub=4)
    z = m.integer("z", lb=0, ub=5)
    b = m.binary("b")
    m.minimize(dm.log(x) + y[0] * y[1] + z + b)
    m.subject_to(x + y[0] >= 1)
    return m, x, y, z, b


def _levels(text):
    return [ln for ln in text.splitlines() if ".l" in ln and ".lo" not in ln]


def test_to_gams_writes_the_supplied_point_as_levels():
    m, x, y, z, b = _gams_model()
    text = to_gams(m, initial_point={x: 1.5, y: [0.25, 3.0], z: 4, b: 1})
    assert _levels(text) == [
        "x.l = 1.5;",
        "y.l('1') = 0.25;",
        "y.l('2') = 3.0;",
        "z.l = 4.0;",
        "b.l = 1.0;",
    ]


def test_to_gams_default_uses_the_attached_point_then_the_midpoint_fallback():
    m, x, y, z, b = _gams_model()
    assert _levels(to_gams(m)) == ["x.l = 2.0;", "z.l = 2.5;"]
    m.set_initial_point({x: 2.75})
    assert _levels(to_gams(m)) == ["x.l = 2.75;", "z.l = 2.5;"]
    # An explicit {} writes no supplied point, as to_nl's contract.
    assert _levels(to_gams(m, initial_point={})) == ["x.l = 2.0;", "z.l = 2.5;"]


def test_to_gams_initial_point_is_validated_like_to_nl():
    m, x, y, z, b = _gams_model()
    # Clamped into the box and integrality-rounded by validate_initial_solution.
    assert "x.l = 3.0;" in _levels(to_gams(m, initial_point={x: 99.0}))
    assert "z.l = 4.0;" in _levels(to_gams(m, initial_point={z: 3.8}))


# ── B-21b: parameters fold to their value in LP/MPS, either spelling ─────────


@pytest.mark.parametrize("writer", [to_lp, to_mps])
def test_parameter_folds_in_both_spellings(writer):
    def build(case):
        m = dm.Model("p")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        p = m.parameter("p", value=2.0)
        if case == "rhs":
            m.subject_to(x[0] + y >= 8 * p)
        else:
            m.subject_to(dm.sum(lambda i: x[i] - 4 * p, over=range(3)) >= 0)
        m.minimize(dm.sum(x) + y)
        return m

    rhs, summed = writer(build("rhs")), writer(build("sum"))
    if writer is to_lp:
        assert "<= -16" in rhs and "<= -24" in summed
    else:
        assert "-16" in rhs and "-24" in summed


# ── B-04: solve(pounce_scaling=) ─────────────────────────────────────────────


def _nlp():
    m = dm.Model("t")
    x = m.continuous("x", lb=0.5, ub=4)
    y = m.continuous("y", shape=(2,), lb=0, ub=4e3)
    m.minimize(1e6 * ((x - 2) ** 2 + dm.exp(y[0] / 1e3) * x) + y[1] / 1e3)
    m.subject_to(x * y[0] >= 1e3)
    return m, x, y


def test_pounce_scaling_reaches_pounce(monkeypatch):
    import pounce

    seen = []
    real = pounce.Problem

    class Spy:
        def __init__(self, *a, **k):
            self._p = real(*a, **k)

        def add_option(self, k, v):
            seen.append((k, v))
            return self._p.add_option(k, v)

        def set_problem_scaling(self, *a):
            seen.append(("scaling", a))
            return self._p.set_problem_scaling(*a)

        def __getattr__(self, n):
            return getattr(self._p, n)

    monkeypatch.setattr(pounce, "Problem", Spy)
    base, *_ = _nlp()
    r0 = base.solve(solver="pounce")
    assert not any(k == "scaling" for k, _ in seen)
    seen.clear()
    m, x, y = _nlp()
    r = m.solve(solver="pounce", pounce_scaling={"objective": 1e-6, "variables": {y: [1e-3, 1.0]}})
    assert ("nlp_scaling_method", "user-scaling") in seen
    (obj_s, x_s, g_s) = next(a for k, a in seen if k == "scaling")
    assert obj_s == 1e-6 and g_s is None
    np.testing.assert_array_equal(x_s, [1.0, 1e-3, 1.0])
    assert r.status == r0.status == "local_optimal"
    # Same point, reported in the model's own units.
    assert r.objective == pytest.approx(r0.objective, rel=1e-7)


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"objective": -1.0}, "finite and positive"),
        ({"objective": float("nan")}, "finite and positive"),
        ({"foo": 1.0}, "unknown keys"),
    ],
)
def test_pounce_scaling_refuses_bad_input(bad, match):
    m, *_ = _nlp()
    with pytest.raises(ValueError, match=match):
        m.solve(solver="pounce", pounce_scaling=bad)


def test_pounce_scaling_refuses_a_wrong_shape_and_a_foreign_variable():
    m, x, y = _nlp()
    with pytest.raises(ValueError, match="shape"):
        m.solve(solver="pounce", pounce_scaling={"variables": {y: [1.0, 2.0, 3.0]}})
    other = dm.Model("o").continuous("w")
    with pytest.raises(ValueError, match="not a variable of this model"):
        m.solve(solver="pounce", pounce_scaling={"variables": {other: 2.0}})


def test_pounce_scaling_refused_off_the_nlp_arm():
    m, *_ = _nlp()
    with pytest.raises(ValueError, match="only to solver='pounce'"):
        m.solve(pounce_scaling={"objective": 2.0})
    lp = dm.Model("lp")
    z = lp.continuous("z", lb=0, ub=1)
    lp.minimize(z)
    with pytest.raises(ValueError, match="convex interior-point engine"):
        lp.solve(solver="pounce", pounce_scaling={"objective": 2.0})
    m2, *_ = _nlp()
    with pytest.raises(ValueError, match="user-scaling"):
        m2.solve(
            solver="pounce",
            pounce_scaling={"objective": 2.0},
            pounce_options={"nlp_scaling_method": "none"},
        )


# ── C-04: the slack form is documented where a user meets the node count ────


def test_highs_options_doc_explains_the_mps_node_gap():
    from discopt.solver import solve_model

    doc = solve_model.__doc__ or ""
    assert "slack column per inequality row" in doc
    assert "to_mps()" in doc
