"""#1775 G2.7: ``gdp.cuttingplane`` (Sawaya & Grossmann 2005; Pyomo's transformation).

Before: no such transformation.

After (NC: a new, user-selected transformation; no route changes): the big-M
lowering plus cuts ``g.x >= v`` that separate the big-M relaxation's point from
the hull relaxation, where ``v`` is a certified lower bound on ``min g.x`` over a
set holding every feasible point -- the hull relaxation when every disjunct row is
linear, and the model itself (``model.clone()``) when one is nonlinear -- so every
cut holds at every feasible point, however inexact the separation. Disjuncts must
be linear or proven convex; anything else is refused, as are models with
Parameters (the cuts would go stale). Fast-API models were refused until #1854
fixed cloning their lowering.
"""

from __future__ import annotations

import itertools
import warnings

import discopt.gdp.cuttingplane as cp
import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.gdp.cuttingplane import _linear, cuttingplane
from discopt.modeling import _relax_integrality
from discopt.validation.feasibility import verify_point

pytestmark = pytest.mark.smoke


def _squares(obj):
    """[x, y <= 2] | [x, y >= 8, x <= 9], and [x + y <= 3] | [x + y >= 15] on [0, 10]^2."""
    m = dm.Model("squares")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x <= 2, y <= 2], [x >= 8, y >= 8, x <= 9]], name="a")
    m.either_or([[x + y <= 3], [x + y >= 15]], name="b")
    m.minimize(obj(x, y))
    return m


def _circles(obj):
    """Two convex disjuncts: a disc of radius 1 at (2, 2), or one at (7, 6)."""
    m = dm.Model("circles")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[(x - 2) ** 2 + (y - 2) ** 2 <= 1], [(x - 7) ** 2 + (y - 6) ** 2 <= 1]], name="c")
    m.minimize(obj(x, y))
    return m


# Optima by hand. squares: A = [0,2]^2 needs x + y <= 3 (B's x + y >= 15 is out of
# reach there); C = [8,9]x[8,10] needs x + y >= 15 (x + y <= 3 is out of reach).
#   min x - 2y: A -> (0, 2) = -4; C -> (8, 10) = -12            -> -12
#   min -x - y: A -> 3; C -> (9, 10) = -19                       -> -19
# circles: min x + y -> 4 - sqrt(2); min -x - y -> -(13 + sqrt(2)).
CASES = [
    (_squares, lambda x, y: x - 2 * y, -12.0),
    (_squares, lambda x, y: -x - y, -19.0),
    (_circles, lambda x, y: x + y, 4.0 - np.sqrt(2.0)),
    (_circles, lambda x, y: -x - y, -(13.0 + np.sqrt(2.0))),
]


def _solve(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return m.solve(time_limit=60)


@pytest.mark.parametrize("norm", [2, "inf"])
@pytest.mark.parametrize("build, obj, truth", CASES)
def test_cuts_keep_the_optimum_and_never_loosen_the_relaxation(build, obj, truth, norm):
    m = build(obj)
    out, rep = cuttingplane(m, norm=norm, max_cuts=10)
    assert rep.relaxation_before is not None and rep.relaxation_after is not None
    assert rep.relaxation_after >= rep.relaxation_before - 1e-6  # min: never looser
    r = _solve(out)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-5)


def test_the_cuts_tighten_the_big_m_relaxation():
    out, rep = cuttingplane(_squares(lambda x, y: x - 2 * y), max_cuts=10)
    assert rep.cuts >= 1
    assert rep.relaxation_before == pytest.approx(-12.08, abs=1e-4)
    assert rep.relaxation_after == pytest.approx(-12.0, abs=1e-4)  # the hull bound
    assert sum(c.name is not None and c.name.startswith("_gdp_cp_cut_") for c in out._constraints)


@pytest.mark.parametrize("build", [_squares, _circles])
def test_every_cut_holds_at_every_feasible_point(build):
    """A grid over the box; every point the declared model accepts satisfies every
    cut. The probe must have checked feasible points and cuts."""
    m = build(lambda x, y: x - 2 * y)
    _, rep = cuttingplane(m, max_cuts=10)
    assert rep.cuts >= 1
    checks = 0
    for xv, yv in itertools.product(np.linspace(0, 10, 41), repeat=2):
        if not verify_point(m, np.array([xv, yv])).ok:
            continue
        for g, v in rep.cut_rows:
            assert g["x"] * xv + g["y"] * yv >= v - 1e-7, (xv, yv, g, v)
            checks += 1
    assert checks >= 20


def test_registered_transformation():
    m = _squares(lambda x, y: -x - y)
    t = dt.create_using("gdp.cuttingplane", m, max_cuts=5)
    assert t is not m and len(m._constraints) == 2  # the source is untouched
    assert _solve(t).objective == pytest.approx(-19.0, abs=1e-5)


def test_max_cuts_zero_is_the_big_m_lowering(monkeypatch):
    def no_solve(*a, **k):
        raise AssertionError("max_cuts=0 must not solve anything")

    monkeypatch.setattr(cp, "_quiet_solve", no_solve)
    m = _squares(lambda x, y: x - 2 * y)
    out, rep = cuttingplane(m, max_cuts=0)
    assert rep.cuts == 0 and rep.stopped == "max_cuts=0 reached"
    assert rep.relaxation_before is None
    assert [str(c) for c in out._constraints] == [
        str(c) for c in reformulate_gdp(m, "big-m")._constraints
    ]


def test_a_model_without_gdp_is_returned_unchanged():
    m = dm.Model("plain")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    out, rep = cuttingplane(m)
    assert out is m and rep.cuts == 0


@pytest.mark.parametrize(
    "row",
    [
        lambda x, y: x * y <= 1,  # bilinear: not convex
        lambda x, y: x**2 + y**2 >= 4,  # convex body, >=: not a convex set
        lambda x, y: x**2 == y,  # nonlinear equality
    ],
)
def test_a_nonconvex_disjunct_is_refused(row):
    m = dm.Model("nc")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[row(x, y)], [x >= 8]])
    m.minimize(x + y)
    with pytest.raises(ValueError, match="not linear or proven convex"):
        cuttingplane(m)


def test_or_semantics_is_refused_by_the_hull_relaxation():
    m = dm.Model("or")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], semantics="or")
    m.minimize(x)
    with pytest.raises(NotImplementedError, match="OR"):
        cuttingplane(m)


def test_bad_options_are_refused():
    m = _squares(lambda x, y: x)
    with pytest.raises(ValueError, match="norm"):
        cuttingplane(m, norm=1)
    with pytest.raises(ValueError, match="max_cuts"):
        cuttingplane(m, max_cuts=-1)


def _pp():
    m = dm.Model("pp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    p = m.parameter("p", value=14.0)
    m.either_or([[x <= 2, y <= 2], [x >= 6, y >= 6]])
    m.subject_to(x + y <= p)
    m.minimize(x - 2 * y)
    return m, p


def test_a_model_with_parameters_is_refused():
    """The cuts are computed at p = 14. Unrefused, ``p.value = 20`` afterwards made
    the output report a certified optimum of -13.714, cutting off the true optimum
    -14 at (6, 10)."""
    m, p = _pp()
    with pytest.raises(ValueError, match="Parameters"):
        cuttingplane(m)


def test_a_parameter_only_in_the_objective_is_refused():
    """x* depends on the objective, so the cuts do too."""
    m = dm.Model("po")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    p = m.parameter("p", value=2.0)
    m.either_or([[x <= 2, y <= 2], [x >= 6, y >= 6]])
    m.minimize(x - p * y)
    with pytest.raises(ValueError, match="Parameters"):
        cuttingplane(m)


def test_a_fast_api_model_is_accepted_since_1854():
    # Refused while cloning a big-M-lowered fast-API model raised (#1854).
    # min X0 - 2 X1, X0 + X1 <= 15, X0 <= 2 or X0 >= 6 on [0, 10]^2: (0, 10) = -20.
    m = dm.Model("fast")
    X = m.continuous("X", shape=(2,), lb=0, ub=10)
    m.add_linear_constraints(np.array([[1.0, 1.0]]), X, "<=", np.array([15.0]))
    m.either_or([[X[0] <= 2], [X[0] >= 6]])
    m.add_linear_objective(np.array([1.0, -2.0]), X)
    out, _report = cuttingplane(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = out.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-20.0, abs=1e-6)


def _back_off_returning(monkeypatch, fake):
    """Patch the back-off solve (the second ``_quiet_solve`` of an iteration) to
    return ``fake(real result)``; the big-M relaxation solves stay real."""
    real = cp._quiet_solve
    calls = {"n": 0, "faked": []}

    def patched(m, time_limit):
        calls["n"] += 1
        r = real(m, time_limit)
        if calls["n"] % 2 == 0:
            f = fake(r)
            calls["faked"].append((r, f))
            return f
        return r

    monkeypatch.setattr(cp, "_quiet_solve", patched)
    return calls


def _as(r, **over):
    from types import SimpleNamespace

    d = {k: getattr(r, k) for k in ("status", "objective", "bound", "bound_valid")}
    d.update(over)
    return SimpleNamespace(**d)


def test_the_cut_constant_is_the_bound_not_the_objective(monkeypatch):
    """A back-off stopped at its time limit: its objective is a primal value above
    min g.x, and only the bound is a valid cut constant."""
    calls = _back_off_returning(
        monkeypatch, lambda r: _as(r, status="time_limit", objective=float(r.bound) + 1.0)
    )
    _, rep = cuttingplane(_squares(lambda x, y: x - 2 * y), max_cuts=1)
    assert calls["faked"] and rep.cuts == 1
    real, fake = calls["faked"][0]
    assert rep.cut_rows[0][1] == float(real.bound)
    assert rep.cut_rows[0][1] < fake.objective - 0.5


def test_an_uncertified_bound_adds_no_cut(monkeypatch):
    calls = _back_off_returning(monkeypatch, lambda r: _as(r, bound_valid=False))
    out, rep = cuttingplane(_squares(lambda x, y: x - 2 * y), max_cuts=3)
    assert calls["faked"]
    assert rep.cuts == 0 and rep.stopped.startswith("no certified bound")
    assert not any(c.name and c.name.startswith("_gdp_cp_cut_") for c in out._constraints)


def _with_global_row(row, nonlinear_disjuncts):
    m = dm.Model("glob")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    if nonlinear_disjuncts:
        m.either_or([[(x - 2) ** 2 + (y - 2) ** 2 <= 1], [(x - 7) ** 2 + (y - 6) ** 2 <= 1]])
    else:
        m.either_or([[x <= 2, y <= 2], [x >= 6, y >= 6]])
    if row is not None:
        m.subject_to(row(x, y))
    m.minimize(0.7 * x - 1.2 * y)
    return m


def _min_gx_bound(m, g, linear):
    """An independent certified bound on min g.x: over the hull relaxation when the
    disjuncts are linear, over the model otherwise (what the back-off must use)."""
    if linear:
        b = reformulate_gdp(m.clone(), "hull")
        _relax_integrality.relax_integrality(b)
    else:
        b = m.clone()
    b.minimize(_linear(b, ["x", "y"], g))
    r = _solve(b)
    assert r.bound_valid
    return float(r.bound)


@pytest.mark.parametrize("nonlinear_disjuncts", [False, True])
@pytest.mark.parametrize(
    "row",
    [
        lambda x, y: 0.9 * x - 0.3 * y <= 1.9,  # linear global row
        lambda x, y: 0.9 * x - 0.3 * y + 0.05 * x**2 <= 1.9,  # convex nonlinear global row
    ],
)
def test_the_back_off_keeps_the_global_rows(row, nonlinear_disjuncts):
    """Each cut constant must be a bound over a set with the global rows: without
    them it is far looser. The probe asserts it saw a cut where the rows matter."""
    m = _with_global_row(row, nonlinear_disjuncts)
    _, rep = cuttingplane(m, max_cuts=3)
    sensitive = 0
    for g, v in rep.cut_rows:
        with_rows = _min_gx_bound(m, g, not nonlinear_disjuncts)
        without = _min_gx_bound(
            _with_global_row(None, nonlinear_disjuncts), g, not nonlinear_disjuncts
        )
        assert v <= with_rows + 1e-6 * max(1.0, abs(with_rows))  # valid
        if with_rows - without > 0.1:
            sensitive += 1
            assert v > (with_rows + without) / 2, (v, with_rows, without)
    assert sensitive >= 1


def test_the_separation_nlp_is_bounded_by_time_limit(monkeypatch):
    import discopt.solvers.nlp_pounce as nlp_pounce

    seen = []
    real = nlp_pounce.solve_nlp_from_model

    def spy(m, *a, options=None, **k):
        seen.append(options)
        return real(m, *a, options=options, **k)

    monkeypatch.setattr(nlp_pounce, "solve_nlp_from_model", spy)
    cuttingplane(_squares(lambda x, y: x - 2 * y), max_cuts=2, time_limit=7.5)
    assert seen and all(o == {"max_wall_time": 7.5} for o in seen)
