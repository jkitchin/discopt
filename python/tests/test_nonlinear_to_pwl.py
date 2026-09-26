"""Tests for ``dm.nonlinear_to_pwl`` -- PWL approximation of nonlinear terms (#1482).

The certificate question is the whole point of this feature, so the tests are
organised around it:

* **Band soundness (the relaxation property).** For many functions and segment
  layouts, ``g(x) - chord(x)`` is sampled densely and must lie inside the
  per-segment band *and* under every taper row. This is the invariant that makes
  ``mode="outer"`` a relaxation.
* **No valid point cut.** With the input fixed at ``t``, the transformed model's
  ``min w`` / ``max w`` must bracket ``g(t)``.
* **Bound validity end to end.** Outer bounds never pass a brute-force optimum,
  whether or not the result certifies; certified objectives equal it.
* **Approximate mode never certifies** -- ``bound=None``, ``gap_certified=False``,
  and ``algorithm_route`` says the model solved was an approximation.
* **Tripwire.** A deliberately unsound band is caught, not certified.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling._pwl_transform import (
    PWLTransformError,
    _segment_band,
    _Univariate,
)

# ---------------------------------------------------------------------------
# Band soundness
# ---------------------------------------------------------------------------

_FUNCS = [
    ("exp", lambda x: dm.exp(x), np.exp, (-2.0, 3.0)),
    ("log", lambda x: dm.log(x), np.log, (0.1, 5.0)),
    ("sqrt_at_zero", lambda x: dm.sqrt(x), np.sqrt, (0.0, 4.0)),
    ("cubic", lambda x: x**3 - 2 * x, lambda v: v**3 - 2 * v, (-2.0, 2.0)),
    ("sin_plus_sq", lambda x: dm.sin(3 * x) + x**2, lambda v: np.sin(3 * v) + v**2, (-2.0, 2.0)),
    ("tanh", lambda x: dm.tanh(2 * x), lambda v: np.tanh(2 * v), (-3.0, 3.0)),
    ("rational", lambda x: 1 / (1 + x**2), lambda v: 1 / (1 + v**2), (-3.0, 3.0)),
    ("x_exp", lambda x: x * dm.exp(-x), lambda v: v * np.exp(-v), (0.0, 6.0)),
    ("atan", lambda x: dm.atan(x), np.arctan, (-4.0, 4.0)),
    ("quartic", lambda x: x**4 - 3 * x**2 + x, lambda v: v**4 - 3 * v**2 + v, (-2.0, 2.0)),
]


@pytest.mark.smoke
@pytest.mark.parametrize("fname,build,npf,dom", _FUNCS, ids=[f[0] for f in _FUNCS])
def test_band_encloses_g_minus_chord_everywhere(fname, build, npf, dom):
    lo, hi = dom
    m = dm.Model()
    x = m.continuous("x", lb=lo, ub=hi)
    u = _Univariate(build(x), x, lo, hi)
    rng = np.random.default_rng(abs(hash(fname)) % 2**32)
    checks = 0
    for n_seg in (1, 2, 3, 5, 8, 17):
        inner = np.sort(rng.uniform(lo, hi, n_seg - 1))
        bps = np.concatenate([[lo], inner, [hi]])
        pts = [u.point(float(t)) for t in bps]
        for i in range(n_seg):
            a, b = float(bps[i]), float(bps[i + 1])
            if b - a < 1e-9:
                continue
            bd = _segment_band(u, a, b, pts[i], pts[i + 1])
            assert bd is not None, (fname, a, b)
            va, vb = pts[i][0], pts[i + 1][0]
            xs = np.linspace(a, b, 401)
            with np.errstate(divide="ignore", invalid="ignore"):
                e = npf(xs) - (va + (vb - va) * (xs - a) / (b - a))
            tol = 1e-12 * max(1.0, float(np.max(np.abs(npf(xs)))))
            assert np.all(e >= bd.lo - tol) and np.all(e <= bd.hi + tol), (fname, a, b)
            if bd.taper is not None:
                ua, ka, ub_, kb, la, ja, lb_, jb = bd.taper
                r = (xs - a) / (b - a)
                assert np.all(e <= ua + ka * r + tol)
                assert np.all(e <= ub_ + kb * (1 - r) + tol)
                assert np.all(e >= la + ja * r - tol)
                assert np.all(e >= lb_ + jb * (1 - r) - tol)
            checks += 1
    assert checks >= 30


@pytest.mark.smoke
def test_band_shrinks_quadratically_for_smooth_terms():
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    u = _Univariate(dm.exp(x), x, 0.0, 1.0)
    widths = []
    for h in (0.2, 0.1, 0.05):
        a, b = 0.4, 0.4 + h
        bd = _segment_band(u, a, b, u.point(a), u.point(b))
        widths.append(bd.hi - bd.lo)
    # Halving h should cut the band by ~4x (O(h^2)); demand at least 3x.
    assert widths[0] / widths[1] > 3.0 and widths[1] / widths[2] > 3.0


# ---------------------------------------------------------------------------
# Transformation structure and refusals
# ---------------------------------------------------------------------------


def _model_quartic_exp():
    m = dm.Model("qe")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-1, ub=3)
    m.minimize(x**4 - 3 * x**2 + x + dm.exp(0.5 * y) - y)
    m.subject_to(x + y >= 0.5, name="link")
    return m, x, y


@pytest.mark.smoke
def test_terms_found_and_original_untouched():
    m, x, y = _model_quartic_exp()
    n_vars, n_cons, obj = len(m._variables), len(m._constraints), m._objective.expression
    t = dm.nonlinear_to_pwl(m, segments=4)
    assert isinstance(t, dm.PWLTransformation)
    assert [tt.input for tt in t.terms] == ["x", "y"]
    assert all(tt.max_band is not None and tt.max_band > 0 for tt in t.terms)
    assert t.terms[0].breakpoints == (-2.0, -1.0, 0.0, 1.0, 2.0)
    assert t.fully_linear and t.skipped == []
    # The original model is never mutated.
    assert (len(m._variables), len(m._constraints)) == (n_vars, n_cons)
    assert m._objective.expression is obj


@pytest.mark.smoke
def test_skips_are_reported_and_multivariate_terms_stay_exact():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=2)
    free = m.continuous("free")  # unbounded
    fixed = m.continuous("fixed", lb=1, ub=1)
    p = m.parameter("p", 2.0)
    m.minimize(x * y + dm.exp(x) + p * dm.sin(y) + free**2 + dm.exp(fixed))
    t = dm.nonlinear_to_pwl(m)
    assert [tt.input for tt in t.terms] == ["x"]
    reasons = " | ".join(s.reason for s in t.skipped)
    assert "Parameter" in reasons and "unbounded" in reasons and "fixed" in reasons
    assert not t.fully_linear  # x*y is left as written


@pytest.mark.smoke
def test_non_algebraic_models_are_refused():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=2)
    m.minimize(dm.exp(x))
    m.either_or([[x <= 0.5], [x >= 1.5]])
    with pytest.raises(PWLTransformError, match="algebraic"):
        dm.nonlinear_to_pwl(m)
    m2 = dm.Model()
    z = m2.continuous("z", lb=0, ub=2)
    m2.minimize(m2.piecewise(z, [0, 1, 2], [0, 1, 0], method="sos2") + dm.exp(z))
    with pytest.raises(PWLTransformError, match="sos2"):
        dm.nonlinear_to_pwl(m2)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kw,exc",
    [
        ({"mode": "exact"}, ValueError),
        ({"segments": 0}, ValueError),
        ({"segments": 2.5}, TypeError),
    ],
)
def test_bad_arguments(kw, exc):
    m, *_ = _model_quartic_exp()
    with pytest.raises(exc):
        dm.nonlinear_to_pwl(m, **kw)


# ---------------------------------------------------------------------------
# No valid point cut: the transformed rows bracket g(t) at fixed inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fname,build,npf,dom", _FUNCS[:6], ids=[f[0] for f in _FUNCS[:6]])
def test_outer_rows_never_cut_a_graph_point(fname, build, npf, dom):
    lo, hi = dom
    m = dm.Model()
    x = m.continuous("x", lb=lo, ub=hi)
    m.minimize(build(x))
    t = dm.nonlinear_to_pwl(m, segments=5)
    assert len(t.terms) == 1
    tm = t.model
    xv = next(v for v in tm._variables if v.name == "x")
    wv = next(v for v in tm._variables if v.name.endswith("_w"))
    checked = 0
    for s in np.linspace(lo, hi, 9)[1:-1].tolist() + [lo, hi]:
        xv.lb = np.float64(s)
        xv.ub = np.float64(s)
        vals, bounds = [], []
        for sense in ("min", "max"):
            tm.minimize(wv) if sense == "min" else tm.maximize(wv)
            r = tm.solve(time_limit=30)
            # Status-independent: a valid bound on min w (max w) that passed g(t)
            # would prove the graph point (t, g(t)) was cut off.
            assert r.x is not None and r.bound is not None, (r.status, s)
            vals.append(float(r.value(wv)))
            bounds.append(float(r.bound))
        g = float(npf(s))
        tol = 1e-7 * max(1.0, abs(g))
        assert bounds[0] <= g + tol and bounds[1] >= g - tol, (fname, s, bounds, g)
        assert vals[0] <= g + tol and vals[1] >= g - tol, (fname, s, vals, g)
        assert vals[1] - vals[0] <= t.terms[0].max_band + 1e-7
        checked += 1
    assert checked == 9


# ---------------------------------------------------------------------------
# End to end: outer bounds are valid; certified results match brute force
# ---------------------------------------------------------------------------


def _brute_quartic_exp():
    xs = np.linspace(-2, 2, 4001)
    best = math.inf
    for xv in xs:
        ylo = max(-1.0, 0.5 - xv)
        ys = np.linspace(ylo, 3.0, 801)
        f = xv**4 - 3 * xv**2 + xv + np.exp(0.5 * ys) - ys
        best = min(best, float(np.min(f)))
    return best


def test_outer_certifies_the_global_optimum():
    m, x, y = _model_quartic_exp()
    r = dm.nonlinear_to_pwl(m).solve(time_limit=120)
    assert r.status == "optimal" and r.gap_certified and r.bound_valid
    want = _brute_quartic_exp()
    assert r.bound <= want + 1e-7
    assert r.objective == pytest.approx(want, abs=1e-3)
    # The incumbent is a point of the ORIGINAL model, keyed by its variables.
    xv, yv = float(r.value(x)), float(r.value(y))
    assert xv + yv >= 0.5 - 1e-6
    assert r.objective == pytest.approx(xv**4 - 3 * xv**2 + xv + math.exp(0.5 * yv) - yv)
    assert "outer" in r.algorithm_route


def test_outer_bound_is_valid_even_when_not_certified():
    m, *_ = _model_quartic_exp()
    r = dm.nonlinear_to_pwl(m, segments=2).solve(max_rounds=1, polish=False)
    want = _brute_quartic_exp()
    assert r.bound is not None and r.bound <= want + 1e-7
    assert not r.gap_certified
    assert r.status in ("feasible", "iteration_limit")


def test_outer_handles_maximize_and_constraints():
    # max sin(x) + cos(y)  s.t.  x**2 + y**2 <= 2 (multivariate row kept exact? no:
    # x**2 and y**2 are separate univariate terms, so the row is linearised too).
    m = dm.Model()
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.maximize(dm.sin(2 * x) + dm.cos(y) - 0.1 * x)
    m.subject_to(x**2 + y**2 <= 2)
    t = dm.nonlinear_to_pwl(m)
    assert {tt.input for tt in t.terms} == {"x", "y"} and len(t.terms) == 4
    r = t.solve(time_limit=120)
    g = np.linspace(-2, 2, 2001)
    X, Y = np.meshgrid(g, g)
    F = np.where(X**2 + Y**2 <= 2, np.sin(2 * X) + np.cos(Y) - 0.1 * X, -np.inf)
    want = float(F.max())
    assert r.bound is not None and r.bound >= want - 1e-7  # an UPPER bound for max
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(want, abs=2e-3)


def test_outer_with_integer_inputs_and_remaining_bilinear_term():
    m = dm.Model()
    n = m.integer("n", lb=0, ub=5)
    x = m.continuous("x", lb=0.5, ub=3)
    m.minimize((n - 2.3) ** 2 + x * n / 4 + 1 / x)
    t = dm.nonlinear_to_pwl(m)
    assert not t.fully_linear  # x*n stays exact
    r = t.solve(time_limit=120)
    best = min(
        (nv - 2.3) ** 2 + xv * nv / 4 + 1 / xv
        for nv in range(6)
        for xv in np.linspace(0.5, 3, 25001)
    )
    assert r.status == "optimal" and r.gap_certified
    assert r.bound <= best + 1e-7
    assert r.objective == pytest.approx(best, abs=1e-3)


def test_outer_infeasible_relaxation_certifies_infeasibility():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    m.subject_to(dm.exp(x) <= 0.5)  # exp >= 1 on [0, 1]
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert r.status == "infeasible" and r.gap_certified


# ---------------------------------------------------------------------------
# Approximate mode: never a certificate
# ---------------------------------------------------------------------------


def test_approximate_mode_never_claims_a_bound():
    m, x, y = _model_quartic_exp()
    t = dm.nonlinear_to_pwl(m, mode="approximate", segments=4)
    r = t.solve(time_limit=60)
    assert r.bound is None and not r.gap_certified and not r.bound_valid
    assert r.status in ("feasible", "local_infeasible")
    assert "APPROXIMATION" in r.algorithm_route
    if r.status == "feasible":
        xv, yv = float(r.value(x)), float(r.value(y))
        assert r.objective == pytest.approx(xv**4 - 3 * xv**2 + xv + math.exp(0.5 * yv) - yv)


def test_approximate_mode_is_uncertified_even_when_exact_at_feasible_points():
    # An integer input gets a breakpoint at every integer, so the interpolant is
    # exact at every feasible point -- and still no certificate is claimed: the
    # mode's contract is about what was solved, not about luck.
    m = dm.Model()
    n = m.integer("n", lb=0, ub=6)
    m.minimize((n - 3.4) ** 2)
    r = dm.nonlinear_to_pwl(m, mode="approximate").solve(time_limit=30)
    assert r.status == "feasible" and not r.gap_certified and r.bound is None
    assert float(r.value(n)) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Tripwire: an unsound band is caught rather than certified
# ---------------------------------------------------------------------------


def test_band_self_check_refuses_an_unsound_band(monkeypatch):
    """A band that excludes the graph is caught at build time, before any solve.

    This is the guard for the case the bound tripwire cannot see: a band shifted
    off the graph can make the relaxation *infeasible*, which would otherwise be
    reported as a certified infeasibility of the original.
    """
    import discopt.modeling._pwl_transform as mod

    real = mod._segment_band

    def shifted(u, a, b, ga, gb):
        bd = real(u, a, b, ga, gb)
        return mod._SegmentBand(bd.hi + 5.0, bd.hi + 6.0, None)

    monkeypatch.setattr(mod, "_segment_band", shifted)
    m, *_ = _model_quartic_exp()
    with pytest.raises(AssertionError, match="excludes the term's value"):
        dm.nonlinear_to_pwl(m)


def test_tripwire_catches_a_bound_passing_a_verified_point(monkeypatch):
    """With the build-time check disabled, an unsound band that still admits some
    points is caught at solve time: its bound passes a verified objective."""
    import discopt.modeling._pwl_transform as mod

    real = mod._segment_band

    def lifted(u, a, b, ga, gb):
        bd = real(u, a, b, ga, gb)
        return mod._SegmentBand(bd.lo + 0.75, bd.hi + 0.75, None)

    monkeypatch.setattr(mod, "_segment_band", lifted)
    monkeypatch.setattr(mod, "_check_band", lambda *a, **k: None)
    m, *_ = _model_quartic_exp()
    with pytest.raises(AssertionError, match="not a relaxation"):
        dm.nonlinear_to_pwl(m).solve(time_limit=60)


def test_refinement_growth_is_bounded():
    """Uniformly wide bands must not double the table every round (measured: the
    valve-point dispatch went 9 -> 849 breakpoints in 8 rounds and was OOM-killed
    before the per-round split was capped); the per-term cap holds."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=20.0)
    m.minimize(dm.abs(dm.sin(3 * x)) + 0.01 * x)
    t = dm.nonlinear_to_pwl(m, segments=8, max_breakpoints=40)
    counts = [len(t.terms[0].breakpoints)]
    r = t.solve(time_limit=60, max_rounds=12)
    counts.append(len(t.terms[0].breakpoints))
    assert counts[-1] <= 40
    history = r.solver_stats["nonlinear_to_pwl"]["history"]
    per_round = [h["breakpoints"][0] for h in history]
    # Never more than ~10% of the segments plus two new points per round.
    for a, b in zip(per_round, per_round[1:]):
        assert b - a <= max(3, math.ceil(0.1 * (a - 1)) + 2), per_round
    assert r.bound is not None and r.bound <= 1e-9  # min is 0 at x = 0


@pytest.mark.smoke
def test_max_breakpoints_must_hold_the_initial_grid():
    m, *_ = _model_quartic_exp()
    with pytest.raises(ValueError, match="max_breakpoints"):
        dm.nonlinear_to_pwl(m, segments=8, max_breakpoints=5)


def test_valve_point_dispatch_bound_brackets_the_literature_optimum():
    """``examples.example_valve_point_dispatch`` (Walters & Sheble 1993): whatever
    the budget allows, the outer bound is below the known optimum 8234.07 and the
    verified cost is above it; the approximate mode's verified cost is above it too."""
    import contextlib
    import io

    from discopt.modeling import examples

    with contextlib.redirect_stdout(io.StringIO()):
        m = examples.example_valve_point_dispatch()
    opt = 8234.07  # published to 2 decimals; allow that rounding
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30, max_rounds=6)
    assert r.bound is not None and r.bound <= opt + 0.01
    assert r.objective is not None and r.objective >= opt - 0.01
    p = np.asarray(r.x["P"])
    assert abs(p.sum() - 850.0) <= 1e-6
    ra = dm.nonlinear_to_pwl(m, mode="approximate").solve(time_limit=30)
    assert ra.bound is None and not ra.gap_certified
    if ra.objective is not None:
        assert ra.objective >= opt - 0.01
