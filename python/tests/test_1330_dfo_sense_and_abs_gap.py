"""#1330: a route's ``objective`` is the USER's objective, and no option is inert.

Two defects on the derivative-free routes and their neighbours.

1. Every evaluator in discopt minimizes, so a ``maximize`` model is handed
   ``-f`` internally. ``solve_direct`` and ``solve_surrogate`` reported that
   internal value verbatim: ``objective = -5.0`` at the point where the user's
   objective is ``+5.0``. The invariant is that ``result.objective`` equals the
   user objective evaluated at ``result.x``, on every route.

2. #1323 promised ``abs_gap_tolerance`` is "honoured or declared, never
   silently inert". Four routes were still silent: ``solver='direct'`` and
   ``solver='surrogate'`` (which warn about ``gap_tolerance`` but had no dual
   bound for either criterion), the deprecated ``gdp_method='oa'`` route, and
   the native Rust spatial kernel, which takes ``min(gap_tolerance,
   abs_gap_tolerance)`` and so silently drops a LOOSER absolute tolerance.
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest


def _concave_max(name="i1330_max"):
    """``max -(x-1)^2 + 5`` on [0, 4]: optimum ``+5`` at ``x = 1``."""
    m = dm.Model(name)
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.maximize(-((x - 1.0) ** 2) + 5.0)
    return m


def _convex_min(name="i1330_min"):
    """The minimize control: same surface, opposite sense."""
    m = dm.Model(name)
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.minimize((x - 1.0) ** 2 + 5.0)
    return m


def _x_value(result):
    if result.x is None:
        return None
    if isinstance(result.x, dict):
        return float(np.ravel(list(result.x.values())[0])[0])
    return float(np.ravel(result.x)[0])


_ROUTE_KWARGS = {
    "default": {},
    "direct": {"solver": "direct", "max_evals": 200},
    # A model-based search is expensive per evaluation; this is enough to land
    # on the optimum and keep the test near a second.
    "surrogate": {"solver": "surrogate", "max_evals": 12, "n_initial": 5, "local_refine": False},
}


# ── 1. the reported objective is the user's objective at the returned point ──


@pytest.mark.correctness
@pytest.mark.parametrize("route", sorted(_ROUTE_KWARGS))
def test_reported_objective_equals_user_objective_at_x_on_maximize(route):
    """``objective == f(x)``. Pre-fix, direct and surrogate returned ``-f(x)``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _concave_max(f"i1330_max_{route}").solve(**_ROUTE_KWARGS[route])

    xv = _x_value(r)
    assert xv is not None, f"{route} returned no point to check the objective against"
    assert r.objective is not None
    f_at_x = -((xv - 1.0) ** 2) + 5.0
    assert r.objective == pytest.approx(f_at_x, abs=1e-4), (
        f"solver={route!r} reported objective={r.objective} at x={xv}, "
        f"where the user objective is {f_at_x}"
    )
    # ...and it is the sign, not just the arithmetic: a maximize optimum here is
    # positive, so a sign error cannot hide behind a near-zero value.
    assert r.objective > 0.0


@pytest.mark.correctness
@pytest.mark.parametrize("route", sorted(_ROUTE_KWARGS))
def test_reported_objective_equals_user_objective_at_x_on_minimize(route):
    """The control: the minimize sense was never wrong and must stay right."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _convex_min(f"i1330_min_{route}").solve(**_ROUTE_KWARGS[route])

    xv = _x_value(r)
    assert xv is not None
    assert r.objective == pytest.approx((xv - 1.0) ** 2 + 5.0, abs=1e-4)


@pytest.mark.unit
def test_reported_objective_helper_flips_only_maximize():
    from discopt.solvers._dfo_common import reported_objective

    assert reported_objective(_concave_max("i1330_h_max"), -5.0) == 5.0
    assert reported_objective(_convex_min("i1330_h_min"), 5.0) == 5.0

    no_obj = dm.Model("i1330_h_none")
    no_obj.continuous("x", lb=0.0, ub=1.0)
    assert reported_objective(no_obj, 3.0) == 3.0


# ── 2. abs_gap_tolerance is declared wherever it cannot be honoured ──────────


def _abs_gap_warnings(recorded):
    return [str(w.message) for w in recorded if "abs_gap_tolerance" in str(w.message)]


@pytest.mark.correctness
@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": "direct", "max_evals": 50},
        {"solver": "surrogate", "max_evals": 8, "n_initial": 4, "local_refine": False},
    ],
    ids=["direct", "surrogate"],
)
def test_dfo_routes_declare_an_ignored_abs_gap_tolerance(kwargs):
    """Neither route has a dual bound, so neither gap criterion can be met."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _convex_min("i1330_dfoabs").solve(abs_gap_tolerance=1e4, **kwargs)
    said = _abs_gap_warnings(w)
    assert said, f"{kwargs['solver']} accepted abs_gap_tolerance and said nothing"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _convex_min("i1330_dfoquiet").solve(**kwargs)
    assert not _abs_gap_warnings(w), "unset must stay silent; None is a default, not a request"


@pytest.mark.correctness
def test_deprecated_gdp_oa_route_declares_an_ignored_abs_gap_tolerance():
    """``gdp_method='oa'`` calls ``solve_mip_nlp``, which takes no absolute gap."""

    def gdp(name):
        m = dm.Model(name)
        x = m.continuous("x", lb=0.0, ub=10.0)
        m.minimize(x)
        m.either_or([[x <= 3], [x >= 7]], name="mode")
        return m

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gdp("i1330_oa").solve(gdp_method="oa", abs_gap_tolerance=1e4)
    assert _abs_gap_warnings(w), "gdp_method='oa' dropped abs_gap_tolerance silently"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gdp("i1330_oa_quiet").solve(gdp_method="oa")
    assert not _abs_gap_warnings(w)


def _native_scope_model(name="i1330_native"):
    """Four scalar variables with two bilinear terms: inside the kernel's subset."""
    m = dm.Model(name)
    xs = [m.continuous(f"x{i}", lb=0.0, ub=5.0) for i in range(4)]
    m.subject_to(xs[0] * xs[1] + xs[2] * xs[3] >= 6.0)
    m.minimize(sum(xs))
    return m


@pytest.mark.correctness
def test_native_spatial_kernel_declares_an_abs_gap_it_declines_to_loosen():
    """``min(gap_tolerance, abs_gap_tolerance)`` drops a looser absolute arm.

    Declining to loosen is the sound direction and stays. What must not stay is
    the silence: the caller asked for 1e6 and got the 1e-4 default, exploring
    exactly as many nodes as with no tolerance at all.
    """
    import discopt.solver as solver_mod

    used_native = []
    original = solver_mod._try_native_spatial_kernel

    def _spy(*args, **kwargs):
        result = original(*args, **kwargs)
        used_native.append(result is not None)
        return result

    solver_mod._try_native_spatial_kernel = _spy
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _native_scope_model("i1330_native_loose").solve(abs_gap_tolerance=1e6)
        loose = _abs_gap_warnings(w)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _native_scope_model("i1330_native_tight").solve(abs_gap_tolerance=1e-9)
        tight = _abs_gap_warnings(w)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _native_scope_model("i1330_native_unset").solve()
        unset = _abs_gap_warnings(w)
    finally:
        solver_mod._try_native_spatial_kernel = original

    assert used_native and all(used_native), (
        "the model must reach the native kernel or this test proves nothing"
    )
    assert loose, "a LOOSER abs_gap_tolerance was dropped silently"
    assert not tight, "a TIGHTER abs_gap_tolerance is honoured, so it must not warn"
    assert not unset, "an unset abs_gap_tolerance must not warn"


@pytest.mark.unit
def test_not_loosened_helper_fires_only_on_a_loosening():
    from discopt.solver import _warn_abs_gap_not_loosened

    fired = 0
    for abs_tol, expect in ((None, False), (1e-9, False), (1e-4, False), (1e6, True)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_abs_gap_not_loosened(abs_tol, 1e-4)
        assert bool(w) is expect, f"abs_gap_tolerance={abs_tol!r} warned={bool(w)}"
        fired += 1
    assert fired == 4
