"""#1323: ``abs_gap_tolerance`` must never be silently inert.

#1315 connected the option to the Rust MILP engine. Two routes still dropped it
without a word, and six more were never consulted at all. None of this produced
a false certificate -- the bounds were valid throughout -- but a stopping
criterion the caller set and the solver ignored is an inert option, and an inert
option that says nothing is worse than one that refuses (CLAUDE.md §3).

So every route now either honours it or says it cannot:

* the #698 re-entry run of the Rust MILP engine honours it (run 1 always did);
* the MIP-NLP family declares it in ``_MIP_NLP_IGNORED_OPTIONS``, which makes
  the #1059 auto-route decline the model -- handing it to the spatial path,
  which does honour the option -- and makes an explicit ``solver="mip-nlp"``
  warn;
* lp_spatial, gp, gp-minlp, signomial, benders/lagrangian and GDPopt-LOA warn.
"""

import contextlib
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import (
    _MIP_NLP_IGNORED_OPTIONS,
    _mip_nlp_ignored_options,
    _warn_abs_gap_ignored,
    solve_model,
)

# ── 1. the Rust MILP engine's re-entry run ───────────────────────────────


def _knapsackish(n=12, seed=7):
    """A small correlated knapsack: integral, solvable, and MILP-routed."""
    rng = np.random.default_rng(seed)
    m = dm.Model("i1323_milp")
    y = m.binary("y", shape=(n,))
    w = rng.integers(3, 40, size=n).astype(float)
    p = w + 5.0
    m.subject_to(dm.sum(w[i] * y[i] for i in range(n)) <= float(w.sum()) * 0.5)
    m.maximize(dm.sum(p[i] * y[i] for i in range(n)))
    return m


@pytest.mark.smoke
def test_the_reentry_run_of_the_rust_milp_engine_gets_abs_gap_tol():
    """``_solve_milp_simplex`` calls the Rust driver twice. Run 1 passed
    ``abs_gap_tol``; run 2 did not, so a re-entry spent its whole remaining
    budget on a tree the caller's absolute criterion had already closed.

    The first call's status is forced to ``feasible`` so the re-entry fires
    deterministically instead of depending on a wall clock.
    """
    import discopt._rust as _rust

    real = _rust.solve_milp_csc_py
    calls: list[dict] = []

    def recording(*args, **kwargs):
        calls.append(dict(kwargs))
        out = real(*args, **kwargs)
        if len(calls) == 1 and out[0] == "optimal":
            # Uncertified feasible -> the #698 re-entry path, with the same point.
            return ("feasible", out[1], out[2], -np.inf, out[4], out[5])
        return out

    m = _knapsackish()
    try:
        _rust.solve_milp_csc_py = recording
        solve_model(
            m, nlp_solver="simplex", time_limit=20.0, abs_gap_tolerance=0.5, verify_incumbent=False
        )
    finally:
        _rust.solve_milp_csc_py = real

    # Only the branch-and-bound calls matter: the root-relaxation probe passes no
    # integer indices and is a pure LP, where an absolute MIP gap means nothing.
    bb_calls = [c for c in calls if "initial_incumbent" in c or "abs_gap_tol" in c]
    assert len(bb_calls) >= 2, f"the re-entry never fired; recorded {len(calls)} driver calls"
    for i, c in enumerate(bb_calls[:2]):
        assert "abs_gap_tol" in c, f"driver call {i} was made without abs_gap_tol"
        assert c["abs_gap_tol"] == pytest.approx(0.5)


# ── 2. the MIP-NLP family declares it instead of dropping it ─────────────


@pytest.mark.smoke
def test_abs_gap_tolerance_is_declared_unhonoured_by_the_mip_nlp_family():
    """Every method in the family converges on the relative gap alone."""
    names = [name for name, _pred in _MIP_NLP_IGNORED_OPTIONS]
    assert "abs_gap_tolerance" in names
    # And the predicate treats "unset" as no preference, so a default solve is
    # still routable.
    assert _mip_nlp_ignored_options({"abs_gap_tolerance": None}) == []
    assert _mip_nlp_ignored_options({"abs_gap_tolerance": 1e4}) == ["abs_gap_tolerance"]


def _convex_minlp():
    """A small convex MINLP: the class the #1059 auto-route serves."""
    m = dm.Model("i1323_cvx")
    y = m.binary("y", shape=(3,))
    x = m.continuous("x", shape=(3,), lb=0.0, ub=4.0)
    for i in range(3):
        m.subject_to(x[i] <= 4.0 * y[i])
    m.subject_to(dm.sum(y[i] for i in range(3)) <= 2)
    m.minimize(dm.sum((x[i] - 2.0) ** 2 + y[i] for i in range(3)))
    return m


@pytest.mark.smoke
def test_the_auto_route_declines_a_solve_that_set_abs_gap_tolerance(monkeypatch):
    """The table drives two consumers; this pins the other one. The #1059
    auto-route refuses to fire when the caller set anything the family drops, so
    the model goes to the spatial path -- which honours the option (the issue
    measured 3 nodes there against 241 on the route).

    Asserted on the TOP-LEVEL call's decision: a solve makes nested
    ``solve_model`` calls for its sub-problems, and those pass no
    ``abs_gap_tolerance`` of their own.
    """
    import discopt.solver as solver_mod

    seen: list[list[str]] = []
    real = solver_mod._mip_nlp_ignored_options

    def spy(values):
        out = real(values)
        seen.append(list(out))
        return out

    monkeypatch.setattr(solver_mod, "_mip_nlp_ignored_options", spy)
    solve_model(_convex_minlp(), time_limit=20.0, abs_gap_tolerance=1e4)

    assert seen, "the auto-route decision never ran"
    assert seen[0] == ["abs_gap_tolerance"], (
        f"the top-level auto-route did not see the option the caller set; decisions were {seen}"
    )


@pytest.mark.smoke
def test_the_auto_route_still_fires_when_nothing_was_set(monkeypatch):
    """No false positive: an unset ``abs_gap_tolerance`` is not caller intent."""
    import discopt.solver as solver_mod

    seen: list[list[str]] = []
    real = solver_mod._mip_nlp_ignored_options
    monkeypatch.setattr(
        solver_mod, "_mip_nlp_ignored_options", lambda v: seen.append(list(real(v))) or real(v)
    )
    solve_model(_convex_minlp(), time_limit=20.0)
    assert seen and seen[0] == []


@pytest.mark.smoke
def test_an_explicit_mip_nlp_solve_warns_about_abs_gap_tolerance():
    with pytest.warns(UserWarning, match="abs_gap_tolerance"):
        solve_model(
            _convex_minlp(),
            solver="mip-nlp",
            time_limit=20.0,
            abs_gap_tolerance=1e4,
        )


# ── 3. the routes that take only gap_tolerance ───────────────────────────


@pytest.mark.smoke
def test_the_shared_notice_is_silent_when_the_caller_set_nothing():
    """``None`` means "use the default", not a request; warning on it would fire
    on every solve and train people to ignore the message."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_abs_gap_ignored("A route", None)


@pytest.mark.smoke
def test_the_shared_notice_names_the_route_and_the_value():
    with pytest.warns(UserWarning, match=r"A route ignores abs_gap_tolerance=0\.25"):
        _warn_abs_gap_ignored("A route", 0.25)


def _tiny_gp():
    m = dm.Model("i1323_gp")
    x = m.continuous("x", lb=0.1, ub=10.0)
    y = m.continuous("y", lb=0.1, ub=10.0)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y)
    return m


@pytest.mark.smoke
def test_the_gp_fast_path_lists_abs_gap_tolerance_as_ignored():
    with pytest.warns(UserWarning, match="abs_gap_tolerance"):
        solve_model(_tiny_gp(), solver="gp", time_limit=20.0, abs_gap_tolerance=1e-3)


@pytest.mark.smoke
def test_the_lp_spatial_engine_warns():
    m = dm.Model("i1323_lps")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.subject_to(x * y <= 4.0)
    m.minimize(-(x + y))
    with pytest.warns(UserWarning, match="lp_spatial.*abs_gap_tolerance"):
        with contextlib.suppress(Exception):
            solve_model(m, time_limit=10.0, abs_gap_tolerance=1e-3, lp_spatial=True)


@pytest.mark.smoke
def test_the_decomposition_routes_warn():
    m = dm.Model("i1323_decomp")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=5.0)
    y = m.binary("y", shape=(4,))
    for i in range(4):
        m.subject_to(x[i] <= 5.0 * y[i])
    m.subject_to(dm.sum(x[i] for i in range(4)) <= 8.0)
    m.minimize(dm.sum(-x[i] + y[i] for i in range(4)))
    with pytest.warns(UserWarning, match="benders.*abs_gap_tolerance"):
        with contextlib.suppress(Exception):
            solve_model(m, time_limit=10.0, abs_gap_tolerance=1e-3, decomposition="benders")


@pytest.mark.smoke
def test_the_gdpopt_loa_route_warns():
    m = dm.Model("i1323_loa")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.binary("y")
    m.subject_to(x <= 5.0 * y)
    m.minimize((x - 2.0) ** 2 + y)
    with pytest.warns(UserWarning, match="LOA.*abs_gap_tolerance"):
        with contextlib.suppress(Exception):
            solve_model(m, time_limit=10.0, abs_gap_tolerance=1e-3, gdp_method="loa")


@pytest.mark.smoke
def test_a_default_solve_with_abs_gap_tolerance_stays_quiet():
    """No false positives: the default engine honours the option, so nothing on
    that path may warn about it."""
    m = dm.Model("i1323_default")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.integer("y", lb=0, ub=4)
    m.subject_to(x + y <= 5.0)
    m.minimize(-(x + 2 * y))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = solve_model(m, time_limit=20.0, abs_gap_tolerance=1e-6)
    assert r.status in ("optimal", "feasible")
    assert not [w for w in caught if "abs_gap_tolerance" in str(w.message)]
