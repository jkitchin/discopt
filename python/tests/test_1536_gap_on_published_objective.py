"""#1536: a certificate is a statement about the PUBLISHED (objective, bound) pair.

The HiGHS MILP route handed HiGHS a constant-free objective (``offset_ = 0``) with
``mip_rel_gap = gap_tolerance``. Under the change of variables ``x = y - 1e6`` the
constant-free part ``c . y`` is ~1e6 at every feasible point, so HiGHS stopped on a
gap of 35.7 / 1e6 and discopt published ``objective=-1, bound=-36.67`` as
``gap_certified=True`` for a model whose optimum is -36.

Two fixes, each tested on its own:

* the route gives HiGHS the objective constant, so its stopping rule measures the
  objective discopt publishes (``test_shifted_milp_*``, ``test_translation_panel``);
* ``solve_model`` refuses to publish a certificate whose own pair fails the
  abs-OR-rel test, whatever the route did (``test_guard_*``), so a route whose
  engine applies its stopping rule to a transformed objective cannot leak one.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from scipy.optimize import linprog


@pytest.fixture(autouse=True)
def _highs_milp_route(monkeypatch):
    """These tests pin the HiGHS MILP route's own mechanics. Since 2026-11-09 the
    pure-MILP default is the Rust engine on every install (PR #2196), so they name
    HiGHS explicitly, the optional ``discopt[highs]`` extra."""
    pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")


OFF = 1e6


def _data(seed: int):
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, size=(3, 5)).astype(float)
    b = rng.integers(2, 10, size=3).astype(float)
    c = rng.integers(-5, 6, size=5).astype(float)
    return A, b, c


def _build(seed: int, off: float, sense: str = "min") -> dm.Model:
    """3 integers in [0, 4], 2 continuous in [0, 5], written as ``y - off``."""
    A, b, c = _data(seed)
    m = dm.Model(f"t{seed}")
    ys = [m.integer(f"i{k}", lb=off, ub=4 + off) for k in range(3)] + [
        m.continuous(f"c{k}", lb=off, ub=5 + off) for k in range(2)
    ]
    x = [v - off for v in ys]
    for r in range(3):
        m.subject_to(sum(A[r, j] * x[j] for j in range(5)) <= b[r])
    obj = sum(c[j] * x[j] for j in range(5))
    if sense == "min":
        m.minimize(obj)
    else:
        m.maximize(-obj)
    return m


def _oracle(seed: int) -> float:
    """Exact minimum of ``c . x``: enumerate the integers, LP over the continuous."""
    A, b, c = _data(seed)
    best = np.inf
    for ii in itertools.product(range(5), repeat=3):
        lp = linprog(
            c[3:],
            A_ub=A[:, 3:],
            b_ub=b - A[:, :3] @ np.array(ii, dtype=float),
            bounds=[(0, 5)] * 2,
            method="highs",
        )
        if lp.status == 0:
            best = min(best, float(c[:3] @ np.array(ii, dtype=float) + lp.fun))
    return best


def _pair_closed(r, rel: float = 1e-4, abs_tol: float = 1e-6) -> bool:
    hi, lo = max(r.objective, r.bound), min(r.objective, r.bound)
    gap = hi - lo
    return gap <= abs_tol or gap / max(abs(hi), abs(lo), 1e-10) <= rel


def _assert_honest(r, truth: float, sense: str = "min") -> None:
    """A certificate must describe the true optimum and its own published pair."""
    if r.gap_certified:
        assert r.status == "optimal"
        assert r.objective == pytest.approx(truth, abs=1e-6, rel=1e-4), (
            f"certified {r.objective}, true optimum {truth}"
        )
        assert _pair_closed(r), f"certified pair ({r.objective}, {r.bound}) fails its own test"
        if sense == "min":
            assert r.bound <= truth + 1e-6
        else:
            assert r.bound >= truth - 1e-6


def test_shifted_milp_issue_repro():
    """The issue's model: certified -1 (true -36) before the fix."""
    truth = _oracle(1)
    assert truth == pytest.approx(-36.0)
    r = _build(1, OFF).solve(time_limit=20)
    _assert_honest(r, truth)
    # With the constant handed to HiGHS the search runs to the true optimum, so
    # the fix is not merely a refusal to certify.
    assert r.gap_certified and r.objective == pytest.approx(truth, abs=1e-6)


def test_shifted_milp_maximize_twin():
    truth = -_oracle(1)
    r = _build(1, OFF, sense="max").solve(time_limit=20)
    _assert_honest(r, truth, sense="max")
    assert r.gap_certified and r.objective == pytest.approx(truth, abs=1e-6)


def test_translation_panel():
    """A certified answer must not depend on where the box sits (x = y - c)."""
    compared = 0
    failures = []
    for seed in range(12):
        truth = _oracle(seed)
        if not np.isfinite(truth):
            continue
        for off in (1e3, 1e6):
            r = _build(seed, off).solve(time_limit=20)
            compared += 1
            try:
                _assert_honest(r, truth)
            except AssertionError as exc:
                failures.append((seed, off, r.status, r.objective, r.bound, str(exc)))
    assert compared >= 20, f"panel compared only {compared} models"
    assert not failures, failures


def test_objective_constant_still_certifies():
    """A plain constant in the objective is not the bug and must keep certifying."""
    for K in (0.0, 1e6, -1e6):
        m = dm.Model("k")
        x = m.integer("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.subject_to(x + y <= 7.5)
        m.minimize(-3 * x - 2 * y + K)
        r = m.solve(time_limit=20)
        assert r.gap_certified and r.objective == pytest.approx(-22.0 + K, abs=1e-6)


def test_guard_decertifies_a_route_that_stops_on_an_open_gap(monkeypatch):
    """A route that certifies with a VALID bound but an open published gap -- #1536's
    exact shape (bound -36.67 valid, incumbent -1) -- is refused by the guard alone,
    end to end through ``Model.solve``, with no help from the route fix."""
    import discopt.solvers.lp_milp_highs as lmh

    real = lmh.solve_milp_std
    calls = []

    def open_gap(*args, **kwargs):
        out = real(*args, **kwargs)
        if out.gap_certified and out.status == "optimal" and out.bound is not None:
            out.bound = out.bound - 35.0  # still a valid lower bound; the gap is now open
            calls.append(out.bound)
        return out

    monkeypatch.setattr(lmh, "solve_milp_std", open_gap)
    r = _build(1, 0.0).solve(time_limit=20)
    assert calls, "the HiGHS route was not exercised, so this test measured nothing"
    assert not r.gap_certified and r.status == "feasible"
    assert r.solver_stats.get("certificate/published_pair_refused") == 1.0


def test_guard_unit_maximize_and_closed_pairs():
    """The guard's arithmetic: it refuses an open pair in either sense and leaves a
    closed pair, an uncertified result and a pairless result untouched."""
    from discopt.modeling.core import SolveResult
    from discopt.solver import _refuse_unclosed_published_pair

    def res(obj, bound, cert=True, status="optimal"):
        return SolveResult(
            status=status, objective=obj, bound=bound, gap_certified=cert, solver_stats={}
        )

    open_min = res(-1.0, -36.67)
    _refuse_unclosed_published_pair(open_min, 1e-4, 1e-6)
    assert (open_min.status, open_min.gap_certified) == ("feasible", False)

    open_max = res(36.0, 37.5)  # maximize: bound is an UPPER bound
    _refuse_unclosed_published_pair(open_max, 1e-4, 1e-6)
    assert (open_max.status, open_max.gap_certified) == ("feasible", False)

    for keep in (res(-36.0, -36.0000001), res(1e6, 1e6 - 50.0), res(5.0, None)):
        before = (keep.status, keep.gap_certified)
        _refuse_unclosed_published_pair(keep, 1e-4, 1e-6)
        assert (keep.status, keep.gap_certified) == before

    unc = res(-1.0, -36.67, cert=False, status="feasible")
    _refuse_unclosed_published_pair(unc, 1e-4, 1e-6)
    assert (unc.status, unc.gap_certified) == ("feasible", False)


def test_guard_judges_amp_by_its_own_rel_gap():
    """``solver="amp"`` is asked to meet ``rel_gap``, not ``gap_tolerance``. A pair it
    closes under ``rel_gap=0.5`` is a legitimate certificate and must survive the
    guard (its 5.7% gap would fail the default 1e-4 that ``gap_tolerance`` carries)."""
    m = dm.Model("circle")
    x = m.continuous("x", lb=0, ub=2, shape=(2,))
    m.subject_to(x[0] ** 2 + x[1] ** 2 >= 2)
    m.minimize(x[0] + x[1])
    r = m.solve(solver="amp", rel_gap=0.5, max_iter=100, time_limit=30)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert "certificate/published_pair_refused" not in (r.solver_stats or {})
    assert _pair_closed(r, rel=0.5)
    assert r.bound <= np.sqrt(2) + 1e-6 <= r.objective + 2e-6
