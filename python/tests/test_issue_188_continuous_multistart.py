"""Regression lock for issue #188: continuous stratified multistart at the root.

``kall_congruentcircles_c51`` (Kallrath circle packing; MINLPLib oracle
``=best= 1.0730``) exposed a nonconvex global-search capability gap: the
default spatial path returned the 1.5371 two-row-packing local optimum and
never escaped its basin. Root cause: pure-continuous nonconvex models had
ZERO basin diversification end to end — the integer-centric primal heuristics
(pump/ILS/diving/RINS/RENS) all no-op with no integers to round or flip, the
root multistart NLP is skipped on the McCormick-LP spatial path, and the
strided node NLP warm-starts from the parent point, so every local solve
stays locked in the basin of the first LP-vertex seed.

The fix (general, class-level — no problem-name special cases): a budgeted
stratified continuous multistart at the root
(``primal_heuristics.continuous_multistart``), wired for nonconvex models with
no integer variables on the McCormick-LP path, behind
``SolverTuning.continuous_multistart`` (``DISCOPT_CONTINUOUS_MULTISTART``,
default ON). Measured: 32 stratified starts reach the 1.0730 global basin on
every seed tried (~2.8 s, ~90 ms/solve), while all 4 deterministic anchors and
the LP-vertex seeds converge to 1.54-3.23 local optima.

The MINLPLib instance is not vendored (#163) and the network-restricted CI
cannot fetch it, so these tests run on a faithful reconstruction of the
Kallrath generator (pattern extracted from the vendored
``kall_circles_c8a.nl``): same variable/constraint structure (the .nl carries
one extra defined objvar variable + defining row), same operator classes
(bilinear area equality, reverse-convex pairwise separation, linear
containment/ordering), and the same certified landscape — the single-row
global at ``n - n*pi/4`` (c51: 1.07301, c41: 0.85841 — MINLPLib ``=best=``)
and the two-row local basin at 1.5371 that #188 reported discopt parking in
(reproduced to all published digits by this reconstruction pre-fix).

Soundness: the heuristic is a primal finder only (heuristic-policy regime,
CLAUDE.md §5) — every point is constraint-re-verified and
``inject_incumbent`` enforces strict improvement, so dual bounds and
certificates are untouched; the assertions below lock that too (bound must
stay a valid underestimator, incumbent must never beat the true optimum).
"""

import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver_tuning import SolverTuning

_R = 0.5

# MINLPLib =best= oracles; analytically n - n*pi/4 (single-row packing, area n).
_C51_GLOBAL = 5 - 5 * math.pi / 4  # 1.0730091830...
_C41_GLOBAL = 4 - math.pi  # 0.8584073464...
# The two-row local basin #188 reported the solver parking in (area 2(1+sqrt 3)).
_C51_TWO_ROW_LOCAL = 2 * (1 + math.sqrt(3)) - 5 * math.pi / 4  # 1.5371107...

# Feasibility tolerance (1e-6 on the separation rows) admits incumbents a hair
# below the analytic optimum; keep the soundness guard just past that slack.
_FEAS_SLACK = 1e-4


def make_congruent_circles(n: int, a_max: float, b_max: float) -> "dm.Model":
    """Kallrath congruent-circles generator (reconstruction, see module doc)."""
    m = dm.Model(name=f"congruentcircles_c{n}1_recon")
    xs = [m.continuous(f"x{i}", lb=_R, ub=a_max - _R) for i in range(n)]
    ys = [m.continuous(f"y{i}", lb=_R, ub=b_max - _R) for i in range(n)]
    a = m.continuous("a", lb=0.0, ub=a_max)
    b = m.continuous("b", lb=0.0, ub=b_max)
    t = m.continuous("t", lb=(2 * _R) ** 2 / 4.0, ub=a_max * b_max)

    m.subject_to(t - a * b == 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            m.subject_to(
                (xs[i] - xs[j]) * (xs[i] - xs[j]) + (ys[i] - ys[j]) * (ys[i] - ys[j])
                >= (2 * _R) ** 2
            )
    for i in range(n):
        m.subject_to(xs[i] - a <= -_R)
        m.subject_to(ys[i] - b <= -_R)
    m.subject_to(xs[0] <= a_max / 2.0)
    m.subject_to(ys[0] <= b_max / 2.0)
    for i in range(n):
        for j in range(i + 1, n):
            m.subject_to(xs[i] - xs[j] <= 0.0)

    m.minimize(t - n * math.pi * _R * _R)
    return m


def _build_c51() -> "dm.Model":
    # Variant-1 box: strip long enough for the single-row global optimum,
    # width admitting the two-row local basin observed in #188.
    return make_congruent_circles(5, a_max=5.0, b_max=2.1)


def test_continuous_multistart_reaches_global_basin_unit():
    """The heuristic itself must find the c51 global row basin (fast, no B&B)."""
    from discopt._jax.primal_heuristics import continuous_multistart

    model = _build_c51()
    result = continuous_multistart(model, n_starts=32, seed=42)
    assert result is not None, "continuous multistart found no feasible point at all"
    x, obj = result
    assert obj <= _C51_GLOBAL + 1e-3, (
        f"multistart best {obj!r} did not reach the global basin {_C51_GLOBAL:.6f} "
        f"(pre-#188 landscape: best reachable local was {_C51_TWO_ROW_LOCAL:.6f})"
    )
    # Soundness: a feasible point can undershoot the analytic optimum only by
    # constraint-tolerance slack, never materially.
    assert obj >= _C51_GLOBAL - _FEAS_SLACK
    assert np.all(np.isfinite(x))


def test_continuous_multistart_noops_on_integer_models():
    """Scope guard: integer models keep their own heuristic arsenal; the
    continuous multistart must decline them untried."""
    from discopt._jax.primal_heuristics import continuous_multistart

    m = dm.Model(name="int_scope_guard")
    x = m.continuous("x", lb=0.0, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x * x + z >= 2.0)
    m.minimize(x * x - z)
    assert continuous_multistart(m, n_starts=4, seed=0) is None


def test_flag_parses_and_defaults_on(monkeypatch):
    monkeypatch.delenv("DISCOPT_CONTINUOUS_MULTISTART", raising=False)
    assert SolverTuning().continuous_multistart is True
    monkeypatch.setenv("DISCOPT_CONTINUOUS_MULTISTART", "0")
    assert SolverTuning().continuous_multistart is False


def test_solve_reaches_c51_global_on_default_path():
    """End-to-end #188 lock: the DEFAULT path must reach the c51 global basin.

    Pre-fix this fails with objective 1.5371107... (the two-row local packing,
    reproduced from #188 to all published digits); post-fix the root multistart
    lands the 1.07301 row packing. Also locks soundness: the dual bound stays a
    valid underestimator and the incumbent never materially beats the analytic
    optimum.
    """
    r = _build_c51().solve(time_limit=30, gap_tolerance=1e-4)

    assert r.objective is not None, "expected a feasible incumbent"
    assert r.objective <= _C51_GLOBAL + 1e-3, (
        f"incumbent {r.objective!r} did not reach the global {_C51_GLOBAL:.6f} "
        f"(#188 regression: parked in a local basin, e.g. {_C51_TWO_ROW_LOCAL:.6f}?)"
    )
    assert r.objective >= _C51_GLOBAL - _FEAS_SLACK, (
        f"incumbent {r.objective!r} is below the true optimum {_C51_GLOBAL:.6f} "
        f"beyond feasibility-tolerance slack — unsound incumbent"
    )
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= r.objective + 1e-6, (
            f"dual bound {r.bound!r} exceeds the incumbent {r.objective!r} — "
            f"certificate invariant violated"
        )


@pytest.mark.slow
def test_solve_sibling_c41_no_regress():
    """#188 acceptance: the smaller sibling (already solved pre-fix) must keep
    reaching its global 0.85841."""
    r = make_congruent_circles(4, a_max=4.0, b_max=2.1).solve(time_limit=25, gap_tolerance=1e-4)
    assert r.objective is not None
    assert r.objective <= _C41_GLOBAL + 1e-3
    assert r.objective >= _C41_GLOBAL - _FEAS_SLACK
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= r.objective + 1e-6
