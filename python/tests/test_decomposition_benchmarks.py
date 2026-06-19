"""Benchmark-style correctness gate for the decomposition solvers.

Runs Benders and Lagrangian on a battery of block-structured / two-stage
instances with known optima, and cross-checks against the monolithic solver.
This is the consolidated correctness check the plan calls "benchmark promotion":
every decomposition result must match the known optimum *and* the monolithic
solve, and every reported bound must be valid.
"""

import discopt.modeling as dm
import pytest

try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = [
    pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed"),
    pytest.mark.correctness,
]

ABS_TOL = 1e-3


# ── canonical structured instances (mirror discopt_benchmarks) ──


def _two_stage_facility():
    m = dm.Model("facility")
    y = m.binary("y")
    x1 = m.continuous("x1", lb=0, ub=10)
    x2 = m.continuous("x2", lb=0, ub=10)
    m.first_stage(y)
    m.minimize(2 * y + x1 + x2)
    m.subject_to(x1 + x2 >= 3)
    m.subject_to(x1 <= 5 * y)
    m.subject_to(x2 <= 5 * y)
    return m, 5.0


def _block_conflict():
    m = dm.Model("conflict")
    x = m.binary("x", shape=(4,))
    m.minimize(2 * x[0] + 3 * x[1] + 2 * x[2] + 4 * x[3])
    m.subject_to(x[0] + x[1] >= 1)
    m.subject_to(x[2] + x[3] >= 1)
    conf = x[0] + x[2] <= 1
    m.subject_to(conf)
    m.mark_coupling(conf)
    return m, 5.0


def _capacitated_two_stage():
    m = dm.Model("cap")
    a = m.binary("a")
    b = m.binary("b")
    x = m.continuous("x", shape=(4,), lb=0, ub=10)
    m.first_stage(a, b)
    m.minimize(4 * a + 4 * b + (x[0] + 2 * x[1] + 2 * x[2] + x[3]))
    m.subject_to(x[0] + x[1] == 3)
    m.subject_to(x[2] + x[3] == 3)
    m.subject_to(x[0] + x[2] <= 5 * a)
    m.subject_to(x[1] + x[3] <= 5 * b)
    return m, 14.0


BENDERS_CASES = [_two_stage_facility, _capacitated_two_stage]
LAGRANGIAN_CASES = [_block_conflict]


@pytest.mark.parametrize("case", BENDERS_CASES)
def test_benders_matches_known_optimum_and_monolithic(case):
    model, opt = case()
    r = model.solve(decomposition="benders", time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(opt, abs=ABS_TOL)
    # Valid lower bound on a minimization.
    assert r.bound is not None and r.bound <= r.objective + 1e-4
    # Cross-check the monolithic solver on the same model.
    mono_model, _ = case()
    mono = mono_model.solve(time_limit=60)
    assert mono.objective == pytest.approx(opt, abs=ABS_TOL)


@pytest.mark.parametrize("case", LAGRANGIAN_CASES)
@pytest.mark.parametrize("method", ["subgradient", "bundle"])
def test_lagrangian_matches_known_optimum(case, method):
    model, opt = case()
    r = model.solve(decomposition="lagrangian", method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(opt, abs=ABS_TOL)
    assert r.bound is not None and r.bound <= r.objective + 1e-4
    mono_model, _ = case()
    mono = mono_model.solve(time_limit=60)
    assert mono.objective == pytest.approx(opt, abs=ABS_TOL)
