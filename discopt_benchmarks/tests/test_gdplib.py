"""Tests for the GDPlib benchmark integration (issue #823).

The GDPlib corpus (https://github.com/SECQUOIA/gdplib) is an optional dependency,
so every test skips cleanly when pyomo/gdplib are absent. The fast path exercises
``jobshop`` — a small *linear* GDP whose big-M and hull reformulations both certify
11.0, cross-checked against HiGHS (the independent oracle for linear models).

Install to run these locally::

    pip install pyomo highspy
    pip install "gdplib @ git+https://github.com/SECQUOIA/gdplib.git"   # from source
"""

from __future__ import annotations

import pytest

from benchmarks import gdplib_runner as gr
from benchmarks.metrics import InstanceInfo, SolveResult, SolveStatus

pytestmark = pytest.mark.skipif(
    not gr.is_available(),
    reason="GDPlib benchmark requires pyomo + gdplib (install gdplib from source)",
)


def _has_highs() -> bool:
    try:
        import pyomo.environ as pyo

        return bool(pyo.SolverFactory("appsi_highs").available(exception_flag=False))
    except Exception:
        return False


def _has_gams() -> bool:
    try:
        import pyomo.environ as pyo

        return bool(pyo.SolverFactory("gams").available(exception_flag=False))
    except Exception:
        return False


def _spec(name):
    (spec,) = gr.discover_models(include=[name])
    return spec


# ── discovery ──────────────────────────────────────────────────────────────


def test_discover_returns_specs():
    specs = gr.discover_models()
    names = {s.name for s in specs}
    assert names, "expected at least one runnable GDPlib model"
    # jobshop is the canonical small model and must always be discoverable.
    assert "jobshop" in names
    for s in specs:
        assert callable(s.builder)


def test_discover_include_filter():
    specs = gr.discover_models(include=["jobshop"])
    assert [s.name for s in specs] == ["jobshop"]


def test_discover_exclude_filter():
    all_names = {s.name for s in gr.discover_models()}
    if "jobshop" not in all_names:
        pytest.skip("jobshop not present")
    filtered = {s.name for s in gr.discover_models(exclude=["jobshop"])}
    assert "jobshop" not in filtered
    assert filtered == all_names - {"jobshop"}


def test_build_failure_captured_as_error():
    """A builder that raises yields an ERROR run with a reason, not an exception."""

    def _boom():
        raise RuntimeError("needs GAMS on PATH")

    spec = gr.GDPModelSpec(name="synthetic_bad", builder=_boom, module="synthetic")
    run = gr.solve_model(spec, method="bigm", time_limit=5, oracle=False)
    assert run.discopt.status == SolveStatus.ERROR
    assert "build failed" in run.note
    assert "needs GAMS" in run.note


# ── the jobshop end-to-end path (linear GDP, HiGHS-checkable) ───────────────


@pytest.mark.correctness
def test_jobshop_bigm_optimal_and_correct():
    """jobshop via big-M solves to the certified optimum 11.0 with no violation."""
    (spec,) = gr.discover_models(include=["jobshop"])
    run = gr.solve_model(spec, method="bigm", time_limit=120, oracle=True)

    assert run.discopt.status == SolveStatus.OPTIMAL
    assert run.discopt.objective == pytest.approx(11.0, abs=1e-2)
    assert run.is_linear is True
    # The non-negotiable gate: no false optimum, no bound crossing.
    assert run.false_optimum is False, run.note
    assert run.bound_crosses is False, run.note


@pytest.mark.correctness
def test_jobshop_hull_matches_bigm():
    """big-M and hull reformulations must certify the same optimum (self-consistency)."""
    (spec,) = gr.discover_models(include=["jobshop"])
    r_bigm = gr.solve_model(spec, method="bigm", time_limit=120, oracle=False)
    r_hull = gr.solve_model(spec, method="hull", time_limit=120, oracle=False)
    assert r_bigm.discopt.objective == pytest.approx(11.0, abs=1e-2)
    assert r_hull.discopt.objective == pytest.approx(11.0, abs=1e-2)
    assert r_bigm.discopt.objective == pytest.approx(r_hull.discopt.objective, abs=1e-2)


@pytest.mark.correctness
@pytest.mark.skipif(not _has_highs(), reason="HiGHS (highspy) not installed")
def test_jobshop_oracle_is_highs():
    """For the linear jobshop model the oracle is HiGHS, and discopt agrees."""
    (spec,) = gr.discover_models(include=["jobshop"])
    run = gr.solve_model(spec, method="bigm", time_limit=120, oracle=True)
    assert run.oracle_source == "highs"
    assert run.oracle_objective == pytest.approx(11.0, abs=1e-2)


def test_highs_declines_when_not_optimal(monkeypatch):
    """HiGHS is trusted as the equality oracle only on a *proven* optimum (#823 #1).

    A non-optimal termination (time limit hit, interrupted) must yield ``None`` — a
    bare incumbent, if trusted, would flag discopt's correct optimum as an
    impossible incumbent. The corpus's linear models solve instantly (jobshop is
    optimal even at ``time_limit=0``), so the interrupted path is injected here.
    """
    import types

    import pyomo.environ as pyo
    from pyomo.opt import TerminationCondition

    class _FakeHighs:
        def __init__(self):
            self.config = types.SimpleNamespace(time_limit=None)

        def available(self, exception_flag=False):
            return True

        def solve(self, m):
            # Load a bare (garbage) incumbent, as an interrupted MILP solve would:
            # without the optimality gate this value would be returned and wrongly
            # trusted as the oracle. With the gate, the non-optimal status wins.
            for v in m.component_data_objects(pyo.Var, active=True):
                v.set_value(v.lb if v.lb is not None else 0.0, skip_validation=True)
            results = types.SimpleNamespace()
            results.solver = types.SimpleNamespace(
                termination_condition=TerminationCondition.maxTimeLimit
            )
            return results

    monkeypatch.setattr(pyo, "SolverFactory", lambda name: _FakeHighs())
    obj = gr._solve_with_highs(_spec("jobshop"), method="bigm", time_limit=0.0)
    assert obj is None


# ── classification & robustness ─────────────────────────────────────────────


def test_max_variables_skips_large_models():
    """A tiny max_variables budget skips the solve without erroring."""
    (spec,) = gr.discover_models(include=["jobshop"])
    run = gr.solve_model(spec, method="bigm", time_limit=10, oracle=False, max_variables=1)
    assert run.discopt.status == SolveStatus.UNKNOWN
    assert "skipped" in run.note


def test_reference_optima_seed_is_sane():
    import math

    ref = gr.reference_optima()
    assert ref.get("jobshop") == pytest.approx(11.0)
    # BARON-confirmed nonlinear seeds are present and finite.
    for name in ("positioning", "cstr", "small_batch", "syngas", "water_network"):
        assert name in ref
        assert math.isfinite(ref[name])
    # cstr: BARON-proven 3.0620 (pyscipopt-.nl's 3.0543 was a false optimum, #823).
    assert ref["cstr"] == pytest.approx(3.0620073, abs=1e-3)
    assert ref["cstr"] > 3.0543118, "cstr must be the true optimum, not the below-true false value"
    # batch_processing is BARON-certified; methanol/gdp_col remain unproven and unseeded.
    assert ref["batch_processing"] == pytest.approx(679365.33, rel=1e-4)
    for unproven in ("methanol", "gdp_col"):
        assert unproven not in ref


# ── Oracle hardening: feasibility verification (#823) ───────────────────────
#
# The core guard is solver-free and runs in CI: an oracle value is trusted only if
# its incumbent is feasible in the real pyomo model. A claimed optimum below the true
# optimum is a claimed feasible point that isn't feasible, so this closes the hole
# where the old pyscipopt-.nl path certified a below-true cstr optimum.


def _tiny_pyomo_model(x_value):
    """min x s.t. x >= 1, 0 <= x <= 10, with x loaded at *x_value*."""
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.c = pyo.Constraint(expr=m.x >= 1)
    m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
    m.x.set_value(x_value, skip_validation=True)
    return m


def test_feasibility_check_accepts_feasible_point():
    m = _tiny_pyomo_model(1.0)  # satisfies x >= 1
    viol = gr._max_constraint_violation(m)
    assert viol is not None and viol <= gr._ORACLE_FEAS_TOL


def test_feasibility_check_rejects_infeasible_point():
    m = _tiny_pyomo_model(0.0)  # violates x >= 1 by 1.0
    viol = gr._max_constraint_violation(m)
    assert viol is not None and viol > gr._ORACLE_FEAS_TOL


def test_feasibility_check_rejects_unset_solution():
    """An incompletely loaded solution is 'not certified feasible', never OK."""
    import pyomo.environ as pyo

    m = _tiny_pyomo_model(1.0)
    m.y = pyo.Var(bounds=(0, 5))  # left unset -> cannot certify feasibility
    m.cy = pyo.Constraint(expr=m.y <= 3)
    assert gr._max_constraint_violation(m) is None


@pytest.mark.skipif(not _has_gams(), reason="GAMS (SCIP/BARON subsolvers) not available")
def test_gams_oracle_certifies_nonlinear_optimum():
    """SCIP via GAMS returns the feasibility-verified optimum for a nonlinear GDP."""
    obj = gr._solve_with_gams(_spec("ex1_linan_2023"), method="bigm", time_limit=60, solver="scip")
    assert obj is not None
    assert obj == pytest.approx(-0.9996, abs=1e-3)


@pytest.mark.skipif(not _has_gams(), reason="GAMS not available")
def test_gams_oracle_declines_when_gap_open():
    """A 0 s budget cannot close the gap -> no oracle value (never a bare incumbent)."""
    assert gr._solve_with_gams(_spec("cstr"), method="bigm", time_limit=0.0, solver="scip") is None


@pytest.mark.correctness
@pytest.mark.slow
@pytest.mark.skipif(not _has_gams(), reason="GAMS not available")
def test_hardened_oracle_rejects_false_cstr_optimum():
    """Regression for the #823 false optimum: the oracle now certifies cstr's *true*
    optimum (~3.0620), not the below-true 3.0543 the pyscipopt-.nl path reported."""
    run = gr.solve_model(_spec("cstr"), method="bigm", time_limit=60, oracle=True)
    assert run.oracle_source in ("scip+baron", "scip-gams", "baron-gams")
    assert run.oracle_objective == pytest.approx(3.0620, abs=1e-3)
    assert run.oracle_objective > 3.0543118, "must be true optimum, not the below-true value"
    assert run.false_optimum is False, run.note
    assert run.bound_crosses is False, run.note


# ── soundness assessment (solver-free, deterministic) ───────────────────────


def _make_run(status, objective, *, minimize=True, oracle=10.0, bound=None):
    r = SolveResult(
        instance="x/bigm", solver="discopt", status=status, objective=objective, bound=bound
    )
    return gr.ModelRun(
        name="x/bigm",
        info=InstanceInfo(name="x/bigm", source="gdplib"),
        discopt=r,
        is_linear=True,
        minimize=minimize,
        oracle_objective=oracle,
        oracle_source="reference",
    )


def test_assess_clean_when_matches_oracle():
    run = _make_run(SolveStatus.OPTIMAL, 10.0)
    gr._assess(run)
    assert not run.false_optimum and not run.bound_crosses


def test_assess_flags_impossible_feasible_incumbent_min():
    """A merely-FEASIBLE incumbent below the true min optimum is a false primal."""
    run = _make_run(SolveStatus.FEASIBLE, 9.0, minimize=True, oracle=10.0)
    gr._assess(run)
    assert run.false_optimum is True
    assert "IMPOSSIBLE INCUMBENT" in run.note


def test_assess_flags_impossible_feasible_incumbent_max():
    """Symmetric max case: a feasible incumbent above the true max optimum."""
    run = _make_run(SolveStatus.FEASIBLE, 11.0, minimize=False, oracle=10.0)
    gr._assess(run)
    assert run.false_optimum is True


def test_assess_flags_optimal_disagreement():
    """Claimed OPTIMAL but converged worse than the oracle -> false optimum."""
    run = _make_run(SolveStatus.OPTIMAL, 12.0, minimize=True, oracle=10.0)
    gr._assess(run)
    assert run.false_optimum is True
    assert "worse-than-oracle" in run.note


def test_assess_feasible_worse_is_not_flagged():
    """A FEASIBLE (unconverged) incumbent worse than the optimum is expected, not a bug."""
    run = _make_run(SolveStatus.FEASIBLE, 12.0, minimize=True, oracle=10.0)
    gr._assess(run)
    assert run.false_optimum is False


def test_assess_flags_bound_crossing_min():
    """A min-sense dual bound above the optimum would fathom it -> crossing."""
    run = _make_run(SolveStatus.TIME_LIMIT, None, minimize=True, oracle=10.0, bound=10.5)
    gr._assess(run)
    assert run.bound_crosses is True


@pytest.mark.slow
def test_run_suite_jobshop_only_no_violation():
    """A one-model suite runs cleanly and reports zero soundness violations."""
    config = gr.GDPLibSuiteConfig(
        include=["jobshop"], methods=("bigm",), time_limit_seconds=120, oracle=True
    )
    results, runs = gr.run_suite(config)
    assert len(runs) == 1
    assert all(not r.false_optimum and not r.bound_crosses for r in runs)
    # Results flow through the standard metrics pipeline.
    assert "jobshop/bigm" in results.instance_info
    assert results.get_results("discopt")
