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


# ── classification & robustness ─────────────────────────────────────────────


def test_max_variables_skips_large_models():
    """A tiny max_variables budget skips the solve without erroring."""
    (spec,) = gr.discover_models(include=["jobshop"])
    run = gr.solve_model(spec, method="bigm", time_limit=10, oracle=False, max_variables=1)
    assert run.discopt.status == SolveStatus.UNKNOWN
    assert "skipped" in run.note


def test_reference_optima_seed_is_sane():
    ref = gr.reference_optima()
    assert ref.get("jobshop") == pytest.approx(11.0)


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
