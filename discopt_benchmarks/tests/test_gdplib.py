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


def test_verified_time_limit_incumbent_still_faces_the_impossible_check():
    """A reported time-limit incumbent must NOT bypass the false-primal check.

    ``SolveResult.is_feasible`` is OPTIMAL|FEASIBLE only, so once a verified
    incumbent from a ``time_limit`` run is reported, the impossible-incumbent
    test — the most dangerous check in this file — would skip it unless the gate
    widens too. Reporting without this is an unchecked path (``CLAUDE.md`` §1).
    """
    run = _make_run(SolveStatus.TIME_LIMIT, 9.0, minimize=True, oracle=10.0)
    run.incumbent_verified = True
    gr._assess(run)
    assert run.false_optimum is True
    assert "IMPOSSIBLE INCUMBENT" in run.note


def test_unverified_time_limit_objective_is_not_assessed_as_an_incumbent():
    """Without verification the run carries no objective, so nothing is claimed.

    This pins the pairing: an objective is reported ONLY when verified, and
    verification is exactly what subjects it to the oracle check above.
    """
    run = _make_run(SolveStatus.TIME_LIMIT, None, minimize=True, oracle=10.0)
    assert run.incumbent_verified is False
    gr._assess(run)
    assert run.false_optimum is False


def test_time_limit_incumbent_is_reported_only_when_it_verifies():
    """The production reporting rule: verified -> reported, unverified -> withheld.

    Measured motivation: on cstr discopt exits ``time_limit`` having found
    3.3700327 with a scale-normalized violation of 1.6e-06 — inside this file's
    own ``_ORACLE_FEAS_TOL`` — and the sweep recorded "no incumbent".
    """
    tl = SolveStatus.TIME_LIMIT

    # cstr's real numbers: inside tolerance -> reported, and flagged for re-check.
    assert gr._decide_objective(tl, 3.3700327, 1.6294483888765542e-06) == (3.3700327, True)
    # Outside tolerance -> withheld, never reported as an incumbent.
    assert gr._decide_objective(tl, 3.3700327, 1e-3) == (None, False)
    # Unevaluable / incompletely loaded solution -> withheld, never treated as OK.
    assert gr._decide_objective(tl, 3.37, None) == (None, False)
    assert gr._decide_objective(tl, None, 0.0) == (None, False)
    # A genuinely feasible run is untouched by all of this: reported, and not
    # marked verified because ``is_feasible`` already arms every soundness check.
    assert gr._decide_objective(SolveStatus.OPTIMAL, 3.06201, None) == (3.06201, False)
    assert gr._decide_objective(SolveStatus.FEASIBLE, 3.06201, None) == (3.06201, False)


def test_assess_flags_bound_crossing_min():
    """A min-sense dual bound above the optimum would fathom it -> crossing."""
    run = _make_run(SolveStatus.TIME_LIMIT, None, minimize=True, oracle=10.0, bound=10.5)
    gr._assess(run)
    assert run.bound_crosses is True


def test_sweep_with_no_oracle_acquired_is_vacuous():
    """§6: zero executed comparisons must not read as a clean sweep."""
    runs = [_make_run(SolveStatus.TIME_LIMIT, None, oracle=None)]
    assert gr.sweep_is_vacuous(runs, oracle_enabled=True) is True


def test_sweep_with_one_oracle_is_not_vacuous():
    """A single executed comparison is enough to make the verdict non-vacuous."""
    runs = [
        _make_run(SolveStatus.TIME_LIMIT, None, oracle=None),
        _make_run(SolveStatus.OPTIMAL, 10.0, oracle=10.0),
    ]
    assert gr.sweep_is_vacuous(runs, oracle_enabled=True) is False


def test_no_oracle_mode_is_exempt_from_vacuity():
    """--no-oracle is an honest declaration that nothing is being checked."""
    runs = [_make_run(SolveStatus.TIME_LIMIT, None, oracle=None)]
    assert gr.sweep_is_vacuous(runs, oracle_enabled=False) is False


def test_main_exits_nonzero_on_a_vacuous_sweep(monkeypatch, capsys):
    """The CLI must fail, not pass, when it verified nothing (CLAUDE.md §6).

    Before this guard, ``violations == 0`` was the only exit criterion, so a sweep
    in which every model errored or was skipped exited 0 and printed a checkmark.
    """
    vacuous = [_make_run(SolveStatus.ERROR, None, oracle=None)]

    def _fake_run_suite(config):
        from benchmarks.metrics import BenchmarkResults

        return BenchmarkResults(suite="gdplib", timestamp="now"), vacuous

    monkeypatch.setattr(gr, "run_suite", _fake_run_suite)
    monkeypatch.setattr(gr, "is_available", lambda: True)

    rc = gr.main(["--models", "jobshop"])
    assert rc == 3, "a sweep that executed zero oracle comparisons must not exit 0"
    assert "oracle-checked=0" in capsys.readouterr().err


def test_main_exits_zero_when_no_oracle_requested(monkeypatch):
    """--no-oracle is deliberate, so it stays a clean exit."""
    runs = [_make_run(SolveStatus.TIME_LIMIT, None, oracle=None)]

    def _fake_run_suite(config):
        from benchmarks.metrics import BenchmarkResults

        return BenchmarkResults(suite="gdplib", timestamp="now"), runs

    monkeypatch.setattr(gr, "run_suite", _fake_run_suite)
    monkeypatch.setattr(gr, "is_available", lambda: True)

    assert gr.main(["--models", "jobshop", "--no-oracle"]) == 0


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


# ── the gdplib_small named suite (#993 Phase D) ─────────────────────────────


def test_gdplib_small_is_fully_discoverable():
    """Every model the preset names must exist in this install.

    This is the test that gives the preset its meaning: ``discover_models``
    filters ``include`` by set intersection, so a preset naming a model this
    gdplib revision does not ship would silently shrink to a smaller panel and
    still report clean. If this fails, install gdplib from source.
    """
    found = {s.name for s in gr.discover_models()}
    missing = sorted(set(gr.GDPLIB_SMALL) - found)
    assert not missing, f"gdplib_small names undiscoverable models: {missing}"
    assert len(gr.GDPLIB_SMALL) == len(set(gr.GDPLIB_SMALL)), "duplicate name in preset"


def test_suite_expands_to_the_preset(monkeypatch):
    seen = {}

    def _fake_run_suite(config):
        from benchmarks.metrics import BenchmarkResults

        seen["include"] = list(config.include or [])
        return BenchmarkResults(suite="gdplib", timestamp="now"), []

    monkeypatch.setattr(gr, "run_suite", _fake_run_suite)
    monkeypatch.setattr(gr, "is_available", lambda: True)
    gr.main(["--suite", "gdplib_small", "--no-oracle"])
    assert seen["include"] == list(gr.GDPLIB_SMALL)


def test_suite_and_models_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        gr.main(["--suite", "gdplib_small", "--models", "jobshop"])
    assert exc.value.code == 2


def test_suite_refuses_a_shrunken_preset(monkeypatch):
    """A preset model absent from the install exits nonzero, never silently."""
    # Resolve the spec BEFORE patching — _spec() calls discover_models() itself.
    only = [_spec("jobshop")]
    monkeypatch.setattr(gr, "is_available", lambda: True)
    monkeypatch.setattr(gr, "discover_models", lambda include=None, exclude=None: only)

    def _must_not_run(config):  # pragma: no cover - reached only on a bug
        raise AssertionError("run_suite must not be reached with a shrunken preset")

    monkeypatch.setattr(gr, "run_suite", _must_not_run)
    assert gr.main(["--suite", "gdplib_small", "--no-oracle"]) == 4


@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_gdplib_small_suite_runs_clean():
    """On-demand wrapper for the named suite: every model runs, nothing unsound.

    Deliberately *not* a CI test — it is the reproducible entry point for the
    GDP panel (``--suite gdplib_small``). The assertions are the two things a
    panel must never get wrong: it ran every model it claims to cover (§6), and
    no run produced a false optimum or a crossed bound (§1).
    """
    config = gr.GDPLibSuiteConfig(
        include=list(gr.GDPLIB_SMALL),
        methods=("bigm",),
        time_limit_seconds=60,
        oracle=True,
    )
    results, runs = gr.run_suite(config)
    assert len(runs) == len(gr.GDPLIB_SMALL), (
        f"suite ran {len(runs)} of {len(gr.GDPLIB_SMALL)} models — a shrunken panel"
    )
    bad = [r.name for r in runs if r.false_optimum or r.bound_crosses]
    assert not bad, f"soundness violations on {bad}"
    checked = sum(1 for r in runs if r.oracle_objective is not None)
    assert checked > 0, "zero oracle-checked runs — this panel verified nothing"
    assert results.get_results("discopt")
