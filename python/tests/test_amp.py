"""Fast AMP regressions for the default PR test battery."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import Model, SolveResult

pytestmark = pytest.mark.smoke


def _make_nlp1() -> Model:
    m = Model("nlp1")
    x = m.continuous("x", lb=1, ub=4, shape=(2,))
    m.subject_to(x[0] * x[1] >= 8)
    m.minimize(6 * x[0] ** 2 + 4 * x[1] ** 2 - 2.5 * x[0] * x[1])
    return m


def _make_circle() -> Model:
    m = Model("circle")
    x = m.continuous("x", lb=0, ub=2, shape=(2,))
    m.subject_to(x[0] ** 2 + x[1] ** 2 >= 2)
    m.minimize(x[0] + x[1])
    return m


def _make_obbt_demo() -> Model:
    m = Model("obbt_demo")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y == 1)
    m.maximize(x * y)
    return m


def _make_shifted_square_infeasible() -> Model:
    m = dm.Model("nlp_mi_007_020")
    x = m.integer("x", lb=-2, ub=3)
    y = m.binary("y")
    m.minimize(x * 0.0)
    m.subject_to((x - 0.5) ** 2 + (4 * y - 2) ** 2 <= 3)
    return m


def _build_relaxation_for_test(
    model: Model,
    part_vars: list[int] | None = None,
    lbs: list[float] | None = None,
    ubs: list[float] | None = None,
    n_init: int = 2,
):
    from discopt._jax.discretization import initialize_partitions
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    terms = classify_nonlinear_terms(model)
    state = initialize_partitions(part_vars or [], lb=lbs or [], ub=ubs or [], n_init=n_init)
    return build_milp_relaxation(model, terms, state, incumbent=None)


def test_amp_integration_suite_is_opt_in():
    """The Alpine/MINLPTests suite must stay out of the default marker selection."""
    text = Path(__file__).with_name("test_amp_integration.py").read_text(encoding="utf-8")

    assert "pytest.mark.slow" in text
    assert "pytest.mark.integration" in text
    assert "pytest.mark.amp_benchmark" in text
    assert "pytest.mark.requires_cyipopt" in text
    assert "pytest.mark.memory_heavy" in text


def _make_dry_run(target: str, env_overrides: dict[str, str] | None = None) -> str:
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    if env_overrides is None:
        env.pop("PYTEST_XDIST_WORKERS", None)
    else:
        env.update(env_overrides)
    result = subprocess.run(
        [
            "make",
            "-n",
            target,
            "PYTHON=python",
            "PYTEST=python -m pytest",
            "MATURIN=python -m maturin",
            "RUFF=python -m ruff",
        ],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return result.stdout


def _dry_run_pytest_args(output: str) -> list[str]:
    for line in output.splitlines():
        if "python -m pytest" in line:
            return shlex.split(line)
    raise AssertionError(f"pytest command not found in dry-run output:\n{output}")


def _dry_run_marker_expression(output: str) -> str:
    args = _dry_run_pytest_args(output)
    marker_idx = [idx for idx, arg in enumerate(args) if arg == "-m"][-1]
    return args[marker_idx + 1]


def test_quick_test_tier_excludes_amp_integration_markers():
    """The quick tier must not select opt-in AMP smoke tests or use prlimit."""
    output = _make_dry_run("test-quick")
    marker_expr = _dry_run_marker_expression(output)

    assert "unit or smoke" in marker_expr
    for marker in ("slow", "integration", "amp_benchmark", "requires_cyipopt", "memory_heavy"):
        assert f"not {marker}" in marker_expr
    assert "python -m pytest python/tests/" in output
    assert "scripts/run_memory_capped_pytest.sh python -m pytest" not in output


def test_pr_fast_tier_excludes_heavy_manual_markers():
    """The PR-fast tier should keep nightly/manual markers out of make test."""
    output = _make_dry_run("test")
    marker_expr = _dry_run_marker_expression(output)
    args = _dry_run_pytest_args(output)

    for marker in (
        "slow",
        "correctness",
        "integration",
        "amp_benchmark",
        "requires_cyipopt",
        "memory_heavy",
    ):
        assert f"not {marker}" in marker_expr
    assert "--ignore=python/tests/test_correctness.py" in output
    assert args[args.index("-n") + 1] == "auto"
    assert "scripts/run_memory_capped_pytest.sh python -m pytest" in output


def test_pr_fast_tier_honors_worker_override():
    """The PR-fast tier should honor CI's explicit xdist worker count."""
    output = _make_dry_run("test", {"PYTEST_XDIST_WORKERS": "2"})
    args = _dry_run_pytest_args(output)

    assert args[args.index("-n") + 1] == "2"


def test_amp_fast_tier_excludes_optional_solver_markers():
    """The local AMP fast target should not require optional NLP solver stacks."""
    output = _make_dry_run("test-amp-fast")
    marker_expr = _dry_run_marker_expression(output)

    for marker in ("slow", "integration", "amp_benchmark", "requires_cyipopt", "memory_heavy"):
        assert f"not {marker}" in marker_expr
    assert "scripts/run_memory_capped_pytest.sh python -m pytest" in output


def test_amp_integration_tier_selects_memory_heavy_marker():
    """The opt-in AMP integration target should include memory-heavy tests under the cap."""
    output = _make_dry_run("test-amp-integration")
    marker_expr = _dry_run_marker_expression(output)

    for marker in ("slow", "integration", "amp_benchmark", "requires_cyipopt", "memory_heavy"):
        assert marker in marker_expr
    assert "scripts/run_memory_capped_pytest.sh python -m pytest" in output


def test_embedding_map_single_partition_uses_no_selector_bits():
    """One SOS2 interval does not need an embedded binary selector."""
    from discopt._jax.embedding import EmbeddingMap, build_embedding_map

    embedding = build_embedding_map(2, encoding="gray")

    assert isinstance(embedding, EmbeddingMap)
    assert embedding.bit_count == 0
    assert embedding.codes == ((),)
    assert embedding.positive_sets == ()
    assert embedding.negative_sets == ()


def test_memory_capped_pytest_wrapper_dry_run():
    """The pytest wrapper should expose the command and resource caps for diagnosis."""
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "run_memory_capped_pytest.sh"
    env = {
        **os.environ,
        "RUN_MEMORY_CAPPED_PYTEST_DRY_RUN": "1",
        "PYTEST_MEMORY_LIMIT_MB": "64",
        "PYTEST_CPU_LIMIT_SECONDS": "5",
    }

    result = subprocess.run(
        [str(script), "python", "-m", "pytest", "python/tests/test_amp.py", "-q"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "python" in result.stdout
    assert "pytest" in result.stdout
    assert "python/tests/test_amp.py" in result.stdout
    if "prlimit" in result.stdout:
        assert "--as=67108864" in result.stdout
        assert "--cpu=5" in result.stdout


def test_memory_capped_pytest_warns_when_prlimit_missing():
    """Missing prlimit should be explicit because the test then runs uncapped."""
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "run_memory_capped_pytest.sh"
    env = {
        **os.environ,
        "PATH": "/tmp",
        "PYTEST_MEMORY_LIMIT_MB": "64",
        "PYTEST_CPU_LIMIT_SECONDS": "5",
    }

    result = subprocess.run(
        ["/bin/bash", str(script), "/bin/true"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "resource caps are NOT enforced" in result.stderr


def test_amp_helper_defaults_cover_semifinite_domains():
    """AMP fallback starts should stay finite on semi-infinite NLP boxes."""
    from discopt.solvers import amp as amp_mod

    lb = np.array([-1e20, 2.0, -1e20, 1.0], dtype=np.float64)
    ub = np.array([1e20, 1e20, -3.0, 5.0], dtype=np.float64)

    start = amp_mod._default_nlp_start(lb, ub)
    np.testing.assert_allclose(start, np.array([0.0, 2.0, -3.0, 3.0]))

    recovery_starts = amp_mod._continuous_recovery_starts(
        np.array([-1e20, 2.0], dtype=np.float64),
        np.array([1e20, 1e20], dtype=np.float64),
        initial_point=np.array([0.5, 10.0], dtype=np.float64),
    )

    assert len(recovery_starts) == 3
    np.testing.assert_allclose(recovery_starts[0], np.array([0.5, 10.0]))
    np.testing.assert_allclose(recovery_starts[1], np.array([0.0, 2.0]))
    np.testing.assert_allclose(recovery_starts[2], np.array([1.0, 2.0]))


def test_exp_univariate_domain_rejects_overflowing_bounds_without_warning():
    """Wide finite exp domains should be rejected without probing exp(ub)."""
    from discopt._jax.milp_relaxation import _univariate_domain_ok

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _univariate_domain_ok("exp", 0.0, 1000.0) is False
        assert _univariate_domain_ok("exp", -1000.0, 10.0) is True


def test_amp_normalizes_initial_point_length_and_bounds():
    """Initial AMP points should be length-checked and clipped to tightened bounds."""
    from discopt.solvers import amp as amp_mod

    lb = np.array([0.0, -1.0], dtype=np.float64)
    ub = np.array([1.0, 2.0], dtype=np.float64)

    clipped = amp_mod._normalize_initial_point(np.array([-3.0, 4.0]), 2, lb, ub)

    np.testing.assert_allclose(clipped, np.array([0.0, 2.0]))
    with pytest.raises(ValueError, match="expected 2"):
        amp_mod._normalize_initial_point(np.array([1.0]), 2, lb, ub)


def test_weymouth_like_squares_extend_builtin_partition_selection():
    """Coupled square constraints should add monomial vars without changing classifier output."""
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt.solvers import amp as amp_mod

    m = Model("weymouth_like_candidates")
    x = m.continuous("x", lb=0.0, ub=10.0, shape=(4,))
    m.minimize(x[0] * x[1])
    m.subject_to(x[2] ** 2 == 3.0 * x[3] ** 2)

    terms = classify_nonlinear_terms(m)

    assert (2, 2) in terms.monomial
    assert (3, 2) in terms.monomial
    assert set(terms.partition_candidates) == {0, 1}
    assert set(amp_mod._equality_square_monomial_partition_candidates(m, terms)) == {2, 3}

    sphere = Model("sphere_candidates")
    y = sphere.continuous("y", lb=-4.0, ub=4.0, shape=(3,))
    sphere.minimize(y[0] * y[2])
    sphere.subject_to(y[0] ** 2 + y[1] ** 2 + y[2] ** 2 <= 10.0)

    sphere_terms = classify_nonlinear_terms(sphere)

    assert set(sphere_terms.partition_candidates) == {0, 2}
    assert set(amp_mod._equality_square_monomial_partition_candidates(sphere, sphere_terms)) == {
        0,
        1,
        2,
    }


def test_partitioned_square_secants_tighten_circle_superlevel_bound():
    """Local square secants should close the Alpine circle MILP lower bound."""
    m = _make_circle()
    part_vars = [0, 1]
    coarse_model, _ = _build_relaxation_for_test(
        m,
        part_vars=part_vars,
        lbs=[0.0, 0.0],
        ubs=[2.0, 2.0],
        n_init=2,
    )
    fine_model, fine_varmap = _build_relaxation_for_test(
        m,
        part_vars=part_vars,
        lbs=[0.0, 0.0],
        ubs=[2.0, 2.0],
        n_init=64,
    )

    coarse = coarse_model.solve()
    fine = fine_model.solve()

    assert set(fine_varmap["monomial_pw"]) == {(0, 2), (1, 2)}
    assert coarse.objective is not None
    assert fine.objective is not None
    assert fine.objective >= coarse.objective + 0.05
    assert fine.objective == pytest.approx(np.sqrt(2.0), abs=1e-4)


def test_shifted_square_constraint_linearizes_and_proves_infeasible(caplog):
    """Affine-square constraints should stay in the AMP MILP relaxation."""
    from discopt._jax.term_classifier import classify_nonlinear_terms

    m = _make_shifted_square_infeasible()
    terms = classify_nonlinear_terms(m)

    assert set(terms.monomial) == {(0, 2), (1, 2)}

    with caplog.at_level("WARNING"):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    assert set(varmap["monomial"]) == {(0, 2), (1, 2)}
    assert result.status == "infeasible"
    assert "omitting constraint" not in caplog.text


def test_issue90_unbounded_square_constraint_linearizes_with_lifted_aux(caplog):
    """The nlp_008_010 square constraint should get aux columns even on wide boxes."""
    m = Model("nlp_008_010_square_core")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize(y**2 + z**3)
    m.subject_to(x**2 <= y**2 + z**2)

    with caplog.at_level("WARNING", logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    assert {(0, 2), (1, 2), (2, 2), (2, 3)} <= set(varmap["monomial"])
    assert milp_model._objective_bound_valid is True
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-8)
    assert "Monomial (0, 2)" not in caplog.text
    assert "omitting constraint" not in caplog.text


@pytest.mark.memory_heavy
def test_amp_reports_shifted_square_minlptests_case_infeasible():
    """Regression for MINLPTests nlp_mi_007_020."""
    m = _make_shifted_square_infeasible()

    result = m.solve(
        solver="amp",
        nlp_solver="ipm",
        max_iter=1,
        time_limit=10.0,
        presolve_bt=False,
        apply_partitioning=False,
    )

    assert result.status == "infeasible"
    assert result.x is None


def test_solve_nlp_subproblem_retries_ipopt_and_restores_bounds(monkeypatch):
    """AMP local NLP recovery should retry Ipopt without mutating model bounds."""
    import discopt._jax.ipm as ipm_mod
    import discopt.solver as solver_mod
    import discopt.solvers.nlp_ipopt as ipopt_mod
    from discopt.solvers import NLPResult, SolveStatus
    from discopt.solvers import amp as amp_mod

    m = Model("nlp_retry")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize((x - 3.0) ** 2)
    original_lb = x.lb.copy()
    original_ub = x.ub.copy()

    class FakeEvaluator:
        _model = m
        _obj_fn = object()

        def evaluate_objective(self, x_flat):
            return float((x_flat[0] - 3.0) ** 2)

    calls = []
    seen_bounds = []

    def fake_ipm(evaluator, x0, options):
        del x0, options
        calls.append("ipm")
        seen_bounds.append(evaluator.variable_bounds)
        return NLPResult(status=SolveStatus.ERROR)

    def fake_ipopt(evaluator, x0, options):
        del options
        calls.append("ipopt")
        seen_bounds.append(evaluator.variable_bounds)
        return NLPResult(status=SolveStatus.OPTIMAL, x=np.array([3.0]), objective=0.0)

    monkeypatch.setattr(amp_mod, "_has_cyipopt", lambda: True)
    monkeypatch.setattr(ipm_mod, "solve_nlp_ipm", fake_ipm)
    monkeypatch.setattr(ipopt_mod, "solve_nlp", fake_ipopt)
    assert solver_mod.solve_nlp is not fake_ipopt

    x_opt, obj = amp_mod._solve_nlp_subproblem(
        FakeEvaluator(),
        x0=np.array([99.0], dtype=np.float64),
        lb=np.array([1.0], dtype=np.float64),
        ub=np.array([5.0], dtype=np.float64),
        nlp_solver="ipm",
        time_limit=10.0,
    )

    assert calls == ["ipm", "ipopt"]
    for lb_seen, ub_seen in seen_bounds:
        np.testing.assert_allclose(lb_seen, np.array([1.0]))
        np.testing.assert_allclose(ub_seen, np.array([5.0]))
    np.testing.assert_allclose(x_opt, np.array([3.0]))
    assert obj == pytest.approx(0.0)
    np.testing.assert_allclose(x.lb, original_lb)
    np.testing.assert_allclose(x.ub, original_ub)


def test_repair_inverted_bounds_snaps_to_midpoint():
    from discopt.solvers import amp as amp_mod

    lb, ub = amp_mod._repair_inverted_bounds(
        np.array([0.0, 2.0], dtype=np.float64),
        np.array([1.0, 1.999999999999], dtype=np.float64),
    )

    assert lb[0] == pytest.approx(0.0)
    assert ub[0] == pytest.approx(1.0)
    assert lb[1] == pytest.approx(1.9999999999995)
    assert ub[1] == pytest.approx(1.9999999999995)


def test_solve_nlp_subproblem_respects_expired_time_limit():
    """Expired local NLP budgets should return immediately."""
    from discopt.solvers import amp as amp_mod

    x_opt, obj = amp_mod._solve_nlp_subproblem(
        evaluator=object(),
        x0=np.array([0.0], dtype=np.float64),
        lb=np.array([0.0], dtype=np.float64),
        ub=np.array([1.0], dtype=np.float64),
        time_limit=0.0,
    )

    assert x_opt is None
    assert obj is None


def test_recover_pure_continuous_solution_uses_best_start_and_maximize_sign(monkeypatch):
    """Pure-continuous recovery should keep the best local NLP start."""
    from discopt.solvers import amp as amp_mod

    m = Model("continuous_recovery")
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.maximize(x)
    starts = [
        np.array([0.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([2.0], dtype=np.float64),
    ]
    objectives = {0.0: 5.0, 1.0: 2.0, 2.0: 4.0}

    monkeypatch.setattr(
        amp_mod,
        "_continuous_recovery_starts",
        lambda flat_lb, flat_ub, initial_point=None: [start.copy() for start in starts],
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_nlp_subproblem",
        lambda evaluator, x0, lb, ub, nlp_solver, time_limit=None: (
            x0.copy(),
            objectives[float(x0[0])],
        ),
    )

    result = amp_mod._recover_pure_continuous_solution(
        m,
        evaluator=object(),
        flat_lb=np.array([0.0], dtype=np.float64),
        flat_ub=np.array([2.0], dtype=np.float64),
        nlp_solver="ipm",
        t_start=amp_mod.time.perf_counter(),
        time_limit=10.0,
    )

    assert result is not None
    assert result.status == "feasible"
    assert result.objective == pytest.approx(-2.0)
    assert float(result.x["x"]) == pytest.approx(1.0)


def test_small_integer_domain_size_edges():
    """Small integer fallback should reject absent, empty, and oversized domains."""
    from discopt.solvers import amp as amp_mod

    continuous = Model("no_integer")
    continuous.continuous("x", lb=0, ub=1)
    assert amp_mod._small_integer_domain_size(continuous, max_assignments=4) is None

    empty = Model("empty_integer_domain")
    empty.integer("y", lb=2, ub=1)
    assert amp_mod._small_integer_domain_size(empty, max_assignments=4) == 0

    oversized = Model("oversized_integer_domain")
    oversized.integer("z", lb=0, ub=10)
    assert amp_mod._small_integer_domain_size(oversized, max_assignments=4) is None


def test_amp_small_helpers_cover_aliases_gaps_and_pruning():
    """Small AMP helpers should preserve public aliases and edge-case math."""
    from discopt.solvers import amp as amp_mod

    assert amp_mod._normalize_partition_method("auto", None) == "auto"
    assert amp_mod._normalize_partition_method("auto", "all") == "max_cover"
    assert amp_mod._normalize_partition_method("auto", "adaptive") == "adaptive_vertex_cover"
    assert amp_mod._normalize_partition_method("auto", 3) == "adaptive_vertex_cover"
    assert amp_mod._normalize_partition_method("auto", lambda ctx: [0]) == "auto"

    with pytest.raises(ValueError, match="Unsupported disc_var_pick string"):
        amp_mod._normalize_partition_method("auto", "missing")
    with pytest.raises(ValueError, match="Unsupported disc_var_pick integer"):
        amp_mod._normalize_partition_method("auto", 9)

    assert amp_mod._normalize_presolve_bt_algo(1) == "lp"
    assert amp_mod._normalize_presolve_bt_algo("lp_obbt") == "lp"
    assert amp_mod._normalize_presolve_bt_algo(2) == "incumbent_partitioned"
    assert amp_mod._normalize_presolve_bt_algo("tmc") == "incumbent_partitioned"
    with pytest.raises(ValueError, match="Unsupported presolve_bt_algo"):
        amp_mod._normalize_presolve_bt_algo("missing")

    assert amp_mod._default_milp_time_limit(remaining=10.0, iteration=1, max_iter=5) == 6.0
    assert amp_mod._default_obbt_time_limit_per_lp(remaining=-1.0, n_orig=2) == 0.0
    assert amp_mod._default_obbt_time_limit_per_lp(remaining=10.0, n_orig=2) == pytest.approx(0.25)
    assert amp_mod._resolve_presolve_bt_time_limits(
        remaining=100.0,
        n_orig=2,
        presolve_bt_time_limit=5.0,
        presolve_bt_mip_time_limit=0.7,
    ) == pytest.approx((5.0, 0.7))
    with pytest.raises(ValueError, match="presolve_bt_time_limit"):
        amp_mod._resolve_presolve_bt_time_limits(10.0, 1, -1.0, None)
    assert amp_mod._compute_relative_gap(None, 1.0) is None
    assert amp_mod._compute_relative_gap(-1.0, 1.0) is None
    assert amp_mod._compute_relative_gap(1.0, 0.0) is None
    assert amp_mod._compute_relative_gap(2.0, -4.0) == pytest.approx(0.5)

    cuts = ["old", "keep1", "keep2"]
    amp_mod._prune_oa_cuts(cuts, max_cuts=2)
    assert cuts == ["keep1", "keep2"]


def test_amp_constraint_helpers_cover_success_and_failure(caplog):
    """Constraint helper failures should reject points instead of accepting them."""
    from discopt.solvers import amp as amp_mod

    no_constraints = Model("no_constraints")
    x = no_constraints.continuous("x", lb=0, ub=1)
    no_constraints.minimize(x)
    assert amp_mod._check_constraints(np.array([0.5]), no_constraints)

    constrained = Model("constrained_check")
    y = constrained.continuous("y", lb=0, ub=1)
    constrained.subject_to(y >= 0.25)
    constrained.minimize(y)
    assert amp_mod._check_constraints(np.array([0.5]), constrained)
    assert not amp_mod._check_constraints(np.array([0.0]), constrained)

    class BadEvaluator:
        n_constraints = 1

        def evaluate_constraints(self, x_flat):
            del x_flat
            raise RuntimeError("boom")

    assert not amp_mod._check_constraints_with_evaluator(
        BadEvaluator(),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
    )
    assert "constraint evaluation failed" in caplog.text


def test_select_best_nlp_candidate_rejects_infeasible_and_expired(monkeypatch):
    """Candidate selection should honor deadlines and constraint feasibility."""
    from discopt.solvers import amp as amp_mod

    m = Model("candidate_constraints")
    m.continuous("x", lb=0, ub=1)

    class InfeasibleEvaluator:
        n_constraints = 1

        def evaluate_constraints(self, x_flat):
            del x_flat
            return np.array([2.0], dtype=np.float64)

    monkeypatch.setattr(
        amp_mod,
        "_build_fixed_integer_bounds",
        lambda x0, model, flat_lb, flat_ub: (flat_lb.copy(), flat_ub.copy()),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_nlp_subproblem",
        lambda evaluator, x0, lb, ub, nlp_solver, time_limit=None: (
            x0.copy(),
            0.0,
        ),
    )

    best_x, best_obj = amp_mod._select_best_nlp_candidate(
        [np.array([0.5], dtype=np.float64)],
        m,
        evaluator=InfeasibleEvaluator(),
        flat_lb=np.array([0.0], dtype=np.float64),
        flat_ub=np.array([1.0], dtype=np.float64),
        constraint_lb=np.array([0.0], dtype=np.float64),
        constraint_ub=np.array([1.0], dtype=np.float64),
        nlp_solver="ipm",
    )
    assert best_x is None
    assert best_obj is None

    expired_x, expired_obj = amp_mod._select_best_nlp_candidate(
        [np.array([0.5], dtype=np.float64)],
        m,
        evaluator=InfeasibleEvaluator(),
        flat_lb=np.array([0.0], dtype=np.float64),
        flat_ub=np.array([1.0], dtype=np.float64),
        constraint_lb=np.array([0.0], dtype=np.float64),
        constraint_ub=np.array([1.0], dtype=np.float64),
        nlp_solver="ipm",
        deadline=amp_mod.time.perf_counter() - 1.0,
    )
    assert expired_x is None
    assert expired_obj is None


def test_solve_amp_validates_public_options_before_solve():
    """Public AMP validation should fail before expensive solver work starts."""
    from discopt.solvers import amp as amp_mod

    m = Model("amp_validation")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)

    with pytest.raises(ValueError, match="partition_scaling_factor"):
        amp_mod.solve_amp(m, partition_scaling_factor=1.0, skip_convex_check=True)
    with pytest.raises(ValueError, match="disc_add_partition_method"):
        amp_mod.solve_amp(m, disc_add_partition_method="bad", skip_convex_check=True)
    with pytest.raises(ValueError, match="convhull_ebd requires"):
        amp_mod.solve_amp(
            m,
            convhull_ebd=True,
            convhull_formulation="facet",
            skip_convex_check=True,
        )


def test_solve_amp_convex_model_delegates_to_continuous_solver(monkeypatch):
    """Convex pure-continuous models should use the single-NLP fast path."""
    import discopt._jax.convexity as convexity_mod
    import discopt.solver as solver_mod
    from discopt.solvers import amp as amp_mod

    m = Model("convex_fast_path")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize((x - 0.25) ** 2)
    captured = {}

    monkeypatch.setattr(convexity_mod, "classify_model", lambda model, use_certificate: (True, {}))

    def fake_solve_continuous(
        model,
        time_limit,
        ipopt_options,
        t_start,
        nlp_solver,
        initial_point,
    ):
        del model, time_limit, ipopt_options, t_start, nlp_solver
        captured["initial_point"] = initial_point.copy()
        return SolveResult(status="optimal", objective=0.0, x={"x": np.array(0.25)})

    monkeypatch.setattr(solver_mod, "_solve_continuous", fake_solve_continuous)

    result = amp_mod.solve_amp(m, initial_point=np.array([2.0]), time_limit=1.0)

    assert result.status == "optimal"
    assert result.convex_fast_path is True
    np.testing.assert_allclose(captured["initial_point"], np.array([1.0]))


@pytest.mark.parametrize(
    ("name", "known_min"),
    [
        ("sqrt", 1.0),
        ("log", 0.0),
        ("exp", 1.0),
        ("abs", 0.0),
    ],
)
def test_supported_univariate_objectives_return_valid_bounds(name, known_min):
    """Supported affine univariate objectives should produce sound MILP bounds."""
    m = Model(f"{name}_obj")
    if name == "sqrt":
        x = m.continuous("x", lb=0.0, ub=3.0)
        m.minimize(dm.sqrt(x + 1.0))
    elif name == "log":
        x = m.continuous("x", lb=1.0, ub=4.0)
        m.minimize(dm.log(x))
    elif name == "exp":
        x = m.continuous("x", lb=0.0, ub=2.0)
        m.minimize(dm.exp(x))
    else:
        x = m.continuous("x", lb=-2.0, ub=3.0)
        m.minimize(dm.abs(x))

    milp_model, varmap = _build_relaxation_for_test(m)
    result = milp_model.solve()

    assert result.status == "optimal"
    assert result.objective is not None
    assert result.objective <= known_min + 1e-8
    assert {r.func_name for r in varmap["univariate_relaxations"]} == {name}


def test_supported_univariate_constraint_tightens_relaxation():
    """Supported operator constraints should be kept instead of omitted."""
    m = Model("exp_constraint")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=1.5)
    m.subject_to(dm.exp(x) <= y)
    m.minimize(-x)

    milp_model, varmap = _build_relaxation_for_test(m)
    result = milp_model.solve()

    assert result.status == "optimal"
    assert result.objective is not None
    assert result.objective > -1.0
    assert len(varmap["univariate_relaxations"]) == 1
    assert varmap["univariate_relaxations"][0].func_name == "exp"


def test_issue71_log_constraint_is_kept_in_relaxation(caplog):
    """Safe affine log constraints should not be omitted from the AMP relaxation."""
    m = Model("issue71_log_constraint")
    x = m.continuous("x", lb=1.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(y <= dm.log(x) - 0.1)
    m.maximize(y)

    with caplog.at_level("WARNING", logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    assert result.status == "optimal"
    assert milp_model._objective_bound_valid is True
    assert result.objective is not None
    assert {r.func_name for r in varmap["univariate_relaxations"]} == {"log"}
    assert "Cannot linearize FunctionCall: log" not in caplog.text
    assert "omitting constraint" not in caplog.text


def test_issue71_maximize_sqrt_objective_uses_real_relaxation_bound(caplog):
    """Safe affine sqrt maximization objectives should not use a feasibility objective."""
    m = Model("issue71_sqrt_max")
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.maximize(dm.sqrt(x + 0.1))

    with caplog.at_level("WARNING", logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    assert result.status == "optimal"
    assert milp_model._objective_bound_valid is True
    assert result.objective is not None
    assert result.objective == pytest.approx(-np.sqrt(4.1), abs=1e-8)
    assert {r.func_name for r in varmap["univariate_relaxations"]} == {"sqrt"}
    assert "falling back to a feasibility objective" not in caplog.text


@pytest.mark.parametrize(
    ("sense", "func_name", "expected_relaxation_obj"),
    [
        ("minimize", "max", 1.5),
        ("maximize", "min", -1.5),
    ],
)
def test_issue64_affine_minmax_objective_lift_adds_correct_rows(
    sense,
    func_name,
    expected_relaxation_obj,
):
    """Objective-level min/max calls should be lifted with epigraph/hypograph rows."""
    m = Model(f"issue64_affine_{func_name}")
    x = m.continuous("x", lb=0.0, ub=3.0)
    expr = dm.maximum(x + 1.0, 2.0 - x) if func_name == "max" else dm.minimum(x + 1.0, 2.0 - x)
    if sense == "minimize":
        m.minimize(expr)
    else:
        m.maximize(expr)

    milp_model, varmap = _build_relaxation_for_test(m)
    result = milp_model.solve()

    lift = varmap["minmax_objective_lift"]
    assert lift is not None
    assert lift["func_name"] == func_name
    assert milp_model._objective_bound_valid is True
    assert result.status == "optimal"
    assert result.objective == pytest.approx(expected_relaxation_obj, abs=1e-8)


@pytest.mark.parametrize(
    ("sense", "func_name", "expected_relaxation_obj"),
    [
        ("maximize", "min", -0.75),
        ("minimize", "max", 0.75),
    ],
)
def test_issue64_minlptests_minmax_objective_uses_lifted_bound(
    sense,
    func_name,
    expected_relaxation_obj,
    caplog,
):
    """The nlp_009 min/max objectives should not force feasibility-objective fallback."""
    m = Model(f"issue64_nlp_009_{func_name}")
    x = m.continuous("x")
    if func_name == "min":
        expr = dm.minimum(0.75 + (x - 0.5) ** 3, 0.75 - (x - 0.5) ** 2)
    else:
        expr = dm.maximum(0.75 + (x - 0.5) ** 3, 0.75 + (x - 0.5) ** 2)
    if sense == "minimize":
        m.minimize(expr)
    else:
        m.maximize(expr)

    with caplog.at_level(logging.WARNING, logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    lift = varmap["minmax_objective_lift"]
    assert lift is not None
    assert lift["func_name"] == func_name
    assert milp_model._objective_bound_valid is True
    assert result.status == "optimal"
    assert result.objective == pytest.approx(expected_relaxation_obj, abs=1e-8)
    assert "falling back to a feasibility objective" not in caplog.text
    assert f"Cannot linearize FunctionCall: {func_name}" not in caplog.text


def test_tan_range_rejects_near_asymptote_endpoints():
    from discopt._jax.milp_relaxation import _tan_range

    near_asymptote = np.pi / 2.0 - 5e-4

    assert _tan_range(near_asymptote - 1e-5, near_asymptote) is None


def test_issue71_milp_wrapper_accepts_nonfatal_highs_warnings():
    """HiGHS passModel warnings from tiny AMP rows must not discard valid bounds."""
    from discopt.solvers import SolveStatus
    from discopt.solvers.milp_highs import solve_milp

    result = solve_milp(
        c=np.array([1.0], dtype=np.float64),
        A_ub=np.array([[1e-15]], dtype=np.float64),
        b_ub=np.array([1e-30], dtype=np.float64),
        bounds=[(1e-30, 1.0)],
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.objective is not None


def test_milp_wrapper_bails_on_fatal_run_status(monkeypatch):
    """A fatal HiGHS run status must not fall through to getSolution()."""
    from discopt.solvers import SolveStatus, milp_highs

    class FakeHighs:
        def setOptionValue(self, *args):
            pass

        def passModel(self, lp):
            del lp
            return milp_highs.highspy.HighsStatus.kOk

        def run(self):
            return milp_highs.highspy.HighsStatus.kError

        def getModelStatus(self):
            return milp_highs.highspy.HighsModelStatus.kOptimal

        def getSolution(self):
            raise AssertionError("getSolution should not be called after fatal run status")

    monkeypatch.setattr(milp_highs.highspy, "Highs", FakeHighs)

    result = milp_highs.solve_milp(c=np.array([1.0], dtype=np.float64), bounds=[(0.0, 1.0)])

    assert result.status == SolveStatus.ERROR


def test_tan_abs_minlptests_objective_linearizes_without_fallback(caplog):
    """The nlp_004-style tan/abs objective should keep a valid MILP objective."""
    m = Model("tan_abs_obj")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-4.0, ub=4.0)
    z = m.continuous("z", lb=-4.0, ub=4.0)
    m.minimize(dm.tan(x) + y + x * z + 0.5 * dm.abs(y))
    m.subject_to(x**2 + y**2 + z**2 <= 10.0)
    m.subject_to(-1.2 * x - y <= z / 1.35)

    with caplog.at_level(logging.WARNING):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    assert result.status == "optimal"
    assert result.objective is not None
    assert {"abs", "tan"} <= {relax.func_name for relax in varmap["univariate_relaxations"]}
    assert not any(
        "could not linearize the objective" in record.message for record in caplog.records
    )


def test_mixed_curvature_tan_relaxation_respects_fixed_argument():
    """Piecewise tan envelopes should tighten a mixed-curvature lifted objective."""
    m = Model("tan_fixed_arg")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(dm.tan(x))
    m.subject_to(x == 0.0)

    milp_model, varmap = _build_relaxation_for_test(m)
    result = milp_model.solve()

    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-8)
    piecewise = varmap["univariate_piecewise_relaxations"]
    assert [relax.relax.func_name for relax in piecewise] == ["tan"]
    assert {interval.curvature for interval in piecewise[0].intervals} == {"concave", "convex"}


def test_disaggregated_piecewise_bilinear_big_m_keeps_negative_endpoint_feasible():
    """Inactive piecewise McCormick rows need enough slack on negative intervals."""
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    z_value = 2.85671038
    z_bound = float(np.sqrt(10.0))
    m = Model("fixed_negative_bilinear")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    z = m.continuous("z", lb=-z_bound, ub=z_bound)
    m.minimize(x * z)
    m.subject_to(x == -1.0)
    m.subject_to(z == z_value)

    state = DiscretizationState(
        partitions={
            0: np.array([-1.0, -0.99, -0.9, 0.0, 1.0], dtype=np.float64),
            1: np.array([-z_bound, 0.0, 2.687936011, 2.956729612, z_bound]),
        }
    )
    milp_model, _ = build_milp_relaxation(
        m,
        classify_nonlinear_terms(m),
        state,
        incumbent=None,
        convhull_formulation="disaggregated",
    )
    result = milp_model.solve()

    assert result.status == "optimal"
    assert result.objective == pytest.approx(-z_value, abs=1e-8)


def test_affine_trig_constraints_are_retained_in_relaxation(caplog):
    """Affine-argument sin/cos constraints should be lifted instead of omitted."""
    m = Model("trig_affine_constraints")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(-x - y)
    m.subject_to(dm.sin(-x - 1.0) + x / 2 + 0.5 <= y)
    m.subject_to(dm.cos(x - 0.5) + x / 4 - 0.5 >= y)

    with caplog.at_level(logging.WARNING, logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    funcs = {relax.func_name for relax in varmap["univariate_relaxations"]}
    assert {"sin", "cos"} <= funcs
    assert "omitting constraint" not in caplog.text
    assert result.status == "optimal"
    assert result.objective is not None


@pytest.mark.parametrize(
    ("objective", "old_range_bound"),
    [
        ("neg_sum", -4.0),
        ("sum", -4.0),
    ],
)
def test_mixed_curvature_affine_trig_uses_piecewise_relaxation(objective, old_range_bound):
    """106-style mixed-curvature sin/cos constraints should not be range-only."""
    m = Model(f"trig_affine_piecewise_{objective}")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    if objective == "neg_sum":
        m.minimize(-x - y)
    else:
        m.minimize(x + y)
    m.subject_to(dm.sin(-x - 1.0) + x / 2 + 0.5 <= y)
    m.subject_to(dm.cos(x - 0.5) + x / 4 - 0.5 >= y)

    milp_model, varmap = _build_relaxation_for_test(m)
    result = milp_model.solve()

    piecewise = varmap["univariate_piecewise_relaxations"]
    assert {relax.relax.func_name for relax in piecewise} == {"sin", "cos"}
    assert all(len(relax.intervals) > 2 for relax in piecewise)
    assert all(
        interval.curvature in {"convex", "concave"}
        for relax in piecewise
        for interval in relax.intervals
    )
    assert result.status == "optimal"
    assert result.objective is not None
    assert result.objective > old_range_bound + 1e-6


def test_trig_piecewise_relaxation_caps_dense_partitions():
    """Dense AMP partitions should not be copied into oversized trig MILPs."""
    from discopt._jax.milp_relaxation import _MAX_TRIG_PIECEWISE_INTERVALS

    m = Model("trig_dense_partition_guard")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(y)
    m.subject_to(dm.sin(x) <= y)

    _milp_model, varmap = _build_relaxation_for_test(
        m,
        part_vars=[0],
        lbs=[-1.0],
        ubs=[1.0],
        n_init=96,
    )

    piecewise = varmap["univariate_piecewise_relaxations"]
    assert len(piecewise) == 1
    assert piecewise[0].relax.func_name == "sin"
    assert len(piecewise[0].intervals) <= _MAX_TRIG_PIECEWISE_INTERVALS
    assert len(piecewise[0].intervals) < 96


def test_dense_bilinear_partitions_fall_back_to_global_relaxation(caplog):
    """Oversized bilinear partition refinements should keep the global McCormick path."""
    from discopt._jax.milp_relaxation import _MAX_RELAXATION_PARTITION_INTERVALS

    m = Model("dense_bilinear_guard")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x * y)

    with caplog.at_level(logging.DEBUG, logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(
            m,
            part_vars=[0],
            lbs=[0.0],
            ubs=[1.0],
            n_init=_MAX_RELAXATION_PARTITION_INTERVALS + 1,
        )

    assert (0, 1) in varmap["bilinear"]
    assert varmap["bilinear_pw"] == {}
    assert varmap["bilinear_lambda"] == {}
    assert any("bilinear piecewise" in note for note in varmap["generation_guardrails"])
    assert "skipped bilinear piecewise refinement" in caplog.text
    assert milp_model._A_ub.shape[0] < 20


def test_dense_monomial_partitions_use_coarse_global_relaxation(caplog):
    """Oversized monomial partitions should avoid allocating one binary per interval."""
    from discopt._jax.milp_relaxation import _MAX_RELAXATION_PARTITION_INTERVALS

    m = Model("dense_monomial_guard")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(x**2)

    with caplog.at_level(logging.DEBUG, logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(
            m,
            part_vars=[0],
            lbs=[-1.0],
            ubs=[1.0],
            n_init=_MAX_RELAXATION_PARTITION_INTERVALS + 1,
        )

    assert (0, 2) in varmap["monomial"]
    assert varmap["monomial_pw"] == {}
    assert any("monomial piecewise" in note for note in varmap["generation_guardrails"])
    assert any("monomial tangent" in note for note in varmap["generation_guardrails"])
    assert "skipped monomial piecewise refinement" in caplog.text
    assert milp_model._A_ub.shape[0] < 20


def test_trig_piecewise_relaxation_skips_huge_argument_span():
    """Very wide trig spans should use range bounds, not many piecewise rows."""
    from discopt._jax.milp_relaxation import _MAX_TRIG_PIECEWISE_SPAN

    span = 2.0 * _MAX_TRIG_PIECEWISE_SPAN
    m = Model("trig_huge_span_guard")
    x = m.continuous("x", lb=-span / 2.0, ub=span / 2.0)
    m.minimize(dm.sin(x))

    milp_model, varmap = _build_relaxation_for_test(m)
    result = milp_model.solve()

    assert varmap["univariate_piecewise_relaxations"] == []
    assert {relax.func_name for relax in varmap["univariate_relaxations"]} == {"sin"}
    assert milp_model._objective_bound_valid is True
    assert result.status == "optimal"
    assert result.objective == pytest.approx(-1.0)


def test_trig_square_constraints_apply_range_bounds():
    """sin(x)^2 and cos(y)^2 constraints should constrain the MILP relaxation."""
    sin_model = Model("sin_square_relax")
    x = sin_model.integer("x", lb=0, ub=4)
    y = sin_model.integer("y", lb=0, ub=4)
    sin_model.maximize(y)
    sin_model.subject_to(y <= dm.sin(x) ** 2 + 2)

    sin_relax, sin_varmap = _build_relaxation_for_test(sin_model)
    sin_result = sin_relax.solve()

    assert len(sin_varmap["univariate_square_relaxations"]) == 1
    assert sin_varmap["univariate_square_piecewise_relaxations"] == []
    assert len(sin_varmap["finite_domain_trig_square_tables"]) == 1
    assert sin_result.status == "optimal"
    assert sin_result.x is not None
    assert sin_result.x[1] <= 3.0 + 1e-8

    cos_model = Model("cos_square_relax")
    z = cos_model.integer("z", lb=1, ub=4)
    b = cos_model.binary("b")
    cos_model.maximize(z)
    cos_model.subject_to(z <= dm.cos(b) ** 2 + 1.5)

    cos_relax, cos_varmap = _build_relaxation_for_test(cos_model)
    cos_result = cos_relax.solve()

    assert len(cos_varmap["univariate_square_relaxations"]) == 1
    assert cos_varmap["univariate_square_piecewise_relaxations"] == []
    assert len(cos_varmap["finite_domain_trig_square_tables"]) == 1
    assert cos_result.status == "optimal"
    assert cos_result.x is not None
    assert cos_result.x[0] <= 2.0 + 1e-8


def test_finite_domain_trig_square_tables_link_integer_arguments_exactly():
    """Small integer trig-square arguments should use selector value tables."""
    sin_model = Model("issue72_sin_square_table")
    x = sin_model.integer("x", lb=0, ub=4)
    y = sin_model.continuous("y", lb=0, ub=4)
    sin_model.maximize(10 * x + y)
    sin_model.subject_to(y <= dm.sin(x) ** 2 + 2)

    sin_relax, sin_varmap = _build_relaxation_for_test(sin_model)
    sin_result = sin_relax.solve()

    sin_tables = sin_varmap["finite_domain_trig_square_tables"]
    assert len(sin_tables) == 1
    assert sin_tables[0].func_name == "sin"
    assert sin_tables[0].domain_values == [0, 1, 2, 3, 4]
    assert len(sin_tables[0].selector_cols) == 5
    assert sin_result.status == "optimal"
    assert sin_result.objective == pytest.approx(-(40.0 + np.sin(4.0) ** 2 + 2.0), abs=1e-8)

    cos_model = Model("issue72_cos_square_table")
    z = cos_model.integer("z", lb=1, ub=4)
    b = cos_model.binary("b")
    cos_model.maximize(z + 10 * b)
    cos_model.subject_to(z <= dm.cos(b) ** 2 + 1.5)

    cos_relax, cos_varmap = _build_relaxation_for_test(cos_model)
    cos_result = cos_relax.solve()

    cos_tables = cos_varmap["finite_domain_trig_square_tables"]
    assert len(cos_tables) == 1
    assert cos_tables[0].func_name == "cos"
    assert cos_tables[0].domain_values == [0, 1]
    assert len(cos_tables[0].selector_cols) == 2
    assert cos_result.status == "optimal"
    assert cos_result.objective == pytest.approx(-11.0, abs=1e-8)


@pytest.mark.parametrize(("func_name", "func"), [("sin", dm.sin), ("cos", dm.cos)])
def test_continuous_trig_square_uses_direct_piecewise_relaxation(func_name, func):
    """Continuous trig-square constraints should not relax to the range-only q <= 1."""
    m = Model(f"{func_name}_square_continuous_piecewise")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(x + y)
    m.subject_to(y <= func(x) ** 2 + 2.0)

    milp_model, varmap = _build_relaxation_for_test(
        m,
        part_vars=[0],
        lbs=[0.0],
        ubs=[4.0],
        n_init=4,
    )
    result = milp_model.solve()

    piecewise = varmap["univariate_square_piecewise_relaxations"]
    assert len(piecewise) == 1
    assert piecewise[0].func_name == func_name
    assert len(piecewise[0].intervals) > 2
    assert all(interval.curvature in {"convex", "concave"} for interval in piecewise[0].intervals)
    assert result.status == "optimal"
    assert result.x is not None
    assert float(result.x[0] + result.x[1]) < 6.95


def test_safe_tan_objective_keeps_relaxation_bound():
    """tan(x) should be lifted when the argument interval avoids asymptotes."""
    m = Model("safe_tan_objective")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(dm.tan(x))

    milp_model, varmap = _build_relaxation_for_test(m)
    result = milp_model.solve()

    assert {relax.func_name for relax in varmap["univariate_relaxations"]} == {"tan"}
    assert result.status == "optimal"
    assert result.objective == pytest.approx(float(np.tan(-1.0)), abs=1e-8)


def test_unsafe_tan_objective_still_falls_back(caplog):
    """tan(x) intervals crossing an asymptote should remain unsupported."""
    m = Model("unsafe_tan_objective")
    x = m.continuous("x", lb=1.0, ub=2.0)
    m.minimize(dm.tan(x))

    with caplog.at_level(logging.WARNING, logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    assert all(relax.func_name != "tan" for relax in varmap["univariate_relaxations"])
    assert milp_model._objective_bound_valid is False
    assert result.status == "optimal"
    assert result.objective is None
    assert "could not linearize the objective" in caplog.text


def test_x_exp_objective_uses_lifted_product_relaxation(caplog):
    """Finite-box x*exp(x) objectives should use exp lift plus McCormick product."""
    m = Model("x_exp_finite")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(x * dm.exp(x))

    with caplog.at_level("WARNING", logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    exp_cols = [
        relax.aux_col for relax in varmap["univariate_relaxations"] if relax.func_name == "exp"
    ]

    assert len(exp_cols) == 1
    assert (0, exp_cols[0]) in varmap["bilinear"]
    assert milp_model._objective_bound_valid is True
    assert result.status == "optimal"
    assert result.objective is not None
    assert result.objective <= -1.0 / np.e + 1e-8
    assert "falling back to a feasibility objective" not in caplog.text


@pytest.mark.parametrize("integer_y", [False, True])
def test_x_exp_minlptests_objective_uses_separable_lower_bound(integer_y, caplog):
    """Unbounded MINLPTests x*exp(x)+cos(y)+z^3-z^2 gets a safe constant bound."""
    m = Model("nlp_001_010_like")
    x = m.continuous("x")
    y = m.integer("y", lb=1, ub=10) if integer_y else m.continuous("y")
    z = m.continuous("z", lb=1.0)
    m.minimize(x * dm.exp(x) + dm.cos(y) + z**3 - z**2)

    with caplog.at_level("WARNING", logger="discopt._jax.milp_relaxation"):
        milp_model, varmap = _build_relaxation_for_test(m)
        result = milp_model.solve()

    assert varmap["univariate_piecewise_relaxations"] == []
    assert milp_model._objective_bound_valid is True
    assert result.status == "optimal"
    cos_lb = min(np.cos(np.arange(1, 11))) if integer_y else -1.0
    assert result.objective == pytest.approx(-1.0 / np.e + cos_lb)
    assert "falling back to a feasibility objective" not in caplog.text


@pytest.mark.parametrize("scale", [1.0, -1.0])
def test_integer_affine_cos_objective_uses_discrete_separable_lower_bound(scale):
    """Finite integer affine cos terms should use their exact enumerated range."""
    m = Model("integer_affine_cos")
    y = m.integer("y", lb=-2, ub=3)
    expr = dm.cos(2.0 * y + 1.0)
    m.minimize(expr if scale > 0 else -expr)

    milp_model, _ = _build_relaxation_for_test(m)
    result = milp_model.solve()

    values = scale * np.cos(2.0 * np.arange(-2, 4) + 1.0)
    assert milp_model._objective_bound_valid is True
    assert result.status == "optimal"
    assert result.objective == pytest.approx(float(np.min(values)))


def test_negative_unbounded_x_exp_objective_keeps_no_bound(caplog):
    """Unsafe x*exp(x) signs must not receive a fake separable lower bound."""
    m = Model("negative_x_exp_unbounded")
    x = m.continuous("x")
    m.minimize(-x * dm.exp(x))

    with caplog.at_level("WARNING", logger="discopt._jax.milp_relaxation"):
        milp_model, _ = _build_relaxation_for_test(m)
        result = milp_model.solve()

    assert milp_model._objective_bound_valid is False
    assert result.status == "optimal"
    assert result.objective is None
    assert "falling back to a feasibility objective" in caplog.text


def test_nested_univariate_objective_still_returns_no_relaxation_bound():
    """Unsupported nested operator arguments should keep the safe no-bound behavior."""
    m = Model("nested_sqrt")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(dm.sqrt(x**2 + 1.0))

    milp_model, varmap = _build_relaxation_for_test(
        m,
        part_vars=[0],
        lbs=[-2.0],
        ubs=[2.0],
    )
    result = milp_model.solve()

    assert varmap["univariate_relaxations"] == []
    assert result.status == "optimal"
    assert result.objective is None
    assert result.x is not None


def test_solve_model_forwards_alpine_amp_aliases(monkeypatch):
    """solve_model should pass Alpine-style AMP aliases through to solve_amp."""
    from discopt.solver import solve_model
    from discopt.solvers import amp as amp_mod

    captured = {}

    def fake_solve_amp(model, **kwargs):
        del model
        captured.update(kwargs)
        return SolveResult(status="infeasible", wall_time=0.0)

    monkeypatch.setattr(amp_mod, "solve_amp", fake_solve_amp)

    m = Model("alias_forwarding")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)

    def update_scaling(context):
        del context
        return 8.0

    solve_model(
        m,
        solver="amp",
        gap_tolerance=1e-3,
        apply_partitioning=False,
        disc_var_pick=1,
        partition_scaling_factor=7.0,
        partition_scaling_factor_update=update_scaling,
        disc_add_partition_method="uniform",
        disc_abs_width_tol=1e-2,
        convhull_formulation="sos2",
        presolve_bt_algo=2,
        presolve_bt_time_limit=12.0,
        presolve_bt_mip_time_limit=0.5,
    )

    assert captured["rel_gap"] == pytest.approx(1e-3)
    assert captured["apply_partitioning"] is False
    assert captured["disc_var_pick"] == 1
    assert captured["partition_scaling_factor"] == pytest.approx(7.0)
    assert captured["partition_scaling_factor_update"] is update_scaling
    assert captured["disc_add_partition_method"] == "uniform"
    assert captured["disc_abs_width_tol"] == pytest.approx(1e-2)
    assert captured["convhull_formulation"] == "sos2"
    assert captured["presolve_bt_algo"] == 2
    assert captured["presolve_bt_time_limit"] == pytest.approx(12.0)
    assert captured["presolve_bt_mip_time_limit"] == pytest.approx(0.5)


def test_amp_custom_partition_hooks_run_inside_amp(monkeypatch):
    """AMP should expose callable selection, scaling, and refinement hooks."""
    import discopt._jax.discretization as disc_mod
    from discopt._jax.discretization import add_adaptive_partition
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    selection_stages = []
    refinement_calls = []
    scaling_calls = []

    def custom_select(context):
        selection_stages.append(context["stage"])
        assert set(context["builtin_pick_partition_vars"]("max_cover")) == {0, 1}
        if context["stage"] == "initial_selection":
            return [0]
        assert "distance" in context
        return [1]

    def custom_scaling(context):
        scaling_calls.append((context["iteration"], context["current_scaling_factor"]))
        return 12.0

    def custom_refine(context):
        refinement_calls.append(
            (
                context["stage"],
                list(context["var_indices"]),
                context["disc_state"].scaling_factor,
            )
        )
        return add_adaptive_partition(
            context["disc_state"],
            context["solution"],
            context["var_indices"],
            context["lb"],
            context["ub"],
        )

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=0.0,
                x=np.array([2.0, 4.0], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([2.0, 4.0], dtype=np.float64), 1.0),
    )
    monkeypatch.setattr(disc_mod, "check_partition_convergence", lambda state: True)

    result = amp_mod.solve_amp(
        _make_nlp1(),
        disc_var_pick=custom_select,
        partition_scaling_factor=10.0,
        partition_scaling_factor_update=custom_scaling,
        disc_add_partition_method=custom_refine,
        presolve_bt=False,
        skip_convex_check=True,
        rel_gap=1e-6,
        max_iter=2,
        time_limit=30,
    )

    assert selection_stages == ["initial_selection", "iteration_selection"]
    assert scaling_calls == [(1, 10.0)]
    assert refinement_calls == [("refinement", [1], 12.0)]
    assert result.status == "feasible"
    assert result.gap_certified is False


def test_amp_adaptive_keeps_monomial_fallback_partitions(monkeypatch):
    """Built-in adaptive selection must not erase monomial-only partitions."""
    import discopt._jax.discretization as disc_mod
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    refined_var_sets = []

    def fake_add_adaptive_partition(state, solution, var_indices, lb, ub):
        del solution, lb, ub
        refined_var_sets.append(list(var_indices))
        return state

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=0.0,
                x=np.array([0.0], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([1.0], dtype=np.float64), 1.0),
    )
    monkeypatch.setattr(disc_mod, "add_adaptive_partition", fake_add_adaptive_partition)
    monkeypatch.setattr(disc_mod, "check_partition_convergence", lambda state: True)

    m = Model("monomial_only_adaptive")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(x**2)

    result = amp_mod.solve_amp(
        m,
        disc_var_pick="adaptive",
        presolve_bt=False,
        skip_convex_check=True,
        rel_gap=1e-6,
        max_iter=2,
        time_limit=30,
    )

    assert refined_var_sets == [[0]]
    assert result.status == "feasible"
    assert result.gap_certified is False


@pytest.mark.memory_heavy
def test_amp_adaptive_refines_weymouth_like_monomials(monkeypatch):
    """Adaptive selection should keep square-balance variables when products also exist."""
    import discopt._jax.discretization as disc_mod
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    refined_var_sets = []

    def fake_add_adaptive_partition(state, solution, var_indices, lb, ub):
        del solution, lb, ub
        refined_var_sets.append(list(var_indices))
        return state

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=0.0,
                x=np.array([0.2, 0.3, 0.4, 0.4], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float64), 1.0),
    )
    monkeypatch.setattr(disc_mod, "add_adaptive_partition", fake_add_adaptive_partition)
    monkeypatch.setattr(disc_mod, "check_partition_convergence", lambda state: True)

    m = Model("weymouth_like_adaptive")
    x = m.continuous("x", lb=0.0, ub=2.0, shape=(4,))
    m.minimize(x[0] * x[1])
    m.subject_to(x[2] ** 2 == x[3] ** 2)

    result = amp_mod.solve_amp(
        m,
        disc_var_pick="adaptive",
        presolve_bt=False,
        skip_convex_check=True,
        rel_gap=1e-6,
        max_iter=2,
        time_limit=30,
    )

    assert refined_var_sets == [[0, 1, 2, 3]]
    assert result.status == "feasible"
    assert result.gap_certified is False


def test_partitioned_presolve_obbt_falls_back_without_incumbent(monkeypatch):
    """Alpine-style mode 2 should use LP OBBT when no feasible incumbent exists."""
    import discopt._jax.obbt as obbt_mod
    from discopt._jax.obbt import ObbtResult
    from discopt.solvers import amp as amp_mod

    m = Model("partitioned_obbt_fallback")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)

    calls = []

    def fake_run_obbt(model, lb, ub, time_limit_per_lp, total_time_limit=None):
        del model
        calls.append((time_limit_per_lp, total_time_limit))
        return ObbtResult(
            tightened_lb=lb + 0.25,
            tightened_ub=ub,
            n_lp_solves=2,
            n_tightened=1,
            total_lp_time=0.01,
        )

    def fail_partitioned(*args, **kwargs):
        del args, kwargs
        raise AssertionError("partitioned OBBT should not run without an incumbent")

    monkeypatch.setattr(obbt_mod, "run_obbt", fake_run_obbt)
    monkeypatch.setattr(amp_mod, "_run_partitioned_obbt", fail_partitioned)

    lb, ub, result = amp_mod._run_amp_presolve_bound_tightening(
        m,
        SimpleNamespace(monomial=[]),
        np.array([0.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        presolve_bt_algo=2,
        remaining=10.0,
        incumbent=None,
        incumbent_obj=None,
        n_init_partitions=2,
        partition_mode="auto",
        partition_scaling_factor=10.0,
        disc_abs_width_tol=1e-3,
        convhull_formulation="disaggregated",
        convhull_ebd=False,
        convhull_ebd_encoding="gray",
        milp_gap_tolerance=None,
        presolve_bt_time_limit=None,
        presolve_bt_mip_time_limit=None,
    )

    np.testing.assert_allclose(lb, np.array([0.25]))
    np.testing.assert_allclose(ub, np.array([1.0]))
    assert result.n_tightened == 1
    assert len(calls) == 1
    assert calls[0][1] is not None
    assert 0.0 < calls[0][1] <= 1.0


def test_partitioned_presolve_obbt_uses_feasible_initial_incumbent(monkeypatch):
    """A feasible initial point should seed the partition-aware OBBT path."""
    from discopt._jax.obbt import ObbtResult
    from discopt.solvers import amp as amp_mod

    m = Model("partitioned_obbt_incumbent")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)

    class FakeEvaluator:
        n_constraints = 0

        def evaluate_objective(self, point):
            return float(point[0])

    incumbent, incumbent_obj = amp_mod._presolve_incumbent_from_initial_point(
        np.array([0.4], dtype=np.float64),
        m,
        FakeEvaluator(),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    )

    captured = {}

    def update_scaling(context):
        del context
        return None

    def fake_partitioned_obbt(model, terms, flat_lb, flat_ub, incumbent, incumbent_obj, **kwargs):
        del model, terms
        captured["flat_lb"] = flat_lb.copy()
        captured["flat_ub"] = flat_ub.copy()
        captured["incumbent"] = incumbent.copy()
        captured["incumbent_obj"] = incumbent_obj
        captured["partition_scaling_factor_update"] = kwargs["partition_scaling_factor_update"]
        return ObbtResult(
            tightened_lb=np.array([0.2], dtype=np.float64),
            tightened_ub=flat_ub.copy(),
            n_lp_solves=2,
            n_tightened=1,
            total_lp_time=0.02,
        )

    monkeypatch.setattr(amp_mod, "_run_partitioned_obbt", fake_partitioned_obbt)

    lb, ub, result = amp_mod._run_amp_presolve_bound_tightening(
        m,
        SimpleNamespace(monomial=[]),
        np.array([0.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        presolve_bt_algo="incumbent_partitioned",
        remaining=10.0,
        incumbent=incumbent,
        incumbent_obj=incumbent_obj,
        n_init_partitions=2,
        partition_mode="auto",
        partition_scaling_factor=10.0,
        disc_abs_width_tol=1e-3,
        convhull_formulation="disaggregated",
        convhull_ebd=False,
        convhull_ebd_encoding="gray",
        milp_gap_tolerance=None,
        presolve_bt_time_limit=2.0,
        presolve_bt_mip_time_limit=0.5,
        partition_scaling_factor_update=update_scaling,
    )

    np.testing.assert_allclose(captured["incumbent"], np.array([0.4]))
    assert captured["incumbent_obj"] == pytest.approx(0.4)
    assert captured["partition_scaling_factor_update"] is update_scaling
    np.testing.assert_allclose(lb, np.array([0.2]))
    np.testing.assert_allclose(ub, np.array([1.0]))
    assert result.n_tightened == 1


def test_partitioned_obbt_applies_scaling_update_before_custom_refinement(monkeypatch):
    """Partitioned OBBT should honor the scaling hook before custom refinement."""
    import discopt._jax.milp_relaxation as milp_mod
    from discopt.solvers import amp as amp_mod

    m = _make_obbt_demo()
    terms = SimpleNamespace(
        partition_candidates=[0, 1],
        bilinear=[(0, 1)],
        trilinear=[],
        multilinear=[],
        monomial=[],
    )
    scaling_calls = []
    refinement_calls = []

    def fake_build_milp_relaxation(*args, **kwargs):
        del args, kwargs
        return (
            SimpleNamespace(
                _A_ub=np.zeros((0, 2), dtype=np.float64),
                _b_ub=np.zeros(0, dtype=np.float64),
                _objective_bound_valid=False,
                _c=np.zeros(2, dtype=np.float64),
                _obj_offset=0.0,
                _bounds=[(0.0, 1.0), (0.0, 1.0)],
                _integrality=np.zeros(2, dtype=np.int32),
            ),
            {},
        )

    def update_scaling(context):
        scaling_calls.append(
            (
                context["stage"],
                context["current_scaling_factor"],
                context["disc_state"].scaling_factor,
            )
        )
        return 14.0

    def refine_partitions(context):
        refinement_calls.append(
            (
                context["stage"],
                context["partition_scaling_factor"],
                context["disc_state"].scaling_factor,
                list(context["var_indices"]),
            )
        )
        return context["disc_state"]

    monkeypatch.setattr(milp_mod, "build_milp_relaxation", fake_build_milp_relaxation)

    result = amp_mod._run_partitioned_obbt(
        m,
        terms,
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        0.25,
        partition_mode="auto",
        n_init_partitions=2,
        partition_scaling_factor=10.0,
        partition_scaling_factor_update=update_scaling,
        disc_abs_width_tol=1e-3,
        convhull_formulation="disaggregated",
        convhull_ebd=False,
        convhull_ebd_encoding="gray",
        total_time_limit=0.0,
        time_limit_per_mip=0.1,
        gap_tolerance=1e-4,
        disc_add_partition_hook=refine_partitions,
    )

    assert scaling_calls == [("presolve_obbt_refinement", 10.0, 10.0)]
    assert refinement_calls == [("presolve_obbt_refinement", 14.0, 14.0, [0, 1])]
    assert result.n_lp_solves == 0
    assert result.n_tightened == 0


def test_partitioned_presolve_obbt_runs_on_bilinear_demo():
    """The real partition-aware OBBT path should solve bounded MILP subproblems."""
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt.solvers import amp as amp_mod

    m = _make_obbt_demo()
    incumbent = np.array([0.5, 0.5], dtype=np.float64)
    evaluator = NLPEvaluator(m)
    incumbent_obj = float(evaluator.evaluate_objective(incumbent))
    terms = classify_nonlinear_terms(m)

    lb, ub, result = amp_mod._run_amp_presolve_bound_tightening(
        m,
        terms,
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([10.0, 10.0], dtype=np.float64),
        presolve_bt_algo=2,
        remaining=5.0,
        incumbent=incumbent,
        incumbent_obj=incumbent_obj,
        n_init_partitions=2,
        partition_mode="auto",
        partition_scaling_factor=10.0,
        disc_abs_width_tol=1e-3,
        convhull_formulation="disaggregated",
        convhull_ebd=False,
        convhull_ebd_encoding="gray",
        milp_gap_tolerance=None,
        presolve_bt_time_limit=1.0,
        presolve_bt_mip_time_limit=0.2,
    )

    assert result.n_lp_solves == 4
    assert result.n_tightened > 0
    assert np.all(lb >= np.array([0.0, 0.0]) - 1e-9)
    assert np.all(ub <= np.array([10.0, 10.0]) + 1e-9)
    assert np.all(lb <= ub)


def test_partitioned_presolve_obbt_maximize_cutoff_uses_relaxation_objective_space(
    monkeypatch,
):
    """Maximization incumbents should be converted to the relaxation minimization space."""
    import scipy.sparse as sp
    from discopt._jax.milp_relaxation import MilpRelaxationModel, MilpRelaxationResult
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt.solvers import amp as amp_mod

    m = _make_obbt_demo()
    incumbent = np.array([0.5, 0.5], dtype=np.float64)
    incumbent_obj = float(NLPEvaluator(m).evaluate_objective(incumbent))
    terms = classify_nonlinear_terms(m)
    captured = {}

    def fake_solve(self, time_limit=None, gap_tolerance=1e-4):
        del time_limit, gap_tolerance
        if "cutoff_row" not in captured:
            A_ub = self._A_ub
            row = A_ub[-1].toarray().ravel() if sp.issparse(A_ub) else np.asarray(A_ub[-1])
            captured["cutoff_row"] = row
            captured["cutoff_rhs"] = float(self._b_ub[-1])
        return MilpRelaxationResult(status="time_limit")

    monkeypatch.setattr(MilpRelaxationModel, "solve", fake_solve)

    amp_mod._run_partitioned_obbt(
        m,
        terms,
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([10.0, 10.0], dtype=np.float64),
        incumbent,
        incumbent_obj,
        partition_mode="auto",
        n_init_partitions=2,
        partition_scaling_factor=10.0,
        disc_abs_width_tol=1e-3,
        convhull_formulation="disaggregated",
        convhull_ebd=False,
        convhull_ebd_encoding="gray",
        total_time_limit=1.0,
        time_limit_per_mip=0.1,
        gap_tolerance=1e-4,
    )

    nonzero_cutoff = captured["cutoff_row"][np.abs(captured["cutoff_row"]) > 1e-12]
    np.testing.assert_allclose(nonzero_cutoff, np.array([-1.0]))
    expected_rhs = -incumbent_obj + 1e-8 * max(1.0, abs(incumbent_obj))
    assert captured["cutoff_rhs"] == pytest.approx(expected_rhs)


def test_amp_accepts_feasible_start_as_incumbent(monkeypatch):
    """A feasible model start should survive when proof search fails immediately."""
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    m = Model("amp_start_incumbent")
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.subject_to(x >= 0.25)
    m.minimize((x - 1.0) ** 2)

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(status="error", objective=None, x=None),
            {},
            [],
            1,
        ),
    )

    result = m.solve(
        solver="amp",
        initial_solution={x: 0.5},
        use_start_as_incumbent=True,
        skip_convex_check=True,
        presolve_bt=False,
        max_iter=1,
        time_limit=30,
    )

    assert result.status == "feasible"
    assert result.objective == pytest.approx(0.25)
    assert result.x is not None
    assert np.asarray(result.x["x"]).item() == pytest.approx(0.5)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_amp_rejects_nonfinite_direct_initial_point(bad_value):
    """Direct AMP initial points must be finite before incumbent checks."""
    from discopt.solvers import amp as amp_mod

    m = Model("amp_nonfinite_start")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)

    with pytest.raises(ValueError, match="finite"):
        amp_mod.solve_amp(
            m,
            initial_point=np.array([bad_value], dtype=np.float64),
            use_start_as_incumbent=True,
            skip_convex_check=True,
            presolve_bt=False,
            max_iter=1,
            time_limit=1.0,
        )


@pytest.mark.memory_heavy
def test_amp_does_not_accept_start_with_nonfinite_objective(monkeypatch):
    """A finite start with NaN objective is not a valid AMP incumbent."""
    import discopt._jax.nlp_evaluator as nlp_eval
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    m = Model("amp_nan_objective_start")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)

    monkeypatch.setattr(nlp_eval.NLPEvaluator, "evaluate_objective", lambda self, x_flat: np.nan)
    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(status="error", objective=None, x=None),
            {},
            [],
            1,
        ),
    )

    result = m.solve(
        solver="amp",
        initial_solution={x: 0.5},
        use_start_as_incumbent=True,
        skip_convex_check=True,
        presolve_bt=False,
        max_iter=1,
        time_limit=1.0,
    )

    assert result.status == "error"
    assert result.objective is None
    assert result.x is None


def test_integer_rounding_candidates_include_floor_and_ceil():
    """Nearest-integer rounding fallback must try floor and ceil alternatives."""
    from discopt.solvers import amp as amp_mod

    m = Model("rounding")
    m.integer("y", lb=0, ub=3, shape=(2,))
    x0 = np.array([1.49, 1.51], dtype=np.float64)

    candidates = amp_mod._integer_rounding_candidates(x0, m)
    rounded = {tuple(float(v) for v in cand) for cand in candidates}

    assert (1.0, 2.0) in rounded
    assert (1.0, 1.0) in rounded
    assert (2.0, 2.0) in rounded


def test_integer_rounding_candidates_enumerate_small_finite_domains():
    """Small integer boxes should be enumerated before falling back to local neighbors."""
    from discopt.solvers import amp as amp_mod

    m = Model("rounding_box")
    m.integer("y", lb=0, ub=4, shape=(2,))

    candidates = amp_mod._integer_rounding_candidates(np.array([4.0, 4.0]), m)
    rounded = {tuple(float(v) for v in cand) for cand in candidates}

    assert len(candidates) == 25
    assert (3.0, 2.0) in rounded


def test_integer_rounding_candidates_cover_continuous_and_large_domains():
    """Rounding helpers should handle continuous models and large integer boxes."""
    from discopt.solvers import amp as amp_mod

    continuous = Model("continuous_rounding")
    continuous.continuous("x", lb=0, ub=1)
    base = np.array([0.25], dtype=np.float64)
    continuous_candidates = amp_mod._integer_rounding_candidates(base, continuous)

    assert len(continuous_candidates) == 1
    np.testing.assert_allclose(continuous_candidates[0], base)

    large = Model("large_integer_rounding")
    large.integer("y", lb=0, ub=10, shape=(3,))
    large_candidates = amp_mod._integer_rounding_candidates(
        np.array([5.2, 5.2, 5.2], dtype=np.float64),
        large,
        max_candidates=4,
    )

    assert len(large_candidates) == 4
    np.testing.assert_allclose(large_candidates[0], np.array([5.0, 5.0, 5.0]))
    np.testing.assert_allclose(
        amp_mod._round_integers(np.array([2.7, 3.2, 4.6]), large),
        np.array([3.0, 3.0, 5.0]),
    )


def test_build_fixed_integer_bounds_clamps_to_integer_domain():
    """Rounded fixed bounds should stay within the realizable integer domain."""
    from discopt.solvers import amp as amp_mod

    m = Model("fixed_bounds_clamp")
    m.integer("y", lb=0.2, ub=2.6)

    nlp_lb, nlp_ub = amp_mod._build_fixed_integer_bounds(
        np.array([2.6], dtype=np.float64),
        m,
        flat_lb=np.array([0.2], dtype=np.float64),
        flat_ub=np.array([2.6], dtype=np.float64),
    )

    assert nlp_lb[0] == pytest.approx(2.0)
    assert nlp_ub[0] == pytest.approx(2.0)


def test_best_nlp_candidate_chooses_lowest_feasible_objective(monkeypatch):
    """Integer rounding fallback should keep the best feasible NLP candidate."""
    from discopt.solvers import amp as amp_mod

    m = Model("best_candidate")
    m.integer("y", lb=0, ub=2)

    candidates = [
        np.array([0.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([2.0], dtype=np.float64),
    ]
    objectives = {0.0: 4.0, 1.0: 1.5, 2.0: 3.0}

    monkeypatch.setattr(
        amp_mod,
        "_integer_rounding_candidates",
        lambda x0, model: [cand.copy() for cand in candidates],
    )
    monkeypatch.setattr(
        amp_mod,
        "_build_fixed_integer_bounds",
        lambda x0, model, flat_lb, flat_ub: (flat_lb.copy(), flat_ub.copy()),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_nlp_subproblem",
        lambda evaluator, x0, lb, ub, nlp_solver, time_limit=None: (
            x0.copy(),
            objectives[float(x0[0])],
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_check_constraints_with_evaluator",
        lambda evaluator, x, lb_g, ub_g: True,
    )

    best_x, best_obj = amp_mod._solve_best_nlp_candidate(
        np.array([0.3], dtype=np.float64),
        m,
        evaluator=object(),
        flat_lb=np.array([0.0], dtype=np.float64),
        flat_ub=np.array([2.0], dtype=np.float64),
        constraint_lb=np.array([], dtype=np.float64),
        constraint_ub=np.array([], dtype=np.float64),
        nlp_solver="ipm",
    )

    assert best_x is not None
    assert float(best_x[0]) == pytest.approx(1.0)
    assert best_obj == pytest.approx(1.5)


def test_best_nlp_candidate_prioritizes_incumbent_start_then_model_start_then_milp(
    monkeypatch,
):
    """AMP local search should try incumbent, model start, then MILP point first."""
    from discopt.solvers import amp as amp_mod

    m = Model("candidate_priority")
    m.continuous("x", lb=0.0, ub=10.0)
    seen_starts = []

    monkeypatch.setattr(
        amp_mod,
        "_solve_nlp_subproblem",
        lambda evaluator, x0, lb, ub, nlp_solver, time_limit=None: (
            seen_starts.append(float(x0[0])) or (None, None)
        ),
    )

    amp_mod._solve_best_nlp_candidate(
        np.array([6.0], dtype=np.float64),
        m,
        evaluator=object(),
        flat_lb=np.array([0.0], dtype=np.float64),
        flat_ub=np.array([10.0], dtype=np.float64),
        constraint_lb=np.array([], dtype=np.float64),
        constraint_ub=np.array([], dtype=np.float64),
        nlp_solver="ipm",
        incumbent=np.array([2.0], dtype=np.float64),
        initial_point=np.array([4.0], dtype=np.float64),
    )

    assert seen_starts[:3] == [2.0, 4.0, 6.0]


def test_best_nlp_candidate_rejects_noninteger_nlp_return(monkeypatch):
    """NLP candidates that violate integrality should be discarded."""
    from discopt.solvers import amp as amp_mod

    m = Model("noninteger_candidate")
    m.integer("y", lb=0, ub=2)

    monkeypatch.setattr(
        amp_mod,
        "_integer_rounding_candidates",
        lambda x0, model: [np.array([1.0], dtype=np.float64)],
    )
    monkeypatch.setattr(
        amp_mod,
        "_build_fixed_integer_bounds",
        lambda x0, model, flat_lb, flat_ub: (flat_lb.copy(), flat_ub.copy()),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_nlp_subproblem",
        lambda evaluator, x0, lb, ub, nlp_solver, time_limit=None: (
            np.array([1.5], dtype=np.float64),
            1.0,
        ),
    )

    best_x, best_obj = amp_mod._solve_best_nlp_candidate(
        np.array([1.0], dtype=np.float64),
        m,
        evaluator=object(),
        flat_lb=np.array([0.0], dtype=np.float64),
        flat_ub=np.array([2.0], dtype=np.float64),
        constraint_lb=np.array([], dtype=np.float64),
        constraint_ub=np.array([], dtype=np.float64),
        nlp_solver="ipm",
    )

    assert best_x is None
    assert best_obj is None


def test_amp_uses_nonlinear_tightened_partition_bounds(monkeypatch):
    """AMP should initialize partitions from the tightened nonlinear box."""
    import discopt._jax.discretization as disc_mod
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    captured = {}
    real_initialize = disc_mod.initialize_partitions

    def spy_initialize(part_vars, lb, ub, **kwargs):
        captured["part_vars"] = list(part_vars)
        captured["lb"] = list(lb)
        captured["ub"] = list(ub)
        return real_initialize(part_vars, lb=lb, ub=ub, **kwargs)

    monkeypatch.setattr(disc_mod, "initialize_partitions", spy_initialize)
    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(status="error", objective=None, x=None),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(amp_mod, "_recover_pure_continuous_solution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        amp_mod,
        "_solve_small_integer_domain_fallback",
        lambda *args, **kwargs: (None, None),
    )

    m = Model("amp_bt_partition_bounds")
    x = m.continuous("x", lb=-1e20, ub=1e20)
    m.subject_to(x**2 <= 4.0)
    m.minimize(x)

    result = m.solve(solver="amp", skip_convex_check=True, max_iter=1, time_limit=30)

    assert result.status == "error"
    assert captured["part_vars"] == [0]
    assert captured["lb"] == pytest.approx([-2.0])
    assert captured["ub"] == pytest.approx([2.0])


@pytest.mark.parametrize(
    ("kind", "lb", "ub", "expected_rule", "expected_reason"),
    [
        ("quadratic", -10.0, 10.0, "sum_of_squares_upper_bound", "negative upper bound"),
        ("sqrt", 0.0, 10.0, "monotone_function_bounds", "sqrt(argument) cannot be <="),
        ("exp", -10.0, 10.0, "monotone_function_bounds", "exp(argument) cannot be <="),
    ],
)
def test_nonlinear_tightening_reports_issue_28_contradictions(
    kind,
    lb,
    ub,
    expected_rule,
    expected_reason,
):
    """Issue #28 contradictions should return an explicit infeasible status."""
    from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

    m = Model(f"issue_28_{kind}_contradiction")
    x = m.continuous("x", lb=lb, ub=ub)
    if kind == "quadratic":
        m.subject_to(x**2 == -1.0)
    elif kind == "sqrt":
        m.subject_to(dm.sqrt(x) <= -1.0)
    else:
        m.subject_to(dm.exp(x) <= -1.0)
    m.minimize(x * 0.0)

    flat_lb = np.array([lb], dtype=np.float64)
    flat_ub = np.array([ub], dtype=np.float64)
    tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(m, flat_lb, flat_ub)

    assert stats.infeasible is True
    assert expected_rule in stats.applied_rules
    assert expected_reason in (stats.infeasibility_reason or "")
    np.testing.assert_allclose(tightened_lb, flat_lb)
    np.testing.assert_allclose(tightened_ub, flat_ub)


def test_reciprocal_argument_interval_uses_explicit_infeasible_sentinel():
    from discopt._jax import nonlinear_bound_tightening as nbt

    interval = nbt.ReciprocalBoundsRule._argument_interval_for_leq(
        numerator=1.0,
        rhs=0.25,
        arg_lo=1.0,
        arg_hi=2.0,
    )

    assert interval is nbt._RECIPROCAL_INTERVAL_INFEASIBLE


def test_nonlinear_tightening_counts_infinite_bounds_without_warning():
    """Unchanged infinite bounds should not warn while counting tightened entries."""
    import warnings

    from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

    m = Model("infinite_bound_counting")
    x = m.continuous("x", lb=0.0, ub=np.inf)
    y = m.continuous("y", lb=0.0, ub=np.inf)
    m.subject_to(x**2 + y**2 <= 4.0)
    m.minimize(x + y)

    flat_lb = np.array([0.0, 0.0], dtype=np.float64)
    flat_ub = np.array([np.inf, np.inf], dtype=np.float64)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(m, flat_lb, flat_ub)

    assert not captured
    assert stats.n_tightened == 2
    np.testing.assert_allclose(tightened_lb, flat_lb)
    np.testing.assert_allclose(tightened_ub, np.array([2.0, 2.0]))


def test_square_difference_tightens_weymouth_like_upstream_pressure():
    """Rows like f^2 = C*(p_from^2 - p_to^2) imply a lower bound on p_from."""
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

    m = Model("weymouth_bound_tightening")
    f = m.continuous("f", lb=6.0, ub=20.0)
    p_to = m.continuous("p_to", lb=5.0, ub=10.0)
    p_from = m.continuous("p_from", lb=0.0, ub=20.0)
    m.subject_to(f**2 == 4.0 * (p_from**2 - p_to**2))
    m.minimize(p_from)

    flat_lb, flat_ub = flat_variable_bounds(m)
    tightened_lb, _tightened_ub, stats = tighten_nonlinear_bounds(m, flat_lb, flat_ub)

    assert "square_difference_lower_bound" in stats.applied_rules
    assert tightened_lb[2] == pytest.approx(np.sqrt(5.0**2 + 6.0**2 / 4.0))


def test_gas_square_difference_tightening_strengthens_root_relaxation():
    """The gas benchmark should start AMP from a tighter Weymouth pressure box."""
    from discopt._jax.discretization import initialize_partitions
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt.benchmarks.problems.gas_network_minlp import build_gas_network_minlp
    from discopt.solvers import amp as amp_mod

    m = build_gas_network_minlp()
    terms = classify_nonlinear_terms(m)
    raw_lb, raw_ub = flat_variable_bounds(m)
    tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(m, raw_lb, raw_ub)

    assert "square_difference_lower_bound" in stats.applied_rules
    assert tightened_lb[4] >= 45.0

    part_vars = sorted(
        set(terms.partition_candidates)
        | set(amp_mod._equality_square_monomial_partition_candidates(m, terms))
    )

    def root_bound(lb, ub):
        state = initialize_partitions(
            part_vars,
            lb=[float(lb[i]) for i in part_vars],
            ub=[float(ub[i]) for i in part_vars],
            n_init=4,
        )
        milp_model, _varmap = build_milp_relaxation(
            m,
            terms,
            state,
            bound_override=(lb, ub),
        )
        result = milp_model.solve()
        assert result.objective is not None
        return float(result.objective)

    raw_bound = root_bound(raw_lb, raw_ub)
    tightened_bound = root_bound(tightened_lb, tightened_ub)

    assert tightened_bound >= raw_bound + 0.1
    assert tightened_bound > 2.3


def test_oa_cut_recovery_drops_oldest_half(monkeypatch):
    """OA recovery should retry with the oldest half of cuts removed."""
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    call_sizes = []

    class FakeMilpModel:
        def __init__(self, status):
            self.status = status

        def solve(self, time_limit=None, gap_tolerance=None):
            return MilpRelaxationResult(
                status=self.status,
                objective=0.0,
                x=np.zeros(1, dtype=np.float64),
            )

    def fake_build(
        model,
        terms,
        disc_state,
        incumbent,
        oa_cuts=None,
        convhull_formulation="disaggregated",
        convhull_ebd=False,
        convhull_ebd_encoding="gray",
        bound_override=None,
    ):
        del model, terms, disc_state, incumbent, bound_override
        assert convhull_formulation == "disaggregated"
        assert convhull_ebd is False
        assert convhull_ebd_encoding == "gray"
        size = len(oa_cuts or [])
        call_sizes.append(size)
        status = "infeasible" if size >= 4 else "optimal"
        return FakeMilpModel(status), {"dummy": True}

    monkeypatch.setattr("discopt._jax.milp_relaxation.build_milp_relaxation", fake_build)

    result, _, kept_cuts, mip_count = amp_mod._solve_milp_with_oa_recovery(
        model=None,
        terms=None,
        disc_state=None,
        incumbent=None,
        oa_cuts=[("c1", 1), ("c2", 2), ("c3", 3), ("c4", 4)],
        time_limit=1.0,
        gap_tolerance=1e-4,
        convhull_formulation="disaggregated",
        convhull_ebd=False,
        convhull_ebd_encoding="gray",
    )

    assert call_sizes == [4, 2]
    assert kept_cuts == [("c3", 3), ("c4", 4)]
    assert result.status == "optimal"
    assert mip_count == 2


def test_oa_cut_generation_receives_convex_constraint_mask(monkeypatch):
    """Evaluator OA cuts should receive the per-constraint convexity filter."""
    import discopt._jax.convexity as convexity_mod
    import discopt._jax.cutting_planes as cutting_planes
    from discopt._jax.convexity.rules import OACutConvexity
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    recorded_masks = []
    recorded_reasons = []
    classify_calls = []

    def fake_classify(model, **kwargs):
        classify_calls.append(kwargs)
        return OACutConvexity(objective_is_convex=True, constraint_mask=[True, False])

    monkeypatch.setattr(
        convexity_mod,
        "classify_oa_cut_convexity",
        fake_classify,
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=0.0,
                x=np.array([1.0, 1.0], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([1.0, 1.0], dtype=np.float64), 2.0),
    )

    def fake_generate_report(*args, **kwargs):
        recorded_masks.append(list(kwargs["convex_mask"]))
        recorded_reasons.append(list(kwargs["skip_reasons"]))
        return cutting_planes.OACutGenerationReport(cuts=[], skipped=[])

    monkeypatch.setattr(
        cutting_planes,
        "generate_oa_cuts_from_evaluator_report",
        fake_generate_report,
    )

    m = Model("amp_oa_mask")
    x = m.continuous("x", lb=0, ub=2, shape=(2,))
    m.subject_to(x[0] + x[1] >= 1.0)
    m.subject_to(x[0] ** 2 - x[1] >= 0.0)
    m.minimize(x[0] + x[1])

    result = m.solve(
        solver="amp",
        apply_partitioning=False,
        skip_convex_check=True,
        presolve_bt=False,
        max_iter=1,
        time_limit=5,
    )

    assert result.status in ("optimal", "feasible")
    assert classify_calls == [{"use_certificate": True}]
    assert recorded_masks == [[True, False]]
    assert recorded_reasons == [[None, "opposite_curvature_for_direct_oa"]]


def test_amp_oa_classification_uses_tightened_bounds_for_reciprocal_rows(monkeypatch):
    """Root bound tightening should feed OA convexity classification."""
    import discopt._jax.cutting_planes as cutting_planes
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    recorded_masks = []

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=0.0,
                x=np.array([1.0, 1.0], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([1.0, 1.0], dtype=np.float64), 2.0),
    )

    def fake_generate_report(*args, **kwargs):
        recorded_masks.append(list(kwargs["convex_mask"]))
        return cutting_planes.OACutGenerationReport(cuts=[], skipped=[])

    monkeypatch.setattr(
        cutting_planes,
        "generate_oa_cuts_from_evaluator_report",
        fake_generate_report,
    )

    m = Model("amp_reciprocal_oa_bounds")
    x = m.continuous("x", lb=0.0)
    y = m.continuous("y", lb=0.0)
    m.minimize(x + y)
    m.subject_to(y >= 1 / (x + 0.1) - 0.5)
    m.subject_to(x >= y ** (-2) - 0.5)
    m.subject_to(4 / (x + y + 0.1) >= 1)

    result = m.solve(
        solver="amp",
        apply_partitioning=False,
        skip_convex_check=True,
        presolve_bt=False,
        max_iter=1,
        time_limit=5,
    )

    assert result.status in ("optimal", "feasible")
    assert recorded_masks == [[True, True, False]]


def _unwrap_minlptests_case(case):
    return case.values[0] if hasattr(case, "values") else case


def test_former_known_failure_nlp_mi_005_runs_in_pr_fast_suite():
    """Representative removed MINLPTests xfail should produce a PR-time AMP signal."""
    from test_minlptests import NLP_MI_INSTANCES

    instances = {
        instance.problem_id: instance
        for instance in (_unwrap_minlptests_case(case) for case in NLP_MI_INSTANCES)
    }
    instance = instances["nlp_mi_005_010"]
    m = instance.build_fn()

    result = m.solve(
        solver="amp",
        nlp_solver="ipm",
        time_limit=10.0,
        gap_tolerance=1e-3,
        apply_partitioning=False,
        max_iter=1,
    )

    assert result.status in ("optimal", "feasible")
    assert result.objective is not None
    assert result.objective == pytest.approx(instance.expected_obj, abs=1e-4)


def _issue91_minlptests_model(group: str, problem_id: str) -> Model:
    from test_minlptests import MINLPTESTS_CVX_BY_ID, NLP_INSTANCES, NLP_MI_INSTANCES

    if group == "cvx":
        return MINLPTESTS_CVX_BY_ID[problem_id].build_fn()
    instances = NLP_MI_INSTANCES if group == "mi" else NLP_INSTANCES
    by_id = {
        instance.problem_id: instance
        for instance in (_unwrap_minlptests_case(case) for case in instances)
    }
    return by_id[problem_id].build_fn()


def _issue91_oa_mask_and_skip_reasons(model: Model) -> tuple[list[bool], list[str | None]]:
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds
    from discopt.modeling.core import VarType
    from discopt.solvers import amp as amp_mod
    from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

    flat_lb, flat_ub = flat_variable_bounds(model)
    int_offsets = []
    int_sizes = []
    offset = 0
    for variable in model._variables:
        if variable.var_type in (VarType.BINARY, VarType.INTEGER):
            int_offsets.append(offset)
            int_sizes.append(variable.size)
        offset += variable.size

    flat_lb, flat_ub, root_infeasible, _root_changed = tighten_root_bounds_with_fbbt(
        model,
        flat_lb,
        flat_ub,
        int_offsets,
        int_sizes,
    )
    assert not root_infeasible

    flat_lb, flat_ub, nonlinear_bt_stats = tighten_nonlinear_bounds(model, flat_lb, flat_ub)
    assert not nonlinear_bt_stats.infeasible
    amp_mod._apply_flat_bounds_to_model(model, flat_lb, flat_ub)

    oa_convexity = classify_oa_cut_convexity(model, use_certificate=True)
    reasons = amp_mod._direct_oa_skip_reasons(
        model,
        oa_convexity.constraint_mask,
        flat_lb,
        flat_ub,
    )
    return oa_convexity.constraint_mask, reasons


@pytest.mark.parametrize(
    ("group", "problem_id", "expected_mask", "expected_reasons"),
    [
        pytest.param(
            "cvx",
            "nlp_cvx_106_010",
            [False, False],
            ["trigonometric_not_certified_convex", "trigonometric_not_certified_convex"],
            id="nlp_cvx_106_010",
        ),
        pytest.param(
            "cvx",
            "nlp_cvx_106_011",
            [False, False],
            ["trigonometric_not_certified_convex", "trigonometric_not_certified_convex"],
            id="nlp_cvx_106_011",
        ),
        pytest.param(
            "nlp",
            "nlp_002_010",
            [False, False],
            ["nonaffine_equality", "nonaffine_equality"],
            id="nlp_002_010",
        ),
        *[
            pytest.param(
                "nlp",
                f"nlp_003_0{suffix}",
                [True, False],
                [None, "trigonometric_not_certified_convex"],
                id=f"nlp_003_0{suffix}",
            )
            for suffix in range(10, 17)
        ],
        pytest.param(
            "nlp",
            "nlp_005_010",
            [True, True, False],
            [None, None, "opposite_curvature_for_direct_oa"],
            id="nlp_005_010",
        ),
        pytest.param(
            "nlp",
            "nlp_008_010",
            [True, False, True],
            [None, "nonconvex_quadratic_alpha_bb_candidate", None],
            id="nlp_008_010",
        ),
        pytest.param(
            "nlp",
            "nlp_008_011",
            [True, False, True],
            [None, "nonconvex_quadratic_alpha_bb_candidate", None],
            id="nlp_008_011",
        ),
        pytest.param(
            "mi",
            "nlp_mi_002_010",
            [True, False],
            [None, "fixed_nonlinear_row"],
            id="nlp_mi_002_010",
        ),
        *[
            pytest.param(
                "mi",
                f"nlp_mi_003_0{suffix}",
                [True, False],
                [None, "trigonometric_not_certified_convex"],
                id=f"nlp_mi_003_0{suffix}",
            )
            for suffix in range(10, 17)
        ],
        pytest.param(
            "mi",
            "nlp_mi_005_010",
            [True, True, False],
            [None, None, "opposite_curvature_for_direct_oa"],
            id="nlp_mi_005_010",
        ),
    ],
)
def test_issue91_minlptests_oa_rows_are_cut_or_explained(
    group,
    problem_id,
    expected_mask,
    expected_reasons,
):
    """Every issue 91 motivating row should be cut or have a stable direct-OA skip reason."""
    model = _issue91_minlptests_model(group, problem_id)

    mask, reasons = _issue91_oa_mask_and_skip_reasons(model)

    assert mask == expected_mask
    assert reasons == expected_reasons


def test_alphabb_quadratic_oa_cut_covers_issue_63_row():
    """The indefinite quadratic row should get a relaxed OA cut, not direct OA."""
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.cutting_planes import generate_alphabb_quadratic_oa_cuts_from_evaluator
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.nlp_evaluator import NLPEvaluator

    m = Model("issue_63_quadratic_cut")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize(x + y**2 + z**3)
    m.subject_to(y >= dm.exp(-x - 2) + dm.exp(-z - 2) - 2)
    m.subject_to(x**2 <= y**2 + z**2)
    m.subject_to(y >= x / 2 + z)

    oa_convexity = classify_oa_cut_convexity(m, use_certificate=True)
    assert oa_convexity.constraint_mask == [True, False, True]

    evaluator = NLPEvaluator(m)
    flat_lb, flat_ub = flat_variable_bounds(m)
    x_star = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    cuts = generate_alphabb_quadratic_oa_cuts_from_evaluator(
        evaluator,
        x_star,
        flat_lb,
        flat_ub,
        constraint_senses=[c.sense for c in m._constraints],
        convex_mask=oa_convexity.constraint_mask,
    )

    assert len(cuts) == 1
    cut = cuts[0]
    assert cut.sense == "<="
    assert np.linalg.norm(cut.coeffs) > 1e-12
    assert np.isfinite(cut.rhs)

    feasible_points = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0, 0.0], dtype=np.float64),
        np.array([-1.0, 1.0, 1.0], dtype=np.float64),
    ]
    for point in feasible_points:
        assert point[0] ** 2 <= point[1] ** 2 + point[2] ** 2 + 1e-12
        assert float(np.dot(cut.coeffs, point)) <= cut.rhs + 1e-8


def test_alphabb_quadratic_oa_skips_nonquadratic_row():
    """Nonquadratic rows must not be accepted by local Hessian coincidence."""
    from discopt._jax.cutting_planes import generate_alphabb_quadratic_oa_cuts_from_evaluator
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.nlp_evaluator import NLPEvaluator

    m = Model("nonquadratic_alphabb_skip")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.subject_to(-(x**4) <= -1.0)
    m.minimize(x)

    evaluator = NLPEvaluator(m)
    flat_lb, flat_ub = flat_variable_bounds(m)
    cuts = generate_alphabb_quadratic_oa_cuts_from_evaluator(
        evaluator,
        np.array([-0.173], dtype=np.float64),
        flat_lb,
        flat_ub,
        constraint_senses=[c.sense for c in m._constraints],
        convex_mask=[False],
    )

    assert cuts == []


def test_alphabb_quadratic_oa_uses_row_col_hessian_support(monkeypatch):
    """Column-only Hessian scans miss nonsymmetric fragments; row/col union must cut."""
    import discopt._jax.cutting_planes as cp

    class FakeEvaluator:
        n_constraints = 1

        def evaluate_constraints(self, x):
            del x
            return np.array([0.0], dtype=np.float64)

        def evaluate_jacobian(self, x):
            return np.zeros((1, len(x)), dtype=np.float64)

    def fake_hessian(evaluator, row_idx, n_vars):
        del evaluator, row_idx, n_vars
        return np.array([[0.0, 0.0], [-2.0, 0.0]], dtype=np.float64)

    monkeypatch.setattr(cp, "_constraint_row_quadratic_hessian", fake_hessian)

    cuts = cp.generate_alphabb_quadratic_oa_cuts_from_evaluator(
        FakeEvaluator(),
        np.array([0.25, -0.25], dtype=np.float64),
        np.array([-1.0, -1.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        constraint_senses=["<="],
        convex_mask=[False],
    )

    assert len(cuts) == 1
    assert np.linalg.norm(cuts[0].coeffs) > 1e-12


def test_objective_cutoff_odd_power_tightening_uses_signed_monotonic_root():
    """Odd monomials are monotone, not symmetric level sets."""
    from discopt.solvers import amp as amp_mod

    assert amp_mod._tighten_simple_power_group({3: 1.0}, 8.0, -10.0, 10.0) == pytest.approx(
        (-10.0, 2.0)
    )
    assert amp_mod._tighten_simple_power_group({3: 1.0}, 8.0, -10.0, -1.0) == pytest.approx(
        (-10.0, -1.0)
    )
    assert amp_mod._tighten_simple_power_group({3: 1.0}, -8.0, -10.0, 10.0) == pytest.approx(
        (-10.0, -2.0)
    )
    assert amp_mod._tighten_simple_power_group({3: -1.0}, 8.0, -10.0, 10.0) == pytest.approx(
        (-2.0, 10.0)
    )


def test_objective_cutoff_negative_odd_power_interval_preserves_feasible_points():
    """A positive cutoff on x**3 over a negative interval must not trim the left side."""
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt.solvers import amp as amp_mod

    m = Model("odd_power_cutoff_counterexample")
    x = m.continuous("x", lb=-10.0, ub=-1.0)
    y = m.continuous("y", lb=100.0, ub=100.0)
    m.minimize(x**3 + y**2)

    flat_lb, flat_ub = flat_variable_bounds(m)
    cutoff = 10001.0
    cutoff_lb, cutoff_ub = amp_mod._tighten_bounds_with_objective_cutoff(
        m,
        flat_lb,
        flat_ub,
        cutoff=cutoff,
    )

    assert (-10.0) ** 3 + 100.0**2 <= cutoff
    np.testing.assert_allclose(cutoff_lb, flat_lb)
    np.testing.assert_allclose(cutoff_ub, flat_ub)


def test_refresh_partitions_for_bounds_prunes_stale_partition_entries():
    """Dropped partition vars must not leave stale active entries in disc_state."""
    from discopt.solvers import amp as amp_mod

    m = Model("partition_refresh_prunes_stale")
    x = m.continuous("x", shape=(3,))
    disc_state = SimpleNamespace(
        partitions={
            0: np.array([-5.0, 0.0, 5.0], dtype=np.float64),
            1: np.array([1.0, 2.0, 3.0], dtype=np.float64),
            2: np.array([-3.0, 0.0, 3.0], dtype=np.float64),
        }
    )
    flat_lb = np.array([-1.0, 2.0, -3.0], dtype=np.float64)
    flat_ub = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    part_vars, part_lbs, part_ubs = amp_mod._refresh_partitions_for_bounds(
        m,
        disc_state,
        flat_lb,
        flat_ub,
        part_vars=[0, 1],
        disc_abs_width_tol=1e-9,
        n_init_partitions=2,
    )

    assert part_vars == [0]
    assert part_lbs == [-1.0]
    assert part_ubs == [1.0]
    assert set(disc_state.partitions) == {0}
    np.testing.assert_allclose(disc_state.partitions[0], np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_allclose(x.lb, flat_lb)
    np.testing.assert_allclose(x.ub, flat_ub)


@pytest.mark.parametrize(
    ("solver_kwargs", "expected_obbt_calls"),
    [
        ({}, 1),
        ({"alphabb_cutoff_obbt": False}, 0),
        ({"alphabb_cutoff_obbt": False, "obbt_with_cutoff": True}, 1),
    ],
)
@pytest.mark.memory_heavy
def test_alphabb_cutoff_obbt_option_controls_prerequisite_pass(
    monkeypatch,
    solver_kwargs,
    expected_obbt_calls,
):
    """The alpha-BB cutoff OBBT pass has its own switch."""
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    calls = []

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=-1.0,
                x=np.array([0.0, 0.0], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([0.0, 0.0], dtype=np.float64), 0.0),
    )

    def fake_cutoff_obbt(**kwargs):
        calls.append(kwargs["iteration"])
        return (
            kwargs["flat_lb"],
            kwargs["flat_ub"],
            kwargs["part_vars"],
            kwargs["part_lbs"],
            kwargs["part_ubs"],
        )

    monkeypatch.setattr(amp_mod, "_run_cutoff_obbt", fake_cutoff_obbt)

    m = Model("alphabb_cutoff_obbt_option")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x + y**2)
    m.subject_to(x**2 <= y**2 + 1.0)

    result = m.solve(
        solver="amp",
        apply_partitioning=False,
        skip_convex_check=True,
        presolve_bt=False,
        max_iter=1,
        time_limit=5,
        **solver_kwargs,
    )

    assert result.status in ("optimal", "feasible")
    assert len(calls) == expected_obbt_calls


def test_run_cutoff_obbt_returns_without_time_after_deadline(monkeypatch):
    """Expired AMP deadlines must not be converted into a fresh OBBT budget."""
    from discopt._jax import milp_relaxation as milp_relaxation_mod
    from discopt.solvers import amp as amp_mod

    build_calls = []

    def fail_build_relaxation(*args, **kwargs):
        build_calls.append((args, kwargs))
        raise AssertionError("cutoff OBBT should not build a relaxation after the deadline")

    monkeypatch.setattr(
        milp_relaxation_mod,
        "build_milp_relaxation",
        fail_build_relaxation,
    )

    m = Model("expired_cutoff_obbt")
    x = m.continuous("x")
    m.minimize(x)
    flat_lb = np.array([-np.inf], dtype=np.float64)
    flat_ub = np.array([np.inf], dtype=np.float64)
    part_vars = [0]
    part_lbs = [-1.0]
    part_ubs = [1.0]

    result = amp_mod._run_cutoff_obbt(
        model=m,
        terms=SimpleNamespace(partition_candidates=[0]),
        disc_state=SimpleNamespace(partitions={}),
        oa_cuts=[],
        convhull_mode="sos2",
        UB=0.0,
        flat_lb=flat_lb,
        flat_ub=flat_ub,
        part_vars=part_vars,
        part_lbs=part_lbs,
        part_ubs=part_ubs,
        n_orig=1,
        obbt_time_limit=30.0,
        partition_scaling_factor=0.1,
        disc_abs_width_tol=1e-9,
        n_init_partitions=2,
        deadline=amp_mod.time.perf_counter() - 1.0,
        iteration=1,
        from_min_space=float,
    )

    result_lb, result_ub, result_part_vars, result_part_lbs, result_part_ubs = result
    assert result_lb is flat_lb
    assert result_ub is flat_ub
    assert result_part_vars is part_vars
    assert result_part_lbs is part_lbs
    assert result_part_ubs is part_ubs
    assert build_calls == []


@pytest.mark.memory_heavy
def test_alphabb_cutoff_obbt_prerequisite_respects_expired_deadline(monkeypatch):
    """The default alpha-BB prerequisite pass must honor the global AMP deadline."""
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    cutoff_obbt_calls = []

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=-1.0,
                x=np.array([0.0, 0.0], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([0.0, 0.0], dtype=np.float64), 0.0),
    )

    def fake_cutoff_obbt(**kwargs):
        cutoff_obbt_calls.append(kwargs["iteration"])
        return (
            kwargs["flat_lb"],
            kwargs["flat_ub"],
            kwargs["part_vars"],
            kwargs["part_lbs"],
            kwargs["part_ubs"],
        )

    monkeypatch.setattr(amp_mod, "_run_cutoff_obbt", fake_cutoff_obbt)

    clock_values = iter([0.0, 0.1, 2.0])

    def fake_perf_counter():
        return next(clock_values, 2.0)

    monkeypatch.setattr(amp_mod.time, "perf_counter", fake_perf_counter)

    m = Model("alphabb_cutoff_obbt_expired_deadline")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x + y**2)
    m.subject_to(x**2 <= y**2 + 1.0)

    result = m.solve(
        solver="amp",
        apply_partitioning=False,
        skip_convex_check=True,
        presolve_bt=False,
        max_iter=1,
        time_limit=1.0,
    )

    assert result.status in ("optimal", "feasible")
    assert cutoff_obbt_calls == []


@pytest.mark.memory_heavy
def test_objective_cutoff_bounds_enable_alphabb_for_issue_63_instances():
    """The exact nlp_008 objective cutoff should create finite alpha-BB bounds."""
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.cutting_planes import generate_alphabb_quadratic_oa_cuts_from_evaluator
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.solvers import amp as amp_mod
    from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

    m = Model("issue_63_exact_cutoff_bounds")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize(x + y**2 + z**3)
    m.subject_to(y >= dm.exp(-x - 2) + dm.exp(-z - 2) - 2)
    m.subject_to(x**2 <= y**2 + z**2)
    m.subject_to(y >= x / 2 + z)

    flat_lb, flat_ub = flat_variable_bounds(m)
    flat_lb, flat_ub, infeasible, _changed = tighten_root_bounds_with_fbbt(
        m,
        flat_lb,
        flat_ub,
        [],
        [],
    )
    assert infeasible is False
    assert not amp_mod.is_effectively_finite(float(flat_ub[0]))
    assert not amp_mod.is_effectively_finite(float(flat_ub[1]))

    cutoff_lb, cutoff_ub = amp_mod._tighten_bounds_with_objective_cutoff(
        m,
        flat_lb,
        flat_ub,
        cutoff=-0.3285279886580375,
    )

    assert amp_mod.is_effectively_finite(float(cutoff_ub[0]))
    assert amp_mod.is_effectively_finite(float(cutoff_ub[1]))

    amp_mod._apply_flat_bounds_to_model(m, cutoff_lb, cutoff_ub)
    cutoff_lb, cutoff_ub, infeasible, _changed = tighten_root_bounds_with_fbbt(
        m,
        cutoff_lb,
        cutoff_ub,
        [],
        [],
    )
    assert infeasible is False
    assert cutoff_lb[0] > -10.0

    oa_convexity = classify_oa_cut_convexity(m, use_certificate=True)
    evaluator = NLPEvaluator(m)
    cuts = generate_alphabb_quadratic_oa_cuts_from_evaluator(
        evaluator,
        np.array([-0.57718987, 0.27099034, 0.55958528], dtype=np.float64),
        cutoff_lb,
        cutoff_ub,
        constraint_senses=[c.sense for c in m._constraints],
        convex_mask=oa_convexity.constraint_mask,
    )

    assert len(cuts) == 1


@pytest.mark.memory_heavy
def test_amp_restores_model_bounds_after_objective_cutoff_tightening(monkeypatch):
    """Internal cutoff/FBBT bounds must not leak back to the caller's model."""
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt.solvers import amp as amp_mod

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=-1.0,
                x=np.array([-0.6, 0.3, 0.5], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([-0.6, 0.3, 0.5], dtype=np.float64), -0.33),
    )

    m = Model("issue_63_bound_restore")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize(x + y**2 + z**3)
    m.subject_to(y >= dm.exp(-x - 2) + dm.exp(-z - 2) - 2)
    m.subject_to(x**2 <= y**2 + z**2)
    m.subject_to(y >= x / 2 + z)

    original_lb, original_ub = flat_variable_bounds(m)

    result = m.solve(
        solver="amp",
        apply_partitioning=False,
        skip_convex_check=True,
        presolve_bt=False,
        alphabb_cutoff_obbt=False,
        max_iter=1,
        time_limit=5,
    )

    restored_lb, restored_ub = flat_variable_bounds(m)
    assert result.status in ("optimal", "feasible")
    np.testing.assert_allclose(restored_lb, original_lb)
    np.testing.assert_allclose(restored_ub, original_ub)


@pytest.mark.memory_heavy
def test_amp_restores_model_bounds_when_callback_raises_after_cutoff(monkeypatch):
    """Propagated callback errors must still restore temporary AMP bounds."""
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt.solvers import amp as amp_mod

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=-1.0,
                x=np.array([-0.6, 0.3, 0.5], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([-0.6, 0.3, 0.5], dtype=np.float64), -0.33),
    )

    m = Model("issue_63_bound_restore_callback_error")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize(x + y**2 + z**3)
    m.subject_to(y >= dm.exp(-x - 2) + dm.exp(-z - 2) - 2)
    m.subject_to(x**2 <= y**2 + z**2)
    m.subject_to(y >= x / 2 + z)

    original_lb, original_ub = flat_variable_bounds(m)

    def fail_callback(_info):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        m.solve(
            solver="amp",
            apply_partitioning=False,
            skip_convex_check=True,
            presolve_bt=False,
            alphabb_cutoff_obbt=False,
            max_iter=1,
            time_limit=5,
            iteration_callback=fail_callback,
        )

    restored_lb, restored_ub = flat_variable_bounds(m)
    np.testing.assert_allclose(restored_lb, original_lb)
    np.testing.assert_allclose(restored_ub, original_ub)


@pytest.mark.memory_heavy
def test_amp_appends_alphabb_cut_for_issue_63_quadratic(monkeypatch):
    """AMP should append an alpha-BB cut for the nonconvex quadratic row."""
    import discopt._jax.cutting_planes as cutting_planes
    from discopt._jax.cutting_planes import OACutGenerationReport
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    captured_cuts = []

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=0.0,
                x=np.array([0.25, 0.5, 0.25], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([0.25, 0.5, 0.25], dtype=np.float64), 1.0),
    )
    monkeypatch.setattr(
        cutting_planes,
        "generate_oa_cuts_from_evaluator_report",
        lambda *a, **k: OACutGenerationReport(cuts=[], skipped=[]),
    )

    def spy_prune(oa_cuts, max_cuts=128):
        captured_cuts[:] = list(oa_cuts)
        return None

    monkeypatch.setattr(amp_mod, "_prune_oa_cuts", spy_prune)

    m = Model("issue_63_amp_cut")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize(x + y**2 + z**3)
    m.subject_to(y >= dm.exp(-x - 2) + dm.exp(-z - 2) - 2)
    m.subject_to(x**2 <= y**2 + z**2)
    m.subject_to(y >= x / 2 + z)

    result = m.solve(
        solver="amp",
        apply_partitioning=False,
        skip_convex_check=True,
        presolve_bt=False,
        max_iter=1,
        time_limit=5,
    )

    assert result.status in ("optimal", "feasible")
    assert len(captured_cuts) == 1
    coeffs, rhs = captured_cuts[0]
    assert np.linalg.norm(coeffs) > 1e-12
    assert np.isfinite(rhs)


@pytest.mark.memory_heavy
def test_amp_keeps_direct_oa_when_alphabb_generation_fails(monkeypatch):
    """Alpha-BB failures should not suppress already generated convex OA cuts."""
    import discopt._jax.cutting_planes as cutting_planes
    from discopt._jax.cutting_planes import LinearCut
    from discopt._jax.milp_relaxation import MilpRelaxationResult
    from discopt.solvers import amp as amp_mod

    captured_cuts = []

    monkeypatch.setattr(
        amp_mod,
        "_solve_milp_with_oa_recovery",
        lambda **kwargs: (
            MilpRelaxationResult(
                status="optimal",
                objective=0.0,
                x=np.array([1.0], dtype=np.float64),
            ),
            {},
            [],
            1,
        ),
    )
    monkeypatch.setattr(
        amp_mod,
        "_solve_best_nlp_candidate",
        lambda *args, **kwargs: (np.array([1.0], dtype=np.float64), 1.0),
    )
    monkeypatch.setattr(
        cutting_planes,
        "generate_oa_cuts_from_evaluator",
        lambda *args, **kwargs: [
            LinearCut(coeffs=np.array([1.0], dtype=np.float64), rhs=2.0, sense="<=")
        ],
    )

    def fail_alphabb(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("alpha-BB failure")

    def spy_prune(oa_cuts, max_cuts=128):
        del max_cuts
        captured_cuts[:] = list(oa_cuts)
        return None

    monkeypatch.setattr(
        cutting_planes,
        "generate_alphabb_quadratic_oa_cuts_from_evaluator",
        fail_alphabb,
    )
    monkeypatch.setattr(amp_mod, "_prune_oa_cuts", spy_prune)

    m = Model("direct_oa_survives_alphabb_failure")
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.subject_to(x <= 2.0)
    m.minimize(x)

    result = m.solve(
        solver="amp",
        apply_partitioning=False,
        skip_convex_check=True,
        presolve_bt=False,
        max_iter=1,
        time_limit=5,
    )

    assert result.status in ("optimal", "feasible")
    assert len(captured_cuts) == 1
    coeffs, rhs = captured_cuts[0]
    np.testing.assert_allclose(coeffs, np.array([1.0], dtype=np.float64))
    assert rhs == pytest.approx(2.0)


@pytest.mark.memory_heavy
def test_amp_root_presolve_preserves_heterogeneous_array_bounds():
    """Root FBBT must not narrow every array element to the first element's bound."""
    m = Model("amp_heterogeneous_array_bounds")
    x = m.continuous(
        "x",
        shape=(2,),
        lb=np.array([0.0, 0.0]),
        ub=np.array([1.0, 10.0]),
    )
    m.subject_to(x[0] ** 2 >= 0.0)
    m.minimize(-x[1])

    result = m.solve(
        solver="amp",
        apply_partitioning=False,
        skip_convex_check=True,
        presolve_bt=False,
        max_iter=2,
        time_limit=10,
    )

    assert result.status in ("optimal", "feasible")
    assert result.x is not None
    assert result.objective is not None
    assert result.objective <= -9.0
    assert result.x["x"][1] >= 9.0


def test_small_integer_domain_fallback_enumerates_complete_domain(monkeypatch):
    """The small-domain fallback should enumerate bounded integer domains directly."""
    from discopt.solvers import amp as amp_mod

    selected_candidates = []

    def fake_select(candidates, *args, **kwargs):
        del args, kwargs
        selected_candidates.extend(candidates)
        return np.asarray(candidates[-1], dtype=np.float64), 0.0

    monkeypatch.setattr(amp_mod, "_select_best_nlp_candidate", fake_select)

    m = Model("small_integer_fallback")
    m.integer("y", lb=0, ub=2)

    x_best, obj_best = amp_mod._solve_small_integer_domain_fallback(
        m,
        evaluator=object(),
        flat_lb=np.array([0.0], dtype=np.float64),
        flat_ub=np.array([2.0], dtype=np.float64),
        constraint_lb=np.array([], dtype=np.float64),
        constraint_ub=np.array([], dtype=np.float64),
        nlp_solver="ipm",
        max_assignments=4,
    )

    assert x_best is not None
    assert obj_best == pytest.approx(0.0)
    assert sorted(float(candidate[0]) for candidate in selected_candidates) == [0.0, 1.0, 2.0]


@pytest.mark.memory_heavy
def test_obbt_presolve_tightens_bilinear_demo_bounds():
    """OBBT should shrink the initial [0, 10]^2 box to the linear hull x + y = 1."""
    from discopt._jax.obbt import run_obbt

    result = run_obbt(_make_obbt_demo())

    np.testing.assert_allclose(result.tightened_lb, np.array([0.0, 0.0]))
    np.testing.assert_allclose(result.tightened_ub, np.array([1.0, 1.0]))
    assert result.n_tightened >= 2


def test_amp_returns_infeasible_for_nonlinear_tightening_contradiction():
    """A tiny end-to-end AMP solve should stop when tightening proves infeasibility."""
    m = Model("amp_contradiction")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(dm.sqrt(x) <= -1.0)
    m.minimize(x)

    result = m.solve(solver="amp", skip_convex_check=True, max_iter=1, time_limit=5)

    assert result.status == "infeasible"
    assert result.x is None


@pytest.mark.memory_heavy
def test_amp_max_iter_without_gap_certificate_returns_feasible():
    """An incumbent without a certified gap should not be labeled optimal."""
    m = _make_nlp1()

    result = m.solve(solver="amp", max_iter=1, time_limit=30)

    assert result.status == "feasible"
    assert result.gap_certified is False
    assert result.objective is not None
    assert result.gap is not None
    assert result.gap > 1e-3
