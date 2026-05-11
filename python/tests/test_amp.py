"""Fast AMP regressions for the default PR test battery."""

from __future__ import annotations

import os
import subprocess
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


def _make_dry_run(target: str) -> str:
    repo = Path(__file__).resolve().parents[2]
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
    )
    return result.stdout


def test_quick_test_tier_excludes_amp_integration_markers():
    """The quick tier must not select opt-in AMP smoke tests."""
    output = _make_dry_run("test-quick")

    assert (
        '-m "(unit or smoke) and not slow and not integration '
        'and not amp_benchmark and not requires_cyipopt"'
    ) in output


def test_pr_fast_tier_excludes_heavy_manual_markers():
    """The PR-fast tier should keep nightly/manual markers out of make test."""
    output = _make_dry_run("test")

    assert (
        '-m "not slow and not correctness and not integration '
        'and not amp_benchmark and not requires_cyipopt"'
    ) in output
    assert "--ignore=python/tests/test_correctness.py" in output


def test_amp_fast_tier_excludes_optional_solver_markers():
    """The local AMP fast target should not require optional NLP solver stacks."""
    output = _make_dry_run("test-amp-fast")

    assert (
        '-m "not slow and not integration and not amp_benchmark and not requires_cyipopt"'
    ) in output


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
    """Square-balance equalities should add monomial vars without changing classifier output."""
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


def test_solve_nlp_subproblem_retries_ipopt_and_restores_bounds(monkeypatch):
    """AMP local NLP recovery should retry Ipopt and restore temporary fixed bounds."""
    import discopt._jax.ipm as ipm_mod
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

    def fake_ipm(evaluator, x0, options):
        del evaluator, x0, options
        calls.append("ipm")
        return NLPResult(status=SolveStatus.ERROR)

    def fake_ipopt(evaluator, x0, options):
        del evaluator, options
        calls.append("ipopt")
        return NLPResult(status=SolveStatus.OPTIMAL, x=np.array([3.0]), objective=0.0)

    monkeypatch.setattr(amp_mod, "_has_cyipopt", lambda: True)
    monkeypatch.setattr(ipm_mod, "solve_nlp_ipm", fake_ipm)
    monkeypatch.setattr(ipopt_mod, "solve_nlp", fake_ipopt)

    x_opt, obj = amp_mod._solve_nlp_subproblem(
        FakeEvaluator(),
        x0=np.array([99.0], dtype=np.float64),
        lb=np.array([1.0], dtype=np.float64),
        ub=np.array([5.0], dtype=np.float64),
        nlp_solver="ipm",
        time_limit=10.0,
    )

    assert calls == ["ipm", "ipopt"]
    np.testing.assert_allclose(x_opt, np.array([3.0]))
    assert obj == pytest.approx(0.0)
    np.testing.assert_allclose(x.lb, original_lb)
    np.testing.assert_allclose(x.ub, original_ub)


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

    monkeypatch.setattr(
        convexity_mod,
        "classify_oa_cut_convexity",
        lambda model: OACutConvexity(objective_is_convex=True, constraint_mask=[True, False]),
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

    def fake_generate(*args, **kwargs):
        recorded_masks.append(list(kwargs["convex_mask"]))
        return []

    monkeypatch.setattr(cutting_planes, "generate_oa_cuts_from_evaluator", fake_generate)

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
    assert recorded_masks == [[True, False]]


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


def test_amp_max_iter_without_gap_certificate_returns_feasible():
    """An incumbent without a certified gap should not be labeled optimal."""
    m = _make_nlp1()

    result = m.solve(solver="amp", max_iter=1, time_limit=30)

    assert result.status == "feasible"
    assert result.gap_certified is False
    assert result.objective is not None
    assert result.gap is not None
    assert result.gap > 1e-3
