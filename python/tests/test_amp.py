"""Fast AMP regressions for the default PR test battery."""

from __future__ import annotations

import os
from pathlib import Path

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


def _make_obbt_demo() -> Model:
    m = Model("obbt_demo")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y == 1)
    m.maximize(x * y)
    return m


def test_amp_integration_suite_is_opt_in():
    """The Alpine/MINLPTests suite must stay out of the default marker selection."""
    text = Path(__file__).with_name("test_amp_integration.py").read_text(encoding="utf-8")

    assert "pytest.mark.slow" in text
    assert "pytest.mark.integration" in text
    assert "pytest.mark.amp_benchmark" in text


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

    solve_model(
        m,
        solver="amp",
        gap_tolerance=1e-3,
        apply_partitioning=False,
        disc_var_pick=1,
        partition_scaling_factor=7.0,
        disc_add_partition_method="uniform",
        disc_abs_width_tol=1e-2,
        convhull_formulation="sos2",
    )

    assert captured["rel_gap"] == pytest.approx(1e-3)
    assert captured["apply_partitioning"] is False
    assert captured["disc_var_pick"] == 1
    assert captured["partition_scaling_factor"] == pytest.approx(7.0)
    assert captured["disc_add_partition_method"] == "uniform"
    assert captured["disc_abs_width_tol"] == pytest.approx(1e-2)
    assert captured["convhull_formulation"] == "sos2"


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
