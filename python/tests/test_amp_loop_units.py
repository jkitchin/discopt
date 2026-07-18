"""Unit tests for AMP loop-support helpers and hook plumbing (#87).

Covers gap arithmetic, OA-cut pruning, MILP/OBBT budget allocation, the
small-integer-domain fallback, user hooks (partition selection / scaling /
refinement), cutoff OBBT, and the maximize-sense result mapping — each
against its documented contract. Every model here is 1-3 variables so the
whole module runs in a few seconds.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._jax.discretization import DiscretizationState
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import Model
from discopt.solvers.amp import (
    _amp_abs_gap_with_bound_tolerance,
    _append_upper_bound_constraint,
    _apply_partition_refinement_hook,
    _apply_partition_scaling_update,
    _build_fixed_integer_bounds,
    _check_constraints,
    _compute_relative_gap,
    _default_milp_time_limit,
    _equality_square_monomial_partition_candidates,
    _presolve_incumbent_from_initial_point,
    _prune_oa_cuts,
    _round_integers,
    _run_cutoff_obbt,
    _select_partition_vars_with_hook,
    _small_integer_domain_size,
    _solve_nlp_subproblem,
)
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Gap arithmetic and cut pruning
# ---------------------------------------------------------------------------


def test_compute_relative_gap_contract():
    assert _compute_relative_gap(0.5, 10.0) == pytest.approx(0.05)
    assert _compute_relative_gap(None, 10.0) is None
    assert _compute_relative_gap(-0.1, 10.0) is None  # negative gap is invalid
    assert _compute_relative_gap(0.5, 0.0) is None  # zero UB -> undefined
    assert _compute_relative_gap(0.5, np.inf) is None


def test_amp_abs_gap_bound_tolerance():
    # Proper ordering passes through.
    gap, ok = _amp_abs_gap_with_bound_tolerance(1.0, 3.0, abs_tol=1e-6)
    assert gap == pytest.approx(2.0) and ok
    # A hair of inversion (numerical noise) clamps to zero, still trustworthy.
    gap, ok = _amp_abs_gap_with_bound_tolerance(1.0 + 1e-9, 1.0, abs_tol=1e-6)
    assert gap == 0.0 and ok
    # A material inversion invalidates the bound.
    gap, ok = _amp_abs_gap_with_bound_tolerance(2.0, 1.0, abs_tol=1e-6)
    assert gap is None and not ok


def test_prune_oa_cuts_keeps_most_recent():
    cuts = list(range(10))
    _prune_oa_cuts(cuts, max_cuts=4)
    assert cuts == [6, 7, 8, 9]
    same = list(range(3))
    _prune_oa_cuts(same, max_cuts=4)
    assert same == [0, 1, 2]


def test_default_milp_time_limit_budgets():
    # Early iteration of a long run: horizon-capped even split, tripled,
    # bounded by 80% of remaining and 60s.
    budget = _default_milp_time_limit(320.0, iteration=1, max_iter=1000)
    assert budget == pytest.approx(min(3 * 320.0 / 32, 0.8 * 320.0, 60.0))
    # Final iteration gets the whole remaining wall (capped).
    assert _default_milp_time_limit(10.0, iteration=50, max_iter=50) == pytest.approx(8.0)
    assert _default_milp_time_limit(1000.0, iteration=1, max_iter=8) == 60.0


# ---------------------------------------------------------------------------
# Small integer domain fallback support
# ---------------------------------------------------------------------------


def test_small_integer_domain_size():
    m = Model("dom")
    m.integer("i", lb=0, ub=3)
    m.binary("b")
    assert _small_integer_domain_size(m, max_assignments=128) == 8

    m2 = Model("dom2")
    m2.continuous("x", lb=0, ub=1)
    assert _small_integer_domain_size(m2, max_assignments=128) is None

    m3 = Model("dom3")
    m3.integer("i", lb=0, ub=63)
    m3.binary("b", shape=(2,))
    # 64 * 4 = 256 > 128 -> abstain
    assert _small_integer_domain_size(m3, max_assignments=128) is None

    m4 = Model("dom4")
    m4.integer("i", lb=2.7, ub=2.2)
    assert _small_integer_domain_size(m4, max_assignments=128) == 0


def test_round_integers_and_fixed_bounds():
    m = Model("rf")
    m.integer("i", lb=0, ub=5)
    m.continuous("x", lb=-1.0, ub=1.0)
    x = np.array([2.4, 0.3])
    rounded = _round_integers(x, m)
    assert rounded[0] == 2.0 and rounded[1] == pytest.approx(0.3)
    nlp_lb, nlp_ub = _build_fixed_integer_bounds(
        np.array([2.6, 0.3]), m, np.array([0.0, -1.0]), np.array([5.0, 1.0])
    )
    # Integer fixed to its rounded value; continuous bounds untouched.
    assert nlp_lb[0] == nlp_ub[0] == 3.0
    assert nlp_lb[1] == -1.0 and nlp_ub[1] == 1.0


# ---------------------------------------------------------------------------
# Hook plumbing
# ---------------------------------------------------------------------------


def _terms_for(model: Model):
    return classify_nonlinear_terms(model)


def _bilinear_model():
    m = Model("bil")
    x = m.continuous("x", lb=0.0, ub=2.0, shape=(2,))
    m.subject_to(x[0] * x[1] >= 1.0)
    m.minimize(x[0] + x[1])
    return m


def test_select_partition_vars_builtin_and_hook():
    m = _bilinear_model()
    terms = _terms_for(m)

    def builtin(t, method="auto", distance=None):
        return [0, 1]

    ctx = {"stage": "initial_selection", "distance": None}
    assert _select_partition_vars_with_hook(
        terms,
        method="auto",
        disc_var_pick_hook=None,
        pick_partition_vars=builtin,
        n_orig=2,
        context=ctx,
    ) == [0, 1]

    seen: dict = {}

    def hook(context):
        seen.update(context)
        # Delegate to the builtin then keep only the first index.
        picked = context["builtin_pick_partition_vars"]()
        return picked[:1]

    out = _select_partition_vars_with_hook(
        terms,
        method="auto",
        disc_var_pick_hook=hook,
        pick_partition_vars=builtin,
        n_orig=2,
        context=ctx,
    )
    assert out == [0]
    assert "partition_candidates" in seen

    def bad_hook(context):
        return [99]

    with pytest.raises(ValueError, match="outside valid range"):
        _select_partition_vars_with_hook(
            terms,
            method="auto",
            disc_var_pick_hook=bad_hook,
            pick_partition_vars=builtin,
            n_orig=2,
            context=ctx,
        )


def test_apply_partition_scaling_update_contract():
    assert _apply_partition_scaling_update(None, current_scaling_factor=10.0, context={}) == 10.0
    # Hook returning None keeps the current factor.
    assert (
        _apply_partition_scaling_update(lambda ctx: None, current_scaling_factor=10.0, context={})
        == 10.0
    )
    # Hook sees the current factor and can update it.
    assert (
        _apply_partition_scaling_update(
            lambda ctx: ctx["current_scaling_factor"] / 2.0,
            current_scaling_factor=10.0,
            context={},
        )
        == 5.0
    )
    with pytest.raises(ValueError, match="partition_scaling_factor_update"):
        _apply_partition_scaling_update(lambda ctx: 0.5, current_scaling_factor=10.0, context={})


def test_apply_partition_refinement_hook_contract():
    disc = DiscretizationState()
    ctx = {"disc_state": disc}
    # Returning a state passes validation.
    other = DiscretizationState(scaling_factor=4.0)
    assert _apply_partition_refinement_hook(lambda c: other, ctx) is other
    # Returning None falls back to context's disc_state.
    assert _apply_partition_refinement_hook(lambda c: None, ctx) is disc
    with pytest.raises(ValueError, match="DiscretizationState"):
        _apply_partition_refinement_hook(lambda c: 42, ctx)


def test_equality_square_monomial_partition_candidates_weymouth():
    m = Model("weymouth")
    f = m.continuous("f", lb=0.0, ub=2.0)
    p = m.continuous("p", lb=0.5, ub=2.0, shape=(2,))
    m.subject_to(f**2 == 0.7 * (p[0] ** 2 - p[1] ** 2))
    m.minimize(f)
    terms = _terms_for(m)
    cands = _equality_square_monomial_partition_candidates(m, terms)
    # All coupled squares partition together.
    assert cands == [0, 1, 2]

    m2 = _bilinear_model()  # bilinear only, no squares
    assert _equality_square_monomial_partition_candidates(m2, _terms_for(m2)) == []


# ---------------------------------------------------------------------------
# Presolve incumbent seeding and constraint checks
# ---------------------------------------------------------------------------


def _circle_model():
    m = Model("circle")
    x = m.continuous("x", lb=0.0, ub=2.0, shape=(2,))
    m.subject_to(x[0] ** 2 + x[1] ** 2 >= 2.0)
    m.minimize(x[0] + x[1])
    return m


def test_presolve_incumbent_from_initial_point_paths():
    m = _circle_model()
    ev = NLPEvaluator(m)
    lb_g, ub_g = _infer_constraint_bounds(m)
    lb_g = np.asarray(lb_g, dtype=np.float64)
    ub_g = np.asarray(ub_g, dtype=np.float64)

    assert _presolve_incumbent_from_initial_point(None, m, ev, lb_g, ub_g) == (None, None)

    feasible = np.array([1.5, 1.5])
    x0, obj = _presolve_incumbent_from_initial_point(feasible, m, ev, lb_g, ub_g)
    np.testing.assert_allclose(x0, feasible)
    assert obj == pytest.approx(3.0)

    infeasible = np.array([0.1, 0.1])  # violates x0^2 + x1^2 >= 2
    assert _presolve_incumbent_from_initial_point(infeasible, m, ev, lb_g, ub_g) == (None, None)


def test_presolve_incumbent_rejects_fractional_integers():
    m = Model("mi")
    m.integer("i", lb=0, ub=3)
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    ev = NLPEvaluator(m)
    point = np.array([1.4, 0.5])  # fractional integer -> rejected
    assert _presolve_incumbent_from_initial_point(point, m, ev, np.array([]), np.array([])) == (
        None,
        None,
    )


def test_check_constraints_accepts_and_rejects():
    m = _circle_model()
    assert _check_constraints(np.array([1.5, 1.5]), m)
    assert not _check_constraints(np.array([0.1, 0.1]), m)
    # Unconstrained model: trivially satisfied.
    m2 = Model("unc")
    x = m2.continuous("x", lb=0.0, ub=1.0)
    m2.minimize(x)
    assert _check_constraints(np.array([0.5]), m2)


# ---------------------------------------------------------------------------
# NLP subproblem success path and failure injection
# ---------------------------------------------------------------------------


def test_solve_nlp_subproblem_solves_tiny_convex_nlp():
    m = Model("nlp")
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.minimize((x - 1.0) ** 2)
    ev = NLPEvaluator(m)
    x_opt, obj = _solve_nlp_subproblem(
        ev, np.array([0.2]), np.array([0.0]), np.array([2.0]), nlp_solver="ipm"
    )
    assert x_opt is not None
    assert x_opt[0] == pytest.approx(1.0, abs=1e-4)
    assert obj == pytest.approx(0.0, abs=1e-6)


def test_solve_nlp_subproblem_rejects_nonfinite_solution(monkeypatch):
    from types import SimpleNamespace

    from discopt.solvers import SolveStatus

    m = Model("nlp2")
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.minimize((x - 1.0) ** 2)
    ev = NLPEvaluator(m)

    def fake_solve_nlp(evaluator, x0, options=None):
        return SimpleNamespace(status=SolveStatus.OPTIMAL, x=np.array([np.nan]))

    import discopt.solvers.nlp_pounce as nlp_pounce

    monkeypatch.setattr(nlp_pounce, "solve_nlp", fake_solve_nlp)
    assert _solve_nlp_subproblem(
        ev, np.array([0.2]), np.array([0.0]), np.array([2.0]), nlp_solver="ipm"
    ) == (None, None)


def test_solve_nlp_subproblem_rejects_nonfinite_objective(monkeypatch):
    from types import SimpleNamespace

    from discopt.solvers import SolveStatus

    m = Model("nlp3")
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.minimize((x - 1.0) ** 2)
    ev = NLPEvaluator(m)

    class _NanObjectiveEvaluator:
        def __getattr__(self, name):
            return getattr(ev, name)

        def evaluate_objective(self, x):
            return np.nan

    def fake_solve_nlp(evaluator, x0, options=None):
        return SimpleNamespace(status=SolveStatus.OPTIMAL, x=np.array([1.0]))

    import discopt.solvers.nlp_pounce as nlp_pounce

    monkeypatch.setattr(nlp_pounce, "solve_nlp", fake_solve_nlp)
    assert _solve_nlp_subproblem(
        _NanObjectiveEvaluator(),
        np.array([0.2]),
        np.array([0.0]),
        np.array([2.0]),
        nlp_solver="ipm",
    ) == (None, None)


# ---------------------------------------------------------------------------
# Upper-bound row stacking
# ---------------------------------------------------------------------------


def test_append_upper_bound_constraint_builds_and_stacks():
    row = np.array([1.0, -1.0])
    A, b = _append_upper_bound_constraint(None, None, row, 3.0)
    assert A.shape == (1, 2) and b.tolist() == [3.0]
    A2, b2 = _append_upper_bound_constraint(A, b, np.array([2.0, 0.0]), 5.0)
    assert A2.shape == (2, 2)
    np.testing.assert_allclose(A2.toarray(), [[1.0, -1.0], [2.0, 0.0]])
    np.testing.assert_allclose(b2, [3.0, 5.0])


# ---------------------------------------------------------------------------
# Cutoff OBBT
# ---------------------------------------------------------------------------


def _cutoff_obbt_args(model, UB, deadline, obbt_time_limit=5.0):
    terms = classify_nonlinear_terms(model)
    from discopt._jax.model_utils import flat_variable_bounds

    flat_lb, flat_ub = flat_variable_bounds(model)
    part_vars = sorted(terms.partition_candidates)
    return dict(
        model=model,
        terms=terms,
        disc_state=DiscretizationState(),
        oa_cuts=[],
        convhull_mode="disaggregated",
        UB=UB,
        flat_lb=flat_lb,
        flat_ub=flat_ub,
        part_vars=part_vars,
        part_lbs=[float(flat_lb[i]) for i in part_vars],
        part_ubs=[float(flat_ub[i]) for i in part_vars],
        n_orig=len(flat_lb),
        obbt_time_limit=obbt_time_limit,
        partition_scaling_factor=10.0,
        disc_abs_width_tol=1e-3,
        n_init_partitions=2,
        deadline=deadline,
        iteration=1,
        from_min_space=lambda v: v,
    )


def test_run_cutoff_obbt_deadline_exhausted_is_noop():
    m = _circle_model()
    args = _cutoff_obbt_args(m, UB=3.0, deadline=time.perf_counter() - 1.0)
    out_lb, out_ub, pv, plb, pub = _run_cutoff_obbt(**args)
    np.testing.assert_allclose(out_lb, args["flat_lb"])
    np.testing.assert_allclose(out_ub, args["flat_ub"])


def test_run_cutoff_obbt_no_candidates_is_noop():
    m = Model("lin")
    x = m.continuous("x", lb=0.0, ub=1.0, shape=(2,))
    m.subject_to(x[0] + x[1] >= 1.0)
    m.minimize(x[0])
    args = _cutoff_obbt_args(m, UB=1.0, deadline=time.perf_counter() + 30.0)
    assert args["part_vars"] == []
    out_lb, out_ub, *_ = _run_cutoff_obbt(**args)
    np.testing.assert_allclose(out_lb, args["flat_lb"])
    np.testing.assert_allclose(out_ub, args["flat_ub"])


def test_run_cutoff_obbt_tightens_and_stays_sound():
    # min x0+x1 s.t. x0*x1 >= 4 on [0,10]^2 with cutoff UB=5: any point with
    # objective <= 5 and x0*x1 >= 4 has each coordinate in [4/5's shadow] —
    # OBBT under the cutoff must keep the true optimum (2,2) inside the box.
    m = Model("tight")
    x = m.continuous("x", lb=0.0, ub=10.0, shape=(2,))
    m.subject_to(x[0] * x[1] >= 4.0)
    m.minimize(x[0] + x[1])
    args = _cutoff_obbt_args(m, UB=5.0, deadline=time.perf_counter() + 30.0)
    out_lb, out_ub, pv, plb, pub = _run_cutoff_obbt(**args)
    # Soundness: the optimum (2,2) survives.
    assert np.all(out_lb <= np.array([2.0, 2.0]) + 1e-6)
    assert np.all(out_ub >= np.array([2.0, 2.0]) - 1e-6)
    # Bounds only ever shrink.
    assert np.all(out_lb >= args["flat_lb"] - 1e-12)
    assert np.all(out_ub <= args["flat_ub"] + 1e-12)
    # Partition metadata stays consistent with the surviving partition vars.
    assert len(pv) == len(plb) == len(pub)


# ---------------------------------------------------------------------------
# Maximize-sense AMP result mapping (functional)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_amp_maximize_sense_returns_original_space_objective():
    m = Model("maxsense")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y == 1.0)
    m.maximize(x * y)
    res = m.solve(solver="amp", max_iter=8, time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(0.25, abs=1e-3)
    # Bound must be reported in the original (maximization) space: an upper
    # bound on the maximum, never below the objective.
    if res.bound is not None:
        assert res.bound >= res.objective - 1e-6
