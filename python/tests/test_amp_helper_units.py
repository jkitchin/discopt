"""Unit tests for the pure-logic helpers in ``discopt.solvers.amp`` (#87).

Each test exercises a documented behavior of a helper directly — matcher
accept/reject shapes, soundness of derived bounds (a computed minimum must
under-estimate sampled true values; a cutoff box must contain every point
whose objective beats the cutoff), and validation error paths. No JAX
compilation beyond what model construction triggers; everything is
sub-second.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.discretization import DiscretizationState
from discopt.modeling.core import Model, VarType
from discopt.solvers.amp import (
    _check_integer_feasible,
    _collect_product_factor_indices,
    _continuous_recovery_starts,
    _dedupe_candidate_points,
    _default_nlp_start,
    _default_obbt_time_limit_per_lp,
    _expr_all_vars_fixed,
    _expr_has_function,
    _expr_variable_indices,
    _flat_index_from_expr,
    _flat_var_index,
    _flatten_objective_power_terms,
    _integer_rounding_candidates,
    _merge_partition_vars,
    _normalize_initial_point,
    _normalize_partition_method,
    _normalize_partition_var_indices,
    _normalize_presolve_bt_algo,
    _polynomial_value,
    _refresh_partitions_for_bounds,
    _remaining_wall_time,
    _repair_inverted_bounds,
    _resolve_presolve_bt_time_limits,
    _scalar_constant,
    _solve_nlp_subproblem,
    _square_monomial_vars_in_expr,
    _tighten_bounds_with_objective_cutoff,
    _tighten_simple_power_group,
    _univariate_polynomial_minimum,
    _validate_partition_scaling_factor,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Bound bookkeeping helpers
# ---------------------------------------------------------------------------


def test_repair_inverted_bounds_snaps_to_midpoint():
    lb = np.array([0.0, 2.0, -1.0])
    ub = np.array([1.0, 1.0, -1.0])
    rlb, rub = _repair_inverted_bounds(lb, ub)
    # Untouched interval stays untouched; inverted one collapses to midpoint.
    assert rlb[0] == 0.0 and rub[0] == 1.0
    assert rlb[1] == rub[1] == pytest.approx(1.5)
    assert rlb[2] == rub[2] == -1.0
    # Inputs are not mutated when a repair happens.
    assert lb[1] == 2.0 and ub[1] == 1.0


def test_repair_inverted_bounds_noop_returns_same_arrays():
    lb = np.array([0.0, 1.0])
    ub = np.array([1.0, 2.0])
    rlb, rub = _repair_inverted_bounds(lb, ub)
    assert rlb is lb and rub is ub


def test_refresh_partitions_seeds_missing_partition_and_prunes_stale():
    m = Model("refresh")
    m.continuous("x", lb=0.0, ub=4.0)
    m.continuous("y", lb=0.0, ub=1.0)
    disc = DiscretizationState()
    disc.partitions[1] = np.array([0.0, 0.5, 1.0])  # becomes stale below
    flat_lb = np.array([0.0, 0.0])
    flat_ub = np.array([4.0, 1e-6])  # y collapses below the width tolerance
    part_vars, part_lbs, part_ubs = _refresh_partitions_for_bounds(
        m, disc, flat_lb, flat_ub, [0, 1], disc_abs_width_tol=1e-3, n_init_partitions=2
    )
    assert part_vars == [0]
    assert part_lbs == [0.0] and part_ubs == [4.0]
    # x had no partition -> seeded with linspace over its interval.
    np.testing.assert_allclose(disc.partitions[0], [0.0, 2.0, 4.0])
    # y's stale partition is dropped with the variable.
    assert 1 not in disc.partitions


def test_refresh_partitions_clips_existing_points_into_new_box():
    m = Model("refresh2")
    m.continuous("x", lb=0.0, ub=10.0)
    disc = DiscretizationState()
    disc.partitions[0] = np.array([0.0, 4.0, 10.0])
    flat_lb = np.array([2.0])
    flat_ub = np.array([6.0])
    _refresh_partitions_for_bounds(
        m, disc, flat_lb, flat_ub, [0], disc_abs_width_tol=1e-3, n_init_partitions=2
    )
    pts = disc.partitions[0]
    assert pts[0] == 2.0 and pts[-1] == 6.0
    assert np.all(np.diff(pts) > 0)
    assert 4.0 in pts  # interior point survives


# ---------------------------------------------------------------------------
# Expression matchers
# ---------------------------------------------------------------------------


def _model_xy():
    m = Model("xy")
    x = m.continuous("x", lb=0.0, ub=2.0, shape=(2,))
    z = m.continuous("z", lb=-1.0, ub=1.0)
    return m, x, z


def test_scalar_constant_accepts_scalar_and_rejects_array_and_var():
    m, x, z = _model_xy()
    two = (2.0 * z).left if hasattr(2.0 * z, "left") else None
    from discopt.modeling.core import Constant

    assert _scalar_constant(Constant(3.5)) == 3.5
    assert _scalar_constant(Constant(np.array([1.0, 2.0]))) is None
    assert _scalar_constant(z) is None
    del two, x


def test_flat_var_index_scalar_vector_and_slice():
    m = Model("fvi")
    x = m.continuous("x", lb=0, ub=1, shape=(2,))
    z = m.continuous("z", lb=0, ub=1)
    w = m.continuous("w", lb=0, ub=1, shape=(2, 2))
    assert _flat_var_index(z, m) == 2  # scalar Variable after x's 2 slots
    assert _flat_var_index(x, m) is None  # whole vector has no single index
    assert _flat_var_index(x[1], m) == 1
    assert _flat_var_index(w[1, 1], m) == 6  # row-major within w's block
    # Partial subscript addresses many scalars -> no flat index.
    assert _flat_var_index(w[1], m) is None


def test_flat_index_from_expr_variants():
    m = Model("fife")
    x = m.continuous("x", lb=0, ub=1, shape=(3,))
    z = m.continuous("z", lb=0, ub=1)
    assert _flat_index_from_expr(z, m) == 3
    assert _flat_index_from_expr(x, m) is None  # size > 1
    assert _flat_index_from_expr(x[2], m) == 2
    assert _flat_index_from_expr(x[0] + z, m) is None


def test_collect_product_factor_indices():
    m = Model("prod")
    x = m.continuous("x", lb=0, ub=1, shape=(2,))
    z = m.continuous("z", lb=0, ub=1)
    assert sorted(_collect_product_factor_indices(x[0] * x[1], m)) == [0, 1]
    assert sorted(_collect_product_factor_indices(2.0 * x[0] * z, m)) == [0, 2]
    # A non-product factor poisons the tree.
    assert _collect_product_factor_indices(x[0] * (x[1] + z), m) is None
    # Fewer than two variable factors is not a product term.
    assert _collect_product_factor_indices(2.0 * x[0], m) is None


def test_square_monomial_vars_detects_power_product_and_recurses():
    m = Model("sq")
    x = m.continuous("x", lb=0, ub=1, shape=(2,))
    z = m.continuous("z", lb=0, ub=1)
    assert _square_monomial_vars_in_expr(x[0] ** 2, m) == {0}
    assert _square_monomial_vars_in_expr(x[1] * x[1], m) == {1}
    # x*y is bilinear, not a square.
    assert _square_monomial_vars_in_expr(x[0] * x[1], m) == set()
    # Recursion through neg, function calls, and sums.
    assert _square_monomial_vars_in_expr(-(z**2), m) == {2}
    assert _square_monomial_vars_in_expr(dm.sin(z**2), m) == {2}
    # SumExpression and SumOverExpression recursion.
    assert _square_monomial_vars_in_expr(dm.sum(z**2), m) == {2}
    assert _square_monomial_vars_in_expr(dm.sum([x[0] * x[0], x[1] * x[1]]), m) == {0, 1}
    assert _square_monomial_vars_in_expr(x[0] ** 3, m) == set()


def test_expr_variable_indices_across_node_types():
    m = Model("evi")
    x = m.continuous("x", lb=0, ub=1, shape=(2,))
    z = m.continuous("z", lb=0, ub=1)
    assert _expr_variable_indices(x, m) == {0, 1}
    assert _expr_variable_indices(x[0] * z, m) == {0, 2}
    assert _expr_variable_indices(-z, m) == {2}
    assert _expr_variable_indices(dm.exp(x[1]), m) == {1}
    assert _expr_variable_indices(dm.sum(x), m) == {0, 1}
    cache: dict[int, frozenset[int]] = {}
    e = x[0] + z
    first = _expr_variable_indices(e, m, cache)
    assert cache and _expr_variable_indices(e, m, cache) == first


def test_expr_has_function_matches_named_calls_only():
    m = Model("ehf")
    x = m.continuous("x", lb=0, ub=1, shape=(2,))
    e = x[0] * dm.exp(x[1])
    assert _expr_has_function(e, {"exp"})
    assert not _expr_has_function(e, {"log"})
    assert _expr_has_function(-dm.sin(x[0]), {"sin"})
    assert _expr_has_function(dm.sum(dm.log(x)), {"log"})
    cache: dict = {}
    assert _expr_has_function(e, {"exp"}, cache) and cache


def test_expr_all_vars_fixed():
    m = Model("fixed")
    x = m.continuous("x", lb=0, ub=1, shape=(2,))
    flat_lb = np.array([0.5, 0.0])
    flat_ub = np.array([0.5, 1.0])
    assert _expr_all_vars_fixed(x[0] ** 2, m, flat_lb, flat_ub)
    assert not _expr_all_vars_fixed(x[0] + x[1], m, flat_lb, flat_ub)
    from discopt.modeling.core import Constant

    # No variables at all -> not "fixed" (constant rows are handled elsewhere).
    assert not _expr_all_vars_fixed(Constant(1.0), m, flat_lb, flat_ub)


# ---------------------------------------------------------------------------
# Separable objective flattening + cutoff tightening
# ---------------------------------------------------------------------------


def test_flatten_objective_power_terms_polynomial_shapes():
    m = Model("poly")
    x = m.continuous("x", lb=-2, ub=2, shape=(2,))
    groups: dict[int, dict[int, float]] = {}
    # 3*x0^2 - x1 + x0/2 + 1  (division folds into the linear coefficient)
    expr = 3.0 * x[0] ** 2 - x[1] + x[0] / 2.0 + 1.0
    const = _flatten_objective_power_terms(expr, m, 1.0, groups)
    assert const == pytest.approx(1.0)
    assert groups[0][2] == pytest.approx(3.0)
    assert groups[0][1] == pytest.approx(0.5)
    assert groups[1][1] == pytest.approx(-1.0)


def test_flatten_objective_power_terms_neg_sum_and_sumover():
    m = Model("neg")
    x = m.continuous("x", lb=-1, ub=1, shape=(2,))
    groups: dict[int, dict[int, float]] = {}
    const = _flatten_objective_power_terms(-(x[0] ** 4), m, 1.0, groups)
    assert const == 0.0 and groups[0][4] == pytest.approx(-1.0)

    groups2: dict[int, dict[int, float]] = {}
    const2 = _flatten_objective_power_terms(dm.sum([x[0] ** 2, x[1] ** 2, 1.5]), m, 1.0, groups2)
    assert const2 == pytest.approx(1.5)
    assert groups2[0][2] == pytest.approx(1.0) and groups2[1][2] == pytest.approx(1.0)

    groups3: dict[int, dict[int, float]] = {}
    const3 = _flatten_objective_power_terms(dm.sum(x[0] ** 2), m, 1.0, groups3)
    assert const3 == 0.0 and groups3[0][2] == pytest.approx(1.0)


def test_flatten_objective_power_terms_rejects_unsupported_shapes():
    m = Model("rej")
    x = m.continuous("x", lb=0.5, ub=2, shape=(2,))
    for bad in (
        dm.exp(x[0]),  # function call
        x[0] * x[1],  # bilinear
        x[0] ** 2.5,  # non-integer exponent
        x[0] ** (-1),  # non-positive exponent
        (x[0] + x[1]) ** 2,  # power of a non-variable
        x[0] / x[1],  # non-constant denominator
    ):
        assert _flatten_objective_power_terms(bad, m, 1.0, {}) is None


def test_univariate_polynomial_minimum_uses_interior_critical_points():
    # x^2 - 2x has its minimum -1 at the interior point x=1.
    terms = {2: 1.0, 1: -2.0}
    assert _univariate_polynomial_minimum(terms, 0.0, 3.0) == pytest.approx(-1.0)
    # Property: the reported minimum under-estimates the polynomial on a grid.
    xs = np.linspace(0.0, 3.0, 301)
    vals = [_polynomial_value(terms, float(v)) for v in xs]
    assert _univariate_polynomial_minimum(terms, 0.0, 3.0) <= min(vals) + 1e-12


def test_tighten_simple_power_group_shapes():
    # Linear, positive coefficient: 2x <= 6 -> ub 3.
    assert _tighten_simple_power_group({1: 2.0}, 6.0, -10, 10) == (-10, 3.0)
    # Linear, negative coefficient: -2x <= 6 -> lb -3.
    assert _tighten_simple_power_group({1: -2.0}, 6.0, -10, 10) == (-3.0, 10)
    # Odd power: x^3 <= 8 -> ub 2.
    lb, ub = _tighten_simple_power_group({3: 1.0}, 8.0, -10, 10)
    assert (lb, ub) == (-10, pytest.approx(2.0))
    # Even power: x^2 <= 4 -> |x| <= 2.
    lb, ub = _tighten_simple_power_group({2: 1.0}, 4.0, -10, 10)
    assert lb == pytest.approx(-2.0) and ub == pytest.approx(2.0)
    # Abstentions: infinite rhs, multiple degrees, negative even-power rhs.
    assert _tighten_simple_power_group({1: 1.0}, np.inf, -1, 1) == (-1, 1)
    assert _tighten_simple_power_group({1: 1.0, 2: 1.0}, 1.0, -1, 1) == (-1, 1)
    assert _tighten_simple_power_group({2: 1.0}, -1.0, -1, 1) == (-1, 1)


def test_cutoff_tightening_is_sound_and_tightens():
    m = Model("cut")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize(x**2)
    flat_lb = np.array([-10.0])
    flat_ub = np.array([10.0])
    new_lb, new_ub = _tighten_bounds_with_objective_cutoff(m, flat_lb, flat_ub, cutoff=4.0)
    assert new_lb[0] > -10.0 and new_ub[0] < 10.0
    # Soundness: every x with x^2 <= cutoff must survive in the box.
    for v in np.linspace(-2.0, 2.0, 41):
        assert new_lb[0] - 1e-9 <= v <= new_ub[0] + 1e-9


def test_cutoff_tightening_abstains_on_maximize_and_infinite_boxes():
    m = Model("cutmax")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.maximize(x**2)
    lb = np.array([-10.0])
    ub = np.array([10.0])
    out_lb, out_ub = _tighten_bounds_with_objective_cutoff(m, lb, ub, cutoff=4.0)
    assert out_lb is lb and out_ub is ub

    m2 = Model("cutinf")
    y = m2.continuous("y")  # unbounded box -> group minimum undefined
    m2.minimize(y**2)
    lb2 = np.array([-np.inf])
    ub2 = np.array([np.inf])
    out_lb2, out_ub2 = _tighten_bounds_with_objective_cutoff(m2, lb2, ub2, cutoff=4.0)
    assert out_lb2 is lb2 and out_ub2 is ub2


def test_cutoff_tightening_abstains_when_cutoff_below_reachable_minimum():
    m = Model("cutlow")
    x = m.continuous("x", lb=1.0, ub=3.0)
    m.minimize(x**2)
    lb = np.array([1.0])
    ub = np.array([3.0])
    # min over the box is 1.0 > cutoff -> abstain (no fabricated empty box).
    out_lb, out_ub = _tighten_bounds_with_objective_cutoff(m, lb, ub, cutoff=0.5)
    assert out_lb is lb and out_ub is ub


# ---------------------------------------------------------------------------
# Start-point construction / normalization
# ---------------------------------------------------------------------------


def test_default_nlp_start_semi_infinite_domains():
    lb = np.array([0.0, 2.0, -np.inf, -np.inf])
    ub = np.array([4.0, np.inf, -1.0, np.inf])
    x0 = _default_nlp_start(lb, ub)
    np.testing.assert_allclose(x0, [2.0, 2.0, -1.0, 0.0])


def test_continuous_recovery_starts_dedupes_and_handles_unbounded():
    lb = np.array([-np.inf, 0.0])
    ub = np.array([np.inf, 0.0])
    starts = _continuous_recovery_starts(lb, ub, initial_point=np.array([0.5, 0.0]))
    # Initial point first, duplicates removed, all within bounds for the
    # bounded coordinate.
    assert np.allclose(starts[0], [0.5, 0.0])
    keys = {tuple(map(float, s)) for s in starts}
    assert len(keys) == len(starts)
    for s in starts:
        assert 0.0 <= s[1] <= 0.0


def test_dedupe_candidate_points_preserves_insertion_order():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    out = _dedupe_candidate_points([a, b, a.copy()])
    assert len(out) == 2
    np.testing.assert_allclose(out[0], a)
    np.testing.assert_allclose(out[1], b)


def test_normalize_initial_point_validation():
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    assert _normalize_initial_point(None, 2, lb, ub) is None
    clipped = _normalize_initial_point(np.array([2.0, -1.0]), 2, lb, ub)
    np.testing.assert_allclose(clipped, [1.0, 0.0])
    with pytest.raises(ValueError, match="length"):
        _normalize_initial_point(np.array([1.0]), 2, lb, ub)
    with pytest.raises(ValueError, match="finite"):
        _normalize_initial_point(np.array([np.nan, 0.0]), 2, lb, ub)


def test_remaining_wall_time():
    import time as _time

    assert _remaining_wall_time(None) is None
    assert _remaining_wall_time(_time.perf_counter() - 1.0) == 0.0
    assert _remaining_wall_time(_time.perf_counter() + 60.0) > 50.0


def test_solve_nlp_subproblem_rejects_exhausted_budget():
    assert _solve_nlp_subproblem(None, np.zeros(1), np.zeros(1), np.ones(1), time_limit=0.0) == (
        None,
        None,
    )


# ---------------------------------------------------------------------------
# Integer rounding helpers
# ---------------------------------------------------------------------------


def _fake_int_model(lb, ub, var_type=VarType.INTEGER):
    var = SimpleNamespace(
        var_type=var_type,
        size=int(np.asarray(lb).size),
        lb=np.asarray(lb, dtype=np.float64),
        ub=np.asarray(ub, dtype=np.float64),
    )
    return SimpleNamespace(_variables=[var])


def test_check_integer_feasible():
    m = _fake_int_model([0.0], [10.0])
    assert _check_integer_feasible(np.array([3.0]), m)
    assert _check_integer_feasible(np.array([3.0 + 5e-6]), m)
    assert not _check_integer_feasible(np.array([3.4]), m)


def test_integer_rounding_candidates_small_domain_nearest_first():
    m = _fake_int_model([0.0], [3.0])
    cands = _integer_rounding_candidates(np.array([1.2]), m)
    values = [float(c[0]) for c in cands]
    # Full enumeration of {0..3}, nearest to 1.2 first.
    assert values[0] == 1.0
    assert sorted(values) == [0.0, 1.0, 2.0, 3.0]
    # All candidates respect the variable bounds.
    assert all(0.0 <= v <= 3.0 for v in values)


def test_integer_rounding_candidates_empty_integer_domain_uses_neighborhood():
    # lb=2.7, ub=2.2 leaves no integer in [ceil(lb), floor(ub)]; the helper
    # falls back to an unclamped rounding neighborhood instead of enumerating.
    m = _fake_int_model([2.7], [2.2])
    cands = _integer_rounding_candidates(np.array([2.4]), m)
    values = [float(c[0]) for c in cands]
    assert values, "neighborhood fallback must still propose candidates"
    assert {2.0, 3.0} <= set(values)  # floor/ceil of the clipped point
    assert len(values) == len(set(values))


def test_integer_rounding_candidates_no_integexcept_returns_base():
    m = _fake_int_model([0.0], [1.0], var_type=VarType.CONTINUOUS)
    x = np.array([0.37])
    cands = _integer_rounding_candidates(x, m)
    assert len(cands) == 1
    np.testing.assert_allclose(cands[0], x)


# ---------------------------------------------------------------------------
# Option validation / normalization
# ---------------------------------------------------------------------------


def test_validate_partition_scaling_factor():
    assert _validate_partition_scaling_factor(4) == 4.0
    for bad in ("abc", None, 1.0, 0.5, np.inf, np.nan):
        with pytest.raises(ValueError, match="finite number > 1.0"):
            _validate_partition_scaling_factor(bad)


def test_normalize_partition_var_indices_validation():
    assert _normalize_partition_var_indices([2, 0, 2], 3, source="disc_var_pick") == [2, 0]
    with pytest.raises(ValueError, match="iterable"):
        _normalize_partition_var_indices(None, 3, source="disc_var_pick")
    with pytest.raises(ValueError, match="iterable"):
        _normalize_partition_var_indices(7, 3, source="disc_var_pick")
    with pytest.raises(ValueError, match="non-integer"):
        _normalize_partition_var_indices([True], 3, source="disc_var_pick")
    with pytest.raises(ValueError, match="non-integer"):
        _normalize_partition_var_indices([1.5], 3, source="disc_var_pick")
    with pytest.raises(ValueError, match="outside valid range"):
        _normalize_partition_var_indices([3], 3, source="disc_var_pick")


def test_merge_partition_vars_keeps_prefix_order():
    assert _merge_partition_vars([3, 1], [1, 2, 3, 4]) == [3, 1, 2, 4]


def test_normalize_partition_method_aliases():
    assert _normalize_partition_method("auto", None) == "auto"
    assert _normalize_partition_method("auto", lambda ctx: [0]) == "auto"
    assert _normalize_partition_method("auto", "all") == "max_cover"
    assert _normalize_partition_method("auto", "adaptive") == "adaptive_vertex_cover"
    assert _normalize_partition_method("auto", 0) == "max_cover"
    assert _normalize_partition_method("auto", 1) == "min_vertex_cover"
    assert _normalize_partition_method("auto", 2) == "auto"
    assert _normalize_partition_method("auto", 3) == "adaptive_vertex_cover"
    with pytest.raises(ValueError, match="disc_var_pick string"):
        _normalize_partition_method("auto", "bogus")
    with pytest.raises(ValueError, match="disc_var_pick integer"):
        _normalize_partition_method("auto", 9)


def test_normalize_presolve_bt_algo_aliases():
    for alias in (1, "1", "lp", "OBBT", "lp-obbt", "linear"):
        assert _normalize_presolve_bt_algo(alias) == "lp"
    for alias in (2, "2", "tmc", "partitioned", "incumbent_partitioned"):
        assert _normalize_presolve_bt_algo(alias) == "incumbent_partitioned"
    with pytest.raises(ValueError, match="presolve_bt_algo"):
        _normalize_presolve_bt_algo("bogus")
    with pytest.raises(ValueError, match="presolve_bt_algo"):
        _normalize_presolve_bt_algo(3)


def test_resolve_presolve_bt_time_limits():
    # Historical default: 10% of remaining, capped at 10s.
    total, per = _resolve_presolve_bt_time_limits(200.0, 4, None, None)
    assert total == pytest.approx(10.0)
    assert per == pytest.approx(_default_obbt_time_limit_per_lp(200.0, 4))
    # Explicit total budget is split across 2n subproblems.
    total, per = _resolve_presolve_bt_time_limits(100.0, 5, 20.0, None)
    assert total == pytest.approx(20.0) and per == pytest.approx(2.0)
    # Per-MIP cap clamps the per-subproblem share.
    _, per = _resolve_presolve_bt_time_limits(100.0, 5, 20.0, 0.5)
    assert per == pytest.approx(0.5)
    # Exhausted budget -> zeros.
    assert _resolve_presolve_bt_time_limits(0.0, 5, None, None) == (0.0, 0.0)
    with pytest.raises(ValueError, match="presolve_bt_time_limit"):
        _resolve_presolve_bt_time_limits(10.0, 2, -1.0, None)
    with pytest.raises(ValueError, match="presolve_bt_mip_time_limit"):
        _resolve_presolve_bt_time_limits(10.0, 2, None, -1.0)
