"""Unit tests for the pure-logic helpers in ``discopt._jax.milp_relaxation`` (#87).

Focus areas:
  * constant folding and product/polynomial decomposition (the linearizer's
    matchers) — accept the documented shapes, abstain on near-misses;
  * the separable objective lower bound and its per-term matchers — every
    computed bound is validated against dense sampling of the true function
    over the box (soundness: bound <= sampled minimum).

All tests are sub-second pure logic; models are 1-3 variables and are never
solved.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.milp_relaxation import (
    _affine_var_base,
    _collect_affine_powers,
    _constant_value,
    _count_distinct_scalar_refs,
    _decompose_product,
    _eval_constant_expr,
    _even_power_term_lower_bound,
    _expr_to_polynomial,
    _integer_affine_cos_lower_bound,
    _integer_domain_values,
    _match_scaled_even_power,
    _normalize_convhull_formulation,
    _polynomial_lower_bound,
    _product_to_monomial,
    _reciprocal_term_lower_bound,
    _safe_x_exp_value,
    _scaled_affine_lower_bound,
    _separable_objective_lower_bound,
    _x_exp_upper_bound,
)
from discopt.modeling.core import Constant, Model

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------


def test_constant_value_scalar_only():
    assert _constant_value(Constant(2.5)) == 2.5
    assert _constant_value(Constant(np.array([1.0, 2.0]))) is None
    m = Model("cv")
    x = m.continuous("x", lb=0, ub=1)
    assert _constant_value(x) is None


def test_eval_constant_expr_folds_composites():
    c = Constant
    assert _eval_constant_expr(-c(2.5)) == -2.5
    assert _eval_constant_expr(abs(c(-3.0))) == 3.0
    assert _eval_constant_expr(c(2.0) + c(3.0)) == 5.0
    assert _eval_constant_expr(c(2.0) - c(3.0)) == -1.0
    assert _eval_constant_expr(c(-3.0) * c(-3.0)) == 9.0
    assert _eval_constant_expr(c(3.0) / c(2.0)) == 1.5
    assert _eval_constant_expr(c(2.0) ** c(3.0)) == 8.0
    # Abstentions: division by zero, complex result, variables.
    assert _eval_constant_expr(c(1.0) / c(0.0)) is None
    assert _eval_constant_expr(c(-1.0) ** c(0.5)) is None
    m = Model("ec")
    x = m.continuous("x", lb=0, ub=1)
    assert _eval_constant_expr(x + 1.0) is None
    assert _eval_constant_expr(-x) is None


# ---------------------------------------------------------------------------
# Product / affine-base / polynomial decomposition
# ---------------------------------------------------------------------------


def _model_xyz():
    m = Model("xyz")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    z = m.continuous("z", lb=0.5, ub=2.0)
    return m, x, y, z


def test_affine_var_base_matches_scaled_variables():
    m, x, y, z = _model_xyz()
    assert _affine_var_base(x, m) == (1.0, 0)
    assert _affine_var_base(2.0 * x, m) == (2.0, 0)
    assert _affine_var_base(y * 3.0, m) == (3.0, 1)
    assert _affine_var_base(y / 4.0, m) == (0.25, 1)
    coeff, idx = _affine_var_base(-(2.0 * z), m)
    assert coeff == -2.0 and idx == 2
    # Abstentions: additive structure, two variables, zero division.
    assert _affine_var_base(x + 1.0, m) is None
    assert _affine_var_base(x * y, m) is None
    assert _affine_var_base(x / 0.0, m) is None


def test_decompose_product_basic_and_negation():
    m, x, y, z = _model_xyz()
    assert _decompose_product(x * y, m) == (1.0, [0, 1])
    scalar, idxs = _decompose_product(2.0 * x * (-y), m)
    assert scalar == -2.0 and idxs == [0, 1]
    # neg(x) * x — the maximize-flip shape — must decompose with the sign peeled.
    scalar, idxs = _decompose_product(-x * x, m)
    assert scalar == -1.0 and idxs == [0, 0]
    # Composite constant factor folds exactly.
    scalar, idxs = _decompose_product((-Constant(3.0)) * x, m)
    assert scalar == -3.0 and idxs == [0]
    # Non-variable leaf -> undecomposable.
    assert _decompose_product(dm.sin(x) * y, m) is None


def test_decompose_product_monomial_collapse_and_pinned_powers():
    m, x, y, z = _model_xyz()
    # x*x*y collapses through the monomial aux column for x**2 (column 10).
    mono_map = {(0, 2): 10}
    scalar, idxs = _decompose_product(x * x * y, m, monomial_var_map=mono_map)
    assert scalar == 1.0 and idxs == [10, 1]
    # Missing aux column -> stays undecomposed but valid (raw repeated factors).
    scalar, idxs = _decompose_product(x * x * y, m, monomial_var_map={})
    assert scalar == 1.0 and sorted(idxs) == [0, 0, 1]
    # Integer power factor resolves through the monomial map.
    scalar, idxs = _decompose_product((y**2) * x, m, monomial_var_map={(1, 2): 11})
    assert scalar == 1.0 and set(idxs) == {11, 0}
    # A pinned fractional-power base folds to an exact constant.
    scalar, idxs = _decompose_product(
        (z**0.5) * x, m, pinned_value=lambda i: 4.0 if i == 2 else None
    )
    assert scalar == pytest.approx(2.0) and idxs == [0]
    # Pinned negative base with fractional power is complex -> undecomposable.
    assert (
        _decompose_product((z**0.5) * x, m, pinned_value=lambda i: -4.0 if i == 2 else None) is None
    )


def test_product_to_monomial():
    m, x, y, z = _model_xyz()
    assert _product_to_monomial(x * y, m) == (1.0, (0, 1))
    assert _product_to_monomial((x**2) * y, m) == (1.0, (0, 0, 1))
    scalar, mono = _product_to_monomial(-2.0 * y * x, m)
    assert scalar == -2.0 and mono == (0, 1)
    assert _product_to_monomial(dm.exp(x) * y, m) is None
    assert _product_to_monomial((x**0.5) * y, m) is None


def test_expr_to_polynomial_distributed_quadratic():
    m, x, y, z = _model_xyz()
    out = _expr_to_polynomial(x**2 - 2.0 * x * y + 1.0 + y / 2.0, m)
    assert out is not None
    const, terms = out
    assert const == pytest.approx(1.0)
    tdict = {}
    for coeff, mono in terms:
        tdict[mono] = tdict.get(mono, 0.0) + coeff
    assert tdict[(0, 0)] == pytest.approx(1.0)
    assert tdict[(0, 1)] == pytest.approx(-2.0)
    assert tdict[(1,)] == pytest.approx(0.5)
    # Non-polynomial leaves abstain.
    assert _expr_to_polynomial(dm.log(z) + x, m) is None
    assert _expr_to_polynomial(x / y, m) is None


def test_collect_affine_powers_finds_scaled_bases_only():
    m, x, y, z = _model_xyz()
    m.subject_to((2.0 * x) ** 3 + y**3 <= 1.0)
    m.minimize(dm.sin((0.5 * y) ** 4))
    found = _collect_affine_powers(m, already_lifted=set())
    keyed = {(var_idx, power): scale for _e, scale, var_idx, power in found}
    # (2x)**3 and (0.5y)**4 are scaled bases; bare y**3 is left to the
    # monomial machinery.
    assert keyed == {(0, 3): 2.0, (1, 4): 0.5}


def test_normalize_convhull_formulation():
    assert _normalize_convhull_formulation("disaggregated") == "disaggregated"
    assert _normalize_convhull_formulation("lambda") == "sos2"
    with pytest.raises(ValueError):
        _normalize_convhull_formulation("bogus")


# ---------------------------------------------------------------------------
# x*exp(x) helpers
# ---------------------------------------------------------------------------


def test_safe_x_exp_value_regimes():
    assert _safe_x_exp_value(-800.0) == 0.0  # exp underflow -> exact limit 0
    assert _safe_x_exp_value(1.0) == pytest.approx(np.e)
    assert _safe_x_exp_value(np.inf) is None
    assert _safe_x_exp_value(1e10) is None  # overflow guard


def test_x_exp_upper_bound_endpoints_and_unbounded():
    lb = np.array([-1.0])
    ub = np.array([2.0])
    # x*exp(x) is increasing for x > -1: max at ub.
    assert _x_exp_upper_bound(0, lb, ub) == pytest.approx(2.0 * np.exp(2.0))
    assert _x_exp_upper_bound(0, np.array([-np.inf]), ub) is None
    assert _x_exp_upper_bound(0, lb, np.array([1e10])) is None


# ---------------------------------------------------------------------------
# Integer-affine cos enumeration
# ---------------------------------------------------------------------------


def test_integer_domain_values():
    m = Model("idv")
    m.integer("i", lb=-1, ub=2)
    m.binary("b")
    m.continuous("x", lb=0, ub=1)
    from discopt._jax.milp_relaxation import _flat_variable_types

    types = _flat_variable_types(m)
    lb = np.array([-1.0, -0.5, 0.0])
    ub = np.array([2.0, 1.5, 1.0])
    assert list(_integer_domain_values(0, types, lb, ub)) == [-1, 0, 1, 2]
    # Binary domain clamps to {0, 1} even with looser bounds.
    assert list(_integer_domain_values(1, types, lb, ub)) == [0, 1]
    assert _integer_domain_values(2, types, lb, ub) is None  # continuous
    assert _integer_domain_values(0, types, np.array([-np.inf, 0, 0]), ub) is None


def test_integer_affine_cos_lower_bound_exact_enumeration():
    m = Model("cosint")
    i = m.integer("i", lb=0, ub=3)
    m.minimize(dm.cos(i * 1.0))
    lb = np.array([0.0])
    ub = np.array([3.0])
    bound = _integer_affine_cos_lower_bound(dm.cos(1.0 * i), 1.0, m, lb, ub)
    # Exact enumeration over i in {0..3}: min cos(i).
    expected = min(np.cos(k) for k in range(4))
    assert bound == pytest.approx(expected)
    # Negative scale flips the extremum.
    bound_neg = _integer_affine_cos_lower_bound(dm.cos(1.0 * i), -2.0, m, lb, ub)
    assert bound_neg == pytest.approx(min(-2.0 * np.cos(k) for k in range(4)))
    # Non-cos expressions abstain.
    assert _integer_affine_cos_lower_bound(dm.sin(1.0 * i), 1.0, m, lb, ub) is None


def test_integer_affine_cos_lower_bound_constant_arg_and_continuous_abstain():
    m = Model("cosconst")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    lb = np.array([0.0])
    ub = np.array([1.0])
    # An affine argument whose variable coefficients are all zero is a
    # constant: the bound is exact without any enumeration.
    const_cos = dm.cos(0.0 * x + 0.5)
    assert _integer_affine_cos_lower_bound(const_cos, 2.0, m, lb, ub) == pytest.approx(
        2.0 * np.cos(0.5)
    )
    # A continuous variable in the argument -> no finite enumeration.
    assert _integer_affine_cos_lower_bound(dm.cos(x), 1.0, m, lb, ub) is None


# ---------------------------------------------------------------------------
# Affine / polynomial / reciprocal / even-power lower bounds
# ---------------------------------------------------------------------------


def test_scaled_affine_lower_bound_vertex_choice():
    m = Model("aff")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=3.0)
    lb = np.array([-1.0, 0.0])
    ub = np.array([2.0, 3.0])
    # scale > 0: minimize 2x - y  -> 2*(-1) - 3 = -5, scaled by 1.
    assert _scaled_affine_lower_bound(2.0 * x - y + 1.0, 1.0, m, lb, ub) == pytest.approx(-4.0)
    # scale < 0: the bound comes from maximizing the affine form.
    assert _scaled_affine_lower_bound(2.0 * x - y + 1.0, -1.0, m, lb, ub) == pytest.approx(-5.0)
    # Unbounded relevant vertex -> abstain.
    assert _scaled_affine_lower_bound(x, 1.0, m, np.array([-np.inf, 0.0]), ub) is None
    # Non-affine expression -> abstain.
    assert _scaled_affine_lower_bound(x * y, 1.0, m, lb, ub) is None


def test_polynomial_lower_bound_soundness_and_abstentions():
    coeffs = {2: 1.0, 1: -2.0, 0: 0.5}  # x^2 - 2x + 0.5, min at x=1 -> -0.5
    bound = _polynomial_lower_bound(coeffs, -3.0, 3.0)
    xs = np.linspace(-3.0, 3.0, 601)
    true_min = min(sum(c * v**p for p, c in coeffs.items()) for v in xs)
    assert bound == pytest.approx(-0.5)
    assert bound <= true_min + 1e-9
    # Even positive leading coefficient is bounded below on an infinite box.
    assert _polynomial_lower_bound({2: 1.0, 1: -2.0}, -np.inf, np.inf) == pytest.approx(-1.0)
    # Unbounded directions with the wrong leading sign abstain.
    assert _polynomial_lower_bound({2: -1.0}, -np.inf, 1.0) is None
    assert _polynomial_lower_bound({2: -1.0}, -1.0, np.inf) is None
    assert _polynomial_lower_bound({3: 1.0}, -np.inf, 1.0) is None
    # Degenerate shapes.
    assert _polynomial_lower_bound({}, -1.0, 1.0) == 0.0
    assert _polynomial_lower_bound({0: 4.2}, -1.0, 1.0) == pytest.approx(4.2)


def test_reciprocal_term_lower_bound_positive_denominator():
    m = Model("recip")
    x = m.continuous("x", lb=0.0, ub=8.0)
    m.minimize(x)
    lb = np.array([0.0])
    ub = np.array([8.0])
    denom = 0.1 + (x - 4.0) ** 2
    # k < 0: minimized at D_lo = 0.1 -> -1/0.1 = -10.
    bound = _reciprocal_term_lower_bound(-1.0, denom, m, lb, ub)
    assert bound == pytest.approx(-10.0, rel=1e-6)
    # Soundness against sampling.
    xs = np.linspace(0.0, 8.0, 801)
    true_min = min(-1.0 / (0.1 + (v - 4.0) ** 2) for v in xs)
    assert bound <= true_min + 1e-9
    # k > 0 with a finite enclosure: minimized at D_hi.
    bound_pos = _reciprocal_term_lower_bound(2.0, denom, m, lb, ub)
    true_min_pos = min(2.0 / (0.1 + (v - 4.0) ** 2) for v in xs)
    assert bound_pos is not None and bound_pos <= true_min_pos + 1e-9
    # Denominator not provably positive -> abstain.
    assert _reciprocal_term_lower_bound(-1.0, (x - 4.0) ** 2, m, lb, ub) is None
    # Tiny numerator short-circuits to zero.
    assert _reciprocal_term_lower_bound(0.0, denom, m, lb, ub) == 0.0


def test_match_scaled_even_power():
    m, x, y, z = _model_xyz()
    matched = _match_scaled_even_power(100.0 * (y - x**2) ** 2, 1.0)
    assert matched is not None
    coeff, base, power = matched
    assert coeff == pytest.approx(100.0) and power == 2
    # Sign folds through the scale.
    matched_neg = _match_scaled_even_power(3.0 * (x + y) ** 4, -2.0)
    assert matched_neg is not None and matched_neg[0] == pytest.approx(-6.0)
    # Rejections: odd power, extra variable factor, two power factors.
    assert _match_scaled_even_power((x + y) ** 3, 1.0) is None
    assert _match_scaled_even_power(z * (x + y) ** 2, 1.0) is None
    assert _match_scaled_even_power(((x + y) ** 2) * ((x - y) ** 2), 1.0) is None


def test_count_distinct_scalar_refs():
    m, x, y, z = _model_xyz()
    assert _count_distinct_scalar_refs(x, m) == 1
    assert _count_distinct_scalar_refs(y - x**2, m) == 2
    assert _count_distinct_scalar_refs(dm.sin(x) + dm.sum([x * y, z]), m) == 3


def test_even_power_term_lower_bound_regimes():
    m, x, y, z = _model_xyz()
    base = y - x**2
    lb = np.array([-2.0, -2.0, 0.5])
    ub = np.array([2.0, 2.0, 2.0])
    # Nonnegative coefficient, base straddles zero -> exact floor 0.
    assert _even_power_term_lower_bound(100.0, base, 2, m, lb, ub) == 0.0
    # Base strictly positive on the box: z in [0.5, 2] -> min z^2 = 0.25.
    assert _even_power_term_lower_bound(1.0, z, 2, m, lb, ub) == pytest.approx(0.25)
    # Negative coefficient with a finite enclosure: -(max |base|)^n.
    bound = _even_power_term_lower_bound(-1.0, z, 2, m, lb, ub)
    assert bound == pytest.approx(-4.0)
    # Negative coefficient without a finite enclosure -> abstain.
    assert (
        _even_power_term_lower_bound(
            -1.0, z, 2, m, np.array([-2, -2, -np.inf]), np.array([2, 2, np.inf])
        )
        is None
    )


# ---------------------------------------------------------------------------
# Separable objective lower bound (integration of the matchers, with sampling)
# ---------------------------------------------------------------------------


def _sampled_min(fn, lb, ub, n=101):
    grids = [np.linspace(lo, hi, n) for lo, hi in zip(lb, ub)]
    mesh = np.meshgrid(*grids)
    pts = np.stack([g.ravel() for g in mesh], axis=1)
    return min(fn(p) for p in pts)


def test_separable_lower_bound_polynomial_soundness():
    m = Model("sep1")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    expr = 3.0 * x**2 - 2.0 * x + 0.5 * y + 1.0
    m.minimize(expr)
    lb = np.array([-2.0, -1.0])
    ub = np.array([3.0, 1.0])
    bound = _separable_objective_lower_bound(expr, m, lb, ub)
    assert bound is not None
    true_min = _sampled_min(lambda p: 3 * p[0] ** 2 - 2 * p[0] + 0.5 * p[1] + 1.0, lb, ub)
    assert bound <= true_min + 1e-9
    # The polynomial path is exact here (vertex/critical-point minimization).
    assert bound == pytest.approx(true_min, abs=1e-3)


def test_separable_lower_bound_rosenbrock_even_power():
    m = Model("rosen")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-1.0, ub=3.0)
    expr = 100.0 * (y - x**2) ** 2 + (1.0 - x) ** 2
    m.minimize(expr)
    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 3.0])
    bound = _separable_objective_lower_bound(expr, m, lb, ub)
    assert bound is not None
    assert bound <= 0.0 + 1e-9  # true minimum is 0 at (1, 1)
    assert bound >= -1e-9  # both terms are nonnegative -> floor at 0 is exact


def test_separable_lower_bound_x_exp_and_reciprocal():
    m = Model("mixed")
    x = m.continuous("x", lb=-3.0, ub=1.0)
    expr = x * dm.exp(x)
    m.minimize(expr)
    lb = np.array([-3.0])
    ub = np.array([1.0])
    bound = _separable_objective_lower_bound(expr, m, lb, ub)
    assert bound == pytest.approx(-1.0 / np.e)

    m2 = Model("recipobj")
    z = m2.continuous("z", lb=0.0, ub=8.0)
    expr2 = -1.0 / (0.1 + (z - 4.0) ** 2)
    m2.minimize(expr2)
    bound2 = _separable_objective_lower_bound(expr2, m2, np.array([0.0]), np.array([8.0]))
    assert bound2 == pytest.approx(-10.0, rel=1e-6)


def test_separable_lower_bound_abstains_on_unbounded_loss():
    # -x*exp(x) on an unbounded box is unbounded below: the matcher must
    # abstain (None), never fabricate a bound.
    m = Model("loss")
    x = m.continuous("x")
    expr = -x * dm.exp(x)
    m.minimize(expr)
    assert (
        _separable_objective_lower_bound(expr, m, np.array([-np.inf]), np.array([np.inf])) is None
    )


def test_separable_lower_bound_cos_terms():
    m = Model("costerm")
    i = m.integer("i", lb=0, ub=3)
    x = m.continuous("x", lb=-1.0, ub=1.0)
    expr = dm.cos(1.0 * i) + x
    m.minimize(expr)
    lb = np.array([0.0, -1.0])
    ub = np.array([3.0, 1.0])
    bound = _separable_objective_lower_bound(expr, m, lb, ub)
    assert bound is not None
    expected = min(np.cos(k) for k in range(4)) - 1.0
    assert bound == pytest.approx(expected)
    # Continuous cos falls back to the -|scale| floor (the additive flattener
    # only peels signs, so the bare/negated call is the recognized shape).
    m2 = Model("coscont")
    y = m2.continuous("y", lb=0.0, ub=6.0)
    expr2 = -dm.cos(y) + y
    m2.minimize(expr2)
    bound2 = _separable_objective_lower_bound(expr2, m2, np.array([0.0]), np.array([6.0]))
    assert bound2 == pytest.approx(-1.0 + 0.0)
    # A constant-scaled cos is not a recognized separable shape: the bound
    # abstains rather than guessing.
    m3 = Model("cosscaled")
    w = m3.continuous("w", lb=0.0, ub=6.0)
    expr3 = 2.0 * dm.cos(w)
    m3.minimize(expr3)
    assert _separable_objective_lower_bound(expr3, m3, np.array([0.0]), np.array([6.0])) is None
