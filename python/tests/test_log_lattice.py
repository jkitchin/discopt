"""Tests for the per-expression log-curvature *lattice* (issue #115).

These exercise :func:`discopt.gp.classify_log_curvature`, the DAG-propagating
log-space analogue of the x-space :class:`~discopt._jax.convexity.Curvature`
lattice. Distinct from ``test_log_curvature.py`` (the flat SymPy classifier):
here the verdict must *compose* through products, ratios, powers, sums and
max/min on the discopt :class:`Expression` DAG, and stay strictly separate from
the x-space verdict.
"""

import discopt.modeling as dm
import pytest
from discopt import Model
from discopt._jax.convexity import Curvature, classify_expr
from discopt._jax.convexity.log_lattice import (
    LogCurvature,
    classify_log_curvature,
    log_combine_product,
    log_combine_sum,
    log_negate,
    log_scale_pow,
)
from discopt.gp import is_log_convex, model_log_curvature
from discopt.modeling.core import Constant

pytestmark = pytest.mark.relaxation

AFF = LogCurvature.LOG_AFFINE
CVX = LogCurvature.LOG_CONVEX
CCV = LogCurvature.LOG_CONCAVE
UNK = LogCurvature.UNKNOWN


@pytest.fixture
def pos_model():
    """A model with two strictly-positive continuous variables."""
    m = Model("pos")
    x = m.continuous("x", lb=0.1, ub=10.0)
    y = m.continuous("y", lb=0.5, ub=20.0)
    return m, x, y


def lc(expr) -> LogCurvature:
    return classify_log_curvature(expr)


# ──────────────────────────────────────────────────────────────────────
# Lattice operators (unit-level, no DAG)
# ──────────────────────────────────────────────────────────────────────


def test_log_negate_swaps_convex_concave():
    assert log_negate(CVX) == CCV
    assert log_negate(CCV) == CVX
    assert log_negate(AFF) == AFF
    assert log_negate(UNK) == UNK


def test_log_combine_product_is_the_log_space_sum():
    # log(f*g) = log f + log g: affine is identity, like reinforce, cross → UNK.
    assert log_combine_product(AFF, AFF) == AFF
    assert log_combine_product(AFF, CVX) == CVX
    assert log_combine_product(AFF, CCV) == CCV
    assert log_combine_product(CVX, CVX) == CVX
    assert log_combine_product(CCV, CCV) == CCV
    assert log_combine_product(CVX, CCV) == UNK
    assert log_combine_product(CVX, UNK) == UNK


def test_log_combine_sum_preserves_log_convexity_only():
    # Sum of log-convex (incl. affine) is log-convex; concavity is NOT preserved.
    assert log_combine_sum(AFF, AFF) == CVX
    assert log_combine_sum(AFF, CVX) == CVX
    assert log_combine_sum(CVX, CVX) == CVX
    assert log_combine_sum(AFF, CCV) == UNK
    assert log_combine_sum(CCV, CCV) == UNK
    assert log_combine_sum(CVX, UNK) == UNK


def test_log_scale_pow():
    assert log_scale_pow(AFF, 2.0) == AFF
    assert log_scale_pow(AFF, -3.0) == AFF
    assert log_scale_pow(CVX, 2.0) == CVX
    assert log_scale_pow(CVX, -1.0) == CCV
    assert log_scale_pow(CCV, 0.5) == CCV
    assert log_scale_pow(CCV, -2.0) == CVX
    assert log_scale_pow(CVX, 0.0) == AFF  # f**0 == 1
    assert log_scale_pow(UNK, 2.0) == UNK


# ──────────────────────────────────────────────────────────────────────
# Leaves and positivity gating
# ──────────────────────────────────────────────────────────────────────


def test_positive_variable_is_log_affine(pos_model):
    _, x, y = pos_model
    assert lc(x) == AFF
    assert lc(y) == AFF


def test_positive_constant_is_log_affine():
    assert lc(Constant(3.0)) == AFF


def test_nonpositive_variable_abstains():
    m = Model("m")
    z = m.continuous("z", lb=-1.0, ub=10.0)  # lb <= 0 → not in GP domain
    w = m.continuous("w", lb=0.0, ub=10.0)  # lb == 0 → not strictly positive
    assert lc(z) == UNK
    assert lc(w) == UNK
    # A monomial built on a non-positive variable is not log-affine either.
    assert lc(2 * z**2) == UNK


def test_nonpositive_constant_abstains():
    assert lc(Constant(-2.0)) == UNK
    assert lc(Constant(0.0)) == UNK


# ──────────────────────────────────────────────────────────────────────
# Monomials → log-affine
# ──────────────────────────────────────────────────────────────────────


def test_monomial_is_log_affine(pos_model):
    _, x, y = pos_model
    assert lc(3 * x**2 * y) == AFF
    assert lc(x / y) == AFF
    assert lc(x**-2) == AFF
    assert lc(5 * x) == AFF
    assert lc(x**0) == AFF


def test_sqrt_of_monomial_is_log_affine(pos_model):
    _, x, y = pos_model
    assert lc(dm.sqrt(x * y)) == AFF


# ──────────────────────────────────────────────────────────────────────
# Posynomials → log-convex (composition is the point)
# ──────────────────────────────────────────────────────────────────────


def test_posynomial_is_log_convex(pos_model):
    _, x, y = pos_model
    assert lc(x * y + 2 * x**2 + 0.5 * y) == CVX
    assert lc(x + 5) == CVX  # monomial + positive constant


def test_product_of_posynomials_composes_to_log_convex(pos_model):
    # The flat SymPy classifier misses this (a Mul of two Adds); the lattice
    # composes LOG_CONVEX * LOG_CONVEX = LOG_CONVEX.
    _, x, y = pos_model
    assert lc((x + y) * (x + 1)) == CVX


def test_sqrt_of_posynomial_is_log_convex(pos_model):
    _, x, y = pos_model
    assert lc(dm.sqrt(x + y)) == CVX


def test_max_of_posynomials_is_log_convex(pos_model):
    _, x, y = pos_model
    assert lc(dm.maximum(x + y, 2 * x)) == CVX


# ──────────────────────────────────────────────────────────────────────
# Log-concave sources
# ──────────────────────────────────────────────────────────────────────


def test_reciprocal_of_posynomial_is_log_concave(pos_model):
    _, x, y = pos_model
    assert lc(1 / (x + y)) == CCV
    assert lc((x + y) ** -1) == CCV


def test_monomial_over_posynomial_is_log_concave(pos_model):
    _, x, y = pos_model
    assert lc((x * y) / (x + y)) == CCV


def test_min_of_monomials_is_log_concave(pos_model):
    _, x, y = pos_model
    assert lc(dm.minimum(x, y)) == CCV


# ──────────────────────────────────────────────────────────────────────
# Abstention (soundness: unknown, never a false log-curvature tag)
# ──────────────────────────────────────────────────────────────────────


def test_difference_abstains(pos_model):
    _, x, y = pos_model
    assert lc(x - y) == UNK


def test_negation_abstains(pos_model):
    _, x, _ = pos_model
    assert lc(-x) == UNK


def test_max_with_concave_arg_abstains(pos_model):
    # max needs every arg log-convex; a log-concave arg kills the verdict.
    _, x, y = pos_model
    assert lc(dm.maximum(x + y, 1 / (x + y))) == UNK


def test_min_with_convex_arg_abstains(pos_model):
    _, x, y = pos_model
    assert lc(dm.minimum(x, x + y)) == UNK


def test_sum_of_log_convex_and_log_concave_abstains(pos_model):
    # x*y (log-affine) + 1/(x+y) (log-concave) → not provably log-convex.
    _, x, y = pos_model
    assert lc(x * y + 1 / (x + y)) == UNK


def test_transcendental_abstains(pos_model):
    _, x, _ = pos_model
    assert lc(dm.exp(x)) == UNK
    assert lc(dm.log(x)) == UNK


def test_variable_exponent_abstains(pos_model):
    _, x, y = pos_model
    assert lc(x**y) == UNK


# ──────────────────────────────────────────────────────────────────────
# Strict separation from the x-space Curvature lattice (the soundness rule)
# ──────────────────────────────────────────────────────────────────────


def test_posynomial_is_log_convex_but_not_x_space_convex(pos_model):
    """A genuine posynomial is log-convex yet NOT convex in x.

    Folding one verdict into the other would mis-gate the x-space convex
    fast path — the invariant the two lattices exist to keep apart.
    """
    _, x, y = pos_model
    posy = x * y + 2 * x**2
    assert lc(posy) == CVX
    # x-space: x*y is indefinite on the positive orthant → not convex.
    assert classify_expr(posy, None) != Curvature.CONVEX


def test_log_curvature_returns_its_own_enum(pos_model):
    _, x, y = pos_model
    result = lc(x + y)
    assert isinstance(result, LogCurvature)
    assert not isinstance(result, Curvature)


def test_shared_subexpression_is_cached(pos_model):
    # A DAG with a reused node must classify consistently (id-cache).
    _, x, y = pos_model
    p = x + y
    assert lc(p * p) == CVX


# ──────────────────────────────────────────────────────────────────────
# Whole-model consumer: model_log_curvature (the #115 payoff)
# ──────────────────────────────────────────────────────────────────────


def _gp_model():
    m = Model("gp")
    x = m.continuous("x", lb=0.1, ub=10.0)
    y = m.continuous("y", lb=0.1, ub=10.0)
    return m, x, y


def test_model_log_curvature_full_gp():
    m, x, y = _gp_model()
    m.minimize(x * y + 0.5 / x)  # posynomial objective
    m.subject_to(x + y <= 5)
    m.subject_to(x * y == 2)
    report = model_log_curvature(m)
    assert report.objective is not None
    assert report.objective.log_convex is True
    assert all(c.log_convex for c in report.constraints)
    assert report.is_log_convex_program is True
    assert report.log_convex_rows == 2


def test_model_log_curvature_certifies_row_that_classify_gp_rejects():
    """`sqrt(x + y) <= t` is convex in y but not a monomial-splitter GP row.

    The lattice consumer certifies it; the whole-model monomial recogniser
    (`classify_gp` / `is_log_convex`) does not — the concrete payoff of a
    composing per-expression lattice.
    """
    m, x, y = _gp_model()
    t = m.continuous("t", lb=0.1, ub=10.0)
    m.minimize(t)
    m.subject_to(dm.sqrt(x + y) <= t)
    report = model_log_curvature(m)
    (row,) = report.constraints
    assert row.lhs == CVX  # sqrt(posynomial) is log-convex
    assert row.rhs == AFF  # t is a monomial
    assert row.log_convex is True
    assert report.is_log_convex_program is True
    # The whole-model monomial recogniser cannot see this structure.
    assert is_log_convex(m) is False


def test_model_log_curvature_reports_partial_log_convexity():
    m, x, y = _gp_model()
    m.minimize(x * y)
    m.subject_to(x + y <= 4)  # log-convex row
    m.subject_to(x - y <= 1)  # difference on the positive side → not certifiable
    report = model_log_curvature(m)
    assert [c.log_convex for c in report.constraints] == [True, False]
    assert report.log_convex_rows == 1
    assert report.is_log_convex_program is False


def test_model_log_curvature_equality_needs_both_sides_monomial():
    # monomial == monomial → affine equality in y (convex); posynomial == monomial → not.
    m, x, y = _gp_model()
    m.minimize(x)
    m.subject_to(x * y == 3)
    assert model_log_curvature(m).constraints[0].log_convex is True

    m2, a, b = _gp_model()
    m2.minimize(a)
    m2.subject_to(a + b == 3)
    assert model_log_curvature(m2).constraints[0].log_convex is False


def test_model_log_curvature_objective_sense_matters():
    # Maximising a monomial is convex-in-y; maximising a posynomial is not.
    m, x, y = _gp_model()
    m.maximize(x * y)
    assert model_log_curvature(m).objective.log_convex is True

    m2, a, b = _gp_model()
    m2.maximize(a + b)
    assert model_log_curvature(m2).objective.log_convex is False


def test_model_log_curvature_no_objective():
    m, x, y = _gp_model()
    m.subject_to(x + y <= 5)
    report = model_log_curvature(m)
    assert report.objective is None
    assert report.is_log_convex_program is True  # objective-free: all rows decide it
