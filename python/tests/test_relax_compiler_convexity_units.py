"""Coverage-driven unit/property tests (#87, final round) for two modules.

(A third, ``relaxation_compiler``, was covered here until it was retired with the
McCormick-"nlp" route in #1873; see docs/dev/nlp-mccormick-route-retirement.md.)

* ``discopt._relax.convexity.patterns`` — every CONVEX/CONCAVE verdict is
  cross-checked against sampled midpoint convexity of the actual function on
  the declared box; near-miss shapes must abstain (``None``).
* ``discopt._relax.differentiable`` — parametric-compiler values and gradients
  are checked against hand-computed values and central finite differences;
  solver-facing error paths must raise; L3 accessors are checked against the
  analytic sensitivities of tiny NLPs.

Suspected genuine bugs are asserted at their CORRECT behavior and marked
``xfail(strict=False)`` (see the individual test docstrings).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import pytest

pytest.importorskip("jax", reason="JAX is the optional discopt[jax] extra (#1873)")

import jax
import jax.numpy as jnp
import numpy as np
from discopt._relax.convexity import Curvature, classify_expr
from discopt._relax.convexity import patterns as pat
from discopt._relax.differentiable import (
    DiffSolveResult,
    DiffSolveResultL3,
    _compile_parametric_constraint,
    _compile_parametric_node,
    _compile_parametric_objective,
    _compute_param_offset,
    _compute_sensitivity_at_solution,
    _differentiable_solve_integer,
    _dispatch_nlp_solve,
    _make_jax_differentiable_solve,
    differentiable_solve,
    differentiable_solve_l3,
    find_active_set,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    UnaryOp,
)

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────
# Shared harnesses
# ──────────────────────────────────────────────────────────────────────


def _grid(lb, ub, n):
    grids = [np.linspace(a, b, n) for a, b in zip(lb, ub)]
    return np.stack([g.ravel() for g in np.meshgrid(*grids)], axis=1)


def _assert_sampled_curvature(f_np, lb, ub, verdict, n_pairs=60, tol=1e-9):
    """Verify a CONVEX/CONCAVE verdict by sampled midpoint convexity."""
    rng = np.random.default_rng(20260718)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    for _ in range(n_pairs):
        a = rng.uniform(lb, ub)
        b = rng.uniform(lb, ub)
        gap = f_np(0.5 * (a + b)) - 0.5 * (f_np(a) + f_np(b))
        if verdict == Curvature.CONVEX:
            assert gap <= tol, f"midpoint convexity violated: gap={gap} at {a}, {b}"
        elif verdict == Curvature.CONCAVE:
            assert gap >= -tol, f"midpoint concavity violated: gap={gap} at {a}, {b}"
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected verdict {verdict}")


def _fd_grad(f, x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g


# ──────────────────────────────────────────────────────────────────────
# convexity/patterns: verdicts checked against sampled curvature
# ──────────────────────────────────────────────────────────────────────


class TestQuadraticCurvature:
    def test_verdicts_match_sampled_curvature(self):
        m = Model("q")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        box = ([-2.0, -2.0], [2.0, 2.0])

        assert pat.quadratic_curvature(x * x + y * y, m) == Curvature.CONVEX
        _assert_sampled_curvature(lambda p: p[0] ** 2 + p[1] ** 2, *box, Curvature.CONVEX)

        assert pat.quadratic_curvature(-(x * x) - 2.0 * (y * y), m) == Curvature.CONCAVE
        _assert_sampled_curvature(lambda p: -(p[0] ** 2) - 2.0 * p[1] ** 2, *box, Curvature.CONCAVE)

        assert pat.quadratic_curvature(2.0 * x + y - 1.0, m) == Curvature.AFFINE
        # Indefinite saddle stays UNKNOWN; non-polynomial abstains entirely.
        assert pat.quadratic_curvature(x * y, m) == Curvature.UNKNOWN
        assert pat.quadratic_curvature(dm.sin(x), m) is None


class TestHomogeneousPsdQuadratic:
    def test_matmul_and_variable_forms(self):
        m = Model("v")
        v = m.continuous("v", shape=(2,), lb=-1.0, ub=1.0)
        A = np.array([[1.0, 2.0], [0.0, 1.0]])
        mm_left = MatMulExpression(Constant(A), v)
        mm_right = MatMulExpression(v, Constant(A))
        assert pat.is_homogeneous_psd_quadratic(dm.sum(mm_left * mm_left), m)
        assert pat.is_homogeneous_psd_quadratic(dm.sum(mm_right * mm_right), m)
        assert pat.is_homogeneous_psd_quadratic(dm.sum(v * v), m)
        # ||Ax||^2 really is convex on the box.
        _assert_sampled_curvature(
            lambda p: float(np.sum((A @ p) ** 2)), [-1.0] * 2, [1.0] * 2, Curvature.CONVEX
        )

    def test_rejects_mismatched_and_foreign_forms(self):
        m = Model("v")
        v = m.continuous("v", shape=(2,), lb=-1.0, ub=1.0)
        w = m.continuous("w", shape=(2,), lb=-1.0, ub=1.0)
        # sum(v*w) = v.w is an indefinite quadratic, not a PSD square sum.
        assert pat._sum_of_squares_linear_matrix(dm.sum(v * w), m) is None
        assert not pat.is_homogeneous_psd_quadratic(dm.sum(v * w), m)
        # Shape-mismatched matmul (either orientation) and foreign variables abstain.
        assert pat._linear_vector_matrix(MatMulExpression(Constant(np.eye(3)), v), m) is None
        assert pat._linear_vector_matrix(MatMulExpression(v, Constant(np.eye(3))), m) is None
        other = Model("other").continuous("z", lb=0.0, ub=1.0)
        assert pat._linear_vector_matrix(other, m) is None
        other_vec = Model("other2").continuous("zv", shape=(2,), lb=0.0, ub=1.0)
        eye2 = Constant(np.eye(2))
        assert pat._linear_vector_matrix(MatMulExpression(eye2, other_vec), m) is None
        assert pat._linear_vector_matrix(MatMulExpression(other_vec, eye2), m) is None
        # A non-affine expression abstains through the final fall-through.
        assert pat._linear_vector_matrix(dm.sum(v * v), m) is None
        # Sum-of-squares rejects a non-product operand and non-linear factors.
        assert pat._sum_of_squares_linear_matrix(dm.sum(v + w), m) is None
        assert pat._sum_of_squares_linear_matrix(dm.sum((v + w) * (v + w)), m) is None

    def test_linear_vector_matrix_1d_forms(self):
        # A 1-D constant vector acts as a single row (left) / column (right).
        m = Model("v1")
        m.continuous("x", lb=0.0, ub=1.0)
        m.continuous("y", lb=0.0, ub=1.0)
        v = m.continuous("v", shape=(2,), lb=-1.0, ub=1.0)
        c = np.array([1.0, 2.0])
        left = pat._linear_vector_matrix(MatMulExpression(Constant(c), v), m)
        right = pat._linear_vector_matrix(MatMulExpression(v, Constant(c)), m)
        np.testing.assert_allclose(left, [[0.0, 0.0, 1.0, 2.0]])
        np.testing.assert_allclose(right, [[0.0, 0.0, 1.0, 2.0]])


class TestProductPattern:
    @pytest.fixture()
    def model(self):
        m = Model("pp")
        m.continuous("x", lb=0.5, ub=2.0)
        m.continuous("y", lb=0.5, ub=2.0)
        return m

    def _xy(self, m):
        return m._variables[0], m._variables[1]

    def test_perspective_of_exp_is_convex(self, model):
        x, y = self._xy(model)
        verdict = pat.classify_product_pattern(y * dm.exp(x / y), model, classify_expr, {})
        assert verdict == Curvature.CONVEX
        _assert_sampled_curvature(
            lambda p: p[1] * np.exp(p[0] / p[1]), [0.5, 0.5], [2.0, 2.0], Curvature.CONVEX
        )

    def test_exp_times_reciprocal_power_is_convex(self, model):
        x, y = self._xy(model)
        verdict = pat.classify_product_pattern(dm.exp(x) * y ** (-1.0), model, classify_expr, {})
        assert verdict == Curvature.CONVEX
        _assert_sampled_curvature(
            lambda p: np.exp(p[0]) / p[1], [0.5, 0.5], [2.0, 2.0], Curvature.CONVEX
        )

    def test_geometric_mean_is_concave(self, model):
        x, y = self._xy(model)
        verdict = pat.classify_product_pattern(x**0.5 * y**0.5, model, classify_expr, {})
        assert verdict == Curvature.CONCAVE
        _assert_sampled_curvature(
            lambda p: np.sqrt(p[0] * p[1]), [0.5, 0.5], [2.0, 2.0], Curvature.CONCAVE
        )

    def test_nonpositive_exponent_monomial_is_convex(self, model):
        x, y = self._xy(model)
        verdict = pat.classify_product_pattern(x ** (-1.0) * y ** (-2.0), model, classify_expr, {})
        assert verdict == Curvature.CONVEX
        _assert_sampled_curvature(
            lambda p: p[0] ** -1.0 * p[1] ** -2.0, [0.5, 0.5], [2.0, 2.0], Curvature.CONVEX
        )

    def test_one_big_exponent_signomial_is_convex(self, model):
        x, y = self._xy(model)
        verdict = pat.classify_product_pattern(x**2.0 * y ** (-1.0), model, classify_expr, {})
        assert verdict == Curvature.CONVEX
        _assert_sampled_curvature(
            lambda p: p[0] ** 2 / p[1], [0.5, 0.5], [2.0, 2.0], Curvature.CONVEX
        )

    def test_negative_leading_constant_flips_curvature(self, model):
        x, y = self._xy(model)
        expr = (-2.0) * (x ** (-1.0) * y ** (-1.0))
        verdict = pat.classify_product_pattern(expr, model, classify_expr, {})
        assert verdict == Curvature.CONCAVE
        _assert_sampled_curvature(
            lambda p: -2.0 / (p[0] * p[1]), [0.5, 0.5], [2.0, 2.0], Curvature.CONCAVE
        )

    def test_rejections(self, model):
        x, y = self._xy(model)
        # A product of constants alone has no core to classify.
        const_prod = BinaryOp("*", Constant(np.array(2.0)), Constant(np.array(3.0)))
        assert pat.classify_product_pattern(const_prod, model, classify_expr, {}) is None
        # Variable exponent breaks the power-factor extraction in both passes.
        assert pat.classify_product_pattern(x * (y**x), model, classify_expr, {}) is None
        # Geometric-mean weights on a box that is not provably nonneg / positive
        # cannot be classified when total exponent exceeds 1.
        m2 = Model("neg")
        a = m2.continuous("a", lb=-1.0, ub=1.0)
        b = m2.continuous("b", lb=-1.0, ub=1.0)
        assert pat.classify_product_pattern(a**0.5 * b**0.6, m2, classify_expr, {}) is None


class TestSqrtAndPerspectivePatterns:
    def test_sqrt_rejects_nonpower_factor(self):
        m = Model("s")
        x = m.continuous("x", lb=0.5, ub=2.0)
        y = m.continuous("y", lb=0.5, ub=2.0)
        assert pat.classify_sqrt_pattern(y**x, m, classify_expr, {}) is None

    def test_perspective_product_rejects_non_product(self):
        m = Model("p")
        x = m.continuous("x", lb=0.5, ub=2.0)
        y = m.continuous("y", lb=0.5, ub=2.0)
        assert pat.classify_perspective_product(x + y, m, classify_expr, {}) is None


class TestAffineNormSquare:
    def test_accepts_weighted_affine_squares_plus_const(self):
        m = Model("ns")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        expr = (x - y) ** 2 + 3.0 * (x - 1.0) ** 2 + 0.5
        assert pat.is_affine_norm_square(expr, m)
        # A SumExpression wrapper flattens into the same recognizer.
        assert pat.is_affine_norm_square(dm.sum(x * x), m)
        # sqrt of it is a Euclidean norm of an affine map: convex.
        _assert_sampled_curvature(
            lambda p: np.sqrt((p[0] - p[1]) ** 2 + 3.0 * (p[0] - 1.0) ** 2 + 0.5),
            [-2.0, -2.0],
            [2.0, 2.0],
            Curvature.CONVEX,
        )

    def test_rejects_negative_const_and_negative_weight_and_nonaffine(self):
        m = Model("ns2")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        assert not pat.is_affine_norm_square((x - y) ** 2 + (-1.0), m)
        assert not pat.is_affine_norm_square((x - y) ** 2 + (-2.0) * (x**2), m)
        assert not pat.is_affine_norm_square((x * y) ** 2, m)

    def test_peel_nonneg_scale_branches(self):
        m = Model("ns3")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        sq = x**2
        # Right-hand constant factor is peeled.
        scale, core = pat._peel_nonneg_scale(BinaryOp("*", sq, Constant(np.array(3.0))))
        assert scale == pytest.approx(3.0)
        assert core is sq
        # Negative constants (either side) abstain.
        assert pat._peel_nonneg_scale(BinaryOp("*", Constant(np.array(-2.0)), sq))[0] is None
        assert pat._peel_nonneg_scale(BinaryOp("*", sq, Constant(np.array(-3.0))))[0] is None


class TestPatternHelpers:
    def test_offsets_and_not_found(self):
        m = Model("h")
        m.continuous("x", lb=0.5, ub=2.0)
        v = m.continuous("v", shape=(3,), lb=0.0, ub=2.0)
        other = Model("o").continuous("z", lb=0.0, ub=1.0)
        assert pat._scalar_var_offset(m, v) is None  # non-scalar
        assert pat._scalar_var_offset(m, other) is None  # not in model
        assert pat._var_offset(m, v) == 1  # after the scalar x
        assert pat._var_offset(m, other) is None

    def test_same_expr_and_contains_var(self):
        m = Model("h2")
        x = m.continuous("x", lb=0.5, ub=2.0)
        v = m.continuous("v", shape=(3,), lb=0.0, ub=2.0)
        assert pat._same_expr(v[0], v[0])
        assert not pat._same_expr(v[0], v[1])
        assert pat._contains_var(dm.sum(v), v)
        sov = dm.sum(v[i] for i in range(2))
        assert pat._contains_var(sov, v)
        assert not pat._contains_var(sov, x)

    def test_scale_power_and_linear_factor(self):
        m = Model("h3")
        x = m.continuous("x", lb=0.5, ub=2.0)
        v = m.continuous("v", shape=(3,), lb=0.0, ub=2.0)
        scaled = pat._scale_expr(x, 2.5)
        assert isinstance(scaled, BinaryOp) and scaled.op == "*"
        assert float(scaled.left.value) == pytest.approx(2.5)
        # base ** nonconstant abstains.
        assert pat._extract_power_factor(x**x) is None
        # Indexed occurrence of the target variable has unit coefficient;
        # a different target abstains.
        got = pat._extract_linear_factor(v[0], v)
        assert isinstance(got, Constant) and float(got.value) == pytest.approx(1.0)
        assert pat._extract_linear_factor(v[0], x) is None

    def test_affine_range_infinite_bounds(self):
        assert pat._affine_range_1d(1.0, 2.0, -np.inf, 3.0) == (-np.inf, 5.0)
        assert pat._affine_range_1d(0.0, 2.0, -np.inf, np.inf) == (2.0, 2.0)
        assert pat._affine_range_1d(-1.0, 0.0, -np.inf, np.inf) == (-np.inf, np.inf)

    def test_expr_struct_eq_branches(self):
        m = Model("h4")
        x = m.continuous("x", lb=0.5, ub=2.0)
        x_again = m._variables[0]
        assert pat._expr_struct_eq(x, x_again)
        assert pat._expr_struct_eq(UnaryOp("neg", x), UnaryOp("neg", x))
        assert not pat._expr_struct_eq(UnaryOp("neg", x), UnaryOp("abs", x))
        pa = Parameter("pp", 1.0, m)
        pb = Parameter("pp", 2.0, Model("o2"))
        assert pat._expr_struct_eq(pa, pa)
        assert pat._expr_struct_eq(pa, pb)  # same-name parameters compare equal
        # Distinct Variable objects compare by name; mixed types fall through.
        x_other = Model("m2").continuous("x", lb=0.0, ub=1.0)
        q_other = Model("m3").continuous("q", lb=0.0, ub=1.0)
        assert pat._expr_struct_eq(x, x_other)
        assert not pat._expr_struct_eq(x, q_other)
        assert not pat._expr_struct_eq(x, Constant(np.array(1.0)))

    def test_sign_domain_checks(self):
        m = Model("h5")
        x = m.continuous("x", lb=0.5, ub=2.0)
        v = m.continuous("v", shape=(3,), lb=0.0, ub=2.0)
        # Out-of-range index must be conservative, not raise.
        bad = IndexExpression(v, 10)
        assert not pat._has_positive_lower_bound(bad, m)
        assert not pat._is_nonneg_domain(bad, m)
        assert pat._is_nonneg_domain(v[0], m)
        # Constant multipliers on either side.
        assert pat._is_nonneg_domain(BinaryOp("*", v[0], Constant(np.array(2.0))), m)
        assert pat._has_positive_lower_bound(BinaryOp("*", Constant(np.array(2.0)), x), m)
        assert pat._has_positive_lower_bound(BinaryOp("*", x, Constant(np.array(2.0))), m)
        # A product of two variables has no constant factor to reason from.
        y = m.continuous("y", lb=0.5, ub=2.0)
        assert not pat._is_nonneg_domain(BinaryOp("*", x, y), m)

    def test_declared_box_cache_invalidation(self):
        m = Model("cache")
        u = m.continuous("u", lb=0.0, ub=1.0)
        lo1, _ = pat._box_bounds(m)
        assert lo1[0] == pytest.approx(0.0)
        u.lb = np.array([0.5])
        lo_stale, _ = pat._box_bounds(m)  # memoized: still the stale value
        assert lo_stale[0] == pytest.approx(0.0)
        pat.clear_declared_box_cache(m)
        lo2, _ = pat._box_bounds(m)
        assert lo2[0] == pytest.approx(0.5)

    def test_clear_cache_propagates_delattr_failure(self):
        """#1520: a cache that cannot be cleared stays stale, and the recognizers
        would then classify against an older, narrower box -- a possible false
        convexity proof. The failure propagates instead of being swallowed."""

        class Weird:
            def __delattr__(self, name):
                raise RuntimeError("nope")

        w = Weird()
        w.__dict__[pat._DECLARED_BOX_CACHE_ATTR] = (1, None, None)
        with pytest.raises(RuntimeError, match="nope"):
            pat.clear_declared_box_cache(w)

    def test_box_bounds_empty_model(self):
        lo, hi = pat._box_bounds(Model("empty"))
        assert lo.size == 0 and hi.size == 0

    def test_box_bounds_broadcasts_scalar_bounds(self):
        m = Model("bb")
        w = m.continuous("w", shape=(2,), lb=0.0, ub=1.0)
        w.lb = np.array(0.25)  # scalar-stored declared bound
        w.ub = np.array(0.75)
        lo, hi = pat._box_bounds(m)
        np.testing.assert_allclose(lo, [0.25, 0.25])
        np.testing.assert_allclose(hi, [0.75, 0.75])


class TestFractionalEpigraph:
    """Quadratic-over-affine epigraph recognition (nlp_cvx_108 family)."""

    @staticmethod
    def _make(mkbody):
        m = Model("epi")
        x = m.continuous("x", lb=0.0, ub=1.0)
        y = m.continuous("y", lb=-10.0, ub=10.0)
        m.subject_to(mkbody(x, y) <= 0.0)
        return m, m._constraints[-1]

    def test_negative_denominator_convex_true(self):
        # -(x+2)*y + x^2 + 1 <= 0  <=>  y >= (x^2+1)/(x+2), epigraph of a
        # convex quadratic-over-affine: recognized as convex.
        m, c = self._make(lambda x, y: -1.0 * (x + 2.0) * y + x * x + 1.0)
        assert pat.classify_fractional_epigraph_constraint(c, m) is True
        # Evidence: g(x) = (x^2+1)/(x+2) is midpoint-convex on the box.
        _assert_sampled_curvature(
            lambda p: (p[0] ** 2 + 1.0) / (p[0] + 2.0), [0.0], [1.0], Curvature.CONVEX
        )

    def test_negative_denominator_nonconvex_false(self):
        # -(x+2)*y - x^2 <= 0  <=>  y >= -x^2/(x+2): epigraph of a strictly
        # concave function, a nonconvex set.
        m, c = self._make(lambda x, y: -1.0 * (x + 2.0) * y - x * x)
        g = lambda t: -(t**2) / (t + 2.0)  # noqa: E731
        # Evidence: the chord midpoint of two feasible points is infeasible.
        assert 0.5 * (g(0.0) + g(1.0)) < g(0.5) - 1e-3
        assert pat.classify_fractional_epigraph_constraint(c, m) is False

    def test_positive_denominator_convex_true(self):
        # (x+2)*y + x^2 + 1 <= 0  <=>  y <= -(x^2+1)/(x+2), the hypograph of a
        # concave function: a convex set that should be recognized.
        m, c = self._make(lambda x, y: (x + 2.0) * y + x * x + 1.0)
        h = lambda t: -(t**2 + 1.0) / (t + 2.0)  # noqa: E731
        # Evidence: h is midpoint-concave, so {y <= h(x)} is convex.
        _assert_sampled_curvature(h, [0.0], [1.0], Curvature.CONCAVE)
        assert pat.classify_fractional_epigraph_constraint(c, m) is True

    def test_positive_denominator_nonconvex_must_not_be_true(self):
        # (x+2)*y - x^2 <= 0  <=>  y <= x^2/(x+2): upper-bounding by a strictly
        # convex function is a NONCONVEX set — classifying it convex is unsound.
        m, c = self._make(lambda x, y: (x + 2.0) * y - x * x)
        g = lambda t: t**2 / (t + 2.0)  # noqa: E731
        # Evidence: (0, g(0)) and (1, g(1)) are feasible but their midpoint is not.
        assert 0.5 * (g(0.0) + g(1.0)) > g(0.5) + 1e-3
        assert pat.classify_fractional_epigraph_constraint(c, m) is not True


# ──────────────────────────────────────────────────────────────────────
# differentiable: parametric compiler + solve plumbing
# ──────────────────────────────────────────────────────────────────────


class TestParametricCompilerOps:
    @pytest.fixture()
    def setup(self):
        m = Model("pc")
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=-5.0, ub=5.0)
        v = m.continuous("v", shape=(3,), lb=-5.0, ub=5.0)
        x_flat = np.array([1.5, 0.5, -1.0, 2.0])
        p_flat = np.array([2.0])
        return m, p, x, v, x_flat, p_flat

    def _check(self, m, expr, f_np, x_flat, p_flat, smooth=True):
        fn = _compile_parametric_node(expr, m)
        got = float(np.asarray(fn(jnp.array(x_flat), jnp.array(p_flat))).ravel()[0])
        assert got == pytest.approx(float(f_np(x_flat, p_flat)), abs=1e-10)
        if smooth:
            scalar = lambda xa, pa: jnp.reshape(fn(xa, pa), ())  # noqa: E731
            gx = np.asarray(jax.grad(scalar, argnums=0)(jnp.array(x_flat), jnp.array(p_flat)))
            gp = np.asarray(jax.grad(scalar, argnums=1)(jnp.array(x_flat), jnp.array(p_flat)))
            np.testing.assert_allclose(gx, _fd_grad(lambda xa: f_np(xa, p_flat), x_flat), atol=1e-5)
            np.testing.assert_allclose(gp, _fd_grad(lambda pa: f_np(x_flat, pa), p_flat), atol=1e-5)

    def test_division(self, setup):
        m, p, x, v, xf, pf = setup
        self._check(m, x / p, lambda xa, pa: xa[0] / pa[0], xf, pf)

    def test_abs(self, setup):
        m, p, x, v, xf, pf = setup
        # x=1.5, p=2: away from the kink, gradients are well-defined. Both the
        # FunctionCall("abs") and UnaryOp("abs") encodings must agree.
        self._check(m, dm.abs(x - p), lambda xa, pa: abs(xa[0] - pa[0]), xf, pf)
        self._check(m, abs(x - p), lambda xa, pa: abs(xa[0] - pa[0]), xf, pf)

    def test_min_max(self, setup):
        m, p, x, v, xf, pf = setup
        self._check(m, dm.minimum(x, p), lambda xa, pa: min(xa[0], pa[0]), xf, pf)
        self._check(m, dm.maximum(x, p), lambda xa, pa: max(xa[0], pa[0]), xf, pf)

    def test_prod(self, setup):
        m, p, x, v, xf, pf = setup
        self._check(m, dm.prod(v), lambda xa, pa: np.prod(xa[1:4]), xf, pf)

    def test_norm2(self, setup):
        m, p, x, v, xf, pf = setup
        self._check(m, dm.norm(v), lambda xa, pa: np.linalg.norm(xa[1:4]), xf, pf)

    def test_matmul(self, setup):
        m, p, x, v, xf, pf = setup
        row = np.array([[1.0, 2.0, 3.0]])
        expr = MatMulExpression(Constant(row), v)
        self._check(m, expr, lambda xa, pa: row[0] @ xa[1:4], xf, pf)

    def test_sum_over(self, setup):
        m, p, x, v, xf, pf = setup
        expr = dm.sum(v[i] * p for i in range(3))
        self._check(m, expr, lambda xa, pa: np.sum(xa[1:4]) * pa[0], xf, pf)

    def test_error_paths(self, setup):
        m, p, x, v, xf, pf = setup
        with pytest.raises(ValueError, match="binary"):
            _compile_parametric_node(BinaryOp("%", x, x), m)
        with pytest.raises(ValueError, match="unary"):
            _compile_parametric_node(UnaryOp("bogus", x), m)
        with pytest.raises(ValueError, match="function"):
            _compile_parametric_node(FunctionCall("bogusfn", x), m)
        with pytest.raises(TypeError, match="Unhandled"):
            _compile_parametric_node("junk", m)
        other = Model("other")
        q = other.parameter("q", value=1.0)
        with pytest.raises(ValueError, match="not found"):
            _compute_param_offset(q, m)
        with pytest.raises(ValueError, match="objective"):
            _compile_parametric_objective(Model("noobj"))
        with pytest.raises(ValueError, match="nlp_solver"):
            _dispatch_nlp_solve("bogus", None, None, {})


class TestDifferentiableSolvePaths:
    def _infeasible_model(self):
        m = Model("inf")
        p = m.parameter("p", value=1.0)
        z = m.continuous("z", lb=0.0, ub=10.0)
        m.subject_to(z >= 5.0)
        m.subject_to(z <= 1.0)
        m.minimize((z - p) ** 2)
        return m, p

    def test_ipopt_backend_active_constraint_gradient(self):
        # min x s.t. x >= p: obj*(p) = p, so d(obj*)/dp = 1 exactly.
        m = Model("ip")
        p = m.parameter("p", value=1.5)
        x = m.continuous("x", lb=0.0, ub=10.0)
        m.subject_to(x >= p)
        m.minimize(x)
        r = differentiable_solve(m, nlp_solver="ipopt")
        assert r.objective == pytest.approx(1.5, abs=1e-6)
        assert r.gradient(p) == pytest.approx(1.0, abs=1e-6)

    def test_nonconvergence_raises(self):
        m, _ = self._infeasible_model()
        with pytest.raises(RuntimeError, match="did not converge"):
            differentiable_solve(m)

    def test_sensitivity_resolve_failure_raises_after_fallback(self):
        m, _ = self._infeasible_model()
        with pytest.raises(RuntimeError, match="did not converge"):
            _compute_sensitivity_at_solution(m, {"z": np.array(3.0)}, nlp_solver="ipm")

    def test_integer_infeasible_returns_zero_sensitivity(self):
        m = Model("bi")
        p = m.parameter("p", value=1.0)
        b = m.binary("b")
        m.subject_to(b >= 0.6)
        m.subject_to(b <= 0.4)
        m.minimize(b * p)
        r = _differentiable_solve_integer(m)
        assert r.status != "optimal"
        np.testing.assert_array_equal(r._sensitivity, np.zeros(1))

    def test_jax_native_constrained_solve_grad(self):
        # min x s.t. x >= p: solve_fn(p) = p and d/dp = 1 via the envelope JVP.
        m = Model("jx")
        m.parameter("p", value=1.0)
        x = m.continuous("x", lb=-5.0, ub=5.0)
        m.subject_to(x >= m._parameters[0])
        m.minimize(x)
        solve_fn = _make_jax_differentiable_solve(m)
        assert float(solve_fn(jnp.array([1.0]))) == pytest.approx(1.0, abs=1e-6)
        g = jax.grad(lambda q: solve_fn(q))(jnp.array([1.0]))
        assert float(np.asarray(g)[0]) == pytest.approx(1.0, abs=1e-6)
        # Cross-check the gradient against a finite difference of the solve.
        fd = (float(solve_fn(jnp.array([1.01]))) - float(solve_fn(jnp.array([0.99])))) / 0.02
        assert float(np.asarray(g)[0]) == pytest.approx(fd, abs=1e-5)

    def test_jax_native_rejects_integer_models(self):
        m = Model("ji")
        m.parameter("p", value=1.0)
        b = m.binary("b")
        m.minimize(b)
        with pytest.raises(ValueError, match="continuous"):
            _make_jax_differentiable_solve(m)
        with pytest.raises(ValueError, match="continuous"):
            differentiable_solve_l3(m)

    def test_l3_nonconvergence_raises(self):
        m, _ = self._infeasible_model()
        with pytest.raises(RuntimeError, match="did not converge"):
            differentiable_solve_l3(m)


class TestFindActiveSet:
    def test_ge_constraint_and_bound_activity(self):
        m = Model("fa")
        m.parameter("p", value=1.0)
        x = m.continuous("x", lb=0.0, ub=2.0)
        # subject_to normalizes >= into <=; build a genuine ">=" row directly.
        m._constraints.append(Constraint(body=x - 1.0, sense=">="))
        cfs = [_compile_parametric_constraint(c, m) for c in m._constraints]
        p_flat = jnp.array([1.0])
        # At x = 1: the >= constraint is tight, no bound is active.
        cons, bounds = find_active_set(jnp.array([1.0]), m, cfs, p_flat)
        assert cons == [0] and bounds == []
        # At x = 2 (the upper bound): constraint slack, ub active.
        cons, bounds = find_active_set(jnp.array([2.0]), m, cfs, p_flat)
        assert cons == [] and bounds == [0]
        # At x = 0 (the lower bound): lb active.
        cons, bounds = find_active_set(jnp.array([0.0]), m, cfs, p_flat)
        assert bounds == [0]


class TestDiffSolveResults:
    def test_value_and_repr_without_solution(self):
        m = Model("r")
        m.parameter("p", value=1.0)
        x = m.continuous("x", lb=0.0, ub=1.0)
        r = DiffSolveResult(
            status="infeasible", objective=None, x=None, _model=m, _sensitivity=np.zeros(1)
        )
        with pytest.raises(ValueError, match="No solution"):
            r.value(x)
        assert "infeasible" in repr(r)

    def test_l3_fallback_accessors(self):
        m = Model("l3f")
        p = m.parameter("p", value=1.0)
        m.continuous("x", lb=0.0, ub=1.0)
        r = DiffSolveResultL3(
            status="optimal",
            objective=1.0,
            x={"x": np.array(0.5)},
            _model=m,
            _sensitivity=np.array([2.5]),
            _dx_dp=None,
            _obj_fn_parametric=None,
            _x_star=None,
            _p_flat=None,
            _l3_failed=True,
        )
        assert r.sensitivity_matrix() is None
        # implicit_gradient falls back to the stored L1 envelope value.
        assert r.implicit_gradient(p) == pytest.approx(2.5)
        assert r.dual_sensitivity(p) is None
        assert r.reduced_hessian() is None
        with pytest.raises(RuntimeError, match="approximate_resolve"):
            r.approximate_resolve([(p, 2.0)])
        with pytest.raises(RuntimeError, match="KKT"):
            r.sensitivity(np.array([1.0]))
        assert "fallback_to_L1" in repr(r)
        # With dx/dp available but no primal solution, approximate_resolve
        # must still refuse.
        r2 = DiffSolveResultL3(
            status="optimal",
            objective=1.0,
            x={"x": np.array(0.5)},
            _model=m,
            _sensitivity=np.array([2.5]),
            _dx_dp=np.zeros((1, 1)),
            _obj_fn_parametric=None,
            _x_star=None,
            _p_flat=jnp.array([1.0]),
            _l3_failed=False,
        )
        with pytest.raises(RuntimeError, match="no solution"):
            r2.approximate_resolve([(p, 2.0)])

    def test_l3_vector_param_implicit_gradient(self):
        # min (a-q0)^2 + (b-q1)^2 with a clipped at ub=1 and q0=2:
        # obj*(q) = (1-q0)^2, so dobj/dq = [2(q0-1), 0] = [2, 0].
        m = Model("l3v")
        q = m.parameter("q", value=np.array([2.0, 0.5]))
        a = m.continuous("a", lb=-1.0, ub=1.0)
        b = m.continuous("b", lb=-5.0, ub=5.0)
        m.minimize((a - q[0]) ** 2 + (b - q[1]) ** 2)
        r = differentiable_solve_l3(m)
        assert not r._l3_failed
        grad = r.implicit_gradient(q)
        assert grad.shape == (2,)
        np.testing.assert_allclose(grad, [2.0, 0.0], atol=1e-6)

    def test_l3_active_constraint_sensitivity_and_reduced_hessian(self):
        # min z s.t. z >= p: z* = p, dz/dp = 1; one active constraint on one
        # variable leaves an empty null space (0x0 reduced Hessian).
        m = Model("l3a")
        p = m.parameter("p", value=1.0)
        z = m.continuous("z", lb=-5.0, ub=5.0)
        m.subject_to(z >= p)
        m.minimize(z)
        r = differentiable_solve_l3(m)
        assert not r._l3_failed
        assert r._n_active == 1
        assert r.gradient(p) == pytest.approx(1.0, abs=1e-6)
        rh = r.reduced_hessian()
        assert rh is not None and rh.shape == (0, 0)
        # KKT back-substitution: dz = dp (1-D and batched forms).
        np.testing.assert_allclose(r.sensitivity(np.array([1.0])), [1.0], atol=1e-8)
        np.testing.assert_allclose(r.sensitivity(np.array([[1.0, 2.0]])), [[1.0, 2.0]], atol=1e-8)
