"""Coverage-driven unit/property tests (#87, final round) for three modules.

* ``discopt._jax.relaxation_compiler`` — envelope containment (``cv <= f <= cc``
  on sampled boxes) for the atom extractors that the existing envelope suite
  does not reach (signed powers, general ``x**y``, partitioned paths, prod,
  norms, sign, indexed tight dispatches, alphaBB arithmetic, the learned-mode
  bilinear dispatch), plus REJECT branches of the ``_try_extract_*`` helpers
  exercised with near-miss shapes.
* ``discopt._jax.convexity.patterns`` — every CONVEX/CONCAVE verdict is
  cross-checked against sampled midpoint convexity of the actual function on
  the declared box; near-miss shapes must abstain (``None``).
* ``discopt._jax.differentiable`` — parametric-compiler values and gradients
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
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.convexity import Curvature, classify_expr
from discopt._jax.convexity import patterns as pat
from discopt._jax.differentiable import (
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
from discopt._jax.relaxation_compiler import (
    _compile_relax_node,
    _resolve_scalar_var_offset,
    _try_extract_arrhenius,
    _try_extract_monod,
    _try_extract_signed_abs_product,
    _try_extract_signed_power,
    _try_extract_signomial_factors,
    _try_extract_trilinear_chain,
    _try_extract_xlogx,
    compile_objective_relaxation,
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


def _assert_envelope(model, f_np, lb, ub, n=6, tol=1e-7, **compile_kw):
    """Compiled relaxation must satisfy cv <= f <= cc on a sampled grid."""
    fn = compile_objective_relaxation(model, **compile_kw)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    for pt in _grid(lb, ub, n):
        cv, cc = fn(pt, pt, lb, ub)
        f = float(f_np(pt))
        assert float(cv) <= f + tol, f"cv {float(cv)} > f {f} at {pt}"
        assert float(cc) >= f - tol, f"cc {float(cc)} < f {f} at {pt}"
    return fn


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
# relaxation_compiler: envelope containment for uncovered atoms
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.relaxation
class TestRelaxationCompilerEnvelopes:
    def test_signed_power_panhandle_envelope(self):
        # f * |f|**(beta-1) with beta = 2.5 (Panhandle generalization of Weymouth).
        m = Model("sp")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        # Builtin abs() maps to UnaryOp("abs"), the shape the extractor matches.
        m.minimize(x * abs(x) ** 1.5)
        _assert_envelope(m, lambda p: p[0] * abs(p[0]) ** 1.5, [-2.0], [2.0], n=13)

    def test_general_pow_envelope(self):
        # x**y with variable exponent routes through exp(y*log(x)).
        m = Model("pw")
        x = m.continuous("x", lb=0.5, ub=2.0)
        y = m.continuous("y", lb=1.0, ub=2.0)
        m.minimize(x**y)
        _assert_envelope(m, lambda p: p[0] ** p[1], [0.5, 1.0], [2.0, 2.0], n=7)

    def test_general_pow_envelope_partitioned(self):
        m = Model("pwp")
        x = m.continuous("x", lb=0.5, ub=2.0)
        y = m.continuous("y", lb=1.0, ub=2.0)
        m.minimize(x**y)
        _assert_envelope(m, lambda p: p[0] ** p[1], [0.5, 1.0], [2.0, 2.0], n=7, partitions=2)

    def test_partitioned_nonsignomial_bilinear_envelope(self):
        # (x+y)*y is a product of non-leaf factors: piecewise-McCormick bilinear.
        m = Model("pb")
        x = m.continuous("x", lb=0.0, ub=2.0)
        y = m.continuous("y", lb=0.5, ub=2.0)
        m.minimize((x + y) * y)
        _assert_envelope(
            m, lambda p: (p[0] + p[1]) * p[1], [0.0, 0.5], [2.0, 2.0], n=7, partitions=3
        )

    def test_partitioned_univariate_envelopes(self):
        for build, f_np, lo, hi in [
            (dm.exp, np.exp, -1.0, 1.5),
            (dm.log, np.log, 0.5, 3.0),
            (dm.sqrt, np.sqrt, 0.1, 4.0),
        ]:
            m = Model("pu")
            x = m.continuous("x", lb=lo, ub=hi)
            m.minimize(build(x))
            _assert_envelope(m, lambda p, f=f_np: f(p[0]), [lo], [hi], n=11, partitions=3)

    def test_prod_multilinear_envelope(self):
        # prod over a length-3 vector on a sign-mixed box (recursive McCormick fold).
        m = Model("pr")
        v = m.continuous("v", shape=(3,), lb=-1.0, ub=2.0)
        m.minimize(dm.prod(v))
        _assert_envelope(m, np.prod, [-1.0] * 3, [2.0] * 3, n=5)

    def test_prod_singleton_is_identity(self):
        m = Model("pr1")
        u = m.continuous("u", shape=(1,), lb=0.5, ub=2.0)
        m.minimize(dm.prod(u))
        fn = compile_objective_relaxation(m)
        pt = np.array([1.3])
        cv, cc = fn(pt, pt, np.array([0.5]), np.array([2.0]))
        assert float(cv) == pytest.approx(1.3)
        assert float(cc) == pytest.approx(1.3)

    def test_norm2_envelope_is_inf_to_one_norm_sandwich(self):
        # cv = max_i |x_i| and cc = sum_i |x_i| at a point evaluation.
        m = Model("n2")
        w = m.continuous("w", shape=(2,), lb=-1.0, ub=2.0)
        m.minimize(dm.norm(w))
        fn = _assert_envelope(m, np.linalg.norm, [-1.0] * 2, [2.0] * 2, n=7)
        pt = np.array([-0.8, 1.5])
        cv, cc = fn(pt, pt, np.full(2, -1.0), np.full(2, 2.0))
        assert float(cv) == pytest.approx(1.5)
        assert float(cc) == pytest.approx(2.3)

    def test_sign_envelope(self):
        m = Model("sg")
        s = m.continuous("s", lb=-1.0, ub=2.0)
        m.minimize(dm.sign(s))
        _assert_envelope(m, lambda p: np.sign(p[0]), [-1.0], [2.0], n=11)

    def test_asinh_indexed_envelope(self):
        # IndexExpression argument takes the flat-index tight-envelope dispatch.
        m = Model("as1")
        v = m.continuous("v", shape=(2,), lb=-1.0, ub=2.0)
        m.minimize(dm.asinh(v[1]))
        _assert_envelope(m, lambda p: np.arcsinh(p[1]), [-1.0] * 2, [2.0] * 2, n=7)

    def test_asinh_composite_fallback_envelope(self):
        # Non-variable argument falls back to envelopes over propagated bounds.
        m = Model("as2")
        a = m.continuous("a", lb=-1.0, ub=1.0)
        b = m.continuous("b", lb=-1.0, ub=1.0)
        m.minimize(dm.asinh(a + b))
        _assert_envelope(m, lambda p: np.arcsinh(p[0] + p[1]), [-1.0] * 2, [1.0] * 2, n=7, tol=1e-6)

    def test_indexed_power_sin_cos_envelope(self):
        # v[1]**3, sin(v[0]), cos(v[1]) all take the flat-index tight dispatches.
        m = Model("ix")
        v = m.continuous("v", shape=(2,), lb=-1.5, ub=2.0)
        m.minimize(v[1] ** 3 + dm.sin(v[0]) + dm.cos(v[1]))
        _assert_envelope(
            m,
            lambda p: p[1] ** 3 + np.sin(p[0]) + np.cos(p[1]),
            [-1.5] * 2,
            [2.0] * 2,
            n=7,
        )

    def test_trilinear_exact_flag_envelope(self, monkeypatch):
        # DISCOPT_TRILINEAR=exact selects the best-of-three nested envelope.
        monkeypatch.setenv("DISCOPT_TRILINEAR", "exact")
        m = Model("tri")
        xs = [m.continuous(f"t{k}", lb=-1.0, ub=1.5) for k in range(3)]
        m.minimize(xs[0] * xs[1] * xs[2])
        _assert_envelope(m, np.prod, [-1.0] * 3, [1.5] * 3, n=5)

    def test_alphabb_arithmetic_envelope(self):
        m = Model("ab")
        x = m.continuous("x", lb=-1.0, ub=2.0)
        m.minimize(x**2 + dm.sin(x))
        _assert_envelope(
            m, lambda p: p[0] ** 2 + np.sin(p[0]), [-1.0], [2.0], n=11, arithmetic="alphabb"
        )

    def test_learned_bilinear_dispatch_uses_registry(self):
        # A learned-mode bilinear must be routed through the registry model with
        # the compositional midpoint as true value: fake model returns
        # (true - 1, true + 1), so the output pins the dispatch wiring exactly.
        class FakeRegistry:
            def get(self, name):
                if name == "bilinear":
                    return lambda xy, xy_lb, xy_ub, true_val: (true_val - 1.0, true_val + 1.0)
                return None

        m = Model("lr")
        x = m.continuous("x", lb=0.0, ub=2.0)
        y = m.continuous("y", lb=0.0, ub=2.0)
        m.minimize((x + y) * (y - x))  # non-signomial => generic bilinear path
        fn = compile_objective_relaxation(m, mode="learned", learned_registry=FakeRegistry())
        pt = np.array([0.5, 1.5])
        cv, cc = fn(pt, pt, np.zeros(2), np.full(2, 2.0))
        true = (0.5 + 1.5) * (1.5 - 0.5)
        assert float(cv) == pytest.approx(true - 1.0)
        assert float(cc) == pytest.approx(true + 1.0)


class TestRelaxationCompilerExtractors:
    @pytest.fixture()
    def model(self):
        m = Model("ex")
        m.continuous("x", lb=0.5, ub=2.0)
        m.continuous("y", lb=0.5, ub=2.0)
        m.continuous("z", lb=0.5, ub=2.0)
        m.continuous("v", shape=(2,), lb=0.5, ub=2.0)
        return m

    def _vars(self, m):
        return m._variables[0], m._variables[1], m._variables[2], m._variables[3]

    def test_trilinear_chain_matches_and_rejects(self, model):
        x, y, z, _ = self._vars(model)
        # Right-associative parsing z*(x*y) resolves all three offsets.
        assert _try_extract_trilinear_chain(z * (x * y), model) == (0, 1, 2)
        # Not a product at all.
        assert _try_extract_trilinear_chain(x + y, model) is None
        # Non-variable leaf inside the inner product.
        assert _try_extract_trilinear_chain((x * (y + 1.0)) * z, model) is None

    def test_signed_product_rejects(self, model):
        x, y, _, _ = self._vars(model)
        assert _try_extract_signed_abs_product(x + y, model) is None
        assert _try_extract_signed_power(x + y, model) is None
        # Non-positive exponent p <= 0 must be rejected.
        assert _try_extract_signed_power(x * abs(x) ** (-1.0), model) is None
        # The positive-exponent UnaryOp("abs") shape does match.
        assert _try_extract_signed_power(x * abs(x) ** 1.5, model) == (0, 2.5)

    def test_xlogx_monod_arrhenius_rejects(self, model):
        x, y, _, _ = self._vars(model)
        assert _try_extract_xlogx(x + y, model) is None
        assert _try_extract_monod(x * y, model) is None  # not a division
        assert _try_extract_arrhenius(dm.log(x), model) is None  # not exp
        # exp(-c / nonvariable) has no scalar offset.
        assert _try_extract_arrhenius(dm.exp(-1.0 / (x * y)), model) is None

    def test_signomial_indexed_factors(self, model):
        _, _, _, v = self._vars(model)
        # Indexed powers, bare indexed factors, and tuple indices all resolve.
        assert _try_extract_signomial_factors(v[0] ** 2.0 * v[1] ** 3.0, model) == [
            (3, 2.0),
            (3 + 1, 3.0),
        ]
        assert _try_extract_signomial_factors(v[0] * v[1], model) == [(3, 1.0), (4, 1.0)]
        assert _try_extract_signomial_factors(v[(0,)] ** 2.0 * v[(1,)], model) == [
            (3, 2.0),
            (4, 1.0),
        ]
        # A multi-component tuple index cannot resolve to one flat slot.
        bad_pow = BinaryOp("**", IndexExpression(v, (0, 1)), Constant(np.array(2.0)))
        assert _try_extract_signomial_factors(BinaryOp("*", bad_pow, v[1]), model) is None

    def test_resolve_scalar_var_offset_tuple_index(self, model):
        x, y, _, v = self._vars(model)
        assert _resolve_scalar_var_offset(v[(1,)], model) == 4
        assert _resolve_scalar_var_offset(x + y, model) is None

    def test_compile_node_raises_on_unknown_shapes(self, model):
        x, _, _, _ = self._vars(model)
        with pytest.raises(ValueError, match="unary"):
            _compile_relax_node(UnaryOp("bogus", x), model)
        with pytest.raises(ValueError, match="binary"):
            _compile_relax_node(BinaryOp("%", x, x), model)
        with pytest.raises(TypeError, match="Unhandled"):
            _compile_relax_node("junk", model)
        with pytest.raises(ValueError, match="Unknown function"):
            _compile_relax_node(FunctionCall("bogusfn", x), model)
        # prod over a non-array argument has no fixed size to fold over.
        y = model._variables[1]
        with pytest.raises(ValueError, match="fixed-size array"):
            _compile_relax_node(FunctionCall("prod", x + y), model)


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

    def test_clear_cache_swallows_delattr_failure(self):
        class Weird:
            def __delattr__(self, name):
                raise RuntimeError("nope")

        w = Weird()
        w.__dict__[pat._DECLARED_BOX_CACHE_ATTR] = (1, None, None)
        pat.clear_declared_box_cache(w)  # must not raise
        assert hasattr(w, pat._DECLARED_BOX_CACHE_ATTR)

    def test_box_bounds_empty_model(self):
        lo, hi = pat._box_bounds(Model("empty"))
        assert lo.size == 0 and hi.size == 0

    @pytest.mark.xfail(
        reason="#757: _box_bounds scalar-bound broadcast branch calls float() on a "
        "1-element array, which raises TypeError under NumPy 2 instead of "
        "broadcasting (patterns.py lines 268-271)",
        strict=False,
    )
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

    @pytest.mark.xfail(
        reason="#757: coeff_lo>0 branch of classify_fractional_epigraph_constraint "
        "has an inverted discriminant test (patterns.py line 1091): the "
        "genuinely convex hypograph y <= -(x^2+1)/(x+2) is reported False",
        strict=False,
    )
    def test_positive_denominator_convex_true(self):
        # (x+2)*y + x^2 + 1 <= 0  <=>  y <= -(x^2+1)/(x+2), the hypograph of a
        # concave function: a convex set that should be recognized.
        m, c = self._make(lambda x, y: (x + 2.0) * y + x * x + 1.0)
        h = lambda t: -(t**2 + 1.0) / (t + 2.0)  # noqa: E731
        # Evidence: h is midpoint-concave, so {y <= h(x)} is convex.
        _assert_sampled_curvature(h, [0.0], [1.0], Curvature.CONCAVE)
        assert pat.classify_fractional_epigraph_constraint(c, m) is True

    @pytest.mark.xfail(
        reason="#757: SOUNDNESS: coeff_lo>0 branch of "
        "classify_fractional_epigraph_constraint (patterns.py line 1091) "
        "returns True for the NONCONVEX set y <= x^2/(x+2); the discriminant "
        "inequality is inverted relative to the coeff_hi<0 branch",
        strict=False,
    )
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
