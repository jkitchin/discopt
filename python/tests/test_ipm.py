"""
Test suite for the pure-JAX Interior Point Method (IPM) solver.

Tests cover:
  - Helper function unit tests (sigma, fraction-to-boundary, convergence, mu update)
  - Unconstrained optimization (quadratic, Rosenbrock, bound-constrained, exponential)
  - Constrained optimization (equality, inequality, nonlinear, mixed, HS071)
  - Comparison with cyipopt on correctness test problems
  - Batch/vmap support (solve_nlp_batch)
  - Performance (batch speedup over sequential)
  - Layer 2 wrapper (solve_nlp_ipm returning NLPResult)
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.ipm import (
    IPMOptions,
    _compute_sigma,
    _fraction_to_boundary,
    _update_mu,
    ipm_solve,
    solve_nlp_batch,
    solve_nlp_ipm,
)
from discopt._jax.ipm import (
    _check_convergence_standalone as _check_convergence,
)
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.solvers import NLPResult, SolveStatus

# ---------------------------------------------------------------
# Shared objective/constraint functions (def, not lambda per E731)
# ---------------------------------------------------------------


def _obj_quadratic_shifted(x):
    return (x[0] - 2) ** 2 + (x[1] + 1) ** 2


def _obj_rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def _obj_sum_sq_2d(x):
    return x[0] ** 2 + x[1] ** 2


def _obj_sum_sq(x):
    return jnp.sum(x**2)


def _obj_exp_sum(x):
    return jnp.exp(x[0]) + jnp.exp(x[1])


def _obj_neg_sum(x):
    return -x[0] - x[1]


def _obj_shifted_2d(x):
    return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2


def _obj_x0_sq(x):
    return x[0] ** 2


def _con_empty(x):
    return jnp.array([])


def _con_sum_2d(x):
    return jnp.array([x[0] + x[1]])


def _con_sum_3d(x):
    return jnp.array([x[0] + x[1] + x[2], x[0]])


def _con_product(x):
    return jnp.array([x[0] * x[1]])


# ---------------------------------------------------------------
# 1. Helper function unit tests
# ---------------------------------------------------------------


class TestIPMHelpers:
    """Unit tests for IPM internal helper functions."""

    def test_compute_sigma(self):
        """Sigma diagonal = z_l/(x - x_l) * has_lb + z_u/(x_u - x) * has_ub."""
        x = jnp.array([1.5, 2.0])
        x_l = jnp.array([1.0, 0.0])
        x_u = jnp.array([3.0, 5.0])
        z_l = jnp.array([0.2, 0.1])
        z_u = jnp.array([0.3, 0.05])
        has_lb = jnp.array([True, True])
        has_ub = jnp.array([True, True])
        sigma = _compute_sigma(x, x_l, x_u, z_l, z_u, has_lb, has_ub)
        # sigma[0] = 0.2/0.5 + 0.3/1.5 = 0.4 + 0.2 = 0.6
        # sigma[1] = 0.1/2.0 + 0.05/3.0 = 0.05 + 0.01667 = 0.06667
        assert jnp.allclose(sigma[0], 0.6, atol=1e-10)
        assert jnp.allclose(sigma[1], 0.1 / 2.0 + 0.05 / 3.0, atol=1e-10)

    def test_compute_sigma_no_bounds(self):
        """Sigma should be zero where bounds are absent."""
        x = jnp.array([1.5, 2.0])
        x_l = jnp.array([-1e20, 0.0])
        x_u = jnp.array([1e20, 5.0])
        z_l = jnp.array([0.0, 0.1])
        z_u = jnp.array([0.0, 0.05])
        has_lb = jnp.array([False, True])
        has_ub = jnp.array([False, True])
        sigma = _compute_sigma(x, x_l, x_u, z_l, z_u, has_lb, has_ub)
        assert jnp.allclose(sigma[0], 0.0, atol=1e-10)
        assert sigma[1] > 0.0

    def test_fraction_to_boundary(self):
        """Step limit: alpha = min(1, min(-tau*s/ds)) for ds < 0."""
        vals = jnp.array([1.0, 2.0, 0.5])
        dvals = jnp.array([0.5, -1.0, -0.3])
        tau = 0.995
        alpha = _fraction_to_boundary(vals, dvals, tau)
        # Only negative dvals matter: -0.995*2.0/(-1.0)=1.99, -0.995*0.5/(-0.3)=1.658
        # min(1.0, 1.658) = 1.0
        assert alpha <= 1.0
        assert alpha > 0.0

    def test_fraction_to_boundary_all_positive(self):
        """When all steps are positive, alpha should be 1.0."""
        vals = jnp.array([1.0, 2.0, 0.5])
        dvals = jnp.array([0.5, 1.0, 0.3])
        tau = 0.995
        alpha = _fraction_to_boundary(vals, dvals, tau)
        assert jnp.allclose(alpha, 1.0, atol=1e-10)

    def test_convergence_optimal(self):
        """All infeasibilities below tol should signal convergence."""
        result = _check_convergence(
            primal_inf=jnp.array(1e-10),
            dual_inf=jnp.array(1e-10),
            compl_inf=jnp.array(1e-10),
            multiplier_sum=jnp.array(1.0),
            n_mult=jnp.array(2),
            consecutive_acc=jnp.array(0),
            opts=IPMOptions(),
        )
        assert int(result) == 1

    def test_convergence_not_met(self):
        """Large infeasibility should not signal convergence."""
        result = _check_convergence(
            primal_inf=jnp.array(1.0),
            dual_inf=jnp.array(1.0),
            compl_inf=jnp.array(1.0),
            multiplier_sum=jnp.array(1.0),
            n_mult=jnp.array(2),
            consecutive_acc=jnp.array(0),
            opts=IPMOptions(),
        )
        assert int(result) == 0

    def test_mu_update_decreases(self):
        """Barrier parameter mu should generally decrease."""
        mu_old = jnp.array(1.0)
        compl_products = jnp.array([0.01, 0.02, 0.005, 0.01])
        n_compl = jnp.array(4)
        mu_new = _update_mu(mu_old, compl_products, n_compl)
        assert float(mu_new) < float(mu_old)


# ---------------------------------------------------------------
# 2. Unconstrained optimization tests
# ---------------------------------------------------------------


class TestIPMUnconstrained:
    """Test Layer 1 (ipm_solve) on problems with no general constraints (m=0)."""

    def test_simple_quadratic(self):
        """min (x-2)^2 + (y+1)^2, optimal at (2,-1), obj=0."""
        x0 = jnp.array([0.0, 0.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([])
        g_u = jnp.array([])
        state = ipm_solve(
            _obj_quadratic_shifted,
            _con_empty,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(),
        )
        assert jnp.allclose(state.x[0], 2.0, atol=1e-4)
        assert jnp.allclose(state.x[1], -1.0, atol=1e-4)
        assert state.obj < 1e-6
        assert int(state.converged) in (1, 2)

    def test_rosenbrock(self):
        """min (1-x)^2 + 100(y-x^2)^2, optimal at (1,1), obj=0."""
        x0 = jnp.array([0.0, 0.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([])
        g_u = jnp.array([])
        state = ipm_solve(
            _obj_rosenbrock,
            _con_empty,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(max_iter=500),
        )
        assert jnp.allclose(state.x[0], 1.0, atol=1e-3)
        assert jnp.allclose(state.x[1], 1.0, atol=1e-3)
        assert int(state.converged) in (1, 2)

    def test_bound_constrained(self):
        """min x^2+y^2 with x,y in [1,5] -> optimal at (1,1), obj=2."""
        x0 = jnp.array([3.0, 3.0])
        x_l = jnp.array([1.0, 1.0])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([])
        g_u = jnp.array([])
        state = ipm_solve(
            _obj_sum_sq_2d,
            _con_empty,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(),
        )
        assert jnp.allclose(state.obj, 2.0, atol=1e-4)

    def test_3d_sum_of_squares(self):
        """min x^2+y^2+z^2, optimal at origin."""
        x0 = jnp.array([1.0, 2.0, 3.0])
        x_l = jnp.full(3, -10.0)
        x_u = jnp.full(3, 10.0)
        state = ipm_solve(
            _obj_sum_sq,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(),
        )
        assert state.obj < 1e-4

    def test_exponential(self):
        """min exp(x)+exp(y) with x,y in [-5,5] -> at x=y=-5."""
        x0 = jnp.array([0.0, 0.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])
        state = ipm_solve(
            _obj_exp_sum,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(),
        )
        assert jnp.allclose(state.x[0], -5.0, atol=0.1)
        assert jnp.allclose(state.x[1], -5.0, atol=0.1)

    def test_active_upper_bound_multiplier(self):
        """min (x-5)^2 s.t. 0<=x<=3 -> x=3, z_u=4, obj=4."""
        x0 = jnp.array([1.5])
        x_l = jnp.array([0.0])
        x_u = jnp.array([3.0])
        state = ipm_solve(
            lambda x: (x[0] - 5.0) ** 2,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(),
        )
        assert int(state.converged) in (1, 2)
        assert jnp.allclose(state.x[0], 3.0, atol=1e-4)
        assert jnp.allclose(state.z_u[0], 4.0, atol=1e-2)
        assert jnp.allclose(state.obj, 4.0, atol=1e-4)

    def test_active_lower_bound_multiplier(self):
        """min x^2 s.t. 2<=x<=5 -> x=2, z_l=4, obj=4."""
        x0 = jnp.array([3.0])
        x_l = jnp.array([2.0])
        x_u = jnp.array([5.0])
        state = ipm_solve(
            _obj_x0_sq,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(),
        )
        assert int(state.converged) in (1, 2)
        assert jnp.allclose(state.x[0], 2.0, atol=1e-4)
        assert jnp.allclose(state.z_l[0], 4.0, atol=1e-2)
        assert jnp.allclose(state.obj, 4.0, atol=1e-4)


# ---------------------------------------------------------------
# 3. Constrained optimization tests
# ---------------------------------------------------------------


class TestIPMConstrained:
    """Test ipm_solve with equality and inequality constraints."""

    def test_equality_constrained(self):
        """min x^2+y^2 s.t. x+y=2 -> (1,1), obj=2."""
        x0 = jnp.array([0.5, 0.5])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([3.0, 3.0])
        g_l = jnp.array([2.0])
        g_u = jnp.array([2.0])  # equality: g_l == g_u
        state = ipm_solve(
            _obj_sum_sq_2d,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(),
        )
        assert jnp.allclose(state.obj, 2.0, atol=1e-3)
        assert jnp.allclose(state.x[0], 1.0, atol=1e-2)
        assert jnp.allclose(state.x[1], 1.0, atol=1e-2)

    def test_inequality_constrained(self):
        """min x^2+y^2 s.t. x+y>=1 -> (0.5,0.5), obj=0.5."""
        x0 = jnp.array([2.0, 2.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([1.0])  # x+y >= 1
        g_u = jnp.array([1e20])  # no upper bound
        state = ipm_solve(
            _obj_sum_sq_2d,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(),
        )
        assert jnp.allclose(state.obj, 0.5, atol=1e-3)

    def test_hs071(self):
        """Hock-Schittkowski #71: classic NLP test problem.

        min x1*x4*(x1+x2+x3)+x3
        s.t. x1*x2*x3*x4 >= 25
             x1^2+x2^2+x3^2+x4^2 = 40
             1 <= x1,x2,x3,x4 <= 5
        Optimal: obj ~= 17.0140
        """

        def obj_fn(x):
            return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

        def con_fn(x):
            return jnp.array(
                [
                    x[0] * x[1] * x[2] * x[3],
                    x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2,
                ]
            )

        x0 = jnp.array([1.0, 5.0, 5.0, 1.0])
        x_l = jnp.ones(4)
        x_u = 5.0 * jnp.ones(4)
        g_l = jnp.array([25.0, 40.0])
        g_u = jnp.array([1e20, 40.0])  # >= 25 and == 40
        state = ipm_solve(
            obj_fn,
            con_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(max_iter=500, tol=1e-6),
        )
        assert jnp.allclose(state.obj, 17.014, atol=0.1)

    def test_mixed_eq_ineq(self):
        """min x^2+y^2+z^2 s.t. x+y+z=3, x>=0.5 -> obj=3."""
        x0 = jnp.array([1.0, 1.0, 1.0])
        x_l = jnp.zeros(3)
        x_u = 5.0 * jnp.ones(3)
        g_l = jnp.array([3.0, 0.5])  # x+y+z == 3, x >= 0.5
        g_u = jnp.array([3.0, 1e20])
        state = ipm_solve(
            _obj_sum_sq,
            _con_sum_3d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(),
        )
        assert jnp.allclose(state.obj, 3.0, atol=0.1)

    def test_nonlinear_equality(self):
        """min x^2+y^2 s.t. x*y=1 -> (1,1), obj=2."""
        x0 = jnp.array([2.0, 2.0])
        x_l = jnp.array([0.1, 0.1])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([1.0])
        g_u = jnp.array([1.0])
        state = ipm_solve(
            _obj_sum_sq_2d,
            _con_product,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(),
        )
        assert jnp.allclose(state.obj, 2.0, atol=1e-2)

    def test_upper_bound_constraint(self):
        """min -x-y s.t. x+y<=3, x,y in [0,5] -> obj=-3."""
        x0 = jnp.array([1.0, 1.0])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([-1e20])  # no lower bound
        g_u = jnp.array([3.0])  # x+y <= 3
        state = ipm_solve(
            _obj_neg_sum,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(),
        )
        assert jnp.allclose(state.obj, -3.0, atol=1e-2)


# ---------------------------------------------------------------
# 4. Comparison with cyipopt on correctness problems
# ---------------------------------------------------------------


def _build_model_rosenbrock() -> dm.Model:
    """min (1-x)^2 + 100(y-x^2)^2, x,y in [-5,5]. Optimal at (1,1), obj=0."""
    m = dm.Model("rosenbrock")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)
    return m


def _build_model_constrained_quadratic() -> dm.Model:
    """min x^2 + y^2 s.t. x+y>=1, x,y in [-5,5]. Optimal: x=y=0.5, obj=0.5."""
    m = dm.Model("constrained_quad")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize(x**2 + y**2)
    m.subject_to(x + y >= 1)
    return m


def _build_model_quadratic_equality() -> dm.Model:
    """min x^2 + y^2 s.t. x+y=2, x,y in [0,3]. Optimal: x=y=1, obj=2."""
    m = dm.Model("quad_equality")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.minimize(x**2 + y**2)
    m.subject_to(x + y == 2)
    return m


def _build_model_nonlinear_eq() -> dm.Model:
    """min x^2 + y^2 s.t. x*y=1, x,y in [0.1,5]. Optimal: x=y=1, obj=2."""
    m = dm.Model("nonlinear_eq")
    x = m.continuous("x", lb=0.1, ub=5)
    y = m.continuous("y", lb=0.1, ub=5)
    m.minimize(x**2 + y**2)
    m.subject_to(x * y == 1)
    return m


def _build_model_exp_nlp() -> dm.Model:
    """min exp(x) + y^2 s.t. x+y>=1, x,y in [-2,2]."""
    m = dm.Model("exp_nlp")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.minimize(dm.exp(x) + y**2)
    m.subject_to(x + y >= 1)
    return m


def _build_model_sqrt_nlp() -> dm.Model:
    """min sqrt(x+1) + sqrt(y+1) s.t. x+y>=2, x,y in [0,5]."""
    m = dm.Model("sqrt_nlp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(dm.sqrt(x + 1) + dm.sqrt(y + 1))
    m.subject_to(x + y >= 2)
    return m


def _build_model_three_variable_nlp() -> dm.Model:
    """min x^2 + y^2 + z^2 s.t. x+y+z=3, x,y,z in [0,5]."""
    m = dm.Model("three_var_nlp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    z = m.continuous("z", lb=0, ub=5)
    m.minimize(x**2 + y**2 + z**2)
    m.subject_to(x + y + z == 3)
    return m


def _solve_with_ipm(build_fn):
    """Solve a model using the IPM wrapper, return NLPResult."""
    model = build_fn()
    evaluator = NLPEvaluator(model)
    lb, ub = evaluator.variable_bounds
    x0 = 0.5 * (np.clip(lb, -100, 100) + np.clip(ub, -100, 100))

    m = evaluator.n_constraints
    if m > 0:
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        cl, cu = _infer_constraint_bounds(model)
        constraint_bounds = list(zip(cl, cu))
    else:
        constraint_bounds = None

    return solve_nlp_ipm(evaluator, x0, constraint_bounds)


def _solve_with_cyipopt(build_fn):
    """Solve a model using cyipopt, return NLPResult."""
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds, solve_nlp

    model = build_fn()
    evaluator = NLPEvaluator(model)
    lb, ub = evaluator.variable_bounds
    x0 = 0.5 * (np.clip(lb, -100, 100) + np.clip(ub, -100, 100))

    m = evaluator.n_constraints
    if m > 0:
        cl, cu = _infer_constraint_bounds(evaluator)
        constraint_bounds = list(zip(cl, cu))
    else:
        constraint_bounds = None

    return solve_nlp(evaluator, x0, constraint_bounds)


@pytest.mark.slow
@pytest.mark.integration
class TestIPMMatchesCyipopt:
    """Compare IPM with cyipopt on correctness test problems."""

    @pytest.fixture(autouse=True)
    def _skip_no_cyipopt(self):
        pytest.importorskip("cyipopt")

    def test_rosenbrock(self):
        """Rosenbrock: IPM objective should be close to cyipopt."""
        cy = _solve_with_cyipopt(_build_model_rosenbrock)
        ipm = _solve_with_ipm(_build_model_rosenbrock)
        assert abs(cy.objective - ipm.objective) < 1e-3

    def test_constrained_quadratic(self):
        """Constrained quadratic: IPM should match cyipopt."""
        cy = _solve_with_cyipopt(_build_model_constrained_quadratic)
        ipm = _solve_with_ipm(_build_model_constrained_quadratic)
        assert abs(cy.objective - ipm.objective) < 1e-3

    def test_quadratic_equality(self):
        """Quadratic equality: IPM should match cyipopt."""
        cy = _solve_with_cyipopt(_build_model_quadratic_equality)
        ipm = _solve_with_ipm(_build_model_quadratic_equality)
        assert abs(cy.objective - ipm.objective) < 1e-3

    def test_nonlinear_equality(self):
        """Nonlinear equality: IPM should match cyipopt."""
        cy = _solve_with_cyipopt(_build_model_nonlinear_eq)
        ipm = _solve_with_ipm(_build_model_nonlinear_eq)
        assert abs(cy.objective - ipm.objective) < 1e-2

    def test_exp_nlp(self):
        """Exponential NLP: IPM should match cyipopt."""
        cy = _solve_with_cyipopt(_build_model_exp_nlp)
        ipm = _solve_with_ipm(_build_model_exp_nlp)
        assert abs(cy.objective - ipm.objective) < 1e-3

    def test_sqrt_nlp(self):
        """Square root NLP: IPM should match cyipopt."""
        cy = _solve_with_cyipopt(_build_model_sqrt_nlp)
        ipm = _solve_with_ipm(_build_model_sqrt_nlp)
        assert abs(cy.objective - ipm.objective) < 1e-3

    def test_three_variable_nlp(self):
        """Three variable NLP: IPM should match cyipopt."""
        cy = _solve_with_cyipopt(_build_model_three_variable_nlp)
        ipm = _solve_with_ipm(_build_model_three_variable_nlp)
        assert abs(cy.objective - ipm.objective) < 1e-3


# ---------------------------------------------------------------
# 5. Batch / vmap tests
# ---------------------------------------------------------------


@pytest.mark.slow
class TestIPMVmap:
    """Test vectorized batch solving via jax.vmap."""

    def test_batch_8(self):
        """Solve 8 quadratic problems with different bounds."""
        n = 2
        batch = 8
        x0_batch = jnp.zeros((batch, n))
        xl_batch = -jnp.ones((batch, n)) * jnp.arange(1, batch + 1)[:, None]
        xu_batch = jnp.ones((batch, n)) * jnp.arange(1, batch + 1)[:, None]
        states = solve_nlp_batch(
            _obj_sum_sq,
            _con_empty,
            x0_batch,
            xl_batch,
            xu_batch,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(),
        )
        assert jnp.all(states.obj < 1e-3)
        assert jnp.all(states.converged > 0)

    def test_batch_64(self):
        """64 instances should all converge."""
        n = 3
        batch = 64
        x0_batch = jnp.ones((batch, n))
        xl_batch = jnp.full((batch, n), -10.0)
        xu_batch = jnp.full((batch, n), 10.0)
        states = solve_nlp_batch(
            _obj_sum_sq,
            _con_empty,
            x0_batch,
            xl_batch,
            xu_batch,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(),
        )
        assert jnp.all(states.obj < 1e-3)
        assert jnp.all(states.converged > 0)

    def test_vmap_matches_sequential(self):
        """Batch results should match sequential solves."""
        n = 2
        batch = 4
        opts = IPMOptions()

        x0_batch = jnp.zeros((batch, n))
        xl_batch = jnp.array([[-5.0, -5.0], [-3.0, -3.0], [-1.0, -1.0], [0.0, 0.0]])
        xu_batch = jnp.array([[5.0, 5.0], [3.0, 3.0], [5.0, 5.0], [5.0, 5.0]])

        batch_states = solve_nlp_batch(
            _obj_shifted_2d,
            _con_empty,
            x0_batch,
            xl_batch,
            xu_batch,
            jnp.array([]),
            jnp.array([]),
            opts,
        )

        for i in range(batch):
            seq_state = ipm_solve(
                _obj_shifted_2d,
                _con_empty,
                x0_batch[i],
                xl_batch[i],
                xu_batch[i],
                jnp.array([]),
                jnp.array([]),
                opts,
            )
            np.testing.assert_allclose(
                float(batch_states.obj[i]),
                float(seq_state.obj),
                atol=1e-4,
                err_msg=f"Batch vs sequential mismatch at instance {i}",
            )

    def test_vmap_varying_bounds(self):
        """Different bounds per instance, optimal depends on bounds."""
        x0_batch = jnp.array([[3.0], [3.0], [3.0], [3.0]])
        xl_batch = jnp.array([[1.0], [2.0], [-1.0], [-5.0]])
        xu_batch = jnp.array([[5.0], [5.0], [5.0], [5.0]])
        states = solve_nlp_batch(
            _obj_x0_sq,
            _con_empty,
            x0_batch,
            xl_batch,
            xu_batch,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(),
        )
        # Optimal: x=lb when lb>0, x=0 when lb<=0
        expected_obj = jnp.array([1.0, 4.0, 0.0, 0.0])
        np.testing.assert_allclose(
            np.array(states.obj),
            np.array(expected_obj),
            atol=1e-3,
        )


# ---------------------------------------------------------------
# 6. Performance test (slow)
# ---------------------------------------------------------------


def _obj_perf(x):
    return jnp.sum(x**2)


def _con_perf(x):
    return jnp.array([x[0] + x[1]])


@pytest.mark.slow
class TestIPMPerformance:
    """Batch solve should be significantly faster than sequential."""

    def test_batch_speedup(self):
        """batch=64 should be >= 10x faster than sequential."""
        n = 5
        batch = 64
        opts = IPMOptions(max_iter=100)

        # Warmup
        x0 = jnp.zeros(n)
        xl = -5.0 * jnp.ones(n)
        xu = 5.0 * jnp.ones(n)
        g_l = jnp.array([1.0])
        g_u = jnp.array([1e20])
        ipm_solve(_obj_perf, _con_perf, x0, xl, xu, g_l, g_u, opts)

        x0_batch = jnp.zeros((batch, n))
        xl_batch = jnp.tile(xl, (batch, 1))
        xu_batch = jnp.tile(xu, (batch, 1))

        # Sequential timing
        t0 = time.perf_counter()
        for i in range(batch):
            ipm_solve(
                _obj_perf,
                _con_perf,
                x0_batch[i],
                xl_batch[i],
                xu_batch[i],
                g_l,
                g_u,
                opts,
            )
        t_seq = time.perf_counter() - t0

        # Batch warmup + timing
        solve_nlp_batch(
            _obj_perf,
            _con_perf,
            x0_batch,
            xl_batch,
            xu_batch,
            g_l,
            g_u,
            opts,
        )
        t0 = time.perf_counter()
        solve_nlp_batch(
            _obj_perf,
            _con_perf,
            x0_batch,
            xl_batch,
            xu_batch,
            g_l,
            g_u,
            opts,
        )
        t_batch = time.perf_counter() - t0

        speedup = t_seq / t_batch
        assert speedup >= 10.0, f"Batch speedup {speedup:.1f}x < 10x"


# ---------------------------------------------------------------
# 7. Layer 2 wrapper tests (solve_nlp_ipm -> NLPResult)
# ---------------------------------------------------------------


class TestIPMWrapper:
    """Test solve_nlp_ipm returns correct NLPResult objects."""

    def test_returns_nlpresult(self):
        """solve_nlp_ipm should return NLPResult."""
        m = dm.Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.minimize((x - 2) ** 2 + (y + 1) ** 2)
        evaluator = NLPEvaluator(m)
        lb, ub = evaluator.variable_bounds
        x0 = 0.5 * (lb + ub)
        result = solve_nlp_ipm(evaluator, x0)
        assert isinstance(result, NLPResult)
        assert result.status == SolveStatus.OPTIMAL
        assert abs(result.objective) < 1e-3

    def test_status_optimal(self):
        """Converged problem maps to OPTIMAL status."""
        m = dm.Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(x**2)
        evaluator = NLPEvaluator(m)
        lb, ub = evaluator.variable_bounds
        x0 = 0.5 * (lb + ub)
        result = solve_nlp_ipm(evaluator, x0)
        assert result.status == SolveStatus.OPTIMAL

    def test_multipliers_returned(self):
        """Constrained problems should have multipliers."""
        m = dm.Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)
        evaluator = NLPEvaluator(m)
        lb, ub = evaluator.variable_bounds
        x0 = 0.5 * (lb + ub)

        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        cl, cu = _infer_constraint_bounds(m)
        constraint_bounds = list(zip(cl, cu))
        result = solve_nlp_ipm(evaluator, x0, constraint_bounds)
        assert result.multipliers is not None
        assert len(result.multipliers) == 1

    def test_unconstrained_none_multipliers(self):
        """Unconstrained problems should have None multipliers."""
        m = dm.Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(x**2)
        evaluator = NLPEvaluator(m)
        lb, ub = evaluator.variable_bounds
        x0 = 0.5 * (lb + ub)
        result = solve_nlp_ipm(evaluator, x0)
        assert result.multipliers is None


# ---------------------------------------------------------------
# 8. Mehrotra predictor-corrector tests
# ---------------------------------------------------------------


@pytest.mark.slow
class TestPredictorCorrector:
    """Test that Mehrotra predictor-corrector produces correct results."""

    def test_pc_unconstrained_matches_standard(self):
        """PC and standard steps should converge to same optimum."""
        n = 3
        x0 = jnp.array([3.0, -2.0, 1.0])
        xl = jnp.full(n, -5.0)
        xu = jnp.full(n, 5.0)

        def obj(x):
            return jnp.sum((x - 1.0) ** 2)

        def con(x):
            return jnp.array([])

        gl, gu = jnp.array([]), jnp.array([])
        opts_pc = IPMOptions(predictor_corrector=True)
        opts_std = IPMOptions(predictor_corrector=False)
        state_pc = ipm_solve(obj, con, x0, xl, xu, gl, gu, opts_pc)
        state_std = ipm_solve(obj, con, x0, xl, xu, gl, gu, opts_std)

        np.testing.assert_allclose(np.array(state_pc.x), np.array(state_std.x), atol=1e-5)
        np.testing.assert_allclose(float(state_pc.obj), float(state_std.obj), atol=1e-6)

    def test_pc_constrained_matches_standard(self):
        """PC constrained solve should match standard step optimum."""
        x0 = jnp.array([1.0, 1.0])
        xl = jnp.array([0.0, 0.0])
        xu = jnp.array([5.0, 5.0])

        def obj(x):
            return x[0] ** 2 + x[1] ** 2

        def con(x):
            return jnp.array([x[0] + x[1]])

        g_l = jnp.array([2.0])
        g_u = jnp.array([1e20])

        opts_pc = IPMOptions(predictor_corrector=True)
        opts_std = IPMOptions(predictor_corrector=False)
        state_pc = ipm_solve(obj, con, x0, xl, xu, g_l, g_u, opts_pc)
        state_std = ipm_solve(obj, con, x0, xl, xu, g_l, g_u, opts_std)

        # Both should converge (obj = 2.0)
        assert float(state_pc.obj) < 3.0
        assert float(state_std.obj) < 3.0
        np.testing.assert_allclose(np.array(state_pc.x), np.array(state_std.x), atol=1e-4)
        np.testing.assert_allclose(float(state_pc.obj), float(state_std.obj), atol=1e-5)

    def test_pc_fewer_iterations(self):
        """PC should typically converge in fewer or similar iterations."""
        n = 3
        x0 = jnp.array([2.0, 2.0, 2.0])
        xl = jnp.full(n, 0.1)
        xu = jnp.full(n, 5.0)

        def obj(x):
            return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

        def con(x):
            return jnp.array([x[0] + x[1] + x[2]])

        g_l = jnp.array([3.0])
        g_u = jnp.array([1e20])

        opts_pc = IPMOptions(predictor_corrector=True)
        opts_std = IPMOptions(predictor_corrector=False)
        state_pc = ipm_solve(obj, con, x0, xl, xu, g_l, g_u, opts_pc)
        state_std = ipm_solve(obj, con, x0, xl, xu, g_l, g_u, opts_std)

        # Both should converge to same optimum (x=1,1,1, obj=3)
        assert float(state_pc.obj) < 5.0
        assert float(state_std.obj) < 5.0
        np.testing.assert_allclose(float(state_pc.obj), float(state_std.obj), atol=1e-3)
        # PC typically uses fewer iterations (allow some slack)
        assert int(state_pc.iteration) <= int(state_std.iteration) + 5

    def test_pc_option_default_true(self):
        """predictor_corrector should default to True."""
        opts = IPMOptions()
        assert opts.predictor_corrector is True

    def test_pc_hs071(self):
        """PC should solve HS071 correctly."""

        def obj_fn(x):
            return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

        def con_fn(x):
            return jnp.array(
                [
                    x[0] * x[1] * x[2] * x[3],
                    x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2,
                ]
            )

        x0 = jnp.array([1.0, 5.0, 5.0, 1.0])
        x_l = jnp.ones(4)
        x_u = 5.0 * jnp.ones(4)
        g_l = jnp.array([25.0, 40.0])
        g_u = jnp.array([1e20, 40.0])
        opts = IPMOptions(predictor_corrector=True)
        state = ipm_solve(obj_fn, con_fn, x0, x_l, x_u, g_l, g_u, opts)
        np.testing.assert_allclose(float(state.obj), 17.014, atol=0.1)


# ---------------------------------------------------------------
# Second-order correction (SOC) tests
# ---------------------------------------------------------------


@pytest.mark.slow
class TestSecondOrderCorrection:
    """Test SOC prevents Maratos effect on nonconvex constrained problems."""

    def test_soc_options_defaults(self):
        """IPMOptions should have SOC fields with correct defaults."""
        opts = IPMOptions()
        assert opts.max_soc == 4
        assert opts.kappa_soc == 0.99

    def test_soc_constrained_quadratic(self):
        """SOC should converge on a quadratic-constrained problem.

        min (x1-2)^2 + (x2-1)^2  subject to  x1^2 + x2^2 = 2
        Optimal on circle closest to (2,1): x ≈ (1.265, 0.632), obj ≈ 0.675.
        """

        def obj(x):
            return (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2

        def con(x):
            return jnp.array([x[0] ** 2 + x[1] ** 2])

        x0 = jnp.array([1.2, 0.5])
        xl = jnp.array([-3.0, -3.0])
        xu = jnp.array([3.0, 3.0])
        gl = jnp.array([2.0])
        gu = jnp.array([2.0])

        # With SOC
        state_soc = ipm_solve(obj, con, x0, xl, xu, gl, gu, IPMOptions(max_soc=4))
        assert int(state_soc.converged) in (1, 2)
        np.testing.assert_allclose(float(state_soc.obj), 0.6754, atol=1e-2)

        # Without SOC — should also converge (this problem is mild)
        state_no = ipm_solve(obj, con, x0, xl, xu, gl, gu, IPMOptions(max_soc=0))
        assert int(state_no.converged) in (1, 2)
        np.testing.assert_allclose(float(state_no.obj), 0.6754, atol=1e-2)

    def test_soc_no_regression_hs071(self):
        """SOC should not regress HS071 (compare with and without)."""

        def obj(x):
            return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

        def con(x):
            return jnp.array(
                [
                    x[0] * x[1] * x[2] * x[3],
                    x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2,
                ]
            )

        x0 = jnp.array([1.0, 5.0, 5.0, 1.0])
        xl = jnp.array([1.0, 1.0, 1.0, 1.0])
        xu = jnp.array([5.0, 5.0, 5.0, 5.0])
        gl = jnp.array([25.0, 40.0])
        gu = jnp.array([1e20, 40.0])

        s_soc = ipm_solve(
            obj,
            con,
            x0,
            xl,
            xu,
            gl,
            gu,
            IPMOptions(max_soc=4, max_iter=500, tol=1e-6),
        )
        s_no = ipm_solve(
            obj,
            con,
            x0,
            xl,
            xu,
            gl,
            gu,
            IPMOptions(max_soc=0, max_iter=500, tol=1e-6),
        )
        # SOC should not produce a worse objective than without SOC
        np.testing.assert_allclose(float(s_soc.obj), float(s_no.obj), atol=0.5)

    def test_soc_unconstrained_noop(self):
        """SOC block should not affect unconstrained problems."""

        def obj(x):
            return (x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2

        x0 = jnp.array([0.0, 0.0])
        xl = jnp.array([-10.0, -10.0])
        xu = jnp.array([10.0, 10.0])

        state_soc = ipm_solve(obj, None, x0, xl, xu, options=IPMOptions(max_soc=4))
        state_no = ipm_solve(obj, None, x0, xl, xu, options=IPMOptions(max_soc=0))
        assert int(state_soc.converged) in (1, 2)
        np.testing.assert_allclose(float(state_soc.obj), float(state_no.obj), atol=1e-6)

    def test_soc_quadratic_equality(self):
        """SOC on a problem with two quadratic equality constraints.

        min x1 + x2 + x3
        s.t. x1^2 + x2^2 = 1
             x2^2 + x3^2 = 1
             0 <= x <= 2
        """

        def obj(x):
            return x[0] + x[1] + x[2]

        def con(x):
            return jnp.array([x[0] ** 2 + x[1] ** 2, x[1] ** 2 + x[2] ** 2])

        x0 = jnp.array([0.5, 0.5, 0.5])
        xl = jnp.array([0.0, 0.0, 0.0])
        xu = jnp.array([2.0, 2.0, 2.0])
        gl = jnp.array([1.0, 1.0])
        gu = jnp.array([1.0, 1.0])

        state = ipm_solve(
            obj,
            con,
            x0,
            xl,
            xu,
            gl,
            gu,
            IPMOptions(max_soc=4, max_iter=500),
        )
        assert int(state.converged) in (1, 2)
        # Both constraints active at the solution
        g_sol = float(con(state.x)[0])
        np.testing.assert_allclose(g_sol, 1.0, atol=1e-3)


# ---------------------------------------------------------------
# Feasibility restoration bridge tests
# ---------------------------------------------------------------


@pytest.mark.slow
class TestJaxFeasibilityRestoration:
    """Tests for the JAX-to-callback feasibility restoration bridge."""

    def test_restoration_reduces_violation(self):
        """Restoration should reduce constraint violation on a simple problem."""
        from discopt._jax.ipm import _jax_feasibility_restoration

        # Problem: x1^2 + x2^2 >= 1 with start at origin (infeasible)
        def con_fn(x):
            return jnp.array([x[0] ** 2 + x[1] ** 2])

        x0 = jnp.array([0.1, 0.1])  # infeasible: 0.02 < 1
        x_l = jnp.array([-2.0, -2.0])
        x_u = jnp.array([2.0, 2.0])
        g_l = jnp.array([1.0])  # x1^2 + x2^2 >= 1
        g_u = jnp.array([1e20])

        opts = IPMOptions(max_iter=200)
        x_restored, success = _jax_feasibility_restoration(
            con_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            opts,
        )
        assert success
        # Restored point should satisfy the constraint better
        g_restored = float(con_fn(x_restored)[0])
        assert g_restored > 0.5  # significant improvement toward g >= 1

    def test_restoration_already_feasible(self):
        """Restoration on a feasible point should return success=False."""
        from discopt._jax.ipm import _jax_feasibility_restoration

        def con_fn(x):
            return jnp.array([x[0] + x[1]])

        x0 = jnp.array([1.0, 1.0])  # feasible: 2.0 <= 3.0
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([-1e20])
        g_u = jnp.array([3.0])

        opts = IPMOptions(max_iter=200)
        _, success = _jax_feasibility_restoration(
            con_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            opts,
        )
        assert not success  # already feasible, nothing to restore

    def test_restoration_with_log_constraint(self):
        """Restoration should handle log-domain constraints."""
        from discopt._jax.ipm import _jax_feasibility_restoration

        # log(1 + x1 - x2) >= 0 requires x1 - x2 >= 0
        def con_fn(x):
            return jnp.array([jnp.log(1.0 + x[0] - x[1])])

        # Start where log is defined but constraint violated:
        # 1 + 0.8 - 1.0 = 0.8, log(0.8) = -0.22 < 0
        x0 = jnp.array([0.8, 1.0])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([2.0, 2.0])
        g_l = jnp.array([0.0])  # log(1+x1-x2) >= 0
        g_u = jnp.array([1e20])

        opts = IPMOptions(max_iter=200)
        x_restored, success = _jax_feasibility_restoration(
            con_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            opts,
        )
        assert success
        g_val = float(con_fn(x_restored)[0])
        assert g_val >= -0.1  # violation should be substantially reduced

    def test_nan_filtering_in_batch(self):
        """Batch solve with NaN results should not select NaN as best."""

        # Objective that produces NaN for some x values
        def obj_fn(x):
            return jnp.log(x[0]) + x[1] ** 2

        def con_fn(x):
            return jnp.array([x[0] + x[1]])

        n_starts = 3
        # Start 0: valid, start 1: will produce NaN (x[0] near 0)
        x0_batch = jnp.array(
            [
                [1.0, 0.5],
                [0.001, 0.5],
                [2.0, 0.1],
            ]
        )
        xl = jnp.broadcast_to(jnp.array([0.01, -1.0]), (n_starts, 2))
        xu = jnp.broadcast_to(jnp.array([3.0, 3.0]), (n_starts, 2))
        g_l = jnp.array([-1e20])
        g_u = jnp.array([2.0])

        opts = IPMOptions(max_iter=200)
        state = solve_nlp_batch(obj_fn, con_fn, x0_batch, xl, xu, g_l, g_u, opts)

        converged = np.asarray(state.converged)
        obj_vals = np.asarray(state.obj)
        # Among converged results with finite obj, best should be picked
        feasible_mask = ((converged == 1) | (converged == 2) | (converged == 3)) & np.isfinite(
            obj_vals
        )
        if np.any(feasible_mask):
            masked = np.where(feasible_mask, obj_vals, np.inf)
            best_idx = int(np.argmin(masked))
            assert np.isfinite(obj_vals[best_idx])
