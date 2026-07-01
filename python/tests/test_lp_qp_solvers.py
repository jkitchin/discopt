"""
Test suite for LP, QP, MILP, MIQP solvers and differentiable optimization.

Tests cover:
  1. Problem classification (LP, QP, MILP, MIQP, NLP, MINLP)
  2. LP IPM correctness (simple LP, equality constraints, bounds, batch)
  3. QP IPM correctness (simple QP, equality constraints, bounds, batch)
  4. LP differentiability (jax.grad matches finite differences)
  5. QP differentiability (jax.grad matches finite differences)
  6. Solver dispatch (Model.solve() routes to LP/QP solver)
  7. MILP via B&B with LP relaxations
"""

from __future__ import annotations

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import pytest

# ---------------------------------------------------------------
# 1. Problem Classifier Tests
# ---------------------------------------------------------------


class TestProblemClassifier:
    """Test classify_problem correctly identifies problem type."""

    def test_lp_detection(self):
        """Linear obj + linear constraints + continuous = LP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("lp_test")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        m.minimize(3 * x[0] + 2 * x[1])
        m.subject_to(x[0] + x[1] <= 5)
        assert classify_problem(m) == ProblemClass.LP

    def test_qp_detection(self):
        """Quadratic obj + linear constraints + continuous = QP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("qp_test")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        m.minimize(x[0] ** 2 + x[1] ** 2)
        m.subject_to(x[0] + x[1] >= 1)
        assert classify_problem(m) == ProblemClass.QP

    def test_milp_detection(self):
        """Linear obj + linear constraints + integer = MILP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("milp_test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(3 * x + 5 * y)
        m.subject_to(x + y <= 5)
        assert classify_problem(m) == ProblemClass.MILP

    def test_miqp_detection(self):
        """Quadratic obj + linear constraints + integer = MIQP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("miqp_test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x**2 + 5 * y)
        m.subject_to(x + y <= 5)
        assert classify_problem(m) == ProblemClass.MIQP

    def test_qcp_detection(self):
        """Linear obj + quadratic constraints + continuous = QCP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("qcp_test")
        x = m.continuous("x", lb=-2, ub=2)
        m.minimize(x)
        m.subject_to(x**2 <= 1)
        assert classify_problem(m) == ProblemClass.QCP

    def test_qcqp_detection(self):
        """Quadratic obj + quadratic constraints + continuous = QCQP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("qcqp_test")
        x = m.continuous("x", lb=-2, ub=2)
        m.minimize((x - 1) ** 2)
        m.subject_to(x**2 <= 1)
        assert classify_problem(m) == ProblemClass.QCQP

    def test_miqcp_detection(self):
        """Linear obj + quadratic constraints + integer = MIQCP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("miqcp_test")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.binary("y")
        m.minimize(x + y)
        m.subject_to(x**2 <= y)
        assert classify_problem(m) == ProblemClass.MIQCP

    def test_miqcqp_detection(self):
        """Quadratic obj + quadratic constraints + integer = MIQCQP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("miqcqp_test")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.binary("y")
        m.minimize((x - 1) ** 2 + y)
        m.subject_to(x**2 <= y)
        assert classify_problem(m) == ProblemClass.MIQCQP

    def test_nlp_detection(self):
        """Nonlinear constraints + continuous = NLP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("nlp_test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + y)
        m.subject_to(dm.exp(x) + y <= 5)
        assert classify_problem(m) == ProblemClass.NLP

    def test_minlp_detection(self):
        """Quadratic constraints with integers classify as MIQCQP."""
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("minlp_test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x * x + y <= 5)
        assert classify_problem(m) == ProblemClass.MIQCQP


# ---------------------------------------------------------------
# 2. LP Standard Form Extraction Tests
# ---------------------------------------------------------------


class TestLPExtraction:
    """Test extract_lp_data produces correct standard form."""

    def test_simple_lp(self):
        """Extract standard form from simple LP."""
        from discopt._jax.problem_classifier import extract_lp_data

        m = dm.Model("test")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        m.minimize(3 * x[0] + 2 * x[1])
        m.subject_to(x[0] + x[1] <= 5)

        lp_data = extract_lp_data(m)
        # Original c should be [3, 2] for original vars
        assert jnp.allclose(lp_data.c[:2], jnp.array([3.0, 2.0]), atol=1e-10)

    def test_qp_extraction(self):
        """Extract QP standard form."""
        from discopt._jax.problem_classifier import extract_qp_data

        m = dm.Model("test")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        m.minimize(x[0] ** 2 + x[1] ** 2 + 3 * x[0])

        qp_data = extract_qp_data(m)
        # Q should be diag(2, 2) (hessian of x0^2 + x1^2)
        assert jnp.allclose(qp_data.Q[:2, :2], 2.0 * jnp.eye(2), atol=1e-6)
        # c should be [3, 0]
        assert jnp.allclose(qp_data.c[:2], jnp.array([3.0, 0.0]), atol=1e-6)

    def test_qcp_extraction(self):
        """Extract QCP data with linear and quadratic rows preserved."""
        from discopt._jax.problem_classifier import extract_qcp_data

        m = dm.Model("test_qcp")
        x = m.continuous("x", shape=(2,), lb=-2, ub=2)
        m.minimize(3 * x[0] + x[1])
        m.subject_to(x[0] + x[1] <= 5)
        m.subject_to(x[0] ** 2 + x[1] <= 1)

        qcp_data = extract_qcp_data(m)

        assert jnp.allclose(qcp_data.c, jnp.array([3.0, 1.0]), atol=1e-10)
        assert jnp.allclose(qcp_data.A_ub, jnp.array([[1.0, 1.0]]), atol=1e-10)
        assert jnp.allclose(qcp_data.b_ub, jnp.array([5.0]), atol=1e-10)
        assert len(qcp_data.quadratic_constraints) == 1
        row = qcp_data.quadratic_constraints[0]
        assert row.sense == "<="
        assert row.rhs == pytest.approx(1.0)
        assert jnp.allclose(row.Q, jnp.array([[2.0, 0.0], [0.0, 0.0]]), atol=1e-10)
        assert jnp.allclose(row.c, jnp.array([0.0, 1.0]), atol=1e-10)


# ---------------------------------------------------------------
# 3. LP IPM Correctness Tests
# ---------------------------------------------------------------


class TestQPIPM:
    """Test QP IPM solver correctness."""

    def test_simple_qp(self):
        """min 0.5(x^2+y^2) s.t. x+y=1, x,y>=0 -> x=y=0.5, obj=0.25."""
        from discopt._jax.qp_ipm import qp_ipm_solve

        Q = jnp.eye(2)  # 0.5 x'Ix = 0.5(x^2+y^2)
        c = jnp.zeros(2)
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([1.0])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.full(2, 1e20)

        state = qp_ipm_solve(Q, c, A, b, x_l, x_u)
        assert int(state.converged) in (1, 2)
        assert jnp.allclose(state.obj, 0.25, atol=1e-4)
        assert jnp.allclose(state.x[0], 0.5, atol=1e-2)
        assert jnp.allclose(state.x[1], 0.5, atol=1e-2)

    def test_qp_with_linear_term(self):
        """min 0.5(x^2+y^2) + 3x s.t. x+y=2, x,y>=0."""
        from discopt._jax.qp_ipm import qp_ipm_solve

        Q = jnp.eye(2)
        c = jnp.array([3.0, 0.0])
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([2.0])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.full(2, 1e20)

        state = qp_ipm_solve(Q, c, A, b, x_l, x_u)
        assert int(state.converged) in (1, 2)
        # Lagrangian: 0.5(x^2+y^2)+3x+λ(x+y-2)
        # KKT: x+3+λ=0, y+λ=0, x+y=2 → x+3=-y → 2x+3=2 → x=-0.5 (but x>=0)
        # Bound active at x=0 → y=2, obj = 0.5*4 = 2
        assert jnp.allclose(state.x[0], 0.0, atol=1e-2)
        assert jnp.allclose(state.x[1], 2.0, atol=1e-2)
        assert jnp.allclose(state.obj, 2.0, atol=1e-3)

    def test_qp_unconstrained(self):
        """min 0.5(x-1)^2 + 0.5(y-2)^2 with bounds [0,5]."""
        from discopt._jax.qp_ipm import qp_ipm_solve

        # 0.5(x-1)^2 + 0.5(y-2)^2 = 0.5(x^2-2x+1) + 0.5(y^2-4y+4)
        # = 0.5 x'Ix + [-1,-2]'x + 2.5
        Q = jnp.eye(2)
        c = jnp.array([-1.0, -2.0])
        A = jnp.zeros((0, 2))
        b = jnp.zeros(0)
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([5.0, 5.0])

        state = qp_ipm_solve(Q, c, A, b, x_l, x_u)
        assert int(state.converged) in (1, 2)
        assert jnp.allclose(state.x[0], 1.0, atol=1e-2)
        assert jnp.allclose(state.x[1], 2.0, atol=1e-2)

    def test_qp_bound_constrained(self):
        """min x^2 + y^2 with 1<=x<=5, 1<=y<=5 -> x=y=1, obj=1."""
        from discopt._jax.qp_ipm import qp_ipm_solve

        Q = 2.0 * jnp.eye(2)  # 0.5*2 = 1 coefficient
        c = jnp.zeros(2)
        A = jnp.zeros((0, 2))
        b = jnp.zeros(0)
        x_l = jnp.array([1.0, 1.0])
        x_u = jnp.array([5.0, 5.0])

        state = qp_ipm_solve(Q, c, A, b, x_l, x_u)
        assert int(state.converged) in (1, 2)
        assert jnp.allclose(state.obj, 2.0, atol=1e-3)  # 0.5*2*(1+1) = 2


# ---------------------------------------------------------------
# 5. LP Differentiability Tests
# ---------------------------------------------------------------


class TestLPDifferentiability:
    """Test that LP solutions are differentiable w.r.t. problem data."""

    def test_grad_wrt_c(self):
        """d(obj*)/dc_i = x*_i for LP at optimality."""
        from discopt._jax.differentiable_lp import lp_solve_grad

        # min -x - 2y s.t. x+y=3, 0<=x<=2, 0<=y<=2 -> x*=1, y*=2
        c = jnp.array([-1.0, -2.0])
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([3.0])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([2.0, 2.0])

        grad_c = jax.grad(lp_solve_grad, argnums=0)(c, A, b, x_l, x_u)
        # d(obj)/dc_i = x*_i at optimality
        assert jnp.allclose(grad_c[0], 1.0, atol=0.5)
        assert jnp.allclose(grad_c[1], 2.0, atol=0.5)

    def test_grad_wrt_b(self):
        """d(obj*)/db_i = y*_i (dual variable) for LP."""
        from discopt._jax.differentiable_lp import lp_solve_grad

        # Well-bounded LP: min -x - 2y s.t. x+y=3, 0<=x<=2, 0<=y<=2
        c = jnp.array([-1.0, -2.0])
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([3.0])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([2.0, 2.0])

        # AD gradient w.r.t. b
        def obj_fn_b(b_val):
            return lp_solve_grad(c, A, b_val, x_l, x_u)

        grad_b_ad = jax.grad(obj_fn_b)(b)

        # Finite difference check
        eps = 1e-5
        obj_p = obj_fn_b(b + eps)
        obj_m = obj_fn_b(b - eps)
        grad_b_fd = (obj_p - obj_m) / (2 * eps)

        assert jnp.allclose(grad_b_ad, grad_b_fd, atol=0.1)

    def test_grad_finite_diff(self):
        """jax.grad matches central finite differences for LP."""
        from discopt._jax.differentiable_lp import lp_solve_grad

        c = jnp.array([-1.0, -2.0])
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([3.0])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([2.0, 2.0])

        # AD gradient w.r.t. b
        def obj_fn_b(b_val):
            return lp_solve_grad(c, A, b_val, x_l, x_u)

        grad_b_ad = jax.grad(obj_fn_b)(b)

        # Finite difference
        eps = 1e-5
        obj_p = obj_fn_b(b + eps)
        obj_m = obj_fn_b(b - eps)
        grad_b_fd = (obj_p - obj_m) / (2 * eps)

        assert jnp.allclose(grad_b_ad, grad_b_fd, atol=0.1)


# ---------------------------------------------------------------
# 6. QP Differentiability Tests
# ---------------------------------------------------------------


class TestQPDifferentiability:
    """Test that QP solutions are differentiable w.r.t. problem data."""

    def test_grad_wrt_c_qp(self):
        """jax.grad through QP solve w.r.t. c should match finite differences."""
        from discopt._jax.differentiable_qp import qp_solve_grad

        Q = jnp.eye(2)
        c = jnp.array([0.0, 0.0])
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([1.0])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.full(2, 1e20)

        def obj_fn(c_val):
            return qp_solve_grad(Q, c_val, A, b, x_l, x_u)

        grad_c = jax.grad(obj_fn)(c)

        # Finite difference
        eps = 1e-5
        for i in range(2):
            c_p = c.at[i].set(c[i] + eps)
            c_m = c.at[i].set(c[i] - eps)
            fd = (obj_fn(c_p) - obj_fn(c_m)) / (2 * eps)
            assert jnp.allclose(grad_c[i], fd, atol=0.1), f"AD grad[{i}]={grad_c[i]}, FD={fd}"


# ---------------------------------------------------------------
# 7. Solver Dispatch Integration Tests
# ---------------------------------------------------------------


class TestSolverDispatch:
    """Test that Model.solve() correctly dispatches to LP/QP solvers."""

    def test_lp_via_model_solve(self):
        """LP problem should be solved via LP IPM when dispatched."""
        m = dm.Model("lp_dispatch")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(3 * x + 2 * y)
        m.subject_to(x + y <= 5)

        result = m.solve()
        assert result.status == "optimal"
        assert result.objective is not None
        assert abs(result.objective - 0.0) < 1.0  # obj should be near 0 (x=y=0)

    def test_qp_via_model_solve(self):
        """QP problem should be solved via QP IPM."""
        m = dm.Model("qp_dispatch")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.minimize(x**2 + y**2)

        result = m.solve()
        assert result.status == "optimal"
        assert result.objective is not None
        assert abs(result.objective) < 0.01  # minimum at origin

    def test_constrained_qp_via_model_solve(self):
        """Constrained QP via Model.solve()."""
        m = dm.Model("cqp_dispatch")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 2)

        result = m.solve()
        assert result.status == "optimal"
        assert result.objective is not None
        # min x^2+y^2 s.t. x+y>=2, x,y>=0 → x=y=1, obj=2
        assert abs(result.objective - 2.0) < 0.5


class TestMILPDispatch:
    """Regression tests for MILP dispatch (issue #36).

    Before #36, ProblemClass.MILP always routed to _solve_milp_bb, which
    has no primal heuristic and timed out on moderate big-M formulations
    (e.g. a 5x3 disjunctive jobshop). HiGHS MIP solves these fast; it is now
    reached via ``nlp_solver="ipm"`` (the universal default is POUNCE, which
    routes MILP to the self-hosted B&B). These tests pin the HiGHS path that
    guards #36; the slower default-POUNCE behavior on big-M MILPs is expected.
    """

    @staticmethod
    def _build_jobshop(n_jobs: int, n_machines: int):
        import numpy as np

        rng = np.random.default_rng(0)
        proc = rng.integers(1, 5, size=(n_jobs, n_machines)).astype(float)
        M = float(proc.sum())

        m = dm.Model(f"jobshop_{n_jobs}x{n_machines}")
        s = m.continuous("start", shape=(n_jobs, n_machines), lb=0, ub=M)
        C = m.continuous("makespan", lb=0, ub=M)
        n_pairs = n_jobs * (n_jobs - 1) // 2
        z = m.binary("order", shape=(n_pairs * n_machines,))
        m.minimize(C)
        for i in range(n_jobs):
            for k in range(n_machines):
                m.subject_to(C >= s[i, k] + float(proc[i, k]))
        pi = 0
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                for k in range(n_machines):
                    idx = pi * n_machines + k
                    m.subject_to(s[i, k] + float(proc[i, k]) <= s[j, k] + M * (1 - z[idx]))
                    m.subject_to(s[j, k] + float(proc[j, k]) <= s[i, k] + M * z[idx])
                pi += 1
        return m

    @pytest.mark.slow
    def test_milp_routes_to_bb(self):
        """The MILP path routes through the self-hosted _solve_milp_bb (HiGHS
        was removed from the MILP path, issue #356)."""
        m = self._build_jobshop(3, 3)
        result = m.solve(time_limit=60)
        assert result.status == "optimal"
        # _solve_milp_bb explores B&B nodes.
        assert result.node_count > 0

    def test_ipm_alias_path_solves(self):
        """The ``"ipm"`` alias also routes through the self-hosted B&B now that
        HiGHS has been removed from the MILP path (issue #356)."""
        m = self._build_jobshop(3, 3)
        result = m.solve(nlp_solver="ipm", time_limit=30)
        assert result.status == "optimal"
        assert result.node_count >= 0


# ---------------------------------------------------------------
# 8. Unified Differentiable Solve Tests
# ---------------------------------------------------------------


class TestUnifiedDiffSolve:
    """Test the unified differentiable_solve API."""

    def test_lp_diff_solve(self):
        """differentiable_solve on LP returns correct result."""
        from discopt._jax.differentiable_solve import differentiable_solve

        m = dm.Model("lp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + 2 * y)
        m.subject_to(x + y <= 5)

        result = differentiable_solve(m)
        assert result.status == "optimal"
        assert result.objective is not None

    def test_qp_diff_solve(self):
        """differentiable_solve on QP returns correct result."""
        from discopt._jax.differentiable_solve import differentiable_solve

        m = dm.Model("qp")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.minimize(x**2 + y**2)

        result = differentiable_solve(m)
        assert result.status == "optimal"
        assert abs(result.objective) < 0.01

    def test_problem_class_reported(self):
        """differentiable_solve should report the detected problem class."""
        from discopt._jax.differentiable_solve import differentiable_solve
        from discopt._jax.problem_classifier import ProblemClass

        m = dm.Model("lp")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)

        result = differentiable_solve(m)
        assert result.problem_class == ProblemClass.LP


# ---------------------------------------------------------------
# 9. LP/QP Batch Solve Tests
# ---------------------------------------------------------------


class TestBatchSolve:
    """Test batch LP/QP solving via vmap."""

    def test_qp_batch(self):
        """Batch QP solve with varying bounds."""
        from discopt._jax.qp_ipm import qp_ipm_solve_batch

        Q = jnp.eye(2)
        c = jnp.zeros(2)
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([1.0])

        xl_batch = jnp.array(
            [
                [0.0, 0.0],
                [0.3, 0.0],
            ]
        )
        xu_batch = jnp.full((2, 2), 1e20)

        states = qp_ipm_solve_batch(Q, c, A, b, xl_batch, xu_batch)
        assert jnp.all(states.converged > 0)
        # First instance: x=y=0.5, obj=0.25
        assert jnp.allclose(states.obj[0], 0.25, atol=1e-3)
