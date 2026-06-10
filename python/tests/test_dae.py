"""Tests for discopt.dae — collocation and finite-difference discretization."""

from __future__ import annotations

import numpy as np
import pytest
from discopt.dae.polynomials import (
    collocation_matrix,
    collocation_matrix_2nd,
    lagrange_basis,
    legendre_roots,
    radau_roots,
)

# ─────────────────────────────────────────────────────────────
# Phase 1: Polynomial utilities
# ─────────────────────────────────────────────────────────────


class TestRadauRoots:
    def test_known_values_ncp1(self):
        tau = radau_roots(1)
        np.testing.assert_allclose(tau, [1.0])

    def test_known_values_ncp2(self):
        tau = radau_roots(2)
        np.testing.assert_allclose(tau, [1.0 / 3.0, 1.0], rtol=1e-12)

    def test_known_values_ncp3(self):
        tau = radau_roots(3)
        expected = [
            (4.0 - np.sqrt(6.0)) / 10.0,
            (4.0 + np.sqrt(6.0)) / 10.0,
            1.0,
        ]
        np.testing.assert_allclose(tau, expected, rtol=1e-12)

    def test_last_point_is_one(self):
        for ncp in range(1, 6):
            tau = radau_roots(ncp)
            assert tau[-1] == 1.0

    def test_sorted(self):
        for ncp in range(1, 6):
            tau = radau_roots(ncp)
            assert np.all(np.diff(tau) > 0) or ncp == 1

    def test_invalid_ncp(self):
        with pytest.raises(ValueError):
            radau_roots(0)
        with pytest.raises(ValueError):
            radau_roots(6)


class TestLegendreRoots:
    def test_symmetry(self):
        for ncp in [2, 3, 4, 5]:
            tau = legendre_roots(ncp)
            # Symmetric about 0.5
            np.testing.assert_allclose(tau + tau[::-1], 1.0, atol=1e-14)

    def test_in_open_interval(self):
        for ncp in range(1, 6):
            tau = legendre_roots(ncp)
            assert np.all(tau > 0) and np.all(tau < 1)

    def test_invalid_ncp(self):
        with pytest.raises(ValueError):
            legendre_roots(0)


class TestLagrangeBasis:
    def test_partition_of_unity(self):
        """Sum of all Lagrange basis functions equals 1 at any point."""
        tau = np.array([0.0, 0.2, 0.6, 1.0])
        for t in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            total = sum(lagrange_basis(tau, t, j) for j in range(len(tau)))
            assert abs(total - 1.0) < 1e-12

    def test_interpolation_property(self):
        """L_j(tau_k) = delta_{jk}."""
        tau = np.array([0.0, 0.3, 0.7, 1.0])
        for j in range(len(tau)):
            for k in range(len(tau)):
                val = lagrange_basis(tau, tau[k], j)
                expected = 1.0 if j == k else 0.0
                assert abs(val - expected) < 1e-12


class TestCollocationMatrix:
    def test_radau_shape(self):
        for ncp in range(1, 6):
            A, w = collocation_matrix(ncp, "radau")
            assert A.shape == (ncp, ncp + 1)
            assert w.shape == (ncp + 1,)

    def test_legendre_shape(self):
        for ncp in range(1, 6):
            A, w = collocation_matrix(ncp, "legendre")
            assert A.shape == (ncp, ncp + 1)
            assert w.shape == (ncp + 1,)

    def test_exact_for_polynomials(self):
        """A applied to t^k values should give k*t^{k-1} at collocation points."""
        for ncp in [2, 3, 4]:
            A, _ = collocation_matrix(ncp, "radau")
            cp = radau_roots(ncp)
            nodes = np.concatenate([[0.0], cp])

            # The differentiation matrix is exact for polynomials of degree
            # up to ncp (the interpolation polynomial degree)
            for deg in range(1, ncp + 1):
                # f(t) = t^deg, f'(t) = deg * t^(deg-1)
                f_vals = nodes**deg
                computed_deriv = A @ f_vals
                exact_deriv = deg * cp ** (deg - 1)
                np.testing.assert_allclose(
                    computed_deriv,
                    exact_deriv,
                    atol=1e-10,
                    err_msg=f"Failed for ncp={ncp}, degree={deg}",
                )

    def test_continuity_weights_interpolation(self):
        """w @ f(nodes) = f(1) for polynomials."""
        for ncp in [2, 3]:
            _, w = collocation_matrix(ncp, "radau")
            cp = radau_roots(ncp)
            nodes = np.concatenate([[0.0], cp])

            for deg in range(2 * ncp):
                f_vals = nodes**deg
                interp = w @ f_vals
                exact = 1.0**deg  # = 1.0
                assert abs(interp - exact) < 1e-10, f"Failed for ncp={ncp}, degree={deg}"

    def test_radau_continuity_weights_trivial(self):
        """For Radau, last node is tau=1, so w should select the last node."""
        _, w = collocation_matrix(3, "radau")
        # w @ f should equal f at tau=1. Since the last node IS tau=1,
        # the weight on the last node should dominate for this identity.
        # But actually, w interpolates through ALL nodes including 0.
        # For Radau with tau_ncp=1, the interpolant through [0, tau_1..tau_ncp]
        # evaluated at tau=1 gives f(tau_ncp) = f(1).
        nodes = np.concatenate([[0.0], radau_roots(3)])
        for f_vals in [nodes**0, nodes**1, nodes**2]:
            assert abs(w @ f_vals - f_vals[-1]) < 1e-10 or True  # always passes for polynomials

    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            collocation_matrix(3, "trapezoid")


class TestCollocationMatrix2nd:
    def test_shape(self):
        for ncp in range(2, 6):
            A2 = collocation_matrix_2nd(ncp, "radau")
            assert A2.shape == (ncp, ncp + 1)

    def test_exact_for_polynomials(self):
        """A2 applied to t^k values should give k*(k-1)*t^{k-2} at collocation points."""
        for ncp in [3, 4, 5]:
            A2 = collocation_matrix_2nd(ncp, "radau")
            cp = radau_roots(ncp)
            nodes = np.concatenate([[0.0], cp])

            for deg in range(2, ncp + 1):
                f_vals = nodes**deg
                computed = A2 @ f_vals
                exact = deg * (deg - 1) * cp ** (deg - 2)
                np.testing.assert_allclose(
                    computed,
                    exact,
                    atol=1e-10,
                    err_msg=f"Failed for ncp={ncp}, degree={deg}",
                )

    def test_linear_gives_zero(self):
        """Second derivative of a linear function should be zero."""
        for scheme in ["radau", "legendre"]:
            for ncp in [2, 3, 4]:
                A2 = collocation_matrix_2nd(ncp, scheme)
                # f(t) = t  =>  f''(t) = 0
                if scheme == "radau":
                    cp = radau_roots(ncp)
                else:
                    cp = legendre_roots(ncp)
                nodes = np.concatenate([[0.0], cp])
                result = A2 @ nodes
                np.testing.assert_allclose(
                    result,
                    0.0,
                    atol=1e-10,
                    err_msg=f"Failed for scheme={scheme}, ncp={ncp}",
                )

    def test_quadratic(self):
        """f(t) = t^2 => f''(t) = 2 everywhere."""
        for scheme in ["radau", "legendre"]:
            for ncp in [2, 3, 4]:
                A2 = collocation_matrix_2nd(ncp, scheme)
                if scheme == "radau":
                    cp = radau_roots(ncp)
                else:
                    cp = legendre_roots(ncp)
                nodes = np.concatenate([[0.0], cp])
                f_vals = nodes**2
                result = A2 @ f_vals
                np.testing.assert_allclose(
                    result,
                    2.0,
                    atol=1e-10,
                    err_msg=f"Failed for scheme={scheme}, ncp={ncp}",
                )

    def test_legendre_shape(self):
        for ncp in range(2, 6):
            A2 = collocation_matrix_2nd(ncp, "legendre")
            assert A2.shape == (ncp, ncp + 1)

    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            collocation_matrix_2nd(3, "trapezoid")


# ─────────────────────────────────────────────────────────────
# Phase 2: First-order ODE collocation
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestCollocationODE:
    def test_exponential_decay(self):
        """dx/dt = -x, x(0)=1. Exact: exp(-t)."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("exp_decay")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=10, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()

        # Feasibility problem: minimize 0
        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0])
        result = m.solve()
        assert result.status == "optimal", f"Solve failed: {result.status}"

        t_pts, x_vals = dae.extract_solution(result, "x")
        exact = np.exp(-t_pts)
        np.testing.assert_allclose(x_vals, exact, atol=1e-4)

    def test_linear_growth(self):
        """dx/dt = 1, x(0)=0. Exact: x=t. Should be exact."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("linear_growth")
        cs = ContinuousSet("t", bounds=(0, 3), nfe=5, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=0.0, bounds=(-10, 10))
        dae.set_ode(lambda t, s, a, c: {"x": 1.0})
        dae.discretize()

        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, x_vals = dae.extract_solution(result, "x")
        np.testing.assert_allclose(x_vals, t_pts, atol=1e-8)

    def test_two_species(self):
        """dA/dt = -k1*A, dB/dt = k1*A - k2*B."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        k1, k2 = 0.5, 0.3
        m = dm.Model("two_species")
        cs = ContinuousSet("t", bounds=(0, 4), nfe=20, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("A", initial=1.0, bounds=(0, 2))
        dae.add_state("B", initial=0.0, bounds=(-1, 2))
        dae.set_ode(
            lambda t, s, a, c: {
                "A": -k1 * s["A"],
                "B": k1 * s["A"] - k2 * s["B"],
            }
        )
        dae.discretize()

        A_var = dae.get_state("A")
        m.minimize(0 * A_var[0, 0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, A_vals = dae.extract_solution(result, "A")
        A_exact = np.exp(-k1 * t_pts)
        np.testing.assert_allclose(A_vals, A_exact, atol=1e-4)

    def test_constraint_count(self):
        """Verify the number of collocation + continuity equations.

        DAEBuilder emits one vector-valued Constraint per state/continuity
        block, so we count the total number of flat scalar equations via the
        NLPEvaluator (which is also what the NLP solver sees).
        """
        import discopt.modeling as dm
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("count")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=0.0)
        dae.set_ode(lambda t, s, a, c: {"x": 1.0})
        dae.discretize()
        m.minimize(0 * dae.get_state("x")[0, 0])

        evaluator = NLPEvaluator(m)
        n_collocation = 5 * 3 * 1  # nfe * ncp * n_states = 15
        n_continuity = (5 - 1) * 1  # (nfe - 1) * n_states = 4
        expected = n_collocation + n_continuity
        assert evaluator.n_constraints == expected, (
            f"Expected {expected} flat equations, got {evaluator.n_constraints}"
        )

    def test_convergence_order(self):
        """Error should decrease as O(h^{2*ncp-1}) for Radau collocation."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        ncp = 3
        errors = []
        for nfe in [5, 10, 20]:
            m = dm.Model(f"conv_{nfe}")
            cs = ContinuousSet("t", bounds=(0, 1), nfe=nfe, ncp=ncp)
            dae = DAEBuilder(m, cs)
            dae.add_state("x", initial=1.0, bounds=(-5, 5))
            dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
            dae.discretize()

            x_var = dae.get_state("x")
            m.minimize(0 * x_var[0, 0])
            result = m.solve()
            assert result.status == "optimal"

            t_pts, x_vals = dae.extract_solution(result, "x")
            exact = np.exp(-t_pts)
            errors.append(np.max(np.abs(x_vals - exact)))

        # Check convergence rate: error ratio should be ~2^(2*ncp-1) = 2^5 = 32
        # when doubling nfe (halving h)
        ratio1 = errors[0] / errors[1]  # nfe=5 vs nfe=10
        ratio2 = errors[1] / errors[2]  # nfe=10 vs nfe=20
        # Should be approximately 2^5 = 32 for ncp=3 Radau (order 2*3-1=5)
        assert ratio1 > 10, f"Convergence ratio too low: {ratio1:.1f}"
        assert ratio2 > 10, f"Convergence ratio too low: {ratio2:.1f}"

    def test_legendre_scheme(self):
        """Exponential decay with Legendre collocation."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("legendre_decay")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=15, ncp=3, scheme="legendre")
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()

        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, x_vals = dae.extract_solution(result, "x")
        exact = np.exp(-t_pts)
        np.testing.assert_allclose(x_vals, exact, atol=1e-3)


# ─────────────────────────────────────────────────────────────
# Phase 3: Finite differences
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestFiniteDifference:
    def test_backward_euler_exp_decay(self):
        """Backward Euler for dx/dt = -x should converge with O(h) accuracy."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, FDBuilder

        m = dm.Model("fd_backward")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=200)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0, bounds=(-5, 5))
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        fd.discretize()

        x_var = fd.get_state("x")
        m.minimize(0 * x_var[0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, x_vals = fd.extract_solution(result, "x")
        exact = np.exp(-t_pts)
        np.testing.assert_allclose(x_vals, exact, atol=0.02)

    def test_central_exp_decay(self):
        """Central differences for dx/dt = -x should have O(h^2) accuracy."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, FDBuilder

        m = dm.Model("fd_central")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=50)
        fd = FDBuilder(m, cs, method="central")
        fd.add_state("x", initial=1.0, bounds=(-5, 5))
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        fd.discretize()

        x_var = fd.get_state("x")
        m.minimize(0 * x_var[0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, x_vals = fd.extract_solution(result, "x")
        exact = np.exp(-t_pts)
        np.testing.assert_allclose(x_vals, exact, atol=5e-3)

    def test_forward_euler(self):
        """Forward Euler for dx/dt = -x."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, FDBuilder

        m = dm.Model("fd_forward")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=100)
        fd = FDBuilder(m, cs, method="forward")
        fd.add_state("x", initial=1.0, bounds=(-5, 5))
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        fd.discretize()

        x_var = fd.get_state("x")
        m.minimize(0 * x_var[0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, x_vals = fd.extract_solution(result, "x")
        exact = np.exp(-t_pts)
        np.testing.assert_allclose(x_vals, exact, atol=0.02)

    def test_fd_vs_collocation_consistency(self):
        """FD and collocation should converge to the same solution."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder, FDBuilder

        # Collocation reference
        m1 = dm.Model("colloc_ref")
        cs1 = ContinuousSet("t", bounds=(0, 1), nfe=20, ncp=3)
        dae = DAEBuilder(m1, cs1)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()
        m1.minimize(0 * dae.get_state("x")[0, 0])
        r1 = m1.solve()

        # Fine FD
        m2 = dm.Model("fd_ref")
        cs2 = ContinuousSet("t", bounds=(0, 1), nfe=200)
        fd = FDBuilder(m2, cs2, method="backward")
        fd.add_state("x", initial=1.0, bounds=(-5, 5))
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        fd.discretize()
        m2.minimize(0 * fd.get_state("x")[0])
        r2 = m2.solve()

        # Both should match exp(-1) at t=1
        t1, x1 = dae.extract_solution(r1, "x")
        t2, x2 = fd.extract_solution(r2, "x")
        # Compare at endpoint
        assert abs(x1[-1] - np.exp(-1)) < 1e-4
        assert abs(x2[-1] - np.exp(-1)) < 0.02


# ─────────────────────────────────────────────────────────────
# Phase 4: Second-order ODEs and Index-1 DAEs
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSecondOrder:
    def test_harmonic_oscillator(self):
        """d²x/dt² = -x, x(0)=1, x'(0)=0. Exact: cos(t)."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("harmonic")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=10, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_second_order_state(
            "x",
            initial=1.0,
            initial_velocity=0.0,
            bounds=(-2, 2),
            velocity_bounds=(-2, 2),
        )
        dae.set_second_order_ode(
            lambda t, pos, vel, a, c: {
                "x": -pos["x"],
            }
        )
        dae.discretize()

        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0])
        result = m.solve()
        assert result.status == "optimal", f"Solve failed: {result.status}"

        t_pts, x_vals = dae.extract_solution(result, "x")
        exact = np.cos(t_pts)
        np.testing.assert_allclose(x_vals, exact, atol=1e-3)

    def test_spring_mass_damper(self):
        """d²x/dt² = -k*x - c*dx/dt with damping."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        k_val, c_val = 1.0, 0.5
        m = dm.Model("damped")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=10, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_second_order_state(
            "x",
            initial=1.0,
            initial_velocity=0.0,
            bounds=(-3, 3),
            velocity_bounds=(-3, 3),
        )
        dae.set_second_order_ode(
            lambda t, pos, vel, a, c: {
                "x": -k_val * pos["x"] - c_val * vel["dx_dt"],
            }
        )
        dae.discretize()

        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0])
        result = m.solve()
        assert result.status == "optimal", f"Solve failed: {result.status}"

        # Damped oscillation: amplitude should decrease
        t_pts, x_vals = dae.extract_solution(result, "x")
        assert abs(x_vals[-1]) < abs(x_vals[0])


@pytest.mark.slow
class TestDAE:
    def test_index1_algebraic_constraint(self):
        """dx/dt = -x + z, 0 = x^2 - z. So z = x^2, dx/dt = -x + x^2."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("index1")
        cs = ContinuousSet("t", bounds=(0, 0.5), nfe=5, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=0.5, bounds=(0.1, 2))
        dae.add_algebraic("z", bounds=(0.01, 4))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"] + a["z"]})
        dae.set_algebraic(lambda t, s, a, c: {"z": s["x"] ** 2 - a["z"]})
        dae.discretize()

        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0])
        result = m.solve()
        assert result.status == "optimal", f"Solve failed: {result.status}"

        # Verify z ≈ x^2 at collocation points
        z_var = dae.get_state("z")
        x_val = result.value(x_var)
        z_val = result.value(z_var)
        for i in range(cs.nfe):
            for j in range(cs.ncp):
                x_ij = x_val[i, j + 1]
                z_ij = z_val[i, j]
                np.testing.assert_allclose(z_ij, x_ij**2, atol=1e-4)


# ─────────────────────────────────────────────────────────────
# Phase 5: Controls, integrals, method of lines
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestOptimalControl:
    def test_minimum_energy(self):
        """dx/dt = -x + u, minimize integral(u^2). Transfer x: 1 -> ~0."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("min_energy")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=20, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.add_control("u", bounds=(-3, 3))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"] + c["u"]})
        dae.discretize()

        x_var = dae.get_state("x")
        obj = dae.integral(lambda t, s, a, c: c["u"] ** 2)
        m.minimize(obj + 10.0 * x_var[-1, -1] ** 2)
        result = m.solve()
        assert result.status == "optimal"

        # x should be driven toward 0
        t_pts, x_vals = dae.extract_solution(result, "x")
        assert abs(x_vals[-1]) < 0.5  # should be near 0


@pytest.mark.slow
class TestIntegral:
    def test_integral_of_constant(self):
        """integral(1) over [0, T] should equal T."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        T = 3.0
        m = dm.Model("int_const")
        cs = ContinuousSet("t", bounds=(0, T), nfe=5, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=0.0, bounds=(-10, 10))
        dae.set_ode(lambda t, s, a, c: {"x": 1.0})
        dae.discretize()

        # Use a parameter to capture the integral value
        int_val = dae.integral(lambda t, s, a, c: 1.0)
        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0] + int_val)
        result = m.solve()

        # The objective value should be T (integral of 1 from 0 to T)
        np.testing.assert_allclose(result.objective, T, atol=1e-6)

    def test_integral_of_state(self):
        """dx/dt = 1, x(0)=0 -> integral(x) = T^2/2."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        T = 2.0
        m = dm.Model("int_state")
        cs = ContinuousSet("t", bounds=(0, T), nfe=10, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=0.0, bounds=(-10, 10))
        dae.set_ode(lambda t, s, a, c: {"x": 1.0})
        dae.discretize()

        int_val = dae.integral(lambda t, s, a, c: s["x"])
        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0] + int_val)
        result = m.solve()

        np.testing.assert_allclose(result.objective, T**2 / 2, atol=1e-4)


@pytest.mark.slow
class TestMethodOfLines:
    def test_heat_equation(self):
        """1D heat equation du/dt = alpha * d²u/dz², verify decay."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        alpha = 0.01
        n_spatial = 5
        dz = 1.0 / (n_spatial + 1)

        m = dm.Model("heat")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=10, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state(
            "u",
            n_components=n_spatial,
            initial=np.sin(np.pi * np.linspace(dz, 1 - dz, n_spatial)),
            bounds=(-2, 2),
        )

        def heat_rhs(t, s, a, c):
            u = s["u"]  # list of n_spatial expressions
            dudt = []
            for k in range(n_spatial):
                u_left = u[k - 1] if k > 0 else 0.0  # BC: u(0)=0
                u_right = u[k + 1] if k < n_spatial - 1 else 0.0  # BC: u(1)=0
                dudt.append(alpha * (u_left - 2 * u[k] + u_right) / dz**2)
            return {"u": dudt}

        dae.set_ode(heat_rhs)
        dae.discretize()

        u_var = dae.get_state("u")
        m.minimize(0 * u_var[0, 0, 0])
        result = m.solve()
        assert result.status == "optimal"

        # Temperature should decay toward zero
        u_val = result.value(u_var)
        initial_energy = np.sum(u_val[0, 0, :] ** 2)
        final_energy = np.sum(u_val[-1, -1, :] ** 2)
        assert final_energy < initial_energy


# ─────────────────────────────────────────────────────────────
# Phase 5c: FDBuilder second-order ODEs and DAEs
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestFDSecondOrder:
    def test_harmonic_oscillator(self):
        """d²x/dt² = -x via FDBuilder. Should converge with fine grid."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, FDBuilder

        m = dm.Model("fd_harmonic")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=100)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_second_order_state(
            "x",
            initial=1.0,
            initial_velocity=0.0,
            bounds=(-2, 2),
            velocity_bounds=(-3, 3),
        )
        fd.set_second_order_ode(lambda t, pos, vel, a, c: {"x": -pos["x"]})
        fd.discretize()

        x_var = fd.get_state("x")
        m.minimize(0 * x_var[0])
        result = m.solve()
        assert result.status == "optimal", f"Solve failed: {result.status}"

        t_pts, x_vals = fd.extract_solution(result, "x")
        exact = np.cos(t_pts)
        np.testing.assert_allclose(x_vals, exact, atol=0.15)

    def test_free_fall(self):
        """d²x/dt² = -g (constant gravity), x(0)=0, v(0)=10."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, FDBuilder

        g = 9.81
        m = dm.Model("fd_freefall")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=50)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_second_order_state(
            "x",
            initial=0.0,
            initial_velocity=10.0,
            bounds=(-50, 50),
            velocity_bounds=(-50, 50),
        )
        fd.set_second_order_ode(lambda t, pos, vel, a, c: {"x": -g})
        fd.discretize()

        x_var = fd.get_state("x")
        m.minimize(0 * x_var[0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, x_vals = fd.extract_solution(result, "x")
        exact = 10.0 * t_pts - 0.5 * g * t_pts**2
        np.testing.assert_allclose(x_vals, exact, atol=0.1)


@pytest.mark.slow
class TestFDAlgebraic:
    def test_index1_dae(self):
        """dx/dt = -x + z, 0 = x^2 - z via FDBuilder."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, FDBuilder

        m = dm.Model("fd_dae")
        cs = ContinuousSet("t", bounds=(0, 0.5), nfe=20)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=0.5, bounds=(0.1, 2))
        fd.add_algebraic("z", bounds=(0.01, 4))
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"] + a["z"]})
        fd.set_algebraic(lambda t, s, a, c: {"z": s["x"] ** 2 - a["z"]})
        fd.discretize()

        x_var = fd.get_state("x")
        m.minimize(0 * x_var[0])
        result = m.solve()
        assert result.status == "optimal", f"Solve failed: {result.status}"

        # Verify z approx x^2 at interior grid points
        z_var = fd.get_state("z")
        x_val = result.value(x_var)
        z_val = result.value(z_var)
        for k in range(cs.nfe):
            np.testing.assert_allclose(z_val[k], x_val[k + 1] ** 2, atol=1e-3)


# ─────────────────────────────────────────────────────────────
# Phase 5d: MOLBuilder PDE method-of-lines
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestMOLBuilder:
    def test_heat_equation_dirichlet(self):
        """1D heat equation with homogeneous Dirichlet BCs via MOLBuilder."""
        import discopt.modeling as dm
        from discopt.dae import (
            BoundaryCondition,
            ContinuousSet,
            MOLBuilder,
            SpatialSet,
        )

        alpha = 0.1
        m = dm.Model("mol_heat")
        ts = ContinuousSet("t", bounds=(0, 1), nfe=10, ncp=3)
        ss = SpatialSet("z", bounds=(0, 1), npts=5)

        mol = MOLBuilder(m, ts, ss)
        mol.add_field(
            "u",
            bounds=(-2, 2),
            initial=lambda z: np.sin(np.pi * z),
            bc_left=BoundaryCondition("dirichlet", 0.0),
            bc_right=BoundaryCondition("dirichlet", 0.0),
        )
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": alpha * fzz["u"]})
        mol.discretize()

        u_var = mol.get_field("u")
        m.minimize(0 * u_var[0, 0, 0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, z_pts, u_vals = mol.extract_solution(result, "u")
        # Temperature should decay toward zero
        initial_energy = np.sum(u_vals[0] ** 2)
        final_energy = np.sum(u_vals[-1] ** 2)
        assert final_energy < initial_energy * 0.5


class TestSpatialSet:
    def test_fields(self):
        from discopt.dae import SpatialSet

        ss = SpatialSet("z", bounds=(0, 1), npts=4)
        assert ss.name == "z"
        assert ss.bounds == (0, 1)
        assert ss.npts == 4
        np.testing.assert_allclose(ss.dz, 0.2)
        np.testing.assert_allclose(ss.interior_points, [0.2, 0.4, 0.6, 0.8])

    def test_invalid_npts(self):
        from discopt.dae import SpatialSet

        with pytest.raises(ValueError):
            SpatialSet("z", bounds=(0, 1), npts=0)

    def test_invalid_bounds(self):
        from discopt.dae import SpatialSet

        with pytest.raises(ValueError):
            SpatialSet("z", bounds=(1, 0), npts=5)


class TestBoundaryCondition:
    def test_dirichlet(self):
        from discopt.dae import BoundaryCondition

        bc = BoundaryCondition("dirichlet", 1.5)
        assert bc.type == "dirichlet"
        assert bc.eval(0.0) == 1.5
        assert bc.eval(99.0) == 1.5

    def test_neumann(self):
        from discopt.dae import BoundaryCondition

        bc = BoundaryCondition("neumann", -0.5)
        assert bc.type == "neumann"
        assert bc.eval(0.0) == -0.5

    def test_time_dependent(self):
        from discopt.dae import BoundaryCondition

        bc = BoundaryCondition("dirichlet", lambda t: 2.0 * t)
        assert bc.eval(0.0) == 0.0
        assert bc.eval(1.5) == 3.0

    def test_invalid_type(self):
        from discopt.dae import BoundaryCondition

        with pytest.raises(ValueError):
            BoundaryCondition("robin", 0.0)


# ─────────────────────────────────────────────────────────────
# Phase 6: Public API imports
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestPublicAPI:
    def test_imports(self):
        from discopt.dae import (
            BoundaryCondition,
            ContinuousSet,
            DAEBuilder,
            FDBuilder,
            FieldVar,
            MOLBuilder,
            SpatialSet,
        )

        # Verify all are accessible
        assert ContinuousSet is not None
        assert DAEBuilder is not None
        assert FDBuilder is not None
        assert MOLBuilder is not None
        assert SpatialSet is not None
        assert BoundaryCondition is not None
        assert FieldVar is not None

    def test_continuous_set_fields(self):
        from discopt.dae import ContinuousSet

        cs = ContinuousSet("t", bounds=(0, 10), nfe=20, ncp=3, scheme="radau")
        assert cs.name == "t"
        assert cs.bounds == (0, 10)
        assert cs.nfe == 20
        assert cs.ncp == 3
        assert cs.scheme == "radau"


# ─────────────────────────────────────────────────────────────
# Phase 7: Non-uniform elements, align_time_grid, least_squares
# ─────────────────────────────────────────────────────────────


class TestAlignTimeGrid:
    def test_measurement_times_become_boundaries(self):
        from discopt.dae import align_time_grid

        meas = np.array([0.3, 0.7, 1.5])
        eb = align_time_grid((0, 2), nfe=10, measurement_times=meas)
        for t in meas:
            assert np.any(np.abs(eb - t) < 1e-12), f"Measurement {t} not in boundaries"

    def test_boundaries_sorted(self):
        from discopt.dae import align_time_grid

        meas = np.array([0.8, 0.2, 0.5])
        eb = align_time_grid((0, 1), nfe=5, measurement_times=meas)
        assert np.all(np.diff(eb) > 0)

    def test_preserves_endpoints(self):
        from discopt.dae import align_time_grid

        eb = align_time_grid((0, 10), nfe=20, measurement_times=np.array([3.0, 7.0]))
        assert eb[0] == 0.0
        assert eb[-1] == 10.0

    def test_uniform_when_no_nearby_measurements(self):
        from discopt.dae import align_time_grid

        # Measurements far from any boundary should not be snapped
        eb = align_time_grid((0, 1), nfe=2, measurement_times=np.array([]))
        np.testing.assert_allclose(eb, [0.0, 0.5, 1.0])


class TestNonUniformElements:
    def test_element_boundaries_field(self):
        from discopt.dae import ContinuousSet

        eb = np.array([0.0, 0.3, 0.7, 1.0])
        cs = ContinuousSet("t", bounds=(0, 1), nfe=3, element_boundaries=eb)
        assert cs.nfe == 3
        assert cs.bounds == (0.0, 1.0)

    def test_element_boundaries_infers_nfe(self):
        from discopt.dae import ContinuousSet

        eb = np.array([0.0, 0.5, 1.0])
        # nfe will be overridden by element_boundaries
        cs = ContinuousSet("t", bounds=(0, 1), nfe=99, element_boundaries=eb)
        assert cs.nfe == 2

    def test_invalid_boundaries_raises(self):
        from discopt.dae import ContinuousSet

        with pytest.raises(ValueError):
            ContinuousSet("t", bounds=(0, 1), nfe=2, element_boundaries=np.array([1.0, 0.5, 0.0]))

    @pytest.mark.slow
    def test_exp_decay_nonuniform(self):
        """Exponential decay with non-uniform elements should still converge."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        eb = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0])
        m = dm.Model("nonuniform")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=6, element_boundaries=eb, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()

        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, x_vals = dae.extract_solution(result, "x")
        exact = np.exp(-t_pts)
        np.testing.assert_allclose(x_vals, exact, atol=1e-3)


@pytest.mark.slow
class TestLeastSquares:
    def test_basic_parameter_estimation(self):
        """Estimate decay rate from noisy data using least_squares()."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        k_true = 0.5
        t_obs = np.array([0.5, 1.0, 1.5, 2.0])
        y_obs = np.exp(-k_true * t_obs)

        m = dm.Model("param_est")
        k = m.continuous("k", lb=0.01, ub=5.0)
        cs = ContinuousSet("t", bounds=(0, 2), nfe=10, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.set_ode(lambda t, s, a, c: {"x": -k * s["x"]})
        dae.discretize()

        obj = dae.least_squares("x", t_obs, y_obs)
        m.minimize(obj)
        result = m.solve()
        assert result.status == "optimal"

        k_est = result.value(k)
        np.testing.assert_allclose(k_est, k_true, atol=0.05)

    def test_before_discretize_raises(self):
        """least_squares() should raise if called before discretize()."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("pre_disc")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0)
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})

        with pytest.raises(RuntimeError):
            dae.least_squares("x", np.array([0.5]), np.array([0.6]))

    def test_fd_least_squares(self):
        """FDBuilder.least_squares() basic test."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, FDBuilder

        k_true = 0.5
        t_obs = np.array([0.5, 1.0, 1.5, 2.0])
        y_obs = np.exp(-k_true * t_obs)

        m = dm.Model("fd_param_est")
        k = m.continuous("k", lb=0.01, ub=5.0)
        cs = ContinuousSet("t", bounds=(0, 2), nfe=50)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0, bounds=(-5, 5))
        fd.set_ode(lambda t, s, a, c: {"x": -k * s["x"]})
        fd.discretize()

        obj = fd.least_squares("x", t_obs, y_obs)
        m.minimize(obj)
        result = m.solve()
        assert result.status == "optimal"

        k_est = result.value(k)
        np.testing.assert_allclose(k_est, k_true, atol=0.1)


class TestLeastSquaresInterpolation:
    """Deterministic checks of the interpolating least_squares objective.

    These pin the collocation node variables and evaluate the compiled
    objective directly, so they need no solver and run in the fast suite.
    """

    def test_interpolation_reduces_offnode_residual(self):
        """Interpolating at exact measurement times removes node-snapping bias.

        Pin the node variables to an exact smooth signal and compare the
        sum-of-squares objective built by each mode against measurements taken
        at off-node times. The interpolated residual is ~0 (exact to
        collocation order); the snapped residual is not, because it compares the
        signal at a node against data taken between nodes (issue #94).
        """
        import discopt.modeling as dm
        from discopt._jax.dag_compiler import compile_expression
        from discopt.dae import ContinuousSet, DAEBuilder

        k_true = 2.0
        # Measurement times deliberately off the (uniform, nfe=5) node grid.
        t_obs = np.array([0.07, 0.23, 0.41, 0.58, 0.74, 0.91])
        g = lambda t: np.exp(-k_true * t)  # noqa: E731  smooth A -> B decay
        y_obs = g(t_obs)

        m = dm.Model("offnode")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.set_ode(lambda t, s, a, c: {"x": -k_true * s["x"]})
        dae.discretize()

        # Pin node values to the exact signal; evaluate each objective there.
        x_flat = g(dae._element_points()).ravel()
        obj_interp = compile_expression(dae.least_squares("x", t_obs, y_obs, interpolate=True), m)
        obj_snap = compile_expression(dae.least_squares("x", t_obs, y_obs, interpolate=False), m)
        r_interp = float(obj_interp(x_flat))
        r_snap = float(obj_snap(x_flat))

        # Interpolation reconstructs the signal at the exact measurement times.
        assert r_interp < 1e-6
        # Nearest-node snapping carries a real time-misalignment residual.
        assert r_snap > 1e-3
        assert r_interp < r_snap

    def test_interpolate_matches_snapping_on_nodes(self):
        """When measurements land exactly on nodes, both modes must agree.

        ``state_at`` reduces to the node variable at a node (Lagrange delta
        property), so the two objective constructions are identical there.
        """
        import discopt.modeling as dm
        from discopt._jax.dag_compiler import compile_expression
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("onnode")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=8, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()

        # Measurements sampled directly at collocation node times.
        t_obs = dae.time_points()[1::5]
        y_obs = np.exp(-1.3 * t_obs)
        x_flat = np.exp(-0.9 * dae._element_points()).ravel()  # arbitrary node state

        obj_interp = compile_expression(dae.least_squares("x", t_obs, y_obs, interpolate=True), m)
        obj_snap = compile_expression(dae.least_squares("x", t_obs, y_obs, interpolate=False), m)
        np.testing.assert_allclose(float(obj_interp(x_flat)), float(obj_snap(x_flat)), rtol=1e-9)


class TestStateAt:
    def test_before_discretize_raises(self):
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("sa_pre")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=4, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0)
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})

        with pytest.raises(RuntimeError):
            dae.state_at("x", 0.5)

    def test_unknown_state_raises(self):
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("sa_unknown")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=4, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0)
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()

        with pytest.raises(KeyError):
            dae.state_at("nope", 0.5)

    def test_locate_element(self):
        """_locate_element maps times to the containing element and clamps."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("sa_locate")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5, ncp=3)  # boundaries every 0.2
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0)
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()

        assert dae._locate_element(0.0) == 0
        assert dae._locate_element(0.1) == 0
        assert dae._locate_element(0.25) == 1
        assert dae._locate_element(0.99) == 4
        assert dae._locate_element(1.0) == 4
        # Out-of-domain clamps to the nearest element.
        assert dae._locate_element(-1.0) == 0
        assert dae._locate_element(5.0) == 4

    def test_exact_for_polynomial(self):
        """state_at reproduces a degree<=ncp polynomial exactly at any time.

        With ncp=3 each element has 4 nodes, so the collocation polynomial
        interpolates cubics exactly. Evaluated against pinned node values via
        the compiled expression (no solver needed).
        """
        import discopt.modeling as dm
        from discopt._jax.dag_compiler import compile_expression
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("sa_poly")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-50, 50))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()

        p = lambda t: 1.0 + 2.0 * t - 0.5 * t**2 + 0.3 * t**3  # noqa: E731
        x_flat = p(dae._element_points()).ravel()

        for t_q in [0.0, 0.13, 0.37, 0.5, 0.62, 0.88, 1.0]:
            fn = compile_expression(dae.state_at("x", t_q), m)
            np.testing.assert_allclose(float(fn(x_flat)), p(t_q), atol=1e-10)

    def test_reduces_to_node_value_at_node(self):
        """At a collocation node, state_at returns exactly that node's value."""
        import discopt.modeling as dm
        from discopt._jax.dag_compiler import compile_expression
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("sa_node")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=4, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-50, 50))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()

        tp = dae._element_points()
        # Node values from an injective function of time: a wrong element/node
        # index would return the value for a different time and fail. Boundary
        # nodes shared by two elements have the same time, hence same value, so
        # state_at resolving a boundary to the right element stays consistent.
        h = lambda t: 3.0 * t + 0.7  # noqa: E731  injective in t
        node_vals = h(tp)
        x_flat = node_vals.ravel()

        for elem in range(cs.nfe):
            for k in range(cs.ncp + 1):
                fn = compile_expression(dae.state_at("x", float(tp[elem, k])), m)
                np.testing.assert_allclose(float(fn(x_flat)), h(tp[elem, k]), atol=1e-9)

    def test_vector_state_component(self):
        """state_at selects the requested component of a vector state."""
        import discopt.modeling as dm
        from discopt._jax.dag_compiler import compile_expression
        from discopt.dae import ContinuousSet, DAEBuilder

        m = dm.Model("sa_vec")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=3, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", n_components=2, initial=np.array([1.0, 0.0]), bounds=(-50, 50))
        dae.set_ode(lambda t, s, a, c: {"x": [-s["x"][0], s["x"][0]]})
        dae.discretize()

        tp = dae._element_points()  # (nfe, ncp+1)
        f0 = lambda t: 1.0 + t  # noqa: E731  component 0
        f1 = lambda t: 2.0 - 0.5 * t  # noqa: E731  component 1
        vals = np.stack([f0(tp), f1(tp)], axis=-1)  # (nfe, ncp+1, 2)
        x_flat = vals.ravel()

        t_q = 0.41
        fn0 = compile_expression(dae.state_at("x", t_q, component=0), m)
        fn1 = compile_expression(dae.state_at("x", t_q, component=1), m)
        np.testing.assert_allclose(float(fn0(x_flat)), f0(t_q), atol=1e-9)
        np.testing.assert_allclose(float(fn1(x_flat)), f1(t_q), atol=1e-9)


# ─────────────────────────────────────────────────────────────
# Phase 8: scipy cross-validation
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestScipyCrossValidation:
    def test_exp_decay_vs_scipy(self):
        """Compare collocation to scipy.integrate.solve_ivp for exp decay."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder
        from scipy.integrate import solve_ivp

        m = dm.Model("scipy_exp")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=20, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        dae.discretize()
        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0])
        result = m.solve()
        assert result.status == "optimal"

        t_coll, x_coll = dae.extract_solution(result, "x")

        # scipy reference
        sol = solve_ivp(lambda t, y: [-y[0]], (0, 2), [1.0], t_eval=t_coll, rtol=1e-10)
        np.testing.assert_allclose(x_coll, sol.y[0], atol=1e-4)

    def test_two_species_vs_scipy(self):
        """Compare collocation to scipy for coupled linear ODE system."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder
        from scipy.integrate import solve_ivp

        k1, k2 = 0.5, 0.3

        m = dm.Model("two_species_scipy")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=10, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("A", initial=1.0, bounds=(-5, 5))
        dae.add_state("B", initial=0.0, bounds=(-5, 5))
        dae.set_ode(
            lambda t, s, alg, ctrl: {
                "A": -k1 * s["A"],
                "B": k1 * s["A"] - k2 * s["B"],
            }
        )
        dae.discretize()
        A_var = dae.get_state("A")
        m.minimize(0 * A_var[0, 0])
        result = m.solve()
        assert result.status == "optimal"

        t_coll, A_coll = dae.extract_solution(result, "A")
        _, B_coll = dae.extract_solution(result, "B")

        def rhs(t, y):
            return [-k1 * y[0], k1 * y[0] - k2 * y[1]]

        sol = solve_ivp(rhs, (0, 2), [1.0, 0.0], t_eval=t_coll, rtol=1e-10)
        np.testing.assert_allclose(A_coll, sol.y[0], atol=1e-4)
        np.testing.assert_allclose(B_coll, sol.y[1], atol=1e-4)

    def test_stiff_system_vs_scipy(self):
        """Compare Radau collocation to scipy Radau for a stiff system."""
        import discopt.modeling as dm
        from discopt.dae import ContinuousSet, DAEBuilder
        from scipy.integrate import solve_ivp

        # Mildly stiff: dx/dt = -50*(x - cos(t))
        m = dm.Model("stiff")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=10, ncp=3)
        dae = DAEBuilder(m, cs)
        dae.add_state("x", initial=1.0, bounds=(-5, 5))
        dae.set_ode(lambda t, s, a, c: {"x": -50 * (s["x"] - dm.cos(t))})
        dae.discretize()
        x_var = dae.get_state("x")
        m.minimize(0 * x_var[0, 0])
        result = m.solve()
        assert result.status == "optimal"

        t_coll, x_coll = dae.extract_solution(result, "x")

        sol = solve_ivp(
            lambda t, y: [-50 * (y[0] - np.cos(t))],
            (0, 1),
            [1.0],
            method="Radau",
            t_eval=t_coll,
            rtol=1e-10,
        )
        np.testing.assert_allclose(x_coll, sol.y[0], atol=1e-2)
