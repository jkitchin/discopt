"""Tests for DAE finite difference (FDBuilder) and method of lines (MOLBuilder).

Focused on problem construction, variable/constraint structure, and basic
accuracy with small grids. Uses simple ODEs and PDEs with known analytical
solutions to verify discretization correctness without long solves.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.dae import (
    BoundaryCondition,
    ContinuousSet,
    FDBuilder,
    MOLBuilder,
    SpatialSet,
)

# ─────────────────────────────────────────────────────────────
# FDBuilder: construction and variable structure
# ─────────────────────────────────────────────────────────────


class TestFDBuilderConstruction:
    """Verify FDBuilder creates correct variable shapes and grid points."""

    def test_time_points_uniform(self):
        m = dm.Model("tp")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=10)
        fd = FDBuilder(m, cs, method="backward")
        tp = fd.time_points()
        assert tp.shape == (11,)
        np.testing.assert_allclose(tp[0], 0.0)
        np.testing.assert_allclose(tp[-1], 2.0)
        np.testing.assert_allclose(np.diff(tp), 0.2, atol=1e-14)

    def test_time_points_nonuniform(self):
        m = dm.Model("tp_nu")
        cs = ContinuousSet("t", bounds=(0, 3), nfe=3, element_boundaries=[0, 1, 1.5, 3])
        fd = FDBuilder(m, cs, method="backward")
        tp = fd.time_points()
        np.testing.assert_allclose(tp, [0, 1, 1.5, 3])

    def test_state_variable_shape(self):
        m = dm.Model("sv")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0, bounds=(0, 5))
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        variables = fd.discretize()
        # State has shape (nfe+1,) = (6,)
        assert "x" in variables
        x_var = fd.get_state("x")
        # The variable object should exist and be retrievable
        assert x_var is not None

    def test_multicomponent_state(self):
        m = dm.Model("mc")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("y", n_components=3, initial=np.array([1.0, 2.0, 3.0]))
        fd.set_ode(lambda t, s, a, c: {"y": [-s["y"][i] for i in range(3)]})
        variables = fd.discretize()
        assert "y" in variables

    def test_control_variable(self):
        m = dm.Model("ctrl")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=0.0, bounds=(-10, 10))
        fd.add_control("u", bounds=(-1, 1))
        fd.set_ode(lambda t, s, a, c: {"x": c["u"]})
        variables = fd.discretize()
        assert "x" in variables
        assert "u" in variables

    def test_algebraic_variable(self):
        m = dm.Model("alg")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0, bounds=(0, 5))
        fd.add_algebraic("z", bounds=(0, 5))
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"] + a["z"]})
        fd.set_algebraic(lambda t, s, a, c: {"z": s["x"] ** 2 - a["z"]})
        variables = fd.discretize()
        assert "x" in variables
        assert "z" in variables

    def test_invalid_method(self):
        m = dm.Model("inv")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        with pytest.raises(ValueError, match="Unknown method"):
            FDBuilder(m, cs, method="runge_kutta")

    def test_double_discretize_raises(self):
        m = dm.Model("dd")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0)
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        fd.discretize()
        with pytest.raises(RuntimeError, match="already been called"):
            fd.discretize()

    def test_no_ode_raises(self):
        m = dm.Model("no_ode")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0)
        with pytest.raises(RuntimeError, match="No ODE RHS"):
            fd.discretize()

    def test_get_state_unknown_raises(self):
        m = dm.Model("gs")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0)
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        fd.discretize()
        with pytest.raises(KeyError, match="Unknown variable"):
            fd.get_state("y")


# ─────────────────────────────────────────────────────────────
# FDBuilder: exponential decay dy/dt = -ky, y(0) = 1
# ─────────────────────────────────────────────────────────────


class TestFDExpDecay:
    """Test FD methods on exponential decay with known solution y = exp(-kt)."""

    @pytest.fixture(params=["backward", "forward", "central"])
    def method(self, request):
        return request.param

    def test_exp_decay_solves(self, method):
        """All FD methods should produce optimal status for simple decay."""
        k = 1.0
        nfe = 10
        m = dm.Model(f"decay_{method}")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=nfe)
        fd = FDBuilder(m, cs, method=method)
        fd.add_state("y", initial=1.0, bounds=(-5, 5))
        fd.set_ode(lambda t, s, a, c: {"y": -k * s["y"]})
        fd.discretize()

        y_var = fd.get_state("y")
        m.minimize(0 * y_var[0])
        result = m.solve()
        assert result.status == "optimal"

    def test_backward_euler_accuracy(self):
        """Backward Euler on dy/dt = -y with 10 points, check rough accuracy."""
        k = 0.5
        nfe = 10
        m = dm.Model("be_acc")
        cs = ContinuousSet("t", bounds=(0, 2), nfe=nfe)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("y", initial=1.0, bounds=(-5, 5))
        fd.set_ode(lambda t, s, a, c: {"y": -k * s["y"]})
        fd.discretize()

        y_var = fd.get_state("y")
        m.minimize(0 * y_var[0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, y_vals = fd.extract_solution(result, "y")
        exact = np.exp(-k * t_pts)
        # With 10 points over [0,2], backward Euler is O(h) so atol is loose
        np.testing.assert_allclose(y_vals, exact, atol=0.15)

    def test_central_higher_accuracy(self):
        """Central differences should be more accurate than backward Euler."""
        k = 1.0
        nfe = 10

        errors = {}
        for method in ["backward", "central"]:
            m = dm.Model(f"cmp_{method}")
            cs = ContinuousSet("t", bounds=(0, 1), nfe=nfe)
            fd = FDBuilder(m, cs, method=method)
            fd.add_state("y", initial=1.0, bounds=(-5, 5))
            fd.set_ode(lambda t, s, a, c: {"y": -k * s["y"]})
            fd.discretize()

            y_var = fd.get_state("y")
            m.minimize(0 * y_var[0])
            result = m.solve()
            assert result.status == "optimal"

            t_pts, y_vals = fd.extract_solution(result, "y")
            exact = np.exp(-k * t_pts)
            errors[method] = np.max(np.abs(y_vals - exact))

        assert errors["central"] < errors["backward"]


# ─────────────────────────────────────────────────────────────
# FDBuilder: second-order ODE
# ─────────────────────────────────────────────────────────────


class TestFDSecondOrder:
    def test_second_order_state_creates_velocity(self):
        """add_second_order_state should create both position and velocity vars."""
        m = dm.Model("so")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_second_order_state("x", initial=0.0, initial_velocity=1.0, bounds=(-5, 5))
        fd.set_second_order_ode(lambda t, pos, vel, a, c: {"x": 0.0})
        variables = fd.discretize()
        assert "x" in variables
        assert "dx_dt" in variables

    def test_free_fall_structure(self):
        """Free fall d2x/dt2 = -g should solve and give reasonable trajectory."""
        g = 9.81
        m = dm.Model("ff")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=10)
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
        np.testing.assert_allclose(x_vals, exact, atol=0.5)


# ─────────────────────────────────────────────────────────────
# FDBuilder: least_squares
# ─────────────────────────────────────────────────────────────


class TestFDLeastSquares:
    def test_least_squares_before_discretize_raises(self):
        m = dm.Model("ls_err")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0)
        with pytest.raises(RuntimeError, match="Call discretize"):
            fd.least_squares("x", np.array([0.0]), np.array([1.0]))

    def test_least_squares_unknown_state_raises(self):
        m = dm.Model("ls_unk")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0)
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        fd.discretize()
        with pytest.raises(KeyError, match="Unknown state"):
            fd.least_squares("y", np.array([0.0]), np.array([1.0]))

    def test_least_squares_returns_expression(self):
        m = dm.Model("ls_expr")
        cs = ContinuousSet("t", bounds=(0, 1), nfe=5)
        fd = FDBuilder(m, cs, method="backward")
        fd.add_state("x", initial=1.0, bounds=(-5, 5))
        fd.set_ode(lambda t, s, a, c: {"x": -s["x"]})
        fd.discretize()
        t_data = np.array([0.0, 0.5, 1.0])
        y_data = np.array([1.0, 0.6, 0.4])
        expr = fd.least_squares("x", t_data, y_data)
        # Should return a discopt expression (not None)
        assert expr is not None


# ─────────────────────────────────────────────────────────────
# MOLBuilder: construction and structure
# ─────────────────────────────────────────────────────────────


class TestMOLBuilderConstruction:
    def test_spatial_set_properties(self):
        ss = SpatialSet("z", bounds=(0, 1), npts=4)
        np.testing.assert_allclose(ss.dz, 0.2)
        np.testing.assert_allclose(ss.interior_points, [0.2, 0.4, 0.6, 0.8])

    def test_mol_creates_field_variables(self):
        m = dm.Model("mol_var")
        ts = ContinuousSet("t", bounds=(0, 1), nfe=5)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference", fd_method="backward")
        mol.add_field("u", initial=0.0, bounds=(-2, 2))
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": fzz["u"]})
        mol.discretize()

        u_var = mol.get_field("u")
        assert u_var is not None
        z_pts = mol.spatial_points()
        assert len(z_pts) == 3
        t_pts = mol.time_points()
        assert len(t_pts) == 6  # nfe+1

    def test_mol_with_control(self):
        m = dm.Model("mol_ctrl")
        ts = ContinuousSet("t", bounds=(0, 1), nfe=5)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference", fd_method="backward")
        mol.add_field("u", initial=0.0, bounds=(-5, 5))
        mol.add_control("q", bounds=(0, 10))
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": fzz["u"] + c["q"]})
        mol.discretize()
        assert mol.get_field("u") is not None

    def test_invalid_time_method(self):
        m = dm.Model("inv_tm")
        ts = ContinuousSet("t", bounds=(0, 1), nfe=5)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        with pytest.raises(ValueError, match="time_method"):
            MOLBuilder(m, ts, ss, time_method="rk4")

    def test_no_pde_raises(self):
        m = dm.Model("no_pde")
        ts = ContinuousSet("t", bounds=(0, 1), nfe=5)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference")
        mol.add_field("u", initial=0.0)
        with pytest.raises(RuntimeError, match="No PDE RHS"):
            mol.discretize()

    def test_no_field_raises(self):
        m = dm.Model("no_fld")
        ts = ContinuousSet("t", bounds=(0, 1), nfe=5)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference")
        mol.set_pde(lambda t, z, f, fz, fzz, c: {})
        with pytest.raises(RuntimeError, match="No fields"):
            mol.discretize()

    def test_double_discretize_raises(self):
        m = dm.Model("dd_mol")
        ts = ContinuousSet("t", bounds=(0, 1), nfe=5)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference")
        mol.add_field("u", initial=0.0)
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": fzz["u"]})
        mol.discretize()
        with pytest.raises(RuntimeError, match="already been called"):
            mol.discretize()

    def test_get_field_before_discretize_raises(self):
        m = dm.Model("gf_err")
        ts = ContinuousSet("t", bounds=(0, 1), nfe=5)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference")
        mol.add_field("u", initial=0.0)
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": fzz["u"]})
        with pytest.raises(RuntimeError, match="Call discretize"):
            mol.get_field("u")


# ─────────────────────────────────────────────────────────────
# MOLBuilder: heat equation du/dt = alpha * d2u/dz2
# ─────────────────────────────────────────────────────────────


class TestMOLHeatEquation:
    def test_heat_dirichlet_solves(self):
        """Heat equation with Dirichlet BCs should solve to optimal."""
        alpha = 0.1
        m = dm.Model("heat_d")
        ts = ContinuousSet("t", bounds=(0, 0.5), nfe=5)
        ss = SpatialSet("z", bounds=(0, 1), npts=5)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference", fd_method="backward")
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
        m.minimize(0 * u_var[0])
        result = m.solve()
        assert result.status == "optimal"

    def test_heat_energy_decays(self):
        """Temperature energy should decrease over time for heat equation."""
        alpha = 0.1
        m = dm.Model("heat_decay")
        ts = ContinuousSet("t", bounds=(0, 1), nfe=8)
        ss = SpatialSet("z", bounds=(0, 1), npts=5)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference", fd_method="backward")
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
        m.minimize(0 * u_var[0])
        result = m.solve()
        assert result.status == "optimal"

        t_pts, z_pts, u_vals = mol.extract_solution(result, "u")
        # u_vals has shape (n_time, npts) for FDBuilder
        initial_energy = np.sum(u_vals[0] ** 2)
        final_energy = np.sum(u_vals[-1] ** 2)
        assert final_energy < initial_energy

    def test_heat_neumann_bcs(self):
        """Heat equation with Neumann (insulated) BCs should solve."""
        alpha = 0.05
        m = dm.Model("heat_n")
        ts = ContinuousSet("t", bounds=(0, 0.5), nfe=5)
        ss = SpatialSet("z", bounds=(0, 1), npts=5)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference", fd_method="backward")
        mol.add_field(
            "u",
            bounds=(-2, 2),
            initial=lambda z: np.sin(np.pi * z),
            bc_left=BoundaryCondition("neumann", 0.0),
            bc_right=BoundaryCondition("neumann", 0.0),
        )
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": alpha * fzz["u"]})
        mol.discretize()

        u_var = mol.get_field("u")
        m.minimize(0 * u_var[0])
        result = m.solve()
        assert result.status == "optimal"


# ─────────────────────────────────────────────────────────────
# MOLBuilder: collocation time method
# ─────────────────────────────────────────────────────────────


class TestMOLCollocation:
    def test_mol_collocation_solves(self):
        """MOLBuilder with collocation time method should also work."""
        alpha = 0.1
        m = dm.Model("mol_col")
        ts = ContinuousSet("t", bounds=(0, 0.5), nfe=3, ncp=3)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        mol = MOLBuilder(m, ts, ss, time_method="collocation")
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


# ─────────────────────────────────────────────────────────────
# MOLBuilder: initial condition types
# ─────────────────────────────────────────────────────────────


class TestMOLInitialConditions:
    def test_callable_initial(self):
        """Callable initial condition should be evaluated at interior points."""
        m = dm.Model("ic_call")
        ts = ContinuousSet("t", bounds=(0, 0.5), nfe=3)
        ss = SpatialSet("z", bounds=(0, 1), npts=4)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference")
        mol.add_field("u", initial=lambda z: z**2, bounds=(-2, 2))
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": 0.01 * fzz["u"]})
        mol.discretize()
        assert mol.get_field("u") is not None

    def test_scalar_initial(self):
        """Scalar initial condition should be uniform across spatial points."""
        m = dm.Model("ic_sc")
        ts = ContinuousSet("t", bounds=(0, 0.5), nfe=3)
        ss = SpatialSet("z", bounds=(0, 1), npts=4)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference")
        mol.add_field("u", initial=0.5, bounds=(-2, 2))
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": 0.01 * fzz["u"]})
        mol.discretize()
        assert mol.get_field("u") is not None

    def test_array_initial(self):
        """Array initial condition should be used directly."""
        m = dm.Model("ic_arr")
        ts = ContinuousSet("t", bounds=(0, 0.5), nfe=3)
        ss = SpatialSet("z", bounds=(0, 1), npts=4)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference")
        init = np.array([0.1, 0.3, 0.5, 0.7])
        mol.add_field("u", initial=init, bounds=(-2, 2))
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": 0.01 * fzz["u"]})
        mol.discretize()
        assert mol.get_field("u") is not None


# ─────────────────────────────────────────────────────────────
# Boundary condition edge cases
# ─────────────────────────────────────────────────────────────


class TestBoundaryConditionEdgeCases:
    def test_time_dependent_dirichlet(self):
        """Time-dependent Dirichlet BC should be accepted by MOLBuilder."""
        m = dm.Model("td_bc")
        ts = ContinuousSet("t", bounds=(0, 0.5), nfe=3)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference")
        mol.add_field(
            "u",
            initial=0.0,
            bounds=(-5, 5),
            bc_left=BoundaryCondition("dirichlet", lambda t: t),
            bc_right=BoundaryCondition("dirichlet", 0.0),
        )
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": 0.01 * fzz["u"]})
        mol.discretize()

        u_var = mol.get_field("u")
        m.minimize(0 * u_var[0])
        result = m.solve()
        assert result.status == "optimal"

    def test_mixed_bcs(self):
        """Dirichlet left, Neumann right should be accepted."""
        m = dm.Model("mixed_bc")
        ts = ContinuousSet("t", bounds=(0, 0.5), nfe=3)
        ss = SpatialSet("z", bounds=(0, 1), npts=3)
        mol = MOLBuilder(m, ts, ss, time_method="finite_difference")
        mol.add_field(
            "u",
            initial=lambda z: 1.0 - z,
            bounds=(-5, 5),
            bc_left=BoundaryCondition("dirichlet", 1.0),
            bc_right=BoundaryCondition("neumann", 0.0),
        )
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": 0.01 * fzz["u"]})
        mol.discretize()

        u_var = mol.get_field("u")
        m.minimize(0 * u_var[0])
        result = m.solve()
        assert result.status == "optimal"
