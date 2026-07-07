"""Regression tests for DAE correctness fixes C1, C2, C3.

- **C1** (`mol.py`): the left Neumann boundary reconstruction had a sign flip, so
  any nonzero-flux left BC gave a wrong PDE solution. Tests are invisible to the
  bug if the flux is 0 (as the pre-existing suite used), so these use nonzero flux.
- **C2** (`collocation.py`): `integral()` under-integrated by exactly 2x for Radau
  `ncp=1` (dropped node-0 weight). The pre-existing suite only exercised `ncp=3`.
- **C3** (`collocation.py`, `finite_difference.py`): two silent no-dynamics paths —
  a second-order state with no acceleration RHS, and an RHS dict key typo — both
  built an under-constrained model with no error.

All fast (direct calls / tiny LP solves), so marked ``smoke``. Each fails on the
pre-fix code.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.dae import (
    BoundaryCondition,
    ContinuousSet,
    DAEBuilder,
    FDBuilder,
    MOLBuilder,
    SpatialSet,
    radau_roots,
)

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------- C1
def _mol_linear_field():
    m = dm.Model("c1")
    ts = ContinuousSet("t", bounds=(0, 1), nfe=2, ncp=2)
    ss = SpatialSet("z", bounds=(0, 1), npts=5)
    mol = MOLBuilder(m, ts, ss)
    # u(z) = z on [0,1]: du/dz = 1. Outward-normal flux is -1 at the left, +1 at
    # the right. The reconstructed boundary values must be the true 0.0 and 1.0.
    mol.add_field(
        "u",
        bounds=(-5, 5),
        initial=lambda z: z,
        bc_left=BoundaryCondition("neumann", -1.0),
        bc_right=BoundaryCondition("neumann", 1.0),
    )
    return mol, ss


def test_c1_left_neumann_reconstruction_sign():
    """Left Neumann reconstruction of a linear field returns the true boundary."""
    mol, ss = _mol_linear_field()
    fv = mol._fields[0]
    u = ss.interior_points.copy()  # u(z) = z at interior points
    dz = ss.dz
    left = mol._get_left_value(fv, 0.0, 0, u, dz)
    # Pre-fix: u[0] - dz*bc_val = u[0] + dz (wrong). Fixed: u[0] + dz*bc_val = 0.0.
    assert left == pytest.approx(0.0, abs=1e-9)


def test_c1_neumann_convention_is_symmetric():
    """Left and right Neumann use the same outward-normal convention."""
    mol, ss = _mol_linear_field()
    fv = mol._fields[0]
    u = ss.interior_points.copy()
    dz = ss.dz
    left = mol._get_left_value(fv, 0.0, 0, u, dz)
    right = mol._get_right_value(fv, 0.0, len(u) - 1, len(u), u, dz)
    assert left == pytest.approx(0.0, abs=1e-9)
    assert right == pytest.approx(1.0, abs=1e-9)


def test_c1_linear_field_has_zero_second_derivative():
    """The reconstructed left boundary yields d2u/dz2 = 0 for a linear field."""
    mol, ss = _mol_linear_field()
    fv = mol._fields[0]
    u = ss.interior_points.copy()
    dz = ss.dz
    left = mol._get_left_value(fv, 0.0, 0, u, dz)
    fzz = (left - 2 * u[0] + u[1]) / dz**2
    assert fzz == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- C2
@pytest.mark.parametrize("ncp", [1, 2, 3, 4, 5])
def test_c2_radau_weights_sum_to_one(ncp):
    """Radau quadrature weights must sum to 1 for every ncp (was 0.5 at ncp=1)."""
    m = dm.Model(f"w{ncp}")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=1, ncp=ncp, scheme="radau")
    w = DAEBuilder(m, cs)._quadrature_weights()
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("ncp", [1, 2, 3, 4, 5])
def test_c2_radau_weights_exact_to_degree_2ncp_minus_2(ncp):
    """Interpolatory Radau weights integrate t^k exactly up to degree 2*ncp-2."""
    m = dm.Model(f"e{ncp}")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=1, ncp=ncp, scheme="radau")
    w = DAEBuilder(m, cs)._quadrature_weights()
    cp = radau_roots(ncp)
    for deg in range(2 * ncp - 2 + 1):
        approx = float((w * cp**deg).sum())
        assert approx == pytest.approx(1.0 / (deg + 1), abs=1e-12)


@pytest.mark.parametrize("ncp", [1, 2, 3, 4, 5])
def test_c2_integral_of_constant_all_ncp(ncp):
    """integral(1) over [0,3] == 3 for every ncp (bug returned 1.5 at ncp=1)."""
    T = 3.0
    m = dm.Model(f"int{ncp}")
    cs = ContinuousSet("t", bounds=(0, T), nfe=4, ncp=ncp)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=0.0, bounds=(-10, 10))
    dae.set_ode(lambda t, s, a, c: {"x": 1.0})
    dae.discretize()
    iv = dae.integral(lambda t, s, a, c: 1.0)
    xv = dae.get_state("x")
    m.minimize(0 * xv[0, 0] + iv)
    result = m.solve()
    assert result.objective == pytest.approx(T, abs=1e-6)


# --------------------------------------------------------------------------- C3
def test_c3a_collocation_second_order_without_rhs_raises():
    m = dm.Model("c3a")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=4, ncp=3)
    dae = DAEBuilder(m, cs)
    dae.add_second_order_state("q", initial=0.0, initial_velocity=1.0)
    with pytest.raises(RuntimeError, match="second-order ODE"):
        dae.discretize()


def test_c3a_fd_second_order_without_rhs_raises():
    m = dm.Model("c3a_fd")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=4)
    fd = FDBuilder(m, cs, method="backward")
    fd.add_second_order_state("q", initial=0.0, initial_velocity=1.0)
    with pytest.raises(RuntimeError, match="second-order ODE"):
        fd.discretize()


def test_c3b_collocation_rhs_key_typo_raises():
    m = dm.Model("c3b")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=4, ncp=3)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=0.0)
    dae.set_ode(lambda t, s, a, c: {"X": 1.0})  # typo: X vs x
    with pytest.raises(ValueError, match="do not match the declared states"):
        dae.discretize()


def test_c3b_fd_rhs_key_typo_raises():
    m = dm.Model("c3b_fd")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=4)
    fd = FDBuilder(m, cs, method="backward")
    fd.add_state("x", initial=0.0)
    fd.set_ode(lambda t, s, a, c: {"X": 1.0})
    with pytest.raises(ValueError, match="do not match the declared states"):
        fd.discretize()


def test_c3b_partial_rhs_raises():
    """A dict that constrains only some states must raise (the rest would float)."""
    m = dm.Model("c3b_partial")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=4, ncp=3)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=0.0)
    dae.add_state("y", initial=0.0)
    dae.set_ode(lambda t, s, a, c: {"x": 1.0})  # y missing
    with pytest.raises(ValueError, match="y"):
        dae.discretize()


def test_c3_correct_models_still_discretize():
    """No false positives: a correctly specified model discretizes cleanly."""
    # collocation
    m = dm.Model("ok_col")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=4, ncp=3)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=0.0)
    dae.set_ode(lambda t, s, a, c: {"x": 1.0})
    dae.discretize()  # must not raise

    # finite difference
    m2 = dm.Model("ok_fd")
    cs2 = ContinuousSet("t", bounds=(0, 1), nfe=4)
    fd = FDBuilder(m2, cs2, method="backward")
    fd.add_state("x", initial=0.0)
    fd.set_ode(lambda t, s, a, c: {"x": 1.0})
    fd.discretize()  # must not raise


def test_c3_second_order_correct_still_works():
    """A properly configured second-order model discretizes cleanly."""
    m = dm.Model("ok_so")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=4, ncp=3)
    dae = DAEBuilder(m, cs)
    dae.add_second_order_state("q", initial=0.0, initial_velocity=1.0)
    dae.set_second_order_ode(lambda t, pos, vel, a, c: {"q": -pos["q"]})
    dae.discretize()  # must not raise
