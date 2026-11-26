"""#2365: post-solve collocation error estimates and mesh refinement.

``DAEBuilder.error_estimate`` reports the defect of the collocation polynomial
between the nodes; ``MeshErrorEstimate.refined_boundaries`` and
``refine_mesh`` refine where it is large while keeping every user boundary.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.dae import (
    BoundaryCondition,
    ContinuousSet,
    DAEBuilder,
    MeshAdequacyWarning,
    MOLBuilder,
    SpatialSet,
    refine_mesh,
)

pytestmark = pytest.mark.filterwarnings("ignore:Variables with very large")


def _decay(mesh, *, k=5.0, lb=-1e20, ncp=3, scheme="radau"):
    """x' = -k x, x(0) = 1, with ``k`` a (fixed) model variable the RHS reads."""
    m = dm.Model("decay")
    kv = m.continuous("k", lb=k, ub=k)
    cs = ContinuousSet(
        "t",
        bounds=(mesh[0], mesh[-1]),
        nfe=len(mesh) - 1,
        ncp=ncp,
        scheme=scheme,
        element_boundaries=np.asarray(mesh, dtype=float),
    )
    d = DAEBuilder(m, cs)
    d.add_state("x", initial=1.0, bounds=(lb, 1e20))
    d.set_ode(lambda t, s, a, c: {"x": -kv * s["x"]})
    d.discretize()
    m.minimize(0 * d.get_state("x")[-1, -1])
    return m, d


def _quiet_estimate(builder, result, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MeshAdequacyWarning)
        return builder.error_estimate(result, **kw)


def test_estimate_bounds_true_interior_error_and_shrinks():
    """The estimate is >= the true error between the nodes and falls with h."""
    checks = 0
    prev = np.inf
    for nfe in (2, 4, 8, 16):
        m, d = _decay(np.linspace(0.0, 2.0, nfe + 1))
        r = m.solve()
        assert r.x is not None
        est = _quiet_estimate(d, r)
        eb = est.element_boundaries
        for i in range(nfe):
            tt = np.linspace(eb[i], eb[i + 1], 51)
            true = np.max(np.abs(d.evaluate_state(r, "x", tt) - np.exp(-5.0 * tt)))
            # Error is accumulated across elements; compare the element's local
            # contribution with the estimate summed over elements up to it.
            assert true <= np.sum(est.error["x"][: i + 1]) * (1 + 1e-9) + 1e-12
            checks += 1
        worst = float(np.max(est.error["x"]))
        assert worst < prev
        prev = worst
    assert checks == 30


def test_polynomial_solution_has_zero_defect_radau_and_legendre():
    """x' = z, z = u (constant control) has a linear solution; collocation is exact."""
    for scheme in ("radau", "legendre"):
        m = dm.Model("lin")
        cs = ContinuousSet("t", bounds=(0.0, 1.0), nfe=3, ncp=2, scheme=scheme)
        d = DAEBuilder(m, cs)
        d.add_state("x", initial=0.0, bounds=(-10, 10))
        d.add_algebraic("z", bounds=(-10, 10))
        d.add_control("u", bounds=(2.0, 2.0))
        d.set_ode(lambda t, s, a, c: {"x": a["z"]})
        d.set_algebraic(lambda t, s, a, c: {"z": a["z"] - c["u"]})
        d.discretize()
        m.minimize(0 * d.get_state("x")[-1, -1])
        r = m.solve()
        with warnings.catch_warnings():
            warnings.simplefilter("error", MeshAdequacyWarning)
            est = d.error_estimate(r)
        assert est.adequate
        assert float(np.max(est.error["x"])) < 1e-8
        np.testing.assert_allclose(d.evaluate_state(r, "x", [0.25, 0.9]), [0.5, 1.8], atol=1e-8)


def test_fast_transient_flags_its_element_and_undershoot():
    """x' = -100 x on 2 elements: the transient sits in element 0. The nodes stay
    above -0.1 (measured min -0.030) but the polynomial between them dips to
    -0.31, so the bound -0.1 is crossed only between nodes, and the warning fires."""
    m, d = _decay(np.linspace(0.0, 1.0, 3), k=100.0, lb=-0.1)
    r = m.solve()
    assert r.x is not None
    with pytest.warns(MeshAdequacyWarning, match="does not resolve"):
        est = d.error_estimate(r)
    assert est.worst_element == 0
    assert est.worst_state == ("x", None)
    assert 0 in est.flagged_elements
    assert est.bound_excursions, "the undershoot between nodes must be reported"
    assert all(e.value < e.bound for e in est.bound_excursions)
    assert "leave the state bounds" in est.summary()


def test_refined_boundaries_keep_user_boundaries_and_split_flagged_only():
    mesh = np.array([0.0, 0.1, 0.35, 1.0])  # deliberately irregular user mesh
    m, d = _decay(mesh, k=30.0)
    r = m.solve()
    est = _quiet_estimate(d, r)
    flagged = set(est.flagged_elements.tolist())
    assert flagged
    new = est.refined_boundaries()
    assert np.all(np.diff(new) > 0)
    assert set(mesh.tolist()) <= set(new.tolist())
    for i in range(len(mesh) - 1):
        inside = np.sum((new > mesh[i]) & (new < mesh[i + 1]))
        assert (inside > 0) == (i in flagged)
    with pytest.raises(ValueError):
        est.refined_boundaries(min_split=1)


def test_warm_start_from_interpolates_previous_polynomials():
    m0, d0 = _decay(np.linspace(0.0, 1.0, 5), k=1.0)
    r0 = m0.solve()
    m1, d1 = _decay(np.linspace(0.0, 1.0, 9), k=1.0)
    guess = d1.warm_start_from(d0, r0)
    x = d1.get_state("x")
    assert set(guess) == {x}
    tp = d1._element_points()
    np.testing.assert_allclose(guess[x], d0.evaluate_state(r0, "x", tp))
    np.testing.assert_allclose(guess[x], np.exp(-tp), atol=1e-5)
    r1 = m1.solve(initial_solution=guess)
    assert r1.x is not None


def test_refine_mesh_reaches_tolerance_and_keeps_boundaries():
    start = np.array([0.0, 0.5, 1.0])
    out = refine_mesh(lambda mesh: _decay(mesh, k=40.0), start, max_rounds=6)
    assert out.adequate
    assert out.history[0].max_error > 1.0
    assert out.estimate.max_error <= 1.0
    assert set(start.tolist()) <= set(out.element_boundaries.tolist())
    assert len(out.history) >= 2
    x_end = out.result.value(out.builder.get_state("x"))[-1, -1]
    assert abs(x_end - np.exp(-40.0)) < 1e-6


def test_refine_mesh_warns_when_limits_stop_it():
    with pytest.warns(MeshAdequacyWarning):
        out = refine_mesh(lambda mesh: _decay(mesh, k=40.0), np.array([0.0, 1.0]), max_rounds=0)
    assert not out.adequate
    assert len(out.history) == 1


def test_mol_forwarding_and_finite_difference_refusal():
    def heat(method):
        m = dm.Model("heat")
        ts = ContinuousSet("t", bounds=(0.0, 0.5), nfe=4, ncp=3)
        ss = SpatialSet("z", bounds=(0.0, 1.0), npts=5)
        mol = MOLBuilder(m, ts, ss, time_method=method)
        dirichlet = BoundaryCondition("dirichlet", 0.0)
        mol.add_field(
            "u",
            bounds=(-2, 2),
            initial=lambda z: np.sin(np.pi * z),
            bc_left=dirichlet,
            bc_right=dirichlet,
        )
        mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": 0.1 * fzz["u"]})
        fields = mol.discretize()
        u = fields["u"]
        m.minimize(0 * (u[-1, -1, 0] if method == "collocation" else u[-1, 0]))
        return m, mol

    m, mol = heat("collocation")
    r = m.solve()
    est = _quiet_estimate(mol, r)
    assert est.error["u"].shape == (4, 5)
    assert np.all(np.isfinite(est.error["u"]))
    m2, mol2 = heat("collocation")
    guess = mol2.warm_start_from(mol, r)
    np.testing.assert_allclose(guess[mol2.get_field("u")], r.value(mol.get_field("u")))

    m_fd, mol_fd = heat("finite_difference")
    r_fd = m_fd.solve()
    with pytest.raises(TypeError, match="collocation"):
        mol_fd.error_estimate(r_fd)


def test_error_estimate_requires_discretize():
    m = dm.Model("x")
    d = DAEBuilder(m, ContinuousSet("t", bounds=(0, 1), nfe=2))
    with pytest.raises(RuntimeError, match="discretize"):
        d.error_estimate(None)
