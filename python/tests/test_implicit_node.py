"""Tests for the implicit-function expression node (issue #379, prototype).

Validates (1) the JAX inner-solve numerics: forward Newton correctness and the
implicit-function-theorem VJP against the analytic derivative and finite
differences; and (2) the CustomCall contract inherited by ``dm.implicit`` --
local-NLP-only with no global certificate, and integers rejected.
"""

import pytest

jax = pytest.importorskip("jax")
import discopt.modeling as dm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from discopt.modeling.implicit import _implicit_solver  # noqa: E402


def _sqrt_residual(u, v):
    # v defined by v**2 - u = 0  ->  v = sqrt(u)  (with a positive initial guess)
    return jnp.array([v[0] ** 2 - u[0]])


def test_forward_solve_matches_sqrt():
    phi = _implicit_solver(_sqrt_residual, x0=jnp.array([1.0]))
    for uval in (0.25, 1.0, 4.0, 9.0, 50.0):
        v = phi(jnp.array([uval]))
        assert float(v[0]) == pytest.approx(uval**0.5, rel=1e-6)


def test_ift_vjp_matches_analytic_and_finite_difference():
    phi = _implicit_solver(_sqrt_residual, x0=jnp.array([1.0]))
    u0 = jnp.array([4.0])
    # jacobian dv/du from the custom IFT VJP
    jac = jax.jacobian(phi)(u0)
    analytic = 1.0 / (2.0 * 4.0**0.5)  # d/du sqrt(u) = 1/(2 sqrt(u)) = 0.25
    assert float(jac[0, 0]) == pytest.approx(analytic, rel=1e-6)
    # finite difference
    eps = 1e-4
    fd = (float(phi(u0 + eps)[0]) - float(phi(u0 - eps)[0])) / (2 * eps)
    assert float(jac[0, 0]) == pytest.approx(fd, rel=1e-4)


def test_vector_block_two_by_two():
    # v0 + v1 = u0 ;  v0 - v1 = u1   ->  v0 = (u0+u1)/2, v1 = (u0-u1)/2  (linear)
    def resid(u, v):
        return jnp.array([v[0] + v[1] - u[0], v[0] - v[1] - u[1]])

    phi = _implicit_solver(resid, x0=jnp.zeros(2))
    v = phi(jnp.array([10.0, 4.0]))
    assert float(v[0]) == pytest.approx(7.0, rel=1e-9)
    assert float(v[1]) == pytest.approx(3.0, rel=1e-9)


def test_implicit_in_model_solves_locally():
    # minimize (v - 3)^2 where v = sqrt(u), u in [0.1, 100]  ->  u* = 9, v* = 3
    m = dm.Model()
    u = m.continuous("u", lb=0.1, ub=100.0)
    v = dm.implicit(_sqrt_residual, [u], n_unknowns=1, x0=jnp.array([1.0]))
    m.minimize((v[0] - 3.0) ** 2)
    r = m.solve(initial_solution={u: 4.0})
    assert r.status in ("optimal", "feasible")
    assert float(r.x["u"]) == pytest.approx(9.0, rel=1e-3)
    # CustomCall contract: local NLP path only, no global certificate.
    assert getattr(r, "gap_certified", False) is False


def test_integers_are_rejected():
    m = dm.Model()
    u = m.continuous("u", lb=0.1, ub=100.0)
    m.integer("k", lb=0, ub=5)
    v = dm.implicit(_sqrt_residual, [u], n_unknowns=1, x0=jnp.array([1.0]))
    m.minimize((v[0] - 3.0) ** 2)
    with pytest.raises(Exception):
        m.solve()


def test_nonconvergence_returns_nan():
    # v**2 + u = 0 has no real root for u > 0 -> Newton cannot converge -> NaN,
    # so the failure propagates instead of returning a wrong finite root.
    phi = _implicit_solver(lambda u, v: jnp.array([v[0] ** 2 + u[0]]), x0=jnp.array([1.0]))
    v = phi(jnp.array([4.0]))
    assert not bool(jnp.isfinite(v[0]))


def test_singular_jacobian_returns_nan():
    # residual independent of v -> dg/dv = 0 (singular block) -> non-finite step
    # -> nonsingular-Jacobian gate trips -> NaN.
    phi = _implicit_solver(lambda u, v: jnp.array([u[0] - 1.0]), x0=jnp.array([1.0]))
    v = phi(jnp.array([4.0]))
    assert not bool(jnp.isfinite(v[0]))


def test_build_time_shape_check():
    # residual returns 2 entries but n_unknowns=1 -> caught at build time.
    with pytest.raises(ValueError):
        dm.implicit(lambda u, v: jnp.array([v[0] - 1.0, v[0]]), [], n_unknowns=1)
    # bad x0 length
    with pytest.raises(ValueError):
        dm.implicit(_sqrt_residual, [], n_unknowns=1, x0=jnp.zeros(2))


def test_nonconvergence_fails_model_solve():
    # A model whose implicit block cannot converge should not report a clean
    # optimum with a bogus value; the NaN poisons the objective.
    m = dm.Model()
    u = m.continuous("u", lb=1.0, ub=10.0)
    v = dm.implicit(
        lambda U, V: jnp.array([V[0] ** 2 + U[0]]), [u], n_unknowns=1, x0=jnp.array([1.0])
    )
    m.minimize((v[0] - 3.0) ** 2)
    r = m.solve(initial_solution={u: 4.0})
    assert r.status not in ("optimal", "feasible") or not bool(jnp.isfinite(r.objective))


def test_model_method_form():
    # m.implicit(...) (the Model-method form) matches the module-level builder.
    m = dm.Model()
    u = m.continuous("u", lb=0.1, ub=100.0)
    v = m.implicit(_sqrt_residual, [u], n_unknowns=1, x0=jnp.array([1.0]))
    m.minimize((v[0] - 3.0) ** 2)
    r = m.solve(initial_solution={u: 4.0})
    assert r.status in ("optimal", "feasible")
    assert float(r.x["u"]) == pytest.approx(9.0, rel=1e-3)


def test_multiple_u_inputs():
    # Two scalar inputs feeding a block: v = u0 + u1 (u concatenated flat).
    m = dm.Model()
    a = m.continuous("a", lb=0.0, ub=10.0)
    b = m.continuous("b", lb=0.0, ub=10.0)
    v = m.implicit(lambda U, V: jnp.array([V[0] - (U[0] + U[1])]), [a, b], n_unknowns=1)
    m.minimize((v[0] - 5.0) ** 2 + a**2)  # want a+b=5 with a small -> a~0, b~5
    r = m.solve(initial_solution={a: 1.0, b: 1.0})
    assert r.status in ("optimal", "feasible")
    assert float(r.x["a"]) + float(r.x["b"]) == pytest.approx(5.0, abs=1e-2)
