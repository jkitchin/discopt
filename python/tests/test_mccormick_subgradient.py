"""Soundness of the reduced-space McCormick subgradient evaluator (#572).

The reduced-space evaluator must produce a GENUINE McCormick underestimator over
the original variables (not the collapse-to-exact-function that the compiled
relaxation fns give at x_cv==x_cc), so that jax.grad yields a VALID subgradient
usable for MAiNGO-style linearized lower-bounding cuts. These tests pin:
  - envelope correctness (x*y at an interior point = the McCormick value, not x*y)
  - validity (cv <= f on the box)
  - convexity of cv
  - subgradient validity (cv is above every tangent)
  - bound validity (min cv <= min f)
  - loud refusal outside the sound v1 scope (no silent invalid bound)
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
from discopt._jax.mccormick_subgradient import (
    build_reduced_relaxation,
    reduced_mccormick_lp_bound,
)

RNG = np.random.default_rng(0)


def _sound(model, lb, ub, f_true, n_samples=300):
    R = build_reduced_relaxation(model, lb, ub)
    u = R.obj_under
    n = R.n
    P = lb + RNG.random((n_samples, n)) * (ub - lb)
    # validity: u(x) <= f(x)
    for x in P:
        assert float(u(jnp.asarray(x))) <= float(f_true(x)) + 1e-6 * (abs(float(f_true(x))) + 1)
    # convexity + subgradient validity
    gu = jax.grad(lambda z: u(z))
    for _ in range(120):
        a = lb + RNG.random(n) * (ub - lb)
        b = lb + RNG.random(n) * (ub - lb)
        mid = 0.5 * (a + b)
        assert (
            float(u(jnp.asarray(mid)))
            <= 0.5 * (float(u(jnp.asarray(a))) + float(u(jnp.asarray(b)))) + 1e-6
        )
        u_a = float(u(jnp.asarray(a)))
        g = np.asarray(gu(jnp.asarray(a)))
        assert float(u(jnp.asarray(b))) >= u_a + g @ (b - a) - 1e-6 * (abs(u_a) + 1)
    # bound validity: min cv <= min f (sampled)
    min_u = min(float(u(jnp.asarray(x))) for x in P)
    min_f = min(float(f_true(x)) for x in P)
    assert min_u <= min_f + 1e-6 * (abs(min_f) + 1)


def test_bilinear_envelope_not_collapse():
    """x*y on [0,4]^2: cv at (1,1) is the McCormick envelope value 0, not 1."""
    m = dm.Model()
    x = m.continuous("x", 2, lb=0.0, ub=4.0)
    m.minimize(x[0] * x[1])
    R = build_reduced_relaxation(m, np.array([0.0, 0.0]), np.array([4.0, 4.0]))
    assert abs(float(R.obj_under(jnp.array([1.0, 1.0])))) < 1e-9


def test_bilinear_sound():
    m = dm.Model()
    x = m.continuous("x", 2, lb=0.0, ub=4.0)
    m.minimize(x[0] * x[1])
    _sound(m, np.array([0.0, 0.0]), np.array([4.0, 4.0]), lambda x: x[0] * x[1])


def test_indefinite_qp_sound():
    m = dm.Model()
    x = m.continuous("x", 2, lb=0.0, ub=4.0)
    m.minimize(x[0] * x[1] - x[0] ** 2 + 2.0 * x[1])
    _sound(
        m, np.array([0.0, 0.0]), np.array([4.0, 4.0]), lambda x: x[0] * x[1] - x[0] ** 2 + 2 * x[1]
    )


def test_qp_with_quadratic_constraint_sound():
    m = dm.Model()
    x = m.continuous("x", 2, lb=-2.0, ub=3.0)
    m.minimize(x[0] ** 2 - 3.0 * x[0] * x[1])
    m.subject_to(x[0] * x[1] - 2.0 <= 0)
    _sound(m, np.array([-2.0, -2.0]), np.array([3.0, 3.0]), lambda x: x[0] ** 2 - 3 * x[0] * x[1])


def test_even_power_sound():
    m = dm.Model()
    x = m.continuous("x", 2, lb=-2.0, ub=3.0)
    m.minimize(x[0] ** 4 - 2.0 * x[0] * x[1])
    _sound(m, np.array([-2.0, -2.0]), np.array([3.0, 3.0]), lambda x: x[0] ** 4 - 2 * x[0] * x[1])


def test_odd_power_positive_base_sound():
    """x**3 over a positive-base box (like st_e36) is monotone convex -> sound."""
    m = dm.Model()
    x = m.continuous("x", 1, lb=1.0, ub=4.0)
    m.minimize(x[0] ** 3)
    _sound(m, np.array([1.0]), np.array([4.0]), lambda x: x[0] ** 3)


def test_odd_power_spanning_zero_now_sound():
    """P0.3: MCBox relaxes x**3 over a sign-spanning base soundly (repeated-mult),
    where the v1 evaluator had to refuse it. The bound must be a valid lower bound."""
    m = dm.Model()
    x = m.continuous("x", 1, lb=-2.0, ub=2.0)
    m.minimize(x[0] ** 3)
    R = build_reduced_relaxation(m, np.array([-2.0]), np.array([2.0]))
    P = -2.0 + RNG.random((300, 1)) * 4.0
    for p in P:
        assert float(R.obj_under(jnp.asarray(p))) <= float(p[0] ** 3) + 1e-6


def test_transcendental_now_sound():
    """P0.3: MCBox relaxes exp(x)*x soundly (exp + general product) where v1 refused."""
    m = dm.Model()
    x = m.continuous("x", 1, lb=0.1, ub=3.0)
    m.minimize(dm.exp(x[0]) * x[0])
    r = reduced_mccormick_lp_bound(m, [0.1], [3.0])
    assert r.status == "optimal"
    true_min = min(float(np.exp(p) * p) for p in np.linspace(0.1, 3.0, 2000))
    assert r.bound <= true_min + 1e-4  # valid lower bound


def _sampled_min(f, lb, ub, n, k=30000, feas=None):
    P = lb + RNG.random((k, n)) * (ub - lb)
    vals = [float(f(x)) for x in P if (feas is None or feas(x))]
    return min(vals)


def test_kelley_bound_bilinear_exact():
    m = dm.Model()
    x = m.continuous("x", 2, lb=0.0, ub=4.0)
    m.minimize(x[0] * x[1])
    r = reduced_mccormick_lp_bound(m, [0.0, 0.0], [4.0, 4.0])
    assert r.status == "optimal"
    assert abs(r.bound) < 1e-6  # true min is 0
    assert r.rounds <= 6  # converges in a few Kelley rounds


def test_kelley_bound_indef_qp_valid_and_tight():
    m = dm.Model()
    x = m.continuous("x", 2, lb=0.0, ub=4.0)
    m.minimize(x[0] * x[1] - x[0] ** 2 + 2.0 * x[1])
    def f(z):
        return z[0] * z[1] - z[0] ** 2 + 2 * z[1]
    r = reduced_mccormick_lp_bound(m, [0.0, 0.0], [4.0, 4.0])
    tmin = _sampled_min(f, np.array([0.0, 0.0]), np.array([4.0, 4.0]), 2)
    assert r.status == "optimal"
    assert r.bound <= tmin + 1e-4  # valid lower bound
    assert r.bound >= tmin - 2.0  # and tight (near-exact on this QP)


def test_kelley_bound_valid_with_quadratic_constraint():
    m = dm.Model()
    x = m.continuous("x", 2, lb=-2.0, ub=3.0)
    m.minimize(x[0] ** 2 - 3.0 * x[0] * x[1])
    m.subject_to(x[0] * x[1] - 2.0 <= 0)
    def f(z):
        return z[0] ** 2 - 3 * z[0] * z[1]
    r = reduced_mccormick_lp_bound(m, [-2.0, -2.0], [3.0, 3.0])
    tmin = _sampled_min(
        f, np.array([-2.0, -2.0]), np.array([3.0, 3.0]), 2, feas=lambda z: z[0] * z[1] - 2.0 <= 0
    )
    assert r.status == "optimal"
    assert r.bound <= tmin + 1e-4  # valid (looser: McCormick constraint relaxation)


def test_kelley_bound_detects_infeasible_relaxation():
    m = dm.Model()
    x = m.continuous("x", 1, lb=0.0, ub=1.0)
    m.minimize(x[0])
    m.subject_to(x[0] >= 5.0)  # infeasible over the box
    r = reduced_mccormick_lp_bound(m, [0.0], [1.0])
    assert r.status == "infeasible"


def test_kelley_bound_unsupported_returns_flag_not_crash():
    # division by a variable is still outside MCBox scope -> clean fallback, no crash
    m = dm.Model()
    x = m.continuous("x", 2, lb=1.0, ub=3.0)
    m.minimize(x[0] / x[1])
    r = reduced_mccormick_lp_bound(m, [1.0, 1.0], [3.0, 3.0])
    assert r.status == "unsupported"  # falls back cleanly; no partial/invalid bound


def test_maximize_sense_valid_lower_bound():
    """For a maximize model the reduced bound (in min form) must stay valid."""
    m = dm.Model()
    x = m.continuous("x", 2, lb=0.0, ub=4.0)
    m.maximize(-(x[0] * x[1]))
    R = build_reduced_relaxation(m, np.array([0.0, 0.0]), np.array([4.0, 4.0]))
    # obj_under is in minimize form: minimizing -(-(x*y)) = minimizing x*y form.
    P = np.array([0.0, 0.0]) + RNG.random((200, 2)) * 4.0
    for x0 in P:
        # obj_under underestimates the minimize-form objective (-f for maximize)
        assert float(R.obj_under(jnp.asarray(x0))) <= -(-(x0[0] * x0[1])) + 1e-6 + 1e-6 * abs(
            x0[0] * x0[1]
        )
