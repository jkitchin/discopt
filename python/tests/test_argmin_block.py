"""``dm.argmin`` -- an inner NLP as a block of an outer model (#1216).

The issue's complaint is not that discopt gets the *stated* model wrong; it is
that hand-writing a follower's KKT conditions states a different model. On the
projection follower

    min_y  ||y - (p, 1)||^2     s.t.  ||y||^2 = 1     ->   y*(p) = (p, 1)/sqrt(p^2+1)

the objective is strictly convex but the feasible set is not, so the KKT system
has two roots: the minimizer (near side, ``y0 > 0`` for ``p > 0``) and a
maximizer (far side). A leader asking for ``y0 = -0.9`` -- which no projection can
produce -- is served the far-side root by the hand-written-KKT route and told it
is optimal.

``test_kkt_block_selects_the_far_side_root`` pins that failure so the motivation
cannot rot; every other test pins that an ``argmin`` block does not have it, and
that its derivatives are the exact sIPOPT sensitivities to *second* order.

The derivative tests compare against closed forms, so they are vacuity-proof only
if they actually run: each parametrized case is one executed comparison and the
module asserts the exact count in ``test_derivative_cases_all_ran``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling import Model

pytestmark = pytest.mark.requires_pounce

# Grid the analytic comparisons run on; also the vacuity counter.
_P_GRID = (0.2, 1.0, 2.0, 2.5)
_RAN: set[float] = set()


def _projection_follower(p_value: float = 1.0) -> tuple[Model, dm.Parameter]:
    """``min ||y - (p,1)||^2 s.t. ||y||^2 = 1`` -- convex objective, nonconvex set."""
    inner = Model("projection")
    y = inner.continuous("y", shape=(2,), lb=-2.0, ub=2.0)
    q = inner.parameter("q", value=p_value)
    inner.minimize((y[0] - q) ** 2 + (y[1] - 1.0) ** 2)
    inner.subject_to(y[0] * y[0] + y[1] * y[1] == 1.0)
    return inner, q


def _exact_y(p: float) -> np.ndarray:
    return np.array([p, 1.0]) / np.sqrt(p * p + 1.0)


def _exact_dy0_dp(p: float) -> float:
    return float((p * p + 1.0) ** -1.5)


def _exact_d2y0_dp2(p: float) -> float:
    return float(-3.0 * p * (p * p + 1.0) ** -2.5)


# ---------------------------------------------------------------------------
# The failure the block exists to remove
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_kkt_block_selects_the_far_side_root():
    """Hand-written KKT is a relaxation: the leader picks the follower's maximizer."""

    def kkt_residual(u, v):
        p = u[0]
        y0, y1, lam = v[0], v[1], v[2]
        return [
            2 * (y0 - p) + 2 * lam * y0,
            2 * (y1 - 1.0) + 2 * lam * y1,
            y0 * y0 + y1 * y1 - 1.0,
        ]

    m = Model("leader")
    p = m.continuous("p", lb=0.05, ub=3.0)
    v = dm.implicit_full_space(
        m,
        kkt_residual,
        [p],
        n_unknowns=3,
        bounds=(np.array([-2.0, -2.0, -10.0]), np.array([2.0, 2.0, 10.0])),
    )
    m.minimize((v[0] + 0.9) ** 2)
    r = m.solve()

    p_star = float(np.asarray(r.value(p)).ravel()[0])
    y_star = np.asarray(r.value(v)).ravel()[:2]
    # It reports success at a point that is not the projection of anything: the
    # returned y0 is negative while the true projection's is positive.
    assert float(r.objective) < 1e-6
    assert y_star[0] < 0.0 < _exact_y(p_star)[0]


@pytest.mark.slow
def test_argmin_block_gives_the_honest_optimum():
    """The same leader, with the follower passed as a model, lands on the truth.

    ``y0(p) = p/sqrt(p^2+1)`` is increasing on the box, so the closest a real
    projection gets to the requested ``-0.9`` is at the *lower* bound of ``p``.
    """
    inner, q = _projection_follower()
    m = Model("leader")
    p = m.continuous("p", lb=0.05, ub=3.0)
    v = dm.argmin(inner, bind={q: p})
    m.minimize((v[0] + 0.9) ** 2)
    r = m.solve()

    p_star = float(np.asarray(r.value(p)).ravel()[0])
    grid = np.linspace(0.05, 3.0, 20001)
    best = grid[np.argmin((grid / np.sqrt(grid**2 + 1.0) + 0.9) ** 2)]
    assert p_star == pytest.approx(best, abs=1e-3)
    assert float(r.objective) == pytest.approx((best / np.sqrt(best**2 + 1.0) + 0.9) ** 2, abs=1e-6)
    # The CustomCall contract: a local solve, honestly reported as one.
    assert r.status == "feasible"
    assert getattr(r, "gap_certified", False) is False
    assert getattr(r, "bound", None) is None


# ---------------------------------------------------------------------------
# Forward value and derivatives against closed forms
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("p_value", _P_GRID)
def test_layer_matches_exact_solution_and_both_derivatives(p_value):
    """Forward value, dx*/dp and d2x*/dp2 against the closed form."""
    import jax
    import jax.numpy as jnp

    inner, q = _projection_follower()
    phi = dm.argmin_layer(inner, [q])

    p = jnp.float64(p_value)
    got = np.asarray(phi(jnp.array([p])))
    assert got == pytest.approx(_exact_y(p_value), abs=1e-7)

    d1 = float(jax.grad(lambda t: phi(jnp.array([t]))[0])(p))
    assert d1 == pytest.approx(_exact_dy0_dp(p_value), rel=1e-5)

    # Forward-over-reverse: the exact nesting the outer solver's Lagrangian
    # Hessian uses. A first-order-only rule (custom_vjp) raises here.
    d2 = float(jax.jacfwd(jax.grad(lambda t: phi(jnp.array([t]))[0]))(p))
    assert d2 == pytest.approx(_exact_d2y0_dp2(p_value), rel=1e-4)

    _RAN.add(p_value)


@pytest.mark.slow
def test_derivative_cases_all_ran():
    """Vacuity guard: the parametrized comparisons above really executed."""
    assert _RAN == set(_P_GRID), f"only {sorted(_RAN)} of {sorted(_P_GRID)} ran"


@pytest.mark.slow
def test_inner_solve_is_memoized_across_derivative_passes():
    """Value, gradient and Hessian at one iterate cost ONE inner solve."""
    import jax
    import jax.numpy as jnp

    inner, q = _projection_follower()
    phi = dm.argmin_layer(inner, [q])
    f = lambda t: phi(jnp.array([t]))[0]  # noqa: E731

    p = jnp.float64(1.3)
    before = phi.inner_solve_count()
    float(f(p))
    float(jax.grad(f)(p))
    float(jax.jacfwd(jax.grad(f))(p))
    assert phi.inner_solve_count() - before == 1


# ---------------------------------------------------------------------------
# Inequality constraints and bounds: the active set enters the KKT system
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_active_inequality_is_differentiated_as_an_equality():
    """``min (y-p)^2 s.t. y <= 1``: dy*/dp is 1 while inactive, 0 once active."""
    import jax
    import jax.numpy as jnp

    inner = Model("clip")
    y = inner.continuous("y", lb=-5.0, ub=5.0)
    q = inner.parameter("q", value=0.0)
    inner.minimize((y - q) * (y - q))
    inner.subject_to(y <= 1.0)
    phi = dm.argmin_layer(inner, [q])

    f = lambda t: phi(jnp.array([t]))[0]  # noqa: E731
    assert float(f(jnp.float64(0.5))) == pytest.approx(0.5, abs=1e-7)
    assert float(jax.grad(f)(jnp.float64(0.5))) == pytest.approx(1.0, abs=1e-6)
    assert float(f(jnp.float64(2.0))) == pytest.approx(1.0, abs=1e-6)
    assert float(jax.grad(f)(jnp.float64(2.0))) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.slow
def test_variable_at_a_bound_has_zero_sensitivity():
    """A variable pinned by its own bound is frozen in the sensitivity system."""
    import jax
    import jax.numpy as jnp

    inner = Model("boxed")
    y = inner.continuous("y", lb=-1.0, ub=1.0)
    q = inner.parameter("q", value=0.0)
    inner.minimize((y - q) * (y - q))
    phi = dm.argmin_layer(inner, [q])

    f = lambda t: phi(jnp.array([t]))[0]  # noqa: E731
    assert float(f(jnp.float64(0.25))) == pytest.approx(0.25, abs=1e-6)
    assert float(jax.grad(f)(jnp.float64(0.25))) == pytest.approx(1.0, abs=1e-6)
    assert float(f(jnp.float64(3.0))) == pytest.approx(1.0, abs=1e-6)
    assert float(jax.grad(f)(jnp.float64(3.0))) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.slow
def test_two_parameters_give_the_full_jacobian():
    """``min ||y - a||^2 s.t. sum(y) == b`` has a closed-form dy*/d(a, b)."""
    import jax
    import jax.numpy as jnp

    inner = Model("proj_plane")
    y = inner.continuous("y", shape=(2,), lb=-10.0, ub=10.0)
    a = inner.parameter("a", value=0.0)
    b = inner.parameter("b", value=1.0)
    inner.minimize((y[0] - a) * (y[0] - a) + (y[1] - 2.0 * a) * (y[1] - 2.0 * a))
    inner.subject_to(y[0] + y[1] == b)
    phi = dm.argmin_layer(inner, [a, b])

    # y* = c + (b - sum(c))/2 * 1, with c = (a, 2a):
    #   y0 = a + (b - 3a)/2 = b/2 - a/2,  y1 = 2a + (b - 3a)/2 = b/2 + a/2
    jac = np.asarray(jax.jacobian(phi)(jnp.array([0.7, 1.3])))
    assert jac == pytest.approx(np.array([[-0.5, 0.5], [0.5, 0.5]]), abs=1e-6)


# ---------------------------------------------------------------------------
# Refusals: the block declines what it cannot represent soundly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_integer_inner_variable_is_refused():
    inner = Model("int_inner")
    y = inner.continuous("y", lb=0.0, ub=5.0)
    z = inner.integer("z", lb=0, ub=5)
    q = inner.parameter("q", value=1.0)
    inner.minimize((y - q) * (y - q) + z)
    m = Model("outer")
    p = m.continuous("p", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="integer/binary"):
        dm.argmin(inner, bind={q: p})


@pytest.mark.unit
def test_maximize_inner_is_refused():
    inner = Model("maxi")
    y = inner.continuous("y", lb=0.0, ub=5.0)
    q = inner.parameter("q", value=1.0)
    inner.maximize(-(y - q) * (y - q))
    m = Model("outer")
    p = m.continuous("p", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="argmax"):
        dm.argmin(inner, bind={q: p})


@pytest.mark.unit
def test_missing_objective_is_refused():
    inner = Model("no_obj")
    inner.continuous("y", lb=0.0, ub=5.0)
    q = inner.parameter("q", value=1.0)
    m = Model("outer")
    p = m.continuous("p", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="no objective"):
        dm.argmin(inner, bind={q: p})


@pytest.mark.unit
def test_empty_bind_is_refused():
    inner, q = _projection_follower()
    with pytest.raises(ValueError, match="non-empty"):
        dm.argmin(inner, bind={})


@pytest.mark.unit
def test_foreign_parameter_is_refused():
    inner, _q = _projection_follower()
    other = Model("other")
    stranger = other.parameter("stranger", value=1.0)
    p = other.continuous("p", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="does not belong"):
        dm.argmin(inner, bind={stranger: p})


@pytest.mark.slow
def test_infeasible_inner_evaluates_to_nan_not_a_wrong_number():
    """A refused inner solve propagates NaN; it never invents a value."""
    import jax.numpy as jnp

    inner = Model("infeasible")
    y = inner.continuous("y", lb=0.0, ub=1.0)
    q = inner.parameter("q", value=1.0)
    inner.minimize(y * y)
    inner.subject_to(y >= 2.0 + q * q)  # unreachable inside [0, 1] for any q
    phi = dm.argmin_layer(inner, [q])
    assert not np.isfinite(np.asarray(phi(jnp.array([1.0])))).all()


@pytest.mark.unit
def test_importing_discopt_does_not_pull_jax():
    """``argmin`` keeps JAX behind the function call, like the rest of modeling."""
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, discopt.modeling as dm; "
            "assert hasattr(dm, 'argmin'); "
            "print(sum(1 for k in sys.modules if k == 'jax' or k.startswith('jax.')))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "0", out.stdout


@pytest.mark.slow
def test_vector_valued_constraint_body_gives_all_its_rows():
    """One Constraint may compile to several rows; each needs its own multiplier.

    ``min sum_i (y_i - q)^2  s.t.  A y = b`` with a 2x3 ``A`` is a single
    Constraint carrying two rows. Its projection has the closed form
    ``y*(q) = (q/3, (3-q)/3, (3+q)/3)``, so a row/multiplier misalignment shows up
    immediately in both the value and the derivative.
    """
    import jax
    import jax.numpy as jnp

    a_mat = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    inner = Model("vec")
    y = inner.continuous("y", shape=(3,), lb=-5.0, ub=5.0)
    q = inner.parameter("q", value=1.0)
    inner.minimize(dm.sum(lambda i: (y[i] - q) * (y[i] - q), over=range(3)))
    inner.subject_to(a_mat @ y == np.array([1.0, 2.0]))

    phi = dm.argmin_layer(inner, [q])
    q_val = 1.4
    got = np.asarray(phi(jnp.array([q_val])))
    assert got == pytest.approx(
        np.array([q_val / 3.0, (3.0 - q_val) / 3.0, (3.0 + q_val) / 3.0]), abs=1e-7
    )
    jac = np.asarray(jax.jacobian(phi)(jnp.array([q_val]))).ravel()
    assert jac == pytest.approx(np.array([1 / 3, -1 / 3, 1 / 3]), abs=1e-6)


# ---------------------------------------------------------------------------
# The lowered arm: KKT reformulation of a CERTIFIED-CONVEX follower (#1216)
# ---------------------------------------------------------------------------


def _convex_follower(bound: float = 0.5):
    """``min (y - q)^2 s.t. y <= bound``, y in [-1, 1] -- convex in y, so its KKT
    conditions characterize its optimum and the lowering is exact."""
    inner = Model("follower")
    y = inner.continuous("y", lb=-1.0, ub=1.0)
    q = inner.parameter("q", value=0.0)
    inner.minimize((y - q) * (y - q))
    inner.subject_to(y <= bound)
    return inner, q


def _leader_with(v, p):
    """Leader: get the follower's answer near 0.3, paying 0.01 p^2 to steer."""
    return (v[0] - 0.3) ** 2 + 0.01 * p * p


@pytest.mark.slow
def test_lowered_arm_agrees_with_the_opaque_arm_and_certifies():
    """Same answer as `dm.argmin`, but with a global certificate."""
    inner_l, q_l = _convex_follower()
    m_low = Model("leader_lowered")
    p_low = m_low.continuous("p", lb=-2.0, ub=2.0)
    v_low = dm.argmin_kkt(m_low, inner_l, bind={q_l: p_low}, multiplier_ub=100.0)
    m_low.minimize(_leader_with(v_low, p_low))
    r_low = m_low.solve()

    inner_o, q_o = _convex_follower()
    m_op = Model("leader_opaque")
    p_op = m_op.continuous("p", lb=-2.0, ub=2.0)
    v_op = dm.argmin(inner_o, bind={q_o: p_op})
    m_op.minimize(_leader_with(v_op, p_op))
    r_op = m_op.solve()

    p_star_low = float(np.asarray(r_low.value(p_low)).ravel()[0])
    p_star_op = float(np.asarray(r_op.value(p_op)).ravel()[0])
    assert p_star_low == pytest.approx(p_star_op, abs=1e-4)
    # The follower is a projection onto [-1, 0.5]; at the optimum it is interior,
    # so y* = p and the leader's trade-off puts p* at 0.3/1.01.
    assert p_star_low == pytest.approx(0.3 / 1.01, abs=1e-4)
    assert float(np.asarray(r_low.value(v_low[0])).ravel()[0]) == pytest.approx(
        p_star_low, abs=1e-5
    )

    # The point of lowering: the certificate the opaque node cannot give.
    assert r_low.status == "optimal"
    assert r_low.gap_certified is True
    assert r_low.bound is not None
    assert r_op.status == "feasible"
    assert r_op.gap_certified is False


@pytest.mark.slow
def test_lowered_arm_exports_to_nl():
    """The whole point of `.nl`: hand the SAME formulation to another solver."""
    inner, q = _convex_follower()
    m = Model("leader")
    p = m.continuous("p", lb=-2.0, ub=2.0)
    v = dm.argmin_kkt(m, inner, bind={q: p}, method="strong_duality")
    m.minimize(_leader_with(v, p))

    text = m.to_nl()
    assert text.startswith("g3")  # AMPL .nl header
    header = text.splitlines()[1].split()
    n_vars, n_cons = int(header[0]), int(header[1])
    # leader p + follower y + one multiplier, and the follower's optimality rows.
    assert n_vars >= 3
    assert n_cons >= 3

    r = m.solve()
    assert float(np.asarray(r.value(p)).ravel()[0]) == pytest.approx(0.3 / 1.01, abs=1e-4)


@pytest.mark.slow
def test_lowered_arm_handles_a_vector_follower_variable():
    """A 1-D follower variable: components become scalars, `dm.sum` folds."""
    inner = Model("alloc")
    w = inner.continuous("w", shape=(2,), lb=0.0, ub=1.0)
    q = inner.parameter("q", value=0.5)
    inner.minimize((w[0] - q) * (w[0] - q) + w[1] * w[1])
    inner.subject_to(dm.sum(w) == 1.0)

    m = Model("leader")
    p = m.continuous("p", lb=0.0, ub=1.0)
    v = dm.argmin_kkt(m, inner, bind={q: p}, method="strong_duality")
    assert v.shape == (2,)
    m.minimize((v[0] - 0.8) ** 2)
    r = m.solve()
    # min (w0-q)^2 + w1^2 s.t. w0 + w1 = 1  ->  w0 = (1+q)/2, so w0 = 0.8 at q = 0.6.
    assert float(np.asarray(r.value(p)).ravel()[0]) == pytest.approx(0.6, abs=1e-4)
    assert float(np.asarray(r.value(v[0])).ravel()[0]) == pytest.approx(0.8, abs=1e-4)


@pytest.mark.slow
def test_nonconvex_follower_is_refused_by_the_lowering():
    """The projection follower: nonlinear equality, so KKT is only necessary.

    This is the failure `test_kkt_block_selects_the_far_side_root` demonstrates.
    Hand-written, it is emitted silently; through `argmin_kkt` it is refused.
    """
    inner, q = _projection_follower()
    m = Model("leader")
    p = m.continuous("p", lb=0.05, ub=3.0)
    with pytest.raises(NotImplementedError, match="affine in the follower variables"):
        dm.argmin_kkt(m, inner, bind={q: p})


@pytest.mark.unit
def test_lowering_refuses_an_opaque_node_in_the_follower():
    import jax.numpy as jnp

    inner = Model("opaque_follower")
    y = inner.continuous("y", lb=-1.0, ub=1.0)
    q = inner.parameter("q", value=0.0)
    weird = dm.custom(lambda t: jnp.sinc(t), name="sinc")
    inner.minimize((y - q) * (y - q) + weird(y))

    m = Model("leader")
    p = m.continuous("p", lb=-1.0, ub=1.0)
    with pytest.raises(ValueError, match="opaque node"):
        dm.argmin_kkt(m, inner, bind={q: p})


@pytest.mark.unit
def test_lowering_refuses_a_multidimensional_follower_variable():
    inner = Model("matrix_follower")
    y = inner.continuous("y", shape=(2, 2), lb=-1.0, ub=1.0)
    q = inner.parameter("q", value=0.0)
    inner.minimize((y[0, 0] - q) * (y[0, 0] - q))

    m = Model("leader")
    p = m.continuous("p", lb=-1.0, ub=1.0)
    with pytest.raises(NotImplementedError, match="shape"):
        dm.argmin_kkt(m, inner, bind={q: p})


@pytest.mark.unit
def test_lowering_refuses_an_integer_follower_and_an_empty_bind():
    inner = Model("int_follower")
    y = inner.continuous("y", lb=0.0, ub=5.0)
    z = inner.integer("z", lb=0, ub=5)
    q = inner.parameter("q", value=1.0)
    inner.minimize((y - q) * (y - q) + z)
    m = Model("leader")
    p = m.continuous("p", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="integer/binary"):
        dm.argmin_kkt(m, inner, bind={q: p})
    with pytest.raises(ValueError, match="non-empty"):
        dm.argmin_kkt(m, inner, bind={})


@pytest.mark.slow
def test_unbound_inner_parameters_are_frozen_at_their_value():
    """Only what `bind` names couples to the leader; the rest are data."""
    inner = Model("two_params")
    y = inner.continuous("y", lb=-5.0, ub=5.0)
    q = inner.parameter("q", value=0.0)
    scale = inner.parameter("scale", value=3.0)
    inner.minimize((y - scale * q) * (y - scale * q))

    m = Model("leader")
    p = m.continuous("p", lb=-1.0, ub=1.0)
    v = dm.argmin_kkt(m, inner, bind={q: p}, method="strong_duality")
    m.minimize((v[0] - 1.5) ** 2)
    r = m.solve()
    # y* = 3 q, so the leader drives q to 0.5.
    assert float(np.asarray(r.value(p)).ravel()[0]) == pytest.approx(0.5, abs=1e-4)
