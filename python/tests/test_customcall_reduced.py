"""P3.1 — reduced-space McCormick relaxation of ``CustomCall`` (hidden-DOF models).

MAiNGO-parity plan §5-P3.1: a ``CustomCall`` whose opaque jax body traces through MCBox
becomes globally relaxable — its internal intermediates stay hidden, so the model can be
bounded (and, per P3.3, branched) on the true degrees of freedom only. These tests pin
the two things that must hold for that hookup to be *sound*:

1. **Soundness** — the reduced bound of a CustomCall flowsheet is a valid lower bound
   (never above a sampled feasible point / the true optimum), and it is a valid
   relaxation of the *same* problem as the flattened/lifted formulation.
2. **Sound-or-refuse** — an opaque body offers no AST to gate on, so anything MCBox does
   not validate (variable-denominator division — the unvalidated non-affine reciprocal,
   nvs22) must REFUSE (status ``unsupported`` → the solver falls back), never emit a
   possibly-invalid bound.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("DISCOPT_REDUCED_LP_BACKEND", "scipy")
import discopt.modeling as dm  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from discopt._jax.mcbox import (  # noqa: E402
    MCBox,
    UnsupportedMcboxOp,
    mcbox_leaves,
    strict_division,
)
from discopt._jax.mccormick_subgradient import reduced_mccormick_lp_bound  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _mexp(x):
    return x.exp() if isinstance(x, MCBox) else jnp.exp(x)


def _unit(c_in, T, a):
    """One reactor stage: carryover = c_in * exp(-a*T); hidden intermediates k, e."""
    return c_in * _mexp(-a * T)


A = (0.8, 0.6)


def _build_reduced():
    m = dm.Model()
    T = m.continuous("T", 2, lb=[0.2, 0.2], ub=[2.0, 2.0])
    z = m.integer("z", 1, lb=[0], ub=[2])
    F0 = 1.0 + 0.5 * z[0]
    u = [dm.custom(lambda c, t, a=A[i]: _unit(c, t, a), name=f"unit{i}") for i in range(2)]
    c1 = u[0](F0, T[0])
    c2 = u[1](c1, T[1])
    m.minimize(-(F0 - c2) + 0.15 * T[0] * (F0 - c1) + 0.15 * T[1] * (c1 - c2))
    m.subject_to((F0 - c2) >= 0.4)
    return m, [0.2, 0.2, 0.0], [2.0, 2.0, 2.0]


def _true_obj(t0, t1, z):
    F0 = 1.0 + 0.5 * z
    c1 = F0 * np.exp(-A[0] * t0)
    c2 = c1 * np.exp(-A[1] * t1)
    reacted = F0 - c2
    return -reacted + 0.15 * t0 * (F0 - c1) + 0.15 * t1 * (c1 - c2), reacted


def test_customcall_reduced_bound_is_valid_lower_bound():
    """The reduced bound of the CustomCall flowsheet must not exceed the true optimum."""
    m, lb, ub = _build_reduced()
    rb = reduced_mccormick_lp_bound(m, lb, ub)
    assert rb.status == "optimal"
    # true optimum by dense feasible scan
    grid = np.linspace(0.2, 2.0, 60)
    best = np.inf
    for z in range(3):
        for t0 in grid:
            for t1 in grid:
                o, r = _true_obj(t0, t1, z)
                if r >= 0.4 - 1e-9 and o < best:
                    best = o
    assert rb.bound <= best + 1e-6, f"reduced bound {rb.bound} exceeds true opt {best}"


def test_customcall_bound_is_valid_over_the_box():
    """Sampled: the reduced convex underestimator never cuts a feasible point."""
    m, lb, ub = _build_reduced()
    rb = reduced_mccormick_lp_bound(m, lb, ub)
    lb, ub = np.array(lb), np.array(ub)
    rng = np.random.default_rng(0)
    worst = np.inf
    for _ in range(4000):
        x = lb + rng.random(3) * (ub - lb)
        o, r = _true_obj(x[0], x[1], x[2])
        if r >= 0.4 - 1e-9:
            worst = min(worst, o)
    assert rb.bound <= worst + 1e-6


def test_customcall_nonaffine_division_refuses():
    """A hidden non-affine denominator (unvalidated reciprocal) must REFUSE, not bound."""

    def bad(x, y, w):
        return x / (y * w)  # non-affine denominator inside the opaque body

    m = dm.Model()
    v = m.continuous("v", 3, lb=[1.0, 1.0, 1.0], ub=[2.0, 2.0, 2.0])
    m.minimize(dm.custom(bad, name="bad")(v[0], v[1], v[2]))
    rb = reduced_mccormick_lp_bound(m, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert rb.status == "unsupported"


def test_customcall_untraceable_body_refuses():
    """A body using a raw jnp intrinsic MCBox can't trace must refuse (sound fallback)."""

    def uses_sin(x):
        return jnp.sin(x)  # MCBox has no jnp.sin dispatch -> not traceable

    m = dm.Model()
    v = m.continuous("v", 1, lb=[0.1], ub=[1.0])
    m.minimize(dm.custom(uses_sin, name="s")(v[0]))
    rb = reduced_mccormick_lp_bound(m, [0.1], [1.0])
    assert rb.status == "unsupported"


def test_strict_division_context_manager():
    """strict_division refuses MCBox/MCBox but leaves constant-denominator division."""
    leaves = mcbox_leaves(jnp.array([2.0, 3.0]), jnp.array([1.0, 1.0]), jnp.array([4.0, 4.0]))
    # constant denominator: allowed even under strict
    with strict_division():
        z = leaves[0] / 2.0
        assert float(z.cv) == 1.0
    # variable denominator: refused under strict, allowed outside
    with strict_division():
        try:
            leaves[0] / leaves[1]
            raise AssertionError("expected UnsupportedMcboxOp under strict_division")
        except UnsupportedMcboxOp:
            pass
    _ = leaves[0] / leaves[1]  # fine outside strict mode


def test_reduced_is_valid_relaxation_of_flattened():
    """Reduced and flattened are relaxations of the SAME model: both bounds valid, and
    reduced is not tighter than flattened (the P2.4 falsification: reduced <= lifted)."""
    m, lb, ub = _build_reduced()
    rb = reduced_mccormick_lp_bound(m, lb, ub)
    # flattened: e_i, c_i explicit
    mf = dm.Model()
    T = mf.continuous("T", 2, lb=[0.2, 0.2], ub=[2.0, 2.0])
    z = mf.integer("z", 1, lb=[0], ub=[2])
    elb = [float(np.exp(-A[i] * 2.0)) for i in range(2)]
    eub = [float(np.exp(-A[i] * 0.2)) for i in range(2)]
    e = mf.continuous("e", 2, lb=elb, ub=eub)
    clb = [1.0 * elb[0], 1.0 * elb[0] * elb[1]]
    cub = [2.0 * eub[0], 2.0 * eub[0] * eub[1]]
    c = mf.continuous("c", 2, lb=clb, ub=cub)
    F0 = 1.0 + 0.5 * z[0]
    mf.subject_to(e[0] - dm.exp(-A[0] * T[0]) == 0)
    mf.subject_to(e[1] - dm.exp(-A[1] * T[1]) == 0)
    mf.subject_to(c[0] - F0 * e[0] == 0)
    mf.subject_to(c[1] - c[0] * e[1] == 0)
    mf.minimize(-(F0 - c[1]) + 0.15 * T[0] * (F0 - c[0]) + 0.15 * T[1] * (c[0] - c[1]))
    mf.subject_to((F0 - c[1]) >= 0.4)
    lbf = [0.2, 0.2, 0.0] + elb + clb
    ubf = [2.0, 2.0, 2.0] + eub + cub
    rbf = reduced_mccormick_lp_bound(mf, lbf, ubf)
    assert rb.status == "optimal" and rbf.status == "optimal"
    # both are valid lower bounds; reduced is looser-or-equal to lifted (never tighter)
    assert rb.bound <= rbf.bound + 1e-4
