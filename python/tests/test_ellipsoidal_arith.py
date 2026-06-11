"""Tests for ellipsoidal arithmetic (M7 of issue #81).

Covers :mod:`discopt._jax.ellipsoidal_arith`:

* **Geometry** — the closed-form ellipsoid calculus: support function, exact
  affine image, and trace-minimising Minkowski-sum outer approximation.
* **Soundness (rigorous-bound invariant)** — every propagated enclosure
  contains the true range of its node on > 10^4 random samples, so a relaxation
  built on it can never exclude a feasible point.
* **Correlation win** — on correlated-affine expressions the ellipsoidal
  enclosure is at least as tight as (and usually strictly tighter than) the
  interval/affine-arithmetic enclosure, on well over 50% of nodes.
* **Villanueva-style reproduction** — for an affine parametric map the
  ellipsoidal enclosure equals the analytic image ellipsoid to ~machine
  precision (the defining exactness property of ellipsoidal calculus), and for
  a nonlinear parametric map it rigorously contains the true reachable set.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.ellipsoidal_arith import (
    Ellipsoid,
    EllipsoidalForm,
    bounding_ellipsoid_of_box,
    forms_from_box,
    forms_from_ellipsoid,
    joint_ellipsoid,
)

jax.config.update("jax_enable_x64", True)

N_SAMPLES = 12_000


def _ball_samples(rng, k, n=N_SAMPLES):
    """Uniform samples in the unit 2-ball of dimension ``k``."""
    z = rng.standard_normal((n, k))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    radius = rng.uniform(0.0, 1.0, size=(n, 1)) ** (1.0 / k)
    return z * radius


# ---------------------------------------------------------------------------
# Geometry of the Ellipsoid object
# ---------------------------------------------------------------------------


def test_support_function_encloses_projections():
    rng = np.random.default_rng(0)
    c = np.array([0.3, -0.7, 1.1])
    a = rng.standard_normal((3, 3))
    ell = Ellipsoid(c, a @ a.T)
    pts = c[None] + _ball_samples(rng, 3) @ np.linalg.cholesky(ell.shape).T
    for _ in range(25):
        s = rng.standard_normal(3)
        lo, hi = ell.support(s)
        proj = pts @ s
        assert proj.min() >= lo - 1e-9
        assert proj.max() <= hi + 1e-9


def test_affine_image_is_exact():
    """``A E + b`` has support exactly ``A``-transformed: tight, not just sound."""
    rng = np.random.default_rng(1)
    c = np.array([1.0, -2.0, 0.5])
    a = rng.standard_normal((3, 3))
    ell = Ellipsoid(c, a @ a.T)
    m = rng.standard_normal((2, 3))
    b = np.array([0.5, -1.0])
    img = ell.affine_image(m, b)
    for _ in range(10):
        s = rng.standard_normal(2)
        lo, hi = img.support(s)
        # support of A E + b along s == support of E along A^T s, plus s.b
        lo0, hi0 = ell.support(m.T @ s)
        shift = float(s @ b)
        assert lo == pytest.approx(lo0 + shift, rel=1e-9, abs=1e-9)
        assert hi == pytest.approx(hi0 + shift, rel=1e-9, abs=1e-9)


def test_minkowski_sum_is_sound_and_trace_optimal():
    rng = np.random.default_rng(2)
    e1 = Ellipsoid(np.array([0.0, 0.0]), np.diag([1.0, 0.25]))
    e2 = Ellipsoid(np.array([0.5, -0.5]), np.diag([0.25, 1.0]))
    summed = e1.minkowski_sum(e2)
    g1 = np.linalg.cholesky(e1.shape)
    g2 = np.linalg.cholesky(e2.shape)
    xa = e1.center[None] + _ball_samples(rng, 2, 4000) @ g1.T
    xb = e2.center[None] + _ball_samples(rng, 2, 4000) @ g2.T
    for p in xa + xb:
        assert summed.contains(p, tol=1e-7)
    # Trace optimality: the chosen kappa minimises trace over a sweep.
    best = min(
        np.trace((1 + 1 / k) * e1.shape + (1 + k) * e2.shape) for k in np.linspace(0.05, 20.0, 4000)
    )
    assert np.trace(summed.shape) <= best + 1e-6


def test_bounding_ellipsoid_of_box_contains_corners():
    lb = np.array([-1.0, -0.5, 2.0])
    ub = np.array([1.0, 1.5, 3.0])
    ell = bounding_ellipsoid_of_box(lb, ub)
    # All 2^3 corners lie inside (boundary) within tolerance.
    from itertools import product

    for combo in product(*zip(lb, ub)):
        assert ell.contains(np.array(combo), tol=1e-7)


# ---------------------------------------------------------------------------
# Soundness of propagated enclosures — the rigorous-bound invariant
# ---------------------------------------------------------------------------


def _nodes_for(x1, x2, X):
    return {
        "x1 - x2": (x1 - x2, X[:, 0] - X[:, 1]),
        "3*x1 + x2": (x1 * 3.0 + x2, 3 * X[:, 0] + X[:, 1]),
        "x1 * x2": (x1 * x2, X[:, 0] * X[:, 1]),
        "exp(x1)": (x1.apply_unary(jnp.exp), np.exp(X[:, 0])),
        "exp(x1) * x2": (x1.apply_unary(jnp.exp) * x2, np.exp(X[:, 0]) * X[:, 1]),
        "log(2 + x1^2)": ((x1 * x1 + 2.0).apply_unary(jnp.log), np.log(2 + X[:, 0] ** 2)),
        "sin(x1 + x2)": ((x1 + x2).apply_unary(jnp.sin), np.sin(X[:, 0] + X[:, 1])),
        "(x1 - x2)^2": ((x1 - x2) * (x1 - x2), (X[:, 0] - X[:, 1]) ** 2),
    }


def test_propagated_enclosures_are_sound_on_dense_samples():
    rng = np.random.default_rng(3)
    c0 = np.array([1.0, -0.5])
    g = np.array([[0.6, 0.2], [0.1, 0.5]])
    ell_in = Ellipsoid(c0, g @ g.T)
    x1, x2 = forms_from_ellipsoid(ell_in)
    samples = c0[None] + _ball_samples(rng, 2) @ np.linalg.cholesky(ell_in.shape).T
    for name, (form, truth) in _nodes_for(x1, x2, samples).items():
        lo, hi = form.bounds()
        assert (truth >= lo - 1e-9).all(), f"{name}: underestimator shaves the node"
        assert (truth <= hi + 1e-9).all(), f"{name}: overestimator below the node"


def test_ellipsoidal_at_least_as_tight_as_interval_on_every_node():
    rng = np.random.default_rng(4)
    c0 = np.array([1.0, -0.5])
    g = np.array([[0.6, 0.2], [0.1, 0.5]])
    ell_in = Ellipsoid(c0, g @ g.T)
    x1, x2 = forms_from_ellipsoid(ell_in)
    samples = c0[None] + _ball_samples(rng, 2, 2000) @ np.linalg.cholesky(ell_in.shape).T
    for name, (form, _truth) in _nodes_for(x1, x2, samples).items():
        lo, hi = form.bounds()
        ilo, ihi = form.interval_bounds()
        assert (hi - lo) <= (ihi - ilo) + 1e-12, f"{name}: ellipsoidal looser than interval"


def test_correlated_affine_strictly_tighter_than_interval_majority():
    """Acceptance: tighter than forward interval AA on >= 50% of nodes."""
    rng = np.random.default_rng(5)
    trials = 200
    strictly_tighter = 0
    for _ in range(trials):
        k = 3
        cin = rng.standard_normal(k)
        gk = rng.standard_normal((k, k)) * 0.4
        forms = forms_from_ellipsoid(Ellipsoid(cin, gk @ gk.T))
        weights = rng.standard_normal(k)
        comb = EllipsoidalForm(0.0, np.zeros(k))
        for w, f in zip(weights, forms):
            comb = comb + f.scaled(float(w))
        elo, ehi = comb.bounds()
        ilo, ihi = comb.interval_bounds()
        if (ehi - elo) < (ihi - ilo) - 1e-9:
            strictly_tighter += 1
    assert strictly_tighter / trials >= 0.5


def test_independent_box_inputs_have_no_spurious_correlation():
    """A single box coordinate's enclosure is exact (no inflation)."""
    x1, x2 = forms_from_box(np.array([-1.0, 2.0]), np.array([3.0, 5.0]))
    assert x1.bounds() == pytest.approx((-1.0, 3.0), abs=1e-9)
    assert x2.bounds() == pytest.approx((2.0, 5.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Villanueva-style parametric reproduction
# ---------------------------------------------------------------------------


def test_affine_parametric_map_matches_analytic_ellipsoid():
    """For an affine map of an ellipsoidal parameter set, the propagated joint
    enclosure equals the analytic image ellipsoid ``E(A c + b, A Q A^T)`` to
    ~machine precision — the exactness property ellipsoidal calculus is built
    on (the 1e-6 reproduction criterion in its strongest form)."""
    rng = np.random.default_rng(6)
    c0 = np.array([0.5, -1.0])
    g0 = np.array([[0.7, 0.1], [0.2, 0.5]])
    q0 = g0 @ g0.T
    ell_in = Ellipsoid(c0, q0)
    p1, p2 = forms_from_ellipsoid(ell_in)

    # Affine outputs y1 = 2 p1 - p2 + 1, y2 = p1 + 3 p2 - 2.
    y1 = p1 * 2.0 - p2 + 1.0
    y2 = p1 + p2 * 3.0 - 2.0
    joint = joint_ellipsoid([y1, y2])

    a = np.array([[2.0, -1.0], [1.0, 3.0]])
    b = np.array([1.0, -2.0])
    analytic = ell_in.affine_image(a, b)

    assert joint.center == pytest.approx(analytic.center, abs=1e-6)
    assert joint.shape == pytest.approx(analytic.shape, abs=1e-6)
    # And both describe the same support in every direction.
    for _ in range(10):
        s = rng.standard_normal(2)
        assert joint.support(s) == pytest.approx(analytic.support(s), abs=1e-6)


def test_nonlinear_parametric_map_rigorously_contains_reachable_set():
    """A nonlinear parametric map's ellipsoidal enclosure contains the true
    reachable set on > 10^4 samples (rigorous, if no longer exact)."""
    rng = np.random.default_rng(7)
    c0 = np.array([0.2, 0.1])
    g0 = np.array([[0.4, 0.05], [0.05, 0.3]])
    ell_in = Ellipsoid(c0, g0 @ g0.T)
    p1, p2 = forms_from_ellipsoid(ell_in)
    # y1 = exp(p1) + p2,  y2 = p1 * p2.
    y1 = p1.apply_unary(jnp.exp) + p2
    y2 = p1 * p2
    joint = joint_ellipsoid([y1, y2])

    samples = c0[None] + _ball_samples(rng, 2) @ np.linalg.cholesky(ell_in.shape).T
    out = np.column_stack([np.exp(samples[:, 0]) + samples[:, 1], samples[:, 0] * samples[:, 1]])
    inside = np.array([joint.contains(p, tol=1e-6) for p in out])
    assert inside.all(), f"{(~inside).sum()} reachable points fell outside the enclosure"
