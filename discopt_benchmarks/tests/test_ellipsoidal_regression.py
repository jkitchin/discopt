"""M7 regression: ellipsoidal arithmetic is a sound, correlation-aware bound
provider that integrates through the polyhedral-OA wrapper and the full solver.

Issue #81 (M7) asks for *ellipsoidal arithmetic* — propagating ellipsoidal
enclosures ``{x : (x-c)^T P^{-1} (x-c) <= 1}`` through the DAG so that affine
correlations interval arithmetic discards are retained. Acceptance criteria
exercised here as a benchmark-suite regression:

1. **Soundness (rigorous-bound invariant).** Every cut the ellipsoidal provider
   emits through :func:`polyhedral_oa.outer_approximation` encloses the true
   graph ``y = f(x)`` on > 10^4 samples — no cut shaves a feasible point.
2. **Correlation win.** On a correlated-affine combination the ellipsoidal
   (2-norm) enclosure is strictly tighter than the interval/affine-arithmetic
   (1-norm) enclosure, the property a forward interval bound cannot reproduce.
3. **End-to-end correctness.** A full spatial-B&B solve with
   ``relaxation_arithmetic="ellipsoidal"`` returns the same correct global
   optimum as plain McCormick, and never reports a bound above the achieved
   objective (``incorrect_count == 0``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt import Model
from discopt import modeling as dm
from discopt._jax import ellipsoidal_arith as ea
from discopt._jax.polyhedral_oa import outer_approximation

jax.config.update("jax_enable_x64", True)

N_REGRESSION_SAMPLES = 12_000

# Smooth-unary coupling instance:  y = exp(x),  minimize y - 2 x  over a box.
# Unconstrained interior optimum of exp(x) - 2x is at x = ln 2.
X_BOX = (-1.0, 2.0)
Y_BOX = (0.0, 10.0)


def _true_optimum() -> float:
    xs = np.linspace(X_BOX[0], X_BOX[1], 2_000_001)
    ys = np.exp(xs)
    feasible = (ys >= Y_BOX[0]) & (ys <= Y_BOX[1])
    return float((ys - 2.0 * xs)[feasible].min())


def _build_model() -> Model:
    m = Model()
    x = m.continuous("x", lb=X_BOX[0], ub=X_BOX[1])
    y = m.continuous("y", lb=Y_BOX[0], ub=Y_BOX[1])
    m.subject_to(dm.exp(x) - y <= 0.0)
    m.minimize(y - 2.0 * x)
    return m


# ---------------------------------------------------------------------------
# 1. Soundness — every ellipsoidal-provider cut globally encloses the graph
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.regression
def test_ellipsoidal_oa_cuts_are_globally_valid():
    oa = outer_approximation(jnp.exp, X_BOX, "ellipsoidal", degree=10, n_slopes=16)

    rng = np.random.default_rng(0)
    xs = rng.uniform(X_BOX[0], X_BOX[1], size=N_REGRESSION_SAMPLES)
    ys = np.exp(xs)  # the true graph y = f(x)
    for k, cut in enumerate(oa.cuts):
        sx, sy = float(cut.coeffs[0]), float(cut.coeffs[1])
        lhs = sx * xs + sy * ys
        if cut.sense == ">=":
            assert (lhs >= cut.rhs - 1e-9).all(), f"cut[{k}] (>=) shaves the graph"
        else:
            assert (lhs <= cut.rhs + 1e-9).all(), f"cut[{k}] (<=) shaves the graph"

    # The cut family sandwiches the graph: lower envelope <= f <= upper envelope.
    grid = np.linspace(X_BOX[0], X_BOX[1], 4001)
    lo = oa.evaluate_lower(grid)
    hi = oa.evaluate_upper(grid)
    fg = np.exp(grid)
    assert (lo <= fg + 1e-9).all()
    assert (hi >= fg - 1e-9).all()


# ---------------------------------------------------------------------------
# 2. Correlation win — ellipsoidal strictly beats interval AA on a correlated
#    combination (2-norm vs 1-norm radius), the property M7 exists to capture.
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_ellipsoidal_strictly_tighter_than_interval_on_correlated_affine():
    rng = np.random.default_rng(5)
    trials = 200
    strictly_tighter = 0
    for _ in range(trials):
        k = 3
        cin = rng.standard_normal(k)
        gk = rng.standard_normal((k, k)) * 0.4
        forms = ea.forms_from_ellipsoid(ea.Ellipsoid(cin, gk @ gk.T))
        weights = rng.standard_normal(k)
        comb = ea.EllipsoidalForm(0.0, np.zeros(k))
        for w, f in zip(weights, forms, strict=True):
            comb = comb + f.scaled(float(w))
        elo, ehi = comb.bounds()
        ilo, ihi = comb.interval_bounds()
        # Sound: ellipsoidal never looser than interval.
        assert (ehi - elo) <= (ihi - ilo) + 1e-12
        if (ehi - elo) < (ihi - ilo) - 1e-9:
            strictly_tighter += 1
    assert strictly_tighter / trials >= 0.5


# ---------------------------------------------------------------------------
# 3. End-to-end solve correctness — relaxation_arithmetic="ellipsoidal" reaches
#    the same global optimum and never reports an invalid bound.
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize("arithmetic", ["mccormick", "ellipsoidal"])
def test_full_solve_reaches_correct_optimum(arithmetic):
    model = _build_model()
    res = model.solve(
        mccormick_bounds="lp",
        relaxation_arithmetic=arithmetic,
        time_limit=60,
        skip_convex_check=True,
    )
    assert res.status == "optimal"
    true_opt = _true_optimum()
    assert res.objective == pytest.approx(true_opt, abs=1e-4, rel=1e-4)
    # Rigorous-bound invariant: the reported bound never exceeds the optimum.
    assert res.bound <= res.objective + 1e-4
