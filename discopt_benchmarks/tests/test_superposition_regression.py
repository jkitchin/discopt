"""M8 regression: multivariate superposition cuts close the LP gap on a
bilinear-of-nonlinear term beyond what compositional McCormick alone achieves.

Issue #81 (M8) asks for a *superposition* relaxation that is provably at least
as tight as the tightest single component and strictly tighter on at least one
instance, without ever violating the rigorous-bound invariant
(``incorrect_count == 0``: a lower bound never exceeds the true global minimum
within abs=1e-6/rel=1e-4).

The univariate polyhedral-OA wrapper (M11) cannot show a *strict* superposition
win: for a single-variable atom ``y = f(x)`` the convex hull is exactly
achievable, so whichever arithmetic gets closer to it dominates and a union is
never strictly tighter. The strict win requires a genuinely multivariate term
where the components are complementary. The canonical case is the
bilinear-of-nonlinear product

    w = f(x) * y ,   handled compositionally as  u = f(x)  then  envelope(u * y).

discopt's compositional-McCormick envelope is exact on the box boundary but
loose in the interior; the superposition family (:mod:`discopt._relax.superposition`)
adds rigorous interior-reference cuts that close that interior gap.

This regression asserts the three M8 acceptance properties on
``exp(x) * y == k``:

1. **Soundness (rigorous-bound invariant).** Every generated cut bounds the true
   product surface on > 10^4 samples — no cut shaves a feasible point.
2. **Strict LP-gap closure.** Through the real :class:`MccormickLPRelaxer`, the
   superposition-enabled root LP bound strictly exceeds the plain-McCormick LP
   bound, while *both* remain valid lower bounds on the true optimum.
3. **End-to-end correctness.** A full spatial-B&B solve with
   ``relaxation_arithmetic="superposition"`` returns the same correct global
   optimum as plain McCormick — the tighter relaxation never prunes a true
   optimum.

.. warning::
   **Criterion 2 is not currently met, and criterion 3 no longer distinguishes
   the two arithmetics.** :func:`build_milp_relaxation` documents
   ``superposition`` as **IGNORED** on the default path since the #632
   uniform-factorable cutover, and ``solver.py`` maps
   ``relaxation_arithmetic="superposition"`` onto exactly that ignored flag. So
   :mod:`discopt._relax.superposition` is presently reachable only from the
   direct cut-generation tests here (criteria 1 and 4), which is why they stayed
   green while the feature went inert.

   The strict-tightness assertion is therefore split out below as a **strict
   xfail** rather than deleted: it is a real acceptance criterion of issue #81,
   and keeping it means re-wiring the flag shows up as an XPASS instead of
   passing unnoticed. The *soundness* half — both bounds valid, superposition
   never looser — is a separate test that passes today.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from discopt import Model
from discopt import modeling as dm
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt._relax.superposition import (
    BilinearNonlinearTerm,
    bilinear_nonlinear_cuts,
    mccormick_references,
    superposition_references,
)

# Bilinear-of-nonlinear instance with a known strict-win objective direction.
# w = exp(x) * y == K ; minimize C_X*x + C_Y*y over the box.
X_BOX = (-1.0, 1.0)
Y_BOX = (0.5, 2.0)
K = 1.126
C_X, C_Y = -0.793409, -0.608689

N_REGRESSION_SAMPLES = 10_000


def _true_optimum() -> float:
    """Exact global optimum of  min C_X*x + C_Y*y  s.t. exp(x)*y == K, box."""
    xs = np.linspace(X_BOX[0], X_BOX[1], 2_000_001)
    ys = K * np.exp(-xs)  # the equality pins y = K * exp(-x)
    feasible = (ys >= Y_BOX[0]) & (ys <= Y_BOX[1])
    return float((C_X * xs + C_Y * ys)[feasible].min())


def _build_model() -> Model:
    m = Model()
    x = m.continuous("x", lb=X_BOX[0], ub=X_BOX[1])
    y = m.continuous("y", lb=Y_BOX[0], ub=Y_BOX[1])
    m.subject_to(dm.exp(x) * y == K)
    m.minimize(C_X * x + C_Y * y)
    return m


# ---------------------------------------------------------------------------
# 1. Soundness — the rigorous-bound invariant (no cut excludes a true point)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.regression
def test_every_superposition_cut_is_globally_valid():
    term = BilinearNonlinearTerm(lambda t: jnp.exp(t), X_BOX, Y_BOX, degree=10)
    cuts = bilinear_nonlinear_cuts(term, superposition_references(X_BOX, Y_BOX))

    rng = np.random.default_rng(0)
    xs = rng.uniform(X_BOX[0], X_BOX[1], size=N_REGRESSION_SAMPLES)
    ys = rng.uniform(Y_BOX[0], Y_BOX[1], size=N_REGRESSION_SAMPLES)
    ws = np.exp(xs) * ys  # the true product surface
    for k, cut in enumerate(cuts):
        ax, ay, aw = (float(c) for c in cut.coeffs)
        lhs = ax * xs + ay * ys + aw * ws
        if cut.sense == ">=":
            assert (lhs >= cut.rhs - 1e-9).all(), f"cut[{k}] (>=) shaves the surface"
        else:
            assert (lhs <= cut.rhs + 1e-9).all(), f"cut[{k}] (<=) shaves the surface"


# ---------------------------------------------------------------------------
# 2. Strict LP-gap closure through the real relaxer, both bounds still valid
# ---------------------------------------------------------------------------


def _root_lp_bounds() -> tuple[float, float, float]:
    """(plain-McCormick bound, superposition bound, true optimum) at the root."""
    model = _build_model()
    lb = np.array([X_BOX[0], Y_BOX[0]])
    ub = np.array([X_BOX[1], Y_BOX[1]])
    base = MccormickLPRelaxer(model, superposition=False).solve_at_node(lb, ub)
    sup = MccormickLPRelaxer(model, superposition=True).solve_at_node(lb, ub)
    assert base.status == "optimal"
    assert sup.status == "optimal"
    return base.lower_bound, sup.lower_bound, _true_optimum()


@pytest.mark.regression
def test_root_lp_bounds_are_valid_under_both_arithmetics():
    """The soundness half of M8's acceptance criteria -- and it still holds.

    Split out from the strict-tightness assertion below so that a feature going
    inert cannot take a *soundness* check down with it. This half passes today;
    the strict half does not, and conflating them hid that for both.
    """
    base_lb, sup_lb, true_opt = _root_lp_bounds()
    assert base_lb <= true_opt + 1e-6, (base_lb, true_opt)
    assert sup_lb <= true_opt + 1e-6, (sup_lb, true_opt)
    # Superposition may not be *looser* than plain McCormick even if the flag is
    # currently inert -- equality is acceptable, regression is not.
    assert sup_lb >= base_lb - 1e-9, (base_lb, sup_lb)


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    reason=(
        "M8 acceptance criterion not currently met: `build_milp_relaxation` "
        "documents `superposition` as IGNORED since the #632 uniform-factorable "
        "cutover, so `MccormickLPRelaxer(superposition=True)` returns a bound "
        "identical to plain McCormick. Kept as strict xfail rather than deleted "
        "so re-wiring the flag reports as an XPASS instead of passing silently."
    ),
)
def test_superposition_root_lp_strictly_tighter():
    base_lb, sup_lb, _ = _root_lp_bounds()
    assert sup_lb > base_lb + 1e-3, (base_lb, sup_lb)


# ---------------------------------------------------------------------------
# 3. End-to-end solve correctness — the tighter relaxation prunes nothing true
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize("arithmetic", ["mccormick", "superposition"])
def test_full_solve_reaches_correct_optimum(arithmetic):
    """Correctness of the *answer*, not of the proof.

    See the sibling test in ``test_ellipsoidal_regression.py``: asserting
    ``status == "optimal"`` inside a 60 s budget is a performance claim, and a
    load-sensitive one. Both arithmetics return ``feasible`` with an objective
    matching the true optimum to ~1e-9.

    Note that ``arithmetic="superposition"`` is currently a **no-op** on this
    path -- ``build_milp_relaxation`` documents ``superposition`` as ignored
    since the #632 uniform-factorable cutover -- so the two parametrizations
    presently run the identical solve. This test is kept parametrized so it
    starts distinguishing them again the moment the flag is re-wired.
    """
    model = _build_model()
    res = model.solve(
        mccormick_bounds="lp",
        relaxation_arithmetic=arithmetic,
        time_limit=60,
        skip_convex_check=True,
    )
    assert res.status in ("optimal", "feasible"), res.status
    assert res.objective is not None, "no incumbent returned"

    true_opt = _true_optimum()
    assert res.objective == pytest.approx(true_opt, abs=1e-4, rel=1e-4)

    # Rigorous-bound invariant, branched explicitly so it cannot silently
    # not-happen when the budget-limited solve returns no dual bound.
    if res.bound is None:
        assert res.status == "feasible", f"status={res.status} must carry a dual bound; got None"
    else:
        assert res.bound <= res.objective + 1e-4


# ---------------------------------------------------------------------------
# 4. Monotonicity — interior references never loosen the corner-only envelope
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_superposition_never_looser_than_corner_family():
    term = BilinearNonlinearTerm(lambda t: jnp.exp(t), X_BOX, Y_BOX, degree=10)
    corner = bilinear_nonlinear_cuts(term, mccormick_references(X_BOX, Y_BOX))
    full = bilinear_nonlinear_cuts(term, superposition_references(X_BOX, Y_BOX))

    xs = np.linspace(X_BOX[0], X_BOX[1], 121)
    ys = np.linspace(Y_BOX[0], Y_BOX[1], 121)
    grid_x, grid_y = np.meshgrid(xs, ys)

    def envelope(cuts):
        lo = np.full_like(grid_x, -np.inf)
        hi = np.full_like(grid_x, np.inf)
        for cut in cuts:
            ax, ay, _ = cut.coeffs
            plane = cut.rhs - ax * grid_x - ay * grid_y
            if cut.sense == ">=":
                lo = np.maximum(lo, plane)
            else:
                hi = np.minimum(hi, plane)
        return lo, hi

    lo_c, hi_c = envelope(corner)
    lo_f, hi_f = envelope(full)
    # Superposition lower envelope >= corner lower; upper <= corner upper.
    assert (lo_f >= lo_c - 1e-9).all()
    assert (hi_f <= hi_c + 1e-9).all()
    # And strictly tighter somewhere in the interior.
    assert (lo_f > lo_c + 1e-6).any() or (hi_f < hi_c - 1e-6).any()
