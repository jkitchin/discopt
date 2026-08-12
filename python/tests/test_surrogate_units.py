"""Unit tests for the numerics in :mod:`discopt.solvers.surrogate`.

Fast and model-free: everything here drives the surrogates, the acquisition and
the search bookkeeping on plain numpy arrays. Nothing solves an NLP, and the two
tests that build a discopt acquisition model only *build* it — solving is the
integration suite's job.

Every property is checked against its **definition** rather than against a second
implementation of the same formula, which is the only way a test of a closed form
can fail for the right reason:

* the kriging predictor is checked against the textbook ``R⁻¹`` expressions, not
  against the Cholesky-whitened ones the module actually evaluates;
* expected improvement is checked against numerical integration of
  ``E[max(f_min - Y, 0)]``;
* the lifting identity the certified acquisition rests on — ``EI = max_u [dΦ(u) +
  sφ(u)]`` — is checked against a brute-force maximization over ``u``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import math

import numpy as np
import pytest
from discopt.solvers.surrogate import (
    AcquisitionNotExpressible,
    KrigingSurrogate,
    RBFSurrogate,
    _rbf_phi,
    _SurrogateSearch,
    build_cors_model,
    build_ei_model,
    expected_improvement,
)

pytestmark = pytest.mark.unit


def _design(m: int, n: int, seed: int = 0) -> np.ndarray:
    """A well-spread design in the unit box (stratified, not clumped)."""
    rng = np.random.default_rng(seed)
    Z = np.empty((m, n))
    for j in range(n):
        strata = (np.arange(m) + rng.uniform(size=m)) / m
        rng.shuffle(strata)
        Z[:, j] = strata
    return Z


# ── RBF surrogate ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kernel", ["cubic", "thin_plate", "linear"])
def test_rbf_interpolates_its_training_points(kernel):
    """The defining property of an interpolant: ``s(x_i) = y_i``, exactly.

    This is the cheapest real check that the saddle-point system is assembled and
    solved correctly. A wrong sign in the ``Pᵀ`` block, a tail of the wrong
    length, or a kernel evaluated on squared distances instead of distances all
    show up here and nowhere cheaper.
    """
    Z = _design(14, 2, seed=1)
    y = np.sin(3 * Z[:, 0]) + Z[:, 1] ** 2
    rbf = RBFSurrogate(kernel=kernel).fit(Z, y)
    np.testing.assert_allclose(rbf.predict(Z), y, atol=1e-8)
    assert not rbf.used_least_squares


def test_rbf_reproduces_a_linear_function_through_its_tail():
    """A linear function is carried entirely by the tail: ``λ = 0``.

    This is what the linear tail is *for*. Without it (or with it mis-assembled)
    the kernel terms have to bend to fit a trend they cannot represent, the
    coefficients blow up, and the surrogate extrapolates back to the mean outside
    the design instead of continuing the trend.
    """
    Z = _design(20, 3, seed=2)
    coef = np.array([2.0, -1.5, 0.25])
    y = Z @ coef + 7.0
    rbf = RBFSurrogate().fit(Z, y)

    assert np.max(np.abs(rbf.lam)) < 1e-8, f"a linear function needs no kernel terms: {rbf.lam}"
    # Exact at points the fit never saw, which a pure interpolation check cannot
    # distinguish from an overfit.
    fresh = _design(11, 3, seed=99)
    np.testing.assert_allclose(rbf.predict(fresh), fresh @ coef + 7.0, atol=1e-8)


def test_rbf_ridge_relaxes_the_interpolation():
    """``ridge > 0`` (the RBF's nugget) must stop forcing the surface through noise.

    Same reason kriging needs a nugget: an exact interpolant of noisy data is a
    model of the noise. The test states the *direction* — a residual appears and
    grows with the ridge — rather than a magnitude, because the magnitude depends
    on the data.
    """
    rng = np.random.default_rng(3)
    Z = _design(25, 2, seed=4)
    clean = Z[:, 0] + Z[:, 1]
    noisy = clean + 0.1 * rng.normal(size=Z.shape[0])

    exact = RBFSurrogate(ridge=0.0).fit(Z, noisy)
    smooth = RBFSurrogate(ridge=1e-2).fit(Z, noisy)
    heavy = RBFSurrogate(ridge=1.0).fit(Z, noisy)

    r_exact = np.max(np.abs(exact.predict(Z) - noisy))
    r_smooth = np.max(np.abs(smooth.predict(Z) - noisy))
    r_heavy = np.max(np.abs(heavy.predict(Z) - noisy))
    assert r_exact < 1e-8, "ridge=0 must still interpolate"
    assert r_smooth > 1e-6, "a ridge must leave a residual, not interpolate"
    assert r_heavy > r_smooth, (r_smooth, r_heavy)


def test_rbf_refuses_a_design_it_cannot_be_fitted_from():
    """Fewer than ``n+1`` points cannot determine a linear tail — refuse, don't guess."""
    with pytest.raises(ValueError, match="at least n\\+1"):
        RBFSurrogate().fit(_design(3, 4, seed=5), np.zeros(3))


def test_rbf_singular_design_falls_back_to_least_squares_loudly(caplog):
    """A duplicated design point makes the system singular; that must be visible.

    Two identical rows make ``Φ`` exactly singular. Silently returning garbage
    coefficients would give a surrogate that looks fine and means nothing, so the
    fallback is recorded on the object *and* logged (CLAUDE.md §3/§7). The driver
    loop never creates this situation — the evaluation cache refuses duplicates —
    but a caller using ``RBFSurrogate`` directly can.
    """
    Z = _design(8, 2, seed=6)
    Z[3] = Z[2]
    y = np.arange(8, dtype=float)
    with caplog.at_level("WARNING"):
        rbf = RBFSurrogate().fit(Z, y)
    assert rbf.used_least_squares
    assert any("singular" in rec.message for rec in caplog.records), caplog.text


def test_rbf_kernel_takes_the_zero_limit_rather_than_producing_a_nan():
    """``r² log r -> 0`` as ``r -> 0``; a naive ``np.where`` still evaluates ``log(0)``."""
    values = _rbf_phi(np.array([0.0, 0.5, 1.0]), "thin_plate")
    assert np.all(np.isfinite(values))
    assert values[0] == 0.0


# ── kriging surrogate ────────────────────────────────────────────────────────


def _dense_kriging(Z_design, y, theta, nugget, Z_new):
    """Textbook DACE predictor and standard error, written with ``R⁻¹``.

    Deliberately a *different* algebra from the module's Cholesky-whitened form,
    so agreement is evidence rather than a tautology.
    """
    m = Z_design.shape[0]
    R = np.exp(-((np.abs(Z_design[:, None, :] - Z_design[None, :, :]) ** 2) * theta).sum(-1))
    R = R + nugget * np.eye(m)
    Rinv = np.linalg.inv(R)
    one = np.ones(m)
    mu = float(one @ Rinv @ y) / float(one @ Rinv @ one)
    res = y - mu
    sigma2 = float(res @ Rinv @ res) / m
    r = np.exp(-((np.abs(Z_new[:, None, :] - Z_design[None, :, :]) ** 2) * theta).sum(-1))
    y_hat = mu + r @ (Rinv @ res)
    s2 = sigma2 * (
        1.0
        - np.einsum("ij,jk,ik->i", r, Rinv, r)
        + (1.0 - r @ (Rinv @ one)) ** 2 / float(one @ Rinv @ one)
    )
    return y_hat, np.sqrt(np.maximum(s2, 0.0))


def test_kriging_interpolates_its_design_points():
    """``ŷ(x_i) = y_i``: the cheapest real check that the MLE and Cholesky solve are right.

    A wrong sign in ``a = L⁻¹(y - μ1)``, a forgotten transpose in the triangular
    solve, or an ``μ`` computed from the wrong system all break this immediately.
    The default nugget is a 1e-8 conditioning jitter, so the interpolation is
    exact to ~1e-4 of the value spread rather than to machine precision — the
    tolerance below is tied to that, not chosen to make the test pass.
    """
    Z = _design(16, 2, seed=7)
    y = np.cos(4 * Z[:, 0]) + 0.5 * Z[:, 1]
    kr = KrigingSurrogate().fit(Z, y)
    y_hat, _ = kr.predict_with_error(Z)
    assert np.max(np.abs(y_hat - y)) < 1e-4 * (y.max() - y.min()), np.abs(y_hat - y).max()


def test_kriging_standard_error_is_zero_at_samples_and_positive_between():
    """The variance must vanish where the function is known and not where it isn't.

    This is the half of the kriging model that EI actually spends: an error that
    is uniformly zero makes EI zero everywhere and the search stalls at its
    initial design, while an error that stays large at sampled points makes it
    re-sample them forever. Both failure modes are silent in a value-only check.

    "Zero" is stated against the value it provably takes rather than against an
    invented constant: with a nugget ``η`` the correlation to a design point is
    ``1 - η``, so ``s(x_i) = σ√η`` exactly. At the default ``η = 1e-8`` that is
    ``1e-4 σ``, which the probe reproduces to three figures. The design is left
    deliberately sparse in the middle — with a dense design kriging is genuinely
    *not* uncertain between points and the contrast would be small for an
    entirely correct model.
    """
    Z = np.array([0.05, 0.15, 0.25, 0.75, 0.85, 0.95]).reshape(-1, 1)
    y = np.sin(6.0 * Z[:, 0])
    kr = KrigingSurrogate().fit(Z, y)
    sigma_total = kr.y_std * math.sqrt(kr.sigma2)

    _, s_at = kr.predict_with_error(Z)
    _, s_grid = kr.predict_with_error(np.linspace(0.0, 1.0, 201).reshape(-1, 1))

    floor = sigma_total * math.sqrt(kr.fitted_nugget)
    assert np.max(s_at) <= 2.0 * floor, (np.max(s_at), floor)
    assert np.min(s_grid) > 0.0
    assert np.max(s_grid) > 100.0 * np.max(s_at), (np.max(s_grid), np.max(s_at))


def test_kriging_predictor_matches_the_textbook_dense_formula():
    """The whitened form must equal the ``R⁻¹`` form it was derived from.

    The whitening (``v = L⁻¹r``, so ``rᵀR⁻¹r = vᵀv``) is what makes the certified
    EI subproblem tractable at all, so an algebra slip there would quietly change
    the model being maximized. Checked against the dense expressions, evaluated
    at the hyperparameters the fit chose.

    Fitted at ``nugget=1e-4`` on purpose. The comparison is about the *algebra*,
    and the reference is the numerically worse of the two forms — it inverts
    ``R`` explicitly and then computes ``s²`` as ``1 - rᵀR⁻¹r``, a cancellation
    of two quantities that are both ~1. At the default ``1e-8`` jitter
    ``cond(R) ≈ 9e8`` and the two agree only to 3% in ``s``; at ``1e-4``
    (``cond ≈ 1e5``) they agree to 2e-8. Loosening the tolerance instead would
    have hidden a real algebra error behind the conditioning.
    """
    Z = _design(12, 2, seed=9)
    y = np.exp(Z[:, 0]) - Z[:, 1] ** 3
    kr = KrigingSurrogate(nugget=1e-4).fit(Z, y)
    fresh = _design(9, 2, seed=10)

    got_y, got_s = kr.predict_with_error(fresh)
    # The module standardizes y internally; the dense reference works on the same
    # standardized values and is mapped back the same way.
    yn = (y - kr.y_mean) / kr.y_std
    ref_y, ref_s = _dense_kriging(Z, yn, kr.theta, kr.fitted_nugget, fresh)
    np.testing.assert_allclose(got_y, kr.y_mean + kr.y_std * ref_y, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got_s, kr.y_std * ref_s, rtol=1e-6, atol=1e-12)


def test_kriging_nugget_stops_forcing_the_surface_through_measurement_error():
    """A noisy sample set must not be interpolated exactly when a nugget is set.

    Vanilla EGO's interpolation assumption is the thing that dates it: with noisy
    observations it fits the noise, and reports zero uncertainty at points whose
    value it does not actually know. The nugget is the fix, and this pins that it
    is wired through rather than merely accepted as an argument.
    """
    rng = np.random.default_rng(11)
    Z = _design(20, 1, seed=12)
    clean = np.sin(3.0 * Z[:, 0])
    noisy = clean + 0.2 * rng.normal(size=Z.shape[0])

    tight = KrigingSurrogate(nugget=1e-8).fit(Z, noisy)
    loose = KrigingSurrogate(nugget=1e-1).fit(Z, noisy)

    r_tight = np.max(np.abs(tight.predict(Z) - noisy))
    r_loose = np.max(np.abs(loose.predict(Z) - noisy))
    spread = float(noisy.max() - noisy.min())
    assert r_tight < 1e-3 * spread, "the default nugget must still interpolate"
    assert r_loose > 1e-2 * spread, "a large nugget must leave the data unfitted"

    # And the uncertainty at a sampled point stops being zero — that is the whole
    # point: with noise, having measured there does not mean knowing the truth.
    _, s_loose = loose.predict_with_error(Z)
    assert np.min(s_loose) > 0.0


def test_kriging_mle_finds_the_anisotropy_it_is_shown():
    """A function that varies only in one coordinate must earn a larger ``θ`` there.

    ``θ_h`` is an inverse squared length scale, so the active dimension should get
    the larger value. This is the cheapest property test of the likelihood that
    does not just restate the formula: a sign error in ``m ln σ̂² + ln|R|``, or
    optimizing the wrong parameterization, gets the ordering backwards.
    """
    Z = _design(24, 2, seed=13)
    y = np.sin(6.0 * Z[:, 0])  # dimension 1 is pure noise-free irrelevance
    kr = KrigingSurrogate().fit(Z, y)
    assert kr.theta[0] > kr.theta[1], kr.theta


def test_kriging_refuses_rather_than_returning_an_unfitted_model():
    with pytest.raises(ValueError, match="at least 2 design points"):
        KrigingSurrogate().fit(_design(1, 2, seed=14), np.zeros(1))


# ── expected improvement ─────────────────────────────────────────────────────


def _ei_by_quadrature(y_hat: float, s: float, f_min: float, half_width: float = 12.0) -> float:
    """``∫ max(f_min - y, 0) N(y; ŷ, s²) dy`` on a fine grid — the definition."""
    grid = np.linspace(y_hat - half_width * s, y_hat + half_width * s, 400001)
    density = np.exp(-0.5 * ((grid - y_hat) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    return float(np.trapezoid(np.maximum(f_min - grid, 0.0) * density, grid))


@pytest.mark.parametrize(
    ("y_hat", "s", "f_min"),
    [
        (0.0, 1.0, 0.0),  # exactly at the incumbent
        (-2.0, 1.0, 0.0),  # predicted much better
        (3.0, 1.0, 0.0),  # predicted much worse: EI small but nonzero
        (0.5, 0.05, 1.0),  # tiny uncertainty
        (10.0, 4.0, -3.0),  # different scale entirely
    ],
)
def test_expected_improvement_matches_numerical_integration(y_hat, s, f_min):
    """The closed form must equal ``E[max(f_min - Y, 0)]`` computed directly.

    Checked against quadrature of the definition rather than against another
    closed form, because the two ways this goes wrong — ``Φ`` and ``φ`` swapped,
    or ``z`` sign-flipped — both produce a plausible-looking positive function
    that a self-consistency check would accept.
    """
    got = float(expected_improvement(np.array(y_hat), np.array(s), f_min))
    want = _ei_by_quadrature(y_hat, s, f_min)
    assert got == pytest.approx(want, rel=1e-6, abs=1e-12)


def test_expected_improvement_is_monotone_in_the_mean_and_in_the_error():
    """``∂EI/∂ŷ = -Φ(z) < 0`` and ``∂EI/∂s = φ(z) > 0``.

    These two identities are exactly what the certified maximization exploits: a
    point is attractive because it is predicted low *or* because it is unknown.
    If either monotonicity broke, maximizing EI would no longer mean what the
    docstring says it means, however well the optimizer worked.
    """
    s = 1.0
    means = np.linspace(-3.0, 3.0, 41)
    ei_vs_mean = expected_improvement(means, np.full_like(means, s), f_min=0.0)
    assert np.all(np.diff(ei_vs_mean) < 0.0), "EI must strictly decrease in the prediction"

    sigmas = np.linspace(0.05, 3.0, 41)
    ei_vs_sigma = expected_improvement(np.full_like(sigmas, 0.7), sigmas, f_min=0.0)
    assert np.all(np.diff(ei_vs_sigma) > 0.0), "EI must strictly increase in the standard error"


def test_expected_improvement_at_zero_error_is_the_deterministic_improvement():
    """``s = 0`` is a limit to be taken, not a division to be guarded with a fudge.

    Every sampled point has ``s = 0`` in an interpolating model, so this branch is
    hit on every single acquisition evaluation. A NaN here would propagate into
    the ranking silently.
    """
    got = expected_improvement(np.array([-2.0, 0.0, 5.0]), np.zeros(3), f_min=1.0)
    np.testing.assert_allclose(got, [3.0, 1.0, 0.0])
    assert np.all(np.isfinite(got))


def test_expected_improvement_xi_demands_a_larger_improvement():
    """A positive ``xi`` shifts the target below the incumbent, so EI falls."""
    plain = expected_improvement(np.array(0.0), np.array(1.0), f_min=0.0, xi=0.0)
    cautious = expected_improvement(np.array(0.0), np.array(1.0), f_min=0.0, xi=0.5)
    assert float(cautious) < float(plain)


def test_the_lifting_identity_the_certified_acquisition_rests_on():
    """``EI = max_u [d Φ(u) + s φ(u)]`` — checked by brute force over ``u``.

    This identity is the whole reason the EI subproblem can be posed without a
    division: ``z = d/s`` is unbounded wherever ``s -> 0``, but the lifted form
    has no division at all and confines ``u`` to a tight box. If it were false,
    the certified acquisition would be maximizing something other than EI while
    reporting success — the exact failure mode CLAUDE.md §6 is about. Also pins
    that the ``[-8, 8]`` truncation the model uses costs nothing measurable.
    """
    grid = np.linspace(-8.0, 8.0, 400001)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(grid / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * grid**2) / math.sqrt(2 * math.pi)
    checked = 0
    for d in (-3.0, -0.5, 0.0, 0.25, 2.0):
        for s in (0.01, 0.3, 1.0, 5.0):
            lifted = float(np.max(d * cdf + s * pdf))
            closed = float(expected_improvement(np.array(-d), np.array(s), f_min=0.0))
            assert lifted == pytest.approx(closed, rel=1e-8, abs=1e-12), (d, s, lifted, closed)
            checked += 1
    assert checked == 20, checked


# ── search bookkeeping ───────────────────────────────────────────────────────


def _sum_oracle(x):
    return float(np.sum(x)), 0.0


def test_search_rejects_a_non_finite_box():
    """A surrogate is fitted on normalized distances; an infinite side makes them NaN."""
    with pytest.raises(ValueError, match="finite box"):
        _SurrogateSearch(np.array([-np.inf]), np.array([1.0]))
    with pytest.raises(ValueError, match="finite box"):
        _SurrogateSearch(np.array([0.0]), np.array([np.inf]))


def test_integer_coordinates_are_rounded_and_stay_in_bounds():
    """Rounding at evaluation time is what lets an integer model be surrogate-modelled.

    The surrogate is fitted on a continuous relaxation of the domain, but every
    value it is fitted to must come from a genuinely integral point, or the
    reported incumbent is not a solution of the model that was posed.
    """
    s = _SurrogateSearch(
        np.array([2.0, 0.0]), np.array([5.0, 1.0]), integer_mask=np.array([True, False])
    )
    for u in np.linspace(0.0, 1.0, 11):
        x = s.to_model_point(np.array([2.0 + 3.0 * u, u]))
        assert abs(x[0] - round(x[0])) < 1e-12
        assert 2.0 <= x[0] <= 5.0


def test_a_repeated_point_is_refused_rather_than_duplicated_in_the_design():
    """Two identical design rows make ``Φ`` and ``R`` exactly singular.

    The cache here is a correctness guard, not an optimization: a duplicate would
    turn the next fit into a rank-deficient solve. ``evaluate`` returning ``None``
    is how the driver learns that no new information arrived.
    """
    s = _SurrogateSearch(np.zeros(1), np.ones(1))
    calls = {"n": 0}

    def counting(x):
        calls["n"] += 1
        return float(x[0]), 0.0

    assert s.evaluate(np.array([0.5]), counting) is not None
    assert s.evaluate(np.array([0.5]), counting) is None
    assert calls["n"] == 1
    assert len(s.X) == 1
    assert s.stats.cache_hits == 1
    assert s.stats.evals == 1


def test_merit_ranks_violation_first_then_objective():
    """The GLce merit the surrogate is fitted to, in both phases.

    Phase A (nothing feasible known) ranks by total violation so the search first
    hunts for feasibility; phase B denies an infeasible point credit for a low
    objective via the ``|f - f_min|`` term, which needs no penalty weight tuned.
    Same construction as ``_DirectSearch.rank_values`` on purpose — the two
    backends must optimize the same thing to be comparable.
    """
    s = _SurrogateSearch(np.zeros(1), np.ones(1))
    s.X = [np.array([0.1]), np.array([0.2])]
    s.f = [100.0, -100.0]
    s.viol = [5.0, 9.0]
    assert s.best_feasible_value is None
    assert s.merits()[0] < s.merits()[1], "phase A must prefer the less-violating point"

    s.best_feasible_value = 10.0
    s.f = [10.0, -1e6]
    s.viol = [0.0, 3.0]
    assert s.merits()[0] < s.merits()[1], "an infeasible point must not win on objective alone"


def test_merit_maps_an_undefined_objective_to_a_finite_worst_case():
    """A black box that returns nothing must not poison the interpolation system.

    ``+inf`` in the right-hand side makes every coefficient ``nan``; dropping the
    point throws away the one thing it does tell us, which is that the region is
    bad. Mapping it to the worst finite merit plus the observed spread keeps both.
    """
    s = _SurrogateSearch(np.zeros(1), np.ones(1))
    s.X = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
    s.f = [1.0, 3.0, np.inf]
    s.viol = [0.0, 0.0, 0.0]
    s.best_feasible_value = 1.0
    merit = s.merits()
    assert np.all(np.isfinite(merit))
    assert merit[2] > merit.max() - 1e-12
    assert merit[2] > merit[1]


def test_maximin_distance_shrinks_as_the_design_fills_the_box():
    """CORS scales its exclusion radius by this, so it must actually track coverage."""
    rng = np.random.default_rng(15)
    s = _SurrogateSearch(np.zeros(2), np.ones(2))
    s.X = [np.array([0.5, 0.5])]
    sparse = s.maximin_distance(rng)
    s.X = [np.array([a, b]) for a in np.linspace(0.05, 0.95, 8) for b in np.linspace(0.05, 0.95, 8)]
    dense = s.maximin_distance(rng)
    assert dense < sparse, (sparse, dense)


def test_escape_point_is_the_most_distant_candidate_not_a_random_one():
    """When a proposal repeats, the substitute must be the maximin point.

    A random restart would be a coin flip; the maximin point of a stratified pool
    is the best single space-filling addition available without another fit.
    """
    rng = np.random.default_rng(16)
    s = _SurrogateSearch(np.zeros(2), np.ones(2))
    # Everything sampled in one corner: the escape must leave it.
    s.X = [np.array([a, b]) for a in (0.0, 0.05, 0.1) for b in (0.0, 0.05, 0.1)]
    point = s.escape_point(rng)
    assert float(np.min(np.linalg.norm(np.asarray(s.X) - point, axis=1))) > 0.5, point


# ── the acquisition ──────────────────────────────────────────────────────────


def _toy_model(n: int, lb, ub):
    """A tiny model with an opaque objective, used only to mirror a box."""
    import discopt.modeling as dm

    m = dm.Model("toy")
    x = m.continuous("x", shape=n, lb=lb, ub=ub)
    m.minimize(dm.custom(lambda v: v[0], name="toy")(x))
    return m


def test_cors_acquisition_cannot_propose_an_already_sampled_point():
    """The exclusion radius makes a sampled point *infeasible*, not merely unattractive.

    This is the difference between CORS and a distance-*penalized* score, and it
    is not cosmetic: an acquisition that can re-propose a sampled point stalls the
    whole method, because the surrogate does not change and the next iteration
    proposes the same point again. Driven here through the multistart optimizer,
    which is the weaker of the two paths — the algebraic path enforces it as a
    hard constraint.
    """
    rng = np.random.default_rng(17)
    lb, ub = np.zeros(2), np.ones(2)
    s = _SurrogateSearch(lb, ub)
    for z in _design(15, 2, seed=18):
        s.evaluate(z, lambda x: (float(np.sum((x - 0.3) ** 2)), 0.0))

    rbf = RBFSurrogate().fit(s.normalize(np.asarray(s.X)), s.merits())
    delta = 0.25
    proposals = s.propose(
        _toy_model(2, lb, ub),
        "rbf",
        rbf,
        delta=delta,
        xi=0.0,
        optimizer="multistart",
        acq_time_limit=1.0,
        acq_gap_tolerance=1e-4,
        n_multistart=256,
        rng=rng,
    )
    assert len(proposals) == 1
    gap = float(np.min(np.linalg.norm(np.asarray(s.X) - proposals[0], axis=1)))
    assert gap > 0.9 * delta, f"proposal sits {gap} from the design, exclusion radius {delta}"
    assert s.stats.acq_multistart == 1


def test_cors_acquisition_lands_on_the_constrained_surrogate_minimum():
    """With a small exclusion radius the proposal must minimize the *surrogate*.

    The complement of the test above: the distance constraint is a floor on
    exploration, not a ban on exploitation. If this failed, the method would be
    pure space filling wearing a surrogate.

    Scored against a dense grid of the surrogate itself rather than against the
    true function's minimizer. The surrogate is not the function — asserting that
    the proposal lands near the true optimum would be testing the *interpolation
    error* of a 20-point cubic RBF, which is a different claim and one that can
    fail while the optimizer is perfect.
    """
    rng = np.random.default_rng(19)
    lb, ub = np.zeros(2), np.ones(2)
    s = _SurrogateSearch(lb, ub)
    target = np.array([0.8, 0.2])
    for z in _design(20, 2, seed=20):
        s.evaluate(z, lambda x: (float(np.sum((x - target) ** 2)), 0.0))
    rbf = RBFSurrogate().fit(s.normalize(np.asarray(s.X)), s.merits())
    delta = 1e-3
    proposals = s.propose(
        _toy_model(2, lb, ub),
        "rbf",
        rbf,
        delta=delta,
        xi=0.0,
        optimizer="multistart",
        acq_time_limit=1.0,
        acq_gap_tolerance=1e-4,
        n_multistart=256,
        rng=rng,
    )
    axis = np.linspace(0.0, 1.0, 241)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    far_enough = np.min(np.linalg.norm(grid[:, None, :] - np.asarray(s.X)[None], axis=-1), axis=1)
    admissible = grid[far_enough >= delta]
    best_on_grid = float(rbf.predict_standardized(admissible).min())
    got = float(rbf.predict_standardized(proposals[0][None, :])[0])
    assert got <= best_on_grid + 1e-6, (got, best_on_grid)


def test_cors_model_is_algebraic_and_carries_one_constraint_per_design_point():
    """The subproblem handed to B&B must contain no opaque body and every exclusion.

    A ``CustomCall`` leaking into the acquisition model would make it unrelaxable
    and the "certified acquisition" claim vacuous, so this pins the negative
    directly on the expression DAG.
    """
    from discopt.modeling.core import CustomCall

    Z = _design(9, 2, seed=21)
    rbf = RBFSurrogate().fit(Z, np.arange(9, dtype=float))
    lb, ub = np.zeros(2), np.ones(2)
    acq = build_cors_model(_toy_model(2, lb, ub), lb, ub, rbf, delta=0.1)
    assert len(acq._constraints) == 9
    assert not _contains_custom_call(acq._objective, CustomCall)


def test_acquisition_mirrors_every_variable_shape_and_type():
    """The acquisition model must reproduce the source box exactly, whatever its shape.

    The flat column order is what ties the surrogate's design (fitted on
    ``_flat_var_box`` coordinates) to the point the subproblem returns. A scalar
    (``shape ()``), a length-1 vector and a multi-element vector all index
    differently, and an integer that came back continuous would let B&B propose a
    fractional value for a variable the model declared integral — silently, since
    the driver clips and rounds afterwards.
    """
    from discopt.modeling.core import VarType

    source = dm_model_with_mixed_shapes()
    Z = _design(9, 5, seed=23)
    rbf = RBFSurrogate(kernel="linear").fit(Z, np.arange(9, dtype=float))
    lb = np.array([0.0, 0.0, -1.0, 0.0, 0.0])
    ub = np.array([6.0, 6.0, 1.0, 1.0, 1.0])
    acq = build_cors_model(source, lb, ub, rbf, delta=0.05)

    assert [v.name for v in acq._variables] == [v.name for v in source._variables]
    assert [v.shape for v in acq._variables] == [v.shape for v in source._variables]
    assert sum(v.size for v in acq._variables) == lb.size
    integral = {v.name for v in acq._variables if v.var_type is VarType.INTEGER}
    assert integral == {"a", "d"}, integral


def dm_model_with_mixed_shapes():
    """A model exercising ``shape ()``, ``shape (1,)``, a vector and a binary."""
    import discopt.modeling as dm
    import jax.numpy as jnp

    m = dm.Model("mixed")
    a = m.integer("a", lb=0, ub=6)
    b = m.continuous("b", lb=0.0, ub=6.0)
    c = m.continuous("c", shape=(1,), lb=-1.0, ub=1.0)
    d = m.binary("d", shape=2)
    m.minimize(
        dm.custom(lambda p, q, r, s: jnp.cos(p) + q + r[0] + s[0], name="opaque")(a, b, c, d)
    )
    return m


def test_ei_model_is_algebraic_and_refuses_a_power_it_cannot_express():
    """``|z|^p`` has no discopt intrinsic for ``p != 2`` — refuse rather than approximate.

    Silently substituting ``(z²)^{p/2}`` would mean the certified subproblem
    optimizes a *different* model from the one that was fitted, and would report
    a certificate for it. That is exactly the silent approximation CLAUDE.md §3
    forbids.
    """
    from discopt.modeling.core import CustomCall

    Z = _design(10, 2, seed=22)
    y = np.sin(3 * Z[:, 0])
    kr = KrigingSurrogate().fit(Z, y)
    lb, ub = np.zeros(2), np.ones(2)
    acq = build_ei_model(_toy_model(2, lb, ub), lb, ub, kr, f_min=float(y.min()))
    assert not _contains_custom_call(acq._objective, CustomCall)
    # r_i, v_i and the s/u lifts, plus the mirrored x.
    assert {v.name for v in acq._variables} >= {"x", "_r", "_v", "_s", "_u"}

    kr.fitted_power = 1.5
    with pytest.raises(AcquisitionNotExpressible, match="power"):
        build_ei_model(_toy_model(2, lb, ub), lb, ub, kr, f_min=float(y.min()))


def _contains_custom_call(node, custom_type) -> bool:
    """True if ``node``'s expression DAG holds a ``CustomCall`` anywhere.

    Written against ``__dict__`` rather than a node-type table so that a new
    expression class cannot quietly slip past it.
    """
    seen: set[int] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, custom_type):
            return True
        for value in getattr(cur, "__dict__", {}).values():
            if isinstance(value, (list, tuple)):
                stack.extend(value)
            else:
                stack.append(value)
    return False
