"""Sharp Neumaier–Shcherbina margin (#309, ``DISCOPT_NS_SHARP_MARGIN``).

The legacy safe-bound evaluation subtracts a flat ``1e-9``-relative margin, which
on magnitude ~1e5 dual decompositions (the gear4 piece LPs) costs 2.9e-4 of every
certified LP bound — enough to push the bound below the certification threshold
at the optimum. The sharp path replaces it with a provably sufficient
forward-error bound (Higham dot-product gammas from the actual data, interval
corners on sign-uncertain reduced costs) and additionally *abstains* when a
sign-uncertain column has an unbounded side after FBBT (a case the legacy path
silently scores as 0, which no flat margin can cover).

Contract under test:
  * sharp is never looser than legacy (when both return a value);
  * sharp never exceeds the true LP optimum (oracle: scipy HiGHS);
  * sharp abstains on sign-uncertain columns with an unbounded side;
  * flag off → dispatcher returns the legacy value bit-for-bit.
"""

from __future__ import annotations

import contextlib

import discopt.solvers.milp_simplex as ms
import numpy as np
import pytest
import scipy.optimize as sopt
import scipy.sparse as sp
from discopt.solver_tuning import SolverTuning, reset_current, set_current


@contextlib.contextmanager
def tuning_context(t):
    token = set_current(t)
    try:
        yield
    finally:
        reset_current(token)


def _std_form(A, b, c, lo, hi):
    m = A.shape[0]
    a_std = sp.hstack([sp.csc_matrix(A), sp.identity(m, format="csc")], format="csc")
    return (
        a_std,
        np.concatenate([c, np.zeros(m)]),
        np.concatenate([lo, np.zeros(m)]),
        np.concatenate([hi, np.full(m, np.inf)]),
    )


def _random_lp(rng):
    n = int(rng.integers(2, 12))
    m = int(rng.integers(1, 10))
    A = rng.normal(size=(m, n)) * (10.0 ** rng.integers(-3, 6))
    A[rng.random(size=A.shape) < 0.4] = 0.0
    b = rng.normal(size=m) * (10.0 ** rng.integers(-2, 5))
    c = rng.normal(size=n) * (10.0 ** rng.integers(-2, 4))
    lo = rng.normal(size=n) * 10
    hi = lo + np.abs(rng.normal(size=n)) * (10.0 ** rng.integers(0, 4))
    lo[rng.random(size=n) < 0.15] = -np.inf
    hi[rng.random(size=n) < 0.15] = np.inf
    return A, b, c, lo, hi


def test_sharp_no_looser_than_std_and_below_oracle_optimum():
    rng = np.random.default_rng(20260716)
    evaluated = 0
    for _ in range(200):
        A, b, c, lo, hi = _random_lp(rng)
        r = sopt.linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
        if r.status != 0:
            continue
        a_std, c_std, lb_std, ub_std = _std_form(A, b, c, lo, hi)
        y = -r.ineqlin.marginals
        for ysign in (1.0, -1.0):  # wrong-sign duals must stay sound too
            std = ms._safe_lp_lower_bound_std(ysign * y, c_std, a_std, b, lb_std, ub_std)
            shp = ms._safe_lp_lower_bound_sharp(ysign * y, c_std, a_std, b, lb_std, ub_std)
            if shp is None:
                continue
            evaluated += 1
            scale = 1.0 + abs(r.fun)
            assert shp <= r.fun + 1e-8 * scale, "sharp bound above the true LP optimum"
            if std is not None:
                assert shp >= std - 1e-12 * scale, "sharp bound looser than legacy"
    assert evaluated >= 50  # the fuzz actually exercised the sharp path


def test_sharp_recovers_flat_margin_loss_at_large_magnitude():
    # The gear4 mechanism, distilled: a dual decomposition whose |contrib| terms
    # sum to ~1e5 (here 1000 fixed columns, each contributing exactly +100).
    # The legacy flat margin loses 1e-9 * 1e5 = 1e-4 of the bound; the sharp
    # forward-error margin loses O(n·u·magnitude) ~ 1e-8. All quantities are
    # exact in float64, so g equals the true optimum and the losses are pure
    # margin.
    n = 1000
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    z = 100.0 * signs  # fixed variables: lb == ub == z
    a_std = sp.csc_matrix(np.ones((1, n)))
    b = np.array([float(z.sum())])  # == 0, keeps A z = b consistent
    y = np.array([7.0])
    c_std = 7.0 + signs  # rc = c - A^T y = signs (exactly +/-1)
    lb_std = ub_std = z
    opt = float(np.dot(c_std, z))  # the only feasible point
    std = ms._safe_lp_lower_bound_std(y, c_std, a_std, b, lb_std, ub_std)
    shp = ms._safe_lp_lower_bound_sharp(y, c_std, a_std, b, lb_std, ub_std)
    assert std is not None and shp is not None
    assert opt - std > 1e-5, "flat margin unexpectedly tight — test premise broken"
    assert opt - shp < 1e-7, "sharp margin did not recover the flat-margin loss"
    assert shp <= opt


def test_sharp_abstains_on_uncertain_sign_with_unbounded_side():
    # z0 - z1 objective on {z0 = z1, z >= 0}: rc computes to exactly 0 on both
    # columns (sign uncertain at any nonzero error radius) and both columns are
    # unbounded above — the true box-min is -inf for any true rc != 0, so the
    # sharp path must abstain. The legacy path scores both terms 0 and returns a
    # finite "bound" whose validity rests on the flat margin, which cannot cover
    # an unbounded term.
    a_std = sp.csc_matrix(np.array([[1.0, -1.0]]))
    b = np.array([0.0])
    c_std = np.array([1.0, -1.0])
    lb_std = np.array([0.0, 0.0])
    ub_std = np.array([np.inf, np.inf])
    y = np.array([1.0])
    std = ms._safe_lp_lower_bound_std(y, c_std, a_std, b, lb_std, ub_std)
    shp = ms._safe_lp_lower_bound_sharp(y, c_std, a_std, b, lb_std, ub_std)
    assert std is not None  # documents the legacy behavior this flag corrects
    assert shp is None


def test_dispatcher_follows_tuning_flag():
    A = np.array([[-1e6]])
    b = np.array([-1e5])
    c = np.array([1.0])
    a_std, c_std, lb_std, ub_std = _std_form(A, b, c, np.array([0.0]), np.array([1.0]))
    y = np.array([-1e-6])
    args = (y, c_std, a_std, b, lb_std, ub_std)
    with tuning_context(SolverTuning(ns_sharp_margin=False)):
        off = ms._safe_lp_lower_bound(*args)
    with tuning_context(SolverTuning(ns_sharp_margin=True)):
        on = ms._safe_lp_lower_bound(*args)
    assert off == ms._safe_lp_lower_bound_std(*args)
    assert on == ms._safe_lp_lower_bound_sharp(*args)
    assert on > off


def test_exact_zero_interval_on_infinite_side_scores_zero():
    # A column with c_k = 0 and an all-zero column has rc = 0 with error radius
    # exactly 0 (point interval): contributes exactly 0 even with infinite
    # bounds — must not abstain, must not produce nan.
    a_std = sp.csc_matrix(np.array([[0.0, 1.0]]))
    b = np.array([1.0])
    c_std = np.array([0.0, 1.0])
    lb_std = np.array([-np.inf, 0.0])
    ub_std = np.array([np.inf, 2.0])
    y = np.array([1.0])
    shp = ms._safe_lp_lower_bound_sharp(y, c_std, a_std, b, lb_std, ub_std)
    assert shp is not None
    assert abs(shp - 1.0) < 1e-9  # g = b^T y + 0 = 1 (rc_1 = 0 too)


@pytest.mark.slow
def test_gear4_piece_bound_clears_certification_threshold():
    # End-to-end on the real gear4 piece LPs: with the sharp margin the root
    # dive bound must clear the 1e-4-relative certification threshold at the
    # optimum (1.64326) — the flat margin left it at 1.64314. Regression for
    # the #309 root-solve mechanism.
    import discopt.modeling as dm
    from discopt._jax.integer_ratio import (
        IntegerRatioPartitioner,
        detect_integer_ratio_specs,
    )

    m = dm.from_nl("python/tests/data/minlplib/gear4.nl")
    specs = detect_integer_ratio_specs(m)
    assert specs
    part = IntegerRatioPartitioner(m, specs)
    lb0 = np.array([v.lb for v in m._variables], dtype=np.float64)
    ub0 = np.array([v.ub for v in m._variables], dtype=np.float64)
    with tuning_context(SolverTuning(ns_sharp_margin=True)):
        bound = part.node_bound(lb0, ub0)
    assert bound is not None
    assert bound >= 1.64330, f"sharp piece bound {bound} below certification threshold"
    assert bound <= 1.6434284740  # never above the true deviation
