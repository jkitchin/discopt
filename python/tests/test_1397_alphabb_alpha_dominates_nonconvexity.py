"""The alphaBB ``alpha`` must dominate the true nonconvexity at any scale (#1397).

Both alphaBB routes build the separable underestimator

    q_under(x) = q(x) - Σ_i α_i (x_i - lb_i)(ub_i - x_i).

``q_under ≤ q`` is unconditional for ``α ≥ 0`` — every bracket is nonnegative in
the box. What ``α`` has to buy is the **convexity** of ``q_under``, because only a
convex function has its tangent as a global underestimator and only a convex body
has its box minimum as a valid bound. That requires

    H + 2·diag(α) ⪰ 0    i.e.    α_i ≥ -λ_min(H)/2

against the *true* minimum eigenvalue. Both routes got this wrong, in mirror-image
ways, and both errors are O(u·‖H‖) — so no absolute constant could fix either:

* ``cutting_planes.generate_alphabb_quadratic_oa_cuts_from_evaluator`` takes
  ``λ_min`` from ``eigvalsh``, which can report a minimum *above* the true one by
  ~5.42·u·‖H‖, and covered that with an absolute ``ALPHABB_SAFETY = 1e-6``. That
  suffices only while ``‖H‖ ≲ 1.7e9``. Measured on the shipped generator, α
  recovered from the cut it returns: **84 of 504 cuts invalid**, onset exactly
  where predicted — 0/72 at ``‖H‖_F = 1e9``, 12/72 at 1e10, 35/72 at 1e12, 37/72
  at 1e14. An invalid cut here can exclude points satisfying ``q(x) ≤ 0``: a cut
  that is not a relaxation.

* ``_alphabb_rigorous.rigorous_alpha`` — which feeds the per-node alphaBB *dual
  bound* — reimplemented the interval Gershgorin bound with a plain
  round-to-nearest ``np.sum`` and no outward rounding, so despite its name it was
  not rigorous: rounding the off-diagonal radius down raises the bound above its
  true value and leaves α short. Graded against the same formula in **exact
  rational arithmetic** over the same float entries: **513 of 1120 rows** had α
  provably below ``-λ_min/2``, worst shortfall 5.76e-4. The correct,
  outward-rounded computation already existed one module away; the duplicate is
  gone and both callers now share
  ``convexity.eigenvalue.gershgorin_row_lower_bounds``.

Exact rational arithmetic is the oracle for the second route because every
binary64 is a rational: ``Fraction`` evaluates the Gershgorin formula over the
*same* float entries with no error of its own, so a violation is a proof rather
than an estimate. For the first route the oracle is the construction
``C = V diag(ev) Vᵀ``, so ``eigvalsh`` is never asked to grade itself.
"""

from fractions import Fraction

import numpy as np
import pytest
from discopt._alphabb_rigorous import rigorous_alpha
from discopt._relax.convexity.eigenvalue import (
    gershgorin_lambda_min,
    gershgorin_row_lower_bounds,
)
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_ad import interval_hessian
from discopt._relax.cutting_planes import (
    generate_alphabb_quadratic_oa_cuts_from_evaluator,
)
from discopt._relax.model_utils import flat_variable_bounds
from discopt._relax.nlp_evaluator import NLPEvaluator
from discopt.modeling import Model

#: The scale sweep. The defect is invisible at 1e0, which is the only place a
#: fixed-matrix test would look.
SCALES = (1e0, 1e3, 1e6, 1e9, 1e10, 1e12, 1e14)


def _quadratic_row_model(n, C):
    """``0.5 xᵀ C x ≤ 1`` over ``[-1, 1]ⁿ``, whose body Hessian is exactly ``C``."""
    m = Model("alphabb_row")
    xs = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(n)]
    m.minimize(sum(xs[1:], xs[0]))
    body = 0.0
    for i in range(n):
        body = body + float(C[i, i]) * 0.5 * xs[i] ** 2
        for j in range(i + 1, n):
            body = body + float(C[i, j]) * xs[i] * xs[j]
    m.subject_to(body <= 1.0)
    return m, body


def _spectrum(n, scale, neg_frac, rng):
    """A symmetric ``C`` whose minimum eigenvalue is *exactly* ``-neg_frac·scale``."""
    lam_true = -neg_frac * scale
    ev = np.concatenate([[lam_true], rng.uniform(0.5, 1.0, size=n - 1) * scale])
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    C = V @ np.diag(ev) @ V.T
    return 0.5 * (C + C.T), lam_true


def _exact_row_lower_bounds(h_lo, h_hi):
    """Interval Gershgorin row bounds in exact rational arithmetic."""
    n = h_lo.shape[0]
    out = []
    for i in range(n):
        radius = Fraction(0)
        for j in range(n):
            if j == i:
                continue
            radius += max(abs(Fraction(float(h_lo[i, j]))), abs(Fraction(float(h_hi[i, j]))))
        out.append(Fraction(float(h_lo[i, i])) - radius)
    return out


# --------------------------------------------------------------------------- #
# Route 1: the cut generator.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("n", [2, 3, 5])
def test_the_alphabb_cut_is_a_valid_relaxation_at_every_hessian_scale(n, scale):
    """α recovered from the *returned cut* must dominate the true nonconvexity.

    The generator sets ``under_grad[curved] = jac - α·(lb + ub - 2x*)`` and
    ``generate_oa_cut`` copies that into ``cut.coeffs`` unchanged, so with ``x*``
    off-centre the division recovers exactly the α it used. Recovering it from the
    shipped output — rather than recomputing the generator's arithmetic here — is
    what makes this test grade the code under test and not a copy of it.
    """
    rng = np.random.default_rng(hash(("cut", n, scale)) % (2**32))
    checked = 0
    for neg_frac in (1e-3, 1e-1, 1.0):
        for _ in range(4):
            C, lam_true = _spectrum(n, scale, neg_frac, rng)
            m, _body = _quadratic_row_model(n, C)
            evaluator = NLPEvaluator(m)
            lb, ub = flat_variable_bounds(m)
            x_star = np.full(n, 0.3) + rng.uniform(-0.1, 0.1, size=n)
            cuts = generate_alphabb_quadratic_oa_cuts_from_evaluator(
                evaluator, x_star, lb, ub, constraint_senses=["<="], convex_mask=[False]
            )
            assert len(cuts) == 1, (
                f"the generator declined to cut a row with λ_min={lam_true:.3e} "
                f"(n={n}, scale={scale:.0e}); the sweep would prove nothing"
            )
            jac = np.asarray(evaluator.evaluate_jacobian(x_star), dtype=np.float64)[0]
            denom = lb + ub - 2.0 * x_star
            assert np.all(np.abs(denom) > 1e-6), "x* landed on a box centre"
            recovered = (jac - np.asarray(cuts[0].coeffs, dtype=np.float64)) / denom
            alpha = float(np.median(recovered))
            spread = float(np.max(np.abs(recovered - alpha)))
            assert spread <= 1e-6 * max(1.0, abs(alpha)), (
                f"α is not uniform across the curved block (spread {spread:.3e}); "
                f"the recovery is wrong, not the code under test"
            )
            checked += 1
            assert alpha >= -0.5 * lam_true, (
                f"αBB cut is not a valid relaxation: n={n}, ‖H‖_F="
                f"{np.linalg.norm(C, 'fro'):.3e}, true λ_min={lam_true:+.6e}, "
                f"α={alpha:.17e} < required {-0.5 * lam_true:.17e} "
                f"(short by {-0.5 * lam_true - alpha:.3e}) — q_under is nonconvex, so "
                f"its tangent can cut off points satisfying q(x) ≤ 0"
            )
    assert checked == 12, f"only {checked} cuts graded — the sweep proved nothing"


# --------------------------------------------------------------------------- #
# Route 2: the per-node dual bound.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("scale", [1e0, 1e3, 1e6, 1e9, 1e12])
@pytest.mark.parametrize("n", [4, 8])
def test_rigorous_alpha_is_rigorous_in_exact_arithmetic(n, scale):
    """``rigorous_alpha`` feeds a dual bound, so its α must be provably sufficient.

    Graded against the interval Gershgorin formula in exact rational arithmetic
    over the very same interval-Hessian float entries, so a failure is a proof.
    """
    rng = np.random.default_rng(hash(("ra", n, scale)) % (2**32))
    checked = 0
    for _ in range(4):
        A = rng.uniform(-1.0, 1.0, size=(n, n)) * scale
        A = 0.5 * (A + A.T)
        m, body = _quadratic_row_model(n, A)
        alpha = np.asarray(rigorous_alpha(body, m), dtype=np.float64)
        iad = interval_hessian(body, m)
        exact = _exact_row_lower_bounds(
            np.asarray(iad.hess.lo, dtype=np.float64),
            np.asarray(iad.hess.hi, dtype=np.float64),
        )
        for i in range(n):
            need = -exact[i] / 2 if exact[i] < 0 else Fraction(0)
            checked += 1
            assert Fraction(float(alpha[i])) >= need, (
                f"α[{i}]={alpha[i]:.17e} is provably below the required "
                f"{float(need):.17e} (short by {float(need - Fraction(float(alpha[i]))):.3e}) "
                f"at scale {scale:.0e} — the alphaBB body is nonconvex, so its box "
                f"minimum is not a lower bound and the node bound can exceed the true one"
            )
    assert checked == 4 * n, f"only {checked} rows graded"


@pytest.mark.unit
def test_rigorous_alpha_reports_infinity_for_an_unbounded_row():
    """The documented abstention survives the switch to the shared helper.

    Two things are asserted, because they are reached differently. An *undeclared*
    bound is materialized as the sentinel ``±1e20``, so the interval Hessian stays
    finite and α merely becomes enormous — no infinity is involved. The genuine
    ``-inf`` row bound is reached through the documented ``box`` override, and must
    become ``α = +inf`` ("no useful alphaBB relaxation exists for this box") rather
    than the NaN that ``(+inf) − (+inf)`` would otherwise produce.
    """
    m = Model("unbounded")
    x = m.continuous("x")  # no declared bounds
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x + y)
    # ``x**3`` has Hessian ``6x``, which tracks ``x``'s box — unlike ``x*y``, whose
    # Hessian is the constant ``[[0,1],[1,0]]`` however wide the box.
    sentinel = np.asarray(rigorous_alpha(x**3 + y, m), dtype=np.float64)
    assert sentinel.shape == (2,)
    assert np.all(np.isfinite(sentinel)), (
        f"the sentinel box is finite, so α must be too; got {sentinel!r}"
    )
    assert sentinel[0] > 1e19, f"a ±1e20 box should give an enormous α; got {sentinel!r}"

    alpha = np.asarray(
        rigorous_alpha(x**3 + y, m, {x: Interval(-np.inf, np.inf), y: Interval(-1.0, 1.0)}),
        dtype=np.float64,
    )
    assert not np.any(np.isnan(alpha)), f"α must never be NaN; got {alpha!r}"
    assert np.isinf(alpha[0]), (
        f"a genuinely unbounded row must yield α = +inf, signalling that no useful "
        f"alphaBB relaxation exists; got {alpha!r}"
    )
    assert alpha[1] == 0.0, f"the bounded, linear variable should need no α; got {alpha!r}"


# --------------------------------------------------------------------------- #
# The shared helper.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_row_bounds_are_valid_lower_bounds_in_exact_arithmetic():
    """Every row bound must sit at or below its exact value, at every scale."""
    rng = np.random.default_rng(13974)
    checked = 0
    for n in (2, 4, 7):
        for scale in (1e0, 1e6, 1e12):
            for _ in range(4):
                lo = rng.uniform(-1.0, 1.0, size=(n, n)) * scale
                lo = 0.5 * (lo + lo.T)
                hi = lo + np.abs(rng.uniform(0.0, 0.1, size=(n, n))) * scale
                hi = 0.5 * (hi + hi.T)
                got = gershgorin_row_lower_bounds(Interval(lo, hi))
                exact = _exact_row_lower_bounds(lo, hi)
                for i in range(n):
                    checked += 1
                    assert Fraction(float(got[i])) <= exact[i], (
                        f"row {i} bound {got[i]!r} exceeds its exact value "
                        f"{float(exact[i])!r} at scale {scale:.0e} — not a lower bound"
                    )
    assert checked == 3 * 3 * 4 * (2 + 4 + 7) / 3, f"only {checked} rows graded"


@pytest.mark.unit
def test_lambda_min_is_the_minimum_of_the_row_bounds():
    """The refactor must be bound-neutral: ``λ_min`` is exactly ``rows.min()``."""
    rng = np.random.default_rng(139745)
    checked = 0
    for n in (1, 2, 5, 9):
        for scale in (1e-6, 1e0, 1e9):
            for _ in range(5):
                lo = rng.uniform(-1.0, 1.0, size=(n, n)) * scale
                lo = 0.5 * (lo + lo.T)
                hi = lo + np.abs(rng.uniform(0.0, 0.5, size=(n, n))) * scale
                hi = 0.5 * (hi + hi.T)
                H = Interval(lo, hi)
                checked += 1
                assert gershgorin_lambda_min(H) == float(gershgorin_row_lower_bounds(H).min()), (
                    "λ_min drifted from the per-row bounds it is defined as"
                )
    assert checked == 60, f"only {checked} comparisons executed"


@pytest.mark.unit
def test_an_unbounded_entry_yields_minus_infinity_not_nan():
    """``(+inf) − (+inf)`` must resolve to ``-inf``, the sound bound, never NaN."""
    lo = np.array([[1.0, -np.inf], [-np.inf, 1.0]])
    hi = np.array([[np.inf, np.inf], [np.inf, np.inf]])
    rows = gershgorin_row_lower_bounds(Interval(lo, hi))
    assert not np.any(np.isnan(rows)), f"NaN row bound: {rows!r}"
    assert np.all(np.isneginf(rows)), f"expected -inf for unbounded rows; got {rows!r}"
    # The global entry point keeps its own non-finite guard.
    assert gershgorin_lambda_min(Interval(lo, hi)) == float("-inf")


@pytest.mark.unit
def test_a_diagonal_interval_hessian_keeps_an_exact_zero_bound():
    """The ``_exact0`` guarantee (#957) survives the extraction.

    A diagonal Hessian has no off-diagonal terms, so its off-diagonal row sum is
    an *exact* zero, and a row whose diagonal lower bound is itself zero must come
    back as exactly ``0.0`` — not one subnormal below, which would lose a boundary
    ``λ_min ≥ 0`` verdict. Nonzero rows are still rounded outward by one ULP, which
    is the whole point of the outward rounding and not a defect.
    """
    lo = np.diag([0.0, 2.0, 5.0])
    hi = np.diag([0.0, 3.0, 5.0])
    rows = gershgorin_row_lower_bounds(Interval(lo, hi))
    assert rows[0] == 0.0, f"an exact-zero row bound was perturbed: {rows[0]!r}"
    for i, exact in ((1, 2.0), (2, 5.0)):
        assert rows[i] <= exact, f"row {i} bound {rows[i]!r} is not a lower bound"
        assert rows[i] >= np.nextafter(exact, -np.inf), (
            f"row {i} bound {rows[i]!r} is more than one ULP below {exact!r}"
        )
    assert gershgorin_lambda_min(Interval(lo, hi)) == 0.0
