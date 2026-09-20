"""The G-convexity certificate's PSD slack must scale with the matrix (#1397).

``certify_g_convex`` decided ``λ_min(H ± ρ·Outer) ≥ 0`` against an absolute
``_PSD_TOL = 1e-10``, whose own comment claimed it was "matching the ordinary
convexity certificate". It stopped matching the moment ``certificate.py`` was
made scale-aware earlier in this same audit, and that stale claim is why the
sibling was nearly missed — so this file pins the *class*, not the instance.

An eigenvalue carries its matrix's units, so an absolute licence fails in both
directions at once, and both are swept here:

* **unsound direction** — at small ``‖aug‖`` a fixed ``1e-10`` is enormous. A
  diagonal matrix ``diag(−1e-11, 1e-6, …)`` has an *exact* ``λ_min = −1e-11``
  and a relative nonconvexity of ``1e-5``; the absolute gate certifies it PSD.
  The scaled slack caps admitted relative nonconvexity at ``O(K·u)`` at every
  scale;
* **refusal direction** — at large ``‖aug‖`` the same ``1e-10`` sits far below
  the ``O(u·‖aug‖)`` widening that outward rounding introduces, so a genuine
  zero eigenvalue reads as a small negative and a valid certificate is lost.

The two arms are deliberately run on the *gate as composed in the shipped
function* — ``_psd_slack(aug, tol)`` against ``gershgorin_lambda_min(aug)`` — and
``test_the_shipped_certificate_uses_the_scaled_slack`` asserts that composition
is what ``certify_g_convex`` actually evaluates, so the unit arms cannot drift
away from the code they are pinning. A fourth arm exercises the whole function
end to end across scales.

Why the unsound arm is a unit arm and not an end-to-end one: for
``certify_g_convex`` a *legitimately* G-convex function is supposed to be
certified once ``ρ`` is large enough, so "the function returned g_convex" is not
by itself evidence of a false positive. Isolating the comparison is what makes
the admitted relative nonconvexity measurable at all.
"""

from __future__ import annotations

import inspect
from fractions import Fraction

import numpy as np
import pytest
from discopt._relax.convexity.eigenvalue import (
    gershgorin_lambda_min,
    interval_magnitude,
    psd_decision_slack,
)
from discopt._relax.convexity.g_convexity import _psd_slack, certify_g_convex
from discopt._relax.convexity.interval import Interval

#: Magnitudes of the tested matrix. The small end is where an absolute licence
#: is relatively huge; the large end is where it is relatively invisible.
MAGNITUDES = (1e-8, 1e-6, 1e-3, 1e0, 1e3, 1e6, 1e9, 1e12)

#: The tolerance the shipped code used before #1397, kept only as the baseline
#: this sweep grades against. Nothing in the fix reads it.
HISTORIC_ABSOLUTE_TOL = 1e-10

#: ``psd_decision_slack`` returns ``K·u·magnitude``; admitted relative
#: nonconvexity cannot exceed that ``K·u`` by more than the Gershgorin
#: enclosure's own widening. A generous ceiling — the point is that it is a
#: *constant*, independent of scale, which an absolute tolerance can never give.
RELATIVE_CEILING = 1e-12

U = float(np.finfo(np.float64).eps)


def _diagonal_interval(diag_entries):
    """Degenerate interval matrix ``diag(entries)``.

    Diagonal, so the Gershgorin row bound is *exact* (no off-diagonal radius to
    round outward) and ``λ_min`` is the least entry with no enclosure slop. That
    removes the enclosure from the measurement and leaves only the gate.
    """
    d = np.asarray(diag_entries, dtype=np.float64)
    m = np.diag(d)
    return Interval(m.copy(), m.copy())


def _exact_min_diagonal(entries):
    """``min`` of the entries in exact rational arithmetic.

    Every binary64 is a rational, so ``Fraction`` over the same float entries
    introduces no error of its own and a violation is a proof rather than an
    estimate.
    """
    return min(Fraction(float(e)) for e in entries)


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_admitted_relative_nonconvexity_is_bounded_at_every_scale(magnitude):
    """The gate may never certify a materially nonconvex matrix as PSD."""
    checked = 0
    worst_rel = 0.0
    worst_detail = None

    # Sweep the negative eigenvalue across the decades that bracket both the
    # historic absolute tolerance and the scaled slack at this magnitude.
    for exponent in range(-22, -3):
        delta = 10.0**exponent
        entries = [-delta] + [magnitude] * 3
        aug = _diagonal_interval(entries)

        mag = interval_magnitude(aug)
        slack = _psd_slack(aug, None)
        admitted = gershgorin_lambda_min(aug) >= -slack

        # The oracle: the exact least eigenvalue of a diagonal matrix.
        exact_min = _exact_min_diagonal(entries)
        assert exact_min < 0, "the sweep must actually probe indefinite matrices"

        checked += 1
        if admitted:
            rel = abs(float(exact_min)) / mag if mag > 0 else float("inf")
            if rel > worst_rel:
                worst_rel = rel
                worst_detail = (delta, mag, slack, rel)
            assert rel <= RELATIVE_CEILING, (
                f"at |aug|_F = {mag:.3e} the gate certified a matrix whose exact "
                f"lambda_min is {float(exact_min):.3e} as PSD — a relative "
                f"nonconvexity of {rel:.3e} ({rel / U:.1f}·u), above the "
                f"{RELATIVE_CEILING:.0e} ceiling. slack={slack:.3e}"
            )

    # §6: a sweep that admitted nothing anywhere has not exercised the gate.
    assert checked == 19, f"expected 19 probes at this magnitude, ran {checked}"
    assert worst_rel > 0.0 or magnitude <= 1e-8, (
        f"at |aug|_F ~ {magnitude:.0e} the gate admitted nothing at all, so the "
        "sweep graded no verdict; the probe, not the solver, is broken"
    )
    if worst_detail is not None:
        delta, mag, slack, rel = worst_detail
        assert rel < 1.0


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_historic_absolute_tolerance_was_scale_dependent(magnitude):
    """Pin *why* the fix was needed: the old gate's relative licence tracked 1/‖aug‖.

    This is the falsifiable statement the fix rests on. It grades the historic
    constant directly — no shipped code involved — so it documents the defect
    permanently even if the implementation moves again.
    """
    # The largest exactly-representable nonconvexity the old absolute gate
    # admitted was HISTORIC_ABSOLUTE_TOL, whatever the matrix's magnitude.
    old_relative_licence = HISTORIC_ABSOLUTE_TOL / magnitude
    new_relative_licence = psd_decision_slack(magnitude) / magnitude

    assert new_relative_licence == pytest.approx(32.0 * U, rel=1e-9), (
        "the scaled slack's relative licence must be a scale-free constant; got "
        f"{new_relative_licence:.3e} at magnitude {magnitude:.0e}"
    )

    if magnitude < 1e-3:
        assert old_relative_licence > new_relative_licence * 1e3, (
            f"at magnitude {magnitude:.0e} the absolute tolerance should be "
            f"far looser ({old_relative_licence:.3e}) than the scaled slack "
            f"({new_relative_licence:.3e}) — that is the unsound direction"
        )
    if magnitude >= 1e9:
        assert old_relative_licence < new_relative_licence, (
            f"at magnitude {magnitude:.0e} the absolute tolerance should be "
            f"*stricter* than the arithmetic's own error — that is the refusal "
            f"direction that silently loses certificates"
        )


@pytest.mark.parametrize("magnitude", (1e6, 1e9, 1e12))
def test_a_genuine_zero_eigenvalue_is_still_admitted_at_large_scale(magnitude):
    """The refusal direction: a PSD matrix must stay certified as ‖aug‖ grows.

    A matrix with an exact zero eigenvalue, perturbed only by the outward
    rounding an interval enclosure introduces, is genuinely PSD. The scaled
    slack must absorb that; an absolute ``1e-10`` does not once
    ``u·‖aug‖ > 1e-10``.
    """
    checked = 0
    # A genuine zero eigenvalue, plus the O(u·‖aug‖) negative excursion that
    # outward rounding of an enclosure at this magnitude produces.
    roundoff = 5.0 * U * magnitude
    for factor in (0.1, 0.5, 1.0, 2.0):
        entries = [-roundoff * factor] + [magnitude] * 3
        aug = _diagonal_interval(entries)
        slack = _psd_slack(aug, None)
        checked += 1
        if factor <= 1.0:
            assert gershgorin_lambda_min(aug) >= -slack, (
                f"at |aug|_F ~ {magnitude:.0e} a zero eigenvalue displaced by "
                f"{roundoff * factor:.3e} (={5.0 * factor:.1f}·u·|aug|) was refused; "
                f"slack={slack:.3e}. An absolute tolerance would have refused it "
                f"once u·|aug| exceeded the tolerance."
            )
    assert checked == 4, f"expected 4 probes, ran {checked}"


def test_the_shipped_certificate_uses_the_scaled_slack():
    """Couple the unit arms above to the code they claim to pin.

    Without this, the sweeps could keep passing while ``certify_g_convex`` went
    back to an absolute comparison — the arms would be measuring a helper nobody
    calls.
    """
    src = inspect.getsource(certify_g_convex)
    assert "_psd_slack(aug, tol)" in src, (
        "certify_g_convex no longer evaluates _psd_slack(aug, tol); the unit "
        f"sweeps in this file are no longer pinning the shipped gate. Source:\n{src}"
    )
    # The absolute comparisons the fix replaced must not come back.
    for dead in (">= -tol", "<= tol:"):
        assert dead not in src, f"an absolute {dead!r} comparison has returned to certify_g_convex"
    # The slack must be taken from the augmented matrix, not the raw Hessian:
    # rho ranges up to _MAX_RHO, so aug can dwarf H.
    helper = inspect.getsource(_psd_slack)
    assert "interval_magnitude(aug)" in helper, (
        "_psd_slack must scale by the tested (augmented) matrix's own magnitude"
    )


@pytest.mark.parametrize("scale", (1e0, 1e3, 1e6, 1e9))
def test_certify_g_convex_still_certifies_a_scaled_convex_body(scale):
    """End-to-end: scaling a convex body must not cost it its certificate.

    ``s·(x² + y²)`` is convex for every ``s > 0``, so the ``ρ = 0`` arm must
    report ``g_convex`` at every scale. This is the visible, end-to-end face of
    the refusal direction.
    """
    import discopt.modeling as dm
    from discopt import Model

    m = Model("scaled_convex")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    cert = certify_g_convex(scale * (x**2 + y**2), m)
    assert cert is not None, f"s·(x²+y²) at s={scale:.0e} lost its G-convexity certificate entirely"
    assert cert.kind == "g_convex", f"expected g_convex at s={scale:.0e}, got {cert.kind}"
    assert cert.rho == 0.0, (
        f"a convex body must certify at rho=0, not via a lift; got rho={cert.rho}"
    )
    # Keep the import used so a future edit cannot silently drop the modeling
    # dependency this arm relies on.
    assert dm is not None


def test_the_sweep_spans_the_scales_it_claims():
    """Guard the probe: a sweep collapsed to one magnitude tests nothing."""
    assert min(MAGNITUDES) <= 1e-8, "the small-scale (unsound) end must be probed"
    assert max(MAGNITUDES) >= 1e12, "the large-scale (refusal) end must be probed"
    assert len(set(MAGNITUDES)) == len(MAGNITUDES) >= 6
    # The relative licence must be scale-free — the whole point of the fix.
    licences = {psd_decision_slack(m) / m for m in MAGNITUDES}
    assert len(licences) == 1, f"the scaled slack's relative licence varies: {licences}"
