"""A semidefiniteness verdict must not be decided by an absolute constant (#1397).

Every convexity verdict in ``_relax/convexity`` compared a computed eigenvalue —
or a rigorous bound on one — against an absolute ``_PSD_TOL = 1e-10``. An
eigenvalue carries the units of the matrix, so the constant stops meaning anything
away from ``‖H‖ ~ 1``, and it failed in *both* directions at once. Measured over a
sweep in ``‖Q‖_F`` (2000 verdicts, ``n ≤ 32``, construction-time oracle
``Q = V diag(ev) Vᵀ``):

==========================================  ===============  ==========
quantity                                    absolute 1e-10   scale-aware
==========================================  ===============  ==========
genuinely PSD, exact zero eigenvalue,          149 / 800         **0**
  refused
worst **relative** nonconvexity admitted      1.97e-12          3.49e-15
  on an indefinite matrix                      (8858·u)         (15.7·u)
admissions materially nonconvex (> 100·u)        25              **0**
verdict changed on a well-scaled matrix           —            0 / 400
==========================================  ===============  ==========

The raw count of admitted indefinite matrices *rises* (240 → 452) because the
slack is wider where ``‖Q‖`` is large. That is not a soundness loss and the count
is the wrong axis: the two gates admit different sets, and every matrix the new
one admits is nonconvex by less than 16·u *relative to its own magnitude*, i.e.
below the resolution of evaluating the form at all. The bound error from treating
a form whose minimum eigenvalue is ``-δ`` as convex is at most ``δ·diam(box)²/2``,
and the form's own magnitude over that box is ``~‖Q‖·diam(box)²/2`` — so bounding
``δ/‖Q‖`` bounds the *relative* bound error independently of the box. The absolute
constant bounded nothing: at small ``‖Q‖`` it admitted 8858·u.

End to end, the user-visible half: ``scale·(x - y)²`` is convex at every scale,
and the certificate silently stopped proving it above ``‖H‖ ≈ 1e5``.
"""

import inspect

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import certificate as CERT
from discopt._relax.convexity.certificate import certify_convex
from discopt._relax.convexity.eigenvalue import (
    _PSD_DECISION_K,
    _UNIT_ROUNDOFF,
    interval_magnitude,
    psd_decision_slack,
)
from discopt._relax.convexity.lattice import Curvature
from discopt._relax.quadratic_form import quadratic_is_nsd, quadratic_is_psd

#: The sweep. Every one of these is an ordinary engineering magnitude; the defect
#: is invisible at 1e0, which is the only place a fixed-matrix test would look.
SCALES = (1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14)

#: An admission is *material* when the matrix is nonconvex by more than 100x the
#: arithmetic's own resolution, relative to its own magnitude. Below that, no
#: floating-point test could have distinguished it from semidefinite.
MATERIAL = 100.0 * float(np.finfo(np.float64).eps)


def _sym(ev, rng):
    """A symmetric matrix with *exactly* the eigenvalues ``ev``, in a random basis.

    The oracle is the construction, not a second call to the code under test.
    """
    n = len(ev)
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Q = V @ np.diag(np.asarray(ev, dtype=np.float64)) @ V.T
    return 0.5 * (Q + Q.T)


def _two_var_model():
    m = dm.Model("scaled")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x)
    return m, x, y


# --------------------------------------------------------------------------- #
# The interval-Hessian / Gershgorin path, end to end.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("scale", SCALES)
def test_a_convex_quadratic_is_certified_convex_at_every_scale(scale):
    """``scale·(x - y)²`` is convex for every positive ``scale``.

    Its Hessian has an exact zero eigenvalue, which outward rounding renders as
    ``≈ -u·‖H‖``. Against an absolute 1e-10 that read as "not provably convex"
    above ``‖H‖ ≈ 1e5``: a silent loss of the certificate, and with it every
    downstream route that needs a convex body.
    """
    m, x, y = _two_var_model()
    assert certify_convex(scale * (x - y) ** 2, m) is Curvature.CONVEX, (
        f"a convex quadratic at scale {scale:.0e} was not certified — the "
        f"certificate is scale-limited, not sound-limited"
    )


@pytest.mark.unit
@pytest.mark.parametrize("scale", SCALES)
def test_a_concave_quadratic_is_certified_concave_at_every_scale(scale):
    """The mirror verdict: ``-scale·(x - y)²`` is concave at every scale."""
    m, x, y = _two_var_model()
    assert certify_convex(-(scale * (x - y) ** 2), m) is Curvature.CONCAVE, (
        f"a concave quadratic at scale {scale:.0e} was not certified concave"
    )


@pytest.mark.unit
@pytest.mark.parametrize("scale", SCALES)
def test_an_indefinite_quadratic_is_never_certified_at_any_scale(scale):
    """The soundness direction, and the reason the slack is ``u·‖H‖`` and not more.

    ``scale·x·y`` has eigenvalues ``±scale``: nonconvex by 100% of its own
    magnitude at every scale. A certificate here would license treating it as its
    own convex underestimator.
    """
    m, x, y = _two_var_model()
    verdict = certify_convex(scale * (x * y), m)
    assert verdict is None, (
        f"an indefinite quadratic at scale {scale:.0e} was certified {verdict!s} — "
        f"a relaxation that does not relax"
    )


# --------------------------------------------------------------------------- #
# The exact-Hessian path (``quadratic_is_psd``), swept.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("n", [2, 8, 32])
def test_a_psd_matrix_with_an_exact_zero_eigenvalue_is_accepted_at_every_scale(n, scale):
    """A quadratic in a *subset* of the variables — the ordinary case — is PSD.

    Its Hessian is singular by construction, so the verdict rests entirely on the
    slack being at least the eigensolver's own error at this magnitude.
    """
    rng = np.random.default_rng(hash(("zero", n, scale)) % (2**32))
    checked = 0
    for nzero in (1, max(1, n // 2)):
        for _ in range(5):
            ev = np.concatenate([np.zeros(nzero), rng.uniform(0.5, 1.0, size=n - nzero) * scale])
            Q = _sym(ev, rng)
            checked += 1
            assert quadratic_is_psd(Q) is True, (
                f"a PSD Q (n={n}, ‖Q‖_F={np.linalg.norm(Q, 'fro'):.3e}, {nzero} exact "
                f"zero eigenvalue(s), computed min "
                f"{float(np.linalg.eigvalsh(Q).min()):+.3e}) was refused"
            )
            assert quadratic_is_nsd(-Q) is True, "the NSD mirror disagreed"
            checked += 1
    assert checked == 20, f"only {checked} verdicts executed — the sweep proved nothing"


@pytest.mark.unit
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("n", [2, 8, 32])
def test_no_materially_indefinite_matrix_is_ever_certified_psd(n, scale):
    """Soundness on the axis that decides it: admitted *relative* nonconvexity.

    A matrix may be admitted when its negative eigenvalue is lost in the
    arithmetic — no test can do better — but never when it is materially
    nonconvex next to its own magnitude.
    """
    rng = np.random.default_rng(hash(("indef", n, scale)) % (2**32))
    checked = 0
    worst = 0.0
    for true_neg in (-1e-12, -1e-9, -1e-6, -1e-3, -1e-1, -1.0):
        for _ in range(5):
            ev = np.concatenate([[true_neg], rng.uniform(0.5, 1.0, size=n - 1) * scale])
            Q = _sym(ev, rng)
            checked += 1
            if quadratic_is_psd(Q) is True:
                rel = abs(true_neg) / float(np.linalg.norm(Q, "fro"))
                worst = max(worst, rel)
                assert rel <= MATERIAL, (
                    f"Q (n={n}, ‖Q‖_F={np.linalg.norm(Q, 'fro'):.3e}) was certified "
                    f"PSD though nonconvex by {rel:.3e} relative to its own magnitude "
                    f"({rel / float(np.finfo(np.float64).eps):.0f}·u > 100·u): true min "
                    f"eigenvalue {true_neg:.2e}"
                )
    assert checked == 30, f"only {checked} verdicts executed"


@pytest.mark.unit
def test_the_verdict_is_unchanged_on_a_well_scaled_matrix():
    """#1397 changes the yardstick, not the tolerance: an O(1) matrix is identical.

    Compared against the removed constant spelled out literally. Both branches
    must occur, or the parity check is vacuous.
    """
    rng = np.random.default_rng(1397)
    checked = 0
    accepted = 0
    for _ in range(400):
        n = int(rng.integers(2, 9))
        Q = _sym(rng.uniform(-1.0, 1.0, size=n), rng)
        legacy = bool(float(np.linalg.eigvalsh(Q).min()) >= -1e-10)
        checked += 1
        accepted += int(legacy)
        assert quadratic_is_psd(Q) is legacy, (
            f"verdict changed on a well-scaled Q (‖Q‖_F="
            f"{np.linalg.norm(Q, 'fro'):.3e}): the removed absolute gate said {legacy}"
        )
    assert checked == 400
    assert 0 < accepted < 400, f"only one branch exercised: {accepted}/400 accepted"


@pytest.mark.unit
def test_degenerate_inputs_still_abstain_rather_than_guess():
    """The ``None``/``True`` contract survives the signature change."""
    assert quadratic_is_psd(np.array([[np.nan, 0.0], [0.0, 1.0]])) is None
    assert quadratic_is_nsd(np.array([[np.nan, 0.0], [0.0, 1.0]])) is None
    assert quadratic_is_psd(np.zeros((2, 3))) is None
    assert quadratic_is_psd(np.zeros((0, 0))) is True
    # An exactly-zero Q has magnitude 0, hence slack 0, and is PSD on the nose.
    assert quadratic_is_psd(np.zeros((3, 3))) is True
    assert quadratic_is_psd(np.eye(3)) is True
    assert quadratic_is_nsd(-np.eye(3)) is True
    assert quadratic_is_psd(np.array([[1.0, 0.0], [0.0, -1.0]])) is False


# --------------------------------------------------------------------------- #
# The yardstick itself.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_slack_is_the_arithmetics_own_error_and_nothing_else():
    """``psd_decision_slack`` must be exactly ``K·u·magnitude``, and safe at the edges."""
    checked = 0
    for mag in (1e-12, 1.0, 1e6, 1e12, 1e300):
        assert psd_decision_slack(mag) == pytest.approx(_PSD_DECISION_K * _UNIT_ROUNDOFF * mag)
        checked += 1
    # No slack rather than an infinite licence on a degenerate magnitude.
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert psd_decision_slack(bad) == 0.0, f"psd_decision_slack({bad!r}) handed out slack"
        checked += 1
    # Monotone in the magnitude: a larger matrix never gets a tighter slack.
    mags = [1e-6, 1.0, 1e3, 1e9, 1e15]
    slacks = [psd_decision_slack(m) for m in mags]
    assert slacks == sorted(slacks), f"slack is not monotone in the magnitude: {slacks}"
    checked += 1
    assert checked == 10, f"only {checked} assertions executed"


@pytest.mark.unit
def test_interval_magnitude_dominates_every_concrete_matrix_in_the_box():
    """The magnitude fed to the slack must bound ``‖A‖₂`` for every ``A`` in ``H``."""
    from discopt._relax.convexity.interval import Interval

    rng = np.random.default_rng(13972)
    checked = 0
    for _ in range(50):
        n = int(rng.integers(2, 6))
        lo = rng.uniform(-10.0, 0.0, size=(n, n))
        hi = lo + rng.uniform(0.0, 10.0, size=(n, n))
        mag = interval_magnitude(Interval(lo, hi))
        for _ in range(10):
            A = lo + rng.random((n, n)) * (hi - lo)
            checked += 1
            assert float(np.linalg.norm(A, 2)) <= mag + 1e-9, (
                f"interval_magnitude {mag!r} does not dominate a concrete ‖A‖₂"
            )
    assert checked == 500, f"only {checked} comparisons executed"


@pytest.mark.unit
def test_the_scale_aware_slack_is_wired_into_both_paths():
    """Guard against the fix being dead code (the #1395 lesson).

    Both the exact-Hessian route and the interval/Gershgorin route must consume
    the shared slack, and no bare comparison against the removed absolute
    constant may survive at a verdict site.
    """
    cert_src = inspect.getsource(CERT)
    qf_src = inspect.getsource(quadratic_is_psd)

    assert "psd_decision_slack" in qf_src, (
        "quadratic_is_psd no longer consults the scale-aware slack"
    )
    assert "tol" not in inspect.signature(quadratic_is_psd).parameters, (
        "quadratic_is_psd took back an absolute tolerance parameter, which is the "
        "defect #1397 removed"
    )
    live = [
        ln for ln in cert_src.splitlines() if "_PSD_TOL" in ln and not ln.lstrip().startswith("#")
    ]
    assert not live, f"an absolute _PSD_TOL comparison survives in certificate.py: {live}"
    assert cert_src.count("psd_decision_slack(") >= 2, (
        "certificate.py must scale BOTH the rank-1 coefficient test and the Gershgorin verdicts"
    )
    assert "interval_magnitude(hess)" in cert_src, (
        "the Gershgorin verdict does not scale by the interval Hessian's magnitude"
    )
