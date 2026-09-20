"""The convex-objective PSD gate must not be fooled by the Hessian's scale (#1397).

``_objective_is_convex_quadratic`` decided convexity with ``eigvalsh(H).min() >=
1e-6`` — an absolute constant against a quantity that carries the problem's scale.
``eigvalsh``'s own error on a symmetric ``H`` is O(eps*||H||), so the constant
stops meaning anything once ``||H|| >~ 1e10``: the computed minimum eigenvalue of
an *indefinite* Hessian drifts positive by more than the margin and the objective
is declared convex. The verdict gates the supporting-hyperplane node bound, so a
false positive emits a lower bound that does not underestimate the objective — a
false dual bound, the class CLAUDE.md §1 calls a regression "full stop".

Measured over a sweep in ``||H||`` with one *known* negative eigenvalue (2190
verdicts, ``n <= 32``, ``||H||_F`` from 1e0 to 1e16):

* before: **194** indefinite Hessians declared PSD — e.g. ``||H||_F=6.2e+13`` with
  a true eigenvalue of ``-1e-6`` computing as ``+1.921e-03``;
* after: **0**, with **0** genuinely positive-definite Hessians newly declined and
  **0** verdicts changed on a well-scaled Hessian.

The tests below are the sweep, not an instance: the defect is invisible at
``||H|| ~ 1``, which is the only place a fixed-matrix test would have looked.
"""

import inspect

import numpy as np
import pytest
from discopt.solver import (
    _CONVEX_OBJ_PSD_TOL,
    _hessian_is_psd_with_margin,
    _objective_is_convex_quadratic,
)

EPS = float(np.finfo(np.float64).eps)

#: The sweep. The defect lives at the top of this range and nowhere else, which is
#: exactly why it survived: every existing convexity test is an O(1) matrix.
SCALES = (1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16)


def _rotated(ev, rng):
    """A symmetric matrix with *exactly* the eigenvalues ``ev``, in a random basis.

    Built as ``Q diag(ev) Q^T``, so whether it is PSD is known independently of
    anything the solver computes — the oracle is the construction, not a second
    call to the code under test.
    """
    n = len(ev)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    H = Q @ np.diag(np.asarray(ev, dtype=np.float64)) @ Q.T
    return 0.5 * (H + H.T)


@pytest.mark.unit
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("n", [2, 8, 32])
def test_an_indefinite_hessian_is_never_declared_psd_at_any_scale(n, scale):
    """The soundness direction: one negative eigenvalue must refuse the gate.

    ``true_neg`` is swept down to ``-1e-12`` — far below the eigensolver's roundoff
    at the large scales, which is the whole point. The gate does not need to
    *resolve* such an eigenvalue; it needs to decline to certify convexity when it
    cannot, rather than read the roundoff as curvature.
    """
    rng = np.random.default_rng(hash((n, scale)) % (2**32))
    checked = 0
    for true_neg in (-1e-12, -1e-9, -1e-6, -1e-3, -1e-1, -1.0):
        for _ in range(6):
            ev = np.concatenate([[true_neg], rng.uniform(0.5, 1.0, size=n - 1) * scale])
            H = _rotated(ev, rng)
            checked += 1
            assert not _hessian_is_psd_with_margin(H), (
                f"indefinite H (n={n}, ||H||_F={np.linalg.norm(H, 'fro'):.3e}, true min "
                f"eigenvalue {true_neg:.3e}, computed "
                f"{float(np.linalg.eigvalsh(H).min()):+.3e}) was declared PSD — the "
                f"convex-objective lower bound would be emitted on a nonconvex objective"
            )
    assert checked == 36, f"only {checked} verdicts executed — the sweep proved nothing"


@pytest.mark.unit
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("n", [2, 8, 32])
def test_a_well_conditioned_positive_definite_hessian_is_still_accepted(n, scale):
    """The usefulness direction: buying soundness by refusing everything is no fix.

    Every eigenvalue is positive with a condition number of 1e3, i.e. the smallest
    sits far above O(eps*||H||) at every scale in the sweep, so the gate has the
    information it needs and must say yes.
    """
    rng = np.random.default_rng(hash((n, scale, "pd")) % (2**32))
    checked = 0
    for _ in range(6):
        ev = np.concatenate([[1e-3 * scale], rng.uniform(0.5, 1.0, size=n - 1) * scale])
        H = _rotated(np.maximum(ev, 1e-5), rng)
        checked += 1
        assert _hessian_is_psd_with_margin(H), (
            f"genuinely PD H (n={n}, ||H||_F={np.linalg.norm(H, 'fro'):.3e}, min "
            f"eigenvalue {ev.min():.3e}) was declined — the gate is now useless at "
            f"this scale, not merely sound"
        )
    assert checked == 6, f"only {checked} verdicts executed"


@pytest.mark.unit
def test_the_verdict_is_unchanged_on_a_well_scaled_hessian():
    """#1397 changes the yardstick, not the tolerance: an O(1) problem is identical.

    Mixed-sign eigenvalues, so both the accept and the refuse branch are compared,
    and the comparison is against the *old* gate spelled out here literally.
    """
    rng = np.random.default_rng(1397)
    checked = 0
    accepted = 0
    for _ in range(400):
        n = int(rng.integers(2, 9))
        H = _rotated(rng.uniform(-1.0, 1.0, size=n), rng)
        absolute = float(np.linalg.eigvalsh(H).min()) >= _CONVEX_OBJ_PSD_TOL
        checked += 1
        accepted += int(absolute)
        assert _hessian_is_psd_with_margin(H) == absolute, (
            f"verdict changed on a well-scaled H (||H||_F="
            f"{np.linalg.norm(H, 'fro'):.3e}): the old gate said {absolute}"
        )
    assert checked == 400
    # Both branches must actually have occurred, or the parity check is vacuous.
    assert 0 < accepted < 400, f"only one branch exercised: {accepted}/400 accepted"


@pytest.mark.unit
def test_the_margin_is_never_looser_than_the_absolute_floor():
    """A near-zero Hessian is still declined: the floor survives the change.

    The absolute constant also served a second purpose — abstaining on a
    (near-)linear objective whose box bound interval arithmetic already gets
    exactly. Tightening for large ``||H||`` must not have relaxed that.
    """
    rng = np.random.default_rng(31397)
    checked = 0
    for mag in (0.0, 1e-18, 1e-12, 1e-9, 1e-7):
        for _ in range(20):
            H = _rotated(np.full(4, mag), rng)
            checked += 1
            assert not _hessian_is_psd_with_margin(H), (
                f"a Hessian with every eigenvalue {mag:.1e} (below the {_CONVEX_OBJ_PSD_TOL:.0e} "
                f"floor) was declared PSD"
            )
    assert checked == 100, f"only {checked} verdicts executed"


@pytest.mark.unit
def test_the_scale_aware_gate_is_actually_wired_into_the_convexity_check():
    """Guard against the fix being dead code (the #1395 lesson).

    A helper nobody calls measures nothing. The gate site must consume it, and it
    must not still carry a bare comparison against the absolute constant.
    """
    src = inspect.getsource(_objective_is_convex_quadratic)
    assert "_hessian_is_psd_with_margin" in src, (
        "the convexity check no longer calls the scale-aware gate — the #1397 fix "
        "would be dead code"
    )
    assert "eig_min >= _CONVEX_OBJ_PSD_TOL" not in src, (
        "the bare absolute comparison is still present at the gate site"
    )
    # ...and the helper really does consult the Hessian's magnitude.
    helper = inspect.getsource(_hessian_is_psd_with_margin)
    assert "norm" in helper, "the gate does not look at ||H|| at all"


@pytest.mark.unit
def test_a_knife_edge_hessian_at_large_scale_refuses_rather_than_guesses():
    """The concrete mechanism, stated as a test: roundoff must not read as curvature.

    At ``||H||_F ~ 1e13`` the eigensolver's error is ~1e-3, a thousand times the old
    1e-6 margin, so the sign of a small true eigenvalue is simply not recoverable.
    The only sound verdict is "not certified convex".
    """
    rng = np.random.default_rng(139700)
    seen_positive_computation = 0
    checked = 0
    for _ in range(60):
        ev = np.concatenate([[-1e-9], rng.uniform(0.5, 1.0, size=7) * 1e13])
        H = _rotated(ev, rng)
        computed = float(np.linalg.eigvalsh(H).min())
        checked += 1
        if computed >= _CONVEX_OBJ_PSD_TOL:
            # This is the defect condition: the old gate would have said "convex".
            seen_positive_computation += 1
        assert not _hessian_is_psd_with_margin(H), (
            f"||H||_F={np.linalg.norm(H, 'fro'):.3e}, true min -1e-09, computed "
            f"{computed:+.3e} — declared PSD"
        )
    assert checked == 60
    # If the roundoff never went positive, this test is not exercising the defect
    # and its passing would mean nothing.
    assert seen_positive_computation > 0, (
        "no trial produced a positive computed eigenvalue for an indefinite H — the "
        "probe did not reach the regime it claims to test"
    )
