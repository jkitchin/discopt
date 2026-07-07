"""Soundness + differential tests for the flagged PSD-on-Q convexity path.

This is the *bound-changing* regime gate (CLAUDE.md §5) for the Phase 4
Q-extraction consumer wired into ``certify_convex``:

``DISCOPT_PSD_QFORM=1`` enables an exact eigenvalue PSD test on the
extracted ``Q`` of a purely quadratic body. Because the body's Hessian is
the *constant* matrix ``2·Q``, this is a rigorous, box-independent
convexity certificate that can prove convexity where the conservative
interval-Hessian + Gershgorin row-sum enclosure abstains.

The tests pin the two soundness obligations of a bound-changing flag:

1. **No mis-certification.** For random *non-convex* (indefinite Q)
   quadratics, the flagged path NEVER returns ``CONVEX`` (and never
   ``CONCAVE`` for non-concave). This is the catastrophic failure mode
   the flag must not have.
2. **Strict tightening / never contradict.** For random quadratics, the
   flag-ON verdict is (a) always a *refinement* of the flag-OFF verdict —
   it only ever turns ``None`` into a verdict, never flips an existing
   ``CONVEX``↔``CONCAVE`` — and (b) always agrees with a direct
   ``numpy.linalg.eigvalsh`` check on the true Q.
3. **Abstention on non-quadratic bodies is identical on/off** — the flag
   only touches the purely quadratic case; everything else falls through
   to the unchanged rigorous path.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.convexity.certificate import certify_convex
from discopt._jax.convexity.lattice import Curvature


@pytest.fixture
def psd_qform_on(monkeypatch):
    monkeypatch.setenv("DISCOPT_PSD_QFORM", "1")


@pytest.fixture
def psd_qform_off(monkeypatch):
    monkeypatch.setenv("DISCOPT_PSD_QFORM", "0")


def _quadratic_body_from_Q(x, Q: np.ndarray):
    """Build the expression ``xᵀ Q x`` from a symmetric matrix ``Q``.

    Uses squares for the diagonal and cross terms ``2·Q[i,j]·x_i·x_j``
    for the strict upper triangle, so the resulting body's Hessian is
    exactly ``2·Q``.
    """
    n = Q.shape[0]
    body = 0.0 * x[0]
    for i in range(n):
        if Q[i, i] != 0.0:
            body = body + float(Q[i, i]) * x[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            coeff = Q[i, j] + Q[j, i]  # = 2·Q[i,j] for symmetric Q
            if coeff != 0.0:
                body = body + float(coeff) * x[i] * x[j]
    return body


def _random_symmetric(rng, n, kind):
    """Random symmetric Q of a requested definiteness ``kind``."""
    A = rng.uniform(-2, 2, size=(n, n))
    S = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(S)
    if kind == "psd":
        w = np.abs(w) + 1e-3
    elif kind == "nsd":
        w = -(np.abs(w) + 1e-3)
    elif kind == "indef":
        # Force at least one positive and one negative eigenvalue.
        if n == 1:
            return None  # cannot be indefinite in 1-D
        w = np.abs(w) + 1e-2
        w[0] = -(abs(w[0]) + 1e-2)
    return (V * w) @ V.T


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(30))
def test_flag_on_never_miscertifies_indefinite(seed, psd_qform_on):
    """The catastrophic case: an indefinite Q must never read as CONVEX/CONCAVE."""
    rng = np.random.default_rng(4000 + seed)
    n = int(rng.integers(2, 6))
    Q = _random_symmetric(rng, n, "indef")
    lam = np.linalg.eigvalsh(Q)
    # Confirm genuinely indefinite (both signs present, away from zero).
    assert lam.min() < -1e-6 and lam.max() > 1e-6

    m = dm.Model("indef")
    x = m.continuous("x", shape=(n,), lb=-1, ub=1)
    body = _quadratic_body_from_Q(x, Q)

    verdict = certify_convex(body, m)
    assert verdict is None, f"indefinite Q (eig {lam}) must NOT be certified; got {verdict}"


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(20))
def test_flag_on_agrees_with_eigvalsh(seed, psd_qform_on):
    """PSD Q -> CONVEX, NSD Q -> CONCAVE, matching a direct eigen check."""
    rng = np.random.default_rng(5000 + seed)
    n = int(rng.integers(1, 6))

    m = dm.Model("psd")
    x = m.continuous("x", shape=(n,), lb=-1, ub=1)

    Q_psd = _random_symmetric(rng, n, "psd")
    body_psd = _quadratic_body_from_Q(x, Q_psd)
    assert certify_convex(body_psd, m) == Curvature.CONVEX

    Q_nsd = _random_symmetric(rng, n, "nsd")
    body_nsd = _quadratic_body_from_Q(x, Q_nsd)
    assert certify_convex(body_nsd, m) == Curvature.CONCAVE


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(30))
def test_flag_on_is_a_refinement_of_flag_off(seed, monkeypatch):
    """ON verdict only ever turns OFF's ``None`` into a verdict; never flips one."""
    rng = np.random.default_rng(6000 + seed)
    n = int(rng.integers(1, 6))
    kind = rng.choice(["psd", "nsd", "indef"])
    Q = _random_symmetric(rng, n, kind)
    if Q is None:
        pytest.skip("1-D cannot be indefinite")

    m = dm.Model("ref")
    x = m.continuous("x", shape=(n,), lb=-1, ub=1)
    body = _quadratic_body_from_Q(x, Q)

    monkeypatch.setenv("DISCOPT_PSD_QFORM", "0")
    off = certify_convex(body, m)
    monkeypatch.setenv("DISCOPT_PSD_QFORM", "1")
    on = certify_convex(body, m)

    # A verdict under OFF must be preserved exactly under ON (never a flip).
    if off is not None:
        assert on == off, f"flag flipped a rigorous verdict {off} -> {on}"
    # ON is at least as decisive as OFF (soundness of the tightening: it
    # only adds verdicts, guaranteed by the eigen agreement test above).
    lam = np.linalg.eigvalsh(Q)
    if lam.min() >= -1e-10:
        assert on == Curvature.CONVEX
    elif lam.max() <= 1e-10:
        assert on == Curvature.CONCAVE
    else:
        assert on is None


@pytest.mark.unit
def test_non_quadratic_body_identical_on_off(monkeypatch):
    """Non-quadratic bodies fall through: verdict is unchanged by the flag."""
    m = dm.Model("nq")
    x = m.continuous("x", shape=(2,), lb=0.5, ub=3.0)
    bodies = [
        dm.exp(x[0]) + x[1] ** 2,  # convex (exp convex + convex sq)
        dm.log(x[0]),  # concave
        x[0] * x[1],  # bilinear, indefinite -> None
        x[0] ** 3,  # cubic -> None
        dm.sin(x[0]),  # neither
    ]
    for body in bodies:
        monkeypatch.setenv("DISCOPT_PSD_QFORM", "0")
        off = certify_convex(body, m)
        monkeypatch.setenv("DISCOPT_PSD_QFORM", "1")
        on = certify_convex(body, m)
        assert off == on, f"flag changed a non-quadratic verdict: {off} -> {on}"


@pytest.mark.unit
def test_flag_default_off_is_conservative(monkeypatch):
    """With no env override, the tight-but-Gershgorin-defeating PSD case abstains."""
    monkeypatch.delenv("DISCOPT_PSD_QFORM", raising=False)
    m = dm.Model("d")
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    # Q with all off-diagonals 0.99: PSD (λ_min ≈ 0.01) but Gershgorin
    # row-sum lower bound is negative, so the default path abstains.
    body = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) + 2 * 0.99 * (
        x[0] * x[1] + x[0] * x[2] + x[1] * x[2]
    )
    assert certify_convex(body, m) is None
    # And ON certifies it.
    monkeypatch.setenv("DISCOPT_PSD_QFORM", "1")
    assert certify_convex(body, m) == Curvature.CONVEX
