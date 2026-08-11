"""Sound box-local G-convexity (convex-transformability) detector.

This is the enabling primitive for the Khajavirad–Michalek–Sahinidis
(2012) *convex-transformable intermediates* program (issue #181, item 1):
recognizing when a nonconvex intermediate ``φ`` is **G-convex** — i.e.
there exists a continuous increasing univariate ``G`` making ``G∘φ``
convex — even though ``φ`` itself is neither convex nor concave.
G-convexity is a *strictly larger* class than the DCP/eigenvalue
convexity the rest of this package recognizes, so detecting it is a
concrete route to tighter *valid* relaxations (a tighter valid dual
bound closes more gaps without ever risking an invalid bound).

The characterization (Fenchel; KMS 2012 §2)
-------------------------------------------
For a twice-differentiable ``φ`` on a convex domain ``C``, ``φ`` is
G-convex iff there is a function ``ρ(x) ≥ 0`` making the **augmented
Hessian**

    ``H(x; ρ) = ∇²φ(x) + ρ(x)·∇φ(x)∇φ(x)ᵀ``

positive semidefinite over ``C``. Because ``∇φ∇φᵀ`` is rank-1 PSD, a
suitable ``ρ ≥ 0`` exists at a point **iff** ``∇²φ`` is PSD on the
subspace orthogonal to ``∇φ`` — the second-order condition for
*pseudoconvexity*, from which the "stationary point ⇒ global minimum"
payoff follows. G-concavity is the same statement for ``-φ``.

Why the sign of ``ρ`` is ``≥ 0``: if ``G∘φ`` is convex for an
increasing ``G`` then ``G'·∇²φ + G''·∇φ∇φᵀ ⪰ 0``; dividing by
``G' > 0`` gives ``ρ = G''/G'``. The *least convexifying* transform
``G*`` is convex (``G'' ≥ 0``), so the least admissible ``ρ₀`` is
nonnegative — the augmented-Hessian test detects exactly the
transforms that could ever tighten a factorable relaxation.

Two regimes, one soundness contract
-----------------------------------
1. :func:`certify_g_convex` — the **sound** box-local certificate. It
   encloses ``∇²φ`` and ``∇φ`` as interval matrices over the box
   (via :func:`interval_hessian`), encloses ``∇φ∇φᵀ`` as an interval
   outer product, and for a *fixed constant* ``ρ ≥ 0`` runs the same
   interval-Gershgorin PSD test the ordinary convexity certificate
   uses on ``H_box + ρ·Outer_box``. Every concrete augmented Hessian
   ``∇²φ(x) + ρ∇φ(x)∇φ(x)ᵀ`` (``x`` in the box) has entries inside
   that interval matrix, so a PSD-certified interior means the
   concrete matrix is PSD — the verdict is a *proof*, never a
   sampling heuristic. A single constant ``ρ`` is a sufficient (not
   necessary) instance of the ``ρ(x)`` freedom, which keeps the
   witness usable by a downstream LP cut (issue #181 item 3).

2. :func:`is_g_convex_pointwise` / :func:`least_convexifying_rho` —
   **floating-point, diagnostic** helpers (exact linear algebra at a
   single point). They are *not* rigorous — an ``eigvalsh`` verdict is
   not outward-rounded — and must never feed a solver bound. They exist
   to reason about / test G-convexity and to seed the constant-``ρ``
   search of the sound certificate.

This module deliberately does **not** hook into node bounding or emit
cuts; that is issue #181 items 2/3 (a bound-*changing* change gated by
the CLAUDE.md §5 graduation panel). Here we ship only the detector.

References
----------
Khajavirad, Michalek, Sahinidis (2012), "Relaxations of factorable
  functions with convex-transformable intermediates," Math. Program.
  Ser. A — §2 (G-convexity, augmented Hessian, least transform ``G*``).
Fenchel (1953), *Convex Cones, Sets and Functions* — augmented-Hessian
  convex-transformability characterization.
Avriel, Diewert, Schaible, Zang (1988), *Generalized Concavity* — the
  pseudoconvexity / restricted-PSD equivalence.
Boyd, Vandenberghe (2004), *Convex Optimization*, §3.1.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt.modeling.core import Expression, Model

from .certificate import certify_convex
from .eigenvalue import gershgorin_lambda_max, gershgorin_lambda_min
from .interval import Interval
from .interval_ad import interval_hessian
from .lattice import Curvature

# Tolerance for accepting "λ ≥ 0" / "λ ≤ 0" despite floating-point slop,
# matching the ordinary convexity certificate (:mod:`certificate`). The
# interval Hessian already outward-rounds, so genuine zero eigenvalues may
# surface as tiny negatives.
_PSD_TOL = 1e-10

# Default fixed-``ρ`` search grid for the sound certificate. The interval
# Gershgorin bound of the augmented matrix is *not* monotone in ``ρ`` (the
# outer-product enclosure widens the off-diagonal Gershgorin radius as fast
# as it lifts the diagonal), so a constant ``ρ`` cannot be found by
# bisection — we probe a geometric grid plus a midpoint-derived estimate and
# accept the first ``ρ`` that certifies. ``0.0`` is included first so the
# certificate degrades gracefully to ordinary convexity.
_DEFAULT_RHO_GRID: tuple[float, ...] = (
    0.0,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    256.0,
    1024.0,
)

# Cap on any ``ρ`` (grid or estimate). A huge ``ρ`` only reflects a gradient
# that nearly vanishes on the box, where the rank-1 lift is numerically
# useless; refusing it keeps the enclosure finite and the search bounded.
_MAX_RHO = 1.0e8


@dataclass(frozen=True)
class GConvexCertificate:
    """A sound box-local G-convexity verdict with its witnessing ``ρ``.

    * ``kind`` — ``"g_convex"`` (``∇²φ + ρ∇φ∇φᵀ ⪰ 0`` on the box) or
      ``"g_concave"`` (``∇²φ - ρ∇φ∇φᵀ ⪯ 0`` on the box).
    * ``rho`` — the nonnegative constant that witnessed the verdict.
      ``rho == 0.0`` means the body is already ordinarily convex/concave
      on the box (G-convexity subsumes convexity); a positive ``rho``
      means the verdict required the augmented (transformation) lift.

    A returned certificate is a *proof* on the box, not a heuristic: the
    concrete augmented Hessian at every point of the box lies inside the
    interval matrix the certificate proved PSD/NSD.
    """

    kind: str
    rho: float

    @property
    def is_convex(self) -> bool:
        return self.kind == "g_convex"

    @property
    def is_concave(self) -> bool:
        return self.kind == "g_concave"

    @property
    def strictly_transformed(self) -> bool:
        """``True`` when the verdict genuinely used the rank-1 lift
        (``ρ > 0``) — i.e. G-convexity beyond ordinary convexity."""
        return self.rho > 0.0


# ──────────────────────────────────────────────────────────────────────
# Floating-point diagnostic helpers (NOT rigorous — see module docstring)
# ──────────────────────────────────────────────────────────────────────


def is_g_convex_pointwise(
    hess: np.ndarray,
    grad: np.ndarray,
    *,
    tol: float = 1e-9,
) -> bool:
    """Exact-at-a-point test: is ``φ`` G-convex where ``∇²φ = hess``,
    ``∇φ = grad``?

    Returns ``True`` iff ``∇²φ`` is PSD on the subspace orthogonal to
    ``∇φ`` — equivalently iff some ``ρ ≥ 0`` makes the augmented Hessian
    PSD (KMS 2012 §2). When ``∇φ = 0`` the rank-1 lift vanishes and the
    condition collapses to ordinary PSD of ``∇²φ``.

    **Floating-point diagnostic only.** This uses ``numpy.linalg.eigvalsh``
    with no outward rounding and must not gate a solver bound; use
    :func:`certify_g_convex` for a rigorous box-local verdict.
    """
    H = np.asarray(hess, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64).ravel()
    n = g.shape[0]
    if H.shape != (n, n):
        raise ValueError(f"hess shape {H.shape} inconsistent with grad length {n}")
    if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
        return False
    Hs = 0.5 * (H + H.T)
    scale = max(1.0, float(np.max(np.abs(Hs))) if Hs.size else 1.0)

    gnorm = float(np.linalg.norm(g))
    if gnorm <= tol * max(1.0, float(np.max(np.abs(g))) if g.size else 1.0):
        # Gradient (numerically) zero: augmented lift is inert, so
        # G-convexity here is exactly ordinary convexity.
        lam = np.linalg.eigvalsh(Hs)
        return bool(lam.min() >= -tol * scale)

    # Project onto g^⊥ with P = I − ĝĝᵀ. ``P Hs P`` has ``g`` in its null
    # space (eigenvalue exactly 0), so its remaining spectrum is the
    # Hessian restricted to g^⊥. "PSD on g^⊥" ⟺ ``λ_min(P Hs P) ≥ 0``.
    gn = g / gnorm
    P = np.eye(n) - np.outer(gn, gn)
    M = P @ Hs @ P
    M = 0.5 * (M + M.T)
    lam = np.linalg.eigvalsh(M)
    return bool(lam.min() >= -tol * scale)


def least_convexifying_rho(
    hess: np.ndarray,
    grad: np.ndarray,
    *,
    tol: float = 1e-9,
    max_rho: float = _MAX_RHO,
) -> Optional[float]:
    """Estimate the least ``ρ₀ ≥ 0`` with ``∇²φ + ρ₀∇φ∇φᵀ ⪰ 0`` at a point.

    Returns ``None`` when ``φ`` is not G-convex at the point (i.e.
    :func:`is_g_convex_pointwise` is ``False``). ``0.0`` is returned when
    ``∇²φ`` is already PSD (ordinary convexity) or ``∇φ`` vanishes.

    Because ``λ_min(∇²φ + ρ∇φ∇φᵀ)`` is nondecreasing in ``ρ`` (adding a
    PSD rank-1 term only lifts eigenvalues, by Weyl), the threshold is
    found by expanding a bracket then bisecting. **Floating-point
    diagnostic only** — used to seed the sound certificate's ``ρ`` search
    and for analysis, never as a rigorous bound.
    """
    if not is_g_convex_pointwise(hess, grad, tol=tol):
        return None
    H = 0.5 * (np.asarray(hess, dtype=np.float64) + np.asarray(hess, dtype=np.float64).T)
    g = np.asarray(grad, dtype=np.float64).ravel()
    scale = max(1.0, float(np.max(np.abs(H))) if H.size else 1.0)
    thresh = -tol * scale

    def lam_min(rho: float) -> float:
        A = H + rho * np.outer(g, g)
        return float(np.linalg.eigvalsh(0.5 * (A + A.T)).min())

    if lam_min(0.0) >= thresh:
        return 0.0
    # Expand an upper bracket where the augmented matrix is PSD.
    hi = 1.0
    while hi <= max_rho and lam_min(hi) < thresh:
        hi *= 4.0
    if hi > max_rho:
        # Should not happen when the pointwise test passed, but guard so a
        # near-singular restricted Hessian never returns a bogus value.
        return None
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if lam_min(mid) >= thresh:
            hi = mid
        else:
            lo = mid
    return float(hi)


# ──────────────────────────────────────────────────────────────────────
# Sound interval outer product ∇φ∇φᵀ
# ──────────────────────────────────────────────────────────────────────


def _interval_outer(grad: Interval) -> Interval:
    """Sound interval enclosure of the outer product ``∇φ∇φᵀ``.

    Off-diagonal ``(i, j)`` entries use the generic interval product
    ``grad_i · grad_j``. Diagonal entries use the *tight* square
    enclosure ``grad_i²`` (``≥ 0``, and strictly tighter than the generic
    product whenever ``grad_i`` straddles zero), which matters because the
    diagonal is exactly what Gershgorin leans on for the PSD lower bound.
    Every concrete ``∇φ(x)∇φ(x)ᵀ`` on the box lies entrywise inside the
    returned matrix, so it is a valid enclosure.
    """
    lo = np.asarray(grad.lo, dtype=np.float64).ravel()
    hi = np.asarray(grad.hi, dtype=np.float64).ravel()
    col = Interval(lo[:, None], hi[:, None])
    row = Interval(lo[None, :], hi[None, :])
    outer = col * row  # (n, n) sound enclosure of grad_i · grad_j
    sq = grad**2  # tight per-component square, lo ≥ 0
    out_lo = np.array(outer.lo, dtype=np.float64, copy=True)
    out_hi = np.array(outer.hi, dtype=np.float64, copy=True)
    np.fill_diagonal(out_lo, np.asarray(sq.lo, dtype=np.float64).ravel())
    np.fill_diagonal(out_hi, np.asarray(sq.hi, dtype=np.float64).ravel())
    return Interval(out_lo, out_hi)


def _rho_candidates(
    hess: Interval,
    grad: Interval,
    rho_grid: Optional[tuple[float, ...]],
    max_rho: float,
) -> list[float]:
    """Ordered, de-duplicated list of nonnegative ``ρ`` values to probe.

    Combines the fixed geometric grid with a midpoint-derived estimate
    (and a couple of multiples of it), so the search adapts to the box's
    actual curvature/gradient scale while remaining a bounded, sound
    enumeration — every candidate is verified rigorously downstream.
    """
    grid = list(_DEFAULT_RHO_GRID if rho_grid is None else rho_grid)
    est = least_convexifying_rho(hess.mid, grad.mid, max_rho=max_rho)
    if est is not None and est > 0.0:
        grid.extend([est, 1.5 * est, 3.0 * est])
    seen: set[float] = set()
    out: list[float] = []
    for r in grid:
        rf = float(r)
        if not np.isfinite(rf) or rf < 0.0 or rf > max_rho:
            continue
        key = round(rf, 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(rf)
    return out


# ──────────────────────────────────────────────────────────────────────
# Sound box-local certificate
# ──────────────────────────────────────────────────────────────────────


def certify_g_convex(
    expr: Expression,
    model: Model,
    box: Optional[dict] = None,
    *,
    rho_grid: Optional[tuple[float, ...]] = None,
    max_rho: float = _MAX_RHO,
    tol: float = _PSD_TOL,
) -> Optional[GConvexCertificate]:
    """Sound box-local G-convexity / G-concavity verdict, or ``None``.

    Args:
        expr: A scalar expression (the factorable intermediate ``φ``).
        model: The model defining the variable layout.
        box: Optional ``{Variable: Interval}`` overriding declared bounds
            — the tightened node/FBBT box when available.
        rho_grid: Optional override of the fixed ``ρ`` probe grid.
        max_rho: Upper cap on any probed ``ρ``.
        tol: PSD/NSD acceptance tolerance (matches the ordinary
            convexity certificate).

    Returns:
        * :class:`GConvexCertificate` with ``kind="g_convex"`` when some
          constant ``ρ ≥ 0`` makes the interval augmented Hessian
          provably PSD on the box.
        * :class:`GConvexCertificate` with ``kind="g_concave"`` when some
          ``ρ ≥ 0`` makes it provably NSD.
        * ``None`` — a deliberate abstention (indefinite, unsupported
          atoms, unbounded enclosure, or Gershgorin too loose for every
          probed ``ρ``). The caller must treat abstention as "not
          G-convex".

    Soundness: for a fixed ``ρ``, ``H_box + ρ·Outer_box`` is an interval
    matrix containing every concrete ``∇²φ(x) + ρ∇φ(x)∇φ(x)ᵀ`` on the
    box; interval Gershgorin certifies *all* of them PSD/NSD at once, so
    the verdict cannot be a false positive. ``rho=0`` reduces exactly to
    the ordinary interval-Hessian convexity test.
    """
    try:
        ad = interval_hessian(expr, model, box=box)
    except ValueError:
        # Array-variable bodies / oversized DAGs (IntervalHessianTooLarge)
        # — abstain rather than guess, exactly as the convexity certificate.
        return None

    H = ad.hess
    g = ad.grad
    if not (np.all(np.isfinite(H.lo)) and np.all(np.isfinite(H.hi))):
        return None
    if not (np.all(np.isfinite(g.lo)) and np.all(np.isfinite(g.hi))):
        return None

    # Prefer the *simplest* verdict: ordinary curvature (``ρ = 0``) is
    # reported before any transformation lift, and in either direction.
    # This matters because ordinarily concave functions (e.g. ``log``) are
    # also G-convex via some increasing convex ``G`` (``G=exp`` gives a
    # linear composite) — reporting ``g_concave, ρ=0`` for them is far more
    # informative than an escalated ``g_convex, ρ>0``.
    #
    # Delegate the ρ=0 test to the full-strength ordinary certificate rather
    # than a bare Gershgorin call, so G-convexity is a *proper superset* of
    # convexity: ``certify_convex`` also carries the exact 2×2-Sylvester and
    # rank-1 PSD fast paths, which certify convex bodies (e.g. ``x²/y``) whose
    # entry-wise Gershgorin enclosure alone is too loose. It is sound (returns
    # proofs) and box-aware.
    ordinary = certify_convex(expr, model, box=box)
    if ordinary == Curvature.CONVEX:
        return GConvexCertificate("g_convex", 0.0)
    if ordinary == Curvature.CONCAVE:
        return GConvexCertificate("g_concave", 0.0)

    outer = _interval_outer(g)
    candidates = [r for r in _rho_candidates(H, g, rho_grid, max_rho) if r > 0.0]

    # G-convex: find ρ > 0 with H + ρ·Outer ⪰ 0 on the box.
    for rho in candidates:
        aug = H + outer * Interval.point(rho)
        if gershgorin_lambda_min(aug) >= -tol:
            return GConvexCertificate("g_convex", rho)

    # G-concave: -φ is G-convex ⟺ H - ρ·Outer ⪯ 0 on the box, i.e.
    # λ_max(H - ρ·Outer) ≤ 0. (Outer(-∇φ) = Outer(∇φ), so the same
    # enclosure serves both directions.)
    for rho in candidates:
        aug = H - outer * Interval.point(rho)
        if gershgorin_lambda_max(aug) <= tol:
            return GConvexCertificate("g_concave", rho)

    return None


__all__ = [
    "GConvexCertificate",
    "certify_g_convex",
    "is_g_convex_pointwise",
    "least_convexifying_rho",
]
