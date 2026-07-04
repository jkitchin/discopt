"""Exact quadratic (Q-matrix) coefficient extraction from the expression IR.

This module upgrades the notion of "quadratic" from a *degree check*
(``ExprArena::is_quadratic`` in Rust — ``max_degree <= 2``, a yes/no
predicate) to an *exact coefficient extraction*: given a scalar
expression, recover the symmetric matrix ``Q``, the linear vector ``c``
and the constant ``d`` such that

    expr(x) == xᵀ Q x + cᵀ x + d      (exactly, for all x)

or return ``None`` when the expression is not purely quadratic.

Design rules (binding — see ``docs/dev/certification-gap-plan.md`` §8
Phase 4 item 3, and CLAUDE.md §5 on the two verification regimes):

* **Exact or abstain.** The extraction is a *recognition*. It either
  returns coefficients that reproduce the expression bit-for-bit (up to
  floating-point evaluation order), or it returns ``None``. It NEVER
  returns an approximate ``Q``. A degree-3+ monomial, a transcendental,
  a variable in a denominator, a fractional power, an unsupported atom
  — any of these make the whole expression non-(purely-)quadratic and
  the function abstains.

* **Trusted foundation.** Extraction is layered on the existing,
  tested polynomial walker
  :func:`discopt._jax.milp_relaxation._expr_to_polynomial` (fed the
  :func:`discopt._jax.term_classifier.distribute_products` normal form),
  the same machinery the edge-concave collector already relies on. That
  walker returns ``None`` on any non-polynomial leaf; we additionally
  reject any monomial of degree > 2. The flat variable indexing is the
  identical prefix-sum layout used by the convexity certificate
  (``interval_ad._var_offset`` == ``term_classifier._compute_var_offset``),
  so a ``Q`` produced here is directly consistent with the coordinate
  system :func:`discopt._jax.convexity.certificate.certify_convex` works
  in.

* **Symmetric-split convention.** For a cross term ``b·x_i·x_j`` with
  ``i != j`` we set ``Q[i,j] = Q[j,i] = b/2`` so that the quadratic form
  ``xᵀ Q x`` reproduces ``b·x_i·x_j`` (the form contributes
  ``Q[i,j]·x_i·x_j + Q[j,i]·x_j·x_i = 2·Q[i,j]·x_i·x_j``). For a square
  ``a·x_i²`` we set ``Q[i,i] = a`` (``xᵀ Q x`` contributes
  ``Q[i,i]·x_i²`` directly).

The Hessian of ``xᵀ Q x + cᵀ x + d`` is the constant matrix ``2·Q``.
That is the payoff for the convexity certificate: on a *purely
quadratic* body the Hessian is constant, so an exact PSD test on ``Q``
(``λ_min(Q) >= 0``) is a rigorous, box-independent convexity proof —
strictly tighter than the conservative interval-Hessian + Gershgorin
row-sum enclosure, which can abstain on an indefinite-looking but
genuinely PSD matrix.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt.modeling.core import Expression, Model


def extract_quadratic(
    expr: Expression, n: int, model: Model
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Extract ``(Q, c, d)`` with ``expr == xᵀ Q x + cᵀ x + d``, or ``None``.

    Args:
        expr: A scalar :class:`~discopt.modeling.core.Expression`.
        n: The flat variable count (dimension of ``x``). ``Q`` is
            ``(n, n)`` and ``c`` is ``(n,)``; monomials must reference
            flat indices in ``[0, n)``.
        model: The model defining the flat variable layout (prefix-sum
            over ``model._variables`` by declaration order — the same
            layout the convexity certificate uses).

    Returns:
        A tuple ``(Q, c, d)`` where ``Q`` is a symmetric ``float64``
        ``(n, n)`` array, ``c`` is a ``float64`` ``(n,)`` array and
        ``d`` is a Python ``float``, such that
        ``expr(x) == x @ Q @ x + c @ x + d`` for every ``x``. Returns
        ``None`` if the expression is not purely quadratic in the model's
        original variables (a degree-≥3 term, a transcendental, a
        variable-in-denominator, a fractional power, an unsupported atom,
        or an out-of-range flat index).

    Notes:
        This is *exact-or-abstain*. It never returns an approximate
        ``Q``. The returned ``Q`` is symmetric by construction (the
        cross-coefficient ``b`` is split evenly onto ``Q[i,j]`` and
        ``Q[j,i]``).
    """
    if n < 0:
        return None

    # Local imports: keep module import cheap and avoid any import cycle
    # with milp_relaxation (which imports broadly from the _jax package).
    from discopt._jax.milp_relaxation import _expr_to_polynomial
    from discopt._jax.term_classifier import distribute_products

    try:
        poly = _expr_to_polynomial(distribute_products(expr), model)
    except Exception:
        # The trusted walker abstains loudly on shapes it cannot reduce
        # (array variables, etc.). Treat any failure as "not recognized"
        # — abstain rather than guess.
        return None

    if poly is None:
        return None

    const, terms = poly

    Q = np.zeros((n, n), dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    d = float(const)

    for coeff, monomial in terms:
        degree = len(monomial)
        if degree == 0:
            # A degree-0 monomial (rare — constants normally fold into
            # ``const``, but honor it if present).
            d += float(coeff)
        elif degree == 1:
            i = int(monomial[0])
            if not (0 <= i < n):
                return None
            c[i] += float(coeff)
        elif degree == 2:
            i, j = int(monomial[0]), int(monomial[1])
            if not (0 <= i < n and 0 <= j < n):
                return None
            if i == j:
                # a·x_i²  ->  Q[i,i] += a
                Q[i, i] += float(coeff)
            else:
                # b·x_i·x_j (i != j)  ->  Q[i,j] = Q[j,i] += b/2
                half = 0.5 * float(coeff)
                Q[i, j] += half
                Q[j, i] += half
        else:
            # Degree >= 3: not purely quadratic. Abstain — never
            # mis-extract a higher-degree expression as quadratic.
            return None

    return Q, c, d


def is_purely_quadratic(expr: Expression, n: int, model: Model) -> bool:
    """Return ``True`` iff :func:`extract_quadratic` recognizes ``expr``.

    A convenience predicate for callers that only need the yes/no verdict
    (a *coefficient-backed* upgrade of the Rust degree check — this says
    "quadratic AND exactly recoverable", where ``is_quadratic`` says only
    "degree ≤ 2"). Purely quadratic includes the affine and constant
    sub-cases (``Q == 0``).
    """
    return extract_quadratic(expr, n, model) is not None


def quadratic_is_psd(Q: np.ndarray, tol: float = 1e-10) -> Optional[bool]:
    """Exact PSD test on a symmetric ``Q``: ``True`` PSD, ``False`` not, ``None`` unusable.

    Uses the symmetric eigenvalue decomposition (``numpy.linalg.eigvalsh``)
    on ``½·(Q + Qᵀ)`` (the symmetric part; ``Q`` is already symmetric by
    construction from :func:`extract_quadratic`, but symmetrizing is
    defensive and free). A matrix is accepted as PSD when its minimum
    eigenvalue is ``>= -tol``.

    The ``tol`` slack absorbs floating-point round-off in the eigenvalue
    routine only; it is a soundness *margin* the caller must reconcile
    with its own certificate tolerance. Returns ``None`` if ``Q`` is not
    finite (the eigen-decomposition would be meaningless), so the caller
    can abstain to its existing rigorous path.

    Args:
        Q: A square, (approximately) symmetric matrix.
        tol: Non-negative slack for accepting ``λ_min >= 0`` despite
            round-off. Must match or be tighter than the caller's
            certificate tolerance.

    Returns:
        ``True`` if ``Q`` is (numerically) PSD, ``False`` if it is
        provably indefinite/negative, ``None`` if ``Q`` is not usable
        (non-finite entries).
    """
    Qa = np.asarray(Q, dtype=np.float64)
    if Qa.ndim != 2 or Qa.shape[0] != Qa.shape[1]:
        return None
    if not np.all(np.isfinite(Qa)):
        return None
    if Qa.shape[0] == 0:
        # The zero-dimensional form xᵀ Q x with n == 0 is the constant 0,
        # trivially PSD (convex).
        return True
    sym = 0.5 * (Qa + Qa.T)
    lam_min = float(np.linalg.eigvalsh(sym)[0])
    return lam_min >= -tol


def quadratic_is_nsd(Q: np.ndarray, tol: float = 1e-10) -> Optional[bool]:
    """Exact NSD (negative semidefinite) test — ``quadratic_is_psd(-Q)``.

    ``True`` if ``Q`` is negative semidefinite (the form is concave),
    ``False`` if not, ``None`` if unusable. Companion to
    :func:`quadratic_is_psd` for certifying concavity.
    """
    Qa = np.asarray(Q, dtype=np.float64)
    if Qa.ndim != 2 or Qa.shape[0] != Qa.shape[1]:
        return None
    if not np.all(np.isfinite(Qa)):
        return None
    return quadratic_is_psd(-Qa, tol=tol)


__all__ = [
    "extract_quadratic",
    "is_purely_quadratic",
    "quadratic_is_psd",
    "quadratic_is_nsd",
]
