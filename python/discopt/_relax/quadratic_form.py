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
  :func:`discopt._relax.milp_relaxation._expr_to_polynomial` (fed the
  :func:`discopt._relax.term_classifier.distribute_products` normal form),
  the same machinery the edge-concave collector already relies on. That
  walker returns ``None`` on any non-polynomial leaf; we additionally
  reject any monomial of degree > 2. The flat variable indexing is the
  identical prefix-sum layout used by the convexity certificate
  (``interval_ad._var_offset`` == ``term_classifier._compute_var_offset``),
  so a ``Q`` produced here is directly consistent with the coordinate
  system :func:`discopt._relax.convexity.certificate.certify_convex` works
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

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    IndexExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)


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

    reduced = extract_quadratic_support(expr, model)
    if reduced is None:
        return None
    support, Q_s, c_s, d = reduced
    if any(not (0 <= i < n) for i in support):
        return None

    Q = np.zeros((n, n), dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    if support:
        idx = np.asarray(support, dtype=np.intp)
        Q[np.ix_(idx, idx)] = Q_s
        c[idx] = c_s
    return Q, c, d


def extract_quadratic_support(
    expr: Expression, model: Model
) -> Optional[tuple[tuple[int, ...], np.ndarray, np.ndarray, float]]:
    """Support-restricted :func:`extract_quadratic`: ``(support, Q, c, d)`` or ``None``.

    Identical recognition and identical arithmetic, but the coefficients come
    back in the coordinate system of the *support* — the sorted tuple of flat
    variable indices the expression actually references — so

        ``expr(x) == x[support] @ Q @ x[support] + c @ x[support] + d``

    with ``Q`` of shape ``(m, m)`` and ``c`` of shape ``(m,)`` for
    ``m = len(support)``. The dense ``(n, n)`` form of :func:`extract_quadratic`
    costs O(n²) memory *per row*, which is prohibitive for a per-row scan over a
    model with tens of thousands of variables; a quadratic row's support is
    typically a handful of variables regardless of ``n``.

    Args:
        expr: A scalar :class:`~discopt.modeling.core.Expression`.
        model: The model defining the flat variable layout (prefix-sum over
            ``model._variables`` by declaration order).

    Returns:
        ``(support, Q, c, d)``, or ``None`` when ``expr`` is not purely
        quadratic. ``Q`` is symmetric by construction, using the same
        symmetric-split convention as :func:`extract_quadratic`. Exact or
        abstain: an approximate ``Q`` is never returned.
    """
    # Local imports: keep module import cheap and avoid any import cycle
    # with milp_relaxation (which imports broadly from the _relax package).
    from discopt._relax.milp_relaxation import _expr_to_polynomial
    from discopt._relax.term_classifier import distribute_products

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
    d = float(const)

    # First pass: collect the support and reject any degree >= 3 monomial,
    # so the reduced matrix is allocated only for a row we can actually use.
    support_set: set[int] = set()
    for _coeff, monomial in terms:
        if len(monomial) > 2:
            # Degree >= 3: not purely quadratic. Abstain — never
            # mis-extract a higher-degree expression as quadratic.
            return None
        for raw in monomial:
            support_set.add(int(raw))
    support = tuple(sorted(support_set))
    position = {flat_idx: pos for pos, flat_idx in enumerate(support)}

    m = len(support)
    Q = np.zeros((m, m), dtype=np.float64)
    c = np.zeros(m, dtype=np.float64)

    for coeff, monomial in terms:
        degree = len(monomial)
        if degree == 0:
            # A degree-0 monomial (rare — constants normally fold into
            # ``const``, but honor it if present).
            d += float(coeff)
        elif degree == 1:
            i = position[int(monomial[0])]
            c[i] += float(coeff)
        else:
            i, j = position[int(monomial[0])], position[int(monomial[1])]
            if i == j:
                # a·x_i²  ->  Q[i,i] += a
                Q[i, i] += float(coeff)
            else:
                # b·x_i·x_j (i != j)  ->  Q[i,j] = Q[j,i] += b/2
                half = 0.5 * float(coeff)
                Q[i, j] += half
                Q[j, i] += half

    return support, Q, c, d


_DEGREE_UNKNOWN = None


def polynomial_degree_bound(expr: Expression, _depth: int = 0) -> Optional[int]:
    """A cheap UPPER BOUND on ``expr``'s polynomial degree, or ``None`` if unknown.

    :func:`extract_quadratic_support` recognizes a row by *expanding* it into a
    monomial list, and the expansion is what costs: a row like st_e36's
    ``(x0² - 6x0 - 11 + 0.8x1) * (-0.62x1 + 3.25x0)² ...`` blows up to degree 6
    and takes 554 ms to expand before the degree check rejects it. Walking the
    DAG and *bounding* the degree without expanding costs O(nodes) and rejects
    that row immediately.

    The bound is structural: ``deg(a*b) <= deg(a) + deg(b)``, ``deg(a±b) <=
    max``, ``deg(a**k) <= k·deg(a)`` for a non-negative integer ``k``. It is an
    upper bound, not the true degree -- an expression whose leading terms cancel
    (``x³ - x³ + x²``) is bounded at 3 but is really quadratic. So a caller may
    use ``bound > 2`` to *skip* a row (a conservative abstention: it tightens
    less, never more) but must NOT use ``bound <= 2`` as proof of quadraticity;
    only the exact extraction decides that.

    Returns ``None`` for anything it cannot bound -- a transcendental, a
    variable denominator, a fractional or variable exponent, an unrecognized
    node -- so the caller falls through to its existing path unchanged.
    """
    if _depth > 200:
        return _DEGREE_UNKNOWN

    if isinstance(expr, Constant):
        return 0
    if isinstance(expr, Parameter):
        return 0
    if isinstance(expr, (Variable, IndexExpression)):
        return 1

    if isinstance(expr, UnaryOp):
        if expr.op in ("neg", "+"):
            return polynomial_degree_bound(expr.operand, _depth + 1)
        return _DEGREE_UNKNOWN

    if isinstance(expr, SumExpression):
        return polynomial_degree_bound(expr.operand, _depth + 1)

    if isinstance(expr, SumOverExpression):
        best = 0
        for term in expr.terms:
            d = polynomial_degree_bound(term, _depth + 1)
            if d is _DEGREE_UNKNOWN:
                return _DEGREE_UNKNOWN
            best = max(best, d)
        return best

    if isinstance(expr, BinaryOp):
        left = polynomial_degree_bound(expr.left, _depth + 1)
        if left is _DEGREE_UNKNOWN:
            return _DEGREE_UNKNOWN
        right = polynomial_degree_bound(expr.right, _depth + 1)
        if right is _DEGREE_UNKNOWN:
            return _DEGREE_UNKNOWN
        if expr.op in ("+", "-"):
            return max(left, right)
        if expr.op == "*":
            return left + right
        if expr.op == "/":
            # Only a constant denominator keeps it polynomial.
            return left if right == 0 else _DEGREE_UNKNOWN
        if expr.op in ("**", "^"):
            # A constant, non-negative integer exponent only.
            if right != 0:
                return _DEGREE_UNKNOWN
            k = _constant_exponent(expr.right)
            if k is None:
                return _DEGREE_UNKNOWN
            return left * k
        return _DEGREE_UNKNOWN

    # MatMulExpression, FunctionCall (exp/log/sin/...), CustomCall, and anything
    # else: not bounded here. Unknown, so the caller keeps its current path.
    return _DEGREE_UNKNOWN


def _constant_exponent(expr: Expression) -> Optional[int]:
    """The exponent as a non-negative int, or ``None`` (fractional/negative/not constant)."""
    value = getattr(expr, "value", None)
    if value is None:
        return None
    try:
        f = float(np.asarray(value).reshape(()))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f) or f < 0.0 or f != int(f):
        return None
    return int(f)


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
    "polynomial_degree_bound",
    "extract_quadratic_support",
    "is_purely_quadratic",
    "quadratic_is_psd",
    "quadratic_is_nsd",
]
