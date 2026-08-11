"""Per-expression log-curvature classification for geometric programming.

This module decides whether a SymPy expression over *positive* variables is
log-convex, log-concave, log-affine (a monomial), or none, after the
geometric-programming change of variables ``u_j = log(x_j)`` (equivalently
``x_j = exp(u_j)``).

Correctness basis
-----------------
Substitute ``x_j = exp(u_j)`` and classify the curvature of the resulting
function of ``u``:

* A monomial ``c * prod_j x_j**a_j`` with ``c > 0`` becomes
  ``exp(log c + sum_j a_j u_j)``; its logarithm ``log c + sum_j a_j u_j`` is
  **affine** in ``u`` -> the monomial is ``"log_affine"``.
* A posynomial (a sum of monomials with strictly positive coefficients) is a
  sum of exp-of-affine terms, which is **convex** in ``u`` -> ``"log_convex"``.
* The reciprocal ``1 / posynomial`` is **log-concave** in ``u`` -> ``"log_concave"``.
* Everything else is ``"none"``.

All free symbols are treated as positive (the GP domain ``x_j > 0``).

References
----------
* Boyd, Kim, Vandenberghe & Hassibi (2007), "A tutorial on geometric
  programming", *Optimization and Engineering* 8(1):67-127.
* Khajavirad, Michalek & Sahinidis (2012), on convexification / G-convexity
  of signomial and geometric-programming expressions.

Addresses issues #115 and #181.

This is a design-time (SymPy) module; no JAX is required.
"""

from __future__ import annotations

import sympy as sp


def is_monomial(expr: sp.Expr) -> tuple[bool, float | None, dict[sp.Symbol, sp.Expr]]:
    """Test whether ``expr`` is a positive monomial in its free symbols.

    A monomial has the form ``c * prod_j sym_j**a_j`` with a strictly positive
    constant ``c`` and real exponents ``a_j`` (rational or float). Products of
    monomials are monomials (exponents add); a monomial divided by a monomial
    is a monomial (exponents subtract). All symbols are treated as positive.

    Under ``u_j = log(x_j)`` the logarithm of a monomial is
    ``log(c) + sum_j a_j u_j``, which is affine in ``u``.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to test.

    Returns
    -------
    (is_mono, log_coeff, exponents) : tuple
        ``is_mono`` is True iff ``expr`` is a positive monomial. When True,
        ``log_coeff`` is ``log(c)`` (float) and ``exponents`` maps each symbol
        to its exponent. When False, ``log_coeff`` is None and ``exponents`` is
        an empty dict.
    """
    expr = sp.sympify(expr)
    coeff = sp.Integer(1)
    exponents: dict[sp.Symbol, sp.Expr] = {}

    # Decompose into multiplicative factors.
    factors = expr.args if isinstance(expr, sp.Mul) else (expr,)

    for factor in factors:
        if factor.is_number:
            # Must be a strictly positive real constant.
            if not (factor.is_real and factor.is_positive):
                return (False, None, {})
            coeff *= factor
        elif isinstance(factor, sp.Symbol):
            exponents[factor] = exponents.get(factor, sp.Integer(0)) + 1
        elif isinstance(factor, sp.Pow):
            base, exp = factor.base, factor.exp
            if not (isinstance(base, sp.Symbol) and exp.is_number and exp.is_real):
                return (False, None, {})
            exponents[base] = exponents.get(base, sp.Integer(0)) + exp
        else:
            # sin, exp, Add, etc. are not monomial factors.
            return (False, None, {})

    if not (coeff.is_real and coeff.is_positive):
        return (False, None, {})

    # Drop any symbols whose net exponent cancelled to zero.
    exponents = {s: e for s, e in exponents.items() if e != 0}
    log_coeff = float(sp.log(coeff))
    return (True, log_coeff, exponents)


def is_posynomial(expr: sp.Expr) -> bool:
    """Test whether ``expr`` is a posynomial.

    A posynomial is a sum of monomials, each with a strictly positive
    coefficient. A single monomial is a (degenerate) posynomial. Posynomials
    are sums of exp-of-affine terms after ``x_j = exp(u_j)`` and are therefore
    convex in ``u``.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to test.

    Returns
    -------
    bool
        True iff every additive term of ``expr`` is a positive monomial.
    """
    expr = sp.sympify(expr)
    terms = expr.args if isinstance(expr, sp.Add) else (expr,)
    return all(is_monomial(term)[0] for term in terms)


def log_curvature(expr: sp.Expr) -> str:
    """Classify the log-curvature of ``expr`` over positive variables.

    Under ``u_j = log(x_j)``: a monomial's log is affine, a posynomial is a sum
    of exp-of-affine terms (hence convex in ``u``), and the reciprocal of a
    posynomial is log-concave in ``u``.

    Parameters
    ----------
    expr : sympy.Expr
        Expression over positive variables.

    Returns
    -------
    str
        One of ``"log_affine"``, ``"log_convex"``, ``"log_concave"``, or
        ``"none"``.
    """
    expr = sp.sympify(expr)

    # A single monomial is log-affine (this also covers monomial ratios).
    if is_monomial(expr)[0]:
        return "log_affine"

    # A posynomial that is not a single monomial is log-convex.
    if is_posynomial(expr):
        return "log_convex"

    # The reciprocal of a posynomial is log-concave: Pow with exponent -1
    # (or any strictly negative exponent) of a posynomial base.
    if isinstance(expr, sp.Pow):
        base, exp = expr.base, expr.exp
        if exp.is_number and exp.is_real and exp.is_negative and is_posynomial(base):
            return "log_concave"

    return "none"
