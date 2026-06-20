"""Symbolic derivation of univariate convex/concave envelopes with SymPy.

Given a univariate SymPy expression ``f(x)`` and a box ``[a, b]`` (the bounds are
kept *symbolic* so the result specializes to any numeric box at code-gen time),
:func:`derive_envelope` classifies the curvature of ``f`` and derives the convex
underestimator ``cv`` and concave overestimator ``cc`` that form the tightest
envelope for the supported curvature classes:

* **CONVEX** (``f'' >= 0``)  →  ``cv = f``,            ``cc = secant``.
* **CONCAVE** (``f'' <= 0``) →  ``cv = secant``,        ``cc = f``.
* **CONCAVO_CONVEX** / **CONVEXO_CONCAVE** — a single inflection point ``c``.
  Over a box that straddles ``c`` the envelope is the classic
  *tangent-line / function* construction: on one side the envelope follows
  ``f`` itself; on the other it follows the line through a box endpoint that is
  tangent to ``f``. The tangent point is found by symbolically solving

  .. math::  f'(t)\\,(t - e) = f(t) - f(e)

  for ``t`` (``e`` is the supporting endpoint). This is the case that unlocks
  gas-network terms such as the Weymouth ``f|f|`` and odd powers ``x^3``.

Multi-inflection functions (more than one sign change of ``f''``) and tangent
equations SymPy cannot solve in closed form are intentionally rejected here with
a clear :class:`EnvelopeDerivationError`; they are handled by later phases
(piecewise / numeric tangent solves).

This module is **design-time only** — it imports SymPy. The resulting
:class:`EnvelopeResult` is consumed by :mod:`discopt._jax.symbolic.codegen`,
which emits a pure-JAX closure for the solver hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import sympy as sp


class Curvature(Enum):
    """Curvature classification of a univariate function over a box."""

    CONVEX = "convex"
    CONCAVE = "concave"
    CONCAVO_CONVEX = "concavo_convex"  # concave on the left of c, convex on the right
    CONVEXO_CONCAVE = "convexo_concave"  # convex on the left of c, concave on the right


class EnvelopeDerivationError(NotImplementedError):
    """Raised when the symbolic engine cannot derive an envelope in closed form."""


@dataclass(frozen=True)
class Tangent:
    """A tangent-line component of a single-inflection envelope.

    Attributes:
        point: Closed-form SymPy expression for the tangent point ``t`` in terms
            of the supporting endpoint symbol, or ``None`` when no closed form was
            found (the code generator then solves it numerically per box).
        from_lower: ``True`` if the tangent line emanates from the lower bound
            ``a``; ``False`` if it emanates from the upper bound ``b``.
        f_on_left: ``True`` if ``f`` itself is used on ``[a, t]`` and the tangent
            line on ``[t, b]``; ``False`` for the mirror image.
        tangent_positive_side: ``True`` if the tangent point lies on the branch
            ``t > c`` (above the inflection). Needed for the numeric solve.
    """

    point: Optional[sp.Expr]
    from_lower: bool
    f_on_left: bool
    tangent_positive_side: bool


@dataclass(frozen=True)
class EnvelopeResult:
    """Symbolic envelope of a univariate function over a symbolic box ``[a, b]``.

    The convex underestimator and concave overestimator are described by the
    curvature class plus, for single-inflection functions, the tangent
    components. :mod:`discopt._jax.symbolic.codegen` turns this into a JAX
    ``(x, lb, ub) -> (cv, cc)`` closure.
    """

    expr: sp.Expr
    var: sp.Symbol
    lower: sp.Symbol
    upper: sp.Symbol
    curvature: Curvature
    f_prime: sp.Expr
    inflection: Optional[sp.Expr] = None
    cv_tangent: Optional[Tangent] = None
    cc_tangent: Optional[Tangent] = None
    name: Optional[str] = None

    @property
    def is_single_inflection(self) -> bool:
        return self.curvature in (Curvature.CONCAVO_CONVEX, Curvature.CONVEXO_CONCAVE)


def _eval_sign(expr: sp.Expr, var: sp.Symbol, point: float) -> Optional[int]:
    """Return the sign (+1/-1/0) of ``expr`` at ``var = point``, or None if not real."""
    try:
        val = complex(expr.subs(var, point))
    except (TypeError, ValueError):
        return None
    if abs(val.imag) > 1e-12:
        return None
    if val.real > 1e-12:
        return 1
    if val.real < -1e-12:
        return -1
    return 0


def _nonsmooth_breakpoints(f: sp.Expr, var: sp.Symbol) -> list[sp.Expr]:
    """Zeros of the arguments of non-smooth atoms (``Abs``/``sign``/``Heaviside``).

    These are kink points where ``f''`` is a (Dirac) discontinuity rather than a
    smooth zero, so ``solve(f'' == 0)`` misses them. Detecting them is essential
    for correctly classifying functions like the Weymouth term ``x*|x|`` (which
    is concave-then-convex with a kink at ``0``) instead of silently treating it
    as convex — an unsound mistake.
    """
    pts: list[sp.Expr] = []
    for atom in f.atoms(sp.Abs, sp.sign, sp.Heaviside):
        arg = atom.args[0]
        try:
            sols = sp.solve(sp.Eq(arg, 0), var, dict=False)
        except (NotImplementedError, sp.PolynomialError):
            continue
        for s in sols:
            s = sp.simplify(s)
            if not s.free_symbols and s.is_real is not False:
                try:
                    if abs(complex(s).imag) <= 1e-12:
                        pts.append(s)
                except (TypeError, ValueError):
                    continue
    return pts


def _real_roots(expr: sp.Expr, var: sp.Symbol) -> list[sp.Expr]:
    """Real solutions of ``expr == 0`` in ``var`` (best effort, constants only)."""
    try:
        sols = sp.solve(sp.Eq(expr, 0), var, dict=False)
    except (NotImplementedError, sp.PolynomialError):
        return []
    out: list[sp.Expr] = []
    for s in sols:
        s = sp.simplify(s)
        if s.is_real is False:
            continue
        if s.free_symbols:
            # Parameter-dependent inflection points are out of Phase-1 scope.
            raise EnvelopeDerivationError(
                f"inflection point {s} depends on parameters {s.free_symbols}; "
                "parameter-dependent inflections are not yet supported"
            )
        if s.is_real or s.is_real is None:
            try:
                if abs(complex(s).imag) <= 1e-12:
                    out.append(sp.nsimplify(s) if s.is_number else s)
            except (TypeError, ValueError):
                continue
    # De-duplicate by float value.
    uniq: list[sp.Expr] = []
    seen: list[float] = []
    for s in out:
        fv = float(s)
        if not any(abs(fv - t) < 1e-12 for t in seen):
            seen.append(fv)
            uniq.append(s)
    return sorted(uniq, key=lambda e: float(e))


def _rewrite_side(f: sp.Expr, var: sp.Symbol, c: float, positive_side: bool) -> sp.Expr:
    """Rewrite non-smooth atoms of ``f`` on one side of the kink ``c``.

    On a given side of ``c`` the argument of each ``Abs``/``sign``/``Heaviside``
    has a definite sign, so the atom collapses to a smooth expression. This makes
    the tangent equation solvable by SymPy on the piece where the tangent point
    lives (``positive_side`` selects ``x > c`` vs ``x < c``).
    """
    test = c + 1e-3 if positive_side else c - 1e-3
    subs: dict = {}
    for atom in f.atoms(sp.Abs, sp.sign, sp.Heaviside):
        arg = atom.args[0]
        s = _eval_sign(arg, var, test)
        if s is None:
            continue
        if isinstance(atom, sp.Abs):
            subs[atom] = arg if s >= 0 else -arg
        elif isinstance(atom, sp.sign):
            subs[atom] = sp.Integer(1) if s >= 0 else sp.Integer(-1)
        else:  # Heaviside
            subs[atom] = sp.Integer(1) if s >= 0 else sp.Integer(0)
    return sp.simplify(f.subs(subs)) if subs else f


def _solve_tangent_point(
    f: sp.Expr,
    var: sp.Symbol,
    endpoint: sp.Symbol,
    *,
    solve_expr: sp.Expr,
    endpoint_expr: sp.Expr,
    inflection: sp.Expr,
    tangent_positive_side: bool,
) -> sp.Expr:
    """Solve ``f'(t)(t - e) = f(t) - f(e)`` for the tangent point ``t``.

    The tangent point lives on the side of the inflection ``c`` opposite the
    supporting endpoint ``e``. To let SymPy resolve the signs of any non-smooth
    atoms, the endpoint is substituted by ``c ± u`` with ``u`` a *positive* dummy,
    ``solve_expr`` is the smooth rewrite of ``f`` on the tangent-point side, and
    ``endpoint_expr`` the smooth rewrite on the endpoint side. The result is
    re-expressed in terms of the original ``endpoint`` symbol.

    Args:
        tangent_positive_side: ``True`` if the tangent point satisfies ``t > c``.

    Returns:
        The closed-form tangent point, or ``None`` if SymPy could not isolate a
        unique real one (the caller then falls back to a numeric solve at
        code-gen time).
    """
    u = sp.Dummy("u", positive=True)
    if tangent_positive_side:
        e_sub = inflection - u  # endpoint below c
        u_back = inflection - endpoint
    else:
        e_sub = inflection + u  # endpoint above c
        u_back = endpoint - inflection

    t = sp.Dummy("t", real=True)
    f_t = solve_expr.subs(var, t)
    fp_t = sp.diff(solve_expr, var).subs(var, t)
    f_e = endpoint_expr.subs(var, e_sub)
    residual = sp.expand(fp_t * (t - e_sub) - (f_t - f_e))
    try:
        sols = sp.solve(sp.Eq(residual, 0), t, dict=False)
    except (NotImplementedError, sp.PolynomialError):
        return None

    c_val = float(inflection)
    keep: list[sp.Expr] = []
    for s in sols:
        s = sp.simplify(s)
        if s.is_real is False:
            continue
        # Evaluate at u=1 to test the side and reject the trivial endpoint root.
        try:
            probe = complex(s.subs(u, 1.0))
        except (TypeError, ValueError):
            continue
        if abs(probe.imag) > 1e-12:
            continue
        tv = probe.real
        on_side = (tv > c_val + 1e-9) if tangent_positive_side else (tv < c_val - 1e-9)
        if not on_side:
            continue
        keep.append(s)

    dedup: list[sp.Expr] = []
    for s in keep:
        if not any(sp.simplify(s - d) == 0 for d in dedup):
            dedup.append(s)

    if len(dedup) != 1:
        return None
    return sp.simplify(dedup[0].subs(u, u_back))


def derive_envelope(
    expr: sp.Expr,
    var: sp.Symbol,
    *,
    lower: Optional[sp.Symbol] = None,
    upper: Optional[sp.Symbol] = None,
    sample_point: float = 1.0,
    name: Optional[str] = None,
) -> EnvelopeResult:
    """Derive the convex/concave envelope of ``expr`` over a symbolic box ``[a, b]``.

    Args:
        expr: Univariate SymPy expression ``f(x)``.
        var: The free symbol ``x``.
        lower, upper: Symbols used for the box bounds in the result. Created as
            ``a``/``b`` if omitted.
        sample_point: A point in the (assumed) domain used to resolve the sign of
            ``f''`` when it has no real roots (e.g. ``1.0`` for ``log``/``sqrt``).
        name: Optional human-readable label propagated to code-gen.

    Returns:
        An :class:`EnvelopeResult` describing ``cv``/``cc``.

    Raises:
        EnvelopeDerivationError: if the curvature pattern or tangent equation is
            outside the closed-form scope of this phase.
    """
    a = lower if lower is not None else sp.Symbol("a", real=True)
    b = upper if upper is not None else sp.Symbol("b", real=True)
    f = sp.sympify(expr)
    fp = sp.diff(f, var)
    f2 = sp.diff(f, var, 2)

    if fp.has(sp.DiracDelta):
        raise EnvelopeDerivationError(
            f"derivative of {f} contains DiracDelta (non-smooth representation via "
            "sign/Heaviside). Rewrite the term using Abs, e.g. 'x*Abs(x)' instead "
            "of 'sign(x)*x**2', so the gradient stays representable."
        )

    # Candidate inflections: smooth zeros of f'' plus non-smooth kinks (Abs/sign
    # arguments), de-duplicated by value.
    candidates = _real_roots(f2, var) + _nonsmooth_breakpoints(f, var)
    roots: list[sp.Expr] = []
    seen: list[float] = []
    for r in candidates:
        rv = float(r)
        if not any(abs(rv - s) < 1e-12 for s in seen):
            seen.append(rv)
            roots.append(r)

    # Determine genuine inflection points: candidates across which the sign of
    # f'' actually changes.
    inflections: list[sp.Expr] = []
    for r in roots:
        rv = float(r)
        left = _eval_sign(f2, var, rv - 1e-3)
        right = _eval_sign(f2, var, rv + 1e-3)
        if left is not None and right is not None and left != 0 and right != 0 and left != right:
            inflections.append(r)

    if len(inflections) == 0:
        # Pure curvature: resolve the constant sign of f''.
        sign = _eval_sign(f2, var, sample_point)
        if sign is None:
            for p in (0.5, 2.0, -1.0, 0.1, 10.0):
                sign = _eval_sign(f2, var, p)
                if sign is not None:
                    break
        if sign is None:
            raise EnvelopeDerivationError(f"could not determine the sign of f'' = {f2} for {f}")
        if sign >= 0:
            curvature = Curvature.CONVEX
        else:
            curvature = Curvature.CONCAVE
        return EnvelopeResult(
            expr=f, var=var, lower=a, upper=b, curvature=curvature, f_prime=fp, name=name
        )

    if len(inflections) > 1:
        raise EnvelopeDerivationError(
            f"{f} has {len(inflections)} inflection points; multi-inflection "
            "envelopes are handled by a later phase"
        )

    c = inflections[0]
    cv_float = float(c)
    left_sign = _eval_sign(f2, var, cv_float - 1e-3)
    # Smooth rewrites of f on each side of the kink, for the symbolic solve.
    right_expr = _rewrite_side(f, var, cv_float, positive_side=True)
    left_expr = _rewrite_side(f, var, cv_float, positive_side=False)
    # CONCAVO_CONVEX: concave (f''<0) on the left, convex on the right.
    if left_sign == -1:
        curvature = Curvature.CONCAVO_CONVEX
        # cv: tangent from lower endpoint a (on left), point on the right (convex).
        cv_pt = _solve_tangent_point(
            f,
            var,
            a,
            solve_expr=right_expr,
            endpoint_expr=left_expr,
            inflection=c,
            tangent_positive_side=True,
        )
        cv_tangent = Tangent(
            point=cv_pt, from_lower=True, f_on_left=False, tangent_positive_side=True
        )
        # cc: tangent from upper endpoint b (on right), point on the left (concave).
        cc_pt = _solve_tangent_point(
            f,
            var,
            b,
            solve_expr=left_expr,
            endpoint_expr=right_expr,
            inflection=c,
            tangent_positive_side=False,
        )
        cc_tangent = Tangent(
            point=cc_pt, from_lower=False, f_on_left=True, tangent_positive_side=False
        )
    else:
        curvature = Curvature.CONVEXO_CONCAVE
        # cv: tangent from upper endpoint b (on right), point on the left (convex).
        cv_pt = _solve_tangent_point(
            f,
            var,
            b,
            solve_expr=left_expr,
            endpoint_expr=right_expr,
            inflection=c,
            tangent_positive_side=False,
        )
        cv_tangent = Tangent(
            point=cv_pt, from_lower=False, f_on_left=True, tangent_positive_side=False
        )
        # cc: tangent from lower endpoint a (on left), point on the right (concave).
        cc_pt = _solve_tangent_point(
            f,
            var,
            a,
            solve_expr=right_expr,
            endpoint_expr=left_expr,
            inflection=c,
            tangent_positive_side=True,
        )
        cc_tangent = Tangent(
            point=cc_pt, from_lower=True, f_on_left=False, tangent_positive_side=True
        )

    return EnvelopeResult(
        expr=f,
        var=var,
        lower=a,
        upper=b,
        curvature=curvature,
        f_prime=fp,
        inflection=c,
        cv_tangent=cv_tangent,
        cc_tangent=cc_tangent,
        name=name,
    )
