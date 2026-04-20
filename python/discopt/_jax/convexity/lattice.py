"""Lattices used by the convexity detector.

Three interacting lattices drive the propagation up the expression DAG:

* :class:`Curvature`    — CONVEX / CONCAVE / AFFINE / UNKNOWN
* :class:`Sign`         — NEG / NONPOS / ZERO / NONNEG / POS / UNKNOWN
* :class:`Monotonicity` — NONDEC / NONINC / CONST / UNKNOWN

Curvature and sign are propagated up the tree per subexpression and
bundled in :class:`ExprInfo`. Monotonicity is *not* propagated — it is
a per-argument property of outer atoms, consulted via an atom table at
composition sites and used by :func:`compose` to apply the DCP rule
(Grant, Boyd, Ye 2006) soundly.

References
----------
Grant, Boyd, Ye (2006), "Disciplined Convex Programming."
Boyd, Vandenberghe (2004), *Convex Optimization*, §3.1.
Ceccon, Siirola, Misener (2020), "SUSPECT," TOP.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Curvature
# ──────────────────────────────────────────────────────────────────────


class Curvature(Enum):
    """Curvature classification of an expression.

    AFFINE  — linear (both convex and concave).
    CONVEX  — epigraph is a convex set.
    CONCAVE — hypograph is a convex set.
    UNKNOWN — no sound verdict; treat as non-convex.
    """

    AFFINE = "affine"
    CONVEX = "convex"
    CONCAVE = "concave"
    UNKNOWN = "unknown"


def negate(c: Curvature) -> Curvature:
    """Curvature of ``-expr`` given curvature of ``expr``."""
    if c == Curvature.CONVEX:
        return Curvature.CONCAVE
    if c == Curvature.CONCAVE:
        return Curvature.CONVEX
    return c


def combine_sum(a: Curvature, b: Curvature) -> Curvature:
    """Curvature of ``a + b``."""
    if a == Curvature.AFFINE:
        return b
    if b == Curvature.AFFINE:
        return a
    if a == b:
        return a
    return Curvature.UNKNOWN


def scale(c: Curvature, sign: int) -> Curvature:
    """Curvature of ``k * expr`` where ``sign = sign(k)`` ∈ {-1, 0, +1}."""
    if sign == 0 or c == Curvature.AFFINE:
        return Curvature.AFFINE
    if sign > 0:
        return c
    return negate(c)


# ──────────────────────────────────────────────────────────────────────
# Sign lattice
# ──────────────────────────────────────────────────────────────────────


class Sign(Enum):
    """Sign classification of an expression's value.

    The lattice refines the real line into six non-overlapping
    (NEG/POS/ZERO) and overlapping (NONPOS/NONNEG) classes. UNKNOWN is
    the top element.

    Order (strongest → weakest):

        ZERO ⊂ {NONPOS, NONNEG}
        NEG  ⊂ NONPOS
        POS  ⊂ NONNEG
        all  ⊂ UNKNOWN
    """

    NEG = "neg"
    NONPOS = "nonpos"
    ZERO = "zero"
    NONNEG = "nonneg"
    POS = "pos"
    UNKNOWN = "unknown"


def sign_from_bounds(lb: float, ub: float) -> Sign:
    """Best sign label from a closed interval ``[lb, ub]``."""
    if not np.isfinite(lb):
        lb = -np.inf
    if not np.isfinite(ub):
        ub = np.inf
    if lb == 0 and ub == 0:
        return Sign.ZERO
    if lb > 0:
        return Sign.POS
    if ub < 0:
        return Sign.NEG
    if lb >= 0:
        return Sign.NONNEG
    if ub <= 0:
        return Sign.NONPOS
    return Sign.UNKNOWN


def sign_from_value(v) -> Sign:
    """Sign of a concrete numeric value (scalar or array)."""
    arr = np.asarray(v)
    if np.all(arr == 0):
        return Sign.ZERO
    if np.all(arr > 0):
        return Sign.POS
    if np.all(arr < 0):
        return Sign.NEG
    if np.all(arr >= 0):
        return Sign.NONNEG
    if np.all(arr <= 0):
        return Sign.NONPOS
    return Sign.UNKNOWN


def sign_negate(s: Sign) -> Sign:
    """Sign of ``-expr``."""
    return {
        Sign.NEG: Sign.POS,
        Sign.NONPOS: Sign.NONNEG,
        Sign.ZERO: Sign.ZERO,
        Sign.NONNEG: Sign.NONPOS,
        Sign.POS: Sign.NEG,
        Sign.UNKNOWN: Sign.UNKNOWN,
    }[s]


# Strict / non-strict distinction matters for reciprocal and log where
# a boundary-zero argument is undefined. ``is_strict`` captures that.
_STRICT = {Sign.NEG: True, Sign.POS: True, Sign.ZERO: True}


def is_strict(s: Sign) -> bool:
    """True if ``s`` excludes zero (NEG, POS, or ZERO)."""
    return _STRICT.get(s, False)


def is_nonneg(s: Sign) -> bool:
    """True if ``s`` proves the value is ≥ 0."""
    return s in (Sign.NONNEG, Sign.POS, Sign.ZERO)


def is_nonpos(s: Sign) -> bool:
    """True if ``s`` proves the value is ≤ 0."""
    return s in (Sign.NONPOS, Sign.NEG, Sign.ZERO)


def is_pos(s: Sign) -> bool:
    """True if ``s`` proves the value is strictly > 0."""
    return s == Sign.POS


def is_neg(s: Sign) -> bool:
    """True if ``s`` proves the value is strictly < 0."""
    return s == Sign.NEG


def sign_add(a: Sign, b: Sign) -> Sign:
    """Sign of ``a + b``.

    NONNEG + NONNEG = NONNEG, POS + NONNEG = POS, etc. Mixed-sign
    combinations collapse to UNKNOWN.
    """
    if a == Sign.ZERO:
        return b
    if b == Sign.ZERO:
        return a
    if is_pos(a) and is_nonneg(b):
        return Sign.POS
    if is_pos(b) and is_nonneg(a):
        return Sign.POS
    if is_neg(a) and is_nonpos(b):
        return Sign.NEG
    if is_neg(b) and is_nonpos(a):
        return Sign.NEG
    if is_nonneg(a) and is_nonneg(b):
        return Sign.NONNEG
    if is_nonpos(a) and is_nonpos(b):
        return Sign.NONPOS
    return Sign.UNKNOWN


def sign_mul(a: Sign, b: Sign) -> Sign:
    """Sign of ``a * b``."""
    if a == Sign.ZERO or b == Sign.ZERO:
        return Sign.ZERO
    # Resolve strictness: POS * POS = POS, NONNEG * NONNEG = NONNEG, etc.
    if is_pos(a) and is_pos(b):
        return Sign.POS
    if is_neg(a) and is_neg(b):
        return Sign.POS
    if (is_pos(a) and is_neg(b)) or (is_neg(a) and is_pos(b)):
        return Sign.NEG
    if is_nonneg(a) and is_nonneg(b):
        return Sign.NONNEG
    if is_nonpos(a) and is_nonpos(b):
        return Sign.NONNEG
    if (is_nonneg(a) and is_nonpos(b)) or (is_nonpos(a) and is_nonneg(b)):
        return Sign.NONPOS
    return Sign.UNKNOWN


def sign_reciprocal(s: Sign) -> Sign:
    """Sign of ``1 / expr``.

    Only defined when the argument is strictly signed (``POS`` or
    ``NEG``); otherwise ``UNKNOWN`` — a reciprocal with a zero in the
    domain is ill-defined and the detector should refuse to reason
    about it.
    """
    if s == Sign.POS:
        return Sign.POS
    if s == Sign.NEG:
        return Sign.NEG
    return Sign.UNKNOWN


# ──────────────────────────────────────────────────────────────────────
# Monotonicity (per outer-function argument)
# ──────────────────────────────────────────────────────────────────────


class Monotonicity(Enum):
    """Monotonicity of a function in a single argument.

    NONDEC  — nondecreasing.
    NONINC  — nonincreasing.
    CONST   — constant in the argument (both NONDEC and NONINC).
    UNKNOWN — neither provably.
    """

    NONDEC = "nondec"
    NONINC = "nonincr"
    CONST = "const"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────────────
# ExprInfo: the per-node propagated tuple
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExprInfo:
    """Propagated analysis of an expression: curvature plus sign."""

    curvature: Curvature
    sign: Sign = Sign.UNKNOWN


AFFINE_UNKNOWN = ExprInfo(Curvature.AFFINE, Sign.UNKNOWN)


# ──────────────────────────────────────────────────────────────────────
# Composition (DCP rule)
# ──────────────────────────────────────────────────────────────────────


def compose(
    f_curv: Curvature,
    f_mono: Monotonicity,
    g_curv: Curvature,
) -> Curvature:
    """Curvature of ``h = f(g)`` under the DCP composition rule.

    A composition ``f(g)`` is convex when ``f`` is convex and

    * ``g`` is affine, or
    * ``f`` is nondecreasing and ``g`` is convex, or
    * ``f`` is nonincreasing and ``g`` is concave.

    Symmetric rules give concavity. Anything else returns
    ``Curvature.UNKNOWN``.

    Reference: Grant, Boyd, Ye (2006), §3.1.
    """
    # Affine inner argument is always safe — outer curvature wins.
    if g_curv == Curvature.AFFINE:
        return f_curv

    if f_curv == Curvature.AFFINE:
        # An affine outer function acts as a sign-preserving scaling;
        # affine-on-anything preserves the inner curvature. With a
        # single argument this is a linear transform and keeps
        # curvature.
        return g_curv

    if f_curv == Curvature.CONVEX:
        if f_mono == Monotonicity.NONDEC and g_curv == Curvature.CONVEX:
            return Curvature.CONVEX
        if f_mono == Monotonicity.NONINC and g_curv == Curvature.CONCAVE:
            return Curvature.CONVEX
        if f_mono == Monotonicity.CONST:
            return Curvature.AFFINE
        return Curvature.UNKNOWN

    if f_curv == Curvature.CONCAVE:
        if f_mono == Monotonicity.NONDEC and g_curv == Curvature.CONCAVE:
            return Curvature.CONCAVE
        if f_mono == Monotonicity.NONINC and g_curv == Curvature.CONVEX:
            return Curvature.CONCAVE
        if f_mono == Monotonicity.CONST:
            return Curvature.AFFINE
        return Curvature.UNKNOWN

    return Curvature.UNKNOWN


# ──────────────────────────────────────────────────────────────────────
# Atom profiles: (curvature, monotonicity) conditioned on argument sign
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AtomProfile:
    """Curvature + monotonicity of a unary atom on a given arg sign."""

    curvature: Curvature
    monotonicity: Monotonicity


def unary_atom_profile(name: str, arg_sign: Sign) -> Optional[AtomProfile]:
    """Profile of a named unary atom given the sign of its argument.

    Returns ``None`` when no profile is known; callers must fall back
    to ``Curvature.UNKNOWN`` to preserve soundness.

    The table follows Boyd & Vandenberghe, *Convex Optimization*
    §3.1.5 (atoms) and CVX's atom library.
    """
    if name == "exp":
        # exp is convex and nondecreasing on all of R.
        return AtomProfile(Curvature.CONVEX, Monotonicity.NONDEC)

    if name in ("log", "log2", "log10"):
        # log is concave and nondecreasing on strictly positive R.
        if is_pos(arg_sign):
            return AtomProfile(Curvature.CONCAVE, Monotonicity.NONDEC)
        return None

    if name == "sqrt":
        # sqrt is concave and nondecreasing on nonneg R.
        if is_nonneg(arg_sign):
            return AtomProfile(Curvature.CONCAVE, Monotonicity.NONDEC)
        return None

    if name == "abs":
        # |x| is convex everywhere; nondec on x>=0, nonincr on x<=0.
        if is_nonneg(arg_sign):
            return AtomProfile(Curvature.CONVEX, Monotonicity.NONDEC)
        if is_nonpos(arg_sign):
            return AtomProfile(Curvature.CONVEX, Monotonicity.NONINC)
        return AtomProfile(Curvature.CONVEX, Monotonicity.UNKNOWN)

    if name == "cosh":
        # cosh is convex everywhere; nondec on x>=0, nonincr on x<=0.
        if is_nonneg(arg_sign):
            return AtomProfile(Curvature.CONVEX, Monotonicity.NONDEC)
        if is_nonpos(arg_sign):
            return AtomProfile(Curvature.CONVEX, Monotonicity.NONINC)
        return AtomProfile(Curvature.CONVEX, Monotonicity.UNKNOWN)

    if name == "sinh":
        # sinh is nondecreasing on all of R; convex on x>=0, concave on x<=0.
        if is_nonneg(arg_sign):
            return AtomProfile(Curvature.CONVEX, Monotonicity.NONDEC)
        if is_nonpos(arg_sign):
            return AtomProfile(Curvature.CONCAVE, Monotonicity.NONDEC)
        return None

    if name == "tanh":
        # tanh is concave on x>=0, convex on x<=0; nondecreasing everywhere.
        if is_nonneg(arg_sign):
            return AtomProfile(Curvature.CONCAVE, Monotonicity.NONDEC)
        if is_nonpos(arg_sign):
            return AtomProfile(Curvature.CONVEX, Monotonicity.NONDEC)
        return None

    return None
