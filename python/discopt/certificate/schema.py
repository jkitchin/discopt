"""Certificate schema constants and exact-rational helpers.

Everything a checker must *trust* is emitted as an exact rational -- a
``[numerator, denominator]`` integer pair -- never a float ``repr``. A Python
``float`` is a dyadic rational, so :class:`fractions.Fraction` captures its exact
value with no rounding; the Lean checker reads the pair back as a ``Rat`` and
reasons about it exactly. Non-finite values (``inf`` for an open bound, ``nan``)
serialize as ``null`` and mean "no bound / not present".
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Optional, Union

# Bumped alongside ``discopt.result_io.SCHEMA_VERSION`` when the certificate
# payload changes shape.
CERTIFICATE_SCHEMA_VERSION = 1

Rational = list  # [int, int] == numerator, denominator (den > 0)


def to_rational(x: Union[float, int]) -> Optional[Rational]:
    """Exact ``[num, den]`` for a finite number, or ``None`` for ``inf``/``nan``.

    A Python ``float`` is a dyadic rational, so :class:`fractions.Fraction`
    captures its *exact* value -- no rounding, no ``limit_denominator``. Emitting
    the exact value is what makes the certificate checkable: the checker verifies
    exactly the point and coefficients the solver produced (the numerical
    tolerance lives in the feasibility comparison, not in the encoding). The pair
    is fully reduced with a positive denominator, so equal rationals encode
    identically.
    """
    xf = float(x)
    if not math.isfinite(xf):
        return None
    fr = Fraction(xf)
    return [fr.numerator, fr.denominator]


def rational_str(r: Optional[Rational]) -> str:
    """Human-readable ``num/den`` (or ``None``) -- for summaries and errors."""
    if r is None:
        return "None"
    num, den = r
    return f"{num}" if den == 1 else f"{num}/{den}"


def as_fraction(r: Optional[Rational]) -> Optional[Fraction]:
    """Inverse of :func:`to_rational` for the Python reference checker."""
    if r is None:
        return None
    num, den = r
    return Fraction(int(num), int(den))
