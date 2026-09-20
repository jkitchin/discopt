"""Exact half-plane rewrite for ``atan2``.

``atan2(y, x)`` is the angle of the point ``(x, y)``, in ``(-pi, pi]``. It is
**discontinuous** across the branch cut ``{y = 0, x <= 0}``, where the value
jumps by ``2*pi``, and that discontinuity is fatal to spatial branch and bound
in a way an ordinary nonconvexity is not: refining a box *onto* the cut makes
the range of ``atan2`` **grow** toward ``2*pi`` rather than shrink toward zero.

Measured (``x in [-2, -1]``, halving the half-width of ``y`` each step;
reproduced by ``test_branch_cut_range_does_not_shrink_under_refinement``)::

    half-width 1.00e+00  ->  range 6.2807
    half-width 3.91e-03  ->  range 6.2832        ( -> 2*pi )
    half-width 4.88e-04  ->  range 6.2832

Every convergence argument for spatial B&B requires the relaxation gap to go to
zero as the box collapses, so no rigorous relaxation over a cut-straddling box
can certify anything, however far the tree branches. That is why ``atan2`` is
not an atom here, and why this module rewrites instead of relaxing.

*Away* from the cut, though, ``atan2`` is not merely relaxable — it is
**exactly expressible** in atoms discopt already relaxes rigorously: ``atan``
(``relax_atan``, convex/concave by the sign of its argument) over a ratio with
a sign-definite denominator (the existing ``ratio`` atom)::

    x > 0:   atan2(y, x) =  atan(y / x)
    y > 0:   atan2(y, x) =  pi/2 - atan(x / y)
    y < 0:   atan2(y, x) = -pi/2 - atan(x / y)

The last two hold on the *whole* upper / lower half-plane, left half included,
so the three cases together cover everything except boxes touching the closed
cut — exactly the region that cannot be certified anyway. Verified against
``math.atan2`` to 4.5e-16 over 900,007 comparisons spanning ``1e-8``..``1e6``
magnitude ratios plus the degenerate edges (``x == 0``, ``y == 0``, denormal
arguments); ``python/tests/test_atan2_rewrite.py`` carries that differential.

The rewrite introduces **no new IR node, no new envelope and no Rust change**:
it emits only atoms every downstream layer — FBBT, the relaxation compiler, the
``.nl`` writer, the arena tape, ``serialize`` — already handles, so a rewritten
``atan2`` certifies like any other smooth model.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple, Optional, cast

import numpy as np

from discopt.modeling.core import Constant, Expression

_LOG = logging.getLogger(__name__)

_HALF_PI = math.pi / 2.0

#: Which half-plane identity applies. ``None`` means the bounds do not prove any
#: of them, i.e. the box may touch the closed branch cut.
_X_POS = "x_pos"
_Y_POS = "y_pos"
_Y_NEG = "y_neg"


class Atan2Classification(NamedTuple):
    """Result of the sign analysis, carrying enough to explain a refusal.

    ``which`` is one of ``_X_POS`` / ``_Y_POS`` / ``_Y_NEG``, or ``None`` when no
    half-plane identity is provable. ``y_enc`` / ``x_enc`` are the scalar
    enclosures the verdict was read off, or ``None`` when one could not be
    computed at all; both are reported in the refusal message so the caller can
    see *which* argument failed to be pinned down.
    """

    which: Optional[str]
    y_enc: Optional[tuple[float, float]]
    x_enc: Optional[tuple[float, float]]


def _enclosure(expr: Expression) -> Optional[tuple[float, float]]:
    """Sound scalar enclosure of every element of ``expr``, or ``None``.

    ``evaluate_interval`` threads its ``model`` argument without ever reading it
    — variable bounds come from ``Variable.lb``/``.ub``, or from the optional
    box — so a bounds walk needs no model, and there is none to pass at
    expression-construction time. ``test_enclosure_ignores_model_argument`` pins
    that, because a ``model`` that silently became load-bearing here would
    misclassify a sign, and a misclassified sign is a *wrong* rewrite rather
    than a loose one.

    Array arguments collapse to the elementwise min of ``lo`` and max of ``hi``:
    the rewrite is emitted only when **every** element sits on one side, which
    is the conservative direction. Infinite endpoints are kept — ``(-inf, -1]``
    still proves negativity — but a NaN endpoint disqualifies the enclosure,
    since NaN compares false against everything and would read as "straddles".
    """
    from discopt._relax.convexity.interval_eval import evaluate_interval

    try:
        enc = evaluate_interval(expr, None)
        lo = float(np.min(np.asarray(enc.lo, dtype=np.float64)))
        hi = float(np.max(np.asarray(enc.hi, dtype=np.float64)))
    except Exception as exc:  # noqa: BLE001 -- deliberately broad; see below
        # This is NOT a swallowed error in the sense CLAUDE.md forbids. The only
        # thing a failure here can do is send the caller down the *refusal*
        # path, which never produces a model: `None` means "cannot prove a
        # sign", and every caller treats that as "do not rewrite". The dangerous
        # direction would be an exception that let a rewrite through, and no
        # such path exists.
        _LOG.debug("atan2: no interval enclosure for %r: %s: %s", expr, type(exc).__name__, exc)
        return None
    if math.isnan(lo) or math.isnan(hi):
        return None
    return lo, hi


def classify_atan2(y: Expression, x: Expression) -> Atan2Classification:
    """Decide which half-plane identity ``atan2(y, x)`` admits on its box.

    The tests are strict (``> 0`` / ``< 0``) with **no tolerance margin**: the
    enclosure is already a sound over-approximation, so ``lo > 0`` is a proof,
    and a margin would only reject provable cases. Zero is excluded on purpose —
    ``atan2`` is undefined at the origin, and each identity divides by exactly
    the argument tested here.
    """
    y_enc = _enclosure(y)
    x_enc = _enclosure(x)
    if y_enc is None or x_enc is None:
        return Atan2Classification(None, y_enc, x_enc)

    y_lo, y_hi = y_enc
    x_lo, _x_hi = x_enc

    # Margin = how far the *denominator* of that identity's ratio stays from
    # zero. Several identities can hold at once (e.g. x > 0 and y > 0); all are
    # exact, so the choice is pure tightness: the larger the margin, the
    # narrower the ratio's enclosure and the better conditioned the quotient.
    options: list[tuple[float, str]] = []
    if x_lo > 0.0:
        options.append((x_lo, _X_POS))
    if y_lo > 0.0:
        options.append((y_lo, _Y_POS))
    if y_hi < 0.0:
        options.append((-y_hi, _Y_NEG))
    if not options:
        return Atan2Classification(None, y_enc, x_enc)

    # Tie-break on the name so the emitted expression is deterministic and does
    # not depend on the order the options were appended in.
    _margin, which = max(options, key=lambda opt: (opt[0], opt[1]))
    return Atan2Classification(which, y_enc, x_enc)


def build_rewrite(cls: Atan2Classification, y: Expression, x: Expression) -> Optional[Expression]:
    """Emit the identity ``cls`` selected, or ``None`` if it selected none.

    Split from :func:`classify_atan2` so a caller that needs both the expression
    and the diagnosis (``dm.atan2``, which reports the enclosures when it
    refuses) pays for the interval walk once.

    The returned expression is built only from ``atan``, division and a
    constant, so it carries no trace of ``atan2`` into the IR.
    """
    from discopt.modeling.core import atan

    if cls.which is None:
        return None
    # The single choke point for emitting a rewrite, so registering here (rather
    # than in each caller) means no doorway -- `dm.atan2`, the GAMS link, or a
    # future one -- can build one without the solve-time guard.
    register_precondition(cls, y, x)
    if cls.which == _X_POS:
        return atan(y / x)
    # `Expression.__sub__` is untyped, so the results below are `Any` to mypy;
    # cast rather than widen this function's return type, which is the contract
    # callers rely on.
    if cls.which == _Y_POS:
        return cast(Expression, Constant(_HALF_PI) - atan(x / y))
    return cast(Expression, Constant(-_HALF_PI) - atan(x / y))


#: Attribute name under which a Model carries the sign assumptions its rewritten
#: ``atan2`` calls were built on. A list of
#: ``(denominator_expr, required_sign, label)`` triples.
PRECONDITIONS_ATTR = "_atan2_preconditions"


def register_precondition(cls: Atan2Classification, y: Expression, x: Expression) -> None:
    """Record the sign the rewrite assumed, so ``Model.validate`` can re-check it.

    The rewrite reads bounds **at construction**, but ``Variable.lb``/``.ub`` are
    mutable afterwards. Widening the bound of the argument the identity divides
    by silently turns the rewrite into a *different function* — measured: with
    ``dm.atan2(y, x)`` built under ``y >= 0.5`` and ``y.lb`` then set to ``-2``,
    the expression returns ``+2.356`` where ``atan2`` is ``-0.785``, an error of
    ``pi``, at 3 of 4 probe points. That is a false model, not a loose one, so
    it cannot be left to documentation (CLAUDE.md §1).

    Branching and FBBT only ever *narrow* a box inside the declared bounds, so
    they cannot trip this; only an explicit post-construction widening can, and
    that is exactly what the solve-time re-check catches.

    Registration needs the owning model, which is found from the argument DAGs.
    A call over pure constants owns no model and needs no guard — there is
    nothing mutable to invalidate it.

    Known limit: the list is model state, not expression state, so it does not
    survive :mod:`discopt.serialize`. A round-tripped model keeps the *rewritten*
    expression (correct for the bounds that were serialized alongside it) but
    loses the guard, so a widening applied after loading is not caught. Closing
    that would mean serializing the preconditions too.
    """
    from discopt.modeling.core import _find_owning_model

    if cls.which is None:
        return
    model = _find_owning_model(y, x)
    if model is None:
        return
    denominator, sign, label = {
        _X_POS: (x, "positive", "x > 0"),
        _Y_POS: (y, "positive", "y > 0"),
        _Y_NEG: (y, "negative", "y < 0"),
    }[cls.which]
    # Tolerate a Model that predates the attribute (e.g. one restored by an
    # older code path) rather than crashing a build that would otherwise work:
    # the guard is added, not assumed.
    recorded = getattr(model, PRECONDITIONS_ATTR, None)
    if recorded is None:
        recorded = []
        setattr(model, PRECONDITIONS_ATTR, recorded)
    recorded.append((denominator, sign, label))


def check_preconditions(model) -> None:
    """Re-verify every recorded ``atan2`` sign assumption; raise if one broke.

    Called from :meth:`discopt.modeling.core.Model.validate`, i.e. once per
    solve. Cost is one interval walk per ``atan2`` in the model — not a walk of
    the model — so it is proportional to the number of rewrites, not to model
    size.
    """
    from discopt.modeling.core import Atan2BranchCutError

    for denominator, sign, label in getattr(model, PRECONDITIONS_ATTR, ()):
        enc = _enclosure(denominator)
        ok = enc is not None and (enc[0] > 0.0 if sign == "positive" else enc[1] < 0.0)
        if not ok:
            raise Atan2BranchCutError(
                f"atan2 was rewritten assuming {label}, but that no longer holds: "
                f"the bound is now {_fmt_enclosure(enc)}. Bounds are read when the "
                "expression is built, so widening them afterwards invalidates the "
                "rewrite -- the expression would differ from atan2 by pi over part "
                "of the new box. Rebuild the atan2 call after setting the bounds, "
                "or restore the bound that made the sign definite."
            )


def rewrite_atan2(y: Expression, x: Expression) -> Optional[Expression]:
    """Exact rewrite of ``atan2(y, x)``, or ``None`` on a cut-straddling box.

    ``None`` is returned exactly when the declared bounds fail to prove that the
    box avoids the closed branch cut ``{y = 0, x <= 0}``. Callers decide what
    that means: :func:`discopt.modeling.atan2` raises
    :class:`~discopt.modeling.core.Atan2BranchCutError`, while the GAMS importer
    keeps the opaque node and solves it on the local path.
    """
    return build_rewrite(classify_atan2(y, x), y, x)


def _fmt_enclosure(enc: Optional[tuple[float, float]]) -> str:
    if enc is None:
        return "not computable from the declared bounds"
    return f"[{enc[0]:g}, {enc[1]:g}]"


def branch_cut_message(cls: Atan2Classification) -> str:
    """Explain a refusal, naming the enclosures the verdict was read off."""
    return (
        "atan2(y, x) crosses its branch cut {y = 0, x <= 0} on this box, where "
        "the value jumps by 2*pi. discopt has no atom for atan2 because that "
        "jump does not shrink under branching -- refining a box onto the cut "
        "drives the range of atan2 toward 2*pi -- so no rigorous relaxation "
        "there can certify a bound.\n"
        f"  bounds seen:  y in {_fmt_enclosure(cls.y_enc)}, "
        f"x in {_fmt_enclosure(cls.x_enc)}\n"
        "Bound one argument away from zero so the box lies in a single "
        "half-plane, and atan2 is rewritten exactly and solves to a certified "
        "global optimum:\n"
        "  x > 0  ->  atan(y / x)\n"
        "  y > 0  ->  pi/2 - atan(x / y)\n"
        "  y < 0  ->  -pi/2 - atan(x / y)\n"
        "For example m.continuous('x', lb=1e-3, ub=10) pins the first case. If "
        "the sign genuinely is not known in advance, reformulate in terms of "
        "atan, or model the quadrant explicitly with a binary variable."
    )
