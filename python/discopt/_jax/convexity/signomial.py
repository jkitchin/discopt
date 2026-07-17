"""Signomial (mixed-sign) canonicaliser and a rigorous box lower bound.

A *signomial* is a sum of signed monomials over strictly positive variables

    S(x) = sum_k c_k * prod_j x_j^{a_kj},        x_j > 0,

where — unlike a posynomial — some coefficients ``c_k`` may be **negative**.
A signomial is generally non-convex and has **no** exact convex reformulation
under ``y = log x`` (``log`` of a mixed-sign sum is not convex), so recognising
one is *not* a proof of anything: it is only the structural precursor to a
global scheme (issue #114, component 5 of #111).

This module is the deliberately-conservative front end for that work:

* :func:`is_signomial` — the *canonicaliser*. Parse an :class:`Expression` into
  a :class:`SignomialForm` (a normalised list of signed :class:`Monomial`
  terms over the model's flat scalar-variable indexing) when — and only when —
  every term is a monomial on the strictly-positive box. Returns ``None``
  otherwise. This reuses the exact same term-flattening / monomial-parsing as
  the posynomial recogniser (:mod:`discopt._jax.convexity.posynomial`); the one
  relaxation is that a term's coefficient may be negative.

* :func:`signomial_box_lower_bound` — a **rigorous** closed-form lower bound on
  ``min_x S(x)`` over the model's positive box, via interval arithmetic on each
  monomial. Sound for a *minimisation* (the bound is ``<=`` the true global
  optimum); it is a valid dual bound, never a certificate of optimality.

* :func:`signomial_dc_terms` — bridge the canonical form to the log-domain
  ``(sigma, log_c, exps)`` term list consumed by the certified DC convex/concave
  envelope in :mod:`discopt._jax.symbolic.signed_signomial`.

Soundness boundary (issue #114).
--------------------------------
Nothing here promotes a mixed-sign signomial to a *convex* or *global-optimal*
verdict. The GP path (:func:`discopt.gp.is_log_convex` / ``classify_gp``)
continues to abstain on any negative coefficient; this module adds only a valid
*lower bound* plus an explicitly-uncertified status. No auto-routing: the
default solve path is untouched.

References
----------
* Lundell, A. & Westerlund, T. Signomial global optimization (SGO): the single /
  positive-power transforms and the difference-of-convex underestimators of the
  alpha-SGO framework.
* Maranas, C. D. & Floudas, C. A. (1997). Global optimization in generalized
  geometric programming. *Computers & Chemical Engineering* 21(4), 351-369.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from discopt._jax.convexity.posynomial import (
    Monomial,
    _flatten_sum_terms,
    _parse_monomial,
)
from discopt.modeling.core import Expression, Model, VarType

# A coefficient with magnitude at or below this is treated as an exact zero and
# dropped (a cancelled term contributes nothing and carries no log).
_ZERO_TOL = 1e-12
# Two exponent-vector entries closer than this are treated as equal when merging
# like terms.
_EXP_TOL = 1e-12


@dataclass
class SignomialForm:
    """A signomial as a normalised list of signed :class:`Monomial` terms.

    Each :class:`Monomial` here may have a **negative** ``coeff`` (that is the
    whole point — a posynomial forbids it). ``exponents`` maps a flat
    scalar-variable offset to its real exponent, exactly as in
    :class:`~discopt._jax.convexity.posynomial.PosynomialForm`.
    """

    monomials: list[Monomial] = field(default_factory=list)

    @property
    def is_mixed_sign(self) -> bool:
        """True iff there is at least one positive AND one negative coefficient.

        A form with all-positive coefficients is really a posynomial; a form
        with all-negative coefficients is ``-posynomial``. Only the mixed case
        is non-trivially non-convex — and the case the GP path must not touch.
        """
        return any(m.coeff < -_ZERO_TOL for m in self.monomials) and any(
            m.coeff > _ZERO_TOL for m in self.monomials
        )

    @property
    def has_negative_term(self) -> bool:
        """True iff any coefficient is negative (not GP-representable as-is)."""
        return any(m.coeff < -_ZERO_TOL for m in self.monomials)

    def variable_offsets(self) -> set[int]:
        """Flat scalar offsets of every variable appearing with nonzero exponent."""
        offsets: set[int] = set()
        for mono in self.monomials:
            for off, exp in mono.exponents.items():
                if abs(exp) > _EXP_TOL:
                    offsets.add(off)
        return offsets

    def evaluate(self, x_by_offset: dict[int, float]) -> float:
        """Evaluate ``S(x)`` at a point given as ``{offset: value}`` (for tests)."""
        total = 0.0
        for mono in self.monomials:
            term = mono.coeff
            for off, exp in mono.exponents.items():
                term *= float(x_by_offset[off]) ** exp
            total += term
        return total


def _merge_like_terms(monomials: list[Monomial]) -> list[Monomial]:
    """Combine monomials that share an exponent vector; drop cancelled terms.

    Merging keeps the canonical form unique (so a round-trip is stable) and lets
    ``x - x`` collapse to nothing rather than masquerade as a two-term
    signomial. The exponent vector is keyed on the sorted ``(offset, rounded
    exponent)`` pairs of its non-zero entries.
    """
    buckets: dict[tuple[tuple[int, float], ...], float] = {}
    order: list[tuple[tuple[int, float], ...]] = []
    exps_by_key: dict[tuple[tuple[int, float], ...], dict[int, float]] = {}
    for mono in monomials:
        nz = {off: e for off, e in mono.exponents.items() if abs(e) > _EXP_TOL}
        key = tuple(sorted((off, round(e / _EXP_TOL) * _EXP_TOL) for off, e in nz.items()))
        if key not in buckets:
            buckets[key] = 0.0
            order.append(key)
            exps_by_key[key] = nz
        buckets[key] += mono.coeff
    merged: list[Monomial] = []
    for key in order:
        coeff = buckets[key]
        if abs(coeff) <= _ZERO_TOL:
            continue
        merged.append(Monomial(coeff, dict(exps_by_key[key])))
    return merged


def is_signomial(expr: Expression, model: Model) -> Optional[SignomialForm]:
    """Return a :class:`SignomialForm` if ``expr`` is a signomial, else ``None``.

    Preconditions (any failure -> ``None``), mirroring the posynomial
    recogniser except that a term's coefficient may be negative:

    * ``expr`` flattens into a sum of monomials (constants, positive-lb
      variable leaves, products, quotients, constant real/integer powers,
      ``sqrt``).
    * every variable leaf has a strictly positive declared lower bound
      (``x_j > 0`` — required for the ``y = log x`` domain to exist).
    * every exponent is a real *constant*, never another expression.

    A non-``None`` return is a genuine signomial on the strictly-positive box.
    It is **not** a convexity or global-optimality certificate.
    """
    terms: list[tuple[float, Expression]] = []
    _flatten_sum_terms(expr, 1.0, terms)

    monomials: list[Monomial] = []
    for scale, term in terms:
        mono = _parse_monomial(term, model)
        if mono is None:
            return None
        coeff = mono.coeff * scale
        if abs(coeff) <= _ZERO_TOL:
            # Exact-zero term (e.g. ``0 * x`` or a cancellation); skip it.
            continue
        monomials.append(Monomial(coeff, mono.exponents))

    monomials = _merge_like_terms(monomials)
    if not monomials:
        return None
    return SignomialForm(monomials)


# ──────────────────────────────────────────────────────────────────────
# Box bounds over flat scalar offsets
# ──────────────────────────────────────────────────────────────────────


def _offset_bounds(model: Model) -> dict[int, tuple[float, float]]:
    """Map every flat scalar offset to its ``(lb, ub)`` in ``x``-space."""
    bounds: dict[int, tuple[float, float]] = {}
    offset = 0
    for v in model._variables:
        lb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        for k in range(v.size):
            bounds[offset + k] = (float(lb[k]), float(ub[k]))
        offset += v.size
    return bounds


def _monomial_extrema(
    mono: Monomial, bounds: dict[int, tuple[float, float]]
) -> Optional[tuple[float, float]]:
    """Return ``(min, max)`` of ``|coeff| * prod x_j^{a_j}`` over the box.

    The magnitude is handled here (always positive); the caller applies the
    sign. For a single monomial ``c>0 * prod x_j^{a_j}`` over ``x_j in
    [lb_j, ub_j]`` with ``lb_j > 0``, monotonicity in each coordinate is fixed
    by the sign of the exponent: increasing when ``a_j > 0`` (use ``lb`` for the
    min, ``ub`` for the max), decreasing when ``a_j < 0`` (the reverse). The box
    extrema are therefore attained at closed-form corners — no search.

    ``ub = +inf`` is handled soundly: a positive-exponent factor makes the box
    max ``+inf`` and drives the min toward its ``lb`` corner; a negative-exponent
    factor makes the box min ``0`` (as ``x -> inf``) and the max its ``lb``
    corner. Returns ``None`` only if a lower bound is non-positive or non-finite
    (violating the strict-positivity precondition), which :func:`is_signomial`
    already forbids.
    """
    magnitude = abs(mono.coeff)
    lo = magnitude
    hi = magnitude
    for off, exp in mono.exponents.items():
        if abs(exp) <= _EXP_TOL:
            continue
        lb, ub = bounds[off]
        if not math.isfinite(lb) or lb <= 0.0:
            return None
        if not math.isfinite(ub):
            if exp > 0.0:
                # x^exp is increasing and unbounded above; min at lb, max -> +inf.
                lo *= lb**exp
                hi = math.inf
            else:
                # x^exp is decreasing to 0 as x -> inf; min -> 0, max at lb.
                lo = 0.0
                hi *= lb**exp
            continue
        lo_pick = lb if exp > 0.0 else ub
        hi_pick = ub if exp > 0.0 else lb
        lo *= lo_pick**exp
        hi *= hi_pick**exp
    return lo, hi


def signomial_box_lower_bound(form: SignomialForm, model: Model) -> float:
    """Rigorous lower bound on ``min_x S(x)`` over the model's positive box.

    Interval arithmetic on the signomial: writing
    ``S = sum_{c_k>0} c_k m_k - sum_{c_k<0} |c_k| m_k`` (each ``m_k`` a positive
    monomial), a valid lower bound is the sum of the positive terms' box minima
    minus the sum of the negative terms' box maxima::

        LB = sum_{+} min_box(c_k m_k) - sum_{-} max_box(|c_k| m_k).

    This is sound because ``S(x) >= sum_+ min - sum_- max`` pointwise on the box
    (each positive term is at least its minimum, each subtracted term at most its
    maximum). It is a **dual bound only** — ``LB <= min_x S(x)`` — never a
    proof of optimality, and it does not depend on any convexity of ``S``.

    Returns ``-inf`` when the box is open in a direction that makes a subtracted
    term unbounded above (an honest, sound "no useful bound"). Raises
    ``ValueError`` if a variable bound is non-positive (violating the signomial
    precondition; :func:`is_signomial` would already have refused such a model,
    so this only guards direct misuse).
    """
    bounds = _offset_bounds(model)
    lower = 0.0
    for mono in form.monomials:
        extrema = _monomial_extrema(mono, bounds)
        if extrema is None:
            raise ValueError(
                "signomial_box_lower_bound requires strictly positive, finite "
                "lower bounds on all variables"
            )
        lo, hi = extrema
        if mono.coeff > 0.0:
            lower += lo  # add the positive term's minimum
        else:
            lower -= hi  # subtract the negative term's maximum
        if lower == -math.inf:
            return -math.inf
    return lower


# ──────────────────────────────────────────────────────────────────────
# Bridge to the certified log-domain DC envelope
# ──────────────────────────────────────────────────────────────────────


def signomial_dc_terms(
    form: SignomialForm,
) -> tuple[list[tuple[float, float, np.ndarray]], list[int]]:
    """Convert to ``(sigma, log_c, exps)`` terms for the DC envelope.

    Produces the exact input format consumed by
    :func:`discopt._jax.symbolic.signed_signomial.signed_signomial_dc_envelope`:
    a dense exponent array over the sorted list of participating variable
    offsets. Returns ``(terms, offsets)`` where ``offsets`` is the ordered list
    of flat scalar offsets defining the ``u = log x`` coordinate order.

    ``sigma = sign(coeff)``, ``log_c = log|coeff|`` (the envelope stores each
    monomial with a positive coefficient and a separate sign, matching the
    difference-of-convex split).
    """
    offsets = sorted(form.variable_offsets())
    index = {off: j for j, off in enumerate(offsets)}
    n = len(offsets)
    terms: list[tuple[float, float, np.ndarray]] = []
    for mono in form.monomials:
        sigma = 1.0 if mono.coeff > 0.0 else -1.0
        log_c = math.log(abs(mono.coeff))
        exps = np.zeros(n, dtype=np.float64)
        for off, e in mono.exponents.items():
            if abs(e) > _EXP_TOL:
                exps[index[off]] = e
        terms.append((sigma, log_c, exps))
    return terms, offsets


# ──────────────────────────────────────────────────────────────────────
# Model-level uncertified relaxation (opt-in; never auto-routed)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SignomialRelaxation:
    """An uncertified signomial relaxation of a minimisation model.

    ``lower_bound`` is a rigorous dual bound on the global optimum (``<=`` it).
    ``certified`` is **always** ``False`` — a mixed-sign signomial is non-convex
    and this relaxation proves only a bound, not optimality. Consumers must
    treat it as a bound, never a certificate.
    """

    form: SignomialForm
    lower_bound: float
    is_mixed_sign: bool
    certified: bool = False


def signomial_relaxation(model: Model) -> Optional[SignomialRelaxation]:
    """Return an uncertified signomial relaxation of ``model``, or ``None``.

    Recognises a single-objective *minimisation* whose objective is a signomial
    over a strictly-positive continuous box and returns a rigorous lower bound
    plus an explicitly-uncertified status. Returns ``None`` when the model is
    not a minimisation signomial (e.g. no objective, maximisation, non-signomial
    objective, or any non-positive / non-continuous variable) — leaving the
    default solve path and the GP abstention entirely untouched.

    This never certifies optimality and never auto-routes; it is an opt-in
    analysis surface for the #114 SGO work.
    """
    if model._objective is None:
        return None
    from discopt.modeling.core import ObjectiveSense

    if model._objective.sense != ObjectiveSense.MINIMIZE:
        return None
    for v in model._variables:
        if v.var_type != VarType.CONTINUOUS:
            return None
        if float(np.asarray(v.lb).min()) <= 0.0:
            return None
    form = is_signomial(model._objective.expression, model)
    if form is None:
        return None
    try:
        lb = signomial_box_lower_bound(form, model)
    except ValueError:
        return None
    return SignomialRelaxation(
        form=form,
        lower_bound=lb,
        is_mixed_sign=form.is_mixed_sign,
        certified=False,  # invariant: a signomial relaxation never certifies
    )


__all__ = [
    "SignomialForm",
    "SignomialRelaxation",
    "is_signomial",
    "signomial_box_lower_bound",
    "signomial_dc_terms",
    "signomial_relaxation",
]
