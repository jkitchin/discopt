"""MCBox — a propagating McCormick relaxation type with rule-based subgradients.

The MC++ analogue, done the JAX way (MAiNGO-parity plan P0). Unlike the compiled
relaxation fns in :mod:`relaxation_compiler` — which collapse to the exact function
value at coincident points, so ``jax.grad`` of them is the *true* (nonconvex)
gradient, not a McCormick subgradient (#572) — an :class:`MCBox` carries the genuine
McCormick relaxation ``(cv, cc)`` over a box **plus its subgradients**, propagated by
per-operator rule through arbitrary jax-traceable code. Evaluate any model function on
``MCBox`` leaves and the convex/concave relaxation and valid supporting hyperplanes
fall out — the way dual numbers give automatic differentiation.

Design (validated by the P0.1 entry experiment):
- Fields ``(cv, cc, lo, hi, sub_cv, sub_cc)`` are **all dynamic pytree leaves**, so a
  single ``jax.jit`` works for every box and ``jax.vmap`` batches over per-node boxes
  (GPU). ``lo/hi`` are the static interval of the subexpression *over the box*, but
  carried as traced values so the compiled fn is box-agnostic.
- Envelope bounds always come from ``lo/hi`` (the box), never from operand cv/cc at
  coincident points (the collapse-bug lesson).
- **Subgradients are propagated by rule, not by autodiff over the whole construction.**
  Arithmetic (sum/product) uses explicit McCormick-plane rules with the piece-selection
  predicate choosing the subgradient of the active plane. Univariate composition reuses
  the tested Tsoukalas–Mitsos kernels (:mod:`multivariate_mccormick`) for the value and
  derives the subgradient by the **chain rule through the kernel**: ``∂kernel/∂(cv_g,
  cc_g)`` (a local, well-defined derivative of a known operator) contracted with the
  incoming subgradients. This avoids ``jax.grad`` over a possibly-nonconvex composite.

Scope: affine arithmetic, general bilinear product (all sign regimes), integer powers
via sign-agnostic repeated multiplication, constant division and sign-definite
variable division (x/y via the reciprocal; no-info bracket when y crosses 0), the
provably-convex-
envelope intrinsics exp/log/log2/log10/sqrt/softplus/abs (kernel-chain subgradient),
and (P1.1) the S-shaped intrinsics tanh/atan/sigmoid/sinh — tight kernel-chain on a
box that doesn't span the inflection, a sound constant-envelope fallback (jnp.where)
on a spanning box (valid but loose; P1.1b can add the tight tangent envelope), and
(P1.4) fractional/signomial powers x**a (non-integer a) over a strictly-positive base
(no-information bracket when the base can reach x<=0, where x**a is undefined). Anything
else refuses (sound-or-refuse). P1 continues coverage (trig, the tight monomial hull
incl. odd-power-over-sign-spanning).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from discopt._jax.multivariate_mccormick import (
    _COMPOSITION_RULES,
    clip_inner,
    compose_even_pow,
    compose_odd_pow,
    compose_pow_frac,
    odd_pow_tangent_coeff,
)


class UnsupportedMcboxOp(Exception):
    """A subexpression is outside the sound MCBox scope — refuse, never approximate."""


# ---------------------------------------------------------------- strict mode
#
# When tracing an OPAQUE user callable (a ``CustomCall`` body, P3.1) through MCBox,
# there is no AST to inspect: the caller cannot apply the ``_is_affine_ast`` guard that
# the reduced-space AST interpreter uses to refuse a non-affine division (whose ``cc``
# subgradient is NOT validated — it excluded the true optimum on nvs22 by ~1.7e5, task
# #69). To keep the CustomCall path sound-or-refuse, tracing runs under
# :func:`strict_division`, in which variable-denominator division (``MCBox / MCBox``)
# refuses loudly rather than emitting a possibly-unsound bound. Division by a numeric
# constant (``MCBox / c``) stays sound and is always allowed. Non-strict (default) mode
# is unchanged — the AST interpreter and the direct-MCBox tests keep their behavior.
_STRICT_DIVISION = False


@contextlib.contextmanager
def strict_division():
    """Refuse variable-denominator (``MCBox / MCBox``) division within this scope.

    Used to trace opaque ``CustomCall`` bodies soundly (P3.1): the non-affine reciprocal
    subgradient is not validated, and an opaque body offers no AST to gate it on, so the
    only sound contract is to refuse. Constant-denominator division is unaffected."""
    global _STRICT_DIVISION
    prev = _STRICT_DIVISION
    _STRICT_DIVISION = True
    try:
        yield
    finally:
        _STRICT_DIVISION = prev


@jax.tree_util.register_pytree_node_class
@dataclass
class MCBox:
    cv: jnp.ndarray  # convex underestimator value at the point
    cc: jnp.ndarray  # concave overestimator value at the point
    lo: jnp.ndarray  # interval lower bound over the box
    hi: jnp.ndarray  # interval upper bound over the box
    sub_cv: jnp.ndarray  # (n,) subgradient of cv
    sub_cc: jnp.ndarray  # (n,) subgradient of cc

    # -- pytree --
    def tree_flatten(self):
        return ((self.cv, self.cc, self.lo, self.hi, self.sub_cv, self.sub_cc), None)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(*ch)

    @property
    def n(self) -> int:
        return self.sub_cv.shape[0]

    # -- affine arithmetic --
    def __add__(self, o):
        o = o if isinstance(o, MCBox) else _const(o, self.n)
        return MCBox(
            self.cv + o.cv,
            self.cc + o.cc,
            self.lo + o.lo,
            self.hi + o.hi,
            self.sub_cv + o.sub_cv,
            self.sub_cc + o.sub_cc,
        )

    __radd__ = __add__

    def __neg__(self):
        return MCBox(-self.cc, -self.cv, -self.hi, -self.lo, -self.sub_cc, -self.sub_cv)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, MCBox) else _const(o, self.n)))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, MCBox):
            return _bilinear(self, o)
        return _scalar_mul(self, jnp.asarray(o, dtype=jnp.float64))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, MCBox):
            if _STRICT_DIVISION:
                raise UnsupportedMcboxOp(
                    "variable-denominator division inside an opaque CustomCall trace: the "
                    "non-affine reciprocal cc-subgradient is not validated (nvs22, task #69) "
                    "and there is no AST to gate it on — refuse (sound-or-refuse)"
                )
            return _bilinear(self, _reciprocal(o))  # x/y = x * (1/y), sign-definite y
        c = jnp.asarray(o, dtype=jnp.float64)
        return _scalar_mul(self, 1.0 / c)

    def __pow__(self, p):
        if isinstance(p, int) or (isinstance(p, float) and float(p).is_integer()):
            n = int(p)
            if n < 1:
                raise UnsupportedMcboxOp(f"non-positive integer power {n}")
            if n == 1:
                return self
            if n % 2 == 0:
                # P1.3: even powers use the TIGHT monomial hull (exact convex
                # envelope cv=x^n, secant cc) — materially tighter than repeated
                # bilinear multiplication (e.g. x^2 on [-2,3]: exact cv=0 at x=0 vs
                # repeated-mult's -4). Valid boundary subgradients via clip_into.
                return _pow_even(self, n)
            # P1.3: odd powers (n>=3) use the tight monomial hull — the convex/concave
            # envelope with the tangent-line construction over sign-spanning boxes,
            # tighter than repeated bilinear multiplication and (unlike relax_pow's
            # piecewise cv) genuinely convex, so the kernel-chain subgradient is valid.
            return _pow_odd(self, n)
        # Non-integer exponent: signomial x**a over a POSITIVE base (P1.4).
        return _pow_frac(self, float(p))

    # -- univariate intrinsics with provably-convex envelopes: the kernel-chain
    #    subgradient is valid directly (see _univariate). --
    def exp(self):
        return _univariate(self, "exp", jnp.exp)

    def log(self):
        return _univariate(self, "log", jnp.log)

    def log2(self):
        return _univariate(self, "log2", jnp.log2)

    def log10(self):
        return _univariate(self, "log10", jnp.log10)

    def sqrt(self):
        return _univariate(self, "sqrt", jnp.sqrt)

    def softplus(self):
        return _univariate(self, "softplus", lambda t: jnp.logaddexp(t, 0.0))

    def abs(self):
        return _univariate(self, "abs", jnp.abs, monotone_inc=False)

    # S-shaped (single inflection at 0, monotone increasing): P1.1 per-regime. On a
    # box that does NOT span the inflection the cv is convex, so the kernel-chain
    # subgradient is valid; on a spanning box the kernel's cv is non-convex (issue
    # #51), so a sound convex fallback (constant f(lo)) is selected by jnp.where.
    def tanh(self):
        return _sigmoidal(self, "tanh", jnp.tanh)

    def atan(self):
        return _sigmoidal(self, "atan", jnp.arctan)

    def sigmoid(self):
        return _sigmoidal(self, "sigmoid", jax.nn.sigmoid)

    def sinh(self):
        return _sigmoidal(self, "sinh", jnp.sinh)


# ---------------------------------------------------------------- helpers


def _const(c, n):
    c = jnp.asarray(c, dtype=jnp.float64)
    z = jnp.zeros(n, dtype=jnp.float64)
    return MCBox(c, c, c, c, z, z)


def _scalar_mul(a, c):
    pos = c >= 0
    return MCBox(
        jnp.where(pos, c * a.cv, c * a.cc),
        jnp.where(pos, c * a.cc, c * a.cv),
        jnp.where(pos, c * a.lo, c * a.hi),
        jnp.where(pos, c * a.hi, c * a.lo),
        jnp.where(pos, c * a.sub_cv, c * a.sub_cc),
        jnp.where(pos, c * a.sub_cc, c * a.sub_cv),
    )


def _pick(pred, va, sa, vb, sb):
    return jnp.where(pred, va, vb), jnp.where(pred, sa, sb)


def _bilinear(a, b):
    """General multivariate McCormick product (McCormick 1976 / MCB 2009) with
    rule-based subgradients: each McCormick plane substitutes the operand cv or cc by
    the sign of the (static) coefficient, and the active plane's subgradient is chosen
    by the same predicate that selects its value."""
    aL, aU, bL, bU = a.lo, a.hi, b.lo, b.hi
    # convex underestimator: max(plane1, plane2), coeff>=0 -> cv else cc
    bv1, bs1 = _pick(aL >= 0, b.cv, b.sub_cv, b.cc, b.sub_cc)
    av1, as1 = _pick(bL >= 0, a.cv, a.sub_cv, a.cc, a.sub_cc)
    cv1, cv1s = aL * bv1 + bL * av1 - aL * bL, aL * bs1 + bL * as1
    bv2, bs2 = _pick(aU >= 0, b.cv, b.sub_cv, b.cc, b.sub_cc)
    av2, as2 = _pick(bU >= 0, a.cv, a.sub_cv, a.cc, a.sub_cc)
    cv2, cv2s = aU * bv2 + bU * av2 - aU * bU, aU * bs2 + bU * as2
    t1 = cv1 >= cv2
    cv, cv_s = jnp.where(t1, cv1, cv2), jnp.where(t1, cv1s, cv2s)
    # concave overestimator: min(plane1, plane2), coeff>=0 -> cc else cv
    bv3, bs3 = _pick(aU >= 0, b.cc, b.sub_cc, b.cv, b.sub_cv)
    av3, as3 = _pick(bL >= 0, a.cc, a.sub_cc, a.cv, a.sub_cv)
    cc1, cc1s = aU * bv3 + bL * av3 - aU * bL, aU * bs3 + bL * as3
    bv4, bs4 = _pick(aL >= 0, b.cc, b.sub_cc, b.cv, b.sub_cv)
    av4, as4 = _pick(bU >= 0, a.cc, a.sub_cc, a.cv, a.sub_cv)
    cc2, cc2s = aL * bv4 + bU * av4 - aL * bU, aL * bs4 + bU * as4
    t1c = cc1 <= cc2
    cc, cc_s = jnp.where(t1c, cc1, cc2), jnp.where(t1c, cc1s, cc2s)
    ps = jnp.stack([aL * bL, aL * bU, aU * bL, aU * bU])
    return MCBox(cv, cc, jnp.min(ps), jnp.max(ps), cv_s, cc_s)


def _univariate(a, name, scalar_f, monotone_inc=True):
    kernel = _COMPOSITION_RULES.get(name)
    if kernel is None:
        raise UnsupportedMcboxOp(f"no composition kernel for '{name}'")
    if monotone_inc:
        interval = (scalar_f(a.lo), scalar_f(a.hi))
    else:  # abs — convex, min at 0 if the base spans it
        e = (scalar_f(a.lo), scalar_f(a.hi))
        interval = (jnp.where((a.lo < 0) & (a.hi > 0), 0.0, jnp.minimum(*e)), jnp.maximum(*e))
    return _univariate_kernel(a, kernel, interval)


def _recip_kernel(cvg, ccg, lb, ub):
    """Composition kernel for 1/y over a SIGN-DEFINITE box (0 not in [lb, ub]).

    y>0: 1/y is convex decreasing -> cv = 1/cc_g (value at the cc branch, the
    argmin of the decreasing envelope), cc = secant at cv_g. y<0: 1/y is concave
    decreasing -> cv = secant at cc_g, cc = 1/cv_g. The secant is the chord of 1/y
    on [lb, ub]. Differentiable in (cv_g, cc_g), so the kernel-chain subgradient is
    valid (cv is convex in both sign regimes)."""
    # McCormick mid-rule clamp: the operand's relaxation values bracket the true y in
    # [lb, ub], but a loose inner relaxation (e.g. y*y) can push cv_g below lb / cc_g
    # above ub; clamp into the valid interval first (still valid bounds on y, and where
    # 1/y is defined) so the reciprocal never extrapolates the secant / 1/(.) outside
    # the domain. Preserves soundness; gradient saturates on the clamped region.
    # ``clip_inner`` (not ``jnp.clip``): the boundary derivative must be the inner
    # one-sided slope (1), else a box-face iterate gets the halved clip-tie subgradient
    # and an invalid reciprocal cut (same soundness bug as the intrinsic envelopes).
    cvg = clip_inner(cvg, lb, ub)
    ccg = clip_inner(ccg, lb, ub)
    width = ub - lb
    slope = (1.0 / ub - 1.0 / lb) / jnp.where(jnp.abs(width) > 1e-12, width, 1.0)

    def sec(t):
        return 1.0 / lb + slope * (t - lb)

    pos = lb > 0.0
    cv = jnp.where(pos, 1.0 / ccg, sec(ccg))
    cc = jnp.where(pos, sec(cvg), 1.0 / cvg)
    return cv, cc


def _reciprocal(b):
    """1/b as an MCBox. Sound-or-refuse: if the denominator interval crosses zero
    the reciprocal is unbounded, so return a no-information bracket (-inf, +inf)
    (any downstream consumer of a non-finite bracket refuses); jit-safe via
    ``jnp.where``. Degenerate (lo==hi) reduces to the exact constant 1/lo."""
    lo, hi = b.lo, b.hi
    interval = (1.0 / hi, 1.0 / lo)  # 1/y is decreasing: min at hi, max at lo
    r = _univariate_kernel(b, _recip_kernel, interval)
    crosses = (lo <= 0.0) & (hi >= 0.0)
    zero = jnp.zeros_like(b.sub_cv)
    return MCBox(
        jnp.where(crosses, -jnp.inf, r.cv),
        jnp.where(crosses, jnp.inf, r.cc),
        jnp.where(crosses, -jnp.inf, r.lo),
        jnp.where(crosses, jnp.inf, r.hi),
        jnp.where(crosses, zero, r.sub_cv),
        jnp.where(crosses, zero, r.sub_cc),
    )


def _pow_even(base, n):
    """``base ** n`` for an even integer ``n >= 2`` (P1.3, tight monomial hull).

    ``x**n`` (even) is convex with its minimum at 0, so the exact convex envelope is
    ``x**n`` itself and the concave overestimator is the secant — materially tighter
    than the repeated-bilinear product, whose convex part is only the max of two tangent
    lines. Value from :func:`compose_even_pow`; subgradient by the kernel chain (valid —
    the envelope is convex and ``clip_into`` gives the valid boundary slope). The
    interval is ``[0, max(lo^n, hi^n)]`` when the box spans 0, else ``[min, max]`` of the
    endpoints (``x**n`` even is U-shaped)."""

    def kernel(cv_g, cc_g, g_lb, g_ub):
        return compose_even_pow(cv_g, cc_g, g_lb, g_ub, n)

    lo, hi = base.lo, base.hi
    e = (lo**n, hi**n)
    spans = (lo < 0.0) & (hi > 0.0)
    interval = (jnp.where(spans, 0.0, jnp.minimum(*e)), jnp.maximum(*e))
    return _univariate_kernel(base, kernel, interval)


def _pow_odd(base, n):
    """``base ** n`` for an odd integer ``n >= 3`` (P1.3, tight monomial hull).

    ``x**n`` (odd) is monotone increasing, so the interval is ``[lo**n, hi**n]``. The
    convex/concave envelope (:func:`compose_odd_pow`) is tight and, over a sign-spanning
    box, genuinely convex/concave via the tangent-line construction — so the kernel-chain
    subgradient is valid (unlike the naive piecewise ``relax_pow`` cv). The tangent
    coefficient is computed once at trace time (static ``n``)."""
    cn = odd_pow_tangent_coeff(n)

    def kernel(cv_g, cc_g, g_lb, g_ub):
        return compose_odd_pow(cv_g, cc_g, g_lb, g_ub, n, cn)

    interval = (base.lo**n, base.hi**n)
    return _univariate_kernel(base, kernel, interval)


def _pow_frac(base, a):
    """``base ** a`` for a non-integer exponent ``a`` (P1.4, signomial).

    ``x ** a`` is real only for ``x > 0`` when ``a`` is non-integer, so this is
    sound-or-refuse on the sign of the base interval: over a strictly-positive box
    (``lo > 0``) it uses the :func:`compose_pow_frac` envelope with the kernel-chain
    subgradient (valid — the envelope is convex/concave per regime and ``clip_inner``
    gives the valid boundary slope); a box that can reach ``x <= 0`` returns a
    no-information bracket ``(-inf, +inf)`` (jit-safe via ``jnp.where``) that any
    downstream consumer refuses. NaN discipline: the envelope is always evaluated on a
    base clamped to ``>= eps > 0`` so the discarded ``jnp.where`` branch never produces
    a NaN that would poison the subgradient."""
    eps = 1e-12

    def kernel(cv_g, cc_g, g_lb, g_ub):
        g_lb_s = jnp.maximum(g_lb, eps)
        g_ub_s = jnp.maximum(g_ub, eps)
        return compose_pow_frac(cv_g, cc_g, g_lb_s, g_ub_s, a)

    lo_s = jnp.maximum(base.lo, eps)
    hi_s = jnp.maximum(base.hi, eps)
    f_lo, f_hi = lo_s**a, hi_s**a
    inc = a >= 0.0  # x**a increasing for a>0, decreasing for a<0
    interval = (jnp.where(inc, f_lo, f_hi), jnp.where(inc, f_hi, f_lo))
    r = _univariate_kernel(base, kernel, interval)
    positive = base.lo > 0.0
    zero = jnp.zeros_like(base.sub_cv)
    return MCBox(
        jnp.where(positive, r.cv, -jnp.inf),
        jnp.where(positive, r.cc, jnp.inf),
        jnp.where(positive, r.lo, -jnp.inf),
        jnp.where(positive, r.hi, jnp.inf),
        jnp.where(positive, r.sub_cv, zero),
        jnp.where(positive, r.sub_cc, zero),
    )


def _sigmoidal(a, name, scalar_f):
    """S-shaped intrinsic (single inflection at 0, monotone increasing).

    NON-spanning box (``lo>=0`` or ``hi<=0``): the composition kernel's cv is convex
    (verified) -> use it with the kernel-chain subgradient. SPANNING box (``lo<0<hi``):
    the kernel's cv is a valid but non-convex underestimator (issue #51), so its
    kernel-chain "subgradient" would not support it. Select instead a sound convex
    fallback -- the constant ``f(lo)`` underestimator (``f`` increasing => ``f>=f(lo)``)
    with the constant ``f(hi)`` overestimator -- via ``jnp.where`` so the whole op stays
    jit/vmap-able. Loose on spanning boxes; P1.1b can add the tight tangent envelope.
    The interval is ``[f(lo), f(hi)]`` (monotone increasing) in both regimes.
    """
    kernel = _COMPOSITION_RULES.get(name)
    if kernel is None:
        raise UnsupportedMcboxOp(f"no composition kernel for '{name}'")
    f_lo, f_hi = scalar_f(a.lo), scalar_f(a.hi)
    ns = _univariate_kernel(a, kernel, (f_lo, f_hi))  # valid (convex) when non-spanning
    spanning = (a.lo < 0.0) & (a.hi > 0.0)
    zero = jnp.zeros_like(a.sub_cv)
    return MCBox(
        jnp.where(spanning, f_lo, ns.cv),
        jnp.where(spanning, f_hi, ns.cc),
        f_lo,
        f_hi,
        jnp.where(spanning, zero, ns.sub_cv),
        jnp.where(spanning, zero, ns.sub_cc),
    )


def _univariate_kernel(a, kernel, interval):
    """Value from the tested composition kernel; subgradient by the chain rule
    through the kernel: ∂kernel/∂(cv_g, cc_g) contracted with the incoming
    subgradients. The kernel is a known convex/concave-preserving operator, so its
    local partials give a valid subgradient of the (convex) cv / (concave) cc."""
    cv, cc = kernel(a.cv, a.cc, a.lo, a.hi)
    dcv = jax.grad(lambda p, q: kernel(p, q, a.lo, a.hi)[0], argnums=(0, 1))(a.cv, a.cc)
    dcc = jax.grad(lambda p, q: kernel(p, q, a.lo, a.hi)[1], argnums=(0, 1))(a.cv, a.cc)
    sub_cv = dcv[0] * a.sub_cv + dcv[1] * a.sub_cc
    sub_cc = dcc[0] * a.sub_cv + dcc[1] * a.sub_cc
    return MCBox(cv, cc, interval[0], interval[1], sub_cv, sub_cc)


# ---------------------------------------------------------------- public API


def mcbox_leaves(x, lb, ub):
    """Seed one MCBox per variable: cv=cc=x[i], lo/hi=lb/ub[i], sub=e_i."""
    x = jnp.asarray(x, dtype=jnp.float64)
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)
    n = x.shape[0]
    eye = jnp.eye(n, dtype=jnp.float64)
    return [MCBox(x[i], x[i], lb[i], ub[i], eye[i], eye[i]) for i in range(n)]


def relax_through(fn, x, lb, ub) -> MCBox:
    """Relax ``fn`` over the box by tracing it on MCBox leaves.

    ``fn`` is any callable written against MCBox-compatible ops (``+ - * / **`` and the
    ``mcbox`` intrinsic namespace / methods), taking the ``n`` leaf MCBoxes and
    returning a scalar MCBox. Returns the composite relaxation; ``jit``/``vmap``-able
    over ``x`` and the box.
    """
    return cast(MCBox, fn(*mcbox_leaves(x, lb, ub)))


# functional intrinsic namespace (mirror of the methods, MC++-style free functions)
def exp(a):
    return a.exp()


def log(a):
    return a.log()


def log2(a):
    return a.log2()


def log10(a):
    return a.log10()


def sqrt(a):
    return a.sqrt()


def softplus(a):
    return a.softplus()


def tanh(a):
    return a.tanh()


def atan(a):
    return a.atan()


def sigmoid(a):
    return a.sigmoid()


def sinh(a):
    return a.sinh()


def abs(a):
    return a.abs()  # noqa: A001
