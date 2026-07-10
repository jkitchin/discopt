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

Scope (P0.2): affine arithmetic, general bilinear product (all sign regimes), integer
powers via sign-agnostic repeated multiplication, constant division, and the
provably-convex-envelope intrinsics exp/log/log2/log10/sqrt/softplus/abs — for these
the kernel-chain subgradient is valid. The S-shaped intrinsics (tanh/atan/sigmoid/sinh)
have a non-convex cv over a sign-spanning box, so they raise
:class:`UnsupportedMcboxOp` until P1.1 adds per-regime subgradient selection. Anything
else also refuses (sound-or-refuse). P1 extends coverage (trig, fractional powers, the
tight monomial hull incl. odd-power-over-sign-spanning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from discopt._jax.multivariate_mccormick import _COMPOSITION_RULES


class UnsupportedMcboxOp(Exception):
    """A subexpression is outside the sound MCBox scope — refuse, never approximate."""


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
            raise UnsupportedMcboxOp("division by a non-constant (P1: sign-definite reciprocal)")
        c = jnp.asarray(o, dtype=jnp.float64)
        return _scalar_mul(self, 1.0 / c)

    def __pow__(self, p):
        if not (isinstance(p, int) or (isinstance(p, float) and float(p).is_integer())):
            raise UnsupportedMcboxOp(f"non-integer power {p} (P1.4: signomial)")
        n = int(p)
        if n < 1:
            raise UnsupportedMcboxOp(f"non-positive power {n}")
        # Repeated multiplication through the (sign-agnostic, validated) bilinear rule:
        # SOUND for every n>=1 and every sign regime. Looser than the tight monomial
        # envelope (P1.3 replaces this with relax_pow's exact hull), but always valid.
        out = self
        for _ in range(n - 1):
            out = out * self
        return out

    # -- univariate intrinsics: only envelopes whose cv is provably convex, so the
    #    kernel-chain subgradient is valid. Regime/S-shaped ops (tanh/atan/sigmoid/
    #    sinh) need per-regime subgradient selection -> P1.1 (refuse for now). --
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

    def _p1(self, name):
        raise UnsupportedMcboxOp(f"{name}: S-shaped envelope needs per-regime subgradient (P1.1)")

    def tanh(self):
        return self._p1("tanh")

    def atan(self):
        return self._p1("atan")

    def sigmoid(self):
        return self._p1("sigmoid")

    def sinh(self):
        return self._p1("sinh")


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
