"""Automated structured-cut derivation by symbolic constraint-chain elimination.

Many global-optimization gaps come not from a single nonconvex atom but from a
*chain* of equality constraints whose joint implication the relaxation misses.
The classic example (issue #15) is a gas pipe + compressor chain

    w^2 = C0 (P_S1^2 - p1^2),   p2 = beta p1,   w^2 = C2 (p2^2 - p5^2),

with a demand-forced lower bound ``p5 >= PN5``. Eliminating the intermediate
pressures yields a *coupling* between the surviving variables,

    beta >= sqrt(phi(w)) ,   phi(w) = C0 (C2 PN5^2 + w^2) / (C2 (C0 P_S1^2 - w^2)),

which, fed into an objective term ``w (beta^kappa - 1)``, collapses to a
**univariate convex underestimator** ``h(w)`` of that term. Injecting ``h`` as a
cut closes the relaxation gap (67% -> 0% on the gas benchmark).

This module automates that pipeline with SymPy:

1. :func:`eliminate_chain_coupling` — eliminate intermediate variables from a set
   of equality constraints and substitute the bounded variables at the extreme
   that yields a **valid lower bound** on a target variable (the substitution
   direction is verified from the sign of the derivative).
2. :func:`power_term_underestimator` — turn a coupling ``target >= g(keep)`` and a
   product objective term ``keep * (target^exponent - 1)`` into the univariate
   underestimator ``h(keep)`` plus a JAX closure and a tangent-cut generator,
   reusing the envelope engine.
3. :func:`verify_cut` — certify soundness of the derived cut by sampling the
   feasible manifold.

Design-time only (imports SymPy); the products are pure-JAX closures + numeric
cut coefficients for the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import sympy as sp


class CutDerivationError(RuntimeError):
    """Raised when the symbolic elimination cannot produce a valid coupling."""


@dataclass(frozen=True)
class ChainCoupling:
    """A derived lower-bound coupling ``target >= target_lower(keep)``.

    Attributes:
        keep: The surviving variable (e.g. flow ``w``).
        target: The coupled variable (e.g. ratio ``beta``).
        target_lower: SymPy expression for the valid lower bound on ``target`` in
            terms of ``keep`` (and any symbolic parameters).
        eliminated: Variables removed by the elimination.
        substituted: Mapping of bounded variables to the bound expression each was
            replaced by (the extreme giving a valid lower bound).
    """

    keep: sp.Symbol
    target: sp.Symbol
    target_lower: sp.Expr
    eliminated: tuple[sp.Symbol, ...]
    substituted: dict = field(default_factory=dict)


def _to_eq(e) -> sp.Eq:
    return e if isinstance(e, sp.Eq) else sp.Eq(e, 0)


def eliminate_chain_coupling(
    equations,
    *,
    target: sp.Symbol,
    keep: sp.Symbol,
    eliminate,
    lower_bounds: dict,
    sample: dict,
) -> ChainCoupling:
    """Derive a valid lower bound ``target >= target_lower(keep)`` from a chain.

    Args:
        equations: Iterable of SymPy ``Eq`` (or expressions implicitly ``== 0``).
        target: The variable to bound from below.
        keep: The variable the bound is expressed in.
        eliminate: Intermediate variables to remove by solving the system.
        lower_bounds: Mapping ``{var: lower_bound_expr}`` for the *bounded*
            intermediate variables; each is substituted at the extreme that gives
            a valid lower bound on ``target`` (direction verified numerically).
        sample: Numeric values (for ``keep``, the bounded vars, and any symbolic
            parameters) used to pick the positive solution branch and to check
            monotonicity signs.

    Returns:
        A :class:`ChainCoupling`.

    Raises:
        CutDerivationError: if no positive real branch is found.
    """
    eqs = [_to_eq(e) for e in equations]
    elim = list(eliminate)
    solset = sp.solve(eqs, [*elim, target], dict=True)
    if not solset:
        raise CutDerivationError("SymPy could not solve the constraint chain")

    # Pick the branch whose target is real & positive at the sample point.
    chosen: Optional[sp.Expr] = None
    for s in solset:
        if target not in s:
            continue
        expr = sp.simplify(s[target])
        try:
            val = complex(expr.subs(sample))
        except (TypeError, ValueError):
            continue
        if abs(val.imag) < 1e-9 and val.real > 0:
            chosen = expr
            break
    if chosen is None:
        raise CutDerivationError("no positive real branch for the target variable")

    # Substitute each bounded variable at the bound that yields a valid lower
    # bound on target: lower bound if target is increasing in it, else upper.
    substituted: dict = {}
    expr = chosen
    for var, lb in lower_bounds.items():
        d = sp.diff(chosen, var)
        try:
            slope = float(d.subs(sample))
        except (TypeError, ValueError) as exc:
            raise CutDerivationError(
                f"could not determine monotonicity of target in {var}"
            ) from exc
        if slope >= 0:
            expr = expr.subs(var, lb)
            substituted[var] = lb
        else:
            raise CutDerivationError(
                f"target decreases in {var}; need an upper bound, not provided"
            )

    return ChainCoupling(
        keep=keep,
        target=target,
        target_lower=sp.simplify(expr),
        eliminated=tuple(elim),
        substituted=substituted,
    )


@dataclass(frozen=True)
class TermUnderestimator:
    """A univariate underestimator ``h(keep)`` of a product objective term.

    Attributes:
        keep: The surviving variable.
        h_expr: SymPy expression for ``h(keep)`` (numeric params substituted).
        h_fn: JAX callable ``h(keep)``.
        is_convex: Whether ``h`` was verified convex on the sampled domain.
    """

    keep: sp.Symbol
    h_expr: sp.Expr
    h_fn: Callable
    is_convex: bool

    def tangent_cut(self, point: float) -> tuple[float, float]:
        """Return ``(value, slope)`` of the tangent to ``h`` at ``keep = point``.

        For convex ``h`` the tangent is a valid global underestimator, so
        ``term >= value + slope * (keep - point)`` is a sound linear cut.
        """
        import jax

        v = float(self.h_fn(point))
        g = float(jax.grad(lambda t: self.h_fn(t))(float(point)))
        return v, g


def power_term_underestimator(
    coupling: ChainCoupling,
    *,
    exponent: float,
    coefficient: float = 1.0,
    param_values: Optional[dict] = None,
    domain: tuple[float, float],
    n_check: int = 200,
) -> TermUnderestimator:
    """Build ``h(keep)`` underestimating ``coefficient * keep * (target^exp - 1)``.

    Uses ``target >= max(1, target_lower(keep))`` (the box floor ``target >= 1``
    combined with the coupling), so

        coefficient * keep * (target^exp - 1)
            >= coefficient * keep * (max(1, target_lower)^exp - 1)_+ =: h(keep).

    Args:
        coupling: The derived coupling ``target >= target_lower(keep)``.
        exponent: The exponent on ``target`` in the objective term.
        coefficient: Scalar multiplier (``K``).
        param_values: Numeric substitutions for symbolic parameters.
        domain: ``(lo, hi)`` range of ``keep`` for convexity checking.
        n_check: Samples for the convexity check.

    Returns:
        A :class:`TermUnderestimator`.
    """
    import jax.numpy as jnp

    keep = coupling.keep
    params = dict(param_values or {})
    lower = coupling.target_lower.subs(params)
    floor = sp.Max(1, lower)
    h_expr = sp.simplify(coefficient * keep * sp.Max(0, floor**exponent - 1))

    h_fn = sp.lambdify([keep], h_expr, modules="jax")

    # Verify convexity of h on the domain via midpoint (Jensen) inequality.
    lo, hi = domain
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.uniform(lo, hi, n_check))
    b = jnp.asarray(rng.uniform(lo, hi, n_check))
    mid = 0.5 * (a + b)
    is_convex = bool(jnp.all(h_fn(mid) <= 0.5 * (h_fn(a) + h_fn(b)) + 1e-6))

    return TermUnderestimator(keep=keep, h_expr=h_expr, h_fn=h_fn, is_convex=is_convex)


def verify_cut(
    term_fn: Callable,
    under: TermUnderestimator,
    coupling_fn: Callable,
    *,
    domain: tuple[float, float],
    target_max: float,
    n: int = 5000,
    seed: int = 0,
) -> dict:
    """Certify ``term >= h(keep)`` over the feasible manifold by sampling.

    Args:
        term_fn: The true term ``(keep, target) -> value``.
        under: The :class:`TermUnderestimator`.
        coupling_fn: ``keep -> target_lower(keep)`` (the lower bound on target).
        domain: Range of ``keep``.
        target_max: Upper bound on ``target`` for sampling.
        n: Number of samples.
        seed: RNG seed.

    Returns:
        ``{"sound": bool, "max_violation": float}`` where a violation is
        ``h(keep) - term`` (should be <= 0).
    """
    rng = np.random.default_rng(seed)
    lo, hi = domain
    ks = rng.uniform(lo, hi, n)
    viol = -np.inf
    for k in ks:
        tl = max(1.0, float(coupling_fn(k)))
        # sample target in its feasible range [tl, target_max]
        if tl > target_max:
            continue
        t = rng.uniform(tl, target_max)
        term = float(term_fn(k, t))
        h = float(under.h_fn(k))
        viol = max(viol, h - term)
    return {"sound": viol <= 1e-7, "max_violation": float(viol)}
