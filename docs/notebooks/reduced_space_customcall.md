# Reduced-Space Global Optimization of Hidden-Function Models

Many engineering models are written as *opaque code*: a flowsheet unit, a trained
surrogate, or a subroutine that maps a handful of decision variables through a long
chain of internal intermediates to an output. In discopt such a function is wrapped
with `dm.custom(...)`, which produces a `CustomCall` node — an arbitrary
JAX-traceable callable that the solver can autodifferentiate but cannot see inside.

This notebook shows how discopt gives these hidden-function models a **certified
global** solve by relaxing the opaque body through the reduced-space McCormick type
(`MCBox`) and branching only on the true **degrees of freedom** (DOF) — the signature
capability of {cite:t}`Bongartz2018`'s MAiNGO solver. The internal intermediates never
become optimization or branching variables, so the branch-and-bound tree stays in the
low-dimensional DOF space.

## Two ways to model a nonlinear term

If you can write a function with discopt primitives (`dm.exp`, `dm.log`, arithmetic on
variables, ...), do so — `dm.udf` keeps full global-solver support. Reach for
`dm.custom` only when the body genuinely cannot be expressed that way (e.g. it calls
into external JAX code). A `CustomCall` is opaque, so:

- **If the body traces soundly through `MCBox`** — arithmetic (`+ - * / **`) and the
  `discopt._jax.mcbox` intrinsic namespace (`exp`, `log`, `sqrt`, fractional powers,
  ...) — a **continuous or integer** model is solved **globally with a certificate**,
  branching on the DOF only (the hidden intermediates stay hidden).
- **Otherwise** (a raw `jnp` intrinsic applied to an argument, a non-affine hidden
  division, a non-scalar leaf, an unbounded box) the model falls back to the **local
  NLP path** — a valid solution, but no global optimality certificate. This is
  *sound-or-refuse*: discopt never returns a partial or invalid global bound.

## A flowsheet as `CustomCall` units

Consider a reactor cascade. Each stage carries a flow `c_in` forward as
`c_out = c_in · exp(-a·T)`, where `T` is the reactor temperature (a DOF) and the rate
factor `e = exp(-a·T)` and the carryover `c_out` are *internal intermediates*. We write
each stage as a `CustomCall`, and chain them:

```python
import jax.numpy as jnp
import discopt.modeling as dm
from discopt._jax.mcbox import MCBox

def mexp(x):
    # dispatch: MCBox for the relaxation, jnp for the local-NLP / value path
    return x.exp() if isinstance(x, MCBox) else jnp.exp(x)

def reactor(c_in, T, a):
    return c_in * mexp(-a * T)   # hidden intermediates: k = a*T, e = exp(-k)

m = dm.Model("cascade")
T = m.continuous("T", 2, lb=[0.2, 0.2], ub=[2.0, 2.0])   # the only DOF
unit = [dm.custom(lambda c, t, a=a: reactor(c, t, a), name=f"r{i}")
        for i, a in enumerate((0.8, 0.6))]

F0 = 1.0
c1 = unit[0](F0, T[0])     # CustomCall
c2 = unit[1](c1, T[1])     # nested CustomCall — hides e_1, e_2, c_1, c_2

reacted = F0 - c2
energy = 0.15 * T[0] * (F0 - c1) + 0.15 * T[1] * (c1 - c2)
m.minimize(-reacted + energy)
m.subject_to(reacted >= 0.3)

result = m.solve()          # certified global solve, branching on T only
```

Because the body of each `reactor` is pure arithmetic plus `exp` (dispatched to
`MCBox`), the model is admitted to the global path. The solver branches on `T1, T2`
alone — never on the four hidden intermediates.

## How the opaque body is relaxed

Under the hood, discopt evaluates the `CustomCall` on `MCBox` *leaves* — one per DOF —
and lets the relaxation propagate by rule, the way dual numbers give automatic
differentiation. Each `MCBox` carries the McCormick convex/concave bracket `(cv, cc)`
over the current box **and its subgradients**, propagated per operator: the general
bilinear product rule of {cite:t}`McCormick1976`, the multivariate composition rules of
{cite:t}`Tsoukalas2014` for the intrinsics, and rule-based subgradient selection in the
spirit of {cite:t}`Mitsos2009` (the active McCormick plane's gradient is chosen by the
same predicate that selects its value — never by autodiff of a possibly non-convex
construction). The resulting convex relaxation is bounded by Kelley cutting planes
solved as a linear program over the DOF only — no auxiliary columns.

You can query this reduced-space bound directly:

```python
from discopt._jax.mccormick_subgradient import reduced_mccormick_lp_bound

rb = reduced_mccormick_lp_bound(m, [0.2, 0.2], [2.0, 2.0])
print(rb.status, rb.bound)   # -> optimal  -0.8868 (a valid lower bound)
```

## Why branch on the DOF only? The tree-size payoff

The value of reduced space is not a tighter per-node bound — on a given box the
reduced-space McCormick bound is no tighter (often looser) than the fully lifted
formulation. The payoff is **dimensional**: spatial branch-and-bound cost grows with
the number of branched variables, and the reduced tree subdivides only the DOF, never
the intermediates.

A controlled experiment makes this concrete. Take a three-unit version of the cascade
(4 continuous DOF + 1 integer feed level) and solve it two ways with the *same*
bounding engine and the *same* best-first spatial branch-and-bound driver, closing the
gap to the same shared optimum:

| Formulation | Branching variables | Nodes to certify |
|---|---|---|
| **Reduced** (`CustomCall` units, DOF only) | 4 | **29** |
| **Full-space** (every `e_i`, `c_i` an explicit variable) | 10 | **174** |

The reduced tree is **6× smaller** and both certify the *same* optimum. Only the
variable set differs; the relaxation engine and the search are identical, so the
node-count ratio isolates the DOF-tree-size effect — the regime
{cite:t}`Bongartz2018` report for MAiNGO. (Both bounds are valid lower bounds ≤ the
true optimum; reduced-space is a bound-changing, opt-in path that never weakens the
certificate.)

## Integer degrees of freedom

The same admission test unlocks **integer** DOF: if every `CustomCall` in the model is
MCBox-relaxable, the tree branches the integer and continuous DOF together with
reduced-space node bounds and certifies globally. A model whose opaque body is *not*
MCBox-relaxable, combined with integers, has no valid node relaxation, so the solver
refuses loudly rather than return an uncertified or invalid result:

```python
m = dm.Model("int_dof")
x = m.continuous("x", lb=-10, ub=10)
k = m.integer("k", shape=(1,), lb=0, ub=4)
m.minimize(dm.custom(lambda x: (x - 3.0) ** 2)(x) + 1.5 * k[0])
result = m.solve()            # certified global: x* = 3, k* = 0
```

## Sound-or-refuse: the boundary of the reduced-space scope

The reduced-space relaxation is deliberately conservative. It **refuses** anything it
cannot bound soundly, and the solver then keeps a sound fallback:

- a body written against raw `jnp` intrinsics (e.g. `jnp.sin(x)`) that `MCBox` does not
  intercept;
- a **non-affine hidden division** (`x / (y*z)`): the general non-affine reciprocal
  subgradient is not validated, so it is refused (continuous models fall back to the
  lifted/local path; integer models raise);
- a fractional power `x**a` (non-integer `a`) over a base that can reach `x ≤ 0`, where
  `x**a` is undefined — a no-information bracket, never a wrong finite bound;
- a non-scalar leaf or an unbounded box.

In every case discopt prefers a loud refusal or a sound fallback to a partial global
bound — a global solver's product is its certificate, and the certificate is never
weakened for coverage.

## Summary

`dm.custom` lets you optimize over opaque, hidden-DOF models — flowsheets, embedded
surrogates, external JAX subroutines — and, when the body is MCBox-relaxable, get a
**certified global** solution that branches only on your true degrees of freedom, with
the internal intermediates hidden. This brings discopt to capability parity with the
reduced-space McCormick approach of {cite:t}`Bongartz2018`, built on the multivariate
composition rules of {cite:t}`Tsoukalas2014` and the McCormick subgradient propagation
of {cite:t}`Mitsos2009`.
