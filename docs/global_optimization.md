# Global optimization: what discopt can and can't certify

discopt's branch-and-bound engine is a *deterministic global* optimizer: when it
returns `status="optimal"` it also returns a **certificate** — a valid lower
bound on the true optimum together with a feasible incumbent — so the reported
solution is provably within the optimality gap of the global optimum
{cite:p}`Belotti2013,Tawarmalani2005`. This is fundamentally different from a
local NLP solve, which only finds a stationary point and can get stuck in a poor
local minimum on a nonconvex problem.

That guarantee is not free, and it does not apply to every model you can write.
This guide explains **which problems are amenable** to certified global solving in
discopt, **which are clearly not**, and what to do about the gap between the two.
It complements rather than repeats the mechanics covered elsewhere: the
{doc}`AMP global solver <notebooks/amp_global_minlp>`, {doc}`convexity detection
<notebooks/convexity_detection>`, {doc}`bound tightening <notebooks/bound_tightening>`,
and the {doc}`relaxation internals <notebooks/solver_internals>`.

## Did I actually get a certified global optimum?

Every `SolveResult` carries the evidence. You do not have to take `status` on
faith — inspect the bound and the gap:

```python
result = model.solve()

result.status          # "optimal" only when the gap is closed within tolerance
result.objective       # the incumbent (best feasible objective found)
result.bound           # the dual bound (valid LB for min / UB for max)
result.gap             # |incumbent - bound| / |incumbent|
result.gap_certified   # True when `bound` is a *sound* global bound
```

A result is a **certified global optimum** when `status == "optimal"`,
`gap_certified is True`, and `gap` is at or below your tolerance. If instead you
see `status == "feasible"`, a non-`None` gap that never closed, or
`gap_certified is False`, then discopt found a feasible point but **could not
prove** it is global — treat it as a (possibly very good) heuristic solution, not
a certificate.

```{important}
`gap_certified` is the single most important field to check on a nonconvex model.
A small `gap` with `gap_certified=False` means the bound itself is not trustworthy
as a global guarantee.
```

## What makes a problem amenable

Certified global solving in discopt rests on two requirements. Meet both and the
spatial branch-and-bound will converge with a sound certificate; violate either
and the guarantee weakens or disappears.

### 1. Finite variable bounds

The relaxations that produce the dual bound are built over the variable box. A
nonconvex term such as a product $x\,y$ is replaced by its McCormick envelope
{cite:p}`McCormick1976`, whose tightness depends entirely on the finite bounds
$[x^L,x^U]\times[y^L,y^U]$. **Without finite bounds there is no valid envelope and
no valid global bound.** discopt warns when declared bounds are missing or
enormous (it treats $\pm 10^{20}$ as "unbounded"), and a model left that way will
typically return `status="feasible"` with no certified gap rather than
`status="optimal"`.

Give every variable that participates in a nonconvex term a realistic finite
range:

```python
x = m.continuous("x", lb=0, ub=1000)     # bounded → envelopes are valid & tight
theta = m.continuous("theta", lb=-3.15, ub=3.15)
```

Bounds need not be tight by hand — discopt's {doc}`FBBT/OBBT presolve
<notebooks/bound_tightening>` will contract them automatically — but they must be
*finite*.

### 2. Factorable structure from supported intrinsics

discopt builds relaxations by decomposing each expression into elementary
operations it knows how to under- and over-estimate — *factorable* programming
{cite:p}`Tawarmalani2005`. The relaxation library covers a broad menu of
intrinsics, each with a convex/concave envelope or αBB underestimator
{cite:p}`Adjiman1998`:

- **Bilinear, trilinear, and general multilinear products** (McCormick / RLT
  {cite:p}`Sherali1990`);
- **Polynomials and quadratics**, including QCQP (PSD/SOC cuts);
- **Fractional terms** — reciprocals, ratios of products, fractional powers;
- **Transcendentals** — `exp`, `log`, the trig family, `asin`/`acos`/`acosh`,
  `erf`; and
- **Saturating activations** — `sigmoid`, `tanh`, `softplus` (used by
  {doc}`neural-network embeddings <notebooks/nn_embedding>`).

If your model is composed of these building blocks over bounded variables, it is
amenable. Integer and binary variables are fully supported — they are branched
just like spatial branching on continuous nonconvex terms — as is
{doc}`generalized disjunctive programming <notebooks/tutorial_gdp>` once
reformulated.

### Amenable problem classes (worked elsewhere)

| Class | discopt support | See |
|-------|-----------------|-----|
| Bilinear / QCQP | McCormick + RLT/PSD/SOC cuts | {doc}`notebooks/solver_internals` |
| Pooling / blending | pq-formulation builder | {doc}`notebooks/pooling_pq` |
| Signomial / posynomial | factorable lift; GP log-transform | {doc}`notebooks/geometric_programming` |
| Polynomial / concave | piecewise-McCormick, αBB | {doc}`notebooks/amp_global_minlp` |
| MINLP (general nonconvex) | spatial B&B; AMP for hard cases | {doc}`notebooks/tutorial_minlp` |
| Embedded ML models | big-M / full-space formulations | {doc}`notebooks/nn_embedding` |

## What is clearly *not* amenable

The following do **not** yield a certified global optimum. In most cases discopt
will still return a feasible point (from its local solves and primal heuristics),
but the global *guarantee* is gone — so do not read `result.objective` as global.

- **Unbounded (or effectively unbounded) variables in nonconvex terms.** No finite
  box ⇒ no valid envelope ⇒ no sound bound. This is the most common cause of a
  `status="feasible"`, `gap_certified=False` result. *Fix:* add finite bounds.

- **Black-box / user-defined functions** (`udf`, `custom`, `CustomCall`). These
  call arbitrary Python and have **no algebraic relaxation**, so the bounding step
  cannot reason about them. They are evaluated pointwise during local NLP solves
  only — useful for modeling convenience, but they reduce the run to local
  optimization with no global certificate.

- **External simulators, table lookups, or interpolated data** used as objective
  or constraint terms. Same reason as black-box functions: discopt cannot
  construct a valid under/over-estimator for something it cannot see symbolically.

- **Operators with no implemented envelope, or genuinely non-factorable
  expressions.** If an intrinsic is outside the supported menu above, or an
  expression cannot be decomposed into supported elementary operations, no sound
  relaxation is available.

- **Discontinuous or pathologically non-smooth terms** (step functions, `floor`,
  expressions with poles inside the variable box, regions where the function is
  undefined / returns `NaN`). Convex relaxation theory assumes the function is
  defined and finite on the box.

- **Very high-dimensional nonconvexity.** This is a *tractability* limit, not a
  correctness one: the certificate remains sound, but spatial branch-and-bound
  scales poorly with the number of nonconvex terms and integer variables. A model
  with hundreds of independent bilinear terms or a deep combinatorial structure
  may not close its gap within any reasonable time budget. Expect
  `status="feasible"` at the time limit with a non-zero gap.

```{warning}
A black-box `udf` does not raise an error — it silently turns a global solve into
a local one. If you need a *certificate*, keep the model algebraic and built from
the supported intrinsics.
```

## A quick amenability checklist

Before expecting a certified global optimum, confirm:

1. **Bounds.** Does every variable in a nonlinear term have finite `lb`/`ub`? (No
   bounds warning at solve time.)
2. **Algebraic.** Is every nonlinearity written with discopt intrinsics — no
   `udf`/`custom`, no external calls, no table lookups?
3. **Supported.** Are all operators in the relaxation menu (products, polynomials,
   fractional, the transcendental/activation list)?
4. **Smooth & defined.** Is every function finite and free of poles/`NaN` across
   the whole box (no `floor`/step inside nonconvex terms)?
5. **Sized sensibly.** Is the count of nonconvex terms and integers small enough to
   close the gap in your time budget?

If all five are "yes," `solve()` should return `status="optimal"` with
`gap_certified=True`. If one is "no," see the remedies below.

## Remediation

- **Add or tighten bounds.** Provide finite `lb`/`ub`; let
  {doc}`FBBT/OBBT presolve <notebooks/bound_tightening>` contract them further.
- **Reformulate to expose structure.** Use the {doc}`pooling pq-formulation
  <notebooks/pooling_pq>` for blending, the {doc}`geometric-programming
  <notebooks/geometric_programming>` log-transform for posynomials, or
  {doc}`GDP <notebooks/tutorial_gdp>` for logical/disjunctive structure — each
  turns a hard or unsupported form into one discopt relaxes well.
- **Replace black boxes with algebra.** If a `udf` wraps something expressible in
  intrinsics, write it out; if it wraps a trained model, embed it as
  {doc}`algebraic NN constraints <notebooks/nn_embedding>` instead.
- **Escalate to AMP for hard nonconvex MINLPs.** The {doc}`Adaptive Multivariate
  Partitioning solver <notebooks/amp_global_minlp>` {cite:p}`Nagarajan2019`
  (`model.solve(solver="amp")`) refines a piecewise relaxation where the gap is
  largest and is often the better choice when the default spatial branch-and-bound
  stalls. Tighter cuts (`rlt_cuts=True`, PSD/SOC separators) can also close stubborn
  gaps — see {doc}`notebooks/solver_internals`.
- **Accept a local solution deliberately.** If global is genuinely out of reach,
  read `result.objective` as a high-quality heuristic value and report it as such
  — but check `gap_certified` so you *know* that is what you have.

## Engine choice at a glance

| Situation | Recommended call |
|-----------|------------------|
| Convex, or bounded factorable nonconvex (default) | `model.solve()` |
| Hard nonconvex MINLP, default B&B stalls | `model.solve(solver="amp")` |
| Hard nonconvex MINLP with Gurobi available for MILP masters | `model.solve(solver="amp", milp_solver="gurobi")` |
| Stubborn bilinear/QCQP gap | `model.solve(rlt_cuts=True)` (+ PSD/SOC cuts) |
| Only a feasible/local point is needed quickly | `model.solve(time_limit=...)`, read the incumbent |

`solver="gurobi"` is a direct matrix backend for LP/MILP/QP/QCP-family models.
It intentionally does not translate arbitrary nonlinear expression DAGs into
Gurobi nonlinear constraints. For general NLP/MINLP, use discopt's global
algorithms and select Gurobi only where it is a matrix subsolver, for example
`solver="amp", milp_solver="gurobi"`.

The throughline: discopt gives a **sound global certificate exactly when the model
is bounded and algebraically factorable from supported intrinsics**. Keep your
formulation inside that envelope and you get a proof; step outside it and you get a
point.
