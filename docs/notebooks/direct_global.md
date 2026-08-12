# Black-box global search with DIRECT (`solver="direct"`)

Most of discopt's machinery rests on being able to *look inside* your model. The
relaxation layer reads the expression DAG, builds convex under- and
over-estimators, and branch-and-bound turns those into a certificate. That is the
product: not just an answer, but a proof.

Sometimes there is nothing to look inside. A flowsheet unit that calls a
simulator, a lookup into tabulated data, a function written with raw `jax.numpy`
intrinsics — wrapped in `dm.custom(...)` these become a `CustomCall` node that
discopt can evaluate and autodifferentiate but cannot relax. When such a body
falls outside the reduced-space `MCBox` scope (see
{doc}`reduced_space_customcall` for when it does *not*), there is no bounding step
to run, and the default path degrades to a single local NLP: a feasible point, no
global search, no certificate.

`solver="direct"` fills that gap. It is the DIRECT algorithm of
{cite:t}`Jones1993` — DIviding RECTangles — with the modifications
{cite:t}`JonesMartins2021` conclude are generally beneficial. It needs only a
finite box and the ability to evaluate the model at a point.

```{warning}
DIRECT returns **no certificate**. `bound` and `gap` are `None`, `gap_certified`
is `False`, and the status is never `"optimal"`. If your model *can* be written
algebraically, use the default solver — it will certify the answer, and DIRECT
will not.
```

## How it works

DIRECT normalizes the box to the unit hypercube and partitions it into
hyperrectangles, each with the objective evaluated at its centre. Each iteration
it selects the *potentially optimal* rectangles, subdivides them, and samples the
new centres.

"Potentially optimal" is the idea that removes the need for a Lipschitz constant.
Given a constant $K$, a rectangle $i$ with centre value $f(c_i)$ and
centre-to-vertex distance $d_i$ has Lipschitz lower bound $f(c_i) - K d_i$. Rather
than fix $K$ — which must be large to be valid, and a large $K$ means almost pure
exploration — DIRECT selects every rectangle that would have the best bound for
*some* $K > 0$:

$$f(c_j) - K d_j \le f(c_i) - K d_i \quad \forall i, \qquad
  f(c_j) - K d_j \le f_{\min} - \varepsilon |f_{\min}|.$$

The first condition is the **lower-right convex hull** of the $(d, f)$ scatter:
plot every rectangle by size and centre value, and the selected ones lie on the
lower boundary running down and to the right. Small-and-good rectangles get
picked for refinement, large-and-mediocre ones for exploration, and the hull
interpolates between them — with no tuning parameter setting the balance.

```python
import numpy as np
import matplotlib.pyplot as plt
from discopt.solvers.direct import select_potentially_optimal

sizes = np.repeat([0.05, 0.09, 0.16, 0.29, 0.5, 0.87], 4)
values = np.array([3.1, 3.4, 3.9, 4.4, 2.4, 3.0, 3.6, 4.1, 2.0, 2.7, 3.4, 4.0,
                   1.6, 2.6, 3.5, 4.2, 1.9, 3.0, 3.9, 4.6, 2.6, 3.6, 4.4, 5.1])

# One candidate per size level: the best-valued rectangle of that size.
best = {}
for i, (d, f) in enumerate(zip(sizes, values)):
    if d not in best or f < values[best[d]]:
        best[d] = i
idx = list(best.values())
sel = [idx[c] for c in select_potentially_optimal(sizes[idx], values[idx], epsilon=1e-4)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(sizes, values, s=28, c="#b0b7c3", label="rectangles")
ax.scatter(sizes[sel], values[sel], s=90, c="#c0392b", zorder=3, label="potentially optimal")
ax.plot(sizes[sel], values[sel], c="#c0392b", lw=1, alpha=0.5)
ax.set_xlabel("centre-vertex distance $d$   (bigger = less explored)")
ax.set_ylabel("centre value $f(c)$")
ax.legend(frameon=False)
```

The three selected rectangles are `(d=0.29, f=1.6)`, `(0.5, 1.9)` and
`(0.87, 2.6)`: the best-valued mid-size rectangle at one end, the largest
rectangle at the other. Everything smaller-and-worse is dominated — no $K > 0$
makes it look best.

## Worked example 1 — the motivating case

An objective discopt cannot relax. The body applies raw `jnp` intrinsics to its
argument, which puts it outside the `MCBox` scope.

```python
import math
import jax.numpy as jnp
import discopt.modeling as dm

def ackley(v):
    n = v.shape[0]
    return (-20 * jnp.exp(-0.2 * jnp.sqrt(jnp.sum(v**2) / n))
            - jnp.exp(jnp.sum(jnp.cos(2 * math.pi * v)) / n) + 20 + math.e)

def build():
    m = dm.Model("ackley")
    x = m.continuous("x", shape=2, lb=-25.768, ub=39.768)
    m.minimize(dm.custom(ackley, name="ackley")(x))
    return m

default = build().solve(time_limit=60)
direct = build().solve(solver="direct", max_evals=2000, time_limit=60)

print(f"default path : {default.objective:.6f}   (local NLP only)")
print(f"solver=direct: {direct.objective:.6f}")
print(f"true optimum : 0.0")
```

```
default path : 15.063514   (local NLP only)
solver=direct: 0.000000
true optimum : 0.0
```

The default path stalls in whichever basin its start point lands in. DIRECT's
systematic subdivision finds the global basin, and local refinement polishes it.

```{note}
The box is deliberately **asymmetric** around the optimum. DIRECT's very first
evaluation is the centre of the box, so a symmetric box around an optimum at the
origin is "solved" at evaluation 1 — flattering and completely uninformative.
This caught us out while building the entry experiment for this backend.
```

## Worked example 2 — samples cluster at *every* global optimum

Branin has three global minima. {cite:t}`JonesMartins2021` highlight that DIRECT's
samples cluster around all of them, which is what makes the sampled set a good
source of local-search starting points — the basis of the `glcCluster` variant.

```python
from discopt.solvers.direct import _DirectSearch

def branin_np(v):
    x1, x2 = v[0], v[1]
    return ((x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6)**2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)

lb, ub = np.array([-5.0, 0.0]), np.array([10.0, 15.0])
search = _DirectSearch(lb, ub)
search.run(lambda x: (float(branin_np(x)), 0.0), max_evals=400)
pts = np.array([search.to_model_point(c) for c in search.part.centers])

for mx, my in [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]:
    d = np.min(np.linalg.norm(pts - np.array([mx, my]), axis=1))
    print(f"nearest sample to ({mx:7.4f}, {my:6.3f}): {d:.4f}")
```

```
nearest sample to (-3.1416, 12.275): 0.0434
nearest sample to ( 3.1416,  2.275): 0.0101
nearest sample to ( 9.4248,  2.475): 0.0003
```

All three minima are found in 400 evaluations, without the search ever being told
there is more than one.

## Worked example 3 — the local-refinement hybrid

{cite:t}`JonesMartins2021` call hybridizing with a local optimizer *the* key
enabler, and the reason shows up plainly: DIRECT finds the basin quickly and
refines slowly. Shubert is the case they use to make the point.

```python
def shubert():
    m = dm.Model("shubert")
    x = m.continuous("x", shape=2, lb=-10.0, ub=10.0)
    body = lambda v: (sum((i + 1) * jnp.cos((i + 2) * v[0] + (i + 1)) for i in range(5))
                      * sum((i + 1) * jnp.cos((i + 2) * v[1] + (i + 1)) for i in range(5)))
    m.minimize(dm.custom(body, name="shubert")(x))
    return m

for refine in (False, True):
    r = shubert().solve(solver="direct", max_evals=1200, local_refine=refine, time_limit=60)
    print(f"local_refine={str(refine):<5}  objective = {r.objective:.6f}")
```

```
local_refine=False  objective = -123.576766
local_refine=True   objective = -186.730909
```

Same evaluation budget. Without refinement the search is still climbing towards
the optimum; with it, the answer is exact to six decimals (the published optimum
is −186.7309).

### Which refiner?

`local_refine_method` chooses between discopt's gradient-based NLP and Powell
{cite:p}`Powell1964`, which uses no derivatives at all. `"auto"` — the default —
runs the NLP and falls back to Powell when it fails to move.

That fallback matters more than it sounds. A `dm.custom` body is JAX-*traceable*
by construction, but traceable is not the same as usefully *differentiable*: a
body containing `jnp.floor`, a table lookup, or a simulator behind
`jax.pure_callback` returns zero or meaningless gradients — and a gradient method
will then sit perfectly still **while reporting success**.

```python
def staircase():
    m = dm.Model("staircase")
    x = m.continuous("x", shape=2, lb=-3.0, ub=5.0)
    body = lambda v: jnp.sum((jnp.floor(v * 4.0) / 4.0 - 1.25)**2) + 0.01 * jnp.sum(v**2)
    m.minimize(dm.custom(body, name="stair")(x))
    return m

for method in ("nlp", "derivative-free", "auto"):
    r = staircase().solve(solver="direct", max_evals=1000,
                          local_refine_method=method, time_limit=60)
    print(f"{method:>16}: {r.objective:.8f}")
```

```
             nlp: 0.03127896
 derivative-free: 0.03125000
            auto: 0.03125000
```

On a smooth objective the ordering reverses — the gradient method wins — and
`"auto"` again takes the better of the two.

## Worked example 4 — why `direct_variant="gl"` is not the default

DIRECT-GL {cite:p}`Stripinis2019` replaces the convex-hull selection with two
steps: a *global* step taking every rectangle Pareto-optimal on (low value, large
size), and a *local* step taking those Pareto-optimal on (close to the incumbent,
large size). It is markedly better on some problems and markedly worse on others,
which is exactly why it is opt-in.

Final objective at a fixed budget cannot show this — given enough evaluations both
converge and the comparison is noise. The meaningful measure is evaluations to a
target accuracy:

```python
import sys; sys.path.insert(0, "../../python/tests")
from support import direct_testfuncs as tfs

def evals_to(tf, variant, tol=1e-2, budget=6000):
    s = _DirectSearch(tf.lb, tf.ub, variant=variant)
    hist = []
    s.run(lambda x: (float(tf.np_body(x)), 0.0), budget,
          on_iteration=lambda se: hist.append((se.stats.evals, se.best_feasible_value)))
    return next((e for e, v in hist if v is not None and tf.relative_error(v) <= tol), None)

for name in ("hartman_6", "shubert"):
    tf = tfs.get(name)
    print(f"{name:<12} classic={evals_to(tf, 'classic'):>5}   gl={evals_to(tf, 'gl'):>5}")
```

```
hartman_6    classic=  105   gl=  277
shubert      classic= 2269   gl=  181
```

GL wins Shubert by more than an order of magnitude and loses Hartman-6 by a
factor of a few — the same asymmetry {cite:t}`JonesMartins2021` report (they
measure 571 vs 8793 on Hartman-6, and 2967 vs 425 on Shubert). With no way to know
in advance which kind of problem you have, the more predictable rule is the safer
default.

## What you do not get

```python
r = build().solve(solver="direct", max_evals=500, time_limit=60)
print(f"status         = {r.status!r}")
print(f"bound          = {r.bound}")
print(f"gap            = {r.gap}")
print(f"gap_certified  = {r.gap_certified}")
```

```
status         = 'feasible'
bound          = None
gap            = None
gap_certified  = False
```

DIRECT is a sampling method with no dual information, so there is no bound and no
gap. An exhausted budget is reported as a limit status — never `"infeasible"`,
because DIRECT cannot prove infeasibility either. Reporting the incumbent as a
bound would be a false certificate, so the backend refuses to.

It also declines to guess where it cannot proceed: a non-finite box raises rather
than silently substituting a big-M, because only you know the real range.

## Where this sits in derivative-free optimization

**This is a strong baseline, not the state of the art**, and it is worth being
plain about that. DIRECT is from 1993.

What it offers: determinism (the same answer every run), essentially one
hyperparameter, no surrogate to fit, no dependency beyond numpy/scipy, native
handling of discopt's integer variables, and sampling that is dense in the limit.
That makes it dependable, and a fair yardstick.

What it is not: competitive with model-based methods on expensive objectives. If
an evaluation costs minutes, a surrogate method reaches a comparable answer in far
fewer calls, because it spends real computation deciding where to sample —
{cite:t}`Jones1998` is the founding idea, modern Bayesian optimization the
developed form. For local refinement, trust-region methods (BOBYQA, DFO-LS) beat
DIRECT badly, which is why the hybrid above exists. For constrained black-box work
at scale, MADS-family solvers {cite:p}`AuditDennis2006` are the mature choice; for
mixed-integer black-box problems, RBF surrogates {cite:p}`CostaNannicini2018` are
strong. {cite:t}`RiosSahinidis2013` is the standard comparison.

The caveat that keeps the family relevant: DIRECT *hybridized with a local solver*
is still competitive — {cite:t}`JonesMartins2021` note `glcCluster` among the top
performers in a comparison on problems up to 300 variables. That hybrid is the
shape implemented here.

### Choosing

| your situation | use |
|---|---|
| The model is algebraic | the default solver — you get a certificate |
| Opaque body that traces through `MCBox` | the default solver — still certified, see {doc}`reduced_space_customcall` |
| Opaque body, evaluation is cheap (ms–s) | `solver="direct"` |
| Opaque body, evaluation is expensive (minutes+) | a surrogate method; DIRECT will spend hundreds of calls |
| Experiments taking hours or days, human in the loop | a sequential design campaign, not a blocking solve |
| You need a *proof*, not a good point | no derivative-free method will give you one |

## Options reference

| option | default | meaning |
|---|---|---|
| `max_evals` | `5000` | evaluation budget — the cost control; cached repeats are free |
| `epsilon` | `1e-4` | Eq. (4) floor; with refinement on, ~`1e-2` is the survey's advice |
| `direct_variant` | `"classic"` | `"gl"` for DIRECT-GL's two-step selection |
| `divide` | `"one"` | `"all"` restores the 1993 all-long-sides rule |
| `break_ties` | `True` | `False` selects every rectangle tied for potentially optimal |
| `local_refine` | `True` | the hybrid; the survey's most-endorsed modification |
| `local_refine_after` | `100` | evaluations between refinement attempts |
| `local_refine_method` | `"auto"` | `"nlp"`, `"derivative-free"`, or auto-fallback |
| `feasibility_tolerance` | `1e-6` | the GLce band treated as feasible |

## References

```{bibliography}
:filter: docname in docnames
```
