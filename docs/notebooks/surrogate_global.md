# Expensive black boxes: the surrogate backend (`solver="surrogate"`)

{doc}`direct_global` covers the case where your objective is opaque but *cheap* —
DIRECT will happily spend thousands of evaluations mapping the box. This page is
the other regime: one evaluation costs minutes or hours, you can afford tens of
them, and it is worth spending real computation deciding where each one goes.

The method is the classical one. Fit a cheap *response surface* to the points
evaluated so far, maximize an acquisition function over that surface to choose
the next point, evaluate there, refit, repeat. {cite:t}`Jones1998` is the
founding paper for the Gaussian-process form; {cite:t}`Gutmann2001` and
{cite:t}`RegisShoemaker2005` for the radial-basis form used here by default.

```{warning}
Like DIRECT, this returns **no certificate** for your problem. `bound` and `gap`
are `None`, `gap_certified` is `False`, the status is never `"optimal"`. A
surrogate is a *model of* the objective, not a bound on it — an interpolant says
nothing rigorous about the function between its data points.
```

## Two families, one interface

`surrogate="rbf"` (**default**) fits

$$s(x) = \sum_i \lambda_i \varphi(\lVert x - x_i \rVert) + p(x)$$

with a cubic, thin-plate or linear kernel and a linear tail. RBF is the default
rather than a Gaussian process for three reasons, in order of weight: **integer
variables work natively** (discopt is a MINLP solver, and a black-box method that
cannot take integers is half a feature); fitting is one symmetric linear solve,
so there is no likelihood to maximize and no silent "the fit did not converge";
and it degrades better with dimension, since kriging must estimate one length
scale per dimension from `m` points.

`surrogate="kriging"` is the {cite:t}`Jones1998` route — a DACE model
{cite:p}`Sacks1989` plus expected improvement — with a **nugget** added so a
noisy objective is not forced through its own measurement error. Use it for
smooth, low-dimensional, genuinely expensive problems.

```python
import sys; sys.path.insert(0, "../../python/tests")
from support import direct_testfuncs as tfs

tf = tfs.get("branin")                 # opaque dm.custom body, true optimum 0.397887
model, _ = tfs.build_model(tf)
r = model.solve(solver="surrogate", max_evals=30, time_limit=900)

print(f"status={r.status}  objective={r.objective:.6f}")
print(f"bound={r.bound}  gap={r.gap}  gap_certified={r.gap_certified}")
```

```
status=feasible  objective=0.613531
bound=None  gap=None  gap_certified=False
```

Thirty evaluations is not convergence — the backend's own test suite reports
branin reaching 1% relative accuracy at 38. The point of the number above is the
second line.

## The cost model — read this before choosing this backend

Nearly all the wall clock is the *acquisition* solve, not your objective. On
branin with a free objective and `max_evals=30`, instrumented per evaluation:

```
evals 1-15  (initial design):   0.77 s total
evals 16-30 (each):            ~20.2 s      <- exactly acquisition_time_limit
total:                          303.8 s
```

That is the intended trade when a single evaluation dwarfs 20 s of solver time.
It is the **wrong** trade when it does not: on a cheap objective `solver="direct"`
is far faster *and* gets a better answer.

```{important}
Do not shorten `acquisition_time_limit` to make it feel faster. With the default
cubic kernel the acquisition never certifies, so the budget *looks* wasted — it
is not. It is buying primal solution quality.
```

Relative error at `max_evals=30`, measured:

| function | `acquisition_time_limit=20` | `=2` |
|---|---|---|
| branin | 0.2156 | 0.2029 |
| six_hump_camel | **0.0164** | 0.8063 |
| hartman_3 | **0.0098** | 0.0103 |

branin on its own says the short budget is free money. six_hump_camel is 49×
worse for it. This table exists because that branin-only reading was briefly
acted on during development and was wrong — a single-instance tuning of exactly
the kind this project forbids.

## The acquisition subproblem is certified — and that is not a certificate

The acquisition is an ordinary algebraic model: a sum of $\lambda_i\varphi(q_i)$
over squared distances, plus one reversed convex quadratic per design point. So
discopt's own spatial branch-and-bound can solve it to **proven global
optimality** — the thing {cite:t}`Jones1998` built a bespoke branch-and-bound for,
and that modern Bayesian-optimization libraries approximate with multistart
gradient ascent.

It certifies *where to sample next*. It says nothing about the answer.

```python
from discopt._relax.primal_heuristics import _generate_starts
from discopt.solvers.surrogate import RBFSurrogate, _SurrogateSearch, build_cors_model
import numpy as np

rng = np.random.default_rng(0)
X = _generate_starts(tf.lb, tf.ub, 8, rng)
y = np.array([float(tf.np_body(x)) for x in X])
search = _SurrogateSearch(tf.lb, tf.ub)
rbf = RBFSurrogate(kernel="linear").fit(search.normalize(X), y)

sub = build_cors_model(model, tf.lb, tf.ub, rbf, delta=0.1).solve(time_limit=20.0)
print(f"acquisition subproblem: status={sub.status} certified={sub.gap_certified}")
```

```
acquisition subproblem: status=optimal certified=True
```

**Whether it certifies is decided by the kernel.** Measured on branin with a real
fitted surrogate, relative gap of the acquisition subproblem at a 20 s budget:

| kernel | m=8 | m=20 |
|---|---|---|
| `linear` | **4e-05** (certified) | **6.9e-09** (certified) |
| `cubic` | 0.2 (not certified) | 23.9 (not certified) |

The mechanism is that cubic-RBF coefficients grow — `max|λ|` runs 10 → 164 across
that range — faster than a McCormick relaxation can bound. `rbf_kernel` stays
`"cubic"` by default because it gives better optimization quality; switch to
`"linear"` when the certificate on the subproblem is what you want:

| kernel | wall (30 evals) | acquisitions certified | objective |
|---|---|---|---|
| `cubic` | 302.5 s | 0 / 15 | 0.6135 |
| `linear` | 249.0 s | 6 / 15 | 1.1298 |

Note the trade in the last column: certifying more often did not produce a better
answer here.

```{note}
Certified **expected improvement** does *not* work, and the attempt is documented
because it was the original motivation for this backend. Built properly — the
division lifted away via $EI(x) = \max_u [d(x)\Phi(u) + s(x)\phi(u)]$ and the
dense $r^\top R^{-1} r$ whitened to $\sum v_i^2$ — branch-and-bound finds the true
acquisition maximum to five significant figures, but the **dual bound never
closes**: on branin it runs 3.82 / 23.1 / 597 / 4871 against optima
0.60 / 0.26 / 0.42 / 1.51 at m = 8/12/20/30. For kriging, discopt is an excellent
*primal* acquisition optimizer and not a certifying one.
```

## Sample efficiency, honestly

Evaluations to 1e-2 relative accuracy, from the backend's test suite:

| function | surrogate | DIRECT | |
|---|---|---|---|
| six_hump_camel | 32 | 137 | 4.3× |
| branin | 38 | 69 | 1.8× |
| hartman_3 | 46 | 79 | 1.7× |
| ackley_2 | 48 | 67 | 1.4× |
| **goldstein_price** | **96** | **75** | **0.8× — a loss** |

The advantage is real but it is roughly **2×, not an order of magnitude**, and
goldstein_price is an outright loss caused by objective dynamic range (3 → ~10⁶);
RBFOpt's monotone objective transformation {cite:p}`CostaNannicini2018` is the
known remedy and is not implemented here. DIRECT is a stronger baseline than the
surrogate literature's framing suggests.

## Choosing

| your situation | use |
|---|---|
| The model is algebraic | the default solver — you get a certificate |
| Opaque body that traces through `MCBox` | the default solver — still certified, see {doc}`reduced_space_customcall` |
| Opaque body, evaluation is cheap (ms–s) | `solver="direct"` |
| Opaque body, evaluation is expensive (minutes+) | `solver="surrogate"` |
| Experiments taking hours or days, human in the loop | a sequential design campaign, not a blocking solve |
| You need a *proof* | no derivative-free method will give you one |

## Where this sits

Closer to current practice than DIRECT, but still not a tuned BoTorch/Ax stack on
a problem that suits one. What this backend offers instead: no extra dependency
(numpy and scipy only), deterministic behaviour, native integer support, and one
modeling layer shared with the certified solver.

Two extensions are designed for but deliberately **not built**: trust-region
restriction of the acquisition domain {cite:p}`Eriksson2019`, the standard answer
to Bayesian optimization's collapse in higher dimensions; and batch proposals, so
several expensive evaluations can run in parallel. For constrained black-box work
at scale, MADS-family solvers {cite:p}`AuditDennis2006` remain the mature choice;
{cite:t}`RiosSahinidis2013` is the standard comparison.

## Options reference

| option | default | meaning |
|---|---|---|
| `max_evals` | `200` | evaluation budget — the cost model |
| `surrogate` | `"rbf"` | `"rbf"` or `"kriging"` |
| `rbf_kernel` | `"cubic"` | `"cubic"`, `"thin_plate"`, `"linear"` (the certifying one) |
| `n_initial` | auto | initial design size; default `max(n+2, min(10n, max_evals//2))` |
| `acquisition_optimizer` | `"auto"` | `"certified"` refuses rather than falling back |
| `acquisition_time_limit` | `20.0` | per-acquisition budget — see the cost model above |
| `nugget` / `estimate_nugget` | `1e-8` / `False` | kriging noise floor |
| `seed` | `0` | two runs with the same seed are identical |
| `on_evaluation` | `None` | progress hook `(evals, best)`; the only way to observe evaluations-to-target |

Full citations are on the {doc}`../references` page.
