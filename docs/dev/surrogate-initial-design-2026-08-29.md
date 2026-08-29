# Surrogate backend — initial-design sizing (2026-08-29)

Entry experiment for issue #1036, run **before** the implementation, per
Dev-Philosophy §4. Subject: the default initial-design rule in
`python/discopt/solvers/surrogate.py`.

Reproduction: the scripts named below, all under `scratchpad/i1036/` —
`sweep.py`, `nesting_probe.py`, `design_experiment.py` (+ `analyze.py`),
`rederive_budgets.py`, `direct_evals.py`, `cost_model.py`. Each prints an
executed-run count and exits non-zero when it is zero (Dev-Philosophy §6).

Module under test asserted in every run: `discopt.solvers.surrogate.__file__` is
printed with the results and is the in-tree file, not an installed copy
(Dev-Philosophy §8).

---

## What #1036 reported, and what reproducing it found

The report: `test_reaches_the_published_optimum_within_a_small_budget[hartman_3-0.01-100]`
fails at relative error 1.1202e-2 against a 1e-2 tolerance, deterministically on
the default seed.

**It does not reproduce on this machine.** The same panel, same seed, same tree,
passes 5/5, and the budget sweep the issue reports as non-monotone comes back
monotone (`sweep.py`):

| max_evals | design | objective | rel_err | issue's reading |
|---|---|---|---|---|
| 46 | 23 | −3.857616 | 1.34e−3 | −3.857687 |
| 60 | 30 | −3.861426 | 3.50e−4 | −3.650674 |
| 80 | 30 | −3.862710 | 1.82e−5 | −3.793509 |
| 100 | 30 | −3.862754 | 6.73e−6 | −3.819510 |
| 120 | 30 | −3.862754 | 6.73e−6 | −3.848625 |
| 150 | 30 | −3.862779 | 1.73e−7 | −3.859252 |
| 200 | 30 | −3.862779 | 1.73e−7 | −3.862737 |

That is not a contradiction of the report; it is the finding. The search is
deterministic given a seed but *chaotic*, so a different BLAS is enough to move
one seed into a neighbouring basin. A single-seed pass/fail on a hand-picked
budget therefore asserts a property of one machine's floating point, not of the
method — which is half of what has to change in the test.

## The mechanism, and a correction to the issue's diagnosis

The issue attributes the non-monotone column to "each row is a *different
trajectory*, not a longer prefix of one trajectory". That is right for the 46
row and **wrong for the rest**. The rule was

```python
n_design = max(n_vars + 2, min(10 * n_vars, max(1, max_evals // 2)))
```

so for `hartman_3` (n = 3) the design is `min(30, max_evals // 2)`: 23 at budget
46, and **30 at every budget from 60 to 200**. Rows 60–200 share one trajectory
and are strictly monotone in both columns above. Only the 46 row is a different
search.

The defect is therefore sharper than "budgets are not comparable": the design
size is a *step function of the budget*, and two budgets are comparable exactly
when they land on the same step. `branin` (n = 2, `10n` = 20) is on the same step
for every budget the panel uses, which is why the panel looked healthy while its
stated rationale was unsound.

Measured directly on the `on_evaluation` traces (`nesting_probe.py`, seed 0,
budgets {40, 46, 60, 80, 100}, all pairs):

| | branin (n=2) | hartman_3 (n=3) | hartman_6 (n=6) | total |
|---|---|---|---|---|
| pairs that diverge | 0/10 | 7/10 | 10/10 | **17/30** |

Every divergence is at **evaluation 1**. `hartman_6` is the worst case because
`10n = 60` and `max_evals // 2` cross inside the budget range.

## Pre-registered hypothesis and kill criterion

> **Hypothesis.** `10n` is a sizing rule for fitting a response surface *once*,
> over a design chosen in advance. It is not a budget-allocation rule for a
> serial adaptive search, where every design point is a point not spent on the
> acquisition. A smaller, dimension-only design therefore reaches a target
> accuracy in **fewer** evaluations across the panel — not only on `hartman_3`.
>
> **Kill criterion.** The replacement must improve the panel mean of censored
> evaluations-to-1e-2 and must win on a **majority of functions** on both
> surrogate families. If it is better only on the RBF path, or only on the
> functions that motivated it, the hypothesis dies and the fix is restricted to
> decoupling the design from the budget.

Design: `max_evals` held **fixed at 100**, so `n_initial` is the only
independent variable; 12 seeds per cell; metric = evaluation at which the
incumbent first reaches 1e-2 relative error, with a non-reaching seed counted at
the full budget (`design_experiment.py`, then `analyze.py`).

## Result — the criterion was met

Censored mean evaluations to 1e-2 (lower is better); `shipped` is the rule in
force at the time, which equals `10n` wherever `10n ≤ max_evals // 2`. That arm
reads the live code, so on the post-fix tree it resolves to `2(n+1)` and
duplicates that column — reproducing the `shipped` column means restoring the old
expression in `solve_surrogate` first.

### RBF (default family), 8 functions × 12 seeds

| function | shipped | n+2 | **2(n+1)** | (n+1)(n+2)/2 | 5n | 10n |
|---|---|---|---|---|---|---|
| branin | 39.1 | 31.3 | **27.8** | 27.8 | 34.9 | 39.1 |
| six_hump_camel | 35.2 | 27.9 | **25.3** | 25.3 | 32.4 | 35.2 |
| ackley_2 | 52.8 | 46.8 | **47.9** | 47.9 | 50.8 | 52.8 |
| hartman_3 | 49.2 | 33.8 | **35.1** | 38.2 | 43.2 | 49.2 |
| goldstein_price | 92.6 | 100.0 | **92.3** | 92.3 | 98.4 | 92.6 |
| hartman_6 | 95.1 | 58.7 | **66.2** | 83.0 | 83.1 | 97.3 |
| rastrigin_2 | 91.5 | 93.5 | **97.2** | 97.2 | 94.0 | 91.5 |
| shubert | 87.0 | 86.2 | **87.8** | 87.8 | 81.1 | 87.0 |
| **panel mean** | **67.8** | 59.8 | **60.0** | 62.4 | 64.7 | 68.1 |
| seeds reaching (of 96) | 64 | 68 | 67 | 63 | 66 | 64 |
| better/worse vs shipped | — | 6/2 | **6/2** | 6/2 | 6/2 | 0/1 |

### Kriging, 6 functions × 12 seeds

| function | shipped | **2(n+1)** | 5n | 10n |
|---|---|---|---|---|
| branin | 27.0 | **24.2** | 22.3 | 27.0 |
| six_hump_camel | 37.1 | **32.2** | 35.0 | 37.1 |
| ackley_2 | 100.0 | 100.0 | 100.0 | 100.0 |
| hartman_3 | 33.4 | **17.1** | 18.7 | 33.4 |
| goldstein_price | 97.3 | **98.8** | 98.6 | 97.3 |
| rastrigin_2 | 98.6 | **94.2** | 98.8 | 98.6 |
| **panel mean** | **65.6** | **61.1** | 62.2 | 65.6 |
| better/worse vs shipped | — | **4/1** | 3/2 | 0/0 |

## Why `2(n+1)` and not `n+2`

`n+2` edges out `2(n+1)` on the RBF panel mean (59.8 vs 60.0 — a 0.3%
difference, well inside the seed noise) and is clearly better on `hartman_6`
(58.7 vs 66.2). It also **collapses on `goldstein_price`**: 0 of 12 seeds reach
the tolerance, against 4 of 12 for the shipped rule. That is the one function in
the panel whose objective spans ~10⁶, and a 4-point design cannot even locate
its scale.

`2(n+1)` never does materially worse than the shipped rule on any function
(worst case `rastrigin_2`, 91.5 → 97.2, on a function where 1–3 seeds in 12
reach the tolerance under any arm), wins on 6 of 8 with RBF and 4 of 6 with
kriging, and stays at or above the `n+1` an RBF with a linear tail needs to be
fitted at all, for every `n`. One rule, both families — the two tables point the
same way, so a family-dependent rule would be unsupported complexity.

**Not a free win, stated as such.** The two functions that lose are the two the
module docstring already names as the hard shapes for this backend: a sharply
scaled objective (`goldstein_price`, whose remedy is the unimplemented monotone
objective transformation) and a densely multimodal one (`rastrigin_2`). Both
prefer a *larger* design, and no single size wins every n = 2 function on this
panel — that heterogeneity is real, not a tuning target. `n_initial` remains the
override and the docstring now says which two shapes to raise it for.

## After the change

* `nesting_probe.py`: **0 of 30** budget pairs diverge, from 17 of 30.
* The head-to-head-against-DIRECT table in the module docstring was re-taken.
  It is a 3-seed slice and moves in both directions: `six_hump_camel` 32 → 23
  and `branin` 38 → 36 improve, `goldstein_price` gets worse, `hartman_3`'s
  median rises 46 → 50 *because* all three seeds now reach the tolerance where
  two did, and `ackley_2`'s falls 48 → 44.5 *because* one seed stopped reaching
  it inside 60 evaluations. It is annotated in place as a slice rather than a
  verdict; the verdict is the 12-seed panel above.
* `rederive_budgets.py`, 8 seeds, budget 200 — every seed reaches 1e-2 on every
  panel function, and because the trajectories now nest, one trace per seed
  yields the first-reach evaluation for *every* budget at once:

  | function | per-seed first reach (seeds 0–7) | 8/8 by | panel budget |
  |---|---|---|---|
  | branin | 30, 36, 41, 42, 16, 36, 23, 22 | 42 | 65 |
  | six_hump_camel | 23, 34, 17, 29, 18, 17, 22, 12 | 34 | 55 |
  | ackley_2 | 42, 64, 47, 46, 46, 48, 52, 53 | 64 | 100 |
  | hartman_3 | 60, 50, 19, 17, 24, 25, 20, 41 | 60 | 90 |
  | goldstein_price | 120, 143, 131, 150, 82, 144, 132, 84 | 150 | 225 |

  The old `goldstein_price` budget of 130 would have been reached by 4 of these
  8 seeds — it was passing on the default seed's luck, exactly as the issue
  suspected of the four budgets it did not investigate.

## Falsifications and corrections recorded here

* The issue's "the result is **non-monotone** in the budget … each row is a
  different trajectory" holds only for the 46-evaluation row. Budgets 60–200 all
  select a 30-point design and form a single monotone trajectory. Corrected
  above; the sharper statement is that the design is a step function of the
  budget.
* The reported `hartman_3` failure does not reproduce on this machine
  (Linux/OpenBLAS, numpy 2.5.2). The failure is real on the reporting machine;
  what it demonstrates is that a single-seed convergence assertion on a chaotic
  search is a platform assertion. This is why the panel test now asserts a
  population statistic.
* The cost-model figure in the module docstring ("the 15-point initial design
  costs 0.8 s") was made false by this change and has been re-measured to the
  6-point design (0.29 s). The neighbouring 20 s-vs-2 s table has **not** been
  re-taken and is now labelled as pre-#1036 rather than left to read as current.

## Not addressed here

The convergence panel is marked `slow` but not `correctness`, and CI selects
`not slow` (fast lane), `correctness and not slow` (PR correctness) or
`correctness and slow` (on-demand). **744 tests in `python/tests/` are `slow`
and not `correctness`, and no lane selects them** — a strictly larger hole than
"the slow tier is on demand". The owner has already ruled on the cadence
question in #1034 ("I don't want to run benchmarks in CI, they take too long,
and are typically something we do prior to releases"), so widening a lane is
their call and is not taken here.
