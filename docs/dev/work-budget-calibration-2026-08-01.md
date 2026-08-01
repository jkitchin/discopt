# Work-unit calibration for the deterministic heuristic budget (issue #912)

Entry experiment and calibration for `discopt._work_budget` — the deterministic
replacement for the wall-clock "how much work" gates that made the search tree a
function of machine speed.

All numbers below were produced on this repo's in-repo MINLPLib corpus
(`python/tests/data/minlplib_nl`, 66 `.nl` files) with

```bash
python -u discopt_benchmarks/scripts/item912_clock_determinism_probe.py --classify --legacy --time-limit 15
python -u discopt_benchmarks/scripts/item912_work_unit_calibration.py --rounds 5
python -u discopt_benchmarks/scripts/item912_clock_determinism_probe.py --clockscale --alphas 1,2,4
```

Every solve runs in a child process that asserts `discopt.__file__`, the compiled
`_rust` extension path, and the arm marker (`SolverTuning.ils_work_budget`, 0 on
the legacy arm) before measuring, and raises rather than swallowing.
`gear2` — the decisive instance in #912 — is **not** in this corpus and
minlplib.org is unreachable from this environment, so everything here is
reproduced on other instances of the same class rather than on gear2 itself.

## 1. Entry experiment: does the mechanism reproduce in-repo?

Yes, on 7 instances. Legacy arm, `time_limit=15` (so the root ILS wall budget is
`min(5.0, 0.15·15) = 2.25 s`), 66 instances, 22 of which fire the root integer
local search:

| instance | nodes | ILS work (units) | ILS wall (s) | stopped on |
|---|---|---|---|---|
| nvs09 | 5 | 49 380 | 2.26 | **deadline** |
| ex1224 | 5 | 35 798 | 2.26 | **deadline** |
| st_e29 | 5 | 32 487 | 2.26 | **deadline** |
| ex1225 | 5 | 30 541 | 2.27 | **deadline** |
| tspn05 | 35 | 16 766 | 2.28 | **deadline** |
| syn05hfsg | 181 | 8 029 | 2.29 | **deadline** |
| fac2 | 39 | 4 087 | 1.06 | **deadline** |
| gkocis | 5 | 11 289 | 0.62 | converged |
| ex1221 | 5 | 10 777 | 0.39 | converged |
| … 13 more | | ≤ 9 289 | ≤ 0.98 | converged |

(Work units in this table are the *first-draft* unit costs; §2 corrects them.
The column that matters here is `stopped on`.)

Seven of the twenty-two ILS-firing instances have their search extent decided by
the wall clock, not by the model. That is the gear2 mechanism, in-repo: on those
rows a faster machine keeps descending and a slower one stops earlier, so the
incumbent handed to the tree — and therefore `node_count` — is a function of the
hardware.

15 of 66 instances hit the overall `time_limit`; those are excluded from every
determinism comparison, because there the clock legitimately decides *when to
stop* (the second, unfixable mechanism #912 separates out).

## 2. Unit costs: one sub-NLP solve = 12 000 evaluations

`item912_work_unit_calibration.py`, 5 interleaved rounds per instance (200 evals
and 20 solves per round), load average 0.32 before / 1.06 after:

| instance | vars | cons | eval (µs) | sd | subnlp (ms) | sd | ratio |
|---|---|---|---|---|---|---|---|
| nvs21 | 3 | 2 | 1.1 | 0.1 | 14.71 | 0.95 | 13 327 |
| ex1221 | 5 | 5 | 1.4 | 0.3 | 13.15 | 0.15 | 9 198 |
| ex1222 | 3 | 3 | 1.1 | 0.1 | 10.03 | 0.09 | 9 391 |
| st_e38 | 4 | 3 | 1.1 | 0.1 | 5.42 | 0.26 | 5 047 |
| nvs09 | 10 | 0 | 0.7 | 0.4 | 1.93 | 0.09 | 2 933 |
| ex1224 | 11 | 7 | 1.4 | 0.2 | 15.50 | 0.67 | 10 902 |
| syn05hfsg | 42 | 58 | 2.5 | 0.2 | 80.95 | 3.54 | 32 297 |
| fac2 | 66 | 33 | 2.7 | 0.3 | 39.13 | 1.27 | 14 435 |
| st_e36 | 2 | 2 | 1.3 | 0.1 | 104.37 | 3.13 | 77 964 |

**Geomean ratio 12 364** (min 2 933, max 77 964) → `NLP_SOLVE_UNITS = 12 000`.

This is a *correction of a published number in this document's own first draft*
(CLAUDE.md rule 11): the module initially charged **250** units per sub-NLP
solve, a guess, and it was wrong by ~50×. The consequence was measurable rather
than cosmetic — under the 250-unit charge the observed "units per second" spread
across instances was 26× (1 411/s on st_e36 to 37 392/s on st_e38), because a
subnlp-dominated search was being charged almost nothing for its dominant cost.
No single default budget can behave sanely across a 26× spread.

A size-scaled charge was tried first and **falsified**: multiplying the charge by
`n_vars + n_cons` made the spread *worse* (67.8× vs 26.5×). On this corpus the
per-operation cost is not driven by model dimension — these models are all tiny —
but by how many sub-NLP solves the search issues. Charging solves correctly is
the fix; scaling by size is not.

The residual ~27× ratio spread is the honest accuracy limit of a deterministic
cost proxy: it is a function of the model only, so it cannot know the machine.

## 3. Default budget

See `SolverTuning.ils_work_budget`. Sized from the re-measured consumption in
§4 under two constraints:

1. **Do not bind where the legacy clock did not bind.** An instance whose ILS
   converged on its own must consume the same work and return the same point, so
   the default must exceed the largest naturally-converging consumption.
2. **Keep the wall-time envelope the legacy default had.** The legacy budget was
   capped at 5 s (`min(5.0, 0.15·time_limit)`), so the default must not exceed
   ~5 s of work on the *slowest*-per-unit instance that the gate actually binds
   on.

## 4. Re-measured consumption at the corrected unit cost (superseded — see §4b)

Same command, same corpus, `NLP_SOLVE_UNITS = 12 000`, legacy arm,
`time_limit=15` (2.25 s wall budget), 22 ILS-firing instances:

| instance | nodes | ILS work | ILS wall (s) | stopped on | units/s |
|---|---|---|---|---|---|
| nvs09 | 5 | 2 630 800 | 2.25 | **deadline** | 1 168 206 |
| ex1224 | 5 | 1 680 548 | 2.26 | **deadline** | 744 594 |
| st_e29 | 5 | 1 668 548 | 2.26 | **deadline** | 738 622 |
| ex1225 | 5 | 1 452 291 | 2.11 | converged | 687 312 |
| tspn05 | 33 | 780 266 | 2.28 | **deadline** | 341 922 |
| gkocis | 5 | 540 039 | 0.61 | converged | 883 861 |
| ex1221 | 5 | 516 027 | 0.38 | converged | 1 343 820 |
| oaer | 3 | 444 039 | 0.61 | converged | 732 738 |
| ex1226 | 3 | 432 051 | 0.33 | converged | 1 305 290 |
| syn05hfsg | 185 | 408 029 | 2.31 | **deadline** | 176 331 |
| nvs06 | 5 | 372 090 | 0.40 | converged | 932 556 |
| st_e38 | 3 | 372 065 | 0.23 | converged | 1 610 671 |
| nvs01 | 3 | 324 138 | 0.38 | converged | 855 245 |
| nvs08 | 3 | 324 078 | 0.23 | converged | 1 433 973 |
| nvs21 | 3 | 276 106 | 0.39 | converged | 709 784 |
| nvs05 | 25 | 228 122 | 0.97 | converged | 233 971 |
| fac2 | 39 | 216 087 | 1.03 | **deadline** | 210 201 |
| ex1222 | 1 | 84 004 | 0.07 | converged | 1 235 353 |
| st_e36 | 85 | 48 017 | 0.72 | converged | 66 230 |
| ex14_1_9, nvs04, st_e11 | | 0 | 0.00 | no-op | — |

### The bistable row is itself the finding

`ex1225` and `tspn05`/`syn05hfsg` are worth reading twice. In the §1 sweep
`ex1225` **stopped on the deadline** at 2.27 s; in this sweep, same machine, same
arm, same `time_limit`, it **converged** at 2.11 s and 1 452 291 units — and
`tspn05` returned 35 nodes in one sweep and 33 in the other, `syn05hfsg` 181 vs
185. Nothing about the input changed between the two runs. That is #912's claim
reproduced accidentally, twice, inside its own calibration: with a wall-clock
extent gate the tree is not a function of the model.

It also makes a limit explicit. On a bistable instance there is no work budget
that "preserves current behaviour", because there is no single current
behaviour to preserve.

### Sizing the default

* Largest consumption by a search that converged on its own, excluding the
  bistable `ex1225`: **540 039** (gkocis). A default above this cannot cut short
  any search the legacy gate let finish.
* Slowest instance the gate actually binds on: **176 331 units/s** (syn05hfsg).
  The legacy budget was capped at 5 s, so staying inside that envelope means
  **≤ 880 000** units.

`_ILS_WORK_BUDGET_DEFAULT = 750_000` is inside `[540 039, 880 000]`. Projected
effect on the seven instances the clock used to cut (legacy wall → work-budget
wall at the measured rate): nvs09 2.25 → 0.6 s, ex1224 2.26 → 1.0 s, st_e29
2.26 → 1.0 s, ex1225 2.11 → 1.1 s, tspn05 2.28 → 2.2 s, syn05hfsg 2.31 → 4.3 s,
fac2 1.03 → 3.6 s. Roughly neutral in total, and every one of them now stops at
a point that is a function of the model.

### What the units do *not* buy

The per-unit cost still varies 24x across instances (66 230 units/s on st_e36 to
1 610 671 on st_e38), so a fixed unit budget is not a fixed wall time. It cannot
be: a deterministic budget is by definition blind to the machine. What it buys is
that the *same* model always gets the *same* search — which is the property the
repo's bound-neutral regime assumes and #912 showed it did not have.

## 4b. The single-currency design was falsified; two counters replaced it

§4 sized one budget in converted units. Two measurements killed that design.

1. **The starvation it caused was real.** At the 750 000-unit default, `nvs09`
   went from 5 nodes to 29 and from `optimal` to `feasible`: its search is cheap
   per operation, so the shared currency ran out long before the search was done,
   while the same number gave the expensive-per-operation `syn05hfsg` three times
   its legacy wall time. One number cannot price operations whose cost ratio
   varies 27x.
2. **The split, once measured, showed why.** Instrumenting the two kinds
   separately (legacy arm, `time_limit=60`, the full 5 s legacy budget) shows the
   sub-NLP count is the real currency and the evaluation count is nearly free:

| instance | nodes | evals | sub-NLP solves | ILS wall (s) | stopped on | solves/s |
|---|---|---|---|---|---|---|
| nvs09 | 5 | 4 752 | 443 | 5.02 | **deadline** | 88.3 |
| ex1224 | 5 | 796 | 217 | 3.45 | converged | 62.9 |
| st_e29 | 5 | 796 | 217 | 3.62 | converged | 60.0 |
| tspn05 | 51 | 460 | 152 | 5.02 | **deadline** | 30.3 |
| ex1225 | 5 | 291 | 121 | 2.20 | converged | 54.9 |
| syn05hfsg | 185 | 47 | 67 | 5.02 | **deadline** | 13.3 |
| gkocis | 5 | 39 | 45 | 0.56 | converged | 80.5 |
| ex1221 | 5 | 27 | 43 | 0.41 | converged | 105.4 |
| oaer | 3 | 39 | 37 | 0.59 | converged | 62.3 |
| ex1226 | 3 | 51 | 36 | 0.35 | converged | 102.6 |
| nvs06 | 5 | 90 | 31 | 0.43 | converged | 71.8 |
| st_e38 | 3 | 65 | 31 | 0.23 | converged | 136.0 |
| nvs01 | 3 | 138 | 27 | 0.31 | converged | 86.8 |
| nvs08 | 3 | 78 | 27 | 0.25 | converged | 109.8 |
| fac2 | 39 | 94 | 23 | 1.30 | **deadline** | 17.7 |
| nvs21 | 3 | 106 | 23 | 0.41 | converged | 55.6 |
| nvs05 | 155 | 122 | 19 | 0.89 | converged | 21.4 |
| ex1222 | 1 | 4 | 7 | 0.07 | converged | 97.2 |
| st_e36 | 85 | 17 | 4 | 0.70 | converged | 5.7 |
| ex14_1_9, nvs04, st_e11 | | 0 | 0 | 0.00 | no-op | — |

Note that `nvs09` is **solve**-dominated too (443 solves) — the "evaluation-
dominated" reading in the first draft was inferred from converted unit totals
without the split, and the split refutes it. Retracted here per CLAUDE.md rule
11; the conclusion (one currency is wrong) survives, the reason changes: it is
not that one search is eval-heavy and another solve-heavy, it is that the *cost
of a solve* varies 5x across instances (13.3/s to 136/s) while the currency
assumed it constant.

**Sizing the two caps.** The constraints are in direct conflict and the conflict
is the bug itself:

* never cut a search the clock let finish ⇒ ≥ **217 solves**, ≥ **796 evals**;
* never exceed the legacy 5 s envelope ⇒ ≤ **67 solves** on the slowest binding
  instance (syn05hfsg, 13.3 solves/s).

The old gate handed ex1224 217 solves and syn05hfsg 67 in the *same five
seconds*. Any deterministic number lands between; `ils_solve_budget = 128` is
roughly the middle (59 % of the largest natural extent; ≈9.6 s on the slowest).
`ils_eval_budget = 20 000` is 25x the largest natural consumption — uncontested,
it only stops a genuinely runaway descent. §7 is the A/B that validated both.

## 5. Determinism panel — tight limit (`time_limit=15`, alpha 1/2/4)

22 ILS-firing instances, new arm, clock scaled 1x / 2x / 4x (a 2x-scaled clock is
a 2x-slower machine). Raw verdict: **41 comparisons, 9 mismatches.**

The 9 mismatches are not scattered — they are exactly six instances, and every
one of them spends ≥ 11 s of its 15 s budget at alpha=1:

| instance | real s @ alpha=1 | alpha=1 | alpha=2 | alpha=4 |
|---|---|---|---|---|
| fac2 | 16.1 | 39 nodes | 27 | (time_limit) |
| nvs05 | 16.0 | 25 | 7 | (time_limit) |
| nvs09 | 17.1 | 29 | 25 | 7 |
| st_e36 | 11.0 | 85 | 67 | (time_limit) |
| syn05hfsg | 16.0 | 155 | 23 | 3 |
| tspn05 | 16.3 | 33 | 13 | 3 |

Scaling the clock 2x on a run that needs 16 s of a 15 s budget halves its real
budget to 7.5 s. The B&B loop's *own* deadline checks then shape the tree — that
is #912's second mechanism (whole-solve budget starvation, the `ex1266` case),
which is the `time_limit` contract and is not what a work budget can or should
fix. Every one of the 16 instances that finishes comfortably is **invariant at
1x / 2x / 4x**, including all five where the new work budget is the binding gate
(ex1224, ex1225, st_e29, tspn05, syn05hfsg all report `stopped_on="work"` with
byte-identical work counts at every alpha).

Rather than leave that attribution as an assertion, the probe now makes it: a
comparison is *in scope* only when neither row hit `time_limit`, consumed ≥ 50 %
of its perceived budget, or had a heuristic cut by the solve deadline. Rows
outside that set are printed with their numbers and with whether they differed —
disclosed, never dropped.

## 6. Determinism panel — generous limit (`time_limit=60`, alpha 1/2)

The headline measurement. 22 ILS-firing instances, work budgets at their
defaults, clock scaled 1x and 2x:

```
out-of-scope comparisons (whole-solve budget starvation): 4
  fac2      alpha=2.0: heuristic cut by the solve deadline — matched anyway
  nvs05     alpha=2.0: deadline-pressured (60.3s of 60s) — and it DIFFERS (155 vs 65 nodes)
  syn05hfsg alpha=2.0: deadline-pressured (44.1s of 60s) — matched anyway
  tspn05    alpha=2.0: deadline-pressured (33.4s of 60s) — and it DIFFERS (51 vs 49 nodes)

=== clock-scale determinism ===
executed comparisons: 18
mismatches:           0
```

**18 in-scope comparisons, 0 mismatches.** Every instance whose solve is not
fighting its own `time_limit` returns the identical tree on a clock running twice
as fast — i.e. on a machine twice as slow.

The four out-of-scope rows are printed with their verdicts rather than dropped,
and two of them *do* differ. That is the point of separating them: those two are
`nvs05` (60.3 s of a 60 s budget) and `tspn05` (33.4 s of 60 s), both squarely in
mechanism 2. Halving their real budget changes their tree, and no work budget
can prevent that — only a larger `time_limit` can.

Compare with §5, the same panel before the scope rule and at a limit tight enough
that six of twenty-two instances were starved: 41 comparisons, 9 mismatches, all
six starved instances.

## 7. Flag ON vs OFF (`time_limit=60`, 22 ILS-firing instances)

`ils_eval_budget=20 000, ils_solve_budget=128` versus the legacy wall gate
(`DISCOPT_ILS_EVAL_BUDGET=0 DISCOPT_ILS_SOLVE_BUDGET=0`), same machine, same
limit, one solve per arm per instance.

**Cert-clean.** 22 instances x 4 reported fields (`status`, `node_count`,
`objective`, `bound`) = **88 field comparisons, 0 differences**. Not "no
regressions" — literally the same trees and the same certificates. The three
instances in this set that carry a reference optimum (nvs04, nvs06, nvs09) have
their bound checked against it on both arms: 6 pairs, 0 bounds above their
optimum, 0 incumbents below it.

**Net effect on cost.** Total wall 203.2 s → 206.4 s (+1.6 %); total root-ILS
wall 30.8 s → 32.5 s (+5.5 %). Redistribution, as designed: the instances the
clock used to cut short get more work (syn05hfsg 5.02 → 9.51 s, fac2 1.30 →
5.33 s), the ones that were burning solves for nothing get less (nvs09 5.02 →
1.58 s, st_e29 3.62 → 2.03 s, ex1224 3.45 → 2.08 s).

The ex1224/st_e29 rows are the informative ones for the sizing argument: both
naturally used 217 solves and both were cut to 128 — and both still return 5
nodes and the same certified optimum. The 89 solves the old gate spent there were
no-ops, which is why the middle of the conflicting range turned out not to cost
anything.

**Determinism.** Five instances moved from clock-decided to work-decided
(`stopped_on = "work:nlp_solve"`: ex1224, nvs09, st_e29, syn05hfsg, tspn05).
`fac2` still reports `deadline` on both arms — it is cut by the *solve/phase*
budget, not by a heuristic's own wall budget, i.e. mechanism 2 again.

## 8. What this does not fix

Stated plainly, because the issue asked for it:

* **Whole-solve budget starvation is untouched and unfixable by this change.**
  When a run is close to `time_limit`, the B&B loop's own deadline checks shape
  the tree, and a slower machine gets a smaller tree. That is the `time_limit`
  contract. The only honest mitigation is the one the panel now applies: refuse
  to make determinism claims about such rows, and say which ones they are.
* **Only the root integer local search is converted.** #912 counts 78 clock
  sites in `python/discopt/` and 57 `Instant::now` in `crates/`. This change
  converts the one the issue names as the decisive mechanism and builds the
  primitive the rest can migrate to. On this corpus, after the conversion, no
  remaining Python-side extent gate moves a tree — but "this corpus does not
  exercise it" is not "it is not there".
* **The Rust gates are out of reach of this instrument.** Clock scaling patches
  Python's `time` module; `Instant::now` in `crates/` is unaffected by it. A row
  that is clock-scale invariant here is invariant *with respect to the Python
  gates* and nothing more.
* **`gear2` itself was never run.** It is not in the in-repo corpus and
  minlplib.org is unreachable from this environment. The mechanism is reproduced
  on seven other instances of the same class; the specific 3-vs-91-node cliff is
  taken from the issue, not re-measured here.

