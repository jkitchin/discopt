# Scenario-decomposed global branch and bound (A8): evaluated, NOT scheduled

**Date:** 2026-09-22
**Status: DECLINED — measurement recorded, implementation not scheduled.** The entry
experiment ran (§5) and the method works as advertised on the class it targets; that
class is not discopt's. Kept as the record so the next reader does not re-derive it.
**Audience:** whoever next considers a decomposition-based global method here.
**Scope:** a new global method for two-stage stochastic nonconvex NLPs, living in
`python/discopt/stochastic/`, reusing `python/discopt/decomposition/` (structure,
parallel) and `python/discopt/_relax/` (relaxation, FBBT/OBBT).
**Relationship to existing docs:**

- `docs/dev/decomposition-remediation-plan.md` §1 registry entry **A8** — "Nonconvex
  models: GBD runs heuristically (`bound=None`). No NGBD-style rigorous alternative."
  This plan is a *second* answer to A8, structurally different from the NGBD spike
  (T3.3, recorded "not run" in that plan's decision log).
- `docs/dev/stochastic-module-plan.md` — the module this lands in. Today's methods
  (`solve_lshaped`, `progressive_hedging`) are convex-exact / heuristic on nonconvex
  models; this is the globally-valid sibling.
- `docs/design/relaxation-catalog.md` — the relaxation layer this method *uses*; it does
  not replace it (see §1.3).

---

## 0. Provenance, and two assumptions that were wrong

The method is Cao & Zavala (2019) — `CaoZavala2019` in `docs/references.bib`,
implemented by the authors as **SNGO** (Julia). **SNoGloDe** (`Stinchfield2025SNoGloDe`)
is the Python/Pyomo successor from the Stinchfield/Laird group, which generalises the
same scheme from two-stage stochastic programs to general block-angular structures; no
public repository or DOI for it was locatable as of 2026-09-21 (it appears as an
optional `import snoglode` inside `sandialabs/sparow`).

This plan was first drafted from the abstract. Reading the full preprint **falsified two
premises of that draft**, both recorded here per CLAUDE.md §4/§11:

1. **"The method needs a small first stage (single digits)." FALSE.** §5.2 of the paper
   solves a parameter-estimation problem with a **48-variable first stage** (temporal
   decomposition: two parameters plus 46 linking states) in 8 min–2 h across 12 real
   datasets, where SCIP exceeds 12 h on 9 of them.
2. **"The non-anticipativity relaxation is the λ=0 Lagrangian bound, so it is weak."**
   Weak *relative to a Lagrangian dual*, yes; but measured against a spatial solver's
   root bound it is not. The paper's root gaps (which are exactly the EVPI) on four
   profiled instances are **36.1 / 14.3 / 1.4 / 9.5 %**, versus SCIP's root gaps of
   **101.5 / 62.6 / ≥10000 / 10.6 %**. The relevant baseline is the monolithic
   relaxation, not the best achievable dual bound.

Both corrections widened the case for *evaluating* it, which is what §4/§5 did. They do
not survive as an argument for building it: see §5's verdict, which turns on scope
rather than on either of these numbers.

---

## 1. The algorithm

### 1.1 Problem and bounds

For `z = min_{x ∈ X0} Σ_s Q_s(x)` with `Q_s(x) = min_{y_s} { f_s(x,y_s) : g_s(x,y_s) ≤ 0 }`
(probability weights folded into `f_s`), at a node with first-stage box `X ⊆ X0`:

- **Lower bound** (paper §2.1) — lift by replicating `x` per scenario and *drop* the
  non-anticipativity constraints `x_s = x_{s+1}`:

      β(X) = Σ_s β_s(X),   β_s(X) = min_{x_s ∈ X, y_s} f_s(x_s, y_s)

  `β(X) ≤ z(X)` because the lifted feasible set contains the original, and `β` is
  monotone: `β(X₁) ≥ β(X₂)` for `X₁ ⊂ X₂`. If any `β_s(X) = ∞` the node is infeasible.
  **Each `β_s` must come from a global solve** — this is the correctness hinge (§3).
- **Upper bound** (paper §2.2) — fix `x̂ ∈ X` and evaluate `α(X) = Σ_s Q_s(x̂)`, again S
  independent global solves.
- **Branching** is on the first-stage variables only; second-stage branching happens
  inside the subproblem solves.

The root gap `α(X0) − β(X0)` is the **expected value of perfect information**. The whole
method is a bet that EVPI is moderate on the target class.

### 1.2 Convergence (paper §3)

Proved by adapting Horst & Tuy (`HorstTuy1996`, already in `references.bib`):

- *Exhaustive subdivision* (branch the longest edge, keep branch points away from the
  bounds) ⇒ "delete by infeasibility" is certain in the limit (Lemma 2) and the lower
  bounding operation is **strongly consistent** (Lemma 3). This half needs only lower
  semicontinuity of `Q_s`, which follows from compactness of `Y_s(x)` (Assumption 2).
- *Upper bound convergence* (Lemma 6) needs **Assumption 3: `Q(·)` Lipschitz continuous
  near `x*`**, which holds if MFCQ holds at the scenario subproblems. This is the one
  constraint qualification in the proof, and it is on the UB half only.
- Bound-improving node selection (always split a node attaining the incumbent lower
  bound) then gives `α_k → z`, `β_k → z` (Theorem 1).

### 1.3 What the reference implementation does *besides* the two bounds (paper §4)

These are not optional extras; they are where SNGO's performance comes from, and four of
the five map onto machinery discopt already has.

| SNGO component | discopt counterpart |
|---|---|
| LP relaxation of the **whole** stochastic problem (convexification + OA, αBB and RLT cuts); node bound = `max(LP bound, β(X))` | `_relax/` relaxation compiler, αBB, RLT — **already ours**; the decomposed bound is a *co-bound*, not a replacement |
| Subproblems warm-started from the LP relaxation solution | `warm_start.py` / `initial_solution=` |
| FBBT + **OBBT at every node** (2·n_x LPs; affordable because subproblem solves dominate) | `tightening.py`, FBBT in `crates/discopt-core` |
| Strong branching on first-stage vars scored by LP-relaxation improvement; fallback to widest range; branch point `Σ x̃_s/|S|`; skip vars with range < 1e-4 | new (small) |
| UB every node for the first 3 levels then every 2 levels | new (trivial) |

Two bound-strengthening tricks worth copying verbatim:

1. **Inherited scenario cut** `f_s(x,y_s) ≥ β_s(X)`, valid for every descendant by the
   monotonicity above. SNGO keeps an auxiliary objective variable per scenario and
   tightens its lower bound down the tree.
2. **Subproblem reuse**: if the parent's minimiser `x̃_s` lies in a child's box, the
   child's `β_s` equals the parent's — no re-solve. Given that **>90 % of runtime is the
   scenario solves** (paper §4), this is the single largest practical win available.

### 1.4 Measured cost model (paper §4, §5.1)

Per-node cost grows **linearly in S**; node count does **not** grow with S, because the
branching dimension is `n_x`. On the PID instance SNGO visited **22,410× fewer nodes**
than SCIP at **1,102× the cost per node**, and won by an order of magnitude in wall time.
The reference implementation is **fully serial**; the authors list parallelisation as
future work and call it "challenging because of memory management and load imbalancing".

---

## 2. What it would have reused (unchanged facts)

- **The structure layer already produces the inputs.** `decomposition/structure.py`'s
  `DecompositionStructure` carries `blocks`, `complicating_vars`, `coupling_constraints`,
  and `restricted_bounds`/`flat_bounds` give the node box.
- **The parallel layer already exists and is what SNGO lacked.**
  `decomposition/parallel/comm.py` maps per-block work with a *fixed reduce order*, so
  `backend="threads"` returns bit-identical bounds. Steps 1, 2, 5, 6 of the paper's cost
  breakdown are all embarrassingly parallel.
- **The `Model.subset` blocker does not apply to the stochastic entry point.**
  `stochastic/scenario.py:8` states it directly: the scenario-creator callback pattern
  means "no expression-graph surgery is needed". A standalone per-scenario `Model` is
  built by calling the user's `recourse_builder` against a *fresh* `Model` — which is why
  the general block-angular case (SNoGloDe's generalisation) stays out of scope (§7).
- **The one API gap** is that today's builder signature closes over the outer model's
  first-stage variables (`python/tests/test_stochastic_phase1.py:32-39` — the inner `recourse`
  closure captures `q`, a variable of the *outer* `m`), so it cannot be retargeted. See T1.1.

---

## 3. Correctness contract (binding on any revival)

1. **`β_s` is a dual bound, never an incumbent.** The node lower bound sums
   `SolveResult.bound` over scenarios, and only when `bound_valid` is true. Summing
   `objective` (or a local NLP value) yields a "lower bound" *above* the true optimum,
   which this solver would then certify — the exact false-`optimal` failure CLAUDE.md §1
   forbids. If any scenario returns no valid bound, the node bound is `-inf`.
2. **A scenario solve that hits a limit still contributes its dual bound**, not its gap
   midpoint, and the node must be marked non-certifying if any subproblem is uncertified.
3. **`α` is only an upper bound once the point is feasible.** Fixed-`x̂` scenario
   solutions go through the existing incumbent verification (`verify_incumbent`) before
   updating the global incumbent.
4. **Exhaustiveness is a correctness property, not a heuristic.** The branch point must
   keep a minimum distance from the box bounds, and the "skip variables with range
   < 1e-4" rule must not be allowed to stall subdivision on a variable that still carries
   gap — if every candidate is below the threshold the node is closed as ε-optimal, not
   silently dropped.
5. **Assumption 3 (MFCQ / Lipschitz `Q`) is an assumption of the UB half of the
   convergence proof.** It cannot be checked cheaply. It does not threaten validity of
   any bound we report — an unverified `α` is still a real feasible point's value — only
   the finite-termination claim. Say so in the docstring rather than asserting
   convergence unconditionally.

---

## 4. Entry experiment (as run)

Per CLAUDE.md §4, the falsifying experiment runs first.

- **Script:** `discopt_benchmarks/scripts/stochastic_evpi_entry_experiment.py`
- **Hypothesis:** on two-stage stochastic nonconvex NLPs, the decomposed root bound
  `β(X0)` is tighter than discopt's monolithic root bound (`SolveResult.root_bound`), and
  the advantage does not decay as S grows.
- **Measurement:** per (family, S) cell — `LB_dec = Σ_s β_s` from S global subproblem
  solves; `α` from the mean-candidate fixed-`x̂` re-solve; discopt's `root_bound` on the
  extensive form, replicated (the baseline is given its *best* root bound across
  replicates, so the comparison is generous to it); both gaps taken against the best
  incumbent either arm found.
- **Families:** PID controller tuning (paper §5.1, `n_x = 3`), temporal-decomposition
  parameter estimation (paper §5.2, `n_x = 2 + (S−1)`), and a two-stage pooling/blending
  problem with a bilinear pool-quality equality (`n_x = 2`) — the nonconvexity class the
  in-repo corpus is actually full of.
- **Known limitation, stated up front:** there are **no two-stage stochastic instances in
  the in-repo corpus** (`python/tests/data/minlplib_nl/`) or in MINLPLib, so this
  experiment cannot be run on real corpus instances the way CLAUDE.md's working-on-an-
  issue §2 prefers. The families are the reference paper's own problem classes plus one
  drawn from our corpus's dominant nonconvexity. The residual risk — that the result does
  not transfer to the instances a user would bring — is real and is the reason the
  implementation phases below stay staged behind further measurement.
- **Kill criterion:** if `decomposed_root_gap ≥ discopt_root_gap` in a majority of cells,
  the method has no bound headroom here and the implementation is **not** scheduled.

Run it with:

```bash
python -u -m discopt_benchmarks.scripts.stochastic_evpi_entry_experiment \
    --families pid,estimation,pooling --scenarios 4,8,16 \
    --time-limit 120 --sub-time-limit 30 --replicates 3 \
    --out reports/stochastic_evpi_entry.json
```

---

## 5. Results (2026-09-22)

Run: `reports/stochastic_evpi_entry.json` (12 cells, 3961 s, 2 monolith replicates) plus
`reports/stochastic_evpi_entry_rerun.json` (the 4 cells whose upper bound the first sweep
suppressed, re-run with the fixed candidate rule at 1 replicate). **`reports/` is
gitignored, so those files are not in-tree — the table below is the durable record**, and
`discopt_benchmarks/scripts/stochastic_evpi_entry_experiment.py` regenerates it
(`--summarize` re-renders the table from a saved report). Budgets: 120 s monolith,
40 s per subproblem, `deterministic=True` throughout. `discopt.__file__` asserted to the
worktree; load average 0.9–1.0 before and after, no competing job.

| family | S | n_x | decomposed LB | discopt root bound | better bound | decomposed root gap | discopt root gap | scenario disagreement | subs certified |
|---|---|---|---|---|---|---|---|---|---|
| pooling | 4 | 2 | −28.0115 | −27.5143 | monolith | 1.81% | **0.0000%** | 0.255 | 4/4 |
| pooling | 8 | 2 | −26.6315 | −25.7894 | monolith | 3.27% | **0.0000%** | 0.406 | 8/8 |
| pooling | 16 | 2 | −27.7079 | −27.1349 | monolith | 2.11% | **0.0000%** | 0.462 | 16/16 |
| pid | 4 | 3 | 0.080456 | 0.025008 | **decomp** | **0.0006%** | 68.92% | 0.000 | 4/4 |
| pid | 8 | 3 | 0.061990 | 0.016807 | **decomp** | **7.75%** | 74.99% | 0.000 | 6/8 |
| pid | 16 | 3 | 0.065083 | **−1e−12** | **decomp** | **3.85%** | 100.00% | 0.000 | 13/16 |
| pid_wide | 4 | 3 | 0.401738 | 0.163690 | **decomp** | **0.0002%** | 59.25% | 0.000 | 4/4 |
| pid_wide | 8 | 3 | 0.248568 | 0.104770 | **decomp** | n/a (x̂ infeasible in 1 scenario) | n/a | 0.000 | 7/8 |
| pid_wide | 16 | 3 | 0.250199 | **−1e−12** | **decomp** | **0.066%** | 100.00% | 0.000 | 15/16 |
| estimation | 4 | 5 | −0.000193 | 0.0000382 | monolith | 104.32% | 99.14% | 0.828 | 2/4 |
| estimation | 8 | 9 | −0.002534 | 0.000973 | monolith | 147.98% | 81.58% | n/a | 1/8 |
| estimation | 16 | 17 | −0.501343 | **none** | (degenerate) | n/a | n/a | n/a | 1/16 |

**Tally.** Bound-level: decomposed better in 7 of 12 cells. Gap-level (the pre-registered
metric): 5 wins of 10 comparable cells. The split is perfectly clean by family — all
decomposed wins are PID, all monolith wins are pooling or estimation.

### 5.1 What the numbers say

- **Where the first stage enters the recourse nonlinearly and the scenarios agree (PID,
  disagreement 0.000), the decomposition is not a speedup — it is the difference between
  an answer and none.** At S=16 discopt on the extensive form returns the *trivial* bound
  `−1e−12` with no feasible point after 120 s; the decomposed arm returns 0.065 and an
  incumbent, gap 3.85 % against the monolith's 100 %.
- **Where the first stage enters the recourse only linearly (pooling), our relaxation is
  already exact at the root** (`nodes=1`, gap 0.0000 %, bit-identical across replicates)
  and the decomposition is pure overhead, losing by 1.8–3.3 points.
- **Where blocks can satisfy themselves independently (estimation, disagreement 0.828),
  EVPI is enormous and grows with S** (104 % → 148 %): each time block fits its own data
  almost perfectly once its linking states are free, so the root bound is worthless.
- **discopt's root bound was bit-identical across replicates in every cell** (sd = 0)
  under `deterministic=True`, so `solver.py`'s reproducibility retraction did not bite
  here.

### 5.2 Caveats that cut *for* the method, stated so the verdict is not read as stronger than it is

- **Uncertified subproblems make the measured decomposed bound an underestimate of the
  method's true bound.** Only 2/4, 1/8 and 1/16 estimation subproblems certified. An
  uncertified subproblem still contributes a *valid* dual bound (nothing here is unsound),
  but one below its true `β_s`. The estimation losses are therefore pessimistic-for-the-
  method, not clean losses.
- **The upper bound used here is the paper's *fallback*, not its primary.** Cao & Zavala
  take x̂ from a local NLP solve of the extensive form and only fall back to the mean of
  the subproblem minimisers. Only the fallback was implemented, which is why `pid_wide`
  S=8 has no α at all.
- **No real-corpus instances exist for this structure.** Neither MINLPLib nor
  `python/tests/data/minlplib_nl/` contains a two-stage stochastic instance, so all three
  families are constructed. This is a named residual risk, not corpus validation.
- **Two bugs in the probe were found and fixed mid-flight** (a positive-feedback sign
  error that made every fixed-gain PID scenario infeasible; an all-S-candidates
  restriction that suppressed the upper bound on 4 cells). Both would have produced
  confident, wrong conclusions — the first a false "PID unmeasurable", the second a false
  "no headroom at S≥8".

### 5.3 Verdict: DECLINED

The method works, on the class it targets. **That class is not discopt's**, and that —
not the bound numbers — is what decides this:

1. **It does not solve MINLPs.** Cao & Zavala assume a **continuous** first stage; an
   integer first stage is their own stated future work, because branching on integers
   interacts with the exhaustive-subdivision argument carrying the convergence proof
   (§1.2). discopt is an MINLP solver. A stochastic-NLP global solver is a *different
   product*, not a capability increment.
2. **The one structure already in this repo that fits, measured worst.**
   `discopt.dae.fit.fit_trajectories` builds one collocation block per trajectory on a
   shared model with shared parameters — first stage plus blocks, exactly this shape. It
   is the `estimation` family above, and its root bound lost at S=4 and S=8 and degraded
   as S grew. The in-repo use case is the anti-case.
3. **Nothing in the corpus exercises it.** There is no two-stage stochastic instance in
   MINLPLib or the in-repo corpus, so the method could never be regression-tested by the
   panels that gate everything else here (CLAUDE.md §5).
4. **The win is confined to low-EVPI instances**, and EVPI is a property of the *user's*
   model that we cannot assume. Both PID families measured disagreement 0.000 — even the
   variant built specifically to be heterogeneous.

**For the MINLP-facing version of A8, the other candidate is the right one.** NGBD
(`LiTomasgardBarton2011`, the unrun T3.3 spike) has finite termination *with an
all-integer first stage* — it targets the class discopt is actually for. Reading the full
Cao & Zavala paper changed which A8 candidate looks more on-mission, in the opposite
direction from what this plan was opened to argue.

### 5.4 What would change this

Any one of these, and this doc becomes a live plan again:

- A user or collaborator brings **two-stage stochastic nonconvex NLPs** as a real workload
  (Laird's group is the obvious source, given SNoGloDe).
- The integer-first-stage extension appears in the literature with a convergence proof, at
  which point the method becomes MINLP-relevant rather than adjacent.
- `discopt.dae.fit` acquires users whose fits are **globally** unsolvable today *and* whose
  block disagreement is small — i.e. the estimation family's numbers invert on real data.

Until then: not scheduled. Phases retained in §6 as a sketch, explicitly not a queue.

---

## 6. If revisited: implementation sketch (NOT a work queue)

Kept because it is cheap to keep and expensive to re-derive. Nothing here is scheduled.

The staging that made sense, in order, was: (0) extend the recourse-builder contract to
`(model, data, s, first_stage)` plus a `first_stage_builder`, so `build_extensive_form`
and a new `build_scenario_model(s)` construct the same first stage in different `Model`s
— the only API gap, since the callback pattern already avoids expression-DAG surgery;
(1) `β(X0)`, `α(X0)` and EVPI through `decomposition/parallel/comm.py`, which is the
whole method's value on its own and its own go/no-go signal; (2) a Python-level
first-stage-only spatial B&B (per-node cost is S global solves, so loop overhead is
noise), then the inherited scenario cut and subproblem reuse as a **bound-neutral** change
(node count and certified objective exactly unchanged), then `max(relaxation, β(X))` as a
co-bound; (3) OBBT at every node, strong branching, the UB schedule, the thread backend;
(4) a `solve_global` surface plus a `MethodKind` member.

The correctness contract in §3 is the part that must survive any revival: **sum subproblem
dual bounds, never incumbents.**

## 7. Out of scope (and why)

- **Integer first-stage variables.** The paper's own future work; branching on integers
  interacts with the exhaustiveness argument and needs its own treatment. `solve_lshaped`
  and `Laporte1993`'s integer L-shaped cuts remain the route for those.
- **Risk measures.** CVaR couples scenarios through the objective and destroys the
  decomposition — the same reason `stochastic/lshaped.py` ships risk-neutral first.
- **General block-angular structures (the SNoGloDe generalisation).** Requires extracting
  a standalone sub-`Model` from a monolith, i.e. the `Model.subset(vars, constraints)`
  primitive whose absence deferred T1.4 in the decomposition remediation plan. The
  stochastic entry point sidesteps it via the builder callbacks; the general case does
  not. Revisit once that primitive exists.
- **MPI.** The thread backend is the shipping target; `comm.py` reserves the slot.

---

## 8. Decision log (append per entry)

- **2026-09-21 — plan drafted from the abstract, then corrected against the full
  preprint.** Two premises falsified; see §0. Neither error would have been caught
  without reading the paper, which is the §4 entry-experiment discipline applied to the
  literature rather than to code.

- **2026-09-22 — DECLINED after the entry experiment.** 12 cells + 4 re-run cells; the
  decomposed root bound beat discopt's in 7 of 12 (all PID), lost in 5 (pooling,
  estimation). The verdict does not turn on the bound numbers: the method assumes a
  continuous first stage (so it is not an MINLP method), nothing in the corpus exercises
  it, and the one in-repo structure that fits (`dae.fit` multi-experiment fitting) is the
  family whose root bound measured worst and degraded with S. Recorded rather than
  scheduled, per CLAUDE.md's rule that a measurement taken must be acted on. NGBD (T3.3)
  is the A8 candidate that targets discopt's actual class.
- **2026-09-22 — the pre-registered metric was a bad choice, replaced and labelled.**
  Normalising both root bounds by an incumbent silenced the probe on 5 of 12 cells,
  including those carrying the largest effect. Two valid lower bounds compare directly.
  The bound-level statistic is post hoc and says so wherever it appears.

---

## 9. References

- Cao, Zavala (2019). *A scalable global optimization algorithm for stochastic nonlinear
  programs.* J. Global Optim. 75(2). `CaoZavala2019`. Implemented as SNGO (Julia).
- Stinchfield, Bhatia, Bynum, Cao, Laird (2025). *SNoGloDe: a structured nonlinear global
  decomposition framework.* `Stinchfield2025SNoGloDe`. Generalises the above to
  block-angular structures; Python/Pyomo.
- Li, Tomasgard, Barton (2011). *Nonconvex generalized Benders decomposition for
  stochastic separable MINLPs.* JOTA 151. `LiTomasgardBarton2011`. The other answer to
  A8 (the T3.3 spike in the decomposition remediation plan).
- Horst, Tuy (1996). *Global optimization: deterministic approaches.* `HorstTuy1996`.
  Supplies the convergence machinery §1.2 adapts.
- Laporte, Louveaux (1993). *The integer L-shaped method…* `Laporte1993`.
