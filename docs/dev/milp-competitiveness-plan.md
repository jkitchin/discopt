# Closing the MILP gap to HiGHS/SCIP — measured plan

Status: entry measurements done 2026-09-05; source review against HiGHS 1.14 and
SCIP 10.0.2 done 2026-09-05. **No stage below has been started.**
Goal: discopt competitive with HiGHS/SCIP on pure MILP.

## 0. The measurement this plan replaces

The prior plan was "raise the root cut budget and clean up slack cuts". It looked
net-positive on a 30-instance synthetic lattice panel (26/30 vs base 25/30).
**On a real corpus it is net-negative and the claim is retracted** (CLAUDE.md §11).

38 MIPLIB 2017 "easy" instances (size-filtered, and every one validated by having
HiGHS reproduce the published optimum from *the converted arrays discopt
receives*), TL = 20 s, arms interleaved per instance, 190/190 runs, no
certification abort:

| arm | solved | wall | med(solved) | nodes | kept/gen |
|---|---|---|---|---|---|
| base_16x1 | **18**/38 | 444.93 s | 0.139 s | 8,312,044 | 0/0 |
| cut200_noprune | 11/38 | 573.62 s | 0.235 s | 1,964,690 | 0/0 |
| cut200_prune | 11/38 | 529.55 s | 0.160 s | 1,327,506 | 2544/5720 |
| cut500_prune | 11/38 | 541.52 s | 0.196 s | 694,144 | 3709/12671 |
| **highs** | **36**/38 | **76.77 s** | 0.802 s | 15,073 | 0/0 |

`DISCOPT_MILP_ROOT_CUTS` therefore **stays default-OFF**. The cleanup
(`root_cut_prune`) is retained only because it is strictly better *within* the
cut path (529 s / 1.33 M nodes vs 573 s / 1.96 M); it does not rescue the budget.

**Both source reviews independently read that row as a lifecycle failure, not a
cut-quality failure**: cuts added under a fixed budget and then carried in every
node LP forever. See §2.1.

### 0.1 Methodology corrections carried into every later panel

- **Matched gap tolerance.** That run gave discopt `gap_tol = 1e-6` (relative —
  `tree_manager.rs:992`) against HiGHS's default `mip_rel_gap = 1e-4`, a 100×
  tighter stopping rule. I first estimated this at "exactly one instance". That
  understated it: at matched 1e-4, `neos-1425699` goes from **640,669 nodes,
  unsolved** to **29 nodes, 0.00 s**. No comparison number may be published at
  mismatched tolerances again.
- **Nodes, not wall, is the diagnostic.** Twelve of the twenty instances discopt
  fails, HiGHS finishes in **one node**.

### 0.2 The sharpest single fact on the panel

HiGHS closes `b-ball`, `sp150x300d`, `beavma`, `nexp-50-20-1-1`,
`nexp-50-20-4-2`, `khb05250`, `gen`, `dcmulti`, `app2-2`, `f2gap40400`, `gr4x6`,
`supportcase14` and `supportcase16` in **one node**. One node means the root LP —
after presolve, root cuts and root heuristics — both *produces* the optimum and
*proves* it. discopt spends 350 k–870 k nodes on several of those same instances.

This was originally written up as "a **root-strength** problem, not a tree-search
problem." I retracted that on 2026-09-05 in favour of a primal/dual/proof split
claiming *"8 of 17 failures already hold a dual bound within ~5 % and still cannot
close."* **That retraction was itself wrong and is withdrawn the same day**
(CLAUDE.md §11). It classified failures by an arbitrary 5 % threshold bearing no
relation to the solver's stopping rule, and was published without checking it
against that rule. Recorded here rather than deleted, because the error mode — a
threshold chosen for readability and then reasoned from as if it were the
convergence test — is worth not repeating.

Re-derived against the actual tolerance (`gap_tol = 1e-4`), over the 17 instances
unsolved at the strongest arm of the post-structural-space panel:

| class | count | meaning |
|---|---|---|
| dual gap already within `gap_tol` | **0** | no pruning / node-selection / cutoff defect exists on this panel |
| PURE-BOUND — incumbent optimal, only bound missing | 3 | `b-ball` 4.5e-4, `neos-3610051-istra` 9.0e-3, `neos-3610040-iskar` 1.0e-2 |
| BOTH SHORT — bound *and* incumbent | 14 | dual 4.6e-3 … 1.0; primal 22 % … no feasible point at all |

**The bound is short on 17 of 17. The incumbent is additionally short on 14 of 17.**

The instance I had held up as the sharpest evidence of a mechanics failure does not
survive contact with the tolerance. `b-ball`'s bound is `-1.500672636` against an
optimum of `-1.5` — a relative dual gap of `4.484e-4` against a `1e-4` stopping
rule. It has not converged. It is short by a factor of ~4.5 in bound and is
behaving exactly as specified. Its 200,839 nodes against HiGHS's 1 measure how
much bound HiGHS gets at the root that discopt never gets anywhere, not a failure
to use a bound it already had.

**Consequence for the stage order.** The original framing was closer to right than
its first correction, but the precise statement is narrower than either: this is a
**bound-strength** problem — root or in-tree — on *every* failing instance, with a
**primal-heuristic** problem layered on 14 of them. Bound work is necessary
everywhere. Primal work is necessary on 14 and sufficient on none, because no
instance closes until the bound arrives regardless of how good the incumbent is.
That is a strict ordering, and the stages below follow it: bound first, primal as
the accelerant that makes the bound's tree cheaper.

## 1. Where the gap is

### 1a. The failures present as primal-short

Decomposing base's 20 failures into primal gap `(inc − opt)/(1+|opt|)` and dual
gap `(opt − bound)/(1+|opt|)`: **15/20 are primal-short**, and two
(`enlight_hard`, `neos-2624317-amur`) never find a feasible point at all. Worst
primal gaps: `neos-911970` 208.5 %, `fiber` 57.7 %, `nexp-50-20-1-1` 56.7 %,
`beavma` 42.2 %, `sp150x300d` 34.3 %. Six have a dual gap ≤ 3 % (`obra` 0.46,
`kaihu` 0.80, `iskar` 1.40, `istra` 2.56, `neos17` 2.56, `fiber` 3.02) — a decent
incumbent alone would close those trees.

§1c argues this is largely a *symptom*: a weak dual bound upstream of the missing
incumbents. It is not, however, dismissed — see Stage 3.

### 1b. The four `mik-250-20-75-*` are a pure dual failure

They already hold the **optimum as incumbent** (primal gap ≈ 0) and still carry a
7–10 % dual gap after ~700 k nodes. HiGHS: 558–1055 nodes, 3.6–5.2 s. `mik` is
the textbook mixed-integer-knapsack (cMIR) family.

discopt's MILP driver separates **cover + Gomory only** (`milp_driver.rs:1065`,
`:1081`; in-tree at `:2300`, cover only), with `cut_select=false` and
`node_cuts=false` as shipped defaults (`lp_bindings.rs:1099`).

### 1c. Presolve, heuristics and symmetry are NOT the difference

Ablation of HiGHS itself over the whole panel, HiGHS alone on the machine,
TL = 20 s, 190 comparisons:

| HiGHS arm | solved | wall | nodes |
|---|---|---|---|
| full | 38/38 | 66.85 s | 15,417 |
| no_presolve | 36/38 | 114.38 s | 36,261 |
| no_heur | 37/38 | 80.12 s | 41,521 |
| no_sym | 38/38 | 71.79 s | 16,368 |
| **bare** (all three off) | **37/38** | **119.93 s** | 54,420 |

Presolve is worth ≤2 instances and ~1.7× wall; symmetry is 0. discopt's own MILP
presolve is dimension-preserving FBBT (`milp_driver.rs:677`) and HiGHS's really
does reduce (`nexp-50-20-1-1` 1030×540 → 728×280) — but the reduction is not what
closes these instances.

**Correction (CLAUDE.md §11).** An earlier draft of this document said HiGHS
"reaches 37/38 with zero heuristic effort". That is wrong.
`mip_heuristic_effort=0` does **not** disable HiGHS's heuristics: feasibility
jump, rounding, shifting and RENS are gated by separate `mip_heuristic_run_*`
booleans (`HighsOptions.h:490-495`, registered **default `true`** at
`:1213-1215`), and `moreHeuristicsAllowed`
(`HighsMipSolverData.cpp:611-673`) grants an unconditional 10,000-iteration
heuristic LP budget even at effort 0 (`:627-629`); incumbents also arrive from
integral node LPs (`HighsSearch.cpp:966-971`) and strong-branch LPs (`:601-606`).
The `bare` arm still had a small primal side. The *direction* survives — cuts +
LP + branching + propagation dominate — but the absolute claim does not.

## 2. What the two source reviews established

Reviews of this plan against `ref/HiGHS/` and `ref/scipoptsuite-10.0.2/`. They
were run independently and **converge on the same reordering**: the plan's target
(cuts) was right, its first action was wrong, and it omitted the two things that
make aggressive root cutting affordable at all.

### 2.1 Cut lifecycle is the missing prerequisite, not a refinement

Neither solver carries root cuts forever, and neither budgets the root by rounds.

- HiGHS: `HighsLpRelaxation::performAging` (`HighsLpRelaxation.cpp:595-642`) ages
  basic cut rows at every node LP and deletes at `mip_lp_age_limit` = 10;
  `removeObsoleteRows` (`:522-539`) drops basic cut rows after the root; the pool
  re-separates by violation with score `viol/(nActiveNzs·sqrt(norm))` and a 0.1
  parallelism cap (`HighsCutPool.cpp:168-364`), pool age limit 30.
- SCIP: `separating/maxroundsroot = -1` (`set.c:471`) — no round budget at all.
  It stops on **stalling**: `maxstallroundsroot` 10 (`:475`), stall test at
  `solve.c:2965-2975` (objreldiff ≤ 1e-4 **and** a fractionality test). Cuts are
  dynamic rows removed after `lp/rowagelimit` = 10 LPs nonbasic (`set.c:263`),
  pool age limit 80. And directly on point for the `root_cut_prune` work on this
  branch: `SCIP_DEFAULT_LP_CLEANUPROWSROOT = TRUE` (`set.c:268`) — SCIP removes
  new **basic** rows after the root LP *by default*, on the same argument;
  `CLEANUPROWS = TRUE` (`:267`) does it after every LP, which is Stage 1's aging
  half. Selection is `cutsel_hybrid` (efficacy 1.0, objparal 0.1,
  intsupport 0.1, minortho root 0.90 — `cutsel_hybrid.c:46-57`).

discopt appends cuts to the matrix permanently (`augment_csc_with_cuts`,
`rebuild_csc_with_cuts`) and budgets by fixed round/cut counts. That is the exact
shape of the §0 regression.

### 2.2 The plan's original Stage 1 would have re-run an experiment already killed

`docs/dev/certification-gap-plan.md` records **CUT-1** and **CUTS-1** as NO-GO,
re-confirmed 2026-07-10 on HEAD `059165fc`: oracle injection of SCIP's *actual*
cMIR cut coefficients closed ≤1.8 % (nvs17) and 0 % (nvs19/24), and
`DISCOPT_CMIR_AGGREGATION` ON/OFF gave **0× node reduction**, bit-identical bound
and node count. `lp/aggregation.rs:1-46` carries the same finding in its header.

**Why this does not settle the MILP case, stated as a precondition rather than
assumed away.** That NO-GO was measured on the MINLP lifted-McCormick class,
where *discopt's default root already closes 99.9 % of the gap* — the separator
was inert because there was nothing left to cut. On `mik` the root leaves 7–10 %.
The premise that made cMIR inert does not hold here. **But that is a hypothesis,
and it is the first thing Stage 2 must test** (§3, Stage 2 entry experiment). If
HiGHS's own root bound on `mik` is also 7–10 % and it wins on nodes instead, then
cuts are not the `mik` lever and Stage 2 is re-scoped to search.

### 2.3 What `separate_mir` actually has, verified in-tree

Not taken on the reviewers' word — read directly:

- deltas = `{1} ∪ {1/|a_j| : j integer}` (`mir.rs:226-232`) — **no** best/2, /4, /8
  refinement (SCIP: `SCIPcutGenerationHeuristicCMIR`, `cuts.c:8339`).
- complementation: two fixed patterns, all-lower and near-upper
  (`mir.rs:236-254`) — **no** per-integer post-hoc flips.
- simple bounds only — **no** VUB/VLB substitution (HiGHS:
  `HighsTransformedLp.cpp:127-330`; SCIP: `VARTYPEUSEVBDS 2`). This is what makes
  cMIR behave like flow cover on fixed-charge structure — i.e. `nexp`,
  `sp150x300d`, `fiber`, `beavma`.
- one cut per row, ranked by raw violation, with no lifted-cover competition
  (HiGHS: `HighsCutGeneration.cpp:1391-1491`).
- **discopt's slack columns are continuous**, so every pure-integer row is
  weakened; SCIP adds integral row slacks to the δ/complementation set.

So "call the existing MIR" is not a small step to HiGHS parity. It is most of
`HighsCutGeneration`.

### 2.4 Divergence between the two reviews, unresolved

HiGHS's reviewer says row-by-row MIR on raw rows is the shape HiGHS deliberately
avoids (separation is on tableau rows or *tight* rows only — `HighsPathSeparator`
types a row with both slacks above feastol as unusable). SCIP's reviewer says
single-row MIR is not intrinsically wrong — SCIP tries the un-aggregated row
first (`sepa_aggregation.c` Step 1 at `naggrs=0`) — and that the real defect is
§2.3's feature list. **Both agree the current `separate_mir` fed raw equality
rows in both orientations would read as "MIR does nothing".** The tight-row
filter is cheap and is adopted; the disagreement does not change Stage 2.

## 2.5 The root-bound attribution — measured 2026-09-05, and it reorders §3

Run before building anything (CLAUDE.md §4). Both arms read the *same* converted
arrays. discopt at the root (`max_nodes` ≤ 3, first finite bound); HiGHS at
`mip_max_nodes=1`. Dual gap as `(opt − bound)/(1+|opt|)`. 14/14 executed, 0
vacuous.

| instance | d_lp | d_root 16×1 | **d_big 200×8** | h_nopre | h_full | own cuts close |
|---|---|---|---|---|---|---|
| mik-250-20-75-1 | 18.99 | 17.29 | **4.63** | 1.94 | 1.84 | 75.6 % |
| mik-250-20-75-2 | 18.16 | 16.03 | **4.81** | 1.64 | 1.52 | 73.5 % |
| mik-250-20-75-3 | 16.13 | 14.03 | **4.53** | 1.68 | 1.76 | 71.9 % |
| mik-250-20-75-5 | 17.46 | 15.86 | **4.47** | 2.12 | 1.72 | 74.4 % |
| fiber | 61.55 | 37.94 | **6.01** | 2.31 | 1.95 | 90.2 % |
| b-ball | 12.73 | 8.24 | **0.99** | 0.01 | −0.00 | 92.2 % |
| sp150x300d | 50.00 | 35.71 | **15.71** | 2.86 | 0.00 | 68.6 % |
| beavma | 58.66 | 52.58 | **31.05** | 1.04 | 0.00 | 47.1 % |
| nexp-50-20-1-1 | 53.33 | 53.33 | **30.00** | −0.00 | −0.00 | 43.8 % |
| neos-3611689-kaihu | 15.10 | 13.78 | **9.31** | 3.75 | 2.50 | 38.3 % |
| neos-3610040-iskar | 8.30 | 7.41 | **6.57** | 4.51 | 5.26 | 20.9 % |
| neos-3610051-istra | 8.89 | 8.73 | **6.55** | 4.11 | 2.00 | 26.3 % |
| neos17 | 12.98 | 12.93 | **10.91** | 4.96 | 1.76 | 16.0 % |
| neos-3118745-obra | 0.46 | 0.46 | **0.29** | 0.46 | 0.39 | 35.7 % |

Three things fall out, and the third overturns the §3 ordering as first drafted.

**(a) HiGHS's root strength is not presolve.** `h_nopre` ≈ `h_full` everywhere —
mik 1.9–2.1 % with presolve off against 1.5–1.8 % with it on, `nexp` 0.00 either
way. Presolve is worth ~0–3 pp of root gap. §1c said ≤2 instances; this says the
root bound specifically owes it almost nothing. **Stage 7 stays last, confirmed.**
(HiGHS exposes no cuts-off switch — the only `mip_*` option records in
`HighsOptions.h` are `allow_restart`, the six `heuristic_run_*` booleans,
`improving_solution_save`, `max_nodes`, `report_level` — so its *cut* share cannot
be ablated this way. An earlier version of this probe labelled
`mip_root_presolve_only` as "cuts off"; it is not a cut switch, and that arm was
deleted rather than reinterpreted.)

**(b) discopt's shipped root budget is drastically undersized.** 16 cuts × 1
round moves mik 18.99 → 17.29 (9 % of the gap) and `nexp-50-20-1-1` 53.33 → 53.33
(nothing at all).

**(c) discopt's EXISTING separators already close 70–92 % of the root gap on the
instances that matter — and this is the finding that reorders the plan.** At
200 × 8, with no new cut family: mik 17.46 → **4.47** (74 %), `fiber` 61.55 →
**6.01** (90 %), `b-ball` 12.73 → **0.99** (92 %), `sp150x300d` 50.00 → **15.71**
(69 %). Cover + Gomory, today, gets mik from 17 % to 4.5 % against HiGHS's 1.7 %.

Set (c) against §0: **the 200-cut arm produced those bounds and still solved 7
fewer instances than base.** discopt is already generating cuts strong enough to
be competitive at the root and then losing the solve by carrying every one of
them in every node LP forever. That is not a cut-quality deficit. It is a cut
*lifecycle* deficit, and §2.1's reviews called it correctly.

**Consequence for §3.** As first drafted, Stage 1 (lifecycle) was framed as a
prerequisite for Stage 2 (cMIR), with cMIR as the large lever — the HiGHS review
put it at 10–15 instances. That is now the wrong weighting and I am recording the
correction here rather than carrying it (CLAUDE.md §4, §11). **Stage 1 is the
lever**; the bound it needs to pay off already exists. Stage 2 is demoted: it is
what takes mik from 4.5 % to HiGHS's 1.7 % *after* Stage 1 makes a large root cut
set affordable, and it is not worth starting until Stage 1 is measured. The
entry experiment this section was written to run — "does HiGHS also leave 7–10 %
at the mik root?" — is answered **no** (it leaves 1.5–1.8 %), so Stage 2 is not
killed; it is deferred behind a bigger, cheaper win.

## 3. Stages

Reordered from the first draft per §2, and re-weighted per §2.5 — Stage 1 is now
the lever and Stage 2 is deferred behind it. Every bound-changing piece goes behind a
default-off flag with the §5 double bar (cert-clean AND net-positive) over the 38
panel *and* the in-repo MINLP corpus. Propagation, aging, restarts and node
selection are contractions or reorderings and need only the bound-neutral or
cert-clean check.

### Stage 0 — flips of things already built (zero code)

Ranked first because the code exists and shipping it off is the whole defect.

1. **`node_propagation=true`** — FBBT per node with children inheriting tightened
   bounds (`milp_driver.rs:2007`), which is HiGHS's `localdom.propagate()`
   (`HighsSearch.cpp:875`) and SCIP's `PROPFREQ 1` (`cons_linear.c:155`).
   Defaults **false** in the PyO3 signature at all three call sites
   (`lp_bindings.rs:1101, :1229, :1388`). Pure contraction ⇒ cert-clean is
   automatic. 

   **MEASURED 2026-09-05, 38 instances, matched `gap_tol=1e-4`, TL = 20 s,
   114/114 runs, zero certification aborts:**

   | arm | solved | wall | med(solved) | nodes |
   |---|---|---|---|---|
   | base | 19/38 | 413.62 s | 0.101 s | 8,969,820 |
   | nodeprop | 19/38 | 395.90 s | 0.081 s | 7,838,764 |
   | highs | 38/38 | 66.35 s | 0.959 s | 15,417 |

   **The HiGHS review predicted "~3-6 instances" from this flag. That is
   falsified: it gains zero.** What it does deliver is −12.6 % nodes, −20 %
   median solve time, and large per-instance wins — `flugpl` 11.0×, `enlight8`
   9.8×, `gt2` 4.8×, `bppc8-02` 1.9× (10.55 s → 2.44 s). On timed-out instances
   the dual bound is better on 8 and worse on 5; the standouts are
   `enlight_hard` 28.95 % → **10.53 %** and `beavma` 28.94 % → 22.33 %, against a
   real regression on `fiber` (2.92 % → 6.28 %). `neos-3610051-istra` newly
   *finds* the optimum (primal 12.00 % → 0.00 %) and ends 1.35 % from proving it.

   Verdict: cert-clean and mildly net-positive, so it graduates ON — but it is
   **not a lever on the gap**, and nothing downstream should be planned as if
   Stage 0 buys instances. Caveat (CLAUDE.md §9): machine load rose from 2.85 to
   13.84 during the run, so the *wall* column is soft; node counts are
   load-independent and carry the conclusion, and arms rotated per instance so
   load hit all three alike. `base` is 19/38 here vs 18/38 in §0 — that one
   instance is the §0.1 tolerance correction, now paid for.

   **The `mik` family is untouched** (1.25–1.30× nodes, still ~600 k against
   HiGHS's 558–1055). Propagation is not the `mik` mechanism; §1b stands.
2. **`cut_select=true`** — `lp/cut_select.rs` exists and ships off
   (`lp_bindings.rs:1099`).
3. **Best-estimate node selection with plunging** — discopt uses
   `SelectionStrategy::BestFirst` (`tree_manager.rs`) and has a best-estimate
   field that is not the driver. SCIP's default is `nodesel_estimate` (verified: `STDPRIORITY 200000` at
   `nodesel_estimate.c:48`, the highest of the five selectors — `bfs` 100000,
   `hybridestim` 50000) with
   plunging; HiGHS alternates best-estimate plunges with a forced best-bound pop
   every 10 leaves (`HighsMipSolver.cpp:504-516`). Improves early incumbents,
   which is §1a.

*Kill criterion:* each flip measured alone on the 38 panel at matched tolerance;
anything that does not improve solved-count or nodes stays off.

### Stage 1 — root cut-loop control — **THE LEVER** (§2.5c)

Replace fixed round/cut budgets with stall-based termination, cut aging, and
efficacy+orthogonality selection; move root cuts into a pool rather than the
matrix.

- Stall rule to replace `root.obj <= prev_obj + 1e-7*(1+|prev_obj|)`, which is
  far tighter than either reference (HiGHS `stall < 3` with a 1.001 test against
  *cumulative* gain; SCIP `objreldiff ≤ 1e-4` plus a fractionality test).
- Delete cut rows that are basic/slack after the root (the `root_cut_prune` work
  already on this branch is the first half of this), then age basic cut rows
  in-tree with a ~10-LP limit, with deleted cuts held in a pool re-checked for
  violation.
- Row deletion must fix up the warm basis (HiGHS falls back to `firstrootbasis`,
  `HighsMipSolver.cpp:635-639`).

*Entry experiment:* re-run the cut200 arm with (a) `cut_select` on, (b) slack-cut
drop, (c) both. *Kill:* if (b) alone does not recover a substantial share of the
7 instances the cut budget lost, aging is not the mechanism and Stage 1 re-scopes.

**RESOLVED — the kill criterion fired, and the mechanism turned out to be
neither aging nor selection.** Both arms are dead:

- **(a) `cut_select` is falsified.** It does not help and on two instances it
  *creates* the pathology. `fiber` at 200×8 goes from 2986.5 to 6457.4
  iterations/node with selection on; `mik-250-20-75-1`, which is perfectly
  healthy at 100 % warm-start acceptance and 6.1 iterations/node, drops to
  **35.0 % acceptance and 515.3 iterations/node**. The textbook
  efficacy+orthogonality fix makes things worse here.
- **(b) slack-cut drop** fired its kill criterion earlier.

The real mechanism is a **column-space defect**, not a lifecycle one.
`separate_gomory_cols` loops `for j in 0..n` over the full working width
(`lp/gomory.rs:215`), so every cut carries coefficients on base slack and
earlier-cut surplus columns. A coefficient on column `ns + r` puts a *second*
nonzero into that column, so it stops being a singleton;
`solve_lp_cols_warm`'s basis reconstruction requires a singleton substitute
(`lp/simplex/primal.rs:2599-2640`) and otherwise returns a **short** basis;
`PreparedDual::prepare` refuses a short basis on shape
(`lp/simplex/dual.rs:521,532`, `DualPrepRejectShape`) and the node cold-solves;
the cold solve returns another short basis, so the state is self-sustaining. A
short basis *also* silently disables GMI separation outright
(`lp/gomory.rs:236`).

Measured directly: `fiber` at 50×2 took **729 shape rejections against 2
acceptances** (0.3 % warm-start acceptance), while the same instance's base
16×1 arm ran at 100 %. That is the 492× per-node cost, and it explains why the
cut arms produce a strong bound and then lose solves: they are paying two to
three orders of magnitude more simplex iterations per node for it.

`root_cut_prune` is a trigger rather than a cause — its dependency-closure sweep
keeps a cut precisely *because* something references its surplus column, so it
preferentially retains the cross-referencing cuts and discards the clean
singleton ones. On `sp150x300d` removing the prune reaches the **identical**
bound (60) at 22.3 instead of 2079.6 iterations/node.

**Fix (implemented):** rewrite every cut into structural space before it is
appended, HiGHS-style. HiGHS never has this problem because
`HighsTransformedLp::transform` expands slack coefficients back into structural
ones *before* generating the cut (`HighsTransformedLp.cpp:447,469-482`). The
same rewrite here uses the exact equality identities the LP rows already supply
(`s_r = (b_r - A_r·x)/alpha` for a base row, `s_i = c_i·x - rhs_i` for a cut
row), so it is an identity, not a relaxation. Two follow-on steps are genuine
weakenings and are bounded accordingly: negligible coefficients left by
cancellation are moved to the right-hand side at their maximum over the box
(HiGHS `HighsCutGeneration.cpp:783`), and a cut whose nonzero count *grows* past
`100 + 0.15·n` is shortened the same way rather than dropped (`:982-1012`).

*Measured result* (6 instances × 4 arms, 400 nodes, `RootCutsSubstDropped` and
the five per-reason counters armed):

| | before | after |
|---|---|---|
| warm-start acceptance | 0.3–1.1 % on the cut arms | **100 % on 22 of 24 arms**, 94.8 % worst |
| `DualPrepRejectShape` | 588–822 per arm | **0** on 23 of 24 arms |
| `fiber` 200×8 iterations/node | 2986.5 | **99.5** |
| `sp150x300d` 200×8 iterations/node | 2079.6 | **10.7** |
| `mik` 200×8 `sel` iterations/node | 515.3 | **5.6** |
| `fiber` 200×8 root bound | 388985 | 381984 |
| `mik` 200×8 root bound | −51927.6 | −51972.4 |
| `blend2` 200×8 root bound | 7.14624 | **7.16601** |

Cut quality is preserved (within 0.02–0.1 % on every instance, better on
`blend2`) while the per-node cost falls 30–240×. A first attempt used a
substitution-density guard instead, which restored warm start but dropped so
many cuts that root bounds collapsed (`mik` to −58655.4, zero cuts kept); that
is recorded here as falsified, and the shortening-not-dropping design replaced
it. A plain dynamism gate failed the same way — 217 of 217 cuts on `fiber` and
75 of 75 on `mik` refused for coefficient range alone — which is why the
negligible terms are removed rather than the cut rejected. HiGHS does not gate
on dynamism at all: its only `dynamism` line sits inside an `#if 0` debug block
(`HighsCutGeneration.cpp:1061`).

Stage 1's remaining items (stall-based termination, in-tree aging, a violation
re-checked pool) stand, but they are now optimizations on a working cut path
rather than the fix for it.

*Why this is now the lever, not a prerequisite.* §2.5c measured discopt's own
existing separators closing **70–92 %** of the root gap at 200 × 8 on exactly the
instances that fail — mik 17.46 → 4.47, `fiber` 61.55 → 6.01, `b-ball` 12.73 →
0.99 — while §0 measured that same 200-cut arm solving **7 fewer** instances. The
bound Stage 2 was going to be built to produce is already being produced and then
thrown away by carrying it in every node LP. Stage 1 is the difference between
having that bound and being able to use it.

*Target:* recover base's 19 and convert the cut arm's root strength into solves.
The mik family is the cleanest probe — a 4.5 % root gap should not need 600 k
nodes.

### Stage 2 — cMIR done properly (DEFERRED behind Stage 1 per §2.5)

*The attribution this stage was gated on has been run* (§2.5). HiGHS's mik root
gap is **1.5–1.8 %**, not 7–10 %, so §2.2's precondition holds and cMIR is not
killed. But §2.5c also showed discopt's existing separators reach 4.5 % on mik
unaided, so cMIR is worth roughly the 4.5 % → 1.7 % remainder — real, but second
to Stage 1, and **not to be started until Stage 1 is measured**. Starting it
first would build a stronger cut into the same lifecycle that is currently
converting strong cuts into lost solves.

When it is started, build in fidelity order, measuring after each: (1) integral row
slacks treated as integer in the δ/complementation set; (2) separate on the
current LP *including* previously added cuts, scored by efficacy `viol/‖α‖` not
raw violation; (3) VUB/VLB substitution; (4) the full δ set incl. best/2,/4,/8
and per-integer complementation flips; (5) tight-row filtering; (6) aggregation
depth ≥3.

*Kill:* root gap on `mik` fails to close ≥3 pp after (1)–(4).

### Stage 3 — primal

Reordered from FJ-first. SCIP has no feasibility-jump at all and still gets first
incumbents; RENS is the cheapest for us because it reuses `milp_driver.rs`
recursively.

1. RENS on the root LP. 2. Propagation-backed rounding (needs Stage 0.1 anyway).
3. Rounding/shifting after *every* root separation round — HiGHS's
`rootSeparationRound` (`HighsMipSolverData.cpp:1783-1790`) is what produces the
one-node solves. 4. Feasibility jump **only if** `enlight_hard` and
`neos-2624317-amur` still lack an incumbent.

### Stage 4 — restarts

Root restart when reduced-cost fixing + propagation has fixed ≥5 % of integers
(SCIP `restartfac` 0.025 / `immrestartfac` 0.05, `solve.c:4601-4603`). discopt
already has `reduced_cost_fix` (`:2766`). Cert-neutral contraction. Also: root
reduced-cost "lurking bounds" (`HighsRedcostFixing.cpp:36-71, :194+`) fire free
tightenings every time the incumbent improves.

### Stage 5 — conflict analysis; Stage 6 — clique/implication tables

Both default-ON in SCIP (`set.c:108-172`). Conflict analysis needs a per-node
bound-change log — a real build. Clique extraction feeds propagation, knapsack
separation and heuristics; `presolve/cliques.rs` and `implied_bounds.rs` exist
and are unwired on the MILP path.

### Stage 7 — dimension-reducing presolve with postsolve

Last, per §1c (≤2 instances, large build — the matrix path has no postsolve
chain). Note restarts (Stage 4) want re-presolve, so there is a coupling.

### Falsified: relaxing the cut-cleanup's numerical gates (2026-09-05)

*Hypothesis.* The substitution's two numerical gates are throwing away usable
cuts. Evidence prompting it: on `gsvm2rl3` the root separator emitted 31 cuts and
kept **zero**, leaving an 85 % root gap, and HiGHS has no dynamism gate at all —
its only `dynamism` line (`HighsCutGeneration.cpp:1061`) sits inside an `#if 0`.

*Entry experiment.* Two gates, tested separately and then together, over all 38
panel instances with a false-bound assertion on every run.

1. **Relaxing the dynamism cap alone** (1e7 → 1e300): 7 instances better, 2
   worse on root gap. `gsvm2rl3` **unchanged** — its cuts never reach the cap.
2. **Keeping an unremovable small coefficient** instead of refusing the cut
   (exact, no weakening): **0 of 38 changed.** Its cuts are then killed by the
   dynamism cap instead.
3. **Both together**: `gsvm2rl3` goes 0 → 99 cuts and 85.16 % → 76.11 % root gap;
   7 better / 2 worse overall. At an intermediate cap of 1e12 the result is 6
   better / 2 worse and `gsvm2rl3` stays at zero cuts — its coefficient range
   exceeds 1e12, i.e. those cuts are genuinely ill-conditioned.

*Kill criterion and result.* The bar is CLAUDE.md §5's second half: net-positive,
not merely sound. Measured at a **fixed 5000-node budget** (deterministic, so the
loaded machine could not contaminate it — load average was ~68 throughout, which
invalidates any wall-clock or time-limited solved-count):

| arm | solved | total nodes | dual gap better | worse |
|---|---|---|---|---|
| base (1e7, refuse) | 14/38 | 133,680 | — | — |
| 1e12 + keep | 14/38 | 132,724 | 3 | 3 |

**Cert-clean but neutral, so it does not ship** — the `DISCOPT_CUT_INHERIT` rule.
The root-bound gain does not survive into tree progress. Not flagged, not
default-off-with-a-knob: reverted, with the measurement recorded in a comment at
the refusal site so the next person does not re-run it.

*What it did leave behind.* One counter covered three distinct failures
(unbounded-bound refusal, non-finite coefficient, ill-conditioned ratio), and I
mis-diagnosed `gsvm2rl3` three times in a row off it — first as "dynamism", then
as "the unbounded branch", before measurement showed it is the ratio test acting
on cuts the unbounded branch let through. The counter is now split into
`SubstDropUnbounded`, `SubstDropNonFinite` and `SubstDropDynamism`. That split is
the shippable part of this experiment.

*Carried forward.* `gsvm2rl3` (42 % dual gap) and `neos-2624317-amur` (zero cuts
generated at all, 100 % dual gap, no feasible point) are not cut-*gating*
failures. They need a cut family whose coefficients are conditioned by
construction, which is Stage 2's business, not a threshold change.

### Explicitly dropped

Symmetry on this panel (0 instances, both reviews agree); GUB and superadditive
cover lifting (OFF by default in SCIP — `cons_knapsack.c:127, :146`); more raw
Gomory budget (measured negative, §0); CSC-native MIR before a dense root-gap
gain is shown; oddcycle/cgmip/closecuts/lagromory (freq −1 in SCIP); local
branching/DINS as first-incumbent tools.

Sequentially-lifted knapsack cover (upgrading `lp/cover.rs` from unlifted) is
*not* dropped but is low priority: ~0.5–1 instance, low risk, and irrelevant to
`mik` (cover fires only on binary rows).

## 4. Success metric

The §0 panel at matched `mip_rel_gap = 1e-4`, TL = 20 s. Base is **18/38 at
444.93 s** (19/38 once the tolerance is matched); HiGHS **36/38 at 76.77 s**
interleaved, 38/38 at 66.85 s alone. "Competitive" is solved-count within a few
instances of HiGHS's with total wall within ~2×. Every stage reports against this
same table.

## 5. Licensing note for the owner

discopt is **EPL-2.0**. HiGHS is MIT (compatible inbound). SCIP 10 is Apache-2.0,
which can be combined but not relicensed. Both reviews were instructed to
describe algorithms, not paste source, and neither returned code. If the cMIR δ
and complementation loop in Stage 2 ends up a close derivation of
`SCIPcutGenerationHeuristicCMIR`, or the transform step of
`HighsTransformedLp::transform`, that is an attribution decision for the owner
before merge, not something to settle in a PR.

## 6. Reproduction

`scratchpad/miplib/` holds `loader.py` (MPS → engine arrays via highspy, so both
solvers see byte-identical matrices), `hs.py` (the shared HiGHS arm), `screen.py`
(the fidelity + tractability gate that built `panel.json`), `miplib_ab.py` (the
§0 arm panel), `attribute.py` (the §1c ablation), `switches_ab.py` (Stage 0) and
`pathology.py` (the `khb05250` 3-nodes-in-20 s probe).
