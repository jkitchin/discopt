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

This is a **root-strength** problem, not a tree-search problem. It is why Stage
0's search-side flips move zero instances, and it is the reason the stage order
below puts root cut control and cMIR ahead of everything on the search side.
Almost everything that matters happens before the first branch.

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

## 3. Stages

Reordered from the first draft per §2. Every bound-changing piece goes behind a
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

### Stage 1 — root cut-loop control (the §0 prerequisite)

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

### Stage 2 — cMIR done properly

*Attribution first, before any code:* compare discopt's post-cut root bound
against HiGHS's root bound on the four `mik` plus the six ≤3 %-dual instances.
**This is the test of §2.2's precondition.** If HiGHS's root gap on `mik` is also
7–10 %, cuts are not the lever there and this stage is re-scoped to search.

If confirmed, build in fidelity order, measuring after each: (1) integral row
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
