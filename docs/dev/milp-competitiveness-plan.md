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
`supportcase14` and `supportcase16` in **one node**. discopt spends 350 k–870 k
nodes on several of those same instances.

*Read §2.6 before drawing conclusions from that count.* "One node" is HiGHS
incrementing `num_nodes` on root *close*, not a statement that the root LP was
integral — and per-instance ablation shows the closer is a **restart** on
`beavma`/`dcmulti`, **integral-objective rounding** on `sp150x300d`, and a
**sub-MIP incumbent** on `nexp-50-20-4-2`. `b-ball`'s 1 node is an artifact of the
engine-form conversion: HiGHS takes 126 nodes on the original MPS.

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

## 2.6 Why HiGHS needs so few nodes — the mechanism, measured

Total panel nodes: HiGHS 15,417, discopt 4,693,240 at its best arm. ~300×.
HiGHS closes **16 of the 38 in a single node**. Before drawing conclusions from
that number, three corrections to how it must be read (all measured 2026-09-05,
`scratchpad/parity/attrib/panelform.py`, highspy 1.12.0, 30 s, the same engine-form
arrays discopt receives):

- **"1 node" does not mean "the root LP was integral."** HiGHS increments
  `num_nodes` on root *close* (`HighsMipSolverData.cpp:1802-1808, 1840-1850,
  1879-1884, 2370-2376`). It means the root loop — LP, cuts, heuristics,
  restarts — reached `mip_rel_gap` before any branching.
- **The closer is often not the cuts.** Per-instance ablation: on `beavma` root
  cuts do 99.4 % and a **restart** finishes it (`norestart`: 7 nodes); on
  `sp150x300d` cuts reach 68.18 of 69 and the bound then jumps to 69 with *no new
  cuts* — that is `computeNewUpperLimit` integral-objective rounding, not
  separation; on `nexp-50-20-4-2` the sub-MIP incumbent is load-bearing
  (`noheur` **times out**). Only `nexp-50-20-1-1` closes on root cuts alone.
- **`b-ball`'s "1 node" is a formulation artifact.** On the original MPS, HiGHS
  takes **126 nodes** and its bound stalls at −1.5996 with 303 conflicts. The
  1-node result is a property of the engine-form conversion, not of HiGHS.

And two of the panel's hardest instances are **not** root-closable by HiGHS
either: `gsvm2rl3` takes **634 nodes** (root 0.012 → 0.2306, 67 % of the gap, then
1394 conflicts and 15,179 strong-branching iterations), and
`neos-2624317-amur`'s HiGHS root bound stays at **0** — no HiGHS cut family fires
at its root, and it needs 3,838 nodes, 9,775 conflicts and 56,531 SB iterations
(38 % of all its LP iterations). **Treating amur as a root-cut problem would be
wrong**; it is a branching-and-conflicts problem for HiGHS too.

With those corrections, the mechanism is two effects multiplying.

**First: the last few percent of root gap is worth orders of magnitude of tree.**
Tree size grows roughly exponentially in the residual gap, so the interesting
comparison is not "discopt closes 74 % of the mik root gap" but what is *left*.
At 200 × 8, discopt leaves 4.5 % on `mik-250-20-75-*`; HiGHS leaves 1.7 %
(§2.5). A 2.6× wider residual gap is not a 2.6× bigger tree. The same pattern
holds on `sp150x300d` (15.71 % vs 0.00 %), `nexp-50-20-1-1` (30.00 % vs 0.00 %)
and `beavma` (31.05 % vs 0.00 %).

**Second: with no incumbent, nothing prunes.** A node is cut off only when its
bound is worse than the incumbent. discopt reaches the end of its budget 22-45 %
above the optimum on the four `mik`, and finds **no feasible point at all** on
`enlight_hard` and `neos-2624317-amur`. This is why `mik` burns 289 k-563 k nodes
against HiGHS's 558-1055 despite a dual gap of only 2.1-2.7 %.

Neither alone explains the ratio; together they do. And they are not independent
— a good incumbent early prunes the tree that would otherwise be needed to lift
the bound.

### 2.6a The separator-level answer: HiGHS has no GMI at all

Worth stating plainly because it inverts the natural assumption. HiGHS's tableau
separator does **not** generate Gomory mixed-integer cuts. It uses tableau rows
only as *base inequalities* fed to cMIR and lifted-cover generation with bound
substitution (`HighsTableauSeparator.cpp:40-244`, `HighsCutGeneration.cpp:505-738,
1391-1491`, `HighsTransformedLp.cpp:20-140, 150-460`), alongside three basis-free
families: Path (`HighsPathSeparator.cpp:170-549`), Mod-k
(`HighsModkSeparator.cpp:30-267`), implied-bound
(`HighsImplications.cpp:583-660`) and clique (`HighsCliqueTable.cpp:1604-1712`).

So the 70-92 % of root gap discopt's GMI already closes is being compared against
a solver that closes its last few percent with a *different and stronger* family.
This is the strongest single argument for A2, and it is why A2 is cMIR/flow-cover
rather than "more GMI".

**HiGHS also never rejects a cut for coefficient range.** Its only dynamism check
is inside `#if 0` (`HighsCutGeneration.cpp:1045-1063`). It gates *source rows*
instead — Tableau drops basis-inverse rows with weight ratio > 1e4, Path follows
only arcs with weight in `[feastol, 1/feastol]` (`checkWeight`, `:222-229`).
SCIP likewise **repairs rather than rejects**: `postprocessCut` (`cuts.c:3748-3810`)
runs `removeZeros(feastol)` → `cutTightenCoefs` → `removeZeros` again, and declares
a cut unusable only when a small coefficient's cancelling bound is infinite
(`:935-1000`). discopt is the outlier in refusing whole cuts on a numerical gate.

## 2.7 Instrument correction: PR #1164's `kept/gen` column is an artifact

`RootCutsGenerated`/`RootCutsKept` increment only inside the cut-cleanup block
(`milp_driver.rs:1226-1229`), which is gated by `prune_base` (`:934-939`) —
`Some(..)` only when `root_cut_prune && root_cuts > 0 && cut_rounds > 1`. **At
`cut_rounds = 1` they never fire.** The `0/0` cells reported for `base_16x1` and
`cut200_noprune` therefore mean *not counted*, not *no cuts generated*. The base
arm does cut: `b-ball`'s root bound is −1.705882 at `root_cuts=16` against
−1.818182 at `root_cuts=0`, and `enlight_hard` goes 21 → 23. This is a CLAUDE.md
§6 failure (a probe that measured nothing and was believed) and PR #1164's body
carries the correction.

## 3. The parity plan

*Goal:* 38/38 on this panel inside HiGHS's wall-clock envelope. That is the
definition of done; every stage below is scored against it.

*Ordering principle, from §1's measured decomposition:* **the bound is short on
17 of 17 failures; the incumbent is additionally short on 14 of 17.** Bound work
is therefore necessary everywhere and primal work is necessary on 14 and
sufficient on none — no instance closes until the bound arrives, however good the
incumbent. That is a strict ordering, and it is why Track A precedes Track B. But
Track B is not optional: on the `mik` family the bound is already within 2.6 % and
the tree still does not close, so Track A alone does not reach 38/38 either.

### Track A0′ — the audit's two candidate defects. One survives; do it first.

Both were found by the 2026-09-05 engine audit and both are **confirmed by
discopt's own split counters** (`scratchpad/miplib/verify_claims.py`, run with
`DISCOPT_PROFILE=1` set before `import discopt`, deltas taken pre/post). Each is
days of work, not weeks, and each unblocks machinery that is already built.

**A0′.1 — the short basis export (`neos-2624317-amur`).**
The cold root LP returns **338 basic variables for m = 342**. Everything
downstream declines silently: `separate_gomory_cols` returns empty at its
`basis.basic_vars.len() != m` guard (`lp/gomory.rs:233-235`) — with **no
counter**, so the panel reads "zero cuts" when the truth is "never looked" — and
`PreparedDual::prepare` refuses the warm start on shape, so nodes cold-solve.
Confirmed here: on amur, zero `RootCuts*` and zero `SubstDrop*` counters fire at
all, and `DualPrepRejectShape = 300` against `DualPrepAccept = 80`.

Origin: `lp/simplex/primal.rs:2585-2627`; the eligibility test at `:2601` is
`stat[j] != BASIC && nb_value(j) == 0.0` with `rows.len() == 1`. 50 of 342 rows
have their slack nonbasic at a *nonzero* value with no zero-valued structural
substitute, so the basis cannot be completed. At a complete basis a numpy replica
finds 201 fractional integer basics and **70 admissible GMI cuts** on the same
instance. Note this is *necessary but not sufficient* for amur: §2.6 shows HiGHS's
root bound on amur is 0 too, so cuts alone will not close it — but nothing else can
start while cuts, warm starts and strong branching are all disabled at once.
*Also add counters at `gomory.rs:233-235, 246-248, 271`: three silent early
returns, none instrumented. CLAUDE.md §6.*

**Result (2026-09-05) — shipped, cert-clean, net-positive on LP work.** The
eligibility test was wrong in its *quantity*, not merely its strictness. Swapping
a nonbasic row singleton `s` into a basic artificial's slot replaces `B`'s column
`±e_r` with `a·e_r` — a scalar multiple, still nonsingular — and leaves `x`
bit-identical whatever `s`'s value, because a basic variable may hold any value.
The primal argument never needed value zero. What the swap *can* break is the
exported `dual`, since a basic column requires `c_j − yᵀA_j = 0`. So the
admissible criterion is the **reduced cost**, and the value test missed every
dual-degenerate case. Fixed by a second pass (`select_row_substitutes`,
`primal.rs`) that fills only the rows pass 1 left empty — strictly monotone, so no
basis that completed before can be shortened.

Counter deltas on amur at the settings above: `DualPrepRejectShape` 300 → **0**,
`DualPrepAccept` 80 → **191**, `RootCutsGenerated` 0 → **102**, `RootCutsKept`
0 → **25**. Three blocked mechanisms unblocked by one criterion change.

38-instance node-limited differential panel (`nodepanel.py`, budget 5000,
`gap_tol=1e-4`, `root_cuts=500`, `cut_rounds=8`, `root_cut_prune=True`):

- *Cert-clean.* 76 bound-vs-optimum comparisons executed, zero violations;
  solved count unchanged at 14/38, no certified instance regressed.
- *Bound-neutral in the tree.* 133,680 → 133,662 nodes (−0.01 %); 36 of 38
  instances bit-identical in nodes **and** iterations. Expected: a warm start does
  not move an LP optimum, so it cannot move the tree.
- *Net-positive in LP work.* Total simplex iterations **6,492,081 → 4,057,317
  (−37.5 %)**, entirely in the four short-basis instances — `neos-3610040-iskar`
  1,898,737 → 29,372 (**−98.5 %**), `neos-2624317-amur` −20.8 %, `neos-911970`
  −6.2 %, `neos-3118745-obra` −1.5 %. The 34 untouched instances moving by exactly
  zero iterations also rules out nondeterminism as the source of the four deltas.

No wall-clock claim: load was 30–70 all day (CLAUDE.md §9). Iterations and nodes
are load-independent, which is why the gate is stated on them.

This is the shape to expect from every A0′ item: it does not move the solved
count by itself, because a node-limited panel spends its budget either way. It
removes a *blockage*, and the items downstream of it (A1's cut budget, A3's
strong branching) are the ones that convert freed LP work into bound. Note also
`convex_kernel.rs:652,1948` carries the same defect on the MINLP path — out of
scope for the MILP panel, and it needs the MINLP corpus to gate.

**A0′.2 — `gsvm2rl3`'s 31 refused cuts. FALSIFIED as a priority item; demoted.**

The audit proposed this as the second high-leverage fix: `gsvm2rl3` separates 31
cuts and discards all 31 at the infinite-bound pin
(`milp_driver.rs:3213-3222`) — confirmed here, `SubstDropUnbounded = 31`,
`SubstDropDynamism = 0`, so the *site* is right and the earlier
dynamism-backstop diagnosis was wrong. The instance has 61 FREE columns carrying
≤ 1.2e-17-relative cancellation noise, and the prescription was "zero before
pinning, as HiGHS does at `HighsTransformedLp.cpp:337-341`."

**Four measurements killed it, in order (2026-09-05):**

1. **The HiGHS mechanism is not what was reported, and is not portable.** The
   `small_matrix_value` cleanup at `:340` runs *after* the free-column refusal at
   `:173`, so it is not what saves HiGHS. What actually saves it is upstream, in
   the separator: `HighsTableauSeparator.cpp:158-165` discards basis-inverse
   weights with `maxAbsRowVal(row) * |weight| <= feastol` **before the row is
   aggregated**, so noise never becomes a nonzero. discopt's GMI path cannot
   adopt that: filtering `w` breaks the `ābar_B = e_i` identity that lets
   `gomory.rs` skip basic columns, and skipping them would then be an
   uncompensated — unsound — drop. HiGHS gets away with it because its filtered
   weights feed a general aggregator (`transform` + cMIR), not a tableau row with
   assumed unit structure.
2. **Compensated summation cannot help.** The residues are 4e-19 … 3e-16, which
   is per-*product* rounding in `f * base.val[k]`, not summation error. Kahan or
   Neumaier accumulation removes the latter and not the former.
3. **FBBT cannot bound the columns.** The SCIP review's explanation — "SCIP would
   refuse these cuts too; the difference is that its presolve bounds the
   variables first" — does not hold here. Ten interval-arithmetic FBBT passes over
   `gsvm2rl3` move it from **61 free structural columns to 61**
   (`scratchpad/miplib/fbbt_free.py`).
4. **The breadth argument evaporates.** The claim was "18 of 100 MPS files have FR
   columns; 6 on this panel." Six panel instances do have them
   (`scratchpad/miplib/freecount.py`) — but five carry exactly **one** free column
   (the objective variable) and **none of them loses a single cut**
   (`scratchpad/miplib/breadth.py`): `SubstDropUnbounded = 0` on all five, with
   24-176 cuts generated each. Only `gsvm2rl3`, with 61, is affected.

So the item reduces to: one instance out of 38, whose cut recovery is **already
measured node-neutral** (3 better / 3 worse at a fixed 5000-node budget — see
"Falsified: relaxing the cut-cleanup's numerical gates"), with no sound fix
identified. The refusal at the pin is **correct as written**: dropping a term
whose maximising bound is infinite is not a relaxation, and no amount of
smallness makes it one in exact arithmetic. It stays.

*What would revive it:* a cut family whose coefficients are conditioned by
construction (A2), or a genuine bound derivation for those 61 columns that FBBT
cannot reach. Not a laxer numerical gate — that road has now been measured shut
twice.

*Lesson recorded (CLAUDE.md §4, §11):* three separate expert-supplied mechanisms
for this one instance — dynamism backstop, zero-then-pin, presolve bounding — were
each plausible, each cited to real source, and each wrong. The counters and a
20-line numpy FBBT settled in minutes what the citations could not.

### Track A — the bound (needed by 17/17)

**A0. Graduate the root cut budget. Ready now; no new code.**
This was net-*negative* before the structural-space fix (`cut200_prune` 11/38
against base 18/38) because every cut broke node warm starts. With that fixed it
is net-positive: `cut500_prune` **21/38 against base 19/38 at 47 % fewer nodes**,
190/190 runs, zero certification aborts, no false bound. That meets CLAUDE.md §5's
double bar — cert-clean *and* net-positive — which it did not before.
*Expected: +2 instances, already measured.*

**GRADUATED default-ON, 2026-09-05.** `_milp_root_cut_budget` now returns the
budget unless `DISCOPT_MILP_ROOT_CUTS=0`; the legacy `root_cuts=16, cut_rounds=1`
single pass stays intact and reachable, which is what the panel A/Bs.

Confirming panel: 38 instances, `gap_tol=1e-4`, 20 s each, both arms interleaved
*within* every replicate, 2 replicates, and the whole panel run **twice** under
different machine load.

| arm | solved | total wall | med(solved) | nodes | gap on the 17 open |
|---|---|---|---|---|---|
| off (16 / 1) | 18-19/38 | 424.9 s ±4.2 | 0.158 s | 9,409,627 | mean 0.2028, med 0.1052 |
| on (500 / 50 / select) | **21**/38 | 399.8 s ±1.0 | 0.135 s | **5,353,150** | mean **0.1494**, med **0.0265** |

*Cert-clean*: 304 bound-vs-reference comparisons across the two runs, zero
violations; `fiber`, `gt2` and `neos-3611689-kaihu` go feasible → optimal and
nothing regresses. *Net-positive*: +3 instances, −43 % nodes, and a materially
better dual bound exactly on §1b's pure-dual family.

**On the wall column, and why it is not the basis of this verdict.** The machine
never idled below load ~5 — two `myst start` dev servers hold ~100 % CPU each,
and a foreign `pytest` arrived mid-run on the first attempt (load peaked at 58).
Per CLAUDE.md §9 the wall numbers are corroboration only. The verdict rests on
solved count, node count and dual bound, which are load-independent, and which
**both runs reproduced to four significant figures** (0.2028 → 0.1494 in each) —
two runs under different load agreeing exactly is stronger evidence than one
quiet run would have been. `gradpanel.py` now samples load per solve so a
contaminated row is recorded rather than inferred, and `gradscore.py` refuses to
issue a timing verdict when it sees contention.

**The cost, stated rather than buried.** Cuts buy the hard instances by taxing
the easy ones: ON is *slower* on 11 of the 18 both arms solve —
`neos-3611447-jijia` 7.5 → 13.9 s, `enlight8` 5.4 → 10.5 s, `22433`
0.31 → 2.51 s — against 7 faster (`bppc8-02` 3.79 → 1.44 s). Net-positive at 38
instances, but the per-instance tax is precisely the argument for A1's
stall-based termination and cut aging. A1 should be scored on removing this tax
without giving back the +3.

**A0 configuration check (2026-09-05) — the shipped setting is not the banked
one, and it had never been measured.** The banked arm is `cut500_prune` =
`root_cuts=500, cut_rounds=8, cut_select=False, root_cut_prune=True`. What
`_milp_root_cut_budget` (`solver.py:21476-21510`) actually hands `solve_milp_py`
is **different**: `root_cuts=500, cut_rounds=50, cut_select=True,
root_cut_time_s=max(0.5, 0.5·engine_budget)`. Flipping the default as written
would therefore have shipped a configuration no panel had ever run — the A0.1
wiring step above silently changes the thing A0 measured.

Measured with a **node**-budget panel (`scratchpad/miplib/armpanel.py`, 38
instances, `max_nodes=2000`, `gap_tol=1e-4`, three arms rotated per instance).
Node budgets rather than time budgets deliberately: the machine was at load ~21
all day, and node counts, iteration counts and dual bounds are load-independent
where wall-clock is not (CLAUDE.md §9). `time_limit_s=300` was present only as a
hang guard and bound on no row.

| arm | config | solved | nodes | simplex iters |
|---|---|---|---|---|
| `off` | 16 / 1 / no-select (today's default) | 11/38 | 59,070 | 1,498,688 |
| `prod` | 500 / 50 / select (what `_milp_root_cut_budget` ships) | 11/38 | 57,524 | 1,618,863 |
| `p500` | 500 / 8 / no-select (the banked `cut500_prune`) | 11/38 | 58,646 | 1,941,550 |

Cert-clean: 114 bound-vs-reference-optimum comparisons, **zero violations**, and
no instance changed solved status in either direction.

The payoff is in the **bound**, on the 27 instances no arm finished within 2000
nodes — mean dual gap `off` 0.2496 → `prod` **0.1883** (−25 %), median 0.1435 →
**0.0404**; `prod` is strictly better than `off` on 16, worse on 4, tied on 7.
The four `mik-250-20-75-*` move 0.143/0.127/0.145/0.162 → 0.038/0.036/0.037/0.038
— the §1b pure-dual family is exactly where the cuts land.

**The banked claim transfers, and the shipped configuration is the better of the
two**: `prod` beats `p500` on mean gap (0.1883 vs 0.1919), on best-arm wins
(19 vs 12) and on simplex iterations (1.62 M vs 1.94 M). So A0.1's wiring does
not need to be re-pointed at `cut500_prune`; `_milp_root_cut_budget` stays as it
is. Caveat on scope: at a 2000-node budget no arm converts a better bound into a
*solve*, so this panel confirms cert-cleanliness and bound quality but says
nothing about the banked "+2 instances", which was a 20 s wall-clock result. That
column still needs the unloaded-machine panel named above before the flip.

**A0.1 — wire the graduated budget to the product path.** The
`_STRONG_CUT_PROFILE` 200 × 10 escalation
(`python/discopt/solvers/milp_simplex.py:66-96, 985-1022`) sits on the OA/relaxer
entry. `Model.solve()` on a pure MILP goes `_milp_engine_default_on`
(`solver.py:21457-21474`) → `_solve_milp_simplex` (`:21563`) → `solve_milp_py`
(`:21692-21706`) with `_cut_opts = {}` unless `DISCOPT_MILP_ROOT_CUTS=1`
(`:21476-21510`). **A graduated cut default that never reaches a plain MILP solve
buys nothing for users.** Near-zero code; do it in the same PR as A0.

**A1. Finish the cut lifecycle (the remainder of the old Stage 1).**
The warm-start half is done. Still missing: stall-based termination instead of the
fixed round budget; cut *aging*; and a cut *pool* rather than the constraint
matrix. Both references agree on the shape and neither uses a fixed budget:
HiGHS runs `while (scaledOptimal && !fractional.empty() && stall < 3)` with
`maxSepaRounds = min(2·sqrt(maxTreeSizeLog2), ∞)` and an LP-iteration cap of
`max(10000, 10·avg)` (`HighsMipSolverData.cpp:1912-1915`); SCIP has
`maxroundsroot = -1` (`set.c:471`) and stops on `objreldiff ≤ 1e-4 AND nfracs ≥
(0.9 − 0.1·nstall)·prev` (`solve.c:2963-2996`), exiting at 10 stalls at the root
and 1 in the tree (`set.c:475, 477`). discopt's `1e-7*(1+|prev_obj|)` test is far
tighter than either. Aging: SCIP `LP_ROWAGELIMIT 10` (`set.c:263`), `cutagelimit
80` (`:483`); HiGHS ages basic cut rows at `mip_lp_age_limit`
(`HighsLpRelaxation.cpp:595-642`) and drops basic cut rows after the root
(`:522-539`). Also: `CUT_MAX_PARALLEL = 0.99` is 10× laxer than both references
(HiGHS `maxpar = 0.1`, `HighsCutPool.cpp:320`; SCIP `MINORTHO 0.90`), so
near-duplicate cuts are being kept.
*Both source reviews independently confirmed A1-before-A2: the lifecycle is the
prerequisite, since a stronger family is only affordable once a large cut set is.*
*Expected: the budget can then go higher than 500 without paying for it.*

**A2. The second cut family — cMIR/flow-cover with bound substitution, and
lifted knapsack covers.**
Correcting an earlier phrasing: GMI is *not* discopt's only separator —
`lp/mir.rs:206`, `lp/aggregation.rs:86` and `lp/cover.rs` all exist. But `mir` and
`aggregation` are reachable only from `bnb/convex_kernel.rs:680, 1972` and
`lp_bindings.rs:285`; **`milp_driver.rs` imports neither.** What is genuinely
absent is flow cover, variable-upper-bound substitution and lifted covers — a
`grep` of `crates/discopt-core/src/lp/*.rs` for flow-cover / VUB / bound
substitution returns nothing.

The structure census says this is exactly the right family for the failures
(executed, `scratchpad/parity/struct_probe.py`): `nexp-50-20-1-1` is 1030 columns,
245 binaries, objective on binaries only, with **245 two-variable VUB rows
`x_a ≤ u_a y_a`** and 50 all-continuous flow rows. `sp150x300d` is 1050 columns,
300 binaries, **150 big-M VUB rows** (binary coefficient ≥ 100× the continuous
one), and **all 750 continuous variables have finite upper bounds**, so
coefficient tightening fires on all 150.

SCIP's named answer for that class, in order: varbound upgrade
(`cons_varbound.c:108`, priority +50000), coefficient tightening
(`cons_linear.c:9003` — `x − M y ≤ 0` with `x ≤ ub_x` reduces M to `ub_x`), then
lifted flow cover and cMIR with VUB substitution (`sepa_aggregation.c:860-965`;
`SCIPcalcFlowCover` `cuts.c:11645`, citing Gu/Nemhauser/Savelsbergh 1999;
`SCIPcalcKnapsackCover` `:924`; `SCIPcutGenerationHeuristicCMIR` `:940`;
`VARTYPEUSEVBDS 2`). For `mik-*`/`beavma`, SCIP's lifted covers come from
`cons_knapsack` at `SEPAPRIORITY +600000` — *before any separator plugin* —
with sequential lifted minimal covers (`:5564, :2581, :5316, :4801, :5035`), and
for non-upgraded mixed rows via `SCIPseparateRelaxedKnapsack`
(`cons_knapsack.c:5781`, called from `cons_linear.c:7538-7620`), which relaxes
continuous and general-integer variables to their bounds or variable bounds.

**Coefficient tightening is the cheapest item in A2 and should be measured
first** — it is presolve-level, needs no separator, and the census says it fires
on all 150 `sp150x300d` VUB rows.

*Entry experiment before building (CLAUDE.md §4, the #727 RLT lesson): separate
cMIR at the root on `mik-*`, `nexp-*`, `sp150x300d`, `beavma` only, and measure
the root gap against the 4.5 %/15.7 %/30.0 %/31.1 % baselines. If it does not move
them on the real instances, it does not get built.*

**A3. Branching quality — the node-48 cliff.**
`sb_active = opts.strong_branch && tm.stats().total_nodes < opts.sb_node_budget`
(`milp_driver.rs:1434`) with a budget of **48 nodes** and ≤ 6 candidates. After
node 48, any variable with fewer than 8 observations falls back to
`default_cost = 1.0` in both directions, so the product score degenerates to
most-fractional for **> 99.99 % of every tree on this panel**.
`reliability_threshold = 8` is hardcoded at `tree_manager.rs:276`. The engine audit
rates this the single biggest quality gap in the driver. For scale, HiGHS spends
15,179 SB iterations on `gsvm2rl3` and 56,531 on `amur` — 38 % of that instance's
total LP work — where discopt spends its entire SB allowance in the first 48
nodes. Reliability branching (SB until a variable has *k* reliable observations,
not until node *k*) is the standard form and is what both references implement.
*Caveat from the b-ball probe: this will do nothing on dual-degenerate instances
— see Track C.*

### Track B — the incumbent (needed by 14/17)

discopt's failures are not marginal on this axis: 22-45 % above optimum on the
`mik` family, and no feasible point at all on two instances.

The inventory is now definitive (2026-09-05 audit). **ON:** per-node rounding
(`try_rounding_csc`, `milp_driver.rs:2299-2311` — nearest then floor, original
rows only) and a single root repair dive (`try_dive_repair`, `:2334-2344`, body
`:3742+`). **INERT:** off-root dives — `DIVE_STRIDE_DEFAULT = 0` (`:317`), and
`dive_batch_eligible` (`:351-360`) requires `!has_incumbent`, so it **can never
improve an existing incumbent**. **ABSENT:** feasibility pump, RINS, RENS, local
branching, any sub-MIP/LNS, shifting, ZI-round, feasibility jump,
propagation-based diving, plunging, restarts — and *any* improvement heuristic
whatsoever once an incumbent exists. That last one is the finding: discopt's first
incumbent is very nearly its last.

**B1. Node selection and plunging.** `SelectionStrategy::BestFirst` is
**hardcoded** at `milp_driver.rs:717` with `export_batch(64)` at `:1370`;
`DepthFirst` and `BestEstimate` exist in `bnb/pool.rs:10-18` (`:66-81`, `:82-96`)
and the driver never uses them. SCIP's default is `nodesel_estimate`
(`STDPRIORITY 200000`, the highest of five, `nodesel_estimate.c:48`) with plunge
parameters at `:56-62` (MINPLUNGEDEPTH `maxdepth/10`, MAXPLUNGEDEPTH
`maxdepth/2`, MAXPLUNGEQUOT 0.25, BESTNODEFREQ 10). Cheapest item in Track B —
the strategy exists and is unreachable — and the one that most directly attacks
"no incumbent".

**B2. Real primal heuristics.** Two shared pieces have to be built before any of
them: a **probing fix-propagate-undo stack** and a **reduced-copy sub-MILP call**.
With those, in order of value-per-unit-work:
1. **Randomized rounding after every root separation round while no incumbent** —
   HiGHS's `randomizedRounding` has no option and always runs
   (`HighsMipSolverData.cpp:1783-1790`); it is the source of `b-ball`'s incumbent.
2. **Shift-and-propagate** — SCIP's answer for the no-feasible-point case
   (`heur_shiftandpropagate.c:66-71`): runs *before* the root LP, relaxes
   continuous variables out of the rows, repeatedly fixes the discrete variable
   that most reduces weighted row violation, propagates in probing mode,
   `onlywithoutsol TRUE`. This is the shape of `enlight_hard` and `amur`.
3. **Feasibility pump**, then **RENS**, then **RINS**/local branching (sub-MIP).
   Measured relevance: on `nexp-50-20-4-2` HiGHS's sub-MIP incumbent is
   load-bearing — `noheur` times out where the default takes 1 node.

*Do not port* `ziRound` or `shifting`: both are **default OFF** in HiGHS
(`HighsOptions.h:1234-1241`).

**B3. Restarts.** Measured as the actual closer on `beavma` (`norestart`: 1 → 7
nodes) and `dcmulti` (two restarts), and `noheur` `beavma` closes via **five**
restarts. SCIP triggers at ≥ 2.5 % root integer fixings (`set.c:363-371`,
`solve.c:4953-4958`). discopt has none. Cheap relative to B2 and independently
measured to matter.

### Track C — measured NOT to be the difference. Do not spend here.

Recorded so this ground is not re-covered:

- **Presolve** — `h_nopre ≈ h_full` everywhere on the root-bound probe; worth
  ~0-3 pp of root gap (§2.5a). §1c independently put it at ≤2 instances.
  (Caveat: SCIP's *coefficient tightening* is presolve-shaped and is **not**
  covered by this negative — it is an A2 item with its own census evidence.)
- **Symmetry** — 0 instances on this panel (§1c); HiGHS `nosym` is identical to
  default on every instance ablated.
- **`node_propagation`** — graduated ON, but gains **zero** instances; the HiGHS
  review's "3-6 instances" is falsified (Stage 0.1).
- **Relaxing the cut numerical gates** — cert-clean but node-neutral; falsified
  and reverted, with the measurement recorded at the refusal site. (A0′.2 is a
  *different* change: zero-then-pin, not keep-and-widen.)
- **`cut_select`** — falsified earlier: `fiber` 2986.5 → 6457.4 it/node, `mik`
  acceptance 100 % → 35.0 %.
- **Integral-objective cutoff tightening** — **already built and wired.**
  `bnb/obj_integral.rs` computes the lattice and `set_objective_lattice`
  feeds `cutoff_value` (`tree_manager.rs:1105-1113`), default ON via
  `DISCOPT_OBJ_INTEGRALITY`. Both source reviews flagged HiGHS's
  `computeNewUpperLimit` as a possible gap; it is not one. It simply does not
  apply to `b-ball`, whose objective variable is continuous
  (`obj_integral.rs:121` correctly declines a lattice).
- **Pruning mechanics** — verified exact. `tree_manager.rs:557-562` prunes on
  `node_lb >= cutoff_value()`; `cutoff_value` is the incumbent exactly;
  `gap_tol` is applied only at `milp_driver.rs:1357, 1872`; the global bound is
  recomputed from the open frontier every batch (`:917, 923-973`); ties are
  explored, not pruned; and **no objective cutoff reaches the node LP** —
  `SimplexOptions` has no such field and a grep of `lp/simplex/*.rs` for
  `cutoff|obj_limit|objective_limit` is empty. SCIP's cutoff test
  (`SCIPsetIsGE(lowerbound, cutoffbound)`, `solve.c:3164`, eps 1e-9, with
  `cutoffbound = upperbound` exactly, `primal.c:428-441`) would not prune those
  nodes either. An earlier "pruning mechanics" framing was retracted and both
  reviews independently confirmed the retraction.
- **`b-ball` by branching or by strong branching** — it is a **dual-degenerate
  plateau, measured**. The bound is flat at −1.705882 from node 127 through
  32,099 nodes under *every* switch tried: whole-tree strong branching with
  16,044 calls, `sb_max_cands=30`, `node_propagation`, heuristics off,
  reduced-cost fixing off, `node_cuts`, `root_cut_prune` off. A degeneracy probe
  found **350/350 child LPs equal to their parent** (176 at depth 1, 174 at
  depth 2). Every pseudocost is 0, so branching is a coin flip. Structure: 8
  assignment rows each picking 5 of 11 over 88 binaries, 11 linking equalities
  `2·x_i = Σ_k x_{i,k}`, 11 rows `x12 ≤ x_i`, objective `min −x12`; the LP gives
  40/11 = 1.818 and the integer answer is `floor(40/11) = 3` → −1.5. **Only cuts
  or presolve can move this instance.** A HiGHS-style gap-aware node prune would
  not help either: its threshold is −1.50015 against a frontier at −1.500673.

### Measurement discipline for every stage below

1. Node-limited panels while machine load is high — a time limit measures the
   machine (CLAUDE.md §9). Load was ~68 during the 2026-09-05 work; every
   conclusion drawn that day rests on node counts and counters, never wall.
2. Every bound-changing item behind a default-off flag with the §5 double bar
   (cert-clean AND net-positive) over the 38 panel *and* the in-repo MINLP corpus.
3. Entry experiment before implementation, on **real corpus instances**, with a
   named kill criterion.
4. Every new silent early return gets a counter. Two of this plan's root causes
   (A0′.1, A0′.2) were invisible for weeks because the code declined without
   counting, and one of them was misdiagnosed three times before the counter was
   split. CLAUDE.md §6.
## 3bis. Stage detail (the original numbering, retained)



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

**A1 entry experiment (2026-09-05) — the stall rule has never fired.** Before
implementing a replacement, the existing rule was instrumented on a 3-row,
11-column knapsack fixture, with the three constants temporarily promoted to
`MilpOptions` knobs (`cut_stall_rel`, `cut_stall_rounds`, `cut_max_parallel`)
defaulting to exactly the values they replaced, so the measurement is
bound-neutral by construction. **Those knobs, and the `RootCutStallRounds`
counter used below, were measurement scaffolding on a throwaway branch and are
deliberately NOT in the tree** — every arm measured neutral-to-harmful, and
CLAUDE.md §3 forbids shipping a flag with no consumer. They are cheap to
reconstruct from this section if a later change needs them:

| arm | rounds run | stalls registered |
|---|---|---|
| shipped `1e-7`, 1 round | **40** (the cap) | **0** |
| SCIP's `1e-4`, 1 round | **14** | 1 |
| `1e-4`, 3 rounds | 40 | 10 |

The historical test `root.obj <= prev_obj + 1e-7*(1+|prev_obj|)` registers **zero
stalls in forty rounds**. It is not merely "far tighter than either reference" as
drafted above — it is inert. `cut_rounds = 50`, which A0 graduated default-ON,
therefore means fifty full separation rounds on *every* instance, whether or not
the bound is still moving. That is the direct mechanism of the per-instance tax
A0 recorded (slower on 11 of the 18 instances both A0 arms solve;
`neos-3611447-jijia` 7.5 → 13.9 s, `enlight8` 5.4 → 10.5 s, `22433` 0.31 →
2.51 s). SCIP's threshold stops the same fixture at 14 rounds.

Orthogonality is loose by the same margin: `CUT_MAX_PARALLEL = 0.99` against
HiGHS's `maxpar = 0.1` and SCIP's `MINORTHO 0.90`. Tightening it to 0.05 takes
the kept set on the fixture from **31 cuts to 12**.

**RETRACTION AND RESULT (2026-09-05).** The entry experiment above was run on a
unit fixture and its conclusion was over-generalized in this document, per
CLAUDE.md §11. Corrected on real corpus instances (the shipped
`RootCutRounds` / `RootCutsGenerated` counters plus a scaffolding
`RootCutStallRounds`, 12 panel instances, shipped A0 config):

| binding exit | count |
|---|---|
| round cap — the only case where the stall threshold can act | **1** (`gr4x6`) |
| 500-cut generation cap, reached at 11–14 rounds | 3 (the whole `mik` family) |
| stall rule already fires at the shipped `1e-7` | 2 (`enlight8`, `neos-911970`) |
| separation stops producing violated cuts | 6 |

The rule is inert *on the fixture*; on real instances it fires occasionally, and
more importantly the loop almost always exits for some other reason long before
50 rounds. `cut_rounds = 50` is therefore not a description of what the solver
does, and tightening its stopping threshold cannot buy much.

**The A1 arms panel confirms that** (38 instances, TL 20 s, arms interleaved,
190 comparisons, cert-clean with 0 bounds above the reference optimum):

| arm | solved | node geomean | iter geomean (21 solved) |
|---|---|---|---|
| `prod` (shipped) | 21 | 1.000 | 1.000 |
| `stall` (SCIP `1e-4`) | 21 | 1.025 | **0.983** |
| `stall3` (`1e-4`, 3 rounds) | 21 | 1.088 | 1.157 |
| `ortho` (maxpar 0.10) | 21 | 1.477 | 1.386 |
| `both` | **20** (loses `gt2`) | 1.654 | 1.049 |

`stall` is worth **1.7 % of simplex iterations** on the instances that solve —
noise, not a lever. Tightening orthogonality is actively harmful (node geomean
1.477), reproducing at the loop level the same result §2.5c already recorded for
`cut_select`. **The kill criterion fires: no arm survives, and the cut-loop
stopping rule is not the mechanism of A0's per-instance tax.**

*A note on the instrument.* The panel's wall column showed `stall` at 0.834,
which reads as a 17 % win and is the number a careless writeup would have
published. Load peaked at 57 during the run (an out-of-session `cargo check
--workspace --all-targets`), so under §9 wall could not carry the verdict;
scoring the same arms on simplex iterations — load-independent — gave 0.983. The
wall figure was mostly artifact. The panel now refuses to start above a load of
6 rather than recording the excursion and hoping the reader discounts it.

*Where A1 goes instead.* The measurement points at the two exits that actually
bind. The `mik` family generates its full 500-cut budget and keeps 88–114 of
them, and those cuts then ride in **every node LP** for the rest of the search;
six other instances stop because separation dries up, which no stopping rule
improves. That is the in-tree half of Stage 1 — cut aging with a ~10-LP basic
limit, and a pool that re-checks violation — not the root loop's termination
test. A1 re-scopes accordingly, which is the fallback this section named in
advance.

*Kill criterion for the A1 arms panel:* an arm must remove A0's per-instance tax
**without giving back A0's +3 solved instances**. If none does, the stopping rule
is not the tax's mechanism and A1 re-scopes onto in-tree aging and the pool.

### A2 — in-tree cut aging. Measured 2026-09-05, and it does NOT survive.

A1 handed off to in-tree aging on the premise, never measured here, that the root
cuts discopt keeps are largely **dead weight** in the tree: costing basis
dimension in every node LP while contributing nothing to that node's bound. Per
CLAUDE.md §4 that premise was tested before any aging code was written.

*Instrument.* Every node LP that reaches optimal classifies each surviving cut row
by its slack column — `x[j] > 1e-7` means the cut is not tight at that node and is
exactly what aging would delete. 36 of 38 panel instances classified (two generate
no cuts), shipped A0 config, 20 s limit.

*Round 1 — the headline ratio, and why it could not decide anything.*

| statistic | value |
|---|---|
| pooled slack share over all (node, cut) pairs | **44.7 %** |
| median per-instance slack share | **51.4 %** |

The pre-registered kill line was *median below 50 % → KILL*. The median is 51.4 %,
so **the criterion as written did not fire**, even though the work-weighted pooled
figure (44.7 %) sits the other side of the line. Recording that rather than
switching to whichever statistic gives a clean answer: post-hoc statistic choice
is the failure §4 exists to prevent, and the honest verdict was MARGINAL.

*Why a slack share cannot settle it.* An aging rule does not delete a row that is
slack *now*; it deletes one non-binding for ~10 **consecutive** node LPs. A 45 %
pooled share is equally consistent with durable dead weight (aging works) and with
rows alternating in and out of the basis (aging thrashes and removes nothing) —
and those imply opposite decisions. The headline probe cannot distinguish them, so
it was extended rather than believed.

*Round 2 — age bucketing.* Each slack observation is bucketed by the row's
consecutive-slack age at the moment of observation; the **aged** bucket is the
fraction an aging rule with threshold 10 could actually have removed. Criterion
pre-registered in the script before the run: ≥ 25 % → build it, < 10 % → kill,
10–25 % → a second marginal answer means the effect is not there.

| statistic | value |
|---|---|
| pooled slack share (reproduces round 1) | 44.5 % |
| median per-instance slack share (reproduces round 1) | 51.4 % |
| **pooled AGED share of all (node, cut) pairs** | **22.5 %** |
| of the slack observations, durable (age ≥ 10) | 50.5 % |

**VERDICT: A2 does not survive.** 22.5 % lands inside the marginal band, below the
25 % line. And 22.5 % is an *upper* bound on what aging could buy, because it
credits aging with deleting every aged row at zero cost — while a row aged out can
be needed again lower in the tree, at which point its bound is simply lost.

*The per-instance spread is the substance, not the pooled number.* The two
populations are cleanly separated and neither is the one aging needs:

| instance | nodes | slack % | aged % |
|---|---|---|---|
| `blend2` | 5.6 k | 87.7 | 79.3 |
| `enlight_hard` | 97 k | 75.7 | 55.3 |
| `p0201` | 438 | 51.1 | **1.1** |
| `mik-250-20-75-5` | 183 k | 42.4 | **12.2** |
| `neos17` | 44 k | 40.8 | **8.3** |
| `beavma` | 79 k | 24.8 | **9.0** |

Where slackness is durable the instance is already small and already solved
(`blend2`, 5.6 k nodes). On the large-tree instances that actually time out — the
`mik` family, `beavma`, `neos17` — slackness is **transient**: rows drift in and
out of the basis, so aging would spend deletions and re-separations without ever
shrinking the working LP much. `p0201` is the extreme: half its cut rows are slack
at any moment and almost **none** of them durably.

*Consequence.* Three measured rounds have now gone into the cut path — root
attribution (§2.5), root-loop termination (A1), in-tree aging (A2) — and none
found the lever. That is itself the finding: **discopt's cut machinery is not
what separates it from HiGHS.** The next question is deliberately upstream of
mechanism choice — which half of `time = nodes × cost_per_node` the gap lives in —
because tree size and per-node cost imply disjoint work and three rounds of
picking a mechanism first have not paid.

*Scaffolding disposition.* The A2 counters are on a probe branch and ship only if
something consumes them; a counter with no consumer is CLAUDE.md §3's dead flag,
which is why A1's stall counter was dropped rather than merged.

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

### A3 — where the gap actually is. Measured 2026-09-05, and it ends the cut search.

Three measured rounds have now gone into the cut path — root attribution
(§2.5), root-loop termination (A1), in-tree aging (A2) — and none produced a
lever. That is itself a result: the next question is not *which cut mechanism*
but whether the cut path is the right place at all. So the probe stopped
choosing mechanisms and measured the shape of the gap directly.

`scratchpad/miplib/nodegap.py`, full §0 panel, 38 instances, `gap_tol=1e-4`,
TL = 20 s, HiGHS and discopt run back-to-back on each instance so both meet the
same machine load. Ratios are taken **only over the 21 instances both solvers
drive to optimality** — a node count is a complete, load-independent quantity
only for a run that *finishes*; under a wall-clock cap a loaded machine explores
fewer nodes, so a timed-out instance's node count is a timing measurement in
disguise. Zero certificate violations (the probe exits non-zero on any bound
above the reference optimum).

| | geomean, discopt ÷ HiGHS, over the 21 both-solved |
|---|---|
| nodes explored | **105×** more |
| nodes per second | **80×** more |

Solved counts on the run were discopt 21/38, HiGHS 37/38; those are wall-clock
gated and therefore load-sensitive, so they are context, not the finding.

**discopt is not slow.** It is roughly two orders of magnitude *faster* per node
than HiGHS and pays for it with a tree two orders of magnitude larger. The
per-instance spread says the same thing without the geomean: `gt2` 2643 nodes at
39052 n/s against HiGHS's 1 node at 39 n/s; `dcmulti` 3593 at 2197 against 5 at
5; `enlight8` 129509 at 43582 against 444 at 339. Two instances run the other
way — `neos-911970` (discopt 31 nodes at 2 n/s vs HiGHS 1907 at 269) and
`neos-3118745-obra` — and they are the exception that shows the measurement is
not an artifact of the harness.

The consequence for planning is direct, and it is why no further cut-mechanism
round should be started before it is used: **discopt has an enormous per-node
budget available to trade for better decisions.** Any mechanism that cuts the
tree by 2× is worth up to ~40× the per-node cost it adds and still comes out
ahead. That is a very different economy from the one the cut rounds were
implicitly optimizing, where the aim was to add bound cheaply.

Two candidates spend that budget, and they want opposite work: a weak
**incumbent** prunes nothing however good the bound is (lever: primal
heuristics — discopt has rounding plus the #1060 continuous-repair dive, and no
RINS/RENS/feasibility-pump/local-branching), while a weak **dual bound**
certifies nothing away however good the incumbent is (lever: branching and
bounding). Both produce the identical symptom measured above, so the node ratio
cannot separate them; `scratchpad/miplib/gapsplit.py` separates them by
splitting the gap still open at the time limit against the reference optimum.

One specific suspect is already visible in the tree and costs nothing to test.
Strong branching is gated by

```rust
let sb_active = opts.strong_branch && tm.stats().total_nodes < opts.sb_node_budget;
// crates/discopt-core/src/bnb/milp_driver.rs:1434
```

and the **Python binding default is `sb_node_budget = 48`**
(`crates/discopt-python/src/lp_bindings.rs:1102`, `:1230`) — which every panel
measurement in this document has silently used. Strong branching therefore runs
for the first 48 nodes, and the pseudocosts seeded from those 48 steer the
remaining 100k–480k. HiGHS instead budgets strong branching in LP *iterations*,
recomputed at every branching decision so that it never fully expires
(`HighsSearch.cpp:1272-1290`). Note that `milp_driver.rs:4573`'s `1024` is a
**test helper**, not a default, and the Rust `Default` impl at `:4211` is `1000`;
neither reaches a Python solve. `scratchpad/miplib/sbbudget.py` sweeps the
budget — it is already a parameter, so the falsifying experiment needs no code.

### A4 — the strong-branching budget is NOT the lever. Falsified 2026-09-05.

A3 established that discopt's gap is tree size (105x more nodes at 80x higher
throughput), so it has a large per-node budget to trade for better decisions.
The cheapest place that budget could go was already a parameter, so it was
tested first.

Strong branching is gated by

```rust
let sb_active = opts.strong_branch && tm.stats().total_nodes < opts.sb_node_budget;
// crates/discopt-core/src/bnb/milp_driver.rs:1483
```

and the **binding default is `sb_node_budget = 48`**, set in the Python binding
(`crates/discopt-python/src/lp_bindings.rs:1102`, and again at `:1230`, `:1389`).
Every panel measurement in this plan has silently used it. (Two other constants
in the tree are *not* the default and must not be cited as such: the Rust
`Default` of `1000` at `milp_driver.rs:4211`, and `1024` in the test helper
`fn opts(..)` at `:4573`. Neither reaches a Python solve.) So strong branching
runs for the first 48 nodes, and the pseudocosts it seeds then steer the
remaining 100k-480k. HiGHS instead budgets SB in **LP iterations** — about a
third of all non-heuristic LP iterations, recomputed at every branching decision,
so it never fully expires (`HighsSearch.cpp:1272-1290`).

**Hypothesis:** the SB budget *shape* — a small fixed node prefix — is a binding
constraint on discopt's tree size.

**Experiment** (`sbbudget.py`, 38-instance MIPLIB easy panel, `gap_tol=1e-4`,
TL 20 s, four arms run back-to-back per instance so they meet the same load):
`sb_node_budget` in {48 (shipped), 1000, 50000, 10^7}. Kill criterion registered
before the run: geomean node reduction over instances **every** arm drives to
optimality, `>= 1.5x` survives, `< 1.25x` falsifies.

**Result — FALSIFIED.** 21 of 38 instances solved by all four arms (the same
21/38 for every arm):

| shipped48 / arm | geomean node reduction |
|---|---|
| n1000 | 1.151x |
| n50k | **1.221x** |
| unlimited | 1.221x |

The best arm lands below even the 1.25x "weak" floor. **Do not spend on the SB
budget shape.**

**The distribution is the finding, and it is bimodal, not flat.** Ten of the 21
instances are *exactly* 1.00x — the budget changes nothing — while a minority
move hard:

| instance | shipped48 | n1000 | n50k | unlimited | 48/best |
|---|---|---|---|---|---|
| `fiber` | 23537 | 3187 | 2081 | 2081 | **11.31x** |
| `dcmulti` | 3593 | 1105 | 1105 | 1105 | **3.25x** |
| `neos-3611447-jijia` | 100541 | 54261 | 55601 | 55601 | 1.85x |
| `23588` | 4449 | 2529 | 2587 | 2587 | 1.76x |
| `p0201` | 441 | 685 | 685 | 685 | **worse** |

So more strong branching is a *large* lever on a few instances, inert on half,
and actively harmful on at least one. That is the shape a per-node cost model
predicts: SB pays where the LP discriminates between candidates and wastes LPs
where it does not. HiGHS acts on exactly that distinction and discopt does not —
`setMinReliable(0)` shuts strong branching off entirely at a degenerate node
(`HighsSearch.cpp:1114-1116`), where `computeLPDegneracy >= 10` is reached by
`varConsRatio >= 2` alone (`HighsLpRelaxation.cpp:438-493`). discopt's
`Pseudocosts` (`crates/discopt-core/src/bnb/branching.rs:51-62`) carries cost
sums and counts only — no inference count, no cutoff count, no degeneracy factor
— so at a degenerate node it has nothing to fall back on and keeps buying SB LPs
that cannot discriminate. **A degeneracy shut-off is a live, general candidate**
("stop paying where the probe cannot separate"), and it is falsifiable on this
very panel: if degeneracy separates `p0201` from `fiber`, it is a mechanism; if
it does not, the explanation is dropped rather than kept as a story.

**Two corrections to this probe's own instrument, recorded per CLAUDE.md §6/§11.**

1. *The run first exited on a gate that was mis-specified, not on a solver
   defect.* The gate required two solved arms to agree on the objective to 1e-6.
   That asserts an exactness the solve never promised: the panel runs
   `gap_tol=1e-4`, so "optimal" means "gap <= 1e-4" and two arms may legitimately
   stop at different incumbents inside that band. `gen` did exactly that —
   `shipped48` returned the reference optimum to 1.3e-16, the larger-budget arms
   returned an incumbent 5.76e-5 **above** it (worse, correct side for a
   minimize, inside the certified gap). The gate is now anchored to the **oracle**
   instead: every solved arm's incumbent must lie in
   `[optimum, optimum + gap_tol*(1+|optimum|)]`. That is not a loosening — it is
   *stricter* in the direction that matters, because an incumbent below the true
   optimum is caught outright, whereas an arm-vs-arm test passes two arms that
   are wrong the same way. The dual-bound gate ran ahead of it and passed on all
   38 x 4 arms, which is why the node table above is readable at all.

2. *A registered caveat, resolved empirically rather than waived.* Before the
   run it was stated that a FALSIFIED verdict would not be acceptable without an
   `SbCalls` counter, because a flat arm cannot be distinguished from "strong
   branching never ran". The table settles it without the counter: node counts
   **change** between `shipped48` and `n1000` on 16 of the 21 instances, so SB
   demonstrably fires more at the larger budgets and moves the search. This is
   not the instrument-measured-nothing failure mode. The counter still ships (it
   sharpens the per-instance picture); it is simply not load-bearing here.

**What A4 leaves standing.** A3's finding is untouched: the gap is tree size.
What A4 removes is the cheapest explanation for it. The remaining candidates,
both upstream of any scoring parameter, are (i) discopt discards a proof it has
already paid for — an infeasible SB probe proves `x >= ceil(x_i)` at that node,
and `strong_branch` turns it into a score constant (`INFEAS_DELTA = 1e7`,
`milp_driver.rs:2720`) and throws the domain reduction away; HiGHS instead
collapses the branch (`branchUpwards`, `HighsSearch.cpp:558-561, 665-669`),
pushing one child and setting the parent's `opensubtrees = 0`, and it needs no
conflict analysis to do it — and (ii) discopt has no plunging at all.

### A5 — the gap is PRIMAL, and specifically the *improving* kind. Measured 2026-09-05.

A3 said the gap is tree size. A4 removed the cheapest explanation for it. This
probe asks the question that decides what to build: on the instances discopt
fails to close, is the open gap the incumbent's fault or the bound's?

**Method** (`gapsplit.py`, same 38-instance MIPLIB easy panel, `gap_tol=1e-4`,
TL 20 s, single arm). For each unsolved instance with a known optimum `z*`,
split the open span `[bound, incumbent]` at `z*`:

```
pshare = (incumbent - z*) / (incumbent - bound)      dshare = (z* - bound) / (incumbent - bound)
```

Registered before the run: median `pshare >= 0.60` -> PRIMAL lever (build
heuristics); `<= 0.20` -> DUAL lever (branching/bounding); between -> both live.

**Result: 17 of 38 unsolved, median primal share 89.6%, median dual share 10.4%
-> PRIMAL.** Zero certificate violations (no bound above an optimum, no incumbent
below one).

**The three-way breakdown is the actionable part, not the median.** The 17 split
cleanly and the groups want different things:

| group | count | instances | what it needs |
|---|---|---|---|
| incumbent found but POOR | **12** | `mik-250-20-75-{1,2,3,5}`, `neos-911970`, `nexp-50-20-{1-1,4-2}`, `sp150x300d`, `neos17`, `beavma`, `neos-3118745-obra`, `gsvm2rl3` | **improving** heuristics + diving |
| no incumbent at all | 2 | `enlight_hard`, `neos-2624317-amur` | find-any-point (RENS / pump class) |
| incumbent IS the optimum | 3 | `neos-3610051-istra`, `neos-3610040-iskar`, `b-ball` | bounding only |

And the poor incumbents are *very* poor — this is not a matter of a few percent:

| instance | incumbent | optimum | off by |
|---|---|---|---|
| `neos-911970` | 215.69 | 54.76 | 294% |
| `beavma` | 593880 | 383285 | 55% |
| `mik-250-20-75-5` | -28299 | -51532 | 45% |
| `gsvm2rl3` | 0.586 | 0.337 | 74% |
| `mik-250-20-75-1` | -34400 | -49716 | 31% |

**This corrects a framing error recorded earlier in this plan.** The working
assumption had been that discopt's primal weakness is a *feasibility* problem,
and RENS was discussed as the answer on the grounds that HiGHS invokes it
precisely under `if (mipdata_->incumbent.empty())`
(`HighsMipSolver.cpp:275-299`). That is true of HiGHS and irrelevant here:
discopt finds a feasible point on 15 of 17 unsolved instances and 12 of those
are simply *bad*. The lever is the **improvement** class — RINS-style
neighborhood search, local branching, and diving — not the find-any-point class.
Only 2 of 17 want a RENS/pump, which independently confirms A4's note that the
feasibility pump is not worth building early.

**It also strengthens the case for plunging** rather than replacing it. Diving
is the structural mechanism that produces *better* incumbents earlier: HiGHS's
dive nodes are the large majority of its nodes, and its rounding heuristics run
*inside* the dive loop. The 12-instance group is exactly the population that
would benefit, and it is the largest group on the panel.

**Instrument defect found and fixed, recorded per CLAUDE.md §6.** The scored set
was computed with `has_inc = st in HAS_INCUMBENT and obj is not None`, with no
finiteness test. `enlight_hard` and `neos-2624317-amur` returned `node_limit`
with `obj = inf`, which passes a None-check; `span` became `inf`, `pshare`
became `nan`, and the summary line then reported "NO INCUMBENT AT ALL: 0" while
two rows printed `nan`. Worse, the wiring guard written to catch exactly this,

```python
if abs(r["pshare"] + r["dshare"] - 1.0) > 1e-6:   # abs(nan) > 1e-6  ->  False
```

was **defeated by the nan**: every comparison against nan is False, so the guard
passed silently and `statistics.median` then sorted a list containing nan, where
order is undefined. Recomputed with the two rows set to what they actually mean
(no incumbent -> `pshare = 1.0`), the median is **unchanged at 89.6%** — the two
values sort to the top and do not move the 9th of 17. The verdict therefore
stands, but it stood by luck and not by correctness. Both defects are fixed: the
finiteness test is in `has_inc`, and the guard is now written `not (… <= tol)`
rather than `… > tol`, which is the form that *fails* on nan instead of passing.

### A6 — the discarded infeasibility proofs are real but too rare. Falsified 2026-09-05.

**The hypothesis.** `strong_branch` (`milp_driver.rs:2740-2795`) turns an
infeasible probe into the score constant `INFEAS_DELTA = 1e7` and throws the
*proof* away. An infeasible down probe proves `x_i ≥ ceil(x_i)` at that node; if
both probes on a candidate come back infeasible the node itself is infeasible and
could be pruned outright. HiGHS harvests exactly this (`branchUpwards`,
`HighsSearch.cpp:558-561`, `:665-669`). The claim was that discopt is paying for a
full warm dual re-solve per probe and discarding the domain reduction it buys.

**Soundness was checked first, and it holds.** Two questions had to clear before
this was even buildable, and both did:

1. *Is the probe's `Infeasible` conditional on a cutoff?* No. The probe closure
   (`milp_driver.rs:2705-2721`) installs no incumbent and no cutoff row, so
   `LpStatus::Infeasible` is a pure feasibility statement. HiGHS's third arm —
   optimal-but-bound-exceeding, needing `other_child_lb` and deferred-subtree
   bookkeeping (`HighsSearch.cpp:613-637`) — does not arise on discopt's path.
2. *Can a numerical false-infeasible become a bound that cuts off the optimum?*
   No. **Every** `LpStatus::Infeasible` this engine can emit is Farkas-certified.
   The primal has exactly two emission sites (`primal.rs:1330`, `:1359`), each
   gated on `farkas_ray_certifies`, returning the honest `Numerical` when the ray
   does not certify; the dual (`dual.rs:940-969`) gates its own the same way and
   returns `None` for a cold primal fallback rather than emit uncertified. Notably
   discopt would collapse on a *stronger* proof than HiGHS, which has the
   equivalent check short-circuited off — `HighsLpRelaxation.cpp:1228` reads
   `if (true || checkDualProof()) return Status::kInfeasible;`.

So the mechanism was cleared on soundness. It is the **frequency** that kills it.

**The measurement.** 38-instance panel, shipped `sb_node_budget = 48`, all 38
instances **node-capped at 3000** rather than wall-capped — deliberately, because
the machine was at load ~30 from unrelated work and a share gathered under a wall
cap is partly a timing measurement (§9). Zero instances hit the wall valve, so
every count below is a function of the model and not of machine speed. Zero
certificate violations.

| bucket | count | share of probes |
|---|---:|---:|
| `SbProbes` | 11582 | — |
| `SbProbeOptimal` | 11240 | 97.05% |
| **`SbProbeInfeasible`** | **342** | **2.95%** |
| `SbProbeOther` | 0 | 0.00% |
| `SbCandBothInfeasible` | **1** | — |

**Against the pre-registered criterion** (written into `profile.rs` before the
run): ≥5% builds, <1% does not, and 1–5% is WEAK — *"build only if the
both-children-infeasible count is itself non-trivial."* The share is 2.95%, and
the both-infeasible count is **1 across the entire panel**. The whole-node-prune
arm, which was the valuable half, essentially never fires. **A6 is falsified and
the collapse is not being built.**

**Retraction (§11).** Earlier in this session the collapse was described as
"cleared for a v1 build". That was a statement about *soundness*, and it remains
true; it was not a statement about frequency, and it was made before the frequency
was measured. The build decision it implied is withdrawn.

**What the same run turned up that is NOT falsified.** Strong branching is gated
by `sb_active = opts.strong_branch && tm.stats().total_nodes < opts.sb_node_budget`
(`milp_driver.rs:1434`), a **global node budget**, defaulting to 48 in the Python
binding. That is why `SbCalls` pins near 31 on every instance regardless of size:
strong branching in discopt is a root-region activity that switches off entirely
after the first 48 nodes, and everything below runs on pseudocosts alone. A4
already swept that budget to unlimited and got only 1.22×, so running SB deeper is
not the answer — but it means the pseudocost model is what steers essentially the
whole tree, which is where A7 goes.

### A7 — the pseudocost default *is* mis-scaled, and fixing it is not enough. 2026-09-05.

**The defect, and it is a real one.** `Pseudocosts::default_cost`
(`branching.rs:108`) is a hardcoded `1.0` in **objective units**, and
`score()` (`branching.rs:179-184`) multiplies it against observed per-unit
pseudocosts — also in objective units — inside the same product:

```rust
let d = self.down_cost(var_index) * frac_part;      // observed, or the 1.0 default
let u = self.up_cost(var_index) * (1.0 - frac_part);
(1e-6 + d) * (1e-6 + u)
```

So on a model whose true pseudocosts sit near `1e-3`, every never-observed
variable scores ~1000× too high and wins every comparison; near `1e3` it loses
every one. **Branching order is a function of the objective's arbitrary scale.**
`tree_manager.rs:723-726` reads the same accessors for `best_estimate`, so
BestEstimate node ordering skews with it. Nothing here touches a bound —
pseudocosts select a *variable* — so this is soundness-free but search-changing.

**Entry experiment first (CLAUDE.md §4).** The defect only matters if unobserved
candidates actually compete at selection time, so that was measured before any
implementation, with the kill criterion committed to `profile.rs` *before* the
run. 38-instance panel, fixed 3000-node cap:

| pre-registered kill threshold | measured | |
|---|---|---|
| unobserved < 10 % of scored candidates | **39.90 %** (675,255 / 1,692,319) | 4× over |
| unobserved winners < 5 % of decisions | **17.40 %** (7,206 / 41,406) | 3.5× over |
| — (A4 secondary) observed costs within one decade of 1.0 | **26.6 %**, spread > 4 decades | |

The constant is not a rarely-touched fallback: it decides roughly **one
branching in six**. This is compounded by the A4 finding that `sb_active`
(`milp_driver.rs:1434`) gates strong branching off after `sb_node_budget` nodes
(48 in the Python binding), so pseudocosts steer essentially the whole tree with
no mechanism ever initializing them below the root region.

**The fix.** Score an unobserved variable at the running arithmetic mean of every
per-unit pseudocost observed so far, across all variables — so it ranks as *an
average variable of this model* rather than *a variable worth 1.0 objective
units*. Zero-gain observations are included in that mean (a branch that gained
nothing is evidence about the scale, not a missing sample). Before any
observation, and when every observation so far is exactly zero, it falls back to
the legacy constant: no scale is knowable there, and all unobserved candidates
tie under either value anyway. Behind `DISCOPT_PCOST_DEFAULT` (`one` = legacy,
`mean` = new), default-off, unrecognized values a hard refusal — which doubles as
the differential panel's wiring gate.

**The differential panel says do not graduate.** 38 instances × 2 arms, 3000-node
cap, order alternated per instance, 120,825 feasibility conditions checked on
incumbents:

| | `one` (legacy) | `mean` (scale-free) |
|---|---|---|
| solved to optimality | 11/38 | **12**/38 |
| geomean dual gap | 0.00158169 | **0.00141174** |
| geomean primal gap | 0.00410782 | **0.00259286** |
| total nodes | 84,900 | 83,966 |
| dual bound better on | — | 8 instances (worse on 7) |
| primal gap better on | — | 2 instances (worse on 1) |

**Cert-clean: yes**, with no slack — no bound above its reference optimum in
either arm, every incumbent independently feasibility-verified against the
*source* MPS rows (not the engine's standard form), and no instance optimal under
`one` that is non-optimal under `mean`. The arms genuinely diverged on 27/38, so
this is not a null instrument.

**Net-positive: no.** The pre-registered bar was ≥ 3 net-better instances on the
dual *or* primal side with neither geomean worsening. Both geomeans improved and
one more instance solved, but the per-instance margin is **1 on each side** —
8-better/7-worse is a coin flip. Worse, the dual-side gains are concentrated in
one family: all four `mik-250-20-75-*` instances improve, which is exactly the
benefit-confined-to-a-class pattern CLAUDE.md §2 rejects. This is the
`DISCOPT_CUT_INHERIT` case again: **sound is not the bar; broadly helpful is.**
The flag ships default-off with this measurement recorded and does not graduate.

**Retraction, CLAUDE.md §11 — this section's first version named the wrong next
step, and it was wrong within the hour.** It claimed that "HiGHS and SCIP do not
guess: they strong-branch the unreliable candidates, and discopt cannot"; that
A4's unlimited-budget sweep "swept SB on *all* candidates at every node"; and
that reliability branching was therefore "a different experiment from A4's".
**All three are false.** `strong_branch` already filters to unreliable
candidates —

```rust
.filter(|c| c.2 < ctx.reliability)   // milp_driver.rs:2678; c.2 is the obs count
```

— and `MilpOptions::strong_branch` is documented at `milp_driver.rs:408` as
"Limited strong branching on unreliable candidates (**reliability branching**)",
feedback loop included. discopt *is* a reliability brancher. A4's `unlimited` arm
therefore already ran reliability branching at full depth and measured **1.221×**,
below even that probe's 1.25× falsification floor. There is no A8 of the kind
described; the claim is withdrawn rather than carried forward.

**What A7 actually leaves standing.** A6's 39.9 % unobserved figure was measured
at the *shipped* `sb_node_budget = 48`, so it is a description of the tree once
reliability branching has expired — not evidence that the mechanism is missing.
And A4 already established, on this same panel, that turning it back on
everywhere is worth only 1.22× in geomean while being **bimodal**: 11.31× on
`fiber`, 3.25× on `dcmulti`, *exactly* 1.00× on ten instances, and actively
**worse** on `p0201`. The lever is not more SB or a better guess between probes —
it is spending SB where the probe can discriminate and not where it cannot. That
is the degeneracy shut-off A4 already registered as its live candidate
(`setMinReliable(0)` at a degenerate node, `HighsSearch.cpp:1114-1116`), with a
falsification test on this very panel: if a degeneracy signal separates `p0201`
from `fiber`, it is a mechanism; if it does not, it is dropped. That, not a
default-value refinement, is the next experiment.

### A14 — RENS is cert-clean and a huge primal win, but FAILS its own net-positive bar. 2026-09-06.

A5 measured the residual gap as primal and specifically the *improving* kind, and
A12 (RINS) attacks it from the incumbent side. RENS {cite:p}`Berthold2014`
attacks it from the side that needs no incumbent at all: fix every integer column
already integral in the **LP relaxation**, restrict each fractional one to
`{floor, ceil}`, solve the sub-MIP. That precondition is exactly what A5's 12
instances holding a merely-poor incumbent do not satisfy for RINS.

**Method.** `scratchpad/a14/rens_panel.py`, one build / two arms toggled by
`DISCOPT_RENS`, arms interleaved, 38-instance MIPLIB easy panel (all 38 with a
reference optimum), `gap_tol=1e-4`, TL 20 s, 2 reps — the same population and
settings A13's `gapsplit.py` used, so the numbers are comparable. The §8 load
gate is *functional*: the panel sets `DISCOPT_RENS="not-a-flag-value"` and
requires the extension to panic (`milp_driver.rs:976`), which a pre-A14 build
does not do. Machine was loaded throughout (`load average 14.33`, another
session holding a core), so **no wall-clock claim is made from this run**;
every bar is stated on bounds, incumbents and statuses.

**Bars, pre-registered in the script docstring before the run.** BAR 1
cert-clean (hard, zero slack). BAR 2 net-positive over the 18 instances not
`optimal` in both arms: (2a) ON median primal gap strictly lower; (2b) incumbent
wins > losses; (2c) dual-bound losses ≤ 20% of the population.

**Result.**

```
BAR 1  reference comparisons executed : 152
       ON incumbents feasibility-verified against the original model : 72
       PASS: 0 violations

BAR 2  population (not optimal in BOTH arms): 18
       median primal gap  OFF 38.74%   ON 1.64%
       incumbent          wins 12   losses 0
       dual bound         wins  1   losses 9   (cap: 3 of 18)

       2a median primal gap strictly better : PASS
       2b incumbent wins > losses           : PASS
       2c dual-bound losses <= 20%          : FAIL
```

The primal effect is the largest this plan has measured: the median primal gap
over the contested population falls **38.74% → 1.64%**, with 12 incumbent wins
and **zero** losses. `mik-250-20-75-2` goes 42.19% → 0.00%, `beavma` 54.94% →
0.29%, `neos17` 140.35% → 40.00%. Bar 1 is clean and non-vacuous: 152 executed
reference comparisons and 72 independently feasibility-verified ON incumbents.

**VERDICT: `DISCOPT_RENS` STAYS DEFAULT-OFF.** 2c failed, the pre-registered kill
criterion fires, and the flag does not graduate on a run that met two of three
bars. This is the `DISCOPT_CUT_INHERIT` outcome recorded a second time: sound and
even dramatically helpful on one axis is still not a graduation.

**Post-hoc, and labelled as such: 2c is partly measuring noise, and the panel
cannot tell how much.** Written after seeing the result, so it justifies a
re-run, not a re-reading of this one. `pick()` compares `rs[-1]` — a *single*
rep per arm — while the dual bound at a fixed time limit is itself a noisy
quantity. Comparing each instance's between-arm difference against the spread
between the two reps of the *same* arm:

| | count |
|---|---|
| dual-bound difference inside same-arm rep-to-rep spread | 5 |
| difference exceeding it | 5 |

`gsvm2rl3` moves 15.94% between two OFF reps while the OFF↔ON difference is
4.28%; `beavma`'s ON bound is *better* than OFF in rep0 and worse in rep1. Five
differences do exceed the noise, but four of those are 0.077–0.339%; only
`neos17` (10.02%) is both real and large. So the true dual-side cost is real but
much smaller than a raw count of 9 losses implies — and a zero-tolerance *count*
under a shared wall-clock budget conflates "RENS harmed the search" with "RENS
spent some of the 20 s in sub-MIPs".

**What that does NOT license.** It does not license graduating the flag. A bar
discovered to be badly specified *after* it fails is not thereby satisfied. The
correct next step is a re-run whose dual-side bar is pre-registered on
*magnitude and repetition* rather than a single-draw count — more reps, a
per-instance spread, and a threshold stated in relative terms — on an unloaded
machine. Until such a panel passes, A14 ships default-OFF with its `=0` opt-out
and the legacy path intact.

## 4. Success metric

The §0 panel at matched `mip_rel_gap = 1e-4`, TL = 20 s. "Competitive" is
solved-count within a few instances of HiGHS's with total wall within ~2×. Every
stage reports against this same table.

Current standing, matched tolerance, 190/190 runs, zero certification aborts:

| arm | solved | wall | med(solved) | nodes |
|---|---|---|---|---|
| base_16x1 | 19/38 | 430.41 s | 0.170 s | 8,853,388 |
| cut200_prune | 20/38 | 400.50 s | 0.130 s | 6,459,226 |
| **cut500_prune** (best) | **21**/38 | 399.97 s | 0.138 s | 4,693,240 |
| **highs** | **38**/38 | **71.95 s** | 1.039 s | **15,417** |

Distance to go: **17 instances, ~5.6× wall, ~300× nodes.** Intermediate targets,
so a stage can be scored before parity is reached:

| stage | target | basis for the target |
|---|---|---|
| A0 + A0.1 | 21/38 | **MET, 2026-09-05 — graduated default-ON.** 18-19/38 → **21/38** at −43 % nodes, dual gap on the 17 open instances 0.2028 → 0.1494; cert-clean over 304 bound-vs-optimum comparisons across two independent runs, zero violations, three instances feasible → optimal and none the reverse. Shipped at `_milp_root_cut_budget`'s configuration, not `cut500_prune`'s — see Track A0 |
| A0′.1 | 21/38 | **MET, 2026-09-05.** No solved-count target — a correctness repair scored on counters and LP work. Delivered: `DualPrepRejectShape` 300 → 0 on amur, `RootCutsGenerated` 0 → 102, and **−37.5 % simplex iterations** panel-wide at unchanged nodes and a cert-clean panel. A0′.2 was falsified and demoted — see Track A0′ |
| A1 | 24/38 | cut lifecycle makes a >500 budget affordable |
| A2 | 30/38 | the family HiGHS actually closes its last few percent with |
| A3 | — | branching quality; scored on nodes at fixed solved-count, not on solves |
| B1 + B3 | 33/38 | node selection + restarts, both cheap and both measured to matter |
| B2 | 35/38 | the sub-MIP/pump ladder |

These are targets, not predictions — a stage that misses its target is re-scoped
against the measurement, per §4 of CLAUDE.md, not carried forward on hope. A0′.1
deliberately claims no solved-count: it is first in the order because it unblocks
three mechanisms at once, not because it is expected to move the panel by itself.

*Caveat on every wall column here:* machine load was 66-69 during the 2026-09-05
panel. Node counts are load-independent and carry the conclusions; the wall
figures need one quiet re-run before they are quoted outside this document.

## 5. Licensing note for the owner

discopt is **EPL-2.0**. HiGHS is MIT (compatible inbound). SCIP 10 is Apache-2.0,
which can be combined but not relicensed. Both reviews were instructed to
describe algorithms, not paste source, and neither returned code.

**Decided 2026-09-05; the standing policy is `docs/dev/reference-provenance.md`.**
Read for the idea, implement independently from the published paper, cite and
acknowledge the source. So Stage 2 builds cMIR from {cite:t}`Marchand2001` and
lifted covers from {cite:t}`Gu1998` / {cite:t}`Gu1999`, using
`SCIPcutGenerationHeuristicCMIR` and `HighsTransformedLp::transform` only to
settle a specific ambiguity the papers leave open -- not as the thing being
transcribed -- and names the file that informed it in the module header.

One caveat that policy file states and that this note previously implied the
wrong way round: **being in Rust is not itself the protection.** A translation
into another language is still a derivative work. What protects an independent
reimplementation is that algorithms and mathematical methods are not
copyrightable, only their expression is -- so the safety comes from working from
the paper, which is a practice that can be skipped, not from the change of
language, which is automatic. The distinction decides which step is load-bearing.

What still comes to the owner is narrower than this note used to say: a routine
with **no published description**, where the reference implementation is the
specification.

## 6. Reproduction

`scratchpad/miplib/` holds `loader.py` (MPS → engine arrays via highspy, so both
solvers see byte-identical matrices), `hs.py` (the shared HiGHS arm), `screen.py`
(the fidelity + tractability gate that built `panel.json`), `miplib_ab.py` (the
§0 arm panel), `attribute.py` (the §1c ablation), `switches_ab.py` (Stage 0) and
`pathology.py` (the `khb05250` 3-nodes-in-20 s probe).
