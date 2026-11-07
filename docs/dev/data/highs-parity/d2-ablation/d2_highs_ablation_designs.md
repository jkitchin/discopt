## C. Port designs for the top components (independent implementations against discopt's data structures)

Each design states what HiGHS does (§A), what discopt has (§B), the gap, the plan, and a
CLAUDE.md §4/§5 entry experiment with a kill criterion. Every one ships behind a default-OFF
`DISCOPT_*` flag with a row in `docs/dev/flag-retirement-audit.md`, and graduates only on a
P4-standard panel (MIPLIB-38 + M, 3 reps, 20 s, interleaved) that is **cert-clean** (`incorrect
= 0`, no bound above its oracle, no `gap_certified` regression, incumbents re-verified) **and
net-positive** (certified ≥ OFF, no incumbent lost on any row, nodes and primal gap not worse
on the both-certified rows, both-certified wall ≤ 1.10× OFF). Nothing below copies HiGHS code;
the behaviour is described and mapped onto discopt's CSC standard form, `TreeManager` batches
and the existing separators.

### C.1 Tableau-row cut generation: lifted cover + c-MIR on aggregated tableau rows, separated at nodes, with LP/pool aging  (accounts for 10 of the 12 plain-B&B failures; necessary on `b-ball`, `neos-911970`; 10–27× nodes on 3 more)

**HiGHS** (§A.1 step 6, §A.4). The separator that does the work is `generateCut` on the
**aggregated tableau row** of every fractional basic integer (and fractional integral row),
both signs, ranked by `f(1−f)/‖B⁻¹ row‖²`. The row is transformed to the bound-substituted,
complemented space (variable bounds included), then a *lifted cover* inequality is tried
(knapsack / mixed-binary / mixed-integer variants by row type) and a *c-MIR* over a small delta
set ({1, the |a_j| of integer columns at nonzero value, the integral scale}, f0 ∈ [0.005, 0.995]);
the more efficacious survives postprocessing (integral scaling, tiny-coefficient removal into the
RHS, violation > 10·tol, coefficient tightening against the box). There is no GMI. The same round
runs at every node taken from the queue; LP rows leave after 10 non-binding epochs, pool rows
after 30 unused agings, and violated pool rows are re-added. Measured here: `pool_soft_1` (stops
the tableau separator and nothing else) fails `b-ball` (164 k nodes, 14.9 %) and `neos-911970`;
`lp_age_0` (aging only) is harmless; `node_cuts_off` fails `neos-911970` and costs 10–14× on
`mik-1`/`mik-5`; `keep_cuts` alone turns plain B&B from 13/25 to 23/25 (`mik` 2–4.6 k nodes,
`b-ball` 7.6 k, `sp150x300d` 114, `istra` 4.8 k, `iskar` 6.0 k, `kaihu` 1.5 k).

**discopt** (§B). Root GMI from refined tableau rows (`gomory.rs`, dense after P4.4's repair),
c-MIR and covers from the *original* rows only (`cmir.rs`, `cover.rs`), node covers only, no aging,
no pool re-separation; every root cut row stays in every node LP for the whole search.

**Gap.** (i) No c-MIR / lifted cover on **tableau** rows — HiGHS's tableau separator is "GMI's
row with c-MIR's rounding and a cover lift", and it is what the analysis doc's L5 deferred;
(ii) no tableau/c-MIR separation at nodes; (iii) no aging, so every root cut is paid for at every
node (the per-node cost population B of the analysis doc).

**Plan** (`crates/discopt-core/src/lp/tableau_cuts.rs`, wired at the two existing separation
points):
1. *Row supply.* Reuse `gomory.rs`'s refined `B⁻ᵀe_i` rows (iterative refinement already there) for
   the basic integers with fractionality in `[1000·tol, 1 − 1000·tol]`; rank by `f(1−f)/‖row‖²`,
   drop rows whose scaled weights span > 1e4 (HiGHS's numerics gate), cap at `200 + 0.1·min(m,
   n_int)` rows per round, and hand each row **and its negation** on.
2. *Cut core.* Extend `cmir.rs`'s bound-substitution/complementation/MIR core (it already does
   variable-bound substitution and the delta set) with (a) HiGHS's delta set and f0 window, and
   (b) a lifted-cover branch: on a row whose integer part is binary and whose continuous part
   is bounded, build the cover from the LP point (largest `x*` first, minimal), lift the
   non-cover binaries sequence-independently (superadditive lifting function of the cover
   weights), and keep cover vs c-MIR by efficacy. Mixed-integer rows go to c-MIR only in the
   first increment.
3. *Postprocessing as HiGHS:* scale to integral coefficients when a scale ≤ 1e4 exists, move
   coefficients < 1e-9·max into the RHS using the column bound (never drop them silently —
   #1236's lesson in `gomory.rs`), require violation > 10·tol, coefficient-tighten against the box,
   then `cut_select.rs` (efficacy + parallelism). Every cut is a combination of original rows
   with real multipliers, so the soundness argument of `cmir.rs` §Soundness carries over
   unchanged; the feasible-point sampling test of §5 is the regression guard.
4. *Node separation.* Run the separator at the first fractional node of each batch, in the slot
   `separate_cover_csc` uses (`milp_driver.rs:3783`), under `node_cut_cap`; fold at the batch
   boundary like node covers.
5. *Aging.* Per cut row an epoch age: a row basic at `lp_age_limit` (10) consecutive batch
   boundaries leaves the node LP (`rebuild_csc_with_cuts` already rebuilds the CSC at boundaries)
   and is parked in the pool; the pool is re-separated at each boundary (violation > tol,
   efficacy-sorted, parallelism-filtered); rows unused for `pool_age_limit` (30) boundaries are
   deleted. The dense P4.4 GMI rows get the same treatment, which is the analysis doc's L2
   density gate without a new gate.
Flags: `DISCOPT_MILP_TABLEAU_CUTS=1` (steps 1–3, root), `DISCOPT_MILP_NODE_TABLEAU=1` (4),
`DISCOPT_MILP_CUT_AGING=1` (5), each with its audit row.

**Entry experiment** (root only, before 4–5): `p44_root_gap.py --all` with the separator ON vs
OFF on the 25 instances here. Bars, against Table 3's HiGHS root with restarts disabled
(`root+restart_off`, i.e. what cuts alone buy HiGHS): `b-ball` root gap ≤ 6 % (HiGHS 4.74 %),
`neos-911970` ≤ 6 % (4.47 %), `fiber` ≤ 3.5 % (2.47 %), `bppc8-02` ≤ 3 % (2.37 %), `neos17` ≤ 2 %
(1.29 %); no root bound above its oracle; feasible-point sampling (the panel's incumbents) cuts
off nothing. **Kill:** `b-ball` or `neos-911970` root gap not within 2 pp of HiGHS's cuts-only
root, or any invalid cut → retire `tableau_cuts.rs` (keep the lifted-cover core if it is
sound, as `cover.rs`'s lifting). Steps 4–5 graduate separately: aging must be exactly
node-count-neutral on the P6.6-certified rows with `lp_age_limit = ∞` (identity test) before
its default is measured on wall; node separation's bar is `neos-911970` and `mik-1`/`mik-5`
certifying with ≤ 0.25× HEAD's nodes.

### C.2 Presolve + restart with a root reduced-cost ladder  (accounts for 9 of 12; necessary for 13–567× on 6 instances, the restart alone for 78–120× on 2)

**HiGHS** (§A.1 steps 1, 5, 8; §A.2). A full dimension-reducing presolve with MIP-only rules
(probing into a clique table and implication graph, coefficient strengthening, dominated
columns, parallel rows/columns, doubleton equations, aggregation, sparsify, dependent
equations, clique substitution); root reduced costs kept as a per-column ladder of
(cutoff → bound) pairs after every separation round, re-applied globally at every incumbent
improvement; when the inactive-integer fraction crosses a threshold (any on the first pass,
2.5 % later, 10 % after the root heuristics, or an in-tree tree-size vote) the model is
re-presolved with the cuts folded in and the root re-evaluated. Measured: on the `mik` family the
root bound is 1.0–1.6 % only with restarts (13.9–16.4 % = the LP gap without, Table 3), 3–4
restarts per instance; `restart_off` 120× on `beavma`, 78× on `bppc8-02`; `presolve_off` 13–567×
on 6 instances; no single presolve rule carries it (Table 4) except doubleton equations on
`bppc8-02` (152×); `keep_presolve` alone rescues 9 of the 12 plain-B&B failures.

**discopt** (§B). The engine's only presolve is FBBT on the marshaled rows (`root_box`); the
Python orchestrator (elimination, FBBT, probing; cliques off) runs once on the model and hands
over bounds; reduced-cost fixing is per node and local to the subtree, no ladder; no restart in
any form (the route's re-entry re-runs the *same* model with a seeded incumbent).

**Gap.** (i) Root reduced costs are not kept, so an incumbent found later never fixes root columns
globally; (ii) nothing re-presolves after fixings; (iii) the engine has no dimension-reducing
presolve on its own rows (no parallel/dominated/doubleton/aggregation, no coefficient
strengthening, no clique substitution).

**Plan** (two increments, the first cheap):
1. *Lurking ladder* (`bnb/redcost_ladder.rs`). After each root cut round that ends `Optimal`, for
   every nonbasic integer column `j` with `|d_j|` above the #1409 deflated tolerance, store
   `(cutoff_k, bound_k)` for `k = 1, 2, 4, …, 2^10` steps from the nonbasic bound
   (`cutoff_k = z_root + |d_j|·(k − 10·tol)`), dropping entries dominated by a tighter bound at a
   lower cutoff. On every incumbent improvement (`inject_incumbent`), apply every entry with
   `cutoff ≥ incumbent − tol` to the **global** root box, FBBT once, and push the box into the
   batch's `tightened` merge, which children already consume. Sound: it is the inequality
   `reduced_cost_fix` already argues, evaluated at the root with the root's safe bound.
2. *Restart* (`bnb/restart.rs`). Count inactive integers (fixed by the ladder, root FBBT or root
   reduced-cost fixing). When ≥ 10 % after the root heuristics or ≥ 2.5 % after the cut loop (any
   on the first pass): rebuild the marshaled problem with fixed columns removed (column deletion
   with RHS adjustment — the only reduction the first increment needs), carry the kept root cuts
   as rows and the root basis, re-run FBBT and the cut loop on the smaller problem, and restart
   the tree from an empty frontier with the incumbent kept; at most 3 restarts (HiGHS's count on
   `mik`). The route's slice/re-entry already restarts trees with a seeded incumbent; the new
   part is the reduced model and the trigger.
3. *Dimension-reducing presolve on the rows* (the analysis doc's L7, last): doubleton equations
   first (the one rule phase 3 singled out), then parallel rows/columns, dominated columns and
   coefficient strengthening with a postsolve stack — port the orchestrator's
   `substitute.rs` / `aggregate.rs` / `coefficient_strengthening.rs` to the CSC standard form
   rather than writing new reductions. Only after 1–2 are measured.
Flags: `DISCOPT_MILP_RC_LADDER=1`, `DISCOPT_MILP_RESTART=1`, later `DISCOPT_MILP_ROW_PRESOLVE=1`.

**Entry experiment** for 1+2 (no presolve port): `beavma`, `bppc8-02`, the four `mik`, `gesa2`,
`iskar`, `issue-2446`, `sp150x300d` at 20 s on the direct driver (`d2_direct_driver_panel.py`),
2 reps. Bars: the ladder fixes ≥ 2.5 % of the integer columns on ≥ 6 of the 10 rows (the
restart trigger), the restarted `mik` root bound closes ≥ 2/3 of the gap between the LP root
and HiGHS's 1.0–1.6 % root (i.e. ≤ 7 % residual from the 16–19 % LP gap), and nodes to certify
on `beavma`/`bppc8-02` ≤ 0.25× HEAD. **Kill:** the ladder fixes < 2 % on every row (then the
restart has nothing to act on and the lever is C.1's cuts), any bound above its oracle (a
soundness bug — stop), or node counts not better on ≥ 4 of the 10 rows → retire both flags and
their code.

### C.3 One budgeted sub-MIP neighbourhood heuristic with a propagating fixing dive  (0 certification losses for HiGHS at 20 s, but ≥ 10× nodes on 11 instances, up to 447×; the four HiGHS heuristics are redundant with each other)

**HiGHS** (§A.3). One machinery, three neighbourhoods. A *dive* on a private copy of the LP
relaxation fixes integers one at a time with domain propagation after each fix (conflict analysis
on failure), re-solving the LP as it goes, until a target fixing rate; what remains is a
**budgeted full sub-MIP** (500 leaves, 200 + nodes/20 nodes, 12 stall nodes, objective cutoff =
incumbent, cliques/implications/pseudo-costs inherited). The fixing rate adapts from observed
successes and infeasibilities. RENS fixes the LP-integral columns first (no incumbent needed),
RINS the LP/incumbent agreement set, `rootReducedCost` the root reduced-cost ladder. They run at
the root and at every plunge start while the heuristic LP-iteration budget allows (10 k free, then
5 % of the tree's LP iterations, front-loaded). Phase 3: removing any one of FJ / RINS / RENS /
root-reduced-cost leaves every instance at its base node count except RENS on `bppc8-02` (104×)
and `mik-5` (18×); removing all of them costs 12–447× nodes on 11 instances.

**discopt** (§B). `try_rins` → `run_sub_mip` exists (agreement neighbourhood pinned by bounds,
recursive driver call with a node budget and ≤ 2 s, strict improvement, re-validated) and is
**default-OFF**; there is no propagation in the neighbourhood construction, no RENS (retired
P4.2), no root-reduced-cost neighbourhood, no adaptive fixing rate; the dive heuristic is an
LP-only most-fractional dive without propagation, cutoff or budget, run once at the root and
only while no incumbent exists. The consequence is the analysis doc's frozen primal (12/12
rows hold the same incumbent at 20 s and 60 s).

**Gap.** (i) No propagation-driven fixing dive (what makes HiGHS's neighbourhoods feasible and
small enough); (ii) no budgeted *tree* schedule; (iii) no RENS / root-reduced-cost
neighbourhoods; (iv) no adaptive fixing rate.

**Plan** (`bnb/sub_mip.rs`, replacing `try_rins`'s neighbourhood builder, keeping `run_sub_mip`):
1. *Fixing dive with FBBT.* A `Neighbourhood` builder takes the batch's node point, the incumbent
   (optional), the ladder of C.2 (optional) and a target fixing rate; it fixes columns in HiGHS's
   order (RINS: agreement set first; RENS: LP-integral first; then fractional ones towards the
   root-direction/objective rounding rule), running `tighten_bounds_csc_with_row` on the sub-box
   after each **block** of `max(1, n_int/50)` fixings (FBBT sweeps all rows, so per-fix
   propagation is too dear); on an infeasible box undo the block and halve the block size (the
   cheap stand-in for conflict analysis). Stop at the target rate, counted on integer columns.
2. *Sub-MIP call.* Add `max_leaves` and `max_stall_nodes` to `MilpOptions` (sub-solves only),
   pass the incumbent as the cutoff (the sub-driver already seeds `initial_incumbent`), keep the
   recursion guard and the re-validation through `validate_seed_incumbent`.
3. *Adaptive fixing rate.* Running means of successful and infeasible rates per kind; draw the
   target uniformly in HiGHS's band, start 0.6; an infeasible sub-MIP at ≤ 1 node halves the
   target and retries once from half the dive depth.
4. *Schedule.* Replace the `!has_incumbent` gate of `dive_batch_eligible` by an LP-pivot budget
   (heuristic pivots ≤ 0.05 × tree pivots + 10 000): RENS while no incumbent, RINS once there is
   one, at the first node of each batch inside the budget; the root-reduced-cost neighbourhood
   once at the root when C.2's ladder exists.
Flag `DISCOPT_MILP_SUBMIP=1` (retires `DISCOPT_RINS` into it, closing its missing audit row).

**Entry experiment** (root-only RENS + RINS, steps 1–3, before the schedule): the 11 instances of
Table 1's heuristics column plus `istra`, `iskar`, `2122`, 20 s direct driver, 2 reps: ≥ 6
instances strictly improve the P6.6 incumbent and ≥ 3 reach the optimum as incumbent (`gt2`,
`iskar`, `kaihu`, `jijia`, `bppc8-02` are the A12/L3a candidates). **Kill:** < 4 improved, any
P6.6-certified row loses its certification, or sub-MIP time > 30 % of the slice on the
both-certified rows → retire the builder (keep `run_sub_mip`). Graduation adds `dcmulti`
certified (the P4.2 RENS failure mode) as an explicit bar.

### C.4 What the ablation could not split: the base machinery (13 of 25 certified by plain HiGHS B&B)

Not a port design — a statement of what §0 item 4 contains, so it is not mistaken for "the LP".
Of the pieces still active in `all_off`, the evidence here points first at **(a) propagation with
implications/cliques plus conflict analysis** (on `gt2`, `fiber`, `nexp-*`, `p0548`, `beavma`
every improving `all_off` incumbent after the first came from node evaluation `T` — the plunge
lands on integral points that Rust's FBBT-only propagation does not reach; the base runs show
183 conflicts on `b-ball`, 9 706 on `neos-911970`) and **(b) the root reduced-cost ladder applied
globally at every incumbent (§A.2, part of C.2 step 1)**; then **(c) randomized rounding with
propagation and an LP re-solve of the continuous part** (`R` is the first incumbent on 6/25 base
runs and FJ `J` on 15/25 — but `fj_off` leaves every tested node count unchanged, so the first
point is not what is load-bearing; what `R` adds is a propagated, LP-repaired rounding at every
dive start while no incumbent exists). Node selection (plunging) and pseudo-costs are the analysis doc's binding
negatives and are not re-proposed. The right experiment for (a) is not an option ablation but a
Rust one: C.2 step 1 plus a clique-table propagator (`presolve/cliques.rs` edges → a
`clique_propagate` pass inside `tighten_bounds_csc_with_row`, bound-neutral by construction)
measured on `gt2`'s 70 k-node tree: bar ≤ 0.25× nodes with the same bound; kill: no change on
`gt2`, `nexp-50-20-1-1`, `fiber`.
