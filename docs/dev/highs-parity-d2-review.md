# D2 review — gap attribution, re-prioritised plan, and the gate (2026-11-08)

Read-only review of branch `claude/happy-hamilton-0vvo3b` at `bee246f8` (PR #2196). Nothing
in the tree was edited; no solver was run. Every number is from a checked-in file (path given)
or a grep of `ref/HiGHS` (HiGHS 1.15.1, `Version.txt`). Analysis scripts and their output are
in this directory: `attrib.py` / `attrib.out` (per-instance attribution, 186 ON solves read)
and `history.py` / `history.out` (certification per instance across the 16 checked-in §5
panels, 2,976 rows read).

**On the HiGHS expert.** This session exposes no Agent tool (ToolSearch for `Agent`/`Task`
returned nothing), so the sub-agent consult the brief asks for could not be spawned. Instead,
every HiGHS mechanism this review ranks was checked by me directly against `ref/HiGHS` at the
cited `file:line` (restart trigger and `performRestart`, root reduced-cost ladder, plunge and
objective limit, per-queue-node separation and aging, tableau/path separators, c-MIR core,
propagation queue, sub-MIP heuristics). Claims about HiGHS that rest only on the earlier
consults (`highs-parity-d2-*-expert.md`) and were not re-verified are marked *(consult)*.

## 0. Where the gap actually is today

The brief's "Rust 16/38" is the P4.9 figure. The latest panel whose ON arm is today's default
(`d2-analysis/tol_robust_panel_v2_r3.json.gz`, build `5ab7b014`, `TOL_ROBUST` graduated ON;
every other graduated D2 lever ON in both arms; no engine change since except the P6.3
packaging) gives, 3 reps, 20 s (`attrib.out`):

| | certified (all 3 reps) | HiGHS (p6.6) | total median wall |
|---|---|---|---|
| MIPLIB-38 | **22/38** | 35/38 | 398 s vs 189 s (2.1×; 16 time-outs = 320 s of the 398) |
| M-24 | 19/24 (+ `p0548` 2/3) | 24/24 | 109 s vs 19 s |
| 62-panel solves | 125/186 | ~177/186 | |

The 16 MIPLIB rows that never certify: `beavma`, `bppc8-02`, `enlight8`, `enlight_hard`,
`gsvm2rl3`, `mik-250-20-75-{1,2,3,5}`, `neos-2624317-amur`, `neos-3118745-obra`,
`neos-911970`, `neos17`, `nexp-50-20-{1-1,4-2}`, `sp150x300d`. HiGHS fails three of them too
(`enlight_hard`, `amur`, `obra`), so **34/38 means flipping 12 of the 13 rows HiGHS certifies
and Rust does not.** M adds `2122`, `gesa2`, `issue-2446`, `p0548`(2/3).

The plan's §0 group table (`highs-parity-d2-plan.md:20-29`) is stale: `b-ball`, `fiber`,
`istra`, `kaihu`, `iskar`, `jijia`, `gt2`, `khb05250`, `neos-1425699` are all 3/3 now
(W1's `CUT_LP`+`SB_ARMED` bundle did that: `history.out`, panel totals 96 → 125).

## 1. Gap attribution (by cause, with the instances and the evidence)

**C1 — a certification regression that graduated: `DISCOPT_MILP_CUT_LP` lost `enlight8`
(and `bppc8-02`'s bound).** `history.out`: `enlight8` is 3/3 in every panel through the
aligned-slacks+sub-MIP bundle (certified in 11–14 s, 45–54k nodes), then **3/0 in both W1
panels** (`w1_cut_lp_panel_r3`, `w1_cut_lp_sb_armed_panel_r3`) and 0/0 ever since. ON: bound
stuck at 20 (optimum 27, incumbent 27 in hand), 9.5–11k nodes in 20 s, `CutLpFolded` 1,390.
`bppc8-02`: OFF bound 495–504.5 at 17–26k nodes, ON bound 486.5 (= the root LP) at 2k nodes,
`CutLpFolded` 282. Both were counted as W1's "2 dual losses (cap 2)" and bar 2a passed because
it **sums** certifications (`d2_flag_panel.py:18`, `:120` — `solved["on"] >= solved["off"]`,
117 → 125). CLAUDE.md §5's cert-clean bar, however, reads "no certification regression (no
`gap_certified=True` instance regresses to uncertified)". Under that wording the CUT_LP
graduation did not meet bar 1. The mechanism (hypothesis, from the code): CUT_LP drops the
node-local augmented re-solve and folds pool cuts into the *shared* batch LP by HiGHS's pool
score (`milp_driver.rs:3089-3103`, `:753-800`); HiGHS has one persistent LP **per worker**
and separates at every queue-popped node into *that* LP immediately
(`HighsMipSolver.cpp:684-693`, `HighsSeparation.cpp:162-166`). On `enlight8` 1,390 rows were
folded and the bound never left the root LP, i.e. the rows a node needs are not the rows the
batch majority admits, and the per-node lift that certified it before W1 is gone.
*Explains:* `enlight8` (+1 MIPLIB, certain once the pre-W1 behaviour is restored), `bppc8-02`'s
bound (not its certification: incumbent 513 vs 507).

**C2 — no in-tree c-MIR-class separation (the dual-only rows).** `mik-250-20-75-{1,2,3,5}`:
incumbent optimal (primal gap 0–0.3 %), dual gap 2.8–3.3 %, 15–18k nodes in 20 s
(`attrib.out`). HiGHS evidence (`highs-parity-d2-highs-ablation.md` Tables 2 and 4): plain B&B
fails all four (26–28k nodes, 1–2 %); `keep_cuts` alone certifies them in 2.1–4.6k nodes;
`node_cuts_off` costs `mik-1`/`mik-5` 10–14×; `pool_soft_1` (tableau separator off, everything
else on) is harmless on `mik-1` (1,163 vs 1,099 nodes) — so the in-tree family that matters is
the **path/aggregated c-MIR class, not the tableau separator**. SCIP agrees
(`d2-analysis/w4_scip_mik_ablation.jsonl`: Gomory off, aggregation only → `mik-1` certified in
8.2 s). What Rust has in the tree is GMI only (node GMI pool), which moved `mik` bounds by 35–70
units of a ~1,800-unit gap. **W4 as planned ("separate aggregated c-MIR at nodes into the same
global aging pool", plan W4 "Change") was never built:** commit `a9bc8d7d` changed `lp/cmir.rs`
and the route's *root* loop only (`git show a9bc8d7d --stat`), the panels that failed were root
adoption, and `e5eac131` deleted it. B1 (`DISCOPT_MILP_NODE_SEP`, retired) did separate c-MIR
at nodes, but with the pre-W4 separator (depth 2, first-8-by-index deltas, no flip, 128-row cap,
matrix fold). *Explains:* the four `mik` (+ `gesa2` and `bppc8-02`'s bound plausibly: the root
W4 bundle certified `gesa2` 3/3 in both of its panels, `history.out`).

**C3 — root closure by path-class cuts (both sides far off).** `nexp-50-20-1-1` (primal 41 %,
dual 14 %), `nexp-50-20-4-2` (7 % / 4 %), `sp150x300d` (29 % / 10 %), `beavma` (34 % / 8 %).
HiGHS closes all four at the root (Table 3, 1 node). On `nexp` the root closes with
`cuts_throttle` *and* with `presolve_off` (0.00 both), so neither the tableau separator nor
presolve is needed there: the non-tableau separators (path c-MIR, implied-bound, clique, mod-k,
`HighsSeparation.cpp:38-40`, `HighsImplications.cpp:583`, `HighsCliqueTable.cpp:1605`) do it;
HiGHS `all_off` root 2.84 %. On `sp150x300d` the no-presolve root is the LP gap (92.9 %) and
the instance certifies in 114 nodes of in-tree separation, or in 59 with presolve (variable
upper bounds). Rust roots after aligned slacks (analysis follow-up 19): `nexp-1-1` 20 %,
`nexp-4-2` 9.7 %, `sp150x300d` 10 %, `beavma` 17.9 %; P4.4 saw Rust c-MIR saturate at 38.6 % on
`sp150x300d`. `cmir.rs` aggregates to depth 2 from rows within 10 % slack
(`highs-parity-d2-phaseB-expert.md` §3) against HiGHS's path length 6 from any start
(`HighsPathSeparator.cpp:170`, `constexpr HighsInt maxPathLen = 6`). The B5 entry (two constants)
was never run. *Explains:* the root half of `nexp`×2, `sp150x300d`, `beavma`; `neos17` partly
(HiGHS root 0.63 % vs Rust dual 4.9 %).

**C4 — base machinery HiGHS cannot switch off: restart with re-presolve, the root
reduced-cost ladder, clique propagation.** `bppc8-02` (HiGHS `all_off` 1,753 nodes; `presolve`
247×, doubleton 152×, RENS 104×, restart 78×), `beavma` (restart 120×), `gesa2` (`all_off`
8,525 nodes in 19 s; heuristics 81×), `neos17` (`all_off` 8,746), `gsvm2rl3` (HiGHS itself needs
21.5 s unloaded, Table 5). Table 4's `presolve_reduce_only` arm (probing, dual fixing,
parallel rows/cols, enumeration, col-stuffing off; `d2_highs_ablation.py:88-95`) stays within
2.2× of base on every row it was run on, so the restart's value does not need the MIP-only
rules. Rust has no restart, no ladder, no clique table (`grep -i restart|ladder
milp_driver.rs`: nothing). HiGHS: `percentageInactiveIntegers`
(`HighsMipSolverData.cpp:1310`), restart at ≥ 10 % after the root heuristics (`:2129-2134`)
and at ≥ 2.5 % (or any fixing on the first pass) after separation (`:2494-2500`),
`performRestart` folds the cut rows into the model and re-presolves (`:1317-1335`,
`appendCutsToModel`); ladder `HighsRedcostFixing.cpp:195` (store), `:37` (apply on every
incumbent). Follow-up 20 ("the Rust route on HiGHS-presolved models fixes only `p0548`")
measured presolve of the *original* model; it says nothing about a restart after root fixings,
which is a different mechanism. P4.5's unseeded probe found 0/57 roots ≥ 10 % fixed with *no
incumbent*; the sub-MIP now lands one on most rows within the first batches.

**C5 — LP engine.** `issue-2446` (M): 1 node in 20 s, the root LP never finishes
(120 µs/pivot, dense `PriceSweep`; HiGHS 973 pivots at 9 µs). `neos-911970`: 973 nodes
(up from 7), primal 90 %, dual 58 %; HiGHS needs tableau + node cuts + one more component
(1,370 nodes, 11–14 s). `bppc8-02`: 1,959 sub-tolerance repairs / 312k primal pivots in 20 s
(plan W5 record). `INCR_PRICING` (steps 1+2) addressed exactly this cost — nodes −16 %, 0 dual
losses — and was retired on `p0548` alone (see "instrument", below).

**C6 — no incumbent.** `2122` (bound at the optimum, 173 nodes, M only), `enlight_hard`,
`amur` (HiGHS fails both). Not gate-relevant on MIPLIB.

**The instrument.** `p0548` is bimodal in *both* arms of every panel since W1
(`history.out`: OFF/ON 0/1, 0/2, 3/3, 3/2, 2/1, 1/3, 2/0, 0/3, 2/3, 3/2). `INCR_PRICING` was
retired on "125 → 123, every loss `p0548`"; `TOL_ROBUST` v1/v2 swung 125 ↔ 126 on `p0548`;
`RAND_ROUND`/`EXPEL` "lost an incumbent" on it. A bar decided by one coin-flip row is
measuring the coin. `istra`/`kaihu` also flip 3/0 under some arms (W4 incr panel), so
trajectory sensitivity is not confined to `p0548`.

**Unknowns and the cheap measurement for each.** (a) Which CUT_LP element starves
`enlight8`: two counters (pool cuts violated at a node optimum but not in the shared LP;
folded rows deleted and later re-violated) on 3 solves of `enlight8`/`bppc8-02`. (b) Whether a
W4-quality c-MIR at nodes moves `mik` at a fixed node budget: needs R3's build; the zero-code
half (HiGHS `pool_soft_1` fine, `node_cuts_off` fatal) is already in hand. (c) The Rust root on
`nexp`/`sp150x300d`/`beavma` with path depth 6: the B5 const rebuild, minutes of compute.
(d) The fraction of integer columns the root ladder would fix with *today's* first incumbent:
re-run `p45_restart_probe.py`'s counter seeded by the sub-MIP's point, zero engine code.
(e) Whether `enlight8` OFF still certifies on today's build (its OFF arm last ran at `5ab7b014`:
yes, 0/3 in both arms there — so confirm the fix target is the pre-W1 path, not something later).

## 2. Prioritised plan

Common rules: every item is a default-OFF flag with an audit row; entry on the named rows
first, 20 s, ≥ 2 reps or a deterministic node budget; graduation on `d2_flag_panel.py` **after
R2's hardening** plus `flag_onoff_corpus_panel.py`. Certificates are never at risk from search
order or from globally valid cuts; the risk items are named per row.

| # | item | hypothesis | target rows (from data) | expected gain (basis) | cost | correctness risk | entry experiment → kill | graduation |
|---|---|---|---|---|---|---|---|---|
| **R1** | **Repair the CUT_LP regression** | the shared-LP fold removed the per-node pool lift; restoring it for nodes whose violated pool cuts are not in the shared LP (node-local augmented re-solve as fallback, or fold at the popped node as HiGHS does) recovers the pre-W1 bound trajectory | `enlight8`, `bppc8-02` (bound) | **+1 MIPLIB** (`enlight8` certified 3/3 at 11–14 s in every pre-W1 panel; `w1_cut_lp_panel_r3` OFF arm); `bppc8-02` bound 486.5 → ~504 | 1–2 days | none to certificates (pool cuts are globally valid, NS-verified; `delete_cut_rows` regression test exists); risk is losing W1's wins (`istra`/`iskar`/`kaihu`/`b-ball`/`fiber`), which the panel scores | `CUT_LP=0` vs `=1` on both rows, 3 reps, with the two new counters; then the fix ON vs OFF on the same rows + `istra`/`b-ball` → kill if `enlight8` is not 3/3 ON or `istra`/`b-ball` lose a rep | §5 panel with the hardened bar; corpus differential |
| **R2** | **Harden the panel instrument** (no solver code) | bar 2a's sum and single-draw rows hide regressions and manufacture kills | — | enables every later decision; retroactively re-reads two kills | ½ day | n/a | n/a | add bar "no instance 3/3 OFF → 0/3 ON" (CLAUDE.md §5 wording); rows bimodal in OFF across ≥ 2 panels get 6 reps or are scored separately from 2a/2d; keep the OFF-fired = 0 check but make it per-flag |
| **R3** | **In-tree c-MIR into the global pool (W4 as designed)** — recover the W4 separator (`a9bc8d7d`, `d31ad1d2`: HiGHS deltas, flip, lifted MIC, small-coef removal, power-of-two scaling) and run it at the k best-bound nodes per batch over original + folded rows aggregated along paths with **global** bounds, output into the existing aging pool / CUT_LP fold | HiGHS and SCIP certify `mik` with in-tree aggregated c-MIR; the tableau separator is not needed (`pool_soft_1`), node separation is (`node_cuts_off`); Rust's only in-tree family is GMI | `mik`×4; `gesa2`, `bppc8-02` bound | **+2–4 MIPLIB** if the entry holds (HiGHS `keep_cuts` 2–4.6k nodes vs Rust 15–18k/20 s; SCIP aggregation-only 8.2 s) | 3–5 days | cuts: validity by the `cmir.rs` sampling tests + feasible-point check against panel incumbents; the W4 LP-numerics failure (sub-tolerance repair pivots on `dcmulti`) must be prevented by the scaling/small-coef rules — a cert-clean but slower trajectory is the realistic risk | direct driver, `max_nodes=5000`, deterministic, ON vs OFF on `mik-1/-2/-3/-5`, `gesa2`, `bppc8-02`: dual gap at the budget must fall ≥ 1 pp on ≥ 3 of 4 `mik`; kill otherwise, or if `dcmulti`/`bell5`/`istra`/`kaihu` show the repair-pivot signature at 20 s | §5 panel (hardened), corpus differential; re-panel `IMPROVE_DIVE` with it as the owner asked (follow-up 8) |
| **R4** | **Path-aggregation c-MIR to HiGHS's design** (B5): depth 2 → 6, any start row, both scales (`HighsPathSeparator.cpp:170`) | the root gap on the fixed-charge/network rows is path-cut gap, not tableau or presolve gap (Table 3 `cuts_throttle`/`presolve_off` roots 0.00 on `nexp`) | `nexp-50-20-1-1`, `nexp-50-20-4-2`, `sp150x300d`, `beavma`; `neos17`, `neos-911970` root | **+0–3** (root half only; the primal half rides on the graduated sub-MIP) | hours for the const-rebuild entry; 2–4 days for a real separator | same as R3 (original-row aggregation, globally valid) | root bound, `max_nodes=2`, on the 6 rows vs today and vs HiGHS `all_off` root (2.8 / 1.3 / 92.9 / — / 6.2 / 7.4 %): kill if < 2 pp gain on ≥ 3 of 5 (B5's pre-registered bar) | as R3 |
| **R5** | **Root reduced-cost ladder + restart** (C.2 steps 1–2): store `(cutoff, bound)` per nonbasic integer after each root round, apply globally on every incumbent; when ≥ 10 % (first pass: any) integers are inactive, rebuild the marshaled problem with fixed columns removed and cuts folded, re-run FBBT/elimination, restart the tree with the incumbent | HiGHS's `mik` root is a restart product (`root+restart_off` 13.9–16.4 %); `restart_off` 120×/78× on `beavma`/`bppc8-02`; `presolve_reduce_only` within 2.2× everywhere | `beavma`, `bppc8-02`, `mik`×4, `gesa2`, `p0548`, `issue-2446` | +0–3, uncertain: HiGHS's restart rides on HPresolve; Rust's re-presolve would be FBBT + elimination | 1–2 weeks | ladder is the inequality `reduced_cost_fix` already uses, at the root with the root's safe bound; restart is a model rewrite → postsolve of the incumbent must be exact (verify against original rows, as the sub-MIP does) | zero code: P4.5's counter seeded with the sub-MIP's first incumbent on the 10 rows → kill if < 2.5 % fixable everywhere (then there is nothing to restart on) | §5 panel; bound-changing |
| **R6** | **Incremental/hyper-sparse primal pricing for cold and repair solves** (re-run of the retired steps 1+2 under R2's instrument) | its kill was `p0548` alone, a row that flips both ways in 9 consecutive panels; nodes −16 %, 0 dual losses | `issue-2446` (M), `bppc8-02`, `neos-911970`, every repair-heavy row | +0–1; broad throughput | recover from `dbf9b898`/`deb1b1f`, 1 day | bound-neutral per LP (same optimum, `updateVerify` guard); owner must agree the kill was an instrument artifact — otherwise this stays retired | same panel, hardened bars → kill on any non-`p0548` loss | §5 panel |
| R7 | Clique table + clique/implied-bound propagation (C.4a) | `bppc8-02`/`p0548`/`gesa2` have set-packing structure HiGHS propagates in `all_off` | `bppc8-02`, `p0548`, `gesa2` | unknown | 2+ weeks | bound-neutral by construction if exact | count packing rows/clique sizes offline first | last |

Honest sum: R1 + R3 + R4 land **~25–29/38** if every entry holds; R5–R7 are what the rest
(`beavma`, `bppc8-02`, `neos17`, `neos-911970`, `gsvm2rl3`) would need, and `gsvm2rl3`/
`neos-911970` sit at HiGHS's own 20 s limit (Table 5; HiGHS 11–14 s). 34/38 is not reachable
by the credible plan; see §3.

## 3. Gate critique: is "34/38 at ≤ 2× wall" the right bar?

No, for four reasons, each from the data.

1. **It demands HiGHS-equivalence on a tuned panel.** 34/38 = 12 of the 13 HiGHS-certified rows
   Rust fails, which span every mechanism family HiGHS has (C1–C5); two of them are at HiGHS's
   own budget limit. Six graduated levers were entry-tested and panelled on these 38; CLAUDE.md
   §2 calls named instances gate probes, not the gate. The size filter that chose the 38 was
   never recorded (`miplib_panel.py:4-5`), so the panel is not reproducible as a sample.
2. **Count-at-20 s is a step function on trajectory-sensitive rows** (`p0548` both ways,
   `gt2`'s 1-ulp flip, `khb05250`'s formulation flip, `istra`/`kaihu` 3/0 under some arms) with
   a ±1–2 swing per rep — the same width as several "levers".
3. **The wall bar is redundant with the count bar:** 320 of Rust's 398 s are the 16 time-outs.
   Over the 22 both-certified MIPLIB rows Rust's total is 75 s vs HiGHS's 61 s, geomean ratio
   **1.25×**, 15/22 within 2× (`attrib.out`; the outliers are `fiber` 9.4×, `gen` 5.1×,
   `b-ball` 4.3×). Speed is not what the gate is missing.
4. **20 s is a dev proxy.** `Model.solve`'s default is 3600 s (`modeling/core.py:9737`);
   the only longer measurement is the 60 s probe (`b4_60s.jsonl`), and nothing at the shipped
   budget has been run on the current engine.

**The narrower end states.** "HiGHS optional by class" has no static predicate
(`highs-parity-d2-analysis.md` §3.1) and its behavioural form (`auto`: Rust slice, HiGHS on
fallback) keeps HiGHS on the default path of an install with the extra, which the owner's
"no fallback" rules out. The owner's 2026-11-08 decision already achieved the packaging goal
(`highspy` optional; Rust the default without it).

**Recommendation.** Treat the remaining question — the default *when `highspy` is installed* —
as a user-harm question, and gate it on a held-out sample rather than on the 38:

- Draw N ≥ 60 MIPLIB 2017 "easy" instances by a **recorded** filter (e.g. HiGHS single-thread
  ≤ 60 s, n ≤ 10k, nnz ≤ 100k), excluding the 38; keep MIPLIB-38 + M as the development panel.
- Run at two budgets, 20 s and 120 s, 3 reps interleaved, Rust vs HiGHS route.
- Bars: 0 incorrect (unchanged); certified ≥ 0.9× HiGHS at both budgets; **no instance HiGHS
  certifies in ≤ 2 s that Rust leaves uncertified at 120 s** (the "`feasible` where HiGHS said
  `optimal` in under a second" harm named in analysis §3.3); SGM wall over both-certified ≤ 2×.
- Until it passes, HiGHS stays the default-when-installed as a documented alternative whose
  docstring names this gate (CLAUDE.md §5 state 3), and D4 (deleting the route code) waits on
  the same measurement.

This keeps the owner's goal reachable by generality (§2) instead of by instance-by-instance
parity, and it is the only gate shape that would survive the trajectory variance measured here.

## 4. What to stop

- **W6 plunging with a hot LP.** HiGHS's `all_off` arm *is* a plunging B&B with propagation,
  reduced-cost fixing and conflicts, and it fails `mik`×4, `iskar`, `istra`, `kaihu`, `b-ball`,
  `sp150x300d` without cuts (Table 2). The primal half of R1-expert is captured by the graduated
  sub-MIP and the entry-confirmed improve-dive; the cutoff saves 8 % of pivots (follow-up 6b).
  The largest build in the plan buys throughput on rows whose problem is the bound.
- **W2(b) feasibility jump.** `fj_off` changes no HiGHS node count (Table 4); its targets
  (`enlight_hard`, `amur`) fail in HiGHS too; the first-incumbent rows are M-panel only.
- **Presolve of the original model** (follow-up 20) and **W3's schedule** — retired on their
  entries; do not re-derive. R5 is a different mechanism (re-presolve after fixings) and must
  be argued as such, not as "presolve".
- **W5(a)/(d) and the EXPEL ceiling** — retired on entry; the record stands.
- **`DISCOPT_MILP_RAND_ROUND`**: its targets are M-only (`2122`, `p0548`); either bundle it with
  R1's panel or retire it then. Do not let it sit.
- **`DISCOPT_MILP_REENTRY_CUTS`**: its win (`gt2`) is 3/3 without it now; retire unless the
  pool-inheritance re-scope is wanted for the 60 s+ budgets (then it is R-budget work, not D2).
- **Judging anything on `p0548`**, and summing certifications in bar 2a — stop until R2 lands.
- The wall bar as a separate target.

## 5. First three things to do next week

1. **R1 entry, then fix.** Six solves (`enlight8`, `bppc8-02`; `CUT_LP=0/1`, 3 reps) on
   today's build with two new counters; build the per-node fallback; re-run on those rows plus
   `istra`/`b-ball`. This is the only item with a known-good endpoint (3/3 in six panels).
2. **R2 instrument hardening** in `d2_flag_panel.py` before any further panel: the
   no-regression bar, bimodal-row handling, per-flag fired check. Re-read the `INCR_PRICING`
   and `TOL_ROBUST` verdicts under the new bar and record the re-reading in the plan.
3. **R4's two-constant root probe** (depth 6, any start) on `nexp`×2, `sp150x300d`, `beavma`,
   `fiber`, `neos-911970` — minutes of compute, decides whether the path separator is the lever
   for C3 — and start **R3's build** against its deterministic node-budget entry on `mik`,
   recovering the W4 separator from `a9bc8d7d`/`d31ad1d2`.

## Appendix — HiGHS references verified in `ref/HiGHS` (1.15.1) for this review

`HighsMipSolverData.cpp:1310` `percentageInactiveIntegers`; `:1317` `performRestart`
(`appendCutsToModel` at `:1335`); restart triggers `:2127-2134` (≥ 10 %) and `:2494-2500`
(≥ 2.5 %, or any on first pass). `HighsRedcostFixing.cpp:37` `propagateRootRedcost`, `:74`
`propagateRedCost`, `:195` `addRootRedcost`. `HighsSearch.cpp:874` `evaluateNode`, `:924`
`setObjectiveLimit`, `:1623` `backtrackPlunge`, `:1870` `dive`; `HighsMipSolver.cpp:684-693`
`separateAndStoreBasis` (per queue-popped node), `:824-858` plunge limit. `HighsSeparation.cpp:38-40`
(tableau, path, mod-k registered), `:43` `separationRound`, `:162-166` add-resolve-age, `:181`
`separate`; `HighsLpRelaxation.cpp:536` `removeObsoleteRows`, `:629` `performAging`.
`HighsTableauSeparator.cpp:47` pool-soft-limit gate, `:232/:237` `generateCut` both signs.
`HighsPathSeparator.cpp:170` `maxPathLen = 6`. `HighsCutGeneration.cpp:25` `determineCover`,
`:280` lifted MIC, `:504` c-MIR heuristic, `:760` `postprocessCut`, `:1058` `generateCut`,
`:1382` `tryGenerateCut`. `HighsDomain.cpp:1936` `markPropagate`, `:2585` `conflictAnalysis`.
`HighsPrimalHeuristics.cpp:77` `solveSubMip`, `:394` RENS, `:661` RINS, `:1117`
`randomizedRounding`; `HighsMipSolverData.cpp:669` `moreHeuristicsAllowed`.
`HighsImplications.cpp:583`, `HighsCliqueTable.cpp:1605` separators. Not re-verified here
*(consult)*: the per-worker LP copy (`HighsMipSolver.cpp:275-301`), `HighsCutPool` scoring
lines, `HighsTransformedLp` global-bound transform.

## Results, 2026-11-08 (owner: "if the default helps leave it on; I want HiGHS parity")

`DISCOPT_MILP_CUT_LP` stays default ON: it nets +8 certifications on the 62-panel even with
the `enlight8` loss. Every run below is `d2_flag_panel.py`, 20 s, 3 reps, interleaved arms;
data in `data/highs-parity/d2-review/`. "OFF" is the shipped default (CUT_LP on).

**R1 entry (`r1_entry.json`).** The regression reproduces: CUT_LP on, `enlight8` 0/3 at
bound 20 (~8.7k nodes); CUT_LP=0, 3/3 at 27 (~46k nodes). A profile-only counter checks the
lazily relaxed rows' cuts at every node point: ~9,800 node LPs per `enlight8` solve
violate one (~193k violations), ~2,500 per `bppc8-02`. The kill (counter ~0) did not fire.

**Fix A, re-admit relaxed rows in place (`DISCOPT_MILP_CUT_LP_READMIT`, `r1_fixA.json`).**
Sound; `bppc8-02` bound 486.5 -> 491-493; `fiber`, `istra`, `kaihu`, `b-ball` keep 3/3;
`enlight8` 20 -> 21, still 0/3 (kill clause "enlight8 not 3/3": A alone is not the fix).

**Fix C, per-node pool lift alongside the fold (`r1_fixAC.json`, A+C): killed.** `fiber`
3/3 -> 0/3, `b-ball` nodes x3, `enlight8` stays 21. Code removed (never committed).

**Why the node rate collapses (`r1_rows.json`).** CUT_LP on, `enlight8`: node LPs carry 339
folded rows on a 64-row model, 95.6 % basic; the deletion pass was blocked 10,652 times
because a relaxed row's slack is nonbasic in some stored frontier basis. 9.2k vs 49.2k
nodes with CUT_LP=0. (`fiber` 1 rep: CUT_LP=0 uncertified, so CUT_LP does help there.)

**Fix D, delete aged relaxed rows and drop the bases that hold their slack nonbasic
(`DISCOPT_MILP_CUT_LP_DROP_RELAXED`, `r1_fixD.json`): fails.** `fiber`/`istra`/`kaihu` lose
certification. Part of that was a mislabel: a node with no stored basis was taken for the
root and ran the root-only dive (`solve_node`'s `is_root`). With a depth-based root test
under the flag (D', `r1_fixDp.json`) every W1 winner keeps 3/3, but `enlight8` falls to 18
at ~3k nodes and `kaihu`'s nodes double: cold re-solves cost more than the dead rows.
The default path never hits the mislabel (`NodeNoBasisOffRoot` = 0 in every default arm).

**HiGHS-expert answer (critique addendum):** HiGHS never repairs a stored basis; its queued
nodes hold none, so discarding a mismatched one is cheap there. The faithful port for
discopt, where every frontier node keeps a basis, is **fix E**: keep D's physical deletion,
but mark affected bases (m'+k basic candidates for m' rows) and repair each one lazily
when its node is popped, with a rank-revealing elimination that leaves k columns without
a pivot (HiGHS's `HFactor` primitive, `util/HFactor.cpp:395-425`, `:1263-1293`). Entry:
repaired-solve pivots vs ordinary warm solves; kill at > 3x or any W1 winner losing a rep.

**R0, keep run 1 going (`DISCOPT_MILP_RUN1_CONTINUE`, `r0_entry.json`).** Checked-in
entries: run 2's bound beat run 1's on 12 of 138 uncertified two-run rows (the expert's
"never" was too strong). Flag ON, 12 instances: cert-clean, certified 6 = 6; dual bound
better on `enlight8` (20 -> 21, 3/3), `beavma` (~352k -> ~356k), `gsvm2rl3`
(0.238 -> 0.253), `nexp-50-20-1-1` (25 -> 26, 2/3), `mik-1`/`mik-2`; worse on `neos17`
(0.101 -> 0.067). Median primal gap 0.126 -> 0.240: run 2's seeded restart (and the #280
swap) was producing the incumbents, as P4.0 recorded. Not graduable alone; it needs a
primal lever that replaces what run 2's restart gave.

**Next:** fix E (lazy basis repair) with A, then a §5 panel of A+E; R0 with a primal
lever (seeded restart heuristics inside the continuing run).
