# SOTA parity analysis: discopt vs SCIP/BARON — evidence, gaps, and the prioritized plan

**Date:** 2026-07-27 · **Tree:** `main @ 0459d406` (post #870/#877/#878/#880/#881)
**Rule for this document:** every claim cites a measurement — a dated run in this repo, a
committed plan-doc table, or a reference CSV. Anything without a number did not make the list.
Negative results (measured dead ends) are listed with the same force as the gaps: **spending
time on them is how the last three weeks under-delivered**, and they are binding under
CLAUDE.md §4 / `baron-gap-plan.md` §8.

## §0. Evidence sources

| source | what it contributes |
|---|---|
| fresh 3-way run, this date (`reports/global50_3way_2026-07-27*`) | current headline parity: discopt (warm daemon) vs BARON (GAMS) vs SCIP (.nl), 50 curated global-opt instances, 60 s, shared oracle |
| `reports/global50_3way_2026-07-20T12-24-28.md` | previous 3-way baseline (pre-#870/#877/#878/#880/#881) |
| `v-baron-remeasure-2026-07-07.md` | 61-instance defaults-only panel vs BARON; per-instance cert ledger |
| `baron-gap-plan.md` §1 | measured gap decomposition `wall ≈ floor × per_node × nodes`; TX0 62-instance attribution histogram |
| `scip-parity-kernel-plan.md` | F1–F3 diagnosis; E0 kernel node-rate bench (real exported node LPs) |
| `performance-plan.md` §1 | CC1–CC5 measured cost centers; falsified hypotheses |
| `scip-gap-nvs-diagnosis.md` | SCIP ablation on nvs17 (cuts vs branching); separator findings |
| `~/Dropbox/…/scip_join.csv` (1,610 rows) | SCIP reference wall/status per MINLPLib instance |
| `~/Dropbox/…/minlplib.solu` (980 `=opt=`) | ground-truth optima used by every soundness check here |
| this session's measured runs (2026-07-25→27) | tln quality (#862/#880), watercontamination scaling (#875), clay certification (#882), #844 instance re-measure |

## §1. Where discopt is at parity today (measured)

**1a. Correctness — the product — is at parity.** July-20 3-way, 50 instances, 60 s:
discopt 48/50 `ok`, **0 violations**; BARON 49/50, 0; SCIP 47/50 (2 GAP), 0. The
61-instance defaults panel: **0 violations**, and every `status=optimal` cross-checked
against `minlplib.solu`. The July-27 rerun (§2) re-confirms on the current tree. Three
days of adversarial probing this week (gear4 presolve `INF`-sentinel #877, the convex-tree
discarded-subtree #870/`1df5f71a`) found and removed two certificate bugs; the backlog
now has **zero open correctness bugs**.

**1b. Root bound strength on the convex families.** GMI closes **75–93 %** of the
convex-panel root spread; `rsyn0805m` root bound **beats SCIP's root** (#781 measurement,
quoted in `scip-parity-kernel-plan.md` F2). The envelope library is at catalog parity
(`relaxation-catalog.md`); cert-gap T1.1 measured every uncovered family as closed-form
box-affine.

**1c. The Rust LP layer already reaches SCIP-class node rates** — when driven correctly.
E0 bench on *real exported node LPs* (2,000 branching-shaped bound flips, release build):

| node LP | m×n | kernel pattern (amortized scale+CSC+LU) |
|---|---|---|
| nvs09 spatial lifted | 292×374 | **25,928 re-solves/s (38 µs p50)** |
| rsyn0805m OA+cuts (post-P1.0) | 537×635 | **1,419/s** (≥ the 500/s kill gate) |
| tanksize spatial lifted | 187×257 | 1,288/s |

**1d. This week's closures, oracle-verified on this machine:** tln4 **exact** / tln5 +4.9 %
/ tln6 +5.9 % (was +12/+213/+327 %, #880 plunging); `clay0303hfsg` certifies 26669.109557
vs `=opt=` 26669.10957 (#882); `rsyn0805m04hfsg`/`rsyn0810m04hfsg` certify at rel 1.4e-7 /
1.8e-7 via the convex kernel (#870); `watercontamination0202` 579 s → 49 s with wall
scaling in `T` (#878).

## §2. Current headline (fresh 3-way, 2026-07-27, 60 s, warm daemon)

Run: `reports/global50_3way_2026-07-27T12-31-43.md` (this tree, `main @ 0459d406`).
**Correctness gate: PASS — 0 violations, all three solvers.**

| solver | ok | GAP | VIOLATION | n/a | total wall (s) | geomean wall (s) |
|---|---|---|---|---|---|---|
| discopt | 48/50 | 0 | **0** | 2 | 340.6 | **0.540** |
| baron | 49/50 | 0 | 0 | 1 | 130.7 | 0.068 |
| scip | 47/50 | 2 | 0 | 1 | 391.3 | 0.188 |

Read against July 20 (pre-#870/#877/#878/#880/#881/#882): correctness identical
(48/49/47, 0 violations), geomean essentially unchanged (0.561 → 0.540 s; BARON
0.077 → 0.068; SCIP 0.198 → 0.188 — run-to-run noise, same machine). **The week's
merges were correctness and capability work off this panel; the headline multiplier —
7.9× vs BARON, 2.9× vs SCIP — is untouched, exactly as the G-A/G-D attribution
predicts: it lives in the per-node loop and the floor, which none of this week's PRs
addressed.** That is the cleanest available demonstration that P1 (the kernel) is
where the wall-clock gap actually is. Illustrative row: `clay0303hfsg` — default path
34.8 s vs BARON 1.4 s / SCIP 5.7 s, while the flag-gated kernel certifies the same
instance in ~7 s (G-C in one row).

## §3. The measured gaps

**G-A. Per-node cost: the node loop runs in the interpreter — 50–500× vs SCIP.**
116-instance profile (`scip-parity-kernel-plan.md` F1): **61 % JAX, 39 % Python, ~0 % Rust
LP wall**. The entire caching campaign (EP1/2/4a/4b/5 + CC1's −22 % gear4 win) took nvs09
from 294 → ~50 ms/node and **plateaued — that hypothesis is spent**. SCIP's node is
0.1–1 ms. nvs05: 20.5 nodes/s vs BARON 1,874 (**90×**), with the bound source being ~780
Python↔JAX scalar round-trips per NLP solve (`baron-gap-plan.md` §1.3). Meanwhile §1c
shows the same machine sustaining 26k LP re-solves/s in Rust on the same LPs. The gap is
architecture, not implementation detail.

**G-B. Primal capability: instances SCIP solves in <6 s where discopt returns *nothing*.**
Re-measured this session on `main`, 60 s, defaults (18 soundness assertions, 0 unsound):

| instance | discopt (60 s) | SCIP (`scip_join.csv`) |
|---|---|---|
| `watercontamination0202` (106,711 vars) | no incumbent, bound ≈ 0 vs opt 125.196 (vacuous) | **optimal, 2.56 s** |
| `gastrans582_cold13` | no incumbent | optimal, 5.38 s |
| `gastrans040` | no incumbent | optimal, 0.06 s |
| `rsyn0805m04hfsg` (default path) | no incumbent | optimal, 1.33 s |
| `ball_mk2_30` | no incumbent, bound −26.9 vs opt 0 | optimal, ~0 s |
| `chimera_k64ising-01` (#843) | no incumbent | optimal, 18.69 s |

The counter-evidence that this is fixable with the machinery already present: plunging
(#880) alone took tln6 from +326.8 % to +5.9 % and certified nvs17 — the node loop simply
never reached exact leaves before. And `rsyn0805m04hfsg` **is** certified in 5.2 s by the
convex kernel — the default path just never routes there (G-C).

**G-C. Routing: capabilities exist but the default path does not reach them.**
`DISCOPT_CONVEX_KERNEL` is default-OFF; with it ON, `rsyn0805m04hfsg`/`rsyn0810m04hfsg`/
`clay0303hfsg`/`syn05*` all certify correctly (15 assertions, 0 unsound, this session).
SCIP/BARON do not have a "flag off" state — routing *is* the product. The inverse case is
also measured: `watercontamination0202` classifies `convex=True` in 2.9 s and the
convex/MIQP route then runs **2001 s with no bound** (vs 49 s spatial) — so graduation is
genuinely a §5 panel question in both directions, not a toggle.

**G-D. The wall-clock multiplier on jointly-proved instances.** July-7 panel: discopt/BARON
geomean **14.4×** on the 40 jointly-proved (median 14.6×, max 180.6×). July-20 global50
geomean: discopt 0.561 s vs BARON 0.077 s (**7.3×**), SCIP 0.198 s (**2.8×**). TX0
attribution over 62 instances: **floor 26, node_count 18, no_bound 8, overrun 4** — i.e.
~42 % of the gap is the per-process floor (import 513 ms; solve itself BARON-competitive
on easy instances), which the warm daemon already halves (residual 3.7× = engine constant
~150 ms + per-node cost).

**G-E. The missing multi-row separator, sequenced honestly.** SCIP's ablation on nvs17:
no-cut SCIP closes in 6,796 nodes; with its `aggregation` separator, **70**. discopt's
single-row GMI/MIR measured: bound −27,795 → −27,291 then **plateau** (still 25× loose) —
discopt has no aggregation c-MIR. But two measured constraints order this *after* cheap
nodes: root-only cutting at certifying intensity starves incumbents (0/5 on
rsyn0810m/tls2, #781 held), and `scip-gap-nvs-diagnosis.md` measured OBBT × cuts ×
throughput as **multiplicative — they must land together**.

**G-G. Presolve/reformulation: the dominant mechanism on the hardest gap instances —
initially missing from this analysis.** (Added same-day after review pushback; measured
with `scip -c "read … optimize display statistics"` on the local SCIP, logs in the
session scratchpad.) Mechanism attribution for all five P2 instances:

| instance | SCIP presolve effect | nodes | incumbent found by | restarts |
|---|---|---|---|---|
| `watercontamination0202` | 106,712 → **566 vars, 107,210 → 560 cons (189×), 0.19 s** | 11 | `relaxation` | 0 |
| `gastrans582_cold13` | 2,186 → 598 vars | 12 | `pscostdiving` | 0 |
| `gastrans040` | 279 → 80 vars | 1 | `feaspump` | 0 |
| `ball_mk2_30` | none needed (30 vars) | 1 | **`trivial`** | 0 |
| `chimera_k64ising-01` | *grew* it: 1,192 ints → binaries + 1,587 product lifts (2,779 vars) | 14 | `relaxation` | 0 |

Three consequences. (1) **`watercontamination0202` is a presolve problem, not a primal
problem** — no incumbent constructor competes with deleting 99.5 % of the model; discopt
runs its entire root setup at full 106k size (#875's measured floor ≈ 27 s is *paying for
the un-presolved model*). **⚠ FALSIFIED 2026-07-27 — see §G-G.1 below; it is both, and
the residual after presolve is primal.** (2) `ball_mk2_30` needs only a trivial-point probe — note the
open discrepancy with #843's claim that the graduated trivial seed already resolves it;
defaults measurement shows no incumbent, so the seed's cold-start gating likely excludes
this class (same gate that excludes chimera, per #843). (3) The do-not-staff entry
"binary expansion (nvs17 7 → 2,751 vars)" is **family-scoped, not universal**: SCIP wins
`chimera` precisely by binarizing + lifting products; the falsification stands for the
nvs family only. **Restarts and conflict analysis measured NOT implicated on these five;
they remain unmeasured on the 40 jointly-proved slow-ratio panel — an open evidence task,
not a claimed non-area.**

**G-G.1. FALSIFICATION (2026-07-27, after #888 landed).** Three claims made above and in
#888's PR body are retracted here per CLAUDE.md §11. All three were contradicted by
measurement *after* the presolve work was scoped on them.

1. **"`watercontamination0202` is a presolve problem, not a primal problem" — FALSE; it is
   both, and the binding half is primal.** End-to-end with `DISCOPT_PRESOLVE_SUBSTITUTE=1`
   (60 s budget, subprocess-isolated, `substitute_targets_e2e.json`): the flag turns
   `time_limit`/**no incumbent** into `feasible` with objective **3190.4506** against
   `=opt= 125.1956151` — a **2449 % primal gap**. Deleting 99.5 % of the model bought a
   first incumbent and nothing more. The reduction is real; the claim that it is what
   stands between us and this instance is not.
2. **The "189×" headline describes one instance, not a class.** Random 300-instance census
   (seed 20260727, mirrored corpus): discopt's substitution pass reduces **nothing on
   64.3 %**, reaches ≥3× on **1/300**, and ≥10× on **0/300**. SCIP 10.0.2's *full* presolve
   on the same sample: nothing on 47.1 %, ≥3× on 8/297, ≥10× on **0/297**. Only 15 of 1610
   corpus instances have ≥10k vars with <5 % nonlinear rows. Independently corroborated by
   #888's own §5 panel: 13 of 66 vendored instances had anything to substitute, and the
   flag gained **0 incumbents and 0 certifications** (48/48 certified, 54/54 incumbents,
   both arms).
3. **"Substitution is single-pass, and coupling it to bound tightening is the residual
   gap" — WRONG MECHANISM.** (My framing when dispatching the follow-up; retracted before
   it was built.) `substitute_to_fixpoint` already iterates, and it **saturates at 2
   sweeps** — 4 and 8 sweeps buy exactly zero on `watercontamination0202`,
   `gastrans582_cold13`, and `gastrans040`. Nor are we doing *less* aggregation than SCIP:
   we eliminate **105,845** vars by aggregation vs SCIP's **67,587**. The residual is that
   SCIP's other 38,396 removals are **bound-driven fixings** (78,959 `ChgBounds`), a
   *different mechanism* we do not have at all — plus dual reductions, implied-free column
   elimination, and coefficient tightening. SCIP needs 6 rounds because its rounds
   alternate mechanisms; ours needs 2 because it only re-runs one. The gap is
   single-*mechanism*, not single-*pass*.

**Consequence for prioritization.** SCIP itself gets no reduction on 47.1 % of the random
300 and never reaches 10×, yet still beats us broadly — so closing our 35.7 % → 52.9 %
reduction-rate gap should not be expected to move the corpus geomean. The FBBT-coupled
propagation/fixing loop remains the honest completion of this mechanism and is cheap (the
`SubstDef::Fixed` postsolve inversion is already built), but it is **deferred, not
top-priority**: it would land a second default-OFF flag with the same measured 0-incumbent,
0-certification profile as #888 — the `DISCOPT_CUT_INHERIT` lesson (sound ≠ helpful).
Priority returns to what #844 is actually titled: **primal constructors** (LP feasibility
pump, pscost diving), which is where the 2449 % gap above lives.

**Open blocker on the #888 flag (must clear before it could ever graduate default-ON):**
`hda` bound quality regresses OFF −64,473 → ON −1.56e8 — sound (weaker, correct direction)
but far looser. Also noted, *not* caused by the flag: `tanksize`'s incumbent violates a
variable bound by 1.60e-6 on the pristine model, bit-identical in both arms, which the
panel's `FEAS_TOL = 1e-5` (`substitute_diff_panel.py:33`) hides — that tolerance is looser
than the repo's abs=1e-6 and should be tightened.

**G-F. `no_bound` family (8/62) — relaxation strength on family D** (tls2/tspn-class,
`baron-gap-plan.md` G5): the dual bound never moves, so no budget helps. Distinct from
G-B (those have vacuous-but-moving bounds); untouched by everything shipped this month.

## §4. The prioritized plan

Ordering rule: (leverage measured on the end metric) ÷ (evidence the approach works),
with measured dead ends excluded. Each area states the falsifiable exit metric up front.

---

### P1 — The compiled branch-and-cut kernel (BCK). *The only fix for G-A; unlocks G-E.*

**Evidence it is required:** the 50–500× per-node gap is the largest single factor and the
Python-side campaign to close it is **measured as spent** (G-A). Evidence the approach
works: E0 passed its own kill criterion on real node LPs (§1c); every ingredient (simplex,
Gomory, c-MIR, cover, cut_select, FBBT, tree) already exists in Rust and profiles at ~0 %
wall because Python never drives it (F3).

**Plan of record already exists** (`scip-parity-kernel-plan.md`): next step is **E1**
(template-refresh parity: analyze-once templates must reproduce Python-built rows ≤1 ulp
on all 62 vendored instances; kill: <90 % of rows templatable), then P1 behind
`DISCOPT_KERNEL` default-OFF.

**Fixed when:** E1 passes its gate; then the P1 gate verbatim — `rsyn*/syn*/clay` panel
certified **≤ 5 s each** (SCIP ≤ ~1 s; 5× interim bar), first-incumbent latency ≤ legacy,
`incorrect_count = 0`, §5 differential clean. End state: global50 geomean within **2×**
of SCIP.

### P2 — The six no-incumbent instances, by measured mechanism (G-B + G-G). *Closes #844, #861, #843.*

**Re-scoped after the G-G attribution:** these six are not one class. The fixes, in
SCIP-measured order of mechanism: **(a) presolve/reduction parity** for
`watercontamination0202` and the `gastrans*` pair — entry experiment: run discopt's
existing Rust presolve passes (`presolve/aggregate.rs`, `eliminate.rs`,
`factorable_elim.rs`) on these three and measure the achieved reduction vs SCIP's
(189×, 3.7×, 3.5×); the passes exist and the solve path shows no reduction (#875 kept
106k vars end-to-end), so the first question is wiring, not new algorithms. **(b) a
trivial-point seed gating fix** for `ball_mk2_30` (and re-test chimera's exclusion) —
likely a one-line gate, blocked on reproducing the #843 discrepancy. **(c) genuine
primal constructors** (LP feasibility pump, pscost diving — both named by SCIP's own
stats) for whatever remains after (a) and (b).

> **P2(a) entry-experiment result (2026-07-27, iteration 2, Contributes to #844).**
> The wiring hypothesis above — *"the first question is wiring, not new algorithms"* —
> is **FALSIFIED**. Existing passes invoked directly on the instances (scripts
> `discopt_benchmarks/scripts/presolve_reduction_{entry,census}.py`) achieve, best
> case across `{eliminate, aggregate, factorable_elim, simplify, fbbt}`:
>
> | instance | discopt achieved (wall/term) | SCIP reference |
> |---|---|---|
> | `gastrans040` | 279 → 269 vars (**1.04×**), 1.6 s NoProgress | 279 → 80 (**3.5×**), 0.06 s |
> | `gastrans582_cold13` | 2,186 → 2,025 vars (**1.08×**), 60 s TimeBudget | 2,186 → 598 (**3.7×**), 5.4 s |
> | `watercontamination0202` | 106,711 → 106,711 vars (**1.00×**) FULL_SET / 106,369 (aggregate-alone, **342 vars in 30 s**) | 106,711 → 566 (**189×**), 0.19 s |
>
> Two independent root causes, **both new-algorithm, not wiring**:
> 1. **No general affine substitution.** `eliminate`/`aggregate`/`factorable_elim`
>    each require the eliminated variable to appear in *exactly one expression*
>    (its defining equality, nowhere else). SCIP substitutes a doubleton-defined
>    variable *out of every row and the objective* it appears in. On `gastrans040`
>    the census finds 105 doubleton equalities but `aggregate` fires on only 10 —
>    the rest define variables that also appear in inequalities, so the precondition
>    rejects them. This is the binding limit where time is *not* the constraint
>    (`gastrans040` finished with NoProgress at 1.04×).
> 2. **Superlinear implementation.** `aggregate_variables_until` rescans every
>    constraint per aggregation (O(applications × rows)); measured ≈11 aggregations/s
>    on `watercontamination0202` (342 in 30 s), projecting **~2.6 h** for the full
>    reduction SCIP does in 0.19 s. In `FULL_SET` the slow `eliminate` pass starves
>    `aggregate` of the shared budget → 1.00× (0 vars) in 60 s.
>
> Plus a **plumbing** gap that blocks *using* any reduction even if achieved: the
> reduced `ModelRepr` is discarded on the solve path (`propagate_bounds_to_model`
> copies bounds only; `solver.py:6043` never consumes the aggregated model), and
> **no postsolve is chained** — `AggregationRecord` is recorded but never inverted
> (`orchestrator.rs:140`, `aggregate.rs:79`: *"does not currently chain post-solve
> recovery"*), so solutions cannot be reported/verified in original variables.
>
> **Structural census (`watercontamination0202`):** 106,201 equalities (0 nonlinear),
> = 15,834 singleton + 81,821 doubleton + 8,546 ≥3-var linear. **92 % of the reduction
> is plain 1-or-2-var equality aggregation** — so the *transform class* SCIP uses is
> not exotic, but discopt has no scalable, substitute-everywhere implementation of it,
> and 8 % (8,546 rows) additionally needs ≥3-var Gaussian elimination that no pass has.
>
> **Kill verdict: NEW-ALGORITHM. STOP — do not build a new presolver in this
> iteration.** Scope for a follow-up (P2(a′)): (i) a batch substitution-graph
> aggregator (union-find over doubleton/singleton equalities, one rewrite pass,
> substitute-everywhere semantics) replacing the O(n²) rescan; (ii) ≥3-var free-column
> Gaussian elimination; (iii) a reduced-model solve entry + postsolve chain that
> inverts `AggregationRecord`/`factorable_elim` records and feasibility-verifies the
> recovered point against the pristine model. Bound-changing ⇒ full §5 differential
> panel before any default.

**Evidence it is required:** the G-B table — six instances at "no incumbent in 60 s" vs
SCIP ≤ 18.7 s, all with `=opt=` oracles. Evidence of tractability: plunging's measured
17× quality jump on the same class (#880); `ball_mk2_30`'s node loop reaches bound −26.9
soundly, it just never finds a leaf; #844 already specifies the SCIP recipe (LP-based
feasibility pump, sub-MIP RENS) and records that all current levers were re-measured and
fail.

**Fixed when:** each of the six returns a **verified-feasible incumbent within 60 s**
(defaults), measured by the #872 quality panel (which is already wired to `minlplib.solu`
and counts false primals); quality bar: primal gap ≤ 10 % on each (tln precedent shows
the machinery reaches ≤ 6 %); `incorrect_count = 0`; and the #844 panel's own gate
(`gains / lost_incumbents / cert_regressions / overshoots / unsound`) stays clean.

### P3 — Route the convex family by default (G-C). *Cheapest large win; removes rsyn from P2.*

**Evidence it is required:** `rsyn0805m04hfsg` certified in 5.2 s by machinery that ships
default-OFF while the default path returns nothing in 60 s. SCIP's 1.33 s on the same
instance is *routing + kernel*, not magic. Evidence of risk (why this is a panel, not a
toggle): the `watercontamination0202` counter-case — `convex=True` classification routes
it into a 2001 s no-bound solve.

**Fixed when:** a §5 graduation panel over the kernel-eligible corpus shows the bars below.
The candidate pool is the `syn*`/`rsyn*`/`clay*`/`cvxnonsep*` families — **136 candidate
`.nl` files** in the Dropbox snapshot (52+48+12+24, counted) — of which per-instance
eligibility must be established by the panel itself (an eligibility sweep of all 136 was
started and did not finish in-session; only the 4 vendored instances are confirmed
eligible, which is why the in-repo corpus alone cannot graduate this flag). Bars: cert-clean (0 crossings vs `.solu`, 0
cert regressions), net-positive (wall/certs strictly better on the family, no default-path
instance worse), **and** a size/time guard that provably excludes the
watercontamination-class counter-case. Then `DISCOPT_CONVEX_KERNEL` defaults ON and
`rsyn0805m04hfsg` certifies ≤ 10 s end-to-end from `solve()` with no env vars.

### P4 — Family-D bound strength (G-F). *The 8/62 `no_bound` instances.*

**Evidence it is required:** TX0's histogram — 8 of 62 instances lose to a bound that
never moves; no throughput or primal work can help them (`baron-gap-plan.md` G5 diagnosis
stands, unchanged by this month's work). BARON proves 4 of them at 60 s (nvs05/nvs09/
tanksize/tls2, July-7 ledger).

**Fixed when:** on the G5-named members, the root relaxation produces a finite,
`.solu`-valid dual bound that **tightens under branching** (trajectory recorder T0.2 shows
strict improvement), and ≥ 2 of the 4 BARON-proved members certify at 60 s with
`incorrect_count = 0`. Bound-changing ⇒ full §5 differential panel.

### P5 — Kill the residual floor honestly (G-D floor share). *Measurement + product, not engine.*

**Evidence it is required:** floor dominates 26/62 of the gap ledger; import tax 513 ms
(86 % of a trivial solve); daemon already halves the easy-class gap to 3.7×, residual =
~150 ms/solve engine constant (Appendix B) — real to users invoking the CLI cold.

**Fixed when:** the benchmark lane reports daemon-mode by default (G4 landed — keep it the
lane of record); the per-solve engine constant is profiled and ≤ 50 ms on `alan`
(currently ~150 ms); cold-start `discopt solve alan.nl` ≤ 250 ms end-to-end. All
bound-neutral by construction (nothing touches the search); verified by byte-identical
node counts on the deterministic panel.

---

### Explicit non-areas — measured dead ends; do not staff

| tempting | why not (measurement) |
|---|---|
| more Python/JAX caching for node cost | EP campaign plateaued at ~50 ms/node; "hypothesis spent" (kernel-plan F1) |
| root-only cutting at certifying intensity | starves incumbents 0/5 (#781 held) |
| single-row GMI/MIR push on nvs family | plateaus 25× loose (`scip-gap-nvs-diagnosis.md`); needs aggregation *inside* cheap nodes (P1) |
| truncating bound-producing solves to meet budgets | §8.1: casctanks bound −99.09 → **+5.70** — the overruns ARE the dual bound |
| ratio-of-`T` wall targets for root-setup instances | §8.6 + #654 test: the bar is "wall scales with `T`"; watercontamination floor ≈ 27 s mirrors sonet23v4's documented 24.5 s |
| capping infinite variable bounds to make NS certify | unsound: shrinks the feasible set (#871, re-scoped) |
| binary-expansion of integer products | nvs17 7 → 2,751 vars (`scip-gap-nvs-diagnosis.md`) |
| blanket-disabling node NLPs | tls2 bound looser, tspn12 incumbent lost, panel wall worse (§8.4) |

## §5. How the last three days actually score against this list

For calibration of the plan, not self-congratulation: #877/#870/#881 were correctness
(the gate, not the goal — but non-negotiable); #880 and #882 were P2-class work (measured
17× and a new certificate); #878 was P5/G-D-overrun class; the #875 ratio-chasing and the
sparse-`LinearContext` branch were **non-area work** (the branch was falsified end-to-end:
40× slower and bound-losing). The lesson encoded above: the non-area table is as
load-bearing as the priority list.
