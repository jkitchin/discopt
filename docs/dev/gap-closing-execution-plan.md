# Gap-Closing Execution Plan — discopt → BARON parity campaign

**Status:** ACTIVE (wave 1 launched 2026-07-10)
**Maintainer directive (2026-07-10, verbatim intent):** *"I don't want the cheapest
path, I want the hard work required to close this gap. All of it."* and *"If BARON
uses branch-and-reduce, we should have that implemented and on the hot solve path,
not on an unmerged branch."*
**Audience:** this document is written to be executed by Opus in stages, task by
task, without additional context. Every task names its entry experiment, kill
criterion, verification regime, and definition of done. Do not start a build step
before its entry experiment's verdict is recorded in §7.

This plan is the *execution layer* over the canonical roadmap
(`docs/dev/certification-gap-plan.md`, whose §0 is a binding contract and §14 the
master task list). Where this plan and that document disagree on *what to build*,
that document wins; where they disagree on *sequencing/current state*, this plan
is newer (2026-07-10) and wins.

---

## §0 Binding constraints (non-negotiable; copied intent from CLAUDE.md + cert-gap-plan §0)

0.1 **Correctness before performance.** `incorrect_count ≤ 0` with zero slack on
    every gate. The certificate invariant (dual bound never crossing the oracle;
    `bound ≤ incumbent` for min) holds on every panel. Never weaken a validation,
    fallback, guard, tolerance, or gate to make a task pass — if a goal can only
    be met that way, the goal loses; stop and record it.
0.2 **Fix the class, not the instance.** Named instances (nvs21, st_e36, gear4,
    tls2, …) are gate probes only. No instance-keyed code, ever.
0.3 **Entry experiment before implementation** (CLAUDE.md §4). State hypothesis,
    evidence, experiment, kill criterion; run it; record the verdict in §7 before
    building. A falsified hypothesis is banked progress — record it in §6 and in
    `docs/dev/performance-plan.md` §6 house style.
0.4 **Two verification regimes:**
    - *Bound-neutral* (refactors, caching, retry-on-failure, marshaling):
      `node_count` and certified `objective` **exactly unchanged** on the cert
      panel (`discopt_benchmarks/scripts/check_cert_neutrality.py`). Any drift —
      even apparent improvement — means the change is wrong.
    - *Bound-changing* (relaxations, cuts, reductions, DAG changes): behind a
      flag default-OFF; differential bound test (new ≥ old AND ≤ oracle on fixed
      boxes); feasible-point sampling (no valid point cut); `incorrect_count = 0`
      on the full panel; adversarial suite green; **3 consecutive green verdicts
      before any default-ON flip** (T2.6 rule — independent held-out gate runs
      with different instance draws/TLs count as verdicts).
0.5 **Workflow.** Feature branch + PR per task, task ID in the title. Every PR:
    `pytest -m smoke`, adversarial suite
    (`pytest -m slow python/tests/test_adversarial_recent_fixes.py`),
    `cargo test -p discopt-core` when Rust touched, regression test that fails
    before/passes after. PR bodies via `--body-file` only. Perf claims carry the
    measurement, not adjectives. Agents work in isolated worktrees; never touch
    the maintainer's checkout.
0.6 **Falsified hypotheses are binding** (§6). Do not relitigate them.
0.7 **Concurrency hygiene.** When several agents run on one machine, gate
    verdicts use deterministic metrics (node counts, objectives, status);
    wall-clock comparisons run back-to-back and note possible contamination.

---

## §1 Measured baseline (2026-07-10, global50, TL=60s, defaults-only main)

Full report: `global_opt_baron_vs_discopt_2026-07-10T11-32-04.md` (regenerable:
`discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py --time-limit 60`).

- **Correctness: discopt VIOLATIONS 0** (ok 46, gap 0, n/a 4). BARON 0 (ok 49).
- **Coverage: discopt certifies 43/50; BARON 49/50.**
- **Speed decomposition** (43 co-certified):
  - *Easy class* (BARON < 1 s; n=40): geomean **25.9×** — an artifact of
    discopt's **fixed ~0.3–1 s per-solve floor** (median 1.02 s, p25 0.33 s):
    imports + JAX init + compile, not the engine. → OVERHEAD-1.
  - *Hard class* (BARON ≥ 1 s; n=3): geomean **4.2×** (clay0303hfsg 41.6×,
    cvxnonsep_nsig30 4.4×, tspn05 **0.4× — discopt wins**). → engine + cuts +
    branch-and-reduce fronts.
- **The 7 non-certified:** nvs05, nvs09, tanksize (optimum found instantly,
  dual bound stalls 60 s), casctanks, tls2 (BARON solves at ROOT, 0.0 s —
  cut/presolve strength), st_miqp5 (binary `.nl`, parser refuses — coverage
  gap, not solve gap), plus tls2/casctanks timeout.
- **Why a week of perf work didn't move this number:** nearly all of it landed
  flag-gated default-OFF (gates blocked by single-instance certificate losses),
  or was capability (reduced-space, opt-in by measurement), or was mandatory
  correctness (C-38..C-44). The blocking cert losses share one root: **LP-engine
  failure behavior under the sparse route / shifted node streams** (§2 front A).

**Campaign definition of done (measured, not vibes):**
| metric | now | target |
|---|---|---|
| global50 incorrect_count | 0 | **0 (invariant)** |
| global50 certified | 43/50 | **≥ 48/50** |
| hard-class geomean vs BARON | 4.2× | **≤ 1.5×** |
| easy-class median wall | 1.02 s | **≤ 0.35 s** |
| root_gap_ratio_vs_baron gate | unevaluable (nulls) | **evaluable and ≤ 1.3** |
| branch-and-reduce | unmerged branch | **default-ON on the hot path** |

---

## §2 Campaign structure — fronts and dependency DAG

```
A  Engine failure-robustness      A-2 dense retry ──────────┐
                                                            ├─► BR-3 re-gates (node_reduce,
BR Branch-and-reduce on hot path  BR-1 merge r2 ── BR-2 ────┤    square_cost_gate, obj_branch_priority)
                                     │                      │
                                     └─ root_fixpoint ON    │
CUTS Cut strength                 CUTS-1 c-MIR ── CUTS-2 ───┼─► V-2 final global50 validation
                                                            │
STRUCT Structure preservation     STRUCT-1 CSE/V-seg ── STRUCT-2 Q-extraction
                                                            │
OVERHEAD Fixed floor              OVERHEAD-1 ───────────────┤
                                                            │
TAIL Certification stalls         TAIL-1 ───────────────────┘
```

Wave 1 (parallel, launched 2026-07-10): A-2, BR-1, CUTS-1, STRUCT-1, OVERHEAD-1.
Wave 2 (auto-unblocked): BR-2 (after BR-1), BR-3 (after A-2 + BR-1), CUTS-2
(after CUTS-1 + BR-1), STRUCT-2 (after STRUCT-1), TAIL-1 (anytime; cheap).
Wave 3: V-2 final validation + default-ON graduations.

Tracker mapping (session task IDs): A-2=#85, BR-1=#78, BR-2=#79, BR-3=#80,
CUTS-1=#81, STRUCT-1=#82, OVERHEAD-1=#83, TAIL-1=#84.

---

## §3 Task specifications

> Anchors below were verified 2026-07-02..07-10; line numbers drift — re-verify
> with grep before editing, never edit blind.

### A-2 — Failure-triggered dense LP retry (engine keystone) — IN FLIGHT

**Context.** #557's density-gated sparse LU route (merged #573, flag
`DISCOPT_LU_DENSITY_ROUTE`, default OFF) wins 1.7–7.8× on wide-McCormick bases
but was blocked from graduating by an nvs21 certificate loss. Task #77's entry
experiment **falsified the conditioning-gate hypothesis** (populations inverted:
failing solves factorize *well-conditioned* bases, cond₁ p50=16; healthy solves
run at 10¹⁰–10¹⁶; `growth` uniformly 1.0 — no signal) and **corrected the
mechanism**: the sparse route triples the LP failure rate (39 vs 12 status=None
exits); failed nodes retain loose-but-valid inherited bounds → final bound stuck
→ `feasible` not `optimal`. Never a corrupted optimum; certification, not
soundness, is what breaks.

**Validated fix (entry experiment green, prototype data):** on sparse-route LP
failure, retry that LP once via the dense route. Cures nvs21
(stuck −1.59e7 → **optimal −5.685216**, 27/30 failures rescued), preserves
st_e07 7.8× and nvs06; naive cold-retry erodes st_e36 1.69×→1.33× — production
version must be leaner (warm retry only if the post-failure basis state is
provably sound; if in any doubt, cold — sound-or-refuse).

**Build.** Rust, LP-solve level (`crates/discopt-core/src/lp/simplex/`) so all
callers inherit it; Python `_solve_lp_warm` None-branch acceptable fallback if
Rust plumbing is awkward (state which and why). Behind the existing flag,
default OFF. **Soundness invariant: the retry may only replace a FAILURE with a
robust-path result — never accept/blend a suspect fast-path result.** Retry
also fails → existing fallback chain unchanged.

**Gates.** `cargo test -p discopt-core`; cert-neutrality flag OFF; #74
graduation-gate re-run flag ON: nvs21 optimal→optimal, `incorrect_count=0`,
st_e36-class wins quantified; regression test for the retained certificate.
Record the #77 falsification in `performance-plan.md` §6 in the same PR.
**Done when:** PR green + gate table in the PR body; flag default-ON eligible
noted (flip in a separate PR after T2.6 verdicts).

### BR-1 — Merge the branch-and-reduce loop; bank `root_fixpoint` default-ON — IN FLIGHT

**Context.** The T2.3/T2.4 orchestration (root cutoff-FBBT↔cutoff-OBBT fixpoint
+ per-node `reduce_node()`, ~1,366 lines, flags `DISCOPT_ROOT_FIXPOINT` /
`DISCOPT_NODE_REDUCE` default-OFF, tests
`python/tests/test_r2_branch_and_reduce.py`) sits on branch
`r2-branch-and-reduce-loop`, unmerged. The T2.0 correctness preflight is
satisfied (C-16 and all listed P0/P1 = fixed in `correctness-issues.md`).
`root_fixpoint` already has **green verdict 1 of 3** (#581 gate: 0 incorrect,
0 cert loss, nodes 3731→3207 = −14%).

**Build.** (1) Rebase onto current origin/main (conflicts expected in
`solver.py` — reduced-space wiring landed near the node loop; flags stay OFF).
(2) Full suite + cert-neutrality flags-OFF (byte-identical). (3) PR; merge on
green. (4) Two more independent held-out gate runs (longer TL / different
instance draw vs the #581 panel; methodology: shared OFF baseline, ON
comparison, oracle cross-check vs `minlplib.solu`; criteria: incorrect=0, zero
cert loss, drift ≤ abs 1e-4/rel 1e-3). (5) Both green → one-line default-ON PR
for `DISCOPT_ROOT_FIXPOINT` citing all three verdicts. Any red → record the
failing instance + mechanism; do not weaken.
**Done when:** loop merged; root_fixpoint default-ON (or its blocker precisely
recorded in §7).

### BR-2 — Make the loop BARON-shaped (Phase-2 completion) — blocked by BR-1

Five independent sub-builds, each its own flag + PR; all bound-changing regime
except (c) which is differential:
a) **T2.4a node-LP duals.** `MccormickLPResult` (`_jax/mccormick_lp.py` ~:188)
   exposes status/lower_bound/x only — callers cannot run DBBT/RC-fixing per
   node. Plumb duals/reduced costs through (backend already computes them).
   Wiring-only PR, bound-neutral.
b) **T2.5 OBBT escalation scoring.** OBBT candidates today = all columns in
   index order (`_jax/obbt.py` ~:860, no scoring). Implement width×|reduced
   cost| top-k selection (needs (a)); budget-aware escalation per cert-gap-plan
   §14 T2.5 spec.
c) **T2.2 warm/persistent OBBT probe LPs** — the measured per-node lever
   (T1.6: the OBBT LP loop dominates per-node cost). Differential regime:
   per-probe bound equality vs cold path to 1e-9 on a sampled A/B set;
   `check_cert_neutrality.py` NEUTRAL.
d) **In-tree probing.** `presolve/probing.rs` runs root-only (default pass list
   `expr_bindings.rs` ~:672). Run bound-strengthening probing at nodes under a
   depth/budget policy.
e) **Root OBBT with incumbent cutoff.** Root sequence (`solver.py` ~:3835-3924)
   runs OBBT without the cutoff; `fbbt_with_cutoff` exists
   (`presolve/fbbt.rs:1098`). Feed the incumbent through.
**Binding negatives:** OBBT-on-aux/`cascade_aux` is measured-dead; gear4 is
reduction-resistant and is NOT a gate probe.
**Done when:** each sub-flag has a green gate; the composite (all ON) passes
the panel; default-ON flips proceed per T2.6.

### BR-3 — Re-gate the nvs21/st_e36-blocked flags — blocked by A-2 + BR-1

Re-run the #581 graduation gates for `node_reduce` (~10% faster, blocked by
st_e36 cert loss), `square_cost_gate`, `obj_branch_priority` (nvs21-blocked)
**with the dense retry ON**. Hypothesis: the cert losses are the same
LP-failure mechanism A-2 cures. Each flag that goes green ×3 verdicts →
default-ON PR. Any still-red flag: root-cause its residual loss (new entry
experiment, not a tweak).
**Done when:** every flag is either default-ON or has a §7-recorded blocker.

### CUTS-1 — Aggregation c-MIR + MIR complementation — IN FLIGHT

**Evidence.** SCIP ablations: aggregation/complemented-MIR = 97–169× node
reduction on nvs17/19/24, growing with size (`scip-gap-closing-plan.md`).
discopt's GMI + Python c-MIR are net-negative at equal node budget. Rust has
GMI/MIR/cover + efficacy/orthogonality selection (`lp/cut_select.rs`) but no
aggregation, and `mir.rs` ~:59 lacks upper-bound complementation.

**Entry experiment (mandatory):** re-run/extend
`discopt_benchmarks/scripts/cut1_cmir_oracle_injection.py` on current main —
inject oracle aggregated c-MIR cuts on nvs17/19/24 at fixed node budget.
**Kill criterion:** <5× node reduction on all three → separator build is dead;
record and stop.
**Build (on GO):** fix MIR complementation (+unit tests); implement SCIP-style
bounded row aggregation (≤k rows, guided by the fractional point) + c-MIR on
the aggregate, in Rust, wired into `cut_select.rs`; flag `DISCOPT_CMIR_AGG`
default OFF. Every emitted cut validated by feasible-point sampling
(`assert_cut_valid` or equivalent); differential bound test; panel + adversarial
green. Measure nodes+wall on nvs17/19/24 + 10-instance integer-heavy draw.
**Done when:** PR green with the ON/OFF table; graduation per T2.6.

### CUTS-2 — Root cut fixpoint + pool aging on the default path — blocked by CUTS-1 + BR-1

Integrate separation into the BR-1 root loop (reduce↔relax↔**separate**
fixpoint — the full BARON root); move cut-pool aging from the opt-in path to
default. Entry experiment: with CUTS-1 ON, does adding a separate-stage to the
root fixpoint close root gap beyond either alone (measure root_gap on the
TAIL-1 instrumentation)? Kill: no additional gap closed → keep loop and cuts
independent.

### STRUCT-1 — CSE/hash-consing + `.nl` V-segment preservation — IN FLIGHT

**Context.** `expr.rs` ~:266 — `add` never dedups (no hash-consing); the `.nl`
parser discards AMPL defined variables (V segments), so shared subexpressions
duplicate into the DAG, JAX compile, lifted LP (duplicate aux columns), and
every FBBT sweep. Multiplier on all other fronts.
**Entry experiment:** duplication census on the 61-file in-repo corpus + 30
snapshot instances: identical-subtree rate (hash over op/children/constants) +
discarded V-segment counts. **Kill:** median dedup gain <10% of DAG nodes AND
negligible V-segments → deprioritize.
**Build (on GO):** (a) hash-consing at construction — try for exact
bound-neutrality (byte-identical panel); if FP-order effects appear, demote to
flag + bound-changing regime (report honestly which regime it landed in);
(b) V-segment retention as shared DAG nodes — flag
`DISCOPT_NL_DEFINED_VARS` default OFF, bound-changing regime. Measure DAG size,
compile time, LP column count, FBBT sweep time, end-to-end wall on the
most-duplicated instances.
**Done when:** PRs green; payoff table recorded; graduation per T2.6.

### STRUCT-2 — Quadratic/Q-matrix structure extraction — blocked by STRUCT-1

Extract Q/bilinear structure in the Rust IR (today: degree checks only;
convexity detection is Python-side). Drives relaxation choice + cut families
from recognized structure (what SCIP/BARON do). Entry experiment: count global50
instances whose objective/constraints are recognizable Q-forms and estimate
bound improvement from PSD/eigen-based relaxation vs current path on 5 of them.
Kill: <20% of panel recognizable or no bound delta → record, stop.

### OVERHEAD-1 — Kill the fixed startup floor — IN FLIGHT

**Evidence:** §1 easy-class decomposition. **Profile first** (importtime, JAX
init, parse, first-compile vs warm, B&B; 5× repeats), then fix top contributors
≥20% of floor only: lazy imports of heavy optional deps; persistent JAX
compilation cache; deferred JAX init; a no-JAX fast path for provably
linear/quadratic instances (strict structural certificate — sound-or-refuse on
routing). Bound-neutral regime (byte-identical panel).
**Binding negative:** the per-NODE Python tax is falsified as a lever (T1.6);
this task is per-SOLVE fixed cost only.
**Done when:** before/after easy-class table (median, p25, geomean vs recorded
BARON times); target median ≤ 0.35 s.

### TAIL-1 — Certification stalls, binary `.nl`, root-gap instrumentation — QUEUED

a) **Stall decomposition** nvs05/nvs09/tanksize: optimum found instantly, dual
   bound flat for 60 s. Per instance: which constraint/operator class keeps the
   relaxation loose (nvs09 has log products; P1.3 tight monomial hulls were
   deferred — revisit), does OBBT move the box, does the BR loop (flag ON) close
   it? Fix the *operator class*. casctanks: revisit after A-2 + BR-2.
   tls2: expected covered by CUTS-1 — verify, don't duplicate.
b) **Binary `.nl` support** in the parser (st_miqp5 unreadable; +1 coverage).
c) **Root-gap instrumentation:** populate `root_gap`/`root_time` in benchmark
   results (currently null → the `root_gap_ratio_vs_baron ≤ 1.3` gate in
   `benchmarks.toml` is unevaluable). Needed to measure BR/CUTS progress; do
   this sub-task FIRST.

### V-2 — Final validation — blocked by all fronts

Re-run global50 (methodology of §1, same TLs) on defaults-only main after each
default-ON graduation lands, and once at campaign end. Compare against §1
baseline + the definition-of-done table. Also re-run the 3-way smoke and the
adversarial suite. Post the table to the tracking issue; update this §7.

---

## §4 Sequencing rules

1. A P0/P1 in `correctness-issues.md` touching a layer freezes perf work on
   that layer until fixed (cert-gap-plan T2.0 rule). Backlog is clean as of
   2026-07-10.
2. Default-ON flips are one-line PRs, separate from feature PRs, citing 3 green
   verdicts each.
3. One task = one PR. Sub-builds in BR-2 are separate PRs.
4. TAIL-1c (root-gap instrumentation) should land early — it is the measuring
   stick for BR/CUTS.
5. After every merged default-path change: `pytest -m smoke` + adversarial on
   main before the next merge (keep main robustly green).

## §5 Gate scripts and oracles

- Cert panel + neutrality: `discopt_benchmarks/scripts/check_cert_neutrality.py`
- Graduation gate: `discopt_benchmarks/scripts/graduation_gate.py` (else
  replicate #581 methodology: shared OFF baseline, ON comparison, oracle
  cross-check, tolerances abs 1e-4 / rel 1e-3, cert-loss = optimal→non-optimal)
- Head-to-head: `discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py`
  (note: its `NL_DIR` is the 61-file in-repo corpus; st_miqp5 absent until
  TAIL-1b — the 2026-07-10 run symlinked it from the snapshot)
- Oracle: `~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu`
- Adversarial: `pytest -m slow python/tests/test_adversarial_recent_fixes.py`

## §6 Binding falsifications (do not relitigate)

| # | falsified hypothesis | evidence | date |
|---|---|---|---|
| F1 | Warm-primal node LP warm-start is a lever | implemented sound, INERT | cert T1.4 |
| F2 | Per-node Python tax is the lever | re-profile: OBBT LP loop dominates | cert T1.6 |
| F3 | Reduced-space McCormick tightens root bounds (QP or pooling/bilinear) | P2.4 sweep: 0 wins, ties or looser | 2026-07-10 |
| F4 | Cut-pool inheritance graduates broadly | ceiling confined to dense-QP; shipped structure-gated | THRU-4/C-44 |
| F5 | OBBT-on-aux (`cascade_aux`) helps | neutral-to-regressive; nvs22 fails to certify | perf-plan §6 |
| F6 | **LU condition estimate discriminates failing bases** | populations inverted (fail p50=16; healthy ≤1e16); growth no-signal; mechanism = 3× LP failure rate, not corruption | 2026-07-10, #77 |
| F7 | gear4 is reducible by cutoff-OBBT/probing | stalls at 2.46M box; not a gate probe | plan §14 |
| F8 | **The tainted tree "proved" its final frontier bound** (DECOMP-1 §3: nvs05 4.87, tanksize 0.881) | per-node accounting (B2-FIX, task #89): the rigorous bound of a tainted tree is min(frontier, tainted nodes' pop-time bounds); on nvs05 the earliest taint (iteration 2) floors it at **1.3521** and on tanksize at 0.8473 — the frontier value is NOT rigorous once those subtrees were removed unproven. The recoverable headroom is marginal (nvs05 1.3481→1.3521), not 3.6×. The real lever is preventing the unsound removal (bound sentinel-fathomed NLP-failure nodes with the always-valid interval bound so they branch instead — the McCormick-LP path already does this rescue), which is bound-changing and separately gated. | 2026-07-10, #89 |
| F9 | **A-RESCUE (task #91): keeping failed-NLP nodes OPEN with a rigorous interval/alphaBB bound on the alphaBB/interval route raises the certified dual bound toward the incumbent** (F8's predicted lever) | Entry experiment (nvs05, tanksize, TL 60, `DISCOPT_NLP_FAIL_INTERVAL_RESCUE`, flag-OFF byte-identical). **KILL.** The rescue bound is *rigorous* but structurally *loose*, and — unlike a removed node — a retained node's bound is the **frontier minimum**, so keeping it open *lowers* the global dual bound. nvs05: flag-OFF frontier climbs 1.352→3.405→**4.875** (surviving well-bounded nodes) then stalls; flag-ON the retained failed-NLP node pins the frontier at its interval bound **2.043** and it freezes there for 57 s (reported bound 1.352→2.043 via taint-floor removal, but the *provable frontier* 4.875→2.043 — strictly worse). tanksize identical signature: reported bound unchanged 0.8680; max frontier glb **0.8809→0.8770** (worse). Root cause: nvs05 has 4 **unbounded** (±inf) continuous vars in the objective, so alphaBB abstains (finite-box only) and the interval enclosure over any subtree is loose; branching the 2 integer vars never tightens the unbounded continuous dims. The lever is a **tighter box bound on unbounded-continuous nodes** (e.g. OBBT/bound-inference to make those vars finite, then a McCormick/interval bound that is competitive with the surviving frontier), NOT node retention. Retaining a node under a loose bound is a net dual-bound *regression*. | 2026-07-10, #91 |
| F10 | **Finitizing the unbounded continuous vars on the nvs05/tanksize/casctanks class is the certification lever** (A-RESCUE #91) | FBBT (`discopt.tightening.fbbt_box`) already finitizes **100 %** of the unbounded vars (nvs05 4/4, tanksize 26/26, casctanks 296/296 — the constraints imply a bounded region), and the solver's root OBBT already applies equivalent bounds: FBBT-preconditioning the model closes **0** additional gap end-to-end (nvs05 root 0.674/final 1.3521; tanksize root 0.847/final 0.868; casctanks root −90.2 — all identical to default). nvs05's objective (`1.10471·x0²·x1+…`) contains **no** unbounded var — its 0.674 root bound is a loose interval enclosure of `x0²·x1`, and its final 1.3521 is the F8 taint floor. Lever 2 (`ROOT_FIXPOINT`+`NODE_REDUCE`) is INERT (both gate on `_mc_lp_relaxer`, which this class lacks per-node). nvs05 (8) and tanksize (47) PASS the `n<=100`/`n<=50` gates — the size-gate hypothesis is false; the per-instance disabler is the McCormick-LP root probe on the raw ±inf box (`solver.py:5176`). This is a **Lever-A (relaxation-strength)** class, not bound-inference. Sub-finding banked (not a lever): FBBT before that probe rescues tanksize (`numerical`→0.840) / casctanks (no-bound→−90.2) — engine robustness only, no gap closed. See `docs/dev/a-unbounded-entry-2026-07-10.md`. | 2026-07-10, #94 |
| F11 | **Exact dual steepest-edge (DSE) pricing is the #606/#598 per-node lever** (reference-solver-guide Lever 1, task #99) | **Two-part KILL.** (a) *Premise wrong:* DSE + BFRT are **not** MISSING from `dual.rs` as the guide's §2/Lever-1 claimed — dual **Devex** pricing (a steepest-edge approximation, Goldfarb–Reid reference weights) *and* the **bound-flipping long-step dual ratio test (BFRT)** both landed in commit `299c28eb` (#178, "dual Devex pricing + bound-flipping ratio test"). The guide row was a keyword miss (scanned for "steepest-edge"/"DSE", not "Devex"). (b) *Lever falsified:* implemented exact DSE (`β_i=‖B⁻ᵀeᵢ‖²`, Forrest–Goldfarb 1992 recurrence; unit-tested against a from-scratch recompute to 1e-9) behind `DISCOPT_DUAL_DSE` (default-OFF). On the reproduced #606 pathology (`_make_alpine_multi4n(n=2,exprmode=1)`→`build_milp_relaxation`→`solve`, `DISCOPT_PROFILE`, deterministic across repeats): DSE-OFF (Devex, = the DECOMP-1 baseline exactly) `DualWarmSolves=761`, `DualDegeneratePivots=18511`, `RefacCap=443`, `NodeLpSolve` 459 @ 1694 ms, obj −26.82204470195638. DSE-ON `DualWarmSolves=634`, `DualDegeneratePivots=16710` (−10 %), `RefacCap=1495` (**3.4×**), `Phase1Pivots` 21471→75178 (**3.5×**), `NodeLpSolve` 321 @ **4239 ms (2.5× slower)**, obj −26.822044701956543 (**LP-objective bound-neutral to ~1e-13**, but node-count NOT preserved — 459 vs 321 — so never a silent default). Root cause: Devex is already a *good* DSE approximation (the classic HiGHS DSE win is over **Dantzig**, not Devex); exact DSE's per-solve seeding (m BTRANs) + reseed-on-refactor cost and different leaving-row choices land on bases the FT-update/refactor machinery handles *worse* (RefacCap 3.4×), a net regression. Shipped sound + flag-gated default-OFF with the DSE weight-recurrence and Devex-vs-DSE-agree-on-optimum unit tests; **do not flip the default**. BFRT already present — not a re-add. | 2026-07-10, #99 |
| F13 | **The MINLPLib ex6_2 Gibbs/log-sum objective is a sum of convex atoms (whole objective convex), so joint outer-approximation (OA) cuts of the full objective collapse the ~300× root gap that summing per-atom tangents leaves** (SOTA-P1 A2, task #98) | **KILL — premise is false; the objective is nonconvex.** The Gibbs free-energy objective carries Wilson activity terms `−x_i·log(a·x+b·y+…)` (the 24 `neg(...)` nodes in ex6_2_5); `−x·log(affine)` is nonconvex at **100 %** of box points in isolation, and the whole objective has a negative Hessian eigenvalue at **78 %/92 %/99.5 %** of box points on ex6_2_5/9/10 (min eig −11.3/−181/−76). A gradient/OA cut of a nonconvex function is **not** a valid underestimator — joint-OA is *unsound*, not merely loose, and cannot be built (certificate-invariant violation). The sound joint alternative, αBB over the whole objective, is **~1e40–1e52× worse** (not tighter): rigorous α (interval-Gershgorin) is ~1e40–1e49 because the `x·log(x)`/`log(x)` terms have `~1/x` Hessian entries that blow up at the `x→1e-7` box edge, so the joint-αBB box-min is ~1e40–1e52 below the per-atom root bound of −89 397. **Per-atom relaxation is used *because* it is the only tractable sound handling of this singular structure — a joint convex underestimator is dominated by the worst single-variable curvature and loses.** ex6_2_5's objective has **no `centropy` token** (raw `log(x/Σ)·x` + `x·log(x)` + `neg(x·log(affine))`; 49 `log(`, 24 `neg(`) — RELAX-1's centropy atom never matches this `.nl` form. **Residual looseness is 100 % in the objective** (constraints are linear equality mass-balances, e.g. `x0+x1+x2 = 40.3071`; 0 nonlinear) but it is **not** the per-atom-vs-joint split hypothesized — it is the intrinsic difficulty of underestimating the **nonconvex `1/x`-singular `x·log(affine)` bilinear-of-log** terms. Real lever: a tighter **per-atom** spatial convex/concave envelope for `x·log(affine)` (Lever A on that composite), consistent with DECOMP-1 §5(ii). No code shipped (joint-OA unsound; joint-αBB a strictly-worse dead flag); no default changed. Entry-experiment root_gap/box-min/α table + reproduction: `docs/dev/p1a2-gibbs-log-sum-oa-entry-2026-07-11.md`. | 2026-07-11, #98 |
| F14 | **tls2's primal miss (global50 only PRIMAL fail: dual 91 % closed, NO incumbent) is a heuristic wiring gap fixable by a small primal-heuristic addition (Fischetti objective feasibility pump / wiring the spatial-path integer-lattice search into NLP-BB)** (SOTA-P4, task #100) | **KILL — feasibility is a genuinely hard combinatorial MIP subproblem the whole existing arsenal cannot crack in-budget; the primal miss is downstream of relaxation weakness (Lever A), not a wiring gap.** Reproduced (TL=60): `time_limit`, obj `None`, 1631 nodes, 33 ints (31 binary + x4,x5∈[1,100]), 4 cont. Structure: obj **linear**, 22/24 constraints **linear** (incl. 6 equalities), only con0/con1 nonlinear (bilinear `x4·x0 + x5·x1`-family trim-loss coupling). Diagnosis — every finder run-and-fails, none is merely un-wired: (a) NLP-BB root heuristics that DID fire — feasibility_pump (×2) and fractional_diving (×2) — both returned None; RENS declined (10/33 ints fractional at the relaxation, max frac 0.45). (b) The spatial-path node suite is NOT the missing piece: forcing `nlp_bb=False` (full `integer_local_search`/`subnlp`/`enumerate`/diving node suite) **also finds no incumbent** (bound 4.3, 1375 nodes). (c) `integer_local_search` standalone (the "relaxation ints violate TRUE constraints" heuristic): None in 1.8 s (stalls at violation local minima; subnlp repair fails every restart). (d) A true **Fischetti objective feasibility pump** (squared-distance-to-rounding objective, continuous relaxation re-solve, prototype): **cycles immediately** — parks at a fixed fractional vertex (12 frac, rounded violation 37.6) that the distance projection cannot escape; None in 30 s / 40 rounds. (e) RENS with `max_free=40` (fix the ~21 integral binaries, solve the whole fractional neighbourhood as a 15 s nested sub-MINLP — the strongest MIP-feasibility move available): None in 39.5 s. (f) 651-way random integer-assignment multistart + full continuous subNLP completion: **0 feasible** in 30 s. The feasible region is a combinatorially isolated set of trim-loss pattern selections; continuous completion of a wrong integer assignment is always infeasible, and the weak McCormick/NLP relaxation parks binaries at ~0.37 so rounding never lands in-basin. **External oracle confirms the kill:** SCIP solves tls2 to `optimal` in **0.21 s** (`scip_join.csv`) and BARON at the **root, 0.0 s** — both via their MIP-feasibility machinery (strong LP relaxation + cuts + full MIP presolve + LP-based diving/pump), exactly the kill-criterion's "BARON/SCIP also lean on their MIP feasibility machinery." The lever is a real MIP-feasibility build on a **stronger integer relaxation** (cut-strengthened LP root that makes the rounding neighbourhood feasible-reachable), i.e. Lever A / a cut+presolve+LP-diving stack — NOT a primal-heuristic wiring patch. No code shipped (any heuristic added would be inert on this class → a dead flag). See `docs/dev/p4-tls2-primal-entry-2026-07-11.md`. | 2026-07-11, #100 |
| F12 | **FBBT-finitizing the box before the McCormick-LP root probe re-engages the strong relaxation on nvs05/tanksize/casctanks** (SOTA-P1 A1, task #97; the F10 banked sub-finding) | **KILL — inert on the real solve path.** Prototype flag `DISCOPT_FBBT_BEFORE_ROOT_PROBE` (sound Rust FBBT via `tighten_root_bounds_with_fbbt`, tightening only the probe box → the `_mc_lp_relaxer` liveness decision, never the per-node box). Full solve (TL=60, ON vs OFF): certified bound + root_bound **byte-identical** on all three (nvs05 1.3520892806701879 / root 0.6740; tanksize 0.8680315476476343 / root 0.8473; casctanks 1.3522533305715752 / root −90.18); node_count differs by ≤ timing noise (casctanks 25==25). The F10 isolated raw-box probe failure **does not reproduce in-situ**: (a) **casctanks** — the factorable-reform pre-reform FBBT (`solver.py:3841-3871`) + domain reductions already finitize the probe box (in-situ n=560, 0 inf), so the probe is already `optimal`/−90.18 **useful** OFF → FBBT-before-probe is a no-op; the relaxer is live but OBBT is *size*-gated (n=560>100), unrelated to boundedness; (b) **tanksize** — FBBT does finitize the probe box (26 ub-inf → 0), but the *reformed* lifted McCormick LP on that finite box then trips the dense-cell densification guard (`_MAX_RELAX_DENSE_CELLS=1e8`, `mccormick_lp.py:737`, #20) → `unbounded`→`skipped_oversize`, **still useless** (relaxer nulled either way); (c) **nvs05** — probe already `optimal`/0.674 useful OFF, relaxer live regardless. So the "root probe on the raw ±inf box disables the relaxer" mechanism is not the binding disabler in the real solve for this class. No code shipped (a flag changing nothing is a dead flag). Re-scope: tanksize is blocked by the dense-cell guard on the finite reformed lift (needs a sparse node-LP path or tighter lift — separate axis), casctanks is OBBT-size-gated with a live relaxer; the class lever remains **Lever A (per-node relaxation strength)** per DECOMP-1 §5, not probe wiring. See `docs/dev/a1-fbbt-before-probe-entry-2026-07-10.md`. | 2026-07-10, #97 |

## §7 Progress ledger (update per task; falsifications also to perf-plan §6)

| task | state (2026-07-10) | verdict/notes |
|---|---|---|
| A-RESCUE (#91) NLP-fail interval bound | **entry KILL** (see §6 F9) | rescue LOWERS the frontier (nvs05 4.875→2.043, tanksize 0.881→0.877): a retained loose-bound node is the frontier minimum. Re-scope to a tighter box bound on unbounded-continuous nodes (OBBT), not node retention. No code shipped (flag would be a dead/harmful lever). |
| DSE pricing (#99) exact dual steepest-edge | **KILL** (see §6 F11) | premise false (Devex+BFRT present since #178, not MISSING); exact DSE regresses the reproduced #606 pathology (2.5× wall, RefacCap 3.4×, −10 % degenerate pivots only) and is not node-count-neutral. Shipped sound + `DISCOPT_DUAL_DSE` default-OFF with unit tests (weight recurrence vs recompute; DSE-vs-Devex same optimum); default stays OFF. |
| A2 (#98) Gibbs/log-sum joint-OA | **entry KILL** (see §6 F13) | premise false: the ex6_2 objective is nonconvex (78–99.5% of box has a negative Hessian eig; the `−x·log(affine)` Wilson terms are 100% nonconvex), so joint-OA cuts are *unsound*, not just loose. Joint αBB (sound) is ~1e40× *worse* (`1/x` Hessian singularity → α~1e40). Constraints are linear mass-balances (0 nonlinear); 100% of looseness is the objective, but the lever is a tighter *per-atom* `x·log(affine)` envelope, not a joint cut. No code shipped. See `docs/dev/p1a2-gibbs-log-sum-oa-entry-2026-07-11.md`. |
| A1 (#97) FBBT-before-root-probe | **entry KILL** (see §6 F12) | inert on the real path: certified bound + root_bound byte-identical ON vs OFF on nvs05/tanksize/casctanks. F10's isolated raw-box probe failure does not reproduce in-situ — casctanks already finite upstream (no-op), tanksize's FBBT-finitized box hits the dense-cell guard (`skipped_oversize`), nvs05 already useful OFF. No code shipped (dead flag). Class lever stays Lever A (relaxation strength). |
| SOTA-P4 (#100) tls2 primal | **entry KILL** (see §6 F14) | tls2 feasibility is a hard combinatorial MIP subproblem; the entire existing heuristic arsenal (pump, Fischetti objective-pump, integer-lattice search, diving, RENS max_free=40, 651× random+subNLP) finds no incumbent in-budget on either the NLP-BB or spatial path. SCIP solves it in 0.21 s / BARON at root via MIP-feasibility machinery over a strong relaxation. The primal miss is downstream of relaxation weakness (Lever A), not a wiring gap. No code shipped (added heuristic would be inert → dead flag). Follow-on scoped: MIP-feasibility stack (cut-strengthened LP root + LP-diving/pump). |
| A-2 dense retry | **in flight** (agent) | entry green (prototype cures nvs21); building |
| BR-1 merge + root_fixpoint | **in flight** (agent) | verdict 1/3 green (#581: −14% nodes, 0 loss) |
| BR-2 a–e | queued (blocked BR-1) | — |
| BR-3 re-gates | queued (blocked A-2+BR-1) | — |
| CUTS-1 c-MIR | **in flight** (agent) | entry experiment running |
| CUTS-2 root separate-stage | queued | — |
| STRUCT-1 CSE/V-seg | **in flight** (agent) | entry experiment running |
| STRUCT-2 Q-extraction | queued | — |
| OVERHEAD-1 startup floor | **in flight** (agent) | profiling |
| TAIL-1 a/b/c | queued | do (c) first |
| RELAX-1 centropy tangent cuts | **entry green → GO, implemented** | ex6_2_5 root bound `None`→−27791 (finite, valid ≤ oracle −70.75); 6/8 `ex6_2_*` unlock feasibility-fallback → valid bound; neutral on 30 non-centropy instances; PR open |
| V-2 final validation | blocked (all) | baseline §1 banked |
