# TX series — closing the 10× vs BARON/SCIP, and nothing else

Status: **ACTIVE** (2026-07-14). Successor to the EP and PF series
(`engine-performance-plan.md`, `sota-proof-plan.md`) and the direct answer to
the 2026-07-14 retrospective: three days of work moved the vendored panel
36→42 proofs but left the end metric — wall-clock ratio vs BARON/SCIP —
essentially untouched, because effort went to subsystems worth <2 % of wall
while the documented family-specific causes sat unaddressed in this repo's own
plan docs.

This plan optimizes exactly one thing: **the measured wall-clock ratio vs the
recorded reference solvers, per instance family.** Nothing lands unless it
moves that number for the instances it names.

## §0. Operating rules (binding — these encode the retrospective's failures)

1. **The metric is the ratio table, not a profile.** Every item must name, *in
   advance*: the instances it targets, the seconds it predicts to recover, and
   the family it generalizes to. The exit gate is that named prediction,
   measured end-to-end. A landed item that does not move its named instances
   is **reverted**, even if harmless — accretion without effect is cost.
2. **Do-not-relitigate list (§2).** No item may re-test a hypothesis already
   killed in `performance-plan.md` (incl. Appendix B's kill list),
   `sota-proof-plan.md` §3, `scip-gap-nvs-diagnosis.md`, or
   `engine-performance-plan.md` — without *new data* that names what changed.
3. **Read-before-derive.** Any new diagnosis starts by citing the existing
   doc(s) covering that territory. The 07-14 retrospective showed hours of
   re-derivation of `scip-gap-nvs-diagnosis.md`, `nlp-solve-volume-2026-07-06.md`,
   and both `bottleneck-profile-*.md`; that is a process failure, not research.
4. **The gap is family-specific — one family per item.** "The 10×" decomposes
   into (at least) four different problems with four different fixes (§1). No
   more single-instance profiles generalized to "the lever".
5. **Measurement discipline.** jobs=1 serial; dual-bound-at-timeout;
   variance-flagged instances (tls2-class wall-budget bimodality) get 2+
   repeats per arm; node counts under contention are never evidence (PF3 rule).
6. **Correctness gates unchanged** (CLAUDE.md §5): byte-neutral items carry
   the fingerprint gate; bound-changing items carry differential + feasible-point
   (0 cuts) + full-panel no-regression. `incorrect_count ≤ 0` with zero slack.
7. **Authoritative numbers come from the maintainer's box** (recorded-BARON
   easy-class harness, `global50`; see `performance-plan.md` Appendix B and
   `config/baron_global50.txt`). In-container proxies: the 62-instance vendored
   panel (`pf_panel.py`, baseline `docs/dev/data/pf-baseline.json`) and the
   Appendix-B floor probes. BARON is not runnable in-container — never present
   an in-container wall as a ratio claim.
8. **One item in flight at a time**, in §4's order. The verdict (GO/KILL/
   REVERTED) is committed to §3's table in the same change as the code.

## §1. What "the 10×" actually is (measured, cited — not assumed)

The single number the goal tracks: **geomean wall ratio vs recorded BARON on
the easy-class panel = 10.3–10.4×** (30 instances BARON proves in <1 s;
measured 2×2 A/B on 2026-07-10, `performance-plan.md` Appendix B). The
codified target already exists: `benchmarks.toml` gate
`geomean_vs_baron ≤ 2.5`. But that geomean is only one of four distinct
sub-problems:

| family | instances (named) | measured state | binding constraint | evidence |
|---|---|---|---|---|
| **A. Easy class (floor-dominated)** | the 30 BARON-<1 s instances; trivial: ex1222, st_test1, gbd, alan, nvs01 | median wall 0.57–0.60 s vs BARON <1 s → geomean 10.3×; nonlinear-class in-window floor ≈ 0.55–0.65 s (jax import 240–300 ms, pounce 125–150 ms, first trace ~75 ms, recurring engine ~150 ms) | **fixed per-solve floor + engine constant**, not search | performance-plan.md Appendix B (OVERHEAD-1) |
| **B. Mid class (proved but slow)** | nvs09, nvs05, st_e36, tspn05, fac2, flay03m | proofs land in 10–30 s where reference is ~1 s; measured on 07-14: node-NLP heuristic OFF halves nvs09 (30→15 s) with identical proof/bound; budget overruns: contvar runs 51–82 s on a 30 s limit, casctanks 68 s, heatexch_gen3 68 s, bchoco08 57 s | **wall burned on zero-bound-stake heuristics + unhonored time limits** | this session's stride experiment (`scratchpad/nlpoff_*.json`); PF5 record |
| **C. Integer-product family (timeout; SCIP seconds)** | nvs17/19/24 (closed #283 but the class stands), graphpart_* (#280) | 0.9 nodes/s vs SCIP ~1,400 no-cut; dual bound frozen at root; LP-node engine EXISTS opt-in (`lp_spatial_bb.py`, PR #290, incremental LP + warm basis + node cuts) but is not default and carries #287 (first incumbent 6 s→12.9 s, soft limit overrun) | **per-node NLP where a pure LP suffices** + engine not routed + OBBT gated off for pure-integer (`solver.py` `_obbt_has_continuous`) | scip-gap-nvs-diagnosis.md (all numbers measured there) |
| **D. Spatial tail (bound frozen or absent)** | heatexch_gen1/2/3, bchoco06/07/08, casctanks, 4stufen, beuster | heatexch: root gap 75 %, LMTD ε-pole **inside the box** → relaxation is *correctly* unbounded (PF4); bchoco06: **no finite bound at all** at 7 nodes (undiagnosed); FBBT (PF1) landed, inert here — stuck *before* reduction bites | **relaxation strength / structural holes**, not speed | sota-proof-plan.md PF4 §3; pf-baseline panel rows |

Families A+B move the geomean directly. Family C flips timeouts to solves
(∞→finite — the ratio's worst term). Family D is research-grade; it gets
entry experiments, not promises.

## §2. Do-not-relitigate list (verbatim citations; new data required to reopen)

- **Persistent JAX compilation cache; lazy-import surgery on `import discopt`;
  deferring JAX init; no-JAX fast path for linear/quadratic** — all killed or
  already-done per Appendix B's ≥20 %-of-floor criterion.
- **"XLA recompilation is the dominant cost"** — falsified
  (performance-plan.md §3, validation patch); confirmed again 07-14
  (steady-state solve = 1 XLA compile).
- **Per-node relaxation rebuild cost / "cheaper node LP"** — already cached
  (EP1) and already incremental+sound (PF2 differential green); EP3's
  patch-table shortcut was UNSOUND and is reverted — do not reintroduce.
- **Branch-point / reliability-threshold tuning** — inert on the stuck set (PF3).
- **A finite LMTD envelope on pole-straddling boxes** — UNSOUND, guard test
  pins it (`test_pf4_lmtd_epsilon_pole.py`). Only pole-*excluded* sub-box
  routes are admissible (TX6).
- **Blanket warm-LP deadline** — regressed 5 proofs by truncating OBBT probes
  (PF5). Only the surgical form (TX3) is admissible.
- **Single-row Gomory/MIR on the McCormick LP** — measured plateau (~2 % bound
  movement, nvs17); the missing piece is an aggregation/c-MIR separator, which
  is **deferred** behind TX5, not part of this plan's critical path
  (scip-gap-nvs-diagnosis.md "Step 3 findings").
- **jit for the separation grad** — nondeterministic (tls2 2.449↔2.1);
  superseded by the landed, gated analytic path (F2′). The separation grad is
  <0.6 s/solve — no further work there, period.

## §3. Items

Verdicts land in this table as they happen.

| item | family | what | entry experiment (before building) | kill criterion | gate | verdict |
|---|---|---|---|---|---|---|
| **TX0 attribution table** | all | One table, one session: for every 62-panel instance + the 5 trivial floor probes, decompose wall into {floor, root build/presolve, node LP, node-NLP heuristic, other} using the existing EP0 probe harness + Appendix-B method, and stamp each instance with its §1 family. Publish `docs/dev/data/tenx-attribution.json`. No fixes in this item. | n/a (it *is* the measurement) | n/a | table committed; every later item must cite its rows | — |
| **TX1 adaptive node-NLP default** | B | The node-NLP is a primal heuristic (bound comes from the LP relaxer; code comment solver.py:6613) costing up to 60 % of wall on stuck trees. Measured: stride→∞ gives identical proofs/bounds, nvs09 30→15 s, st_e36 17→14 s. The `fast` profile already ships stride 8 — the *default* (4) is too eager. Make the default adaptive: always at root + when the last K attempts improved the incumbent; exponential back-off when they don't. | rerun the 07-14 stride experiment on the full 62 panel (stride 4 vs adaptive prototype), jobs=1 | any lost proof or looser bound (2 repeats on flagged instances) | full panel: proofs ≥ 42, no LOOSER/CROSSED, total wall strictly down | — |
| **TX2 honor the time limit** | B | Budget overruns are pure ratio loss: contvar 51–82 s, casctanks 68 s, heatexch_gen3 68 s on a 30 s limit — the benchmark charges us the overrun. PF5's blunt fix is falsified; the surgical form: per-phase deadline checks in the node loop + OBBT probe LPs get their **own sub-budget** so range reduction is never truncated by the outer deadline. | instrument where the overrun accrues on contvar/casctanks (which phase ignores the deadline) | the PF5 trap: any bound looser on the panel | panel: no instance exceeds budget+grace; bounds/proofs unchanged (differential on OBBT-affected instances) | — |
| **TX3 easy-class floor: JAX-free small-model path** | A | The remaining floor (Appendix B, post-sympy-fix) is jax import 240–300 ms + first trace ~75 ms + pounce import ~140 ms on a ~0.6 s median — ~40–50 % of easy-class wall. Appendix B calls `import jax` "out of repo", which assumes JAX must be imported. F2′ proved the engine's own IR + `interval_ad` can produce values/gradients JAX-free. Hypothesis: for models under a size/atom threshold, the full root relaxation + B&B can run with **no JAX import at all** (interval_ad envelopes + Rust simplex + POUNCE-native NLP where needed). | count how many of the 30 easy-class instances use only interval_ad-covered atoms; hand-build a JAX-free root bound on 3 of them and compare bound + build time vs the JAX path | <15/30 instances qualify, or JAX-free root bound is looser, or build >2× slower | byte-level: bounds/nodes identical to the JAX path on qualifying instances (it computes the same envelopes); easy-class A/B geomean strictly down; panel no-regression | — |
| **TX4 route family C to the LP-node engine** | C | All parts exist: `lp_spatial_bb.py` (incremental warm LP, node cuts, opt-in via #290), the scope test, OBBT (contracts [0,200]→[0,40] on nvs17 when un-gated). Work: (a) fix #287 (dive/primal incumbent surfaced early; soft limit honored between phases — overlaps TX2); (b) un-gate root OBBT for pure-integer models (scip-gap plan Phase 1; predicted nvs17 root −65,842→−6,790); (c) route in-scope integer-product models to the engine by default. | reproduce #287 (kall_congruentcircles_c72 first-incumbent 12.9 s vs 8 s limit); rerun the engine on nvs17/19/24 at current HEAD to refresh the Step-1 numbers | engine (with OBBT + latency fix) fails to beat the default path on its own family, or any out-of-family panel regression | family gate: nvs17/19/24 solved or bound within 2× of true (SCIP no-cut solves in 4.7 s — that is the parity bar, not 0.88 s); graphpart improves; full panel + differential green; #286-class false-unbounded regression tests pass | — |
| **TX5 c-MIR / aggregation separator** | C | **Deferred.** The 97× node multiplier, but scip-gap diagnosis shows throughput (TX4) must come first — no-cut SCIP still solves the family. Opens only if TX4 lands and its family gate shows node-count (not throughput) binding. | — | — | — | deferred |
| **TX6 spatial-tail entry experiments** | D | Two bounded diagnoses, no build commitment: (a) **bchoco06 has no finite dual bound** — find the unbounded relaxation hole (which term class, why; 1 session); (b) **heatexch pole-excluded sub-boxes** — PF4's only sound route: branch once on the `a=ε+b` pole line, measure the root bound of the two pole-free children with the (sound-there) AM/GM envelope, by hand. | (a) and (b) *are* entry experiments | (a) hole is structural à la LMTD with no sound finite bound → document, close; (b) child-box bound improves gen1's 38,184 root by <10 % → KILL the route | any build that follows gets the full bound-changing gate | — |

## §4. Sequencing, arithmetic, and the stop condition

**Order: TX0 → TX1 → TX2 → TX3 → TX4 → TX6 (TX5 deferred).**
TX1/TX2 are low-risk measured wins that also make every later measurement
cleaner (no heuristic noise, no overruns). TX3 attacks the recorded geomean's
largest component. TX4 is the biggest single family flip with mostly-built
parts. TX6 is research and goes last deliberately.

**Honest arithmetic (predictions to be checked against TX0, not adjectives):**
- TX1: mid-class wall −30…−50 % where the heuristic is idle waste (nvs09-class);
  no new proofs by itself; frees budget inside fixed limits so some timeouts
  may flip.
- TX2: removes 20–50 s of charged overrun on ≥4 tail instances; ratio-table
  effect immediate.
- TX3: if the entry holds, easy-class nonlinear floor drops ~0.3–0.45 s of a
  ~0.6 s median → geomean 10.3× → **plausibly 4–6×**. This is the single
  largest predicted mover of the tracked number.
- TX4: family C from timeout/frozen to solved-in-seconds (∞→finite terms in
  the ratio table); parity bar is no-cut SCIP (4.7 s), not full SCIP.
- TX6: no prediction — that is the point of entry experiments.

**Stop condition (anti-circling):** after TX4, re-measure the easy-class
geomean and the panel on the maintainer's box. If geomean ≤ 2.5× (the
benchmarks.toml gate), the plan closes. If not, the residual must be
*attributed* (TX0 refresh) to named instances/causes and becomes new named
items — no unattributed grinding, no profile-of-the-day.
