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

1. **The metric is the ratio table, not a profile — but it gates STACKS, not
   items.** The repo's own data forbids per-item end-metric gating:
   `scip-gap-nvs-diagnosis.md` measured that OBBT alone leaves nvs17 6× loose,
   cuts alone are useless at 0.9 nodes/s, throughput alone can't close loose
   McCormick — "*they are multiplicative and must be addressed together*." A
   solo revert-if-no-effect rule would kill necessary-but-insufficient
   components (and past try-and-revert cycles did exactly that; reverting also
   destroys the component, making the next composition attempt archaeology).
   So gating is two-level:
   - **Mechanism gate (per item):** did the item do the mechanical thing it
     claims (calls drop, phase respects deadline, engine routes, box
     contracts) *and* pass its soundness regime. This admits the component —
     behind a default-OFF flag when bound-changing (the
     `flag-graduation-protocol.md` convention), directly when byte-neutral or
     zero-bound-stake. No end-metric demand at this level.
   - **Stack gate (per family checkpoint, §5):** the named family metric is
     evaluated for the *composed stack*, with attribution by **ablation arms**
     (stack minus one component, reusing `generality_sweep.py`'s arm
     machinery) — a component's value is its marginal contribution *in
     context*, not solo. A component with zero marginal contribution in the
     full stack is then removable; that replaces revert-on-no-solo-effect.
   - **Revert** is reserved for regressions: unsoundness, lost proofs, looser
     bounds, or metric-worsening. Never for "no solo effect."
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
| **TX0 attribution table** | all | One table, one session: for every 62-panel instance + the 5 trivial floor probes, decompose wall into {floor, root build/presolve, node LP, node-NLP heuristic, other} using the existing EP0 probe harness + Appendix-B method, and stamp each instance with its §1 family. Publish `docs/dev/data/tenx-attribution.json`. No fixes in this item. | n/a (it *is* the measurement) | n/a | table committed; every later item must cite its rows | **DONE (2026-07-14)** — `data/tenx-attribution.json`, 62 rows (16 cProfile'd, rest family-extrapolated; wall/status/bound/nodes from the f2p jobs=1 30 s panel). **Families:** **A** 22 inst / 27.9 s (floor 41 %, node-NLP 42 %; all 22 proved) — floor-bound; **B** 14 / 226.7 s (node-NLP 70 %, sep 14 %, presolve 12 %; 10 proved / 4 feasible) — node-NLP-bound, separation concentrated in nvs09; **C** 10 / 58.6 s (node-NLP 77 %; **9 proved, only nvs12 times out**) — native per-node NLP where an LP suffices; **D** 16 / 556.9 s (node-NLP 69 %, other/engine 16 %, presolve 12 %; 11 timeout / 4 feasible / 1 hang). **Panel-wide split of ~870 s:** node-NLP **600.6 s (69 %)**, root presolve/build 103.6 s (12 %), other/engine 91.2 s (10 %), separation 40.3 s (5 %), floor 34.5 s (4 %), **node-LP 0.5 s (0.06 %)**. **Findings:** (1) the node LP is *not* the cost anywhere (0.06 %); per-node NLP dominates every family — confirms PF2. (2) that node-NLP is mostly the per-node relaxation *solve* (bound source), not skippable heuristic: measured skippable waste (`nlpoff/nlpdef` stride) is 14.3 s on nvs09 (integer ⇒ whole node-NLP removable → TX1) but only 1.2 s heatexch_gen1 / 0 s bchoco06 (the D-tail node-NLP is the relaxer → TX3/TX4, not TX1). (3) **§1 correction:** the integer-product class (C) is *not* stuck on the vendored panel — nvs17/19/24 are absent; 9/10 in-panel C members prove (0.25–9.3 s), only nvs12 shows the frozen-bound pathology, so TX4's family-C win is barely visible here. (4) **§1 correction:** nvs05/nvs09/tspn05 are *feasible, not proved* at HEAD (default-off F2′). (5) presolve/build is a top cost on casctanks (15.5 s) and st_e36 (8 s), larger than §1's 'relaxation-strength' framing for D. **Top single attributions (all node-NLP):** contvar 43 s, heatexch_gen1 39 s, bchoco08 36 s, heatexch_gen3 32 s, tanksize 29 s, nvs12 29 s. |
| **TX1 adaptive NLP-heuristic budget** | B | **RE-SCOPED by TX0.** TX0 falsified the original "node-NLP = 60 % of wall, zero bound stake" framing: the 600 s / 69 % node-NLP bucket is *mostly the non-skippable per-node relaxation solve*, and only **2 instances** carry `binding_constraint=heuristic_waste`. The `DISCOPT_NODE_NLP_STRIDE` knob touches only the strided *in-tree* node-NLP (nvs09 14.3 s skippable; heatexch_gen1 1.2 s; bchoco06 0 s) — a small, integer-family-concentrated slice. The real primal-heuristic budget is the *set* (`_solve_root_node_multistart`, feasibility_pump, integer_local_search, subnlp — the ~600 s minus the bound-source solve). Re-scoped item: adaptive back-off across **all** NLP primal heuristics keyed on incumbent improvement, not just the node-NLP stride. Entry experiment must first *measure the addressable set* per family (TX0 only stride-tested 3 instances). | on the full 62 panel, per-heuristic on/off attribution (which of {multistart, fpump, ils, subnlp, strided node-NLP} is idle waste, by family) — do NOT assume it's 60 % | addressable idle-heuristic wall < 40 s panel-wide (then TX1 is not worth the risk — demote), OR any lost proof / looser bound | full panel: proofs ≥ 42, no LOOSER/CROSSED, wall strictly down on the instances it names | **GO — LANDED gated (2026-07-14)** (`DISCOPT_ADAPTIVE_NLP=1`, default OFF; `solver.py` `solve_model` node loop + `solver_tuning.adaptive_nlp`; regression `test_tx1_adaptive_nlp.py`). **Scope:** re-scoped again — TX0's addressable set is the *strided in-tree node-NLP* (a pure primal heuristic gated to nonconvex + LP-relaxer nodes, `_gate_node_nlp`); the root-fire heuristics (multistart/fpump/ils/subnlp) fire once and are not a call-volume waste. Mechanism: effective stride starts at the base (4), doubles (cap 256) after 2 consecutive fired-but-non-improving batches, resets to base the instant the node-NLP improves the incumbent. Convex / no-LP-relaxer bound-source path untouched (byte-identical, verified). **Entry experiment** (full 62 panel, node-NLP fully OFF in the gated regime vs default): addressable idle-waste is family-B-concentrated — nvs09 +14.7 s (feasible→proved), tspn05 +7.3 s (feasible→proved), st_e36 +5.2 s, fac2 +2.3 s (family B 226.7→199.4 s). Panel-wide < 40 s addressable — BUT blanket-off is **unsound/harmful**: tls2 bound LOOSER (2.10 vs 2.449), tspn12 feasible→timeout (lost incumbent), nvs12 +40.5 s / heatexch_gen3 +18.3 s slower, net wall 870→893 s WORSE. This is why the mechanism is *adaptive* (reset-on-improvement), not blanket-off. **Soundness gate (`--vs` flag-ON vs `f2p_full_eval.json`): GREEN** — proofs 41→**43** (gained nvs09, tspn05; **0 lost**), **0 LOOSER/0 CROSSED**; tls2 bound byte-identical to reference across 2 repeats (adaptive keeps its node-NLP productive where full-off loosened it); nvs09 stable-proved across 2 repeats. **Benefit gate: PASS** — total wall **870.2→824.6 s (−45.6 s)**; family B −16.0 s; named nvs09 −3.6 s + proof, st_e36 −5.1 s, tspn05 +proof, fac2 −1.8 s. Composes with a future TX2 deadline scheduler (Stack B); landed standalone. |
| **TX2 honor the time limit** | B | Budget overruns are pure ratio loss: contvar 51–82 s, casctanks 68 s, heatexch_gen3 68 s on a 30 s limit — the benchmark charges us the overrun. PF5's blunt fix is falsified; the surgical form: per-phase deadline checks in the node loop + OBBT probe LPs get their **own sub-budget** so range reduction is never truncated by the outer deadline. | instrument where the overrun accrues on contvar/casctanks (which phase ignores the deadline) | the PF5 trap: any bound looser on the panel | panel: no instance exceeds budget+grace; bounds/proofs unchanged (differential on OBBT-affected instances) | **STOP — NOT LANDED (2026-07-14)** (`tx2-honor-deadline-spike.md`). Entry experiment (phase timers + caller lines, `scratchpad/tx2_phase_probe.py`/`tx2_callsite.py`) falsified the premise. All 4 overrun instances process **nodes=1** — the overrun is inside root/first-node processing, not the between-node loop (which already gates launches, `:6471`/`:6592`). **Phase→overrun:** contvar = 21 s discarded root **probe** `solve_at_node` @`:5265` (handed `time_limit=3.0s`, ignored) + 8 s #138 root-fallback bound @`:2631` (the reported bound, also overruns); casctanks = 48 s root **OBBT** @`:4721` — one persistent probe LP (`solve_lp_warm_csc_py`, no wall deadline) spins ~33 s and feeds the −99.09 bound; bchoco08 = 12 s node-LP dual bound; heatexch_gen3 = ~50–77 s POUNCE batch node-NLP overrunning its `max_wall_time` clamp (uninterruptible XLA compile, F4). **Common cause:** the Python scheduler already passes correct per-phase budgets everywhere (`solve_at_node` even builds a deadline, `mccormick_lp.py:1223`); the **native** solver (Rust simplex / POUNCE) ignores them — the exact PF5 seam. **PF5 trap reproduced** (`tx2_obbt_sens.py`, casctanks OBBT probe iter-cap): bound **−99.09 → +5.70**, wall 78→105 s — truncating the OBBT/dual-bound solve changes the bound chaotically. The overrun IS bound-producing native math; the only mechanism (blanket native truncation) regresses bounds. The seam is not cleanly separable at the layer the fix must live (one shared untimed `milp.solve` across discarded-probe + dual-bound + OBBT-probe). Real fixes = PF5 open #1 (OBBT-probe budgeting, casctanks) + #4 (contvar simplex conditioning) + a native deadline in Rust simplex/POUNCE — out of TX2's Python scope, unsafe as a blanket. Per the kill criterion + CLAUDE.md §1: land nothing. |
| **TX2b native-deadline engine (TX2 follow-up)** | B | Ship the native piece TX2's STOP named: an *optional* per-call wall-clock deadline in the Rust LP/MILP simplex bindings (`solve_lp_warm_csc_py`/`solve_lp_warm_py`), returning a valid non-improving `IterLimit` on expiry. Then wire it into TX2's one *discarded* caller — the root probe `solve_at_node` @`solver.py:5265` (result only sets `_probe_useful`) — for contvar; diagnose hda's HEAD hang. Invariant: a deadline may touch ONLY a discarded/zero-stake call, NEVER an OBBT/dual-bound/#138 solve (the PF5 trap). | rebuild the extension; verify the probe result is discarded (trace `_probe_useful`); cProfile hda to locate the hang phase | any bound/node changed on the panel = a non-discarded site got truncated | full 62-panel `--vs`: bounds+nodes **byte-identical** except the deadline-affected walls; contvar wall down; no regression | **PARTIAL — engine landed, Python activation FALSIFIED (2026-07-14).** **Landed (byte-neutral):** the Rust native deadline — `deadline_s: Option<f64>` on both warm-LP bindings (`crates/discopt-python/src/lp_bindings.rs` + `native_deadline` helper); the core primal/dual loops already poll `SimplexOptions::deadline`. `None` = byte-identical (verified: `deadline_s=None`→`optimal`, `=0.0`→`iter_limit`). Core unit test `deadline_fires_returns_iterlimit` (446+4+1 `cargo test` green) + Python regression tests (`test_lp_certificates.py`: None byte-identical, expired→`result is None`). Thin Python surface `solve_lp_warm_std(deadline_s=…)`. **Activation FALSIFIED by measurement (reverted, nothing wired to a caller):** (1) **contvar probe @`:5265` is NOT zero-stake.** The probe's result gates `_probe_useful` (keep-vs-drop the McCormick relaxer). Truncating the warm solve returns `lb=None`→`_probe_useful` flips false→**relaxer dropped**→solve diverts to the POUNCE root NLP: **>100 s XLA-Hessian HANG** and the dual bound (HEAD 183430.5) is LOST — strictly worse than the 36 s HEAD timeout. **Fail-open** (treat a truncated probe as useful, sound since engaging the relaxer only ever adds valid bounds) removes the hang but the bound moves **LOOSER** (183430.5→180782.9) and there is **no wall win** (contvar is budget-capped at 30 s; the freed probe time is re-spent in-budget) — the probe warm solve also seeds the relaxer's warm-start/cut state, so cutting it shifts every downstream node's timeout bound = the PF5 trap. Fails the `--vs` gate → reverted. (2) **hda hang is NOT a native LP.** cProfile/faulthandler: hda hangs in `_build_convexity_box`/`_try_convex_lift` (the convexity-certificate interval-Hessian sweep, pure Python) inside `build_milp_relaxation`, reached via the **bound-producing** #138 root-fallback `_root_relaxation_lower_bound` (`solver.py:2649`). The LP deadline cannot reach it and it feeds the dual bound → leave it. **Net:** TX2's STOP re-confirmed at the caller layer with the native mechanism now built — the deadline is sound, but has **no byte-neutral in-repo caller** (every warm-LP caller is bound-producing or a control-flow gate). Remaining: casctanks OBBT (`:4721`), bchoco08 dual LP, heatexch_gen3 POUNCE/XLA — unchanged, need OBBT-filtering / dual-evaluator / a native POUNCE deadline (separate items). hda relaxation-build cost is a build-optimization item, not a deadline item. |
| **TX3 easy-class floor: JAX-free small-model path** | A | The remaining floor (Appendix B, post-sympy-fix) is jax import 240–300 ms + first trace ~75 ms + pounce import ~140 ms on a ~0.6 s median — ~40–50 % of easy-class wall. Appendix B calls `import jax` "out of repo", which assumes JAX must be imported. F2′ proved the engine's own IR + `interval_ad` can produce values/gradients JAX-free. Hypothesis: for models under a size/atom threshold, the full root relaxation + B&B can run with **no JAX import at all** (interval_ad envelopes + Rust simplex + POUNCE-native NLP where needed). | count how many of the 30 easy-class instances use only interval_ad-covered atoms; hand-build a JAX-free root bound on 3 of them and compare bound + build time vs the JAX path | <15/30 instances qualify, or JAX-free root bound is looser, or build >2× slower | byte-level: bounds/nodes identical to the JAX path on qualifying instances (it computes the same envelopes); easy-class A/B geomean strictly down; panel no-regression | **KILL (2026-07-14)** (`tx3-jax-free-small-spike.md`). Ran step 5 (reachability) first per the item. **Decisive:** `import jax` is **unavoidable on the nonlinear path even with the relaxation swapped**. Easy-class split (family A, 62-panel): **16 nonlinear / 6 MILP-MIQP** (the 6 already jax-free, §2) → addressable set = 16 nonlinear. `interval_ad` *is* jax-free (nvs03 enclosure `[-200,4000]` in ~0.9 ms, no `jax` in `sys.modules`) and ≥14/16 pass atom coverage — but that is **moot**: `import jax` is the *first* op of both nonlinear entry points, `_make_evaluator(model)` @ `solver.py:9179` (`_solve_nlp_bb`) / `:8959` (`_solve_continuous`) → `_jax/nlp_evaluator.py:19` (`import jax`), *before* any relaxation. The evaluator is the **point-mode f/grad/jac provider** for the per-node POUNCE NLP bound source (TX0's dominant cost) + every NLP heuristic + constraint-bound inference (`:9183`) — not the relaxation. `interval_ad` yields *interval enclosures*, not point-mode Jacobians, so it can't replace it; and the McCormick relaxer is itself jax at module load (`relaxation_compiler.py:14`, `dag_compiler.py:33`, `mccormick_lp.py`). Forgone floor = cold `import jax` 0.40–0.43 s here (Appendix B: 240–300 ms), ~40 % of the 0.6 s median — real but unreachable by a relaxation swap. **To remove the floor** needs a jax-free numpy point-mode evaluator (replacing `nlp_evaluator`+`dag_compiler`) + jax-free relaxation builder + routing POUNCE/all heuristics through it, at byte-level parity — an engine rewrite, not this item. Recommendation: **demote TX3**. |
| **TX4 route family C to the LP-node engine** | C | All parts exist: `lp_spatial_bb.py` (incremental warm LP, node cuts, opt-in via #290), the scope test, OBBT (contracts [0,200]→[0,40] on nvs17 when un-gated). Work: (a) fix #287 (dive/primal incumbent surfaced early; soft limit honored between phases — overlaps TX2); (b) un-gate root OBBT for pure-integer models (scip-gap plan Phase 1; predicted nvs17 root −65,842→−6,790); (c) route in-scope integer-product models to the engine by default. | reproduce #287 (kall_congruentcircles_c72 first-incumbent 12.9 s vs 8 s limit); rerun the engine on nvs17/19/24 at current HEAD to refresh the Step-1 numbers | engine (with OBBT + latency fix) fails to beat the default path on its own family, or any out-of-family panel regression | family gate: nvs17/19/24 solved or bound within 2× of true (SCIP no-cut solves in 4.7 s — that is the parity bar, not 0.88 s); graphpart improves; full panel + differential green; #286-class false-unbounded regression tests pass | — |
| **TX5 c-MIR / aggregation separator** | C | **Deferred.** The 97× node multiplier, but scip-gap diagnosis shows throughput (TX4) must come first — no-cut SCIP still solves the family. Opens only if TX4 lands and its family gate shows node-count (not throughput) binding. | — | — | — | deferred |
| **TX6 spatial-tail entry experiments** | D | Two bounded diagnoses, no build commitment: (a) **bchoco06 has no finite dual bound** — find the unbounded relaxation hole (which term class, why; 1 session); (b) **heatexch pole-excluded sub-boxes** — PF4's only sound route: branch once on the `a=ε+b` pole line, measure the root bound of the two pole-free children with the (sound-there) AM/GM envelope, by hand. | (a) and (b) *are* entry experiments | (a) hole is structural à la LMTD with no sound finite bound → document, close; (b) child-box bound improves gen1's 38,184 root by <10 % → KILL the route | any build that follows gets the full bound-changing gate | — |

## §4. Sequencing, arithmetic, and the stop condition

**Re-sequenced by TX0 (2026-07-14).** TX0's binding-constraint histogram over
the 62 panel: **floor 26**, node_count 18, no_bound 8, overrun 4,
bound_frozen 3, **heuristic_waste 2**, throughput 1. This re-orders the plan:
the biggest addressable bucket is the **floor** (TX3, and it *is* the codified
easy-class metric), not the node-NLP heuristic (TX1, now 2 instances). The
29 instances in {node_count 18, no_bound 8, bound_frozen 3} are
relaxation-strength-limited — the genuinely hard core (TX4/TX6), which no cheap
item touches.

**Order: TX0 ✓ → TX2 → TX3 → TX1(re-scoped) → TX4 → TX6 (TX5 deferred).**
Lead with the two highest-certainty, metric-aligned items: TX2 (overrun, 4
named instances, trivial + safe) and TX3 (floor, 26 instances, the codified
geomean). TX1 drops behind them — TX0 shrank its premise from 60 % to 2
instances; it runs only if its re-scoped entry experiment finds a ≥40 s
addressable set. TX4 is the family flip (barely in-container observable per
TX0 — its gate needs the out-of-panel nvs17/19/24 + graphpart). TX6 last.

**Honest arithmetic (now anchored to TX0's numbers, not the pre-TX0 guesses):**
- TX2: removes the charged overrun on the 4 `overrun` instances (contvar 43 s
  over, casctanks/heatexch_gen3/bchoco08 similar) — immediate ratio-table win,
  no proof change; the single most certain item.
- TX3: family A is 41 % floor / 42 % node-NLP (TX0). If the JAX-free entry
  holds, the floor share (~0.25 s of the ~0.6 s easy-class median) drops →
  geomean 10.3× → **plausibly 5–7×**. Still the largest single mover of the
  codified metric, but TX0 shows node-NLP is an equal chunk of family A, so
  TX3 alone does not reach ≤2.5× — it stacks with TX1's easy-class share.
- TX1 (re-scoped): addressable only where NLP heuristics are idle waste;
  TX0 measured this as small + integer-concentrated. Predict < 50 s panel-wide;
  kill-if-<40 s. No longer a headline item.
- TX4: family C is *not stuck in-container* (TX0: 9/10 panel-C prove); the win
  is on the absent nvs17/19/24 + graphpart — measured out-of-panel, ∞→finite.
- TX6: no prediction. But TX0 re-weights it *up*: 29 instances are
  relaxation-strength-limited, the largest hard bucket — the real spatial 10×
  lives here, and it is research, not a lever.

**Stop condition (anti-circling):** after the Stack-C checkpoint, re-measure
the easy-class geomean and the panel on the maintainer's box. If geomean ≤
2.5× (the benchmarks.toml gate), the plan closes. If not, the residual must be
*attributed* (TX0 refresh) to named instances/causes and becomes new named
items — no unattributed grinding, no profile-of-the-day.

## §5. Composition: declared stacks, checkpoints, ablation

Items are NOT independent; the dependency structure is declared here so
composition is designed, not discovered. Components land behind env flags (the
`flag-graduation-protocol.md` convention: `DISCOPT_X=0` always restores old
behavior), so stack checkpoints and ablations are a **flag matrix**, not
reverts and re-lands.

**Stack B (mid-class wall): TX1 + TX2.**
Interaction: TX1 frees wall *inside* a budget that TX2 makes real — with
overruns unhonored, TX1's savings don't change what the benchmark charges; with
the heuristic uncapped, TX2's deadline fires mid-heuristic. Shared
implementation surface: both touch the node loop's phase boundaries — build as
one deadline-aware phase scheduler, not two patches.
*Checkpoint B:* 62-panel jobs=1, arms {off, TX1, TX2, TX1+TX2}: proofs ≥
baseline, no LOOSER, mid-class wall and overrun-seconds strictly down in the
composed arm; per-component marginal from the ablation arms recorded in §3.

**Stack C (integer-product family): TX1 + TX2 + TX4a(#287 latency) +
TX4b(OBBT un-gate) + TX4c(engine routing).**
This is the stack the scip-gap diagnosis calls multiplicative: OBBT tightens
the box the engine's McCormick LP is built on (necessary, insufficient alone);
LP-node throughput exploits the tightened box (insufficient on a loose box);
TX4a's early incumbent enables pruning that throughput makes affordable; TX1
matters here because the engine's scope test falling through to the default
path must not re-enter an unthrottled NLP loop.
*Checkpoint C:* family gate on nvs17/19/24 + graphpart_2g/3pm (bar: solved or
bound within 2× of true at 60 s — parity with no-cut SCIP's 4.7 s, not 0.88 s)
+ full-panel no-regression + differential; ablation arms = stack minus each of
{TX4a, TX4b, TX4c}, with TX1+TX2 held on (they graduate at Checkpoint B and
become part of the control). Marginal contributions recorded; a component with
zero marginal in-stack is removed.

**Independent items (solo gates are correct for these):**
- **TX3** (JAX-free floor): additive cost, no coupling to search behavior —
  its gate is byte-level equivalence + easy-class A/B geomean. Interacts with
  nothing above (it changes *how* the same envelopes are computed, not which).
- **TX0 / TX6**: measurements; nothing to compose.

**Interactions to watch (named now so they're tested, not discovered):**
1. TX2's deadline machinery vs OBBT probes — the PF5 trap lives exactly at
   this seam; the OBBT sub-budget is *part of* TX2's design, and Stack C's
   ablation must include a TX2-off arm on OBBT-heavy instances.
2. TX1's throttle changes incumbent arrival times → changes pruning →
   changes node counts everywhere. All Stack B/C comparisons therefore use
   dual-bound-at-budget and proofs as the signal, never node counts (§0.5).
3. TX4b (OBBT on pure-integer) widens OBBT's reach exactly where TX2 budgets
   it — sequence TX2 before TX4b so the sub-budget exists first.

## §5.5. Running synthesis after TX0–TX3 (2026-07-14) — the wins are engine-layer

Three items in (TX0 data; TX2 STOP; TX3 KILL), a pattern the pre-plan guessing
missed: **every large cost bucket is gated by the native engine (Rust simplex,
POUNCE, the JAX evaluator), not by Python orchestration — which is already
correct or already optimized.**

- **TX2:** the 4 overruns are single native bound-producing solves; the Python
  scheduler already passes correct per-phase budgets. Fix needs a **native
  deadline in the Rust simplex/POUNCE**, caller-tagged safe-vs-unsafe.
- **TX3:** the easy-class floor is `import jax` (~0.4 s, ~40 % of the median),
  imported unconditionally as the *first* solve op to build the NLP evaluator
  (node bound source + all heuristics). Unremovable without a jax-free
  **point-mode NLP evaluator** — and the F2′ microbench (this session,
  `scratchpad/f2prime_speed.py`) shows the deciding constraint: **jitted JAX
  ≈ 45 µs/call vs interval_ad/numpy ≈ 500 µs/call.** A jax-free evaluator is
  ~10× slower *per call*; it only wins on import-dominated tiny models (few
  calls) and loses badly on anything with node volume. So JAX is the *right*
  per-call tool; its fixed import cost is the price, and that price only bites
  the easy class. The floor is a genuine import-vs-per-call tradeoff, not an
  inefficiency to fix cheaply.
- **TX0:** node-NLP is 69 % of wall (the per-node bound source, jitted-fast per
  call but high call *volume*); node-LP is 0.06 %.

**Implication for the plan.** The codified metric (easy-class geomean) is
**floor-blocked** — no cheap Python item moves it; the options are all heavy
(dual numpy/JAX evaluator for tiny models; upstream `import jax` speedup, §2
"out of repo"; or accept it). The reachable wins are:
1. **TX1** — throttle the NLP-heuristic *call volume* where it is idle waste
   (Python, bounded, the one un-blocked item; TX0 measured nvs09 = 14.3 s).
2. **TX4** — route family C to the existing LP-node engine (native LP per node
   instead of NLP) — the one place we already have a jax-light per-node path.
3. **A native-deadline engine item** (enables TX2) — Rust work, high certainty.
4. **The jax-free point-mode evaluator** — the biggest general lever *and* the
   biggest cost; would touch floor + node-NLP dispatch + deadline honoring at
   once, but it is an engine rewrite (TX3's "what would have to change"), and
   the 10×-slower-per-call constraint means it must be a *dual* path (numpy for
   tiny, JAX for the rest), not a replacement. Needs its own entry experiment
   before any commitment.

This is a strategic fork (items 3/4 are engine-scope, not Python) — flagged for
a direction decision, not auto-dispatched.

## §6. Alignment with the reference solvers (evidence-tiered)

Claims about SCIP/BARON in this plan carry one of three evidence tiers; no
untiered claim about a reference solver is admissible in TX items.

- **Tier 1 — measured in this repo.** SCIP was run directly on the family-C
  instances with statistics captured (`scip-gap-nvs-diagnosis.md`): nvs17
  70 nodes / 0.88 s default; **6,796 nodes / 4.72 s with cuts off** (≈1,440
  nodes/s — the TX4 throughput bar); presolve-off unchanged; separator stats
  aggregation(c-MIR) 15–22, Gomory 1–6, RLT 0. BARON is **closed source**: we
  observe only outcomes (recorded walls, Appendix B / `baron_global50.txt`);
  all claims about its internals are Tier-2 reconstructions.
- **Tier 2 — published literature (keys in `docs/references.bib`, distilled
  in `.crucible/wiki/methods/{bound-tightening,spatial-branch-and-bound}.org`).**
  BARON = branch-and-reduce: LP/polyhedral relaxations of a factorable
  reformulation + per-node range reduction [Ryoo1996; Tawarmalani2005;
  Tawarmalani2002; Kilinc2018]. SCIP MINLP = LP-based branch-and-cut on an
  extended formulation [Vigerske2018; Bestuzheva2023]. OBBT is explicitly
  budgeted/filtered in reference practice [Gleixner2017]. Shared substrate:
  [McCormick1976; Smith1999; Belotti2009]; branching [Achterberg2005];
  heuristic budgeting [Berthold2006; Achterberg2007].
- **Tier 3 — inference (must be labeled).** E.g. "BARON does not run an
  expensive NLP at every node" is an inference from its published LP-polyhedral
  architecture, not an observable; BARON does run local NLPs for incumbents.
  The LMTD-convexity literature (Mistry & Misener) is not yet in the bib —
  ingest before citing in TX6.

Per-item alignment: **TX1** (NLP as budgeted heuristic) and **TX2** (budgeted
OBBT) move *toward* reference practice [Berthold2006; Vigerske2018;
Gleixner2017] — our stride-4-unconditional default is the deviation. **TX4**
is the published core of both solvers [Ryoo1996; Tawarmalani2005; Vigerske2018;
Kilinc2018] plus Tier-1 observation. **TX5** matches SCIP's observed working
separator (aggregation/c-MIR) and is deferred for the Tier-1 reason (no-cut
SCIP still solves the family). **TX3** has no reference analog — it removes a
substrate artifact (Python/JAX floor) reference solvers never had. **TX6** is
not alignment-driven: it is our own PF4 finding; no evidence either reference
solver handles the ε-pole class better. Deliberate divergence: the certificate
regime (outward rounding, zero-slack `incorrect_count`, differential gates) is
stricter than default floating-point SCIP/BARON practice — a product choice,
not a gap.
