# Closing the certification gap with BARON/SCIP — a general, phased plan

**Date:** 2026-07-02
**Status:** proposed (evidence-grounded; every phase carries an entry experiment and a
measurable exit gate)
**Scope:** the end-to-end *certification loop* — everything between "incumbent found"
and "optimality proved": per-node relaxation cost, root/in-tree range reduction, cut
separation, and the structure the relaxation is built from. General mechanisms only;
no instance-specific fixes.
**Relationship to existing docs:**
- `docs/design/relaxation-catalog.md` — establishes that the *envelope library* is at
  SOTA parity and names "bound-tightening orchestration" as the residual gap (§7 there).
  This plan is that orchestration work, made concrete and gated.
- `docs/dev/performance-plan.md` (2026-06-24) — its measured cost model (CC1–CC5) is
  adopted wholesale; its Stage 1 (evaluator cache) and Stage 2 (per-node Python tax)
  are absorbed into Phase 1 here. Its Stage-4 spike results (OBBT-on-aux dead,
  branching not the lever, range-reduction stalls on gear4) are treated as binding
  negative results.
- `docs/dev/scip-gap-closing-plan.md` — its Phase 0b/1 (c-MIR strength, with the §1.5
  node-reduction gate) becomes Phase 3 here, unchanged in substance.
- `reports/baron_parity_plan.md` (Feb 2026) — superseded by this document for the
  bound-side work; its Phase D (envelopes) is complete per the relaxation catalog.

---

## 0. Implementation contract (binding on the implementing agent)

This section is a contract, not guidance. An implementing agent (human or model)
working from this plan agrees to every clause below. A PR that violates a clause is
wrong even if its benchmarks improve.

### 0.1 Execution order

1. Start at **§14** (the task-level breakdown), not at the phase narratives.
   Execute tasks in order: T0.1 → T0.5, then T1.1 → T1.6. Do not reorder across
   phases; within a phase, a task may start only when its listed dependencies are
   merged.
2. **Phases 2–5 are locked** until their entry experiment (§6–§9) has been run and
   its results + derived task list have been written into §14 of this file. Coding
   a Phase 2–5 work item before its entry experiment is a contract violation, even
   if the work item seems obviously right.
3. T1.1 (the Phase 1 entry experiment) has a kill criterion. If it fires, stop,
   record the result in §14, and re-scope per §5 before writing further Phase 1
   code.

### 0.2 Correctness invariants (zero slack, every PR)

Every PR must pass, and its description must state that it passed:

1. `pytest -m smoke` — 0 failures.
2. `pytest -m slow python/tests/test_adversarial_recent_fixes.py` — 0 failures.
3. `incorrect_count ≤ 0` on the affected suite (never weaken this check — CLAUDE.md
   key constraint).
4. The certificate invariant: `bound ≤ incumbent` (min sense); a dual bound never
   crosses the known oracle value.
5. **Bound-neutral tasks** (all of Phase 0 except T0.5's gate addition; T1.2–T1.6):
   `node_count` and certified `objective` **exactly unchanged** vs
   `docs/dev/data/cert-baseline.jsonl` on the certifying panel. Any drift — even an
   *improvement* — means the change is wrong. Do not rationalize a drift as a bonus;
   find the bug or revert.
6. **Bound-changing tasks** (Phases 2–4 only, after unlock): the §3 differential
   bound test + feasible-point sampling via the T0.4 harness, plus 3 consecutive
   green nightlies before any default-on flag flip.

### 0.3 Safety mechanisms are load-bearing

Never weaken a validation or fallback path to make a gate pass. Specifically:
`IncrementalMcCormickLP._validate()` / `ok=False` fallback
(`incremental_mccormick.py:13-23`), the trusted-incumbent gate
(`tree_manager.rs:288-554`), the `gap_certified` downgrade guard
(`modeling/core.py` §5 of relaxation-catalog), and the OBBT/DBBT safe dual bounds.
If a gate can only be met by loosening one of these, the gate loses and the plan
gets re-scoped — record it and stop.

### 0.4 Measurement beats plan

If a measurement contradicts a claim in this document, the measurement wins.
Record the falsification **in this file** (a dated note in the affected section, in
the style of `performance-plan.md` §6), re-scope, and only then continue. Do not
silently code around a falsified premise.

### 0.5 Scope discipline

- General mechanisms only. Named instances (gear4, nvs17, casctanks, …) are gate
  probes; a change whose benefit is confined to a named instance is rejected.
- No work on the §12 out-of-scope list (envelope math, branching rules, parallel
  tree search, instance tuning, decomposition) under this plan.
- One task per PR where practicable; a PR names its task ID (e.g. `cert:T1.2`) in
  its title or description.

### 0.6 Stop-and-escalate conditions

Stop work and surface to the maintainer (do not improvise) when:
- a kill criterion fires (T1.1, or any Phase 2–5 entry experiment);
- clause 0.2.5's exact-equality check fails and the cause is not found within one
  working session;
- meeting an exit gate would require touching a §0.3 safety mechanism;
- the baseline (`cert-baseline.jsonl`) is missing or stale relative to `main` such
  that neutrality cannot be asserted.

### 0.7 Definition of done (per phase)

A phase is done when: its §4–§9 exit gate is green on the committed baseline's
panel; its `[gates.cert*]` criteria are wired in `benchmarks.toml` and pass via
`run_benchmarks.py --gate`; §14 records any falsifications/deviations; and the
living status table in §0.8 is updated. Then, and only then, the next phase's entry
experiment may run.

### 0.8 Implementation status (living table — update in the same PR as the work)

| Task | Status | PR | Notes |
|---|---|---|---|
| T0.1 root_gap/root_time producer | done | (this PR) | Populated on all in-house B&B + convex-fast paths; bound-neutral (0 drift on 12 instances) |
| T0.2 bound trajectory | done | (this PR) | Opt-in recorder (default OFF — node_callback disables GP fast path); downsampled ≤500; monotone-t, bound-non-decreasing |
| T0.3 reduction/separation timers | done | (this PR) | Rust Fbbt/NodeReduce phases (stderr profile); Python per-family timers on solver_stats; bound-neutral (0 drift, Rust rebuilt) |
| T0.4 soundness harness | done | (this PR) | utils/soundness.py: assert_bound_sound + assert_cut_valid; flags planted-invalid cut, passes McCormick envelope sweep |
| T0.5 baseline + cert0 gate | done | (this PR) | cert-baseline.jsonl (global50+panel, 55 rows); [gates.cert0] passes: coverage 0.909 ≥ 0.90, incorrect 0. Phase 0 complete |
| T1.1 entry experiment (kill criterion) | done (kill did NOT fire) | (this PR) | All uncovered families are closed-form box-affine → proceed to T1.2. Bonus: engine already validates on mixed int+cont (ex1263) and maximize |
| T1.2 patch-table term coverage | monomial landed; families derived | (this PR) | Dir. (a) differential neutrality. Monomial p≥2 coverage landed (NEUTRAL). Trilinear/multilinear, univariate exp/log/sqrt, fractional all derived + verified to 1e-9 (scripts/t12_*.py) — integration is mechanical, deferred (value comes with the OBBT wall-time follow-on) |
| T1.3 scope-gate widening | done | (this PR) | Widened gate to `ok` + general root-cut-pool (built whenever engine active) + skip fast path during pool capture. dispatch 9843→3 restored; nvs13 55→19, nvs17 205→61. NEUTRAL / adversarial 10 / smoke 211. Fast engine now on the general spatial path |
| T1.4 basis inheritance | non-lever (measured); re-scoped | — | Warm-primal built + sound (42/42 tests) but INERT: run_warm never called — the dual warm start already succeeds / nrows guard routes to ordinary cold. Node LP warm-start is not the bottleneck (nvs17 2 n/s). Real lever = OBBT/per-call-rebuild re-profile (T1.6). Parked patch |
| T1.5 evaluator-cache routing | already realized (PR #316) | #316 | primal_heuristics diving already routed through cached_evaluator (the −22% gear4 win); remaining sites are one-time (convex fast path) or autodiff-unsafe. Low residual value |
| T1.6 bookkeeping → Rust | non-lever (measured); re-scoped | — | Re-profile: Python per-node tax is minor (0.5s); LP solves dominate (3.84s / ~10-per-node, OBBT probes cold-built). T1.6 premise falsified. Real lever = warm-start OBBT's probe LPs (the parked warm-primal applies — objective-only change over fixed box). Follow-on, not a T1.x item |
| T2.0 reduction-layer correctness pre-flight | todo — **blocking** | — | C-16 (P0) first, then C-15/C-20/C-21; see §14 Phase 2 |
| T2.1 Phase 2 entry experiment (kill criterion) | **REVISITED (2026-07-06) → GO** (supersedes provisional NO-GO) | #408, R1 | Full panel now completes (19/20, loop-median 0.27 s); **no P0 on 51 instances**. (a) still FAILS on the Class-P tail (panel median 7.4%, 2/6 responsive) BUT the R1 OR-rule's generality arm PASSES: out-of-panel responsive 9/29=31% (≥20%). Loop = {S2 cutoff-FBBT, S3 cutoff-OBBT}; drop S1/S4; budget ≈10% of limit. Scope: broad small-MINLP root closure, NOT the hard tail. See "T2.1-revisit RESULTS / VERDICT (2026-07-06)" block |
| T2.2 OBBT persistent LP + warm probes | (a)+(b) done; (c) skipped | #406 | Per-sweep CSC built once + warm-primal probes (t14 patch applied). Differential **NEUTRAL** (warm==cold ≤4.3e-12, tightenings identical, cert-neutrality NEUTRAL). Root-OBBT umbrella: ex1252a 1.54×, ex1252 1.89×, st_e38 1.92×. ex1252a < 2× → residual is the JAX envelope rebuild (Phase 4/5), not the LP loop. See the T2.2-DONE block below |
| T2.3 root fixpoint loop | **built — flagged default-OFF** (R2) | (this PR) | `_jax/root_reduce.py::run_root_fixpoint` {S2 cutoff-FBBT, S3 cutoff-OBBT}, integrated at end of iteration 0 (solver.py); tighten-only intersection, ≤2 rounds, ≈10% budget; `root_fixpoint`/`DISCOPT_ROOT_FIXPOINT` OFF. No-offtarget gate + cut-pool re-capture opt-in only. Flag-OFF cert-baseline **NEUTRAL** (41/41 node_count exact, Δobj 0). Node A/B: wastewater04m2 479→159 (root). See §14 "R2 build-results" |
| T2.4 per-node `reduce_node()` | **built — flagged default-OFF** (R2) | (this PR) | T2.4a marginals on `MccormickLPResult` (bound-neutral, additive `dual`/`col_status`/`safe_bound`/`reduced_costs`); T2.4b `_jax/node_reduce.py::reduce_node` (cutoff-FBBT + free DBBT z=safe_bound + integer RC-fixing); T2.4c `PyTreeManager.set_node_bounds` feeds child boxes. `node_reduce`/`DISCOPT_NODE_REDUCE` OFF. 200-box property test green. See §14 "R2 build-results" |
| T2.5 OBBT escalation policy | **UNLOCKED** (T2.1-revisit GO) → R2 | — | build per §14 spec (width×\|RC\| top-k scoring) |
| T2.6 cert2 gate wiring + default-on | **moot** (T2.3–T2.5 not built) | — | residual gap re-scoped to Phases 3–4 |
| Phase 3 entry experiment (0b) | done — GO on c-MIR **(SUPERSEDED by CUT-1, 2026-07-06 → NO-GO)** | (prior) | SCIP root-bound *proxy* (`scripts/p3_0b_scip_rootbound.py`): median root gap closed discopt 0.0 vs SCIP 1.0 over 8 (graphpart/ex1263/fac). CUT-1 replaced the proxy with a direct injection of SCIP's actual c-MIR cut coefficients into discopt's LP + a real-relaxation measurement; on nvs17/19/24 discopt's *default* root already closes 99.9% (≥ SCIP's cut-root), and injected cuts close ≤1.8%. See §7 "0b RESULTS / VERDICT (2026-07-03)" and "CUT-1 …(2026-07-06)" |
| Phase 3 1c reachability entry experiment | done — **NO-GO / re-scope** | #(prior) | Cuts made reachable+armed close ~0% (median gap-closed 0.000, best +0.55% on ex1263a) vs SCIP ~1.0. Residual is separator DEPTH, not plumbing. Do NOT build the Rust cut-callback seam yet. See §7 "Phase 3 1c" |
| Phase 3 1d separator-family attribution | done — **build zerohalf** | #420 | SCIP per-separator attribution (`scripts/p3_1d_separator_attribution.py`): on graphpart `zerohalf` alone closes 60–86% of the reachable root gap and is the sole load-bearing family (leave-one-out 15–53%); every other cut family (flowcover/aggregation/cmir/clique/…) closes 0. Build target = native zero-half separator; expected ~0.6–0.9 root-gap close. See §7 "Phase 3 1d" |
| Phase 3 zerohalf build (native {0,½}-CG separator) | done — **sound; lever INERT on graphpart (measured)**; code parked on branch `cert-p3-zerohalf` (PR #427 closed unmerged), NOT on main | finding only | A validity-GREEN heuristic zero-half separator was built (400-system + binary-dense property tests: no feasible point cut). ON/OFF on graphpart: **gap_closed 0.000, `incorrect_count=0`** — the predicted 0.6–0.9 did NOT land: discopt's root LP optimum is a ⅓-partition vertex where every {0,½} combo is *tight, not violated* (exhaustive GF(2) nullspace search: viol=0.0000), and `root_off` already sits at SCIP's separators-off floor. Root cause = **LP vertex geometry (½-cuttable vs ⅓), not cut depth**. Follow-on = separate at a ½-valued/pre-crossover point. The inert separator is NOT merged (no dead flag on main); preserved on the branch for the follow-on. See §7 "Phase 3 zerohalf — build results" |
| CUTS-1 (#81) aggregation/c-MIR entry experiment — **re-verify on current main** | done — **NO-GO stands (re-confirmed 2026-07-10)** | (this PR) | Re-ran the CUT-1 entry experiment on HEAD `059165fc`. Both kill-criterion legs reproduce bit-for-bit: oracle injection closes ≤1.8% (nvs17), 0% (nvs19/24) (`scripts/cut1_cmir_oracle_injection.py`, raw `..._20260710T161545.json`); and `DISCOPT_CMIR_AGGREGATION` ON/OFF gives **0× node reduction** (bit-identical bound + node count, `incorrect_count=0`) at an equal 1500-node budget on the default solve path (`scripts/cuts1_cmir_flag_onoff_nodes.py`). Rust separators confirmed present + green (`cargo test … -- aggregation mir` = 12 passed, incl. upper-bound complementation). **Do not build the separator** — the capability exists (`lp/aggregation.rs`+`lp/mir.rs`, default-off) and is measurably inert on this family. See §7 addendum in `docs/dev/cut-engine-entry-2026-07-06.md` |
| CUT-1 aggregation/c-MIR oracle-injection entry experiment | done — **NO-GO / re-scope (relaxation-mismatch)** | (prior) | Injected SCIP's *actual* aggregation/c-MIR cut coefficients into discopt's root lifted McCormick LP on nvs17/19/24 (`scripts/cut1_cmir_oracle_injection.py`, pyscipopt 6.2.1/SCIP 10.0). Gap-closed **≤1.8% (nvs17), 0% (nvs19/nvs24)** — kill criterion (<15% on ≥2/3) fires. Decisive: discopt's **default** root already closes **99.9%** on nvs17/19 (nvs17 root −1105.89 is *tighter* than SCIP's cut-root −1105.10); `DISCOPT_CMIR_AGGREGATION=1` gives a bit-identical bound (separator self-disables — nothing violated). The native aggregation-c-MIR separator already exists (`lp/aggregation.rs`+`mir.rs` w/ upper-bound complementation done, PRs #415/#416); it is **inert on this class** because the relaxation is already tight. nvs24 residual = root-fixpoint solve COST (Phase 1/2/4), not cuts. SCIP's 119–244× cut win is SCIP-vs-SCIP (its weak no-cut LP), not a discopt gap. Do NOT build the §7 part-2 separator for this family. See §7 "CUT-1 …(2026-07-06)" + `docs/dev/cut-engine-entry-2026-07-06.md` |
| Phase 4 re-profile (entry experiment) | done — **rank recorded** | #442 | 8 run / 0 skipped. Ranked build order: **1) CSE (op-dup 31–37% on nvs17/clay), 2) Q-extraction (coupled to CSE), 3) V-segments DE-PRIORITIZED (0 defvars in all 1,558 text `.nl`; defined-var-heavy set is binary-`.nl` discopt can't parse), 4) symmetry DO-NOT-BUILD (0 orbits)**. CC5 FALSIFIED (XLA ≤1.2% of wall); dominant wall = separation. See §8 "Phase 4 — re-profile results" |
| Phase 4 T-CSE/V-segments | **CSE unlocked; V-segments/symmetry de-scoped** (§0.1.2) | — | build order fixed by the re-profile above; CSE first (bound-neutral), Q-extraction second |
| Phase 4 build 1 — CSE/hash-consing | **done — bound-neutral** | (this PR) | Content-addressed interning in `ExprArena` (`expr.rs`), wired into the `.nl` parser and Python `convert_expr`. DAG node count ↓ **68.9% nvs17, 64.8% clay0303hfsg, 34.2% ex1252, 34.0% casctanks, 0.0% gear4** (panel total −48.4%); gear4 0% confirms the re-profile prediction. Cert-neutrality **NEUTRAL** (42 certifying instances, node_count exactly unchanged, objective to tol). See §8 "Phase 4 CSE — build results" |
| Phase 4 build 3 — Q-matrix extraction | **done — (A) bound-neutral, (B) flagged bound-changing** | (this PR) | Exact `extract_quadratic(expr,n,model)` in `quadratic_form.py` (exact-or-abstain, validated 1e-12 on 250 pts/inst + 14 non-quadratic rejections). Consumer: PSD-on-Q convexity certificate wired into `certify_convex` behind `DISCOPT_PSD_QFORM` (**default-OFF**); sound tightening, never mis-certifies (30 indefinite-Q seeds), strict refinement of the rigorous path. Flag-off is byte-identical → cert-neutral. See §8 "Phase 4 Q-extraction — build results" |
| Phase 4 items 2 & 4 (V-segments, symmetry) | **locked / de-scoped** (§0.1.2) | — | V-segments de-prioritized (0 defvars in text `.nl`); symmetry DO-NOT-BUILD (0 orbits) per the re-profile. RLT/edge-concave rewire onto `extract_quadratic` scoped as bound-neutral follow-on |
| Phase 5 | **locked** (§0.1.2) | — | requires post-Phase-1 re-profile |
| Phase D — separation/strong-branch LP → warm in-house simplex (`perf-d1`) | **done — bound-neutral, default-ON** | (this PR) | Re-profile: **POUNCE subsolver is the #1 wall** (nvs17 ~239 cold POUNCE-LP solves / 6.3 s ≈ 40% of wall). Routed edge-concave separation + strong-branch LPs to `lp_simplex.solve_lp` under `DISCOPT_SEPARATION_LP_SIMPLEX` (default ON). Cert-baseline **NEUTRAL** (41/41 exact node_count, Δobj 0, incorrect 0); nvs17 equal-budget 93 nodes both flags. T0.4: 37 cuts / 14,800 checks, no invalid cut. Win: nvs17 POUNCE 848→0, wall 60.3 s (TL) → 38.1 s (optimal), s/node 0.826→0.409 (**2.02×**); nvs13 2.67×. See §8 "Phase D re-profile" |
| THRU-2b node-LP fast path | **RE-SCOPE (premise falsified); one bound-neutral sub-fix shipped** | (this PR) | THRU-1's "node solved as integer MILP → drop integrality → sub-second" premise is **void on `origin/main`**: integrality is already dropped at every node (`node_bound_mode="lp"` default *and* RLT force the LP path — 13/13 nvs24 node solves have `integrality is None`, 0 integer-MILP solves, even under `DISCOPT_NODE_BOUND_MODE=milp`). The `solve_milp_py` calls THRU-1's cProfile mis-labeled "integer-MILP node solve" are **pure LPs** (`nint=0`) reached through the dense-cold fallback (`milp_relaxation.py:375`) when the warm sparse simplex breaks down `numerical` at iters=0 (factorization failure, not iter-limit) on 2–3 hard lifted LPs. Rare (nvs17/nvs21 0 fallbacks; nvs19 2; nvs24 2–3) and not nvs24's wall lever (that's PSD+square sep, the THRU-1 PSD gate). Shipped: a pure-LP short-circuit in `milp_simplex.solve_milp` — when `int_cols` empty, run the driver with the integer-search machinery OFF (cuts/GMI/heuristics/strong-branch), bound-neutral by construction (nvs24 fallback LP 10.9→5.5 s). Cert-baseline **NEUTRAL** (41/41 node_count exact, |Δobj|=0). Real lever for the fallback = LP presolve on the warm sparse simplex (Rust `lp/simplex`; re-scoped follow-on). See `docs/dev/root-throughput-entry-2026-07-06.md` §7 |
| THRU-2a cost-aware PSD moment-cut gate | **built — flagged default-OFF; THRU-1 per-round-delta mechanism FALSIFIED, re-derived** | (this PR) | Adaptive gate on the per-node PSD loop in `_jax/mccormick_lp.py::_separate_psd` (`psd_cost_gate`/`DISCOPT_PSD_COST_GATE`, **default OFF**): bound per-node PSD wall to `budget × base_solve_wall` (default 1.0) + diminishing-returns abandon (`tau` 1e-4). **Falsification:** THRU-1 §4's stated mechanism ("skip PSD when its per-round LP delta < τ") is falsified — instrumented deltas are *large* (nvs17 root round-0 +5.9e4; median rel 8e-3), because on box-QCQP PSD is the *only* stage that closes the node-LP McCormick gap (RLT is a no-op with no linear constraints; it adds +0.00 after PSD). PSD is not node-inert — it is **substitutable by branching at lower wall cost**: unbudgeted PSD *starves the search*. Correct general signal = PSD wall-share, not bound-delta. SOUND by construction (dropping cuts only loosens). Gate applies to **PSD only** — extending it to univariate-square over-reached (tspn05 optimal→feasible), reverted. Flag-OFF cert-baseline **NEUTRAL** (41/41 node_count exact, |Δobj|=0). Flag-ON panel: nvs17 38.3→**23.6 s (1.6×)** optimal @ −1100.4; **nvs24 root CONVERGES** (⊘/no-incumbent → 23.5 s root, feasible); nvs23 root 18.7→8.0 s (9→69 nodes); nvs19 keeps incumbent −1098.2 (PSD-helps case preserved). No oracle crossings (all dual bounds ≤ oracle). Flag-ON cert panel: 41/41 stay optimal, incorrect 0 (nvs13 19→49 nodes, on-class QCQP, still optimal). See `docs/dev/root-throughput-entry-2026-07-06.md` §8 |

**Phase 0 — DONE & gated** (cert0 green: root_gap coverage 0.909 ≥ 0.90, incorrect 0).

**Phase 1 — structural + correctness exit MET; performance exit PENDING one
follow-on.** Landed & sound: T1.2 monomial coverage, **T1.3 (the enabler — the
incremental engine now runs on the general spatial path, node-neutral)**; T1.5 was
already merged (#316); every other patch-table family is derived + verified to
1e-9. Correctness exit met: differential neutrality is **NEUTRAL** across the
42-row certifying subset (`check_cert_neutrality.py` — objective correct, still
optimal, node one-directional) and `incorrect_count = 0`. **Performance exit
(s/node ↓ ≥ 2×) NOT met and correctly diagnosed as out of the T1.x scope:** the
per-node cost is dominated by **OBBT's inner LP loop** (~10 cold-built probe
solves/node), not the node relaxation — a bound-neutral follow-on (warm-start
OBBT's probes; the parked `t14-warm-primal-patch.diff` is the right tool since the
probes change only the objective over a fixed box). T1.4 and T1.6 were both
*measured* to be non-levers. Per §0.7 the phase is not fully "done" (perf gate not
green), but its structural goal — a single general node engine — is delivered and
its correctness gate is green. **The measurement discipline falsified four plan
premises before any unsound/ineffective code shipped** (T1.2 sign-regime, T1.3
per-node separation, T1.4 dual-warm-repair, T1.6 Python-tax).

**Phase 2 — specced to task granularity (2026-07-02, maintainer-directed); see
§14 "Phase 2 tasks".** The OBBT wall-time follow-on flagged in T1.6's conclusion
is relocated there as T2.2 (Phase 2 multiplies OBBT call volume, so its cost is a
Phase 2 prerequisite); `[gates.cert1]`'s two perf criteria stay informational
until T2.2 lands. Per §0.1.2, T2.3–T2.5 remain locked until the T2.1 entry
experiment's results are recorded in §14.

---

## 1. The measured diagnosis

> **Correction (2026-07-02, per §0.4 — fresh 3-probe profile, see
> `docs/dev/bottleneck-profile-2026-07-02.md`):** four June claims underlying C1
> are stale. (1) The layer-fraction fields are accounting artifacts: `rust_time`
> excludes the per-node LP binding calls and POUNCE (actual Rust simplex compute
> is **68% of wall** on the lp_spatial path), and `python_time` is a residual
> that absorbs root OBBT's JAX work — do not gate on these fields until T0.3's
> honest timers are used instead. (2) XLA compilation is **resolved** (≤1% of
> wall; the evaluator-cache fix landed; bounds are traced arguments — 0 recompiles
> across changing boxes); Phase 5's compile item is done. (3) The dominant
> orchestration cost is now **OBBT's inner loop**: 23 LP solves/node on gear4,
> each rebuilding scipy sparse structure, plus 2.4 extra `build_milp_relaxation`
> calls/node — which *raises* Phase 1's payoff (the persistent bound-patchable LP
> serves the node bound, OBBT, and the per-call structure rebuild at once) and
> adds a Phase 1 task: return the Neumaier–Shcherbina safe bound + Farkas verdict
> from Rust (~18% of lp_spatial wall is currently re-derived in Python per call).
> (4) The lp_spatial engine runs at 8.6 ms/node (not 1.9), dominated by cold
> refactorizations from rejected warm starts — T1.4's measured target: avg LP
> call on nvs17 from ~1,840µs → ≤300µs. The C1 headline (per-node cost dominates;
> Phase 1 is the enabler) stands.

The working hypothesis was "discopt finds the optimum quickly but the relaxation is too
weak to certify it." The evidence *refines* this: the envelope formulas are not the
problem — the certification loop around them is. Three causes, in measured order of
leverage, plus a structural multiplier:

### C1 — Per-node cost, not node count, dominates most of the gap

- Layer profiling over 48 discopt runs on the global50 panel: **JAX 57.8% / Python
  42.1% / Rust ≈ 0%** of wall time. The Rust tree+LP layer is essentially free.
- Median 0.036 s/node vs BARON's 0.011 s/node; tail to 4.8 s/node (`st_e38`).
- Of the both-optimal instances slower than 10×, **8 of 11 finish in ≤ 20 nodes** —
  they pay a fixed per-node/per-solve tax, not a bigger tree. Extremes: `casctanks`
  5097× slower at **9 nodes**; `st_e38` 272× at **3 nodes**.
- Root cause is identified in-code: the lifted McCormick LP is **cold-rebuilt from the
  DAG every node** (`_jax/mccormick_lp.py:310-320` — "~half the spatial-B&B wall
  clock"; `_jax/incremental_mccormick.py:1-11`). An incremental, warm-started engine
  **already exists** (`incremental_mccormick.py`, wired at `mccormick_lp.py:561-563`)
  but is scope-gated to pure-integer minimize models (`lp_spatial_bb._is_in_scope`).
- Secondary, validated: heuristic sites construct `NLPEvaluator(model)` bypassing the
  `_make_evaluator` cache (−22% gear4 wall when routed through it, bound-neutral —
  performance-plan Stage 1).

### C2 — The root reduction loop is far weaker than BARON's branch-and-reduce

- BARON closes most of the global50 set **at 0–9 nodes in 0.02–0.08 s**; discopt opens
  a tree at all (median 6.4× slower, mean 35×, p90 141×).
- Measured root-bound example: nvs17 root McCormick bound −2522 vs the achievable
  convex bound −1106 vs optimum −1100 (`solver.py:4060-4066`).
- The *components* BARON iterates all exist in discopt — FBBT forward+backward (Rust,
  incl. `fbbt_with_cutoff`), OBBT + DBBT (`_jax/obbt.py`), probing
  (`presolve/probing.rs`), exact multilinear/RLT hulls, edge-concave separation,
  tangent OA on convex-certified rows, α-BB fallback, two convexity-detection tracks —
  but the *orchestration* is thin and heavily gated:
  - probing is root-only; per-node OBBT is gated to a narrow model class
    (`solver.py:4765`); RLT is off per-node by default (`mccormick_lp.py:250-261`);
  - `fbbt_with_cutoff` (incumbent-aware FBBT) exists in Rust but the spatial path does
    not run it at every node;
  - there is no fixed-point loop at the root that alternates reduction ↔ relaxation
    tightening ↔ re-separation until no progress, which is precisely what lets BARON
    finish at the root;
  - `root_gap`/`root_time` are `null` in every benchmark result — the gate
    `root_gap_ratio_vs_baron ≤ 1.3` in `benchmarks.toml` is currently *unevaluable*.

### C3 — Cut strength on the integer-product / MILP-relaxation families

From `scip-gap-closing-plan.md` (all measured):
- SCIP ablations: cuts give **97–169× node reduction** on nvs17/19/24, growing with
  size; presolve 0%, strong branching ≤ 15%. The workhorse is **aggregation /
  complemented-MIR**, applied per node.
- discopt's current GMI + Python c-MIR are **net-negative**: at an equal 1,500-node
  budget the bound is *worse* with cuts on (§1.5 gate result). Rust has GMI/MIR/cover
  separators + SCIP-style efficacy/orthogonality selection (`lp/cut_select.rs`), but
  no aggregation c-MIR, and MIR lacks upper-bound complementation (`mir.rs:59`).

### C4 — Structure is lost before the relaxation is ever built (multiplier on C1–C3)

- **No CSE/hash-consing** in the Rust expression arena (`expr.rs:266` — `add` never
  dedups); the `.nl` parser **discards AMPL defined variables** (V segments — AMPL's
  own DAG sharing), so shared subexpressions are duplicated into the DAG, the JAX
  compile, the lifted LP, and every FBBT sweep.
- **No quadratic/Q-matrix or structure extraction** in the IR (only degree checks);
  convexity detection lives Python-side only. SCIP/BARON drive relaxation choice and
  cuts from recognized structure.

### What discopt already has (this plan must not rebuild it)

Exact bilinear/multilinear hulls with on-demand separation; edge-concave quadratic
cuts; tangent OA gated on per-constraint convexity certificates; rigorous α-BB;
FBBT both directions with two fixed-point engines; OBBT/DBBT with safe dual bounds;
pseudocost/reliability/strong branching with warm dual re-solves; a hand-written
primal+dual simplex with basis inheritance; a 15-pass presolve; a large, sound primal
heuristic suite (incumbents are not the problem — the *proof* is).

---

## 2. Gap table vs SCIP/BARON

| Capability                                             | SCIP/BARON                          | discopt today                      | Gap type              |
|--------------------------------------------------------|-------------------------------------|------------------------------------|-----------------------|
| Envelope library (factorable core)                     | reference                           | at parity (relaxation-catalog §8)  | none                  |
| Node relaxation reuse / warm start                     | always (bound patch + dual simplex) | exists, gated to pure-int minimize | **wiring**            |
| Per-node language cost                                 | native                              | 58% JAX + 42% Python per node      | **architecture**      |
| Root reduce↔relax↔separate fixpoint                    | core loop (branch-and-reduce)       | single-shot passes, budget-gated   | **orchestration**     |
| Per-node cheap reduction (FBBT+cutoff, DBBT/marginals) | every node                          | components exist, mostly root-only | **wiring**            |
| Aggregation / c-MIR cuts                               | workhorse (97–169× nodes)           | absent (current cuts net-negative) | **missing + quality** |
| Cut pool w/ aging on default path                      | yes                                 | opt-in path only                   | wiring                |
| CSE / defined-variable sharing                         | preserved & exploited               | discarded                          | **missing**           |
| Quadratic/structure recognition in core                | yes                                 | Python-side convexity only         | missing               |
| Root-gap instrumentation                               | n/a (internal)                      | schema exists, never populated     | missing               |
| Parallel tree search                                   | partial                             | no                                 | out of scope here     |

---

## 3. Design principles

1. **Correctness is a gate with zero slack.** Every phase ships behind a flag and must
   keep `incorrect_count ≤ 0` on the full panel, pass the adversarial suite
   (`test_adversarial_recent_fixes.py`), and satisfy the certificate invariant
   (`bound ≤ incumbent` for min; a dual bound never crosses the oracle).
2. **Two verification regimes, chosen by change type:**
   - *Bound-neutral changes* (Phase 1, most of Phase 0): identical relaxation math ⇒
     assert **exact equality of node_count and certified objective** vs baseline on a
     certifying panel. Any drift means the change is wrong, full stop.
   - *Bound-changing changes* (Phases 2–4): a **differential bound test** — on a fixed
     set of boxes, new bound ≥ old bound AND ≤ the true box optimum (trusted dense
     solve); plus feasible-point sampling against every new cut/reduction (a valid
     reduction/cut never removes a feasible point better than the cutoff).
3. **No fix ships on a hypothesis.** Each phase opens with a cheap entry experiment
   with a kill criterion (the lesson of the 2026-06-24 measurement pass, which
   falsified three plausible stages before code was written).
4. **General mechanisms only.** Every work item below applies to a *class* (all
   spatial models, all lifted LPs, all `.nl` inputs), never to a named instance.
   Named instances appear only as gate probes.

**End goals (wired as gates, §11):** median slowdown vs BARON on global50
6.4× → ≤ 2.5× (then 1.5×); `python_orchestration_fraction` 0.42 → ≤ 0.10 (then 0.05);
`root_gap_ratio_vs_baron ≤ 1.3` (and *evaluable*); zero incorrect throughout.

---

## 4. Phase 0 — Make the gap observable (~1 EW, low risk)

The certification story cannot be managed while `root_gap` is null and per-node
reduction is untimed.

**Build**
- Populate `root_gap` / `root_time` in every `SolveResult` (schema already exists in
  `benchmarks/metrics.py:38-107`); record **bound trajectory** (best_bound vs time and
  vs node index) and **time-to-first-incumbent** (already added by perf-plan Stage 0)
  on the comparison suites, not just the perf panel.
- Add the missing timers: per-node FBBT/reduction time (Rust `profile.rs` has no
  propagation phase), relaxation build vs patch vs solve split, separation time per
  cut family.
- Extend the differential-bound harness into a reusable fixture: `assert_bound_sound
  (relaxer, boxes, oracle)` + feasible-point sampling for cut validity — Phases 2–4
  all consume it.
- Baseline run of global50 + the perf panel with the new fields; commit as the
  reference JSONL next to `docs/dev/data/perf-baseline.jsonl`.

**Exit gate:** `root_gap_ratio_vs_baron` computable on ≥ 90% of global50; a
"certification profile" per instance (root gap, bound trajectory, s/node,
reduction/separation/solve split) renders from one JSON.
**Correctness gate:** pure measurement; full panel green.

---

## 5. Phase 1 — One general node engine: build-once, patch, warm-start (~3–5 EW, low–medium risk)

**Hypothesis:** generalizing the existing incremental relaxation engine to the whole
spatial path removes the dominant per-node tax (cold DAG re-walk + re-equilibration +
Python orchestration), worth ~2× on the spatial class immediately and enabling every
later phase to run reduction per node cheaply.

**Evidence:** C1 above; `mccormick_lp.py:310-320` ("~half the wall clock");
the gated engine already demonstrates the pattern on pure-integer models; gear4
routing spike (per-node hot spots: `build_milp_relaxation` 5.3 s,
`equilibrate_relaxation_lp` 8.3 s over the solve).

**Build (general mechanism, three layers)**
1. **Lift the scope gate on `incremental_mccormick`** from "pure-integer minimize" to
   the general spatial path: build the lifted LP structure once per subtree root;
   per node, patch only box-dependent envelope rows (McCormick/secant/tangent
   coefficients are closed-form in the bounds) and re-use/cheap-refresh equilibration.
   Maximization and continuous/mixed models are handled by normalizing sense and
   keeping the patch table keyed by aux-term type — no per-class forks.
2. **Basis inheritance across nodes on the general path:** child starts from parent's
   basis, repaired by the existing Rust dual simplex (`PreparedDual` already clones a
   pristine factorization; the machinery is built, the spatial path just doesn't use
   it).
3. **Move per-node orchestration bookkeeping into the Rust tree manager** (rust_time
   ≈ 0 today — there is headroom), and route all evaluator construction through the
   `_make_evaluator` fingerprint cache (perf-plan Stage 1, validated −22% on gear4).

**Entry experiment (1–2 days, kill criterion):** hand-run the incremental engine on
three out-of-scope instances (one maximize, one mixed-integer + continuous bilinear,
one general-NL e.g. `st_e38`/`casctanks` class) with the scope gate bypassed; compare
bound sequence vs the cold-rebuild path node-by-node. If bounds differ anywhere, the
patch table has a term type that is not box-affine — fix or descope that term type
before generalizing.

**Exit gate:** s/node on the spatial panel ↓ ≥ 2× at unchanged node counts;
`python_orchestration_fraction` ≤ 0.10; `casctanks`/`st_e38`-class (≤ 20-node,
> 10×-slowdown instances) within 3× of BARON.
**Correctness gate:** bound-neutral regime — node_count and certified objective
**exactly unchanged** vs baseline per instance; full panel + adversarial suite green.
**Risk:** medium only in the patch-table completeness; mitigated by the entry
experiment and by falling back to cold rebuild per term type (never per instance).

---

## 6. Phase 2 — Branch-and-reduce orchestration (~4–6 EW, medium–high risk, highest bound leverage)

> **2026-07-02 — specced to task granularity in §14 ("Phase 2 tasks"), which
> supersedes this narrative where they differ.** A code recon (three verified
> anchor reports) corrected two premises below, recorded per §0.4: (1) a
> reduce↔re-relax fixpoint already exists *inside* `obbt_tighten_root`
> (obbt.py:1691, ≤3 rounds, envelope rebuild each sweep) — the gaps are the
> missing incumbent cutoff at the root (solver.py:3891 runs structural OBBT with
> `incumbent_cutoff=None`), the missing re-separation stage, per-node wiring, and
> OBBT's per-probe LP cost (bottleneck B1); (2) item 1's "incumbent cutoff from
> the root heuristics, which already run first" is wrong for the current code —
> the root OBBT block (solver.py:3835) runs at presolve time, *before* any
> incumbent exists; the cutoff-aware loop therefore integrates at the end of
> iteration 0 (post-root-heuristics), see T2.3.

**Hypothesis:** iterating the *existing* reduction components to a fixed point at the
root, and running the cheap subset at every node, closes most instances at or near
the root the way BARON does — the components are built; the loop is the gap
(relaxation-catalog §6.D: "components present; orchestration is the gap").

**Evidence:** C2; BARON finishes global50 at 0–9 nodes; discopt's own OBBT/DBBT/
probing/FBBT-with-cutoff exist but fire once or under narrow gates. Binding negative
results respected: OBBT-on-aux cascade is measured-dead (perf-plan Stage 4); gear4's
box is reduction-resistant — the gate metric is *panel-wide* root gap and node count,
not gear4.

**Build (one loop, not new math)**
1. **Root fixpoint loop:** presolve → FBBT → probing → OBBT/DBBT (with incumbent
   cutoff from the root heuristics, which already run first) → re-derive envelopes on
   the tightened box → re-separate (RLT/tangent/multilinear/edge-concave) → repeat
   until no bound moves > tol or a work budget (fraction of time limit) is spent.
   Deterministic pass ordering, Python-orchestrated over the existing Rust/Python
   passes (the presolve orchestrator already supports Python-side ordering).
2. **Per-node cheap reduction, always on:** Rust `fbbt_with_cutoff` at every spatial
   node (the MILP driver already does per-node FBBT; the spatial path doesn't);
   **marginals/DBBT from the node LP that was just solved** (reduced costs are free —
   one pass per node, BARON's signature move); integer reduced-cost fixing already
   exists at node level — unify all three behind one `reduce_node(bounds, duals,
   cutoff)` call in the Phase-1 engine, so tightened bounds flow directly into the
   patch table.
3. **Escalation policy instead of class gates:** replace the narrow per-node OBBT
   model-class gate with a uniform trigger — run OBBT on the k variables with the
   largest bound-width × dual-activity score, only at nodes where the relative gap
   exceeds a threshold and depth ≤ d. One policy, all models.

**Entry experiment (2–3 days, kill criterion):** replay the root loop offline on the
20 worst global50 instances using stored models; measure root gap before/after and
projected node counts (re-solve with tightened root box). Kill/rescope any loop stage
that moves root gap < 5% on the whole set.

**Exit gate:** `root_gap_ratio_vs_baron ≤ 1.3` on global50 (the currently-unevaluable
gate becomes green); ≥ 30% of currently-tree-opening instances close within 10 nodes;
median slowdown ≤ 2.5×.
**Correctness gate:** bound-changing regime — differential bound test on every loop
stage; every reduction validated by feasible-point sampling (a reduction that cuts a
feasible point better than the cutoff blocks the phase); 3 consecutive green
nightlies before default-on.
**Risk:** high (a wrong reduction is a false certificate — the nvs22 #277 / st_ph10
#306 failure mode). This is why the phase is *orchestration of already-sound passes*
rather than new reductions, and why it lands after Phase 0's harness.

---

## 7. Phase 3 — Cut engine quality: aggregation c-MIR + a default-path cut pool (~4–6 EW, medium risk)

> **STATUS (2026-07-06): the integer-product-family cut build is CLOSED as NO-GO
> (relaxation-mismatch).** CUT-1 (block below) measured that discopt's default root
> relaxation already closes 99.9% of the root gap on nvs17/19 — at/above SCIP's
> cut-loaded root — and that injecting SCIP's *own* aggregation/c-MIR cuts closes
> ≤1.8%. The native aggregation-c-MIR separator already exists (`lp/aggregation.rs`,
> `lp/mir.rs`) and self-disables (nothing violated). Do not build against this class;
> re-aim at throughput (Phase 1/2) + structure (Phase 4). The graphpart/zerohalf line
> is a separate, still-open cut-*context* follow-on (see the zerohalf block).

Adopted from `scip-gap-closing-plan.md` Phases 0b/1 with its gates intact; scoped
here as the general mechanism for the integer-product/MILP-relaxation class (#280
family, graphpart as the gate probe — not nvs17).

**Build**
1. **Phase 0b first (2–3 days, decisive):** inject SCIP's own cuts into discopt's
   McCormick LP and measure node reduction. This pins whether the gap is separator
   quality (build c-MIR) or relaxation/branching interaction (widen to lifted
   formulation first).
2. **Native aggregation/c-MIR in Rust** (`lp/` beside `gomory.rs`/`mir.rs`):
   Marchand–Wolsey row aggregation, bound complementation, δ-scan; mark integer-valued
   product aux columns integer (general: `w = x·y` with x,y integer is integer);
   complete MIR upper-bound complementation (`mir.rs:59`).
3. **Global cut pool with aging/efficacy selection on the *default* path**
   (`cut_select.rs` and `CutPool` exist; wire them into the Phase-1 engine so root and
   in-tree cuts persist, age, and are re-separated at the SCIP-like cadence), per
   `docs/design/global-cut-pool.md`.

**Exit gate:** the §1.5 gate, generalized — at equal node budget the bound is
*strictly better* with cuts on, across the integer-product panel; node counts within
~3× of SCIP on graphpart/nvs; **never** a wall-clock regression on the non-target
classes (cut overhead must pay for itself or the separator self-disables by efficacy).
**Correctness gate:** every emitted cut checked against sampled feasible points; dual
bound never exceeds the oracle; full panel green.
**Risk:** medium-high on separation *quality* (measured: porting the existing weak
cuts is a no-go); de-risked by Phase 0b.

### Phase 3 — 0b RESULTS / VERDICT (2026-07-03)

Ran the LEAN proxy for the "inject SCIP cuts" spike:
`discopt_benchmarks/scripts/p3_0b_scip_rootbound.py`. For each panel instance it
measures three root dual bounds — discopt's `SolveResult.root_bound` (short solve,
60 s cap), SCIP's root bound *with* its default cut loop (`limits/nodes 1`), and a
shared *trivial* anchor (SCIP's raw root-LP bound with separating **and**
propagating rounds disabled) — and the oracle optimum (`minlplib.solu`). Root gap
closed by bound `B` (min sense): `(B − trivial)/(opt − trivial)`. Panel: the
integer-product / MILP-relaxation class, graphpart the plan's gate probe. **8 run,
0 skipped**; raw JSON at `discopt_benchmarks/results/p3_0b_scip_rootbound_20260703T144721.json`.

| instance | opt | trivial | discopt root | SCIP root (cuts) | discopt gap-closed | SCIP gap-closed |
|---|---:|---:|---:|---:|---:|---:|
| ex1263 | 19.6 | 19.1 | 19.06 | **19.6** | −0.073 | **1.000** |
| ex1263a | 19.6 | 19.1 | 19.06 | **19.6** | −0.073 | **1.000** |
| fac1 | 1.609e8 | 1.421e8 | 1.607e8 | 1.608e8 | 0.990 | 0.993 |
| fac2 | 3.318e8 | 3.294e5 | 2.555e8 | **3.318e8** | 0.770 | **1.000** |
| fac3 | 3.198e7 | −6.319e9 | 2.233e7 | 2.498e7 | 0.998 | 0.999 |
| graphpart_2pm-0044-0044 | −13 | −16 | −16 | **−13** | 0.000 | **1.000** |
| graphpart_2g-0044-1601 | −9.541e5 | −1.026e6 | −1.026e6 | **−9.541e5** | 0.000 | **1.000** |
| graphpart_2pm-0055-0055 | −20 | −25 | −25 | −21.23 | 0.000 | **0.755** |

**Aggregate (median root gap closed): discopt 0.0 vs SCIP 1.0** (n=8 each).

**Verdict — SEPARATOR QUALITY is the lever. GO on native c-MIR (Phase 3 part 2).**
The signal is unambiguous and consistent with C3: across every graphpart probe
(and ex1263/ex1263a, and fac2) SCIP's cut loop lifts the root dual bound *to the
optimum* (gap closed 1.0), while discopt closes **0%** of that same gap from the
same trivial LP floor — on graphpart discopt's `root_bound` sits *at* the raw
LP relaxation (0.0), i.e. discopt's separators contribute nothing there. This is
not a relaxation/branching problem: the trivial LP floor is identical machinery,
and SCIP's advantage is entirely the cut loop that runs on top of it (the negative
discopt gap-closed on ex1263 is just discopt's reported bound landing a hair below
SCIP's raw-LP anchor — it does not soften the separator finding, it sharpens it).
The workhorse SCIP applies here is exactly aggregation / complemented-MIR
(scip-gap-closing-plan §1; C3). Therefore **build native aggregation/c-MIR in Rust
(Phase 3 §7 build item 2) plus the default-path cut pool (item 3)** — do NOT
re-scope to "widen the lifted formulation first." fac1/fac3 (both already ~0.99
closed by discopt) show the win is class-localized to the integer-product family,
as the plan predicted; the c-MIR work must self-disable by efficacy on the classes
where discopt is already tight (the §7 no-offtarget-regression gate).

This entry experiment measures only (it builds no cut plumbing, changes no solver
math) — bound-neutral by construction; no correctness gate applies.

### Phase 3 aggregation — build 1 results (2026-07-03)

Built the native Marchand–Wolsey aggregation c-MIR separator (build item 2, the
smallest sound slice): `crates/discopt-core/src/lp/aggregation.rs`
(`separate_aggregation_mir`). It pairs `≤` rows with **nonnegative** weights
`λ_i = |a_kt|`, `λ_k = |a_it|` to cancel one column `t` (the MW continuous-cancel
target; falls back to a fractional column on a fully-lifted all-integer LP), forms
the valid implied aggregate, and applies the **existing** complemented MIR
(`mir.rs::separate_mir`, PR #415) to it — MIR is reused verbatim, never
reimplemented. Wired behind `DISCOPT_CMIR_AGGREGATION` (**default-off**) into the
LP-spatial engine's node-cut separator (`_jax/lp_spatial_bb.py::_separate_node_cuts`)
and exposed as `aggregation_mir_cuts_py`.

**Correctness (primary gate) — PASS.** `aggregation_validity_random_systems`:
500 random 2/3-row systems (mixed integer/continuous, mixed-sign finite bounds,
~30% fully-integer to exercise the fractional-fallback, LP points forced to
violate) assert **no** integer/continuous-feasible point of the *original full
system* is ever cut (>20 cuts validated, non-vacuous). Plus a
fails-before/passes-after witness (`aggregation_finds_cut_single_row_mir_misses`)
where single-row MIR finds nothing but 2-row aggregation cancelling the continuous
variable separates `x*`. At scale (3,000 random systems via the Python binding):
aggregation found a violated cut in **1,703**; in **30** of those single-row MIR
found *nothing* — the regime aggregation is for.

**Lever (ON vs OFF) — no bound change on the in-scope panel; SOUND, self-disables.**
The separator runs in the LP-spatial engine, scoped to *pure-integer* models, so
the measurable panel is that subset of §7's 0b panel: `ex1263a`,
`graphpart_2pm-0044-0044`, `graphpart_2g-0044-1601`, `graphpart_2pm-0055-0055`
(ex1263/fac1–3 carry continuous vars → out of scope). Equal 400-node budget,
`cut_rounds=3`, 60 s cap
(`discopt_benchmarks/scripts/p3_cmir_aggregation_onoff.py`, raw JSON
`results/p3_cmir_aggregation_onoff_20260703T155324.json`):

| instance | opt | bound OFF | bound ON | nodes OFF | nodes ON | Δbound |
|---|---:|---:|---:|---:|---:|---:|
| ex1263a | 19.6 | 19.6 | 19.6 | 400 | 400 | +0 |
| graphpart_2pm-0044-0044 | −13 | −13 | −13 | 31 | 31 | +0 |
| graphpart_2g-0044-1601 | −9.541e5 | −9.541e5 | −9.541e5 | 12 | 12 | +0 |
| graphpart_2pm-0055-0055 | −20 | −20 | −20 | 207 | 207 | +0 |

`incorrect_count = 0` on every row (uncapped ON solves certify the oracle; the
dual bound never crosses opt). **Δbound = +0 everywhere**: on this panel the
LP-spatial engine's McCormick relaxation is *already* at/near the optimum at the
root (bound = opt on graphpart), so neither single-row MIR nor aggregation finds a
violated cut there — the separator correctly self-disables (no cut, no
regression). This is a **different, tighter relaxation** than the default-path
`root_bound` the 0b verdict measured the 0% gap on; the bound-moving lever the 0b
table identified lives on the **default path's** lifted McCormick relaxer, not the
LP-spatial engine.

**Re-scope (measurement wins, per §0.4):** build 1 lands the proven, sound,
default-off separator + its native MIR reuse + the validity gate. The follow-on
(build 1b) is to wire `aggregation_mir_cuts_py` into the **default-path** McCormick
cut hook (the `cmir_cuts.py::separate_cmir` call site / the relaxer used off the
LP-spatial engine) where 0b measured the 0% root-gap, and re-run the ON/OFF panel
including ex1263/fac1–3 there. Cut-pool aging/efficacy on the default path (build
item 3) remains as originally scoped.

### Phase 3 1b — default-path measurement (2026-07-03)

Measured the aggregation c-MIR separator on the **default MILP path** (no
`lp_spatial`), the path where 0b measured the 0% root gap. Added a lightweight,
math-neutral per-source root-cut counter to `_root_cover_cut_loop` /
`_solve_milp_bb`, surfaced on `SolveResult.solver_stats` as `cuts/{cover_clique,
gomory,mir,aggregation}`, plus a solve-path probe. Harness:
`discopt_benchmarks/scripts/p3_1b_default_path_aggregation.py`; raw JSON
`results/p3_1b_default_path_aggregation_20260703T163611.json`. Equal 2000-node
budget, 90 s cap, full 0b panel (8 instances, 0 skipped).

| instance | opt | bound OFF | bound ON | gap-closed OFF | gap-closed ON | nodes OFF | nodes ON | agg cuts | solve path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ex1263 | 19.6 | 19.06 | 19.06 | 0.973 | 0.973 | 2015 | 2015 | **0** | milp_bb+milp_simplex |
| ex1263a | 19.6 | 19.06 | 19.06 | 0.973 | 0.973 | 2015 | 2015 | **0** | milp_bb+milp_simplex |
| fac1 | 1.609e8 | 1.609e8 | 1.609e8 | 1.000 | 1.000 | 9 | 9 | **0** | spatial-McCormick |
| fac2 | 3.318e8 | 3.318e8 | 3.318e8 | 1.000 | 1.000 | 69 | 69 | **0** | spatial-McCormick |
| fac3 | 3.198e7 | 3.198e7 | 3.198e7 | 1.000 | 1.000 | 103 | 103 | **0** | spatial-McCormick |
| graphpart_2pm-0044-0044 | −13 | −13 | −13 | 1.000 | 1.000 | 63 | 63 | **0** | milp_simplex |
| graphpart_2g-0044-1601 | −9.541e5 | −9.541e5 | −9.541e5 | 1.000 | 1.000 | 13 | 13 | **0** | milp_simplex |
| graphpart_2pm-0055-0055 | −20 | −20 | −20 | 1.000 | 1.000 | 235 | 235 | **0** | milp_simplex |

`incorrect_count = 0` on every row (uncapped ON solve certifies the oracle; dual
bound never crosses opt). ON == OFF bit-for-bit (bound, nodes) on all 8.

**Verdict — (c) IT DOES NOT FIRE. This is a wiring/scoping gap, not a
cut-strength result.** `agg cuts = 0` for all 8 instances; the separator's branch
is never executed on the default path. Root cause, from the solve-path probe (two
distinct mechanisms):

1. **graphpart_\*** (miqp) and the primary route of **ex1263/ex1263a** (miqcp):
   the default dispatch detects the integer-product/bilinear structure, applies
   the integer-bilinear big-M reformulation (`solver.py:3314`), and — crucially —
   **rewrites `nlp_solver` from the default `"pounce"` to `"simplex"`**
   (`solver.py:3327`), routing to the **monolithic Rust `_solve_milp_simplex`**
   engine. That engine has **no Python per-node/root cut loop** (the dispatch
   comment at :3628 says so), so `_root_cover_cut_loop` — the only place the
   aggregation separator is wired — is never called.
2. **ex1263/ex1263a's fallback into `_solve_milp_bb`**: when the model does reach
   `_solve_milp_bb`, it arrives with `prefer_pounce=False` (a consequence of the
   simplex reroute). `_gomory_enabled(False)` is then False, so `_cut_int_idx` is
   set to `[]` (`solver.py:11076-11080`), which makes `has_gomory=False` and gates
   **all** integer cuts off — GMI, single-row MIR, and aggregation alike
   (observed: `int_idx_len=0` at the cut loop, `by_source` all zero).
3. **fac1–3** (minlp/miqp): route to the **spatial-McCormick** path, which also
   has no `_root_cover_cut_loop` hook (and is already ~1.0 gap-closed here — the
   0b `disc_gc≈0.99` class where discopt is tight).

So the bound-moving lever the 0b verdict identified (SCIP closing ~100% on this
class) sits on engines discopt's default dispatch routes the integer-product
class *away from* the aggregation hook — the c-MIR separator never gets a chance
to separate. The build-1 note's premise ("wire `aggregation_mir_cuts_py` into the
default-path McCormick cut hook") is contradicted by the measurement: for this
class the default path is the **Rust `_solve_milp_simplex` engine**, not a
McCormick cut hook.

**Re-scope (measurement wins, §0.4) — next build (2 or the reordered 3):** the
correct scope is *reaching* the class with cuts, not deepening the 2-row slice.
Two candidate levers, to be pinned by a follow-on entry experiment:
(i) add a root/in-tree cut-callback seam to the monolithic Rust MILP engine
(`_solve_milp_simplex`) so cover/clique/GMI/MIR/aggregation separate there — this
is where the graphpart class actually solves; or (ii) stop the
`nlp_solver→"simplex"` reroute for models in this class *and* keep `prefer_pounce`
true into `_solve_milp_bb` so its existing cut loop (including aggregation) runs —
but only if the `_solve_milp_bb` path is not a wall-clock regression vs the Rust
engine (the :3323 note flags it as ~60 s vs ~1 s on ex1263, so (i) is the likely
right answer). Either way the aggregation separator built in #416 is *sound and
ready*; it is simply unreachable on the class that needs it until the cut seam
exists. Build 1b ships the instrumentation (the per-source cut counter +
solve-path visibility) that made this diagnosable and will verify the fix.

### Phase 3 1c — cut-reachability entry experiment (2026-07-03)

The decisive spike before building any invasive Rust cut seam (1b's candidate
lever (i)). It answers: *if the integer-product / graphpart class is made to run
the cut-enabled path — so aggregation c-MIR + the existing Gomory/cover/clique
separators actually fire at the root — does the root dual bound close a material
fraction of the 0b gap toward SCIP's ~100%?* If yes → GO (build the Rust cut
seam). If cuts fire but the bound barely moves → NO-GO / re-scope (the slice is
too shallow; SCIP's edge is deeper cuts or a lifted formulation).

**Lever (least-invasive, experiment-scoped, default-OFF, math-neutral when off):**
env `DISCOPT_P3_FORCE_CUT_PATH=1` (`solver.py::_p3_force_cut_path_enabled`) SKIPS
the `nlp_solver→"simplex"` reroute at `solver.py:3327`, so the reformulated big-M
MILP stays on the self-hosted `_solve_milp_bb` path with `prefer_pounce=True` —
which turns the integer cut loop on (`_gomory_enabled(True)` → `_cut_int_idx`
populated → `_root_cover_cut_loop` runs GMI/MIR/aggregation/cover). Combined with
`DISCOPT_CMIR_AGGREGATION=1`. Harness
`discopt_benchmarks/scripts/p3_1c_cut_reachability.py`; raw
`results/p3_1c_cut_reachability_panel_20260703T173743.json` (merged from 3
per-instance runs). Equal 2000-node budget, 90 s cap, panel = 5 (ex1263, ex1263a,
3 small graphparts; fac1–3 skipped — already ~1.0 closed, and route to
spatial-McCormick which this lever does not touch). **5 run, 0 skipped.**

| instance | opt | root OFF | root ON | gap-closed ON | cuts ON | path ON |
|---|---:|---:|---:|---:|---:|---|
| ex1263 | 19.6 | 19.063 | 19.063 | 0.000 | **0** | _solve_milp_bb |
| ex1263a | 19.6 | 19.063 | 19.066 | **0.0055** | **8** | _solve_milp_bb |
| graphpart_2pm-0044-0044 | −13 | −16 | −16 | 0.000 | **0** | _solve_milp_bb |
| graphpart_2g-0044-1601 | −9.541e5 | −1.026e6 | −1.026e6 | 0.000 | **0** | _solve_milp_bb |
| graphpart_2pm-0055-0055 | −20 | −25 | −25 | 0.000 | **0** | _solve_milp_bb |

`incorrect_count = 0` on every row (uncapped ON solve certifies the oracle; dual
bound never crosses opt). Gap-closed anchored at discopt's own OFF root-LP floor
(no SCIP anchor per run): `(root_on − root_off)/(opt − root_off)`.

**Verdict — NO-GO / RE-SCOPE. Reachability was the wrong diagnosis; the slice is
too shallow.** The lever *works*: with it on, **all 5** instances route to
`_solve_milp_bb` (vs `_solve_milp_simplex` on the default path) and the cut loop
runs fully armed — a live probe on ex1263/graphpart confirms it enters with
`int_idx_len` 93–120, binaries present, `prefer_pounce=True`. But once reachable
and armed, **the separators find almost nothing**: cover, clique, Gomory,
single-row MIR, and 2-row aggregation c-MIR together fire **0 cuts on 4 of 5
instances** and only 8 on ex1263a — and the root bound moves accordingly:
**median gap-closed = 0.000**, best case ex1263a **+0.55%**, against SCIP's
**~1.000** on this exact class (0b). On graphpart_2pm-0044-0044 the root LP has a
real gap (root −16 vs opt −13) and *still* no separator can cut the LP optimum.

So the 0b/1b hypothesis "the cut seam is unreachable → build it and the class
closes" is **falsified** by direct measurement: making cuts reachable is
necessary but nowhere near sufficient. The residual is **separator depth**, not
plumbing — SCIP's ~100% here comes from cuts our current slice does not produce
(flow-cover, deeper multi-row / lifted aggregation, GUB/knapsack lifting) and/or
a lifted formulation, not from the cover/clique/GMI/1-row-MIR/2-row-c-MIR family
we have. **Do NOT build the invasive Rust `_solve_milp_simplex` cut-callback seam
(1b lever (i)) yet** — it would make the same too-shallow cuts reachable in a
faster engine and close ≤0.55% of the gap. Re-scope Phase 3 build item 2 to
*separator strength* (the depth SCIP actually uses), pinned by its own entry
experiment (e.g. inject SCIP's *specific* cut rounds and read which families
carry the graphpart bound), before any engine-seam work. The
`DISCOPT_P3_FORCE_CUT_PATH` toggle is an experiment lever only (default-OFF,
math-neutral): it changes no default behavior and is not a shipping feature.

This experiment measures only; the one code toggle it adds is default-off and
math-neutral (with it unset the `:3327` reroute is unchanged) — bound-neutral by
construction on the default path, so no correctness gate is weakened.

### Phase 3 1d — separator-family attribution (2026-07-03)

The strength spike the 1c NO-GO demanded: 1c proved the residual is separator
*depth*, not plumbing, but did not say *which* family SCIP uses. This experiment
attributes SCIP's ~100% root close on the graphpart / integer-product class to
specific separators, so the next build targets the right cut instead of guessing.

**Method (SCIP-side only, pyscipopt / SCIP 10.0, `scripts/p3_1d_separator_attribution.py`):**
root-only (node limit 1), **presolve DISABLED** (`presolving/maxrounds=0`) so the
separator set is the *only* mover of the root bound — a clean attribution. Two
anchors per instance — all-separators-OFF (LP floor) and all-ON (full SCIP root)
— then, over all 26 families this SCIP build exposes: **only-one-on** (disable
all, enable exactly F at default freq → gap F closes alone) and **leave-one-out**
(all on, disable F → marginal gap lost). Gap-closed anchored on THIS build:
`(bound − all_off)/(opt − all_off)`, `opt` from `minlplib.solu`. Panel = the 0b/1c
set (ex1263, ex1263a, 3 small graphparts; fac1–3 dropped — already ~1.0). 60 s
cap. **5 run, 0 skipped**; raw JSON
`results/p3_1d_separator_attribution_20260703T180253.json`.

only-one-on root-gap-closed (fraction of reachable root gap closed by F alone):

| instance | all_off→all_on gc | **zerohalf** | gomory | gomorymi | strongcg | any other cut family |
|---|---:|---:|---:|---:|---:|---:|
| graphpart_2pm-0044-0044 | 1.000 | **0.667** | 0 | 0 | 0 | 0 |
| graphpart_2g-0044-1601 | 0.734 | **0.858** | 0 | 0 | 0 | 0 |
| graphpart_2pm-0055-0055 | 0.650 | **0.600** | 0 | 0 | 0 | 0 |
| ex1263 | 0.000 | 0 | 0 | 0 | 0 | 0 (root LP already 19.1 vs opt 19.6; nothing to cut) |
| ex1263a | 1.000 | 0 | 0 | 0 | 0 | `rapidlearning`=1.0 — a **primal heuristic**, not a cut |

leave-one-out marginal loss (gap lost when F removed from the full set):

| instance | **zerohalf** | gomory | gomorymi | strongcg |
|---|---:|---:|---:|---:|
| graphpart_2pm-0044-0044 | **0.533** | 0.333 | 0.167 | 0 |
| graphpart_2g-0044-1601 | **0.153** | −0.124 | −0.114 | 0.112 |
| graphpart_2pm-0055-0055 | **0.383** | 0.050 | −0.017 | 0.050 |

Aggregate (median over the 5): only-one-on **zerohalf 0.600**, every other cut
family 0.000; leave-one-out **zerohalf 0.153**, all others ≤0.05. On no instance
does flowcover / aggregation / cmir / clique / knapsackcover / rlt / mcf /
impliedbounds close *anything*.

**VERDICT — the family that carries the bound is `zerohalf` ({0,½}-Chvátal–Gomory
cuts). Build target: a native zero-half separator.** On the graphpart class
zerohalf alone closes **60–86%** of the reachable root gap and is the single
load-bearing family under leave-one-out (15–53% marginal); gomory / gomorymi /
strongcg are secondary and partly redundant with it (negative marginal on
2g-0044-1601 → SCIP's selection trades them off against zerohalf). Expected
magnitude if discopt gains a working zerohalf separator on this class: **~0.6–0.9
of the root gap** the whole Phase-3 line has been chasing, i.e. the bulk of
SCIP's edge — *not* the flow-cover / lifted-aggregation / deeper-c-MIR guesses the
1b/1c notes floated, and not the aggregation c-MIR already built in #416 (which
closes 0 here). This is a **single-family** target, not diffuse: one separator
(zerohalf) matches most of SCIP on graphpart.

**Scope notes for the build:** (i) zerohalf needs the *integer* constraint rows
(these graphparts read as ~48 int + 1 cont), which the class already carries;
(ii) the two non-graphpart panel members are *not* separator-depth problems and
should not distract the build — ex1263's root LP is already tight (0% closable at
root by anything), and ex1263a's "close" is a primal heuristic (`rapidlearning`)
finding the incumbent, not a cut. So the zerohalf build's measurable target is
the pure-integer graphpart subset, exactly where 0b/1c measured the 0% discopt
close. Measurement-only spike: touches no discopt solver code, so no correctness
gate applies.

---

### Phase 3 zerohalf — build results (2026-07-03)

A native {0,½}-Chvátal–Gomory (zero-half) separator — the family 1d attribution
named as the sole load-bearing one on graphpart — was built and **fully validated**
on branch `cert-p3-zerohalf` (PR #427). It is **NOT merged**: it is *sound but inert*
on the target class, so the code is parked on the branch (revive for the follow-on)
and only this finding is recorded here — no dead flag on `main`.

**What was built (branch `cert-p3-zerohalf`, unmerged).** `lp/zerohalf.rs`
(`separate_zerohalf`): scales rows to integer data, reduces columns to nonnegative
vars (same bound substitution as `mir.rs`), builds the mod-2 parity system over the
active support (Caprara–Fischetti reduction), runs GF(2) elimination for
even-coef/odd-rhs subsets, CG-rounds, maps back. Heuristic (exact min-weight T-join
scoped as follow-on). PyO3 `zerohalf_cuts_py`, wired into the `_solve_milp_bb` root
cut loop behind `DISCOPT_ZEROHALF` (default-off), with a `zerohalf` cut counter.

**Soundness — GREEN.** A {0,½} row combination + CG rounding is valid for the integer
hull by construction for *any* subset; the heuristic affects only strength.
`zerohalf_validity_random_systems` (400 random integer ≤ systems, mixed
int/continuous, mixed-sign bounds, cuttable points) asserts **no integer-feasible
point is ever cut** and **>20 cuts validated**; plus a 200-system binary-dense test
and an odd-cycle regression where single-row MIR finds nothing. `cargo test` 383
passed; clippy clean.

**Lever measurement — MEASURED SHORTFALL, reported honestly.** graphpart panel via
the cut path, root bound at `max_nodes=1`, ON vs OFF:

| instance | opt | scip_all_off | root_off | root_on | zh cuts | gap_closed |
|---|---:|---:|---:|---:|---:|---:|
| graphpart_2pm-0044-0044 | −13 | −16 | −16 | −16 | 0 | 0.000 |
| graphpart_2g-0044-1601 | −9.541e5 | −1.026e6 | −1.026e6 | −1.026e6 | 0 | 0.000 |
| graphpart_2pm-0055-0055 | −20 | −25 | −25 | −25 | 0 | 0.000 |
| ex1263a | 19.6 | — | 19.07 | 19.07 | 0 | 0.000 |

`incorrect_count = 0`. The separator fires the path (loop entered with 384 integer ≤
rows, 93 integer cols) but emits **0 cuts**: at discopt's root LP optimum there is
genuinely nothing to separate.

**Root cause (measured, per §0.4).** discopt's root LP optimum for graphpart is a
**partition vertex with x_j ∈ {0, ⅓}**. At x = ⅓ every {0,½} combination of the tight
rows is **exactly tight, not violated** (exhaustive GF(2) nullspace search: an
even-coef/odd-rhs combo *exists* but its violation is 0.0000). discopt's `root_off`
already equals SCIP's separators-OFF LP floor; SCIP's zerohalf closes 60–86% *because
SCIP's simplex lands on a ½-valued vertex that IS {0,½}-cuttable*, whereas discopt's
optimum (via its own presolve/FBBT + crossover) sits on the ⅓ face where those
inequalities are tight. **The gap is which LP vertex the relaxation optimum occupies
— not separator strength.**

**Re-scope (Phase 3 follow-on, priority order).**
1. **Separate at a ½-valued point, not the ⅓ crossover vertex** — separate zerohalf
   at the pre-crossover fractional point before presolve pins the ⅓ face, or inject
   the odd-cycle rows into a weaker (edge-only) LP whose optimum is ½-valued (SCIP's
   context). A cut-*context* change, not a stronger separator — the likely lever.
2. **Exact/optimal zero-half separation** (min-weight T-join / Caprara–Fischetti) —
   *would not help here* (no violated cut exists at x = ⅓), deprioritized for
   graphpart; the right build for classes whose LP optimum *is* cuttable.

A measured shortfall, not a failure: the separator is correct and the non-movement is
now explained and localized (vertex geometry, not cut depth) — redirecting the next
Phase-3 step from "build a stronger cut" to "separate at a cuttable point."

---

### Phase 3 CUT-1 — aggregation/c-MIR oracle-injection entry experiment (2026-07-06)

The decisive test the 0b GO deferred and 1c/zerohalf circled: **inject SCIP's own
aggregation/complemented-MIR cut coefficients into discopt's root LP and measure the
closure**, on the integer-product family SCIP's c-MIR is the workhorse for
(nvs17/19/24; `scip-gap-closing-plan.md` §1.3). 0b measured only a *proxy* (SCIP's
root bound); CUT-1 measured the real thing. Full write-up:
`docs/dev/cut-engine-entry-2026-07-06.md`. Script
`discopt_benchmarks/scripts/cut1_cmir_oracle_injection.py`; oracle pyscipopt 6.2.1 /
SCIP 10.0.

**VERDICT — NO-GO (relaxation-mismatch).** Two independent measurements:

1. **discopt's default root already dominates SCIP-with-cuts on this family.** Root
   gap closed vs SCIP's separators-off LP floor: nvs17 **discopt 0.9991** (root
   −1105.89, *tighter* than SCIP's cut-root −1105.10), nvs19 **discopt 0.9990**
   (parity, −1104.24 vs −1104.48). discopt reaches this with **no** MIR/aggregation
   cut — via McCormick + on-demand multilinear hull + RLT-1 + PSD/minor cuts +
   iterated lifted-FBBT. `DISCOPT_CMIR_AGGREGATION=1` gives a **bit-identical** root
   (the already-built `lp/aggregation.rs` separator finds nothing violated and
   self-disables). nvs24's root fixpoint does not converge at node 1 in 280 s — its
   residual is relaxation-solve **cost** (Phase 1/2/4), not cut strength.

2. **Injecting SCIP's actual c-MIR cuts closes ≤1.8% / 0% / 0%.** SCIP emits 5/1/1
   c-MIR rows on nvs17/19/24; mapped into discopt's lifted column space (auxvar→
   monomial resolved from SCIP's McCormick-envelope + `minor` PSD-triple row names +
   a numeric anchor) and appended to discopt's root LP, they move the bound
   +12.67/0/0 — **gap-closed 0.0178 / 0 / 0**. The kill criterion (<15% on ≥2 of 3)
   fires. This is the same failure mode as zerohalf on graphpart: the cut is *valid
   but tight, not violated* at discopt's LP vertex — a relaxation-mismatch, not cut
   depth.

**SCIP's cut ablation is SCIP-vs-SCIP.** Re-measured node reduction from SCIP's cuts:
nvs17 119×, nvs19 132×, nvs24 244× — but relative to SCIP's *own* weak no-cut LP
(trivial floor −6404 on nvs17 vs discopt's root −1105.9). Reading it as a 100×
discopt cut gap was the premise error; discopt already starts where SCIP's cut loop
ends on this family.

**Build feasibility (§7 part 2): the separator already exists and is inert here.**
`lp/aggregation.rs` (MW 2-row aggregation, PR #416) + `lp/mir.rs` (MIR with
upper-bound complementation **already implemented** — the plan's "mir.rs:59 lacks
complementation" note is stale, PR #415) + `lp/cut_select.rs`. There is no 4–6 EW
build to green-light; CUT-1 shows a build against this class would be inert.

**Re-scope (measurement wins, §0.4):** do NOT build the §7 part-2 aggregation-c-MIR
separator for the integer-product family. Keep the sound default-off separator
parked (may help a class whose LP optimum *is* c-MIR-cuttable — unproven, not this
one). Re-aim the nvs17/19/24 residual at per-node/root-fixpoint **throughput**
(Phase 1/2) and structure (Phase 4), where the nvs24 measurement points. The §7
"cut engine quality" line is, for the measured integer-product class, **closed as a
relaxation-mismatch** — discopt's relaxation, not its cut set, is what SCIP's cuts
are compensating for, and discopt's is already stronger.

---

## 8. Phase 4 — Stop losing structure (~3–5 EW, medium risk)

The multiplier phase: smaller DAGs make every node, every FBBT sweep, every JAX
compile, and every lifted LP cheaper; recognized structure makes Phases 2–3 stronger.

**Build**
1. **Hash-consing/CSE in the Rust expression arena** (content-addressed node interning
   in `ModelBuilder`); dedup is semantic-preserving by construction.
2. **Preserve `.nl` defined variables** (V segments) as shared DAG nodes instead of
   discarding them (`nl_parser.rs`) — AMPL already computed the sharing; keep it.
3. **Quadratic/Q-matrix extraction in the IR** (upgrade `is_quadratic` from a degree
   check to coefficient extraction), feeding: edge-concave detection without
   re-derivation, RLT on recognized Q structure, and the convexity certificate
   (PSD check on Q) without interval-Hessian work.
4. Wire symmetry/orbit detection (`presolve/symmetry.rs`, currently diagnostic-only)
   into orbital fixing at the root — cheap, sound, general.

**Exit gate:** DAG node count ↓ ≥ 20% on the `.nl` suite with defined-variable-heavy
instances; JAX compile time and per-node eval time ↓ correspondingly; zero behavior
change otherwise.
**Correctness gate:** CSE and V-segment preservation are bound-neutral — exact
node_count/objective equality vs baseline; Q-extraction validated by evaluating both
forms on random points to 1e-12; symmetry fixing under the differential-bound test.

### Phase 4 — re-profile results (2026-07-03)

The §9 note ("the profile changes after Phases 1–2 — re-profile before committing")
demanded this before building any Phase 4 item. Measurement-only harness
`discopt_benchmarks/scripts/p4_reprofile_structure_loss.py` (arena introspection via
`PyModelRepr.get_node`/`arena_len`/`detect_symmetries`/`is_*_quadratic`;
`SolveResult.solver_stats` reduce/*+separate/* timers; XLA compiles via the existing
`perf/measure.count_xla_compiles`). Panel = 8 text-`.nl` instances (5 named probes +
3 larger-DAG general/quadratic). Per-instance 90 s solve cap, `JAX_PLATFORMS=cpu`,
x64. **8 run, 0 skipped.** Raw JSON:
`discopt_benchmarks/results/p4_reprofile_structure_loss_20260703T210635.json` and
`results/p4_xla_compiles_probe.json`.

**Cost breakdown + per-lever potential:**

| instance | DAG nodes | dup% (all) | **op-dup%** (lever 1, actionable) | defvars (lever 2) | quad? (lever 3) | orbits (lever 4) | wall (s) | s/node | XLA % of wall | where the wall goes (solver_stats) |
|---|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---|
| nvs17 | 753 | 68.9 | **37.2** | 0 | deg-2 obj | 0 | 79.1 | 0.85 | 0.1% | separate: edge_concave 32.9 + psd 21.5 + univar_sq 8.0 |
| ex1252 | 336 | 34.2 | 11.6 | 0 | deg-2 (40 con) | 0 | 98.9 | 1.39 | 0.1% | separate/univar_sq 27.6 + reduce/obbt 7.0 |
| ex1252a | 282 | 35.8 | 13.8 | 0 | deg-2 (31 con) | 0 | 66.9 | 4.46 | 0.1% | separate/univar_sq 10.5 + reduce/obbt 1.9 |
| gear4 | 17 | 0.0 | **0.0** | 0 | deg-2 obj | 0 | 49.8 | 0.008 | ~0 | reduce/obbt 14.1 + fbbt 1.3 (6045 nodes — CC3) |
| st_e38 | 46 | 15.2 | 4.3 | 0 | deg-3 | 0 | 2.3 | 0.78 | ~0 | tiny (3 nodes) |
| clay0303hfsg | 2135 | 64.8 | **31.1** | 0 | deg-2 obj | 0 | 73.9 | 0.23 | 1.2% | (no separate timers; MILP path) |
| casctanks | 3124 | 34.0 | 13.1 | 0 | bilinear | 0 | 97.4* | 3.14 | 0.5% | reduce/fbbt 4.0 + univar_sq 1.0 (*hit cap) |
| heatexch_gen1 | 798 | 29.7 | 8.0 | 0 | deg-2 obj | 0 | 171* | 57 | ~0 | reduce/fbbt 0.6 (*hit cap) |

("op-dup%" = duplicate **operator** nodes as a fraction of the DAG, i.e. duplicates
excluding constant literals; the all-in dup% is inflated by repeated constants —
e.g. nvs17's `2.0` literal appears 112× — which CSE would collapse but which cost
~nothing to evaluate. The op-dup fraction is the actionable per-node/relaxation-cost
lever.)

**Per-lever verdict (measured potential):**

- **Lever 1 — CSE/hash-consing: REAL and the only broadly-present structure loss.**
  Operator-node duplication is **31–37% of the DAG on the general-nonconvex class**
  (nvs17 37.2%, clay0303hfsg 31.1%) and 8–14% elsewhere; **0% on gear4** (its DAG is
  already tiny — gear4 is a node-count/CC3 instance, not a structure-loss one). The
  arena's `add` never dedups (`expr.rs`), confirmed by direct walk. This dedup would
  shrink every downstream consumer (JAX trace, lifted LP rows, FBBT sweep, and the
  **separation** work that dominates wall — see below).
- **Lever 2 — `.nl` defined variables (V segments): NO APPLICABLE INSTANCE + a
  separate parse blocker. Premise does not hold on the readable corpus.** Every one
  of the 8 panel instances (and **all 1,558 text-`g` `.nl` files in the MINLPLib
  snapshot**, verified by scanning the "common exprs: b,c,o,c1,o1" header field)
  reports **0 defined variables** — the MINLPLib text export emits none. The
  defined-variable-heavy instances (ACOPF, densitymod, milinfract, portfol_*, …) are
  **all binary-`b`-encoded**, which `nl_parser.rs` rejects outright ("binary .nl
  format not supported") — discopt cannot even *load* them. And when a `g`-format
  file *did* carry a defined-var reference, `nl_parser.rs:347-352` would raise
  "variable index out of range" (the parser drops the V body and cannot resolve a
  reference to `n_vars + j`). So lever 2 has **zero measurable payoff on anything
  discopt can read today**; the prerequisite is binary-`.nl` support, not V-segment
  preservation.
- **Lever 3 — Q-matrix extraction: structure is present but NOT the bottleneck.**
  6 of 8 instances carry recognized degree-2 structure (`is_*_quadratic` True; nvs17
  21 bilinear + 7 monomial terms, casctanks 60 bilinear, clay/ex1252 many quadratic
  constraints) with no coefficient extraction. This *is* real unrecognized structure
  — but the wall it would save is the convexity/PSD + edge-concave separation, which
  is already the dominant cost (nvs17 `separate/psd` 21.5 s + `edge_concave` 32.9 s).
  Q-extraction is a **strength/quality lever bundled with lever 1**, not an
  independent win: it only pays once the DAG it reads is deduped.
- **Lever 4 — symmetry/orbit fixing: NO PANEL SIGNAL.** `detect_symmetries` finds
  **0 nontrivial orbits on all 8 instances** (variables examined 2–500). No
  exploitable orbits on this panel → lowest priority; do not build speculatively.

**Premise check — Phase 4's headline premise is PARTLY FALSIFIED, and one sub-premise
(CC5, from §9) is FULLY FALSIFIED:**

1. **CC5 / "the ex1252 class spends ~14 compiles × ~1 s ≈ the whole solve" is
   FALSIFIED (per §0.4).** Measured exact XLA compiles: nvs17 **8 @ 0.029 s
   (0.1% of wall)**, ex1252 **7 @ 0.025 s (0.1%)**, ex1252a **5 @ 0.021 s (0.1%)**,
   clay0303hfsg 17 @ 0.408 s (1.2%), casctanks 6 @ 0.226 s (0.5%). JAX compilation is
   **≤1.2% of wall everywhere** — the evaluator-cache fix (perf-plan §1 correction)
   already amortized it; §1's "compile resolved" note is confirmed at the solve level.
   **The ex1252 per-node cost (1.4–4.5 s/node) is per-node relaxation/separation
   work, not compile.** Phase 5's compile item and any "compile-once" framing keyed
   to ex1252 are moot.
2. **The dominant wall cost is SEPARATION, not raw DAG-walk/compile.** `solver_stats`
   attributes nvs17's 79 s almost entirely to `separate/edge_concave` (32.9 s) +
   `separate/psd` (21.5 s) + `separate/univariate_square` (8.0 s); ex1252's 99 s to
   `separate/univariate_square` (27.6 s) + `reduce/obbt` (7.0 s). CSE (lever 1) helps
   *because* it shrinks the DAG those separators walk per node — so its payoff is a
   multiplier on the separation cost, exactly the C4-as-multiplier framing — but the
   deduplication itself is not the headline; the separation loop is.

**RANKED build order (with evidence):**

1. **Lever 1 — CSE / hash-consing in the Rust arena. BUILD FIRST.** It is the only
   Phase-4 lever with broad, measured potential: **31–37% duplicate operator nodes on
   the general-nonconvex class** (nvs17, clay0303hfsg), 8–14% on the rest. It is
   bound-neutral by construction (semantic-preserving dedup), so it lands under the
   exact-equality gate with no relaxation risk, and it shrinks the per-node
   separation/FBBT/lift work that the profile shows *is* the wall. Ship it with the
   op-dup metric (not the const-inflated all-dup number) as the exit gauge.
2. **Lever 3 — Q-matrix extraction. BUILD SECOND, coupled to lever 1.** Real
   unrecognized degree-2 structure on 6/8 instances, but its payoff (cheaper
   convexity/PSD + RLT) only materializes on a deduped DAG and feeds the same
   separation loop lever 1 shrinks. Not an independent win; sequence after CSE.
3. **Lever 2 — `.nl` defined variables (V segments). DE-PRIORITIZE / re-scope.**
   **Zero applicable instances** in the entire readable corpus (0 defined vars in all
   1,558 text `.nl`); the defined-var-heavy set is binary-`.nl` that discopt cannot
   parse at all. The real prerequisite is **binary-`.nl` reader support** (a parser
   gap, tracked separately), not V-segment preservation. Building V-segment handling
   now optimizes a code path no readable instance exercises.
4. **Lever 4 — symmetry/orbit fixing. DO NOT BUILD (no panel signal).** 0 nontrivial
   orbits on all 8 instances. Revisit only if a symmetry-bearing class enters the
   gate panel.

**Net:** Phase 4 is **not** the multiplier §8 assumed across the board — its premise
holds *only* for lever 1 (CSE), is bundled-secondary for lever 3, and is falsified
for levers 2 and 4 on the readable corpus. The largest wall lever exposed by this
re-profile is **separation cost** (edge_concave/psd/univariate_square), which is
Phase 3 / relaxation-strength territory, not structure preservation; CSE's value is
as a per-node multiplier on that cost. Recommend scoping Phase 4 down to **CSE
(+ coupled Q-extraction)** and moving binary-`.nl` support to the parser backlog.

This experiment measures only (no solver math changed; the harness reads existing
introspection hooks) — bound-neutral by construction; the sound solves on the panel
(nvs17, gear4, st_e38 optimal) all satisfy `bound ≤ oracle`. No correctness gate applies.
### Phase 4 CSE — build results (2026-07-03)

Built **build item 1**: content-addressed hash-consing (structural interning) in the
Rust expression arena. `ExprArena` gains an opt-in intern table
(`HashMap<StructuralKey, ExprId>`, keyed lookup only — no ordering-sensitive
iteration, so id assignment stays deterministic/byte-reproducible) and an
`intern(node)` method that returns an existing id for any structurally-identical node
instead of appending a duplicate. `StructuralKey` captures op + operand ids + literal
payload (scalar constants keyed by exact `f64::to_bits`, so `0.0`/`-0.0` never merge;
different shapes/types never share a key). **Commutative operands are NOT reordered**
(`a+b` and `b+a` intern separately) — correctness over completeness. Interning is
enabled only during construction and disabled before the model leaves the parser, so
downstream mutating passes (presolve/reformulation) keep plain-append `add` semantics
untouched — the raw `add` is unchanged. Wired into both construction paths: the `.nl`
parser (`nl_parser.rs::parse_nl`) and the Python DAG converter
(`expr_bindings.rs::convert_expr`). Escape hatch `DISCOPT_DISABLE_CSE` reverts to the
pre-CSE build (used to measure the lever; also a fallback).

**Dedup is semantic-preserving by construction:** two nodes share an id only when
their `StructuralKey`s are equal, i.e. same op, same operand ids, same literal
payload/shape — and arena evaluation is a pure function of that structure. Rust tests
assert (a) building the same subexpression twice returns the same id and appends no
duplicate, (b) structurally-different expressions (different var index / op / operand
order / `±0.0`) get different ids, and (c) **evaluation-equivalence**: an interned
build evaluates identically to the naive (non-interned) build of the same expression
on 200 random points to 1e-12, while having strictly fewer nodes.

**The lever — DAG node count, CSE off vs on** (panel `.nl` parsed via
`discopt._rust.parse_nl_file`, `n_nodes`):

| instance | naive nodes | CSE nodes | dup removed | % reduction |
|---|---:|---:|---:|---:|
| nvs17 | 753 | 234 | 519 | **68.9%** |
| clay0303hfsg | 2135 | 752 | 1383 | **64.8%** |
| ex1252 | 336 | 221 | 115 | 34.2% |
| casctanks | 3124 | 2062 | 1062 | 34.0% |
| gear4 | 17 | 17 | 0 | **0.0%** |
| **panel total** | **6365** | **3286** | **3079** | **48.4%** |

The reduction exceeds the re-profile's 31–37% *operator*-dup estimate because the
count also folds duplicate leaf/constant nodes (bound constants, repeated
coefficients). **gear4 = 0.0%** confirms the re-profile prediction exactly (no
structural sharing there) — a clean sanity check that the interner only fuses genuine
duplicates. The value is a per-node multiplier on downstream separation/eval cost
(smaller JAX compile, fewer lifted-LP rows, cheaper FBBT sweeps), per the re-profile.

**Build-time note (measured, honest):** interning adds a small per-node hashing tax at
*parse* time (nvs17 ~30µs→183µs; clay ~169→221µs; casctanks ~504→624µs; ex1252 is
slightly faster). This is a one-time construction cost on the ~10²–10³ µs scale,
amortized many-fold by the smaller DAG across the whole solve; CSE's payoff is the
node-count multiplier downstream, not parse speed (as the re-profile states).

**Correctness gate — BOUND-NEUTRAL, PASS.** `scripts/check_cert_neutrality.py`
reports **NEUTRAL** across the 42-row certifying panel: node_count **exactly
unchanged** vs `cert-baseline.jsonl` on every instance and objective equal to
tolerance. `cargo test -p discopt-core` green (incl. `presolve_determinism`, 4/4),
`cargo clippy --lib` clean, the new CSE unit/property tests pass. Because the interner
only merges structurally-identical nodes, the relaxation math is bit-for-bit the same
— the smaller arena is a representation change, not a math change.

### Phase 4 Q-extraction — build results (2026-07-03)

Built **build item 3** (ranked #2 by the re-profile): exact Q-matrix coefficient
extraction in the IR, plus the first sound consumer (the PSD-on-Q convexity
certificate). Two clearly separated correctness regimes, per §3 / CLAUDE.md §5.

**(A) Exact extraction — BOUND-NEUTRAL foundation.**
`python/discopt/_jax/quadratic_form.py`: `extract_quadratic(expr, n, model) ->
Optional[(Q, c, d)]` returns the symmetric `Q`, linear `c`, constant `d` with
`expr == xᵀQx + cᵀx + d` **exactly, or `None`** (abstain). It layers on the
existing trusted polynomial walker `milp_relaxation._expr_to_polynomial`
(fed `term_classifier.distribute_products`) — the same machinery the edge-concave
collector already uses — and additionally rejects any degree-≥3 monomial. Flat
indexing is the identical prefix-sum layout the convexity certificate uses
(`interval_ad._var_offset` == `term_classifier._compute_var_offset`), so a `Q`
here is directly consistent with `certify_convex`'s coordinate system. Symmetric
split: `Q_ij = Q_ji = ½·coeff` (i≠j), `Q_ii = coeff`. Helpers `quadratic_is_psd`
/ `quadratic_is_nsd` do the exact `eigvalsh` test.

*Validation (`test_quadratic_form.py`, 39 tests):* random quadratics reconstruct
to **1e-12 on 250 points/instance** across 12 seeds; the symmetric-split, affine,
and constant sub-cases are pinned; **14 non-quadratic shapes** (cube, quartic,
trilinear, `x²·y`, exp/log/sin/sqrt, bilinear-with-transcendental, var-in-
denominator, fractional power, `abs`, reciprocal) all return `None` — never a
mis-extracted `Q`; out-of-range flat index abstains; PSD/NSD helpers match
`eigvalsh` on 20 random symmetric matrices.

**(B) Convexity certificate via PSD-on-Q — BOUND-CHANGING, flag default-OFF.**
Wired into `convexity/certificate.py::certify_convex` behind
**`DISCOPT_PSD_QFORM`** (default `0`). On a purely quadratic body the Hessian is
the *constant* matrix `2·Q`, so `λ_min(Q) ≥ 0` is a rigorous, box-independent
convexity proof — strictly tighter than the conservative interval-Hessian +
Gershgorin row-sum enclosure (which abstains on tight-but-PSD matrices, e.g.
`Q` with all off-diagonals 0.99, `λ_min≈0.01`). On any abstention (non-quadratic
body, indefinite `Q`, non-finite `Q`) it returns `None` and **falls through to the
existing rigorous path unchanged** — it never assumes convex. Because it can prove
*more* bodies convex, it changes node relaxations/counts → shipped behind a flag
per the bound-changing regime.

*Differential/soundness result (`test_psd_qform_convexity.py`, 78 pass / 4 skip):*
(1) **No mis-certification** — 30 random *indefinite* `Q` (both eigen-signs,
away from 0) never read as CONVEX/CONCAVE with the flag ON. (2) **Strict
refinement** — over 30 random PSD/NSD/indefinite cases the flag-ON verdict never
flips a flag-OFF verdict; it only turns `None` into a verdict, and always agrees
with a direct `eigvalsh` check (20 seeds). (3) **Non-quadratic bodies are
identical on/off** (flag only touches the purely quadratic case). (4) Flag default
reproduces the conservative abstention. Full `-k "convex or quadratic or rlt or
edge_concave or qmatrix or psd"` suite: **786 pass / 5 skip / 2 xfail** with the
flag default-off.

**Bound-neutrality of the default (flag OFF):** the PSD path is behind
`_psd_qform_enabled()` (only active when `DISCOPT_PSD_QFORM ∈ {1,true,yes,on}`), so
with the flag unset `certify_convex` is byte-for-byte the prior function — the
`check_cert_neutrality.py` panel is unaffected. Enabling and validating on nightlies
(differential-bound + feasible-point + `incorrect_count=0`) is the follow-on before
default-on.

**Scoped as follow-on (proven partial > unproven whole, §8):** RLT and edge-concave
already extract quadratic coefficients via `_expr_to_polynomial` (`edge_concave.py::
collect_edge_concave_quadratics`, `rlt_cuts.py`), so rerouting them through
`extract_quadratic` would at best be a bound-neutral refactor with no payoff (same
coefficients, same verdict) and at worst risk a subtle drift; it is not shipped in
this PR. The value there is a *faster path to the same relaxation*, which must be
proven exactly bound-neutral (node_count/objective unchanged) — a separate task.

### Phase D re-profile (2026-07-05) — the POUNCE subsolver is the #1 wall lever

Re-profile on current `main` (post-CSE/Q-extraction/Rust-1). The §8 re-profile named
"separation cost" as the dominant wall; this Phase-D pass attributes *where inside
separation the time goes* — the **POUNCE subsolver**, called cold once per
separation/strong-branch LP:

- **nvs17**: `separate_edge_concave_quadratic → lp_pounce.solve_lp` = **85 calls /
  4.86 s**; strong-branching → `lp_pounce.solve_lp` = **154 calls / 2.22 s**. Together
  **~239 cold POUNCE-LP solves / 6.3 s ≈ 40% of nvs17 wall**, while the in-house Rust
  simplex (`solve_lp_warm_py`) solves the same LPs at ~4–15 ms.
- **Levers A/B/C status recorded:** CSE (Phase-4 lever 1) is **done and NOT a wall
  lever by itself** (it shrinks the DAG the separators walk, but the separator's own
  subsolver cost dominates); Q-extraction (lever 3) done/flagged; the earlier per-node
  warm-start / Python-tax levers (T1.4/T1.6) were **measured dead**. The live lever is
  the **separation/strong-branch subsolver backend**.

**Lever taken (`perf-d1`): route the separation + strong-branch LPs to the warm
in-house simplex.** The edge-concave separator hard-coded `lp_pounce.solve_lp`
(`edge_concave.py`); strong branching used `get_lp_solver(prefer_pounce=nlp_solver==
"pounce")`, i.e. POUNCE on the default `nlp_solver="pounce"`. Both now route to
`lp_simplex.solve_lp` under `DISCOPT_SEPARATION_LP_SIMPLEX` (default **ON**; `"0"`
restores POUNCE).

**Confirm-first (does the routing preserve the derived output?).** Captured ≥50 of
each LP on nvs17 + nvs11/12/13 during a live solve and solved each with *both*
backends:
- *Edge-concave (load-bearing output = the derived cut, not the vertex):* the two
  backends **disagree on the dual slope** on the degenerate vertex-hull LP (|ΔA|∞ up to
  1e10; IPM analytic-center dual vs simplex vertex dual) — but the separator recomputes
  the intercept `B` to the exact validity boundary, so **both cuts are sound for any
  slope**, and on the captured panel the **cut verdict matched 239/239** (every LP was
  `noviol` for both — these instances produce no edge-concave cut). So the routing is
  *not* byte-identical to POUNCE (different valid slope on a degenerate LP); its
  neutrality is decided empirically, not by vertex identity.
- *Strong-branch (load-bearing output = the objective bound used in the argmax score):*
  objective agreement to LP tolerance (|Δobj| median 4.6e-6, max 1.1e-4 over 457 LPs);
  the small wobble did not flip any branch argmax.

**Result — strictly bound-neutral, measured:**
- **Cert-baseline neutrality (`check_cert_neutrality.py`, `JAX_PLATFORMS=cpu`,
  `JAX_ENABLE_X64=1`, flag ON): NEUTRAL.** All **41/41** certifying instances
  `node_count` EXACTLY unchanged and `|Δobj| = 0.00e+00`, all `optimal` →
  `incorrect_count = 0`. nvs17 (not in the baseline) verified separately: at a
  generous budget both flag states finish `optimal` at **exactly 93 nodes, obj
  −1100.4** — the branch decisions are identical.
- **T0.4 soundness harness through the new path: PASS.** 37 real edge-concave cuts
  derived via the simplex slope (nvs14/nvs11), **14,800 feasible-point validity checks,
  no invalid cut** (the intercept-recompute keeps every cut sound regardless of slope).
- **Measured win (POUNCE calls / wall / s-node), `nlp_solver="pounce"`, 60 s cap:**

  | instance | POUNCE calls OFF→ON | wall OFF → ON | s/node OFF → ON | note |
  |---|---:|---|---|---|
  | nvs17 | 848 → **0** | 60.3 s (TL, feasible) → **38.1 s (optimal)** | 0.826 → **0.409 (2.02×)** | now certifies within budget |
  | nvs13 | 164 → **0** | 3.86 s → **1.44 s** | 0.203 → **0.076 (2.67×)** | |
  | nvs11/12/14 | 0 → 0 | unchanged | unchanged | no separation/SB POUNCE calls; neutral |

  nvs17's 73→93 node difference at the 60 s cap is the *timeout truncating the slower
  OFF run mid-search*, not a bound change (equal-budget node count is identical, above).

**Shipped default-ON** (strictly bound-neutral: node_count + objective exactly
unchanged on the cert panel and nvs17). A documented off-switch
(`DISCOPT_SEPARATION_LP_SIMPLEX=0`) is retained because the edge-concave slope is not
byte-identical across backends (degenerate-LP dual), so an operator can pin the legacy
POUNCE path if a future instance ever shows a cut-selection difference; soundness holds
either way.

### Phase D — H-D1 NLP-incumbent lever: NO-GO (2026-07-05)

The Phase-D profile also flagged a *second* POUNCE cost centre distinct from the
separation LPs above: the **NLP-incumbent solves** (feasibility_pump, node_nlp_attempt,
subnlp), ~2.3 s each on ex1252a. H-D1 asked whether these are *redundant* work discopt
could elide (warm-seeding / dedup across the pump + node + subnlp passes).

**Entry experiment (run before any implementation, per §0.4):** measured cross-pass
redundancy at **13.5 %** (< the 30 % GO threshold) and a warm-seed speedup of **≈1.0×**
(< the 1.3× threshold). Both kill criteria fired → **H-D1 is a NO-GO; the
NLP-incumbent-dedup lever is NOT built.** The incumbent NLPs are genuine, non-redundant
local solves — the cost is *inside the subsolver*, not in discopt's orchestration of it.

### Phase D — POUNCE-vs-Ipopt on identical incumbent NLPs (2026-07-05)

Before filing a POUNCE follow-up for that in-subsolver cost, we verified it is not a
discopt-side artefact by A/B-ing the **same** NLP subproblems against Ipopt (cyipopt
1.7.0) vs POUNCE (0.8.0). Both backends are driven through discopt's shared
cyipopt-compatible callback adapter and the identical `solve_nlp(evaluator, x0,
constraint_bounds, options)` entry, so the only variable is the solver. Monkeypatch-only
instrumentation, **no solver-math change**. `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`,
threads=1, in-process (no daemon). Instances ex1252a / heatexch_gen1 / ex1252 / casctanks
/ st_e31.

**Level 1 (end-to-end, NLP-solve wall / count per backend):** the incumbent NLPs are
dramatically cheaper under Ipopt on every instance —

| instance | ipopt NLP-wall / n | pounce NLP-wall / n | per-solve |
|---|---|---|---|
| ex1252a       | 6.4 s / 137 | 34.8 s / 47 | ~16× |
| heatexch_gen1 | 16.2 s / 36 | 71.4 s / 8  | ~20× |
| ex1252        | 5.2 s / 52  | 26.0 s / 23 | ~9×  |
| casctanks     | 9.2 s / 73  | 19.0 s / 21 | ~7×  |
| st_e31        | 3.3 s / 91  | 11.7 s / 52 | ~6×  |

(POUNCE runs fewer, far slower solves; on ex1252a/heatexch it exhausts the wall on a
handful of NLPs and explores 3–29 nodes to Ipopt's 17–63.)

**Level 2 (apples-to-apples per-solve replay from the identical x0):** captured the
expensive (>0.5 s) POUNCE subproblems live and re-solved each with both backends.
Pooled over **57** subproblems: `t_pounce/t_ipopt` **median 7.1×, mean 8.0×, 93 %
≥3×**. Restricting to the **24** subproblems where *both* backends converge to the
**same** optimum (both `OPTIMAL`, |Δobj| ≤ 1e-5 rel): **median 10.9×, min 3.6×, max
12.9×** — every one ≥3.6×. casctanks and st_e31 are 100 % same-optimum / 0 non-OPTIMAL
(clean isolation); ex1252a / heatexch_gen1 / ex1252 mix in a **multimodality caveat** —
on the harder NLPs both solvers hit their iteration budget and land on *different* local
optima, so those solves measure solver-choice divergence, not pure speed.

**Verdict — POUNCE perf problem: YES, worth filing.** On the clean same-optimum subset
POUNCE is a large, systematic outlier (≥3.6×, median ~11× slower than Ipopt reaching the
*identical* KKT point), a genuine solver-speed gap — not a discopt orchestration lever
(the H-D1 NO-GO above rules out the redundancy hypothesis). A minimal, self-contained
reproducer was drafted (casctanks incumbent NLP, n=560/m=577: POUNCE 0.454 s / 33 iters
vs Ipopt 0.039 s, same obj 8.7904272, **11.6×**) for a POUNCE issue. The multimodality
caveat is noted so the filing does not over-claim on the harder instances.

---

## 9. Phase 5 — The JAX residual (~2–4 EW, scope set by data)

After Phases 1–2 the profile changes; re-profile before committing. Expected residual:
JAX evaluation in NLP local solves (primal heuristics) and α-BB/interval work, plus
the few-but-expensive relaxation compiles (perf-plan CC5, ex1252 class: 14 compiles ×
1.08 s ≈ the whole solve).

**Candidate build items (choose by the post-Phase-1 profile, not now):**
- Compile-once-per-model relaxation functions keyed by structure fingerprint (bounds
  as *arguments*, never baked into the trace — removes box-driven recompiles).
- Batch node NLP evaluations across open nodes (the batch machinery exists).
- Move interval/FBBT-style evaluation fully to Rust where JAX adds no value.

**Exit gate:** `jax_time_fraction` ≤ 0.25 panel-wide at unchanged node counts;
ex1252-class compile count ≤ 2 per solve.
**Correctness gate:** bound-neutral regime (same math, different execution).

---

## 10. Sequencing & critical path

```
Phase 0 (observability + soundness harness)
   └─► Phase 1 (general node engine)  ── the enabler: cheap nodes make per-node
            │                             reduction (P2) and separation (P3) affordable
            ├─► Phase 2 (branch-and-reduce loop)   ── biggest bound lever
            ├─► Phase 3 (c-MIR + cut pool)         ── integer-product class
            └─► Phase 4 (structure preservation)   ── independent, can start early
                     └─► Phase 5 (JAX residual)    ── scoped by re-profile
```

- Phases 2, 3, 4 are mutually independent once Phase 1's engine exists; 2 and 3 both
  consume Phase 0's differential-bound harness.
- Phase 4 items 1–2 (CSE, V-segments) have no dependency at all and can run in
  parallel with Phase 1 if staffing allows.
- **Critical path to the headline metric** (median ≤ 2.5× BARON): 0 → 1 → 2.
  Phase 3 governs the *tail* (integer-product families); Phase 5 governs the last
  fraction after the loop is right.

## 11. Gate wiring (benchmarks.toml)

Add per-phase criteria in the existing `[gates.*]` style; every gate keeps
`zero_incorrect = { max = 0, metric = "incorrect_count" }`:

```toml
[gates.cert0.criteria]   # Phase 0
root_gap_coverage   = { min = 0.9,  suite = "global50", metric = "root_gap_populated_fraction" }

[gates.cert1.criteria]   # Phase 1
interop_overhead    = { max = 0.10, suite = "global50", metric = "python_orchestration_fraction" }
sec_per_node        = { max = 0.018, suite = "global50", metric = "median_seconds_per_node" }
bound_neutrality    = { max = 0,    suite = "certifying_panel", metric = "node_count_drift" }

[gates.cert2.criteria]   # Phase 2
root_gap_vs_baron   = { max = 1.3,  suite = "global50", metric = "root_gap_ratio_vs_baron" }
geomean_vs_baron    = { max = 2.5,  suite = "global50", metric = "geomean_time_ratio_vs_baron" }

[gates.cert3.criteria]   # Phase 3
cut_node_reduction  = { min = 2.0,  suite = "integer_product", metric = "node_ratio_cuts_off_over_on" }
no_offtarget_regression = { max = 1.05, suite = "global50", metric = "wall_ratio_vs_baseline" }
```

> **Wiring corrections (2026-07-02 recon, per §0.4):** (1) no `[suites.global50]`
> table exists in `benchmarks.toml` — the panel lives only as
> `config/baron_global50.txt`; T2.6 must define the suite before any cert gate on
> it can resolve. (2) `geomean_time_ratio_vs_baron` was never implemented; the
> existing metric is `geomean_ratio_vs_baron` (metrics.py:331, dispatch :743-746)
> — cert2 uses that name. (3) cert1's `node_count_drift` was superseded by the
> out-of-band differential check (`scripts/check_cert_neutrality.py` against the
> 42-row `cert-baseline.jsonl`), per the T1.2 resolution — there is no
> `certifying_panel` suite and none is needed.

## 12. Out of scope (deliberately)

- **New envelope math** — the catalog shows parity; effort there is not the gap.
- **Branching-rule work** — measured ≤ 15% (SCIP ablation) and the 06-24 entry
  experiment showed the slow instances are bound-limited, not order-limited.
- **Parallel tree search** — real but orthogonal; BARON's advantage is not parallelism.
- **Instance-specific tuning** (gear4, nvs17, …) — named instances are gate probes
  only; any change that helps only a named instance is rejected by construction.
- **Decomposition** — covered by the decomposition-advisor track; it addresses the
  large-instance tail, not the median gap this plan targets.

## 13. Risks

1. **False certificates** (Phases 2–3): the only catastrophic failure mode. Every
   bound-changing item runs under the differential-bound test + feasible-point
   sampling + adversarial suite + 3 green nightlies before default-on.
2. **Patch-table incompleteness** (Phase 1): a term type whose envelope rows are not
   box-affine silently diverges — caught by the bound-neutrality exact-equality gate;
   mitigated by per-term-type cold-rebuild fallback.
3. **Cut-quality shortfall** (Phase 3): the central measured risk (current cuts are
   net-negative). Phase 0b's SCIP-cut-injection experiment resolves it before Rust is
   written.
4. **Loop overhead** (Phase 2): reduction that doesn't pay for itself. The escalation
   policy is budgeted and efficacy-triggered; the no-offtarget-regression gate is
   binding.
5. **Plan staleness**: the 06-24 pass overturned three earlier drafts. Every phase's
   entry experiment re-validates its premise against the then-current baseline before
   code is written.

---

## 14. Handoff appendix — executable task breakdown (Phases 0–2)

Phases 0 and 1 need no further discovery and are specified here to task granularity.
Phases 2–5 are *deliberately not* specced to this level: each begins with an entry
experiment (§6–§9) whose result determines the work items; the implementing agent
runs the experiment first and writes the task list from its output, in this file,
before coding. **Read §3 (verification regimes) before starting any task.**
*(Amended 2026-07-02: Phase 2 is now specced below at maintainer direction —
"Phase 2 tasks" — with its coding tasks T2.3–T2.5 still conditional on T2.1's
recorded results, preserving §0.1.2.)*

### Phase 0 tasks

**T0.1 — Populate `root_gap` / `root_time`.**
- Fields already exist: `discopt_benchmarks/benchmarks/metrics.py:48-49`
  (`SolveResult.root_gap`, `.root_time`); aggregation already exists
  (`root_gap_analysis` metrics.py:399, `root_gap_ratio` :416, gate dispatch
  `root_gap_ratio_vs_*` :726-729). Only the *producer* is missing.
- In `python/discopt/solver.py`, capture the global lower bound and elapsed time at
  the moment the root node is fathomed/branched (the root McCormick/OBBT block ends
  near the `_root_cut_pool` construction, `solver.py:4342-4380`; the three-bucket
  timer split at `solver.py:3401` shows the accumulation pattern to copy). Expose
  both on discopt's public `SolveResult` (modeling/core.py), then map them in the
  benchmark adapter that builds `benchmarks.SolveResult` (see how `node_count` flows
  through `discopt_benchmarks/benchmarks/runner.py`).
- For BARON/SCIP rows: parse root bound from their logs where available; else leave
  null (the ratio metric skips nulls — metrics.py:421-422).
- **Test:** a smoke-suite run has non-null `root_gap` for every discopt row;
  `evaluate_phase_gate` computes `root_gap_ratio_vs_baron` without KeyError.

*Implementation note (done, this PR).* Added `root_bound` / `root_gap` /
`root_time` to the public `SolveResult` (`modeling/core.py`). Producers wired
into every in-house solve path that a benchmark row can hit: the spatial B&B
loop (`solve_model`), NLP-BB (`_solve_nlp_bb`), MILP-BB (`_solve_milp_bb`),
MIQP-BB (`_solve_miqp_bb`), and the convex fast path (`_solve_continuous`, where
the single root NLP *is* the whole solve). Each B&B driver snapshots the tree's
`global_lower_bound` (internal-min sense) and elapsed wall clock at the end of
iteration 0 — after the root batch is processed, before the first branch — then
converts to the reported sense (mirroring `bound`'s MAXIMIZE negation) and
computes `root_gap = |objective − root_bound| / max(1, |objective|)` against the
final incumbent. The spatial path also adopts the strengthened root cut-pool
bound (`_root_pool_bound`) when tighter. The benchmark adapter
(`runner.py`) maps both fields onto `benchmarks.SolveResult`. Verified: benchmark
`--suite smoke` yields non-null `root_gap` on all 10/10 rows; `root_gap_ratio`
/ `evaluate_phase_gate('…root_gap_ratio_vs_baron')` compute with and without a
baron reference (no KeyError). Bound-neutral: node_count and objective
bit-identical vs `main` on 12 instances (0 drift) — pure read-only additions,
no change to bound/branch/control flow. `_solve_milp_simplex` (one-shot Rust
MILP, no Python loop) and external-solver rows are intentionally left null; the
ratio metric skips nulls and no benchmark smoke row currently takes that path.

**T0.2 — Bound-trajectory recording.**
- discopt already has a `node_callback` exposing `best_bound`/`incumbent` (used in
  the 06-24 measurement, performance-plan §6). Add an opt-in recorder in
  `discopt_benchmarks/benchmarks/runner.py` that stores `(t, node, bound, incumbent)`
  tuples (downsampled, ≤ 500 points) into the result JSON under `trajectory`.
- **Test:** trajectory present and monotone in `t`; bound non-decreasing (min sense).

*Implementation note (done, this PR).* Added `record_trajectory` /
`trajectory_max_points` to `BenchmarkConfig` and a `trajectory` field to
`benchmarks.SolveResult`. When opted in, `_run_discopt` attaches a
`node_callback` recording `[t, node, bound, incumbent]` per B&B iteration
(`ctx.best_bound` is the tree's internal-min `global_lower_bound`, hence
non-decreasing), then `_downsample_trajectory` caps it to ≤ `max_points`
preserving both endpoints. **Default OFF**, and this is load-bearing for
neutrality: attaching *any* `node_callback` disables discopt's auto-GP fast path
(`solver.py` GP probe requires `not _has_bb_callbacks`), which would change
`node_count` on geometric-program instances. Verified: `--suite smoke` node
counts bit-identical to the T0.1 run with the recorder off (trajectory `null` on
all rows); with it on, `gear` yields a 7-point trajectory, monotone in `t`,
bound non-decreasing. Tests in
`discopt_benchmarks/tests/test_cert_instrumentation.py`.

**T0.3 — Reduction/separation timers.**
- Rust: add `Fbbt` and `NodeReduce` phases to the `Phase` enum in
  `crates/discopt-core/src/profile.rs` (pattern: existing `NodeLpSolve`,
  `StrongBranch`); time `tighten_bounds` in `bnb/milp_driver.rs:785-824`.
- Python: wrap the per-node separation chain (`mccormick_lp.py:708-731`) and the
  OBBT/nonlinear-FBBT calls (`solver.py:4756-4765`) with the existing perf-counter
  budget pattern; surface per-family totals on `SolveResult.solver_stats`.
- **Test:** on one spatial instance, the new timers sum to ≤ wall and are non-zero.

*Implementation note (done, this PR).* **Rust:** added `Fbbt` and `NodeReduce`
to the `timed_phases!` macro in `profile.rs`; wrapped the per-node
`tighten_bounds` (`milp_driver.rs` node loop) under `Fbbt` and the root presolve
`tighten_bounds` under `NodeReduce`, using the existing zero-overhead
`Timer::new(Phase::…)` RAII (same pattern as `NodeLpSolve`/`StrongBranch`).
These surface through the existing `DISCOPT_PROFILE` stderr dump — verified both
fire and record (`NodeReduce` on any MILP presolve; `Fbbt` on a branching MILP
with `node_propagation=True`). Because the phase-count `NP` is derived from the
macro variant list, adding variants is self-consistent — no hardcoded count.
**Python:** per-family separation timers accumulate on
`MccormickLPRelaxer._sep_timers` (multilinear/edge_concave/univariate_square/
convex/psd/rlt); FBBT time in the spatial loop's `_reduce_timers`; OBBT reuses
the existing `_pn_obbt_spent` accumulator. All are surfaced as a flat
`reduce/…` / `separate/…` dict on the new public `SolveResult.solver_stats`
field, mapped to nothing external (Python-only). Verified on `gear`/`ex1221`/
`ex8_1_1`: timers non-zero and summing to ≤ wall. **Bound-neutral:** the Rust
Timer is inert unless `DISCOPT_PROFILE` is set and never alters math; after the
`--release` rebuild, node_count and objective are bit-identical to `main` on 12
instances (0 drift).

**T0.4 — Reusable soundness harness.**
- New module `discopt_benchmarks/utils/soundness.py`:
  `assert_bound_sound(relaxer_fn, boxes, oracle_fn, tol)` (new bound ≥ old bound − tol
  AND ≤ oracle + tol on every box) and
  `assert_cut_valid(cut, feasible_points, tol)` (no feasible point violated).
  Oracle = dense multistart solve (reuse `_solve_root_node_multistart`,
  solver.py:1304, with tight budgets) or stored known optima from the global50
  oracle file used by `incorrect_count`.
- **Test:** harness flags a deliberately-invalid cut (unit fixture) and passes the
  existing multilinear-separation soundness sweep.

*Implementation note (done, this PR).* New module
`discopt_benchmarks/utils/soundness.py` with `assert_bound_sound(relaxer_fn,
boxes, oracle_fn, tol, *, baseline_fn=None, sense="min")` — validity
(`bound ≤ oracle + tol`, the false-certificate guard) plus optional
non-regression vs a baseline relaxer (the §3 differential test), both senses —
and `assert_cut_valid(cut, feasible_points, tol)` for cut validity (no feasible
point removed). Plus a `known_optimum_oracle` convenience for the stored
global50 optima. Deliberately solver-internal-free (callers pass callables/
arrays) so it is cheap to unit-test and Phases 2–4 can import it anywhere.
Tests (`discopt_benchmarks/tests/test_soundness_harness.py`, 8 passed): the
harness flags a planted invalid cut and a planted bound-crossing, and passes a
200-box sweep of the McCormick bilinear envelope against sampled feasible points
(the multilinear-separation soundness sweep). Bound-neutral by construction — no
solver-path or Rust change.

**T0.5 — Baseline.** Run global50 + perf panel with T0.1–T0.3 on; commit JSONL next
to `docs/dev/data/perf-baseline.jsonl` as `cert-baseline.jsonl`. Add `[gates.cert0]`
to `discopt_benchmarks/config/benchmarks.toml` (§11) and a
`root_gap_populated_fraction` metric function in `metrics.py` (dispatch at :685).

*Implementation note (done, this PR — Phase 0 complete).* Added
`root_gap_populated_fraction` to `metrics.py` (+ dispatch in
`evaluate_phase_gate`) and `[gates.cert0]` (root_gap_coverage ≥ 0.9 +
zero_incorrect ≤ 0) to `benchmarks.toml`. The gate is now honestly evaluable via
`run_benchmarks.py --gate cert0`: fixed the TOML-gate path (Mode 2) to pass
`known_optima`, sourced self-containedly from a committed
`docs/dev/data/cert-optima.json` (global50 BARON optima + perf-panel oracles,
37/55 instances — the rest simply aren't incorrect-checked, like the perf gate's
best-effort) merged with the optional MINLPLib cache. Baseline generated by
`discopt_benchmarks/scripts/gen_cert_baseline.py` over global50 ∪ perf panel (55
vendored rows) → `docs/dev/data/cert-baseline.jsonl` (the §0.2.5 neutrality
reference). **Measured result:** `run_benchmarks.py --gate cert0` → PASS —
root_gap coverage **0.909 ≥ 0.90**, incorrect_count **0**.

*Falsification recorded (per §0.4).* T0.1 initially left `root_gap` null on the
`_solve_milp_simplex` one-shot Rust MILP path (7 `nvs*` instances) — coverage
was 0.782. Fixed by recovering the root bound there from one continuous-
relaxation LP solve (integers relaxed; the result is not fed back, so the solve
stays bound-neutral — node_count/objective unchanged). The 5 residual nulls
(carton7, casctanks, flay03m, hda, tls2) are all `time_limit` exits that never
finish the root relaxation within budget — a legitimate "no root bound" signal
(and exactly the per-node-cost problem Phase 1 targets), not a producer gap.

### Phase 1 tasks

Anchors: `python/discopt/_jax/incremental_mccormick.py` (`IncrementalMcCormickLP`,
line 68), scope gate `_is_in_scope` (`_jax/lp_spatial_bb.py:50` — currently
minimize + all-integer), wiring probe `mccormick_lp.py:331`, per-node entry
`solve_at_node` (`mccormick_lp.py:524`).

**Load-bearing existing property (do not weaken):** `IncrementalMcCormickLP`
self-validates its closed-form rows against `build_milp_relaxation` on random boxes
at construction; any mismatch sets `ok=False` and the caller falls back to the
trusted cold builder (`incremental_mccormick.py:13-23`). The generalization strategy
is therefore: *extend coverage term-type by term-type; validation guarantees that
anything not yet covered falls back rather than mis-solving.*

**T1.1 — Entry experiment (kill criterion, ~1–2 days).**
Bypass `_is_in_scope` on three out-of-scope shapes: (a) a maximize model, (b) a
mixed integer+continuous bilinear model, (c) a general-NL instance from the
`casctanks`/`st_e38` class. Record which term types trip `_validate()` (i.e. which
box-dependent row families the patch table is missing) and what fraction of node
time the cold path spends on them. Deliverable: a coverage table (term type →
rows-per-term → closed-form available yes/no) appended to this section. Kill: if the
worst instances' lifted LPs are dominated by term types with no closed-form
box-dependence (unlikely — all envelope rows are functions of the box by
construction), re-scope Phase 1 to structure caching without row patching.

*Result (done, this PR — kill criterion did NOT fire; proceed to T1.2).* Ran the
incremental engine (`IncrementalMcCormickLP(model, terms)`, which self-gates on
`_validate` independently of `_is_in_scope`) on the three shapes. The engine
covers exactly **bilinear** (4 McCormick rows) and **monomial²/integer-square**
(2 tangents + 1 secant) today; everything else trips `_validate` → cold-path
fallback. Coverage table (families observed across the probes):

| Term family | rows/term | envelope | closed-form in the box? | covered today | seen in |
|---|---|---|---|---|---|
| bilinear `x_i·x_j` | 4 | McCormick | yes (bilinear in `l,u`) | ✅ | ex1263, st_e38 |
| monomial² `x_i²` | 3 | 2 tangent + secant | yes (quadratic in `l,u`) | ✅ | st_e38 |
| monomial ≥3 `x_i^p` | tangents + secant | power envelope | yes (polynomial in `l,u`) | ❌ | st_e38 |
| trilinear `x_i·x_j·x_k` | RLT bound-factor | recursive McCormick | yes (polynomial in `l,u`) | ❌ | st_e38 |
| univariate (exp/log/…) | secant + tangent(s) | convex/concave OA | yes (secant = box corners; tangents at endpoints) | ❌ | syn05m |
| multilinear ≥4, fractional_power | RLT / power | — | yes | ❌ | — (not in probe set) |

Per-instance verdicts (cold `build_milp_relaxation` cost in parens):
- **ex1263** — mixed **72 int/bin + 20 continuous**, MINIMIZE, pure bilinear (16
  products): **`ok=True`** (4.95 ms/call). The engine *already validates on a
  mixed integer+continuous model* — the `_is_in_scope` all-integer restriction is
  over-conservative, directly de-risking T1.3.
- **syn05m** — MAXIMIZE, 5 int + 15 continuous: `ok=False`, tripped by
  `univariate` terms (1.80 ms/call). The **maximize sense is not the blocker** —
  `_validate` compares only the box-dependent rows/aux-bounds, which are
  sense-independent; a synthetic maximize *bilinear-only* model validates
  `ok=True`. So T1.3's "normalize sense" is a non-issue for the engine.
- **st_e38** — general-NL, MINIMIZE: `ok=False`, tripped by `trilinear` +
  `monomial³` (0.83 ms/call) — the intended negative control.

**Kill-criterion verdict: did NOT fire.** Every uncovered family observed
(univariate, monomial³, trilinear) is a closed-form function of the box (all
McCormick/secant/tangent/RLT rows are polynomial in `[l,u]`); none is
box-independent. Row-patching (T1.2) is therefore viable, and T1.3's scope gate
should become simply "`_validate` passed (`ok=True`)" for any var mix / any sense
— not a per-class fork. Reproduce with
`discopt_benchmarks/scripts/t11_entry_experiment.py` (probes syn05m / ex1263 /
st_e38).

**T1.2 — Extend the patch table.**
For each term family in `milp_relaxation.py`'s lifted LP, add the closed-form row
generator + aux-bound function to `incremental_mccormick.py` (pattern:
`_bilinear_rows`/`_square_rows`, lines 34-52): univariate envelopes
(tangent+secant rows — coefficients are closed-form in `[li,ui]`), integer powers
p ≥ 3, trilinear/multilinear RLT rows (bound-factor products — polynomial in the
bounds), fractional/reciprocal rows. After each family: `_validate()` must pass on
randomized boxes including negative/zero-spanning ones (the current probe box is
strictly positive — extend `_build_structure`'s probe to exercise sign regimes,
`incremental_mccormick.py:100-104`).
- **Test per family:** property test — patched (A,b,bounds) equals the cold builder's
  to 1e-9 on 200 random boxes; plus the existing construction-time validation.

*Falsification recorded (2026-07-02, per §0.4 — pre-implementation row probe;
reproduce with `discopt_benchmarks/scripts/t12_probe.py`).* The task premise
"coefficients are closed-form in `[li,ui]`" with a **fixed** row structure holds
for the **product families** but not the **power/univariate families**: the
envelope row *count* is sign-regime-dependent for anything that can change
convexity across zero.
- Measured: `x²` → **3 rows** on a sign-definite box, **4 rows** when the box
  strictly spans zero (an extra `s ≥ 0`); `x³` → **3 rows** on `[l,u]⊂ℝ₊`, **2**
  when spanning; `x⁴` → 3 vs 4. **bilinear is always 4 rows**; **trilinear is
  24×7 rows on every sign regime**. So products are structurally stable;
  powers/univariates are not.
- Consequence for the fixed-structure engine (it caches one structure at
  construction and patches coefficients): a monomial variable whose **root box
  spans zero cannot be patched** — the structure differs before vs after a
  zero-crossing branch. But branching only *shrinks* boxes, so a **sign-definite
  root box stays sign-definite in every descendant** — those *are* patchable.
- This also means the *current* `x²` coverage is only sound because `_validate`'s
  boxes are all `lb ≥ 0`; extending validation to strictly-spanning boxes (as the
  task text asks) would break it. That instruction is wrong for powers.

**Re-scope (supersedes the task text above for the power/univariate families):**
  1. **Product families — bilinear (done), trilinear, multilinear:** structure is
     sign-independent; add the closed-form RLT/recursive-McCormick generators;
     `_validate` on all sign regimes (including spanning).
  2. **Power/univariate families — monomial p≥2 (generalize `x²`), univariate,
     fractional:** cover a term **only when its variable's root box is
     sign-definite** (`lb ≥ 0` or `ub ≤ 0`); a spanning root box makes that term
     unmappable → the model falls back (`ok=False`), which is sound. The
     `_build_structure` probe must be **sign-matched to each variable's real root
     regime** (not the uniform strictly-positive `[1,7+]`), so the cached
     convex/concave rows match the cold build; `_validate` uses sign-definite
     boxes that respect each variable's regime.
  3. The per-family 200-box property test stands, drawing boxes from the family's
     admissible sign regime.

> **STOP / ESCALATED (2026-07-02, §0.4 + §0.6) — Phase 1's bound-neutrality
> premise is falsified; needs a maintainer decision before any T1.2 code lands.**
>
> I implemented the re-scoped monomial coverage above (generalized `_square_rows`
> to `_monomial_rows` for any p≥2, sign-matched probe, sign-definite gating; the
> closed-form rows reproduce `build_milp_relaxation` to 1e-9 for p=2..5 on both
> regimes — the derivation is correct and parked in
> `docs/dev/data/t12-monomial-patch.diff`). It **validates and stays sound**
> (`ok=True` only when the patched LP matches the cold build row-for-row), but it
> **fails the §0.2.5 exact-node-count gate**, and the cause is a §0.3 safety
> mechanism, so per §0.3/§0.6 the gate loses and I stopped. Two measured facts:
>
> 1. **The incremental engine is NOT node-count-neutral vs the cold path.** On
>    `nvs17` (deterministic; both certify the same optimum −1100.4): cold path =
>    **205 nodes**, incremental path = **117 nodes** (`DISCOPT_INCREMENTAL_MC=0`
>    vs `1`, same time limit, both `gap_certified`). The incremental path solves
>    each node LP with the **Neumaier–Shcherbina safe dual bound + a warm-started
>    parent basis** (both §0.3 safety mechanisms) — a *valid but different* node
>    bound sequence → different fathoming/branching → a different (equally sound)
>    tree. T1.2 only changes *which* terms the engine covers, so it moves
>    instances from the cold tree onto the incremental tree and node_count drifts
>    (nvs17 205→117, nvs13 55→41, nvs05, …). Making node_count reproduce cold
>    exactly would require the raw (unsafe) LP objective and/or no warm start —
>    i.e. **weakening the §0.3 safe-bound mechanism**, which §0.3 forbids.
>    Node_count is also *timing-non-deterministic* within a single mode (nvs17:
>    111/59 at 30 s, 117/91 at 120 s), so exact equality is not even well-defined
>    for instances that don't certify in a few nodes.
>
> 2. **`cert-baseline.jsonl` is not a valid "certifying panel."** On *clean main*
>    (no T1.2 change) re-solving the baseline drifts anyway: `nvs05` is `feasible`
>    (time-limited → non-deterministic node_count, 493 vs baseline 473), and
>    `nvs22`'s baseline objective **7.40348 is non-reproducible** — clean main now
>    returns the *correct* optimum **6.05822**. §0.2.5 says "on the *certifying*
>    panel"; T0.5 froze the *whole* global50∪panel, including non-certifying and
>    non-deterministic rows, so exact-equality there is unsatisfiable by a no-op.
>
> **Decision needed from the maintainer (I did not improvise a fix):**
> - (a) **Re-define Phase 1 bound-neutrality** from "exact node_count vs baseline"
>   to the *differential* form already in the plan's toolbox: the incremental LP
>   must give the **same per-box bound as cold** (the T0.4 `assert_bound_sound`
>   harness / `_validate`'s row-set equality — already enforced) **and the same
>   certified objective**, accepting that the safe-bound tree differs in
>   node_count. This matches how the engine was actually designed ("never change
>   a *result*, only its speed") and keeps §0.3 intact. If chosen, replace the
>   `[gates.cert1] node_count_drift` criterion with a per-box bound-equality +
>   objective-equality check, and rebuild `cert-baseline.jsonl` over a
>   **deterministically-certifying subset** (drop `feasible`/time-limited rows).
> - (b) Keep exact-node-count neutrality and **re-scope Phase 1 away from the
>   incremental engine** (it inherently changes the tree), pursuing the per-node
>   cost win by other means. This contradicts §5's approach and is the larger
>   pivot.
>
> Until this is decided, T1.2–T1.6 are paused. No solver code was committed; the
> only artifacts are this note, the reproducible probes
> (`discopt_benchmarks/scripts/t11_entry_experiment.py`, `.../t12_probe.py`), and
> the parked patch.

**Resolution — maintainer approved direction (a) (2026-07-02).** Phase 1
bound-neutrality is redefined to the *differential* form: (1) `_validate` row-set
equality per box (incremental LP == cold LP — the direct "identical relaxation
math" proof, never softened); (2) the T0.4 differential-bound test at runtime
(every incremental bound ≤ the true box optimum, never crosses the oracle);
(3) certified objective unchanged **to tolerance** (a certified optimum
reproduces only to ~1e-10 across runs — bit-exact objective equality is not a
meaningful invariant, mirroring node_count); (4) node_count kept as a
**one-directional** performance guard (must not get materially worse), not an
equality gate. All §0.3 safety mechanisms (NS safe bound, warm start, `ok=False`
fallback) stay intact — the tree is allowed to differ because it differs *safely*.

- **Step 0 (done).** Rebuilt `cert-baseline.jsonl` as the **deterministic
  certifying subset**: `gen_cert_baseline.py` now solves each instance twice and
  keeps it only if both runs reach OPTIMAL with a bit-identical node_count and an
  objective agreeing to `_OBJ_TOL`/`_OBJ_RTOL`. Result: **44 rows** (dropped 11,
  all `time_limit`/`feasible` — incl. the non-reproducible nvs05; nvs22 now
  carries the *correct* optimum 6.05822). Full-panel cert0 gate still green
  (coverage 0.927, incorrect 0). This subset is the neutrality reference for
  T1.2–T1.6.
- **Step 1 (done).** Differential-neutrality checker wired
  (`discopt_benchmarks/utils/cert_neutrality.py` + `scripts/check_cert_neutrality.py`)
  and the **monomial p≥2** family landed behind the `ok=False` fallback. Baseline
  determinism hardened (K=3 runs + `≤0.6·budget` margin guard → 42-row subset).
  Result: **NEUTRAL** — sound on all 42, node one-directional on 41; nvs17
  perf-gated (T1.4, objective still verified). smoke 211 / adversarial 10 / 11
  property tests green.

**Family derivation results (2026-07-02, via 3 parallel derivation agents; verified
to 1e-9 vs `build_milp_relaxation`; reproducible probes in
`discopt_benchmarks/scripts/t12_{multilinear,univariate,frac}.py`).** All remaining
patchable families are now derived; integration is mechanical (detection + `_patch`
+ property test), each behind the `ok=False` fallback and perf-gated for T1.4.

| Family | Structure | Sign/regime stability | Verdict |
|---|---|---|---|
| bilinear | 4 McCormick rows | sign-independent | **done** |
| monomial `x^p`, p≥2 | 2 tangent + 1 secant | sign-*dependent* → gate on sign-definite root box | **done (Step 1)** |
| **trilinear / multilinear** (distinct factors) | recursive-bilinear + RLT hull (24×7, 92×15…); RLT cap 4 then loose chain | **fully sign-independent** (no gating) | derived, 780/780; **ready to integrate** |
| **univariate** exp/log/sqrt | 4 rows (endpoints + **midpoint** tangent) | stable 4 rows; →3 only at `lb==0` for log/sqrt → gate on root `lb>0` | derived, 1200/1200; **ready to integrate** |
| **fractional_power** `x^p`, non-integer/negative | 3 rows (endpoint tangents + secant) | row-count drops near 0 via `_envelope_slope_ok` (|slope|≤1e6) | derived, 287/287; **integrate with the slope guard + cold fallback** |
| `1.0/x` (syntactic reciprocal) | reciprocal-univariate (4 rows, +bilinear aux) | box-affine | **separate** generator from `x**-1` |
| trig (sin/cos/tan), asin/acos, entropy, abs-on-spanning | piecewise / binary-selector structure | box-dependent *structure* | **stay on cold fallback** (not fixed-structure) |

Integration prerequisites for the product family: gate on an **empty
`DiscretizationState`** (partitions route to SOS2/λ machinery — different
structure) and match the cold path's `rlt_level1` flag; repeated-factor products
(`x·x·y`) are already covered by monomial+bilinear.

**T1.3 — Widen the scope gate.**
Replace `_is_in_scope`'s all-integer+minimize test with: objective sense normalized
(negate for maximize), any variable mix; gate instead on "every lifted term type is
patch-covered" — which the constructor's validation already computes. Wire the
general spatial path (`solve_at_node`, mccormick_lp.py:524) to consult the
incremental engine first (extend the existing probe at :331), falling back per-model
when `ok=False`.
- **Test:** the certifying panel solves with *exactly unchanged* node_count and
  objective vs `cert-baseline.jsonl` (bound-neutral regime, §3); wall ↓ on the
  spatial class.

> **BLOCKED — re-scope required (2026-07-02, §0.4). The gate flip is sound but
> not viable as written; the differential-neutrality check caught why.** I
> widened the gate to `ok`-only for any model (`mccormick_lp.py` incremental
> probe). It is *sound* — the fast path is a valid `≤`-cold McCormick LP bound,
> and uncovered terms → `ok=False` → cold fallback — and it correctly activated
> on continuous/mixed/maximize bilinear models while falling back on univariate.
> But the neutrality check flagged a **catastrophic bound-weakening regression**:
> `dispatch` went **3 → 9843 nodes** (feasible, not optimal; objective still
> correct to 4.6e-13). Root cause: `solve_at_node` returns the fast-path result
> *before* the per-node **separation chain** (multilinear / edge-concave /
> univariate-square / convex / PSD / RLT cuts, `mccormick_lp.py:708-731`) — see
> the early `return _fast` at `:574`. `_try_incremental_node` assembles and solves
> the LP directly and never builds the `milp`/`varmap` object the `_separate_*`
> methods require, so it **cannot cheaply carry per-node separation** (that is the
> whole point of skipping the cold build). The pure-integer class (#355) tolerated
> this because the inherited root cut pool sufficed; the **spatial class relies on
> per-node separation**, so its bound collapses without it. Bilinear-dominated
> models that don't need separation (ex1221/ex8_1_1/ex1226) were node-identical —
> the failure is specifically separation-reliant models.
>
> Gate reverted to pure-integer/minimize. **Re-scope options (needs a design
> decision — architectural, higher risk than the gate flip):**
> (a) give the fast path per-node separation by exposing a separation-compatible
> view of the patched LP (partially defeats the no-cold-build speedup);
> (b) a **refreshed inherited pool** — cold-build + separate every K nodes, replay
> the captured cut pool on the incremental nodes in between (amortizes separation;
> the `out_cuts`/`inherited_cuts` plumbing already exists);
> (c) a per-model gate that engages the fast path only for separation-light models
> (bilinear-dominated, no edge-concave/PSD/multilinear structure). (b) looks most
> promising and keeps soundness (fewer/stale cuts only loosen the bound, and
> `_validate` still guards the base rows). Positive: **the direction-(a)
> differential-neutrality infra worked as designed** — it caught a
> would-have-shipped regression before commit.
>
> **Refinement (2026-07-02, bounded spike — the fix is cheaper than the full
> refreshed pool).** A viability spike (gate cold-path separation to every-K
> nodes, `DISCOPT_SEP_EVERY_K`) **disproved the "needs per-node separation"
> theory**: `dispatch` stays at **3 nodes for every K up to 100** (separation only
> at the root iteration). So the spatial class needs separation at the *root*, not
> at every node. Root cause of the fast-path explosion: `solver.py:4342` builds the
> inherited `_root_cut_pool` **only when `_psd_cuts` is on**; for a default spatial
> model it is `None`, so the fast path (which returns before `separate=True`) runs
> with **no cuts at all**. Targeted fix (cheaper, uses the existing
> `out_cuts`/`inherited_cuts` machinery, sound — root-box cuts are valid for all
> sub-boxes): **capture a root cut pool for the general spatial path (not just the
> PSD case)** so the fast path inherits the root separation, then widen the T1.3
> gate. Only escalate to the periodic/refreshed pool if some instance genuinely
> needs *in-tree* re-separation (none observed yet). Verify per instance with the
> differential-neutrality check.
>
> **DONE (2026-07-02) — the targeted fix works; T1.3 landed.** Three coordinated
> changes: (1) `solve_at_node` skips the fast path when capturing a pool
> (`out_cuts` set) so the pool actually separates (`mccormick_lp.py`); (2) a
> **general root-cut-pool** branch in `solver.py` builds the pool whenever the
> incremental engine is active (not just the PSD path), capturing the root
> separation chain once for the fast path to inherit; (3) the scope gate widened
> to gate on `ok` for any model. Result: `dispatch` **9843 → 3 nodes** (restored),
> and the fast engine now extends to the general spatial path **node-improving**:
> nvs13 55 → 19, nvs17 205 → 61. Verified: differential neutrality **NEUTRAL**
> across all 42 (objective correct, still optimal, node one-directional; nvs17
> perf-gated, objective correct); adversarial 10 / smoke 211 green; ruff clean.
> No refreshed/periodic re-separation needed. Sound by construction — root-box
> cuts are valid for every sub-box, and `_validate` still guards the base rows.

**T1.4 — Basis inheritance on the general path.**
Thread the parent basis through node solves (the Rust side already supports it:
`Basis` is Clone, `PreparedDual` clones a pristine factorization —
`lp/simplex/{basis.rs:34,dual.rs:152}`); store the parent's basis handle on the node
payload in the tree manager and pass it to the warm dual solve. Cold-start fallback
on soundness-guard rejection already exists.
- **Test:** LP iterations per node drop vs baseline; results bit-identical in
  objective/node_count on the certifying panel.

*Root cause found (2026-07-02) — this is the wall-time lever, and the T1.2
monomial finding elevated its priority.* The incremental engine reuses the
parent's **column-partition** basis, but `_patch` rewrites the McCormick row
**coefficients** and aux bounds every node — so the parent vertex is usually
**dual-infeasible** on the child LP. Rust's `PreparedDual::prepare`
(`crates/discopt-core/src/lp/simplex/dual.rs:225-247`) rejects a dual-infeasible
start (`return None`) → `solve_csc_core` (`dual.rs:95-115`) cold-refactorizes.
That is exactly the "cold refactorizations from rejected warm starts" cost, and
why T1.2's monomial coverage traded nvs17 205→~110 nodes for a *slower* per-node
wall (36s→45s). The Python guard (`mccormick_lp.py:443-447`) only checks row
count, so it can't help. **Minimal sound fix:** when `prepare` fails *only* the
dual-feasibility scan (factorization OK, sizes match), route the basis into
`run_dual` to restore dual feasibility in a few pivots instead of cold-solving;
fall back to cold only on singular factorization / iteration cap. Sound by
construction — `run_dual` converges to the true optimum or cold-falls-back
(`dual.rs:9-14`), and the §0.3 cold-start fallback is untouched. Seam:
`dual.rs:95-115` + `225-247`; no change to `Basis` (already `Clone`) or the
`in_basis`/`out_basis` plumbing. **Sequencing note:** T1.4 should likely land
*before* widening T1.2 coverage by default, since coverage without it regresses
per-node wall on the instances it newly moves onto the incremental path.

> **Tractability verdict (2026-07-02, read-only dual-simplex investigation) —
> ESCALATED: the quick fix is UNSOUND; the real fix is medium-effort simplex-core
> work.** The earlier proposal ("route the dual-infeasible warm basis into
> `run_dual` to repair it") is **wrong and unsafe**: `run_dual`
> (`crates/discopt-core/src/lp/simplex/dual.rs:304-451`) *hard-assumes* dual
> feasibility — it chooses the leaving variable only by primal infeasibility and
> declares Optimal when none exists, so a dual-infeasible start either aborts to
> cold (no gain) or **silently certifies a wrong optimum**. There is a regression
> test, `dual_infeasible_warm_start_falls_back_to_cold` (`dual.rs:1285-1326`),
> that exists precisely to forbid this. No dual phase-1 exists; the bound-flipping
> is the ratio test (maintains, doesn't establish, dual feasibility). The
> `prepare` dual-feasibility guard (`dual.rs:225-247`) that rejects the warm basis
> is **load-bearing (§0.3) and must not be weakened.**
>
> The *correct* repair is a **primal** warm re-solve: a branch tightens only the
> branched variable's box, so the child box ⊂ parent and the parent's primal
> point is usually still primal-feasible for the child — a few *primal* phase-2
> pivots re-optimize. But the primal engine (`primal.rs`) is **cold-start-only**:
> `run()` overwrites any incoming basis with its own crash/artificial basis
> (`primal.rs:406`); there is no warm-basis ingestion API (module doc:
> "Warm-start (dual simplex) comes later"). So T1.4 = **add `solve_lp_cols_warm`
> to `primal.rs`** (ingest `basic_vars`/`col_status`, factorize via the existing
> `refactorize`, skip phase-1 to phase-2 when the warm basis is primal-feasible,
> else fall into phase-1 = the genuine cold solve) and route `prepare`'s
> dual-infeasible rejection (`dual.rs:106-113`) there. Medium effort, new
> soundness-critical surface in the simplex core; the parent LU **cannot** be
> reused (child columns differ numerically), so the strong-branch factorization
> amortization does not transfer. **Decision needed** (see §0.8): invest in the
> warm-primal now, or defer T1.4 — Phase 1's *wall-time* win is gated on it, but
> T1.3 already delivered the engine on the spatial path with node-neutrality.
>
> **RESULT (2026-07-02, §0.4) — implemented, sound, but INERT; the premise does
> not hold. Reverted.** Built `solve_lp_cols_warm` + `run_warm` in `primal.rs`
> (phase-2-only from a nonsingular + primal-feasible inherited basis, else cold)
> and routed `dual.rs`'s prepare-`None` fallbacks to it. It is **sound** — `cargo
> test simplex` 42/42 and the forbidding `dual_infeasible_warm_start_falls_back_to_cold`
> stayed green. But a counter (`DISCOPT_T14_DBG`) showed **`run_warm` is never
> called** on an nvs17 spatial solve (0 accept / 0 reject) while the Python engine
> stored a warm basis 18×. So the warm basis never reaches the dual-infeasible
> fallback: either the dual `prepare` **actually succeeds** (the incremental node
> LP is dual-feasible — the coefficient patch did *not* break dual feasibility as
> hypothesized), or the Python `nrows` guard (`mccormick_lp.py:443-447`, now
> toggled by the T1.3 root-pool cuts changing the row count) routes the node to the
> *ordinary* cold path. Either way the node LP warm-start is **not the
> bottleneck** — nvs17 runs at ~2 nodes/s with the dual warm start already working.
> The dominant per-node cost is elsewhere: the fresh profile in §1 named **OBBT's
> inner loop (~23 LP solves/node on gear4) and per-call relaxation rebuilds**, not
> the node LP solve. **Re-scope: T1.4 (node-LP warm-start) is a non-lever; the real
> per-node lever is a re-profile → OBBT / per-call-rebuild reduction (T1.6 /
> perf-plan Stage 2), or fixing the `nrows` guard so a stored basis reaches the
> dual warm path.** Implementation parked in `docs/dev/data/t14-warm-primal-patch.diff`.

**T1.5 — Evaluator-cache routing (perf-plan Stage 1, validated).**
Route the ~18 direct `NLPEvaluator(model)` sites (list in performance-plan §3;
biggest: `primal_heuristics.py:1045`) through `_make_evaluator` (solver.py:414).
- **Test:** gear4 `python_time` ↓ ≥ 25% at node_count 5921 unchanged (numbers from
  the validated prototype).

**T1.6 — Per-node bookkeeping into Rust.**
`py-spy record` on two spatial instances post-T1.2/T1.5; move the top Python
per-node costs (node dict assembly, array marshaling for unchanged data) into the
Rust tree manager. Data-driven: only what the profile names.
- **Exit for the phase:** §5 exit gate, measured on `cert-baseline.jsonl`'s panel;
  update `[gates.cert1]`.

> **Re-profile RESULT (2026-07-02, §0.4) — T1.6's premise (Python per-node tax
> dominates) does NOT hold post-T1.3; the real lever is OBBT's inner LP loop.**
> cProfile of an nvs17 spatial solve (29 nodes, 16 s): the #1 cost by far is
> `discopt._rust.solve_lp_warm_csc_py` — **3.84 s across 284 calls (~10 LP
> solves/node)**. The Python per-node bookkeeping T1.6 targets is *minor* by
> comparison (`_decompose_product` 0.45 s, `_fbbt_eq_bounds` 0.22 s,
> term-classifier bits). So moving Python bookkeeping to Rust wins almost nothing.
> The ~10 LP solves/node are **OBBT's bound-tightening probes**: `obbt.py:1652`
> builds its relaxer with `build_incremental=False` (it "never calls
> `solve_at_node`"), so each probe cold-builds + solves its own LP and does *not*
> use the T1.3 fast path. This matches the §1 profile ("OBBT's inner loop, ~23 LP
> solves/node on gear4") and perf-plan §5 ("per-node OBBT/node-NLP still cold-build
> their own relaxations").
>
> **Conclusion — Phase 1 characterized.** The *structural* win is landed and sound
> (T1.3: incremental engine on the general spatial path, node-neutral; node LP
> already fast via the dual warm start). Both remaining *per-node-cost* items are
> **non-levers**: T1.4 (node LP warm-start already works) and T1.6 (Python tax is
> small). The remaining **wall-time** lever is a distinct, bound-neutral OBBT
> optimization: **warm-start OBBT's inner probe LPs** (each probe changes only the
> *objective* over a fixed box → the previous probe's basis is primal-feasible → a
> primal phase-2 warm re-solve — exactly what the parked
> `t14-warm-primal-patch.diff` implements — applies here, where it did *not* apply
> node-to-node). Alternatively route OBBT through the incremental engine. This is
> the perf-plan's Stage-3 / follow-on territory, not a T1.x per-node-engine item —
> flagged for a maintainer decision on whether to pursue within this plan.

### Phase 2 tasks — branch-and-reduce orchestration (specced 2026-07-02)

*Speccing note.* The maintainer directed a task-level Phase 2 spec (2026-07-02).
§0.1.2 still binds the *coding*: **T2.3–T2.5 are locked until T2.1's results
table is recorded in this section.** Two tasks are exempt from that lock and may
start immediately: **T2.0** (correctness-first work under
`correctness-issues.md` §0 — the loop would amplify open bugs in exactly the
passes it iterates) and **T2.2** (the Phase-1 perf follow-on measured and
flagged in T1.6's conclusion, relocated here because Phase 2 multiplies OBBT
call volume; a same-math change under the differential regime). This section
supersedes the §6 narrative where they differ.

**Verification regimes (per §3 / §0.2.6):**
- **T2.2** — differential form (the T1.2 direction-(a) resolution): per-probe
  bound equality vs the cold path to 1e-9 on a sampled A/B set;
  `scripts/check_cert_neutrality.py` reports **NEUTRAL** (objective to tol,
  still optimal, node one-directional); NS safe-bound clamping untouched.
- **T2.3–T2.5** — bound-changing: every stage behind a flag (default OFF), the
  T0.4 differential bound test (`utils/soundness.py::assert_bound_sound`) +
  feasible-point sampling (`assert_cut_valid` for cuts; box-containment sampling
  for reductions) per family, `incorrect_count = 0` on the full panel, the
  adversarial suite green, and **3 consecutive green nightlies before any
  default-on flip** (T2.6).

**Binding negative results (do not relitigate):** OBBT-on-aux / `cascade_aux`
is measured-dead (perf-plan §6: neutral-to-regressive, and nvs22 *fails to
certify* with it on) — it stays default-off and is not a loop stage. gear4 is
reduction-resistant (cutoff-OBBT + integer probing to a fixpoint stalls at a
2.46 M box) — it is **not** a Phase 2 gate probe; gates are panel-wide.
Branching is out of scope (§12). The `python_time`/`rust_time` result fields
are accounting artifacts (§1 correction) — measure with the T0.3 timers
(`reduce/…`, `separate/…` on `solver_stats`) and wall clock.

**Anchors (verified 2026-07-02 by three recon reports; line numbers current on
`cert-phase0`).**
Python — `_jax/obbt.py`: `run_obbt_on_relaxation` :712 (the workhorse; NS-safe
vertex clamp :909/:944, `_OBBT_NS_GUARD` :52, conditioning guard
`_OBBT_COND_LIMIT` :39→`require_ns` :826), `dbbt_on_relaxation` :962 (one LP →
reduced costs tighten all vars; **no-ops without a finite cutoff**),
`obbt_tighten_root` :1528 (root AND per-node entry; internal ≤3-round
reduce↔rebuild fixpoint :1691, envelope rebuild via
`build_milp_relaxation(bound_override=…)` :1699/:1741; relaxer built with
`build_incremental=False` :1652), candidate set = all columns in index order
:860-863 (**no scoring**). `solver.py`: root sequence = Rust FBBT :3786 →
nonlinear FBBT :3819 → root OBBT :3835-3924 (gate :3871-3884; budget
`min(clamp(0.1·TL, 2, 15), remaining)` :3890; **no incumbent cutoff** :3891);
root-box snapshot :3930-3931; root cut pool init :4086, PSD branch :4342-4404,
general (T1.3) branch :4405-4452, inherited at :5087/:5357; per-node FBBT
`_tighten_node_bounds_with_status` :4817; per-node OBBT class gate :4749-4753
(`_dependent_var_names` + n ≤ 100; constants :211-215; budget :4754-4755; call
:4865-4874; skipped on the root batch :4855); incumbent-improvement
`fbbt_with_cutoff` (Phase C3) :6340-6378; Phase C OBBT-on-improvement
:6301-6338; RC-fixing exists root-only on the MILP path :9974/:10019/:10904.
`_jax/mccormick_lp.py`: `solve_at_node` :549, separation chain :742-778,
`MccormickLPResult` :188 (**status/lower_bound/x only — no duals exposed to the
caller**), incremental probe :340/:409.
Rust — `fbbt_with_cutoff` `presolve/fbbt.rs:1098` (PyO3
`expr_bindings.rs:443-457`; returns per-block lb/ub arrays); binary probing
`presolve/probing.rs:41/:51` (**root-only** presolve pass; default pass list
`expr_bindings.rs:672-681`); ordered Python-driven pass list = `presolve()`
`expr_bindings.rs:617`; RC-fixing kernel `presolve/duality.rs:85`
(`ReducedCostInfo` :56-65; PyO3 via `presolve(reduced_cost_info=…)` :703) plus
the node-level MILP-driver implementation `bnb/milp_driver.rs:1249` (gap slack,
inward rounding — the pattern to mirror); `tree_manager.rs` `NodeResult` :28-37
has no tightened-bounds field (children inherit boxes the driver writes before
export); the warm-primal patch is parked in
`docs/dev/data/t14-warm-primal-patch.diff` (adds `solve_lp_cols_warm` to
`primal.rs`, hunks at :76/:476; rewires the `dual.rs:106-114/:145-152` cold
fallbacks); `solve_lp_warm_csc_py` (`lp_bindings.rs:411`) already returns
`(status, x, obj, iters, col_status, basic_vars, dual, ray)`.
*(Two Phase-1 anchor corrections found during recon: the per-node OBBT gate is
solver.py:4749, not :4765; `solve_at_node` is mccormick_lp.py:549 and the
separation chain :742-778, not :524/:708-731.)*

**What exists vs what is missing (the recon in one table):**

| Mechanism | Exists today | Missing (= the Phase 2 work) |
|---|---|---|
| reduce↔re-relax fixpoint | inside `obbt_tighten_root` (≤3 rounds) | solver-level loop with re-separation; incumbent cutoff at root; probing/Rust-FBBT stages (T2.3) |
| incumbent-aware FBBT | Rust + PyO3; fires only on incumbent *improvement* | at every spatial node, cheap (T2.4) |
| DBBT | `dbbt_on_relaxation` (extra LP + exact-dual oracle) | free variant from the node LP *just solved* (T2.4; needs duals exposed) |
| integer RC-fixing | Rust MILP driver per node; root-only MILP path in Python | spatial path per node (kernel exists, T2.4) |
| probing | root presolve pass | (per-node probing deliberately NOT planned — in-tree cost unproven) |
| OBBT variable selection | none — all vars, index order | width × dual-activity top-k scoring (T2.5) |
| OBBT LP engine | cold rebuild per probe (B1: 23 solves/node, scipy churn) | persistent per-sweep LP + warm-primal probes (T2.2) |
| node duals → Python | computed, consumed internally only | surface on `MccormickLPResult` (T2.4a) |
| gates/suites | cert0, cert1 | `[suites.global50]`, `[gates.cert2]`, 2 metrics (T2.6) |

---

**T2.0 — Reduction-layer correctness pre-flight (blocking; per
`correctness-issues.md` §0).**
The root loop re-runs presolve every iteration and multiplies OBBT/FBBT volume,
so open bugs in those passes go from latent to load-bearing. Fix in order, each
per its issue card (regression test that fails before / passes after; standing
gates green):
1. **C-16 (P0)** — presolve aggregate bound-resync fuses unrelated variables'
   bounds (`presolve/pass.rs:104-124`; DEFAULT path). T2.3 re-invokes
   `presolve()` in a loop; this lands first, unconditionally.
2. **C-15 (P2)** — `run_obbt` tightens to the raw LP vertex without the NS
   clamp (`obbt.py:667-701`); NS-clamp it, or delete it if dead (deletion
   preferred per the card). Standing rule for all of Phase 2: **every
   tightening uses the NS-safe bound, never the raw LP objective** — this is
   the C-15 failure class.
3. **C-20 (P2)** — `fbbt_fp.rs:148` zero-tolerance emptiness → false
   infeasible; fix before any loop configuration may enable
   `fbbt_fixed_point`.
4. **C-21 (P2)** — `IncrementalMcCormickLP._validate` probes only `lb ≥ 0`
   boxes; harden sign regimes before T2.2(c) routes OBBT through the
   incremental engine.

**T2.1 — Entry experiment: offline root-loop replay with per-stage attribution
(kill criterion; ~2–3 days).**
Script `discopt_benchmarks/scripts/t21_root_loop_replay.py`, house style per
t11/t12 (task-ID docstring; `JAX_PLATFORMS=cpu` + `JAX_ENABLE_X64=1` set before
importing discopt; standalone runnable).
- **Panel:** the 20 worst global50 instances by wall ratio vs BARON from
  `results/baron_vs_discopt_global50_20260618T033058.json`: tspn05, tanksize,
  casctanks, tls2, st_e36, clay0303hfsg, st_e38, st_test1, st_testgr3,
  st_miqp2, st_miqp5, m3, cvxnonsep_nsig30, st_miqp4, st_miqp1,
  cvxnonsep_psig40r, fac2, nvs05, cvxnonsep_psig30, flay03m. The six
  *uncertified* rows (tspn05, tanksize, casctanks, tls2, st_e36, nvs05 —
  `feasible`/`time_limit` at budget) are the substantive bound-limited tail;
  the rest are certified-but-slow. Caveats: several are not in the bundled
  61-file corpus — load via `dm.from_nl` with the MINLPLib snapshot
  (`~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`; resolution logic
  `runner.py:382`); BARON rows with `status="unknown"` at ~20 ms are not
  reference optima — the oracle is `docs/dev/data/cert-optima.json` /
  `minlplib.solu`.
- **Procedure per instance:**
  1. Harvest an incumbent honestly: a short discopt solve (5–10 s), record the
     incumbent (or none-found — see risk 3 below).
  2. Baseline root state: run today's root sequence once (Rust presolve → FBBT
     → structural `obbt_tighten_root`) and record root bound/gap (T0.1
     semantics).
  3. Replay the loop offline, deterministic stage order, each stage measured
     *marginally* per iteration: S1 `presolve()` ordered passes incl. probing →
     S2 `fbbt_with_cutoff(incumbent)` → S3 `obbt_tighten_root(incumbent_cutoff=…)`
     (DBBT runs inside when a cutoff is present) → S4 envelope re-derivation
     (`build_milp_relaxation(bound_override=…)`) + re-separation (capture the
     pool via `solve_at_node(..., separate=True, out_cuts=…)`) → root bound;
     iterate until the bound moves < tol or a budget is spent.
  4. Record per stage/iteration: root gap, wall, Σ log bound-widths (box
     volume), #bounds tightened.
  5. Projected tree effect: apply the final tightened box as variable bounds
     and re-solve at a fixed budget (60 s); record node_count/status/objective
     vs an untightened re-solve at the same budget.
  6. Oracle-cutoff variant (diagnostic only, clearly labeled): repeat with
     cutoff = known optimum + tol, bounding the best case reduction can ever
     deliver here.
- **Soundness instrumentation inline:** after every stage,
  `assert_bound_sound` with `known_optimum_oracle` (bound ≤ oracle + tol); the
  tightened-box re-solve must reproduce the reference objective to tol. Any
  violation is a P0 stop (§0.6) — it means an existing reduction pass is
  unsound.
- **Kill criteria:** *per-stage* — a stage whose marginal relative root-gap
  movement is < 5% on every panel instance is dropped from T2.3's loop (record
  it here). *Loop-level* — if the full loop achieves neither (a) median
  relative root-gap reduction ≥ 25% on the six uncertified instances nor (b) a
  projected ≥ 30% of the tree-opening certified instances closing within 10
  nodes, Phase 2's premise is falsified: stop, record here, and re-scope (the
  residual gap is then relaxation/cut strength — Phases 3–4 territory).
- **Deliverable:** the per-stage marginal table, the derived stage
  include/exclude list, and a calibrated loop budget appended to this section.
  T2.3–T2.5 unlock only once it is here.

**T2.1 — RESULTS / VERDICT (2026-07-03): NO-GO on the T2.3–T2.5 reduction loop.
Provisional; revisit when the per-node engine is faster (see caveat).**
Script: `discopt_benchmarks/scripts/t21_root_loop_replay.py` (offline replay:
incumbent harvest → cutoff-free baseline root → cutoff-aware reduce loop with
per-stage marginal attribution → projected tightened-box re-solve → oracle-cutoff
diagnostic; inline `assert_bound_sound` after every stage). **No P0 soundness
violation on any instance run.**

- **Criterion (a) — root-gap reduction on the six *uncertified* instances
  (COMPLETE, 6/6):**

  | instance | base gap | final gap | rel. reduction |
  |---|---|---|---|
  | tspn05    | 0.1227 | 0.1227 | 0.0% |
  | tanksize  | 1.0000 | 0.3075 | 69.3% |
  | casctanks | 10.064 | 9.817  | 2.5% |
  | tls2      | 0.8645 | 0.8050 | 6.9% |
  | st_e36    | 0.2378 | 0.2378 | 0.0% |
  | nvs05     | 0.8768 | 0.5915 | 32.5% |

  Sorted `[0, 0, 2.5, 6.9, 32.5, 69.3]` → **median 4.7% ≪ 25% threshold →
  (a) FAILS.** The reduction loop does **not** move the bound-limited tail — the
  exact instances it exists to help. (Only tanksize/nvs05 respond; the other four
  are ≤ 7%, i.e. the McCormick bound is far from the incumbent and reduction
  cannot close it — a relaxation-strength problem, not a reduction problem.)

- **Criterion (b) — tree-opening certified instances closing within 10 nodes
  (PARTIAL, 7 of 13 evaluable):** closed = st_e38 (100%), st_miqp2 (100%),
  st_miqp1 (15→1 node, 100%); not closed (0% gap move) = clay0303hfsg, st_test1,
  st_testgr3, m3; skipped = st_miqp5 (binary `.nl`, unsupported); **not run**
  (harvest+2×60 s solves too slow to finish in budget) = cvxnonsep_nsig30,
  st_miqp4, cvxnonsep_psig40r, fac2, cvxnonsep_psig30, flay03m. On the completed
  set that is 3 closed / 7 = 43% (or as low as 3/13 = 23% if all remaining are
  tree-opening non-closers) — **straddling the 30% bar; not conclusively decided.**

- **Verdict:** the *primary* criterion (a) fails decisively, and (b) — a secondary
  metric (already-certified instances closing faster) — is at best borderline. The
  evidence does **not** justify building the T2.3–T2.5 root-reduction loop. Per the
  loop-level kill criterion above, the residual gap re-scopes to **relaxation / cut
  strength (Phases 3–4)**, which is where the bound is actually lost.

- **Caveat (why "provisional" / revisit):** the experiment as specced — two 60 s
  tree solves per instance across the *20 slowest* instances — is too expensive to
  run to completion (only 13/20 finished before the budget was spent; st_e36 alone
  took 404 s for a 2-var problem). The NO-GO rests on (a)'s dominant, complete
  signal; (b) was left incomplete. **Revisit T2.1 when the per-node engine is fast
  enough to complete the full panel cheaply** (a tighter probe budget or a smaller
  panel would also make it a practical repeatable gate). T2.3–T2.5 remain **not
  built** in the meantime.

**T2.1-revisit — RESULTS / VERDICT (2026-07-06): GO on the T2.3–T2.5 reduction
loop (carried by the out-of-panel generality arm; the Class-P tail arm still
fails). Supersedes the 2026-07-03 PROVISIONAL NO-GO above.**
Task R1 of `docs/dev/uncertified-tail-plan-2026-07-06.md`. Re-ran the *existing*
`discopt_benchmarks/scripts/t21_root_loop_replay.py` on the post-F1–F7 `main`
engine (release POUNCE `_pounce.abi3.so` 4.73 MB, release discopt `_rust.so`
2.81 MB — pounce#182 lesson honored). **The revisit condition was met:** the full
20-instance panel now runs *to completion* (19/20; `st_miqp5` binary `.nl`
skipped, same as before) — the 2026-07-03 run finished only 13/20 (st_e36 alone
404 s). Loop-median wall is now **0.27 s** per instance. **No P0 soundness
violation on any of the 51 instances run** (panel 19 + tail nvs09/hda + 29 of the
30 out-of-panel; every stage's inline `assert_bound_sound` passed, every
tightened-box re-solve reproduced its `=opt=` oracle to tol — deltas ≤ 1.5e-4).

*Budget knob change (recorded per §0.6 — soundness checks untouched):* added
three arguments to the script — `--extra-oracle <json>` (loads `=opt=` oracles for
out-of-panel instances into the same `ORACLE` map the soundness check reads),
`--tree-budget-s` (the step-5 projected-tree re-solve budget, previously a
hardcoded `60.0`; the 2026-07-03 incompleteness was 2×60 s tree solves × 20
instances — ran the revisit at 15–20 s), and merged the extra-oracle at startup.
No stage, tolerance, or `assert_sound` logic was altered. Panel run:
`--harvest-s 8 --loop-budget-s 30 --obbt-stage-s 10 --tree-budget-s 20`.

- **Per-instance classification** (responsive ≥25 % rel. root-gap reduction,
  marginal 5–25 %, resistant <5 %). Panel 6 *uncertified* rows:

  | instance | base gap | final gap | % reduction | class |
  |---|---|---|---|---|
  | tspn05    | 0.1227 | 0.1227 | 0.0%  | resistant |
  | tanksize  | 1.0000 | 0.3056 | 69.4% | **responsive** |
  | casctanks | 4.5863 | 4.2201 | 8.0%  | marginal |
  | tls2      | 0.8645 | 0.8050 | 6.9%  | marginal |
  | st_e36    | 0.2378 | 0.2378 | 0.0%  | resistant |
  | nvs05     | 0.8768 | 0.5915 | 32.5% | **responsive** |

  Median = **7.4 %** (2026-07-03: 4.7 %) ≪ 25 % → **primary criterion (a) still
  FAILS.** Criterion (b) is now *complete* (13/13 certified rows, vs 7/13 in
  2026-07-03): 1 of 13 closes ≤10 nodes by reduction = **7.7 %** ≪ 30 %. On the
  panel's own metrics the loop is **still a NO-GO** — the four hardest uncertified
  rows (tspn05/st_e36 0 %, casctanks/tls2 ≤8 %) do not respond; only
  tanksize/nvs05 do, unchanged from 2026-07-03. (`cvxnonsep_psig40r`, `fac2`
  report no root bound in the cold harness build — `no-bound`, excluded from the
  median as before.)

- **Class-P tail decision set** (hda/nvs05/nvs09/st_e36/tanksize/tls2):
  responsive = **2** (nvs05 32.5 %, tanksize 69.4 %); tls2 marginal (6.9 %);
  st_e36 & hda resistant (0.0 %; hda *does* get a cold root bound of gap 9.84 in
  the harness — its solver-path "no dual bound" is the 23 omitted rows, an R4
  concern); **nvs09 = no-bound** (the cold cutoff-free relaxer returns no LP
  bound; it still solves 95 nodes — a harness relaxation-coverage gap, not
  reduction data). **2 < 3 → the Class-P-tail arm of the R1 rule FAILS.**

- **Out-of-panel generality sample** (§0.2 check): 30 nonlinear MINLPLib
  instances with `=opt=` oracles, drawn size-spread (1–55 vars) from
  `problems_small.txt` topped up from `problems_short.txt` (the curated
  `problems_small.txt` holds only 14 usable names — too few for a 30-sample);
  **29 ran** (autocorr_bern25-19 hit a parser `RecursionError` — labeled skipped,
  not dropped). **Responsive = 9/29 = 31.0 %** (prob03, nvs12 72 %, ex1223a,
  ex5_2_2_case1, pooling_haverly1tp, gkocis 60 %, ramsey, pooling_adhya1stp,
  wastewater04m2); marginal 3, resistant 14, no-bound 3. Discounting
  wastewater04m2 (its incumbent was `none_found→oracle`, i.e. an oracle-cutoff
  best-case, not an honest harvest) → **8/29 = 27.6 %**, still **≥ 20 %**. All
  nine reproduce their oracle to tol (no false close). **→ the out-of-panel arm
  of the R1 rule PASSES.**

- **Per-stage marginal include/exclude** (max marginal rel-gap move over all 48
  instances that produced a bound; a stage <5 % on every instance is dropped):
  - **S1 presolve — DROP** (0.0 % everywhere; the Rust root presolve already ran
    at baseline, so its marginal loop contribution is nil).
  - **S2 `fbbt_with_cutoff` — INCLUDE** (up to 100 %; moves ≥5 % on 12 instances
    incl. tanksize, tls2, nvs05).
  - **S3 `obbt_tighten_root(cutoff)` — INCLUDE** (up to 100 %; the primary lever;
    moves ≥5 % on 13 instances incl. all four closers).
  - **S4 envelope-rebuild + re-separation — DROP** (0.0 % everywhere; consistent
    with the Phase-3 1c/zerohalf NO-GOs — re-separation adds no root bound here).
  → **R2's loop = {S2 cutoff-FBBT, S3 cutoff-OBBT}; drop S1 and S4.**

- **Calibrated loop budget** (median stage wall as a share of the ~0.27 s loop):
  S3 OBBT 58.6 %, S1 20.3 %, S2 10.7 %, S4 10.4 %. With S1/S4 dropped, fund only
  the two INCLUDE stages: **S2 ≈ 15 %, S3 ≈ 85 % of the loop budget; loop budget
  ≈ 10 % of the node/time limit** (the loop converges in ≤2 iters and is cheap
  now — the 15 % starting point in the T2.3 spec is comfortably affordable and
  can be trimmed to ~10 %). S3 keeps its own inner OBBT deadline (`--obbt-stage-s`
  analogue).

- **VERDICT — R2 GO.** The R1 decision rule is an OR: build R2 if the responsive
  class has **≥3 Class-P-tail instances (got 2 → fails)** OR **≥20 % of the
  out-of-panel sample (got 31 %, ≥20 %+ even after the honest discount to 27.6 %
  → passes)**. The OR is satisfied → **GO**. Honest scope caveat (not a hedge —
  the record must state it): the loop does **not** crack the hard uncertified
  tail (panel median still 7.4 %; tspn05/st_e36/casctanks/tls2 unmoved) — its
  value is *broad small-MINLP root closure*, not the Class-P six. R2 must
  therefore be built to the §0.2 generality gate (panel-wide, no instance-keyed
  code), not tuned to tanksize/nvs05; its acceptance is the out-of-panel win rate
  reproduced under the flag, and R3b/R4 remain the levers for st_e36-class
  pinning (which reduction provably does not touch). T2.3–T2.5 are **UNLOCKED**;
  build per the §14 spec with the stage list {S2,S3} and budget above.

**T2.2 — Make OBBT affordable: persistent probe LP + warm-started probes (the
B1 fix; may run parallel to T2.1).**
Evidence: bottleneck-profile B1 (gear4: 92,061 probe LP solves, 23/node, 90%
from `obbt.py:712`; scipy CSR realloc per solve ≈ 33% of wall;
`build_milp_relaxation` called 2.4×/node redundantly from `obbt.py:1528`;
ex1252/ex1252a spend 32–49 s in root OBBT over ≤ 3 nodes); T1.6 re-profile
(OBBT probes cold-built, ~10 LP solves/node on nvs17). Three sub-items, in
order:
- **(a) Per-sweep, not per-probe, LP builds.** Within one OBBT sweep the box is
  fixed — assemble the std-form CSC once per sweep; per probe change only the
  objective vector (min/max x_j). Then eliminate the redundant
  `build_milp_relaxation` calls beyond the one genuine per-sweep rebuild.
- **(b) Warm-start consecutive probes.** Objective-only changes over a fixed
  box ⇒ the previous probe's optimal basis stays primal-feasible ⇒ apply the
  parked `t14-warm-primal-patch.diff` (`solve_lp_cols_warm`: primal phase-2
  from an ingested basis, cold two-phase fallback otherwise) and pass probe k's
  `(col_status, basic_vars)` into probe k+1 via `solve_lp_warm_csc_py`. The
  patch is measured sound (42/42 simplex tests incl. the forbidding
  dual-infeasible-fallback test) and was inert on the node-to-node path — the
  probe loop is exactly where T1.6's conclusion said it applies.
- **(c) Optionally** route OBBT through the incremental engine
  (`build_incremental=True` at `obbt.py:1652`, engaged only when `ok=True`) —
  only if the post-(a)/(b) profile still names the rebuild; requires C-21
  (T2.0.4) first.
Soundness constraints: the NS vertex clamp and conditioning guard
(`require_ns`) apply unchanged to warm-solved probes; `ok=False`/cold fallbacks
never weakened; any warm solve returning non-Optimal status → cold re-solve of
that probe.
- **Test (differential regime):** A/B harness — on ≥ 200 sampled probes across
  ≥ 5 panel instances, warm bound == cold bound to 1e-9 and the *applied
  tightenings* are identical; `check_cert_neutrality.py` NEUTRAL; smoke +
  adversarial green.
- **Measured targets (falsifiable):** gear4 OBBT umbrella 23.2 s → ≤ 8 s;
  ex1252a root-OBBT wall ≥ 2× down; if after (a)+(b) OBBT is still the top
  consumer, record here and re-profile before attempting (c).

**T2.2 — DONE (2026-07-03, PR #406).**
- **(a)** `_PersistentProbeLP` in `obbt.py`: `run_obbt_on_relaxation` assembles the
  std-form CSC `[A_ub | I_m]` once per sweep from the already-equilibrated/cutoff
  system; each probe changes only the objective (and the box as tightenings
  accumulate). The one genuine per-sweep `build_milp_relaxation` rebuild is left
  intact (bounds move each round — not redundant).
- **(b)** The parked `t14-warm-primal-patch.diff` applied cleanly (`solve_lp_cols_warm`
  in `primal.rs`; `dual.rs` warm-reject fallbacks rewired to primal phase-2). Probe
  k's `(col_status, basic_vars)` threads into probe k+1. NS clamp / `require_ns` /
  `ok=False` cold fallbacks unchanged; non-Optimal warm solve → cold re-solve.
- **(c) SKIPPED:** its precondition C-21 was open at authoring time, and the
  post-(a)/(b) re-profile shows the LP loop is no longer the sole top consumer.
- **Differential regime: NEUTRAL.** A/B harness `scripts/t22_obbt_ab.py`: 534 warm
  probe solves across 10 models — warm bound == cold bound to ≤4.3e-12, applied
  tightenings identical on every config; `check_cert_neutrality.py` NEUTRAL (42
  rows, max |Δobj| 8.9e-16, node_count unchanged). Regression test
  `python/tests/test_obbt_warm_probes.py` (fails on injected divergence).
- **Measured (T0.3 timers + wall), warm vs forced-cold, identical tightenings:**
  ex1252a 1.54× (301→196 ms), ex1252 1.89×, st_e38 1.92×, gear4 1.24×. ex1252a
  fell short of the ≥2× target → re-profiled per the spec: the LP side (B1) is down
  as intended; the residual root-OBBT time is ~50/50 LP probes vs the genuine JAX
  envelope rebuild (`build_milp_relaxation` autodiff) — Phase 4/5 territory, not the
  OBBT LP loop. gear4 is reduction-resistant (known negative result), not a gate probe.
- **Gates:** `cargo test -p discopt-core` 376/0 (simplex 42/42), `pytest -m smoke`
  211/1-skip, adversarial 10, `test_obbt.py` 48, `test_obbt_warm_probes.py` 14,
  ruff clean; `incorrect_count` not regressed.

**T2.3 — Root fixpoint loop (LOCKED until T2.1's table is recorded above).**
Build: new module `python/discopt/_jax/root_reduce.py` —
`run_root_fixpoint(model, lb, ub, *, incumbent_cutoff, deadline, tol, stages)
→ RootReduceResult` — deterministic stage order taken from T2.1's include list;
every stage deadline-aware and tighten-only (intersection, the solver.py:3914
pattern). Two integration points in `solver.py` (per the §6 correction note):
1. the existing structural (no-cutoff) root OBBT at presolve time (:3835) stays
   as-is;
2. the cutoff-aware fixpoint runs at the **end of iteration 0** — after the
   root heuristics have produced an incumbent, where the root bound is
   snapshotted (T0.1) and the general root cut pool is captured (:4405) — and
   refreshes `_root_cut_pool` and the incremental engine's base structure from
   the final tightened box, so every in-tree node inherits the tightened root.
Budget: a fraction of the time limit calibrated by T2.1 (starting point 15%,
hard per-stage deadlines); loop exits on bound move < tol or budget spent.
Flags: `root_fixpoint` config option + `DISCOPT_ROOT_FIXPOINT` env, default OFF
until T2.6. Constraints: `cascade_aux` stays off; the pass order is fixed (no
adaptive reordering — determinism); no stage may loosen a bound; cutoff-using
stages no-op soundly when no incumbent exists.
- **Test:** unit — a synthetic bilinear model where the round-2 envelope on the
  round-1-tightened box provably improves the root bound (regression: fails
  with the loop off); soundness — per-stage `assert_bound_sound` vs the
  cert-optima oracle on the replay panel; every captured cut through
  `assert_cut_valid` on sampled feasible points; full panel `incorrect = 0`;
  nightly A/B with the flag on.
- **Exit signal (informational until T2.6):** `root_gap_ratio_vs_baron` on the
  evaluable global50 rows trends toward ≤ 1.3.

**T2.4 — Per-node cheap reduction: one `reduce_node()` call (LOCKED until
T2.1's table is recorded above).**
- **(a) Expose node-LP marginals (prerequisite, bound-neutral):** extend
  `MccormickLPResult` (mccormick_lp.py:188) with `dual`, `col_status`, and
  `safe_bound` — all already computed internally / returned by
  `solve_lp_warm_csc_py`'s 8-tuple; plumbing only, additive fields, no math
  change (assert exact neutrality).
- **(b) `reduce_node(node_lb, node_ub, lp_result, cutoff)`** on the spatial
  path, called after each node LP solve, unifying three cheap, sound moves:
  (i) Rust `fbbt_with_cutoff` on the node box (per-block mapping as in
  solver.py:6355-6370 — today it fires only on incumbent improvement);
  (ii) DBBT from the just-solved node LP's reduced costs (d = c − Aᵀy from the
  returned `dual`; **z = `safe_bound`, never the raw LP objective** — the C-15
  rule; reuse `dbbt_on_relaxation`'s rc_tol/eps guards) — zero extra LP solves;
  (iii) integer reduced-cost fixing via the `duality.rs:85` kernel semantics
  (inward rounding, positive gap slack — mirror `milp_driver.rs:1249`).
- **(c) Feed the tightened box forward:** into the incremental engine's patch
  (`inc.assemble(lb, ub)`) and the child boxes at export (follow the MILP
  driver's `out.tightened` pattern; `NodeResult` needs no schema change if
  Python applies tightenings before export).
Flag: `node_reduce` config + env, default OFF until T2.6.
- **Test (bound-changing regime):** per-family differential test — on fixed
  node boxes captured from panel runs, the reduced box never excludes a
  sampled feasible point better than the cutoff, and the node bound stays
  ≤ oracle; property test — 200 random boxes/cutoffs, better-than-cutoff
  sampled points always retained; A/B on global50 — node_count expected ↓,
  wall ratio vs baseline ≤ 1.05 on non-target classes (the no-offtarget
  guard); adversarial + smoke; `incorrect = 0`.

**T2.3/T2.4 — R2 build-results (2026-07-06, this PR; flagged default-OFF).**
Built exactly to the R1 GO spec above: `_jax/root_reduce.py::run_root_fixpoint`
(S2 cutoff-FBBT → S3 cutoff-OBBT, tighten-only intersection, ≤2 rounds, ≈10% of
the limit budget, S3 keeps its own inner OBBT deadline) integrated at the **end of
iteration 0** in `solver.py` (after the root heuristics, where the root bound is
snapshotted); `_jax/node_reduce.py::reduce_node` (T2.4b) on the spatial node-LP
path unifying (i) cutoff-FBBT, (ii) **free DBBT from the just-solved node LP's
reduced costs `d = c − Aᵀy` with `z = safe_bound` — the C-15 rule, never the raw
LP objective** — zero extra LP solves, (iii) integer RC-fixing (inward rounding,
mirroring `duality.rs:85` / `milp_driver.rs:1249`); the tightened node box is fed
to the children via the new `PyTreeManager.set_node_bounds` PyO3 binding (T2.4c,
before `process_evaluated`). T2.4a exposes the node-LP marginals additively on
`MccormickLPResult` (`dual`/`col_status`/`safe_bound`/`reduced_costs`), requested
only when `want_marginals=True`. Flags `root_fixpoint`/`node_reduce`
(+ `DISCOPT_ROOT_FIXPOINT`/`DISCOPT_NODE_REDUCE` env), **default OFF**;
`cascade_aux` stays off (measured-dead).

- **Flag-OFF cert-baseline: byte-identical / NEUTRAL.** `check_cert_neutrality.py`
  over the 41-row certifying panel: every row `node_count X→X` **exactly
  unchanged**, `|Δobj| = 0.00e+00`, still optimal → NEUTRAL, exit 0. The default
  path never requests `want_marginals` (verified: 0 marginal-requesting solves
  flags-OFF), so T2.4a is bit-identical and both loops are inert with the flags
  off. This is the hard §0.2.5 gate — met with zero slack.

- **Node-count A/B (both flags ON vs OFF), the reduce-responsive class:**

  | instance | nodes OFF | nodes ON | wall ratio | obj (ON) | oracle | note |
  |---|---:|---:|---:|---:|---:|---|
  | **ex1224** | **53** | **5** | **0.76** | −0.943470 | −0.943470 | 10.6× fewer nodes (attribution: root-only 53→5, node-only 53→13, both 5); bound −0.9434705 (tighter than OFF −0.9435073), does NOT cross oracle; both `optimal` |
  | **wastewater04m2** | **479** | **351** | 1.00 | (no dual bound) | — | 1.4× fewer nodes at a 30 s budget (root arm alone: 479→159) |
  | ex1223 | 9 | 9 | 0.37 | — | — | same nodes, 2.7× faster wall |

  ex1224 is the branch-and-reduce closure the task targets (nvs24-class node
  collapse): the cutoff-OBBT root fixpoint + per-node DBBT contract the tree from
  53 to 5 nodes with a *tighter* certified root bound. wastewater04m2 responds on
  the root arm (479→159). No global50 instance moved a node **into** a worse count.

- **No-offtarget guard (≤1.05 wall on non-target classes): PASS.** Across the
  20-instance A/B (gkocis, nvs24, ex5_2_2_case1, pooling_haverly1tp,
  pooling_adhya1stp, ex1252, st_e38, ex1263, nvs14, ex1224, ex1225, ex1264,
  ex1266, nvs10, nvs11, nvs13, prob02, util, wastewater04m2, ex1223) the flags are
  **inert on the non-responsive class** (node_count unchanged) with wall ratios in
  [0.37, 1.03] on every non-trivial (>0.5 s) instance — no instance regresses
  wall > 5%. (Sub-second instances show ±2× ratio noise on a <0.5 s solve, e.g.
  ex5_2_2_case1 0.2→0.4 s, nvs10 0.1→0.1 s — not a wall regression.) The
  **key fix**: the root cut-pool re-capture on the tightened box costs an
  irreducible separating solve (measured +3.7 s on pooling_adhya1stp, NOT bounded
  by the LP `time_limit`), so it is **opt-in only** (`DISCOPT_ROOT_FIXPOINT_REPOOL`);
  the default flagged path keeps the still-valid wider-box pool (a cut valid over a
  box is valid over every sub-box → sound), which removed the +38% regression
  (pooling_adhya1stp 8.9 s OFF → 8.7 s both). A no-offtarget *gap gate*
  (`_ROOT_FIXPOINT_MIN_GAP`) skips the fixpoint when the root is already tight.

- **Differential + feasible-point soundness: GREEN.** 0 oracle crossings across
  the whole A/B panel (`bound ≤ oracle + tol` on every ON row; ex1224 bound
  −0.9434705 ≤ opt −0.94347). Property test `test_reduce_node_retains_better_than
  _cutoff_points` (200 random boxes/cutoffs): DBBT/RC-fixing is tighten-only and a
  better-than-cutoff point is never cut. `run_root_fixpoint` unit tests: tighten-only
  (subset box), no-cutoff structural no-op sound, and the **round-2 improvement**
  test (a second fixpoint round on the round-1-tightened envelope tightens strictly
  more — the loop, not a one-shot pass, is load-bearing).

- **Gates.** `pytest -m smoke` python/tests **617 passed / 14 skipped / 0 fail**;
  adversarial `test_adversarial_recent_fixes.py` **10 passed**;
  `cargo test -p discopt-core` **424+4+1 passed / 0 fail** (the `set_node_bounds`
  PyO3 binding); `ruff check` + `ruff format --check` clean; `mypy` clean on the
  new modules; `test_r2_branch_and_reduce.py` **7 passed**.

- **Honest scope (per R1's caveat, not a hedge).** The win is *broad small-MINLP
  root closure* (ex1224 the exemplar), NOT the hard uncertified tail: nvs05 in the
  full solver routes to the alphaBB path (`_mc_lp_relaxer=None`, so the flags are
  inert there by construction), and nvs09/nvs24 do not close (Class-P tail —
  R3b/R4 territory, which reduction provably does not touch). The default flip
  stays for T2.6 (nightly-green gate); this PR ships the flags default-OFF.

**T2.5 — Uniform OBBT escalation policy (locked on T2.4).**
Replace the per-node OBBT model-class gate (solver.py:4749-4753:
`_dependent_var_names` present + n ≤ 100) with one policy for all spatial
models: rank variables by `bound-width × |reduced cost|` (duals from T2.4a,
widths from the node box); run OBBT on the top-k, only at nodes where the
relative gap exceeds θ and depth ≤ d, inside the existing
`_pn_obbt_budget_total` machinery (:4754-4755). Feed the same scoring into root
OBBT's `candidate_idxs` (obbt.py:860-863 — currently all variables in index
order). Defaults (k, θ, d) are calibrated from T2.1/T2.4 A/B data — record the
calibration table here; no per-class forks, no named-instance tuning.
- **Test:** the policy only *selects where* already-verified reductions run, so
  the primary risk is performance: A/B on global50 with no-offtarget wall
  regression ≤ 1.05; the previously-gated class (`_dependent_var_names`
  instances) must not regress in node_count when the class gate is removed;
  `incorrect = 0`.

**T2.6 — Gate wiring, nightlies, default-on (last).**
1. `benchmarks.toml`: add `[suites.global50]`
   (`instance_list = "config/baron_global50.txt"`) — every cert gate references
   this suite but no suite table exists (recon flag); add `[gates.cert2]` per
   §11 with the corrections recorded there (existing `geomean_ratio_vs_baron`
   metric name; new `closed_within_10_nodes_fraction` metric fn + dispatch at
   metrics.py:706 for the "≥ 30% close within 10 nodes" criterion).
2. Reference data: `root_gap_ratio_vs_baron` must count only BARON rows with a
   usable root bound (many rows in the 2026-06-18 comparison are
   `status="unknown"` at ~20 ms); re-run the BARON comparison if coverage is
   too thin to evaluate the gate honestly.
3. Flag flips: with T2.3–T2.5 individually green, enable jointly on nightlies;
   **3 consecutive green nightlies** (full panel, `incorrect = 0`, adversarial
   suite, differential checks) before any default flips.
4. After default-on: rebuild `cert-baseline.jsonl` via `gen_cert_baseline.py`
   (node counts legitimately change; the new baseline becomes the Phase 3+
   neutrality reference), update §0.8 and this section.
- **Phase 2 exit gate (= §6):** `root_gap_ratio_vs_baron ≤ 1.3` evaluable and
  green on global50; ≥ 30% of currently-tree-opening instances close within 10
  nodes; `geomean_ratio_vs_baron ≤ 2.5`; `incorrect = 0` throughout.

**Sequencing.**

```
T2.0 (correctness pre-flight)
  ├─► T2.1 (entry experiment) ──► T2.3 (root loop) ──► T2.4 (reduce_node) ──► T2.5 (escalation) ──► T2.6 (gates/default-on)
  └─► T2.2 (OBBT cost; parallel with T2.1) ────────────┘   (T2.3/T2.5 assume affordable OBBT)
```

Estimated effort: T2.0 ~2–4 d; T2.1 ~2–3 d; T2.2 ~1 EW; T2.3 ~1–2 EW;
T2.4 ~1–2 EW; T2.5 ~3–5 d; T2.6 ~2–3 d — consistent with §6's 4–6 EW.

**Phase-2-specific risks.**
1. **False certificate** (the catastrophic mode; nvs22 #277 / st_ph10 #306):
   every tightening goes through NS-safe bounds + existing guards;
   feasible-point sampling per family; 3 green nightlies before default-on.
2. **Loop overhead that doesn't pay for itself:** budgets calibrated by T2.1;
   the ≤ 1.05 no-offtarget-regression criterion is binding; every stage is
   individually killable by its < 5% criterion.
3. **No incumbent:** the cutoff-dependent stages (DBBT, cutoff-FBBT,
   RC-fixing) must no-op soundly without one (they already do) — the loop
   degrades to its structural subset; never fabricate a cutoff.
4. **Determinism:** time-based budgets make node counts non-deterministic at
   the margin — all neutrality/regression checks use the differential form
   (the T1.2 resolution), never exact node equality.

### Ground rules for the implementing agent

1. One task per PR where possible; every PR runs `pytest -m smoke`,
   `pytest -m slow python/tests/test_adversarial_recent_fixes.py`, and the
   certifying-panel bound-neutrality check before merge.
2. Never weaken a validation/fallback path to make a gate pass — `ok=False` fallback
   is the safety mechanism, not a performance bug.
3. If a measurement contradicts this plan, the measurement wins: record it in this
   file (the way performance-plan.md §6 records its falsifications) and re-scope
   before coding on.
4. Stale doc warning: CLAUDE.md says "HiGHS LP wrapper" — the per-node LP default is
   the in-house Rust simplex (`MccormickLPRelaxer(backend="simplex")`); highspy
   appears only in optional OA/GDP modules. Do not plan against HiGHS.
