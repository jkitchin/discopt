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
| T0.3 reduction/separation timers | not started | — | |
| T0.4 soundness harness | not started | — | |
| T0.5 baseline + cert0 gate | not started | — | |
| T1.1 entry experiment (kill criterion) | not started | — | |
| T1.2 patch-table term coverage | not started | — | blocked by T1.1 |
| T1.3 scope-gate widening | not started | — | blocked by T1.2 |
| T1.4 basis inheritance | not started | — | |
| T1.5 evaluator-cache routing | not started | — | |
| T1.6 bookkeeping → Rust | not started | — | blocked by T1.5 profile |
| Phase 2 entry experiment | **locked** (§0.1.2) | — | unlocks on Phase 1 done |
| Phase 3 entry experiment (0b) | **locked** (§0.1.2) | — | unlocks on Phase 0 done |
| Phase 4 T-CSE/V-segments | **locked** (§0.1.2) | — | may parallel Phase 1 once specced |
| Phase 5 | **locked** (§0.1.2) | — | requires post-Phase-1 re-profile |

---

## 1. The measured diagnosis

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

| Capability | SCIP/BARON | discopt today | Gap type |
|---|---|---|---|
| Envelope library (factorable core) | reference | at parity (relaxation-catalog §8) | none |
| Node relaxation reuse / warm start | always (bound patch + dual simplex) | exists, gated to pure-int minimize | **wiring** |
| Per-node language cost | native | 58% JAX + 42% Python per node | **architecture** |
| Root reduce↔relax↔separate fixpoint | core loop (branch-and-reduce) | single-shot passes, budget-gated | **orchestration** |
| Per-node cheap reduction (FBBT+cutoff, DBBT/marginals) | every node | components exist, mostly root-only | **wiring** |
| Aggregation / c-MIR cuts | workhorse (97–169× nodes) | absent (current cuts net-negative) | **missing + quality** |
| Cut pool w/ aging on default path | yes | opt-in path only | wiring |
| CSE / defined-variable sharing | preserved & exploited | discarded | **missing** |
| Quadratic/structure recognition in core | yes | Python-side convexity only | missing |
| Root-gap instrumentation | n/a (internal) | schema exists, never populated | missing |
| Parallel tree search | partial | no | out of scope here |

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

## 14. Handoff appendix — executable task breakdown (Phases 0–1)

Phases 0 and 1 need no further discovery and are specified here to task granularity.
Phases 2–5 are *deliberately not* specced to this level: each begins with an entry
experiment (§6–§9) whose result determines the work items; the implementing agent
runs the experiment first and writes the task list from its output, in this file,
before coding. **Read §3 (verification regimes) before starting any task.**

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

**T0.5 — Baseline.** Run global50 + perf panel with T0.1–T0.3 on; commit JSONL next
to `docs/dev/data/perf-baseline.jsonl` as `cert-baseline.jsonl`. Add `[gates.cert0]`
to `discopt_benchmarks/config/benchmarks.toml` (§11) and a
`root_gap_populated_fraction` metric function in `metrics.py` (dispatch at :685).

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

**T1.4 — Basis inheritance on the general path.**
Thread the parent basis through node solves (the Rust side already supports it:
`Basis` is Clone, `PreparedDual` clones a pristine factorization —
`lp/simplex/{basis.rs:34,dual.rs:152}`); store the parent's basis handle on the node
payload in the tree manager and pass it to the warm dual solve. Cold-start fallback
on soundness-guard rejection already exists.
- **Test:** LP iterations per node drop vs baseline; results bit-identical in
  objective/node_count on the certifying panel.

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
