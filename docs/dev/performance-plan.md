# discopt performance plan — measured, staged, correctness-preserving

> Status: proposed, **grounded in a measurement pass (2026-06-24)** that
> overturned the first draft's cost model. The raw numbers are in the Appendix;
> read them before trusting any stage. Unifies the scattered performance issues
> (#309, #287, #282, #280, #267, #196, #208, #187). The narrow
> `scip-gap-closing-plan.md` (integer-product cuts, nvs17/19/24) studied a
> *different family* whose cost profile (LP-bound) does **not** match the spatial
> panel measured here — see §1.1.

## 0. The one non-negotiable: correctness is a gate, not a goal

Every change ships **only if the correctness panel stays green** — speed is the
objective, correctness is a hard constraint with zero slack:

1. `pytest -m smoke` (209) — 0 failures.
2. `pytest -m slow python/tests/test_adversarial_recent_fixes.py` (PR #315) — the
   10 adversarial problems on the week's soundness fixes: no false-feasible /
   -infeasible / -unbounded / -optimal, sound certificate, no crash.
3. **`incorrect_count == 0`** on the perf panel (`benchmarks/metrics.py` already
   encodes the invariants: an incumbent never beats the oracle; a `gap=0`
   "optimal" sits at the optimum).
4. The **certificate invariant** per instance: `bound ≤ incumbent` (min) /
   `bound ≥ incumbent` (max); a valid dual bound never crosses the oracle.

A perf change that improves wall time but trips any of (1)–(4) is a **regression**.
This discipline is what let the week of 2026-06-18..24 land speed/robustness work
without re-introducing the false-optimal / false-feasible bugs it fixed.

## 1. The measured cost model (this replaces the first draft's guesses)

I profiled four panel instances (gear4, ex1252, kall_congruentcircles_c72,
rsyn0810m) with the built-in `jax_time / rust_time / python_time / node_count`
split and an XLA-compile counter (`jax_log_compiles`). **Three findings overturned
the first draft:**

1. **`rust_time ≈ 0` everywhere on the spatial panel.** gear4 0.01s, ex1252 0.00s,
   kall 0.00s. There is **no LP-solve / Python→Rust round-trip cost** to cut on
   these instances. The first draft's C1 ("per-node McCormick LP + Python→Rust
   round-trip"), inherited from the scip-gap doc's §1.1, is **not supported** here.
   (That doc studied nvs17/19/24 — a different, genuinely LP/cut-bound family. Its
   conclusions are scoped to that family, not the spatial panel.)

2. **There are two JAX cost modes, and the Stage-1 entry experiment + a validated
   patch corrected my initial read of both.** Measured compile counts (20 s):
   ex1252 14 @ 1.08 s; gear4 **412 caught, only 6 distinct signatures** (e.g.
   `concat_constraints` compiled 164×, `lagrangian`/`fn` 82× each, all at the
   *identical* shape `float64[6]`). Identical-shape recompiles ⇒ **not** shape
   variance (my first guess, *falsified*). Attribution: **110 of 111 evaluators on
   gear4 are constructed by `primal_heuristics.py:1045` (`diving`)**, which calls
   `NLPEvaluator(model)` directly, bypassing the existing `_make_evaluator` cache.
   **But** a patch routing those through the cache cut gear4 wall **70.7 s → 55.0 s
   (−22 %, bound-neutral: 5921→5921 nodes, identical obj)** while `jax_time` *barely
   moved* (22.8 → 22.2 s) — the 15 s saving was **entirely `python_time`
   (47.7 → 32.5 s)**. So the evaluator-rebuild cost is the **Python** cost of
   constructing evaluators (DAG/trace/jit setup), *not* the XLA compiles it
   triggers (those were cheap). And gear4's 22 s of JAX is dominated by **per-node
   evaluation × 5921 nodes** (jax scales linearly with node count), which caching
   does not touch.

3. **Python orchestration is the largest single cost on gear4.** `python_time
   47.7 s` > JAX 22.9 s; ~15 s of it (32 %) is the avoidable evaluator-rebuild
   above (validated), the rest (~32 s) is per-node orchestration × 5921 nodes.

### Re-derived cost centers (measured, not hypothesized)

| # | Cost center | Measured evidence | Issues |
|---|---|---|---|
| **CC1** | **Evaluator-rebuild Python cost** — heuristic sites call `NLPEvaluator(model)` directly, bypassing the `_make_evaluator` cache; the cost is Python DAG/trace/jit *construction* (not the XLA compiles) | gear4: 110/111 evaluators from `primal_heuristics.py:1045 (diving)`; **patch → −22 % wall, all from `python_time`** (bound-neutral) | #309, #196 |
| **CC2** | **Per-node orchestration + evaluation × node count** — the bulk of both `python_time` and `jax_time` scale linearly with nodes | gear4 wall 15 s→71 s as nodes 1381→5921; ~8 ms/node Python + ~4 ms/node JAX-eval | #309, #208 |
| **CC3** | **Node count** — the multiplier on CC2 | gear4 5921 nodes (BARON ~handful) | #309, #208 |
| **CC4** | **Incumbent latency** — first feasible point too late under short budgets | kall first incumbent ~12.9 s (mostly JAX before the first node) | #287, #282, #280, #267, #188 |
| **CC5** | **Few but expensive relaxation compiles** — large lifted-McCormick / Jacobian fns compile ~1 s each | ex1252: 1 (cached) evaluator yet 14 @ 1.08 s, triggered from `_tighten_node_bounds_with_status` / the relaxation path | #187, #196 |
| ~~LP / Rust round-trip~~ | **removed for this panel** — measured ≈ 0 | rust_time 0.00–0.01 s | (still applies to nvs* per scip-gap doc) |

rsyn0810m is the one panel member that is plainly **Python-bound** (python 8.8 s,
jax 2.1 s, 127 nodes), reinforcing CC1/CC2.

**Corrected leverage ranking (after the entry experiment):** the *validated*
quick win is CC1 (evaluator caching, −22 % gear4 wall, bound-neutral, low risk —
ship first). But it is **not** dominant: the bulk of gear4 is CC2/CC3 (per-node
cost × 5921 nodes), so **node-count reduction is the largest lever**, not the
recompilation story the first two drafts told. CC5 (ex1252's expensive relaxation
compiles) is a separate, smaller-scoped investigation and is where #187's
"architectural compile" actually lives — narrower than I claimed.

## 2. Stage 0 — Observability & the perf gate (prerequisite, ~1 PR)

> **Implemented.** `discopt_benchmarks/perf/` (`measure.py`, `panel.py`,
> `gate.py`), baseline at `docs/dev/data/perf-baseline.jsonl`, `make perf-gate` /
> `make perf-baseline`. The two previously-missing metrics — `xla_compile_count`
> and `time_to_first_incumbent` — are now recorded per solve. The baseline already
> shows the Stage-1 win: gear4 went from **810 → 5** XLA compiles
> (`compiles/node` ~0.0008). Gate logic is unit-tested in
> `discopt_benchmarks/tests/test_perf_gate.py`.

**Work items**
- **Perf panel**: version-controlled ~25 instances spanning CC1–CC4 (gear4,
  ex1252/ex1252a, kall_*/graphpart_* latency set, syn*/rsyn*, nvs17/19/24,
  autocorr_bern25-25, carton7). < 10 min to run.
- **Record the split that actually matters**: `jax_time`, `python_time`,
  `rust_time`, **`xla_compile_count`** and **`xla_compile_seconds`** (new — this
  pass shows compilation is the cost; we must track it directly), `node_count`,
  `time_to_first_incumbent` (new — CC4 is currently unmeasured).
- **`make perf-gate`**: runs the panel, writes `docs/dev/data/perf-baseline.jsonl`,
  **fails** on any correctness-panel failure or > 15 % regression on wall **or**
  node_count **or** xla_compile_count. Nightly, not on the fast path.

**Exit gate**: `make perf-gate` green on `main`; baseline committed; the compile
counter confirmed (the entire plan now hinges on it, so it must be a first-class,
trusted metric, not a one-off `jax_log_compiles` hack).

**Risk**: low (pure measurement). **This stage must land first** — every claim
below is falsifiable only against it.

## 3. Stage 1 — Route heuristic evaluators through the cache (CC1; the validated quick win)

**Entry experiment — done (2026-06-24), and it falsified two of my hypotheses:**
1. *Shape variance* (draft 2's premise) — **falsified**: the recompiled functions
   carry *identical* shapes (`concat_constraints` float64[6] recompiled 164× on
   gear4), so it is not a shape-keyed cache miss.
2. *XLA recompilation is the dominant cost* — **falsified by a validation patch**:
   routing `diving`'s evaluator through a per-model cache cut gear4 wall −22 % but
   `jax_time` barely moved; the win was **all Python** (evaluator *construction*
   cost), and the dominant remaining cost is per-node work × node count (CC2/CC3),
   not compilation.

**What is actually true (measured):** `primal_heuristics.py:1045` (`diving`) — and
~17 other sites (pounce_layer ×5, differentiable ×8, primal_heuristics ×3, …) —
call `NLPEvaluator(model)` directly, **bypassing the existing fingerprint cache in
`_make_evaluator` (solver.py:414)**. Each rebuild re-pays the Python DAG/trace/jit
construction. On gear4 that is 110 rebuilds ≈ 15 s of `python_time`.

**Work item (small, mechanical, bound-neutral):** route the cache-bypassing
`NLPEvaluator(model)` sites through `_make_evaluator(model)` (or thread the
already-built evaluator down the call chain). The evaluator is stateless in the
evaluation point (you pass `x` to `evaluate_*`), so reuse across dives/heuristics
is exact.

**Exit gate (measured against Stage 0 baseline):** gear4 `python_time` ↓ ≥ 25 % at
**unchanged node_count and identical objective** (validated prototype: 47.7→32.5 s,
nodes 5921→5921, obj identical); the same on the heuristic-heavy panel members.

**Correctness gate:** the run must be **bound-neutral** — assert `node_count` and
certified `objective` are *unchanged* vs baseline on a certifying panel (a cached
evaluator is the same math; if either moves, the cache is wrong). Full correctness
panel green. (The prototype already showed 5921→5921 / identical obj.)

**Risk:** low. Same evaluator object, same numerics; the only failure mode is a
stale cache across a genuinely different model, guarded by the existing fingerprint.

> Note on size: this is a **real but moderate** win (~22 % on gear4, less where
> heuristics fire less). It is *not* the headline lever — CC2/CC3 (per-node cost ×
> node count) is. It ships first because it is measured, safe, and free of search
> changes.

## 4. Stage 2 — Cut Python per-node orchestration (CC2)

gear4 spends **47.7 s in Python** for 5921 nodes (> its JAX). Even after CC1, an
8 ms/node Python tax caps throughput. Profile the per-node Python path
(`py-spy record` on a gear4 run) and attribute the 8 ms: node bookkeeping, array
marshaling to/from JAX, the tree-management glue.

**Likely work items** (decided by the `py-spy` attribution, not guessed)
- Batch node processing (already partially present) so per-node Python fixed costs
  amortize over a batch.
- Move the hot per-node bookkeeping into the existing Rust tree manager (the Rust
  side is nearly idle — `rust_time ≈ 0` — there is headroom to do *more* there).
- Avoid per-node Python⇄numpy⇄JAX array copies for unchanged data.

**Exit gate**: gear4 `python_time` ↓ ≥ 2× at unchanged node_count; nodes/s up
correspondingly.

**Correctness gate**: bookkeeping/marshaling changes are bound-neutral — the
returned optimum and node count must be **unchanged** (assert exact equality of
`objective` and `node_count` on a certifying panel). Full panel green.

**Risk**: low–medium (mechanical), *if* gated by the unchanged-result assertion.

## 5. Stage 3 — Incumbent-first (CC4)

CC4 is now understood as CC1 at the front of the search: kall spends 12.2 s
**compiling** before its first incumbent (not "the spatial node loop is
expensive", as #287 hypothesized). Stage 1 attacks the root cause; Stage 3 adds the
latency-specific safety net.

**Work items**
- Run dive / RENS (#302) / RINS (#276) at the **root**, before the heavy relaxation
  compiles, and **return the incumbent the moment it is found** (#287's suggested
  direction) so a hard-timeout still yields the best feasible point.
- Make the primal deadline-aware so it never becomes the overrun.

**Exit gate**: `time_to_first_incumbent` on kall_congruentcircles_c72 ≤ 6 s (was
12.9 s); zero no-incumbent hard-timeouts on the #287 panel (kall_*, graphpart_*,
tln4, flay04h) at a 10 s budget; syn*/rsyn* (#282) median gap improves (better
early incumbent + more budget for the bound).

**Correctness gate**: incumbents pass the existing feasibility gate (they already
do — these are sound suboptimal points); nothing is certified optimal without the
bound (`gap_certified` discipline). Low correctness risk.

> **Incumbent *quality* sub-gap closed (2026-07-16, #188).** CC4's latency framing
> did not cover the basin-quality gap on pure-continuous nonconvex models: that
> class had zero diversification end to end (pump/ILS/diving/RINS/RENS no-op with
> no integers; root multistart skipped on the MC-LP path; node NLPs warm-start
> from the parent point), so kall_congruentcircles_c51 parked at the 1.5371
> two-row local packing forever. Fix: root stratified continuous multistart
> (`primal_heuristics.continuous_multistart`, `DISCOPT_CONTINUOUS_MULTISTART`,
> default ON) — c51 reconstruction now reaches the 1.07301 global (c41 sibling
> and kall_circles_c8a C-38 lock unregressed).
> **Falsified in passing:** #188's "random multistart is a confirmed dead end
> (40/40 infeasible)" does not hold on the current POUNCE backend — 54/64
> *stratified* starts converge to constraint-verified feasible KKT points
> (~90 ms/solve), 3/32 in the global basin on every seed tried; the 4
> deterministic anchors and LP-vertex seeds never leave 1.54–3.23. The dead end
> was the sampling scheme + old backend, not multistart per se.

> **Generalizes to two more families (2026-07-17, #267).** The #188 lever was
> validated only on *congruent*-circle packing. #267 recorded two sibling gaps;
> both are now closed by the same merged lever with **no code change**, verified
> on reconstructions (`test_issue_267_global_search_gap.py`): a
> **heterogeneous-radii** packing (default parks in a two-row local ~1.841 with
> the flag off, reaches the single-row global 1.525520 with it on — stable to 5
> digits across 10 seeds) and **inscribedsquare02**, a structurally *different*
> family (largest square inscribed in a parametric curve, a maximization,
> `sin`/`cos`/poly constraints). The inscribedsquare02 reconstruction is
> digit-faithful — it reproduces both the published global 0.968017 (reached on
> the default path) and the 0.50885 local basin #267 reported parking in
> (n_starts=32/seed=123). Evidence that the diversification gap, not a
> geometry-specific trick, was the real lever: the fix is class-level.

## 6. Stage 4 — Node-count reduction (CC3; highest leverage, strictest gate)

> **Entry experiment (2026-06-24) — falsified the framing below; read this first.**
> I measured the bound trajectory (`node_callback` → `best_bound`/`incumbent`) and
> the solve path/wall for the node-heavy panel. Three results overturn "branching
> is the safe node-count win":
>
> | instance | nodes | wall | bound behavior | reading |
> |---|---|---|---|---|
> | gear4 | 5921 | **70 s** | **best_bound pinned at 0** for all 5921 nodes, jumps to opt only at the end | bound problem, not branching |
> | nvs17 | 43 | — | bound diverges below opt (uninformative) | bound problem |
> | ex1263 | **15335** | **0.9 s** | Rust MILP B&B ~17 000 nodes/s | node_count ≠ wall — *not slow* |
> | tln4 | 6970 | 13.8 s | Rust MILP path | MILP-side, separate |
> | nvs22 / clay0303hfsg | 103 / 229 | 16 / 30 s | bound *climbs* to opt | branching-lever, but tiny |
>
> 1. **node_count ≠ wall.** ex1263's 15 335 nodes solve in **0.9 s** — chasing its
>    node count would optimize a non-problem. The metric that matters is wall, and
>    the gate must read it that way (it gates node_count only for *slow* certifying
>    instances).
> 2. **The one slow node-heavy instance (gear4) is bound-pinned at 0** — no branch
>    order can fathom against a 0 bound, which is why the *already-implemented*
>    strong/pseudocost/reliability branching does not help it. This is the hard,
>    partially-unsolved **#196** relaxation problem (lift the pinned bound via
>    per-node lifted-FBBT / OBBT-on-aux / SOS1 recognition), **not** branching.
> 3. Where the bound *does* climb (nvs22, clay), node counts are small — modest
>    wins, not the lever.
>
> **Consequence:** there is no clean, safe, high-impact Stage-4 win via branching.
> The real lever is **bound-strengthening on the pinned-bound class** (#196/#208) —
> high value but high risk (a wrong bound is a false certificate, the nvs22 #277 /
> st_ph10 #306 failure mode), so it is a scoped, differential-bound-gated,
> multi-PR effort, not a quick change. A branching PR was **not** shipped because
> the measurement shows it would not move the slow instances.
>
> **Bound-lifting follow-up (2026-06-24) — the one *ready* lever is measured-dead.**
> `obbt.obbt_tighten_root(cascade_aux=True)` already implements OBBT-on-aux (#208):
> capture OBBT's tightening of the lifted product/ratio aux columns and reverse-FBBT
> it back onto the originals. It is **sound** (every optimum preserved). But an A/B
> with the Stage-0 harness confirms it does **not** pay off — neutral on nvs11/nvs12,
> and a regression on nvs13 (35→37 nodes), nvs17 (43→61), and **nvs22 (optimal/103
> nodes → feasible/9.907, fails to certify)**. The perf gate would reject it. So
> `cascade_aux` stays default-off; OBBT-on-aux is **not** the win.
>
> | instance | cascade off | cascade on (#208) |
> |---|---|---|
> | nvs11 / nvs12 | 33 / 13 nodes | 33 / 13 (no change) |
> | nvs13 | 35 nodes, optimal | 37 nodes, optimal |
> | nvs17 | 43 nodes, feasible | 61 nodes, feasible |
> | nvs22 | **optimal, 6.058, 103 nodes** | **feasible, 9.907, did not certify** |
>
> **Correction (2026-06-24, after "SCIP/BARON manage this" pushback) — the
> framing above mis-diagnosed *why* BARON wins.** BARON solves gear4 in 0.18 s.
> It does **not** lift the continuous bound either — gear4's continuous bound is
> genuinely 0 (the continuous ratio can hit the target exactly), confirmed: full
> RLT + all separators + optimal cutoff still give bound 0. BARON wins by
> **branch-and-reduce**: a tight relaxation + *aggressive range reduction*
> collapses the integer box, then the tiny remainder is enumerated. The
> certificate is "no better integer point exists in the reduced box," **not** a
> lifted LP bound.
>
> The measured gap is therefore **range-reduction strength**, not bound-lifting:
> - gear4 optimum is `x0=19, x1=16, x2=49, x3=43`.
> - discopt's cutoff-OBBT collapses x0,x1 to [12,43] but leaves x2,x3 at [12,60]
>   → a ~32×32×49×49 ≈ **2.5 M** box → 5921 nodes of mostly-unfathomable spatial
>   branching.
> - It stalls there because the ratio-of-products McCormick envelope is too loose
>   for OBBT to bite further (5 OBBT rounds tighten nothing more; cutoff=50 and
>   cutoff=1.6434 give the *same* box).
>
> So Stage 4 **is** bridgeable, via the SOTA **branch-and-reduce** stack, which
> discopt has only weakly:
> 1. **Tighter ratio-of-products / bilinear relaxation** (#185 `r·q=m` envelope,
>    #201 reciprocal-quadratic, integer-aware products) so range reduction bites.
> 2. **Aggressive range reduction at every node** — integer-rounded cutoff-OBBT,
>    **probing on integer values**, and **duality/marginal-based reduction
>    (DBBT)** — to collapse the box the way BARON does.
> 3. (1) and (2) **compound**: a tighter relaxation makes reduction cut deeper,
>    which tightens the relaxation again.
>
> This corrects two earlier errors: over-generalizing from the one `cascade_aux`
> negative result to "no win," and conflating "continuous bound is 0" with
> "unsolvable fast" (BARON is fast *without* lifting that bound). The work is
> substantial and still strictly correctness-gated (every reduction must be valid
> over the relaxation polytope — a wrong reduction cuts the optimum = a false
> certificate), but it is **identifiable, SOTA-aligned work, not a dead end**.
>
> **Recommended first step (measured spike, with a kill criterion):** prototype
> integer probing + integer-rounded cutoff-OBBT to a fixpoint on gear4 and confirm
> the box collapses well below 2.5 M (kill the spike if it does not), then add the
> #185 ratio envelope and re-measure. Ship only what the perf gate shows reduces
> nodes/wall while staying 0-incorrect + differential-bound-sound.
>
> **SPIKE RESULT (2026-06-24) — kill criterion fired, and it relocated the lever.**
> Range reduction (cutoff-OBBT + integer probing/shaving to a fixpoint) on gear4
> with the *optimal* cutoff **stalls at a 2.46 M box** (5.76 M → 2.46 M, then
> fixpoint; probing matched OBBT exactly — they share the loose McCormick
> relaxation). And a tighter continuous ratio envelope would not help either:
> x0=43 is genuinely continuously feasible at the target ratio (x1=12, x2≈x3≈59.8).
> So **neither reduction nor bound-lifting is gear4's lever.**
>
> The per-node measurement reveals the real gap — **per-node speed, not node count:**
>
> | instance | path | nodes | wall | **ms/node** | split |
> |---|---|---|---|---|---|
> | gear4 | spatial / JAX | 5921 | 54.9 s | **9.28** | 40 % JAX, 59 % Python, 0 % Rust |
> | ex1263 | Rust MILP B&B | 15335 | 0.9 s | **0.059** | ~all Rust |
>
> gear4's nodes are **~157× slower** than the Rust path's. At Rust-path speed its
> 5921 nodes would take **~0.35 s ≈ BARON's 0.18 s** — i.e. **per-node speed alone
> closes the gap**, regardless of BARON's exact node count. gear4 is on the slow
> spatial/JAX path (per-node McCormick-LP rebuild + JAX eval + Python orchestration
> at ~9 ms/node); ex1263 is on the fast native path.
>
> **Corrected lever (measured): CC1/CC2 — speed up the spatial per-node path**, not
> CC3 (node count) and not bound-lifting.
>
> **ROUTING SPIKE (2026-06-24) — found the precise bottleneck, and corrected a
> magnitude error.** The per-node LP *solve* is **already** routed through the Rust
> warm-started simplex (`MccormickLPRelaxer(backend="simplex")` is the default —
> that's why `rust_time ≈ 0`: the solve is already fast). So "route the solve to
> Rust" is done and is *not* the remaining cost. A cProfile of gear4's per-node
> path shows the real bottleneck: **the McCormick relaxation is rebuilt from
> scratch every node.** `solve_at_node` calls `build_milp_relaxation(...)` per
> call — 6244 times (≈ once/node):
>
> | per-node hot spot | cumtime |
> |---|---|
> | `build_milp_relaxation` (DAG walk → rows) | 5.3 s |
> | `equilibrate_relaxation_lp` (Ruiz scaling) | 8.3 s |
> | constraint-matrix build + product decompose | ~3 s |
>
> ≈ **half the wall is rebuilding the same relaxation structure** (only the bound
> box changes node-to-node). The lever is therefore **incremental relaxation
> reuse**: build the structure once, and per node update only the bound-dependent
> McCormick envelope coefficients (reuse/cheap-refresh the equilibration). This is
> exactly the #316 pattern (stop rebuilding per call), applied to the node-LP.
> Bound-neutral (identical relaxation math → identical bound → identical search),
> so low correctness risk, gate-validated on node_count-unchanged + wall-down.
>
> **Honest magnitude (correcting the earlier "→0.35 s ≈ BARON" claim):** that
> comparison was apples-to-oranges — ex1263's 0.06 ms/node is a *pure-MILP* node
> (no McCormick relaxation at all, because #285 linearizes it); gear4 is a genuine
> ratio that **cannot** be reformulated to a pure MILP, so it will always carry a
> per-node McCormick LP. Incremental reuse is a realistic **~2× per-node win
> (gear4 ~55 s → ~25–30 s)**, not full BARON closure. It is a real, low-risk,
> measurable improvement for the whole spatial-nonconvex class — but the residual
> gap to BARON also involves tighter relaxations/reduction that compound, which
> remain harder, separate work.

CC3 multiplies CC1+CC2: gear4's 5921 nodes are *why* it pays its Python cost. But
per the entry experiment, fathoming those nodes needs a non-zero bound first — so
the work below is gated on the bound lifting off 0, and a tighter bound or a
mis-scored branch is exactly how a **false certificate** (nvs22 #277 / st_ph10
#306) is born, hence the heaviest correctness gate.

**Work items** (each behind a flag, each gated by node-count *and* the cert invariant)
1. **Pseudocost / reliability branching** on the spatial + integer columns (#309).
2. **OBBT-on-auxiliaries** (#208): rebuild McCormick envelopes from OBBT-tightened
   aux bounds and cascade back — currently discarded.
3. **Envelope tightening** for the loose fractional-power / ratio relaxations (#189,
   gear4 class).
4. **Complemented-MIR / aggregation cuts** for the integer-product family — this is
   the existing `scip-gap-closing-plan.md` Phase 1, *with its §1.5 node-reduction
   gate* (cuts must demonstrably cut nodes, not just exist).

**Exit gate**: gear4 5921 → **< 500 nodes** and wall **70 s → < 10 s**; ≥ 2× node
cut on the bilinear-integer class (ex1263a/tln4/clay0303hfsg).

**Correctness gate (strict)**: a **differential bound test** — the new relaxation's
bound at a fixed set of boxes must be ≥ the old bound *and* ≤ the true box optimum
(trusted dense solve); a bound that ever exceeds the box optimum is a false
certificate and blocks the change. Branching changes are bound-neutral — assert the
certified optimum is unchanged under a branching-rule A/B. Adversarial suite + full
panel green for 3 consecutive nightlies before default-on.

**Risk**: high (correctness-sensitive). Flag-gated, A/B'd, default-off until green.

> **Binary-multilinear MILP route follow-up (2026-07-16) — "sparse-MILP LP
> throughput ⇒ autocorr 25-25 certifies" FALSIFIED.** After #187's exact
> linearization (PR #667) routed the autocorr class to the MILP engine and #663's
> sparse CSC engine landed, the hypothesis was that node-LP throughput was the
> remaining blocker to *certifying* autocorr_bern25-25 in budget. Measured on the
> reformed 1,224-row MILP (synthetic Bernasconi n=25 dense):
>
> | lever | result |
> |---|---|
> | sparse engine, 600 s | 8,415 nodes, **dual bound frozen at 12.0** (= the parity floor: one per odd-length lag) — no visible progress toward the optimum 36 |
> | generic root cuts (cover/clique/GMI/MIR via `DISCOPT_P3_FORCE_CUT_PATH`) | bound unchanged at 12.0; root ~10× slower |
> | perfect incumbent (36) | frontier bound stays ~12 → pruning threshold barely matters for tree size |
>
> The LP relaxation sits at the parity floor because the LP can hold every
> ``y_k`` at its parity-nearest-zero value with fractional ``b = 1/2`` and
> loose Fortet ``z``; branching individual bits barely moves it until deep in
> the tree. This is the **same bound-pinned phenomenon as gear4 above**, now on
> the lifted binary-product (Boolean-quadric) polytope: certification needs
> BQP/PSD-class strengthening of the ``z``-polytope (triangle inequalities,
> PSD moment cuts à la #663's `X_ii = x_i` recognition — currently only on the
> spatial path), a scoped research effort, not an engine or throughput fix.
> **What DID pay** (shipped in the follow-up PR): incumbent seeding — the
> class-gated 1-flip local search finds the true optimum 36 in ~0.5 s, and with
> it the 30 s answer improves from `feasible 84 / bound 12` to `incumbent 36 /
> bound 12`; n=13 dense *certification* drops 3.7 s → 0.4 s (the seed collapses
> the proving phase where the bound does move).

## 7. Sequencing & rationale (revised by the measurement)

```
Stage 0 (gate) ─► Stage 1 (kill recompilation — the dominant measured cost)
                      │
                      ├─► Stage 2 (Python per-node tax)
                      ├─► Stage 3 (incumbent-first; a CC1 symptom at the root)
                      └─► Stage 4 (node count; multiplies CC1+CC2, strictest gate)
```

- **Stage 1 (evaluator caching) ships first because it is validated, safe, and
  search-neutral — but it is *not* the main lever.** The entry experiment showed
  the dominant cost is CC2/CC3 (per-node work × node count); Stage 1 buys a
  measured ~22 % on gear4 with near-zero risk while the harder node-count work
  proceeds. (Two earlier drafts called recompilation "the main event"; the
  validation patch disproved that — `jax_time` barely moved.)
- **The largest lever is Stage 4 (node count)** — gear4's 5921 nodes are why it
  pays CC2 5921 times. That is also the strictest correctness gate, so it comes
  after the cheap, safe wins (1–3) harden the cutoff and the panel.
- **Stage 0 still first** — every claim above is only believable against the
  baseline harness and the bound-neutral / certificate gates.
- **2 and 3 are independent** of 1's internals and can proceed in parallel once the
  panel exists; both are bound-neutral / low-risk.
- **Stage 4** last among the substantive stages: biggest lever, strictest gate, and
  it benefits from cheaper nodes (1+2) and a tight early cutoff (3).

## 8. What I am NOT claiming

- The panel is **4 instances** (plus the Stage-0 set). The CC1/CC2 findings are
  strong for the **spatial-relaxation path** (gear4/ex1252/kall). The nvs*
  integer-product family (scip-gap doc) is LP/cut-bound and is **out of this
  cost model's scope** — do not assume Stage 1 helps it without measuring.
- The **targets** (gear4 < 500 nodes, ex1252 jax ≤ 5 s) are goals, not proven
  reachable. Each is a falsifiable exit gate; if a stage's entry experiment kills
  its hypothesis (e.g. the recompiles turn out shape-invariant), the stage pivots,
  documented, rather than pushing a fix that doesn't move the metric.
- This is the lesson of the week: **no fix ships on a hypothesis.** Every stage
  above names the experiment that must confirm its premise *before* code, and the
  metric + correctness gate that must move *after* it.

> **Falsified (2026-07-10, task #94 — A-UNBOUNDED, F9):** "Finitizing the unbounded
> continuous vars on the nvs05/tanksize/casctanks certification-stall class is the
> lever." FBBT already finitizes 100 % of them (nvs05 4/4, tanksize 26/26,
> casctanks 296/296 — the constraints imply a bounded region), and the solver's root
> OBBT already applies equivalent bounds, so FBBT-preconditioning closes **0**
> additional gap end-to-end. nvs05's objective contains no unbounded var (root bound
> 0.674 is a loose interval enclosure of `x0²·x1`; final 1.3521 is the F8 taint floor).
> `DISCOPT_ROOT_FIXPOINT`+`DISCOPT_NODE_REDUCE` are inert (both gate on the McCormick-LP
> relaxer this class routes away from). This is a **Lever-A (relaxation-strength)**
> class, not bound-inference — consistent with DECOMP-1. Full record + reproduction:
> `docs/dev/a-unbounded-entry-2026-07-10.md`; ledgered as F9 in
> `gap-closing-execution-plan.md` §6.

> **Falsified (2026-07-11, task #98 — P1-A2, F13):** "The MINLPLib ex6_2 Gibbs/log-sum
> objective is a sum of convex atoms (whole objective convex), so joint outer-approximation
> cuts of the full objective collapse the ~300× root gap left by summing per-atom tangents."
> The premise is false: the objective is **nonconvex** — its Wilson activity terms
> `−x_i·log(a·x+b·y+…)` (the 24 `neg(...)` nodes in ex6_2_5) are nonconvex at 100% of box
> points, and the whole objective has a negative Hessian eigenvalue at 78%/92%/99.5% of box
> points (ex6_2_5/9/10). Joint-OA is therefore **unsound** (a gradient cut of a nonconvex
> function is not a valid underestimator), and the sound joint alternative (αBB over the whole
> objective) is ~1e40–1e52× **worse** — the `x·log(x)`/`log(x)` `~1/x` Hessian singularity at
> the box edge drives rigorous α to ~1e40. Per-atom relaxation is used *because* it is the only
> tractable sound handling of this singular structure. Residual looseness is 100% in the
> objective (constraints are linear mass-balances), but the lever is a tighter **per-atom**
> `x·log(affine)` envelope, not a joint cut — Lever A on that composite. No code shipped.
> Record + reproduction: `docs/dev/p1a2-gibbs-log-sum-oa-entry-2026-07-11.md`; ledgered as
> F13 in `gap-closing-execution-plan.md` §6.

## 9. Engine layer — density-aware LU route (#557): the nvs21 certificate loss

> **Entry experiment (2026-07-10, task #77) — the conditioning-gate hypothesis is
> falsified; the mechanism is the LP failure rate, and the fix is
> failure-triggered (task #85).**
>
> Context: the `DISCOPT_LU_DENSITY_ROUTE` route (routes m∈(16,256] sparse
> McCormick bases to feral's sparse LU; #573) failed its graduation gate on one
> instance: **nvs21** goes `optimal` → `feasible` with the route ON — same
> correct incumbent (−5.68478…), but the final dual bound sticks at
> **−15 901 749** (vs −5.68522 OFF). Hypothesis under test: the offending bases
> are ill-conditioned, so a factorization-time condition estimate can divert
> them to the dense LU.
>
> Instrumented every sparse factorization (feral `condition_estimate_1`
> Hager–Higham κ₁ + `growth()`, incl. mid-solve refactorizations) and attributed
> them to their LP solve's outcome:
>
> | population (route ON) | n(fact) | κ₁ p50 | p90 | p99 | max |
> |---|---|---|---|---|---|
> | nvs21, factorizations in FAILING LP solves (`Numerical`/`IterLimit`) | 59 | **1.6e1** | 1.5e9 | 7.6e18 | 7.6e18 |
> | nvs21, factorizations in OPTIMAL LP solves | 391 | 2.1e5 | 3.1e7 | 2.4e9 | 1.0e10 |
> | st_e36 (certifies optimal), OPTIMAL solves | 832 | 2.8e6 | **7.9e10** | 3.3e12 | **3.2e13** |
> | nvs06 (certifies optimal), OPTIMAL solves | 72 | 5.0e8 | 7.5e12 | — | **1.2e16** |
>
> 1. **The populations are inverted, not merely overlapping**: most failing
>    solves factorize *beautifully conditioned* bases (κ₁ p50 = 16), while
>    healthy instances routinely succeed at κ₁ 10¹⁰–10¹⁶. No threshold — let
>    alone one with the required ≥2-orders margin — separates them. A gate tight
>    enough to catch nvs21's failures diverts essentially everything (killing
>    the st_e36-class win); a gate loose enough to spare the healthy population
>    catches 2 of 39 failures. `growth()` is 1.0 uniformly — zero signal.
>    **Conditioning-gate: KILLED.** The failures develop during the *iteration*
>    (a κ₁=8 basis factorizes cleanly, then the solve breaks), invisible at
>    factorization time.
> 2. **The −1.6e7 values are not corrupted LP optima.** They appear in the
>    route-OFF run too — they are legitimately loose early lifted-McCormick
>    relaxation bounds. The pathology is that the node carrying one never gets
>    closed.
> 3. **The real ON-vs-OFF delta is the LP failure rate**: `Numerical`/`IterLimit`
>    exits 39 ON vs 12 OFF (3.25×). A node whose LP fails is abandoned with its
>    inherited loose bound → final bound stuck → certificate lost. Sound (the
>    bound stays valid) but uncertified.
>
> **Consequence (task #85):** the fix is **failure-triggered, not predictive** —
> when a solve fails with the route ON, re-solve that LP once, cold, with the
> route suppressed (the robust dense-preferring path). It uses the failure
> signal the solve already reports, needs no tunable threshold, and is sound by
> construction (only ever replaces a *failure*; never accepts or blends a
> suspect fast-path result; cold because the failed run's warm state is exactly
> what is in doubt). Validated: nvs21 ON → `optimal`, bound −5.68522, 30
> retries / 27 rescues; st_e36 win preserved at 1.65× (24.8 s → 15.0 s);
> panel: 10/10 `optimal→optimal`, node counts identical, incorrect_count 0.
> Implemented in `lp/simplex/primal.rs` (`dense_retry`), counters
> `LpDenseRetries`/`LpDenseRetryRescues`, still behind
> `DISCOPT_LU_DENSITY_ROUTE` (default OFF).
## Appendix B — per-solve fixed startup floor (OVERHEAD-1, task #83; 2026-07-10, `JAX_PLATFORMS=cpu`, x64)

Decomposition of a fresh-process trivial solve (5× stable, ex1222/st_test1/gbd/alan/nvs01;
timer convention of the global50 harness: starts **after** `import discopt`, so the
measured window is `from_nl` + `solve()` and everything `solve()` lazily imports):

| component | ms | in measured window? | class it hits |
|---|---|---|---|
| python+site+numpy+`import discopt` | ~110–130 | **no** (excluded by harness) | all |
| `.nl` parse (`from_nl`) | 1–4 | yes | all |
| `import jax` + devices + first tiny jit | ~240–300 | yes | nonlinear-relaxation class only — MILP/MIQP solves never import jax (measured) |
| `import sympy` via `cut_recognizer` (solver.py structure-cut presolve) | ~100–120 | yes | **was: every solve**; now only models with a nonlinear `==` row (fixed this task) |
| `import pounce` → `scipy.optimize` | ~125–150 | yes | every solve that reaches an NLP/QP relaxation or heuristic (incl. the MIQP node path `_pounce_qp_relaxation_nodes`) |
| first-solve JAX trace/compile beyond imports | ~75 | yes | nonlinear class |
| recurring engine work (trivial instance) | ~150 | yes | all (not floor) |

Per-class in-window floor before the fix: MILP/MIQP ≈ 0.28–0.34 s (sympy 35–40%,
pounce ~40%), nonlinear ≈ 0.55–0.65 s (jax ~40–45%, pounce ~25%, sympy ~18%).

**Shipped (this task):** lazy SymPy in `cut_recognizer` behind its own sympy-free
`has_square_difference_candidate` pre-check (the `symbolic/` package's lazy-SymPy
invariant already mandated this). MILP/MIQP-class wall: st_test1 0.30→0.19 s,
gbd 0.27→0.16 s, alan 0.34→0.23 s; ex1222 (no nonlinear `==` rows) 0.82→0.70 s.
Verified **exactly bound-neutral** by a full differential over the 41-instance
cert panel (pre-fix vs post-fix code, same machine, back-to-back): node_count,
objective, and status bitwise-identical on all 41. (The committed
`cert-baseline.jsonl` is stale vs current `main` on nvs02/nvs11/nvs12 —
pre-existing at the branch base, confirmed by identical node counts with the
fix reverted.)

Easy-class panel (BARON-optimal-in-<1 s per the 2026-06-18 record → 30 local
instances; TL=60, two back-to-back A/B interleaved runs each, loaded machine —
load avg ~5–8 from sibling agents, identical conditions for A and B):

| pass | median wall | p25 wall | geomean vs recorded BARON |
|---|---|---|---|
| before run 1 | 0.709 s | 0.273 s | 13.42× |
| before run 2 | 0.770 s | 0.271 s | 13.75× |
| after run 1 | 0.600 s | 0.163 s | 10.32× |
| after run 2 | 0.570 s | 0.175 s | 10.42× |

(−20% median, −37% p25, −23% geomean; every MIQP-class instance −32…−40%.)

**Killed by the ≥20%-of-floor criterion (do not relitigate without new data):**
- *Persistent JAX compilation cache* (`jax_compilation_cache_dir`): the cacheable
  XLA-compile share of the floor is inside the ~75 ms trace+compile residue
  (≈12% of the nonlinear-class floor; tracing, which dominates it, is not cached).
- *Lazy-import surgery on `import discopt`* (scipy.sparse via `discopt.decomposition`
  ≈63 ms): outside the harness window and ≈9% of the total user-facing floor.
- *Deferring JAX init*: already the case — jax is imported only when a nonlinear
  relaxation path runs; pure MILP/MIQP solves never touch it (measured).
- *No-JAX fast path for linear/quadratic*: already exists (same measurement).

**Out of repo (recorded, not actionable here):** (a) `import jax` ~240–300 ms is the
single biggest remaining floor item on the nonlinear class — it is upstream cost
(plugin discovery via `importlib.metadata` over the venv is a large slice);
(b) `import pounce` spends ~90% of its ~140 ms importing `scipy.optimize` inside
`pounce/_minimize.py` — a pounce-repo fix (defer scipy.optimize there) would cut
~40% of the MILP/MIQP-class floor for every consumer.

## Appendix — raw measurement pass (2026-06-24, `JAX_PLATFORMS=cpu`, x64)

Built-in split (first pass):

| inst | status | wall | nodes | jax_t | rust_t | py_t | nodes/s |
|---|---|---|---|---|---|---|---|
| gear4 | optimal | 70.7 | 5921 | 22.81 | 0.01 | 47.68 | 83.7 |
| ex1252 | time_limit | 72.5 | 7 | 54.18 | 0.00 | 18.23 | 0.10 |
| kall_congruentcircles_c72 | feasible | 15.3 | 5 | 12.22 | 0.00 | 3.07 | 0.33 |
| rsyn0810m | feasible | 10.9 | 127 | 2.07 | 0.02 | 8.80 | 11.7 |

Compile-vs-eval (jax_time scaling):

| inst | tl | nodes | jax_t | jax/node | reading |
|---|---|---|---|---|---|
| ex1252 | 15/30/60 | 3/3/7 | 11.8/26.5/53.4 | 3.9/8.8/7.6 | jax grows at constant nodes ⇒ not eval |
| gear4 | 15/40/90 | 1381/3463/5921 | 4.3/12.7/22.9 | 0.003/0.004/0.004 | jax/node constant ⇒ scales with nodes |

XLA compile counts (20 s budget):

| inst | nodes | jax_t | XLA compiles | s/compile | reading |
|---|---|---|---|---|---|
| ex1252 | 3 | 15.1 | **14** | 1.078 | few, expensive compiles ≈ whole solve |
| gear4 | 1373 | 4.7 | **810** | 0.006 | many cheap recompiles ≈ whole jax_time; ~0.6/node |

Stage-1 entry experiment (the recompiles are identical-shape ⇒ re-creation, not
shape variance):

| inst | distinct compile signatures | example | evaluator constructions (top site) |
|---|---|---|---|
| gear4 | 6 (of 412 caught) | `concat_constraints` float64[6] ×164 | **110/111 from `primal_heuristics.py:1045 diving`**; 1 from cached `_make_evaluator` |
| ex1252 | 5 (of ~14) | `concat_constraints` float64[45] ×2 | **1** (cached) — recompiles come from the relaxation/`_tighten_node_bounds`, not evaluator rebuilds ⇒ CC5 |

Stage-1 validation patch (route `diving` through a per-model evaluator cache):

| metric | gear4 baseline | gear4 cached | Δ |
|---|---|---|---|
| wall | 70.7 s | **55.0 s** | **−22 %** |
| python_time | 47.7 s | **32.5 s** | **−15.2 s (the entire win)** |
| jax_time | 22.8 s | 22.2 s | ≈0 (compilation was *not* the cost) |
| node_count | 5921 | 5921 | unchanged (bound-neutral) |
| objective | 1.6434285 | 1.643428 | identical (sound) |
