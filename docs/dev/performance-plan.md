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

> *Measured 2026-07-17 (`issue-282-syn-rsyn-diagnosis-2026-07-17.md` §R2) — the #282
> clause above is half-right and should be scored accordingly.* CC4 genuinely owns
> #282's **short-budget** regime: at the issue's 5 s budget **6/7 syn/rsyn return no
> incumbent at all** because the root alone costs 5–21 s (`presolve` ≈ 2.6 s/call). But
> "better early incumbent" does **not** close #282 at 60 s+: the gap there is
> **dual-dominated 7/7**, and on `syn15m02hfsg`/`syn30hfsg` the incumbent is already the
> proven optimum. For #282, score CC4 on *no-incumbent-at-short-budget*, not on median gap
> — the median gap is the bound's to close (§R2-6).

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
> **GNN-branching consequence (2026-07-18) — scaffold removed, #236 closed
> not-planned.** The untrained GNN branching path (`branching_policy="gnn"`)
> was removed rather than trained: it had always been inert (hardcoded
> `params=None` → most-fractional fallback), and the entry experiment above
> shows its best case — a cheap imitation of strong branching — cannot move
> the slow tail, since *actual* strong/pseudocost/reliability branching
> already does not. Revisit only if the slow tail stops being bound-limited
> (post-#196 branch-and-reduce work).

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

> **Falsified (2026-07-17, issue #673 — "z-polytope (BQP/PSD) cuts certify the
> autocorr class").** The 2026-07-16 entry above conjectured that
> BQP/PSD-class strengthening of the lifted binary-product (Boolean-quadric)
> `z`-polytope would move the reformed-autocorr root dual bound off the parity
> floor. Issue #673 scoped three strengthenings "in increasing order of
> ambition": (1) Padberg triangle inequalities, (2) PSD moment cuts on
> `[1 b; bᵀ Z]` with `Z_ii = b_i` (the #663 recognition), (3) square-linkage RLT
> coupling the `y_k` epigraphs with the `z` vars. The entry experiment measured
> the reformed-autocorr root LP bound at the **full closure** of each family
> (whole family added, LP re-solved to the polytope optimum — the strongest the
> family can give), before writing any cut code:
>
> | instance | parity floor | base | +triangle (closure) | +PSD moment (Shor closure) | +square-RLT | opt |
> |---|---|---|---|---|---|---|
> | n=6 dense | 3 | 3.0 | 3.0 | 3.0 | 3.0 | 7 |
> | n=8 dense | 4 | 4.0 | 4.0 | 4.0 | 4.0 | 8 |
> | n=10 dense | 5 | 5.0 | 5.0 | 5.0 | 5.0 | 13 |
> | n=13 dense | 6 | 6.0 | 6.0 | 6.0 | 6.0 | 6 |
> | **n=25 dense** (the issue's 1,224-row instance) | **12** | **12.0** | **12.0** (all 2,300 triangles) | **12.0** | — | 36 |
>
> **All three directions leave the bound exactly at the parity floor**, including
> the concrete thing the issue pointed to (#663's `Z_ii=b_i` PSD recognition
> ported to this route — tested here as the full pairwise Shor closure). The
> triangle cuts *do* separate the LP vertex (8 violated at the n=8 optimum, max
> 0.18), but an alternate optimal face at the same objective satisfies them, so
> the closure bound does not move. **Root cause:** the sum-of-squares objective
> is relaxed square-by-square through the *exact* 2D convex hull of
> `{(y_k, y_k²)}` (the secant envelope — already the tightest possible 2D
> relaxation of each `C_k²`), and `y_k` is *affine* in `(b, z)`. Σ`t_k` reaches
> the parity floor by driving each `y_k` independently to its parity-nearest
> attainable value; every proposed strengthening constrains only the pairwise
> `(b, z)` polytope and its affine link to `y_k`, none of them the **joint**
> realization of `(C_1,…,C_K)`. That joint coupling — "the correlations cannot
> all be near-zero at once" — is a degree-≥4 property absent from the pairwise
> moment matrix (the flat degree-4 Fortet lift is *worse*, −529 vs 3.0 on n=6,
> so the secant hull is the right relaxation, not the lever). It is the
> LABS/merit-factor combinatorial lower bound, which is (a) not a Boolean-quadric
> property, so the issue's entire proposed avenue cannot deliver it, and (b)
> autocorr-class-specific (a Fourier sum-rule), which Dev-Philosophy #2 forbids
> as a single-problem solution. **No cut code shipped** — shipping a family that
> provably does not move the metric would fail the issue's own exit gate
> ("root bound moves materially above the parity floor") and Dev-Philosophy #3/#4.
> Reproduction: `discopt_benchmarks/scripts/bqp673_zpolytope_falsification.py`;
> pinned by `python/tests/test_bqp_zpolytope_falsification.py`. Re-scope: the
> lever for this class is cross-square/joint-correlation coupling, a distinct
> higher-risk research direction — not `z`-polytope cuts.

> **Falsified (2026-07-17, issue #677 — "joint-correlation / degree-4 moment
> coupling certifies the autocorr class").** #673's re-scope pointed at the only
> remaining lever: coupling the *joint* realization of `(C_1,…,C_K)` via the
> degree-4 moment (Lasserre **level-2**) relaxation over `s ∈ {±1}`. Entry
> experiment (before any engine integration, per #677's kill criterion): solve
> the **combined** relaxation — the strongest form — the level-2 moment matrix
> `M(y) ⪰ 0` (which *does* see the cross-square degree-4 couplings the pairwise
> matrix misses) **plus** the per-square parity secant cuts `t_k ≥ (u+v)y_k−uv`
> (pseudo-moments need not put `(E[C_k],E[C_k²])` in the integer hull, so the
> secant cuts genuinely add to the SDP). Solved with CLARABEL/SCS:
>
> | n | parity floor | level-2+secant | movement | opt |
> |---|---|---|---|---|
> | 6 | 3 | 5.00 | **+2.0** | 7 |
> | 8 | 4 | 4.76 | +0.76 | 8 |
> | 10 | 5 | 5.77 | +0.77 | 13 |
> | 13 | 6 | 6.00 | +0.0 (=opt) | 6 |
> | 20 | 10 | 10.001 | ~0 | — |
> | **25** (target) | **12** | **11.9999** | **~0** | 36 |
>
> The lever is **real but decays to zero by the target scale**: it moves the
> bound above the floor only for small n (n=6 +2.0, n=8/10 ≈+0.76), the absolute
> movement shrinks with n, and at n=25 the combined bound sits exactly at the
> parity floor (11.9999, optimum 36). Worse, the mechanism is **intractable
> in-solver**: the level-2 moment matrix is dense `326×326` at n=25 (15,275
> moment vars), ~2 min for a *single* root relaxation at loose accuracy — hopeless
> against the 60 s budget, before even considering per-node use. #677 direction 2
> (the LABS Fourier sum-rule) is a single `vᵀM v ≥ 0` inequality **subsumed by
> the level-1 PSD closure** #673 already found inert (12.0), so it is dead by
> domination. Pure level-2 *without* the secant cuts is actually **weaker** than
> the floor on most n (n=8 3.67<4, n=13 3.47<6) — it drops the per-square
> integrality the reformed model exploits. **No code shipped.** This settles the
> autocorr **dual-side** wall: no proposed relaxation lever (BQP/PSD `z`-polytope,
> degree-4 moment, sum-rule) moves the n=25 bound off the parity floor tractably;
> the LABS/merit-factor lower bound is genuinely hard (consistent with the weak
> LP/SDP relaxations in that literature). The shipped practical state stands —
> incumbent seeding returns the true optimum 36 in ~0.5 s; only certification is
> open, and it is open on a hard-bound, not an engine, gap. Reproduction:
> `discopt_benchmarks/scripts/joint_correlation_moment_probe.py`.

> **Falsified (2026-07-18, issue #707 — "cutoff-driven range reduction certifies
> the ex1252 class").** #707 shipped the flow-aware integer-multilinear envelope
> (`DISCOPT_INTEGER_MULTILINEAR_REFORM`), which lifts ex1252's dual bound off its
> structural 5134 floor by exact-linearizing the objective's integer-multilinear
> terms `(c+1800·x15)·x0·x3·x18`. That closes the barrier the issue *diagnosed*,
> but does not certify ex1252 — the reformed dual climbs only to ~48k (opt
> 128893.74). The natural next hypothesis was that discopt's dual lags because it
> never gets an incumbent early, so cutoff-driven OBBT/DBBT never fires; the entry
> experiment fed the **known optimum in as an objective cutoff** and ran root OBBT
> before writing any code:
>
> | box | OBBT no-cutoff | OBBT + cutoff=opt | note |
> |---|---|---|---|
> | root (all indicators free) | 0.0 | **0.0** | obj relaxes to 0 → cutoff never binds |
> | line-1 selected (x18=1) | 12658 | **12658** | identical; cutoff still slack (12658 ≪ 128893) |
> | + subdivide continuous x12 | 12658 | 12658 | branching the flows does not move it |
> | + fix integer x0=2,x3=1 (loosest node) | 12658 | 12658 | vs true ≈128893 → **~10× loose** |
>
> **The cutoff is inert at every level**: the relaxation is so loose (12658 on the
> binding node) that `obj ≤ 128893` is trivially satisfied, so it propagates
> nothing — range reduction / incumbent cutoff is **not** the lever. **Root
> cause:** the continuous cost rows `x15 = a·x6³ + b·x6²·x12 + c·x12²·x6`
> (`x6∈[0,2950]`, `x12∈[0,350]`) are relaxed term-wise, and the `w=x6²` lift secant
> alone spans `w∈[0,8.7M]` — enormously loose, feeding every downstream term. Even
> with the line selected *and* the integer flow factors fixed, the McCormick
> relaxation of the cubic gives 12658 vs a true value ~10× higher; subdividing the
> continuous flows does not tighten it. SCIP's dual reaches 128438 (0.35% gap) at
> 120 s — its cubic relaxation + cuts are ~10× tighter on exactly this block; and
> **SCIP itself does not certify ex1252 in 120 s**, so this is a SOTA-frontier
> bound gap, not an engine/throughput fix. Re-scope (issue #721): the lever is a
> **stronger relaxation of the wide-range cubic block** — auto piecewise-McCormick
> on wide monomial factors (the `x6²` secant first), edge-concave/vertex-polyhedral
> envelopes extended to cubic (non-quadratic) blocks (catalog §7's open item), and
> RLT tying the cubic equality to the bound/bilinear rows — *not* cutoff/OBBT
> orchestration, which only compounds once the relaxation is strong enough to make
> the cutoff bind. Reproduction:
> `discopt_benchmarks/scripts/ex1252_cutoff_obbt_falsification.py`.

> **Falsified + root-caused (2026-07-18, issue #723 lever 3 — "cap per-node LP
> solves; nvs05 does ~59 LP/node").** #723's first three levers are handled:
> convexity re-classification is de-duped (commit `ecff43e`), the RENS primal
> heuristic is throttled by the default-ON G2 governor (`heuristic_governor.py`,
> `EXPENSIVE_SOURCES = {"rens"}`), and the interval-`__mul__` / JIT-recompile
> overheads are cut (`ecff43e`, `bfc9d55`). Lever 3 — "nvs05 does ~59 per-node LP
> solves; find why and *bound it*" — was the open one, and the entry experiment
> (measurement before code, Dev-Philosophy #4) **falsified the "cap it" framing**
> and relocated the cost.
>
> The per-node LP solves are **per-node OBBT** (`obbt_tighten_root` on each
> branched node's box against its McCormick relaxation): the load-bearing lever
> that lets the nvs05/welded-beam class certify at modest node counts (it pins the
> functionally-dependent continuous outputs once the integer drivers are fixed).
> Two findings (`nvs05_obbt_probe_cost_measurement.py`):
>
> 1. **The probes are productive, not wasteful — capping rounds trades bound for
>    wall.** Per-node OBBT runs up to `_PER_NODE_OBBT_ROUNDS` (3) sweeps, stopping
>    at a `sweep_tight == 0` fixpoint. On nvs05 the sweeps do **not** diminish —
>    over a representative solve (≥15 s, reaching branched nodes; the first few
>    root-adjacent nodes fixpoint faster) rounds 1/2/3 each tighten ~18 bounds/call
>    and almost every such call runs the full 3 sweeps without hitting the fixpoint
>    (the McCormick envelope keeps tightening as the box shrinks). So "the LPs are
>    far too many" is the wrong diagnosis:
>    capping rounds/probes would loosen the per-node bound → **more** B&B nodes, a
>    bound-for-wall trade, not a free cut. The bound-neutral affordability levers on
>    the probe loop are already taken (fixed-column skip at `width <= min_width`,
>    warm-start basis threaded probe→probe via `_PersistentProbeLP`, equilibrate-
>    once-per-sweep, `build_incremental=False`).
>
> 2. **The per-probe cost is degenerate simplex *pivoting*, not factorization —
>    and factorization reuse was implemented and measured NET-NEUTRAL (a second
>    falsification).** The per-probe wall is *bimodal* (nvs05, ~635 probes/12 s):
>    p50 ≈ 3.5 ms, p90 ≈ 12.7 ms, p99 ≈ 20 ms — ~66 % are cheap warm re-solves and
>    the **~29 % expensive tail (≥5 ms) is 62 % of all probe wall**. The expensive
>    probes are warm-start *rejections*: OBBT applies its tightenings mid-sweep and
>    the objective flips each probe (`min x_i` → `min x_{i+1}`), so the threaded
>    previous basis is both primal-infeasible (box shrank) and dual-infeasible
>    (objective changed) → the warm dual and warm primal are both rejected and the
>    probe does a near-cold two-phase re-solve of **~220 primal pivots** (per-call
>    `DISCOPT_PROFILE`: `FtUpdate`/`AlphaFtran`/`PriceBtran` ≈ 220 each). Note the
>    `iters` field returned by `solve_lp_warm_csc_py` reports **0** for these
>    (the cold/primal-fallback path does not surface its pivot count) — which is why
>    an earlier reading mislabeled the probes "0-pivot"; the phase counters are the
>    ground truth. The LP is tiny (m ≈ 325, n ≈ 90, **nnz ≈ 1116, 0.8 % dense**), so
>    its sparse basis factorization is only ~0.4 ms/call, equilibration ~0.003 ms,
>    and Python↔Rust marshaling negligible — none is the bottleneck; the ~220
>    degenerate pivots are.
>
>    A bit-identical **factorization-reuse** engine change was built to test the
>    "re-factorization dominates" hypothesis: a persistent `discopt._rust.ProbeLp`
>    handle + `ProbeFactorCache` that reuses a *fresh* `factorize_sparse` across
>    probes sharing the sweep matrix (mirroring `PreparedDual::reoptimize`'s
>    factorization-clone), behind `DISCOPT_OBBT_FACTOR_REUSE` (default-OFF), with
>    cargo cached-vs-stateless differential tests. It was **cert-clean** (byte-
>    identical status + `node_count` + objective bits on the certifying panel
>    ex1224/st_e29/st_e36/fac2, each in a fresh process; 0 mismatches over a 609-probe
>    in-situ A/B) but **net-neutral (~1.0×)**: reusing a ~0.4 ms factorization can
>    not move a ~220-pivot cost. Per the `DISCOPT_CUT_INHERIT` lesson (sound ≠
>    helpful — a cert-clean but neutral change stays out, measurement recorded), the
>    change was **reverted**, not shipped.
>
> **Re-scope (the real lever):** cut the warm-start *rejection* tail — reduce the
> ~29 % of probes that fall to a ~220-pivot near-cold re-solve because box-tightening
> **and** objective-flip together invalidate the threaded basis. Directions (each a
> distinct, higher-risk LP/OBBT-algorithm change needing its own entry experiment +
> byte-identical panel, and a bounded per-probe pivot budget so the tail can never
> blow up): a probe/basis strategy that keeps a warm start feasible across the two
> simultaneous changes (e.g. re-optimize box changes with the dual from a *shared*
> per-sweep basis before swapping the objective, rather than threading probe→probe);
> or a bounded phase-1 from the node's LP-optimal basis. Factorization/scaling/
> marshaling are all measured-small and are **not** the lever. `clay0303hfsg` is a
> *different* bottleneck (0 per-node OBBT probes on the same panel — FBBT/JAX-bound
> per the issue's own split), so this lever is scoped to the OBBT-probe class only.
> Reproduction: `discopt_benchmarks/scripts/nvs05_obbt_probe_cost_measurement.py`;
> pinned by `python/tests/test_perf_723_obbt_probe_cost.py`.

> **Falsified (2026-07-18, #723 lever 3 re-scope — "cut the OBBT warm-start
> rejection tail with a better warm-start strategy").** Entry experiment run
> before any implementation (kill criterion set in advance: tail probe wall must
> drop ≥ 50 %): capture every real OBBT probe from an nvs05 solve (per-sweep
> matrix, per-probe objective/box/threaded basis), replay the captured stream
> offline under each candidate strategy, and compare wall + results against the
> baseline threading. Five measurements, one shared capture
> (`nvs05_obbt_warmstart_replay.py`):
>
> 1. **Two-stage** (dual re-opt of the box change under the *previous* objective,
>    then swap objectives from that primal-feasible basis — the direction the
>    block above proposed): **1.10× slower on the tail**, oracle-gated ≤ 5 %.
>    Diagnosis via the stage split: stage 1 (box change, same objective) costs
>    ~0.85 ms — the box change was never the expensive part. The **objective flip
>    itself** is: walking from `min x_i`'s optimal vertex to `min x_j`'s costs the
>    ~220 pivots regardless of how fresh the starting basis is.
> 2. **Self-warm floor** (re-solve each probe from its *own* optimal basis):
>    p50 ~0.86 ms even on the tail, 100 % optimal — the LP is EASY from the right
>    basis, ruling out degeneracy-stall; but the right basis isn't known ahead.
> 3. **Per-objective basis memory** (warm `min/max x_j` from the basis that solved
>    the *same objective* previously): the predicted per-probe win appears
>    (tail-hit p50 7.0 → 1.3 ms) but ~1/3 of hits fail to transfer — the envelope
>    matrix is rebuilt each round at the tightened box, and the coefficient drift
>    rejects the basis → cold fallback — and those failures carry the wall.
>    Aggregate: **1.02–1.07×**.
> 4. **Hybrid upper bound** (memory + threaded fallback on rejection, i.e. the
>    best an in-engine "try basis A, else basis B" could do, +0.3 ms detection):
>    **1.34–1.36× probe wall ≈ ~8 % of nvs05 solve wall. This is the ceiling of
>    the whole warm-start-strategy family** — below the ≥ 50 % kill criterion and
>    below the bar for a correctness-critical engine change.
> 5. **Solution-based probe filtering** (Gleixner/Berthold/Müller 2017 — skip a
>    probe when a previously returned optimal point sits at the probe's bound):
>    **unsound as naively implemented** — the returned vertex is feasible only to
>    ~1e-6·scale, and on nvs05's ~1e4-scale boxes that slack fakes witnesses; the
>    replay audit found **~45 % of "filterable" probes would actually have
>    tightened** (a silent bound loosening, i.e. a correctness bug had it
>    shipped). Even optimistically it caps at ~11 % of probe wall. A rigorous
>    variant would need exactly-verified witnesses (directed-rounding feasibility
>    check) and would keep only the sound remainder of that 11 %.
>
> **Standing conclusion for #723 lever 3 (three falsifications deep):** nvs05's
> per-node OBBT LP wall (~1/3 of solve wall) is genuine simplex pivot work whose
> per-probe floor (self-warm, ~1.3 ms mean incl. marshal) is ~3× below today's
> ~4.4 ms mean, but no implementable warm-start/filter strategy tested reaches
> that floor — the family caps at ~8 % of solve wall. Closing the remaining
> BARON/SCIP gap on this class is NOT a warm-start orchestration problem; it
> would need engine-level pivot-throughput work (pricing, degenerate-walk cost)
> or a fundamentally different bounding scheme for the dependent-variable class
> — both large, separate efforts with their own entry experiments. Lever 3 is
> closed as measured-and-bounded; the issue's remaining wall-time gap on nvs05
> is attributed, not mysterious. Reproduction:
> `discopt_benchmarks/scripts/nvs05_obbt_warmstart_replay.py`.

> **Falsified (2026-07-18, issue #721 — "piecewise-McCormick auto-trigger on
> wide-range cubic/monomial blocks certifies ex1252").** #707's re-scope (record
> above) pointed at a stronger cubic-block relaxation, with #721's most localized
> direction being auto-triggered piecewise McCormick on the wide flow factors
> (`x6,x7,x8 ∈ [0,2950]`), asserting the `x6²` secant is "the weakest single link".
> The entry experiment measured the reformed-ex1252 dual bound (with #707's reform
> applied) on the *actual* per-node engine (`MccormickLPRelaxer`) at the canonical
> loosest node (LINE1 fixed, OBBT-tightened, `x0=2, x3=1`). **The bound is pinned at
> `12658.06` across every available lever:**
>
> | lever at the loosest node | dual bound |
> |---|---|
> | baseline (standard McCormick) | 12658.06 |
> | subdivide `x6` / subdivide `x12` (halves) | 12658.06 (from the #707 probe: 12658.1 both) |
> | RLT cuts / level-1 RLT | 12658.06 |
> | PSD (moment) cuts | 12658.06 |
> | superposition cuts | 12658.06 [^superposition-inert] |
> | OBBT + optimum cutoff | 12658 (the #707 record) |
>
> [^superposition-inert]: Retracted as a measurement (#1035). The `superposition`
>     switch this row toggled had been inert since the #632 cutover —
>     `build_milp_relaxation` accepted the parameter and never read it — so this
>     arm re-measured the baseline rather than a superposition-cut relaxation. The
>     *conclusion* (the bound is pinned at 12658.06) stands on the other rows; the
>     superposition lever was never actually tested. #1035 deleted the switch and
>     the unreachable cut generator with it, so the arm is not re-runnable as
>     written.
>
> Two corrections to the issue's framing fall out. **(1) `x6` is not the lever, and
> neither is any flow.** At the *root* the flows are wide but the objective relaxes
> to 0 (indicators free); at any *binding* node OBBT has already narrowed
> `x6 → [1823,2950]` and `x12 → [116.7,175]`, so partitioning those narrow ranges is
> inert. "Wide-range" and "binding" never coincide, so direction #1 (piecewise on
> wide monomial factors) cannot bite on the real path. (A transient +27% signal from
> partitioning `x12` on the *AMP MILP* engine — `build_milp_relaxation`, SOS2
> partition binaries — proved to be a node-definition artifact: it appears only at a
> *looser* box where the MILP is free to re-choose the active line, and vanishes on
> the canonical box, where the AMP build is infeasible. It is not a cubic-block
> tightening.) **(2) The wall is the objective coupling, not the cubic rows.** The
> bound equals the objective's constant term `6329.03·x0·x3·x18 = 6329.03·2 =
> 12658.06` *exactly*, yet the relaxed `x15 = 12.44 ≠ 0`: the reformed
> `x15·(x0·x3·x18)` aux relaxes to its lower bound, so the `1800·x15` cost
> contributes 0 to the bound regardless of `x15`. The cubic cost rows #721 targets
> only *define* `x15`; tightening them cannot lift the bound while `x15`'s coupling
> into the objective is itself loose in-relaxation. No wide-range-monomial partition
> trigger shipped — it would be inert on the real path (and, keyed on range width,
> would select `x6`, the *most* inert flow), per the `DISCOPT_CUT_INHERIT` lesson
> (sound ≠ helpful). The real lever is the **objective coupling**, addressed next.
> Reproduction: `discopt_benchmarks/scripts/ex1252_piecewise_lever_probe.py`; pinned
> by `python/tests/test_ex1252_piecewise_lever.py`.

> **Implemented, default-OFF (2026-07-18, issue #721 — objective-coupling RLT,
> `DISCOPT_MULTILINEAR_COUPLING_RLT`).** Following the record above (the wall is the
> `x15·(x0·x3·x18)` coupling, not the cubic rows), the entry experiment measured
> `min x15` over the reformed loosest-node relaxation = **12.44** — the cubic/flow
> rows *do* force `x15` up; the bound sits at the objective constant `12658.06` only
> because the reformed `v_k = z_k·x15` big-M products decouple (with the reform's own
> expansion bits fractional in the LP, every `v_k` relaxes to 0). Since the objective
> is `12658.06 + 3600·x15` at the node, a valid coupling link makes `12658.06 +
> 3600·12.44 = 57435` a **sound** bound the current relaxation simply leaves on the
> table. The fix is RLT (issue direction #3): multiply each integer factor's exact
> bit-linking equality (`x_i = lo + Σ2^k e_k`) and each AND hull (`z ≤ b`,
> `z ≥ Σb−(n−1)`) by the non-negative continuous factor, tying `Σ2^k(e_k·c)` to
> `x_i·c` and `v = z·c` to the per-bit products. Both levels are needed — the AND-hull
> RLT alone does nothing (the leak is one level down, in the fractional expansion
> bits); adding the bit-linking RLT (where `x_i·c` is McCormick-exact once `x_i` is
> fixed) lifts the loosest-node bound to **57434.96**, matching the entry-experiment
> prediction to the penny. Sound throughout (RLT rows are products of valid
> identities/inequalities — never cut a feasible point; verified `bound ≤ opt` on
> every run). **Kept default-OFF, not graduated:** the flag-OFF path is
> byte-identical to #707 (same 90-column model, same 12658.06). The deterministic
> node-budget A/B settles net effect (equal node count removes the wall-clock
> nondeterminism that made the time-limited global dual erratic):
>
> | ex1252, ~400 B&B nodes | global dual | incumbent |
> |---|---|---|
> | flag OFF | 16071 | 143555 |
> | flag ON | 16304 (+1.4%) | **134471** (closer to opt 128894) |
>
> So the large *node-level* lift (4.5× at a line-selected node) translates to only
> **+1.4% on the global dual** — because the global dual is set by the *shallow*
> indicator nodes near the root (objective still relaxes to ~0 there), not the deep
> line-selected nodes the coupling RLT tightens. It does improve the *primal* side
> (a better incumbent). **ex1252 stays a hard bound gap** — this is a SOTA-frontier
> instance (SCIP does not certify it in 120 s either), and the coupling RLT does not
> lift the *global* dual materially above the #707 ~48k plateau within a practical
> budget. Net-positivity for graduation would need **deep-node gating** — apply the
> RLT only once the integer factors are fixed by branching (a per-node cut, not an
> upfront model transform), so shallow nodes don't pay for rows that cannot bite
> there — plus the CLAUDE.md §5 corpus differential panel. The lever is sound and a
> foundation; graduation is future work. `python/tests/test_ex1252_coupling_rlt.py`
> pins soundness + the node-level lift + the byte-identical OFF path.
> **Follow-up (same day):** the compounding probe
> (`discopt_benchmarks/scripts/ex1252_compounding_probe.py`) confirms the RLT
> *unlocks* the previously-falsified levers — `x6` subdivision now lifts a child
> bound 57435→62071 and OBBT pins `x12` exactly / caps `x15` at 30.89 within
> seconds (both provably inert pre-RLT) — and exposes an engine fragility
> (0.0/`numerical` fallback bounds on narrow boxes). Full anatomy + staged
> certification plan: `docs/dev/ex1252-certification-plan.md`.

> **Shipped + one falsification (2026-07-18, issue #741 — SGO constrained node
> tightening; certifies the ≥4-var signomial class).** The #736 blocker (bound
> −3e5-class vs opt on ex7_2_3 / ex3_1_2 while the incumbent is found) was
> reproduced on faithful reconstructions of the two probes (the container had no
> corpus access; ex3_1_2 ≡ Himmelblau/g04, ex7_2_3 ≡ HS106/g10 — coefficients
> from the literature, oracle values matching the issue table). Root-caused into
> THREE stacked losses, each measured at the root before its fix: (1) the
> single-supporting-hyperplane certification of the Lagrangian is arbitrarily
> loose when SLSQP stalls (ex3_1_2 root: relaxed opt −31499, corner certificate
> **−122428**, plain interval floor −32217 — the certificate, not the
> relaxation, was the binding loss); (2) child bounds were not monotone (a
> failed child solve reported −1.1e10 *below* its parent); (3) the DC secants
> over wide boxes need range reduction the tree wasn't doing (ex7_2_3 root
> relaxed opt 2100 vs opt 7049). Shipped, all certified-sound and inside the
> already-flagged `DISCOPT_SGO` path (issue #741, lever 1): iterated log-domain
> OBBT with the incumbent objective cut, every coordinate bound proven by the
> same Lagrangian-corner mechanism (never the subsolver's word) and backed off
> by a margin; a rigorous interval floor on the objective AND on the fitted
> Lagrangian; monotone parent-bound inheritance; certified infeasibility/cut
> pruning (empty-box ⇒ `status="infeasible"` + `gap_certified=True`);
> secant-gap-guided branching at the relaxation point; frozen-pack evaluation
> (the secant part is affine on a fixed box — precompute per node; ~13×
> node-rate). Measured: 4-var wide-box class instance **certifies** (1e-2 in
> 562 nodes / ~31 s, 1e-4 in ~1.9k nodes) where the pre-#741 path sits at
> **−327 vs opt 10.90** at the same node budget; ex3_1_2 tree bound
> −1.1e10 → **−30701** (opt −30665.5, rel gap 1.2e-3 @ 900 s); ex7_2_3 finds
> the optimal incumbent (was: none) and bound −3e5 → **+3233** (opt 7049) —
> converging but not certified at 1e-4 in-budget; certification of the 8-var
> 100×-range class needs LP-strength relaxation throughput (future work).
> **Falsified in passing (lever 3, "tighter Pminus"):** certifying tighter
> secant *argument* ranges (per-negative-monomial ξ-OBBT via the same
> `_cert_min_linear` machinery, incumbent cut included) returns ranges WIDER
> than box-implied on essentially every row of both probes (e.g. ex7_2_3 box
> ξ∈[9.21,16.12] → "cert" [7.91,17.35]; the single exception tightens one
> ex3_1_2 row by 0.06 log-units — noise) — the Lagrangian corner certificate
> is too weak in non-coordinate directions, so ξ-OBBT through this machinery
> is a dead end;
> a tighter Pminus needs either a certified LP dual or piecewise secants with
> branching on the argument, each its own entry experiment. Differential-bound
> + feasible-point-sampling + infeasibility-certificate regression tests:
> `python/tests/test_signomial_global.py` (§6 block). `DISCOPT_SGO` remains
> default-OFF — graduation (issue Task 3) needs the corpus panel.

> **Shipped (2026-07-18, issue #741 Task 2 — integer signomial MINLPs; admits
> the `cvxnonsep_nsig*` family).** Entry experiment on the in-repo
> `cvxnonsep_nsig30` (the corpus `.nl` is present; 15 continuous + 15 integer
> vars, one mixed-sign signomial row) found the constraint is
> `1 − 0.2·exp(a·u) ≤ 0` — a *single* negative monomial — so it is EXACTLY
> convex-representable: dividing by the positive `0.2·exp(a·u)` gives the
> log-linear halfspace `a·u ≥ log 5` (identical feasible set). A direct convex
> solve of the reformulated continuous relaxation returns **130.479973** =
> exactly the issue's stated continuous relaxation value, confirming the exact
> transform (Task 1 lever 2 / Lundell–Westerlund single-sign power transform) is
> the "tighter continuous relaxation prerequisite" Task 2 names — the DC secant
> of that same row bounds only ~55.9 at 30-dim. Shipped both pieces, all
> certified-sound and default-OFF behind `DISCOPT_SGO`: (1) `_exact_convex_pack`
> replaces every single-negative-monomial row with its exact convex posynomial
> form (its node relaxation is then exact, not a secant), keeping the untransformed
> body for genuine feasibility verification; (2) integer branching wrapping the
> continuous node relaxation — integer bound rounding (empty enclosed-integer
> range ⇒ certified integer-infeasible prune), most-fractional branch with an
> integer-domain-split fallback, and integer-feasible incumbent recovery (fix
> each integer to an enclosed integer, solve the continuous remainder, verify
> every true constraint). The node relaxation relaxes integers to the continuous
> box, so every bound is a valid lower bound on the integer optimum; a fully
> pruned tree returns a rigorous `status="infeasible"` certificate. Measured:
> small integer MINLPs and box-only integer programs certify to their
> brute-force optima in ≤6 nodes; `cvxnonsep_nsig30` is admitted and its
> integer-feasible incumbent reaches the exact oracle **130.62871264** (bound
> climbs to ~101 at ~140 nodes / 150 s — sound, converging; full certification of
> the 30-var box is the same wide-box corner-certification frontier as ex7_2_3,
> not in-budget). The exact transform additionally collapses the Task-1 4-var
> probe to a **root** certification (0 branch nodes). Classifier abstains on
> binary / 0-lb variables (the log lift needs `x > 0`). Regression tests (§8
> block, `python/tests/test_signomial_global.py`): exact-transform feasible-set
> equivalence + convexity, integer-MINLP certification vs brute force, integer
> node bound ≤ every integer-feasible point, certified integer infeasibility,
> box-only integer, flag-gated `Model.solve()`, and a slow `nsig30` admission
> probe. Reproduction: `discopt_benchmarks/scripts/sgo_741_tightening_probe.py
> nsig`. `DISCOPT_SGO` stays default-OFF (Task 3 graduation still pending the
> corpus panel).

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
## 10. Relaxation layer — sparse-bilinear RLT auto-gate (#727): the medium-pooling root-bound gap

> **Entry experiment (2026-07-18) — confirmed the framing; the fix is a
> structure-aware auto-gate, behind a flag (`DISCOPT_RLT_SPARSE_AUTO`, default
> OFF), pending the corpus graduation panel.**
>
> Context: issue #727 (SOTA wall-time gap on 70 medium MINLPs) attributes the
> pooling / bilinear-flow-network cluster (haverly, pooling_bental5stp,
> genpooling_lee2, gasprod_sarawak01, wastewater0*m*) to a **weak McCormick root
> bound**. The full MINLPLib snapshot is not in-repo, so the entry experiment uses
> a controlled proxy: `k` independent Haverly-I pooling blocks in one model (the
> canonical pooling nonconvexity and weak McCormick bound, with a *known* optimum
> `400·k` and `6·k` continuous vars / `4·k` bilinear terms — sparse: products grow
> linearly with variables).
>
> **Hypothesis:** RLT (build-time level-1 + per-node cut separation) closes the
> pooling root bound in seconds, but the default RLT auto policy gates on a raw
> *variable count* (`_AUTO_RLT_LEVEL1_MAX_VARS = 50`, `_AUTO_CUTS_MAX_VARS = 40`),
> which excludes medium pooling instances even though their RLT relaxation is small
> and root-closing. The variable count is a poor cost proxy.
>
> **Measurement (TL=30 s, warm; `rlt` control vs the k-block proxy):**
>
> | k | vars | default (`auto`) | `rlt=on` |
> |---|---|---|---|
> | 8 | 48 | **feasible**, bound 7022 (opt 3200), 533 nodes, TL | **optimal**, root, 1 node, 1.5 s |
> | 12 | 72 | **feasible**, bound 13650 (opt 4800), 591 nodes, TL | **optimal**, root, 1 node, 2.8 s |
> | 16 | 96 | feasible, bound 18787 | feasible, bound 8950 (RLT helps; TL-bound) |
>
> Isolating the levers: at k=8 (48 vars) the decisive combination is **build-time
> level-1 RLT** (which includes the Phase-2 quadratic-constraint RLT that multiplies
> the bilinear pool-quality equality by bound factors, and lifts the product
> columns) *plus* per-node RLT-cut separation; per-node separation alone (multiplying
> only *linear* constraints) does not close k=12. Both levers are gated by the raw
> caps, so medium pooling falls back to the loose McCormick bound and times out —
> exactly the #727 symptom.
>
> **Why the raw cap is the wrong proxy (the discriminator):** RLT cost is driven by
> the number of lifted product columns/rows, not the variable count. Measured
> `nbil/nv`:
>
> | model | nbil/nv | density `nbil/(n·(n−1)/2)` | RLT relaxation (cols / ub rows) |
> |---|---|---|---|
> | k-block pooling (sparse) | ≈ 0.67 (constant) | → 0 as n grows | k=24, n=144: 472 / 1786 (trivially solvable) |
> | dense box-QP | ≈ n/2 (grows) | 1.00 | grows ~n² |
>
> A sparse-bilinear pooling network keeps a small RLT relaxation past the raw cap;
> a dense QCQP grows its products quadratically and is correctly excluded. The
> variable-count cap (motivated by the dense casctanks blow-up, 500 vars → 359 s/node)
> conflates the two.
>
> **Fix:** `SolverTuning.rlt_sparse_auto` (`DISCOPT_RLT_SPARSE_AUTO`, default OFF)
> widens the build-time-level-1 and per-node-cut auto-gates to additionally admit a
> model whose *product-term count* ≤ `rlt_sparse_max_terms` (300; the lifted-column
> budget — sparse bilinear vs dense QCQP) **and** whose variable count ≤
> `rlt_sparse_max_vars` (200; a ceiling on the enlarged per-node re-solve cost).
> `solver._rlt_sparse_admit` implements the gate; flag OFF ⇒ byte-identical dispatch.
>
> **Follow-up measurement (2026-07-18) — structure alone is NOT sufficient; the
> productivity gate.** Surveyed the in-repo instances for an independent
> sparse-bilinear family to validate the class-level fix. Found the heat-exchanger
> network-synthesis family (`heatexch_gen{1,2,3}`: nonconvex, sparse bilinear —
> gen1 32 bilin / 112 vars, gen2 40 / 148) — *and RLT is net-**negative** there*:
>
> | instance | default | structure-only gate ON |
> |---|---|---|
> | heatexch_gen1 | bound 41621 (31 nodes) | **looser** 38183 (3 nodes) |
> | heatexch_gen2 | incumbent 824551 (feasible) | **incumbent lost** (`obj=None`) |
>
> Root-bound probe (single-node LP, RLT off vs on) explains it — the relative root
> gain `|b_rlt − b_noRLT| / (|b_noRLT| + 1)` splits the two families by **ten orders
> of magnitude**:
>
> | model | root LB no-RLT | root LB +RLT | relative root gain |
> |---|---|---|---|
> | pooling (kpool12) | −15000 | **−4800** (= optimum) | **0.68** |
> | heatexch_gen1 | 38183.53174599 | 38183.53174550 | 1.3e-11 |
> | heatexch_gen2 | 543496.0185139 | 543496.0185124 | 2.6e-12 |
>
> RLT helps *iff it closes / near-closes the root* (pooling: paid once, tree
> collapses); when the root stays open (heatexch) the heavier node LP only starves
> branching. This is the `DISCOPT_CUT_INHERIT` lesson again (sound ≠ helpful).
>
> **Refinement:** `rlt_sparse_root_probe` (`DISCOPT_RLT_SPARSE_ROOT_PROBE`, default ON
> when the widening is on) adds a **productivity** condition on top of structure — a
> bounded root probe (`solver._rlt_root_gain`, memoized once per solve) must show a
> relative root gain ≥ `rlt_sparse_min_root_gain` (`DISCOPT_RLT_SPARSE_MIN_ROOT_GAIN`,
> default 1e-2). Pooling (0.68) engages; heatexch (~1e-11) declines and falls back to
> the exact default path — no regression. On any probe failure the widening is
> declined (never regress). Verified: kpool8/12 still certify at the root; heatexch
> gen1/gen2 match their default runs bit-for-bit in behavior.
>
> **Soundness:** RLT is valid regardless of engagement (a constraint×bound-factor
> product is non-negative at every feasible point), so this only ever trades
> relaxation size for bound tightness — `incorrect_count` cannot change, and the
> downstream soundness guards (C-43 pool retry, the per-node unboundedness cross-check,
> the `gap_certified` guard) are untouched. Verified: flag ON, k=8/12 certify at the
> root at the true optimum with a valid dual bound; flag OFF, the smoke suite (664)
> and the k-block proxy are unchanged.
>
> **Kill criterion / graduation:** bound-changing (CLAUDE.md §5), so it stays
> default-OFF until the owner's corpus-wide differential panel passes **both** bars
> (cert-clean: `incorrect_count = 0`, no bound above its reference optimum, no
> certification regression; and net-positive on the pooling/bilinear cluster without
> a regression on the dense-QCQP class the raw cap was protecting). The flag is
> *killed* if the panel shows the widened gate is net-negative on any covered class
> (the `DISCOPT_CUT_INHERIT` lesson: sound ≠ helpful).
> Regression-pinned by `python/tests/test_rlt_sparse_auto.py`.

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

## 10. hda root throughput (#671 follow-up): where the root time actually goes

> **Diagnosis + one falsification (2026-07-18).** After #671 gave hda a sound,
> tight *root dual bound* (≈ −6.47e4), the natural next question was closing its
> optimality gap to opt −5964.53. Measurement shows the blocker is **not**
> relaxation strength — it is **root throughput**: hda explores **3 B&B nodes**
> and the root node consumes the *entire* budget (`root_time` ≈ wall at every
> `time_limit`), so the tree never branches and no primal incumbent is found
> (`objective=None`, gap undefined). A tighter relaxation is moot while the tree
> cannot move.
>
> Clean attribution (no cProfile overhead), `time_limit=40`, flag OFF:
> **wall 42.2 s, jax 6.9 s, python 35.3 s, rust 0.0 s**, nodes 3. The `python`
> bucket is dominated by *synchronous Rust-extension calls* (counted as Python):
>
> | cost centre | ~seconds | nature |
> |---|---|---|
> | Rust presolve (`run_root_presolve` → `PyModelRepr.presolve`) | ~10 | one-time |
> | McCormick LP/MILP solves (`solve_lp_warm_csc_py` ~2.8 s each ×4, `solve_milp_csc_py`) | ~13 | ill-conditioned relaxation (the #671 hard LP) |
> | FBBT (`reduce/fbbt`) | 5.1 | per fixpoint round |
> | JAX XLA compile of the relaxation/Jacobian evaluator | ~7 | one-time per solve |
> | relaxation build + convexity boxes (`build_uniform_relaxation`, `_build_convexity_box`/`eigvalsh`) | ~7 | — |
>
> `separate/*` cut timers are all **0.000** — the root never reaches cut
> separation. There is **no single wasteful Python hotspot**; the time is genuine
> presolve + slow LP solves on the near-singular McCormick relaxation + FBBT.
>
> **Falsified — dense→sparse Jacobian routing (the one clean candidate).**
> `evaluate_jacobian` routes to the sparse coloring path only above
> `_DENSE_JACOBIAN_COMPILE_LIMIT = 1e6` (m·n); hda is 718×722 = 5.18e5 < 1e6, so it
> takes the **dense** `jax.jacfwd` (722 JVPs) despite a **0.4 %-dense** Jacobian —
> `should_use_sparse` (density < 15 %, ≥50 vars) is plainly true. Hypothesis: a
> density-aware route to the sparse path (identical values ⇒ bound-neutral) cuts
> the ~12 s the profile attributed to `_evaluate_dense_jacobian`. **Entry
> experiment (force hda through sparse via a lowered limit): FALSIFIED.** Bound and
> node_count are **identical** (bound-neutral, as predicted) but wall time did
> **not** improve — **44.1 s (sparse) vs 42.3 s (dense)**, `jax_time` 6.6 s both.
> The profile's ~12 s was one-time XLA *compile* of the Jacobian program, which the
> sparse coloring evaluator also pays; the per-iterate dense eval itself is cheap
> at this size, and hda re-evaluates the Jacobian only a handful of times (FBBT's
> two-point linearity test). Do not re-try density-aware Jacobian routing as an
> hda root-speedup lever. Reproduction: `/tmp` experiment mirrored in the #671
> session; the routing gate lives at `python/discopt/_relax/nlp_evaluator.py:783`.
>
> **Re-scope.** Meaningful hda root speedup requires the *hard* levers, not a Python
> micro-fix: (a) faster/robust simplex on the ill-conditioned McCormick relaxation
> (the #671 factorization-hardening research — also the thing that makes the LP
> solves ~2.8 s each and forces the numerical-failure path), and/or (b) profiling
> and reducing the ~10 s Rust presolve. Both are Rust-engine efforts with their own
> bound-neutral gates; neither is a quick win. Recorded here so the next attempt
> starts from the measurement, not the guess.

> **Falsified (2026-07-18, issue #727 sub-cluster C — "sound FBBT/activity
> finite-bound derivation certifies the 1-node-stall unbounded-declared nonconvex
> NLPs demo7/chakra").** Hypothesis (from the #727 small-nonconvex diagnosis): these
> pure-continuous nonconvex NLPs stall at 1 node because they declare unbounded
> variable sides (`[0,inf]`), so the root McCormick relaxer is unbounded and cannot
> bound/branch; SCIP derives finite bounds by constraint/activity propagation, so
> discopt should too — derive the missing bounds (FBBT/activity) so the relaxer
> builds and spatial B&B engages, with a >50 % root-gap reduction on demo7 as the
> GO bar. **Entry experiment falsified BOTH prongs of the kill criterion.**
>
> 1. **The feasible region is genuinely unbounded — the constraints do NOT imply
>    the missing bounds.** demo7 (70 vars, quadratic obj + one convex-quadratic
>    equality) has 18 linear "slack" columns (x32–x43, x53, x54, …) that are
>    genuinely unbounded: `bootstrap_finite_bounds` (exact LP `max x_i` over the
>    full linear subsystem, incl. equalities) returns unbounded for them, and
>    demo7's only nonlinear constraint (constraint 0) does not contain them, so no
>    nonlinear propagation can bound them either. FBBT + linear-OBBT + nonlinear
>    tightening iterated to a fixpoint — even with the incumbent objective cutoff
>    `obj ≤ V` added as an explicit constraint — leaves them open. chakra (62 vars,
>    all-equality, `−Σ cᵢ xᵢ^0.1` **anti-coercive concave** objective, relaxer
>    `bound=None`) is worse: its objective *wants* the vars to grow, so no cutoff
>    bounds them, and every finite bound would have to flow through the fractional
>    -power equalities; linear bootstrap finitizes 0/60.
>
> 2. **Any finite cap on the genuinely-unbounded columns is demonstrably UNSOUND;
>    the sound sub-derivation that IS available gives 0 % gap reduction.** Measured
>    on demo7 (opt −1,589,042): capping the open columns at **1e6 yields dual bound
>    −1,568,959 — ABOVE the optimum, i.e. it CUTS the optimum (unsound)**; 1e8
>    happens to land sound (−1,589,053, 0.001 % gap) only because the largest
>    optimal slack value is x67 ≈ 1.65e6; 1e10+ fails numerically. There is no
>    principled sound cap — the working window is luck, exactly the arbitrary cap
>    the task forbids.
>
> **Genuine sound sub-finding (recorded for independent follow-up, NOT a #727
> fix).** `nonlinear_bound_tightening._flatten_sum` does not distribute `neg` over
> a nested sum, so `neg(Σ −aᵢ xᵢ² + bᵢ xᵢ)` (how the .nl reader renders demo7's
> constraint-0 convex quadratic) stays an opaque atomic term and the
> `SeparableQuadraticUpperBoundRule` never sees its squares. Teaching `_flatten_sum`
> to recurse through `UnaryOp("neg")` (sound, exact) lets that rule bound demo7's
> **nonlinear** vars from constraint 0 alone (x0≤12480, x1≤3531, x4≤14116 —
> independently verified NOT to cut the optimum: opt x0/x1/x4 = 1844/892/3794).
> But this ALONE moves the final #727 bound by **0 %** (stays −6.35e6, 299 % gap):
> the derived bounds are ~7× the optimum (loose McCormick envelope), and OBBT
> cannot sharpen them because it bails on the 18 still-open genuinely-unbounded
> slack columns. So the fix is *sound but not net-positive for this cluster*
> (the `DISCOPT_CUT_INHERIT` pattern); it may help other separable-convex-quadratic
> instances whose ONLY blocker is the nonlinear var (not unbounded slacks) — worth
> a scoped FBBT-completeness issue on its own merits, gated per §5, but it does not
> close sub-cluster C.
>
> **Re-scope.** demo7/chakra are not one lever and not a bounded FBBT fix. Closing
> them soundly needs either (a) a McCormick relaxer that handles genuinely-free
> (obj-coeff-0, nonlinear-free) columns as free LP columns instead of requiring a
> finite cap — a relaxer-internals effort whose payoff (demo7 certifies at 0.00 %
> once x0–x5 are additionally OBBT-tightened) is real but out of scope here — or
> (b) accepting these as genuinely-hard residuals. Recorded as such; no ship, no
> arbitrary cap. Reproduction: the `_flatten_sum` neg-distribution monkeypatch +
> `bootstrap_finite_bounds`/`tighten_nonlinear_bounds`/`MccormickLPRelaxer.solve_at_node`
> probes in the #727 sub-cluster-C session.

## 11. Root-relaxation fallback wall overrun (#832/#814): the residual is the Python DCP build, not the Rust LP

> **Diagnosis + falsification (2026-07-21, issue #832).** #832 was filed (as a
> #814 residual) claiming the `gastrans582` root-relaxation overrun is a Rust LP
> setup/factorization that ignores `SimplexOptions::deadline` before the first
> pivot. The §4 entry experiment on current `main` (post-#831) **falsified the
> premise.** Direct call `_root_relaxation_lower_bound(gastrans582_mild11,
> budget=3.0s)`: `bound=None  total_wall=15.46s  build_time=14.31s (3 builds)
> Rust_LP=1.15s → 5.2× overrun`. Every faulthandler sample lands in
> `build_milp_relaxation` (Python DCP convexity classification), and the Rust
> `DISCOPT_PROFILE` shows the basis factorize at **2.77 ms** (`DualPrepare`) with the
> pivot loop already deadline-polled (`DualPivotLoop` 1.07 s). Threading a deadline
> into the Rust LU would fix **0.0 s** of the 14.3 s — the #727 RLT trap
> (mechanism validated only against a mis-attributed proxy). Reproduction:
> `scratchpad/repro_832.py` (localization) and the base-build split showing the base
> build alone is ~10.5 s and un-skippable (feeds `_objective_bound_valid`),
> consistent across variants (mild11 10.34 s, cold13 10.10 s → the class).
>
> **Fix (Lever 1, measured not predicted).** #694 already deadlines the *separated*
> build but left its companion base build WHOLE. `DISCOPT_ROOT_BUILD_DEADLINE`
> (default off, §5 bound-changing) deadlines the base build too, making the whole
> fallback honor its grant: a truncated base build is a valid **weaker** relaxation
> (dropped rows only enlarge the feasible set) or trips the existing
> `_objective_bound_valid` gate to `None` (weaker, never falsified). gastrans582:
> **15.46 s → 3.56 s (5.2× → 1.2×)**, same `None` outcome — a pure wall win. A static
> size predictor was rejected as fragile (per-term build cost spans 185 µs–14 ms, so
> a term-count threshold would falsely skip cheap-but-numerous-term instances); the
> wall-clock deadline measures reality instead. Graduation gated on the flag-ON-vs-OFF
> differential panel (`generality_sweep` arm `root_build_deadline`).

## 12. #818 B1 throughput: FBBT structural-match caching is sound-but-not-helpful (falsified 2026-07-21)

> **Entry experiment + falsification (#818).** cProfile of the cleanest B1
> micro-benchmarks (ex9_1_1 8.3s, ball_mk2_10 4.8s, ex14_2_7 9.9s, nvs21 1.7s — SCIP
> ~noise) decomposed the slowdown: node NLP-solve cost is **node-count-bound** (a full
> NLP per node where SCIP uses an LP; ball_mk2_10 opens **2046 nodes** on 10 vars),
> per-node nonlinear FBBT is 20-27% (`tighten_nonlinear_bounds`), root OBBT dominates
> a few (ex14_2_7 4.9s / 104 probes), and JAX dispatch is ~3s on ex14_2_7.
>
> **Hypothesis (bound-neutral lever): cache the per-node FBBT structural matching.**
> The rule matchers re-walk the immutable constraint DAG every node (`_constant_value`
> **1.03M calls**, `scalar_flat_index` 474k, `match`/`walk`) over a structure that is
> invariant across B&B nodes — only the box changes. Memoized the bound-INDEPENDENT
> matchers (`_match_scaled_linear_var/_square_var/_affine_var`, `scalar_flat_index`) in
> a per-metadata `_match_cache` (attached via `object.__setattr__` on the frozen
> dataclass; invalidated with the metadata on constraint-DAG change; `is`-identity
> checked against `id()` reuse).
>
> **Result: bound-neutral but NOT net-positive → not shipped.** Node_count and
> objective were **exactly** unchanged on a 10-instance panel (ex9_1_1 233, ball_mk2
> 1023, nvs02 337, … bit-identical) — the cache is sound. But it moved wall by **<3%**:
> `tighten_nonlinear_bounds` cumtime 2.174s→2.046s (~0.13s), and the decorator overhead
> (`wrapper` 0.078s) plus a counterproductive `scalar_flat_index` memo (0.093→0.113s —
> the function is too cheap to cache) nearly cancel it. FBBT matching is simply **not
> the wall bottleneck** — the node NLP-solve count is. This is the `DISCOPT_CUT_INHERIT`
> pattern (sound ≠ helpful; §5): recorded and reverted rather than shipped as complexity
> for no gain. Baseline/verify harness: `scratchpad/fbbt_baseline.json`,
> `scratchpad/prof_818.py`.
>
> **Standing conclusion for #818 B1.** The lever is fewer/cheaper node relaxations —
> the in-progress convex **LP-OA branch-and-cut kernel (#799/#804)**, not per-node
> Python-overhead trimming. Root-OBBT budgeting (ex14_2_7 class) is a separate
> bound-CHANGING candidate (needs the §5 panel).

## 13. #917 budget policy: the #844 reserve is forfeited, and the bound merge that would spend it is dead (falsified 2026-07-31)

> **The defect.** `Model.solve` deducts `0.35 * time_limit` from the caller's budget
> for every model the #844 no-incumbent fallback *could* serve, and spends it in one
> case only — the primary returned nothing. A primary that *does* find an incumbent
> and then hits its reduced deadline forfeits the slice: nobody spends it, and the
> caller gets `time_limit` at 65% of the limit they stated.
>
> **Entry measurement** (19 in-scope instances across both in-repo corpora, 60 s
> budget, isolated subprocesses; `scratchpad/issue917_entry_panel_T60.json`): 15
> certify inside the reduced 39 s budget, **1** (nvs24) is the no-incumbent case the
> reserve exists for, and **3** (nvs17/nvs19/nvs23) stop at ~39 s holding an
> incumbent with 21 s discarded. `nvs18` certifies at **38.9 s** of the 39 s primary
> budget — 0.1 s of margin — so the reserve is a latent *certification* regression on
> the family, not only lost wall.
>
> **Falsified: candidate 1, "spend the residual on the fallback and keep the tighter
> dual bound."** Issue #917 proposed it with an explicit kill criterion ("if the
> merged bound is no tighter on a corpus panel, this buys nothing"). The criterion is
> met on every instance of the class — the fallback's bound is 5-50x *looser*:
>
> | instance | primary bound (39 s) | reserve bound (21 s) | tighter? |
> |---|---|---|---|
> | nvs17 | -1105.89 | -5838.02 | no |
> | nvs19 | -1104.24 | -16377.19 | no |
> | nvs23 | -1130.70 | -57328.53 | no |
>
> Its primal is worse too on every one (-1068.8 vs -1100.4, -931.2 vs -1097.6, none
> vs -1124.8). This is the documented ~250x root looseness of the McCormick
> relaxation on this family (`docs/dev/lp-node-primal-quality.md`), so the merge was
> **recorded and not shipped**. The budget belongs to the primary.
>
> **Shipped instead: candidate 2, the incumbent-conditional budget extension**
> (`_extend_budget_for_incumbent`, `DISCOPT_LP_SPATIAL_RESERVE_EXTENSION`, default ON
> after its panel). The search reclaims the reserve at its reduced deadline and only
> while it already holds an incumbent — the one state in which the #844 fallback
> provably has nothing to contribute — so #844 keeps its exact budgets and ordering,
> and a path with no reclaim point (the #764 native kernel is one uninterruptible
> Rust call) simply keeps today's reduced budget rather than silently losing the
> fallback's reserve. Panel: 133 cells over a uniform budget grid,
> `cert_regressions=0 lost_incumbents=0 unsound=0`; nvs18@45 s goes uncertified 0/3 →
> **certified 3/3** and nvs18@30 s closes 92% of its dual gap (-783.8528 → -778.8359
> against an incumbent of -778.4), both with zero spread over 3 reps.
>
> **Standing negative result.** On nvs17/nvs19/nvs23 the reclaimed time is
> **neutral**: nvs17 at 60 s goes from 27 to 7391 nodes and the dual bound does not
> move a digit. That is the NLP-per-node bound freeze on dense integer-bilinear
> models — the pathology the `lp_spatial=True` engine exists for — and no budget
> policy can fix it. Anyone measuring a bound improvement on that family from *more
> time* is measuring noise.
>
> **Method note (cost me a wrong conclusion once).** A wall-limited search is **not**
> node-reproducible: nvs13 at a 6 s budget gives 454 vs 475 nodes, different bounds,
> on two runs of the identical build. Single-run OFF/ON diffs on time-limited cells
> are noise — the first 1-rep panel flagged four "looser bound" cells, and every one
> was a cell where the extension never fired, i.e. two runs of identical code. Score
> only cells where the mechanism fired, and repeat them.

## 14. #917 follow-up: the warm pure-LP node path drops its caller's deadline (mechanism built, panel FAILED 2026-07-31)

> **The defect.** `MilpRelaxationModel.solve` takes a `time_limit`, and its DEFAULT
> `backend="simplex"` pure-LP fast path dropped it: `_solve_lp_warm` /
> `_solve_lp_warm_equilibrated` / `solve_lp_warm_std` took no deadline and
> `lp_bindings.rs` hardcoded `SimplexOptions { deadline: None }` — while the MILP
> route (`solve_milp_csc_py(time_limit_s=…)`) wired it up and the dual/primal pivot
> loops already poll it every 256 pivots. ~13 call sites in `_relax/mccormick_lp.py`
> plus `lp_spatial_bb.py` and `integer_ratio.py` compute a per-LP budget and pass it
> here, so the drop was general.
>
> **Evidence** (`scratchpad/nvs24_arm.py`, `scratchpad/nvs24_profile_evidence.txt`).
> nvs24 at a 3.9 s budget: ~53 s (13.5x), reproducibly, of which
> `separate/univariate_square` is 47.9 s — ONE call,
> `solve(time_limit=0.202)` → `_solve_lp_warm` → 47.03 s, a single `DualPivotLoop`
> **59 494 degenerate dual pivots** deep with `DualBlandActivations=0`. Hard cliff:
> m=1324 rows solves in 1.1 s, m=1334 takes 45 s.
>
> **Built:** `time_limit_s` on `solve_lp_warm_csc_py` → `SimplexOptions.deadline`,
> threaded back up through `solve_lp_warm_std` → `_solve_lp_warm` →
> `MilpRelaxationModel.solve`, which now spends ONE shared budget across its warm /
> equilibrated / cold attempts instead of handing each a fresh copy. Behind
> `DISCOPT_LP_WARM_DEADLINE`.
>
> **Panel FAILED → stays OFF** (66 in-repo instances, 15 s budget, OFF/ON interleaved).
> *Cert-clean*: **no** — 1 certification regression (cvxnonsep_psig40r) and 2 bounds
> lost outright (bchoco08 1.0 → None, contvar 171244.81 → None), 2 looser vs 3 tighter.
> *Net-positive*: **no** — total overrun 79.2 s → 85.2 s (flat; the OFF arm alone
> ranged 78–176 s across runs, so this metric is noisy).
>
> **The blocker is precise.** Honouring the deadline costs a *bound*, not just time.
> `_time_limit_result` does return a sound Neumaier–Shcherbina floor when the yielded
> simplex leaves a usable dual, but it never reaches the reported bound: the consumer
> in `_relax/mccormick_lp.py:758,903` adopts a node bound only from a
> `status == "optimal"` result. Teaching it to accept a rigorous safe bound from a
> non-optimal node would change the verdict — sound by weak duality, but it touches
> the bound-adoption logic certification rests on, so: own change, own panel.
>
> **Two falsifications of my own work, recorded per §11.**
>
> 1. *The recovery cascade.* The first cut had no timeout guard, so a deadline exit
>    fell into the equilibrated retry and then the ~170x-slower cold `solve_milp`.
>    That **doubled** the corpus overrun the change exists to remove (175.6 s →
>    310.3 s), all of it from one instance: heatexch_gen3 25.7 s → **254.5 s**
>    (1.7x → 17.0x). With the guard, 30.7 s. A deadline is not a numerical failure and
>    must not trigger the recovery meant for ill-conditioning.
> 2. *The scoring hole.* The panel first compared bound quality only where BOTH arms
>    were finite, so `1.0 → None` read as clean and it printed CERT_CLEAN=True twice
>    before the hole was found. `lost_bound`/`gained_bound` exist because of that. An
>    instrument that cannot see the most severe outcome in its own domain is worse
>    than no instrument.
>
> **Standing conclusion.** The overrun is real and the plumbing to fix it now exists
> and is tested; what blocks it is that the solver has no way to bank a dual bound
> from an LP it cut short. Fix that first, then re-run this panel.
>
> **Separate, still open:** the dual-simplex degeneracy stall itself — 59 494
> degenerate pivots with Bland's rule never activating.

### 14a. Update: the blocker is cleared — the panel is now cert-clean (2026-07-31)

> §14 closed with "the solver has no way to bank a dual bound from an LP it cut
> short". That is now fixed, and the fix was **not** where the Python-side reasoning
> put it. Three links had to change, and only the third mattered:
>
> 1. `MilpRelaxationModel._stash_deadline_bound` banks the NS floor of a yielded LP
>    (ungated — #517's flag guards a *numerically broken* dual, a different concern).
> 2. `_relax/mccormick_lp.py` adopts a finite bound from a `status == "time_limit"`
>    node result, the same shape as the existing #517 branch beside it.
> 3. **The actual blocker, in Rust.** `primal.rs` exported its `y = B⁻ᵀc_B` dual
>    candidate on a `Numerical` exit but kept an EMPTY dual on `IterLimit`, on the
>    stated grounds that there is "no usable factorization to btran against". That is
>    true of the `failed()` path but not of an iteration/deadline exit, which stops a
>    healthy solve early with its factorization intact. Measured directly:
>    `solve_lp_warm_csc_py(..., time_limit_s=0.0)` returned `dual=[]`, so steps 1-2
>    were banking nothing and both showed no effect at all when tested.
>
> With the dual exported (`is_ok()` guard retained), the two lost bounds return:
> bchoco08 `None` -> 1.0000000000006 and contvar `None` -> 98924.53 (looser than the
> OFF arm's 171244.81, but finite and sound). Panel: **CERT_CLEAN=True** —
> `cert_regressions=0  lost_incumbents=0  lost_bound=0  unsound=0`, bounds 3 tighter /
> 2 looser.
>
> **Still not graduated.** Net-positive is not demonstrated: total overrun 82.1 s ->
> 70.9 s (-14%) sits inside the metric's own noise — three runs of the OFF arm alone
> gave 82.1 / 79.2 / 175.6 s — and cells over budget went 15 -> 18. The corpus average
> is diluted by the ~50 instances that finish far inside 15 s and are bit-identical
> either way. What would settle it: a load-gated, multi-rep panel over the instances
> where the deadline actually binds.
>
> **Method note.** The Rust change touches an exit path on the DEFAULT route, so it
> was checked for bound-neutrality independently: 13 certifying instances byte-identical
> in node_count/objective/bound/status/gap_certified with the flag OFF
> (`scratchpad/issue917_neutrality.py`), plus `cargo test -p discopt-core` 575 passed.
> The new dual is consumed only behind flags (`_certify` reads `safe_bound` only from
> an `optimal` result), which is why the default path does not move.

### 14b. #928: the deadline exit is bound-preserving; graduation still fails, and the residual moved seams (2026-08-09)

> §14a left two things open: the banked floor's *quality* (a deadline exit that
> banks -141697 when -64473 is seconds away) and the net-positive bar. The first
> is now fixed; the second still fails, and the measurement says the remaining
> overrun is no longer this seam's.
>
> **Root cause of the weak floor, measured** (`scratchpad/issue928_{capture_lps,
> replay}.py`, replaying the real hda separated-relaxation node LP): on a deadline
> the warm dual loop returned `None`, discarding its dual-feasible basis; the cold
> primal fallback then ran on a spent budget and its `IterLimit` dual export is
> **identically zero through phase 1**, so the recovered NS floor was exactly
> `g(y=0)` — the trivial box bound, -141697.4335, at 15/40/75% deadline fractions
> alike, gap to the -64473.4 LP optimum never shrinking with budget.
>
> **The fix** (`SimplexOptions::bank_deadline_duals`, set ONLY by
> `solve_lp_warm_csc_py` when a `time_limit` is passed — the MILP driver's own
> deadline route keeps `false`, so the default B&B pivot path is untouched;
> verified BOUND_NEUTRAL flag-OFF on 13 certifying instances vs the merge-base,
> `scratchpad/issue928_neutrality.py`, marker-gated per §8): the dual loop's
> deadline exit returns `IterLimit` carrying its current `y = B⁻ᵀc_B` (NS value =
> the monotone best-so-far dual objective); near-zero-pivot breakdowns retry in
> place (refactorize + exact recompute, the loop's existing soundness anchor,
> capped at 5) instead of abandoning the loop; the last exact refresh's duals
> survive a breakdown → cold-fallback → deadline-cut sequence; and a COLD solve
> carrying a finite deadline starts the dual simplex from the sign-matched slack
> basis when dual-feasible (`_dual_start_slack_basis`; `PreparedDual::prepare`
> re-verifies the precondition), because the primal proves no usable floor
> mid-run. Measured on the hda LP: floor -141697 -> -118439/-118783 at every
> binding deadline (~30% of the gap closed, monotone in budget); whole-solve hda
> @10 s ON: bound -141697 -> -123462 in 3/3 reps, wall 11.2-11.8 s.
>
> **Seven §5 panel runs, ALL cert-clean** — 0 cert_regressions / lost_incumbents
> / lost_bound / unsound in every run; §14a's bound-losing trade is gone (contvar
> is now often TIGHTER than OFF; tspn12 and tls2 gained incumbents). Net-positive
> is where it dies (`discopt_benchmarks/results/issue928_*.json`):
>
> | panel | budget | ON−OFF overrun |
> |---|---|---|
> | full corpus (66), 1 run | 15 s | 73.8 -> 31.6 s (**-57%**) |
> | binding subset (19), 3 reps | 15 s | -2.5 / 0.0 / -8.2 s |
> | binding subset (19), 3 reps | 20 s | **+325.4 / +68.5 / +12.7 s** |
>
> The sign flips with budget — not "measurably helpful broadly" (the
> `DISCOPT_CUT_INHERIT` rule) — so **the flag stays OFF**.
>
> **Where the ON overrun actually lives** (`scratchpad/issue928_{contvar_probe,
> blowup_hunt}.py`): not in this seam. Zero per-LP deadline violations across all
> probes. contvar ON spends **2.8 s in 30 relaxation solves** against a 27-33 s
> wall; OFF spends 6.3 s in 20 solves for a ~22 s wall. The budget-honoring LPs
> return sooner, the enclosing separation loop fits ~10 more rounds, and each
> round's non-LP cost (~1.1 s `build_uniform_relaxation` etc.) is not clamped by
> the round's grant. On top of that, downstream phases with coarse global-deadline
> compliance produce rare severe modes in BOTH arms: ON contvar 500.6 s and
> bchoco08 80.9 s; OFF heatexch_gen3 200.5 s in the same rep set (the same
> pre-existing class §14 recorded as 78-176 s OFF-arm noise). The next actionable
> seam is therefore **caller-side round-budget accounting** (clamp a separation
> round's build cost against the phase grant, and the downstream phases' deadline
> compliance), not further work in the LP engine.

> §13 graduated `DISCOPT_LP_SPATIAL_RESERVE_EXTENSION` default-ON on a 133-cell panel
> whose net-positive case was three cells: nvs18@45 s (uncertified 0/3 → **certified
> 3/3**), nvs18@30 s (92% of its dual gap closed, zero spread over 3 reps) and
> nvs13@9 s. **That case no longer exists**, and the cause is not a flaw in the
> measurement but a change underneath it: #919 re-graduated the #764 native spatial
> kernel to default-ON, and the kernel certifies all three instances in 2.4–5.4 s —
> long before the reduced budget can bind.
>
> | cell | pre-#919 (the case) | post-#919 (both arms) |
> |---|---|---|
> | nvs18@45 s | uncertified 0/3 → certified 3/3 | certifies in 5.4 s |
> | nvs18@30 s | bound −783.85 → −778.84 | certifies in 5.2 s |
> | nvs13@9 s | certified 1/3 | certifies in 2.4 s |
>
> Re-running the identical panel on the new base: **133 cells, extension fired 0
> times** (`issue917_reserve_extension_panel_postkernel.json`). The flag is inert, so
> ON vs OFF is a no-op — and a bound-changing flag with no measurable benefit defaults
> OFF (§5 bar 2). Flipped back, graduation retracted in the flag docstring.
>
> **The defect is not fixed — it moved.** The native kernel is one uninterruptible
> Rust call that never enters the Python node loops where
> `_extend_budget_for_incumbent` lives, so a kernel-routed solve still forfeits the
> reserve. Measured on the new base, nvs17 at a 60 s budget spends **39.4 s in both
> arms** — the 0.65×T signature exactly, extension never firing. Independently
> corroborated: 8 of the 19 in-scope instances route to the kernel.
>
> **What closing it requires.** The budget decision has to be made up front *inside*
> the kernel — pass the reserve into the Rust spatial driver and have it extend its own
> deadline once when it holds an incumbent, mirroring `_extend_budget_for_incumbent` in
> the Python loops. Issue #917 anticipated exactly this for this path. Until then the
> flag can only act on models the kernel declines.
>
> **Standing lesson (§11).** A graduation panel certifies a flag *against the tree it
> ran on*. This one was invalidated four days later by an unrelated default flip in
> another PR, with no code change to the flag itself. When a panel's net-positive case
> narrows to a handful of instances, it is worth asking which *other* default decides
> whether those instances even reach the mechanism.

### 13b. Re-graduation: the kernel-side reclaim point passes on a far broader panel (2026-08-01)

> §13a retracted the reserve extension because #919 moved the affected instances onto
> the native spatial kernel, where the Python-side guard cannot reach — leaving the
> defect intact (nvs17: 39.4 s of a 60 s budget in both arms) and the flag inert
> (133 cells, 0 firings). The kernel now has its own reclaim point:
> `SpatialTreeConfig.incumbent_time_extension` pushes the tree's deadline out once, and
> only while `incumbent.is_some()`.
>
> **Entry experiment first, with a kill criterion**: is the forfeited 35% worth
> anything on this path? Tested without writing code, by disabling the reserve so the
> primary got the full budget. It survived easily — nvs17 −1149.20 → −1100.40,
> nvs19 −4017.37 → −2303.40, nvs23 −23735.23 → −18951.65.
>
> **Panel** (same 133 cells, OFF/ON interleaved). Extension fired in **27**.
> *Cert-clean*: `cert_regressions=0 lost_incumbents=0 lost_bound=0 looser_bound=0
> worse_objective=0 unsound=0`; no ON-arm bound above a reference optimum.
> *Net-positive*: **all 27 firing cells improve the bound**, six from no bound at all.
>
> | cell | OFF | ON |
> |---|---|---|
> | nvs17@60 s | −1136.00 | **−1100.40** (onto its incumbent) |
> | nvs19@60 s | −3838.44 | **−2289.47** (40% tighter) |
> | nvs23@60 s | −23757.40 | **−19262.40** (19% tighter) |
> | nvs24@60 s | −138238.27 | **−111833.39** (19% tighter) |
> | nvs24@20 s | none | **−174436.22** |
>
> Budget utilisation 0.176 → 0.241. Cost: six overshoots OFF did not have, all at the
> two smallest budgets and all 6–8% (the post-loop tail, now measured against the full
> limit instead of 0.65×T). **Flag graduated default-ON**, `=0` opt-out retained.
>
> **Instrument note (§6), the second such this session.** The first run of this panel
> reported `extension_fired: 0` while the A/B plainly showed the mechanism working:
> `budget/incumbent_extension_s` was set only in the Python loops, so the panel was
> blind on exactly the path being added. That run was killed rather than interpreted,
> and `SpatialTreeResult.incumbent_extension_s` now carries the reclaimed seconds
> through to `solver_stats`. Separately, this panel's scorer gained the
> `lost_bound`/`gained_bound` check that the sibling lp-warm-deadline panel needed —
> a finite bound going to `None` was invisible to it too. Both panels now score it.

### 14c. #966: the caller-side seams measured, the severe mode caught in flight, and an eager-Hessian fallback falsified (2026-08-09)

> #928's residual (recorded in §14b, on `claude/issue-928-as1vvj` until that
> branch merges) named two caller-side deficiencies. Both are now measured on
> this tree and fixed behind flags (`scratchpad/issue966_phase_probe.py`; all
> probes print executed-comparison counts per §6).
>
> **Seam 1 — a round's grant clamps only its LPs.** A `solve_at_node` round
> granted 2.0 s runs 5.2–5.8 s in BOTH `DISCOPT_LP_WARM_DEADLINE` arms on
> contvar @ 20 s — ~3.2 s of it the cold `build_uniform_relaxation`, spent
> after the admission check, unclamped (6/6 reps). The internal node deadline
> is anchored AFTER the build, so the build also restarts the round's clock.
> Fix (`DISCOPT_NODE_ROUND_BUDGET`, default OFF, §5): the spatial node loops
> pass the global deadline into `solve_at_node` as an absolute
> `round_deadline` (clamps the build via the #694 anytime mechanism AND caps
> the internal anchor), and the round admission check declines a round whose
> remaining grant cannot cover the relaxer's measured cold-build EMA
> (`expected_build_cost()`). The serial loop previously had NO past-deadline
> skip at all — it gets one under the flag.
>
> **Seam 2 — the 200–500 s severe modes are one uninterruptible XLA compile.**
> Caught in flight with `faulthandler.dump_traceback_later` (§10):
> heatexch_gen3 @ 20 s ran 162.5 s with the stack inside
> `_solve_root_node_multistart → solve_nlp(pounce) → evaluate_hessian_values →
> sparse_hess_values → jit_batch_hvp` and XLA's own slow-compile alarm
> reporting the compile at **124 s**. The F4 gate exists for exactly this risk
> but the per-node NLP entries (root multistart on the no-relaxer class,
> strided node NLP, JAX-callback batch POUNCE) bypass it.
>
> **Falsified (§4, kill criterion hit): the eager fallback.** Hypothesis: an
> uncompiled `jax.disable_jit()` Hessian evaluation bounds the phase at usable
> per-call cost. Measured (`scratchpad/issue966_eager_hessian_entry.py`):
> heatexch_gen3 eager walls 62.7 / 11.7 / 10.4 s, contvar 29.7 / 7.7 / 8.3 s —
> the steady state alone dwarfs a per-iteration budget, so the eager route is
> dead. A compile can neither be truncated nor cheaply avoided; **entry
> refusal is the whole treatment** (the F4 rationale). Fix
> (`DISCOPT_HESS_COMPILE_GATE`, default OFF, §5): `_hess_compile_refuses`
> declines a NONCONVEX per-node NLP entry when the conservative first-compile
> estimate exceeds the remaining budget, checked at the loop sites where
> `_model_is_convex` is authoritative (the multistart chain does not thread
> convexity down; on the convex path the NLP is the bound producer and is
> never refused — rule 1).
>
> **A/B with both flags ON** (3 reps × {contvar, heatexch_gen3, bchoco08} ×
> both #928 arms @ 20 s): every wall in **[20.1, 22.1] s** — the baseline's
> worst same-container mode was 162.5 s and §14b records 500.6 s. No bound
> lost (heatexch_gen3 `None` in both regimes; bchoco08 identical; contvar OFF
> 183430.5 vs 183632.0 baseline — weaker by declined rounds, as the mechanism
> predicts, and never above). The gate's refusal was confirmed non-vacuous
> (2 firings logged on a heatexch_gen3 run, §6).
>
> **Honest residuals.** (1) The one-time ROOT probe build (grant 2.0 s, wall
> ~5.5 s on contvar) is deliberately NOT clamped: it feeds the
> `_probe_useful` decision and the incremental structure every later round
> reuses; truncating it can drop the relaxer entirely. The overrun is one
> bounded in-flight op (#654 policy). (2) The compile estimate is
> measured-unpredictable (1–186 s, R² ≈ 0, see `estimate_hessian_compile_s`),
> so an entry admitted with remaining budget above the risk floor can still
> overrun — the flag kills the observed late-entry severe modes, not the
> early-entry gamble the F4 floor deliberately accepts. (3) On THIS tree the
> loop fits few rounds (contvar: 1 probe round per run); the "more rounds fit"
> amplification §14b measured needs the #928 branch's banked dual floor, so
> the corpus-wide differential panel for these two flags is coupled to the
> #928 graduation panel re-run (issue #966 item 3).

### 14b. #928 + #966 coupled graduation panel: FAILED, and the loss is an *interaction* (2026-08-09)

> §14a left `DISCOPT_LP_WARM_DEADLINE` cert-clean but not net-positive, with a named
> next step: "a load-gated, multi-rep panel over the instances where the deadline
> actually binds." That panel has now run, and it grew a second question first.
>
> **Why the three flags were scored as one change.** #928's net-positive failure had a
> *measured* cause, not a mysterious one: the LP layer honours its grant, but the
> enclosing separated-relaxation round did not, so budget-honouring LPs merely let the
> loop fit more unclamped rounds and the ON arm's wall went **up** (ON−OFF +325.4 /
> +68.5 / +12.7 s at a 20 s budget). #966 closed that seam behind
> `DISCOPT_NODE_ROUND_BUDGET` (a `round_deadline` threaded into `solve_at_node`, min-
> combined with the build deadline) and `DISCOPT_HESS_COMPILE_GATE` (refuse to *start*
> a per-node NLP whose first-time sparse-Hessian XLA compile — measured 124 s on
> heatexch_gen3 against a 20 s budget — cannot fit; a compile is uninterruptible, so
> entry refusal is the whole treatment). Graduating #928 without #966 re-creates the
> known regression, so the panel has three arms rather than two:
> `base` (all OFF) / `seam` (#966 only) / `cand` (all three), interleaved per instance,
> every flag named explicitly in every arm so no cell can inherit a default.
>
> **Panel**: 19 binding instances, 20 s budget, 3 reps, `wtpanel` built from
> `origin/main` and gated by four §8 markers per cell (`discopt.__file__`,
> `_extend_budget_for_incumbent`, `_dual_start_slack_basis`, `round_deadline` in
> `solve_at_node.__code__.co_varnames`). `COMPARISONS_EXECUTED=19` each rep;
> `loadavg` recorded into every artifact.
> Raw: `discopt_benchmarks/results/issue966_coupled_binding20_rep{1,2,3}.json`.
>
> **The wall regression is gone.** The entry question passes cleanly and reproducibly:
>
> | pair | rep1 | rep2 | rep3 | mean ± sd |
> |---|---|---|---|---|
> | cand − base overrun | −2.6 s | −2.7 s | −2.7 s | **−2.7 ± 0.0 s** |
> | seam − base overrun | −6.0 s | −3.6 s | −5.2 s | **−4.9 ± 1.0 s** |
>
> The sign flip is attributable to #966; #928 gives back ~2.3 s of #966's gain.
>
> **`CERT_CLEAN=False` in 3/3 reps, on exactly one item**: contvar's bound goes
> `183632.766 → None` in the `cand` arm — not looser, *absent*. Nothing is unsound
> (`unsound=[]`, `incumbent_verification_failed=[]`, no cert regressions, no lost
> incumbents in any rep), so this is a claim-quality failure, not a soundness one.
> **Neither flag set graduates.**
>
> **The 3-arm panel could not attribute it, so a 4th arm was run.** `cand` loses the
> bound and `seam` keeps it, which attributes the loss to #928 *given the seam* — not
> to #928 by itself. The panel has no `warm`-only arm, so one was added
> (`discopt_benchmarks/scripts/issue928_contvar_attribution_probe.py`, 2 reps,
> `CELLS_EXECUTED=16`, bit-identical across reps):
>
> | contvar @20 s | nodes | status | bound | incumbent |
> |---|---|---|---|---|
> | base | 7 | time_limit | 183632.766 | none |
> | warm (#928 alone) | 3 | **feasible** | 98924.530 (looser, sound) | **813745.125** |
> | seam (#966 alone) | 7 | time_limit | 183632.766 | none |
> | cand (all three) | **287** | time_limit | **none** | none |
>
> **#928 alone does not destroy the bound — it trades bound quality for an incumbent**
> (the 98924.53 figure is exactly the one §14a reported). The destruction is an
> **interaction**: with the round budget also clamping, the node count explodes 7 → 287
> and no node ever certifies anything, so the tree spends the whole budget on cheap
> uncertified nodes. The suspect is the pair "LP yields on its deadline" ×  "round
> yields on *its* deadline" compounding into a node result that carries no adoptable
> bound at all; that is the thing to fix before either flag is re-panelled.
>
> **A second cost, from #966 alone, not surfaced in PR #968**: casctanks
> `2.9098 → −56.5001` in the `seam` arm, identical in all three reps and in both probe
> reps (and −57.2/−60.1/−60.7 with all three on). Sound (looser) and therefore
> invisible to `cert_clean`, but a dual bound crossing from +2.9 to −56.5 is a
> collapse, not a drift — the round budget is declining the rounds that were producing
> casctanks' bound. #928 alone costs it only 2.9098 → 2.4598.
>
> **Verdict.** `seam` (#966 alone) *is* cert-clean in 3/3 reps and is the only arm that
> buys wall time (−4.9 ± 1.0 s), but it fails the net-positive bar on the other half of
> the metric set: its bound ledger over three reps is 4 looser (casctanks every rep,
> tls2 ×2, nvs05, tspn10) against 3 tiny non-reproducible tighter entries, and node
> counts rise (1329 → 1811). This is the `DISCOPT_CUT_INHERIT` shape again — sound, and
> genuinely better on the metric it was built for, while paying for it in the metric
> the solver's product actually is. **All three flags stay default-OFF.**

### 14b-qual. Retraction of §14b's soundness evidence — the oracle was 1/19 (2026-08-09)

> CLAUDE.md §11: a measurement that contradicts a claim I already published gets
> retracted in writing before anything else proceeds. This qualifies §14b above.
>
> §14b asserts "**Nothing is unsound** (`unsound=[]`, `incumbent_verification_failed=[]`,
> …)". The `unsound=[]` half of that was **not** a measurement over 19 instances. The
> panel's `reference_optimum()` read `python/tests/_optima.py` behind a bare
> `except Exception: return None` — §7's exact failure mode — and that module records
> 27 curated instances. A counted control over the panel's own 19 names
> (`old_oracle_covered`) returns **one**: `clay0303hfsg`. Every other instance's oracle
> arm was skipped, and the panel printed `unsound: []` regardless. The other soundness
> checks in the same function *did* fire on all 19 (bound-crosses-incumbent,
> incumbent verification), so §14b's soundness conclusion was never empty — but its
> bound-vs-oracle component rested on one instance's three arms per rep (9 comparisons
> over the three reps, counted) and was reported as though it covered all 19.
>
> **The claim survives re-measurement, and is now stated at its real strength.** The
> three saved artifacts were re-scored against `minlplib.solu` — the library-wide
> oracle — without re-running anything, since the cells already carry every arm's
> bound (`--rescore`, and the standalone
> `discopt_benchmarks/scripts/issue966_rescore_against_solu.py`):
>
> | | merged §14b | re-scored |
> |---|---|---|
> | instances with an oracle | 1 / 19 | **16 / 19** |
> | bound-vs-oracle comparisons | 3 per rep, 9 total | **44 per rep, 132 over 3 reps** |
> | violations | 0 | **0** |
>
> Tightest margin over all 132: `syn05hfsg`, −1.99e-09 — inside the 1e-4 relative
> tolerance and attributable to LP arithmetic, not a crossed bound. `bchoco06`,
> `bchoco07` and `bchoco08` are named in neither oracle and remain **uncovered**; they
> are reported, never counted as clean. §14b's verdict (all three flags default-OFF)
> is unchanged — that verdict turned on `cert_clean` and the bound ledger, not on this
> line.
>
> **Instrument fixed, not just the sentence.** `reference_optimum()` now consults
> `minlplib.solu` first and falls back to `_optima`; only `KeyError` (the one genuine
> "no oracle" outcome) is swallowed, so a missing `.solu` or a bad import crashes.
> `soundness()` returns `oracle_comparisons_executed` and `instances_without_oracle`,
> both printed, and the panel **exits non-zero** when it made zero oracle comparisons
> (§6) — the state that produced this retraction can no longer report success.

### 14d. #928: the interaction is a round that throws away the floor it holds (2026-08-11)

> §14b's verdict named the residual precisely — "the pair *LP yields on its deadline*
> x *round yields on its deadline* compounding into a node result that carries no
> adoptable bound at all; that is the thing to fix before either flag is
> re-panelled." It is fixed, and the mechanism turned out to be simpler and more
> general than "an interaction of two clamps".
>
> **The corpus cell does not travel, so the mechanism was probed directly.** contvar
> @ 20 s is wall-clock-shaped: on the container this work ran in, all four arms of
> the §14b attribution probe return the same bound (171244.81, 3 nodes) and the 287-node
> collapse does not reproduce at all. Chasing the cell would have measured the
> container. So instead: hand `solve_at_node` a `round_deadline` that is already
> spent — the state #966's clamp produces once the cold build has eaten the round's
> grant — and compare against an unclamped control, over the same 19 binding
> instances, in both `DISCOPT_LP_WARM_DEADLINE` arms
> (`scratchpad/issue928_round_cut_short_entry.py`, 114 counted cells).
>
> **16 of 114 cells return NO bound** where the control certifies one — bchoco06/07/08
> (spent and nearly-spent alike), hda, tls2 — and they return it as `uncertified`, not
> `time_limit`. **Both flag arms are affected identically**, so the loss is not the
> interaction of the two clamps: it is the *round* clamp alone, and the LP clamp only
> ever made it more reachable. The truncated build leaves a **0-row** relaxation which
> solves to LP optimality and which every certification route then declines (no NS safe
> bound; the conditioning/free-column guards refuse the vertex).
>
> **And the floor was in hand the whole time** (`scratchpad/issue928_floor_inventory.py`,
> 10 counted rounds). Every lost cell's truncated relaxation carries a valid finite
> `_objective_floor`:
>
> | instance | rows built | `_objective_bound_valid` | floor | unclamped control |
> |---|---|---|---|---|
> | bchoco06 | 0 / 134 | True | **-1.0** | -0.9999918 |
> | bchoco07 | 0 / 153 | True | **-1.0** | -1.0000000 |
> | bchoco08 | 0 / 190 | True | **-1.0** | -1.0000000 |
> | tls2 | 0 / 24 | True | **0.0** | 0.0 |
> | hda | 0 / 718 | True | **-172201.82** | -64675.25 |
>
> On four of the five the discarded floor is the control bound to within rounding. The
> fix is therefore not a new bound source — it is *reporting the one already computed*.
>
> **Shipped (both inside the opt-in flags' blast radius, no new flag).**
> 1. `_solve_at_node_impl` reports that box-interval floor whenever the round was cut
>    short (`_build_truncated`, or an LP that exited `time_limit`) and no route
>    certified anything, and takes `max(banked deadline dual, box floor)` on a deadline
>    exit — the two are not ordered a priori, since §14a measured the hda node LP
>    banking exactly `g(0)`, the box floor itself. Sound: the floor comes from the cost
>    columns of the same column box the LP is solved over, independent of which rows
>    were emitted, so it is below the node's true optimum however truncated the build;
>    `_objective_bound_valid` gates out the un-relaxable and the #248 garbage-wide cases
>    (a garbage floor would anchor the tree on a bound that never certifies — the
>    "sound but harmful" trade §14b measured).
> 2. The #966 round-admission check no longer declines the **ROOT** round. A branched
>    node always carries its parent's bound, so declining forgoes tightening only; the
>    root has no parent, and declining it leaves the tree with no bound source at all —
>    which is exactly the shape of §14b's contvar collapse (7 → 287 nodes, nothing ever
>    certified, `bound=None`). This is rule 1 of `_fb_stop`, applied where it was
>    missing: never skip while no valid bound is in hand.
>
> **Verification.** Post-fix re-run of the entry experiment: `LOST_BOUND_CELLS` 16 → 0,
> with a counted bound-vs-control control on every cell. Default-path
> bound-neutrality with all three flags OFF: **13/13 certifying instances byte-identical**
> in `node_count`/`objective`/`bound` against the pre-fix tree, marker-gated in both
> directions (`scratchpad/issue928_round_floor_neutrality.py` — the marker is
> `_cut_short_floor` in `_solve_at_node_impl.__code__.co_varnames`). Five mechanism
> tests in `python/tests/test_928_round_cut_short_floor.py` fail on the pre-fix tree
> and pass after; a sixth pins the default-path decline that must NOT change.
>
> **Oracle caveat, stated up front (§14b-qual's rule).** `minlplib.solu` is not
> reachable from this container — the network policy denies minlplib.org — so the
> curated `known_optima.toml` covers **1 of the 19** binding instances. The re-panel
> below therefore adds a **cross-arm primal ceiling**: every arm's verified incumbent
> bounds the optimum from above, so the best incumbent found by any arm in any rep is a
> sound ceiling for every other arm's bound on that instance. Both coverages are
> counted and printed, and a run making zero ceiling comparisons exits non-zero.
>
> **Re-panel: the wall verdict flips clean, the bound ledger flips clean, and the one
> residual is #966's — the flags still do NOT graduate.** 19 binding instances, 20 s,
> 3 reps, three arms interleaved per instance
> (`discopt_benchmarks/scripts/issue928_round_floor_panel.py`; raw
> `discopt_benchmarks/results/issue928_round_floor_binding20.json`;
> `COMPARISONS_EXECUTED=57`, `CEILING_COMPARISONS_EXECUTED=63`).
>
> | metric (cand vs base) | rep1 | rep2 | rep3 |
> |---|---|---|---|
> | overrun delta | **−94.4 s** | **−313.5 s** | **−16.7 s** |
> | total overrun base → cand | 169.4 → 75.0 | 367.4 → 53.9 | 67.2 → 50.5 |
> | bounds tighter / looser | 8 / 1 | 7 / 2 | 8 / 1 |
> | bounds lost / gained | 0 / 1 | 0 / 0 | 0 / 0 |
> | nodes base → cand | 501 → 823 | 615 → 877 | 537 → 851 |
>
> * **Net-positive: passes.** The delta is negative in 3/3 reps (mean −141.5, sd 153.9 —
>   large because rep2's base arm hit a 313 s severe mode, not because the sign is in
>   doubt) where §14b measured **+325.4/+68.5/+12.7**. The sign flip is gone.
> * **The bound ledger is now positive, not merely non-negative.** `lost_bound` is empty
>   in 3/3 — §14b's contvar `183632.766 → None` does not recur — and rep1 *gains* one
>   (clay0303hfsg `None → −1.23e-05`). The big movers are exactly the cut-short class:
>   hda `−2.07e13 → −124296.9`, tspn12 `183.33 → 192.92`, beuster `6352.06 → 17698.96`,
>   heatexch_gen2 `555767.8 → 578091.9`. Against 1–2 looser cells per rep, all tiny
>   (tspn05 178.1165 → 177.8928, tls2 2.4487 → 2.1000).
> * **Soundness: 0 violations**, over 63 counted bound-vs-ceiling comparisons plus the
>   2–3 curated-oracle comparisons per rep, plus the bound-crosses-incumbent and
>   incumbent-verification arms, which fire on every cell. Coverage is reported, not
>   assumed.
> * **`CERT_CLEAN=False` in reps 2 and 3, on one item: tspn12's incumbent.** The base
>   arm finds 282.24 (bound 183.33); seam and cand find none (bound 192.92). **This is
>   the seam's, not #928's**: `seam vs base` shows the same loss (rep3 adds tls2) while
>   `cand vs seam` loses **no** incumbent in any rep. The round budget declines the
>   rounds whose search trajectory happened to land that incumbent — the same shape as
>   §14b's casctanks bound collapse, one level over in the metric set.
>
> **Verdict: all three flags stay default-OFF.** Bar 1 is a hard gate with no slack and
> it fails, even though the failing item is a primal one owned by #966 and is paid for
> by a materially tighter dual bound in the same cells. The thing this issue names —
> the callee dropping the deadline, and then the round throwing away the floor it
> holds — is fixed and measured; what remains is #966's round-admission policy losing
> a primal trajectory, which belongs on that issue.

### 14e. #966: a short-granted round YIELDS instead of skipping — §14b's bound ledger root-caused and fixed; graduation still fails (2026-08-11)

> §14b's coupled panel failed on one thing: the `seam` arm (#966's two flags) bought
> wall time (−4.9 ± 1.0 s, cert-clean 3/3) and paid in the metric the solver's product
> actually is — 4 looser bounds against 3 non-reproducible tighter, node counts
> 1329 → 1811, casctanks collapsing `2.9098 → −56.5001` in every rep. Its closing
> sentence named the fix: *"a round that yields on its deadline must still be able to
> bank a bound, the way §14a taught an LP cut short to export its dual."* That is what
> this entry implements — after two hypotheses died on the way.
>
> **Base note (2026-08-11, after the merge with §14d).** Everything measured below
> was run on `aca13dd`, i.e. BEFORE §14d's cut-short floor and root-round exemption
> landed. The two changes are complementary — §14d makes a cut-short round *report*
> the box floor it already holds, this entry makes a short-granted round *run* in
> the first place (banking an LP point as well as a bound), and the merged code
> yields only on non-root nodes so §14d's rule-1 root exemption governs both. The
> panel numbers below therefore understate the merged tree, and the §5 verdict for
> the merged code needs its own panel run: see the residual note at the end.
>
> **§14d does not subsume this entry, measured on the merged base.** Re-running the
> A/B there (nvs05 @ 20 s, forced-admission regime) with §14d's cut-short floor
> present: base 3.5425 / incumbent 8.7320 / 39 nodes, **skip 0.6323 / 6.3870 / 77**,
> **yield 3.5067 / 8.7320 / 59** (38 yields counted). The floor rescues a round that
> RUNS and is cut short; it can do nothing for a round that never runs, which is why
> the skip arm still loses the bound and the incumbent on the merged tree.
> Flag-OFF bound-neutrality was re-verified against the merged base as well
> (13/13 byte-identical, markers both ways).

> **FALSIFIED (H1): the cheap tier is the incremental fast path.** A round the grant
> cannot afford could fall back to `_try_incremental_node` (patch + warm start, ~ms).
> Measured (`scratchpad/issue966_inc_scope.py`, 5/5 instances the flag hurt):
> `IncrementalMcCormickLP` is **out of scope on every one** — nvs05, tspn10, casctanks,
> contvar, tls2 all construct with `_inc = None`, and the whole-solve probes see
> `fast_path_hits = 0`. There is no incremental tier on this class.
>
> **CONFIRMED (H2): the #694 anytime build already is the cheap tier**
> (`scratchpad/issue966_truncated_floor.py`, 2 reps/instance). A build whose deadline
> is already spent emits zero constraint rows but still linearizes the objective, so it
> returns a valid weaker relaxation *and* the rigorous box-interval `_objective_floor`:
>
> | instance | full build | truncated build | full round | truncated-round bound |
> |---|---|---|---|---|
> | contvar | 0.39–1.83 s | **0.008 s** | 19.6–23.9 s | 87754.998 (full: 164927.75) |
> | casctanks | 0.06–0.16 s | **0.025 s** | 0.16–0.27 s | 0.0 (full: *no bound at all*) |
> | tspn10 | 0.18–0.91 s | **0.068 s** | 2.2–16.0 s | 0.0 (full: 161.16) |
> | nvs05 | 0.74 s | **0.002 s** | 0.93 s | 0.674 (full: −323.05) |
>
> Truncation is effective at the *build* and nowhere else, so a yield must skip the
> separation chain too (tspn10's truncated-build round still cost 1.85–1.94 s with
> separation on).
>
> **What the flag's skip actually cost** (`scratchpad/issue966_yield_vs_decline.py`,
> nvs05 @ 20 s, every round forced through the admission branch): a skipped round banks
> no bound **and no LP point**, and the point is what the spatial brancher and the
> primal heuristics consume:
>
> | arm | bound | incumbent | nodes | wall |
> |---|---|---|---|---|
> | base (flag off) | 3.5139 | 8.7320 | 29–39 | 20.9 s |
> | every round SKIPPED | 0.6842 | **523.69** | **1** | **4.1 s** (16 s of budget unused) |
> | every round YIELDED | 1.3529 | 8.7320 | 45 | 21.0 s |
>
> **The fix** (`DISCOPT_NODE_ROUND_BUDGET`, still default OFF). A round whose grant
> cannot cover the measured cold-build EMA now runs with `yield_round=True` —
> separation off, build truncated at the grant — instead of being skipped. Sound by the
> same argument as the truncation: dropping cuts and rows only enlarges the relaxed
> set. Shipped-code A/B (`scratchpad/issue966_yield_fix_ab.py`, same forced regime):
>
> | nvs05 @ 20 s | bound | incumbent | nodes |
> |---|---|---|---|
> | base | 3.5139 | 8.7320 | 39 |
> | skip (pre-fix) | 0.6323 | 6.9358 | 35 |
> | yield (shipped) | **3.5061** | **8.7320** | 59 |
>
> Second half: a yielded round that banks *nothing* must not leave its node holding the
> failure sentinel. The slot carries `_INFEASIBILITY_SENTINEL` when the per-node NLP
> failed and the LP round is what replaces it; leaving it hands the node to the Rust
> tree's bound-cut — a fathom with no proof, which `_nonrigorous_sentinel_fathom` turns
> into a *discarded global dual bound* (§14b's contvar `183632.766 → None` has exactly
> this shape). `_yield_keeps_node_open` imports `-inf` instead: the node stays open,
> floored at its proved parent bound, and nothing is decertified.
>
> **Flag-OFF bound-neutrality** (§5 regime 1): 13 certifying instances,
> `node_count`/`objective`/`bound`/`status`/`gap_certified` byte-identical against the
> pre-change tree, markers asserted in both directions (§8) —
> `scratchpad/issue966_neutrality.py`, `BOUND_NEUTRAL=True  COMPARED=13`.
>
> **Coupled panel re-run**, same script and instance list as §14b (19 binding
> instances, 20 s, 3 arms interleaved per instance, 2 reps;
> `discopt_benchmarks/results/issue966_yield_binding20_rep{1,2}.json`):
>
> | pair | rep1 | rep2 |
> |---|---|---|
> | total overrun, base | 104.9 s | 300.8 s |
> | seam − base overrun | **−60.6 s** | **−261.7 s** |
> | cand − base overrun | −53.1 s | −269.3 s |
> | seam bound ledger vs base | 6 tighter / 1 looser | 6 tighter / 1 looser |
> | cand bound ledger vs base | 11 tighter / 2 looser | 10 tighter / 2 looser |
> | `CERT_CLEAN` (cand vs base) | **True** | **False** (tspn12 lost incumbent) |
>
> §14b's two named casualties are gone: casctanks is *tighter* in both flag arms in
> both reps (no collapse), and contvar keeps a finite bound in `cand` (180782.86 →
> 171244.81, sound) instead of losing it. The severe modes the base arm still hits are
> what the overrun deltas are made of — heatexch_gen3 base 210.1 s (10.5×) vs seam
> 21.8 s in rep2, bchoco08 base 86.2 s (4.31×) vs 21.0 s, tspn12 base 57.8 s (2.89×)
> vs 20.8 s in rep1 — each with an equal or *tighter* flag-arm bound.
>
> **Verdict: the flags stay default-OFF.** §14b's failure mode is fixed (the bound
> ledger is now positive and the wall win is large and reproducible), but §5 bar 1 is
> per-run and rep2's graduation pair is not cert-clean: tspn12 loses its incumbent
> (rep1 lost tspn10's in the `seam` arm). Lost incumbents are the residual, and they
> are not reproducible rep-to-rep — which is itself the signal that the *anytime*
> character of a yielded round is now trading primal luck, not bound quality.
>
> **Two honest limits on this measurement, both environmental.** (1) It ran in a
> container where POUNCE lacks the tape NLP evaluator, so every cell used the JAX
> evaluator; absolute walls and even base bounds differ from §14b's machine
> (casctanks base is −90.1786 here vs 2.9098 there), so this is the same panel on a
> different machine, not a reproduction of §14b's cells. (2) `minlplib.solu` is not
> reachable here, so oracle coverage was **1/19 instances (3 comparisons)** — the very
> state §14b-qual retracted a claim over. It is declared explicitly
> (`DISCOPT_MINLPLIB_SOLU=none`, recorded as `solu_oracle` in every artifact) rather
> than swallowed, and the panel's other soundness checks (bound-crosses-incumbent,
> incumbent verification) fired on all 19 × 3 cells with `unsound=[]`. **No soundness
> claim over the corpus may rest on these two runs**; re-run on the benchmark machine
> with the .solu oracle before graduating anything.
>
> **Residual, and how it meets §14d.** §14d's own verdict ends "what remains is #966's
> round-admission policy losing a primal trajectory, which belongs on that issue" —
> the same lost-incumbent item this panel's rep2 failed on (tspn12), from the other
> side. Yield mode is the candidate treatment for exactly that residual: an admission
> policy that runs a weaker round instead of discarding it keeps the primal
> trajectory alive (nvs05's incumbent survives at its base value where a skip lost it
> 8.73 → 523.69). Whether it *does* on the corpus is unmeasured on the merged base —
> both changes were panelled separately, each against `aca13dd`. The graduation panel
> owed to #966 item 3 must now be run once, on the merged tree, on the benchmark
> machine.

## 15. #956 envelope outward rounding: the defect is real, but it is NOT what drives `n_undecided` (falsified 2026-08-08)

> **The defect (confirmed).** The McCormick row generators in
> `bnb/mccormick_patch.rs` did no outward rounding, so an envelope row could cut a
> point EXACTLY on the graph of the term it relaxes: the rhs is a cancelling
> combination of box-endpoint quantities (`slope*x0 - f(x0)`) while the auxiliary
> bound is an independent rounding of the same quantity. Entry experiment, sweeping
> all four generators over ten orders of magnitude and evaluating `(x, f(x))`:
>
> | family | worst residual | vs 1 ulp of the row magnitude |
> |---|---|---|
> | cubic monomial | 6.4e+1 | ~0.8 |
> | affine square | 1.3e-1 | ~1.0 |
> | affine-form product | 2.0e-3 | ~1.0 |
> | square monomial | 1.2e-4 | ~0.9 |
> | bilinear | 9.5e-7 | ~0.4 |
>
> Kill criterion was "worst < 1e-9" (ordinary LP tolerance); it missed by nine orders
> of magnitude. A cubic lift needs a box of only `~5e3` to cut its own graph by
> `6.1e-5`, and the resulting node LP has no feasible point at all
> (`corner_pinned_cubic_node_lp_is_solvable` pins this).
>
> **FALSIFIED: this is not the mechanism behind #956's headline measurement.** The
> issue reports `nvs20: n_undecided=2056 over 4192 kernel nodes` (49%) and proposes
> `n_undecided` as the net-positive metric. It does not move:
>
> | arm | undecided / nodes | fraction | bound |
> |---|---|---|---|
> | guard ON | 407 / 885 | 46.0 % | 225.342 |
> | guard OFF (legacy) | 413 / 904 | 45.7 % | 224.858 |
>
> Sweeping the guard size settles it: at 1e6 ulp (2.2e-10 relative) the rate is
> *unchanged* (431/860); only at 1e10 ulp — a **2.2e-6 relative** relaxation, a
> million times larger than any envelope rounding can explain — does it collapse to
> 1/1420, and then the bound degrades 225.34 → 209.55. Two further causes were tested
> and excluded: propagation (43 % undecided with `run_propagation=False`) and OBBT
> (`run_obbt` defaults false on this path, so it never ran). A conflict-localizing
> probe — re-solving each undecided node LP with only the rows relaxed, then only the
> column bounds relaxed — leaves ~2/3 of them `Numerical` even at 1e-6 relative.
>
> **Standing conclusion.** `nvs20`'s undecided nodes are a **simplex robustness**
> problem on a 712x912 node LP, not an envelope-rounding problem. The lever for that
> counter is in `lp/simplex` (phase-1 Farkas certification and conditioning on the
> assembled node LP), not in the relaxation. #956's fix stands on its own invariant —
> a relaxation must contain the graph of what it relaxes — and must not be credited
> with, or judged by, the `n_undecided` counter.
>
> **Panel (CLAUDE.md §5).** Corpus-wide, 119 in-repo instances, ON vs OFF: only **21
> ever reach the native kernel**, the sole code path the flag exists on. On those 21
> at a 30 s budget the panel is **cert-clean** — 0 incorrect, 0 bounds above a
> reference optimum, 0 certification regressions, 0 objective drift, flag firing on
> 17/21 rows — with 15 of 21 bound deltas at `1e-13`–`1e-15` relative (the guard's
> designed cost) and the remaining 6 mixed in direction under a wall-clock budget.
> **Not net-positive**: `n_undecided` 1636 → 1559 (−4.7 %), node count +0.1 %, bounds
> better on 2 / worse on 4. The 5 s corpus arm flagged 2 certification regressions and
> 4 objective drifts; an **ON-vs-ON control** (identical configuration, run twice)
> reproduced the same two regressions and a *larger* nvs20 spread (10.5 vs 7.4
> absolute), so those flags are wall-clock noise, not flag effects. Three of the six
> flagged instances never call the kernel in either arm and so cannot be affected at
> all.
>
> **RETRACTION (2026-08-09): the flag ships default-OFF, and the claim that it is
> merely "neutral" was wrong.** An earlier revision of this section argued for
> default-ON on the grounds that the guard repairs an invariant and was only
> bound-neutral. A better measurement — the interleaved-replicate method this repo
> already validated for exactly this situation (`solver.py`, #902: `max_nodes` is
> unusable because `node_limit` routes the kernel back to the Python path, so the
> arms would compare OFF against OFF) — shows the guard is **harmful**, not neutral:
>
> | instance | ON bound (3 reps) | OFF bound (3 reps) | verdict |
> |---|---|---|---|
> | nvs17 | -1333.79 ± 4.8 | -1308.60 ± 1.2 | REGRESSION 3/3 |
> | nvs19 | -5841.67 ± 130 | -5679.50 ± 30 | REGRESSION 3/3 |
> | nvs20 | 225.342 ± 0 | 225.428 ± 0 | REGRESSION 3/3 |
> | nvs23 | -25667.3 ± 25 | -25570.6 ± 9.3 | REGRESSION 3/3 |
> | nvs24 | -171599 ± 600 | -170776 ± 100 | REGRESSION 3/3 |
> | tanksize | 1.25788 ± 9.4e-5 | 1.25823 ± 2.1e-4 | REGRESSION 3/3 |
>
> 6/6 decisive instances regress in 3/3 replicates, outside the replicate spread;
> `nvs20` has **zero** spread in both arms, so it is deterministic, not noise. And on
> `ex1252` under the #707 reform flags — *the configuration #956 was filed about* —
> the guard makes the counter the issue proposed measurably WORSE:
>
> | arm | undecided | fraction | nodes | bound |
> |---|---|---|---|---|
> | ON | 5222 / 6388 | **81.7 %** | 6388 | 12658 |
> | OFF | 8554 / 15456 | 55.3 % | 15456 | 20520 |
>
> **Mechanism, and it is not what the issue assumed.** It is not build cost — the
> guarded relaxation build is if anything faster (median 6.09/6.15 ms ON vs
> 6.18/7.00 ms OFF on nvs20, interleaved). It is not root-relaxation weakening on
> most instances either: root bounds move only ~6e-14 relative on nvs17/19/23/24
> (nvs20 is the exception at 1.8e-6, because the guard is sized relative to the ROW
> and nvs20's rows are ~1e10 against an objective of ~87). What happens is that a
> uniformly *looser* relaxation prunes slightly less at every node, the trees
> diverge, and under a fixed wall-clock budget the bound lands lower — which is why
> the direction is systematic rather than 50/50. On ex1252 the loosening also makes
> node LPs *more* degenerate, so a larger share come back undecided. That inverts
> the issue's premise: those LPs are not unsolvable because rows cut their own
> graph, they are unsolvable because they are ill-conditioned, and loosening them
> makes conditioning worse.
>
> So this is the `DISCOPT_CUT_INHERIT` case exactly as §5 describes it — sound but
> harmful — and the flag stays **OFF**, with the measurement recorded.
> `DISCOPT_ENVELOPE_OUTWARD_ROUND=1` opts in. With it off, `outward_slack` is
> identically `0.0` and `widen` is the identity, so all three engines are bit-for-bit
> the pre-#956 arithmetic; the invariant tests opt in explicitly rather than
> asserting a configuration that does not ship.
>
> **What would graduate it.** Not a bigger guard — a *cheaper* invariant. The guard
> is currently sized by the row's whole term magnitude; sizing it by the actual
> cancellation in each rhs (which is what genuinely needs repair) would be far
> smaller on well-scaled rows and might cost no bound at all. Alternatively, repair
> only the rows that demonstrably cut their own graph rather than all of them.
>
> **The Python twin is now closed too (2026-08-09).** `uniform_relax._emit_1d` /
> `_emit_mccormick` and `incremental_mccormick`'s generators carry the identical
> guard, behind the same flag. The constraint that shaped it: `_validate` compares
> the two engines through `_rowset`, which rounds each rhs absolutely to 6 decimals,
> so a guard applied to one and not the other — or computed from different
> intermediates — silently drops the incremental fast path to `ok=False`. The guard
> is therefore a pure function of quantities all three engines hold, computed once in
> `_relax/outward_rounding.py` and mirrored term-for-term in Rust. Verified: monomial
> and bilinear rows are BIT-IDENTICAL across the two Python engines, and the
> pre-existing affine-square divergence is unchanged (1265 mismatches, worst |delta|
> 1.133e-01, identical with the guard on and off). The invariant now holds on the
> Python generators: 1 950 000 comparisons, worst residual -3.3e-15, previously
> +1.3e+5.
>
> **Two defects in the Rust-only first pass, both found by doing the Python side.**
> `widen` sized both ends of an aux interval by the larger end's magnitude (the
> enclosure of `x**5` over [-18.3,-0.64] is [-2.06e6,-0.106], so the small end moved
> 7.3e-9 — a 7e-8 relative widening); and `bounded_mag` zeroed anything past the
> `1e20` sentinel, which also zeroed legitimately large DERIVED values — `x**3` over
> a box reaching 9e6 encloses 7.3e20, whose ulp is 1.3e5, left entirely unguarded.
> The sentinel means "this BOX is unbounded" and is now tested on box endpoints only.
>
> **A pre-existing gap this work exposed, worth its own issue.** The spatial path
> cannot certify infeasibility of `x*y >= c, x + y <= 1` over `[0,1]^2` for ANY `c`
> tested — 0.3, 0.45, 0.49, 0.55, 0.6, 0.75, 1.0, 1.5, 2.0 all run ~12 000 nodes to
> the time limit in BOTH arms. `test_debug_adversarial`'s infeasible fixture passed
> only because `c = 0.5` puts the root relaxation's feasible set at exactly one
> vertex — `(0.5,0.5,0.5)` satisfies every McCormick facet with equality — so the
> verdict came from whichever way that razor-thin LP rounded, and the solver closed
> it in 1 node without ever branching, contradicting the fixture's own docstring.
> The fixture now uses a model whose verdict is deterministic in both arms.

### 15a. Retraction and graduation: the "harmful" verdict on #956 was noise, and the guard ships ON (2026-08-09)

Everything in §15 above marked as evidence that
`DISCOPT_ENVELOPE_OUTWARD_ROUND` is *harmful* is **retracted**. Both measurements
were re-run on the current tree after the #956 T3′ fix (a proven-empty region can
now be fathomed without an incumbent) and neither survives.

**The 6/6-regressions table was noise, and the missing piece was a control.** That
table has no arm in which the two builds are known to be identical, so it had no
way to distinguish a real effect from run-to-run variation. The T6 panel supplied
one: comparing this branch against `origin/main` on six instances where the only
differing code path *provably never executes* (`TreeCertInfeasPrunes = 0` on all
six), the same harness still reported "WIN 3/3" for one instance and "REGRESSION
4/7" for two others, with bound spreads of 0.1–0.7 % at `time_limit=20`. That is
the noise floor. Re-run against it, the guard's own A/B is a wash: 3 identical, 1
win (tanksize, every replicate), 2 unresolved, 2 "regressions" at 2/3 — every one
of them inside the floor. `nvs20`'s "deterministic, not noise" reading also fails:
it is now *identical* in both arms.

**The ex1252 undecided-fraction table can no longer be measured at all.** That
configuration does not route through the native spatial kernel any more — the
panel's wrapper counts 0 kernel calls for it, while counting 21 elsewhere in the
same run, so the probe is live and the answer is real. Across the 21 corpus
instances that do reach the kernel, `n_undecided` is **0 in both arms** over
12 636 (ON) / 12 255 (OFF) kernel nodes. There is no undecided fraction left for
the guard to worsen. The mechanism story §15 built on that table — "loosening
makes conditioning worse, so more LPs come back undecided" — has no surviving
measurement behind it and should not be cited.

**Cert-clean panel, ON vs OFF, one build, 119 instances:** 149 executed
assertions, **0 false bounds, 0 false objectives, 0 false infeasibles, 0 new
errors**. The single flagged certification difference (`nvs12`) does not reproduce
— 6/6 interleaved replicates return `optimal`, 231 nodes, bound −481.2 in both
arms.

**Graduated default-ON.** §5's net-positive bar exists to keep sound-but-useless
*performance* machinery off the default path. The outward-rounding guard is not
that — it is the repair for the reported defect (unguarded generators cut
`(x, f(x))` out of their own envelope, a route to a false bound; six Rust
regression tests fail with `=0`), and its cost is not measurable above the noise
floor. §1 governs: correctness before performance. `=0` remains the opt-out and
the legacy arithmetic behind it is untouched. §15's "what would graduate it"
paragraph — a cheaper, cancellation-sized guard — is still the right *next*
refinement, but it is no longer a precondition for shipping.

**Also closed here:** §15's last paragraph called the `x*y >= c, x + y <= 1`
non-certification "a pre-existing gap worth its own issue". It was fixed instead,
in this branch: the tree pruned only on `lower_bound >= incumbent`, and with no
incumbent that test is never true, so every certified-empty node was branched
rather than fathomed. It now returns `infeasible` in **1 node** for every `c` from
0.3 to 2.0, against ~12 000 nodes to the time limit before.

### 15b. #956 T2′: the bound excursions are authored by a drifted `x_B` — and repairing that does not reduce them (2026-08-09)

Recorded here because it is a **negative result about the LP engine**, and §15's
cost model is where those belong.

The primal loop maintains `x_B` incrementally and re-derives it exactly only on
the ≤48-update refactorizations. The Harris ratio test reads that estimate to pick
the leaving variable and the step, so a drifted estimate authorises a step that is
infeasible in the values the basis actually holds — and no later refresh repairs
it, because a refresh corrects `x_B`, not the basis. Sweeping an exact-refresh
cadence `K` on `nvs20` moves the phase-1 handoff violation rate monotonically over
a 30× range: **50.3 % (K=0) → 28.1 % (16) → 21.8 % (8) → 14.9 % (4) → 10.1 % (2) →
1.7 % (K=1)**. EXPAND is excluded independently: suppressing it on every solve
moves the rate 49.7 % → 48.2 %, and 97 % of the violations exceed its per-pivot
ceiling by over 100× with **none** at or below it.

**The falsification.** Driving that rate to 1.7 % barely touches the failures it
was supposed to explain: `LpAuditBoundsFail` 788 → 764, `LpVerdictNumerical`
864 → 856. So a box-infeasible phase-1 handoff is real but is **not** what makes
node LPs come back `Numerical`. Any future work that proposes to cut the
`Numerical` population by fixing primal feasibility should treat this as settled
and look elsewhere (discopt#364's class).

**Not shipped, and why.** The cadence is sound (it only replaces an estimate with
the value the basis already holds — pinned by
`xb_refresh_cadence_cannot_move_the_optimum`) but not net-positive under a
wall-clock budget: the extra ftran costs throughput, and fewer nodes in the same
20 s means a lower bound. K=1 is 2 wins / 4 regressions (nvs19, nvs23, nvs24 at
4σ–9σ outside the replicate spread); K=16 is 2 wins / 2 regressions / 2 unresolved.
`DISCOPT_PRIMAL_XB_REFRESH` therefore defaults to `0` and
`DISCOPT_SIMPLEX_NO_EXPAND` to off — kept as the control arms, since eliminating
drift or suppressing EXPAND is the only way to judge a future ratio-test
hypothesis, and each is pinned by a test rather than left as dead configuration.

One measurement worth keeping for CC-style costing: on `nvs20` at K=1 the engine
performs 92 116 extra ftrans over a 20 s solve and still completes 973 nodes
against 902 at K=0 — the exact-`x_B` refresh is far cheaper per pivot than the
node-level work around it, so the throughput cost that decides against it is small
and instance-dependent, not structural.

## 16. #1008 LP throughput: the gap is per-iteration linear algebra, and six in-repo levers are falsified (2026-08-12)

**Claim under test.** The in-house simplex is 8.7x–29x slower than HiGHS on
identical LPs (issue #1008, measured on QPLIB_1157's root relaxation:
`rlt=0 rows=3273 highs=0.034 s in-house=0.292 s`, `rlt=1 rows=3937
highs=0.231 s in-house=6.696 s`). Panel throughout: 18 relaxation LPs captured
from the spatial B&B root on manifest-selected continuous nonconvex QPLIB
instances (`m` 2550–8000), 45 s per-LP limit, arms interleaved *within* each
repetition, `uptime` recorded per run.

**Entry attribution (`samply --rate 999`, QPLIB_1157 `rlt=1`, `m = 3937`).**

| region | share |
|---|---:|
| `FeralLU::factorize_sparse` | 59.5% |
| — numeric `SparseLu::factor` | 45.7% |
| — symbolic `SparseLuSymbolic::analyze` | 13.5% |
| Forrest–Tomlin `update` | 18.7% |
| ftran / btran | 16.4% |
| pricing + ratio test | 1.7% |

Iteration counts: **in-house 4965 pivots vs HiGHS 2153 — 2.3x the pivots but
12.5x the time per pivot.** Pricing and the ratio test, the issue's named
suspects, are 1.7% of wall combined; Devex pricing is already present in both
loops. This is the binding decomposition: the gap is *per-iteration linear
algebra*, and any future attempt that starts from the pivot rule is starting in
the 1.7%.

**Falsified levers.** Each was pre-registered with a kill criterion before
implementation.

| lever | outcome |
|---|---|
| refactorization cadence (`updates >= 48`, constant in `m`) | median 1.12x at interval 100/200, and QPLIB_0911 goes 1.9 s `optimal` → 47 s `iter_limit` |
| LU threshold pivoting (`u = 0.1`) to cut fill | fill −2% to −18%, wall flat; `u = 0.5` and `u = 0.01` both regress QPLIB_0911 to `iter_limit` |
| *is there any fill headroom at all?* | **no** — SuperLU/COLAMD on the same 12 final bases is median **0.94x**; feral's factor is already smaller on 8 of 12. The planned maximum-transversal work was cancelled on this measurement |
| symbolic-ordering reuse (`DISCOPT_LU_SYMBOLIC_REUSE`) | reuse rate 77.2% (1016/1316), so the mechanism operated — median **1.009x** vs a 1.05x bar, plus two `optimal` → `iter_limit` regressions. Objectives clean (0/11 pairs above 1e-9) |
| dual-loop Harris stability pass (`DISCOPT_LP_DUAL_HARRIS`) | wall median **0.999x** (0.774x–1.228x), median Δiter **−0.3%** vs a −5% bar, fires on only 10/18, and regresses QPLIB_1745_rlt1 `optimal` (42.5 s) → `iter_limit` |

Both flags were **deleted**, not shipped default-OFF (CLAUDE.md §3, no dead
flags) — the `DISCOPT_CUT_INHERIT` outcome with a status regression on top.
Retraction (§11): an earlier "roughly 1.10x expected from ordering reuse"
estimate is withdrawn.

**Phase split, the one artifact kept (PR #1012, bound-neutral instrumentation).**
`Phase::LuSymbolic`/`LuNumeric` plus `LuSparseFactorizations`/`LuBasisNnz`/
`LuFactorNnz` decompose the 59.5% bar over the 18-LP panel: numeric ≈**50%** of
total LP wall (18%–74%), symbolic ≈**11%** (2.7%–25.5%), everything else
(ftran/btran + FT update) ≈**38%**.

**Where the residual lives — and it is not in this repository.** feral's numeric
kernel is a median **2.1x** slower than `scipy splu` on the engine's own final
bases (n=18, min 0.6x, max 4.1x, biased in feral's favour), and at ~50% of wall a
*perfect* numeric fix is worth at most `1/(0.5/2.1 + 0.5) = 1.36x`. The other
≈38% is ftran/btran/FT-update, and feral's `src/lu/sparse_solve.rs` implements
these as **dense-vector loops over all `m`, not hyper-sparse** (Hall & McKinnon)
— on `m ≈ 4000–8000` bases with single-digit-nnz right-hand sides that is the
textbook order-of-magnitude case. So ≈88% of per-pivot cost is external to
discopt. **Binding conclusion: no in-repo scheduling change (cadence, ordering,
pivot threshold, ratio-test stability) moves this gap.** Closing #1008 requires
hyper-sparse triangular solves and a faster numeric kernel *in `feral`*, or a
different LU backend — a scope decision, not a tuning exercise.

**Two methodological notes worth generalizing.**

1. *Iteration counts on time-limited rows measure throughput, not convergence.*
   On a row that hits the limit in both arms, **fewer** iterations means
   **slower** per iteration. The dual-Harris panel's headline "−20% iterations"
   on QPLIB_1451_rlt1 is a 20% throughput **loss**. This is the most attractive
   misreading available in any deadline-capped A/B.
2. *A single pivot-path change is not an effect size.* One repivot in 6121 pivots
   moved QPLIB_1619_rlt1's wall by 1.23x. Cells in these panels are only
   interpretable in aggregate.

**Separable finding (issue #1013).** Five *independent* rounding-level
perturbations of the dual pivot path (refactor interval 100 and 200; LU pivot
threshold 0.5 and 0.01; ordering reuse) each drive QPLIB_0911's root LP from 1279
pivots / 1.9 s `optimal` to >6000 pivots / >45 s `iter_limit`, with
`DualDegeneratePivots` = 553/1279 and `DualStallTrips` = **0** throughout — the
warm-stall guard cannot fire on a cold dual solve. The dual-Harris pass clears it
(46.2 s `iter_limit` → 4.4 s `optimal`, 10.5x) even though it is corpus-neutral,
which is why the mechanism is recorded there rather than discarded.

## 17. #1004 GDP primal: the "~80% false-infeasible" rate is a property of the start *sampler*, not of the constructor (falsified 2026-08-13)

**Claim under test.** Issue #1004: `one_hot_config_subnlp` "rejects ~80% of
genuinely feasible configurations" because it solves **one** start per candidate
configuration. Evidence given: with the integers pinned to a known-feasible
configuration on `syngas`, 12 of 67 starts (B1) and 2 of 6 (#993 C2) produced a
feasible point.

**Entry attribution, before any clock.** The constructor does not draw its start
from a family. At a fixed configuration its seed is `clip(0, lb, ub)` on every
continuous slot plus the candidate configuration on every one-hot and residual
binary — a function of the model and the candidate alone (`x_relax` survives only
in *general* integer slots outside every group, which on `gdplib_small` means
`modprodnet` and nothing else). A detection rate obtained by sampling starts
measures the sampler. Both cited probes sampled.

**E1 — detection panel, all 12 `gdplib_small` models, plus a deep second pass on
the three whose first pool held no feasible configuration: 12,180 executed
feasibility tests.** Three arms per configuration: **Z** the constructor's own start, **X**
the relaxation point's continuous slots, **R** 8 stratified random starts.
Feasibility oracle = "any arm succeeded", so it is symmetric across arms. The
configuration pool deliberately excludes dive-derived configurations from the
verdict table: `one_hot_config_dive` accepts by trying the zero start *first*, so
a dive-sourced witness set is enriched with exactly what arm Z can already solve —
a bias that sank this probe's first version and forced the rebuild.

| arm | detection |
|---|---|
| Z — the constructor's start | **192/193 (99%)** |
| X — relaxation-point start | 186/193 (96%) |
| R — stratified random start | 981/1544 (64%) |
| **Z, restricted to configurations proven feasible *without* it** | **188/189 (99%)** |

The deep pass — dive disabled, so every pooled configuration comes from a channel
that never consults the zero start — adds 9 feasible configurations out of 915
sampled (2 in 460 on `small_batch`, 1 in 447 on `cstr`: the rare-event regime is
real) and the constructor's start solves **9 of 9**. Combined: **201/202**.

`batch_processing` is the mechanism in one row: **64/64** for the constructor's
start on the same 64 configurations where **0/512** random starts succeed. A
sampling probe on that model would report 0% and conclude the constructor rejects
everything.

**E2 — the escape hatch (restart cost), 420 timed solves across two arms.** The
issue's own budget argument (`(1−(1−p)^k)/(k·p) ≤ 1`) flips sign only if starts
2..k on an already-built sub-problem are materially cheaper than start 1. The
same-start control — vary nothing but repetition — measures **0.967 / 0.954 /
1.026 / 1.010 / 1.005** on small_batch / cstr / spectralog / batch_processing /
gdp_col. A restart costs what the first solve costs; `subnlp` shares its evaluator
across every plan already and retains nothing per configuration.

The different-starts arm carries the trap worth generalizing: `batch_processing`
reads **0.018** there (271 ms → 4.9 ms), which looks exactly like the escape
hatch. Those are the 0/512 random starts *failing immediately*. **A ratio below 1
in a multistart timing arm tracks how fast a start fails, not how cheap a restart
is** — the two are indistinguishable without a same-start control, and the control
for that model is 1.010.

**Break-even, at the measured `p = 0.995`.** With restart cost ratio `ρ`, `k`
starts per candidate cover `B/(1+(k−1)ρ)` candidates and yield
`(1−(1−p)^k)/[(1+(k−1)ρ)·p]` relative to single-start: break-even needs `ρ ≤
0.52%` at `k = 2` (0.26% at `k = 3`, 0.13% at `k = 5`). At the measured `ρ ≈ 1.0`,
`k = 2` yields **0.50×** — it halves the answer rate. Even a restart costing 10%
of a first solve loses (0.91×). **Binding conclusion: no multistart-per-candidate
ships; the single-start design is correct and the margin is not close.**

**Retraction (§11).** #1004's headline is withdrawn — the B1/C2 measurements stand
as measurements, the inference to the constructor does not. Nothing here says
`zero_start` is *optimal*: `water_network` costs it 1 of 7 configurations and
`gdp_col`'s dive pool 2 of 5 (where the relaxation-point start gets 5/5). The
union of two starts would detect more, at exactly the coverage cost the arithmetic
above rejects.

Full record and artifacts: `docs/dev/data/issue1004-gdp-config-start-detection.md`.
Pinned by `python/tests/test_issue1004_gdp_config_start.py` (start is
relaxation-point independent; each candidate tested exactly once; the wave's
`_WAVE_SOLVE_CAP` stays a *candidate* cap) — all three mutation-checked.

## 18. #1013 dual degeneracy stall: both escapes are unreachable, and the fix is the cold hand-off (2026-08-13)

**Claim under test.** Issue #1013: a rounding-level perturbation of the dual
pivot path multiplies the iteration count several-fold and turns `optimal` into
`iter_limit`, while `DualStallTrips` reads 0 through 43% degenerate pivots. Its
suggested direction: instrument the ratio test, re-land #1008's dual Harris pass
*scoped to the stall*, and make the stall detector able to fire.

**Environment caveat, first.** `QPLIB_0911` — the instance the issue is written
around — is unreachable here: the `~/Dropbox` MINLPLib/QPLIB snapshot is absent,
`qplib.zib.de` is blocked by the network policy, and `scratchpad/i1008/lps/*.npz`
was never committed. The issue's headline cell (1279 pivots → `iter_limit`; the
10.5x Harris recovery) was **not** re-measured and is neither confirmed nor
contradicted below. The panel is built instead from the vendored corpora: 102
root relaxations from all 9 in-repo QPLIB instances and all 68 in-repo MINLPLib
`.nl` instances at `rlt_lineq` off/on, of which 100 have a dual-feasible slack
start (`scratchpad/i1013/`, `FINDINGS.md`).

**Entry measurement — the escapes are not merely quiet, they are unreachable.**
Over all 100 LPs: `DualStallTrips` = **0** and `DualBlandActivations` = **0**, on
every single LP, including cells that are 98.7% degenerate and exhaust their
budget. Bland's rule engages at `2·(n+1)` consecutive degenerate pivots (58 194
on the worst cell) and the F2 guard at the size-derived pivot cap (~10⁶). This
generalizes the issue's point 3 past its instance: on lifted relaxations the dual
loop has **no** anti-degeneracy escape at all.

**The mechanism is not the hypothesized tiny pivots.** Per-pivot trace
(`DISCOPT_LP_DUAL_TRACE=1`, new) on `QPLIB_3815_rlt1`, 8192 pivots: 98.6%
degenerate, chosen `|pivot|` median **exactly 1.0** (min 1.1e-2, max 4.6e2),
**zero** pivots below 1e-4, and the primal step is never 0. The degeneracy is
*dual* — `d_q ≈ 0` because a lifted relaxation's objective is sparse — so a
stability tie-break on `|α_rj|` has nothing to discriminate.

**Falsified levers** (100-LP panel, arms interleaved within each rep, per-LP
20 s limit; each pre-registered against a status-regression kill criterion).

| lever | outcome |
|---|---|
| dual Harris pass, **armed only inside a degeneracy stall** (the issue's ask) | fires on 22/100, wall median **0.995x**, iteration median **0**, and regresses `tspn10_rlt1` `optimal` → `iter_limit`. Same verdict as #1008 on a different corpus |
| Bland's rule at a *reachable* run length (512) | fires on 8/100, median 1.007x, regresses `tspn10_rlt1` the same way |
| a second progress measure (bail only if the total primal infeasibility is *also* flat) | on every LP traced, `max_noprog == max_run` exactly — the primal term never reset a counter the degeneracy test had not already reset. No discriminating power; removed rather than shipped as an untested branch |

**What ships.** `SimplexOptions::dual_stall_patience` (default **2048**,
`DISCOPT_LP_DUAL_STALL_BAIL=<n>` overrides, `=0` disables): after that many
consecutive degenerate pivots the warm loop returns `None`, i.e. the caller's
cold two-phase primal — the action every other difficulty in this loop already
takes, and one that self-verifies its own verdict. The threshold separates two
non-overlapping measured populations: every LP whose warm loop converges peaks at
a **902**-pivot run (median 120); every cell that does not converge runs
**≥ 1274**.

**Graduation panel (2 reps, `off` vs default).**

| gate | result |
|---|---|
| unchanged cells | **98 / 100** identical status *and* identical iteration count |
| status regressions | **0** |
| status improvements | **1** — `QPLIB_3814_rlt1` `infeasible` → `optimal` |
| objective drift on optimal/optimal | **0.00e+00** (n=96) |
| wall, LPs ≥ 50 ms | 0.96x–1.20x; the two cells where it fires gain 1.13x and 1.20x |
| wall, LPs < 50 ms (83 of 100) | 0.74x–1.47x spread at **identical iteration counts** — sub-millisecond noise, not effect |

**Tree-level differential panel.** Every vendored `.nl` with a recorded optimum
(16 instances, 60 s each, `scratchpad/i1013/tree_panel.py`), default vs off: 96
assertions (no bound past its reference optimum, no incumbent beyond it, no
certification or status regression), **0 issues**, and objective, bound and node
count **bit-identical on all 16** — the bail does not fire in these trees, whose
longest degenerate runs are 47–87 pivots. `pytest -m smoke` (1008 passed),
`pytest -m slow python/tests/test_adversarial_recent_fixes.py` (10 passed) and
`cargo test -p discopt-core` (604 passed) are green.

**The `QPLIB_3814_rlt1` cell is why this is not only a throughput issue.** On the
default path that LP returns **`infeasible`** after 5543 pivots (3352 of them one
degenerate run). SciPy/HiGHS returns `optimal` at 0.238394628159; every perturbed
arm of our own engine returns `optimal` at 0.238394628195; the returned point
satisfies every row and bound to 2e-8; an elastic LP (`min t` s.t. `Ax − t ≤ b`)
returns `t* = 0.0`. The certificate that decides it turns on `bᵀy = 3.0e-8`
against a term magnitude `Σ|b_i·y_i| = 600` — a relative margin of **5e-11** —
tested against a Neumaier–Shcherbina margin, `1e-9·(1+|bᵀy|+Σ|boxmax|)`, built
from *result* magnitudes rather than accumulation magnitudes. **That is a
separate defect and is filed separately** (the bail only stops the stalled warm
loop from being the thing that decides it; it does not repair the margin).

### 18a. Re-measured on the post-#1017 base: the status improvement was #1017's (2026-08-13)

`main` merged #1019 (the Farkas accumulation-margin fix for #1017) while this
branch was open. That fix repairs the certificate that produced the
`QPLIB_3814_rlt1` false `infeasible`, so on the merged base **that LP returns
`optimal` with the stall bail OFF as well as ON** (obj `0.23839462819531176`,
identical). The "1 status improvement" this section credited to the bail is
therefore **#1017's result, not this mechanism's**, and the claim is withdrawn
(CLAUDE.md §11).

Panel re-run on the merged base (100 LPs, 2 reps, `scratchpad/i1013/postmerge_panel.jsonl`):

| gate | pre-merge base | post-#1017 base |
|---|---|---|
| unchanged cells (status *and* iterations) | 98 / 100 | **99 / 100** |
| status regressions | 0 | **0** |
| status improvements | 1 (`QPLIB_3814_rlt1`) | **0** — the cell is now `optimal` in both arms |
| objective drift, optimal/optimal | 0.00e+00 (n=96) | **0.00e+00** (n=97) |
| cells where the bail fires | 3 | 3 — 1.34x, 1.13x, and one neutral |

Tree panel re-run on the same base: objective, bound and node count identical on
all 16 instances, 96 assertions, 0 issues.

**What the mechanism is now worth, stated plainly.** It is a *tail* guard, not a
throughput change: inert on 97 of 100 panel LPs (bit-identical iteration counts),
and on the three where it fires it converts a stalled warm grind into a cold
solve worth 1.34x (`QPLIB_3814_rlt1`, 1.78 s → 1.33 s) and 1.13x
(`tspn10_rlt1`, 16.18 s → 14.36 s), with the third neutral. That is the same
shape of value as the F2 stall guard it sits beside — bounding a pathological
tail — and it is **not** a broad net-positive, which is how it should be read.

**A second methodological note (§8), because it produced a false regression.**
The first tree panel reported `nvs12` 2.8 s → 47.8 s and `nvs11` 0.7 s → 6.6 s.
That was the harness: its "on" arm set `DISCOPT_LP_DUAL_STALL_BAIL=1` and the
flag parsed `1` literally as *a patience of one pivot*, bailing on the first
degenerate pivot of every node LP. The counters said so — 16 bails with
`DualDegenerateRunMax` = 1, which is impossible under the intended semantics —
and reading them back is what caught it. The flag now reads `1`/`true`/`on` as
"enabled at the default patience" and only integers ≥ 2 as an explicit patience.

**Methodological note (§6 again, and it cost a design).** The first version of
this fix was built on a Python probe that reconstructed the engine's progress
test *without its tolerance filter*, so sub-tolerance noise read as progress and
the probe reported `QPLIB_3871_rlt0`'s no-progress window as 24 pivots when the
engine's own counter says 663. A whole conjunctive test was designed around that
number, shipped into a panel, and measured as useless. The rule that would have
caught it earlier: read the quantity **the code tests**, out of the code, not a
reimplementation of it.

### 18b. The bail is default-OFF: "never the value" was false (#1008, 2026-08-13)

The soundness argument for graduating this flag default-ON, in §18 and in the
shipped doc comments, was: the bail returns to the cold two-phase primal, which
self-verifies its own verdict, **so it can change only *which* engine finishes a
solve, never the value.** That argument silently assumes the cold solve finishes.
All three bailing cells of the 100-LP panel had a cold solve that did. **Measured
on QPLIB root relaxations outside that panel, two do not** — and the guarantee
fails in the worst available direction (CLAUDE.md §11; this withdraws the claim
in §18, in `SimplexOptions::dual_stall_patience`, and in the CHANGELOG).

Env A/B on the shipped binary, no rebuild, `time_limit=None`, oracle = HiGHS:

| LP | `DUAL_STALL_BAIL=0` | default (`=1`) | bails | cold verdict |
|---|---|---|---:|---|
| `QPLIB_2738` | **`optimal` −5.0587686**, 9.6 s | **no solution**, 12.5 s | 1 | `Numerical` |
| `QPLIB_2170` (`time_limit=40`) | **`optimal` 0**, 1.7 s | **no solution**, 0.2 s | 1 | **`Unbounded`** |
| `QPLIB_3225` | no solution | no solution | 0 | `Numerical` — unrelated; cause still unfound (see below) |

> **The `QPLIB_3225` attribution above is RETRACTED, 2026-08-14 (§11).** It
> originally read "needs feral #160", i.e. that the LU ordering was the cause.
> Upstream ran the A/B and closed **feral #166** as not reproducible: the AMD arm
> and the peel arm return the *same* objective (511.52671247757985), differing
> only in bound (508.88462071047877 / 508.88769205717875) and nodes (4625 / 5357).
> `QPLIB_3225` does not track the ordering flip. The real cause is still unfound;
> nothing in the row above depends on it, since the row is a zero-bail control.

The `QPLIB_2170` row is the sharper one: the cold path does not refuse, it
**claims `Unbounded`** on an LP HiGHS certifies as `optimal 0` in 81 pivots and
that our own warm loop solves to 0. A bail whose fallback can produce a false
verdict is not a neutral change of engine.

**The detector cannot separate "stalled" from "converging slowly."** QPLIB_2170
reaches its optimum after **~22 800** consecutive degenerate pivots with 16 435
Bland activations — an order of magnitude past the 2048-pivot patience, and past
the "every converging warm loop peaks at 902" population that derived it. That
population was measured on the same 100 LPs that contained no cold-failing cell;
it is a property of that sample, not of the loop.

Against the mechanism's own measured worth — inert on 97 of 100, 1.34x and 1.13x
on two cells, one neutral — §1 does not permit trading a certificate for that.
The flag is **default-OFF**; `DISCOPT_LP_DUAL_STALL_BAIL=1` opts in and the
mechanism is untouched, so a future panel that gates on *bound retention* as well
as wall-clock can re-graduate it. #1013's own PR body had already withdrawn
"broadly net-positive" and named this as the alternative.

Pinned by `dual_stall_bail_can_cost_a_bound_when_the_cold_solve_fails` against the
vendored `qplib2170_cold_fail_lp.json` (1755×3193, captured by measured outcome
rather than by name, §2). It returns `Unbounded` on the old default and
`optimal 0` on this one, in ~2 s, with no Python and no corpus dependency.

**Still open, and not fixed here:** the false `Unbounded` itself, and the
`bank_deadline_duals` coupling that makes QPLIB_2170 solvable only when the caller
passes a `time_limit` (#1008 R1) — both tracked in #1008.

### 18c. The false `Unbounded` is a broken ftran read as a verdict (#1008, 2026-08-13)

§18b left this open. It is not a property of the stall bail: with
`DISCOPT_LP_DUAL_STALL_BAIL=0` (the default since #1021) and `time_limit=None`,
QPLIB_2170's root relaxation still reaches the ratio test's unbounded exit —
`UnboundedRejectRowResidual = 2` end to end.

**What the engine was doing.** `Infeasible` is issued only after
`farkas_ray_certifies_cols` accepts a dual ray. `Unbounded` was issued on the
strength of "the ratio test found no blocking row" alone. But that sentence is a
statement *about* `α = B⁻¹A_q`, so an ftran that silently returns zeros produces
it verbatim — which is exactly what happens here. The captured ray:

| quantity | required | measured |
|---|---|---|
| nonzeros in `d` | ≥ 1 basic + entering | **1** (the entering column only) |
| `max_i |A d|_i` | 0 | **1.0** |
| `cᵀd` | < 0 | **0.0** |
| box-recession violations | 0 | 0 |

Three of the four conditions fail. The LP is `optimal 0` (HiGHS, 81 pivots).

**Blast radius, by driver.** `spatial_tree::verdict_for` already maps
`Unbounded`, `IterLimit` and `Numerical` alike to `NodeVerdict::Undecided`, so
there the cost was only the bypassed `dense_retry` — which is gated on
`Numerical | IterLimit` and so never ran. `milp_driver` is the serious one:
`out.unbounded` at any node sets `hit_unbounded`, breaks the search loop, and
short-circuits `decide_status` ahead of every other branch, returning
`MilpStatus::Unbounded` for a bounded MILP and discarding any incumbent. One
false node verdict is enough.

**The fix** is the primal mirror of the Farkas path: `ray_certifies_unbounded`
checks `A d = 0`, `cᵀd < 0` under the cost the *phase* is minimizing (not
`self.c` — phase 1 is minimizing infeasibility), and box recession, refusing with
`Numerical` when any fails. Margins are built from accumulation magnitudes rather
than results (#1017). The relative slack is `1e-7`, deliberately not
`gamma(nnz)`: the residual carries the LU solve's error, not just summation
rounding. It rejects the observed breakdown by a factor of 10⁷.

`Numerical` is also the status that routes into `dense_retry`, so the honest
refusal is strictly more useful than the false claim.

**Not a bound recovery.** On QPLIB_2170 at `time_limit=None` the outcome stays
"no solution" — the engine simply stops claiming a certificate it cannot support.
The remaining loss on that cell is the `bank_deadline_duals` coupling (#1008 R1,
still open): the same LP solves to `optimal 0` at `time_limit=40.0`. The four-cell
BAIL × time_limit A/B is otherwise unchanged from §18b, so nothing regressed.

Pinned by `cold_primal_does_not_claim_unbounded_on_a_bounded_lp` (returns
`Unbounded` obj 351 without the certifier, `Numerical` with it) and by the
pre-existing `unbounded_detected` / `unbounded_emits_a_valid_primal_ray`, which
confirm a genuine unbounded ray still certifies untouched.

### 18d. A NaN bound is read as both open and closed (#1008, 2026-08-13)

The certifier in §18c turned a CI job red on `test_c3_unbounded_recourse_reported`
(`unbounded` → `iteration_limit`). The measurement — profile counters on the
Benders solve, `DISCOPT_PROFILE=1` — put the rejection at
`UnboundedRejectBox = 2`, not at the residual or objective margins. The LP behind
it, captured by wrapping the recourse oracle:

```
min -w   s.t.  -w <= 0,   w in [0, nan]
```

The **upper bound is NaN**. `Model.continuous(ub=None)` stores `array(nan)`; the
Rust simplex reads the sentinel `±1e20`. Nothing translated between the two, and
NaN does not announce itself, because every comparison against it is false — so
each guard reads the same bound as whichever answer its comparison is written for:

| guard | question asked | NaN answer | reading |
|---|---|---|---|
| ratio test | `ub < INF` — does this bound block? | false | **open** → `t = INF` → `unbounded` |
| §18c ray certifier | `ub >= INF` — is this side open? | false | **closed** → refuse |

So the pre-#1022 `unbounded` on that LP was *correct but underived*: right answer,
resting on which of the two readings the ratio test happened to use, over a box
the engine could not certify as recessive. The certifier did not break it — it
made a latent ambiguity observable. This is the `INF`-is-`1e20` hazard already in
`CLAUDE.md`, in its other guise: there the sentinel silently survives a
multiplication; here it is silently absent.

**The fix has two halves**, and needs both. `lp_simplex._finite_box` translates
the box (NaN, `±inf`, and any magnitude past the sentinel → `±1e20`) so the
simplex sees one convention; the PyO3 LP/MILP entry points refuse a NaN bound
with a `ValueError` naming the index, so a caller that skips the translation gets
a loud error rather than a verdict derived from two incompatible readings. `±inf`
is deliberately *not* refused — it satisfies `>= INF` and fails `< INF`, so both
readings already agree it is open.

**Method note.** The regression was found by CI and diagnosed by counter, not by
reading control flow: the first probe (`c3_probe.py`) printed every
`Unbounded*`/`Farkas*` counter and the *box* counter was the one that moved. A
control-flow reading would have blamed the residual margin, which is where the
§18c work had been concentrated. Baseline separation followed CLAUDE.md §8 — the
pre-#1022 `.so` was asserted to *lack* the `UnboundedRejectRowResidual` marker
before being trusted as a baseline, and it reproduced the pass.

Pinned by `python/tests/test_1008_nan_lp_bound.py` (3 tests; 2 fail with
`ValueError: ub[0] is NaN` when `_finite_box` is neutered).

### 18e. The unstable-pivot recovery was gated on the caller's `time_limit` (#1008 R1, 2026-08-13)

§18c closed the false certificate on QPLIB_2170 but left the bound loss standing,
and named the suspect: `SimplexOptions::bank_deadline_duals`, set as
`deadline.is_some()`, was doing double duty. Besides banking the dual loop's
anytime floor (#928 — its actual job) it also switched on the recovery that
re-tries a near-zero pivot instead of abandoning the warm re-solve.

**Entry experiment** (CLAUDE.md §4), run before the implementation and read from
counters rather than from control flow — two new counters exist precisely so this
is a measurement:

| call | `DualUnstablePivotRecoveries` | `DualUnstablePivotBails` | ray rejects | result |
|---|---|---|---|---|
| `time_limit=None` | 0 | **1** | 2 | no solution |
| `time_limit=40.0` | **1** | 0 | 0 | **optimal 0** |

One unstable pivot separates a certified optimum from nothing, and which side you
land on is decided solely by whether the caller bounded the LP in time. QPLIB_2738
fires neither counter — its loss was the #1013 stall bail, already fixed — so this
is a distinct mechanism, not a second reading of the same cell.

**The fix** is an unbundling, not a new mechanism: `recover_unstable_pivot` gets
its own field and its own env gate (`DISCOPT_LP_UNSTABLE_PIVOT_RECOVERY`), and
`bank_deadline_duals` goes back to banking duals. The deadline path keeps the
recovery unconditionally (`deadline.is_some() || unstable_pivot_recovery_default()`)
so it stays bit-identical to what its own panel judged.

**Status: default-OFF for deadline-free callers.** Taking a different pivot is
bound-changing, so §5 applies and the flag stays off until a corpus differential
panel clears *both* bars. The graduation panel (88 captured relaxation LPs,
`scratchpad/i1008/r1_panel.py`, one process per arm because the flag is read once
via `OnceLock`, HiGHS as oracle, every LP at `time_limit=None` — the exact call
shape that lost the recovery) has since **run to completion on both arms**. Result:

| bar | verdict | evidence |
|---|---|---|
| **cert-clean** | **PASS** | 82 LPs compared; 0 bounds below the HiGHS optimum, 0 certification regressions, 0 objective drift, 0 LPs at the iteration cap |
| **net-positive** | **FAIL** | 0 bounds recovered; retention identical at 72/82 in both arms |

**GRADUATE: NO — the flag stays default-OFF.** The mechanism is *sound*, not
*helpful*: precisely the `DISCOPT_CUT_INHERIT` outcome, and per §5 a cert-clean
but neutral flag stays off with the measurement recorded.

The probe was **not empty**, which is what makes the FAIL meaningful (§6): the
recovery fired on 5 ON-arm LPs (`QPLIB_2590_rlt0/rlt1`, `QPLIB_2819_rlt0`,
`QPLIB_2823_rlt1`, `QPLIB_3089_rlt1`, 12 recoveries total) against 7 OFF-arm bails.
So the ON arm really did take different pivots on those LPs and still recovered no
bound the OFF arm lacked — a measured neutral, not an arm compared against itself.
6 of the 88 LPs were skipped (`_dual_start_slack_basis` rejected the start, so they
never enter the warm path) and are named in the log rather than silently dropped.

Wall was 2829.81s OFF vs 2835.45s ON — a single unreplicated run, on a machine
that was also running the §18f interval experiment (load I created myself), so it
is directional only and supports no timing claim in either direction. It is
recorded because a 0.2% spread is worth knowing is *not* a signal.

Pinned by `unstable_pivot_recovery_is_not_gated_on_a_deadline` (flag off → bails 1,
not optimal; flag on → recoveries 1, optimal 0; both arms assert
`deadline.is_none()`, which is the coupling under test).

### 18f. LP wall is LU factorization, and the 48-update cap is load-bearing (#1008 D3, 2026-08-13)

Two results: an attribution that redirects #1008's planned work, and a
falsification that kills the fix that attribution most obviously suggests.

**The attribution.** Phase profile over the 19 captured relaxation LPs the R1
panel had finished (`scratchpad/i1008/attrib.py`, share-of-wall over 600.5s):

| phase | share of LP wall |
|---|---|
| **LuNumeric** (`SparseLu::factor`) | **72.6%** |
| Refactorize (primal, outer timer) | 7.7% |
| LuSymbolic (`SparseLuSymbolic::analyze`) | 5.0% |
| FtUpdate | 1.9% |
| PriceBtran + AlphaFtran + PriceSweep | **1.5%** |

`PriceSweep` alone is **0.1%**. The planned D2 work — column-wise PRICE at
O(nnz(A)) per pivot — therefore targets a phase that is 1.5% of wall on this
corpus and cannot pay for the 8.7x–29x gap in the issue title. **D2 is
deprioritized**; the cost is the LU.

Shares, not speeds: the two panel arms ran concurrently, so absolute ms are
contention-inflated (§9). Contention inflates all phases alike, so the shares hold
and no speed claim is made from them. Per-LP, the `LuNumeric` share tracks the
fill ratio directly — fill 1.0 → ~5%, fill ~6–7 → ~50%, `QPLIB_1451_rlt0` at fill
**19.1x** → **74.8%**.

**The hypothesis this suggested, and its falsification.** Both simplex loops
refactorized on a hardcoded `updates >= 48`. On `QPLIB_1451_rlt0` all 265 primal
refactorizations were cap-triggered (`RefacCap`; 12764 pivots / 48 = 266) and the
adaptive `refac_work_budget` gate beside it never fired, because the fixed cap
always trips first. With `FtUpdate` at 1.9% against `LuNumeric` at 72.6%, the
updates the cap truncates looked ~38x cheaper than the factorizations it forces,
so raising the interval should have cut the dominant cost.

It does the opposite. Entry experiment over 16 captured LPs
(`scratchpad/i1008/refac_entry.py` + `refac_report.py`, one process per interval —
the gate is a `OnceLock`), counters exact and load-independent:

| interval | factorizations | factor nnz | basis nnz | fill | cap-trig | FT-fail |
|---|---|---|---|---|---|---|
| **48** (default) | **667** | **75.1M** | 14.1M | **5.32x** | 651 | 0 |
| 100 / 200 / 500 | 792 | 309.4M | 22.5M | **13.7x** | 0 | 776 |

**+19% factorizations and +312% factor nonzeros.** Above ~100 the arms are
*identical*: the cap stops firing entirely and feral's product-form update reaches
its own stability limit instead (`DualRefacCap` 651 → 0, `DualRefacFtFail` 0 →
776). Letting updates accumulate to that limit lands the search on denser bases
(`basis_nnz` 14.1M → 22.5M — a different pivot path, not merely different factors)
and then produces far denser factors from them. Directional wall agreed: 30.5s →
195s.

**Kill criterion met; the direction is dead.** The 48-update cap is not mistuned,
it is load-bearing — it refactorizes *before* the update degrades, and the cheap
fresh factor it produces is the point. "Refactorize less often" is not available
as a #1008 fix. Correctness was unaffected throughout (64 arm×LP comparisons
against HiGHS, **0 deviations**), so this is a performance verdict only.

**The single-instance trap, for the record (§2).** On `QPLIB_1423_rlt1` alone,
raising the interval *helped* — 66 → 55 factorizations, 11.37M → 9.55M factor nnz.
That one LP was read first and looked like confirmation. The corpus aggregate
reverses it 4x. Nothing here was claimed until the 16-LP aggregate ran.

**What shipped.** Instrumentation and a measurement knob, no default change:
`DISCOPT_LP_REFAC_INTERVAL` (default **48** — unset is byte-identical to the
previous engine, unparseable input refused loudly rather than read as the default,
which would make an A/B harness measure the baseline twice), and three counters
that made the falsification visible at all — `DualRefactorizations`,
`DualRefacCap`, `DualRefacFtFail`. The dual loop's refactorizations were
previously **unattributed**: the `Refactorizations`/`Refac*` counters are
incremented only by the primal, so an LP the dual solved outright showed 100+
`LuSparseFactorizations` against zero refactorization events, and the aggregate
above could not have been computed.

**Corrections to statements made earlier in this issue's work (§11).**

1. `QPLIB_1451_rlt0` was described mid-investigation as "545 seconds at zero
   pivots". It is not: `Phase1Pivots 9` + `Phase2Pivots 12755` = **12,764 pivots**.
   The panel's `iters=0` is the *cold-fallback* signature (`DualColdFallbacks 1`;
   dual.rs documents `iters == 0` as exactly that) — the dual was abandoned and the
   cold primal did the work without reporting its iteration count. The reported
   `iters` on a cold-fallback solve is an observability gap, not a pivot count.
2. The `piv` column of `attrib.py` is a **primal-only** count
   (`Phase1Pivots + Phase2Pivots`). It reads 0 for the 18 LPs the dual solved
   because dual pivots are not in those counters — not because those LPs pivoted
   zero times.

**What the attribution leaves open.** The cost is `LuNumeric` and the lever is
fill, not frequency. feral's `analyze` already triangularizes and runs AMD on the
residual bump, so the 19.1x fill on `QPLIB_1451_rlt0` is *after* a fill-reducing
ordering — the next question for #1008 is whether that fill is intrinsic to these
bases or an artifact of static ordering under partial pivoting (the usual remedy
being threshold-Markowitz pivoting, which chooses pivots dynamically against both
sparsity and stability). That is a feral-side question and is **not** opened here.

> **RETRACTED, 2026-08-13 (§11).** The sentence "feral's `analyze` already
> triangularizes and runs AMD on the residual bump" is **false for the feral
> version discopt links**. It was read off feral's unreleased `main`;
> `git grep "fn triangularize" v0.15.1` returns nothing — triangularization landed
> in feral `1217992` (PR #160), seven commits past the pinned tag. Ordering was
> therefore never eliminated as a #1008 lever; it was never tested. §18g measures
> it.

### 18g. The ordering lever was never actually tested: feral bumped, peel measured, shipped OFF (#1008, 2026-08-13)

**First, the retraction (§11).** §18f closed by eliminating *ordering* as a #1008
lever — "feral's `analyze` already triangularizes and runs AMD on the residual
bump, so the 19.1x fill on `QPLIB_1451_rlt0` is *after* a fill-reducing ordering."
That is wrong for the build discopt links. `git grep "fn triangularize" v0.15.1`
returns nothing; triangularization landed in feral `1217992` (PR #160), **seven
commits past the pinned tag**. The claim was read off feral's unreleased `main`
and generalized to the shipped crate. `SparseLuSymbolic::analyze` at 0.15.1 runs
AMD over the **whole basis**, no Suhl–Suhl peel. The lever was not eliminated; it
was never tried. §18f's in-place note records this.

**The bump.** `crates/discopt-core/Cargo.toml` now pins feral to the git rev
`e00aa7060bb87359a8ed014a4a71a9f22fff0167` (head of feral's `claude/issue-161-w05b75`,
PR #162) rather than crates.io `0.15.1`. There is no release carrying any of this
work — 0.15.1 is still crates.io's latest. The rev carries feral #160/#162/#163/
#164/#165: Suhl–Suhl basis triangularization behind the parameterized
`SparseLuSymbolic::analyze_with(a, LuOrderingParams { triangularize })`, the
`LuParams::dense_bump_max_dim` route that sends a peeled residual bump to the
dense kernel, the sparse-rhs ftran/btran entry points, and the
`hyper_sparse_max_density` 0.25 → 0.10 default fix. The pin comment records that
this breaks the 0.15.1 alignment with POUNCE 0.10.0's workspace and must revert to
a version requirement once feral cuts a release.

**The bump changes nothing by default** — the Cargo.toml acceptance regime
(CLAUDE.md §5, bound-neutral) demands exactly that, and it is measured, not
assumed. Arms: `base` = 0.15.1, `off` = e00aa706 with the new flag unset, `on` =
e00aa706 with the peel on; one process each over the captured relaxation LPs,
`scratchpad/i1008/feral_arm.py` + `feral_report3.py`. Build identity is asserted
per §8 on a marker string (`sparse_triangular` present in the bumped extension,
absent at 0.15.1), not on a path.

- `off` vs `base`: **31 LPs × 8 fields (`status`, bit-level `obj`, `facs`,
  `basis_nnz`, `factor_nnz`, `p1`, `p2`, `cold_fallback`) — 0 differences.**
- `cargo test -p discopt-core --lib` at e00aa706, flag off: **614 passed, 0 failed**.
- All three arms' objectives against HiGHS at the repo tolerance, zero slack: **93
  arm×LP comparisons, 0 deviations.**
- The certifying panel, the acceptance regime the Cargo.toml pin comment mandates
  (`scratchpad/i1008/feral_cert_arm.py` + `feral_cert_report.py`, both arms
  concurrent under equal load): **52 instances × 4 fields (`status`,
  `node_count`, `objective`, `bound`) — all 52 rows bit-identical.**

**What the peel does (`on` vs `base`; counters are exact integers and
load-independent, wall is directional only because the arms ran concurrently, §9).
A representative slice of the 31:**

| LP | fill base | fill on | facs b→on | wall b→on | cold |
|---|---|---|---|---|---|
| **QPLIB_1451_rlt0** | **19.12x** | **2.62x** | 683→757 | **569.5→31.8** | 1→1 |
| QPLIB_1675_rlt0 | 10.61x | **1.06x** | 67→95 | 5.66→0.50 | 0 |
| QPLIB_1661_rlt1 | 10.15x | 8.67x | 326→323 | 155.5→60.4 | 0 |
| QPLIB_1493_rlt0 | 3.57x | **1.01x** | 30→22 | 0.34→0.07 | 0 |
| QPLIB_1055_rlt0/1 | 3.60x | **1.00x** | 19→20 | 0.37→0.07 | 0 |
| QPLIB_1143_rlt0 | 4.81x | **1.09x** | 30→31 | 0.92→0.15 | 0 |
| QPLIB_1437_rlt1 | 7.05x | 6.51x | 179→175 | 25.4→12.0 | 0 |
| QPLIB_1353_rlt1 | 6.15x | 6.70x | 111→106 | 7.00→4.80 | 0 |
| **QPLIB_0911_rlt0/1** | 6.79x | 2.93x | **27→474** | **2.00→13.06** | **0→1** |

Aggregate over the 31: fill **11.09x → 6.51x**, total factor nonzeros
**2.81e9 → 1.92e9 (−31.6%)**, per-LP factor nonzeros better on 25, worse on 5,
unchanged on 1; directional wall 1809.5s → 584.4s. The peel takes the bases whose
fill was moderate down to ~1.0x — the factor becomes as sparse as the basis — and
takes a third off the worst one in the corpus (`QPLIB_1451_rlt0`, 19.12x → 2.62x,
the instance §18f's attribution singled out at 74.8% `LuNumeric`).

*Correction to an interim figure stated in this session (§11).* The first 18 LPs
of this run gave fill 5.84x → 3.71x and total factor nnz **+65%**, and that
prefix was reported as the aggregate. It is not: the +65% was `QPLIB_0911`
dominating a prefix that had not yet reached the large instances. On the full
31-LP set the sign reverses to −31.6%. `QPLIB_0911` is still a real regression —
it is just not the largest term once the corpus is.

**Two blockers keep it default-OFF.**

1. **A bound is lost, and the pairing does not rescue it.**
   `bchoco06_illcond_scaled_path_recovers_bound_649` fails with the peel on —
   `Numerical` where the test asserts `Optimal`. Root-caused by a same-build A/B:
   flag on FAIL, flag unset PASS, so it is the ordering and not the bump.
   Upstream's own table (feral `895ef65`, `dev/research/lu-ordering-and-kernel-2026-08-13.md`)
   records whole-basis AMD PASS, peel-**without**-cap FAIL (`Numerical`), and
   peel-**plus**-`dense_bump_max_dim = 4096` **PASS** — the dense kernel reorders
   the bump's arithmetic again and lands back on a certifying path. discopt pairs
   the cap with the peel exactly as prescribed and **still fails at e00aa706**.

   That is not a wiring failure, and it is not assumed to be one (§6). The two
   counters added with this change measure the pairing directly on that fixture,
   profiling on, same build, flag the only difference:

   | | factorizations | `LuBumpDim` (sum) | `LuDenseBumpFactorizations` | status |
   |---|---|---|---|---|
   | flag off | 32 | 26,656 (≈833 = `m`, nothing peeled) | 0 | **Optimal**, obj −0.999989 |
   | flag on | 48 | **2,036** (≈42/factorization, ~95% peeled) | **42** | **Numerical** |

   The peel peels and the dense route fires on 42 of 48 factorizations; the bound
   is lost anyway. So the divergence from upstream's table is real and is the
   feral rev — `e00aa706` is #163's revert plus #164/#165, and their PASS was
   measured at `895ef65`. Reported as a fact here, not diagnosed: it is a
   feral-side numerical question. This is a §1 stop regardless — fill is not
   tradeable against a lost bound, whatever the geomean says.
2. **A cold fallback appears.** `QPLIB_0911` goes 27 → 474 factorizations with
   `DualColdFallbacks` 0 → 1 — the dual abandons and the cold primal redoes the
   work (§18f correction 1 documents that signature). Wall 2.0s → 13.1s. Fill fell
   (6.79x → 2.93x) and the LP still got 6x slower, so fill is not a sufficient
   objective on its own.

**Verdict (§5).** The dependency bump ships. The ordering ships as
`DISCOPT_LU_TRIANGULARIZE`, **default OFF** — matching feral's own
`analyze` default — with `dense_bump_max_dim = 4096` paired to it (feral #163
pairs the cap with the peel; unpaired, upstream's own table has it failing #649's
instance too) and `0` when the peel is off. Unparseable input is refused loudly
rather than read as the default, per the `DISCOPT_LP_REFAC_INTERVAL` precedent —
a harness that silently measures the baseline twice is the §6 failure mode. Two
counters were added so "the pairing engaged" is checkable rather than assumed:
`LuBumpDim` (sum of residual-bump dimensions; equals the basis dimension when
nothing peeled) and `LuDenseBumpFactorizations`.

**What remains for #1008.** Upstream's own 14-LP interleaved measurement (feral
`b8906a6`, all arms in one binary) puts the ordering at **1.306x** geomean, the
sparse-rhs ftran/btran entry points at **1.067x** alone, and both together at
**1.497x** — discopt's gap to HiGHS 14.4x → 9.6x. The sparse-rhs entry points are
a discopt-side call-site change with no ordering risk and are **not** part of this
bump; they are the next increment. The ordering itself graduates only when #649's
instance survives it — that is a feral-side numerical question (their recorded
counterexample `QPLIB_2055` at 0.389x is the same shape) and belongs upstream.

> **Both sentences above are RETRACTED, 2026-08-14 (§11).** See §18h: the
> sparse-rhs increment measures ~1.00x end to end, and the ordering's blocker is
> not "#649 vs a 1.71x we could otherwise take" — the 1.71x *is* the ordering.

### 18h. There is no in-repo increment left on #1008: the sparse-rhs lever is ~1.00x and the 1.71x cannot be decoupled from the bound loss (2026-08-14)

Two questions left open by §18g, both now settled — and both settled *against*
§18g's stated plan.

**1. The sparse-rhs entry points are not worth a call-site change (§11
retraction).** §18g sized them at **1.067x** from upstream's 14-LP panel and
named them "the next increment". feral PR #162's own end-to-end table, discopt
held fixed and only feral varying, on the QPLIB_1157 root LP:

| feral arm | wall | vs crates.io |
|---|---|---|
| 0.15.1 | 6.088 s | 1.00x |
| `main` + `dense_bump_max_dim = 4096` | 3.567 s | 1.71x |
| #162 (sparse-rhs) + bump + cap 0.10 | 3.607 s | **1.69x** |

**~1.00x for the sparse-rhs work**, against 10.8x on the `ftran` component.
Upstream also retracted issue #161's "93.1% of wall in the LU layer" as a
profile-attribution artifact — triangular solves are ~1% of that solve. A 10.8x
component speedup that moves the total by 1% *is* a direct measurement that the
component was 1% of the total. The 1.067x panel figure and the 1.497x stack are
component/panel numbers and must not be read as downstream predictions; §18g read
them that way and was wrong to.

**2. The 1.71x cannot be taken without the peel, and the peel loses the bound.**
`feral/src/lu/sparse_factor.rs:334` makes `symbolic.triangularized` a *hard
precondition* of the dense-bump route, so a "cap without peel" configuration does
not exist by construction. Upstream `895ef65` recorded that the paired
configuration is nonetheless safe (`peel + cap 4096` → PASSED on #649's
instance), which if true would have decoupled the speedup from the regression.
**It does not reproduce.** Same discopt commit `bce881ff`, unmodified, only feral
varying, patched exactly as `895ef65` describes:

| feral rev | ordering | cap | `bchoco06_illcond_..._649` | dense-bump firings |
|---|---|---|---|---|
| `e00aa706` | whole-basis AMD | 0 | **ok** | **0** |
| `e00aa706` | peel | 0 | FAILED — `Numerical` | 0 |
| `e00aa706` | peel | 4096 | **FAILED — `Numerical`** | **26** |
| `895ef65` | peel | 4096 | **FAILED — `Numerical`** | **26** |

Rows 1–2 reproduce `895ef65` exactly; row 3 refutes it; row 4 shows it is not a
regression from the three `src/lu/` commits since. §6: the firing counter is
patched into `sparse_factor.rs` immediately after `want_dense_bump`, so the
failing arm is one where the dense route demonstrably ran — 26 firings where the
route should fire and 0 where it should not. This also rules out a silently
inert patch, which would have reproduced peel-no-cap instead. It agrees with the
independent counter measurement on this branch (42/48 factorizations dense-bump,
`LuBumpDim` 26,656 → 2,036, `Numerical`).

Filed upstream as **feral #168**, with a comment on feral PR #162.

**Conclusion.** `DISCOPT_LU_TRIANGULARIZE` default-OFF is confirmed correct, and
it is not a conservative placeholder waiting on a graduation panel: the
configuration it gates is the one that loses a dual bound, and the speedup that
would justify the risk is unreachable without it. **#1008 has no remaining
in-repo increment.** Both candidate levers §18g left are closed — one measures
~1.00x, the other is inseparable from a certificate loss. §16's binding
conclusion ("closing #1008 requires work in feral or a different LU backend")
survives both, and its own prescription needs the same correction: it named
hyper-sparse triangular solves as half the fix, and upstream has now measured
that half at ~1.00x end to end. What is left is the numeric kernel, upstream.

### 18i. Upstream replaced the ordering lever entirely: threshold-Markowitz makes `DISCOPT_LU_TRIANGULARIZE` dead (#1008, 2026-08-14)

feral PR #172 (closing feral #171) changes `SparseLu::factor`'s **default** from
AMD-on-AᵀA + Gilbert–Peierls to **threshold-Markowitz** pivoting, which chooses
its column order *during* factorization. It therefore **ignores the `symbolic`
argument** beyond a dimension check. `LuPivoting::GilbertPeierls` restores the
old rule; the change is breaking and ships as 0.16.0 (not yet on crates.io —
0.15.1 is still the latest published version as of this entry).

**Audit of this repo for silent ordering substitution** (upstream's four hazard
classes, run at `bce881ff` and on this branch):

| hazard | shipped `main` | this branch (#1025) |
|---|---|---|
| `SparseLu::factor(&a, &sym, …)` with a deliberately chosen `sym` | **clean** — both sites pass plain `SparseLuSymbolic::analyze`, a throwaway AMD | **HIT** — both sites pass `analyze_with(a, LuOrderingParams { triangularize })` |
| assertions on `reach_visits()` | none | none |
| assertions on `used_dense_bump()` | none | **HIT** — `linsolve.rs:745` counts it |
| tests pinning a pivot row or permutation | none | none |

`main`'s `FeralLU::params()` builds with `..LuParams::default()`, so the new
`pivoting` field is not a build break either. **Shipped discopt has zero
exposure**; both hits are in the unmerged #1025 branch.

**Both hits resolve by deletion, not by `pivoting: LuPivoting::GilbertPeierls`.**
Pinning Gilbert–Peierls at those sites would keep `DISCOPT_LU_TRIANGULARIZE`
functioning, but §18h already established what it gates: the configuration that
loses bchoco06's dual bound, for a speedup unreachable without that loss. Keeping
a superseded lever alive by pinning the LU back to the slower rule is the "dead
flag" §3 forbids. The `used_dense_bump()` counter is the sharper case: under the
Markowitz default the dense-bump route is unreachable, so `LuDenseBumpFactorizations`
would read a permanent 0 — a non-vacuity guard that has itself gone vacuous, which
is exactly the instrument §6 exists to prevent. A guard that can only report 0 is
worse than no guard.

**De-risk run, measured here, not taken on report** (CLAUDE.md §8/§6). Worktree
at `bce881ff` + `[patch.crates-io] feral = { git = …, branch = "main" }`,
resolving to feral `c9c3adc`. §8 markers asserted present in the fetched source:
`pub enum LuPivoting`, `used_markowitz`, and `pivoting: LuPivoting::Markowitz`
as the `LuParams::default()` value (`src/lu/mod.rs:457`).

| check | result |
|---|---|
| `cargo test -p discopt-core --lib --no-fail-fast` | **602 passed, 0 failed** |
| `cargo test -p discopt-core --lib lp::` | **112 passed, 0 failed** |
| `bchoco06_illcond_scaled_path_recovers_bound_649` | **ok** |
| §6 probe `used_markowitz()` at the factor site | **46 true, 0 false** |

The 112/112 and the 46 firings reproduce upstream's numbers exactly, from a
different harness. The probe matters because both routes are silent: a passing
test proves nothing about *which* rule ran, and `--lib` captures `eprintln!` from
passing tests, so the first run of it read 0 firings for a purely instrumental
reason — `-- --nocapture` is required. Neither the probe nor the `[patch.crates-io]`
stanza is committed; they were reverted after the run.

**Consequence for #1008 and for PR #1025.** Upstream's own corpus reports fill
(`factor_nnz/nnz(B)`) geomean **2.77x → 1.06x**, never worse, faster on 15 of 16
bases, best case 1066.64 ms → 10.98 ms. That is the numeric-kernel work §18h
named as "what is left, upstream" — arriving as a default, requiring no discopt
lever at all. It supersedes the peel: it gets bchoco06 green *and* better fill,
where the peel had to trade one for the other. The 0.16.0 bump is therefore the
only remaining #1008 action, and it is a one-line version change plus a re-run of
the panel above, not a flag.

### 18j. feral 0.16.0 taken: fill 8.32x → 1.28x on discopt's own bases, and it will barely move MINLP wall (#1008, 2026-08-14)

feral 0.16.0 published; discopt is bumped from `0.15.1` (the git-rev pin on
`e00aa706` and the `DISCOPT_LU_TRIANGULARIZE` peel it carried are both gone — see
§18i for why the peel is deleted rather than kept alive with a `GilbertPeierls`
pin). Two arms, `feral-0.15.1` vs `feral-0.16.0`, each asserting the other arm's
marker **absent** in the loaded extension (§8).

**Which §5 regime applies was the first question, and it is not a formality.**
The `Cargo.toml` pin comment demanded `node_count` and `objective` be EXACTLY
unchanged for any bump that moves LU arithmetic. 0.16.0 fails that flatly: 28 of
208 exact comparisons drift, including `nvs02` 345 → 421 nodes and `nvs14`
129 → 839. But that rule was written for bumps that *claim* neutrality, and
0.16.0 replaces the pivoting rule outright — a different factorization means a
different rounding trajectory means different degenerate pivot choices. Held to
the letter, the rule forbids ever adopting a better LU. The pin comment now
splits the regime explicitly rather than being quietly reinterpreted.

**Gate 1, cert-clean: PASS.** 52-instance certifying panel, 103 checks, 0
violations, 51 oracle-backed. No `optimal` instance left `optimal`, no dual bound
above its reference optimum, all objectives inside abs 1e-6 / rel 1e-4. 47 of 52
node counts bit-identical; total nodes +1.3%.

**Gate 2 on the certifying panel would read FAIL — and that reading is wrong.**
Node counts improve on 0 instances and worsen on 5. That is the `DISCOPT_CUT_INHERIT`
shape, and taken at face value it kills the bump. It is the wrong instrument:
`baron-gap-plan.md` §1.3 measures node-LP at **0.06%** of that panel's wall, so a
fill improvement is invisible there by construction while the rounding reshuffle
it also causes is fully visible. The panel reports this change's cost and none of
its benefit. Quoting it as the verdict would repeat the §18g error — a component
ratio read as a downstream prediction — with the sign flipped.

**Gate 2 on an instrument that can see it: PASS, decisively.** 82 captured
relaxation LPs, counter-based (exact integers, load-independent):

| | 0.15.1 | 0.16.0 |
|---|---:|---:|
| aggregate fill (`LuFactorNnz/LuBasisNnz`) | 8.32x | **1.28x** |
| factor nonzeros | 4,597,979,997 | **836,938,845** (−81.8%) |
| factorizations | 11,628 | 13,558 |
| per-LP factor nnz | — | better 58, worse 18, within 1% 6 |
| objectives at the HiGHS optimum | 72/72 | 72/72 |

Upstream's 2.77x → 1.06x geomean reproduces here as 8.32x → 1.28x aggregate on
discopt's own bases — a different corpus and a different harness. Wall was
2751.9s → 493.0s; that is **not** a timing claim (the arms ran concurrently with
unequal contention, §9) and nothing here rests on it.

**Robustness is a wash, one status change each way.** `QPLIB_2480_rlt1` regresses
`optimal` → `numerical` (its fill was already 1.03x, so Markowitz had nothing to
win and only moved the trajectory: 276 → 823 factorizations, 31,798 phase-1
pivots, then a refusal). `QPLIB_1675_rlt1` goes the other way, `numerical` →
`optimal`, and larger: fill 12.84x → 2.00x, wall 446.56s → 35.42s — an LP 0.15.1
could not solve at all. `QPLIB_2823_rlt1` moves `numerical` → `iter_limit`, both
non-solutions. Cold primal fallbacks 28 → 30 across 82 LPs. `numerical` is a
refusal, not a false answer, and **neither arm produced a false verdict in 144
comparisons** — but the cold path is exactly the one §18b/§18c measured returning
a false `Unbounded` on `QPLIB_2170` and a false `Numerical` on `QPLIB_2738`, so
the fallback count is the number to watch on the next bump, not a footnote.

**What this does NOT do, stated plainly.** It does not close #1008's headline gap
and it will barely move discopt's MINLP wall. The LP layer is 0.06% of panel wall
on those families; an 81.8% cut in factor work inside 0.06% is unmeasurable
end-to-end. This is the right change on its own terms — it is the fill lever §18f
named, delivered upstream, and it is worth taking because the LP layer is now
genuinely fast rather than because the solver got faster. §16's conclusion stands
unchanged: discopt's gap to BARON is node-NLP and Python marshaling (69% and
82.5% respectively per `baron-gap-plan.md` §1.3), not LU throughput.

## 19. #1064 `squfl` primal gap: the structure was never *seen*, and the model rewrite trades the certificate for the incumbent (2026-08-19)

The `squfl` family (separable quadratic uncapacitated facility location) returned
incumbents 69–115% above optimum at 600 s. The issue framed this as "no incumbent
after 600 s"; that headline was already resolved by earlier work — every instance
now returns a feasible point — so the residual was *quality*, and it root-caused to
two things, neither of them the relaxation.

### 19.1 The recursion limit was silently reclassifying models (fixed)

A `.nl` body is a left-leaning `((((a+b)+c)+d)+…)` chain, one AST node per term.
`_extract_quadratic_terms._walk` and `_extract_linear_coefficients_sparse._walk`
both recursed once per term, so a long body raised `RecursionError` — and the
callers report that as *"not quadratic"* / *"not linear"*, which is
indistinguishable from a genuinely nonlinear body. A convex separable MIQP
therefore lost its Hessian and its semicontinuity structure **purely because it was
long**, with no diagnostic anywhere.

Measured over all 1610 MINLPLib `.nl` instances: 2 hit the quadratic walk
(`squfl025-040` at 1000 terms, `squfl015-080` at 1200), **17** hit the linear one
(`sporttournament30`…`40`, `edgecross10-060`…`24-057`, `autocorr_bern*`, `ibs2`).
After the work-stack rewrite the same scan reports `ERRORS 0`.

This is the §6 lesson in a production code path rather than an instrument: the
failure mode was not a wrong answer, it was a *capability that silently did not
exist*, reported as a legitimate classification.

Bit-identity (§5 regime 1) was preserved deliberately — children are pushed onto
the work stack in reverse so they pop in source order, keeping the floating-point
accumulation order unchanged. SHA-256 over the extracted `(Q, c, const)` and
`(terms, const)` matches on every corpus instance the recursive walk could process
at all; the only hashes that move are the ones that previously raised (`ibs2` goes
from `{ok: 1810, RecursionError: 1}` to `{ok: 1811}`).

### 19.2 The OA objective cut was the plain tangent; perspective-strengthening it is where the primal gain is

The OA master's objective epigraph row is the aggregate tangent
`grad^T x − eta <= rhs`. It discards the perspective of every separable convex
square over a semicontinuous variable. For binary `y` with a model row `x <= u*y`,
`y = 0` forces `x = 0`, so the Frangioni–Gentile cut `eta >= 2q z x − q z² y` is
valid using **only** `y ∈ {0,1}` — no box enters, so it is globally valid and safe
to keep across nodes (unlike anything read off a node relaxation's rows).

In row form, strengthening term `i` subtracts `q_i x̄_i²` from *both*
`coeffs[y_col_i]` and `rhs`. At `y_i = 1` the two cancel and the row is identical to
the plain tangent; at `y_i = 0` it reads `eta >= 0` instead of `eta >= −q x̄²`.

Measured at 60 s, interleaved, with a §6 counter proving the strengthening fired.
**Same bound and same node count on both arms** — the gain is purely primal:

| instance | applied | obj OFF → ON | bound | nodes | gap OFF → ON |
|---|---|---|---|---|---|
| squfl010-025 | 5750 | 214.111 → 214.111 | 214.111 | 371 | −0.0% → −0.0% (optimal both) |
| squfl015-060 | 9000 | 619.389 → 375.591 | 307.514 | 1055 | 68.9% → **2.4%** |
| squfl015-080 | 10800 | 821.264 → 465.460 | 296.717 | 703 | 104.0% → **15.6%** |
| squfl025-040 | 11000 | 423.980 → 233.818 | 127.705 | 1023 | 114.9% → **18.5%** |

Structure population: exactly **20 of 1610** corpus instances (§2 — the fix is to
the class, and every other instance is a provable no-op because the detector
returns an empty term list).

### 19.3 FALSIFIED: the model rewrite. It is strictly better on the primal and strictly worse on the certificate

The obvious formulation-level move is to lift `q x²` to `q s` and add `x² <= s y`.
Its primal effect is larger than the cut's — 68.9% → 0.00%, 114.8% → 0.00%,
114.9% → 0.05% — and it is tempting for exactly that reason.

It is not shipped. The lifted row `x² <= s y` is a **nonconvex** quadratic
constraint, so the rewritten model leaves the convex relaxation and routes to the
spatial path. The dual bound came out **looser on every** `squfl` instance. That is
a certification regression under §5, and §1 settles it: a change that improves the
incumbent while degrading the certificate is a regression, however large the primal
number looks. The primal-only figure is the trap here — read alone it reports a
1000× improvement on a change that makes the solver worse at its actual job.

### 19.4 FALSIFIED: a spatial row separator for the same structure

A separator that added perspective rows at the node relaxation was built and
measured `sep_calls = 0` / `nodes_via_mccormick = 0` on reformed models — it never
fired, because the reformed model does not reach the McCormick path where the
separator was installed. Dropped rather than relocated, since §19.3 removes the
reformulation it depended on.

### 19.5 The §5 graduation panel: `DISCOPT_PERSPECTIVE_OA_CUT` is now default-ON (2026-08-20)

Twenty instances, flag OFF then ON back-to-back per instance (interleaved, §9),
60 s each, threads pinned. Every `squfl*` in the corpus, plus the four
non-`squfl` instances the detector fires on (`st_miqp2`, `st_miqp4`, `st_test3`,
`meanvar-orl400_05_e_8`) and the two it does *not*
(`watercontamination0202/0303`, `persp_applied = 0`) as the neutral control.
All twenty are MINIMIZE, so a **higher** bound is tighter.

| instance | applied | gap OFF | gap ON | bound OFF | bound ON | nodes OFF→ON |
|---|---|---|---|---|---|---|
| squfl010-025 | 5750 | −0.000 | −0.000 | 214.11 | 214.11 | 371→371 |
| squfl010-040 | 8000 | −0.000 | −0.000 | 240.60 | 240.60 | 293→293 |
| squfl010-080 | 8000 | −0.000 | −0.000 | 509.71 | 509.71 | 593→593 |
| squfl015-060 | 9000 | 68.945 | **2.446** | 307.51 | 307.51 | 1055→1055 |
| squfl015-080 | 10800 | 104.047 | **15.645** | 296.72 | 296.72 | 703→703 |
| squfl020-040 | 8800 | 85.592 | **0.000** | 183.76 | **184.14** | 1503→1535 |
| squfl020-050 | 9000 | 129.453 | **2.284** | 165.44 | 165.44 | 1151→1151 |
| squfl020-150 | 18000 | 114.749 | **14.082** | 253.21 | **346.46** | 31→31 |
| squfl025-025 | 8125 | 44.806 | **−0.000** | 141.58 | 141.58 | 2303→2303 |
| squfl025-030 | 10500 | 29.997 | **0.543** | 151.04 | 151.04 | 1503→1503 |
| squfl025-040 | 11000 | 114.854 | **18.488** | 127.70 | 127.70 | 1023→1023 |
| squfl030-100 | 15000 | 317.248 | **14.137** | 123.89 | **189.55** | 3→3 |
| squfl030-150 | 9000 | 243.690 | **10.209** | 158.93 | **238.17** | 3→3 |
| squfl040-080 | 9600 | 498.274 | **35.569** | 91.49 | **132.45** | 3→3 |
| meanvar-orl400_05_e_8 | 800 | 6.433 | 6.433 | 98.12 | 98.12 | 31→31 |
| st_miqp2 | 11 | 0.000 | 0.000 | 2 | 2 | 0→0 |
| st_miqp4 | 6 | 0.000 | 0.000 | −4574 | −4574 | 0→0 |
| st_test3 | 1 | 0.000 | 0.000 | −7 | −7 | 0→0 |
| watercontamination0202 | 0 | — | — | none | none | 0→0 |
| watercontamination0303 | 0 | — | — | none | none | 0→0 |

```
gap: better=11 worse=0 unchanged=7
dual bound: tighter=5 looser=0
nodes: more=1 fewer=0
CERT-CLEAN: PASS   NET-POSITIVE: PASS   GRADUATE: YES   CHECKS_EXECUTED 20
```

**Bar 1 (cert-clean).** No bound above its reference optimum, no `optimal →
non-optimal` regression (the three `optimal` instances stay optimal, bit-equal),
no dual bound loosened, no arm losing a bound the other had, and no `infeasible`
where the OFF arm found a point. **Bar 2 (net-positive).** The gap improves on
11 of 18 scored instances and worsens on none; five dual bounds tighten. The two
instances #1064 reported as returning *nothing* after 600 s — `squfl020-150` and
`squfl025-040` — both return incumbents at 60 s.

The single node-count change (`squfl020-040`, 1503→1535) is expected and not a
§5 regime-1 violation: this is a bound-*changing* flag, so the tree is allowed to
move. `watercontamination*` applies zero cuts and is byte-identical on both arms,
which is the §6 evidence that "no change" there means "the detector correctly
declined", not "the panel never ran".

Per §5 the flag is now **default-ON**, with `DISCOPT_PERSPECTIVE_OA_CUT=0` kept
as the opt-out and the plain-tangent path intact.

### 19.5 The round-budget regression test measured the machine, not the budget (2026-08-29)

`test_1064_round_fix_resolve.py::test_the_time_budget_is_what_bounds_the_spend`
failed on `main` at `1b8866ec` with *"budget frac made no difference: 7 attempts
budgeted vs 7 unbudgeted"*, and reproduced on a rerun. It was not a solver
regression, and the budget it tests is correct.

The test ran two full `solve_model` calls with a stub that declines and sleeps
0.3 s, and compared how many times each reached the round gate — the budgeted
arm bounded by `_ROUND_TIME_FRAC * T = 2.0 s` (7 attempts), the unbudgeted arm
expected to run to the solve's own limit. That comparison is only meaningful if
the search reaches the gate many more than 14 times, and how often a
wall-clock-bounded search reaches it is not reproducible. Ten identical runs of
the fixture, one box, one build, arms interleaved (load1 4.5–10.0):

| frac | gate visits | nodes | status | wall |
|---|---|---|---|---|
| 0.2 | 7, 7, 0, 0, 0 | 97, 97, 0, 0, 0 | optimal | 8.6–6.0 s |
| 1e6 | 15, 15, 15, 0, 0 | 31, 31, 31, 0, 0 | feasible/optimal | 10.9–5.7 s |

The `0` rows are the important ones: the model does not always route to
`_solve_miqp_bb` at all, so on this box the test usually failed its own
"round-fix-resolve never ran" precondition — the invariant was untested on any
developer machine and only ever evaluated on CI, where it eventually drew
`7 vs 7`. Entering `_solve_miqp_bb` **directly** is by contrast exactly
reproducible (5/5 runs: 91 nodes, `optimal`, budget built once, gate consulted
once), which is what the wiring test now does.

**Retraction (§11).** Mid-session I told the owner the CI failure was "a real
failure, not flaky" because a rerun reproduced it. The reproduction was real; the
conclusion was wrong. Reruns land on similar runners, so reproducing a
load-sensitive race is not evidence against it being one.

The rule (`_RoundBudget`, `_round_fix_resolve_attempt`) was lifted out of
`_solve_miqp_bb`'s node loop so it can be exercised directly instead of raced to.
That is strictly more coverage, not less: the extracted gate also pins the
opt-out and no-incumbent conjuncts, which nothing tested before, and the whole
file dropped from ~25 s (two flaky tests) to 0.7 s (none). Five mutations of the
production rule — dropping the time term from `may_attempt`, handing an attempt
the solve deadline, dropping the `enabled`/`has_incumbent` conjuncts, not
recording the spend, and bypassing the gate entirely — each fail at least one
test.

**Standing lesson.** A wall-clock-bounded search is not a fixture. Any assertion
that counts how many times such a search reaches an internal gate is measuring
the machine; extract the rule and assert on it directly.


## 20. #1061 root cuts: `DISCOPT_NLPBB_ROOT_CUTS` is sound but not helpful — stays OFF (rejected 2026-08-20)

> **Superseded by §21 (2026-08-20).** The re-run panel below — on a build with
> the #1062 stall fix and the #1098 `gap_certified` correction — passes both §5
> bars, and the flag is now default-ON. §20 is retained as the record of the
> verdict *for the build it was run on* (CLAUDE.md §11), not as current policy.

**Why this was run.** #1061 reports discopt's root dual bound sitting 27.1x above
the reference optimum on `syn40m` and 8.5x on `rsyn0840m`. The root cutting-plane
stage (#781, `DISCOPT_NLPBB_ROOT_CUTS`, default-OFF) was the nearest built lever,
and PR #1094 fixed a real defect in it: `_root_cuts` switched *itself* off on any
model whose objective arrived as a vector block, so on that whole class the stage
had never run at all. With the stage actually reaching those models, the §5
graduation gate became answerable for the first time.

**The panel.** Flag ON vs OFF over the in-repo corpus, 6 shards, 60 s/instance,
**153 instances with complete pairs in both arms**.

| gate | result |
|---|---|
| dual bound | tighter 31, looser **13**, flat 109 |
| primal shortfall | better 4, worse **6**, same 93 |
| total wall | OFF 3015.9 s, ON 3020.8 s (+0.2%) |
| bound above its reference optimum (`RAISED`) | **0 instances** |
| certification regression | **`clay0303hfsg`** (`gap_certified` True → False) |
| **CERT-CLEAN** | **FAIL** |
| **NET-POSITIVE** | **FAIL** |

**Both bars fail, and they fail for different reasons.** Soundness in the narrow
sense holds — no bound in the ON arm rose above its reference optimum in 153
pairs, so the cuts are valid. But §5's cert-clean bar is broader than "no false
bound": `clay0303hfsg` regresses from `gap_certified=True` to uncertified, which
is a certificate lost, and that alone disqualifies the flag.

**The looseners are a coherent family, not scatter.** Every large regression is
`clay*`:

| instance | dual bound OFF → ON |
|---|---|
| `clay0205hfsg` | 1889.99 → **26.89** |
| `clay0304m` | 39347.56 → **6545.0** |
| `clay0303hfsg` | 26669.11 → **19513.27** |
| `clay0304hfsg` | 3915.87 → **1840.0** |

plus two primal losses on the same family (`clay0205m` 0.0% → 233.4% short,
`clay0304m` 0.0% → 4.62% short). A cut round cannot *loosen* a valid dual bound
on its own; what it does is spend root budget and move the trajectory, and on
`clay*` that trade is severely negative. The gains, meanwhile, are real but small
and concentrated in the `fo*`/`m*`/`no*` families (e.g. `fo9_ar2_1` 14.34 → 20.38,
`fo7_ar3_1` 8.08 → 11.10) — 31 tighter against 13 looser is not the broad win §5
asks for, and the primal column is net *negative*.

**Verdict: `DISCOPT_NLPBB_ROOT_CUTS` stays default-OFF.** This is the
`DISCOPT_CUT_INHERIT` shape exactly — sound is not the same as helpful — and §5
says a cert-clean-but-neutral flag stays off. Here it is not even cert-clean. PR
#1094's fix ships on its own merits (the stage silently disabling itself on
vector-block models is a defect whether or not the flag ever graduates) and
carries `Contributes to #1061`, not `Closes`.

**What this does NOT establish.** It does not say root cuts are the wrong idea for
the syn/rsyn class; it says *this* implementation, at this budget, is not ready to
default on. And it leaves #1061's headline untouched: the 27.1x on `syn40m` is a
property of the big-M relaxation itself. The lever that *does* move #1061 turned
out to be elsewhere entirely — a free-column pricing defect in the simplex that
silently disabled OBBT's exact-LP oracle on this whole class — and it ships on its
own branch (`fix/1061-phase1-open-column`), not here.

### 20.1 Amendment: the `clay*` looseners were a *budget* sink, not a bound effect (#1062, 2026-08-20)

§20's REJECT verdict stands as the record of the flag *as it stood on that
panel*. What it could not say is **why** `clay0303hfsg` lost `gap_certified` under
the flag, and the answer turns out not to be about the cuts at all.

A valid cut cannot loosen a valid dual bound, so "the flag produced looser
bounds" was never explicable by the cuts' arithmetic. The entry experiment
(hypothesis, prediction and kill criterion fixed in advance —
`scratchpad/entry_1062_budget.py`) measured the *stage*, not the bound, at
`time_limit=30` (stage budget 6.0 s):

| instance | rounds | improving | cuts kept | wall | % of TL | stop |
|---|---|---|---|---|---|---|
| clay0205hfsg | 16 | **0** | **0** | 7.43 s | **25.8%** | budget |
| clay0303hfsg | 19 | **0** | **0** | 2.39 s | 8.0% | no_cuts |
| syn40m | 20 | 18 | 61 | 1.12 s | 3.7% | no_cuts |
| rsyn0840m | 30 | **30** | 69 | 3.50 s | 11.4% | **rounds cap** |

The loop broke only on budget, the round cap, a dead LP, or a round selecting no
cuts. None detects the case that costs: cuts keep being **found** while the bound
never **moves**. `productive_rounds` counts rounds that *chose* cuts — 16 of 16 on
`clay0205hfsg` — so it reads as fully productive throughout. Both `clay*` hold
their root LP bound at exactly `0.0` in every trace entry and then have every cut
discarded by the end-of-loop quality gate. `clay0205hfsg` spent a quarter of the
whole solve's time limit to hand back nothing, and overran its own 6.0 s budget
(the budget check sits at the loop top, so a round starting under budget runs to
completion). **The flag's looser bounds were time-starved bounds.**

The same table shows the converse defect: `rsyn0840m` stopped on `ROUNDS = 30`
while improving in 30 of 30 rounds, 43% of its budget unspent. The round cap was
binding precisely on the instances the stage helps.

Both are fixed on `fix/1062-root-cut-stall` (a two-round stall guard against the
quality gate's own tolerance; the cap demoted to a runaway backstop; per-round
`bound_trace` / `improving_rounds` / `stop_reason` on `RootCutResult`, which the
loop previously lacked entirely). After it: `clay0205hfsg` 7.43 s → **0.10 s**,
`clay0303hfsg` 2.39 s → **0.10 s**, both with identical output; `syn40m`
unchanged at bound 366.4403; `rsyn0840m` runs to natural cut exhaustion at 36
rounds with its bound tightening 861.77 → 852.11.

**Status.** This does *not* retroactively graduate the flag. It removes the
measured cause of the panel's `cert-clean` failure, which makes a re-run
meaningful; the flag stays default-OFF until a fresh §5 panel clears **both**
bars. Until that panel is scored, §20's verdict is the operative one.

**Update (2026-08-20): that panel has now been scored — see §21.** It clears both
bars and the flag graduated default-ON, so the sentence above ("§20's verdict is
the operative one") no longer holds; it was correct when written and is retracted
here rather than edited, per CLAUDE.md §11.

## 21. #1061/#1062 root cuts, re-run: `DISCOPT_NLPBB_ROOT_CUTS` passes both §5 bars and graduates ON (2026-08-20)

**Why this was re-run.** §20 rejected the flag on a panel that failed *both* §5
bars. Two things changed underneath that verdict:

1. **#1062's stall fix.** The convex B&B's stalled node abstained instead of
   being excluded (#1082), which suppressed the sub-NLP primal heuristic on
   exactly the convex models this flag targets. §20's panel predates it.
2. **#1098's `gap_certified` narrowing.** The §20-era field carried *two*
   meanings. On the OA/mip-nlp route it meant "the printed gap is arithmetically
   valid" — true at a 430% open gap. On the NLP-BB route (`solver.py`) any
   `feasible` exit cleared it, i.e. "the gap is *closed*". A differential panel
   whose two arms take different routes therefore reads a **false certification
   regression** whenever the flag changes which route wins.

**The panel.** Flag ON vs OFF over the in-repo corpus, 6 shards, 60 s/instance,
arms adjacent per instance (§9), threads pinned. **153 instances with complete
pairs in both arms**; every shard reported a non-zero executed-comparison count
and no tracebacks (§6).

| gate | result |
|---|---|
| dual bound | tighter **22**, looser 9, flat 122 |
| primal shortfall | better **6**, worse **1**, same 102 |
| node count | OFF 111 702 → ON 96 860 (**−13.3%**); 58 instances >2% fewer, 2 more, 93 within 2% |
| total wall | OFF 5744.0 s, ON 5774.8 s (**+0.5%**) |
| bound above its reference optimum (`RAISED`) | **0 instances** |
| certification regression (#1098 semantics) | **0 instances** |
| proven-optimal count | OFF 69, ON 69 (no certificate lost) |
| **CERT-CLEAN** | **PASS** |
| **NET-POSITIVE** | **PASS** |

**The two flagged regressions were artifacts, and this was verified rather than
asserted.** The raw scorer — reading the pre-#1098 `gap_certified` straight out
of the logs — flagged `rsyn0805m02m` and `syn40m`. Re-adjudicating all 153 pairs
under the corrected predicate (`gap_certified == (status == "optimal")`, which is
what #1098 makes both routes report) gives:

```
raw cert regressions (pre-#1098 logs) : ['rsyn0805m02m', 'syn40m']
cert regressions under #1098 semantics: []
  -> INVENTED by the loose reading    : ['rsyn0805m02m', 'syn40m']
  -> HIDDEN by the loose reading      : []
```

The "hidden" column matters: a *narrowing* can in principle expose a regression
the loose reading concealed (OFF stays certified, ON turns out to have been an
open gap mislabelled). Zero were hidden, so the correction only removes false
positives here.

`syn40m` is the clearest case — the ON arm is better on **every real quantity**
and lost only the label:

| arm | route | status | objective | bound | `gap_certified` |
|---|---|---|---|---|---|
| OFF | OA (`mip_nlp_trace` present) | feasible | 55.713 (17.7% short) | 292.22 | True |
| ON | NLP-BB (no `mip_nlp_trace`) | feasible | **67.71325557 = the optimum** | **187.88** | False |

Turning root cuts on makes the NLP-BB answer better, so it wins route selection
(`_route_is_better`) and inherits that route's stricter label. Confirmed by direct
re-run of both instances × both arms on the post-#1098 build: both report
`gap_certified=False` in *both* arms, so there is no regression, and ON is better
on both (`syn40m` objective 55.713 → 67.713; `rsyn0805m02m` 1059.08 → 1226.09
with a tighter bound 4386.54 → 4373.31).

**Verdict.** Both bars pass, so under §5 the flag graduates default-ON, keeping
the `DISCOPT_NLPBB_ROOT_CUTS=0` opt-out and the legacy path intact. This
supersedes §20's rejection: §20 remains the correct verdict *for the build it was
run on* (pre-#1062-stall-fix, pre-#1098), and is retained per §11 rather than
edited away.

### 21.1 The panel was not the last gate: the smoke suite caught a false certificate the corpus could not see

Flipping the default to ON with the panel in hand failed `pytest -m smoke`
immediately, on `test_945_nlp_box_and_gap_closure.py::
test_default_solve_path_does_not_certify_a_super_optimal_incumbent`:

```
flag=0: status=optimal obj=2.999999998034762  super-optimal by 1.965e-09
flag=1: status=optimal obj=2.998978315393790  super-optimal by 1.022e-03
```

on the MindtPy constraint-qualification fixture `(x-3)^2 <= 50(1-y)`, whose
exact optimum is 3.0. This is CLAUDE.md §1's worst class — a certificate on a
point that cannot exist — and it outranks §5: a flag does not graduate onto the
default path while an in-repo gate demonstrates one.

**Why 153 corpus instances missed it.** The corpus oracle (`minlplib.solu`) is
compared at a relative tolerance, and the error here is 3.4e-4 relative. It is
also structural rather than statistical: it needs an *active degenerate* row,
where the constraint residual is quadratic in the variable error, so 1e-6 of
violation is 1e-3 of `x`. A panel over well-posed instances has no power against
it. The lesson generalizes past this flag: **a corpus panel is a §5 instrument,
not a §1 one.** The §1 gates are the suites.

**The chain, measured end to end** (`$SP/cq_probe{2,3,4}.py`, each printing an
executed-comparison count per §6):

1. Root cuts tighten the root LP to 2.998981, so the node-1 NLP relaxation
   returns `x = 2.998978` at `y = 1 - 2.1e-8`. That point is genuinely feasible
   — the sliver of `y` buys `50 * 2.1e-8 = 1.05e-6` of big-M slack, exactly
   covering `(x-3)^2 = 1.04e-6`.
2. `y` is integral within tolerance, so the node is integer-feasible and the
   relaxation value equals the incumbent: gap 0 at **node 1**.
3. Snapping `y` to 1 removes that slack: row 0 is now violated by 1.044e-6. The
   1e-6 exit gate sees 9.938e-7 after the `1e-9 * 50` term-scaled forgiveness,
   and passes it by 0.6%.
4. The terminal refine (integers fixed, `bound_relax_factor=0`) returned
   `x = 2.9999999931`, violation **4.8e-17** — the honest point — and the
   adoption rule **rejected it**, because its objective differs from the
   incumbent's by 1.02e-3, outside the rule's ±1e-4 window.

Step 4 is the defect, and it is not specific to root cuts: the ±1e-4 objective
window is the wrong arbiter when the two points *disagree*, because it keeps
whichever is more optimistic — including one the refine has just shown to be
unattainable at that integer assignment. Root cuts only supply a relaxation
tight enough to land on the degenerate boundary, where the flaw is visible.

Fixed on `feat/1061-1062-graduate-nlpbb-root-cuts`: when the objectives disagree
beyond the window, both points are measured with the same arbiter the exit gate
uses (`_nonlinear_point_excess`, declared rows and declared box) and a *strictly
more feasible* point is adopted even though its objective is worse. It cannot
admit a point the exit gate would refuse, and when the two agree within the
window the original branch still applies — so nothing moves on a healthy solve.
After it, flag ON reports `obj = 2.9999999931` with row 0 residual 4.7e-17, and
`pytest -m smoke` is 1079 passed / 1 skipped / 2 xpassed.

**Does this invalidate §21's panel?** No, and the direction matters: the fix can
only ever *replace* an incumbent with a strictly more feasible one at the same
integer assignment. It cannot loosen a bound, cannot manufacture a bound above
an optimum, and cannot turn a certified instance uncertified. The panel's
cert-clean verdict therefore survives the change a fortiori. Its net-positive
column is measured on the pre-fix build and is not re-run; the fix touches only
the terminal point of a solve, not the search.

### 21.2 `rsyn0820m02m`: the row-count hypothesis is falsified, and the cost is per-node NLP effort

The graduation panel's net-positive column is a population statistic, so it is
worth naming what it is averaging over. Re-running #1062's four named instances
plus `rsyn0820m02m` (which is **not** in the 153-instance panel) at 60 s, both
arms adjacent per instance, found one clear loser:

| instance | OFF | ON |
|---|---|---|
| `rsyn0820m02m` (max, opt 1092.09) | obj 244.20, 31 nodes | obj **-39.53**, **3 nodes** |

**Hypothesis (stated before measuring):** the stage appends so many rows that
per-node NLP cost collapses throughput. **Kill criterion:** if the applied cut
count is small relative to the model's rows, the hypothesis is wrong.

**Falsified.** `$SP/probe_0820.py`, `CHECKS_EXECUTED 4`:

```
rsyn0820m02m: declared rows=1074 vars=510
root-cut stage: n_cuts=69 rounds=26 productive=26 stop='budget'
                lp_bound=5220.25 stage_wall=10.6s
rows after solve = 1143  (+69 cut rows, 6.4% of the model)
result: status=feasible obj=-39.531 bound=4786.63 nodes=3 subnlp=1 wall=60.2
stage consumed 10.6s of a 60s budget (18%)
```

69 cuts on 1074 rows cannot be a 10x throughput effect by row count. The stage's
own wall is bounded and behaved (`max(2, min(10, 0.2*T))`, 10.6 s of 60 s), so
the lost time is in the tree: **~1.9 s/node OFF vs ~16 s/node ON**, an ~8.5x
per-node cost increase bought by 6.4% more rows. The cost is therefore in how
much *effort* each node NLP spends on the cut-tightened feasible region, not in
its size. Every round here was productive (26/26) and the loop stopped on
budget, so the stall detector had nothing to catch — this is not the #1062 stall.

No soundness consequence: the bound stays valid (4786.63 >= 1092.09) in both
arms, and the panel's `RAISED` count is 0.

**Not acted on in the graduation PR, deliberately.** Any change to the cut cap
or the per-node effort limit is itself bound-changing and would invalidate the
153-instance panel that the graduation rests on. Recorded here as the entry
measurement for whoever tunes the cut stage next: the lever to test first is
per-node NLP iteration effort under appended GMI/c-MIR rows, **not** the cut
count.

> **Falsified (2026-08-20, issue #1066 — "the throttled MILP default cut budget
> is what leaves the convex rsyn/syn/squfl class uncertified").** After #1102
> made the root cut loop warm dual re-optimize each round instead of cold-solving
> the augmented LP, the Python-facing default (`root_cuts=16, cut_rounds=1,
> cut_select=False`, set in the three pyo3 signatures at
> `crates/discopt-python/src/lp_bindings.rs:1009/1116/1238`) looked like a
> throttle held on for a cost that no longer exists. Entry experiment before any
> implementation, per Dev-Philosophy #4: `/tmp/mono/ab_cutbudget.py`, 16 instances
> drawn by *family prefix* (`rsyn`, `syn`, `clay`, `batchs`, `fac`) filtered to
> those with a `minlplib.solu` oracle and evenly strided — never by naming
> instances (#2) — arms interleaved per instance (#9), arm B injecting
> `root_cuts=500, cut_rounds=15, cut_select=True` by wrapping the Rust entry point
> so nothing but the four cut arguments differs. **Kill criterion, stated in the
> script docstring before the run:** arm B must certify *strictly more* instances
> than arm A with no soundness or certification regression.
>
> **Result: `SOLVED_OPTIMAL A=5 B=5 of 16`, `CHECKS_EXECUTED 72`, `VIOLATIONS 0`.**
> Load gate (#9): the first attempt was **discarded** at load 73 on a 14-core box
> — an unrelated `cargo test --workspace --release` held ~10 cores and starved the
> probe to 26% CPU — and re-run behind a load gate; the scored run sampled
> load1 every 60 s across its own execution: `n=23 min=3.90 median=9.59 max=12.25`.
>
> The budget **does** work as a cut mechanism, and that is exactly what makes the
> negative result informative. Over the 9 instances with a nontrivial root gap,
> arm B closes a **median 23.7%** of arm A's root gap (max 36.3% on
> `rsyn0820m02m`, 33.7% on `rsyn0815m02m`). It is not enough to matter, because
> the gaps it is closing are enormous:
>
> | instance | root gap, arm A | root gap, arm B | closed | nodes in 60 s |
> |---|---|---|---|---|
> | rsyn0805m02m | 57.1% | 44.7% | 21.6% | 3 |
> | rsyn0810m02m | 165.7% | 136.7% | 17.5% | 7 |
> | rsyn0815m02m | 147.4% | 97.7% | 33.7% | 15 |
> | rsyn0820m02m | 338.4% | 215.5% | 36.3% | 3 |
> | rsyn0830m02m | 399.8% | 304.5% | 23.8% | 3 |
> | rsyn0840m02m | 516.2% | 394.0% | 23.7% | 3 |
> | syn30m02m | 126.9% | 95.7% | 24.6% | 31 |
> | syn40m02m | 312.7% | 312.7% | 0.0% | 15 |
>
> Arm B is also **not** uniformly sound-and-neutral elsewhere: it lost
> `clay0205hfsg`'s incumbent outright (`feasible` obj 24140 → `time_limit` obj
> `nan`, on *more* nodes: 337 vs 305), returned a `nan` dual bound on `faclay35`
> where arm A returned a finite one, and cost 21x wall on the easy `fac1`
> (0.1 s / 0 nodes → 2.1 s / 7 nodes) and 5.7x on `syn15m02m`. Cert-clean but
> neither broadly helpful nor free — the `DISCOPT_CUT_INHERIT` disposition
> exactly (Dev-Philosophy #5: sound ≠ helpful). **The default is unchanged.**
>
> **Re-scope — the measurement points somewhere else entirely.** The column that
> matters in the table above is the last one. The rsyn family explores **3 nodes
> in 60 seconds**: 0.05 nodes/s, i.e. ~20 s *per node*, and `faclay35` manages
> 1 node in 62 s (0.02 nodes/s). No root cut of any strength closes a 394% gap
> from three nodes; SCIP and BARON solve these same instances in 0.2–48 s. The
> binding constraint on the #1066 class is **per-node cost on the convex-MINLP
> route**, not cut strength, and any further work on root cut budgets for this
> class is premature until node throughput is within an order of magnitude of the
> reference solvers. Measured throughput on the 11 time-limited rows for whoever
> picks this up: 0.02–0.24 nodes/s on rsyn/faclay, 0.25–0.51 on syn30/40m02m,
> 1.77 on batchs101006m, 5.05 on clay0205hfsg.

### #1066 — the per-node cost was a duplicated LU factorization (2026-08-20)

Following the re-scope above ("the binding constraint on the #1066 class is
per-node cost"), the hypothesis was that the convex-MINLP route pays a redundant
basis factorization per node. Stated before the run: the entry experiment is a
`DISCOPT_PROFILE` counter split of one 60 s solve of `rsyn0820m02m`; the kill
criterion is that the LU-factorization count per warm dual solve comes out at or
below 1.0, which would mean nothing is duplicated and the cost is intrinsic.

It did not. The counters read **2.15–2.20 sparse LU factorizations per warm dual
solve** across three reps — one full refactorization per node beyond the one the
warm dual start already pays. The extra one was `reduced_cost_fix`
(`bnb/milp_driver.rs`): it rebuilt the node basis' sparse LU from scratch and
btran'd it to recover `y = B⁻ᵀc_B`, a vector **the LP solve had already computed
from its own final factorization** and exported as `LpSolve::dual` (both the warm
dual and the cold primal fill it on `Optimal`). A new `Phase::RedCostFix` timer —
the pass runs *outside* `NodeLpSolve`, which is why every earlier phase split
booked it as unaccounted wall — priced it directly.

#### Retraction: the first cut of this change broke the node Farkas fathom

The first implementation unscaled `sol.dual` **in place** right after the node
solve, on the reasoning that reduced-cost fixing was "the only consumer below".
It is not. The `LpStatus::Infeasible` arm verifies the node's Farkas ray against
the *scaled* batch CSC (`ctx.csc`, `ctx.sb`, scaled node bounds), so an unscaled
ray no longer matches the data it is checked against: every verification failed
and provably empty nodes stopped being fathomed. Sound — the arm's failure path
returns a non-pruning `-inf` bound, so no optimum can be cut — but ruinous, and
it wedged a `pytest -m smoke` run for >20 minutes inside a single MILP solve.

The numbers first recorded here (2.17/2.15 LU per warm solve, +71% throughput)
were measured on that broken cut and are **withdrawn**; the table below replaces
them. The A/B gate did not catch it because both arms carried the in-place
unscale — `DISCOPT_MILP_RC_FIX_REFACTOR` only selects the dual *source*, so a
defect in code common to both arms is invisible to it. Lesson for the next
differential flag: an env-var A/B tests the branch, never the code the branch
shares with its control.

The shipped fix leaves `sol.dual` in solve-space and unscales a private copy at
the reduced-cost-fixing call site. `rc_fix_dual_unscale_does_not_break_the_node_farkas_fathom`
pins it: an LP-infeasible model with non-unit row factors that the root ray must
refute in one node — 3 nodes with the in-place unscale restored, 1 without.

#### Measurement (corrected)

Interleaved A/B, same binary, three reps, `DISCOPT_MILP_RC_FIX_REFACTOR=1`
(legacy) vs default (reuse), `rsyn0820m02m`, 60 s, thread-summed phase times.
Load average 9.9 before / 13.0 after — elevated by an unrelated process on the
box, which is what the `pA2` outlier is:

| metric | A r1 | A r2 | A r3 | B r1 | B r2 | B r3 |
|---|---|---|---|---|---|---|
| `RedCostFix` ms | 103,246 | 79,893 | 105,083 | **339** | **338** | **328** |
| node LPs (`NodeLpSolve`) | 34,239 | 18,990 | 38,335 | 59,433 | 61,266 | 58,559 |
| `DualWarmSolves` | 37,668 | 22,499 | 41,912 | 64,241 | 66,376 | 64,019 |
| `LuSparseFactorizations` | 81,368 | 49,551 | 89,912 | 71,373 | 73,525 | 71,167 |
| LU per warm solve | 2.16 | 2.20 | 2.15 | **1.111** | **1.108** | **1.112** |
| wall (s) | 60.32 | 60.50 | 60.20 | 60.31 | 60.21 | 60.23 |
| objective | −39.531007606231725 in all six runs | | | | | |
| MINLP nodes | 3 in all six runs | | | | | |

Reduced-cost fixing went from **96.1 s of mean thread time to 335 ms** (~290x).
Node-LP throughput rose from a median 34.2k to 59.4k in the same ~60.3 s wall,
**+74% on medians** (+96% on means; arm A's spread is 10.2k against arm B's 1.4k,
so the median is the number to quote). The LU-per-warm-solve ratio is the robust
statistic here — it is a within-run ratio, immune to the load that moved arm A's
absolute counts, and it drops 2.15–2.20 → 1.108–1.112.

Wall is pinned at the limit on this instance either way, so throughput — not wall
— is the signal; scoring time-limited and certified instances together is how a
panel like this reads as "no change".

Bound-neutrality gate (`python/tests/data/minlplib_nl`, 66 instances, 20 s each,
scored only where BOTH arms certify): `CERTIFIED_BOTH 48`, `CHECKS_EXECUTED 96`,
`DRIFT 0` — every certified objective within 1e-9 relative and every node count
exactly equal.

Disposition: bound-neutral (§5 regime 1), default-on, legacy path retained as the
fallback for an empty `dual` and reachable via `DISCOPT_MILP_RC_FIX_REFACTOR=1`
for the differential test.

## 22. #1066 route budget: the wall is blind, but capping the OA master is not the fix (2026-08-21)

The #1066 reporter panel — 15 convex MINLPLib instances (`syn`/`rsyn`, `squfl`,
`portfol_classical050_1`, `alan`) at default settings, `gap_tolerance=1e-4`,
`time_limit=60` (600 s on two rows) — still returned **7/15 certified** on
`main` after every linked work item (#1059–#1065) had merged. Two hypotheses
were raised against those rows. One survived; the other is falsified here, and a
claim published earlier in the same session is retracted with it.

### 22.1 Confirmed: the fixed 50 % route budget cuts off routes that are converging

`_CONVEX_ROUTE_BUDGET_FRACTION = 0.5` hands the #1059 auto-route half the
caller's limit and reserves the rest for the spatial fallback. Attribution on
`syn40m` (three arms, interleaved, load recorded):

| arm | status | objective | wall |
|---|---|---|---|
| default (route gets 30 s) | feasible, uncertified | 58.210 | 61.7 s |
| route budget = 1.0 | **optimal** | 67.71325583 | 32.9 s |
| explicit `solver="mip-nlp"` | **optimal** | 67.71325583 | 42.3 s |

Published optimum 67.713256. OA certifies this instance at 33–43 s; the wall
lands at 30 s, so neither path certifies and the panel reports a 17 % gap.

The justification recorded on the constant no longer holds either. It cites
`alan` as a case where "the routed OA path burned the ENTIRE 180 s budget";
`alan`'s OA arm now returns in **0.5 s** with `bound=None`, and the spatial path
certifies 2.925 in 0.2 s / 13 nodes. The constant is defending against a failure
that has since been fixed elsewhere, while causing a new one.

Full 15 × 2 characterization (`t_oa`/`cert_oa` = explicit OA on the whole budget,
`t_sp`/`cert_sp` = the spatial path with the route disabled):

| instance | T | t_oa | cert_oa | t_sp | cert_sp |
|---|---|---|---|---|---|
| syn40m | 60 | 43.5 | **True** | 60.9 | False |
| rsyn0820m | 60 | 60.9 | False | 63.9 | False |
| rsyn0830m | 60 | 60.3 | False | 64.2 | False |
| rsyn0840m | 60 | 60.6 | False | 62.2 | False |
| rsyn0820m02m | 60 | 60.4 | False | 71.8 | False |
| squfl025-040 | 60 | 60.2 | False | 60.6 | False |
| portfol_classical050_1 | 60 | 60.4 | False | 63.0 | False |
| alan | 60 | 0.5 | False | 0.2 | **True** |
| rsyn0805m | 60 | 1.1 | **True** | 60.4 | False |
| rsyn0810m | 60 | 1.3 | **True** | 63.1 | False |
| rsyn0815m | 60 | 15.9 | **True** | 60.2 | False |
| syn30m | 60 | 1.5 | **True** | 28.4 | **True** |
| syn20m02m | 60 | 10.7 | **True** | 60.7 | False |
| squfl020-150 | 600 | 602.0 | False | 609.7 | False |
| squfl015-060 | 600 | 600.8 | False | 37.0 | **True** |

**Handing the route the whole budget is not the fix.** Scored offline against
these traces, `f = 1.0` gains `syn40m` and *loses* `squfl015-060`, where OA burns
all 600 s uncertified and the spatial path certifies in 37.0 s — a wash on count
and a loss on wall. The wall is blind in both directions, and both directions
cost something.

The discriminator is **progress, not gap level**. At the 50 % checkpoint
`syn40m` sits at relative gap 0.147 but is moving fast (0.770 → 0.391 → 0.147),
while `portfol_classical050_1` sits at 0.0045 and is *completely stalled* —
identical bounds from iteration 30 through 33, 154 cuts per iteration buying
nothing, final `bound=None`. A gap-*level* rule (θ = 0.2) extends `portfol` and
costs it its dual bound; a progress rule does not. Scored over all 15 traces:

| policy | certified | wall | rows with no dual bound |
|---|---|---|---|
| fixed f = 0.5 (shipping) | 7 | 1388.2 s | 0 |
| fixed f = 0.75 | 8 | 1521.6 s | 0 |
| fixed f = 1.0 | 7 | 1634.6 s | 1 |
| gap-level f0 = 0.5, θ = 0.2 | 8 | 1372.1 s | 1 |
| **progress-gated f0 = 0.5, δ = 0.25** | **8** | **1376.4 s** | **0** |

Shipped as `DISCOPT_CONVEX_ROUTE_GUARD` / `_RouteProgressGuard`: the route gets
the whole limit and an injected OA `termination_hook` that never fires before
the old checkpoint and then hands back whatever the route has not earned.
**Insufficient evidence is abandon, not continue** — on the four `rsyn` rows the
OA loop completes two iterations in 60 s, offering one finite gap observation and
no trend, and treating that as progress would leave the fallback nothing.

### 22.2 FALSIFIED: capping the OA master to force more iterations

**Hypothesis.** On `rsyn0820m`/`rsyn0830m`/`rsyn0840m`/`rsyn0820m02m` the OA loop
completes only two iterations in 60 s because a single master MILP consumes
almost the whole budget (`_MASTER_NO_INCUMBENT_BUDGET_FRAC = 0.9`, #1062). A
truncated master still yields a valid dual bound for a relaxation of the MINLP,
so capping it per call should buy many more cut rounds and a tighter bound.

**Kill criterion** (stated before the run): *if capping the master leaves the
dual bound no better, or costs the incumbent, on these rows, the hypothesis is
dead.*

**Result: both arms fired the kill criterion.** All four instances are
MAXIMIZE, so the dual bound is an upper bound and *smaller is tighter*:

| instance | optimum | arm | objective | dual bound | iterations |
|---|---|---|---|---|---|
| rsyn0840m | 325.55 | base | −11.413 | 759.439 | 2 |
| | | cap 0.15 | −11.413 | 793.233 ✗ | 7 |
| | | cap 0.30 | −11.413 | 669.653 ✓ | 4 |
| rsyn0830m | 510.07 | base | **497.875** | 728.056 | 2 |
| | | cap 0.15 | 296.403 ✗ | 849.534 ✗ | 7 |
| | | cap 0.30 | 296.403 ✗ | 780.073 ✗ | 4 |
| rsyn0820m02m | 1092.09 | base | −84.863 | 5136.400 | 2 |
| | | cap 0.15 | −83.720 | 5406.735 ✗ | 7 |
| | | cap 0.30 | −83.720 | 5152.472 ✗ | 4 |
| rsyn0820m | 1150.30 | base | 1120.012 | **1184.479** | 2 |
| | | cap 0.15 | 1120.012 | 1436.551 ✗ | 7 |
| | | cap 0.30 | 1120.012 | 1426.337 ✗ | 4 |

`CHECKS_EXECUTED 24`, all rows sound against `minlplib.solu`. The mechanism
works — capping does produce 2 → 4 or 7 iterations — but **more rounds bought
weaker cuts, not better bounds**: the bound is worse on 7 of 8 arm/instance
pairs, and the single improvement (`rsyn0840m` at cap 0.30) is contradicted by
cap 0.15 on the same instance, so it is not an effect. `rsyn0830m`'s incumbent
regresses by 200 units under both caps. The master cap does not ship.

The generalizable lesson: on these instances the OA master's *time* is not the
scarce resource — the *strength of the cut set* is. A master truncated before it
proves optimality returns a bound from a partially explored MILP tree, and
iterating that faster compounds the weakness instead of amortizing it.

### 22.3 Retraction

Earlier in the same session I stated that OA performs "exactly one master MILP
solve in the entire 60 s" on `rsyn0830m`/`rsyn0840m`/`rsyn0820m02m`. That was
inferred from characterization traces that carried a single trace point, and it
is **wrong**: the cleaner low-load `base` arm above measures **two** iterations
and two master calls on all four `rsyn` rows. The substantive observation — that
the OA loop barely iterates on this class — stands; the count does not.

### 22.4 The guard needs OA to check in, or it arrives too late (2026-08-21)

The first graduation panel for `DISCOPT_CONVEX_ROUTE_GUARD` (v1, log
`scratchpad/issue1066/grad_panel_v1_stale.log`) was stopped after ten rows
because those ten rows falsified the v1 design. Both arms of the same
interleaved run, at `time_limit=60`, `gap_tolerance=1e-4`, load 8.1–9.9. Every
instance here is a **maximization** model, so a higher objective is better and a
*lower* dual bound is tighter:

| instance | OFF obj | ON obj | OFF bound | ON bound | OFF wall | ON wall | guard |
|---|---|---|---|---|---|---|---|
| syn40m | 58.2096 | **67.7133 (cert)** | 68.2186 | 67.7133 | 61.6 | **33.5** | 4 calls, no fire |
| rsyn0820m | 1130.1620 | 1120.0124 | 1420.8056 | 1184.4534 | 61.8 | 60.8 | 2 calls, no fire |
| rsyn0830m | 497.8750 | 497.8750 | 766.8364 | 729.1202 | 64.4 | 70.6 | fired 54.61 |
| rsyn0840m | 151.9691 | **−11.4133** | 814.2617 | 764.2488 | 61.0 | 64.9 | fired 55.13 |
| rsyn0820m02m | −84.8635 | −84.8635 | 5257.0875 | 5091.9553 | 60.3 | 70.5 | fired 54.73 |

The bound moved the right way on every row and `syn40m` gained its certificate,
but `rsyn0840m` lost its incumbent outright and `rsyn0820m` lost 10 units. Under
§5 bar 1 ("objective drift within tolerance") that is a failing panel, and the
v1 flag would have stayed OFF.

**Attribution.** The `fired` column is the mechanism. `_MASTER_NO_INCUMBENT_BUDGET_FRAC`
gives the first master `0.9 × remaining`, so widening the route's budget from
`0.5 × time_limit` to the whole limit also widened the first master from ~27 s to
~54 s. The `termination_hook` only runs at the top of an OA iteration, so on the
`rsyn` rows the guard's *second* call did not arrive until ~55 s of a 60 s limit.
Abandoning there is worse than never having tried: the spatial fallback inherits
~5 s instead of the 30 s the fixed split used to hand it, and never finds the
incumbent it finds under OFF. The guard was not wrong about those rows — it was
right and late.

**Why the offline replay missed it.** `scratchpad/issue1066/guard_replay.py`
scored the recorded OA gap traces for certificates and wall only. It never scored
incumbent quality, so it was structurally blind to a policy that trades the
fallback's incumbent for a tighter dual bound, and it scored all four `rsyn` rows
"same". A replay can only falsify what it scores; this is §6 in a form that
passes its own executed-assertion count.

**The fix, and why it is not the falsified master cap.** `solve_oa` takes an
optional `master_checkin_deadline` (seconds of elapsed OA time). `_master_time_budget`
clamps a master solve so the loop returns to the top of an iteration by that
deadline, and the clamp lifts once the deadline passes. The auto-route sets it to
the guard's checkpoint. §22.2 falsified a *different* intervention: capping every
master to a fraction of remaining, which forced more rounds of weaker cuts and cost
`rsyn0830m` 200 units of objective. This truncates at most the one master that
would cross the point where the caller has already said it will decide, and only
while the route has shown no progress. It is refused without a `termination_hook`
to read it, because a master truncation nobody acts on is exactly §22.2.

**Entry experiment** (stated kill criterion: if `rsyn0840m`'s ON incumbent does
not return to ≈151.97 while `syn40m` still certifies, the design is dead and the
flag stays OFF). Interleaved, `CHECKS_EXECUTED 8`, `VIOLATIONS 0`,
log `scratchpad/issue1066/entry_v2.log`:

| instance | arm | status | objective | bound | wall | guard |
|---|---|---|---|---|---|---|
| syn40m | off | feasible | 58.2096 | 68.2186 | 61.7 | — |
| syn40m | on | **optimal, certified** | 67.7133 | 67.7133 | 41.3 | 5 calls, no fire |
| rsyn0840m | off | feasible | 151.9691 | 818.2426 | 62.7 | — |
| rsyn0840m | on | feasible | **151.9691** | 802.0252 | 61.2 | fired **30.53** |

The fire time moved from 55.1 s to 30.5 s, `rsyn0840m`'s incumbent is bit-identical
to its OFF arm, and `syn40m` still certifies. `syn40m`'s masters never approach the
checkpoint, so the clamp never binds there — which is the property that lets one
mechanism serve both rows. Graduation is scored on the full v2 panel
(`grad_panel_v2.json`), not on this experiment.

### 22.5 `DISCOPT_CONVEX_ROUTE_GUARD` graduates default-on (2026-08-22)

Panel: 79 instances -- the #1066 reporter's 15 at their reported limits (60 s,
600 s on `squfl015-060` and `squfl020-150`) plus the 64 in-repo corpus instances
at 20 s. Arms run back to back per instance so a load excursion hits both, each
behind a `load1 <= 10` gate, every row scored against `minlplib.solu` in the
model's own sense. Log `scratchpad/issue1066/grad_panel_v2.log`, data
`grad_panel_v2.json`.

| criterion | OFF | ON |
|---|---|---|
| certified | 55 / 79 | **56 / 79** |
| total wall | 1817.3 s | **1799.5 s** |
| certificates gained / lost | — | 1 / 0 |
| incumbents better / worse | — | 1 / 0 |
| dual bounds tighter / looser | — | 6 / 3 |

**Bar 1, cert-clean: PASS.** 290 soundness checks executed, 0 violations. No
incumbent beat its reference optimum, no dual bound crossed one, no
`gap_certified=True` row regressed to uncertified, no row lost a dual bound it
had, and the only objective that moved moved the right way (`syn40m`
58.2096 -> 67.7133 on a maximization model). The three looser dual bounds
(`casctanks` 6.3109 -> 6.2497, `rsyn0840m` 808.84 -> 820.55,
`squfl020-150` 373.65 -> 353.18) are all on rows uncertified in **both** arms,
and all remain on the valid side of their reference optimum.

**Bar 2, net-positive: PASS,** though modestly: +1 certificate, −17.8 s total
wall, 6 bounds tighter against 3 looser, nothing regressed. The certificate
gained is `syn40m`, the row #1066 was opened about. The guard *fired* on 11
instances spanning syn/rsyn/squfl/portfol/clay/cvxnonsep/tls, so the budget it
reclaims is a class-wide effect and not a `syn40m` special case — on 10 of those
11 it cut a dead route short without costing anything, which is the half of the
mechanism that shows up in the wall column rather than the certificate column.

Default flipped to on. `DISCOPT_CONVEX_ROUTE_GUARD=0` restores the fixed
`_CONVEX_ROUTE_BUDGET_FRACTION` split, and that path stays tested
(`python/tests/test_1066_route_progress_guard.py::TestGraduatedDefault`,
`test_1059_route_fallback.py::test_auto_route_gets_a_fraction_of_the_limit`).

## 17. #1119 singular-tangent binding trigger: the observable is anti-correlated with payoff (falsified 2026-08-23)

#1115 shipped the singular-endpoint tangent (`SolverTuning.singular_tangent`,
`singular_tangent_lazy`) **default-OFF** because its timing panel failed gate 2:
sound and bound-helpful on the `kriging_peaks` family, but +25.6 % wall on
`eq6_1` and +54.2 % on `maxmin` for no bound at all. #1119 was the successor
hypothesis: the separator fires on the **operator** (a `sqrt` / `asin` / `acos` /
`acosh` / fractional-power atom with a singular endpoint), and the fix is to fire
on whether the recovered facet actually **binds** — keep the rows that constrain
the LP, drop the ones that do not, and the cost goes away where the gain is
absent.

Kill criterion, quoted from the issue: *if the binding fraction does not separate
the gaining instances (`kriging_peaks-full*`) from the paying ones (`eq6_1`,
`maxmin`) — i.e. if rows bind at a similar rate in both groups — then binding-rate
is not the predictor, a hit-rate gate cannot recover the cost, and this direction
is dead. Record the falsification and stop rather than reaching for a second
heuristic.*

### The entry experiment

`MccormickLPRelaxer` now tallies, per emitted batch, how many singular-tangent
rows are tight at the optimum of the LP they were added to
(`singular_tangent_stats()`; `_ST_BINDING_TOL = 1e-7` relative, the separator's
own violation scale). Verified **bound-neutral** first, per §5: 40/40 exact
`node_count` / `objective` / `bound` / `status` identities across five instances
× {flag OFF, flag ON}, instrumentation present vs stashed. Panel:
`scratchpad/issue1119/binding_probe.py`, `max_nodes=200`, `time_limit=120`,
`deterministic=True`, flag + lazy placement ON.

| instance | #1115 class | rows | binding | **binding_frac** | Σ LP obj gain | gain/row |
|---|---|---|---|---|---|---|
| `eq6_1` | pays +25.6 % wall, gains nothing | 22 874 | 22 874 | **1.0000** | −1.4e−13 | ~0 |
| `maxmin` | pays +54.2 % wall, gains nothing | 23 189 | 22 214 | **0.9580** | 0.2116 | 9.1e−6 |
| `kriging_peaks-full050` | gains 1.9 % of gap | 5 418 | 4 610 | **0.8509** | 300.2 | 5.5e−2 |
| `kriging_peaks-full100` | gains 1.6 % of gap | 8 976 | 8 234 | **0.9173** | 594.4 | 6.6e−2 |

### Verdict: falsified, and in the worst of the three available ways

1. **Binding is near-universal, because it is near-tautological.** A row is
   emitted precisely because it is violated at the current LP point, so the
   re-solve that follows lands on it. The whole population sits in 0.85–1.00.
2. **What separation exists runs backwards.** The two instances that *pay* have
   the two *highest* binding rates; the two that *gain* have the two lowest. A
   gate that keeps high-binding rows keeps exactly the rows that buy nothing.
3. **The gate cannot recover the cost even in principle.** `eq6_1` has **zero**
   non-binding rows: a rule that drops non-binding rows drops 0 of 22 874 and
   saves 0 % of the +25.6 % it exists to remove. `maxmin` drops 975 of 23 189 —
   4.2 % of the rows against +54.2 % wall. Meanwhile on `kriging_peaks-full050`
   the same rule discards 808 of 5 418 (14.9 %) of the rows on the instance the
   feature is *for*. Zero saving where it must save; collateral where it must not
   cut.

The direction is dead. Per the issue's own instruction, stopping here rather than
reaching for a second heuristic.

### What is deliberately NOT being proposed

Σ LP-objective gain per row does separate the two groups, by three to four orders
of magnitude (~0 and 9.1e−6 for the payers; 5.5e−2 and 6.6e−2 for the gainers), and
an age-out gate reading it is the obvious next thought. It is not being built, for
three reasons and none of them is effort:

* It is a **different hypothesis** from the one #1119 authorized, and the issue's
  kill criterion explicitly forbids substituting one when the first dies.
* The figure is **unnormalized across objective scales** (`eq6_1`'s bound is
  −0.497, `kriging_peaks-full100`'s is −345.6), so the four-point spread is not
  yet evidence of anything scale-free.
* The population is **tiny**. A prefilter over MINLPLib for instances whose `.nl`
  contains `o39`/`o51`/`o52`/`o53` or a fractional-power exponent yields 164
  candidates; screening the 96 smallest (`max_nodes=1`, 15 s, flag + lazy ON;
  stopped at `tls7`, whose root does not finish inside the budget) registers a
  spec on 18 and emits a row on **8**:

  | instance | rows | binding | binding_frac | Σ obj gain |
  |---|---|---|---|---|
  | `eq6_1` | 300 | 300 | 1.0000 | −5.6e−17 |
  | `kriging_peaks-full050` | 170 | 80 | 0.4706 | 3.8e−05 |
  | `kriging_peaks-full030` | 136 | 76 | 0.5588 | 5.6e−05 |
  | `kriging_peaks-full020` | 24 | 24 | 1.0000 | 1.2e−04 |
  | `kall_ellipsoids_tc03c` | 16 | 16 | 1.0000 | 0 |
  | `kriging_peaks-full010` | 8 | 8 | 1.0000 | 3.8e−05 |
  | `tspn12` | 2 | 2 | 1.0000 | 8.5e−14 |
  | `tspn15` | 1 | 1 | 1.0000 | −8.5e−14 |

  Same shape at the root as at 200 nodes, and more starkly: the two payer-profile
  instances (`eq6_1`, `kall_ellipsoids_tc03c` — many rows, zero gain) bind at
  1.0000, while the two `kriging_peaks` members with enough rows to have a rate
  bind at 0.47 and 0.56. The in-repo 66-instance corpus tells the same story from
  the other end: 12 files contain a singular opcode, 2 emit any lazy row at all
  (`tspn08` 1 row, `tspn12` 3), both at ~1e−13 of bound. A threshold fitted to
  four named instances out of a population this size is the single-problem
  solution CLAUDE.md §2 rejects, not a class fix.

`singular_tangent` therefore stays **default-OFF**, which is already its shipped
state; #1115's flag and its lazy-placement variant remain in tree, tested, and
opt-in. The accounting stays in `MccormickLPRelaxer` — it costs one dot product
per emitted batch on a path that only runs when the default-off flag is on, and it
is how this verdict would be re-checked.

## 23. #1066 master cut budget: raising it is falsified; probe-then-escalate is the fix (2026-08-29)

§22 established that the #1066 rows lose because a single OA master MILP eats
almost the whole 60 s budget, and that capping the master is not the fix. This
section resolves what the master's cost actually is, records the falsification of
the obvious remedy, and reports the policy that replaced it.

### 23.1 The stale constant

`milp_simplex` is the single Python funnel through which every in-house MILP
reaches the Rust driver, and it passed the driver **no cut options at all** — so
every solve silently inherited the binding's defaults, `root_cuts=16`,
`cut_rounds=1`, `cut_select=False`. Those numbers were set in #334 (commit
`03f84a62`, 2026-06-27) against the *cost* of cutting: back then each round of
the root loop re-derived the augmented LP from a cold slack basis, so a second
round cost a full root solve — 14 cold root solves were 23.1 s of a 24.2 s cut
loop on the `rsyn0840m` master. #1102 removed that cost by warm-starting each
round from the previous round's basis. The budget chosen against it was never
revisited.

### 23.2 Entry experiment: raising the budget, 10 instances

Hypothesis: the legacy budget is now mis-set, and raising it to
`root_cuts=200, cut_rounds=10, cut_select=True` lets the master close inside the
OA loop's per-iteration share. Kill criterion: dead if no row gains a certificate
and no dual bound tightens materially. `CHECKS_EXECUTED 40`, `VIOLATIONS 0`:

| instance | OFF (legacy 16/1) | ON (200/10/select) |
|---|---|---|
| `rsyn0820m` (max) | feasible, bnd 1435.84, 63.6 s | **optimal, certified** 1150.3005, 20.4 s |
| `rsyn0830m` (max) | obj 497.87, bnd 744.10 | obj 504.59, bnd 523.45 |
| `rsyn0840m` (max) | obj 151.97, bnd 839.93 | obj 325.55, bnd 500.03 |
| `rsyn0820m02m` (max) | obj −65.28, bnd 5227.41 | obj −24.15, bnd 4359.66 |
| `syn40m` | certified, 30.0 s | certified, 1.4 s |
| `syn20m02m` | certified, 7.9 s | certified, 2.9 s |
| `rsyn0805m` | certified, 1.1 s | certified, 2.8 s (**cost**) |
| `squfl025-040` (min) | bnd 144.64 | bnd 122.91 (**regression**) |
| `portfol_classical050_1` | obj −0.09071, bnd −0.09743 | tie |
| `alan` | certified, 0.5 s | certified, 0.5 s |

The kill criterion was not met — a certificate was gained and four bounds moved
substantially — so the mechanism was carried to a graduation panel.

### 23.3 FALSIFIED: no static budget is safe (graduation panel, 79 instances)

The corpus-wide panel **failed §5 bar 1**. `CHECKS_EXECUTED 291`,
`VIOLATIONS 1`, `ROWS_EXERCISED 37/79`. Certified 56/79 in both arms: the raised
budget *gained* `rsyn0820m` and **lost `tls2`** — a certification regression, which
bar 1 forbids with zero slack. Total wall 1797.6 s → 1722.4 s; five bounds
tighter, four looser.

Root cause, measured standalone on `tls2`'s masters (tiny: 46 ub rows, 6 eq rows,
37 cols, 33 integer):

| `tls2` master0 | status | bound | nodes | wall |
|---|---|---|---|---|
| legacy 16/1 | **optimal 3.10** | 3.10 | 241 | 0.0 s |
| 200/10/select | feasible | 2.43 | 58 923 | 60.0 s |
| 100/10/select | feasible | 2.43 | 64 705 | 60.0 s |
| 64/5/select | feasible | 2.30 | 31 631 | 16.1 s |
| 32/5/select | feasible | 2.30 | 31 631 | 16.0 s |

Same shape on masters 3 and 6 (515 and 595 nodes, 0.0 s at the legacy budget; the
strong arms on master 6 find a *worse* incumbent, 8.30 vs 5.30). No soundness
violation — the optimum stays inside `[bound, incumbent]` on every arm — but the
conclusion is binding: **a master that closes in a few hundred nodes at 16 cuts is
overhead territory, and the extra rows derail its search.** Neither budget
dominates, and no static raise is safe. Treat this as a negative result: do not
re-propose raising `root_cuts`/`cut_rounds` unconditionally.

Sub-hypotheses falsified in the same sweeps, also binding:

* `gmi_cuts=False` is far worse than legacy on 3 of 4 `rsyn` masters.
* `node_propagation=True` is neutral-to-harmful.
* `root_cuts >= 500` collapses node throughput.
* `root_cuts=16, cut_rounds=10, cut_select=True` is insufficient — the *cap*
  matters, not merely the number of rounds.

### 23.4 The design that replaced it: probe, then escalate

Since nothing in a master's shape predicts which class it is in, the policy
measures instead of guessing: run the cheap budget under a **node cap**, and
spend the strong budget only on a master that cap fails to close. Both attempts
bound the same MILP, so either one's dual bound is valid and either one's
incumbent is feasible; the merge takes the better of each and can never invent a
bound neither attempt proved. The second attempt is seeded with the probe's
incumbent (the driver re-validates a seed and silently drops one it cannot prove
feasible, so seeding cannot manufacture a certificate).

Entry experiment on 10 captured masters, cap 20 000 nodes, 60 s:

| master | escalates? | policy | legacy alone | strong alone |
|---|---|---|---|---|
| `tls2` masters 0/3/6 | **no** | optimal, 0.0 s | optimal, 0.0 s | feasible, 60 s |
| `rsyn0820m` m0 | yes | optimal 1.6 s | optimal 11.4 s | optimal 0.7 s |
| `rsyn0830m` m0 | yes | optimal 1.1 s | optimal 49.2 s | optimal 0.3 s |
| `rsyn0840m` m0 | yes | **optimal 33.5 s** | feasible 63.3 s | optimal 32.4 s |
| `rsyn0820m02m` m0 | yes | bnd −4152.9 | bnd −5015.8 | bnd −4156.8 |
| `squfl025-040` m0/m5 | **no** | optimal (1 n, 23 n) | optimal | optimal |
| `portfol_classical050_1` m0 | **no** | optimal, 0.0 s | optimal | optimal |

The policy keeps every `rsyn` win and declines to escalate on `tls2`, `squfl` and
`portfol` — the three classes the static profile hurt.

**Retraction (CLAUDE.md §11).** The probe-overhead half of the stated kill
criterion — "dead if probe overhead exceeds ~15 % of the strong arm's wall on the
`rsyn` masters" — **fails as written**: the probe costs 0.9 s against a 0.7 s
strong wall on `rsyn0820m` (129 %). The criterion was unmeasurable, not the design
unsound: a node-capped probe costs a roughly fixed ~1 s while the strong wall
ranges 0.3 s → 32 s, so the ratio tracks the denominator rather than the policy.
The decision-relevant quantity is absolute overhead — ~0.8–1.1 s per *escalating*
master against a 54 s master budget, and zero on a master that declines — and the
end-to-end panel in §23.5.

The shipped cap is **5 000 nodes**, chosen from the measured node counts rather
than taste: every master the legacy budget closes fast closes well inside it
(`tls2` at 241/515/595 nodes), and every master that needs the strong budget was
still open at 20 000 (all four `rsyn`). 5 000 sits ~8× above the first class and
far below the second, so it classifies identically to the 20 000 cap the entry
experiment ran at while costing ~4× less where it escalates.

### 23.5 Graduation panel: both bars pass (2026-08-29)

Same 79-instance manifest, same scorer and same machine as the §23.3 panel that
the static profile failed, so the two are directly comparable. Arms interleaved
per instance behind a load gate (CLAUDE.md §9); `DISCOPT_MILP_CUT_BUDGET`
off/on.

`CHECKS_EXECUTED 292`, `VIOLATIONS 0`, `ROWS_EXERCISED 37/79`.

**Bar 1 — cert-clean.** Zero violations: no incumbent past its `minlplib.solu`
reference, no dual bound crossing one, no row losing a bound it had, and — the
clause the static profile broke — **no certification regression**. Certificates
56 → 57: `rsyn0820m` gained, nothing lost. `tls2`, whose certificate the static
profile destroyed, is now bit-identical across arms (`obj 5.299999999988933`,
`bound 5.299999976086654`, 21.4 s both) because the probe closes each of its 11
masters and declines to escalate. `portfol_classical050_1` is likewise identical.

**Bar 2 — net-positive.** Every direction moves the right way:

| | OFF (legacy only) | ON (probe-then-escalate) |
|---|---|---|
| certified | 56/79 | **57/79** |
| dual bounds moved | — | **5 tighter, 0 looser** (of 78 compared) |
| total wall, all rows | 1796.6 s | **1712.3 s** (−4.7 %) |

Tighter bounds: `rsyn0820m` 1435.8 → 1150.3, `rsyn0830m` 738.3 → 517.1,
`rsyn0840m` 816.8 → 499.3, `rsyn0820m02m` 5216.2 → 4360.5 (all max sense, so
lower is tighter), `squfl025-040` 144.882 → 144.993 (min sense) — the row the
static profile *loosened* to 122.9.

Largest wall movements, all improvements, none a regression:

| instance | OFF | ON | |
|---|---|---|---|
| `rsyn0820m` | 63.5 s uncertified | **20.2 s certified** | −43.2 s |
| `syn40m` | 29.1 s | 2.2 s | −26.8 s |
| `rsyn0815m` | 13.5 s | 3.4 s | −10.0 s |
| `syn20m02m` | 7.5 s | 3.4 s | −4.1 s |

No exercised row got more than 2 s slower, and `rsyn0805m` — which the static
profile taxed 1.1 → 2.8 s — is 1.1 → 1.0 s, because its masters close inside the
probe cap and never reach the strong budget. This is the point of the design: the
cost is paid only where the cheap budget has already been shown to fail.

Both bars clear on one run, which under the 2026-07-17 policy suffices, so
`_MILP_CUT_BUDGET_DEFAULT` graduates to `True`. `DISCOPT_MILP_CUT_BUDGET=0`
remains the opt-out and restores the single legacy solve exactly; that path stays
covered by tests.

### 23.6 Known defect uncovered, not fixed here

The Rust root cut loop (`crates/discopt-core/src/bnb/milp_driver.rs`, the
`for _round in 0..opts.cut_rounds` loop) **never consults `time_limit_s`**. At the
legacy `cut_rounds=1` this is invisible; at the strong budget's 10 rounds a master
can in principle spend its whole deadline cutting before a single B&B node runs.
The escalation is not exposed to it in any measurement here — the strong arm is
entered only after a node-capped probe has already returned, and no panel row hit
it — but the loop should honour the deadline regardless of who calls it.

## 24. #1066 root cuts: the stage's time budget never covered the stage (2026-08-29)

§23's probe-then-escalate fixed the OA *master*. It did not close `rsyn0830m`
or `rsyn0840m` at default settings, so this is where the remaining wall went.

### The measurement

A default solve of `rsyn0830m` (`time_limit=150`, `gap_tolerance=1e-4`) under
`cProfile`, against the escalation-ON tree. Top of the cumulative profile:

```
   ncalls  tottime  cumtime  filename:lineno(function)
        1   81.281   81.281  discopt/solvers/_root_cuts.py:276(_solve_lp)
        1    0.003   75.230  discopt/solvers/mip_nlp.py:662(solve_mip_nlp)
        8   74.469   74.469  {built-in method discopt._rust.solve_milp_csc_py}
```

**A single `_solve_lp` call cost 81.3 s of a 150 s solve.** The two figures are
disjoint (81.3 + 75.2 ≈ the 150.6 s wall): the root cutting-plane stage runs to
completion, and only then does the OA search get what is left. The instance
finishes `feasible`, not `optimal`, having spent 54% of its budget in a stage
whose caller allots it `max(2.0, min(10.0, 0.2 * time_limit))` = **10.0 s**.

### The defect

`generate_root_cuts`' docstring says "``time_budget_s`` bounds the stage's wall
time." It did not, in two independent ways:

1. **The clock started in the wrong place.** `t0 = _time.perf_counter()` sat
   *below* the initial `obj, x, duals, h = oa_converge()`. The whole OA
   prologue — up to `OA_MAX_ITERS` = 60 LP solves — ran before the budget was
   armed. Nothing measured it and nothing could stop it.
2. **The check was between rounds, and the overrun is inside one call.** Even
   with the clock started correctly, `if perf_counter() - t0 > time_budget_s`
   is only consulted at the top of the round loop. `_solve_lp` passed no
   `time_limit` to HiGHS, so one LP could run arbitrarily long — and on
   `rsyn0830m` exactly one did. **A between-rounds budget cannot bound a stage
   whose unit of work is unbounded.** This is the same shape as the Rust
   root-cut-loop deadline defect recorded in §23.6.

Note which of these the 81.3 s was: `ncalls` is **1**. Not 60 cheap LPs adding
up past a budget — one LP that never had a deadline.

### The fix (`DISCOPT_ROOT_CUT_DEADLINE`, shipped default-OFF; default-ON since §25.9)

- The stage clock (`t_stage`) starts before the prologue, and `_remaining()`
  is what every LP is bounded by.
- `_solve_lp` gains a `time_limit` and sets HiGHS' own `time_limit` option,
  floored at 1e-3 — HiGHS reads `<= 0` as *no limit*, which would silently
  restore the overrun.
- `oa_converge` checks the remaining budget between its iterations.
- Budget exhausted before the first LP closes → the stage returns no cuts,
  which is identical to the stage being switched off.

**Soundness.** Truncation can only ever *remove* cuts. A partially converged OA
is an outer approximation over *fewer* constraints than the true feasible set —
a relaxation of a relaxation — so its bound and every cut separated from it stay
valid; only their strength is given up. There is no arm of this change that can
produce a bound tighter than the truth.

**Why it still ships default-off.** It is bound-changing (CLAUDE.md §5 regime 2):
on an instance whose prologue outruns the budget it changes which cuts the stage
separates, hence that instance's root bound. Soundness is not the open question —
*net-positive* is. Spending 81 s of root cuts is not obviously worse than not
spending it; the graduation panel is what answers that, and until it clears both
bars the flag stays off.

### Verification

Deterministic and load-immune: a fake `perf_counter` the stub advances, so no
assertion here depends on machine load (CLAUDE.md §9). Five new tests in
`python/tests/test_nlpbb_root_cuts_781.py`. Mutation battery, each restored to
identical afterwards:

| mutation | result |
| --- | --- |
| round-loop clock starts late in both arms (the original bug) | 1 failed |
| prologue LPs unbounded (deadline only inside the round loop) | 1 failed |
| HiGHS never gets the `time_limit` option | 1 failed |
| no positive floor on the limit handed to HiGHS | 1 failed |
| the flag defaults ON | 1 failed |
| flag OFF also gets a deadline (legacy path not intact) | 1 failed |

A first battery run reported the bug-revert mutation as *surviving*. It was the
mutation that was wrong, not the test: it neutralised `_remaining()` but left
`t0 = t_stage`, so the round-loop check still fired at the same point. Recorded
because "the mutation survived" and "the test is vacuous" look identical from
the exit code, and only reading the mutation apart from the result tells them
apart.

### The owed panel (tracked in #1141) — PAID, and the flag graduated: see §25.9

> **Settled 2026-08-31.** The panel below was run, failed bar 1 on `tls2`, the
> mechanism behind that failure was found and fixed, and the re-run cleared both
> bars. `DISCOPT_ROOT_CUT_DEADLINE` is **default-ON** as of §25.9; everything
> from here to the end of §24 is the debt as it stood, kept for the record. Note
> that §25.9 also **retracts** the root cause the first panel published.

`DISCOPT_ROOT_CUT_DEADLINE` merged default-OFF, which meant the shipped default
was the arm where the docstring's promise is false. That is acceptable only for
as long as the graduation panel is actually owed to someone, so it is recorded
here and in **#1141**: the flag graduates or is deleted, and "neither" is not an
outcome. A flag that ships off and is never panelled is the dead flag CLAUDE.md
§3 forbids; the difference between this and that is whether the debt is written
down.

Re-confirmed live on `main` at 2026-08-30, before merging — the defect is not an
artefact of the #1066-era tree:

- `_root_cuts.py` calls `oa_converge()` and only *then* sets `t0`, so the
  prologue is still outside the budget.
- `_solve_lp` still sets `output_flag` and nothing else, so no LP carries a
  deadline.
- `nlpbb_root_cuts_enabled()` is **default-ON** (graduated 2026-08-20, §21) and
  `solver.py` hands the stage `max(2.0, min(10.0, 0.2 * time_limit))`, so this
  is a default-path stage with a budget it cannot enforce, not an opt-in one.

The panel to run is the standard §5 regime-2 A/B — `DISCOPT_ROOT_CUT_DEADLINE`
ON vs OFF over the in-repo corpus — under a load gate (§9), since *net-positive*
here is entirely a wall-clock claim. Soundness is not what the panel is for:
truncation only removes cuts, so neither arm can produce a bound tighter than
the truth. The question is whether 81 s of root cuts buys more than 81 s of
branch-and-bound, and only the panel answers it.

## 25. #1141 convex-MINLP class: the missing capability, a false dual bound, and a restoration that never converged (2026-08-31)

#1066 closed at 14/15. The row it could not close, `portfol_classical050_1`, is not
closeable by tuning the OA loop: on the certified-convex class discopt solves an
**NLP at every node** where SCIP solves an **LP with gradient cuts** — 38.9 ms/node
against 3.0 — and explores *more* nodes for it. #1141 records four tuning-level
fixes already falsified against that row. This section resolves what the missing
capability was, reports two soundness defects that building it uncovered, root-causes
the restoration failure the issue also reported, and records which of the three new
flags graduated.

Outcome, up front:

| flag | verdict | evidence |
|---|---|---|
| `DISCOPT_OA_NODE_CUTS` | **default ON** | §25.4 |
| `DISCOPT_OA_ELASTIC_RESTORATION` | **default ON** | §25.6 |
| `DISCOPT_OA_INFEASIBLE_NOGOOD` | **stays OFF** | §25.7 |
| `DISCOPT_ROOT_CUT_DEADLINE` (§24's) | **default ON** | §25.9 |
| convex route target: HiGHS → `"oa"` | **retargeted** | §25.10–25.11 |

### 25.1 The capability

`oa.py`'s `node_callback` — gradient (ECP) cuts separated at **fractional** node LP
solutions, which is exactly what SCIP does — has existed since the SHOT work. It was
wired to nothing usable:

| backend | fractional-node hook |
|---|---|
| `gurobi.py` | implemented (MIPNODE) |
| `milp_highs.py` | refused; HiGHS 1.12 declares `kCallbackMipDefineLazyConstraints` and binds nothing to it |
| `milp_simplex.py` | refused; the Rust driver's hook fired only at integer-feasible points |

So without a commercial license every node either paid a full NLP (expensive, what we
do) or ran on a relaxation that ignored the nonlinear constraint (weak). SCIP is cheap
*and* tight because it re-separates at each node's fractional LP.

The Rust driver now has the hook (`MilpNodeHook` / `MilpNodeVerdict` /
`solve_milp_node_hooked`): it fires at fractional node relaxations, stages the
globally-valid rows the separator returns, and re-queues the node so it re-solves
against them. Two budgets bound it — `node_hook_rounds` (separation rounds per node)
and `node_hook_cut_cap` (rows per solve) — and a zero budget is treated as no hook at
all, so an unhooked search stays bit-for-bit identical.

The asymmetry with the lazy hook is the design point. A lazy veto is **mandatory**: it
is the only thing keeping a point out of the incumbent, so a vetoed node is re-queued
and exhausting the re-queue cap costs certification. A node separation is
**optional**: a fractional LP solution is not an incumbent candidate, it only tightens
a relaxation, so exhausting either budget imports the node's own valid LP bound and
leaves certification untouched.

The SHOT profile no longer requires Gurobi — the hook it needs now exists in-house —
while the HiGHS master is still refused for SHOT, since it separates only at
integer-feasible incumbents.

### 25.2 Soundness defect 1: `bound = objective` at an `"optimal"` exit

Building the differential panel produced an arm that certified `optimal` at
−0.10088167 on an instance whose optimum is −0.10091959. The cuts were not at fault:
215 separator rows checked against the verified optimum and against sampled feasible
points, 0 invalid; 100 random MILPs with valid non-cutting node rows left the
certified objective unchanged; 200 brute-force-checked solves over 100 randomized
convex MINLPs with genuinely cutting rows agreed with enumeration on every arm.
Replaying the exact system the driver was handed — static rows plus every separator
row — as a plain MILP found −0.10091959, so the driver's own exit was the false one.

Root cause, and it is **pre-existing and not about #1141's hook at all**:
`milp_simplex` published `bound = objective` on a driver exit of `"optimal"`, on the
reasoning that a proven optimum has incumbent == dual bound. That holds only at gap
**zero**. `"optimal"` from the driver means optimal *within* `gap_tol`
(`decide_status` takes `tm.gap() <= opts.gap_tol`), and `TreeManager::gap` normalises
by `max(|incumbent|, 1.0)` — so on an objective of magnitude 0.1 a 1e-4 "relative"
tolerance is 1e-4 **absolute**, i.e. 1e-3 relative. `solve_lp_nlp_bb` then republished
that number as the MINLP's dual bound.

`_certified_bound` now publishes the engine's own `global_lower_bound` (already
floored by the #598 unresolved-fathom floor and capped at the incumbent), and the lazy
entry point reports the real gap rather than a hardcoded `0.0`. A genuinely drained
tree loses nothing — its frontier bound equals the incumbent. Regression tests fail
before the fix (`published dual bound -0.5420000000000001 is above the true optimum
-0.5430000000000001`) and pass after.

The same `bound = objective`-at-`"optimal"` reading should be audited in the other
MILP wrappers (`milp_highs`, `milp_pounce`, `gurobi`); only `milp_simplex` was touched
here.

### 25.3 Soundness defect 2: an integer-free OA loop certified a local minimum

The corpus panel flagged `trig` (MINLPLib: one continuous variable on `[-2, 5]`, one
nonconvex row): `mip_nlp_method="lp_nlp_bb"` returned `status="optimal"`,
`objective=-2.479027828`, `bound=-2.479027828`, `gap=0.0`. The true minimum over the
declared box is **−3.762500358** (MINLPLib's value, reproduced by brute force at
x = 2.667).

An integer-free OA loop is a single local NLP solve, and both `solve_lp_nlp_bb` and
`solve_oa` reported that solve as a global proof unconditionally. A local minimum is
the global one only on a convex model. `_continuous_model_is_certified_convex` now
gates it; without the certificate the result is `feasible` with `bound=None`. The
mirror claim is gated too — a local NLP that found no point has not *proved* the model
infeasible — so that arm returns `no_feasible_point` rather than `infeasible`.

This predates #1141 and is identical with its flags on or off; the panel found it, it
did not cause it. `test_oa.py::test_no_discrete_short_circuit` asserted the old
behaviour and is updated, with a convex companion that still certifies.

### 25.4 GRADUATED: `DISCOPT_OA_NODE_CUTS` clears both §5 bars

Corpus panel over the 119 vendored MINLPLib instances (`scratchpad/1141/panel_corpus.py`,
both `python/tests/data/minlplib_nl/` and `python/tests/data/minlplib/`),
`lp_nlp_bb` / simplex, OFF vs ON interleaved per instance, incumbents
feasibility-verified from the model's own evaluator rather than taken on the solver's
word:

| | value |
|---|---|
| rows that exercised the flag | 37/119 |
| soundness checks / violations | 128 / **0** |
| certificates | **23 → 25**, none lost |
| total wall | 601.5 s → 538.2 s (**−10.5 %**) |
| dual bounds moved | **10 tighter, 2 looser** |
| incumbents failing an independent feasibility check | 0 |

(An earlier run of the same panel, before §25.3's certificate gate landed, read
42 → 44. The gate correctly demotes ~19 integer-free nonconvex rows from `optimal`
to `feasible` in **both** arms, so the absolute count fell and the delta did not.
The numbers above are the shipping build's.)

Objective drift appears on two rows, `cvxnonsep_nsig30` (130.6628 → 130.6921) and
`cvxnonsep_psig30` (78.9989 → 79.0024), both `feasible`-at-the-limit. Neither is
attributable: `cvxnonsep_nsig30`'s **OFF** arm moved by more than that between two
runs of the same arm (130.6513 vs 130.6628), so incumbent drift on a time-limited row
of this panel is run-to-run variability, not a flag effect. Only bound and status
movements are attributable here.

The two looser bounds are `clay0303hfsg` (1700.0 → 3.98e-12) and `fac2`
(259 641 263 → 257 873 972), neither certified by either arm. `clay0303hfsg` was
investigated rather than waved through: its bound is **frozen across budgets** in both
arms (identical at 25 s, 30 s and 35 s), so it is an early plateau, not degradation
over time; the separator costs 0.02 s of a 30 s solve, so it is not a cost effect; and
the obvious mechanical explanation was **falsified** — see §25.5. The largest single
gain is `st_miqp1`, `feasible` after 30 s with bound 244.07 → **optimal in 0.03 s** at
the oracle 281.0.

Both bars clear on one run, which under the 2026-07-17 policy suffices, so the default
flips to ON. `DISCOPT_OA_NODE_CUTS=0` remains the opt-out and restores the pre-#1141
master exactly; `DISCOPT_OA_NODE_CUT_ROUNDS` / `DISCOPT_OA_NODE_CUT_CAP` tune it.

### 25.5 FALSIFIED: the re-queue was not discarding the bound that mattered

Hypothesis: `clay0303hfsg` loses its bound because re-opening a separated node throws
its evaluation away and leaves it on the parent-inherited bound, and the frontier
minimum is what the driver reports. `TreeManager::raise_node_bound` was added to keep
the bound the evaluation proved (sound and monotone — it never lowers a bound and
never raises one above what an evaluation proved).

It moved the node count (283 → 351) and **did not move the bound at all**
(3.979039320256561e-12, to every digit). The hypothesis is dead. The change is kept
because discarding a proved bound was wrong regardless, but it is not the explanation,
and `clay0303hfsg`'s plateau remains unattributed.

### 25.6 GRADUATED: `DISCOPT_OA_ELASTIC_RESTORATION` — item 3's root cause

#1141 measured 0 of 60 restorations converging on `portfol_classical050_1`, 57 of them
`Error_In_Step_Computation`, and recorded that switching the merit norm
(L1 / L2 / L∞) changed nothing. The reason it changed nothing is that the norm is not
what is broken. `_FeasibilityEvaluator` poses restoration as an **unconstrained**
minimization of a violation merit and reports an **identically zero Hessian** for it.
With no constraints an interior-point method's KKT matrix is `σ_f·∇²f + Σ`, the first
term is identically zero, and away from the variable bounds the matrix is numerically
singular — inertia correction runs out and the solve exits code −3. The norm never
enters.

`_ElasticFeasibilityEvaluator` is the textbook formulation instead (Fletcher–Leyffer;
what BONMIN and SHOT solve), over `z = [x | u]`:

```
min ‖u‖   s.t.   cl ≤ g(x) ± u ≤ cu,   u ≥ 0,   integers fixed
```

one slack per row for L1/L2, one shared slack for L∞. It is smooth, it has real
constraints and the original problem's Lagrangian Hessian, and its start is
elastic-feasible by construction (`u` initialised to the violation), so the IPM begins
inside its own feasible set.

Entry experiment on **real corpus instances** — every restoration the OA loop actually
requested across 27 vendored instances, 400 replays, each through both formulations
(`scratchpad/1141/probe_restoration_sweep.py`):

| | shipped merit | elastic |
|---|---|---|
| converged (code 0/1) | 194/400 | **395/400** |
| `Error_In_Step_Computation` (−3) | **162/400** | **0/400** |
| lower violation reached | — | better on 211, worse on 4, tied on 185 |
| improved on the clipped master point | 205 | 218 |

**The convexity gate is load-bearing, and it is not a tuning knob.** The first
corpus panel of the ungated flag cost +4.3 % wall. Every row where the elastic form
was slower was **nonconvex and produced no incumbent in either arm** (`bchoco06/07/08`,
`beuster`, `heatexch_gen2`: +3.6 to +14.1 s each); every convex row was neutral or
faster. That is the condition under which the elastic subproblem means anything: on a
convex feasible set it is a convex NLP, so its solution is the *global*
minimum-violation point and the restoration certifies something; on a nonconvex model
it is one more local solve, and a more expensive one. Gated on the model's constraints
all being certified convex, the corpus panel is:

| | value |
|---|---|
| rows where the elastic form actually runs | 7/119 |
| soundness checks / violations | 128 / **0** |
| certificates | 23 / 23 |
| total wall | 464.3 s → 457.2 s (−1.5 %, noise) |
| dual bounds moved | **0** |

On all 7 rows the status, objective and wall are unchanged while step-computation
failures become convergences (`m3` 313 → 0, `clay0303hfsg` 46 → 0). Corpus-neutral,
then — and it pays on the class the issue is about:

| instance | merit | elastic |
|---|---|---|
| `meanvarx` (real, MINLPLib) | 0.68 s | **0.30 s** (same certificate) |
| portfolio n=40, K=6 | 49.3 s | **6.2 s** |
| portfolio n=50, K=5 | 19.2 s | **3.7 s** |
| portfolio n=60, K=6 | 52.2 s | **7.8 s** |

Graduated ON with `=0` as the opt-out. The honest limit on this evidence: the vendored
corpus contains exactly **one** real row of the target class (`meanvarx`); the other
three are reconstructions of `portfol_classical050_1` built with the modeling API
because MINLPLib is not reachable from the environment this was measured in. The
MINLPLib convex family is the panel that should confirm it.

The outcome is also no longer invisible. Restoration falls back to the clipped master
point on failure, so a run where it *never* converged looked exactly like one where it
always did; `_RESTORATION_OUTCOMES` now records the subsolver's own verdict per
formulation and the trace summary reports it.

### 25.7 REJECTED as a default: `DISCOPT_OA_INFEASIBLE_NOGOOD` (items 2 and 4)

An OA cut excludes the *point* it is taken at, not the *assignment* — with a linear
objective and no epigraph nothing stops the master returning the same integers at a
different continuous point, which is item 4 (7 of 172 assignments re-proposed, one six
times). The exclusion mechanism is a no-good cut, and it is sound exactly when the
assignment is **proven** infeasible.

Item 2 as literally proposed — map Ipopt code 2 to `SolveStatus.INFEASIBLE` — is
**refused**, not deferred. Restoration converging to a local minimizer of the
constraint violation proves infeasibility only on a **convex** subproblem, and
`_IPOPT_STATUS_MAP` also serves `solve_model`'s pure-NLP path, which would then
publish `status="infeasible"` for a nonconvex model it never proved infeasible. That
is a §1 violation, not a bound change. The sound version is to read the raw code where
a convexity certificate is held, which is why `NLPResult.raw_status` now carries it.

`_assignment_proven_infeasible` requires all three: the subsolver's own code is 2
(*not* merely "the NLP returned nothing" — on `portfol_classical050_1` that was 57
step-computation failures, whose exclusion would delete a subtree that may hold the
optimum); **every** constraint defines a convex feasible set, so fixing the integers
leaves a convex set on which a local minimizer of the violation is global; and the
point the subsolver exited at still violates a row by more than 1e-6.

Validated: 8 brute-force comparisons over 4 draws of a cardinality-constrained
portfolio, every assignment enumerated and solved independently, the flag firing up to
216 exclusions per run — **0 wrong answers, 0 invalid bounds**.

Rejected on bar 2. Corpus panel: 7/119 rows fire it (up to 1709 exclusions on
`st_miqp1`), 128 soundness checks, 0 violations — and **nothing moves**: certificates
23/23, 0 bounds moved, wall +0.8 %. On the class it is worse than inert: the portfolio
n=40 row went from `optimal` in 49 s to `iteration_limit` at 60 s **with no incumbent
at all**, because the no-good rows steer the master away from the assignments where a
feasible point lives. The `DISCOPT_CUT_INHERIT` rule applies: sound, fires, not
helpful — stays OFF, with the measurement recorded.

### 25.8 The class measurement, and what would close #1141's own row

`lp_nlp_bb`, 60 s cap, gap 1e-4. Three arms A/B/C **interleaved**, three repetitions
each, mean ± sd (`scratchpad/1141/timing_class.py`, 27 runs, load average 0.68 before /
1.08 after — CLAUDE.md §9). This is the node-cut arm measured before the elastic
restoration graduated, so it isolates that one flag:

| instance | HiGHS master (the route's target) | simplex master | simplex **+ node cuts** |
|---|---|---|---|
| n=50, K=5 | 17.84 ± 0.22 s | 19.83 ± 0.49 s | **1.41 ± 0.06 s** |
| n=40, K=6 | 54.53 ± 1.45 s | 48.33 ± 0.94 s | **2.49 ± 0.12 s** |
| n=60, K=6 | 7.58 ± 0.03 s | 49.88 ± 0.66 s | **6.20 ± 0.23 s** |

Every arm exits `optimal` and the three dual bounds agree to ~1e-8 per row
(−0.10193369, −0.10091960, −0.10095279), so this is a wall comparison at equal
certificate. With both graduated flags on, the same rows are 0.67 s, 2.00 s and 2.59 s.

Closing #1141's own row additionally needs the **route** to reach this master.
`_convex_minlp_auto_route` sent the certified-convex class to `lp_nlp_bb` on the
**HiGHS** master, chosen in #1066 because the in-house master lost on `rsyn*`
(`rsyn0840m`: in-house root loop 0 % of the gap closed, HiGHS 86.1 %). The table above
shows the in-house master **with** fractional separation beating HiGHS on the
portfolio family; whether it also holds on `rsyn*` is the measurement that would
decide the retarget on speed, and `rsyn*` is not vendored.

> **Superseded by §25.11 (2026-08-31).** The route was retargeted, but not on the
> portfolio evidence and not on a speed argument — on a *dependency-policy* one:
> HiGHS is an opt-in extra and the default route must never name it. §25.11 also
> **retracts** the claim, made in this section's spirit and stated outright in the
> route-panel commit, that #1066's HiGHS evidence was "stale by construction"
> because it predated the fractional-node hook. Measured head to head on the 26
> routed instances, node cuts do **not** close the gap between the masters
> (simplex 97.3 s vs simplex+node 97.7 s). HiGHS is the stronger MILP engine and
> the portfolio table above does not generalise.


### 25.9 The added work item: `DISCOPT_ROOT_CUT_DEADLINE` GRADUATES ON (2026-08-31)

#1141 gained a second work item after #1129 merged: graduate or delete
`DISCOPT_ROOT_CUT_DEADLINE`, the §24 flag that bounds the root cutting-plane
stage's individual LPs, with the bar stated as "the flag graduates or it is
deleted; 'neither' is not an acceptable outcome." The first panel had it failing
bar 1. The mechanism behind that failure was then found and fixed, and the
re-run panel clears both bars. **The flag graduates default-ON**, keeping the
`=0` opt-out and the legacy path.

**Populating the panel.** The stage runs only when the model is
convexity-certified and has integer variables, on a top-level solve, and
`generate_root_cuts` returns before its first LP when there are no integers.
That population is 25 of the 119 vendored instances. Run on the plain default
path the stage is nearly unreachable — `DISCOPT_CONVEX_MINLP_ROUTE` diverts
convex MINLPs to `mip-nlp`, so only **2 of 25** rows enter the stage and only
**1** reaches an LP, with the deadline never biting (122 LPs, 0 declined). A
panel over that population would have printed "0 violations" while measuring
nothing (CLAUDE.md §6). With the convex route off, so the NLP-BB path that owns
the stage actually runs it, the population is real: **12/25** rows enter the
stage, **8/25** run LPs under a deadline.

#### RETRACTED: the first panel's stated root cause was wrong

CLAUDE.md §11. The 2026-08-31 first draft of this section, and PR #1142's body,
both stated:

> an LP that stops on the deadline returns the all-`None` declined tuple, which
> inside `oa_converge` sets `x = None`, breaks the loop, and makes
> `generate_root_cuts` return an **empty** result — one truncated LP discards
> every cut the stage had accumulated.

Measured on `tls2` (pre-fix tree, deadline ON, convex route off): the stage runs
240 LPs, declines 1, exits `no_lp`, and returns **90 cuts**. Not an empty
result — the opposite. On a `no_lp` exit `x` is `None`, which SKIPS the
end-of-loop binding-cut filter, so the entire applied set ships into the model.
That is precisely the row flood `generate_root_cuts`' own docstring exists to
prevent ("the full applied set — measured: ~170 dense rows on rsyn0805m —
collapses node NLP throughput"), and it is why `tls2` lost its certificate: not
too few cuts, **too many**, priced into every node NLP.

The empty-result path is real, but it needs the decline to land *mid*-convergence,
where `oa_converge` overwrote `obj, x, duals, h` in place and lost an LP from the
same call that had already closed optimally. `tls2` does not take that path.

A second claim made in the same session — that the deadline arm was live-shipping
invalid GMI cuts from a stale basis — is also retracted. `separate_gmi` does pair
`binv[r]` / `row_st[r]` with `a_all[r]` positionally, so a row system wider than
the basis would multiply an equality row's basis entry by a cut row, but the
mid-convergence break fires on `_remaining() <= 0` and the round loop's
top-of-loop check reads the same clock, so it exits before another separation
round. Measured live on `tls2`: **0 mismatches in 38 `separate_gmi` calls**. It
is a hazard the retention fix would have introduced, not one that existed.

#### The fix

Three parts, all confined to the deadline arm so the A/B measures exactly one
change (with the flag off a decline is a structural or numerical LP failure
rather than a budget stop, and restoring an earlier solve there would change the
default path's cuts — a bound-changing edit owed its own panel):

1. `oa_converge` keeps the last LP that reached **optimality** instead of
   returning the declined tuple. That LP is over a *subset* of the OA rows, hence
   a relaxation, so its optimum is a valid root bound and cuts separated from it
   are valid.
2. It rolls `cuts_a` / `cuts_b` back to the rows that LP was solved from, so the
   returned basis and row system agree — the invariant `separate_gmi` depends on
   and cannot check for itself.
3. `separate_gmi` now refuses a mismatched basis outright rather than reading
   equality-row entries as cut rows.

The round loop's `no_lp` exit keeps the previous round's optimum for the binding
filter instead of leaving `x = None`.

Effect on `tls2`, everything else held fixed: same 240 LPs, same 38 rounds, same
`lp_bound` — and **90 cuts → 19**, stage-to-answer wall 35.5 s → 30.4 s.

#### Bar 1: cert-clean

119 vendored instances, `DISCOPT_CONVEX_MINLP_ROUTE=0`, 30 s per arm, OFF vs ON
interleaved per instance, incumbents feasibility-verified from the model's own
evaluator (`scratchpad/1141/panel_root_cut_deadline.py`, log and JSON committed):
**331 executed soundness checks, 1 flagged row**, plus **40** stage-replay checks
with **0** violations (`scratchpad/1141/panel_budget_contract.py`).

The flagged row, `clay0303hfsg` (`optimal` → `feasible`), is a time-limit
boundary race, not a regression — 5 interleaved reps, load gate checked
(`scratchpad/1141/reps_tls2.py`):

| `clay0303hfsg`, 5 reps | `optimal` | `feasible` | wall |
|---|---|---|---|
| OFF | 3/5 | 2/5 | 30.01 ± 0.81 s |
| ON | **4/5** | 1/5 | 29.17 ± 0.94 s |

Both arms miss the certificate on some reps; ON misses it *less* often. And the
instance that failed 6/6 before the fix now certifies in both arms, with ON
faster — 3 reps at 60 s: OFF `optimal` 3/3 in 34.00 ± 1.14 s, ON `optimal` 3/3 in
**26.20 ± 7.04 s**.

#### Bar 2: net-positive — measured as contract enforcement, and why

State the limitation first. This is **not** the broad corpus speed-up bar 2
normally asks for, and no such evidence exists here. The flag's benefit shows
only on an instance whose OA prologue outruns the stage budget at the 2–10 s
`solver.py` hands it, and **no vendored instance does** — 0/119 deadline bites.
#1066 measured that pathology on `rsyn0830m` (one LP burned 81.3 s of a 150 s
solve against a 10 s budget) and `rsyn*` is not vendored, nor is minlplib.org
reachable from this environment. So the corpus can neither confirm nor falsify a
speed-up, and claiming one would be inventing the measurement.

What *is* measurable is the contract itself: does the stage return within
`time_budget_s`? That is a wall-clock question, so it reads the same wherever it
is exercised — the #727 lesson (a mechanism validated on a synthetic proxy can be
a no-op on the real class) is about *gains*, not contracts. Measured by replaying
the stage over a budget range on the real instances that run it, with the
arguments **captured from the real caller** so the docstring's caller contract
("`lb`/`ub` are the FBBT-tightened root bounds") holds by construction:

| | OFF | ON |
|---|---|---|
| worst-case overrun past its own budget | **+0.297 s** | **+0.076 s** |
| stage runs overrunning by >20% | 2/35 | **0/35** |
| root bound at the budgets `solver.py` uses (≥2 s) | — | identical on 3 of 4 instances, 9.4e-6 on the 4th |
| corpus wall, 119 instances | 1065.0 s | 1063.5 s (−0.1 %) |
| certificates | 89 | 88, the one difference falsified above |

Bound differences elsewhere on the corpus are noise, and the panel calibrates
it: 9 of 119 rows show a bound difference, but only **2** of those are rows the
flag can touch at all. The other 7 are search jitter on rows where the stage
never ran.

So the flag costs nothing measurable and closes a documented promise that is
otherwise false on a default-ON stage. The alternative #1141 permits — deleting
it — would delete the deadline mechanism and reopen #1066. It graduates.

**A probe that measured nothing, and the guard that caught it.** The first cut of
the contract panel rebuilt `is_int` from a guessed `v.var_type` spelling
(`v.vtype`, which does not exist), so the mask was all-False,
`generate_root_cuts` took its `not np.any(is_int)` early return, and all 28 rows
measured a 0.00 s no-op. The executed-check counter and its non-zero exit are
what surfaced it (CLAUDE.md §6), and the real spelling was one `grep` away (§
"look up an API before calling it"). Its second cut then rebuilt `lb`/`ub` from
`flat_variable_bounds` — the **raw declared box**, not the FBBT-tightened one the
caller contract specifies. On `cvxnonsep_psig40r` that box leaves 42 of 82
columns unbounded, the separators substitute a fake `1e5` for an infinite bound,
and the stage returned root bounds up to 32092 against a verified incumbent of
86.5: **8 "violations" in both arms that were the probe's contract breach, not a
defect in the stage** (the shipped path solves that instance to `optimal` at
86.539). Capturing the caller's own arguments removed all 8.


### 25.10 The route was pinning an opt-in dependency (2026-08-31)

Owner directive, mid-#1141: *"`_convex_minlp_auto_route` returns
`{"milp_solver": "highs"}` — that should never happen, we should always route to
a discopt solver. highs is only available as opt-in."*

It is a defect independent of any performance argument, and the codebase already
said so one layer down. `_resolve_lp_nlp_bb_backend`:

> `"auto"` deliberately does **not** pick HiGHS. Routing it there would make the
> default depend on an optional package and would move every existing caller's
> node counts; the opt-in keeps #356 and the current defaults intact.

`_convex_minlp_auto_route` then pinned `{"milp_solver": "highs"}` and did exactly
that from above. The visible symptom: the default **algorithm** for a whole
problem class changed with whether `highspy` happened to be installed — with it,
`lp_nlp_bb`/HiGHS; without it, a silent fall back to `"oa"`. Two users on the
same model and the same version got different algorithms, and neither was told.

It also silently voided §25.4. `DISCOPT_OA_NODE_CUTS` graduated ON, but the HiGHS
master has no fractional-node hook at all, so on the **default path** for the very
class #1141 is about, the capability the issue exists to add never ran. Every
panel in §25.4 measured it with `mip_nlp_method="lp_nlp_bb"` forced onto the
in-house master, i.e. with the route bypassed. That is not a wrong measurement,
but it is a measurement of a configuration no default solve reached.

**And it still does not reach it.** §25.11 retargets the route to `"oa"` on the
evidence, not to `lp_nlp_bb`, so the fractional-node hook remains off the default
path — it is reached by `mip_nlp_method="lp_nlp_bb"`, which no longer needs
Gurobi or HiGHS to get there. Closing that gap needs the in-house MILP master to
become competitive, which §25.11's last section states as the remaining work
rather than leaving implied.

### 25.11 The route targets `"oa"` — and two of my own claims, retracted (2026-08-31)

The route now returns **no** `milp_solver` key at all, and targets `"oa"`.
`milp_solver="highs"` stays available as an explicit caller opt-in and still
outranks the route's pick (the call site uses `setdefault`).

Getting here took two wrong answers, both published before they were checked.

#### The measurement that was wrong (CLAUDE.md §8)

The first panel forced `mip_nlp_method="lp_nlp_bb"` and `milp_solver=…`, so it
measured the **master**. On the default path the route also brings the #1066
progress guard, a fixed 50 % budget reserve and a spatial fallback, and **none of
them ran**. On that panel `bb_inhouse` scored 26/26 certificates and `"oa"` came
last, and I reported "for a default install this change is a strict improvement,
24 → 26 certificates". That sentence describes a configuration no default solve
takes. **Retracted.**

What a plain `solve(time_limit=30)` actually did on `tls2` under that target:

```
route: ...in-house master... did not certify in 15.12s -> fell back with 14.88s
status=time_limit  objective=None  bound=2.87
```

against `optimal` 5.3 in 0.58 s before the retarget — a lost certificate *and* a
lost incumbent.

#### The hypothesis that was wrong (CLAUDE.md §4, §11)

`clay0303hfsg` certifies in ~30 s driven directly but returns `feasible` through
the route under **both** guard settings, and both hand the route ~50 % of the
limit. So: *the fixed fallback reserve is truncating a master that would have
finished.* Kill criterion, written next to the arm before running it — if
removing the reserve does not recover those rows, or costs certificates on rows
the fallback rescues, the hypothesis is wrong.

It is wrong. Removing the reserve is the **worst** arm of five. The spatial
fallback is *rescuing* rows, not stealing from them.

#### The measurement that decided it

Five arms, plain `solve(time_limit=30)` with no kwargs so guard, reserve and
fallback all participate, interleaved per instance over the 26 instances the
router itself diverts, on an idle machine, 215 executed soundness checks and
**0 violations** (`scratchpad/1141/panel_route_default.py`):

| arm | certificates | incumbents | total wall |
|---|---|---|---|
| `pre` (route → HiGHS) | 26 | 26 | 35.6 s |
| `lp_nlp_bb`, in-house master | 23 | 25 | 174.2 s |
| …same, guard off | 24 | 25 | 162.3 s |
| **`oa`** | **24** | **26** | 118.7 s |
| …in-house, no fallback reserve | 20 | 25 | 200.7 s |

Rows losing a certificate against `pre`: `lp_nlp_bb` loses
`clay0303hfsg`/`cvxnonsep_nsig30`/`tls2` and `tls2`'s incumbent entirely; `"oa"`
loses `clay0303hfsg`/`tls2` and **no incumbent**; no-reserve loses six.

`"oa"` is therefore the best target reachable without an optional dependency, and
the master-only ranking that put it last **inverts** once the route's own
machinery is in play — the guard's `master_checkin_deadline` limb was built for
`"oa"`, while `lp_nlp_bb` on the in-house master is truncated by a budget policy
calibrated on a master that certified in a fraction of its budget.

#### Who wins and who pays

`"oa"` is exactly what an install **without** `highspy` already got, so for the
default install this is a **no-op**. Only callers who had `highspy` present are
affected: they lose two certificates (`clay0303hfsg`, `tls2`) and no incumbents,
and `mip_nlp_method="lp_nlp_bb", milp_solver="highs"` opts them straight back in.
There is no reading on which this is a speed-up; it buys a default path that does
not silently depend on an optional package.

#### What was kept, and what remains

The in-house master's **termination check-in** stays, and it is a real capability
rather than scaffolding: `lp_nlp_bb` now runs on the in-house master with
fractional-node cuts *and* a progress hook, where it previously required Gurobi
or HiGHS. The blocker had been a stale refusal — "the driver enforces
`time_limit` itself and has no callback-termination hook" — whose second half
was false: the Rust driver has had a per-iteration checkpoint carrying
`incumbent`/`bound`/`gap`/`elapsed` with a `Stop` control, exposed to Python as
`debug_hook`, since the interactive debugger landed. The check-in is a
composition over it, with no Rust change; a hook-stopped search reports
uncertified, as an interrupted tree must.

What remains is the honest #1141 headline: **the in-house MILP master is ~3×
slower than HiGHS on this population and the fractional-node hook does not close
that gap** (97.3 s vs 97.7 s over the same rows). Until it does, the default
route cannot use `lp_nlp_bb`. That is a master-engine problem — root-gap closure,
cut management, node throughput — not a separation-hook one, and it is what a
follow-up should attack.

#### A panel thrown away

One run of this panel was invalidated and is kept as
`panel_route_default_CONTAMINATED.json` rather than deleted. A leftover
subprocess from a killed `pytest` run — the JAX-fallback test, whose command line
did not match the `pkill` pattern used — ran at **100 % CPU for 15 minutes**
across the whole panel, with load averaging 2.5–2.9 for a single-process
measurement. This is the incident CLAUDE.md §9 already records, repeated. The
re-run waited for load below 0.4 and checks for stray load at the **end** of the
run, not only at the start.

### 25.12 ROOT CAUSE: the lazy separator was called on INFEASIBLE nodes (2026-08-31)

This is #1141's item 4, and the reason the certified-convex class was unusable on
our own MILP master. Everything in §25.10–25.11 about "the in-house master is
~3× slower" was reasoning about a symptom.

`INFEAS_SENTINEL` is a **finite** `1e30`, so the lazy path's admission test

```rust
let integral = out.result.lower_bound.is_finite()
    && (out.result.is_feasible || solution_is_integral(&out.result.solution, &is_int));
```

admitted an **infeasible** node. Such a node carries a placeholder solution
vector of zeros, which `solution_is_integral` accepts. The separator was handed a
point that is not a solution of anything, returned a cut for it, and the node was
re-queued — against a matrix the point still violates, because it was never a
solution of that matrix. The loop runs to `LAZY_REQUEUE_CAP`, which sets
`gap_certified = false` and `search_incomplete = true`.

Measured on MINLPLib `tls2`, in-house master, 30 s:

| | before | after |
|---|---|---|
| separations | 1477 | **27** |
| distinct assignments | **1** | 25 |
| re-proposals | 1476 | **2** |
| fixed-NLP subproblems | 1537, all infeasible | — |
| result | `iteration_limit`, no incumbent | **`optimal` 5.3** |

Confirmed from inside the driver: the returned point violated **31 of the 35 cut
rows already present in its own LP**. The same path on the HiGHS master needed
**13** subproblems and reached `optimal` 5.3 — identical NLP layer, identical cut
logic, only the master differs, which is what localised the defect to ours.
`cvxnonsep_nsig30` falls to 6 re-proposals across 1219 assignments (the issue
reported "7 of 172" — the same defect, sampled on an instance where it happened
not to lock up).

The fix is the test the **fractional** hook already applies
(`node_result_usable`), added in this same issue. §25.4 recorded "the lazy path's
`integral` predicate is left byte-identical" as a *safety* property. It was the
bug: the new path was guarded and the old one left carrying it.

**Effect on the route question.** Re-running §25.11's arm comparison on the fixed
master (26 routed instances, 30 s, interleaved, 160 checks / 0 violations):

| arm | certificates | incumbents | total wall |
|---|---|---|---|
| `spatial` | 24 | 26 | 96.0 s |
| `oa` (the shipped target) | 24 | 26 | 108.3 s |
| `lp_nlp_bb`, in-house master | **24** | **26** | 102.4 s |
| `lp_nlp_bb`, HiGHS (opt-in) | 26 | 26 | 30.9 s |

`lp_nlp_bb` on the in-house master rises from 23 certificates / 25 incumbents to
parity with `"oa"` — so the shipped `"oa"` target is **not** a regression, and the
choice between the two native options is now a coin flip rather than a
2-certificate gap. Pointing the route at `lp_nlp_bb` would additionally put
fractional-node separation on the default path; that is the natural next step and
it now rests on a tie rather than on a deficit.

**Method note.** Two earlier rounds of this investigation attributed the symptom
to budget policy, guard calibration and master speed in turn, and each was
falsified. What ended it was instrumenting the driver to ask the matrix directly
whether the returned point satisfied its own rows — a question with a yes/no
answer, unlike "is the master too slow". Reach for the invariant check earlier.

## CC-1143 The convex-MINLP route's abstain cost: three hypotheses falsified, one survived

**Context.** #1143: `_convex_minlp_auto_route` sends a convex MINLP to an OA
master; when the route does not certify, the caller pays at least
`_CONVEX_ROUTE_BUDGET_FRACTION` (half the limit) before the default path starts
from scratch. On the cert panel that is routinely 3-8x the entire unrouted solve.
The issue named three candidate mechanisms. All three are **falsified**; the
survivor came out of the measurements rather than the list.

Measured 2026-08-31, 14-core box, idle, `_rust.abi3.so` rebuilt from the tree
under test and asserted at import (§8), 60 s budgets, corpus instances from
`python/tests/data/minlplib_nl` and the MINLPLib snapshot.

### Falsified 1 — gate the route on the default path's root gap

The issue's reasoning: `cvxnonsep_nsig30`'s root gap is 0.0011, so "a model
already almost closed at the root has little for an OA master to win". Measured
the default path's `root_gap` (route OFF) for 24 instances and classified each by
what the route actually did:

    route-won root gaps : min 0        max 32.01   (n=19)
    abstained root gaps : min 0.001139 max  8.109  (n=5)

**Complete overlap; no separating threshold exists.** `cvxnonsep_nsig30`
(abstains, 0.001139) sits *between* `cvxnonsep_psig40r` (won, 6.8e-05) and
`st_miqp4` (won, 5.5e-04). The same overlap kills the variant that predicts the
fallback's cost from the root gap: `rsyn0805m` (gap 0.2173) needs 60.2 s unrouted
while `flay02m` (gap 0.1875) needs 1.64 s.

### Falsified 2 — an early checkpoint on the "no finite dual-bound observation" verdict

The issue's candidate 2: fire early on the cheap verdict the guard already
implements. Measured the guard's observation count at 10% of the budget:

| instance | obs by 10% | what the rule would do |
|---|---|---|
| `fac2` | 5 | **not fire** — but this is a target |
| `cvxnonsep_nsig30` | 155 | **not fire** — also a target |
| `rsyn0840m` | 0 | **fire** — a row #1066 protects |
| `rsyn0820m02m` | 0 | **fire** — ditto |

The rule is **anti-correlated with what it needs to do**: it misses both expensive
abstainers and fires on the two rows the #1066 record says to keep.

### Falsified 3 — moving `master_checkin_deadline` earlier on its own

Necessary but not sufficient, and harmful alone. With the check-in at 10% and the
verdict left at 50%, the first master is shortened but the route still keeps the
budget: **`fac2` optimal 35.8 s -> feasible 60.7 s** and **`squfl015-060` optimal
-> feasible**, against `rsyn0840m` feasible -> optimal. Two certificates lost, one
gained. A half-measure is worse than either endpoint.

### Survived — the decision point (implemented, default-off)

The measurements say the route **either certifies fast or not at all**. Over 24
classified instances every row the route certifies it certifies within **2.96 s**
of a 60 s limit (`syn20m02m` 2.91 s is the slowest, `syn40m` 2.05 s, `syn15m03m`
0.90 s, `rsyn0805m` 0.82 s); every abstain costs **>= 28 s**. Win time and waste
are separated by an order of magnitude, so the decision needs neither a model
property nor a gap trend — only to be taken early.

`_CONVEX_ROUTE_DECISION_POINT_FRACTION = 0.10` is bounded on both sides by
measurement, which is why it is not smaller. At **0.05** the cut lands inside the
win distribution: `syn20m02m` loses its certificate outright (optimal 2.91 s ->
feasible 60.4 s) and `tls2` regresses 30.3 s -> 57.6 s. At **0.10** both recover
and the savings survive (flag OFF vs ON, interleaved, shipped flag):

| instance | OFF | ON | |
|---|---|---|---|
| `cvxnonsep_nsig30` | optimal 29.71 s | optimal **6.69 s** | -23.0 s |
| `clay0303hfsg` | optimal 43.16 s | optimal **22.23 s** | -20.9 s |
| `fac2` | optimal 35.78 s | optimal **14.69 s** | -21.1 s |
| `squfl015-060` | optimal 60.20 s | optimal **40.05 s** | -20.2 s |
| `tls2` | optimal 30.48 s | optimal 30.47 s | neutral |
| `syn20m02m` | optimal 2.90 s | optimal 2.91 s | win preserved |
| `rsyn0805m` | optimal 0.76 s | optimal 0.76 s | win preserved |

Zero lost certificates, zero objective disagreements. The gap-trend test is
deliberately **not** consulted at the decision point: it is what kept
`cvxnonsep_nsig30` running 28.2 s while its gap improved the whole way
(0.298 -> 0.0013) without ever certifying. "Is it improving?" is the wrong
question; answering it correctly still costs the budget.

### The floor the in-repo panel could not see

A fraction alone is unsound on a short budget, and the first panel missed it. A
route's win time is **absolute** (`syn20m02m` certifies in ~2.9 s at every limit)
while a fraction **shrinks** with the limit: at a 20 s budget the bare 10% gives
2.0 s, which cuts inside the win distribution. Measured, `syn20m02m` goes
**optimal 3.16 s -> feasible 20.12 s**. The 61-instance in-repo corpus is blind to
this because it holds no `syn`/`rsyn` instance — the class the route exists for —
so the 66-pair panel passed both bars while the policy carried a latent lost
certificate.

`_CONVEX_ROUTE_DECISION_POINT_FLOOR_S = 5.0` clears the slowest measured win
(2.96 s) with margin, and the decision point is additionally capped at
`_CONVEX_ROUTE_BUDGET_FRACTION` so that on a budget too short for the floor the
policy degrades to the #1066 behavior rather than to something worse:

    limit    2 s ->  1.00 s   (= #1066)     limit   20 s ->  5.00 s  (#1066: 10 s)
    limit    8 s ->  4.00 s   (= #1066)     limit   60 s ->  6.00 s  (#1066: 30 s)

With the floor, `syn20m02m` at 20 s is back to **optimal 2.91 s**.

**Generalisable lesson.** A policy parameterised as a fraction of the caller's
budget must be checked against the *absolute* quantity it is racing. The in-repo
corpus cannot falsify a claim about a class it does not contain; #1059 knew this
and supplemented with `syn`/`rsyn`, and the same supplement is what caught this.

### Status: default-ON (graduated 2026-08-31)

Graduation panel — in-repo corpus, 66 pairs at 20 s, OFF/ON interleaved:

| criterion | result |
|---|---|
| certificates lost | **0** |
| certificates gained | 1 (`clay0303hfsg`) |
| objective vs oracle beyond correctness tol | **0** |
| total wall OFF / ON | 417.9 / 407.9 s |
| materially faster / slower | 3 / 1 |

Cert-clean **and** net-positive. The single slower row, `bchoco08`
(20.01 -> 20.96 s), certifies in neither arm at a 20 s limit — a wall difference
between two time-limited runs, not a lost result. Supplemented with the
`syn`/`rsyn`/`squfl` class at 20 s and 60 s: no route win lost at either budget.
`DISCOPT_CONVEX_ROUTE_DECISION_POINT=0` restores the #1066 policy, which is kept
intact and tested.

## 26. #1182 exact continuous (simplex/CNF) lowering: the speed motive is falsified; a capability motive survives (2026-09-05)

**Hypothesis under test** (RFC #1123, deferred to #1182, from Theorem 1 of
[arXiv:2601.03906v1](https://arxiv.org/abs/2601.03906v1)): replacing each disjunction
by its exact continuous simplex lowering removes the selector binaries and therefore
gives a *faster certified* solve than discopt's big-M / hull lowering.

**Kill criterion, fixed before running.** On the in-repo native GDP corpus
(`benchmarks.gdplib_native`, SCIP/BARON-certified optima) the simplex arm must reach
the same certified optimum as big-M/hull on at least one instance with strictly less
wall time or fewer nodes. If it certifies nothing the classical lowerings certify, the
hypothesis is falsified for certified global solving.

**Probes** (`scratchpad/issue1182/`, each printing an executed-comparison count and
exiting non-zero at zero, per CLAUDE.md §6): `E1_entry_experiment.py` (real GDP
corpus), `E2_paper_class.py` (the paper's own obstacle-avoidance class),
`E3_bigm_refusal.py` (capability), `E5_corpus_refusal_scan.py` (GDPlib scan). Logs
are committed beside them. Both arms go through the same `Model.solve` certified path
— which is the comparison the paper does *not* make: its §5 runs local Ipopt on both
sides, and its "Big M" baseline also eliminates the binaries continuously.

### E1 — real corpus, 60 s cap, `load average 0.17` at start

| instance | arm | status | objective | bound | nodes | wall | clauses | literal occ. | weight vars | rows | Jac. nnz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jobshop | big-m | optimal | 11 | 11 | 13 | 0.32 s | – | – | 6 | 12 | 30 |
| jobshop | hull | optimal | 11 | 11 | 7 | 0.04 s | – | – | 6 | 42 | 96 |
| jobshop | **simplex** | optimal | 11 | 11 | **251** | **4.82 s** | 3 | 6 | 6 | 6 | 18 |
| ex1_linan_2023 | big-m | optimal | −0.9996 | −0.9996 | 15 | 5.31 s | – | – | 9 | 20 | 45 |
| ex1_linan_2023 | hull | optimal | −0.9996 | −0.9996 | 9 | 5.86 s | – | – | 9 | 40 | 92 |
| ex1_linan_2023 | **simplex** | **feasible** | −0.9996 | −1.0166 | 437 | **60.1 s (cap)** | 48 | 224 | 224 | 96 | 144 |
| small_batch | big-m | optimal | 167427.66 | 167427.65 | 3 | 7.21 s | – | – | 9 | 34 | 73 |
| small_batch | hull | optimal | 167427.66 | 167427.65 | 3 | 10.07 s | – | – | 9 | 55 | 121 |
| small_batch | **simplex** | optimal | 167427.66 | 167427.65 | 3 | 11.66 s | 24 | 72 | 72 | 48 | 100 |

Two independent runs gave identical node counts (251 / 437 / 3), so the ordering is
not a timing artefact. **0 of 3 instances** meet the kill criterion, and
`ex1_linan_2023` regresses from certified to uncertified. The size table is also the
answer to #1182's requirement 4: on `ex1_linan_2023` the *declared* rows barely move
while clauses go 0 → 48 and weight variables 9 → 224, because each `a == v` disjunct
is two predicates and CNF distribution is multiplicative (2⁴ + 2⁵ = 48 clauses). A
clause count alone would have hidden that.

### E2 — the paper's own class, where CNF distribution is a no-op

Obstacle avoidance for a discrete-time double integrator: one 4-way disjunction of
*single* linear predicates per step, i.e. 1 clause and 4 weights, the shape most
favourable to the lowering. Two interleaved repetitions, 60 s cap:

| steps | arm | status | objective | bound | nodes | wall (rep 0 / rep 1) |
|---|---|---|---|---|---|---|
| 3 | big-m | optimal | 15.2 | 15.2 | 97 | 0.89 s / 0.53 s |
| 3 | hull | time_limit | – | 14.4 | 3999 / 4191 | 60.4 s / 60.1 s |
| 3 | **simplex** | feasible | 15.2 | 15.156 | 761 | 31.8 s / 32.3 s |
| 5 | big-m | optimal | 2.4414 | 2.4414 | 347 | 2.10 s / 2.28 s |
| 5 | hull | time_limit | – | 2.4 | 2015 | 60.2 s / 60.2 s |
| 5 | **simplex** | feasible | 2.4414 | 2.4 | 1433 / 1401 | 60.0 s / 60.1 s |

The lowering finds the optimal incumbent and cannot close the gap; big-M certifies in
under 3 s. So the loss is not an artefact of instance selection — it holds on the
class the mechanism was designed for. **The speed motive is falsified and recorded as
such: `"simplex"` is opt-in, is never selected by `gdp_method="auto"`, and this
section is the reason.**

### E3/E5 — what survives: a capability, not a speed claim

discopt's big-M pass refuses a disjunct row whose interval enclosure is unbounded;
the Furman–Sawaya–Grossmann hull refuses a row that is not finite at the origin
(`HullPerspectiveOriginError`). Theorem 1 needs neither. E3 measures all four cells:

| fixture | big-m | hull | simplex |
|---|---|---|---|
| unbounded `x` in `x <= 1` | **refused** | optimal | optimal |
| finite box (control) | optimal | optimal | optimal |
| `log(x) <= 0`, `x` unbounded above | optimal | **refused** | optimal |
| `1/x <= 1` on a box straddling 0 | **refused** | **refused** | **optimal, bound 0** |

The last row is a class no lowering in this tree could handle before. E5 asks how
often it occurs on a real corpus rather than in a constructed fixture (the #727 RLT
lesson), scanning every disjunct row of 17 GDPlib models with Pyomo's own interval
propagation and an origin evaluation:

| | disjunct rows | unbounded enclosure (big-M refuses) | non-finite at origin (hull refuses) | both |
|---|---|---|---|---|
| **17 GDPlib models** | **11,058** | 18 | 79 | **18** |

The 79 are `gdp_col` (28), `hda` (33) and `stranded_gas` (18); the 18 that hit both
are `stranded_gas`, where the row is `log` of a capacity sum whose box includes 0 —
an economy-of-scale sizing idiom, not a one-off. That is the concrete fixture #1182's
entry condition asks for, and it is a *capability* fixture: on those rows the exact
continuous lowering is not faster than the alternatives, it is the only one there is.

### E4 — the SOS1 reference, which only exists on MPEC models

Requirement 4 of #1182 asks for a benchmark against "exact GDP/SOS1 references".
E1/E2 cover the exact GDP references. SOS1 in this tree is not a general
disjunction lowering — it is one of `discopt.mpec`'s complementarity encodings — so
the SOS1 comparison exists only where a complementarity does. A relation
`0 <= f ⊥ g >= 0` is reachable through four exact encodings, all through the same
certified `Model.solve`; `method="scholtes"` is deliberately absent, being a
homotopy of *local* solves whose result is not a certificate.

| model | arm | status | objective | nodes | wall | source `min(f, g)` | certified |
|---|---|---|---|---|---|---|---|
| distance | sos1 | optimal | 1 | 3 | 0.36 s | 9.3e−11 | yes |
| distance | gdp/big-m | optimal | 1 | 3 | 0.04 s | 6.1e−11 | yes |
| distance | gdp/hull | optimal | 1 | 3 | 0.05 s | 7.5e−12 | yes |
| distance | **gdp/simplex** | optimal | 1 | 11 | **15.70 s** | **1.4e−17** | yes |
| chain4 | sos1 | optimal | 1 | 29 | 0.06 s | 2.2e−11 | yes |
| chain4 | gdp/big-m | optimal | 1 | 9 | 0.06 s | 4.9e−12 | yes |
| chain4 | gdp/hull | optimal | 1 | 9 | 0.18 s | 6.6e−12 | yes |
| chain4 | **gdp/simplex** | optimal | 1 | **5** | **14.65 s** | 5.5e−08 | yes |

All four certify the same optimum, so the encoding is sound here too. The
node-count signal is **mixed** — the simplex arm wins on `chain4` (5 vs 9) and
loses on `distance` (11 vs 3) — while the wall-time signal is uniformly bad:
100–250× slower, and near-constant at ~15 s across two models of different size,
which suggests a per-node cost in the spatial path rather than search. Stated
plainly so it is not read as a partial win: a node-count win that costs 244× wall
does not meet §5's *net-positive* bar, and one of the two models is a synthetic
chain I wrote for this probe rather than a corpus instance (`distance` is the
repo's own `test_mpec.py` fixture). The one column where the lowering is
consistently better is the **source complementarity residual** on `distance`
(1.4e−17 against 6.1e−11), which is a property of the encoding being exact rather
than regularized — worth recording, not worth a default change.

**Retraction of the framing in #1182's own text.** The issue asks for "a model where
the deferred lowering is expected to beat the exact GDP/SOS1 path" and reads that as a
performance question. E1/E2 answer it negatively and that answer is binding: no
default may be changed on this mechanism, and any future claim that it is faster owes
a measurement that contradicts these two tables first.
## 27. #1180 per-node layer split, post-tape: the marshaling era is over, and the remaining Python is worth 1.20× (2026-09-06)

**Claim under test.** `baron-gap-plan.md` §1.3's attribution of `nvs05` —
`python 82.5 % / jax 12.3 % / rust 3.4 %`, with `pounce.Problem.solve` marshaling
scalars across the Python/JAX boundary one point at a time — and its conclusion
that "`solve_lp_warm_csc_py`: 0.67 s — the node LP is nothing".

**Instrument.** cProfile self time aggregated across binding boundaries (never a
single-layer label), with a clean unprofiled control arm reporting the
`discopt._timing` FFI split and the profiler's own distortion (median 1.23×
wall). 66 in-repo instances at 20 s, 5185 nodes, 333 executed assertions.
Probes in `discopt_benchmarks/scripts/issue1180_*.py`; full record in
`docs/dev/issue-1180-per-node-layer-split-2026-09-05.md`.

**Result: the split has inverted.**

| layer | corpus (wall-weighted) | `nvs05` | §1.3's `nvs05` |
|---|---:|---:|---:|
| POUNCE native (IPM + tape) | **47.7 %** | 26.1 % | 0.1 % |
| `discopt._rust` (LP/MILP) | 15.0 % | **50.8 %** | 3.4 % |
| `discopt` Python | 15.4 % | 8.3 % | — |
| evaluator callback glue | 9.2 % | 4.7 % | — |
| other Python + numpy/scipy | 12.8 % | 10.2 % | — |
| **jax / XLA** | **0.00 %** | **0.00 %** | 12.3 % |

The dominant seam is the NLP solve itself (`nlp_pounce.solve_nlp`, 57.4 % of
corpus wall; the native IPM alone is 44.9 % of *all* self time). 1.95 M
derivative callbacks cost **11.5 s of tape arithmetic wrapped in 47.7 s of Python
frames** — so the whole marshaling story is now worth at most 1.10× corpus-wide,
against the 82.5 % it used to be. `nvs05`'s LP dominance is a two-instance
phenomenon (`nvs05` 41.6 % OBBT, `nvs09` 30.0 %, every other instance ≤ 10.6 %).

**Falsified, with the replacement measured.**

| withdrawn | measurement |
|---|---|
| "the node LP is nothing" (0.06 % of panel wall) | 9.3 % + 3.6 % corpus-wide; 50.8 % on `nvs05` |
| #764's "~70 % of an OBBT probe is Python marshaling" (tanksize, 2026-07) | **1–3 %** on four instances (nvs05 7.82 ms/probe = 7.77 native + 0.05 Python) — the persistent-CSC + warm-basis work removed it |
| "every node re-solves from scratch" | the parent point already beats a cold midpoint on 7 of 10 instances (gkocis 3.4×, tanksize 3.2×) |
| a full `pounce.WarmStart` across nodes would help | 0.53×–1.95×, **median 0.99×** on iteration count — 4 better, 4 worse, 2 neutral. Kill criterion met, not built. It is also structurally blocked: on `tls2`, 10 of 12 consecutive node NLPs have different `(n, m)` because the cut pool changes the row count |

**What shipped (bound-neutral).** The one in-repo lever the measurement leaves:
`_timing.charge` as a `__slots__` context manager instead of a generator
(1.93 µs → 1.05 µs per `with`, entered once per derivative callback), and
`TapeNLPEvaluator._x` handing pounce a contiguous `float64` array instead of a
per-callback Python list (`nvs05` n=15: 2.11 µs → 0.20 µs; `4stufen` n=157:
12.29 µs → 1.52 µs — the cost scaled with `n` while the arithmetic under it did
not). Gate: 1610 bit-identity comparisons over 66 instances, 0 mismatches; smoke
1133 passed; A/B interleaved in one process, **52 of 53 comparable rows exactly
neutral** on nodes/objective/bound, median **1.198×** on rows with ≥ 1 s of wall
(1.221× on the callback-heavy panel).

**Three methodological results worth carrying forward.**

1. *A deterministic budget makes node counts comparable but makes wall clock
   meaningless for any phase whose stopping rule **is** the wall budget.* On
   `beuster`, `deterministic=True` produced a reproducible **0.516×** that was
   pure artifact: the two arms issued **3858 OBBT probes against 942** for the
   same 3 nodes. On the ordinary wall budget the same instance honours its limit
   and returns **5× the nodes at a 30 % tighter dual bound**. Read neutrality and
   speed from different arms, and exclude limit-terminated rows from both.
2. *`node_callback` is a routing signal, not an observer.* `alan`, fresh
   subprocesses, both orders, same 13 nodes and objective: **54 POUNCE solves and
   11 130 tape evaluations without it, 1 and 0 with it.** Any profile taken with
   one attached is a profile of a different engine — `profile_instance.py`
   attaches one by default.
3. *A replay-based "pure-binding floor" compares the wrong population.* An
   earlier version of the probe-LP instrument replayed one captured call against
   the raw binding and reported a **negative** Python overhead, because the
   captured call cold-starts while the in-loop population is warm-started. Two
   nested timers in the same run have no such failure mode. This is the same
   shape as the measurement #764's 70 % figure came from.

**Open, not fixed here.** `clay0303hfsg` is **not reproducible under
`deterministic=True`**: four identical repeats on unmodified code give two
different objectives (55092.52 ×3, then 46785.55) at an identical 27 nodes, with
the dual bound agreeing to 12 significant figures — the *incumbent* moves, so the
nondeterministic component is a primal heuristic. That is a `deterministic=`
contract defect and a live hazard for every bound-neutrality gate that trusts the
flag.
