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

## 6. Stage 4 — Node-count reduction (CC3; highest leverage, strictest gate)

CC3 multiplies CC1+CC2: gear4's 5921 nodes are *why* it pays 810 recompiles and
47.7 s of Python. Fewer nodes wins on all three axes at once — but a tighter bound
or a mis-scored branch is exactly how a **false certificate** (nvs22 #277 / st_ph10
#306) is born, so this carries the heaviest correctness gate.

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
