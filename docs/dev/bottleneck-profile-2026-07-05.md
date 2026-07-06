# Bottleneck profile — 2026-07-05 (easy-instance per-node cost + dual-bound stall vs BARON)

**Date:** 2026-07-05
**Status:** measured (measurement-only; no production code changed)
**Scope:** the two problem classes the 2026-07-05 BARON comparison
(`reports/global_opt_baron_vs_discopt_2026-07-05T10-47-11.json`, 61 MINLPLib
instances, 60 s) exposed: **Class A** (massive apparent per-node cost on
instances BARON solves in <1 s: fac2, nvs01, m3, nvs13) and **Class B**
(correct incumbent, dual bound doesn't close: flay03m, st_e36, tls2, nvs05,
nvs09, hda, clay0303hfsg). Also: the budget-overrun mechanism (contvar,
heatexch_gen3) and the fixed JIT/setup tax.
**Referenced by:** follows `docs/dev/bottleneck-profile-2026-07-02.md` (B1–B6)
and the Phase-D sections of `docs/dev/certification-gap-plan.md` (§8, dated
2026-07-05). This pass explicitly tests the user challenge to the Phase-D
conclusion: *"POUNCE is not that much slower than Ipopt, especially on smaller
problems — there must be substantial room on the discopt side."*

> **Method.** Branch `perf-profile-2026-07-05` at `origin/main` (31683a25).
> Apple M4 Pro (14-core, arm64), Python 3.12.11, JAX 0.10.2 (`JAX_PLATFORMS=cpu`,
> `JAX_ENABLE_X64=1`), pounce **0.7.0** (note: Phase-D used 0.8.0), numpy 2.5.1,
> scipy 1.18.0, `maturin develop --release`. Instances from
> `python/tests/data/minlplib_nl/`, `from_nl(path).solve(time_limit=60,
> gap_tolerance=1e-4)`. Three independent measurement modes per instance via
> `discopt_benchmarks/scripts/profile_instance.py` (new, reusable):
> 1. **clean** — uninstrumented wall + `SolveResult` phase fields
>    (`root_time`, `root_bound`, T0.3 `solver_stats` timers) + a
>    `node_callback` bound/incumbent trace (verified node-count-neutral on
>    nvs13: 19 nodes, identical objective, with and without the callback);
> 2. **cprofile** — full cProfile with fixed attribution buckets (C-extension
>    entries included, so POUNCE/Rust builtins are timed directly); overhead
>    vs clean ≈ +10–85 % depending on Python-callback density — used for
>    *shares*, never for totals;
> 3. **sampled** — in-process 100 Hz stack sampler (py-spy needs root on
>    macOS and was unusable; the sampler is `--sample-hz` in the same script).
>    Native time is attributed to the calling Python frame.
> `faulthandler.dump_traceback_later` (`--dump-stack-after`) gave mid-run
> stacks for the deadline-overrun runs. Raw artifacts (JSON, pstats, collapsed
> stacks) were produced in the session scratchpad; the durable numbers are
> below.

> **Reproducibility caveat (honesty).** On this machine two "Class B" probes
> from the 07-05 report actually **certify inside the budget**: flay03m
> `optimal` in 46.5 s / 111 nodes (report: time-limit, 18.4 % gap, 69 nodes)
> and clay0303hfsg `optimal` in 25.6 s / 229 nodes (report: time-limit,
> 19.1 % gap). The report run was evidently slower per unit work (load /
> different machine); the *shape* of where time goes is stable across all
> three measurement modes here, and the class conclusions do not depend on
> whether the 60 s line is crossed. fac2/nvs01/m3/nvs13/st_e36/tls2/nvs05/
> nvs09/hda reproduce the report's statuses.

---

## 1. Executive summary — ranked bottlenecks (measured)

**The POUNCE-hypothesis verdict up front:** the user's challenge is
**confirmed for Class A and for most of Class B**. On the profiled panel,
POUNCE's *per-solve* latency is 10–64 ms — unremarkable for these problem
sizes and nowhere near the 0.5–2.3 s/solve outliers of the Phase-D
(ex1252a/heatexch_gen1) class. Where POUNCE seconds dominate the wall, the
driver is **call volume commanded by discopt-side policy** (thousands of
heuristic sub-NLPs), not subsolver speed. Two instances (nvs01, st_e36) are
dominated by the **in-house Rust simplex**, with POUNCE at 15 %/22 %. The
Phase-D claim "the POUNCE subsolver is the #1 wall lever" does **not**
transfer to these classes (§5).

Ranked bottlenecks:

1. **B7 — Unbudgeted LNS `local_branching` enumeration** (Class A fac2;
   Class B flay03m, and via RENS the root of tls2). `_lns_k_schedule =
   (2, 5, 10)`; with 12 binaries the enumeration path of
   `primal_heuristics.local_branching` issues `sum(C(12,r), r=0..k)` full
   POUNCE sub-NLPs: **79 at k=2, 1586 at k=5**. Measured: fac2 = 1665 subnlp
   calls (= exactly 79+1586) / **22.2 s of a 23.5 s solve (85.8 % of samples)**;
   flay03m = 3330 calls (= 2×1665) / **48.9 s of 46.5–52.5 s (96.5 % of
   samples)**. The caller passes a ≤2 s slice (`submip_time_limit=_lb_slice`)
   but the ≤`max_binaries` enumeration branch **ignores any time budget and
   never polls the deadline** — a k=5 call runs ~22 s against an intended 2 s
   slice, even when the incumbent is already the known optimum (it was, on
   both). Counterfactual `rens=False` does *not* fix it (fac2 27.3 s,
   flay03m 91.7 s): the same enumeration migrates to the in-tree LNS call
   site. The fix has to be at `local_branching` itself.
2. **B8 — Rust dual-simplex warm-restart stall** (nvs01; also the st_e36
   volume cost). nvs01: of 41 `solve_lp_warm_std` calls totalling 17.5 s,
   **two warm re-solves take 8.71 s and 8.70 s each** on a 117×25 / 118×25
   tangent-augmented LP (the other 39 calls: 4–6 ms). That is **79 % of
   nvs01's 22.0 s wall**, inside `_separate_univariate_square`'s
   append-rows-and-re-solve loop. st_e36: 2 398 warm-simplex calls at
   13.3 ms mean = 31.9 s = **53 % of wall** on a 2-variable problem (85 % of
   wall under `solve_at_node`, incl. relaxation rebuilds).
3. **B9 — Separation re-solve volume, and one separator still on the POUNCE
   LP IPM** (nvs09, st_e36, nvs01). nvs09: **92.6 % of 60 s** is
   `_separate_multilinear` → **`lp_pounce._solve_core`** (99.5 % of that
   separator's samples) — the multilinear separator's per-round re-solves go
   through the POUNCE LP IPM on this path; the #484 Phase-D routing
   (`DISCOPT_SEPARATION_LP_SIMPLEX`, default ON) covered edge-concave +
   strong-branching only. This is the "if separation LPs STILL dominate,
   that's news" case. st_e36/nvs01: same 8-round separate-and-re-solve
   pattern through the warm simplex (B8).
4. **B10 — Root-phase heuristic/NLP binge on the "no relaxation" flowsheet
   class** (hda, contvar, heatexch_gen3; root = 66–100 % of wall). hda:
   root 42–51 s of 64 s = Rust `PyModelRepr.presolve` **15.1 s in one
   call** + 21 root multistart POUNCE NLPs at ~600–780 ms (12.6 s tottime,
   20.4 s incl. JAX callbacks) + ~10.4 s JAX trace/dispatch (11 419 pjit
   cache misses, 784 traces) — and the MILP relaxation **omits 23
   constraints** ("cannot linearize log(...)/x**expr"), so there is **no
   dual bound at all** (`best_bound = 1e30` sentinel for all 7 nodes).
   contvar/heatexch_gen3 are the same class (§4: budget overrun).
5. **B11 — Weak root relaxation on specific structures** (Class B bound
   stalls; §3). st_e36's root bound (−304.5) is **bit-identical** for
   x1 ∈ [15,25] and x1 ∈ [24,25] — the pinning term's envelope is independent
   of the integer's box; only *fixing* x1 lifts it (fix x1=22 → certifies in
   5 nodes; 897 unfixed nodes go nowhere). nvs05/tls2/nvs09 root gaps
   87.7 % / 86.4 % / 69.0 % with slow-but-real climbs; clay0303hfsg root
   bound pinned at ~0 (big-M gated objective) for ~100 nodes, then climbs
   and certifies.
6. **B12 — POUNCE as the per-node NLP bound engine** (nvs05, tls2 — the one
   sub-class where the subsolver itself is the direct cost). nvs05: 834
   node-NLP `_attempt` calls = 53.3 s ≈ **64 ms/solve, 91 % of wall**; tls2:
   1497 node NLPs at 29 ms = 70 % of wall. Per Phase-D's own A/B (median
   7–11× vs Ipopt), an Ipopt-class engine would cut this to roughly 5–9 s
   (nvs05) — worth having — but the root gap (87.7 %) is what makes the node
   count high in the first place.
7. **Fixed setup tax** (visible only on trivial instances): `import discopt`
   0.10–0.14 s + **lazy imports inside the first `solve()` 0.4–0.6 s** +
   JIT/warmup ≈ 0.5 s (m3 second in-process solve: 3.41 s → 2.88 s, same 61
   nodes). Net of tax, m3 is still ~2.7 s vs BARON 0.04 s — the gap is real
   work (B7/B12-class), not tax.

**Where the prior "per-node cost" framing was wrong:** fac2's benchmark line
(32.5 s / 69 nodes ≈ 0.47 s/node) is a mis-attribution. Measured: root
processing = **22.1 s (94 %)**, B&B loop = 1.4 s / 69 nodes = **20 ms/node**,
already at the 0.018 s/node cert-gate target. Same for tls2 (root 25.5 s;
tree 18 ms/node × 1903 nodes) and clay0303hfsg (root 12.2 s = 48 %). nvs01's
"1.3 s/node" is actually two 8.7 s LP calls (B8).

### POUNCE wall-share vs per-solve latency (the hypothesis test, per instance)

| instance | POUNCE incl. callbacks (cum) | pure-Rust IPM (tottime) | calls | ms/solve | dominant caller | verdict |
|---|---:|---:|---:|---:|---|---|
| fac2 | 23.3 s (80 %) | 15.8 s (54 %) | 1 744 | 13.4 | LNS `subnlp` (1 665) | **volume, not speed** |
| flay03m | 49.4 s (94 %) | 34.2 s (65 %) | 3 557 | 13.9 | LNS `subnlp` (3 330) | **volume, not speed** |
| tls2 | 43.8 s (70 %) | 29.2 s (47 %) | 1 516 | 28.9 | node NLPs (1 497) | volume + engine |
| nvs05 | 55.1 s (91 %) | 32.5 s (54 %) | 1 206 | 45.7 | node NLPs (834 @ 64 ms) | **engine (B12)** |
| m3 | 2.8 s (60 %) | 1.6 s (35 %) | 258 | 10.7 | node NLPs (204) | volume, small |
| hda | 20.4 s (30 %) | 12.6 s (19 %) | 21 | ~600–780 | root multistart | engine on 722 vars |
| st_e36 | 13.4 s (22 %) | 11.5 s (19 %) | 1 928 | 7.0 | mixed | **not POUNCE (B8/B9)** |
| nvs01 | 3.5 s (15 %) | 2.6 s (11 %) | 430 | 8.1 | heuristics | **not POUNCE (B8)** |
| nvs13 | 0.48 s (19 %) | 0.13 s (5 %) | 44 | 10.9 | node NLPs | not POUNCE |
| nvs09 | ~4.8 % of samples | — | — | — | — | **not POUNCE-NLP (B9: `lp_pounce` LP-IPM 92.6 %)** |

(Shares are of the cProfile-profiled total; sampler cross-checks on fac2
[88.6 % pounce frames], flay03m [98.0 %], st_e36 [24.0 %], nvs09 [4.8 %]
agree within a few points.)

**Verdict, stated plainly:** on Class A the prior "POUNCE is the residual
bottleneck" conclusion is **falsified** — fac2/m3/nvs13 POUNCE per-solve is
10–14 ms (near-Ipopt territory for these sizes) and nvs01 is a simplex
pathology; the room is on the discopt side, as the user hypothesized. On
Class B it is falsified for flay03m (heuristic volume), st_e36 (simplex +
separation volume), and nvs09 (separation LP-IPM routing), and **survives
only in a refined form** on nvs05/tls2 (per-node NLP engine, 29–64 ms/solve)
and hda's 722-var root NLPs — the class where Phase-D's Ipopt A/B actually
measured it.

---

## 2. Per-instance phase breakdown

Wall/phase columns from the **clean** runs; attribution shares from
cProfile + sampler. "tree ms/node" = (solve − root_time)/nodes.

### Class A

| phase | fac2 | nvs01 | m3 | nvs13 |
|---|---:|---:|---:|---:|
| total wall (clean) | 23.45 s | 21.97 s | 3.25 s | 1.51 s |
| `from_nl` parse | 0.002 s | 0.002 s | 0.002 s | 0.007 s |
| root processing (`root_time`) | 22.06 s (94 %) | 4.09 s (19 %) | 1.95 s (60 %) | 0.91 s (60 %) |
| B&B loop | 1.39 s | 17.88 s | 1.30 s | 0.60 s |
| nodes / tree ms-per-node | 69 / **20 ms** | 17 / 1 052 ms | 61 / 21 ms | 19 / 32 ms |
| status / verdict | optimal, exact | optimal, exact | optimal, exact | optimal, exact |
| BARON wall | 0.04 s | 0.1 s | 0.04 s | 0.1 s |

Attribution (share of profiled wall):

- **fac2** (convex-quadratic objective + quadratic constraints; routed through
  the generic `_solve_nlp_bb`, *not* the convex-MIQP fast path, which gates on
  `problem_class == MIQP` — fac2's nonlinear constraints exclude it):
  POUNCE-cum 80 % — of which `rens` → nested `_solve_nlp_bb` → LNS
  `local_branching` → 1 665 `subnlp` calls = 22.2 s. The *entire* Class-A
  story is one root RENS call whose sub-solve runs two enumeration rounds
  (k=2: 79, k=5: 1 586). Remainder: JIT 0.75 s, lazy imports 0.54 s.
- **nvs01**: `rust_lp_solve` 17.5 s (75 %): 41 calls, **two at 8.7 s each**
  (warm dual-simplex re-optimization after `_separate_univariate_square`
  appended tangent rows; LP is 117–118 rows × 25 cols; all other calls
  4–6 ms). T0.3 timer agrees: `separate/univariate_square` = 17.45 s.
  POUNCE 15 %.
- **m3**: POUNCE 60 % (204 node NLPs @ 11.6 ms + 49 diving); JIT 0.48 s;
  lazy imports 0.43 s. Root 1.95 s = multistart + root separation; root
  bound starts at ~0 (pinned, gear4-style) and climbs to optimal by node 61.
- **nvs13**: `solve_at_node` 42 % (of which `build_milp_relaxation` 34 % —
  30 calls @ 29 ms on a 5-var problem), rust LP 16 %, POUNCE 19 %, lazy
  imports 17 %, JIT 18 %. At 1.5 s total, the fixed taxes are ~⅔ of the
  BARON gap.

### Class B

| phase | flay03m | st_e36 | tls2 | nvs05 | hda |
|---|---:|---:|---:|---:|---:|
| total wall (clean) | 46.5 s | 60.1 s (TL) | 60.3 s (TL) | 60.1 s (TL) | 64.1 s (TL) |
| root processing | 22.4 s (48 %) | 2.6 s (4 %) | 25.5 s (42 %) | 3.2 s (5 %) | 42.0 s (66 %) |
| B&B loop | 24.1 s | 57.5 s | 34.8 s | 56.9 s | 22.1 s |
| nodes / tree ms-per-node | 111 / 217 ms | 897 / 64 ms | 1 903 / **18 ms** | 821 / 69 ms | 7 / 5 525 ms |
| dominant sink (share) | LNS subnlp 94 % | simplex 53 % + sep 28 % | POUNCE 70 % | node NLPs 91 % | presolve 22 % + POUNCE 30 % + JAX ~15 % |
| status | optimal here | feasible, gap 23.8 % | feasible at 58.7 s | feasible, gap open | no incumbent, no bound |
| BARON wall | 0.3 s | 0.1 s | 0.04 s | 0.6 s | 0.2 s |

- **flay03m**: root 22.4 s = RENS (24.2 s cum) with the same LNS enumeration
  inside; the 23.5 s mid-search "stall" (bound flat at 40.0, nodes 49→87) is
  in-tree LNS calls, not bound trouble — the bound then climbs and
  **certifies at node 111**. Root gap 36.8 % (root bound 30.98 vs 48.99) is
  real but survivable: nodes are ~20 ms without the heuristic tax.
- **st_e36** (2 vars, x1 integer [15,25]): 85 % under `solve_at_node`; rust
  simplex 31.9 s tottime (2 398 calls) + `build_milp_relaxation` 9.2 s
  (506 calls) + POUNCE 13.4 s. `separate/multilinear` = 11.65 s (T0.3).
  Bound: see §3.
- **tls2**: root 25.5 s (RENS/feasibility hunt: `primal_heuristics` 26.6 s
  cum); tree itself runs at 18 ms/node and the bound climbs steadily
  0.72 → 5.2 (opt 5.3) — with the root tax removed it would have closed at
  roughly the same node count. `rens=False` counterfactual: 41.1 s.
- **nvs05**: 91 % node NLPs (834 @ 64 ms, incl. retry-from-alternative-start
  policy: ~1.6 NLP per node with `subnlp` extras). Root gap 87.7 %; bound
  climbs 0.67 → 5.32 (trace) but the **reported** final bound is 1.35 (see
  §6 anomaly).
- **hda** (722 vars): root 42–51 s = 15.1 s Rust presolve (one call) +
  21 multistart POUNCE NLPs (~600–780 ms each) + one 8.05 s `subnlp` +
  ~10.4 s JAX pjit dispatch/trace (11 419 cache misses / 784 traces — this
  class **does** re-trace; §5) + rust LP/MILP 10 s. 23 constraints omitted
  from the relaxation → no dual bound ever (sentinel 1e30) → nothing can
  fathom; 7 nodes at TL. Python churn is also visible (366 791
  `np.asarray`, 32.2 M `core.py:size` calls = 3.3 s + 0.8 s).

---

## 3. Class B — root gap and the binding deficiency

"Moves?" = does the global dual bound improve during the 60 s budget.

| instance | root bound | known opt | root gap | bound at TL (trace) | moves? | binding deficiency (evidence) |
|---|---:|---:|---:|---:|---|---|
| flay03m | 30.984 | 48.9898 | 36.8 % | 48.99 (certified @111 nodes, 46.5 s) | yes | **(none — heuristic tax B7 masks a working search).** Root gap from big-M/McCormick-div is real but the tree closes it in ~1 s of actual node work. |
| st_e36 | −304.500 | −246.0 | 23.8 % | −304.498 after 897 nodes | **no** | **(a) envelope pinning.** Root bound bit-identical (−304.5000003055) for x1 ∈ [15,25], [20,25], and [24,25]; only *fixing* lifts it: x1=22 → root −278.9, certifies in 5 nodes; box [5,5.5]×[20,22] → root = −246.0000 exactly. The pinning term's under-envelope (nested `pow(·,2)` composites with negative coefficients) is independent of the integer's interval, so no amount of interval branching moves the bound. |
| tls2 | 0.718 | 5.3 | 86.4 % | 5.2 | yes, steadily | **(root-cost starvation + envelope).** 42 % of budget burned at root; tree itself climbs 0.72→5.2 at 18 ms/node. sqrt/mult structure; bound would close with ~2× more node budget. |
| nvs05 | 0.674 | 5.4709 | 87.7 % | 5.32 (trace) / 1.35 (reported — §6) | yes, slowly | **(a)+(engine).** Loose product/power envelopes give an 87.7 % root gap; 64 ms/node POUNCE NLPs (B12) cap throughput at ~14 nodes/s. |
| nvs09 | −72.900 | −43.134 | 69.0 % | −59.24 @159 nodes | yes, slowly | **(c) separation cost, not cut absence:** 92.6 % of wall is the multilinear separator's LP re-solves through `lp_pounce._solve_core` (B9) → ~2.6 nodes/s. The cuts work (bound climbs every trace sample); they are just catastrophically expensive to separate. |
| hda | none (1e30) | −5964.5 | ∞ | none | n/a | **(a′) relaxation coverage:** 23 rows omitted ("cannot linearize `log(<general expr>)` / `x**expr`") → no dual bound exists; plus B10 root binge. BARON reports this instance infeasible in 0.2 s (its own issue); discopt neither bounds nor finds a point. |
| clay0303hfsg | ≈ 0 (−1.2e−5) | 26 669.1 | ~100 % | 26 669.1 (certified @229 nodes, 25.6 s) | yes (after ~100 nodes flat) | **(d)-flavored:** big-M/disjunctive structure — objective gated by binaries, root bound structurally pinned at 0 (gear4-pattern per performance-plan §6); bound lifts only once enough binaries are branched. Root cost 12.2 s (48 %) on top. |

Two structural sub-classes for the bound stalls:

- **Pinned-at-a-constant** (st_e36, clay0303hfsg, m3's root, gear4 from the
  06-24 record): the root relaxation value is a structural constant that
  interval-shrinking cannot lift — only fixing (st_e36) or branching the
  gating binaries (clay) moves it. For st_e36 this is an *envelope* defect
  (fix class: even-power composite envelopes); for clay/gear4 it is the
  known branch-and-reduce gap (cert-gap-plan C2).
- **Loose-but-live** (nvs05, tls2, nvs09): the bound climbs monotonically;
  wall per node (B8/B9/B12) is what prevents closure inside 60 s.

FLay/CLay big-M check (asked in the brief): flay03m's nonlinearity is only 3
`div` rows (area constraints); the rest is linear big-M rows. discopt's tree
handles them adequately *once it gets CPU* (111/229 nodes to certify); the
big-M weakness shows up as the pinned root bound (clay: 0; flay: 30.98), not
as an unsolvable tree. BARON's <1 s comes from a tight root (its probing +
big-M-aware bounds), but with B7 removed the discopt gap on flay03m is
~5 s vs 0.3 s, not 64 s vs 0.3 s.

---

## 4. Budget overrun (contvar 221 s, heatexch_gen3 81.5 s vs a 60 s limit)

Measured with `--dump-stack-after 70` (faulthandler, repeat):

- **contvar**: `solve()` returned after **221.0 s** (root_time = 220.9 s).
  All three dumps (t = 70/140/210 s) show the identical stack:
  `solve_model` (solver.py:6068) → `feasibility_pump`
  (primal_heuristics.py:290) → `nlp_pounce.solve_nlp` → POUNCE Hessian
  callback → `sparse_hessian.sparse_hess_values` (line 162) → **XLA
  `backend_compile_and_load`**. The overrun is a single, uninterruptible
  first-time XLA compilation of the sparse Hessian for the root
  feasibility-pump NLP (~150 s+ of compile on this 297-var/flowsheet model).
  No deadline poll can fire inside `jit` compilation, and the POUNCE
  `max_wall_time` option cannot help either — the time is spent inside one
  Python callback, before the solver iterates.
- **heatexch_gen3**: same class — 81.5 s, root_time = 81.1 s, the t = 70 s
  dump is inside `nlp_pounce.solve_nlp:148` (jax_time = 58.8 s). No bound
  (1e30 sentinel; AMP-style constraint omissions like contvar/hda).

So the loop that "fails to poll the deadline" is not a loop: it is the root
heuristic's NLP solve whose *first Hessian evaluation* compiles for longer
than the whole budget. Any fix must gate *entering* the compile (estimate or
budget-check before requesting the Hessian path / choose a Hessian-free
fallback when the remaining budget is small), not poll inside it.

---

## 5. Falsifications & corrections of prior claims

Per performance-plan §6 house style — superseded claims recorded, not
overwritten:

1. **"Phase D: the POUNCE subsolver is the #1 wall lever"**
   (certification-gap-plan §8, 2026-07-05). *Correction:* true only for the
   NLP-heavy class it was measured on (ex1252a/heatexch_gen1/casctanks) and,
   in refined form, for nvs05/tls2/hda root NLPs. On the easy Class A and on
   flay03m/st_e36/nvs09 the #1 levers are discopt-side: unbudgeted LNS
   enumeration (B7), simplex warm-restart stall (B8), and separation
   re-solve volume/routing (B9). POUNCE per-solve on these instances is
   7–64 ms — the pooled "median ~7× vs Ipopt" does not make it the wall
   driver when the calls are 13 ms and number in the thousands *by policy*.
2. **"XLA compilation: resolved (≤1 % of wall), 0 recompiles across
   boxes"** (bottleneck-profile-2026-07-02 §2). *Correction:* verified there
   for the gear4/nvs17 class and still true for it — but it does **not
   generalize**: hda re-traces per solve family (784 traces / 11 419 pjit
   cache misses, ~10.4 s ≈ 15 %), and contvar spends >150 s (>68 % of its
   run) inside a single `backend_compile_and_load` (§4). The compile-cache
   win is class-local.
3. **"OBBT inner loop is the dominant orchestration cost"**
   (bottleneck-profile-2026-07-02 §1 item 1). *Not contradicted, but not
   general:* on all 11 instances profiled here OBBT is ≤4 % (obbt_root
   ≤1.0 s everywhere). The 07-02 statement holds for its spatial panel only.
4. **The benchmark report's per-node framing for Class A** ("fac2 0.47 s/node,
   nvs01 1.3 s/node"). *Correction:* fac2 tree = 20 ms/node (94 % of wall is
   the root RENS/LNS binge); nvs01 = two 8.7 s LP calls, not uniform node
   cost. "s/node = wall/nodes" is a misleading metric when root/heuristic
   phases dominate; gates should use root-phase and tree-phase timers
   (T0.3) separately.
5. **"cold simplex ≈ 1 840 µs average"** (07-02 §1 item 2) as a
   characterization of simplex cost. *Refinement:* the tail, not the mean,
   is the problem — a warm dual-simplex re-optimization after row appends
   can take **8.7 s** on a 118×25 LP (nvs01, twice). Averages hide the
   pathology.
6. **Phase-D LP routing "shipped default-ON" closing the separation-LP
   story** (#484). *Correction:* the multilinear separator still drives
   its re-solves through `lp_pounce._solve_core` on the nvs09 path
   (92.6 % of that instance's wall). The flag covered edge-concave and
   strong-branching call sites only.
7. **"F1 (LNS enumeration budget) removes tls2's ~15 s root cost"**
   (§1.1 "via RENS the root of tls2"; §7 F1 acceptance "tls2 root −≥15 s").
   *Correction (F1 implementation, `perf-f1-lns-enumeration-budget`, pounce
   0.7.0, M4 Pro):* on this machine tls2 never enters the `local_branching`
   enumeration path at its root — an instrumented solve records **0**
   `subnlp` calls / **0** `local_branching` invocations, and the single
   in-tree `local_branching` call costs 1.36 s. tls2's ~25 s root sink is
   **`rens` itself (2 calls = 25.7 s)** — RENS's own nested `_solve_nlp_bb`
   root/heuristic phase — which F1 does not own. Budgeting the enumeration
   correctly bounds fac2 (1665→158 sub-NLPs, 23.5 s→5.8 s) and flay03m
   (3330→395, 47.8 s→6.8 s), both still certifying with identical node
   count/objective, but leaves tls2's root unchanged (25.6 s→25.1 s). The
   tls2 root lever is **F4** (budget-gate the root heuristic NLP phase), not
   F1; the §7 F1 tls2 line is reassigned to F4. F1's class win (the
   `sum_r C(n,r)` enumeration blow-up on the ≤12-binary early-incumbent
   class) is confirmed and delivered.
8. **"F3 collapses nvs09's separation cost ~10–50× (nodes/s ≥10×, share
   <20 %)"** (§7 F3 acceptance; §1.3 "the same swap that gave 10–50× on the
   other separators applies"). *Correction (F3 implementation,
   `perf-f3-multilinear-separator-simplex`, pounce 0.7.0, M4 Pro):* the entry
   experiment falsifies the uniform-10× premise for **this** LP class. The
   multilinear hull LP is `min/max f(v)·λ s.t. Vᵀλ = x*, 1ᵀλ = 1, λ ≥ 0` with
   `2^n` λ columns — **not** a row-augmented re-solve, so the "warm" in the
   swap is cold-simplex-vs-IPM, exactly like #484's edge-concave routing (no
   cross-call basis reuse; `lp_simplex.solve_lp` is documented cold). On
   nvs09, `n`≈10 → **1024-column** hull LPs: a per-LP A/B shows the cold Rust
   simplex is **bimodal** — it solves ~81 % of them in **≈1 ms (100× faster
   than POUNCE's ~150 ms)** but on the remaining ~19 % (degenerate/
   ill-conditioned widest LPs) it **stalls to its 100 000-pivot default
   (~1.6 s)**, worse than POUNCE. (Bonus soundness note: the hull LP's
   feasible set `{λ ≥ 0, Σλ = 1}` is a simplex — *provably bounded* — so
   POUNCE's occasional `UNBOUNDED` verdict on the wide LP is a numerical
   false-negative that **drops a valid cut**; the exact simplex recovers it,
   and the intercept-recompute keeps every cut valid regardless of engine.)
   The sound fix is F2's philosophy applied to the *cold* path: a size-derived
   pivot cap → POUNCE per-LP fallback (F2's guard only covers the *warm* dual
   re-solve, so the plan's "F2 protects the new traffic" holds only partially).
   Result on nvs09 @60 s: simplex serves 564/698 hull LPs in 3.8 s (6.4 % of
   wall), but the **128 POUNCE fallbacks cost 44.3 s (73 %)** — that hard
   subset is *irreducible* (POUNCE is the better engine for it), so the
   separation share stays ~89 % and the wall floor does not move 10×. Measured
   deltas: **nodes/s 2.62 → 3.65 (1.4×)**, **bound@60 s −59.24 → −57.67
   (strictly tighter, valid vs the −43.134 oracle)**, incumbent unchanged.
   The `<20 %` share / `≥10×` nodes/s targets are **not met** and were premised
   on separation being removable *overhead*; on nvs09 it is intrinsic bound
   work whose hard tail is IPM-favourable. The class win (the ≤4-arity-cap
   narrow hull LPs, ~81 % here, → ~100× per LP) is real and sound; the residual
   nvs09 wall is the wide-LP IPM cost (a POUNCE-engine question, cf. F6/upstream
   jkitchin/pounce#182), not a routing one. Cert panel: **provably byte-identical**
   — no certifying instance reaches the multilinear separator (0 calls at 60 s
   on all 41, incl. the heavy cvxnonsep/fac2/tspn05/flay02m).
9. **"F4: gate the root heuristic NLP by a fitted `compile ~ f(model size)`
   curve; contvar 221 s"** (§4; §7 F4 mechanism/acceptance). *Correction (F4
   implementation, `perf-f4-root-budget-gate`, pounce 0.7.0, M-series arm64):*
   two premises fell to the entry experiment.
   (a) **The 221 s contvar overrun does not reproduce on this build:** contvar
   now returns in **60.7 s** (root 29.3 s) at `time_limit=60` — already within
   the §0.7 envelope. The 221 s was pounce 0.8.0 / an earlier snapshot; the
   reproducing overrun on this build is **heatexch_gen3 (80.7 s)** and, at
   `time_limit=60`, **hda (64.2 s)**. The measurement wins.
   (b) **Compile time is not a function of cheap model size** (the plan asked
   for a fitted curve). First-time sparse-Hessian XLA compile, standalone, over
   7 MINLPLib models: tls2 0.15 s (n=37, dense), fac2 0.15 s (n=66, dense),
   heatexch_gen1 1.1 s, hda 2.5 s (n=722), casctanks 5.0 s, heatexch_gen3 49 s
   (n=580), **contvar 186 s (n=296)**. Regressing `log(compile)` on `n_vars`
   gives **R² = 0.002** (contvar at n=296 compiles ~74× slower than hda at
   n=722); the same model varies 2.5→8.3 s run-to-run. The cost is governed by
   the DAG's shape/depth (contvar's nested `log/exp/division` chains), not any
   cheap scalar, and is not predictable in advance. Consequently the compile
   estimate is a **conservative floor** (dense path → 0.5 s; sparse
   compressed-HVP path, uncompiled → a 15 s risk-headroom constant), not a point
   predictor — used only to gate *primal-heuristic* entry, where over-estimating
   merely skips a heuristic (sound; never touches the dual bound).
   (c) **The real overrun on this build is not one uninterruptible compile but
   repeated post-deadline heuristic-NLP launches** that each overrun their own
   `max_wall_time` clamp (a nominal 3 s POUNCE solve runs ~10–15 s because each
   IPM iteration's exact Hessian is expensive). The fix gates *entry* into every
   root heuristic NLP by the remaining budget (a self-calibrating worst-case
   observed per-solve cost) and threads the absolute deadline into the looping
   heuristics (`diving`/`rins` poll before each sub-NLP; the multistart caps its
   extra starts by the observed per-start cost). Result: **heatexch_gen3
   80.7 → 60.9 s**, **contvar 60.7 s (unchanged)**, **hda 64.2 → 64.1 s**;
   full 61-instance panel at `time_limit=30`: **61/61 within §0.7** with **0
   objective changes and 0 lost incumbents** vs the gate-off baseline.
   (d) **KILL-CRITERION HIT (a second overrun site):** the out-of-panel large
   flowsheet **super3t** overruns `time_limit=30` at **74 s ungated / 40 s
   gated** — but its residual overrun is **not** a heuristic NLP (an instrumented
   solve makes **zero** `solve_nlp` calls). Its root time is spent in
   `_jax/term_classifier._compute_var_offset` — the McCormick relaxation
   term-classification build, an uninterruptible O(n·terms) pass that predates
   the heuristic phase. This is a **distinct overrun site** outside F4's scope
   (F4 halves it as a side effect but cannot close it); it needs its own
   entry/streaming gate and is filed as F4 follow-up **#507**.

---

## 6. Observations out of scope (flagged, not fixed)

- **Reported bound ≪ tree bound on nvs05:** the `node_callback` trace ends
  with `best_bound = 5.32`, but `SolveResult.bound = 1.348` (gap reported
  75.4 % instead of ~2.7 %). Either the frontier bound is not being adopted
  into the final result on this path, or the callback's `best_bound` is not
  the certified global bound. Both readings matter for benchmark scoring —
  worth a correctness-side look (docs/dev/correctness-issues.md candidate).
- **`best_bound = 1e30` sentinel** leaks into the `node_callback` API on
  the no-relaxation class (hda/heatexch_gen3) instead of `None`/`-inf`.
- **hda Python churn:** 32.2 M `core.py:size` calls / 21 M `abs` calls in a
  67 s solve (term_classifier / relaxation walk) — ~2 s, secondary to B10
  but a sign of per-call DAG re-walks on big models.

---

## 7. Fix candidates, ranked by measured expected wall-clock win

Ordered by (expected win on the measured panel) × (confidence). Regimes per
CLAUDE.md: *bound-neutral* changes must keep node_count/objective exactly
unchanged; heuristic-policy changes touch incumbent discovery (node counts
may legitimately change) but never the dual bound — they still face the
`incorrect_count ≤ 0` panel gate.

1. **F1 — Give `local_branching`'s enumeration path a real budget** (B7).
   Mechanism: poll `deadline`/slice inside the flip loop; when
   `sum_r C(n_bin, r) × ~14 ms` exceeds the slice, truncate or dispatch to
   the existing `_local_branching_submip` (which already takes
   `time_limit`). Affects: fac2, flay03m, tls2 root, every ≤12-binary
   instance with an early incumbent. Expected (from component arithmetic,
   not vibes): fac2 23.5 s → ~2.5–5 s (root 1.0 s [measured with rens off] +
   tree 1.4 s + capped heuristics ≤2 s/call); flay03m 46.5 s → ~7–12 s;
   tls2 −20 s at root. Note the `rens=False` counterfactual (fac2 27.3 s,
   flay03m 91.7 s) proves disabling callers just relocates the cost — the
   budget must live in the enumeration itself. Risk: heuristic-policy
   (primal-only); dual bound untouched; needs the cert-panel
   incorrect-count gate + a wall-clock regression probe.
2. **F2 — Simplex warm-restart stall guard** (B8). Mechanism: iteration or
   time cap on the warm dual-simplex re-optimize; on trip, fall back to a
   cold solve of the same LP (soundness unchanged — same optimum, different
   path). Evidence: nvs01's 2×8.7 s calls vs 4–6 ms siblings on
   near-identical LPs; a cold solve of these LPs costs ~5 ms. Expected:
   nvs01 22.0 s → ~4.5 s. Bound-neutral regime (objective/node_count must
   be exactly unchanged — the LP optimum is the same; only who computes it
   changes). Rust-side; needs `cargo test -p discopt-core` + cert-panel
   neutrality run.
3. **F3 — Route the multilinear separator's re-solves to the warm simplex**
   (B9), i.e. extend #484's `DISCOPT_SEPARATION_LP_SIMPLEX` to this call
   site (`mccormick_lp._separate_multilinear` → `milp.solve(backend=…)`
   resolution). Expected: nvs09 60 s (gap 39 %) → separation cost drops
   ~10–50× per Phase-D's own measurements of the same swap elsewhere;
   st_e36's 13.3 ms×2 398 volume also shrinks with F2's warm path.
   Bound-neutral-with-caveat exactly as #484 documented (degenerate-LP dual
   may differ; cut *validity* is intercept-recomputed): decide by the same
   empirical neutrality protocol.
4. **F4 — Budget-gate the root heuristic NLP/compile phase** (B10, §4).
   Mechanism: before a root heuristic NLP that will trigger a first-time
   Hessian compile, check remaining budget vs a measured compile-cost
   estimate for the model size; skip or use the Hessian-free path when it
   doesn't fit; cap root multistart count by remaining time. Expected:
   restores the 60 s contract on contvar (221 s → ≤60 s) and
   heatexch_gen3 (81.5 → ≤60), cuts hda's root 42 s substantially. Risk:
   heuristic-policy; no dual-bound effect. (The hard alternative —
   interruptible XLA compiles — does not exist.)
5. **F5 — Even-power composite envelope for the st_e36 pinning class**
   (B11). Mechanism: secant/tangent envelopes for `±(affine+quadratic)²`
   composites that actually contract with the participating variables'
   boxes (relaxation-catalog check first per its "do not rebuild" rule).
   Evidence: fixing x1 certifies in 5 nodes; the current envelope is
   interval-independent. Expected: st_e36 60 s/897 nodes → ~2 s/≤10 nodes.
   **Bound-changing regime**: feature-flagged, differential bound test +
   feasible-point sampling, default-off until nightly-green.
6. **F6 — Per-node NLP engine cost on the nvs05/tls2 class** (B12): either
   the Phase-D POUNCE performance filing (per-solve 7–11× vs Ipopt on
   identical problems) or reducing attempts (retry policy adds ~1.6
   NLP/node on nvs05). Expected: nvs05 55 s → ~8–15 s *of solver time* —
   but without F5-class root-gap work it still won't certify in 60 s
   (bound at 5.32/5.47 after 821 nodes). Pair with relaxation work.
7. **F7 — Fixed-tax trims** (lazy-import hoisting; JIT warmup reuse):
   ~0.9–1.1 s on every first solve; only matters for the sub-5 s class
   (m3, nvs13, ex1224, nvs08). Low risk, low ceiling.

Non-candidates (measured dead here): OBBT tuning (≤4 % everywhere on this
panel); further CSE/Q-extraction-style DAG shrinking for Class A (the DAG
walk is not where the seconds are, except hda's churn note in §6).

---

## 8. Artifacts

- `discopt_benchmarks/scripts/profile_instance.py` — the reusable
  measurement harness used for every number in this doc (clean/cprofile
  modes, 100 Hz in-process sampler with collapsed-stack output, bound/node
  `node_callback` trace, `--second-solve` JIT-tax probe,
  `--dump-stack-after` overrun stack dumps).
- Raw per-instance JSON/pstats/collapsed stacks were produced in the session
  scratchpad (not committed; regenerate with the script above — every table
  states its instance + mode).
