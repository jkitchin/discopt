# DECOMP-1 (task #88): certification-effort decomposition — which lever dominates?

**Date:** 2026-07-10 · **Base:** `main` @ `16f25505` (includes root_gap/root_time #589,
dense-retry counters #591, centropy relaxation #597) · **Measurement-only** — no solver
changes in this PR.

## 0. TL;DR

The maintainer's directive was: *"we are still working way too hard to certify and not
succeeding."* This diagnostic decomposes that effort on a 10-instance MINLPLib panel +
the #598 AMP multi4N MILP, TL = 60 s, with per-node-LP status counting, Rust simplex
profile counters, and a bound/incumbent trajectory tap on the B&B tree.

**Verdict: Lever A (relaxation/dual-bound strength) dominates — 7 of 10 uncertified
runs — but two narrow, cheap Lever-B certification-plumbing defects throw away dual
bound the solver already proved, and should be fixed first.** Lever C (B&B machinery
with a tight relaxation) appears on exactly one instance (nvs19). One instance (tls2)
fails on the *primal* side — a lever the A/B/C taxonomy didn't anticipate.

Headline numbers:

- **10/11 runs found the exact oracle optimum; only 1/11 certified it** (st_e36).
- **Node-LP numerical failure is a non-factor on the spatial path: 0 numerical /
  iteration-limit / error exits in 711 McCormick node-LP solves** across the panel
  (2 sound root-probe declines: one `optimal`-with-no-safe-bound, one `unbounded`).
  The "node LPs take numerical exits → gap poisoned" hypothesis is **falsified as a
  panel-wide driver** — it is real but *specific to the in-house MILP B&B* (#598),
  and it reappears in a different guise as node-**NLP** taint (below).
- **Root gap > 20 % on 7 of the 8 uncertified spatial runs that had an incumbent**
  (clay0303hfsg 100 %, casctanks 1 084 %, nvs05 88 %, nvs09 69 %, tanksize 33 %,
  ex6_2_5 39 179 %, ex6_2_9 33 205 %). The exception is nvs19 (0.53 %).
- **The incumbent was already optimal for 52–96 % of the wall time** on those 7
  instances (`stall_frac` column) — the entire tail is pure dual-bound effort.
- **Two plumbing defects convert proved bound into reported failure:**
  1. **#598 (in-house MILP B&B):** finds `obj == bound == −26.822045` (gap exactly 0,
     HiGHS-verified optimum) in 0.8 s but exits `iteration_limit`, uncertified —
     459 node LPs, 22 833 pivots, 81 % of dual pivots degenerate. scipy-HiGHS closes
     the identical matrix (161 cols / 582 rows / 44 ints) in **one call, 0.24 s,
     19 nodes**. `LpDenseRetries = 0` → confirmed **not** the #591 dense-route class.
  2. **Decertify-and-discard (spatial path):** on nvs05 the tree *proved* a global
     lower bound of **4.8746** (89 % of the 5.4709 optimum) but the run *reported*
     **1.3481** — the `_nonrigorous_sentinel_fathom` taint sets
     `_tree_bound_valid = False` and the whole tree bound is dropped for the root
     pool fallback. Same on tanksize (proved 0.8811, reported 0.8680). The taint is
     the correct soundness response to an unproven *fathom*; discarding the valid
     *frontier bound* of the surviving tree is over-broad plumbing, not soundness.

## 1. Method

- Panel: clay0303hfsg, casctanks, tls2, nvs05, nvs09, tanksize, st_e36, nvs19,
  ex6_2_5, ex6_2_9 (`~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`,
  oracle = `minlplib.solu`), plus the #598 AMP multi4N MILP relaxation
  (`_make_alpine_multi4n(n=2, exprmode=1)` → `build_milp_relaxation`, `n_init=2`,
  solved via the default in-house route; identical matrix cross-solved with
  `scipy.optimize.milp` / HiGHS).
- `Model.solve(time_limit=60)` per instance in a fresh subprocess with
  `DISCOPT_PROFILE=1`; harness: `scripts/decomp1_cert_effort_run.py` (instance
  runner), `scripts/decomp1_cert_effort_drive.py` (driver + profile-dump parser),
  `scripts/decomp1_cert_effort_analyze.py` (attribution).
- Instrumentation (monkeypatch, measurement-only): every
  `MccormickLPRelaxer._solve_at_node_impl` call counted by terminal status
  (including C-42/C-43 retries); `_compute_alphabb_bound` / `_compute_interval_bound`
  / `solve_mccormick_batch` counted to identify each instance's actual node-bound
  engine; `_pounce_recover_node_bound` counted (MILP-path decertify recovery);
  `PyTreeManager` proxied so every `stats()` call records
  `(t, global_lower_bound, incumbent)` — the bound/incumbent trajectory.
- `t_inc*` = first time the incumbent matched the oracle (rel 1e-4);
  `stall_frac` = fraction of wall time *after* that point;
  `closed` = fraction of the root→oracle dual gap closed by the final tree bound.
- Caveats: wall numbers from a shared laptop (sequential runs, but machine
  contention possible); DISCOPT_PROFILE + the Python taps add small overhead —
  node/LP/status counts and bounds are deterministic, wall times are indicative.

## 2. Per-instance results (TL = 60 s)

| instance | oracle | status | found opt? | root_gap | final gap | nodes | node-LP solves (failures) | node-bound engine | LP pivots | stall_frac | dual gap closed | attribution |
|---|---:|---|:--:|---:|---:|---:|---:|---|---:|---:|---:|---|
| clay0303hfsg | 26669.1 | feasible | yes | **1.00** | 0.268 | 305 | 0 (0) | NLP-BB (POUNCE) | 0 | 0.67 | 0.73 | **A** |
| casctanks | 9.1635 | feasible | yes | **10.84** | 0.852 | 25 | 8 (0) | McCormick LP + interval | 28 576 | 0.52 | 0.92 | **A** |
| tls2 | 5.3 | time_limit | **no incumbent** | n/a | n/a | 1663 | 0 (0) | NLP-BB | 0 | n/a | 0.91 (bound 4.90/5.3) | **primal** |
| nvs05 | 5.4709 | feasible | yes | **0.88** | 0.754 | 533 | 2 (0)¹ | alphaBB + interval | 12 159 | **0.96** | 0.14 (reported)² | **A + B** |
| nvs09 | −43.134 | feasible | yes | **0.69** | 0.540 | 31 | 10 (0) | LP + OBBT (~1.9 s/node) | 9 930 | **0.95** | 0.22 | **A** |
| tanksize | 1.2686 | feasible | yes | **0.33** | 0.316 | 171 | 2 (0)¹ | alphaBB + interval | 55 097 | 0.84 | 0.05² | **A + B** |
| st_e36 | −246.0 | **optimal** | yes | 0.24 | 7.6e-5 | 153 | 152 (0) | McCormick LP | 46 228 | — | 1.00 | certified |
| nvs19 | −1098.4 | feasible | yes | **0.0053** | 0.0034 | 215 | 158 (0) | McCormick LP | 166 732 | 0.03³ | 0.37 | **C** |
| ex6_2_5 | −70.752 | feasible | yes | **391.8** | 118.7 | 409 | 202 (0) | McCormick LP | 900 238 | **0.95** | 0.70 | **A** |
| ex6_2_9 | −0.03407 | feasible | yes | **332.1** | 9.98 | 351 | 177 (0) | McCormick LP | 59 994 | **0.95** | 0.97⁴ | **A** |
| amp_multi4n (#598) | −26.8220⁵ | iteration_limit | yes (gap = 0!) | n/a | 0 (uncertified) | 459⁶ | 459 (≥1 numerical)⁶ | in-house MILP B&B | 22 833 | n/a | n/a | **B** |

¹ Root probes only: nvs05's LP declined a bound (`optimal`, no safe bound — sound
refusal) and tanksize's returned `unbounded`, so both fell back to alphaBB/interval
per-node bounding — the node engine that then stalls (lever A on *that* engine).
² Reported bound after decertify-and-discard; the tree actually proved 4.8746 (nvs05,
closed = 0.88) and 0.8811 (tanksize) — see §4.2.
³ nvs19's exact incumbent −1098.4 only arrived at t = 58.3 s (a −1098.2 incumbent sat
from t = 2.6 s); its bound was within 0.34 % at exit — a pure tail-closing failure.
⁴ ex6_2_9 closed 97 % of an enormous root gap (−332 → −10.01) and still ends 294×
away from the −0.0341 optimum in relative terms — the root gap is simply that large.
⁵ Optimum of the lifted MILP itself, certified by HiGHS on the identical matrix
(status 0, one `milp()` call, 0.24 s, 19 nodes, dual bound = objective).
⁶ From `DISCOPT_PROFILE` phase counters: `NodeLpSolve = 459` (1.67 s total),
`Phase1+Phase2 pivots = 22 833`, `DualWarmSolves = 761`, `DualColdFallbacks = 23`,
`DualDegeneratePivots = 18 511` (81 % of dual pivots), `RefacCap = 443`,
`LpDenseRetries = 0`. Exit after 0.85 s at `iteration_limit` with `obj == bound`.

## 3. Lever attribution — the evidence

### Lever A (relaxation/dual-bound strength) — DOMINANT, 7/10 uncertified

Seven instances have root_gap ≥ 33 % (median root_gap across the 8 uncertified
spatial runs with incumbents: **0.94**, i.e. ~94 % of the incumbent value), the
incumbent in hand within seconds (t_inc* = 2.5–29.5 s), and then 30–58 s of pure
dual-bound grinding that closes 5–97 % of the gap but never enough. Three distinct
sub-classes — the lever is *not* one relaxation:

- **alphaBB/interval-routed (nvs05, tanksize):** the McCormick LP declines at the
  root (soundly), so nodes are bounded by interval arithmetic + alphaBB. That engine
  closes 14 % (nvs05, tree bound) and 5 % (tanksize) of the gap in ~55 s over
  533/171 nodes. The per-node relaxation, not the tree policy, is the binding
  constraint.
- **x·log(x) / Gibbs family (ex6_2_5, ex6_2_9):** root gaps of **392×** and
  **332×** the unit scale. Measured *with* #597's centropy tangent underestimator
  in the base commit — the root on these two is still orders of magnitude loose.
  (ex6_2_9's bound moves −332 → −10.0, i.e. 97 % closed, and the residual is still
  ~300× the optimum — no tree policy survives that starting point.)
- **NLP-BB / root-bound-absent (clay0303hfsg; casctanks root):** clay is detected
  convex → NLP-BB; its "root bound" is ≈ 0 vs an optimum of 26 669 (root_gap 1.0),
  and 305 POUNCE NLP nodes close 73 % in 60 s. casctanks burns **29.4 s of 61 s in
  the root** (fixpoint/OBBT — consistent with T1.6's finding that the OBBT LP loop
  is the per-node lever) and starts the tree from root_gap 10.8.

### Lever B (LP numerics / certification robustness) — REAL BUT NARROW; fix first

The panel-wide form hypothesized in the directive — node LPs taking numerical exits
— **does not occur on the spatial path**: 0 failures in 711 node-LP solves, and
`LpDenseRetries = 0` everywhere. B is instead two specific plumbing defects:

1. **#598, in-house MILP B&B:** gap **exactly 0** (obj = bound = HiGHS-verified
   optimum) yet `iteration_limit`, because one lifted-McCormick node LP took an
   uncertified exit and `decide_status` demands `gap_certified`. Cost of the miss:
   an entire found-and-proved optimum reported as a limit exit. The engine also
   burns 459 nodes / 22 833 pivots where HiGHS needs 19 nodes in one call — the
   node-count ratio is ~24× *and* the exit label is wrong. (The pivot profile —
   81 % degenerate dual pivots, RefacCap = 443 refactor-cap trips over 761 warm
   solves — localizes the inefficiency to degeneracy handling + FT-update churn on
   partitioned/lifted formulations.)
2. **Decertify-and-discard on the spatial path:** `_nonrigorous_sentinel_fathom`
   taint (a node NLP failed and was sentinel-pruned without proof) correctly blocks
   an "optimal" label, but the result-build then treats the *entire tree bound* as
   tainted (`_tree_bound_valid = False`) and reports the far weaker root-pool /
   root-relaxation fallback: nvs05 reports 1.3481 where the tree proved 4.8746
   (**3.6× weaker**, reported gap 75 % vs provable 11 %); tanksize reports 0.8680
   vs proved 0.8811. A node pruned without proof taints *optimality*, and any
   *ancestral* bound contribution of that node — but the frontier minimum over
   surviving open nodes is a function of which nodes were (possibly wrongly)
   *removed*; the sound fix is per-node (floor tainted nodes into the frontier at
   their inherited parent bound) rather than discarding the frontier wholesale.
   That is a certification-accounting fix, not a relaxation fix.

### Lever C (B&B machinery with a tight relaxation) — 1/10

Only **nvs19**: root_gap 0.53 % (consistent with #587 — the root is already
SCIP-with-cuts-tight on the integer-product family, so A-via-cuts is dead there),
final gap 0.34 %, 215 nodes at ~3.8 nodes/s with zero LP failures. The tree grinds
the last half-percent and cannot finish. Note its *exact* incumbent only arrived at
t = 58 s (a 0.02 %-off incumbent sat for 55 s) — tail-gap closing needs both the
last bound sliver and the last incumbent sliver.

### Unclassified by A/B/C: primal failure (tls2)

tls2's *dual* side works: bound 0.718 → 4.90 vs optimum 5.3 (91 % closed, 1 663
nodes). It never finds **any** incumbent in 60 s, so there is nothing to certify.
The binding lever is primal heuristics on this structure, and no relaxation or
certification work changes that.

## 4. Counterfactual: "if the root bound were exact, how much would fathom?"

Using the incumbent trajectory: with an oracle-exact root bound, the solve ends at
t_inc\* (incumbent = optimum, bound = optimum → gap 0). Panel-wide that recovers
**52–96 % of the wall time on 7 of the 9 uncertified runs with incumbents**
(clay 67 %, casctanks 52 %, nvs05 96 %, nvs09 95 %, tanksize 84 %, ex6_2_5 95 %,
ex6_2_9 95 %; nvs19 3 % — its bound is already near-exact; amp 100 % — its bound
IS exact and only the label is withheld). The nvs05/nvs09/tanksize "finds the
optimum instantly, stalls 60 s" pattern from the directive is confirmed and is a
pure dual-bound stall, not incumbent search.

## 5. Recommendation

1. **Fix the two Lever-B plumbing defects first** (small, bounded, converts already-
   proved results into certificates / honest gaps):
   a. #598: harden the in-house node-LP so a zero-gap search can certify (iterative
      refinement / anti-degeneracy on partitioned formulations — the profile points
      at degenerate dual pivots + refactor-cap churn), or at minimum re-verify the
      single poisoned node instead of withholding the label for the whole solve.
   b. Replace decertify-and-**discard** with per-node taint accounting so the
      reported bound is the tree's valid frontier bound (nvs05: 4.87 not 1.35).
      This alone moves reported gaps on the taint-affected class from ~75 % to ~11 %
      without touching any relaxation.
2. **Then attack Lever A as the strategic thrust — but per sub-class, not
   generically:** (i) the alphaBB/interval fallback engine that pure-integer /
   LP-declined models route to (nvs05, tanksize) — these nodes close ≤ 14 % of the
   gap in a minute; (ii) the x·log(x)/Gibbs root (ex6_2_5/9, root gaps 300–400×)
   where #597's tangent planes are demonstrably not yet enough; (iii) the NLP-BB
   root bound (clay: root_gap 1.0 — NLP-BB opens the tree with essentially no dual
   bound). Do **not** spend A-effort on integer-product roots (#587: already
   SCIP-tight; c-MIR NO-GO stands).
3. **Lever C is not the panel's problem** (1/10). Revisit after A/B.
4. Track the tls2-style primal gap separately — it is invisible to all three levers.

## 6. Falsifications recorded

- "Node-LP numerical exits poison certification panel-wide" — **falsified** on the
  spatial path (0 failures / 711 node LPs, `LpDenseRetries = 0`). The failure class
  is confined to the in-house MILP B&B on lifted/partitioned MILPs (#598) and to
  node-*NLP* sentinel fathoms whose taint handling over-discards (§3, Lever B).
- "The tree's reported bound reflects what the search proved" — **false** on the
  taint-affected class (nvs05, tanksize): reporting discards up to 3.6× of proved
  bound.
- Binding negatives respected, not relitigated: per-node Python tax (T1.6 — the
  OBBT LP loop is the per-node lever; casctanks' 29 s root is consistent),
  reduced-space (no tightening), c-MIR on integer-product families (#587 NO-GO).

## 7. Reproduction

```bash
DISCOPT_PROFILE=1 python scripts/decomp1_cert_effort_drive.py 60          # full panel
python scripts/decomp1_cert_effort_drive.py 60 nvs05,amp_multi4n          # subset
python scripts/decomp1_cert_effort_analyze.py <results-dir>               # attribution rows
```

Raw per-instance JSON (result fields, LP status counts, Rust profile counters,
bound/incumbent trajectories) is written as `res_<name>.json`; this report's table
is generated from those files by the analyze script. Runs for this report were on
macOS arm64, sequential, 2026-07-10.
