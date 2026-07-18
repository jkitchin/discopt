# #764 — Closing `tanksize`: from root-relaxation research to the closure plan

Status: **RESOLVED 2026-07-18 — `tanksize` is a per-node-*throughput* problem, full stop; the
relaxation is a red herring and the "infeasible" framing below was overstated (see the STRATEGY
CORRECTION).** The #764 relaxer-discard fix (`DISCOPT_ROOT_LP_PROBE_TIGHT`, graduated default-ON)
shipped and lifted `tanksize`'s frontier 0.853→0.92 and certified `syn05hfsg`.

> **STRATEGY CORRECTION (2026-07-18) — the decisive measurement.** discopt reaches bound 1.138 at
> **363 nodes / 150 s** and is climbing ~0.11 per 60 nodes, so it closes `tanksize` in **~450 nodes**
> — *fewer* than **BARON's 3477**. discopt's search is MORE node-efficient. The entire deficit is
> per-node speed: ~450 × ~420 ms ≈ 190 s vs BARON's 3477 × ~0.8 ms ≈ 2.7 s (~**500× per node**). At
> BARON's node speed, discopt's ~450 nodes would close in **under a second**. So `tanksize` is not
> special in its math — it is simply the rare instance that needs a **several-hundred-node tree**;
> the problems discopt solves well close in **< ~50 nodes**, where the 500× per-node penalty is
> invisible. This **invalidates two claims made earlier in this doc**: (1) the Phase-B "~2.3× ceiling
> / KILL on tanksize" — that ceiling assumed discopt keeps doing **95 OBBT LPs/node**, which is a
> *strategy choice*, not a floor; BARON closes with **~1 LP/node + free reduced-cost reduction** and
> more nodes (discopt does ~450×95 ≈ 43 k LP solves total vs BARON's few thousand). (2) "both axes
> exhausted / infeasible." The untested lever is exactly the **BARON per-node strategy** — cheap
> reduced-cost DBBT (≈1 LP/node) in place of brute-force OBBT (95 LPs/node), taking more but far
> cheaper nodes. discopt *had* this (T2.4 `reduce_node`, removed #581 as net-negative on
> ex5_3_3/spring/qapw) but it was **never evaluated on the loose-root / big-tree class** where the
> cheap-node-more-nodes tradeoff should win. This is the real next entry experiment; see "The closure
> plan". The relaxation findings below stand as recorded, but they are **not** the blocker.
>
> **REFINEMENT (2026-07-18) — "cheap nodes" ≠ "no reduction".** Turning per-node OBBT fully OFF
> (`_PER_NODE_OBBT_MAX_VARS=0`, ~1 LP/node) does NOT help: the frontier bound **stalls at ~0.93**
> (851 nodes @150 s, never climbs) vs full OBBT's 1.138 @363. The per-node **box tightening is
> load-bearing** — without it the McCormick relaxation is too loose for branching alone to progress.
> So the lever is not *less* reduction, it is *cheaper-per-unit* reduction: SCIP/BARON get the same
> box tightening from **reduced-cost propagation (free from the one node LP's duals) + selective OBBT
> on a few variables at select nodes, warm-started in native code**, where discopt does **exhaustive
> OBBT (all ~47 vars × 2 bounds = ~95 cold-ish LPs) at every branched node in Python/JAX**. Same
> technique (OBBT/DBBT), done ~50× more often than needed and ~100× slower per LP. The
> `certification-gap-plan.md` names both halves: **Phase 2 branch-and-reduce orchestration** (make
> reduction selective/cheap — reduced-cost DBBT as the default, OBBT only where it pays) and
> **Phase B native node kernel** (warm dual-simplex LPs, no marshaling). Neither is a research
> unknown; both are engineering the substrate SCIP/BARON already have.

> **CORRECTION (2026-07-18, same day).** An earlier revision of this doc claimed the T2.4
> `reduce_node` flag (`DISCOPT_NODE_REDUCE`) as "a measured first lever (+bound, +52 % throughput
> on tanksize)". That claim is **retracted as a measurement artifact**: the flag was
> **removed in #581** (its held-out N=20 graduation gate in PR #685 came back net-negative —
> benefit 24 % / regression 18 %, regressing ex5_3_3/spring/qapw — and the module was deleted;
> nothing reads the env var). The apparent A/B difference (0.8898/95 nodes vs 0.9221/143 nodes
> @60 s) was **machine load**: the "OFF" arms ran under the resource pressure that shortly killed
> the session worker. Re-measured under clean conditions, default and `DISCOPT_NODE_REDUCE=1` are
> indistinguishable (0.9221 @ 139–147 nodes across 4 runs). There is no T2.4 flag to graduate; a
> reduce_node-style mechanism would be a *rebuild* that must first overturn the #685 net-negative
> verdict with new class-level evidence. `DISCOPT_ROOT_FIXPOINT=1` (T2.3, which *does* still
> exist) was re-verified clean: 0.9221 @143 — inert on tanksize, unchanged conclusion.

## Goal / definition of done

`tanksize` (MINLPLib, in-repo `python/tests/data/minlplib_nl/tanksize.nl`, oracle 1.2686437540,
minimize) certifies at default tolerances in seconds (BARON 2.66 s / SCIP 0.75 s order of
magnitude), `incorrect_count = 0` throughout. The incumbent is already the optimum and is found at
**node 0**, so the entire task is closing the dual bound.

## The lever is real (measured)

BARON's root dual bound is **0.955**; it is a valid lower bound, so the continuous-relaxation global
optimum C satisfies `0.955 ≤ C ≤ 1.2686` (the MINLP incumbent, which is continuous-feasible). discopt's
root LP is **0.838** — it leaves **≥ 0.117** on the table at the root. A stronger relaxation
demonstrably exists.

## What is falsified — every existing discopt mechanism is inert at the root

Measured on the root McCormick LP (`MccormickLPRelaxer.solve_at_node` over the FBBT box, integers
relaxed). These negatives are **binding** (CLAUDE.md §4) — do not re-tread them:

| mechanism | root LP | note |
|---|---|---|
| McCormick (baseline) | **0.8382** | — |
| build-time level-1 RLT (`rlt_level1=True`) | 0.8382 | exactly no-op |
| targeted RLT cut separation (`rlt_cuts=True`, 6 rounds) | 0.8382 | exactly no-op — links to existing product cols, still nothing violated/tightening |
| `obbt_tighten_root`, 30 rounds (121 bounds tightened) | 0.8382 | box tightening cannot move it; x0/x6/x7 stay wide |
| Shor SDP (`DISCOPT_SHOR_SDP_ROOT_BOUND=1`) | 0.840 | inert |
| piecewise/bisection on widest bilinear var, up to 8 leaves | 0.8382 | looseness is **distributed**, not concentrated — naive spatial partitioning does not tighten |

RLT being a bit-identical no-op on a *real* instance is the #727 lesson confirmed: do not invest in
an RLT variant here without first measuring root-gain > 0 on `tanksize` itself.

## Honest coupling with throughput (read before committing to relaxation-only)

BARON closes `tanksize` in **3477 nodes / 2.66 s ≈ 1300 nodes/s** — its root (0.955) is only modestly
better than discopt's (0.838) and still 25 % off the optimum, so **BARON's win is primarily
throughput, not a tight root.** Consequences:

- A stronger root relaxation *reduces* the node count needed but, at discopt's ~2–10 nodes/s (the
  #723 residual, closed as largely genuine per-node simplex work), is **necessary-but-not-sufficient**
  to reach "seconds" on its own. Reaching BARON's node count at BARON's speed needs the engine-level
  throughput lever #723 scoped out.
- The two levers *multiply*: root 0.838→~0.95 might cut the tree by a large factor, and even a modest
  node-rate gain then compounds. The relaxation lever is the tractable-first half **because** it also
  helps every other spatial instance and does not depend on the Rust-simplex engine rewrite.
- Do not promise `tanksize`-in-seconds from the relaxation lever alone; scope it as "materially
  fewer nodes + a tighter certificate," with the throughput multiplier tracked separately.

## Structural diagnosis

Dense, bipartite, pooling-like bilinear program: 47 products `w_ij = x_i·x_j` with heavily shared
variables — the flow group {x0,x1,x2, x6..x14} multiplies the split group {x18..x26}, plus an
objective chain through x15·x16, x16·x17 and `x15 = 0.3271·(√x3+√x4+√x5)`. FBBT bounds the flows
finitely (x0∈[1,40], x6∈[0,992], x7∈[0,1680]) but they stay *wide*; the single-term McCormick hull
is exact per product yet the **joint** relationship among products sharing a variable is unmodeled,
and OBBT self-limits (loose relaxation → weak OBBT → loose relaxation).

## Candidate mechanisms, ranked — each needs a `tanksize`-real entry experiment first

1. **Pooling PQ-relaxation — FALSIFIED (entry experiment, 2026-07-18).** PQ is a special case of
   level-1 RLT (multiply linear rows by variable/bound factors, lift the products). Tested the
   **full level-1 RLT closure**, not just PQ: added `(-g)·(x_j-l_j) ≥ 0` and `(-g)·(u_j-x_j) ≥ 0`
   for every linear form `g ≤ 0` × every continuous variable (2190 valid product constraints, 2170
   lifted product columns, solved sparse via `DISCOPT_SPARSE_LARGE_LP=1`). Result: root LP
   **0.8382 → 0.8382, delta = 0.0000** (bit-identical). Also tested the RLT-over-existing-products
   subset (110 constraints): delta 0.0000. Construction validated — injecting an over-tightening
   `x17 ≥ 1.0` correctly moves the root to 1.0, so the RLT rows genuinely reach the LP; they are
   simply all non-binding at the McCormick optimum. **tanksize's McCormick bound already equals its
   level-1 RLT bound.** Additional decisive sub-finding: with all 9 integers *fixed to their optimal
   leaf* the root is still 0.8382, so the entire 0.838→1.2686 gap is **continuous×continuous
   bilinear** (x0·x6/x9/x12, x1·x7/x10/x13, x2·x8/x11/x14, x15·x16, x16·x17), not integrality.
   Repro: `scratchpad` `pq_exp.py` / `pq_exp2.py`. Kill criterion met — do not pursue PQ or any
   level-1 RLT variant here.
2. **Level-2 RLT / higher-degree products — FALSIFIED (entry experiment, 2026-07-18).** Added
   level-2 RLT rows `(-g)·(x_i-l_i)·(x_k-l_k) ≥ 0` for every linear form `g ≤ 0` × each loose product
   pair (i,k), **expanded into explicit clean monomials** so discopt's classifier actually lifts them
   (naive `(-g)*fi*fk` is dropped — `tri=0`; the expanded form yields `bil 47→293, tri 0→213`, i.e.
   213 trilinear terms genuinely relaxed via trilinear McCormick). 121 constraints, solved sparse.
   Result: root LP **0.8382 → 0.8382, delta = 0.0000**. Level-2 RLT is inert too. (Caveat: this uses
   discopt's *trilinear-McCormick* relaxation of the degree-3 terms — a tighter trilinear hull could
   in principle bind, but that is a second, more speculative layer; the standard hierarchy does
   nothing here.) Repro: `scratchpad/rlt2b.py`. Kill criterion met.
3. **Structured/higher-order SDP on the shared-variable blocks.** Diagonal Shor was inert (0.840);
   a moment/SDP relaxation on each shared-variable star (e.g. {x0, x6, x9, x12} and their products)
   might capture the joint constraint level-1 RLT misses. Entry experiment: solve the block-moment
   SDP offline (scs) over one leaf and measure.

**Reality check — the RLT hierarchy is exhausted.** Everything tractable is now measured inert at
the root, bit-identically:

    McCormick 0.838 = level-1 RLT 0.838 = level-2 RLT 0.838 (213 trilinears) ;
    Shor SDP 0.840 ; 30-round OBBT 0.838 ; integers-fixed 0.838 ; naive bisection 0.838 (→8 leaves)

Yet the continuous optimum is ≥ 0.955. That combination is the important signal: the gain is **not**
in the polyhedral RLT hierarchy discopt can build, so candidate 3 (structured/higher-order SDP) is
now the only untested relaxation route and is a long shot given Shor's inertness. The realistic path
to the ≥0.955 bound is most likely **smart spatial branching** — the full B&B *does* climb
(0.906→1.079 over 199 nodes) because it branches on the right variables, unlike the inert
naive-widest bisection — i.e. this is a **branching-quality + throughput** problem more than a
root-relaxation one. Reinforced #727 lesson: level-1 AND level-2 RLT are exact no-ops on the real
instance; do not ship an RLT-family relaxation for this class.

Branching-quality is a *secondary* finding: naive widest-variable bisection is inert (0.838 to 8
leaves) while real B&B climbs, so the effective spatial-branching variable selection here is worth a
separate look.

## Soundness / graduation contract (non-negotiable)

Any new envelope/cut is **bound-changing** (CLAUDE.md §5, Regime 2): behind a default-OFF flag until
the corpus differential panel is BOTH cert-clean (no bound above its reference optimum, no
certificate-invariant violation, incumbents independently feasibility-verified) AND net-positive.
A bilinear envelope that ever over-tightens is a false optimal — the single worst failure in this
codebase. No single-instance / problem-name special-casing (§2).

## Where the sound increment already landed

`python/discopt/solver.py` `_root_lp_probe_tight_enabled()` (graduated default-ON, #764);
regression tests in `python/tests/test_issue282_root_lp_probe.py`; panel artifact
`discopt_benchmarks/results/issue764_root_lp_probe_tight_graduation_panel.json`; falsification record
in `docs/dev/issue-282-syn-rsyn-diagnosis-2026-07-17.md` §R3-5.

---

## The closure plan (2026-07-18) — branch-and-reduce throughput, not relaxation

### Where we are (all measured this campaign)

| fact | value |
|---|---|
| incumbent | = optimum (1.2686), found at **node 0** — pure dual-closure problem |
| root LP | **0.838, immovable**: = level-1 RLT = level-2 RLT (213 trilinears lifted); Shor SDP 0.840; 30-round OBBT, integers-fixed, naive bisection all 0.838 |
| BARON | root 0.955 (also a 25 % root gap!) → closes at **3477 nodes / 2.66 s ≈ 1300 n/s** |
| SCIP | closes in 0.75 s |
| discopt search | **does climb**: 0.92 @71 → 1.079 @199 → 1.145 @677 nodes — search quality is plausibly within small factors of BARON's; the deficit is **throughput: 2–10 n/s (100–600×)** |
| current default (clean re-measure ×4) | 0.9221 @ 139–147 nodes/60 s, ~2.4 n/s (probe-tight graduated in) |
| per-node cost (~860 ms) | ~32 % Rust LP solves (**~95 LPs/node** ≈ 2n OBBT probes @~3 ms each), ~10 % pounce NLP, ~35 % Python/JAX orchestration (asarray ~11 k, Python DAG-walk ~58 k, interval ~7 k calls/node) |

### What BARON/SCIP actually do (maps to `certification-gap-plan.md` §2 gap table)

Neither closes `tanksize` at the root — BARON's root gap is 25 %. They win by making nodes
~1000× cheaper and reducing *at* every node:

- **BARON (branch-and-reduce)**: per node = bound-patch + warm dual-simplex LP (µs–ms), then
  **marginals-based range reduction for free** (reduced costs + incumbent cutoff ⇒ tightened
  bounds, no extra LPs), probing at the root, *full* OBBT rarely. Native code throughout.
- **SCIP**: LP warm start + domain propagation at every node in C, reliability branching, cuts
  (measured **inert** for discopt's relaxation of this family — CUT-1/CUTS-1 NO-GO).

discopt has the *components* (warm incremental LP engine T1.3, Rust `fbbt_with_cutoff`,
`MccormickLPResult` marginals from T2.4a — the result-surface half of T2.4, which survived #581,
OBBT probes) but spends ~95 LPs and ~40 ms+ of Python per node where BARON spends 1–3 LPs and zero
interpreter time.

### Verdict on this doc's original direction

Root-relaxation research is **closed as falsified** here: candidates 1 (PQ/level-1 RLT) and
2 (level-2 RLT) killed by entry experiments; candidate 3 (structured SDP) not run — low odds
(Shor inert), high cost, and now unnecessary: the search closes the gap when given nodes.
T2.3 root fixpoint (`DISCOPT_ROOT_FIXPOINT=1`): **inert on tanksize** (0.9221 @143 = default,
re-verified under clean conditions) — binding negative for this class. T2.4 `reduce_node`: **no
longer exists** (removed #581 after the #685 held-out gate measured it net-negative); see the
CORRECTION note in the header — it is NOT a Phase A item.

### Phase A — scheduling + orchestration diet (days; target ≥ 20–50 n/s, certify in minutes)

1. **Per-node LP diet — schedule the 2n OBBT probes — FALSIFIED (entry experiment, 2026-07-18).**
   ~95 LPs/node (~32 % of wall) is the biggest line item, so the hypothesis was that thinning the
   probes (stride / top-k columns / fewer rounds) buys net throughput. Measured the bound trajectory
   (`node_callback`) for default vs `stride=2` vs `top_k=20`. **Bound-per-node (deterministic,
   relaxation quality): full OBBT wins** — @160 nodes default 0.8957 vs stride2 0.8793 vs topk20
   0.8858; thinning strictly loosens the box. Throughput rises (~72 s→~50 s to reach 160 nodes) but
   the **net bound-at-equal-wall is a wash** — all three converge to ~1.00 by 90 s and no config
   robustly leads (an apparent stride2 lead at 60 s was a single fathoming-jump landing just before
   the sample; it did not reproduce). This reproduces #738 (probes load-bearing) and #723 lever-3
   (probes productive, no free thinning) *on tanksize specifically*. Kill criterion met — do not
   schedule/thin per-node OBBT for this class. Repro: `scratchpad/a1_traj.py`.
   *(Corollary: even the best config reaches only ~1.00 @90 s vs the 1.2686 optimum — probe
   scheduling cannot close tanksize regardless; the bound climb past ~1.0 is the long tail.)*
2. **NLP starvation on stalled incumbents — SMALL / already near floor (measured).** pounce NLPs
   are only **~10 % of wall, 2/node**, and `DISCOPT_ADAPTIVE_NLP` on-vs-off is a **no-op on
   tanksize** (identical 55 NLPs, 27 nodes) — it is already at its floor and part of the 2/node is
   the node-bound path, not pure waste. Ceiling ~10 %; not a closure lever. Repro:
   `scratchpad/a2.py`.
3. **Python orchestration diet — DIFFUSE, no cacheable hotspot (measured).** The ~14–35 % Python/JAX
   cost is spread across thousands of tiny calls: `tighten_nonlinear_bounds` is ~34 ms/node
   fragmented over `_constant_value` (220 k calls), `match` (87 k), `walk` (63 k) — already partially
   cached (`_get_struct_cache`/`_cached_flat_terms`) — plus `asarray` marshaling (13.6 k/node) at the
   Python↔Rust↔JAX boundary. There is no single hotspot to cache; this is death-by-a-thousand-cuts,
   exactly the residual #723 closed as needing the native kernel, not micro-opts.

**Phase A verdict — NOT a viable path to the needed throughput (2026-07-18).** All three levers are
now measured and none delivers the ~10× the plan's arithmetic assumed: A1 (the 32 % OBBT line item)
is falsified — probes are load-bearing, thinning is a wash; A2 (10 %) is at its floor; A3 (~14–35 %)
is diffuse and un-cacheable in Python. The tanksize per-node cost is genuinely ~⅓ productive-required
OBBT LPs + ~⅓ diffuse Python orchestration + ~⅓ Rust LP/NLP — irreducible without moving the node
loop out of Python. **Skip Phase A micro-optimization; the necessary investment is Phase B.** (This
independently reconfirms the `certification-gap-plan.md` C1 "per-node language cost — architecture"
diagnosis and the #723 closed verdict.)

4. **(Parallel, cheap) Branching-quality A/B**: the full B&B climbs while naive widest-var
   bisection is inert — so selection matters. A/B current spatial selection vs pseudocost/
   reliability on bilinear-violation (BARON's violation transfer). Node-count multiplier if it
   lands; entry experiment offline on captured trees.

### Phase B — the Rust node kernel — ENTRY EXPERIMENT RUN (2026-07-18): regime-split verdict

The entry experiment measures the **ceiling** speedup a Rust node loop can give = `total_wall /
native_compute`, where native = the work a Rust loop cannot remove (Rust LP solves + pounce NLP;
everything else — Python interpreter, JAX Python-side dispatch, numpy marshaling, Python interval
arithmetic / McCormick build — is removable). Kill if ceiling < ~5×. Measured under cProfile:

| instance | native (stays) | removable | ceiling | regime |
|---|---|---|---|---|
| **tanksize** | 42.8 % | 57.2 % | **2.3×** | OBBT-LP-bound (95 native Rust LP probes/node) |
| **nvs05** | 50.7 % | 49.3 % | **2.0×** | OBBT-LP-bound |
| **casctanks** | 7.0 % | 93.0 % | **14.2×** | cold-McCormick-rebuild-bound (`interval.__post_init__` 489 k calls, `_build_convexity_box`) |

**Two regimes, opposite verdicts:**
- **Cold-rebuild-bound class (casctanks-like — the `certification-gap-plan.md` C1 median, "lifted
  McCormick LP cold-rebuilt from the DAG every node ~ half the wall"): GO.** Ceiling ~10–14×; the
  Python McCormick/convexity-box rebuild dominates and a Rust incremental kernel removes it. Worth
  the weeks — but it belongs to `certification-gap-plan` Phase 1/5, not #764.
- **OBBT-LP-bound class (tanksize, nvs05): KILL for the ≥5× target.** Ceiling ~2.0–2.3× — ~half the
  wall is *native Rust LP solves* (the productive OBBT probes), which #723 already showed do not
  warm-start past 1.36×. Even a perfect Rust loop takes tanksize ~2 n/s → ~4.6 n/s: **thousands of
  nodes to close ⇒ minutes, still not seconds.**

**Consequence for #764's literal DoD.** tanksize sits in the KILL regime on *both* axes: its root
relaxation is exhausted (all inert at 0.838) AND its per-node cost is dominated by genuine,
irreducible OBBT simplex work (Phase B ceiling 2.3×). **Certifying tanksize "in seconds" is not
reachable by any lever discopt can implement** without a fundamentally different node-bounding scheme
that needs far fewer than 2n LP probes per node (the open, unowned research lever #723 named — a
*different bounding scheme* for the functionally-dependent class, not a faster implementation of the
same one). What is reachable: Phase B certifies tanksize in **~minutes** (2.3× × node-count help),
and delivers the big win on the cold-rebuild-bound majority of the slow-spatial class.

### Gates (unchanged, binding)

Phase A items 3–4 are Regime-1 (bound-neutral: exact node-count + objective equality). Items 1–2
and Phase B are Regime-2 (flagged default-OFF → ON-vs-OFF corpus panel, cert-clean AND
net-positive, `incorrect_count = 0`, no certification regressions). No instance-keyed logic
anywhere; `tanksize`/`nvs05` are gate probes only.

---

## Recommendation to the maintainer (2026-07-18)

The corrected diagnosis: **`tanksize` is a pure per-node-throughput problem.** It needs ~450 nodes
(fewer than BARON's 3477); each node is ~500× slower than BARON's. Not a relaxation problem.

1. **Ship** (done): the relaxer-discard fix, `DISCOPT_ROOT_LP_PROBE_TIGHT` default-ON — a sound,
   panel-graduated win (tanksize frontier 0.853→0.92, `syn05hfsg` newly certified, byte-stable
   elsewhere). This is the only code change #764 warranted and it is landed.
2. **Falsified, recorded as binding negatives** (do not re-tread): the whole root-relaxation
   hierarchy (level-1/2 RLT, Shor SDP, OBBT, partitioning — root immovable at 0.838, and it does not
   matter because the tree size is already fine). NLP starvation at floor. Python orchestration
   diffuse. These are real and recorded, but none is the blocker.
3. **The untested lever — the real next step (do NOT close #764 as infeasible):** the **BARON
   per-node strategy**. discopt spends ~95 OBBT LPs/node to buy a small (~450-node) tree; BARON
   spends ~1 LP/node + free reduced-cost reduction and takes ~3477 cheaper nodes, closing in 2.7 s.
   discopt's total LP work is ~5–10× BARON's *and* each LP is slower. Entry experiment: on the
   loose-root/big-tree class (tanksize + siblings), run cheap reduced-cost DBBT (≈1 LP/node) in place
   of full per-node OBBT and measure **total wall to close** (not bound-per-node — the point is the
   tradeoff favors many cheap nodes here). discopt *had* this (T2.4 `reduce_node`, removed #581 for
   net-negative on ex5_3_3/spring/qapw — a *different* class); the #685 gate never covered this class.
   Kill: total-wall-to-close not improved vs full-OBBT default. This is the honest correction to the
   earlier "Phase-B ceiling 2.3× / infeasible" — that ceiling assumed the 95-LP strategy is fixed,
   which it is not.
4. **Also worth doing** (own issue, `certification-gap-plan` scope): the **Phase B Rust node kernel**
   — but note the two levers compose: the strategy switch (item 3) cuts LPs/node ~50×, and the Rust
   kernel makes each remaining node native. On the cold-rebuild-bound class (casctanks) the kernel
   alone is ~10–14×; on tanksize the *combination* (fewer LPs × native loop) is the plausible route
   to BARON-order wall, not the kernel alone.

The through-line: measurement killed four plausible directions before any unsound code shipped — but
it also over-reached into an "infeasible" verdict that a single node-count measurement overturned.
The §4 discipline cuts both ways: the correcting experiment (nodes-to-close) should have run before
the "infeasible" framing, not after.
