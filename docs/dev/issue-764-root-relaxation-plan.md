# #764 — Closing `tanksize`: from root-relaxation research to the closure plan

Status: **direction resolved 2026-07-18 — see "The closure plan" at the end.** The #764
relaxer-discard fix (`DISCOPT_ROOT_LP_PROBE_TIGHT`, graduated default-ON) shipped and lifted
`tanksize`'s frontier 0.853→0.92 and certified `syn05hfsg`. The root-relaxation research direction
this doc originally scoped was then **run to ground and falsified** (level-1 RLT, level-2 RLT,
Shor SDP, OBBT-to-fixpoint, naive partitioning — all exact no-ops at the root; §candidates below).
The evidence now identifies the closure path as **branch-and-reduce throughput** — the same
diagnosis as `certification-gap-plan.md` C1/C2 — with a measured first lever: the already-built
T2.4 `reduce_node` (`DISCOPT_NODE_REDUCE`, default-OFF) moves BOTH axes on `tanksize`
(bound 0.8898→0.9221 @60 s AND +52 % node throughput).

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
| best current config | probe-tight (default) + `DISCOPT_NODE_REDUCE=1`: 0.9221 @143 nodes/60 s, 2.38 n/s |
| per-node cost (~860 ms) | ~32 % Rust LP solves (**~95 LPs/node** ≈ 2n OBBT probes @~3 ms each), ~10 % pounce NLP, ~35 % Python/JAX orchestration (asarray ~11 k, Python DAG-walk ~58 k, interval ~7 k calls/node) |

### What BARON/SCIP actually do (maps to `certification-gap-plan.md` §2 gap table)

Neither closes `tanksize` at the root — BARON's root gap is 25 %. They win by making nodes
~1000× cheaper and reducing *at* every node:

- **BARON (branch-and-reduce)**: per node = bound-patch + warm dual-simplex LP (µs–ms), then
  **marginals-based range reduction for free** (reduced costs + incumbent cutoff ⇒ tightened
  bounds, no extra LPs), probing at the root, *full* OBBT rarely. Native code throughout.
- **SCIP**: LP warm start + domain propagation at every node in C, reliability branching, cuts
  (measured **inert** for discopt's relaxation of this family — CUT-1/CUTS-1 NO-GO).

discopt has the *components* (warm incremental LP engine T1.3, Rust `fbbt_with_cutoff`, free DBBT
marginals T2.4a, OBBT probes) but spends ~95 LPs and ~40 ms+ of Python per node where BARON spends
1–3 LPs and zero interpreter time.

### Verdict on this doc's original direction

Root-relaxation research is **closed as falsified** here: candidates 1 (PQ/level-1 RLT) and
2 (level-2 RLT) killed by entry experiments; candidate 3 (structured SDP) not run — low odds
(Shor inert), high cost, and now unnecessary: the search closes the gap when given nodes.
T2.3 root fixpoint (`DISCOPT_ROOT_FIXPOINT=1`): **bit-identical on tanksize** (bound 0.8898,
95 nodes, unchanged) — binding negative for this class.

### Phase A — wiring + scheduling (days; target ≥ 20–50 n/s, certify in minutes)

1. **Graduate T2.4 `reduce_node`** (`DISCOPT_NODE_REDUCE`) through the CLAUDE.md Regime-2 panel.
   Measured on tanksize: bound 0.8898→0.9221 @60 s AND 1.57→2.38 n/s — the only lever this
   campaign that moved both axes. It is the BARON free-reduction step, already built and
   property-tested (200-box); it needs the ON-vs-OFF corpus panel (cert-clean + net-positive).
2. **Per-node LP diet — replace the 2n OBBT probes with scheduled OBBT + free DBBT.** ~95 LPs/node
   is the single biggest line item. With T2.4's free DBBT as the every-node reduction, full OBBT
   probing can be scheduled (depth-gated / strided / marginal-filtered: probe only variables whose
   `width × |reduced cost|` suggests payoff — T2.5's scorer exists, parked). Entry experiment: the
   LPs/node vs bound-trajectory tradeoff curve on tanksize + nvs05 (nvs05 is the class where probes
   are load-bearing — #738; the diet must not regress it). Kill: no schedule beats always-probe on
   frontier-bound-at-equal-wall.
3. **NLP starvation on stalled incumbents.** ~10 % of wall is pounce NLPs that cannot improve an
   already-optimal incumbent. `DISCOPT_ADAPTIVE_NLP` (default-ON) should starve them; verify it
   engages on tanksize and fix the gap if not (bound-neutral).
4. **Python orchestration diet** (~35 % of wall): cache the static expression-DAG walk in
   `nonlinear_bound_tightening` (~58 k calls/node re-deriving a static structure), batch the
   `asarray` marshaling (~11 k/node). Bound-neutral Regime-1 changes (exact node-count/objective
   equality gates).

Arithmetic: ~10 ms/node ⇒ ~100 n/s ⇒ a BARON-shaped 3.5–14 k-node tree closes in **35–140 s**.
Phase A alone plausibly certifies tanksize at TL = 300 s and repairs the whole slow-spatial class;
it does NOT reach "seconds".

5. **(Parallel, cheap) Branching-quality A/B**: the full B&B climbs while naive widest-var
   bisection is inert — so selection matters. A/B current spatial selection vs pseudocost/
   reliability on bilinear-violation (BARON's violation transfer). Node-count multiplier if it
   lands; entry experiment offline on captured trees.

### Phase B — the Rust node kernel (weeks; target ~1 ms/node, certify in seconds)

The `certification-gap-plan.md` §2 "per-node language cost — **architecture**" gap: move the
spatial node inner loop (bound patch → warm dual simplex → cutoff-FBBT/DBBT reduce → branch
select) entirely into `discopt-core`, calling back to Python only for NLP heuristics and
separation events. This is the only route to BARON-order µs–ms nodes and the issue's literal
"certifies in seconds" DoD. Scope it as its own certification-gap-plan phase with the usual
entry experiment (a Rust prototype loop on the already-Rust LP/FBBT components, measured on the
global50 spatial subset) before committing.

### Gates (unchanged, binding)

Phase A items 3–4 are Regime-1 (bound-neutral: exact node-count + objective equality). Items 1–2
and Phase B are Regime-2 (flagged default-OFF → ON-vs-OFF corpus panel, cert-clean AND
net-positive, `incorrect_count = 0`, no certification regressions). No instance-keyed logic
anywhere; `tanksize`/`nvs05` are gate probes only.
