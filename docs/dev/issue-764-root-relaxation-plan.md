# #764 — Stronger root relaxation for the dense-bilinear / pooling class

Status: **research direction, foundation laid 2026-07-18.** The #764 relaxer-discard fix
(`DISCOPT_ROOT_LP_PROBE_TIGHT`, graduated default-ON) shipped and lifted `tanksize`'s frontier
0.853→0.92 and certified `syn05hfsg`. It did **not** close `tanksize` — this doc scopes the residual
and the research lever that would.

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
