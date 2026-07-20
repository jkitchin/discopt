# Native-Warm-LP Convex Kernel — implementation plan (issue #807)

**Status:** not started. This is the durable, loop-executable spec for the #807
rewrite: one persistent LP, bounds modified in place across the whole tree,
dual-warm reoptimized per node — the SCIP/BARON per-node-throughput architecture.
It survives context loss: each work iteration reads this doc in full, takes the
first unfinished task, runs its entry experiment BEFORE implementing (CLAUDE.md
§4), honors the kill criterion, and records measured results in the work log
(newest first). It is self-contained: every anchor number a task needs is in §2.

**Provenance (do not re-derive).** #800 closed on its scoped win (the convex
panel certifies cert-clean within the 120 s production budget; net-positive vs
NLP-BB) after falsifying all three of its per-node/search levers by entry
experiment — T1 row-order, T2 reliability branching, T3 warm-child restart
(records: PR #805, `docs/dev/convex-kernel-plan.md` work log, 2026-07-20
entries). #801 closed pinning BARON's edge on this class as *"native warm LPs +
reduced-cost range reduction — the only deficit is per-node throughput."* #807 is
the intersection of those two findings: the throughput gap is architectural, and
this plan is the architecture.

## 1. Why the current kernel cannot be warm (binding — the T3 finding)

The shipped kernel (`crates/discopt-core/src/bnb/convex_kernel.rs::solve_node_cut`)
**re-assembles and cold-solves the node LP from scratch at every node** (and
rebuilds the CSC every OA/separation round via `assemble`). T3 proved cross-node
warm starts cannot be bolted onto this design:

- the parent's *cut-converged* optimal basis lives on a **different, larger LP**
  (its tangent/cut set) than the child's base-only first solve — dimensions and
  meaning don't match;
- the parent's *base-only* basis (right dimensions) is **neither dual-feasible
  for the child nor bound-neutral**: warming from it drifted `node_count`
  950→1120 (+18%, different degenerate vertex — the T1 degeneracy lesson) AND
  raised wall +12% (prepare + dual-feasibility check + frequent cold fallback
  cost more than the cold solve). Reproduces iter-8's negative via the dual path.

**The enabling fact for the fix** (verified in `lp/simplex/dual.rs`,
`PreparedDual` doc, lines ~276–280): dual feasibility of a basis depends on the
objective, the basis, the matrix, and *which bound each nonbasic sits at* — **not
on the bound values**. Therefore, in a **shared** LP (same rows, columns,
objective), any node's optimal basis is a dual-feasible warm start for **any**
other node's box: a node differs from its neighbor only in the `l`/`u` vectors,
and `PreparedDual::reoptimize(l, u, b, opts)` takes fresh bounds per call. T3
failed because the LP was *not* shared. Sharing the LP is the whole architecture.

## 2. Anchors (pinned by #800 T0 — cite these, do not re-measure as baselines)

Scripts: config A = `discopt_benchmarks/scripts/issue800_t0_baseline.py`
(production/unseeded, budget 120 s); config B =
`discopt_benchmarks/scripts/issue798_k2_tree_gate.py` (seeded nodes-to-certify).
Oracle: `~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu`.

| instance  | A: wall s | A: nodes | A: inc-latency s | B: nodes (seeded) | prototype |
|-----------|----------:|---------:|-----------------:|------------------:|----------:|
| rsyn0805m |      7.81 |      353 |             5.52 |               353 |        67 |
| rsyn0810m |      8.29 |      315 |             6.96 |               177 |        60 |
| rsyn0815m |      6.70 |      237 |             4.90 |               281 |        46 |
| rsyn0820m |     80.52 |     1805 |            80.43 |                 — |         — |
| rsyn0830m |     50.80 |     1172 |            46.70 |                 — |         — |
| syn05m    |      0.00 |        3 |             0.00 |                 — |         — |
| syn10m    |      0.00 |        1 |             0.00 |                 — |         — |
| syn15m    |      0.15 |       19 |             0.13 |                 — |         — |
| syn20m    |      0.58 |       43 |             0.47 |                 — |         — |
| syn40m    |     17.93 |      631 |             7.42 |               139 |        55 |

- Config-B seeded panel total **950 nodes / ~24 s** → **~26 ms/node** average;
  each node runs ~20–50 LP solves (OA rounds × separation rounds), every round
  rebuilding the CSC.
- SCIP solves the 4-instance seeded panel class at **~0.5–0.8 s/instance**
  (~2–3 s total; #798 iter-6 measurement). The #807 wall target is ~2× SCIP.
- Cert-clean on every run is mandatory and non-negotiable (§5 below).

## 3. The architecture (one LP, bounds-in-place)

### 3.1 Row classes and their validity (the C-43 discipline, made structural)

| class | validity | lifetime in the persistent LP |
|---|---|---|
| base rows (`le_rows` ‖ `eq_rows`) | global | permanent |
| OA tangents of convex `≤` rows | **global** (a tangent underestimates its convex `g` everywhere; `oa_tangent(x̄)` does not reference the box) | permanent, shared by ALL nodes — the pool **amortizes OA convergence across the tree** (a node whose ancestors already cut its region typically needs 0–2 new tangents, vs 20–50 solves/node today) |
| integrality cuts (GMI / cover / MIR) | **local**: valid only for boxes ⊆ the derivation box (they use node variable bounds) | box-tagged; **active iff node box ⊆ derivation box**, else deactivated (below). Never active in a sibling/disjoint box — C-43 enforced by construction |

### 3.2 Deactivation without structural change (the SCIP row-aging trick)

Standard form is `[A | I] z = b` with every row owning a slack: `≤` rows
`s ∈ [0, cap]`, `=` rows `s = 0` (see `assemble`). To **deactivate** a cut row,
set its slack upper bound to the row's *global-box violation cap*
`cap_off = max(0, max-activity over the ROOT box − rhs)` — finite because the
FBBT'd root box is finite (the K1d NS lesson) — so the row can never bind for any
`x` in the root box: it is vacuous, the matrix is untouched, the basis stays
valid, and the NS safe bound stays finite. To **activate**, restore
`s ∈ [0, cap_on]` with `cap_on` = min-activity cap over the root box (as
`assemble` computes today, but over the ROOT box, not the node box — cap
tightness only affects the NS margin ~1e-12·cap, not validity). Everything is a
bound change; bound changes are exactly what the dual warm path reoptimizes.

### 3.3 `PersistentLp` (new module `crates/discopt-core/src/bnb/warmlp.rs`)

```rust
pub(crate) enum RowKind {
    Base,
    Tangent,                                  // globally valid, always active
    Cut { dlo: Vec<f64>, dhi: Vec<f64> },     // derivation box (activation test)
}

pub struct PersistentLp {
    n: usize,                    // structural columns (fixed)
    rows: Vec<AsmRow>,           // base ‖ appended (append-only between GCs)
    kinds: Vec<RowKind>,
    n_base: usize,
    csc: SparseCols,             // cached; rebuilt only on append / GC
    b: Vec<f64>,                 // rhs per row (grows on append)
    c_std: Vec<f64>,             // sign·c ‖ zeros for slacks (grows on append)
    cap_on: Vec<f64>,            // per row: active slack cap (root box)
    cap_off: Vec<f64>,           // per row: deactivation cap (root box)
    basis: Option<Basis>,        // carried across nodes AND across heap jumps
    l: Vec<f64>, u: Vec<f64>,    // current full bounds (structural ‖ slacks)
}

impl PersistentLp {
    fn new(spec: &ConvexKernelSpec, root_lo: &[f64], root_hi: &[f64]) -> Self;
    /// Set structural bounds to the node box and (de)activate cut rows by
    /// box containment. Pure bound writes; O(rows·nnz_row) for the caps test.
    fn enter_node(&mut self, lo: &[f64], hi: &[f64]);
    /// Dual-warm solve from the carried basis (PreparedDual::prepare on the
    /// cached CSC → reoptimize with current l/u/b); cold `solve_lp_cols_scaled`
    /// fallback whenever prepare fails or the warm path stalls. Stores the
    /// optimal basis back. Scaled path only (NS certification).
    fn solve_warm(&mut self, opts: &SimplexOptions) -> LpSolve;
    /// Append a row (tangent or cut): extend csc/b/c_std/caps, extend the
    /// carried basis with the new slack basic (`extend_basis` idiom).
    fn append_row(&mut self, row: AsmRow, kind: RowKind);
    /// Drop deactivated/aged cut rows when rows.len() exceeds the GC threshold;
    /// rebuild csc compactly; invalidate the carried basis (next solve cold).
    fn gc(&mut self);
}
```

`solve_tree` keeps its entire existing logic — FBBT per node, safe-bound
fathoming, pseudocost branching, incumbent acceptance (`is_integer_feasible`),
honest `TimeLimit`/`Exhausted`, the leaf-dual reported-bound rule — and swaps
`solve_node_cut` for the persistent-LP node flow:

1. `enter_node(FBBT'd box)` — bound writes only.
2. `solve_warm` — a few dual pivots typically; cold fallback is always sound.
3. OA loop: violated convex rows → `append_row(tangent)` (permanent) → warm
   re-solve, to OA convergence. Expected ~0–2 appends/node once the pool warms.
4. Separation rounds (as today: `separate_cover_csc` + `separate_gomory_cols` +
   `separate_mir` under `select_cuts`): each selected cut →
   `append_row(cut, derivation box = current node box)` → warm re-solve.
   `assert_cut_valid`-style guard per cut stays.
5. Final solve → `ns_safe_bound_csc` (scaled) → the node's rigorous dual bound.
6. Fathom/branch exactly as today. The carried basis goes to the next *popped*
   node — best-bound jumps are fine (dual feasibility holds for any bounds; a
   far jump just costs more pivots, measured in W0/W3).

**Development flag:** the new path sits behind `DISCOPT_CVX_NATIVELP`
(default-OFF) until W5; `solve_node_cut` stays intact as the byte-check
reference and the fallback. The outer `DISCOPT_CONVEX_KERNEL` routing flag and
the Python-side #779 incumbent verification are untouched.

### 3.4 Known risks (each has a task/gate below)

- **Degenerate-vertex drift:** the new path lands on different LP vertices than
  the old kernel → the tree differs. That is *expected* (search-order regime,
  flavor ii) — the guard is cert-clean + a node-blowup bar, not node parity.
- **NS under wide slack caps:** root-box caps are wider than node-box caps;
  the NS margin grows by ~1e-12·cap. W1's byte-check gates that bounds still
  certify (finite, sound) on the panel boxes.
- **Basis vs edited rows:** `PreparedDual::prepare` refactorizes from the CSC on
  every call, so appends/GC are safe; a dual-infeasible carried basis (e.g.
  after GC) → documented cold fallback, never wrong.
- **Pool growth:** tangents are permanent by design (they are the relaxation);
  cuts are GC'd. GC threshold ~4× base rows; GC forces one cold solve.
- **Warm stalls:** keep `warm_stall_guard` ON and `expel_zero_artificials: true`
  (P1.0) — stall → bounded cost → cold fallback.

## 4. Correctness contract (standing, binding — CLAUDE.md §1/§5)

- `incorrect_count ≤ 0`, zero slack. Dual bound = oracle optimum exactly on the
  panel when `optimal` is claimed; `bound ≥ incumbent` (max sense); never a
  false `optimal`; honest `TimeLimit`/`Exhausted`.
- Every node bound is `ns_safe_bound_csc` on the **scaled** solve (warm path
  equilibrates — `dual.rs` scale→solve→unscale; keep it).
- Every accepted incumbent is integer-integral + OA-tight AND #779-verified
  against the pristine model on the Python side (unchanged).
- A cut row may be ACTIVE only in a node whose box ⊆ its derivation box (§3.1).
  Never weaken this to make a gate pass; if a gate can only pass that way, the
  gate loses — stop and surface.
- Fallbacks (cold solve, old `solve_node_cut` path, NLP-BB routing) stay intact
  at every stage.

## 5. Reusable assets (do NOT rebuild)

- `lp/simplex/dual.rs`: `PreparedDual::{prepare, reoptimize}` (bound-change dual
  reoptimize — the core primitive), `solve_lp_warm_scaled_csc`, warm stall
  guard, deadline polling.
- `convex_kernel.rs`: `ConvexKernelSpec`, `ConvexRow::oa_tangent` (globally
  valid), `assemble` (fork it for the persistent build), `extend_basis`,
  `substitute_slacks`, `propagate_fbbt`, `is_integer_feasible`, `select_branch`
  + `Pseudocosts`, the whole `solve_tree` shell, the 10 unit tests.
- Separators: `lp/gomory.rs`, `lp/cover.rs`, `lp/mir.rs`, `lp/cut_select.rs`.
- Marshaling: `convex_bindings.rs` (`SpecArrays`), Python producer
  `_convex_kernel.build_convex_spec`, `issue798_k1_bytecheck.build_convex_arrays`.
- Gates: `issue798_k2_tree_gate.py` (config B), `issue800_t0_baseline.py`
  (config A), `issue798_convex_family_certclean.py`, `issue798_regime2_panel.py`.

## 6. Falsified — do NOT re-walk (binding negatives)

- Row-order / cut-append-order / branching tie-break as node-count levers
  (#800 T1: 9 variants, none beat baseline; unbiased degeneracy variance).
- Reliability/strong branching as a standalone lever (#800 T2: net-negative on
  nodes AND wall with cold probes). Revisit ONLY as W4b after W1–W2 land.
- Warming the *per-node-reassembled* first solve from the parent's base-only
  basis (#800 T3 / #798 iter-8): node drift + wall regression. The fix is the
  shared LP, not a better per-node warm heuristic.
- Root-relaxation tightening on this class beyond the OA+GMI family (#781 HOLD,
  #801's RLT/moment falsifications on tanksize).

## 7. #807 — executable task list (loop-executable)

**Loop protocol.** ONE feature branch `feat/807-native-warmlp` off `main`; ONE
draft PR ("feat(#807): native-warm-LP convex kernel (W0–W5)") opened after W0;
one scoped commit per task, pushed each task; CI green before the next task.
Before each commit: `cargo test -p discopt-core`, `pytest -m smoke`, the
adversarial suite (`pytest -m slow python/tests/test_adversarial_recent_fixes.py`),
plus the task's own gate script. Append a work-log entry (§8, newest first) with
MEASURED numbers in the same commit. Kill criterion hit → revert scaffolding (no
dead flags), record the falsification, commit+push, STOP and surface. Any
correctness regression (a bound above its reference, a false optimal, an
unverified incumbent, cert-fail) → STOP immediately and surface.

### W0 — entry experiment: in-place dual reoptimize on the real instance. **BLOCKING.**
- **Goal.** Validate the core claim — same-LP bound-change dual reoptimize is
  much cheaper than today's per-node cold pipeline — on a REAL panel instance
  before any architecture is built.
- **Hypothesis + evidence.** In a shared LP, the previous node's optimal basis is
  dual-feasible for any node box (§1, `PreparedDual` contract); SCIP/BARON
  reoptimize children in a few dual pivots (#801: ~500×/node edge). Today's
  per-node cost is ~26 ms (§2).
- **Entry experiment.** Temporary PyO3 probe `convex_warmlp_probe_py`
  (pattern: K1d) + script `issue807_w0_probe.py`, on **rsyn0815m** (heaviest
  per-node) and **syn40m** (most nl rows): (a) build the root LP = base rows +
  root-OA-converged tangent pool over the FBBT'd root box; (b) generate ≥20
  realistic child boxes by replaying pseudocost branching + FBBT from the tree
  (branch var fixed down/up, FBBT-propagated); (c) per child, measure — cold:
  today's full `solve_node_cut` (sep=0) wall; warm: bounds-in-place +
  `PreparedDual::prepare`+`reoptimize` from the previous solve's basis — wall,
  **pivots**, bound agreement (≤1e-6), NS certification (finite safe bound),
  and the number of NEW tangents needed to re-converge OA (pool-amortization
  check). Include at least 5 *non-adjacent* child pairs (simulated best-bound
  jumps).
- **Kill criterion.** Warm (prepare+reoptimize+OA-reconverge) not ≥**2×** faster
  than the cold node solve on the median child, OR warm bound ≠ cold bound
  beyond 1e-6, OR NS fails to certify on any warm solve. Kill ⇒ #807 is
  falsified at the architecture level — record, revert the probe, close the
  issue on the record; there is no fallback lever.
- **Verification regime.** Measurement only; bound parity + NS certification
  asserted within the probe.
- **Done-when.** Probe numbers (median/p90 warm vs cold wall, pivots, new
  tangents per child, jump-pair penalty) recorded in the work log; GO/KILL
  stated explicitly.

### W1 — `PersistentLp` core + tree integration, OA-only.
- **Goal.** The persistent LP + bounds-in-place + dual-warm node solve wired
  into `solve_tree` behind `DISCOPT_CVX_NATIVELP` (default-OFF), OA tangents
  only (`max_sep_rounds=0`), old path untouched.
- **Hypothesis + evidence.** W0's measured speedup transfers to the tree;
  tangent-pool sharing collapses per-node OA rounds (W0 measures this).
- **Entry experiment.** None beyond W0 (this is the build W0 de-risked).
- **Kill criterion.** Per-node wall (OA-only, node-capped seeded runs on the
  4 config-B instances) not ≥2× better than the old kernel at the same setting,
  or bound parity/NS gates fail. Kill ⇒ back to W0's data to find the discrepancy;
  if irreconcilable, surface.
- **Verification regime.** Bound-parity byte-check (K1-style, flavor i on the
  *node relaxation*): script `issue807_w1_bytecheck.py` — new-path node bound =
  old `solve_node` bound to ≤1e-6 over the root box + ≥6 child boxes × 4
  instances, NS certifying on all. Tree-level: flavor ii (vertices may differ) —
  cert-clean on node-capped runs. `cargo test -p discopt-core` green; new unit
  tests for `enter_node` activation logic, `append_row`+`extend_basis`, GC, and
  cold-fallback-on-dual-infeasible.
- **Done-when.** Byte-check PASS; per-node OA-only wall ≥2× better (measured,
  in the log); flag-OFF path bit-identical (smoke + adversarial green).

### W2 — box-tagged cut rows + full branch-and-cut tree on the panel.
- **Goal.** Integrality cuts in the persistent LP with box-tagged activation
  (§3.1–3.2) + GC; the full tree runs the panel under the flag.
- **Hypothesis + evidence.** Cuts drive the 15–18× node reduction (#797); the
  activation rule preserves C-43 by construction; deactivation is pure bound
  writes so the warm chain survives.
- **Entry experiment.** None separate — W2 is gated by its panel run (the
  build is W0/W1-de-risked; the open question is only the measured wall).
- **Kill criterion.** Seeded config-B panel wall not ≤**12 s** (≥2× vs 24 s
  anchor), OR nodes > **1.5×** the config-B anchor per instance (tree-quality
  blowup guard: 353/177/281/139 → caps 530/266/422/209), OR any cert failure.
  Wall-kill ⇒ profile (pivots/node, appends/node, GC frequency), fix, re-gate
  once; still failing ⇒ surface with the profile.
- **Verification regime.** Search-order (flavor ii): cert-clean mandatory (dual
  = oracle exactly, `bound ≥ incumbent`, per-cut validity guard, incumbents
  #779-verified via the production path), nodes within the blowup bar, wall the
  measured target. Scripts: `issue798_k2_tree_gate.py` under the flag +
  `issue800_t0_baseline.py` under the flag (config A: full 10-instance panel
  cert-clean; rsyn0820m ≤40 s, rsyn0830m ≤25 s).
- **Done-when.** Both panel runs cert-clean with the bars met; numbers vs §2
  anchors in the log. Stretch (record, don't gate): seeded panel ≤8 s.

### W3 — cross-node jump cost + plunging order (optional, entry-first).
- **Goal.** If best-bound jumps dominate pivot counts, recover wall with SCIP
  plunging (process one child of the just-solved node first, bounded-depth DFS
  bursts inside the best-bound loop).
- **Hypothesis + evidence.** Dual reoptimize cost scales with bound distance
  from the previous solve; W0's jump-pair measurement quantifies it. SCIP
  plunges for exactly this reason.
- **Entry experiment.** From W2's instrumented runs: pivots-per-node vs
  box-distance-from-previous-node. If adjacent nodes average <⅓ the pivots of
  jump nodes AND jumps are >30% of nodes, plunging has headroom; else skip W3
  (record why).
- **Kill criterion.** Plunging does not cut panel wall ≥**10%** vs post-W2, or
  hurts nodes >1.5× anchors, or any cert failure ⇒ revert, record.
- **Verification regime.** Search-order (flavor ii): cert-clean + node bar +
  wall target. Best-bound *fathoming/termination* logic must stay intact
  (plunging changes the processing order, never the bound logic).
- **Done-when.** ≥10% wall gain recorded, or the skip/kill recorded.

### W4 — reduced-cost range reduction (RCRR) — the #801 second lever.
- **Goal.** At each node optimum with an incumbent, tighten variable bounds from
  reduced costs: nonbasic-at-lower `j` with reduced cost `d_j > 0` (min-sense
  standard form) admits `x_j ≤ l_j + gap/d_j` (symmetric at upper); integers
  floor the tightened bound. Tightenings feed FBBT and shrink child boxes.
- **Hypothesis + evidence.** #801 names RCRR as BARON's second ingredient; it is
  the one *node-count* mechanism not touched by the #800 falsifications (it is
  a bound-tightening, not an ordering/branching heuristic). Free at a solved
  node (reduced costs are already there).
- **Entry experiment.** Instrument-first on the seeded panel (post-W2 path):
  count applicable tightenings/node and the box-volume reduction, WITHOUT
  applying them. If <5% of nodes get any tightening, kill before building.
  Then apply behind a sub-flag and measure nodes + wall.
- **Kill criterion.** Applied RCRR improves neither panel nodes nor wall by
  ≥**10%** (cert-clean) ⇒ revert, record. **Soundness bar (non-negotiable):
  the `gap` used must be the certified one** (incumbent − NS-safe dual bound,
  directed-rounded outward); never the raw LP objective. A tightening that cuts
  the oracle optimum on any panel instance is an immediate stop-and-surface.
- **Verification regime.** Bound-changing (§5 Regime-2 flavor): sub-flag
  default-OFF until the panel run is cert-clean AND net-positive; differential
  check — no tightened bound may exclude the instance's oracle optimum vector
  (assert per node on the panel: oracle x within the tightened box whenever it
  was within the parent box).
- **Done-when.** ≥10% node or wall gain cert-clean with the differential guard
  green → sub-flag folds into the W2 path; else the falsification is recorded.

### W4b — (optional) reliability branching, revisited with cheap warm probes.
Only if W1–W2 landed and W4 is resolved. #800 T2's record permits exactly this
revisit: strong-branch probes via `PreparedDual::reoptimize` (bounds-only, a few
pivots each, cut-consistent by construction — the shared LP *is* the node LP).
Entry: η=8, K=8 on the seeded panel. **Kill:** <15% node cut or any wall
regression ⇒ revert, record, done. Search-order regime (flavor ii).

### W5 — checkpoint + graduation.
- **Goal.** Confirm the #807 bars; graduate the flags.
- **Bars.** (a) cert-clean everywhere (mandatory); (b) **wall:** seeded 4-panel
  ≤ ~2× SCIP (≤6 s total; SCIP ~2–3 s, §2) and config-A full panel with
  rsyn0820m/0830m well inside the 120 s budget (≤40 s each; stretch ≤15 s);
  (c) **nodes:** report the ratio vs prototype (67/60/46/55) honestly.
  *Decision point for the owner:* #800's ≤2× node gate was set for a
  cutting-driven design; with node levers falsified (T1/T2) and RCRR (W4) the
  only remaining node mechanism, propose graduating on cert-clean +
  net-positive wall with the node ratio *recorded*, not gating. Do not silently
  drop the gate — surface it.
- **Graduation.** Flip `DISCOPT_CVX_NATIVELP` default-ON inside the kernel (old
  path kept as byte-check reference + `=0` opt-out). Then run the outer-flag
  Regime-2 gate (inherited from #798/#800 T7): `issue798_regime2_panel.py` +
  `issue798_convex_family_certclean.py` + config A, flag-ON vs flag-OFF —
  cert-clean (incorrect 0, no bound above reference, no certification
  regression, incumbents verified) AND net-positive (wall/nodes) ⇒ propose
  `DISCOPT_CONVEX_KERNEL` default-ON to the owner (keep the opt-out and the
  NLP-BB fallback intact).
- **Done-when.** Both flag decisions executed (or the owner's explicit deferral
  recorded), all numbers in the log, PR flipped ready-for-review.

## 8. Work log (append newest first)

*(empty — W0 not started)*
