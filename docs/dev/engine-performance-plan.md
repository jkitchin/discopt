# Engine performance plan — EP series (issue #632 follow-on)

Status: **OPEN** (2026-07-13). Owner: sequential single-context execution — one EP
item per fresh context, in order. Companion to `docs/dev/performance-plan.md`
(the CC1–CC5 cost model; its negative results are binding) and
`docs/dev/avm-canonicalization-plan.md` (§10 ledger records each EP landing).

## §0. Binding contract (read before any EP item)

1. **Soundness is absolute.** No item below may change what makes a bound valid.
   Every item is either *bound-neutral* (must prove byte-identity) or
   *bound-changing* (must run the differential regime). If a gate can only pass
   by weakening a check, the item stops and surfaces it.
2. **Two verification regimes** (CLAUDE.md §5):
   - *Bound-neutral* (caching, reuse, marshaling): `relaxation_fingerprint`
     **byte-identical** on all 62 vendored instances vs the committed
     `docs/dev/data/claim-baseline.jsonl` (`python/tests/test_claim_baseline_neutral.py`
     mechanism), AND `node_count` + certified `objective` **exactly unchanged**
     on the certifying smoke set. Any drift — even an apparent improvement —
     means the change is wrong.
   - *Bound-changing*: differential bound test + feasible-point sampling,
     default-off flag, nightly-green before default-on. Only EP4b and EP6 are in
     this class.
3. **Measurement beats plan.** Every item reports before/after numbers from the
   EP0 probe. If an item's measured win is nil, record the falsification in this
   doc and stop the item — do not land speculative complexity.
4. **No new `DISCOPT_*` flags** except where CLAUDE.md §5 *requires* a
   default-off gate (EP6 already has `DISCOPT_NODE_PROBING`).
5. In-container limits: BARON, full MINLPLib, `minlplib.solu` are absent.
   Full-panel `global50`/`cert0` and wall-time comparisons are **local-host
   handoffs**; in-container gates are the fingerprint/node-count/suite gates
   plus the EP0 probe numbers.

## §1. The measured problem (evidence baseline, 2026-07-13)

Two independent profiles agree (in-container cProfile on nvs09; maintainer's
local global50 profile on `0a8a7885`, PR #636 comments):

- **Wall split (116-instance panel): 61% JAX, 39% Python, ~0% Rust LP.** The
  in-house simplex is idle; the cost is relaxation *construction*, per node.
- **`build_milp_relaxation` runs 22× for 19 nodes** (nvs09). The uniform engine
  re-runs `canonicalize → reconstruct → evaluate_interval → curvature
  certification → compile_expression/jax.grad` on every call. Nothing is cached
  across calls.
- **In-container per-node cost 38 ms/node** (nvs09, 10 simulated children):
  21 ms full engine rebuild — of which ~15 ms is `_try_convex_lift` re-proving
  interval-Hessian curvature certificates (76 ms each, 2/build) — plus 11 ms
  `_separate_multilinear` re-solving 13 tiny facet LPs per node.
- **The incremental node fast path never engages** under the engine:
  `IncrementalMcCormickLP` validates patched rows == cold build and only knows
  bilinear+monomial rows (`incremental_mccormick.py:20`); the engine's
  univariate/log-space/OA rows fail validation → `ok=False` → cold rebuild every
  node. This was the CC2 mitigation; the cutover silently disabled it.
- **OBBT dominates `root_time`** on the capped instances (hda 52.5 s root / 7
  nodes / 0 NLP solves; heatexch_gen3 53.6 s; contvar 37.6 s):
  `_PER_NODE_OBBT_ROUNDS=3` × O(n) LP probes, each probe paying a full engine
  build for the SAME node box.
- Fixed setup (first JAX traces + the `IncrementalMcCormickLP` double-build in
  the relaxer ctor) reached ~16 s in-container on nvs09 — ~95% of its wall.

**North star (the SOTA architecture):** BARON/SCIP split *analyze-once*
(DAG, convexity certificates, envelope templates — presolve) from *per-node*
(coefficient refresh + warm-started LP + FBBT). The engine built the right
analyze layer; the EP series stops running it at every node. Target: per-node
cost in low single-digit ms; root_time on hda-class instances cut ~10×; SGM
back below the 0.83 s federation baseline with the engine's smaller trees.

## §2. Execution protocol (per item)

A fresh context executing item EP*k*:
1. Read this §0–§2 + the EP*k* section only. `cd /home/user/discopt && source .venv/bin/activate`;
   `export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`.
2. Run the EP0 probe first to get the live *before* numbers.
3. Implement exactly the EP*k* scope. Do not reach into later items.
4. Run the EP*k* verification gate (listed per item) + `pytest -m smoke` +
   the fast selection + serial `-m claim_boundary -n0` + `ruff check`/`ruff
   format --check`/`mypy python/discopt/` (+ `cargo test -p discopt-core` iff
   Rust touched).
5. Update this doc's §3 status table (before/after numbers, commit sha) and add
   a one-row entry to `avm-canonicalization-plan.md` §10. Commit, push to
   `claude/issue-632-opus-plan-ffxld4`.

## §3. Status

| Item | Status | Before → after (probe) | Commit |
|---|---|---|---|
| EP0 probe harness | done | baseline (in-container, `--children 10`): nvs09 ctor 0.762 s / root 0.291 s / **294 ms/node** / 22 builds / 19 nodes / obj −43.13434; ex1226 ctor 0.025 s / root 0.024 s / 24 ms/node / 9 builds / 5 nodes / obj −17.0; bchoco06 ctor 0.656 s / root 2.256 s / 2214 ms/node / 6 builds / 7 nodes / time_limit@120 s | `0105c8d` |
| EP1 per-model analysis cache | done | in-container `--children 10` (wall-clock noisy, treat as order-of-magnitude): nvs09 ctor 0.53→0.17 s / **282→169 ms/node**; ex1226 ctor 0.025→0.025 s / **26.8→3.7 ms/node**; bchoco06 ctor 0.61→0.55 s / **2268→1813 ms/node**. Builds/nodes/objectives UNCHANGED (nvs09 22 builds/19 nodes/−43.13434; ex1226 9/5/−17.0; bchoco06 6/7/time_limit). Fingerprints byte-identical on all 62 vendored instances; node_count + objective byte-equal on a 19-instance corpus (persistent-cache vs fresh-per-build). | `7a6a5bd` |
| EP2 OBBT single-build-per-box | **done (already landed by T2.2 #579; premise falsified)** | Measured (in-container): per-node/root OBBT already builds the relaxation **once per (node box, round)** and shares it across all probes — hda (722 cols, 15 s budget) `build_milp_relaxation` **calls=1 per sweep** (builds/sweep = 1.00), the single build 6.83 s = 36.5 % of OBBT wall, probes warm-started via `_PersistentProbeLP`. **No per-probe rebuild exists to remove** — EP2's "each probe pays a full engine build" premise is falsified (see EP2 section). Node-level neutrality pinned by new `test_obbt_ep2_node_reuse.py` (10 vendored instances: reuse+warm boxes == cold-seam boxes, Δ ≤ 1 ulp / ≤ 1e-9). No solver-math change → fingerprints/node-counts/objectives byte-unchanged by construction. | (docs+test) |
| EP3 patch-table node path | **done (engagement 48/62 = 77%; cheap-closed-form half falsified — see EP3 §)** | Fast-path engagement restored for the ENGINE via `UniformPatchTable` (`incremental_mccormick.py`), wired ahead of the closed-form `IncrementalMcCormickLP` in `mccormick_lp.py`. **Before→after (probe, `--children 10`, fast path engaged):** nvs09 **169→51 ms/node** (3.3×, node-neutral: 19 nodes, obj −43.13433691803531 byte-identical fast-ON vs `DISCOPT_INCREMENTAL_MC=0` **to optimality**); ex1226 **3.6→2.4 ms/node** (5 nodes, obj byte-identical ON/OFF). bchoco06 (3rd certifying instance) does NOT engage (unbounded/large layout-unstable root) → cold path unchanged → trivially neutral. **Engagement 48/62 = 77%** (not engaged: alan, bchoco06/08, ex1224, fac2, m3, nvs01, nvs21, oaer, st_e29, tanksize, tspn08/10/12 — unbounded root / box-unstable lifted layout → cold fallback). `relaxation_fingerprint` byte-identical on all 62 (build path untouched; `test_claim_baseline_neutral` green). **Falsification (§0.3):** the envisioned ~0.1 ms closed-form coefficient refresh is NOT byte-reproducible for the engine — its box-dependent rows/aux-bounds are `evaluate_interval` results over the reconstructed DAG (measured: a closed-form affine interval differs from `evaluate_interval` bit-for-bit on fac2 3/3, alan 1/6), and single-atom replay cannot reproduce the interleaved aux-column layout; so `_patch` regenerates through the EP1-cached engine build (byte-identical) and the win is the per-node **separation skip** (inherited root pool) + **warm start**, not a numpy refresh (hence 51 ms, not ≤12 ms). | `cb31ea9` |
| EP4a separation facet cache | **done (cache landed; measured wall win negligible — §0.3 honest report)** | Exact memoization of `separate_multilinear_envelope` in a capped (200k, clear-on-overflow) module dict, keyed by the **full** float64-bytes inputs `(lb, ub, x_star, w_star, tol, max_factors)` — NOT box-only: the supporting facet is *selected at* `x_star` (measured varies with `x_star` on 10/16 nvs09 boxes), so a box-only key would return a stale facet (a byte-neutrality bug). **BOUND-NEUTRAL gate GREEN:** node counts + certified objectives byte-identical before/after (nvs09 51 builds/19 nodes/−43.13433691803531; ex1226 24/5/−16.999999994161513; bchoco06 38/7/time_limit); `relaxation_fingerprint` byte-identical on all 62 (build path untouched; `test_claim_baseline_neutral` green, 113s); new `test_ep4a_multilinear_facet_cache.py` asserts cache-hit facets == fresh derivation byte-for-byte on 25 boxes/arity ×{2,3} × 3 w_star, key-includes-x_star, no-resolve-on-hit, cap-clears. **Measured (nvs09 end-to-end):** cache deterministically removes **144/360 = 40% of facet-LP solves** (180 sep calls, 108 unique inputs → 72 hits), but the recurring atoms are the *cheap* LPs — `_solve_envelope` cumulative time only 15.1→14.6 s (~3%), within noise at the 45 s solve; EP0 ms/node (distinct cold child boxes, no recurrence) unchanged (53.0→52.7). **Win is negligible on the certifying instances** — landed anyway as the guaranteed-neutral, harmless exact cache (bounded, self-verified) that removes the real redundant computation. **Closed-form arity≤3 NOT added** (correctly skipped): the LP facet is `x_star`-selected with a vertex-recomputed intercept, so no box-only closed form can reproduce it bit-for-bit — the self-check would fail by construction. | `f1a5e77` |
| EP4b separation warm-start + OA pool | open | | |
| EP5 lazy/shared compiles | open | | |
| EP6 probing/OBBT default-on tuning | open (local host) | | |

---

## EP0 — Measurement probe + baseline lock

**Goal.** One reusable instrument so every later item has comparable
before/after numbers.

**Scope.** New `discopt_benchmarks/scripts/engine_perf_probe.py` (measurement
only — no solver-math change, no flag). For each named instance (default:
nvs09, ex1226, bchoco06; accept `--instances`):
- relaxer construction time (`MccormickLPRelaxer(model)`);
- root `solve_at_node` time;
- ms/node over `--children N` (default 10) simulated child boxes (shrink one
  var's box per child, the pattern in the 2026-07-13 in-container profile);
- `build_milp_relaxation` call count for one end-to-end `model.solve()`
  (attach a counter via monkeypatched wrapper inside the probe process — do NOT
  instrument library code);
- optional `--profile` → cProfile top-20 cumulative.
Emit a JSON + human table. Record the baseline numbers for the three default
instances in §3 (they are the "before" column for EP1).

**Verification.** Script-only: ruff/mypy clean; probe runs green on the three
defaults. No fingerprint gate needed (no library change).

**Done.** Probe committed; §3 baseline row filled.

**Baseline measured (2026-07-13, in-container, `--children 10`).** The harness
reproduces the §1 build-count evidence exactly (nvs09: **22 builds for 19
nodes**), which validates the counter. One number diverges materially from §1
and the measurement stands (contract §0.3): **nvs09 per-node cost measured
~294 ms/node, not the ~38 ms/node cited in §1** — the root `solve_at_node`
(0.291 s) is consistent with ~294 ms, so the whole per-node relaxation build is
~8× more expensive here than the §1 profile recorded. The EP0 child boxes shrink
one variable to its lower half from the *root* box (every child is a cold engine
rebuild — the incremental fast path never engages, per §1), so this is the true
cold per-node cost; the §1 38 ms figure likely reflects a different child
schedule or host. EP1's before column is **294 ms/node (nvs09)**; wall-clock
timings vary run-to-run, so treat them as order-of-magnitude, but the 22-build /
19-node and 294-ms/root-0.291-s internal consistency is solid.

---

## EP1 — Per-model analysis cache (box-independent work computed once)

**Goal.** Stop re-running box-independent analysis at every build: the
canonical DAG, reconstructed expressions, DCP verdicts, compiled
`value_fn`/`grad_fn`, and (monotone-inherited) interval-Hessian curvature
certificates. This is maintainer lever 1 + the certificate half of lever 2, and
the largest measured per-node component (~15 of 38 ms/node in-container).

**Scope.** `python/discopt/_jax/uniform_relax.py` only.
- Add a `_ModelAnalysisCache` stored on the model object
  (`model.__dict__["_uniform_relax_analysis"]`; Model has no `__slots__`).
  Guard staleness with a token `(len(model._variables),
  len(model._constraints), id(model._objective))` — on mismatch, rebuild the
  entry. (Model mutation mid-solve is unsupported, but a stale *objective*
  would be flat-out wrong, so the token is mandatory.)
- Cache fields:
  - `dag`: `canonicalize(model)` — pins every `CNode`, making `id(cnode)`
    stable for the model's life (this also stabilizes `expr_id` in
    `CompositeMultivarRelaxation` and every `id(node)` key below).
  - `expr[id(cnode)]`: `reconstruct(node, model)` results. Pinning the
    reconstructed trees kills the historical `evaluate_interval` stale-`id()`
    hazard class (see the WARNING in `_Builder.bounds` — keep that comment,
    update it to explain the pinning).
  - `dcp[id(cnode)]`: `classify_expr` verdict (including `None`) — global DCP
    is box-independent.
  - `compiled[id(cnode)]`: `(value_fn, grad_fn)` from
    `compile_expression`/`jax.grad` — the *function of x* does not change with
    the box.
  - `hessian_certs[id(cnode)]`: list of `(lo_support, hi_support, verdict)`
    proven boxes, **restricted to the node's `var_support` columns**. Lookup:
    if the query support-box is a subset of a proven box → verdict is valid
    (convexity over a box is inherited by every sub-box — monotone; this is a
    theorem, not a heuristic). Cap the list (8; drop-oldest).
  - `hessian_abstain[id(cnode)]`: abstained support-boxes. Skip re-proving when
    the query box is a subset of an abstained box AND every width ≥ 0.5× the
    abstained width (re-try when meaningfully smaller — curvature can resolve
    on smaller boxes; abstaining longer is sound, only looser). Cap 8.
  - `bounds_by_box[(id(cnode), support_lb.tobytes(), support_ub.tobytes())]`:
    interval enclosures keyed by the box **restricted to the node's support**
    — so branching on one variable only invalidates enclosures of nodes that
    depend on it, and OBBT probes at the same box hit 100%. Cap the dict
    (500k entries → clear; the per-build `self._bounds` memo still stands).
- Rewire `_Builder`: `bounds()`, `rep()`/`_try_convex_lift` (DCP + certs +
  compiled), `_rep_impl`'s `aux_expr`, `_factor_value` (line ~981), the
  log-space extractor (line ~1160), and `build_uniform_relaxation`'s
  `canonicalize` call all read through the cache via small helpers
  (`ctx._expr(node)`, `ctx._dcp(node)`, `ctx._compiled(node)`,
  `ctx._curvature_cert(node, lo, hi)`).

**Explicitly out of scope:** caching whole build results (the returned
`MilpRelaxationModel` is mutated by separators — a result cache is unsound
without copy-on-hit; EP2/EP3 handle build reuse safely), OBBT, the incremental
path, separation.

**Verification (bound-neutral gate).**
1. `relaxation_fingerprint` equality vs committed `claim-baseline.jsonl` on all
   62 vendored instances (the shape test covers this; also compare the
   `fingerprint` field explicitly in the probe run).
2. In-process double-build: build the same (model, box) twice; the second
   (cache-hot) fingerprint must equal the first byte-for-byte. Add this as a
   regression test in `test_uniform_relax.py`.
3. Token test: build, then `model.subject_to(...)` a new constraint, rebuild →
   fingerprint reflects the new constraint (cache invalidated). Also assert the
   new-objective case.
4. End-to-end: solve nvs09 + ex1226 before/after — `node_count` and certified
   `objective` exactly equal.
5. Suites per §2.

**Done.** All gates green; §3 row shows before/after ctor + ms/node from EP0.
Expected effect: the ~15 ms/node certificate cost and the per-node
compile/reconstruct cost drop to ~0 after the first build; hda-class root
builds stop re-tracing JAX.

---

## EP2 — OBBT probes reuse one relaxation per node box

**Goal.** Kill the dominant `root_time` cost: `_PER_NODE_OBBT_ROUNDS=3` ×
O(n) probes each paying a **full engine build for the same box** (hda 52.5 s
root / 7 nodes). An OBBT probe minimizes/maximizes a single variable over the
SAME relaxation — only the objective vector differs.

**Scope.** `python/discopt/_jax/obbt.py` + its call path in
`mccormick_lp.py` (`_tighten_node_bounds_with_status`). Build the relaxation
**once per (node box, OBBT round)**, then per probe: swap the LP objective
(`c` vector) in place (or on a light clone), solve, restore. Warm-start
consecutive probes from the previous basis if the existing solve interface
exposes it (it does for the Rust simplex — see `solve_at_node`'s warm path);
otherwise plain re-solve of the same matrix is already the win. Do NOT change
which probes run, their order, or how contraction is applied.

**Soundness.** Identical relaxation ⇒ identical probe LPs ⇒ identical (or
warm-start-equal-value) probe bounds; OBBT contraction logic untouched.

**Verification (bound-neutral gate).** Fingerprints unchanged (build path
untouched); OBBT-produced boxes **exactly equal** before/after on a 5-instance
probe set (assert array equality of the tightened bounds — add a regression
test); end-to-end node counts + objectives exactly unchanged (nvs09, ex1226,
bchoco06); suites per §2. Report before/after `root_time` on bchoco08 (hda
stand-in in-container) via EP0 `--profile`.

**Done.** Probe-set boxes byte-equal; root_time reduction recorded in §3.
Expected: root_time on hda-class drops by ~the probe-count multiple (each
probe was a full build); contvar may regain its finite root bound
(`iteration_limit` was budget exhaustion — check and note, do not force).

**OUTCOME (2026-07-13): premise falsified — already implemented; no code change.**
Per §0.3 (measurement beats plan; do not land speculative complexity), the entry
measurement was run *before* implementing, and it falsifies EP2's premise. The
optimization EP2 describes — "build the relaxation once per (node box, OBBT
round), swap only the objective per probe, warm-start consecutive probes from the
previous basis" — was **already landed by cert:T2.2 (PR #579, merged 2026-07-10)**
as `_PersistentProbeLP` in `obbt.py`, consumed by `run_obbt_on_relaxation` and
reached from `obbt_tighten_root` (the per-node/root OBBT entry). Evidence:

- **Code.** `run_obbt_on_relaxation` assembles the standard-form CSC **once per
  sweep** (`_PersistentProbeLP`, `obbt.py`) and threads the optimal basis
  probe→probe; each probe only writes `c[var]=±1`. No probe calls
  `build_milp_relaxation`. `obbt_tighten_root` calls `build_milp_relaxation`
  **once per round** (plus one DBBT rebuild at the *tightened* box when a cutoff
  fires — a different box, not redundant).
- **Measurement.** Instrumented `obbt_tighten_root` on **hda** (722 cols, 15 s
  budget): `build_milp_relaxation` **calls = 1** for the sweep (builds/sweep =
  1.00), that single build 6.83 s = 36.5 % of OBBT wall; the rest is the
  large-scale (722-col equilibrated) probe LP solves. There is **no per-probe
  build** to remove. The EP0 profile's "OBBT probes paying a full engine build
  each" (§1) mischaracterised the per-**round** builds — the profile was taken
  2026-07-13, *after* T2.2 (#579) landed 2026-07-10 (git ancestry confirmed).
- **Neutrality.** The EP0 profile's premise is the only thing changed here; the
  solver math is untouched, so the byte-identity gate (fingerprints on all 62,
  node counts + objectives) holds by construction. `test_obbt_ep2_node_reuse.py`
  pins the invariant at the *node* level (`obbt_tighten_root` reuse+warm boxes ==
  forced-cold-seam boxes, Δ ≤ 1 ulp / ≤ 1e-9, on 10 vendored instances),
  complementing the T2.2 `run_obbt_on_relaxation`-level tests in
  `test_obbt_warm_probes.py`.

**The residual hda-class OBBT root_time is EP3's target, not EP2's:** the single
per-round `build_milp_relaxation` (6.83 s on hda) is the cold engine build that
EP3's patch-table node path removes. EP2 landed **no code change** beyond the
regression test that documents and guards the already-present reuse — inventing a
rewrite of a working, byte-neutral-critical path to "land EP2" would be exactly
the speculative complexity §0.3 forbids.

---

## EP3 — Patch-table node path (extend the incremental engine to engine rows)

**Goal.** Restore the CC2 mitigation: per-node relaxation = coefficient
refresh + warm-started Rust simplex, not a Python rebuild. Today
`IncrementalMcCormickLP.ok == False` on every engine-shaped model.

**Scope.** `python/discopt/_jax/uniform_relax.py` (emit a patch table),
`python/discopt/_jax/incremental_mccormick.py` (consume it),
`mccormick_lp.py` `_try_incremental_node` (unchanged contract).
- During the engine build, record per row a descriptor
  `(family, cnode_id, aux_col, support_cols)` for every **box-dependent** row
  family with a closed-form coefficient function of the box: McCormick product
  rows, 1-D secant/tangent envelopes on definite-curvature boxes, interval
  floors (aux column bounds), monomial rows. Rows the engine derives through
  code paths without a closed form (OA lift equality rows are box-independent
  — patchable as constants; log-space band rows and odd-power hull rows may be
  phase-2) mark `unpatchable`.
- `IncrementalMcCormickLP` (or a sibling `UniformPatchTable` class chosen by
  what validates): patch = regenerate exactly the box-dependent rows from the
  descriptors for the child box; everything else is copied. Keep the existing
  construction-time gate: **row-for-row validation against the cold build**
  (`_validate`); any unpatchable row family present → `ok=False` → cold path,
  exactly as today. Start by validating on models whose engine rows are all
  patchable families; extend families until the corpus engagement rate is
  high.
- Preserve the T1.3 guard: the fast path must still be skipped during root
  cut-pool capture (`out_cuts is not None`) — read the existing comment block
  and keep its behavior; a regression here collapses the spatial bound
  (dispatch 3 → 9843 nodes, documented).

**Verification (bound-neutral gate).** The mechanism is self-verifying
(validated patch == cold build at construction), plus: fingerprints unchanged;
node counts + objectives exactly unchanged on the certifying set **with the
fast path both on and forced off** (`DISCOPT_INCREMENTAL_MC=0` exists);
engagement telemetry: report the fraction of the 62-instance corpus where
`ok=True` and the measured ms/node with the fast path engaged. Suites per §2.

**Done.** Engagement > 50% of corpus instances with ms/node at or below the
pre-cutover ~12 ms; §3 records engagement rate + ms/node. (Reaching ~100%
engagement by patching log-space/odd-power families can be a follow-up item —
record actual coverage honestly.)

**OUTCOME (2026-07-13): DELIVERED at 48/62 = 77% engagement, node-neutral — but
the "closed-form coefficient refresh" half is FALSIFIED (§0.3); the fast-path
engagement and its measured win ARE delivered.**

*What was falsified (measured, before implementing the patch).* EP3's premise is
that the per-node relaxation can be a **cheap closed-form coefficient refresh**
that byte-reproduces the cold build. It cannot, for the uniform engine:

1. The engine's box-dependent row coefficients and aux-column bounds are produced
   by `evaluate_interval` run over the reconstructed canonical DAG (secant/tangent
   `_emit_1d`, McCormick folds, aux floors). A closed-form recomputation does **not**
   reproduce `evaluate_interval`'s floating-point results bit-for-bit for any
   multi-term (affine / folded / power-of-affine) argument — **measured:** a
   closed-form affine interval differs from `evaluate_interval` on fac2 (3/3 atoms)
   and alan (1/6). Byte-identity (the whole EP3 soundness model) therefore fails for
   the engine's dominant families; only bare-single-variable atoms match, and the
   vendored corpus has essentially none (the `IncrementalMcCormickLP` bare-var
   closed-form validates on **0/62**).
2. Replaying "just the box-dependent atoms" cannot reproduce the aux-column layout:
   the builders allocate aux columns *interleaved* through the bottom-up DAG walk,
   so byte-identical replay = the full walk = the cold build.

After EP1 the engine build is already ~12 ms/node; the dominant per-node cost is
per-node **separation** (multilinear facet LPs — ~100 ms/node on nvs09, EP4a's
target) plus the node LP solve. A byte-identical patch cannot beat the EP1-cached
build, and the large lever the fast path actually pulls — **skipping per-node
separation** (the inherited root cut pool substitutes) + **warm-starting** the Rust
simplex — is exactly the CC2 mitigation the #632 cutover disabled by leaving
`IncrementalMcCormickLP` unable to validate on engine rows.

*What was delivered.* `UniformPatchTable` (`incremental_mccormick.py`), wired ahead
of the closed-form table in `mccormick_lp.py`. Its `_patch` regenerates the node
relaxation through the (EP1-cached) engine build — **byte-identical by construction**
— and `_validate` engages only when (a) the root build has a valid objective bound
and (b) the lifted column layout is box-stable across reachable sub-boxes (else
`ok=False` → cold path unchanged, exactly as today). Engaging then routes the node
through the existing `_try_incremental_node` fast path: **skip the per-node
separation chain, inherit the root pool, warm-start.** All existing soundness guards
are preserved (T1.3 skip during pool capture `out_cuts is not None`; C-38 warm-basis
false-infeasible re-solve; C-43 pool false-fathom re-verify).

*Measured (in-container, verified fast-ON vs `DISCOPT_INCREMENTAL_MC=0`).*
Engagement **48/62 = 77%**; nvs09 **169→51 ms/node** (3.3×), node count 19 and dual
bound/objective −43.13433691803531 **byte-identical to optimality** with the fast
path both ON and OFF; ex1226 5 nodes / obj byte-identical, 3.6→2.4 ms/node; bchoco06
declines engagement (cold path unchanged). `relaxation_fingerprint` byte-identical on
all 62 (no build path changed). The 51 ms (vs the plan's ≤12 ms target) is the direct
consequence of the falsification: `_patch` still pays the ~12 ms EP1-cached build, so
the win is separation-skip + warm-start, not a 0.1 ms refresh.

*Follow-up (honestly out of EP3's soundly-deliverable scope).* A genuinely cheaper
patch would require either (i) an engine that emits its box-dependent coefficients in
a closed form it *also* uses to build (so the closed form is byte-identical by
definition), or (ii) accepting a *non*-byte-identical (differential-regime) fast
path. Both are larger than EP3. The 14 non-engaging instances are unbounded-root or
box-unstable-layout (log-space/composite columns that appear only on a strict
sub-box) — closing them needs the layout-stability analysis or the differential
regime, a follow-up.

---

## EP4a — Multilinear facet cache (exact, bound-neutral)

**Goal.** Stop re-deriving multilinear envelope facets per node: 132 tiny
facet LPs / 10 nodes (11 ms/node) in `_separate_multilinear` →
`multilinear_separation.py::_solve_envelope`.

**Scope.** `python/discopt/_jax/multilinear_separation.py`. Cache facets per
`(sorted var cols, support-box bytes)` in a capped module dict — the facet set
is a pure function of the atom and its box. For arity ≤ 3, add the closed-form
facets (standard trilinear hull formulae) guarded by an equality check against
the LP-derived facets on a sample (construction-time self-check, like the
incremental validator), falling back to the LP derivation on mismatch.

**Verification (bound-neutral gate).** Same facets ⇒ same cuts ⇒ node counts +
objectives exactly unchanged (certifying set); fingerprints untouched (root
build unchanged); a unit test asserting cache-hit facets == freshly derived
facets on 20 random boxes. Suites per §2.

**Done.** ms/node contribution of `_separate_multilinear` in the EP0 profile
drops to ~0 on cache-hot nodes; §3 updated.

**OUTCOME (2026-07-13): exact cache landed, byte-neutral; measured wall win
NEGLIGIBLE on the certifying instances — honest §0.3 report, no manufactured
win. Closed-form arity≤3 correctly NOT added.**

*Key correction to the plan's premise.* EP4a assumed "the facet set is a pure
function of the atom and its box." The *full hull* is; but
`_solve_envelope` returns the **single supporting facet selected at the query
point `x_star`**, which is NOT box-only — measured on nvs09, the returned
`(a, b)` varies with `x_star` on **10 of 16** boxes queried at multiple points.
A box-only cache key would therefore return a stale facet for a different
`x_star` at the same box — a byte-neutrality (correctness) bug. So the cache is
keyed by the **exact float64 bytes of all of `_separate_multilinear_envelope`'s
inputs** (`lb, ub, x_star, w_star, tol, max_factors`) — a pure-function
memoization, byte-identical by construction. This is the safe cache the plan's
own fallback clause mandates ("if the closed form is at all uncertain, land only
the exact cache").

*Closed-form arity≤3 skipped (correctly).* For the same reason, no box-only
closed-form facet can reproduce the `x_star`-selected LP facet (with its
vertex-recomputed intercept) bit-for-bit, so the construction-time self-check
would fail by definition. Adding it would be dead code; skipped per the
directive.

*Measured (nvs09, in-container).* The cache is exact and removes real
redundancy — **144/360 = 40% of the facet-LP solves** are eliminated (180 sep
calls, 108 distinct inputs → 72 hits). But the recurring atoms are the *cheap*
LPs: cumulative `_solve_envelope` time drops only ~15.1 → 14.6 s (~3%), within
run-to-run noise at the 45 s solve, and the EP0 ms/node metric (which solves
distinct cold child boxes with no recurrence) is unchanged (53.0 → 52.7). The
post-EP1/EP3 facet LPs are simply already cheap where they recur; the large
per-node separation cost is `_separate_convex`'s JAX-grad Kelley loop (EP4b/EP5),
not the multilinear hull LP. **Node counts, objectives, and fingerprints are
byte-identical**, so the cache is landed as a guaranteed-neutral, harmless,
bounded (200k, clear-on-overflow) removal of redundant work — not for a wall win
that the measurement does not support.

---

## EP4b — Separation warm-start + OA cut pool (bound-changing-in-path)

**Goal.** Make the `_separate_convex` Kelley loop incremental: warm-start each
round's LP from the previous basis; append rows without a full matrix rebuild
(`_append` currently `np.vstack`/`sp.vstack`s the whole matrix per round);
pool OA cuts for child inheritance the way RLT/PSD root cuts already are.

**Class.** Warm starts and pooled cuts can change which (equally valid) vertex
the LP returns and therefore the cut sequence and tree path — sound but not
byte-identical. This item runs the **bound-changing regime**: differential
bound test (per-node bound ≥ the cold-loop bound on fixed boxes, never above
the true box optimum on the audit set), feasible-point sampling (0 cuts),
end-to-end `incorrect_count = 0` on the smoke certifying set; land default-on
only if node counts on the perf panel do not regress — otherwise behind the
existing separation options, and surface it.

**Scope.** `mccormick_lp.py` (`_separate_convex`, `_append`, pool plumbing —
mirror the RLT/PSD pool mechanism).

**Done.** Per-round separation cost measured before/after; §3 updated;
differential + feasible-point results recorded.

---

## EP5 — Lazy and shared JAX compiles (setup cost, CC5)

**Goal.** After EP1, compiles happen once per model; EP5 makes them happen
only when *needed*: `grad_fn` is only consumed when `_separate_convex`
actually separates that spec — defer `jax.grad`/trace until first use
(callable wrapper), share compiled fns across structurally identical CNodes
(the DAG is hash-consed — identical structure ⇒ same CNode ⇒ already shared;
verify and note). Kill the remaining ctor double-build cost: the
`IncrementalMcCormickLP` constructor runs `_full_build` twice to validate;
after EP3 this validates the patch table (needed); ensure it reuses the EP1
cache so the second build is cheap, and skip the *second* build when the first
already produced the row descriptors needed for validation.

**Scope.** `uniform_relax.py` (lazy grad wrapper), `incremental_mccormick.py`
(single-build validation), no API changes.

**Verification (bound-neutral gate).** Fingerprints + node counts +
objectives exactly unchanged; the lazy wrapper must be exception-transparent
(a compile failure at first use degrades to skipping that spec's separation —
the same sound no-op as today's construction-time failure). Report ctor +
root-solve time before/after on the EP0 set.

**Done.** In-container nvs09 ctor+root < 2 s (from ~16 s); §3 updated.

---

## EP6 — Cash in the tree: probing/OBBT budgets + default-on (local host)

**Goal.** With per-node cost cut (EP1–EP5), the expensive-node×small-tree
economics flip: tune `DISCOPT_NODE_PROBING` / `probe_max_vars` /
`in_tree_presolve_stride` / `_PER_NODE_OBBT_ROUNDS` against the perf panel and
decide default-on for per-node probing per CLAUDE.md §5 (default-off until
green on consecutive nightly runs).

**Class.** Bound-changing (tree-changing). **Local-host item**: needs
`global50` + `cert0` (`incorrect_count = 0` hard gate), SGM/wall comparison,
and the nightly cadence. In-container prep: a sweep script over the budget
knobs on the vendored corpus reporting nodes/wall per setting.

**Done.** Maintainer-ratified default flip (or a recorded decision not to),
with panel numbers in §3 and the ledger.
