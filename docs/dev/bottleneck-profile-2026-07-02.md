# Bottleneck profile — 2026-07-02 (JAX / Rust–Python boundary / orchestration)

**Date:** 2026-07-02
**Status:** measured (fresh 3-probe pass on branch `cert-phase0`)
**Scope:** the certification loop's *runtime* cost — per-node LP engine, the
Rust–Python boundary, and orchestration (OBBT, structure rebuild, certificate
re-derivation). This is a measurement record, not a plan; the actionable
follow-ups are relocated into `docs/dev/certification-gap-plan.md` (Phase 1/2
tasks). Correctness invariants are untouched by anything here.
**Referenced by:** `docs/dev/certification-gap-plan.md` §1 (the dated correction
note), and tracking issue #397.

> **Method.** Three probes on the spatial/LP-bound panel (gear4, ex1252, kall,
> nvs17, and the `lp_spatial` path), using the T0.3 honest reduction/separation
> timers (Rust FBBT/NodeReduce stderr profile + Python per-family timers on
> `solver_stats`) rather than the `rust_time`/`python_time` residual split that
> the June draft leaned on. Raw artifacts (per-instance `PerfRecord`s, cProfile
> pstats, microbench scripts) were produced in the session scratchpad; only the
> durable numbers below are retained. Where a June claim is superseded, §4
> records the correction rather than silently overwriting it.

---

## 1. Ranked bottlenecks (measured)

In measured order of leverage:

1. **OBBT inner loop** (spatial class). 23 LP solves/node on gear4, each
   reallocating scipy sparse structure (≈33% of wall is scipy/numpy churn), plus
   +2.4 extra `build_milp_relaxation` calls/node. Ruiz scaling is recomputed
   (~10 ms × 1,127). This is the dominant orchestration cost on the spatial
   panel.
2. **Cold simplex refactorizations** (`lp_spatial` path). The Rust LP call is
   **68% of wall** at ~1,840 µs average vs ~100 µs warm; warm starts are
   frequently rejected on lifted coefficients, and 3.2 solves/node are
   re-marshaled. The `lp_spatial` engine runs at **8.6 ms/node** (not 1.9),
   dominated by these cold refactorizations.
3. **Python-side certificate re-derivation** around every LP call
   (Neumaier–Shcherbina safe bound + Farkas verdict), ≈18% of `lp_spatial` wall.
   Rust already holds the row duals / dual ray and could return the safe bound +
   Farkas verdict directly, eliminating the Python re-derivation.
4. **Root-tightening mis-attribution.** `python_time` is a *residual* that
   absorbs root OBBT's JAX work and Rust presolve; it is not an honest layer
   timer. Do **not** gate on the layer-fraction fields (`rust_time` /
   `python_time`) until the T0.3 honest timers are used instead.
5. **Per-node JAX evaluation dispatch** (NLP-bound class). Up to 1.46 s/node on
   `kall`; the batch evaluator never actually ran on the profiled instances.
6. **Per-call structure rebuild.** scipy csr→csc→`hstack(I)` per LP call (≈12%)
   despite identical sparsity across all nodes.

---

## 2. Not bottlenecks (verified — do not spend time here)

- **PyO3 / FFI boundary:** ~2 µs/call floor. Not a lever.
- **XLA compilation:** resolved (≤1% of wall). The evaluator-cache fix landed;
  bounds are traced arguments, so there are **0 recompiles across changing
  boxes**. Phase 5's compile-caching item is done.

---

## 3. Follow-ups (relocated into the certification-gap plan)

- **Items 1, 2, 3, 6** are all served by the certification-gap-plan Phase 1
  mechanism (a persistent, bound-patchable LP held Rust-side): tasks T1.2 / T1.3
  / T1.4, plus the added Phase 1 task **"return the Neumaier–Shcherbina safe
  bound + Farkas verdict from Rust"** (serves item 3). The persistent
  bound-patchable LP serves the node bound, OBBT, and the per-call structure
  rebuild at once, which *raises* Phase 1's payoff.
  - **T1.4 target from this profile:** nvs17 average LP call **~1,840 µs →
    ≤300 µs**.
- **Item 4:** switch gates/reports to the T0.3 honest timers (now implemented)
  instead of the `rust_time` / `python_time` residuals. Until then the
  layer-fraction fields stay informational.
- **Item 5:** re-scope certification-gap-plan Phase 5 toward eval
  dispatch/batching after the post-Phase-1 re-profile (the compile-caching item
  is already done).

The C1 headline of the certification-gap plan — *per-node cost dominates the
gap; Phase 1 is the enabler* — stands under this profile.

---

## 4. Stale-claims record (corrections of the June draft)

Per the certification-gap-plan §0.4 house style, superseded claims are recorded,
not overwritten. Four June claims underlying C1 are stale:

1. **"`rust_time ≈ 0` everywhere on the spatial panel"**
   (`docs/dev/performance-plan.md` §"first draft", item 1). *Correction:* the
   layer-fraction fields are accounting artifacts — `rust_time` **excludes** the
   per-node LP binding calls and POUNCE. The actual Rust simplex compute is
   **68% of wall** on the `lp_spatial` path (§1 item 2). `python_time` is a
   residual that absorbs root OBBT's JAX work and Rust presolve (§1 item 4). Do
   not gate on these fields until the T0.3 honest timers are used.
2. **"~1.9 ms/node"** (`docs/dev/scip-gap-closing-plan.md` §1.1, the nvs17
   cProfile). *Correction:* the `lp_spatial` engine runs at **8.6 ms/node**,
   dominated by cold refactorizations from rejected warm starts (§1 item 2).
3. **"91% in dense re-marshal"** (`docs/dev/scip-gap-closing-plan.md` §1.1 / §2:
   91% in `solve_lp_warm_py`, full-matrix re-marshaling per node).
   *Correction:* the LP *solve itself* is the cost (68% of wall), not the
   marshaling; the dominant orchestration cost is now **OBBT's inner loop** (§1
   item 1). The per-call structure rebuild is a separate, smaller ≈12% (§1
   item 6).
4. **XLA compilation as an open cost.** *Correction:* resolved — ≤1% of wall,
   0 recompiles across changing boxes (§2).

These corrections are surfaced in `docs/dev/certification-gap-plan.md` §1's dated
correction note; this section is the durable measurement record behind it.
