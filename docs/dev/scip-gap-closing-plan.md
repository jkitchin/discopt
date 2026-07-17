# Closing the SCIP gap on integer-product MINLPs — a data-driven development plan

**Status:** plan (not started). Builds on the merged opt-in LP-node engine
(`solve(lp_spatial=True)`, PR #290) and the diagnosis in
`scip-gap-nvs-diagnosis.md`.

---

## 0. Reassessment against the live issues 280–287 (priority correction)

The sections below (§1–§5) optimize the **nvs17/19/24 dense-integer-product
SCIP-speed gap**. Checked against the actual open issues, that target is the wrong
anchor: nvs is issue **#283, which is CLOSED**. The structure of the *live* issues
(measured — `classify_nonlinear_terms` + the engine's scope test + reproduction):

| issue | family / class | integer-bilinear (c-MIR target)? | engine in-scope? | real need |
|---|---|---|---|---|
| **#280** | graphpart_* binary-QP (240 int-bilinear, 96 int) | **yes** | yes | global-search gap (≫ nvs17 size) |
| **#281** | smallinvDAX_* MIQP (+1 continuous) | present | **no** | **incumbent polish / gap-tol**, not search |
| **#282** | syn_*/rsyn_* MINLP (0 bilinear, general-NL, max) — **mixed convex (rsyn*/syn40m → NLP-BB) + nonconvex (`*hfsg` → spatial McCormick)** | **no** | **no** | **root dual bound** (4×–500× looser than SCIP; dual-dominated 7/7) — *not* a primal/search gap; see §R2 |
| **#287** | kall_*/graphpart_* (kall: 0 integers, continuous) | **no** | **no** | **incumbent latency** regression |

So the §1–§5 priorities **do not map onto 280–287**:
- **Priority 1 (native c-MIR for integer-bilinear)** is relevant to **only #280**.
  #282 has *no* bilinear; #287's kall is *continuous*; #281 is polish, not search.
- **Priority 2 (native node loop)** is throughput on an **opt-in** engine that 3 of
  the 4 live issues don't even use.
- **#287 reproduces on `main`'s *default* path** (kall first incumbent 10.9 s vs the
  prior ~6 s, soft 8 s limit overrun) — a **primal-latency + time-limit** regression
  (same class as the #291 deadline fix), untouched by making the engine opt-in.

**Corrected priority order for resolving 280–287:**
1. **#287 — incumbent latency + honor the soft `time_limit`** (default path). Live,
   broad (kall/graphpart/tln4/flay04h → `hard_timeout`, no incumbent), cheap. Fix:
   surface the root/dive incumbent early; check time between per-node phases.
2. **#281 — incumbent polish** (terminal NLP polish on the best incumbent +
   absolute-gap tolerance for near-zero optima). The cheapest win per the issue.
3. **#280 — integer-bilinear binary-QP search gap.** *Here* the c-MIR/engine work of
   §1–§5 genuinely applies — but re-aimed at **graphpart** (96 vars / 240 products),
   not nvs17, and the §1.5 gate (do the cuts reduce nodes?) must be validated **on
   graphpart**. Higher-risk/effort; after the cheap primal wins above.
4. **#282 — general-NL MINLP (syn/rsyn) search gap.** c-MIR irrelevant (0 bilinear).
   *Measurement update 2026-07-17 **round 2**, on the real corpus
   (`issue-282-syn-rsyn-diagnosis-2026-07-17.md` §R2):* **#282 is a root-relaxation problem,
   not a search or primal problem.** Attribution over all seven named instances against the
   `=opt=` oracle: **dual-dominated 7/7** at 60 s, and on `syn15m02hfsg`/`syn30hfsg` the
   incumbent is *already the proven optimum* — only the bound is missing. discopt's **root
   bound is 4×–500× looser than SCIP's on all seven** (e.g. `syn40m` +2609 % vs SCIP +5.2 %),
   while SCIP proves the whole panel optimal in 0.5–1.6 s.
   Two round-1 claims are **falsified**: (a) "the family is convex" — it is not, `syn15m02hfsg`
   /`syn30hfsg`/`syn40hfsg` are **nonconvex** and run **spatial McCormick** (round 1
   generalized from `syn05m`, the only vendored sibling); (b) consequently "McCormick
   strengthening is not the lever" — it *is* back on the table for the nonconvex `hfsg`
   subfamily, exactly where the root bound is worst. Primal/LNS (#276) is **not** the lever
   (the incumbent is the healthy half); #268 throughput is a secondary multiplier.
   Re-pointed plan in §R2-6 of the diagnosis doc. F-1 (no OA auto-routing) still stands.

**Sequencing:** the LP-node engine has been shedding regressions (#286, #291, #287);
the primal-side wins (#287, #281) should land before any multi-week native cut
separator, which is one track (for #280) among several — not the headline.

The §1–§5 evidence (per-node profile, SCIP ablations, the cut gate) remains valid
and is retained below as the substantiation for the #280 track specifically.

---

**Problem.** The LP-node spatial branch-and-cut engine *closes* nvs17 to proven
optimality (78 s) but is **~90× slower than SCIP** (0.88 s) and does **not** close
the larger siblings nvs19/nvs24 within 300 s. This plan identifies, with measured
evidence, exactly which development closes that gap — and, just as importantly,
which work would **not** help and is therefore out of scope.

Every workstream below is justified by a measurement, names the experiment that
will validate it, and states a quantitative success criterion. No speculative work.

---

## 1. Evidence base (measured)

### 1.1 Per-node cost is the LP solve, via a Python→Rust round-trip
cProfile of a 20 s nvs17 `lp_spatial` solve (9,551 nodes, ~1.9 ms/node):

| component | time | share |
|---|---|---|
| `discopt._rust.solve_lp_warm_py` (the LP solve) | 16.4 s | **91%** |
| `_patch` (incremental McCormick row update) | 0.80 s | 4.5% |
| everything else (assemble, aux bounds, child mgmt) | ~0.7 s | ~4% |

The relaxation **build** is already cheap (Step 2's incremental patch). The cost is
the **LP solve itself** — ~1.7 ms for a 35-column / ~112-row LP, against SCIP's
~µs. Each node crosses Python→Rust and re-marshals the full dense `A`, `b`, bounds.

### 1.2 Cuts are the dominant lever, and it grows with problem size
SCIP node counts, default vs ablations (120 s limit), measured this session:

| instance | default | **no cuts** | cut speedup | no presolve | no strong-branch |
|---|---|---|---|---|---|
| nvs17 | 70 | 6,796 | **97×** | 70 | 78 |
| nvs19 | 122 | 16,242 | **133×** | 122 | 143 |
| nvs24 | 468 | 79,104 | **169×** | 468 | 530 |

- **Cuts dominate**: 97–169× fewer nodes, and the multiplier *grows* with size.
- **Presolve is irrelevant** here (identical node counts).
- **Strong/reliability branching barely matters** (+10–15% nodes without it).

### 1.3 The cut that does the work is complemented-MIR (aggregation)
SCIP separator statistics (applied cuts):

| instance | aggregation (c-MIR) | gomory | zerohalf | RLT |
|---|---|---|---|---|
| nvs17 | 15 applied | 1 | 5 found | **0** |
| nvs19 | 6 | 6 | 4 found | **0** |
| nvs24 | **22 applied** | 41 found | 7 found | **0** |

`aggregation` (multi-row complemented MIR, Marchand–Wolsey) is the workhorse;
gomory is secondary; **RLT contributes nothing** on this family. On nvs24 the
separator is called 379× (201 at the root, ~178 in the tree) — so it is applied
**per node**, not only at the root.

### 1.4 discopt's current cuts are both slow and weak
Measured on nvs17's McCormick LP this session:
- Python GMI (from the optimal basis) and the Python complemented-MIR plateau:
  root bound moves only −27,795 → −27,291 (~2%), 25× loose, then stalls.
- Per-node separation (crossover + GMI binding + c-MIR) costs **~seconds/node** and
  crashed throughput (31 nodes vs 5,928) — net-**negative** at every setting.
- Root-only inherited cuts also net-negative (LP-row bloat slows every node).

So the existing cut path fails on **two** axes — too slow (Python crossover/binding)
*and* too weak (modest root tightening). SCIP's aggregation is fast (native) *and*
strong (closes in 70–468 nodes).

### 1.5 discopt's current cuts do NOT improve node-efficiency (gate result)
The §1.4 numbers are root-bound tightening; the decisive test is node *count*.
Node-limited experiment (nvs17, equal 1,500-node budget, global bound compared so
LP-solve speed is factored out):

| config | bound @ 1,500 nodes |
|---|---|
| cuts OFF | −2,280.9 |
| root c-MIR/GMI ON (`lp_spatial_cut_rounds=20`) | **−2,564.0 (worse)** |

Adding discopt's current cuts makes the bound *looser* at equal node count. So the
existing cut family is not merely slow (§1.4) — it does **not** deliver node
reduction at all, the opposite of SCIP's 97–169×. **A pure "port the existing cuts
to Rust" approach is therefore ruled out by measurement.** The native separator
must be genuinely *stronger* (match SCIP's aggregation quality), not just faster —
this is the central risk Phase 0/1 below now front-load.

---

## 2. What this implies (and what to NOT build)

The arithmetic that frames everything: with SCIP-grade cuts the family solves in
**70–468 nodes**. At discopt's *current* 1.7 ms/node that is **0.1–0.8 s** — i.e.
**cuts alone would close nvs17/19/24 in ~1 s without any throughput work.** The
node-throughput gap only bites in the *no-cut* regime (79k nodes × 1.7 ms = 134 s,
which is exactly why nvs24 times out today).

Therefore:
- **Priority 1: a *stronger* native cut separator** (complemented-MIR/aggregation)
  — the dominant, size-scaling lever. Note the §1.5 gate: this is **not** a port of
  the existing cuts (those don't reduce nodes) — it requires matching SCIP's
  aggregation *quality*, so a separation-quality track precedes/accompanies the
  Rust implementation.
- **Priority 2: native node loop / faster LP solve** — turns "within ~10×" into
  "within ~3×"; not required to *close* the family once cuts work.
- **De-scoped: branching and presolve.** Evidence (§1.2) shows ≤15% / 0% effect.
  Pseudocost branching (already in the engine) is sufficient. Do **not** invest in
  reliability branching, strong branching, or presolve for this class.

---

## 3. Phased plan

### Phase 0 — Gate: do the cuts reduce node count? (DONE — outcome below)
**Result (§1.5):** discopt's current cuts make the bound *worse* at equal node
count. **Go/no-go: NO-GO for a pure Rust port** of the existing separators — they
lack the strength. Phase 1 is therefore re-scoped to cut *strength*, and gated by
one more isolating experiment:

**Phase 0b — isolate strength vs separator-quality (2–3 days):** extract the cuts
SCIP generates on nvs17 (`SCIP> write … cuts`, or its log) and add them directly to
discopt's McCormick LP; measure node reduction in the engine.
- SCIP's cuts *do* reduce discopt's nodes → discopt's *relaxation* is sound; the
  gap is purely separator **quality** → build a faithful c-MIR (Phase 1).
- SCIP's cuts *don't* help discopt's nodes → the relaxation/branching interaction
  is implicated → widen Phase 1 to the lifted formulation (extra aux structure)
  before separation.
This pins down exactly what Phase 1 must deliver before any Rust is written.

### Phase 1 — Native complemented-MIR / aggregation separator (2–4 weeks)
**Hypothesis:** a fast, strong c-MIR separator delivers SCIP's 97–169× node
reduction; at current node speed that closes nvs17/19/24 in ~1 s.
**Evidence:** §1.2 (cuts dominate), §1.3 (aggregation is the workhorse, per-node).
**Build:**
- Implement complemented-MIR (Marchand–Wolsey) in Rust under
  `crates/discopt-core/src/lp/` next to the existing `gomory.rs`/`mir.rs`:
  tableau-based row aggregation (B⁻¹ combinations), bound complementation, the
  δ-scan, the MIR function; numerically safe (coefficient snapping + rhs margin,
  as the existing GMI does).
- Mark the **product aux columns integer** (`w_ij = x_i x_j` is integer-valued) —
  the established prerequisite for the fractional `w` to be a cut target.
- Integrate into the node loop (root rounds + a periodic in-tree cadence matching
  SCIP's ~half-non-root call ratio), with cuts inherited down each subtree (valid,
  per the diagnosis).
**Measurement:** node count + time on nvs17/19/24 vs SCIP; **soundness gates** —
dual bound never exceeds the true optimum, no sampled feasible point is cut.
**Success:** nvs17/19/24 **close**; node counts within ~3× of SCIP; wall-clock
within ~10× of SCIP (≈ a few seconds).

### Phase 2 — Native node loop + warm LP (3–6 weeks)
**Hypothesis:** moving the B&B node loop into Rust — LP structure resident, rows
patched in place, basis persisted across parent→child, no per-node PyO3 marshaling
— cuts the 1.7 ms/node by ~10×.
**Evidence:** §1.1 (91% in `solve_lp_warm_py`, full-matrix re-marshaling per node).
**Build:** a Rust spatial-B&B driver over the McCormick-extended LP that reuses the
existing simplex with a persistent basis and in-place incremental row updates;
single PyO3 entry. Reuse the existing tree-manager infrastructure where possible.
**Measurement:** ms/node and nodes/s vs SCIP.
**Success:** <0.2 ms/node; combined with Phase 1, total time within ~3× of SCIP.

### Phase 3 — Primal for the nvs24 class (1–2 weeks)
**Hypothesis:** nvs24 finds *no* incumbent via dive/feasibility-pump because
feasibility is hard; SCIP gets one immediately (and the default discopt path finds
−1026 via its NLP solves).
**Evidence:** nvs24 `obj=None` at 300 s; nvs19 22% gap; default NLP path finds a
feasible point.
**Build:** a stronger primal for this class — NLP-polish of rounded points (reuse
the existing node-NLP), or a RENS-style sub-MIP on the cut-tightened relaxation.
**Measurement:** nodes-to-first-incumbent.
**Success:** an incumbent on nvs24 well before the node budget; reported gap closes.

---

## 4. Standing benchmark harness (evidence for every phase)
Build a small reproducible harness (extends `discopt_benchmarks`): nvs10–nvs24 +
the ex126x family + carton7, each run through the engine and SCIP, recording
**nodes, wall-time, gap-over-time, root bound, incumbent-over-time**. Every phase is
gated on it; regressions (correctness or node count) block the phase. This keeps the
effort data-driven end to end and prevents the "lever that doesn't pan out" pattern
documented in the diagnosis.

## 5. Risks
- **Cut strength (CONFIRMED, the central risk).** §1.5 measured that discopt's
  current cuts do not reduce nodes. Phase 1 is a separation-*quality* effort
  (matching SCIP's aggregation), de-risked first by Phase 0b (do SCIP's own cuts
  reduce discopt's nodes?). This is harder than a port and is the schedule driver.
- **Soundness:** native cuts must never cut a feasible point — enforced by the
  bound≤opt monitor and feasible-point sampling on every instance (the existing
  c-MIR already validates bound≤opt).
- **Scope creep into branching/presolve:** explicitly resisted — §1.2 shows they
  are not the lever for this class (≤15% / 0%).
- **Throughput may not even be needed first:** §2 shows cuts alone (70–468 nodes ×
  1.7 ms) would close the family in ~1 s, so Phase 2 (native loop) is sequenced
  *after* Phase 1 and only if the within-3× target requires it.
