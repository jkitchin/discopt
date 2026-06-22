# Why discopt loses to SCIP on the nvs17/19/24 family — a data-backed diagnosis

**Status:** diagnosis complete; implementation plan proposed (not started).
**Scope:** dense all-integer polynomial MINLPs (nvs17/19/24: 7–10 integer vars in
`[0,200]`, every pair multiplied — `C(n,2)` bilinear terms + integer powers).

SCIP solves these in **70–468 nodes, ~1–2 s**. discopt **times out** with a
feasible-but-suboptimal incumbent. This note records what was *measured* (not
assumed) and corrects three hypotheses that the data falsified.

## Measurements (nvs17, the smallest)

| experiment | result | source |
|---|---|---|
| discopt full solve, 30 s | feasible −1091.4 (true −1100.4), **bound −65,842**, **31 nodes**, **0.9 nodes/s** | `solve()` |
| discopt dual bound trajectory | **frozen at −65,842** (root value) across all 31 nodes | `solve()` |
| Root OBBT *called directly* | [0,200] → [0,~40], 35 tightenings, 0.9 s | `obbt_tighten_root` |
| Root OBBT *in the solver* | **never runs** — gated off for pure-integer models (`solver.py:3312`, `_obbt_has_continuous`) | code |
| Solve with OBBT box applied | bound −65,842 → **−6,790** (10× better, still 6× loose), still times out | experiment |
| Cutoff-OBBT w/ optimal incumbent | box barely moves ([0,46]→[0,43]) — McCormick is the wall, not bounds | experiment |
| SCIP default | 70 nodes, 0.88 s | scip |
| SCIP **no cuts** (`separating/maxrounds 0`) | **6,796 nodes, 4.72 s — still solves** | scip |
| SCIP no presolve | 70 nodes, 0.97 s (presolve irrelevant) | scip |
| SCIP separators that fire | **aggregation (MIR)** 15–22 applied, **gomory** 1–6, zerohalf; **RLT = 0** | scip stats |
| discopt profile (12 s slice) | **3 B&B nodes**; time in POUNCE NLP solves (6.3 s), NLP primal heuristics (feasibility_pump 3.8 s + integer_local_search 2.0 s + subnlp 1.8 s), JAX trace/Hessian (~5 s) | cProfile |

## Three hypotheses the data **falsified**

1. **"OBBT isn't contracting the boxes."** False — OBBT contracts [0,200]→[0,40]
   fine; it's just *gated off* for pure-integer models. And applying it only gets
   the bound to 6× loose — necessary, not sufficient.
2. **"RLT cuts are SCIP's edge."** False — SCIP's RLT separator finds **0 cuts**
   here. The work is done by MIR (`aggregation`) + Gomory.
3. **"Cuts are the binding constraint."** Partly false — cuts give a real **97×
   node reduction** (6,796→70), but **no-cut SCIP still solves in 4.7 s** while
   discopt cannot. Cuts are a strong multiplier on top of a fast engine discopt
   does not have.

## The actual root cause (three compounding factors)

1. **Node throughput: 0.9 nodes/s vs SCIP's ~1,400 (no-cut).** discopt solves a
   **continuous NLP relaxation per node** (POUNCE IPM + JAX Hessians, ~0.2 s) and
   burns most of the budget in **NLP-based primal heuristics**. SCIP solves a
   **pure LP** per node (~µs). For all-integer polynomial problems the per-node
   NLP and the NLP heuristics are pure waste — the McCormick **LP** + integer
   branching suffices.
2. **Dual bound frozen.** Bisecting a 7-D `[0,200]` box 31 times barely dents an
   envelope this loose, so the global bound never leaves the root value.
3. **Loose relaxation, no integer cuts in this path.** discopt *has* Gomory/MIR
   generators, but only on the **pure-MILP root LP** — never on the McCormick
   relaxation of a nonconvex model.

No single cheap fix closes the gap: OBBT alone → 6× loose; cuts alone → useless at
0.9 nodes/s; throughput alone → loose McCormick won't close without cuts/OBBT.
They are multiplicative and must be addressed together.

## What SCIP does (and discopt lacks): LP-based spatial branch-and-cut

SCIP keeps **one aux var per product** (McCormick envelope, polynomial size — it
does **not** binary-expand), solves a **pure LP** at each node, branches on the
integer variables (driving products exact at the leaves), and separates **MIR +
Gomory** cuts on that extended LP. discopt has the two *halves* but not the
combination:

- **LP-based MILP branch-and-cut** (fast Rust simplex + Gomory/MIR/cover cut loop)
  — but only for problems that are *already* linear (or binary-expanded, which
  blows up: nvs17 7→2,751 vars).
- **NLP-based spatial B&B** (McCormick lift + spatial branching) — handles
  nonconvexity but at 0.9 nodes/s with a frozen bound.

The missing engine is **LP-based spatial branch-and-cut**: McCormick LP per node
(no NLP), integer/spatial branching, MIR/Gomory cuts on the extended LP, OBBT at
the root including pure-integer models.

## Proposed plan (phased, each phase independently verifiable)

**Phase 0 — de-risk. DONE (positive).** A standalone LP-node spatial B&B over the
McCormick relaxation (`build_milp_relaxation` per box, integer-fractional + product-
tightness spatial branching, rounding heuristic verified via collapsed-box LP — no
NLP anywhere) was prototyped and measured:

| | discopt today | LP-node prototype (30 s) |
|---|---|---|
| nvs17 | bound frozen −65,842, inc −1091, 0.9 nodes/s | bound **−1,247**, inc **−1,085**, ~37 LP/s |
| nvs19 | frozen, inc −1092 | bound −1,700, inc −1,036 |

Conclusions: (a) the LP-node architecture **unfreezes the dual bound** (−65,842 →
≈−1,250 vs true −1,100) and finds near-optimal primals — the central hypothesis
holds. (b) It does **not yet close**: the gap is the two already-proven levers —
MIR/Gomory cuts (97× node reduction) and node throughput. (c) Throughput is bounded
by **rebuilding the relaxation each node** (~27 ms, JAX-trace-dominated), NOT the LP
solve (~0.4 ms). A real implementation must update rows/bounds incrementally
(`MilpRelaxationModel` already carries a warm basis) rather than calling
`build_milp_relaxation` per node.

**Phase 1 — un-gate OBBT for pure-integer models** (cheap, but only as part of the
whole). Verify: nvs17 root bound −65,842 → −6,790.

**Phase 2 — route integer-product MINLPs to an LP-node spatial B&B.** Reuse the
MILP branch-and-cut tree; swap binary-expansion for the McCormick LP relaxation
with per-node envelope refinement. Verify value-equivalence on the existing
integer-bilinear test basket + ex126x + nvs14.

**Phase 3 — MIR/Gomory on the McCormick extended LP** (the 97× multiplier).
Verify node-count drop on nvs17/19/24 toward SCIP's 70–468.

**Non-goal:** matching SCIP's raw LP speed (~µs). Target is *solving* the family
in seconds, not microsecond LP throughput.
