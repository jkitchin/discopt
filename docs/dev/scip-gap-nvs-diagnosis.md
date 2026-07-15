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

## Step 3 findings (cuts on the McCormick LP) — measured

Step 1 landed (`_jax/lp_spatial_bb.py`); Step 3 (cuts) was investigated next because
the SCIP ablation showed cuts are the *closing* lever (70 vs 6,796 nodes). Two
concrete discoveries on nvs17, both data-backed:

1. **The fractionality is in the product vars, not the originals.** The McCormick
   LP optimum has *integral* original variables `x` (30/35 columns integral); the
   looseness lives in the continuous product columns `w_ij`. So Gomory/MIR keyed on
   the original integers find nothing. Fix: mark the product aux columns **integer**
   (`w_ij = x_i*x_j` is integer-valued when the factors are — a sound implied
   integrality). Then `w_ij`'s fractional envelope value becomes a cut target and
   MIR begins to separate.

2. **discopt's single-row MIR/Gomory don't tighten the bound here.** With a
   crossover vertex + aux integrality, MIR separates a cut that is valid and cuts
   off the *vertex* (16.5 > 16) but **not the LP optimum** (16.0 = 16.0). The
   McCormick LP has a large optimal face; trimming a vertex leaves the optimal
   *value* unchanged, so the bound does not move. This matches SCIP's separator
   stats exactly: the work there is done by the **`aggregation`** separator
   (multi-row *complemented* MIR), not plain Gomory — and discopt has **no
   row-aggregation MIR**. Gomory's basis recovery on the augmented McCormick LP is
   also finicky (returns no cut).

Extending the test: feeding GMI a *proper simplex basic vertex* (not a crossover
point) does make it separate (9 cuts), but a full GMI+MIR root loop **plateaus** —
the bound moves −27,795 → −27,291 in two rounds then stalls (still 25× loose, ~2%
total). So even the textbook bound-improving cut (GMI from the optimal basis) is too
weak on this McCormick LP; discopt genuinely lacks SCIP's `aggregation` (multi-row
complemented MIR).

**Synthesis / corrected priority.** The cut path is a dead end *with discopt's
current separators*. But the SCIP ablation already showed **no-cut SCIP closes
nvs17 in 6,796 nodes via branching alone** (4.7 s). The Step-1 LP-node engine is
exactly that no-cut analog and already reached bound −1,157 in 652 nodes — it simply
can't reach ~6,800 nodes at ~92 ms/node. Therefore the correct, general, performant
closer is **node throughput** (Step 2: incremental warm-started LP nodes →
~1–2 ms/node → tens of thousands of nodes → close by branching, matching no-cut
SCIP), **not** cuts. Cuts (a future c-MIR/aggregation separator) are a node-count
optimization (6,796 → 70), valuable later but neither necessary nor currently
attainable. **Step 2 is therefore the real next step; Step 3 is deferred** until a
proper aggregation-MIR separator exists.

## G3 / TX4 update (2026-07-15) — routing FALSIFIED; false-optimal fixed instead

Entry experiments for baron-gap-plan G3 (route family C to the LP-node engine by
default) measured on `main` @`0f3ebd7d` (post-#636 uniform engine). Two findings
overrode the plan.

**1. The default NLP path already solves the family (#636 superseded the freeze).**
The frozen-bound pathology this diagnosis documented (nvs17 stuck at −65,842) is
gone on the default path — the uniform engine + already-un-gated root OBBT
(`solver.py:4711`, `_obbt_has_nonlinear and n_vars <= 50`) close it:

| instance | default path (60 s budget) | true opt |
|---|---|---|
| nvs17 | **optimal −1100.4**, 131 nodes, 10.5 s | −1100.4 |
| nvs19 | **optimal −1098.4**, 277 nodes, 30.2 s | −1098.4 |
| nvs24 | feasible −1031.8, bound −1034.55, gap 0.27 %, 517 nodes, 60 s | −1033.2 |

Root OBBT probe (EE3): post-#636 the root McCormick bound is −395,450 over [0,200]
→ −34,715 after OBBT (box → ~[0,49..65]). The Phase-1 prediction (−65,842 →
−6,790) was against the *pre-#636* relaxation and does not reproduce; the number
changed but the default path solves regardless, so OBBT un-gating (plan part b) is
already effective on `main`.

**2. The opt-in LP-node engine returns CERTIFIED FALSE OPTIMA at `main`** — a P0
regression, not just a throughput gap:

| instance | `solve(lp_spatial=True)` at `main` | true opt |
|---|---|---|
| nvs17 | status=**optimal** obj=**−1836.2**, gap_certified=True (infeasible point) | −1100.4 |
| nvs19 | status=**optimal** obj=**−2520.4** | −1098.4 |

Cause: #636 lifts bilinear `x_i*x_j` via **univariate squares**, so
`build_milp_relaxation`'s `info["bilinear"]` is empty for nvs17-class models. The
engine's `_worst_product_var` then never sees the (loose) products, declares "all
products tight", and accepted the loose McCormick node bound as the incumbent's
true objective. `IncrementalMcCormickLP.ok` is also False (cuts + pump disabled),
and the collapsed-box `verify()` returned None (`_objective_bound_valid` False).
The existing `test_nvs17_dual_bound_is_valid_and_tight` catches this but is
`slow`+`requires_pounce`, so CI smoke never ran it.

**Action taken (this PR).** Routing (part c) is NOT landed — it would inject false
optima into the default path, and the engine cannot beat the default even once made
sound (inc.ok broken → no throughput; McCormick bound too loose). Instead the
engine's **soundness** was fixed generally (exact ground-truth incumbent
verification; a valid dual-bound floor for nodes it cannot branch; "optimal" only
on a genuine gap closure). Post-fix the engine is sound on the family (nvs17
feasible −1092.4 / valid bound −1836.2 / honest `time_limit`, no false optimum).

**Kill criterion (met):** the engine, even with OBBT + soundness fixes, does not
beat the default path on its own family. Full engine restoration to the #636
relaxation (repopulate the product map so `inc.ok`/product-branching/cuts work
again) is a separate, larger effort and the prerequisite for any future routing.
