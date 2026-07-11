# Principled Reference-Solver Guide

**Purpose.** A literature- and source-grounded map of what the winning solvers —
HiGHS (LP/MILP), SCIP (MINLP / spatial), BARON (spatial global MINLP,
branch-and-reduce) — actually do to be fast, mapped against discopt's current
state, yielding a *derived* prioritized roadmap. The maintainer's directive:
stop probing discopt instance-by-instance and guessing at levers; work from an
authoritative reference map instead.

**Grounding rule.** Every performance claim about a reference solver cites a
reference in `docs/references.bib` or a verified URL; every claim about discopt
cites a `file:symbol`/`file:line` in this repo. Citations use the
`docs/references.bib` keys (capitalized: `Ryoo1996`, `Puranik2017`, …), which are
a *different* key namespace from the Crucible KB articles (`.crucible/wiki/`,
lowercase keys). Both are cross-referenced below.

**Status:** strategy document, 2026-07-10. Consumes and reconciles with
`certification-effort-decomposition-2026-07-10.md` (DECOMP-1) and
`gap-closing-execution-plan.md` §6 (falsifications F1–F8). Companion KB articles:
`.crucible/wiki/concepts/{baron,scip,highs,maingo}-solver.org`,
`.crucible/wiki/comparisons/maingo-vs-discopt.org`.

---

## 1. Per-solver performance anatomy

### 1.1 BARON — spatial global MINLP by branch-and-reduce

BARON's product is a **rigorous global certificate** (dual bound ≤ incumbent),
and its defining algorithmic idea is *branch-and-reduce*: classical spatial
branch-and-bound augmented with a **range-reduction phase applied at every node**
{cite:p}`Ryoo1996`. Tighter variable ranges feed directly into tighter convex
relaxations, which is the virtuous cycle BARON is built around
{cite:p}`Tawarmalani2002`.

**The range-reduction "poll" (what runs, where, when).** The authoritative
description is Puranik & Sahinidis, *Domain reduction techniques for global NLP
and MINLP* {cite:p}`Puranik2017`; the marginals-based mechanism traces to
{cite:p}`Ryoo1996`. In the GAMS/BARON interface these are exposed as named,
default-on options
([GAMS/BARON docs](https://www.gams.com/latest/docs/S_BARON.html)):

| BARON option | Reduction | Default | Mechanism |
|---|---|---|---|
| `TDo` | Nonlinear feasibility-based (FBBT) | 1 (on) | interval propagation through the factorable DAG |
| `LBTTDo` | Linear feasibility-based | 1 (on) | bound propagation on the linear relaxation rows |
| `OBTTDo` | Optimality-based (OBBT) | 1 (on) | min/max each variable over the relaxation LP |
| `MDo` | Marginals-based (duality/reduced-cost) | 1 (on) | Lagrange-multiplier / reduced-cost range contraction {cite:p}`Ryoo1996` |
| `PDo` | Probing | −2 (auto) | tentative fixings + re-propagation |

The *marginals-based* reduction (`MDo`) is the Ryoo–Sahinidis contribution
{cite:p}`Ryoo1996`: from the lower-bounding LP's dual multipliers and reduced
costs, if a variable sits at a bound with a strictly signed multiplier, the
opposite bound can be contracted "for free" — no extra LP solve. This is cheaper
than OBBT (which solves 2n auxiliary LPs) and runs at every node.

**Relaxation hierarchy.** BARON constructs relaxations by the
**auxiliary-variable method** (AVM): every factorable expression is decomposed
into elementary operations, an auxiliary variable is introduced per intermediate
node, and McCormick envelopes {cite:p}`McCormick1976` + RLT rows
{cite:p}`Sherali1990` are written in the *lifted* space, plus univariate
convex/concave envelopes {cite:p}`Tawarmalani2005`. The polyhedral
branch-and-cut of {cite:p}`Tawarmalani2005` refines this with supporting
hyperplanes of the convex relaxation generated at LP solutions. RLT gives the
**exact multilinear hull** for polynomial terms. Integrality is exploited by
adding MILP relaxations to the LP/NLP portfolio {cite:p}`Kilinc2018`.

**Branching.** Spatial branching on the variable with the largest relaxation
error (violation-based), with integrality-aware rules for discrete variables
{cite:p}`Kilinc2018`.

**FINITE-DOMAIN requirement (load-bearing for §5).** BARON's global guarantee
*assumes finite lower/upper bounds on all variables and on all nonlinear
subexpressions*
([GAMS/BARON docs](https://www.gams.com/latest/docs/S_BARON.html)):
> "All nonlinear expressions … must be bounded below and/or above."

Finite variable bounds do **not** imply finite expression bounds (e.g. `1/x` on
`x∈[0,1]`). When the user omits bounds, **BARON attempts to infer them from the
problem constraints**; if inference fails it prints
*"User did not provide appropriate variable bounds"* and downgrades the claim to
*"Globality is therefore not guaranteed"* — i.e. it **returns no lower bound**
rather than a false certificate. (A commonly cited fallback is to clamp missing
bounds near ±10^10 for a numerically-constructible relaxation while dropping the
optimality claim.) The takeaway: BARON's stall/degradation mode on unbounded
problems is *exactly* discopt's `feasible`/`gap_certified=False` mode (§5).

### 1.2 SCIP — spatial B&B in a constraint-integer-programming framework

SCIP is an open-source (Apache-2.0 since 9.0) branch-and-cut platform whose
MINLP extension is described in {cite:p}`Vigerske2018` and, for version 8, in
{cite:p}`Bestuzheva2023`. Its performance rests on four pillars:

**Presolve.** A large library of reductions applied to a fixed point before
search: bound tightening, coefficient strengthening, aggregation, singleton /
implied-free elimination, clique extraction, probing, dual reductions
{cite:p}`Achterberg2020`. Presolve quality is a first-order determinant of MIP
performance, not a preprocessing afterthought.

**The nonlinear estimator framework (nlhdlrs).** SCIP 8 consolidates all
nonlinear constraints into a *single* `nonlinear` constraint handler that
dispatches relaxation/propagation/separation to specialized **nonlinear
handlers** — bilinear, convex, concave, quotient, perspective, and the default
(McCormick) handler {cite:p}`Bestuzheva2023`. Each expression is offered to all
registered nlhdlrs; the one reporting the tightest relaxation or cheapest
propagation wins. This is how SCIP *detects and exploits convex substructure at
the expression level* and beats AVM-only solvers on convex-heavy models
{cite:p}`Bestuzheva2023`.

**The cut / separation loop.** SCIP generates cuts in rounds interleaved with LP
re-solves: the full Chvátal–Gomory family (GMI, MIR, mod-k, flow-cover,
implied-bound, clique) for the integer part, plus — new in SCIP 8 — automatic
**RLT cuts** and **intersection cuts** for the nonlinear part
{cite:p}`Bestuzheva2023`. Cuts are added, the LP re-optimized, and the loop
repeats subject to efficacy/orthogonality filtering.

**OBBT scheduling.** SCIP's OBBT propagator minimizes/maximizes each variable over
the relaxation. Crucially it is **scheduled, not run everywhere**: by default
SCIP applies OBBT *at the root node* to tighten bounds globally, with a
per-round LP-iteration budget, interrupts for cheaper propagation between LP
solves, and can trigger extra separation/propagation rounds after each OBBT-LP
(`propagating/obbt/separatesol`, `propagating/obbt/propagatefreq`; verified
against the SCIP OBBT propagator documentation) {cite:p}`Gleixner2017`. FBBT
(reverse propagation through the expression graph) runs at every node
{cite:p}`Vigerske2018`. Branching is reliability/pseudocost-based
{cite:p}`Achterberg2005` for the integer part and violation-based for spatial
branching {cite:p}`Vigerske2018`.

### 1.3 HiGHS — the LP/MILP engine (discopt's #557/#606 pain)

HiGHS's defining contribution is the **parallel dual revised simplex** of
{cite:p}`Huangfu2018`. discopt's per-node bottleneck is its *own* dual simplex on
wide dense McCormick-lifted bases (#557, #606); HiGHS is the reference for the
robustness/speed techniques discopt underperforms. The numerics that matter
(Huangfu & Hall, and the HiGHS `HEkkDual*` implementation
[ERGO-Code/HiGHS](https://github.com/ERGO-Code/HiGHS)):

1. **Dual steepest-edge (DSE) pricing** for CHUZR (row/leaving-variable
   selection) — the single largest robustness lever; it selects the leaving
   variable by true steepest-edge weight, drastically reducing iteration counts
   versus Dantzig pricing {cite:p}`Huangfu2018`.
2. **Bound-flipping ratio test (BFRT)** for CHUZC (entering-variable / column
   selection) — a long-step ratio test that flips bounded variables across the
   ratio-test breakpoints in one pivot, reducing degenerate pivots
   {cite:p}`Huangfu2018`.
3. **Hypersparse FTRAN/BTRAN** — exploits the sparsity of RHS/price vectors after
   update so solves against the basis inverse touch only the nonzero support
   {cite:p}`Huangfu2018`.
4. **Forrest–Tomlin (and product-form) basis updates** with the novel
   single-precision-stable variants of {cite:p}`Huangfu2018`, plus
   refactorization triggers that bound accumulated error.
5. **Presolve** shared with the MIP layer (dominated cols, singleton rows,
   doubleton eq, forcing constraints) {cite:p}`Achterberg2020`.
6. **MIP engine:** reliability pseudocost branching {cite:p}`Achterberg2005`,
   the full cut family, and a primal-heuristic portfolio (feasibility jump,
   RENS/RINS, feasibility pump, ZI-round) {cite:p}`Achterberg2020`.

The **degeneracy + refactorization churn** DECOMP-1 measured on discopt's
in-house engine (81% degenerate dual pivots, 443 refactor-cap trips over 761 warm
solves on the #598 MILP) is *precisely* the failure mode DSE + BFRT are designed
to suppress. That is not a coincidence — it is the reference telling us which two
techniques are missing.

---

## 2. Component × discopt matrix

Legend — **HAS** (default-on, competitive) · **PARTIAL** (present but flag- or
class-gated) · **WEAK** (exists but underperforms the reference) · **MISSING**.
All discopt evidence is `file:symbol` in this repo; solver evidence in §1.

| Component | Reference relies on | discopt state | Evidence (discopt) |
|---|---|---|---|
| **FBBT** (feasibility bound tightening) | BARON `TDo`, SCIP per-node | **HAS** (default-on, root + per-node) | `crates/discopt-core/src/presolve/fbbt.rs`, `passes.rs:FbbtPass` (default); `_jax/mccormick_lp.py` lifted FBBT; per-node in `solver.py` reduce loop |
| **OBBT** (optimality bound tightening) | BARON `OBTTDo`, SCIP root-scheduled {cite:p}`Gleixner2017` | **PARTIAL** (class-gated: needs MC-LP relaxer + dependent vars + `n_vars ≤ _PER_NODE_OBBT_MAX_VARS`) | `_jax/obbt.py:obbt_tighten_root`; `solver.py` per-node gate (~5290–5330); Rust `presolve/obbt.rs` **not yet orchestrator-wrapped** (`passes.rs` note: "OBBT … integration deferred") |
| **DBBT / marginals** (reduced-cost range reduction) | BARON `MDo` {cite:p}`Ryoo1996` | **HAS** (Rust kernel + Python root/per-node) | `crates/discopt-core/src/presolve/duality.rs:reduced_cost_fixing`; `solver.py:_root_reduced_cost_fixing` / `_reduced_cost_fixing` |
| **Probing** | BARON `PDo`, SCIP presolve | **HAS** (default-on, deadline-bounded) | `crates/discopt-core/src/presolve/probing.rs`; `passes.rs:ProbingPass` |
| **Branch-and-reduce loop (reduce every node)** | BARON core {cite:p}`Ryoo1996` | **PARTIAL** (root fixpoint via presolve orchestrator; per-node reduce is inline Python, not a first-class loop) | `presolve/orchestrator.rs` fixpoint (`~84–136`); per-node FBBT/OBBT inline in `solver.py` B&B loop; `root_fixpoint` flag in flight (BR-1/#78) |
| **Auxiliary-variable lifted relaxation (AVM)** | BARON, SCIP default | **HAS** (this *is* discopt's default node engine) | `_jax/mccormick_lp.py` ("bilinears lifted to aux columns"); `solver.py` `_mc_mode="lp"` on nonconvex+objective |
| **RLT / exact multilinear hull** | BARON, SCIP 8 | **HAS** (auto / class-gated) | `_jax/rlt_cuts.py`; `solver.py` `rlt` param ("auto") |
| **Expression-level convexity detection (nlhdlr-style)** | SCIP 8 nlhdlrs {cite:p}`Bestuzheva2023` | **PARTIAL/WEAK** (convexity module + α-BB/PSD/SOC exist, but no single dispatcher that picks the tightest handler per subexpression) | `_jax/convexity/`, `alphabb.py`, `psd_cuts.py`, `soc_cuts.py`; problem-level classifier `problem_classifier.py` (not per-expression) |
| **Reduced-space McCormick (no aux lifting)** | MAiNGO signature (not BARON/SCIP) | **PARTIAL** (exists, not default) | `_jax/multivariate_mccormick.py`, `mccormick_nlp.py`; **F3 falsified as a root-bound tightener** |
| **Integer / MIR / c-MIR cuts** | HiGHS + SCIP full CG family | **PARTIAL/WEAK** (root-only; c-MIR default-OFF, net-negative on integer-product roots) | `crates/discopt-core/src/lp/mir.rs`, `_jax/cmir_cuts.py`; `solver.py` c-MIR default-off (#587 NO-GO) |
| **Gomory (GMI)** | HiGHS/SCIP | **PARTIAL** (POUNCE-gated, off by default) | `crates/discopt-core/src/lp/gomory.rs`; `solver.py` GMI gate |
| **Cover cuts** | HiGHS/SCIP | **PARTIAL** (root only) | `crates/discopt-core/src/lp/cover.rs`, `_jax/cover_cuts.py` |
| **Cut separation *loop*** (rounds + re-solve + filtering) | SCIP separation loop {cite:p}`Bestuzheva2023` | **WEAK** (cuts added mostly at root as a pool, not an efficacy/orthogonality-filtered multi-round loop) | root cut assembly in `solver.py`; no per-node round loop |
| **Dual simplex — steepest-edge pricing** | HiGHS {cite:p}`Huangfu2018` | **PARTIAL** (dual **Devex** — a steepest-edge *approximation* — present since #178; exact DSE tried under `DISCOPT_DUAL_DSE` and **KILLED** — F11/#99: regresses the #606 pathology 2.5× wall, RefacCap 3.4×) | `dual.rs` `gamma` Devex weights + Goldfarb–Reid update (`select_leaving`); exact DSE `recompute_dse_weights` flag-gated default-OFF |
| **Dual simplex — bound-flipping ratio test** | HiGHS {cite:p}`Huangfu2018` | **HAS** (long-step BFRT since #178) | `dual.rs` `build_candidates` + the breakpoint-flipping loop in `run_dual` |
| **Hypersparse FTRAN/BTRAN + FT update** | HiGHS {cite:p}`Huangfu2018` | **PARTIAL** (sparse LU + product-form update via `feral`; iterative refinement opt-in) | `simplex/linsolve.rs` (`SparseLu`, `update`, `ftran_refined`) |
| **Dense-LU fallback route for wide bases** | (discopt-specific fix for the AVM dense-column pathology) | **PARTIAL** (flag `DISCOPT_LU_DENSITY_ROUTE`, default-OFF; blocked from graduating) | `simplex/linsolve.rs:188–195` `density_route_enabled`; A-2/#85 retry work in flight |
| **Presolve reduction library** | HiGHS/SCIP {cite:p}`Achterberg2020` | **HAS** (aggregate, coeff-strengthen, implied-bounds, eliminate, cliques, symmetry, redundancy) | `crates/discopt-core/src/presolve/*.rs` via `orchestrator.rs` |
| **Reliability / pseudocost branching** | HiGHS/SCIP {cite:p}`Achterberg2005` | **WEAK** (most-fractional among priority vars; strong-branching exists but no pseudocost accumulation) | `solver.py:_select_priority_branch_var`; `_jax/strong_branching.py` |
| **Primal heuristic portfolio (FJ, RENS/RINS, FP)** | HiGHS {cite:p}`Achterberg2020` | **PARTIAL** (multi-start NLP + a feasibility-pump-style rounding; no feasibility jump / RINS / RENS) | `_jax/primal_heuristics.py:MultiStartNLP`; `pounce_layer.py` |
| **Finite-bound inference from constraints** | BARON (infers from constraints) | **HAS** (LP-based bootstrap + equality propagation) | `_jax/obbt.py:bootstrap_finite_bounds` (~1597), wired into `obbt_tighten_root` |
| **Rigorous certificate / safe dual bounds** | all three (Neumaier–Shcherbina style) | **HAS** | `_jax/obbt.py` safe-bound handling; `_numeric.py:is_effectively_finite` |

**Honest read.** discopt already *has* the branch-and-reduce menu (FBBT, DBBT,
probing, OBBT, bootstrap bound inference) — the range-reduction poll is largely
present. Where it is thin is (a) the **LP engine numerics** (DSE + BFRT missing;
this is the measured per-node cost), (b) the **cut separation loop** as a
filtered multi-round process rather than a root pool, (c) **pseudocost
branching**, and (d) **expression-level convexity dispatch** (nlhdlr-style). The
*relaxation math* is at parity or richer (α-BB, PSD/SOC, edge-concave, learned
ICNN — see `maingo-vs-discopt.org`); the *engine and search control* are the gap.

---

## 3. Derived, prioritized roadmap

Levers ranked by **(reference-solver evidence it matters) × (discopt gap size) ×
(tractability)**. Each is a *class* fix, not an instance probe.

### Lever 1 — Dual-simplex steepest-edge pricing + bound-flipping ratio test — **RESOLVED / KILLED (F11, #99)**
- **Reference grounding.** The two techniques HiGHS's dual revised simplex is
  built on {cite:p}`Huangfu2018`; DSE is *the* iteration-count lever and BFRT is
  *the* degeneracy lever — **over Dantzig pricing**.
- **What was actually there (this row was wrong above).** BFRT and dual **Devex**
  (a steepest-edge *approximation*) both landed in #178 (`dual.rs`); they are
  *not* missing. The only genuine delta was Devex → *exact* DSE.
- **Verdict (F11/#99).** Exact DSE was implemented (Forrest–Goldfarb 1992
  recurrence, unit-tested vs a from-scratch weight recompute to 1e-9) behind
  `DISCOPT_DUAL_DSE`, default-OFF, and measured on the reproduced #606 pathology.
  It **regresses**: 2.5× wall, RefacCap 3.4×, Phase1Pivots 3.5×, only −10%
  degenerate dual pivots, and is not node-count-neutral (459→321 node LPs).
  LP-objective bound-neutral to ~1e-13 (the sound part), but a net loss — Devex is
  already a good DSE approximation and exact DSE's seeding/refactor cost dominates.
  **Do not relitigate.** The #606/#598 per-node cost is *not* a pricing gap; it is
  degeneracy geometry + FT-update/refactor churn on lifted bases (see DECOMP-1 §5.1a:
  iterative refinement / anti-degeneracy on partitioned formulations, and the
  primal-side `RefacCap`).

### Lever 2 — Graduate the dense-LU route + failure-triggered retry (A-2/#85)
- **Reference grounding.** Not a reference *feature* but the discopt-specific
  manifestation of the AVM dense-column pathology that HiGHS avoids structurally;
  the fix restores the robustness HiGHS's numerics provide.
- **discopt gap.** Flag exists but default-OFF and blocked from graduating
  (`linsolve.rs:188`); the conditioning-gate hypothesis was **falsified** (F6/#77)
  and the mechanism corrected to *3× LP-failure rate → lost certificate*, with a
  green-entry retry fix.
- **Why #2.** In flight, entry-green, converts `feasible`→`optimal` on the
  nvs21-class; complements Lever 1 (same engine). Soundness-gated: retry may only
  replace a failure with a robust result.

### Lever 3 — Per-sub-class relaxation strength for the loose-root class (Lever A of DECOMP-1)
- **Reference grounding.** BARON/SCIP close large root gaps with **tight
  relaxations + range reduction feeding them** {cite:p}`Tawarmalani2002`,
  {cite:p}`Bestuzheva2023`. DECOMP-1 shows relaxation strength is the dominant
  uncertified driver (7/10), but *not one relaxation*: (i) the α-BB/interval
  fallback engine that LP-declined pure-integer models route to (nvs05, tanksize);
  (ii) the x·log(x)/Gibbs root (ex6_2_5/9, root gaps 300–400×) where the centropy
  tangent cuts (RELAX-1) are not yet enough; (iii) the NLP-BB root that opens with
  ~no dual bound (clay).
- **discopt gap.** The *math* exists; the *routing and tightness per sub-class* is
  weak. This is an expression-level convexity-dispatch gap (nlhdlr analogue, §2).
- **Why #3.** Biggest attributed share of certification effort, but heterogeneous
  and per-class — lower tractability than 1–2. Do **not** spend this on
  integer-product roots (already SCIP-tight; see Lever-NO-GO below).

### Lever 4 — A cut *separation loop* (rounds + efficacy/orthogonality filter) at the root, staged
- **Reference grounding.** SCIP's separation loop is a filtered multi-round
  process, not a one-shot pool {cite:p}`Bestuzheva2023`; this is where root dual
  bound is actually won on structured MILP/MIQCP.
- **discopt gap.** **WEAK** — cuts are assembled largely as a root pool without an
  efficacy/orthogonality-filtered re-solve loop (§2). c-MIR/GMI are gated off
  because, *without* a proper loop + filter, they are net-negative.
- **Why #4.** High reference evidence, real gap, but medium tractability
  (touches the root LP driver) and partially entangled with CUTS-1/CUTS-2 in
  flight. Sequence after the engine (Levers 1–2) so cut rounds run on a robust LP.

### Lever 5 — Pseudocost / reliability branching
- **Reference grounding.** The post-2000 canon for both SCIP and HiGHS
  {cite:p}`Achterberg2005`; reliability branching is the standard variable-selection
  default.
- **discopt gap.** **WEAK** — most-fractional among priority vars; strong
  branching exists but no pseudocost history accumulation (§2).
- **Why #5.** Solid reference evidence and a genuine gap, but node-count wins here
  are second-order to fixing the *per-node cost* (Levers 1–2) — a cheaper node is
  worth more than fewer expensive nodes until the engine is fixed. DECOMP-1 finds
  pure B&B-machinery (Lever C) binding on only 1/10 instances.

**Explicit NO-GO (kept from binding falsifications, do not relitigate):**
c-MIR / aggregation cuts on **integer-product roots** (nvs17/19/24) — the root is
already SCIP-with-cuts-tight there (#587), so cut strength is dead on that class
(F-class NO-GO). Reduced-space McCormick as a *root-bound tightener* — F3.

---

## 4. Reconciliation with this session's results

DECOMP-1 (`certification-effort-decomposition-2026-07-10.md`) and
`gap-closing-execution-plan.md` §6 (F1–F8) record five session KILLs. For each,
did the reference literature *predict* it? If yes, this guide would have saved the
probe.

| Session KILL | Falsification | Literature predicted it? | Grounding |
|---|---|---|---|
| **Reduced-space McCormick auto** (tightens root bounds) | **F3** — "0 wins, ties or looser" | **YES.** | MAiNGO's *known* trade-off: reduced-space/composed envelopes are **weaker per node** than lifted AVM/RLT — they are not the convex hull `maingo-vs-discopt.org`; {cite:p}`Bongartz2018`. Predicting a *root-bound* win from the looser method was backwards. |
| **c-MIR on integer-product roots** (CUTS-1 NO-GO, #587) | root already cut-tight → cut strength dead | **YES.** | The root is already at "SCIP-with-cuts" tightness on that family (DECOMP-1 §3, nvs19 root_gap 0.53%). Adding cuts to an already-tight root cannot help — SCIP itself gates cuts by efficacy {cite:p}`Bestuzheva2023`. |
| **Conditioning-gate** (LU cond estimate discriminates failing bases) | **F6/#77** — populations inverted; mechanism = 3× LP-failure rate | **PARTIAL.** | HiGHS's answer to ill-conditioned lifted bases is *not* a cond-number gate but DSE + BFRT + FT-update stability {cite:p}`Huangfu2018`. The literature says "fix the pivoting/refactorization", which predicts a cond-gate is the wrong instrument — but does not itself pre-compute the inverted populations. Half-predicted. |
| **A-RESCUE / tainted-tree node retention** (recover the frontier bound of a decertified tree) | **F8/#89** — frontier not rigorous once subtrees removed unproven; headroom marginal (nvs05 1.348→1.352, not 3.6×) | **YES — this is the BARON finite-domain law.** | A node pruned *without proof* cannot contribute a rigorous frontier bound, exactly as BARON *refuses to report a lower bound* when its finite-domain assumption is violated ([GAMS/BARON](https://www.gams.com/latest/docs/S_BARON.html); §1.1, §5). The rigorous fix is *preventing the unsound removal* (bound sentinel-fathomed nodes with the always-valid interval bound so they branch), not recovering a bound the search never proved. |
| **root_fixpoint neutral** (default-ON reduce fixpoint at root) | BR-1/#581: −14% nodes, 0 loss — a *modest* win, not the lever | **YES.** | BARON's reduce phase runs **every node**, and its payoff is the *bound-feeds-relaxation cycle over the tree* {cite:p}`Ryoo1996`, {cite:p}`Puranik2017` — a one-shot *root* fixpoint captures only a fraction. The literature predicts a root-only reduce is helpful-but-not-transformative, which is exactly the −14%-nodes verdict. |

**Score: the literature predicted 4 of 5 KILLs outright (F3, c-MIR, A-RESCUE,
root_fixpoint) and half-predicted the 5th (conditioning-gate).** Every one of
these was reached by an instance-by-instance probe that the reference map would
have pre-empted:

- F3 (reduced-space) — `maingo-vs-discopt.org` already states, with citation, that
  reduced-space envelopes are looser per node. The probe re-derived a documented
  trade-off.
- A-RESCUE — the BARON finite-domain law (§5) says an unproven prune yields no
  rigorous bound. The probe re-discovered a soundness axiom.
- root_fixpoint — Ryoo–Sahinidis put the reduce phase at *every* node for a
  reason; a root-only fixpoint is knowably partial.

This is the validation the maintainer asked for: **the guide would have saved the
probing.** The one genuinely new thing the probes found — that the *per-node LP
engine numerics*, not relaxation math, gate the #598-class certification — is
itself the thing the HiGHS literature points straight at (Lever 1).

---

## 5. The unbounded-variable question (task #94; nvs05 / tanksize stall class)

**What BARON/SCIP require.** Both spatial global solvers require a **finite box**
to build valid convex relaxations — a McCormick envelope over an unbounded box
does not exist. BARON states this explicitly: finite bounds on all variables *and*
all nonlinear subexpressions
([GAMS/BARON](https://www.gams.com/latest/docs/S_BARON.html)); finite variable
bounds are necessary but not sufficient (`1/x` on `[0,1]`).

**How they infer bounds when the user doesn't supply them.**
- **BARON:** *"attempts to infer appropriate bounds from problem constraints."* If
  that fails, it prints *"User did not provide appropriate variable bounds"* and
  **withholds the global-optimality claim / lower bound** rather than fabricate a
  certificate. A numeric fallback (~±10^10) may be used only to construct a
  *feasible* relaxation, with the optimality claim dropped.
- **SCIP:** derives finite bounds via **FBBT reverse-propagation through the
  expression graph** and **root OBBT** {cite:p}`Vigerske2018`,
  {cite:p}`Gleixner2017`; if a nonlinear expression's argument stays unbounded, the
  relaxation for that expression is not built and the affected dual bound is not
  claimed.

**What discopt already does (and it maps cleanly onto BARON).**
- Default box for an unbounded continuous variable is `[-9.999e19, 9.999e19]`
  (`python/discopt/modeling/core.py`), with `EFFECTIVE_INF = 1e19` and
  `is_effectively_finite` (`_jax/_numeric.py`); the McCormick relaxation numeric
  cap is `_RELAX_NUMERIC_CAP = 1e10` (`_jax/milp_relaxation.py`).
- **discopt has BARON's "infer bounds from constraints" step already:**
  `bootstrap_finite_bounds` (`_jax/obbt.py:~1597`) minimizes/maximizes each
  unbounded variable over the **linear feasible region** with an exact LP solver
  (sound: only shrinks the box), wired into `obbt_tighten_root` with a two-round
  fixpoint interleaved with `propagate_equality_defined_bounds` to unlock bounds
  through equality-defined intermediates (divisions, logs).
- When a **nonlinear-participating** variable stays `≥ _RELAX_NUMERIC_CAP`, the
  incremental McCormick path is refused and it routes to a cold rebuild + HiGHS
  unboundedness cross-check (`_jax/mccormick_lp.py:~429`), degrading soundly to
  `status="feasible", gap_certified=False` — **the same behavior BARON exhibits**
  (a feasible point, no global claim).

**Why nvs05 / tanksize still stall (the residual gap vs BARON).** These have
*division/sqrt slacks* like `x4 = 4243.28/(x0·x1)` (`nonlinear_bound_tightening.py`
comments name the nvs05/gear4 class) whose argument product is bounded only
*implicitly* by other constraints. The **linear** bootstrap
(`bootstrap_finite_bounds`) cannot see a bound that is only implied *nonlinearly*,
so the slack stays effectively unbounded, the McCormick LP declines at the root
(DECOMP-1: nvs05's root LP is a *sound refusal*), nodes fall back to
α-BB/interval, and the tree grinds (Lever A sub-class (i)).

**The BARON-aligned fix (informs task #94).** BARON closes this gap not with a
richer linear bootstrap but with **nonlinear FBBT/OBBT that propagates the implicit
bound through the nonlinear constraint** and a marginals-based contraction
{cite:p}`Ryoo1996`, {cite:p}`Puranik2017`. discopt's `nonlinear_bound_tightening.py`
is the right home; the gap is that the bootstrap runs on the *linear* system only.
The correct, sound extension is a **nonlinear bootstrap** that derives finite
bounds for division/sqrt-slack arguments from the nonlinear constraints — feeding
exactly the bound-tightening → tighter-relaxation cycle BARON relies on — rather
than any change to the ±10^10 clamp (which BARON itself uses only to *drop* the
guarantee). This is Lever 3 sub-class (i) with a concrete, reference-grounded
mechanism.

---

## 6. References used (all verified present in `docs/references.bib`)

- `Ryoo1996` — Ryoo & Sahinidis, *A branch-and-reduce approach to global
  optimization*, J. Glob. Opt. (range reduction / marginals).
- `Puranik2017` — Puranik & Sahinidis, *Domain reduction techniques for global
  NLP and MINLP*, Constraints (the BARON range-reduction survey).
- `Tawarmalani2002` — Tawarmalani & Sahinidis, *Convexification and Global
  Optimization* (the branch-and-reduce monograph).
- `Tawarmalani2005` — *A polyhedral branch-and-cut approach to global
  optimization*, Math. Prog.
- `Kilinc2018` — Kılınç & Sahinidis, *Exploiting integrality … with BARON*,
  Opt. Methods & Software (**added this PR**).
- `McCormick1976`, `Sherali1990` — McCormick envelopes; RLT.
- `Vigerske2018` — Vigerske & Gleixner, *SCIP: global optimization of MINLP in a
  branch-and-cut framework* (**added this PR**).
- `Bestuzheva2023` — *Global optimization of MINLP with SCIP 8* (nlhdlrs, RLT /
  intersection cuts) (**added this PR**).
- `Gleixner2017` — *Three enhancements for optimization-based bound tightening*
  (OBBT scheduling).
- `Huangfu2018` — Huangfu & Hall, *Parallelizing the dual revised simplex method*
  (DSE, BFRT, hypersparse FTRAN/BTRAN, FT update).
- `Achterberg2020` — Achterberg et al., *Presolve reductions in MIP*.
- `Achterberg2005` — *Branching rules revisited* (reliability/pseudocost).
- `Bongartz2018` — MAiNGO technical report (reduced-space McCormick) (**added
  this PR**).

Web sources (verified live 2026-07-10):
[GAMS/BARON options](https://www.gams.com/latest/docs/S_BARON.html) ·
[ERGO-Code/HiGHS](https://github.com/ERGO-Code/HiGHS) · SCIP OBBT propagator
documentation at scipopt.org (`prop_obbt.c`; parameter names confirmed via search).
