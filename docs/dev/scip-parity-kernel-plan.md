# SCIP-parity plan: the compiled branch-and-cut kernel (BCK)

**Date:** 2026-07-19
**Status:** proposed (supersedes incremental throughput work; companions:
`tenx-plan.md` §0 operating rules, `engine-performance-plan.md` (EP; its
measurements are the evidence base here), `performance-plan.md` CC1–CC5,
`docs/dev/issue-282-syn-rsyn-diagnosis-2026-07-17.md`, issues #781/#786/#764)
**Owner intent:** *"SCIP/BARON solve these problems easily; we simply must
obtain that level of performance. Do not get stuck in small, incremental
approaches."*

---

## §1. High-level diagnosis: why months of correct work has not moved the end metric

The end metric is wall-clock vs SCIP/BARON per family. Current best cases:
syn30m certified in **60 s** vs SCIP **0.8 s**; tanksize **10 nodes/s** vs
BARON **1300 nodes/s**; rsyn0805m **~15% gap at 60 s** vs SCIP closed in 0.5 s.

Three measured facts explain the whole gap, and none of them is fixable by
another flag:

**F1 — The per-node loop runs in the interpreter.** The 116-instance profile
(EP §1): **61% JAX, 39% Python, ~0% Rust LP wall** — the in-house simplex is
idle while every node pays Python/JAX relaxation *construction* and
*separation*. The entire EP caching campaign (EP1/EP2/EP4a/EP4b/EP5, all
landed or honestly falsified) took nvs09 from 294 → ~50 ms/node — and
plateaued there, because what remains IS the interpreter and tracing layer.
SCIP's node is 0.1–1 ms. **No further caching inside Python closes a 50–500×
per-node gap; that hypothesis is spent** (EP status table is its record).
The NLP-BB path is worse: it solves a full NLP per node.

**F2 — The bound side is now adequate but cannot compose at speed.** This
session validated the missing separator family (GMI: closes 75–93% of the
convex-panel root spread; rsyn0805m root beats SCIP's) and shipped it sound —
and its graduation was **held** (#781): root-only cutting at certifying
intensity reliably starves first-incumbent discovery (0/5 incumbents on
rsyn0810m/tls2). The EP3 revert recorded the dual lesson: skipping per-node
tightness for speed loses proofs — *a real solver never trades tightness for
node rate*. And `scip-gap-nvs-diagnosis.md` measured that OBBT, cuts, and
throughput are **multiplicative — they must land together**. Cutting
throughout the tree (what SCIP actually does) requires nodes cheap enough
that separation and heuristics fit inside them. Cheap nodes require F1 fixed.

**F3 — The kernel ingredients already exist in Rust, unwired.**
`crates/discopt-core` already contains: warm-startable primal/dual simplex
with basis management (`lp/simplex/`, `lp/basis.rs`), **Gomory** (`lp/gomory.rs`),
**c-MIR** (`lp/mir.rs`), **cover** (`lp/cover.rs`), **cut selection**
(`lp/cut_select.rs`), aggregation, a B&B tree (`bnb/`), FBBT (`presolve/`),
the expression IR (`expr.rs`), and the .nl parser. They sit idle (F1's "~0%
Rust wall") because the loop that would drive them is Python. This repeats
the #282 pattern at architecture scale: the mechanism exists; the plumbing
never lets it run.

**Conclusion.** The missing artifact is not another lever — it is a **single
compiled node loop**: analyze-once in Python, then a Rust branch-and-cut
kernel that owns {node select → FBBT → row refresh from box → warm dual
simplex → in-tree separation from a pool → primal rounding/diving → branch}
with Python re-entered only for rare, batched oracle work and final
verification. Everything this repo has built — envelope library at SOTA
parity (relaxation catalog), validated GMI+selection (#781), FBBT, the
simplex — is necessary-but-insufficient until that loop exists. This is the
BARON/SCIP architecture already named as the north star in EP §1; the EP
series proved it cannot be reached by caching from the Python side.

## §2. Target architecture

**Analyze once (Python, per solve):** parse, classify, convexity
certificates, presolve/coefficient tightening (#780), envelope **templates**
— every relaxation row becomes a box-parametric generator (McCormick
bilinear, univariate secant/tangent, OA tangent stencils); static
`ProblemData` (linear rows, templates, integrality, objective) marshaled to
Rust **once**.

**Per node (Rust, the kernel):** refresh row coefficients from the node box
(a few flops per row — no JAX), FBBT, warm dual-simplex from parent basis,
separation rounds from the existing Rust separators (Gomory from own tableau,
c-MIR, cover) through `cut_select` + a pool, LP-driven rounding/diving primal
heuristics, pseudocost/spatial branching, certificate-safe fathoming
(existing tree-manager invariants).

**Python callbacks (rare, batched):** OA tangent refresh at new LP optima /
incumbent candidates (JAX, batched); NLP polish of promising integer points;
multilinear facet rows where no closed form exists (or in-kernel tiny LPs —
they are LPs). **Final incumbent verification stays the #779 pristine-model
guard.** Dual bounds come from the in-house simplex — already the trusted
default LP engine, so the certificate story does not change.

## §3. Phases — each gated on the END metric, entry-experiment-first

**E0 (days) — kernel node-rate ceiling.** Rust bench: warm dual-simplex
re-solves under bound flips on the *real* LPs (rsyn0805m root LP + cuts ≈
230×170; tanksize McCormick LP; nvs09). **Kill:** < 500 re-solves/s on the
rsyn-class LP → the in-house simplex cannot carry the kernel; re-scope
(in-process HiGHS as kernel LP is the named fallback) before any loop code.

**E1 (days) — template-refresh parity.** For the envelope families covering
the certifying panel: analyze-phase templates + Rust refresh must reproduce
the Python-built rows (≤1 ulp) on all 62 vendored instances at the root box
and at perturbed child boxes. **Kill:** < 90% of panel rows templatable →
scope the per-node Python residue explicitly before proceeding.

**P1 (weeks) — convex-family kernel (LP/OA branch-and-cut,
Quesada–Grossmann).** Linear rows + OA tangent rows + integrality; in-tree
Gomory/c-MIR/cover under `cut_select` + pool; rounding/diving primal; Python
callbacks for tangent refresh + NLP incumbent verification. Behind
`DISCOPT_KERNEL` (default OFF), legacy path intact. **Gate:** rsyn*/syn*/clay
panel certified ≤ **5 s** each (SCIP ≤ ~1 s; 5× is the interim bar),
first-incumbent latency ≤ legacy, `incorrect_count = 0`, all suites, §5
Regime-2 panel. This retires #781's HOLD the right way and delivers #786.

**P2 (weeks) — spatial kernel.** E1 templates + in-loop FBBT + OBBT probes on
the warm LP + spatial branching + multilinear facets via in-kernel LPs.
First run in **bound-neutral mode** (fixed branching, legacy cut set) to
byte-compare trees vs the legacy driver on the certifying panel; then full
mode. **Gate:** nvs09 ≤ 1 s; tanksize certified ≤ 10 s; global50 easy-class
SGM ratio vs recorded BARON ≤ 3×.

**P3 — graduation.** Per-family default flip behind the §5 panel
(cert-clean + net-positive + first-incumbent latency, the #781 lesson baked
into the panel metric); legacy path retained as `DISCOPT_KERNEL=0`.

**Secondary track (independent, bound-side):** #764's RLT-2 / fractional
envelope lever for the pooling class — BARON's tanksize root (0.955 vs our
0.92) is relaxation strength the kernel does not supply; entry experiment
per the falsification record already on #764.

## §4. Explicitly rejected (with the record that kills each)

- More Python-side caching/incrementalism for throughput — EP plateau (§1 F1).
- Root-only cut intensification — #781 HOLD (primal starvation, measured 0/5).
- Trading per-node tightness for node rate — EP3 revert (lost proofs).
- Single-lever pushes ("just cuts", "just OBBT", "just throughput") —
  `scip-gap-nvs-diagnosis.md`: multiplicative, must compose.
- NLP-per-node retention for the convex family — F1; SCIP/Bonmin use LP/OA.

## §5. Correctness contract (unchanged, binding)

`incorrect_count ≤ 0` zero slack; certificate invariant every panel; kernel
bound-neutral mode must byte-match legacy trees before full mode lands
anything; every bound-changing stage default-OFF → Regime-2 panel; #779
pristine-model incumbent guard is the last line for every path; no safety
mechanism weakened to make a gate (CLAUDE.md §0.3-equivalents apply verbatim).
