# #282 — syn_*/rsyn_* process-synthesis global-search gap: diagnosis + a correctness fix found en route

**Task:** [#282](https://github.com/jkitchin/discopt/issues/282) — the process-synthesis
(`syn*` / `rsyn*`) MINLP global-search gap: discopt returns sound feasible incumbents
(~19 % median gap on the issue's panel) but does not close to the optimum within budget,
while SCIP closes the same instances in < 1 s. **Date:** 2026-07-17. **Regime:** entry
experiment / diagnosis, plus one localized correctness fix in the opt-in OA path
(`python/discopt/solvers/oa.py`) with a regression test. **Env:** in-repo build
(`maturin develop --release`), `JAX_ENABLE_X64=1`.

---

## Headline

1. **The perf gap itself is NOT reproducible in this environment**, so no perf/relaxation
   change is shipped here (CLAUDE.md §4 — no fix ships on an un-run entry experiment). The
   named instances (`rsyn0805m`, `syn40m`, `syn15m02hfsg`, …) are absent: the Dropbox
   MINLPLib snapshot is not mounted, `minlplib.org` is blocked by the network policy
   (`connect_rejected` 403), and the in-repo corpus vendors only the smallest sibling,
   `syn05m.nl`. `syn05m` **solves to proven optimality in ~1–3 s / 9 nodes** on the current
   default path, so it does not exhibit the gap.

2. **The plan's #282 characterization is partly corrected by measurement.**
   `scip-gap-closing-plan.md` §0 lists #282 as needing "general-NL relaxation strengthening
   + LNS heuristics." Measured on `syn05m` and synthetic convex-synthesis proxies:
   - The family **is convex and is already detected as convex** and routed to NLP-BB on the
     **default** path (`_classify_model_convexity` → `_solve_nlp_bb`, `solver.py:4777`). So
     relaxation *selection* is not the missing piece, and McCormick strengthening is not the
     lever for this (convex) family.
   - **Outer Approximation is the natural method** for convex MINLP (how SCIP/DICOPT/BONMIN
     close syn/rsyn fast): on `syn05m`, OA converges in **2 major iterations / 0.74 s** vs
     NLP-BB's 9 nodes / 2.6 s.
   - **BUT auto-routing convex MINLPs to OA is falsified as a general win** on harder
     proxies (see F-1 below) — OA did not converge (86 major iters at 20 binaries) and, worse,
     **mis-reported a feasible problem** (see the correctness fix). So the lever is *NLP-BB
     throughput + primal/LNS on the default path*, not a naive switch to OA — and OA needs the
     robustness fix below before it could ever be considered for auto-routing.

3. **A correctness bug was found and fixed** while probing the OA angle: the multitree OA
   path returned `status="infeasible"` for a **feasible** model whenever it terminated with
   no incumbent for **any** reason other than a genuine master-infeasibility proof —
   including hitting the time limit, the iteration limit, or a user termination hook. A
   solver that merely ran out of budget must never claim infeasibility (CLAUDE.md §1). Fixed;
   regression test added.

---

## Measurements

### M-1 `syn05m` (only vendored real instance) — default path

| path | status | obj | bound | nodes | wall |
|---|---|---|---|---|---|
| default (`nlp_bb` auto) | optimal | 837.7324 | 837.7324 | 9 | 2.6 s |
| spatial (`nlp_bb=False`) | optimal | 837.7324 | 837.7324 | 7 | 1.2 s |
| OA (`solver="mip-nlp"`, `oa`) | optimal | 837.7324 | 837.7324 | 0 (2 major iters) | 0.74 s |

`_classify_model_convexity(syn05m)` → `known=True, is_convex=True` (all 28 constraints
convex). The default path routes it through `_solve_nlp_bb`. Known optimum 837.7324 (max).

### M-2 Synthetic convex process-synthesis proxy (`scratchpad/hard_synth.py`)

`max Σ revᵢ·log(1+xᵢ) − fixᵢ·yᵢ` s.t. big-M unit logic `xᵢ ≤ U·yᵢ`, precedence coupling
`yᵢ ≤ yᵢ₋₁ + yᵢ₋₂`, and shared budget/cardinality rows — a convex-max-of-concave MINLP that
mirrors the syn/rsyn class (all runs `convex_known=True`, 20 s limit):

| n (binaries) | default: obj / bound | OA: obj / bound / major-iters |
|---|---|---|
| 20 | 22.34 / 45.92 (159 nodes, TL) | 32.84 / 55.61 / **86** (TL) |
| 30 | 24.15 / 61.67 (287 nodes, TL) | **infeasible / None / 1** ← wrong |

Neither path closed the proxy in budget. The default (NLP-BB) has the **tighter dual bound**;
OA has the **better incumbent** (consistent with the issue's "sound suboptimal incumbents").
The n=30 OA "infeasible" is the correctness bug below.

---

## F-1 (falsification) — "auto-route detected-convex MINLPs to OA"

**Hypothesis:** since the family is convex and OA closes `syn05m` in 2 iterations, routing all
detected-convex MINLPs to OA on the default path closes #282. **Verdict: KILL.** On the M-2
proxy OA needed 86 major iterations at 20 binaries (no convergence) and returned a wrong
status at 30 binaries. OA's convergence on convex MINLP is not uniformly fast — it depends on
master-MILP hardness and cut quality — and its incumbent-vs-bound trade-off is the mirror of
NLP-BB's, not a strict improvement. Recorded so this dead-end is not re-walked. The productive
levers for #282 remain **(a) NLP-BB per-node throughput and (b) primal/LNS incumbent quality
on the default path** (companion to #268/#276), to be measured against the real corpus.

---

## The correctness fix (shipped here)

**File:** `python/discopt/solvers/oa.py`, multitree `solve_oa` terminal return.

**Bug:** when the loop ended with no incumbent and no "unresolved NLP failure" (the C-35
case), it fell through to `status="infeasible"` **regardless of why it stopped**. So a
resource-limited or user-terminated run on a demonstrably feasible model reported it as
infeasible — a false infeasibility certificate. Deterministic repro (now a regression test):
a feasible convex MINLP (`max x+y s.t. x²+y²≤10`, integer) stopped at iteration 0 by a
termination hook returned `status="infeasible"`.

**Fix:** report `infeasible` only when infeasibility was actually **proven** — i.e. the master
MILP (a valid relaxation of the integer feasible set, carrying globally-valid OA / rigorous
no-good cuts) was itself infeasible: `final_reason ∈ {"master_infeasible",
"master_infeasible_unrepaired"}`. Every other no-incumbent exit (`time_limit`,
`iteration_limit`, `user_termination`, `master_error`, `master_unbounded`, `cycling`,
`stalling`, repair-loop give-up) is **inconclusive** → `status="unknown"`, matching the C-35
unresolved-config precedent directly above it in the same function. The genuine-infeasible
path (`test_infeasible_model`: `x∈[0,1], x≥2` → `master_infeasible`) is unchanged.

**Scope check:** the single-tree path (`lp_nlp_bb`) already mapped these correctly
(`time_limit`/`iteration_limit`/`no_feasible_point`); only the multitree terminal fallthrough
was affected. The **default** solve path never calls `solve_oa` (OA is opt-in via
`solver="mip-nlp"`), so no default-path behavior changed.

**Tests:**
- New: `test_oa.py::test_no_incumbent_on_limit_is_not_reported_infeasible` (fails before,
  passes after).
- Updated incidental status assertions in 4 mechanism tests that hit `iteration_limit`
  (`max_iterations=0/1`, monkeypatched masters) and previously encoded the false
  "infeasible": `test_oa.py` (×3, no-good-cut mechanism) and
  `test_mip_nlp.py::test_oa_rnlp_initialization_adds_cuts_at_relaxation_point`. Their
  mechanism assertions are untouched. `master_infeasible_unrepaired` stays `infeasible`
  (`test_mip_nlp.py::test_mip_nlp_shot_master_repair_failure_records_diagnostic`).

**Verification:** `test_oa.py`, `test_mip_nlp.py`, `test_c35_oa_nogood_nlp_failure.py`,
`test_gdpopt_loa.py` → 204 passed / 2 skipped; `test_oa.py -m slow` → 10 passed; smoke
suite → 651 passed / 13 skipped; adversarial suite (`-m slow`) → 9 passed, 1 pre-existing
**environmental** deadline-overrun (`test_large_dense_jacobian_no_crash`, 69 s vs 48 s budget
on this slower container — default spatial-B&B path, never touches `solve_oa`).

---

## Next experiments (need the real corpus; run before any perf build)

Draw `rsyn0805m/0810m/0815m`, `syn15m02hfsg`, `syn30hfsg`, `syn40hfsg`, `syn40m` from the
MINLPLib snapshot (`minlplib.solu` as oracle). For each:

1. **Attribute the gap.** Log NLP-BB node count, per-node NLP wall, first-incumbent time,
   and root dual bound vs oracle. Decide whether the gap is dual (bound too loose → cuts /
   OBBT) or primal (incumbent too weak → LNS/RENS/#276) or throughput (node/s → #268).
2. **OA as a bounded primal heuristic**, not a router: run OA for a small iteration cap to
   harvest incumbents, inject into the default tree, keep the NLP-BB dual bound. Kill if it
   does not improve the incumbent within a fixed fraction of budget.
3. **Graduation** for any bound/primal-changing change follows the CLAUDE.md Regime-2 panel
   gate (flag default-OFF; `incorrect_count = 0`; no bound above oracle; incumbents
   independently feasibility-verified) before default-ON.

Kill criterion for this whole track: if measured NLP-BB node throughput and first-incumbent
latency on the real instances are already competitive and only the *bound* lags, the work
collapses to cut/OBBT strengthening and #282 should be re-pointed at that.
