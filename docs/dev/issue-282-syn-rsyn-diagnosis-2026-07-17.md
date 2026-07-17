# #282 — syn_*/rsyn_* process-synthesis global-search gap: diagnosis + a correctness fix found en route

> **STATUS 2026-07-17 (round 2): the round-1 conclusions below are PARTLY FALSIFIED.**
> The MINLPLib snapshot is now mounted, so the entry experiment round 1 could not run has
> been run on all seven real instances. Measured: the gap is **dual-dominated on 7/7**, the
> family is **not uniformly convex** (3/7 are nonconvex), and the round-1 dismissal of
> relaxation strengthening does not survive contact with the corpus. **Read §R2 first** —
> it supersedes Headline items 1–2 and F-1's forward-looking claim. The OA correctness fix
> (Headline 3) is unaffected and stands.

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

## §R2 — round 2 (2026-07-17, real corpus): the gap is dual-dominated

**What changed:** the Dropbox MINLPLib snapshot is mounted, so all seven named instances are
available. Round 1's blocker is gone and its entry experiment ("attribute the gap") has been
run. **Env:** discopt `0.6.1.dev0`, `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu`, 14-core darwin.
**Harness:** `discopt_benchmarks/scripts/issue282_gap_attribution.py`. **Raw:**
`discopt_benchmarks/results/issue282/`.

All seven carry an `=opt=` tag (a *proven* optimum, both fences at once), so for these
maximization instances the decomposition is exact, not heuristic:

    reported_gap = bound − objective = (bound − opt) + (opt − objective)
                                        └dual excess┘   └primal deficit┘

Both terms are ≥ 0 for a sound solver. (Note `generality_sweep.load_solu` reads only
`=best=`/`=bestdual=` and returns *nothing* for these instances — key on `=opt=`.)

### R2-1 Attribution at 60 s — dual-dominated on 7/7

| instance | convex | path | opt | obj | bound | dual excess | primal deficit | nodes | root_t |
|---|---|---|---|---|---|---|---|---|---|
| `rsyn0805m` | yes | NLP-BB | 1296.1 | 1116.5 | 1685.9 | **+30.1 %** | +13.9 % | 319 | 8.1 s |
| `rsyn0810m` | yes | NLP-BB | 1721.4 | 1548.6 | 2486.4 | **+44.4 %** | +10.0 % | 191 | 10.2 s |
| `rsyn0815m` | yes | NLP-BB | 1269.9 | 1047.3 | 1886.8 | **+48.6 %** | +17.5 % | 319 | 13.4 s |
| `syn15m02hfsg` | **no** | spatial | 2832.7 | **2832.7** | 3343.5 | **+18.0 %** | **0.0 %** | 411 | 21.2 s |
| `syn30hfsg` | **no** | spatial | 138.2 | **138.2** | 1375.0 | **+895 %** | **0.0 %** | 469 | 11.6 s |
| `syn40hfsg` | **no** | spatial | 67.7 | 64.4 | 1853.7 | **+2638 %** | +4.8 % | 375 | 16.6 s |
| `syn40m` | yes | NLP-BB | 67.7 | 33.2 | 1231.2 | **+1718 %** | +51.0 % | 479 | 4.2 s |

**Every instance is dual-dominated at 60 s** (dual excess > 2 × primal deficit). On
`syn15m02hfsg` and `syn30hfsg` the incumbent **is the proven optimum** — discopt has already
found the answer and simply cannot prove it. **Soundness: clean** — no bound on the wrong
side of `=opt=`, no incumbent beating it, on any of the 14 runs. The issue's "sound
suboptimal incumbents" framing is confirmed as sound, but its *implied lever* (better
incumbents) is wrong: the incumbent is the healthy half.

**Honest caveat — the 60 s dual-dominance is budget-dependent on the convex half.** A 120 s
`rsyn0805m` run narrows it to parity:

| budget | obj | bound | nodes | dual | primal | verdict |
|---|---|---|---|---|---|---|
| 60 s | 1116.458 | 1685.90 | 319 | +30.07 % | +13.86 % | dual_dominated (2.17×) |
| 120 s | 1116.458 | 1467.99 | 1245 | +13.26 % | +13.86 % | mixed (0.96×) |

So for `rsyn*` both halves matter in absolute terms. But note *what moved*: over that extra
60 s the incumbent improved by **exactly zero** (1116.4583 → 1116.4583; the node trace shows
it frozen from t ≈ 49.9 s to t ≈ 119.6 s) while the bound gained 218 units. **The marginal
return on budget is entirely dual**, and the improver heuristics ran for those 70 s and
produced nothing. The `hfsg` trio is unambiguous at any budget (primal deficit ≡ 0).

**Environment caveat:** this box is shared and was under concurrent load (a recurring ~10-core
`expt_budget.py`; load avg 7–15 throughout). **Wall-clock-derived numbers here — `root_time`,
`node_count`, the 5 s no-incumbent result — are soft and should be re-measured on a quiet box
before being cited as throughput targets.** The R2-2 root-bound comparison below is
timing-independent (it compares relaxation *values* at the root, not speed), so the headline
conclusion does not rest on the contended timings.

### R2-2 The root relaxation is the gap (SCIP reference)

SCIP 6.2.1 on the same `.nl` files, same box. `scip_root` = dual bound at a 1-node limit:

| instance | opt | SCIP root | **discopt root** | SCIP full solve |
|---|---|---|---|---|
| `rsyn0805m` | 1296.1 | +16.1 % | **+62.9 %** | optimal, 0.50 s |
| `rsyn0810m` | 1721.4 | +9.5 % | **+72.1 %** | optimal, 0.51 s |
| `rsyn0815m` | 1269.9 | +17.9 % | **+103.6 %** | optimal, 0.60 s |
| `syn15m02hfsg` | 2832.7 | +6.9 % | **+124.7 %** | optimal, 0.69 s |
| `syn30hfsg` | 138.2 | +43.3 % | **+955 %** | optimal, 1.12 s |
| `syn40hfsg` | 67.7 | +281.6 % | **+3041 %** | optimal, 1.55 s |
| `syn40m` | 67.7 | **+5.2 %** | **+2609 %** | optimal, 0.87 s |

discopt's root bound is looser than SCIP's on **all seven**, by 4×–500× in relative excess.
SCIP proves optimality on the whole panel in 0.5–1.6 s. This is a **root-relaxation quality**
problem first; the tree cannot recover what the root gives away.

### R2-3 Corrections to round 1 (measurement wins)

1. **"The family is convex" — FALSE as stated.** Round 1 generalized from `syn05m`, the one
   vendored sibling. Measured on the real seven (both `_classify_model_convexity` and
   `_jax.convexity.classify_model(use_certificate=True)` agree): `rsyn0805m`, `rsyn0810m`,
   `rsyn0815m`, `syn40m` are **convex** → NLP-BB; `syn15m02hfsg`, `syn30hfsg`, `syn40hfsg`
   (the `hfsg` variants) are **nonconvex** → **spatial McCormick B&B**. The family spans both
   paths.
2. **"McCormick strengthening is not the lever" — FALSE.** It followed only from (1). The
   three nonconvex instances run the McCormick relaxation, and that is exactly where the root
   bound is worst (+124 %/+955 %/+3041 %) while the incumbent is already at/near optimal.
   Relaxation strengthening is back on the table for the `hfsg` subfamily — as
   `scip-gap-closing-plan.md` originally had it, before round 1 removed it.
3. **"The live levers are NLP-BB throughput (#268) + primal/LNS incumbent quality (#276)"
   — the primal half is FALSE.** The primal side is the healthy one (0 % deficit on 2/7).
   Throughput is genuinely poor but is a *secondary* multiplier, not the lever.
4. **F-1 stands as a falsification** (auto-routing convex MINLPs to OA remains killed), but
   its closing sentence — naming throughput + primal/LNS as "the productive levers" — is
   superseded by R2-1/R2-2.

### R2-4 The issue's 5 s panel measures root latency, not search

At the issue's 5 s budget, **6 of 7 return no incumbent at all** (`root_time` 5.0–21.2 s ≥ the
whole budget; 3 nodes). The issue's 5 s table (e.g. `rsyn0805m` obj 1025.85) does **not**
reproduce here on `0.6.1.dev0`. A 120 s trace shows discopt's first incumbent on `rsyn0805m`
is 1025.845 — the issue's exact value — but it does not arrive until **t ≈ 8.5 s**. So the
issue's panel and this one differ in *when* the root finishes, not in what it finds. This is
either slower hardware or a root-latency regression since `0.4.1.dev0`; **not established
here**, and worth its own bisect before anyone reads the 5 s column as a search result.

### R2-5 Where the wall clock actually goes (`rsyn0805m`, 60 s, cProfile)

`python_time` is **not** overhead. It is computed as `wall − rust − jax` (`solver.py:10557`)
while `jax_time` wraps only the *batch node NLP* (`solver.py:10008`–`10154`) — so every
primal-heuristic NLP solve lands in the residual and reads as "Python time". Profiled:

| frame | calls | cum |
|---|---|---|
| `_solve_nlp_bb` | — | 57.7 s |
| `nlp_pounce.solve_nlp` | 353 | 30.6 s |
| `primal_heuristics.local_branching` | 3 | 18.3 s |
| `primal_heuristics.diving` | 8 | 18.2 s |
| `primal_heuristics.rins` | 7 | 17.0 s |
| `PyModelRepr.presolve` (tottime) | 4 | 10.5 s |
| `_tighten_node_bounds_with_status` | 79 | 16.4 s |

The improver heuristics dominate `_solve_nlp_bb`, and `presolve` at 2.6 s/call explains the
8.5 s root latency. Given R2-1 — the incumbent is the *healthy* half — this budget is being
spent on the wrong side of the gap.

### R2-7 (H1, tested) — the improver contingent is mis-calibrated on this family

**H1:** the improver-role LNS heuristics (RINS + local branching) consume the bulk of the
NLP-BB budget and still leave the incumbent far from the optimum; capping them converts
budget into nodes and a tighter dual bound. **Test:** A = default (`DISCOPT_HEUR_QUOT=0.5`)
vs B = `DISCOPT_HEUR_QUOT=0` (improvers blocked once an incumbent exists — an existing knob,
no code change), 60 s, all seven. **Kill if** B does not raise nodes AND tighten the bound,
or if B's incumbent regresses enough to lose the gap.

| instance | nodes A→B | dual excess A→B | primal deficit A→B | **reported gap A→B** |
|---|---|---|---|---|
| `rsyn0805m` | 319 → **1247** | +30.1 % → **+13.3 %** | +13.9 % → +20.9 % | 43.9 % → **34.1 %** |
| `rsyn0810m` | 223 → **1119** | +43.9 % → **+33.4 %** | +10.0 % → +19.4 % | 53.9 % → **52.8 %** |
| `rsyn0815m` | 351 → **607** | +47.8 % → **+44.9 %** | +17.5 % → +18.0 % | 65.4 % → **62.9 %** |
| `syn40m` | 415 → **1535** | +1732 % → **+1619 %** | +54.3 % → +65.8 % | 1786 % → **1684 %** |
| `syn30hfsg` | 281 → **623** | +919 % → **+895 %** | 0.0 % → **0.0 %** | 919 % → **895 %** |
| `syn40hfsg` | 251 → **501** | +2638 % → **+2589 %** | +4.8 % → **+4.8 %** | 2642 % → **2594 %** |
| `syn15m02hfsg` | 459 → 427 | +123 % → +123 % | 0.0 % → 0.0 % | 123 % → 123 % (tie) |

**Verdict: H1 CONFIRMED, but it is a trade-off, not a free win.** B gives more nodes on 6/7,
a tighter bound on 6/7, and a **smaller reported gap on 6/7 (A wins 0/7)** — the bound gain
outweighs the incumbent loss every time it trades. On the `hfsg` trio B is *strictly* better
(incumbent already optimal, so there is nothing to lose). But B's incumbent genuinely
regresses on the convex half (`rsyn0805m` 1116.5 → 1025.8).

Sharpest datum: **`rsyn0805m` with improvers off reaches at 60 s the exact bound (1467.995)
and node count (1247 vs 1245) the default reaches at 120 s.** The default is spending ~half
its wall clock on improvers. (B's incumbent, 1025.845, is precisely the value this issue
reports — it is the root diving incumbent, i.e. what you get with no improver on top.)

**This is NOT a "turn the improvers off" recommendation, and must not be read as one.**
The governor exists *because* of the mirror-image failure: #347 (`clay0303hfsg`) had the
improvers starving the tree, and #321 ported them in without the contingent. Turning them off
globally would re-break the families they were built for. The measurement instead indicts the
**calibration**, and that is H2:

> **H2 (untested, general — now [#704](https://github.com/jkitchin/discopt/issues/704)):** `_improver_allowed` (`solver.py:9844-9855`) charges *fixed
> abstract cost units* (`_HEUR_COST = {"rins": 5.0, "lbranch": 10.0}`) against a
> node-proportional contingent — it never measures wall time. On syn/rsyn one local-branching
> call costs ≈ 6 s (profile: 3 calls / 18.3 s), so ~10 improver calls burn ~35 s before a
> node-count-based contingent can throttle them; and the gate is bypassed entirely before the
> first incumbent (`tree.incumbent() is None → return True`), which on `rsyn0805m` is the
> first 8.5 s. A **wall-time-denominated** contingent (charge measured seconds, cap as a
> fraction of remaining budget) would self-calibrate across families instead of assuming every
> sub-MIP costs the same. Entry experiment before any build; then the Regime-2 panel gate.

**Caveat:** 7 instances, contended box, single run per arm, no variance estimate. Enough to
re-point the diagnosis; **not** enough to move a default. `DISCOPT_HEUR_QUOT` is an existing
knob, so nothing here required a code change.

---

## Measurements (round 1)

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

## Next experiments (round 1's list — superseded by §R2-6 below; kept for the record)

Draw `rsyn0805m/0810m/0815m`, `syn15m02hfsg`, `syn30hfsg`, `syn40hfsg`, `syn40m` from the
MINLPLib snapshot (`minlplib.solu` as oracle). For each:

1. **Attribute the gap.** *(DONE in §R2-1 — verdict: dual-dominated 7/7.)*
2. **OA as a bounded primal heuristic**, not a router: run OA for a small iteration cap to
   harvest incumbents, inject into the default tree, keep the NLP-BB dual bound. Kill if it
   does not improve the incumbent within a fixed fraction of budget.
   *(§R2-1 makes this low-value: the incumbent is the healthy half of the gap. Deprioritized,
   not killed — it would still help `syn40m` (+51 % deficit).)*
3. **Graduation** for any bound/primal-changing change follows the CLAUDE.md Regime-2 panel
   gate (flag default-OFF; `incorrect_count = 0`; no bound above oracle; incumbents
   independently feasibility-verified) before default-ON. *(Unchanged; still binding.)*

Round 1's kill criterion for this track was: *"if measured NLP-BB node throughput and
first-incumbent latency on the real instances are already competitive and only the bound lags,
the work collapses to cut/OBBT strengthening and #282 should be re-pointed at that."*
**It has fired, on the bound half.** The bound lags badly (§R2-2: root 4×–500× looser than
SCIP). Throughput/first-incumbent latency are *not* competitive (root 8–21 s vs SCIP's 0.5–1.6 s
whole solve), so this is a partial fire — but the dominant term is unambiguous.

---

## §R2-6 — re-pointed plan for #282

**#282 is a root-relaxation problem.** Re-point it from "search gap / primal incumbents"
(#276) to **root dual-bound strengthening**, split by subfamily:

1. **Nonconvex `hfsg` (spatial McCormick).** Root bound +124 %/+955 %/+3041 % vs SCIP's
   +6.9 %/+43 %/+282 %, with the incumbent already optimal on 2/3. Highest-value, clearest
   target. Entry experiment: root-bound-only A/B of the existing reduction/cut machinery
   (OBBT, `nonlinear_bound_tightening`, cut families) on these three, scored purely on
   `root_bound` vs `=opt=` — no wall-clock claims. Kill if no family moves the root bound
   ≥ 10 % of the SCIP−discopt spread.
2. **Convex `rsyn*`/`syn40m` (NLP-BB).** Root bound +63 %/+72 %/+104 %/+2609 %. The NLP-BB
   node bound is the NLP objective; there is no cut loop tightening the *root*. Question to
   answer before building: does NLP-BB have a root cut/OBBT stage at all, and is `syn40m`'s
   +2609 % a different failure from `rsyn*`'s ~+70 % (its `root_time` is 4.2 s — the fastest
   root, and the worst bound)?
3. **Certification — split out to [#703](https://github.com/jkitchin/discopt/issues/703).**
   `gap_certified=False` on **14/14 runs, including the convex NLP-BB ones**, where the gap
   *should* certify. Cause is `solver.py:10027` — any untrusted (non-KKT) node in a batch
   decertifies. This is sound-conservative and correct per roadmap P0.3, but it means #282's
   convex half never emits a certificate even when the bound is valid. The open question is
   *why* those nodes are untrusted.
4. **Root latency (feeds #268).** `presolve` at 2.6 s/call × 4 = 10.5 s dominates the
   8–21 s root. At the issue's 5 s budget this is the *entire* story (§R2-4).
5. **Improver contingent calibration — split out to
   [#704](https://github.com/jkitchin/discopt/issues/704)** (H2, §R2-7). Measured: improvers
   cost ~half the wall clock on this family and the budget is better spent on the tree (gap
   smaller on 6/7 with them off; A wins 0/7). The general fix is a **wall-time-denominated**
   contingent, not a family switch — the current one charges fixed abstract units and cannot
   see that one `lbranch` call costs 6 s here. Entry experiment first, then the Regime-2 panel.

**Not the lever (do not re-walk):** OA auto-routing (F-1), McCormick strengthening for the
*convex* half (it doesn't run there), primal/LNS incumbent quality as the headline (§R2-1).
