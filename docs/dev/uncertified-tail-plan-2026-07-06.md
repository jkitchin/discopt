# Closing the uncertified tail — branch-and-reduce revisit + bound-responsiveness plan (2026-07-06)

**Status:** planned (no task started)
**Basis:** `docs/dev/perf-followup-results-2026-07-06.md` (V1: the tail that
F1–F7 did not move), `docs/dev/certification-gap-plan.md` §6 + §14 "Phase 2
tasks" (the branch-and-reduce spec and its **T2.1 NO-GO — provisional**
verdict), `docs/dev/bottleneck-profile-2026-07-05.md` §3/§5 (the pinned-bound
diagnosis and the F5/F6 falsifications). Every task cites a measurement; every
premise carries the experiment that would kill it.
**Executor:** a fresh Claude (Opus) session with no prior context. Everything
needed is in this doc, the docs above, and CLAUDE.md.
**Relationship to certification-gap-plan.md:** this plan *is* the Phase-2
revisit that T2.1's verdict called for, plus the new levers the 2026-07-05/06
evidence exposed. Cert-gap-plan §0 (implementation contract) binds here in
full; §14's T2.3–T2.5 specs are incorporated by reference, not duplicated.
When work lands, update cert-gap-plan §0.8 and this doc in the same PR.

---

## §0 Binding contract (additions to cert-gap-plan §0 — read that first)

0.1 **Everything here is bound-changing or tree-changing.** Unless a task says
    otherwise: feature flag, default OFF; differential bound test
    (`utils/soundness.py::assert_bound_sound` — new bound ≥ old AND ≤ oracle)
    + feasible-point sampling per family; `incorrect_count = 0` on the full
    panel; adversarial suite green; **3 consecutive green nightlies before any
    default-on flip**. A tightening/branching change that cuts a feasible
    point better than the cutoff blocks its whole track.

0.2 **The tail is the target, the panel is the gate.** Wins must be measured
    on the named tail instances but *gated* panel-wide: no instance that
    certifies today may lose certification; no-offtarget wall regression
    ≤ 1.05; certified objectives unchanged. Named instances are probes, not
    tuning targets (no instance-keyed code, ever).

0.3 **Entry experiment before implementation, always.** Four of the last six
    perf tasks were re-scoped or killed by their entry experiments (F1-tls2,
    F3, F5, F6). That is the system working. Run the experiment, honor the
    kill criterion, record falsifications in the source doc's §5-style before
    re-scoping.

0.4 **Do not relitigate recorded negatives** (§4 below is the consolidated
    list). If new evidence contradicts one, the reopen must cite the specific
    new measurement (cert-gap-plan §0.4: measurement beats plan) and state a
    kill criterion — exactly as R3 does for the §12 branching exclusion.

0.5 **Tooling:** `discopt_benchmarks/scripts/profile_instance.py` for
    wall/phase; `scripts/t21_root_loop_replay.py` for the reduce-loop replay
    (house style already established); `scripts/check_cert_neutrality.py` for
    neutrality; oracle = `docs/dev/data/cert-optima.json` / `minlplib.solu`.
    The per-node engine is post-F1–F7 `main` — verify release POUNCE (~5 MB
    `_pounce*.so`) before trusting any timing (the pounce#182 lesson).

---

## §1 The target, split by evidence

V1 (2026-07-06, 61 instances, 60 s): **19 uncertified**, in two classes.

**Class P — BARON-instant, discopt-stuck (the primary target, 6 instances):**

| instance | V1 status / gap | BARON | T2.1 root-loop response | known root cause (measured) |
|---|---|---|---|---|
| tanksize | feasible, 32 % | 2.7 s | **69.3 %** gap reduction | reduction-responsive (T2.1) |
| nvs05 | feasible, 75 % | 0.5 s | **32.5 %** | partially reduction-responsive; residual = product relaxation (87.7 % root gap) |
| st_e36 | feasible, 24 % | 0.09 s | **0.0 %** | zero-spanning factor `f1 = x0²−6x0−11+0.8x1 ∋ 0` → product relaxation spans 0 for every x1-box; bound responds only to x0 (F5 kill analysis: x0→[5,5.5] gives root −246.04 ≈ optimum) |
| tls2 | feasible | 0.03 s | 6.9 % | root bound climbs steadily (0.72→5.2/5.3) — throughput+strength mix |
| nvs09 | feasible, 34 % | 0.02 s | n/a (not in T2.1 panel) | 69 % root gap, bound climbs slowly; separation now cheap (F3) |
| hda | time_limit, no bound | 0.2 s* | n/a | 23 rows omitted from the relaxation → **no dual bound exists**; *BARON calls it infeasible (its own issue) |

**Class H — BARON-also-stuck (13 instances, secondary):** 4stufen, bchoco06/07/08,
beuster, casctanks, contvar, heatexch_gen1/2/3, tspn08/10/12. BARON hits the
same 60 s wall (casctanks: T2.1 response 2.5 % — reduction-resistant). These
are genuinely hard; this plan treats them as *beneficiaries, not gates* — any
Class-P lever is measured on them too, but no task's kill criterion depends on
them.

**Why "branch-and-reduce" is three tracks, not one.** T2.1 proved the pure
reduce-loop moves only part of Class P (tanksize, nvs05). The F5 kill proved
st_e36's stall is not an envelope defect but a *zero-spanning-factor product
relaxation* + *branching that never touches the responsive variable*. BARON's
branch-and-reduce is the loop **plus** bound-aware spatial branching **plus**
factorable lifting — R1/R2 cover the loop, R3/R4 cover the rest.

---

## §2 Ordering

```
R1 (T2.1-revisit: complete replay on the cheap engine)   — entry gate for R2
R3a (bound-responsiveness fingerprint)                    — entry gate for R3b/R4, independent of R1
   │
R2 (root fixpoint loop T2.3 + reduce_node T2.4 + escalation T2.5 — per §14 spec, unlocked by R1)
R3b (bound-responsive spatial branching — flagged)
R4  (zero-spanning-factor lifting — flagged)
   │
R5 (zerohalf pre-crossover follow-on — independent, optional, targets #280)
   │
V2 (gate wiring [suites.global50]/[gates.cert2] + BARON re-run)
```

R1 and R3a are cheap experiments; run both first (parallelizable). R2, R3b,
R4 are builds conditional on their experiments. **T2.0 (the reduction-layer
correctness pre-flight) is COMPLETE** — C-15/C-16/C-20/C-21 are all `fixed` in
`correctness-issues.md`, so the §14 lock is lifted; do not redo it.

---

## §3 Tasks

### R1 — T2.1 revisit: complete the root-loop replay on the post-F1–F7 engine

- **Why reopened:** T2.1's NO-GO is explicitly *provisional* with the revisit
  condition "when the per-node engine is fast enough to complete the full
  panel cheaply" (cert-gap-plan §14, T2.1 RESULTS). F1–F4 satisfied it:
  flay03m 63.8→7.5 s, fac2 32.5→5.8 s, nvs01 22.5→5.5 s; 7 of T2.1's 20
  panel instances could not even run in budget then — all are cheap now.
- **Procedure:** re-run `scripts/t21_root_loop_replay.py` (the existing
  harness; same stage order S1 presolve → S2 `fbbt_with_cutoff` → S3
  `obbt_tighten_root(incumbent_cutoff=…)` → S4 envelope-rebuild +
  re-separation) on: (i) the full original 20-instance panel — this time to
  completion; (ii) the current Class-P tail (add nvs09, tanksize, hda; drop
  rows that now certify). Keep the inline `assert_bound_sound` after every
  stage (any violation = P0 stop). Record the per-stage marginal table.
- **What changed since the NO-GO (check each):** T2.2's warm OBBT probes cut
  S3's cost; F2's stall guard removes the 8.7 s simplex tail from S4's
  re-separation; A1/A2 fixed the bound bookkeeping the replay reads; C-15/16/
  20/21 fixed the passes the loop iterates. The *response* percentages may
  also change, not just the cost — that is the point of re-running rather
  than reusing the 2026-07-03 table.
- **Verdicts to produce (sharper than the original single median):**
  1. Per-instance classification: *reduction-responsive* (relative root-gap
     reduction ≥ 25 %), *marginal* (5–25 %), *reduction-resistant* (< 5 %).
  2. Per-stage include/exclude list for R2's loop (a stage < 5 % marginal on
     every instance is excluded — same rule as before).
  3. A calibrated loop budget (fraction of time limit, per stage).
- **Decision rule for R2 (replaces the old median-only kill):** build R2 if
  the reduction-responsive class contains **≥ 3 instances of the current
  Class-P tail** OR **≥ 20 % of a 30-instance out-of-panel MINLPLib sample**
  (drawn from `problems_small.txt`, oracle `minlplib.solu` — the §0.2
  generality check). If only tanksize/nvs05 respond and nothing out-of-panel
  does, the loop is a 2-instance feature → do NOT build R2; record the final
  (non-provisional) NO-GO in cert-gap-plan §14 and skip to R3/R4.
- **Effort:** ~1–2 days (the harness exists; the engine is fast now).

### R2 — Conditional build: root fixpoint loop + per-node reduction (T2.3–T2.5 per the §14 spec)

- **Unlock:** R1's decision rule, recorded in cert-gap-plan §14 under a
  "T2.1-revisit RESULTS" block. Do not start before that block exists.
- **Spec:** cert-gap-plan §14 **T2.3** (root fixpoint loop —
  `_jax/root_reduce.py::run_root_fixpoint`, integration at end of iteration 0
  post-root-heuristics, flags `root_fixpoint`/`DISCOPT_ROOT_FIXPOINT`
  default-OFF), **T2.4** (per-node `reduce_node()` — (a) expose
  `dual`/`col_status`/`safe_bound` on `MccormickLPResult` [bound-neutral
  prerequisite, assert exact neutrality], (b) unify cutoff-FBBT + free DBBT
  from the just-solved node LP + integer RC-fixing [the C-15 rule: **z =
  safe_bound, never the raw LP objective**], (c) feed tightened boxes to the
  incremental patch + child export), **T2.5** (width × |reduced-cost| top-k
  OBBT scoring replacing the model-class gate). The anchors, tests, and
  regime in §14 are current — follow them verbatim; do not re-derive.
- **Additions to the §14 spec from this plan:**
  - Stage list and budgets come from R1's table, not the 2026-07-03 one.
  - The T2.4 differential tests should reuse F5's box-containment sampling
    pattern (`assert no sampled feasible point better than the cutoff is
    excluded`) — the machinery exists from the F-series.
  - Acceptance: the R1-responsive instances certify (tanksize) or halve
    their V1 gap (nvs05) at 60 s with the flags ON; panel-wide §0.2 gates.
- **Effort:** T2.3 ~1–2 EW, T2.4 ~1–2 EW, T2.5 ~3–5 d (per §14).

### R3 — Bound-responsive spatial branching (the F5-derived lever)

**R3a — Entry experiment: box-shrink responsiveness fingerprint (~2 days).**
- **Evidence justifying a §12 reopen:** cert-gap-plan §12 excludes
  "branching-rule work — measured ≤ 15 % (SCIP ablation)". That measurement
  was about *cut/presolve-style ablations on integer instances*; the F5 kill
  produced direct contrary evidence for the pinned class: on st_e36 the root
  bound is **bit-identical for every x1-box** (the branched variable) but
  jumps −304.5 → **−246.04 (≈ the optimum)** when x0's box shrinks to
  [5.0,5.5] — the tree branches 897 nodes on the wrong variable while the
  responsive one is never split. Per §0.4, measurement beats plan; this is a
  narrowly-scoped reopen with a kill criterion, not a general branching
  project.
- **Procedure:** for each Class-P instance (and 5 R1-resistant Class-H ones),
  compute a *responsiveness fingerprint*: for each variable, halve its box
  (both halves) and measure the root-bound movement of the better half —
  `score(v) = |Δroot_bound(v)|`. Then instrument one 60 s solve and record
  which variables the current policy actually branches (spatial candidates =
  most-fractional / pseudocost per `bnb/branching.rs` + hints).
- **Deliverable:** a table per instance: top-3 responsive variables vs top-3
  actually-branched variables, and the overlap.
- **KILL CRITERION:** if on ≥ 4 of the 6 Class-P instances the current policy
  already branches the responsive variables (overlap ≥ 2/3), selection is not
  the lever — record and skip R3b. Also kill R3b per-instance-class if
  responsiveness is flat (no variable moves the bound > 1 % — that class is
  purely relaxation-limited → R4 territory).

#### R3a RESULTS / VERDICT (2026-07-06)

**Harness:** `discopt_benchmarks/scripts/r3a_responsiveness_fingerprint.py`
(reuses `t21_root_loop_replay.root_lp_bound` for the fingerprint; branching read
from a new behavior-neutral Rust counter `PyTreeManager.branch_var_counts()`
armed via `solver._R3A_BRANCH_COUNT_SINK`). Release POUNCE verified
(`_pounce.abi3.so` = 4.7 MB). Both measurements are taken in the *reformed* flat
variable space (factorable lift appends `_fr_aux_*` after the originals), so
`score(v)` and branch-frequency index the same columns. Instrumentation
neutrality confirmed: st_e36 node count / objective / bound bit-identical with
the sink on vs off (87 nodes, −246.0, −304.49044273); `cargo test -p
discopt-core` bnb suite 75/75 green.

**`score(v) = |root_bound(better half) − root_bound(full box)|`** (internally-
minimized sense, so the "better" half is the one that *raises* the root bound).
`maxΔ%` = max `score` relative to `max(1, |root_bound|)`. `flat` = `maxΔ% < 1 %`.
One 60 s instrumented solve per instance for the branch counts.

| instance | cls | root LP bound | top-3 responsive (by score) | top-3 branched (by freq) | overlap | maxΔ% | flat? |
|---|---|---:|---|---|:--:|--:|:--:|
| st_e36 | P | −304.5 | **x0**, _fr_aux_0, x1 | **x0**, _fr_aux_0, _fr_aux_1 | **2/3** | 24.8 % | no |
| nvs05 | P | 0.675 | _fr_aux_3, x0, _fr_aux_0 | x0, x1, x3 | 1/3 | (near-0 bound) | no |
| nvs09 | P | −72.9 | _fr_aux_1, x2, x0 | x2, x3, x1 | 1/3 | 38.3 % | no |
| tanksize | P | **none** | — | x0, x1, x2 | 0/3 | 0.0 % | **YES** |
| tls2 | P | ≈0 | x5, x4, x15 | x31, x35, x24 | **0/3** | (near-0 bound) | no |
| hda | P | **none** | — | x719, x709 | 0/3 | 0.0 % | **YES** |
| casctanks | H | **none** | — | x244, x245, x243 | 0/3 | 0.0 % | **YES** |
| 4stufen | H | 10 604 | x129, x130, x131 | x114, x117, x111 | 0/3 | 17.0 % | no |
| beuster | H | 5 942 | x156, x155, x154 | x122, x132, x131 | 0/3 | 31.9 % | no |
| heatexch_gen1 | H | **none** | — | x107, x104, x100 | 0/3 | 0.0 % | **YES** |
| bchoco06 | H | **none** | — | x9, x8, x0 | 0/3 | 0.0 % | **YES** |

(For nvs05/tls2 the root LP bound is ≈0, so the `maxΔ%` denominator `max(1,|b|)`
inflates the relative move; the **absolute** score is large — nvs05 `_fr_aux_3`
score 288, tls2 x5 score 5.66 — so neither is flat. `flat` is driven entirely by
the **no-root-bound** instances, not by any real bound that fails to move.)

**Two distinct Class-P sub-classes, not one:**

1. **Responsive-but-under-branched — the R3b lever (nvs05, nvs09, tls2).** A root
   bound exists, responsive variables exist, but the top responsive variables are
   *not* the top-branched. **tls2 is the clearest case in the whole panel:**
   overlap **0/3** — its most-responsive variables `x5`/`x4` (score 5.66/5.01)
   are branched **2/2** times, while its three most-branched columns `x31`/`x35`/
   `x24` (123/98/87 branches) have **score exactly 0** — the policy spends its
   entire branching budget on variables that provably do not move the root bound.
   nvs05/nvs09 show the same shape more mildly (the single most-responsive column
   is a lifted `_fr_aux_*` that is **never branched** — nvs05 `_fr_aux_3` score
   288, 0 branches; nvs09 `_fr_aux_1` score 27.9, 0 branches).

2. **Relaxation-limited / no-root-bound — R4 territory, not R3b (tanksize, hda).**
   The cold McCormick relaxation produces **no LP bound at all** (the AMP
   constraint-omission class — hda's 23 unlinearizable rows per bottleneck-
   profile §3; tanksize's `4243.28/(x0·x1)` and other non-constant divisions).
   Responsiveness is *undefined* (there is no bound for box-halving to move), so
   these are flagged **flat → R4**, exactly as the kill criterion prescribes.
   Branching selection cannot be their lever; they need relaxation coverage
   (R4 lifting), not a branching score.

**st_e36 — the F5 prediction is REFUTED (measurement beats plan, §0.4).** F5
(bottleneck-profile §3/§5) predicted x0 responsive / x1 branched — "the tree
branches 897 nodes on x1 while the responsive x0 is never split." On the
post-F1–F7 engine this **does not reproduce**: x0 is the top responsive variable
(score 75.6) **and** the top-branched variable (**267** branches vs x1's 14).
The tree already spends most of its budget on the responsive variable; the
overlap is **2/3** (x0 + `_fr_aux_0`, the 2nd-most-responsive lifted column,
which is now also branched, 56×). So st_e36 is *not* a selection-limited
instance — its residual gap is the zero-spanning-factor product relaxation (R4),
consistent with F5's own re-scoping conclusion. This is the one Class-P instance
that "already branches the responsive vars."

**Class-H (context only, not a gate):** 4stufen/beuster are responsive-but-
under-branched (overlap 0/3; responsive `x129`.../`x156`... never branched) — R3b
beneficiaries. casctanks/heatexch_gen1/bchoco06 are no-root-bound / flat → R4.

**KILL-CRITERION EVALUATION.** Class-P instances already branching the responsive
variables (overlap ≥ 2/3): **1 of 6** (st_e36 only). SKIP requires ≥ 4. Not met.
Relaxation-limited (flat) Class-P instances flagged for **R4**: **tanksize, hda**
(both no-root-bound). The three Class-P instances with a live, responsive-but-
under-branched signal are **nvs05, nvs09, tls2**.

> **VERDICT: BUILD R3b.** Selection *is* a lever on nvs05/nvs09/tls2 — the
> current policy demonstrably branches non-responsive columns while the
> responsive ones (including lifted `_fr_aux_*` products) go untouched (tls2:
> the top-3 branched have score 0; nvs05/nvs09: the top-responsive `_fr_aux_*` is
> never branched). Scope R3b to *raise the branching priority of high-`score`
> columns, including lifted aux variables* (which R4 also needs branchable — the
> two tasks reinforce). Exclude tanksize/hda from R3b's acceptance (no bound to
> respond to → R4). st_e36 already branches its responsive variable, so R3b is
> not expected to help it; its lever is R4 (zero-spanning-factor lifting). The
> §12 branching-exclusion reopen is thereby *narrowed*, not broadened: the win is
> confined to the pinned/loose-product class R3 was scoped for, with a live
> per-instance signal, not a general branching-rule project.

**R3b — Conditional build: responsiveness-aware spatial branching score
(flagged).**
- **Mechanism:** blend a bound-responsiveness term into the *spatial*
  branching score only (integer branching untouched): candidates get
  `score = pseudocost_score × (1 + λ·normalized_responsiveness)`, where
  responsiveness is estimated **cheaply from existing per-node data** — the
  interval-arithmetic bound's partial sensitivity to each variable's width
  (the `_compute_interval_bound` machinery already walks the DAG per node) or
  a periodic strong-branch-style probe on the top-k width×activity variables
  (reuse F2's guarded warm simplex). NO per-instance tuning; one λ, one k,
  calibrated on the R3a table and validated out-of-panel.
- **Regime:** tree-policy change — soundness is untouched (branching any
  variable is always valid), but node counts change by design, so:
  certified objectives unchanged panel-wide; `incorrect_count = 0`; the
  no-offtarget wall gate (≤ 1.05); flag `DISCOPT_RESPONSIVE_BRANCHING`
  default-OFF until nightly-green. Regression test: a synthetic model with a
  pinned product bound (an st_e36-shaped 2-var model built in the test, not
  the named instance) certifies in ≤ 10 nodes with the flag ON and does not
  with it OFF.
- **Acceptance:** ≥ 2 Class-P instances certify at 60 s (F5's data predicts
  st_e36 in ~5 nodes if x0 is branched); no panel regression.
- **Effort:** ~1–2 EW.

### R4 — Zero-spanning-factor lifting (factorable-reformulation extension, flagged)

- **Evidence:** F5's root-cause on st_e36 — `C0 = f1·(positive expr)` with
  `f1 ∋ 0` on every box, so the product's McCormick interval spans 0
  regardless of the factors' envelope quality; and nvs05/nvs09's 70–88 % root
  gaps in the same loose-product family. The factorable reformulation
  already lifts *some* structures (relaxation-catalog §3); the gap is
  products whose nonlinear factor is not lifted to a bounded auxiliary.
- **Mechanism:** when a product's factor is a non-atomic expression `f(x)`
  whose FBBT interval spans 0, lift `w = f(x)` to an auxiliary variable with
  its FBBT-tightened bounds, relax the product as McCormick on `(w, g)`, and
  let FBBT/branching act on `w` directly (BARON's standard factorable move —
  branching on `w` splits the zero-spanning interval, which un-pins the
  product bound; combined with R3b, `w` becomes a branchable, responsive
  candidate).
- **Entry experiment (~2–3 days):** hand-lift st_e36's `f1` (construct the
  lifted model via the modeling API in a script) and measure: root bound on
  the original box; root bound after branching `w` once at 0; nodes to
  certify. **KILL CRITERION:** if the hand-lifted model's bound does not
  strictly improve after one `w`-split (i.e. the pinning survives lifting),
  the mechanism is wrong — record and stop. Also check the relaxation catalog
  first (§0.4 of this doc): if the lifting path exists and is merely gated,
  the task is re-gating, not building.
- **Regime:** bound-changing (a reformulation changes the relaxation):
  flag `DISCOPT_LIFT_ZERO_SPANNING_FACTORS` default-OFF; differential bound
  test (lifted root bound ≥ unlifted, ≤ oracle) + feasible-point sampling
  (the lifted model must not cut any original-feasible point — test maps
  points through `w = f(x)`); `incorrect_count = 0` panel-wide with the flag
  ON; byte-identical with it OFF. Out-of-panel witnesses: ≥ 2 instances with
  zero-spanning nonlinear factors found by a corpus scan (the F5 methodology).
- **Acceptance:** st_e36-class certifies at 60 s with flag ON (+R3b);
  nvs05/nvs09 root gap reduced ≥ 25 %; no panel regression.
- **Effort:** ~1–2 EW.
- **Note on hda:** hda's problem is different — 23 constraints *omitted* from
  the relaxation entirely ("cannot linearize log(<general expr>)/x**expr"),
  so no bound exists at all. If R4's lifting machinery naturally covers
  those rows (lift the log/power argument, relax over the aux), extend it to
  them — that would give hda its first dual bound. If not, file the
  relaxation-coverage gap as its own issue; do not force it into R4.

### R5 — Zerohalf at a pre-crossover point (optional, parallel; targets #280 graphpart)

- **Basis:** cert-gap-plan §7/§0.8 — a validity-GREEN native {0,½}-CG
  separator exists on branch `cert-p3-zerohalf` (PR #427 closed unmerged, no
  dead flag on main). It was INERT because discopt's root LP optimum is a
  ⅓-partition vertex where every {0,½} combination is tight-not-violated;
  the recorded follow-on is to **separate at a ½-valued / pre-crossover
  interior point** instead of the vertex.
- **Entry experiment:** on 3 graphpart instances (full corpus), take the
  parked separator and separate at (a) the analytic center / a few IPM
  iterations before convergence, (b) a convex combination of vertex and
  center. Measure violated-cut count and root-gap-closed vs the recorded 0.
  **KILL CRITERION:** if no interior point yields violated {0,½} cuts with
  positive gap closure, the family is dead for this geometry — close #280's
  cut angle and record.
- **Regime:** cut validity is point-independent (the separator's validity
  tests are already GREEN) — but re-run `assert_cut_valid` sampling anyway;
  flag default-OFF; panel gates as §0.1.
- **Effort:** ~3–5 d. Independent of R1–R4; schedule opportunistically.

### V2 — Gate wiring + the measurement of record

1. **Wire the long-missing suite/gate** (cert-gap-plan §14 T2.6 items 1–2):
   `[suites.global50]` in `benchmarks.toml` + `[gates.cert2]`
   (`root_gap_ratio_vs_baron ≤ 1.3` — computed from A3's now-live bound
   fields; `closed_within_10_nodes_fraction` metric + dispatch). Re-run the
   BARON comparison for reference root bounds where coverage is thin.
2. **Flag flips** per §0.1: each of R2/R3b/R4 individually green → jointly on
   nightlies → 3 consecutive green → default-on → rebuild
   `cert-baseline.jsonl` (node counts legitimately change) → update
   cert-gap-plan §0.8.
3. **The measurement:** re-run
   `discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py --time-limit 60`
   against the 2026-07-06 V1 baseline (`reports/…2026-07-06T03-25-20.json`).
   - **Hard gates:** 0 violations; no certification lost; time-limit contract
     holds; bound-vs-oracle clean.
   - **Targets (falsifiable, from the per-task acceptances):** proved-optimal
     42 → **≥ 46** (st_e36, tanksize, +2 of nvs05/nvs09/tls2); uncertified
     tail 19 → ≤ 15; hda reports a finite dual bound. Class H may not move —
     say so honestly if it doesn't.

---

## §4 Do-not-relitigate (consolidated binding negatives)

- **OBBT-on-aux / `cascade_aux`** — measured dead (perf-plan §6; nvs22 fails
  to certify with it on). Not a loop stage.
- **gear4** — reduction-resistant (2.46 M box after cutoff-OBBT+probing to a
  fixpoint). Never a gate probe.
- **New envelope math for the pinned class** — F5 kill: the composite
  even-power envelopes are already exact; st_e36's defect is the product
  relaxation + branching, not envelopes. (R4 lifts factors — a reformulation,
  not new envelope math.)
- **Tree-warm-starting node NLPs** — F6 kill: iteration-neutral-to-worse
  (tls2 0.58×); release POUNCE ≈ 1.1× Ipopt (pounce#182 resolved). Node
  *speed* is not the tail's limiter.
- **Per-node probing** — deliberately unplanned (§14 table); in-tree cost
  unproven.
- **The 2026-07-03 T2.1 stage table** — superseded by R1's re-run when it
  lands; until then T2.3–T2.5 stay locked.
- **Phase 3 cut plumbing** (1c NO-GO: reachable+armed cuts close ~0 %) and
  **zerohalf at the vertex** (inert at the ⅓-partition vertex) — R5's
  interior-point variant is the only live cut thread for graphpart.
- **V-segments / symmetry** — de-scoped by the Phase 4 re-profile (0 defvars
  in text `.nl`; 0 orbits).
- **`python_time`/`rust_time` result fields** — accounting artifacts; use the
  T0.3 timers.

## §5 Effort summary

R1 ~1–2 d; R3a ~2 d (parallel with R1); R2 ~2.5–4.5 EW (conditional);
R3b ~1–2 EW (conditional); R4 ~1–2 EW (entry experiment ~2–3 d first);
R5 ~3–5 d (optional); V2 ~2–3 d. Expected critical path if everything
unlocks: R1/R3a → (R2 ∥ R3b ∥ R4) → V2, ≈ 4–7 EW. If R1 confirms the NO-GO
and only R3/R4 build: ≈ 2.5–4.5 EW.
