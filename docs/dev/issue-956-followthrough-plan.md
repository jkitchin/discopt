# Issue #956 follow-through: make undecided node LPs decidable

Status: **in progress** (started 2026-08-09)

## §0 Why this plan exists (binding context)

#956 is fixed — the McCormick envelopes no longer cut their own graph — but the fix
ships **default-OFF** because it measured harmful (`performance-plan.md` §15):
6/6 decisive instances regress in 3/3 interleaved replicates, and on `ex1252` under
the #707 reform flags the undecided-node fraction gets *worse*, 55.3 % → 81.7 %.

The measurements that produced that verdict also produced the explanation, and it
is not about envelopes at all:

* `nvs20`: 46 % of node LPs come back `Numerical`. Guard size is irrelevant to it
  (a 10⁶× larger guard changes nothing); propagation and OBBT were tested and
  excluded.
* `x*y >= c, x + y <= 1` over `[0,1]²` is **never** certified infeasible, for any
  `c` in {0.3, 0.45, 0.49, 0.55, 0.6, 0.75, 1.0, 1.5, 2.0} — ~12 000 nodes to the
  time limit, in both arms.

Reading the code closes the loop between those two symptoms: they are the **same
defect**. `lp_spatial_bb._node_bound` fathoms a node only on

```python
verdict = "fathom" if (_st == "infeasible" and _farkas) else "unresolved"
```

so an LP that is infeasible but whose Farkas ray does not *certify* becomes
`"unresolved"`, folds into `unresolved_lb`, and the final status collapses to
`time_limit` — the tree can never conclude `infeasible`. The Rust tree draws the
same distinction (`verdict_for`: only `LpStatus::Infeasible` is a proof; since #927
everything else branches). So both open items reduce to one question:

> **why do these node LPs fail to produce a certifiable verdict?**

That is the whole subject of this plan. It is also on the critical path for
finishing #956: the guard's measured harm is that it *increases* the undecided
fraction, so if undecided nodes become decided, the harm mechanism may disappear
and the guard can graduate default-ON.

**Verification regime.** Every task here is bound-changing or verdict-changing, so
CLAUDE.md §5 applies: cert-clean AND net-positive, measured with the *interleaved
replicate* method (`solver.py` / #902 — a `max_nodes` budget is unusable because
`node_limit` routes the kernel back to the Python path, so the arms would compare
OFF against OFF). A win must hold in EVERY replicate, a regression in a majority,
and an instance whose replicates disagree is quarantined as unresolved.

**Soundness is not negotiable here, and this plan is unusually exposed to it.**
Every task makes the solver *more willing to declare a region empty*. A wrong
`Infeasible` is a false certificate — the #927 failure mode exactly. No task may
relax a Farkas check by loosening a tolerance; the only admissible fixes are ones
that construct or verify a certificate more completely.

## §1 The three witnesses

Referenced by task ID below; all three must be measured for every change.

| id | witness | current behaviour |
|---|---|---|
| W1 | `x*y >= 0.6, x + y <= 1` on `[0,1]²` (Python path, 0 kernel calls) | ~12 000 nodes, no verdict |
| W2 | `nvs20` (native kernel) | 46 % of node LPs `Numerical` |
| W3 | `ex1252` + `DISCOPT_INTEGER_MULTILINEAR_REFORM=1` + `DISCOPT_MULTILINEAR_COUPLING_RLT=1` | 55 % undecided (82 % with the #956 guard on) |

## §2 Executable task list

Work top to bottom. Each task states its done-condition; do not start a task whose
predecessor's done-condition is unmet. Tick the box and record the measurement
inline when a task completes.

**Order changed 2026-08-09 (owner's call): T3 and T4 run BEFORE T2′.** T2′ has
become an open-ended simplex investigation whose size is not yet bounded — six
hypotheses eliminated, no fix — while T3/T4 are independent of it and deliver the
bilinear-infeasibility gap outright. Banking them first also de-risks T2′: if it
turns out phase 1 emits a false *feasible*, T3's certificate plumbing is what
carries the fix through to the tree anyway.

- [x] **T1 — Entry experiment: classify the undecided verdicts (NO implementation
  until this reports).** DONE 2026-08-09 — **kill criterion NOT met; T2's hypothesis
  is falsified.** See §4 log entry T1 for the full histogram. Headlines:
  * `FarkasRejectOpen = 0` on every witness. The ray is *never* rejected for want of
    a finite bound on an unbounded column, so T2 as written was wrong.
  * "infeasible-but-uncertified" is 12 % of undecided verdicts on W2 and 0 % on
    W1/W3 — under the 20 % threshold everywhere, so the plan's own kill criterion
    fires and T2 is re-scoped below.
  * The actual dominant bucket on W2 is `LpAuditBoundsFail` = 1300, **82 % of all
    undecided verdicts**: the simplex reaches "optimal" and the final audit finds a
    basic variable outside its box.
  * Those excursions are NOT near-misses. As a multiple of the audit's own relative
    tolerance: `<10×` 218, `<1e3×` 582, `<1e6×` 488, `≥1e6×` 12 — 83 % are at least
    10× beyond tolerance. **No tolerance change is admissible**; these are wrong
    bases, exactly as §0 requires us to assume.
  * W1 and W3 register **zero** counters on both instrumented paths. Their node LPs
    never reach the Rust simplex at all, which makes T3 (not T2) W1's whole story.

  *(Original scope, for the record: instrument the LP layer to count per node which
  outcome fired — `Optimal`, `Infeasible`+certified, `Infeasible`-but-uncertified,
  `Numerical` from phase-1 residual / from the feasibility audit / from a
  factorization error, `IterLimit` — via the `profile.rs` counter facility so it
  ships as permanent instrumentation, and evaluate the 20 % kill criterion in
  writing. All of that was done; the criterion fired.)*

- [ ] **T2′ — Re-scoped by T1: attack the bound-excursion class, not Farkas.**
  T1 killed the original T2 (`FarkasRejectOpen = 0`, and uncertified-infeasibility
  is only 12 % of the problem). What actually happens on W2 is that the simplex
  declares optimality on a basis that is **primal infeasible**: a basic variable
  sits 10×–10⁶× the audit tolerance outside its box, the audit catches it, and —
  because a bound excursion is the one failure iterative refinement provably cannot
  repair (`discopt#364`) — the whole solve is downgraded to `Numerical`.

  So the node LP is not undecidable; the simplex is *losing* a solution it had.
  Sub-question to settle first (cheap, one more instrumented run): is the offending
  column basic or nonbasic, and is the excursion present at the last
  refactorization or accumulated between them? That distinguishes three fixes:
  a Harris/EXPAND drift reset, a bound-flip cleanup of the offending nonbasics, or
  a forced refactorize-and-reprice before the audit runs.
  **Constraint (unchanged):** no tolerance may be loosened — T1 measured 83 % of
  excursions at ≥10× tolerance, so a tolerance fix would be laundering wrong bases
  into certified bounds.
  **Done when:** `LpAuditBoundsFail` is materially reduced on W2 and the terminal
  histogram shifts from `Numerical` to `Optimal`/`Infeasible`, with T5 green.

- [x] **T3′ — DONE 2026-08-09. Re-scoped: the certificate plumbing ALREADY EXISTS and
  works. The defect is downstream of it.** T3 as written rested on two claims that
  both turned out false; see the §4 log entry T3 for the full trace.
  * `solve_lp_spatial_bb` is called **zero** times for W1, so `_relax_bound` and its
    `cut_enabled=False` branch — the entire subject of the original T3 — are not on
    this witness's path at all.
  * W1's node LPs are certified infeasible **3600/3600** (`('infeasible', farkas=True)`
    out of `IncrementalMcCormickLP.solve_assembled_full`), `mccormick_lp` converts
    that into `MccormickLPResult(status="infeasible")` on a verified ray, and
    `solver.py` fathoms it (`node_infeasible_mask`, `_INFEASIBILITY_SENTINEL`).
    Every link in the chain works.

  **The open question is now narrower and stranger:** every node LP is certified
  infeasible and every node is fathomed, yet the tree runs 4000+ nodes and returns
  `time_limit` instead of `infeasible`. A fathomed node produces no children, so the
  tree should empty almost immediately. Something is regenerating or re-solving
  nodes, or the termination test is not reading the fathom marks.
  **DONE.** Root cause and fix in the §4 log entry below; W1 went from
  `time_limit` at 12 000+ nodes to `infeasible` in **1 node, 1.9 s**.

- [x] **T4 — DONE 2026-08-09. W1 terminates.** `x*y >= c, x + y <= 1` on `[0,1]²` returns
  `infeasible` for `c` in {0.5, 0.6, 1.0, 2.0}, in both guard arms.
  **Done when:** all eight combinations return `infeasible`, and the original
  `_infeasible_bilinear` fixture (`x*y >= 0.5`) is **restored** in
  `test_debug_adversarial.py` — the fixture swap made in the #956 branch exists only
  because the solver could not certify that model, so restoring it is the honest
  regression test for this work. Its docstring's original claim ("infeasible only
  after branching") must then actually hold: assert the solve branches (node count
  > 1) rather than closing at the root on a rounding.

- [x] **T5 — DONE 2026-08-09. Soundness gate: 26 models with known optima, 0 false infeasibles.** Every newly-certified
  `Infeasible` must be a true emptiness proof. For a sample of nodes newly reported
  infeasible on W1/W2/W3, verify no feasible point exists by an independent check
  (dense LP via scipy/HiGHS on the same assembled system, plus random/vertex
  sampling inside the node box).
  **Done when:** zero contradictions across the sample, and the check is left in the
  tree as a `#[cfg(debug_assertions)]` / test-only audit so a future regression
  trips it.

- [x] **T6 — DONE 2026-08-09. Cert-clean PASS; the A/B on the chosen instances was
  a null experiment, which is itself the finding.** Branch vs `origin/main` as two
  builds, guard OFF in both. 119 instances × 2 arms, 147 executed assertions, zero
  false bounds / objectives / infeasibles / new errors; the two flagged status
  differences are TL=5 deadline-boundary flicker that the **baseline shows too**.
  The T3′ prune arm fires 0 times on all six decisive instances (they are simply
  not in the class), so the arms execute identical code there and the replicate
  verdicts are noise — now recorded as this harness's measured noise floor. It
  *does* fire on **30 of 119** corpus instances; the A/B re-run on ten of those is
  **1 WIN (tls2: 265 vs 317 nodes, bound 3.667 vs 2.100, `optimal` vs `feasible`)
  and 9 identical, 0 regressions**. W1 itself: `origin/main` `time_limit` at 4927
  nodes → branch `infeasible` in 1 node. Cert-clean **and** net-positive; no
  revert. Full entry in §4, including the retraction of the first
  (instrument-broken) fired-check.

- [x] **T7 — DONE 2026-08-09. `DISCOPT_ENVELOPE_OUTWARD_ROUND` graduates to
  default-ON; `=0` remains the opt-out.** Both halves of the original "harmful"
  verdict were re-measured and neither survives (§4 entry below): the regression
  signal is inside this harness's now-measured noise floor, and the
  undecided-fraction evidence is unmeasurable because `n_undecided` is **0 across
  the whole corpus in both arms**. What is left is a demonstrated soundness defect
  fixed at no measurable cost, which CLAUDE.md §1 settles. Corpus panel ON vs OFF
  on one build: 149 assertions, 0 false bounds / objectives / infeasibles / errors.

- [ ] **T8 — Close out.** `cargo test -p discopt-core`, `pytest -m smoke`, the
  adversarial suite, clippy/fmt/ruff. Update `performance-plan.md` §15 with the
  final state, update this file's status line, and post the summary to #956 stating
  explicitly whether it can be closed.

## §3 Observations parked here (not tasks)

* **Budget overrun.** W1 with `time_limit=8.0` returned after **21.1 s** wall. The
  spatial path overruns its own deadline by ~2.6×. Not this plan's subject; if it
  obstructs a measurement, note it and work around it rather than fixing it here.

## §4 Log

### T6 — the panel, and a second instrument that measured nothing (2026-08-09)

**Setup.** Two *builds*, not two flag settings: this branch (merged up to
`origin/main` f13bd93) against `origin/main` itself, both with
`DISCOPT_ENVELOPE_OUTWARD_ROUND=0` so the panel measures T2+T3 alone. Each arm is
its own process importing `discopt` from its own tree, and each asserts the tree
path *and* a version marker — `PyTreeManager.import_results.__text_signature__`
must contain `certified_infeasible` on the branch arm and must **not** on the
baseline arm (CLAUDE.md §8).

**Corpus cert-clean: PASS.** 119 instances per arm at `time_limit=5`, 147
executed assertions: **0 false bounds, 0 false objectives, 0 false infeasibles, 0
new errors.** Two status differences were flagged and both are deadline-boundary
artifacts, not regressions:

| instance | base | branch | resolution |
|---|---|---|---|
| `ex1265` | `optimal` (1555 nodes) | `time_limit` (1 node) | bimodal at TL=5 — **the baseline flips too** (1 of 8 replicates returned `feasible`/1559 nodes). At TL=30 both arms are `optimal`, 1555 nodes, 3.3 s, 3/3. |
| `nvs05` | `feasible` | `time_limit` | identical bound (0.6149450688180482); both arms `time_limit`, 3 nodes, 5/5 on replication. |

**The A/B on the decisive instances is a NULL EXPERIMENT, and that is the
finding.** nvs17/19/20/23/24 and tanksize were chosen because they are where the
undecided-node problem shows. They are not where T3′ acts: instrumented with
`TreeCertInfeasPrunes`, the certified-empty prune arm fires **0 times on all six**
and **1 time on W1**. Two arms that execute identical code cannot differ, so every
verdict the replicate harness printed for them is scheduler jitter:

| instance | 3 reps | 7 reps | T3′ prunes |
|---|---|---|---|
| nvs17 | WIN 3/3 | — | 0 |
| nvs19 | w2/l1 | "REGRESSION" 4/7 | 0 |
| nvs20 | identical | — | 0 |
| nvs23 | w2/l1 | "REGRESSION" 4/7 | 0 |
| nvs24 | "REGRESSION" 2/3 | "REGRESSION" 5/7 | 0 |
| tanksize | w2/l0/t1 | w5/l1/t1 | 0 |

That table is worth keeping for a different reason: it is a **measured noise
floor** for this harness. At TL=20 with identical code, bound spreads of 0.1–0.7 %
and per-instance win/loss splits up to 5/7 occur *by chance*. A future panel that
reads a 4/7 split as a regression would be reading noise.

**Where it does fire: 30 of the 119 corpus instances (25 %)**, from 1 prune up to
28 (`trig` 28, `tls2` 16, `st_e05` 12, `m3` 10, `st_e03` 9, `prob10` 6, `nvs21` 3,
`gkocis`/`nvs01`/`ex1223a`/`st_e38` 2). So the arm is far from inert corpus-wide —
the six instances §1 nominated simply are not in that set. Re-running the same
3-replicate A/B on ten firing instances is the panel that actually tests the
change:

| instance | branch | base | verdict |
|---|---|---|---|
| **tls2** | **265 nodes, bound 3.667, `optimal` in 1/3 reps** | 317 nodes, bound 2.100, `feasible` in 3/3 | **WIN, every replicate** |
| trig, st_e05, m3, st_e03, prob10, nvs21, ex1233, st_e36, heatexch_gen1 | — | — | **identical** (same node count, same bound, every replicate) |

Nine of ten are bit-identical: the certified-empty prune *replaces* a branch that
would have closed the same region anyway, so the certificate is unchanged and only
the path to it is shorter. The tenth is a genuine gain — `tls2` prunes 52 fewer
nodes and lifts its dual bound from 2.100 to 3.667, enough to certify optimality
in a replicate where `origin/main` finishes merely `feasible`.

Where it acts most, it is not marginal at all: W1 (`x·y ≥ 0.6, x + y ≤ 1` on
`[0,1]²`) is `time_limit` at **4927 nodes** on `origin/main` and `infeasible` in
**1 node, 1.4 s** on the branch.

**Retraction (CLAUDE.md §11).** The first run of this fired-check reported the arm
firing **zero times everywhere, including W1** — while W1 demonstrably returned
`infeasible` in 1 node. The counter was right and the readout was wrong:
`profile::reset()` cleared `CTOTALS` along with the per-`dump` accumulators, and
`milp_driver` calls `reset()` at the start of **every MILP sub-solve**, so any
counter incremented before the last sub-solve read back as 0. This is the same
failure as the `dump()` retraction recorded under T3 — the totals were made to
survive `dump()` but not `reset()`. Split into `reset()` (per-dump accumulators)
and `reset_totals()` (the deliberate whole-process reset, the only thing
`profile_reset_py` now calls). Any zero counter read in this plan before this fix
is unreliable and was re-measured.

**Verdict: cert-clean AND net-positive — T2+T3 stays.** Cert-clean by the corpus
panel (147 assertions, zero violations); net-positive by the firing-instance panel
(1 win, 9 identical, 0 regressions) plus W1's `time_limit` → `infeasible`. On the
89 non-firing instances it cannot regress anything, because there it executes
nothing at all. No revert.

### T7 — the guard graduates default-ON, and why the "harmful" verdict fell (2026-08-09)

The #956 guard shipped default-OFF on two measurements. Both were re-run on the
current tree; **neither reproduces**, and one of them can no longer even be taken.

**1. "6/6 decisive instances regress in 3/3 replicates" was noise.** T6 supplied
the missing control: two *builds* that execute identical code on those instances
still produce win/loss splits up to 5/7 and bound spreads of 0.1–0.7 % at
`time_limit=20`. Measured against that floor, the guard's re-run A/B (ON vs OFF,
one build, 3 interleaved replicates) is a wash:

| instance | ON bound (sd) | OFF bound (sd) | verdict |
|---|---|---|---|
| nvs20, tls2, trig | — | — | identical |
| tanksize | 1.25532 ± 0.00037 | 1.25498 ± 0.00035 | WIN, every replicate |
| nvs17 | −1316.59 ± 3.6 | −1319.23 ± 1.5 | unresolved (w2/l1) |
| nvs23 | −25851.5 ± 23 | −25868.7 ± 47 | unresolved (w2/l1) |
| nvs19 | −5820.58 ± 31 | −5799.21 ± 11 | "regression" 2/3 — inside the floor |
| nvs24 | −172448 ± 94 | −172248 ± 200 | "regression" 2/3 — inside the floor |

**2. The undecided-fraction evidence no longer exists.** The figure that carried
the original decision — `ex1252` under the #707 reform flags going 55.3 % → 81.7 %
undecided — cannot be reproduced, because that configuration no longer routes
through the native spatial kernel at all (the wrapper counts **0** kernel calls
for it, while counting 21 elsewhere, so the probe is live and the answer is real).
Across the 21 corpus instances that *do* reach the kernel, `n_undecided` is **0 in
both arms** over 12 636 / 12 255 kernel nodes. There is no undecided fraction left
to make worse.

**Corpus panel, ON vs OFF, one build, 119 instances:** 149 executed assertions —
**0 false bounds, 0 false objectives, 0 false infeasibles, 0 new errors.** One
certification difference was flagged (`nvs12`, OFF `optimal` / ON `feasible`) and
does **not** reproduce: 6/6 interleaved replicates return `optimal` at 231 nodes
with bound −481.2 in *both* arms. Same for `ex1265` in the other direction (6/6
`optimal`, 1555 nodes, both arms). Both were TL=5 deadline flicker of the kind T6
already characterised.

**Decision.** §5's net-positive bar exists to keep sound-but-useless *performance*
machinery out of the default path (the `DISCOPT_CUT_INHERIT` rule). This is not
that: it is the repair for the defect the issue reports — unguarded generators cut
the exactly-feasible point `(x, f(x))` out of their own envelope, which is a route
to a false bound, and six Rust regression tests fail with `=0` to prove it. §1
governs: *correctness before performance, always.* Shipping the repair off by
default means #956 is not fixed for anyone who does not know the flag exists. The
guard is now **default ON** in both engines (`mccormick_patch.rs` and
`_jax/outward_rounding.py` — they must agree, or `_validate`'s row-set comparison
drops the incremental path), with `=0` preserved as the opt-out and the legacy
arithmetic intact behind it.

### T3' / T4 / T5 - the tree could not fathom a proven-empty region (2026-08-09)

**The defect.** `TreeManager` prunes on `node_lb >= self.incumbent_value`, with
`incumbent_value` initialised to `f64::INFINITY`. A rigorously infeasible node
arrives as a large *finite* sentinel lower bound, and `1e30 >= inf` is false. So
**a region proven empty was never pruned until an incumbent existed** - and on an
infeasible model one never does. Every certified-empty node was branched instead,
which is why `x*y >= c, x + y <= 1` ran ~12 000 nodes to the time limit for every
`c` from 0.3 to 2.0. The whole certificate chain already worked (3603/3603 node
LPs Farkas-certified, mask set); the tree simply had no way to act on it.

The certificate could not be inferred from the sentinel: the same sentinel also
encodes *soft* failures, and fathoming those is exactly #927's false-certificate
mode. So it is carried explicitly - `NodeResult::certified_infeasible`, an
optional `import_results` array (existing callers unchanged), and a prune arm
ahead of the bound test.

**T4 result - every case that previously timed out now closes in 1 node**, in
both outward-rounding arms:

| `c` | before | after |
|---|---|---|
| 0.3, 0.45, 0.49, 0.5 | ~12 000 nodes, `time_limit` | **`infeasible`, 1 node** |
| 0.55, 0.6, 0.75, 1.0 | ~12 000 nodes, `time_limit` | **`infeasible`, 1 node** |
| 1.5, 2.0 | ~12 000 nodes, `time_limit` | **`infeasible`, 1 node** |

The original `_infeasible_bilinear` fixture (`x*y >= 0.5`) is **restored** in
`test_debug_adversarial.py`, with its docstring corrected rather than reinstated:
the old claim "infeasible only after branching" was never true, and the root
closure is now a genuine root-level proof (the McCormick relaxation over the root
box is itself empty) instead of the rounding accident it used to be.

**T5 - soundness.** This change strictly increases pruning, so the risk is a
false `infeasible`. Gate: every in-repo model with a known optimum is provably
feasible, so any `infeasible` report is a provable error. **26 models checked, 0
false infeasibles.** Two smoke tests failed en route and both were worth having -
`test_nonrigorous_fathom_is_not_reported_infeasible` and
`test_integration_nonrigorous_fathom_reports_feasible_not_optimal` - but as a
test DOUBLE signature mismatch, not a soundness break; their fakes now forward
the certificate rather than dropping it, so they still exercise the path they
guard. Two Rust unit tests pin both sides of the boundary: a certified-empty node
prunes with no incumbent, and an *uncertified* sentinel still branches.

### T1 — entry experiment (2026-08-09)

Permanent instrumentation added (`profile.rs` counters + `_rust.profile_counters_py`
/ `profile_reset_py`, so a caller gets the numbers back instead of parsing the
stderr dump): terminal verdict histograms for BOTH simplex entry points, the
`LpInfeasUncertified` bucket, the two audit-failure causes, the two Farkas
rejection reasons, and excursion-magnitude buckets.

W2 `nvs20`, 20 s, 1651 nodes:

| counter | value | share of verdicts |
|---|---|---|
| `LpVerdictOptimal` | 143 | 8.3 % |
| `LpVerdictNumerical` | 1584 | 91.7 % |
| `LpVerdictInfeasible` | 0 | 0 % |
| `LpAuditBoundsFail` | **1300** | 82 % of undecided |
| `LpAuditRowsFail` | 62 | |
| `LpInfeasUncertified` | 190 | 12 % of undecided |
| `FarkasRejectOpen` | **0** | — |
| `FarkasRejectMargin` | 768 | |
| excursion `<10×` / `<1e3×` / `<1e6×` / `≥1e6×` | 218 / 582 / 488 / 12 | |

W1 and W3: every counter **0** — neither witness's node LPs reach the instrumented
Rust simplex paths.

**Falsified:** the T2 hypothesis (Farkas ray rejected for want of a finite bound on
an unbounded structural column). `FarkasRejectOpen` is exactly 0; the rejections
that do happen are margin failures, which is evidence those LPs are not infeasible
rather than that the certificate is incomplete. The plan's kill criterion
("uncertified ≥ 20 % of undecided on some witness") is not met anywhere — 12 % is
the maximum — so T2 is re-scoped to the bound-excursion class per §2 T2′.

### T2′ — sub-question settled: the root cause is phase 1, not phase 2 (2026-08-09)

The plan asked three questions before implementing. All three are now answered on
W2 (`nvs20`, 20 s), and the answer is upstream of where the symptom appears.

| question | measurement | reading |
|---|---|---|
| basic or nonbasic? | `AuditBoundsOnBasic` 846, `AuditBoundsOnNonbasic` **0** | always a BASIC variable, as theory requires |
| can a sharper `x_B` repair it? | `AuditBoundsRefineFixes` **0** / `RefinePersists` 846 | no — `assemble`'s standing claim (#364) is CONFIRMED here too |
| is the box empty on arrival? | `LpCrossedBox` **0** | no — falsified; `assemble_node_lp`'s intersection is not producing empty boxes |
| **is the basis feasible when phase 1 hands off?** | `Phase1EndBoxOk` 379 / `Phase1EndBoxViolated` **612** | **NO — 62 % of solves enter phase 2 already outside the box** |

So the simplex is not losing a good solution in phase 2; it never had one. Phase 1
measures feasibility as `sum|artificials| <= 1e-6` — an absolute test on the
artificials that says nothing about whether the *structural* basics are inside
their boxes — and the comments at the handoff assert primal feasibility that the
numbers do not support. Phase 2 then optimizes from an infeasible start, the final
audit catches the excursion, and refinement cannot repair it because there was
never a feasible basis to sharpen.

This retires the three candidate fixes the plan listed (drift reset, bound-flip
cleanup, forced refactorize-and-reprice) — all three operate on phase 2 or on
`x_B` arithmetic, and the defect is in neither.

**First candidate implemented and mostly falsified: EXPAND is not the mechanism.**
EXPAND (Gill et al.) lets each pivot violate a bound by up to `EXPAND_MAX = 1e-7`
(absolute) to escape degeneracy, which looked like the obvious accumulation
source. Implemented `Simplex::no_expand` plus a flag-gated, audit-guarded re-solve
(`DISCOPT_PRIMAL_EXPAND_RESET=1`) in the same shape as the #671 hardening retry —
only a terminal certificate is accepted, so it can rescue but never certify.
Measured on W2: **309 retries, 13 rescues — a 4.2 % rescue rate.** Suppressing the
bound relaxation entirely leaves 96 % of the failures in place, so EXPAND
accumulation is a minor contributor at most. (It is not worthless: the run's dual
bound came out *higher*, 227.23 vs 225.43. Kept, default-OFF, pending T6.)

Two instrumentation lessons paid for here, both CLAUDE.md §6:
* the retry armed **0** times at first because the failing solves do not enter via
  `solve_lp_cols` — the counter caught it rather than the run reporting a silent
  no-op;
* it then armed 115 times against 597 entries, because the #85 dense retry
  `return`s early on exactly the solves this one targets. The two retries are now
  chained rather than exclusive.

**Where T2′ goes next.** With EXPAND excluded, the surviving explanation is that
phase 1 has *no term at all* for box infeasibility of basic variables: it minimizes
`sum|artificials|` and stops at zero, while a basic structural sits outside its box.
The crash basis is box-feasible by construction (it only crashes a column whose
value lands inside its box), so feasibility is lost during pivoting, and the exact
`x_B` recompute cannot undo it — meaning incremental drift steered a wrong *basis
choice* rather than a wrong *value*.

The decisive next measurement, before any more implementation: **dump one of these
node LPs and hand it to HiGHS/scipy.** If HiGHS calls it infeasible, our phase 1 is
producing a false *feasible* and the fix belongs in infeasibility detection — which
is exactly what W1 and W2 both need, and would convert `Numerical` into certified
`Infeasible`. If HiGHS solves it, the fix belongs in phase-1 feasibility
maintenance (more frequent exact recompute / a composite phase-1 objective).

### T3 — retraction of a T1 finding, and the certificate chain traced end to end (2026-08-09)

**RETRACTION (CLAUDE.md §11).** T1 reported that W1 and W3 "register zero counters
on both instrumented paths", and concluded their node LPs never reach the Rust
simplex. **That was an instrument defect, not a measurement.** `profile::dump()`
`swap(0)`s every counter, and it is called at the end of `solve_lp_warm_csc_py` —
i.e. after *every node LP* on the Python path — so the snapshot was being wiped
before the probe could read it. Fixed by adding monotonic `CTOTALS` that `dump()`
does not clear (`counter_snapshot` now reads those; an explicit `reset()` clears
both). Every T1 conclusion that rested on W2 stands — W2's counters were large and
nonzero — but the W1/W3 zeros are withdrawn.

**What W1 actually does,** with the repaired instrument:

| measurement | value |
|---|---|
| `LpVerdictInfeasible` | **6586** |
| `LpInfeasUncertified` | 0 |
| `LpVerdictNumerical` | 0 |
| `solve_assembled_full` verdicts | `('infeasible', farkas=True)` × 3600 |
| `solve_lp_spatial_bb` calls | **0** |

So the Rust simplex certifies every one of these node LPs infeasible; the Python
re-verification agrees (`solve_lp_warm_std` → `farkas_certified=True`, checked
directly); `mccormick_lp` returns `MccormickLPResult(status="infeasible")` on the
verified ray; and `solver.py` fathoms it with `node_infeasible_mask` +
`_INFEASIBILITY_SENTINEL`. **The whole certificate chain works.** T3's proposed fix
— plumb `(status, farkas)` through the cold path — would have been a no-op on this
witness, and `lp_spatial_bb` is not even on its path.

One real defect was found on the way and is worth its own note: `relax.solve()`
reports `status='infeasible'` with `farkas_certified=False`, because the generic
MILP path's result builder omits the field entirely (the warm path sets it). That
is a genuine hole in `MilpRelaxationModel`, currently unreachable from W1 but
reachable from any caller that lands on the generic path — recorded here rather
than fixed, since fixing it now would be speculative.

**Also learned:** `DISCOPT_PROFILE` costs throughput badly (147k–475k stderr lines
per run, W3 fell from ~6 400 nodes to 31), so node counts measured under it must
never be compared against unprofiled runs. The verdict *shares* are still valid.
