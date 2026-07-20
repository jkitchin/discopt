# #801 — `tanksize` root relaxation (research-grade): execution plan

Status: **RESOLVED 2026-07-20 — all residuals FALSIFIED; recommend closing #801.**
Every genuinely-untested residual is measured inert on the real `tanksize` root
(entry experiments, converged solvers): the order-1 dense moment/Shor SDP (0.8401),
the order-2 star moment/Lasserre relaxation — the dominating test of the "tighter
trilinear hull" caveat — (0.8400), and PQ (no target: the split group is integer,
integers-fixed → 0.838). The 0.838→≥0.955 gap is **not accessible to any convex
relaxation of the products through the moment/SDP hierarchy up to order 2**, which
decisively extends the #764 "the gain is NOT in the RLT hierarchy" conclusion. The
recorded path for `tanksize` itself remains throughput (#800 / `certification-gap-plan`
Phase B), not relaxation. Full falsification record in §9 below. Execution plan
retained below for provenance.

This document was the step-by-step plan for executing issue #801 (split from #798 §3).

Goal of the issue: the root bound on `tanksize` is `lb(x17)` = **0.8382** (McCormick
LP over the FBBT box) vs BARON's root of **0.955**. #801 asks whether any residual
relaxation mechanism can close part of that root gap — entry-experiment-first, on the
real instance, per CLAUDE.md §4.

## 0. Read these first (binding context — do not skip)

| doc | why |
|---|---|
| `docs/dev/issue-764-root-relaxation-plan.md` | The complete prior campaign on exactly this question, including the falsification records this plan builds on and the 2026-07-18 strategy correction ("tanksize is a per-node-throughput problem"). |
| `docs/dev/performance-plan.md` §6 | House style for recording falsifications. |
| CLAUDE.md §1, §4, §5 | Correctness gates, entry-experiment discipline, Regime-2 (bound-changing) graduation contract. |
| Issue #798 §3–§4 (closed tracker) | Where #801 came from; the binding negatives list. |

## 1. Reframe first — the issue premise is partly stale (do this before any code)

#801 names the residual lever as "**RLT-2 / fractional-envelope / pooling-PQ**".
Checked against the committed record (`issue-764-root-relaxation-plan.md`,
"Candidate mechanisms, ranked"), that list is mostly already dead:

- **Pooling-PQ / level-1 RLT — already FALSIFIED** (entry experiment 2026-07-18).
  The *full* level-1 RLT closure (2190 product rows, 2170 lifted columns, sparse
  solve) left the root bit-identical: 0.8382 → 0.8382. Construction validated by
  injection (`x17 ≥ 1.0` moves the root to 1.0, so the rows genuinely reach the LP).
- **Level-2 RLT — already FALSIFIED** (entry experiment 2026-07-18). 121 degree-3
  rows, expanded to clean monomials so the classifier lifts them (`bil 47→293,
  tri 0→213`): 0.8382 → 0.8382. **One caveat was recorded**: the 213 trilinear
  terms were relaxed with discopt's *trilinear McCormick*; "a tighter trilinear
  hull could in principle bind." That caveat is the only polyhedral door left open
  — it is Stage 2a below.
- **Fractional envelopes — structurally vacuous as literally stated.**
  `tanksize.nl` contains **zero division operators** (operator census: 84 `o2`
  multiply, 3 `o39` sqrt, no `o3` divide). There are no fractional terms to
  envelope. The only meaningful reading is a PQ-style *reformulation* that
  introduces ratio/fraction structure — and whether any such structure escapes the
  already-falsified level-1 RLT closure is a structural question Stage 1 answers
  before any build.

**Action:** post a reconciliation comment on #801 stating the above (with pointers
to the falsification records) and the corrected scope: the genuinely untested
residuals are (a) tighter trilinear hulls on the RLT-2 lift, (b) block moment/SDP
on the shared-variable stars (the recorded candidate 3, never run), and (c) a
PQ-reformulation cross-check *only if* the Stage-1 structural audit finds product
families the tested RLT-1 closure did not cover. Do **not** rebuild the falsified
experiments.

## 2. Definition of done — falsify-and-close is a legitimate completion

Honest expectation-setting (all measured, see #764 doc): BARON's *own* root is
0.955 — still a 25 % gap — and BARON wins on per-node throughput (~1300 nodes/s),
not root tightness; discopt's search already closes tanksize in *fewer* nodes than
BARON when given time. The prior probability that any residual mechanism both
moves the root materially *and* shrinks the tree is **low**. This issue is
research-grade: either outcome finishes it.

- **GO outcome:** a mechanism lifts the root materially (Stage 3 thresholds),
  generalizes beyond tanksize, and ships default-OFF behind a flag per the
  Regime-2 contract, with graduation tracked as a follow-up.
- **KILL outcome (expected):** all residuals measured inert → append the
  falsification records to this doc and `issue-764-root-relaxation-plan.md`,
  comment on #801 with the records, and **recommend closing #801** — the recorded
  path for tanksize itself remains throughput (#800 / `certification-gap-plan.md`
  Phase B / the LP-OA kernel line), not relaxation.

Time-box: ≤ 3 working days total. Stage 0–1 ≈ half a day; each Stage-2 experiment
≈ half a day to a day; stop a stage the moment its kill criterion fires.

## 3. Stage 0 — re-baseline on current HEAD (gate for everything else)

The prior experiments' scratchpad scripts (`pq_exp.py`, `pq_exp2.py`, `rlt2b.py`)
were **not committed and are gone**. Recreate the probe as a committed script this
time: `discopt_benchmarks/scripts/issue801_root_probe.py`.

1. Standard probe = `MccormickLPRelaxer.solve_at_node` on
   `python/tests/data/minlplib_nl/tanksize.nl` over the FBBT root box, integers
   relaxed (the #764 measurement setup; use `DISCOPT_SPARSE_LARGE_LP=1` for any
   lifted construction).
2. Confirm the baseline root LP is still **0.8382** on current HEAD. If it is not,
   STOP: something graduated since 2026-07-18 changed the root; re-diagnose before
   trusting any prior negative.
3. Wire two standing validity checks into the probe harness, used by every
   Stage-2 experiment:
   - **Injection sanity:** adding `x17 ≥ 1.0` must move the root to 1.0 (proves a
     construction's rows reach the LP — the #764 technique).
   - **No-valid-point-cut:** the known optimum (oracle **1.2686437540**, and the
     node-0 incumbent point) plus sampled feasible points must satisfy every added
     row/cut to tolerance. Any violation = the construction is unsound; stop.
4. Env trap (recorded in #798/#802): `discopt.pth` is a single shared path — drive
   runs via `PYTHONPATH` and verify which build you're measuring before trusting
   numbers.

## 4. Stage 1 — gap attribution + structural audit (cheap; steers Stage 2)

No prior experiment measured *where* the 0.117+ gap lives. Do that before choosing
targets:

1. **Residual attribution.** At the root McCormick LP optimum, compute
   `w_ij − x_i·x_j` for all 47 lifted products and rank by |residual| (and by a
   dual-weighted measure if marginals are available —
   `MccormickLPResult.reduced_costs` exists since #764 Phase 2 step 1). Known
   prior: with integers fixed the root is still 0.8382, so the gap is
   continuous×continuous — expect the mass on `x0·x6/x9/x12, x1·x7/x10/x13,
   x2·x8/x11/x14, x15·x16, x16·x17`. The top blocks define the SDP cliques for
   Stage 2b and the trilinear targets for 2a.
2. **Measure the ceiling `C`.** The continuous-relaxation global optimum satisfies
   `0.955 ≤ C ≤ 1.2686`. Multistart local NLP (IPOPT/pounce) on the
   integer-relaxed instance gives an upper estimate of `C`. This bounds the value
   of *all* root-relaxation work: if `C ≈ 0.955`, even a perfect hull gains only
   ~0.117 and the tree is unavoidable regardless.
3. **PQ coverage audit.** Determine whether the split group {x18..x26} is a
   fraction simplex (sum-to-one linear rows) and enumerate the product families a
   literature pooling-PQ formulation would add (fraction×fraction, fraction×flow
   in both directions, sum-to-one × variable products). Check each family against
   what the falsified full level-1 closure already contained. Only a family
   **outside** that closure justifies Stage 2c; otherwise record "PQ definitively
   covered and dead" and skip 2c.

## 5. Stage 2 — residual entry experiments (hypothesis / experiment / kill, each)

Run in this order (cheapest and most-likely-informative first); all offline in the
probe script — **no product code until a GO**.

### 2a. Tighter trilinear hulls on the RLT-2 lift (the recorded open caveat)

- **Hypothesis:** the level-2 RLT rows were inert *because* their 213 trilinear
  terms were relaxed with recursive/trilinear McCormick; the facet-defining
  trilinear convex hull (Meyer & Floudas 2004 extreme-point facets) is strictly
  tighter and might make some row bind.
- **Experiment:** rebuild the `rlt2b` construction (level-2 rows expanded into
  clean monomials — the classifier drops naive `(-g)*fi*fk` products, `tri=0`;
  expansion is mandatory to get `bil 47→293, tri 0→213`). Then replace/augment the
  trilinear relaxation with explicit extreme-point facet cuts per trilinear box —
  either enumerate hull facets offline per term, or run a separation loop that
  adds violated facets at each LP optimum. Validate with both Stage-0 checks.
  Measure the root.
- **Kill:** root gain < +0.005 over 0.8382 → dead, and record that this closes the
  polyhedral (RLT) hierarchy on this class *including* the tighter-hull caveat.

### 2b. Block moment/SDP on shared-variable stars (#764 candidate 3 — never run)

- **Hypothesis:** the unmodeled tightness is the *joint* constraint among products
  sharing a variable (the #764 structural diagnosis); a moment relaxation on each
  shared-variable clique (e.g. {x0, x6, x9, x12} ∪ their products) captures
  second-order joint information that per-term McCormick and diagonal Shor
  (0.840, near-inert) cannot. Low prior — Shor's inertness is evidence against —
  but it is the last untested relaxation route on record.
- **Experiment:** offline with cvxpy/scs: sparse Shor-with-blocks / order-1
  Lasserre on the Stage-1 top cliques, *including the model's linear rows*, over
  the root box, minimizing x17. Measure the SDP bound.
  - If the SDP bound ≤ ~0.85: KILL immediately (no LP-transferable value).
  - If materially above 0.85: second step — extract linear certificates
    (eigenvector cuts `(vᵀ[1;x])² ≥ 0` expanded over the lift, or dual-derived
    valid inequalities), add them to the McCormick LP, re-validate
    (injection + no-valid-point-cut + NS-safe bounding for any shipped variant),
    and measure the *LP* root. The LP number is the decision number — an SDP
    bound that does not transfer to the LP node relaxation does not help the
    B&B.
- **Kill:** block-SDP bound < 0.85, or LP-transferred gain < +0.005, or the cut
  extraction cannot be made sound (any sampled feasible point violated).

### 2c. PQ-reformulation residual (conditional — only if Stage 1 opened it)

- **Hypothesis:** a product family from the pooling PQ formulation lies outside
  the falsified level-1 closure and binds.
- **Experiment:** add exactly the uncovered family (lifted variables + their
  McCormick + the RLT identities linking them), validate, measure the root.
- **Kill:** gain < +0.005, or Stage 1 shows full coverage (then this stage never
  runs — record why).

Record every result — GO or kill — in this doc in the `performance-plan.md` §6
house style (hypothesis, measurement, verdict, repro pointer), and keep the probe
script committed.

## 6. Stage 3 — decision gate for any GO

A root gain alone does not ship. In order:

1. **Generality (CLAUDE.md §2 — no single-instance mechanisms).** Re-run the
   winning construction on ≥ 2 other dense-bilinear/pooling-structured instances —
   draw from the full MINLPLib corpus (`~/Dropbox/projects/discopt-minlp-benchmark/`,
   filter by type/structure via `minlplib_types.csv`); in-repo fallbacks:
   `casctanks`, `heatexch_gen1..3`, `4stufen`. The mechanism must help the *class*
   (or come with a class-level structural trigger, never a problem-name key).
2. **Tree impact.** Root gain that doesn't shrink the tree is recorded but not
   shipped: measure bound-at-60 s and nodes-to-close on tanksize + siblings with
   the mechanism applied at the root (and, if cheap enough, per-node). Compare
   against the default trajectory (0.9221 @ ~143 nodes/60 s baseline from #764).
3. **Ship path (Regime-2, binding — CLAUDE.md §5).** Default-OFF flag; every added
   row/cut is bound-changing: differential bound test + feasible-point sampling;
   corpus-wide flag ON-vs-OFF panel that is BOTH cert-clean (`incorrect_count = 0`,
   no bound above its reference optimum, no certification regression, incumbents
   independently verified) AND net-positive (nodes/wall/bound — sound ≠ helpful,
   the `DISCOPT_CUT_INHERIT` lesson). A bilinear envelope that over-tightens is a
   false optimal — the single worst failure in this codebase; NS-safe bounding on
   any solved sub-relaxation that feeds a certificate.

## 7. Stage 4 — close the loop (either outcome)

- Update this doc and `issue-764-root-relaxation-plan.md` with the verdicts.
- Comment on #801: the Stage-1 reconciliation, each experiment's
  hypothesis/measurement/verdict, and an explicit statement of **whether #801 can
  be closed** — on all-kill, recommend close with the falsification record as the
  deliverable and the pointer that tanksize's own path remains throughput
  (#800 / certification-gap-plan Phase B); on a GO, name the follow-up flag and
  its graduation status.

## 8. Assets and pointers

| asset | where |
|---|---|
| Instance + oracle | `python/tests/data/minlplib_nl/tanksize.nl`; optimum 1.2686437540 (min), incumbent found at node 0 — pure dual-closure |
| Probe pattern | `MccormickLPRelaxer.solve_at_node` (root FBBT box, integers relaxed); `DISCOPT_SPARSE_LARGE_LP=1` for lifted LPs |
| Prior falsification detail | `issue-764-root-relaxation-plan.md` "Candidate mechanisms" (constructions, row counts, injection technique) |
| Term classifier behavior | naive degree-3 products are dropped (`tri=0`) — expand to clean monomials first (`python/discopt/_jax/term_classifier.py`) |
| Shor SDP path (reference for 2b) | `DISCOPT_SHOR_SDP_ROOT_BOUND` wiring in `python/discopt/solver.py` |
| Marginals for attribution | `MccormickLPResult.reduced_costs` / `row_dual` (#764 Phase 2 step 1, `python/tests/test_phase2_cold_marginals.py`) |
| Existing #764 scripts | `discopt_benchmarks/scripts/issue764_*.py` (node-cost decomposition, cut attribution) |
| Results home | `discopt_benchmarks/results/issue801/` (create; JSON + md, timestamped) |

---

## 9. Falsification record (2026-07-20) — measured, binding negatives

Executed per the plan; all probes committed under
`discopt_benchmarks/scripts/issue801_*.py`, artifacts under
`discopt_benchmarks/results/issue801/`. House style: hypothesis / measurement /
verdict (performance-plan §6). **Do NOT re-walk these.**

### Stage 0 — baseline reproduced on HEAD
Root McCormick LP over the FBBT box, integers relaxed = **0.8382369708575** — exactly
the #764 figure. Notable: it equals `fbbt lb(x17)` to 13 digits, i.e. the McCormick
LP does not improve on interval propagation for the objective. Box-injection plumbing
validated (forcing `lb(x17)=1.0` moves the root to 1.0).
`issue801_root_probe.py`.

### Stage 1 — attribution + structural audit (steers/kills Stage 2)
- **Integers-fixed root = 0.8382369708575** (bit-identical). Fixing the 9 integer
  split vars x18..x26 to their optimal leaf `{1,0,0,0,1,0,0,0,1}` leaves the root
  unchanged → the entire gap is the **continuous×continuous core** (11 products):
  the flow stars x0:{6,9,12}, x1:{7,10,13}, x2:{8,11,14} and the objective chain
  x16:{15,17}. Reconfirms #764 on HEAD.
- **Ceiling C:** reliable bracket **C ∈ [0.955 (BARON root), 1.2686 (incumbent)]**;
  a bounded multistart local NLP was uninformative (landed at a 9.35 local min), so
  it did not tighten the bracket. Max root gain available ≤ ~0.117–0.43.
- **PQ audit:** the split group x18..x26 is **integer** (selection vars, 13
  simplex-like rows), not fractional proportions, and integers-fixed→0.838, so a
  pooling-PQ formulation has **no target in the continuous core**. Combined with the
  #764 finding that the full level-1 RLT closure (which already contains every
  product family) is bit-identical inert, **Stage 2c does not run** (no product
  family outside the falsified closure).
  `issue801_stage1_attribution.py`.

### Stage 2b — order-1 dense moment (Shor) SDP — FALSIFIED
- *Hypothesis:* the joint constraint among products sharing a variable (the #764
  candidate 3, never run) is captured by a moment/SDP relaxation over the stars; the
  **dense** order-1 moment matrix over all 47 vars dominates every block/star version.
- *Construction:* self-contained QCQP moment SDP built from the instance's own exact
  quadratic forms (extracted by finite differences; faithfulness gate: the
  McCormick-only QCQP-LP reproduces 0.8382, ✓). PSD on M=[[1,x'],[x,X]] + McCormick
  box, solved three ways for trust.
- *Measurement:* **x17 = 0.8401**, agreeing across (1) a **converged** scaled SCS
  (status `solved`, M PSD to 1e-6, feasibility residuals ~1e-7, M00=1.0), (2) a
  rigorous scaled PSD cutting-plane (HiGHS per round, frozen at 0.8401), and (3) a
  rigorous unscaled cutting-plane (0.8401). The +0.0018 over 0.8382 is entirely from
  dropping the sqrt equality (con 18), not from the PSD. **A caution recorded for the
  next agent:** an *unconverged* SCS run on the *unscaled* problem reported a spurious
  1.143 — a non-convergence/scaling artifact (the moment entries span 0..8e6). Never
  trust an `inaccurate`/`max_iters` SCS objective; rescale to [0,1] and audit
  feasibility.
- *Verdict:* **KILL** (gain 0.0018 < 0.005 threshold, and not from the PSD). Dense
  order-1 moment SDP inert ⟹ every block/star order-1 SDP inert.
  `issue801_stage2b_moment_sdp.py`, `issue801_stage2b_psd_cuts.py`,
  `issue801_stage2b_scaled.py`.

### Stage 2a — order-2 star moment (Lasserre), the tighter-trilinear-hull test — FALSIFIED
- *Hypothesis:* #764 left one caveat open — its level-2 RLT used recursive trilinear
  McCormick; a tighter trilinear hull "could in principle bind." The **dominating**
  test is a sparse order-2 moment (Lasserre) relaxation on the stars: its PSD moment
  matrix relaxes the degree-3/4 monomials (the trilinear terms) tighter than recursive
  McCormick and keeps them consistent, so it dominates both the literal hull
  refinement and the recorded RLT-2.
- *Construction:* base 48×48 moment PSD + one order-2 moment PSD block per star
  (dims 15,15,15,10 over the degree-≤2 monomial basis), stars variable-disjoint so no
  cross-star coupling; linking equalities tie the blocks' degree-≤2 entries to the
  base. Scaled to [0,1].
- *Measurement:* **x17 = 0.8400** (converged SCS, status `solved`); corroborated by a
  rigorous order-2 cutting-plane frozen at 0.8401 while actively separating star
  eigenvector cuts.
- *Verdict:* **KILL** (gain 0.0018 < 0.005, not from the higher-order moments). The
  trilinear/higher-order polyhedral+moment route is closed.
  `issue801_stage2a_order2.py`, `issue801_stage2a_scs.py`.

### Consolidated reality check (extends the #764 "RLT hierarchy exhausted" table)
```
McCormick 0.8382 = level-1 RLT 0.8382 = level-2 RLT 0.8382 (#764)
  = order-1 dense moment/Shor SDP 0.8401 (converged) = order-2 star Lasserre 0.8400 (converged)
diagonal Shor 0.840 (#764) ; 30-round OBBT 0.838 (#764) ; integers-fixed 0.8382
Yet C ∈ [0.955, 1.2686].
```
The 0.838→≥0.955 gap is **not accessible to the polyhedral (RLT) or moment (SDP,
order ≤2) hierarchy** on the real instance. This is the important signal, now
rigorously extended two full moment-hierarchy levels beyond #764. `tanksize`'s own
path to "seconds" is **per-node throughput** (#800 / `certification-gap-plan` Phase B),
not a stronger root relaxation.

### Recommendation
**Close #801.** Its literal DoD (a residual root-relaxation lever for `tanksize`) is
falsified: every genuinely-untested candidate is measured inert with converged
solvers, on the real instance, entry-experiment-first. No sound, net-positive
bound-changing mechanism exists to ship, so there is nothing to graduate. The
throughput lever lives in #800 / the certification-gap plan.
