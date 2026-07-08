# G4 — hard-tail bounding strength: gap decomposition + research roadmap (2026-07-07)

**Status:** entry experiment complete. **Decision: research-roadmap KILL — no
bounding prototype shipped.** The measured root-gap decomposition shows the hard
tail's uncertified rows are **not, in the main, root-envelope-limited**: the two
QCQP-style integer-product instances have a **near-tight root** (0.2–0.5 % gap)
and fail on **node throughput + primal**, not bound strength; one flowsheet has
**no finite underestimator at all** (no dual bound emitted); and the genuinely
envelope-loose instances are **already covered by shipped/flagged work (TD-A)** or
are **OBBT/branch-and-reduce-limited (R1/R2), not lifting-limited**. There is no
tractable bounding lift here that moves a bound not already moved by existing work
— per the task's own bar ("do NOT ship a token relaxation that doesn't move the
bound") the honest deliverable is this table + the ranked roadmap.

**Branch:** `g4-hard-tail-bounding` (from `origin/main`).
**Executor:** fresh Opus session, release build (`_pounce.abi3.so` 4.7 MB),
`PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`.
**Method:** `Model.solve(max_nodes=1)` returns the solver's own `root_bound`
(the strongest rigorous dual bound proved at the root box); `max_nodes=200` and a
60 s throughput run show how much branching closes it. Root gap is measured
against the **true MINLPLib `=opt=` oracle** (not the incumbent). Harness:
`scratchpad/root_gap_measure.py`, raw JSON `scratchpad/g4_rootgap.json`.

---

## Part 1 — the root-gap decomposition (measured)

`ROOT GAP` = `|oracle − root_bound| / max(1, |oracle|)`.

| instance | structure (dominant nonconvex terms) | root_bound | oracle | **ROOT GAP** | bound @ ~200 nodes | @60 s: status / nodes / bound | limiter |
|---|---|---:|---:|---:|---:|---|---|
| **nvs19** | dense **integer-product** QP: 8 int vars ∈[0,200], every pair `xᵢ·xⱼ` (28 bilinears) + int powers | −1104.24 | −1098.4 | **0.53 %** | −1103.07 (0.43 %) | feasible / **219** / −1102.10 | **throughput + primal** (root already tight) |
| **nvs24** | dense integer-product QP: 10 int vars ∈[0,200], 45 bilinears + int powers | −1035.66 | −1033.2 | **0.24 %** | −1035.66 (0.24 %) | feasible / **9** / −1035.66 | **throughput + primal** (root already tight) |
| **tanksize** | flowsheet, 47 var / 9 int / 21 nonlinear con (bilinear + sqrt sizing rows) | 0.847 | 1.269 | **33.2 %** | 0.868 (31.6 %) | feasible / 203 / 0.868 | **OBBT / branch-and-reduce** (R1/R2) |
| **nvs05** | welded-beam: rational + `sqrt` signomial obj `1.10471·x0²·x1 + 0.04811·x2·x3·(14+x1)` | 0.674 | 5.471 | **87.7 %** | 1.348 (75.4 %) | feasible / 207 / 1.348 | **OBBT** (obj envelope is *exact*; constraints don't push vars up) |
| **nvs09** | all-int, `Σᵢ[log(xᵢ−2)² + log(10−xᵢ)²] − (∏xᵢ)^0.2` — squared univariate logs, dropped by the linearizer | −72.90 | −43.13 | **69.0 %** | −60.09 (39.3 %) | feasible / 159 / −60.09 | **loose envelope — but TD-A already fixes it (69→27 %, flagged)** |
| **hda** | flowsheet: `log(ratio)`, `sqrt(ratio)`, `x/(x^0.23)`, `(1−x)^-1.544`, `1/(…sqrt…)` products | **None** | −5964.5 | **∞ (no bound)** | None | time_limit / 3 / **None** | **no finite underestimator** (23 rows dropped) |

**Key measurements that overturn the "the tail is bounding-limited" framing:**

1. **nvs19 / nvs24 have a near-tight root** (0.53 % / 0.24 % vs oracle). A tighter
   bilinear/RLT/SDP envelope buys essentially nothing — the McCormick-LP root is
   already within half a percent. What they lack is (a) **node throughput**
   (nvs24 reached **9 nodes in 60 s** ≈ 0.15 nodes/s; nvs19 **219 nodes** ≈ 3.6
   nodes/s — the per-node NLP + NLP primal heuristics dominate, exactly the
   `scip-gap-nvs-diagnosis.md` finding) and (b) **primal**: the incumbent is stuck
   at −1098.2 / −1031.8, short of the true −1098.4 / −1033.2, so even a perfect
   bound would not fathom. **These belong to G2 / the LP-node spatial engine, not
   G4.** (`scip-gap-nvs-diagnosis.md` already prototyped an LP-node engine that
   unfreezes throughput; the closer there is Step 2 incremental warm LP nodes, not
   a tighter root.)

2. **nvs05's objective envelope is already exact.** Its 87.7 % root gap is that the
   *constraints* leave x0..x3 at the box corner at the root (root min = 0.674 at
   x=0.01; true opt is at x0=0.68). Confirmed on record by TD-A (§6 of
   `uncertified-tail-plan-results-2026-07-06.md`): the objective-only relaxation
   over the box **equals 0.674 and is certified optimal** — no product lift moves
   it. This is **OBBT / branch-and-reduce (R2)**, not a lifting problem.

3. **tanksize is reduction-responsive** (R1: "nvs05 = 32.5 %, tanksize
   reduction-responsive"); its 33 % root gap needs R2's root-fixpoint OBBT loop,
   not a tighter envelope.

4. **nvs09 is the one genuinely envelope-loose instance — and TD-A already closes
   most of it** (root gap 69.0 % → 27.1 %, a 60.7 % relative reduction), shipped
   behind `DISCOPT_LIFT_LOOSE_PRODUCTS` (default OFF), pending graduation. No new
   G4 bounding work is needed; the lever is the **flag-graduation pipeline (G1)**,
   not a new relaxation.

5. **hda emits no dual bound at all.** The MILP relaxer drops **23 rows** it cannot
   linearize safely and then declines the bound (a dropped row leaves a nonlinear
   column unbounded → `_has_unbounded_nonlinear_col` → `bound=None`). The dropped
   rows, by term type (deduplicated over one solve):

   | dropped-row term type | count | example |
   |---|---:|---|
   | non-constant division `x/(x^0.23)` (fractional-power denominator) | 6 | `x257 / (x213^0.230769)` |
   | `sqrt(<ratio>)` / `log(<ratio>)` (transcendental of a nested division) | 9 | `sqrt(((x/x)·x)/x)`, `log((((0.0001+x)/…)/…))` |
   | undecomposable product (reciprocal of a `sqrt`-chain, squared) | 7 | `(1/(…·sqrt(…)·(…)^-1.5+1))²` |
   | variable-exponent power `(1−x)^-1.544`, `log(4^(…x))` | 3 | `(1−x43)^-1.544`; `log(0.333·4^(1+0.333·x)−0.333)` |

   Every one of these has **no finite convex underestimator under the current
   McCormick/factorable machinery** on hda's root box (arguments span zero;
   variable exponents need the `exp(y·log x)` signomial path with `x_lb>0`). This
   is the F5 (#509) / TD-B (#520, KILLED) structural class, re-confirmed here.

### Term-type attribution — which structures dominate the looseness

Pooling across the hard set, the root-bound slack is dominated by **three
distinct causes, only one of which is a loose-envelope problem**:

- **(A) Loose envelope on a specific term type — 1 instance (nvs09).**
  `pow(univariate-transcendental, int≥2)` (squared logs) is distributed to a
  `log·log` product the linearizer drops → the whole objective is dropped → bound
  from the αBB/NLP fallback only. **Already fixed by the shipped-but-flagged TD-A
  lift** (lift `t=log(·)`, rewrite `t²` exactly). Root gap 69 → 27 %.

- **(B) No finite underestimator — 1 instance (hda) + the wider Class-H
  flowsheet family** (heatexch_gen1/2/3, beuster, 4stufen per the tail results
  doc). Dominant terms: `log`/`sqrt` **of a zero-spanning ratio**, **fractional-
  power denominators**, **variable-exponent powers**, and **reciprocal-of-sqrt
  products**. No lift produces a bound (TD-B proved this on the log-arg subclass).

- **(C) Not a bounding problem at all — the largest share by instance count
  (nvs19, nvs24, nvs05, tanksize).** nvs19/nvs24 have a **near-tight root**
  (0.2–0.5 %) and fail on throughput/primal (G2 / LP-node engine). nvs05/tanksize
  are **OBBT / branch-and-reduce-limited** (R1/R2): the envelope is exact or
  near-exact; the missing lever is range reduction, not a tighter relaxation.

**Bottom line of Part 1:** across the six probes, exactly **one** (nvs09) is a
loose-envelope problem, and it is **already solved by flagged code**. The rest are
throughput (2), reduction (2), or no-finite-underestimator (1). The G4 bounding
lever, as a *new relaxation*, has **no tractable target left on this set**.

---

## Part 2 — direction ranking (gap contribution × tractability) and the decision

Ranked by (root-gap contribution the direction could actually recover) ×
(tractability within a scoped task):

| # | direction | gap it can recover | tractability | verdict |
|---|---|---|---|---|
| 1 | **(a) TD-A lift-before-distribute, extended** | nvs09 69→27 % **(already banked, flagged)**; no *new* composite family found on this set | high — but **nothing new to build**; the lever is *graduating* the flag (G1), not a new lift | **already shipped; route to G1 graduation, not G4** |
| 2 | **throughput / LP-node engine + integer-product cuts** (not strictly "bounding") | closes nvs19/nvs24 (root already tight; needs to *reach* the bound at speed) | medium–large (months; the `scip-gap-nvs-diagnosis` Step-2 engine + aggregation-MIR) | **the real nvs19/24 lever — G2/engine, out of G4 scope** |
| 3 | **(c) bound-only fallback for no-underestimator rows (hda)** | would give hda *a* (loose) finite bound where it has none | **low tractability, high risk** — the dropped rows leave columns genuinely unbounded; a "loose bound" needs interval/αBB enclosure of `log/sqrt(zero-spanning ratio)` + variable-exponent powers, i.e. the αBB-as-primary build (#4) | **research; do not prototype now** |
| 4 | **(b) αBB-as-primary on the no-underestimator structures** | the general answer for the Class-H flowsheet tail (hda, heatexch_gen*, beuster, 4stufen) | large (months) — rigorous-α exists; the open work is **routing + per-node cost + zero-spanning-argument handling**; TD-B already showed the naive log-lift path is a dead end | **the honest long pole — research roadmap** |
| 5 | **branch-and-reduce (R2) on the reformed model (nvs05, tanksize)** | nvs05 87.7 %, tanksize 33.2 % | medium — R1 already GO'd R2; deferred as "broad, non-tail" | **route to R2, not a new G4 relaxation** |

**Decision: KILL the "ship a G4 bounding prototype" arm; deliver this
decomposition + roadmap.** No direction on this hard set is a *tractable new
bounding lift that moves a bound not already moved*:

- The only loose-envelope instance (nvs09) is **already fixed by flagged TD-A**;
  the outstanding action is **graduation (G1)**, not new relaxation code.
- The largest-count failures (nvs19/nvs24) have a **near-tight root** — a tighter
  envelope is measurably almost worthless; they need **throughput/primal (G2 /
  LP-node engine)**.
- nvs05/tanksize need **OBBT/branch-and-reduce (R2)**, not lifting (envelope is
  exact/near-exact).
- hda needs **αBB-as-primary / a bound-only fallback for genuinely-unrelaxable
  rows** — a months-scale research build, and TD-B already falsified the cheap
  version.

Shipping a token relaxation here would either be inert (structures absent), a
regression (contvar-style aux blowup, per TD-B §5.2), or duplicate TD-A. That
fails the task's explicit bar. **The gap-decomposition table + the roadmap below
is the deliverable.**

---

## Part 3 — the ranked research roadmap (the months-scale long pole)

In priority order (payoff × leverage across the tail), for the next executor:

1. **Graduate TD-A (and R4) via the G1 pipeline — cheap, highest certainty,
   FIRST.** nvs09's 60 % root-gap reduction is *built and validated* but inert
   because the flag never flips. This is a **process** action
   (`flag-graduation-protocol.md`), not a bounding build. Expected: nvs09 root gap
   69 → 27 % reaches the default user; a materially tighter tail bound at zero new
   risk. *(This is where the one real envelope win on the hard set actually
   lands.)*

2. **LP-node spatial engine + incremental warm LP (throughput) — the nvs19/nvs24
   closer.** Their root is already 0.2–0.5 % tight; the wall is 0.15–3.6 nodes/s
   (per-node NLP + NLP heuristics). `scip-gap-nvs-diagnosis.md` Phase-0 already
   prototyped an LP-node B&B that unfreezes the bound; the remaining work is
   **Step 2** (incremental row/bound updates on the warm simplex basis → ~1–2
   ms/node → tens of thousands of nodes) + a **primal** pass that reaches the true
   integer optimum (both incumbents are 0.02–0.13 % short). Expected: nvs19/nvs24
   certify in seconds, matching no-cut SCIP (6,796 nodes / 4.7 s). *This is a
   throughput/engine build, not a bounding build — but it is where the nvs QCQP
   certs live.*

3. **Branch-and-reduce (R2: root-fixpoint OBBT + `reduce_node`) on the reformed
   model — the nvs05 / tanksize closer.** R1 already returned GO. nvs05's envelope
   is exact; its 87.7 % gap is that the root LP sits at the box corner —
   OBBT/probing that pushes x0..x3 up is the lever. tanksize is reduction-
   responsive (33 % → needs the loop). Expected: both move materially once the
   root-fixpoint loop runs on the model the tree actually branches (R3b's lesson:
   measure on the reformed, not factorable-only, model).

4. **αBB-as-primary for the no-finite-underestimator flowsheet class (hda,
   heatexch_gen1/2/3, beuster, 4stufen) — the true research pole (months).** These
   rows (`log`/`sqrt` of zero-spanning ratios, fractional-power denominators,
   variable-exponent powers, reciprocal-of-sqrt products) have **no McCormick
   underestimator**. Rigorous interval-Hessian αBB *exists* (`alphabb.py`); the
   open problems are: **(i) routing** — detect a row the factorable relaxer drops
   and emit an αBB underestimator for it instead of dropping it (so the relaxer
   emits a finite, if loose, bound rather than `None`); **(ii) per-node cost** —
   αBB on 262-nonlinear-column hda is expensive; **(iii) zero-spanning arguments**
   — many arguments have no positive lower bound even for αBB's interval Hessian,
   needing a spatial split that establishes sign first. TD-B (#520) already
   falsified the cheap log-arg-lift shortcut; this is the hard, right fix.
   Expected payoff: the first *finite dual bound* for the Class-H flowsheet tail —
   BARON's remaining certs live here. Effort: months. Kill criterion for the
   entry experiment: if αBB on the dropped rows still yields `bound=None`
   (unbounded columns persist after enclosure) or regresses per-node wall beyond
   the budget on hda, record and re-scope to a spatial-sign-establishing presolve
   first.

5. **(Lower priority) integer-product aggregation-MIR cuts** — a node-count
   multiplier for nvs19/24 *after* the LP-node engine (SCIP's 97× reduction comes
   from `aggregation` multi-row complemented MIR, which discopt lacks). Neither
   necessary nor attainable before the throughput engine exists
   (`scip-gap-nvs-diagnosis.md` Step 3, deferred).

**Effort shape:** #1 is days (process). #2/#3 are engine/reduction builds already
scoped elsewhere (weeks–months). #4 is the genuine research long pole (months) and
is where a future G4 *bounding* task should focus — but only after an entry
experiment establishes that αBB can produce a finite bound on the dropped-row
class at all (the open question TD-B left).

---

## Appendix — reproduction

```
# release build, 4.7 MB pounce
maturin develop --release
PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python scratchpad/root_gap_measure.py nvs05,nvs09,nvs19,nvs24,tanksize,hda out.json
```

`root_gap_measure.py` runs each instance at `max_nodes=1` (root only → `root_bound`)
and `max_nodes=200`, printing `root_gap_vs_oracle` and `bound_gap_vs_oracle`
against the MINLPLib `=opt=` oracle. The 60 s throughput probe (nvs19/nvs24 at
`max_nodes=1_000_000`) confirmed the near-tight-root / low-throughput split.

No source changed — this task is a measurement + decision. No flag, no relaxation,
no gate run required beyond the reproduction above (nothing to regress).
