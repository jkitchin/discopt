# discopt performance roadmap

**Status:** living document. **Created:** 2026-07-28.
**Companion to** `performance-plan.md` (the append-only *measured record*) and
`certification-gap-plan.md` (the certification workstream). This document is the
**forward-looking, modular index**: it says what is worth doing next, what is
provably not, and how to tell the difference. It does not restate their contents
— it links to them.

---

## §0 How to use this document

Each workstream in §3 is a **self-contained card**. You should be able to pick up
any one card, read only it plus this section, and start work. Cards do not depend
on each other's prose; where they depend on each other's *results*, the card says
so in `Depends on`.

**Every card has the same six fields, and they are not decoration:**

| field | what it is for |
|---|---|
| **Hypothesis** | The falsifiable claim. If you cannot state it falsifiably, it is not ready to be a card. |
| **Evidence** | What is already measured, with provenance. Distinguishes "we know" from "we suspect". |
| **Entry experiment** | The measurement that must run **before any implementation code**. |
| **Kill criterion** | The result that ends the card. Written *before* the experiment, so it cannot be rationalised afterwards. |
| **Gates** | What must pass before it ships. Correctness gates are never negotiable. |
| **Depends on** | Other cards whose *results* (not code) this one needs. |

### The three rules that generated this document

1. **Correctness is a gate, not a goal.** `incorrect_count <= 0` has zero slack.
   A change that makes anything faster while risking a false
   optimal/infeasible/bound is a regression, full stop. See
   `CLAUDE.md` §1 and `performance-plan.md` §0.
2. **No fix ships on a hypothesis.** Every card names its entry experiment and
   kill criterion. Run the experiment first. If it kills the card, record the
   falsification in `performance-plan.md` and stop — that is a successful
   outcome, not a wasted one.
3. **Sound is not the same as helpful.** This repo has now killed *six* changes
   that were provably correct and measurably useless (§4). A cert-clean but
   neutral change stays out, and the measurement gets recorded. Budget for this:
   historically a substantial fraction of plausible-looking levers die here.

### The failure mode this document exists to prevent

Two workstreams in July 2026 (#861, #896) were scoped from **cost-per-call
microbenchmarks** without checking **call counts** or **what else dominates**.
Both were killed by measurement after the scoping, not before. The specific
error: `hda`'s relaxation build costs ~18 ms, which is genuinely large — but hda
runs **3 node solves in 81 s**, so the build is 3.9% of its solve and eliminating
it entirely is worth nothing.

> **Cost-per-call is not impact. Impact is cost-per-call x call-count / total.**
> A microbenchmark accurate in isolation can still support a completely wrong
> scoping decision. Before opening a card, profile the *whole solve*.

---

## §1 Evidence base and confidence

Everything in §2–§3 is traceable to one of these. Confidence is stated so a
reader knows what is load-bearing and what is indicative.

| source | date | what it establishes | confidence |
|---|---|---|---|
| `performance-plan.md` §1 | 2026-06-24 | The CC1–CC5 cost model | high (measured) |
| `performance-plan.md` §10 (2nd) | 2026-07-18 | hda is **root-throughput bound**: 3 nodes, root eats the budget, no primal | high |
| `performance-plan.md` §12 | 2026-07-21 | Node cost is **node-count-bound**; a full NLP per node where SCIP uses an LP | high |
| `certification-gap-plan.md` Phase D | 2026-07-05 | POUNCE subsolver was the #1 wall lever; fixed (2.0–2.7x on nvs17/nvs13) | high |
| #861 / #896 close-outs | 2026-07-28 | Incremental-McCormick admission is ~1.0x end-to-end; relaxation build is 0.8–6.1% of wall | high — **independently re-verified** |
| This document's §2 table | 2026-07-28 | Phase attribution on hda + the tail | high — **independently re-verified** |

**Independent verification (2026-07-28).** The #861/#896 numbers were re-measured
from scratch by a separate agent that was instructed to refute them. Result: **all
claims confirmed, no refutations**, with these corrections adopted here:

* `hda` relaxation build is **17.84 ms** (not 20.41) and **3.9%** of wall (not 4.1%).
* **Six** declined instances exceed a 1 ms build, not five — `st_e35` at 1.02 ms
  straddles the line.
* `nvs20` and `util` **do not time out** — they solve to proven optimality in
  9.2 s / 7.6 s. Their small build shares are of a *complete* solve, so they were
  never throughput candidates at all.
* Admission trends **0–5% slower**, not neutral, on the small class — the ON arm
  pays structure build + validation at construction. Resolvability was quantified:
  relative sd 1–8%, so a 1.3x effect would sit 4–30 sigma outside the observed
  spread. The ~1.0x null is resolvable; anything under ~5% is not.
* The node-parity panel was proven **not blind**: corrupting one envelope
  coefficient *after* construction (bypassing `_validate`) produced **76
  disagreements in 120 comparisons** and a non-zero exit.

---

## §2 The cost model — where solve time actually goes

Phase attribution, `m.solve(time_limit=60)` on `hda` under cProfile
(independently reproduced; shares are robust to the machine load present):

| phase | share |
|---|---|
| NLP subsolves (`solve_nlp`) | **45.0%** |
| Root presolve (Rust `run_root_presolve`) | **18.0%** |
| Convexity classification | **15.7%** |
| Node LP solving (`solve_at_node`, **3 calls**) | 12.3% |
| Relaxation building (`build_uniform_relaxation`, 8 calls) | **3.9%** |

Relaxation build share across the rest of the large-model tail: `ex1233` 5.8%,
`kall_circles_c8a` 6.1%, `nvs20` 1.7%, `util` 0.8%.

**The headline: relaxation building is 0.8–6.1% of wall everywhere.** It is not a
lever, and two workstreams have now died proving it.

### External calibration against SCIP (`hda`, same `.nl`, same machine)

| | discopt 60 s | SCIP 60 s | SCIP 300 s |
|---|---|---|---|
| nodes | **3** | 4,693 | 27,546 |
| incumbent | **none** | **-5964.534** (= reference optimum) | **-5964.534** |
| dual bound | **-64,473** | -6,721.9 | -6,656.0 |
| gap | n/a (no primal) | 12.70% | 11.59% |

Read this carefully, because it cuts both ways. **hda is genuinely hard** — SCIP
does not close it either, and 300 s of SCIP moves the dual bound about 1%. But
discopt is behind on three axes that have nothing to do with relaxation building:
**no incumbent at all** where SCIP finds the optimum; **3 nodes vs thousands**;
and a dual bound **9.6x weaker**.

---

## §3 Workstreams

Ordered by expected value. **Card C is the top card** — it carries the largest
measured effect (a 3-order-of-magnitude NLP/node spread). Reorder freely as
evidence lands; that is what makes this modular.

---

### Card A — Primal: find an incumbent where SCIP finds one in seconds

> **Status:** OPEN. **Owner:** unassigned. **Depends on:** nothing.
> **Related:** #862 (incumbent quality), #844 (no-primal fallback).

**Hypothesis.** On the no-primal class, discopt's failure is not search *depth*
but the heuristics never producing a feasible integer point — and a solver that
returns no incumbent has no gap, cannot fathom, and cannot report progress,
making this the highest-value defect independent of speed.

**Evidence.** `hda`: 60 s, ~20 s spent in `integer_local_search` /
`feasibility_pump` / `subnlp` (8 `solve_nlp` calls, 37.4 s cumulative) and still
`objective=None`, while SCIP finds the true optimum and keeps 12 solutions in the
same budget. `kall_circles_c8a`: incumbent 2.727 vs reference 2.541.

**Entry experiment.** For each no-primal instance, instrument the heuristic chain
and answer: does it (a) never run, (b) run and fail to find a feasible point, or
(c) find one that fails verification? These are three different bugs with three
different fixes. Print an executed-attempt count per heuristic — a heuristic that
silently never fires is the most likely single cause and the cheapest to fix.

**Kill criterion.** If the heuristics *are* running and *are* finding feasible
points that then fail an over-strict verification, this card converts to a
verification-tolerance card and the primal framing is wrong. If they run and
genuinely cannot find a feasible point on constraint systems SCIP satisfies in
seconds, the card stands and points at heuristic quality.

**Gates.** Any incumbent must be verified feasible by the exact evaluator — never
trusted from a relaxation value (this is the nvs17 false-optimum lesson,
`lp_spatial_bb.py`). `incorrect_count = 0`. New incumbents may not degrade
`node_count` on the certifying panel.

---

### Card B — Dual: the bound-quality outliers

> **Status:** OPEN. **Owner:** unassigned. **Depends on:** nothing.

**Hypothesis.** A dual bound 10x from the optimum is a relaxation-strength defect
localisable to specific atom classes, not a generic tightness problem.

**Evidence.** `hda` bound -64,473 vs optimum -5,964.5 (9.6x weaker than SCIP's
-6,721.9). `kall_circles_c8a` bound **0.0** against optimum 2.541 — a bound of
zero is a total failure of the relaxation on that structure, not a loose one.
Per `performance-plan.md` §10 (1st), `#671` already gave hda a *tight root bound*
(-6.47e4) — so the -64,473 seen here needs explaining: either the tight path is
not being taken by default, or it is lost later.

**Entry experiment.** Start with `kall_circles_c8a` — a 0.0 bound is the most
diagnosable signal available. Identify which atom's envelope collapses. Then,
separately, determine whether hda's default path takes the #671 tight-bound route
at all (this is a routing question and may be cheap).

**Kill criterion.** If the weak bounds trace to atoms already known-hard (the
`ex6_2` Gibbs/log-sum family is **falsified F13** — joint OA there is *unsound*,
and rigorous alpha-BB is ~1e40 worse), stop and record. Do not re-attempt F13.

**Gates.** Bound-changing regime (`CLAUDE.md` §5): differential bound test
(new >= old AND <= true box optimum), feasible-point sampling (no valid point
cut), flag-gated default-off until a corpus panel is both cert-clean and
net-positive.

---

### Card C — Node regime: NLP-per-node is the dominant cost, and it is not admission

> **Status:** OPEN — **evidence complete, this is the top card.**
> **Depends on:** nothing. **Supersedes:** the "is admission the lever?" question.

**Finding.** discopt has two node regimes separated by ~100-800x throughput, and
the separator is **how many NLP solves each node costs** — not relaxation
building, and not incremental-McCormick admission.

Measured 2026-07-28 (25 s limit, load avg 4.4-5.4; ratios are the load-robust
quantities here):

| instance | nodes | nodes/s | solve_nlp | **NLP/node** | admitted |
|---|---|---|---|---|---|
| nvs17 | 13,555 | 798.6 | 82 | **0.01** | True |
| nvs19 | 4,913 | 296.5 | 61 | **0.01** | True |
| nvs24 | 1,353 | 81.0 | 33 | **0.02** | True |
| util | 501 | 74.6 | 2 | **0.00** | False |
| nvs20 | 105 | 12.3 | 33 | 0.31 | False |
| kall_circles_c8a | 95 | 3.8 | 68 | 0.72 | False |
| st_e35 | 31 | 1.2 | 43 | 1.39 | False |
| ex1225 | 5 | 6.5 | 133 | **26.60** | False |
| nvs01 | 3 | 3.5 | 69 | **23.00** | False |
| nvs21 | 3 | 1.7 | 60 | **20.00** | False |
| gear4 | 3 | 8.5 | 124 | **41.33** | False |
| hda | 3 | 0.1 | 0 | 0.00 (root-bound) | False |

**The spread is three orders of magnitude: 0.01 vs 20-41 NLP solves per node.**

**Admission correlates but does NOT cause.** All 3 admitted instances are fast
and all 8 slow ones are declined (`admitted among FAST 3/4, among SLOW 0/8`) —
a strong association. But two independent pieces of evidence say admission is a
**marker**, not the mechanism:

* **`util` is declined and fast** (74.6 nodes/s, 2 NLP calls total). Admission is
  not necessary for the fast regime.
* **The causal test says no.** #861's A/B toggled `DISCOPT_INCREMENTAL_MC` on the
  *same* instances and measured **1.0x** (independently re-verified; if anything
  0-5% slower). Toggling admission changes the per-node *build* path but does not
  move an instance between the NLP-per-node and LP-per-node engines — so it
  cannot and does not change the regime.

The honest reading: admission and the fast regime share a common cause — the
model being in scope for the McCormick-LP-per-node engine. Nice structures (clean
bilinear/integer QCQP) both admit *and* route to the LP loop. **This vindicates
the #861/#896 closures** (admission itself is not a lever) while pointing at a
much larger one.

**Hypothesis.** The lever is *routing and NLP count*: either widen how many
instances reach the LP-per-node engine, or cut the 20-41 NLP solves per node the
default path pays.

**Entry experiment.** For the 20-41 NLP/node instances (gear4, ex1225, nvs01,
nvs21), attribute those calls: which caller issues them (strong branching? bound
tightening? heuristics? per-node NLP relaxation?), and how many are *required*
for correctness versus opportunistic. Then determine why each instance does not
route to the LP-per-node engine — is it out of scope structurally, or declined by
a gate that could be widened soundly? Print per-caller counts; a single dominant
caller would make this a narrow, high-value fix.

**Kill criterion.** If the NLP calls are irreducible (each genuinely required for
a sound bound, no cheaper LP equivalent) AND the routing is a true structural
scope limit rather than a conservative gate, then this is an architectural
rewrite rather than a card, and it should be re-scoped as such rather than
attempted incrementally. Also: `certification-gap-plan.md` Phase D already
removed the *separation/strong-branch* POUNCE calls (2.0-2.7x on nvs17/nvs13) —
confirm you are not re-treading that; the remaining calls are a different caller.

**Gates.** Any routing widening is bound-changing (`CLAUDE.md` §5): differential
bound test, feasible-point sampling, flag-gated default-off until a corpus panel
is both cert-clean and net-positive. Reducing NLP *count* while preserving the
same bounds is bound-neutral and must prove exact `node_count`/objective
invariance.

**Caveat.** 12 instances, one machine, under load. The three-order-of-magnitude
NLP/node split is far too large to be a load artifact, but the ordering *within*
each regime is not reliable at this sample size.

### Card D — Root throughput on the presolve-bound class

> **Status:** OPEN. **Depends on:** nothing. **Related:** `performance-plan.md` §10 (2nd).

**Hypothesis.** On instances where the root consumes the whole budget, the tree
never branches — so *root* cost, not per-node cost, is the lever.

**Evidence.** hda: root presolve 18.0% + convexity classification 15.7% = **~34%
of wall before the search meaningfully starts**, and only 3 node solves happen.
§10 attributes the rest to ill-conditioned McCormick LP solves (~13 s, the #671
class), FBBT (5.1 s) and one-time JAX XLA compile (~7 s).

**Entry experiment.** Split the root budget into one-time costs (JAX compile,
presolve, classification) versus per-round costs (FBBT fixpoint, OBBT probes).
One-time costs are amortisable or cacheable; per-round costs need algorithmic
change. Measure which dominates **before** choosing.

**Kill criterion.** If the root cost is irreducible presolve + genuinely
ill-conditioned LP work, this card becomes "improve the LP conditioning" (which
is #671's territory) rather than "make the root faster", and should be re-scoped
rather than pushed.

**Gates.** Bound-neutral regime: `node_count` and certified objective **exactly
unchanged** on the certifying panel. Convexity-classification caching in
particular must be proven not to change any verdict.

---

### Card E — Convexity classification cost

> **Status:** OPEN. **Depends on:** Card D's split (it may be subsumed).

**Hypothesis.** 15.7% of hda's wall in convexity classification is
disproportionate for a **box-independent, one-time** analysis.

**Evidence.** `_classify_model_convexity` 15.7% of hda wall, 3 calls, ~12 s.
Three calls of a model-structural analysis suggests recomputation.

**Entry experiment.** Are the 3 calls on identical inputs? If yes, this is a
caching problem and cheap. If they differ (different boxes/subproblems), it is
not, and the card closes.

**Kill criterion.** **Read `performance-plan.md` §12 first.** FBBT
structural-match caching was exactly this shape — bound-neutral, sound, and moved
wall <3%, so it was reverted rather than shipped. If the entry experiment
projects <5% wall, close this card *without implementing*. That precedent is
binding.

---

## §4 Binding negative results — do not re-propose

These are **falsified or measured-not-helpful**. Re-proposing one requires new
evidence that the original measurement was wrong, stated explicitly.

| # | result | where |
|---|---|---|
| F9 | Finitizing unbounded continuous vars closes the certification stall — **falsified** (FBBT already finitizes 100%) | `performance-plan.md` §8 |
| F13 | Joint OA on the ex6_2 Gibbs/log-sum objective — **falsified and unsound** (objective is nonconvex; alpha-BB ~1e40 worse) | `performance-plan.md` §8 |
| — | Dense→sparse Jacobian routing on hda — **falsified** (bound-neutral, wall unchanged) | `performance-plan.md` §10 (2nd) |
| — | FBBT structural-match caching — sound, bound-neutral, **<3% wall → reverted** | `performance-plan.md` §12 |
| — | `DISCOPT_CUT_INHERIT` — cert-clean but neutral-or-harmful → stays OFF | `CLAUDE.md` §5 |
| — | #861 generic patch tape — **~1.0x measured** (0–5% *slower*); staging also backwards (55% of specs need the aux tape first) | #861 close-out |
| — | #896 hda aux-bound coverage — **3.9% ceiling**; hda runs 3 nodes | #896 close-out |

**The pattern worth internalising:** four of these were *sound but not helpful*.
Correctness is necessary and nowhere near sufficient. Budget entry experiments
accordingly.

---

## §5 Shared measurement infrastructure

Built during #861; reusable by any card. **Prefer extending these to writing new
ad-hoc probes.**

| tool | purpose |
|---|---|
| `discopt_benchmarks/scripts/incremental_admission_sweep.py` | Admission meter; `--baseline` exits non-zero on any regression |
| `discopt_benchmarks/scripts/incremental_node_parity_panel.py` | Per-node patched-vs-cold LP parity; **proven sensitive** by a firing control (76/120) |
| `discopt_benchmarks/scripts/incremental_oracle_check.py` | Parses `minlplib.solu`; **never accepts a typed reference value**; separates certified objectives from `time_limit` incumbents; reports `NO-ORACLE` rather than passing silently |
| `discopt_benchmarks/data/local_oracle.json` | The 2 corpus instances MINLPLib does not carry (`st_e17` = 376.291905403861, `meanvar` = 5.24339865067014), established by SCIP/BARON/Couenne with provenance |
| `IncrementalMcCormickLP.decline_reason` | Decline reason as a first-class attribute |

### Measurement rules with teeth (from `CLAUDE.md` §6–§11, all earned)

* **Prove the probe fired.** Print an executed-assertion count; exit non-zero at
  zero. A probe that measures nothing reports "0 violations" and reads as a pass.
* **Verify the operation SUCCEEDED before timing it.** During #896 a "warm-start
  gives 1.00x" result was produced by an LP that had **returned `None`** both
  times — it timed a failure twice, with `in_basis` silently `None`.
* **Never hand-type a reference value.** Two false-optimum reports in #861 came
  from typed oracle values; one instance had no reference at all.
* **Timing needs interleaving, a load gate, and a spread** — and state your
  *resolvability*: at 1–8% relative sd, a 1.3x effect is 4–30 sigma out (so the
  null is resolvable) but anything under ~5% is not.
* **Do not create your own load.** Running three profiling agents concurrently
  invalidates all three. Serialise timing work, or report shares/ratios only.

---

## §6 Adding a workstream

1. Write the card **before** the code, with all six fields. If the hypothesis is
   not falsifiable, or you cannot write a kill criterion you would actually
   honour, it is not ready.
2. Check §4 — if it is a binding negative, you need new evidence that the
   original measurement was wrong.
3. Profile the **whole solve** first. Never scope from a microbenchmark
   (this is the #861/#896 lesson, §0).
4. Run the entry experiment. If it kills the card, record the falsification in
   `performance-plan.md` and close it. **That is a success.**
5. Ship behind the gates in `CLAUDE.md` §5 — bound-neutral changes prove exact
   `node_count`/objective invariance; bound-changing changes need the differential
   test, feasible-point sampling, and a corpus panel that is both cert-clean and
   net-positive.
