# Certifying ex1252 and the gated-configuration MINLP class — anatomy & staged plan

> Status: proposed 2026-07-18, grounded in the #707/#721 measurement chain
> (`docs/dev/performance-plan.md` §6, four falsification/implementation records) and
> the compounding probe run the same day
> (`discopt_benchmarks/scripts/ex1252_compounding_probe.py`). Successor to the #721
> re-scope; every stage below carries an entry experiment and a kill criterion per
> CLAUDE.md Dev-Philosophy #4. Correctness gates (`incorrect_count = 0`, bound ≤
> incumbent) bind on every stage — a stage that can only pass by weakening them is
> dead on arrival.

## 0. Verdict up front

ex1252 is not blocked by one wall but by a **stack of five**, and the top three are
now measured, understood, and (as of PR #726) the load-bearing one is removed. The
remaining distance to certification is **search orchestration + engine robustness**,
not new relaxation mathematics. There is a credible, staged path; each stage is
individually falsifiable and none requires a research-grade invention. Calibration
honesty: SCIP 10.0 reaches 0.35% gap on ex1252 in 120 s **and still does not
certify** — so full certification at default tolerances is a stretch goal; the
binding target is the #721 gate ("global dual materially above ~48k"), with
certification the plausible end state if Stages 1–4 land cleanly.

## 1. Why we cannot certify today: the five-layer anatomy (all measured)

ex1252 (Westerlund pump-configuration): choose line configurations (indicators
`x18–x20`, small integer pump counts `x0–x5 ∈ [0,3]`) so the per-line cubic
head/flow/power physics (`c0–c5`) meets demand at minimum cost. The objective is
integer-multilinear: `(6329.03 + 1800·x15)·x0·x3·x18 + …` per line.

| layer | mechanism | measured effect | status |
|---|---|---|---|
| **L1** — objective multilinear decoupling | term-wise trilinear McCormick relaxes the whole objective product to ~0 at fractional indicators | dual floor **5134** | **fixed** — #707 exact linearization (`DISCOPT_INTEGER_MULTILINEAR_REFORM`) → ~48k plateau |
| **L2** — product-aux decoupling | the reform's own expansion bits go fractional in the LP, so every `v_k = z_k·x15` big-M product relaxes to 0; the cost var never enters the bound (`bound = 6329.03·2` exactly, while relaxed `x15 = 12.44 ≠ 0`) | config-node bound stuck at **12658** vs true ≈128894; RLT lifts it to **57435** | **fixed** — PR #726 coupling RLT (`DISCOPT_MULTILINEAR_COUPLING_RLT`, default-OFF) |
| **L3** — search order | the global dual = min over *open* nodes; shallow indicator-fractional nodes bound at ~13.8–16k and the tree does not prioritize reaching integer-complete config nodes | 4.5× node-level lift → only **+1.4%** global dual at equal 400-node budget | **open — the dominant layer** |
| **L4** — residual cubic gap at configs | at a config node the bound is `12658 + 3600·min(x15)`; the cubic rows floor `x15` at 12.44 vs the 32.28 certification needs (2.6×) | pre-RLT: provably inert to every lever; **post-RLT: compounding unlocked** (see §2) | **open — now tractable** |
| **L5** — engine fragility on narrow boxes | OBBT-pinned / subdivided child boxes return `0.0` / `(numerical)` / `(error)` fallback bounds — sound but weak | a single 0.0 child collapses the proved min-child bound at exactly the nodes certification needs | **open — prerequisite** |

The essential geometry: ex1252's discrete configuration space is **tiny** (3
indicators × pump counts ≤ 4² per line), and each configuration's continuous
subproblem is a ~4-variable nonconvex block that the now-unlocked machinery
squeezes hard. Certification = reach the configs fast (L3), bound them tight
(L4), and never drop a bound to a fallback (L5).

## 2. The compounding measurement (2026-07-18) — the plan's empirical basis

#721 predicted an ordering: *"once the relaxation is strong enough that the
objective cutoff binds, cutoff-driven OBBT/DBBT and primal heuristics will
compound — but relaxation strength must come first."* Both halves are now
measured, on the canonical config node (LINE1, OBBT-tightened, `x0=2, x3=1`):

**Pre-RLT (falsification records, §6 of the performance plan):** flow
subdivision, piecewise partitioning (LP *and* MILP), RLT/PSD/superposition cuts,
and OBBT-with-known-optimum-cutoff **all leave the bound at 12658.06 exactly** —
zero movement, every lever.

**Post-RLT (`ex1252_compounding_probe.py`):**

| lever at the config node (RLT ON, base bound 57435) | result |
|---|---|
| subdivide `x12` | children all **infeasible** except one — OBBT has pinned `x12 = 175.0` *exactly* (flow is determined by the config; not a branching dimension) |
| subdivide `x6` into 4 | live child at **62071** (+8% per split; pre-RLT provably 0%) — the speed axis is the real spatial dimension |
| OBBT (with or without cutoff) | `x12 → [175, 175]`, `x15 → [12.44, 30.89]` within seconds — pre-RLT it moved nothing |
| fragility | two children return `0.0`, others `(numerical)`/`(error)` — the fallback floor destroys the proved min-child bound |

The OBBT result is qualitatively new information: capping `x15 ≤ 30.89 < 32.28`
at this config means the machinery is now generating exactly the kind of
per-configuration eliminations certification is made of. The compounding is
real; what remains is plumbing it into the search and not fumbling the bounds
(L5).

## 3. What SCIP does that we don't (calibration, not aspiration)

SCIP's 128438 (0.35%) at 120 s versus our ~48k plateau is **not** a tighter cubic
envelope — it is (a) hybrid/reliability branching that immediately identifies the
indicators and pump-count integers as the high-pseudocost dimensions and drives
the tree through the config space first, (b) per-node OBBT/propagation that
squeezes each config box the way §2 shows ours now can, and (c) numerically
robust node LPs that never hand back a 0.0. That is Stages 1–3 below, exactly.
SCIP *still* fails to close the last 0.35% in 120 s — which is why certification
is the stretch goal and "materially above 48k" is the binding gate.

## 4. The staged plan

Stages are ordered by dependency; each has an entry experiment (cheap, before
implementation) and a kill criterion. All bound-changing pieces stay behind
default-OFF flags until the CLAUDE.md §5 differential panel passes.

### Stage 0 — objective-coupling RLT *(done — PR #726)*
The unblocking lever. Default-OFF `DISCOPT_MULTILINEAR_COUPLING_RLT`; OFF path
byte-identical to #707; sound (RLT of valid identities, `bound ≤ opt` verified);
pinned by `python/tests/test_ex1252_coupling_rlt.py`.

### Stage 1 — engine hardening on narrow/pinned boxes *(DONE 2026-07-18)*
**Problem:** child solves on OBBT-pinned or finely subdivided boxes returned
`0.0`/`(numerical)`/`(error)` fallbacks (§2), collapsing proved bounds.
**Root causes found (bisect: deterministic, order-independent — not warm/pool
state):**
1. *Directional-widening bug in the conditioning clamp* (`solve_at_node` +
   `sanitize_relaxation_for_conditioning`): a bound crossing the 1e10 numeric cap
   was mapped to ±inf **by sign**, so a large-*positive lower* bound became
   `+inf` — a pinned `[+inf, +inf)` box, not the documented widening. The ex1252
   `x6³` monomial aux (lb 1.9e10 on high-speed sub-boxes) hit exactly this; the
   simplex reported the nonsense LP `unbounded` and the objective-floor fallback
   collapsed the child bound to 0.0. **Fixed**: a crossing bound now always
   widens outward (lo → −inf, hi → +inf); the ±sentinel cases the clamp was
   written for behave identically.
2. *Crossed (empty) boxes crashed the build* into a diagnostic-free `error`:
   OBBT correctly flags a crossed tightening as `infeasible=True` (a proof the
   box is empty — at this config it fires at rounds=8, pruning the whole config
   outright), but a crossed box handed to `solve_at_node` crashed
   (`Interval lo > hi`). **Fixed**: `solve_at_node` now answers a genuinely
   crossed box with the definitionally correct `infeasible`, and repairs
   hair-crossings (float round-off) by widening — which can never false-prune.

**Measured after the fix** (same battery): children 2/3 go `0.0 → 66932 / 92706`
(certified, monotone in `x6`; proved 4-way min-child bound 62071 vs 0.0 before),
and the OBBT-crossed box returns `infeasible` (correct prune) instead of `error`.
Both LP relaxations only widen or prune-on-empty, so the change is sound by
construction; `python/tests/test_narrow_box_bounds.py` pins fails-before/
passes-after (4 tests, incl. the false-prune guard for hair-crossings).

### Stage 2 — configuration-first branching *(entry experiments run 2026-07-18 —
kill criterion FIRED for the hint route; re-scoped to the disjunctive route)*
**Hypothesis:** branching indicators → reform expansion bits (`_ipx_e*`) → pump
counts *before* continuous/spatial dimensions collapses the shallow-node floor.
**Measured** (`ex1252_stage2_branching_probe.py`, reform + RLT ON):
- *Experiment A (instrumented 400-node tree):* the tree spends its branching on
  the reform's **continuous** big-M product auxes (`_ipx_v*`, top column 29
  branchings) whose subdivision provably cannot pin the coupling; the existing
  `DISCOPT_OBJ_BRANCH_PRIORITY` set contains only 6 vars post-reform (its
  detector keys on nonlinear-term participation, and the `_ipx_e` bits — the L2
  leak drivers — appear only in *linear* big-M rows, invisible to it).
- *Experiment B (all-binaries priority, monkeypatch):* **kill criterion fired**
  — 400-node global dual 12658 (below the 16304 baseline, far below the 33k
  gate), incumbent worse. Root cause: the priority hint is *fractionality-gated*
  and the node LP vertex often has **integral binaries while the coupling stays
  loose through the v-auxes** — no fractional signal, no hint, and the standard
  selector keeps branching the useless v columns. Hint-based config-first
  branching is falsified for this tree (the mechanism cannot express "branch a
  config dichotomy regardless of fractionality").
- *Experiment C (disjunctive route — the recorded alternative, now primary):*
  enumerate the 2³ line patterns, OBBT each, min over configs = valid root bound:

  | config | bound |
  |---|---|
  | (0,0,1) | 90592 — **pruned outright** under the incumbent cutoff |
  | (0,1,1) | 59806 |
  | (1,0,0) | 38524 |
  | (1,0,1), (1,1,0), (1,1,1) | ~0 — the residual wall |
  | (0,0,0) | infeasible; (0,1,0): numerical |

  Single-line configs certify far above the tree's 16.3k; the multi-line configs
  re-create the L2 decoupling **one level down**: their coupling runs through the
  pump-count integer-**bilinear** products (`x9·x3 = 400·x18` …), which the
  reform expands on its *bilinear* path — **not covered by the #721 coupling RLT**
  (it keys on ≥3-factor multilinear products with a continuous factor).

**Re-scoped Stage 2 deliverables (in order):**
1. *(DONE 2026-07-18)* Extend the bit-linking coupling RLT to the
   **integer-bilinear** expansion path (`_try_expand_mul`) — same identity, same
   default-OFF flag. Measured per-config amplification (OBBT'd config boxes,
   with the incumbent cutoff): (0,0,1) 90592 → **115466** (and pruned outright
   under the cutoff), (0,1,1) 59806 → **90429**, (1,0,0) 38524 → **71644**; the
   line-1-only config node lifts 12658 → **65654** (5.2×). Multi-line configs
   stay ~0 — correctly: their pump-count integers are free, so the coupling
   grounds out at a genuinely loose McCormick; they need one recursion level
   (deliverable 2). Two side-findings: (a) the extension tightened the *old*
   test anchor (LINE1 + `x0=2, x3=1`) enough to expose that the config is
   **genuinely infeasible** (OBBT rounds=8 proves the box empty; the raw node LP
   now goes `numerical`-inconclusive rather than bounding a vacuous 57435) — the
   test pins moved to the feasible line-1-only config; (b) the raw-node simplex
   cannot Farkas-certify that emptiness (ill-conditioning) — an engine-hardening
   item to fold into Stage 1's scope if it recurs on feasible boxes.
2. A root **disjunctive-bound pass** over the indicator patterns (class-gated,
   default-OFF): global dual = max(root bound, min over configs of the per-config
   OBBT+LP bound), recursing one level (pump counts) on configs that stay weak.
   With deliverable 1's amplification, the surviving weak set is exactly the
   multi-line configs — the recursion targets their pump-count integers.
3. Only then revisit in-tree branching (the correct mechanism is dichotomy
   branching on config variables regardless of fractionality — a tree-side
   change, larger scope).

### Stage 3 — per-config compounding loop (OBBT + cutoff at integer-complete nodes)
**Hypothesis:** at integer-complete nodes, a short OBBT loop (now potent, §2)
with the incumbent cutoff prunes or tightens configs toward their true cost;
the incumbent exists (RLT ON found 134471 unaided).
**Entry experiment:** replay the §2 loop across all feasible `(line, x0, x3)`
configs; table of per-config bounds/infeasibilities; how many survive below the
incumbent?
**Kill criterion:** if the per-config loop leaves > half the config space both
feasible and > 20% below incumbent, the cubic gap (Stage 4) is bigger than the
search gap and Stage 4 is promoted above Stage 2 in effort.

### Stage 4 — cubic-block tightening at configs (L4; #721's original directions, now unblocked)
In increasing effort, measured at the OBBT-tightened config box:
1. **Spatial branching on `x6`** — already works (+8%/split measured); Stage 2's
   priority order just needs to reach it after the integers.
2. **Piecewise-McCormick auto-trigger gated on integer-complete nodes** — #721's
   direction #1, provably inert pre-RLT, now expected to bite; entry experiment:
   k = 4/8 partitions on `x6` at the config box, measure the `min x15` lift.
3. **Edge-concave / αBB envelope for the cubic rows** (catalog §7's open item) —
   one-shot tighter envelope of `x15 = a·x6³ + b·x6²·x12 + c·x12²·x6` over the
   config box; highest effort, only if 1–2 stall.
**Kill criterion per option:** < 10% `min x15` lift at the config box → drop that
option (recorded, not retried).

### Stage 5 — graduation (CLAUDE.md §5)
Class detector: `has_integer_multilinear_reformulation_work(model)` (the #707
trigger) — i.e. the gated-configuration class, *not* an ex1252 special case.
Panel: corpus differential run, flag ON vs OFF — cert-clean (`incorrect_count =
0`, no bound above reference, incumbents feasibility-verified) AND net-positive
(node count / wall / bound). ex1252a rides along as the sibling instance. The
coupling RLT + config-first branching + per-config OBBT graduate together or
stay default-OFF with the measurement recorded.

### Alternative route (recorded, not primary): disjunctive/hull bound
Because the config space is tiny, a root-level **disjunctive bound** — enumerate
indicator patterns, bound each with the §2 machinery, take the min — would
short-circuit L3 entirely; the GDP hull/perspective machinery
(`gdp_reformulate.py`, the AMP convhull formulations) is the principled version
(perspective-tighten the gated cost rows). Higher integration cost; becomes the
primary route only if Stage 2's kill criterion fires.

## 5. Generality — "and similar problems"

The class this plan serves (and the detector gates on): **small discrete
configuration space gating nonconvex continuous unit models, coupled to cost
through integer×continuous products.** ex1252/ex1252a are the named probes;
pump/compressor/network-synthesis instances in MINLPLib share the shape, and the
#707 trigger already recognizes it structurally. Per Dev-Philosophy #2, every
stage keys on the structure (integer-multilinear work + gated blocks), never on
an instance name; the graduation panel is corpus-wide.

## 6. Honest end-state assessment

- **High confidence:** Stages 1–3 lift the global dual well past the 48k plateau
  (the per-config machinery demonstrably prunes/tightens; the config space is
  enumerable).
- **Medium confidence:** Stage 4 closes config gaps far enough that the cutoff
  prunes all non-optimal configs — the shape of certification.
- **Known residual risk:** the optimal config's own gap must close to tolerance;
  SCIP's 0.35%-and-stuck shows this tail is where the hard part lives. If the
  tail resists Stage 4, the recorded fallback is the αBB/edge-concave envelope
  (Stage 4.3) and, beyond it, honesty: record the frontier, as #673/#677 did for
  the autocorr class.
