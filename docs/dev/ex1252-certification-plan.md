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
2. *(DONE 2026-07-18)* The root **disjunctive-bound pass**
   (`discopt/_jax/disjunctive_config_bound.py`, flag
   `DISCOPT_DISJUNCTIVE_CONFIG_BOUND`, default-OFF): enumerate the reform's
   indicator patterns, bound each configuration box by **FBBT → OBBT → LP**
   (the per-box FBBT closes the `numerical` leaves in ~4 ms), best-first
   **unit-peel** the weakest leaf on the configuration count variables, under a
   leaf/wall budget. Anytime-valid: children inherit the parent bound until
   certified, so min-over-leaves is valid at any cutoff. Wired via the
   integer-ratio-partitioner precedent: the floor is stashed on the model and
   `MccormickLPRelaxer.solve_at_node` max-combines it into every optimal node
   bound (a root-box bound is valid on every sub-box), flowing through the
   tree's existing plumbing with no new threading. **Measured:** standalone
   root pass (48 leaves, no incumbent, no tree) certifies **37945** — the
   re-scoped Stage-2 gate (≥ 33k) passes; end-to-end at equal solve settings
   (85 nodes, 240 s) the reported global dual goes **0.0 → 42725** with the
   same incumbent (134471). Two operational findings: (a) *incumbent-cutoff
   OBBT inside the pass is net-negative* — it degrades LP conditioning enough
   that leaves stay uncertified and the min collapses to the inherited floor
   (160-leaf run returned exactly the floor) — so the wiring runs the pass
   cutoff-free; (b) the surviving weak leaves are the deep all-pumps-on chains,
   whose climb is bounded by the per-leaf LP tightness (Stage 4's cubic
   levers pick up from here). Tests:
   `python/tests/test_disjunctive_config_bound.py` (decline, anytime validity,
   the ≥ 33k gate, and the relaxer floor plumbing).
   *Recursion entry experiment (2026-07-18,
   `ex1252_pump_recursion_probe.py`, config (1,1,0), zero-dichotomies `{0}` vs
   `[1,3]` on the 4 active pump vars):* **12/16 children prune** (LP-infeasible)
   — the dichotomy eliminates hard; 2 children bound (62027 and 6814 — the
   all-pumps-on child needs one value-split level `{1},{2},{3}` to climb); 2
   children are `numerical` — genuinely infeasible (head equation
   `x18 = 0.0025·x9·x3` with both line-1 pumps zero) but the ill-conditioned LP
   cannot Farkas-certify it and OBBT does not cross. **The same contradiction is
   provable by interval FBBT alone**, so the pass must run a per-box FBBT step
   before the LP — the Rust binding exposes exactly this
   (`tighten_var_bounds` + `fbbt`/`fbbt_with_cutoff`). Implementation shape:
   per config box, FBBT → OBBT(cutoff) → LP; prune on any of the three; recurse
   zero-dichotomies then value-splits on the weakest surviving child, under a
   node/LP budget; return max(root bound, min over leaves).
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

### Stage 4 — cubic-block tightening at configs *(option 1 DONE 2026-07-18 — the
plateau breaks; #721's acceptance bar exceeded)*
In increasing effort, measured at the OBBT-tightened config box:
1. **Spatial bisection of the cubic block at count-complete leaves — DONE.**
   Kill bar (≥ 10% leaf lift) was already met by the Stage-1 battery (config-box
   x6 children: parent 65654 → min-child 73848, +12.5%). Implemented inside the
   disjunctive pass: once a leaf has no configuration count left to peel,
   bisect the **widest-relative continuous nonlinear participant** (structural:
   nonlinear-term participants minus counts/indicators — the x6-class), capped
   at `max_spatial_depth` bisections per path. **Measured:** the standalone pass
   goes 37945 (48 leaves — byte-identical to Stage 2, spatial only engages
   deeper) → **63080 at 120 leaves** (129 s, no incumbent, no tree) — through
   the ~48k plateau, i.e. **the original #721 acceptance criterion ("global
   dual climbs materially above ~48k") is formally exceeded**. Wiring budget
   raised to `min(25% of time_limit, 150 s)` with a deadline-governed leaf
   budget (the module's 48-leaf default was capping the in-solve pass at the
   shallow regime). **End-to-end: the reported global dual on a 600 s solve goes
   0.0 (flag OFF) → 74915 (flag ON)** — 58% of the optimum, with a better
   incumbent as well (131124 vs 134471); the in-solve pass beats the standalone
   63080 because the root box is presolve/FBBT-tightened before it runs. Session
   trajectory of the certified dual: 5134 (pre-#707) → ~16k (#707) → 42725 →
   **74915**. Pinned by `test_ex1252_spatial_recursion_clears_721_bar` (≥ 48k).
2. **Piecewise-McCormick auto-trigger gated on integer-complete nodes** — #721's
   direction #1; held in reserve — option 1 did not stall, so per the plan's
   ordering this stays unbuilt until the spatial lever plateaus (record the
   measurement then).
3. **Edge-concave / αBB envelope for the cubic rows** (catalog §7's open item) —
   highest effort, only if 1–2 stall.
**Kill criterion per option:** < 10% `min x15` lift at the config box → drop that
option (recorded, not retried).

### Stage 5 — graduation panel *(RUN 2026-07-18 — verdict: NOT eligible; flags
stay default-OFF, measurement recorded)*

**Instruments:** `check_cert_neutrality` OFF (environment sanity) and ON (the
flag stack: reform + coupling RLT + disjunctive bound) over the 49-instance cert
panel, plus a 9-instance ON-vs-OFF differential on the reform-firing subset
(nvs01/05/09/16/22, st_e36/e40, ex1252, ex1252a) at the 60 s fair budget.

**Findings:**
1. *Environmental noise:* clay0303hfsg / st_e38 / tls2 violate the committed
   baseline **in the OFF arm too** (this container is slower than the baseline
   machine) — not flag-attributable.
2. *nvs09 loses its certificate ON* (optimal in 9 s OFF → 0 nodes, wall overrun
   ON). **Attributed to the pre-existing #707 reform flag**: reform-only
   reproduces it identically (78 s, 0 nodes) with none of this branch's code —
   a root-phase stall on the reformed nvs09 needing its own diagnosis.
3. *Short-budget dual regression on the flagship:* at 60 s, reform+RLT costs
   ex1252's tree bound (14347 → ~0; bigger LPs → fewer nodes at equal time)
   while improving the primal side (incumbent 204321 → 134471). The stack pays
   off only at generous budgets (600 s: 0 → 74915).
4. *Two defects the panel caught were fixed on the spot:* the pass's leaf
   solves carried no time limit (nvs09 ON overran a 60 s budget to 115 s; with
   the fix, nvs09@300 s **certifies optimal with the full stack ON**, 103 s),
   and the pass now engages only when its budget is ≥ 45 s (below that the
   stack's pass is byte-identical to OFF — the 60 s panel behavior is restored
   to reform+RLT-only).
5. *Genuine wins where structure fires:* nvs05 bound 3.81 → 4.10 (equal
   nodes); ex1252 primal 204321 → 134471; ex1252 dual 0 → 74915 at 600 s;
   6 of 9 firing instances byte-identical (guards reject the reform).

**Verdict (per the `DISCOPT_CUT_INHERIT` discipline):** sound throughout (no
bound ever crossed its reference; every violation is environmental or
pre-existing-#707), but **not net-positive at the fair budget** — the flags stay
default-OFF. Graduation blockers, in order: (a) the pre-existing #707 nvs09
root-phase stall, (b) the short-budget dual cost of reform+RLT on the
ex1252 class (candidate fix: budget-aware reform adoption). Re-run this panel
after those land.

### Stage 5 — blockers resolved *(2026-07-18, second pass)*

Both graduation blockers were root-caused and fixed; the panel was re-run.

**Blocker (a) — nvs09 "root-phase stall" was a reform-build monomial blowup.**
The stall is not in the LP/search but in the reformulation *build*: nvs09's
objective carries a **10-factor** integer-multilinear product (`[3,9]` each →
`4^10 ≈ 1.05M` distributed binary monomials), each ≥2-bit one minting an AND aux,
so `_try_expand_multilinear` explodes for minutes before the post-build column
guard can reject it (faulthandler lands in `and_product`). **Fix:** an early
estimate of the distributed monomial count (`∏_i(1+nbits_i)`, from the factor
ranges alone, no columns minted) aborts the whole pass once the cumulative
estimate exceeds `max(5000, 12·n_orig)` — caught by `expand_integer_products`'
existing `except → return model`, so a blown-up reform degrades to exactly the
flag-off model (bound-neutral: any reform that trips this would be rejected by
the post-build/adoption guards anyway; the cap sits ~46× above the largest
observed legitimate reform, ex1252 cum-est 108, and ~280× below the nvs09
blowup). **Measured:** nvs09 flag-ON now certifies optimal in ~11 s / 5 nodes
(`gap_certified=True`), was hanging > 150 s. ex1252/ex1252a/nvs05 reform to
byte-identical column counts. Pinned by
`test_integer_multilinear_reform.py::{test_wide_multilinear_reform_guard_degrades_instead_of_hanging,
test_nvs09_reform_on_certifies_and_terminates}`.

**Blocker (b) — the short-budget dual cost is now avoided by budget-aware
adoption.** On the gated-configuration class the reform's payoff is the
disjunctive config-bound floor / deep spatial recursion, which only engages at a
generous budget (the disjunctive pass engages at `min(0.25·time_limit,150) ≥ 45`
s, i.e. `time_limit ≥ 180` s). Below that the exact-linearization is pure
per-node-LP cost: measured on this container, ex1252@60 s flag-ON collapsed the
tree dual **9273 → 0** and lost the incumbent, vs the flag-off spatial path's
9273. **Fix:** a non-pure-MILP reform that carries configuration indicators is
adopted only when the budget affords the payoff pass; below that it keeps the
flag-off path. A non-config reform (nvs05: payoff is the direct node-LP
tightening, +0.19 dual at equal nodes @60 s) and pure-MILP reforms are
unaffected. **Measured after:** ex1252@60 s flag-ON is byte-identical to flag-off
(dual 9273, 31 nodes); ex1252@200 s still adopts and lifts (dual 31459, incumbent
134471). Pinned by
`test_integer_multilinear_reform.py::test_ex1252_short_budget_declines_reform_no_regression`
(deterministic node-limited byte-identity). A hygiene rider silences the spurious
`milp_simplex` divide overflow on the ill-conditioned reform boxes (bound-neutral
`errstate`, finishing Stage 1's robust-node-LP mandate).

**Re-run differential (ON vs OFF, 60 s, this container —
`discopt_benchmarks/scripts/ex1252_stage5_differential.py`):**

| instance | OFF dual / incumbent | ON dual / incumbent | note |
|---|---|---|---|
| nvs01, nvs16, nvs22, st_e36, st_e40 | (optimal, certified) | identical | byte-identical, guards reject the reform |
| **nvs09** | optimal −43.13 / 5 nodes / **cert** | identical, **cert** | **blocker (a) fixed** — ON now certifies (was hanging, uncertified) |
| **ex1252** | 9273 / 204321 | **identical (9273 / 204321)** | **blocker (b) fixed** — the 9273 → 0 collapse is gone; reform declined below its payoff budget |
| nvs05 | 3.81 / 8.732 | identical | the Stage-5 "3.62 → 3.81 win" was **wall-clock-limit noise**; both arms reach 3.81 here |
| ex1252a | 14086 / 177861 | 14086 / **183660** | dual identical; ON incumbent slightly worse — a *sound* primal wobble from the **narrow-box-branch rider** (the reform is declined at 60 s, so this is the only ON/OFF delta) |

**Soundness: clean** — no dual above its optimum, no `bound > incumbent`, no
certificate regression, on any arm. The two Stage-5 regressions (nvs09 cert loss,
ex1252 dual collapse) are **both eliminated**.

**Generous-budget check (200 s, config instances):** the reform stack is
**net-negative on this container** — ex1252 OFF **46875** vs ON **31458**; ex1252a
OFF **46368** vs ON **38746** (ON incumbent also worse, 147745 vs 131564). Root
cause (logged): the disjunctive pass engages correctly (reported dual *equals* its
floor, no wiring bug) but is **leaf-throughput-limited on this slow container** —
only **29 leaf solves in its 50 s budget** → floor 31458, versus the baseline's
48–120 leaves → 37945–63080. Meanwhile the **flag-off spatial path has become
strong** (46875 @ 200 s, climbing — vs the plan's earlier "OFF ≈ 0 @ 600 s", a
different/older tree), so the pass's 50 s tax + the reform's heavier per-node LPs
are not repaid here. **This falsifies "the reform pays off at generous budgets"
_on this container_** (Dev-Philosophy #4 — measurement recorded); whether it still
holds on the faster baseline machine (where the pass reaches 120 leaves) is
untested here.

**Verdict (`DISCOPT_CUT_INHERIT` discipline): flags stay default-OFF, measurement
recorded.** The two graduation *blockers are resolved* — the stack is now **safe**
(no reform-build hang, no certificate loss, no short-budget dual collapse) and
sound throughout — but it is **not net-positive on this container**: neutral at
the 60 s fair budget (the reform is correctly inert on the config class below its
180 s adoption budget) and net-negative at 200 s (container-limited pass
throughput + a strong OFF path). This is a strict improvement over the prior
verdict (which had *active* regressions), and the flag is now safe to opt into,
but it does not clear the net-positive bar. Re-running on a baseline-speed panel
(where the disjunctive pass reaches its 48–120-leaf regime) is the remaining
graduation question; the blocker fixes ship regardless as a default-OFF increment.
**All three fixes are sound and kept: the monomial-blowup guard (a real hang +
certificate fix, valuable independent of graduation), the budget-aware adoption
gate (prevents the measured 60 s collapse; unchanged at ≥ 180 s), and the
`milp_simplex` errstate hygiene (bound-neutral).**

### Stage 5-original — graduation (CLAUDE.md §5)
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
