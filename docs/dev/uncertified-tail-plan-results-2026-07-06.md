# Uncertified-tail plan — V2 results (2026-07-06)

**Status:** measured. The V2 gate-wiring + measurement-of-record task of
`docs/dev/uncertified-tail-plan-2026-07-06.md` §3 V2 (and cert-gap-plan §11 /
§14 T2.6 items 1–2). Two parts: (1) wire the long-missing
`[suites.global50]` / `[gates.cert2]`; (2) the flag-ON BARON re-run that tallies
what the tail plan's R-series actually shipped.

**Headline (honest):** **R4 (#518) is the only shipped tail win.** With
`DISCOPT_LIFT_ZERO_SPANNING_FACTORS=1` (default OFF), **st_e36** moves
feasible/TL → **optimal** (153 nodes, ~17 s), lifting **proved-optimal 42 → 43**
vs the 2026-07-06 V1 baseline. This is *not* the plan's original ≥46 target —
that assumed R2 + R3b landed. R3b was killed by its own entry experiment (which
also revealed R3a's fingerprint was measured on the wrong model — the
factorable-only reform, not the integer-bilinear model the tree actually
branches — so the tail is relaxation-limited, not selection-limited), R2 is
deferred (broad, non-tail), R5 was not built, and F5/F6 were
already killed. The hard branch-and-reduce / no-bound tail (nvs05, nvs09, hda,
tanksize, casctanks, …) is **unchanged** — those need deeper lifting /
relaxation coverage (#517), out of this plan's scope.

**Method.** `discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py
--time-limit 60 --out-dir reports`, 61 vendored MINLPLib `.nl`, discopt
(isolated subprocess, **release** maturin build) vs full-license BARON via
GAMS 53, MINLPLib `primalbound` oracle, abs=1e-6/rel=1e-4. **Release POUNCE**
(4.7 MB `_pounce.abi3.so`) — the debug-build artifact (pounce#182) does not
apply. The flag-ON run sets `DISCOPT_LIFT_ZERO_SPANNING_FACTORS=1` in the
environment; it propagates to every discopt subprocess through the worker's
`env=dict(os.environ, …)`. Reports:

- flag-ON (this run): `reports/global_opt_baron_vs_discopt_2026-07-06T06-15-16.{json,md}`
- V1 baseline (flag-OFF, on main): `reports/global_opt_baron_vs_discopt_2026-07-06T03-25-20.json`

---

## 1. Gate wiring (Part 1)

Every `[gates.cert*]` in `benchmarks.toml` named `suite = "global50"`, but no
`[suites.global50]` table existed (the recon flag in cert-gap-plan §11 "Wiring
corrections 2026-07-02"). V2 closes that.

**`[suites.global50]`** — points at the vendored `config/baron_global50.txt`
(50 instances). 49 have a `.nl` in the local corpus
(`python/tests/data/minlplib_nl/`) and run with no `--fetch`; st_miqp5 is listed
but ships only as a binary `.nl` discopt cannot parse (R1 skipped it for the same
reason), so the suite resolves to 49 locally / 50 with `--fetch`. 60 s
per-instance cap matches the measurement of record.

**`[gates.cert2]`** — wired per §11 with the recorded corrections:

| criterion | metric | bound | notes |
|---|---|---|---|
| `zero_incorrect` | `incorrect_count` | max 0 | the zero-slack correctness gate (live) |
| `root_gap_vs_baron` | `root_gap_ratio_vs_baron` | max 1.3 | computed from A3's now-live `SolveResult.bound` (null before A3) |
| `geomean_vs_baron` | `geomean_ratio_vs_baron` | max 2.5 | **existing** metric name (§11 correction 2 — `geomean_time_ratio_vs_baron` was never implemented) |
| `tree_closed_fraction` | `closed_within_10_nodes_fraction` | min 0.30 | new T2.6 metric + dispatch; the "≥ 30% close within 10 nodes" exit |

**Reference-bound caveat (§14 T2.6 item 3).** Many BARON rows carry no usable
root/dual bound (`status="unknown"` at ~20 ms), so their `root_gap` is `None`.
`root_gap_ratio` excludes null-`root_gap` rows from **both** numerator and
denominator, guards the denominator with `>1e-10`, and skips a non-finite
discopt gap — it never divides by a non-bound. When no instance has a usable
bound on both sides it returns NaN → the gate reads NaN → fail (an honest "not
evaluable", never a spuriously-passing empty mean). Docstring records this.

**New metric — `closed_within_10_nodes_fraction`.** Denominator = rows that
reached a *proven* optimum (`status == OPTIMAL`); numerator = the subset that did
so in ≤ 10 B&B nodes (`node_count == 0`, e.g. a root fathom / convex fast path,
counts as the tightest close). A feasible-only (uncertified-tail) or errored row
is not in the denominator — the metric measures how tightly the proofs close,
not the proof rate.

**Tests (all green, 6 new).** `discopt_benchmarks/tests/test_cert_instrumentation.py`:
a fixture with known node counts (0, 7, 10, 153) → known fraction 0.75; the
boundary (node_count == 10 closes, 11 does not, 0 is tightest); the
denominator-is-proved-only guard (a 900-node *feasible* row does not dilute);
empty/tail-only → 0.0; the `[suites.global50]` + `[gates.cert2]` config presence;
and end-to-end `evaluate_phase_gate('cert2', …)` dispatch of the new metric
(0.75 → passes ≥ 0.30) with the zero-slack correctness guard.

**cert2 is informational until the R-flags flip default-on** (3 green nightlies,
out of V2 scope): T2.3–T2.5 are not built and R4 ships default-OFF, so
`root_gap_vs_baron` / `geomean_vs_baron` are not expected to pass on `main` yet.
`zero_incorrect` is the live gate.

---

## 2. The measurement of record (Part 2) — flag-ON BARON re-run

### 2.1 baseline (V1, flag-OFF) → V2 (flag-ON)

| metric | V1 baseline (flag-OFF) | **V2 (flag-ON)** | Δ |
|---|---:|---:|---:|
| proved optimal | 42 | **43** | **+1** (st_e36) |
| correct (`ok`) | 51 | 51 | — |
| `GAP` (honest suboptimal) | 1 | 1 | — |
| `n/a` (no oracle / no incumbent) | 9 | 9 | — |
| **VIOLATIONS** | 0 | **0** | — (hard gate) |
| uncertified tail (not proved-opt) | 19 | **18** | **−1** |
| rows with a live dual bound | 51 / 61 | 51 / 61 | — (A3) |
| total discopt wall | 1130.0 s (18.8 min) | **1094.8 s (18.2 min)** | −35 s (st_e36 faster) |

Exactly **one** instance changed status — **st_e36** (feasible → optimal).
Everything else is bit-for-bit the V1 baseline (same status, same verdict), as
the single-instance thesis predicted (§2.4).

### 2.2 Hard gates (all hold)

- **0 correctness violations** — no run claimed a certified global at the wrong
  value, returned an incumbent better than the proven global, or reported a dual
  bound crossing its oracle. The A3 bound-vs-oracle check is live on **51/61**
  rows and found **0 crossings**.
- **No certification lost** — proved-optimal *increased* 42 → 43; the set of
  baseline-proved instances is a subset of the flag-ON proved set (0 lost).
- **`time_limit` contract holds** — cap = 60·1.1 + 5 = **71 s**; **0 instances
  over cap** (slowest is bchoco08 at 66.7 s, a hard time_limit both ways).
- **Bound-vs-oracle clean** — 0 dual bounds cross the oracle (checked above).

### 2.3 The st_e36 row (the one instance that moves)

In the full flag-ON run st_e36 is `optimal`, obj −246.0, bound −246.02, **153
nodes, 15.8 s** (was `feasible`/TL, bound −304.49, 883 nodes, 60.1 s in the V1
baseline). The cheap single-instance flag-OFF↔ON confirmation (same build, BARON
skipped) reproduced the delta independently:

| flag | status | bound | nodes | wall |
|---|---|---:|---:|---:|
| OFF | feasible (TL) | −304.49 | 883 | 60.2 s |
| **ON** | **optimal** | **−246.0** | 153 | **17.3 s** |

This is exactly the R4/#518 acceptance: branching the zero-spanning product
factor `w = f1` (removed from the branch-deprioritized set only when the flag is
on) un-pins the McCormick product bound. Relaxation math and feasible set are
untouched — only branch *ordering* changes, always sound.

### 2.4 Why one flag-ON full run is sufficient (no separate flag-OFF full run)

R4 already proved flag-OFF is **byte-identical** to the V1 baseline on the
41-instance cert panel (`check_cert_neutrality.py`: all `nodes X→X`, `|Δobj|=0`,
NEUTRAL). The F5-style corpus scan (1610 MINLPLib `.nl` ≤ 200 KB) found the
zero-spanning product-factor structure **only in st_e36**, so the flag tags an
aux — and therefore changes behaviour — on **exactly one** instance. The V1
baseline JSON is the flag-OFF full tally; the flag-ON run differs from it only at
st_e36. The §2.3 single-instance flag-OFF↔ON delta confirms the difference is
confined there.

---

## 3. What did NOT move — and why (honest scope)

The uncertified tail shrank by exactly one (19 → 18: st_e36 left). The remaining
**18** are unchanged vs the V1 baseline (same status, same bound):

```
4stufen  bchoco06  bchoco07  bchoco08  beuster  casctanks  contvar  hda
heatexch_gen1  heatexch_gen2  heatexch_gen3  nvs05  nvs09  tanksize  tls2
tspn08  tspn10  tspn12
```

The hard tail is unchanged by design:

- **nvs05 / nvs09** — R4 tags **no** aux (bound identical ON/OFF). Their stall is
  a different structure (nvs09: `log·log` product that can't be linearized;
  nvs05: loose multilinear/signomial), not a zero-spanning product factor. R4's
  lever is specific to the st_e36 class.
- **hda** — no zero-spanning product factor to tag; its 23 omitted rows are a
  `log(<general expr>)` / `x**expr` relaxation-coverage gap
  (`_LIFTABLE_CALL_OUTER = {sqrt, exp}` deliberately excludes `log`). Filed as
  **#517**; not forced into R4. **TD-B (§5) investigated the `log`-argument lift
  and KILLED it**: the reform pass is gated off on hda (convexity classification
  times out on 722 vars), the reform's root LP is pre-existingly infeasible, and
  the no-bound flowsheet class's log arguments all span zero (unrelaxable). Still
  no dual bound.
- **tanksize, tls2** — reduction-responsive (T2.1/R1) but need R2's root fixpoint
  loop, which is deferred (its generality arm passed but its value is broad
  small-MINLP root closure, not the hard tail).
- **casctanks, 4stufen, bchoco06/07/08, beuster, contvar, heatexch_gen1/2/3,
  tspn08/10/12** — Class H (BARON-also-hard); this plan treats them as
  beneficiaries, not gates. Unchanged.

---

## 4. Ledger of R-tasks (shipped / killed / deferred)

| task | outcome | evidence |
|---|---|---|
| **R1** (T2.1 root-loop replay revisit) | **DONE → GO on R2** | full panel completes (19/20, loop-median 0.27 s); no P0 on 51 instances; generality arm 9/29=31%≥20% (cert-gap-plan §14 T2.1-revisit) |
| **R2** (root fixpoint loop + reduce_node) | **DEFERRED** | unlocked by R1 but broad / non-tail; value is small-MINLP root closure, not the hard tail — not built in this arc |
| **R3a** (responsiveness fingerprint) | **DONE — verdict BUILD R3b** | overlap 1/6 < 4 → BUILD; but the fingerprint was measured on the factorable-only reform (a methodology flaw R3b later caught); #515 |
| **R3b** (responsiveness-aware branching) | **KILLED** (its own entry experiment) | force-branching the "responsive" var made it *worse* (nvs05 821→853, nvs09 bound worsened, tls2 unchanged); the branched model is 15-var integer-bilinear where R3a's columns don't exist → relaxation-limited, defer to R4; #516 |
| **R4** (zero-spanning-factor lifting) | **SHIPPED (flagged, default-OFF)** | st_e36 feasible/TL → optimal (153 nodes, ~17 s); differential-bound + 20 000-pt feasible sampling + flag-OFF byte-identical all green; #518 |
| **R5** (zerohalf pre-crossover) | **NOT BUILT** | optional / opportunistic; graphpart cut geometry (cert-gap-plan §7) — out of this arc |
| **F5 / F6** | **KILLED (prior)** | envelope math / node-NLP warm-start are not the tail's levers (perf-followup-results-2026-07-06 §5) |
| **V2** (this doc) | **DONE** | gate wiring + flag-ON tally: proved-optimal 42 → 43 |
| **TD-B** (`log(<expr>)` call-argument lift, §5) | **KILLED** (its own entry experiment) | hda gets no bound: reform gated off (classify timeout, 722 vars) + pre-existing root-LP infeasibility + 17/23 omitted rows outside a `log`-arg lift; no-bound flowsheet class's log args span zero; contvar *regresses* (timeout). Mechanism sound but no bound benefit → not shipped; #517 |

**Follow-ups filed:** **#517** (hda / `log`-argument relaxation-coverage gap —
TD-B (§5) re-scoped it to the true blockers: reform gating on large models + the
reform's root-LP infeasibility on hda; the `log`-arg lift is *not* the lever);
R2 remains available if the small-MINLP root-closure value is prioritized later.

**Default flip (R4):** NOT in this arc — `DISCOPT_LIFT_ZERO_SPANNING_FACTORS`
stays OFF until 3 consecutive green nightlies per §0.1.

---

## 5. TD-B — lift `log(<expr>)` call arguments for relaxation coverage (KILLED by entry experiment; #517)

**Track:** deeper-tail relaxation-coverage arm — close the gap that leaves **hda**
(and the no-bound flowsheet class) with NO dual bound by extending the factorable
reform's liftable-call set (`factorable_reform._LIFTABLE_CALL_OUTER = {sqrt, exp}`)
to `log`-family, lifting `w == <inner expr>` (FBBT-bounded) and relaxing the outer
`log(w)` over the aux with the existing concave-log secant/tangent envelope, guarded
by a certified strictly-positive argument lower bound.

**Regime:** bound-changing (a reformulation changes the relaxation). Feature flag
`DISCOPT_LIFT_CALL_ARGUMENTS`, default OFF, was the plan.

### 5.1 Entry experiment (run BEFORE building — it gated, and killed, the task)

**hda baseline (flag-OFF, stock `main`):** `status=time_limit, bound=None,
node_count=7` at 20–60 s — the tree has no dual bound and can never fathom.
Confirmed the 23 omitted rows via the `AMP: omitting constraint … cannot be
linearized safely` path:

| bucket | count | example |
|---|---|---|
| `Cannot linearize FunctionCall: log(<expr>)` | 6 | `log((0.333333·4^(0.0001+0.333·x0)) − 0.333333)`; `log(((x99/x156)·x100)/x159)` |
| `Cannot linearize FunctionCall: sqrt(<expr>)` | 3 | `sqrt(((x56/x57)·x54)/x55)` (already in `_LIFTABLE_CALL_OUTER`) |
| `Cannot decompose product` | 6 | `(1/(…·sqrt(…)·(…)^-1.5+1))²` |
| `Cannot linearize non-constant division: x/(x^0.230769)` | 6 | `x246/(x204^0.230769)` |
| `Cannot linearize power expression: (1−x)^-1.544` | 2 | `(1−x43)^-1.544` |

So the **`log` rows are only 6 of 23** dropped rows; the other 17 are
divisions / fractional powers / products entirely outside a `log`-argument lift.

**Hand-lift result — KILL CRITERION FIRED (hda gets NO finite bound):**

1. **The reform pass never runs on hda.** `factorable_reformulate` is gated behind
   `has_factorable_work(model) and _classify_model_convexity(model) == (ok=True,
   convex=False)` (`solver.py:3565/3597`). On hda (722 vars) the convexity
   classifier **exhausts its time budget** → `ok=False` → the whole
   `_LIFTABLE_CALL_OUTER` machinery is dead on hda regardless of what is added to
   it. (Measured: `has_factorable_work=True`, `classify → ok=False`.)
2. **Forcing the reform on still yields no bound.** With the gate bypassed and the
   `log`-lift patched in, 3 of the 6 log rows lift (`log_fired=3`; the other 3 are
   `nvars=1` and contain a *variable-exponent* `4^(…x0)` the lift correctly leaves
   alone). The root McCormick LP is **`infeasible` on the root box** — and this is
   **pre-existing**: stock (unpatched) reform already returns `infeasible` on hda's
   root box. Separately, `MccormickLPRelaxer` declines the bound because
   `_has_unbounded_nonlinear_col` is True (262 nonlinear columns, many with `±inf`
   bounds from the 17 still-dropped rows). Result: `bound=None` either way.
3. **The no-bound flowsheet class is structurally unrelaxable by this lift.** Every
   multivariate `log(<expr>)` row in the target class has an argument interval that
   **spans zero** (`lo ≤ 0`), so `log` has no finite convex underestimator and the
   lift must abstain soundly: heatexch_gen1 8/8, heatexch_gen2 10/10, heatexch_gen3
   50/50, beuster 4/4, 4stufen 4/4 span-zero (0 liftable). This is exactly the
   obstacle #517 anticipated ("`log` needs a certified strictly-positive argument
   lower bound that interval arithmetic does not always supply").

### 5.2 Out-of-panel — no witness benefits; one regresses

Corpus scan (`~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl`) for
multivariate `log(<expr>)` rows and end-to-end solves (flag ON vs OFF):

| instance | log-lift fires? | flag-OFF | flag-ON | verdict |
|---|---|---|---|---|
| ex6_2_14 (4 var) | yes (12 args) | feasible, **bound=None** | feasible, **bound=None** (nodes 2031→507) | no bound; log args are ratios `x/(x+y)→0` (loose) and appear in `log(w)·x` products that stay nonlinear |
| ex6_2_12 (4 var) | yes (10 args) | feasible, **bound=None** | feasible, **bound=None** | same structure |
| contvar (296 var) | yes (36 args) | feasible, **bound=173 560** | **TIMEOUT** (>90 s vs 30 s) | already had a bound; lift adds 36+ aux columns → per-node LP blowup → **REGRESSION** |
| heatexch_gen1/2/3, beuster, 4stufen | no (abstain, span-zero) | — | — | structurally unrelaxable (§5.1.3) |

No instance in the corpus scan gains a finite dual bound it lacked; contvar shows
the shipped lift would *regress* an instance that already certifies-progress. The
lift mechanism itself is **sound** (unit-verified: `log(x·y)` over `x,y∈[1,3]` →
`log(w)`, `w==x·y`, `w∈[1,9]>0`, exact bilinear defining equality) — but soundness
without a bound benefit, plus a measured regression, fails the acceptance bar.

### 5.3 Verdict and what remains

**KILLED — no source change to the relaxation shipped** (shipping the `log`-lift
would be dead on the target class, useless on the witnesses, and a regression on
contvar; per Dev-Philosophy #3 / #4 and the plan §0.3 kill protocol, a lift that
cannot produce a bound is not shipped). hda's no-bound has **three independent
causes**, none of which the `log`-argument lift addresses:

1. **Gating** — convexity classification times out on 722 vars, so the reform
   (and any liftable-call extension) never runs. *This is the first-order blocker;
   a call-argument lift cannot help until the reform is reachable on large models.*
2. **Pre-existing root infeasibility** — stock reform makes hda's root McCormick LP
   `infeasible` (a valid relaxation of a feasible box must not be infeasible). A
   separate correctness concern, filed as a note on #517.
3. **Coverage breadth** — 17 of 23 omitted rows are divisions / fractional powers /
   products, not `log` arguments; and the no-bound flowsheet class's log arguments
   span zero (unrelaxable). Even a perfect `log`-lift touches at most 6/23 hda rows.

**#517 disposition:** the `log(<general expr>)` relaxation-coverage lift is
**not the lever for hda / the no-bound flowsheet class**. #517 stays open, re-scoped
to the true blockers (reform gating on large models + the reform's root-infeasibility
on hda). The `x**expr` (variable-exponent) rows are likewise out of scope: hda's
`4^(0.0001+0.333·x0)` needs the `exp(expr·log x)` signomial path with `x_lb>0`, a
separate treatment.
## 6. TD-A — deeper-tail loose-product lift (nvs05/nvs09) · **BUILD (nvs09), KILL (nvs05)**

**Track:** deeper-tail follow-on to §3 ("unchanged tail"): extend the factorable
lift to the nvs05/nvs09 loose-product structure R4 does not tag. Bound-changing
regime (§0.1). Flag `DISCOPT_LIFT_LOOSE_PRODUCTS`, **default OFF**.

### Entry experiment (diagnose → hand-lift → measure) — the kill gate

**Diagnosed structure (measurement beat the plan's characterisation):**

- **nvs09** (all-integer, x0..x9 ∈ {3..9}): the objective is a *separable* sum of
  **squared univariate logs** minus a signomial:
  `Σ_i [ log(x_i−2)² + log(10−x_i)² ] − (∏_i x_i)^0.2`. It is **not** a "log·log
  product" in the multivariate sense — each `log(·)²` is `pow(log(·), 2)`, an
  *integer power of a univariate call*. `distribute_products` expands it to the
  `log·log` product the MILP linearizer cannot decompose ("Cannot decompose
  product: log·log"), so the whole objective is **dropped** → feasibility
  objective, root bound from the fallback NLP/αBB path only (**69.0 %** root gap).
- **nvs05** (welded-beam): a rational + sqrt signomial. Its objective
  `1.10471·x0²·x1 + 0.04811·x2·x3·(14+x1)` is a monomial + trilinear the RLT hull
  already relaxes; all constraints are already lifted (free auxes x4..x7 hold the
  ratios/sqrts). No dropped/loose product remains in the objective.

**Hand-lift + measurement (root bound; oracle nvs05 5.471, nvs09 −43.134):**

| instance | native root bound | hand-lifted root bound | root gap native → lifted | rel. gap reduction | verdict |
|---|---:|---:|---:|---:|---|
| **nvs09** | −72.90 (69.0 %) | **−54.83 (27.1 %)** | 69.0 % → 27.1 % | **60.7 %** | **PASS** (≥ 25 %) |
| **nvs05** | 0.674 (87.7 %) | 0.674 (unchanged) | — | **0 %** | **KILL** (criterion #2) |

- **nvs09 hand-lift:** introduce `u_i == log(x_i−2)`, `v_i == log(10−x_i)` as
  bounded auxes (`u,v ∈ [0, log 7]` on x ∈ [3,9] — strictly positive log args, no
  sign change, so the nvs09 kill clause does *not* fire) and rewrite each squared
  log as the exact univariate square `u²`/`v²`. Root bound −72.90 → −54.83, a
  **60.7 % relative** gap reduction — well over the 25 % bar. Sound: −54.83 ≤
  oracle −43.13 (no crossing).
- **nvs05 KILL (criterion #2 — "already exactly relaxed; gap is elsewhere"):**
  the *objective-only* relaxation over the native box equals **0.674 and is
  certified optimal** — i.e. `min x0²·x1 + …` over the box is exactly 0.674,
  achieved at the corner x0=x1=0.01. The objective envelope is not loose; the
  87.7 % root gap is entirely that the **constraints do not push x0,x1,x2,x3 up
  from the box corner at the root** (the true optimum is x0=0.68, x1=2.79). That
  is a **bound-tightening / branch-and-reduce (OBBT)** problem — exactly R1's
  "nvs05 = 32.5 % reduction-responsive" finding — **not** a lifting problem. No
  product-factor lift moves it. Recorded and STOPPED for nvs05.

### The general structure (not instance-keyed)

The lift keys on **`g(x)**n`** where `g` is a *single-argument transcendental*
(`log`/`log2`/`log10`/`exp`/`sqrt`/`sin`/`cos`/`tan`/`atan`/`tanh`/`log1p`) whose
base is **not a bare variable** and `n` is a **positive integer ≥ 2**. It lifts
`t == g(x)` (a bounded aux via the existing `_Lifter.expression`) and rewrites the
node as the monomial `t**n` — relaxed exactly (even `n`, `relax_square`) or
3-regime (odd `n`) by the existing monomial path. **Reuses the existing hull
machinery; rebuilds nothing** (relaxation-catalog do-not-rebuild rule honoured).

**Out-of-panel witnesses (corpus scan, 1226 `.nl` ≤ 150 KB, structurally
distinct):** the same `pow(call, int≥2)` structure appears in **mathopt5_1**
(`sin(x)**3`), **mathopt5_6** (`sin(x)**2`), **mathopt3** (`sin(x)**2`) — the
lift fires on all three (`t == sin(x)`), confirming it is not log-specific. (Only
nvs09 shows a *large* bound move; mathopt5_6 has an additional unrelated
`sqrt(abs(x))` block that still drops its bound — that is a separate coverage gap,
not a lift failure.)

### Implementation (flagged, default OFF)

`python/discopt/_jax/factorable_reform.py`:
- `_lift_loose_products_enabled()` — env flag `DISCOPT_LIFT_LOOSE_PRODUCTS`
  (default `"0"`).
- `_liftable_call_power_base()` — the structural matcher (`call ** int≥2`,
  non-atomic base).
- `_prelift_call_powers()` — a pre-pass (runs *before* `distribute_products` so
  the `**` node is intact) that lifts `g(x)**n → t**n`; a flag-gated identity
  no-op when OFF.
- `_scan_for_liftable_call_power()` + one gated line in
  `_has_factorable_work_inner()` so the reform fires on instances whose *only*
  liftable term is this structure.
- Wired as the first step of both the constraint and objective reform paths.

### Gates (BOUND-CHANGING, §0.1) — all green

- **Differential bound (nvs09):** flag ON root bound (−54.83) ≥ OFF (−72.90) AND
  ≤ oracle (−43.13); a ≥ 25 % relative reduction asserted (60.7 %).
  `test_tda_nvs09_root_bound_tightens`.
- **Feasible-point sampling (2000 pts):** the lift `t == g(x)` is an exact
  identity — the lifted objective at `(x, t=g(x))` equals the original at `x`
  (max err < 1e-6) and every aux equality holds (residual < 1e-6); cuts no
  feasible point. `test_tda_lift_is_exact_identity_feasible_sampling`.
- **Flag-OFF byte-identical:** `check_cert_neutrality.py` **NEUTRAL** — all 41
  cert-panel instances `nodes X→X`, `|Δobj| = 0` (the new code is an identity
  no-op when the flag is off).
- **Suite:** `pytest -m smoke` 614 passed / 14 skipped; adversarial slow 10
  passed; `test_factorable_reform.py` 65 (+5 TD-A) passed; ruff check + format
  clean; mypy clean (numpy-stub env mismatch excepted). No Rust touched.

### Acceptance

**nvs09 root gap 69.0 % → 27.1 % (60.7 % relative reduction) with the flag ON —
meets the ≥ 25 % bar.** At 60 s nvs09 is still `feasible` (per-node cost of the
`(∏x_i)^0.2` 10-way signomial is the remaining limiter, 7 nodes) — the win is a
*materially tighter dual bound*, not (yet) certification, which the acceptance
allows ("a materially tighter bound is the bar; certifying is a bonus"). **nvs05
is honestly KILLED** — its gap is OBBT/branch-and-reduce (R2 territory), not a
lifting problem. Flag stays OFF pending 3 green nightlies per §0.1.

**Ledger addition:** **TD-A** (loose-product / call-power lift) — **BUILD
(nvs09, flagged default-OFF) / KILL (nvs05)**; nvs09 root gap 69.0 % → 27.1 %;
witnesses mathopt5_1/5_6/mathopt3; PR — see title
`perf(relaxation): TD-A — lift loose-product factors (nvs05/nvs09, flagged)`.
