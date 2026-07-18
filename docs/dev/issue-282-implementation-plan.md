# Issue #282 — implementation plan (root/dual-bound strengthening for `syn*`/`rsyn*`)

**Target issue:** [#282](https://github.com/jkitchin/discopt/issues/282) — Global-search gap:
`syn*`/`rsyn*` process-synthesis MINLP (sound suboptimal incumbents).

**Audience:** the implementer (Opus). This is an execution plan, not a diagnosis — the
diagnosis is done. Read it top to bottom; every workstream states its entry experiment,
kill criterion, soundness regime, and graduation gate. **Do not write implementation code for
a workstream before running its entry experiment (CLAUDE.md §4).**

**Prior art you must not re-walk:**
- `docs/dev/issue-282-syn-rsyn-diagnosis-2026-07-17.md` — the three-round diagnosis. Read
  §R2 and §R3 first.
- `docs/dev/scip-gap-closing-plan.md` §0 — the live-issue priority table and the §1.5
  node-reduction gate.
- `docs/dev/performance-plan.md` §10 — the #727 RLT productivity lesson (**sound ≠ helpful**),
  binding.
- `docs/design/relaxation-catalog.md` §4–§7 — what relaxations/cuts already exist and the
  soundness rules.
- `docs/dev/certification-gap-plan.md` §0 — the binding correctness contract.

---

## 0. Framing: what this issue actually is, and what is already done

Three rounds of measurement (all on the mounted MINLPLib snapshot) established:

1. **The gap is dual-dominated on 7/7 instances.** With each instance carrying a proven
   `=opt=` tag, the split `reported_gap = (bound − opt) [dual excess] + (opt − obj) [primal
   deficit]` is exact. Every instance is dual-dominated at 60 s; on `syn15m02hfsg` and
   `syn30hfsg` the incumbent **is already the proven optimum** (primal deficit 0.0 %) — discopt
   has the answer and cannot prove it. **Soundness clean on all 14 runs.** So the issue's
   original "sound suboptimal incumbents" framing is true but its implied lever (better
   incumbents / LNS #276) is the **wrong half**. This is a **root-relaxation quality** problem.

2. **discopt's root bound is 4×–500× looser than SCIP's** on all seven (SCIP proves the whole
   panel optimal in 0.5–1.6 s). The tree cannot recover what the root gives away.

3. **The family splits by convexity → by solve path:**
   - **Nonconvex `*hfsg`** (`syn15m02hfsg`, `syn30hfsg`, `syn40hfsg`) → **spatial McCormick
     B&B**. Root cause of their loose bound was a **discarded relaxer** (fixed, flag-gated —
     Workstream A below).
   - **Convex** (`rsyn0805m`, `rsyn0810m`, `rsyn0815m`, `syn40m`) → **NLP-BB** (`_solve_nlp_bb`),
     which has **no root cut/OBBT stage at all**. Their root bound is the convex continuous
     relaxation with a weak big-M; box tightening (OBBT) moves it **0.0 pts** because it does not
     cut the fractional-`y` relaxation point. Closing this needs **root cutting planes** — the
     open campaign, Workstream C below.

### Already shipped (merged, both default-OFF, neither graduated)

| lever | PR | flag | scope | status |
|---|---|---|---|---|
| tightened-box root LP probe | [#715](https://github.com/jkitchin/discopt/pull/715) | `DISCOPT_ROOT_LP_PROBE_TIGHT` | nonconvex `*hfsg` | merged, default-OFF |
| iterate root OBBT to convergence | [#720](https://github.com/jkitchin/discopt/pull/720) | `DISCOPT_OBBT_ITERATE` | wide-box dense QCQP | merged, default-OFF |

### Already resolved, do not re-open

- **Certification (#703, closed not-a-defect).** `gap_certified=False` on the convex half is
  **correct**: the node NLPs are 100 % trusted (0 untrusted nodes, all clean-KKT `OPTIMAL`);
  the `False` comes from the feasible-exit reset (`solver.py` ~10919–10921) because the gap is
  still open at the time limit. `gap_certified` means *optimality proven*, not *bound valid*
  (the rigorous dual `bound` is surfaced and `obj < opt < bound` holds soundly). It flips to
  `True` **only** when the root/dual bound closes the gap — i.e. Workstream C is also the fix
  for #703. **Do not weaken the guard.**
- **Improver wall-time contingent (H2, #704, closed).** Entry experiment CONFIRMED (improvers
  burn 65–69 % of budget on rsyn; the contingent is denominated in abstract units, not seconds),
  design-only. This is a **throughput/primal** lever, orthogonal to the dual-bound work here —
  out of scope for #282. If pursued it is its own issue (build not yet done).

### Do-not-re-walk (falsified levers — measurement already killed these)

- **OA auto-routing for detected-convex MINLPs (F-1):** KILLED (86 major iters at 20 binaries,
  wrong status at 30). Do not route convex MINLPs to `solve_oa`.
- **Root OBBT / range reduction for the nonconvex `*hfsg` bound:** moves the reported
  `root_bound` ~0 pts (955.4 % → 955.4 %). Range reduction is not the lever there — the relaxer
  discard was (Workstream A).
- **Root OBBT for the convex half:** moves the bound **0.0 pts** (box tightening does not cut
  the fractional-`y` point). Cutting planes, not tightening, are the lever (Workstream C).
- **Primal/LNS incumbent quality (#276) as the #282 headline:** the incumbent is the healthy
  side (already optimal on 2/7). Not the lever.

---

## 0.1 Environment preconditions (READ BEFORE STARTING)

**The seven-instance panel requires the MINLPLib snapshot.** Only `syn05hfsg.nl` is vendored in
`python/tests/data/minlplib_nl/`; **`rsyn0805m/0810m/0815m`, `syn15m02hfsg`, `syn30hfsg`,
`syn40hfsg`, `syn40m` are NOT vendored.** The plan's entry experiments and the graduation panel
read them from the snapshot at:

```
~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/*.nl   # the .nl instances
~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu      # the =opt= oracle
```

Confirm before running any panel step:

```bash
ls ~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/rsyn0805m.nl \
   ~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu
```

If that path is absent, the entry experiments below **cannot be run** — stop and mount the
snapshot (or point `$DISCOPT_MINLPLIB_CACHE` at a populated cache). Do **not** substitute a
synthetic proxy: the #727 RLT lesson (synthetic root-gain 0.68, real gain 0.0) is the standing
warning that a mechanism validated only on a proxy can be a no-op on the real class.

Standard env for every run: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=python`.

Existing harness (reuse, do not rebuild):
- `discopt_benchmarks/scripts/issue282_gap_attribution.py` — per-instance dual-excess /
  primal-deficit / throughput attribution at chosen budgets. Reads `=opt=` directly.
- `discopt_benchmarks/results/issue282/` — round-1..3 raw data
  (`attribution_5s_60s_20260717.json`, `scip_reference_20260717.json`,
  `root_lp_probe_ab_20260717T183044.json`, `heuristic_ab_60s_20260717.json`).
- `discopt_benchmarks/scripts/gp_minlp_graduation_panel.py` — the canonical **two-bar**
  (cert-clean + net-positive) graduation-panel template; copy its affected/inert split.
- `discopt_benchmarks/scripts/graduation_gate.py` — the general per-flag gate + durable ledger.
- `discopt_benchmarks/utils/soundness.py` — `assert_bound_sound` / `assert_cut_valid`
  (feasible-point-not-removed). **Every new cut family must pass `assert_cut_valid`.**

---

## Workstream A — graduate `DISCOPT_ROOT_LP_PROBE_TIGHT` (nonconvex `*hfsg`)

**Mechanism is already merged (#715); this workstream is the graduation, not a build.** The
keep/discard probe for the McCormick LP relaxer now runs over the FBBT/OBBT-tightened root box
instead of the raw declared bounds (the `*hfsg` family declares flows `[0, inf]`, so the raw-box
probe made the McCormick LP unbounded → `_mc_mode="none"` → relaxer dropped → loose fallback
bound). Sound by construction: the probe only decides *whether* to keep a rigorous
outer-approximation relaxer; every node still solves its own sub-box.

Measured effect (30 s, root dual excess vs `=opt=`):

| instance | root % OFF→ON | dual % OFF→ON |
|---|---|---|
| `syn30hfsg` | +955.4 → **+571.1** | +919.2 → **+542.4** |
| `syn40hfsg` | +3041.4 → **+2350.4** | +2667.1 → **+2260.0** |
| `syn15m02hfsg` | +124.7 → **+118.1** | +123.0 → **+115.2** |
| convex `rsyn*`/`syn40m` | unchanged (bound-neutral) | unchanged |

### Steps

1. **Re-confirm the A/B on the real panel** (corpus required). Reproduce
   `results/issue282/root_lp_probe_ab_20260717T183044.json` on a quiet box; the value-based
   root/dual numbers above are load-independent, so they must reproduce. If any `*hfsg` root
   fails to tighten, stop — the merged mechanism regressed.
2. **Run the Regime-2 graduation panel** (CLAUDE.md §5, single passing run meeting BOTH bars):
   - *cert-clean:* flag ON vs OFF over the vendored 65-instance corpus + the seven-instance
     `*hfsg`/convex panel — `incorrect_count = 0`, no dual bound above `=opt=`, no
     `gap_certified=True → False` regression, certified objective within tolerance (abs 1e-6 /
     rel 1e-4), every ON incumbent independently feasibility-verified. #715 already reported
     0/65 soundness violations; re-run under the panel harness and record the verdict JSON.
   - *net-positive:* measurably helpful broadly — the `*hfsg` root/dual bound must tighten
     (it does, above) and no covered class may regress. The convex half is bound-neutral
     (structurally OK). Score against the `casctanks` change too (it tightens — confirms the
     mechanism is general, not `*hfsg`-keyed, per CLAUDE.md §2).
   - Clone `gp_minlp_graduation_panel.py` into
     `discopt_benchmarks/scripts/issue282_root_lp_probe_graduation_panel.py`; the affected set =
     spatial-McCormick instances where the probe verdict changes; the inert set = everything
     `_mc_mode` doesn't touch (must be byte-identical status/obj/node_count).
3. **Kill criterion:** if the panel shows the flip is cert-clean but **not** net-positive (e.g.
   the tighter root does not reduce nodes / wall on any covered instance — the `DISCOPT_CUT_INHERIT`
   / #727 failure mode), it **stays default-OFF** and you record the neutral verdict. Soundness
   alone does not graduate a flag.
4. **On a passing panel:** graduate the default the way `DISCOPT_CUT_INHERIT` did — route the
   flag through a `SolverTuning` tri-state field (`python/discopt/solver_tuning.py`;
   `True`=force-on / `False`=force-off-shipped-default / `None`=structure-gated opt-in), flip its
   `default_factory` to the graduated value, and keep `_root_lp_probe_tight_enabled`
   (`solver.py:446–470`) as the env opt-out. Land the verdict JSON under
   `discopt_benchmarks/results/issue282/` and update the flag docstring (`solver.py:448–464`).
   **Graduation-gate caution (the CUT_INHERIT precedent, `docs/dev/cut-inherit-grad-2026-07-08.md`):
   a single flag-ON false-optimal (nvs22) blocked CUT_INHERIT from graduating** — the bar is zero
   soundness regressions AND no covered instance materially slower. Do not flip on a
   cert-clean-but-neutral result.

**Regression test already exists:** `python/tests/test_issue282_root_lp_probe.py` (vendored
`syn05hfsg`; differential-bound + feasible-point; fails before the fix). Keep it.

---

## Workstream B — graduate `DISCOPT_OBBT_ITERATE` (wide-box QCQP root box)

**Mechanism merged (#720), default-OFF.** Raises the root OBBT sweep cap 3→50 with a
min-improvement early-stop when the model is quadratically structured AND the box is wide. The
OFF path is byte-identical (`rounds=3`, `min_improvement=None`). Each tightening reuses the
existing Neumaier–Shcherbina safe clamp — soundness unchanged; iterating only applies more sound
tightenings.

**Scope caveat:** #720's headline instance is `nvs24` (dense integer QCQP, #283 family), not the
`syn*`/`rsyn*` panel — the `syn/rsyn` family is `0 bilinear`, so this lever is expected
**bound-neutral on the #282 panel itself** and its real corpus is the wide-box dense-QCQP class.
It is #282-tagged only as a sibling root-bound lever. Treat it as an independent graduation.

### Steps

1. **Entry experiment (corpus required):** confirm the ON-vs-OFF root-bound A/B on wide-box
   QCQP (nvs24: root −394,504 OFF → −14,443 ON, 27× tighter) and confirm **byte-neutrality on
   the `syn*`/`rsyn*` panel** (it must not change those — they have no quadratic structure).
2. **Regime-2 graduation panel:** flag ON vs OFF over the vendored corpus + the quadratic
   n≤40 slice (#720 already ran 35 instances, 0 violations, neutral there because they already
   solve OFF). The class that benefits (wide-box non-converging QCQP, e.g. nvs24) is **not** in
   the n≤40 corpus, so the panel must draw the wide-box QCQP instances from the snapshot to
   demonstrate net-positive. Clone the panel template as
   `discopt_benchmarks/scripts/issue282_obbt_iterate_graduation_panel.py`.
3. **Kill criterion:** if end-to-end the reported dual bound does not improve on the wide-box
   class (because the node-engine's alphaBB path — governed by Workstream A's flag — makes the
   reported bound time-limit-dependent, per #720's "why default-OFF"), the two flags may need to
   graduate **together**. Record the interaction; do not flip B in isolation if A being OFF masks
   its benefit.
4. **On a passing panel:** flip the default at `solver.py:438`, keep the `=0` opt-out and the
   `min_improvement=None` legacy path.

**Regression test already exists:** `python/tests/test_obbt_iterate_root.py`. Keep it.

---

## Workstream C — root cutting planes on the NLP-BB path (the open campaign)

**This is the substantive new work.** The convex `rsyn0805m/0810m/0815m` and `syn40m` route to
`_solve_nlp_bb` (`solver.py:10788`). Its GDP structure is materialized as a **big-M**
linearization by `reformulate_gdp(model, method="big-m")` (`solver.py:10823`), and its only root
stage is `tighten_root_bounds_with_fbbt` (`:10859–10870`) — **FBBT + integer-bound rounding only:
no OBBT, no LP relaxation, no cut/OA stage.** Each node then bounds by solving the **original
continuous NLP over the node's box** (`:11156–11298`) — a *full NLP, not an LP*. The big-M gives a
**fractional-`y`** root relaxation that box tightening cannot cut, so the root sits at
+63 %/+72 %/+104 %/+2609 % while SCIP — using MIR / flow-cover / knapsack-cover / OA cuts on the
big-M rows — reaches +16 %/+10 %/+18 %/+5 %.

**The architectural gap, precisely.** Because NLP-BB bounds by NLP (not LP), you cannot simply
"add cut rows to the node LP" the way the spatial McCormick path does — there is no node LP. Two
consumer paths exist, and Stage C.1 must pick one:

- **(a) Augmented node NLP (preferred first cut).** For a **convex** MINLP, an OA cut generated at
  the root NLP solution is a **globally-valid supporting hyperplane** — valid at every node. Add
  such cuts as extra **linear constraints** to the node NLP via the existing `_AugmentedEvaluator`
  (`solver.py:717`), which already wraps an evaluator with a `CutPool`'s `to_constraint_arrays()`.
  This reuses `generate_oa_cuts_from_evaluator` / `generate_objective_oa_cut`
  (`_jax/cutting_planes.py:271`,`:613`) against the `NLPEvaluator` that `_solve_nlp_bb` already
  builds (`_make_evaluator`, `:10902`). **Lowest-risk lever: no new LP, no relaxer, cuts are
  globally valid by convexity.**
- **(b) Root LP relaxation.** Build a `MccormickLPRelaxer` (`_jax/mccormick_lp.py:353`) or a big-M
  MILP root LP, separate a root cut pool once (MIR/cover/GMI via `milp_driver.rs` /
  `_root_cover_cut_loop` at `solver.py:14570`, or `separate_cmir`), and adopt the tighter root
  dual bound as `_root_pool_bound` (the adoption path the spatial route already uses at
  `solver.py:6838`). Heavier; only if (a) leaves a large residual on `rsyn*`/`syn40m`.

This is a campaign, not a bounded increment. Stage it, and gate every stage on an entry
experiment scored **purely on root bound vs `=opt=`** before any wiring into the search.

### What already exists to build on (do not rebuild)

- **OA cut generators + `CutPool`:** `python/discopt/_jax/cutting_planes.py` — `CutPool` (`:893`),
  `generate_oa_cuts_from_evaluator` (`:271`), `generate_objective_oa_cut` (`:613`),
  `generate_alphabb_quadratic_oa_cuts_from_evaluator` (`:535`), `LinearCut` (`:36`). The multitree
  `solve_oa` (`python/discopt/solvers/oa.py`, `_add_oa_cuts:2145`, `_add_ecp_cuts:2278`) drives
  exactly this master→cut→re-solve loop already. **Reuse the generators; do not route the solve to
  OA** — F-1 killed auto-routing.
- **`_AugmentedEvaluator`** (`solver.py:717`) — wraps an `NLPEvaluator` with a `CutPool`'s
  `to_constraint_arrays()`, i.e. the ready mechanism to feed cuts into a node NLP as linear rows
  (consumer path (a) above).
- **MIR / GMI / cover (Rust LP path):** `crates/discopt-core/src/lp/{mir,gomory,aggregation,
  cover,cut_select}.rs` + Python `_root_cover_cut_loop` (`solver.py:14570`, separates Gomory
  `:14680`, single-row MIR `:14688`, aggregation c-MIR `:14704` gated `DISCOPT_CMIR_AGGREGATION`,
  cover/clique `:14717`) and `separate_cmir` (`_jax/cmir_cuts.py:95`). All wired to the
  **MILP/spatial** cutting LP, **not** `_solve_nlp_bb`. **C-4** (fixed) is the binding soundness
  lesson: an integer-MIR cut with a fractional integer lower bound is invalid — reuse the fixed
  path, do not re-derive. Relevant only under consumer path (b).
- **Knapsack cover cuts:** `python/discopt/_jax/cover_cuts.py`, Rust `lp/cover.rs`.
- **Flow-cover cuts: DO NOT EXIST anywhere in the tree.** The one genuinely new cut family the
  campaign may need. If added, it must pass `assert_cut_valid` and carry its own differential test.

### Entry experiment C.0 (run FIRST — corpus required; no code beyond a throwaway probe)

For each convex panel instance, at a 1-node root limit, measure the **standalone** root-bound
gain of each cut family, in isolation:

1. Take the root NLP solution (`_solve_root_node_multistart`, `solver.py:11219`) and the big-M/OA
   linearization (`reformulate_gdp(big-m)` output; the `solve_oa` first-iteration master is a
   ready source of the linear relaxation).
2. Separate one round of each of {**root OA cuts** via `generate_oa_cuts_from_evaluator` +
   `generate_objective_oa_cut` (consumer path (a)), MIR (via `mir.rs`), knapsack-cover
   (`cover_cuts.py`), flow-cover (prototype)} against the fractional root point; re-bound; record
   the new root bound.
3. Report per-family **relative root gain** = `(bound_with_cuts − bound_root) / (opt −
   bound_root)` — the same metric the #727 probe used to split pooling (0.68) from heatexch
   (1e-11). OA cuts are the cheapest and are globally valid for these convex models, so measure
   them first; only reach for the LP-relaxation families (MIR/cover/flow-cover, consumer path (b))
   if OA leaves a large residual.

**Kill criterion (per family, binding — CLAUDE.md §4 + #727 lesson):** a family is dead for
#282 unless it moves the root bound **≥ 10 % of the discopt→SCIP root spread** on at least one
convex instance. A cut that is valid but closes ~0 of the root only starves branching
(**sound ≠ helpful**, the #727 / `DISCOPT_CUT_INHERIT` law) — record it as falsified and drop it.
Rank the surviving families by root gain; only the survivors get implemented.

**Expected split to verify, not assume:** `syn40m`'s +2609 % (fastest root, worst bound) is
likely a *different* failure from `rsyn*`'s ~+70 % (the diagnosis flagged this open question). If
one cut family fixes `rsyn*` but not `syn40m`, that is a real result — do not force one mechanism
onto both; report the split.

### Stage C.1 — root cut loop on `_solve_nlp_bb`, behind `DISCOPT_NLPBB_ROOT_CUTS` (default-OFF)

Only for the families that survived C.0. Add a root-only cutting stage to `_solve_nlp_bb`,
inserted **after the root FBBT presolve and tree creation** (`solver.py:10870–10907`) and around
the `iteration==0` root NLP solve (`:11218–11228`). `root_bound` is already wired onto
`SolveResult` by cert-plan T0.1, so you have the measurement surface. The stage:

1. Generate a root cut pool once (the surviving families from C.0) — for consumer path (a), OA
   cuts from the root NLP solution into a `CutPool`; for (b), separate against a built root LP.
2. **Consume the cuts.** Path (a): wrap the node evaluator in `_AugmentedEvaluator` (`:717`) so
   every node NLP carries the root cuts as extra linear constraints (globally valid by convexity).
   Path (b): adopt the tighter root LP bound as `_root_pool_bound` (reuse the spatial adoption at
   `:6838`) and/or inherit the pool at nodes.
3. Iterate root separation to a min-improvement early-stop (mirror `obbt_tighten_root`'s
   convergence structure).

**Soundness (the hard gate — this is bound-changing, CLAUDE.md §5 / cert-plan §0.2–§0.3):**
- Every cut must pass `assert_cut_valid` (no feasible integer point, including the global
  optimum, removed). Add a feasible-point-sampling test per family on a vendored instance.
- An LP-infeasible fathom from the cut LP prunes only on a **verified Farkas ray** (C-38 lesson)
  — never trust a bare simplex `infeasible` status to fathom.
- MIR reuses the C-4-fixed generator (no cut from a fractional integer lower bound).
- The bound may never exceed `=opt=` on any box; the differential test asserts
  `new_bound ≥ old_bound` AND `new_bound ≤ true box optimum` on fixed boxes.

**Architectural warning (cert-plan T1.3, binding):** the incremental/spatial fast path returns
its LP bound *before* the per-node separation chain, and widening a scope gate there caused a
catastrophic bound-weakening regression (`dispatch` 3→9843 nodes). Keep the NLP-BB root cut
stage **root-only** first; do **not** wire cuts into every node until a separate entry experiment
shows per-node separation helps without collapsing the bound elsewhere.

### Stage C.2 — the §1.5 node-reduction gate (binding, from `scip-gap-closing-plan.md`)

A tighter root is necessary but not sufficient. `scip-gap-closing-plan.md` §1.5 ruled out
"port the cuts and hope" by measurement: discopt's *current* cuts can make the bound **looser at
equal node count**. So Stage C.1's flag graduates only if the root cuts **reduce node count to
close the gap** on the convex panel — not merely tighten the root value. Score both root bound
and node-count-to-close; a root that tightens but does not reduce nodes is the #727 heatexch
failure mode (heavier node LP starves branching) and stays OFF.

### Stage C.3 — graduation

Full Regime-2 panel (`gp_minlp_graduation_panel.py` template), affected set = NLP-BB convex
MINLPs where cuts fire, inert set = everything else (byte-identical). BOTH bars: cert-clean
(0 violations, no bound past oracle, no cert regression, incumbents feasibility-verified) AND
net-positive (root bound tightened AND nodes-to-close reduced, no covered class regressed). Land
the verdict JSON under `discopt_benchmarks/results/issue282/`. Default-OFF until it passes.

**Honest-frontier clause (house style, cf. #673/#677/#732):** if the convex root gap resists all
surviving cut families (e.g. `syn40m`'s +2609 % proves structurally hard), record the frontier —
the residual root spread and which families moved it how far — rather than shipping a weaker
relaxation or a name-keyed special case. A recorded falsification is a valid outcome (CLAUDE.md
§4).

---

## Verification checklist (every PR in this plan)

- `pytest -m smoke` — 0 failures.
- `pytest -m slow python/tests/test_adversarial_recent_fixes.py` — 0 failures.
- `cargo test -p discopt-core` — only if Rust touched (Workstream C MIR/flow-cover may touch
  `crates/discopt-core/src/lp/simplex/`).
- `ruff check python/` + `ruff format --check python/` clean; `mypy python/discopt/`.
- New behavior ⇒ a regression test that **fails before** the change and passes after.
- Bound-changing flags ship **default-OFF**; the differential-bound + feasible-point tests are
  the gate; graduation is one passing Regime-2 panel meeting **both** bars (CLAUDE.md §5).
- `incorrect_count = 0` with zero slack throughout — never weaken a validation/fallback/guard to
  make a gate pass.

## Definition of done for #282

- Workstreams A and B graduated (or a recorded neutral verdict keeping them OFF).
- Workstream C: the convex-half root cutting-plane stage lands (default-OFF → graduated if the
  panel passes), materially closing the discopt→SCIP root spread on `rsyn*` and (separately)
  `syn40m`, with node-count-to-close reduced — OR a recorded honest frontier if a subfamily
  resists. When the root bound closes the convex-half gap, `gap_certified` legitimately flips
  `True` (this is also the #703 resolution).
- Closing summary on #282 states what graduated, the measured root/dual/node numbers, and
  whether the issue can close or exactly what remains (per CLAUDE.md "Working on an issue" §5).

## File / symbol map (quick reference)

Line numbers are current as of the diagnosis reading (`solver.py` is ~16320 lines; re-grep before
editing — earlier diagnosis comments cite a stale layout, e.g. "`:10027`"/"`:9844`" are the
`solve_model` twins, not the NLP-BB sites).

| what | where |
|---|---|
| `DISCOPT_OBBT_ITERATE` reader | `python/discopt/solver.py:438` |
| `DISCOPT_ROOT_LP_PROBE_TIGHT` reader + docstring | `python/discopt/solver.py:446–470` (bool `:465`) |
| `DISCOPT_ROOT_LP_PROBE_TIGHT` probe site (spatial only) | `python/discopt/solver.py:6700–6767` |
| `_solve_nlp_bb` (def) | `python/discopt/solver.py:10788` |
| big-M GDP reform (materializes structure) | `python/discopt/solver.py:10823` (`reformulate_gdp(method="big-m")`) |
| NLP-BB root stage — FBBT only, **insert C.1 cut loop after here** | `python/discopt/solver.py:10859–10907`, root NLP `:11218–11228` |
| NLP-BB node bounding (full NLP over box, not LP) | `python/discopt/solver.py:11156–11298` |
| NLP-BB `_improver_allowed` (H2 / #704 site) | `python/discopt/solver.py:10990–11001` |
| NLP-BB certification + feasible-exit reset (#703 site) | `python/discopt/solver.py:11814–11943` (reset `:11886–11888`) |
| batch/serial trust decert (do not weaken) | `python/discopt/solver.py:11173` (batch), `:11256–11267` (serial), `_batch_trusted` |
| `_AugmentedEvaluator` (feed cuts into a node NLP) | `python/discopt/solver.py:717` |
| spatial root cut-pool capture + adoption | `python/discopt/solver.py:6790–6935`, `_root_pool_bound` `:6838` |
| Python MILP root cut loop (Gomory/MIR/c-MIR/cover) | `python/discopt/solver.py:14570` (`_root_cover_cut_loop`) |
| root OBBT to convergence | `python/discopt/_jax/obbt.py` (`obbt_tighten_root`) |
| OA cut generators + `CutPool` | `python/discopt/_jax/cutting_planes.py` (`CutPool:893`, `generate_oa_cuts_from_evaluator:271`, `generate_objective_oa_cut:613`) |
| OA multitree driver (reuse generators, do not auto-route) | `python/discopt/solvers/oa.py` (`_add_oa_cuts:2145`, `_add_ecp_cuts:2278`) |
| McCormick LP relaxer + incremental rows | `python/discopt/_jax/mccormick_lp.py:353` (ctor), `solve_at_node:900` |
| c-MIR / aggregation separator | `python/discopt/_jax/cmir_cuts.py:95` |
| MIR / GMI / cover / cut-select (Rust LP path) | `crates/discopt-core/src/lp/{mir,gomory,cover,aggregation,cut_select}.rs`, `bnb/milp_driver.rs` |
| knapsack cover cuts (Python) | `python/discopt/_jax/cover_cuts.py` |
| cut-validity harness (mandatory) | `discopt_benchmarks/utils/soundness.py` (`assert_cut_valid`) |
| graduation flag tri-state | `python/discopt/solver_tuning.py` (`SolverTuning`, cf. `DISCOPT_CUT_INHERIT`) |
| graduation-panel template | `discopt_benchmarks/scripts/gp_minlp_graduation_panel.py` |
| general per-flag gate + ledger | `discopt_benchmarks/scripts/graduation_gate.py` |
| gap attribution harness | `discopt_benchmarks/scripts/issue282_gap_attribution.py` |
| round-1..3 raw data | `discopt_benchmarks/results/issue282/` |
| NLP-BB = "no cut separation" (design context) | `docs/design/global-cut-pool.md` §2a |
