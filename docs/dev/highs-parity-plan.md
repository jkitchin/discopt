# HiGHS parity plan — the in-house Rust engine in every role HiGHS plays

Status: **proposed, nothing implemented.** Written 2026-11-03 against `main` @ `1fe78b1`;
every `file:line` below was read at that commit. Branch: `claude/happy-hamilton-0vvo3b` (long-running,
§2). Owner priority (2026-11-03): an early, explicit path to closing **#1671** (§4, M-1671).

Goal: the Rust engine (`crates/discopt-core/src/lp/simplex/`, `bnb/milp_driver.rs`) at
parity with or better than HiGHS in **every role HiGHS plays in discopt**, so `highspy`
becomes an optional extra and the HiGHS-only code can be deleted. "Parity" is made
quantitative per role in §1.3. This plan does not re-run the MILP levers
`milp-competitiveness-plan.md` already killed (§1.5).

---

## §0 Loop protocol (binding — this doc is loop-executable)

Execute §5's checklist strictly in order, one task per loop iteration (a task marked
*(≤ k iterations)* may take that many; nothing else may).

1. Pick the **first unchecked** `[ ]` task in §5. Do only that task. Read its **Scope**,
   **Entry**, **Done**, **Kill** and **Regime** before touching code.
2. **Entry experiment first** (CLAUDE.md §4). A task whose Entry names a hypothesis runs
   the falsifying measurement on the real instances it names *before* the implementation.
   If the measurement contradicts the task, write the falsification in §7 and re-scope
   the task (or retire it) **in the same iteration**; do not build on it.
3. Implement. Every new solver-math lever goes behind a `DISCOPT_*` flag, default-OFF,
   with a row added to `docs/dev/flag-retirement-audit.md` in the same commit (CLAUDE.md
   §5). A bound-neutral port needs no flag but needs its exact-identity gate (§1.4).
4. **Verify before committing** — standing gates, all of them, every task:
   - `maturin develop --release` from the repo root when `crates/` changed, then
     `cargo test -p discopt-core`;
   - `pytest -m smoke`; `pytest -m slow python/tests/test_adversarial_recent_fixes.py`;
   - `ruff check python/ && ruff format --check python/ && mypy python/discopt/`;
   - a new regression test that **fails before and passes after** (prove it: run it on
     the pre-change tree or a mutated module with the load marker asserted absent);
   - the task's own **Done** criterion, run exactly as written, with its
     executed-comparison count printed and non-zero;
   - a new test's call phase is < 15 s or it is marked `slow` (+ `correctness` when it
     checks a known optimum) and added to the right allowlist in the same PR
     (CLAUDE.md "Tests").
   If any gate is red, fix it in the same iteration. **Never advance on red.**
5. Commit `feat(highs-parity): Pn.m <title>` (or `fix:`/`test:`/`docs:`), flip the box
   `[x]` in §5 **in the same commit**, and record the one-line result (numbers, not
   adjectives) under the task. Update the §6 status table.
6. **Kill criterion.** Every task names the measurement that falsifies it. When it
   fires: stop, write the falsification in §7 (date, commit, numbers, what it kills),
   apply the task's stated *on-kill* action (usually: retire the flag **and its
   implementation** in one commit, CLAUDE.md §5 "Retired"), and re-scope §5 before
   continuing. The measurement wins.
7. **Decision points (D1–D5, §3)** are tasks that may end in "stop and ask the owner".
   On that outcome the loop halts: write the decision memo under the task and do not
   pick the next task until the owner answers.
8. Soundness is a hard gate with zero slack: `incorrect_count = 0` on every panel, no
   bound above a reference optimum, no `gap_certified=True` instance regressing to
   uncertified. A task that can only meet its Done by weakening a validation, tolerance,
   verifier or fallback has failed; say so and stop (CLAUDE.md §1, §3).

---

## §1 Where HiGHS is, and what parity means

### 1.1 The roles, verified in the tree (corrections to the brief are flagged ⚠)

| Role | Path | Default? | Entry points (file:line) |
|---|---|---|---|
| **R1** pure LP/MILP at `Model.solve()` | `_lp_milp_backend()` reads `DISCOPT_LP_MILP_BACKEND`, default `highs` (`solver.py:37494`); `_highs_takes_pure_lp_milp` (`:37532`) skips root presolve for the route; LP `_solve_lp_highs` (`:37822`) → `lp_milp_highs.solve_lp_std` (`lp_milp_highs.py:1657`); MILP → `solve_milp_std` (`:2583`) / `_solve_milp_std` (`:3494`). Opt-out `=rust` → `_solve_milp_simplex` (`solver.py:38295`) → `solve_milp_csc_py` (`lp_bindings.rs:1271`) → `milp_driver.rs` (9,714 lines). `milp_backend="native"` (#1535) is the Python B&B `_solve_milp_bb` (`solver.py:39006`). | **yes** | contract: `lp-milp-highs-routing-plan.md` §3 |
| **R2a** ⚠ convex-MINLP route OA/LP-NLP-BB master | `_convex_route_oa_master()` is **always HiGHS** (`solver.py:9916`; the `auto` Rust arm was *retired* 2026-10-04, setting it raises). Route is on by default (`DISCOPT_CONVEX_MINLP_ROUTE`, `solver.py:9910`), target `lp_nlp_bb` since #1658 → `milp_highs.solve_milp_with_lazy_cuts` (`milp_highs.py:1418`), whose certifying verdict is **already confirmed by the Rust driver** `_confirm_lazy_master_independently` (`milp_highs.py:1260`, #2069). | **yes** | the brief's "Rust is already the default" is true only of R2c |
| **R2b** ⚠ decomposition masters | `get_decomposition_master_solver()` returns HiGHS whenever `highspy` imports (`lp_backend.py:249-262`, #1614); callers `decomposition/benders/solver.py:238`, `benders/gbd.py:312`, `lagrangian/solver.py:147`. | **yes** (when installed) | Rust is the fallback only |
| **R2c** GDP native master, AMP masters | `DEFAULT_MASTER_BACKEND = "auto"` = Rust (`gdp/algorithms/master.py:40`); AMP `milp_solver="auto"` (`amp.py:974`). HiGHS is per-call opt-in. | no | G3.2 measurement at `master.py:26-39` |
| **R3** NLP-BB root cutting planes | `solvers/_root_cuts.py` (`DISCOPT_NLPBB_ROOT_CUTS`, **default ON** since 2026-08-20, header `:46-52`): root LP via HiGHS `_solve_lp` (`:410`), GMI from `getBasicVariables`/`getReducedRow`/`getBasisInverseRow` (`:532-582`). A missing `highspy` degrades to "no cuts" (`:714`, #1514). | **yes** | Rust has `gomory.rs::separate_gomory_cols` and the binding `gomory_cuts_py` (`lp_bindings.rs:209`) |
| **R4** convex node bound dual source | `convex_node_bound._lp_dual` (`:215`): HiGHS on the slack form → row-equilibrated HiGHS (`:132`) → Rust simplex `_simplex_dual` (`:170`). The certificate `certify_lp_dual` (`lp_milp_highs.py:718`) is valid for any dual. | **yes** | #2067 |
| **R5** MILP IIS engine | `_iis_milp._Engine` = one persistent HiGHS model, `changeRowsBounds`/`changeColsBounds`/`changeColsIntegrality` (`:236-242`), presolve-on/off probes (`:244-271`, `:400-416`), #1634 presolve-free confirmation. Reached only by `compute_iis(include_integrality=True)` (`infeasibility.py:253-258`, default `False`; dispatch `:487`). | **opt-in** ⚠ | Rust has no persistent modifiable MILP handle |
| **R6** ⚠ verified-LP uses on the **MINLP** path | `solve_lp_std` is also the arbiter for POUNCE `primal_infeasible` labels (#1537, `_linear_node_verified_empty` `solver.py:34903`, guarded by `_highs_lp_verifier_available` `:34732` with a Rust-simplex fallback), the linear-node rescue `_linear_node_lp_rescue` (#1611 D-30, `solver.py:37710`), and the gated free-column node certificate (`DISCOPT_FREE_COLUMN_LP_CERT`, `mccormick_lp.py:446`, `:535`). | **yes** (#1537, #1611) | not in the brief |
| R7 mentions only | `_relax/partition_selection.py:67` (docstring), `cli.py:76` (deps listing), `amp.py:2294` (docstring), `_tape_nlp_evaluator.py:87` (docstring). | — | text edits in P6 |

Packaging/CI: `highspy>=1.11,!=1.12.0` is a **core** dependency (`pyproject.toml:44`); the
`highs` extra is a no-op alias (`:73`). CI lanes: `HiGHS route (highspy …)`
(`.github/workflows/ci.yml:721`) and the template for a no-dependency lane, `Python
without JAX` (`:780`, with its corpus guard `python/tests/test_1873_jax_optional.py`).
HiGHS-route-only flags that die with the route: `DISCOPT_HIGHS_MASTER_LP_CUTS`,
`DISCOPT_HIGHS_MASTER_NO_SUBMIP`, `DISCOPT_HIGHS_MASTER_LP_CUT_ROUNDS`,
`DISCOPT_HIGHS_DROP_NEGLIGIBLE`, `DISCOPT_MILP_ROW_FORM`, `DISCOPT_MILP_IMPLIED_BOUNDS`,
`DISCOPT_MILP_COEF_TIGHTEN`, `DISCOPT_MILP_CONCURRENT_CROSS`, `DISCOPT_MILP_FIXPROP_SEED`.

### 1.2 What the Rust engine has and lacks (verified by grep at `1fe78b1`)

Has: dual simplex with warm start (`lp/simplex/dual.rs`), primal (`primal.rs`), FeralLU
sparse LU + product-form updates (`linsolve.rs`), scaling, crossover, Farkas, NS safe
bounds (`refine.rs`), exact dual projection (`project.rs`, #2115 graduated), FBBT
presolve (`presolve.rs`, dimension-preserving), CSC end to end (`sparse-milp-plan.md`
T0–T4). MILP driver: cover, GMI, aggregated c-MIR (#2107, `DISCOPT_MILP_CMIR` graduated
ON per `flag-retirement-audit.md:416`), cut selection, pseudocost + strong branching,
diving, RINS (`DISCOPT_RINS`), RENS (`DISCOPT_RENS`, default OFF — failed bar 2c, A14),
reduced-cost fixing, node NS bound + projection, lazy-constraint entry
(`solve_milp_lazy_csc_py`), deadline polling (51 sites).

Lacks (0 hits in `lp/` and `bnb/milp_driver.rs`): interior point (POUNCE is the only IPM
and it is rejected for rigor, #145), hypersparse ftran/btran, LP presolve with postsolve,
ranging, conflict analysis, restarts, clique/implication tables, implied-bound cuts,
feasibility-jump/shifting heuristics, a persistent modifiable MILP handle. `mir.rs` is
not wired into the driver (`cmir.rs` header).

### 1.3 Parity, quantitatively — the end state (binding thresholds)

Every threshold cites the measurement it comes from; the P0 baselines re-measure each at
`HEAD` before any engine task runs, and **a P0 number replaces the figure here**.

| Role | Panel (TL, tolerance) | HiGHS today | Rust today | **Parity bar** | Basis |
|---|---|---|---|---|---|
| R1-LP | L panel, 25 LPs from `ref/HiGHS/check/instances` @ v1.15.1 (P0.1; was 24) (20 s) | **P0.3:** H 25/25 certified, 21.15 s; H0 0.32 s (was 24/24, 3.16 s) | **P0.3:** 24/25 (`perold` `feasible`), 47.42 s, geomean R/H **1.41×**, R/H0 ≈ 155× (was 21/24, 116.8 s, 11.9×) | **25/25 certified, `incorrect_count=0`, per-instance geomean wall ratio ≤ 3.0** | the existing `lp_vs_highs_geomean ≤ 3.0` gate, `benchmarks.toml:218` |
| R1-MILP (small) | M panel, 24 MILPs, `discopt_benchmarks/data/lp_milp/manifest.json` (20 s, gap 1e-4) | **P0.3:** H 24/24, 32.54 s, 518 nodes; H0 4.90 s, 328 nodes (was 24/24, 5.35 s, 217) | **P0.3:** 18/24, 119.92 s, 117,596 nodes, geomean R/H 1.73× (was 19/24, 104 s, 610,256) | **≥ 23/24 certified, geomean wall ratio ≤ 3.0, 0 incorrect** | one instance of slack because `issue-2173` already favours Rust 0.6× |
| R1-MILP (MIPLIB-38) | MIPLIB 2017 easy, size-filtered, validated (20 s, `mip_rel_gap` 1e-4) | **P0.3:** H 35/38, 253.17 s, 14,015 nodes; H0 35/38, 171.04 s (was 38/38, 72 s, 15,417) | **P0.3:** 14/38, 344.89 s, 708,937 nodes, geomean R/H 1.78× (was 21/38, 400 s, 4.69 M) | **≥ 34/38 solved, total wall ≤ 2× HiGHS, 0 incorrect** | §4 of that plan: "within a few instances … wall within ~2×"; D2 (§3) decides what happens short of it |
| R2a | §76.4 route panel: 26 in-repo diverted instances × spans {0,3,6} + 16 portfolios = 94 (30 s) | 92 optimal, 339.5 s (`performance-plan.md` §76.4) | retired `auto` arm: tied on syn/rsyn with kernel ON (20/1040 s vs 20/1040 s), lost with kernel OFF (22 vs 31) (`solver.py:9922-9931`) | **certified ≥ HiGHS arm's count, total wall ≤ 1.25×, 0 lost certificates** | the 1.25× is the noise floor the §76.4 spans show (20–30 s per span on six rows) |
| R2b | every decomposition test + the GDPlib/`gdp_parity` masters replayed (`gdp-parity-baseline.md`) | HiGHS wins large masters: total 81 s vs 114 s | Rust SGM 2.8× faster on small (`master.py:26-39`) | **identical certified set; SGM ≤ 1.0 (Rust no slower by SGM) and total ≤ 1.5×** | G3.2's own bar, inverted |
| R3 | NLP-BB convex corpus subset (the #781 probe set + §21 Regime-2 panel, 153 instances) | — | — | **cert-clean on the Regime-2 panel; root bound after the cut loop ≥ HiGHS-engine bound − 1e-6(1+\|b\|) on ≥ 90 % of instances and never looser by > 1e-4 relative; nodes/wall not worse in total** | this is a *bound-changing* port (§1.4) |
| R4 | captured node LPs (P0.5 fixture, ≥ 50) + convex corpus end to end | — | — | **node_count and certified objective exactly unchanged on the corpus; on fixtures NS(Rust dual) ≥ NS(HiGHS dual) − (CERT_ABS + CERT_REL·\|v\|) on every LP, 0 bounds lost** | CERT constants `lp_milp_highs.py:63-64` |
| R5 | every `test_iis*`/`test_1634*` IIS fixture + the MIPcc27 smallset if on disk | — | — | **identical IIS element sets on every fixture (same `order`), `proven_irreducible` identical, wall ≤ 2×** | deletion filtering is order-deterministic, so the set must match exactly |
| R6 | #1537 / #1611 / #1869 regression files + corpus | — | — | **verdicts identical on the regression files; corpus node_count/objective exactly unchanged** | bound-neutral |
| Package | `pytest -m smoke` with `highspy` uninstalled; corpus guard | — | — | **green; 0 recorded `highspy` import attempts on the default path** | mirror of `test_1873_jax_optional.py` |

### 1.4 Verification regimes (CLAUDE.md §5)

- **Bound-neutral** (R4, R6, marshalling, harness, the persistent handle, the LP-route
  engine seam): `node_count` and certified `objective` **exactly** unchanged on the
  certifying panel; LP objectives equal to 1e-9 relative. Any drift is a bug. Exception
  written down now: R4's Rust dual is a *different* vector, so its NS value can differ by
  roundoff; the gate is still exact node_count/objective on the corpus, and a drift that
  traces to a strictly tighter bound is re-filed as bound-changing, never waved through.
- **Bound-changing** (R3 — a different optimal basis yields different GMI cuts; every P4
  lever; P2.2 LP presolve if duals change): flag, default-OFF, differential panel ON vs
  OFF over the named corpus requiring BOTH *cert-clean* and *net-positive*
  (`issue2115_graduation_panel.py` is the template: executed-check counter, mechanism-
  fired counter, interleaved subprocess arms, false-primal screen). Pre-register the bars
  in the script docstring before the run (A14's lesson: a bar found to be mis-specified
  after it fails is not thereby met).

### 1.5 Binding negative results — do not schedule these again

From `milp-competitiveness-plan.md` (A2–A14, B1): symmetry, strong-branching budget,
node selection, pseudocost scaling, in-tree cut aging, branching on unreliable
pseudocosts, relaxing the cut-cleanup numerical gates, "more GMI". From
`sparse-milp-plan.md`: NS-on-POUNCE-IPM, exact-simplex RLT at qap scale. From
`gdp/algorithms/master.py:26-39`: a size-gated master default is "a new hypothesis, not
measured" — it may be scheduled only with an entry experiment. From the routing plan
(#1600, owner 2026-10-03): native rows vs `StdForm` on the HiGHS route — not planned.

---

## §2 Branch model

- One long-running branch `claude/happy-hamilton-0vvo3b` from `main`. **Sync (owner,
  2026-11-04): rebase onto `origin/main` once a day** to stay current (this replaces the
  original "merge after every `main` change to the engine paths, at least weekly"). The
  branch is single-owner, so the rebased branch is pushed with `--force-with-lease`.
  A rebase changes the code under test, so it is **never** done while a measurement run is
  in flight (the panel workers re-import `discopt` per solve): it waits for the run to
  finish, then `maturin develop --release` / `pip install -e .` and the standing gates
  re-run before the next task. Baseline data is labelled by the commit it measured; after
  a rebase that commit survives only in the reflog, so a run JSON's
  `header.commit` is the measurement's identity, not a branch commit. Last sync:
  2026-11-04 — a parallel session rebased the plan onto `main` and merged this branch back
  (`ff3da5b`); this tree synced to it at `07f71bbd` after P0.3's run. Next due 2026-11-05.
- **Checkpoint PRs to `main`** (each independently reviewable, CI-green): the P0
  harness; each bound-neutral port once its exact-identity gate is green; each flag that
  *graduates* (default flip + audit row) or *retires* (deletion + audit row); the P6
  packaging flip. Cut a short-lived PR branch from `claude/happy-hamilton-0vvo3b`, containing only
  that task's commits, and merge it to `main`; then sync `main` back. Flags that are
  built but not yet graduated stay on the feature branch until their panel runs.
- Name the task ID in every PR title (`feat(highs-parity): P1.3 …`). Use `Contributes to
  #1671` until M-1671's Done holds; never a closing keyword near a negation.

---

## §3 Decision points (the loop stops and asks)

| ID | After | Question for the owner |
|---|---|---|
| **D0** (M-1671) | P0.7 | Which of §4's three closure routes for #1671 to take, given the P0.3/P0.7 numbers. |
| **D1** | P2.6 | R1-LP parity reached → flip the pure-LP class to Rust (P6.1). If missed after P2.1–P2.5: keep HiGHS for LP (HiGHS stays *optional* only if R1-MILP also lands) or fund one more LP task. |
| **D2** | P4.9 | R1-MILP: MIPLIB-38 ≥ 34/38 at ≤ 2× wall reached → flip. If not: accept the *narrower* end state "HiGHS optional for R1 by class" (Rust default for instances the harness classifies as small, HiGHS opt-in otherwise — needs its own entry experiment), or keep HiGHS as the MILP default and stop the MILP track. |
| **D3** | P5.3 | R2a/R2b masters on Rust — and whether the #2069 independent confirmation (today: Rust confirms HiGHS) is still required when the primary *is* Rust, and what plays the second engine. |
| **D4** | P6.6 | Delete HiGHS entirely vs keep `discopt[highs]` as a documented opt-out (CLAUDE.md §5 state 3) with the route code retained. |

---

## §4 Milestone M-1671 — close #1671 (and what it does to #1634)

**The issue.** HiGHS 1.15.1's MIP presolve cycles forever in `HPresolve::fastPresolveLoop`
on the #1667 unit-commitment hand-off (2904 × 4344, 240 binaries), polling neither
`time_limit` nor callbacks; the route works around it by switching off presolve rule 14
(`_PRESOLVE_RULE_SPARSIFY`, `lp_milp_highs.py:97-117`), and rule 13 for #1634
(`:75-96`). The same `MILP_PRESOLVE_RULE_OFF` is pinned on every `milp_highs` master
solve (`milp_highs.py:656, 993, 1595`). `_iis_milp.py` does **not** set it (its probes
run presolve `choose`/`off`, `:244-271`), so the IIS engine is exposed to the cycle on an
instance of that class. The upstream fix is unreleased (ERGO-Code/HiGHS PR 2962 per the
canary's bisect; the brief names #3367); the issue's own close condition ("re-measure on
a release with the fix") is outside our control.

**Our close condition:** *no default discopt path depends on HiGHS presolve.* Checkable
as:

1. `python/tests/test_1667_highs_sparsify_presolve_cycle.py::test_unit_commitment_returns_within_its_time_limit`
   passes on the **default route** (no `DISCOPT_LP_MILP_BACKEND` set) with the sparsify
   bit removed from the default path — `optimal`, objective `450623.7692` (rel 1e-6),
   wall < 60 s, as the test already asserts;
2. the default `Model.solve()` of a pure MILP sets no `presolve_rule_off` (new assertion
   in that file; `test_sparsify_is_off_on_every_milp_route` is re-scoped to the opt-in
   HiGHS route only);
3. `test_1671_highs_presolve_cycle_canary.py` is re-scoped to the opt-in route or deleted
   with its measurement recorded (`:19-39` of the canary); the sparsify-off bit stays
   documented as the opt-in route's workaround (CLAUDE.md §5 state 3);
4. wall time is bounded by the Rust driver's deadline polling (the issue's "discopt cannot
   bound a HiGHS call that never polls"), pinned by a test that gives the Rust route the
   UC model with `time_limit=5` and asserts return within 5 s + the certificate overhead;
5. the closing comment is drafted under M-1671 in §5 (not posted).

**Three routes to it, ordered by cost; D0 picks after P0.7 and P0.3 have numbers.**

- *(a) Rust default for the pure-MILP class* — the full P4 track (far; D2).
- *(b) Rust default for the sub-class the cycle reaches* — a measured routing predicate
  (e.g. by row count / binaries / slack-form density) under which the Rust driver already
  certifies at HiGHS-comparable wall on the M panel + MIPLIB-38, with HiGHS for the rest.
  Needs its own entry experiment (the predicate must be a general property, CLAUDE.md
  §2). The witness (2904 × 4344) would need to fall inside it; P0.7 tells us whether Rust
  certifies it at all in 20 s.
- *(c) `presolve=off` on the default HiGHS MILP route* — removes every dependence on
  HiGHS presolve at once (the cycle and the #1634 rule), at HiGHS's own measured cost:
  no_presolve 36/38 at 1.7× wall vs full 38/38 (`milp-competitiveness-plan.md` §1c). It
  is loud and not an approximation, but it is a performance regression on the default
  path; the owner decides whether that is acceptable as an interim state.
- *Rejected as the closure:* a subprocess hard-kill watchdog around `h.run()`. It bounds
  wall time (a legitimate loud refusal, CLAUDE.md §3) but leaves the default path
  dependent on HiGHS presolve, so it does not meet the close condition; it may be built
  as the interim time-limit guard if the owner wants (ask in D0, do not presume).

**#1634.** Its presolve-free cross-check (`_cross_check_presolve`, `lp_milp_highs.py:3168`;
`milp_highs.py:792`) and rule-13 bit exist because HiGHS's parallel-column rule produced
false bounds. On route (a) or (b) the default path no longer runs HiGHS presolve, so #1634
becomes **moot on the default path** and closable as "workaround retained on the opt-in
HiGHS route"; on (c) it is moot outright. The Rust driver has no dimension-reducing
presolve (`presolve.rs` is FBBT only), so it does not inherit the defect class; P4.8
(dimension-reducing presolve) must re-open this question with a postsolve-verified
certificate before it graduates. Record the #1634 disposition in M-1671's closing memo.

---

## §5 Tasks

Format per task: **Scope** (files) · **Entry** (hypothesis + falsifying measurement on real
instances, where the task rests on one) · **Done** (commands + numeric thresholds) ·
**Kill** (what falsifies it, and the on-kill action) · **Regime** (§1.4).

### Phase 0 — measurement first (no engine work before every baseline exists)

- [x] **P0.1 — Harness audit and the L-panel manifest.** Scope:
  `discopt_benchmarks/scripts/lp_milp_panel.py`, `lp_milp_loader.py`,
  `discopt_benchmarks/data/lp_milp/manifest.json`. The manifest holds only the 24 M-panel
  members + `i1183`; the L panel's 24 LPs came from `ref/HiGHS/check/instances`, which is
  **absent on this machine** (0 files), and `run_benchmarks.py:853` records "No Netlib
  loader exists in this tree" — so `lp_vs_highs_geomean` (`benchmarks.toml:218`) is
  unmeasurable today. Add a `fetch` subcommand that clones ERGO-Code/HiGHS at the tag of
  the installed `highspy` into `ref/HiGHS` (git-ignored; record the commit in the
  manifest), re-`screen` the L panel (24 LPs) and the infeasible/unbounded panel (18)
  into the manifest, and add: a per-row **route marker** assertion (`solver_stats
  ["route/lp_milp_backend"]` must equal the arm), an executed-comparison counter per
  panel, `--load-gate` (refuse to report wall when 1-min load > gate at start or end;
  statuses/nodes still reported), and a `--rounds` interleave with order rotation.
  Done: `python -u discopt_benchmarks/scripts/lp_milp_panel.py run --manifest
  … --panel L --panel M --panel INF --arms R,H,H0 --rounds 1 --time-limit 5` prints
  `executed=` ≥ 66·3 and exits 0; `screen` lists every exclusion with its reason; a unit
  test pins that a wrong-route row raises. Kill: an L-panel instance the loader cannot
  round-trip (`Model.to_mps` → HiGHS) → exclude and list it; if > 4 of 24 are excluded,
  the panel is re-sourced (Netlib `.mps` via HiGHS `readModel`) before P0.3. Regime:
  harness (bound-neutral by construction).
  **Result (2026-11-04):** `fetch` pins HiGHS `v1.15.1` @ `04024d70` (100 instance
  files); `screen` gives **L 25 / M 24 / INF 16 = 65** (the plan's 24/24/18 was screened
  at an older HiGHS tree, §7), every exclusion listed with its reason (1 LP round-trip
  exclusion, `nan0`: NaN reaches the `.nl` writer — kill not fired). Done run (`--rounds 1
  --time-limit 5 --load-gate 2`): `executed=195` = 65 × 3, i.e. every member × arm,
  `incorrect_count=0`, exit 0, walls withheld on 10/65 rows by the gate. Statuses R/H/H0:
  L 24+1 `feasible` / 25 / 25 optimal; M 16 (+3 `time_limit`, 5 `feasible`) / 24 / 24;
  INF 16/16/16 correct verdicts. The route check fired on its first real run:
  `gams10am` is answered by root presolve on **both** arms (neither engine runs), now
  reported as route `presolve` rather than raised or counted as an engine row.
  Regression test `python/tests/test_highs_parity_p01_panel_harness.py` (21 tests; 19
  error before — load marker `check_route` absent — 21 pass after).
- [x] **P0.2 — Rebuild the MIPLIB-38 harness and check it in.** Scope: new
  `discopt_benchmarks/scripts/miplib_panel.py` (+ `data/miplib/panel.json`). The
  §6 harness of `milp-competitiveness-plan.md` is not on disk (`scratchpad/` has no
  `miplib/`). Rebuild per its §0: MIPLIB 2017 "easy" set fetched from miplib.zib.de
  (sha256 recorded; files not vendored), size-filtered as there (so the 38 can be
  reproduced — if the filter cannot be recovered, pick the 38 names from §0/§2.5/§4 of
  that plan, which lists 20+ by name, and complete the set by the same size rule, recording
  the choice), each instance **validated** by raw HiGHS reproducing the published optimum
  from the converted arrays; arms `R` (`DISCOPT_LP_MILP_BACKEND=rust`), `H` (default),
  `H0` (raw highspy), matched `mip_rel_gap=1e-4`, TL 20 s, fresh subprocess per solve,
  interleaved, node counts as the primary metric, executed counter, load gate, the
  `_assert_discopt_under_test` marker. Done: `screen` produces a 38-member manifest with
  0 oracle mismatches; a 3-instance `run --time-limit 5` prints `executed=9`, exits 0.
  Kill: fewer than 30 instances pass validation → record and run with what validates;
  thresholds in §1.3 scale as `⌈0.9·n⌉`. Regime: harness.
  **Result (2026-11-04):** the size filter is not recorded anywhere, but the 38 *names* are:
  `scratchpad/a14/rens_panel.json` (A14's run on this panel) lists exactly 38, and each of
  its reference optima equals the `miplib2017-v31.solu` `=opt=` value, so the panel is
  **recovered, not reconstructed** (no size rule had to be guessed). `miplib_panel.py
  fetch` caches the 38 `.mps.gz` + `.solu` in git-ignored `ref/miplib2017/` (764 KB) and
  pins 39 sha256s in `data/miplib/panel.json` (a re-fetch verifies them); `screen` →
  **38 members, 0 oracle mismatches**. `neos-3118745-obra` was *unvalidated* at the 60 s
  validation limit (raw HiGHS timed out; not a wrong objective) and validated at 900 s
  (255, 41,713 nodes, 146 s) — it is the panel's hardest instance for HiGHS on this host.
  `run` delegates to `lp_milp_panel.py run` (P0.1's route marker, counters, load gate,
  rotation) with the new `--h0-subprocess`, so every arm is a fresh process. Done run
  (`gr4x6,p0201,neos-911970`, TL 5 s): `executed=9`, `incorrect_count=0`, exit 0.
  Regression test `python/tests/test_highs_parity_p02_miplib_panel.py` (6 error before,
  6 pass after). Branch synced with `origin/main` (#2125 `lp/` commits) in this iteration.
- [x] **P0.3 — Fresh R1 baselines at HEAD.** Scope: results under
  `docs/dev/data/highs-parity/baseline-<commit>/`. Run P0.1's L/M/INF and P0.2's
  MIPLIB-38, arms R/H/H0, 3 rounds, TL 20 s, load gate < 2 (if the gate never opens, run
  anyway and mark walls *indicative*; nodes/statuses carry). Done: a table per panel
  (solved, certified, total wall, median over solved, nodes, incorrect) written into §6
  with `incorrect_count = 0` on H; the R column replaces the "Rust today" figures in §1.3;
  `executed` ≥ 3·(25+24+16+38)·3 (panel sizes re-screened by P0.1, §7). Kill: `incorrect_count > 0` on **either** arm → that is
  a P0-blocking correctness defect: file it, fix it before any other task (CLAUDE.md
  correctness-first), and re-run. Regime: measurement.
  **Result (2026-11-04):** solver at `dcdac02` (= `53d0e42`; the run spans two
  container restarts, resumed under the code-under-test fingerprint check, P0.3 harness
  commits), data in `docs/dev/data/highs-parity/baseline-dcdac02/`. **`executed` = 585 +
  342 = 927** = the Done bound, **`incorrect_count = 0` on every arm** (kill not fired).
  The load gate (< 2) never held for the whole run on this 4-core host (the run's own
  solves keep the 1-min load near 2): timing-valid rows L 14/25, M 4/24, INF 11/16,
  MIPLIB 12/38, so every wall below is *indicative*; statuses, certificates and nodes
  carry. Table in §6; §1.3's "today" columns replaced. What the numbers say:
  - **The H route's own overhead dominates its walls here**: geomean H/H0 = 110× (L),
    20× (M), 56× (INF), 3.3× (MIPLIB). `25fv47` takes 4.35 s on H (sd 0.1 s over 3
    rounds) vs 0.16 s raw; the routing plan measured ~0.82 s. So R/H is 1.41× (L),
    1.73× (M), 1.72× (INF), 1.78× (MIPLIB), while R/H0 is two orders larger on LP.
    **Open for the owner (raised at D0):** §1.3's "wall ratio ≤ 3.0" does not say whether
    it is against H (the route Rust would replace) or H0 (HiGHS's own speed); on L the
    two readings are 1.41× vs ≈ 155×. P2.1 should attribute the H − H0 gap first.
  - Rust uncertified: L `perold` (`feasible`); M `2122`, `issue-2446`, `p0548`
    (`time_limit`), `gesa2`, `gt2`, `sp150x300d` (`feasible`); INF `gas11` (unbounded,
    uncertified on R **and** H). MIPLIB: 24/38 unsolved, 21 of them `feasible` — many
    well before 20 s (`neos-911970` after 7 nodes), which P4.1 must explain;
    `neos-1425699` hits `time_limit` where `milp-competitiveness-plan.md` §0.1 recorded
    29 nodes / 0.00 s at matched gap (different entry: the plan drove the Rust driver
    directly; this is `Model.solve`).
  - HiGHS on this host: H misses `enlight_hard`, `neos-2624317-amur`,
    `neos-3118745-obra` (`feasible`); H0 misses `amur`, `gsvm2rl3`, `obra` at 20 s —
    35/38 each, not the 38/38 of the 14-core dev machine.
  - **Sync after the run** (`07f71bbd`: the branch as rebased onto `main` by a parallel
    session, plus these commits) brought #1917 G3.2 into `milp_driver.rs` (coefficient
    tightening, tableau c-MIR, both default OFF), `gomory.rs`, `fbbt.rs`, `lp_pounce.py`.
    Drift check on the R arm at the synced HEAD, L + M, 1 round: **49/49 identical
    status, every optimal row's node count equal to a baseline round** — the baseline
    describes HEAD on those panels (MIPLIB not re-checked). `cargo test` 980 + smoke 4482
    passed after the sync.
- [ ] **P0.4 — Master baselines (R2a/R2b/R2c).** Scope: new
  `discopt_benchmarks/scripts/master_parity_panel.py`. (i) R2a: replay the captured
  masters (`python/tests/data/oa_masters/*.npz`, `issue2069/clay0203_master.mps.gz`,
  `milp_default_box_clay0203_master.npz`) through `milp_highs.solve_milp` and
  `milp_simplex.solve_milp` (what `_run_simplex_engine`, `milp_highs.py:1170`, already
  does), seeded identically; plus the §76.4 route panel end to end as a **single arm**
  (HiGHS master, the shipped default) to refresh the 92/94, 339.5 s figure. (ii) R2b: every
  Benders/GBD/Lagrangian test model plus the GDPlib masters of `gdp-parity-baseline.md`,
  `get_decomposition_master_solver` forced to each engine via a new test-only env
  `DISCOPT_DECOMP_MASTER={highs,simplex}` (read in `lp_backend.py:260`; default
  unchanged). Done: per-engine table (status, objective, bound, nodes, wall) with a
  comparison counter ≥ number of masters × 2 and 0 objective disagreements where both
  certify. Kill: a disagreement where both certify → correctness defect, P0-blocking as
  in P0.3. Regime: measurement.
- [ ] **P0.5 — Fixture capture for R3/R4/R6.** Scope: new `python/tests/data/highs_parity/`
  + a capture script. Instrument (temporary, not committed) `_root_cuts._solve_lp`,
  `convex_node_bound._lp_dual` and `solve_lp_std`'s MINLP-path callers to dump each LP
  (`StdForm` arrays) with HiGHS's `x`, `y`, status, and for R3 the GMI cuts emitted, over
  the convex NLP-BB subset of the 74-file corpus. Keep ≥ 50 R4 node LPs, ≥ 20 R3 root LPs
  (with cuts), every #1537/#1611 arbitration LP reached. Done: fixtures ≤ 5 MB total,
  loader test asserts shapes and that HiGHS re-solve reproduces the stored objective to
  1e-9. Kill: none (capture). Regime: measurement.
- [ ] **P0.6 — IIS baseline.** Scope: new `discopt_benchmarks/scripts/iis_parity_panel.py`.
  Run `compute_iis(include_integrality=True)` on every IIS-exercising test model in
  `python/tests/` (grep `compute_iis`/`milp_iis`) and the MIPcc27 smallset **if** on disk
  (else record its absence), storing element sets, `n_solves`, `proven_irreducible`,
  wall. Done: ≥ 20 fixtures recorded; a re-run reproduces identical element sets
  (determinism check, counter printed). Kill: non-deterministic sets between two HiGHS
  runs on the same fixture → record; that fixture's parity bar becomes "within the
  observed HiGHS spread" and is listed. Regime: measurement.
- [ ] **P0.7 — Entry experiment for M-1671: the UC witness on the Rust driver.** Scope:
  `scripts`-free: run `test_1667_…::test_unit_commitment_returns_within_its_time_limit`'s
  `UC_SCRIPT` with `DISCOPT_LP_MILP_BACKEND=rust` at `time_limit` 20/60/120 s, 3 reps,
  fresh subprocess each, load gate; also with `presolve=False` and with
  `DISCOPT_MILP_CMIR=0` as attribution arms. Record status, objective, bound, nodes,
  wall. Done: a table in §6; Rust certifies `450623.7692` within 20 s **or** the table
  states the gap (bound/incumbent at 20/60/120 s). Kill: a Rust bound above
  `450623.7692 + 1e-6·|·|` or a wrong `optimal` → correctness defect, P0-blocking. Then
  **D0** (§3): present routes (a)/(b)/(c) of §4 with these numbers and stop.

### Phase 1 — bound-neutral ports and the engine seam (R6, R4, R3)

- [ ] **P1.1 — Engine seam for the verified LP route.** Scope: `lp_milp_highs.py`
  (`solve_lp_std` `:1657`, `_pass_model` `:1452`, `_new_highs` `:1638`,
  `phase1_infeasibility_proof` `:1011`, `recession_ray` `:1065`), new
  `python/discopt/solvers/lp_std_engine.py`. Hypothesis: the certificate ladder
  (`readback_problem`, `feasibility_problem`, `ns_bound`, `fbbt_box`, `exact_ns_bound`,
  `farkas_verified`, `primal_ray_verified`, `certify_lp_dual`) is engine-independent and
  only the raw solve (`x`, `y`, status, dual/primal ray, presolve-off re-solve) is HiGHS.
  Refactor `solve_lp_std(sf, *, time_limit, engine="highs")` to call an
  `LpEngine` protocol; implement `HighsEngine` (today's calls, byte-for-byte behaviour)
  and `RustEngine` over `solve_lp_warm_csc_py` (`lp_bindings.rs:552`; returns `x`, row
  duals, ray — verify the ray and the infeasible/unbounded status mapping against
  `lp_simplex.solve_lp` `:134`). Done: with `engine="highs"` every test in
  `test_lp_milp_highs_route.py` (39) and `test_lp_huge_finite_box_937.py`,
  `test_850_…`, `test_issue_1229_…`, `test_solver_duals.py` passes unchanged **and** the L
  panel H arm reproduces P0.3's statuses/objectives/provenance labels exactly (counter ≥
  24); with `engine="rust"` the L panel certifies ≥ P0.3's R count with 0 incorrect and
  every `optimal` carries a provenance label from the same ladder. Kill: a ladder helper
  that turns out to read HiGHS state (not `sf`/`y`) → list it; if the Rust engine cannot
  supply a verifiable Farkas ray on ≥ 1 of the 18 INF-panel instances where HiGHS can,
  the phase-1 proof path (`:1011`) must close it, else record the gap in §7 and the LP
  default stays HiGHS. Regime: bound-neutral (HiGHS arm exact; Rust arm measured).
- [ ] **P1.2 — R6: MINLP-path verified LPs on the Rust engine, HiGHS as fallback.**
  Scope: `solver.py` `_linear_node_verified_empty` (`:34903`),
  `_highs_lp_verifier_available` (`:34732`), `_linear_node_lp_rescue` (`:37710`),
  `mccormick_lp._verified_free_column_bound` (`:535`). Switch the three callers to
  `engine="rust"` first, falling to `highs` only on a Rust `Numerical`/`IterLimit` (logged
  with a counter in `solver_stats`). Done: the #1537, #1611, #1869 regression files pass;
  the P0.5 R6 fixtures give identical verdicts (`infeasible`/`bound`/`none`) on every LP
  (counter printed); the 74-file corpus at 60 s, `deterministic=True`, has **identical**
  status/objective/node_count/bound to a `main` run on every instance (the H4 guard
  script `lp_milp_minlp_guard.py` pattern, compared=74, drift=0). Kill: any corpus drift →
  root-cause; if it is a Rust verdict `none` where HiGHS certified (lost bound), the
  fallback order stays HiGHS-first for that caller and the gap goes to §7 with the LP
  attached to P2.1's profile list. Regime: bound-neutral.
- [ ] **P1.3 — R4: Rust dual first in `_lp_dual`.** Scope: `convex_node_bound.py:215-260`.
  Reorder to `_simplex_dual` → HiGHS → row-equilibrated HiGHS; `_simplex_dual` gains the
  slack-form `sf` input so all three solve the same LP. Done: on the P0.5 R4 fixtures,
  `certify_lp_dual(y_rust)` ≥ `certify_lp_dual(y_highs)` − (CERT_ABS + CERT_REL·|v|) on
  every LP and no `None` where HiGHS gave a bound (counter ≥ 50); convex corpus subset
  (NLP-BB instances) at 60 s, `deterministic=True`: node_count and objective identical to
  `main` on every instance; `test_2067*` passes. Kill: ≥ 1 fixture where the Rust dual
  loses a bound HiGHS found → keep HiGHS first *for that LP class* (state the class),
  record in §7, attach the LP to P2.1. Regime: bound-neutral with the §1.4 R4 exception.
- [ ] **P1.4 — R3: root-cut LP and GMI on the Rust simplex (flagged).** Scope:
  `_root_cuts.py` `_solve_lp` (`:410`), `separate_gmi` (`:532`), new
  `DISCOPT_NLPBB_ROOT_CUTS_ENGINE={highs,rust}` (default `highs`), Rust: a binding that
  solves the root LP cold, returns `x`, duals, basis and the GMI cuts via
  `separate_gomory_cols` (`gomory_cuts_py`, `lp_bindings.rs:209`, is the existing entry —
  verify its tableau convention equals the one documented at `_root_cuts.py:534-541`).
  Entry: on the P0.5 R3 fixtures, the Rust GMI cut set from the Rust basis vs HiGHS's from
  its basis — measure per-fixture root-bound after one round with each; hypothesis: the
  Rust round closes ≥ 90 % of what the HiGHS round closes on ≥ 90 % of fixtures. Done:
  the §21 Regime-2 panel (153 instances, the script named in `performance-plan.md` §21)
  re-run ON (`rust`) vs OFF (`highs`): cert-clean, and net-non-inferior (dual bound
  tighter-or-equal within 1e-6 relative on ≥ 90 %, never looser by > 1e-4 relative, total
  nodes and wall ≤ 1.05×); `test_nlpbb_root_cuts_781.py`'s exact-enumeration soundness
  test runs on the Rust engine (0 unsound cuts). Kill: entry < 90 % → the Rust basis is
  systematically worse for GMI; record, keep HiGHS default, and move the task after P2
  (a better cold-start basis may fix it). Panel cert-clean but not non-inferior → flag
  stays OFF as state 3 ("HiGHS-less installs get this; HiGHS installs keep HiGHS cuts")
  until P6 forces the question. Regime: bound-changing.

### Phase 2 — the LP engine (R1-LP)

- [ ] **P2.1 — Attribution of the L-panel losses.** Scope: `scratchpad`-style profile under
  `docs/dev/data/highs-parity/p2.1/`. On the 3 R failures (25fv47, shell, standmps per
  P0.3's refresh) and the 10 largest R/H wall ratios: `samply`/`perf` on the Rust route,
  split into Python marshalling (`_highs_std_form`, `_solve_milp_simplex`), cold
  factorization, refactor interval (`DISCOPT_LP_REFAC_INTERVAL`), pivot loop
  (pricing/ratio test), degenerate stalls, and iteration count vs HiGHS's
  `simplex_iteration_count`. Done: a table with ≥ 13 rows, each attributed to one bucket
  with a % of wall, and a ranked list of P2.2–P2.5 by expected gain **with the ordering
  of those tasks re-written here from the data**. Kill: none (measurement) — but if
  > 50 % of the loss is Python marshalling, insert a bound-neutral marshalling task
  before P2.2 and record it.
- [ ] **P2.2 — Cold-start basis and dual-slack warm start for the LP route.** Scope:
  `milp_driver.rs::dual_slack_basis`, `lp/simplex/primal.rs`, `basis.rs`. Hypothesis
  (test in Entry on the P2.1 instances): a crash basis (slack basis + triangular
  structural crash, CPLEX/HiGHS style) cuts cold iterations ≥ 30 % vs the all-slack
  start. Flag `DISCOPT_LP_CRASH_BASIS` default OFF. Done: on the L panel, ON vs OFF
  interleaved 3 rounds: identical objectives (1e-9 rel), identical certification, total
  iterations −30 % or wall −20 %; `cargo test` golden tests (`driver_matches_golden`,
  `branching_golden`) **unchanged with the flag OFF**. Kill: < 10 % iteration saving →
  retire flag + implementation. Regime: bound-neutral in objective (vertex may differ —
  it is an LP optimum either way; node-count golden must hold with the flag OFF; MILP
  node drift with it ON makes it bound-changing and it then needs the P4-style panel).
- [ ] **P2.3 — Hypersparse FTRAN/BTRAN and row-wise pricing.** Scope: `linsolve.rs`,
  `dual.rs` (Devex pricing loop), `feral` interface. Entry: on the P2.1 instances, measure
  the density of `B⁻¹a_q` and `eᵀB⁻¹` results; hypothesis: ≥ 50 % of solves have < 10 %
  fill, so a hypersparse path pays. Flag `DISCOPT_LP_HYPERSPARSE` default OFF. Done:
  bit-identical pivot sequence (`lp_iters` golden unchanged ON vs OFF on the `cargo`
  panel) and wall −25 % on the instances where the entry held. Kill: entry < 50 % or wall
  gain < 10 % → retire. Regime: bound-neutral (exact).
- [ ] **P2.4 — LP presolve with postsolve (row/column singletons, dominated/empty
  rows, doubleton equations), duals restored.** Scope: new `lp/simplex/lppresolve.rs`,
  `lp_std_engine.RustEngine`. Flag `DISCOPT_LP_PRESOLVE` default OFF. Entry: count
  removable rows/cols on the L panel with a scan script; hypothesis: ≥ 20 % of rows on ≥ 8
  of 24 instances. Done: ON vs OFF on L: identical objectives (1e-9), identical
  certification **through the full ladder on the postsolved duals** (NS gap ≤
  CERT_ABS + CERT_REL·|obj| on every certified instance), wall −20 % where the entry
  held; Rust unit tests cover each reduction's postsolve on crafted LPs. Kill: entry fails
  or any certificate lost → retire. Regime: bound-neutral (postsolve must be exact; a
  dual that fails the ladder is a bug, not a tolerance to widen).
- [ ] **P2.5 — Degeneracy: bound-flipping ratio test + cost perturbation on the LP
  route** *(≤ 2 iterations)*. Scope: `dual.rs` (`cost_perturbation_*` tests exist), the
  stall guard. Entry: per P2.1, the share of iterations spent in degenerate pivots on the
  three failures. Flag `DISCOPT_LP_DEGEN_PERTURB` default OFF. Done: the three failures
  certify in < 20 s with the ladder closing; iterations −30 % on degenerate instances;
  cert-clean L. Kill: < 10 % → retire. Regime: bound-neutral in objective (perturbation
  removed before the final verdict; the ladder certifies the unperturbed LP).
- [ ] **P2.6 — R1-LP graduation gate → D1.** Scope: `solver.py:37494` (`_lp_milp_backend`
  split: new `DISCOPT_LP_BACKEND`, default `highs`, for the pure-LP class only;
  `DISCOPT_LP_MILP_BACKEND` keeps governing MILP), docs. Done: L panel, Rust engine via
  P1.1's seam, flags of P2.2–P2.5 ON: **24/24 certified, 0 incorrect, geomean wall ratio
  ≤ 3.0**, INF panel 18/18 with verified certificates, 3 rounds under the load gate. Then
  flip `DISCOPT_LP_BACKEND` default to `rust`, graduate each P2 flag that contributed
  (audit rows), keep `=highs` as the opt-out, and run the H4 MINLP guard (drift=0). Kill:
  bar missed → **D1**.

### Phase 3 — persistent MILP handle and IIS (R5)

- [ ] **P3.1 — Rust `MilpHandle`.** Scope: new `crates/discopt-python/src/milp_handle.rs`,
  `milp_driver.rs` (`MilpOptions.presolve: bool` already exists, `:473`, "root
  feasibility-based bound tightening"; expose it per solve on the handle),
  `python/discopt/solvers/milp_handle.py`. A pyclass holding the CSC problem with `change_col_bounds(idx, lo, hi)`,
  `change_row_bounds`, `change_integrality(idx, is_int)`, `solve(time_limit, presolve,
  max_nodes)` → status/x/bound/nodes, `presolve_only()` (FBBT verdict), `set_solution`.
  Done: Rust tests: a handle re-solved after each toggle equals a fresh `solve_milp_csc`
  on the same data bit-identically (status, obj, bound, nodes) on the `sparse_milp_diff`
  panel + 20 random toggle sequences (counter printed); Python binding round-trips.
  Kill: none — this is scaffolding; if the handle cannot avoid touching MINLP-shared
  code (`lp/simplex/*`, `tree_manager.rs`), say which code and stop (owner froze MINLP
  code). Regime: bound-neutral.
- [ ] **P3.2 — `_iis_milp._Engine` on the handle, HiGHS as the opt-in.** Scope:
  `_iis_milp.py:111-470`; new `iis_engine="rust"|"highs"` kwarg on `milp_iis`/`compute_iis`
  (default `rust`), `_solve`'s presolve ladder mapped to `presolve_only()` →
  `solve(presolve=True)` → `solve(presolve=False)`. Done: P0.6 fixtures: identical element
  sets and `proven_irreducible` on every fixture (counter ≥ 20), wall ≤ 2× the HiGHS
  engine; `test_iis*`/`test_1634_*` pass on both engines (`test_1634_highs_parallel…`
  keeps HiGHS). Kill: a fixture whose IIS differs **and** whose Rust set is not an IIS
  (a kept element is removable, checked by one extra probe) → correctness defect; a
  differing but valid IIS on a fixture P0.6 flagged non-deterministic is accepted and
  listed. Regime: bound-neutral (verdict-identical).
- [ ] **P3.3 — IIS time bounding.** Scope: `_iis_milp.py` `_GRACE_S`, interrupt callback
  plumbing. The handle polls its deadline, so the HiGHS-specific overrun guard
  (`:64-69`) becomes route-specific. Done: a test that gives the Rust engine a probe with
  `time_limit=0.5` on `enlight`-class data returns within 1.0 s; `n_solves` and sets
  unchanged on P0.6. Kill: none. Regime: bound-neutral.

### Phase 4 — MILP search (R1-MILP), ordered by measured expected gain

Standing rule for every P4 task: flag default-OFF; entry experiment on MIPLIB-38
instances (never synthetic only); graduation panel = MIPLIB-38 + M panel, ON vs OFF,
interleaved, 3 rounds, bars pre-registered: (1) cert-clean, (2) solved-count ≥ OFF, total
nodes −15 % **or** solved +2 with nodes not worse, dual-bound losses ≤ 10 % of the open
population *by magnitude* (> 1 % relative, exceeding the OFF arm's own rep-to-rep spread —
the A14 lesson). Each graduation or retirement adds its audit row. **Order is provisional
and is re-written by P4.1.**

- [ ] **P4.1 — Re-attribution at HEAD.** Scope: `miplib_panel.py attribute`. On P0.3's
  unsolved R instances, classify at `gap_tol=1e-4`: no incumbent / incumbent short by
  (opt − inc)/(1+|opt|) / bound short by (opt − bound)/(1+|opt|) at 20 s and 60 s (the
  A5/A14 method, both budgets so the "frozen primal" finding is re-tested after #2107's
  c-MIR). Done: a table over every unsolved instance, and the **order of P4.2–P4.8
  rewritten here** with one line of justification each. Kill: none.
- [ ] **P4.2 — RENS re-run with the pre-registered magnitude bar.** Scope:
  `DISCOPT_RENS` (`milp_driver.rs`), no new code unless the bar demands a sub-MIP budget
  cap. A14 left RENS cert-clean, 12 incumbent wins / 0 losses, failing only a
  single-draw count bar it later called noise. Done: the panel above, 3 reps, dual-side
  bar by magnitude. Graduate (default ON, audit row) or **retire flag + implementation**
  (RINS stays). Kill: bar 2 fails → retire. Regime: bound-changing.
- [ ] **P4.3 — Cut repair instead of rejection.** Scope: `milp_driver.rs` cut cleanup
  (the numerical gates), `cut_select.rs`. Entry: count cuts rejected by range/dynamism
  gates at the MIPLIB roots and the root bound with those cuts *repaired*
  (SCIP-style: drop |a| < feastol by charging the box, tighten coefficients, re-check
  validity on the box) vs rejected — on ≥ 10 instances; hypothesis: ≥ 5 pp more root gap
  closed on ≥ 5 instances. Flag `DISCOPT_MILP_CUT_REPAIR`. Done: panel bars; feasible-point
  sampling (≥ 1e5 integer-feasible points over the panel) 0 violations. Kill: entry < 5
  instances → retire. Regime: bound-changing.
- [ ] **P4.4 — Tableau-row c-MIR / lifted cover (HiGHS's family).** Scope: `cmir.rs`
  (aggregation from tableau rows, not only original rows), `cover.rs` (lifting with
  bound substitution), driver wiring. Entry: on the four `mik-250-20-75-*` + `fiber` +
  `sp150x300d` roots (where §2.5 measured 4.5 % residual vs HiGHS 1.7 %), root gap with
  tableau-based c-MIR added; hypothesis: residual ≤ 2.5 % on mik. Flag
  `DISCOPT_MILP_TABLEAU_CMIR`. Done: panel bars; exact-enumeration soundness on small
  seeded MILPs 0 unsound cuts. Kill: mik residual > 3.5 % → retire. Regime: bound-changing.
- [ ] **P4.5 — Root restart.** Scope: `milp_driver.rs` root loop. After the root cut loop,
  if reduced-cost fixing + FBBT fixed ≥ 10 % of columns, re-run presolve + root loop on the
  reduced problem once (HiGHS's closer on `beavma`/`dcmulti`, §2.6). Entry: count such
  roots on MIPLIB-38; hypothesis ≥ 8 instances qualify and the restarted root bound
  improves ≥ 1 pp on ≥ 4. Flag `DISCOPT_MILP_ROOT_RESTART`. Done: panel bars. Kill: entry
  fails → retire. Regime: bound-changing.
- [ ] **P4.6 — Feasibility jump + shifting primal heuristics.** Scope: new
  `bnb/primal_fj.rs`. Entry: on P4.1's no-incumbent instances, a standalone FJ run (10 %
  of TL) finds a feasible point on ≥ half of them. Flag `DISCOPT_MILP_FJ`. Done: panel bars
  with the incumbent-verified screen; FJ points are verified before adoption. Kill: entry
  < half → retire. Regime: bound-changing (primal only; bounds untouched — still panelled).
- [ ] **P4.7 — Conflict analysis.** Scope: `milp_driver.rs` infeasible-node handling
  (Farkas ray → conflict constraint over the branching bounds), `pool.rs`. Entry: on
  `gsvm2rl3`/`neos-2624317-amur`-class instances (§2.6: branching-and-conflicts), count
  infeasible nodes whose Farkas ray involves ≤ 10 branching bounds; hypothesis ≥ 30 % of
  infeasible nodes. Flag `DISCOPT_MILP_CONFLICT`. Done: panel bars; each conflict
  constraint validated by feasible-point sampling. Kill: entry < 30 % → retire. Regime:
  bound-changing.
- [ ] **P4.8 — Dimension-reducing MILP presolve with postsolve** *(≤ 3 iterations, last:
  worth ≤ 2 instances / 1.7× in HiGHS's own ablation, §1c)*. Scope: reuse P2.4's LP
  reductions + integer-aware rounding, `presolve/substitute.rs` where it applies. Flag
  `DISCOPT_MILP_PRESOLVE_REDUCE`. Entry: reductions ≥ 20 % of rows on ≥ 8 instances. Done:
  panel bars; the postsolved incumbent re-verified on the **original** rows; the #1634
  question (§4) answered in the task record. Kill: entry fails → retire. Regime:
  bound-changing.
- [ ] **P4.9 — R1-MILP graduation gate → D2.** Scope: `solver.py:37494`. Done: MIPLIB-38
  ≥ 34/38, total wall ≤ 2× H, M panel ≥ 23/24 at geomean ≤ 3.0, 0 incorrect, INF 18/18;
  the Rust MILP route passes the routing plan's §3.2 incumbent verification and
  root-bound cross-check **through P1.1's seam on the Rust engine** (the verifier is
  engine-independent; the tree bound stays labelled `rust-fp`). Then flip
  `DISCOPT_LP_MILP_BACKEND` default to `rust`, keep `=highs`, audit rows, H4 guard
  drift=0, and record M-1671 route (a) as met if D0 chose it. Kill: bar missed → **D2**.

### Phase 5 — masters (R2a/R2b)

- [ ] **P5.1 — Re-enable the Rust lazy master on the convex route behind a flag.** Scope:
  `solver.py:9916` (`_convex_route_oa_master` accepts `DISCOPT_CONVEX_ROUTE_OA_MASTER=
  simplex` again, default `highs`; `auto` stays refused), `oa.py` lazy-callback selection
  (`:6578`). Done: the §76.4 panel (94) + syn/rsyn (51, kernel ON and OFF) ON vs OFF
  interleaved: cert-clean; certified ≥ HiGHS arm, total wall ≤ 1.25×. Kill: lower certified
  count or wall > 1.25× after P4's graduated levers → flag kept OFF as state 3 (the
  documented opt-out for HiGHS-less installs), recorded in §7; D3 asks whether to continue.
  Regime: bound-changing (route-level panel).
- [ ] **P5.2 — Decomposition masters default to Rust.** Scope: `lp_backend.py:249-262`
  (`get_decomposition_master_solver` → Rust first; HiGHS when `DISCOPT_DECOMP_MASTER=
  highs`), `names_decomposition_route`. Done: P0.4's R2b table re-run: identical certified
  set, SGM ≤ 1.0, total ≤ 1.5× (counter ≥ masters × 2). Kill: SGM > 1.0 → keep HiGHS
  default when importable; record. Regime: bound-neutral per master (same optimum) —
  wall-gated.
- [ ] **P5.3 — The independent confirmation question → D3.** Scope: memo only.
  `_confirm_lazy_master_independently` (`milp_highs.py:1260`) confirms HiGHS with Rust
  (then SCIP). With Rust primary: confirm with HiGHS when installed, else with a
  Rust re-solve at a different seed and presolve off (weaker). Present the measured
  false-certificate catch rate of each option on the #2069/#2115 captured masters and
  stop at **D3**.

### Phase 6 — HiGHS optional, then removable

- [ ] **P6.1 — `discopt/_optional_highs.py` and the inventory.** Scope: move
  `HighsUnavailable`/`require_highspy` (`lp_milp_highs.py:122-135`) and
  `milp_highs.HighsBackendUnavailable` (`:91`) to one module with `MissingHighsError`
  (`ModuleNotFoundError`, `name="highspy"`, message names `pip install 'discopt[highs]'`);
  new `docs/dev/highs-optional-inventory.md` modelled on `jax-optional-inventory.md`
  listing every remaining `highspy` reach with its feature. Done: `grep -rn "import
  highspy" python/discopt` lists only sites behind `require_highspy`; smoke green. Kill:
  none. Regime: bound-neutral.
- [ ] **P6.2 — CI lane "Python without highspy" + corpus guard.** Scope:
  `.github/workflows/ci.yml` (clone of the `:780` lane), new
  `python/tests/test_highs_optional.py` mirroring `test_1873_jax_optional.py` (import
  blocker that **records the stack of every attempt**; the 74-file corpus + the M-panel
  MILPs solved in fresh interpreters). Done: lane green with `pytest -m smoke` and the
  guard reporting 0 attempts on the default path; a deliberately HiGHS-only call raises
  `MissingHighsError` naming the extra. Kill: an attempt recorded on the default path →
  that site is a remaining role; add it to §1.1 and schedule its port before proceeding.
  Regime: packaging.
- [ ] **P6.3 — Packaging flip.** Scope: `pyproject.toml:44` (remove `highspy` from
  `dependencies`), `:73` (real `highs` extra; `dev` and `all` include it), `cli.py:76`,
  docs, `CLAUDE.md` Architecture note, the R7 docstrings. Done: `pip install .` without
  the extra → `pytest -m smoke` green; `wheel_smoke.py` passes without `highspy`; release
  workflow runs the no-highspy lane. Preconditions: D1 and D2 both resolved to a Rust
  default (or the owner's narrower acceptance), P1.2–P1.4, P3.2, P5.1–P5.2 landed. Kill:
  any precondition unmet → stop; this task cannot be partially done.
- [ ] **P6.4 — Retire HiGHS-only flags and dead paths (routing plan §8).** Scope: the
  nine route-only flags listed in §1.1, `_cross_check_presolve` (both files),
  `MILP_PRESOLVE_RULE_OFF`, `_root_cuts` HiGHS branch, `_lp_dual` HiGHS rungs,
  `_iis_milp` HiGHS engine — **only** if D4 chooses removal; under "kept as opt-out" they
  stay with a state-3 docstring each. Done: audit rows updated (retired or state 3);
  `cargo test`, smoke, adversarial green; `flag_audit_scan.py` shows no default-OFF
  solver-math gate without a state. Kill: none.
- [ ] **P6.5 — M-1671 closure record** *(may run earlier, as soon as D0's route is met)*.
  Scope: §4's five checks; `test_1667_…`, `test_1671_…` re-scoped; closing comment
  drafted here (not posted): "Closed on discopt's side: no default path runs HiGHS
  presolve since <commit>; the UC witness certifies 450623.7692 on the default route in
  <t> s with the Rust driver's deadline polling; the sparsify-off workaround remains
  documented on the opt-in `DISCOPT_LP_MILP_BACKEND=highs` route; re-measure upstream on a
  release ≥ <version> and drop the bit there." Plus the #1634 disposition per §4. Done:
  the five checks pass in CI. Kill: none.
- [ ] **P6.6 — Final gate → D4.** Re-run every §1.3 panel at the final commit, fill §6,
  and stop for D4.

---

## §6 Status

| Task | State | Result (one line) | Commit |
|---|---|---|---|
| P0.1 | [x] | L 25 / M 24 / INF 16 manifest at HiGHS v1.15.1; Done run executed 195/195, incorrect 0; route marker, per-panel counter, load gate, rotation added | `69ead3f` |
| P0.2 | [x] | 38/38 recovered from the A14 artifact, validated against `.solu` (0 mismatches; obra needs > 60 s); Done run executed 9, incorrect 0 | see git log |
| P0.3 | [x] | 927 executed, 0 incorrect; walls indicative (gate open on 41/103 rows); R/H geomean 1.41 (L) / 1.73 (M) / 1.78 (MIPLIB) but H/H0 = 110× on L | see git log |
| P0.4 | [ ] | | |
| P0.5 | [ ] | | |
| P0.6 | [ ] | | |
| P0.7 / D0 | [ ] | | |
| P1.1–P1.4 | [ ] | | |
| P2.1–P2.6 / D1 | [ ] | | |
| P3.1–P3.3 | [ ] | | |
| P4.1–P4.9 / D2 | [ ] | | |
| P5.1–P5.3 / D3 | [ ] | | |
| P6.1–P6.6 / D4 | [ ] | | |

Baselines (filled by P0.3/P0.4/P0.7; walls carry the load-gate verdict):

| Panel | Arm | solved/certified | total wall | median (solved) | nodes | incorrect | load gate |
|---|---|---|---|---|---|---|---|
| L (25) | R / H / H0 | 24 / 25 / 25 (cert 24 / 25 / —) | 47.42 / 21.15 / 0.32 s | 0.205 / 0.197 / 0.002 s | — | 0 / 0 / 0 | 14/25 valid → indicative |
| M (24) | R / H / H0 | 18 / 24 / 24 (cert 18 / 24 / —) | 119.92 / 32.54 / 4.90 s | 0.255 / 0.424 / 0.038 s | 117,596 / 518 / 328 | 0 / 0 / 0 | 4/24 valid → indicative |
| INF (16) | R / H / H0 | 16 / 16 / 16 verdicts (cert 15 / 15 / —; `gas11` unbounded uncertified) | 23.25 / 5.39 / 0.38 s | 0.192 / 0.264 / 0.003 s | — | 0 / 0 / 0 | 11/16 valid → indicative |
| MIPLIB-38 | R / H / H0 | 14 / 35 / 35 (cert 14 / 35 / —) | 344.89 / 253.17 / 171.04 s | 2.023 / 3.602 / 1.358 s | 708,937 / 14,015 / 18,685 | 0 / 0 / 0 | 12/38 valid → indicative |
| UC witness (P0.7) | R @ 20/60/120 s | | | | | | |

---

## §7 Falsification log

Format (CLAUDE.md §4, §11; house style `performance-plan.md` §6): one entry per
falsified hypothesis or retracted claim, newest last.

```
**<YYYY-MM-DD> — <task id>: <one-line claim that died>.** Commit `<hash>`, load <x>,
instances <names/count>, executed comparisons <n>. Measured: <numbers>. Expected:
<numbers>. Consequence: <retired flag+impl | re-scoped task | threshold replaced |
decision point raised>. Retraction of: <earlier statement, if any>.
```

**2026-11-04 — P0.1: "the L/M/INF panels are 24/24/18 = 66 instances".** Commit
`2224cef` + P0.1 harness, load ≤ 1.6, instances: all 100 `.mps`/`.lp` files of HiGHS
`check/instances` at `v1.15.1` (`04024d70`), executed comparisons 195. Measured: L **25**
(net +1: `perold`, absent from the old tree's file list in the old manifest), M **24**
(the old `M-HiGHS` 23 + `bound_implied`; no member lost), INF **16** (15 infeasible + 1
unbounded, `gas11`). Expected: 24/24/18 (routing plan §12, screened at an older HiGHS
checkout that is no longer on disk; its exclusion list was not recorded, so the 2-instance
INF difference cannot be attributed). Consequence: threshold replaced — P0.1's Done count
is 65·3 = 195 (every member × arm, met) instead of 66·3 = 198; P0.3's `executed` bound and
the §1.3/§6 panel sizes now read 25/24/16. The `M-HiGHS` panel label is renamed `M`.

**2026-11-04 — P0.1: "every R/H row is answered by the arm's engine".** Same run.
Measured: `gams10am` (INF) is proved infeasible by root presolve (`termination =
"presolve"`, separable-quadratic bound tightening of an equality row) on **both** the R and
the H arm, so neither LP engine runs. Consequence: the route marker has a third value,
`presolve`; such a row keeps its oracle verdict check and is counted
(`presolve_answered`), but is never attributed to an engine. Any parity statistic over
INF compares 15 engine rows, not 16.

**2026-11-04 — P0.3: §1.3's "HiGHS today" / "Rust today" figures.** Solver `dcdac02`,
4-core host, load gate open on 41/103 instances (walls indicative), executed 927.
Measured: HiGHS route (H) L 25/25 21.15 s, M 24/24 32.54 s, MIPLIB 35/38 253.17 s; raw
HiGHS (H0) MIPLIB 35/38 171.04 s; Rust L 24/25, M 18/24, MIPLIB 14/38. Expected (14-core
dev machine, earlier commits): H L 3.16 s, M 5.35 s, MIPLIB 38/38 72 s; Rust L 21/24, M
19/24, MIPLIB 21/38. Consequence: §1.3's columns replaced by the P0.3 numbers; the parity
*bars* are unchanged, but whether "wall ratio" means against H or H0 is put to the owner
(they differ by ~100× on LP here). Retraction of: none — the earlier figures were
measured on a different host and commit; they are superseded, not wrong.

---

## §8 Facts in this plan that contradict the brief it was written from

Recorded so the next reader does not re-derive them (verified at `1fe78b1`):

1. **R2 is not "Rust already the default".** The convex-MINLP route's OA/LP-NLP-BB master
   is *always* HiGHS (`solver.py:9916`; the Rust `auto` arm was retired 2026-10-04), and
   Benders/GBD/Lagrangian masters pick HiGHS whenever it imports (`lp_backend.py:249`).
   Only the GDP native master and AMP default to Rust. R2 is therefore a default-path role.
2. **R3 is a default-path MINLP role** (`DISCOPT_NLPBB_ROOT_CUTS` default ON since
   2026-08-20), and its port is **bound-changing**, not bound-neutral: GMI cuts depend on
   the optimal basis, which differs between engines on degenerate LPs.
3. **R5 (IIS) is opt-in** (`include_integrality=False` default), so it is not on the
   path to M-1671 and is scheduled after the LP engine, not before.
4. **#1229 is not live.** Root cause fixed in `dual_slack_basis` and `dual.rs::
   select_leaving` (routing plan §12, 2026-09-15); `test_issue_1229_free_column_sentinel.py`
   pins it. The 2122 false `infeasible` (`fbbt_row`) is also fixed. Nothing to schedule.
5. **A sixth role (R6) exists on the MINLP path**: `solve_lp_std` arbitrates POUNCE
   infeasibility labels (#1537), rescues linear nodes (#1611) and certifies free-column
   node LPs (#1869), with a Rust fallback already present for #1537.
6. **The L panel and `lp_vs_highs_geomean` are unmeasurable today**: `ref/HiGHS/check/
   instances` is absent on this machine and no Netlib loader exists
   (`run_benchmarks.py:853`); the checked-in manifest holds only the M panel.
7. **`DISCOPT_MILP_CMIR` is default ON** (graduated on introduction,
   `flag-retirement-audit.md:416`), so the MIPLIB-38 baseline of `milp-competitiveness-plan.md`
   §4 predates two graduated levers (c-MIR, NS projection) and is stale — P0.3 replaces it.
8. **The in-repo corpus is 74 `.nl` files**, not 66 (CLAUDE.md is stale on this).
9. **The Rust driver already confirms HiGHS lazy-master claims** (#2069,
   `milp_highs.py:1260`), which is the precedent P5.3's D3 question turns around.
