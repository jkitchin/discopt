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
diving, RINS (`DISCOPT_RINS`), RENS (`DISCOPT_RENS`, retired 2026-11-05 by P4.2: failed bar 2a on `dcmulti`),
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
  (`ff3da5b`); this tree synced to it at `07f71bbd` after P0.3's run. 2026-11-05: synced
  to `origin/main` `8ebd87bf` by **merge** (`e776577d`), not rebase -- the branch already
  carries merge commits from those syncs, and `git rebase` replays every commit they
  brought in (it stopped on a #2143 commit that is not on `main`). Conflicts resolved to
  `main`'s side except this branch's own audit rows and panel entries; rebuilt; `cargo
  test` 981/0, smoke 4484 passed, adversarial 19 passed. One P0.6 IIS re-pinned (below).
  2026-11-05 (later, owner request): merged `origin/main` `db232311` (#2143 landed on
  `main`, #2149/#2159/#2160/#2161) at `fe28c7bc`; conflicts resolved to `main`'s side
  plus this branch's two audit rows; rebuilt; `cargo test` 984/0, smoke 4506, adversarial
  19, audit/inventory/parity/route tests 530 passed. 2026-11-05 (third, owner request
  "rebase first, then 4.0"): merged `origin/main` `a0b3b5e1` (#2158/#2163-#2166) at
  `0912f6c6` -- by merge, for the reason above; one conflict, `run_milp_hooked` in
  `lp_bindings.rs`, kept `main`'s ReLU-separator guard and this branch's shared
  `milp_options` builder (no `MilpOptions` field differs). `cargo test` 986/0, smoke
  4518, adversarial 19, parity/IIS/audit tests 618 passed (the one failure was P4.0's
  new gate, marker updated). `cargo clippy -D warnings` is clean on `discopt-core`;
  `discopt-python` has 11 pre-existing lints in files neither side touched
  (`expr_bindings.rs`, `bnb_bindings.rs`, `convex_bindings.rs`, `decomp_bindings.rs`).
  2026-11-05 (fourth, owner request "rebase on origin, and continue with P4.2"): merged
  `origin/main` `e39975e1` (#2162, #2169) at `13880cbc`; one conflict (audit history
  lines, both kept; the marker stays 49). It brought in #2142's fixture that exposed the
  exact-Farkas false `Infeasible` (fixed, P3.2 retraction). Next due 2026-11-06.
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
| **D0** (M-1671) | P0.7 | Which of §4's three closure routes for #1671 to take, given the P0.3/P0.7 numbers. **Decided 2026-11-04: (a).** |
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
- [x] **P0.4 — Master baselines (R2a/R2b/R2c).** Scope: new
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
  **Result (2026-11-04): the kill fired once, was fixed, and the re-run is clean.**
  Harness `discopt_benchmarks/scripts/master_parity_panel.py` (`fixtures`, `capture-decomp`,
  `capture-gdp`, `replay`, `recheck`, `route`, `decomp-e2e`). `DISCOPT_DECOMP_MASTER` is
  in `lp_backend.py:get_decomposition_master_solver` (test-only lever, default
  unchanged, a typo raises). Data: `docs/dev/data/highs-parity/masters-67530f14/`
  (pre-fix) and `masters-6351f589/` (post-fix). (Shrunk 2026-11-05 for `main`'s
  `test_docs_dev_data_size` guard: `replay.json` gzipped to `replay.json.gz`, the raw
  `replay.log` cut to its last 300 lines, `replay_tail.log`.)
  - **Kill fired (pre-fix replay, 3050 masters, 6100 compared, 1 contradiction):**
    `milp_highs.solve_milp` certified OPTIMAL with bound 41573.26 on
    `milp_default_box_clay0203_master` against the 41420.272 the Rust engine verifies
    to 1e-14. Cause: HiGHS was handed the default box `±9.999e19` as a finite bound;
    both the primary solve and the #1634 cross-solve were wrong, so the cross-check
    confirmed it. Filed as jkitchin/discopt#2146 and fixed (`6351f589`): every HiGHS
    master opens the default box, as the Rust adapter already did. 999 of the 3042
    captured decomposition masters carry that box, so this is a default-path class
    (convex route + decomposition masters); the fix changed exactly 1 HiGHS answer
    out of 3050.
  - **Post-fix replay: 3103 masters (8 fixtures + 3042 decomposition + 53 GDP),
    compared 6206 (≥ 2 × 3103), contradictions 0, raised 0.** Each engine's point is
    verified by the script; every verified incumbent is held against every bound;
    infeasible/unbounded splits are checked.

    | set (n) | optimal H / S | total wall H / S | SGM wall S/H (both optimal) |
    |---|---|---|---|
    | R2a fixtures (8) | 8 / 8, identical objectives | 13.7 / 22.2 s | 0.84 |
    | R2b decomposition (3042) | 3040 / 3040 (+2 infeasible each) | 55.7 / 4.5 s | **0.08** |
    | R2c GDP, vendored Pyomo (53) | 51 / 51 (+2 infeasible each) | 3.5 / 1.0 s | 0.30 |

    Per-fixture (status / objective / bound / nodes / wall, H vs S) is in the
    commit `docs(highs-parity): P0.4` message and `replay.json`; HiGHS needs far
    fewer nodes on the hard ones (m6 749 vs 24,254; rsyn0820m 148 vs 11,540).
  - **R2a route panel, single arm (HiGHS master), 94 models:** post-fix 91/94
    certified, 332.7 s; pre-fix 92/94, 325.9 s (§76.4: 92 optimal, 339.5 s). Route
    fired 94/94, 169 bound checks, 0 violations in both. The one difference,
    `portfol_roundlot@0`, is budget-edge noise: interleaved A/B, 3 reps at 30 s,
    pre-fix 2/3 certified (one `feasible` at 30.5 s), post-fix 3/3 (20.8–21.2 s).
  - **R2b end to end (132 decomposition tests × each forced engine):** simplex
    132/132; HiGHS 131/131 with `test_benders::test_pounce_only_no_highspy`
    deselected — that test simulates a missing `highspy`, so forcing `highs` makes it
    raise by design. Recorded test-isolation hazard: when it fails, its
    `sys.modules` surgery (it deletes every module named `*highs*`) leaves the session
    such that `test_1611_lagrangian::test_requested_lagrangian_is_not_replaced_by_bnb_fallback`
    (iteration-capped Kelley, HiGHS tie-breaking sensitive) returns 323.67 instead of
    331.6; it passes alone and after any other decomposition file. The capture
    plugin's run showed the same 323.67. **Retraction:** I first attributed that
    failure to CPU contention; the loop is capped by iterations, not wall clock.
  - **Not measured:** the 17 GDPlib PR58 instances (`gdplib` must be installed from
    source; that install was not permitted in this session). R2b's G3.2 finding that
    HiGHS wins *large* GDP masters is therefore untested here; the captured masters
    are small, which is where Rust wins.
- [x] **P0.5 — Fixture capture for R3/R4/R6.** Scope: new `python/tests/data/highs_parity/`
  + a capture script. Instrument (temporary, not committed) `_root_cuts._solve_lp`,
  `convex_node_bound._lp_dual` and `solve_lp_std`'s MINLP-path callers to dump each LP
  (`StdForm` arrays) with HiGHS's `x`, `y`, status, and for R3 the GMI cuts emitted, over
  the convex NLP-BB subset of the 74-file corpus. Keep ≥ 50 R4 node LPs, ≥ 20 R3 root LPs
  (with cuts), every #1537/#1611 arbitration LP reached. Done: fixtures ≤ 5 MB total,
  loader test asserts shapes and that HiGHS re-solve reproduces the stored objective to
  1e-9. Kill: none (capture). Regime: measurement.
  **Result (2026-11-04):** `discopt_benchmarks/scripts/capture_highs_parity_fixtures.py`
  wraps the HiGHS calls at runtime (no solver edit; the plan's "temporary instrumentation"
  lives in the script) and captured **4,447 LPs**: R3 181, R4 3,997, R6 269. Three passes:
  the 132-file in-repo corpus by default, the corpus again with
  `DISCOPT_CONVEX_KERNEL=0 DISCOPT_CONVEX_MINLP_ROUTE=0` (label `@nlpbb`: by default
  the kernel/route answer convex instances first and only 3 root LPs were reached), and
  the #1537/#1611/#1869 regression files under pytest. Fixtures in
  `python/tests/data/highs_parity/`, **1.31 MB**: R3 30 (all with GMI cuts, 9
  instances), R4 60 (round-robin over instances), R6 178 (#1537 arbitration 94, #1869
  free-column 81, #1611 rescue 3; de-duplicated; verdict-only LPs carry the status HiGHS
  must reproduce), plus **5 hard R6 LPs** under `hard/`. Plain HiGHS (presolve off)
  can't reproduce those: `Unknown` ×2, `Infeasible` against the route's point (feasible
  to 1.2e-7) ×1, objective off by ~5e-8 relative ×2. All are nvs16 #1869 LPs with
  columns boxed at 1.6e9, i.e. the inputs P1.2 must survive.
  `test_highs_parity_p05_fixtures.py`: 274 tests, 1.5 s total, each fixture's shapes,
  and HiGHS reproducing its objective to 1e-9 relative (or its verdict); the hard ones
  hold the route's point to 1e-6 row-scaled. Measurement note: `_lp_dual`'s returned
  value excludes the slack form's `obj_const` (100/100 sampled LPs differ by exactly it);
  the fixtures store the row form's objective, constant included.
- [x] **P0.6 — IIS baseline.** Scope: new `discopt_benchmarks/scripts/iis_parity_panel.py`.
  Run `compute_iis(include_integrality=True)` on every IIS-exercising test model in
  `python/tests/` (grep `compute_iis`/`milp_iis`) and the MIPcc27 smallset **if** on disk
  (else record its absence), storing element sets, `n_solves`, `proven_irreducible`,
  wall. Done: ≥ 20 fixtures recorded; a re-run reproduces identical element sets
  (determinism check, counter printed). Kill: non-deterministic sets between two HiGHS
  runs on the same fixture → record; that fixture's parity bar becomes "within the
  observed HiGHS spread" and is listed. Regime: measurement.
  **Result (2026-11-04):** `discopt_benchmarks/scripts/iis_parity_panel.py` (`capture`,
  `run`). The 14 IIS-exercising test files handed `compute_iis` 75 models: 1 non-linear (no
  MPS, skipped), 56 distinct linear models, of which 3 are feasible (those tests check the
  refusal) and are listed as excluded. **53 fixtures** in
  `python/tests/data/highs_parity/iis/` (244 KB, MPS + `expected.json`), 38 with
  integers, 7 whose IIS keeps integrality elements. Two fresh-process runs each:
  **deterministic 53/53** (identical constraint/bound/integrality sets), all
  `proven_irreducible` and `complete`, 0 confirm fallbacks, 1,058 solves, 1.30 s total
  (max 0.16 s, n ≤ 62). MIPcc27 smallset: **absent on this host** (`~/Dropbox/…` does not
  exist), recorded in the run JSON; the panel is therefore all small test models, and
  the plan's "wall ≤ 2×" bar for R5 will be judged at sub-second scale. Kill not fired.
  `test_highs_parity_p06_iis_baseline.py` pins every set exactly (54 tests, 1.5 s); data
  `docs/dev/data/highs-parity/iis-3e5dbfa3/`.
  **Re-pinned 2026-11-05 (sync to `main` `8ebd87bf`):** `6e890b49289ca18d`. #2145 (IIS
  exact-reduction second route) returns rows {0,1,2} + bounds {x0 upper, x2 lower} + x3
  integrality instead of rows {0,1} + {x0 upper, x1 lower, x2 lower} + x3 integrality.
  Both checked independently in raw HiGHS on the `from_nl` model: infeasible, and feasible
  after dropping any single row, bound or integrality, so both are irreducible; IIS are
  not unique. Reproduced in two fresh processes; the old set is kept in the entry's
  `rebaselined` field. The other 52 unchanged.
- [x] **P0.7 — Entry experiment for M-1671: the UC witness on the Rust driver.** Scope:
  `scripts`-free: run `test_1667_…::test_unit_commitment_returns_within_its_time_limit`'s
  `UC_SCRIPT` with `DISCOPT_LP_MILP_BACKEND=rust` at `time_limit` 20/60/120 s, 3 reps,
  fresh subprocess each, load gate; also with `presolve=False` and with
  `DISCOPT_MILP_CMIR=0` as attribution arms. Record status, objective, bound, nodes,
  wall. Done: a table in §6; Rust certifies `450623.7692` within 20 s **or** the table
  states the gap (bound/incumbent at 20/60/120 s). Kill: a Rust bound above
  `450623.7692 + 1e-6·|·|` or a wrong `optimal` → correctness defect, P0-blocking. Then
  **D0** (§3): present routes (a)/(b)/(c) of §4 with these numbers and stop.
  **Result (2026-11-04):** `discopt_benchmarks/scripts/uc_witness_rust.py` (the test's own
  `UC_SCRIPT`, solve line parametrised), fresh subprocess per solve, 3 interleaved reps,
  idle 4-core host (load 0.0 at start); data `docs/dev/data/highs-parity/uc-a937847b/`.
  **18 executed, 0 violations** (no Rust bound above 450623.7692; kill not fired).

  | arm | status (3/3) | incumbent | bound | nodes | wall (s) |
  |---|---|---|---|---|---|
  | Rust @ 20 s | `time_limit` | **none** | 447,483.14 | 3, 3, 3 | 21.4, 20.8, 20.6 |
  | Rust @ 60 s | `time_limit` | **none** | 448,231.82 / 448,369.12 | 63, 95, 95 | 61.4, **88.0, 88.2** |
  | Rust @ 120 s | `time_limit` | **none** | 448,817.54 (gap 0.40 %) | 159 ×3 | **142.2, 141.3, 147.3** |
  | Rust, `presolve=False` @ 60 s | `time_limit` | none | same two modes | 63, 95, 63 | 60.8, 86.7, 62.8 |
  | Rust, `DISCOPT_MILP_CMIR=0` @ 60 s | `time_limit` | none | same two modes | 95, 95, 63 | 85.7, 87.5, 60.3 |
  | HiGHS route @ 20 s (default) | `optimal`, certified | 450,623.7692 | 450,587.00 | 7 ×3 | 9.9, 10.3, 10.5 |

  - **Rust does not certify the witness in 20 s, nor in 120 s, and finds no incumbent
    at all.** Its bound rises 447,483 → 448,818 (gap 0.40 % at 120 s).
  - **What actually runs:** every Rust row's route is `milp-tree: discopt MILP branch and
    bound (milp_backend='rust')`. The monolithic Rust driver exhausts its budget with
    no incumbent and **defers** (`_solve_milp_simplex` returns `None`, handing back its
    bound; `solver.py` ~38880), and the Python MILP tree runs after it. So the
    "Rust route" on this class is driver + Python tree.
  - **Wall time is not bounded by the limit:** +18–23 % at 120 s, and up to +47 % at
    60 s (the 95-node runs). That fails §4 close condition 4 ("return within the
    limit + the certificate overhead") as the route stands today.
  - **Attribution:** `presolve=False` and c-MIR off show the same two modes (63/95
    nodes) as the default. Neither is the lever; the bimodality follows wall-clock
    timing, not the option.
  - Kill not fired; nothing to fix before D0.

  **D0 decision memo — the loop stops here for the owner.** Close condition for #1671:
  *no default discopt path depends on HiGHS presolve.* The routes, with these numbers:
  - **(a) Rust default for every pure MILP** (the full P4 track): today Rust certifies
    14/38 MIPLIB, 18/24 M, and **0/1 on the witness** (no incumbent in 120 s). Far; it
    waits on D2.
  - **(b) Rust default for a measured sub-class containing the witness**: not viable
    yet. The witness is exactly where Rust fails worst (no primal at all), so no general
    predicate can put it on the Rust side today. It would first need a primal
    heuristic (P4.6 feasibility jump/shifting) and the deadline fix below; it then needs
    its own entry experiment.
  - **(c) `presolve=off` on the default HiGHS MILP route**: closes #1671 *and* makes
    #1634 moot at once, at HiGHS's measured cost (no_presolve 36/38 at 1.7× wall vs
    38/38 in `milp-competitiveness-plan.md` §1c; not yet re-measured on this host).
    Loud, not an approximation, but a default-path slowdown.
  - **Interim guard (ask, don't presume):** a subprocess hard-kill watchdog around the
    HiGHS `run()` bounds wall time but leaves the dependence; it is not a closure.
  - **Independent of D0** (recommend scheduling either way): the Rust route overruns its
    time limit by up to 47 % because the Python tree runs after the driver's deferral.
    That is a correctness-adjacent contract bug (`time_limit`), small and general.

  **D0 DECIDED (owner, 2026-11-04): route (a)** — the Rust engine becomes the default
  for every pure MILP, via the full P4 track (gated by D2); and "fix the time-limit
  overrun, then continue". #1671 closes when P4.9/D2 flips the default (P6.5 drafts the
  closing record). Route (c) is not taken; the HiGHS route stays the default, presolve
  on, until then.
  **Time-limit overrun — fixed (2026-11-04).** Cause, measured with a per-call probe on the
  witness at TL 30: the Rust driver ran 11 s and deferred; the Python MILP tree then
  checked the deadline only once per loop iteration, and an iteration solves a whole
  batch of ~1 s node LPs. Node LPs started at 30.3, 31.6, 32.5, 33.2 and 33.7 s; wall
  34.3–38.5 s. Fix (`_solve_milp_bb`): the deadline is checked before every node after
  the first; nodes not reached are handed back with `tree.requeue_nodes` (pop-time bound
  kept, never pruned). Witness at TL 30 after the fix: 30.8 / 31.2 s, same bound
  (448,108.54). Regression test `test_milp_tree_time_limit_per_node.py` (fails before: 7
  tree node solves started after a 3 s limit; passes after).

  **Also for the owner (raised by P0.3/P0.4):** (1) does §1.3's "wall ratio ≤ 3.0" mean
  against the HiGHS route (H) or raw HiGHS (H0)? On LP they differ ~100× here (Rust/H
  1.41×, Rust/H0 ≈ 155×). (2) GDPlib's 17 instances need `gdplib` installed from source
  (`pip install "gdplib @ git+https://github.com/SECQUOIA/gdplib.git"`), which this
  session was not permitted to do; R2b's "large GDP masters" question is untested
  until then.

### Phase 1 — bound-neutral ports and the engine seam (R6, R4, R3)

- [x] **P1.1 — Engine seam for the verified LP route.** Scope: `lp_milp_highs.py`
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
  **Result (2026-11-04):** entry check first: the ladder helpers (`readback_problem` …
  `exact_ns_bound`, lines 227–1010) touch **no** HiGHS object; only `solve_lp_std`'s raw
  solve, `phase1_infeasibility_proof` and `recession_ray` do. Hypothesis confirmed, no
  helper to list. New `python/discopt/solvers/lp_std_engine.py`: `LpEngine` protocol,
  `HighsEngine` (today's option lists and calls, rays fetched lazily) and `RustEngine`
  (`solve_lp_warm_csc_py` on `A x = b` directly; no presolve, so `solve_lp_std` starts at
  the presolve-free step; its `numerical` exit maps to `undecided` and reaches the
  phase-1 proof, as in `lp_simplex`). `solve_lp_std`, `phase1_infeasibility_proof` and
  `recession_ray` take `engine=` (`None` reads `DISCOPT_LP_STD_ENGINE`, default
  `highs`; audit row added as a documented opt-out until P2.6/D1). The Rust engine sets
  `lp/std_engine_rust`, never the HiGHS route marker.
  - **HiGHS engine, exact:** the named files pass unchanged (96 tests); old vs new
    `solve_lp_std` on the 37 L/INF standard forms built by `_highs_std_form`: **37/37
    identical in every field** (status, objective repr, bound, certificate, labels,
    message, `highs_status`, stats). Counter 37 ≥ 24.
  - **Rust engine, measured** (same 37): L **24/25 certified** (P0.3 Rust route: 24;
    `perold` `feasible`, the NS bound is −∞ for its dual on every rung, as on the old
    route), **0 wrong answers**; INF **12/12 verdicts proved** (11 Farkas, `gas11`
    unbounded) — the kill (a Farkas ray HiGHS has and Rust cannot supply) did not fire.
    Provenance: every `optimal` is certified by a rung of the same ladder, 13 by rung 1
    (NS over the declared box, unlabelled by design — HiGHS's `blending`/`shell` are
    unlabelled too) and the rest by exact dual correction; on `standata` the Rust dual
    certifies on rung 1 where HiGHS's needed the FBBT box.
  - Script `discopt_benchmarks/scripts/lp_engine_seam_check.py`; data
    `docs/dev/data/highs-parity/p11-engine-seam/`. Test
    `test_highs_parity_p11_lp_engine_seam.py` (5 tests; all fail before the seam).
- [x] **P1.2 — R6: MINLP-path verified LPs on the Rust engine, HiGHS as fallback.**
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
  **Result (2026-11-04): the kill fired for two of the three callers; one moved.** Entry
  experiment first: P0.5's 183 R6 fixtures through both engines (`p12-rust-first/
  entry_verdicts.json`), verdict = certified infeasible / certified bound / none:
  - **#1611 rescue: 3/3 identical** → `_linear_node_lp_rescue` now calls
    `lp_milp_highs.solve_lp_std_rust_first` (Rust, then HiGHS on anything not decisive:
    the plan said "only on Numerical/IterLimit"; any non-decisive Rust answer falls back
    instead, which can only add a proof).
  - **#1537 arbitration: 93/94 identical; 1 differs** (`nvs0`, scaled): HiGHS proves the
    node LP empty with a verified Farkas ray, Rust returns a point feasible to 4.0e-7 on
    rows with |b| up to 1.2e9 (≈ 3e-16 relative), certified optimal. Both certificates
    are valid at their tolerance. The arbitration needs the emptiness proof, so it
    **stays HiGHS-first** (kill).
  - **#1869 free-column certificate: 61/85 identical; Rust loses the bound on 23 and
    certifies a looser one on 1** (22 of the 24 are nvs16's LPs with columns boxed at
    1.6e9, plus `ex8_5_3@nlpbb` and two `test_1537` scalings; the hard ones of P0.5).
    **Stays HiGHS-first** (kill); these LPs go to P2.1's profile list.
  - Corpus: the 132-file corpus at 20 s with a call counter reaches the rescue **0
    times** (P0.5's capture: also 0 at 10 s, default and `@nlpbb`), so no corpus drift
    is possible from this change; the probe fires (3 rescue calls, all Rust-decisive,
    under `test_issue_1611_remainder.py`). Regression files #1537/#1611/#1869 pass (56).
    Test `test_highs_parity_p12_rust_first_rescue.py` (5; all fail before).
- [x] **P1.3 — R4: Rust dual first in `_lp_dual`.** **KILLED 2026-11-04 (see result).** Scope: `convex_node_bound.py:215-260`.
  Reorder to `_simplex_dual` → HiGHS → row-equilibrated HiGHS; `_simplex_dual` gains the
  slack-form `sf` input so all three solve the same LP. Done: on the P0.5 R4 fixtures,
  `certify_lp_dual(y_rust)` ≥ `certify_lp_dual(y_highs)` − (CERT_ABS + CERT_REL·|v|) on
  every LP and no `None` where HiGHS gave a bound (counter ≥ 50); convex corpus subset
  (NLP-BB instances) at 60 s, `deterministic=True`: node_count and objective identical to
  `main` on every instance; `test_2067*` passes. Kill: ≥ 1 fixture where the Rust dual
  loses a bound HiGHS found → keep HiGHS first *for that LP class* (state the class),
  record in §7, attach the LP to P2.1. Regime: bound-neutral with the §1.4 R4 exception.
  **Result (2026-11-04): kill fired at the entry experiment; no code shipped.** P0.5's 60
  R4 node LPs, each solved by both engines of the P1.1 seam (presolve off, as `_lp_dual`
  does) and both duals certified by `certify_lp_dual` on the same slack form
  (`p13-r4-dual/`): **0 bounds lost, 55 within tolerance, but 3 looser than the bar** —
  Rust's certified bound is 1.0–1.25e-6 below HiGHS's at |v| ≈ 1.5e-6, against a
  tolerance of CERT_ABS + CERT_REL·|v| ≈ 1.0e-6. All 3 are node LPs of the translated
  `clay0303hfsg` (`test_1537_invariance` `shift` cell). Rust is also **strictly tighter on
  2** LPs, so the reorder is not bound-neutral in either direction. One instance does not
  define a general LP class to carve out (CLAUDE.md §2), so R4 **stays HiGHS-first**
  (`_lp_dual` unchanged; the Rust simplex remains its third source). The 3 LPs join
  P2.1's profile list; re-open after P2 improves the Rust dual (P2.2–P2.5).
- [x] **P1.4 — R3: root-cut LP and GMI on the Rust simplex (flagged).** Scope:
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
  **Result (2026-11-04): cert-clean, not non-inferior → state 3.** `gomory_cuts_py` takes
  the standard-form vertex `z = [x, s]` and returns `coeffs·z ≥ rhs`; with `s = b_le −
  A_le x` that is projected exactly to the x-space `≤` row the HiGHS separator emits, then
  filtered identically (norm, dynamism 1e6, rhs safety 1e-9 relative, violation 1e-6).
  - **Entry** (P0.5's 30 R3 fixtures, one GMI round per engine, gain re-measured by a
    HiGHS re-solve with the cuts): on the 14 fixtures where HiGHS's round gains, the Rust
    round closes ≥ 90 % of it on **13 (92.9 %)**; the miss is `syn05m` (3.98 vs 25.02);
    Rust gains more on `m3` (0.25 vs 0) and `flay02m` (0.033 vs 0.0007).
  - **Population:** the §21 panel's 153 instances no longer describe where the stage
    runs: P0.5's capture shows it fires on **2** corpus instances by default (the convex
    kernel/route answer the rest first) and 9 with them off (`@nlpbb`). Panel = those 11
    instance-configurations (`p14-root-cuts/population.json`), 3 reps, 60 s, arms in
    their own subprocesses, interleaved; script `root_cuts_engine_panel.py`.
  - **Panel:** 66 executed, **414 bound/feasibility checks, 0 violations**, identical
    certified set (10 certified in both, `portfol_roundlot@nlpbb` uncertified in both).
    Bounds tighter-or-equal (1e-6 rel) on **11/11**, never looser; total nodes **0.999×**
    (209,296 → 209,087); total wall **1.078×** (246.4 → 265.7 s), above the 1.05× bar,
    almost all `tls2@nlpbb` (48.3 → 65.4 s, 137 → 181–189 nodes).
  - **Outcome (the task's state-3 clause):** `DISCOPT_NLPBB_ROOT_CUTS_ENGINE` stays
    `highs` on HiGHS installs; **unset with no highspy it now selects `rust`**: such an
    install used to get no root cuts at all (#1514). Audit row added. The 781 file's
    exact-enumeration soundness test passes on the Rust arm (0 unsound cuts); its two
    HiGHS-plumbing tests (`…deadline_reaches_highs…`, the total-stall-rounds count) are
    HiGHS-arm properties: the Rust deadline is pinned in
    `test_highs_parity_p14_root_cuts_engine.py`, and the stall test's arithmetic assumes
    non-improving rounds come last (Rust: 66 rounds, 63 improving, one mid-run).

### Phase 2 — the LP engine (R1-LP)

- [x] **P2.1 — Attribution of the L-panel losses.** Scope: `scratchpad`-style profile under
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
  **Result (2026-11-04): the loss is Python, not the simplex — the >50 % clause fired.**
  Script `discopt_benchmarks/scripts/lp_route_profile.py` (each Rust LP call timed from
  Python, its `DISCOPT_PROFILE` phase dump parsed), per-function split by `cProfile`;
  data `docs/dev/data/highs-parity/p2.1/`. The 13 rows (`perold` + the 12 largest R/H
  wall ratios of P0.3; the plan's "3 R failures" were pre-P0.3 names):

  | instance | Rust route wall (s) | raw HiGHS | iters Rust / HiGHS | python (bucket) | Rust LP phases | root presolve, R only (s) | cert re-check R / H (s) | declared-box tightening R / H (s) |
  |---|---|---|---|---|---|---|---|---|
  | `scrs8` | 7.01 | 13 ms | 902 / 361 | **99%** | 1% | 4.84 | 1.93 / 2.08 | 1.09 / 1.12 |
  | `stair` | 5.39 | 14 ms | 1293 / 291 | **99%** | 1% | 3.93 | 0.92 / 0.95 | 0.61 / 0.61 |
  | `etamacro` | 2.18 | 9 ms | 1518 / 419 | **97%** | 3% | 1.36 | 1.12 / 1.11 | 0.54 / 0.54 |
  | `perold` | 9.31 | 57 ms | 3684 / 1149 | **96%** | 4% | 4.73 | 0.00 / 1.39 | 1.00 / 0.89 |
  | `25fv47` | 9.87 | 160 ms | 5123 / 2583 | **93%** | 7% | 4.75 | 4.97 / 5.00 | 2.56 / 2.46 |
  | `standgub` | 2.38 | 4 ms | 118 / 34 | **100%** | 0% | 1.12 | 1.52 / 1.58 | 0.76 / 0.73 |
  | `standata` | 2.05 | 4 ms | 118 / 34 | **100%** | 0% | 0.86 | 1.37 / 1.60 | 0.78 / 0.78 |
  | `standmps` | 2.33 | 5 ms | 401 / 163 | **99%** | 1% | 0.87 | 1.78 / 1.66 | 0.95 / 0.95 |
  | `shell` | 3.34 | 14 ms | 1127 / 325 | **99%** | 1% | 0.94 | 1.24 / 1.17 | 0.79 / 0.81 |
  | `e226` | 1.11 | 8 ms | 549 / 335 | **98%** | 2% | 0.16 | 1.21 / 1.22 | 0.49 / 0.51 |
  | `egout-ac` | 0.36 | 1 ms | 50 / 0 | **99%** | 1% | 0.01 | 0.13 / 0.13 | 0.11 / 0.10 |
  | `dD2e` | 0.25 | 0 ms | 2 / 0 | **99%** | 1% | 0.00 | 0.00 / 0.00 | 0.01 / 0.01 |
  | `avgas` | 0.31 | 0 ms | 9 / 6 | **100%** | 0% | 0.00 | 0.02 / 0.02 | 0.01 / 0.01 |

  - **Every row's bucket is `python`, 93–100 % of the route's wall.** The Rust simplex
    (all its phases) is 0–7 %; it takes 2–4× HiGHS's iterations, which matters only once
    the Python overhead is gone.
  - Three pieces make up most of the Python time (columns above): **root presolve**,
    which the Rust route runs and the HiGHS route skips (`_highs_takes_pure_lp_milp`), up
    to 4.8 s; the **post-solve certificate re-check** `_withhold_unverified_certificate`,
    which rebuilds a generic affine evaluator expression by expression (`affine_of` /
    `_combine`: 491 rows, 9,461 combines on `scrs8`), up to 5.0 s **on both routes**; and
    the **declared-box nonlinear tightening** `_declared_box_tightening`, up to 2.6 s on
    a purely linear model, **on both routes**. The last two are also most of the HiGHS
    route's own overhead: the "H/H0 = 110×" of P0.3 (raw HiGHS 4–160 ms against a route
    wall of 1–4 s).
  - **Re-ranked P2 by expected gain (from this data):** **P2.0 (new, below) first** —
    every L row is ≥ 93 % Python; then P2.2 (cold-start basis: Rust needs 2–4× HiGHS's
    iterations) > P2.5 (degeneracy, same symptom) > P2.3 (hypersparse; FTRAN/BTRAN are
    ≤ 3 % of wall here) > P2.4 (LP presolve; the route already presolves in Python).
- [x] **P2.0 — Python overhead on the pure LP/MILP route (bound-neutral; inserted by
  P2.1).** Scope: `solver.py` (`_highs_takes_pure_lp_milp`, the Rust LP/MILP dispatch,
  `_declared_box_tightening`), `modeling/core.py` (`_withhold_unverified_certificate`),
  `_affine_evaluator.py`. (i) The Rust LP route skips root presolve exactly when the HiGHS
  route does. (ii) The post-solve certificate re-check builds its evaluator for a model
  that is already linear from the standard-form arrays the route used (or reuses them),
  not by recombining expressions. (iii) `_declared_box_tightening` returns at once on a
  model with no nonlinear term. Done: L/M/INF panels R and H arms, statuses,
  certificates and objectives **identical** to P0.3 on every instance (counter printed),
  and both routes' wall on the 13 rows above reduced by ≥ 50 % (median); the #1229 /
  #1537 / #1611 / #1869 files and the route tests pass. Kill: any changed status,
  certificate or bound → root-cause; a piece that cannot be made bound-neutral is dropped
  from the task and recorded. Regime: bound-neutral (H arm exact; R arm exact).
  **Progress (2026-11-04):**
  - **(i) done.** `_pure_route_presolve_skip_backends()` = `("highs", "rust")`. L/M/INF,
    Rust arm, 1 round, 20 s: **65/65 identical status, certificate and objective** to
    P0.3, 0 incorrect; Rust arm total wall **190.6 → 137.4 s (−28 %)**. Node counts
    unchanged on every certified instance; the 5 M instances whose count moved are
    time-limited/uncertified ones that now get more nodes into the same 20 s (`2122`
    127 → 247, `gesa2`, `gt2`, `p0548`, `sp150x300d`). Data `p2.0/`. Test
    `test_highs_parity_p20_rust_route_presolve.py`; `test_highs_route_skips_discopt_root_
    presolve` now uses the native route as its positive control.
  - **(iii) re-scoped by measurement:** `_declared_box_tightening` is **not** a no-op on
    linear models: on `scrs8` it tightens 677 bounds (`defined_variable_forward`,
    `separable_quadratic_upper_bound`), and it is what proves `gams10am` infeasible in
    presolve on both arms (P0.1). Skipping it is not bound-neutral, so (iii) becomes "make
    the pass cheaper with identical outputs" (standalone it costs 0.20–0.29 s on `scrs8`,
    ~1.1 s inside a solve: the extra is not import cost, warm repeat solves measure the
    same).
  - **(ii) done (2026-11-04).** The evaluator was already the #1743 affine walk, not a
    tape; its cost was `affine_of` building one scipy matrix per expression node (~0.2 ms
    each: 9,461 nodes, 1.7 s on `scrs8`). `affine_array.scalar_linear_of` reads a scalar
    `+ - neg * / **1` row over scalar variables and constants in one dict walk, and
    declines everything else to `affine_of` unchanged; `affine_constraint_rows` and
    `AffineEvaluator.try_build` try it first. Same-input probe on the 65 L/M/INF
    instances: `J`, `g0`, `c`, `c0` and row map **bit-identical** to the `affine_of` build
    (325 comparisons; 160 nonlinear corpus models declined by both), build 14.2 → 0.57 s
    in total. Panel L/M/INF R+H, 1 round, 20 s (`p2.0/scalar_affine_lmi.json`): 130/130
    status, certificate and objective identical to P0.3 (H) and P2.0 (i) (R), every
    objective **bit-identical**, node counts unchanged on every certified row, 0
    incorrect. Interleaved A/B of this piece alone, on vs off, 13 rows × 2 routes × 3
    reps (`p20_scalar_affine_ab.py`, `p2.0/scalar_affine_ab.json`, load 1.3–2.1): 130
    comparisons, 0 differences; median wall ratio **R 0.625 (sd 0.16), H 0.589 (sd
    0.15)**, totals R 18.8 → 11.9 s, H 17.7 → 10.5 s. Test
    `test_highs_parity_p20_scalar_affine.py`.
  - **Done criterion against P0.3** (13 rows, median of per-row wall ratio, this run vs
    `baseline-dcdac02`): **R 0.262, H 0.477** — both ≥ 50 % reductions. Caveat (§9): P0.3
    is a different session, not an interleaved control; the interleaved evidence is the
    two A/Bs ((i) and (ii) each against its own off arm). H's 0.477 is close to the bar,
    so (iii) is still done before P2.0 is checked off.
  - **Retracted 2026-11-04 (§11): the H figure above.** Re-measured with P0.3's own
    protocol -- 13 rows, R+H, **3 rounds**, `--load-gate 2.0`
    (`p2.0/p20_done_13rows.json`): 26/26 verdicts identical to P0.3; median per-row wall
    ratio **R 0.351** (passes) but **H 0.600** (does **not** pass the ≥ 50 % bar; per-row
    0.47–1.05, the three ~0.08 s rows unchanged). The 1-round 0.477 was an optimistic
    draw. P2.0 stays open on the H arm.
  - **Lazy `scipy.optimize` (2026-11-04).** `_dual_recovery` imported it at module load:
    0.16 s of every fresh process's first solve (`-X importtime`), on a route that never
    calls `lsq_linear`. Now imported on first use. Real for a user's first solve; **not
    visible on the panel** (its child process has imported it before the timed solve):
    13 rows, 3 rounds, H 0.611 (`p2.0/p20_done_13rows_lazy.json`). Test
    `test_highs_parity_p20_lazy_scipy_optimize.py`.
  - **(iii) done as "cheaper, identical outputs" (2026-11-04).** Three exact changes:
    `has_factorable_work` returns False at once when the objective and every row are
    scalar-linear over `Constant` coefficients (`affine_array.is_scalar_linear`; every
    scan it skips is False on that grammar -- argued per scan in the code); the
    univariate-polynomial rule caches its expansion per row instead of re-expanding every
    round; the defined-variable rule reads a bare variable leaf's enclosure from its box
    (`_leaf_interval`, the same object `evaluate_interval` returns). Identity, same
    inputs, old vs new code: `tighten_nonlinear_bounds` `lb`/`ub`/stats bit-identical on
    197 instances (L/M/INF + every in-repo `.nl`, linear and nonlinear);
    `has_factorable_work` 225/225 (49 True in both), scan total 2.95 → 1.16 s. Test
    `test_highs_parity_p20_linear_presolve_passes.py`.
  - **State after (i)–(iii), 13 rows, 3 rounds, load-gated** (`p2.0/p20_done_13rows_iii.
    json`): 26/26 verdicts identical to P0.3; median per-row wall ratio vs P0.3 **R 0.323
    (passes), H 0.556 (misses the 0.5 bar)**. What is left on the H route, warm profile of
    the median row: the declared-box pass is still the largest piece (the two rules that
    fire on linear rows, `defined_variable_forward` and `separable_quadratic_upper_bound`,
    re-scanning every row each of ~4.6 rounds), then a tail of 0.03–0.15 s items across
    subsystems (`_is_fast_linear_quadratic_family` degree walks, the affine evaluator
    build, `_withhold_default_box_certificate`, `find_functionally_dependent_names`,
    `validate`, `readonly_model_repr`). **Owner question:** the H-arm bar measures the
    HiGHS route's own Python overhead, not Rust parity; accept P2.0 at H −44 % and move
    on, or keep cutting that tail first?
  - **Closed 2026-11-05 by owner decision:** accepted at R −68 % / H −44 % (the H bar is
    the HiGHS route's own overhead, not Rust parity); the remaining tail is not pursued.
- [x] **P2.2 — Cold-start basis and dual-slack warm start for the LP route. KILLED 2026-11-05 (see result).** Scope:
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
  **Result (2026-11-05): killed at its kill bar; flag and implementation retired
  (never merged).** Entry (`p22_entry_phase_split.py`, `p2.2/entry_phase_split.json`, L
  panel, Rust route, every LP of each solve): phase 1 is **49.6 %** of the 15,549 cold
  pivots (median 41 %, up to 65 % on `perold`/`shell`), and Rust pivots 2.6× HiGHS's
  iterations, so a crash could in principle reach −30 %. The cold primal already crashes
  structural **singletons** into the start basis (`primal.rs` `run`): 70–80 % of rows
  start structural (`perold` 425/625, `25fv47` 628/820). Built: a triangular crash on top,
  kept primal feasible because phase 1 here minimizes the sum of artificials and needs
  in-box basics -- a multi-nonzero column crashed into an untouched artificial row only
  when its value and every singleton it shifts stay in their boxes, never touching
  another crashed column's row (block-triangular, so nonsingular). Measured, flag ON vs
  OFF, same panel: v1 (column must touch only artificial rows) total pivots **−0.06 %**;
  v2 (singleton rows allowed) **−3.4 %** (15,549 → 15,025; phase 1 −5.7 %), placing 1–50
  columns into 127–257 open rows (`p2.2/crash_v{1,2}_*.json`). The remaining rows have no
  column whose value lands in its box: that is the ceiling of a feasibility-preserving
  crash. Below the 10 % kill bar → retired, no code kept but the entry instrument. A
  Bixby/CPLEX crash that ignores feasibility needs a composite phase 1 that tolerates
  bound-infeasible basics -- a different task, not this hypothesis; and the wall term was
  never reachable (the whole simplex is ~35 ms of `scrs8`'s 1.2 s route wall).
- [x] **P2.3 — Hypersparse FTRAN/BTRAN and row-wise pricing. KILLED AT ENTRY 2026-11-05.** Scope: `linsolve.rs`,
  `dual.rs` (Devex pricing loop), `feral` interface. Entry: on the P2.1 instances, measure
  the density of `B⁻¹a_q` and `eᵀB⁻¹` results; hypothesis: ≥ 50 % of solves have < 10 %
  fill, so a hypersparse path pays. Flag `DISCOPT_LP_HYPERSPARSE` default OFF. Done:
  bit-identical pivot sequence (`lp_iters` golden unchanged ON vs OFF on the `cargo`
  panel) and wall −25 % on the instances where the entry held. Kill: entry < 50 % or wall
  gain < 10 % → retire. Regime: bound-neutral (exact).
  **Result (2026-11-05): killed at entry; nothing built.** Instrument: temporary
  profile-gated counters in `FeralLU::ftran`/`btran` (`linsolve.rs`; reverted) -- per
  solve, nonzeros in and out (`|v| > 1e-14`) and whether the result has < 10 % fill;
  driver `p2.3/entry_probe.py`, Rust route, 20 s. **The 13 P2.1 instances (the entry's
  population): FTRAN results < 10 % fill on 24 % of 15,168 solves, BTRAN on 21 % of
  25,415** -- below the 50 % entry bar (`p2.3/entry_fill_L13.json`). Extra look at the M
  panel, where the Rust wall actually is: 60 % / 45 % of 3.0M / 4.9M solves (pooled ~50 %,
  `p2.3/entry_fill_M.json`), the big hypersparse ones being `sp150x300d`, `issue-2446`,
  `2122` (75–96 % low-fill FTRAN); but FTRAN/BTRAN phases are 20.4 s of 112.6 s of wall on
  the 9 heaviest M instances (18 %, an over-count -- phases sum across threads), so even
  halving the low-fill half saves ~5–9 % of M wall, under the 10 % kill bar. **Side
  finding for P4.1:** the largest single phase on the M panel is **FBBT** (`gt2` 27.7 s,
  `gesa2` 14.0 s, `bell5` 9.3 s, `p0548` 8.3 s, `dcmulti` 6.8 s of phase time), ahead of
  the node LP solves.
- [x] **P2.4 — LP presolve with postsolve (row/column singletons, dominated/empty
  rows, doubleton equations), duals restored. NOT PURSUED (owner, 2026-11-05).** Scope: new `lp/simplex/lppresolve.rs`,
  `lp_std_engine.RustEngine`. Flag `DISCOPT_LP_PRESOLVE` default OFF. Entry: count
  removable rows/cols on the L panel with a scan script; hypothesis: ≥ 20 % of rows on ≥ 8
  of 24 instances. Done: ON vs OFF on L: identical objectives (1e-9), identical
  certification **through the full ladder on the postsolved duals** (NS gap ≤
  CERT_ABS + CERT_REL·|obj| on every certified instance), wall −20 % where the entry
  held; Rust unit tests cover each reduction's postsolve on crafted LPs. Kill: entry fails
  or any certificate lost → retire. Regime: bound-neutral (postsolve must be exact; a
  dual that fails the ladder is a bug, not a tolerance to widen).
  **Not pursued (owner decision 2026-11-05):** its Done bar (wall −20 % where the entry
  holds) is unreachable by construction -- after P2.0 the Rust simplex is ~10 % of an L
  instance's route wall (P2.1: 0–7 % before P2.0's Python cuts), so removing every pivot
  could not reach −20 %. No code, no flag.
- [x] **P2.5 — Degeneracy: bound-flipping ratio test + cost perturbation on the LP
  route** *(re-scoped by the owner 2026-11-05 to the exact dual correction; done)* *(≤ 2 iterations)*. Scope: `dual.rs` (`cost_perturbation_*` tests exist), the
  stall guard. Entry: per P2.1, the share of iterations spent in degenerate pivots on the
  three failures. Flag `DISCOPT_LP_DEGEN_PERTURB` default OFF. Done: the three failures
  certify in < 20 s with the ladder closing; iterations −30 % on degenerate instances;
  cert-clean L. Kill: < 10 % → retire. Regime: bound-neutral in objective (perturbation
  removed before the final verdict; the ladder certifies the unperturbed LP).
  **Entry findings (2026-11-05) -- the failure premise does not hold; awaiting owner re-scope.**
  (1) The plan's "three failures" (25fv47, shell, standmps) are pre-P0.3 names; all three
  certify on the Rust route at HEAD. The one L failure now is `perold`. (2) `perold` is
  **not a degeneracy failure**: the Rust simplex reaches the optimum (−9380.755278, equal
  to HiGHS's) and the result is withheld at the certificate (`lp/ns_bound_missing`; the
  pure Rust route reports "no NS safe bound from its dual", #1953). Through
  `solve_lp_std(engine="rust")` the full ladder fails the same way: the Rust dual's worst
  wrong-sign reduced cost on an open side is 2.9e-13 (HiGHS's: exactly 0), NS is −inf, and
  the exact dual correction over the FBBT box grows its exact-zero set 150 → 196 columns
  and stops at `EXACT_MAX_TOTAL_WORK` (4e8) in round 6; on HiGHS's dual it converges in 5
  rounds at 2.2e8. One or two steps of float refinement of the Rust dual on its own basis
  (residual 2.9e-13 → 3.4e-13 / 2.4e-13) certify on one step and not on the next: the
  instance sits at the correction's work budget, and the cost is the correction
  re-solving its whole growing set every round, not the dual's quality. (3) Degeneracy as
  such: 35 % of the L panel's Rust pivots are degenerate (`p2.2/entry_phase_split.json`;
  46–49 % on `etamacro`, `stair`) -- an iteration-only target, since the simplex is ≤ 7 %
  of the route's wall (P2.1).
  **Re-scope (owner, 2026-11-05):** fix the exact dual correction rather than degeneracy;
  the degeneracy work (bound-flipping ratio test, cost perturbation,
  `DISCOPT_LP_DEGEN_PERTURB`) is not pursued -- it could only cut iterations, and the
  simplex is ≤ 7 % of the route's wall.
  **Result (2026-11-05): `perold` certifies on the Rust route; L is 25/25 on both.** The
  cost was `_exact_solve`'s **dense** rational Gaussian elimination: every row update
  swept every column of a block made of LP columns with a few nonzeros each. Now sparse
  (row maps, the pivot in each column the remaining row with the fewest nonzeros).
  `perold`'s Rust-dual correction: dense ran out of `EXACT_MAX_TOTAL_WORK` (4e8) in round
  6, sparse needs **1.9e7** and certifies −9380.755278236511 (oracle −9380.755278235187);
  on HiGHS's dual the bound is bit-identical and the correction ~2× faster in wall. No
  flag and no cap changed: a nonsingular system has exactly one rational solution, and
  elimination finds no pivot exactly when the system is singular, so the change can only
  turn a work-cap refusal into the answer the dense form would have reached. Evidence:
  300 random rational systems (half singular by construction) identical to a dense
  reference; `solve_lp_std` on every P0.5 fixture with equality rows + L + INF, both
  engines, old vs new code: **379/380 bit-identical** (status, certificate, bound,
  objective, provenance), the one change `perold|rust` → certified. Panel L/M/INF R+H,
  1 round (`p2.5/sparse_exact_lmi.json`): 130 executed, 0 incorrect; vs the P2.0 (ii)
  panel the only verdict change from this commit is `perold` R; `bell5` H (objective
  8966635.6 → 8966413.7, both certified) and `rgn` R (2461 → 2565 nodes) moved with the
  2026-11-05 `main` merge (#2136/#2151) -- re-run 3× each with the old and the new
  `_exact_solve`: identical, and `exact_ns_bound` is never called on either. Rust
  certifies L 25/25, M 18/24, INF 15/16 (H: 25, 24, 15). Test
  `test_highs_parity_p25_sparse_exact_solve.py` (the `perold` case fails before).
- [x] **P2.6 — R1-LP graduation gate → D1.** *(D1 decided 2026-11-05: flip)* Scope: `solver.py:37494` (`_lp_milp_backend`
  split: new `DISCOPT_LP_BACKEND`, default `highs`, for the pure-LP class only;
  `DISCOPT_LP_MILP_BACKEND` keeps governing MILP), docs. Done: L panel, Rust engine via
  P1.1's seam, flags of P2.2–P2.5 ON: **24/24 certified, 0 incorrect, geomean wall ratio
  ≤ 3.0**, INF panel 18/18 with verified certificates, 3 rounds under the load gate. Then
  flip `DISCOPT_LP_BACKEND` default to `rust`, graduate each P2 flag that contributed
  (audit rows), keep `=highs` as the opt-out, and run the H4 MINLP guard (drift=0). Kill:
  bar missed → **D1**.

  **Result (2026-11-05): measured; awaiting D1.** Harness arm **S** added (the verified LP
  route with `DISCOPT_LP_STD_ENGINE=rust`, P1.1's seam; route marker `seam-rust`, pure LP
  only -- `test_seam_arm_requires_the_rust_std_engine_on_the_verified_route`). Run
  (`p2.6/gate_L_INF.json`): L (25) + INF (16), arms R/S/H/H0, **3 rounds, load gate 2.0
  (0/41 rows gated), 480 executed, 0 incorrect.** P2.2–P2.5 contributed no flag (P2.2,
  P2.3 killed; P2.4 not pursued; P2.5's fix is unflagged), so there is nothing to turn ON.

  | | R (native Rust route) | S (Rust engine via seam) | H (HiGHS route) |
  |---|---|---|---|
  | L certified, every round | **25/25** | **25/25** | 25/25 |
  | INF decided correctly | 16/16 (15 certified) | 12/12 LPs (11 certified) | 16/16 (15 certified) |
  | geomean wall vs H, L / INF | **1.07 / 0.97** | **1.05 / 1.25** | 1 |
  | geomean wall vs raw HiGHS (H0), L / INF | 84.9 / 55.1 | 83.4 / 85.6 | 79.5 / 56.7 |

  The one uncertified INF row is `gas11` (`unbounded`), identical on all three arms. Bar
  reading: against the HiGHS route the ≤ 3.0 bar holds with a wide margin on both Rust
  arms; against raw HiGHS no route passes -- the HiGHS route itself is 80× (the
  certificate ladder, verification and the modeling layer, P0.3/P2.0). Which one §1.3
  means is the owner question still open, so the gate goes to **D1**. The bar's "24/24"
  and "18/18" predate the P0.1 manifest (L has 25, INF 16).

  **D1 (owner, 2026-11-05): the §1.3 wall bar is measured against the HiGHS route →
  parity met → flip the pure-LP class to Rust.** Done in `7a851adc`, `bb29aee3`,
  `c1d9a0a3`:
  - `solver._lp_backend()`: `DISCOPT_LP_BACKEND` (`rust` default, `highs` opt-out); if
    unset, an explicitly set `DISCOPT_LP_MILP_BACKEND` still decides an LP (so every A/B
    run and opt-out keeps its meaning); MILP reads only `DISCOPT_LP_MILP_BACKEND`
    (default `highs`, unchanged). Used at the LP dispatch in `_solve_impl` (only when no
    `milp_backend` keyword is given -- an explicit keyword still decides), in AMP's and
    MIP-NLP's pure-LP paths. Arm R (native Rust route) is the default: it is what the
    gate measured and it needs no `highspy` (P6).
  - The default Rust LP route names itself (`rust-lp: discopt LP solve, Rust simplex`),
    as the HiGHS route and `native` always did.
  - AMP's pure-LP shortcut existed only for the HiGHS route; with the Rust default it
    fell to the NLP path (`iteration_limit` on `nlp_cvx_002_010`). It now hands the LP
    to whichever engine `_lp_backend()` names.
  - **H4 MINLP guard** (`lp_milp_minlp_guard.py`, now taking the env var to vary;
    `d1/minlp_guard_lp_backend.json`): `DISCOPT_LP_BACKEND` rust vs highs vs rust, 74
    `.nl` instances, **drift 0**, 7 timing-noisy (R ≠ R2) listed.
  - Gates: `-m "not slow"` full lane (20,142 passed; the 12 failures triaged: 3 were D1
    -- two HiGHS-route tests now pin the route, the AMP gap fixed; 5 are the container
    lacking `sympy`/`cyipopt`; 2 bookkeeping -- the #912 inventory entry for P0.7's
    per-node deadline (`contract`) and the audit's live-gate count, 49 on this branch
    with #2143's `DISCOPT_SUBNLP_REDUCED`; 1 the AMP test's misplaced reference check),
    smoke 4485 passed, adversarial 19 passed. Test `test_highs_parity_p26_lp_backend.py`.
  - No P2 flag contributed (P2.2/P2.3 killed, P2.4 not pursued, P2.5 unflagged), so no
    flag graduates. `DISCOPT_LP_BACKEND=highs` is an opt-out for a shipped default (out
    of scope of the flag audit).

### Phase 3 — persistent MILP handle and IIS (R5)

- [x] **P3.1 — Rust `MilpHandle`.** Scope: new `crates/discopt-python/src/milp_handle.rs`,
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
  **Result (2026-11-05): done; no MINLP-shared code touched** (`lp/simplex/*`,
  `tree_manager.rs` unchanged).
  - `discopt-core` `bnb/milp_handle.rs`: `MilpHandle` owns the equality standard form
    and edits it in place (`change_col_bounds`, `change_integrality`, `set_solution`);
    `solve` calls `solve_milp_node_hooked` -- the function `solve_milp_csc_py` runs --
    with no hooks on the current arrays, so each solve is a fresh driver run (no warm
    start carried; the handle saves marshaling, not search). `presolve_only` calls the
    new `milp_driver::milp_presolve_verdict`, which shares the driver's own entry stage
    (`prepare_entry`: #2107 noise relaxation, #1296 gate) and root stage (`root_box`:
    #1287 rounding + root presolve) with the search -- both extracted from
    `solve_milp_node_hooked` / `solve_milp_node_search` as a bound-neutral refactor
    (the two merged early `Infeasible` returns were identical structs).
  - `discopt-python` `milp_handle.rs`: the `MilpHandle` pyclass; its options come from
    `lp_bindings::milp_options`, now also what `solve_milp_csc_py` uses, so the same
    keywords give the same `MilpOptions` by construction.
  - `python/discopt/solvers/milp_handle.py`: the IIS engine's shape -- ranged rows
    `row_lo <= A x <= row_hi`, one logical per row (`A_i x - s_i = 0`,
    `s_i ∈ [row_lo_i, row_hi_i]`), so `change_row_bounds` is a logical's box and
    dropping a row is `(-inf, inf)`.
  - Evidence: Rust test `handle_after_toggles_matches_a_fresh_solve_bit_for_bit` --
    the `sparse_milp_diff` panel + 20 random toggle sequences, **156 comparisons, all
    bit-identical** (status, obj, bound, nodes, lp_iters, x; statuses Optimal /
    Infeasible / Unbounded), and `presolve_only = true` ⇒ the `presolve=true` solve is
    `Infeasible` with 0 nodes; Python `test_highs_parity_p31_milp_handle.py` -- 20
    random toggle sequences, **120 comparisons against a fresh `solve_milp_csc_py`,
    all bit-identical, and 120/120 toggled problems' verdict and optimum equal to
    HiGHS** (the ranged-row layout is the intended problem). Gates: `cargo test` 988/0,
    clippy `-D warnings` on `discopt-core` (both feature sets), smoke 4506, adversarial
    19, the 71 Rust-MILP-route test files 1101 passed.
- [x] **P3.2 — `_iis_milp._Engine` on the handle, HiGHS as the opt-in.** *(done with gaps, owner 2026-11-05: flag graduated, IIS default stays HiGHS)* Scope:
  `_iis_milp.py:111-470`; new `iis_engine="rust"|"highs"` kwarg on `milp_iis`/`compute_iis`
  (default `rust`), `_solve`'s presolve ladder mapped to `presolve_only()` →
  `solve(presolve=True)` → `solve(presolve=False)`. Done: P0.6 fixtures: identical element
  sets and `proven_irreducible` on every fixture (counter ≥ 20), wall ≤ 2× the HiGHS
  engine; `test_iis*`/`test_1634_*` pass on both engines (`test_1634_highs_parallel…`
  keeps HiGHS). Kill: a fixture whose IIS differs **and** whose Rust set is not an IIS
  (a kept element is removable, checked by one extra probe) → correctness defect; a
  differing but valid IIS on a fixture P0.6 flagged non-deterministic is accepted and
  listed. Regime: bound-neutral (verdict-identical).
  **Progress (2026-11-05): engine ported; kill criterion hit on 2/53 fixtures -- awaiting
  the owner.**
  - `discopt/_iis_backend.py`: the engine's solver calls behind one interface
    (`configure`, `set_cols`, `run(presolve, stop, max_nodes)`, `presolve_only`,
    `point`, `last_nodes`, `mip_tol`, `lp_tolerance`). `HighsBackend` is the pre-P3.2
    code moved verbatim (interrupt guard, run-clock offset, node cap, tolerance swap);
    `RustBackend` is P3.1's `MilpHandle` (+ bulk `set_col_bounds_all` /
    `set_integrality_all`, one Rust call per probe). `_iis_reduce.reduced_infeasible`
    takes `engine=` for its reduced-model solve. `iis_engine="rust"|"highs"` on
    `milp_iis` / `milp_iis_arrays` / `compute_iis`, carried into portfolio children.
  - **Refactor neutral:** with `iis_engine="highs"` P0.6's baseline passes 54/54
    element for element; the 272 `test_iis*` / `test_1634*` / mipcc tests pass (10 that
    reached into `eng.h` / HiGHS status enums now use `eng.backend.h` / backend
    statuses).
  - **Rust engine, first run:** 31/53 identical; all 22 others a strict superset of the
    baseline with a removable bound (validated independently in raw HiGHS; the baseline
    sets validate as IIS), each reported `proven_irreducible=False` -- never a false
    claim. Root cause: the driver cannot certify an LP infeasibility whose Farkas ray
    carries roundoff on a free column (`FarkasRejectOpen` x60, `LpInfeasUncertified`,
    in `lp/simplex/primal.rs::farkas_ray_certifies` -- frozen MINLP-shared code), so it
    returns `node_limit` with bound `-inf` after 1 node and the probe is undecided.
  - **Fallback outside the frozen code** (in `RustBackend`): an undecided probe with no
    integer column goes through the verified LP ladder, `solve_lp_std(engine="rust")`
    (phase-1 proof + P2.5's exact correction; certificates only). Panel
    (`iis_engine_panel.py`, 3 reps interleaved, `p3.2/iis_engine_panel.json`):
    **51/53 identical**; the 2 left (`1ec8e0a9f8f1c72c`, `97083706a8966d15`) keep a
    removable bound -- integer probes, decided by the driver's B&B whose node LPs hit
    the same frozen check. Wall: median Rust/HiGHS **1.01x**, max 4.7x on a 0.01 s
    fixture, total 1.31 s vs 1.05 s (bar ≤ 2x holds on the median and total).
  - Default stays **`"highs"`** until decided; `"rust"` is an opt-in. Tests
    `test_highs_parity_p32_iis_engine.py` (the 2 known-gap fixtures are pinned as
    "superset, still infeasible, not claimed irreducible").
  - **Owner decision needed:** (a) unfreeze `lp/simplex/primal.rs` for an exact
    Farkas correction on open columns (closes the class at its source; bound-changing
    for MINLP fathoming → its own §5 panel); (b) keep the frozen code and extend the
    backend fallback to integer probes (e.g. HiGHS-free cross-check of an undecided
    MILP probe by enumerating the driver's undecided node LPs through the ladder --
    more code, narrower); (c) accept 51/53 and flip the default with the 2 listed; or
    (d) leave `"highs"` the default and move on to P3.3.
  **Owner decision (2026-11-05): unfreeze the Farkas check.** Built
  `lp/simplex/farkas_exact.rs` (`DISCOPT_FARKAS_EXACT_OPEN`, default OFF, audit row):
  after the float verifier and the #1355 cleanup reject a ray, it is re-decided over the
  rationals (`num-rational`, new dependency) -- the reduced costs selecting an open side
  are driven to exactly zero by sparse rational elimination (P2.5's algorithm and caps),
  then the Farkas inequality is checked exactly. Rust tests: certifies the P3.2 probe's
  noisy ray the float check rejects; never certifies a feasible variant (4 rays);
  refuses an unreachable zero. **With the flag ON the Rust IIS engine returns P0.6's
  baseline on 53/53 fixtures** (`iis_engine_panel.py`; 51/53 OFF; without the
  backend's LP-ladder fallback 51/53 even ON, so both stay), median wall **0.87x**
  HiGHS, total 0.79 s vs 1.02 s; `test_exact_farkas_closes_the_gap`. Gates: `cargo
  test` 992/0, clippy `-D warnings` clean (both feature sets), smoke 4506, adversarial
  19, IIS + parity tests 639. Next: the flag's §5 corpus panel (it changes what every
  Rust LP proves infeasible, so MINLP / MILP fathoming), then -- on a pass -- graduate
  it and flip `iis_engine` to `"rust"`.
  **§5 panel for `DISCOPT_FARKAS_EXACT_OPEN` (2026-11-05).** (1) In-repo MINLP corpus,
  132 instances, ON vs OFF interleaved, 20 s, 3 jobs (`flag_onoff_corpus_panel.py`,
  `p3.2/farkas_exact_corpus_panel.jsonl`): **cert-clean** -- 0 incorrect, no bound past
  the oracle, 244 incumbent feasibility checks, no errors; certified 108 -> 107, the one
  "lost" (`ex1252a`) timing noise -- re-run 3x per arm interleaved both arms time out
  uncertified (nodes 976-1505 either way) and the exact check runs **0 times** on it;
  no gains. (2) L/M/INF on the Rust route, flag OFF vs ON (`p3.2/farkas_exact_lmi_flag
  {0,1}.json`): 130 executed, 0 incorrect, certified L 25/25, M 18/24, INF 15/16 in
  both arms, verdicts identical; node counts differ only on five time-limited
  uncertified M rows. So: sound everywhere, inert where float certification works, and
  decisive on the class it targets (the Rust IIS engine: 51/53 -> 53/53). Graduation
  is an owner call: "net-positive broadly" is met only on that class.
  **Retracted 2026-11-05 (§11): the §5 panel above and the 53/53 IIS result measured a
  partly inert implementation.** `farkas_exact_open` required `l.len() == n`; the primal
  simplex calls the verifier with its working box, which carries its `m` artificial
  columns too (`len == n + m`), so on that path the exact check returned `false` before
  doing anything (`FarkasExactAttempt` 0 on a reproduced node LP that logged 16
  `FarkasRejectOpen`). Only the dual simplex's call site reached it. Found when the
  default flip broke `test_batch_and_single_filters_both_give_certified_iis` (a node LP
  of a random MILP left uncertified). Fixed (the real columns' sides are read, as the
  float verifier does; Rust test `accepts_a_working_box_that_carries_artificials`);
  every P3.2 measurement is being re-run.
  **Re-measured with the fix (2026-11-05, `cb3e576f`):** (1) IIS panel, flag ON
  (`p3.2/iis_engine_panel_exact_v2.json`): **53/53 identical** to P0.6, median wall
  0.91x, total 0.87 s vs 1.16 s. (2) §5 corpus panel, 132 instances ON vs OFF
  (`p3.2/farkas_exact_corpus_panel_v2.jsonl`): **0 problems**, certified 107 -> 107, no
  gains or losses, 244 incumbent checks, no errors. (3) L/M/INF Rust route OFF vs ON
  (`p3.2/farkas_exact_lmi_v2_flag{0,1}.json`): 130 executed, 0 incorrect, certified L
  25 / M 18 / INF 15 both arms; node counts move only on five time-limited uncertified
  M rows. Cert-clean everywhere; net-positive only on its class (Rust IIS 51 -> 53/53,
  and the random-MILP IIS test's 131 -> 0 failed node certificates).
  **The broader IIS suite on `iis_engine="rust"`** (flag ON): 13 of 639 fail. 8 test
  HiGHS-engine internals (deadline guard, run clock, raw `Highs.run` logs, a slowed
  `Highs.run`) and would pin `iis_engine="highs"`; 5 are behaviour: the big-M
  `kedron_like` model (3 tests) -- the driver accepts an integer 6e-11 off
  (`INT_TOL` 1e-5) whose slack through an 8e6 coefficient makes a removal probe look
  feasible, the witness repair rejects it as an artifact, so the element stays
  undecided where HiGHS proves it; and pigeonhole PHP(10,9) (1 test, plus 1 variant) --
  the driver cannot prove it infeasible within a 6 s budget where HiGHS does (Phase 4
  MILP strength).
  **RETRACTION + soundness fix (2026-11-05, later): the graduated exact check issued a
  false `Infeasible`.** `main`'s #2142 fixture `captured_cstr_bigm_node_lp_stays_undecided_2142`
  (a GDPlib cstr big-M node LP that HiGHS solves; merged here at `13880cbc`) failed on
  this branch: with the flag ON the warm dual returned `Infeasible`, with `=0` it stayed
  undecided. Cause: `certify_one` certified on a bare exact `g > 0`, which proves only the
  *exact* system empty; this LP is infeasible by ~1e-8 but feasible within the solver's
  tolerance. The corpus panel above did not contain such an LP, so "cert-clean" was true
  of the panel and false of the mechanism. **Fix:** certify only when
  `g > Σ_i |y_i| tol_i + Σ_j |rc_j| tol_j` with `tol = AUDIT_FEAS (1 + |side|)` -- the row
  and bound tests of `primal::audit_feasibility` -- which proves that no point the
  solver itself would accept as feasible exists (derivation in the code). Tests: the
  #2142 fixture and a new `tolerance_feasible_system_is_not_certified_2142` both fail
  with the bare `g > 0` and pass with the margin; `cargo test` 1004/0. Re-measured: Rust
  IIS still returns P0.6's set on **53/53** (`p3.2/iis_engine_panel_exact_v3_margin.json`,
  median wall ratio 0.87), P3.2 tests 7/7. Corpus ON vs OFF with the margin
  (`p3.2/farkas_margin_corpus.*`, 20 s, 1 rep): 74 rows, 0 problems, certified 56 = 56,
  130 incumbent checks. The flag stays default ON *with the margin*;
  the claim "cert-clean" is restated as "cert-clean on the panel; the mechanism was not
  sound until the margin".
  **Closed (owner, 2026-11-05): graduate `DISCOPT_FARKAS_EXACT_OPEN`; `iis_engine` stays
  `"highs"`.** The flag is default ON (`=0` opt-out; audit row *Graduated*).
  `iis_engine="rust"` is a supported opt-in that matches HiGHS on P0.6's 53 fixtures;
  it is not the default because of two measured Rust-MILP gaps, both handed to Phase 4:
  big-M integrality slack (`INT_TOL` 1e-5 lets a 6e-11-off integer through an 8e6
  coefficient look feasible; the engine correctly refuses the artifact and leaves the
  element undecided) and pigeonhole-class infeasibility proofs within a short budget.
  The bar's "default `rust`" is therefore not met; the plan's other P3.2 bars are
  (identical sets and `proven_irreducible` on every fixture with the flag ON, wall ≤ 2x:
  median 0.91x). Gates with the flag ON: `cargo test` 993/0, clippy `-D warnings`
  clean (both feature sets), smoke 4506, adversarial 19, IIS + parity + audit 650.
- [x] **P3.3 — IIS time bounding.** Scope: `_iis_milp.py` `_GRACE_S`, interrupt callback
  plumbing. The handle polls its deadline, so the HiGHS-specific overrun guard
  (`:64-69`) becomes route-specific. Done: a test that gives the Rust engine a probe with
  `time_limit=0.5` on `enlight`-class data returns within 1.0 s; `n_solves` and sets
  unchanged on P0.6. Kill: none. Regime: bound-neutral.

  **Result (2026-11-05): done.** The HiGHS overrun guard already became route-specific in
  P3.2 (it lives in `HighsBackend`; `_iis_reduce`'s HiGHS solve installs it too); the
  Rust backend relies on the driver polling its own deadline. `_iis_milp._GRACE_S` is now
  single-sourced from `_iis_backend.GRACE_S`. **Entry measurement:** a Rust probe with a
  0.5 s budget on enlight-class lights-out models (8x8 .. 14x14, presolve on and off)
  returned in **<= 0.58 s** (bar: 1.0 s), reported `INTERRUPTED`; a whole Rust-engine IIS
  of pigeonhole PHP(10,9) under `total_time_limit` 2 s / 6 s raised `TimeoutError` at
  2.01 s / 6.01 s (nothing confirmable in time -- on time). **`n_solves` on P0.6:** the
  HiGHS engine's counts are 53/53 identical to the tree before P3.1 (a worktree at
  `d553b789`, its `discopt` path and the absence of `_iis_backend` asserted); they
  differed from P0.6's recorded counts on 52/53 because the 2026-11-05 `main` merge
  (#2145/#2159) changed the IIS engine, not the backend refactor -- re-recorded with
  history, and now **pinned** for the HiGHS engine in
  `test_highs_parity_p06_iis_baseline.py`. Tests
  `test_highs_parity_p33_iis_time_bound.py` (the bar, presolve on/off; the total
  budget; the single-sourced grace). Gates: smoke 4505 (+1 timing-conditional skip,
  `test_1386`, at a loaded 513 s run), IIS + parity 643, ruff/mypy clean.

### Phase 4 — MILP search (R1-MILP), ordered by measured expected gain

Standing rule for every P4 task: flag default-OFF; entry experiment on MIPLIB-38
instances (never synthetic only); graduation panel = MIPLIB-38 + M panel, ON vs OFF,
interleaved, 3 rounds, bars pre-registered: (1) cert-clean, (2) solved-count ≥ OFF, total
nodes −15 % **or** solved +2 with nodes not worse, dual-bound losses ≤ 10 % of the open
population *by magnitude* (> 1 % relative, exceeding the OFF arm's own rep-to-rep spread —
the A14 lesson). Each graduation or retirement adds its audit row. **Order is provisional
and is re-written by P4.1.**

- [x] **P4.1 — Re-attribution at HEAD.** Scope: `miplib_panel.py attribute`. On P0.3's
  unsolved R instances, classify at `gap_tol=1e-4`: no incumbent / incumbent short by
  (opt − inc)/(1+|opt|) / bound short by (opt − bound)/(1+|opt|) at 20 s and 60 s (the
  A5/A14 method, both budgets so the "frozen primal" finding is re-tested after #2107's
  c-MIR). Done: a table over every unsolved instance, and the **order of P4.2–P4.8
  rewritten here** with one line of justification each. Kill: none.
  **Result (2026-11-05).** `miplib_panel.py attribute` (new; `classify` unit-tested):
  the 24 instances P0.3's Rust route left uncertified, each re-solved on the Rust route
  at 20 s and 60 s in a fresh process, classed against the published optimum by the
  larger shortfall, `|opt − v| / (1 + |opt|)` (`p4.1/attribution.json`, 48 executed):

  | instance | n / m / int | 20 s: class, primal, dual, wall | 60 s: class, primal, dual, term |
  |---|---|---|---|
  | `neos-1425699` | 105 / 89 / 85 | no-incumbent, —, 4.2e-05, 20.1 s | no-incumbent, —, 4.2e-05, time_limit |
  | `mik-250-20-75-5` | 270 / 195 / 250 | primal, 0.514, 0.031, 10.6 s | dual, 5.0e-04, 0.101, time_limit |
  | `neos-911970` | 888 / 107 / 840 | primal, 3.08, 0.565, 10.4 s | primal, 3.08, 0.565, time_limit |
  | `mik-250-20-75-3` | 270 / 195 / 250 | primal, 0.248, 0.0292, 10.6 s | dual, 0.0026, 0.0852, time_limit |
  | `fiber` | 1298 / 363 / 1254 | primal, 0.548, 0.541, 10.8 s | primal, 0.548, 0.276, time_limit |
  | `khb05250` | 1350 / 101 / 24 | primal, 0.0439, 0.0244, 11.1 s | solved, 0, 1.5e-12, gap |
  | `nexp-50-20-1-1` | 490 / 540 / 245 | primal, 0.633, 0.4, 10.3 s | primal, 0.633, 0.367, time_limit |
  | `mik-250-20-75-2` | 270 / 195 / 250 | primal, 0.411, 0.0334, 10.6 s | dual, 1.4e-16, 0.1, time_limit |
  | `nexp-50-20-4-2` | 1225 / 540 / 980 | dual, 0.125, 0.292, 10.5 s | dual, 0.125, 0.194, time_limit |
  | `gt2` | 188 / 29 / 188 | primal, 0.139, 9.9e-13, 20.1 s | primal, 0.139, 9.9e-13, time_limit |
  | `mik-250-20-75-1` | 270 / 195 / 250 | primal, 0.889, 0.0332, 10.7 s | dual, 0, 0.117, time_limit |
  | `enlight_hard` | 200 / 100 / 200 | no-incumbent, —, 0.289, 20.0 s | no-incumbent, —, 0.263, time_limit |
  | `sp150x300d` | 600 / 450 / 300 | dual, 0.343, 0.429, 10.5 s | dual, 0.343, 0.386, time_limit |
  | `neos17` | 535 / 486 / 300 | primal, 0.214, 0.094, 10.4 s | primal, 0.214, 0.0459, time_limit |
  | `beavma` | 390 / 372 / 195 | primal, 0.412, 0.396, 10.3 s | primal, 0.412, 0.367, time_limit |
  | `neos-3611447-jijia` | 472 / 377 / 85 | primal, 0.241, 0.0278, 10.2 s | primal, 0.241, 0.0278, node_limit |
  | `neos-3610051-istra` | 805 / 709 / 76 | primal, 0.12, 0.02, 10.4 s | primal, 0.12, 0.02, node_limit |
  | `neos-3611689-kaihu` | 421 / 323 / 88 | primal, 0.217, 0.025, 10.2 s | primal, 0.217, 0.00833, time_limit |
  | `neos-3610040-iskar` | 430 / 335 / 85 | primal, 0.132, 0.0263, 10.2 s | primal, 0.132, 0.0263, node_limit |
  | `neos-2624317-amur` | 524 / 342 / 524 | no-incumbent, —, 0.779, 20.0 s | no-incumbent, —, 0.779, time_limit |
  | `b-ball` | 100 / 30 / 88 | dual, 3.1e-15, 0.12, 10.1 s | dual, 3.1e-15, 0.12, time_limit |
  | `bppc8-02` | 232 / 59 / 230 | dual, 0.0118, 0.0404, 10.4 s | dual, 4.5e-16, 0.0404, time_limit |
  | `gsvm2rl3` | 241 / 180 / 60 | primal, 0.187, 0.114, 10.3 s | primal, 0.187, 0.0932, time_limit |
  | `neos-3118745-obra` | 1131 / 144 / 1130 | primal, 0.215, 0.00391, 10.5 s | primal, 0.215, 0.00391, time_limit |

  - **Counts.** 20 s: primal 17, dual 4, no-incumbent 3. 60 s: primal 12, dual 8,
    no-incumbent 3, solved 1 (`khb05250`, 37 s).
  - **The 20 s budget is half spent (new finding).** 20/24 stop at ~10.5 s of 20 s,
    uncertified, `termination=None`. `_solve_milp_simplex` gives run 1 a #291 stall
    slice of `min(0.5·remaining, 10 s)` and re-enters (#698) only when
    `remaining > slice` -- at `time_limit=20` that is 9.6 s ≤ 9.9 s, so re-entry is
    skipped and ~48 % of the budget is discarded; at 60 s re-entry runs. Every
    `time_limit` up to ~21 s is affected, including P0.3's 20 s R1 measurement.
  - **Frozen primal re-confirmed after #2107.** All 12 primal-class rows at 60 s hold
    the *same* incumbent at 20 s and 60 s; the dual bound moves, the incumbent does
    not. `mik-250-20-75-{1,2,3,5}` reach the optimum (or within 0.3 %) by 60 s and
    become dual-class (bound 8–12 % short).
  - **No incumbent:** `neos-1425699`, `enlight_hard`, `neos-2624317-amur` at both
    budgets.
  **Re-ordered P4 (from this data):**
  1. **P4.0 (new) — spend the budget:** re-enter on an uncertified `feasible` whenever
     any useful time remains (not only when it exceeds run 1's slice); 20/24 rows lose
     ~half their time at ≤ 20 s. Cheapest, broadest; flagged + §5 panel (it changes
     the search, though adoption is only on certification or strict improvement).
  2. **P4.6 — feasibility jump + shifting:** 12/24 primal-dominated at 60 s with the
     incumbent frozen 20 → 60 s, plus 3 with none -- the largest class.
  3. **P4.2 — RENS (pre-registered bar):** same class as P4.6, an existing mechanism.
  4. **P4.4 — tableau c-MIR / lifted cover:** 8/24 dual-dominated at 60 s, the `mik`
     family's bound 8–12 % short with the optimum in hand.
  5. **P4.3 — cut repair:** same dual class, cheaper than P4.4 if cuts are being
     dropped on cleanup.
  6. **P4.5 — root restart:** dual class; after P4.3/P4.4 change the root.
  7. **P4.8 — dimension-reducing presolve:** helps both classes, largest effort.
  8. **P4.7 — conflict analysis:** targets infeasibility-heavy search (`enlight_hard`,
     no-incumbent rows); 3/24.
- [x] **P4.0 — Spend the budget: re-enter on run 1's leftover. GRADUATED 2026-11-05.** Scope:
  `solver._solve_milp_simplex` (#698 re-entry gate), flag `DISCOPT_MILP_REENTRY_SPEND`
  (default OFF, audit row). The legacy gate re-enters only when the remainder exceeds run
  1's stall slice; below the 10 s cap the slice is half the remaining time, so on any
  `time_limit` under ~21 s the re-entry never runs (P4.1: 20/24 uncertified rows returned
  at 10.1-11.1 s of 20 s). The flag re-enters whenever `_MILP_REENTRY_MIN_S` (1 s)
  remains; adoption unchanged; `deterministic` keeps the legacy gate (#1116).
  Hypothesis: strictly better primal or dual on >= 3 of the 24 rows, no row worse beyond
  noise. Kill: < 3 → retire flag + implementation. Regime: bound-changing.
  **Entry (2026-11-05, `docs/dev/data/highs-parity/p4.0/`, `p40_reentry_entry.py`, 20 s,
  OFF/ON interleaved, fresh subprocess per solve, load < 1.5):**
  - *v1* (`entry_v1_bound_replaced.*`, 1 rep): executed 48, fired 20 (the 4 others have no
    incumbent, or used the whole budget). primal+ 4, dual+ 6, **dual- 6**. The four
    `mik-250-20-75-*` rows reached the optimum ON but their dual gap went 0.03 → 0.12.
  - **Defect found (default path too):** the #698 adoption branch *replaced* run 1's bound
    with run 2's whenever run 2 won on the primal side, though its comment (and the
    `node_limit` branch beside it) keeps the tighter of two valid bounds. Fixed in
    `163d725a` (regression
    test fails before / passes after). It is not flag-gated: it makes the code do what its
    comment already stated, and the kept value is a bound the solve itself proved. It also
    applies to the legacy re-entry at `time_limit` >= ~21 s.
  - *v2* (`entry_v2.*`, with the fix, 2 reps): executed 96, fired 40. The four `mik` rows
    go from primal gap 0.25-0.89 to <= 0.003 (`-1`, `-2` to the optimum) on **both**
    reps, with the dual gap unchanged (|Δ| <= 5e-4, run-1 timing noise). Other
    dual+/dual- verdicts are |Δ| <= 0.015 in both directions on rows whose OFF arm also
    varies between reps (`fiber`, `khb05250`, `jijia`, `neos17`): noise. Cert-clean on
    all 144 MIPLIB rows of v1+v2 (no bound past the optimum, no incumbent below it).
  - Entry bar met (4 rows strictly better primal, reproducibly). Next: the §5 corpus panel
    (`flag_onoff_corpus_panel.py --time-limit 20 --reps 2`; the flag is inert above ~21 s).
  **§5 corpus panel (2026-11-05, `corpus_panel.{jsonl,log}`):** in-repo MINLP corpus,
  `flag_onoff_corpus_panel.py --time-limit 20 --jobs 2 --reps 2`, load 1.5-4.9. **Cert-clean:**
  148 rows, 0 with a problem, certified 112 → 112 (none gained, none lost), 260
  incumbent feasibility checks, 0 errors. **Neutral on the corpus:** 13 rows differ, all
  on time-limited spatial-B&B / OA routes the flag does not reach, in both directions and
  swapping between reps (`beuster` 6352 ↔ 6372): timing noise. The three corpus instances
  on the native MILP route (`nvs02`, `nvs07`, `nvs14`) certify early and are identical.
  Net-positive evidence is the MIPLIB entry (4/24 rows reproducibly 25-89 % → <= 0.3 %
  primal gap, cert-clean on 144 rows). **Graduated 2026-11-05 (owner: "Graduate, then
  P4.6")**: default ON, `DISCOPT_MILP_REENTRY_SPEND=0` restores the legacy gate; audit
  row Graduated, live-gate marker 50 → 49. After the flip: smoke 4518 passed (444 s vs
  474 s before), adversarial 19, re-entry/audit/MILP tests 62 passed.
  Full fast lane (`-m "not slow" --time-budget 15`, 4 workers): 20252 passed, 29 failed --
  none from the flip: 6 need `sympy`/`cyipopt`, absent in this container (fail with `=0`
  too); `test_issue1729…[shape1]` takes 14.2 s with the flag ON *and* OFF, so it crosses the
  15 s budget only under load; the other 22 pass on a quiet re-run (the full run shared
  the machine with a runaway unit test of P4.6's prototype).
- [x] **P4.2 — RENS re-run with the pre-registered magnitude bar. RETIRED 2026-11-05.** Scope:
  `DISCOPT_RENS` (`milp_driver.rs`), no new code unless the bar demands a sub-MIP budget
  cap. A14 left RENS cert-clean, 12 incumbent wins / 0 losses, failing only a
  single-draw count bar it later called noise. Done: the panel above, 3 reps, dual-side
  bar by magnitude. Graduate (default ON, audit row) or **retire flag + implementation**
  (RINS stays). Kill: bar 2 fails → retire. Regime: bound-changing.
  **Result (2026-11-05): RETIRED (owner: "Retire per the bar").** `p42_rens_panel.py`
  (bars pre-registered in its docstring before the run, owner's choice "A14 bars +
  magnitude"), data `docs/dev/data/highs-parity/p4.2/`, build `56861877`: MIPLIB-38 + M
  (62 instances; `dcmulti` is in both), 3 reps, 20 s, Rust MILP route, 372 solves,
  functional gate passed (a garbage `DISCOPT_RENS` refused), `RensRun` 791 /
  `RensImproved` 77. Load was high (median 1-min load 6.2 on 4 cores), so no wall claim.
  - BAR 1 cert-clean: **pass** (372 checks, 0 problems, `RensRejected` 0).
  - 2a solved ON >= OFF: **FAIL**, 90 vs 95. All five are `dcmulti`: OFF certifies 5/6
    runs in 7-19 s, ON 0/6. Not budget -- its sub-MIPs used 63-120 nodes per solve --
    but a search interaction: ON ends with a worse incumbent (primal gap 0.0043 vs 0)
    and never closes.
  - 2b median primal gap over the 29-instance open population 0.214 → 0.064: pass.
  - 2c incumbent wins 10 / losses 1 (`dcmulti`): pass.
  - 2d dual losses by magnitude 0 (cap 2): pass. A14's dual-side failure did not
    reproduce once measured on magnitude and repetition.
  Retired per the pre-registered kill (bar 2 fails → retire, RINS stays): `rens_enabled`,
  `parse_rens_flag`, `rens_box`, `try_rens`, the call site and its state,
  `SubMipKind::Rens`, `Phase::Rens`, the eight `Rens*` counters, their unit tests, the
  two RENS integration-test binaries and their vertex-cover fixture. `cargo test` 1001/0
  (1004 minus the three RENS unit tests), RINS arms pass. Gates after retirement: smoke 4524,
  adversarial 19, parity/MILP/RINS tests 572, audit tests 11, clippy clean. The Python MINLP RENS (#281,
  `primal_heuristics.py`) is a different mechanism and is untouched. Recorded: RENS was a
  large primal win that cost a certification on one instance; a future attempt should
  start from why its incumbent derails `dcmulti`'s search.
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
  **Entry (2026-11-05, in progress -- decision point).** Instrument `p44_root_gap.py`:
  `Model.solve` on the Rust route at `max_nodes=2`, recording the FIRST driver call's bound
  (the shipped options, root cut loop included). Two probe defects found and fixed before
  any number was used: the solve's *reported* bound is a fallback engine's when a
  node-capped run without an incumbent defers (`sp150x300d` read 91.6 %, worse than its
  LP); and at `max_nodes=1` the tree's global bound is still `-inf` (root created, not
  solved). Root gap `(opt - bound)/(1+|opt|)`, % (data `p4.4/`):

  | arm | mik-1 | mik-2 | mik-3 | mik-5 | fiber | sp150x300d |
  |---|---|---|---|---|---|---|
  | HEAD (shipped route) | 4.11 | 3.99 | 3.79 | 4.16 | **61.55** | **50.0** |
  | + `cmir_cuts` (#2107 separator) | 4.15 | 4.35 | 3.89 | 4.12 | 32.72 | 38.57 |
  | `root_cuts` 2000 (= 5000 × 200 rounds) | 4.09 | 3.76 | 3.39 | 4.01 | 61.55 | 50.0 |
  | 5000 × 200 + `cmir_cuts` | 3.46 | 3.45 | 3.19 | 3.71 | 32.72 | 38.57 |
  | HiGHS (§2.5, Sept) | 1.84 | 1.52 | 1.76 | 1.72 | 1.95 | 0.00 |

  Findings that change the premise:
  1. **The pure-MILP route never runs c-MIR.** #2107 graduated `DISCOPT_MILP_CMIR` on the
     OA-master path only (`milp_simplex._strong_cut_kwargs`); `_solve_milp_simplex`
     passes no `cmir_cuts`, and every Rust entry defaults it `false`. Turning the
     existing separator on: `fiber` 61.55 → 32.72, `sp150x300d` 50.0 → 38.57; mik flat.
  2. **`fiber` and `sp150x300d` get ZERO root cuts at HEAD**: one round, cover finds
     nothing, GMI declines (`SepGomoryShortBasis` -- the root basis is shorter than m).
     Setting `expel_zero_artificials` in `milp_options` did not change it (the root basis
     GMI receives does not come from that option's path); not yet traced.
  3. **mik is budget-saturated, not family-starved:** GMI emits 946 cuts in 13 rounds,
     the loop stops on the 500 generated-cut cap; 2000 helps 0.0-0.4 pp and 5000 × 200
     adds nothing more; with c-MIR the best existing-mechanism mik residual is 3.2-3.7 %,
     between the hypothesis (≤ 2.5 %) and the kill line (> 3.5 %).
  **Re-scoped (owner, 2026-11-05: "Fix route gaps first").** Built
  `DISCOPT_MILP_ROOT_GMI_REPAIR` (default OFF, `2a0504cf`): (1) a short root basis is
  completed with unit columns of the rows a partial-pivoting elimination leaves
  unpivoted (structural matching was tried first and gave a numerically singular basis
  on `fiber` -- recorded, replaced; the dense elimination is capped at `m·k² <= 5e8`,
  matching above); (2) the slack rewrite maps a slack column to the row it is a clean
  singleton slack OF (on `sp150x300d` all 30 completed-basis cuts had been dropped with
  `SubstDropNoSlack`: slackless rows shift the `ns + r` convention). Root gap with the flag
  (`p4.4/root_repair*.json`): `fiber` 61.55 → 19.76 %, `sp150x300d` 50.0 → 30.0 %, `mik`
  bit-identical; c-MIR on top adds little (`fiber` 18.27 %) and makes `mik-2` worse, so it
  is not part of this flag. Soundness: 400 seeded MILPs with short bases, 319 cuts ×
  4193 integer-feasible points by exact enumeration, 0 violations.
  **§5 panel (2026-11-05/06, `p4.4/p44_panel.json`, bars pre-registered in
  `p44_gmi_repair_panel.py`):** MIPLIB-38 + M, 3 reps, 20 s, 372 solves, fired ON
  2,517,336 / OFF 0; median load 6.2 on 4 cores, so no wall claim.
  - BAR 1 cert-clean: **pass** (372 checks, 0 problems).
  - 2a solved: **96 → 99**. 2b nodes over the 30 always-certified instances:
    **157,233 → 127,323 (−19 %)**. 2c dual losses: **1** (`neos17` 0.097 → 0.115), cap 2.
  - Dual gains: `fiber` 0.556 → 0.194, `b-ball` 0.120 → 0.0002, `beavma` 0.396 → 0.161,
    `khb05250` 0.029 → 0.0015, `nexp-50-20-1-1` 0.400 → 0.267, `nexp-50-20-4-2` 0.292 →
    0.181, `sp150x300d` 0.429 → 0.300, `neos-3610040-iskar` 0.026 → 0.000.
  - MINLP corpus (`p4.4/p44_corpus.jsonl`, 20 s): 74 rows, 0 problems, certified 56 = 56,
    130 incumbent checks.
  Both bars met → graduation is the owner's call. Tableau c-MIR (the original P4.4
  scope) is deferred: with the repair, `mik` stays at 3.8-4.2 %; its limiter is the
  500-cut cap (2000 helps 0-0.4 pp), not a missing family.
- [ ] **P4.5 — Root restart.** Scope: `milp_driver.rs` root loop. After the root cut loop,
  if reduced-cost fixing + FBBT fixed ≥ 10 % of columns, re-run presolve + root loop on the
  reduced problem once (HiGHS's closer on `beavma`/`dcmulti`, §2.6). Entry: count such
  roots on MIPLIB-38; hypothesis ≥ 8 instances qualify and the restarted root bound
  improves ≥ 1 pp on ≥ 4. Flag `DISCOPT_MILP_ROOT_RESTART`. Done: panel bars. Kill: entry
  fails → retire. Regime: bound-changing.
- [x] **P4.6 — Feasibility jump + shifting primal heuristics. KILLED AT ENTRY 2026-11-05.** Scope: new
  `bnb/primal_fj.rs`. Entry: on P4.1's no-incumbent instances, a standalone FJ run (10 %
  of TL) finds a feasible point on ≥ half of them. Flag `DISCOPT_MILP_FJ`. Done: panel bars
  with the incumbent-verified screen; FJ points are verified before adoption. Kill: entry
  < half → retire. Regime: bound-changing (primal only; bounds untouched — still panelled).
  **Result (2026-11-05): killed at its entry bar; implementation retired.** Instrument:
  `p46_fj_entry.py` (data `docs/dev/data/highs-parity/p4.6/`). A Rust Feasibility Jump
  (Luteberget & Sartor 2023: weighted `<=` rows, jump values by a slope walk over each
  column's breakpoints, candidates from up to 25 sampled violated rows plus 5 random
  columns, +1 weight on every violated row at a local minimum; commit `4aef7331`), run
  standalone at 10 % of the 20 s limit (2 s), 5 seeds, every point re-verified on the
  original rows / box / integrality:
  - `neos-1425699`: found 5/5 in 0.03-0.17 s; objective 6.97e9-8.19e9 against the
    optimum 3.18e9 (2.2-2.6x).
  - `enlight_hard`: 0/5 (~300k moves, ~55k weight steps each).
  - `neos-2624317-amur`: 0/5 (~150k moves, ~47k weight steps each).
  1/3 < the pre-registered half. A full 20 s run (1.6-3.2M moves, 2 seeds) finds
  nothing on either miss, so the failure is not budget. A Python prototype of the same
  search (60 s, ~2.5-6k moves) found nothing on any of the three; it is too slow to judge
  and is kept in the script only as a reference implementation. Retired: `bnb/primal_fj.rs`
  and `fj_search_py` deleted (recoverable from `4aef7331`); no flag was ever added.
  Not tested by this entry: FJ on the 12 *primal-class* rows (incumbent frozen 20 → 60 s),
  which P4.2 (RENS) targets next.
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
| P0.4 | [x] | kill fired: false HiGHS master optimum (#2146), fixed; post-fix 3103 masters, 6206 compared, 0 contradictions; Rust SGM 0.08× HiGHS on decomposition masters; route 91–92/94 | `6351f589` + see git log |
| P0.5 | [x] | 268 fixtures + 5 hard (R3 30 with cuts, R4 60, R6 178), 1.31 MB, HiGHS reproduces all 268 to 1e-9 | see git log |
| P0.6 | [x] | 53 IIS fixtures (38 with integers), deterministic 53/53, all proven irreducible; MIPcc27 absent | see git log |
| P0.7 / D0 | [x] / **awaiting owner** | Rust: no incumbent on the witness at 20/60/120 s (bound gap 0.40 %), overruns its limit by up to 47 % (driver defers to the Python tree); HiGHS 9.9 s certified; 0 violations | see git log |
| P1.1–P1.4 | [x] (P1.3 killed; P1.4 state 3) | P1.1: seam; HiGHS 37/37 byte-identical; Rust L 24/25, INF 12/12. P1.2: #1611 rescue Rust-first (3/3 identical); #1537 (1/94 differs) and #1869 (24/85) stay HiGHS-first. P1.3 killed: Rust dual looser on 3/60 R4 LPs. P1.4: Rust root cuts cert-clean, bounds 11/11, wall 1.078x → opt-out; HiGHS-less installs now get root cuts | see git log |
| P2.1–P2.6 / D1 | P2.1 [x], P2.0 [x], P2.2 killed, P2.5 [x] (re-scoped), P2.3 killed at entry, P2.4 not pursued (owner), P2.6 [x] → **D1: flipped** | P2.1: 93–100 % of the Rust LP route is Python (root presolve, post-solve re-check, declared-box tightening). P2.0 (owner-accepted 2026-11-05): 13 slowest L rows vs P0.3, R 0.323 / H 0.556, 26/26 verdicts identical; L Rust wall 47.4 → 12.0 s. P2.2 killed: feasibility-preserving triangular crash −3.4 % pivots (bar −10 %). P2.5 (re-scoped): sparse exact dual correction, `perold` certifies, Rust L 25/25. P2.3 killed at entry: 24 % / 21 % low-fill FTRAN/BTRAN on the P2.1 rows (bar 50 %) | see git log |
| P3.1–P3.3 | P3.1 [x], P3.2 [x] (gaps), P3.3 [x] | P3.1: Rust `MilpHandle`, 276 fresh-solve comparisons bit-identical, 120/120 vs HiGHS. P3.2: IIS engine behind a backend; `DISCOPT_FARKAS_EXACT_OPEN` graduated (corpus cert-clean, Rust IIS 53/53 vs P0.6, median wall 0.91x); IIS default stays HiGHS (big-M slack, pigeonhole → Phase 4). P3.3: Rust probe 0.5 s budget returns ≤ 0.58 s; HiGHS `n_solves` pinned, refactor-neutral 53/53 | see git log |
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
| UC witness (P0.7) | R @ 20/60/120 s; H @ 20 s | R 0/3 at every limit (no incumbent); H 3/3 certified | R 21 / 61–88 / 141–147 s; H 9.9–10.5 s | — | R 3 / 63–95 / 159; H 7 | 0 | idle (load 0.0) |

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

**2026-11-04 — P0.4: "the HiGHS master's #1634 cross-check keeps its bounds valid".**
Commit `67530f14`, 3050 masters replayed, compared 6100. Measured: on
`milp_default_box_clay0203_master` HiGHS's primary bound 41595.42 and presolve-free bound
41573.26 both exceed the true optimum 41420.272 (Rust-verified point, feasible to 1e-14),
and the cross-check published 41573.26 as OPTIMAL. Expected: bound ≤ 41420.272.
Consequence: P0-blocking correctness defect jkitchin/discopt#2146, fixed in `6351f589`
(every HiGHS master opens the default box; #1917's narrower fallback removed); P0.4 re-run
on the fix: 6206 compared, 0 contradictions. Retraction of: none in this plan; it limits
#1634's "min of the two bounds is valid whenever either one is" to inputs where at least
one configuration is right.


**2026-11-04 — P1.2: "the Rust engine reaches HiGHS's verdicts on the MINLP-path LPs".**
Commit `ac54f5d9` + P1.2, P0.5's 183 R6 fixtures, compared 183. Measured: 158 identical;
#1537 arbitration 1/94 differs (HiGHS: verified Farkas proof of emptiness; Rust: a point
feasible to ~3e-16 relative); #1869 free-column 24/85 differ (Rust: no certified bound on
23, a looser one on 1; 22 are nvs16 LPs boxed at 1.6e9). Expected: identical verdicts.
Consequence: #1537 and #1869 stay HiGHS-first (P1.2 kill); only the #1611 rescue (3/3
identical) is Rust-first. The 24 #1869 LPs and the nvs0 arbitration LP are P2.1 inputs.


**2026-11-04 — P1.3: "the Rust dual certifies at least HiGHS's bound on every R4 LP".**
Commit `d9a6009a`, P0.5's 60 R4 fixtures, compared 60. Measured: 0 lost, 55 within
CERT_ABS + CERT_REL·|v|, 3 looser by 1.0–1.25e-6 (translated `clay0303hfsg`), 2 strictly
tighter. Expected: none looser. Consequence: P1.3 killed, R4 stays HiGHS-first, the 3 LPs
go to P2.1.

**2026-11-05 — P2.2: "a crash basis cuts the Rust LP route's cold iterations ≥ 30 %".**
Commit `cada2c4e` + an uncommitted crash, L panel, Rust route, pivots summed over every LP
of each solve. Measured: phase 1 is 49.6 % of 15,549 pivots, but a crash that keeps phase
1's start feasible (the only kind this sum-of-artificials phase 1 accepts) cut total
pivots by 0.06 % (strict) and 3.4 % (singleton rows allowed); it placed 1–50 columns into
127–257 open rows, the rest having no in-box column. Expected: ≥ 30 % (kill < 10 %).
Consequence: P2.2 killed, flag and implementation retired unmerged; a feasibility-ignoring
crash would need a composite phase 1 (not scheduled).

**2026-11-05 — P2.3: "≥ 50 % of FTRAN/BTRAN solves on the P2.1 instances have < 10 % fill".**
Commit `b8d89b7f` + temporary counters, Rust route, 13 P2.1 LPs. Measured: 24 % of 15,168
FTRAN and 21 % of 25,415 BTRAN results under 10 % fill. Expected: ≥ 50 %. Consequence: P2.3
killed at entry, nothing built. On the M panel ~50 % are low-fill but the solves are ≤ 18 %
of wall (≤ ~5–9 % gain, under the 10 % bar); FBBT is the larger M-panel phase (P4.1).

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
