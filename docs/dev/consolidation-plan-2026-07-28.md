# Consolidation & gap-closure plan — 2026-07-28

**Companion to** `docs/dev/architecture-review-2026-07-28.md` (the evidence base;
file:line citations below refer to it and to the tree at `main @ 41fd72c4`).
**Execution model:** phases implemented by Claude Opus sessions. Each phase card is
self-contained — an implementer reads **§0 (the contract) plus their card only** and
can start work. Cards inside a phase are ordered; phases marked independent can
interleave.

**Goals, in priority order:** (1) preserve every correctness certificate,
(2) close the measured performance/capability gaps vs SCIP/BARON,
(3) consolidate redundancy and make the code modular enough that (1) and (2) stay
cheap to verify. Consolidation is *subordinate* to gap closure: where a redundancy
lives on a path that kernel convergence (Phase 5) will retire, we do not hand-polish
it first.

## Refresh delta (what changed since the review snapshot)

- **#904 resolved #902 (Card F).** The native-kernel quality regression (seed
  enumerated 0/1 over general integers) is fixed; the re-run graduation panel passes
  all three gates (quality-clean, cert-clean, net-positive → GRADUATE YES), with a
  standing panel-quality gate test (`test_902_panel_quality_gate.py`). **Card C
  (kernel engagement) is unblocked** — Phase 5 is now executable.
- **#905 landed the #843 QUBO/Ising primal default-ON**, extracted at graduation to
  a JAX-free `discopt/qubo_primal.py`. One of the six G-B no-incumbent instances
  (`chimera_k64ising-01`) is closed. The extraction is the **house pattern for
  Phase 6 of this plan**: a capability graduates *out of* the monolith into a
  dependency-light module, with a structural no-fire proof over the corpus.
- Re-verified on `41fd72c4`: `solve_model` is 7,443 lines; the per-node double FBBT
  is still live at `solver.py:8973` and `:12762`; `cascade_aux` is still resolved at
  only one of six `obbt_tighten_root` sites; 28 `os.environ` reads remain in
  `solver.py`. All review findings stand.

---

## §0. Binding contract (read before any phase)

These rules bind every card. They restate and extend CLAUDE.md §1–§5; where they
conflict with convenience, they win.

1. **Two verification regimes, chosen per change, stated in the PR:**
   - **Regime N (bound-neutral):** refactors, dead-code deletion, wiring that must
     not change math. Gate: `node_count` and certified `objective` **exactly
     unchanged** on the certifying panel (Phase 0's `panel_baseline.json`), plus
     `pytest -m smoke` and the adversarial suite. Any drift — even improvement —
     fails the card.
   - **Regime C (bound-changing):** tightenings, cuts, routing changes, deletions of
     *active* tightenings. Gate: differential panel ON-vs-OFF per CLAUDE.md §5 —
     cert-clean (incorrect_count = 0, no bound above reference, no certification
     regression, incumbents feasibility-verified) AND net-positive — **plus the
     #902 incumbent-quality gate** (no instance's incumbent degrades vs reference,
     per `test_902_panel_quality_gate.py`). Behind a flag, default-OFF until the
     panel passes.
2. **Deletion of an active mechanism is Regime C, not Regime N** — removing a
   tightening can loosen bounds. Deletion of provably-unreachable code is Regime N.
   "Provably unreachable" means: no call site sets the gate true anywhere in
   `python/`, `crates/`, `discopt_benchmarks/`, or `scripts/`, verified in the PR.
3. **Entry experiment before build** (CLAUDE.md §4): any card resting on a
   hypothesis names its falsifying experiment and kill criterion, runs it on real
   corpus instances first, and records the result in this file (§6, falsification
   log) whichever way it goes.
4. **No card weakens a validation, gate, or safety guard.** If a gate can only pass
   by weakening, the card stops and reports.
5. **Instrumentation discipline** (CLAUDE.md §6–§11) applies to every measurement a
   card produces: executed-assertion counts, no swallowed exceptions, verify the
   loaded module, interleaved timing with spread, incremental output.
6. **Every card ends with:** what was run, the numbers, and an explicit
   can-this-card-close statement. Follow-up work goes in a named issue, not an
   implicit TODO.
7. **Flags:** new behavior ships behind a `DISCOPT_*` flag using the Phase 1 helper
   (once landed), default-OFF, graduating per Regime C. Graduated flags keep the
   `=0` opt-out.
8. **Do not rebuild what exists.** `docs/design/relaxation-catalog.md` lists the
   relaxation layer as done; the review §1 lists heuristics/branching/simplex at
   parity. A card needing one of these *wires* it, never re-implements it.

**Panel infrastructure note.** Until Phase 0 lands, cards use the existing
`issue764_native_kernel_graduation_panel.py` machinery as the Regime C harness and
the 61-instance in-repo corpus for Regime N. After Phase 0, its artifacts are
mandatory.

---

## Phase 0 — Measurement substrate (prerequisite for everything)

> **Status:** LANDED 2026-07-29 (tasks 1 and 3 complete; task 2's script + wiring
> complete, its list awaits a machine with the snapshot). **Depends:** nothing.
> **Regime:** N (pure tooling — no solver behavior touched).
> **Est:** 1–2 sessions.
>
> **What exists now.** `discopt_benchmarks/scripts/panel_baseline.py` (producer +
> `--check`), `reports/panel_baseline_f154dcff.json` (119 instances, 45 s budget,
> defaults, clean tree: 87 optimal / 16 feasible / 15 time_limit / 1 child_timeout,
> 2166 s wall, 85 rows in the Regime-N comparable population),
> `discopt_benchmarks/scripts/select_heldout50.py` + `[suites.heldout50]` +
> `run_benchmarks.py`'s `requires_corpus_snapshot` refusal, and
> `discopt_benchmarks/tests/test_panel_baseline.py` (12 tests).
>
> **Use it like this.** A Regime-N card runs
> `python -u discopt_benchmarks/scripts/panel_baseline.py --check
> reports/panel_baseline_f154dcff.json` and pastes the *executed-comparison count*
> into the PR alongside the verdict. A count of zero is a failure, not a pass. When
> a card legitimately re-baselines (a new default lands), generate a fresh
> `panel_baseline_<sha>.json` and say in the PR which baseline the next card gates
> against.
>
> **Two things the card did not close.** (a) `heldout50` cannot be drawn in an
> environment without the MINLPLib snapshot, so the committed list is a placeholder
> and every consumer refuses loudly (`heldout50: SKIPPED — local only`, exit 3)
> rather than passing empty; the owner materializes it with one command. (b) The
> baseline surfaced a solver finding, not a tooling one: `heatexch_gen3` under
> `time_limit=45` runs **233.4 s** (5.2× its own budget, re-measured standalone).
> Phase 0 labels it `child_timeout` and changes nothing; it wants its own issue.

The review found one systemic evaluation risk: `global50` is both the iteration set
and the graduation gate set (`benchmarks.toml:82-95`), and Regime N currently has no
frozen baseline artifact. Every later card's gate quality depends on fixing this
first.

**Tasks.**
1. `discopt_benchmarks/scripts/panel_baseline.py`: run the in-repo 61-instance
   corpus (plus the 147-instance `.nl` set used by #865) on defaults; emit
   `reports/panel_baseline_<sha>.json` with per-instance `node_count`, certified
   `objective`, `status`, `bound`, wall. Add `--check <baseline.json>` mode that
   fails non-zero on any node-count/objective drift, printing the executed
   comparison count (§0.5).
2. **Held-out graduation panel:** a seeded, rotating draw of ~50 instances from
   `~/Dropbox/projects/discopt-minlp-benchmark/` (stratified by `minlplib_types.csv`
   type and `problems_*.txt` runtime band), disjoint from `global50`. Wire it into
   `benchmarks.toml` as `suite = "heldout50"`; Regime C gates require BOTH
   `global50` (regression watch) and `heldout50` (generalization). When the Dropbox
   snapshot is unavailable (CI), the gate records `heldout50: SKIPPED — local only`
   loudly rather than passing silently.
3. Populate the `root_gap` instrumentation schema (`certification-gap-plan` gap
   table row "schema exists, never populated") from the baseline run — later cuts
   /propagation cards need root-gap deltas as their primary metric.

**Exit:** baseline artifact checked in under `reports/`; `--check` proves itself by
detecting a deliberate 1-node perturbation (test); `heldout50` resolvable locally.

---

## Phase 1 — Mechanical hygiene (bound-neutral, immediately shippable)

> **Status:** LANDED 2026-07-29 (Cards 1a and 1b both complete).
> **Depends:** Phase 0 for the check artifact. **Regime:** N throughout.
>
> **What exists now (1a).** `python/discopt/_env.py`
> (`env_bool`/`env_int`/`env_float`/`env_str`/`env_enum` — one truth table,
> `ValueError` on anything outside it) and its Rust twin
> `crates/discopt-core/src/env.rs` (same table; unparseable warns on stderr and
> takes the default, since a kernel has no exception channel). All 57 inline
> Python reads migrated (27 in `solver.py`); `solver_tuning.py`'s helpers rewired,
> `_env_cut_inherit` keeping its tri-state. The three Rust presence-test flags
> (`DISCOPT_PROFILE`, `DISCOPT_DISABLE_CSE`, `DISCOPT_T14_DBG`) now honour `=0`.
> `python/discopt/_flag_registry.py` holds 73 rows plus `solver_tuning_flags()`,
> which recovers the 46 `SolverTuning` flags from the dataclass by AST instead of
> a hand-copy; `scripts/gen_flag_docs.py` generates `docs/reference/flags.md`
> (119 flags), linked from README and `docs/_toc.yml`.
> `python/tests/test_flag_registry.py` (17 tests) locks the truth table, the
> registry closure, the daemon suffix list, the doc's freshness, and the
> **CI grep-gate**: zero raw `os.environ.get("DISCOPT_` outside
> `_env.py`/`solver_tuning.py`. Every default is unchanged; CHANGELOG
> (Unreleased → Fixed) lists every flag whose non-`0`/`1` spellings changed.
>
> **What exists now (1b).** −4,889 net lines: `presolve/obbt.rs`, the A3
> Rust↔Python handshake (`run_orchestrated_presolve` + `ConvexReformPass` +
> `ReverseADPass` + `SeparabilityPass` + the two protocol helpers only they used),
> the three never-enabled Rust passes (`scaling.rs`, `duality.rs`,
> `reduction_constraints.rs`) with their adapters / pass names /
> `reduced_cost_info` plumbing / `row_scales`+`col_scales` delta fields, five
> orphaned PyO3 exports (`solve_lp_py`, `crossover_to_vertex_py`,
> `recover_basis_py`, `PyBatchDispatcher`+`batch.rs`, `parse_nl_string`), and three
> orphaned `_jax` modules (`embedding.py`, `operator_relaxations.py`,
> `soc_cuts.py`). `fbbt_fp.rs` and `symmetry.rs` are parked with header comments
> naming Card 3b and Phase 7.3. `nn/presolve.py` is kept, with its "informational
> v0, not wired into any solve path" caveat moved to the first line of the class
> docstring.
>
> **What the card said to delete and re-verification saved.** Six targets had live
> callers in the directories §0.2 names, so they are NOT provably unreachable:
> `solve_convex_node_py` and `convex_warmlp_probe_py`
> (`discopt_benchmarks/scripts/issue798_k1_bytecheck.py` — the K1 gate — and
> `issue807_w0_probe.py`), `SubstitutionChain` (a CLAUDE.md §8 load-marker
> assertion in `presolve_roundloop_census.py`, and the return type of the live
> `PyModelRepr.substitute`), `_jax/claim_audit.py` (two benchmark scripts),
> `_jax/symbolic/domains/power.py` + `_jax/symbolic/registry.py` +
> `_jax/symbolic/certified_learned.py` + `_jax/pounce_layer.py` (all imported by
> notebooks that are in `docs/_toc.yml`), `_jax/symbolic/signed_signomial.py`
> (imported by the live `_jax/symbolic/patterns.py`), and `_jax/icnn_trainer.py`
> (named in `learned_relaxations.py`'s error message as the way to produce its
> inputs). `parse_nl_string`'s tests were **migrated, not deleted** — they now
> round-trip through the live `parse_nl_file`, so no `.nl`-parser coverage was
> traded for the deletion (§0.4).
>
> **Verification (both cards, final tree, 2026-07-29).** The Rust extension was
> rebuilt from the edited sources and its loaded state asserted before any
> measurement (§0.5 / CLAUDE.md §8): `discopt.__file__` and `_rust.__file__` under
> the repo, the five deleted PyO3 exports **absent**, five kept exports present,
> and `presolve(passes=["scaling"])` raising `unknown presolve pass` — so the panel
> exercised **new Python + new Rust**, not a stale prebuilt `.so`.
> - `panel_baseline.py --check reports/panel_baseline_f154dcff.json`:
>   **comparisons executed: 255** (node_count 85, certified objective 85, status 85)
>   over 85 comparable of 119 baseline rows — **PASS: no node-count or
>   certified-objective drift.** 1943.3 s wall, load start 0.21 peak 2.73. 20
>   non-comparable rows reported and not gating (budget-dependent `time_limit` /
>   `feasible` statuses, plus `heatexch_gen3`'s known `child_timeout`).
> - `pytest -m smoke python/tests`: 844 passed, 16 skipped, 2 xpassed (846 before
>   1b; the −2 are the two tests of the deleted `_jax` modules).
>   `pytest -m smoke discopt_benchmarks/tests`: 51 passed, 1 skipped.
> - `pytest -m slow python/tests/test_adversarial_recent_fixes.py`: **10 passed**,
>   including the load-sensitive `test_large_dense_jacobian_no_crash`.
> - `cargo test -p discopt-core`: 536 + 4 + 1 passed (567 before 1b; the −31 are the
>   deleted kernels' own tests). `ruff check` / `ruff format --check` clean on every
>   changed file.
>
> **One finding for Phase 2, not fixed here.** `_jax/superposition.py` looks
> test-only but is not dead: `MccormickLPRelaxer` threads
> `superposition=(relaxation_arithmetic == "superposition")` into
> `build_milp_relaxation`, which **accepts the argument and ignores it**
> (`_SUPERPOSITION_FUNCS` is likewise defined and never read). That is a wiring
> defect of the Card 2a/2c family, not dead code; the module is kept.

### Card 1a — One flag helper, one truth table

Seven incompatible boolean parse idioms exist (review §2.4): `DISCOPT_RLT=false`
enables RLT (`solver_tuning.py:33` is `raw != "0"`); `DISCOPT_CONVEX_KERNEL=off`
enables it; `DISCOPT_PROFILE=0` enables profiling (Rust presence test,
`profile.rs:23`); `DISCOPT_SGO=2` disables it.

1. `python/discopt/_env.py`: `env_bool(name, default)` accepting exactly
   `1/true/yes/on` → True, `0/false/no/off` → False (case-insensitive), empty/unset
   → default, **anything else raises `ValueError` naming the flag and the accepted
   values** (loud refusal per CLAUDE.md §3). `env_int`/`env_float`/`env_enum`
   likewise.
2. Migrate all Python inline reads (57 flags; 28 sites in `solver.py`) and
   `solver_tuning.py`'s four `_env_*` helpers to it. **Compatibility note in the
   CHANGELOG**: strings like `false` change meaning for registry flags — that is
   the point; the old parse was a defect, and the new one errors rather than
   silently flipping (no silent behavior change is possible).
3. Rust: one `env_bool` in `discopt-core` with the same table; fix the three
   presence-test flags (`DISCOPT_PROFILE`, `DISCOPT_DISABLE_CSE`,
   `DISCOPT_T14_DBG`) to honor `=0`.
4. Registry shim: every non-`SolverTuning` flag gets a row in a static
   `FLAG_REGISTRY` (name, default, kind: graduated/parked/permanent/debug, owning
   issue, one-line doc) — including the 12 f-string daemon flags. A test asserts
   every `env_bool` call site's name is in the registry (greppable via the helper).
5. Generate `docs/reference/flags.md` from the registry + `SolverTuning` fields;
   link from README. Fix the stale `flag-graduation-protocol.md:29` parked table
   (`DISCOPT_NODE_REDUCE` was removed in #581) and the `node_reduce.py:30` /
   `solver_tuning.py:894-903` contradiction (module exists, flag doesn't).

**Exit:** zero raw `os.environ.get("DISCOPT_` outside `_env.py`/`solver_tuning.py`
(CI grep-gate); Regime N panel clean; a new test locks the truth table in both
languages.

### Card 1b — Delete the provably dead

Per review §2.3.4 and the audit, all verified zero-caller (re-verify per §0.2 before
each deletion):

- `crates/discopt-core/src/presolve/obbt.rs` (623 lines, zero callers).
- The A3 handshake: `_jax/presolve/orchestrator.py` `run_orchestrated_presolve`
  (unreachable — no caller passes `python_passes`) and the passes only it can run
  (`ReverseADPass` duplicate wrapper, `SeparabilityPass`, `ConvexReformPass`), ~700
  lines. Keep `nn/presolve.py` (documented informational v0, user-visible) but move
  its "not wired" caveat into the class docstring's first line.
- Orphaned PyO3 exports (8 of ~28: `solve_lp_py`, `crossover_to_vertex_py`,
  `recover_basis_py`, `solve_convex_node_py`, `convex_warmlp_probe_py`,
  `PyBatchDispatcher`, `parse_nl_string`, `PySubstitutionChain`) — delete the
  bindings; keep any Rust internals another path uses.
- Never-enabled Rust passes: `scaling.rs`, `duality.rs`, `reduction_constraints.rs`
  — delete pass + plumbing (their concepts live on in the live implementations:
  MILP-driver RCF, etc.). **Exception — `fbbt_fp.rs` and `symmetry.rs` are NOT
  deleted**; they are claimed-superior or SOTA-relevant and become entry
  experiments (Cards 3b, 8c). Park them with a header comment naming the card.
- `_jax/symbolic/domains/power.py` (true orphan) and the 11 test-only `_jax`
  modules: delete the orphan; for test-only modules, either promote (wire the test
  into a user API) or delete module+test — decided per module in the PR, defaulting
  to delete.

**Exit:** `cargo test -p discopt-core` and full pytest green; Regime N panel clean;
LOC delta reported in the PR (~5,000 expected).

---

## Phase 2 — Wiring defects (small, measured, mostly Regime C)

> **Status:** LANDED 2026-07-29 — but **two of the three cards landed as
> measured NOs**, which is the outcome their own gates produced, not a shortfall.
> **Depends:** Phase 0. **Regime:** C (2a), C→killed (2b), N (2c.1), measurement
> only (2c.2).
>
> **Card 2a — LANDED, extra sites measured NET-NEGATIVE and are off explicitly.**
> `SolverTuning.obbt_cascade_aux` is now the single resolution point for
> `DISCOPT_OBBT_CASCADE_AUX` (moved out of `_flag_registry`; `docs/reference/flags.md`
> regenerated), all six `obbt_tighten_root` call sites pass `cascade_aux`
> **explicitly**, and `TestCascadeAuxWiredAtEverySite` fails any future call site
> that omits it (AST walk over `python/discopt`, prints its 6 executed assertions).
> The differential panel (`discopt_benchmarks/scripts/card2a_cascade_aux_panel.py`,
> artifact `reports/card2a_cascade_aux_on.json`) was **cert-clean, 0 violations over
> 578 executed comparisons**, and **failed the #902 quality gate**; per-site
> attribution by interleaved replicated A/B against the pre-card tree put the harm
> at the root and incumbent sites and the only per-node gain at one instance. All
> five newly-wired sites therefore ship `cascade_aux=False` with the measurement
> cited at the call site. Numbers in §6.
>
> **Card 2b — KILLED by its entry experiment (do NOT delete).** The hypothesis was
> that Rust `in_tree_presolve` subsumes the per-node Python Jacobian FBBT. It does
> not: **278 of 1,495 decided nodes (18.6 %)** carry a Python-only inference — 37×
> the card's 0.5 % kill criterion — over 82,316 individual bound comparisons on 25
> corpus instances. Both per-node call sites stay. Probe:
> `discopt_benchmarks/scripts/card2b_fbbt_subsumption_entry.py`. Numbers, the
> attribution of *which* half of the pass Rust is missing, and the follow-up scope
> in §6.
>
> **Card 2c.1 — LANDED (Regime N).** The plan's premise ("the identical pass runs
> twice") is **falsified**: the two calls differ in input box, model state and
> position relative to dispatch, so neither can replace the other. What was really
> being discarded — the first call's tightened *box* — is now intersected into the
> root working box, guarded by model identity and array shape. Regime-N spot check
> clean; full panel below.
>
> **Card 2c.2 — measured, filed, NOT built (as the card instructs).** Over all 119
> corpus instances the three passes produce **zero** bound tightenings and
> **8,614 non-bound rewrites on 32 instances (27 %)**: `simplify` removes 8,439
> rows on 31 instances, `redundancy` 149 rows on 2, `coefficient_strengthening`
> rewrites 26 rows on 2. That is far past "near-zero", so removing the passes is
> wrong and repr-level adoption is the fix — which the card explicitly scopes out.
> Filed as **Phase 3 sub-card 3d** below. Probe:
> `discopt_benchmarks/scripts/card2c_presolve_rewrites_entry.py`, artifact
> `reports/card2c_presolve_rewrites.json`.
>
> **Superposition wiring defect (Phase 1's hand-off) — resolved as: dead table
> deleted, live-but-unfinished parameter kept.** `_SUPERPOSITION_FUNCS` was defined
> and never read (the #632 cutover deleted its consumer) and is gone. The
> `superposition=` parameter is *not* removed: `build_milp_relaxation` documents
> five arguments as ignored-for-signature-compatibility since #632, and
> `_jax/superposition.py` is a live, tested cut generator the uniform engine has not
> re-adopted — an unfinished feature, not a removed one. Retiring the whole
> ignored-argument family belongs to Card 4b's signature cleanup.
>
> **`heldout50`: SKIPPED — local only.** This environment has no MINLPLib snapshot,
> so the Regime-C gate for Card 2a ran on the in-repo 119-instance corpus (both
> `minlplib_nl` and `minlplib`, the union Phase 0 froze) plus targeted interleaved
> replicated A/B. The generalization arm the plan asks for was **not** run and this
> verdict is therefore corpus-local. Card 2a's outcome is a *negative* (sites stay
> off), so the missing arm cannot have admitted an unmeasured bound change.
>
> **Verification (final tree `f557b056`, 2026-07-29).** The committed defaults are:
> cascade ON at `root_reduce` only (unchanged from before the phase), explicit
> `cascade_aux=False` at the other five sites, both per-node Python FBBT calls
> retained, the declared-box tightening carried forward, `_SUPERPOSITION_FUNCS`
> deleted. No Rust source was touched, so `cargo test -p discopt-core` was not
> required; the prebuilt extension was asserted loaded from the repo before every
> measurement (CLAUDE.md §8).
> - `panel_baseline.py --check reports/panel_baseline_f154dcff.json`:
>   **comparisons executed: 255** (node_count 85, certified objective 85, status 85)
>   over 85 comparable of 119 rows — **PASS: no node-count or certified-objective
>   drift.** 1914.2 s wall, load start 0.17 peak 2.03. 20 non-comparable rows
>   reported and not gating.
> - `pytest -m smoke python/tests`: **844 passed, 16 skipped, 2 xpassed** (identical
>   to Phase 1's counts). `pytest -m smoke discopt_benchmarks/tests`: **51 passed,
>   1 skipped**. `pytest -m slow python/tests/test_adversarial_recent_fixes.py`:
>   **10 passed**. `ruff check` / `ruff format --check`: clean on all 12 changed
>   files.
> - **Isolation note.** A concurrent session began editing this working tree
>   (Phase 3 Card 3a's `tightening_schedule`) at 12:08 UTC, after the panel and the
>   `python/tests` smoke run had completed on the clean tree. The adversarial suite,
>   the benchmark smoke suite and the lint were **re-run in a `git worktree` pinned
>   to `f557b056`**, with the child asserting the Card 2c.1 marker present and the
>   foreign `tightening_schedule` import absent before running. Nothing in this
>   phase's commits contains that session's work (verified against the commit stat).
>
> **Two findings that are not Phase 2's, surfaced by Phase 2's panel.**
> (a) `contvar` blows well past its budget on **both** trees (>400 s against a 45 s
> limit, interleaved A/B, and Phase 1's own `--check` already recorded
> `contvar: child_timeout`), so it joins `heatexch_gen3` as a budget-overrun
> instance wanting its own issue. (b) The Rust presolve orchestrator terminates on
> `IterationCap` for 48 of 119 instances — it never reaches a fixpoint inside 16
> sweeps — while `simplify` reports row removals on every sweep. Worth a look
> alongside 3d.

### Card 2a — `cascade_aux` at all six sites

Documented graduated-ON (`obbt.py:1884-1895`) but wired at 1 of 6 call sites
(`root_reduce.py:393-404`; missing at `solver.py` root/per-node/incumbent OBBT,
`lp_spatial_bb.py:502`, `disjunctive_config_bound.py:241`). Resolve the flag once
(via `SolverTuning`) and pass it everywhere. **Regime C** — the original
graduation measured one site; re-run the `ab_cascade_aux.py` gate with all sites
live on `global50` + `heldout50`. If net-negative at the extra sites, keep those
sites off *explicitly* (a `cascade_aux=False  # measured net-negative, <ref>`
argument), not implicitly.

### Card 2b — Retire the per-node Python Jacobian FBBT (entry experiment first)

`_tighten_node_bounds_with_status` (`solver.py:2362`, O(m·n²) Python triple loop)
runs at every node on both Python B&B paths (`solver.py:8973`, `:12762`)
immediately before Rust `in_tree_presolve`, which should subsume it from the exact
DAG. Given G-A (per-node cost is the dominant gap), this is the highest-leverage
small deletion in the repo — but §0.2 makes it Regime C.

- **Entry experiment:** instrument both mechanisms on the 61-corpus node streams;
  count nodes where the Python pass tightens *strictly beyond* the Rust pass's
  fixpoint (print executed-comparison count). **Kill criterion:** if >0.5% of nodes
  show Python-only inferences, do NOT delete — port the missing inference class to
  `in_tree_presolve` first, then re-run.
- On pass: delete the per-node calls (keep the function for its two setup-path
  uses, or inline them), Regime C panel (deleting an active tightening), plus
  per-node wall-time delta reported.

### Card 2c — Stop discarding computed tightenings

Two verified compute-then-discard defects:
1. `_declared_box_tightening` (`solver.py:6368-area`): full 17-rule pass whose
   tightened box is used only for a warning and an infeasibility boolean, then
   recomputed at `:6950` where it *is* applied. Restructure to run **once**: apply
   the result, keep the infeasibility check. Regime N target (identical fixpoint ⇒
   identical nodes); if the panel shows drift, the two passes were not identical —
   stop and diagnose before shipping (that would be a live ordering bug worth its
   own issue).
2. Rust orchestrator model rewrites dropped (`propagate_bounds_to_model` copies
   bounds only; `_root_presolve.py:206-214`). **Entry experiment:** count
   instances on the corpus where `simplify`/`coefficient_strengthening`/
   `redundancy` rewrites change anything beyond bounds. If ≥ a handful, implement
   repr-level adoption (solve from the presolved repr, keeping the postsolve
   chain — the `DISCOPT_PRESOLVE_SUBSTITUTE` machinery is the pattern); else
   remove the three passes from the default list and record the measurement.
   Either branch is Regime C.

---

## Phase 3 — One tightening pipeline (the presolve consolidation)

> **Status:** PARTIAL — **CLOSED 2026-07-29 by owner decision.** 3a(a), 3b and 3c
> LANDED; **3a(b) and 3d DEFERRED-BY-OWNER** (see below). Phase 4 starts next and
> takes exclusive access to `solver.py`.
> **Depends:** Phase 2 (its measurements decided what survives).
> **Gated against:** `reports/panel_baseline_f154dcff.json` — still the current
> baseline. Phase 2 did **not** re-baseline: its own verification block records
> `--check` PASS with 255 comparisons on `f557b056`, i.e. Card 2c.1's declared-box
> intersection turned out node-neutral on the panel, so the Phase 0 artifact
> remains the gate. Phase 3's final `--check` on `acd5feaf` reproduces it exactly:
> **255 comparisons, PASS**, 1944.3 s, load start 0.57 peak 2.43.
>
> ### Why this phase closes here — the finding that re-sequenced it
>
> Across Phases 2–3 the **mechanical** redundancy proved real and condensed
> cleanly: seven boolean parse idioms became one truth table (1a), ~4,900 lines of
> provably-dead passes and orphan modules went (1b). But **every semantic
> redundancy that was actually measured turned out not to be redundant**:
>
> | card | the claimed duplicate | what the measurement found |
> |---|---|---|
> | 2b | Rust `in_tree_presolve` subsumes the per-node Python Jacobian FBBT | **two distinct inference classes** the kernel lacks — 278/1,495 nodes (18.6 %) |
> | 2c.1 | "the identical 17-rule pass runs twice" | **not the same computation** — different input box, model state and position relative to dispatch |
> | 3b | `fbbt` and `fbbt_fp` are the same FBBT, keep the faster | **no winner** — different, complementary fixpoints; composing beats either |
>
> So the tightening layer is a **differentiated stack that was undocumented**, not
> an accreted duplicate. The right deliverable there is the schedule object plus
> the documentation of what each stage is for and why it is skippable — which is
> Card 3a(a), landed — **not** deletion. That reframing is why 3a(b) and 3d are
> deferred rather than merely unfinished.
>
> ### Landed
>
> - **3a(a) — `TighteningSchedule` declared and enforced.**
>   `python/discopt/_jax/tightening_schedule.py` declares four schedules
>   (root 9 stages, NLP-BB root 1, spatial node 6, NLP-BB node 2 — **18 stages**),
>   each with its anchor, its gate in the source's own terms, and a soundness note.
>   An AST conformance test asserts the anchors occur in the declared order inside
>   the host functions; `record()` at each site makes `schedule.explain()` report
>   the real last-run verdict. Regime N by construction — nothing moved.
>   The AST probe earned its keep immediately (it rejected a wrong first draft
>   ordering) and surfaced **three stages §2.3's listing never named**: the
>   pre-factorable-reform FBBT, the Phase C incumbent-cutoff OBBT and the Phase C3
>   incumbent-cutoff FBBT.
> - **3b — entry experiment ran; premise falsified, no winner.** 11,088 executed
>   bound comparisons. Nothing deleted, nothing adopted; the real finding is
>   **Card 3e**. §6 has the full attribution.
> - **3c — standing parity test landed.** 175 decided nodes, 5,612 bound
>   comparisons, **0 violations on all three hard invariants** (contraction,
>   soundness floor, kernel monotonicity); pooled Card-2b asymmetry 5.6 %.
>
> ### DEFERRED BY OWNER — 3a(b), the Regime-C de-duplications
>
> Single `tighten_root_bounds_with_fbbt` invocation policy; one Python
> reduced-cost fixer; OBBT entry points 3 -> 1. **Not started, and not to be
> started as consolidation work.** Rationale: the table above. Every semantic
> de-duplication this plan proposed and then measured was wrong about the
> mechanisms being duplicates, so these three are to be treated as *hypotheses to
> falsify first*, not cleanups to perform.
>
> **What a future session needs to pick this up cold.** The prerequisite is now in
> place: the root schedule is declared, so the sites are enumerable rather than
> hunted. It already shows the card's premise is doubtful — there are **three**
> distinct `tighten_root_bounds_with_fbbt` stages with **different gates**, not one
> repeated call:
> `pre_factorable_fbbt` (`solver.py:5905`, gated `has_factorable_work(model)`,
> runs *before* the factorable reform so its interval checks see finite bounds,
> issue #138), `root_fbbt` (`:7038`, unconditional, last step before tree creation,
> budget-capped per #863) and `nlp_bb_root_fbbt` (`:12543`, at `_solve_nlp_bb`
> entry). Run the Card 2c.1-shaped entry experiment — do the calls see the same
> input box and the same model state? — **before** writing any consolidation. Each
> arm is Regime C and needs its own differential panel (~32 min on this machine).
>
> ### DEFERRED BY OWNER — 3d, adopt the Rust presolve's model rewrites
>
> **Re-classified: this is gap-closure work belonging next to Phase 5, not
> consolidation work ahead of the modularization.** Its premise is measured and
> stands (Card 2c.2: 8,614 non-bound rewrites on 32/119 instances, zero bound
> contribution), and the card body below is unchanged and still accurate. What
> changed is its *position*: solving from the presolved repr changes what the
> relaxation compiler sees, which is a capability change of the same family as
> Phase 5's kernel coverage expansion, and it should be sequenced against that
> work rather than run before `solve_model` is carved up. Not started here — per
> §0 and the session brief, a partially-wired repr adoption is the most dangerous
> change in this plan and half of it is worse than none of it.
>
> ### Filed, not built
>
> - **Card 3e** — the wired-in `fbbt` pass cannot compose with any other presolve
>   pass's tightenings (measured 0 composed bounds in 7/7 instances against 48 for
>   `fbbt_fp`). Regime C, one-line mechanism (a seed parameter on
>   `fbbt_with_cutoff_until`). Evidence is durable in three places: this card, the
>   §6 Card 3b entry, and the `fbbt_fp.rs` header.
>
> **`heldout50`: SKIPPED — local only.** No MINLPLib snapshot in this environment,
> so every measurement above is corpus-local (119 in-repo instances). All three
> landed cards are Regime N or measurement-only, so the missing generalization arm
> cannot have admitted an unmeasured bound change.

The audit found ~30 mechanisms, 6 FBBTs, 5 reduced-cost fixers, no sequencer beyond
the Rust orchestrator, and `tighten_root_bounds_with_fbbt` invoked from 5
uncoordinated sites.

### Card 3a — `TighteningSchedule` object — **(a) LANDED, (b) DEFERRED-BY-OWNER**

One module (`python/discopt/_jax/tightening_schedule.py` or promote
`presolve_pipeline.py`) declaring the **root schedule** and the **node schedule**
as ordered lists of named stages with their gates, replacing the inline hand-ordering
in `solve_model` (review §2.3 wiring listing). Rules:
- Pure mechanical extraction first (Regime N, exact-node-count) — same stages, same
  order, same gates, now introspectable (`schedule.explain()` prints stage, gate,
  and last-run stats; feeds Card 4a's `explain_routing`).
- Then, separately (Regime C), the de-duplications Phase 2 justified: single
  `tighten_root_bounds_with_fbbt` invocation policy, one reduced-cost-fixing
  implementation for the Python side (delegating to the live MILP-driver kernel
  semantics; delete `node_reduce._dbbt_from_reduced_costs` and
  `solver._reduced_cost_fixing` in favor of it), OBBT entry points 3 → 1
  (`obbt_tighten_root` becomes the single door with modes; `run_obbt` /
  `run_obbt_on_relaxation` become internal).

### Card 3b — `fbbt` vs `fbbt_fp` (entry experiment) — **RAN; premise falsified, no winner**

> **Status:** LANDED 2026-07-29 as a measured NO. The card's premise — "both are
> FBBT, so the fixpoints must be identical; pick the faster, delete the loser" —
> is falsified on both halves. 11,088 executed bound comparisons over 119
> instances: 11 instances disagree on 183 bounds, `fbbt_fp` tighter on 170 and
> `fbbt` tighter on 13, with **neither dominating** (on `st_e03` running both in
> either order beats either alone). `fbbt.rs` was never deletable — `fbbt_fp.rs`
> imports its kernels — and `fbbt_fp` cannot replace the wired-in pass because it
> reports `IterationCap`, not `NoProgress`, on 11 instances, so its result depends
> on `max_iterations`, and it costs 6.2× on `util` for that. `fbbt_fp.rs` stays
> parked with its header rewritten from an unmeasured claim into the measurement.
> Probe: `card3b_fbbt_vs_fbbt_fp_entry.py` (+ `--diagnose`, 22 checks), artifact
> `reports/card3b_fbbt_vs_fbbt_fp.json`. Full attribution in §6.
>
> **What it found instead, and it is bigger than the card:** the wired-in `fbbt`
> pass **structurally cannot compose with any other presolve pass's tightenings**
> (0 composed bounds in 7/7 instances, against 48 for `fbbt_fp`). Filed as Card 3e.

`fbbt_fp.rs:15-20` claimed to supersede the wired-in sweep FBBT ("wasteful…
oscillates in the tail") yet was never enabled. The A/B ran on the corpus root
streams: fixpoint equality and wall time, 3 interleaved replicates.

### Card 3e — the root FBBT pass cannot see the other passes' bounds (filed by Card 3b)

**Measured, not hypothesised** (Card 3b, 119 instances, 14 executed composition
checks). `FbbtPass::run` (`crates/discopt-core/src/presolve/passes.rs`) calls
`fbbt_with_cutoff_until(&ctx.model, …)`, which seeds `var_bounds` from
`model.variables` — the **declared** box — and never reads `ctx.bounds`; the
adapter then intersects the result back in. The orchestrator deliberately never
writes tightened bounds into `ctx.model`'s `VarInfo` (documented at
`orchestrator.rs`: mutating declared bounds can flip an inactive bound active and
change LP duals). The two facts together mean the wired-in FBBT pass **re-derives
the same box on every sweep** and can never propagate from what `eliminate`,
`simplify`, `implied_bounds`, `coefficient_strengthening` or `probing` just
proved.

Evidence: at `max_iterations=1` — one sweep, so no pass can re-run and launder the
result — `[implied_bounds, X]` beats `intersect(X@1, implied_bounds@1)` on **0**
bounds for `fbbt` across 7/7 instances and on **48** bounds for `fbbt_fp`. Every
disagreeing corpus row shows `fbbt[NoProgress, iters=2]`: the orchestrator's
fixpoint loop stops at sweep 2 because sweep 2 reproduces sweep 1 exactly.

**Scope.** Add a seed parameter to `fbbt_with_cutoff_until` (initialise
`var_bounds` as `declared ∩ seed` instead of `declared`) and pass `ctx.bounds`
from `FbbtPass::run`. `fbbt_fp.rs` is the working reference for the seeded form.
**Regime C** — it strictly tightens the root box, so it is bound-changing; ships
behind a default-OFF `DISCOPT_*` flag per §0.7 and graduates on a differential
panel with the #902 quality gate. Do **not** instead adopt `fbbt_fp` wholesale:
Card 3b measured its result to be `max_iterations`-dependent.

### Card 3d — Adopt the Rust presolve's model rewrites (filed by Card 2c.2) — **DEFERRED-BY-OWNER, re-sequenced next to Phase 5**

**Measured, not hypothesised** (Card 2c.2, 119 instances, 8,206 per-pass deltas
examined, `reports/card2c_presolve_rewrites.json`): `simplify`,
`redundancy` and `coefficient_strengthening` contribute **zero bound tightenings**
— so `propagate_bounds_to_model`, the only bridge back to the Python DAG, carries
**nothing** from them — while producing **8,614 non-bound rewrites on 32 of 119
instances (27 %)**: 8,439 rows removed by `simplify` (31 instances), 149 by
`redundancy` (`carton7`, `tanksize`), 26 rows rewritten by
`coefficient_strengthening` (`gbd`, `hda`).

Their output is not *entirely* wasted — the presolved repr is kept and drives the
Rust root FBBT and in-tree FBBT — but nothing reaches the Python relaxation
compiler, which is what builds every node LP. This is also *why* a second, stronger
Python coefficient-tightener exists (`solvers/_root_presolve.py`'s "NOTE ON
LOCATION").

**Scope:** solve from the presolved repr with a postsolve chain, using the
`DISCOPT_PRESOLVE_SUBSTITUTE` machinery as the pattern (§6a names the same
mechanism). Regime C. **Do not** instead delete the three passes: the measurement
says they do real work, just not work that reaches the consumer.

### Card 3c — Node-tightening parity across kernels — **LANDED**

> **Status:** LANDED 2026-07-29. `python/tests/test_node_tightening_parity.py`
> (10 tests: 6 per-instance `slow`, 2 native-kernel `slow`, 1 totals `slow`, 1
> `smoke`). Node streams are captured by wrapping the two real node-tightening
> entry points during real solves, so the boxes are the ones the engines actually
> decide. Measured on the current tree: **175 decided nodes, 5,612 bound
> comparisons, 491 contraction checks, 158 monotonicity checks, 108 soundness
> checks**, 4 spatial-loop + 2 NLP-BB-loop instances.
>
> Two engines branch differently, so their node *streams* cannot be aligned. What
> is compared is the stack **as a function on a box**: for each node,
> `P = Python(B0)`, `S = Kernel(Python(B0))` (what ships) and the counterfactual
> `K = Kernel(B0)`. Four invariants:
>
> | | invariant | kind | result |
> |---|---|---|---|
> | I1 | every stack only shrinks its input box | hard | 0 violations / 491 |
> | I2 | a box containing a known-feasible point still contains it after tightening, on every stack | hard, never to be relaxed | 0 violations / 108 |
> | I3 | `Kernel(Python(B0))` inside `Kernel(B0)` — the kernel is monotone in its input box | hard | 0 violations / 158 |
> | I4 | the Card 2b asymmetry, as a **ceiling** with counts | ledger | pooled 5.6 % (11/175); per-instance ceiling 60 %, applied only above 10 decided nodes |
>
> I4 is deliberately a ceiling, not an equality: Card 2b established that the
> kernel does **not** subsume the Python pass, so equality would fail on the known
> gap and a floor would fail when Phase 5 *closes* it. The test fails on a **new**
> divergence — a kernel edit that starts losing inferences it used to make.
>
> The I2 witness is built by variable **name** against the evaluator's own model,
> so a reformulated column order cannot make the soundness check silently vacuous;
> auxiliary coordinates with no reported value are skipped rather than assumed.
>
> **Two measured findings recorded for Phase 5.**
> (a) The **native spatial kernel served ZERO solves** on both arms of the
> end-to-end test (`nvs05`, `st_e05`; 2 producer calls, 0 served) — the producer
> declined and the ON arm ran the Python loop. That arm therefore proves the
> *fallback* is certificate-safe, not that the kernel agrees, and the test prints
> that rather than reading the agreement as parity. This is review §2.5.2 ("the
> Python fallback is load-bearing") measured, and it is exactly the population
> Phase 5's coverage census must rank.
> (b) `propagate_spec_fixpoint` has **no PyO3 binding**, which is *why* the native
> kernel cannot be compared box-by-box at all. Phase 5 should add one and upgrade
> this test's native arm from end-to-end to the box-level comparison; the harness
> is already written and shared.
>
> The MILP driver — the fourth stack — ships `node_propagation=false`, so it
> decides no node boxes on defaults. Rather than ignore it, a `smoke` test asserts
> that default straight off `solve_milp_py.__text_signature__`, so the day it
> graduates ON this file fails and forces the driver into the comparison.
>
> `ex1264`/`ex1263` were dropped from the instance list after measuring **zero**
> decided nodes at this budget — their arm of the comparison was vacuous. The
> totals test is what keeps that from recurring silently: it fails on zero decided
> nodes, zero bound comparisons, or a run that exercised only one Python loop.

Per review §2.5.1, the Python spatial path, the native kernel, and the MILP driver
run different node-tightening stacks with nothing asserting equivalence.

---

## Phase 4 — Routing made explicit (the solver.py decomposition, part 1)

> **Status:** **4a LANDED**, **4b PARTIAL (1 of 5 modules)**, **4c NOT STARTED**
> (2026-07-29). **Depends:** Phase 1 (flag helper). **Regime:** N throughout.
>
> **What exists now (4a).** `python/discopt/routing.py` declares **29 routes** —
> every dispatch gate in `solve_model`, in the order the function evaluates them —
> each with the gate *verbatim in the source's own terms*, the handler it
> dispatches to, why it exists, and (for 12 of them) what makes it decline.
> `python/tests/test_routing.py` (14 `smoke` tests) is what makes the declaration
> authoritative rather than decorative:
>
> | test | what a regression looks like |
> |---|---|
> | markers occur in the declared order | a gate added / deleted / moved / renamed |
> | one `entered()` per route, closure both ways | an undeclared gate, or a route nobody records |
> | declared handler is called in the route's own region | the declaration drifting into fiction |
> | **the three #740/#748 guards are still in the source** | a fall-through turned back into an early `return` |
> | the two branch-level fall-throughs still call `fell_through()` | the runtime walk silently losing a decline |
>
> The last two are soundness tests, not style. `_solve_milp_bb`, `_solve_miqp_bb`
> and `_solve_nlp_bb` neither receive nor consult `lazy_constraints` /
> `incumbent_callback`; for a lazy constraint the callback *defines* the feasible
> set, so an engine that never consults it would accept a point outside it and
> certify it. `class_milp`, `class_miqp` and `nlp_bb_auto` therefore match their
> gate and decline, and the guards are pinned at source level.
>
> Recording is one line per route at branch entry (`_rt.entered(...)`) plus a
> `_rt.fell_through(...)` at each branch-level #748 decline — a thread-local dict
> write with no control-flow effect. The open/close bracket is a **decorator**
> (`_routing_walk`), which covers all 64 early returns without touching a single
> `return`. `discopt.explain_routing(model, **kw)` and
> `Model.solve(explain_routing=True)` render the walk with each gate's real
> verdict, composing Card 3a's `tightening_schedule.explain()` so one call shows
> both the engine chosen and the tightening stages inside it.
>
> **One design decision worth recording.** The marker says `entered`, not
> `taken`, and *which route dispatched* is derived in `explain()` as the last
> route to record (a dispatching branch returns, so nothing after it can record).
> Marking entry as "taken" would have been an instrument reporting a result it
> never measured: 11 of the 29 routes run a second, finer classification inside
> the branch (`classify_gp`, a convexity check, an engine that may return `None`)
> and continue on when it declines. The naive version printed **three** routes as
> "TAKEN" on a small MINLP.
>
> **What exists now (4b).** `python/discopt/solver.py` is now the package
> `python/discopt/solver/` (`git mv`, rename recorded, so `git log --follow` is
> preserved) and the first module is carved out: **`solver/native_kernel.py`,
> 796 lines** — the eight-function #764 native-spatial-kernel cluster (engagement
> gate, feature safety, seed construction + rigorous seed verification, the
> driver) plus its three constants. Provably a pure move: 7 of the 8 moved
> functions are **AST-identical** to their pre-move selves and the 8th differs
> only by the deferred `_unpack_solution` import that breaks the package cycle;
> all **124** stay-behind functions are AST-identical.
>
> `solver/__init__.py` is **17,569 lines** (from 18,211 at the head of Phase 4).
> `solve_model` is unchanged at **7,622 lines** and remains the largest function.
> The card's "no file > 2,500 lines / no function > 300 lines" target is **not**
> met and cannot be met by moving whole functions — see the §6 coupling census.
>
> **Verification.** 4a and 4b were panel-gated **separately, each on its own
> tree**, against `reports/panel_baseline_f154dcff.json`; numbers in §6.
>
> **What remains, exactly.**
> - **4b:** the four other modules the card names (`setup`, `reformulate`,
>   `root`, `results` — and `spatial_loop`). §6 shows why they are not next:
>   they are *inline statement blocks of `solve_model`*, not functions, sharing a
>   closure of 200+ locals, so moving one is a signature-design problem rather
>   than a move. The tractable next step is a **`solver/_common.py` leaf-helper
>   layer** (`_unpack_solution`, `_pack_solution`, `_extract_variable_info`,
>   `_gap_converged`, `_decompose_eq_slack_form`, the shared constants), which
>   drops the engine functions' module-level dependency counts into single digits
>   and unblocks `milp.py` (`_solve_milp_bb` + `_solve_miqp_bb` +
>   `_solve_milp_simplex`, ~1,500 lines) and `matrix_backends.py`.
> - **4c:** not started. `lp_spatial_bb.py`, `gp/solve_gp_minlp` and
>   `signomial_global` still reimplement node selection with raw `heapq`.

### Card 4a — Dispatch table extraction

`solve_model`'s ~20 sequential gates (review §2.1; dispatch tree in the audit)
become an ordered registry of `Route(name, predicate, handler, gate_reason)` in
`python/discopt/routing.py`, evaluated in declared order. Strictly mechanical:
predicates and order are copied, not redesigned; the three #740/#748 soundness
fall-throughs stay fall-throughs (encoded as `Route(..., fallthrough=True)`).
Regime N exact-node-count. Deliverables: `Model.solve(explain_routing=True)` (or
`discopt.explain_routing(model)`) printing the route walk with each gate's verdict
— this is also the debugging tool every later graduation panel wants.

### Card 4b — Carve `solve_model` into modules

With routing extracted, split the monolith along its existing phase banners into
`solver/` submodules: `setup.py` (validation, classification, convexity cache),
`reformulate.py` (the 9 reformulation stages), `root.py` (root presolve/OBBT/cuts,
Card 3a's schedule), `spatial_loop.py` (the 2,470-line inline loop → the 5th
Python loop file, honestly named), `results.py`. Target: no file > 2,500 lines, no
function > 300 except the loop body (its own follow-up). Pure moves + imports;
Regime N; `git log --follow` preserved by moving whole functions. Multiple PRs,
one module each — **never one big-bang PR**.

### Card 4c — Three stray B&B loops onto `PyTreeManager`

`lp_spatial_bb.py`, `gp/solve_gp_minlp`, `signomial_global` reimplement node
selection with raw `heapq`. Port each to `PyTreeManager` (same selection policy →
Regime N per loop; where the local policy differs, either adopt it as a
`PyTreeManager` option or match existing behavior exactly and note the diff).
Value: every certificate-critical pruning decision now flows through one audited
tree manager. If Phase 5 retires `lp_spatial_bb`'s class first, skip its port.

---

## Phase 5 — Kernel convergence (the G-A architecture gap; now unblocked)

> **Status:** **5.1 LANDED** (the coverage census — the ranking every other
> sub-card is sequenced against), **5.4 PARTIAL** (the misroute guard landed; the
> flag does **not** graduate — its deciding population is not in this
> environment), **5.2 / 5.3 OPEN** (2026-07-29). **Depends:** Phase 0; Card 3c
> (the safety net). **Est:** the long pole; many sessions, incremental by design.
>
> ### What exists now (5.1 — the census)
>
> `discopt_benchmarks/scripts/phase5_kernel_coverage_census.py` +
> `reports/phase5_kernel_coverage_census_c346fd73.json`, built on decline reason
> codes added to the producer (`_jax/spatial_producer.py`: 18 coded sites,
> `producer_stats()`) and to the engagement path (`solver/native_kernel.py`: 8
> gate contracts + 9 driver outcomes, `kernel_engagement_stats()`). The producer
> is **tapped at its real call site** — the presolved root box — because a static
> pre-filter reading declared bounds gets the answer wrong on exactly the
> instances that matter (#902 measured it dropping `tanksize`).
>
> **119 in-repo instances, 45 s, `DISCOPT_NATIVE_SPATIAL_KERNEL=1`. Executed: 119
> rows, 82 feature-gate calls, 82 producer calls, 61 declines, 20 served, 0
> unclassified.**
>
> | bucket | inst | baseline wall (s) | % wall | unsolved at 45 s |
> |---|---|---|---|---|
> | `producer` (a relaxation feature the kernel cannot build) | 61 | 1328.8 | 69.0 % | 21 |
> | `served` (the kernel solved it) | 20 | 397.7 | 20.6 % | 9 |
> | `never_reached` (another route dispatched first) | 37 | 197.2 | 10.2 % | 2 |
> | `driver` (spec built, answer rejected) | 1 | 2.3 | 0.1 % | 0 |
>
> **Ranked producer gaps — this ordering, not intuition, sequences 5.2:**
>
> | # | decline code | inst | wall (s) | % | notes |
> |---|---|---|---|---|---|
> | 1 | `term_trilinear` | 17 | 437.6 | 22.7 % | nvs*/bchoco*/st_e3*/ex1252* — needs a trilinear `EnvTerm` in Rust |
> | 2 | `infinite_aux_bounds` | 9 | 379.6 | 19.7 % | 4stufen, beuster, carton7, contvar, ex14_1_9, gear4, hda, heatexch_gen1/2 |
> | 3 | `probe_real_shape_mismatch` | 3 | 158.9 | 8.2 % | casctanks, st_e04, st_e35 |
> | 4 | `probe_objective_bound_invalid` | 4 | 155.3 | 8.1 % | the whole `tspn*` family — the G-F `no_bound` class |
> | 5 | `fixed_row_box_dependent:coeffs` | 9 | 109.7 | 5.7 % | an unmodelled box-dependent envelope family; the guard is working, the coverage is missing |
> | 6 | `blf_row_count` (5 or 6) | 7 | 45.0 | 2.3 % | #861 over-claiming; the Python incremental engine already solves this **numerically** (`incremental_mccormick._select`) and the producer only declines |
> | 7 | `term_ratio` | 3 | 9.6 | 0.5 % | gear, gear2, gear3 |
> | 8 | `term_univariate:{log,exp,cos,sin}` | 6 | 11.4 | 0.6 % | one atom each |
>
> **The census falsifies Card 3c's headline framing, and that matters for
> sequencing.** Card 3c recorded "the native spatial kernel served ZERO solves"
> — true of its 2-instance end-to-end arm (`nvs05` declines `term_trilinear`,
> `st_e05` declines `blf_row_count:6`), and *not* true of the corpus: the kernel
> already serves **20 of 119** instances carrying 20.6 % of the baseline wall
> (dispatch, fuel, nvs13/17/18/19/20/23/24, prob06, st_e06/07/13/17/18/27/31,
> st_ph10, tanksize, util). The coverage problem is real and is 69 % of the wall
> — but it is an expansion problem, not a from-zero problem, and Card 3c's parity
> test should draw its native arm from the **served** list above so that arm stops
> being vacuous.
>
> ### What exists now (5.4 — the misroute guard; the flag stays OFF)
>
> The graduation blocker Phase 5.4 names is the misroute class. Its mechanism is
> now measured, reproduced in-repo, fixed and guarded — see the §6 entries dated
> 2026-07-29 (`Phase 5.4 entry experiment` and `Phase 5.4 fractional budget`).
> Summary: `Model.solve` gave the kernel `min(time_limit, 120)` s and then handed
> `solve_model` the caller's **full** budget again, so an eligible-but-uncertifiable
> model paid the attempt *on top of* its whole default budget —
> `clay0303hfsg` at a 10 s budget: ON **25.24 s** (sd 0.12) vs OFF 13.52 s
> (sd 0.05). `_convex_kernel.last_attempt_seconds()` now publishes the attempt
> wall and `Model.solve` deducts it (16.01 s post-fix, excess over OFF
> −80.6 %), with `python/tests/test_phase5_convex_kernel_budget.py` as the
> standing guard (it fails on the pre-fix arithmetic).
>
> **The flag does not graduate, and the reason is population, not evidence
> quality.** Measured over all 119 in-repo instances, `build_convex_spec` accepts
> exactly **four**: `clay0303hfsg`, `cvxnonsep_psig40r`, `syn05hfsg`, `syn05m`.
> The family this flag exists for is 136 `.nl` files in the MINLPLib snapshot and
> the deciding counter-case (`watercontamination0202`) is snapshot-only. Both are
> recorded as **SKIPPED — local only**. What the in-repo panel *can* and does
> settle is the other direction — that turning the flag on does not harm the 115
> instances it never routes — and the four it routes; the verdict is in §6.
>
> ### What 5.2 should take next
>
> Not `term_trilinear` first despite its rank: it is a new Rust envelope family
> (patcher + differential fixtures + parity extension). The cheap, general,
> already-solved-elsewhere row is **#6, `blf_row_count`** — the producer declines
> models whose lifted model constraints share a term's support, while
> `IncrementalMcCormickLP._select` identifies exactly the same envelope rows
> *numerically* on the probe box and has done since #861. Porting that matcher is
> a Python change worth 7 instances, and it is the prerequisite shape for every
> later family. `term_trilinear` is the right *second* card, and it is where the
> wall is.

The review's central finding: the 50–500× per-node interpreter cost is the dominant
wall-clock gap, the machinery to fix it (native spatial kernel, `spatial_propagate`)
exists, and the hard tail is structurally routed to the slow path via
`_native_kernel_feature_safe` declines. Strategy: **expand kernel coverage
feature-by-feature, graduating each expansion on the panel**, exactly as
#764/#865/#879/#904 have been doing. This phase is a pipeline, not a single card:

1. **Coverage census (measurement card):** over the corpus + heldout50, tabulate
   *why* the producer declines each instance (`_native_kernel_feature_safe` reason
   codes — add them if not present). Output: ranked list of missing features by
   instance-count and by wall-clock at stake. This ranking, not intuition, orders
   the sub-cards.
2. **Per-feature sub-cards** (template): implement the feature in the kernel
   producer/relaxation, Regime C panel with the #902 quality gate, extend Card 3c's
   parity test. Known candidates from the docs: richer node-LP row families (the
   trusted build's separable objective floor, convex-lift OA — the
   `issue-764` "quality ratchet" list), more propagation atoms + ordering,
   violation/reliability branching in-kernel.
3. **Retirement dividend:** each coverage expansion shrinks the population served
   by the Python spatial loop. When a class fully migrates, delete its
   special-case handling from the Python loop (Regime C, since it deletes active
   code). This — not hand-refactoring — is how the 2,470-line loop body actually
   gets smaller.
4. **Convex kernel graduation (G-C, cheapest large win — can run first and in
   parallel):** `DISCOPT_CONVEX_KERNEL` certifies `rsyn*`/`clay*`/`syn05*` in
   seconds; blockers were verification wiring (#779/#798), coverage has widened
   (#865/#879). Run the §5 panel both directions — including the measured
   misroute guard (`watercontamination0202` classifies convex and then runs 2001 s
   with no bound: the kernel budget + certified-only adoption + fallback must make
   this class strictly safe). Graduate default-ON or record what blocks it.

---

## Phase 6 — Missing mechanisms (gap closure beyond the kernel)

> **Status:** OPEN. **Depends:** Phase 0; independent of Phases 3–4; 6b after 5.4.
> **Est:** 3+ sessions per card; each card is optional-but-valuable on its own.

### Card 6a — Presolve reduction scale (G-G class)

SCIP presolves `watercontamination0202` 106,712 → 566 vars and solves in 2.56 s;
discopt returns nothing, and reduces nothing on 64% of a 300-instance census.
`DISCOPT_PRESOLVE_SUBSTITUTE` exists default-OFF and once produced a 2449% primal
gap on this class — so the *mechanism* exists and the *quality* failed.
Entry experiment: on the G-B instance list, run substitution-to-fixpoint and
measure (a) reduction ratio, (b) certified-answer parity on the reduced model.
Root-cause the 2449% incident (it is a postsolve or substitution-validity bug —
find it, don't tune around it). Kill criterion: if reductions on this class stay
<2× after the fix, the SCIP 189× is presolver tech discopt doesn't have — record
it and file the specific missing pass (aggregation/stuffing) as its own card.

### Card 6b — CSE / defined-variable sharing through the stack

The IR gap table row: parser-level CSE exists (`DISCOPT_DISABLE_CSE`), but sharing
is discarded downstream — duplicated into the modeling DAG, JAX compile, lifted LP,
and every FBBT sweep. Entry experiment: measure shared-subexpression multiplicity
on the 20 largest corpus instances (parser stats vs DAG node counts). Build only
if multiplicity is material (kill: <1.5× median duplication). Implementation goes
in the *kernel-facing* path (Phase 5's substrate), not the Python evaluator being
retired.

### Card 6c — Aggregation / c-MIR cuts (G-E) — **sequenced last deliberately**

Measured: SCIP's aggregation separator takes nvs17 6,796 → 70 nodes; discopt's
single-row GMI/MIR plateaus. Also measured: root-only cutting at certifying
intensity starved incumbents (#781), the prior c-MIR attempt was a NO-GO, and
OBBT × cuts × throughput are **multiplicative — they must land together**. Do not
start this card until Phase 5 provides cheap nodes to multiply against. Then:
multi-row aggregation separator in Rust (root + in-tree, pool with aging), Regime
C with root-gap (Phase 0.3) as the primary metric and the #781 incumbent-starvation
check as a gate arm.

### Card 6d — Process floor (G-D; product work)

42% of the BARON gap attribution is the ~513 ms import/startup floor. Tasks:
lazy-import audit (`DISCOPT_EAGER_IMPORTS` exists — invert the default where
Regime N proves neutrality), a `discopt solve` CLI fast path that defers JAX until
a path needs it (the #905 JAX-free-gate pattern), benchmark harness defaults to
the warm daemon. Metric: cold `python -c "import discopt"` and cold single-instance
solve wall, before/after, interleaved (§0.5).

---

## Phase 7 — Product shape: islands, docs, and the long tail

> **Status:** OPEN. **Depends:** none (can run anytime). **Regime:** N.
> **Est:** 1–2 sessions.

1. **Tier the islands explicitly.** `mo/`, `ro/`, `stochastic/`, `bilevel/`,
   `dae/`, `interfaces/` (+ `mpec.py`) — zero inbound imports, 41 of 71
   `NotImplementedError`s. Add a support-tier statement to each package docstring
   and a docs page: Tier 1 (certified core, correctness-gated) vs Tier 2
   (functional islands, API may change, not covered by solver gates). Promote or
   demote nothing silently.
2. **The stochastic invalid-cuts raise** (`multistage.py:47`) and the gdpopt_loa
   UNSOUND caveat (`gdpopt_loa.py:628`): verify each refusal actually triggers on
   the documented inputs (a test per refusal — refusals are load-bearing safety
   guards and currently untested).
3. **SOTA long tail — explicitly deferred, with entry experiments filed as
   issues, not built:** restarts, conflict analysis, orbital fixing
   (`symmetry.rs` is parked for this), parallel tree search. The tanksize ablation
   (§764: cuts/OBBT off made SCIP *faster*) is the standing warning against
   staffing features because SOTA solvers have them. Each issue names the corpus
   class that would prove the feature matters and the kill criterion.

---

## §5. Sequencing summary

```
Phase 0 (baseline + heldout panel)
  ├─► Phase 1 (hygiene: flags, dead code)        [independent cards]     DONE
  ├─► Phase 2 (wiring defects)  ─► Phase 3 (3a(a)+3b+3c only)            DONE
  │        └─► Phase 4 (routing + decomposition)  ◄── NEXT, exclusive solver.py
  ├─► Phase 5 (kernel convergence; 5.4 convex-kernel graduation may run first)
  │        └─► Card 3d (adopt the presolved repr) — MOVED here from Phase 3
  ├─► Phase 6a/6b/6d (presolve scale, CSE, floor) [6c strictly after Phase 5]
  └─► Phase 7 (islands/docs/deferred-tail)       [anytime]
```

**Re-sequencing, 2026-07-29 (owner).** Phase 4 now follows Phase 3's *landed*
cards rather than a complete Phase 3: 3a(b) and 3d are deferred (see the Phase 3
block), so nothing else in Phase 3 needs `solver.py` and Phase 4 can take it
exclusively. **Card 3d moves adjacent to Phase 5** — adopting the presolved repr
changes what the relaxation compiler sees, which is gap-closure of the same family
as kernel-coverage expansion, not consolidation to be done ahead of the
modularization. Card 3e (the FBBT composition defect) is unscheduled and may run
any time after Phase 4 releases `solver.py`; it touches `crates/` only.

Parallelism guidance for Opus sessions: Phases 1, 2, 7 are safe concurrently
(disjoint files). Phases 3 and 4 both edit `solver.py` — serialize them. Phase 5
touches `crates/` + producer files — safe alongside 1/2/7, coordinate with 3c
(whose parity test is the guard Phase 5's expansions must keep green).

## §6. Falsification log (append-only, per §0.3)

### 2026-07-29 — Card 2a: "the graduated cascade should be on at all six sites" — **FALSIFIED**

**Hypothesis.** `DISCOPT_OBBT_CASCADE_AUX` graduated default-ON in #208 but is
resolved at 1 of 6 `obbt_tighten_root` sites; wiring the other five should reproduce
the #208 verdict (cert-clean and net-positive) at those sites too.

**Kill criterion.** The §0.1 Regime-C gate: cert-clean AND net-positive AND the
#902 incumbent-quality gate.

**Experiment 1 — differential panel.** `card2a_cascade_aux_panel.py`, 119 in-repo
instances (both corpus dirs), 45 s, defaults, cascade live at all six sites, against
the frozen Phase 0 baseline `panel_baseline_f154dcff.json` (whose reproducibility on
this tree Phase 1's `--check` had just established, 255 comparisons PASS).
The child taps `obbt_tighten_root` and records caller `file:line` + the `cascade_aux`
it received, so "no change" cannot be confused with "never ran".

- **Sites actually engaged** (487 cascade-ON calls at the new sites):
  `solver.py:7145` root OBBT — 79 calls / 79 instances; `solver.py:10987`
  incumbent-improvement OBBT — 91 / 70; `solver.py:9042` per-node OBBT — 317 / 12;
  `root_reduce.py:400` (the pre-existing site) — 82 / 48.
  **`lp_spatial_bb.py` and `disjunctive_config_bound.py`: ZERO calls** — the corpus
  does not reach them (the latter sits behind a default-OFF flag).
- **cert-clean: PASS**, 0 violations over **578 executed comparisons**
  (objective agreement 87, optimality regression 87, bound-vs-proven-oracle 67,
  certification 117, quality 103, node 117).
- **quality-clean: FAIL** — `ex1252` incumbent 134263.585 → 204321.632 (+52 %).
- **Net:** nodes 37,110 → 37,139 (+29, +0.1 %) over 117 paired rows; certifications
  +1 (`nvs18`) / −0 ; 22 rows changed node count.

**Experiment 2 — attribution, interleaved replicated A/B vs the pre-card tree**
(`git worktree` at `HEAD~1`, each child asserting `discopt.__file__` and the
presence/absence of the `obbt_cascade_aux` marker before solving, arms interleaved
per replicate, load 0.09–0.43):

| instance | pre-card | all sites ON | per-node OFF | per-node+incumbent OFF |
|---|---|---|---|---|
| ex1252 obj | 134263.585 (3/3) | 204321.632 (3/3) | 204321.6 / 134471.6 (unstable) | 134471.559 (2/2) |
| ex1252a obj | 177860.742 (3/3) | 134263.587 (3/3) | 134263.587 | 134263.587 |
| ex1252a bound | 14086.222 | **4.257e-319** | 4.257e-319 | 4.257e-319 |
| nvs06 nodes | 5 | 3 | 3 | 5 |
| st_e36 nodes | 85 | 83 | 83 | 85 |
| st_e04 nodes | 103 | 97 | 103 | — |

So: the **incumbent-improvement site** owns the `nvs06`/`st_e36` node gains *and*
`ex1252`'s 52 % primal regression; the **per-node site** owns `st_e04` alone (317
calls over 12 instances to buy 6 nodes); the **root site** alone still collapses
`ex1252a`'s dual bound 14086 → ~0. Every row above reproduced exactly across its
replicates.

**Verdict.** Sound everywhere, helpful essentially nowhere: −10 nodes of 37,110 on
the certifying population against a reproduced 52 % incumbent regression and a
reproduced dual-bound collapse. This is the `DISCOPT_CUT_INHERIT` lesson again —
cert-clean but not net-positive stays off. All five sites ship `cascade_aux=False`
**explicitly**, with the measurement at the call site; the two zero-call sites are
off as *unmeasured* (§0.1 forbids shipping an unmeasured bound change), not as
measured-harmful.

**What did land:** one resolution point (`SolverTuning.obbt_cascade_aux`), five
accidental defaults turned into five documented decisions, and a standing AST test
that makes the omission a failure. Behaviourally the tree is identical to the
pre-card tree at all five sites, so `panel_baseline_f154dcff.json` remains the gate.

**Not attributable to this card:** the panel's `contvar` ON-arm `child_timeout`.
Interleaved A/B kills **both** trees at >400 s (2 replicates each), and Phase 1's
own `--check` log already recorded `contvar: child_timeout`.

### 2026-07-29 — Card 2b: "Rust `in_tree_presolve` subsumes the per-node Python Jacobian FBBT" — **FALSIFIED, deletion abandoned**

**Hypothesis.** `_tighten_node_bounds_with_status` (O(m·n²) Python, every node, both
Python B&B loops) is redundant against the Rust kernel that runs immediately after
it from the exact DAG.

**Kill criterion (from the card).** >0.5 % of nodes showing a Python-only inference
⇒ do not delete.

**Experiment.** `card2b_fbbt_subsumption_entry.py`. Today a node's box becomes
`Rust(Python(B0))`; after the deletion it becomes `Rust(B0)`. The probe computes
**both**, per node, on real node streams — legitimate because Rust
`in_tree_presolve` takes `&self` and is a pure function of (repr, box, depth,
incumbent), so the counterfactual arm cannot perturb the search it observes. A
Python-only inference is a coordinate of `Rust(Python(B0))` strictly inside
`Rust(B0)` at 1e-12 relative, or a node Python fathomed that `Rust(B0)` does not.
25 instances drawn from the Phase 0 baseline by `node_count > 0 and not
convex_fast_path`, covering both loops, 20 s each.

- **executed: 1,458 node counterfactuals + 37 fathom counterfactuals = 1,495 decided
  nodes; 82,316 individual bound comparisons.**
- **Python-only inference nodes: 278 / 1,495 = 18.60 %** — 37× the kill criterion.
  Also 3 Python-only fathoms and 186 unmatched boxes (counted, not dropped).
- Concentrated but not exotic: `flay03m` 212/218 nodes, `clay0303hfsg` 39/78,
  `syn05hfsg` 9/170, `m3` 8/86, `tanksize` 6/17, `st_e05` 3/55, `nvs22` 1/27.
- Typical shape — Python *pins* a variable the Rust kernel leaves open:
  `[0, 1.4387] → [1.4387, 1.4387]`, `[7.5, 10] → [10, 10]`.

**Which half is missing** (attribution re-run over the 7 affected instances, 648
node counterfactuals, 47,460 bound comparisons): of 274 Python-only bound nodes,
**147 come from the Jacobian linear-row FBBT alone**, **83 from the structural /
interval nonlinear rules alone** (`_apply_nonlinear_tightening_with_status`), **44
from both**. So the Rust kernel is missing *two* distinct inference classes, not
one, and the larger is the Jacobian-linearized row FBBT with its structural-
linearity mask.

**Verdict.** Do not delete; both per-node call sites stay. Porting is not the small
job the card's "port it if small" branch anticipated — it is two mechanisms
(a Jacobian-sampled linear-row FBBT with the #27a structural mask, and the 17-rule
`tighten_nonlinear_bounds` set including `DefinedVariableForwardRule`). It belongs
with Phase 3's one-tightening-pipeline work / Phase 5's kernel convergence, where
the Rust node kernel is the substrate being extended anyway. **The G-A per-node cost
finding stands, but this deletion is not the way to collect it.**

### 2026-07-29 — Card 2c.1: "the identical 17-rule pass runs twice" — **FALSIFIED (premise), consolidated differently**

**Claim under test** (plan Card 2c.1 / review §2.3): `_declared_box_tightening`
discards its box, then "the identical pass re-runs" later and is applied; running it
once should be Regime-N.

**Falsification (static, verified against the tree).** The two calls are not the
same computation:

| | first call | second call |
|---|---|---|
| input box | `flat_variable_bounds(model)` — the **declared** box | the working box, **after** `propagate_bounds_to_model` and the Rust root FBBT |
| model state | pre-dispatch, pre-materialization | post-dispatch; the model may have been mutated (builder rows materialized) or **rebound** (`factorable_reformulate(clear_only=True)` on the convex fast path) |
| deadline | a fresh 15 %-of-budget slice | the solve's absolute `_solve_deadline` |
| position | **before** LP/QP/convex dispatch, several of which return without ever reaching the second call | after it |

Neither can replace the other: the first must precede dispatch because the paths
that return early depend on its infeasibility proof, and the second sees a strictly
tighter input box. Running "once" as written would have been a behaviour change, not
a refactor.

**What shipped instead.** The *discard* is what was wrong, and that is fixed: the
first call's tightened box is intersected into the root working box before the
second pass, guarded by (a) `model` object identity — the one place the model can be
replaced — and (b) array-shape equality, which catches any reformulation that
changed the variable count. Sound in both directions: every mutation between the two
points only adds rows (feasible set shrinks, so a valid box stays valid), and an
intersection can only tighten. An empty intersection is routed through the existing
infeasible return rather than introducing a new terminal path. The payoff is not a
saved pass — it is that the earlier pass's work survives when the second is
truncated by its #875 deadline, which on the large-model class is exactly when it
truncates.

### 2026-07-29 — Card 2c.2: "the dropped Rust presolve rewrites are near-zero" — **FALSIFIED; filed as Card 3d, not built**

**Experiment.** `card2c_presolve_rewrites_entry.py` builds each corpus repr exactly
as `solve_model` does and runs the orchestrator with `solve_model`'s own arguments,
tabulating per-pass deltas. **119 instances, 8,206 per-pass deltas examined, 0
errored.**

| pass | invocations | rows removed | rows rewritten | **bounds tightened** | instances with non-bound rewrites |
|---|---|---|---|---|---|
| `simplify` | 912 | 8,439 | 0 | **0** | **31** |
| `redundancy` | 912 | 149 | 0 | **0** | 2 (`carton7`, `tanksize`) |
| `coefficient_strengthening` | 912 | 0 | 26 | **0** | 2 (`gbd`, `hda`) |

**32 of 119 instances (27 %) carry a non-bound rewrite**, and all three passes
tighten **zero** bounds — so `propagate_bounds_to_model`, the only bridge into the
Python DAG, carries literally nothing from them. Both halves of the card's decision
rule are therefore answered: removing the passes is wrong (they do real work), and
adoption is the fix. Per the card's own instruction, adoption is **not built here**;
it is filed as **Card 3d** with these counts.

**Incidental:** the orchestrator terminates on `IterationCap` for 48/119 instances
(never reaching a fixpoint in 16 sweeps), `NoProgress` 67, `TimeBudget` 3,
`Infeasible` 1.

### 2026-07-29 — Card 3b: "`fbbt` and `fbbt_fp` are both FBBT, so their fixpoints must be identical; pick the faster and delete the loser" — **FALSIFIED on both halves**

**Hypothesis (the card's own).** `fbbt_fp.rs:15-20` claims to supersede the
wired-in sweep FBBT ("wasteful … oscillates in the tail") yet was never enabled.
A/B the two on the corpus; fixpoint equality is *expected* (both are FBBT over the
same DAG, and `fbbt_fp` literally imports `forward_propagate` /
`backward_propagate` from `fbbt`), so any bound difference is a bug in one of them.
The faster becomes the only Rust DAG-FBBT; the loser is deleted.

**Kill criterion.** A fixpoint disagreement that is *not* a bug in either kernel
kills the "pick a winner, delete the loser" plan, because the two passes are then
not substitutes.

**Experiment.** `discopt_benchmarks/scripts/card3b_fbbt_vs_fbbt_fp_entry.py`. Both
passes run **alone**, through the same orchestrator entry (`PyModelRepr.presolve`),
from the same repr, same tolerance (1e-8), `max_iterations=16`; 3 interleaved
A/B/A/B replicates per instance; 119 in-repo corpus instances; load 0.18 at start,
1.35 at end. Artifact `reports/card3b_fbbt_vs_fbbt_fp.json`.

- **executed: 11,088 bound comparisons over 119 instances, 0 errored.**
- **11 instances disagree on 183 bounds** — `fbbt_fp` tighter on 170, `fbbt`
  tighter on 13. Not equality, and not one-directional.
- Wall on the **104 both-converged** instances: `fbbt` 0.617 s vs `fbbt_fp`
  0.253 s summed — **ratio 0.410**, i.e. the watch-list pass is 2.4× faster
  *where it converges*.

**Attribution** (`--diagnose`, 22 executed checks). Three separate causes, none of
which is a soundness bug in either kernel:

| # | cause | evidence |
|---|---|---|
| 1 | **The wired-in `fbbt` pass cannot compose.** `FbbtPass::run` calls `fbbt_with_cutoff_until(&ctx.model, …)`, which seeds from the **declared** box and never reads `ctx.bounds`; the adapter then intersects. The orchestrator deliberately never writes tightened bounds back into `ctx.model` (documented in `orchestrator.rs`, for LP-dual validity), so the pass re-derives the same box every sweep. | At `max_iterations=1` (so no pass can re-run and launder the result), `[implied_bounds, X]` beats `intersect(X@1, prior@1)` on **0** bounds for `fbbt` across 7/7 instances, and on **48** bounds for `fbbt_fp`. Every disagreeing row shows `fbbt[NoProgress, iters=2]` — the signature of a pass computing the same thing every sweep. |
| 2 | **Fixpoint order-dependence on cyclic nonconvex DAGs.** FBBT has a unique greatest fixpoint only on monotone/linear systems. Both boxes are valid outer approximations; **neither dominates**. | `st_e03` block2.lo: `fbbt` 0.012676, `fbbt_fp` 0.012381, `fbbt→fp` **0.013482**, `fp→fbbt` 0.013468 — composing beats *either* singleton. This is the mechanism `fbbt_fp`'s own header cites (Belotti–Cafieri–Lee–Liberti 2010). |
| 3 | **Float accumulation at ~1e-10 relative** from a different visit order (`ex1252`, `ex1252a`, `hda`), below any decision threshold. | e.g. `ex1252` block10.hi 385.8840406063522 vs 385.8840406893461. |

**A third finding, fatal to adopting `fbbt_fp` as-is.** Its header claims it
"terminates the moment the queue is empty — that's the true fixed point, no
`max_iter` artefact". Inside the orchestrator it reports **`IterationCap`, not
`NoProgress`**, on 11 instances: per-sweep `bounds_tightened` plateaus at 35
(`4stufen`) / 89 (`util`) and never reaches zero — asymptotic sub-tolerance drift.
Its output therefore **depends on `max_iterations`**, and it costs **6.2×** the
sweep pass on `util` (261 ms vs 42 ms) to buy that drift.

**Verdict — no winner; nothing deleted, nothing adopted.** `fbbt.rs` was never
deletable (`fbbt_fp.rs` imports its kernels; `fbbt_with_cutoff` backs the root FBBT
and `in_tree_presolve`), and `fbbt_fp` cannot replace the `fbbt` pass while its
result is budget-dependent. `fbbt_fp.rs` stays parked, with its header rewritten
from an unmeasured claim into this measurement, and with the note that it is the
working reference implementation of a *seeded, composing* DAG-FBBT.

**What the card actually found, and it is bigger than the card.** The wired-in FBBT
pass — the one every solve runs — **cannot see any other presolve pass's
tightenings**. `eliminate`, `simplify`, `implied_bounds`, `probing` and
`coefficient_strengthening` all tighten `ctx.bounds`; `fbbt` then re-derives from
the declared box and intersects. That is a defect in exactly the "one tightening
pipeline" this phase exists to fix, it is general (no instance keying), and the fix
is a seed parameter on `fbbt_with_cutoff_until`. Filed as **Card 3e** (Phase 3);
Regime C, not built here.

### 2026-07-29 — Card 3c: "the four node-tightening stacks reach equivalent fixed points, or the kernel's are tighter" — **CONFIRMED where measurable, with two gaps recorded**

**Claim under test** (review §2.5.1). The Python spatial loop, the Python NLP-BB
loop, the native Rust spatial kernel and the MILP driver run different per-node
tightening stacks and nothing asserts they agree. The card's invariant: the boxes
agree, or the kernel's are tighter — **never looser**.

**Experiment.** `python/tests/test_node_tightening_parity.py`, a standing test
rather than a script. Node streams are captured by wrapping the two real node
entry points during real solves, so the boxes compared are the ones the engines
actually decide. Per node: `P = Python(B0)`, `S = Kernel(Python(B0))` (what
ships), `K = Kernel(B0)` (counterfactual — sound because `in_tree_presolve` takes
`&self` and is a pure function of repr/box/depth/incumbent).

- **executed: 175 decided nodes, 5,612 bound comparisons, 491 contraction checks,
  158 monotonicity checks, 108 soundness checks**; 4 spatial-loop and 2
  NLP-BB-loop instances (the totals test fails if either loop is unexercised).
- **I1 contraction — 0 violations / 491.** No stack ever grows a box.
- **I2 soundness floor — 0 violations / 108.** A box containing a known-feasible
  witness still contains it after every stack. The witness is matched by variable
  **name** against the evaluator's own model, so a reformulated column order
  cannot make the check silently vacuous.
- **I3 kernel monotonicity — 0 violations / 158.** `Kernel(Python(B0))` is inside
  `Kernel(B0)`; without this, "the kernel is at least as tight" is unprovable.
- **I4 the Card 2b asymmetry** — pooled 5.6 % (11/175 nodes) on this subset, in
  line with Card 2b's 18.6 % over a larger sample. Encoded as a **ceiling with
  counts**, not an equality: equality would fail on the known gap and a floor
  would fail when Phase 5 *closes* it.

**Two measured gaps recorded for Phase 5, not papered over.**

1. **The native spatial kernel served ZERO solves** on both end-to-end arms
   (`nvs05`, `st_e05`; 2 producer calls, 0 served): `_native_kernel_feature_safe`
   declined and the ON arm ran the Python loop. So that arm currently proves the
   *fallback* is certificate-safe, not that the kernel agrees — and the test says
   so in its output rather than reading the agreement as parity. This is review
   §2.5.2 ("the Python fallback is load-bearing, not legacy") measured, and it is
   exactly the population Phase 5.1's coverage census must rank.
2. **`propagate_spec_fixpoint` has no PyO3 binding**, which is *why* the native
   kernel cannot be compared box-by-box at all. Phase 5 should add one and upgrade
   this test's native arm from end-to-end to the box-level comparison; the harness
   is already written and shared.

**Incidental:** `ex1264`/`ex1263` were dropped from the instance list after
measuring **zero** decided nodes at the test's budget — their arm of the
comparison was vacuous, the §6-of-CLAUDE.md failure mode. The totals test now
fails on zero decided nodes, zero comparisons, or a single-loop run, so that
cannot recur silently.

### 2026-07-29 — Phase 3 close-out: what was verified on the final tree, and what was not

Recorded because §0.6 asks for what was run, and because a partially-verified tree
described as verified is the same defect as an instrument that measures nothing.

**Verified on the final tree `acd5feaf`:**
- `panel_baseline.py --check reports/panel_baseline_f154dcff.json`:
  **comparisons executed: 255** (node_count 85, certified objective 85, status 85)
  over 85 comparable of 119 rows — **PASS: no node-count or certified-objective
  drift.** 1944.3 s, load start 0.57 peak 2.43; 19 non-comparable rows reported
  and not gating.
- `pytest python/tests/test_tightening_schedule.py`: 12 passed.

**Verified on `4f78b8cc`** (the tree one commit earlier; the only later change is
`acd5feaf`, a dict-key correction confined to a `record()` keyword argument):
- `pytest python/tests/test_node_tightening_parity.py`: 10 passed.

**Verified on `14a40a05`** (Card 3a(a)'s tree, before 3b/3c):
- `pytest -m smoke python/tests`: **856 passed, 16 skipped, 2 xpassed**
  (844 before this phase; the +12 are `test_tightening_schedule.py`).

**Verified on the edited Rust sources:** `cargo test -p discopt-core`
**536 + 4 + 1 passed** (the only Rust change in this phase is a comment block).

**NOT re-run on the final tree, and the owner's close-out instruction was to start
no further measurement:** `pytest -m smoke python/tests`,
`pytest -m smoke discopt_benchmarks/tests`, and
`pytest -m slow python/tests/test_adversarial_recent_fixes.py`. The residual risk
is bounded — between `14a40a05` and `acd5feaf` the only `solver.py` change is the
one-key `record()` fix, and the Regime-N panel (which solves all 119 corpus
instances end to end) **did** run on the final tree and passed with 255
comparisons — but the smoke and adversarial suites should be re-run at the head of
Phase 4 before any new `solver.py` edit, and this note is here so that is not
forgotten.


### 2026-07-29 — Phase 4 Step 0: Phase 3's three un-re-run suites, re-run on the untouched tree

Phase 3's close-out flagged three suites it could not re-run on its final tree and
asked that they run at the head of Phase 4 before any new `solver.py` edit. Done,
on `33e06fb6` with nothing modified:

- `pytest -m smoke python/tests` — **857 passed, 16 skipped, 2 xpassed**, 656 s.
  The +1 against Phase 3's recorded 856 is Card 3c's own `smoke` test in
  `test_node_tightening_parity.py`, which landed *after* the tree that 856 was
  measured on; not a new test and not a regression.
- `pytest -m smoke discopt_benchmarks/tests` — **51 passed, 1 skipped** (matches).
- `pytest -m slow python/tests/test_adversarial_recent_fixes.py` — **10 passed**,
  233 s (matches).

Phase 3's residual risk is therefore closed rather than merely bounded.

### 2026-07-29 — Card 4a: dispatch table extraction — **LANDED, Regime N clean**

**Scope check first.** The card says "~20 sequential gates". The AST census found
**29**, because four of them are the `problem_class` sub-branches (LP / QP / MILP /
MIQP) that the review's dispatch tree counts as one. All 29 are declared; the extra
nine are not new behaviour, only previously-uncounted gates.

**Verification.** `panel_baseline.py --check reports/panel_baseline_f154dcff.json`
on the Card 4a tree: **comparisons executed 255** (node_count 85, certified
objective 85, status 85) over 85 comparable of 119 rows — **PASS**, no node-count
or certified-objective drift. 2226.7 s wall, load start 0.20 peak 2.33. 17
non-comparable rows reported and not gating, all budget-dependent
(`status=time_limit`/`feasible`, plus `tls2` certified at 97 % of its budget);
Phase 3's run reported 19 of the same character.
`pytest python/tests/test_routing.py` — 14 passed, executed assertions
`{order: 1, marker: 180, handler: 23, guard: 6, recorder: 12}` = 222.

**A finding worth keeping.** The first version recorded each gate as `taken` at
branch entry and rendered the first such record as "the route that dispatched". On
a 2-variable MINLP it printed **three** routes as TAKEN (`auto_gp`, `nlp_bb_auto`,
`spatial_branch_and_bound`), because 11 of the 29 routes run a second, finer
classification *inside* the branch and continue when it declines. The marker is now
`entered` and the dispatcher is derived as the last route to record — sound because
a dispatching branch returns, so nothing after it can record. This is CLAUDE.md §6
applied to a debugging instrument: the naive version would have answered "which
engine solved this?" with a number it never measured.

### 2026-07-29 — Card 4b entry experiment: "`solve_model` splits into five modules by pure moves" — **FALSIFIED for four of the five; one module landed**

**Hypothesis (the card's own).** With routing extracted, `solve_model` splits along
its phase banners into `setup.py` / `reformulate.py` / `root.py` / `spatial_loop.py`
/ `results.py` as **pure moves plus imports**, Regime N, one module per PR, target
no file > 2,500 lines and no function > 300 lines.

**Kill criterion.** A candidate whose move is not expressible as relocating whole
functions — i.e. one that would require inventing a parameter list or a context
object — is not a "pure move" and falsifies the card's method for that module.

**Experiment 1 — what the five named modules actually are.** AST census of
`solve_model` on the Phase-4 head tree (`33e06fb6`, before Card 4a):
**7,576 lines, 318 top-level statements, 64 `return`s (63 of them early), 558
`if`s, and one 2,495-line inline `while` loop.** Four of the five
named modules (`setup`, `reformulate`, `root`, `spatial_loop`) are *inline
statement blocks of `solve_model`*, not functions: they read and write a shared
closure of 200+ locals (`_mc_lp_relaxer`, `_root_lb_snapshot`, `rust_time`,
`_gap_certified`, …). Moving one is a signature-design problem — decide what
crosses the boundary, in which direction, and prove the choice bound-neutral — not
a `git mv` of a `def`. `results.py` is the same shape: the return block reads ~40
locals accumulated across the whole function.

**Experiment 2 — coupling census over the movable *functions*.** For each
top-level function that is a plausible module member, the number of
`solver.py`-level names it references (constants, sibling helpers, imports),
computed by AST with local bindings subtracted:

| function | LOC | module-level deps |
|---|---|---|
| `_solve_nlp_bb` | 1,256 | **44** (including `solve_model` itself — a cycle) |
| `_solve_milp_bb` | 632 | 28 |
| `_solve_miqp_bb` | 500 | 23 |
| `_solve_continuous` | 230 | 21 |
| `_solve_qp_matrix` | 193 | 16 |
| `_solve_milp_simplex` | 354 | 13 |
| **native-kernel cluster (8 fns)** | **694** | **1** (`_unpack_solution`) |

Every engine function reaches 13–44 sibling names, so moving one alone creates an
import cycle back into `solver`. The native-kernel cluster is the single exception
— its dependency closure touches exactly one solver-level helper — which is why it
is the module that landed.

**Experiment 3 — the hazard the card did not anticipate: monkeypatch seams.** A
census over `python/tests` and `discopt_benchmarks` found **30+ distinct
`discopt.solver` attributes that tests replace** (`_solve_batch_pounce` ×6,
`_solve_milp_simplex` ×4, `_solve_lp_matrix` ×4, `_root_cover_cut_loop` ×4,
`_native_kernel_seed` ×2, …). A pure move plus a re-export leaves
`monkeypatch.setattr(discopt.solver, "_f", fake)` **succeeding and doing nothing**:
the moved call site resolves `_f` in its own module. `raising=True` does not help,
because the re-exported attribute exists. At least one such seam is a *negative*
assertion (`must_not_run` in `test_expired_outer_deadline_skips_native_seed_work`)
which would then pass vacuously — a CLAUDE.md §6 instrument-measures-nothing
failure introduced by a refactor.

**Rule adopted, and applied to the landed module:** a moved function that any test
*patches* is **not** re-exported from `discopt.solver`, so a stale patch raises
`AttributeError` instead of silently no-op'ing; a moved function that tests only
*call* may be re-exported using the explicit `x as x` form. Under that rule
`_native_kernel_seed` and `_native_kernel_seed_candidates` are not re-exported and
their three test sites were repointed at `discopt.solver.native_kernel`. **Any
future 4b move must run this census for its own cluster.**

**What landed, and its proof of purity.** `solver.py` → `solver/__init__.py` (git
rename recorded) plus `solver/native_kernel.py` (796 lines). **7 of the 8 moved
functions are AST-identical** to their pre-move selves; the 8th
(`_try_native_spatial_kernel`) differs by exactly three added lines, the deferred
`from discopt.solver import _unpack_solution` that breaks the package cycle. All
**124** stay-behind functions are AST-identical.

**Verdict.** The card's method holds for *function clusters* and is falsified for
the four inline-block modules. The next tractable step is **not** one of the card's
five names: it is a `solver/_common.py` leaf-helper layer, which drops the engine
functions' dependency counts into single digits and unblocks `milp.py` (~1,500
lines) and `matrix_backends.py`. Carving the inline blocks needs `solve_model`'s
locals turned into an explicit state object first — a larger, separate piece of
work that should be scoped on its own rather than smuggled into a "pure move" card.

### 2026-07-29 — Phase 5.1: "the native spatial kernel serves nothing; coverage starts from zero" — **FALSIFIED; it already serves 20 of 119**

**Claim under test.** Card 3c's recorded finding, inherited by this phase as its
premise: *"the native spatial kernel served ZERO solves on both end-to-end arms
(nvs05, st_e05; 2 producer calls, 0 served)"*. Phase 5.1's census was commissioned
to rank *why* the producer declines — which presupposes it mostly declines.

**Instrument.** Reason codes did not exist, so they were added first (Regime N: a
thread-local dict write in place of each bare `return None`, every branch returning
exactly what it returned before): 18 coded producer sites
(`_jax/spatial_producer.py`, `producer_stats()`), 8 feature-gate contracts and 9
driver outcomes (`solver/native_kernel.py`, `kernel_engagement_stats()`). The
census (`discopt_benchmarks/scripts/phase5_kernel_coverage_census.py`) taps the
producer at its **real call site**, i.e. on the presolved root box after root FBBT
+ OBBT — deliberately not a static pre-filter, which #902 measured dropping
`tanksize` because the filter sees declared bounds and the producer sees presolved
ones. Each child exits the moment its classification is decided, so a decline costs
root-setup wall rather than a full budget; wall-at-stake is therefore joined from
the frozen `reports/panel_baseline_f154dcff.json`, never from the census run.

**Executed: 119 rows, 82 feature-gate calls, 82 producer calls, 61 declines, 20
served, 0 unclassified.** 753 s wall, load 1.10 → 3.10. Artifact
`reports/phase5_kernel_coverage_census_c346fd73.json`.

**The falsification.** The kernel serves **20 of 119** instances (dispatch, fuel,
nvs13, nvs17, nvs18, nvs19, nvs20, nvs23, nvs24, prob06, st_e06, st_e07, st_e13,
st_e17, st_e18, st_e27, st_e31, st_ph10, tanksize, util) carrying **397.7 s = 20.6 %**
of the baseline wall. Card 3c's "zero" was true of its two instances and false of
the corpus: `nvs05` declines `term_trilinear` and `st_e05` declines
`blf_row_count:6`, so its native arm happened to draw two decline cases out of a
population where 1 in 6 is served. Recorded per CLAUDE.md §11 — Card 3c's own
statement was that the *arm* proved the fallback safe rather than the kernel, which
was scrupulous; what this falsifies is the phase-level premise that coverage starts
from zero. **Consequence for Card 3c:** its `test_native_spatial_kernel_agrees_end_to_end`
parametrization should be drawn from the served list, or its arm stays vacuous.

**The ranking (the deliverable).**

| bucket | inst | wall (s) | % | unsolved |
|---|---|---|---|---|
| `producer` | 61 | 1328.8 | 69.0 % | 21 |
| `served` | 20 | 397.7 | 20.6 % | 9 |
| `never_reached` | 37 | 197.2 | 10.2 % | 2 |
| `driver` | 1 | 2.3 | 0.1 % | 0 |

| # | producer decline code | inst | wall (s) | % |
|---|---|---|---|---|
| 1 | `term_trilinear` | 17 | 437.6 | 22.7 % |
| 2 | `infinite_aux_bounds` | 9 | 379.6 | 19.7 % |
| 3 | `probe_real_shape_mismatch` | 3 | 158.9 | 8.2 % |
| 4 | `probe_objective_bound_invalid` | 4 | 155.3 | 8.1 % |
| 5 | `fixed_row_box_dependent:coeffs` | 9 | 109.7 | 5.7 % |
| 6 | `blf_row_count:{5,6}` | 7 | 45.0 | 2.3 % |
| 7 | `term_ratio` | 3 | 9.6 | 0.5 % |
| 8 | `term_univariate:{log,exp,cos,sin}` | 6 | 11.4 | 0.6 % |

`never_reached` splits `nlp_bb_auto` 10 / `class_milp` 13 / `class_miqp` 10 / four
singletons — i.e. 10.2 % of the wall is routing, not kernel coverage, and no
coverage work touches it. **`heldout50`: SKIPPED — local only** (no MINLPLib
snapshot in this environment), so the census covers the 119-instance in-repo union
only.

**Two incidental findings, recorded rather than fixed.** (a) `prob10` is the single
`driver:objective_disagreement` row — the producer built a spec, the kernel solved,
and the driver rejected its own answer because the recomputed objective disagreed
beyond 1e-4 relative. That is the #789 guard working, and it is also a bug lead.
(b) The entire `tspn*` family declines on `probe_objective_bound_invalid`, which
means the G-F `no_bound` family and the kernel-coverage gap are the *same*
population on this corpus.

**A by-product #902 should read, labelled as the weak evidence it is.** The census
ran `DISCOPT_NATIVE_SPATIAL_KERNEL=1` at the *same* 45 s budget the frozen Phase-0
baseline used with the flag off, so on the 20 served rows the two artifacts form an
unreplicated, non-interleaved ON/OFF pair. Cross-checked (27 executed comparisons,
sense-aware, oracle from `utils.reference_optima`): **0 objective regressions, 0
bounds past a proven reference**, and **4 certifications gained** —

| instance | OFF (baseline, flag off) | ON (census, flag on) |
|---|---|---|
| `st_e31` | `feasible` 47.9 s | **`optimal` 6.4 s** |
| `nvs18` | `feasible` 29.8 s | **`optimal` 7.2 s** |
| `nvs20` | `feasible` 45.8 s | **`optimal` 8.9 s** |
| `util` | `feasible` 46.1 s, obj 1059.32 | **`optimal` 15.4 s, obj 999.58** |

plus two strict incumbent improvements under ON (`nvs19` −1097.6 → −1098.2 against
`=opt=` −1098.4; `util` above). This is **not** a graduation verdict: the arms are
not interleaved, not replicated, and come from two different runs on a machine
whose load differed. It is a reason for #902's re-graduation panel to be run, and a
prediction of what it should find.

### 2026-07-29 — Phase 5.4 entry experiment: "the convex-kernel misroute is an instance quirk" — **FALSIFIED; it is the budget arithmetic, and it reproduces in-repo**

**Hypothesis.** `watercontamination0202`'s counter-case (classifies convex in 2.9 s,
then runs 2001 s with no bound vs 49 s spatial; `sota-parity-analysis-2026-07-27.md`
§3 G-C) is not instance-specific. `Model.solve` gives the kernel
`min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET=120)` s, adopts the result **only**
when it certifies, and then calls `solve_model` with the caller's **full**
`time_limit` again (`modeling/core.py`) — the elapsed attempt is never deducted. So
*any* eligible-but-uncertifiable model pays its whole default budget **plus** the
attempt.

**Kill criterion (the card's own).** If no eligible in-repo instance's ON-arm wall
materially exceeds its budget while OFF's does not, the hazard is not reproducible
here and the fix is re-scoped rather than built.

**Experiment.** `discopt_benchmarks/scripts/phase5_convex_kernel_budget_entry.py`.
Eligibility is **measured, not assumed** — `build_convex_spec` over all 119 in-repo
instances accepts exactly four (`clay0303hfsg`, `cvxnonsep_psig40r`, `syn05hfsg`,
`syn05m`; 119 executed probes, 0 errored). Arms interleaved within each replicate,
2 replicates, one subprocess per (instance, arm), each child asserting the arm the
loaded module reports rather than the arm requested.

*Round 1, 45 s budget (16 solves, 4 paired comparisons, load 0.25 → 2.78):*

| instance | OFF wall | ON wall | Δ | statuses |
|---|---|---|---|---|
| `clay0303hfsg` | 46.66 s | 42.53 s | −4.13 s | OFF feasible×2 → **ON optimal×2** |
| `cvxnonsep_psig40r` | 16.29 s | 17.45 s | +1.16 s | optimal both |
| `syn05hfsg` | 23.40 s | 1.13 s | **−22.27 s** | optimal both |
| `syn05m` | 3.07 s | 0.92 s | −2.14 s | optimal both |

→ **hazard KILLED at 45 s**, and the flag looks strongly positive on the eligible
set (one certification gained, one 20.7× speedup).

*Round 2, 10 s budget — a budget the kernel provably cannot meet on `clay0303hfsg`
(16 solves, load 2.07 → 1.63):*

| instance | OFF wall | ON wall | Δ |
|---|---|---|---|
| `clay0303hfsg` | 13.52 s (sd 0.05) | **25.24 s (sd 0.12)** | **+11.71 s on a 10 s request** |
| `cvxnonsep_psig40r` | 11.10 s | 11.21 s | +0.10 s |
| `syn05hfsg` | 10.68 s | 1.31 s | −9.37 s |
| `syn05m` | 3.34 s | 0.94 s | −2.40 s |

→ **hazard CONFIRMED**, reproduced in both replicates, 2.5× the stated limit. The
mechanism is general and the two rounds together say precisely what it is: the
overrun appears exactly when the attempt *fails*, which a 45 s budget on this
corpus never provokes and a 10 s budget always does.

**Fix.** `_convex_kernel.last_attempt_seconds()` publishes the attempt wall — the
spec build **included**, because `build_convex_spec` is the convexity
classification and on this class it is 2.34 s (`clay0303hfsg`) / 1.16 s
(`cvxnonsep_psig40r`), not a rounding error — and `Model.solve` subtracts it from
`_primary_tl`. It is **exactly 0.0** when the flag is off (reset on entry; the
clock starts after the flag check), so the default path subtracts a literal zero
and every deadline below it is bit-identical: Regime N on defaults, by construction
rather than by hope.

**Measured after:** `clay0303hfsg` at 10 s goes **25.24 → 16.01 s**; excess over
OFF **+11.71 → +2.27 s (−80.6 %)**. The residual +2.27 s is the one-off convexity
classification, which is not removable without declining before classifying.
Standing guard: `python/tests/test_phase5_convex_kernel_budget.py`, which fails on
the pre-fix arithmetic (24.69 s for an 8 s request, verified by simulating the old
`_primary_tl`) and passes after (13.79 s), and which refuses to pass vacuously —
it asserts the attempt was actually declined.

### 2026-07-29 — Phase 5.4: "cap the kernel attempt at a fraction of the budget" — **FALSIFIED before it was built**

**Hypothesis (this card's first design).** The clean guard for the misroute class
is a fractional share: give the kernel `min(f·T, 120)` s so the trusted default
path always keeps `(1−f)·T`. `f = 0.35` was the candidate, matching the existing
`_fb_reserve = 0.35 * time_limit` precedent in the same function.

**Kill criterion.** If the measured attempt cost of any instance the flag *helps*
exceeds `f·T` at the panel budget, the cap buys safety by destroying the win, and
the design dies.

**Experiment (attempt-cost decomposition, 120 s cap, one process per instance).**

| instance | parse | `build_convex_spec` | `solve_convex_tree` | total | kernel status |
|---|---|---|---|---|---|
| `clay0303hfsg` | 0.01 s | 2.34 s | **39.55 s** | **41.9 s** | `optimal`, 211 nodes |
| `cvxnonsep_psig40r` | 0.00 s | 1.16 s | 0.00 s | 1.16 s | `exhausted` at the root, bound −inf |
| `syn05hfsg` | 0.00 s | 0.92 s | 0.01 s | 0.93 s | `optimal`, 2 nodes |
| `syn05m` | 0.00 s | 0.75 s | 0.02 s | 0.77 s | `optimal`, 3 nodes |

**Verdict — killed.** `clay0303hfsg` needs **~93 %** of a 45 s budget to certify, so
every `f < 0.93` turns the corpus's only certification win (OFF `feasible` → ON
`optimal`) back into `feasible`. The cap was dropped rather than shipped as a knob
defaulting to a no-op (CLAUDE.md §3: no dead flags). What ships instead is the
deduction alone, which bounds the *contract* (`solve(time_limit=T)` stays ≈ T)
without touching the attempt itself.

Two things this measurement also settles. (a) `cvxnonsep_psig40r`'s +1.16 s ON-arm
delta at 45 s is **exactly** its spec-build cost — the kernel declines at the root
in 0.00 s — so the "producer probe overhead" on the non-routed population is the
convexity classification and nothing else. (b) The docs' "`clay0303hfsg` certifies
in ~7 s" (plan §5.4, from #882) does **not** reproduce on this machine at this
tree: it is 41.9 s. Recorded per CLAUDE.md §11 for whoever plans against that
number next.
