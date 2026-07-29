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

> **Status:** OPEN. **Depends:** Phase 0. **Est:** 2–3 sessions.

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

> **Status:** OPEN. **Depends:** Phase 2 (its measurements decide what survives).
> **Est:** 3–4 sessions.

The audit found ~30 mechanisms, 6 FBBTs, 5 reduced-cost fixers, no sequencer beyond
the Rust orchestrator, and `tighten_root_bounds_with_fbbt` invoked from 5
uncoordinated sites.

### Card 3a — `TighteningSchedule` object

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

### Card 3b — `fbbt` vs `fbbt_fp` (entry experiment)

`fbbt_fp.rs:15-20` claims to supersede the wired-in sweep FBBT ("wasteful…
oscillates in the tail") yet was never enabled. A/B on the corpus root + node
streams: fixpoint equality (must be identical — both are FBBT; any bound difference
is a bug in one of them, investigate before proceeding) and wall time. Winner
becomes the only Rust DAG-FBBT; loser is deleted. Regime N if fixpoints match
(expected), with the wall delta recorded.

### Card 3c — Node-tightening parity across kernels

Per review §2.5.1, the Python spatial path, the native kernel, and the MILP driver
run different node-tightening stacks with nothing asserting equivalence. Add a
**differential node-tightening test**: for a sample of corpus instances, run N
nodes on each engaged path and assert the propagated child boxes agree (or that
the kernel's are tighter — never looser) within tolerance. This is a standing
test, not a one-off; it becomes the guard that Phase 5's coverage expansion cannot
silently weaken node tightening.

---

## Phase 4 — Routing made explicit (the solver.py decomposition, part 1)

> **Status:** OPEN. **Depends:** Phase 1 (flag helper). **Est:** 3–5 sessions.

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

> **Status:** OPEN — **unblocked by #904**. **Depends:** Phase 0; Card 3c
> strongly recommended first (it is the safety net). **Est:** the long pole;
> many sessions, incremental by design.

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
  ├─► Phase 1 (hygiene: flags, dead code)        [independent cards]
  ├─► Phase 2 (wiring defects)  ─► Phase 3 (tightening pipeline)
  ├─► Phase 4 (routing + decomposition)          [after 1a]
  ├─► Phase 5 (kernel convergence; 5.4 convex-kernel graduation may run first)
  ├─► Phase 6a/6b/6d (presolve scale, CSE, floor) [6c strictly after Phase 5]
  └─► Phase 7 (islands/docs/deferred-tail)       [anytime]
```

Parallelism guidance for Opus sessions: Phases 1, 2, 7 are safe concurrently
(disjoint files). Phases 3 and 4 both edit `solver.py` — serialize them. Phase 5
touches `crates/` + producer files — safe alongside 1/2/7, coordinate with 3c.

## §6. Falsification log (append-only, per §0.3)

*(empty — every entry experiment records its outcome here, pass or kill.)*
