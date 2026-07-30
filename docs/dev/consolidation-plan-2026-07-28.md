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

### Phase 0 addendum — the hardened Regime-N gate (open-ledger item 15, 2026-07-30)

> **Status:** LANDED. **Every Regime-N card from here on invokes the gate as
> specified below**; the pre-item-15 single-shot form is no longer sufficient
> evidence for a bound-neutrality claim.

**Why the gate needed hardening.** Phase 0 shipped a `comparable` filter that
excludes rows whose *terminal status* is budget-dependent. That is necessary and it
is not sufficient. The root-cause experiment (§6, "open-ledger item 15") measured
what the filter cannot see: **the solver's search path is a function of the wall
clock at 78 Python decision sites**, and on `gear2` the one that fires is the root
primal heuristic. `solver/__init__.py:9660` hands `integer_local_search`
`time_budget = min(5.0, 0.15·time_limit)`, and its descent runs
`while improved and time.perf_counter() < deadline` until that wall deadline expires
— measured at **5.02 s consumed of a 5.00 s budget**, i.e. it never converges, it
always runs out. Forcing that budget alone moves the gated node count as a step
function (5.0 s → **3 nodes**, ≤3.0 s → **91**, 0.5 s → **93**) and the default sits
directly on the cliff. `gear2` is `optimal`, `certified`, and done in 15 % of its
budget, so *every* static filter admits it — and it still moves under ambient load.
That is why two runs of the same tree flagged different single instances.

**How a Regime-N card invokes the gate now.**

```bash
python -u discopt_benchmarks/scripts/panel_baseline.py \
    --check reports/panel_baseline_f154dcff.json
```

Nothing else may be running on the box. The defaults are the gate:

| knob | default | what it does |
|---|---|---|
| `--replicates` | 3 | a **flagged** row is re-run this many times *alone*, then adjudicated. Clean rows cost nothing. |
| `--max-transient` | 3 | more than this many environmentally-excused rows fails the RUN as *too noisy to gate*. |
| `--max-load` | 2.0 | refuses to **start** above this 1-minute load average (exit 4). `--allow-load` overrides and stamps the run NOT gate-quality. Honest about its reach: it would *not* have caught either observed failure (run A started at load 0.25 and the contention arrived afterwards) — it catches only "started a panel while something was already running". |
| `--replicates 0` | — | escape hatch to the pre-item-15 single-shot gate. Prints a warning; **not acceptable as card evidence.** |

Adjudication verdicts, and what each means for the card:

* **TRANSIENT** — the flagged row's replicates unanimously reproduce the baseline.
  The first-pass deviation was the container. Does not fail; is **printed in full**
  and must be **pasted into the card's close-out** along with the counts. A card
  that reports a PASS while hiding its transients has not reported its measurement.
* **CONFIRMED** — the replicates unanimously disagree with the baseline. Real
  drift; the card **fails Regime N**. A bound-neutrality violation is deterministic
  (the changed code runs every time), so this is the arm every genuine regression
  lands in — which is why the hardening does not weaken the gate (§0.4). That is
  tested, not argued: `test_check_detects_a_perturbed_node_count` injects a
  one-node perturbation and requires the verdict to be `CONFIRMED` and explicitly
  *not* `TRANSIENT`.
* **NONDETERMINISTIC** — the replicates disagree with *each other*. The instance
  does not reproduce itself, so nothing can be gated on it. **Fails**, under its own
  label; it is never averaged into a pass.

**What a card must paste.** The `comparisons executed (total): N = A first-pass + B
adjudication` line, the flagged/adjudicated/transient counts, the `load start …
peak …` line, and every TRANSIENT row in full. Zero comparisons is a failure, not a
pass (§0.5).

**What this does NOT fix, stated so no later card mistakes it for fixed.** The
solver still decides how much work to do by reading a clock. Adjudication tells you
*whether* a deviation is code-induced; it does not make the tree reproducible. The
real fix is deterministic work budgets (LP iterations / sub-NLP counts) in place of
wall-clock budgets on the root heuristic and its siblings — a solver change, Regime
C, sized as its own card. It is filed as ledger row 15b, not done here.

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

### Card 3a — `TighteningSchedule` object — **(a) LANDED; (b) RAN 2026-07-30 — all three de-duplications answered NO, no code change**

> **Status: (b) COMPLETE as a measured NO (2026-07-30).** All three
> de-duplications the card names were taken to an entry experiment on real corpus
> instances *before* any consolidation was written (§0.3). None of the three
> survived. That is the outcome, not a deferral: per the card's own framing and
> Phase 2/3's standing lesson, "these duplicates keep turning out to be
> differentiated", and forcing a merge to satisfy the card is explicitly
> disallowed. Full write-ups in §6; summary:
>
> | item | hypothesis | verdict | evidence |
> |---|---|---|---|
> | **2. one reduced-cost fixing** | `_dbbt_from_reduced_costs` ≡ `_reduced_cost_fixing`; delete one | **BOTH STAY** | 85 real node-LP inputs, **291 executed integer-column comparisons, 0 disagreements** — but DBBT also tightens **continuous** columns (8 observed) and returns an infeasibility verdict, which RCF cannot express, so deleting it is a Regime-C bound-*loosening* (§0.2); and RCF's only consumer (`_solve_milp_bb`) is unreachable — **147 classifications, ZERO MILP** on the corpus — so the reverse swap cannot be gated. The 291/291 agreement is population-dependent: the two floor different quantities (step vs bound) and *do* differ on a fractional integer `lb` (demonstrated, 3 cases, 1 differs). |
> | **1. one root-FBBT invocation policy** | >1 call site fires per solve, so the root FBBT runs redundantly | **REDUNDANCY IS REAL BUT NOT UNIFORM — no policy** | 66 solves, **94 executed invocations**, 0 failures. Per-solve histogram `{0:7, 1:32, 2:23, 4:4}` — **27 solves invoke it more than once**. Of the 35 second-plus calls, **29 changed nothing** *but* **6 tightened bounds**. A "run it once" policy would therefore drop 6 real tightenings — a Regime-C bound-*loosening*. The only clean win is the 29 no-op passes, which is **wall time with zero certificate content**, and this session's benchmark is for correctness. |
> | **3. OBBT doors 3 → 1** | three doors to one room | **BOTH STAY** | 12 instances, **3,386 executed bound comparisons, 803 disagreements**: `obbt_tighten_root` tighter on 786, **`run_obbt` tighter on 17**. Neither dominates, so collapsing either into the other loses tightenings. They are different *relaxations* (model linear rows vs the McCormick LP envelope), not two spellings. `run_obbt_on_relaxation` is not even type-compatible (it takes a pre-built relaxation object, not a `Model`) and `obbt_tighten_root` already delegates to it internally, so it cannot be made private without breaking `amp.py`'s two call sites. |
>
> **Regime classification, per the card's requirement.** All three would have been
> **Regime C** where they change bounds (item 2 deletes an active tightening; item
> 3 changes which relaxation a caller gets) and Regime N only for the pure
> re-export of an entry point. Since none was consolidated, **no Regime-C gate was
> needed and none was claimed** — the only verification owed is that the tree is
> unchanged, which it is: item (b) landed **zero** production edits, only the three
> entry-experiment scripts and their reports.

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

### Card 3e — the root FBBT pass cannot see the other passes' bounds (filed by Card 3b) — **MECHANISM LANDED, DEFAULT-OFF; GRADUATION REFUSED**

> **Status:** 2026-07-29. The seed mechanism is implemented and shipped behind
> `DISCOPT_FBBT_SEED` (**default-OFF**, `parked`). It does **not** graduate, and the
> graduation panel was **not** run — the entry experiment stopped it first. Below is
> what was measured; the full transcript is the §6 entry of the same date.
>
> **What exists now.** `fbbt_with_cutoff_until_seeded(…, seed: Option<&[Interval]>)`
> starts from `declared ∩ seed`; `fbbt_with_cutoff_until` delegates with `None`, so
> the unseeded path is untouched (asserted bit-for-bit by the new Rust test).
> `FbbtPass.seed_from_ctx` (default `false`) passes `ctx.bounds`. Reachable from
> Python via the new `fbbt_seed_from_ctx` kwarg of `PyModelRepr.presolve`, read from
> `DISCOPT_FBBT_SEED` in `run_root_presolve`. Entry probe:
> `discopt_benchmarks/scripts/card3e_fbbt_seed_entry.py`, artifact
> `reports/card3e_fbbt_seed_entry_be705694.json`.
>
> **Entry experiment, 119 instances, 11,058 executed bound comparisons.**
>
> | | result |
> |---|---|
> | mechanism is a no-op (the kill criterion) | **NO** — 129 bounds tightened on 9 of 119 instances (`util` +76, `heatexch_gen3` +28, `hda` +12, `casctanks` +5, `4stufen`/`beuster`/`st_e03` +2, `st_e11`/`st_e17` +1) |
> | containment invariant (seeded box ⊆ unseeded box) | **VIOLATED on 1 instance** — `casctanks`, 5 bounds looser |
> | end-to-end effect (3 instances, both arms) | `st_e03` and `hda` identical; **`util` 159 → 217 nodes at an identical bound and objective** |
>
> **The containment violation is not a budget artifact, and that is why this stops
> here.** Re-run at 11.25 s / 60 s / 300 s presolve budgets, the split is identical
> every time: OFF terminates `IterationCap` at 16 orchestrator iterations, ON
> terminates **`Infeasible` at iteration 2**. So the seeded pass declares the
> `casctanks` root box *empty*. The end-to-end solve does not currently act on it
> (`DISCOPT_FBBT_SEED=1` returns the same `time_limit` / bound `-90.17862569095436`
> as OFF), so **no false-infeasible ships** — but a presolve that can call a root box
> empty is one consumer away from doing exactly that, and CLAUDE.md §1 does not let a
> flag graduate over that signal.
>
> **Verdict: GRADUATE NO, and the differential panel was deliberately not run.**
> Running a three-hour panel to decide the net-positive question would have been the
> wrong order of work: the card already has (a) an unexplained infeasibility
> declaration and (b) the only end-to-end datum pointing the wrong way. Per §0.3 the
> entry experiment is what decides whether to build further, and it said stop.
>
> **What remains — UPDATED 2026-07-29, the blocker is gone.** The `casctanks`
> `Infeasible` is root-caused and fixed; see **Card 3e-RC** above. It was neither
> hypothesis: not an invalid seed (the distinguishing experiment cleared), and not a
> genuinely empty box (the crossing was **1.1e-16**, against `FEAS_TOL` 1e-6). It was
> the *emptiness test* — a zero-tolerance `is_empty()` in `orchestrator::any_empty`,
> which also fired on `util` (6.8e-13) and, with **no flag set**, on `heatexch_gen3`
> (8.5e-14). With that fixed, the seeded arm terminates `Infeasible` on **0 of 119**
> instances and the containment invariant holds corpus-wide.
>
> So the soundness signal that stopped this card is **cleared**, and the only open
> question is the one the card never got to: **net-positive**. The evidence still
> points the wrong way and is unchanged by the fix — `util` spent 159 → **217 nodes**
> for an identical bound and objective, i.e. the tightening reorders branching without
> improving the relaxation, and `st_e03`/`hda` were byte-identical. That is why the
> §5 differential panel is *now permissible* but still not obviously worth 3 hours:
> the entry data predicts `net-positive: FAIL`. A session picking this up should
> re-run `card3e_fbbt_seed_entry.py` (its containment assertion now passes) and then
> decide whether to spend the panel on a mechanism whose only end-to-end datum is
> negative. `casctanks` is also census rank #3 (`probe_real_shape_mismatch`), so it is
> an instance this plan is going to keep meeting.

### Card 3e-RC — the emptiness test, not the seed: a live false-fathom on the default path — **LANDED**

> **Status:** LANDED 2026-07-29. Card 3e's soundness signal is root-caused, and the
> root cause is **neither** of the two hypotheses the card named. It is a defect in
> shipped code that has nothing to do with `DISCOPT_FBBT_SEED`, fires with **no flag
> set**, and is acted on as a node fathom. Full transcript: the §6 entry of the same
> date. Probe: `discopt_benchmarks/scripts/card3e_infeasible_root_cause.py`,
> artifact `reports/card3e_infeasible_root_cause_ed2da7bd.json`.
>
> **Both hypotheses falsified.** (a) `ctx.bounds` is *not* an invalid seed: the
> distinguishing experiment — the production FBBT kernel run from the **unseeded**
> orchestrator's own certified final box, on its own final model — clears on 118 of
> 119 instances, and the one exception is the same defect, not a seed problem. (b) The
> composed box is *not* genuinely empty to within `FEAS_TOL`: measured corpus-wide,
> **0 instances** have a bound crossing above `FEAS_TOL` (1e-6), while the crossings
> that trigger the abort are **1.1e-16 to 8.5e-14** — seven to ten orders of magnitude
> below tolerance.
>
> **The actual defect.** `orchestrator::any_empty` and `bnb::in_tree_presolve` tested
> emptiness with `Interval::is_empty()` — strict `lo > hi`, **zero tolerance** — while
> `fbbt.rs`, `fbbt_fp.rs` and `probing.rs` all gate on `is_empty_beyond(FEAS_TOL)`,
> whose own doc-comment says the strict form "mistakes that numerical noise for
> infeasibility". The two outliers were the two that **act** on the verdict: the
> orchestrator aborts the whole presolve sweep, and the in-tree kernel sets
> `infeasible`, which the B&B loop treats as a *rigorous fathom* and prunes the
> node's entire subtree (`solver/__init__.py:8457` and `:12254`).
>
> **It fires on the default path.** `heatexch_gen3` terminates root presolve
> `Infeasible` with **`DISCOPT_FBBT_SEED` unset**, on a crossing of **8.53e-14**
> (`[226.7, 226.6999999999999]`), and the in-tree kernel declares that instance's own
> certified root box empty by the same amount. `casctanks` (1.1e-16) and `util`
> (6.8e-13) do it under the flag. Every abort the strict test produced corpus-wide was
> spurious.
>
> **`util` was invisible to Card 3e's own probe** — it terminated `Infeasible` at
> iteration 7 and the containment-only check still scored it as a *net tightening*
> (+76 bounds), so the entry experiment reported 1 affected instance where there were
> 3. This is why the root cause was chased on the corpus rather than on `casctanks`.
>
> **The fix, and why it is not a weakened validation (§0.4).** Two paired changes at
> both sites: declare infeasibility only via `any_empty_beyond(bounds, FEAS_TOL)`, and
> repair sub-tolerance crossings with `repair_subtol_crossings` to
> `[min(lo,hi), max(lo,hi)]` — the smallest interval containing both endpoints, so
> whichever derivation was sound keeps its endpoint and no feasible point is cut. The
> repair is the necessary second half: declining to *declare* emptiness is not enough
> if an inverted interval is still handed to an LP column bound. Loosening a fathom is
> the **sound** direction (the node is explored, not discarded); the unsound direction
> would be to prune more. Both new counters (`subtol_crossings_repaired`,
> `subtol_repaired`) are surfaced through PyO3 so a rising count is visible rather
> than silently absorbed.
>
> **Guard against over-permissiveness.** Two of the four new Rust tests assert that a
> crossing *beyond* `FEAS_TOL` still terminates `Infeasible` / still fathoms. Without
> them the fix would be exactly the tolerance-tweak §0.4 forbids.
>
> **A third site, found by the new test rather than by the corpus.** The probing
> branch of `in_tree_presolve` gates its own infeasibility call on `opts.tol`, so it
> can leave `new_lb > new_ub` by up to `opts.tol` *without* setting `infeasible` — the
> same inverted-box hole on the `DISCOPT_NODE_PROBING` (default-OFF) branch. Closed
> with a repair on the box that actually exits the function, plus
> `probing_branch_cannot_return_an_inverted_box`. With probing OFF the exit repair is
> provably a no-op — FBBT runs on the patched model whose declared box *is* the node
> box, so its result is contained in it, and `max(node_lo, iv.lo) <= min(node_hi,
> iv.hi)` holds whenever both are non-inverted, which the two earlier repairs
> guarantee. That is why the Regime N panel below, which started before this hunk
> landed, still measures the shipped default path.
>
> **Verified.** The corpus probe re-run on the fixed tree, identical population (119
> of 119, 22,116 executed bound comparisons): OFF `Infeasible` **1 → 0**, ON
> `Infeasible` **2 → 0**, instances with a noise crossing reaching the caller
> **3 → 0**, instances with a genuine crossing **0 → 0** (the guard did not become
> permissive), E1 **1 → 0**. `cargo test -p discopt-core` **548 passed** (+6).
>
> **Card 3e itself is unchanged by this: `DISCOPT_FBBT_SEED` stays default-OFF.** The
> containment violation that stopped it was a symptom of *this* defect, so removing it
> makes the flag's net-positive question answerable — but not answered. See the Card
> 3e note below for what remains.

### Card 3d — Adopt the Rust presolve's model rewrites (filed by Card 2c.2) — **NOT BUILT 2026-07-30; entry experiment says the benefit is throughput, and the bound-relevant half already exists elsewhere**

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

> **Status: NOT BUILT, 2026-07-30.** Entry experiment run first per §0.3
> (147 `.nl` files = `minlplib_nl` 66 + `minlplib` 81; 147 three-pass runs +
> 147 full-pass runs, 0 errors; §6 entry "Card 3d entry experiment"). Three
> measured findings, in the order that decides the card:
>
> 1. **The card's residual content is throughput, not bound.** Across all 147,
>    the three indicted passes tighten **zero** bounds — Card 2c.2's headline,
>    reproduced. Their output is 6,635 rows removed by `simplify`, 149 by
>    `redundancy`, and 52 rows rewritten by `coefficient_strengthening`. A row
>    that `redundancy` removes is implied and cannot tighten an LP relaxation;
>    `simplify` moved no bound anywhere on this corpus. So ~99 % of the 8,614
>    rewrites buy *smaller node LPs* — real, but pure wall-clock, which the
>    2026-07-30 owner re-sequencing put off the critical path (same reasoning
>    that moved Card 6d).
> 2. **The one bound-relevant component already exists, in the right place, and
>    stronger.** Coefficient strengthening (52 rows, `gbd` + `hda`) is already
>    implemented on the Python side as `solvers/_root_presolve.py`
>    (`DISCOPT_COEF_TIGHTEN`, parked). Its "NOTE ON LOCATION" says exactly why it
>    lives there — "rewriting the Python model at the root is the only place the
>    tightened coefficients actually reach the relaxation" — and records that the
>    Rust pass is *weaker*: it reads declared bounds (so it bails on the `[0,∞)`
>    flows) and skips negative fixed-charge binary coefficients. Adopting the
>    Rust rewrite would import the weaker of the two.
> 3. **The safe scoping and the useful scoping are disjoint.** The three passes
>    run *alone* are variable-preserving on **all 147** instances, so that
>    adoption needs no postsolve at all — but in isolation `simplify` finds only
>    1,440 of its 6,635 rows (78 % of its effect depends on the other passes'
>    tightenings). The full pass list *does* change the variable count
>    (`hda` 722→719, `casctanks` 500→490, `4stufen` 149→148, `util` 145→144;
>    never grows), so the useful version needs a postsolve chain — and one cannot
>    be assembled from what the orchestrator records. `substitute.rs` is the
>    **only** pass that emits an inversion payload (`SubstitutionRecord` +
>    `block_map`); `EliminatePass`/`AggregatePass`/`FactorableElimPass` record
>    index lists in the delta stream, which say what happened, not how to undo it.
>
> **Decision.** Not built, and not left half-wired (the card's own warning). To
> reopen it, the prerequisite is a postsolve payload on every variable-changing
> Rust pass — a multi-session change to `crates/discopt-core/src/presolve/` in its
> own right — plus a reason to want it that is not wall-clock. Card 6a's finding
> that the substitution machinery **is** sound removes the soundness objection to
> reuse but not the benefit objection.

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

> **Status: modules 2–5 DROPPED by owner, 2026-07-30.** Module 1
> (`solver/native_kernel.py`) landed and stays. The rest is closed, not deferred:
> the entry experiment proved four of the five named modules are *inline statement
> blocks* sharing a closure of 200+ locals, so they cannot be moved without first
> extracting an explicit state object — a project in its own right that buys
> maintainability, not correctness. With the benchmark's purpose confirmed as
> **correctness validation**, that trade does not earn a session. Re-open only as a
> deliberately scoped state-object card, never as "finish the carve-up".

> **Addendum 2026-07-30 (open-ledger item 11 — the state-object card, run).** The
> deliberately scoped state-object card above was opened and its entry experiment
> measured the thing this status block asserted. **"A closure of 200+ locals" is
> not the coupling.** `solve_model` binds **851** names in its own scope, but only
> **153** are bound in one would-be module region and read in another, and they
> are not spread evenly — they pile up on two boundaries:
>
> | bound in ↓ / read in → | setup | reformulate | root | loop | results |
> |---|---|---|---|---|---|
> | **setup** | 42 | 4 | **10** | 3 | 1 |
> | **reformulate** | 4 | 33 | **12** | 3 | 3 |
> | **root** | 6 | 5 | 151 | **85** | **44** |
> | **loop** | 1 | 0 | 13 | 287 | 24 |
> | **results** | 0 | 0 | 6 | 5 | 44 |
>
> That reframes the dropped work substantially. `setup.py` and `reformulate.py`
> are **nearly self-contained** — 14 and 18 outbound names respectively — so they
> were never the hard part; `root`→`loop` (85) and `root`→`results` (44) are, and
> both are boundaries of the *same* module. Two further measured properties make
> the remaining migration provable rather than hopeful: **zero `nonlocal` writes**
> across all 41 nested scopes (the 40 closure captures are reads only, so no
> write-back paths have to be invented), and **no `locals()`/`globals()`/`eval`/
> `exec` anywhere in `solve_model`** (so every reference is statically resolvable
> and a migration can be *proved* by AST comparison rather than argued from a
> green test run).
>
> **What has landed.** `python/discopt/solver/state.py` — four cohesive
> `slots=True` dataclasses (`PhaseTimers`, `PrimalHeuristicState`,
> `LazyStallSeparationState`, `PerNodeOBBTBudget`) carrying **21** of the 153,
> threaded through `solve_model` with **no code moved out of the function**.
> Cross-region names 153 → 136; `root`→`loop` 85 → 68. Both increments are
> Regime-N clean on the hardened gate; numbers in §6.
>
> **What that unblocks, stated precisely so a later card does not over-claim it.**
> The census says the carve order the card assumed (setup first) is right for the
> wrong reason, and that `results.py` is the cheapest real module — its inbound set
> is 44 + 24 names, of which the certificate cluster is the only hard one. It does
> **not** yet unblock `spatial_loop.py`: 68 names still cross `root`→`loop`, and
> until they are grouped that signature is unreviewable. Finishing the remaining
> ~115 is mechanical (the tool and the gate both exist now) but it is bounded work,
> not a free consequence of this card.

With routing extracted, split the monolith along its existing phase banners into
`solver/` submodules: `setup.py` (validation, classification, convexity cache),
`reformulate.py` (the 9 reformulation stages), `root.py` (root presolve/OBBT/cuts,
Card 3a's schedule), `spatial_loop.py` (the 2,470-line inline loop → the 5th
Python loop file, honestly named), `results.py`. Target: no file > 2,500 lines, no
function > 300 except the loop body (its own follow-up). Pure moves + imports;
Regime N; `git log --follow` preserved by moving whole functions. Multiple PRs,
one module each — **never one big-bang PR**.

### Card 4c — Three stray B&B loops onto `PyTreeManager`

> **Status: CLOSED (2026-07-30). Task 2 LANDED; Task 1 RETIRED and REPLACED.**
> The vector-constraint corpus gap is **closed**. The three ports are **retired**
> — option (b) of the two exits this card offered, taken by the owner on the
> measurement below. In their place the card ships the thing the ports were *for*:
> a standing certificate-invariant suite over all three loops, and the
> default-inactive audit hook that makes their pruning decisions observable.
>
> ### Task 1 — RETIRED (owner decision, 2026-07-30)
>
> **The card's premise is falsified, and that is why it closes rather than
> waiting.** The premise was that consolidating onto one audited tree manager
> improves auditability. The measurement (§6, two entries, 2026-07-30) says a
> faithful port would add **five policy switches** to `PyTreeManager` — selection
> tie-break, prune slack, branch-variable rule, split-point rule,
> failed-relaxation handling — plus **two contract extensions**
> (`signomial_global` branches in **log space**; its per-node OBBT *replaces* the
> child box) that `export_batch`/`import_results` cannot express at all. That is a
> net *increase* in the complexity of the audited component, verified by a gate
> that cannot see the change. The premise does not survive its own consequences,
> so option (a) — materialise a GP/signomial population large enough to gate the
> port — is not merely blocked by this environment; it would be buying a worse
> design with a better gate.
>
> The **policy characterisation table below is preserved verbatim** as the durable
> deliverable: it is the reference anyone re-opening this question needs, and it
> is the specification the invariant suite audits against.
>
> ### Task 1's replacement — what actually serves the goal
>
> What makes a certificate-critical pruning decision auditable is that it is
> **observable**, not that it is centralised. Two pieces, both landed:
>
> - **`python/discopt/validation/fathom_audit.py`** — a default-inactive hook
>   (`record_fathom` / `fathom_audit()`); one module-global read and an `is None`
>   test per decision when nobody is listening. Each of the three loops now
>   reports **every** bound-fathom decision — *both* arms, kept and fathomed — in
>   its internal minimisation sense. Recording both arms is the point: a hook that
>   fires only on the fathom makes `node_bound >= incumbent - slack` a tautology.
> - **`python/tests/test_stray_bb_loop_invariants.py`** (`slow`) — the standing
>   regression watch none of the three loops had. Flags **forced ON**
>   (`DISCOPT_GP_MINLP`, `DISCOPT_SGO`, `lp_spatial=True`), because the entry
>   experiments established the panel never executes two of them on defaults.
>   Three invariants per loop, on real corpus instances:
>   **I1** no node is fathomed while its bound beats the incumbent by more than
>   the optimality tolerance — where the admissible slack is **re-derived by the
>   test from the declared `gap_tolerance`**, never read off the loop's own
>   reported slack; **I2** the reported dual bound never crosses the oracle from
>   `discopt_benchmarks/utils/reference_optima` (`.solu` when present, else the
>   vendored `known_optima.toml` / `cert-optima.json`, so it scores in CI);
>   **I3** the final incumbent passes `validation/feasibility.verify_point`.
>
> **Measured, 7 tests, all pass:** `i1_fathom_decisions` **12,905**,
> `i1_fathomed` **3**, `i2_bound_vs_oracle` **3**, `i3_incumbent_verified` **5**,
> and all three loops observed (`gp_minlp` 2 runs, `signomial_global` 2,
> `lp_spatial_bb` 1). A module-scoped finalizer fails the suite if any counter is
> zero, so a loop that quietly stops being reachable turns this file red instead
> of passing on nothing — the exact failure mode that killed Card 4c's own gate.
> A planted-violation control (`test_i1_checker_rejects_a_planted_wrongful_fathom`)
> proves the I1 assertion discriminates rather than accepting everything.
>
> **Finding worth recording: these loops almost never fathom by bound.** Only
> **3** of 12,905 audited decisions were fathoms. `lp_spatial_bb` fired its
> decision site **12,747** times on `nvs17` and fathomed **zero** times — under
> best-first the popped node *is* the frontier minimum, so the gap test fires
> before the bound test ever can, and that site is a safety net rather than a
> working pruner. `gp_minlp` and `signomial_global` fathom at most once per solve
> by construction (both `break` out of the loop on the frontier test). The
> invariant is therefore cheap to hold and the suite's value is as a *watch* on
> future policy edits, not as a stress test of today's code — stated plainly so
> nobody reads "3 fathoms" as thin coverage of a hot path.
>
> ### What landed (Task 2 — the vector-constraint corpus gap)
>
> `python/tests/vector_constraint_corpus.py` (7 cases, modeling-API built) and
> `python/tests/test_vector_constraint_corpus.py` (37 tests, all `smoke`), plus a
> vector arm in `test_node_tightening_parity.py` (Card 3c's guard) fed by
> `solvable_vector_cases()`.
>
> Cases carry mixed senses (`<=`/`>=`/`==`), non-zero `Constraint.rhs`, a
> multi-row equality, an integer vector, and a branching model, each with a known
> feasible and a known **infeasible** point. Every infeasible point is asserted
> in-bounds and integral first, so the constraint-row check is the *only* thing
> that can reject it — a mis-indexing verifier cannot be rescued by its bounds
> check.
>
> **Non-vacuity, per CLAUDE.md §6.** `_pre55_verify` transcribes the row loop
> verbatim from `030b44f4~1` (per-*object* index, `rhs` ignored, self-referential
> tolerance). Measured: it **wrongly ACCEPTS 6 of 7** infeasible points; the
> shipped verifiers reject 7 of 7 across all three entry points; and the scalar
> control is rejected by both, proving the discriminator measures alignment rather
> than failing everything. Parity arm: 2 solvable cases, **18 executed containment
> checks** on I2 (`vec_branching` captures 137 boxes). The arm's vacuity guard is
> not decorative — it *fired twice* during development, before a branching case
> existed.
>
> **Incidental finding (own issue, not fixed here).** A `Constraint` appended
> directly to `model._constraints` with a **non-zero `rhs` is honoured by the NLP
> evaluator and every verifier but IGNORED by the solver**: measured,
> `Constraint(w, ">=", 5.0)` appended by hand solves to `w = 0`, while
> `m.subject_to(w >= 5.0)` (which folds the offset into the body) solves to
> `w = 5`. The public path is correct; the private-append path — which
> `test_incumbent_verifier_scale.py` already uses, harmlessly, since it never
> solves — silently yields a model the solver and the verifier disagree about.
> `solvable_vector_cases()` filters those cases out of the solving suites
> *structurally* rather than by convention, and the parity arm re-asserts the
> all-zero-`rhs` property before solving.
>
> ### Why Task 1 (the three ports) did not proceed
>
> Two entry experiments, run before any port was written (§0.3), both landed
> against it:
>
> 1. **The Regime-N panel cannot see these loops.** `panel_baseline.py` sets no
>    `DISCOPT_*` flag, and `routing.py` gates `solve_gp_minlp` behind
>    `DISCOPT_GP_MINLP` and `solve_signomial_global` behind `DISCOPT_SGO`, both
>    **default-OFF**. A `--check` PASS after porting them would be the "0
>    violations = pass" failure §6 exists to stop. (`lp_spatial_bb` *is*
>    default-reachable, but only as `modeling/core.py`'s no-incumbent fallback.)
> 2. **Forcing the flags ON does not rescue the gate.** Over all 119 corpus
>    instances (357 executed classifications, 0 errors) `classify_gp_minlp`
>    accepts **3** and `classify_signomial_global` accepts **5**. Of the 3 GP
>    instances exactly **one** (`prob03`, 5 nodes) is a budget-independent
>    Regime-N comparable; the other two time out. A port gated on a single 5-node
>    tree cannot distinguish a tie-break change, a pseudocost change or a
>    fathom-slack change — measured: the exact-pruning arm left `prob03` at 5
>    nodes while moving both time-limited instances.
>
> Against that, the ports are **not** mechanical. `PyTreeManager` *owns* branching
> (variable **and** split point) and pruning; each loop diverges on several axes at
> once, enumerated below with file:line. Making any port bound-neutral means adding
> those policies as options to the one audited tree manager — i.e. paying
> complexity in the audited component, verified by a panel that cannot see the
> change. That trade fails §0.4's spirit and the card's own rationale, so **no loop
> was ported and no drift was accepted**, per the standing instruction to treat an
> unverifiable port as a finding.
>
> ### The characterised policies (the deliverable that does carry forward)
>
> | axis | `PyTreeManager` | `gp/solve_gp_minlp` | `signomial_global` | `lp_spatial_bb` |
> |---|---|---|---|---|
> | selection | `BestFirst`, tie → **deeper**, then lower `NodeId` (`pool.rs:50-64`) | best-first, tie → insertion **FIFO** (`gp/__init__.py:1000`) | best-first, insertion FIFO (`signomial_global.py:1333`) | best-first **plus a LIFO plunge stack** with depth cap + gap gate (`lp_spatial_bb.py:927-957`) |
> | prune | `node_lb >= incumbent`, **exact** (`tree_manager.rs:459`) | `>= incumbent - fathom_slack()` (`:1004`, `:1029`) | `gap_ok()` relative slack (`:1320`) | `>= inc_val - 1e-9*(1+|inc|)` (`:986`) |
> | branch var | pseudocost/reliability, **default ON**, no Python setter (`:560-574`) | most-fractional (`_most_fractional_offset`) | integer-first, else max-width or **DC secant-gap score** (`:1377-1385`) | own pseudocost `_branch_var`, else `_worst_product_var` |
> | split point | `floor(val)` / `floor+1` (`branching.rs:299-317`) | `floor(v)` / `ceil(v)`, **empty child suppressed** (`:1053-1058`) | **log-space** `log(fl)` / `log(fl+1)`; continuous at midpoint *or* clipped `ustar[j]` | integer `floor((lb+ub)/2)`/`mid+1`; continuous shared midpoint |
> | failed relaxation | floored at parent bound, then **branched** (`:382-420`) | **abandoned** immediately into `abandoned_bound` (`:1017-1026`) | `min_fathomed` frontier accounting | `unresolved_lb` floor |
> | per-node box rewrite | none | none | **OBBT rewrites the child box** (`eval_constrained`) | basis/cut inheritance |
>
> Two of these are not options-shaped at all: `signomial_global` branches in
> **log space** and lets per-node OBBT *replace* the child box, neither of which
> `PyTreeManager`'s `export_batch`/`import_results` contract can express.
>
> ### What remains in Card 4c — **NOTHING. The card is closed.**
>
> The decision the card could not make for itself was made by the owner on
> 2026-07-30: **option (b), retire.** For the record, the exits as the card framed
> them were **(a)** materialise a GP-MINLP / signomial population large enough to
> gate a port (the MINLPLib snapshot at `~/Dropbox/projects/discopt-minlp-benchmark/`
> has the instances; this environment does not), then port `gp` first — the only
> loop whose divergences are all options-shaped; or **(b)** accept the finding and
> retire, on the grounds that adding five policy switches to the audited tree
> manager to absorb three callers makes the audited component *harder* to audit.
>
> (b) was taken because (a) buys a better gate for a worse design: the five
> switches and two contract extensions are a cost the port pays *whatever* gate
> certifies it. `lp_spatial_bb` was skippable either way per the card's own
> proviso if Phase 5 retires its class, and `signomial_global` could not be ported
> at all without a `PyTreeManager` supporting caller-supplied split points and
> caller-rewritten child boxes — a much larger change than this card scoped.
>
> **Should the question re-open**, the prerequisites are unchanged and now
> written down: the policy table above is the specification, and
> `test_stray_bb_loop_invariants.py` is the differential harness a port would have
> to keep green. Re-opening needs a population, not a re-analysis.

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
> | 1 | `term_trilinear` | 17 | 437.6 | 22.7 % | ~~needs a trilinear `EnvTerm` in Rust~~ **FALSIFIED 2026-07-29 (Card 5.2-T):** the base envelope is already a nested McCormick *bilinear* chain the kernel expresses; and only **7 of 17** would reach row-claiming, so the recoverable wall is **118.5 s / 6.2 %** |
> | 2 | `infinite_aux_bounds` | 9 | 379.6 | 19.7 % | 4stufen, beuster, carton7, contvar, ex14_1_9, gear4, hda, heatexch_gen1/2 |
> | 3 | `probe_real_shape_mismatch` | 3 | 158.9 | 8.2 % | casctanks, st_e04, st_e35 |
> | 4 | `probe_objective_bound_invalid` | 4 | 155.3 | 8.1 % | the whole `tspn*` family — the G-F `no_bound` class |
> | 5 | `fixed_row_box_dependent:coeffs` | 9 | 109.7 | 5.7 % | an unmodelled box-dependent envelope family; the guard is working, the coverage is missing |
> | 6 | `blf_row_count` (5 or 6) | 7 | 45.0 | 2.3 % | #861 over-claiming. ~~the Python incremental engine already solves this numerically (`incremental_mccormick._select`)~~ **FALSIFIED 2026-07-29:** `_select` covers bilinear/monomial/affine_square only and never sees `bilinear_linform_specs` — a build, not a port |
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
> recorded as **SKIPPED — local only**.
>
> The §5 differential panel ran anyway, over all 119, both arms, ×3 replication on
> the decisive rows (10,734 s; artifact
> `reports/phase5_convex_kernel_diff_panel_670911ed.json`; full verdict verbatim in
> §6):
>
> ```
> cert-clean    : PASS (0)          [613 executed checks]
> quality-clean : PASS (0)
> net-positive  : FAIL (engaged 3, helped 0, median non-engaged wall delta
>                       +0.060s over 116, overhead_ok=True)
> GRADUATE      : NO
> ```
>
> So the two questions the in-repo corpus *can* answer are answered — turning the
> flag on is cert-clean and quality-clean corpus-wide, and it costs **+0.060 s
> median** on the 116 instances it never routes (that is the convexity
> classification and nothing else). `helped = 0` because the one instance that
> would carry the bar, `clay0303hfsg`, was **quarantined as unresolved** (ON
> `optimal` ×3, OFF `feasible`/`time_limit`/`feasible`) — the #902 replication
> machinery refusing to let ambient load decide a verdict. The cleaner measurement
> of the same win is the entry experiment (quiet machine, interleaved, 2
> replicates): OFF `feasible` ×2 → ON `optimal` ×2, 46.66 s → 42.53 s.
>
> **`DISCOPT_CONVEX_KERNEL` therefore stays default-OFF.** What would graduate it,
> stated so the next session does not re-derive it: the same panel over the 136-file
> convex family plus `watercontamination0202`, on a machine with the snapshot, with
> `clay0303hfsg` resolving stably.
>
> ### What 5.2 should take next — **REWRITTEN 2026-07-29; both prior claims measured false**
>
> The two sentences this block used to contain were entry-experimented and **both
> failed**. They are kept below, struck, because §0.3 says a falsification is recorded
> where the claim lived, not only in §6.
>
> ~~"`IncrementalMcCormickLP._select` identifies exactly the same envelope rows…
> porting that matcher is a Python change worth 7 instances."~~ **FALSIFIED.**
> `_select` has exactly **3** call sites — `bilinear`, `monomial`, `affine_square`.
> The class `blf_row_count` declines on is `rel.bilinear_linform_specs` (a product of
> two *affine forms*), and `incremental_mccormick.py` **never references that field**;
> it is consumed by `spatial_producer.py` alone. There is no matcher to port. Probe:
> `discopt_benchmarks/scripts/phase52_blf_select_entry.py`, artifact
> `reports/phase52_blf_select_entry_ed2da7bd.json`, 55 BLF terms examined across the 7
> declining instances plus `tanksize` as a served control (19/19 terms at exactly 4
> rows — the producer's predicate is right about the instance it serves).
>
> ~~"`term_trilinear` … needs a trilinear `EnvTerm` in Rust … it is where the wall
> is."~~ **FALSIFIED on both halves** — see the Card 5.2-T scoping card below. The
> base relaxation of a trilinear product is already a nested McCormick *bilinear*
> chain the kernel can express, and the recoverable wall is **118.5 s / 6.2 %**, not
> 437.6 s / 22.7 %.
>
> **What is actually true, and what the next session should do.** `blf_row_count` is
> still the right *shape* of card, and the direction is viable — the measurement shows
> the extra candidate rows are exactly what a matcher needs them to be, namely
> box-**independent**: aux bound rows (`{w: ±1} <= 0` on `st_e40`, `syn05hfsg`) and
> lifted model constraints (`600·x₀ − 50·x₁ − w <= −5000` on `st_e11`; `2x₀ + 2x₁ +
> 4w <= 3` on `st_e09`). But it is a **build**, not a port, and it has a prerequisite
> the old framing hid: a `_select`-style match needs the closed-form expected rows,
> which for `w = A·B` are McCormick rows in the *form enclosures* `(aL, aH, bL, bH)`
> — and `bilinear_linform_specs` does **not** record them. `_emit_mccormick` receives
> them as `ba`/`bb` from `ctx.bounds(node)` (an `evaluate_interval` on the original
> DAG, *not* a LinForm interval over column bounds, so they cannot be recomputed
> faithfully from the spec). So the card is: (1) extend the spec tuple to carry the two
> enclosures, (2) add a BLF expected-row generator mirroring `_emit_mccormick`'s four
> rows, (3) match numerically and require exactly one hit per expected row — declining
> on ambiguity, as `_select` does — leaving the unmatched rows as fixed rows rather
> than claiming them. Regime C, 7 instances, `45.0 s / 2.3 %`.

### Phase 5.2 — the census re-read by ladder position (open-ledger item 8) — **MEASURED; NOTHING BUILT, AND THAT IS THE RESULT**

> **Status:** three entry experiments run 2026-07-30, **all three killed**. **Regime:**
> none — no solver code changed, so neither N nor C applies and **no flag was
> registered and no default moved**. **Depends:** Phase 5.1 (the census). **Est:** done
> as scoped; what remains is a different card, named below.
>
> #### What item 8 asked, and what the corpus answered
>
> The ledger's item-8 row and this phase's own scoping note both argued census rank #2
> (`infinite_aux_bounds`, 9 instances, **379.6 s / 19.7 %**) was the better next card
> than rank #1, *because* rank #1's headline had been measured to be inflated by
> decline-ladder masking. **The same masking applies to rank #2, and harder.**
>
> | experiment | hypothesis | kill criterion | measured | verdict |
> |---|---|---|---|---|
> | **E8.1** framing | ≥ 6 of 9 reach row-claiming with the gate bypassed | ≤ 4 | **1 of 9** | KILLED |
> | **E8.2** repairability | infinities repairable by FBBT on the relaxation's own rows | ≥ 5 of 9 survive | **9 of 9 survive**; 0 of 229 infinite columns finitized | KILLED |
> | **E8.3** re-pick (ranks #3+#5) | offending rows belong to *registered* lifts → producer claiming-predicate fix | ≥ 6 of 12 over unregistered aux | **9 of 9** rank-#5 instances | KILLED |
>
> Executed counts, artifacts and the full mechanism are in §6 (three entries dated
> 2026-07-30). Headlines:
>
> * **Rank #2 is worth 0.0 %, not 19.7 %.** The one instance that reaches row-claiming
>   with the gate bypassed (`carton7`) is the gate being *right*: 24 infinite original
>   columns, an unbounded presolved root box, and a bypassed spec carrying 24 infinite
>   entries in `global_lo`/`global_hi` — an unbounded node LP whose safe-bound
>   evaluation is invalid. The other 8 decline immediately on another missing feature.
> * The code name is a misnomer hiding **two disjoint classes, neither producer-side**:
>   **6 of 9** have an unbounded *presolved root box* on original columns (a McCormick
>   envelope over an unbounded box does not exist — this is the presolve
>   propagation-quality lever `issue-764-scip-comparison.md` names), and **3 of 9**
>   carry free aux columns from atoms the relaxer refused to envelope, 13/17, 16/20 and
>   3/6 of which appear in **no relaxation row at all**, so FBBT has nothing to
>   propagate through by construction.
> * **Ranks #3 and #5 — the re-pick — are relaxer-side.** Every rank-#5 instance's
>   box-dependent rows sit on aux columns registered in **no** structural map: the
>   `power` kind whose lift lands in none of `monomial_map` /
>   `univariate_square_map` / `affine_square_map`, plus factorable-intermediate partial
>   products. On 6 of 9 the arithmetic is exact — offending rows = 4 × (unregistered
>   `power` lifts); `kall_circles_c8a` is 72 lifts × 4 = **288 rows**. Rank #3 is the
>   same root seen from the other side: identical family counts and coverage kinds
>   across both builds, with the real build emitting strictly *fewer* rows
>   (`casctanks` −16, `st_e35` −7, `st_e04` −1, columns unchanged).
>
> #### The reading protocol this phase now binds (the transferable result)
>
> The producer's decline ladder is **ordered**, so an attributed wall is an upper bound
> exactly to the extent the code is tested early — a static property readable off
> `spatial_producer.build_spatial_kernel_spec` without running anything. Ranks **#3**
> and **#5** are the producer's *last two* tests, so every instance in them has already
> passed every other decline test and their wall is recoverable **by construction**.
>
> | rank | code | ladder pos | inst | attributed | **recoverable** |
> |---|---|---|---|---|---|
> | #3 | `probe_real_shape_mismatch` | 15 (unmaskable) | 3 | 158.9 s / 8.2 % | **158.9 s / 8.2 %** |
> | #1 | `term_trilinear` | 8 | 17 | 437.6 s / 22.7 % | 118.5 s / 6.2 % |
> | #5 | `fixed_row_box_dependent` | 16 (unmaskable) | 9 | 109.7 s / 5.7 % | **109.7 s / 5.7 %** |
> | #2 | `infinite_aux_bounds` | 7 | 9 | 379.6 s / 19.7 % | **0.0 s / 0.0 %** |
>
> Two of the top five have now been re-measured and both collapsed. The two rows that
> were never inflated were ranked third and fifth. **Staff the census by ladder
> position, not by attributed wall.**
>
> #### What the next card is (named so it is not re-derived)
>
> Not a producer change. Give the unregistered lifts a structural registration in
> `uniform_relax` — the non-monomial/non-square `power` family and the McCormick fold's
> factorable-intermediate partial products — so the producer's map-driven claiming can
> see their columns, plus `EnvTerm` support in the Rust kernel for any family it does
> not already implement, then make the box-conditional row emission (rank #3)
> unconditional or claimable. Regime C, 12 instances, **268.6 s / 13.9 %** — the
> largest genuinely-recoverable block in the census. It is a **relaxer + kernel** card,
> materially larger than a producer-gate change, and it is *not* what item 8 scoped;
> E8.3's pre-stated kill criterion said item 8 stops at the measurement, and it does.
>
> The only remaining producer-side entry in the top five is **Card 5.2-T** (rank #1,
> 7 instances / 6.2 %), below — owner-scoped, ledger row 9.
>
> #### Verification run on the final tree (2026-07-30)
>
> No solver code changed this session — the diff is two probe scripts, their artifacts
> and this document — so the Regime-N question is whether the tree still behaves as
> item 11 left it, not whether a change is bound-neutral.
>
> | suite | result |
> |---|---|
> | `pytest -m smoke` (python/tests) | see the item-8 close-out §6 entry |
> | `pytest -m smoke` (discopt_benchmarks/tests) | ditto |
> | adversarial (`test_adversarial_recent_fixes.py -m slow`) | ditto |
> | node-tightening parity (smoke + slow) | ditto — **no newly-served class, so nothing was added to it**; Card 2b's 147/83/44 asymmetry is unchanged |
> | vector corpus / routing / tightening schedule / solver state / flag registry | ditto |
> | `ruff check` + `ruff format --check` + `mypy` (2.1.0) | ditto |
> | `cargo test -p discopt-core` | **not run — no Rust touched** |
> | Regime-N `panel_baseline.py --check` | ditto |
> | `heldout50` / MINLPLib snapshot | **SKIPPED — local only** |
>
> **Parity-guard note.** `python/tests/test_node_tightening_parity.py` is the standing
> guard that coverage expansion cannot silently weaken node tightening. This session
> expanded coverage by **zero** classes, so there is no new class to cover and the file
> is untouched (its `pytest.mark.xdist_group` pin intact). Whichever session takes the
> relaxer-side card above owes it the new `power`-family arm.

### Card 5.2-T — `term_trilinear` scoping (census rank #1) — **SCOPED, NOT STARTED**

> **Status:** scoping only, per the owner's instruction not to implement. Two
> load-bearing measurements, both cheap, both contradicting the census's framing of
> this row.
>
> **1. It is not a new Rust envelope family.** `uniform_relax._fold_product` relaxes
> `x_i·x_j·x_k` as a **nested chain of McCormick bilinear lifts** — `t = x_i·x_j` then
> `w = t·x_k`, two `_emit_mccormick` calls, hence two `bilinear_linform_specs` entries.
> `trilinear_map[(i,j,k)] = w` is registered *in addition*, purely so the separators
> (trilinear RLT, Meyer-Floudas/Rikun hull, `multilinear_separation.py`) can attach
> tighter cuts on top. So the producer's `if rel.trilinear_map: return
> _decline("term_trilinear")` rejects the whole model on the mere **registration** of a
> trilinear product, not on the presence of rows the kernel cannot regenerate — the
> `BlfTerm` family it already implements covers the base envelope.
>
> Residual, and this is the real work: **4–13 rows per instance** touch the trilinear
> aux column and are claimed by no BLF / monomial / affine-square term (measured:
> `st_e03` 6, `ex1224` 4, `nvs01` 6, `mathopt3` 6, `nvs22` 8, `bchoco06` 13, `nvs09`
> 0). Those are lifted model constraints (box-independent — they pass through as fixed
> rows, fine) mixed with **trilinear RLT rows**, which are box-**dependent** and
> `DISCOPT_TRILINEAR_RLT` defaults **ON**. Those must be excluded from the producer's
> build the same way `skip_separable_floor` and `skip_convex_lift` already are —
> options that exist in `_build` for precisely this reason. Dropping them loosens the
> relaxation (valid, weaker); claiming them would be unsound.
>
> **2. The prize is 6.2 %, not 22.7 %.** The producer's decline ladder is ordered, and
> `term_trilinear` is tested **before** `term_multilinear` / `term_ratio` /
> `term_univariate` / `infinite_aux_bounds`, so it *masks* whatever else each model
> would decline on. Re-running the ladder with the trilinear test removed, over all 17
> instances: only **7 reach row-claiming** (`ex1252`, `ex1252a`, `nvs01`, `nvs06`,
> `nvs21`, `st_e03`, `st_e38`). The other 10 decline immediately on a different missing
> feature — **6** `infinite_aux_bounds` (`bchoco06/07/08`, `mathopt3`, `nvs05`,
> `nvs22`), **3** `term_univariate` (`ex1224`/`st_e29` log, `st_e36` exp), **1**
> `term_multilinear` (`nvs09`). Baseline wall of the reachable 7 is **118.5 s of
> 1,926.0 s = 6.2 %** (and `ex1252`/`ex1252a` are 92.0 s of that 118.5 s, both sitting
> at the 45 s budget ceiling). The census's 437.6 s / 22.7 % is correct as an
> attribution of the *first* decline code — reproduced exactly — but it is an **upper
> bound**, not recoverable wall.
>
> **Verdict, one sentence:** this is "wire up what exists" (a Python producer change:
> drop the blanket `trilinear_map` decline, add a `skip_trilinear_rlt`-style build
> option, let the nested `BlfTerm`s be claimed) rather than a new Rust envelope family
> — **one to two sessions** including the Regime C panel and the Card 3c parity
> extension, for **7 instances / 6.2 % of corpus wall**, which after this re-measure
> makes it comparable to rather than dominant over rank #2 `infinite_aux_bounds` (9
> instances, 19.7 %, and the blocker for 6 of the trilinear 17 as well) — so
> `infinite_aux_bounds` is arguably the better next card, and this scoping does not
> claim otherwise.

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

## Phase 5.5 — the incumbent verifier is scale-blind (filed by the 5.4 panel) — **LANDED**

> **Status:** LANDED 2026-07-29. **Depends:** nothing. **Regime:** neither N nor C
> as the plan defines them — see "Which regime" below. **Est:** done.
>
> Phase 5.4's differential panel first scored `cert-clean: FAIL(2)` and the failures
> were the **verifier's**, not the solver's. Phase 5 rescoped its gate to asymmetric
> failures (correct for a differential panel) and filed the underlying defect as a
> named follow-up, because it is a solver-wide correctness question and CLAUDE.md §1
> puts correctness before performance. This card is that follow-up, closed.
>
> ### The defect, stated exactly
>
> Every certificate-gating incumbent verifier used the tolerance
> `abs_tol + rel_tol * |residual|`. That is **self-referential** — keyed on the very
> quantity being judged. Solving it for an equality row: the row passes iff
> `|r| <= abs + rel·|r|` iff `|r| <= abs/(1 - rel)`. With `abs=1e-6, rel=1e-4` that
> is `1.0001e-6`: a pure **absolute** 1e-6 on every row scale, and the `rel_tol` term
> is arithmetically dead. The same collapse holds for `<=` / `>=`.
>
> **`nvs22`'s certificate is not the problem — the verifier was.** Measured (probe
> `verifier_scale_sweep.py`, and the §6 entry): status `optimal`, objective
> `6.058219942618198` against MINLPLib `=opt= 6.05822` (5.7e-8), and the two rows
> that failed carry **relative** residuals 8.1e-9 and 1.5e-8 on row scales 2.1e3 and
> 1.7e4. The opposite conclusion the card was told to consider — "the certificate is
> issued at a tolerance the repo's own verifiers do not accept" — is falsified by
> those numbers, and it is worth saying loudly that it *was* checked.
>
> ### Three more defects the fix surfaced, all in the wrongly-ACCEPT direction
>
> Found by an entry experiment written to falsify the assumption that the two
> verifiers examined the rows they claimed to (probe transcript in §6):
>
> | # | defect | measured consequence |
> |---|---|---|
> | 2 | one row index per *constraint object* vs one evaluator row per *flat element* | a point violating row 2 of a size-3 vector constraint **by 5.0** was returned FEASIBLE by both verifiers |
> | 3 | `model._constraints` misses builder-resident linear rows (`add_linear_constraints`) | such a model looks unconstrained to the verifier |
> | 4 | `Constraint.rhs` ignored (body compared against 0); non-`Constraint` classes skipped (native kernel) or `AttributeError` mid-loop (convex kernel) | wrong verdict in both directions; a whole constraint class vouched for without being read |
>
> `_incumbent_is_feasible` additionally checked **neither variable bounds nor
> integrality** — a fractional value on an INTEGER variable passed.
>
> ### What exists now
>
> `python/discopt/validation/feasibility.py` — one verifier, `verify_point`, exported
> from `discopt.validation`. Rows are enumerated from the **evaluator's own row map**
> (`_source_constraints` × `_constraint_flat_sizes`), which is what makes defects 2
> and 3 structurally impossible rather than fixed-by-hand. Tolerance:
>
> ```
> violation_i <= abs_tol * scale_i ,  scale_i = max(1, |b_i|, max_j |J_ij|·max(1,|x_j|))
> ```
>
> `scale_i` is the **examiner's** row scale (`validation/examiner.py`, "Examiner's
> scaled mode") — reused, not reinvented (§0.8). Consumers rewired:
> `solver/native_kernel._native_kernel_verify_point` (the native-kernel seed gate),
> `solvers/_convex_kernel._incumbent_is_feasible` (the #779 adoption gate, and the
> independent verifier both Regime-C panels call), and `warm_start.check_feasibility`
> (alignment/rhs only — see below).
>
> ### Which regime, and why this is stricter rather than looser (§0.4)
>
> Not Regime C: it changes no bound, cut or routing. Not plain Regime N either: it
> deliberately changes a *verdict function*. The bar it was held to instead is the one
> §0.4 implies — **the fix must be more correct, not more permissive**, demonstrated
> both ways:
>
> * **Four wrongly-accept holes closed** (defects 2–4 plus the missing
>   bounds/integrality), each with a regression test that fails on the old code.
> * **The one widening is bounded and calibrated.** The relative coefficient is
>   `abs_tol` (1e-6), **not** the repo's `rel_tol` (1e-4) — reusing `rel_tol` would
>   have put the floor at 1.01e-4 and loosened every unit-scale row 100×. With
>   `scale_i >= 1` and coefficient `abs_tol`, a unit-scale row's tolerance is
>   *exactly* the pre-existing absolute 1e-6. The same function's variable-bound check
>   has carried a scale term with the 100× looser `rel_tol` coefficient since #764;
>   this makes the row check consistent with it and tighter.
> * **Four naive widenings that would have accepted a bad point are each killed by a
>   test**: a model-global `max_j |x_j|` scale (accepts a unit row violated by 1e-2
>   next to a 1e9 variable); a 1-norm row scale (accepts a 0.5 violation on a row of
>   10^3-magnitude cancelling terms); `rel_tol` as the coefficient (accepts 5e-5 on a
>   unit row); simply raising the absolute floor to 1e-3 (accepts 5e-4 on a unit row).
> * **Refusal beats approximation** (§3): unknown sense, unevaluable constraint class,
>   evaluator failure, non-finite value, row-count disagreement and point-length
>   mismatch all return *not verified with a reason*, never an optimistic pass. When
>   the Jacobian cannot be formed the scale degrades to `max(1, |b_i|)` — the strict
>   direction; a verifier that cannot measure a row's scale must not widen for it.
>
> `warm_start.check_feasibility` got the alignment/rhs fix but **keeps its flat
> absolute tolerance on purpose**: it is a user-facing warm-start diagnostic with no
> certificate to protect, so the strict reading is the useful one. Stated in the code.
>
> ### Verification
>
> `python/tests/test_incumbent_verifier_scale.py`, 16 tests (15 `smoke`/`unit`, 1
> `slow`+`correctness` on the real `nvs22`). **7 of the 16 fail on the pre-fix
> consumers** (proved in a worktree at `bb3b6c73` carrying only the new module and the
> new test file, so the failure is attributable to the consumers and not to a missing
> import); all 16 pass after. The 9 that pass in both are the §0.4 locks — they assert
> the strictness that must *not* move.
>
> Corpus evidence: `discopt_benchmarks/scripts/verifier_scale_sweep.py` +
> `reports/verifier_scale_sweep_<sha>.json` score every in-repo incumbent under both
> tolerance forms on one tree (the old form re-implemented inline, so the comparison
> is of *forms* rather than of git revisions). Verdict in §6.
>
> Panel re-measurement: Phase 5.4's own child, `nvs22`, both arms, ×2 — `verified`
> flips **false → true in both arms**, i.e. the panel's single symmetric
> `verification note` disappears, and it disappears for the right reason (worst
> relative row violation 1.5e-8, not a widened tolerance). The panel's differential
> scoping is left as Phase 5 wrote it and its comment updated to record the fix.
>
> **Can this card close: YES.** No follow-up filed; nothing deferred.

---

## Phase 6 — Missing mechanisms (gap closure beyond the kernel)

> **Status:** OPEN. **Depends:** Phase 0; independent of Phases 3–4; 6b after 5.4.
> **Est:** 3+ sessions per card; each card is optional-but-valuable on its own.

### Card 6a — Presolve reduction scale (G-G class) — **ROOT-CAUSED 2026-07-30; the premise was wrong, one real defect fixed, flag stays PARKED**

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

> **Status: root-caused 2026-07-30** (§6 entry "Card 6a: the 2449 % primal gap is
> not a postsolve bug"). `watercontamination0202` is snapshot-only, so the card was
> worked by construction plus in-repo reproducers.
>
> 1. **"It is a postsolve or substitution-validity bug" — FALSIFIED.** The
>    transform is exact on every in-repo instance it reduces (13 of 66):
>    13,086 surviving-row comparisons, 3,642 dropped-row identity checks, 52
>    box-soundness and 26 box-exactness checks, plus 104 objective-identity and
>    104 Python-bridge-fidelity comparisons — **zero failures**. All four named
>    candidate mechanisms are dead.
> 2. **The incident is not a wrong answer.** The reported point is recomputed on
>    the pristine model and feasibility-verified there at 1e-5 before it is
>    returned, and the run's status was `feasible` under a time limit. A 2449 %
>    gap on a *verified-feasible* incumbent is a primal-quality outcome, not a
>    certificate defect. Retracted accordingly (CLAUDE.md §11).
> 3. **What the flag actually does** (12-instance ON/OFF panel, interleaved,
>    replicated, bit-identical within arm): it changes the problem the relaxation
>    and the heuristics see, **large and not monotone** — `hda`'s dual bound 132×
>    *looser*, `4stufen`'s ~5× *tighter*, `casctanks` 32 % tighter,
>    `heatexch_gen2` gains a first incumbent where the default path has none
>    (the `watercontamination0202` signature, reproduced in-repo), `heatexch_gen3`
>    gains a first *bound* where the default path produces none. No bound crossed
>    a proven optimum on any of the four instances that have one.
>    **The cost is wall, not nodes:** on `st_e11` the flag cuts the tree from 27
>    nodes to 5 and still takes 4–5× longer (6.7/8.2 s OFF vs 30.5/40.6 s ON), so
>    at the panel's 30 s budget it turned `optimal` into `feasible`. At 120 s both
>    arms certify.
> 4. **One real defect, fixed:** the #779 postsolve guard was integrality-blind
>    (`ModelRepr.evaluate_point` checks rows and bounds only), so a lifted point
>    with a fractional integer was reported as `optimal`. Closed in
>    `_presolve_substitute.integrality_violation`; regression test fails before,
>    passes after. Not reachable through today's substitution pass — reachable
>    through any *other* repr transform reusing the guard, which is Card 3d.
> 5. **Flag verdict: PARKED, graduation REFUSED, with two named blockers.**
>    (a) the 4–5× wall cost on `st_e11`, which under a tight budget *is* a
>    certification regression — §0.1's hard Regime-C bar — and which no node-count
>    saving compensates; (b) no predictor of *when* substitution helps the
>    relaxation: `hda` and `4stufen` reduce comparably and move the bound in
>    opposite directions by orders of magnitude, so the measurement does not
>    supply one and inventing one would be a hypothesis-driven fix (CLAUDE.md §4).
>    Not removed: it is the **only** pass in the tree that records an inversion
>    payload, and both Card 3d and the deferred FBBT-coupled fixing loop name it
>    as the pattern.
> 6. **The card's own kill criterion fires.** Reductions on this class stay far
>    below 2× on the corpus (§G-G.1's 300-instance census: nothing on 64.3 %,
>    ≥3× on 1/300), so per the card text the SCIP 189× is presolver tech discopt
>    does not have. The specific missing mechanism is already named in §G-G.1 —
>    bound-driven fixing (SCIP's 78,959 `ChgBounds`), dual reductions and
>    implied-free column elimination — and stays deferred, not built.

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

**Re-sequencing, 2026-07-30 (owner) — the benchmark is for CORRECTNESS, not speed.**
That reprioritizes what is left:
- **Card 4c: CLOSED (2026-07-30).** Task 2 landed (7 modeling-API vector cases
  wired into the verifier and Card 3c parity suites; the pre-5.5 row loop wrongly
  accepts 6 of 7, so the coverage is proven non-vacuous). **Task 1 RETIRED** —
  owner took exit (b). Two entry experiments (§6) showed the Regime-N panel never
  invokes the GP-MINLP or signomial loops on defaults (both behind default-OFF
  flags) and that forcing the flags ON yields exactly **one** budget-independent
  comparable (`prob03`, 5 nodes); worse, a faithful port would add five policy
  switches plus two contract extensions to the *audited* component, inverting the
  card's own premise. Replaced by what actually serves the goal:
  `validation/fathom_audit.py` (default-inactive observability of every
  bound-fathom decision) and `test_stray_bb_loop_invariants.py` (I1 no wrongful
  fathom / I2 bound never crosses the oracle / I3 incumbent verifies, flags forced
  ON, **12,905 audited decisions**). The per-loop policy characterisation is
  preserved on the card as the specification a future port would have to meet.
- **The `Constraint.rhs` silent wrong answer (filed by Card 4c Task 2): FIXED.**
  Root cause `_jax/dag_compiler.compile_constraint`, which compiles `body` and
  drops `rhs`; **26 modules** read `.body` and never `.rhs` while the verifier,
  the exporters and the Rust `ConstraintRepr` all honour it. **The public API is
  affected** — `m.subject_to(Constraint(w, ">=", 5.0))` solved to `w = 0`; the
  earlier "the public path is correct" note is retracted in §6. Direction:
  **refuse loudly** at `Model.subject_to` *and* `Model.validate` (CLAUDE.md §3),
  because half-honouring `rhs` across 26 modules would replace a wrong-answer
  hazard with a soundness hazard.
- **Card 4b modules 2–5: DROPPED** (see its card) — maintainability, not
  correctness, behind a state-object prerequisite.
- **Card 6d (process floor): OFF the critical path** — pure wall-clock, no
  certificate content. Keep for a later performance pass.
- **Card 6a rises**: root-causing substitution's measured 2449 % primal gap is a
  correctness investigation wearing a performance label.
- Order from here: **4c → 3a(b) → 6a → 3d** (3d last and default-OFF; it is still
  the most dangerous change in the plan), with 6b/6c/7 as capacity allows.

**Close-out, 2026-07-30 (the last substantive session).** That order was worked to
its end. **4c CLOSED**, **3a(b) CLOSED** (all three de-duplications answered NO),
**6a ROOT-CAUSED** — the 2449 % incident is not a postsolve or substitution-validity
bug (16,800+ executed exactness comparisons, zero failures) and is not a wrong
answer at all; the one real defect was an integrality-blind postsolve guard, now
fixed, and `DISCOPT_PRESOLVE_SUBSTITUTE` stays parked with a named blocker
(`st_e11` `optimal → feasible`). **3d NOT BUILT** — its entry experiment (147
instances) shows the deliverable is throughput, its one bound-relevant component
already exists stronger in `_root_presolve.py`, and the postsolve chain it needs
cannot be assembled from what the Rust orchestrator records. Both verdicts are in
§6 with their numbers. What remains open across the whole plan is listed under
**"Open at close-out"** at the end of this section.
- Standing requirement, after a container rollback silently destroyed a session:
  every card asserts `HEAD == origin` **and** rebuilds/verifies the compiled `.so`
  marker before measuring. Source-file checks alone missed a stale binary once.

### Open at close-out (2026-07-30) — everything the plan has not finished

One line each on what it would take to close. Nothing here is blocked on a defect;
every item is either a deliberate deferral with its measurement recorded, or work
this environment cannot adjudicate.

| # | item | state | what would close it |
|---|---|---|---|
| 1 | **Card 3d** — adopt the presolved repr | NOT BUILT, entry experiment recorded | A postsolve payload on every variable-changing Rust pass (`EliminatePass`, `AggregatePass`, `FactorableElimPass`), then a Regime-C panel — **and a benefit that is not wall-clock**, which the 147-instance measurement does not currently supply |
| 2 | **Card 3e / `DISCOPT_FBBT_SEED`** | mechanism landed default-OFF, graduation refused | A corpus population where seeding the root FBBT from `ctx.bounds` is net-positive; the in-repo panel showed sound-but-not-helpful |
| 3 | **Card 6a / `DISCOPT_PRESOLVE_SUBSTITUTE`** | sound, parked, graduation refused | A *predictor* of when substitution helps the relaxation. `hda` (132× looser) and `4stufen` (~5× tighter) reduce comparably and move oppositely, and `st_e11` regresses `optimal → feasible` |
| 4 | **Card 6a's missing presolve mechanism** | named, not built | Bound-driven fixing (SCIP's 78,959 `ChgBounds`), dual reductions, implied-free column elimination — §G-G.1 names them; each is its own card, and §G-G.1's own census says closing the reduction-rate gap should not be expected to move the corpus geomean |
| 5 | **Card 6b** — CSE / defined-variable sharing | not started | Its entry experiment: shared-subexpression multiplicity on the 20 largest instances (kill: <1.5× median duplication), then build in the kernel-facing path |
| 6 | **Card 6c** — aggregation / c-MIR cuts | deliberately last | Phase 5 first (cheap nodes to multiply against — OBBT × cuts × throughput measured multiplicative), then a Rust multi-row separator with the #781 incumbent-starvation check as a gate arm |
| 7 | **Card 6d** — process floor (~513 ms import) | off the critical path | Pure wall-clock, no certificate content. Lazy-import audit + `discopt solve` fast path, Regime N on cold-start wall |
| 8 | **Card 5.2 / 5.3** — kernel coverage expansion | **MEASURED 2026-07-30, NOTHING BUILT — and that is the result.** The card as scoped (`infinite_aux_bounds`) is **dead**; this row's own premise was wrong | Three entry experiments, three kills (§6, and the Phase 5.2 block). Rank #2's recoverable wall is **0.0 %, not 19.7 %** — 6 of 9 have an unbounded *presolved root box* (presolve propagation, not the kernel), 3 of 9 have free aux columns from atoms the relaxer refused to envelope. The re-pick (ranks #3+#5, **268.6 s / 13.9 %**, unmaskable by ladder position) is **relaxer-side**: unregistered `power` lifts + factorable-intermediate partial products, plus box-conditional row emission. Closing it means structural registration in `uniform_relax` + `EnvTerm` support, Regime C — a materially larger card than item 8 scoped, named in full in the Phase 5.2 block. **Binding protocol added: staff the census by LADDER POSITION, not by attributed wall** |
| 9 | **Card 5.2-T** — `term_trilinear` | SCOPED, NOT STARTED (owner) | Not a new envelope family — the producer declines on mere *registration* of a trilinear product. Closing it is a producer-gate change plus a Regime-C panel |
| 10 | **`DISCOPT_CONVEX_KERNEL`** | cert-clean and quality-clean, graduation NO | Net-positive on a population this environment does not have (Phase 5.4 verdict) — needs the heldout/local corpus |
| 11 | **Card 4b modules 2–5** → **the state-object prerequisite** | **STARTED AND PARTLY LANDED 2026-07-30; the card itself does NOT close** | The prerequisite is built and proven: `python/discopt/solver/state.py` (4 `slots=True` dataclasses), **21 of 153** cross-region locals threaded, two Regime-N-clean panel runs, an AST-proof migration tool (`thread_solve_model_state.py`) and a both-directions conformance test. Closing it means threading the remaining ~115 — chiefly the `root`→`loop` 68 — then the carve is a separate card. The census also **corrects this row's own premise**: the coupling is 153 names on two boundaries, not "200+ locals", and `setup`/`reformulate` are nearly self-contained (14 / 18 outbound) |
| 12 | **Phase 7** — islands, refusal tests, deferred tail | not started, unblocked, cheap | Support-tier docstrings on the six zero-inbound packages; one test per load-bearing refusal (`multistage.py:47`, `gdpopt_loa.py:628`); file the SOTA long-tail entry experiments as issues |
| 13 | **heldout50 panel** | never run here | Local-only corpus. Every card that names it has recorded "SKIPPED — local only" |
| 14 | **28 parked flags** | default-OFF, each with its own record | Each needs its own Regime-C graduation panel; several (this one, `DISCOPT_CUT_INHERIT`, `DISCOPT_CONVEX_KERNEL`, `DISCOPT_FBBT_SEED`) have measured refusals recorded and should not be re-proposed without new evidence |
| 15 | **The Regime-N panel is no longer reproducible instance-for-instance here** (filed 2026-07-30) | **ROOT-CAUSED AND CLOSED 2026-07-30** — and the premise was wrong: it is *not* "Phase 0 work, not solver work" | The cause is the solver, not the harness: the search reads the wall clock at **78** Python decision sites, and the root primal heuristic (`integer_local_search`, `time_budget=min(5.0, 0.15·time_limit)`) runs to its deadline every time (5.02 s consumed of 5.00 s). Forcing that budget alone steps `gear2` 3 → 91 → 93 nodes with nothing else changed. Fixed at the gate by **replicate-and-agree adjudication** (Phase 0 addendum); the alternative — widening the `comparable` filter — was **rejected**: it would have to exclude `gear2`-class rows (`optimal`, certified, 15 % of budget) permanently, weakening detection in violation of §0.4, and it does not even cover `ex1266`, whose flake was whole-solve starvation with a 0.03 s root |
| 15b | **The solver decides how much work to do by reading a clock** (filed 2026-07-30 by item 15) | measured, not fixed — the *real* fix behind item 15's gate-level remedy | Deterministic work budgets (LP iterations, sub-NLP counts) in place of wall-clock budgets on `integer_local_search` and its 77 siblings, so an identical model + `time_limit` gives an identical tree on any machine. Solver change, Regime C (it changes the search), its own card. Until then adjudication tells a card *whether* a deviation is code-induced, but the tree is genuinely not reproducible across machine speeds |

Parallelism guidance for Opus sessions: Phases 1, 2, 7 are safe concurrently
(disjoint files). Phases 3 and 4 both edit `solver.py` — serialize them. Phase 5
touches `crates/` + producer files — safe alongside 1/2/7, coordinate with 3c
(whose parity test is the guard Phase 5's expansions must keep green).

## §6. Falsification log (append-only, per §0.3)

### 2026-07-30 — Open-ledger item 8 / Phase 5.2 entry experiment: "`infinite_aux_bounds` (census rank #2, 9 instances, 19.7 % of wall) is the better next kernel-coverage card than `term_trilinear`" — **FALSIFIED. Recoverable wall is 0.0 %, not 19.7 %, and the decline is not producer-side at all**

The ledger's item-8 row and the Phase 5 scoping note both argue rank #2 is the better
next card *because* rank #1's headline was measured to be inflated by decline-ladder
masking. **The same masking applies to rank #2, and harder.** Two pre-stated
hypotheses, both killed.

Probe: `discopt_benchmarks/scripts/phase52_infinite_aux_entry.py`; artifact
`reports/phase52_infinite_aux_entry_4f3cd17d.json`. Method mirrors the 5.1 census —
one subprocess per instance, `DISCOPT_NATIVE_SPATIAL_KERNEL=1`, the producer **tapped
at its real call site** (the presolved root box, not declared bounds). The child does
not wrap the diagnosis in `try` (§7); it asserts `discopt.__file__`, `producer_stats`,
and that `spatial_producer` contains **exactly one** `np.isinf(` call site before
installing the bypass shim arm E8.1 depends on (§8). Executed: **9 instances
diagnosed, 0 crashed, 9 reproduce calls, 9 bypass calls, 9 relaxations built, 3,846
columns examined, 229 infinite columns examined, 115,766 FBBT row scans, 856 FBBT
tightenings.** Load 0.35 → 1.29 on 4 cores.

**E8.1 — framing.** Hypothesis: with the `infinite_aux_bounds` gate bypassed, **≥ 6 of
9** reach row-claiming. Kill: ≤ 4. **Measured: 1 of 9.**

| instance | next decline with the gate bypassed | baseline wall |
|---|---|---|
| `carton7` | **NONE — spec built** | 78.0 s |
| `4stufen` | `blf_row_count:0` | 47.2 s |
| `beuster` | `term_trilinear` | 50.2 s |
| `contvar` | `term_trilinear` | 51.1 s |
| `ex14_1_9` | `fixed_row_box_dependent:coeffs` | 1.4 s |
| `gear4` | `term_ratio` | 2.0 s |
| `hda` | `term_ratio` | 56.6 s |
| `heatexch_gen1` | `term_univariate:log` | 46.8 s |
| `heatexch_gen2` | `term_univariate:log` | 46.3 s |

And the one that reaches row-claiming **is the gate being right, not the gate being
wrong**: `carton7`'s 24 infinite columns are all *original* columns, its presolved root
box carries 24 infinite entries, and the bypassed spec's `global_lo`/`global_hi` carry
**24 infinite entries** — an unbounded node LP whose safe-bound evaluation is invalid.
So the producer-side recoverable wall at rank #2 is **0.0 s / 0.0 %**, against an
attributed **379.6 s / 19.7 %**.

**E8.2 — repairability.** Hypothesis: the infinities are a forward-interval-propagation
artifact, repairable by FBBT over the relaxation's **own rows** at the root box (sound:
a bound implied by `A_ub x <= b_ub` cuts no LP-feasible point, and McCormick
relaxations are nested under box inclusion so a root-derived aux bound stays valid at
every child). Kill: ≥ 5 of 9 survive. **Measured: 9 of 9 survive — FBBT to fixpoint
finitized ZERO of the 229 infinite columns**, despite applying 856 tightenings
elsewhere.

| instance | cols | inf cols | orig/aux | in no row | presolved-box inf | inf after FBBT |
|---|---|---|---|---|---|---|
| `4stufen` | 230 | 17 | 0/17 | 13 | 0 | 17 |
| `beuster` | 282 | 20 | 0/20 | 16 | 0 | 20 |
| `carton7` | 528 | 24 | 24/0 | 0 | 24 | 24 |
| `contvar` | 1095 | 64 | 8/56 | 38 | 8 | 64 |
| `ex14_1_9` | 5 | 1 | 1/0 | 0 | 1 | 1 |
| `gear4` | 16 | 2 | 2/0 | 0 | 2 | 2 |
| `hda` | 1131 | 6 | 0/6 | 3 | 0 | 6 |
| `heatexch_gen1` | 220 | 40 | 16/24 | 16 | 16 | 40 |
| `heatexch_gen2` | 339 | 55 | 20/35 | 25 | 20 | 55 |

**Why it cannot be repaired, and this is the transferable finding: the code name is a
misnomer, and it hides two disjoint classes, neither of them producer-side.**

1. **6 of 9 — the *presolved root box itself* is unbounded** (`carton7` 24,
   `heatexch_gen2` 20, `heatexch_gen1` 16, `contvar` 8, `gear4` 2, `ex14_1_9` 1
   infinite entries on **original** columns). The solver's own FBBT/OBBT could not
   bound those variables. No relaxation-side or producer-side change reaches this:
   a McCormick envelope over an unbounded box does not exist. This is the
   propagation-quality lever `docs/dev/issue-764-scip-comparison.md` names, and it
   lives in presolve, not in the kernel producer.
2. **3 of 9 (`4stufen`, `beuster`, `hda`) — free aux columns from atoms the relaxer
   *refused to envelope***. 13 of `4stufen`'s 17, 16 of `beuster`'s 20 and 3 of
   `hda`'s 6 infinite columns have `row_nnz == 0` — they appear in **no relaxation row
   at all**. That is `uniform_relax`'s documented behaviour ("when a nonlinear atom
   cannot be enveloped … its aux column is a free interval-floor column"), so FBBT has
   nothing to propagate through **by construction**, and the kernel has no envelope
   rows to regenerate. The producer is not being conservative; there is no relaxation
   to port.

**Verdict:** item 8's card, as scoped, is dead. The correct sequencing statement — and
the reason the census ranking must not be read off the first decline code — is below.

### 2026-07-30 — Open-ledger item 8: the census ranking re-derived by ladder position — **the unmaskable tail is ranks #3 and #5, not #1 or #2**

Filed by the entry experiment above, because two of the census's top five have now been
measured to be masked and the plan has twice been surprised by it.

The producer's decline ladder is **ordered**, so an attributed wall is an upper bound
*exactly to the extent the code is tested early*. That is a static property of
`spatial_producer.build_spatial_kernel_spec` and it can be read off the source without
running anything:

| ladder position | code | maskable by an earlier code? |
|---|---|---|
| 4 | `probe_objective_bound_invalid` (rank #4, `tspn*`) | **yes** (everything below it) |
| 7 | `infinite_aux_bounds` (rank #2) | **yes** |
| 8–12 | `term_trilinear` (#1) / `term_multilinear` / `term_ratio` (#7) / `term_univariate` (#8) | **yes** |
| 13 | `blf_row_count` (#6) | **yes** |
| 15 | `probe_real_shape_mismatch` (rank #3) | **NO** |
| 16 | `fixed_row_box_dependent` (rank #5) | **NO** |

Ranks #3 and #5 sit at the *bottom* of the ladder: every instance in them has already
passed **every other** decline test, so their attributed wall is recoverable wall by
construction — no re-measurement needed to establish it, unlike #1 and #2.

| code | inst | attributed | measured recoverable | source |
|---|---|---|---|---|
| `probe_real_shape_mismatch` (#3) | 3 | 158.9 s / 8.2 % | **158.9 s / 8.2 %** | unmaskable by ladder position |
| `term_trilinear` (#1) | 17 | 437.6 s / 22.7 % | 118.5 s / 6.2 % | Card 5.2-T |
| `fixed_row_box_dependent:coeffs` (#5) | 9 | 109.7 s / 5.7 % | **109.7 s / 5.7 %** | unmaskable by ladder position |
| `infinite_aux_bounds` (#2) | 9 | 379.6 s / 19.7 % | **0.0 s / 0.0 %** | the entry experiment above |

So the next card is **rank #3 + rank #5 — 12 instances, 268.6 s, 13.9 % of corpus
wall** — and they are worth taking together because both are *the same shaped failure*:
the producer's probe-vs-real row comparison finds a row it did not claim as an envelope
whose **shape** (rank #3) or **coefficients** (rank #5) move with the box. Card 5.2-T
independently predicted this mechanism for the trilinear residual ("trilinear RLT rows,
box-dependent, `DISCOPT_TRILINEAR_RLT` defaults ON … those must be excluded from the
producer's build the same way `skip_separable_floor` and `skip_convex_lift` already
are"). Whether that prediction is the actual cause on ranks #3/#5 is the next entry
experiment, recorded below.

### 2026-07-30 — Open-ledger item 8 re-pick / Phase 5.2 entry experiment: "the unmaskable tail (ranks #3 + #5) is a producer *claiming-predicate* fix" — **FALSIFIED. The rows belong to lifts the relaxer registers in no map at all; the fix is relaxer-side, not producer-side**

The re-pick from the ranking above. Ranks #3 (`probe_real_shape_mismatch`, 3 instances,
158.9 s) and #5 (`fixed_row_box_dependent`, 9 instances, 109.7 s) are the producer's
last two tests, so their 268.6 s / 13.9 % is recoverable wall by construction. Both are
the same shaped failure: the producer identifies structure on a probe box and validates
it against the real box, and every row it did not claim as a term envelope must be
box-independent.

**E8.3.** Hypothesis: the offending rows are envelope rows of terms the relaxer *does*
register, which the producer failed to *claim* because of its support/row-count
predicate — so the fix is a claiming-predicate change inside the producer. Kill: ≥ 6 of
12 have offending rows over aux columns registered in **no** map. **Measured: 9 of
9 rank-#5 instances — every single one. KILLED.**

Probe: `discopt_benchmarks/scripts/phase52_boxdep_rows_entry.py`; artifact
`reports/phase52_boxdep_rows_entry_603ff43a.json`. The producer's own state is read,
never re-implemented: `_decline` is wrapped so that on the decline of interest it
captures the **caller frame's locals** (`env_rows`, both builds, the claim
bookkeeping) — re-deriving row ownership in the probe would measure the probe.
Executed: **12 instances diagnosed, 0 crashed, 12 producer calls, 12 frames captured,
717 unclaimed rows checked, 393 offending rows found, 3 shape comparisons.** Load
0.20 → 0.95.

**Rank #5 — the offending rows, and what owns them.** The producer returns on the
*first* offender; the probe walks every unclaimed row so the class is characterized:

| instance | unclaimed rows | offending | aux owners of the offenders |
|---|---|---|---|
| `ex1221` | 11 | 4 | `unregistered_aux` |
| `ex1225` | 34 | 15 | `unregistered_aux` ×12, `blf,unregistered_aux` ×3 |
| `ex1226` | 32 | 18 | `unregistered_aux` ×15, `blf,unregistered_aux` ×3 |
| `ex1233` | 228 | 48 | `unregistered_aux` |
| `kall_circles_c8a` | 376 | 288 | `unregistered_aux` |
| `nvs04` | 4 | 4 | `affine_square,unregistered_aux` |
| `nvs08` | 11 | 8 | `univariate:sqrt,unregistered_aux` ×4, `unregistered_aux` ×4 |
| `st_e02` | 10 | 4 | `unregistered_aux` |
| `st_e15` | 11 | 4 | `unregistered_aux` |

**The mechanism, named.** Cross-tabulating the relaxer's own `coverage` (which
relaxation *kind* it applied to each node) against the structural maps the producer
reads localizes it exactly: it is the **`power` kind whose lift lands in none of
`monomial_map` / `univariate_square_map` / `affine_square_map`**, plus
factorable-intermediate partial products from the McCormick fold. On **6 of 9** the
arithmetic is exact — offending rows = **4 × (unregistered `power` lifts)**:

| instance | `power` kinds applied | registered non-BLF aux | unregistered | 4× | offenders |
|---|---|---|---|---|---|
| `kall_circles_c8a` | 72 | 0 | 72 | 288 | **288** |
| `ex1233` | 28 | 16 | 12 | 48 | **48** |
| `ex1221` / `st_e15` | 2 | 1 | 1 | 4 | **4** |
| `nvs04` | 3 | 2 | 1 | 4 | **4** |
| `st_e02` | 3 | 2 | 1 | 4 | **4** |

The other three (`ex1225` 15, `ex1226` 18, `nvs08` 8) carry additional offenders over
factorable-intermediate columns — `ex1225` applies **no** `power` kind at all and still
has 15, so partial-product intermediates are a second, independent unregistered
population.

**Rank #3 is the same root cause seen from the other side.** All three instances have
**identical** term-family counts and coverage kinds across the two builds, and the real
build has strictly *fewer* rows — `casctanks` 2594 → 2578 (−16), `st_e35` 274 → 267
(−7), `st_e04` 29 → 28 (−1), with the column count unchanged in every case. So no lift
appeared or vanished; an **envelope row emission is conditional on the box** (a
degenerate or coincident row dropped at the real box). That is relaxer-side too.

**Verdict.** Closing ranks #3 and #5 means giving those lifts a structural registration
(and, where the family is not already an `EnvTerm`, kernel support) inside
`uniform_relax` — a relaxer + kernel change. The pre-stated kill criterion said item 8
stops with the measurement rather than starting that card, and it does. Nothing was
built; no flag was registered; no default changed.

### 2026-07-30 — Open-ledger item 8 close-out: what was run, and what item 8 leaves open

Recorded per §0.6. Tree: `4f3cd17d` (item 11's close-out) + this session's commits.
`HEAD == origin` asserted before every commit.

**Three entry experiments, three kills, zero solver-code changes.** The session
produced no behavior change, so there is **no Regime C panel and no graduation** — per
§0.3 a card that cannot survive its entry experiment does not proceed to a build, and
per the item-8 brief a graduation shipped default-ON without a clean panel is the worst
possible outcome. What the corpus now says, in one table:

| census rank | code | inst | attributed | **measured recoverable** | owner of the fix |
|---|---|---|---|---|---|
| #1 | `term_trilinear` | 17 | 437.6 s / 22.7 % | 118.5 s / 6.2 % | producer gate (Card 5.2-T, ledger row 9) |
| #2 | `infinite_aux_bounds` | 9 | 379.6 s / 19.7 % | **0.0 s / 0.0 %** | presolve propagation (6/9) + relaxer (3/9) — **not the kernel** |
| #3 | `probe_real_shape_mismatch` | 3 | 158.9 s / 8.2 % | 158.9 s / 8.2 % | **relaxer** (box-conditional row emission) |
| #4 | `probe_objective_bound_invalid` | 4 | 155.3 s / 8.1 % | not measured this session | G-F bound strength |
| #5 | `fixed_row_box_dependent` | 9 | 109.7 s / 5.7 % | 109.7 s / 5.7 % | **relaxer** (unregistered `power` lifts + partial products) |

**The single most useful thing this session establishes:** the census ranking is a
ranking of *first decline codes*, and its rows must be read by **ladder position**
before they are staffed. Two of the top five have now been re-measured and both
collapsed (#1 to 27 % of its headline, #2 to zero). The two rows that were **never**
inflated — #3 and #5, the producer's last two tests — were ranked third and fifth and
are now the largest genuinely-recoverable entries. Nothing in the census file needs
correcting; the reading protocol does, and it is written down here.

**Suites run on the final tree** — see the verification block appended to the Phase 5.2
card below.

### 2026-07-30 — Open-ledger item 11 close-out: what was run on the final tree

Recorded per §0.6. Tree: `d7b374f6` (item 15's close-out) + this session's
commits. `HEAD == origin` asserted before every commit — the branch had not moved
under this session at any point. Compiled `.so` marker asserted before **every**
panel run: `strings _rust.cpython-311-x86_64-linux-gnu.so` finds
`subtol_crossings_repaired` (CLAUDE.md §8), and `discopt.__file__` resolves to
`/home/user/discopt/python/discopt/__init__.py`, i.e. the tree under test and not
a site-packages copy.

**What landed.**

| commit | what |
|---|---|
| `ab8235dc` | the census: `discopt_benchmarks/scripts/solve_model_locals_census.py`, `reports/solve_model_locals_census.{json,txt}` on the unmodified tree, plus the AST-proof migration tool `thread_solve_model_state.py`. No solver code touched |
| `4665bc40` | `python/discopt/solver/state.py` + `PhaseTimers` (4 locals, 46 sites) and `PrimalHeuristicState` (7 locals, 44 sites) threaded; `python/tests/test_solver_state.py` |
| `d7964ece` | the Regime-N gate artifact for `4665bc40` |
| `820246ec` | `LazyStallSeparationState` (6 locals, 29 sites) and `PerNodeOBBTBudget` (4 locals, 15 sites) threaded; the migration table made two-directional |

**No code was moved out of `solve_model`.** That was the scope boundary and it
held: `solve_model` is still one function, still in `solver/__init__.py`, and the
carve is untouched. What changed is that 21 of its locals are now fields of named
typed objects instead of members of an implicit closure.

**Proof of purity — why this is a rename and not an edit.** Every one of the four
migrations was performed by `thread_solve_model_state.py`, which does the textual
rewrite from AST node positions and then *proves* it: it applies the same
substitution to the original AST with a transformer and refuses to write unless
`ast.dump(transformed_original) == ast.dump(reparsed_rewritten)` for
`solve_model`, and unless all **125** sibling top-level definitions are
AST-identical to their pre-rewrite selves. Executed comparisons per run: 126.

| holder | locals | sites | reference substitutions | siblings proved identical |
|---|---|---|---|---|
| `_timers` | 4 | 46 | 46 | 125 |
| `_heur` | 7 | 44 | 44 | 125 |
| `_lazy` | 6 | 29 | 29 | 125 |
| `_pn_obbt` | 4 | 15 | 15 | 125 |

The tool also refuses when a migrated name exists at module level, because there a
*missed* site would silently read the global instead of raising `NameError`
(CLAUDE.md §7). None of the 21 collided; the refusal arm was never needed but is
the reason a missed site is loud rather than silent.

It refused once, correctly: `_lazy_glb_ref` was an `AnnAssign`, and rewriting the
target to an attribute changes the node's `simple` flag, so the proof would not
close. Converted to a plain assignment by hand first — behaviour-neutral by
PEP 526 (annotations on *local* variables are never evaluated at runtime) — and
the type now lives on the dataclass field.

**Regime N — the hardened gate, twice, once per threading commit.**

| tree | verdict |
|---|---|
| `4665bc40` (11 locals) | **PASS.** `comparisons executed (total): 255 = 255 first-pass + 0 adjudication over 85 comparable row(s); flagged 0, adjudicated 0, transient 0`. 1988.5 s, load start 0.17 peak 2.34. Artifact `reports/item11_panel_check_commit2.log` |
| `820246ec` (21 locals) | **PASS.** `comparisons executed (total): 255 = 255 first-pass + 0 adjudication over 85 comparable row(s); flagged 0, adjudicated 0, transient 0`. 1949.1 s, load start 0.48 peak 1.81. Artifact `reports/item11_panel_check_commit3.log` |

Zero rows were flagged on either run, so nothing reached adjudication and there
are **no TRANSIENT rows to disclose** — per the Phase 0 addendum a card that hides
transients has not reported its measurement, and here there are none. The 16
non-comparable drift rows the gate prints are budget-dependent terminal statuses
and are not gating; they are the same population Phase 3, Phase 5, Card 3e, Card
4a, Card 4c and item 15 all saw. **Ledger row 15b applies and was honoured**: the
verdict quoted is the adjudicator's, not a raw first-pass flag — there was simply
nothing to adjudicate.

**Suites.**

| suite | result |
|---|---|
| `pytest python/tests -m smoke` | **PASS — 953 passed**, 16 skipped, 7,756 deselected, 2 xpassed (456.8 s). Pre-session 947; the +6 are `test_solver_state.py` |
| `pytest discopt_benchmarks/tests -m smoke` | **PASS — 53 passed, 1 skipped**, 378 deselected (32.3 s) |
| `pytest -m slow python/tests/test_adversarial_recent_fixes.py` | **PASS — 10 passed** (208.4 s) |
| `test_routing.py` + `test_tightening_schedule.py` + `test_node_tightening_parity.py` + `test_vector_constraint_corpus.py` + `test_solver_state.py` | **PASS — 73 passed**, 12 deselected (12.7 s) |
| `test_flag_registry.py` | **PASS — 17 passed** |
| `mypy 2.1.0 python/discopt/ --ignore-missing-imports` | **Success — 320 source files.** Version asserted, not assumed: a local shim would have reported a false Success (CLAUDE.md §8 applied to the type checker) |
| `ruff check python/`, `ruff format --check python/` | clean, apart from two offenders (`test_log_square_relaxation.py`, `test_mo_augmecon2.py`) verified dirty on `d7b374f6` before this session touched anything |
| `cargo test -p discopt-core` | **not run — `crates/` untouched** |
| heldout50 | **SKIPPED — local only.** Not available in this environment |

The two AST conformance suites Card 3a(a) and Card 4a left behind
(`test_tightening_schedule.py`, `test_routing.py`) assert marker *order inside
`solve_model`*; they pass unchanged, which is the expected result for a change
that renames locals and moves no statements. Neither was edited or silenced.

**The monkeypatch hazard Card 4b measured did not apply, and that is a fact rather
than an assumption.** The 30+ patched `discopt.solver` attributes are module-level
*functions*; this card moved no function and re-exported nothing, so no
`monkeypatch.setattr` seam changed meaning and the negative assertion in
`test_expired_outer_deadline_skips_native_seed_work` cannot have gone vacuous. The
standing re-export policy test in `test_spatial_native_kernel.py` passes.

**Effect, measured on the census.**

| | before | after |
|---|---|---|
| cross-region locals | 153 | **136** |
| `root`→`loop` | 85 | **68** |
| `root`→`results` | 44 | **41** |
| `loop`→`results` | 24 | **17** |

The arithmetic closes exactly: −21 migrated, +4 holders (the holders are
themselves cross-region locals).

**Can item 11 close? NO — and the honest-outcome clause does not apply either.**
The census's kill criterion came back negative: the crossing names *do* cluster,
so the object is not a cosmetic rename and the work is worth doing. But 21 of 153
is a fifth of it. What remains is bounded and mechanical rather than uncertain —
the tool, the gate, the conformance test and the naming policy all exist and are
proven — and the next groups are named by the census itself: the McCormick
relaxation handles (`_mc_lp_relaxer`, `_mc_mode`, `_mc_obj_relax_fn`,
`_mc_con_relax_fns`, `_mc_con_senses`, `_mc_negate`, `_mc_obj_eval`,
`_mc_nlp_period`), the dual-bound certificate flags (`_gap_certified`,
`_nonrigorous_fathom`, `_taint_floor_internal`, `_tree_bound_poisoned`,
`_root_glb_internal`, `_convex_bound_untrusted`, `_root_rigorously_infeasible`,
`_root_pool_bound`, `_debug_quit`), and the read-only root context (`evaluator`,
`tree`, `t_start`, `_deadline`, `n_vars`, `int_offsets`, `int_sizes`, `cl_list`,
`cu_list`, `constraint_bounds`, `opts`).

The certificate cluster is deliberately **not** in this session's increment. It is
the soundness-critical group — it decides whether a bound may be reported as
certified — it has the widest lifetime in the function (13 rebinds of
`_gap_certified` alone, across all three regions), and it deserves a commit whose
gate run is about nothing else. Bundling it behind two cheap increments would have
made a `CONFIRMED` verdict unattributable, which is the whole reason this card
worked in small commits.

**One thing this card did NOT establish, stated so no later card assumes it.**
Threading a local onto a `slots=True` dataclass replaces a `LOAD_FAST` with a
`LOAD_ATTR`, and per ledger row 15b the solver decides how much work to do by
reading a clock at 78 Python sites — so a *large enough* slowdown could move a
node count without any logic changing. Two panel runs at 21 locals show no such
effect (0 flagged rows both times), but that is evidence at this scale, not a
proof that the remaining ~115 are free. A later increment that threads a local
read inside the innermost node loop should watch for it rather than assume it
away.

### 2026-07-30 — Open-ledger item 11 entry experiment: "`solve_model`'s locals are a 200+ closure that cannot be separated" — **HALF FALSIFIED. The closure is real; 200+ is not the coupling, and the coupling is concentrated on two boundaries**

**Hypothesis (Card 4b's own, restated as item 11's premise).** The four dropped
modules (`setup`, `reformulate`, `root`, `spatial_loop` — and `results`) share "a
closure of 200+ locals", so carving them is a signature-design problem that needs
an explicit state object first.

**Kill criterion, stated before the measurement.** If the mutable search state and
the read-only configuration are entangled such that no cohesive grouping exists —
i.e. the crossing names do not cluster, or nearly every local crosses nearly every
boundary — then any object is a cosmetic rename, item 11 buys nothing, and it
stops with that recorded (the honest-outcome clause).

**Instrument.** `discopt_benchmarks/scripts/solve_model_locals_census.py`. It
walks `solve_model`'s own scope (nested `def`/`lambda`/`class`/comprehension
scopes handled as separate scopes, as Python does), classifies every binding, and
prints an executed-classification count that must be non-zero (CLAUDE.md §6).
Regions are anchored on **the source's own phase banners**, not line numbers, and
the script refuses if an anchor matches zero or more than one line rather than
guessing. Artifacts: `reports/solve_model_locals_census.json` (per-name rows) and
`reports/solve_model_locals_census.txt` (the printed summary), both produced on
the unmodified tree `d7b374f6`.

**Result 1 — the shape of the function.** 7,622 LOC, 319 top-level statements, 51
parameters, 41 nested scopes, one 2,495-line inline `while`. **851** names bound in
its own scope:

| class | count | definition |
|---|---|---|
| `CONFIG` | 205 | bound once, pre-loop, never augmented or rebound |
| `STATE` | 404 | rebound, augmented, bound inside the loop, or `del`'d |
| `SINGLE_USE` | 104 | bound once, read once |
| `DEAD` | 6 | bound, never read |
| `CALLABLE` | 132 | nested `def`/`class`/function-local `import` |

**Result 2 — "200+ locals" is not the coupling; 153 is, and it is lopsided.**
Regions named for Card 4b's five modules (`setup` 1,235 / `reformulate` 582 /
`root` 2,685 / `loop` 2,495 / `results` 625 LOC). Bind-region → read-region:

| bound in ↓ / read in → | setup | reformulate | root | loop | results |
|---|---|---|---|---|---|
| **setup** | 42 | 4 | 10 | 3 | 1 |
| **reformulate** | 4 | 33 | 12 | 3 | 3 |
| **root** | 6 | 5 | 151 | **85** | **44** |
| **loop** | 1 | 0 | 13 | 287 | 24 |
| **results** | 0 | 0 | 6 | 5 | 44 |

Only **153** names are bound in one region and read in another. Two boundaries
carry almost all of it — `root`→`loop` (85) and `root`→`results` (44) — and the
two modules Card 4b listed first are nearly self-contained (`setup` has 10 + 3 + 1
outbound; `reformulate` 12 + 3 + 3). The premise that the whole function is one
undifferentiated closure is **wrong in the direction that matters**: it is one
*hard* boundary, not five.

**Result 3 — two properties that make the migration provable rather than hopeful.**

* **Zero `nonlocal` writes** across all 41 nested scopes. The 40 closure captures
  are **reads only**. A carve never has to invent a write-back path, and a state
  object can be passed by reference with no aliasing subtleties.
* **No `locals()`, `globals()`, `eval` or `exec` anywhere in `solve_model`.** Every
  reference is statically resolvable, so a migration can be *proved* by AST
  comparison instead of being argued from a passing test run.

**Result 4 — the crossing names cluster, which is what decides the object's
shape.** The 153 are not an arbitrary set; they fall into cohesive groups that
traverse `root`→`loop`→`results` **together** — Rust/JAX timing accounting;
sub-NLP and LNS heuristic budgets; lazy-constraint arming and separation counters;
the per-node OBBT budget; the McCormick relaxation handles; and the dual-bound
certificate flags (`_gap_certified`, `_nonrigorous_fathom`, `_taint_floor_internal`,
`_tree_bound_poisoned`, …). This is the kill criterion coming back **negative**:
a grouping exists, so the object is not a cosmetic rename.

**Verdict.** The premise survives in substance — an explicit state object is
genuinely the prerequisite for the carve — and is falsified in its arithmetic. The
number to design against is **153 crossing names on two dominant boundaries**,
grouped into roughly half a dozen clusters, not "200+ locals" spread evenly. Item
11 proceeds, and it proceeds as *a set of small cohesive dataclasses*, which is a
conclusion the census produced and intuition did not: before measuring, a single
`SolveModelState` god-object was the obvious design, and the matrix says it would
have been the wrong one.

**Six dead bindings** were also found and are recorded rather than deleted (a
deletion is Regime N but belongs in its own commit, not smuggled into a rename):
`_`, `_frac_parts`, `_has_continuous_var`, `_lns_swap_applicable`, `_scores`,
`inc_sol`.

### 2026-07-30 — Card 4c close-out / Card 3a(b) close-out: what was run on the final tree

Tree: `6f815214` + the three commits of this session. `HEAD == origin` asserted
before each commit. Compiled `.so` marker asserted **before every measurement**:
`strings` on `_rust.cpython-311-x86_64-linux-gnu.so` finds `subtol_repaired` and
`subtol_crossings_repaired`, the PyO3 strings unique to the newest Rust commit
`8532ce2d` — so the binary matches the newest Rust sources. **Rust untouched this
session**, therefore `cargo test -p discopt-core` was **not run** (nothing in
`crates/` changed).

| gate | result |
|---|---|
| `pytest python/tests -m smoke` | **PASS — 946 passed, 16 skipped, 7752 deselected, 2 xpassed** (478 s). Pre-session baseline 922 passed; the +24 is exactly `test_constraint_rhs_refusal.py`. |
| `pytest discopt_benchmarks/tests -m smoke` | **PASS** (exit 0) |
| `pytest -m slow test_adversarial_recent_fixes.py` | **PASS — 10 passed** (222 s) |
| `pytest test_node_tightening_parity.py` | **PASS — 4 passed, 12 deselected** |
| `pytest -m slow test_node_tightening_parity.py` | **PASS — 12 passed, 4 deselected** (77 s) — Card 3c's guard, including the vector arm |
| `pytest test_vector_constraint_corpus.py` | **PASS — 37 passed** |
| `pytest test_flag_registry.py` | **PASS — 17 passed.** No flag was added this session, so `docs/reference/flags.md` needs no regeneration and the staleness test confirms it. |
| `pytest -m slow test_stray_bb_loop_invariants.py` | **PASS — 7 passed** (148 s). Final-tree counts: `i1_fathom_decisions` **13,758**, `i1_fathomed` 3, `i2_bound_vs_oracle` 3, `i3_incumbent_verified` 5, all three loops observed. (The decision count is budget-dependent through `nvs17`, which is time-limited: an earlier run on the same tree recorded 12,905 with 12,747 from that instance. The *invariant* verdict is not budget-dependent; only how many decisions get audited is.) |
| `pytest test_constraint_rhs_refusal.py` | **PASS — 24 passed**; executed counts `refused_subject_to` 10, `refused_validate` 7, `refused_solve` 2, `normalized_solved` 4, `public_api_reachable` 1 |
| `ruff check python/` + `ruff format --check` | **PASS** on every file this session touched. Two files unrelated to this session (`test_log_square_relaxation.py`, `test_mo_augmecon2.py`) are pre-existing format drift and were left alone. |
| Regime N `panel_baseline --check reports/panel_baseline_f154dcff.json` | **PASS — `comparisons executed: 255` (node_count 85, certified objective 85, status 85) over 85 comparable of 119 baseline rows; "PASS: no node-count or certified-objective drift", `PANEL_EXIT=0`.** Ran alone after the suite block, as queued. Same 255/85 population as every prior run (Phase 3, Phase 5, Card 3e, Card 4a). Verdict recorded by the orchestrator after the implementing session ended; log `scratchpad/v_panel.log`. |
| heldout50 | **SKIPPED — local only.** Not available in this environment. |

**One non-comparable row worth a look, flagged rather than buried.** In the passing
panel, `tls2` moved `nodes 421→373, obj 5.299999922109238→9.29999987524207`. It is
correctly excluded from gating — it certifies at 43.8 s = 97 % of the 45 s budget, so
it is inherently irreproducible — but a ~76 % objective move is large enough that
"non-comparable" should not be read as "fine". Nothing this session touches explains
it, and the same row has been non-comparable in every prior panel. **Follow-up:** re-run
`tls2` alone at a 300 s budget against its reference optimum to establish which value is
right; if the 9.3 arm is the reproducible one, the baseline row is the stale one.

**Why Regime N is the right regime for everything here.** The three loop edits
hoist an existing condition into a named local and call a default-inactive hook
between — same boolean, same use, no math. The `rhs` refusal is unreachable from
any producer in the corpus or the suite (66 models / 3,705 comparisons / 0
violations; 1,947 suite constructions / 58 non-zero, all from test fixtures). Card
3a(b) landed **zero** production edits. So no bound can move, and the panel is a
confirmation rather than a differential.

### 2026-07-30 — Card 3a(b) item 3: "OBBT entry points 3 → 1" — **FALSIFIED; both stay**

**Hypothesis (H-OBBT).** `run_obbt`, `run_obbt_on_relaxation` and
`obbt_tighten_root` are three doors to one room, so two can be made private behind
the third.

**Kill criterion.** Two doors given the same model and the same starting box
returning different boxes.

**Experiment** (`discopt_benchmarks/scripts/card3a_obbt_doors_entry.py`,
`reports/card3a_obbt_doors_entry.json`). Only the type-compatible pair can even be
compared: `run_obbt(model, lb, ub, …)` vs `obbt_tighten_root(model, lb, ub, …)`,
same model, same root box, one round each, 20 s cap per door. 12 instances, **0
errors, 3,386 executed bound comparisons.**

| measure | value |
|---|---|
| disagreements | **803** |
| `obbt_tighten_root` tighter | 786 |
| **`run_obbt` tighter** | **17** |

**Verdict: FALSIFIED, and in the way that matters.** `obbt_tighten_root` is tighter
far more often — but `run_obbt` is tighter on **17** bounds, so **neither
dominates** and collapsing either into the other *loses* tightenings (Regime C,
bound-loosening, §0.2). They are different relaxations, not different spellings:
`run_obbt` tightens against the model's **linear rows**, `obbt_tighten_root`
against the **McCormick LP envelope** it builds itself. Even the return contracts
differ (`ObbtResult(tightened_lb, tightened_ub, …)` vs
`RootObbtResult(lb, ub, …, infeasible)`).

**And the third door cannot be made private on structural grounds alone.**
`run_obbt_on_relaxation` takes a **pre-built relaxation object**, not a `Model`, so
it is not substitutable for either; `amp.py` calls it at two sites with a
relaxation it already owns; and `obbt_tighten_root` **already delegates to it**
(obbt.py:2119, :2567). Making it internal would break `amp.py` and would hide the
helper the "single door" is itself built on.

**Consequence: all three stay, documented. No code change.**

### 2026-07-30 — Card 3a(b) item 1: "a single `tighten_root_bounds_with_fbbt` invocation policy" — **the redundancy is real but NOT uniform; no policy shipped**

**Hypothesis (H-FBBT-INV).** More than one of the call sites fires within a single
solve, so the root FBBT runs redundantly and an invocation policy removes real work.

**Kill criterion.** If every solve invokes it at most once, the "5+ sites" are five
*engines* each doing its own root presolve once and there is nothing to coordinate.

**Experiment** (`discopt_benchmarks/scripts/card3a_root_fbbt_invocations.py`,
`reports/card3a_root_fbbt_invocations.json`). Wrapper counts invocations per solve
with the caller's `file:line` and each call's `root_changed`. 66 instances, **0
solve failures, 94 executed invocations.**

| measure | value |
|---|---|
| invocations per solve | `{0: 7, 1: 32, 2: 23, 4: 4}` |
| solves invoking more than once | **27** |
| caller sites reached | `solver:6390` ×41, `solver:5234` ×27, `solver:11901` ×17, `solver:17108` ×9 |
| second-plus calls that changed **nothing** | **29** |
| second-plus calls that **tightened** | **6** (`4stufen`, `beuster`, `clay0303hfsg`, `cvxnonsep_psig40r`, `fac2`, `nvs05`) |

**Verdict: H-FBBT-INV holds — the repeats are real — but the card's remedy does
not follow.** The second-plus calls are **not uniformly redundant**: 6 of 35 moved
bounds (a later engine entry sees a box a prior pass had already improved, and FBBT
is not idempotent across a changed box). A "single invocation" policy that
suppresses repeats would therefore **drop 6 real tightenings** — a Regime-C
bound-loosening, which §0.2 forbids and which no gate in this environment could
excuse. The only clean saving is the 29 no-op passes: **pure wall time with zero
certificate content**, on a session whose benchmark is explicitly for
**correctness, not speed** (§5, 2026-07-30 re-sequencing).

**Consequence: measured, characterised, not consolidated.** If this is revisited as
a *performance* item, the shape is not "invoke once" but "skip a repeat whose input
box is bit-identical to the previous call's" — which is bound-neutral by
construction (Regime N) and would recover 29 of 35 repeats. Recorded as the
follow-up shape; not built here, because it is wall-clock work outside this
session's scope.

### 2026-07-30 — Card 3a(b) item 2: "the two Python reduced-cost-fixing implementations are duplicates; delete one" — **BOTH STAY. Same on the integer columns, differentiated everywhere else, and the swap cannot be gated**

**Hypothesis (H-RCF).** `_jax/node_reduce._dbbt_from_reduced_costs` and
`solver._reduced_cost_fixing` compute the same tightening on the integer columns,
so one can be deleted in favour of the other. (Regime C — deleting an *active*
tightening can loosen bounds, per §0.2.)

**Kill criterion.** One integer bound on which they disagree, on inputs drawn from
real corpus node LPs.

**Experiment** (`discopt_benchmarks/scripts/card3a_rcf_dedup_entry.py`,
`reports/card3a_rcf_dedup_entry.json`). Both entry points wrapped so whichever the
corpus reaches supplies the population; every live invocation's exact arguments
captured and replayed through **both** implementations. 66 instances, 0 solve
failures, `DISCOPT_PHASE2_DBBT=1`.

| measure | value |
|---|---|
| captured real inputs | **85** (all from the node-DBBT seam) |
| executed integer-column comparisons | **291** |
| integer lb / ub disagreements | **0 / 0** |
| continuous columns tightened by DBBT **only** | **8** |
| captures from the root-RCF seam | **0** |

**Verdict: H-RCF holds on the integer columns — and the deletion is still wrong,
in both directions.**

1. **Deleting `_dbbt_from_reduced_costs` loses an active tightening.** It also
   tightens **continuous** columns (8 observed on real inputs) and returns an
   infeasibility verdict; `_reduced_cost_fixing` iterates `int_idx` and can express
   neither. That is a Regime-C bound-*loosening* deletion, which §0.2 forbids.
2. **Deleting `_reduced_cost_fixing` cannot be gated here.** Its only consumer is
   `_root_reduced_cost_fixing` inside `_solve_milp_bb`, which routes on
   `problem_class == ProblemClass.MILP`. Measured over the whole in-repo corpus:
   **147 executed classifications, ZERO MILP** (73 MINLP, 21 MIQCQP, 15 MIQP, 14
   NLP, 11 MIQCP, 8 QCP, 3 QCQP, 2 QP) — hence 0 captures from that seam. A Regime-C
   swap with no population to certify it on is exactly the unverifiable change
   Card 4c was retired for.
3. **The 291/291 agreement is a property of the population, not of the code.** The
   two floor *different quantities*: `_reduced_cost_fixing` floors the **step**
   (`lb + floor(gap/d)`), `_dbbt_from_reduced_costs` floors the resulting **bound**
   (`floor(lb + gap/d)`), and their acceptance thresholds differ (`0.5` vs `_EPS`).
   Demonstrated directly, 3 constructed cases, 1 differs: with a *fractional*
   integer lower bound `lb=0.3, gap=3.5, d=1` they give `ub = 3.3` vs `ub = 3.0`
   (DBBT tighter, and still valid for an integer column). Every captured node had
   integral integer lower bounds, which is why the corpus never separated them.
   Recorded so nobody reads "0 disagreements" as "provably identical".

**Consequence: both stay, documented.** No code change; the card's item 2 is
answered as a measured NO, consistent with Phase 2/3's standing lesson that these
"duplicates" keep turning out to be differentiated.

### 2026-07-30 — Card 4c Task 1: "consolidating the three loops onto `PyTreeManager` improves auditability" — **FALSIFIED (the card's own premise); port RETIRED**

**Hypothesis.** Card 4c's stated value was "every certificate-critical pruning
decision now flows through one audited tree manager". That presumes the ports make
the audited component *easier* to audit.

**Kill criterion.** If a faithful port requires adding policy switches to
`PyTreeManager` — rather than removing divergence — the premise inverts: the
audited component grows, and it grows to serve three callers the certifying panel
cannot execute.

**Evidence** (the two entries below, unchanged): five policy axes diverge per loop
(selection tie-break, prune slack, branch-variable rule, split-point rule,
failed-relaxation handling), and two are not options-shaped at all —
`signomial_global` branches in **log space**, and its per-node OBBT *replaces* the
child box, neither expressible in the `export_batch`/`import_results` contract. The
gate that would certify the ports sees exactly **one** budget-independent instance
(`prob03`, 5 nodes).

**Verdict: FALSIFIED.** Five switches plus two contract extensions is a net
increase in the complexity of the audited component, and it is a cost the port pays
whatever gate certifies it — so exit (a) (materialise a population) buys a better
gate for a worse design. **Owner decision: option (b), retire.** Card 4c is
CLOSED. The per-loop policy characterisation table is preserved on the card as the
durable deliverable.

**Replacement, built and landed the same session** (the goal survives the port;
auditability comes from *observability*, not centralisation):
`python/discopt/validation/fathom_audit.py` (default-inactive hook, both arms of
every bound-fathom decision recorded in internal minimisation sense) plus
`python/tests/test_stray_bb_loop_invariants.py` (`slow`), which forces
`DISCOPT_GP_MINLP` / `DISCOPT_SGO` / `lp_spatial=True` ON — precisely because the
first entry below proved the panel never invokes two of them — and asserts I1
(no wrongful fathom, admissible slack re-derived from the declared
`gap_tolerance`, never from the loop's reported slack), I2 (dual bound never
crosses the `reference_optima` oracle), I3 (incumbent passes `verify_point`).

**Measured: 7 tests pass. 12,905 audited fathom decisions; 3 fathoms; 3 oracle
comparisons; 5 incumbents verified; all three loops observed.** A planted-violation
control proves I1 discriminates; a module-scoped finalizer fails the suite if any
counter reaches zero.

**Secondary finding — these loops almost never fathom by bound.** 3 of 12,905.
`lp_spatial_bb` fired its decision site **12,747** times on `nvs17` and fathomed
**zero** times: under best-first the popped node *is* the frontier minimum, so the
gap test fires before the bound test can, making that site a safety net rather
than a working pruner. `gp_minlp` and `signomial_global` fathom at most once per
solve (both `break` on the frontier test). Recorded so "3 fathoms" is not misread
as thin coverage of a hot path — it is complete coverage of a cold one.

### 2026-07-30 — the `Constraint.rhs` silent wrong answer: "only the private-append path is affected, and the public path is correct" — **FALSIFIED; the public API is affected too**

**Restatement.** The 2026-07-30 entry further down recorded, as an incidental
Card 4c Task 2 measurement, that a hand-appended `Constraint` with a non-zero
`rhs` is honoured by the verifier and ignored by the solver, and concluded "the
public `subject_to` path folds the offset into the body and is correct". **That
conclusion is wrong and is retracted here** (CLAUDE.md §11). `Constraint` is
exported in `discopt.modeling.__all__`; `subject_to` accepted the object verbatim
without normalising it. Measured on this tree, all three arms, one script:

    m.subject_to(w >= 5.0)                       -> w = 5.0    (operators normalize)
    m.subject_to(Constraint(w, ">=", 5.0))       -> w = -5e-09  ← PUBLIC API
    m._constraints.append(Constraint(w,">=",5.0))-> w = -5e-09

This is a **silent wrong answer through documented public API**, not a
private-attribute footgun. It is more serious than the original write-up believed.

**Root cause.** `_jax/dag_compiler.compile_constraint` / `compile_constraint_params`
compile `constraint.body` and discard `rhs`. That is the seam, but it is not an
isolated slip: the tree is **split** on whether `rhs` means anything.

**Hypothesis (H-RHS) and kill criterion.** H-RHS: a non-zero `rhs` is unreachable
through supported construction — every producer normalizes, exactly as
`Constraint`'s docstring declares ("always 0.0 in normalized form"). Kill
criterion: one constraint with `rhs != 0` reached from a supported constructor
means refusing would break working models, and the solve path must be taught to
honour `rhs` instead.

**Experiment, two arms** (`discopt_benchmarks/scripts/card4c_rhs_entry.py`; and a
pytest plugin wrapping `Constraint.__init__` over the whole smoke suite):

- Static/corpus: **66 models loaded, 3,705 executed rhs comparisons, 0 load
  failures, 0 violations.** A planted control arm registers its non-zero `rhs`, so
  the probe demonstrably sees what it claims to measure; an operator arm confirms
  the DSL normalizes.
- Dynamic/suite: **1,947 executed `Constraint` constructions across `pytest -m
  smoke`; 58 carried a non-zero `rhs`, and every one of the 58 originates in a
  TEST fixture** (`vector_constraint_corpus.py`, `test_incumbent_verifier_scale.py`
  — both deliberately built to exercise the verifier, neither of which solves).
  **Zero production producers emit one.**

**Verdict: H-RHS HOLDS.** Non-zero `rhs` is out-of-contract construction.

**The deciding measurement for the direction of the fix.** Which way to go turns
on how large the "does not honour `rhs`" surface is, so it was counted rather than
guessed: **26 modules read `Constraint.body` and never read `.rhs`** —
`dag_compiler`, `relaxation_compiler`, `milp_relaxation`, `mccormick_subgradient`,
`term_classifier`, `nonlinear_bound_tightening`, `dependent_vars`,
`implied_integer`, `differentiable`, `canonical_expr`, `sparsity`,
`edge_concave`, `integer_ratio`, `primal_heuristics`, the four
`_jax/convexity/*` certificate modules, `bilevel/{kkt,problem,strong_duality}`,
`decomposition/{_linear,benders/gbd,benders/solver}`, `ro/formulations/_common`,
`gdp_advisor`. Against them, `validation/feasibility` (`signed = body - rhs`), the
`.nl`/LP/MPS/GAMS exporters, `problem_classifier`, `_jax/obbt`, `solvers/gurobi`
and the Rust `ConstraintRepr` (**114 references** across the presolve crate) *do*
honour it.

**Direction chosen: refuse loudly at the model boundary** (CLAUDE.md §3). Teaching
the solve path to honour `rhs` means correcting all 26, each of which encodes the
`body sense 0` form *structurally* rather than arithmetically. A partial job is
strictly worse than the status quo: today the relaxation stack is uniformly
rhs-blind, so its McCormick envelope still relaxes the same row the verifier
checks; half-honoured, the envelope would be built for a **different row** than the
one being verified — a soundness hazard replacing a wrong-answer hazard. Refusing
preserves the one invariant all 26 modules already rely on and costs the caller a
one-line rewrite the error message spells out.

**Shipped.** `modeling/core._reject_unnormalized_rhs`, raised from **both** doors:
`Model.subject_to` (so the error lands on the offending line) and `Model.validate`
(the collection-level enforcement — it catches direct `_constraints.append` *and*
the internal rebuilds that propagate `c.rhs` verbatim, and `solve` always
validates). Array `rhs` is handled with `np.any`, not `float()`, which would have
raised a confusing `TypeError` on a vector row. Regression suite
`python/tests/test_constraint_rhs_refusal.py` (`smoke`, 23 tests) tests **both**
directions the finding named: the refusal fires at both doors for scalar/vector,
named/unnamed, `<=`/`>=`/`==`; and normalized rows still solve to the right answer
— the control that proves the guard discriminates rather than rejecting
everything. The vacuity guard was demonstrated to fire (deselect the arms → the
module finalizer errors).

**Not a Regime-C change.** No bound moves: the refusal is unreachable from any
producer in the corpus or the suite (measured above), so this is Regime N and the
panel must be exactly unchanged.

### 2026-07-30 — Card 4c: "the Regime-N panel can gate the three stray-loop ports" — **FALSIFIED**

**Hypothesis.** Card 4c's per-loop gate is `panel_baseline.py --check` with exact
node/objective match. That presumes the 119-instance panel actually *invokes*
`lp_spatial_bb.solve_lp_spatial_bb`, `gp.solve_gp_minlp` and
`signomial_global.solve_signomial_global`.

**Kill criterion.** Any loop the panel cannot reach makes its own Regime-N PASS
vacuous (CLAUDE.md §6: a checker that compares nothing reads exactly like a pass),
and the port must be verified another way or not attempted.

**Experiment** (`discopt_benchmarks/scripts/card4c_reachability.py`,
`reports/card4c_reachability.json`). Static: `panel_baseline.py:202` states
"NOTHING here sets a `DISCOPT_*` flag", and `routing.py:218-245` gates
`auto_gp_minlp` behind `env_bool("DISCOPT_GP_MINLP", False)` and
`auto_signomial_global` behind `env_bool("DISCOPT_SGO", False)` — both
**default-OFF**, both short-circuited *before* their classifier runs. Dynamic: all
119 corpus instances loaded and classified, **357 executed classifications, 0 load
errors, 0 classifier errors**.

- `classify_gp_minlp` accepts **3**: `cvxnonsep_nsig30`, `cvxnonsep_psig30`, `prob03`.
- `classify_signomial_global` accepts **5**: `cvxnonsep_nsig30`, `prob02`, `prob03`,
  `prob06`, `st_e38`.
- `classify_gp` accepts **0** (context: the pure-GP route never fires on this corpus).

**Verdict: FALSIFIED.** On defaults the panel invokes neither the GP-MINLP nor the
signomial loop at all, so a post-port `--check` PASS would be evidence of nothing.
`lp_spatial_bb` is the exception — it is default-reachable, but only through
`modeling/core.py:4293`'s no-incumbent fallback (`_fb_reserve > 1.0 and
result.objective is None`), i.e. on the handful of instances the primary path
leaves without an incumbent. **Consequence:** the ports were not attempted; see
the next entry for why forcing the flags ON does not rescue the gate either.

### 2026-07-30 — Card 4c: "the GP loop's divergences from `PyTreeManager` are immaterial, so its port is bound-neutral" — **NOT ESTABLISHED (population too small)**

**Hypothesis.** `gp/solve_gp_minlp` is the most tractable of the three loops (pure
integer best-first, no spatial branching, no per-node OBBT), so its port should be
bound-neutral with `PyTreeManager`'s stock policy.

**Kill criterion.** Any single policy divergence that moves `node_count` or the
certified objective on a budget-independent corpus instance proves the port needs a
matching `PyTreeManager` option and cannot be silently normalised (§0.1).

**Experiment** (`discopt_benchmarks/scripts/card4c_gp_divergence.py`,
`reports/card4c_gp_divergence.json`). `DISCOPT_GP_MINLP=1`, 60 s, the 3 instances
the previous entry identified. Baseline vs one divergence applied alone —
`PyTreeManager`'s **exact** prune (`node_lb >= incumbent`, `tree_manager.rs:459`)
in place of the loop's `>= incumbent - fathom_slack()` (`gp/__init__.py:1004`).
**6 executed comparisons, 0 errors.**

| instance | baseline | exact-prune arm | comparable? |
|---|---|---|---|
| `prob03` | optimal, **5 nodes**, obj 9.999999905 cert | optimal, **5 nodes**, same obj | **yes** (budget-independent) |
| `cvxnonsep_nsig30` | time_limit, 69 nodes, no incumbent | time_limit, 93 nodes | no |
| `cvxnonsep_psig30` | time_limit, 78 nodes, no incumbent | optimal, 85 nodes, obj 78.9989 | no |

**Verdict: the hypothesis is neither confirmed nor killed, and that is the
finding.** The divergence moved both time-limited instances, but those are
budget-dependent by construction and cannot fail a Regime-N gate; the *only*
budget-independent GP comparable in the entire corpus is `prob03`, a **5-node**
tree, and it was neutral. A 5-node tree cannot discriminate a best-first tie-break
(`PyTreeManager` prefers **deeper** on a bound tie, `pool.rs:57-60`; the loop is
insertion-FIFO), pseudocost-vs-most-fractional branching, empty-child suppression,
or failed-relaxation handling. **The gate, not the port, is what is missing.**

**Consequence.** No loop was ported. Porting behind a gate this weak would be
exactly the "measurement that never happened and reported success anyway" the
Measurement & instrumentation discipline section was written for. Card 4c's status
block records the full per-loop policy characterisation (the reusable half of the
work) and the two exits available.

### 2026-07-30 — Card 4c Task 2: a hand-appended `Constraint` with non-zero `rhs` is solved as though `rhs = 0`

> **PARTIALLY RETRACTED, same day — see the "the `Constraint.rhs` silent wrong
> answer" entry above.** The claim below that "**the public path is correct**" is
> **WRONG**: `Constraint` is exported in `discopt.modeling.__all__` and
> `m.subject_to(Constraint(w, ">=", 5.0))` reproduces the defect exactly. This is
> a public-API silent wrong answer, not a private-attribute footgun. The rest of
> this entry stands. **FIXED** by a loud refusal at `Model.subject_to` and
> `Model.validate` (CLAUDE.md §3); regression suite
> `python/tests/test_constraint_rhs_refusal.py`.

**Not a hypothesis — an incidental measurement**, recorded because it silently
invalidates any test that both solves and verifies the same hand-built model.

`model._constraints.append(Constraint(body, sense, rhs))` is the pattern
`test_incumbent_verifier_scale.py` already uses. The NLP evaluator and therefore
`validation/feasibility.verify_point` honour `rhs`; the **solver does not**.
Measured on this tree:

    Constraint(w, ">=", 5.0) appended by hand  -> solve() returns w = 0
    m.subject_to(w >= 5.0)                     -> solve() returns w = 5

The public `subject_to` path folds the offset into the body and is correct. The
existing verifier tests are unaffected (they never solve). Card 4c Task 2's parity
arm *does* solve, so `vector_constraint_corpus.solvable_vector_cases()` filters
non-zero-`rhs` cases out structurally and the parity arm re-asserts the all-zero
property before solving. **This wants its own issue**: either the private-append
path should be honoured by the solver too, or appending a `Constraint` with a
non-zero `rhs` should refuse loudly rather than silently drop it (§0.4 / CLAUDE.md
§3 — a silent approximation is the failure mode).

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

**Read this ranking as an upper bound per row, not as recoverable wall.** The producer's
decline ladder is ordered, so each code's wall includes instances that would decline on
a *later* code the moment that one is served. Re-measured for rank #1 on 2026-07-29
(Card 5.2-T): only **7 of 17** `term_trilinear` instances would reach row-claiming —
**118.5 s / 6.2 %**, not 437.6 s / 22.7 %. No row below was re-measured; assume the
same caveat applies to all of them.

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

### 2026-07-29 — Phase 5.4 graduation panel for `DISCOPT_CONVEX_KERNEL` — **GRADUATE: NO** (cert-clean and quality-clean PASS; net-positive unproven on a population this environment does not have)

**The panel.** `discopt_benchmarks/scripts/phase5_convex_kernel_diff_panel.py`, all
**119** in-repo instances (union of both corpus dirs), 45 s budget, OFF vs ON with
both arms written explicitly in every child, one subprocess per (instance, arm),
stage-2 replication ×3 with the arms interleaved on the decisive rows. Wall
**10,734 s**, load start 0.82 peak 4.72. Artifact
`reports/phase5_convex_kernel_diff_panel_670911ed.json`.

**Verdict, verbatim after the rescore described below:**

```
## VERDICT
  cert-clean    : PASS (0)
  verification notes (symmetric, NOT charged to the flag): 1
      - nvs22: incumbent fails independent verification in BOTH arms (pre-existing, not attributable to the flag)
  quality-clean : PASS (0)
  net-positive  : FAIL (engaged 3, helped 0, median non-engaged wall delta +0.060s over 116, overhead_ok=True)
  GRADUATE      : NO

  eligible : 4 -> ['clay0303hfsg', 'cvxnonsep_psig40r', 'syn05hfsg', 'syn05m']
  adopted  : 3 -> ['clay0303hfsg', 'syn05hfsg', 'syn05m']
  helped   : 0 -> []
  unresolved: 1
      - clay0303hfsg: replicates disagree — OFF=['feasible', 'time_limit', 'feasible'] ON=['optimal', 'optimal', 'optimal']
  EXECUTED CHECKS : 613 {'objective': 86, 'optimality': 86, 'bound_oracle': 132, 'quality': 103, 'verify': 206}
```

**What the panel settles.** (1) **Cert-clean and quality-clean over the whole
corpus**: 613 executed checks — 86 certified-objective agreements, 86
certification-regression checks, 132 dual-bound-vs-proven-oracle checks, 103
incumbent-quality comparisons, 206 independent incumbent verifications — with
**zero** violations. (2) **The cost of turning the flag on for the 116 instances it
does not route is +0.060 s median.** That is the convexity classification and
nothing else, corroborated exactly by the attempt-cost decomposition above
(`cvxnonsep_psig40r`: spec 1.16 s, tree 0.00 s). Both were open questions and both
are now answered.

**Why it does not graduate — two independent reasons, neither of which is "the
evidence looks bad".**

1. **Population.** `build_convex_spec` accepts **4 of 119** in-repo instances
   (measured: 119 executed probes, 0 errored). The family this flag exists for is
   136 `.nl` files in the MINLPLib snapshot and the deciding counter-case
   `watercontamination0202` is snapshot-only. **Full convex-family panel: SKIPPED —
   local only. `watercontamination0202` misroute counter-case: SKIPPED — local
   only.** `sota-parity-analysis-2026-07-27.md` §4 P3 said this in advance ("the
   in-repo corpus alone cannot graduate this flag"); this run measured it rather
   than assuming it.
2. **`helped = 0` under this session's load.** The one instance that would have
   carried the bar, `clay0303hfsg`, is **quarantined as unresolved**: ON is
   `optimal` in all three replicates, OFF is `feasible`/`time_limit`/`feasible`.
   The replication machinery did exactly what #902 built it for — load moved the row
   to unresolved instead of making the verdict wrong. The cleaner measurement of the
   same instance is the entry experiment above (quiet machine, 2 interleaved
   replicates, 45 s): OFF `feasible` ×2 / ON `optimal` ×2, 46.66 s → 42.53 s. So the
   win is real and the *panel* could not certify it on a busy machine; per §0.1 an
   unresolved row awards nothing, and the flag stays **default-OFF**.

**A harness defect this panel found and fixed in itself — the first verdict was
`cert-clean: FAIL (2)` and it was wrong.** Both violations were `nvs22`, one per
arm, i.e. **identical in OFF and ON**. Root-caused rather than tuned around: the
incumbent fails on two *defined-variable equality* rows,

| row | residual | variable value | relative |
|---|---|---|---|
| `x4 = 4243.28/(x2·x3)` | 1.71e-5 | 2121.64 | 8.1e-9 |
| `x5 = ((59405.9+2121.64·x3)·x6)/(…)` | 2.64e-4 | 10782.7 | 2.4e-8 |

and the reported objective matches `=opt= 6.05822` to **5.7e-8**. The verifier's
tolerance is `abs_tol + rel_tol·|residual|`, which on an equality row degenerates to
a pure absolute 1e-6 **no matter how large the row is** — so a defined variable of
magnitude 1e4 can never pass. Nothing was relaxed: the gate was rescoped to the
question a *differential* panel actually asks — a verification failure is a cert
violation when it is **asymmetric** (ON fails, OFF passes); a failure reproduced in
both arms is by construction not flag-caused and is reported as a
`verification note` with the same prominence. Re-scored from the stored rows
(`--rescore`, no re-solve: the rows are the measurement, the verdict is a function
of them).

**Named follow-up, not fixed here (out of Phase 5's scope, and it is a solver-wide
question):** *the incumbent verifiers' tolerance has no row-scale term.*
`solver/native_kernel._native_kernel_verify_point` and
`solvers/_convex_kernel._incumbent_is_feasible` both reject `nvs22`'s certified
optimum. Either the tolerance should carry a scale term (`abs + rel·‖row terms‖`
rather than `abs + rel·|residual|`), or `nvs22`'s certificate is being issued at a
tolerance the repo's own verifiers do not accept — and which of those is true is a
correctness question that deserves its own investigation with its own panel. The
measurement above is the entry evidence for it.

**Two harness defects also fixed for the next run** (neither changed this verdict):
`_obj_match(None, None)` returned False, so every no-incumbent-in-both-arms row
counted as "decisive" — 20 rows were replicated where 6 were real, which is where
most of the 10,734 s went; and the panel gained a `--rescore` mode so a corrected
gate never requires another three-hour run.

### 2026-07-29 — Phase 5 close-out: what was run on the final tree

Recorded because §0.6 asks for it, and because Phase 3's close-out established the
house rule that a partially-verified tree described as verified is the same defect
as an instrument that measures nothing.

**Regime N (the gate for every bound-neutral change in this phase).**
`panel_baseline.py --check reports/panel_baseline_f154dcff.json` on the Phase-5
tree: **comparisons executed 255** (node_count 85, certified objective 85, status
85) over 85 comparable of 119 rows — **PASS: no node-count or certified-objective
drift.** 18 non-comparable rows reported and not gating, all budget-dependent, the
same character Phase 3 (19) and Card 4a (17) reported. This is the load-bearing
result of the phase's safety argument: the census instrumentation only executes
behind `DISCOPT_NATIVE_SPATIAL_KERNEL` (default OFF), and the 5.4 deduction
subtracts a value that is **exactly** `0.0` with `DISCOPT_CONVEX_KERNEL` off, so
the default path is unchanged *by construction* — and the panel confirms it
*by measurement*.

**Suites.**

| suite | result |
|---|---|
| `pytest -m smoke python/tests` | **871 passed**, 16 skipped, 2 xpassed, 503 s |
| `pytest -m smoke discopt_benchmarks/tests` | **51 passed**, 1 skipped, 32 s |
| `pytest -m slow python/tests/test_adversarial_recent_fixes.py` | **10 passed**, 225 s |
| `pytest python/tests/test_node_tightening_parity.py -m slow` | **12 passed**, 77 s |
| `pytest python/tests/test_phase5_convex_kernel_budget.py` (incl. `-m slow`) | **3 passed** |
| `ruff check` / `ruff format --check` on every changed file | clean |
| `cargo test -p discopt-core` | **not run — no Rust source was touched in this phase** |

The smoke count moved 857 → 871 against Phase 4's record. That is **not** this
phase: `pytest -m smoke python/tests --collect-only` reports **877 collected on
this tree and 877 on the base commit `c346fd73`** (checked in a worktree), i.e.
Phase 5 adds zero smoke tests — the delta belongs to Card 4b's tree, which landed
after the 857 was recorded.

**Card 3c's parity arm is no longer vacuous.** With the served instances added it
reports **served 3 of 5 producer calls** (was 0 of 2), over 140 decided nodes,
4,070 bound comparisons, 394 contraction checks, 127 monotonicity checks, 100
soundness checks and 17 native checks; pooled Python-only-inference rate 5.0 %.

**Disclosed measurement condition (CLAUDE.md §9).** The Regime-N check and the test
suites ran *concurrently* with the differential panel on a 4-core box; load peaked
at 4.72. This is disclosed rather than hidden because the panel's `net-positive`
bar is wall-based: `clay0303hfsg`'s quarantine is plausibly a consequence of it,
and the +0.060 s median non-engaged delta should be read as an upper bound rather
than a precise figure. Neither affects the gates that decided the verdict —
cert-clean and quality-clean are status/objective comparisons, and the Regime-N
gate is node-count and certified-objective equality, which returned the same 85
comparable rows and the same 255 comparisons as every prior run.

### 2026-07-29 — Phase 5.5 entry experiment 1: "`nvs22`'s certificate is issued at a tolerance the repo's own verifiers do not accept" — **FALSIFIED; the verifier is wrong, the certificate is right**

The card was explicitly told this could go either way, and that the opposite
conclusion would mean a shipped certificate is wrong. It was run before any code was
written. Probe: solve `nvs22` (`python/tests/data/minlplib/nvs22.nl`, 45 s, defaults),
take the returned incumbent, and score every row against its own scale. Marker
asserted: `discopt.__file__ = /home/user/discopt/python/discopt/__init__.py`.
**EXECUTED CHECKS: 10.**

```
status optimal  obj 6.058219942618198  bound 6.058219942618198  certified True
VERDICT native_kernel._native_kernel_verify_point: False None
VERDICT _convex_kernel._incumbent_is_feasible    : False

row                                             sense    violation      scale     relative
((neg((((59405.9 + (2121.64*x3))*x6)/((x…  ==   2.6410e-04  1.7329e+04  1.5240e-08
((neg((4243.28/(x2*x3))) + x4) - 0)        ==   1.7080e-05  2.1216e+03  8.0506e-09
((neg(sqrt(((0.25*(x3**2)) + (((0.5*x0)…  ==   2.6426e-08  3.1623e+00  8.3566e-09
((neg(((0.5*x3)/x6)) + x7) - 0)            ==   2.2108e-09  1.0000e+00  2.2108e-09

rows violating pure-absolute 1e-6        : 2
rows violating abs 1e-6 + rel 1e-4·scale : 0
n_constraints (evaluator rows): 9   Constraint instances: 9   max flat size: 1
```

**Kill criterion and verdict.** The hypothesis dies if the offending residuals are
small *relative to their rows*. They are: 1.5e-8 and 8.1e-9, against an objective
that matches MINLPLib `=opt= 6.05822` to 5.7e-8. So the certificate is sound and the
verifier's tolerance is the defect. Recorded loudly per the card's instruction,
because the alternative reading would have been a shipped-certificate bug.

**The arithmetic, since it is the whole finding.** `tol = abs + rel·|residual|` on an
equality row accepts iff `|r| <= abs + rel·|r|` iff `|r| <= abs/(1 - rel)` =
**1.0001e-6** for `abs=1e-6, rel=1e-4`. The `rel_tol` term is dead: the tolerance is a
pure absolute 1e-6 at every row scale. A residual-scaled tolerance is self-referential
and *shrinks toward the absolute floor as the residual shrinks*, which is exactly
backwards from what a relative tolerance is for.

### 2026-07-29 — Phase 5.5 entry experiment 2: "the two verifiers examine the rows they claim to" — **FALSIFIED; both ACCEPT a point violating a row by 5.0**

Run because a tolerance fix that leaves the row enumeration wrong fixes nothing. Four
constructed models, each with a grossly infeasible point; kill criterion stated in the
probe ("if a grossly infeasible point is REJECTED by both on every case, the
hypothesis is dead"). **EXECUTED CHECKS: 8.**

```
[A vector<=, row2 violated by 5.0] rows=3  nk=True   ck=True      <-- BOTH WRONG
[B two scalars, 2nd violated by 5.0] rows=2  nk=False  ck=False
[C SOS then equality, eq violated by 4.0] rows=1 nk=False ck=RAISED AttributeError(
      "'_SOSConstraint' object has no attribute 'sense'")
[D rhs=5 constraint, w=50] rows=1  nk=False  ck=False
FINDING: A: native_kernel WRONGLY ACCEPTED an infeasible point
FINDING: A: convex_kernel WRONGLY ACCEPTED an infeasible point
FINDINGS: 2
```

Case A is the row-alignment defect: `NLPEvaluator.evaluate_constraints` emits one row
per **flat element**, both loops advanced one index per **constraint object**, so a
size-3 vector constraint had rows 1 and 2 never read. Case C is the same class one
level up — `model._constraints` carries classes the evaluator's row set does not, and
the two loops respectively skipped them silently and crashed on them. Case D returns
the right verdict for the wrong reason (`rhs` is ignored; the body is compared against
0), which flips to a wrong verdict as soon as the feasible side is tested — locked by
`test_nonzero_rhs_is_subtracted_in_both_directions`.

After the fix the same probe prints **FINDINGS: 0** with all four rejected.

### 2026-07-29 — Phase 5.5: the regression suite fails on the pre-fix consumers

Per the workflow rule that new behaviour needs a test failing before and passing
after, and per §8 (verify which code you loaded): a worktree at `bb3b6c73` was given
**only** the new `validation/feasibility.py` and the new test file, leaving
`native_kernel.py` / `_convex_kernel.py` at their pre-fix state — so a failure is
attributable to the consumers and not to a missing import.

```
BEFORE (bb3b6c73 + new module + new test):  6 failed, 9 passed  (+ the slow nvs22 test: 1 failed)
AFTER  (this tree):                        15 passed            (+ the slow nvs22 test: 1 passed)
```

The 7 failures are exactly the defect set: vector-row alignment, builder-resident
rows, `rhs`, unevaluable classes, missing bounds/integrality on the convex path, the
synthetic large-scale row, and real `nvs22`. The **9 that pass in both arms are the
§0.4 locks** — unit-scale tolerance unchanged, and the four naive widenings
(model-global scale, 1-norm scale, `rel_tol` coefficient, raised absolute floor) each
still rejecting the bad point they were built to catch. A fix that had merely loosened
the tolerance would have failed those nine.

### 2026-07-29 — Phase 5.5 corpus sweep: "the row-scale term buys permissiveness" — **FALSIFIED, and the widening is bounded at 1.5e-8 relative**

`discopt_benchmarks/scripts/verifier_scale_sweep.py`, all **119** in-repo instances,
20 s budget, one subprocess per instance. Every returned incumbent is scored under
**both** tolerance forms *on one tree* — the old `abs + rel·|residual|` loop is
re-implemented inline in the probe, so the comparison is of forms rather than of git
revisions. Artifact `reports/verifier_scale_sweep_030b44f4.json`. Wall 1,329.9 s,
load 0.68 → 1.65.

```
## VERDICT
  EXECUTED COMPARISONS : 99
  agree                : 98
  False -> True (old wrongly rejected) : 1
      nvs22   worst_abs=2.6410e-04  worst_rel=1.5240e-08  (row scale 1.7329e+04)
              status=optimal  certified=True
  True -> False (new rejects)          : 0
  worst RELATIVE violation newly accepted: 1.5240e-08
```

**What this bounds.** Corpus-wide the two forms disagree on **exactly one** incumbent,
the one that started this card, and the largest relative violation the new form
accepts anywhere is **1.52e-8** — 65× inside the 1e-6 relative tolerance. `True →
False` is **0**: nothing that verified before stops verifying. So the widening is not
a blank cheque; it is one instance, at a relative residual three orders of magnitude
below the tolerance, on a certificate that matches the published optimum to 5.7e-8.

**Two things this sweep honestly does *not* show, said rather than implied.**
(1) The 20 s budget leaves 20 of 119 instances without an incumbent, so the scored
population is 99, not 119. (2) The **row-alignment** defects (§6 entry 2) do not fire
on this corpus at all: every `.nl` instance parses to scalar rows (`max flat size 1`)
and none uses the builder fast path, so `from_nl` cannot reach them. The single
"rows the OLD loop never examined" row (`nvs22`, 1 of 9) is the old loop's *early
exit* on its first failing row, not the alignment bug. Defects 2 and 3 are reachable
through the Python modeling API and `add_linear_constraints` — which is precisely
what `test_vector_constraint_rows_beyond_the_first_are_checked` and
`test_builder_resident_linear_rows_are_checked` cover, and why they, not this sweep,
are the evidence for those two.

### 2026-07-29 — Phase 5.5: `nvs22` re-measured on Phase 5.4's own panel child

Not a re-derivation from the unit tests: the actual differential-panel child, both
arms, ×2, 45 s, on the fixed tree.

```
OFF rep1  status=optimal obj=6.058219942618198 node_count=35 verified=true
OFF rep2  status=optimal obj=6.058219942618198 node_count=35 verified=true
ON  rep1  status=optimal obj=6.058219942618198 node_count=35 verified=true  attempt_s=0.112
ON  rep2  status=optimal obj=6.058219942618198 node_count=35 verified=true  attempt_s=0.106
```

`verified` was **false in both arms** before this card and is **true in both** after.
The panel's single symmetric `verification note` therefore disappears, and it
disappears for the right reason — the incumbent's worst relative row violation is
1.5e-8, not because a tolerance was widened to swallow it. Phase 5.4's differential
scoping is left exactly as Phase 5 wrote it (it is the right question for a
differential panel); only its comment is updated to record the resolution.

### 2026-07-29 — Card 3e entry experiment: "seeding the root FBBT from `ctx.bounds` is a sound, helpful tightening" — **HALF CONFIRMED, HALF KILLED; flag stays OFF**

`discopt_benchmarks/scripts/card3e_fbbt_seed_entry.py`, all **119** in-repo
instances, root presolve only, one subprocess per instance, 11.25 s presolve budget
(what production gives the pass on a 45 s solve). Marker asserted before any
measurement: `run_root_presolve` must expose `fbbt_seed_from_ctx`, else the child
exits 2 rather than measuring the wrong tree. Artifact
`reports/card3e_fbbt_seed_entry_be705694.json`. Wall 223.7 s, load 0.95 → 1.58.

```
## VERDICT
  EXECUTED BOUND COMPARISONS : 11058
  comparable instances       : 119 of 119
  instances tightened by seed: 9
      util +76   heatexch_gen3 +28   hda +12   casctanks +5
      4stufen +2   beuster +2   st_e03 +2   st_e11 +1   st_e17 +1
  TOTAL bounds tightened     : 129
  budget-dependent rows (arms stopped differently): 8
  SOUNDNESS: instances loosened : 1
      casctanks  [{var 339: off_lo 0.021650635 -> on_lo 0.017677670}, …]
FAIL: the seeded arm LOOSENED a bound — the seed is not a valid box
```

**Confirmed half.** The kill criterion was "zero bounds tightened corpus-wide". It is
not zero: **129 bounds on 9 of 119 instances**. Card 3b's mechanism claim survives
contact with the real corpus — this is not a synthetic-only effect (the #727 RLT
failure mode), and the composition gap it names is real.

**Killed half, and it is the one that decides the card.** The probe's containment
assertion — seeding can only shrink the starting box and `backward_propagate` only
tightens, so the seeded result must be *contained* in the unseeded one — fails on
`casctanks`. Run down rather than attributed to the budget:

```
budget= 11250ms  OFF[IterationCap, iters=16]  ON[Infeasible, iters=2]  tightened=5  loosened=5
budget= 60000ms  OFF[IterationCap, iters=16]  ON[Infeasible, iters=2]  tightened=5  loosened=5
budget=300000ms  OFF[IterationCap, iters=16]  ON[Infeasible, iters=2]  tightened=5  loosened=5
```

Identical at 27× the budget: **the seeded pass declares the `casctanks` root box
empty**, and the orchestrator stops at iteration 2 with `Infeasible` instead of
running its 16. The "loosening" is that early stop. `casctanks` has no reference
optimum in the in-repo oracle (`reference_oracle("casctanks") -> None`) and neither
arm finds an incumbent at 45 s, so there is no feasible witness on hand to call this
a *proven* false infeasible — but there is equally nothing supporting the box being
empty, and a presolve that can empty a root box is one consumer away from a false
`infeasible` certificate. Checked, and recorded because the negative matters:
end-to-end `DISCOPT_FBBT_SEED=1` on `casctanks` returns the **same**
`time_limit` / bound `-90.17862569095436` as OFF, so **no false infeasible ships
today**.

**The only end-to-end datum points the wrong way too.** Three instances, both arms,
45 s:

| instance | OFF | ON |
|---|---|---|
| `util` | `feasible` obj 1059.3160203762593 bound 999.5538990472674 **159 nodes** | same obj, same bound, **217 nodes** |
| `st_e03` | `optimal` −1161.3366118602887, 53 nodes | identical |
| `hda` | `time_limit`, bound −64473.44240243703, 3 nodes | identical |

`util` is the instance with the largest presolve gain (+76 bounds) and it costs 58
nodes for an identical bound — the tightening changes the branching order without
improving the relaxation. Three instances is not a net-positive verdict and is not
offered as one; it is the reason the three-hour panel was **not** spent.

**Verdict.** `DISCOPT_FBBT_SEED` ships **default-OFF** and does not graduate. Per
§0.3 the entry experiment exists to decide whether to keep building, and it said
stop: an unexplained infeasibility declaration outranks an unmeasured net-positive
question. The mechanism, its Rust test, and this measurement are landed so the next
session starts from evidence rather than from the hypothesis. What must happen before
any panel: run the seeded kernel from the *unseeded* orchestrator's own final box —
the box that run already certified — and determine whether `casctanks` is an
orchestrator bookkeeping bug (`ctx.bounds` not a valid seed for `ctx.model`) or an
FBBT `FEAS_TOL` question on one instance.

### 2026-07-29 — Phase 5.5 / Card 3e close-out: what was run on the final tree

Recorded because §0.6 asks for it, and because Phase 3's close-out established the
house rule that a partially-verified tree described as verified is the same defect as
an instrument that measures nothing.

**Regime N.** `panel_baseline.py --check reports/panel_baseline_f154dcff.json` on the
final tree — with the Rust `.so` **rebuilt** by `maturin build --release`, which is
the reason this gate is not optional here: `DISCOPT_FBBT_SEED` is default-OFF and
`seed=None` is asserted bit-for-bit identical in the Rust test, but a rebuilt binary
is still a rebuilt binary.

```
comparisons executed: 255 (node_count 85, certified objective 85, status 85)
                      over 85 comparable of 119 baseline rows
PASS: no node-count or certified-objective drift.
```

Identical population and comparison count to Phase 5's run (255 / 85). **14**
non-comparable rows reported and not gating, all budget-dependent — the same
character as Phase 5 (18), Phase 3 (19) and Card 4a (17). Wall 2,008.5 s, load start
0.82 peak 2.89.

**Suites.**

| suite | result |
|---|---|
| `pytest -m smoke python/tests` | **882 passed**, 16 skipped, 2 xpassed, 487 s |
| `pytest -m smoke discopt_benchmarks/tests` | **51 passed**, 1 skipped, 33 s |
| `pytest -m slow python/tests/test_adversarial_recent_fixes.py` | **10 passed**, 225 s |
| `pytest python/tests/test_node_tightening_parity.py -m slow` | **12 passed**, 76 s |
| `pytest python/tests/test_incumbent_verifier_scale.py` (incl. `-m slow`) | **16 passed** |
| `pytest python/tests/test_flag_registry.py` | **17 passed** |
| `cargo test -p discopt-core` | **542 passed**, 0 failed (Rust was touched) |
| `ruff check` / `ruff format --check` on every changed file | clean |
| `cargo fmt` on the changed crates | clean (an unrelated `delta.rs` reformat was reverted to keep the diff scoped) |

The smoke count moved 871 → **882**, and that delta is attributable rather than
mysterious: `test_incumbent_verifier_scale.py` contributes exactly **11** `smoke`
tests (its other 5 are 4 `unit` and 1 `slow`). 871 + 11 = 882.

**heldout50: SKIPPED — local only** (the MINLPLib snapshot is not in this
environment). Card 3e's §5 differential panel: **not run**, deliberately — see the
Card 3e §6 entry; the entry experiment ended the card before the panel was the right
spend.

**Disclosed measurement condition (CLAUDE.md §9).** The Regime-N check ran
*concurrently* with `discopt_benchmarks` smoke, the adversarial suite and the parity
suite (load start 0.82, peak 2.89); the `python/tests` smoke suite ran alone
afterwards. No verdict on this tree is wall-based — Regime N gates on node-count and
certified-objective equality, the sweeps gate on verdict flips and bound counts — so
contention can only move a row out of the comparable population, and it did not: the
same 85 comparable rows and 255 comparisons as every prior run.

### 2026-07-29 — Card 3e-RC: "`casctanks`'s seeded `Infeasible` is either an invalid seed (a) or a genuinely empty box (b)" — **BOTH FALSIFIED; it is the emptiness test, and it fires with no flag set**

`discopt_benchmarks/scripts/card3e_infeasible_root_cause.py`, all **119** in-repo
instances, both arms, one subprocess per instance, 11.25 s presolve budget (identical
to the Card 3e entry probe, so the OFF arms are directly comparable). Artifact
`reports/card3e_infeasible_root_cause_ed2da7bd.json`. **22,116 executed bound
comparisons**, 119 of 119 comparable. **Artifact caveat:** the pre- and post-fix runs
share a git SHA (the fix was uncommitted while measuring), so the `.json` holds the
**post-fix** run only; the pre-fix transcript is preserved verbatim as
`reports/card3e_rootcause_sweep.log` and the post-fix one as
`reports/card3e_rootcause_sweep_postfix.log`, both committed alongside it. Marker asserted before any measurement: the
**compiled** `PyModelRepr.presolve` must accept `fbbt_seed_from_ctx` — not the Python
signature, which is what the original Card 3e probe checked. That distinction was not
academic: this session's container came up with a `.so` from 07:44 against sources
restored at 23:13, and a signature-only marker passes on that stale binary.

**Pre-fix verdict.**

```
## VERDICT
  EXECUTED BOUND COMPARISONS : 22116
  comparable instances       : 119 of 119
  wall 204.2s   load 0.35 -> 1.24
  OFF terminated Infeasible  : 1  ['heatexch_gen3']
  ON  terminated Infeasible  : 2  ['casctanks', 'util']
  --- crossing census, split at FEAS_TOL ---
  OFF instances w/ NOISE crossings (0 < lo-hi <= 1e-6) : 1
      heatexch_gen3            n=3 worst=8.526513e-14 var4 [226.7, 226.6999999999999]
  OFF instances w/ GENUINE crossings (lo-hi > 1e-6)    : 0
  ON  instances w/ NOISE crossings                     : 2
      casctanks                n=1 worst=1.110223e-16 var179 [0.75, 0.7499999999999999] last_pass=implied_bounds@it1
      util                     n=2 worst=6.821210e-13 var103 [1726.551859626083, 1726.5518596260822] last_pass=implied_bounds@it6
  ON  instances w/ GENUINE crossings                   : 0
  --- E1: FBBT kernel from the UNSEEDED arm's own certified final box ---
  E1 declared the certified box EMPTY on : 1 instance(s)
      heatexch_gen3            worst_cross=8.526513e-14 noise=3 genuine=0
  E1 declared the DECLARED box empty on  : 0 instance(s) (control)
```

**Hypothesis (a) — `ctx.bounds` is an invalid seed for `ctx.model` — FALSIFIED.** E1
is the distinguishing experiment the Card 3e block specified: the production FBBT
kernel (`in_tree_presolve`, which takes an explicit `(node_lb, node_ub)` box and
patches the model with it, so it *is* "the kernel from a supplied box") run from the
**unseeded** orchestrator's own final box on its own final model. It clears on 118 of
119. There is no indexing or renumbering fault: `casctanks` has 490 scalar blocks, one
per column, and the E1 control from the declared box also clears on all 119.

**Hypothesis (b) — the composed box is genuinely empty to within `FEAS_TOL` —
FALSIFIED.** **0 instances** have a crossing above `FEAS_TOL` (1e-6), in either arm.
The three that abort cross by **1.1e-16, 6.8e-13 and 8.5e-14** — the last ulp.

**What it actually is.** The *emptiness test*. `orchestrator::any_empty` used
`bounds.iter().any(|b| b.is_empty())` (strict `lo > hi`, zero tolerance) and
`bnb::in_tree_presolve` used the same, while `fbbt.rs` (3 sites), `fbbt_fp.rs` (2, one
carrying the comment "gate this one on FEAS_TOL to avoid a false 'infeasible'") and
`probing.rs` all gate on `is_empty_beyond(FEAS_TOL)`. The two zero-tolerance outliers
were precisely the two whose verdict is *acted on*: the orchestrator aborts the sweep,
and `in_tree_presolve` sets `infeasible`, which the B&B loop treats as a **rigorous
fathom** and prunes the subtree (`solver/__init__.py:8457`, `:12254`;
`tightening.py:218`). Mechanism on `casctanks`, traced pass by pass: at iteration 0 the
seeded FBBT derived `ub = 0.7499999999999999` for var 179 where the unseeded arm
derived `0.75`; at iteration 1 `implied_bounds` — a different kernel, different
arithmetic, same quantity — derived `lb = 0.75`; `lo > hi` by one ulp; abort.

**This is a live defect on the default path, and that is the finding that outranks the
flag.** `heatexch_gen3` aborts root presolve with `DISCOPT_FBBT_SEED` **unset**, and
the in-tree kernel calls that instance's own certified root box empty by 8.5e-14. The
root-presolve consequence is contained today (the solver ignores `terminated_by`, and
`propagate_bounds_to_model` reads the repr's declared bounds rather than `ctx.bounds`),
but the in-tree consequence is not contained: it is a false fathom, i.e. a false bound.

**`util` was below Card 3e's own reporting threshold.** It terminates `Infeasible` at
iteration 7, and the entry probe's containment-only check still scored it a net
tightening (+76 bounds), so the original measurement reported 1 affected instance where
there are 3. Recorded per §6/§11: the earlier statement "the containment violation
fails on `casctanks`" was true but incomplete, and "1 instance loosened" is retracted
in favour of **3 instances abort, 1 of them with no flag set**.

**Fix and its guard.** `repair_subtol_crossings(bounds, FEAS_TOL)` +
`any_empty_beyond(bounds, FEAS_TOL)` in `fbbt.rs`, applied at both sites; the repair
snaps a sub-tolerance crossing to `[min(lo,hi), max(lo,hi)]`, the smallest interval
containing both endpoints, so no feasible point either derivation admitted is cut and
no inverted interval reaches an LP column bound. In `in_tree_presolve` the *incoming*
node box is sanitized before anything reads it — the first version repaired only the
FBBT output and still returned an inverted box, because `new_lb`/`new_ub` start as
copies of the input and the flooring loop only tightens; the new test caught that.
Four Rust tests, two of which are the anti-permissiveness guard (§0.4): a crossing
beyond `FEAS_TOL` must still abort / still fathom. Verified fail-before-pass-after by
reverting the behaviour with the tests in place —
`subtol_crossing_is_repaired_not_declared_infeasible` fails with "a
1.1102230246251565e-16 crossing is rounding noise, not an infeasibility" on the old
code and passes on the new; `crossing_beyond_feas_tol_is_still_infeasible` passes both
ways, which is what makes it a control rather than a ratchet.

**Post-fix, same probe, same population.**

```
  EXECUTED BOUND COMPARISONS : 22116        comparable: 119 of 119
  wall 224.5s   load 0.61 -> 1.12
  OFF terminated Infeasible  : 0  []        (was 1: heatexch_gen3)
  ON  terminated Infeasible  : 0  []        (was 2: casctanks, util)
  OFF/ON instances w/ NOISE crossings   : 0 / 0   (was 1 / 2)
  OFF/ON instances w/ GENUINE crossings : 0 / 0   (was 0 / 0 — guard unchanged)
  E1 declared the certified box EMPTY on : 0     (was 1)
```

A note on the instrument, because it is the reason this was found: the probe reports
the **magnitude** of every crossing split at `FEAS_TOL`, not a boolean. A probe that
only reported "empty / not empty" would have reproduced Card 3e's dead end exactly.

### 2026-07-29 — Phase 5.2: "`IncrementalMcCormickLP._select` already solves `blf_row_count`; porting it is a Python change worth 7 instances" — **FALSIFIED; no matcher exists to port**

`discopt_benchmarks/scripts/phase52_blf_select_entry.py`, artifact
`reports/phase52_blf_select_entry_ed2da7bd.json`. **65 executed assertions**, **55 BLF
terms** examined across the 7 `blf_row_count` instances (`st_e01`, `st_e05`, `st_e08`,
`st_e09`, `st_e11`, `st_e40`, `syn05hfsg`) plus `tanksize` as a served control.

```
  `_select` call sites            : 3
  `_select` classes               : ['affine_square', 'bilinear', 'monomial']
  BLF class covered by `_select`  : False
  spec records form bounds ba/bb  : False
  BLF terms examined              : 55
  BLF terms claiming != 4 rows    : 11
  CLAIM FALSIFIED
```

The claim is wrong on the **identity of the class**. `_select` has three call sites —
`bilin_rows`, `mono_rows`, `affsq_rows`. `blf_row_count` declines on
`rel.bilinear_linform_specs`, the product of two *affine forms*, and
`incremental_mccormick.py` never references that field: it is consumed by
`spatial_producer.py` alone. `_select` solves the analogous problem for three *other*
families; there is nothing to port. Per §0.3 the card stops here rather than starting
an envelope build on a false premise.

**Retracted predicate in this probe (CLAUDE.md §11).** Its first version OR-ed a
case-insensitive `"linform"` substring over the whole module into the verdict and
printed **"CLAIM HOLDS"**. The module's single hit is the word `LinForm` inside a
docstring about `_emit_1d` on an affine base — unrelated. The verdict was wrong, was
retracted before being carried anywhere, and the predicate now rests solely on
`"bilinear_linform_specs" in src`; the loose count is kept as a recorded field that no
verdict reads. Recorded because a probe that confidently prints the opposite of what
its own data says is the §6 failure mode, and this one did it to me.

**What the measurement does establish, which is why the direction survives.** The
control is clean — `tanksize`, the instance #764 validated the BLF path on, has
**19/19** terms claiming exactly 4 rows, so the producer's `!= 4` predicate is right
about what it serves. And the 11 declining terms' extra candidates are exactly what a
matcher needs them to be, box-**independent**: aux bound rows (`{w: +1} <= 0` and
`{w: -1} <= 0` on `st_e40`; `{42: 1.0} <= 0` on `syn05hfsg`) and lifted model
constraints (`600·x₀ − 50·x₁ − w <= −5000` plus its negation on `st_e11`; `2x₀ + 2x₁ +
4w <= 3` on `st_e09`). So the card is a **build**, not a port, with one prerequisite
the old framing hid: the expected closed-form rows need the form enclosures
`(aL, aH, bL, bH)`, `bilinear_linform_specs` does not record them, and they cannot be
recomputed from the spec because `_emit_mccormick` takes them from `ctx.bounds(node)`
(an `evaluate_interval` on the original DAG, not a LinForm interval over column
bounds). Full card shape in the Phase 5 "What 5.2 should take next" block.

### 2026-07-29 — Card 5.2-T scoping: "`term_trilinear` needs a trilinear `EnvTerm` in Rust, and it is where the wall is (17 instances, 22.7 %)" — **FALSIFIED on both halves**

Scoping only (implementation explicitly out of scope this session). Two measurements.

**(1) Not a new Rust envelope family.** `uniform_relax._fold_product` relaxes
`x_i·x_j·x_k` as a **nested McCormick bilinear chain** — `t = x_i·x_j`, then `w = t·x_k`
— two `_emit_mccormick` calls, hence two `bilinear_linform_specs` entries, a family the
kernel's `BlfTerm` already implements. `trilinear_map` is registered *additionally*, so
the separators (trilinear RLT, Meyer-Floudas/Rikun, `multilinear_separation.py`) can
attach tighter cuts. The producer's `if rel.trilinear_map: return
_decline("term_trilinear")` therefore rejects on the mere **registration**, not on
inexpressible rows. Residual work, measured: 4–13 rows per instance touch the trilinear
aux and are claimed by no BLF/monomial/affine-square term (`st_e03` 6, `ex1224` 4,
`nvs01` 6, `mathopt3` 6, `nvs22` 8, `bchoco06` 13, `nvs09` 0) — lifted model
constraints (box-independent, pass through) mixed with **trilinear RLT rows**, which
are box-dependent and default ON (`DISCOPT_TRILINEAR_RLT`). Those need the treatment
`skip_separable_floor` / `skip_convex_lift` already get in the producer's `_build`.
`multilinear_separation.py` is **not** reachable from the kernel producer — it is a
per-node Python separator requiring LP solves, and the kernel runs the node entirely in
Rust; `DISCOPT_TRILINEAR` likewise selects among `relaxation_compiler.py` strategies,
while the producer builds from `uniform_relax`.

**(2) The prize is 6.2 %, not 22.7 %.** The producer's decline ladder is ordered and
`term_trilinear` is tested **before** `infinite_aux_bounds` / `term_multilinear` /
`term_ratio` / `term_univariate`, so it *masks* the other reasons. Re-running the ladder
with the trilinear test removed, over all **17** instances (17 classified, executed
count printed):

| next decline reason | n | instances |
|---|---|---|
| **NONE — would reach row-claiming** | **7** | `ex1252`, `ex1252a`, `nvs01`, `nvs06`, `nvs21`, `st_e03`, `st_e38` |
| `infinite_aux_bounds` (census rank #2) | 6 | `bchoco06/07/08`, `mathopt3`, `nvs05`, `nvs22` |
| `term_univariate:log` / `:exp` (rank #8) | 3 | `ex1224`, `st_e29` / `st_e36` |
| `term_multilinear` | 1 | `nvs09` |

Baseline wall of the reachable 7 is **118.5 s of 1,926.0 s = 6.2 %**, of which
`ex1252` + `ex1252a` are **92.0 s** (both at the 45 s ceiling, i.e. unsolved). The
census's 437.6 s / 22.7 % **reproduced exactly** from
`reports/panel_baseline_f154dcff.json`, so it is a correct attribution of the *first*
decline code — but it is an upper bound, not recoverable wall. Every ranked row in the
Phase 5.1 census carries this same caveat and none of them said so; that is the
transferable lesson.

**Verdict:** "wire up what exists" — a Python producer change (drop the blanket
`trilinear_map` decline, add a `skip_trilinear_rlt`-style build option, let the nested
`BlfTerm`s be claimed) — **one to two sessions** including the Regime C panel and the
Card 3c parity extension, for 7 instances / 6.2 %. That makes it comparable to rather
than dominant over rank #2 `infinite_aux_bounds` (9 instances, 19.7 %, and the blocker
for 6 of these 17 as well), so `infinite_aux_bounds` is arguably the better next card.

### 2026-07-29 — Card 3e re-measured on the fixed tree: the containment violation is gone, the net-positive question is not

`card3e_fbbt_seed_entry.py` re-run unchanged on the Card 3e-RC tree, all **119**
instances, **11,058 executed bound comparisons**, 119 of 119 comparable, wall 227.5 s.
Artifact `reports/card3e_fbbt_seed_entry_ed2da7bd.json` (supersedes the `be705694`
artifact for the containment question only; console transcript
`reports/card3e_entry_postfix.log`).

```
  instances tightened by seed: 8
      util +76   casctanks +8   hda +4   4stufen +2
      beuster +2   st_e03 +2   st_e11 +1   st_e17 +1
  TOTAL bounds tightened     : 96
  budget-dependent rows (arms stopped differently): 5
  SOUNDNESS: instances loosened : 0
```

**`SOUNDNESS: instances loosened : 0`** — was 1 (`casctanks`, 5 bounds). The
containment invariant now holds corpus-wide, which is exactly what the Card 3e block
said had to happen before the §5 panel could be spent. The probe no longer exits
non-zero.

**Two of the old numbers are retracted as artifacts of the defect (§11).** The totals
moved 129 → **96** bounds on 9 → **8** instances, and both deltas are attributable
rather than mysterious:

- `heatexch_gen3 +28` **disappears entirely**. Pre-fix its *unseeded* arm aborted
  `Infeasible` at an 8.5e-14 crossing, so the OFF box it was compared against was
  artificially loose and the seeded arm "beat" it by 28 bounds. With OFF running its
  full sweep the gain is zero. That +28 was never a tightening; it was the defect
  measured from the other side.
- `casctanks` moves +5 → **+8** and `hda` +12 → +4: both arms now run more sweeps, so
  the fixpoints compared are different (and closer) fixpoints.

**Graduation: still NO, and now for one reason instead of two.** The soundness blocker
is gone; the net-positive question is untouched and its only end-to-end evidence still
points the wrong way — `util` 159 → **217 nodes** at an identical bound and objective,
`st_e03`/`hda` byte-identical. The §5 differential panel is therefore **permissible**
for the first time, and was **not run this session**: it is a ~3 h spend (cf. the
Phase 5.4 panel's 10,734 s) that the entry data predicts will return
`net-positive: FAIL`, and it would have displaced the Regime N verification of the
Card 3e-RC fix, which is the change that actually ships. Stated plainly so the next
session does not re-derive it: **run the panel if you want the negative on record;
nothing in the evidence suggests it will graduate.**

### 2026-07-30 — Card 3e-RC close-out: what was run on the final tree

Recorded per §0.6, and following the house rule Phase 3's close-out established: a
partially-verified tree described as verified is the same defect as an instrument that
measures nothing.

**Regime N.** `panel_baseline.py --check reports/panel_baseline_f154dcff.json`, with
the Rust `.so` rebuilt by `maturin build --release`.

```
instances       : 119
statuses        : {'child_timeout': 1, 'feasible': 18, 'optimal': 88, 'time_limit': 12}
comparable rows : 85/119 (certified terminal within 60% of budget — the Regime-N population)
wall            : 1956.6s total; load start 0.78 peak 3.29

comparisons executed: 255 (node_count 85, certified objective 85, status 85)
                      over 85 comparable of 119 baseline row(s)
PASS: no node-count or certified-objective drift.
```

**Identical population and comparison count to every prior run** (255 / 85 — Phase 5,
the Card 3e close-out, and Card 4a all report the same). 20 non-comparable rows
reported and not gating, all budget-dependent — the same character as Phase 5 (18),
Phase 3 (19) and Card 4a (17).

**The honest limit of that PASS, stated because it matters here more than usual.** All
three instances this fix actually changes are **outside** the Regime-N comparable
population, and were before the change: `heatexch_gen3` is `child_timeout`,
`casctanks` and `util` are budget-dependent (`time_limit` / `feasible`). So the panel
confirms the fix disturbed nothing on the 85 rows it can adjudicate — which is the
question Regime N exists to answer — but it is **not** the evidence that the fix works.
That evidence is the corpus probe (22,116 executed comparisons, 3 → 0 spurious aborts)
and the six Rust tests, two of them anti-permissiveness controls.

**Suites** (final tree, final `.so`).

| suite | result |
|---|---|
| `pytest -m smoke python/tests` | **882 passed**, 16 skipped, 2 xpassed, 379.8 s |
| `pytest -m smoke discopt_benchmarks/tests` | **51 passed**, 1 skipped, 22.7 s |
| `pytest -m slow python/tests/test_adversarial_recent_fixes.py` | **10 passed**, 187.4 s |
| `pytest python/tests/test_node_tightening_parity.py -m slow` | **12 passed**, 1 deselected, 68.5 s |
| `pytest python/tests/test_incumbent_verifier_scale.py` | **15 passed**, 1 deselected |
| `pytest python/tests/test_flag_registry.py` | **17 passed** |
| `cargo test -p discopt-core` | **548 passed**, 0 failed (Rust was touched); +6 vs 542 |
| `ruff check` / `ruff format --check` on every changed file | clean |
| `cargo fmt` on the changed crates | clean (the pre-existing `delta.rs` deviation left untouched, as in the Card 3e close-out) |

Smoke stayed at **882** — this change adds no Python tests, and the count matching the
previous close-out exactly is the intended signal.

The parity suite is the guard that Card 2b's measured Jacobian/nonlinear asymmetry has
not regressed (it asserts a ceiling on the pooled Python-only-inference node rate plus
non-vacuity counters, and that the native arm served > 0). It passes unchanged at 12,
and nothing new is served this session, so no parity extension was required.

**No new flag.** This is a soundness fix removing a false fathom; gating it default-OFF
would ship the defect. Precedent: Phase 5.5's incumbent-verifier fix landed the same
way. `_flag_registry.py` and `docs/reference/flags.md` are therefore untouched, and
`test_flag_registry.py` passes unchanged.

**heldout50 / MINLPLib snapshot: SKIPPED — local only.** Card 3e's §5 differential
panel: **not run** — see the Card 3e re-measurement entry; its soundness blocker is now
cleared, but the spend was judged against evidence predicting `net-positive: FAIL`.

**Disclosed measurement condition (CLAUDE.md §9).** The Regime-N check ran concurrently
with the benchmarks-smoke / flag-registry / parity / adversarial suites (load start
0.78, peak 3.29); `python/tests` smoke and the final corpus probe ran afterwards
(load 1.02 → 3.98). No verdict reported here is wall-based — Regime N gates on
node-count and certified-objective equality, the corpus probe on abort counts and
crossing magnitudes — so contention can only move a row out of the comparable
population, and it did not: the same 85 comparable rows and 255 comparisons as every
prior run.

**One sequencing caveat, disclosed rather than smoothed over.** The probing-branch exit
guard and the `repair_subtol_crossings_contract` unit test landed *after* the Regime-N
panel started, so the panel measured the build without them. The unit test is
test-only; the exit guard is provably a no-op under the panel's default settings
(`DISCOPT_NODE_PROBING` is OFF, and with probing off the returned box is
`max(node_lo, iv.lo)` / `min(node_hi, iv.hi)` where `iv ⊆ node box`, so it cannot
invert). The corpus probe **was** re-run on the final build after the rebuild, with a
marker assertion, and returned byte-identical counts (22,116 comparisons, 0 aborts in
both arms).

### 2026-07-30 — Card 6a: "the 2449 % primal gap is a postsolve or substitution-validity bug" — **FALSIFIED on every candidate; the incident is not a wrong answer, and the one real defect is elsewhere**

**Hypothesis (the card's own, and the session brief's).** `DISCOPT_PRESOLVE_SUBSTITUTE=1`
returned objective **3190.4506** against `=opt= 125.1956151` on
`watercontamination0202` — a 2449 % primal gap (`sota-parity-analysis-2026-07-27.md`
§G-G.1). A presolve that returns a grossly wrong answer is a correctness defect, so
the mechanism is one of: postsolve not inverting every substitution; substituted
variables not restored into the reported point; bounds on eliminated variables
dropped; the objective evaluated on the reduced model rather than the original.

**Kill criterion.** If the substitute → postsolve chain satisfies the exactness
identities on every instance the pass reduces, all four candidates are dead and the
incident must be explained by something other than the transform.

**Why it was worked by construction.** `watercontamination0202` is snapshot-only and
absent here, so the entry experiment tests the *identities the transform claims*
rather than replaying the instance. Marker asserted before every measurement:
`subtol_crossings_repaired` present in `_rust.cpython-311-x86_64-linux-gnu.so`
(CLAUDE.md §8). Rust untouched all session, so the binary matches `8532ce2d`.

**Experiment 1 — repr-level and bridge-level identities.** 66 in-repo `.nl`
instances; 13 carry a reduction. Per instance, 8 random points inside the reduced
box, lifted through `chain.postsolve`:

| arm | what it asserts | executed |
|---|---|---|
| E1 | `reduced.obj(x_red) == pristine.obj(postsolve(x_red))` | **104** |
| E2 | max constraint violation identical in both spaces | **104** |
| E3 | a point inside the reduced box lifts inside the pristine box | **104** |
| E4 | `model_to_repr(model_from_repr(reduced))` evaluates identically — the model actually SOLVED is the Python one, not the repr | **104** |
| E5 | same scalar-var and row count across that bridge | **13** |

**0 failures, 0 errors.**

**Experiment 2 — per-row and per-variable, because a max-violation comparison can be
masked by one dominating row** (CLAUDE.md §6: an instrument that cannot fail is not
an instrument). Row names are `None` on `.nl` reprs, so the rows are compared
alignment-free: the pristine residual multiset at the lifted point must equal the
reduced residual multiset padded with exactly one zero per dropped row. Six points
per instance, the last two deliberately placed *outside* the reduced box:

| arm | what it asserts | executed |
|---|---|---|
| R1 | a surviving row has the same value in both spaces | **13,086** |
| R2 | a dropped row is satisfied at the lifted point for an ARBITRARY reduced point (it is definitional, so it must hold identically, not only at feasible points) | **3,642** |
| B1 | inside the reduced box ⟹ inside the pristine box (soundness of the transferred bound) | **52** |
| B2 | outside the reduced box ⟹ outside the pristine box (EXACTNESS — an over-tightened transfer would cut the optimum) | **26** |

**0 failures.** B1 and B2 together say the boxes correspond in both directions, so
the reduced model is neither a relaxation nor a restriction of the original; R1 and
R2 say the same for the rows. **All four candidate mechanisms are FALSIFIED.**

**Consequence 1 — the incident is not a wrong answer, and §G-G.1's framing is
retracted (CLAUDE.md §11).** `lift_result` recomputes the objective on the *pristine*
repr and feasibility-verifies the lifted point there at 1e-5 before returning it, and
the incident run's status was `feasible` under a 60 s limit. A 2449 % gap on a
verified-feasible incumbent is a primal-quality outcome. The flag did not return a
grossly wrong answer on `watercontamination0202`; it returned a *sound, poor* first
incumbent where the default path returned none.

**Consequence 2 — that signature reproduces in-repo.** 12-instance ON/OFF panel over
the reducing set, one subprocess per (instance, arm), arms interleaved, 30 s budget
(`hda` additionally at 45 s with two replicates per arm, order-balanced OFF/ON/ON/OFF).
Every arm was **bit-identical across replicates**, so the spread on the reported
quantity is zero and the verdict is not wall-based.

| instance | sense | bound OFF | bound ON | effect of the flag |
|---|---|---|---|---|
| `hda` (45 s) | min | −64,473.442402437 | −8,530,983.3036501 | **132× LOOSER** |
| `4stufen` | min | 20,282.0507 | 100,992.3292 | ~5× tighter |
| `casctanks` | min | −149.87196 | −102.49921 | 32 % tighter |
| `beuster` | min | 6,352.0632 | 6,431.0426 | 1.2 % tighter |
| `bchoco06` | max | 0.99997757 | 1.0000000000 | 2.2e-5 looser |
| `bchoco07` | max | 1.000000000000286 | 1.000000000000212 | 7e-14 tighter |
| `bchoco08` | max | 1.000000000002788 | 1.000000000000547 | 2.2e-12 tighter |
| `heatexch_gen1` | min | 38,183.5317460179 | 38,183.5317460091 | neutral |
| `heatexch_gen2` | min | 555,767.79028573 | 555,767.79028575 | bound neutral — **ON gains a first incumbent, 814,343.765, where OFF has none** |
| `heatexch_gen3` | min | **no bound at all** | 783.0180645 | **ON gains a bound where OFF produces none** |
| `gkocis` | min | −1.9230988280923 (`optimal`) | −1.9230988280949 (`optimal`) | neutral, both certified |
| `syn05hfsg` | max | 837.732412391107 (`optimal`) | 837.7324093630 (`optimal`) | both certified, ON's gap 6.3e-8 vs 0 |
| `st_e11` | min | 189.31162974 **`optimal`** | 189.36307863 **`feasible`** | certification lost **at this budget** — see the retraction below |

`heatexch_gen2` and `heatexch_gen3` are the `watercontamination0202` signature
reproduced on instances that exist here: deleting rows and variables lets the primal
heuristics and the root bound reach something the full model does not. `hda` and
`4stufen` reduce comparably (303 and 33 eliminations) and move the bound in
**opposite directions by orders of magnitude**, so the effect is not monotone and no
predictor of it is available from this measurement.

**Soundness across the panel.** Four of the 13 have a proven reference optimum
(`reference_optima.oracle_table`, from `cert-optima.json`): `hda` −5964.534084,
`gkocis` −1.923098738, `st_e11` 189.3116297, `syn05hfsg` 837.7324009. **No ON-arm
bound crossed its optimum on any of them**, and no ON-arm incumbent beat one by more
than the oracle's own precision (`gkocis` 3.5e-8 below, matching the OFF arm's
4.2e-8 — an oracle-precision artifact, not the flag).

**Consequence 3 — the one real defect, and it is in the guard, not the transform.**
`ModelRepr.evaluate_point` — the whole of the #779 postsolve guard — checks rows and
variable bounds and **nothing else**. A lifted point whose integral survivor is
fractional passes both arms and is reported as the answer: the new regression test
demonstrates `lift_result` returning `SolveResult(status='optimal', obj=8.0)` at
`y = 3.5` before the fix, and `None` after. Not reachable through today's
substitution pass (`substitute.rs` "Scope (v0)" never eliminates an integral block),
but the guard is the last check before a point becomes the reported solution and
Card 3d proposes to reuse it for a transform where that upstream invariant does not
hold. Fixed in `_presolve_substitute.integrality_violation` as an O(n) numpy pass, not
by delegating to `verify_point`, so the guard's own measured reason for using the Rust
evaluator (0.008 s vs >119 s for the JAX path on this pass's target models) survives.

**RETRACTION, same session (CLAUDE.md §11): "`st_e11` is a categorical certification
regression" — WRONG, it is budget-sensitive, and the cost is WALL, not nodes.** The
30 s panel row above reads as a hard regression. A clean A/B at **120 s**, run alone
after everything else finished, two replicates per arm, order-balanced OFF/ON/ON/OFF,
says otherwise:

| instance | arm | status | objective | bound | nodes | wall |
|---|---|---|---|---|---|---|
| `st_e11` | OFF | `optimal` | 189.31162974131917 | 189.31162968661766 | **27** | 6.65 s / 8.18 s |
| `st_e11` | ON | `optimal` | 189.31162991242556 | 189.31162943443508 | **5** | **30.45 s / 40.61 s** |
| `syn05hfsg` | OFF | `optimal` | 837.7324123911069 | 837.7324123911069 | **185** | 30.30 s / 28.39 s |
| `syn05hfsg` | ON | `optimal` | 837.7323567390654 | 837.7324093629704 | **89** | **15.94 s / 15.68 s** |

Both certify in both arms given room. What the flag actually costs on `st_e11` is
**wall**: it cuts the tree from 27 nodes to 5 and still takes **4–5× longer**, which
is what pushed it past a 30 s budget. And even that is not monotone — on
`syn05hfsg` the same flag halves both the nodes (185 → 89) and the wall (~29 s →
~16 s). Every dimension of this flag's effect (bound, nodes, wall) moves in **both**
directions by large factors.

**Flag verdict: PARKED, graduation REFUSED, two blockers named.** Sound — 16,800+
executed comparisons, zero failures, no bound above a proven optimum anywhere in the
panel. But (a) under a *fixed* panel budget it converts `st_e11` from
`gap_certified=True` to `False`, and a graduation panel is by construction run at a
fixed budget, so §0.1's certification bar bites; and (b) there is **no predictor** of
which way it will move any given instance — `hda` vs `4stufen` on the bound,
`st_e11` vs `syn05hfsg` on the wall. `DISCOPT_CUT_INHERIT` again: sound ≠ helpful.
**Not removed** — it is the only pass in the tree that records an inversion payload,
and Card 3d plus the deferred FBBT-coupled fixing loop both name it as the pattern.
Building the predictor is what would unblock graduation; inventing one from this
measurement would be a hypothesis-driven fix (§4).

**Recorded, not fixed:** `substitute.rs`'s `eliminable` predicate tests
`!variables[b].lb.is_empty()` but line 415 indexes `variables[e].ub[0]`. Provably
unreachable — `model_to_repr` extracts `lb` and `ub` from the same Python variable, so
they are always the same length — and closing it would mean a Rust rebuild for a
panic that cannot occur, so it is written down rather than patched.

### 2026-07-30 — Card 3d entry experiment: "adopting the presolved repr delivers the 8,614 dropped rewrites to the relaxation compiler" — **the rewrites are real, the benefit is throughput, and the bound-relevant half already exists elsewhere. NOT BUILT**

**Hypothesis (the card's, from Card 2c.2).** `simplify`, `redundancy` and
`coefficient_strengthening` produce 8,614 non-bound rewrites that
`propagate_bounds_to_model` cannot carry, so solving from the presolved repr with a
postsolve chain would deliver them to the Python relaxation compiler and tighten
what every node LP is built from.

**Kill criteria, stated before the run.** (a) If the rewrites are overwhelmingly row
*removal* with zero bound movement, the delivery buys smaller LPs, not tighter ones —
wall-clock, which the 2026-07-30 re-sequencing put off the critical path. (b) If the
adoption cannot be scoped variable-preserving, it needs a postsolve chain, and the
card must show one can be built from what the orchestrator records.

**Experiment.** 147 `.nl` files (`python/tests/data/minlplib_nl` 66 +
`python/tests/data/minlplib` 81). Each run twice through `PyModelRepr.presolve`: once
with the three indicted passes **alone**, once with the **full** list `solve_model`
runs. 147 + 147 runs, **0 errors**. Marker asserted before measuring.

*Q1 — the three passes ALONE (17 unique instances carry any rewrite):*

| pass | rows removed | rows rewritten | **bounds tightened** |
|---|---|---|---|
| `simplify` | 1,440 | 0 | **0** |
| `redundancy` | 149 | 0 | **0** |
| `coefficient_strengthening` | 0 | 52 | **0** |

Variable count **unchanged on all 147** — so an isolated-three adoption would need no
postsolve at all.

*Q2 — the FULL pass list:*

| pass | rows removed | rows rewritten | bounds tightened |
|---|---|---|---|
| `simplify` | 6,635 | 0 | 0 |
| `redundancy` | 149 | 0 | 0 |
| `coefficient_strengthening` | 0 | 52 | 0 |
| `factorable_elim` | 45 | 0 | 0 |
| `aggregate` | 18 | 0 | 18 |
| `eliminate` | 9 | 0 | 18 |
| `probing` | 0 | 0 | 4,111 |
| `fbbt` | 0 | 0 | 3,181 |
| `implied_bounds` | 0 | 0 | 3,142 |

Variable count changes on **4** unique instances — `hda` 722→719, `casctanks`
500→490, `4stufen` 149→148, `util` 145→144 — and **never grows** (0 of 147).

**Both kill criteria fire.**

1. **Zero bounds from all three passes across 147 instances** — Card 2c.2's headline,
   reproduced at a different scale. A row `redundancy` removes is implied and cannot
   tighten an LP relaxation; `simplify` moved no bound anywhere. So ~99 % of the
   rewrite volume is row removal, i.e. smaller node LPs.
2. **The safe scoping and the useful scoping are disjoint.** Run alone the three
   passes are variable-preserving (no postsolve needed) but `simplify` finds only
   1,440 of its 6,635 rows — **78 % of its effect depends on the other passes'
   tightenings**. The full list changes the variable space, so the useful version
   needs a postsolve chain, and one cannot be assembled from what exists:
   `substitute.rs` is the **only** pass that emits an inversion payload
   (`SubstitutionRecord` + `block_map`), while `EliminatePass`, `AggregatePass` and
   `FactorableElimPass` record index lists in the delta stream — what happened, not
   how to undo it.

**And the one bound-relevant component is already built, in the right place, and
stronger.** Coefficient strengthening (52 rows, `gbd` + `hda`) exists on the Python
side as `solvers/_root_presolve.py` (`DISCOPT_COEF_TIGHTEN`, parked). Its "NOTE ON
LOCATION" gives the reason — "rewriting the Python model at the root is the only place
the tightened coefficients actually reach the relaxation" — and records the Rust pass
as *weaker*: it reads declared bounds, so it bails on the `[0,∞)` flows this family
declares, and it skips negative fixed-charge binary coefficients. Adopting the Rust
rewrite would import the weaker of the two.

**Decision: NOT BUILT.** "The most dangerous change in the plan" is not worth a
throughput gain on a correctness benchmark, and a partially-wired repr adoption is
worse than none (the card's own warning). Prerequisite to reopening: a postsolve
payload on every variable-changing Rust pass, plus a benefit that is not wall-clock.
Card 6a's finding that the substitution machinery **is** sound removes the soundness
objection to reusing it as the pattern, but not the benefit objection.

### 2026-07-30 — the `tls2` anomaly: "the 9.3 arm may be the reproducible one, and the baseline row the stale one" — **NO. 5.3 is right, 9.3 is a sound truncated incumbent, and the row needs no re-record**

**The flag.** The passing Regime-N panel moved `tls2` `nodes 421→373,
obj 5.299999922109238→9.29999987524207` — correctly non-gating (the baseline row
certifies at 43.8 s = 97 % of the 45 s budget, so it is marked `comparable=false`)
but a ~76 % objective move, which the Card 4c close-out flagged rather than buried.
The open question was whether 9.3 was a **certified** optimum, which for a minimize
instance with a proven optimum of 5.3 would be a false optimum and a hard defect.

**Experiment.** `tls2` alone, subprocess-isolated, nothing else running: 3 replicates
at the panel's 45 s budget, then 2 at 300 s. Reference optimum **5.3, proven**
(`docs/dev/data/cert-optima.json` via `reference_optima.oracle_table`).

| budget | rep | status | objective | bound | `gap_certified` | nodes | wall |
|---|---|---|---|---|---|---|---|
| 45 s | 0 | `feasible` | 5.299999922109238 | 3.3746135 | **False** | 421 | 46.18 s |
| 45 s | 1 | `feasible` | 10.299999865440752 | 2.4487126 | **False** | 353 | 48.70 s |
| 45 s | 2 | `feasible` | 9.29999987524207 | 2.5000000 | **False** | 373 | 46.30 s |
| 300 s | 0 | **`optimal`** | **5.299999922109238** | **5.3** | **True** | **421** | 63.06 s |
| 300 s | 1 | **`optimal`** | **5.299999922109238** | **5.3** | **True** | **421** | 71.62 s |

**Answer: 5.3 is right**, to the last digit of the proven optimum, and it is
*deterministic* — both 300 s runs certify at the same objective, the same bound and
the same **421** nodes. The instance simply needs ~63–72 s on this machine, which is
1.4–1.6× the panel's 45 s budget.

**The 9.3 (and 10.3) arms are NOT false optima.** Every 45 s run returns
`status=feasible` with `gap_certified=False`: a sound, time-limited incumbent above
the optimum, from a search that ran out of budget at a different point. Nothing
crossed the oracle in any arm. The 45 s node counts (421 / 353 / 373) measure how far
the search got, not a nondeterministic tree — the completed search is 421 nodes every
time.

**No re-record, and no issue.** The baseline row's *objective* is confirmed correct by
the certifying runs; only its terminal status is a lucky machine-minute, and the panel
already excludes it for exactly that reason ("certified at 43.8 s = 97 % of the 45 s
budget"). The `MARGIN_FRAC` filter did its job. Rewriting the frozen artifact would
break the 255/85 comparison history that every Regime-N run since Phase 3 shares, for
no gain. There is no solver nondeterminism to file: this is budget starvation on an
instance whose true cost is above the panel budget, which is what `comparable=false`
means.

### 2026-07-30 — Card 6a / Card 3d / `tls2` close-out: what was run on the final tree

Recorded per §0.6. Tree: `9bd93ca3` + this session's commits; `HEAD == origin` asserted
before each commit (origin stayed at `9bd93ca3` throughout — nothing else moved it).
Compiled `.so` marker asserted **before every measurement**: `strings` on
`_rust.cpython-311-x86_64-linux-gnu.so` finds `subtol_crossings_repaired`, the PyO3
string unique to the newest Rust commit `8532ce2d`, so the binary matches the newest
Rust sources. **Rust untouched this session**, therefore `cargo test -p discopt-core`
was **not run** (nothing in `crates/` changed).

**Suites** (final tree, final `.so`).

| suite | result |
|---|---|
| `pytest python/tests -m smoke` | **PASS — 947 passed**, 16 skipped, 7,756 deselected, 2 xpassed (540.6 s). Pre-session baseline 946; the +1 is exactly `test_postsolve_guard_rejects_a_lifted_point_with_a_fractional_integer` |
| `pytest discopt_benchmarks/tests -m smoke` | **PASS — 51 passed, 1 skipped** |
| `pytest -m slow test_adversarial_recent_fixes.py` | **PASS — 10 passed** (219.9 s) |
| `pytest test_node_tightening_parity.py` | **PASS — 4 passed**, 12 deselected |
| `pytest -m slow test_node_tightening_parity.py` | **PASS — 12 passed**, 4 deselected (77.1 s) — Card 3c's guard, vector arm included |
| `pytest test_vector_constraint_corpus.py` | **PASS — 37 passed** |
| `pytest test_constraint_rhs_refusal.py` | **PASS — 24 passed** |
| `pytest -m slow test_stray_bb_loop_invariants.py` | **PASS — 7 passed** (154.2 s). Counts: `i1_fathom_decisions` **11,864**, `i1_fathomed` 3, `i2_bound_vs_oracle` 3, `i3_incumbent_verified` 5, all three loops observed |
| `pytest test_presolve_substitute.py` | **PASS — 7 passed**, 4 deselected |
| `pytest -m slow test_presolve_substitute.py` | **PASS — 4 passed**, 7 deselected (50.4 s). The corpus-exactness arm prints `instances_reduced=13 surviving_row_comparisons=8724 dropped_row_checks=2428 box_in=26 box_out=26 problems=0` |
| `pytest test_flag_registry.py` | **PASS — 17 passed.** A flag row's doc changed, so `docs/reference/flags.md` was regenerated by `scripts/gen_flag_docs.py`; the staleness test confirms they match |
| `ruff check python/` + `ruff format --check` | clean on every file this session touched |
| `cargo test -p discopt-core` | **not run — `crates/` untouched** |
| heldout50 | **SKIPPED — local only.** Not available in this environment |

**Regime N — reported exactly as it came out, not laundered. The panel returned FAIL
in BOTH runs, on DIFFERENT single instances, and neither is attributable to this
session.**

| run | condition | verdict |
|---|---|---|
| A | ran **concurrently** with the suite block (load start 0.25, peak >3 on a 4-core box) — my own contention, disclosed | **FAIL, 4 violations, all on `ex1266`**: `optimal → time_limit`, certification lost, nodes 6005 → 1279, certified objective 16.3 → None |
| B | re-run **alone**, nothing else on the box (load start 1.16, peak 3.48, wall 2341.8 s) | **FAIL, 1 violation**: `gear2` nodes 3 → 91. `ex1266` PASSES here — `optimal`, nodes **6005**, obj **16.3**, `cert=Y`, 9.1 s, comparable |

Both flagged instances were then re-measured **alone**, and both reproduce the frozen
baseline **exactly**:

* `ex1266` × 3: `optimal`, obj **16.3**, bound 16.3, **6005 nodes**, certified, 8.5–9.0 s
  (baseline: 6005 nodes, obj 16.3, certified in 6.1 s);
* `gear2` × 5: `optimal`, obj **1.155529714729433e-07**, **3 nodes**, certified, 6.5–6.7 s
  — bit-identical to the baseline's `1.155529714729433e-07` / 3 nodes.

**And this session's only code change is *measured* unreachable on the panel's default
settings**, not merely argued to be. Instrumenting
`solvers/_presolve_substitute.{build_reduced,lift_result,integrality_violation}` and
solving both instances with `DISCOPT_PRESOLVE_SUBSTITUTE` unset:
`build_reduced` entered **2/2**, returned `None` **2/2**, `lift_result` called **0**,
`integrality_violation` called **0**. The Card 6a arm cannot execute on this path, so
it cannot move a node count on it.

**Comparison population unchanged:** run B executed `comparisons executed: 255
(node_count 85, certified objective 85, status 85) over 85 comparable of 119 baseline
rows` — the same 255/85 as Phase 3, Phase 5, Card 3e, Card 4a and Card 4c. 18
non-comparable rows reported and not gating, the same character as every prior run.

**Conclusion, stated as evidence rather than as a pass.** Two runs of the same tree
each flagged a *different single* instance; each flagged instance is bit-identical to
the baseline when run alone; and the only code change is measured unreachable on the
path being gated. The drift is **environment sensitivity of the harness on this
4-core container**, not a bound-neutrality violation of this session's work — but the
panel's literal exit code was FAIL both times and is recorded as such. **New open
item (ledger row 15): the Regime-N panel is no longer reproducible instance-for-
instance in this environment**; before it can gate again it needs either a
per-instance replicate-and-agree rule or a `comparable` filter that also excludes rows
whose *preprocessing* phases are budget-sensitive. That is a measurement-substrate
task (Phase 0), not a solver task.

> **RETRACTED 2026-07-30 (CLAUDE.md §11).** The last sentence above is wrong. The
> root-cause experiment (see the final §6 entry, "open-ledger item 15") measured the
> mechanism: it is the **solver's** wall-clock-bounded root primal heuristic, not the
> harness's bookkeeping, and the flagged rows are not "preprocessing"-budget-sensitive
> in the sense meant here — `gear2`'s root heuristic runs to a 5 s wall deadline every
> time, and forcing that budget alone steps its node count 3 → 91 → 93. The remedy
> shipped is a gate-level replicate-and-agree rule (Phase 0 addendum); the *substantive*
> fix is deterministic work budgets in the solver, filed as ledger row 15b. The rest of
> this entry's observations stand.

**Disclosed measurement conditions (CLAUDE.md §9).** Run A's contention was
self-inflicted and is the reason it is reported as invalidated rather than as a
result; run B was the remedy. The Card 6a bound A/B (12 instances) and the `tls2`
probe were each run with nothing else on the box, and the Card 6a A/B arms were
**bit-identical across replicates within each arm**, so no verdict in this session's
§6 entries is wall-based except the explicitly-labelled `st_e11`/`syn05hfsg` wall
comparison, which was measured alone at 120 s with two order-balanced replicates per
arm.

### 2026-07-30 — Open-ledger item 15: "the Regime-N panel's instance-for-instance irreproducibility is a harness artifact, closable in Phase 0" — **FALSIFIED. It is the solver, the panel was right to flag it, and the ledger's own framing was wrong**

**Hypothesis** (ledger row 15, as filed): two runs of the same tree flagging a
*different single* instance is "environment sensitivity of the harness on this 4-core
container", closable by "a per-instance replicate-and-agree rule, or extending the
`comparable` filter to exclude rows whose *preprocessing* phases are budget-sensitive.
**Phase 0 work, not solver work.**"

**Kill criterion.** If a *specific* wall-clock-gated phase can be shown to move
`node_count` on a flagged instance with everything else held identical, then the
irreproducibility is a property of the solver's budget arithmetic, not of the
harness's bookkeeping — the last sentence of the hypothesis is false, and any remedy
that only edits the `comparable` filter is treating a symptom.

**Experiment.** `discopt_benchmarks/scripts/item15_root_budget_probe.py`, four arms,
artifacts `reports/item15_*.json`. Every child asserts `discopt.__file__` *and* the
`subtol_crossings_repaired` marker inside `_rust*.so` before it measures anything
(CLAUDE.md §8) and raises rather than swallowing (§7); a child that emits no
`RESULT_JSON` is a hard error, not a skipped row.

**Static census first.** `time.perf_counter()` / `time.monotonic()` / `deadline`
comparisons that gate a *branch* (not merely a report) in `python/discopt/`:
**78 sites**, led by `solver/__init__.py` (26), `_jax/primal_heuristics.py` (10),
`_jax/mccormick_lp.py` (6), `solver/native_kernel.py` (5), `_jax/obbt.py` (5). Plus
57 `Instant::now` sites in `crates/`. The search reads the clock everywhere.
(Derivation, so the figure is auditable rather than asserted: `grep -rnE
"perf_counter\(\) *[<>=]|perf_counter\(\) *- *[A-Za-z_0-9.]+ *[<>]|monotonic\(\) *>=|
_deadline_exhausted\(\)|deadline_exceeded\(\)|_remaining_budget\(\)"` over
`python/discopt/` excluding `llm/` matches **82** lines; **4** of those are the helper
*definitions* and their docstrings, leaving **78** that gate a branch.)

**Arm 1 — `--arm observe`: the first candidate is ruled OUT.** The ledger and the
plan both suspected root OBBT (`_obbt_budget = min(min(max(time_limit·0.1, 2.0),
15.0), _remaining_budget())`, `solver/__init__.py:6549`, and the candidate sweep
breaks on `deadline` at `obbt.py:1199/1229`). Measured on `gear2`, `ex1266`, `gear`,
`gear3`, `gear4` at 45 s, idle box: **5 observations, 9 `obbt_tighten_root` calls
observed, 0 of them returned at or past their deadline.** Consumed 2–45 ms against
4.5–5.0 s budgets — root OBBT converges long before its clock and cannot be the
mechanism. `ex1266` never calls it at all (`root_time` 0.03 s of a 6.1 s solve).

**Arm 2 — `--arm forcebudget --phase ils`: the causal arm.** `gear2` spends 5.96 s of
its 6.56 s wall in the *root*, and a `cProfile` of the whole solve puts 5.0 s of it in
`primal_heuristics.integer_local_search` (called from `solver/__init__.py:9660` with
`time_budget=min(5.0, 0.15·time_limit)`, descending
`while improved and time.perf_counter() < deadline`, `primal_heuristics.py:768`).
Forcing that one budget, everything else identical, 2 replicates each, idle box —
**12 executed comparisons, 0 errors** (`reports/item15_forcebudget_ils_gear2.json`):

| forced ILS budget | node_count | certified objective |
|---|---|---|
| 5.00 s (the default) | **3**, 3 | 1.155529714729433e-07 |
| 4.00 s | **3**, **91** | 1.1555e-07 / 1.2492e-07 |
| 3.00 s | 91, 91 | 1.2492237047931615e-07 |
| 2.00 s | 91, 91 | 1.2492e-07 |
| 1.00 s | 91, 91 | 1.2492e-07 |
| 0.50 s | 93, 93 | 1.2492e-07 |

and the heuristic **consumed 5.02 s of its 5.00 s budget** — it never converges, it
always runs out. So the default sits directly on a cliff edge, and which side of it
the run lands on is decided by how fast the machine was for those five seconds. The
objective moves too (1.1555e-07 → 1.2492e-07, Δ = 9.4e-9) but stays *inside* the
panel's `1e-8 + 1e-9·|obj|` tolerance — which is exactly why the recorded failure
reported one violation (node count) and not two.

**Arm 3 — `--arm clockscale`: does it generalise?** Scaling `time.perf_counter` /
`time.monotonic` by `alpha` is, from the solver's point of view, indistinguishable
from a machine `alpha` times slower, and reaches all 78 Python sites at once. Over
the **entire 85-row Regime-N comparable population** at alphas 1.0/1.25/1.5/2.0 —
**255 executed comparisons** (`reports/item15_clockscale_comparable.json`):

* **alpha = 1.0 control: 85 of 85 rows reproduce the FROZEN baseline exactly, 0
  mismatches.** The tree is unchanged, the harness is sound, and the two recorded
  failures were not the tree.
* alpha = 1.25: **0** of 85 moved. alpha = 1.5: **0** of 85 moved.
* alpha = 2.0: **1** of 85 moved — `gear2`, 3 → 91 nodes, status and certification
  unchanged. The same step the forced-budget arm produced, from an independent knob.

The emulation is a **lower bound**: it cannot reach `crates/`. `ex1266` does not move
even at alpha = 8 (`reports/item15_clockscale_ex1266.json`, 4 comparisons) because its
whole-solve budget is enforced Rust-side.

**Arm 4 — `--arm load`: both recorded failure modes, reproduced on demand.** 24 busy
processes on 4 cores, 2 replicates per condition — **8 executed observations**
(`reports/item15_load_reproduction.json`):

| instance | idle | loaded (24 spinners) | which recorded failure this is |
|---|---|---|---|
| `ex1266` | `optimal`, **6005**, **6005** | **`time_limit`**, **1**, **7** | run A (status flip, certification lost, node collapse) |
| `gear2` | `optimal` cert, **3**, **3** | `optimal` cert, **91**, **93** | run B (status and certificate intact, node count drifts) |

`gear2` under load is not even self-consistent (91 vs 93) — `NONDETERMINISTIC` by the
adjudicator's own definition. Two mechanisms, not one: `ex1266` is whole-solve budget
starvation (its root is 0.03 s — **not a preprocessing phase at all**), `gear2` is the
root primal heuristic's wall budget.

**Verdict: FALSIFIED, on the part that matters.** The irreproducibility is real and it
is the **solver's**, not the harness's. The ledger's "Phase 0 work, not solver work" is
retracted in place (see the retraction block in the Card 6a/3d/`tls2` close-out entry,
CLAUDE.md §11).

**Which remedy, and why not the other.** The ledger offered two.

* **Rejected — widen the `comparable` filter to exclude budget-sensitive rows.**
  Three independent reasons, in order of severity. (1) *It cannot be implemented
  without violating CLAUDE.md §2.* There is no static property that marks `gear2`:
  it is `optimal`, `gap_certified`, and finishes in **15 %** of its budget, so every
  filter Phase 0 has admits it. The only way to *know* it is sensitive is to run it
  at several clock scales — i.e. the replicate machinery — and the only cheap
  alternative is a hardcoded instance list keyed to problem names, which is exactly
  what §2 forbids. (2) *It is a permanent weakening (§0.4).* An excluded row never
  gates again, including against a real regression; adjudication costs a row's
  gating only on the run where it actually deviated. (3) *It does not even cover the
  observed defect set.* `ex1266`'s failure is whole-solve starvation with a 0.03 s
  root, so a filter on *preprocessing* budget-sensitivity — the ledger's own wording
  — would not have excluded it.
* **Chosen — per-instance replicate-and-agree adjudication** (Phase 0 addendum;
  `_adjudicate` in `panel_baseline.py`). It keeps every row in the gate and decides
  per *observation* rather than per *instance*.

**Proof it does not weaken detection of real drift (§0.4 is not satisfied by
argument).** A bound-neutrality violation is deterministic — the changed code runs on
every replicate — so it lands in `CONFIRMED` and still fails. Demonstrated end to end
on the *flaky instance itself*: `gear2`'s frozen row perturbed by **one node** (3 → 4),
`--check --subset gear2`:

```
comparisons executed (total): 12 = 3 first-pass + 9 adjudication over 1 comparable row(s)
  gear2 replicate 1/3: optimal nodes=3 ... 2/3: optimal nodes=3 ... 3/3: optimal nodes=3
  -> gear2: CONFIRMED — 3/3 isolated replicates agree with each other and DISAGREE
                        with the baseline — reproducible drift.
FAIL: gear2: NODE COUNT drift 4 -> 3
```

and the clean arm on the same row PASSES with 3 executed comparisons, so the failure
is attributable to the injected perturbation and not to ambient flakiness. The same
property is pinned in CI by `test_check_detects_a_perturbed_node_count`, which now
requires the verdict to be `CONFIRMED` and explicitly **not** `TRANSIENT`, and by five
pure-logic arms covering all three verdicts, the zero-replicate refusal, and the
signature's field set.

**The residual, stated rather than hidden.** A *rare* real drift that fires in the
first pass and in none of the replicates is recorded as `TRANSIENT`. That is why every
transient row is printed in full, lands in the exit summary and the check artifact,
and is capped at `--max-transient`. It is a disclosure, not a dismissal — and the
substantive fix is ledger row 15b (deterministic work budgets), not a bigger `R`.

### 2026-07-30 — Open-ledger item 15 close-out: what was run on the final tree

Recorded per §0.6. Tree: `08543244` (a concurrent session's test-parallelism commit,
**no file overlap** with this work) + this session's three commits. `HEAD == origin`
asserted before every commit. Compiled `.so` marker asserted **inside every probe
child, not once at the start**: `strings` on
`_rust.cpython-311-x86_64-linux-gnu.so` finds `subtol_crossings_repaired`, and
`item15_root_budget_probe.py::_assert_loaded_build` re-checks both
`discopt.__file__` and that marker in each of the ~400 subprocesses it spawned,
raising `SystemExit` rather than measuring a stale build (CLAUDE.md §8).
**Nothing under `python/` or `crates/` was touched**, so `cargo test -p discopt-core`
and `mypy python/discopt/` were **not run** — the change is confined to
`discopt_benchmarks/` and `docs/`.

| suite | result |
|---|---|
| `pytest discopt_benchmarks/tests/test_panel_baseline.py` | **PASS — 21 passed** (10.97 s). Was 12 before; the 9 new arms are the three adjudication verdicts, the zero-replicate refusal, the signature's field set, the two comparator arms, the load-gate refusal, and the `--replicates 0` escape hatch |
| `pytest discopt_benchmarks/tests -m smoke` | **PASS — 53 passed, 1 skipped** (31.5 s). Pre-session 51; +2 are the load-gate and `--replicates 0` arms |
| `pytest python/tests -m smoke` | **PASS — 947 passed**, 16 skipped, 7,756 deselected, 2 xpassed (453.1 s) — identical to the pre-session count |
| `pytest -m slow python/tests/test_adversarial_recent_fixes.py` | **PASS — 10 passed** (207.5 s) |
| `ruff check` + `ruff format --check` on the three touched files | clean |
| `cargo test -p discopt-core` | **not run — `crates/` untouched** |
| `mypy python/discopt/` | **not run — `python/` untouched** |
| heldout50 | **SKIPPED — local only.** Not available in this environment |

**Regime N — the hardened gate, both directions.**

| arm | result |
|---|---|
| full 119-instance `--check` on the unmodified tree, nothing else on the box | **PASS.** `comparisons executed (total): 255 = 255 first-pass + 0 adjudication over 85 comparable row(s); flagged 0, adjudicated 0, transient 0`. 1931.6 s, load start 0.47 peak 2.39. 15 non-comparable drift rows reported and not gating. `ex1266` (6005 nodes) and `gear2` (3 nodes) both reproduced the frozen baseline exactly. Artifact `reports/item15_panel_check_hardened.json` |
| `--check --subset gear2` against the clean frozen baseline | **PASS**, 3 executed comparisons |
| `--check --subset gear2` against a baseline whose `gear2` node count was perturbed by **one** (3 → 4) | **FAIL, exit 1.** 12 executed comparisons = 3 first-pass + 9 adjudication; 3/3 isolated replicates `nodes=3`; verdict **CONFIRMED**; transient 0 |

The same 255/85 comparison population as Phase 3, Phase 5, Card 3e, Card 4a and
Card 4c. The gate now passes clean *and* still fails on a one-node injection — on
the instance whose flakiness opened the item.

**Disclosed measurement conditions (CLAUDE.md §9).** The box is a 4-core container
and was **shared with another Claude session this session** (it landed
`08543244` mid-run); that is disclosed rather than assumed away, and it is why the
`--arm clockscale` population sweep carries an explicit `alpha = 1.0` control —
85/85 rows reproducing the frozen baseline is the evidence that no foreign load
contaminated it. Nothing else was run during the clockscale sweep, the load-arm
reproduction, or the full panel `--check`; the suites were run only after the panel
finished. The `--arm load` "idle" condition ran at a *decaying* 1-minute average of
1.9–4.1 (residue of the preceding sweep plus the panel child's own JAX/BLAS
threads) and still reproduced the baseline exactly on both instances, which is
itself the calibration behind the `--max-load` default of 2.0. All 24 spinner
processes were killed and verified gone (`ps --sort=-pcpu`) before any subsequent
measurement — the §9 "check for stray load you created yourself" step, which has
invalidated a round in this repo before.

**Can item 15 close? YES.** The defect is root-caused with executed counts, the
remedy is implemented with the alternative explicitly rejected and why, the "does
not weaken the gate" constraint is demonstrated rather than argued, and the plan
carries the addendum every later card must follow. What is *not* closed, and is
filed rather than hidden: **ledger row 15b** — the solver still decides how much
work to do by reading a clock, and until that is replaced with deterministic work
budgets, an identical model and `time_limit` are not guaranteed to give an identical
tree across machines. Adjudication makes the gate trustworthy; it does not make the
solver reproducible.
