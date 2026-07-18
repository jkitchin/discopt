# Issue #671 — resolution plan (finish & close)

**Audience:** the implementing agent (Opus) on branch
`claude/issue-671-resolution-lm4bdh`.
**Goal:** make hda's tight dual bound the *default* behavior, panel-clean, and
close #671.

## 0. State of the world (verified 2026-07-18 against `main` @ `075c904`)

The issue's target — hda reporting a tight finite dual bound (≈ −6.45e4 vs
candidate A's −1.80e10) with no panel regression — is *mechanically achieved*
but *not shipped as the default*, and the owner's graduation panel (issue
comment 2026-07-18) returned **HOLD**. Three facts drive everything below:

1. **PR #746 was squash-merged from an early point of branch
   `claude/discopt-issue-671-46isrd`** — the *always-on build-time* row filter
   (`build_milp_relaxation` tail, `milp_relaxation.py:1957`). That is exactly
   the first cut PR #746's own body says was **KILLED** (10/66 in-repo
   regressions, nvs09 losing its certificate).
2. **The failure-triggered refactor exists but is stranded, unmerged**, on the
   tip of `origin/claude/discopt-issue-671-46isrd`:
   - `4ebade5` fix(#671): make the row filter FAILURE-TRIGGERED (was always-on)
   - `8de0365` test(#671): panel separates HARD from SOFT changes
   - `645f7cf` docs(#671): failure-triggered panel verdict (**ACCEPTANCE
     PASS** — 0 UNSOUND, 0 HARD, 7 SOFT on both-arms-timeout instances)
   That branch **predates the #732 work now on `main`** (empty-box guard,
   directional bound widening, disjunctive-config floor, overflow-safe
   trigger guard), so it cannot be merged wholesale — the refactor must be
   ported hunk-by-hunk without reverting #732.
3. **The owner's graduation harness lives on
   `origin/feat/graduate-relax-row-filter-671`** (`584eeab`):
   `discopt_benchmarks/scripts/issue671_rowfilter_graduation_panel.py` + raw
   data + VERDICT.md. Its HOLD verdict reproduced the build-time first-cut
   regressions on `main` (filter fires on 30/66, regresses 4 already-solving:
   nvs05/nvs09 lose certificates, nvs13/nvs21 drift node_count/objective) and
   explicitly defined what closes the issue:

   > Make the filter genuinely failure-triggered — fire only when a node LP
   > breaks down numerically, so already-solving nodes are byte-identical …
   > Re-run this panel; graduation needs cert-clean = 0 already-solving
   > regressions AND the hda gain.

Secondary levers already on `main`, both default-OFF, both stay opt-in:
`DISCOPT_LP_ITERATIVE_REFINEMENT` (τ-regularized resolve + NS bound; delivers
hda ≈ −6.45e4 but ~doubles hda wall time) and
`DISCOPT_LP_FACTORIZATION_HARDENING` (validated primitive; measured NOT to
rescue hda). The row filter is the graduating lever: it is cheaper (one
re-solve per failed node, no τ-sweep), tighter on hda (−64675.25, exactly the
τ-homotopy limit, NS-certified by feral's own dual at τ=0), and
failure-triggered by design.

## 1. Definition of done

1. `DISCOPT_RELAX_ROW_FILTER` is **failure-triggered** (fires only after a node
   LP exits without a certified verdict: not `optimal` and not
   Farkas-certified `infeasible`), never at build time.
2. The §5 graduation panel (the owner's harness, same protocol) passes both
   bars: **cert-clean** — `incorrect_count = 0`, zero already-solving
   regressions (nvs05/nvs09/nvs13/nvs21 byte-identical: status, certificate,
   `node_count`, objective), no bound above its reference optimum, no
   certification regression, incumbents independently feasibility-verified —
   AND **net-positive** — hda OFF → ON: −1.80e10 → ≈ −6.45e4, sound
   (≤ opt −5964.534084), without broad harm elsewhere.
3. The flag graduates **default-ON** with the `DISCOPT_RELAX_ROW_FILTER=0`
   opt-out and the legacy (no-filter) path intact (CLAUDE.md §5 graduation
   policy, 2026-07-17).
4. PR merged; issue closed with the close-out summary (§8 below), explicitly
   stating the disposition of the two opt-in sibling flags.

If (2) fails again, the issue does **not** close by fiat — see §7
(contingencies). Correctness rules; never weaken the panel to pass.

## 2. Step 1 — port the failure-triggered refactor onto current `main`

Branch: `claude/issue-671-resolution-lm4bdh`, reset onto latest `origin/main`.

Attempt `git cherry-pick 4ebade5 8de0365 645f7cf` first; expect conflicts in
`mccormick_lp.py` / `milp_relaxation.py` / `solver_tuning.py`. Whether
cherry-picking or hand-porting, the port consists of exactly four production
hunks plus tests/docs:

- **`python/discopt/_jax/mccormick_lp.py`** — in `_solve_at_node_impl`, after
  the cold-path solve (and after the C-42 pool-drop re-solve block, currently
  ending near line 1344): if `_tuning().relax_row_filter` and the result is
  neither `optimal` nor Farkas-certified `infeasible`
  (`getattr(res, "farkas_certified", False)` — the attribute exists on `main`,
  `solvers/milp_simplex.py:185`), call `_filter_unresolvable_rows(milp)` and,
  if it dropped rows, re-solve once with the remaining time budget.
- **`python/discopt/_jax/milp_relaxation.py`** — remove the build-time filter
  call at the tail of `build_milp_relaxation` (lines 1955–1959 on `main`);
  keep `_filter_unresolvable_rows` itself (it becomes the solve-layer
  primitive). Add the comment explaining why the filter is NOT applied at
  build time (panel evidence).
- **`python/discopt/solver_tuning.py`** — replace the `relax_row_filter`
  docstring with the failure-triggered contract (take the branch's wording).
- **`python/tests/test_relax_row_filter.py`** — take the branch's updated
  tests (build-time no-op; failure-triggered firing; byte-identical on clean
  solves).
- Tests/docs/panel from `8de0365`/`645f7cf`
  (`discopt_benchmarks/results/issue671/rowfilter_diff_panel.py`, the
  ACCEPTANCE-PASS note in
  `docs/dev/hda-certification-rowfilter-entry-2026-07-18.md`).

**Porting guardrail (the reason wholesale merge is forbidden):** after the
port, `git diff origin/main -- <each touched file>` must show **only**
row-filter-related changes. Specifically it must NOT revert any of `main`'s
#732 hunks, which the stale branch predates:

- the empty-box guard + `_EMPTY_BOX_TOL` in `mccormick_lp.py` (`solve_at_node`
  wrapper);
- the *directional* out-of-cap bound widening (`lo → -inf`, `hi → +inf`) in
  both `mccormick_lp.py` and `sanitize_relaxation_for_conditioning`;
- the disjunctive-config floor plumbing (`_disjunctive_floor`);
- the cross-multiplied (overflow-safe) `_RELAX_FALSE_INFEAS_TRIGGER` guard in
  `milp_relaxation.py`.

If any of those appear in the diff, the port is wrong — redo it.

## 3. Step 2 — verify the design assumptions on today's tree

Cheap checks before running anything heavy:

1. **Node-locality of the mutation.** `_filter_unresolvable_rows` mutates
   `milp` in place. On the cold path `milp` is built fresh per node
   (`build_milp_relaxation` call at `mccormick_lp.py:1198`), so the mutation
   is node-local. Confirm nothing pools/caches that object across nodes
   (the "pool" in this file is a *cut* pool of rows appended per node, not a
   model-object pool — verify that is still true).
2. **Incremental fast path.** `_try_incremental_node` returns before the cold
   build. Confirm a fast-path result cannot surface an uncertified failure
   (`numerical` / non-Farkas `infeasible`) that bypasses the filter — i.e. the
   fast path falls through to the cold build on such failures, or its
   failures are already routed there. If a bypass exists, either route those
   through the cold path (preferred if trivially safe) or record the gap in
   the doc — do not silently expand the filter into the fast path without
   re-running the panel.
3. **Re-solve budget.** The re-solve must use the remaining node time budget
   (`_remaining()`-style), not a fresh full budget.
4. **hda end-to-end smoke** (before the panel): flag ON,
   `dm.from_nl("hda.nl").solve()` (instance from
   `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`, TL ~120 s) →
   tight sound bound ≈ −6.45e4; flag OFF → candidate-A floor
   `−18016528426.5` byte-identical. Also nvs05 + nvs09 ON-vs-OFF: identical
   status/certificate/node_count/objective.

## 4. Step 3 — standard verification battery

- `pytest python/tests/test_relax_row_filter.py -v`
- `pytest python/tests/test_issue_671_lp_iterative_refinement.py -m "not slow"`
  (candidate-A / refinement plumbing untouched)
- `pytest python/tests/test_issue_517_numerical_dual_bound.py
  python/tests/test_issue362_decline_ns_safe_bound.py` (failure-branch
  regressions)
- `pytest -m smoke` and the adversarial suite
  (`pytest -m slow python/tests/test_adversarial_recent_fixes.py`)
- `cargo test -p discopt-core` (no Rust change expected in the port; run it
  anyway — the refine/regularized_lu modules live there)

## 5. Step 4 — re-run the owner's graduation panel

Use the owner's harness, not the branch's older `rowfilter_diff_panel.py`
(the owner's is the §5 two-bar gate, sense-aware — it carries the syn05hfsg
max-sense fix — and does independent incumbent feasibility re-verification):

```
git checkout origin/feat/graduate-relax-row-filter-671 -- \
  discopt_benchmarks/scripts/issue671_rowfilter_graduation_panel.py
```

Same protocol as the HOLD run: 66-instance in-repo corpus
(`python/tests/data/minlplib_nl/*.nl`) + hda; single warm-JAX process;
corpus TL = 60 s, hda TL = 120 s; both arms record dual bound, objective,
status, `gap_certified`, `node_count`, wall, rows-dropped, independent
incumbent-feasibility verification. Write results to
`discopt_benchmarks/results/issue671/graduation_panel_<date>/`
(new dir; keep the HOLD run's data as the baseline record).

**Required outcome:** 0 UNSOUND; 0 HARD — every already-solving instance
byte-identical (in particular nvs05, nvs09, nvs13, nvs21, which the
failure-triggered design makes inert *by construction*: their node LPs solve
`optimal`, so the filter never fires); hda ON ≈ −6.45e4 sound. SOFT changes
(instances that time out in BOTH arms) are permitted but must be enumerated
in the verdict doc with their bound deltas — the prior failure-triggered
panel saw 7 (hda huge gain; bchoco/casctanks losses).

**On SOFT losses:** they arise where a failed node's filtered re-solve yields
a bound looser than the fallback bound the node would otherwise have reported.
If bchoco/casctanks-style losses persist and look material, the surgical fix
is to make the lever **monotone**: accept the filtered re-solve's bound only
via max-combine with whatever bound the failure path already produces (both
are valid lower bounds, so `max` is rigorously sound — the same "max of valid
bounds" argument the τ-sweep uses). That turns net-positive from a judgment
call into a structural property. Implement it only if the panel shows the
need; re-run the panel after.

## 6. Step 5 — graduate and protect

1. Flip the default: `relax_row_filter` → `_env_flag("DISCOPT_RELAX_ROW_FILTER",
   default=True)`. Keep the `=0` opt-out and the legacy no-filter path intact.
2. Update the docstring + `docs/dev/hda-certification-rowfilter-entry-2026-07-18.md`
   with the graduation-panel record (house style: measurement, not adjectives),
   and add the falsification/measurement note to
   `docs/dev/performance-plan.md` §6 if any hypothesis died along the way.
3. Regression tests for the new default:
   - default-ON is inert on a cleanly-certifying instance (byte-identical
     node_count/objective vs `DISCOPT_RELAX_ROW_FILTER=0`) — fast, not slow;
   - hda gets a finite tight bound (> −1e7 and ≤ opt) at *default* settings —
     mark `slow` (needs the Dropbox corpus instance; skip if absent, as the
     existing hda tests do);
   - opt-out restores the legacy path (candidate-A floor reproduced).

## 7. Contingencies

- **Panel shows a HARD regression** (an already-solving instance whose nodes
  *do* fail numerically mid-tree and now drift): tighten the trigger — fire
  only when the failure path would otherwise produce *no* certified verdict at
  all — and/or apply the max-combine guard from §5. Re-run. If it still fails,
  the flag stays opt-in: post the panel data on #671, state exactly what
  failed, and do NOT close the issue. Never trade a certificate for the hda
  bound.
- **hda no longer reaches ≈ −6.45e4** (drift since the branch's measurement):
  diagnose before proceeding — the entry-experiment reproducers are in
  `discopt_benchmarks/results/issue671/` (`export_hda_root_lp.py`,
  `rowfilter_entry_experiment.py`).
- **Cherry-pick too conflict-ridden:** hand-port the four hunks of §2 directly;
  the branch's test file usually applies clean.

## 8. Step 6 — PR and close-out

1. PR from `claude/issue-671-resolution-lm4bdh` titled
   `fix(cert): #671 failure-triggered row filter — graduate the tight hda dual bound`.
   Body: what was ported (and from where), the porting guardrail diff check,
   the full verification battery results, and the graduation panel numbers
   (both bars, HARD/SOFT tables). `Closes #671` **only if** §1 is fully met.
2. Issue close-out comment (§ "Working on an issue" step 5): what changed, how
   verified, explicit "can be closed" statement, plus disposition of the
   siblings — `DISCOPT_LP_ITERATIVE_REFINEMENT` and
   `DISCOPT_LP_FACTORIZATION_HARDENING` remain **opt-in research primitives**
   (the row filter supersedes the τ-sweep as hda's default lever; refinement
   remains the principled path once feral gains a rank-revealing LU — the
   "remaining principled follow-up" of
   `docs/dev/issue-671-gsw-iterative-refinement-2026-07-18.md`). No new issue
   is needed for those unless the owner asks.
3. Scope honesty in both PR and comment: this tightens hda's *certificate*;
   the residual gap (−6.47e4 → opt −5964.53) is the genuine root McCormick
   relaxation gap — an orthogonal branch-and-reduce/relaxation-strength
   effort, explicitly out of scope for #671.

## References

- #671 + comments (entry experiment CONFIRMED 2026-07-17; graduation HOLD
  2026-07-18), #662 (candidate A), #708 (entry experiment), #746 (merged
  first cut).
- Branches: `origin/claude/discopt-issue-671-46isrd` (failure-triggered
  refactor, stranded), `origin/feat/graduate-relax-row-filter-671` (owner's
  panel harness + HOLD data).
- `docs/dev/hda-certification-rowfilter-entry-2026-07-18.md`,
  `docs/dev/issue-671-gsw-iterative-refinement-2026-07-18.md`,
  `docs/dev/candidate-b-phase2-scaling-entry-2026-07-16.md`.
