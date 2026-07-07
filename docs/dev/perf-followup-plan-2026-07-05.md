# Performance follow-up plan — 2026-07-05 (executable task list for the B7–B12 bottlenecks)

**Status:** planned (no task started)
**Basis:** every task below is grounded in a measurement in
`docs/dev/bottleneck-profile-2026-07-05.md` (same branch/PR). Do not re-derive
those numbers; do re-run the per-task *entry experiment* before implementing —
if it no longer reproduces, the measurement wins: record the falsification in
the profile doc's §5 style and re-scope.
**Executor:** a fresh Claude (Opus) session with no prior context. Everything
needed is in this doc, the profile doc, and the canonical planning docs listed
in CLAUDE.md.
**Tracking:** file one GitHub issue per task on first pickup (title = the task
ID + name below); this doc is the specification, the issue is the workflow
handle.

---

## §0 Binding contract (do not reinterpret)

0.1 **Correctness before performance.** `incorrect_count ≤ 0` on every panel is
    a hard gate with zero slack. A task that can only hit its wall-clock target
    by weakening a validation, fallback, tolerance, or safety guard **loses**;
    stop and surface it. The A-tasks (correctness) merge before any F-task that
    touches the same layer.

0.2 **Fix the class, not the instance.** The named instances (fac2, nvs01,
    flay03m, st_e36, nvs09, hda, contvar, …) are *probes*. No change may key on
    an instance name, shape hash, or benchmark membership. Each task names its
    *class* and at least one out-of-panel witness instance from
    `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/` to confirm the
    fix generalizes (oracle: `minlplib.solu`).

0.3 **Verification regimes** (CLAUDE.md §Development-Philosophy 5, applied per
    task below):
    - *bound-neutral*: `node_count` and certified `objective` **exactly
      unchanged** on the certifying panel (`docs/dev/data/cert-baseline.jsonl`
      protocol; regenerate the comparison with the same seeds/flags). Any
      drift, including apparent improvement, means the change is wrong.
    - *heuristic-policy*: may change incumbent discovery timing (node counts
      may legitimately change) but must never touch the dual bound path. Gate:
      certified `objective` values unchanged, `incorrect_count = 0`, and a
      wall-clock regression probe (no instance on the 61-instance panel slows
      by >10 % — heuristics that stop finding incumbents show up here).
    - *bound-changing*: feature flag, default-OFF; differential bound test
      (new bound ≥ old bound AND ≤ true box optimum on fixed boxes) +
      feasible-point sampling (no valid point cut); flips default only after
      green on consecutive nightly runs.

0.4 **Per-PR gates** (all tasks): `pytest -m smoke`,
    `pytest -m slow python/tests/test_adversarial_recent_fixes.py`,
    `cargo test -p discopt-core` when Rust is touched, `ruff check` +
    `ruff format --check`, `mypy python/discopt/`. New behavior requires a
    regression test that fails before / passes after. One task per PR, task ID
    in the title. **Watch CI fully green before merging** (standing lesson:
    #490 merged on unstable → broken main).

0.5 **Measurement tooling.** Use
    `discopt_benchmarks/scripts/profile_instance.py` (added with the profile
    doc) for every before/after number: `clean` mode for totals, `--cprofile`
    for shares, `--second-solve` for tax, `--dump-stack-after` for overruns.
    Perf claims in PRs quote instance, mode, and machine.

0.6 **Falsification discipline.** If an entry experiment contradicts this
    plan, the profile doc, or `docs/dev/performance-plan.md` §6, the
    measurement wins. Record it (profile doc §5 house style) before
    re-scoping. Do not silently skip a task.

0.7 **Time-limit semantics.** `solve(time_limit=T)` is a *contract*: wall
    time ≤ T + a small constant (target: +10 % or +5 s, whichever is larger).
    F4 restores it; no other task may regress it.

---

## §1 Ordering and dependencies

```
A1 (nvs05 bound adoption)  ──┐
A2 (1e30 sentinel)         ──┤  correctness pre-flight: merge first
A3 (benchmark lb field)    ──┘  (A3 is script-only, trivial)
        │
F4 (root budget gate / time_limit contract)   — independent
F1 (LNS enumeration budget)                   — independent
F2 (simplex warm-restart stall guard)         — independent (Rust)
F3 (multilinear separator LP routing)         — after F2 lands (shares the
                                                 warm-simplex path; F2's cap
                                                 protects F3's new call site)
        │
F5 (even-power composite envelope)            — bound-changing, after A1
                                                 (needs trusted bound
                                                 reporting to evaluate)
F6 (node-NLP volume/engine, nvs05/tls2 class) — after F5 entry experiment
                                                 (F5 may collapse the node
                                                 count and change F6's value)
F7 (fixed-tax trims)                          — anytime, lowest priority
        │
V1 (full BARON head-to-head re-run + gate)    — after F1–F4 merged
```

A1/A2/A3, F1, F2, F4 are mutually independent — parallelize across
worktrees/sessions if desired, but each in its own PR.

---

## §2 Tasks

### A1 — nvs05-class: final `SolveResult.bound` is far below the tree's best bound  (P1, correctness)

- **Evidence** (profile §6): on nvs05 the `node_callback` trace ends with
  `best_bound = 5.32`, but `SolveResult.bound = 1.348` → reported gap 75.4 %
  where the tree achieved ~2.7 %. One of two things is true, and both matter:
  (a) the frontier/global bound is not adopted into the final result on this
  path (under-reporting — makes discopt look worse and mis-scores every
  benchmark), or (b) the callback's `best_bound` is *not* a certified global
  bound (over-reporting in the callback API — a soundness question).
- **Hypothesis:** (a) — a result-assembly path on `time_limit` exit takes a
  stale root/incumbent-era bound instead of the tree frontier minimum.
- **Entry experiment:** run nvs05 60 s with the callback trace
  (`profile_instance.py --trace`); dump both the final frontier
  (min over open nodes of node lower bounds) and `SolveResult.bound`.
  *Kill criterion:* if the frontier min at exit is actually 1.348 (i.e. the
  callback's 5.32 was never a valid global bound), this becomes a **callback
  API soundness bug** — re-file as (b), fix the callback, and add it to
  `docs/dev/correctness-issues.md` as a new C-item instead.
- **Implementation sketch:** find where `SolveResult.bound` is assembled on
  the `time_limit`/`feasible` exit path of `_solve_nlp_bb`
  (`python/discopt/solver.py`; grep `bound =` near the result construction
  and the `1e30` sentinel handling). The certified global bound at exit is
  `min(node.lower_bound for node in open_frontier)` (min sense), or the last
  completed-node bound when the frontier is empty. Ensure maximize-sense
  negation is handled (solver.py:853 pattern).
- **Class witness:** any time-limited spatial instance with open nodes at
  exit; check tls2 (bound trace 5.2 at TL) reports ≈5.2, not less.
- **Regime/gates:** correctness fix; the reported bound may only become
  *tighter and still valid* (never exceeds true optimum on the oracle panel).
  Add a regression test: solve a small instance with a tiny node budget,
  assert `result.bound == frontier_min` and `bound ≤ oracle optimum`.
  Full §0.4 gates + the 61-instance panel: `incorrect_count = 0` and no
  reported bound crossing its `minlplib.solu` oracle.
- **Acceptance:** nvs05@60 s reports gap ≈ (5.47−5.32)/5.47 ≈ 3 %, not 75 %;
  panel-wide reported gaps never loosen.

### A2 — `best_bound = 1e30` sentinel leaks into the public callback API  (P2, correctness/API)

- **Status: DONE** (branch `fix-a2-sentinel-callback-audit`). The full audit found
  the sentinel could still escape through `SolveResult.bound`/`.gap` and
  `root_bound`/`root_gap` (the callback surface was already fixed by A1/#498): the
  result-assembly paths (`solver.py` spatial ~7137, nlp_bb ~8380) set
  `bound_val = stats["global_lower_bound"]` and only guarded with `np.isfinite`,
  which passes the finite `1e30`. Fixed centrally in `SolveResult.__post_init__`
  (`modeling/core.py`) — the single chokepoint every construction path funnels
  through — by mapping any sentinel-magnitude `bound`/`root_bound` (either sense) to
  `None` and clearing its gap. Callback `gap` is now gated on `best_bound is not
  None` at all three `CallbackContext` sites. Bound-neutral: node_count and
  certified objective exactly unchanged on the nonconvex cert subset; the one
  intended output delta is the no-relaxation class now reporting `bound=None`
  instead of `1e30`. Tests: `test_a2_sentinel_bound_api.py` (6),
  `test_nonrigorous_fathom_decertifies_optimal.py` (5, incl. the previously-untested
  false-*optimal* decertification lock).
- **Evidence** (profile §6): on the no-relaxation class (hda, heatexch_gen3)
  `node_callback` receives `best_bound = 1e30` instead of "no bound".
- **Implementation sketch:** at the callback marshaling site, map the
  sentinel to `None` (preferred; matches `SolveResult.bound: Optional`) —
  and audit every consumer of `best_bound` inside the repo (gap computation,
  commentary, dashboard) for arithmetic on the sentinel. Do **not** change
  the internal sentinel representation — only the API boundary.
- **Regime/gates:** bound-neutral (representation-only). Regression test:
  callback on an hda-like model (a model whose relaxation omits rows — build
  synthetically with an unsupported operator, not named-instance) receives
  `None`. §0.4 gates.
- **Acceptance:** no `1e30` observable through any public API; gap fields
  `None` when no bound exists.

### A3 — benchmark worker reads a nonexistent `res.lower_bound`  (P3, benchmark-infra, trivial)

- **Evidence:** `global_opt_baron_vs_discopt.py`'s `DISCOPT_WORKER` does
  `lb = getattr(res, "lower_bound", None)` — `SolveResult` has `bound`
  (modeling/core.py:1186), so `lower_bound` was `null` for all 61 rows of the
  2026-07-05 report. `gap` was real; only the `lower_bound` column is dead.
- **Implementation sketch:** in the worker string, read `res.bound` (keep the
  JSON key `lower_bound` for report compatibility, or rename both — either,
  consistently). Also add a `bound-vs-oracle` violation check to the report
  verdicts now that the field is live (bound crossing the `.solu` dual bound
  = a VIOLATION row, same as an objective violation).
- **Regime/gates:** script-only; smoke gate. **Blocks V1** (V1's report should
  carry live bounds).
- **Acceptance:** a 2-instance `--instances` run emits non-null bounds.
- **Status: DONE** (branch `fix-a3-benchmark-bound-oracle`). Worker reads
  `getattr(res, "bound", None)` (the `.lower_bound` attr never existed; `None`
  now means "no dual bound" post-A2, not "read failed"). Added
  `bound_violates_oracle(bound, known, maximize)` and threaded the reported
  bound into `classify()` — a dual bound crossing the oracle (`bound > opt` for
  min, `bound < opt` for max, beyond tol) is now a `VIOLATION` even with a
  correct/absent incumbent, and the violation report distinguishes a bad bound
  from a bad incumbent. Tests in
  `discopt_benchmarks/tests/test_nl_solvers_verdict.py` (6 new: bound-crossing
  → VIOLATION both senses, fires without an incumbent, valid-LB-stays-GAP,
  None-bound backward-compat, the predicate, and the worker-attr regression).
  Script-only; ruff clean.

### F1 — Budget `local_branching`'s enumeration path  (B7; the #1 wall lever)

- **Evidence** (profile §1.1, §2): `_lns_k_schedule = (2, 5, 10)`
  (solver.py:4999, :7610; consumed at :6617/:8123 →
  `_jax/primal_heuristics.py:1426 local_branching`). With 12 binaries the
  ≤`max_binaries` enumeration branch issues `ΣC(12,r)` full POUNCE sub-NLPs
  — 79 at k=2, 1 586 at k=5 — ignoring its ≤2 s slice
  (`submip_time_limit=_lb_slice`) and never polling the deadline. fac2:
  1 665 calls = 85.8 % of 23.5 s; flay03m: 3 330 calls = 96.5 % of wall;
  runs even when the incumbent is already optimal. `rens=False`
  counterfactual (fac2 27.3 s, flay03m 91.7 s) proves the budget must live
  **inside `local_branching`**, not at call sites.
- **Hypothesis:** polling the slice/deadline inside the flip-enumeration loop
  (and skipping enumeration when its predicted cost exceeds the slice)
  removes ≥80 % of fac2/flay03m wall with zero effect on certified
  objectives.
- **Entry experiment** (run before implementing): monkeypatch a counter into
  `local_branching` to log (n_binaries, k, calls issued, wall) on fac2 and
  flay03m; confirm the 79/1 586 arithmetic and the ≥20 s attribution.
  *Kill criterion:* enumeration <50 % of wall on both ⇒ re-profile before
  proceeding.
- **Implementation sketch** (all inside `local_branching` +
  its enumeration helper):
  1. Accept/honor a hard `deadline` (absolute) in addition to the slice;
     poll every sub-NLP (they are ~14 ms — per-iteration polling is free).
  2. Before starting a k-round, predict cost = `ΣC(n_bin, r≤k)` × the
     *measured* mean sub-NLP time from the previous round (or 15 ms prior);
     if it exceeds the remaining slice, truncate k or dispatch to the
     existing `_local_branching_submip` (which already takes `time_limit`).
  3. Do not start any round when the incumbent already matches the node
     relaxation bound within `gap_tolerance` (nothing to improve).
  Keep the k-schedule policy itself; this task adds budget enforcement, not
  new heuristics.
- **Class witness:** any ≤12-binary MINLP with an early incumbent; pick one
  from `problems_small.txt` outside the 61-panel (e.g. a small `flay`/`clay`
  sibling or `st_` family member with binaries) and show the same call-count
  collapse.
- **Regime/gates:** heuristic-policy (§0.3). Panel run: certified objectives
  unchanged, `incorrect_count = 0`, no instance >10 % slower, and the
  primal-side check that incumbent *quality* at exit is not degraded on the
  panel (same or better objective at TL on the time-limited instances).
  Regression test: a synthetic 12-binary model with a deadline of 1 s
  asserts `local_branching` returns within 1.2 s and issues fewer sub-NLPs
  than the unbudgeted count.
- **Acceptance** (profile §7 arithmetic): fac2 23.5 s → ≤5 s;
  flay03m 46.5 s → ≤12 s (certifying); tls2 root −≥15 s. Measured with
  `profile_instance.py` clean mode, same machine, quoted in the PR.

### F2 — Warm dual-simplex stall guard  (B8; Rust)

- **Evidence** (profile §1.2, §2): nvs01 — of 41 `solve_lp_warm_std` calls
  totalling 17.5 s, two warm re-solves after `_separate_univariate_square`
  row-appends take **8.71 s / 8.70 s** on 117–118×25 LPs whose siblings take
  4–6 ms (79 % of wall). st_e36 — 2 398 warm calls @ 13.3 ms = 53 % of wall.
  A cold solve of the same LPs costs ~5 ms.
- **Hypothesis:** the warm dual simplex enters a degenerate cycling/stalling
  regime from a stale basis after row appends; an iteration cap with
  cold-restart fallback recovers the 4–6 ms cost with an identical optimum.
- **Entry experiment:** instrument (or log at `debug!`) iteration counts in
  `crates/discopt-core/src/lp/simplex/dual.rs::solve_lp_warm*`; re-run nvs01;
  confirm the two 8.7 s calls have pathological iteration counts (≫ the
  4–6 ms calls) rather than per-iteration cost. *Kill criterion:* iterations
  are normal and time is per-iteration (e.g. dense refactorization cost) ⇒
  different fix (factorization reuse), re-scope before writing the cap.
- **Implementation sketch** (`dual.rs`, possibly `primal.rs`):
  1. Add `max_iterations`/`max_time` to `SimplexOptions` (respect existing
     option plumbing; no new global).
  2. Default cap: `K × (rows+cols) + C` (pick K from the entry experiment's
     healthy-call distribution, e.g. p99 × 10 — record the number).
  3. On cap trip: discard the warm basis, cold-solve the same LP
     (`solve_lp` path), return that result; count trips in the LP stats.
  4. EXPAND/anti-degeneracy already exists (CLAUDE.md notes) — check first
     whether the stalled calls bypass it; if the root cause is a bypass,
     fix that instead of adding the cap (§0 band-aid rule), keeping the cap
     as a backstop.
- **Class witness:** st_e36's 2 398-call volume mean should drop; pick one
  out-of-panel spatial instance with separation-heavy traffic (from
  `problems_short.txt`) and show no regression.
- **Regime/gates:** **bound-neutral** (same LP optimum, different path):
  cert-baseline node_count/objective exactly unchanged. `cargo test -p
  discopt-core` + a Rust regression test that constructs a stall-prone
  warm-start (from the entry experiment's captured LP + basis — serialize
  it) and asserts the guarded path returns the same optimum as cold within
  the cap. §0.4 gates.
- **Acceptance:** nvs01 22.0 s → ≤5 s; no cert-panel drift; trip counter > 0
  on nvs01 and = 0 on the healthy majority.

### F3 — Route the multilinear separator's LP re-solves to the warm simplex  (B9; extends #484)

- **Evidence** (profile §1.3): nvs09 — 92.6 % of 60 s inside
  `_separate_multilinear` (`_jax/mccormick_lp.py:994`, called at :759) →
  `lp_pounce._solve_core` (the POUNCE LP IPM). #484's
  `DISCOPT_SEPARATION_LP_SIMPLEX` (default ON) covered edge-concave
  (`edge_concave.py:58`) and strong branching (`solver.py:1748`) only.
- **Hypothesis:** the same swap that gave 10–50× on the other separators
  applies; nvs09's separation cost collapses and its node throughput rises
  from ~2.6 nodes/s to the panel-normal range.
- **Entry experiment:** count LP solves and mean ms inside
  `_separate_multilinear` on nvs09 (one `--cprofile` run isolates it).
  Confirm ≥80 % of wall and that the LPs are re-solves of an augmented LP
  (warm-startable). *Kill criterion:* the LPs are structurally fresh each
  round (no basis to reuse) ⇒ the win is only cold-simplex vs IPM; re-check
  the expected gain before implementing (still likely worth it, but state
  the new number).
- **Implementation sketch:** mirror #484's mechanism at this call site: honor
  `DISCOPT_SEPARATION_LP_SIMPLEX` in `_separate_multilinear`'s LP
  construction, reuse the warm-basis plumbing the edge-concave path uses,
  and (as #484 did) recompute each cut's intercept to the exact validity
  boundary so cut *validity* never depends on which engine produced the
  duals. **Depends on F2** (this adds warm-simplex traffic; the stall guard
  should land first).
- **Class witness:** an out-of-panel multilinear-heavy instance
  (`minlplib_types.csv` → pick a small signomial/multilinear one).
- **Regime/gates:** same protocol as #484 — nominally bound-neutral with the
  documented degenerate-dual caveat; decide by the empirical neutrality run
  (cert-baseline exact-match; if a degenerate instance drifts, investigate
  before shipping — do not widen the tolerance). §0.4 gates + off-switch
  test (`DISCOPT_SEPARATION_LP_SIMPLEX=0` restores the old path).
- **Acceptance:** nvs09 separation share <20 % of wall; nodes/s ≥ 10× prior;
  bound at 60 s strictly better than −59.24 (it may now certify; fine).
- **Status: DONE with a falsified premise** (branch
  `perf-f3-multilinear-separator-simplex`, pounce 0.7.0, M4 Pro; profile §5
  item 8). The entry experiment falsified the uniform-10× assumption for this
  LP class: the `2^n`-column hull LP is cold-simplex-vs-IPM (no warm re-use,
  like #484), and on nvs09's 1024-column LPs the cold simplex is **bimodal** —
  ~81 % solve in ≈1 ms (≈100× vs POUNCE's ~150 ms), ~19 % stall to the
  100 000-pivot default. The kill-criterion "state the revised number" branch
  applies. Implemented soundly: mirror #484's env
  (`DISCOPT_SEPARATION_LP_SIMPLEX`, default ON) + intercept-recompute in
  `multilinear_separation._solve_envelope`, plus a size-derived cold-simplex
  pivot cap → POUNCE per-LP fallback (F2's warm-only guard does not cover the
  cold hull LP). Measured nvs09 @60 s: **nodes/s 2.62 → 3.65 (1.4×, not 10×)**;
  **bound −59.24 → −57.67 (strictly tighter, valid vs the −43.134 oracle)**;
  separation share stays ~89 % because 128 wide-LP POUNCE fallbacks (44 s) are
  IPM-favourable and irreducible by routing (a POUNCE-engine matter, cf. F6 /
  jkitchin/pounce#182). The `<20 %`/`≥10×` targets are **not met**; the class
  win (narrow hull LPs, ~81 % here → ~100× per LP) is real and sound. Cert
  panel provably byte-identical (0 multilinear-separator calls at 60 s on all
  41 certifying instances). Off-switch + soundness + fallback locked by
  `python/tests/test_f3_multilinear_separation_lp_backend.py` (7 tests).

### F4 — Budget-gate the root heuristic NLP/compile phase  (B10 + §4; restores the `time_limit` contract)

- **Evidence** (profile §4): contvar returns after **221 s** against
  `time_limit=60` — all stack dumps inside `feasibility_pump`
  (primal_heuristics) → POUNCE Hessian callback →
  `sparse_hessian.sparse_hess_values:162` → one uninterruptible first-time
  **XLA compile** (~150 s+). heatexch_gen3 81.5 s, same class. hda's root:
  42 s of 64 s (15.1 s one-call Rust presolve + 21 multistart NLPs
  @600–780 ms + ~10 s re-tracing).
- **Hypothesis:** gating *entry* into compile-triggering root work by
  remaining budget (compiles cannot be interrupted) restores the contract
  with no dual-bound effect, because all gated work is primal-heuristic.
- **Entry experiment:** re-run contvar with `--dump-stack-after 70`; confirm
  the overrun stack. Measure hda's root multistart count/costs. *Kill
  criterion:* overrun reproduces somewhere other than a heuristic-NLP first
  compile ⇒ that site needs its own gate; enumerate before coding.
- **Implementation sketch:**
  1. Thread the absolute deadline into the root-heuristic phase (it exists in
     the B&B loop; the root phase predates it on this path).
  2. Before a heuristic NLP whose evaluator family is not yet compiled
     (query the JAX compile cache — the evaluator knows), require
     `remaining_budget ≥ compile_estimate(model_size)`; the estimate comes
     from a measured curve (fit on ~5 sizes with `profile_instance.py`;
     record the fit in the PR). Otherwise skip the heuristic or use its
     Hessian-free/first-order path if one exists (check
     `nlp_pounce.solve_nlp` options) — skipping a *primal heuristic* is
     always sound.
  3. Cap root multistart count by remaining time (measured per-start cost ×
     starts ≤ a root-phase fraction, e.g. 50 % of budget — a policy constant,
     documented, not tuned per instance).
  4. Poll the deadline between multistarts and between heuristic families
     (feasibility_pump, RENS, diving).
- **Class witness:** any large flowsheet-class `.nl` from the full corpus
  with `time_limit=30`; assert wall ≤ 33 s + report which phases were
  skipped.
- **Regime/gates:** heuristic-policy. Panel: certified objectives unchanged,
  `incorrect_count = 0`; time-limited instances must not lose incumbent
  quality (compare objective-at-TL vs baseline). Regression test: a model
  with an artificially expensive compile (large sparse Hessian) +
  `time_limit=5` returns within the §0.7 envelope. §0.4 gates.
- **Acceptance:** contvar wall ≤ 66 s; heatexch_gen3 ≤ 66 s; hda root phase
  ≤ 50 % of budget; §0.7 holds on the full 61-instance panel.
- **Status: DONE, two premises falsified** (branch `perf-f4-root-budget-gate`,
  pounce 0.7.0, M-series arm64; profile §5 item 9). (1) The 221 s contvar
  overrun does not reproduce on this build — contvar is 60.7 s, already inside
  the envelope; the reproducing overruns are heatexch_gen3 (80.7 s) and hda
  (64.2 s). (2) Compile time is **not** a function of cheap model size
  (`log(compile)` vs `n_vars`: R² = 0.002; contvar n=296 → 186 s vs hda n=722 →
  2.5 s; noisy run-to-run), so the estimate is a conservative *floor* (dense
  0.5 s; uncompiled sparse 15 s risk-headroom), not a fitted curve. The overrun
  on this build is repeated **post-deadline heuristic-NLP launches** (each
  overrunning its own `max_wall_time`), fixed by gating *entry* into every root
  heuristic NLP on the remaining budget + threading the absolute deadline into
  the looping heuristics (`diving`/`rins` poll each sub-NLP; multistart caps
  extra starts by observed per-start cost). Result: **heatexch_gen3
  80.7 → 60.9 s**, contvar 60.7 s, hda 64.2 → 64.1 s; **full panel 61/61 within
  §0.7 at TL=30 with 0 objective changes / 0 lost incumbents** vs gate-off. hda
  root is 42 s (70 % of a 60 s budget) — the ≤50 % target is **not** met, but
  hda's *wall* respects the contract; its root is Rust presolve + re-tracing,
  not the gated heuristics. **Kill-criterion hit:** the out-of-panel flowsheet
  super3t overruns via a *second* site (`term_classifier._compute_var_offset`
  relaxation build, 0 `solve_nlp` calls) — outside F4's scope; filed as F4
  follow-up #507. Off switch: `DISCOPT_ROOT_BUDGET_GATE=0`.

### F5 — Even-power composite envelopes for the pinning class  (B11; bound-changing)

- **Evidence** (profile §3): st_e36's root bound (−304.5000003055) is
  bit-identical for x1 ∈ [15,25], [20,25], [24,25] — the under-envelope of
  its nested `pow(·,2)` composites with negative coefficients is independent
  of the integer's box; only fixing lifts it (x1=22 → certifies in 5 nodes;
  897 unfixed nodes go nowhere). nvs05/nvs09 root gaps 87.7 %/69 % are the
  same loose-product/power family (live but slow).
- **Precondition (binding):** check `docs/design/relaxation-catalog.md` first
  — do not rebuild anything it lists as done. C-34/FR-1 (even powers p≥4
  straddle bounds) and C-17 (alphaBB rigorous-α) are adjacent; this task is
  about *composite* `±(inner expr)²` envelopes that contract with the
  participating variables' boxes, not the bare-power case.
- **Hypothesis:** a secant/tangent envelope for `c·(affine+quadratic)²`
  (c<0 concave side) that is a function of the *composite argument's*
  interval — propagated from member boxes — lifts st_e36's root bound
  strictly monotonically with box shrinking, collapsing its tree.
- **Entry experiment:** extract st_e36's pinning term (2 vars — print the
  DAG); compute by hand/numpy the convex under-envelope over x1 ∈ [15,25]
  vs [24,25]; verify they *differ* (i.e. the information exists and the
  current relaxation discards it). *Kill criterion:* the true envelope is
  also interval-independent on this structure ⇒ the defect is elsewhere
  (e.g. the argument's interval never tightens in FBBT); re-diagnose before
  building anything.
- **Implementation sketch:** in the relaxation compiler
  (`python/discopt/_jax/mccormick.py::relax_pow` at :203 and the composite
  path that feeds it), thread the composite argument's interval (already
  computed by interval propagation) into the even-power envelope instead of
  falling back to the variable-box-independent form. Flag:
  `DISCOPT_COMPOSITE_POW_ENVELOPE` (default OFF until nightly-green).
- **Class witness:** grep the full corpus for `o5`/pow-of-expression
  structures (the `.nl` opcode scan tooling from C-5 work) — pick 2
  out-of-panel instances; differential bound test on both.
- **Regime/gates:** **bound-changing** (§0.3): differential bound test on
  fixed boxes (new ≥ old, ≤ true box optimum via dense sampling) +
  feasible-point sampling (no valid point cut) + the flag discipline.
  Panel with flag ON in a manual run: `incorrect_count = 0`.
- **Acceptance:** st_e36 root bound strictly improves when x1's box shrinks;
  st_e36 certifies ≪897 nodes (profile predicts ~5–10). Default flip only
  after two green nightlies.
- **Status: KILLED at the entry experiment — premise falsified, nothing
  shipped** (branch `perf-f5-composite-power-envelope`, pounce 0.7.0,
  M4 Pro; profile §5 item 10). The composite even-power envelope is **not**
  st_e36's defect, on three grounds: (a) the pinned root bound
  `−304.5000003055` *is* the objective's own true box-minimum (a plain
  polynomial `2x0²+0.008x1³−3.2x0x1−2x1` whose argmin is the shared corner
  `(5.5,25)`; dense-sampled box-min = −304.5000 for all of [15,25]/[20,25]/
  [24,25]) — the relaxation is already essentially exact for the objective,
  so there is no envelope slack to recover; (b) every `(inner)²` in st_e36
  (all in constraint C0) has coefficient **+1** (convex), whose true convex
  under-envelope *equals the function* (scipy convex-hull LP: gap 0 on both
  boxes) — and `relax_pow`'s even branch already threads the argument
  interval and returns exactly that, so the "variable-box-independent
  fallback" the task posited does not exist; (c) st_e36 makes **0**
  `relax_pow` calls (the LP relaxer lifts squares to auxes) and, decisively,
  `f1 = x0²−6x0−11+0.8x1` spans 0 on **every** box (`[−8,6.25]`/`[−4,6.25]`/
  `[−0.8,6.25]`), so `C0 = f1·(positive)` relaxes to an interval spanning 0
  regardless of the square factors' tightness — no even-power envelope can
  cut the corner. The bound *does* move with the box, but via the objective/
  f1 quadratic-bilinear structure (shrink x0→[5,5.5] lifts −304.5→−246.0;
  fix x1=22→−278.9), i.e. a **branch-and-reduce / product-relaxation**
  lever, not an envelope one. Per CLAUDE.md §3 (no dead flags) the
  `DISCOPT_COMPOSITE_POW_ENVELOPE` flag was **not** added and no relaxation
  code changed. Re-scope st_e36 under the branch-and-reduce roadmap
  (cert-gap-plan): tighten C0's product-of-factors / f1-crossing relaxation.

### F6 — Node-NLP volume/engine on the nvs05/tls2 class  (B12; paired with upstream)

- **Evidence** (profile §1.6): nvs05 — 834 node NLPs @64 ms = 91 % of wall
  (retry policy adds ~1.6 NLP/node); tls2 — 1 497 @29 ms = 70 %. The engine
  gap itself is filed upstream (jkitchin/pounce#182; Phase-D A/B: median
  7–11× vs Ipopt).
- **Scope here (discopt-side only):** (1) audit the retry-from-alternative-
  start policy — measure its incumbent hit-rate on the panel; if retries
  find <5 % of incumbents, cap retries by a deadline-aware policy;
  (2) re-measure after F5: if the root-gap work collapses node counts, this
  task may be moot — **run the F5-dependent entry experiment before any
  code**. *Kill criterion:* retry hit-rate is material (≥15 % of incumbents)
  ⇒ leave the policy alone; only the upstream engine work remains, and this
  task closes as "blocked-upstream".
- **Regime/gates:** heuristic-policy; §0.4 gates + panel neutrality on
  certified objectives.
- **Acceptance:** stated after the entry experiment (record the numbers in
  the PR); do not pre-commit a wall target for a task whose scope is
  conditional.
- **Status: RE-SCOPED then KILLED at the entry experiment — nothing shipped**
  (branch `perf-f6-tree-warmstart-nlp`, pounce 0.7.0 RELEASE wheel, M-series
  arm64; profile §5 item 11). F6 was re-scoped away from the "POUNCE engine
  gap" (proven a *debug-build artifact* — a release POUNCE is ~1.1× Ipopt with
  fewer iters, pounce#182 comment 4889424028) to the maintainer-endorsed
  lever: **warm-start incumbent NLPs across the B&B tree** (thread the parent
  node's converged primal + duals + μ into the child NLP via POUNCE's
  `Problem.solve(x0, lagrange=, zl=, zu=)` + `warm_start_init_point`/`mu_init`).
  Entry experiment (real captured parent→child NLP pairs from live 60 s nvs05
  and tls2 B&B runs, cold vs warm re-solve): warm-start is **incumbent-quality
  safe** (every replay reached the identical KKT point / objective) but its
  **iteration count fails the kill criterion on the class** — **tls2 uniformly
  worse (8/8 pairs, median cold/warm 0.58×, i.e. warm needs ~1.7× more iters);
  nvs05 helps only on ≤2-bound-diff pairs (2.29×) and flips to a net loss on
  realistic ≤4-bound pairs**, and the only positive recipe (`x + duals + μ`) is
  exactly the one that is catastrophic on tls2 (dropping μ or the duals makes
  nvs05 merely neutral). Per the plan's "<1.3× median ⇒ warm-start is not the
  lever" kill criterion the task stops. Retry sub-audit: alternative-start
  retries *rescue* only 2.4% (nvs05) / 0.2% (tls2) of nodes — below the 15%
  "material" and 5% "cap it" thresholds — so the retry policy is left unchanged
  (F4 already deadline-polls between retries). Per CLAUDE.md §3 (no dead flags)
  **no warm-start API, feature flag, or `NLPResult.mu` field was added and no
  solver code changed.** The nvs05/tls2 stall is a weak-bound / branch-and-
  reduce problem (as F5 also concluded), not a per-node NLP speed problem —
  re-scope under the cert-gap-plan branch-and-reduce roadmap. pounce#182 closes
  as a resolved build artifact.

### F7 — Fixed-tax trims  (lowest priority)

- **Evidence** (profile §1.7): `import discopt` 0.10–0.14 s; lazy imports
  inside first `solve()` 0.4–0.6 s; JIT warmup ≈0.5 s. Only matters for the
  sub-5 s class (m3, nvs13, ex1224, nvs08) — and net of tax m3 is still
  ~2.7 s vs BARON 0.04 s, so this is polish, not strategy.
- **Sketch:** hoist the lazy imports to module import time behind a
  `DISCOPT_EAGER_IMPORTS=1` env (opt-in for benchmark/CLI runs); persistent
  JAX compilation cache (`jax.config.compilation_cache_dir`) for the warmup
  family. Measure with `--second-solve`.
- **Regime/gates:** bound-neutral; §0.4 gates.
- **Acceptance:** first-solve tax ≤0.3 s with the env set; zero panel drift.

### V1 — Re-run the BARON head-to-head and gate the outcome  (after F1–F4)

- **Procedure:** rebuild `main`, re-run
  `discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py --time-limit 60`
  (A3 must be merged so bounds are live). Same machine class as the
  2026-07-05 baseline; note load conditions.
- **Gates (hard):** `incorrect_count = 0` / zero violations (as before);
  no instance that certified on 2026-07-05 loses certification; §0.7
  time-limit contract panel-wide.
- **Targets (measured expectations, not promises — from §7 arithmetic):**
  discopt total wall 23.1 min → **≤ ~12 min**; "hit TL" count 18 → ≤ 12;
  fac2/nvs01/flay03m/nvs09 leave the slow list; contvar/heatexch_gen3
  respect the budget. Compare per-instance vs the 2026-07-05 JSON and
  record the delta table in `docs/dev/` (this doc's §3 as a results
  appendix, or a dated results note).
- **If a target is missed:** that is data, not failure — record which
  bottleneck's share did not move (per-task acceptance already measured
  them individually) and re-profile the residual before adding new tasks.

---

## §3 What NOT to do (measured dead ends — binding per §0.6)

- **OBBT tuning** for these classes: ≤4 % of wall on all 11 profiled
  instances (profile §5.3). The 07-02 OBBT finding applies to its spatial
  panel only.
- **More CSE/DAG-shrinking for Class A**: the DAG walk is not where the
  seconds are (except hda's churn note, profile §6 — secondary to B10).
- **Blaming POUNCE per-solve latency on Class A / flay03m / st_e36 / nvs09**:
  falsified (profile §1, verdict table). The "POUNCE engine gap" (Phase-D
  7–11× vs Ipopt) was itself falsified as a **debug-build artifact** — a
  release POUNCE is ~1.1× Ipopt with fewer iters (pounce#182 comment
  4889424028; profile §5 item 11). There is no per-solve engine gap to close,
  upstream or down. **Warm-starting node NLPs across the tree (F6)** was the
  endorsed alternative and was **killed at its entry experiment** (worse
  iteration count on tls2, fragile on nvs05); nvs05/tls2 are a weak-bound /
  branch-and-reduce problem, not a node-NLP-speed problem.
- **`rens=False` or disabling heuristics as a "fix" for B7**: measured to
  relocate the cost, not remove it (fac2 27.3 s, flay03m 91.7 s).
- **Polling inside XLA compiles** (F4): not possible; gate entry instead.
