# Skeptical architecture review — 2026-07-28

**Prompt for this review** (owner): *"discopt has a lot of built-in functionality, but
it is a bit of an unwired mess from months of iterative bug fixes, with a complicated
flag system, multiple solvers that combine paths through Python and Rust, presolves,
etc. I am worried about solutions that solve specific problems rather than general
approaches."*

**Method.** Five independent static audits over `main` (flag system, solve dispatch,
presolve/tightening layers, instance-specific tuning, unwired functionality), plus
full-history git archaeology (1,669 commits, 2026-02 → 2026-07) and hand verification
of the sharpest claims. Everything below carries file:line evidence; nothing is a
runtime measurement unless marked as quoted from an in-repo report.

---

## 0. Verdict, claim by claim

| Owner's claim | Verdict | One-line basis |
|---|---|---|
| "Solutions that solve specific problems, not general approaches" | **Wrong** — the strongest negative result of the review | Zero instance-keyed logic anywhere in non-test source (AST-verified); instance-motivated fixes were consistently generalized to structural classes |
| "Complicated flag system" | **Right**, with an important nuance | 104 live env flags, **7 mutually incompatible boolean parse idioms**, 56% never migrated into the `SolverTuning` registry, no user-facing enumeration, zero interaction coverage — but the *graduation discipline* around the flags is real and enforced |
| "Unwired mess from months of iterative bug fixes" | **Half right** | The *core solve path* is an accreted monolith (a 7,459-line `solve_model`, 10 B&B loops, ~30 tightening mechanisms, 6 FBBTs) with several genuine wiring defects; but the *periphery* is cleaner than the claim implies — dead code surface is small, duplication is mostly deliberate and documented |
| "Multiple solvers combining paths through Python and Rust" | **Right** | 19 terminal solve paths; per-node tightening is **inconsistent between the Python B&B paths and the Rust native kernel** — flipping one flag silently swaps three node mechanisms for one different one |
| "Works, but not uniformly, and not as fast as SOTA" | **Confirmed — by the repo's own documents** | `docs/dev/sota-parity-analysis-2026-07-27.md`: 7.9× BARON / 2.9× SCIP geomean at correctness parity; performance is bimodal (43 optimal / 8 feasible-only / 10 no-incumbent on the 61-instance panel); the doc's own words: *"capabilities exist but the default path does not reach them… SCIP/BARON do not have a 'flag off' state — routing IS the product"* |

The one-sentence synthesis: **the codebase does not have an overfitting problem; it has
a composition problem.** The individual mechanisms are general, sound, and unusually
well-evidenced. What has accreted is the *plumbing between them* — dispatch, flags,
and tightening orchestration — and that is exactly where the SOTA gap lives.

---

## 1. Where the skepticism is wrong (and worth retiring)

### 1.1 No instance-specific hacks — verified hard

An AST-level scan (separating docstrings/comments from code) of every non-test module
in `python/discopt` and `crates` found **zero** string comparisons, dicts, or branches
keyed on a MINLPLib instance name, and **zero** hardcoded oracle objective values. The
~200 instance-name mentions are all in comments, docstrings, and benchmark corpus
lists. 73 of 1,468 non-merge commits name an instance; sampled diffs
(`4fd2d0ef` kall_circles false-optimal, `ed7eb5ef` nvs22 objective floor,
`0844434f` gear4 integer-ratio partition) all use the instance as a *reproducer* and
ship a mechanism quantified over a structural class, gated default-OFF, and graduated
on a panel. The CLAUDE.md §2 discipline was actually followed, not merely written down.

### 1.2 The evidence culture is real

- Tombstoned flags with recorded kill measurements (`solver_tuning.py:894-902`
  `NODE_REDUCE` removed as "sound but not helpful"; `:451-457` two more).
- `docs/dev/performance-roadmap.md` §4 lists six provably-correct-but-useless changes
  killed before shipping; `performance-plan.md` records three falsified cost-model
  claims; `sota-parity-analysis-2026-07-27.md` §G-G.1 retracts three of its *own*
  earlier claims.
- Zero TODO/FIXME/HACK markers repo-wide — debt is carried as `NotImplementedError`
  (71 instances) and issue-referenced justification comments instead, which is a
  policy choice, not concealment (ruff's TD/FIX rules are simply not enabled).

This matters for the review's framing: the repo *knows* most of what is written below.
The failure mode is not ignorance; it is that consolidation work (16 `refactor:`
commits vs 294 `fix:` + 267 `feat:`) has never been prioritized.

---

## 2. Where the skepticism is right

### 2.1 The dispatcher is one function, not an architecture

- `solve_model` (`solver.py:4316-11774`) is **7,459 lines — 41% of the file** — with
  555 `if` statements, 63 nested early returns, max nesting depth 11, and ~65
  `# --- banner ---` phase sections whose naming schemes span at least five eras
  (`Phase E`, `B2-FIX (task #89)`, `C-1`, `TX1`, `Lever A`, `PF1`, `M6/M9 of #51`).
- There is **no dispatch table, registry, or strategy object**. The routing policy is
  the implicit ordering of ~20 sequential gates; adding a path means inserting an
  `if` at the right index. Three soundness fall-throughs retrofitted for #740/#748
  (`solver.py:6638,6697,6704`) turn "fast engine" branches into silent redirects into
  the 2,470-line generic loop.
- The file grew 43× (415 → 17,987 lines) across 358 commits — one in five commits in
  the repo's history touches it.
- The problem classifier itself (`_jax/problem_classifier.py:62-115`, 54 lines, 10
  classes) is clean; its *consumer* re-qualifies the result through seven additional
  ad-hoc predicates before selecting anything.

### 2.2 Ten B&B loops

Seven Python loops (inline spatial in `solve_model` at `solver.py:8677`; `_solve_nlp_bb`;
`_solve_milp_bb`; `_solve_miqp_bb`; `lp_spatial_bb.py:380`; `gp/__init__.py:905`;
`signomial_global.py:1141`) plus three complete Rust implementations
(`milp_driver.rs`, `spatial_tree.rs`, `convex_kernel.rs`). Three of the Python loops
(`lp_spatial_bb`, `gp_minlp`, `signomial_global`) reimplement node selection and
branching with their own `heapq` despite `PyTreeManager` existing for exactly that.
Every new loop is a new place where a certificate invariant must be re-proven.

### 2.3 The tightening stack is the clearest accretion casualty

~30 distinct mechanisms across 5 layers, including **six FBBT implementations** and
**five reduced-cost-fixing implementations**, with no single sequencer —
`run_root_presolve` orchestrates only the Rust kernel set; everything else is
hand-ordered inline at ~20 call sites. Specific defects, in priority order:

1. **Per-node double FBBT** (verified): `_tighten_node_bounds_with_status`
   (`solver.py:2209`) — a pure-Python O(m·n²) Jacobian-sampled linear-row FBBT — runs
   at **every node on both Python B&B paths** (`solver.py:8836`, `12625`) immediately
   before the Rust `in_tree_presolve` kernel (`solver.py:8885`, `12664`) that computes
   a strict superset of its inferences from the exact DAG. No comment explains why
   both run. Given the repo's own finding that the per-node cost gap vs SCIP is
   50–500×, this is a direct candidate for deletion (as a bound-neutral change with
   the exact-node-count gate of CLAUDE.md §5).
2. **`cascade_aux` mis-wiring** (verified): documented as graduated default-ON
   (`obbt.py:1884-1895`), but the env flag is resolved and passed at exactly **one of
   six** `obbt_tighten_root` call sites (`root_reduce.py:393-404`); the root, per-node,
   incumbent-improvement, `lp_spatial_bb`, and `disjunctive_config_bound` sites all
   take the function default `False`. A graduated feature is off at 5 of 6 sites.
3. **Rust orchestrator model rewrites are dropped**: `propagate_bounds_to_model`
   copies bounds only (`_root_presolve.py:206-214` documents this as a known gap), so
   `coefficient_strengthening`, `simplify` big-M work, and `redundancy` row removal
   do work with no downstream effect except via bounds — and a second, stronger
   Python coefficient-tightener exists *because* of this leak.
4. **~2,770 lines of never-enabled Rust presolve** (`fbbt_fp.rs` — which its own
   header says supersedes the wired-in `fbbt.rs` — `scaling.rs`, `duality.rs`,
   `reduction_constraints.rs`, `symmetry.rs`) plus 623 lines of outright-dead
   `presolve/obbt.rs` (zero callers), plus the entire unreachable A3 Python-pass
   layer (~700 lines: `run_orchestrated_presolve` and the four passes only it can
   construct).
5. **Stale docs contradicting code**: `solver_tuning.py:894-903` claims
   `node_reduce.py` was removed; the module exists, is imported at `solver.py:8498`,
   and its header still advertises the deleted flag. `flag-graduation-protocol.md:29`
   lists `DISCOPT_NODE_REDUCE` as a live parked flag.

### 2.4 The flag system: good policy, no infrastructure

- **104 live flags** (of 121 grep-visible tokens; 17 are never read from env). 46 are
  registered in `SolverTuning` — typed, validated, thread-safe, zero dual-reads —
  and **57 are inline `os.environ.get` calls** (27 in `solver.py` alone), plus 8 in
  Rust with no registry, plus 12 daemon flags whose names are built by f-string and
  are invisible to grep.
- **Seven incompatible boolean parse idioms** (verified). The registry's own helper
  (`solver_tuning.py:33`) is `raw != "0"`, so **`DISCOPT_RLT=false` turns RLT ON**
  for the 43 registry flags, while `DISCOPT_CONVEX_KERNEL=false` correctly turns it
  off — and `DISCOPT_CONVEX_KERNEL=off` turns it *on* (`_convex_kernel.py:652`).
  Empty string is true under three idioms and false under four. `"2"` enables under
  five idioms and silently *disables* under two (`DISCOPT_SGO=2` is off). Two Rust
  flags use presence tests, so **`DISCOPT_PROFILE=0` enables profiling** and
  `DISCOPT_DISABLE_CSE=0` disables CSE — the opposite of the repo's own documented
  `=0` escape-hatch convention.
- **No enumeration anywhere**: README has zero `DISCOPT_` mentions; no reference doc
  exists; the graduation docs are process narratives, not an index.
- **No interaction coverage, by design**: the graduation protocol
  (`flag-graduation-protocol.md:38-45`) deliberately tests each flag in isolation
  against all-defaults after the N=20 bundled pilot contaminated attribution. That
  was the right call for attribution, but it means the shipped configuration space
  (2^~75 for the graduation-track flags) has exactly two tested points per flag plus
  one `all` bundle. Four-flag cross-language chains exist
  (`CONVEX_KERNEL` → `CVX_NATIVELP` → `CVX_DOMINATED_COLS` → `CONVEX_KERNEL_BUDGET`)
  with no shared declaration.
- The *policy* around flags — default-OFF until a differential panel passes, `=0`
  opt-out retained after graduation, tombstones with measurements — is coherent and
  followed. The mess is the *mechanics*, not the governance.

### 2.5 Python/Rust hybrid: mostly deliberate, two real risks

Most duplication is documented and intentional (`mccormick_patch.rs` declares itself
a byte-for-byte port with a bound-neutrality contract; `decomposition/graph/kernels.py`
is an explicit pure-Python reference). The two genuine risks:

1. **Per-node tightening is inconsistent across kernels.** The Python spatial path
   runs Jacobian FBBT + Rust `in_tree_presolve` + per-node OBBT (n≤100); the Rust
   native kernel (default-ON since 2026-07-27) runs only `spatial_propagate`, with
   its OBBT sweep default-OFF; the MILP driver runs reduced-cost fixing but node
   propagation default-false. `DISCOPT_NATIVE_SPATIAL_KERNEL=1` therefore changes
   *which tightenings a node sees*, not just which language runs them. Nothing
   asserts these stacks reach equivalent fixed points.
2. **The Python fallback is load-bearing, not legacy.** The native kernel's producer
   declines feature-unsafe models (`_native_kernel_feature_safe`, `solver.py:571`),
   so the 2,470-line inline Python loop is the permanent path of record for the hard
   tail — which is precisely the tail where the 50–500× per-node interpreter cost
   bites. The two-kernel strategy structurally guarantees the slowest instances get
   the slowest engine.

### 2.6 Non-uniform performance is structural, and the repo has already diagnosed it

Quoting the repo's own `sota-parity-analysis-2026-07-27.md`: 48/50 ok at 60 s but
geomean 0.540 s vs BARON 0.068 s; per-node profile 61% JAX / 39% Python / ~0% Rust;
`nvs05` at 20.5 nodes/s vs BARON's 1,874; six instances SCIP solves in <6 s where
discopt returns nothing; `clay0303hfsg` at 34.8 s on the default path while the
flag-gated convex kernel does it in ~7 s. The bimodal 61-instance panel
(43 optimal / 8 feasible-only / 10 nothing) is the "works, but not uniformly"
observation, measured. The doc's §G-C names the cause in the same words as this
review: routing. The best capabilities (`CONVEX_KERNEL`, the OBBT sweep, `SGO`) are
parked default-OFF, and the default path cannot reach them.

### 2.7 Feature islands (the "lot of built-in functionality")

Only `gp/` is on the default solve path. Six subpackages (`mo`, `ro`, `stochastic`,
`dae`, `interfaces`, `bilevel` — plus `mpec.py`, consumed only by `bilevel/`) have
**zero inbound imports** from the rest of the package. All are tested and documented
(so: islands, not dead code), but they carry 41 of the 71 `NotImplementedError`s and
the only "Phase 0"/"v0" self-labels (`stochastic/multistage.py:47` ships a headline
algorithm as a raise: *"cuts are invalid; solve via build_extensive_form"*). These are
a maintenance liability out of proportion to their integration: they enlarge the
surface the correctness gates do *not* cover while sharing the modeling core whose
churn (116 commits to `modeling/core.py`) can break them silently.

### 2.8 One systemic overfitting risk survives — but it is about *evaluation*, not code

`global50` is both the iteration set and the graduation gate set
(`benchmarks.toml:82-95`; every `[gates.cert*]` resolves to it). No gate names an
instance and no code keys on one, but a frozen panel that drives development *and*
grades it is textbook selection pressure, and it quietly discounts every panel number
cited in `solver_tuning.py` docstrings. One flag (`ils_solve_cap`,
`solver_tuning.py:741`) was validated on a held-out sample — the discipline exists
but is not applied to the panel itself. The ~4,800-instance Dropbox snapshot makes a
rotating held-out draw cheap. Separately: ~14 magic-number gates
(`_AUTO_CUTS_MAX_VARS = 40` at `solver.py:3692` with no measurement; three unrelated
`50`s; an uncommented duplicated `15` in `partition_selection.py:280,285`) are sound
(perf-only) but unfitted — asserted, not derived.

---

## 3. What I would actually do (prioritized)

Ordered by (certificate risk × leverage) ÷ effort. Items 1–4 are bound-neutral by
construction and gated by the exact-node-count check; none is a rewrite.

1. **One `_env_bool` with one truth table** (accept `1/true/yes/on` and `0/false/no/off`,
   reject everything else *loudly*), migrated flag-by-flag; fix the two Rust
   presence-test flags to honor `=0`. Small, mechanical, and it closes the
   "user sets `DISCOPT_RLT=false`, gets RLT, publishes a wrong benchmark" hole.
2. **Generate a flag reference** from `SolverTuning` + a registry shim for the 57
   inline flags (name, default, status: graduated/parked/permanent, owner issue).
   The registry docstrings already contain the content; it just isn't enumerable.
3. **Fix the verified wiring defects**: `cascade_aux` at the five unpassed sites
   (then re-run its graduation panel — it graduated on evidence from one site);
   delete or justify the per-node double FBBT (`solver.py:2209`); reconcile the
   `node_reduce` docstring/protocol-doc contradictions.
4. **Delete the dead layer**: `presolve/obbt.rs`, the never-enabled Rust passes (or
   wire `fbbt_fp` in place of `fbbt` if its own supersession claim is true — that is
   a measurable experiment), the A3 handshake, the orphaned Rust bindings (8 of ~28
   exports), and the 11 test-only `_jax/` modules. ~5,000 lines at near-zero risk.
5. **Turn the routing table explicit.** Not a rewrite of `solve_model` — extract the
   ~20 sequential gates into an ordered list of (predicate, path, gate-reason)
   entries that is *introspectable* (`model.explain_routing()`), so the ordering
   stops being implicit in `if`-position. This is also the precondition for the
   thing the parity doc says matters most: graduating the parked kernels
   (`CONVEX_KERNEL` first — its blockers #779/#798 are verification wiring, not math)
   so the default path reaches the fast engines.
6. **Split the evaluation set**: keep `global50` for iteration, add a held-out
   rotating panel drawn from the Dropbox snapshot as the graduation gate. Cheap
   insurance for every future flag flip.
7. **Freeze the islands explicitly.** Mark `mo/ro/stochastic/bilevel/dae/interfaces`
   as tier-2 (no correctness-gate coverage, API-stability not promised) in docs, or
   promote them deliberately. The worst outcome is the current implicit one, where
   README-level parity is implied between the certified core and a `Phase 0` island
   that raises on its headline path.

### What I would explicitly *not* do

- **Not rewrite the solver around the Rust kernels.** The Python fallback is the
  soundness net for the feature-unsafe tail, and the repo's own falsification record
  (six sound-but-useless changes; the `#861` patch-tape 1.0×) argues against big
  speculative restructuring. Extend `_native_kernel_feature_safe` coverage
  incrementally instead.
- **Not purge the magic constants wholesale.** They are perf-gates, not soundness
  gates. Replace them opportunistically when a measurement touches one
  (`_AUTO_CUTS_MAX_VARS = 40` first, since it has no measurement at all).
- **Not add a combinatorial flag-interaction test matrix.** 2^75 is not testable;
  the right fix is *fewer standing flags* — the graduation policy already trends
  parked flags toward deletion or default-ON, and item 5 reduces how many
  routing-relevant flags exist at all.

---

## 4. Bottom line

The skeptical intuition is right about the *shape* of the problem but wrong about its
*substance*. This is not a pile of per-problem hacks — the audits found literally zero
instance-keyed logic, and the fix-generalization discipline is verifiably followed.
It is, however, exactly what "months of iterative, evidence-gated, individually-sound
changes with almost no consolidation" produces: a 7,459-line implicit routing policy,
ten B&B loops, six FBBTs, a two-regime flag system where the word `false` means true
for 43 flags, per-kernel tightening stacks that differ silently, and best-in-repo
capabilities parked behind default-OFF flags while the default path loses to BARON by
7.9×. The certificate culture kept the accretion *sound*; nothing kept it *composed*.
The good news is that the highest-leverage fixes (§3 items 1–5) are plumbing, not
mathematics, and the repo's own verification machinery (bound-neutral gates,
differential panels) is precisely the tool that makes that plumbing work safe to do.
