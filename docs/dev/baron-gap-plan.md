# BARON-gap plan — five independent, stacking tasks (G1–G5)

**Status:** G1–G5 **executed and merged** (2026-07-15) — see the verdict column
in §9 and the follow-on items in §10. Successor to `tenx-plan.md`, incorporating
the 2026-07-15 layer-correct profiling session (post-#636 uniform engine on `main`
@`0f3ebd7d`). Every number below was measured in that session or is cited to a
committed record; nothing is assumed. Net outcome: G1/G2/G4 landed net-positive;
G3 routing was falsified (kill criterion) but surfaced+fixed a P0 false-optimal;
G5 diagnosed family D and re-scoped bchoco06 to an LP-conditioning problem (G6).

**Audience:** each task section (§3–§7) is written to be implemented by an
independent agent session with no other context than this document plus the
repo. Read §0 and §8 first regardless of which task you take.

---

## §0. Operating rules (binding)

1. **CLAUDE.md governs.** Correctness before performance: `incorrect_count ≤ 0`
   with zero slack; never weaken a validation to make a gate pass; measurement
   beats hypothesis; entry experiment *before* implementation; record
   falsifications in this file's task table.
2. **Layer-correct attribution or it didn't happen.** This repo has repeatedly
   mis-attributed cost between Python/JAX/Rust (see §1.3 — June's
   "jax=97.7%" was actually Python↔JAX *marshaling*). Any performance claim in
   a G-task PR must state its measurement method (cProfile with
   binding-boundary aggregation, phase timers, or fresh-subprocess decomposition)
   and which layer boundary it crosses.
3. **Verification regimes** (CLAUDE.md §5): G2/G3 are behavior-changing →
   family-scoped differential gates (proofs never lost, bounds never looser,
   `--vs` comparison). G1 is bound-neutral by construction → byte-identical
   bounds/nodes gate. G4 is measurement-only → no solver gate. G5 is
   diagnosis-first → any build that follows gets the full bound-changing gate.
4. **Do-not-relitigate list is in §8.** TX2/TX2b (deadline truncation), TX3
   (JAX-free relaxation swap), PF5 (blanket native truncation) were falsified
   with recorded evidence. Do not re-propose them in G-task PRs; G1 explicitly
   *supersedes* TX3's kill with a different mechanism (see §3).

---

## §1. The measured decomposition (why we are 10–20× slower than BARON)

`wall_gap ≈ floor × per_node_cost × node_count`, with different factors
dominating different instance families. TX0's binding-constraint histogram
(62-instance panel, `docs/dev/data/tenx-attribution.json`): **floor 26
instances, node_count 18, no_bound 8, overrun 4**.

### §1.1 Factor F — the per-process floor (dominates 26/62 instances)

Fresh-subprocess decomposition, median of 3, trivial instance (alan), measured
2026-07-15 on `main`:

| phase | ms |
|---|---|
| `import jax` | 299 |
| `import pounce` | 148 |
| `import discopt` | 66 |
| parse `.nl` | 2 |
| **actual solve (21 nodes)** | **80** |
| **total** | **595** |

86% import tax; the solve itself is BARON-competitive (BARON solves alan in
~80ms *total*). Every fresh-process invocation — including every row of every
benchmark sweep, which runs discopt in an isolated subprocess — pays the 513ms.

**The warm daemon (`python/discopt/daemon.py`, CLI `discopt solve`) already
amortizes this.** Measured 2026-07-15, easy class, median of 3 vs BARON (June
baseline times):

| instance | daemon | fresh-proc | BARON | daemon/BARON |
|---|---|---|---|---|
| alan | 0.17s | 0.30s | 0.08s | 2.2× |
| gbd | 0.16s | 0.28s | 0.06s | 2.4× |
| ex1222 | 0.24s | 0.73s | 0.04s | 6.0× |
| st_test1 | 0.16s | 0.29s | 0.05s | 3.6× |
| nvs01 | 0.53s | 1.18s | 0.07s | 7.3× |
| st_miqp2 | 0.16s | 0.29s | 0.05s | 3.2× |
| **geomean** | **3.7×** | **7.5×** | 1× | |

→ The daemon halves the easy-class gap today. The residual ~3.7× is engine
time (F2) plus the ~150ms recurring per-solve engine constant
(performance-plan.md Appendix B). G4 makes the benchmark measure this honestly.

### §1.2 Factor N — node count (real, smallest factor)

June both-optimal panel: discopt/BARON node ratio **geomean 3.7×** (median
3.7×). The uniform engine (#636) already attacks this: nvs09 215→19 nodes,
nvs22 certifies in 35 nodes, st_miqp2 obj=2.0 in 9 nodes. Family D's frozen
bounds are a *relaxation-strength* problem, not a speed problem (G5).

### §1.3 Factor C — per-node cost: Python↔JAX **marshaling**, not JAX, not Rust

nvs05, 20s budget, cProfile aggregated across binding boundaries (2026-07-15,
`main`): 411 nodes → 20.5 nodes/s. BARON (June): 3,561 nodes in 1.9s →
**1,874 nodes/s**. Per-node gap ≈ **90×** on this instance.

| layer | share of 20s | what it is |
|---|---|---|
| python | **82.5%** | `pounce.Problem.solve` 515 calls, **16.0s cumulative** |
| jax/xla | 12.3% | kernel execution |
| rust (LP) | 3.4% | `solve_lp_warm_csc_py`: **0.67s — the node LP is nothing** |
| pounce native | 0.1% | the optimizer itself |

Inside those 515 NLP solves: `evaluate_constraints` called **402,669 times**,
`evaluate_objective` 400,329 times — **~780 evaluator callbacks per NLP
solve**, each a Python→JAX→`float()` round-trip. Interop overhead alone:
`np.asarray` 1.71s, `jax array._value` 1.53s, `array.__float__` 2.17s
cumulative. **The optimizer spends its life marshaling scalars across the
Python/JAX boundary one point at a time.** TX0 confirms panel-wide: node-NLP =
69% of all wall (600.6s of ~870s); node-LP = 0.06%.

These NLP solves are the **bound source**, not a skippable heuristic: with the
strided node-NLP fully off (`DISCOPT_NODE_NLP_STRIDE=1000000`), nvs05 wall and
bound are essentially unchanged (verified 2026-07-15). So the fix is making
each solve cheap (G1) or not needing an NLP at the node at all (G3), never
skipping bound-producing work (§8).

### §1.4 The "heuristics that never end" hypothesis — verdict

Tested (TX0/TX1/TX2, re-verified 2026-07-15): **mostly falsified as stated.**
The long-running calls are bound-producing native/XLA work (casctanks = one 48s
OBBT probe LP; contvar = 21s root probe that seeds relaxer state; heatexch =
uninterruptible XLA compile). Truncating them changes bounds chaotically
(casctanks bound −99.09 → +5.70 under an iteration cap — the PF5 trap). The
genuinely idle-heuristic slice is ~2 instances and already has a landed,
soundness-gated fix: TX1's adaptive back-off, default-OFF → G2 graduates it.

---

## §2. Independence and stacking (confirmed at code + behavior level)

**Code independence** (can be implemented/merged in any order; no file-region
conflicts):

| task | files touched | overlaps |
|---|---|---|
| G1 | `_jax/nlp_evaluator.py`, POUNCE call sites (`solver.py` ~10620–10640 `_IpoptCallbacks`/`pounce.Problem`) | none with G2–G5 |
| G2 | `solver_tuning.py:325` (one default) + docs/tests | `solver.py` untouched |
| G3 | `solver.py` routing predicate + OBBT gate (~4681), `_jax/lp_spatial_bb.py` | different `solver.py` region from G2's flag |
| G4 | `discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py` only | none (measurement) |
| G5 | `_jax/uniform_relax.py` (specific atom classes), diagnosis scripts | none |

**Behavioral stacking** (each is net-positive alone; combined benefits stack
but **sub-additively where two tasks touch the same family** — do not sum the
individual claims):

| | family A (floor) | family B (mid) | family C (int-product) | family D (tail) |
|---|---|---|---|---|
| G1 marshaling | small (solve is 80ms) | **large** | large *until G3 reroutes* | large where bound exists |
| G2 adaptive NLP | small | **moderate** (measured −16s/B) | small | mixed (nvs12 +40s if blanket-off — adaptive avoids this) |
| G3 LP-node routing | — | — | **large** (supersedes G1/G2 here) | — |
| G4 daemon/harness | **halves measured gap** | honest floor split | honest floor split | honest floor split |
| G5 relaxation strength | — | — | — | **unlocks** (no_bound/frozen instances) |

Known sub-additive pairs: **G1×G2** (G2 removes calls G1 makes cheaper — both
help family B, combined < sum), **G1×G3** (G3 moves family C off the NLP path,
so G1's family-C share vanishes; G1 keeps its B/D benefit). No pair conflicts;
each task's gate is family-scoped so landing order is free. G4 changes only
*measurement*, so it stacks exactly (a fixed subtraction of the floor from
every comparison).

**Recommended order if serialized** (highest certainty first): G4 → G2 → G1 →
G3 → G5. But the point of this document is that order is not required.

---

## §3. G1 — kill the Python↔JAX marshaling in the point-mode evaluator

> **Landed #645 (2026-07-15).** The literal "fuse `(f,g,c,J)` tuple +
> `jax.device_get`" implementation sketch below was **falsified** in the entry
> experiment (0.76–0.90×, slower; 87.8% of iterates want obj+cons only) and
> replaced by **co-occurrence fusion** (concat `FC=[f,c]` / `GJ=[g,J]`, single
> `np.asarray`, iterate-memo). Read the §9 verdict for the shipped design and
> numbers; the sketch below is retained as the original hypothesis.

**Objective.** Reduce per-NLP-solve callback overhead by ≥5× on the
nvs05-class measurement (§1.3) with **byte-identical bounds and node counts**
(the evaluator computes the same numbers, cheaper).

**Evidence.** §1.3: ~400k callback round-trips per 20s; interop (`asarray`/
`_value`/`__float__`) ≈ 5.4s of 20s wall on nvs05 alone; per-node gap 90×.

**Code anchors.**
- `python/discopt/_jax/nlp_evaluator.py` — `class NLPEvaluator` (line ~188):
  `evaluate_objective` (:499), `evaluate_gradient` (:503),
  `evaluate_constraints` (:511), `_evaluate_dense_jacobian` (:517),
  `evaluate_lagrangian_hessian` (:488), `_current_params` (:480). Each wraps a
  jitted JAX callable and converts the result per call
  (`float(...)`/`np.asarray(...)`).
- `python/discopt/solver.py` ~10620–10640: `_IpoptCallbacks(proxy)` →
  `pounce.Problem(n=..., m=..., problem_obj=callbacks, lb=..., ub=..., cl=...,
  cu=...)`. POUNCE calls `problem_obj.objective/gradient/constraints/jacobian/
  hessian` per iterate — one Python round-trip *per quantity per iterate*.
- `_BoundOverrideEvaluator` (same region) is a thin per-node proxy — any change
  must preserve its box-override semantics.

**Entry experiment (do this before building anything).**
1. Instrument one nvs05 NLP solve: count callbacks per solve by quantity
   (obj/grad/cons/jac/hess) and measure µs/callback split into {jit-call,
   device→host transfer, float()/asarray conversion, Python frame overhead}.
   A `time.perf_counter_ns` wrapper around each `NLPEvaluator` method is
   sufficient; run with `time_limit=20`.
2. Prototype (throwaway) a **fused evaluation**: one jitted function returning
   `(f, g, c, J)` as a single pytree, called once per iterate, converted to
   numpy once (`jax.device_get` of the whole tuple). Measure the same solve.
   - If POUNCE's `problem_obj` protocol forces per-quantity calls, memoize by
     iterate: cache key = `x.tobytes()`; first quantity computes the fused
     tuple, the rest hit the cache. This requires **no POUNCE change**.
3. **Kill criterion:** fused/memoized path < 2× callback-overhead reduction on
   the nvs05 probe, or any bound/node deviates from the default path.

**Implementation sketch** (after a passing entry experiment):
1. Add an iterate-memoized fused path inside `NLPEvaluator`: a single jitted
   `_fused(x, params) -> (f, g, c, J)` compiled per model; per-method entry
   points check the memo before dispatching. Keep the Hessian separate (it is
   called ~10× less — 45,710 vs 402,669 on the probe).
2. Convert device→host **once per iterate** (`jax.device_get` on the tuple),
   store numpy in the memo; per-quantity methods slice from numpy (no more
   per-call `__float__`/`asarray` on JAX arrays).
3. `_BoundOverrideEvaluator` and `_IpoptCallbacks` stay as-is (they wrap the
   evaluator; the memo lives below them, so Rayon-concurrent proxies each get
   their own evaluator instance — verify reentrancy: one memo per evaluator
   instance, no module-level state).
4. Optional follow-up (separate PR, only if the memo lands green): batch the
   *convex-separation* evaluations in `mccormick_lp.py` `_separate_convex` the
   same way (nvs09: `separate/convex` = 1.5s of 4.15s).

**Verification gate.**
- Bound-neutral regime: `node_count` and certified `objective` **exactly
  unchanged** on the 11-instance corpus (alan, ex1221/2/5, gkocis, nvs03/06/09,
  st_miqp1/2, st_e13) + nvs05/nvs22/ex14_1_9.
- `pytest -m smoke` green; adversarial suite green.
- Perf claim in the PR: nvs05 nodes/s before/after + the callback-count table
  from the entry experiment.
- **This is what would have caught the June mis-attribution:** report the layer
  split (cProfile binding-boundary aggregation) before/after.

**Why this does not relitigate TX3.** TX3 asked "remove `import jax` by
swapping the relaxation"; the kill stands (the *evaluator* needs JAX). G1 keeps
JAX and removes the *per-call marshaling*. A later "numpy point-mode evaluator
for small models" (which would also remove the import floor) remains open but
is NOT this task — it is an engine rewrite requiring its own plan; do not
scope-creep G1 into it.

---

## §4. G2 — graduate `DISCOPT_ADAPTIVE_NLP` to default-ON

**Objective.** Flip the TX1 adaptive node-NLP back-off from opt-in to default.

**Evidence.** TX1 landed 2026-07-14, default-OFF, with a green soundness gate:
full 62-panel `--vs` — proofs 41→**43** (gained nvs09, tspn05; 0 lost), **0
LOOSER / 0 CROSSED**, wall 870.2→824.6s (−45.6s); tls2's blanket-off looseness
does NOT occur under the adaptive form (bound byte-identical across repeats).
Re-verified on merged `main` 2026-07-15: nvs09 4.15→3.70s; nvs05 +88 nodes and
a tighter bound (5.4107 vs 5.3795) in the same 20s budget.

**Code anchors.**
- `python/discopt/solver_tuning.py:325`:
  `adaptive_nlp: bool = field(default_factory=lambda: _env_flag("DISCOPT_ADAPTIVE_NLP", default=False))`
  → flip `default=False` to `default=True`. Env semantics after the flip:
  `DISCOPT_ADAPTIVE_NLP=0` restores the fixed stride (the documented
  flag-graduation convention in the field's docstring — update that docstring).
- Mechanism (do not touch): `solver.py` node loop `_eff_nlp_stride` — stride
  starts at base 4, doubles (cap 256) after 2 consecutive non-improving
  batches, resets on incumbent improvement.
- `python/tests/test_tx1_adaptive_nlp.py` — tests currently assert the
  default-OFF behavior; invert the default-direction assertions.

**Entry experiment.** None needed beyond re-running the recorded gate — the
TX1 row in `tenx-plan.md` §3 *is* the entry experiment. Re-run its `--vs`
panel comparison at current `main` to confirm the numbers survived #636's
merge (they were measured on the PR branch).

**Verification gate.** The TX1 gate, re-run: full 62-panel default-ON vs
default-OFF — proofs ≥ unchanged (expect +2), 0 LOOSER/CROSSED bounds, wall
strictly down panel-wide; `pytest -m smoke` + adversarial green;
`test_tx1_adaptive_nlp.py` updated and green. **Kill criterion:** any lost
proof or looser bound at current `main` → keep default-OFF and record why in
this table.

---

## §5. G3 — route the integer-product family to the LP-node engine (TX4)

**Objective.** Family C (integer-product models: nvs17/19/24 class,
graphpart_*) burns per-node NLP where a pure LP node suffices (TX0: family C
node-NLP = 77% of its wall; SCIP solves this family at ~1,400 nodes/s with no
cuts). Route in-scope models to the existing LP-node engine by default.

**Evidence.** TX0 family table; `scip-gap-nvs-diagnosis.md` (all numbers);
TX4 row in `tenx-plan.md` §3 (planned, never executed — verdict blank).

**Code anchors.**
- Engine exists opt-in: `python/discopt/_jax/lp_spatial_bb.py` (incremental
  warm LP, node cuts, GMI with safety margin — its C-10 tests were un-deferred
  in #636's review and pass). Opt-in wiring per PR #290.
- OBBT gate: `solver.py:4681` — `_obbt_has_continuous = any(v.var_type ==
  VarType.CONTINUOUS ...)`; pure-integer models currently skip root OBBT
  (`:4683`). scip-gap plan Phase 1 predicts nvs17 root −65,842 → −6,790 when
  un-gated.
- Known blocker: **#287** — first incumbent latency on the LP engine
  (kall_congruentcircles_c72: 12.9s vs an 8s soft limit; dive/primal incumbent
  must surface early and the soft limit be honored between phases).

**Entry experiment (from the TX4 row, still required).**
1. Reproduce #287 at current `main` (first-incumbent time on
   kall_congruentcircles_c72).
2. Re-run the LP engine on nvs17/19/24 at current `main` (they are OUT of the
   62-panel — fetch from the corpus at
   `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`) to refresh the
   PR-#290-era numbers.
3. Measure root-OBBT un-gated on nvs17: does the root bound move
   −65,842 → ≈−6,790 as predicted?

**Implementation steps.**
1. Fix #287 (surface dive/primal incumbent early; honor the soft limit between
   phases).
2. Un-gate root OBBT for pure-integer models with nonlinear terms
   (`solver.py:4683` — extend the existing `_obbt_has_nonlinear` escape).
3. Add the routing predicate: models in the engine's proven scope (the scope
   test in `lp_spatial_bb.py`) route there by default; everything else
   untouched. Ship behind `DISCOPT_LP_SPATIAL=auto` with explicit
   on/off override, default `auto`.

**Verification gate (family gate from TX4).** nvs17/19/24 solved or bound
within 2× of true optimum (SCIP no-cut parity bar: 4.7s, not 0.88s);
graphpart_* improves; **zero out-of-family panel regressions** (full 62-panel
`--vs`: bounds/proofs unchanged off-family); #286-class false-unbounded
regression tests pass. **Kill criterion:** the engine (with OBBT + latency
fix) fails to beat the default path on its own family.

---

## §6. G4 — measurement truth: harness fixes + daemon benchmark lane

**Objective.** Make the benchmark measure what BARON comparisons need: real
BARON node counts, and discopt wall with the floor separated (fresh-process vs
warm-daemon lanes). Measurement-only; no solver behavior change.

**Evidence.** §1.1 (daemon halves the measured easy-class gap); this session's
sweeps recorded `baron node_count=0` on **150/150 rows** while the June
baseline (same GAMS/BARON) recorded real counts (nvs05: 3,561) — the current
harness never populates BARON's node count.

**Code anchors** (all in
`discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py`):
- `parse_lst` (~line 216 populates `node_count` from the discopt result;
  the BARON `.lst` side never extracts iterations) and `parse_baron_root`
  (~:310, parses the `lo=3` iteration log for the root bound — the same log
  carries per-iteration node counts).
- `run_baron` (~:355): runs `gams ... lo=3`; keep `proc.stdout` for node
  parsing.
- The discopt side runs one fresh subprocess per instance (isolation) — the
  513ms floor is charged to every row.

**Implementation steps.**
1. **BARON nodes:** parse the final iteration line of BARON's `lo=3` log
   (columns: iteration, nodes-in-tree, ...) and/or the `.lst` "Total no. of
   BaR iterations"/"Best solution found at node" lines into
   `SolverRun.node_count`. Validate against the June baseline on 3 instances
   (alan=3, nvs05=3561, nvs01=7).
2. **Floor-separated discopt timing:** inside the per-instance subprocess,
   report `(import_s, parse_s, solve_s)` alongside wall (the imports are
   already the first statements; wrap with `perf_counter`). Emit both
   `wall_time` (unchanged, back-compat) and `solve_time` in the JSON row.
3. **Daemon lane (opt-in `--via-daemon`):** route solves through
   `discopt.daemon.solve_via_daemon` (see `python/discopt/daemon.py`
   docstring; CLI equivalent `discopt solve <nl>`). First call warms the
   daemon (excluded warm-up solve); report daemon wall per instance. Fall back
   to in-process on daemon failure exactly as the CLI does.
4. Report table gains three ratio columns: `wall/baron` (today's number),
   `solve/baron` (floor-excluded), `daemon/baron` (deployment-realistic).

**Verification gate.** No solver code touched (diff confined to
`discopt_benchmarks/`); BARON node counts match the June baseline spot checks;
`solve_time + import_s + parse_s ≈ wall_time` within 5%; the three ratios
reproduce §1.1's measurements (daemon ≈ 3.7× easy-class geomean) on a 6-instance
smoke panel. **Kill criterion:** none (measurement); if BARON's log format
resists parsing, record the format sample in the PR and keep `node_count=None`
(never fabricate).

---

## §7. G5 — family-D relaxation strength (TX6 diagnoses, unchanged)

**Objective.** Family D (heatexch_gen1/2/3, bchoco06/07/08, casctanks,
4stufen, beuster) is stuck on **bound strength** (frozen/absent dual bounds),
not speed. Two bounded diagnoses, no build commitment.

**Evidence.** TX0 family D: 556.9s, 11 timeout / 4 feasible / 1 hang;
`sota-proof-plan.md` PF4 §3 (heatexch LMTD ε-pole *inside* the box →
relaxation is *correctly* unbounded); bchoco06 has **no finite dual bound at
7 nodes** (undiagnosed).

**Tasks (each one session, diagnosis-first).**
1. **bchoco06 unbounded-relaxation hole:** which atom class of which
   constraint leaves the root LP unbounded, and why (instrument
   `build_uniform_relaxation`'s per-atom envelope emission — the `audit_build`
   ownership report in `_jax/claim_audit.py` gives per-atom columns; find the
   free column with objective/constraint stake). Kill: hole is structural à la
   LMTD with no sound finite bound → document and close.
2. **heatexch pole-excluded sub-boxes:** branch once by hand on the `a=ε+b`
   pole line; measure the two children's root bounds with the (sound there)
   AM/GM envelope. Kill: children improve gen1's 38,184 root bound by <10%.
3. Any *build* that follows either diagnosis gets the full bound-changing gate
   (differential bound test + feasible-point sampling, feature-flagged
   default-off — CLAUDE.md §5).

**Verification gate.** For the diagnoses: a written record in this file (per
falsification house style). For any subsequent build: the bound-changing
regime, family-D bound improvements quantified against
`~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu`, zero regressions
elsewhere.

---

## §8. Do-not-do list (binding falsifications — new data required to reopen)

1. **Do not truncate bound-producing native solves** (TX2 STOP + PF5): capping
   casctanks' OBBT probe flipped its bound −99.09 → +5.70. The overruns ARE
   the dual bound.
2. **Do not re-add the Rust LP native deadline** (TX2b REVERTED, preserved at
   git `b8c591f`): every warm-LP caller is bound-producing or a control-flow
   gate; the contvar "discarded" probe seeds relaxer state (truncating it =
   looser bound + no wall win).
3. **Do not attempt the JAX-free relaxation swap** (TX3 KILL): `import jax` is
   the first op of both nonlinear entry points via `nlp_evaluator.py:19`; the
   evaluator, not the relaxation, is the JAX dependency. (G1 attacks the
   marshaling *within* that dependency; a full numpy point-mode evaluator is a
   separate, unplanned engine rewrite.)
4. **Do not blanket-disable the node-NLP** (TX1 entry experiment): tls2 bound
   went LOOSER, tspn12 lost its incumbent, net panel wall got *worse*
   (870→893s). Only the adaptive form (G2) is admissible.
5. **Do not trust single-layer profiler labels** (June's "jax=97.7%"): always
   aggregate cProfile across binding boundaries (pounce/_rust/jax/python) as
   in §1.3, or use the fresh-subprocess phase decomposition of §1.1.

---

## §9. Task table (verdicts land here)

| task | scope | entry experiment | gate | verdict |
|---|---|---|---|---|
| G1 marshaling | `nlp_evaluator.py` fused/memoized point-mode | callback census + fused prototype on nvs05 | byte-identical bounds/nodes; nodes/s up ≥2× on nvs05 probe | **LANDED #645** (2026-07-15). Literal `(f,g,c,J)`-tuple + `device_get` design **falsified** (0.76–0.90×, slower — census: 87.8% of iterates want obj+cons only). Landed **co-occurrence fusion** instead (concat `FC=[f,c]`, `GJ=[g,J]`, one `np.asarray`, iterate-memo): **2.45×** callback-overhead reduction on the access-pattern replay, byte-identical across the 13-instance panel (nodes/obj/bound/status exact). End-to-end nvs05 nodes/s **1.26×** (26.6→33.5) with a not-looser bound — marshaling is one component of node cost, so end-to-end < the callback-layer 2.45×. |
| G2 adaptive NLP default-ON | `solver_tuning.py:325` | re-run TX1 `--vs` panel at `main` | proofs ≥ baseline (+2 expected), 0 LOOSER/CROSSED, wall down | **LANDED #643** (2026-07-15). `adaptive_nlp` default `False→True`; env `DISCOPT_ADAPTIVE_NLP=0` restores fixed stride. Gate green. |
| G3 LP-node routing (TX4) | #287 fix + OBBT un-gate + routing | #287 repro; nvs17/19/24 refresh; OBBT root-bound probe | family: ≤2× of true on nvs17/19/24; zero off-family regression | **ROUTING KILLED #646** (2026-07-15) — kill criterion met: engine cannot beat the default path on its own family (default already solves nvs17/19/24: −1100.4/−1098.4 optimal, nvs24 0.27% gap; #636 + already-un-gated root OBBT superseded the freeze). Entry experiment instead surfaced a **P0 false-optimal** in the opt-in engine (nvs17 reported optimal −1836.2 at an infeasible point; #636's univariate-square bilinear lifting emptied `info["bilinear"]` so `_worst_product_var` declared "all products tight"). Landed the **soundness fix** (ground-truth incumbent verification + `unresolved_lb` floor; opt-in path only, default byte-identical). Engine restoration → G7 below. |
| G4 harness truth | benchmarks script only | June-baseline node-count spot check | ratios reproduce §1.1; no solver diff | **LANDED #642** (2026-07-15). BARON node parse (`Total no. of BaR iterations`; alan=3 validated exact vs June), floor-split timing (`import/parse/solve`), `--via-daemon` lane, 3 ratio columns. No solver diff. |
| G5 family-D diagnoses (TX6) | bchoco06 hole; heatexch pole children | they *are* the experiments | written record; builds get full gate | **LANDED #644** (2026-07-15, `g5-family-d-diagnoses.md`). bchoco06 unbounded-**hole FALSIFIED** — re-attributed to **in-house LP conditioning** (HiGHS gives root bound ~1.0; discopt simplex `iteration_limit`s on a 1e10‥5e-324 coefficient spread) → G6 below. heatexch pole-children **KILLED** (0.00% root-bound improvement). No build shipped. |

---

## §10. Follow-on items surfaced during G1–G5 execution

These were *discovered* by the G1–G5 sessions (measured, recorded), not part of
the original plan. Each is diagnosis-first and independent; scope before building
per §0.

**Status (2026-07-15):** G6 **in progress** — entry experiment reproduced (the
subnormal `-4.941e-324` lower bounds on `x²`/product aux columns are underflow
artifacts of a true 0 and, with the 1e10 coefficient side, break the simplex's
equilibration); a subnormal-flush fix is being implemented + gated. G7
**deferred** (rationale below — no default-path value while G3 routing is killed).

### G6 — in-house LP simplex conditioning (from G5's bchoco06 re-attribution)

**Hypothesis (evidence-backed, from `g5-family-d-diagnoses.md`).** bchoco06's
"missing dual bound" is **not** an unbounded-relaxation hole — the same root
relaxation solved by HiGHS returns a finite bound ~1.0, while the in-house
simplex hits an `iteration_limit` on a coefficient spread spanning ~1e10 down to
subnormal (5e-324). The defect is **LP conditioning / numerics in
`crates/discopt-core/src/lp/simplex/`**, not the relaxation layer.

**Entry experiment (before any code).** On the bchoco06 root LP: (a) confirm the
HiGHS-vs-simplex bound gap reproduces at `main`; (b) characterize the failing
basis (which columns carry the 1e10‥5e-324 spread; is it a scaling problem the
simplex should equilibrate, or a genuine ill-conditioning that needs a numerics
fix); (c) does column/row equilibration (geometric scaling) inside the simplex
close the gap without perturbing well-conditioned instances? **Kill criterion:**
scaling does not recover a finite bound, or it perturbs any byte-neutral panel
instance → escalate to a linear-algebra (refactorization/tolerance) fix with its
own plan. **Layer note:** this is a Rust-layer change — profile and gate in Rust
(`cargo test -p discopt-core`), do not conflate with the JAX evaluator work.

### G7 — restore the LP-node engine's product map — **DEFERRED (2026-07-15)**

**Context (from G3/#646).** #636's univariate-square bilinear lifting left the
LP-node engine's `info` product map empty for integer-product models, which (a)
was the root of the P0 false-optimal #646 fixed by falling back to ground-truth
verification, and (b) disables the engine's product branching + cuts
(`IncrementalMcCormickLP.ok=False`), so it has no throughput on family C.
Repopulating the product map to the #636 relaxation would restore spatial
branching + cuts.

**Disposition — deferred, not built.** The LP-node engine is **opt-in only**
(`solver.py:3209`, `if kwargs.get("lp_spatial", False)`); nothing routes to it on
the default path. Its only consumer would be G3 routing, which is **KILLED** (§9:
the engine cannot beat the default path on its own family even once made sound).
So G7 restores *throughput on an engine nothing uses by default and that loses to
the default path* — a benefit confined to an unrouted opt-in engine, which
CLAUDE.md §2 rejects. The #646 soundness fix already made that engine **safe**
(no false optima), which is the only property that mattered while it stays opt-in.
G7 is therefore parked: it becomes worth doing only if new evidence reopens G3
routing (a measured case where the LP-node engine, with a restored product map,
*beats* the default path on some family) — that evidence is the entry experiment
that would un-defer it. Until then the sound default path owns family C and this
item ships nothing.
