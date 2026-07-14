# SOTA proof plan — PF series (issue #632 follow-on; supersedes the EP series cadence)

Status: **OPEN** (2026-07-14). Successor to `docs/dev/engine-performance-plan.md`
(EP series, CLOSED — retrospective in §1). Companion invariants:
`CLAUDE.md` (correctness absolute), `docs/dev/performance-plan.md` (CC model),
`docs/dev/avm-canonicalization-plan.md` §10 (ledger).

## §0. Operating rules (what changed vs the EP series)

1. **The metric is proofs, not microseconds.** Every item is judged by
   `proved`/`nodes`/`wall` on the panel (in-container: the 62-instance vendored
   proxy panel via the PF0 harness; authoritative: the maintainer's `global50`
   runs, which land on the PR within hours). An item that does not move
   proofs/nodes dies at spike stage. `incorrect_count ≤ 0` remains the absolute,
   non-negotiable gate at every step — nothing here relaxes it.
2. **Spike before item.** No item is planned-in-full or assigned a build context
   until a bounded spike (≤ ~1 hour of agent time) has verified its premise and
   measured a candidate win on ≥3 unproved/slow instances. The EP series wrote 7
   items from one profile and 3 premises were false — the spike rule is the fix.
3. **Bound-changing is the point, not the exception.** Sound-by-construction
   bound-changing levers (valid cuts, sound box reduction, branching) are where
   SOTA wins live. Gate: the PF0 differential harness — (a) fixed-box
   differential bound check (new ≥ old, never crossing the oracle where known),
   (b) feasible-point sampling (0 cuts), (c) panel outcome (proved ≥ before;
   nodes not exploding on any instance class). When the in-container gate is
   green, the change lands **default-ON on this PR branch** — the maintainer
   reviews every push with global50 and a regression gets bisected/reverted
   (EP3 demonstrated the loop works). This intentionally replaces the EP-era
   "default-off + nightly" cadence **on this development branch**; the
   default-off discipline still applies to anything merged to `main` before its
   global50 confirmation. (Maintainer-visible deviation — recorded here.)
4. **Parallel by subsystem.** Items in disjoint subsystems (B&R tuning /
   node-LP path / branching / relaxation classes) run as concurrent worktree
   agents, converging through the PF0 gate. Sequential only within a subsystem.
5. **Measurement beats plan** (unchanged). Falsifications are recorded here.

## §1. EP retrospective (evidence for the rules above)

| EP item | Outcome | Lesson |
|---|---|---|
| EP1 analysis cache | ✅ real (294→169 ms/node; SGM 0.96→0.68 with EP5) | analyze-once caching was genuinely missing |
| EP5 jaxpr-eval grads | ✅ real (nvs09 45→33 s; jit correctly rejected) | bit-identity is checkable — check it |
| EP2 OBBT reuse | ✗ premise false (already existed) | spike first |
| EP4a facet cache | ~0 (3%, within noise) | spike first |
| EP4b sep warm-start | ✗ premise false (already existed) | spike first |
| EP3 patch-table | ✗ REVERTED — skipped per-node separation = looser bounds, 3 lost proofs | a "cheaper node" that skips work is not a technique; per-node tightness is sacred |

Panel state after EP + EP3 revert (expected ≈ pre-EP3 proofs with EP1/EP5 speed):
82–85/116 proved, SGM ~0.68 s, `incorrect_count = 0`. BARON/SCIP prove nearly
all of these quickly. The gap is **tree size** (branch-and-reduce built but not
cashed in) and **per-node LP reuse** (no sound incremental node LP), then
residual root-gap classes.

## §2. The PF items

### PF0 — Outcome + differential harness (the shared gate; 1 context, main tree)
`discopt_benchmarks/scripts/pf_panel.py`: run the 62-instance vendored corpus
(configurable time budget, default 30 s/instance, `--instances`/`--budget`),
emit per-instance `{status, objective, bound, nodes, wall}` JSON + a diff mode
(`--vs REF.json`) reporting proved-delta, node-ratio, bound-direction per
instance, and flagging any instance whose bound went LOWER (looser) or crossed
a known objective. Plus `--differential`: fixed-box root+child bound comparison
between two env configurations, and the feasible-point sampler reused from the
engine validation harness. Commit a reference JSON at current HEAD as the
standing baseline. This is the fast in-container stand-in for global50 and the
gate every PF item runs.

### PF1 — Cash in branch-and-reduce (B&R subsystem; the biggest expected win)
Premise (verified machinery, unverified payoff): per-node probing
(`DISCOPT_NODE_PROBING`, default-off), OBBT rounds/budgets, DBBT/reduced-cost
fixing all exist. BARON's trees are small because range reduction runs HARD at
every node. Spike: on ~15 unproved/slow proxy instances, sweep
probing on/off × `probe_max_vars` × `in_tree_presolve_stride` × OBBT rounds;
measure proved/nodes/wall. If the spike shows proofs: full item = pick the
winning config, run the PF0 gate, flip default-ON in-branch, ledger row.
Kill: no config proves anything new or shrinks trees ≥20% on the spike set.

### PF2 — Sound incremental node LP (node-LP subsystem; EP3 done right)
The correct version of what EP3 faked: persistent node LP with **parent cut +
basis inheritance** and **bounded per-node separation** — inherit everything
the parent separated (valid at the child box), warm-start, then run the
separation loop only on cuts VIOLATED at the child LP point (cap rounds), so
the child is never looser than "parent cuts + new violated cuts" and converges
to the same tightness the cold path reaches, at a fraction of the LP/JAX work.
Spike: prototype on nvs09 + tspn05 + st_e04 (the EP3 victims) — child bound
must be ≥ the cold child bound on every sampled box (differential), wall must
drop materially. This is the hardest item; it only proceeds if the spike holds
the differential on the EP3 victims specifically.

### PF3 — Branching quality (B&B subsystem; cheap spike, possibly large)
Machinery exists (pseudocost + reliability + spatial selection,
`branching.rs`). Unmeasured against SOTA. Spike: per-instance node counts on
the proxy panel vs published BARON node counts where available (plan §14 /
literature); sweep spatial branch-point rule (midpoint vs LP-point vs
convex-combination) and reliability threshold. Any config that cuts nodes ≥2×
on a class without losing proofs graduates through PF0.

### PF4 — Residual root-gap classes (relaxation subsystem)
The still-unproved families with known-looser roots: sign-mixed high-arity
multilinear (simultaneous/exponential hull), general linear-fractional
`A(x)/B(x)` (heatexch class), cvxnonsep sum-of-signomials. Spike per class:
measure the root gap on its instances, implement the class envelope only if
the gap explains the timeout (bound-changing → PF0 differential + panel gate).

### PF5 — Incumbents + LP robustness (mop-up)
(a) Incumbent latency (CC4): feasibility-pump/diving cadence on unproved
instances — earlier incumbents = more pruning. (b) contvar-class simplex
iteration budget on large tightened LPs (scaling/restart, not a bumped cap).
Spike each; only build what the spike shows.

## §3. Status

| Item | Stage | Result | Commit |
|---|---|---|---|
| PF0 harness + baseline | **DONE (2026-07-14)** | `scripts/pf_panel.py` — panel (subprocess-isolated, `--jobs N`, fresh env/worker) + diff (`--vs`, sense-aware LOOSER/CROSSED catcher, exits non-zero) + differential (`--env-a/--env-b` per-box bound compare + feasible-point soundness, 0 cuts). Baseline `docs/dev/data/pf-baseline.json` @ HEAD, 30 s/inst, jobs=4: **proved 36 / feasible 11 / timeout 10 / hung 5** (bchoco08, casctanks, contvar, hda, heatexch_gen3 — `solve()` overran budget+grace=90 s and were killed; isolation kept the panel alive), total_wall 1067 s. Notable: ex1226 optimal (obj −17.0, 5 nodes, 3 s); nvs09 feasible (bound −51.97, obj −39.60, 15 nodes); tspn05 feasible (bound 178.27, obj 191.26, 31 nodes). ruff/format clean; 5-inst smoke green; regression + differential gates verified firing on synthetic inputs. | branch `claude/issue-632-opus-plan-ffxld4` (#632) |
| PF1 branch-and-reduce ON | **DONE — wiring LANDED (2026-07-14)** (spike: `docs/dev/pf1-branch-and-reduce-spike-2026-07-14.md`) | **LANDED:** the FBBT-only in-tree reduce kernel is now wired into the GLOBAL spatial B&B node loop (`solver.py:~6134`, gated by `in_tree_presolve_stride`), passing the REAL node box + incumbent-as-cutoff + real `node_depth` (new `TreeManager::node_depth` / `PyTreeManager.node_depths`, replacing the hardcoded 0 on both paths); default `in_tree_presolve_stride` flipped 0→1 on BOTH paths (FBBT-only; probing stays OFF). Contraction is intersect-only, outward-rounded, cutoff-aware — sound by construction (a proven-empty box fathoms; a reduced box always still contains the whole feasible region). Telemetry `_in_tree_presolve_global_calls()` + regression test `test_pf1_global_in_tree_presolve.py` pin the wiring (stride≥1 ⇒ >0 firings on a spatial model, stride=0 ⇒ 0). Confirmed firing on the global path: nvs05 150 calls, tspn05 29, bchoco06 1, st_e38 3. **Controlled OFF-vs-ON (identical binary, spike set, 40 s, jobs 4):** m3 **feasible→OPTIMAL** (proof gained, 61→47 nodes), fac2 **−43 %** nodes (69→39), clay0303hfsg **−28 %** (223→159) — these three are the NLP-BB stride flip; on the GLOBAL path, **nvs05 dual bound 1.3521→4.3023 (3.2× tighter, sound: bound ≤ obj 5.47)**, tspn05 bound unchanged (179.44, 39→37 nodes), bchoco06/07 + heatexch_gen1/2 unchanged (stuck ≤3 nodes at TL — bottleneck is upstream of range reduction, not FBBT). **Gates all GREEN:** (a) `--vs pf-baseline.json` — m3 proof gained, 0 lost, no LOOSER/CROSSED bound; (b) differential (`--env-a "" --env-b "DISCOPT_NODE_PROBING=1"`, 7 inst ×5 boxes) — env-b at-least-as-tight per box, feasible-point worst violation 1.8e-11 « 1e-5, **0 cuts**; (c) `bound ≤ objective` 0 violations on every ON and OFF row. `pf-baseline.json` regenerated at the new ON default (**full 62-instance corpus, 30 s: proved 36→41, +5 gained [cvxnonsep_nsig30, m3, nvs02, nvs11, st_e36], 0 lost, no LOOSER/CROSSED bound, hangs 5→1, wall 1067→963 s**). Suites: cargo test `discopt-core` 445+4+1 green; smoke 634; fast selection 5271 passed/0 fail; claim_boundary 360; bilevel+GDP soundness 17 (rebased #638 holds); ruff/format/mypy clean. **Verdict:** the global B&R wiring is a SOUND, no-regression positive (nvs05 bound 3× tighter) but is NOT by itself the BARON-closing lever on the hardest global instances (bchoco/heatexch are stuck *before* reduction can bite); the biggest measured wins remain the NLP-BB stride flip (m3 proof, fac2/clay node cuts). Range reduction is a real cash-in but the BARON gap on the stuck spatial instances needs a different lever (relaxation strength / branching / first-incumbent), tracked in later PF items. **Spike detail (retained):** 87 subprocess runs, 0 soundness violations. **Central finding: the B&R kernel (`in_tree_presolve_stride`/`DISCOPT_NODE_PROBING`/`PROBE_MAX_VARS`) is wired ONLY into `_solve_nlp_bb` (solver.py:9300, node_depth hardcoded 0 at :9324); the GLOBAL spatial B&B path — where every unproved instance lives (tspn*, bchoco*, contvar, heatexch*, nvs05) — never invokes it** (call-count monkeypatch = 0; node-identical across all configs). So the BARON-gap premise is neither confirmed nor killed — the flags can't reach the code path that matters. Where reachable (NLP-BB/convex): **`fbbt_s1` (stride=1, probing OFF) gains a proof (m3 feasible→optimal, 61→47 nodes), cuts fac2 −43%, no losses, no overhead**; probing is net-negative off that path (clay0303hfsg 60 s: 63 nodes / no incumbent/bound vs baseline 251 / both; psig40r +40 % wall) → keep probing OFF pending a scored/budgeted policy (confirms the P3 caveat). **The real PF1 item:** (1) wire the in-tree FBBT(+reduce) kernel into the global spatial B&B node loop (near solver.py:6155 / `node_reduce`), passing real `node_depth`; (2) flip NLP-BB default `in_tree_presolve_stride` 0→1 (FBBT-only, data-supported); (3) fix the `node_depth=0` hardcode; (4) `_PER_NODE_OBBT_ROUNDS` (solver.py:237) needs an env/kwarg before it can be swept. All bound-changing (box reduction) → PF0 differential + feasible-point + panel gate; a reduced box must never exclude a feasible point. | `perf(pf1)` on `claude/issue-632-opus-plan-ffxld4` (#632) |
| PF2 sound incremental node LP | open (spike) | | |
| PF3 branching tune | **SPIKE DONE → KILL (both axes)** (`docs/dev/pf3-branching-spike.md`) | ~30 runs, 0 soundness violations. Current rules (read from code): global spatial B&B = the Rust `PyTreeManager` (`tree_manager.rs::process_evaluated`); spatial branch VAR = longest normalized edge, **branch POINT = box midpoint HARDCODED** `0.5*(lb+ub)` (`branching.rs:478/550`, the node LP solution is in scope but never consulted — the "blind midpoint" the premise flagged); integer reliability threshold = **8 HARDCODED** (`tree_manager.rs:216`). Neither axis is env/kwarg-tunable, so both were prototyped in an ISOLATED worktree (env gates, default-off byte-identical, `cargo build --release`, worktree PYTHONPATH — main tree / shared venv untouched). **Result: both inert on the spike set** — LP-anchored branch point: fac2 (only proof) byte-identical across midpoint/lp/lpmid; nvs05/heatexch node counts fall but the **dual bound is UNCHANGED** (fewer nodes = more per-node time, not better search); reliability threshold {1,4,8,16}: completely inert (fac2 = 69 nodes at every value). **KILL** per the criterion (no axis cuts nodes ≥2× on a class or gains a proof; the one proof is invariant). **Method caveat (binding for all future branching/node measurement): node counts are NON-DETERMINISTIC under a wall-time budget** — solver.py's wall-fraction-budgeted root heuristics (RENS #281) + CPU contention move fac2 101→69 nodes with branch code UNSET; use **jobs=1 serial + dual-bound-at-timeout** as the robust signal, never raw node counts under contention. **Redirect (evidence-backed, corroborated by PF1's verdict): the spike timeouts trace to PF4 (frozen McCormick bound: nvs05 on i1·i2, tspn05) and PF2 (per-node cost starving trees to 3–31 nodes/40 s: heatexch*/bchoco*), NOT branch-point.** No PF1 conflict (branching.rs vs solver.py node loop — different layer). Revisit PF3 only if a later panel surfaces a class with a large, *moving-bound* tree. | KILL (worktree prototype, not pushed) |
| PF4 root-gap classes | open (spike) | | |
| PF5 incumbents + LP robustness | open (spike) | | |
