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
| PF0 harness + baseline | open | | |
| PF1 branch-and-reduce ON | open (spike) | | |
| PF2 sound incremental node LP | open (spike) | | |
| PF3 branching tune | open (spike) | | |
| PF4 root-gap classes | open (spike) | | |
| PF5 incumbents + LP robustness | open (spike) | | |
