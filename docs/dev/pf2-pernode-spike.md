# PF2 SPIKE — sound incremental node LP / per-node cost (2026-07-14)

Status: **DONE → KILL** on this spike set. Measurement-only; prototype work
lived in an **isolated worktree** (reset to the PF branch tip so PF1's landed
node-loop code was present) and is **NOT pushed**. Companion:
`docs/dev/sota-proof-plan.md` §2 PF2; premise source: `docs/dev/pf3-branching-spike.md`
(redirect) + the EP3 revert (`docs/dev/engine-performance-plan.md`).

Isolation: `_rust.so` copied from the main tree (already built at PF1 state,
`PyTreeManager.node_depths` present); everything run with
`PYTHONPATH=<worktree>/python` so the shared `.venv` / main tree (PF4's
workspace) were never touched. No library code was edited — the spike falsified
its premise before any prototype was warranted (PF §0.2: spike before item).

## 1. Premise under test

PF2 premise (from the PF3 redirect): *on the starved spatial instances
(heatexch\*/bchoco\* at 3–31 nodes/40 s) the per-node relaxation + separation
cost is so high the tree cannot explore; a sound "cheaper node" (parent cut +
basis inheritance, warm-start, bounded per-node separation — "EP3 done right")
lets the tree explore without losing tightness.*

**This premise is FALSIFIED by measurement.** The per-node relaxation build and
separation are *negligible* on these instances; the tree is starved by costs
that PF2's node-LP-inheritance lever does not touch (a primal-heuristic node NLP,
a one-time root presolve, and genuine large node-LP simplex work). And every
mechanism PF2 proposes to build **already exists** and is already sound.

## 2. Per-node cost breakdown (cProfile tottime + built-in `solver_stats`)

Real `model.solve(time_limit=40)` per instance, in-container, jobs=1. The
solver already surfaces per-family timers on `SolveResult.solver_stats`
(`reduce/fbbt` = FBBT + PF1 in-tree presolve; `reduce/obbt`; `separate/<family>`).
cProfile supplies the build / Rust-LP / node-NLP split.

**heatexch_gen2** (112–148 cols; 3–7 nodes; ~48 s; time_limit; bound frozen):

| component | cost | note |
|---|---|---|
| Ipopt node NLP (`Problem.solve` = cyipopt) | **26.3 s cum / 11.7 s tot, 10 calls** | primal heuristic — NOT a bound in nonconvex LP mode |
| root `PyModelRepr.presolve` (Rust) | **10.0 s, 1 call** | one-time root cost |
| interval evals (`evaluate_constraints`) | 2.6 s | FBBT/OBBT + NLP callback |
| relaxation build (`build_uniform_relaxation`) | 5.0 s cum / **8 builds total** | EP1-cached; not per-node-dominant |
| FBBT (`reduce/fbbt`, incl. PF1) | 2.0 s | |
| node LP (`solve_milp_py`+`solve_lp_warm_csc`) | ~1.6 s | |
| **separation (convex+multi+edge)** | **0.62 s** | the classic PF2/EP3 target — negligible |
| eigvalsh (convexity certs) | 0.75 s | |

**bchoco06** (118 cols; 7 nodes; 44.5 s; time_limit):

| component | cost | note |
|---|---|---|
| root `PyModelRepr.presolve` (Rust) | **7.85 s, 1 call** | one-time |
| Ipopt node NLP (`Problem.solve`) | **17.2 s cum / 5.8 s tot, 1 call** | one hard primal NLP |
| node LP (`solve_lp_warm_csc` 4.56 s/9 + `solve_milp` 3.5 s/4) | **8.06 s** | genuine simplex work, distinct tightened boxes |
| pounce `solve_problem_batch` | 2.4 s | |
| interval evals | 2.5 s | |
| FBBT (`reduce/fbbt`) | 2.83 s | |
| relaxation build | 2.98 s cum / **6 builds total** | |
| **separation (convex)** | **0.41 s** | negligible |

**nvs05** (8 cols; 259 nodes; 40 s; feasible; **bound frozen** — PF4 class):

| component | cost | note |
|---|---|---|
| Ipopt node NLP (`Problem.solve`) | **24.1 s cum / 9.95 s tot, 344 calls = 60 % of wall** | primal heuristic |
| node LP (`solve_lp_warm_csc`) | 9.84 s, 212 calls | |
| FBBT (`reduce/fbbt`) | 3.09 s | of which **PF1 `in_tree_presolve` (Rust) = 0.16 s / 182 calls** |
| relaxation build | 1.2 s / 9 builds | |
| separation | negligible | |

**EP0 steady-state `solve_at_node` ms/node** (build + separation + node LP only,
warm children — i.e. exactly the surface PF2 would optimize):
nvs05 **15**, nvs09 **153**, tspn05 **596**, heatexch_gen1 **121**,
heatexch_gen2 **308**, bchoco06 **1834**, bchoco07 **3058** ms/node.

**Crux.** For heatexch_gen2 the EP0 `solve_at_node` figure is 308 ms/node but the
*real* per-node wall is ~16 000 ms/node — a ~50× gap. That gap is entirely the
node NLP + root presolve + interval evals that live **outside** `solve_at_node`.
PF2 can only make `solve_at_node` cheaper, and `solve_at_node` is a small slice
of real per-node time; within it, separation is already <0.6 s and the build is
EP1-cached.

## 3. What dominates (and which PF item owns it)

1. **Ipopt node NLP local solve** (`_solve_node_nlp_ipopt`, `Problem.solve`):
   17–26 s / 40 s. A *primal heuristic* — in nonconvex LP-relaxer mode its
   objective is not a valid bound (the McCormick LP bounds every node), so its
   cost has **zero tightness stake** and cutting it can never loosen a child
   bound. Already gated by `node_nlp_stride` (default 4, `solver.py:6577`,
   `DISCOPT_NODE_NLP_STRIDE`). This is **PF5(a) incumbents/latency**, not PF2.
2. **Root `PyModelRepr.presolve`** (Rust): 8–10 s, one-time. A fixed *root*
   cost, not per-node — a distinct root-cost item, not PF2.
3. **Node LP simplex** (`solve_lp_warm_csc_py` / `solve_milp_py`): genuine work
   on distinct tightened boxes — **exactly EP4b's conclusion** ("the Rust LP is
   the in-house simplex doing genuine work on distinct cold child boxes, not
   redundant re-computation"). Already warm-started across nodes and
   pool-inheriting (see §4). Residual → **PF5(b)** simplex robustness, if
   anything.
4. **PF1's added FBBT is NOT a burden.** `reduce/fbbt` is 1.4–3 s total and the
   *PF1-added* Rust `in_tree_presolve` is only **0.16 s / 182 calls** on nvs05.
   No stride/depth-gating of the reduction is warranted (the alternative lever
   the task flagged "if FBBT dominates" — it does not dominate).

## 4. PF2's mechanisms already exist and are already sound

Every lever PF2-as-specified would build is already present:

- **Parent cut inheritance** → the general root cut pool (`solver.py`
  `_root_cut_pool`, `inherited_cuts`), with the C-42/C-43 "pool is an
  accelerator, never a dependency" soundness guards
  (`mccormick_lp.py:solve_at_node`).
- **Cross-node basis inheritance / warm-start** → `_inc_warm_basis` /
  `_inc_basis_nrows` (`mccormick_lp.py:568,739–757`), with the C-38 guard that
  re-solves cold when a carried warm basis would certify a *false* infeasible.
- **Bounded per-node separation** → the incremental fast path
  (`_try_incremental_node`) inherits the pool + warm-starts + separates only
  what the child point violates.

EP4b already measured warm-start at **100 % hits** and the root pool inheriting
the full separation chain. There is no material "cheaper node" left to build in
the PF2 subsystem.

## 5. Differential (mandatory EP3-trap gate) — GREEN

`pf_panel.py --differential --env-a "DISCOPT_INCREMENTAL_MC=0" --env-b ""`
(env-a = cold path = reference tightness; env-b = default incremental warm+pool
path; env-b must be ≥ env-a per box), 5 boxes/instance, on the EP3 victims +
starved set:

| instance | root_a (cold) | root_b (incremental) | boxes | result |
|---|---|---|---|---|
| nvs05 | (LP not optimal) | (LP not optimal) | 6 | at-least-as-tight |
| tspn05 | 167.79 | 167.79 | 6 | at-least-as-tight (identical) |
| nvs09 | −51.1443 | −51.1443 | 6 | at-least-as-tight (identical) |
| heatexch_gen2 | 543496 | 543496 | 6 | at-least-as-tight (identical) |
| bchoco06 | (max; LP not optimal) | (LP not optimal) | 6 | at-least-as-tight |

Feasible-point soundness on the env-b relaxations: **worst violation 1.82e-11**
(« 1e-5), **0 cuts** → SOUND. **PF0 differential gate GREEN.** The incremental
node LP is byte-for-byte as tight as cold on the EP3 victims — no EP3 loosening
exists in the shipped default, confirming §4's machinery is sound (and that
there is nothing to "fix" there).

## 6. Sound-lever probe: node-NLP budget does not unstarve

The one data-picked sound lever (raise `node_nlp_stride`, which only throttles a
primal heuristic — zero differential risk) was tested 4 vs 16, jobs=1, tl=40:

| instance | stride 4 (default) | stride 16 |
|---|---|---|
| nvs05 | feasible, 299 nodes, obj 5.47093 | feasible, **299 nodes**, obj 5.47093 |
| heatexch_gen2 | time_limit, 7 nodes | time_limit, **7 nodes** |

**Inert.** nvs05 explores the same 299 nodes and its bound stays frozen (its
timeout is the loose McCormick envelope on the integer product `i1·i2` — a
**PF4 root-gap**, per the PF3 spike, so *more* nodes would not close it anyway).
heatexch_gen2 stays at 7 nodes — the binding costs there are the one-time root
presolve + the primal NLP that the stride does not fully govern.

## 7. Verdict: **KILL**

Per PF §0.2 (an item that does not move proofs/nodes dies at spike stage; write
no item from a false premise): **KILL PF2** on this evidence.

- The classic PF2 lever (inherit cuts + basis, bounded separation) has **no
  material cost to save** — separation is <0.6 s and the build is EP1-cached on
  every starved instance; the ms that PF2 optimizes (`solve_at_node`) is a small
  slice (~2 %) of real per-node time.
- Every PF2 mechanism **already exists** (root cut pool, `_inc_warm_basis`,
  incremental fast path) and is **already sound** (differential GREEN, §5).
- The real starvation costs belong to **other PF items**: the Ipopt node NLP
  primal heuristic → **PF5(a)**; the one-time root presolve → a **root-cost
  item** (new); genuine node-LP simplex work → **PF5(b)**; and the tree-moving
  instance (nvs05) is **bound-frozen → PF4**.
- PF1's FBBT is cheap (0.16 s Rust in-tree presolve) — **not** a per-node burden;
  no reduction stride/depth-gate is needed.

**No prototype was built** (the premise failed the entry measurement), so no
bound-changing change was produced and the EP3 trap cannot be sprung. **No PF1
conflict**: nothing here touches `solver.py:~6134` (PF1's node-loop reduce
wiring). *If* a future item attacks the dominant node-NLP cost (the real lever),
it lives in the **PF5** subsystem at the `node_nlp_stride` gate
(`solver.py:6577–6670`) — which **shares the node-loop file with PF1's landed
code**, so that item (not this one) must be sequenced against PF1.

## 8. Recommendation to `main`

Redirect PF2's budget: the per-node lever that actually starves the stuck
spatial instances is the **node NLP primal-heuristic cost** (sound to throttle;
already `node_nlp_stride`-gated) + the **one-time root presolve** — score/budget
these under PF5 (and open a root-presolve-cost item), while the still-frozen
bounds (nvs05/tspn05) remain PF4. Do **not** build a "sound incremental node LP"
item — it is already shipped and sound.
