# Session Consolidation — 2026-07-10 (gap-closing campaign)

Single source of truth for what the gap-closing campaign produced, what is merged,
what is parked, and what is at risk. Created in response to "consolidate what we
have and make sure it is not lost."

## 1. Merged this session (all on `main`, CI-green)

| PR | what |
|---|---|
| #582 | reduced-space evaluator refuses non-finite boxes (nvs22 root cause #1) |
| #583 | P2.3 wire `DISCOPT_RELAX_SPACE=reduced` (MAiNGO parity, default-OFF, sound on nvs22) |
| #584 | P2.4 entry experiment — reduced-space never beats lifted; auto-select KILLED |
| #585 | gap-closing execution plan (`docs/dev/gap-closing-execution-plan.md`) |
| #586 | STRUCT-1 — CSE verified already-on-main (median 25% DAG reduction) |
| #587 | CUTS-1 — aggregation c-MIR NO-GO (separator already sound + root already tight) |
| #588 | MAIN-HEALTH — cert:T0.3 `solver_stats` schema (restore green main) |
| #589 | TAIL-1c — `root_gap_ratio_vs_baron` now evaluable (added BARON root ref; fails at 2.69×) |
| #590 | TAIL-1a — certification-stall attribution (diagnostic) |
| #591 | **A-2 — failure-triggered dense LP retry; CURES the nvs21 certificate loss** |
| #592 | **OVERHEAD-1 — lazy-SymPy; −20% median / −37% p25 startup on the easy class, bound-neutral** |
| #593 | TAIL-1b — binary `.nl` parsing (transcode-to-text; st_miqp5 readable; +1 coverage) |
| #594 | rescue MAiNGO Crucible KB (task #66 — was uncommitted/at-risk) |

## 2. Flag inventory (authoritative from `main` `solver_tuning.py`, 2026-07-10)

**Graduated / default-ON** (on the hot path today):
`RLT_QUAD`, `MULTILINEAR_SEPARATE`, `TRILINEAR_RLT`, `SQUARE_SEPARATE`,
`EDGE_CONCAVE`, `PSD_COST_GATE`, `LP_WARMSTART`, cut-pool inheritance
(structure-gated, #55/#61), root-heuristic governor/RENS (#44).

**Parked / default-OFF** (implemented + wired, awaiting graduation or killed):
| flag | status |
|---|---|
| `ROOT_FIXPOINT` | branch-and-reduce root loop (in main via #524). #581 gave a −14% verdict but it did **not reproduce** under re-test; graduation OPEN. |
| `NODE_REDUCE` | per-node FBBT + reduced-cost DBBT (in main via #524). Blocked by an st_e36 cert loss (LP-failure class) — re-gate now that A-2/#591 is merged (BR-3). |
| `LU_DENSITY_ROUTE` | density-aware sparse LU + **A-2 dense retry (#591) that cures nvs21**. Re-gate for default-ON (BR-3). |
| `SQUARE_COST_GATE` | sound; nvs21 cert loss (LP-failure class) — re-gate post-A-2. |
| `OBJ_BRANCH_PRIORITY` | sound; nvs21 cert loss — re-gate post-A-2. |
| `LIFTED_FBBT` | sound; st_e30 cert loss. |
| `ALPHABB_WITH_LP` | sound; no perf gain when LP relaxer active (KEEP-OPT-IN). |
| `RLT` | default-OFF (RLT_QUAD is the graduated variant). |
| `RELAX_SPACE=reduced` | MAiNGO reduced-space; sound, opt-in; auto-select KILLED (P2.4, #584). |

Nothing here is lost — every flag is implemented in `main` and tracked. Several
nvs21/st_e36 cert losses share ONE mechanism (sparse-route LP failures) that A-2
(#591) now cures → BR-3 re-gates them together.

## 3. Binding falsifications this session (do not relitigate)

- Reduced-space McCormick never tightens the root bound vs lifted (P2.4, #584).
- Aggregation c-MIR already exists + discopt root already at SCIP-with-cuts strength
  (#587).
- CSE/hash-consing already in main (#453); V-segments have 0 applicable instances
  (#586).
- LU condition estimate does NOT discriminate failing bases; the mechanism is LP
  failure-rate; fix is failure-triggered retry (perf-plan §9, #591).
- **`root_fixpoint` in isolation is ≈neutral** (re-tested 2026-07-10). Whether the
  full branch-and-reduce STACK (root+node reduce + strengthen, run together) beats
  baseline is UNDER INVESTIGATION — the correct experiment is combined, not
  per-flag (see §4).

## 4. Open threads

- **Branch-and-reduce as a SYSTEM.** Per-flag testing shows each reduce flag is
  marginal alone. Combined-stack experiment (root_fixpoint + node_reduce +
  lifted_fbbt + density-retry + RLT/strengthen, all ON) vs baseline on the
  tree-opening class is running. Early signal: the stack is NOT collapsing trees
  the way BARON does → likely a genuinely missing capability (candidates:
  per-node OBBT is class-gated to n≤100; OBBT candidate scoring is index-order
  not width×|RC| (T2.5 unbuilt); OBBT budget ~10% TL). This is the real gap to
  branch-and-reduce parity — BR-2 scope.
- **BR-3** — re-gate the nvs21/st_e36-blocked flags (node_reduce, density route,
  square_cost_gate, obj_branch_priority, TD-A/lift-loose-products for nvs09) now
  that A-2 cures the LP-failure class.
- **BR-2** — Phase-2 completion (per-node OBBT declass, T2.5 scoring, warm probes,
  in-tree probing). Note: T2.4a node-LP duals ARE already exposed (main), contrary
  to the plan's stale note.

## 5. Workspace hygiene (needs the maintainer)

The primary checkout `/Users/jkitchin/projects/discopt` is on
`fix-467-unbounded-root` — **195 commits behind `main`** — with uncommitted work
(the MAiNGO Crucible KB, now rescued in #594; plus benchmark result JSONs and a
stray `tim.lst`). Recommendation: after #594 merges, reconcile this checkout onto
`main` (stash/commit the remaining `.crucible` MANIFEST/index edits or regenerate
them via the crucible CLI). No solver code is at risk — origin/main has all merged
work; this is a local-branch hygiene item.
