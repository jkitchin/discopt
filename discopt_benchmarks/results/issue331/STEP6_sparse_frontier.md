# Issue #331 — Step 6: the sparse frontier — cuts, blocked on dense cut rows

Step 5's sparse instances exposed the open frontier: discopt is 24–53× SCIP nodes
on sparse-row knapsacks. Step 6 grounds *what* closes that gap and builds the
relevant primitive. The result: **the lever is cuts (not clique/conflict), the
cuts close the node gap to SCIP's level, but discopt cannot afford them in wall
time because its cut rows are dense and defeat the #334 sparse engine.** Closing
the sparse regime needs sparse cut-augmented LP handling — a data-layer item, not
an algorithmic cut change.

## Grounding 1 — it is NOT clique/conflict propagation

The natural hypothesis (sparse ⇒ conflict structure ⇒ clique cuts) is **wrong**
here, caught before building:

- **Zero conflict pairs** in every sparse instance (`w_i + w_j > cap` in a shared
  row never holds — `cap = ½Σw` is too loose for pairwise conflicts, exactly as
  on the dense instances).
- **SCIP propagator decomposition** (sparse, no restarts) — separation dominates;
  conflict analysis is negligible:

  | smdk60x20 | full | no-separation | no-propagation | no-conflict | no-redcost |
  |---|---|---|---|---|---|
  | nodes | 9 | **637** | 25 | 11 | 21 |

  Removing **cuts** explodes SCIP's tree 70× (9→637); removing conflict analysis
  does almost nothing (9→11). On sparse, SCIP's advantage is **cuts**.

## Grounding 2 — cuts close the node gap, but the wall trade is negative

Cranking discopt's existing cuts on sparse reaches SCIP-level **node** counts:

| instance | prod | cuts++ (64/10 + node cuts) | SCIP |
|---|---|---|---|
| smdk80x25 | 36 957 n / 3.9 s | **737 n** / 9.7 s | 753 n / 1.9 s |

`cuts++` matches SCIP's node count (737 ≈ 753) — the algorithm works — but takes
**5× SCIP's wall** (9.7 s vs 1.9 s). The per-node cost with cuts is the problem:
SCIP ≈ 2.5 ms/node *with cuts*, discopt ≈ 13 ms/node.

## What was built — cut selection / management (`lp/cut_select.rs`)

SCIP keeps a small, diverse active cut set. This step adds the standard primitive
— efficacy (`violation/‖coeffs‖`) + orthogonality (drop near-parallel duplicates)
selection, capped at the `root_cuts` budget — wired into the root cut loop behind
`cut_select` (default off; sound, never modifies a cut). With a small cap and
many rounds it keeps the active set small while iterating.

It is **not enough on its own.** Searched the config space on the worst sparse
instance (smdk80x25, SCIP 753 n / 1.85 s):

| config | nodes | wall |
|---|---|---|
| prod | 36 957 | 3.9 s |
| select 25/3 | 38 111 | 5.8 s |
| select 50/3 | 23 709 | 7.9 s |
| select 75/3 | 14 211 | 8.7 s |

Every config that cuts nodes meaningfully **regresses wall** (no wall-neutral
sweet spot). Capping the *count* doesn't help, because the kept cuts are still
**dense** (lifted cover/GMI fill many columns), so the cut-augmented matrix is
dense and #334's sparse simplex no longer applies — every node pays a dense
re-solve.

## The real blocker — dense cut rows defeat the sparse engine

#334 made the *original* sparse rows fast, but separated cuts are dense, so the
moment cuts are added the per-node LP is dense again. That is why SCIP (which
keeps cuts sparse and manages/ages them aggressively) is ~5× cheaper per
cut-node. Closing the sparse regime therefore needs **sparse cut representation
in the LP layer** (keep the cut-augmented matrix sparse: sparse cut rows, sparse
factor updates, cut aging/removal) — "Regime C, part 2", a data-layer item — with
`cut_select` as the management primitive on top. It is not reachable by tuning
cut *quantity* or *selection* alone (measured).

## Status

- **Lever for the sparse frontier identified and grounded:** cuts (the node
  benefit reaches SCIP's level), blocked on dense cut-row LP cost.
- **`cut_select` primitive built** (sound, tested, default-off) — the management
  layer the eventual sparse-cut work needs.
- **Not yet a wall win on sparse** — that requires the sparse cut-augmented LP
  data layer, the concrete next project for closing #331's sparse regime.

### Reproducing

SCIP decomposition via pyscipopt toggles (`separating`, `propagating/*`,
`conflict/enable`); discopt sweeps via `solve_milp_py(cut_select=…, root_cuts=…,
cut_rounds=…)` over `gen_sparse_mdk` instances.
