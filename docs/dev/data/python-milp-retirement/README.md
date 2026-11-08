# Python MILP tree retirement: measurements

Plan and inventory: `docs/dev/python-milp-tree-retirement.md`.

## Step 1: which solves reach `_solve_milp_bb`, and why (2026-11-08)

`solver_stats["route/python_milp_tree_entry"]` now records the reason code
(`solver._PY_MILP_TREE_REASONS`) whenever the tree runs.

- **In-repo corpus** (`tree_entry_census.py`, 74 `.nl` instances, 20 s,
  `DISCOPT_LP_MILP_BACKEND=rust`; `tree_census_rust_20s.jsonl`): **0 of 74** solves reach
  the tree, MINLP lifts included. 74 executed, 0 errors.
- **MIPLIB-38+M panel** (route labels already in the D2 panels,
  `../highs-parity/d2-analysis/tol_robust_panel_v2_r3.json.gz`, 62 instances x 3 reps x 2
  arms): 31 of 372 solves reach the tree. Five instances arrive through the #291 stall
  handoff (`enlight_hard`, `neos-2624317-amur`, `2122`, `issue-2446`, `p0548`): every one ends
  at the time limit **with no incumbent**, so the tree has 0 wins on about 60 runs. The sixth,
  `fixed-binary`, has no rows (`no_constraints`); the Rust driver solves 0-row MILPs
  correctly (probe: `y = floor(4.5) = 4`, objective -11, certified).
- The stall handoff is also the mechanism behind `p0548`'s coin flip: when the driver
  finds an incumbent inside its 10 s slice it keeps the search; when it does not, the
  Python tree gets the rest and finds nothing.

Kill criterion (step 1): a reason with zero wins switches straight to the engine. Both
measured reasons qualify.
