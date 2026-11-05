# hull: presolve aggregation to certify cstr

*Draft follow-up of #2142 (not filed). Data: `docs/dev/data/2142-scip-presolved/`,
`docs/dev/data/2142-scip-hull-ablation/`, `docs/dev/performance-plan.md` §115-§116.*

## Goal

GDPlib `cstr` (oracle 3.0620073) certifies under `gdp_method="hull"` on the default path within
the gdp_parity budget (300 s). Big-M already certifies it (§112, ~85 s). Under the
division-free hull (`DISCOPT_HULL_EXACT_PERSPECTIVE`, §115-§116) the native kernel is reached
with magnitudes 1e3. It is not certified at 300 s (bound about 0.94-1.58 depending on load;
deterministic 1.61 @10k and 2.19 @20k nodes).

## Evidence

The model is the lever, not the tree. SCIP 10 (PySCIPOpt 6.3.0) on discopt's captured homog
kernel spec (`2142-exact-hull/scip/cstr_homog_spec.npz`, 285 cols / 779 rows / 5 square
terms / 29 BLF) certifies in **7,470 nodes / 12 s**. Its presolve reduces the spec to
110 vars / 174 conss: 51 columns fixed and 124 aggregated.

**discopt's own tree on SCIP's presolved model certifies.** `translate.py` turns the `.cip`
back into a kernel spec, and `arm_pre.py` swaps it in through a spy and expands every point
back. Results, deterministic:

| arm | 10k bound | 20k |
|---|---|---|
| discopt, original spec | 1.6108 | 2.1886 |
| discopt, 106 doubleton identities aggregated (`arm_aggr.py`) | 1.6998 | 2.4254 |
| **discopt, SCIP's presolved model** | **2.8834** | **certified at 12,931 nodes (43 s kernel)** |
| same, original rows mapped back in | | certified at 18,943 |
| same, discopt's own root-FBBT box | | certified at 12,087 |
| big-M default (reference line) | 2.2486 | 2.7875 |

The deleted rows and SCIP's bound tightening are therefore not the lever. The column
eliminations are. The doubleton-identity subset alone recovers only about 40 %.

SCIP ablations (nodes to certify, baseline 7,470):

| arm | nodes |
|---|---|
| presolve off | 48,521 |
| `donotaggr` | 50,269 |
| `cons_linear` presolve off | 60,371 |
| varbound presolve off | 12,617 |
| probing presolve off | 11,274 |
| conflict off | 29,538 |

discopt already has the conflict counterpart, dual proofs (§112).

## Which reductions matter

These are the ones `arm_aggr` lacked, read from the `.cip`:

1. **Products with a fixed factor become linear.** For example `w = x0 * x78` with `x0 = 1`
   becomes `w = x78`, which is then aggregated. Likewise a square or product whose input is
   fixed fixes its output.
2. **Aggregations whose representative is a term or BLF output.** When `x_j = a x_k + b` and
   `x_k` is an output, eliminate the non-output side instead. When both are outputs and
   `b = 0`, rescale one BLF side and merge (SCIP: `x160*x255 - 2*x284 == 0`). `arm_aggr`
   skipped all 5 such cases because it always kept the lower index.
3. **Multi-aggregations** `x_j = sum_k a_k x_k + b` (8 of them) into rows and into BLF
   affine sides. A BLF side is an affine form, so this is directly expressible.
4. **Negated binaries** `y' = 1 - y` (9).
5. **Two product definitions of one quantity onto one output column** (`x269`).
6. **Product outputs eliminated into the linear context.** SCIP keeps
   `-0.99 + x62 + x66*x160 - x225 == 0` with no separate output column. The kernel
   equivalent is one BLF output column shared by every row that used the eliminated copies.

## Proposed design

- **Module.** `python/discopt/_relax/spatial_presolve.py`:
  `presolve_spec(spec) -> (spec2, expand) | None`. It is called in
  `solver._try_native_spatial_kernel` on the MAIN kernel call only. The original `spec` stays
  for the pricer's `_kernel_lift_agrees_at`, `_kernel_max_open_nodes` and the probe calls.
  `incumbent_value` and `primal_hook` are wrapped with `expand`. LP points carry envelope
  columns after `n_cols`, so `expand` reads `x[:n_cols]`. `res["incumbent_x"]` is expanded.
- **Flag.** `DISCOPT_KERNEL_PRESOLVE`: a `SolverTuning` field, `_env_flag(default=False)`,
  plus a row in `docs/dev/flag-retirement-audit.md`.
- **Algorithm**, iterated to closure:
  1. `_rust.propagate_spec_py` on the global box (15 rounds). Fixed columns are those with
     `lo == hi` exactly.
  2. Substitute fixed columns into rows, BLF sides and term inputs:
     - a BLF with a constant side becomes two linear rows;
     - kind 0 with a fixed input becomes linear;
     - a kind-1/2/3/4 term with a fixed input fixes its output.
  3. Doubleton equalities: pairs of opposite parallel `<=` rows of support 2, plus support-1
     pairs, which give fixings. Then `x_j := a x_k + b`:
     - choose the representative so that outputs, integers and objective columns survive;
     - an affine input of a kind-1 term becomes kind 4 (kind 2 for `p = 2`);
     - a kind-0 term becomes a BLF;
     - reject `|a|` outside `[1e-3, 1e3]`;
     - for integrality, only `a = ±1` with integral `b` between integers.
  4. Multi-aggregation of equality rows into a BLF output (item 6): optional second stage,
     measured separately.
  5. Drop empty rows, rows implied by the box and parallel duplicates (keep the minimum rhs).
     This is optional, since the `prerows` arm shows it is not the lever, but it is cheap.
  6. Eliminated columns: `lo = hi = 0`, integrality 0, removed from `obbt_candidates`.
     `n_cols` is unchanged, so the kernel and verifier index space is stable.
- **Soundness.** Every substitution is an equality implied by globally valid rows or exact
  fixings, so the reduced problem equals the original on the FBBT box. The expanded point is
  re-verified by `_native_kernel_verify_point` (already on the pricer path).
- **Regime.** Bound-changing, so the full §5 panel applies: corpus-74, GDP parity big-M and
  hull, tanksize, nous1.

## Tests

`python/tests/test_kernel_presolve.py`:

- **(a)** `x0 = x1` encoded as two rows aggregates away, and `expand` restores it;
- **(b)** a negated binary, and `x_j = 2 x_k + 1` into a kind-1 square (becoming kind 4)
  and into a BLF side;
- **(c)** an output/output non-identity pair (rescale when `b = 0`, skip otherwise);
- **(d)** a BLF with a fixed factor is linearised;
- **(e)** a differential test: random feasible points of the reduced spec, expanded, satisfy
  the original; the original optimum, compressed, satisfies the reduced spec;
- **(f)** slow: cstr homog with the flag ON certifies within 20k deterministic nodes. The
  entry arm's number is 12,931 nodes on SCIP's model; the target is the deterministic 20k
  bound > 2.7875.

## Entry experiment and kill (pre-register before building)

Build stages 1-3 first as a spy, as `arm_aggr.py` did. Kill if cstr homog at 20k nodes is
<= 2.7875. If it is killed, add stage 4 and re-measure before concluding.
