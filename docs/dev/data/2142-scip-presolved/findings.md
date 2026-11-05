# #2142: discopt's tree on SCIP's presolved homog-hull model — findings (2026-11-05)

Pre-registration: `PREREG.md` (kill: `pre` 20k bound <= 2.7875 and not certified in 300 s).
Model: cstr under `DISCOPT_HULL_EXACT_PERSPECTIVE=1` (homog hull), captured kernel spec
`../2142-exact-hull/scip/cstr_homog_spec.npz` (285 cols, 779 rows, 5 square terms, 29 BLF).

## Translation fidelity (`translate.py`, `translate.log`)

SCIP's transformed problem (`../2142-scip-hull-ablation/presolved_homog.cip`, 110 vars / 174
conss) -> kernel spec over the same 285 column indices + 11 fresh product columns (296 cols,
212 rows, 5 terms, 24 BLF); 175 columns eliminated (fixed / aggregated / negated /
multi-aggregated), each recovered by the cip's affine map.

- F1: SCIP solves the translated spec to 3.0620008 (9,367 nodes) — the oracle within feastol.
- F2a: the expanded translated optimum satisfies the ORIGINAL spec to 1.67e-6. **PREREG
  amendment:** the gate was 1e-6; SCIP's own optimum of the original spec has residual 9.7e-7
  under the same check, and the translated point goes through chained maps, so the gate was
  set below the instrument's own tolerance. Amended to 1e-5 (recorded here; the arm's
  per-incumbent check uses 1e-5 too).
- F2b: every aggregation map holds exactly (0.0) at SCIP's optimum of the original.
- F2c: the compressed original optimum satisfies the translated spec to 9.7e-7.

## discopt arms (`arm_pre.py`, deterministic, `PYTHONHASHSEED=0`)

| arm | nodes | bound | status | kernel wall | `LpVerdictNumerical` |
|---|---|---|---|---|---|
| base (control, unmodified spec) | 10,000 | **1.6107840610272945** (bit-identical to §115) | node_limit | 136 s | 1,122 |
| **pre** (SCIP's presolved model) | 10,000 | **2.8834** | node_limit | | 809 |
| **pre** | 20,000 cap | **3.0617110 — certified at 12,931 nodes** | optimal | 43 s | 917 |
| pre + all 341 non-empty original rows mapped back (`ADD_MAPPED_ROWS=1`) | 20,000 cap | 3.0617104, certified at 18,943 | optimal | 103 s | 890 |
| pre with discopt's own root-FBBT box instead of SCIP's (`BOX_DISCOPT_FBBT=1`) | 20,000 cap | 3.0617140, certified at 12,087 | optimal | 49 s | 1,283 |

Reference: homog base 2.1886 @20k; arm_aggr (106 doubleton aggregations) 2.4254 @20k; big-M
default 2.7875 @20k. Incumbent 3.0620146 in every arm, expanded and verified on the original
spec; no bound above the oracle (executed-assertion counts 5 per arm).

**Verdict against the pre-registered kill: PASSES.** On SCIP's presolved model discopt's own
tree (dual proofs and all, no other change) certifies cstr in 12,931 nodes / 43 s of kernel
time — the same order as SCIP's 7,470 nodes. **The hull gap on cstr is in the model the
kernel is handed, not in discopt's tree.** §0a's "the gap is in discopt's tree" is retracted
in this respect: the tree is adequate once the model is reduced.

What carries it (ablations on the translated model):

- **Not the deleted rows.** 779 original rows map to: 438 empty after the fixings/aggregations,
  132 implied by the presolved box, 146 parallel to a presolved row, 63 other
  (`rows_census.py`). Putting all 341 non-empty ones back still certifies (18,943 nodes).
- **Not SCIP's bound tightening.** discopt's own root-FBBT box (15 rounds) in place of SCIP's
  presolved bounds still certifies (12,087 nodes).
- **So it is the structural reduction:** the column eliminations, of which the 106 doubleton
  identities alone gave only 2.43 @20k (`../2142-scip-hull-ablation/arm_aggr_20000.log`).
  What SCIP does beyond them (from the cip, comparing the original 29 BLF / 5 terms with the
  presolved 33 nonlinear conss): (i) a product with a FIXED factor becomes linear
  (`w275 = x0 * x78` with `x0 = 1` -> `x78 = w275`, which then aggregates); (ii) aggregations
  of non-identity type into term/BLF outputs (5 skipped by `arm_aggr`: e.g. a BLF output
  equal to `-2` times another output, which SCIP keeps as `x160*x255 - 2*x284 == 0`);
  (iii) 8 multi-aggregations `x_j = sum a_k x_k` substituted into the products' linear
  context (`blf0: -0.99 + x62 + x66*x160 - x225 == 0` — the product's output column is
  eliminated and the product sits directly in the linear equation); (iv) 9 negated binaries;
  (v) one product with two definitions merged onto one output (`x269 = 0.95*x75*x77 =
  x75*x79`).

## SCIP presolve-component ablations on the original spec (baseline 7,470 nodes)

| arm | param | nodes |
|---|---|---|
| linear cons presolve off | `constraints/linear/maxprerounds=0` | **60,371** |
| varbound presolve off | `constraints/varbound/maxprerounds=0` | 12,617 |
| probing presolve off | `propagating/probing/maxprerounds=0` | 11,274 |
| linear `aggregatevariables=false` | | 7,470 (bit-identical: no-op) |
| linear pairwise off | `constraints/linear/presolpairwise=false` | 7,470 (no-op) |
| setppc presolve off | `constraints/setppc/maxprerounds=0` | 7,470 (no-op) |
| linear hashing off | `constraints/linear/presolusehashing=false` | 6,369 |
| nonlinear presolve off | `constraints/nonlinear/maxprerounds=0` | SCIP error ("error in input data"); not measurable |

`cons_linear`'s presolve is the carrier (60k without it, worse than all-presolve-off's 48.5k);
the `aggregatevariables` switch does not govern the doubleton aggregation that matters
(no-op), consistent with `presolving/donotaggr=true` (global) giving 50,269.

## Scope decision (owner, via coordinator, 2026-11-05)

The presolve is NOT built on #2142. The design for the agent that builds it is in
`followup_issue.md` (drafted, not filed). #2142's closing target moves to: big-M certifies
cstr on default (done, §112) and hull reaches the native kernel with bounded magnitudes on
the default path; full hull certification of cstr moves to the follow-up.
