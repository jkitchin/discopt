# #2142 — discopt's tree on SCIP's PRESOLVED homog-hull model: pre-registration

Written 2026-11-05 before any discopt arm on the translated model was run (CLAUDE.md §4).

## Question

SCIP certifies discopt's captured homog kernel spec (`../2142-exact-hull/scip/cstr_homog_spec.npz`)
in 7,470 nodes; with presolve off it needs 48,521 and sits at discopt's 20k bound (2.25 vs
2.19). A spec-level emulation of the 106 doubleton aggregations recovered only 40 % of the gap
(2.43 @20k, killed). Is the rest of the gap *in the presolved model* (so a presolve that reproduces
SCIP's transformed problem would transfer it to discopt's tree), or in discopt's tree?

## Experiment

`translate.py`: parse SCIP's transformed problem (`../2142-scip-hull-ablation/presolved_homog.cip`,
110 vars / 174 conss) into a discopt `SpatialKernelSpec` over the same 285 column indices:
active SCIP vars keep their column; fixed / aggregated / negated columns are fixed to `[0,0]`
in the kernel and recovered by the cip's affine maps (`expand`); every product in a nonlinear
constraint becomes a term (square) or BLF (bilinear) output, re-using a linear var as the
output when the constraint is exactly `w = s * prod`, a fresh aux column otherwise.
Fidelity gates (any failure = the translation is wrong, not a result):
(F1) SCIP solves the translated spec to 3.0620073 +- 1e-5;
(F2) SCIP's optimum of the ORIGINAL spec, compressed, satisfies every translated row/term to 1e-6,
and the expanded translated optimum satisfies every original row/term to 1e-6.

`arm_pre.py`: cstr hull with `DISCOPT_HULL_EXACT_PERSPECTIVE=1`, deterministic, `PYTHONHASHSEED=0`;
a spy on `solve_spatial_tree_py` asserts the spec it receives equals the captured npz and
replaces it with the translated one (incumbent_value / primal_hook wrapped with `expand`,
expanded incumbent asserted feasible on the original spec). Arms: `pre` 10k, 20k, and 300 s
(not deterministic); control `base` 10k must reproduce 1.6107840610272945 bit for bit (§8).

## Kill criterion (pre-registered)

- **Pass (build a presolve):** `pre` 20k bound > **2.7875** (big-M default @20k) OR `pre`
  certifies within 300 s.
- **Kill (the gap is in the tree, not the model):** `pre` 20k bound <= 2.7875 and not certified
  in 300 s. Then the next step is a node-by-node comparison with SCIP on the presolved model.
- Soundness: bound <= 3.0620073 + 1e-6; incumbent >= 3.0620073 - 1e-5.

## SCIP presolver ablation (attribution, runs in parallel)

On the original spec (7,470 baseline), one presolver / cons-handler presolve step disabled at a
time, to name the reductions carrying the 6.5x.
