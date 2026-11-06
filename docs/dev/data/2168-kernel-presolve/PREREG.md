# #2168 kernel presolve: entry experiment pre-registration

Written 2026-11-05 before any arm with `DISCOPT_KERNEL_PRESOLVE=1` was run (CLAUDE.md §4).

**Hypothesis.** Stages 1-3 of the issue's design (root-propagation fixings, fixed-factor
linearisation, doubleton aggregation with output/integer/objective-aware representative
choice), applied to the kernel spec of the MAIN kernel call, transfer most of the gain
discopt's tree showed on SCIP's presolved model (`../2142-scip-presolved/`: certified at
12,931 nodes) to the default solve.

**Evidence.** SCIP ablation: `donotaggr` == presolve off (50,269 vs 48,521 nodes, baseline
7,470); discopt's tree on SCIP's presolved model certifies at 12,931 nodes; the doubleton
identities alone (`arm_aggr.py`, lower-index representative, no output/term handling)
reached 2.4254 @20k.

**Experiment.** `arm.py`, GDPlib cstr, `gdp_method="hull"` (homogenized hull, default
ON), `deterministic=True`, `PYTHONHASHSEED=0`, `max_nodes` 10k and 20k, flag OFF vs ON.
The OFF 10k control must reproduce 1.6107840610272945 (§8, branch tree identity).

**Kill.** ON 20k bound <= 2.7875 (big-M default @20k) and not certified. If killed, add
stage 4 (multi-aggregation into BLF outputs) and re-measure before concluding.

**Soundness gates (any failure is a bug, not a result).** bound <= 3.0620073 + 1e-6,
incumbent >= 3.0620073 - 1e-5, the expanded incumbent verifies on the model (the solver's
`_native_kernel_verify_point` path, unchanged).
