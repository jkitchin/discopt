# #2142 §114: the 94 "SCIP-empty" cstr-hull boxes (performance-plan §114)

Boxes: `2142-affinepow-reverse/b1_capture.py OUT.npz 1000 10` (every 10th of the first 1,000
deterministic cstr-hull nodes; 94 boxes, 320 spec columns). Spec evaluator independent of SCIP:
`specio.py`. Every script prints an executed-assertion/propagation count (CLAUDE.md §6). SCIP
6.3's "SCIPrationalChgInfinity() not thread safe" stderr lines (printed once per model when
`numerics/infinity` is changed) are filtered out of the logs with `grep -v rational.cpp`;
nothing else is filtered.

| file | what |
| --- | --- |
| `c1_scip_opt.*` | SCIP optimum of the spec, verified row by row by numpy (residual 8.9e-7): x*, all four perspective binaries = 1 |
| `c2_instrument.*` | §113 instrument on boxes x* +- delta: non-empty down to 1e-6; empty at 1e-8 / 0 (x* is only 1e-6 feasible) |
| `c3_widen.*` | widening the 94 boxes by up to 1e-4: still 94/94 SCIP-empty (not a sliver effect) |
| `c4_minimal.*` | a minimal empty set: ONE column, col 318 = w316*w317 in [~1e16, 1e16] (both perspective w of binary 8 at 1e8) |
| `scip_prop2.py` | the §113 instrument with `big` = numerics/infinity / infinite-bound threshold |
| `c5_witness.*` | binary 8 = 0 is INFEASIBLE to SCIP at infinity 1e20 (presolve, 0 nodes) and feasible at 1e30 (verified point, obj 3.1336, col 313 = 1e24) |
| `c7_scip_hull.py`, `c7_*.log` | `2142-scip-expert/run.py` with env BIG: default at 1e20 reproduces "optimal, 1,772 nodes"; at 1e30 SCIP reports "optimal 3.1302" in 0 nodes (WRONG: oracle 3.0620, x* verified); emu @10k: dual 0.0, with nonlinear propagation off also 0.0 |
| `c6_boxes.*` | per box: §113 instrument 94 empty; at 1e30 91 empty; SCIP witness search at 1e30, verified by numpy: 2 boxes (0, 2) non-empty, both called empty by §113 |
| `c8_lsq_witness.*` | local least-squares witness search (independent of SCIP): finds nothing even where c6 verified a point -- too weak; recorded, not used |
| `c10_reduced.*`, `c10_reduced_cut.*` | exact ground truth with integral binaries: fix integers, forward-evaluate every lifted column whose operands are fixed, check against the box: **91/94 empty by an explicit violated value** (e.g. box 4: binary 6 = 0 forces w296 = 1e8, box has w296 in [9.517e7, 9.585e7]); box 2 has a verified witness; boxes 0/1 have free binaries (box 0 has c6's verified witness) |
| `c11_kernel_snap.*` | the kernel's own propagation: 0/94 empty as captured, **91/94** when the integer bounds are rounded first |
| `PREREG.md` | pre-registered E1/E2 for `DISCOPT_INT_ROUND_EXACT` |
| `e1_instrument.log` | E1: flag OFF 0/94, ON 91/94 (no pre-rounding) |
| `e_arm.py`, `e2_hull10k_f{0,1}.log` | E2: cstr hull, deterministic 10k nodes, OFF vs ON |
