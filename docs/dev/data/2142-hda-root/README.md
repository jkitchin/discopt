# #2142 item 1: the hda "root-bound loss" under `DISCOPT_HULL_EXACT_PERSPECTIVE` was a misreading — hda MAXIMIZES

§115 kept the flag OFF because hda's root bound went 121140.98 -> 118678.93 with it ON and read
that as "looser". **hda's objective sense is `maximize`** (`gdp_baseline._spec("hda")`, Pyomo
`sense maximize`), so its dual bound is an UPPER bound and 118678.93 is the TIGHTER one.
Retracted in performance-plan §116 (CLAUDE.md §11).

Instrument (`lp.py`): the pure McCormick LP of the hull-lowered model at its declared box, no
tree and no FBBT (`MccormickLPRelaxer(m).solve_at_node(lb, ub)`), internal minimization of
`-f`:

| lowering | LP `lower_bound` (internal, = -UB) | published upper bound |
|---|---|---|
| eps-perspective (flag `=0`) | -121140.9826 | 121140.98 |
| homogenized (flag `=1`) | -118678.9293 | **118678.93** (tighter by 2.0 %) |
| homogenized + perspective-McCormick linear rows (experiment, not shipped) | -117453.4634 | 117453.46 |

Same numbers from the full solve (`root.py`, `n1_*.log`, `tl600_*.log`; identical at 120 s and
600 s, so deterministic and not time-limited). The degree-2 rows carry the change (`deg2_hda.log`:
homogenizing only the 37 degree-2 rows reproduces 118678.93; only the 10 degree-3 rows gives
121113.10).

Experiments tried and not shipped:

- **Keep the eps row as well as the homogenized row** (`keep_*.log`). On hda the bound came
  back None, and on cstr there were 7 nodes in 120 s with bound 0.0: the eps rows bring back
  the 1e27 magnitudes. Rejected.
- **Perspective-McCormick linear rows** (`pmc_*.log`, `n1_pmc.log`, `tl600_pmc.log`):
  `sum c_m W_m = rhs*y` with `W_m = y*prod t` relaxed by McCormick on the disjunct box.
  - hda: tighter still (117453).
  - gdp_col: LP bound unchanged (8663.16, `lpcol_*.log`).
  - cstr at 5,000 deterministic nodes: worse, 0.455 against 0.728 without the rows
    (`pmc_cstr_f1.log`, `nopmc_cstr_f1.log`).

  This is not net-positive, so it was not shipped. The code was reverted, and the
  measurement is recorded here.

A real defect was found and fixed along the way (`convex_err.py`, `convex_err2.py`). The ON
lowering calls `classify_constraint` mid-hull, and the declared-box memo in
`_relax/convexity/patterns.py` kept the pre-growth size. The solver's classification then
raised `IndexError: index 801 is out of bounds for axis 0 with size 801` and marked the whole
hda model convexity-unknown. The memo is now invalidated when the variable count changes:
3173/3493 rows are proven convex, against 0 before because of the raise. The fix did not move
hda's bound (`fix_f*.log`: 121140.98 / 118678.93, unchanged).
