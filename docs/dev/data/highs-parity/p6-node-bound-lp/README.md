# P6: convex node-bound LP and the P1.2 rescue, measured

`probe_rows.jsonl` comes from `lp_switch_probe.py`, a pytest plugin, run over every
test that reaches the two sites (the 2067, 1605, 1953, 1611 and P1.2 rescue files,
`test_highs_optional`; slow ones included):

    LP_PROBE_OUT=rows.jsonl PYTHONPATH=<dir of plugin> pytest -p lp_switch_probe ... -m ""

- `site=cnb`: for each linearised node LP, the in-house simplex alone vs the old
  HiGHS-first source. 252 calls: Rust returned a dual on all 252, and the LP values
  agree within 3.0e-9 relative. That compared LP values, not certified bounds, and none
  of these calls came from `test_1537_invariance`. Re-running P1.3's entry
  (`../p13-r4-dual/p13_entry.py`, 60 node LPs, certified bounds) on 2026-11-08 found 3
  `clay0303hfsg` LPs whose Rust-dual bound is 1.0-1.25e-6 looser than HiGHS's, so the
  first switch (68863423) was reverted. The cause and the fix are below; with the fix,
  `p13_entry_with_zero_dual.py` gives 60 compared, 0 lost, 0 looser, Rust tighter on 10,
  and #8 is Rust-only.
- `site=p12`: the rust-first rescue (`solve_lp_std_rust_first`). 8 calls, 7 decisive on
  Rust. The one non-decisive call is the P1.2 test's own synthetic engine failure (it
  monkeypatches the Rust engine to return `error`), so no real rescue LP needed HiGHS.
  The fallback was removed; an undecided answer leaves the node open.

## Why the 3 P1.3 LPs certified looser on the Rust dual

The three are the same translated `clay0303hfsg` node LP (`test_1537_invariance` shift
cell; 249 columns, 150 rows, boxes up to 3.6e6, objective constant ~8.4e8 that cancels
to an optimum of ~0):

| engine | LP value | max abs dual | cheap NS (y / 0) | exact correction (y / 0) |
|---|---|---|---|---|
| HiGHS | -1.5497e-6 | 0 | -8.428e-4 / -8.428e-4 | -1.5497e-6 / -1.5497e-6 |
| Rust | -2.5034e-6 / -2.0266e-6 | 200 / 150 | -8.441e-4 / -8.428e-4 | -2.8024e-6 / -2.5699e-6 // -1.5497e-6 |

- Both duals are optimal. HiGHS's is the zero vector; Rust's has norm ~200. The cheap rung
  pays ~8.4e-4 of rounding slack on either; only the exact correction gets near 0, and
  the correction of a norm-200 dual keeps ~1.25e-6 more slack than that of the zero dual.
- Fix (`convex_node_bound._safe_bound`): the zero vector is a second candidate in both
  rungs (weak duality makes any multiplier valid); the larger bound is kept. Neither
  candidate is skipped because the first looks "closed": closedness is judged against
  the engine's vertex value, which here (-2.50e-6) is below the true optimum, so y's
  exact bound looks closed while the zero dual's is 1.25e-6 tighter.

### Retraction (2026-11-08)

An earlier version of this section (commit 2d9d9253) said the Rust "optimal" basis
left 17 / 12 open-sided nonbasic columns with a wrong-sign reduced cost (|d| up to 300),
and named the Rust simplex's optimality on free and one-sided columns as the hardening
target. **That was wrong.** `p13_rc_optimality_check.py` judges each reduced cost
`d = c - Aᵀy` by where `x` sits (at its lower bound, at its upper, or strictly inside;
tol 1e-7(1+|c_j|)) on all 6 Rust answers in the panel's clay files: **0 violations** at
any position, 1494 columns checked, and the open-side-only count it used is also 0.
The earlier count was not reproduced (nor by flipping the dual's sign, which gives 5-6
columns, all at a finite bound) and its convention was not recorded. There is no
free-column optimality defect here; the looser bounds were the dual's norm.
