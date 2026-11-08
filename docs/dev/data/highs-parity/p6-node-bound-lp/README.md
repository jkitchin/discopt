# P6: convex node-bound LP and the P1.2 rescue, measured

`probe_rows.jsonl` comes from `lp_switch_probe.py`, a pytest plugin, run over every
test that reaches the two sites (the 2067, 1605, 1953, 1611 and P1.2 rescue files,
`test_highs_optional`; slow ones included):

    LP_PROBE_OUT=rows.jsonl PYTHONPATH=<dir of plugin> pytest -p lp_switch_probe ... -m ""

- `site=cnb`: for each linearised node LP, the in-house simplex alone vs the old
  HiGHS-first source. 252 calls: Rust returned a dual on all 252, and the LP values
  agree within 3.0e-9 relative. That compared LP values, not certified bounds, and none
  of these calls came from `test_1537_invariance`. Re-running P1.3's entry
  (`../p13-r4-dual/p13_entry.py`, 60 node LPs, certified bounds) on 2026-11-08 still
  finds 3 `clay0303hfsg` LPs whose Rust-dual bound is 1.0-1.25e-6 looser than HiGHS's,
  past the ~1e-6 certificate tolerance: P1.3's kill holds, so the switch of #8 to
  Rust-only (68863423) was reverted and HiGHS stays first there.
- `site=p12`: the rust-first rescue (`solve_lp_std_rust_first`). 8 calls, 7 decisive on
  Rust. The one non-decisive call is the P1.2 test's own synthetic engine failure (it
  monkeypatches the Rust engine to return `error`), so no real rescue LP needed HiGHS.
  The fallback was removed; an undecided answer leaves the node open.

## Why the 3 P1.3 LPs certify looser on the Rust dual (diagnosed 2026-11-08)

The three are the same translated `clay0303hfsg` node LP (`test_1537_invariance` shift
cell; 249 columns, 150 rows, boxes up to 3.6e6, 126-132 columns open on one side):

| engine | LP value | certified | max abs dual | open columns with wrong-sign reduced cost |
|---|---|---|---|---|
| HiGHS | -1.5497e-6 | -1.5497e-6 | 0 | 0 |
| Rust | -2.5034e-6 / -2.0266e-6 | -2.8024e-6 / -2.5699e-6 | 200 / 150 | 17 / 12 (\|d\| up to 300) |

- The optimum is ~0: both values are noise against these boxes, and the 1.0-1.25e-6
  gap is ~100x below any certification gap.
- The defect is the Rust "optimal" basis: it leaves columns that are open on one side
  nonbasic with a reduced cost of the wrong sign. An optimal basis has none, and HiGHS's
  has 0. The exact dual correction must then repair 12-17 columns and pays for it in
  bound.
- Hardening target before #8 can move: the Rust simplex's optimality on free and
  one-sided nonbasic columns (the #1061 class).
