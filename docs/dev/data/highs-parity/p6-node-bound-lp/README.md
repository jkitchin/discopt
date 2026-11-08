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
