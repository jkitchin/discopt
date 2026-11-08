# P6: convex node-bound LP and the P1.2 rescue, measured

`probe_rows.jsonl` comes from `lp_switch_probe.py`, a pytest plugin, run over every
test that reaches the two sites (the 2067, 1605, 1953, 1611 and P1.2 rescue files,
`test_highs_optional`; slow ones included):

    LP_PROBE_OUT=rows.jsonl PYTHONPATH=<dir of plugin> pytest -p lp_switch_probe ... -m ""

- `site=cnb`: for each linearised node LP, the in-house simplex alone vs the old
  HiGHS-first source. 252 calls: Rust returned a dual on all 252, and the LP values
  agree within 3.0e-9 relative. #8 now uses the Rust simplex only.
- `site=p12`: the rust-first rescue (`solve_lp_std_rust_first`). 8 calls, 7 decisive on
  Rust. The one non-decisive call is the P1.2 test's own synthetic engine failure (it
  monkeypatches the Rust engine to return `error`), so no real rescue LP needed HiGHS.
  The fallback was removed; an undecided answer leaves the node open.
