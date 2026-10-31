# #1917 G4.2 `DISCOPT_GDP_ENUMERATE_CUTOFF`: 3-rep run on the 7 Pyomo GDPopt models

`run.sh`: `gdp_baseline.py --worker <instance> discopt-enumerate 300`, flag set in the
child's environment, arms interleaved and their order alternated per rep, main
`4517c7b` + this branch (`9020e43`). Load1 at most 1.70 throughout (4 cores, one solve
at a time). `rows.json`: one record per (instance, flag, rep) with the panel verdict
(`gdp_baseline.score` against `gdplib_runner.panel_oracle`).

| instance | OFF wall median (sd) | ON wall median (sd) | verdicts |
|---|---|---|---|
| strip_packing_concrete | 73.2 (0.7) | 23.7 (0.8) | solved 3/3 both |
| eight_process | 14.5 (0.4) | 5.7 (0.2) | solved 3/3 both |
| eight_process_logical | 17.9 (0.7) | 8.9 (0.2) | solved 3/3 both |
| constrained_layout_clay0203 | 128.8 (3.0) | 131.5 (9.0) | solved 3/3 both |
| ex_633_trespalacios | 1.4 (0.0) | 2.8 (0.1) | solved 3/3 both |
| nonconvex_hen | 1.3 (0.0) | 1.0 (0.0) | solved 3/3 both |
| jobshop_small | 0.3 (0.0) | 0.2 (0.0) | solved 3/3 both |

Cert-clean on all 42 runs. Net-positive on 4 of 7 (3x on strip_packing, 2.5x on
eight_process, 2x on eight_process_logical); neutral on clay0203 (its restricted
nonconvex subproblems are no cheaper to refute); 2x slower on ex_633 (1.4 s -> 2.8 s,
the restricted solves of 3 configurations cost more than the plain ones). Not a
graduation panel: 7 of the 24 `gdp_parity` instances (no git GDPlib here).

**Correction to commit `fd6a213`'s message.** It says the GDP test files
(`test_*gdp*`, `test_1775*`, `test_1917*`) pass. They do not all pass here: 1405
passed, 1 failed. The failure is
`test_1775_g0_4_pyomo_gdp.py::test_parity_checker_catches_a_wrong_row`
(`AttributeError: 'NoneType' object has no attribute '_parent'`, inside Pyomo's
`parent_block`). It fails identically on the base commit `4517c7b`, checked in a
worktree with `discopt.__file__` asserted, so it is an environment failure
(this container has Pyomo 6.10.1), not this change.
