# #2183: fixed-integer LP check, bounded verdict

These panels back the #2183 change to `_refutation_check` in
`python/discopt/solvers/lp_milp_highs.py` (helper `_fixed_integer_lp_cannot_refute`).
All runs: 2026-11-06, highspy 1.15.1, Linux, Python 3.13, base `5975672`. The load
average stayed under 1.6 throughout. No timing claims are made; every comparison below
is of status, certificate, objective and bound.

**Arms.** `new` is this branch. `old` is the same process with
`_fixed_integer_lp_cannot_refute` patched to return `False`, which is exactly the
pre-#2183 code path. The `elif pt is None` branch then declines, as before. The two
arms are interleaved per instance. The M-HiGHS panel instead ran the stashed (unmodified)
`lp_milp_highs.py` against this branch, through `lp_milp_panel.py`.

| panel | script | result |
|---|---|---|
| med_term_purchasing big-M (vendored Pyomo example) | `medterm_probe.py` | before: `feasible`, uncertified, bound 8358.999 (the issue's decline). After: `optimal`, certified, obj 6797.539701513318, bound 6798.198610641284 (oracle 6797.539701513444, max) |
| M-HiGHS (23 HiGHS `check/instances`) + i1183, `--arms H`, 60 s | `discopt_benchmarks/scripts/lp_milp_panel.py` | `mhighs_before.json` / `mhighs_after.json`: 24/24 certified on both. Status, objective, bound and `gap_certified` identical on all 24, `incorrect_count=0` on both |
| #1509-family substitute (156 instances) | `family_1509.py` → `family_1509.json` | 0 false certificates on both arms, 156/156 certified on both, 0 differences. The new predicate was never called: every own-integer LP came back `optimal` |
| gdp_parity big-M/hull, 14 of 24 instances buildable here, 60 s | `gdp_ab.py` → `gdp_ab.json` | The only cell where the new arm fired is med_term big-M (`feasible` → certified, same objective). All other cells are identical, except clay0203 big-M (see below) |

**The #1509 family could not be regenerated.** Its script (`scratchpad/i1509/panel.py`)
was never committed. `family_1509.py` rebuilds the family from the
`lp-milp-highs-routing-plan.md` description. It uses `test_1509`'s `_pwl_block`:
120 hand-written PWL MILPs over c ∈ {1e2, 1e3, 1e4, 3e4, 1e5}, K ∈ {4, 8, 12, 16},
log band ∈ {8.67e-8, 5e-7, 2e-6}, with sq-band taper on and off. It adds 36
`nonlinear_to_pwl` surrogates of `log`/`sqrt` at 3 segment counts. The oracle is c².
This is the strongest substitute available. On this HiGHS build, however, it produced
no false certificate on either arm. So it shows neutrality, not that the falsifier
still catches. That part rests on `test_1509_highs_milp_refuted_certificate.py`, whose
`test_bound_above_a_verified_point_is_refuted` injects a false HiGHS answer and still
passes. It also rests on the construction itself: the new arm accepts only when the
own-integer LP's NS bound already passes the margin test, and then the LP optimum
(≥ that bound) passes it too.

**gdp_parity coverage.** The installed gdplib is PyPI 20.6.2, older than the panel's,
and is missing `med_term_purchasing` and the data files of several models.
`biofuel`, `stranded_gas` and `syngas` lack data files, and `gdp_col`'s builder returns
`None`. These are listed in the run's `EXCLUDED` line. `batch_processing`,
`disease_model` and `med_term_purchasing` come from the vendored Pyomo examples
(`discopt_benchmarks/data/pyomo_gdp_examples`). The vendored `batch_processing` is
refused at big-M/hull reformulation (an unbounded disjunct body, #1617) on both arms.
Not covered here: cstr, ex1_linan_2023, hda, kaibel, methanol, mod_hens, modprodnet,
positioning, small_batch, spectralog. A cell can change only if it reaches
`solve_milp_std` and its own-integer LP returns `feasible`. In this run the cells that
reached `solve_milp_std` were jobshop_small, strip_packing_concrete, disease_model and
med_term_purchasing, each under both methods (8 cells). The new arm fired in one of
them: med_term big-M.

**clay0203 big-M** differs between arms, `old` `feasible` at 63.9 s and `new` `optimal`
at 52.3 s, with the same objective and 127 nodes on both. It is on the `nlp-bb` route and
never calls `solve_milp_std` (`milp_route_calls=0`), so the change cannot reach it. An
earlier, interrupted run of the same script certified it on both arms with identical
numbers. This is a 60 s wall-clock artifact.
