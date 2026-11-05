# #2168 — column-eliminating presolve of the native spatial kernel's spec

Code: `python/discopt/_relax/spatial_presolve.py` (`presolve_spec`), wired into
`solver._try_native_spatial_kernel` on the MAIN kernel call behind
`DISCOPT_KERNEL_PRESOLVE` (default OFF). Plan: `docs/dev/performance-plan.md` §119.

## Why the entry runs at the kernel level

The issue's entry was to run GDPlib `cstr` through `Model.solve(gdp_method="hull")`.
The container this was built in has the PyPI `gdplib` 20.6.2, which has no `cstr`
module; installing gdplib from source (the `discopt_benchmarks[gdplib]` route) was
refused by the environment. So the entry runs on the captured kernel spec
`../2142-exact-hull/scip/cstr_homog_spec.npz` — the spec SCIP certifies in 7,470 nodes
and whose SCIP-presolved translation discopt's tree certifies in a full solve at
12,931 nodes (`../2142-scip-presolved/`).

`kernel_arm.py` builds the main call's options from the solver's own helpers (the same
defaults a `Model.solve` uses) and solves the spec with the native kernel. What it
cannot run is the Pyomo-model side: the NLP seed, the primal hook and the pricer.
`seeded` stands in for the seed by handing every arm `initial_incumbent =
3.0620145766187727`, the value of the verified incumbent that seed finds in the full
solve (a real feasible value, so a sound cutoff).

**Retracted (2026-11-05): "the harness tracks the full solve".** The full-solve runs
(`gdp_panel/`) put the presolved arm at 1.6168 / 2.2395 at 10k / 20k against this harness's
2.2822 / 2.7936, and the original arm at 0.6624 / 0.9314 against 1.4385 / 1.9044. The control
below matched, the presolve arms did not; the seeded harness overstates the full solve, and its
20k pass of the kill line is a harness result only. The full solve passes the cstr bar on its
"or certifies" branch instead (certified at 41,949 nodes, 374 s). The original claim was: the positive control `scip` (SCIP's presolved
model, `../2142-scip-presolved/cstr_homog_scippre_spec.npz`) reaches 2.9144 @10k
(full solve 2.8834) and certifies at 12,049 nodes (full solve 12,931).

## Entry results (deterministic, `PYTHONHASHSEED=0`)

Pre-registration: `PREREG.md` (kill: 20k bound <= 2.7875, the big-M default's 20k
bound, and not certified).

| arm | 2k | 10k | 20k |
|---|---|---|---|
| `orig` (captured spec), seeded | 0.3421 | 1.4385 | 1.9044 |
| **`pre` (stages 1-3), seeded** | 0.3412 | **2.2822** | **2.7936** |
| `scip` (SCIP's presolved model, control), seeded | | 2.9144 | certified at 12,049 |
| `orig`, unseeded | | 1.1888 | 1.6651 |

**Verdict (harness only; see the retraction above): passes the kill line**, by 0.006 (2.7936 > 2.7875). It does not certify at
20k; SCIP's model does. At 10k the presolve recovers 0.84 of the 1.48 bound gap
between the original spec and SCIP's model (57 %); at 20k, 0.89 of at least 1.16
(SCIP's model has certified by then, so at most 77 %).

What the presolve does on this spec (`presolve_spec` info, 0.03 s): 51 columns fixed
by the kernel's root propagation (SCIP: 51), 112 aggregated (SCIP: 115 aggregated + 9
negated, 8 of which are multi-aggregations), 1 product linearised, rows 779 -> 292,
free columns 234 -> 122 (SCIP's presolved model: 110 variables).

Probe `probe_rep.py`: the representative tie-break flipped to SCIP's (keep the higher
index): 10k 2.2766 vs 2.2822, neutral; the default stays.

## Stage 4 (multi-aggregation): built, measured, removed

The issue made stage 4 optional, to be added if 1-3 were killed. It was built anyway
(commit 921b450), because SCIP's model still certifies where 1-3 do not:

| stage 4 variant | 10k, seeded |
|---|---|
| none (stages 1-3) | 2.2822 |
| eliminate a non-output column into rows AND BLF sides (44 multi-aggregations) | 0.1719 (20k: 0.3280) |
| eliminate into rows only (46) | 1.7415 |
| implied-free columns only, as SCIP does | nothing qualifies |

Into a BLF side, a multi-column form makes the side's interval a sum of column
intervals, and every envelope built on it loosens. SCIP's eight multi-aggregations on
cstr eliminate the BLF *output* (`w := x62 - x67 - 0.99`, `w` implied free), which
the kernel cannot express: a BLF needs a single output column. Stage 4 was removed;
logs `karm_multi_10000_seeded.log` (rows only) and
`superseded/karm_multi_blfsides_*_seeded.log`.

## Files

- `PREREG.md` — pre-registration (written before any ON arm).
- `kernel_arm.py` — the harness; `queue.sh` — the batch that ran the seeded arms.
- `karm_<arm>_<nodes>[_seeded].log` — one run each; `runs.out` — exit codes.
- `superseded/` — the first `pre` arms, run before the OBBT-inheritance change
  (unseeded 10k 1.4186; seeded 10k 2.2822, identical to the final code).
- `probe_rep.py`, `probe_rep_10000.log` — the tie-break probe.
- `panel/` — the CLAUDE.md §5 corpus panel (`issue2115_graduation_panel.py --flag
  DISCOPT_KERNEL_PRESOLVE`).

## Outcome

Kept as a documented opt-out (owner decision, 2026-11-05), after the owner's GDP parity panel
and full-solve cstr runs (`gdp_panel/`): cert-clean throughout, cstr hull certifies with the flag
ON (41,949 nodes, 374 s), GDP panel net 0 against a pre-registered >= 2. See the audit row in
`docs/dev/flag-retirement-audit.md` and performance-plan §119.
