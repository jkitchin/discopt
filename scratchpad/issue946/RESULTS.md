# Issue #946 — measurements

Build: `pounce-solver 0.9.0` (PyPI wheel), numpy 2.4.6, Python 3.11.

## 1. Entry experiment — does the degeneracy reproduce here?

`entry_experiment.py`, two in-process arms of
`test_gbd::test_linear_objective_nonlinear_constraint`, differing only in the
recourse NLP's `bound_relax_factor`. Every `solve_nlp` return is tapped.

| arm | recourse x at y=0 | g(x) | multiplier | status | bound |
|---|---|---|---|---|---|
| default `bound_relax_factor` | `7.07e-05` | `+1.00e-08` (row violated) | `7.07e3` | optimal | −1.0000 |
| `bound_relax_factor = 0` | `6.93e-09` | `9.60e-17` | `9.82e7` | **iteration_limit** | **−2.0000** |

Reproduces the issue's table to the digit, including the 101 recourse solves the
stalled arm burns. 105 measurements executed.

Post-fix, arm B: **optimal, bound −1.0000, 3 recourse solves.**

## 2. Differential panel — 40 instances x 2 arms

`differential_panel.py` (deterministic family: binary and 0..2 integer first
stages, quadratic/linear objectives, conic and equality coupling, scale 1e-1 to
1e2, some degenerate at a first-stage point), oracle = monolithic `Model.solve()`
with the reference point's own constraint violation recorded. `compare_panel.py`
applies the CLAUDE.md §5 gates. 80 comparisons executed, gate exit 0.

| | default arm | `bound_relax_factor = 0` arm |
|---|---|---|
| certified `optimal` before | 38/40 | 33/40 |
| certified `optimal` after | 38/40 | **38/40** |
| recourse NLP solves before → after | 113 → 113 | **604 → 117** |
| wall before → after | 16.2 s → 15.9 s | **117.8 s → 16.7 s** |

* NEW unsound bounds: **0**. Certification regressions: **0**. Incumbent
  regressions: **0**. Bounds weakened: **0**.
* Gains: 5 instances `iteration_limit` → `optimal`, 5 bounds strengthened, 4
  incumbents improved (all four then match the monolithic optimum exactly).
* 13 of 80 runs return `|mu| > 1e6`, so the degenerate class is well covered.

### The one flagged instance, and why it is not this change

Seed 16, **default** arm, before *and* after identically: `bound = 128.7391 >
incumbent = 128.7322`. The incumbent violates the (degenerate) row by `1.0e-8`,
i.e. it is the slightly-infeasible point Ipopt's `bound_relax_factor` produces —
the #940/#945 dependency, not a bound defect. Confirmed pre-existing by running
the comparator before-vs-before, which flags the same instance with the same
numbers.

The same seed in the **exact** arm is the change working as intended: post-fix
GBD certifies `128.74599`, which is the true optimum of the *exactly* feasible
problem, while the monolithic reference reports `128.73218` from a point `1.0e-8`
outside the row. Verified by hand: at `y = (0,0)` the recourse collapses to
`x = 0` and `f(0) = 128.745989`.

### A hypothesis that was measured and killed

First read of seed 16 was "the 1.1e11-coefficient degenerate cut rows are
wrecking the master MILP's numerics, so the master returns a wrong bound".
`master_tap.py` taps every master `(c, A_ub, b_ub)` and brute-forces all 2^n
first-stage points against those rows by hand: the master's answer agrees with
its own rows exactly (`y=(0,0)`, `eta=128.745989`). The hypothesis is false, and
no cut-dropping-on-degeneracy logic was written on the strength of it.

## 3. Non-binary first stage (issue item 3)

`nonbinary_probe.py`, same recourse with a `0..3` integer first stage and
`bound_relax_factor = 0`: no integer L-shaped cut is available, so the outcome
stays the honest `iteration_limit` — but GBD now stops after 2 master iterations
(3 recourse solves instead of 101) and logs both the stall reason and the reason
no multiplier-free cut could be built.

## 4. Test batteries

* `pytest python/tests/test_gbd.py python/tests/test_946_* python/tests/test_benders*.py
  python/tests/test_decomposition_{adversarial,solve_equivalence,benchmarks}.py -m ""`
  → 109 passed, 4 skipped.
* `pytest python/tests/ -m smoke` → 909 passed, 15 skipped, 2 xpassed.
* `pytest -m slow python/tests/test_adversarial_recent_fixes.py` → 10 passed.
  (A first run of this file showed 2 failures — `carton7` and
  `large_dense_jacobian_no_crash`, both wall-clock-deadline assertions — while
  the differential panel was still running on the same box; the file took 298 s
  against a 178 s baseline. Re-run on an idle machine: 10/10 in 192 s. Neither
  test touches the decomposition code path.)
* `ruff check python/`, `ruff format --check`, `mypy` on the changed module: clean.
* Rust untouched.
