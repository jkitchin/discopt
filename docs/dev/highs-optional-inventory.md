# HiGHS as an optional dependency — inventory (highs-parity P6.1)

Status: **done (P6.3, 2026-11-08): `highspy` is the optional `discopt[highs]` extra.**
No default path needs it. Without it the pure-MILP route (#1 below) falls back to the
in-house MILP engine, and the explicit HiGHS opt-ins raise `MissingHighsError` naming the
extra. D2 (whether the in-house engine should be the MILP default *with* HiGHS installed)
stays open: HiGHS remains the MILP default where it is present, because the in-house
engine still closes fewer MIPLIB instances within a time limit (P4.9). This document lists
every path that can reach HiGHS and what a HiGHS-less install gets. Modelled on
`jax-optional-inventory.md`.

The mechanism is `python/discopt/_optional_highs.py`:

- `require_highspy(feature)` imports and returns `highspy`, or raises `MissingHighsError`,
  a `ModuleNotFoundError` with `name="highspy"` whose message names the feature and
  `pip install 'discopt[highs]'`. Any existing `except ImportError` behaves as it did when
  `import highspy` failed on its own.
- `highs_available()` answers without importing (`importlib.util.find_spec`; a
  `sys.modules["highspy"] = None` block counts as absent; not cached, because the tests
  that simulate a HiGHS-less install toggle it within one process).
- The two historical exception names are aliases of `MissingHighsError`:
  `lp_milp_highs.HighsUnavailable` and `milp_highs.HighsBackendUnavailable`.
- `import highspy` appears in exactly one module, `_optional_highs.py`
  (`grep -rn "import highspy" python/discopt` lists it alone, plus prose).

Importing `discopt.solvers.lp_milp_highs` or `milp_highs` does **not** import HiGHS: both
reach it lazily through `require_highspy`. Most modules that import from `lp_milp_highs`
use only its HiGHS-free helpers (`StdForm`, `primal_ray_verified`, `fbbt_box`,
`certify_lp_dual`, `phase1_infeasibility_proof`, ...) and are not HiGHS reaches.

## Reaches

"Default" is the shipped behavior with `highspy` installed. "Without highspy" is what a
HiGHS-less install gets today.

| # | reach (entry point) | default | without highspy | removed by |
|---|---|---|---|---|
| 1 | Pure-**MILP** route (`solver` dispatch → `lp_milp_highs.solve_milp*`) | HiGHS when `highspy` is installed (`DISCOPT_LP_MILP_BACKEND` unset or `=highs`) | the in-house MILP engine (P6.3: `_lp_milp_backend` falls back when the variable is unset; the route label ends `(default; highspy is not installed)`); an explicit `=highs` raises | fallback done (P6.3); the default *with* HiGHS installed is **D2**, still open (P4.9 gate missed: Rust 16/38 MIPLIB vs 36/38) |
| 2 | Pure-**LP** route | Rust (`DISCOPT_LP_BACKEND`, D1) | unaffected | done (D1) |
| 3 | `solve_lp_std` engine seam (`lp_std_engine`; `DISCOPT_LP_STD_ENGINE`) — the verified LP solve under the MILP/LP routes and `mccormick_lp`'s exact LP certification | **Rust** (default since P6.4, 2026-11-07); HiGHS only when the user passes `highs_options` or selects the HiGHS LP route | unaffected | done (P6.4) |
| 4 | Convex-MINLP route OA / LP-NLP-BB **master** (`solver._convex_route_oa_master`) | **Rust** (`simplex`, default since D3, 2026-11-07) | unaffected; `DISCOPT_CONVEX_ROUTE_OA_MASTER=highs` opt-in raises | done (D3, owner-accepted residual of 6 instances: plan D3 status) |
| 5 | #2069 independent **confirmation** of a HiGHS master (`milp_highs._confirm_lazy_master_independently`); Lagrangian `lazy_confirm` | only when the master is HiGHS | n/a (no HiGHS master) | follows #4; P5.3 memo |
| 6 | **Decomposition** masters (`lp_backend.get_decomposition_master_solver`) | Rust (P5.2) | unaffected; `DISCOPT_DECOMP_MASTER=highs` opt-in raises | done (P5.2) |
| 7 | NLP-BB **root cuts** LP (`_root_cuts.root_cuts_engine`; `DISCOPT_NLPBB_ROOT_CUTS_ENGINE`) | **Rust** (default since P6, 2026-11-08); `=highs` is the opt-out | unaffected; the `=highs` opt-out raises | done (P6). Panel over the 18 instance-configurations where the stage fires (3 reps, 60 s, interleaved; `data/highs-parity/p6-root-cuts-default/`): 708 bound/feasibility checks, 0 violations; certified runs 49 HiGHS vs 50 Rust; bounds equal or tighter within 1e-6 relative on 18/18 (Rust looser only on `flay02m`, by 2.9e-9 relative); nodes 0.982x, total wall 1.006x. Low-load re-run of the 7 slowest or most variable rows: wall 1.013x, nodes 1.009x. `tls2@nlpbb` sits at the 60 s cliff, and its certification flips between runs in both arms (HiGHS 3/6, Rust 2/6). At 120 s both certify 4/4, HiGHS at a median of 77.7 s and 257–359 nodes, Rust at 66.5 s and 159–189 nodes, so P1.4's "1.35x on `tls2`" does not reproduce. The largest per-row cost is `cvxnonsep_psig40r@nlpbb`: 1.4 s to 2.7 s, with a tighter bound |
| 8 | Convex **node bound** LP (`convex_node_bound`, #2067) | **Rust** simplex only (P6, 2026-11-08) | unaffected | done (P6). P1.3's kill (3 translated `clay0303hfsg` LPs 1.0-1.25e-6 looser) came from the Rust dual's norm (~200) on an LP whose ~8.4e8 objective constant cancels, not from a wrong dual: `_safe_bound` now also certifies the zero multiplier in both rungs and keeps the larger. P1.3's entry re-run with that pipeline (`data/highs-parity/p6-node-bound-lp/p13_entry_with_zero_dual.py`): 60 compared, 0 lost, 0 looser, Rust tighter on 10. An earlier "wrong-sign reduced cost" diagnosis was wrong and is retracted in that directory's README. |
| 9 | POUNCE `primal_infeasible` corroboration (`_linear_node_verified_empty`) and the multiplier-free MIQP node bound | the verified LP route on the in-house engine (P6.4); the `_highs_lp_verifier_available()` gate that sent HiGHS-less installs to weaker fallbacks was removed (P6) | identical (the route needs no HiGHS) | done (P6) |
| 10 | **IIS** engine (`compute_iis(iis_engine=...)`, `DISCOPT_IIS_ENGINE`) | **Rust** (default since P6.4, 2026-11-07) | unaffected; `iis_engine="highs"` opt-in raises | done (P6.4). The P3.2 residual (big-M `INT_TOL`, pigeonhole) is accepted; tests whose budgets were tuned on HiGHS pin `=highs` |
| 11 | `discopt.mipcc`: the MPS **reader** | in-house since P6 (`discopt._mps`, Rust `discopt_core::mps`, an exact reproduction of HiGHS's `readModel` -> `getLp`: 176/176 files identical, `docs/dev/data/highs-parity/mps_reader/`) | unaffected (the reader); the post-answer checker-settings re-solve, which imitates the competition's HiGHS checker, is skipped with a message and the doubly proven answer stands | done (reader). The checker is HiGHS-specific by design |
| 12 | Opt-in `milp_solver="highs"` for MIP-NLP / OA (`lp_backend.get_milp_solver(backend="highs")`) | opt-in only | raises when asked | stays an opt-in |
| 13 | CPLEX-LP-format (`.lp`, `.lp.gz`) **reader**: `discopt._lpfile.read_lp` and the benchmark loader `lp_milp_loader.read_instance` (which reached HiGHS for every non-MPS file) | in-house since P6 (Rust `discopt_core::lp_file`, HiGHS's staged reader ported stage by stage, objective Hessian included: 294/294 files and synthetic cases identical, kError included, `docs/dev/data/highs-parity/lp_reader/`) | unaffected; the loader reads `.mps` and `.lp` without HiGHS (its `highs_reference` oracle stays HiGHS by design) | done |

## What a HiGHS-less install gets (measured, P6.3)

Nothing to configure: `pip install discopt` (no extra) solves every default path
in-house. Measured 2026-11-08 with `highspy` blocked at import:

- `test_highs_optional.py::test_corpus_never_reaches_highs`: the 74 in-repo `.nl`
  instances at 10 s each, **74 solved, 0 errors, 0 attempts to import `highspy`**
  other than the 77 `highs_available()` probes.
- `pytest -m smoke`: green once the tests that exercise a HiGHS route skip
  (`requires_highs` / `pytest.importorskip("highspy")`); before that marking it showed
  23 failures and 3 collection errors, every one a test that asks for a HiGHS route
  explicitly (`milp_solver="highs"`, `highs_options`, `DISCOPT_LP_BACKEND=highs`, a module
  that pins `DISCOPT_CONVEX_ROUTE_OA_MASTER=highs`) or asserts the `highs-milp` route
  label, plus `test_1619`, now engine-aware.
- `.github/scripts/wheel_smoke.py --no-highs` (run by `release.yml` on the bare wheel,
  then `--highs` after installing `wheel[highs]`).
- `highs_options` without `highspy` is refused (`MissingHighsError`), not dropped with
  the "was not used" warning: they are a request for HiGHS.

CI's `Python without HiGHS` lane (`ci.yml` `python-nohighs`) runs the smoke tier and the
corpus guard with `highspy` uninstalled.
