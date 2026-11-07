# HiGHS as an optional dependency — inventory (highs-parity P6.1)

Status: **inventory and mechanism done; `highspy` is still a core dependency.** Making it
optional is P6.3, and that is gated on decisions D2 (the pure-MILP route) and D3 (the
convex-route OA master), both reopened by the owner on 2026-11-06 ("carry the plan to the
end, then bridge the gaps or accept them"). This document lists every path that still
reaches HiGHS, what it would cost a HiGHS-less install, and what removes it. Modelled on
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
| 1 | Pure-**MILP** route (`solver` dispatch → `lp_milp_highs.solve_milp*`) | HiGHS (`DISCOPT_LP_MILP_BACKEND=highs`) | `=rust` is the opt-in; the dispatch default would raise `MissingHighsError` | **D2** (P4.9 gate missed: Rust 16/38 MIPLIB vs 36/38) |
| 2 | Pure-**LP** route | Rust (`DISCOPT_LP_BACKEND`, D1) | unaffected | done (D1) |
| 3 | `solve_lp_std` engine seam (`lp_std_engine.HighsEngine`; `DISCOPT_LP_STD_ENGINE`, default `highs`) — the verified LP solve under the MILP/LP routes and `mccormick_lp`'s exact LP certification | HiGHS | `DISCOPT_LP_STD_ENGINE=rust` | flip the seam default (P1.x measured parity on L/M/INF) — P6.4 |
| 4 | Convex-MINLP route OA / LP-NLP-BB **master** (`solver._convex_route_oa_master`) | **Rust** (`simplex`, default since D3, 2026-11-07) | unaffected; `DISCOPT_CONVEX_ROUTE_OA_MASTER=highs` opt-in raises | done (D3, owner-accepted residual of 6 instances: plan D3 status) |
| 5 | #2069 independent **confirmation** of a HiGHS master (`milp_highs._confirm_lazy_master_independently`); Lagrangian `lazy_confirm` | only when the master is HiGHS | n/a (no HiGHS master) | follows #4; P5.3 memo |
| 6 | **Decomposition** masters (`lp_backend.get_decomposition_master_solver`) | Rust (P5.2) | unaffected; `DISCOPT_DECOMP_MASTER=highs` opt-in raises | done (P5.2) |
| 7 | NLP-BB **root cuts** LP (`_root_cuts._root_cuts_engine`; `DISCOPT_NLPBB_ROOT_CUTS_ENGINE`) | HiGHS when available, else Rust | Rust, automatically | P6.4 (Rust root LP measured 1.35x on `tls2`) |
| 8 | Convex **node bound** LP (`convex_node_bound`, #2067) | HiGHS, then the Rust simplex | Rust simplex, automatically (catches the error, records why) | none needed |
| 9 | `solver._highs_available()` WASM fallbacks (POUNCE `primal_infeasible` corroboration, multiplier-free node bound) | HiGHS path | in-house simplex / box-only bound (#1537) | none needed |
| 10 | **IIS** engine (`compute_iis(iis_engine=...)`, `_iis_backend.HighsBackend`, `_iis_reduce._highs_reduced`) | `iis_engine="highs"` | `iis_engine="rust"` (P3.2: 53/53 vs P0.6) | flip the default — big-M INT_TOL and pigeonhole gaps (P3.2) |
| 11 | `discopt.mipcc` (`read_mps`, the HiGHS checker) | HiGHS | `MissingHighsError` naming the feature | an in-house MPS reader; the checker is a HiGHS-specific tool by design |
| 12 | Opt-in `milp_solver="highs"` for MIP-NLP / OA (`lp_backend.get_milp_solver(backend="highs")`) | opt-in only | raises when asked | stays an opt-in |

## What a HiGHS-less install needs, today

Set `DISCOPT_LP_MILP_BACKEND=rust`, `DISCOPT_LP_STD_ENGINE=rust` and
`DISCOPT_CONVEX_ROUTE_OA_MASTER=simplex` (or `DISCOPT_CONVEX_MINLP_ROUTE=0`), and pass
`iis_engine="rust"`. Reaches 6–9 already work. #11 has no substitute. P6.2 adds the CI
lane that runs the corpus and `pytest -m smoke` with `highspy` uninstalled, which turns
this table into a measured one.
