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

## What a HiGHS-less install needs, today

Set `DISCOPT_LP_MILP_BACKEND=rust`, `DISCOPT_LP_STD_ENGINE=rust` and
`DISCOPT_CONVEX_ROUTE_OA_MASTER=simplex` (or `DISCOPT_CONVEX_MINLP_ROUTE=0`), and pass
`iis_engine="rust"`. Reaches 6–9 already work. #11's reader is in-house (P6); only its
HiGHS-imitating checker step is skipped. #13 (the LP-format reader) is in-house too. P6.2 adds the CI
lane that runs the corpus and `pytest -m smoke` with `highspy` uninstalled, which turns
this table into a measured one.
