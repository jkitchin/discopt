<!-- Owner direction 2026-11-08: port the useful parts of the Python MILP tree (`_solve_milp_bb`) to the Rust driver and retire the Python path. Inventory below compiled read-only (grep/read, nothing executed); items marked unverified are open. Spot-checked: the #2163 deferral (`solver.py` ~40397) and the #291 stall route. -->

# Retiring the Python MILP tree (`_solve_milp_bb`): inventory and porting plan

Branch `claude/happy-hamilton-0vvo3b`, read-only survey, 2026-11-08. All line numbers are
`python/discopt/solver.py` unless another file is named. "Rust driver" means
`crates/discopt-core/src/bnb/milp_driver.rs` (`solve_milp_node_search`, :2833; `MilpOptions`,
:1433). Nothing was run except grep/read; claims marked **(unverified)** were not checked by
execution.

---

## 1. Every route into `_solve_milp_bb`

`_solve_milp_bb` is defined at :40797 (ends :41996). It has **one call site in solver.py**
(:18992) and three outside it (`solvers/milp_pounce.py:137`, `solvers/mip_nlp.py:510,522`).
Every route below ends at one of those four calls.

### 1a. The `solve_model` MILP branch (:18818-19050). Single call at :18992

Reaching it requires `problem_class == MILP` and **no** `lazy_constraints`,
`incumbent_callback` or `cut_callback` (:18840). With any of those, the MILP goes to spatial
B&B (#748), not to this tree (:19136-19149). The call runs when the HiGHS route
(:18847-18866) did not return **and** the Rust engine either was not tried or deferred.

| # | Condition that lands in `_solve_milp_bb` | Code | Who |
|---|---|---|---|
| R1 | `milp_backend="native"`: `_want_engine` is False (`_lpm_backend != "native"`, :18884), HiGHS is skipped. Route label `native-milp: discopt MILP branch and bound (...)` (:19021-19027) | :18884-18890, :19021 | explicit opt-in (#1535; teaching/inspection; documented at :15059-15076, :38746-38752) |
| R2 | `node_callback is not None` with no backend named: `_observe_tree` (:18839) skips HiGHS (:18851) and the engine (:18886). Label `milp-tree: ... (node_callback set; ...)` (:19028-19032) | :18839, :18851, :18886 | explicit opt-in. `milp_backend="highs"` + `node_callback` is refused (:16042-16051) |
| R3 | `lagrangian_bound=True`: skips HiGHS (:18850). The engine is skipped unless `nlp_solver="simplex"` (:18889); with `"simplex"` the engine runs and ignores the flag with a warning (:18915-18920) | :18850, :18889 | explicit opt-in |
| R4 | `abs_gap_tolerance` given **and** the resolved `abs_gap_tol < gap_tolerance`: `_want_engine=False` (:18908-18919), because the engine's relative gap floors its denominator at 1.0 (`tree_manager.rs:1582`). HiGHS still takes it first when it is the backend, so this route fires only when `DISCOPT_LP_MILP_BACKEND=rust`, when `highspy` is missing, when HiGHS declined, or with `nlp_solver="simplex"` | :18908-18919 | opt-in parameter, but it hits the **default** route in the no-HiGHS install |
| R5 | `DISCOPT_MILP_ENGINE=0` (`_milp_engine_default_on`, :38665-38680) with HiGHS not taking the model (rust backend / no highspy) | :18889 | env opt-out ("the `_solve_milp_bb` path stays intact", :38675) |
| R6 | **Engine deferral** (`_solve_milp_simplex` returns `None`; the caller folds `_deferred["bound"]` and the stats into the tree's result, :19043-19046). Deferral sites in `_solve_milp_simplex` (:39962): | | **default path** whenever the Rust engine is the MILP engine (no `highspy`, or `DISCOPT_LP_MILP_BACKEND=rust`, or `nlp_solver="simplex"`, or the MINLP reformulations in R8) |
| R6a | the binding fails to import | :40010-40012 | unreachable in practice: `_solve_milp_bb` also needs the Rust `TreeManager` (`_tree_manager`, :40989) **(unverified that no build lacks just `solve_milp_csc_py`)** |
| R6b | `_milp_is_exactly_linear(model)` is False | :40023 | see the note under the table |
| R6c | **no constraints** (`_A_m == 0`) | :40039 | **default (rust route)**. This is the chimera_k64ising class (1192 ints, 0 rows), cited at :41007-41012. No deferred bound is recorded on this route |
| R6d | ReLU spec cannot be formed (`relu_cuts` is set but `_relu_native_engine` returns None) | :40044-40046 | not into the tree: the ReLU branch goes to spatial B&B on a decline (:19133-19149) |
| R6e | **#291 stall guard.** Run 1 is capped at `max(0.5, min(0.5·remaining, 10 s))` (:40067-40071, `_SIMPLEX_MILP_BUDGET_CAP_S=10.0` at :38590). When it ends `node_limit` with no incumbent, the function returns None and stores `deferred["bound"]`/`["stats"]` (:40613-40647). `DISCOPT_MILP_RUN1_CONTINUE` (default OFF, :39723-39739) moves the guard into the driver as `stall_time_s` (`milp_driver.rs:1457`); it still defers when no incumbent was found | :40613-40647 | **default (rust route)**. This is the main production route into the Python tree: every hard MILP the engine opens without an incumbent |
| R6f | **#2163**: the engine ends `feasible` with a non-finite bound (`withdraw_certificate`, `milp_driver.rs:2679`; `has_unscalable_tiny_entry`, :2459), not on nodes and not on a debugger stop. Its point is handed over as `deferred["incumbent"]` | :40388-40426 | **default (rust route)**, for the big-M tiny-entry class (#1380/#1414) |
| R6g | the engine's optimal/feasible point fails `_point_feasible` and no debugger stop is pending | :40427-40445 | default (rust route), defensive |
| R7 | **HiGHS route declined.** `_solve_milp_highs` returns None only for a model that is not exactly linear, or for a ReLU spec it cannot form (:39371-39392). The engine then also declines (R6b), and the tree runs with the label `the HiGHS route declined` (:19036) | :18863-18866 | default, but only through R6b |
| R8 | **MINLP reformulations** that rewrite `nlp_solver` to `"simplex"`: the binary-multilinear lift (:17637-17648) and the integer-bilinear/product big-M lift (:18175-18189). They take the engine and land in the tree on any R6 deferral. Inside a #1940 race **lift slice**, #1959 (`DISCOPT_IPX_RACE_STALL_EXIT`, default ON, :12232-12290) ends the slice at the deferral instead (:18962-18985) | :17637, :18180, :18962 | **default MINLP path** (ex12xx / nvs12 class) |
| R9 | `DISCOPT_P3_FORCE_CUT_PATH=1` (cert:P3.1c lever, default OFF, :13065-13087): skips the R8 `nlp_solver→"simplex"` rewrite, documented as "keeping the model on `_solve_milp_bb`". **Finding:** since #1229 (HiGHS default) and the engine graduating default-ON (:38665), `nlp_solver="pounce"` goes to HiGHS or to the engine, so this flag no longer reaches `_solve_milp_bb` unless `DISCOPT_MILP_ENGINE=0` is also set. Its docstring and the audit row (`docs/dev/flag-retirement-audit.md:761`) are stale **(unverified by execution)** | :13065, :17644, :18187 | experiment lever |
| R10 | `nlp_solver` other than `"pounce"`/`"simplex"` (`"ipm"`, ipopt, ...): the routing is the same as the default, but in the tree `prefer_pounce=False` (:19000), so `_gomory_enabled` turns GMI off (:37905-37912) unless `GOMORY_CUTS_ENABLED` is set. Pinned by `test_p1_milp_bb_soundness.py::test_ipm_alias_milp_routes_to_self_hosted_bb` (:406) | :18999-19000 | changes the tree's behaviour, not the route |

Note on R6b/R7: `classify_problem` returned MILP, yet `_milp_is_exactly_linear` is False. If
that combination is reachable, `_solve_milp_bb` runs on `extract_lp_data(model)` (:40845), the
same linear projection that #286 says silently drops nonlinear terms. The tree has no
linearity gate of its own. **Unverified** whether any model reaches it; if one does, it is a
soundness hole, and today the Python tree is where it lands.

### 1b. Outside `solve_model`

| # | Route | Code | Who |
|---|---|---|---|
| R11 | **Matrix-MILP "POUNCE" backend**: `milp_pounce.solve_milp` builds a `Model` and calls `_solve_milp_bb(..., 16, "best_first", max_nodes, t0, prefer_pounce=True)` with the **default `node_engine="pounce"`**, so node LPs run on the POUNCE IPM, not the simplex (:40866-40876) | `solvers/milp_pounce.py:99-146` | `get_milp_solver(backend="pounce")` or `prefer_pounce=True` (`solvers/lp_backend.py:141-152, 220-228`), reached from `milp_solver="pounce"` in AMP (`amp.py:2359`), OA (`oa.py:2995, 5051, 5423`), the GDP master (`gdp/algorithms/master.py:196`) and `milp_relaxation` (`_relax/milp_relaxation.py:785`). It is also the "auto" fallback when the simplex binding is missing (unreachable in practice, R6a). No in-tree caller passes `prefer_pounce=True` (grep) |
| R12 | **SHOT profile direct MILP**: `_try_direct_milp`. `milp_solver="pounce"` calls `_solve_milp_bb(prefer_pounce=True, node_engine="simplex")`; **`"auto"` (the default) and every other non-gurobi/non-simplex value** call `_solve_milp_bb(prefer_pounce=nlp_solver=="pounce", node_engine=...)` | `solvers/mip_nlp.py:483-530` (gate :700-706, dispatch :761-772) | `solver="mip-nlp"` with `mip_nlp_profile="shot"` on a model classified MILP, unless `direct_quadratic_routing="off"` |
| R13 | Tests and scripts that call `_solve_milp_bb` directly (§3) | | test-only |

---

## 2. Feature matrix: Python tree vs Rust driver

Legend: **=** equivalent exists, **~** partial, **✗** missing in Rust, **R+** the Rust driver
has it and the Python tree does not.

| Capability | Python tree (`_solve_milp_bb`) | Rust driver | Status |
|---|---|---|---|
| Node selection `strategy` | `best_first`/`depth_first`/`best_estimate`, passed to `_tree_manager(..., strategy)` (:40989), validated #1722 | Hard-coded `SelectionStrategy::BestFirst` (`milp_driver.rs:2906`). `bnb/pool.rs:10-19` already implements all three | **✗**, cheap: plumbing only |
| `branching_rule` | `pseudocost` (the tree default), `most_fractional`, `least_fractional`, `strong` (full SB over every candidate, probe count reported), via `tree.set_branch_hints` (`_milp_branching_hints` :40663-40740, applied :41477-41500) | Reliability pseudocost plus limited SB (`strong_branch`, `sb_max_cands`, `sb_node_budget`, `MilpOptions` :1513-1533; `strong_branch` fn :6016) | **~**. Pseudocost is equivalent. Most/least-fractional are missing. Full SB is approximable with unlimited `sb_max_cands`/`sb_node_budget` **(unverified as identical)** |
| `node_callback` (observe after each batch) | `_fire_milp_node_callback` (:40743-40794), called at :41533 after `process_evaluated`. `CallbackContext` carries node_count, incumbent_obj, best_bound (certified, user sense), gap, elapsed, `x_relaxation` and `node_bound` of the batch's best node (`None` if the batch has no relaxation). A raising callback is logged and the search continues | `MilpDebugHook` fires `AfterProcess` with total_nodes, incumbent, bound, gap, elapsed, batch ids/lineage (`MilpDebugState`, :151-185; fire sites :3668, :4508, :4706). No relaxation point or per-node bound in the state | **~**. A hook exists; the state lacks `x_relaxation`/`node_bound`, and the certified-bound semantics (`_certified_callback_bound`) would need mapping |
| `lagrangian_bound` / `lagrangian_frequency` | `LagrangianNodeBounder` (`decomposition/lagrangian/node_bounder.py`): root multipliers once (:41060-41075), then `max(LP, L)` per node every `freq` batches (:41446-41456). It solves an integer subproblem per node | none | **✗**. Measured value only on 5 tiny synthetic GAP instances: nodes off 13/11/25/101/89 vs on 1/1/1/31/15 (`tests/test_lagrangian_bb_hook.py:276-292`). No corpus panel found |
| Root cut loop | `_root_cover_cut_loop` (:38241), up to 5 rounds (:40961): knapsack cover + **conflict-graph clique** (`_extract_clique_edges` :38114, #1603 guard :40921-40928), GMI (only when `prefer_pounce` or `GOMORY_CUTS_ENABLED`, :37905), MIR (`_separate_mir_cuts` :38062). Every cut is screened against verified points (`_verified_cut_screen_points` :38190, #1603 backstop) | Cover (`lp::cover`), GMI off the native basis, aggregated c-MIR (`cmir_cuts`, :1493), cut selection, root rounds and time cap, slack-cut pruning, node cut pool (`MilpOptions` :1476-1503, :1553-1600). The default budget is 500 cuts / 50 rounds (`_milp_root_cut_budget`, :39521-39587) | **~**. **Clique cuts are missing**: no `clique` in `milp_driver.rs`; `presolve/cliques.rs` extracts edges but the driver does not use them. Rust has c-MIR, selection, pruning and a node pool, which the Python tree lacks (**R+**). Note that `_solve_milp_simplex` does not pass `cmir_cuts`, so the binding default `false` applies on the `solve_model` route (`lp_bindings.rs:1267`) **(unverified whether that is intended)** |
| `milp_cuts=False` | `max_rounds=0` (:40961) | `root_cuts=0` / `cut_rounds=0` binding kwargs | **=** (plumbing) |
| Root reduced-cost fixing | Root only, POUNCE duals plus a purified incumbent (`_root_reduced_cost_fixing` :37735, called :40884) | At **every node** (`reduced_cost_fixing`, :1527; `reduced_cost_fix` :6250, use :5463) | **R+** |
| Node bound propagation (FBBT) | none in the loop | `node_propagation` (:1519; :5075) plus root presolve | **R+** |
| Root LP bound seed | `tree.seed_root_bound` (:41005, #933) | The root bound is native | **=** |
| Root dive (continuous repair) | `_root_dive` (:38467, called :41077) | `try_dive_repair` (:7244); no-incumbent dive schedule `dive_stride` (:304-352); improving dive (:354-390) | **= / R+** |
| Node incumbent heuristics | Snap-fix-resolve at each node LP (`_maybe_inject_snapped` :41146 → `_pounce_snap_incumbent` :37267, POUNCE re-solve) | Rounding (`try_rounding*` :6970/:7058), randomized rounding (:1948), sub-MIP RINS/RENS (`try_sub_mip` :1699, default ON), dives | **R+** (snap-fix is not ported; Rust has more) |
| Stalled-node recovery | A stalled/unavailable node LP is re-solved with POUNCE on the original rows (`_recover_or_decertify` :41108 → `_pounce_recover_node_bound` :36259). If that fails, `_gap_certified=False` (C-47) | Unresolved-fathom floor (parent bound inherited, #598), #649 unscaled retry, Farkas-verified infeasibility (:5861-6015), `withdraw_certificate` (:2679) | **~**, by a different mechanism. Rust does not try an IPM re-solve; it floors or decertifies |
| Tolerance-integral claim (#1414) | `_integral_claim_branch_col` (:41187): if the integral realisation leaves the rows, sentinel the node as a non-exclusion and branch on the offending column (:41404-41414, hints :41504-41513) | Snapped-incumbent verification and fix-and-resolve (#2163 doc, `tests/test_2163_rust_engine_relu_bigm.py:12-16`; `snap_integral` :7533, `polish_integral_vertex` :7572). When it cannot resolve, the engine **withdraws the certificate** and the solve defers (R6f) | **~**. **The Python tree is the only engine that certifies this class**: on `min -x+3z, x<=1e12 z` the engine returns `feasible 0.0` with bound -inf and the tree certifies -7 (:40394-40400). This is the one Python-only capability with a cited measurement on a correctness case |
| Exit integral-realisation recovery (#1380/#1606) and #952 exit gate | `_integral_claim_recovery` (:41658), "repolished" (:41678-41694), raises on no realisation (:41731) or an infeasible point (:41756) | `_point_feasible` gate in the caller (:40427-40445), `snap_integer_columns` (:40447-40452), driver verification | **~**. Python refuses loudly; the engine path defers (R6g) |
| #1331 objective recomputed at the returned point | :41786-41815 | The driver recomputes the objective from `c` on a seed (:1537-1541) and the caller reports the engine `obj` | **~** **(unverified for node incumbents)** |
| #1383 stale-certificate check | `_withhold_stale_certificate` (:41955) | not applied on the engine route (:40536-40560 builds the result directly) | **✗ on the engine route (unverified whether needed)** |
| `max_nodes` | `total_nodes >= max_nodes` checked per batch (:41563). No incumbent gives `node_limit`; an incumbent gives `feasible` | `max_nodes` is a node-creation cap (:1438). Under `DISCOPT_MILP_ENGINE_FULL_BUDGET` (default ON, :39840) the caller makes run 1 uncapped unless the caller named a cap, then one total cap over both runs (:40088-40099, :40243-40249). Without an incumbent it **defers** instead of returning `node_limit` (:40613) | **~**. The semantics differ, and a no-incumbent node limit becomes a fallback |
| Absolute/relative gap | `_gap_values_converged`: abs `UB-LB <= abs_gap_tol` (default 1e-6, `_DEFAULT_ABS_GAP_TOL`) **or** rel with denominator `max(|ub|,|lb|,1e-10)` (:8341-8358) | rel denominator `max(|inc|,1.0)` (`tree_manager.rs:1582`), abs only when given (`gap_closed`, `milp_driver.rs:1427-1428`) | **~**. A different formula. Near-zero optima under an abs-tighter request go to Python (R4) |
| Time limit | Checked per node; unevaluated nodes `requeue_nodes` (:41381, :41438; highs-parity P0.7) | Per batch, per node and per LP deadline (`time_limit_s`, :1450) | **=** |
| `incumbent_time_extension` (#917) | `_extend_budget_for_incumbent` (:41306) | `_reclaim_withheld_budget` in the caller under full-budget (:40232-40240) | **=** (caller-side) |
| Open-frontier cap (#2092) | `_tree_max_open_nodes` (:40991) | `max_open_nodes` (:1444) | **=** |
| Incumbent seeding (`initial_point`) | Checks integrality, box (#1324) and rows, then `inject_incumbent` (:41014-41048). Also the engine's deferred point (#2163) | `initial_incumbent`, validated in the driver (`validate_seed_incumbent` :7146) | **=** |
| Debugger | Checkpoints ITER_START, AFTER_SELECT, **BEFORE_IMPORT (steer point: inject incumbent / branch hint, `debug/checkpoints.py:24`)**, AFTER_PROCESS, TERMINATED (:41320-41620) | IterStart, AfterSelect, AfterProcess, IncumbentFound, Terminated (`MilpCheckpoint` :132-145). No steer point | **~**. BEFORE_IMPORT steering is missing |
| Duals at the incumbent | `_mip_recover_relaxation_duals` (:41769) | The same helper in the caller (:40540-40556) | **=** |
| Root metrics (`root_bound`/`root_gap`/`root_time`) | tree snapshot (:41545-41556) | an extra integer-free LP solve (:40472-40530) | **=** |
| `bound_source` / anytime bound (#933/#1244) | `_finalize_reported_bound` with root-LP / untainted root snapshot fallback (:41906) | `bnb_tree` or None (:40553-40554) | **~** |
| `termination` (#1585) | `_search_termination` (:41571) | Named only for gap/node/time (:40562-40574) | **~** |
| solver_stats keys | `cuts/cover_clique`, `cuts/gomory`, `cuts/mir` (:41934-41936), `incumbent/repolished` (:41939), `budget/incumbent_extension_s` (:41946), `branching/strong_probe_lps` (:41950); plus `rust_time`/`jax_time`/`python_time` fields | `lp/iters`, `lp/driver_nodes` (:40135-40145), plus `milp_driver/iters`, `milp_driver/nodes` on a defer (:40642-40647). No cut counts reported | **~**. Tests and panels that read `cuts/*` need Rust counters |
| `batch_size` | Nodes per exported batch | internal batching | parameter goes away |
| `node_engine`/`prefer_pounce` (IPM node LPs) | `"pounce"` IPM nodes (R11 default) or simplex | simplex only | intentionally not ported (§4) |
| Lazy / fractional-node hooks, ReLU separator, `MilpHandle` | none (MILPs with callbacks go to spatial, #748) | `MilpLazyHook` (:236), `MilpNodeHook` (:287), ReLU (`solve_milp_relu_csc_py`), `MilpHandle` (`crates/discopt-python/src/milp_handle.rs`) | **R+** |

**Evidence for and against the Python tree as a fallback:**
- **Against.** After a #1940 lift-slice deferral the tree produced neither an incumbent nor a
  certificate in 77 of 77 runs (:12251-12257). On the highs-parity UC witness it ran after the
  driver deferred, found no incumbent in 120 s and overran the time limit by up to 47%
  (`docs/dev/highs-parity-plan.md:521-532, 560-567`; the overrun is since fixed). The engine
  beat the tree on a 10-instance MILP panel: 7/10 vs 6/10 solved, total wall 275→174 s
  (:38665-38672).
- **For.** #2163/#1414 big-M certification (:40394-40400). The #291 comment says the fallback
  "solves nvs12's integer-bilinear reformulation in ~0.4 s where this engine stalls"
  (:40616-40620). Which fallback engine produced that number is **unverified**: the comment
  says "spatial / POUNCE path", but the code falls into `_solve_milp_bb`.

---

## 3. Tests that pin Python-tree behaviour (`python/tests/`)

Grouped by what each would need when the tree is removed.

**A. Call `_solve_milp_bb` or its private helpers directly: delete or re-point**
- `test_debug_adversarial.py`: imports `_solve_milp_bb` (:26), drives it at :575, ":663 Loop 3 (_solve_milp_bb) is fallback-only, so drive it directly".
- `test_crossover.py:126-133` (IPM nodes in `_solve_milp_bb`), and `_root_cover_cut_loop`.
- `test_milp_tree_time_limit_per_node.py::test_tree_stops_within_one_node_of_the_limit` (frame name `_solve_milp_bb`, :54; patches `_solve_node_lp_simplex`).
- `test_pounce_node_solves_respect_deadline.py` (spies `_solve_milp_bb`, :94-101).
- `test_1959_race_stall_exit.py::test_the_python_tree_still_runs_outside_a_lift_slice` and others that monkeypatch `S._solve_milp_bb` (:94-108); `scripts/panel_1959_race_stall_exit.py:69-91`.
- `test_1456_presolve_work_inventory.py:137` (`_TERMINAL_ENTRIES` names it).
- `test_1535_milp_backend_native.py`: spies `_solve_milp_bb` (:75-86), unit-tests `_milp_branching_hints` and `_fire_milp_node_callback`. All 23 tests are #1535 native-tree API: `test_native_milp_runs_the_python_tree_with_the_callers_options`, `test_milp_cuts_false_skips_the_root_cut_loop`, `test_node_callback_fires_with_values_in_the_users_sense`, `test_branching_rules_change_the_search_but_not_the_answer`, `test_node_callback_on_a_default_milp_runs_the_tree` (route string `milp-tree`), `test_node_callback_reports_no_bound_for_a_batch_without_a_relaxation`, and others.
- Helper unit tests: `test_clique_cuts.py`, `test_cover_cuts.py`, `test_issue_1603_native_milp_cuts.py` (`_root_cover_cut_loop`, `_extract_clique_edges`, `_verified_cut_screen_points`, `_binary_mask`; 16 tests), `test_root_dive.py`, `test_bigm_route_coef_tighten_1414.py` (`_root_dive`, plus source-marker asserts for `_integral_claim_branch_col`/`set_branch_hints`, :245-262), `test_p1_milp_bb_soundness.py` (`_root_reduced_cost_fixing`; `test_milp_bb_certifies`, `test_default_milp_routes_to_self_hosted_bb` :399, `test_ipm_alias_milp_routes_to_self_hosted_bb` :406, `test_milp_pounce_only_is_highs_free`), `test_rust_crossover.py` (`_gomory_enabled`, `GOMORY_CUTS_ENABLED`), `test_1514_bound_handlers_propagate.py` (`_extract_clique_edges`).

**B. Route or env assertions that pin the Python tree**
- `test_milp_engine_routing.py`: `test_opt_out_keeps_the_python_path`, `test_lagrangian_bound_is_not_silently_rerouted`, and the `_merge_engine_bound` group (`test_tighter_engine_bound_is_kept_for_a_minimization` ... `test_deferred_dict_is_optional`), which exist only because a deferral falls into the tree.
- `test_issue_1722_strategy_validation.py::test_native_milp_tree_honours_strategy_without_warning`, and `test_rust_engine_route_warns_that_strategy_is_unused` (this one flips once strategy is ported).
- `test_lagrangian_bb_hook.py` (10 tests; `test_hook_sound_and_helpful_on_a_branching_tree` pins `DISCOPT_MILP_ENGINE=0`, `DISCOPT_LP_MILP_BACKEND=rust`, :293-294; `test_end_to_end_hook_matches_hook_off`, `test_hook_flag_noops_on_maximize_end_to_end`).
- `test_issue_1874_milp_engine_budget.py`, `test_952_miqp_incumbent_feasibility.py` (uses `DISCOPT_MILP_ENGINE`; exit-gate tests on the Python tree, :260, :442, :522-557).
- `milp_backend="native"` users: `test_issue_1617.py`, `test_1585_textbook_findings.py`, `test_issue_1603_native_milp_cuts.py`, `test_issue1754_native_lp_iters.py`, `test_1953_native_lp_ns_gate.py`, `test_lp_milp_highs_route.py`, `test_milp_tree_time_limit_per_node.py` (several are LP-native and unaffected; check each).
- `lagrangian_bound` also appears in `test_solver_root_bound_paths.py`, `test_highs_parity_p20_rust_route_presolve.py`, `test_rlt_root_bound.py`, `test_1680_book_review.py`, `test_1520_relax_classifier_handlers.py`, `test_1614_route_provenance.py`.

**C. Behavioural tests that silently depend on a deferral into the tree (rust backend)**
- `test_1380_integral_realisation.py::test_no_route_reports_an_objective_below_the_true_optimum[rust-*]` and `test_bigm_route_coef_tighten_1414.py::test_rust_route_certifies_the_true_optimum_with_the_flag_off` (`FLAG_OFF_CASES` up to M=1e12, :225-229, asserts `status=="optimal"`). By the #2163 note these are the R6f class. Which cases currently pass only via the tree is **unverified**; an `algorithm_route` probe would settle it.
- `test_1331_milp_incumbent_objective.py`, `test_1324_warm_start_injection.py` (`test_milp_bb_*`), `test_no_incumbent_certification.py`, `test_954_nlpbb_incumbent_feasibility.py`, `test_912_wall_budget_inventory.py:489`, `test_milp_simplex_dive.py:65`, `test_lp_qp_solvers.py:459-501`.
- Matrix "POUNCE" MILP (R11): `test_milp_pounce.py` (6), `test_milp_bb_pounce_no_jax_ipm.py`, `test_lp_backend_select.py`, `test_amp_simplex_backend.py`, `test_issue_1060_open_bounds_marshal.py`, `test_1236_gmi_drops_no_term.py`, `test_milp_simplex.py`, `test_pounce_bound_snap.py`.
- SHOT direct MILP (R12): `test_mip_nlp.py`, `test_1619_remaining.py`, `test_oa_internals_units.py` (`mip_nlp_profile="shot"`).
- Spatial/kernel `node_callback` tests (`test_callbacks.py`, `test_solver_node_paths.py`, ...) mostly target the spatial tree; only their MILP cases matter (`test_callbacks.py:876, :917` mention `_solve_milp_bb`).
- Benchmark scripts naming it: `discopt_benchmarks/scripts/{coverage_map.py:78, p3_1b_default_path_aggregation.py:99, p3_1c_cut_reachability.py:137, issue933_anytime_bound_entry_experiment.py:107}`.

---

## 4. Proposed sequence (smallest and highest-value first)

Each step is bound-neutral unless marked bound-changing (CLAUDE.md §5).

1. **Instrument before porting (entry experiment, §4).** Add a counter
   `solver_stats["route/python_milp_tree_entry"] = <reason>` at :18992, with a reason enum
   (R1-R12). Run the in-repo corpus plus the rust-backend tests in §3C with
   `DISCOPT_LP_MILP_BACKEND=rust` and count, per reason, how often the tree returns something
   better than the engine's deferred result (an incumbent, or `optimal`). Kill criterion: any
   reason with zero wins can switch straight to a refusal or an engine-only result. This
   tells us whether R6e (the #291 deferral) has *any* measured value; the 77/77 lift-slice
   result suggests not.
2. **R6c, no constraints.** Make `_solve_milp_simplex` handle `_A_m == 0`: the driver with a
   zero-row CSC, or a trivial per-column optimum (each column at its cost-optimal integral
   bound, a closed form). Parity test: chimera-style 0-row integer model, rust backend,
   `status=="optimal"`, the same objective as HiGHS. Switch R6c in the same PR.
3. **`strategy` → `MilpOptions.selection`.** Replace `SelectionStrategy::BestFirst` at
   `milp_driver.rs:2906` with an option, add a binding kwarg, and pass it from
   `_solve_milp_simplex`. Default BestFirst keeps it bound-neutral. Tests: re-point
   `test_issue_1722_strategy_validation.py` (the rust route no longer warns), and require
   `node_count` and objective identical for the default on a certifying panel.
4. **`milp_cuts=False` and `branching_rule`.** `root_cuts=0`. Add a `branch_rule` enum
   (`Pseudocost`, `MostFractional`, `LeastFractional`, `StrongFull`) at the driver's
   branching-variable choice. Report `branching/strong_probe_lps` from the driver. Parity
   tests: `test_1535_milp_backend_native.py::test_branching_rules_change_the_search_but_not_the_answer`
   and `test_milp_cuts_false_skips_the_root_cut_loop`, re-pointed at the engine (assert
   `cuts/*` counters from new driver stats).
5. **`node_callback` via an observer hook.** Extend `MilpDebugState` at AfterProcess with the
   batch's best relaxation bound and its structural point, or add a dedicated
   `MilpObserverHook` with the same zero-effect-when-absent rule. Wrap it as the
   `CallbackContext` that `_fire_milp_node_callback` builds today (sense mapping, certified
   bound, `None` without a relaxation). Tests: the 5 `node_callback` tests in
   `test_1535_milp_backend_native.py`. Then switch R2, and R1 for these options.
6. **Gap semantics, R4.** Add `MilpOptions.rel_gap_denominator` with value `Floor1` (today)
   or `MaxAbs` (Python's `max(|ub|,|lb|,1e-10)`). Select `MaxAbs` whenever
   `abs_gap_tol < gap_tol` is requested. That case is bound-changing only in the stopping
   test, and only for the requests that route to Python today. Tests:
   `test_1243_abs_gap_tolerance.py::test_a_near_zero_optimum_is_not_declared_converged_at_a_tightened_tolerance`
   and `test_1323_abs_gap_routes.py` on the engine. Switch R4.
7. **#1414/#2163 class (R6f, R6g): the one correctness-bearing port.** In the driver, where
   the tiny-entry decertification fires today, use the #1414 rule instead: sentinel the node
   as a non-exclusion (floor at the parent) and branch on the column whose rounding broke a
   row (`process_evaluated` already honours such a hint, `tree_manager.rs` "#1414 actionable"
   marker). Bound-changing, so flag it, default OFF, and run the §5 panel. Parity tests:
   `test_1380_integral_realisation.py[rust]`, `test_bigm_route_coef_tighten_1414.py::test_rust_route_certifies_the_true_optimum_with_the_flag_off`
   (M=1e12), `test_2163_rust_engine_relu_bigm.py`. Switch R6f once the panel passes. Until
   then a defer must become a loud uncertified result, not a silent Python run.
8. **Clique cuts (bound-changing, flag).** Feed `presolve/cliques.rs` edges into a clique
   separator in the root loop. Re-point `test_clique_cuts.py` and the #1603 tests
   (`test_issue_1603_native_milp_cuts.py::test_clique_edges_are_flat_binary_columns`,
   `test_non_binary_clique_edge_is_refused_loudly`) at the Rust separator. Port only if the
   panel is net-positive; otherwise drop it (Rust already has cover, GMI and c-MIR).
9. **Retire the #291 deferral (R6e) and R8's fallback.** If step 1 shows no wins, the driver
   returns its `node_limit`/`time_limit` result with the valid bound (the #1959 path already
   does this for lift slices, :18962-18985) instead of `None`. Tests: re-point
   `test_milp_engine_routing.py`'s `_merge_engine_bound` group (delete it with
   `_merge_engine_bound`/`_merge_engine_stats`, :38593-38650), and
   `test_1959_race_stall_exit.py::test_the_python_tree_still_runs_outside_a_lift_slice`.
10. **Matrix "POUNCE" backend (R11) and SHOT direct (R12).** Point `backend="pounce"` and
    `prefer_pounce` at `milp_simplex.solve_milp`, or refuse `"pounce"` as a MILP backend
    with a message naming `"simplex"`. Make `_try_direct_milp` "auto"/"pounce" call
    `_solve_milp_simplex`. Tests: `test_milp_pounce.py` against `milp_simplex`;
    `test_lp_backend_select.py`; `test_mip_nlp.py` direct-MILP cases.
11. **`milp_backend="native"` (R1)** maps to the engine with steps 3-5 options; route label
    `native-milp: discopt monolithic Rust MILP engine (...)`. Update the docstrings at
    :15059-15076 and :38746-38752 and `docs/dev/solve-routing.md:83-89`.
12. **Delete.** Remove the R5 opt-out (`DISCOPT_MILP_ENGINE=0` becomes a loud error or a
    no-op deletion; retire the flag), R9's `DISCOPT_P3_FORCE_CUT_PATH`, `_solve_milp_bb`, its
    exclusive helpers and `milp_pounce.py`.

### What deletion removes (approximate, from line spans)

| Item | Lines |
|---|---|
| `_solve_milp_bb` (:40797-41996) | ~1,200 |
| `_milp_branching_hints` + `_fire_milp_node_callback` (:40663-40796) | ~134 |
| `_solve_node_lp_simplex` (:37155-37263), the only caller is the tree (`_root_dive` uses it too, also exclusive) | ~109 |
| `_reduced_cost_fixing` + `_root_reduced_cost_fixing` (:37671-37848) | ~178 |
| `_binary_mask`, cover/MIR augment and separate, `_extract_clique_edges`, `_cut_loop_relaxation_x`, `_verified_cut_screen_points`, `_root_cover_cut_loop`, `_root_dive`, `_gomory_enabled` (:37849-38589, excluding the GMI helpers :37915-38028 that `_relax/lp_spatial_bb.py` still uses) | ~630 |
| `_merge_engine_stats`/`_merge_engine_bound` (:38593-~38650) and the deferral plumbing in `_solve_milp_simplex` | ~80 |
| Routing branches in `_solve_impl` (:18959-19046, partially) | ~60 |
| `solvers/milp_pounce.py` | 168 |
| **Total** | **~2,500 lines of Python** |

Several neighbouring helpers are shared and **stay**: `_solve_node_lp_pounce`,
`_pounce_snap_incumbent`, `_pounce_recover_node_bound`, `_round_fix_resolve_attempt`,
`_integral_claim_recovery` (used by `_solve_miqp_bb` :41997ff), `_decompose_eq_slack_form`,
`_tree_manager`, `_finalize_reported_bound`, `_withhold_stale_certificate`,
`_search_termination`. Note that `_solve_miqp_bb` is a structurally identical Python tree for
convex MIQP; it is out of scope here but a natural next target. Whether `PyTreeManager`
methods used only by this tree (`set_branch_hints`, `requeue_nodes`, `seed_root_bound`) can
also go is **unverified**: the spatial path may use them.

### Recommend NOT porting: refuse loudly instead (§3)

- **`node_engine="pounce"` / IPM node LPs (`prefer_pounce`).** CLAUDE.md and the code itself
  say an IPM is the wrong engine for linear node LPs (:18981-18986; #145 analytic-centre
  drift). `milp_solver="pounce"` for a MILP should raise, naming `"simplex"`.
- **`batch_size` for MILP.** An internal batching knob with no semantic content. It should be
  ignored with the existing "unused option" warning, or dropped.
- **`lagrangian_bound`.** Recommend **retire, as a refusal** (`ValueError` naming the
  standalone `solver="lagrangian"`/decomposition route) unless the owner wants to keep it.
  The only measurement is 5 synthetic instances in a test docstring, with no corpus panel.
  It needs an integer subproblem solve per node through a Python callback, and it
  silently no-ops outside linear minimization with coupling (:41066-41075). If kept, the
  port is a `MilpBoundHook` trait called back across PyO3 like `MilpLazyHook`, plus a §5
  panel.
- **Debugger BEFORE_IMPORT steering (inject incumbent or branch hint mid-search).** Port only
  if a debugger user needs it. Otherwise document the Rust checkpoint set as the supported
  one and refuse the `steer`/`import` checkpoint names on the MILP route.
- **POUNCE stalled-node recovery.** The Rust floor-or-decertify is sound. An IPM re-solve
  adds a POUNCE dependency to the default MILP path for no measured gain **(unverified: no
  measurement found)**.
- **`DISCOPT_P3_FORCE_CUT_PATH`.** Already unreachable for its stated purpose (R9); retire
  it per the §5 "Retired" state.

## Progress log

- **2026-11-08, steps 4, 6, 11 and the stall route.** The Rust driver takes
  `branch_rule` (`Reliability`/`MostFractional`/`LeastFractional`/`Strong`) and
  `rel_gap_max_abs` (the tree's `max(|ub|,|lb|,1e-10)` denominator), and reports per-solve
  `root_cut_rows` / `strong_probe_lps` (`_rust.milp_last_solve_stats()`, surfaced as
  `cuts/root_rows` and `branching/strong_probe_lps`). `_solve_milp_simplex` passes
  `branching_rule` and sets `rel_gap_max_abs` when `abs_gap_tolerance < gap_tolerance`, so
  **R4 is gone** (`test_milp_engine_tight_abs_gap.py`). **R1 is gone**:
  `milp_backend="native"` runs the engine (route
  `native-milp: discopt monolithic Rust MILP engine (milp_backend='native', ...)`); the
  `test_1535` branching and options tests are re-pointed at it. Tests of tree internals
  (`test_issue_1603_native_milp_cuts.py` clique/backstop/#1606-repolish,
  `test_milp_tree_time_limit_per_node.py`) reach the tree through the `DISCOPT_MILP_ENGINE=0`
  opt-out until deletion. **R6e (the #291 stall) is gone** by default:
  `DISCOPT_MILP_NO_STALL_TREE` and `DISCOPT_MILP_RUST_OPEN_DEFAULT_BOX` graduated together
  (open-box differential 148 checks / 0 violations, `data/python-milp-retirement/`).
  Remaining routes into the tree: `withdrawn_bound_2163` (`DISCOPT_MILP_RUST_COEF_ENTRY`,
  panels owed; it replaced the retired `..._COEF_RESCUE`), `lagrangian_bound`, the `DISCOPT_MILP_ENGINE=0` opt-out, R11 (matrix POUNCE
  backend), R12 (SHOT direct MILP), and engine declines (ReLU spec unavailable, feasibility
  gate failures).
- **2026-11-08, step 10 (R11, R12).** `milp_pounce.solve_milp` is now an alias for
  `milp_simplex.solve_milp` that raises a `FutureWarning` (the IPM-node tree is not
  ported, §4); `get_milp_solver(backend="pounce")` returns it and `prefer_pounce` no longer
  changes the engine, so `milp_solver="pounce"` in AMP/OA/GDP and `milp_relaxation` keeps
  working on the Rust engine. `_try_direct_milp` (SHOT profile) runs `_solve_milp_simplex`
  for `"auto"`, `"simplex"` and `"pounce"`; a deferral falls back to the SHOT loop.
  Tests that used `milp_pounce` as a *reference* would have compared the engine to itself
  (vacuous, CLAUDE.md §6) and now use exhaustive enumeration
  (`test_milp_simplex.py`, `test_amp_simplex_backend.py`); tree-internal tests of that
  backend are deleted (`test_rust_crossover` GMI/MIR-in-pounce-mode,
  `test_1236`'s POUNCE node-budget arm, the POUNCE node-deadline plumbing, replaced by a
  wall-clock check on the alias). **Capability not carried over, recorded:** the POUNCE
  backend seeded the root LP bound even under a ~zero budget (#15); at `time_limit=1e-6`
  the engine returns no bound. `"auto"` has always been the engine, so AMP's default
  never had that seeding; it was `test_milp_bb_pounce_no_jax_ipm.py`, now removed.
- **2026-11-08, step 12: the tree is deleted.** `_solve_milp_bb` (~1,200 lines) and every
  helper only it used are gone from `solver.py` (net -2,620 lines there, -3,900 with tests):
  the root cover/clique/MIR cut loop, `_root_dive`, Python reduced-cost fixing, the POUNCE
  node-LP path, the branching-hint and node-callback adapters, `_merge_engine_bound` /
  `_merge_engine_stats`, the GMI gate (`GOMORY_CUTS_ENABLED`). Every remaining hand-off
  now returns the engine's own result: the #291 stall returns no incumbent with its valid
  bound (the `DISCOPT_MILP_NO_STALL_TREE=0` opt-out and `DISCOPT_IPX_RACE_STALL_EXIT`, whose
  only job was to skip the tree, are removed); a withdrawn certificate (#2163) returns the
  verified incumbent `feasible` with no bound; a point that fails the model's rows returns
  `unknown` with neither point nor bound (`route/milp_engine_point_refused`). A decline
  for a missing binding or nonlinear terms on the MILP route raises (a routing defect).
  `DISCOPT_MILP_ENGINE` (the opt-out to the tree) is removed. Tests of tree internals are
  deleted; tests of user-visible properties are re-pointed at the engine. The #1414 witness at M >= 1e12 on the
  Rust route, which the tree used to certify, certifies on the engine because
  `DISCOPT_MILP_RUST_COEF_ENTRY` graduated (two §5 panels, `data/python-milp-retirement/`);
  with `=0` it returns the honest uncertified `feasible` instead.
