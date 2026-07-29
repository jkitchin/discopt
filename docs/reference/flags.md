# Environment flags

<!-- GENERATED FILE — do not edit by hand.
     Regenerate with `python scripts/gen_flag_docs.py`. The source of truth is
     `discopt._flag_registry.FLAG_REGISTRY` plus the `SolverTuning` dataclass. -->

Every `DISCOPT_*` environment variable discopt reads, in one place.

## The truth table

All flags parse the same way (`python/discopt/_env.py`):

| value (case-insensitive, whitespace-trimmed) | result |
| --- | --- |
| `1`, `true`, `yes`, `on` | true |
| `0`, `false`, `no`, `off` | false |
| unset, or empty | the flag's default |
| anything else | `ValueError` naming the flag and the accepted values |

The Rust core (`crates/discopt-core/src/env.rs`) uses the same table; because a
solver kernel has no exception channel, an unparseable value there warns on stderr
and falls back to the default instead of aborting.

## Kinds

| kind | meaning |
| --- | --- |
| `graduated` | Default-**ON** after a differential panel; keeps its `=0` opt-out forever. |
| `parked` | Default-**OFF** opt-in: implemented and sound, awaiting graduation. |
| `permanent` | Infrastructure knob (budgets, paths, sockets, process lifecycle). |
| `debug` | Developer instrumentation or an entry-experiment lever. |

Per CLAUDE.md §5, new behavior ships default-OFF behind a `parked` flag and only
becomes `graduated` when a corpus-wide differential panel is both cert-clean and
net-positive.

## Flags (73)

### `graduated` (13)

| flag | default | side | issue | description |
| --- | --- | --- | --- | --- |
| `DISCOPT_CVX_DOMINATED_COLS` | `1` (on) | python | #879 | Dominated-cost-column upper bound inside the convex kernel. |
| `DISCOPT_HEURISTIC_GOVERNOR` | `1` (on) | python | G2 | Hit-rate-adaptive governor throttling primal heuristic sources. `=0` restores the pre-governor byte-identical behaviour. |
| `DISCOPT_HEUR_BUDGET` | `1` (on) | python | #347 | SCIP-shaped success-weighted contingent gating the heavy primal improvers (enumeration, RINS, local branching). `=0` restores always-on. |
| `DISCOPT_INCREMENTAL_MC` | `1` (on) | python | cert:T1.3 | Build the incremental McCormick LP (row-for-row self-validated) instead of cold-building the relaxation per node. |
| `DISCOPT_INTEGER_RATIO_PARTITION` | `1` (on) | python | — | Integer-ratio partitioning reformulation (graduated 2026-07-16 on a 66-instance differential panel). |
| `DISCOPT_LIFT_LOOSE_PRODUCTS` | `1` (on) | python | TD-A/T2.6 | Lift integer powers of transcendental univariates into `t == g(x)` auxiliaries. |
| `DISCOPT_LIFT_ZERO_SPANNING_FACTORS` | `1` (on) | python | — | Lift zero-spanning product factors during factorable reformulation. `=0` restores the byte-identical no-tagging behaviour. |
| `DISCOPT_LP_SPATIAL_FALLBACK` | `1` (on) | python | — | Allow the LP-spatial path to fall back to the generic spatial loop. |
| `DISCOPT_LU_DENSITY_ROUTE` | `1` (on) | rust | #602/#612 | Density-based routing between the sparse and dense LU factorizations. `=0` restores the historical dense-preferring routing byte-identically. |
| `DISCOPT_MILP_SWAP_RESEED` | `1` (on) | python | — | One-hot swap reseeding of the MILP incumbent on re-entry (graphpart family). `=0` is the opt-out. |
| `DISCOPT_QUBO_PRIMAL` | `1` (on) | python | #843 | Greedy-1opt + tabu local search seeding an incumbent on unconstrained binary quadratic models. Graduated 2026; `=0` restores the no-seed path. |
| `DISCOPT_ROOT_BUDGET_GATE` | `1` (on) | python | — | Refuse to launch a root heuristic NLP that cannot fit in the remaining wall budget. Primal-only, so skipping is always sound. |
| `DISCOPT_SEPARATION_LP_SIMPLEX` | `1` (on) | python | — | Solve separation / strong-branching LPs with the in-house warm simplex instead of a cold POUNCE IPM solve. `=0` restores the caller's backend. |

### `parked` (28)

| flag | default | side | issue | description |
| --- | --- | --- | --- | --- |
| `DISCOPT_ANALYTIC_SEPGRAD` | `0` (off) | python | — | Use the compiled analytic separation gradient instead of the JAX path (falls back to JAX on any construction failure). |
| `DISCOPT_CMIR_AGGREGATION` | `0` (off) | python | cert:P3 | Marchand-Wolsey aggregation c-MIR separator. Validity-gated, so enabling it can only add valid cuts. |
| `DISCOPT_COEF_TIGHTEN` | `0` (off) | python | — | Python coefficient strengthening at the root. Sound (the LP relaxation only shrinks) but not yet panel-graduated. |
| `DISCOPT_CONVEX_KERNEL` | `0` (off) | python | #798/#779 | Route certifiable convex MINLPs into the native convex kernel; the result is adopted only when it certifies optimality and verifies feasible. |
| `DISCOPT_CVX_NATIVELP` | `0` (off) | rust | #807 | Route convex-kernel node solves through the shared persistent LP (bounds-in-place dual-warm reoptimize) instead of a cold per-node solve. |
| `DISCOPT_ENTROPY_ATOM` | `0` (off) | python | — | Recognize `x*log(x)` on `x>0` and emit its exact 1-D convex envelope. |
| `DISCOPT_FBBT_SEED` | `0` (off) | python | consolidation-plan Card 3e | Seed the Rust root FBBT pass from the presolve orchestrator's running box instead of only the model's declared box (read Python-side in `_jax/presolve_pipeline.run_root_presolve`, forwarded as the `fbbt_seed_from_ctx` … |
| `DISCOPT_GP_MINLP` | `0` (off) | python | — | Geometric-programming MINLP fast path (`discopt.gp.solve_gp_minlp`). |
| `DISCOPT_G_CONVEX_CUTS` | `0` (off) | python | — | Inject cuts derived from g-convexity certificates. |
| `DISCOPT_LOGSUMEXP_ATOM` | `0` (off) | python | — | Emit the convex softmax-tangent OA for `log(sum exp(.))` instead of the loose concave `log` relaxation. |
| `DISCOPT_LP_FACTORIZATION_HARDENING` | `0` (off) | rust | #671 | Failure-triggered hardened retry: build the basis factor with a singular perturbation so a near-singular basis completes. |
| `DISCOPT_LP_SPATIAL_MIXED` | `0` (off) | python | #860 | Extend the LP-spatial fallback to mixed integer/continuous models. |
| `DISCOPT_MULTILINEAR_COUPLING_RLT` | `0` (off) | python | #721 | Objective-coupling RLT on top of the integer-multilinear reformulation. |
| `DISCOPT_NARROW_BOX_BRANCH` | `0` (off) | python | #732 | Convert a failed *branchable* narrow-box node into an open node instead of a sentinel fathom. |
| `DISCOPT_NATIVE_SPATIAL_KERNEL` | `0` (off) | python | #764 | Route the spatial B&B tree into the native Rust kernel. |
| `DISCOPT_NLPBB_ROOT_CUTS` | `0` (off) | python | — | Root cut loop on the NLP-B&B path. |
| `DISCOPT_NLP_NATIVE` | `0` (off) | python | — | Use the native (pyo3) NLP problem object instead of the Python evaluator; blocked on `PyNlProblem` being Send-safe. |
| `DISCOPT_NODE_PROBING` | `0` (off) | python | cert:P3 | Per-node probing on discrete variables (sound: contracts only on proven infeasibility); costs O(discrete) extra FBBT solves per firing. |
| `DISCOPT_NORM_ATOM` | `0` (off) | python | — | Emit the convex OA of `sqrt(sum t^2)` instead of the loose concave sqrt. |
| `DISCOPT_OBBT_ITERATE` | `0` (off) | python | #282 | Iterate root OBBT to convergence instead of a fixed round budget. |
| `DISCOPT_OBBT_TOPK` | `0` (off) | python | T2.5 | Scored top-k per-node OBBT de-gate; awaiting the differential + panel gates. |
| `DISCOPT_PRESOLVE_SUBSTITUTE` | `0` (off) | python | — | Solve from the presolved representation (substitution + postsolve chain) rather than copying bounds only. |
| `DISCOPT_PSD_QFORM` | `0` (off) | python | — | PSD quadratic-form convexity certificate. |
| `DISCOPT_RELENT_ATOM` | `0` (off) | python | — | Jointly-convex OA of the relative entropy `x*log(x/y)`. |
| `DISCOPT_ROOT_FIXPOINT_REPOOL` | `0` (off) | python | — | Re-separate the root cut pool after a root-fixpoint bound tightening. |
| `DISCOPT_SGO` | `0` (off) | python | #114/#741 | Signomial global fast path: spatial B&B on the certified log-domain DC envelope for mixed-sign signomials over a positive box. |
| `DISCOPT_TRIVIAL_PRIMAL` | `0` (off) | python | #827 | Seed the root with verified-feasible trivial points (origin, box centre, all-lb, all-ub) on pure-continuous models. |
| `DISCOPT_XEXP_ATOM` | `0` (off) | python | — | Recognize `t*exp(t)` on its convex region `t>=-2` and emit the exact 1-D convex envelope. |

### `permanent` (27)

| flag | default | side | issue | description |
| --- | --- | --- | --- | --- |
| `DISCOPT_CONVEX_KERNEL_BUDGET` | `120.0` | python | #798 | Wall-clock budget (s) for a convex-kernel attempt before falling back. |
| `DISCOPT_DECOMP_STORE` | _unset_ | python | — | Path to the decomposition-advisor outcome store (`RecordStore`). |
| `DISCOPT_DISABLE_JAX_CACHE` | `0` (off) | python | — | Skip enabling JAX's persistent on-disk compilation cache at import. |
| `DISCOPT_EAGER_IMPORTS` | `0` (off) | python | — | Import the whole solve path at `import discopt` instead of lazily. |
| `DISCOPT_GAMS_BENCHMARK` | `0` (off) | python | — | GAMS-link daemon: Benchmark mode: no solve cap, no lifetime cap, 30 min idle timeout. |
| `DISCOPT_GAMS_IDLE_TIMEOUT` | `600.0 (1800.0 under BENCHMARK)` | python | — | GAMS-link daemon: Seconds of idleness before the daemon exits. |
| `DISCOPT_GAMS_JAX_CLEAR_EVERY` | `0` | python | — | GAMS-link daemon: Drop JAX compilation caches every N solves; 0 disables. |
| `DISCOPT_GAMS_MAX_LIFETIME` | `3600.0 (0.0 under BENCHMARK)` | python | — | GAMS-link daemon: Seconds of total lifetime before the daemon recycles; 0 disables. |
| `DISCOPT_GAMS_MAX_RSS_MB` | `0` | python | — | GAMS-link daemon: Peak RSS (MiB) before the daemon recycles; 0 disables. |
| `DISCOPT_GAMS_MAX_SOLVES` | `500 (0 under BENCHMARK)` | python | — | GAMS-link daemon: Solves served before the daemon recycles; 0 disables. |
| `DISCOPT_GAMS_NO_DAEMON` | `0` (off) | python | — | GAMS link: solve in-process instead of via the warm daemon. |
| `DISCOPT_GAMS_SOCKET` | _unset_ | python | — | Explicit AF_UNIX socket path for the GAMS-link daemon. |
| `DISCOPT_HEUR_OFFSET` | `0.0` | python | #347 | Root contingent (sub-NLP-solve-equivalents) for `DISCOPT_HEUR_BUDGET`. |
| `DISCOPT_HEUR_QUOT` | `0.5` | python | #347 | Per-processed-node contingent accrual for `DISCOPT_HEUR_BUDGET`. |
| `DISCOPT_LLM_MODEL` | _unset_ | python | — | litellm model id for the optional LLM features (`discopt.llm`). |
| `DISCOPT_LP_SPATIAL_PLUNGE` | `require_incremental` | python | — | Depth-first plunging in the LP spatial B&B loop; unset defers to the caller's `require_incremental`. |
| `DISCOPT_MINLP_BENCH` | _unset_ | python | — | Path to the MINLPLib snapshot used by the benchmark harness and the corpus-drawing tests. |
| `DISCOPT_NODE_PROBE_MAX_VARS` | `32` | python | cert:P3 | Budget for `DISCOPT_NODE_PROBING`: discrete variables probed per firing. |
| `DISCOPT_ROOT_CUT_MAX` | `200` | python | — | Default for `solve_model(root_cut_max=...)`: root cut-pool size cap. |
| `DISCOPT_ROOT_CUT_ROUNDS` | `0` | python | — | Default for `solve_model(root_cut_rounds=...)`: root cut-pool rounds. |
| `DISCOPT_SOLVE_BENCHMARK` | `0` (off) | python | — | solve daemon: Benchmark mode: no solve cap, no lifetime cap, 30 min idle timeout. |
| `DISCOPT_SOLVE_IDLE_TIMEOUT` | `600.0 (1800.0 under BENCHMARK)` | python | — | solve daemon: Seconds of idleness before the daemon exits. |
| `DISCOPT_SOLVE_JAX_CLEAR_EVERY` | `0` | python | — | solve daemon: Drop JAX compilation caches every N solves; 0 disables. |
| `DISCOPT_SOLVE_MAX_LIFETIME` | `3600.0 (0.0 under BENCHMARK)` | python | — | solve daemon: Seconds of total lifetime before the daemon recycles; 0 disables. |
| `DISCOPT_SOLVE_MAX_RSS_MB` | `0` | python | — | solve daemon: Peak RSS (MiB) before the daemon recycles; 0 disables. |
| `DISCOPT_SOLVE_MAX_SOLVES` | `500 (0 under BENCHMARK)` | python | — | solve daemon: Solves served before the daemon recycles; 0 disables. |
| `DISCOPT_SOLVE_SOCKET` | _unset_ | python | — | Explicit AF_UNIX socket path for the solve daemon. |

### `debug` (5)

| flag | default | side | issue | description |
| --- | --- | --- | --- | --- |
| `DISCOPT_DISABLE_CSE` | `0` (off) | rust | — | Build the `.nl` expression arena with plain append instead of interning (evaluation-identical; quantifies the CSE node-count lever). |
| `DISCOPT_P3_FORCE_CUT_PATH` | `0` (off) | python | cert:P3.1c | Entry-experiment lever: skip the big-M `nlp_solver -> 'simplex'` reroute so the integer-product class stays on the cut-bearing `_solve_milp_bb` path. |
| `DISCOPT_PROFILE` | `0` (off) | rust | — | Enable the Rust core's internal phase profiler. |
| `DISCOPT_REDUCED_LP_BACKEND` | `simplex` | python | — | Backend for the reduced-space McCormick Kelley LP: `simplex` (default) or `scipy`. |
| `DISCOPT_T14_DBG` | `0` (off) | rust | T14 | Print the warm-basis accept/reject decision in the primal simplex. |

## `SolverTuning` fields

These 47 flags are the *legacy* spelling of a typed
`discopt.solver_tuning.SolverTuning` field. Prefer the object — it is
per-solve, thread-safe, validated, and discoverable:

```python
from discopt import SolverTuning
model.solve(tuning=SolverTuning(rlt_quad=False, node_bound_mode="milp"))
```

The env var supplies the field's *default* when it is not passed explicitly.

| flag | field | default | description |
| --- | --- | --- | --- |
| `DISCOPT_ADAPTIVE_NLP` | `adaptive_nlp` | `1` (on) | Adaptive back-off for the *strided in-tree node NLP* (`DISCOPT_ADAPTIVE_NLP`, **default ON** since G2 — flag-graduation convention: `=0` restores today's fixed `node_nlp_stride`). TX1 (`docs/dev/tenx-plan.md` §3). The … |
| `DISCOPT_ANYTIME_ROOT_BUILD` | `anytime_root_build` | `0` (off) | Make the root-relaxation *fallback* build anytime/incremental so its dual bound accrues and the build can honor the grant (`DISCOPT_ANYTIME_ROOT_BUILD`, default **off**; §5 bound-changing; issue #694). #654 left a … |
| `DISCOPT_CONTINUOUS_MULTISTART` | `continuous_multistart` | `1` (on) | Stratified continuous multistart at the root for pure-continuous nonconvex models (`DISCOPT_CONTINUOUS_MULTISTART`, default ON; issue #188). The primal-heuristic suite is integer-centric: on a model with no integer … |
| `DISCOPT_CUT_INHERIT` | `cut_inherit` | `0` (off) | Root-cut-pool inheritance for the per-node square/PSD separation loops — **tri-state, opt-in** (`DISCOPT_CUT_INHERIT`: unset / `0` ⇒ force-off = the shipped default; `gated`/`auto` ⇒ structure-gated opt-in; `1` ⇒ … |
| `DISCOPT_DISJUNCTIVE_CONFIG_BOUND` | `disjunctive_config_bound` | `0` (off) | Root disjunctive configuration bound for the gated-configuration class (`DISCOPT_DISJUNCTIVE_CONFIG_BOUND`, default **OFF**; #732 Stage 2). When the integer-multilinear reform (#707) applies, enumerate the reform's … |
| `DISCOPT_EDGE_CONCAVE` | `edge_concave` | `1` (on) | Edge-concave aggregation cuts (`DISCOPT_EDGE_CONCAVE`, default on). |
| `DISCOPT_ILS_SOLVE_CAP` | `ils_solve_cap` | `2` | Sub-NLP solve cap for the `integer_local_search` objective-descent (`DISCOPT_ILS_SOLVE_CAP`, **default 2 = ON** since ILS-DEFAULT, #530-followup). `integer_local_search._objective_improve` runs a first-improvement … |
| `DISCOPT_INTEGER_MULTILINEAR_REFORM` | `integer_multilinear_reform` | `0` (off) | Flow-aware exact linearization of *integer-multilinear* products — products of >=3 variable factors where every factor but at most one is integer- or binary-valued (declared or implied), e.g. `(c + … |
| `DISCOPT_LP_ITERATIVE_REFINEMENT` | `lp_iterative_refinement` | `0` (off) | When a node LP breaks down numerically (the hda-class ill-conditioned McCormick relaxations whose near-singular bases the float64 simplex cannot certify), recover a *tight* dual bound by re-solving a few RHS-regularized … |
| `DISCOPT_LP_WARMSTART` | `lp_warmstart` | `1` (on) | Warm-start the node LP from the parent basis (`DISCOPT_LP_WARMSTART`). |
| `DISCOPT_MULTILINEAR_RLT_MAX` | `multilinear_rlt_max` | `4` | Max arity for multilinear RLT lifting (`DISCOPT_MULTILINEAR_RLT_MAX`). |
| `DISCOPT_MULTILINEAR_SEPARATE` | `multilinear_separate` | `1` (on) | Separate multilinear McCormick cuts (`DISCOPT_MULTILINEAR_SEPARATE`). |
| `DISCOPT_NODE_BOUND_MODE` | `node_bound_mode` | `lp` | Per-node dual bound: `"lp"` (default, lifted-McCormick LP) or `"milp"` (legacy nested integer MILP node solve) — `DISCOPT_NODE_BOUND_MODE`. |
| `DISCOPT_NODE_NLP_STRIDE` | `node_nlp_stride` | `4` | Solve the node NLP every k-th node (`DISCOPT_NODE_NLP_STRIDE`, default 4). |
| `DISCOPT_NODE_NUMERICAL_DUAL_BOUND` | `node_numerical_dual_bound` | `1` (on) | Attach a Neumaier–Shcherbina safe lower bound from the in-house simplex's *own* dual candidate when the node LP solve breaks down numerically (`DISCOPT_NODE_NUMERICAL_DUAL_BOUND`, default ON since the #362 graduation — … |
| `DISCOPT_NS_SHARP_MARGIN` | `ns_sharp_margin` | `1` (on) | Replace the flat `1e-9`-relative Neumaier–Shcherbina evaluation margin with a rigorous forward-error bound computed from the actual data (Higham dot-product gammas + interval corners on sign-uncertain reduced costs) … |
| `DISCOPT_OBBT_CASCADE_AUX` | `obbt_cascade_aux` | `1` (on) | #208 reverse-FBBT aux cascade inside :func:`obbt_tighten_root` (`DISCOPT_OBBT_CASCADE_AUX`, **default ON** since the #208 graduation — `=0` restores the legacy OFF behaviour at every site). OBBT tightens the *lifted … |
| `DISCOPT_OBJ_BRANCH_PRIORITY` | `obj_branch_priority` | `1` (on) | Prioritize branching on objective-defining variables (`DISCOPT_OBJ_BRANCH_PRIORITY`, default ON). Graduated per T2.6 with 3 consecutive green held-out verdicts (composed with the density LU route): BR-3 #602 (verdict … |
| `DISCOPT_PHASE2_DBBT` | `phase2_dbbt` | `0` (off) | Per-node cheap reduced-cost DBBT + cutoff-FBBT (Phase 2, issue #764). `DISCOPT_PHASE2_DBBT` (**default OFF**, bound-changing / Regime-2). After each spatial node LP solve, run `reduce_node` — free duality-based bound … |
| `DISCOPT_PSD_COST_GATE` | `psd_cost_gate` | `1` (on) | Adaptive cost-aware gate on the per-node PSD (moment) cut separation loop (`DISCOPT_PSD_COST_GATE`, default **ON** since G1.3; `DISCOPT_PSD_COST_GATE=0` is the escape hatch). PSD separation dominates the QCQP root wall … |
| `DISCOPT_PSD_COST_GATE_BUDGET` | `psd_cost_gate_budget` | `1.0` | PSD wall budget per node as a multiple of that node's base LP-solve wall (`DISCOPT_PSD_COST_GATE_BUDGET`, default 1.0). The PSD loop stops once its cumulative wall this node exceeds `budget × base_solve_wall`. Only … |
| `DISCOPT_PSD_COST_GATE_TAU` | `psd_cost_gate_tau` | `0.0001` | Relative diminishing-returns threshold for the PSD loop (`DISCOPT_PSD_COST_GATE_TAU`, default 1e-4). A round whose LP-bound improvement `Δ ≤ tau × (1 + |lb_before|)` abandons the remaining PSD rounds at that node. Only … |
| `DISCOPT_RELAX_ROW_FILTER` | `relax_row_filter` | `1` (on) | **Failure-triggered**, **default ON** (#671 graduated 2026-07-18; opt out with `DISCOPT_RELAX_ROW_FILTER=0`). When a node LP breaks down without a certified verdict (`numerical`, or a spurious `infeasible` with no … |
| `DISCOPT_RELAX_SPACE` | `relax_space` | `lifted` | Per-node relaxation *space* for the McCormick dual bound (`DISCOPT_RELAX_SPACE`, MAiNGO-parity plan §2 P2.3). Values: - `"lifted"` (**default**, byte-identical to pre-P2.3): today's lifted McCormick LP with auxiliary … |
| `DISCOPT_RLT` | `rlt` | `0` (off) | Legacy whole-relaxation RLT toggle (`DISCOPT_RLT`). The `rlt=` argument to :meth:`Model.solve` is the primary control; this OR-s in alongside it. |
| `DISCOPT_RLT1_LAGRANGIAN` | `rlt1_lagrangian` | `0` (off) | Compute the RLT-1 root bound by the **Lagrangian dual** of the coupling rows instead of the monolithic LP (`DISCOPT_RLT1_LAGRANGIAN`, default off; §5). Same rigorous RLT-1 bound as :attr:`rlt1_root_bound`, but reached … |
| `DISCOPT_RLT1_LAGRANGIAN_MAX_ITER` | `rlt1_lagrangian_max_iter` | `300` | Subgradient iteration budget for :attr:`rlt1_lagrangian` (`DISCOPT_RLT1_LAGRANGIAN_MAX_ITER`, default 300). More iterations tighten the bound toward the RLT-1 optimum; each iterate is already a valid lower bound, so an … |
| `DISCOPT_RLT1_MAX_PAIRS` | `rlt1_max_pairs` | `60000` | Size guard for :attr:`rlt1_root_bound`: skip (sound no-op) when the all-pairs lift `n(n-1)/2` exceeds this (`DISCOPT_RLT1_MAX_PAIRS`, default 60000 — admits qap's 25200 pairs, blocks a runaway build on a much larger … |
| `DISCOPT_RLT1_ROOT_BOUND` | `rlt1_root_bound` | `0` (off) | Add an RLT level-1 lower bound at the root for constrained **binary** QPs (`DISCOPT_RLT1_ROOT_BOUND`, default off; §5 bound-changing). Term-wise McCormick envelopes on an indefinite `x'Qx` are trivially loose — every … |
| `DISCOPT_RLT_QUAD` | `rlt_quad` | `1` (on) | Quadratic RLT row generation (`DISCOPT_RLT_QUAD`, default on). |
| `DISCOPT_RLT_QUAD_MAX` | `rlt_quad_max` | `256` | Column cap for quadratic RLT (`DISCOPT_RLT_QUAD_MAX`, default 256). |
| `DISCOPT_RLT_SPARSE_AUTO` | `rlt_sparse_auto` | `0` (off) | Structure-aware widening of the RLT auto-engage gate for **sparse-bilinear** models (`DISCOPT_RLT_SPARSE_AUTO`, default **off**; issue #727). The default RLT auto policy gates build-time level-1 RLT and the per-node RLT … |
| `DISCOPT_RLT_SPARSE_MAX_TERMS` | `rlt_sparse_max_terms` | `300` | Product-term (lifted-column) budget for the sparse-bilinear RLT widening (`DISCOPT_RLT_SPARSE_MAX_TERMS`, default 300). Counts bilinear + trilinear + multilinear product terms; caps the RLT relaxation size directly, so … |
| `DISCOPT_RLT_SPARSE_MAX_VARS` | `rlt_sparse_max_vars` | `200` | Variable-count ceiling for the sparse-bilinear RLT widening (`DISCOPT_RLT_SPARSE_MAX_VARS`, default 200). Bounds the per-node re-solve cost of the enlarged relaxation when a model does not close at the root. Only … |
| `DISCOPT_RLT_SPARSE_MIN_ROOT_GAIN` | `rlt_sparse_min_root_gain` | `0.01` | Minimum relative root-bound improvement for the productivity gate to engage the sparse RLT widening (`DISCOPT_RLT_SPARSE_MIN_ROOT_GAIN`, default 1e-2 = 1%). The RLT-inert vs RLT-productive populations are ~10 orders of … |
| `DISCOPT_RLT_SPARSE_ROOT_PROBE` | `rlt_sparse_root_probe` | `1` (on) | Root-**productivity** gate on the sparse-bilinear RLT widening (`DISCOPT_RLT_SPARSE_ROOT_PROBE`, default **on** when `rlt_sparse_auto` is on; issue #727). Structure alone is *necessary but not sufficient*: a … |
| `DISCOPT_ROOT_BUILD_DEADLINE` | `root_build_deadline` | `1` (on) | Deadline the **base** root-relaxation `build_milp_relaxation` (`DISCOPT_ROOT_BUILD_DEADLINE`, default **ON** — GRADUATED per §5, set `=0` to opt out; §5 bound-changing; issues #832/#814). The #694 `anytime_root_build` … |
| `DISCOPT_ROOT_FIXPOINT` | `root_fixpoint` | `1` (on) | Run the cutoff-aware root branch-and-reduce fixpoint (cert:T2.3) at the end of iteration 0: iterate {FBBT-with-cutoff, OBBT/DBBT-with-cutoff} to a fixpoint on the root box, refreshing the root cut pool + incremental … |
| `DISCOPT_ROOT_LP_PROBE_TIGHT` | `root_lp_probe_tight` | `1` (on) | Probe the spatial-path McCormick LP relaxer over the FBBT/OBBT-**tightened** root box rather than the raw declared model bounds when deciding whether to keep it for the whole search (`DISCOPT_ROOT_LP_PROBE_TIGHT` … |
| `DISCOPT_SHOR_SDP_MAX_DIM` | `shor_sdp_max_dim` | `400` | Size guard for :attr:`shor_sdp_root_bound`: skip (sound no-op) when the moment-matrix dimension `n + 1` exceeds this (`DISCOPT_SHOR_SDP_MAX_DIM`, default 400 — admits qap's 226, blocks a runaway SDP on a much larger … |
| `DISCOPT_SHOR_SDP_ROOT_BOUND` | `shor_sdp_root_bound` | `0` (off) | Add a **strong-Shor SDP** lower bound at the root for constrained all-binary QPs (`DISCOPT_SHOR_SDP_ROOT_BOUND`, default off; root-only, §5 bound-changing). The global moment-matrix PSD constraint `M = [[1, x'],[x, X]] … |
| `DISCOPT_SHOR_SDP_TIME_LIMIT` | `shor_sdp_time_limit` | `120.0` | Wall-clock budget in seconds for the SCS solve behind :attr:`shor_sdp_root_bound` (`DISCOPT_SHOR_SDP_TIME_LIMIT`, default 120 — covers qap's ~86 s root solve). An early stop is sound: the safe dual bound is valid at any … |
| `DISCOPT_SOS1_SELECTOR_BRANCH` | `sos1_selector_branch` | `0` (off) | Spatially branch continuous SOS1 selectors before drilling aux-binaries (`DISCOPT_SOS1_SELECTOR_BRANCH`, default OFF; issue #196). A continuous one-of-N selector `s` (member of a selection row `Σ s_i = 1`, upper-coupled … |
| `DISCOPT_SPARSE_LARGE_LP` | `sparse_large_lp` | `0` (off) | Solve the per-node McCormick LP even when its lift exceeds the `_MAX_RELAX_DENSE_CELLS` dense-cell guard (`DISCOPT_SPARSE_LARGE_LP`, default off). The whole per-node path is now sparse — relaxation build (CSR) … |
| `DISCOPT_SQUARE_SEPARATE` | `square_separate` | `1` (on) | Separate tightened square (`x**2`) cuts (`DISCOPT_SQUARE_SEPARATE`). |
| `DISCOPT_TRILINEAR` | `trilinear_nested` | `0` (off) | Force the legacy nested-bilinear trilinear path (`DISCOPT_TRILINEAR=nested`; equivalent to the default unless another trilinear selector is explicitly set). |
| `DISCOPT_TRILINEAR_RLT` | `trilinear_rlt` | `1` (on) | Trilinear RLT rows (`DISCOPT_TRILINEAR_RLT`, default on). |

**Total: 120 flags** — 73 in the registry, 47 `SolverTuning` fields.
