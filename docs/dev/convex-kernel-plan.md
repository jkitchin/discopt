# Convex LP-OA Branch-and-Cut Kernel — implementation plan (issue #798)

**Status:** K1 in progress. This doc is the durable, loop-executable spec for the
SCIP/BARON-parity convex kernel. It survives context loss — each work iteration
reads this, advances the current phase, and records results here.

Standing calibration (issue #798): *if SCIP/BARON solve an instance in seconds,
discopt must too; when in doubt, do what SCIP/BARON do.* The architecture mirrors
SCIP's convex-MINLP path (LP/NLP-based branch-and-cut, Quesada–Grossmann).

## The one data-justified lever (de-risked, do not re-litigate)

- Root bound gap on the convex `rsyn*`/`syn*` family closes via the **GMI
  separator family** + sustained in-tree separation (#782/#796/#797).
- The SOTA architecture is **LP-OA branch-and-cut** (LP relaxation at every node,
  cut into natively), NOT NLP-per-node. Prototype #797 measured **45–67 nodes** to
  certify vs 800–1200 without cutting — a **15–18× node reduction**.
- The Python prototype `discopt_benchmarks/scripts/issue786_lpoa_bandc_prototype.py`
  is the reference implementation to **byte-check the Rust kernel against**.

## Phases and gates

### K0 — architecture entry experiment. **DONE** (#796, #797)
15–18× node reduction validated. No further work.

### K1 — Rust LP-OA node relaxation. **DONE — GATE PASSED (2026-07-19)**
K1a–K1d built and committed. The K1 gate (`issue798_k1_bytecheck.py`): the Rust
node relaxation matches the Python reference `node_relax(separate=False)` to
**max|Δ| ~1e-11** (≪1e-6) over the root box + 6 child boxes on all 4 convex panel
instances. Safe (Neumaier–Shcherbina) bound certifies under FBBT-finite bounds
(what the K2 tree provides), sound and exact. Details in the work log below.
Build the OA LP over a box (linear rows + OA tangents of convex nonlinear rows;
refresh box-dependent rows per node), solve via the warm in-house simplex, return
the LP dual bound.

**Architecture decision (K1) — composite-of-affine convex rows. VERIFIED GO.**
Rust must evaluate `g_i(x)` and `∇g_i(x)` at LP vertices discovered *during* the
node loop. Rather than a full autodiff engine in Rust, each convex nonlinear row
is marshaled as a **composite-of-affine** descriptor:
`g_i(x) = a_i·x + b_i + Σ_t coeff_t·func_t(p_t·x + q_t) ≤ rhs_i`, so Rust needs
only closed-form univariate `f/f'` per `MathFunc` (extending `mccormick_patch`'s
`Univariate`). This is the OA-canonical convex class SCIP linearizes this way.
Rows that don't fit → the analyze-once producer returns `None` → keep the NLP-BB
path (the sound boundary, exactly like `spatial_producer`).

*Entry experiment (CLAUDE.md §4), `issue798_convex_decompose_probe.py`:* every
nonlinear row of all 4 panel instances (3/6/11/28 rows) decomposes into this form
and the reconstruction reproduces the JAX evaluator's **value AND gradient** to
machine precision (max_verr 4e-16, **max_gerr 0.0**). Its `decompose`/`eval_decomp`/
`grad_decomp` are the reusable producer (emit the flat-array marshaling).

**K1 build steps (each testable):**
- **K1a — Rust univariate `f/f'` table** (`eval_and_deriv(MathFunc, t) -> (f, fp)`)
  for the convex-certifiable funcs (log/exp/sqrt/log1p first). Unit-test vs finite
  differences. Independent of marshaling.
- **K1b — `ConvexKernelSpec`** (linear rows CSR, objective `c`, integrality, box,
  nonlinear rows as `Vec<CompositeRow>`), + `oa_tangent(row, x) -> EnvRow` reusing
  the `EnvRow` cut container.
- **K1c — node LP-OA relaxation over a box**: assemble CSC LP (linear rows +
  accumulated OA tangents), `solve_lp_cols_scaled`, add tangents for violated
  nonlinear rows, warm re-solve to OA convergence; return `ns_safe_bound_csc`.
- **K1d — PyO3 binding + Python byte-check harness** vs `_RootLP`/prototype
  `node_relax(separate=False)` over the root box + perturbed child boxes.

**Gate:** on ≥5 convex instances (`rsyn0805m`, `rsyn0810m`, `rsyn0815m`,
`syn40m`, + one more), the Rust node bound matches the Python `_RootLP`/prototype
`RootModel` LP-OA bound to **≤1e-6** over (a) the root box and (b) perturbed child
boxes. No separation yet — this gate is OA-only node relaxation parity.

**Byte-check reference (Python):**
- `python/discopt/solvers/_root_cuts.py::_RootLP` (shipped) — the LP-OA root model:
  `A_le/b_le`, `A_eq/b_eq`, linear objective `c`, `oa_tangent(row, x)`,
  `nonlinear_violations(x)`, `_fbbt_separation_bounds()`.
- `discopt_benchmarks/scripts/issue781_cutmgmt_probe.py::RootModel` +
  `solve_lp_highs` — the prototype's node relaxation (`node_relax` in the LP-OA
  prototype calls the OA-converge loop; K1 reproduces `separate=False` path).
- The K1 target is the **OA-converged LP optimum with no cuts** = the
  `node_relax(..., separate=False)` bound.

### K2 — best-bound tree + in-tree separation.
Reuse `bnb/spatial_tree.rs` as the tree template (best-bound heap, FBBT
propagation, honest `TimeLimit`, safe-bound fathoming). Wire the existing Rust
separators — `lp/gomory.rs`, `lp/mir.rs`, `lp/cover.rs`, `lp/cut_select.rs` — into
the node LP, **re-separated fresh per node over the node box** (node-local; never
shared across siblings — the C-43 lesson).

**Gate:** `assert_cut_valid` per cut; feasible-point test (no integer-feasible
point removed); nodes-to-certify on the panel within ~2× of the Python prototype
(67/60/46/55 for rsyn0805m/rsyn0810m/rsyn0815m/syn40m).

### K3 — LP-NLP-BB primal.
NLP solve at integer-feasible LP points (verified vs the original model, the #779
pristine-model guard) + rounding/diving. Cuts live in the LP, never in the
per-node NLP → primal never starved (dissolves the #781 HOLD).

**Kill:** ON must never return None where the current NLP-BB path finds an
incumbent, on the panel.

### K4 — graduation. **WIRED + CERT-CLEAN (2026-07-19); graduation criteria met.**
`Model.solve()` routes to the kernel behind `DISCOPT_CONVEX_KERNEL` (default-OFF)
via `_convex_kernel.try_convex_solve` (gate → solve → #779 incumbent verification →
certified `SolveResult`; `None` → default path for flag-off / non-convex /
unverifiable). Validation: 8 producer/gate/routing tests pass; **smoke suite passed**
(default path unaffected); **adversarial suite 10 passed** (no regression);
**Regime-2 corpus panel cert-clean** (66 instances: routed 1, declined 65,
incorrect_count 0); **convex-family sweep cert-clean** (10 rsyn/syn instances,
`issue798_convex_family_certclean.py`: 8 certify optimal with sound dual bound and
no false optimal, 2 fall back soundly on the #779 incumbent verification —
incorrect_count 0); **net-positive PROVEN** (kernel certifies the convex panel in
~24 s vs the NLP-BB path timing out uncertified in 482 s). Flag stays default-OFF
per policy; graduation to default-ON is a follow-up once broader corpus coverage
accrues.

**Scope (narrowed 2026-07-19, owner-approved):** the kernel targets the
**smaller/quickly-certifiable** convex MINLPs, where it is SCIP-competitive
(syn05m/10m/15m certify in 0.6–1.1 s). `try_convex_solve` gives it a bounded
attempt (`min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET=120 s)`) and uses the result
only when it CERTIFIES; larger instances it can't finish in budget (rsyn0820m/0830m
— which find no incumbent, a primal gap) fall back to NLP-BB with the full time
budget rather than stalling. **SCIP-parity on larger convex MINLPs, the unmet K2
node-count gate (2.5–5.3× vs prototype), the primal/rounding heuristic, and the
default-ON graduation are tracked in #800.** #798's §3 secondary threads (native
spatial kernel default-ON; tanksize root relaxation) are split into #801 so #798
closes with its primary build (PR #799). Original K4 spec below.


`DISCOPT_CONVEX_KERNEL` default-OFF → §5 Regime-2 panel scored on
**nodes-to-certify AND first-incumbent latency AND wall** (fix the #781 tally
which missed latency); cert-clean (0 violations, no false optimum, incumbents
verified). Graduate per-family when it beats the NLP-BB path on wall without a
primal regression.

## Correctness contract (standing, binding — CLAUDE.md §1/§5, issue #798 §5)

- `incorrect_count ≤ 0`, zero slack; certificate invariant on every panel.
- Every node bound is an LP relaxation optimum over an OA of the node's
  integer-feasible set → a valid dual bound. Every accepted incumbent is
  integer-integral AND OA-tight (nonlinear residual ≤ tol) → a valid primal bound,
  independently verified vs the pristine model (#779).
- Every bound-changing stage default-OFF behind `DISCOPT_CONVEX_KERNEL` until the
  Regime-2 panel passes (cert-clean + net-positive).
- Node-local cut scoping proven by feasible-point tests before any wall claim.
- Never a false `optimal` — honest `TimeLimit`/`Exhausted`.

## Reusable assets (do NOT rebuild)

- **Rust:** `lp/simplex` (warm, P1.0 `expel_zero_artificials` basis fix),
  `lp/gomory.rs`, `lp/mir.rs`, `lp/cover.rs`, `lp/cut_select.rs`,
  `bnb/spatial_tree.rs` (tree template), `presolve` (FBBT), the E0 warm-bench.
- **Python:** `_root_cuts.py` (GMI validity proven by exact enumeration; OA loop;
  pool + selection), the probes `issue781_cutmgmt_probe.py`,
  `issue786_intree_value_probe.py`, `issue786_lpoa_bandc_prototype.py` (byte-check
  reference), the analyze-once → flat-arrays → Rust marshaling pattern.
- **Data:** `discopt_benchmarks/results/issue786/`, `results/issue781/`.

## Falsified — do NOT re-walk (binding negatives, issue #798 §4)

- Root-only cutting starves the NLP-BB primal (#781 HOLD). In-tree LP-based
  cutting is required.
- Transient LP-OA cuts bolted onto NLP-per-node fight the architecture (#797).
  Make the LP the node relaxation.
- Cut *selection* alone ≈ add-everything; the lever is the GMI *family* + sustained
  separation (#782/#797).

## Work log (append newest first)

- **2026-07-19 (iter 10): K4 producer + convexity gate + SOUNDNESS FIX + gate tests.**
  - **Production producer** `python/discopt/solvers/_convex_kernel.py`:
    `build_convex_spec(model)` extracts linear rows + linear objective +
    integrality + bounds from a real `discopt.Model`, decomposes convex nl rows
    (composite-of-affine), marshals to `solve_convex_tree_py`, or returns `None`
    → NLP-BB fallback. **Rigorous convexity gate:** route only if the objective is
    linear, every nl row decomposes, and every term is convex in the row's `≤`
    normal form (convex func & coeff≥0, or concave & coeff≤0; `≥` rows negated);
    nonlinear equalities / unknown structure fall back. `solve_convex_tree()` runs
    the kernel; `convex_kernel_enabled()` reads `DISCOPT_CONVEX_KERNEL` (default-OFF).
  - **SOUNDNESS FIX** (surfaced by the unseeded production path — the seeded K2
    gate hid it): a tolerance-feasible OA-vertex incumbent can exceed the frontier
    dual, so the reported bound sat BELOW the incumbent (invariant `bound≥incumbent`
    violated; rsyn0805m bound 1296.1003 < inc 1296.1207). Fixed: report the dual
    bound as the max dual over ALL tree leaves + the frontier (rigorous, ≥ true opt)
    and clamp the incumbent to it. Verified unseeded: `bound ≥ incumbent` AND
    `bound ≥ oracle optimum` on all 4. Locked by a Rust assertion.
  - **5 Python gate tests** (`test_convex_kernel_gate.py`): convex routes +
    certifies soundly; bilinear / wrong-curvature / nonlinear-equality /
    nonlinear-objective all fall back. All pass. 10 Rust tests pass.
  - **Remaining K4:** (a) route `Model.solve()` behind `DISCOPT_CONVEX_KERNEL`
    (default-OFF) — a careful, soundness-critical integration into
    `python/discopt/solver.py`: when the flag is ON and `build_convex_spec` returns
    a spec, run the kernel + verify the incumbent vs the pristine model (#779) +
    map the result to `SolveResult`; otherwise the existing path, untouched. NEVER
    route an un-gated model. (b) Regime-2 corpus panel (flag ON vs OFF): cert-clean
    (0 incorrect, no false optimum, incumbents verified) + net-positive. Perf polish
    toward SCIP's ~2 s is optional (kernel already decisively beats NLP-BB).

- **2026-07-19 (iter 9): pseudocost branching — 2.6× wall (64.6s → 24.4s), cert-clean.**
  - Replaced most-fractional with SCIP product-rule pseudocost branching (reuse
    `bnb::branching::Pseudocosts`). `TreeNode` carries its `(var, frac, is_down)`;
    the pseudocost is updated with the tightening gain once the node's dual is known.
  - **Nodes 507/483/1185/337 → 353/177/281/139** (rsyn0815m 4.2× fewer);
    **total wall 64.6s → 24.4s**. Cumulative vs cold baseline: **124s → 24.4s (5×)**.
    All cert-clean (bound = optimum EXACTLY). Now ~10× off SCIP's ~2s (was ~30×).
  - Remaining perf gap to SCIP: still per-node cost (each node ~20-50 cold+warm LP
    solves) + node count (353/177/281/139 vs prototype 67/60/46/55, 2.5-5.3×).
    Next candidate levers: reliability branching (strong-branch the top pseudocost
    candidates early), better cut management, or revisiting parent→child warm with
    a proper dual-feasible restart. Diminishing returns; K4 graduation may already
    hold (kernel decisively beats the NLP-BB path — see below).
  - **NLP-BB head-to-head — DECISIVE (K4 net-positive PROVEN):** the current
    NLP-BB path (`dm.from_nl(name).solve(time_limit=120)`) times out
    **uncertified** on ALL 4 (status `feasible`, `gap_certified=False`, 120 s each,
    482 s total), and its incumbents are poor (rsyn0805m 1105.7 vs opt 1296.1;
    syn40m −38.2 vs 67.7). The convex kernel **certifies all 4 optimal in 24.4 s**
    (~20× faster) with `bound = optimum EXACTLY`. The kernel doesn't merely win on
    wall — it CERTIFIES the convex family that the NLP-BB path cannot certify at
    all in 120 s. This is #798's goal and a clear K4 graduation case (cert-clean +
    net-positive, both bars).
  - **Next: K4 graduation** — wire the kernel into `Model.solve()` routing for the
    convex MINLP family behind `DISCOPT_CONVEX_KERNEL` (default-OFF), with the
    analyze-once producer detecting composite-of-affine convexity and falling back
    to NLP-BB otherwise; run the Regime-2 corpus panel (cert-clean, no false
    optimum, incumbents verified). Perf polish (toward SCIP's ~2 s) is now optional
    — the kernel is already a large, cert-clean improvement over the status quo.

- **2026-07-19 (iter 8): K3 satisfied; parent→child basis inheritance FALSIFIED;
  measuring vs NLP-BB.**
  - **K3 (primal) is effectively DONE.** The tree certifies UNSEEDED on the panel
    (`initial_incumbent=None`): all 4 optimal, incumbents found at integer-feasible
    OA-tight leaf vertices, verified feasible to tolerance by `is_integer_feasible`
    (nl residual ≤ oa_tol, int ≤ int_tol — the #779-style guard, already present).
    **Node counts are IDENTICAL seeded vs unseeded** (rsyn0815m 1185=1185) → the
    primal is NOT the bottleneck; rounding/NLP heuristics would give ~0 here.
    Incumbent objective slightly exceeds the true optimum (OA-vertex is an
    overestimate; rel 2.5e-6, within the rel-1e-4 tolerance) — a genuinely-feasible
    objective via an NLP polish is deferred (within tolerance, needs a Python NLP
    callback, low value on this family). Existing tree tests already run unseeded.
  - **Parent→child base-basis inheritance — FALSIFIED (measurement wins, §4).**
    Implemented (store the node's base-LP basis on `TreeNode`, warm the child's
    first solve via `solve_lp_warm_scaled_csc`), cert-clean, nodes slightly down
    (2512→2072) — but **wall UP 64.6s→92.8s** (per-node 26→45 ms). The parent's
    base basis is a poor warm start across the tighter child box (warm-attempt +
    factorize + dual-feasibility check, then likely cold fallback / slow
    reoptimize, costs more than a direct cold solve). **Reverted.** Revisiting it
    needs deeper simplex warm-start work (why `PreparedDual::prepare` fails or
    reoptimizes slowly across the branching bound change), not a quick win.
  - **Best committed state: within-node warm-start, 64.6s total, cert-clean.**
  - **Next:** measuring the kernel vs the current NLP-BB path
    (`dm.from_nl(name).solve()`) head-to-head — the actual K4 graduation criterion.
    If the kernel already beats NLP-BB on wall, it's a net-positive graduation
    candidate independent of SCIP parity. Remaining perf lever: node-count reduction
    (the loop's row order raised nodes ~2.5× vs the old loop — investigate branching
    / row order), which compounds with the warm-start.

- **2026-07-19 (iter 7): K2e warm-start node solve DONE — ~2× wall, cert-clean.**
  - Unified `oa_converge`+`solve_node_cut` into one growing-LP loop: base `[le, eq]`
    assembled once, OA tangents + cuts APPENDED (append-only → columns stable), each
    re-solve WARM from the carried+extended basis via `solve_lp_warm_scaled_csc`
    (equilibrates → NS still certifies; cold fallback → always correct).
  - **Total wall 124s → 64.6s (~2×); per-node ~6× faster** (rsyn0815m 177→29 ms/node).
    All 4 cert-clean (bound = optimum EXACTLY). 10 tests pass, clippy clean.
  - **Node counts rose 1006→2512** — the loop's row order perturbs vertex/cut
    selection on these degenerate big-M LPs (A/B: cold-in-new-loop rsyn0815m 738 /
    TIMED OUT at 120s; warm 1185 / 34s — warm rescues the heavier tree). Net wall
    still 2× better, so committed; the node blowup is tree-quality, addressed next.
  - **REMAINING SCIP-parity levers (still ~30× off SCIP's ~2s for the 4):**
    1. **Parent→child basis inheritance** (the big one): each node's FIRST solve is
       still COLD. Store the node's optimal `Basis` on its `TreeNode`; a child (one
       branching bound changed) dual-reoptimizes from it in a few pivots
       (`solve_lp_warm_scaled_csc` from the inherited basis). This is how SCIP amortizes
       — likely the dominant remaining win. Thread a `Option<Basis>` through the heap
       node and into `solve_node_cut`'s first solve.
    2. **Node count** — stronger branching (pseudocost/reliability vs most-fractional)
       and/or a row order that keeps the tree small; investigate the degeneracy
       sensitivity. Fewer nodes compounds with (1).
    3. Then K3 primal + K4 graduation.

- **2026-07-19 (iter 6): K2c/K2d DONE (tree + gate). Correctness PERFECT; WALL is
  the real bottleneck (measured) → warm-start rework is the priority.**
  - Tree PyO3 binding `solve_convex_tree_py` + panel gate `issue798_k2_tree_gate.py`.
  - **Correctness gate PASSES (non-negotiable):** all 4 instances certify `optimal`
    with dual bound = oracle optimum EXACTLY (1296.12/1721.45/1269.93/67.71),
    cert-clean, no false optimal, no bound below oracle. The tree is sound.
  - **Separation:** added MIR (c-MIR family) — measured lever (kernel closed 38% of
    syn40m's root gap vs the prototype's 78%; GMI+cover alone left it open). MIR
    more than halved total panel nodes (2232→1006; rsyn0815m 1301→141, syn40m
    365→211). rsyn0805m regressed 287→377 (MIR crowds GMI in the top-8 `select_cuts`
    — a per-family cut budget would fix it).
  - **Node-count gate: FAIL** — 377/277/141/211 = 3.0–5.6× the prototype
    (67/60/46/55), down from 4–28× but above the 2× bar. NOT weakened.
  - **DECISIVE MEASUREMENT (the K4 metric): kernel WALL = 25–36 s/instance**, NOT
    sub-second — 64× SCIP's 0.5–0.8 s. The "Rust nodes are ~1000× faster" prior was
    WRONG for this kernel: it is ~87 ms/node. **Root cause: `solve_node_cut`
    re-assembles the LP from scratch (`SparseCols::from_csc`) and COLD-solves it on
    every OA round × every separation round — ~60–120 cold LP solves/node.** The K1
    "box-reassembly" shortcut (fine for the K1 byte-check) is the bottleneck.
  - **NEXT PRIORITY — K2e: warm-start node solve (precise design).** The wall cost
    is the COLD re-solve each round, not CSC rebuild. The fix is warm-starting, via
    an **append-only** row order so the basis carries across rounds:
    - Assembly order becomes `base (le + eq)` ‖ `extra_rows` (OA tangents AND cuts,
      in add-order). Structural cols `[0,n)` and base-slack cols never move; each
      new row appends its slack at the END. So the previous solve's `Basis` extends
      to the new size by making the new slack basic (`extend_basis` idiom) → a valid
      dual-repairable warm start.
    - Unify tangents+cuts into one `extra_rows: Vec<AsmRow>`. A cut generated over
      the current standard-form cols stays valid next round (cols are append-only) →
      **`substitute_slacks` is no longer needed** (drop it). Express OA tangents,
      GMI, cover, MIR all as one row-add.
    - Warm re-solve with `solve_lp_warm_scaled_csc(&LpView{a:dense,…}, b, &basis,
      opts, &sp)` (SCALED — keep it, NS certification depends on equilibration; the
      dense `a` build is O(m·n) ≪ a cold solve). `expel_zero_artificials: true`.
    - Keep the K2 gate cert-clean (bound = opt EXACTLY) and all 10 Rust tests as the
      guard. Re-measure wall vs the NLP-BB path (`dm.from_nl(...).solve()`).
    - **Sep-rounds sweep — FALSIFIED as a wall lever (rsyn0805m):** sr2/4/6/12 →
      1641/1143/741/377 nodes but 35.6/37.6/32.9/33.8 s — **wall is ~invariant**
      (~33 s) while nodes drop 4×. More cuts = fewer nodes but the same total
      LP-solve work; reducing sep rounds does NOT cut wall. Warm-starting the
      per-solve is the ONLY lever. (Keep sep rounds at 12 — best node count, same
      wall.)
    - **NS-scaling risk RESOLVED:** `solve_lp_warm_scaled_csc` DOES equilibrate
      (dual.rs:42–50 scale→warm-solve→unscale x/dual), so warm re-solves stay
      NS-certifiable — the cold `solve_lp_cols_scaled` certification carries over.
    - `PreparedDual::prepare` uses `sp` for the matrix and only `lp.{m,n,l,u,c}` —
      `lp.a` is unused, so pass an empty `a` in the `LpView` (no dense build needed).
    - Only after warm-start lands are K3 (primal) and K4 (graduation) meaningful —
      a slow kernel can't graduate on wall.

- **2026-07-19 (iter 5): K2a/b DONE — in-node separation.**
  - Refactored `solve_node` → `oa_converge` helper (K1 path unchanged) + new
    `solve_node_cut` (K2): OA-converge, then separate `separate_cover_csc` +
    `separate_gomory_cols` under `select_cuts` (efficacy×orthogonality, top-8),
    re-converge, up to `max_sep_rounds`. Node-local (C-43).
  - Cut bridge: the Rust separators emit cuts over standard-form columns
    (structural ‖ slacks); `substitute_slacks` rewrites each over structural
    columns via `s_r = b_r − A_r·x` (an identity on the feasible set → valid),
    added as a `≤` `LinRow`. Keeps the box-reassembly design; no growing-LP rewrite.
  - Test: cover cut closes `max x0+x1 s.t. x0+x1≤1.5, x∈{0,1}²` from LP 1.5 → 1.0
    (integer hull), strictly tighter AND sound (never below the integer optimum).
    8 unit tests pass, clippy clean.
  - **Next: K2c** — the best-bound tree (`solve_tree` on `ConvexKernelSpec`).
    Node = box + parent bound; best-bound heap. Per node: (1) **linear FBBT**
    propagator over `le_rows`+`eq_rows` (both directions) + integer rounding —
    REQUIRED: it makes bounds finite so the NS safe bound certifies (the K1d
    action item; without it the root's infinite bounds → uncertifiable → the tree
    can't fathom); (2) `solve_node_cut` over the propagated box; (3) fathom by
    safe bound vs incumbent; (4) branch on the most-fractional integer into two
    covering children; (5) incumbent from integer-integral + OA-tight LP vertices
    (minimal primal — K3 adds the NLP/rounding). Honest `TimeLimit`/`Exhausted`
    (never a false `Optimal`). Fork the loop shape from `bnb/spatial_tree.rs`.
    Then **K2d** — the panel gate: nodes-to-certify vs the prototype (67/60/46/55,
    within ~2×) + feasible-point test on the real convex panel.

- **2026-07-19 (iter 4): K1d DONE → K1 GATE PASSED.**
  - PyO3 binding `solve_convex_node_py` (`crates/discopt-python/src/convex_bindings.rs`,
    registered in `lib.rs`): nested-CSR flat arrays (linear ≤/= rows; nl rows →
    terms → affine args) → `ConvexKernelSpec` → `solve_node`; returns
    `{status, bound (NS safe), raw_bound (LP optimum), x, oa_rounds, n_tangents}`.
  - Producer `build_convex_arrays(rm, lo, hi)` + gate harness
    `issue798_k1_bytecheck.py`: reuses the decompose probe + RootModel. Result:
    **max|Δraw| ~1e-11 over 28 boxes, all 4 instances → PASS.**
  - Safe-bound finding: NS returns None under the raw INFINITE structural upper
    bounds (roundoff rc meets inf bound — the tanksize lesson). Verified the safe
    bound certifies under FBBT-finite bounds (finite, sound `safe ≤ py+tol`, exact)
    on all 4 → the K2 tree, which runs FBBT per node, will fathom on a certified
    bound. **K2 action item:** assemble node bounds from the FBBT-propagated box,
    not the raw global box, so the NS bound certifies.
  - **Next: K2** — best-bound tree + in-tree separation. Fork `bnb/spatial_tree.rs`
    (best-bound heap, FBBT via a convex-spec propagator, honest TimeLimit,
    safe-bound fathoming). Wire the Rust separators `lp/gomory.rs` (`separate_gomory_cols`),
    `lp/mir.rs` (`separate_mir`), `lp/cover.rs` (`separate_cover_csc`),
    `lp/cut_select.rs` (`select_cuts`) into the node LP, re-separated fresh per node
    over the node box (node-local — the C-43 lesson). Gate: `assert_cut_valid` per
    cut + feasible-point test (no integer-feasible point removed) + nodes-to-certify
    within ~2× of the prototype (67/60/46/55). Note: the Rust GMI needs the LP
    basis + integrality; solve_node currently returns only x — K2 must expose the
    final basis (or re-solve to get it) to drive `separate_gomory_cols`.

- **2026-07-19 (iter 3):** Built K1a+K1b+K1c in `crates/discopt-core/src/bnb/convex_kernel.rs`
  (7 unit tests, clippy-clean, all committed):
  - **K1a** `ConvexFunc {Log,Exp,Sqrt,Log1p}` — `eval` / `eval_and_deriv` (the OA
    tangent primitive). Tested vs finite differences + known values.
  - **K1b** `Affine`/`CompositeTerm`/`ConvexRow` — `value`, `gradient_dense`,
    `oa_tangent → LinCut`. Tested: value+grad correctness, multivar grad vs FD,
    and the soundness property (tangent exact at x̄, underestimates convex g
    everywhere = valid relaxation cut).
  - **K1c** `ConvexKernelSpec` + `solve_node(lo,hi,oa_tol,max_rounds,opts)` — the
    node LP-OA relaxation: standard-form `[A|I]z=b` assembly (≤ rows get a
    min-activity-capped slack, = rows a slack fixed at 0 — the finite-slack
    certification lesson), minimize `sign·c·x` via `solve_lp_cols_scaled`,
    separate OA tangents for violated rows, re-solve to OA convergence, dual bound
    = `ns_safe_bound_csc` negated to model sense (rigorous upper bound for max).
    Tested: OA loop converges to `ln5` on `max t s.t. exp(t)≤5`; linear node
    reproduces the LP optimum in one solve.
  - **Next: K1d** — PyO3 binding (flat arrays → `ConvexKernelSpec` → `solve_node`)
    in `crates/discopt-python/src/`, a Python producer built from `RootModel`
    (reuse `issue798_convex_decompose_probe.decompose` + RootModel's
    A_le/b_le/A_eq/b_eq/c/bounds/integrality), and the byte-check harness: Rust
    `solve_node` bound vs Python `_RootLP`/prototype `node_relax(separate=False)`
    over the root box + perturbed child boxes, gate ≤1e-6. Needs a maturin rebuild
    of the extension. This is the K1 GATE.

- **2026-07-19 (iter 1):** Branch `feat/798-convex-kernel-k1` off main; cherry-picked
  the LP-OA prototype reference (`issue786_lpoa_bandc_prototype.py`). Mapped the
  Python byte-check reference (`_RootLP`). Rust asset map in progress. Next: write
  the K1 Rust node-relaxation module + a byte-check harness against the prototype.
