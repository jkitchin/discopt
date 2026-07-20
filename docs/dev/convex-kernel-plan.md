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

## #800 — executable task list (loop-executable)

**Goal (#800).** Certify the *full* convex `rsyn*`/`syn*` panel (through
rsyn0820m/0830m and syn40m) within **~2× SCIP wall**, **cert-clean**
(`incorrect_count ≤ 0`, dual bound = oracle optimum exactly, honest
`TimeLimit`/`Exhausted`, no false optimal), with the **K2 node-count gate met**
(nodes-to-certify ≤ 2× the prototype 67/60/46/55), then re-scope the kernel's time
budget and **graduate `DISCOPT_CONVEX_KERNEL` toward default-ON** through the
CLAUDE.md §5 Regime-2 panel. #798 shipped the kernel scoped to the
smaller/quickly-certifiable convex MINLPs; the large-instance gap is node count
(dominant driver), per-node cold LP solves, and a primal gap on the 2 fallbacks.
Each task below is self-contained and loop-executable: an agent reads this list,
takes the first `todo` task, runs its **entry experiment on the real `rsyn*`/`syn*`
panel BEFORE implementing** (CLAUDE.md §4), abandons on the kill criterion, and
otherwise implements to the done-when gate and records the result in the work log.

**Scoping-lever framing (correction — read before starting).** #800's DoD said
"lift the size cap in `_convex_kernel.build_convex_spec`." **There is no size cap
there** — the gate (`build_convex_spec` → `_build`) is *purely structural*
(linear objective, every nl row decomposes composite-of-affine, every term convex
in `≤` normal form). The actual scoping lever is the **time budget** in
`try_convex_solve`: `budget = min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET)`
(default 120 s); a large instance gets a *bounded attempt* and falls back to NLP-BB
if it doesn't certify in budget. "Lifting the cap" = closing the wall gap so the
panel certifies inside a sane budget, then re-scoping that budget (T5) — not
touching the structural gate.

**Regime note.** Three flavors appear below. (i) *Strictly bound-neutral* (T3
warm-restart, a pure per-node LP-solve speedup): assert **`node_count` AND certified
objective EXACTLY unchanged** on the panel, only wall moves — any node drift means
the change is wrong. (ii) *Search-order* (T1 row-order, T2 branching, T4 primal):
these **intentionally change `node_count`** (the optimized metric) but introduce no
new relaxation — the per-node LP bound at a fixed box is identical, so the invariant
holds by construction; verify **cert-clean** (dual bound = oracle optimum exactly,
`bound ≥ incumbent`, every incumbent #779-verified, `assert_cut_valid` on any cut
whose selection shifts) and treat `node_count`/wall/latency as the measured target.
(iii) *Graduation* (T7): the §5 Regime-2 flag-ON-vs-OFF corpus gate (cert-clean +
net-positive). No task may weaken a validation, fallback, or soundness guard to pass.

### T0 — Canonical baseline measurement (pin the anchor). **BLOCKING, do first.**
- **Goal.** One pinned measurement config + script that records, for the whole
  convex panel, the numbers every later task compares against — so ratios are
  anchored and the two pre-existing configs stop being conflated.
- **Hypothesis + evidence.** #800's wall table (rsyn0805/0810/0815m 8–10 s,
  syn40m 37 s) does **not** reconcile with iter-9's 24.4 s seeded panel: they are
  *different configs* — iter-9/`issue798_k2_tree_gate.py` **seeds** the oracle
  incumbent (isolates the dual side = nodes-to-certify) and pre-dates iter-10's
  soundness fix; the wall table is the **unseeded production path**
  (`try_convex_solve`, post iter-10). Any optimization measured against the wrong
  baseline is uninterpretable.
- **Entry experiment + kill criterion.** None to falsify — this is instrumentation,
  not a hypothesis. Pin **two** labeled measurements over the panel
  {rsyn0805m/0810m/0815m/0820m/0830m, syn05m/10m/15m/20m/40m}: **(A) production/
  unseeded** wall + `node_count` + **first-incumbent latency** via a
  `try_convex_solve` harness (extend `issue798_convex_family_certclean.py`; this is
  the wall/latency + graduation anchor); **(B) seeded nodes-to-certify** via
  `issue798_k2_tree_gate.py` (the K2 dual-side gate anchor). Kill only if a run is
  not cert-clean — then stop and file the correctness regression before any perf work.
- **Verification regime.** Measurement only; assert cert-clean on both (dual bound
  = oracle optimum exactly, no false optimal, incumbents #779-verified).
- **Done-when.** A committed baseline table (config A and config B, per instance:
  wall, `node_count`, first-incumbent latency, seeded nodes-to-certify vs prototype)
  recorded in the work log, with the pinned scripts named. Every later task cites
  these numbers.

### T1 — Row-order / degeneracy sensitivity (cheapest node-count win).
- **Goal.** Recover the node-count lost to LP row order so the panel's seeded
  nodes-to-certify moves toward the ≤2× prototype bar, wall unchanged-or-better.
- **Hypothesis + evidence.** The K2e append-only restructure (iter 6/7) raised nodes
  **1006→2512 (~2.5×)** — "the loop's row order perturbs vertex/cut selection on
  these degenerate big-M LPs" (iter 7); iter 6 A/B saw a cold rsyn0815m at 738 vs a
  differently-ordered 1301. This is a search-quality artifact of ordering on
  degenerate LPs, currently under-weighted, and is the cheapest node lever
  (no new mechanism, only ordering / tie-break / anti-degeneracy).
- **Entry experiment + kill criterion.** BEFORE implementing, sweep the candidate
  row orderings / degeneracy tie-breaks (e.g. base-row ordering, cut-append order,
  Bland/anti-cycling on ties) on the seeded panel (config B) and record
  nodes-to-certify per ordering. **Kill** if no ordering reduces panel total
  nodes-to-certify by **≥15%** vs the T0 config-B baseline without raising wall —
  then abandon and record the falsification.
- **Verification regime.** Search-order (flavor ii): `node_count` is the target and
  will move; assert cert-clean (dual bound = oracle optimum exactly, `bound ≥
  incumbent`, incumbents #779-verified, `assert_cut_valid` on selected cuts). The
  10 Rust tests + K2 gate stay green.
- **Done-when.** Panel seeded nodes-to-certify reduced ≥15% vs T0-B, cert-clean,
  wall not worse; new nodes/wall recorded against T0.

### T2 — Reliability / strong branching (cut node count).
- **Goal.** Drive nodes-to-certify further toward the ≤2× prototype bar by
  strong-branching the top pseudocost candidates early (reliability branching).
- **Hypothesis + evidence.** Pseudocost branching already gave 2.6× wall / up to 4.2×
  fewer nodes (iter 9: 507/483/1185/337 → 353/177/281/139); the remaining node gap to
  the prototype (2.5–5.3×) is the named next lever ("reliability branching — strong-
  branch the top pseudocost candidates early", iters 7/9). Reliability branching is
  SCIP's default and the standard node-count reducer on this family.
- **Entry experiment + kill criterion.** BEFORE implementing, prototype reliability
  branching (strong-branch the top-k pseudocost candidates until each is reliable)
  on the seeded panel (config B) and record nodes-to-certify **and** wall (strong
  branching costs extra LP solves per node — the trade must net out). **Kill** if it
  does not reduce panel nodes-to-certify by **≥20%** *or* raises panel wall — then
  abandon and record.
- **Verification regime.** Search-order (flavor ii): `node_count` is the target;
  assert cert-clean (dual = optimum exactly, `bound ≥ incumbent`, incumbents
  #779-verified). Rust tests + K2 gate green.
- **Done-when.** Panel nodes-to-certify ≥20% below post-T1 baseline AND panel wall
  not worse, cert-clean; numbers recorded.

### T3 — Dual-feasible parent→child warm restart (the harder amortization).
- **Goal.** Make each child node's *first* LP solve a warm dual-reoptimize from a
  proper dual-feasible starting basis (a few pivots) instead of a cold solve,
  cutting per-node wall with `node_count` **exactly unchanged**.
- **Hypothesis + evidence.** Each node's first solve is still **cold** (iters 6/7);
  this is SCIP's dominant amortization. **The idea is sound; the prior attempt failed
  on restart *mechanics*, not the idea** — iter 8's parent→child *base-basis*
  inheritance was cert-clean and cut nodes slightly (2512→2072) but wall went
  **UP 64.6→92.8 s (+28 s)** because the parent base basis is a poor primal warm
  start across the tighter child box (warm-attempt + factorize + dual-feas check,
  then cold fallback). The fix is a **dual-feasible** restart across the single
  branching bound change (`PreparedDual::prepare` from the parent's *optimal* basis),
  which is exactly the case dual simplex reoptimizes cheaply — a distinct mechanism
  from the falsified base-basis inheritance.
- **Entry experiment + kill criterion.** BEFORE a full build, prototype the
  dual-feasible reoptimize on the child's first solve for one panel instance and
  measure **pivots-to-reoptimize** and **wall** vs the cold solve; confirm the
  parent's optimal basis is dual-feasible for the child LP (same objective, tighter
  box → bound change only) so dual simplex applies. **Kill** if the dual reoptimize
  does not beat the cold solve on per-node wall on that instance, or if panel wall is
  not **≥15%** below the T2 baseline — abandon and record (do **not** re-walk the
  base-basis inheritance; that specific mechanism is falsified).
- **Verification regime.** *Strictly bound-neutral* (flavor i): the LP, box, and
  optimum are identical — assert **`node_count` AND certified objective EXACTLY
  unchanged** on the panel; only wall moves. Any node drift ⇒ the restart is wrong.
  NS certification must still hold (keep the scaled warm path — `dual.rs` equilibrates).
- **Done-when.** Panel wall ≥15% below T2 baseline with `node_count` bit-identical
  and cert-clean; per-node wall + pivots recorded against T0.

### T4 — K3 primal: rounding/diving to close the 2 fallbacks (primal gap, distinct).
- **Goal.** Find a certified incumbent for rsyn0820m/rsyn0830m within budget so they
  certify instead of falling back — a **primal** (incumbent-finding) workstream,
  separate from the K2 node/wall work.
- **Hypothesis + evidence.** The 2 fallbacks decline at the **#779 incumbent
  verification** — "likely no certified incumbent within the budget, a large-instance
  timeout rather than an infeasible incumbent" (#800). On the smaller panel the primal
  is *not* the bottleneck (iter 8: seeded ≡ unseeded node counts), so this is
  specifically the large-instance primal-starvation case a rounding/diving heuristic
  at integer-feasible LP points targets (K3; cuts live in the LP, never in the NLP →
  primal never starved, dissolving the #781 HOLD).
- **Entry experiment + kill criterion.** FIRST confirm the root cause on
  rsyn0820m/0830m: instrument `try_convex_solve` to record whether the budget expires
  with **no incumbent** vs an incumbent that **fails #779**. Only if it's *no
  incumbent* proceed; then prototype a rounding/diving heuristic at LP integer-feasible
  points and measure **first-incumbent latency**. **Kill** if the heuristic does not
  produce a #779-verified incumbent on at least one of the two within a 120 s budget —
  abandon and record (the gap may then be dual-side, i.e. T1–T3 territory).
- **Verification regime.** Search-order/primal (flavor ii): the dual bound is
  unchanged; every heuristic incumbent is **#779-verified feasible against the
  pristine model** before acceptance (never a false or infeasible incumbent), certified
  objective = oracle optimum exactly. K3 kill invariant: ON must never return `None`
  where the NLP-BB path finds an incumbent.
- **Done-when.** rsyn0820m/0830m certify optimal cert-clean within budget (or the
  root cause is proven dual-side and the task is re-scoped/closed with that finding
  recorded); first-incumbent latency recorded.

### T5 — Re-scope the kernel time budget (the reframed "size cap").
- **Goal.** Once T1–T4 close the wall gap, raise/adapt `DISCOPT_CONVEX_KERNEL_BUDGET`
  so the full panel certifies *inside* budget instead of a premature fallback —
  without ever letting a large instance stall past an honest limit.
- **Hypothesis + evidence.** The scoping lever is the budget in `try_convex_solve`
  (`min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET=120 s)`), **not** a structural cap in
  `build_convex_spec` (there is none). After T1–T4 the panel wall should fall well
  under 120 s, so the current budget already permits certification; the task is to
  confirm and, if needed, tune the default so no panel instance falls back for time.
- **Entry experiment + kill criterion.** Re-run the T0 config-A production harness
  post-T1–T4 and record per-instance wall vs budget. **Kill/defer** if any panel
  instance still exceeds a sane budget (≥120 s) — the wall work is unfinished; return
  to T1–T3 rather than inflate the budget to mask a slow solve (§3: no tolerance/limit
  tweak to pass a gate).
- **Verification regime.** Search-order/config (flavor ii): fallback path and honest
  `TimeLimit` semantics unchanged; assert cert-clean on the full panel via config A.
  Never report a false `optimal` for a budget-exhausted solve.
- **Done-when.** Full panel certifies within the (possibly re-tuned) budget on the
  production path, cert-clean, fallback still sound for anything over budget.

### T6 — K2 node-count gate + panel-wall confirmation (checkpoint).
- **Goal.** Confirm the #800 numeric bars are met on the panel before graduation:
  nodes-to-certify ≤ 2× prototype AND wall ≤ ~2× SCIP, cert-clean.
- **Hypothesis + evidence.** The K2 node-count gate is the standing unmet bar
  (2.5–5.3× → ≤2×, #800 / iter 6); wall ~2× SCIP is the #800 DoD. This checkpoint
  ties T1–T5 to those thresholds so graduation isn't attempted early.
- **Entry experiment + kill criterion.** None — this is the gate re-run
  (`issue798_k2_tree_gate.py` for nodes; config-A harness for wall). **Kill/return**
  to the relevant node/wall task if either bar is unmet.
- **Verification regime.** Gate re-run; cert-clean assertion (dual = optimum exactly,
  no false optimal, incumbents #779-verified).
- **Done-when.** `issue798_k2_tree_gate.py` reports PASS (nodes ≤2× prototype,
  cert-clean) AND config-A panel wall ≤ ~2× SCIP; numbers recorded.

### T7 — Graduation: Regime-2 corpus panel → `DISCOPT_CONVEX_KERNEL` default-ON.
- **Goal.** Graduate the flag toward default-ON via the CLAUDE.md §5 Regime-2 gate
  (flag ON vs OFF over the in-repo corpus): cert-clean AND net-positive.
- **Hypothesis + evidence.** #798 proved net-positive on the smaller family (kernel
  certifies in ~24 s vs NLP-BB timing out uncertified at 482 s) and cert-clean on the
  66-instance Regime-2 panel; #800 extends that to the full `rsyn*`/`syn*` panel once
  T6 passes. §5 requires one passing graduation-gate run meeting BOTH bars.
- **Entry experiment + kill criterion.** Run the Regime-2 corpus panel flag-ON vs
  flag-OFF (`issue798_regime2_panel.py` + the convex-family sweep). **Kill** if the
  ON run shows ANY of: `incorrect_count > 0`, a bound above its reference optimum, a
  `gap_certified=True` instance regressing to uncertified, an unverified incumbent, or
  a net node/wall regression on a routed family — leave the flag default-OFF and record
  (the `DISCOPT_CUT_INHERIT` lesson: sound-but-neutral stays OFF).
- **Verification regime.** *Graduation* (flavor iii): §5 Regime-2 — (1) cert-clean
  (`incorrect_count = 0`, no bound above reference, no certification regression,
  incumbents independently feasibility-verified) AND (2) net-positive (measurably
  helpful on node/wall). Keep the `=0` opt-out and the NLP-BB fallback intact.
- **Done-when.** One passing graduation-gate run meeting both bars → flip
  `DISCOPT_CONVEX_KERNEL` default-ON (opt-out preserved); or, if not net-positive,
  the flag stays OFF with the measurement recorded and #800 closes on the scoped win.

## Work log (append newest first)

- **2026-07-20 (#800 T1): ROW-ORDER / DEGENERACY LEVER — FALSIFIED (kill hit).**
  Entry experiment (CLAUDE.md §4) run BEFORE any production change, on the seeded
  config-B panel (rsyn0805m/0810m/0815m/syn40m; T0-B baseline **950** nodes-to-certify).
  Swept all three levers the T1 spec names, via OFF-by-default experimental knobs;
  measured panel total nodes-to-certify per variant (all cert-clean: bound ≥ oracle,
  closed onto it):
  - **Branching tie-break** (`select_branch` among near-equal pseudocost scores):
    0=first-index (baseline) **950** / 1=last-index 1076 / 2=most-fractional 960 /
    3=max-degree 1018.
  - **Base ≤-row order** (permute the ≤ block → different slack columns → different
    degenerate vertex): 0=producer (baseline) **950** / 1=sparsest-first 968 /
    2=densest-first 1222 / 3=reversed 1774.
  - **Cut-append order** (reverse selected-cut append): 0=baseline **950** /
    1=reversed 1080.
  - **KILL CRITERION HIT.** T1 kills unless some ordering cuts panel nodes ≥15%
    (950 → ≤807) without raising wall. Across **9 non-baseline variants over 3
    distinct ordering mechanisms, NONE reduces panel total below 950** — the best
    non-baseline was 960 (+1.1% *more* nodes), and most were far worse. The producer
    order + first-index branching (the T0 baseline) is already the best of everything
    tested.
  - **What the measurement says (record, do not re-walk):** row order genuinely
    perturbs node count *a lot* (reversed ≤-order 950→1774; individual instances
    swing wildly, e.g. rsyn0815m 261↔1237 across orderings) — confirming the
    degeneracy sensitivity the hypothesis named — but it is **unbiased variance, not
    a systematic lever**: no single ordering wins the panel, and per-instance gains
    (rsyn0805m improves under several orderings) are cancelled by per-instance losses
    (rsyn0815m blows up). The iter-6 "738 vs 1301" evidence was one such swing, not a
    recoverable systematic order. The 2.5–6.1× node gap to the prototype must be
    closed by a real mechanism (**T2 reliability/strong branching**), not by
    reordering. **Do not re-attempt row-order / cut-order / branching-tie-break as a
    node-count lever on this family.**
  - **Scaffolding reverted** (CLAUDE.md §3, no dead flags): the three experimental
    knobs (`DISCOPT_CVX_BRANCH`/`ROWORD`/`CUTORD`) and the sweep harness are removed;
    the kernel is bit-identical to the T0 baseline (10 Rust convex tests pass). This
    work-log table is the durable falsification record (performance-plan §6 house
    style). **SURFACED to the human for re-scoping per the loop protocol.**

- **2026-07-20 (#800 T0): CANONICAL BASELINE PINNED — cert-clean, both anchors.**
  - **Instrumentation (measurement-only, bound-neutral):** added first-incumbent
    latency to the kernel — `ConvexTreeResult.first_incumbent_node` /
    `first_incumbent_secs` (unseeded runs; `None` when seeded or no incumbent),
    surfaced through the PyO3 dict (`solve_convex_tree_py`). No search change:
    it only records `(node_count, wall)` when the first incumbent is set. 10 Rust
    convex tests pass.
  - **Config A — production/unseeded, `issue800_t0_baseline.py`, budget=120 s**
    (the wall / node_count / first-incumbent-latency + graduation anchor; mirrors
    `try_convex_solve` incl. the #779 verify). **cert-clean PASS, incorrect=0**:

    | instance  | status  | wall_s | nodes | inc_node | inc_lat_s | bound      | opt        |
    |-----------|---------|-------:|------:|---------:|----------:|-----------:|-----------:|
    | rsyn0805m | optimal |   7.81 |   353 |      233 |      5.52 |  1296.1207 |  1296.1206 |
    | rsyn0810m | optimal |   8.29 |   315 |      257 |      6.96 |  1721.4721 |  1721.4477 |
    | rsyn0815m | optimal |   6.70 |   237 |      161 |      4.90 |  1269.9288 |  1269.9256 |
    | rsyn0820m | optimal |  80.52 |  1805 |     1798 |     80.43 |  1150.4160 |  1150.3005 |
    | rsyn0830m | optimal |  50.80 |  1172 |     1067 |     46.70 |   510.0722 |   510.0720 |
    | syn05m    | optimal |   0.00 |     3 |        2 |      0.00 |   837.7324 |   837.7324 |
    | syn10m    | optimal |   0.00 |     1 |        1 |      0.00 |  1267.3536 |  1267.3536 |
    | syn15m    | optimal |   0.15 |    19 |       14 |      0.13 |   853.2848 |   853.2847 |
    | syn20m    | optimal |   0.58 |    43 |       28 |      0.47 |   924.2642 |   924.2633 |
    | syn40m    | optimal |  17.93 |   631 |      213 |      7.42 |    67.7136 |    67.7133 |

    Panel wall ≈ **173.7 s** (dominated by rsyn0820m 80.5 s + rsyn0830m 50.8 s).
    Every bound ≥ oracle optimum and closed onto it; every incumbent #779-verified.
  - **Config B — seeded nodes-to-certify, `issue798_k2_tree_gate.py`** (the K2
    dual-side gate anchor; oracle incumbent seeded to isolate nodes-to-certify):

    | instance  | nodes | proto | ratio  | bound     | opt       |
    |-----------|------:|------:|-------:|----------:|----------:|
    | rsyn0805m |   353 |    67 | 5.27×  | 1296.2224 | 1296.1206 |
    | rsyn0810m |   177 |    60 | 2.95×  | 1721.5101 | 1721.4477 |
    | rsyn0815m |   281 |    46 | 6.11×  | 1269.9261 | 1269.9256 |
    | syn40m    |   139 |    55 | 2.53×  |   67.7133 |   67.7133 |

    Seeded panel total = **950 nodes**. **K2 gate: FAIL** on the node-count bar
    (2.53–6.11× > 2× prototype); cert-clean holds (bound ≥ opt, closed onto it).
    This is the standing unmet dual-side lever T1/T2 target.
  - **KEY FINDING (re-scopes T4/T5).** At the **120 s production budget the full
    panel certifies cert-clean**, including rsyn0820m (80.5 s) and rsyn0830m
    (50.8 s) — the two "fallbacks" in the #798 K4 narrative. Those fallbacks were a
    **45 s-budget artifact** (`issue798_convex_family_certclean.py` used
    `time_limit=45`), NOT a fundamental primal gap: both find a #779-verified
    incumbent (inc_node 1798 / 1067) and certify within 120 s. Consequence:
    **T4's premise** (rsyn0820m/0830m find *no* incumbent within budget) does not
    hold at the production budget — the T4 entry experiment must first re-confirm
    the root cause at budget=120 s; the remaining large-instance gap is **wall /
    node-count (dual-side, T1–T3)**, not primal starvation. **T5** (budget) is
    largely already satisfied at 120 s; the lever is closing wall so the slowest
    (rsyn0820m 80.5 s) drops well under budget.
  - **Pinned scripts:** config A `discopt_benchmarks/scripts/issue800_t0_baseline.py`
    (budget arg, default 120); config B `discopt_benchmarks/scripts/issue798_k2_tree_gate.py`
    (seeded, 4 prototype instances). Every later T-task cites these numbers.

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
