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

### K1 — Rust LP-OA node relaxation. **IN PROGRESS**
Build the OA LP over a box (linear rows + OA tangents of convex nonlinear rows;
refresh box-dependent rows per node), solve via the warm in-house simplex, return
the LP dual bound.

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

### K4 — graduation.
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

- **2026-07-19 (iter 1):** Branch `feat/798-convex-kernel-k1` off main; cherry-picked
  the LP-OA prototype reference (`issue786_lpoa_bandc_prototype.py`). Mapped the
  Python byte-check reference (`_RootLP`). Rust asset map in progress. Next: write
  the K1 Rust node-relaxation module + a byte-check harness against the prototype.
