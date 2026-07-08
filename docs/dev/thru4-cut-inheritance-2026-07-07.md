# THRU-4 — root-cut-pool inheritance for square/PSD separation (2026-07-07)

**Task.** THRU-3 (`docs/dev/thru3-node-decomp-2026-07-07.md`, PR #550) measured
that the dominant per-node cost on the dense integer-QP class is the pair of
per-node point-separation loops — `_separate_univariate_square` (73% of the
nvs24 solve wall) and `_separate_psd` (12%) — each re-deriving its cut family
via up to 8 full MILP re-solves at EVERY node; with both off, nvs24 runs 9→309
nodes (36×) at essentially unchanged bound-at-TL. THRU-4 implements the lever
(BARON's architecture): separate fully at the ROOT, pool the cuts, and INHERIT
them at every node instead of re-separating — keeping the root tightness while
approaching the 36× throughput ceiling.

Base: branched from `origin/main` @ `3d736c9a` (PR #550 still OPEN at branch
time — its default-OFF `DISCOPT_SQUARE_COST_GATE` prototype, measured
throughput-inert, was absent, so the baseline behaviour equals THRU-3's
baseline, reproduced below to the node). #550 merged mid-task; the branch was
**rebased onto `2d4d1efa`** (post-#550 main) with both features kept
side-by-side (the square-cost-gate remains default-OFF and orthogonal), and the
flag-ON fire test re-run post-rebase reproduces the pre-rebase result exactly
(nvs19: optimal, 367 nodes, 53.0 s, pool 70/357/357, root −1104.2400022094807).

Build: release (`maturin develop --release`), pounce `_pounce.abi3.so` 4.73 MB.
Env: `PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, TL 60 s.
Oracle (`minlplib.solu`): nvs24 = −1033.2, nvs19 = −1098.4, nvs17 = −1100.4,
nvs23 = −1125.2, nvs10 = −310.8, nvs13 = −585.2.

## Part 1 — per-family validity classification (the soundness crux)

Every cut family the node separation chain can emit, classified for
inheritance from a ROOT-box pool into descendant sub-boxes. The governing
fact: a pool row is inherited soundly iff it is valid at every feasible lifted
point of the *root* box — every child's feasible set is a subset of the
root's, so root-box validity implies validity on every descendant. The hazard
is the reverse direction only (a cut separated on a CHILD box need not hold on
a sibling), which the design excludes by pooling **only rows captured at the
root box**.

| family | cut form | classification | inherit? |
|---|---|---|---|
| `_separate_univariate_square` | tangent `s ≥ 2·x0·x − x0²` at the LP point | **globally valid** — a tangent of the convex `x²` under-estimates it on all of ℝ; coefficients carry no box bound | yes, directly |
| `_separate_psd` | eigencut `vᵀMv ≥ 0`, `M = [[1,xᵀ],[x,X]]`, linearized on `(x, X)` | **globally valid** — `vᵀMv = (v₀ + v_rᵀx)² ≥ 0` wherever the lifting relations `X = x xᵀ` hold; the eigenvector `v` fixes coefficients, no box bound enters (verified in `psd_cuts.py::psd_cut_from_submatrix`) | yes, directly |
| `_separate_convex` (#358 composite lifts) | supporting tangent of a convex (resp. concave) claimed subexpression | **root-box valid** — the claimer certifies curvature over the root box; the tangent under/over-estimates `g` on that box, hence on every sub-box | yes (root capture only) |
| `_separate_multilinear` | over-cap multilinear envelope facets built from `milp._bounds` | **box-dependent** — envelope of the *capture-time* box; valid on that box and all its sub-boxes, invalid on points outside it | yes (root capture only); NEVER from node solves |
| `_separate_edge_concave` | vertex-polyhedral hull hyperplanes of the current box | **box-dependent** — same as above | yes (root capture only); NEVER from node solves |
| `_separate_rlt` | `(b − aᵀx)(x_j − l) ≥ 0` linearized, `l`/`u` the capture-time bounds | **box-dependent** — the bound factor is nonnegative only inside the capture-time box | yes (root capture only); NEVER from node solves |
| static child-box relaxation rows (e.g. secant `s ≤ (l+u)x − lu`) | not emitted by any separator, but the canonical counter-example | **box-dependent and NOT pooled** — the regression test demonstrates a child-box secant cutting a root-feasible point | excluded by construction |

No family had to be excluded from *root-pool* inheritance: root-box validity
is sufficient for descent, and both measured-dominant families (square, PSD)
are in fact globally valid — box-independent — so their pooled rows are exact,
not loosened, on every child. The kill criterion ("every material family
box-dependent and un-inheritable") did not trigger.

**Lazy re-separation not needed (measured).** The pool rows are appended to
each node's relaxation before the solve, so the node LP optimum satisfies them
by construction; the only "lazy" trigger available is *fresh point separation*
at the node LP optimum, which is exactly the loop being removed. THRU-3's
`both_off` control already measured the no-re-separation end point: bound-at-TL
essentially unchanged (nvs24 −1035.66→−1035.38 with NO cuts at all; here the
pool retains the root cuts so the flag-ON bound-at-TL is *identical*, see
below). Re-separation is therefore skipped entirely at nodes, per that
measurement.

## Part 2 — design

Default-OFF flag: `DISCOPT_CUT_INHERIT` / `SolverTuning(cut_inherit=True)`.

* **Root (unchanged behaviour):** the existing general root-cut-pool capture
  (cert:T1.3, `solve_at_node(..., out_cuts=...)`) runs the full default
  separation chain once on the root box and captures every appended row. With
  the flag on this capture is additionally performed when the incremental
  engine is unavailable (the cold-path instances are exactly where THRU-3
  measured the drag). Root separation rounds/parameters are untouched, so root
  tightness is retained — verified: the reported root bound is byte-identical
  flag-ON vs flag-OFF on both targets (nvs24 −1035.6600020723224,
  nvs19 −1104.2400022094807).
* **Nodes:** each node still receives `inherited_cuts=_root_cut_pool`
  (pre-existing mechanism, column-layout-gated), and with the flag on passes
  `skip_pool_separators=True` to `solve_at_node`, which skips ONLY the
  univariate-square and PSD point-separation loops (the two measured, pooled
  families). The rest of the chain (multilinear / edge-concave / convex / RLT)
  is untouched — each was <1% of the node wall in THRU-3's decomposition.
  Skipping separation only *loosens* a node's relaxation; it can never cut a
  feasible point or raise a bound — sound by construction.
* **Instrumentation** (`solver_stats`): `pool/size` (rows in the root pool),
  `pool/inherited_nodes`, `pool/inherited_rows` (fast + cold paths),
  `pool/skipped_separations` (node solves that skipped the square/PSD loops).

## Measurements (TL 60 s, single runs, same machine)

### Targets

| instance | arm | status | nodes | nodes/s | bound@TL | incumbent | root bound | pool |
|---|---|---|---:|---:|---:|---:|---:|---|
| nvs24 | OFF (default) | feasible | 9 | 0.14 | −1035.66 | −1031.80 | −1035.6600020723224 | — |
| nvs24 | **ON** | feasible | **49** | **0.76 (5.3×)** | **−1035.66 (same)** | −1031.80 | −1035.6600020723224 (identical) | size 30, inherited 38, skipped 38 |
| nvs19 | OFF (default) | feasible | 197 | 3.28 | −1102.10 | −1098.20 | −1104.2400022094807 | — |
| nvs19 | **ON** | **optimal (CERTIFIED)** | **367** | **6.93 (2.1×)** | **−1098.40 = opt** | **−1098.40 = opt** | −1104.2400022094807 (identical) | size 70, inherited 357, skipped 357 |

* **nvs19 now CERTIFIES at 60 s** (gap 0.0, wall 52.9 s) — the incumbent
  reaches the oracle optimum −1098.4 and the dual bound closes on it. Baseline
  times out at 0.35% gap.
* **nvs24**: 5.3× node throughput at *identical* bound-at-TL and incumbent.
  Not the 36× ceiling: THRU-3 already showed nvs24's residual floor is the
  base per-node MILP simplex solve (48% of samples) plus the root separation
  wall (~16 s of the 60 s TL is the flag-ON root probe + pool capture — the
  retained root-tightness cost); `both_off`'s 309 nodes came from a
  *structurally smaller* LP (no lifted aux columns), which inheritance
  deliberately does not give up. 49 nodes at the same tight bound strictly
  dominates the baseline 9.
* Per-node separation wall collapses as designed: nvs24
  `separate/univariate_square` 45.7 s → 16.1 s (all remaining time is the
  root probe + root pool capture, i.e. unchanged root behaviour),
  `separate/psd` 7.7 s → 0.0 s; nvs19 24.9 s → 0.2 s and 12.2 s → 0.04 s.

### QCQP control set (root tightness + certificates retained)

| instance | arm | status | nodes | wall (s) | bound@TL | incumbent | root bound | oracle |
|---|---|---|---:|---:|---:|---:|---:|---:|
| nvs17 | OFF | optimal | 173 | 23.3 | −1100.41 | −1100.4 | −1105.89 | −1100.4 |
| nvs17 | **ON** | optimal | 213 | **14.1 (1.65× faster cert)** | −1100.40 | −1100.4 | −1105.89 (identical) | −1100.4 |
| nvs23 | OFF | feasible | 69 | 60.1 | −1130.50 | −1125.2 | −1130.70 | −1125.2 |
| nvs23 | **ON** | feasible | **147 (2.1×)** | 60.2 | **−1130.41 (tighter)** | −1125.2 | −1130.70 (identical) | −1125.2 |
| nvs10 | OFF | optimal | 5 | 0.1 | −310.8 | −310.8 | −397.0 | −310.8 |
| nvs10 | ON | optimal | 5 | 0.1 | −310.8 | −310.8 | −397.0 (identical; no pool — sub-second cert) | −310.8 |
| nvs13 | OFF | optimal | 49 | 1.9 | −585.2 | −585.2 | −590.37 | −585.2 |
| nvs13 | ON | optimal | 53 | 1.5 | −585.2 | −585.2 | −590.37 (identical) | −585.2 |

No control regressed: every certificate is retained (nvs17 certifies 1.65×
faster), every root bound is identical flag-ON vs flag-OFF, every dual bound
stays ≤ the oracle optimum and ≤ its incumbent, and the two TL-bound instances
gain nodes (nvs23 2.1×) at same-or-tighter bound-at-TL. Pool fire-proof on all
pool-eligible controls: nvs17 size 59 / inherited 209 / skipped 209; nvs23
size 82 / inherited 109 / skipped 109; nvs13 size 36 / inherited 53 /
skipped 53. (nvs17/nvs13/nvs23 already built a pool flag-OFF via the
pre-existing cert:T1.3 incremental-path capture — the flag adds the *skip*;
on nvs19/nvs24, where the incremental engine declines, the flag adds the pool
capture itself.)

## Verification (bound-changing regime, CLAUDE.md §5)

* **Differential:** on every arm of every instance above, the reported dual
  bound is ≤ the oracle optimum (min sense) and ≤ the incumbent — no bound
  crossed the optimum. The ROOT bound is byte-identical flag-ON vs flag-OFF on
  nvs19/nvs24 (root behaviour unchanged).
* **Feasible-point sampling:** the captured root pools on nvs19 (70 rows),
  nvs24 (95 rows), nvs17 (59 rows) were evaluated at 2000 random integer
  feasible points each (exact lifted vectors): max violation 4.9e−32 — no
  pooled cut removes any feasible point, hence none can cut a point of any
  child box.
* **Regression tests** (`python/tests/test_cut_inherit_pool.py`, smoke-marked):
  (1) every pooled root cut holds at EVERY feasible lifted point of a dense
  integer-QP box (exhaustive); (2) the box-dependence hazard is pinned — a
  CHILD-box relaxation row (sub-box secant/envelope) provably cuts a
  root-feasible point, the shipped root pool does not, and the flag-ON solve
  still certifies the brute-force optimum naive child-row inheritance would
  have fathomed; (3) default-off runs perform no pool-skip.
* **Cert-neutrality (default OFF):** `check_cert_neutrality.py` — **40/41
  byte-identical** (`nodes n->n`, `|Δobj|=0.00e+00` on every row). The one
  flagged row — `nvs13 node_count 19 -> 49` — is the SAME pre-existing
  baseline drift on `main` that THRU-3 (#550) already root-caused: re-solving
  nvs13 on the BARE branch point (this diff stashed) yields **49 nodes,
  objective −585.2** — identical to this branch's default-OFF run (49 nodes,
  −585.2, see the control table). The committed `cert-baseline.jsonl` (19
  nodes) predates a since-merged main commit that moved nvs13; the default-OFF
  path of THIS change is verifiably byte-identical (49 → 49, |Δobj| = 0).
  Refreshing the stale baseline remains out of scope (unrelated main drift,
  same disposition as #550).

## Gates

* `pytest -m smoke`: **620 passed, 14 skipped** (includes the 3 new
  smoke-marked pool tests); re-run post-rebase.
* Adversarial: `pytest -m slow python/tests/test_adversarial_recent_fixes.py`
  — **10 passed**; re-run post-rebase.
* `ruff check` + `ruff format --check`: clean (re-run post-rebase).
* Pre-commit `mypy` (the actual hook, whole-package): **Passed** (re-run
  post-rebase).
* `cargo test -p discopt-core`: n/a — no Rust touched.
* Fire-proof: `pool/size` 30–95, `pool/inherited_nodes` and
  `pool/skipped_separations` fire on every pool-eligible instance (tables
  above); `pool/skipped_separations == 0` on every default-OFF run.

## Follow-on (recorded, not shipped here)

* Differential nightly greening before any default-ON discussion (CLAUDE.md
  §5 bound-changing regime).
* nvs24's residual ceiling: the base per-node MILP simplex solve (THRU-3's
  `solve_milp` 48% floor) and the ~16 s root separation wall. The next
  throughput levers on this class are a cheaper base node LP (warm-start /
  structure) and a root-separation budget — not more per-node cut work.
