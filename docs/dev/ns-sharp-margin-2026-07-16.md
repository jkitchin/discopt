# Sharp Neumaier–Shcherbina margin (#309 follow-up, `DISCOPT_NS_SHARP_MARGIN`)

Status: implemented, flag-gated default-OFF (bound-changing regime,
CLAUDE.md verification regime 2). Companion to
`integer-ratio-partition-2026-07-16.md` — this is the "residual blocker"
its §5 identified, root-caused and fixed.

## 1. Falsification first: the §5 diagnosis was wrong

`integer-ratio-partition-2026-07-16.md` §5 recorded the 2.8856e-4 safe-bound
loss on the gear4 piece LPs as "the dual vector's residual, not conditioning",
and proposed iterative dual refinement. The decomposition experiment
(`exp_ns_decompose.py`, scratchpad) measured, on the optimal root piece:

| component | value |
|---|---:|
| flat margin `1e-9·(1+\|bᵀy\|+Σ\|contrib\|)`, magnitudes 1.44e5 | **2.8856e-4** |
| true dual residual (raw LP optimum − g_raw) | 1.0e-6 |
| float64 evaluation error (g_raw vs longdouble re-evaluation) | 9e-12 |
| reduced-cost sign flips f64 vs longdouble | 0 |

The loss **is the flat margin** — `1e-9 × ~2.9e5` of decomposition magnitude.
That also explains every §5 invariance observation (bit-identical across piece
width, box depth, equilibration): a magnitude-scaled constant, not a residual.
No dual refinement is needed; the dual is already good to 1e-6.

## 2. The fix: a provably sufficient forward-error margin

`_safe_lp_lower_bound_sharp` (`python/discopt/solvers/milp_simplex.py`)
replaces the flat margin with the classical first-order forward-error bound
assembled from the actual data (Higham 2002, §3.1):

- per-column reduced-cost error `E_k ≤ γ_{nnz_k+1}·(|c_k| + (|A|ᵀ|y|)_k)`,
  `γ_p = p·u/(1−p·u)` (valid for any summation order, so numpy's pairwise
  reduction is covered);
- columns with `|rc_k| ≤ E_k` have an **uncertain sign**: their box term is
  enclosed by the four interval corners `{rc_k∓E_k}×{lb_k, ub_k}` instead of a
  side selection;
- summation errors `γ_n·Σ|term|`, `γ_{m+1}·Σ|b_j y_j|`, plus final-add slop;
- the whole margin ×1.0625 (exact power of two) to dominate the dropped O(u²)
  terms and the float64 evaluation of the margin expression itself.

Measured on the gear4 optimal piece: loss 2.8856e-4 → **1.0e-6** (the genuine
dual residual). Dispatch: `_safe_lp_lower_bound` routes to sharp/legacy on
`SolverTuning.ns_sharp_margin` (`DISCOPT_NS_SHARP_MARGIN`, default OFF); the
legacy path is byte-identical when the flag is off. All three NS consumers go
through the dispatcher: the optimal-solve certificate, the #517 broken-basis
floor, and the Farkas `g₀(y) > 0` verification.

## 3. Latent soundness gap found on the way (legacy path)

The legacy evaluation scores a column with computed `rc_k == 0` as contributing
0 even when a box side is **infinite**. If the true (exact-arithmetic) reduced
cost is a rounding-error-sized nonzero, the true box-min term is −∞ and `g(y)`
is not a valid bound — a flat margin cannot cover an unbounded term. Reachable
only when a reduced cost rounds to a value within ~1e-16·magnitude of zero on a
column FBBT cannot bound (free lifted aux), with a dual that still certifies —
never observed in the corpus, but not impossible. The sharp path closes it
(uncertain-sign columns require both sides finite after FBBT, else abstain —
`test_sharp_abstains_on_uncertain_sign_with_unbounded_side` documents both
behaviors). Fixing the *default* path is a correctness-backlog decision, not a
silent ride-along here: it belongs in `docs/dev/correctness-issues.md` triage
(the fix would be to make the sharp path the default once graduated, which
subsumes the gap).

## 4. End-to-end results (gear4, this container, tol 1e-4)

| config | nodes | wall | root bound | certifies |
|---|---:|---:|---:|:--:|
| flag off (bit-identical baseline: obj/bound/nodes) | 2487 | 41.8 s | −1e-9 | yes |
| partition ON, sharp margin ON | **505** | **41.6 s** | **1.6434274718** | yes |
| + optimal incumbent injected at node 1 (ceiling) | **3** | **1.5 s** | 1.6434274718 | yes |

The root bound is now optimum − 1.0e-6 — above the 1e-4-relative certification
threshold (1.64326) — so every node fathoms by gap the moment the incumbent is
known. The 505-node run is **pure incumbent latency**: the bound side of gear4
is solved, and the remaining lever is primal (the dive's enumeration already
produces the best achievable witness; a feasibility-checked root injection is
the follow-up, same class as #287/#281).

## 5. Verification

- `python/tests/test_ns_sharp_margin.py`: randomized differential fuzz vs
  scipy-HiGHS oracle (sharp ≤ true optimum, sharp ≥ legacy, wrong-sign duals
  included; 0 violations in 400 trials × 2 signs), the distilled
  large-magnitude construction (1e5-scale decomposition, exact float64
  quantities), the uncertain-sign abstain case, the exact-zero-interval case,
  dispatcher flag routing, and a slow gear4 piece-bound regression
  (`bound ≥ 1.64330` with the flag on).
- Flag-off gear4: objective/bound/node-count bit-identical to baseline.
- Smoke + certificate-layer guard suites green (see PR description).
- mypy: no new errors (33 pre-existing in other files, unchanged).

## 6. Not done here

- `discopt._jax.obbt._ns_safe_lp_lower_bound` mirrors the flat-margin constant
  and keeps it — OBBT bounds are per-variable-box, not objective certificates,
  and the 1e-9-relative slack there costs box width, not certification. Sharing
  the sharp evaluator is a possible later cleanup with its own neutrality test.
- Flag graduation (both `DISCOPT_INTEGER_RATIO_PARTITION` and
  `DISCOPT_NS_SHARP_MARGIN`) per regime 2: corpus-wide differential panel on
  consecutive nightly runs.
