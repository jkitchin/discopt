# nvs05 (#362 part b) — the certification edge is one declined node LP, not bound strength

**Task:** issue #362's remaining half — "dual bound ≥ 5.0 / certify nvs05" —
after PF1 (#632) resolved part (a) and lifted the reported bound off the 1.35
taint floor. **Date:** 2026-07-16. **Regime:** diagnosis + entry experiment,
then a flag-gated bound-changing fix (CLAUDE.md §5 regime 2).
**Env:** in-container (4 cores, ~10× slower per node than the reference host;
all numbers below are controlled OFF-vs-ON on this container, not comparable to
the reference-host absolutes in earlier docs). `JAX_PLATFORMS=cpu
JAX_ENABLE_X64=1`, jax 0.10.2, main @ `fd5db1c`.

## Headline

nvs05 under the **default config** now *exhausts its tree* (in-container: 173
nodes, 105 s wall at tl=180) with the correct incumbent 5.4709341 and frontier
≈ 5.4696 — the dual-bound-strength problem the issue describes **no longer
exists** (F17's "needs thousands of nodes" was overtaken by PF1 + the uniform
engine + adaptive NLP). What remains is **exactly one node** at the
certification edge: its LP defeats both in-house warm simplex attempts, the
generic cold path that then solves it `optimal` (LP optimum **5.470728**)
carries no certificate, `_certify` soundly refuses the raw vertex
(`_max_finite_magnitude` 2.3e9 > 1e7), the node produces no bound, its
NLP-failure sentinel survives, and it is **non-rigorously sentinel-fathomed**
carrying pop-time floor **5.469616** — 2.4e-4 below the incumbent, just outside
the 1e-4 tolerance. One declined node ⇒ `feasible` instead of `optimal`, and
the reported bound is the floor instead of the frontier.

Both failed warm attempts had **already computed rigorous Neumaier–Shcherbina
safe bounds from their own duals** (bare 5.288; equilibrated **5.4658**) and
threw them away: the #517 stash only attached them when the whole chain
produced *no* bound, and `_certify` ignores the generic path's `res.bound`
(auto-RLT drops integrality).

## Diagnosis chain (each step measured)

1. Default @180 s (in-container): `feasible`, obj 5.47093412640568 (= optimum),
   bound 5.469616074027518, 173 nodes, wall 105 s, tree exhausted. Deterministic
   across repeats.
2. Debug taint log: **one** `Non-rigorous sentinel fathom at node 169
   (iteration 30): pop-time bound 5.46962` — no other taint in the whole run
   (PF1 removed the early-iteration taints the 2026-07-10 status comment and
   `test_b2fix_taint_floor_bound.py`'s pinned floor 1.3521 described).
3. `solve_at_node` instrumentation: the tainting node's LP returns
   `status=optimal, lower_bound=None` on a **tiny, fully-finite box**
   (x0∈[0.681,0.688], x1∈[2.757,2.793], x2=5, x3=1 pinned; all 99 lifted
   columns finite — `_has_unbounded_nonlinear_col` = False,
   `objective_bound_valid` = True).
4. Decline branch: `safe_bound=None` → free-var/conditioning chain →
   `_max_finite_magnitude` = 2.33e9 (> 1e7) → decline. The 2.3e9 entries are 2
   envelope rows over the shear-scale aux (col value ~1.85e8 = 13600²).
5. Offline replay of the extracted node LP (99 cols / 378 rows, vendored at
   `python/tests/data/nvs05_node171_decline_lp.npz`):
   - bare warm simplex: **fails** (returns None) — but its dual's NS bound is
     **5.288263** (valid);
   - equilibrated warm simplex: **fails** — NS bound **5.465813** (valid);
   - generic cold path: `optimal` 5.470728, no certificate produced.

## Entry experiment (before any solver change)

Monkeypatch-simulated attach of the stashed NS bound as `safe_bound` on the
generic path's `optimal` result, nvs05 default config + flag, tl=180:

| arm | status | obj | bound | nodes | wall |
|---|---|---|---|---|---|
| default (flag OFF) | feasible | 5.47093412640568 | 5.469616074027518 | 173 | 105 s |
| attach (flag ON) | **optimal** | 5.47093412640568 | **5.4705684830157715** | 179 | 108 s |

13 attach events; early wide-box nodes get loose NS bounds (−35, −140 — they
just branch, floored at their parent bound on import), the certification-edge
nodes get 5.4655–5.4706 and *branch* instead of tainting; the tree exhausts
rigorously. **GO** (kill criterion — still-feasible/tainted at 180 s — not met).

## Shipped fix (this PR)

`MilpRelaxationModel.solve` (`_jax/milp_relaxation.py`): on an `optimal`
generic-path solve, surface the stashed NS bound as the result's
`safe_bound` — same flag as #517 (`DISCOPT_NODE_NUMERICAL_DUAL_BOUND`,
default **OFF**, bound-changing regime), since it is the same mechanism (NS
bound from the in-house simplex's own dual on a numerically-failed attempt)
consumed at one more point. `_certify` then certifies the node through its
existing `safe_bound` branch; the driver's existing NLP-failure rescue adopts
the bound and the node **branches** instead of being sentinel-fathomed.

Soundness: the NS bound is valid for ANY multiplier vector (weak duality), so a
drifted-basis dual only loosens it; a finite NS value is itself a proof the LP
is bounded, so this can never fabricate a bound on a genuinely unbounded
relaxation (himmel16 class) — the vertex-trust guards are untouched. This is
NOT the F9 node-retention trap: the node keeps a *tight* rigorous bound
(≥ the taint floor it would otherwise leave behind, in the endgame), and it is
branchable — F9's kill was retention under a *loose interval* bound on the
alphaBB route.

Verified end-to-end (real implementation, in-container):
- flag ON: `optimal` / 5.4705684830157715 / 179 nodes — nvs05's **first full
  rigorous certificate** (issue #362 acceptance: bound ≥ 5.0, status optimal);
- flag OFF: byte-identical to pre-change default (`feasible` /
  5.469616074027518 / 173 nodes).
- regression tests: `python/tests/test_issue362_decline_ns_safe_bound.py`
  (vendored-LP unit tests fail pre-fix / pass post-fix; flag-OFF unchanged;
  @slow end-to-end certify).
- differential panel (15-instance #517 set, tl=25 s, subprocess-isolated,
  flag OFF vs ON): see `panel gate` record in the PR description — bounds
  never cross an in-repo-attested oracle on either arm; no proof lost flag-ON.

## Stale-test reconciliation

`test_b2fix_taint_floor_bound.py::test_nvs05_tainted_tree_reports_rigorous_floored_bound`
pinned the 2026-07-10 taint trajectory (floor exactly 1.3520892806701879 at
20 s). PF1 (merged 2026-07-14) removed that early taint, so the pin is
falsified on any host (this container @20 s: bound ≈ 3.5; reference host
@20 s: 5.41 per the G2 record). Rewritten trajectory-free: bound ≤ oracle,
bound ≤ incumbent, incumbent never below the optimum, and `optimal` only with
a genuinely closed gap (#27a). The exact floor arithmetic remains unit-tested
in `test_spatial_cert_taint_floor_certify.py`.

## Falsification record (plan-doc house style)

> **F18 — "nvs05 part (b) needs additional bound-tightening (stronger RLT /
> per-node cuts)" is FALSIFIED on current main.** The issue's suggested
> direction (b) presumed the dual bound was pinned by relaxation strength
> (1.35 at 180 s when filed). After PF1 + #636 + G2, the default tree already
> drives the frontier to 5.4696 and *exhausts*; the certification gap is a
> single numerically-declined node LP at the edge, and the missing 2.4e-4 is
> recovered by surfacing an NS dual bound the engine had already computed and
> discarded. No new relaxation machinery is warranted for this instance class.

## Follow-up (measured, not shipped here)

**Per-node OBBT (Lever A) is now net-negative on its own target class**
(in-container, tl=60 s, default config): OBBT ON = 91 nodes / bound 3.62 /
obj 8.73 (root-heuristic incumbent never improved); OBBT budget-zeroed = 153
nodes / bound 3.88 / obj 5.89; budget-zeroed + `DISCOPT_ANALYTIC_SEPGRAD=1` =
221 nodes / bound 5.4698 / obj = optimum, and the tree *completes* at ~63 s
wall vs 105 s+ with OBBT on. Lever A predates PF1's in-tree FBBT, which now
does the dependent-aux pinning it was built for at ~1/30 the cost
(`reduce/fbbt` 0.7 s vs `reduce/obbt` 19.2 s of a 60 s run). De-gating or
striding per-node OBBT is a **default-path bound-changing change** needing the
full panel gate on the reference host — deliberately not ridden on this PR.
Separation cost is the other half (`separate/convex` 10.6 s +
`separate/univariate_square` 9.9 s at 91 nodes); `DISCOPT_ANALYTIC_SEPGRAD`
(#632 F2', default-OFF) is already positioned for it pending its jobs=1 panel.
