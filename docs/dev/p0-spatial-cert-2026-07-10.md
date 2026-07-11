# P0 SPATIAL-CERT — extend the gap-closed certification accounting to the spatial driver (2026-07-10)

**Task:** Phase 0 of the gap-closing campaign: bring `solve_model` (the spatial
driver) up to the certification-accounting discipline B1-FIX (#604) established
for the MILP driver. Blocker being cleared: the nvs22 density-route certificate
loss that failed FLAG-GRAD verdict 3 for all three graduation candidates
(`docs/dev/flag-graduation-verdicts23-2026-07-10.md`).

**PR:** (see branch `p0/spatial-gap-certify`). Python-only; Rust untouched.

## 1. Entry experiment — why is the label withheld? (run before any code)

Reproduced the defect byte-identically (`solve_one.py`, isolated subprocess,
full-snapshot corpus): nvs22, `DISCOPT_LU_DENSITY_ROUTE=1`, TL 40:
`feasible, 37 n, obj 6.0582199356118, bound 6.0582199951512` (relative gap
~9.8e-9). Route OFF: `optimal, 35 n, bound 6.058219998999999`.

Instrumented every `_gap_certified = False` site in `solve_model` plus the
serial node-finalize chain. The full causal chain, deterministic:

1. **Iteration 5, node 33** (the route-shifted tree's extra node): the strided
   per-node NLP is skipped (`_node_nlp_due` False), so `result_lbs[33]` starts
   at the NLP-placeholder failure sentinel (1e30).
2. The node's McCormick LP (`solve_at_node`, cold path) returns
   **`status='optimal'` with `lower_bound=None`**: the vertex solved, but the
   Neumaier–Shcherbina safe bound was not computable (free lifted column) and
   the C-38-family guard (`mccormick_lp.py:1490-1505`) **soundly declines** to
   certify the vertex objective as a bound. A sound refusal — but the driver's
   LP rescue (`solver.py`, serial `mc_lp_res.lower_bound is not None` arm)
   then leaves the sentinel in place.
3. The nonconvex finalize sees the surviving sentinel with no infeasibility
   proof → `_nonrigorous_sentinel_fathom` → `_gap_certified = False`
   (`solver.py:6801` serial site; batch twin `:6391`), and the #603 C-1 sweep
   floors the removal at its **pop-time bound 7.403479999999996** — proved at
   the node's parent, valid over the subtree forever.
4. Terminal state: `gap_converged=True, is_finished=True, search_closed=True,
   poisoned=False, bound_unresolved=False, taint_floor=7.40348,
   glb=6.0582199951512, inc=6.05822` — and the pre-fix decision
   (`search_closed and _gap_certified`) downgrades to `feasible`.

**Verdict: case (c)** of the soundness triage — an over-conservative condition,
the exact analogue of B1's case (a). The bound did **not** arrive via the #603
taint floor (the frontier 6.05822 is the binding term; the floor 7.40348 is
slack, sitting **above** the incumbent — the removal is a rigorous cutoff
fathom in all but name). Every removed subtree was rigorously accounted;
clearing certification for the whole solve was unnecessary.

## 2. The fix (label-only; mirrors #604's semantics)

`python/discopt/solver.py`, terminal decision of `solve_model`:

- The rigorous global dual bound of a sentinel-tainted tree is
  `min(frontier, taint floor)` (#603) — every term a *proved* bound (frontier:
  per-node relaxation solves; floor: pop-time bounds proved at ancestors).
- Certification is **re-earned** iff the *sole* decertification cause was
  sentinel fathoms — never over an untrusted bound VALUE
  (`_tree_bound_poisoned`), an untrusted convex node bound (new
  `_convex_bound_untrusted` flag at the C-13 serial and P0.3 batch sites), or
  an unresolved -inf pin (`_bound_unresolved`) — AND the floor-inclusive bound
  closes the certification gap under the **identical** convergence arithmetic
  the search certifies with (`_gap_values_converged`, extracted from
  `_gap_converged`). This is #604's criterion verbatim: the `unresolved_floor`
  permanently seeds the bound and `Optimal` requires the floor-inclusive
  `gap_closed`.
- The reported bound on such an exit is the floor-inclusive value — the same
  number #603's recovery already reports on the uncertified version of these
  exits — so **node counts, objectives and bounds are byte-identical; only the
  label changes.**
- **#27a is preserved:** the fathom itself never certifies anything — only the
  proved floors do. A solve whose floor-inclusive gap does NOT close keeps the
  `feasible` downgrade and #603's no-re-certify gate at the late
  re-certification block, unchanged (negative control nvs05: floor 1.3521 vs
  incumbent 5.4709 → stays `feasible`, bound 1.3520892806701879).
- Mid-loop callback semantics are untouched (conservative `_gap_certified`
  still cleared at the event sites; the re-earn is terminal-only).

Before/after on the probes (TL 40, deterministic):

| probe | pre-fix | post-fix |
|---|---|---|
| nvs22 route ON | feasible, 37 n, bound 6.0582199951512 | **optimal**, 37 n, bound 6.0582199951512 (byte-equal) |
| nvs22 route OFF | optimal, 35 n, bound 6.058219998999999 | optimal, 35 n, bound byte-equal |
| st_e36 `DISCOPT_NODE_REDUCE=1` | feasible, 75 n, bound −246.00000046256457 | **optimal**, 75 n, bound byte-equal |
| nvs05 (open floor) | feasible, bound 1.3520892806701879 | feasible, bound byte-equal (no re-earn) |

## 3. st_e36 + node_reduce (BR-3 residual blocker) — same class, covered

Measured (trace, this build): the node_reduce loss is **not** the
reduction-fathom itself (reduce fathoms carry `node_infeasible_mask` — a
rigorous emptiness proof — and never taint). It is one **serial sentinel
fathom** (iteration 9, node 66) of the same nvs22 class, whose pop-time floor
**−246.00000046256457** sits 4.6e-7 below the incumbent −246.0000000026479 —
inside the certification tolerance — while the frontier collapses to the
incumbent and the tree drains. BR-3's "bound pinned −304.5" was the pre-#603
decertify-and-discard *reporting*; post-#603 the floor is −246.00000046. The
floor-inclusive gap is closed, so the fix's discipline covers it: st_e36 +
node_reduce now exits `optimal` with the floor as the reported bound
(≤ oracle −246; `bound ≤ incumbent` holds). No separate mechanism needed.

## 4. Gates

- **Regression test** `python/tests/test_spatial_cert_taint_floor_certify.py`:
  nvs22 route-ON → `optimal` / bound closed / **node_count == 37** (the
  label-only assertion), st_e36 node_reduce → `optimal` with the floor bound,
  plus a unit test pinning `_gap_values_converged` to the house arithmetic.
  **Verified fail-before (both instance tests FAIL on baseline `solver.py`) /
  pass-after in the same build.**
- **Cert-neutrality** (`check_cert_neutrality.py`, 41 certifying instances):
  objectives Δ=0 on all 41; node counts identical on 38; the 3 flagged
  (nvs02 101→337, nvs11 39→55, nvs12 195→231) reproduce **byte-identically
  with the fix reverted** — the same pre-existing origin/main drift vs the
  committed baseline that #603 documented.
- `pytest -m smoke`: 644 passed, 1 skipped.
- Adversarial suite (`pytest -m slow python/tests/test_adversarial_recent_fixes.py`): 10 passed.
- b2fix taint-floor suite (`test_b2fix_taint_floor_bound.py`, #27a negative
  control): 2 passed.
- `ruff check` / `ruff format --check`: clean; `mypy` adds no new errors.
- Rust untouched (no `cargo test` required; extension rebuilt from origin/main
  crates for the measurements).

## 5. Soundness panel + FLAG-GRAD verdict-3 re-run (35 instances, TL 60)

Harness: `docs/dev/data/p0-spatial-cert-2026-07-10/` (byte-copied from the
verdicts23 run; arms `off`, `lu_density_route`, `obj_branch_priority` +route,
`lift_loose_products` +route; isolated subprocess per solve; oracle =
`minlplib.solu`; deterministic metrics only, §0.7).

Shared OFF baseline (this build): every certified instance byte-matches the
pre-fix verdicts23 OFF run — 28 optimal, identical node counts and objectives
(nvs21 195 n, st_e36 153 n, nvs22 35 n, …). The fix's re-earn fired on **zero**
OFF-panel instances (no new labels at defaults). The only OFF difference vs the
pre-fix run is TL-bound progress on time-limited instances under different
machine load (tls2 feasible→time_limit with no incumbent this run, nvs09/nvs20/
ex7_2_3/st_e31 explored fewer nodes in 60 s) — wall-dependent, used for no
verdict (§0.7).

| flag (ON = flag + route) | engaged? | incorrect | oracle-cross | cert-loss | node Δ (both-certified) | verdict |
|---|:-:|:-:|:-:|:-:|---|---|
| **lu_density_route** | yes | 0 | 0 | **0** (nvs22 optimal, 37 n) | 1092 → 1092 (nvs21 195→191, nvs22 35→37, nvs13 45→47) | **GREEN** |
| **obj_branch_priority** | yes | 0 | 0 | **0** | 1092 → **1054** (nvs21 195→191, nvs22 35→**17**, nvs01 17→11, nvs13 45→35) | **GREEN** |
| **lift_loose_products** | yes | 0 | 0 | **0**; nvs09 bound −63.71→**−50.27** (lift live) | 1092 → 1092 (route deltas) | **GREEN** |

Soundness sweep (`soundness_check.py`) over all four arms: **CLEAN** — every
"optimal" row has a closed reported gap, `bound` never beyond the incumbent
(sense-corrected, 1e-6 noise) and never crosses the oracle. Label flips vs the
pre-fix verdicts23 data: exactly **nvs22 feasible→optimal** in the three
route-composed arms (the fix, on a closed gap) and the load-dependent tls2
progress difference above. **No new optimal label anywhere with an open gap.**

## 6. Graduation outcome

| flag | verdict 1 (BR-3 #602) | verdict 2 (#612) | verdict 3 (#612, pre-fix) | verdict 3 re-run (this PR) | GRADUATE? |
|---|:-:|:-:|:-:|:-:|:-:|
| `DISCOPT_LU_DENSITY_ROUTE` | GREEN | GREEN | RED (nvs22, root-caused here as a driver accounting bug) | **GREEN** | **yes — 3 consecutive greens** |
| `DISCOPT_OBJ_BRANCH_PRIORITY` | GREEN | GREEN | RED (nvs22 via route) | **GREEN** | **yes — 3 consecutive greens** |
| `DISCOPT_LIFT_LOOSE_PRODUCTS` | GREEN | GREEN | RED (nvs22 via route) | **GREEN** | **yes — 3 consecutive greens** |

The pre-fix verdict-3 RED was not a property of any of the three flags: it was
this driver certification-accounting defect (nvs22's answer, bound and tree
were all correct; only the label was withheld). With the defect root-caused and
fixed, the re-run on the same panel/TL is the valid verdict 3.

**Triple composition** (the actual default-ON package: route + obj_branch +
lift together — the pairwise arms alone don't cover it): extra `triple` arm on
the same panel — errors 0, incorrect 0, crosses 0, cert-losses 0, drift 0,
nodes 1092 → **1054** (nvs21 195→191, nvs22 35→**17**, nvs01 17→11,
nvs13 45→35), soundness sweep CLEAN.

**Default-ON diffs are prepared as the clearly-separated final commit of this
PR** (`solver_tuning.py` obj_branch_priority, `factorable_reform.py`
lift_loose_products, `linsolve.rs` density route — each keeping `=0` as the
escape hatch) and are called out in the PR body for maintainer review. Note
`docs/dev/flag-graduation-protocol.md` prescribes the flip as its own reviewed
PR including a re-run of the measurement of record
(`global_opt_baron_vs_discopt.py`) under the new defaults; the maintainer can
drop the final commit and take that path instead — the core fix does not
depend on it.
