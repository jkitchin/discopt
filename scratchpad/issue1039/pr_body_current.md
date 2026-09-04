## Summary

Works #1039 — the 18 slow/correctness-tier failures left after #1037, #1038 and
#1036 took their buckets. All 18 are dispositioned: **11 repaired, 4 pinned as
strict xfails against defects that are real and out of scope for a test-repair
PR, 3 folded into repairs of their neighbours.** The tier is green.

One production-code fix, one new instrument, and a lot of measurement. The
guiding rule throughout: a failing assertion is repaired only when the *test* was
wrong. Where the code was wrong, nothing was loosened.

## The one real code defect (bucket E)

### Retraction: the first version of this fix was a correctness regression (§11)

**Superseded by the review on this PR, and corrected in `ac42014a`.** This
section originally described widening `_is_linear` with a `SumOverExpression`
case. That was wrong and I am retracting it. `_is_linear` is a shared gate with
~10 consumers, and the hull substitution family (`_hull_linear_substitute` →
`_substitute_vars` → `_collect_variables`) and `_bound_expression` have no case
for the node. Admitting it made hull emit the disjunct body *globally* with its
selector coefficient collapsed to zero, and on

```python
m.either_or([[dm.sum(x[i] - 1 for i in range(3)) <= 0.0], [x[0] >= 8.0]])
m.minimize(-(x[0] + x[1] + x[2]))     # xi in [0, 10], true optimum -30.0
```

both `auto` and `hull` returned `status=optimal, objective=-3.0, bound=-3.0`. A
dual bound **above** the true minimum of a minimization is invalid. Three loud
refusals on `main` had become two silent false certificates — a §1 violation.
Verified on both arms with §8 marker counts before acting on the review.

### What actually ships

The defect is real: `_relax/gdp_reformulate.py` silently dropped its exactly-one
row, so a GDP whose exactly-one constraint is built as `Σ[N terms]` reformulated
without it and LOA returned `unknown`. But `_is_linear` was never the right
question. OA and LOA both gated their linear row block behind **two** predicates
— `_is_linear(c.body)` first, then `_extract_body_coeffs` — and the conservative
one wins. The extractor answers with the row itself, so it cannot promise more
capability than it delivers; it is the correct and only gate.

- `_is_linear` — `SumOverExpression` case **reverted**, with a comment recording
  why admitting it is unsound for the hull consumers.
- `_extract_body_coeffs` — keeps the term-by-term fold. This is the actual fix.
- `gdpopt_loa.py` **and** `oa.py` — redundant pre-gate dropped. `oa.py:1348` had
  the byte-identical shape, so both sites are fixed (§2, the class not the site).

Completing the hull family so those routes *solve* this model instead of
declining it is **#1154**: it is bound-changing (it moves the FBBT structural
mask at `solver.py:3068` and the aux-lift gate at `factorable_reform.py:684`) and
owes the §5 differential panel, so it does not ride along in a test-repair PR.
With the widening reverted, those consumers are byte-identical to `main`.

### Regression test

`python/tests/test_1039_gdp_sumover_rows.py` pins **both** defects. Fail-before /
pass-after, run in the installed tree with §8 marker counts confirming which
sources each arm loaded:

| sources at | marker | result |
|---|---|---|
| `bf079fc1` (the widening) | 4 | **2 failed** — `[auto]`, `[hull]`: bound `-3.0` above the optimum |
| `main` | 2 | **2 failed** — extractor returns `None`; LOA returns `unknown` |
| `HEAD` | 3 | **5 passed** |

The blocker arm asserts the refusal rather than skipping on it (a skip on all
three routes would make it a no-op, §6), and is phrased as "no route may report a
bound above the true optimum" so #1154 strengthens it rather than breaks it.
354 GDP tests pass; 911 pass on the gdp/hull/bigm/oa/benders/gbd/mpec selection.

## The instrument (bucket A)

`MccormickLPRelaxer` now keeps `_row_filter_stats` (invocations / rows_dropped),
surfaced through `solver_stats` as `row_filter/*`. Sweeping all 66 vendored
instances with it: exactly three open the failure-triggered branch — `bchoco07`
(2 inv / 158 rows), `bchoco08` (2/144), `hda` (2/356). Every other instance is 0.

### Retraction: this issue's own bucket-A comment is contradicted (CLAUDE.md §11)

The retraction comment on #1039 measured `filter_invocations=0` in **both** arms
on `hda` and concluded the mechanism was dormant, revising bucket A to "not a
contract defect". With the counter surfaced, the count on `hda` is **2, not 0**,
and the arms are far apart:

```
DISCOPT_RELAX_ROW_FILTER=0 -> bound -13992288065.86   inv=0
DISCOPT_RELAX_ROW_FILTER=1 -> bound      -64509.85    inv=2, 356 rows dropped
```

(2 reps interleaved.) The opt-out is live and load-bearing. What *was* right in
that comment is the diagnosis of the tests: they encoded a stale premise. But the
premise was subtler than "hda no longer false-fails" — each test varied the flag
it names while leaving `DISCOPT_RELAX_ROW_FILTER` at its graduated default-ON
(#671, 2026-07-18), so a **later-graduated mechanism answered for both arms** and
no test isolated its own flag.

With the other rescue flags pinned (3 reps interleaved, sd 0 in every arm):

| flag | OFF | ON | verdict |
|---|---|---|---|
| #671 refinement | -141647.03848991546 | -64509.850876897275 | live |
| #517 dual bound | -141697.43348991615 | -141697.43348991545 | agrees to 12 digits |

So `#671`'s test now asserts the **differential** (ON tighter than OFF, both
sound) instead of a hardcoded floor a third mechanism can move. `#517`'s flag is
still consulted but no longer load-bearing on `hda`; it is disposed per the
C-42/C-43/C-44 precedent — assert inertness *with the measurement*, rather than
demand it fire, which would be asserting a premise known false. Its ON path keeps
coverage in `test_hda_gets_first_finite_dual_bound`. **Re-pointing it needs a
`#517` invocation counter; that is recorded in the docstring as the one piece
left open.**

`bchoco07` also moves out of the "never fires" list — it fires twice, so
asserting zero on it was asserting a false premise — and the real-instance
positive control the retraction comment asked for is added.

Whether the branch opens is itself budget-dependent (a #1116 role-2 effect), so
the firing test is pinned to the measured plateau with the curve in the
docstring: `bchoco07` fires at tl=15/30 but not tl=60. `deterministic=True` does
not remove the dependence, it moves it (0 nodes at tl<=60, fires at tl=120).

## Retraction: my own bucket-G review recommendation was wrong (§11)

My review of #1039 said the test was right and the `gap_certified` docstring
needed correcting. That was wrong, and #1059 had already settled it on
2026-08-19 with a measurement: `gap_certified` means "the reported gap is
*mathematically valid*", not "the gap is closed", so `True` at a rigorous 37.7%
gap on nvs17 is **correct**. The **test** and the **graduation panel** were both
wrong. Both are fixed; `_route_result_is_certified` (solver.py:5760) is the right
predicate for "did this actually finish", and the panel now uses an equivalent.

## Pinned, not fixed — 4 strict xfails

Each preserves every assertion and threshold. `strict=True` means none can pass
by having its goalposts moved, and each fails the suite the moment it is fixed.

Per §3 each now ships with a filed issue tracking the real fix, linked from the
xfail's own `reason`: **#1151** (the super-optimal division objective),
**#1152** (the root-setup overrun and the sonet23v4 contradiction — one contract
hole, two tests), **#1153** (nvs19's budget non-monotonicity).

**1. A super-optimal reported objective (new, and the most severe thing here).**
Bucket E listed `assert 1.998683979470214 == 2.0 +- 1e-4` as an accuracy miss.
Widening the tolerance would have masked a soundness defect. `minimize x/y + y/x`
over a positive box has global minimum **exactly 2** by AM-GM, so 1.9987 is a
value no feasible point attains — returned with `status=optimal`. The returned
point is fine (true objective 2.000002247829649); the reported number is wrong by
-1.318268e-03 against its own incumbent.

The error is the absolute feasibility tolerance amplified by 1/denominator. Only
the quotient misbehaves — affine and **bilinear** objectives (the latter also
auxiliary-backed) match an outside oracle to 4e-16 — so it is specific to the
division reformulation, not to auxiliaries generally. Scaling the box floor
confirms the 1/y law, `|delta| x denominator` flat at ~1.9e-6 (~2x the 1e-6
tolerance, two quotient terms):

```
floor 1e-3  denom 0.00140525  delta -1.318268e-03  product 1.852e-06
floor 1e-2  denom 0.0106986   delta -1.850776e-04  product 1.980e-06
floor 1e-1  denom 8.13524     delta -4.360956e-13  product 3.548e-12
floor 1e+0  denom 479.758     delta -4.440892e-16  product 2.131e-13
```

A trace over the whole solve found **zero Python frames** returning the bad
value, so it is produced in the Rust B&B incumbent path. The general fix is to
recompute the incumbent objective from the original expression at the incumbent
point — one evaluation, and it can only correct the number, since the point is
feasible and its true objective is a valid upper bound. That is a §5
bound-changing solver change, not a test repair.

**2. Root setup still overruns its deadline (bucket B).** Re-measured at load
3.33, so not the §9 load artifact three other failures in that sweep turned out
to be: 30s budget -> 59.7s (2.0x), 60s -> 89.4s (1.5x). #875 took this from 579.3s
(19.3x) with `nodes=0`, so most of the class is fixed and a residual remains. The
1.25x threshold is deliberately **not** relaxed. Not one instance — the same
overrun appeared incidentally on `casctanks` (~321s/120s, 2.7x), `nvs19`
(80.7s/60s, 1.35x) and `sonet23v4` (4.6s/2s, 2.32x).

**3. `sonet23v4` vs bucket B — a contract contradiction, now machine-checked.**
At tl=2.0 the bound is `None`; at tl=8.0 it is -473508.0000005656 and at tl=30.0
-50587.97955518025, both sound against the -22747.5 oracle. The bound is not lost
to a truncation *bug* — the #654 gating correctly declines to **start** a 17s
operation. So this test ("spend the 17s, keep the bound") and the bucket B test
("never exceed 1.25x") assert opposing contracts on one mechanism, and
`sonet23v4` already overruns 2.32x *while declining* the op. Both hold only once
the root-relaxation LP build is made interruptible so a short budget yields a
weaker-but-present bound. That is the §3 hard fix.

**4. `nvs19`: more budget buys a worse answer (bucket F).** The test's "~11s
locally" premise is false — the optimum -1098.4 is never reached at any budget to
480s, measured on a quiet machine (load 2.63):

```
tl    wall     status      nodes    objective   bound
30    30.2s    time_limit   38403   -1098.2     -2076.33
60    80.7s    time_limit    7619   -1001.2     -7401.66
120  110.7s    feasible    100009   -1097.6     -1104.24
240  188.4s    feasible    100001   -1097.6     -1104.21
480  344.1s    feasible    100013   -1097.6     -1103.83
```

tl=30 finds -1098.2; tl=60 finds -1001.2 while exploring **5x fewer nodes** on
twice the wall clock — the #1116 role-2 signature. A performance pathology, not a
budget shortfall; raising the budget cannot fix it. At tl>=120 the search stops
on a ~100,000 **node** cap with wall under the limit. Everything is sound: every
bound is valid, every incumbent feasible and >= the optimum. No false
certificate — which is what the soundness sibling guards, and it passes.

## Repaired, because the test was wrong

* **Bucket C** — three "mechanism never fired" assertions re-pointed. The
  pool-drop retry and re-separation triggers went inert *because the source fix
  removed the condition* (C-42/C-43/C-44 precedent: the right assertion is
  `inherited_nodes >= 1` + `dropped_nodes == 0`, not `>= 1`). TX1's back-off does
  fire, just not on `nvs09` — measured `nvs09` 0/0, `casctanks` 0/0, `bchoco07`
  0/0, `tspn05` 2/2 — so the test is split and re-pointed at `tspn05`, carrying
  its own soundness assertions.
* **Bucket D** — node-count thresholds re-derived from an 18-draw sweep instead
  of nudged. Only 3 draws branch at all; the rest solve in 1-5 nodes. PSD's
  no-harm and optimum-preservation properties hold 18/18; `auto == selected
  family` 8/8. Both tests are now parametrized over the draws that actually
  branch, with a §6 probe-fired guard (`base.node_count > 20`).
* **Bucket E gauss_newton** — asserted `status=='optimal'`; **both** arms return
  `feasible` with `node_count==0` and `bound=None`, and 120 -> 600s changes
  nothing, so the root relaxation structurally yields no dual bound for a
  sum-of-squares-of-exponentials objective. The assertion is now
  soundness-shaped: `feasible` is accepted, but a claim of `optimal` must carry
  `bound <= objective`. Coverage is *strengthened* — it now also checks the
  reported objective is attained at the returned point and that both backends
  recover the generating parameters (2.0, 1.3) against a numpy oracle. Its actual
  subject (the two Hessian backends agree) holds to ~1e-12.
* **Bucket E bchoco07 drift** and **bucket A byte-identity** — the old
  comparisons ran both arms under a bare `time_limit`, which is not a comparison:
  a wall-truncated solve is cut at a machine-speed-dependent point, so the two
  arms are two different *amounts of search*. `beuster` produced **both** outcomes
  in **both** arms, proving the variation is within an arm. Both now run under
  `deterministic=True` on instances that terminate on work, with the precondition
  **asserted** (§6) rather than assumed.

## A note on the issue's prerequisite

#1039 closes with "Prerequisite for all of it: #1034." #1034 was closed by
*declining* to add a schedule — `ci.yml:15` records "NO `schedule:` trigger" — so
that line is stale. It does not block this PR, but the underlying worry is real
and unaddressed: nothing runs these tiers automatically, which is how #1038's
guard broke and stayed broken.

## Verification

* `pytest -m smoke` — **1165 passed**, 1 skipped, 2 xpassed (both pre-existing).
* `pytest -m slow python/tests/test_adversarial_recent_fixes.py` — **19 passed**.
* `cargo test -p discopt-core` — **not run; no Rust was touched** (`git diff
  main...HEAD -- crates/` is empty).
* Per-file, all at `-m "slow or not slow"`: row filter **18 passed**; #517 + #671
  **11 passed**; gp_corpus **21 passed / 1 xfailed**; gauss_newton **32 passed**;
  incumbent-injection + #875 **18 passed / 3 xfailed**; #654 **9 passed / 1
  xfailed**; GDP **322 passed** / 1 skipped / 2 xpassed; C-42 **7 passed**; cut
  policy + PSD **14 passed**; tainted-tree **2 passed**.

Every probe prints an executed-assertion count and exits non-zero at zero; none
swallows an exception; the ones that made timing claims ran interleaved with a
load gate and report a spread. Probes are in `scratchpad/issue1039/`.

## Disposition

Closes #1039 — all 18 failures triaged and the tier is green.

It should **not** be closed silently, though: the 4 strict xfails stand for 4
real defects, and per §3 each ships with a filed issue tracking the real fix,
linked from its xfail `reason`:

| issue | defect | xfail |
|---|---|---|
| #1151 | reported objective BELOW the global minimum on quotients (false certificate) | `test_gp_corpus` |
| #1152 | `time_limit` role-1: root-setup overrun vs deadline-gated bound loss, contradictory as tested | `test_875_root_setup_budget`, `test_issue654_deadline_root_setup` |
| #1153 | incumbent quality non-monotone in `time_limit` (nvs19: 2x budget, 5x fewer nodes, worse answer) | `test_incumbent_injection_soundness` |

**#1154** additionally tracks completing the hull substitution family for
`SumOverExpression` — the bound-changing half of the bucket-E fix that this PR
deliberately does not ship.

Found while verifying, and **not** part of this PR:
`test_oa.py::TestOAEdgeCases::test_infeasible_model` asserts `result.x == {}` and
gets `None`. Confirmed pre-existing by running it against `main`'s sources in
this tree (§8 marker=2). It is outside the 18 this issue triaged, so it is left
alone rather than widening the PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_014E81XM1j2J5sydzj9nFFFp

