# #1147 — complementarity provenance: the measurements

Run everything from the repository root.

* `repro_1147.py` — the entry experiment. Prints the pair count before/after each
  rebuilding pass. On `main`: `after=0` for GDP (big-m/hull/mbigm),
  `expand_integer_products` and `factorable_reformulate`; `binary_multilinear`
  abstains. After the fix: `after=1` on all of them.
* `panel_1147.py` — the bound-neutral panel over the 66-instance in-repo `.nl`
  corpus. Asserts the loaded module and a version marker (CLAUDE.md §8), prints
  per-instance progress (§10) and an executed-solve count (§6).
  Usage: `python -u scratchpad/i1147/panel_1147.py <arm-name> <out.json> [time_limit]`.
* `probe_two.py` — re-runs the instances whose `node_count` moved between panel
  arms, several times per arm, to separate a real bound change from a wall-clock
  artifact on a time-limited run (CLAUDE.md §9).
* `verify_review.py` / `verify_review2.py` — reproduce the findings of the two
  PR #1149 reviews and re-check them after each fix.

## The panel that stands

`panel_before3.json` / `panel_after3.json` — `main` vs. the fixed head, 10 s per
instance, **both arms on the same boot of an idle host** (`marker: false` /
`marker: true` respectively):

* status identical **66/66**;
* every recorded field (status, objective, bound, `node_count`, `gap_certified`)
  identical on all **45** instances that converged in both arms;
* the 7 differing rows are all unconverged (`time_limit` / `feasible`) ones whose
  node count tracks how much wall clock the fixed budget buys — not a bound change.

### Two superseded pairs, and why

Two earlier arm-pairs were measured and then **retracted** (CLAUDE.md §11); their
raw JSON is not kept, since only the pair above supports a claim.

1. The first pair compared arms measured under *different* machine load — the
   container restarted between them — so their unconverged rows were never
   comparable. 13 rows moved across those conditions, and the pre-review-fix arm
   tracked `main` on almost all of them: the host talking, not the code. It had
   been reported as "identical 66/66, 44 converged, 2 `node_count` differences".
2. The second pair was matched (both arms idle) but on a boot that has since been
   replaced, so it could not speak for the current head. It had been reported as
   "45 converged, 4 differences".

The lesson, recorded because it cost three rounds: an A/B over a **wall-clock
budget** is only comparable when both arms run on the same host in the same
state, and only its *converged* rows can carry a bound-neutrality claim at all.

## Full non-slow suite

`pytest python/tests/ -m "not slow" -n 4`: **8 failed, 9119 passed, 149 skipped,
9 xfailed, 2 xpassed** (638 s). Triaged, none in a file this change touches:

* 4 are load-induced flakes of the 4-way parallel run — `test_lp_huge_finite_box_937`
  and the three `test_decomposition_adversarial::test_rand_lagrangian_dual_is_valid_lower_bound`
  parameters all PASS when re-run serially on this branch.
* 4 are pre-existing in this container and fail **identically on `main`** (verified by
  `git checkout <main> -- python/discopt`, asserting the version marker absent, then
  restoring): `test_convex_nlp_certificate_853::test_neglog_interior_stall_not_false_optimal[ipopt]`,
  `test_75_tape_nlp_evaluator::test_solve_degrades_to_jax_when_pounce_is_missing`,
  `test_relax_compiler_convexity_units::TestDifferentiableSolvePaths::test_ipopt_backend_active_constraint_gradient`,
  `test_issue_1066_master_bound_inversion::test_lp_nlp_bb_refuses_a_master_bound_above_its_incumbent`.
  They depend on optional backends (cyipopt / POUNCE) this container does not provide;
  CI is the authority on them.

## What the reviews found

`verify_review.py` — all eight findings of the first review, each reproduced
before being fixed. HIGH 1 returned a **certified `optimal` at `z = 1`** on
`max z s.t. mcp(z+1, z), z in [-1,1]`, where the MCP's `z = u` branch requires
`F <= 0` and `F = +2`.

`verify_review2.py` — the two blocking findings of review 5115268644:

* **Blocking 1** — `flat_source_indices` read offsets off `Variable._index` (the
  *declaring* model's position). A target model holding a two-element `x` and a
  scalar `y` as `[y, x]` returned `[0, 1, 1]`; its true layout is `[1, 2, 0]`.
  Reproduced exactly, including the predicted `[0, 1, 1]`.
* **Blocking 2** — the lowering *method* was a plain field on the shared relation.
  GDP into `m1` then SOS1 into `m2` left it reading `"sos1"`; and
  `carry_complementarities` took any non-`None` value as proof that `src` was
  lowered, so an unlowered `src` handed `dst` a lowered mark for rows neither
  model carried — walking past the solve guard added for the first review's HIGH 1.
