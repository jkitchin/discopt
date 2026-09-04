# #1147 — complementarity provenance: the measurements

* `repro_1147.py` — the entry experiment. Prints the pair count before/after each
  rebuilding pass. On `main`: `after=0` for GDP (big-m/hull/mbigm),
  `expand_integer_products` and `factorable_reformulate`; `binary_multilinear`
  abstains. After the fix: `after=1` on all of them.
* `panel_1147.py` — the bound-neutral panel over the 66-instance in-repo `.nl`
  corpus. Asserts the loaded module and a version marker (CLAUDE.md §8), prints
  per-instance progress (§10) and an executed-solve count (§6).
  `panel_before.json` / `panel_after.json` are its two arms at a 10 s limit
  (`marker: false` / `marker: true`). Diff: status and objective identical
  66/66; every recorded field identical on all 44 converged instances.
* `probe_two.py` — the two instances whose `node_count` differed on the panel
  (`bchoco07`, `tls2`). Both are unconverged (`time_limit` / `feasible`) runs,
  and both values reproduce *within* an arm across interleaved repeats
  (2 rounds x 3 reps per arm), so the panel difference is a wall-clock artifact,
  not a bound change (CLAUDE.md §9).

Run from the repository root.

## Full non-slow suite

`pytest python/tests/ -m "not slow" -n 4` on this branch: **8 failed, 9119 passed,
149 skipped, 9 xfailed, 2 xpassed** (638 s). Triaged:

* 4 are load-induced flakes of the 4-way parallel run — `test_lp_huge_finite_box_937`
  and the three `test_decomposition_adversarial::test_rand_lagrangian_dual_is_valid_lower_bound`
  parameters all PASS when re-run serially on this branch.
* 4 are pre-existing in this container and fail **identically on `main`** (verified by
  `git checkout <main> -- python/discopt`, asserting the version marker absent, then
  restoring): `test_convex_nlp_certificate_853::test_neglog_interior_stall_not_false_optimal[ipopt]`,
  `test_75_tape_nlp_evaluator::test_solve_degrades_to_jax_when_pounce_is_missing`,
  `test_relax_compiler_convexity_units::TestDifferentiableSolvePaths::test_ipopt_backend_active_constraint_gradient`,
  `test_issue_1066_master_bound_inversion::test_lp_nlp_bb_refuses_a_master_bound_above_its_incumbent`.
  They depend on optional backends (cyipopt / POUNCE behaviour) this container does not
  provide; CI is the authority on them.

Zero failures in any file this change touches.
