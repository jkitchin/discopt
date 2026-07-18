<!--
Title: scope the PR and name the task/issue ID, e.g.
  fix(correctness): C-16 …   |   feat(#282): …   |   perf(sparse-milp): …
Keep the PR to one task/issue. See CLAUDE.md → Workflow.
-->

## Summary

<!-- What changed and why, in a few sentences. Link the issue: "Closes #NNN" /
"Contributes to #NNN". Use "Contributes to" (not "Closes") when the change is
partial or ships behind a default-off flag. -->

## Correctness

<!-- The product of a global solver is its certificate. Confirm this change cannot
produce a false optimal / infeasible / bound. If it touches a bound, cut, or
relaxation, say why it stays sound (a strengthened bound must never exceed the
true optimum). -->

- [ ] Cannot produce a false certificate (or: N/A — no solver-math change)
- [ ] No validation, fallback, or safety guard was weakened to make a check pass

## Tests run (state the result — numbers, not adjectives)

- [ ] `pytest -m smoke` → <!-- e.g. 666 passed, 1 skipped -->
- [ ] `pytest -m slow python/tests/test_adversarial_recent_fixes.py` → <!-- e.g. 10 passed -->
- [ ] `cargo test -p discopt-core` → <!-- required only if Rust was touched; else N/A -->
- [ ] `ruff check` / `ruff format --check` clean

## Regression test

<!-- New behavior requires a test that FAILS before this change and PASSES after.
Name it, and confirm the fail-before/pass-after. -->

- [ ] Added a regression test that fails before / passes after (or: N/A — docs/refactor only)

## Bound-changing / performance change? (delete this section if N/A)

<!-- CLAUDE.md §5. A bound-changing change ships behind a DEFAULT-OFF flag until a
graduation panel passes. Include the measurement — suite, baseline, numbers — not
adjectives. -->

- [ ] Default-OFF flag: `DISCOPT_…` (flag-OFF path byte-identical)
- [ ] Differential-bound test (new bound ≥ old AND ≤ true box optimum) + feasible-point sampling
- [ ] Graduation panel evidence attached (flag ON vs OFF: cert-clean `incorrect_count=0` AND net-positive), or: not graduating — default-off only
- Measurement: <!-- suite / baseline / numbers -->
