# #1160 — `dm.sum(X, axis=k)` was solved as the axis-collapsed model

Measurement harnesses for the fix. Every script prints an executed-comparison
count and exits non-zero when it made none (CLAUDE.md §6).

| script | what it measures |
|---|---|
| `repro.py` | the issue's reproducer verbatim (`sum(A, axis=1) <= 2`, true optimum -4) |
| `probe_paths.py` | which LP extractor claims the model and how many rows each emits |
| `panel.py` | 6 axis-reduced models (linear axis=0/1, convex quadratic, nonconvex `>=`, bilinear, `exp`) against hand-derived optima, checking objective **and** `bound <= true optimum` |
| `panel2.py` | 7 more (MILP, MIQP, QP, least-squares, `==` sense, 3-D operand, 1-D `axis=0`) |
| `nl_neutrality.py` | bound-neutrality on the in-repo 66-instance `.nl` corpus: counts guard invocations during **real solves** (with an instrument self-test that fails if the counter is dead) and dumps per-instance status/objective/bound/node_count for an A/B diff |

## Results

Baseline (`570ca9a`), `panel.py` + `panel2.py`: **4 of 12 wrong** —
`lin_axis1` -2 (true -4), `lin_axis0` -1 (true -3), `milp_axis1` -2 (true -4),
`lin_axis_3d` -1 (true -4). Each returned `status="optimal"` with the wrong
value *as the dual bound*, while `verify_point` confirmed the true optimum's
point feasible. After the fix: **12 of 12 correct**, every bound at or below the
true optimum.

`nl_neutrality.py` on the 66 in-repo `.nl` instances (10 s / 3000 nodes each):
`GUARD_CALLS: 0` — the guarded branch is not merely inert, it is never reached,
measured during real solves (the instrument's self-test on a Python-API model
counts 3 calls / 2 refusals in the same process). The A/B over 264 compared
fields differs in 8, all `bound`/`node_count` on time-limited instances; three
repetitions of *each* arm produce the other arm's values (see the PR).
