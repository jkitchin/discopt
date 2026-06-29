# MIP-NLP upstream readiness gate

This checklist records the integration-branch state for the MIP-NLP issue series
before opening the final upstream PR from `bernalde:feature/mip-nlp-solver` to
`jkitchin:main`.

## Branch hygiene

- Fork repository: `bernalde/discopt`.
- Integration branch: `feature/mip-nlp-solver`.
- Child PR base: `bernalde:feature/mip-nlp-solver`.
- Final upstream path: one later PR from `bernalde:feature/mip-nlp-solver` into
  `jkitchin:main`.
- Current upstream baseline checked for this gate: `upstream/main` at `5d593ae`.
- Fork `main` was synced with `upstream/main` by merge commit `35a63db`.
- Integration branch includes that synced fork-main merge by merge commit
  `14db514`.

## Child issue disposition

| Issue | Scope | Integration PR | Status |
| --- | --- | --- | --- |
| #111 | Route `solver="mip-nlp"` to OA/ECP and establish the two-axis API | #122 | Merged into `feature/mip-nlp-solver` |
| #112 | OA/ECP parity baselines against MindtPy-style fixtures | #123 | Merged into `feature/mip-nlp-solver` |
| #113 | Initialization strategies: `rNLP`, `initial_binary`, `max_binary` | #124 | Merged into `feature/mip-nlp-solver` |
| #114 | OA robustness: slack, no-good cuts, feasibility norms, cycling | #125 | Merged into `feature/mip-nlp-solver` |
| #115 | Feasibility-pump method and FP initialization | #126 | Merged into `feature/mip-nlp-solver` |
| #116 | Level-set regularized OA variants | #127 | Merged into `feature/mip-nlp-solver` |
| #117 | Lagrangian and SQP regularized OA variants | #128 | Merged into `feature/mip-nlp-solver` |
| #118 | Global OA routing | #129 | Merged into `feature/mip-nlp-solver` |
| #119 | LP/NLP branch-and-bound method | #130 | Merged into `feature/mip-nlp-solver` |
| #120 | MIP-NLP solution-pool support | #131 | Merged into `feature/mip-nlp-solver` |

## Pyomo MindtPy fixture disposition

Pyomo MindtPy is a behavior and test reference only. Do not copy Pyomo source
into discopt without a separate license and attribution review.

- `MINLP3_simple.py`: incorporated in PR #128 as a cheap convex-guarantee
  regularized-OA case. It remains part of the final checklist.
- `MINLP4_simple.py`: deferred. The model is nominally a convex
  regularization-paper example, but discopt's current classifier recognizes only
  part of its reciprocal/fractional-power/geometric-mean structure as OA-valid.
  A local regularized-OA probe returned infeasible, so this belongs in a future
  convexity-classifier or regularized-OA follow-up rather than in the passing
  #117 set.
- `MINLP5_simple.py`: not a convex-OA guarantee case under the current OA
  contract. It belongs with nonconvex GOA coverage, not with the #117 convex
  regularized-OA guarantee tests.

## Validation commands

Run these before opening the upstream PR:

```bash
python -m ruff check python/
python -m ruff format --check python/
PYTHONPATH=python python -m pytest python/tests/test_mip_nlp.py python/tests/test_oa.py -q
make test
make docs
```

Optional solver-specific checks depend on installed commercial or compiled
backends:

```bash
PYTHONPATH=python python -m pytest python/tests/test_gurobi_backend.py -q
make test-amp-fast
make test-correctness
```

If a solver extra is unavailable, record the skipped backend and the reason in
the upstream PR body. Do not mark a backend as validated when the command
skipped because Gurobi, cyipopt, Ipopt libraries, or another optional dependency
was unavailable.

## Upstream PR gate

Before opening the upstream PR:

1. Confirm `feature/mip-nlp-solver` contains the current `jkitchin:main` tip.
2. Confirm all child PRs in the table above are merged or explicitly deferred.
3. Run the validation commands above on the integration branch.
4. Post the validation results on issue #121.
5. Open one upstream PR from `bernalde:feature/mip-nlp-solver` to
   `jkitchin:main`.
