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
- Current upstream baseline checked for this gate: `upstream/main` at
  `4fd2d0e`.
- Fork `main` was last synced with `upstream/main` by merge commit `35a63db`.
- Integration branch includes that synced fork-main merge by merge commit
  `14db514` and currently points at `245a55a`.
- `feature/mip-nlp-solver` does not yet contain current `upstream/main`
  (`4fd2d0e`). Sync the integration branch with `jkitchin:main` before opening
  the final upstream PR.

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
| #138 | SHOT parity audit and benchmark baseline | #155 | Merged into `feature/mip-nlp-solver` |
| #139 | SHOT MIP-NLP profile and trace foundation | #154 | Merged into `feature/mip-nlp-solver` |
| #140 | Unified MIP-NLP cut provenance | #156 | Merged into `feature/mip-nlp-solver` |
| #141 | SHOT reformulation controls | #157 | Merged into `feature/mip-nlp-solver` |
| #142 | Initial POA and relaxation-phase seeding | #158 | Merged into `feature/mip-nlp-solver` |
| #143 | Interior-point and rootsearch services | #159 | Merged into `feature/mip-nlp-solver` |
| #144 | ESH and objective-rootsearch hyperplanes | #160 | Merged into `feature/mip-nlp-solver` |
| #145 | SHOT-style MultiTree master orchestration | #161 | Merged into `feature/mip-nlp-solver` |
| #146 | Fixed-NLP candidate manager | #162 | Merged into `feature/mip-nlp-solver` |
| #147 | General-integer no-good cuts and solution-pool parity | #165 | Merged into `feature/mip-nlp-solver` |
| #148 | Master repair and nonconvex reduction cuts | #166 | Merged into `feature/mip-nlp-solver` |
| #149 | Convex bounding master for nonconvex/local-cut runs | #167 | Merged into `feature/mip-nlp-solver` |
| #150 | SingleTree callback parity | #168 | Merged into `feature/mip-nlp-solver` |
| #151 | Direct NLP/LP/MILP/MIQP/QCP/MIQCQP routing | #169 | Merged into `feature/mip-nlp-solver` |
| #152 | External MIP-NLP event hooks | #170 | Merged into `feature/mip-nlp-solver` |
| #153 | Integration docs, benchmark refresh, and release checklist | This PR | Merge this checklist/docs PR before upstream readiness signoff |

Supporting guard PR #164 also landed on `feature/mip-nlp-solver` to keep local
SHOT cuts from excluding known incumbents.

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

Run these default-CI and release-documentation checks before opening the
upstream PR:

```bash
ruff check python/
ruff format --check python/
pytest python/tests/test_mip_nlp.py python/tests/test_oa.py python/tests/test_shot_parity_baseline.py -q
PYTHONPATH=python python scripts/shot_parity_baseline.py \
  --output docs/dev/data/shot-parity-baseline.json
make test
make docs
```

Run these broader release gates when the integration branch is ready for the
upstream PR:

```bash
make test-all
make test-amp-fast
make test-correctness
make bench-smoke
make perf-gate
```

Optional solver-specific checks depend on installed commercial, compiled, or
external backends:

```bash
pytest python/tests/test_gurobi_backend.py -q
pytest python/tests/test_mip_nlp.py -q -k "gurobi or single_tree or solution_pool"
SHOT_EXECUTABLE=/home/bernalde/repos/SHOT/build/SHOT \
PYTHONPATH=python python scripts/shot_parity_baseline.py \
  --output docs/dev/data/shot-parity-baseline.json \
  --workdir /tmp/discopt-shot-baseline
make gams-verify
```

If a solver extra is unavailable, record the skipped backend and the reason in
the upstream PR body. Do not mark a backend as validated when the command
skipped because Gurobi, cyipopt, Ipopt libraries, or another optional dependency
was unavailable. Default CI must remain free of commercial solver requirements.

## Upstream PR gate

Before opening the upstream PR:

1. Sync `bernalde:main` with current `jkitchin:main`, then merge that updated
   main into `feature/mip-nlp-solver`.
2. Confirm `feature/mip-nlp-solver` contains the current `jkitchin:main` tip.
3. Confirm all child PRs in the table above are merged or explicitly deferred.
4. Regenerate `docs/dev/data/shot-parity-baseline.json` and keep the SHOT rows
   only if a built SHOT executable was actually used.
5. Run the validation commands above on the integration branch and record exact
   skipped optional backends.
6. Post the validation results on issue #121.
7. Open one upstream PR from `bernalde:feature/mip-nlp-solver` to
   `jkitchin:main`.
