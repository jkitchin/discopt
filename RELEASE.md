# Release Checklist

This is the authoritative procedure for cutting a discopt release. Copy this
checklist into the release tracking issue/PR and tick items as you go. Do not
skip steps -- CI gates are defense-in-depth, not a substitute for working
through this list.

For each release, fill in:

- `vX.Y.Z` -- the target version
- `vA.B.C` -- the previous released version

The next planned release is **`v0.2.6`** (patch on top of `v0.2.5`).

---

## 1. Pre-flight

- [ ] Working tree is clean (`git status`) and on `main`, up to date with `origin/main`.
- [ ] No open issues tagged `release-blocker` on GitHub.
- [ ] Decide the version bump (major / minor / patch) against SemVer; record the rationale in the release tracking issue.
- [ ] Create a release branch: `git checkout -b release/vX.Y.Z`.

## 2. Correctness and tests (must all pass -- zero tolerance)

- [ ] `cargo test -p discopt-core` -- all Rust tests green.
- [ ] `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest python/tests/ -v` -- full Python suite green.
- [ ] `pytest python/tests/ --cov=discopt` -- coverage >= 85%.
- [ ] `pytest discopt_benchmarks/tests/ -v` -- benchmark suite green.
- [ ] `pytest python/tests/test_correctness.py -v` -- **`incorrect_count == 0`**. Non-negotiable per `CLAUDE.md`.
- [ ] `make bench-smoke` -- smoke benchmark passes.
- [ ] Phase gates relevant to this release. Each `--gate phaseN` check reads the matching suite's most recent results in `results/`, so run `--suite phaseN` first. These runs are expensive (each suite uses a 3600 s per-instance time limit) and are typically a CI-only step for patch releases:
  - [ ] `python discopt_benchmarks/run_benchmarks.py --suite phase1 && python discopt_benchmarks/run_benchmarks.py --gate phase1`
  - [ ] `python discopt_benchmarks/run_benchmarks.py --suite phase3 && python discopt_benchmarks/run_benchmarks.py --gate phase3`
  - [ ] `phase4` currently has gate criteria but no registered suite in `discopt_benchmarks/config/benchmarks.toml`. Skip until a `[suites.phase4]` entry is added; if relevant to the release, fix the suite definition first.
  - [ ] Review `reports/phase*_gate_report.md` -- no regressions vs. the previous release.

## 3. Lint, format, types

- [ ] `pre-commit run --all-files` -- clean. This runs `ruff` (lint + format), `mypy`, and `cargo fmt`.
- [ ] Confirm `ruff` version in `.pre-commit-config.yaml` and `.github/workflows/ci.yml` still match (currently pinned to `0.14.6`).

## 4. Documentation

- [ ] `README.md` reviewed -- install commands, feature list, minimum Python/Rust versions, and badges all accurate.
- [ ] `jupyter-book build docs/` -- **zero warnings** (project rule per `CLAUDE.md`; fix any that appear).
- [ ] New features since the last release have a notebook or tutorial entry under `docs/notebooks/` and are linked from `docs/_toc.yml`.
- [ ] New citation keys added to `docs/references.bib`; rendered page `docs/references.md` still builds.
- [ ] `make notebooks` -- every notebook in `docs/notebooks/` and `manuscript/` executes end-to-end without error (600 s timeout per notebook).
- [ ] `docs/intro.md` landing page reflects the headline features of this release.
- [ ] API docs (`autoapi`) regenerated as part of the Jupyter Book build -- spot-check that a few new public symbols appear.

## 5. Manuscript

The manuscript source of truth is `manuscript/discopt.org` (org-mode). The
`discopt.tex` file is **generated** from the .org file -- never edit `.tex`
directly; always re-export from org.

- [ ] `manuscript/discopt.org` reviewed -- narrative covers any new features in this release.
- [ ] Re-export `discopt.org` -> `discopt.pdf` in one step with:
      `scimax export manuscript/discopt.org --format pdf`
      (This re-runs the org export and compiles the .tex to a fresh .pdf. You can also use `C-c C-e l p` from inside Emacs / scimax if you prefer.)
- [ ] Confirm no LaTeX warnings about undefined references or missing citations in the export output.
- [ ] Re-execute manuscript notebooks against current code:
  - [ ] `manuscript/benchmark_lp_qp_speedup.ipynb`
  - [ ] `manuscript/minlplib_smoke_test.ipynb`
  - [ ] `manuscript/piecewise_gap_reduction.ipynb`
- [ ] Numbers and figures in `discopt.pdf` match the latest phase gate reports in `reports/` and the refreshed manuscript notebooks above.
- [ ] Author list, affiliations, and acknowledgments still correct.
- [ ] Citation keys resolve against `manuscript/references.bib`.
- [ ] If preparing an arXiv submission for this release: build a tarball with `discopt.tex`, `references.bib`, `manuscript/figures/`, and `discopt.bbl`, then compile once in a clean directory to confirm it builds standalone.

## 6. Roadmap and project tracking

- [ ] `ROADMAP.md` updated -- completed phases/items marked, next phase highlighted.
- [ ] `tasks.org` reconciled (if used).
- [ ] Any `reports/*.md` that describe project state refreshed for this release.

## 7. Changelog

- [ ] `CHANGELOG.md` has a new `## [X.Y.Z] - YYYY-MM-DD` section with entries grouped under **Added / Changed / Deprecated / Removed / Fixed / Security**.
- [ ] Everything previously under `## [Unreleased]` has been moved into the new version section.
- [ ] Entries cite PR numbers / commit hashes where helpful.
- [ ] Compare-link footnotes at the bottom of `CHANGELOG.md` updated:
  `[X.Y.Z]: https://github.com/jkitchin/discopt/compare/vA.B.C...vX.Y.Z`

## 8. Version bump

- [ ] `pyproject.toml` -- bump `version = "X.Y.Z"`.
- [ ] `python/discopt/__init__.py` -- bump `__version__ = "X.Y.Z"`.
- [ ] `CITATION.cff` -- bump `version: X.Y.Z` to match the tag. (`.zenodo.json` carries no version; Zenodo derives it from the git tag when the GitHub release is archived.)
- [ ] Grep for hard-coded version strings that might have been missed:
  `rg -n '\bA\.B\.C\b' --glob '!CHANGELOG.md' --glob '!docs/_build' --glob '!target'`
  (substitute the previous version `A.B.C`).
- [ ] Rebuild the Rust extension locally (`make build` or `cd crates/discopt-python && maturin develop`) and re-run `pytest python/tests/ -m "not slow"` to confirm the bump didn't break imports.
- [ ] `python -c "import discopt; print(discopt.__version__)"` prints the new version.

## 9. Review and merge

- [ ] Open a PR from `release/vX.Y.Z` -> `main`, titled `release: vX.Y.Z`.
- [ ] CI green on the PR (all matrix cells).
- [ ] At least one reviewer approval.
- [ ] Merge into `main` (squash or merge-commit, per project convention).

## 10. Tag and publish

- [ ] `git checkout main && git pull`.
- [ ] `git tag -a vX.Y.Z -m "discopt vX.Y.Z"` -- annotated tag on the merge commit.
- [ ] `git push origin vX.Y.Z` -- this triggers `.github/workflows/release.yml`.
- [ ] Watch the release workflow: wheels build for Linux (x86_64 + aarch64), macOS (x86_64 + aarch64), and Windows (x64); sdist built; PyPI upload succeeds.
- [ ] Verify on PyPI that `https://pypi.org/project/discopt/X.Y.Z/` exists and lists the expected wheels.
- [ ] In a clean virtualenv: `pip install discopt==X.Y.Z` then `python -c "import discopt; print(discopt.__version__)"`.

## 11. GitHub Release

- [ ] Draft a GitHub Release for tag `vX.Y.Z`.
- [ ] Title: `discopt vX.Y.Z`.
- [ ] Body: paste the new `CHANGELOG.md` section plus a short "Highlights" paragraph.
- [ ] Attach any relevant phase gate reports or benchmark plots.
- [ ] Publish the release (not draft).

## 12. Post-release

- [ ] Bump `pyproject.toml` and `python/discopt/__init__.py` to the next dev version (e.g. `X.Y.(Z+1).dev0`) on a follow-up commit to `main`.
- [ ] Add a fresh `## [Unreleased]` header to `CHANGELOG.md`.
- [ ] Announce the release on the project's usual channels.
- [ ] Close the release tracking issue.
- [ ] Open issues for any known gaps surfaced during the release that were not blockers.
