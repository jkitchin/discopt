---
description: Grade the student's exercises (and writing if present) for a lesson
argument-hint: <track>/<id> [--milestone N]
---

The student has asked for assessment of `$ARGUMENTS`.

**Argument parsing.** `$ARGUMENTS` is `<track>/<id>` plus optional flags:

- `--milestone <N>` — for the advanced capstone (`advanced/30_capstone_advanced`),
  grade only milestone N (1, 2, or 3): M1 = proposal, M2 = method + initial
  results, M3 = final paper. Look for `course/<track>/<id>/milestone_<N>.md`
  and grade against the milestone-specific subset of the rubric. Reject
  with a clear message if `--milestone` is passed for a non-capstone lesson.

**Default flow.** Invoke the `course-assessor` skill. The skill knows the
procedure:

- Read `course/<track>/<id>/rubric.md`.
- Read the student's `course/<track>/<id>/exercises.ipynb`.
- Read the optional `course/<track>/<id>/writing_response.md`.
- Use the reference solution at `course/solutions/<track>/<id>/exercises.ipynb`
  (instructor-only path; the `solutions/` tree is gitignored on student
  distributions, so this read may fail in a student clone — degrade
  gracefully and grade only against the rubric in that case).

**Workspace safety.** The student's `exercises.ipynb` is the live file they
edit. Do NOT overwrite it during grading. Do NOT regenerate it via
`python course/_build/nbgen.py`; nbgen has overwrite-safety that detects
student edits, but the assessor must never touch the file regardless.

**Progress write.** After grading, atomically update
`course/progress.yaml` under `scores.<track>/<id>`:

- preserve any pre-existing `exercises`, `writing`, `attempts`, `hints_used`
  fields you don't overwrite;
- recompute `total = exercises + writing` only when both subscores exist;
- append `last_assessed: <ISO date>`.

If `course/progress.yaml` doesn't yet exist, copy
`course/progress.template.yaml` first.

Produce the structured feedback report and update `progress.yaml`.
