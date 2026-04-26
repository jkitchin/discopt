---
description: Grade the student's exercises (and writing if present) for a lesson
argument-hint: <track>/<id>
---

The student has asked for assessment of lesson `$ARGUMENTS`.

Invoke the `course-assessor` skill. The skill knows the procedure: read the
rubric at `course/$ARGUMENTS/rubric.md`, the student's
`course/$ARGUMENTS/exercises.ipynb`, the optional
`course/$ARGUMENTS/writing_response.md`, and grade against the reference
solution at `course/solutions/$ARGUMENTS/exercises.ipynb`.

Produce the structured feedback report and update `course/progress.yaml`.
