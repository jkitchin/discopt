---
description: Grade only the writing assignment for a lesson
argument-hint: <track>/<id>
---

Grade only the writing assignment for lesson `$ARGUMENTS`.

Invoke the `course-assessor` skill. Read:

- `course/$ARGUMENTS/writing.md` — the prompt.
- `course/$ARGUMENTS/rubric.md` — the writing-specific criteria (typically the
  bottom 30 points).
- `course/$ARGUMENTS/writing_response.md` — the student's response.

If `writing_response.md` does not exist, ask the student to create it (you may
suggest a starter outline based on the prompt) and stop.

Otherwise, score it on:
- clarity
- technical correctness
- use of citations (verify every `{cite:p}` key resolves in
  `docs/references.bib`; flag invented or unverifiable keys explicitly)
- engagement with the prompt

**Preserve the exercises score.** Read the existing
`scores.<lesson>` block first. Update only the `writing` field. Recompute
`total = exercises + writing` ONLY if `exercises` is already set; if
`exercises` is unset, write `total: null` and leave a note that the student
should run `/course:assess <lesson>` before the writing total is meaningful.

Append `last_assessed_writing: <ISO date>`.
