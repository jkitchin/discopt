---
description: Get a graduated hint for a course exercise (does not reveal the answer)
argument-hint: <track>/<id> <exercise-number>
---

The student is stuck on an exercise. Arguments: `$ARGUMENTS` (lesson id and
exercise number).

Use the `course-assessor` skill in **hint mode** (per its SKILL.md). The
protocol is graduated:

1. First time asked → give a general nudge ("re-read section X").
2. Second time asked → give a specific pointer ("count your dual variables").
3. Third time asked → give pseudocode/structure but never the literal answer.

Look up the previous hint level in `course/progress.yaml` under
`scores.<lesson>.hints_used` (it may not yet exist; treat absence as 0).
Increment after giving the hint.

If the student asks for the answer outright, decline politely and explain
that struggling productively with the exercise is the point.
