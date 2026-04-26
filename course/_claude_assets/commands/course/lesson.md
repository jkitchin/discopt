---
description: Load a discopt-course lesson and walk the student through the reading
argument-hint: <track>/<id>   (e.g. basic/02_lp_fundamentals)
---

You are guiding a student through a discopt-course lesson. The lesson id is
`$ARGUMENTS`. Steps:

1. Read `course/$ARGUMENTS/reading.ipynb`. (Use Read; for notebooks the tool
   returns cell contents.)
2. Give a 5-bullet preview: the lesson's learning objectives.
3. Ask the student which they'd like to do first:
   (a) work through the reading interactively here in the chat,
   (b) open the notebook in JupyterLab and read it themselves and ask Claude
       questions as they go,
   (c) skip straight to the exercises.
4. If (a), step through the markdown sections in order, pausing for questions.
5. If the student finishes the reading, suggest they run
   `/course:hint $ARGUMENTS <ex#>` if they get stuck on any exercise, and
   `/course:assess $ARGUMENTS` when done.

Update `course/progress.yaml` under `current_lesson` to `$ARGUMENTS`.
