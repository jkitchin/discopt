# discopt: An Optimization Course

A self-paced, Claude-Code-assisted course in mathematical optimization, taught
through the `discopt` solver. Three tracks (basic, intermediate, advanced)
totalling **30 lessons**, each pairing a reading notebook with an exercise
notebook, a writing prompt, and a rubric. Claude Code acts as your TA: it
delivers material, answers questions, hints when you're stuck, and grades your
work against the rubric.

## How the course works

Each lesson lives in `course/<track>/<id>/` and contains:

| File             | Purpose                                                       |
| ---------------- | ------------------------------------------------------------- |
| `reading.ipynb`  | The lesson — math, examples, runnable code with `discopt`.    |
| `exercises.ipynb`| 3-6 exercises with `# TODO` cells you fill in.                |
| `writing.md`     | A short essay or analysis prompt (≈ 1 page).                  |
| `rubric.md`      | The criteria Claude uses to assess your work.                 |

Reference solutions live under `course/solutions/<track>/<id>/`. Try the
exercise yourself before peeking — Claude can give hints without revealing
the answer.

## Quick start

```bash
# 1. Install discopt with the optional notebook deps
pip install -e ".[dev,nn,llm]"
jupyter lab

# 2. Install the course slash commands and assessor skill into .claude/
bash course/install_skills.sh             # project-scope (./.claude/)
# or
bash course/install_skills.sh --user      # global (~/.claude/)

# 3. Open the syllabus
open course/SYLLABUS.md

# 4. In Claude Code, start lesson 1
/course:lesson basic/01_intro_to_optimization

# 5. When you finish the exercises, ask Claude to assess
/course:assess basic/01_intro_to_optimization
```

The slash commands and assessor skill ship as source files under
`course/_claude_assets/`. The install script copies them into `.claude/`
where Claude Code will pick them up. Source-of-truth lives under
`_claude_assets/` so they're version-controlled with the rest of the course.

## Slash commands

The course ships with these Claude Code commands:

- `/course:lesson <id>` — load a lesson, get a guided tour of the reading.
- `/course:hint <id> <ex#>` — get a graduated hint for a specific exercise.
- `/course:assess <id>` — Claude grades your exercises against the rubric.
- `/course:grade-writing <id>` — Claude grades your `writing.md` response.
- `/course:progress` — show what you've completed and what's next.
- `/course:cite-check` — verify all `{cite:p}` keys resolve in `references.bib`.

The grading behaviour is implemented by the `course-assessor` skill in
`.claude/skills/course-assessor/`, so it is consistent across sessions.

## Tracks

- **Basic (1–10)**: modeling, LP, duality, IP, B&B, convexity, NLP. Aimed at
  someone who has had calculus and linear algebra and wants a working
  knowledge of optimization plus the `discopt` API.
- **Intermediate (11–20)**: simplex internals, IPM, cuts, branching rules,
  presolve, relaxations, conic optimization, decomposition, stochastic
  programming. Algorithms-first; you'll implement small pieces and benchmark
  against `discopt`'s defaults.
- **Advanced (21–30)**: spatial B&B, global optimization theory, GDP,
  NN-embedded MINLP, robust optimization, ML-for-OR, differentiable
  optimization, bilevel, reformulations. Capstone is a paper-style writeup
  with computational results.

## Progress tracking

Your state is recorded in `course/progress.yaml` (created on first
`/course:progress` call from `progress.template.yaml`). Lessons unlock in
order within a track, but tracks can be done in parallel.

## Citations

Every reading cites the literature with `{cite:p}` / `{cite:t}` keys
resolving to entries in `docs/references.bib`. New course-specific entries
were added when this course was built. Run `/course:cite-check` to verify.

## Building / regenerating notebooks

The notebooks are generated from Python content modules in
`course/_build/lessons/`:

```bash
python course/_build/nbgen.py
```

This rewrites every `reading.ipynb` and `exercises.ipynb`. Edit the `.py`
modules — not the JSON — and rebuild.

## Course-API shim

The lesson code cells use a slightly-friendlier façade
(`m.add_variable / .add_variables / .add_constraint / .is_convex / .solve`
with kwargs like `mode`, `verbose`, `x0`) that maps onto the real `discopt`
API (`m.continuous / .binary / .integer / .subject_to / .solve`). The shim
lives in `course/_compat.py`; `nbgen.py` auto-injects a loader cell at the
top of every notebook so the shim is installed before the lesson code runs.
You can remove the shim when comfortable working directly with the real
`discopt.modeling.core.Model` API.
