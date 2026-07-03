# Tutor Module Review — Correctness, Thoroughness, and Design Assessment

**Date:** 2026-07-03
**Scope:** `python/discopt/tutor.py` (534 lines), the packaged course it fronts
(`python/discopt/course/` — 30 lessons, `_claude_assets/` slash commands +
`course-assessor` skill, `install_skills.sh`, progress template), and
`python/tests/test_cli_tutor.py` (542 lines).
**Method:** Full read of the CLI, the six slash-command prompts, the assessor
skill, and the course tree structure; logic suspects exercised directly; the
packaging promises verified by **building a wheel** and inspecting it. Baseline:
**46 tests passed** (0.4 s).

This is the healthiest module reviewed in this series. The findings are minor
CLI-logic and UX items plus two system-level design observations; nothing touches
solver math, and the load-bearing promises all verify.

---

## 1. Verified correct (the load-bearing claims)

- **Wheel packaging** — the module's core promise ("ships as package data so a
  pip install includes it") **holds**: a locally built wheel contains all 133
  course files, including all 60 notebooks, the slash commands, the skill, and
  the progress template. (The explicit `[tool.maturin] include` entries for
  `gams/data` had suggested data files might need listing; they don't for
  package-directory data — hypothesis tested and refuted.)
- **No instructor-solutions leak** — `solutions/` is absent from the repo,
  `.gitignore`d, and a planted `course/solutions/leak_probe.txt` did **not**
  enter a rebuilt wheel: maturin honors `.gitignore` for wheel contents. The
  same protects a stray instructor `progress.yaml`.
- **Course completeness** — all 30 lessons (3 tracks × 10) carry all four
  artifacts (`reading.ipynb`, `exercises.ipynb`, `rubric.md`, `writing.md`);
  numbering is globally sequential (01–30) so shorthand resolution has no
  cross-track collisions; rubric weights (exercises 70 + writing 30 = 100) are
  consistent with the assess template and make the ≥70 pass threshold coherent.
- **Careful prompt engineering** in the slash commands: `lesson.md` explicitly
  forbids clobbering `current_lesson` on a preview; `assess.md` has a
  workspace-safety section (never overwrite the student's notebook) and a
  graceful-degradation path for the instructor-only solutions; the assessor
  skill's feedback template demands specific, criterion-level feedback.
- **CLI hygiene**: env-var > walk-up > packaged resolution precedence (tested);
  read-only-packaged detection with an actionable hint; `install` idempotent
  (skip-unless-`--force`) and deliberately sourced from the packaged copy (with
  the reason documented in code); slash-command-missing pre-flight warning; the
  test suite covers resolution, ordering, dashboards, reset, and install paths
  properly.

---

## 2. Findings (all P2–P3)

| ID | Loc | Finding |
|----|-----|---------|
| TU-1 | `tutor.py:256-262` | `_next_lesson`: when `current` is passed it returns `lessons[idx+1]` **without skipping already-passed lessons** — `discopt tutor next` relaunches a completed lesson for students who worked out of order [CONFIRMED: current=01 passed, 02 passed → returns 02]. Fix: first *unpassed* lesson after `idx`. The no-current branch already does this; make them consistent. |
| TU-2 | `tutor.py:350` | `discopt tutor reset` (all) calls `input()` — on piped/CI/non-interactive stdin it dies with an **EOFError traceback** [CONFIRMED]. Catch `EOFError`/`KeyboardInterrupt` → "aborted", and add `--yes` for scripted use. |
| TU-3 | `tutor.py:94-112` | Resolution UX: `"2"` fails (no zero-pad normalization), `"lp"`/`"capstone"` fail (prefix-only, no substring fallback), and *ambiguous* input produces the same "unknown lesson" message as garbage — the user can't tell they nearly matched. Fix: zero-pad bare numbers, add substring fallback, and print "ambiguous: matches a, b" on multi-match. |
| TU-4 | `tutor.py:525-531` | With `DISCOPT_COURSE_DIR` set but invalid, `_find_course_dir` short-circuits to `None` (a defensible choice, and tested) — but the error message then advises "…or set DISCOPT_COURSE_DIR", which the user already did. Detect the case and say the path in the variable is missing `SYLLABUS.md`. |
| TU-5 | `tutor.py:246`, `progress.md:21`, 30× `rubric.md` | The pass threshold (70) and the 70/30 split are duplicated across the CLI, the prompt files, and every rubric. Single-source them (e.g. a small `course/course.yaml` the CLI and prompts both read) so a future re-weighting can't half-apply. Low priority while the numbers are stable. |
| TU-6 | `tutor.py:163-181` | `_slash_command_installed` checks `./.claude` and `~/.claude` only, but Claude Code resolves project commands by walking **up** from cwd — running from a subdirectory of a project that has `.claude/` at its root triggers a spurious "not installed" warning before a launch that would actually work. Walk parents like `_find_course_dir` does. |
| TU-7 | `tutor.py:370-372` | Per-lesson reset assumes `yaml.safe_load` returns a dict — a hand-mangled `progress.yaml` (list/string) produces an `AttributeError` traceback. One `isinstance` guard. |
| TU-8 | `assess.md:31-34` | References `python course/_build/nbgen.py` — no `_build/` exists in the repo or wheel (instructor-only or stale). The mention is inside a "do NOT run" instruction so it's harmless, but a student's Claude reading it can't verify what nbgen is; mark it explicitly as instructor-tooling or drop it. |
| TU-9 | cosmetic | `_copy_tree` prints one line per file (133 lines on install — summarize instead); the packaged tree ships its `.gitignore` (harmless, arguably useful in the materialized copy). |

**Performance:** nothing of note — file walks and a subprocess launch; the CLI
defers heavy imports correctly (`cli.py` only imports the tutor subparser when
the `tutor` subcommand is requested).

---

## 3. Two system-level observations (the substantive part)

### 3.1 Grading is LLM-judgment-only for every student — by design, but unstated

`assess.md`/the assessor skill grade against the reference solution **only when
`course/solutions/` exists** — and it never ships (verified). So for every
actual student, all 100 points ride on the LLM's rubric interpretation:
non-deterministic run to run, un-appealable, and sensitive to model changes.
The prompts are well-written, but nothing pins grading behavior.

**Recommendation (the one genuinely valuable enhancement):** make the
exercises-70 portion deterministic. Each `exercises.ipynb` already has a fixed
structure; embed per-exercise assertion cells (or a `checks.py` per lesson) that
the assessor *runs* rather than judges — LLM grading then covers only the
writing-30 and partial-credit narration. This is the nbgrader lesson applied to
an LLM-native course: deterministic where possible, generative where valuable.
It also gives the course a self-test (see 3.2).

### 3.2 Nothing executes the course notebooks — 60 notebooks will silently rot

No test or CI job runs `reading.ipynb`/`exercises.ipynb` against the current
discopt API (`test_cli_tutor.py` is pure CLI; the Jupyter-Book build covers
`docs/notebooks/` only). The course teaches the discopt API across 30 lessons —
every API change is a chance for a lesson to break invisibly, discovered only
by a student mid-lesson. **Add a CI smoke job** (nbmake or a
`pytest`-parameterized executor, `-m slow`, possibly a weekly schedule rather
than per-PR) that executes at least all `reading.ipynb`. This is the
highest-value test addition for the module.

---

## 4. Design assessment

The architecture is genuinely well-conceived for what it is: an **LLM-native
courseware system** where the CLI is deliberately thin (locate tree, resolve
lesson, launch `claude` with a slash command) and all pedagogy lives in
versioned, packaged prompt files + notebooks. Compared to the established
alternatives — nbgrader (deterministic autograding, no tutoring),
Jupyter-Book/Carpentries (static content, no assessment) — the conversational
tutoring + rubric-based LLM grading + progress file is a real differentiator,
and things this design usually gets wrong are handled: state lives in one
gitignored YAML with a checked-in template; the packaged copy is read-only with
an explicit materialization step; prompts guard against the obvious clobbering
failure modes. The two §3 items (deterministic exercise checks, notebook CI)
are what separate it from being robust courseware rather than a good demo; the
§2 items are an afternoon of fixes.

---

## 5. Implementation plan (for Opus)

Single small PR for the mechanics, plus one CI addition:

**PR 1 — `fix(tutor): TU-1..TU-7`** (each with a regression test in
`test_cli_tutor.py`, failing first):
- TU-1 skip-passed in `_next_lesson`; TU-2 EOF-safe confirm + `--yes`;
  TU-3 zero-pad/substring/ambiguity-message in `_resolve_lesson`;
  TU-4 specific env-var error; TU-6 parent-walking command check;
  TU-7 dict guard in reset; TU-9 summarized install output.
- Baseline recorded: 46 passed.

**PR 2 — `ci(course): notebook execution smoke`** (§3.2): nbmake-style
execution of all 30 `reading.ipynb` (marker `slow`, JAX_PLATFORMS=cpu, timeout
per notebook); fix whatever it flushes out on first run.

**Design round (no code yet)** — deterministic exercise checks (§3.1): pick 2–3
lessons as pilots, add assertion cells + assessor-skill wiring
("run the checks, report pass/fail per exercise, grade only partial credit and
writing yourself"), evaluate whether scores stabilize across repeated
assessments of the same submission. TU-5 (single-sourcing thresholds) rides
along with whatever config file this introduces.
