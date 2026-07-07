# The discopt Optimization Course (`discopt tutor`)

The self-paced optimization course — a 30-lesson curriculum across
**basic**, **intermediate**, and **advanced** tracks, taught *through* the
solver and delivered interactively via [Claude Code](https://claude.com/claude-code) —
now lives in the standalone
[discopt-course](https://github.com/jkitchin/discopt-course) plugin (#430),
mirroring the DoE extraction (#389). It is no longer bundled with the core
`discopt` package.

## Installation

```bash
pip install discopt-course
```

Installing the plugin registers the `tutor` subcommand through discopt's
`"discopt.cli"` entry-point group, so it is discovered automatically by the
core CLI — exactly like `discopt doe`. No changes to core are needed.

## Usage

Once installed, drive the course from the same `discopt` command line:

```bash
discopt tutor list                # show the lesson library
discopt tutor start basic-01      # begin a lesson
discopt tutor resume              # continue where you left off
discopt tutor next                # advance to the next lesson
discopt tutor install             # install the packaged /course: slash commands
```

`discopt tutor` is a thin wrapper: it locates the packaged course tree,
resolves a lesson id, and launches a `claude` session running the matching
`/course:` slash command. The lesson library, grading rubrics, and your
progress all live inside the plugin.

See the
[discopt-course documentation](https://github.com/jkitchin/discopt-course)
for the full curriculum and workflow.
