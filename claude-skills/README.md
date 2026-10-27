# discopt Claude Code Skills

The project's shareable Claude Code slash commands, agent personas and
Agent Skill live **inside the Python package** at `python/discopt/skills/`,
so they ship with the wheel and are available to anyone who has
`pip install`ed discopt.

## Install with knowhow (no discopt install needed)

The repository root has `commands/`, `agents/` and `skills/` symlinks into
`python/discopt/skills/`, which is the layout
[knowhow](https://github.com/jkitchin/knowhow) (v0.3+) discovers:

```bash
uvx knowhow add jkitchin/discopt --list          # 8 commands, 17 agents, 1 skill
uvx knowhow add jkitchin/discopt --all           # install everything
uvx knowhow add jkitchin/discopt --skill discopt # just the overview skill (any agent)
uvx knowhow check / update / remove ...          # lock-file managed
```

The `discopt` skill follows the Agent Skills spec, so knowhow installs it for
any supported agent (Claude Code, Codex, Cursor, opencode, ...). Commands
and subagents are Claude Code only. On Windows, clone with
`git config core.symlinks true` or the root links are plain files; installing
from the subpath (`jkitchin/discopt/python/discopt/skills`) works either way.

## Install with the discopt CLI

```bash
# Install into ~/.claude/ (affects every Claude Code project)
discopt install-skills

# Or install into the current project's .claude/
discopt install-skills --project-scope

# Symlink instead of copy (edits show up live; useful with pip install -e)
discopt install-skills --dev

# Overwrite existing files
discopt install-skills --force
```

A legacy wrapper `bash claude-skills/install.sh` is kept for backward
compatibility and simply forwards to `discopt install-skills --project-scope`.

## What ships

### Skill (`~/.claude/skills/discopt/`)

`discopt`: an overview skill covering how to build a model, solve it, read a
`SolveResult` (what `optimal`, `gap_certified` and the local statuses do and do
not claim) and load `.nl`/GAMS files. It loads when the task needs it.

### Slash commands (`~/.claude/commands/`)

| Command | What it does |
|---------|--------------|
| `/formulate`        | Natural-language problem → runnable discopt model |
| `/debug`            | Troubleshoot a broken setup, model, or solve (install/daemon/infeasibility/numerics) |
| `/diagnose`         | Interpret a `SolveResult` and recommend next steps |
| `/reformulate`      | Strengthen relaxations / exploit structure for a faster solve |
| `/explain-model`    | Generate a formal math write-up of a model |
| `/convert`          | Translate a model to/from Pyomo, GAMS, AMPL, JuMP (and file formats) |
| `/estimate`         | Fit model parameters to experimental data |
| `/benchmark-report` | Narrative performance report from benchmark JSON |

(The dev-only `/adversary` and `/discoptbot` commands ship separately via
`discopt-dev`, not in the package bundle. The `/discopt-doe` skill and the
DoE expert agents are provided by the [discopt-doe](https://github.com/jkitchin/discopt-doe)
plugin: `discopt-doe-install-skill`.)

### Agents (`~/.claude/agents/`)

17 domain-expert subagents covering modeling and the solver stack:
`amp-expert`, `benchmarking-expert`, `convex-relaxation-expert`,
`convexity-detection-expert`, `differentiability-expert`, `estimation-expert`, `heuristics-expert`,
`highs-expert`, `ipopt-expert`,
`llm-feature-expert`, `minlp-solver-expert`,
`modeling-expert`, `multiobjective-expert`, `ml-embedding-expert`,
`presolve-expert`, `robust-opt-expert`, `scip-expert`.

The source of truth for these files is `python/discopt/skills/commands/`,
`python/discopt/skills/agents/` and `python/discopt/skills/skills/`. Edit them there; downstream users
pick up updates on the next `pip install --upgrade discopt`.
