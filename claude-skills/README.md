# discopt Claude Code Skills

The project's shareable Claude Code slash commands and agent personas now
live **inside the Python package** at `python/discopt/skills/`, so they
ship with the wheel and are available to anyone who has `pip install`ed
discopt.

## Install

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

| Slash commands (`~/.claude/commands/`) | Agents (`~/.claude/agents/`) |
|----------------------------------------|------------------------------|
| `adversary`        | `highs-expert` |
| `benchmark-report` | `scip-expert`  |
| `convert`          |                |
| `diagnose`         |                |
| `discoptbot`       |                |
| `doe`              |                |
| `estimate`         |                |
| `explain-model`    |                |
| `formulate`        |                |
| `reformulate`      |                |

The source of truth for these files is `python/discopt/skills/commands/`
and `python/discopt/skills/agents/`. Edit them there; downstream users
pick up updates on the next `pip install --upgrade discopt`.
