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
| `/doe`              | Design optimal experiments (identifiability, D/A/E-optimal, active learning) |
| `/benchmark-report` | Narrative performance report from benchmark JSON |

(The dev-only `/adversary` and `/discoptbot` commands ship separately via
`discopt-dev`, not in the package bundle.)

### Agents (`~/.claude/agents/`)

22 domain-expert subagents covering modeling and the solver stack:
`amp-expert`, `benchmarking-expert`, `convex-relaxation-expert`,
`convexity-detection-expert`, `differentiability-expert`, `doe-expert`,
`estimability-expert`, `estimation-expert`, `heuristics-expert`,
`highs-expert`, `identifiability-expert`, `ipopt-expert`, `jax-ipm-expert`,
`llm-feature-expert`, `minlp-solver-expert`, `model-discrimination-expert`,
`modeling-expert`, `multiobjective-expert`, `nn-embedding-expert`,
`presolve-expert`, `robust-opt-expert`, `scip-expert`.

The source of truth for these files is `python/discopt/skills/commands/`
and `python/discopt/skills/agents/`. Edit them there; downstream users
pick up updates on the next `pip install --upgrade discopt`.
