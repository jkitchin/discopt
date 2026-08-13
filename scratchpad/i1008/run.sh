#!/bin/bash
# Run a probe with the worktree's python package on the path.
WT=/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077
export PYTHONPATH="$WT/python"
cd "$WT"
uptime
exec python -u "$@"
