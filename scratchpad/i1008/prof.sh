#!/bin/bash
set -e
WT=/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077
export PYTHONPATH="$WT/python"
cd "$WT"
cp target/release/lib_rust.dylib python/discopt/_rust.cpython-312-darwin.so
uptime
samply record --save-only -o "$WT/scratchpad/i1008/prof.json.gz" --rate 999 -- \
  python -u scratchpad/i1008/profile_probe.py
echo "PROFILE WRITTEN"
