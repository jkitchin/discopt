#!/bin/bash
# #1008 H2: LU threshold-pivoting sweep. One process per threshold (OnceLock knob).
WT=/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077
export PYTHONPATH="$WT/python"
cd "$WT"
LOG="$WT/scratchpad/i1008/pivsweep.log"
: > "$LOG"
uptime
for u in 1.0 0.5 0.1 0.01; do
  echo "=== u=$u ===" | tee -a "$LOG"
  DISCOPT_LU_PIVOT_THRESHOLD=$u DISCOPT_PROFILE=1 I1008_TL=45 \
    python -u "$WT/scratchpad/i1008/fill2.py" 2>/dev/null | tee -a "$LOG"
done
grep '^JSONP ' "$LOG" | sed 's/^JSONP //' > "$WT/scratchpad/i1008/pivsweep.jsonl"
uptime
echo "PIVSWEEP DONE lines=$(wc -l < "$WT/scratchpad/i1008/pivsweep.jsonl")"
