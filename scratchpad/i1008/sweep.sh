#!/bin/bash
# #1008 refactorization-cadence sweep. One process per interval (OnceLock knob).
# A per-LP wall limit keeps one pathological arm from stalling the whole sweep;
# an LP that trips it records status=iter_limit, which is itself a datum.
WT=/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077
export PYTHONPATH="$WT/python"
cd "$WT"
OUT="$WT/scratchpad/i1008/sweep.jsonl"
LOG="$WT/scratchpad/i1008/sweep.log"
: > "$OUT"
: > "$LOG"
uptime
for iv in 48 100 200 400 800 1600 3200; do
  echo "=== interval=$iv ===" | tee -a "$LOG"
  DISCOPT_LP_REFACTOR_INTERVAL=$iv I1008_HIGHS=1 I1008_REPS=1 \
    I1008_TL="${I1008_TL:-45}" I1008_MAXROWS="${I1008_MAXROWS:-6000}" \
    python -u "$WT/scratchpad/i1008/arm.py" 2>&1 | tee -a "$LOG"
done
grep '^JSON ' "$LOG" | sed 's/^JSON //' > "$OUT"
uptime
echo "SWEEP DONE lines=$(wc -l < "$OUT")"
