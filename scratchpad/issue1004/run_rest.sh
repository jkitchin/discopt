#!/usr/bin/env bash
# #1004 — the two remaining passes, run strictly serially.
#
# Pass 2 (deep): the five models whose E1 unbiased pool contained no feasible
# configuration at all get a much larger pool and no dive, so the arm comparison
# is sourced only by channels that never consult the zero start.
# Pass 3 (E2): restart cost. Timing, so it runs alone on a quiet machine (§9) —
# nothing else may be launched while it runs.
set -u
cd /home/user/discopt || exit 1
D=scratchpad/issue1004

echo "=== pass 2: deep unbiased pool, no dive ($(date)) ==="
timeout 5400 python -u "$D/E1_detection_rate.py" \
    --models jobshop small_batch cstr syngas gdp_col \
    --dive-seconds 0 --random-starts 4 \
    --pool-random 200 --pool-neighbours 200 \
    --out "$D/E1_deep_results.json" > "$D/E1_deep.log" 2>&1
echo "pass 2 exit=$? ($(date))"

echo "=== pass 3: E2 restart cost, different starts ($(date)) ==="
timeout 3600 python -u "$D/E2_restart_cost.py" --dive-seconds 45 --starts 5 --reps 3 \
    --configs 3 --out "$D/E2_results.json" > "$D/E2.log" 2>&1
echo "pass 3 exit=$? ($(date))"

echo "=== pass 3b: E2 control, same start repeated ($(date)) ==="
timeout 3600 python -u "$D/E2_restart_cost.py" --dive-seconds 45 --starts 5 --reps 3 \
    --configs 3 --same-start --out "$D/E2_samestart_results.json" \
    > "$D/E2_samestart.log" 2>&1
echo "pass 3b exit=$? ($(date))"
echo "ALL PASSES DONE"
