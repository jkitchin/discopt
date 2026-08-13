#!/usr/bin/env bash
# #1004 — the two remaining passes, run strictly serially.
#
# Pass A (E2, first): restart cost — the issue's one escape hatch. This is a
# TIMING measurement, so it runs alone on a quiet machine (CLAUDE.md §9) and
# nothing else may be launched while it runs. Two arms: different starts, and a
# same-start control that separates caching from start-dependence.
# Pass B (deep pool): the models whose E1 unbiased pool contained no feasible
# configuration get a much larger pool and no dive, so the arm comparison is
# sourced only by channels that never consult the zero start. Restricted to the
# cheap models — gdp_col and syngas cost ~1.2 s per test, so a 400-configuration
# pool there is hours for an expected yield near zero (E1 measured 0 feasible in
# 60 and 58 configurations respectively).
set -u
cd /home/user/discopt || exit 1
D=scratchpad/issue1004

echo "=== pass A: E2 restart cost, different starts ($(date)) ==="
timeout 3600 python -u "$D/E2_restart_cost.py" --dive-seconds 45 --starts 5 --reps 3 \
    --configs 3 --out "$D/E2_results.json" > "$D/E2.log" 2>&1
echo "pass A exit=$? ($(date))"

echo "=== pass A2: E2 control, same start repeated ($(date)) ==="
timeout 3600 python -u "$D/E2_restart_cost.py" --dive-seconds 45 --starts 5 --reps 3 \
    --configs 3 --same-start --out "$D/E2_samestart_results.json" \
    > "$D/E2_samestart.log" 2>&1
echo "pass A2 exit=$? ($(date))"

echo "=== pass B: deep unbiased pool, no dive ($(date)) ==="
timeout 5400 python -u "$D/E1_detection_rate.py" \
    --models jobshop small_batch cstr \
    --dive-seconds 0 --random-starts 4 \
    --pool-random 300 --pool-neighbours 300 \
    --out "$D/E1_deep_results.json" > "$D/E1_deep.log" 2>&1
echo "pass B exit=$? ($(date))"
echo "ALL PASSES DONE"
