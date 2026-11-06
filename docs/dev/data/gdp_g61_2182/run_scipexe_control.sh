#!/bin/bash
# #2182: like-for-like control for the #2149 deviation. Pyomo bigm/hull arms on the SCIP
# executable (GDP_BASELINE_GLOBAL=scip, #2149's setting) vs this run's scip_direct.
until grep -q G61_DONE /tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/g61/run.log; do sleep 30; done
cd /tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/wt_main
export PYTHONPATH=/tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/wt_main/python GDP_BASELINE_GLOBAL=scip
echo "=== scipexe control start $(date)"; uptime
taskset -c 0-2 /tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/venv312/bin/python -u discopt_benchmarks/scripts/gdp_baseline.py --out /tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/g61/control_pyomo_scipexe.jsonl --arms pyomo-bigm,pyomo-hull --jobs 3 --time-limit 300 --grace 120
echo "=== scipexe control end $(date) exit $?"
echo SCIPEXE_DONE
