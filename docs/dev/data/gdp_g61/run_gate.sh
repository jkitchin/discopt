#!/bin/bash
# #1917 G6.1 gate: 14 family arms x 24 gdp_parity instances x 3 reps, jobs=4, 300 s.
WT=/Users/jkitchin/projects/discopt-g61
S=/private/tmp/claude-501/-Users-jkitchin-projects-discopt/5793622b-ea73-47f3-a307-5878f6abb68d/scratchpad/g61
ARMS=discopt-big-m,pyomo-bigm,discopt-hull,pyomo-hull,discopt-loa,pyomo-loa,discopt-gloa,pyomo-gloa,discopt-ric,pyomo-ric,discopt-enumerate,pyomo-enumerate,discopt-lbb,pyomo-lbb
cd $WT
export GDP_BASELINE_GLOBAL=scip PYTHONPATH=$WT/python
for rep in 1 2 3; do
  echo "=== rep $rep start $(date)"; uptime
  /Users/jkitchin/Dropbox/uv/.venv/bin/python -u discopt_benchmarks/scripts/gdp_baseline.py \
    --out $S/rep$rep.jsonl --arms $ARMS --jobs 4 --time-limit 300 --grace 120
  echo "=== rep $rep end $(date) exit $?"
done
echo GATE_DONE
