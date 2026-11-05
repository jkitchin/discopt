#!/bin/bash
# #2142 graduation re-run (section 116): GDP parity hull + big-m, fire then solve.
cd /home/user/discopt
D=docs/dev/data/2142-exact-hull/panel2
F=DISCOPT_HULL_EXACT_PERSPECTIVE
P=discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py
python -u $P --flag $F --phase fire --out $D/gdp_hull.jsonl > $D/gdp_fire.log 2>&1
python -u $P --flag $F --phase solve --reps 1 --time-limit 120 --max-nodes 5000 --deterministic --max-load 99 --out $D/gdp_hull.jsonl > $D/gdp_solve.log 2>&1
python $P --flag $F --report --out $D/gdp_hull.jsonl > $D/gdp_report.txt 2>&1
echo DONE
