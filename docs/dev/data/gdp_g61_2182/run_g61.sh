#!/bin/bash
# #2182 task 1: G6.1 re-run on main (worktree at origin/main), same design as #2149:
# rep 1 = all 14 arms x 24 gdp_parity instances; reps 2-3 = both-solved pairs only.
WT=/tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/wt_main
S=/tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/g61
PY=/tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/venv312/bin/python
ARMS=discopt-big-m,pyomo-bigm,discopt-hull,pyomo-hull,discopt-loa,pyomo-loa,discopt-gloa,pyomo-gloa,discopt-ric,pyomo-ric,discopt-enumerate,pyomo-enumerate,discopt-lbb,pyomo-lbb
export PYTHONPATH=$WT/python
cd $WT
echo "=== rep1 start $(date) head $(git rev-parse --short HEAD)"; uptime
taskset -c 0-2 $PY -u discopt_benchmarks/scripts/gdp_baseline.py --out $S/rep1.jsonl --arms $ARMS --jobs 3 --time-limit 300 --grace 120
echo "=== rep1 end $(date) exit $?"
$PY - <<'PY' > $S/pairs.txt
import json
F={"big-m":("discopt-big-m","pyomo-bigm"),"hull":("discopt-hull","pyomo-hull"),"loa":("discopt-loa","pyomo-loa"),"gloa":("discopt-gloa","pyomo-gloa"),"ric":("discopt-ric","pyomo-ric"),"enumerate":("discopt-enumerate","pyomo-enumerate"),"lbb":("discopt-lbb","pyomo-lbb")}
v={}
for l in open("/tmp/claude-0/-home-user-discopt/7dc8155b-ccf3-5b20-9dd0-fc10a1fe7043/scratchpad/g61/rep1.jsonl"):
    r=json.loads(l); v[(r["instance"],r["arm"])]=r["verdict"]
for fam,(d,p) in F.items():
    inst=sorted({i for (i,a) in v if v.get((i,d))=="solved" and v.get((i,p))=="solved"})
    if inst: print(fam, d+","+p, ",".join(inst))
PY
cat $S/pairs.txt
for rep in 2 3; do
  echo "=== timing rep $rep start $(date)"
  while read fam arms inst; do
    taskset -c 0-2 $PY -u discopt_benchmarks/scripts/gdp_baseline.py --out $S/timing_rep$rep.jsonl --arms $arms --include $inst --jobs 3 --time-limit 300 --grace 120
  done < $S/pairs.txt
done
echo "=== serial control start $(date)"
while read fam arms inst; do
  sub=$(echo $inst | tr ',' '\n' | head -2 | paste -sd, -)
  taskset -c 0-2 $PY -u discopt_benchmarks/scripts/gdp_baseline.py --out $S/serial.jsonl --arms $arms --include $sub --jobs 1 --time-limit 300 --grace 120
done < $S/pairs.txt
echo G61_DONE $(date)
